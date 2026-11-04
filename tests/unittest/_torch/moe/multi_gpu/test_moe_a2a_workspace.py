# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank regression for alternating quantized/unquantized A2A payloads."""

import faulthandler
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm as tllm
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm.mapping import Mapping

# Match the neighboring MPI tests: workers must receive this module by value.
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_EP_SIZE = 4
_WORKER_TIMEOUT_S = 180


@pytest.fixture
def workspace_mpi_pool():
    """Use fresh workers for each parametrization."""
    # Unlike the shared module-scoped fixture, isolate each parametrization.
    # Keep the parent deadline active through startup, results, and shutdown.
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S + 60, exit=True)
    try:
        with MPIPoolExecutor(_EP_SIZE) as executor:
            yield executor
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_worker(capture, in_workspace, low_precision):
    """Bound hangs in native calls, MPI barriers, and collective cleanup."""
    faulthandler.dump_traceback_later(_WORKER_TIMEOUT_S, exit=True)
    communicators = []
    aborting = False
    try:
        with pytest.MonkeyPatch.context() as patch:
            # Pin worker-side policy rather than inheriting user/CI overrides.
            patch.delenv("TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB", raising=False)
            try:
                rank = tllm.mpi_rank()
                assert tllm.mpi_world_size() == _EP_SIZE
                torch.cuda.set_device(rank)
                MnnvlMemory.initialize()
                reason = None
                if not MnnvlMemory.supports_mnnvl():
                    reason = "Requires an NVLink fabric supported by MNNVL"
                # The large race reproducer retains graph pools and expert/output
                # tensors in addition to its workspace; leave conservative headroom.
                if torch.cuda.mem_get_info()[0] < 24 * 1024**3:
                    reason = "Requires at least 24 GiB free memory per GPU"
                reasons = MPI.COMM_WORLD.allgather(reason)
                if any(reasons):
                    return "; ".join(sorted({r for r in reasons if r})), []

                hidden, capacity = 6144, 32768
                mapping = Mapping(
                    world_size=_EP_SIZE,
                    rank=rank,
                    tp_size=_EP_SIZE,
                    moe_ep_size=_EP_SIZE,
                    gpus_per_node=_EP_SIZE,
                    enable_attention_dp=True,
                )
                for _ in range(2):
                    communicators.append(
                        NVLinkOneSided(
                            mapping,
                            256,
                            8,
                            capacity,
                            payload_in_workspace=in_workspace,
                            hidden_size=hidden,
                            dtype=torch.bfloat16,
                            use_low_precision_combine=low_precision,
                        )
                    )
                first, second = communicators
                assert first.workspace.data_ptr() == second.workspace.data_ptr()
                failures = _run_rounds(first, second, rank, hidden, capacity, capture, in_workspace)
                return None, failures
            except Exception:
                # A peer may already be in a collective. Abort this test's MPI
                # workers before attempting CUDA-synchronizing teardown.
                traceback.print_exc()
                aborting = True
                MPI.COMM_WORLD.Abort(1)
                raise
            finally:
                if not aborting:
                    for comm in reversed(communicators):
                        comm.destroy()
    finally:
        faulthandler.cancel_dump_traceback_later()


def _run_rounds(first, second, rank, hidden, capacity, capture, in_workspace):
    """Exercise the original mixed-layout/skew reproducer without extra barriers."""
    ids = torch.tensor([0, 64, 128, 192, 1, 65, 129, 193], dtype=torch.int32, device="cuda").repeat(
        capacity, 1
    )
    scales = torch.full((capacity, 8), 0.125, device="cuda")
    small = torch.full((capacity, hidden // 2), 85, dtype=torch.uint8, device="cuda")
    sf = torch.ones((capacity, hidden // 16), dtype=torch.uint8, device="cuda")
    big = torch.full((capacity, hidden), float("nan"), dtype=torch.bfloat16, device="cuda")
    expert = torch.full(
        (4 * capacity, hidden), float(rank + 1), dtype=torch.bfloat16, device="cuda"
    )
    lengths_list = (
        [266, 9, 19034, 9],
        [266, 9, 19034, 9],
        [0, 1, 32, 0],
        [9, 0, capacity, 1],
        [1, 9, 64, 0],
        [1, 9, 64, 0],
    )

    def rounds():
        checks = []
        for i, lengths in enumerate(lengths_list):
            count, maximum = lengths[rank], max(lengths)
            comm = first if i % 2 == 0 else second
            comm.dispatch(
                small[:count] if i % 2 == 0 else big[:count],
                sf[:count] if i % 2 == 0 else None,
                ids[:count],
                scales[:count],
                lengths,
            )
            if in_workspace:
                payload = comm.get_combine_payload_tensor_in_workspace(
                    maximum, hidden, torch.bfloat16
                )
                payload.fill_(rank + 1)
            else:
                payload = expert[: 4 * maximum]
            if rank == 2:
                torch.cuda._sleep(200000)
            output = comm.combine(payload)
            checks.append(output.eq(10).all())
        return torch.stack(checks)

    # Initialize kernels before capture; no host synchronization inside rounds.
    rounds()
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    records = []
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            checks = rounds()
        for _ in range(20):
            graph.replay()
            records.append(checks.clone())
    else:
        for _ in range(20):
            records.append(rounds())
    valid = torch.stack(records).cpu()
    failures = (~valid).nonzero().tolist()
    MPI.COMM_WORLD.Barrier()
    return failures


@pytest.mark.skipif(torch.cuda.device_count() < _EP_SIZE, reason="Requires four local GPUs")
@pytest.mark.threadleak(enabled=False)  # MPI pool shutdown has known thread timing issues.
@pytest.mark.parametrize("low_precision", [False, True])
@pytest.mark.parametrize(
    "capture,in_workspace", [(False, False), (False, True), (True, False), (True, True)]
)
def test_mixed_dispatch_layout_preserves_previous_combine(
    workspace_mpi_pool, capture, in_workspace, low_precision
):
    """Run four ranks under the regular pytest/CI MPI pool launcher."""
    results = list(
        workspace_mpi_pool.map(
            _run_worker,
            *zip(*[(capture, in_workspace, low_precision)] * _EP_SIZE),
        )
    )
    reasons = [reason for reason, _ in results if reason]
    if reasons:
        assert len(reasons) == _EP_SIZE, "Workers must agree to skip before native collectives"
        pytest.skip(reasons[0])
    for rank, (_, failures) in enumerate(results):
        assert not failures, (rank, capture, in_workspace, low_precision, failures)
