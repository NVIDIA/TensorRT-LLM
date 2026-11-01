# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eight MPI ranks: TP8 SA-1R and two-round sampling must be bit-identical."""

import gc
import json
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("the sampled exchange requires CUDA", allow_module_level=True)
# Select the MPI-local device before TensorRT-LLM initializes CUDA libraries.
torch.cuda.set_device(
    int(os.environ.get("SLURM_LOCALID", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
)

from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size  # noqa: E402
from tensorrt_llm.mapping import Mapping  # noqa: E402

if mpi_world_size() != 8:
    if os.environ.get("M3_REQUIRE_TP8_EXCHANGE") == "1":
        raise RuntimeError(f"expected eight MPI ranks, got {mpi_world_size()}")
    pytest.skip("run this regression with eight MPI ranks", allow_module_level=True)

# Every rank makes the same eligibility decision before creating workspaces.
_capabilities = mpi_comm().allgather(torch.cuda.get_device_capability())
if any(capability != (10, 7) for capability in _capabilities):
    if os.environ.get("M3_REQUIRE_TP8_EXCHANGE") == "1":
        raise RuntimeError(f"expected SM 10.7 on every rank, got {_capabilities}")
    pytest.skip("the TP8 exchange requires SM 10.7 on every rank", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops import (  # noqa: E402,E501
    m3_sampled_accept as sa,
)

_VS = 25008
_DRAFT = 3
_CALLS = 4
_CASES = [
    (4, "spiked", 0.95),
    (16, "spiked", 0.5),
    (48, "spiked", 0.95),
    (4, "tail", 0.02),
    (16, "tail", 0.5),
    (48, "tail", 0.95),
]


def _logits(rows, mode, rank, device):
    generator = torch.Generator(device=device).manual_seed(20261101 + rows * 31 + rank)
    if mode == "tail":
        return (torch.randn(rows, _VS, generator=generator, device=device) * 0.3 + rank * 0.5).to(
            torch.bfloat16
        )
    out = torch.full((rows, _VS), -30.0, dtype=torch.bfloat16, device=device)
    hot = (torch.arange(12, device=device) * 97 + rank * 43 + 5) % _VS
    out[:, hot] = (torch.linspace(4.0, 0.0, 12, device=device) + rank * 0.5).to(out.dtype)
    return out


def _snapshot(logits, mapping, draft, batch, temperatures, top_ks, top_ps, seeds, offsets, ws):
    next_new = torch.full((draft.shape[0], _DRAFT + 1), -7, dtype=torch.int32, device=logits.device)
    result = sa.sampled_accept(
        logits,
        mapping.tp_rank * _VS,
        mapping,
        draft,
        batch,
        temperatures,
        top_ks,
        top_ps,
        None,
        seeds,
        offsets,
        0,
        next_new=next_new,
        ws=ws,
        use_pdl=True,
    )
    assert result is not None
    return tuple(t.clone() for t in (*result, next_new, ws.tokens[: logits.shape[0]]))


def _assert_collective(condition, message):
    failures = mpi_comm().allgather(None if condition else message)
    assert not any(failures), f"rank comparison failures: {failures}"


def _assert_same(left, right):
    _assert_collective(
        len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right)),
        "exchange variants differ",
    )


def test_comparison_failure_is_collective():
    left = torch.tensor([1], device="cuda")
    right = torch.tensor([2 if mpi_rank() == 3 else 1], device="cuda")
    with pytest.raises(AssertionError, match="exchange variants differ"):
        _assert_same((left,), (right,))
    mpi_comm().Barrier()


@pytest.mark.parametrize("rows,mode,top_p", _CASES)
def test_one_round_exchange_matches_two_rounds(monkeypatch, rows, mode, top_p):
    assert "one_round_k" in sa._op_args() and "decided" in sa._op_args()
    rank = mpi_rank()
    comm = mpi_comm()
    device = torch.device("cuda", torch.cuda.current_device())
    mapping = Mapping(world_size=8, rank=rank, gpus_per_node=torch.cuda.device_count(), tp_size=8)
    logits = _logits(rows, mode, rank, device)
    nreq = rows // (_DRAFT + 1)
    temperatures = torch.linspace(0.8, 1.2, rows, device=device)
    top_ks = torch.zeros(rows, dtype=torch.int32, device=device)
    top_ps = torch.full((rows,), top_p, device=device)
    seeds = torch.arange(rows, device=device, dtype=torch.int64) * 17 + 20261101
    base_offsets = torch.arange(rows, device=device, dtype=torch.int64)
    offsets = base_offsets.clone()
    draft = torch.zeros(nreq, _DRAFT, dtype=torch.int32, device=device)
    batch = torch.arange(nreq, dtype=torch.int32, device=device)
    masses = comm.allgather(torch.logsumexp(logits[0].float() / temperatures[0], 0).item())
    assert max(masses) - min(masses) > 0.5, "the test needs unequal shard masses"
    workspaces = {}
    try:
        for variant in (0, 82):
            monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
            workspaces[variant] = sa.SampledAcceptWorkspace.create(mapping, device)
        warm = []
        for variant in (0, 82):
            monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
            warm.append(
                _snapshot(
                    logits,
                    mapping,
                    draft,
                    batch,
                    temperatures,
                    top_ks,
                    top_ps,
                    seeds,
                    offsets,
                    workspaces[variant],
                )
            )
        torch.cuda.synchronize()
        _assert_same(*warm)
        # Reuse the exact Philox inputs, but exercise every strict-accept prefix.
        sampled = warm[0][-1].reshape(nreq, _DRAFT + 1)
        draft.copy_(sampled[:, :_DRAFT])
        prefixes = torch.arange(nreq, device=device) % (_DRAFT + 1)
        for request in range(nreq):
            prefix = request % (_DRAFT + 1)
            if prefix < _DRAFT:
                draft[request, prefix] = (draft[request, prefix] + 1) % (_VS * 8)
        observations = []
        for call in range(_CALLS):
            offsets.copy_(base_offsets + call * 1000)
            pair = []
            for variant in (0, 82):
                monkeypatch.setattr(sa, "ONE_ROUND_K", variant)
                pair.append(
                    _snapshot(
                        logits,
                        mapping,
                        draft,
                        batch,
                        temperatures,
                        top_ks,
                        top_ps,
                        seeds,
                        offsets,
                        workspaces[variant],
                    )
                )
            observations.append(pair)
        torch.cuda.synchronize()
        for call, pair in enumerate(observations):
            _assert_same(*pair)
            if call == 0:
                _assert_collective(
                    torch.equal(pair[0][1], (prefixes + 1).to(torch.int32)),
                    "strict-accept prefixes differ",
                )
            record = [tensor.cpu().tolist() for tensor in pair[0]]
            peers = comm.allgather(record)
            assert all(peer == peers[0] for peer in peers), "ranks disagree on sampled outputs"
        for ws in workspaces.values():
            _assert_collective(
                ws.counters.cpu().tolist() == [0, _CALLS + 1],
                "workspace counter did not exercise both halves",
            )
        decided = workspaces[82].decided.item()
        if mode == "spiked":
            _assert_collective(decided > 0, "the spiked case must exercise SA-1R decisions")
        if rank == 0:
            print(
                json.dumps(
                    dict(
                        rows=rows,
                        mode=mode,
                        top_p=top_p,
                        world=8,
                        calls_per_variant=_CALLS + 1,
                        both_halves=True,
                        unequal_log_masses=masses,
                        decided=decided,
                        all_outputs_bitwise=True,
                    )
                ),
                flush=True,
            )
    finally:
        torch.cuda.synchronize()
        comm.Barrier()
        # Release multicast memory while each workspace communicator is still live.
        for ws in workspaces.values():
            ws.uc = None
            ws.handle = None
        gc.collect()
        comm.Barrier()
        for ws in workspaces.values():
            if ws.comm is not None:
                ws.comm.Free()
