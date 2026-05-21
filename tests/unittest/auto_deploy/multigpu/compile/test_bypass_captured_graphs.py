# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU regression test for ``_bypass_captured_graphs`` in the AD CUDA-graph backend.

Exercises the divergence pattern that causes the conc=32 NaN regression in PR #13723
(MoE all-to-all with attention-DP). Pre-fix: a captured graph for ``bs=1`` bakes
``int(batch_info_host[13].item())`` from capture time into the kernel launch as a
scalar argument, so at replay the captured constant is used regardless of the
post-``tp_allgather`` slot-13 value. The wrapper ``maybe_pad_for_cuda_graph``
toggles ``_bypass_captured_graphs`` so all ranks fall through to eager and read
slot 13 fresh, restoring cross-rank consistency.

This test reproduces that pattern with a tiny module (no MoE kernel needed).
Each rank locally checks both code paths.
"""

import pytest
import torch
import torch.nn as nn
from _dist_test_utils import get_device_counts

from tensorrt_llm._torch.auto_deploy.compile.backends.torch_cudagraph import CapturedGraph
from tensorrt_llm._torch.auto_deploy.distributed.common import spawn_multiprocess_job
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import _set_bypass_captured_graphs


class _SlotReadingModel(nn.Module):
    """Mimics the ``int(batch_info_host[13].item())`` scalar read in the MoE A2A op.

    The Python int returned by ``.item()`` is consumed as a scalar kernel argument
    by ``torch.full``; under cuda graph capture that argument value is recorded
    into the graph and reused at replay irrespective of the current tensor value.
    """

    def __init__(self, batch_info_host: torch.Tensor):
        super().__init__()
        # The host tensor is a singleton stand-in for SequenceInfo's batch_info_host.
        self._batch_info_host = batch_info_host

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Read a Python int from a CPU/pinned tensor (no GPU sync required).
        runtime_value = int(self._batch_info_host[0].item())
        bs = input_ids.shape[0]
        # ``torch.full`` issues a kernel whose ``value`` is baked into the launch.
        return torch.full((bs,), float(runtime_value), device="cuda")


def _run_bypass_test(rank: int, world_size: int) -> None:
    device = "cuda"
    # Pinned host scalar standing in for batch_info_host[13].
    batch_info_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)

    inner = _SlotReadingModel(batch_info_host)
    cg = CapturedGraph(inner, num_batched_inputs=1)

    capture_bs = [1, 2, 4]

    def get_args_kwargs(bs: int):
        # Mimic SequenceInfo.set_capture_batch / nest_sequences seeding slot 13
        # with the local total at capture time. Each captured graph thus bakes
        # in its own runtime_value (= bs).
        batch_info_host[0] = bs
        return (torch.zeros(bs, device=device),), {}

    cg.capture_graph(get_args_kwargs, capture_bs)

    # Mimic ADEngine.forward(): tp_allgather updates slot 13 to a cross-rank max
    # that is unrelated to any individual rank's capture-time value.
    runtime_value = 7919  # unique sentinel
    batch_info_host[0] = runtime_value

    # --- Path 1: pre-fix behaviour reachable when bypass is OFF ---
    # Replay path: shape matches captured bs=1 -> kernel arg is the captured int (1).
    out_replay = cg(torch.zeros(1, device=device))
    # Eager path: shape (3,) does not match any captured graph -> Python re-runs,
    # reads slot 13 fresh -> kernel arg is the runtime int.
    out_eager = cg(torch.zeros(3, device=device))

    expected_capture = 1  # the captured graph for bs=1 baked in 1
    assert int(out_replay[0].item()) == expected_capture, (
        f"rank={rank}: replay path should use baked-in capture-time value "
        f"{expected_capture}, got {out_replay.tolist()}"
    )
    assert int(out_eager[0].item()) == runtime_value, (
        f"rank={rank}: eager path should use fresh slot value {runtime_value}, "
        f"got {out_eager.tolist()}"
    )

    # --- Path 2: with bypass ON, both paths must use the fresh slot value ---
    _set_bypass_captured_graphs(cg, True)
    try:
        out_replay_bypassed = cg(torch.zeros(1, device=device))
        out_eager_bypassed = cg(torch.zeros(3, device=device))
    finally:
        _set_bypass_captured_graphs(cg, False)

    assert int(out_replay_bypassed[0].item()) == runtime_value, (
        f"rank={rank}: replay shape with bypass=True must fall through to eager "
        f"and read slot fresh ({runtime_value}), got {out_replay_bypassed.tolist()}"
    )
    assert int(out_eager_bypassed[0].item()) == runtime_value, (
        f"rank={rank}: eager shape with bypass=True must still use fresh slot "
        f"({runtime_value}), got {out_eager_bypassed.tolist()}"
    )

    # --- Path 3: bypass cleared -> behaviour returns to Path 1 ---
    out_replay_after = cg(torch.zeros(1, device=device))
    assert int(out_replay_after[0].item()) == expected_capture, (
        f"rank={rank}: replay should resume after bypass cleared, got {out_replay_after.tolist()}"
    )


@pytest.mark.parametrize("device_count", get_device_counts([2]))
def test_bypass_captured_graphs(device_count):
    spawn_multiprocess_job(job=_run_bypass_test, size=device_count)
