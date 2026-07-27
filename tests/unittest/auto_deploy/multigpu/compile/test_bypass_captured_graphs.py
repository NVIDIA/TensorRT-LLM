# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Multi-GPU regression test for the captured-graph bypass mechanism.

Guards both layers of the fix that landed in PR #13723 for the conc=32 NaN
regression on MoE all-to-all under attention-DP:

  Level 1 (leaf): ``CapturedGraph.forward`` honours the
    ``BypassCapturedGraphs()`` context manager. A capture-time scalar read inside
    the captured graph (``int(host_tensor[0].item())``, mirroring how the MoE
    all-to-all op in ``trtllm_moe.py`` derives ``runtime_max_tokens_per_rank``
    from the capture-time input shape) is baked into the captured kernel-launch as
    a scalar argument. Replay reuses it regardless of post-capture host updates.
    While inside ``BypassCapturedGraphs()`` (i.e.
    ``cuda_graph_state.in_bypass() == True``), the wrapper short-circuits to
    eager and reads the host fresh.

  Level 2 (wrapper): ``maybe_pad_for_cuda_graph`` correctly detects cross-rank
    state mismatch via ``tp_allgather`` and enters ``BypassCapturedGraphs()``
    around the call when ANY rank reports ``can_run_cuda_graph == False``.
    Captured graphs whose shapes happen to match are bypassed too — this is the
    actual cross-rank divergence the conc=32 NaN bug exposed when one rank ran
    prefill (eager) while another ran decode (replay) with a stale capture-time
    ``runtime_max_tokens_per_rank``.

The cross-rank decision is propagated via the process-wide
``cuda_graph_state.BYPASS`` flag toggled by ``BypassCapturedGraphs()``, NOT by
traversing ``model.modules()`` and toggling a per-instance flag (the previous
approach was both fragile to renames/subclasses and an unpythonic instance-state
mutation). End-to-end MoE-A2A coverage is provided by
``TestNemotronSuperV3::test_accuracy[*-4-attn_dp_on-trtllm]`` (post-merge).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from _dist_test_utils import get_device_counts

from tensorrt_llm._torch.auto_deploy.compile.backends.torch_cudagraph import CapturedGraph
from tensorrt_llm._torch.auto_deploy.distributed.common import spawn_multiprocess_job
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import maybe_pad_for_cuda_graph
from tensorrt_llm._torch.auto_deploy.utils.cuda_graph import BypassCapturedGraphs, cuda_graph_state


class _SlotReadingModel(nn.Module):
    """Mimics a capture-time scalar read baked into the MoE A2A op's kernel launch.

    The Python int returned by ``.item()`` is consumed as a kernel-launch
    argument by ``torch.full``; under cuda graph capture that argument value is
    recorded into the graph and reused at replay irrespective of the current
    host tensor value. The eager path (uncaptured shape / under
    ``BypassCapturedGraphs()``) re-runs the Python body and reads the host fresh.
    """

    def __init__(self, host: torch.Tensor):
        super().__init__()
        self._host = host

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = int(self._host[0].item())
        return torch.full((x.shape[0],), float(v), device="cuda")


def _run_bypass_test(rank: int, world_size: int) -> None:
    device = "cuda"

    # ----- Level 1 setup: tiny CapturedGraph over the slot-reading model -----
    host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    inner = _SlotReadingModel(host)
    cg = CapturedGraph(inner, num_batched_inputs=1)

    capture_bs = [1, 2, 4]

    def get_args_kwargs(bs: int):
        # Mimic the MoE A2A op deriving its per-rank dispatch budget from the
        # capture-time input shape. Each captured graph thus bakes its own value.
        host[0] = bs
        return (torch.zeros(bs, device=device),), {}

    cg.capture_graph(get_args_kwargs, capture_bs)

    # ----- Level 2 setup: minimal fake ADEngine for maybe_pad_for_cuda_graph --
    # The wrapper accesses: cuda_graph_used, enable_attention_dp, dist_config.tp_size,
    # dist.tp_allgather, padding_dummy_request, cuda_graph_batch_sizes.
    # We fake all of them with MagicMock and assign concrete values where the
    # wrapper's conditionals depend on the value.
    fake_engine = MagicMock()
    fake_engine.cuda_graph_used = True
    fake_engine.enable_attention_dp = True
    fake_engine.dist_config = SimpleNamespace(tp_size=world_size)
    # Non-None so the wrapper's "create dummy" branch is skipped on first call.
    fake_engine.padding_dummy_request = MagicMock()
    fake_engine.cuda_graph_batch_sizes = capture_bs
    fake_engine.max_total_draft_tokens = 0
    fake_engine.max_beam_width = 1

    def real_tp_allgather(local_obj):
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_obj)
        return gathered

    fake_engine.dist = SimpleNamespace(tp_allgather=real_tp_allgather)

    def fake_forward(self, scheduled_requests, resource_manager, *args, **kwargs):
        # Mimic ADEngine.forward: gather cross-rank totals, write to slot, run model.
        # The captured graph's `int(host[0].item())` happens INSIDE cg(...) — so
        # under replay it returns the baked value, under bypass it reads fresh.
        all_locals = list(self.dist.tp_allgather(scheduled_requests.local_total))
        host[0] = max(all_locals)
        return cg(torch.zeros(scheduled_requests.batch_size, device=device))

    wrapped = maybe_pad_for_cuda_graph(fake_forward)

    def mk_scheduled(can_cg: bool, bs: int, local_total: int):
        sr = MagicMock()
        sr.can_run_cuda_graph = can_cg
        sr.batch_size = bs
        sr.local_total = local_total
        # Real list — the wrapper's padding branch calls .extend / [:-num_padding]
        sr.generation_requests = []
        return sr

    # ===== Scenario A (Level 1): all ranks can run cuda graph -> REPLAY =====
    # All ranks have can_cg=True, bs=1, local_total=1; cross-rank max also = 1.
    # Wrapper allows replay; captured graph for bs=1 returns its baked value (= 1).
    sr_a = mk_scheduled(can_cg=True, bs=1, local_total=1)
    out_a = wrapped(fake_engine, sr_a, MagicMock())
    assert int(out_a[0].item()) == 1, (
        f"rank={rank} scenario A (REPLAY): expected baked capture-time value 1, "
        f"got {out_a.tolist()}"
    )
    assert cuda_graph_state.in_bypass() is False, (
        f"rank={rank} scenario A: bypass state should remain False after replay"
    )

    # ===== Scenario B (Level 2): rank 0 prefill, rank 1 decode -> BYPASS =====
    # Cross-rank state diverges; wrapper must allgather, see at least one False,
    # enter BypassCapturedGraphs() around the call, and run eager. Eager reads
    # host fresh -> value = max(local_total across ranks).
    bypass_state_seen_during_call: list[bool] = []

    def fake_forward_probed(self, scheduled_requests, resource_manager, *args, **kwargs):
        # Records the global bypass state at call time. Captured graphs honour
        # this same state via cuda_graph_state.in_bypass() in their forward().
        bypass_state_seen_during_call.append(cuda_graph_state.in_bypass())
        all_locals = list(self.dist.tp_allgather(scheduled_requests.local_total))
        host[0] = max(all_locals)
        return cg(torch.zeros(scheduled_requests.batch_size, device=device))

    wrapped_probed = maybe_pad_for_cuda_graph(fake_forward_probed)

    sr_b = mk_scheduled(
        can_cg=(rank == 1),  # only rank 1 says yes
        bs=1,
        local_total=(99 if rank == 0 else 1),  # cross-rank max should be 99
    )
    out_b = wrapped_probed(fake_engine, sr_b, MagicMock())
    assert int(out_b[0].item()) == 99, (
        f"rank={rank} scenario B (BYPASS-EAGER): expected fresh cross-rank max 99, "
        f"got {out_b.tolist()}"
    )
    assert bypass_state_seen_during_call == [True], (
        f"rank={rank} scenario B: cuda_graph_state.in_bypass() must be True during the "
        f"wrapped call (so CapturedGraph submodules skip replay), got history "
        f"{bypass_state_seen_during_call}"
    )
    assert cuda_graph_state.in_bypass() is False, (
        f"rank={rank} scenario B: bypass state must be RESTORED to False after the "
        f"BypassCapturedGraphs() context exits (try/finally), got "
        f"{cuda_graph_state.in_bypass()}"
    )

    # ===== Scenario C: after bypass cleared, replay resumes -> baked value ====
    sr_c = mk_scheduled(can_cg=True, bs=1, local_total=1)
    out_c = wrapped(fake_engine, sr_c, MagicMock())
    assert int(out_c[0].item()) == 1, (
        f"rank={rank} scenario C: replay should resume baked behaviour after bypass cleared, "
        f"got {out_c.tolist()}"
    )

    # ===== Scenario D: direct context-manager smoke test (no wrapper) =========
    # Confirms BypassCapturedGraphs() alone is sufficient to short-circuit a
    # CapturedGraph instance to eager — independent of maybe_pad_for_cuda_graph.
    host[0] = 7919
    out_d_no_bypass = cg(torch.zeros(1, device=device))
    assert int(out_d_no_bypass[0].item()) == 1, (
        f"rank={rank} scenario D pre-context: baked replay should yield 1, got {out_d_no_bypass}"
    )
    with BypassCapturedGraphs():
        assert cuda_graph_state.in_bypass() is True
        out_d_bypass = cg(torch.zeros(1, device=device))
    assert cuda_graph_state.in_bypass() is False
    assert int(out_d_bypass[0].item()) == 7919, (
        f"rank={rank} scenario D inside-context: eager should read fresh slot 7919, "
        f"got {out_d_bypass}"
    )


@pytest.mark.parametrize("device_count", get_device_counts([2]))
def test_bypass_captured_graphs(device_count):
    spawn_multiprocess_job(job=_run_bypass_test, size=device_count)
