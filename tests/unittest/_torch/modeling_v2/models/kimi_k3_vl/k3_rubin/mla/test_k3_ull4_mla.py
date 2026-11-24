# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi-K3 ULL MLA attention-side scheduling switch against today's path, one GPU.

``TRTLLM_K3_ULL4_MLA_PLAN_EARLY``: the step's FP4 MLA metadata (generation lengths fused with the
packed split plan, the device page table) written ahead on a side stream
(``prelaunch_fp4_mla_generation_metadata``): the layer's KV update and decode then launch no
metadata kernel and give the same cache bytes, outputs and statistics.

Helpers: ``test_k3_ull2_mla.py`` (FP4 chain, same directory).
"""

import math

import pytest
import test_k3_ull2_mla as u2
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)


@pytest.fixture(autouse=True)
def _k3_decode_kernel(monkeypatch):
    u2.qf.use_k3_decode_kernel(monkeypatch)


def _plan(md):
    """The packed split plan workspace of the step."""
    return md.fp4_mla_state.workspaces["_fp4_mla_packed_plan"]


@u2.qf.needs_k3_decode
@pytest.mark.parametrize("n", [1, 2])
def test_step_metadata_prelaunch(n, monkeypatch):
    """The step metadata written ahead on a side stream: the chain's KV update and decode launch
    no metadata kernel, and the cache bytes, decode output and statistics are bitwise those of
    the chain that writes the metadata itself; also captured in a CUDA graph."""
    from tensorrt_llm._torch.attention.backends import fp4_mla

    monkeypatch.setenv("TRTLLM_FP4_MLA_SPLIT_KV", "8")
    monkeypatch.setenv(u2.PDL_ENV, "1")
    chain = u2._Chain(n, 5000 + 1000 * n, seed=80 + n)
    md = chain.md
    state = md.fp4_mla_state
    try:
        # What the metadata resets at every step (Fp4MlaState.invalidate_generation_lengths).
        state.invalidate_generation_lengths()
        chain.run(early_trigger=True)
        # Eager plans hold spare task slots the plan kernel leaves untouched: compare against a
        # reference written over the same fill.
        _plan(md).fill_(-1)
        state.invalidate_generation_lengths()
        chain.run(early_trigger=True)
        ref = chain.result()
        ref_plan = _plan(md).clone()
        ref_lens = state.generation_kv_lens.clone()
        pools = [p.clone() for p in u2.qf._pools(chain.mgr)]
        assert state.packed_plan_key is not None

        def prelaunched_run():
            state.invalidate_generation_lengths()
            side = torch.cuda.Stream()
            fork, done = torch.cuda.Event(), torch.cuda.Event()
            fork.record()
            with torch.cuda.stream(side):
                fork.wait()
                launched = fp4_mla.prelaunch_fp4_mla_generation_metadata(
                    md, num_heads=u2.H, use_helix=True, device=torch.device("cuda")
                )
                done.record()
            torch.cuda.current_stream().wait_event(done)
            before = state.metadata_launches
            chain.run(early_trigger=True)
            return launched, state.metadata_launches - before

        chain.out.fill_(torch.nan)
        _plan(md).fill_(-1)
        plan_ptr = _plan(md).data_ptr()
        launched, extra_launches = prelaunched_run()
        assert launched and extra_launches == 0
        # An index-less device must not reallocate the plan workspace.
        assert _plan(md).data_ptr() == plan_ptr
        u2._assert_bitwise(chain.result(), ref)
        assert torch.equal(_plan(md), ref_plan)
        assert torch.equal(state.generation_kv_lens, ref_lens)
        for got, want in zip(u2.qf._pools(chain.mgr), pools):
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
        # Captured: the prelaunch, the join and the chain in one graph.
        prelaunched_run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prelaunched_run()
        for _ in range(2):
            chain.out.fill_(torch.nan)
            _plan(md).fill_(-1)
            graph.replay()
            u2._assert_bitwise(chain.result(), ref)
            assert torch.equal(_plan(md), ref_plan)
        assert math.isfinite(float(ref[0].float().abs().max()))
    finally:
        chain.mgr.shutdown()
