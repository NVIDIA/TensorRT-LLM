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
"""The Mamba2 decode-side JIT-prefetch planner covers other batches.

A recorded warmup-like decode call of each launcher (the native selective
state update, the Triton conv update, the MTP state promote) fills the
records; the planner's replays must then cover the real launches of other
decode batch sizes whose per-request inputs start at other rows of their
buffers (generation rows behind any number of context rows).
"""

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.modules.mamba import jit_prefetch_decode as jpd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HEADS, HEAD_DIM, DSTATE, NGROUPS, CONV_DIM, WIDTH = 8, 80, 128, 1, 2304, 4
CACHE, MAX_ROWS = 64, 64


def _keys(calls):
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key

    keys = set()
    for call in calls:
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        for kw in variants:
            kw = dict(kw)
            kw["debug"] = kw.get("debug", jit_fn.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            _, kkc, _, _, binder = jit_fn.device_caches[driver.active.get_current_device()]
            _, spec, opts = binder(*call.args, **kw)
            keys.add((jit_fn.fn.__name__, compute_cache_key(kkc, spec, opts)))
    return keys


class _Buffers:
    """Engine-sized buffers; a decode batch is a row slice of them."""

    def __init__(self):
        dev = torch.device("cuda")
        f32 = dict(dtype=torch.float32, device=dev)
        self.state = torch.zeros(CACHE, HEADS, HEAD_DIM, DSTATE, dtype=torch.bfloat16, device=dev)
        self.xbc = torch.zeros(MAX_ROWS, CONV_DIM, dtype=torch.bfloat16, device=dev)
        self.dt = torch.zeros(MAX_ROWS, HEADS, dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros(MAX_ROWS, HEADS * HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.state_indices = torch.arange(MAX_ROWS, dtype=torch.int32, device=dev)
        self.A = torch.zeros(HEADS, HEAD_DIM, DSTATE, **f32)
        self.dt_bias = torch.zeros(HEADS, HEAD_DIM, **f32)
        self.D = torch.zeros(HEADS, HEAD_DIM, **f32)
        self.conv_state = torch.zeros(CACHE, CONV_DIM, WIDTH - 1, dtype=torch.bfloat16, device=dev)
        self.conv_state = self.conv_state.transpose(1, 2).contiguous().transpose(1, 2)
        self.conv_w = torch.zeros(CONV_DIM, WIDTH, dtype=torch.bfloat16, device=dev)
        self.conv_b = torch.zeros(CONV_DIM, dtype=torch.bfloat16, device=dev)

    def ssu(self, nctx, ngen):
        """The mixer's non-MTP decode: x/B/C from xbc rows nctx.., dt rows."""
        from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update

        rows = slice(nctx, nctx + ngen)
        di = HEADS * HEAD_DIM
        x = self.xbc[rows, :di].contiguous().view(ngen, HEADS, HEAD_DIM)
        B = self.xbc[rows, di : di + NGROUPS * DSTATE].contiguous().view(ngen, NGROUPS, DSTATE)
        C = self.xbc[rows, di + NGROUPS * DSTATE :].contiguous().view(ngen, NGROUPS, DSTATE)
        dt = self.dt[rows].unsqueeze(-1).expand(ngen, HEADS, HEAD_DIM)
        selective_state_update(
            self.state,
            x,
            dt,
            self.A,
            B,
            C,
            self.D,
            z=None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            state_batch_indices=self.state_indices[rows],
            out=self.out[rows].view(ngen, HEADS, HEAD_DIM),
        )

    def conv(self, nctx, ngen):
        from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update

        rows = slice(nctx, nctx + ngen)
        causal_conv1d_update(
            self.xbc[rows].unsqueeze(-1),
            self.conv_state,
            self.conv_w,
            self.conv_b,
            activation="silu",
            conv_state_indices=self.state_indices[rows],
        )


def _record_then_plan(buf):
    jpd._RECORDS.clear()
    jpd.start_recording()
    try:
        buf.ssu(0, 4)
        buf.conv(0, 4)
        torch.cuda.synchronize()
    finally:
        jpd.stop_recording()
    provider = jpd.MambaDecodeProvider(max_batch_size=MAX_ROWS)
    assert provider, "warmup-like calls recorded nothing"
    return _keys(provider.plan_all())


def _real(fn, nctx, ngen):
    with jp.shadow_launches() as rec:
        fn(nctx, ngen)
    return _keys(rec)


def test_planned_keys_cover_other_batches_and_row_offsets():
    buf = _Buffers()
    planned = _record_then_plan(buf)
    for nctx, ngen in [(0, 1), (0, 2), (0, 16), (0, 17), (1, 3), (3, 16), (5, 33), (2, 40)]:
        for name, fn in (("ssu", buf.ssu), ("conv", buf.conv)):
            real = _real(fn, nctx, ngen)
            assert real, f"{name} nctx={nctx} ngen={ngen}: no launches"
            missing = real - planned
            assert not missing, f"{name} nctx={nctx} ngen={ngen}: unplanned {sorted(missing)}"


def test_recording_stops_and_restores_launchers():
    from tensorrt_llm._torch.modules.mamba import selective_state_update as ssu

    orig = ssu.selective_state_update
    jpd.start_recording()
    assert ssu.selective_state_update is not orig
    jpd.stop_recording()
    assert ssu.selective_state_update is orig
