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

HEADS, HEAD_DIM, DSTATE, NGROUPS, WIDTH = 8, 80, 128, 1, 4
CONV_DIM = HEADS * HEAD_DIM + 2 * NGROUPS * DSTATE  # [x | B | C], as in Mamba2Mixer
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


def test_per_request_inputs_aligned_independently():
    """In one MTP state promote, ``src_state_indices`` starts at row 0 (always
    aligned) while ``dst_state_indices`` and the accepted counts start after
    the context rows, and the launcher passes the destination indices again
    as its placeholder pointers. Warmup records one alignment mix; serving
    sees the others. Every one must be planned."""
    from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager as mcm

    dev = torch.device("cuda")
    dst = torch.zeros(2, CACHE, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    inter = torch.zeros(2, MAX_ROWS, 4, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    slots = torch.arange(MAX_ROWS, dtype=torch.int32, device=dev)
    accepted = torch.ones(MAX_ROWS, dtype=torch.int32, device=dev)

    def promote(nctx, ngen):
        mcm._promote_mamba_state_triton(
            dst,
            inter,
            slots[:ngen],
            accepted[nctx : nctx + ngen],
            slots[nctx : nctx + ngen],
            position_source_is_token_count=True,
        )

    jpd._RECORDS.clear()
    jpd.start_recording()
    try:
        promote(0, 4)  # warmup: no context rows, everything aligned
        torch.cuda.synchronize()
    finally:
        jpd.stop_recording()
    planned = _keys(jpd.MambaDecodeProvider(max_batch_size=MAX_ROWS).plan_all())
    for nctx, ngen in [(0, 1), (1, 1), (1, 3), (3, 16), (4, 5), (5, 33)]:
        missing = _real(promote, nctx, ngen) - planned
        assert not missing, f"nctx={nctx} ngen={ngen}: unplanned {sorted(missing)}"


def test_a_second_warmup_on_another_pool_is_planned_too():
    """The KV-cache estimation pass and the final executor warm up on pools
    of different sizes (the conv update specializes on the pool's size). A
    provider enumerated after each warmup plans the second pool's launches,
    and does not plan the first pool's again."""
    estimation, final = _Buffers(), _Buffers()
    final.conv_state = torch.zeros(
        2 * CACHE, WIDTH - 1, CONV_DIM, dtype=torch.bfloat16, device="cuda"
    ).transpose(1, 2)  # the same layout as _Buffers, twice the cache lines
    jpd._RECORDS.clear()
    provider = jpd.MambaDecodeProvider(max_batch_size=MAX_ROWS)
    for warmup in (estimation, final):
        jpd.start_recording()
        try:
            warmup.conv(0, 4)
            torch.cuda.synchronize()
        finally:
            jpd.stop_recording()
        planned = _keys(provider.plan_all())
        assert planned, "a warmup with new launch classes planned nothing"
    for nctx, ngen in [(0, 1), (3, 16), (5, 33)]:
        missing = _real(final.conv, nctx, ngen) - planned
        assert not missing, f"final pool nctx={nctx} ngen={ngen}: unplanned {sorted(missing)}"
    assert not provider.plan_all(), "launch classes planned twice"


def test_helper_compiles_every_planned_decode_kernel(tmp_path):
    """The CPU-only Triton helper imports a kernel's module with stand-ins for
    the rest of tensorrt_llm; every kernel the decode provider plans (also the
    cache manager's state promote) must compile there."""
    import json
    import os
    import subprocess
    import sys

    from tensorrt_llm._torch.pyexecutor.kv_cache import mamba_cache_manager as mcm

    buf = _Buffers()
    jpd._RECORDS.clear()
    jpd.start_recording()
    try:
        buf.ssu(0, 4)
        buf.conv(0, 4)
        dst = torch.zeros(2, CACHE, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        inter = torch.zeros(2, MAX_ROWS, 4, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        rows = buf.state_indices[:4]
        mcm._promote_mamba_state_triton(dst, inter, rows, rows, rows)
        torch.cuda.synchronize()
    finally:
        jpd.stop_recording()
    assert set(jpd.recorded_launchers()) == {
        "selective_state_update",
        "causal_conv1d_update",
        "_promote_mamba_state_triton",
    }
    specs, names = [], set()
    for call in jpd.MambaDecodeProvider(max_batch_size=MAX_ROWS).plan_all():
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        found = jp._lookup_or_serialize(jit_fn, call.args, variants[0])
        if found is not None and jit_fn.fn.__name__ not in names:
            names.add(jit_fn.fn.__name__)
            specs.append(found[1])
    assert "_promote_mamba_state_kernel" in names, names
    helper = os.path.join(os.path.dirname(os.path.abspath(jp.__file__)), "jit_prefetch_helper.py")
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(jp.__file__))))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1",
               TRITON_CACHE_DIR=str(tmp_path))  # fmt: skip
    env.pop("PYTHONPATH", None)
    lines = "".join(json.dumps({"tag": i, "spec": s}) + "\n" for i, s in enumerate(specs))
    out = subprocess.run([sys.executable, "-u", helper, pkg_root], input=lines,
                         capture_output=True, text=True, env=env, timeout=600).stdout  # fmt: skip
    replies = [json.loads(line) for line in out.splitlines()][1:]
    assert len(replies) == len(specs) and all(r["ok"] for r in replies), replies


def test_recording_stops_and_restores_launchers():
    from tensorrt_llm._torch.modules.mamba import selective_state_update as ssu

    orig = ssu.selective_state_update
    jpd.start_recording()
    assert ssu.selective_state_update is not orig
    jpd.stop_recording()
    assert ssu.selective_state_update is orig


def _mixer_decode(buf, nreq_ctx, nctx_tokens, ngen):
    """``Mamba2Mixer.forward_core``'s non-speculative decode call of the
    native state update, for ``ngen`` generation rows behind ``nctx_tokens``
    context tokens of ``nreq_ctx`` context requests."""
    from einops import rearrange, repeat

    from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update

    rows = slice(nctx_tokens, nctx_tokens + ngen)
    di = HEADS * HEAD_DIM
    x_d, B_d, C_d = torch.split(buf.xbc[rows], [di, NGROUPS * DSTATE, NGROUPS * DSTATE], dim=-1)
    x_d = rearrange(x_d, "b (h p) -> b h p", p=HEAD_DIM).contiguous()
    dt_d = repeat(buf.dt[rows].to(torch.float32), "b h -> b h p", p=HEAD_DIM)
    B_d = rearrange(B_d, "b (g n) -> b g n", g=NGROUPS).contiguous()
    C_d = rearrange(C_d, "b (g n) -> b g n", g=NGROUPS).contiguous()
    selective_state_update(
        buf.state, x_d, dt_d, buf.A, B_d, C_d, buf.D, z=None, dt_bias=buf.dt_bias,
        dt_softplus=True, state_batch_indices=buf.state_indices[nreq_ctx : nreq_ctx + ngen],
        out=buf.out[rows].view(ngen, -1, HEAD_DIM),
    )  # fmt: skip


def test_seeded_native_decode_class_covers_the_mixer_decode():
    """Before any decode step, the native state-update class seeded from a
    mixer and its cache manager plans the launches the mixer's
    non-speculative decode makes, at any batch size and behind any number of
    context rows; a speculative cache manager seeds nothing."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.mamba import selective_state_update as ssu
    from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer

    buf = _Buffers()
    mixer = object.__new__(Mamba2Mixer)
    torch.nn.Module.__init__(mixer)
    for name, value in (("selective_state_update_func", ssu.selective_state_update),
                        ("layer_idx", 0), ("tp_nheads", HEADS), ("head_dim", HEAD_DIM),
                        ("tp_ngroups", NGROUPS), ("d_state", DSTATE), ("delta_softplus", True),
                        ("norm", SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))),
                        ("_A_expanded", buf.A), ("_D_expanded", buf.D),
                        ("_dt_bias_expanded", buf.dt_bias)):  # fmt: skip
        object.__setattr__(mixer, name, value)
    model = torch.nn.Sequential(mixer)
    layer = SimpleNamespace(temporal=buf.state)
    jpd._RECORDS.clear()
    speculative = SimpleNamespace(is_speculative=lambda: True, mamba_layer_cache=lambda i: layer)
    assert jpd.seed_native_decode_records(model, speculative) == 0
    manager = SimpleNamespace(is_speculative=lambda: False, mamba_layer_cache=lambda i: layer)
    assert jpd.seed_native_decode_records(model, manager) == 1
    provider = jpd.MambaDecodeProvider(max_batch_size=MAX_ROWS)
    (batch,) = list(provider.early_batches())
    planned = _keys(provider(batch))
    for nreq_ctx, nctx_tokens, ngen in ((0, 0, 1), (0, 0, 16), (1, 3, 5), (2, 9, 16), (0, 0, 37)):
        with jp.shadow_launches() as rec:
            _mixer_decode(buf, nreq_ctx, nctx_tokens, ngen)
        assert _keys(rec) <= planned, (nreq_ctx, nctx_tokens, ngen)
    jpd._RECORDS.clear()
