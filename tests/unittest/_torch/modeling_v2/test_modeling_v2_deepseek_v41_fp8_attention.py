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
"""The DeepSeek-V4.1-Flash-NVFP4 target's opt-in FP8 attention (TRTLLM_DSV41_FP8_ATTENTION=1),
on the real kernels.

* The KV scales: one power of two per compressed-source group (every layer that reads a
  source shares it, because one attention call takes one scale for both pools) and one per
  window-only layer, the smallest that keeps the norm-weight bound below 448 x scale.
* The window read: the E4M3 read buffer is, row for row, the bf16 read buffer divided by the
  layer's scale and rounded to E4M3, with the same renumbered index table.
* The compressed read in both forms of the one read buffer: per request (every entry of every
  request) and per query (the selected top-k rows only). Each E4M3 form is the bf16 one scaled
  and rounded, and the FP8 attention over either form is bitwise the same.

Each case drives the target's own methods on a stand-in core holding just the attributes
they read, and a stand-in prefill target holding just the entries they call.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    modeling,
)

Core = modeling.DeepseekV41UnifiedModelingV2
Prefill = modeling.PrefillTarget
SRC = 20
E4 = torch.float8_e4m3fn


def _e4m3_of(bf16_rows: torch.Tensor, scale: float) -> torch.Tensor:
    """uint8 codes of bf16 rows divided by a power-of-two scale (exact), rounded to E4M3."""
    return (bf16_rows.float() / scale).clamp(-448.0, 448.0).to(E4).view(torch.uint8)


def _core(max_t: int, fp8: bool, scale: float):
    return SimpleNamespace(
        w={"final_norm": torch.empty(1, device="cuda")},
        head_dim=512,
        ratios={SRC: 1},
        kv_sources=[SRC],
        kv_src={SRC: SRC},
        pp_layer_ids=list(range(40)),  # one pipeline stage: every layer is local
        local=frozenset(range(40)),
        rank=0,
        fp8_attn=fp8,
        _fp8_kv_scale={SRC: scale},
        _const={"one_f32": torch.ones(1, dtype=torch.float32, device="cuda")},
    ), SimpleNamespace(swa_local_indices_cuda=torch.empty(max_t, 128, device="cuda"))


def _target(core):
    """A stand-in prefill target: the core it reads and the read-back entries it calls."""
    return SimpleNamespace(
        core=core,
        _swa_read=modeling.Mxfp8Dequantize(),
        _swa_read8=modeling.Mxfp8ToE4m3(),
        _comp_read=modeling.Nvfp4KvToBf16(),
        _comp_read8=modeling.Nvfp4KvToE4m3(),
    )


def test_kv_scales_group_by_compressed_source():
    """Layers 0-1 window-only, 2-19 ratio 2 over sources 2 / 8 / 14, 20-39 ratio 1 over source 20
    (the checkpoint's role table), with per-layer norm-weight maxima spread over 0.4 .. 6.4."""
    n = 40
    ratios = [0, 0] + [2] * 18 + [1] * 20
    kv_sources = (2, 8, 14, 20)
    kv_src = [None if not r else max(s for s in kv_sources if s <= i) for i, r in enumerate(ratios)]
    g = torch.Generator().manual_seed(0)
    wmax = [0.4 + 6.0 * float(torch.rand(1, generator=g)) for _ in range(n)]
    cmax = {s: 0.4 + 6.0 * float(torch.rand(1, generator=g)) for s in kv_sources}
    w = {}
    for i in range(n):
        w[f"l{i}_kv_norm"] = torch.full((512,), wmax[i]).to(torch.bfloat16)
        w[f"l{i}_kv_norm"][7] = -wmax[i]  # the bound is on |w|
    for s in kv_sources:
        w[f"l{s}_comp_norm"] = torch.full((512,), cmax[s]).to(torch.bfloat16)
    core = SimpleNamespace(
        w=w,
        num_layers=n,
        pp_layer_ids=list(range(n)),
        ratios=ratios,
        kv_src=kv_src,
        rank=1,
        _fp8_kv_scale=[None] * n,
    )
    Core._derive_fp8_kv_scales(core)
    s = core._fp8_kv_scale
    groups = {}
    for i in range(n):
        groups.setdefault(kv_src[i] if ratios[i] else -1 - i, []).append(i)
    assert len(groups) == 2 + len(kv_sources)
    for key, layers in groups.items():
        bound = max(float(w[f"l{i}_kv_norm"].abs().max()) for i in layers)
        if key >= 0:
            bound = max(bound, float(w[f"l{key}_comp_norm"].abs().max()))
        bound *= modeling._FP8_KV_BOUND
        scales = {s[i] for i in layers}
        assert len(scales) == 1, (key, scales)
        sc = scales.pop()
        assert math.log2(sc) == int(math.log2(sc)), sc
        assert bound <= 448.0 * sc < 2 * bound, (
            key,
            bound,
            sc,
        )  # never saturates, and is the smallest


@skip_pre_blackwell
def test_window_read_is_the_bf16_read_scaled_to_e4m3():
    torch.manual_seed(1)
    pages, T, scale = 64, 300, 2.0**-4
    pool = torch.zeros(pages, 128, 528, dtype=torch.uint8, device="cuda")
    x = torch.randn(pages * 128, 16, 32, device="cuda") * torch.exp2(
        torch.randint(-10, 5, (pages * 128, 16, 1), device="cuda").float()
    )
    amax = x.abs().amax(-1).clamp_min(1e-4)
    m, e = torch.frexp(amax.double() / 448.0)
    e = torch.where(m == 0.5, e - 1, e).to(torch.int32)
    q = (x / torch.exp2(e.double()).float().unsqueeze(-1)).clamp(-448, 448).to(E4)
    flat = pool.view(-1, 528)
    flat[:, :512] = q.view(-1, 512).view(torch.uint8)
    flat[:, 512:] = (e + 127).to(torch.uint8)
    codes, scales = pool[..., :512], pool[..., 512:]
    gidx = torch.randint(0, pages * 128, (T, 128), dtype=torch.int32, device="cuda")
    gidx[::7, 100:] = -1
    outs = {}
    for fp8 in (False, True):
        core, md = _core(8192, fp8, scale)
        core._fp8_kv_scale = {0: scale}
        md.compress_block_tables = {1: torch.zeros(4, 8, dtype=torch.int32, device="cuda")}
        pools = Core._native_read_buffers(core, md)
        pools["swa_q"] = {0: (codes, scales)}
        ren, swa4 = Prefill._native_window(_target(core), 0, gidx, T, pools)
        assert swa4.dtype == (E4 if fp8 else torch.bfloat16)
        outs[fp8] = (ren.clone(), pools["swa_c"][: T * 128].clone())
    assert torch.equal(outs[False][0], outs[True][0])
    assert torch.equal(outs[True][1].view(torch.uint8), _e4m3_of(outs[False][1], scale))


@skip_pre_blackwell
def test_compressed_read_both_forms_and_attention_are_consistent():
    """The bf16 and E4M3 buffers agree row for row (E4M3 = bf16 / scale, rounded) in both forms, and
    the FP8 attention over the per-request and the per-query form is bitwise the same."""
    torch.manual_seed(0)
    H, D, bk, pages, cols, scale = 16, 512, 128, 200, 40, 2.0**-1
    lens = [3000, 5100, 4097, 1]
    seqs = len(lens)
    pool = torch.randint(0, 256, (pages, bk, 288), dtype=torch.uint8, device="cuda")
    pool[..., 256:] = torch.randint(0, 0x70, (pages, bk, 32), dtype=torch.uint8, device="cuda")
    codes, scales = pool[..., :256], pool[..., 256:]
    comp_q4 = {SRC: (codes.reshape(pages, bk, 1, 256), scales.reshape(pages, bk, 1, 32))}
    comp_q1 = {SRC: (codes.reshape(pages * bk, 1, 1, 256), scales.reshape(pages * bk, 1, 1, 32))}
    tbl = (
        torch.randperm(pages, device="cuda")[: seqs * cols]
        .view(seqs, cols)
        .to(torch.int32)
        .contiguous()
    )
    klens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    topk = torch.full((seqs, modeling._TOPK), -1, dtype=torch.int32, device="cuda")
    for b, n in enumerate(lens):
        pick = torch.randperm(n, device="cuda")[: min(n, modeling._TOPK)].sort().values
        topk[b, : pick.numel()] = pick.to(torch.int32)
    count = (topk >= 0).sum(1).to(torch.int32)
    valid = topk >= 0
    e = topk.clamp(min=0).long()
    b_idx = torch.arange(seqs, device="cuda").view(-1, 1)
    per_request = torch.where(valid, b_idx * cols * bk + e, -1).to(torch.int32)
    slots = torch.where(valid, tbl.long()[b_idx, e // bk] * bk + e % bk, -1).to(torch.int32)

    swa = (torch.randn(4, 1, 128, D, device="cuda") * 4).to(E4)
    win = torch.randint(0, 4 * 128, (seqs, 128), dtype=torch.int32, device="cuda")
    q = (torch.randn(seqs, 1, H, D, device="cuda") * 8).to(E4)
    sinks = torch.randn(H, device="cuda")
    ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    topk_lens = (128 + count).contiguous()
    bmm1 = D**-0.5 * 2.0**-3 * scale

    def attend(gidx, comp_pool):
        return modeling.TrtllmSparseMlaDsv4Fp8()(
            q, swa, comp_pool, ws, gidx, topk_lens, klens.clamp(min=128), sinks, bmm1, scale
        )

    rows = {}
    out = {}
    for fp8 in (False, True):
        # One buffer that holds either form (4 request slots of 40 x 128 rows, 40 query rows).
        core, md = _core(64, fp8, scale)
        md.compress_block_tables = {1: tbl}
        md.compressed_kv_lens_cuda = {1: klens}
        pools = Core._native_read_buffers(core, md)
        assert pools["comp_req_fit"] == seqs and pools["comp_q_fit"] >= seqs
        pools.update(comp_q4=comp_q4, comp_q1=comp_q1)
        for form in ("request", "topk"):
            c4 = pools["comp_c4_req" if form == "request" else "comp_c4_q"][1]
            assert c4.dtype == (E4 if fp8 else torch.bfloat16)
            if form == "request":
                Prefill._native_compressed(_target(core), SRC, {"md": md, "B": seqs}, pools)
                gidx = torch.cat([win, per_request], 1)
                sel = per_request
            else:
                pools["topk_out"].view(torch.uint8).fill_(0x5A)  # stale finite rows, never attended
                gidx = Prefill._native_topk(
                    _target(core), SRC, torch.cat([win, slots], 1), topk, seqs, pools, {}
                )
                assert torch.equal(gidx[:, 128:], pools["topk_ren"][:seqs])
                sel = pools["topk_ren"][:seqs]
            flat = c4.reshape(-1, D)
            rows[form, fp8] = flat[sel.clamp(min=0).long()][valid].clone()
            if fp8:
                out[form] = attend(gidx, c4)
    for form in ("request", "topk"):
        assert torch.equal(
            rows[form, True].view(torch.uint8), _e4m3_of(rows[form, False], scale)
        ), form
    assert torch.equal(rows["request", False], rows["topk", False])
    assert torch.equal(out["request"], out["topk"])
    assert torch.isfinite(out["request"].float()).all()
