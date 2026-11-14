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
"""Kimi-K3 fused FP4 MLA query (``k3_mla_q_fused``) against the default query chain.

The FP4 cache, HP ring and Helix per-token metadata are built through the real
``Fp4MlaKVCacheManagerV2`` (context scatter, then one generation step of ``q_len`` tokens per
request), with the per-batch FP4 MLA state on ``metadata.fp4_mla_state``; the decode is K3's own
(``ops/attention/_fp4_mla_decode``). The default chain is the production one: the q_b low-M GEMM,
the BF16 W_UK BMM and the fused FP4 generation update with its query rows. Checks:

* The KV append without query rows (``kernels/k3r_fp4_mla/fp4_mla_kv_append``) writes the cache
  bytes of the backend's full update (bitwise).
* The fused kernel's query quantization equals the Triton kernel's on the same BF16 values
  (bitwise), including the RoPE rotation with a non-identity table.
* Its q_b and W_UK products agree with the production GEMMs up to BF16 rounding of a different
  FP32 summation order.
* With the q_a RMSNorm inside the kernel the query rows are those after FlashInfer's separate
  norm (bitwise), and ``k3_mla_latent_row`` is bitwise FlashInfer's norm plus the ``k_pe`` copy.
* ``generation.forward_gen_q_fused`` (multi-stream, KV half on the aux stream) gives
  the default chain's cache bytes and, through K3's FP4 decode, its attention output within the
  FP4 query quantization.
* The ``k3_mla_q_fused`` entry on its ``.CELLS`` is within one FP4 step of its plain-torch
  reference.

``test_k3_fp4_mla_decode.py`` builds on the FP4 cache helpers here.
"""

import functools
import math
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["../../../.."]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

H, QL, LORA, ROPE, PAGE = 96, 1536, 512, 64, 128
Q_LEN = 8
EPS = 1e-6


def k3_decode_unavailable() -> bool:
    """Whether the packed CuTe DSL FP4 MLA decode cannot run here (SM107 with a CuTe DSL that
    builds the FP4 MLA kernels)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return True
    from tensorrt_llm._torch.attention.backends.fp4_mla.config import _cutedsl_backend_available

    return not _cutedsl_backend_available()


# Tests that run K3's FP4 MLA decode (``ops/attention/_fp4_mla_decode``).
needs_k3_decode = pytest.mark.skipif(
    k3_decode_unavailable(),
    reason="needs an SM107 GPU and a CuTe DSL that builds the FP4 MLA decode kernels",
)


def k3_decode():
    """K3's FP4 MLA decode module."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (  # noqa: E501
        _fp4_mla_decode as fp4_mla_decode,
    )

    return fp4_mla_decode


def _pow2_scales(rows, cols, gen, lo, hi):
    grid = (math.ceil(rows / 128), math.ceil(cols / 128))
    return torch.exp2(torch.randint(lo, hi, grid, generator=gen).float())


@functools.lru_cache(maxsize=1)
def _weights():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_gemm import (
        k3_fp8w_lowm_gemm as lowm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_q import (
        k3_mla_q_fused as fused,
    )

    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(7)
    w_qb = (torch.randn(H * 192, QL, generator=gen) * 2).to(torch.float8_e4m3fn)
    s_qb = _pow2_scales(H * 192, QL, gen, -8, -6)
    w_qb, packed = lowm.prepare_k3_fp8w_lowm_weight(w_qb.to(dev), s_qb.to(dev))
    return SimpleNamespace(
        w_qb=w_qb,
        s_qb_f32=s_qb.to(dev),
        s_qb_packed=packed,
        qb_exp=fused.block_exponents(s_qb.to(dev), H * 192, QL),
        w_uk=(torch.randn(H, LORA, 128, generator=gen) * 0.06).to(torch.bfloat16).to(dev),
        q_norm=(1 + 0.1 * torch.randn(QL, generator=gen)).bfloat16().to(dev),
        kv_norm=(1 + 0.1 * torch.randn(LORA, generator=gen)).bfloat16().to(dev),
    )


def _rotary_table(max_pos, kind):
    dev = torch.device("cuda")
    rot = torch.zeros((max_pos, ROPE, 2), dtype=torch.float32, device=dev)
    if kind == "identity":
        rot[..., 0] = 1.0
    else:
        gen = torch.Generator(device=dev).manual_seed(5)
        angle = torch.rand((max_pos, ROPE), device=dev, generator=gen) * 6.28
        rot[..., 0] = torch.cos(angle)
        rot[..., 1] = torch.sin(angle)
    return rot.view(max_pos, ROPE * 2).contiguous()


def _fp4_case(n, kv, seed):
    """Real FP4 cache of ``n`` requests with ``kv`` tokens each (or the ``n`` lengths in ``kv``;
    the last ``Q_LEN`` of them this step's), Helix speculative per-token metadata; returns
    (manager, metadata)."""
    import tensorrt_llm
    from tensorrt_llm._torch.attention.backends.fp4_mla import (
        FP4_MLA_KV_GLOBAL_SCALE,
        FP4_MLA_Q_GLOBAL_SCALE,
        scatter_fp4_mla_kv_cache,
    )
    from tensorrt_llm._torch.attention.backends.fp4_mla.cache_manager import Fp4MlaKVCacheManagerV2
    from tensorrt_llm._torch.attention.backends.fp4_mla.state import Fp4MlaState
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig as LlmKvCacheConfig
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
    from tensorrt_llm.mapping import Mapping

    torch.manual_seed(seed)
    dev = torch.device("cuda")
    seq_lens = list(kv) if isinstance(kv, (list, tuple)) else [kv] * n
    assert len(seq_lens) == n
    num_blocks = [(s + PAGE - 1) // PAGE for s in seq_lens]
    max_seq_len = max(seq_lens) + 32 * PAGE
    max_tokens = sum(num_blocks) * PAGE
    max_tokens += max(max_tokens // 2, n * 64 * PAGE)
    ctx_lens = [s - Q_LEN for s in seq_lens]
    mgr = Fp4MlaKVCacheManagerV2(
        LlmKvCacheConfig(
            max_tokens=max_tokens, dtype="nvfp4", enable_block_reuse=False, host_cache_size=0
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=LORA + ROPE,
        tokens_per_block=PAGE,
        max_seq_len=max_seq_len,
        max_batch_size=n,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.NVFP4,
        spec_config=MTPDecodingConfig(max_draft_len=Q_LEN - 1),
        max_num_tokens=max_tokens,
        pretrained_config=SimpleNamespace(kv_lora_rank=LORA),
    )
    mgr.add_dummy_requests(list(range(n)), seq_lens)
    kv_cache, sf_cache = mgr.get_fp4_mla_cache_buffers(0)
    kv_cache.zero_()
    sf_cache.zero_()
    request_ids = list(range(n))
    block_ids = mgr.get_batch_cache_indices(request_ids, layer_idx=0)
    spec = mgr.get_fp4_mla_page_table_spec(0)
    hp_block_ids = mgr._get_batch_cache_indices_by_pool_id(
        request_ids, pool_id=spec.hp_pool_id, is_kv_aggregate=False
    )
    stride = max(num_blocks)
    rows, hp_rows = [], []
    for i in range(n):
        b = block_ids[i][: num_blocks[i]]
        rows.extend(b + [0] * (stride - len(b)))
        hb = hp_block_ids[i][: num_blocks[i]]
        hp_rows.extend(hb + [0] * (stride - len(hb)))
    idx = torch.tensor(rows, dtype=torch.int32, device=dev)
    hp_idx = torch.tensor(hp_rows, dtype=torch.int32, device=dev)
    indptr = torch.arange(n + 1, dtype=torch.int32, device=dev) * stride
    state = Fp4MlaState(
        hp_pool=mgr.get_fp4_mla_hp_pool(),
        hp_page_indices=hp_idx,
        v_scale_pool=mgr.get_mla_v_scale_pool(),
        q_global_scale=torch.tensor([FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device=dev),
        kv_global_scale=torch.tensor([FP4_MLA_KV_GLOBAL_SCALE], dtype=torch.float32, device=dev),
        generation_kv_lens=torch.empty(n, dtype=torch.int32, device=dev),
        generation_append_lens=torch.empty(n, dtype=torch.int32, device=dev),
        generation_lengths_num_tokens=-1,
        generation_lengths_num_seqs=-1,
        generation_lengths_num_contexts=-1,
        generation_lengths_capture_recorded=False,
        device_page_table=True,
        device_page_table_valid=True,
        page_table_stride=stride,
        context_repack_max_touched_pages=stride,
        _paged_kv_indices=idx,
        _paged_kv_indptr=indptr,
        paged_kv_indptr_decode=indptr.clone(),
        num_sequences=n,
        num_blocks=None,
        num_context_blocks=n * stride,
        num_generation_blocks=0,
    )
    md = SimpleNamespace(
        kv_cache_manager=mgr,
        page_size=PAGE,
        num_contexts=n,
        num_seqs=n,
        beam_width=1,
        request_ids=request_ids,
        runtime_features=SimpleNamespace(has_speculative_draft_tokens=False),
        is_cuda_graph=False,
        fp4_mla_state=state,
    )
    v_packed = mgr.get_mla_v_packed_pool_base()
    if v_packed is not None:
        v_packed.zero_()
    state.v_scale_pool.zero_()
    md.kv_lens_cuda_runtime = torch.tensor(ctx_lens, dtype=torch.int32, device=dev)
    md.prompt_lens_cuda_runtime = md.kv_lens_cuda_runtime.clone()
    md.prompt_lens_cpu_runtime = torch.tensor(ctx_lens, dtype=torch.int32)
    state.batch_indices = torch.cat(
        [torch.full((s,), i, dtype=torch.int32, device=dev) for i, s in enumerate(ctx_lens)]
    )
    state.positions = torch.cat([torch.arange(s, dtype=torch.int32, device=dev) for s in ctx_lens])
    latent = (torch.randn(sum(ctx_lens), LORA + ROPE, device=dev) * 0.5).clamp_(-3, 3).bfloat16()
    scatter_fp4_mla_kv_cache(
        md, latent, layer_idx=0, token_offset=0, phase="context", local_layer=0, v_head_dim=LORA
    )
    md.num_contexts = 0
    state.num_context_blocks = 0
    state.num_generation_blocks = n * stride
    md.kv_lens_cuda_runtime = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    md.prompt_lens_cuda_runtime = torch.full((n,), Q_LEN, dtype=torch.int32, device=dev)
    md.prompt_lens_cpu_runtime = torch.full((n,), Q_LEN, dtype=torch.int32)
    state.batch_indices = torch.arange(n, dtype=torch.int32, device=dev).repeat_interleave(Q_LEN)
    state.positions = torch.cat(
        [torch.arange(c, s, dtype=torch.int32, device=dev) for c, s in zip(ctx_lens, seq_lens)]
    )
    md.num_tokens = n * Q_LEN
    md.num_ctx_tokens = 0
    md.num_generations = n
    md.max_num_tokens = n * Q_LEN
    md.max_num_sequences = n
    md.max_total_draft_tokens = Q_LEN - 1
    md.max_seq_len = max_seq_len
    md.runtime_features.has_speculative_draft_tokens = True
    md.helix_position_offsets = state.positions.clone()
    md.helix_local_slots = state.positions.clone()
    md.helix_kv_bounds = state.positions + 1
    md.helix_is_inactive_rank = torch.zeros(n, dtype=torch.bool, device=dev)
    md._helix_spec_tokens_valid = True
    return mgr, md


def _pools(mgr):
    kv, sf = mgr.get_fp4_mla_cache_buffers(0)
    pools = [kv, sf, mgr.get_fp4_mla_hp_pool(), mgr.get_mla_v_scale_pool()]
    v_packed = mgr.get_mla_v_packed_pool_base()
    return pools + ([v_packed] if v_packed is not None else [])


def _rows_per_seq(md):
    """Query rows one request owns in the decode's Q buffers (packed or one row per head)."""
    return int(md.fp4_mla_state.prequantized_q_rows_per_seq or 0) or Q_LEN * H


def _q_bytes(md, num_tokens):
    from tensorrt_llm._torch.attention.backends.fp4_mla import _get_fp4_mla_swizzled_scale_size

    state = md.fp4_mla_state
    rows = _rows_per_seq(md) * (num_tokens // Q_LEN)
    sf_bytes = _get_fp4_mla_swizzled_scale_size(rows, 640)
    return (
        state.prequantized_q[:rows].clone(),
        state.prequantized_q_sf.view(torch.uint8)[:sf_bytes].clone(),
    )


def _scatter(md, latent, rot, q_pe, q_rope_out, q_input):
    from tensorrt_llm._torch.attention.backends.fp4_mla import scatter_fp4_mla_kv_cache

    assert scatter_fp4_mla_kv_cache(
        md,
        latent,
        0,
        token_offset=0,
        phase="generation",
        local_layer=0,
        v_head_dim=LORA,
        rotary_cos_sin=rot,
        q_pe=q_pe,
        q_rope_out=q_rope_out,
        q_quant_input=q_input,
        helix_position_offsets=md.helix_position_offsets,
        helix_is_inactive_rank=md.helix_is_inactive_rank,
    )


def _attention(rot=None):
    """The attributes of the FP4 MLA layer's ``TrtllmAttention`` that the KV-only generation
    update and K3's generation decode use."""
    return SimpleNamespace(
        num_heads=H,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        qk_nope_head_dim=128,
        q_scaling=1.0,
        uses_fp4_mla_attention=True,
        rotary_cos_sin=rot,
        layer_idx=0,
        _ensure_rope_table_size=lambda _: None,
        get_fp4_mla_local_layer_idx=lambda _: 0,
    )


def _external_scatter(md, latent, rot):
    """The KV-only generation update: cache rows and the packed-Q buffers, no query rows."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_fp4_mla import (
        fp4_mla_kv_append,
    )

    fp4_mla_kv_append.fp4_mla_kv_append_generation(_attention(rot), latent, md, H)


def _run_fused(md, w, x, rot, debug=None, norm=None):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_q import (
        k3_mla_q_fused as fused,
    )

    state = md.fp4_mla_state
    fused.k3_mla_q_fused(
        x,
        w.w_qb,
        w.qb_exp,
        w.w_uk,
        state.q_global_scale,
        rot,
        md.helix_position_offsets,
        state.prequantized_q,
        state.prequantized_q_sf.view(torch.uint8),
        gen_len=Q_LEN,
        rows_per_seq=_rows_per_seq(md),
        debug=debug,
        norm=norm,
    )


def _bf16_ulps(a, b):
    """``|a - b|`` in units of the BF16 spacing at ``b``."""
    a, b = a.float(), b.float()
    ulp = torch.exp2(torch.floor(torch.log2(b.abs().clamp_min(2.0**-126))) - 7)
    return (a - b).abs() / ulp


def _inputs(num_tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, QL, device="cuda", generator=gen).bfloat16()
    latent = (torch.randn(num_tokens, LORA + ROPE, device="cuda", generator=gen) * 0.5).bfloat16()
    return x, latent


@pytest.mark.parametrize("n", [1, 2, 4])
def test_kv_append_without_query_rows_is_bitwise(n):
    mgr, md = _fp4_case(n, 1000, seed=n)
    try:
        num_tokens = n * Q_LEN
        rot = _rotary_table(md.max_seq_len, "identity")
        _, latent = _inputs(num_tokens, 3)
        before = [p.clone() for p in _pools(mgr)]
        q_input = torch.randn(num_tokens, H, LORA + ROPE, device="cuda").bfloat16()
        q_pe = torch.randn(num_tokens, H, ROPE, device="cuda").bfloat16()
        _scatter(md, latent, rot, q_pe, q_input[..., LORA:], q_input)
        full = [p.clone() for p in _pools(mgr)]
        for p, p0 in zip(_pools(mgr), before):
            p.copy_(p0)
        _external_scatter(md, latent, rot)
        torch.cuda.synchronize()
        for p, ref in zip(_pools(mgr), full):
            assert torch.equal(p, ref)
        assert md.fp4_mla_state.prequantized_q is not None
        assert md.fp4_mla_state.prequantized_q_sf is not None
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("rope", ["identity", "random"])
@pytest.mark.parametrize("n", [1, 2])
def test_query_quantization_matches_cache_update(n, rope):
    """The fused kernel's FP4 query rows are the generation update's on the kernel's own BF16
    absorbed / q_b values."""
    w = _weights()
    mgr, md = _fp4_case(n, 1000, seed=10 + n)
    try:
        num_tokens = n * Q_LEN
        rot = _rotary_table(md.max_seq_len, rope)
        x, latent = _inputs(num_tokens, 4)
        _external_scatter(md, latent, rot)
        dbg_qb = torch.zeros(num_tokens, H * 192, device="cuda", dtype=torch.bfloat16)
        dbg_q = torch.zeros(num_tokens, H, LORA + ROPE, device="cuda", dtype=torch.bfloat16)
        _run_fused(md, w, x, rot, debug=(dbg_qb, dbg_q))
        torch.cuda.synchronize()
        q_fused, sf_fused = _q_bytes(md, num_tokens)
        md.fp4_mla_state.prequantized_q.fill_(0)
        q_input = dbg_q.clone()
        q_rope_out = torch.empty(num_tokens, H, ROPE, device="cuda", dtype=torch.bfloat16)
        q_pe = dbg_qb.view(num_tokens, H, 192)[..., 128:]
        _scatter(md, latent, rot, q_pe, q_rope_out, q_input)
        torch.cuda.synchronize()
        q_ref, sf_ref = _q_bytes(md, num_tokens)
        assert torch.equal(q_rope_out, dbg_q[..., LORA:])
        assert torch.equal(q_fused, q_ref)
        assert torch.equal(sf_fused, sf_ref)
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("num_tokens", [5, 8, 16, 24])
def test_fused_norm_matches_flashinfer_norm(num_tokens):
    """The q_a RMSNorm inside the kernel gives the query rows of the kernel run after
    FlashInfer's norm, bit for bit; so does the KV latent row kernel."""
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_q import (
        k3_mla_q_fused as fused,
    )

    w = _weights()
    n = max(1, num_tokens // Q_LEN)
    mgr, md = _fp4_case(n, 600, seed=40 + num_tokens)
    try:
        rot = _rotary_table(md.max_seq_len, "identity")
        gen = torch.Generator(device="cuda").manual_seed(8)
        qkv = torch.randn(num_tokens, QL + LORA + ROPE, device="cuda", generator=gen).bfloat16()
        q, ckv, kpe = qkv.split([QL, LORA, ROPE], -1)
        latent = fused.k3_mla_latent_row(ckv, kpe, w.kv_norm, EPS)
        ref_latent = torch.cat([rmsnorm(ckv, w.kv_norm, EPS), kpe], dim=-1)
        torch.cuda.synchronize()
        assert torch.equal(latent, ref_latent)
        if num_tokens % Q_LEN or num_tokens > fused.MAX_TOKENS:
            return
        _external_scatter(md, latent, rot)
        _run_fused(md, w, rmsnorm(q, w.q_norm, EPS), rot)
        torch.cuda.synchronize()
        ref = _q_bytes(md, num_tokens)
        md.fp4_mla_state.prequantized_q.fill_(0)
        _run_fused(md, w, q, rot, norm=(w.q_norm, EPS))
        torch.cuda.synchronize()
        got = _q_bytes(md, num_tokens)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("num_tokens", [8, 16])
def test_projections_match_production_gemms(num_tokens):
    """q_b against the low-M FP8 GEMM, W_UK against the BF16 BMM: the same products, another FP32
    summation order, so BF16 results agree to rounding."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_gemm import (
        k3_fp8w_lowm_gemm as lowm,
    )

    w = _weights()
    n = num_tokens // Q_LEN
    mgr, md = _fp4_case(n, 600, seed=20 + n)
    try:
        rot = _rotary_table(md.max_seq_len, "identity")
        x, latent = _inputs(num_tokens, 5)
        _external_scatter(md, latent, rot)
        dbg_qb = torch.zeros(num_tokens, H * 192, device="cuda", dtype=torch.bfloat16)
        dbg_q = torch.zeros(num_tokens, H, LORA + ROPE, device="cuda", dtype=torch.bfloat16)
        _run_fused(md, w, x, rot, debug=(dbg_qb, dbg_q))
        ref_qb = torch.empty_like(dbg_qb)
        lowm.run_k3_fp8w_lowm_gemm(x, w.w_qb, w.s_qb_packed, ref_qb)
        q = dbg_qb.view(num_tokens, H, 192)
        ref_uk = torch.bmm(q[..., :128].transpose(0, 1), w.w_uk.transpose(1, 2)).transpose(0, 1)
        torch.cuda.synchronize()
        for got, ref in ((dbg_qb, ref_qb), (dbg_q[..., :LORA], ref_uk)):
            ulps = _bf16_ulps(got, ref)
            assert float(ulps.max()) <= 2.0
            assert float((got != ref).float().mean()) < 0.02
    finally:
        mgr.shutdown()


class _DecodeAsAttention:
    """``generation.attn_forward_gen`` stand-in: K3's FP4 generation decode of every head (no
    Helix exchange), as ``generation.gen_attention`` runs it with the helix8 settings."""

    def __init__(self, md):
        self.md = md

    def __call__(self, layer, fused_q, md, *, helix_v_up=False, hooks=None, **kw):
        assert helix_v_up and kw["latent_cache"] is not None
        assert md.fp4_mla_state.generation_cache_scattered
        stats = torch.empty(fused_q.shape[0], H, 2, device="cuda", dtype=torch.float32)
        out = k3_decode().run_generation(
            _attention(),
            fused_q,
            md,
            softmax_stats_tensor=stats,
            page_plan=True,
            reduce_pdl=True,
        )
        assert not md.fp4_mla_state.generation_cache_scattered
        return out


def _mla_stand_in(w, md, rot, qkv):
    """The attributes of the MLA layer that ``generation.forward_gen_q_fused`` reads, with the
    real kernels behind them."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_q.q_norm_pdl import (  # noqa: E501
        rmsnorm_release_at_launch,
    )

    return SimpleNamespace(
        num_heads_tp=H,
        num_heads_tp_cp=H,
        v_head_dim=LORA,
        q_lora_rank=QL,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        mqa=_attention(rot),
        aux_stream=torch.cuda.Stream(),
        _kv_events=(torch.cuda.Event(), torch.cuda.Event()),
        kv_a_layernorm=SimpleNamespace(weight=w.kv_norm, variance_epsilon=EPS),
        q_a_layernorm=SimpleNamespace(weight=w.q_norm, variance_epsilon=EPS),
        q_b_proj=SimpleNamespace(weight=w.w_qb, weight_scale=w.s_qb_f32),
        k_b_proj_trans=w.w_uk,
        k_b_proj_trans_dequant=None,
        _q_fused_qb_exp=None,
        project_qkv_a=lambda _: qkv,
        q_a_norm=lambda q: rmsnorm_release_at_launch(q, w.q_norm, EPS),
    )


@needs_k3_decode
@pytest.mark.parametrize("n", [1, 2])
def test_forward_gen_q_fused_matches_default_chain(n, monkeypatch):
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_gemm import (
        k3_fp8w_lowm_gemm as lowm,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_q.q_norm_pdl import (  # noqa: E501
        rmsnorm_release_at_launch,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_nvfp4__sm_107__helix8_moeep8 import (
        modeling as generation,
    )
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    w = _weights()
    mgr, md = _fp4_case(n, 2000, seed=30 + n)
    try:
        num_tokens = n * Q_LEN
        rot = _rotary_table(md.max_seq_len, "identity")
        gen = torch.Generator(device="cuda").manual_seed(6)
        qkv = torch.randn(num_tokens, QL + LORA + ROPE, device="cuda", generator=gen).bfloat16()
        before = [p.clone() for p in _pools(mgr)]

        # Default chain: q RMSNorm, q_b GEMM, W_UK BMM, generation update with the query rows.
        q, ckv, kpe = qkv.split([QL, LORA, ROPE], -1)
        latent = torch.cat([rmsnorm(ckv, w.kv_norm, EPS), kpe], dim=-1)
        qb = torch.empty(num_tokens, H * 192, device="cuda", dtype=torch.bfloat16)
        lowm.run_k3_fp8w_lowm_gemm(rmsnorm_release_at_launch(q, w.q_norm, EPS), w.w_qb,
                                   w.s_qb_packed, qb)  # fmt: skip
        fused_q = torch.empty(num_tokens, H, LORA + ROPE, device="cuda", dtype=torch.bfloat16)
        qv = qb.view(num_tokens, H, 192)
        torch.bmm(qv[..., :128].transpose(0, 1), w.w_uk.transpose(1, 2),
                  out=fused_q[..., :LORA].transpose(0, 1))  # fmt: skip
        _scatter(md, latent, rot, qv[..., 128:], fused_q[..., LORA:], fused_q)
        assert md.fp4_mla_state.generation_cache_scattered
        ref_out = _DecodeAsAttention(md)(None, fused_q.view(num_tokens, -1), md, helix_v_up=True,
                                         latent_cache=latent)  # fmt: skip
        ref_pools = [p.clone() for p in _pools(mgr)]
        for p, p0 in zip(_pools(mgr), before):
            p.copy_(p0)

        mla = _mla_stand_in(w, md, rot, qkv)
        decode = _DecodeAsAttention(md)
        monkeypatch.setattr(generation, "attn_forward_gen", decode)
        k3_decode().begin_step(md)
        with with_multi_stream(True):
            out = generation.forward_gen_q_fused(mla, qkv, md)
        torch.cuda.synchronize()
        for p, ref in zip(_pools(mgr), ref_pools):
            assert torch.equal(p, ref)
        rel = float((out.float() - ref_out.float()).norm() / ref_out.float().norm())
        assert rel < 0.05, rel
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------------------------------------
# The k3_mla_q_fused entry on its cells, against its plain-torch reference
# ---------------------------------------------------------------------------------------------


def _q_fused_entry():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention import (  # noqa: E501
        k3_mla_q_fused as entry_module,
    )

    return entry_module


def _dequant_rows(entry_module, q_fp4, q_sf, rows, gs):
    """FP32 ``[rows, 576]`` values of FP4 query rows (RoPE: main + residual) and their FP32
    ``[rows, 40]`` group scales, in units of the unscaled query."""
    cols = 40
    col = torch.arange(cols, device=q_fp4.device)
    offsets = entry_module.swizzled_sf_offset(rows[:, None], col[None, :], cols)
    sf = q_sf.view(torch.uint8).view(-1)[offsets.flatten()].view(-1, cols)
    sf = sf.view(torch.float8_e4m3fn).float()
    codes = entry_module.unpack_e2m1(q_fp4[rows])
    values = entry_module.e2m1_values(codes).view(-1, cols, 16) * sf.unsqueeze(-1)
    prefix = values[:, :32].flatten(1)
    rope = values[:, 32:].view(-1, 4, 2, 16)
    return torch.cat((prefix, (rope[:, :, 0] + rope[:, :, 1]).flatten(1)), -1) / gs, sf / gs


_Q_FUSED_CELLS = _q_fused_entry().k3_mla_q_fused.CELLS


@pytest.mark.parametrize("cell", _Q_FUSED_CELLS, ids=[c.why[:48] for c in _Q_FUSED_CELLS])
def test_k3_mla_q_fused_cells(cell):
    """The entry's ``.CELLS`` under ``validating``: every FP4 group of the written query rows
    within one FP4 step (two units of its scale) of the reference's, codes equal in nearly every
    group (the q_b MMA's summation order moves a BF16 value by an ulp now and then), and every
    byte outside the tokens' rows untouched."""
    import zlib

    from _validating import validating

    from tensorrt_llm._torch.attention.backends.fp4_mla import FP4_MLA_Q_GLOBAL_SCALE
    from tensorrt_llm._torch.attention.backends.fp4_mla.config import (
        FP4_MLA_Q_LOGICAL_DIM,
        _fp4_mla_packed_rows_per_seq,
    )
    from tensorrt_llm._torch.attention.backends.fp4_mla.layout import (
        _get_fp4_mla_swizzled_scale_size,
    )

    entry_module = _q_fused_entry()
    op = entry_module.k3_mla_q_fused
    spec = cell.spec
    w = _weights()
    requests, gen_len = spec["requests"], spec["gen_len"]
    tokens = requests * gen_len
    seed = zlib.crc32(cell.why.encode()) % 2**31
    x, _ = _inputs(tokens, seed)
    rows_per_seq = _fp4_mla_packed_rows_per_seq(H, gen_len)
    num_rows = requests * rows_per_seq
    q_fp4 = torch.zeros((num_rows, 320), dtype=torch.uint8, device="cuda")
    q_sf = torch.zeros(
        _get_fp4_mla_swizzled_scale_size(num_rows, FP4_MLA_Q_LOGICAL_DIM),
        dtype=torch.uint8,
        device="cuda",
    )
    gs = torch.tensor([FP4_MLA_Q_GLOBAL_SCALE], dtype=torch.float32, device="cuda")
    rot = _rotary_table(4096, "identity")
    positions = (1000 + torch.arange(tokens, device="cuda")).to(torch.int32)
    norm = (w.q_norm, EPS) if spec["norm"] else None
    if norm is None:
        x = _flashinfer_norm(x, w.q_norm)
    args = (x, w.w_qb, w.qb_exp, w.w_uk, gs, rot, positions, q_fp4, q_sf, gen_len, rows_per_seq,
            True, norm)  # fmt: skip
    ref_q, ref_sf = op.reference(*args)
    with validating(op):
        op(*args)
    torch.cuda.synchronize()

    t = torch.arange(tokens, device="cuda")
    token_rows = t // gen_len * rows_per_seq + t % gen_len * H
    rows = (token_rows[:, None] + torch.arange(H, device="cuda")[None, :]).flatten()
    written = torch.zeros(num_rows, dtype=torch.bool, device="cuda")
    written[rows] = True
    assert torch.equal(q_fp4[~written], ref_q[~written]), "bytes outside the tokens' rows moved"
    got, got_sf = _dequant_rows(entry_module, q_fp4, q_sf, rows, float(gs))
    ref, ref_sf = _dequant_rows(entry_module, ref_q, ref_sf, rows, float(gs))
    step = 2 * torch.maximum(got_sf, ref_sf)
    step = torch.cat((step[:, :32].repeat_interleave(16, -1),
                      step[:, 32::2].repeat_interleave(16, -1)), -1)  # fmt: skip
    assert bool(((got - ref).abs() <= step).all()), cell.why
    groups_equal = torch.eq(q_fp4[rows].view(-1, 40, 8), ref_q[rows].view(-1, 40, 8)).all(-1)
    assert float(groups_equal.float().mean()) >= 0.95, cell.why


def _flashinfer_norm(x, weight):
    from flashinfer.norm import rmsnorm

    out = torch.empty_like(x)
    rmsnorm(x, weight, EPS, out=out)
    return out
