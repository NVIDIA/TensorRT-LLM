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
"""Kimi-K3 fused FP4 MLA decode query (``k3_mla_q_fused``) against the default query chain.

The FP4 cache, HP ring and Helix per-token metadata are built through the real
``Fp4MlaKVCacheManagerV2`` (context scatter, then one generation step of ``q_len`` tokens per
request). The default chain is the production one: the q_b low-M GEMM, the BF16 W_UK BMM and
the fused FP4 generation update with its query rows. Checks:

* The KV append without query rows writes the cache bytes of the full update (bitwise).
* The fused kernel's query quantization equals the Triton kernel's on the same BF16 values
  (bitwise), including the RoPE rotation with a non-identity table.
* Its q_b and W_UK products agree with the production GEMMs up to BF16 rounding of a different
  FP32 summation order.
  (Both read the kernel's BF16 intermediates through ``_ProbedQFusedKernel``.)
* With the q_a RMSNorm inside the kernel the query rows are those after FlashInfer's separate
  norm (bitwise), and ``k3_mla_latent_row`` is bitwise FlashInfer's norm plus the ``k_pe`` copy.
* ``KimiK3MLAAttention._forward_generation_fused_query`` (multi-stream, KV half on the aux
  stream) gives the default chain's cache bytes and, through the FP4 decode, its attention output
  within the FP4 query quantization.
"""

import functools
import math
import types
from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass import Int32, Int64

from tensorrt_llm._torch.kernels.blackwell import k3_mla_q_fused as fused_mod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

H, QL, LORA, ROPE, PAGE = 96, 1536, 512, 64, 128
Q_LEN = 8
EPS = 1e-6


def _pow2_scales(rows, cols, gen, lo, hi):
    grid = (math.ceil(rows / 128), math.ceil(cols / 128))
    return torch.exp2(torch.randint(lo, hi, grid, generator=gen).float())


@functools.lru_cache(maxsize=1)
def _weights():
    from tensorrt_llm._torch.kernels.blackwell import k3_mla_q_fused as fused
    from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout

    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(7)
    w_qb = (torch.randn(H * 192, QL, generator=gen) * 2).to(torch.float8_e4m3fn)
    s_qb = _pow2_scales(H * 192, QL, gen, -8, -6)
    # Power-of-two scales: the packed UE8M0 scale is exact (no resmoothing needed).
    w_qb = w_qb.to(dev)
    packed = transform_sf_into_required_layout(
        s_qb.to(dev), mn=H * 192, k=QL, recipe=(1, 128, 128), is_sfa=False
    )
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
    """Real FP4 cache of ``n`` requests with ``kv`` tokens each (the last ``Q_LEN`` of them this
    step's), Helix speculative per-token metadata; returns (manager, metadata)."""
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
    seq_lens = [kv] * n
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
        device_page_table=True,
        device_page_table_valid=True,
        page_table_stride=stride,
        context_repack_max_touched_pages=stride,
        _paged_kv_indices=idx,
        _paged_kv_indptr=indptr,
        paged_kv_indptr_decode=indptr.clone(),
        num_sequences=n,
        num_context_blocks=n * stride,
        num_generation_blocks=0,
    )
    md = SimpleNamespace(
        kv_cache_manager=mgr,
        fp4_mla_state=state,
        page_size=PAGE,
        num_contexts=n,
        num_seqs=n,
        request_ids=request_ids,
        runtime_features=SimpleNamespace(has_speculative_draft_tokens=False),
        is_cuda_graph=False,
        is_warmup=False,
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


def _q_bytes(md, num_tokens):
    from tensorrt_llm._torch.attention.backends.fp4_mla import _get_fp4_mla_swizzled_scale_size

    rows = (int(md.fp4_mla_state.prequantized_q_rows_per_seq or 0) or Q_LEN * H) * (
        num_tokens // Q_LEN
    )
    sf_bytes = _get_fp4_mla_swizzled_scale_size(rows, 640)
    return (
        md.fp4_mla_state.prequantized_q[:rows].clone(),
        md.fp4_mla_state.prequantized_q_sf.view(torch.uint8)[:sf_bytes].clone(),
    )


def _scatter(md, latent, rot, q_pe, q_rope_out, q_input, external):
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
        q_rows_external=external,
    )


def _external_scatter(md, latent, rot):
    stub = latent.new_empty(1)
    num_tokens = latent.shape[0]
    q_pe = stub.expand(num_tokens, H, ROPE)
    _scatter(md, latent, rot, q_pe, q_pe, stub.expand(num_tokens, H, LORA + ROPE), True)


def _run_fused(md, w, x, rot, norm=None):
    fused_mod.k3_mla_q_fused(
        x,
        w.w_qb,
        w.qb_exp,
        w.w_uk,
        md.fp4_mla_state.q_global_scale,
        rot,
        md.helix_position_offsets,
        md.fp4_mla_state.prequantized_q,
        md.fp4_mla_state.prequantized_q_sf.view(torch.uint8),
        gen_len=Q_LEN,
        rows_per_seq=int(md.fp4_mla_state.prequantized_q_rows_per_seq or 0) or Q_LEN * H,
        norm=norm,
    )


class _ProbedQFusedKernel(fused_mod.K3MlaQFusedKernel):
    """The fused kernel, also storing its BF16 intermediates at the device addresses ``qb_addr``
    (q_b output ``[t_pad, 96 * 192]``, q_pe before RoPE) and ``ob_addr`` (absorbed query
    ``[t_pad, 96, 512]``), set on a subclass per launch."""

    qb_addr = 0
    ob_addr = 0

    @cute.jit
    def absorb(self, warp: Int32, lane: Int32, wuk_s: Int32, qn_s: Int32, ob_s: Int32):
        fused_mod.K3MlaQFusedKernel.absorb(self, warp, lane, wuk_s, qn_s, ob_s)
        # Both CTAs of the head hold all of q_nope; both store it.
        bidx, _, _ = cute.arch.block_idx()
        head = bidx // Int32(2)
        vectors = self.t_pad * fused_mod.NOPE_DIM * 2 // 16
        workers = fused_mod._ABSORB_WARPS * 32
        for i in cutlass.range_constexpr(-(-vectors // workers)):
            item = warp * Int32(32) + lane + Int32(i * workers)
            if item < Int32(vectors):
                t = item // Int32(16)
                v = item - t * Int32(16)
                words = fused_mod._lds_v4_b32(qn_s + t * Int32(fused_mod._QN_PITCH) + v * Int32(16))
                dst = Int64(self.qb_addr) + (Int64(t) * (H * 192) + Int64(head) * 192) * 2
                fused_mod._stg_v4(dst + Int64(v) * 16, *words)

    @cute.jit
    def store_prefix_group(self, t, grp, rank, head, gs, ob_s, qfp4_addr, qsf_addr, gen_len,
                           rows_per_seq):  # fmt: skip
        fused_mod.K3MlaQFusedKernel.store_prefix_group(
            self, t, grp, rank, head, gs, ob_s, qfp4_addr, qsf_addr, gen_len, rows_per_seq
        )
        src = ob_s + t * Int32(fused_mod._OB_PITCH) + grp * Int32(32)
        col = rank * Int32(16) + grp
        dst = Int64(self.ob_addr) + ((Int64(t) * H + Int64(head)) * LORA + Int64(col) * 16) * 2
        fused_mod._stg_v4(dst, *fused_mod._lds_v4_b32(src))
        fused_mod._stg_v4(dst + Int64(16), *fused_mod._lds_v4_b32(src + Int32(16)))

    @cute.jit
    def store_rope_group(self, t, rg, head, gs, qpe_s, cs_s, qfp4_addr, qsf_addr, gen_len,
                         rows_per_seq):  # fmt: skip
        fused_mod.K3MlaQFusedKernel.store_rope_group(
            self, t, rg, head, gs, qpe_s, cs_s, qfp4_addr, qsf_addr, gen_len, rows_per_seq
        )
        src = qpe_s + t * Int32(fused_mod._QPE_PITCH) + rg * Int32(32)
        dst = Int64(self.qb_addr) + (
            (Int64(t) * (H * 192) + Int64(head) * 192 + 128 + Int64(rg) * 16) * 2
        )
        fused_mod._stg_v4(dst, *fused_mod._lds_v4_b32(src))
        fused_mod._stg_v4(dst + Int64(16), *fused_mod._lds_v4_b32(src + Int32(16)))


def _run_probed(md, w, x, rot):
    """:func:`_run_fused` with :class:`_ProbedQFusedKernel`: returns its BF16 q_b output
    ``[T, 96 * 192]`` (q_pe before RoPE) and absorbed query ``[T, 96, 512]``."""
    num_tokens = x.shape[0]
    t_pad = 8 if num_tokens <= 8 else 16
    qb = torch.zeros(t_pad, H * 192, device="cuda", dtype=torch.bfloat16)
    ob = torch.zeros(t_pad, H, LORA, device="cuda", dtype=torch.bfloat16)
    probe = type(
        "_Probe", (_ProbedQFusedKernel,), {"qb_addr": qb.data_ptr(), "ob_addr": ob.data_ptr()}
    )
    production = fused_mod.K3MlaQFusedKernel
    fused_mod._compiled.cache_clear()
    fused_mod.K3MlaQFusedKernel = probe
    try:
        _run_fused(md, w, x, rot)
        torch.cuda.synchronize()
    finally:
        fused_mod.K3MlaQFusedKernel = production
        fused_mod._compiled.cache_clear()
    return qb[:num_tokens], ob[:num_tokens]


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
        _scatter(md, latent, rot, q_pe, q_input[..., LORA:], q_input, False)
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
        qb, absorbed = _run_probed(md, w, x, rot)
        q_fused, sf_fused = _q_bytes(md, num_tokens)
        md.fp4_mla_state.prequantized_q.fill_(0)
        q_input = torch.zeros(num_tokens, H, LORA + ROPE, device="cuda", dtype=torch.bfloat16)
        q_input[..., :LORA] = absorbed
        q_rope_out = torch.empty(num_tokens, H, ROPE, device="cuda", dtype=torch.bfloat16)
        q_pe = qb.view(num_tokens, H, 192)[..., 128:]
        _scatter(md, latent, rot, q_pe, q_rope_out, q_input, False)
        torch.cuda.synchronize()
        q_ref, sf_ref = _q_bytes(md, num_tokens)
        assert torch.equal(q_fused, q_ref)
        assert torch.equal(sf_fused, sf_ref)
    finally:
        mgr.shutdown()


@pytest.mark.parametrize("num_tokens", [5, 8, 16, 24])
def test_fused_norm_matches_flashinfer_norm(num_tokens):
    """The q_a RMSNorm inside the kernel gives the query rows of the kernel run after
    FlashInfer's norm, bit for bit; so does the KV latent row kernel."""
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch.kernels.blackwell import k3_mla_q_fused as fused

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
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as lowm

    w = _weights()
    n = num_tokens // Q_LEN
    mgr, md = _fp4_case(n, 600, seed=20 + n)
    try:
        rot = _rotary_table(md.max_seq_len, "identity")
        x, latent = _inputs(num_tokens, 5)
        _external_scatter(md, latent, rot)
        qb, absorbed = _run_probed(md, w, x, rot)
        ref_qb = torch.empty_like(qb)
        lowm.run_low_m_gemm(x, w.w_qb, w.s_qb_packed, ref_qb)
        q = qb.view(num_tokens, H, 192)
        ref_uk = torch.bmm(q[..., :128].transpose(0, 1), w.w_uk.transpose(1, 2)).transpose(0, 1)
        torch.cuda.synchronize()
        for got, ref in ((qb, ref_qb), (absorbed, ref_uk)):
            ulps = _bf16_ulps(got, ref)
            assert float(ulps.max()) <= 2.0
            assert float((got != ref).float().mean()) < 0.02
    finally:
        mgr.shutdown()


class _DecodeAsAttention:
    """``_attn_forward_gen`` stand-in: the FP4 decode of every head (no Helix exchange)."""

    def __init__(self, md):
        self.md = md
        self.output = None

    def __call__(self, mqa, fused_q, k, v, position_ids, md, helix_v_up=False, **kwargs):
        from tensorrt_llm._torch.attention.backends import fp4_mla

        assert helix_v_up and kwargs["latent_cache"] is not None
        num_tokens = fused_q.shape[0]
        assert md.fp4_mla_state.generation_cache_scattered
        out = torch.empty(num_tokens, H, LORA, device="cuda", dtype=torch.bfloat16)
        fp4_mla.run_fp4_mla_attention_decode(
            md,
            layer_idx=0,
            local_layer=0,
            q=fused_q.view(num_tokens, H, LORA + ROPE),
            output=out,
            sm_scale=1.0 / math.sqrt(192.0),
            kv_lora_rank=LORA,
            qk_rope_head_dim=ROPE,
            prequantized_q=md.fp4_mla_state.prequantized_q,
            prequantized_q_sf=md.fp4_mla_state.prequantized_q_sf,
            q_batch_capacity=md.fp4_mla_state.q_batch_capacity,
        )
        return out.view(num_tokens, H * LORA)


def _mla_stand_in(w, md, rot, qkv):
    """The attributes and methods of ``KimiK3MLAAttention`` that
    ``_forward_generation_fused_query`` uses, with the real kernels behind them."""
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    mqa = SimpleNamespace(
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        uses_fp4_mla_attention=True,
        rotary_cos_sin=rot,
        layer_idx=0,
        _ensure_rope_table_size=lambda _: None,
        get_fp4_mla_local_layer_idx=lambda _: 0,
    )
    mqa.fp4_mla_kv_append_generation = types.MethodType(
        TrtllmAttention.fp4_mla_kv_append_generation, mqa
    )
    kv_norm = RMSNorm(hidden_size=LORA, eps=EPS, dtype=torch.bfloat16).cuda()
    kv_norm.weight.data.copy_(w.kv_norm)
    q_norm = RMSNorm(hidden_size=QL, eps=EPS, dtype=torch.bfloat16).cuda()
    q_norm.weight.data.copy_(w.q_norm)
    cute_norm = k3.KimiK3MLAAttention._runs_flashinfer_cute_rmsnorm
    mla = SimpleNamespace(
        num_heads_tp=H,
        q_lora_rank=QL,
        kv_lora_rank=LORA,
        qk_rope_head_dim=ROPE,
        out_scale=None,
        mqa=mqa,
        aux_stream=torch.cuda.Stream(),
        ln_events=[torch.cuda.Event(), torch.cuda.Event()],
        kv_a_layernorm=kv_norm,
        q_a_layernorm=q_norm,
        _fused_latent_row=cute_norm(kv_norm),
        _fused_decode_query_norm=cute_norm(q_norm),
        q_b_proj=SimpleNamespace(weight=w.w_qb, weight_scale=w.s_qb_f32),
        k_b_proj_trans=w.w_uk,
        k_b_proj_trans_dequant=None,
        _q_b_block_exponents=None,
        _helix_tail_gate_behind_kv=False,
        _project_qkv_a=lambda hidden_states, prequantized_input: qkv,
        _q_a_layernorm_maybe_fused=lambda q: rmsnorm(q, w.q_norm, EPS),
        _attn_forward_gen=_DecodeAsAttention(md),
    )
    for name in ("_q_b_exponents", "_absorption_weight_bf16"):
        setattr(mla, name, types.MethodType(getattr(k3.KimiK3MLAAttention, name), mla))
    return mla


@pytest.mark.parametrize("n", [1, 2])
def test_forward_generation_fused_query_matches_default_chain(n):
    from flashinfer.norm import rmsnorm

    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm as lowm
    from tensorrt_llm._torch.modules.kimi_k3_mla import kimi_k3_mla_attention as k3
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
        lowm.run_low_m_gemm(rmsnorm(q, w.q_norm, EPS), w.w_qb, w.s_qb_packed, qb)
        fused_q = torch.empty(num_tokens, H, LORA + ROPE, device="cuda", dtype=torch.bfloat16)
        qv = qb.view(num_tokens, H, 192)
        torch.bmm(qv[..., :128].transpose(0, 1), w.w_uk.transpose(1, 2),
                  out=fused_q[..., :LORA].transpose(0, 1))  # fmt: skip
        _scatter(md, latent, rot, qv[..., 128:], fused_q[..., LORA:], fused_q, False)
        md.fp4_mla_state.generation_cache_scattered = True
        ref_out = _DecodeAsAttention(md)(None, fused_q.view(num_tokens, -1), None, None, None,
                                         md, helix_v_up=True, latent_cache=latent)  # fmt: skip
        ref_pools = [p.clone() for p in _pools(mgr)]
        for p, p0 in zip(_pools(mgr), before):
            p.copy_(p0)

        mla = _mla_stand_in(w, md, rot, qkv)
        out = torch.empty(num_tokens, H * LORA, device="cuda", dtype=torch.bfloat16)
        with with_multi_stream(True):
            k3.KimiK3MLAAttention._forward_generation_fused_query(mla, qkv, md, out, None)
        torch.cuda.synchronize()
        for p, ref in zip(_pools(mgr), ref_pools):
            assert torch.equal(p, ref)
        rel = float((out.float() - ref_out.float()).norm() / ref_out.float().norm())
        assert rel < 0.05, rel
    finally:
        mgr.shutdown()
