# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3_8_flash_next_fp8__sm_103__unified target's thop_attention variant
(ops/attention/thop_attention__rank_heads).

A copy of the catalog entry's test (`attention/test_modeling_v2_thop_attention.py`) cut to
the paths this target calls: standard packed-QKV causal GQA at head_size 256 over a bf16,
fp8-e4m3 or NVFP4 paged pool on the paged-context path, at the per-rank head counts of its
members -- 24 q / 2 kv (tp1, dep2, dep4), 12 / 1 (tp2, tep2) and 6 / 1 (tp4, tep4) -- plus
the entry's own cells. The environment is built for real: a caller-owned paged pool or a
real KVCacheManagerV2 (one pool shared by several layers, or the engine's two-pool hybrid
48-row mapping), the int32 host/device length tensors, and the pool read back bitwise after
every append. The MLA, attention-sink and sliding-window sections of the catalog test belong
to targets that call the catalog entry, not this variant.

The cells run every head geometry at the target's batch and history envelope: up to 128
requests per rank, histories up to 2047 tokens (the dense path ends at the 2048-token QSA
cutoff), MTP verify blocks of P = 2, 4 and 8 rows, and an 8192-row prefill (four 2048-token
contexts, max_num_tokens). The non-cell tests pin what a cell cannot: the within-block mask
of a verify step read out one-hot at every head geometry that drafts, with the full-block
rival measured outside the band; the cached prefix reaching the FMHA; the sparse /
skip-softmax group inert; and local_layer_idx over the 48-row mapping.
"""

import math
import zlib
from types import SimpleNamespace
from typing import Dict, List, NamedTuple, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.thop_attention__rank_heads import (  # noqa: E501
    QM_FP8_KV_CACHE,
    QM_NVFP4_KV_CACHE,
    QWEN4_DENSE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.thop_attention__rank_heads import (  # noqa: E501
    thop_attention as op,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import (  # noqa: E402,F401 — needs the path declared above
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "thop_attention requires a CUDA device"

# The output is a softmax-weighted combination of unit-scale bf16 v rows;
# one bf16 ulp at magnitude 1 is 2^-8 ~= 3.9e-3. The kernel's online-softmax
# fp32 accumulation order differs from the reference and K/V round-trip
# through the bf16 KV cache. Observed on sm_100 across all cases: max abs
# err 1.6e-2, on magnitude-~1 elements (within rtol); at most 64% of the
# combined atol + rtol*|ref| allowance is used (the maximum is the page-32
# cached-KV no-append MLA context case at 63.9%, then the fp8-pool
# multi-layer context at 62% and the MLA chunked-prefill final causal pass
# at 61%; the other MLA cases, at 8, 16 and 32 query heads and page sizes 32
# and 64, sit at 36-53%).
ATOL = 5e-3
RTOL = 1.6e-2  # torch default for bf16

MASK_CAUSAL = 1  # AttentionMaskType.causal
MASK_PADDING = 0  # AttentionMaskType.padding: no mask (bidirectional)

QUANT_MODE_FP8_KV_CACHE = 128  # QuantMode.FP8_KV_CACHE (sole bit set)
QUANT_MODE_NVFP4_KV_CACHE = 8192  # QuantMode.NVFP4_KV_CACHE (sole bit set)

# fp8-e4m3 KV-pool decode tolerance. The reference dequantizes the mirrored
# e4m3 pool content and puts q through the same e4m3 round-trip the kernel
# applies (quantized with kv_scale_orig_quant — pinned by a non-power-of-2
# scale sweep: only that scale collapses the error). The residual is the
# kernel's internal e4m3 handling of the softmax probabilities before BMM2
# (decode kernel variant QkvE4m3...: every MMA operand is e4m3) plus fp8-MMA
# accumulation order. e4m3's max relative rounding error is 2^-4; observed
# max abs err vs this reference on sm_100: 3.6e-2 across 10 seeds x 2 decode
# steps (magnitude-~1 outputs), at most 53% of the 2^-4 * (1 + |ref|)
# allowance, while a reference that skips the kv dequant (the wrong-scale
# failure mode) sits at ~4.4e-1 — several times past the gate.
FP8_DECODE_ATOL = 2**-4
FP8_DECODE_RTOL = 2**-4

# NVFP4: one e4m3 block scale per this many values; e2m1 magnitudes by code
# and the round-to-nearest-even steps between them (inclusive where the upper
# neighbour is the even code).
NVFP4_BLOCK = 16
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_STEPS = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False))
E2M1_STEPS += ((3.5, True), (5.0, False))


def _nvfp4_quantize(rows: torch.Tensor, global_scale: torch.Tensor):
    """[n, D] rows -> (packed e2m1 bytes [n, D/2], e4m3 scale bytes [n, D/16]):
    the attention op's append, in torch (`cvt_warp_fp16_to_fp4`)."""
    n, d = rows.shape
    x = rows.float().reshape(n, d // NVFP4_BLOCK, NVFP4_BLOCK)
    block_max = x.abs().amax(-1)
    sixth = torch.tensor(6.0, dtype=torch.float32, device=rows.device).reciprocal()
    code = (global_scale * (block_max * sixth)).to(torch.float8_e4m3fn)
    out_scale = torch.where(
        block_max != 0,
        (code.float() * global_scale.reciprocal()).reciprocal(),
        torch.zeros_like(block_max),
    )
    scaled = x * out_scale[..., None]
    magnitude = scaled.abs()
    codes = torch.zeros(scaled.shape, dtype=torch.int32, device=rows.device)
    for threshold, inclusive in E2M1_STEPS:
        codes += (magnitude >= threshold if inclusive else magnitude > threshold).to(torch.int32)
    codes |= torch.signbit(scaled).to(torch.int32) << 3
    # 0 x inf, in a block whose scale underflows to zero: the conversion makes +6.
    codes = torch.where(torch.isnan(scaled), torch.full_like(codes, 0x7), codes)
    codes = codes.reshape(n, d // 2, 2)
    return (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8), code.view(torch.uint8)


def _nvfp4_dequantize(packed: torch.Tensor, scales: torch.Tensor, global_scale) -> torch.Tensor:
    """Inverse of `_nvfp4_quantize`: fp32 [n, D]."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    lut = torch.cat([lut, -lut])
    raw = packed.long()
    values = torch.stack([lut[raw & 0xF], lut[raw >> 4]], -1).flatten(-2)
    scale = scales.view(torch.float8_e4m3fn).float().repeat_interleave(NVFP4_BLOCK, -1)
    return values * scale / global_scale


def _nvfp4_roundtrip(kv: torch.Tensor, global_scale) -> torch.Tensor:
    """[T, Hkv, D] bf16 -> the bf16 values an NVFP4 pool returns for them."""
    t, h, d = kv.shape
    packed, scales = _nvfp4_quantize(kv.reshape(t * h, d), global_scale)
    return _nvfp4_dequantize(packed, scales, global_scale).reshape(t, h, d).to(kv.dtype)


def _check_nvfp4_cache(env, request_id: int) -> None:
    """An NVFP4 pool holds each token's e2m1 bytes and its block scales -- K's
    token-major, V's in four-token groups -- bitwise as the append writes them."""
    tpb, hkv, d = env.tokens_per_block, env.num_kv_heads, env.head_dim
    groups = d // NVFP4_BLOCK
    for role, history in ((0, env.k_history[request_id]), (1, env.v_history[request_id])):
        seq = torch.cat(history)
        total = seq.shape[0]
        packed, scales = _nvfp4_quantize(
            seq.reshape(total * hkv, d), env.kv_scale_orig_quant[1 + role]
        )
        packed = packed.reshape(total, hkv, d // 2)
        scales = scales.reshape(total, hkv, groups)
        for j, page in enumerate(env.pages[request_id]):
            n = min(tpb, total - j * tpb)
            if n <= 0:
                break
            rows = slice(j * tpb, j * tpb + n)
            got = env.kv_cache[page, role, :, :n].permute(1, 0, 2)
            assert torch.equal(got, packed[rows]), f"request {request_id} role {role} page {j}"
            sf_page = env.kv_block_scales[page, role]
            if role == 0:
                got_sf = sf_page[:, :n].permute(1, 0, 2)
            else:
                grouped = sf_page.reshape(hkv, tpb // 4, groups, 4).permute(0, 1, 3, 2)
                got_sf = grouped.reshape(hkv, tpb, groups)[:, :n].permute(1, 0, 2)
            assert torch.equal(got_sf, scales[rows]), (
                f"request {request_id} role {role} page {j}: block scales"
            )


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """A softmax chain, so the band is the longest in this catalog.

    Attention accumulates twice -- once over the KV range inside the
    softmax, once over the same range in the value product -- and the
    kernel splits both differently from a single fp32 pass. 8 ulp
    element-wise against the row scale and 4 of relative RMS, the same band
    a two-GEMM MoE chain gets, and for the same reason.

    Scored in bf16 ulp whatever the output dtype: on the fp8 paths the
    operands are e4m3 and the accumulation is bf16-limited, so fp32's ulp
    would measure the container. The standard configuration's fp8 paths
    also round the softmax probabilities to e4m3, which this band does not
    absorb (5-6 ulp RMS measured); `reference(e4m3_probabilities=True)`
    models it instead. Measured on sm_103: the qwen4_exp, encoder and fp8
    context cells use at most 0.25 of the element gate and 0.16 of the RMS
    one; the fp8 200-token decode, whose KV range spans tiles, 0.58 of both.
    """
    assert_within_ulp(out, ref, element_ulp=8.0, rms_ulp=4.0, ulp=ULP[torch.bfloat16])


class _PagedAttnEnv:
    """Real op state: caller-owned paged KV pool + explicit metadata tensors.

    Also mirrors every K/V fed to the op per request, so references can be
    built from the true sequence history, and the paged cache can be checked
    against it.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tokens_per_block: int = 32,
        num_blocks: int = 64,
        max_batch: int = 4,
        max_blocks_per_seq: int = 8,
        pool_dtype: torch.dtype = torch.bfloat16,
        quant_mode: int = 0,
        kv_scaling_factor: Optional[float] = None,
        attention_window_size: Optional[int] = None,
        page_ring: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_batch = max_batch
        self.max_seq_len = max_blocks_per_seq * tokens_per_block
        self.quant_mode = quant_mode
        # None = no sliding window (window == max_seq_len). page_ring is the
        # caller-side page budget: with it set, absolute page index j of a
        # sequence is served by the j % page_ring-th physical page, so the pool
        # holds a bounded window of the sequence and genuinely wraps.
        self.attention_window_size = (
            self.max_seq_len if attention_window_size is None else attention_window_size
        )
        self.page_ring = page_ring
        self.rings: dict[int, List[int]] = {}

        # Single-layer, single-pool paged KV cache (HND layout). An NVFP4 pool
        # holds e2m1 pairs, and its e4m3 block scales live in a second pool
        # whose pointer is the third axis of the pool pointers.
        self.nvfp4 = bool(quant_mode & QUANT_MODE_NVFP4_KV_CACHE)
        self.kv_cache = torch.zeros(
            num_blocks,
            2,
            num_kv_heads,
            tokens_per_block,
            head_dim // 2 if self.nvfp4 else head_dim,
            dtype=torch.uint8 if self.nvfp4 else pool_dtype,
            device="cuda",
        )
        if self.nvfp4:
            self.kv_block_scales = torch.zeros(
                num_blocks,
                2,
                num_kv_heads,
                tokens_per_block,
                head_dim // NVFP4_BLOCK,
                dtype=torch.uint8,
                device="cuda",
            )
            self.pool_pointers = torch.zeros(1, 2, 2, dtype=torch.int64, device="cpu")
            self.pool_pointers[0, 0, 0] = self.kv_cache.data_ptr()
            self.pool_pointers[0, 0, 1] = self.kv_block_scales.data_ptr()
        else:
            self.pool_pointers = torch.zeros(1, 2, dtype=torch.int64, device="cpu")
            self.pool_pointers[0, 0] = self.kv_cache.data_ptr()
        # KV-cache scaling factor s: dequant = quantized * s. Mirrors the
        # production construction (quant_orig = s, orig_quant = 1/s, both
        # fp32 [1] CUDA, [3] -- q, K, V -- over an NVFP4 pool); None =
        # unquantized pool, scale args passed as None.
        if kv_scaling_factor is None:
            self.kv_scale_quant_orig: Optional[torch.Tensor] = None
            self.kv_scale_orig_quant: Optional[torch.Tensor] = None
        else:
            factor = torch.full(
                (3 if self.nvfp4 else 1,), kv_scaling_factor, dtype=torch.float32, device="cuda"
            )
            self.kv_scale_quant_orig = factor
            self.kv_scale_orig_quant = 1.0 / factor
        self.pool_mapping = torch.zeros(1, 2, dtype=torch.int32, device="cpu")
        self.block_offsets = torch.zeros(
            1, max_batch, 2, max_blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        # Auto-resized in place by the op on first call.
        self.workspace = torch.empty(0, dtype=torch.int8, device="cuda")

        self._next_free_page = 0
        self.pages: dict[int, List[int]] = {}
        self.prompt_lens: dict[int, int] = {}
        self.k_history: dict[int, List[torch.Tensor]] = {}
        self.v_history: dict[int, List[torch.Tensor]] = {}

    def add_request(self, request_id: int, prompt_len: int) -> None:
        self.pages[request_id] = []
        self.prompt_lens[request_id] = prompt_len
        self.k_history[request_id] = []
        self.v_history[request_id] = []

    def cached_len(self, request_id: int) -> int:
        return sum(t.shape[0] for t in self.k_history[request_id])

    def _ensure_pages(self, request_id: int, total_tokens: int) -> None:
        tpb = self.tokens_per_block
        needed = (total_tokens + tpb - 1) // tpb
        pages = self.pages[request_id]
        if self.page_ring is not None:
            ring = self.rings.setdefault(request_id, [])
            while len(ring) < self.page_ring:
                ring.append(self._next_free_page)
                self._next_free_page += 1
            while len(pages) < needed:
                pages.append(ring[len(pages) % self.page_ring])
            return
        while len(pages) < needed:
            pages.append(self._next_free_page)
            self._next_free_page += 1

    def call_op(
        self,
        qkv: torch.Tensor,
        seq_lens: List[int],
        num_contexts: int,
        request_ids: List[int],
        mask_type: int,
        attention_sinks: Optional[torch.Tensor] = None,
        record: bool = True,
        seq_lens_override: Optional[List[int]] = None,
        ctx_lens_override: Optional[List[int]] = None,
        max_seq_len_override: Optional[int] = None,
        use_paged_context_fmha: bool = False,
        predicted_tokens_per_seq: int = 1,
        alias_pages: bool = False,
        op_overrides: Optional[Dict] = None,
    ) -> torch.Tensor:
        """One thop_attention call over explicitly constructed batch state.

        record=False skips the K/V history mirror, so the identical call can be
        repeated (the append is idempotent) to compare code paths bit for bit.
        The *_override arguments feed the op varied batch state
        (sequence_length, context_lengths, max_seq_len) to pin what each one
        actually has to carry — deliberately wrong values in the batch-state
        probes, and the correct per-call context length on the paged-context
        path, where a context row's context_lengths is this call's new-token
        count rather than the registered prompt length.
        use_paged_context_fmha=True selects the paged-context execution path
        (see the paged-context section).
        predicted_tokens_per_seq > 1 makes each generation sequence contribute
        that many q rows (one speculative-decoding step); seq_lens then carries
        P for a generation row, which is also its new-token count, so the page
        bookkeeping and the K/V mirror need no special case.
        alias_pages points every entry of a sequence's offsets row at its first
        physical page -- the torn-append arming sequence, used only as a control
        that the pool comparison can see a racing append.
        op_overrides replaces named arguments of the op call itself, for cases
        that sweep an argument this helper does not otherwise expose.
        """
        ns = len(request_ids)
        kv_lens = []  # cached + new, per sequence
        ctx_lens = []  # prompt length, per sequence
        for rid, new in zip(request_ids, seq_lens):
            total = self.cached_len(rid) + new
            assert total <= self.max_seq_len
            self._ensure_pages(rid, total)
            kv_lens.append(total)
            ctx_lens.append(self.prompt_lens[rid])
            # K page offset of page p is 2*p, V is 2*p + 1 (single-layer pool).
            row = self.block_offsets[0, len(kv_lens) - 1]
            for j, p in enumerate(self.pages[rid]):
                page = self.pages[rid][0] if alias_pages else p
                row[0, j] = 2 * page
                row[1, j] = 2 * page + 1
        if seq_lens_override is not None:
            kv_lens = list(seq_lens_override)
        if ctx_lens_override is not None:
            ctx_lens = list(ctx_lens_override)

        req_types = [0 if i < num_contexts else 1 for i in range(ns)]
        total_ctx_kv = sum(kv_lens[:num_contexts])
        total_gen_kv = sum(kv_lens[num_contexts:])
        num_ctx_tokens = sum(seq_lens[:num_contexts])

        num_tokens = sum(seq_lens)
        output = torch.empty(
            num_tokens,
            self.num_heads * self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )
        op_kwargs = dict(
            q=qkv,
            k=None,  # packed QKV rides inside q
            v=None,
            output=output,
            output_sf=None,
            workspace=self.workspace,
            sequence_length=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
            host_past_key_value_lengths=torch.tensor(kv_lens, dtype=torch.int32),
            host_total_kv_lens=torch.tensor([total_ctx_kv, total_gen_kv], dtype=torch.int32),
            context_lengths=torch.tensor(ctx_lens, dtype=torch.int32, device="cuda"),
            host_context_lengths=torch.tensor(ctx_lens, dtype=torch.int32),
            host_request_types=torch.tensor(req_types, dtype=torch.int32),
            max_context_q_len_override=None,
            kv_cache_block_offsets=self.block_offsets,
            host_kv_cache_pool_pointers=self.pool_pointers,
            host_kv_cache_pool_mapping=self.pool_mapping,
            cache_indirection=None,
            kv_scale_orig_quant=self.kv_scale_orig_quant,
            kv_scale_quant_orig=self.kv_scale_quant_orig,
            out_scale=None,
            rotary_inv_freq=None,
            rotary_cos_sin=None,
            latent_cache=None,
            q_pe=None,
            block_ids_per_seq=None,
            attention_sinks=attention_sinks,
            is_fused_qkv=True,
            update_kv_cache=True,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            local_layer_idx=0,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            tokens_per_block=self.tokens_per_block,
            max_num_requests=self.max_batch,
            max_context_length=self.max_seq_len,
            max_seq_len=(
                self.max_seq_len if max_seq_len_override is None else max_seq_len_override
            ),
            attention_window_size=self.attention_window_size,
            beam_width=1,
            mask_type=mask_type,
            quant_mode=self.quant_mode,
            q_scaling=1.0,
            position_embedding_type=0,  # no in-kernel RoPE
            rope_dim=0,
            rope_base=10000.0,
            rope_scale_type=0,
            rope_scale=1.0,
            rope_short_m_scale=1.0,
            rope_long_m_scale=1.0,
            rope_max_positions=1024,
            rope_original_max_positions=1024,
            use_paged_context_fmha=use_paged_context_fmha,
            attention_input_type=0,  # mixed
            is_mla_enable=False,
            chunked_prefill_buffer_batch_size=1,
            q_lora_rank=None,
            kv_lora_rank=None,
            qk_nope_head_dim=None,
            qk_rope_head_dim=None,
            v_head_dim=None,
            rope_append=None,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            attention_chunk_size=None,
            softmax_stats_tensor=None,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_spec_dec_tree=False,
            spec_decoding_generation_lengths=None,
            spec_decoding_position_offsets=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=None,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=0,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
        )
        op_kwargs.update(op_overrides or {})
        with validating(op):
            op(**op_kwargs)
        torch.cuda.synchronize()

        # Record the K/V slices just appended to the cache, per request.
        if record:
            q_w = self.num_heads * self.head_dim
            kv_w = self.num_kv_heads * self.head_dim
            start = 0
            for rid, sl in zip(request_ids, seq_lens):
                rows = qkv[start : start + sl]
                self.k_history[rid].append(
                    rows[:, q_w : q_w + kv_w].view(sl, self.num_kv_heads, self.head_dim)
                )
                self.v_history[rid].append(
                    rows[:, q_w + kv_w :].view(sl, self.num_kv_heads, self.head_dim)
                )
                start += sl
        return output

    def reference(
        self,
        qkv: torch.Tensor,
        seq_lens: List[int],
        request_ids: List[int],
        cached_lens: List[int],
        mask_type: int,
        q_transform=None,
        kv_transform=None,
        window: Optional[int] = None,
        full_block: bool = False,
    ) -> torch.Tensor:
        """fp32 SDPA over each sequence's full K/V history (must be called
        after call_op so the history includes this call's K/V). q_transform /
        kv_transform, when given, map the bf16 q / K/V history first (e.g.
        the e4m3 round-trip a quantized cache imposes). window, when given,
        additionally drops keys older than the sliding window.
        full_block=True is the rival model for a predicted_tokens_per_seq > 1
        call -- every q row of a sequence sees its whole cached+new range, i.e.
        a draft row also sees its later siblings -- and exists only to be gated
        against."""
        q_w = self.num_heads * self.head_dim
        outs = []
        start = 0
        for rid, sl, cached in zip(request_ids, seq_lens, cached_lens):
            q_seq = qkv[start : start + sl, :q_w].view(sl, self.num_heads, self.head_dim)
            if q_transform is not None:
                q_seq = q_transform(q_seq)
            k_seq = torch.cat(self.k_history[rid])
            v_seq = torch.cat(self.v_history[rid])
            if kv_transform is not None:
                k_seq = kv_transform(k_seq)
                v_seq = kv_transform(v_seq)
            skv = k_seq.shape[0]
            if mask_type == MASK_CAUSAL:
                mask: Optional[torch.Tensor] = torch.zeros(
                    sl, skv, dtype=torch.bool, device=qkv.device
                )
                for i in range(sl):
                    lo = 0 if window is None else max(0, cached + i + 1 - window)
                    mask[i, lo : skv if full_block else cached + i + 1] = True
            else:  # padding: every query attends to every key
                assert window is None, "sliding window certified for causal only"
                mask = None
            o = F.scaled_dot_product_attention(
                q_seq.transpose(0, 1).float(),
                k_seq.transpose(0, 1).float(),
                v_seq.transpose(0, 1).float(),
                attn_mask=mask,
                enable_gqa=True,
            )
            outs.append(o.transpose(0, 1).reshape(sl, -1))
            start += sl
        return torch.cat(outs).to(qkv.dtype)

    def expected_cache(self, kv: torch.Tensor) -> torch.Tensor:
        """Map bf16 K/V fed to the op to the expected pool content: identity
        for a bf16 pool; fp32-scale-then-RN-cast for a quantized pool."""
        if self.kv_cache.dtype == torch.bfloat16:
            return kv
        assert self.kv_scale_orig_quant is not None
        return (kv.float() * self.kv_scale_orig_quant).to(self.kv_cache.dtype)

    def cache_pages_content(self, request_id: int) -> Tuple[torch.Tensor, ...]:
        """(pool K rows, pool V rows, expected K, expected V), token-major."""
        k_seq = torch.cat(self.k_history[request_id])  # [total, Hkv, D]
        v_seq = torch.cat(self.v_history[request_id])
        total = k_seq.shape[0]
        tpb = self.tokens_per_block
        got_k, got_v = [], []
        for j, p in enumerate(self.pages[request_id]):
            n = min(tpb, total - j * tpb)
            if n <= 0:
                break
            got_k.append(self.kv_cache[p, 0, :, :n, :].permute(1, 0, 2))
            got_v.append(self.kv_cache[p, 1, :, :n, :].permute(1, 0, 2))
        return (
            torch.cat(got_k).contiguous(),
            torch.cat(got_v).contiguous(),
            self.expected_cache(k_seq),
            self.expected_cache(v_seq),
        )

    def check_cache(self, request_id: int) -> None:
        """The paged cache must hold exactly the expected rows, bitwise (the
        bf16 K/V fed so far, or their scaled e4m3 casts for an fp8 pool, or
        their NVFP4 bytes and block scales for an NVFP4 pool)."""
        if self.nvfp4:
            _check_nvfp4_cache(self, request_id)
            return
        got_k, got_v, exp_k, exp_v = self.cache_pages_content(request_id)
        assert torch.equal(got_k.view(torch.uint8), exp_k.view(torch.uint8)), (
            f"K cache mismatch: request {request_id}"
        )
        assert torch.equal(got_v.view(torch.uint8), exp_v.view(torch.uint8)), (
            f"V cache mismatch: request {request_id}"
        )

    def check_unwritten_pool_zero(self) -> None:
        """Pool bytes no append should have written must still hold the
        initial zeros: whole pages never allocated to a request, and the
        unwritten tail rows of each request's partially filled last page."""
        tpb = self.tokens_per_block
        written: dict[int, int] = {}
        for rid, pages in self.pages.items():
            total = self.cached_len(rid)
            for j, p in enumerate(pages):
                written[p] = min(tpb, total - j * tpb)
        for p in range(self.kv_cache.shape[0]):
            n = written.get(p, 0)
            if n >= tpb:
                continue
            tail = self.kv_cache[p, :, :, n:, :].contiguous()
            assert (tail.view(torch.uint8) == 0).all(), f"page {p} written beyond token row {n}"

    def random_qkv(self, num_tokens: int) -> torch.Tensor:
        width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        return torch.randn(num_tokens, width, dtype=torch.bfloat16, device="cuda")


def test_rejects_non_divisible_head_counts() -> None:
    """Hq must be an integer multiple of Hkv, and the wrapper must be the
    one to say so: a context-only call at 6q/4kv d128 returns without
    raising, having computed only the first (6 // 4) * 4 = 4 head columns
    and left the other two all-zero (the decode path does raise, from
    xqaDispatcher: 'numQHeads should be multiple of numKVHeads')."""
    with pytest.raises(AssertionError, match="must be a multiple of"):
        op.is_valid(num_heads=6, num_kv_heads=4)


# ─── Standard configuration, multi-layer shared pool ───────────────────


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise tensor equality (byte view), dtype-agnostic: float equality
    would miss NaN-payload or signed-zero byte changes in an fp8 pool."""
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class _MultiLayerPagedAttnEnv:
    """Real multi-layer op state: a trtllm KVCacheManagerV2 hosting one paged
    pool shared by several layers, whose pool pointers, layer->pool mapping,
    and block offsets the op consumes exactly as produced. The pool element
    type follows the manager dtype (bf16, or fp8-e4m3 with the same quant
    args as the single-layer fp8 env).

    The manager owns page allocation; the test mirrors every K/V fed per
    (layer, request) so references and bitwise cache checks come from the
    true per-layer history. Page ids are re-derived from the manager's
    offsets, pinning the multi-layer stride: page p of an L-layer pool has
    K-slab offset p * L * 2 and V-slab offset p * L * 2 + 1.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        tokens_per_block: int = 32,
        num_blocks: int = 64,
        max_batch: int = 4,
        max_seq_len: int = 256,
        dtype: DataType = DataType.BF16,
        quant_mode: int = 0,
        kv_scaling_factor: Optional[float] = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.quant_mode = quant_mode

        self.mgr = KVCacheManagerV2(
            KvCacheConfig(
                max_tokens=num_blocks * tokens_per_block,
                enable_block_reuse=False,
            ),
            CacheTypeCpp.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=dtype,
        )
        # KV-cache scaling factor s, same construction as the single-layer
        # env: quant_orig = s, orig_quant = 1/s, both fp32 [1] CUDA; None =
        # unquantized pool, scale args passed as None.
        if kv_scaling_factor is None:
            self.kv_scale_quant_orig: Optional[torch.Tensor] = None
            self.kv_scale_orig_quant: Optional[torch.Tensor] = None
        else:
            factor = torch.full((1,), kv_scaling_factor, dtype=torch.float32, device="cuda")
            self.kv_scale_quant_orig = factor
            self.kv_scale_orig_quant = 1.0 / factor
        assert self.mgr.num_pools == 1, "test expects one pool holding all layers"
        pool_mapping = self.mgr.kv_cache_pool_mapping
        assert pool_mapping is not None
        self.pool_mapping: torch.Tensor = pool_mapping
        self.block_offsets = torch.zeros(
            1,
            max_batch,
            2,
            self.mgr.max_blocks_per_seq,
            dtype=torch.int32,
            device="cuda",
        )
        # Per-layer HND views into the shared pool, each
        # [num_blocks, 2, Hkv, tokens_per_block, D] strided over whole pages.
        self.layer_views: List[torch.Tensor] = []
        for idx in range(num_layers):
            view = self.mgr.get_buffers(idx, kv_layout="HND")
            assert view is not None
            page_scale = self.mgr.get_layer_page_index_scale(idx)
            assert page_scale == num_layers * 2
            view = view[:: page_scale // 2]
            view.zero_()
            self.layer_views.append(view)
        # Auto-resized in place by the op on first call.
        self.workspace = torch.empty(0, dtype=torch.int8, device="cuda")

        self.prompt_lens: Dict[int, int] = {}
        self.k_history: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        self.v_history: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    def add_request(self, request_id: int, prompt_len: int) -> None:
        """Register the request with the manager, allocating its prompt pages."""
        self.prompt_lens[request_id] = prompt_len
        added = self.mgr.add_dummy_requests([request_id], token_nums=[prompt_len])
        assert added is not None, "KV cache manager out of blocks"
        for layer_idx in range(self.num_layers):
            self.k_history[layer_idx, request_id] = []
            self.v_history[layer_idx, request_id] = []

    def add_decode_token(self, request_id: int) -> None:
        """Extend the manager's allocation by one generated token."""
        cache = self.mgr.kv_cache_map[request_id]
        assert cache.resize(cache.capacity + 1)

    def refresh_offsets(self, request_ids: List[int], num_contexts: int) -> None:
        """Copy the manager-produced block offsets for this batch to device."""
        self.mgr.copy_batch_block_offsets(
            self.block_offsets, request_ids, 1, num_contexts, len(request_ids)
        )
        torch.cuda.synchronize()  # the copy is staged non-blocking

    def cached_len(self, layer_idx: int, request_id: int) -> int:
        return sum(t.shape[0] for t in self.k_history[layer_idx, request_id])

    def call_op(
        self,
        layer_idx: int,
        qkv: torch.Tensor,
        seq_lens: List[int],
        num_contexts: int,
        request_ids: List[int],
        pool_mapping: Optional[torch.Tensor] = None,
        record: bool = True,
        attention_window_size: Optional[int] = None,
        use_paged_context_fmha: bool = False,
        ctx_lens: Optional[List[int]] = None,
        pool_pointers: Optional[torch.Tensor] = None,
        block_offsets: Optional[torch.Tensor] = None,
        local_layer_idx: Optional[int] = None,
        predicted_tokens_per_seq: int = 1,
    ) -> torch.Tensor:
        """One thop_attention call for one layer of the shared pool.

        Offsets must have been refreshed for exactly this request order.
        pool_mapping overrides the manager's mapping (doctored-mapping
        probe); record=False skips the K/V history mirror for such calls.
        attention_window_size defaults to max_seq_len (no sliding window) and
        is a per-call scalar, so layers sharing the pool may differ in it.
        use_paged_context_fmha=True selects the paged-context execution path;
        ctx_lens then carries this call's per-context-row new-token count,
        which a cached prefix makes differ from the registered prompt length.

        pool_pointers and block_offsets override the manager's, which is what
        lets one env stand in for one *pool* of a several-pool layout: the
        caller composes the tables across envs and this call addresses its own
        rows through them. A single manager cannot produce that -- it owns one
        pool -- and the pool-index column of the mapping is unexercised
        without it.

        local_layer_idx defaults to layer_idx and separates the two meanings
        that coincide in a single-pool layout: layer_idx is this env's own
        layer (its K/V history and buffer view), while local_layer_idx is the
        row of the composed mapping. Across pools they differ.

        predicted_tokens_per_seq > 1 makes each generation sequence contribute
        that many token-major q rows; seq_lens then carries P for a generation
        row, which is also its new-token count.
        """
        ns = len(request_ids)
        kv_lens = [self.cached_len(layer_idx, rid) + new for rid, new in zip(request_ids, seq_lens)]
        if ctx_lens is None:
            ctx_lens = [self.prompt_lens[rid] for rid in request_ids]
        req_types = [0 if i < num_contexts else 1 for i in range(ns)]
        num_ctx_tokens = sum(seq_lens[:num_contexts])
        if pool_mapping is None:
            pool_mapping = self.pool_mapping
        if pool_pointers is None:
            pool_pointers = self.mgr.kv_cache_pool_pointers
        if block_offsets is None:
            block_offsets = self.block_offsets
        if local_layer_idx is None:
            local_layer_idx = layer_idx

        output = torch.empty(
            sum(seq_lens),
            self.num_heads * self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )
        with validating(op):
            op(
                q=qkv,
                k=None,  # packed QKV rides inside q
                v=None,
                output=output,
                output_sf=None,
                workspace=self.workspace,
                sequence_length=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
                host_past_key_value_lengths=torch.tensor(kv_lens, dtype=torch.int32),
                host_total_kv_lens=torch.tensor(
                    [sum(kv_lens[:num_contexts]), sum(kv_lens[num_contexts:])],
                    dtype=torch.int32,
                ),
                context_lengths=torch.tensor(ctx_lens, dtype=torch.int32, device="cuda"),
                host_context_lengths=torch.tensor(ctx_lens, dtype=torch.int32),
                host_request_types=torch.tensor(req_types, dtype=torch.int32),
                max_context_q_len_override=None,
                kv_cache_block_offsets=block_offsets,
                host_kv_cache_pool_pointers=pool_pointers,
                host_kv_cache_pool_mapping=pool_mapping,
                cache_indirection=None,
                kv_scale_orig_quant=self.kv_scale_orig_quant,
                kv_scale_quant_orig=self.kv_scale_quant_orig,
                out_scale=None,
                rotary_inv_freq=None,
                rotary_cos_sin=None,
                latent_cache=None,
                q_pe=None,
                block_ids_per_seq=None,
                attention_sinks=None,
                is_fused_qkv=True,
                update_kv_cache=True,
                predicted_tokens_per_seq=predicted_tokens_per_seq,
                local_layer_idx=local_layer_idx,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                tokens_per_block=self.tokens_per_block,
                max_num_requests=self.max_batch,
                max_context_length=self.max_seq_len,
                max_seq_len=self.max_seq_len,
                attention_window_size=(
                    self.max_seq_len if attention_window_size is None else attention_window_size
                ),
                beam_width=1,
                mask_type=MASK_CAUSAL,
                quant_mode=self.quant_mode,
                q_scaling=1.0,
                position_embedding_type=0,  # no in-kernel RoPE
                rope_dim=0,
                rope_base=10000.0,
                rope_scale_type=0,
                rope_scale=1.0,
                rope_short_m_scale=1.0,
                rope_long_m_scale=1.0,
                rope_max_positions=1024,
                rope_original_max_positions=1024,
                use_paged_context_fmha=use_paged_context_fmha,
                attention_input_type=0,  # mixed
                is_mla_enable=False,
                chunked_prefill_buffer_batch_size=1,
                q_lora_rank=None,
                kv_lora_rank=None,
                qk_nope_head_dim=None,
                qk_rope_head_dim=None,
                v_head_dim=None,
                rope_append=None,
                mrope_rotary_cos_sin=None,
                mrope_position_deltas=None,
                helix_position_offsets=None,
                helix_is_inactive_rank=None,
                attention_chunk_size=None,
                softmax_stats_tensor=None,
                is_spec_decoding_enabled=False,
                use_spec_decoding=False,
                is_spec_dec_tree=False,
                spec_decoding_generation_lengths=None,
                spec_decoding_position_offsets=None,
                spec_decoding_packed_mask=None,
                spec_decoding_bl_tree_mask_offset=None,
                spec_decoding_bl_tree_mask=None,
                spec_bl_tree_first_sparse_mask_offset_kv=None,
                sparse_kv_indices=None,
                sparse_kv_offsets=None,
                sparse_attn_indices=None,
                sparse_attn_offsets=None,
                sparse_attn_indices_block_size=0,
                num_contexts=num_contexts,
                num_ctx_tokens=num_ctx_tokens,
            )
        torch.cuda.synchronize()

        if record:
            q_w = self.num_heads * self.head_dim
            kv_w = self.num_kv_heads * self.head_dim
            start = 0
            for rid, sl in zip(request_ids, seq_lens):
                rows = qkv[start : start + sl]
                self.k_history[layer_idx, rid].append(
                    rows[:, q_w : q_w + kv_w].view(sl, self.num_kv_heads, self.head_dim)
                )
                self.v_history[layer_idx, rid].append(
                    rows[:, q_w + kv_w :].view(sl, self.num_kv_heads, self.head_dim)
                )
                start += sl
        return output

    def reference(
        self,
        layer_idx: int,
        qkv: torch.Tensor,
        seq_lens: List[int],
        request_ids: List[int],
        cached_lens: List[int],
        q_transform=None,
        kv_transform=None,
        window: Optional[int] = None,
    ) -> torch.Tensor:
        """fp32 causal SDPA over this layer's full K/V history (must be
        called after call_op so the history includes this call's K/V).
        q_transform / kv_transform, when given, map the bf16 q / K/V history
        first (e.g. the e4m3 round-trip a quantized cache imposes). window,
        when given, additionally drops keys older than the sliding window."""
        q_w = self.num_heads * self.head_dim
        outs = []
        start = 0
        for rid, sl, cached in zip(request_ids, seq_lens, cached_lens):
            q_seq = qkv[start : start + sl, :q_w].view(sl, self.num_heads, self.head_dim)
            if q_transform is not None:
                q_seq = q_transform(q_seq)
            k_seq = torch.cat(self.k_history[layer_idx, rid])
            v_seq = torch.cat(self.v_history[layer_idx, rid])
            if kv_transform is not None:
                k_seq = kv_transform(k_seq)
                v_seq = kv_transform(v_seq)
            mask = torch.zeros(sl, k_seq.shape[0], dtype=torch.bool, device=qkv.device)
            for i in range(sl):
                lo = 0 if window is None else max(0, cached + i + 1 - window)
                mask[i, lo : cached + i + 1] = True
            o = F.scaled_dot_product_attention(
                q_seq.transpose(0, 1).float(),
                k_seq.transpose(0, 1).float(),
                v_seq.transpose(0, 1).float(),
                attn_mask=mask,
                enable_gqa=True,
            )
            outs.append(o.transpose(0, 1).reshape(sl, -1))
            start += sl
        return torch.cat(outs).to(qkv.dtype)

    def expected_cache(self, kv: torch.Tensor) -> torch.Tensor:
        """Map bf16 K/V fed to the op to the expected pool content: identity
        for a bf16 pool; fp32-scale-then-RN-cast for a quantized pool."""
        pool_dtype = self.layer_views[0].dtype
        if pool_dtype == torch.bfloat16:
            return kv
        assert self.kv_scale_orig_quant is not None
        return (kv.float() * self.kv_scale_orig_quant).to(pool_dtype)

    def check_caches(self, request_ids: List[int], layers: Optional[List[int]] = None) -> None:
        """Every layer's slabs must hold exactly the expected rows, bitwise
        (the bf16 K/V fed to that layer, or their scaled e4m3 casts for an
        fp8 pool), at the pages the manager assigned (offsets are
        layer-agnostic and count slabs: one page = num_layers * 2 slabs).
        layers restricts the check to the layers that were driven: an undriven
        layer has no history to compare against."""
        tpb = self.tokens_per_block
        slabs_per_page = self.num_layers * 2
        for layer_idx in range(self.num_layers) if layers is None else layers:
            view = self.layer_views[layer_idx]
            for s, rid in enumerate(request_ids):
                k_seq = self.expected_cache(torch.cat(self.k_history[layer_idx, rid]))
                v_seq = self.expected_cache(torch.cat(self.v_history[layer_idx, rid]))
                total = k_seq.shape[0]
                for j in range((total + tpb - 1) // tpb):
                    k_off = int(self.block_offsets[0, s, 0, j].item())
                    v_off = int(self.block_offsets[0, s, 1, j].item())
                    assert k_off % slabs_per_page == 0, "multi-layer K offset stride"
                    assert v_off == k_off + 1, "V slab follows K within the page"
                    page = k_off // slabs_per_page
                    n = min(tpb, total - j * tpb)
                    k_blk = view[page, 0, :, :n, :].permute(1, 0, 2)
                    v_blk = view[page, 1, :, :n, :].permute(1, 0, 2)
                    assert _bitwise_equal(k_blk, k_seq[j * tpb : j * tpb + n]), (
                        f"K cache mismatch: layer {layer_idx}, request {rid}, page {j}"
                    )
                    assert _bitwise_equal(v_blk, v_seq[j * tpb : j * tpb + n]), (
                        f"V cache mismatch: layer {layer_idx}, request {rid}, page {j}"
                    )

    def random_qkv(self, num_tokens: int) -> torch.Tensor:
        width = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        return torch.randn(num_tokens, width, dtype=torch.bfloat16, device="cuda")


# Relative tolerance for the exact per-key attention weights read out by the
# one-hot probe below. Those weights are pure fp32 softmax constants rounded
# once to bf16: half an ulp is 2^-9 = 2.0e-3 relative, and the kernel's own
# fp32 softmax rounding lands on top. 6e-3 is ~1.5 bf16 ulp; the worst
# observed on sm_100 across the probe cases is 3.6e-3, 59% of it.
SWA_WEIGHT_RTOL = 6e-3


def _paged_ctx_call(
    env: _PagedAttnEnv,
    qkv: torch.Tensor,
    seq_lens: List[int],
    num_contexts: int,
    request_ids: List[int],
    **kwargs,
) -> torch.Tensor:
    """One use_paged_context_fmha=True call. A context row's context_lengths
    is this call's new-token (q-row) count — what the engine puts in
    prompt_lens for a context request served over a cached prefix — while a
    generation row keeps the registered prompt length."""
    ctx_lens = [
        seq_lens[i] if i < num_contexts else env.prompt_lens[rid]
        for i, rid in enumerate(request_ids)
    ]
    return env.call_op(
        qkv,
        seq_lens,
        num_contexts,
        request_ids,
        MASK_CAUSAL,
        ctx_lens_override=ctx_lens,
        use_paged_context_fmha=True,
        **kwargs,
    )


def _band_fraction(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> float:
    """How much of the atol + rtol*|ref| allowance the largest error uses —
    elementwise, the statistic torch.testing.assert_close itself gates on, so
    a value above 1.0 is exactly an assert_close failure, and by how much."""
    diff = (out.float() - ref.float()).abs()
    return float((diff / (atol + rtol * ref.float().abs())).max())


def _assert_far_outside_band(
    out: torch.Tensor,
    rival: torch.Tensor,
    threshold: float,
    label: str,
    atol: float,
    rtol: float,
) -> None:
    """A rival hypothesis has to fail assert_close by at least this factor."""
    ratio = _band_fraction(out, rival, atol, rtol)
    assert ratio > threshold, f"{label}: only {ratio:.4g}x outside the band"


# ─── GQA 24/x/256 and the encoder-only mode: shared helpers ────────────


def _history(env, rid: int, layer: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """The K/V one request's cache holds (for one layer of a shared pool)."""
    if layer is None:
        return torch.cat(env.k_history[rid]), torch.cat(env.v_history[rid])
    return torch.cat(env.k_history[layer, rid]), torch.cat(env.v_history[layer, rid])


def _op_reference(
    env,
    qkv: torch.Tensor,
    seq_lens: List[int],
    request_ids: List[int],
    causal: bool = True,
    e4m3: bool = False,
    layer: Optional[int] = None,
    nvfp4: bool = False,
    partition: Optional[Tuple[int, Optional[int], int]] = None,
    e4m3_probabilities: Optional[bool] = None,
) -> torch.Tensor:
    """`op.reference` over the history the env mirrored for these requests.

    nvfp4=True reads K and V as an NVFP4 pool returns them (the history
    through `_nvfp4_roundtrip`) before the e4m3 arithmetic `e4m3` selects.

    Call it after the op call, so each history ends with this call's K/V:
    `reference`'s bottom-right alignment then puts this call's q rows at the
    end of it. causal=False on a generation call is the full-block rival --
    every draft row seeing its later siblings. e4m3=True is the standard
    configuration's fp8 arithmetic: q, K, V and the softmax probabilities all
    e4m3 operands; `partition` is `(kv_tile, kv_chunk, kv_insts)`, the KV
    partition `reference` rounds those probabilities over (None: the row max).
    e4m3_probabilities, when given, overrides e4m3 for the probabilities alone.
    """
    kv_tile, kv_chunk, kv_insts = (None, None, 1) if partition is None else partition
    hq, hkv, d = env.num_heads, env.num_kv_heads, env.head_dim
    ks, vs = zip(*(_history(env, rid, layer) for rid in request_ids))
    if nvfp4:
        ks = tuple(_nvfp4_roundtrip(k, env.kv_scale_orig_quant[1]) for k in ks)
        vs = tuple(_nvfp4_roundtrip(v, env.kv_scale_orig_quant[2]) for v in vs)
    return op.reference(
        qkv[:, : hq * d],
        torch.cat(ks).reshape(-1, hkv * d),
        torch.cat(vs).reshape(-1, hkv * d),
        list(seq_lens),
        [k.shape[0] for k in ks],
        num_heads=hq,
        num_kv_heads=hkv,
        qk_head_dim=d,
        v_head_dim=d,
        softmax_scale=1.0 / math.sqrt(d),
        causal=causal,
        e4m3_inputs=e4m3,
        e4m3_probabilities=e4m3 if e4m3_probabilities is None else e4m3_probabilities,
        kv_tile=kv_tile,
        kv_chunk=kv_chunk,
        kv_insts=kv_insts,
    )


#: The KV partitions a quantized-pool kernel rounds its softmax probabilities over
#: (`reference`'s `kv_tile`, `kv_chunk`, `kv_insts`): 128-key tiles in one CTA or in
#: CTA chunks of 256..1024 keys, dealt to one or two softmax instances; None is the
#: row max. The launch picks one by shape and batch, so a cell is gated against the
#: closest -- every member is the same attention and differs only in where e4m3
#: rounds a probability, which moves the output by at most 3.7 bf16 ulp RMS over the
#: cells here (the row max against the closest member), against 10.8-33.4 for K/V
#: read unquantized, 4.6-4.8 for the probabilities left in fp32 and 139-169 for each
#: sequence's newest key repeated (`test_quantized_decode_partition_gate_rejects_its_rivals`).
_E4M3_PARTITIONS: Tuple[Optional[Tuple[int, Optional[int], int]], ...] = (
    None,
    (128, None, 1),
    (128, None, 2),
    (128, 256, 1),
    (128, 256, 2),
    (128, 512, 1),
    (128, 512, 2),
    (128, 1024, 1),
    (128, 1024, 2),
)


def _closest_partition(out: torch.Tensor, reference_at) -> Tuple[torch.Tensor, str]:
    """`reference_at(partition)` at the member of `_E4M3_PARTITIONS` closest to
    `out` in relative RMS, and a label naming that member beside the row max's
    distance."""
    scored = []
    for partition in _E4M3_PARTITIONS:
        ref = reference_at(partition)
        diff = (out.float() - ref.float()).pow(2).mean().sqrt()
        scored.append((diff.item() / ref.float().pow(2).mean().sqrt().item(), partition, ref))
    rel, partition, ref = min(scored, key=lambda s: s[0])
    row = scored[0][0] / ULP[torch.bfloat16]
    return ref, f"partition {partition} (row max: {row:.3f} ulp RMS)"


def _packed_reference(
    qkv: torch.Tensor,
    seg_lens: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
) -> torch.Tensor:
    """`op.reference` over each segment's own packed rows and nothing else:
    what the no-KV-cache mode computes, and the rival a cached-prefix call
    must be separated from (its prefix ignored)."""
    q_w, kv_w = num_heads * head_dim, num_kv_heads * head_dim
    return op.reference(
        qkv[:, :q_w],
        qkv[:, q_w : q_w + kv_w],
        qkv[:, q_w + kv_w :],
        list(seg_lens),
        list(seg_lens),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        softmax_scale=1.0 / math.sqrt(head_dim),
        causal=causal,
    )


def _std_onehot_qkv(
    env: _PagedAttnEnv, n_tokens: int, positions: List[int], t0: int
) -> torch.Tensor:
    """Packed QKV that turns the op into an attention-weight read-out: K is all
    zero, so every unmasked logit is 0 and the softmax is uniform over exactly
    the attended set, and V[t] = e_{t - t0}, so output column d of a query row
    IS the weight that row gives key t0 + d. 1.0 and 0.0 are both exact in
    e4m3, so the probe survives the quantized pool unchanged."""
    q_w = env.num_heads * env.head_dim
    kv_w = env.num_kv_heads * env.head_dim
    qkv = torch.zeros(n_tokens, q_w + 2 * kv_w, dtype=torch.bfloat16, device="cuda")
    qkv[:, :q_w] = torch.randn(n_tokens, q_w, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(n_tokens, env.num_kv_heads, env.head_dim, dtype=torch.bfloat16, device="cuda")
    for i, t in enumerate(positions):
        d = t - t0
        if 0 <= d < env.head_dim:
            v[i, :, d] = 1.0
    qkv[:, q_w + kv_w :] = v.reshape(n_tokens, kv_w)
    return qkv


def _assert_draft_row_weights(
    out: torch.Tensor,
    env: _PagedAttnEnv,
    row: int,
    last_key: int,
    label: str,
    weight_rtol: float,
) -> None:
    """Under the one-hot probe, output column d of `row` is the weight it gives
    key d. A bottom-right-aligned causal block puts the last attended key at
    `last_key`: every column above it must be an exact zero (a later sibling
    contributing anything at all is the failure this exists to catch) and every
    column at or below it carries the uniform 1 / (last_key + 1)."""
    w = out.view(-1, env.num_heads, env.head_dim)[row].float()
    assert bool((w[:, last_key + 1 :] == 0).all()), (
        f"{label}: keys above {last_key} carry weight {w[:, last_key + 1 :].abs().max().item():.4g}"
    )
    torch.testing.assert_close(
        w[:, : last_key + 1],
        torch.full_like(w[:, : last_key + 1], 1.0 / (last_key + 1)),
        rtol=weight_rtol,
        atol=0.0,
        msg=lambda m: f"{label}: attended weights are not uniform\n{m}",
    )


# How far a rival model has to sit outside a band to count as separated.
FP8_STD_MIN_SEPARATION = 3.0

# ─── predicted_tokens_per_seq > 1 in the standard configuration ────────
#
# One speculative-decoding step: each generation sequence contributes P q rows
# (its draft chain), token-major. On sm_103 a linear draft leaves every
# spec_decoding_* tensor None and is_spec_decoding_enabled False, so the
# within-block mask has to come from predicted_tokens_per_seq alone. The cells
# gate P against `op.reference`; these read the mask out directly and measure
# the full-block rival.

# One configuration per head geometry and pool this target drafts over, with the
# verify heights its members run: 6/1 at tp4 / tep4 (MTP depth 1..7, P up to 8),
# 24/2 at dep4 (MTP3, P = 4) over bf16 and over fp8 at s = 1.0. The weight
# tolerance is the bf16 one-hot bound over a bf16 pool; over fp8 the softmax
# probabilities are an e4m3 MMA operand and carry its 2**-4 relative error --
# the zeros stay exactly zero, which is the part that pins the mask.
_DRAFT_POOLS = [
    ("bf16 6/1/256", 6, 1, torch.bfloat16, 0, None, SWA_WEIGHT_RTOL, (2, 4, 8)),
    ("bf16 24/2/256", 24, 2, torch.bfloat16, 0, None, SWA_WEIGHT_RTOL, (2, 4)),
    ("fp8 24/2/256", 24, 2, torch.float8_e4m3fn, QUANT_MODE_FP8_KV_CACHE, 1.0, 2**-4, (2, 4)),
]


def _draft_env(
    num_heads: int, num_kv_heads: int, pool_dtype, quant_mode: int, scale
) -> _PagedAttnEnv:
    return _PagedAttnEnv(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        tokens_per_block=32,
        num_blocks=128,
        max_blocks_per_seq=16,
        pool_dtype=pool_dtype,
        quant_mode=quant_mode,
        kv_scaling_factor=scale,
    )


@pytest.mark.parametrize("pool", _DRAFT_POOLS, ids=[c[0] for c in _DRAFT_POOLS])
def test_std_predicted_tokens_mask_is_bottom_right_causal(pool) -> None:
    """The within-block mask, read out rather than inferred.

    Cache rows carrying a one-hot V turn the output into the attention-weight
    vector. Draft row t of a sequence with L cached tokens attends keys
    [0, L + t] uniformly and its later siblings' columns come back bitwise
    zero, at every P the pool's members verify with. Rows are token-major
    within a sequence, all P tokens are appended, and mask_type 0 and 1 are
    bitwise identical.
    """
    name, hq, hkv, pool_dtype, quant_mode, scale, weight_rtol, heights = pool
    for p in heights:
        torch.manual_seed(450 + p)
        env = _draft_env(hq, hkv, pool_dtype, quant_mode, scale)
        cached = 40
        env.add_request(0, cached)
        env.add_request(1, cached)
        prefill = torch.cat([_std_onehot_qkv(env, cached, list(range(cached)), 0)] * 2)
        env.call_op(prefill, [cached, cached], 2, [0, 1], MASK_CAUSAL)
        gen = torch.cat([_std_onehot_qkv(env, p, [cached + t for t in range(p)], 0)] * 2)
        out = env.call_op(gen, [p, p], 0, [0, 1], MASK_CAUSAL, predicted_tokens_per_seq=p)
        for n in range(2 * p):
            _assert_draft_row_weights(
                out,
                env,
                n,
                cached + (n % p),
                f"{name} P={p} row {n} (sequence {n // p}, draft {n % p})",
                weight_rtol,
            )
        assert env.cached_len(0) == cached + p and env.cached_len(1) == cached + p
        env.check_cache(0)
        env.check_cache(1)
        env.check_unwritten_pool_zero()
        # mask_type is not what produces the within-block mask.
        gen2 = torch.cat([_std_onehot_qkv(env, p, [cached + p + t for t in range(p)], 0)] * 2)
        out_causal = env.call_op(
            gen2, [p, p], 0, [0, 1], MASK_CAUSAL, predicted_tokens_per_seq=p, record=False
        )
        out_padding = env.call_op(gen2, [p, p], 0, [0, 1], MASK_PADDING, predicted_tokens_per_seq=p)
        assert _bitwise_equal(out_causal, out_padding), (
            f"{name} P={p}: mask_type 0 and 1 differ in the standard configuration"
        )


@pytest.mark.parametrize("pool", _DRAFT_POOLS, ids=[c[0] for c in _DRAFT_POOLS])
def test_std_predicted_tokens_full_block_rival(pool) -> None:
    """Realistic inputs at each verify height (and a mixed step): inside
    `_compare` of the bottom-right causal reference, while the full-block
    model -- every draft row seeing its later siblings -- sits outside the
    fp8 decode band by at least FP8_STD_MIN_SEPARATION (the ratios are
    printed). At P = 1 the two coincide."""
    name, hq, hkv, pool_dtype, quant_mode, scale, _, heights = pool
    fp8 = pool_dtype == torch.float8_e4m3fn
    atol, rtol = (FP8_DECODE_ATOL, FP8_DECODE_RTOL) if fp8 else (ATOL, RTOL)
    for p in heights:
        torch.manual_seed(460 + p)
        env = _draft_env(hq, hkv, pool_dtype, quant_mode, scale)
        env.add_request(0, 70)
        env.add_request(1, 35)
        _paged_ctx_call(env, env.random_qkv(105), [70, 35], 2, [0, 1])
        gen = env.random_qkv(2 * p)
        out = env.call_op(
            gen,
            [p, p],
            0,
            [0, 1],
            MASK_CAUSAL,
            predicted_tokens_per_seq=p,
            use_paged_context_fmha=True,
        )
        _compare(out, _op_reference(env, gen, [p, p], [0, 1], e4m3=fp8))
        rival = _op_reference(env, gen, [p, p], [0, 1], causal=False, e4m3=fp8)
        print(f"  {name} P={p}: full-block rival {_band_fraction(out, rival, atol, rtol):.1f}x")
        _assert_far_outside_band(
            out,
            rival,
            FP8_STD_MIN_SEPARATION,
            f"{name} P={p}: a draft row saw its later siblings",
            atol,
            rtol,
        )

    # A context row ahead of two draft blocks of different histories.
    torch.manual_seed(470)
    p = heights[-1]
    env = _draft_env(hq, hkv, pool_dtype, quant_mode, scale)
    env.add_request(0, 31)
    env.add_request(1, 64)
    env.add_request(2, 20)
    _paged_ctx_call(env, env.random_qkv(95), [31, 64], 2, [0, 1])
    mixed = env.random_qkv(20 + 2 * p)
    out = _paged_ctx_call(env, mixed, [20, p, p], 1, [2, 0, 1], predicted_tokens_per_seq=p)
    _compare(out, _op_reference(env, mixed, [20, p, p], [2, 0, 1], e4m3=fp8))
    rival = _op_reference(env, mixed, [20, p, p], [2, 0, 1], causal=False, e4m3=fp8)[20:]
    print(
        f"  {name} mixed P={p}: full-block rival {_band_fraction(out[20:], rival, atol, rtol):.1f}x"
    )
    _assert_far_outside_band(
        out[20:],
        rival,
        FP8_STD_MIN_SEPARATION,
        f"{name} mixed step: a draft row saw its later siblings",
        atol,
        rtol,
    )


# ─── qwen4_exp's full-attention cell: GQA 24/2/256 over a bf16 pool ────
#
# 24 query / 2 kv heads / head_size 256, tokens_per_block 32, causal, no sinks
# or window, the paged-context path on, P up to 4. 2 kv heads put 12 query heads
# in a group where 24/4 puts 6, which moves the compiled decode kernel's q-tile
# (VarSeqQ16 against VarSeqQ8), so 24/2/256 is certified as a triple of its own.
# Every rival here is a reference model computed on the host, never a second
# deliberately wrong call.

QWEN_HQ, QWEN_HKV, QWEN_D = 24, 2, 256
QWEN_TPB = 32

# How far a rival model has to sit outside the bf16 band. Measured on sm_103:
# 45x-118x (a draft row seeing its later siblings), 183x (a context call's
# cached prefix dropped), 17.8x-582x (an encoder's frame segmentation dropped).
QWEN_MIN_SEPARATION = 3.0


def _qwen_bf16_env(**kwargs) -> _PagedAttnEnv:
    """The checkpoint's full-attention cell over a bf16 paged pool."""
    kwargs.setdefault("num_heads", QWEN_HQ)
    kwargs.setdefault("num_kv_heads", QWEN_HKV)
    kwargs.setdefault("head_dim", QWEN_D)
    kwargs.setdefault("tokens_per_block", QWEN_TPB)
    kwargs.setdefault("num_blocks", 128)
    kwargs.setdefault("max_blocks_per_seq", 16)
    return _PagedAttnEnv(**kwargs)


def test_bf16_gqa24_2_cached_prefix_rival() -> None:
    """A 96-token request served by two paged-context calls, 48 then 33 over
    the 48-token prefix: each inside `_compare`, and the second far outside
    the band around the prefix-ignored model -- every row seeing only this
    call's own 33 tokens, which is what a call that never reached the pool
    computes (183x on sm_103). On the fresh call the two coincide."""
    torch.manual_seed(523)
    env = _qwen_bf16_env()
    env.add_request(0, 96)
    for new_tokens in (48, 33):
        cached = env.cached_len(0)
        qkv = env.random_qkv(new_tokens)
        out = _paged_ctx_call(env, qkv, [new_tokens], 1, [0])
        _compare(out, _op_reference(env, qkv, [new_tokens], [0]))
        if cached:
            _assert_far_outside_band(
                out,
                _packed_reference(qkv, [new_tokens], QWEN_HQ, QWEN_HKV, QWEN_D, causal=True),
                QWEN_MIN_SEPARATION,
                f"the {cached}-token cached prefix never reached the FMHA",
                ATOL,
                RTOL,
            )
        env.check_cache(0)
    env.check_unwritten_pool_zero()


def test_bf16_gqa24_2_sparse_group_is_inert() -> None:
    """The sparse / skip-softmax arguments, at and off their inert values.

    In the standard configuration nothing in the group moves a byte: each of
    nine settings -- the four index/offset tensors filled with a real
    8-key-per-row selection, a non-zero sparse_attn_indices_block_size, a
    num_sparse_topk, sparse_attn_kv_lens, and the two skip-softmax thresholds
    with and without a stats buffer -- returns output bitwise identical to the
    all-inert call. So a caller passing something here is not silently getting
    sparse attention either.

    A negative result needs a control: the same byte comparison over arguments
    the op does read (q_scaling 2.0, mask_type 0) must fail.
    """
    torch.manual_seed(522)
    env = _qwen_bf16_env()
    env.add_request(0, 40)
    qkv = env.random_qkv(40)

    def again(mask_type: int = MASK_CAUSAL, **overrides) -> torch.Tensor:
        # record=False throughout: the history mirror is what would turn a
        # repeat into a cached-prefix call, and these stay one fresh prefill.
        return env.call_op(
            qkv,
            [40],
            1,
            [0],
            mask_type,
            record=False,
            use_paged_context_fmha=True,
            op_overrides=overrides,
        )

    base = again()
    assert _bitwise_equal(again(), base), (
        "the identical call is not reproducible, so no byte comparison below means anything"
    )

    keep = 8  # a selection that could not reproduce full causal attention
    attn_idx = torch.tensor([list(range(keep))] * 40, dtype=torch.int32, device="cuda").reshape(-1)
    attn_off = torch.tensor([keep * i for i in range(41)], dtype=torch.int32, device="cuda")
    off_inert = {
        "sparse_attn_indices + sparse_attn_offsets": dict(
            sparse_attn_indices=attn_idx, sparse_attn_offsets=attn_off
        ),
        "... + sparse_attn_indices_block_size": dict(
            sparse_attn_indices=attn_idx,
            sparse_attn_offsets=attn_off,
            sparse_attn_indices_block_size=1,
        ),
        "... + num_sparse_topk": dict(
            sparse_attn_indices=attn_idx,
            sparse_attn_offsets=attn_off,
            sparse_attn_indices_block_size=1,
            num_sparse_topk=keep,
        ),
        "sparse_kv_indices + sparse_kv_offsets": dict(
            sparse_kv_indices=attn_idx, sparse_kv_offsets=attn_off
        ),
        "num_sparse_topk alone": dict(num_sparse_topk=8),
        "sparse_attn_indices_block_size alone": dict(sparse_attn_indices_block_size=64),
        "sparse_attn_kv_lens": dict(
            sparse_attn_kv_lens=torch.zeros(1, dtype=torch.int32, device="cuda")
        ),
        "skip_softmax thresholds": dict(
            skip_softmax_threshold_scale_factor_prefill=0.9,
            skip_softmax_threshold_scale_factor_decode=0.9,
        ),
        "skip_softmax thresholds + stat": dict(
            skip_softmax_threshold_scale_factor_prefill=0.9,
            skip_softmax_threshold_scale_factor_decode=0.9,
            skip_softmax_stat=torch.zeros(40, QWEN_HQ, 2, dtype=torch.float32, device="cuda"),
        ),
    }
    for label, overrides in off_inert.items():
        assert _bitwise_equal(again(**overrides), base), (
            f"{label} changed the output: the sparse / skip-softmax group is not inert here"
        )
    assert not _bitwise_equal(again(q_scaling=2.0), base), (
        "q_scaling 2.0 is bitwise identical to 1.0: this case's byte comparison is blind"
    )
    assert not _bitwise_equal(again(mask_type=MASK_PADDING), base), (
        "mask_type 0 is bitwise identical to 1 on a 40-token context call: "
        "this case's byte comparison is blind"
    )


# ─── local_layer_idx over qwen4_exp's 48-row mapping ───────────────────
#
# The 48-layer tower carries a full-attention layer every fourth position, and
# the target passes the layer's `layer_offsets` entry, so it drives rows 3, 7,
# ..., 47. Two arrangements, because the mapping's two columns move
# independently and only one of them is the row index:
#
#   * the engine's own -- a 48-row mapping whose twelve attention rows sit on
#     pool 0 at layer-in-pool 0..11 and whose other 36 rows name a second pool,
#     so the row index (to 47) and the layer-in-pool column (to 11) differ;
#   * a real 48-layer KVCacheManager, whose identity mapping drives both to 47.
#
# Both assert the pair a single-layer pool cannot: the named layer's slabs
# moved, AND every sibling layer's slabs are bitwise untouched.

QWEN_NUM_LAYERS = 48
QWEN_ATTN_LAYERS = list(range(3, QWEN_NUM_LAYERS, 4))


def _qwen_layer_pools(
    attn_blocks: int = 48,
    other_blocks: int = 32,
    max_seq_len: int = 256,
    max_batch: int = 4,
    num_heads: int = QWEN_HQ,
    num_kv_heads: int = QWEN_HKV,
    **pool_kw,
) -> Tuple[_MultiLayerPagedAttnEnv, _MultiLayerPagedAttnEnv]:
    """The two pools of the hybrid mapping, at the full-attention cell.

    The second holds the 36 rows this cell never drives; it exists so those
    rows name a real, distinct pool base and the untouched assertion has
    somewhere to look. Both managers take the same max_seq_len and max_batch,
    which makes their block-offset tables concatenable into the per-pool table
    the op indexes by the mapping's pool column. `pool_kw` (dtype, quant_mode,
    kv_scaling_factor) goes to both, and so does the rank's head geometry.
    """
    kw = dict(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=QWEN_D,
        tokens_per_block=QWEN_TPB,
        max_seq_len=max_seq_len,
        max_batch=max_batch,
        **pool_kw,
    )
    attn = _MultiLayerPagedAttnEnv(num_layers=len(QWEN_ATTN_LAYERS), num_blocks=attn_blocks, **kw)
    other = _MultiLayerPagedAttnEnv(
        num_layers=QWEN_NUM_LAYERS - len(QWEN_ATTN_LAYERS), num_blocks=other_blocks, **kw
    )
    return attn, other


def _compose_engine_mapping(
    attn: _MultiLayerPagedAttnEnv, other: _MultiLayerPagedAttnEnv
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool pointers, the 48-row layer->pool mapping and the per-pool block
    offsets of the hybrid arrangement. Row i is what local_layer_idx=i selects:
    rows 3, 7, ..., 47 are pool 0 at layer-in-pool 0..11, every other row is
    pool 1. Recompose after each refresh_offsets: offsets are concatenated by
    value."""
    pool_pointers = torch.zeros(2, 2, dtype=torch.int64, device="cpu")
    pool_pointers[0] = attn.mgr.kv_cache_pool_pointers[0]
    pool_pointers[1] = other.mgr.kv_cache_pool_pointers[0]
    attn_rows = set(QWEN_ATTN_LAYERS)
    rows, in_attn, in_other = [], 0, 0
    for i in range(QWEN_NUM_LAYERS):
        if i in attn_rows:
            rows.append([0, in_attn])
            in_attn += 1
        else:
            rows.append([1, in_other])
            in_other += 1
    mapping = torch.tensor(rows, dtype=torch.int32, device="cpu")
    offsets = torch.cat([attn.block_offsets, other.block_offsets], dim=0)
    return pool_pointers, mapping, offsets


def _assert_only_this_layer_moved(
    env: _MultiLayerPagedAttnEnv,
    snapshot: List[torch.Tensor],
    layer: int,
    label: str,
) -> None:
    """This layer's slabs moved, and every other layer's slabs are bitwise as
    they were. The complement's size is asserted too, so the sibling loop
    cannot pass by being empty."""
    assert not _bitwise_equal(snapshot[layer], env.layer_views[layer]), (
        f"{label}: the named layer's slabs did not move at all"
    )
    siblings = [m for m in range(env.num_layers) if m != layer]
    assert len(siblings) == env.num_layers - 1 and siblings, f"{label}: empty complement"
    for m in siblings:
        assert _bitwise_equal(snapshot[m], env.layer_views[m]), (
            f"{label}: the call wrote layer {m}'s slabs as well as its own"
        )


def test_bf16_gqa24_2_layer_index_engine_48_row_mapping() -> None:
    """24/2/256 at the twelve local_layer_idx values the target drives.

    Each of rows 3, 7, ..., 47 runs a fresh context chunk, a context chunk
    over its own cached prefix on the paged-context path, and generation at
    P = 1 and 2. Each call is inside `_compare` of its own layer's history,
    so a read landing in a sibling's slabs cannot pass; each append moves its
    own layer's slabs and leaves the other eleven bitwise untouched; and the
    36-layer second pool comes back bitwise as it started.
    """
    torch.manual_seed(530)
    attn, other = _qwen_layer_pools()
    other_before = [v.clone() for v in other.layer_views]
    _, mapping, _ = _compose_engine_mapping(attn, other)
    assert mapping.shape == (QWEN_NUM_LAYERS, 2)
    assert [mapping[row].tolist() for row in QWEN_ATTN_LAYERS] == [
        [0, k] for k in range(len(QWEN_ATTN_LAYERS))
    ], "the twelve driven rows are pool 0, layer-in-pool 0..11"

    attn.add_request(0, 96)  # the manager allocates the whole prompt up front
    attn.add_request(1, 40)
    attn.refresh_offsets([0, 1], num_contexts=2)
    ptrs, mapping, offsets = _compose_engine_mapping(attn, other)
    assert int(ptrs[0, 0]) != int(ptrs[1, 0]), "the two pools must have distinct bases"

    for chunk in ([64, 25], [32, 15]):
        for layer, row in enumerate(QWEN_ATTN_LAYERS):
            qkv = attn.random_qkv(sum(chunk))
            snapshot = [v.clone() for v in attn.layer_views]
            out = attn.call_op(
                layer,
                qkv,
                chunk,
                2,
                [0, 1],
                pool_mapping=mapping,
                pool_pointers=ptrs,
                block_offsets=offsets,
                local_layer_idx=row,
                use_paged_context_fmha=True,
                ctx_lens=chunk,
            )
            _compare(out, _op_reference(attn, qkv, chunk, [0, 1], layer=layer))
            _assert_only_this_layer_moved(
                attn, snapshot, layer, f"context chunk {chunk} at local_layer_idx {row}"
            )
    attn.check_caches([0, 1])

    for p in (1, 2):
        for _ in range(p):
            attn.add_decode_token(0)
            attn.add_decode_token(1)
        attn.refresh_offsets([0, 1], num_contexts=0)
        ptrs, mapping, offsets = _compose_engine_mapping(attn, other)
        for layer, row in enumerate(QWEN_ATTN_LAYERS):
            gen = attn.random_qkv(2 * p)
            snapshot = [v.clone() for v in attn.layer_views]
            out = attn.call_op(
                layer,
                gen,
                [p, p],
                0,
                [0, 1],
                pool_mapping=mapping,
                pool_pointers=ptrs,
                block_offsets=offsets,
                local_layer_idx=row,
                use_paged_context_fmha=True,
                predicted_tokens_per_seq=p,
            )
            _compare(out, _op_reference(attn, gen, [p, p], [0, 1], layer=layer))
            _assert_only_this_layer_moved(
                attn, snapshot, layer, f"P={p} decode at local_layer_idx {row}"
            )
    attn.check_caches([0, 1])

    for before, after in zip(other_before, other.layer_views):
        assert _bitwise_equal(before, after), (
            "a call on a pool-0 row wrote the pool the mapping's other 36 rows name"
        )


def test_bf16_gqa24_2_layer_index_identity_48_layer_pool() -> None:
    """The same question with the mapping's other column moved: a real 48-layer
    KVCacheManager's identity rows take local_layer_idx and the layer-in-pool
    column to 47 together, so the pool-base shift runs 47 * kv_factor slabs
    deep. Rows 0, 3, 23 and 47 over context, cached-prefix context and P = 2
    generation, each inside `_compare` of its own history, with the
    moved/untouched pair over all 48 layers."""
    torch.manual_seed(532)
    env = _MultiLayerPagedAttnEnv(
        num_layers=QWEN_NUM_LAYERS,
        num_heads=QWEN_HQ,
        num_kv_heads=QWEN_HKV,
        head_dim=QWEN_D,
        tokens_per_block=QWEN_TPB,
        num_blocks=24,
        max_seq_len=128,
    )
    assert env.pool_mapping.tolist() == [[0, i] for i in range(QWEN_NUM_LAYERS)], (
        "a real 48-layer manager produces identity mapping rows"
    )
    driven = [0, 3, 23, 47]

    env.add_request(0, 64)
    env.refresh_offsets([0], num_contexts=1)
    for chunk in ([40], [24]):
        for layer in driven:
            qkv = env.random_qkv(sum(chunk))
            snapshot = [v.clone() for v in env.layer_views]
            out = env.call_op(
                layer, qkv, chunk, 1, [0], use_paged_context_fmha=True, ctx_lens=chunk
            )
            _compare(out, _op_reference(env, qkv, chunk, [0], layer=layer))
            _assert_only_this_layer_moved(
                env, snapshot, layer, f"context chunk {chunk} at identity row {layer}"
            )
    env.check_caches([0], layers=driven)

    p = 2
    for _ in range(p):
        env.add_decode_token(0)
    env.refresh_offsets([0], num_contexts=0)
    for layer in driven:
        gen = env.random_qkv(p)
        snapshot = [v.clone() for v in env.layer_views]
        out = env.call_op(
            layer, gen, [p], 0, [0], use_paged_context_fmha=True, predicted_tokens_per_seq=p
        )
        _compare(out, _op_reference(env, gen, [p], [0], layer=layer))
        _assert_only_this_layer_moved(env, snapshot, layer, f"P={p} decode at identity row {layer}")
    env.check_caches([0], layers=driven)


def test_bf16_gqa24_2_layer_index_ab_single_layer_vs_row_47() -> None:
    """A/B: the same call, once in a single-layer pool and once at row 47.

    Identical operands and identical layer content either side, and every
    other op argument equal (max_seq_len, max_context_length,
    attention_window_size, max_num_requests, tokens_per_block), so the only
    variable is where the call's slice lives -- a one-layer pool at base, or
    layer-in-pool 11 of a twelve-layer pool selected by mapping row 47 of 48.
    The outputs are bitwise equal on both the paged-context prefill and the
    decode that reads the history back.

    The control is the degeneracy itself: the same decode sent to row 43
    (layer-in-pool 10), whose slabs hold no history, must not come back equal.
    """
    torch.manual_seed(531)
    single = _qwen_bf16_env()
    single.add_request(0, 64)
    qkv = single.random_qkv(64)
    out_single = _paged_ctx_call(single, qkv, [64], 1, [0])
    qkv_d = single.random_qkv(1)
    out_single_d = _paged_ctx_call(single, qkv_d, [1], 0, [0])
    single.check_cache(0)

    attn, other = _qwen_layer_pools(attn_blocks=64, other_blocks=64, max_seq_len=512)
    assert attn.max_seq_len == single.max_seq_len
    deep_layer, deep_row = len(QWEN_ATTN_LAYERS) - 1, QWEN_ATTN_LAYERS[-1]
    assert deep_row == 47
    attn.add_request(0, 64)
    attn.refresh_offsets([0], num_contexts=1)
    ptrs, mapping, offsets = _compose_engine_mapping(attn, other)
    tables = dict(pool_mapping=mapping, pool_pointers=ptrs, block_offsets=offsets)
    out_multi = attn.call_op(
        deep_layer,
        qkv,
        [64],
        1,
        [0],
        local_layer_idx=deep_row,
        use_paged_context_fmha=True,
        ctx_lens=[64],
        **tables,
    )
    attn.add_decode_token(0)
    attn.refresh_offsets([0], num_contexts=0)
    ptrs, mapping, offsets = _compose_engine_mapping(attn, other)
    tables = dict(pool_mapping=mapping, pool_pointers=ptrs, block_offsets=offsets)
    out_multi_d = attn.call_op(
        deep_layer,
        qkv_d,
        [1],
        0,
        [0],
        local_layer_idx=deep_row,
        use_paged_context_fmha=True,
        **tables,
    )
    attn.check_caches([0], layers=[deep_layer])

    for what, a, b in (
        ("paged-context prefill", out_single, out_multi),
        ("decode", out_single_d, out_multi_d),
    ):
        assert _bitwise_equal(a, b), (
            f"{what}: a single-layer pool and layer-in-pool 11 of a twelve-layer "
            f"pool at mapping row 47 disagree (max|delta| "
            f"{(a.float() - b.float()).abs().max().item():.6g})"
        )

    wrong_row = QWEN_ATTN_LAYERS[-2]
    wrong = attn.call_op(
        deep_layer - 1,
        qkv_d,
        [1],
        0,
        [0],
        local_layer_idx=wrong_row,
        use_paged_context_fmha=True,
        record=False,
        **tables,
    )
    assert not _bitwise_equal(wrong, out_single_d), (
        f"a decode routed to mapping row {wrong_row}, whose slabs hold no history, "
        f"is bitwise identical to the correct one: this A/B cannot see a wrong layer index"
    )


# ─── Encoder-only: the op with no KV cache at all ──────────────────────
#
# A vision tower has no KV cache: one bidirectional pass over each frame's
# patch tokens. The op is told so by receiving no cache addressing at all --
# kv_cache_block_offsets, host_kv_cache_pool_pointers and
# host_kv_cache_pool_mapping all None -- with update_kv_cache still True.
# Frame segmentation is the batch: one context sequence per frame, and
# mask_type 0 makes each frame's tokens attend to exactly their own frame.


def _encoder_call(
    qkv: torch.Tensor,
    seg_lens: List[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask_type: int,
    workspace: torch.Tensor,
    **overrides,
) -> torch.Tensor:
    """One context-phase call with every KV-cache argument absent, batch state
    as the vision tower builds it (`max_num_requests` = the segment count)."""
    ns = len(seg_lens)
    total = sum(seg_lens)
    output = torch.empty(total, num_heads * head_dim, dtype=qkv.dtype, device="cuda")
    op_kwargs = dict(
        q=qkv,
        k=None,
        v=None,
        output=output,
        output_sf=None,
        workspace=workspace,
        sequence_length=torch.tensor(seg_lens, dtype=torch.int32, device="cuda"),
        host_past_key_value_lengths=torch.tensor(seg_lens, dtype=torch.int32),
        host_total_kv_lens=torch.tensor([total, 0], dtype=torch.int32),
        context_lengths=torch.tensor(seg_lens, dtype=torch.int32, device="cuda"),
        host_context_lengths=torch.tensor(seg_lens, dtype=torch.int32),
        host_request_types=torch.tensor([0] * ns, dtype=torch.int32),
        max_context_q_len_override=None,
        # The three whose absence selects the no-KV-cache mode.
        kv_cache_block_offsets=None,
        host_kv_cache_pool_pointers=None,
        host_kv_cache_pool_mapping=None,
        cache_indirection=None,
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
        out_scale=None,
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        latent_cache=None,
        q_pe=None,
        block_ids_per_seq=None,
        attention_sinks=None,
        is_fused_qkv=True,
        update_kv_cache=True,
        predicted_tokens_per_seq=1,
        local_layer_idx=0,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        tokens_per_block=None,
        max_num_requests=ns,
        max_context_length=max(seg_lens),
        max_seq_len=max(seg_lens),
        attention_window_size=max(seg_lens),
        beam_width=1,
        mask_type=mask_type,
        quant_mode=0,
        q_scaling=1.0,
        position_embedding_type=0,
        rope_dim=0,
        rope_base=10000.0,
        rope_scale_type=0,
        rope_scale=1.0,
        rope_short_m_scale=1.0,
        rope_long_m_scale=1.0,
        rope_max_positions=1024,
        rope_original_max_positions=1024,
        use_paged_context_fmha=False,
        attention_input_type=0,
        is_mla_enable=False,
        chunked_prefill_buffer_batch_size=1,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_nope_head_dim=None,
        qk_rope_head_dim=None,
        v_head_dim=None,
        rope_append=None,
        mrope_rotary_cos_sin=None,
        mrope_position_deltas=None,
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        attention_chunk_size=None,
        softmax_stats_tensor=None,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets=None,
        spec_decoding_packed_mask=None,
        spec_decoding_bl_tree_mask_offset=None,
        spec_decoding_bl_tree_mask=None,
        spec_bl_tree_first_sparse_mask_offset_kv=None,
        sparse_kv_indices=None,
        sparse_kv_offsets=None,
        sparse_attn_indices=None,
        sparse_attn_offsets=None,
        sparse_attn_indices_block_size=0,
        num_contexts=ns,
        num_ctx_tokens=total,
    )
    op_kwargs.update(overrides)
    with validating(op):
        op(**op_kwargs)
    torch.cuda.synchronize()
    return output


# ─── The entry's own cell list ──────────────────────────────────────────


def _seed(cell) -> None:
    """A cell's own seed, stable across processes (str hash is not)."""
    torch.manual_seed(zlib.crc32(cell.why.encode()))


#: Rows one uncertified history call carries: the engine's max_num_tokens.
_FILL_TOKENS = 8192


def _distance(out: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
    """`_compare`'s two statistics in bf16 ulp: the largest element against its
    row's scale, and the relative RMS."""
    flat_out = out.float().reshape(-1, out.shape[-1])
    flat_ref = ref.float().reshape(-1, ref.shape[-1])
    row_scale = flat_ref.abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
    elt = ((flat_out - flat_ref).abs() / row_scale).max().item() / ULP[torch.bfloat16]
    rms = (flat_out - flat_ref).pow(2).mean().sqrt() / flat_ref.pow(2).mean().sqrt()
    return elt, rms.item() / ULP[torch.bfloat16]


def _report(label: str, out: torch.Tensor, ref: torch.Tensor, extra: str = "") -> None:
    elt, rms = _distance(out, ref)
    print(f"  {label[:72]}: {elt:.2f} ulp element-wise, {rms:.3f} ulp RMS {extra}".rstrip())


class _PagedRun(NamedTuple):
    env: _PagedAttnEnv
    qkv: torch.Tensor
    out: torch.Tensor
    new: List[int]
    rids: List[int]
    e4m3: bool
    nvfp4: bool
    causal: bool


def _run_paged(spec) -> _PagedRun:
    """A standard-configuration call over a single-layer paged pool, from a spec.

    Uncertified context calls fill each sequence's `cached` tokens, at most
    `_FILL_TOKENS` rows a call as the engine's max_num_tokens chunks a
    prefill; the certified call then appends `new` rows per sequence (P for a
    generation row). Over an fp8 or NVFP4 pool every operand that call reads
    went through e4m3, except on the packed context path, which never reads
    the pool. The block table is as wide as the longest sequence needs (at
    least 16 pages), and `max_seq_len` with it.
    """
    fp8 = bool(spec["quant_mode"] & QUANT_MODE_FP8_KV_CACHE)
    nvfp4 = bool(spec["quant_mode"] & QUANT_MODE_NVFP4_KV_CACHE)
    cached, new, nc = spec["cached"], spec["new"], spec["num_contexts"]
    tpb = spec["tokens_per_block"]
    longest = max(c + n for c, n in zip(cached, new))
    env = _PagedAttnEnv(
        num_heads=spec["num_heads"],
        num_kv_heads=spec["num_kv_heads"],
        head_dim=spec["head_size"],
        tokens_per_block=tpb,
        num_blocks=spec["num_blocks"],
        max_batch=spec["max_num_requests"],
        max_blocks_per_seq=max(16, -(-longest // tpb)),
        pool_dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        quant_mode=spec["quant_mode"],
        kv_scaling_factor=spec["kv_scale"] if fp8 or nvfp4 else None,
    )
    p, paged = spec["predicted_tokens_per_seq"], spec["use_paged_context_fmha"]
    assert all(n == p for n in new[nc:]), "a generation row carries P q rows"
    assert paged or not any(cached[:nc]), "a cached prefix needs the paged-context path"
    rids = list(range(len(new)))
    for rid in rids:
        env.add_request(rid, cached[rid] + (new[rid] if rid < nc else 0))
    groups: List[List[int]] = []
    for rid in (rid for rid in rids if cached[rid]):
        if not groups or sum(cached[r] for r in groups[-1]) + cached[rid] > _FILL_TOKENS:
            groups.append([])
        groups[-1].append(rid)
    for prior in groups:
        hist = [cached[rid] for rid in prior]
        env.call_op(
            env.random_qkv(sum(hist)),
            hist,
            len(prior),
            prior,
            MASK_CAUSAL,
            ctx_lens_override=hist,
            use_paged_context_fmha=paged,
        )
    qkv = env.random_qkv(sum(new))
    out = env.call_op(
        qkv,
        list(new),
        nc,
        rids,
        spec["mask_type"],
        ctx_lens_override=[new[i] if i < nc else env.prompt_lens[i] for i in rids],
        use_paged_context_fmha=paged,
        predicted_tokens_per_seq=p,
    )
    # One e4m3 flag covers the whole call: the packed path's context rows read
    # bf16, its generation rows e4m3, so a cell never mixes the two there.
    assert not (fp8 and not paged and 0 < nc < len(rids))
    assert not (nvfp4 and not paged), "an NVFP4 cell runs the paged-context path"
    e4m3 = (fp8 or nvfp4) and (paged or nc == 0)
    causal = spec["mask_type"] == MASK_CAUSAL
    return _PagedRun(env, qkv, out, list(new), rids, e4m3, nvfp4, causal)


def _cell_paged(cell, arm) -> None:
    """A standard-configuration cell over a single-layer paged pool, gated
    against `op.reference` over the history the call should have attended --
    over e4m3 operands at the closest of `_E4M3_PARTITIONS` -- with the pool
    read back bitwise."""
    arm(op)
    _seed(cell)
    run = _run_paged(cell.spec)
    env, qkv, out = run.env, run.qkv, run.out

    def reference_at(partition):
        return _op_reference(
            env,
            qkv,
            run.new,
            run.rids,
            causal=run.causal,
            e4m3=run.e4m3,
            nvfp4=run.nvfp4,
            partition=partition,
        )

    ref, label = _closest_partition(out, reference_at) if run.e4m3 else (reference_at(None), "")
    _report(cell.why, out, ref, label)
    _compare(out, ref)
    for rid in run.rids:
        env.check_cache(rid)
    env.check_unwritten_pool_zero()


def _cell_layered(cell, arm) -> None:
    """A qwen4_exp cell at one row of a 48-row mapping.

    `hybrid48` is the engine's arrangement (rows 3, 7, ..., 47 on pool 0 at
    layer-in-pool 0..11, the other 36 rows on pool 1); `identity48` a real
    48-layer manager's single pool. The history goes in through the same row
    by an uncertified context call over the sequences that have one. Beside
    `_compare`: the call moved its own layer's slabs and no sibling's, and the
    second pool stayed untouched.
    """
    spec = cell.spec
    arm(op)
    _seed(cell)
    row, p = spec["local_layer_idx"], spec["predicted_tokens_per_seq"]
    cached, new, nc = spec["cached"], spec["new"], spec["num_contexts"]
    assert all(cached[nc:]), "a generation row needs a history"
    heads = dict(num_heads=spec["num_heads"], num_kv_heads=spec["num_kv_heads"])
    fp8 = bool(spec["quant_mode"] & QUANT_MODE_FP8_KV_CACHE)
    pool_kw = (
        dict(dtype=DataType.FP8, quant_mode=spec["quant_mode"], kv_scaling_factor=spec["kv_scale"])
        if fp8
        else {}
    )
    if spec["pool_layout"] == "hybrid48":
        env, other = _qwen_layer_pools(max_batch=spec["max_num_requests"], **heads, **pool_kw)
        layer = QWEN_ATTN_LAYERS.index(row)
    else:
        env = _MultiLayerPagedAttnEnv(
            num_layers=QWEN_NUM_LAYERS,
            head_dim=spec["head_size"],
            tokens_per_block=spec["tokens_per_block"],
            num_blocks=24,
            max_batch=spec["max_num_requests"],
            max_seq_len=128,
            **heads,
            **pool_kw,
        )
        other, layer = None, row

    def tables() -> Dict:
        if other is None:
            return {}
        ptrs, mapping, offsets = _compose_engine_mapping(env, other)
        return dict(pool_pointers=ptrs, pool_mapping=mapping, block_offsets=offsets)

    other_before = [] if other is None else [v.clone() for v in other.layer_views]
    rids = list(range(len(new)))
    for rid in rids:
        env.add_request(rid, cached[rid] + (new[rid] if rid < nc else 0))
    prior = [rid for rid in rids if cached[rid]]
    hist = [cached[rid] for rid in prior]
    env.refresh_offsets(prior, num_contexts=len(prior))
    env.call_op(
        layer,
        env.random_qkv(sum(hist)),
        hist,
        len(prior),
        prior,
        local_layer_idx=row,
        use_paged_context_fmha=True,
        ctx_lens=hist,
        **tables(),
    )
    for rid in rids[nc:]:
        for _ in range(p):
            env.add_decode_token(rid)
    env.refresh_offsets(rids, num_contexts=nc)
    qkv = env.random_qkv(sum(new))
    snapshot = [v.clone() for v in env.layer_views]
    out = env.call_op(
        layer,
        qkv,
        list(new),
        nc,
        rids,
        local_layer_idx=row,
        use_paged_context_fmha=True,
        ctx_lens=[new[i] if i < nc else env.prompt_lens[i] for i in rids],
        predicted_tokens_per_seq=p,
        **tables(),
    )
    if fp8:
        ref, label = _closest_partition(
            out,
            lambda partition: _op_reference(
                env, qkv, list(new), rids, layer=layer, e4m3=True, partition=partition
            ),
        )
    else:
        ref, label = _op_reference(env, qkv, list(new), rids, layer=layer), ""
    _report(cell.why, out, ref, label)
    _compare(out, ref)
    env.check_caches(rids, layers=[layer])
    _assert_only_this_layer_moved(env, snapshot, layer, cell.why)
    for before, after in zip(other_before, [] if other is None else other.layer_views):
        assert _bitwise_equal(before, after), "a pool-0 row wrote the second pool"


def _cell_encoder(cell, arm) -> None:
    """The no-KV-cache mode: one context sequence per frame, no cache
    addressing, gated against `op.reference` over each frame's own rows."""
    spec = cell.spec
    arm(op)
    _seed(cell)
    seg_lens = list(spec["seg_lens"])
    h, hkv, d = spec["num_heads"], spec["num_kv_heads"], spec["head_size"]
    qkv = torch.randn(sum(seg_lens), (h + 2 * hkv) * d, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(0, dtype=torch.int8, device="cuda")
    out = _encoder_call(qkv, seg_lens, h, hkv, d, spec["mask_type"], workspace)
    causal = spec["mask_type"] == MASK_CAUSAL
    _compare(out, _packed_reference(qkv, seg_lens, h, hkv, d, causal=causal))
    del qkv, out
    torch.cuda.empty_cache()


#: The qwen4_exp full-attention call over an fp8-e4m3 pool at the production kv
#: scale 1.0 (fp32 [1] scale tensors, quant_mode 128): what the trunk layers and
#: the MTP draft layer pass when the engine serves an fp8 KV cache. Same chosen
#: pool sizes as `QWEN4_DENSE`.
_QWEN4_FP8 = dict(QWEN4_DENSE, quant_mode=QM_FP8_KV_CACHE, kv_scale=1.0)
#: The same call over an NVFP4 pool: e2m1 data and a second pool of e4m3 block
#: scales, fp32 [3] unit kv scales, `[1, 2, 2]` pool pointers, bf16 output --
#: what the trunk layers and the MTP draft layer pass when the engine serves an
#: NVFP4 KV cache. Same chosen pool sizes as `QWEN4_DENSE`.
_QWEN4_NVFP4 = dict(QWEN4_DENSE, quant_mode=QM_NVFP4_KV_CACHE, kv_scale=1.0)
#: `QWEN4_DENSE` and `_QWEN4_FP8` for a verify step of up to 32 requests, the captured-batch
#: ceiling: one block-offset row per request and pages for 32 histories (the largest cell
#: needs 133 of the 144).
_QWEN4_DENSE_G32 = dict(QWEN4_DENSE, max_num_requests=32, num_blocks=144)
_QWEN4_FP8_G32 = dict(_QWEN4_FP8, max_num_requests=32, num_blocks=144)


def _spread(batch: int, longest: int) -> tuple[int, ...]:
    """`batch` histories rising evenly from 1 to `longest` tokens: one-page, page-boundary
    and multi-page sequences in one verify step."""
    return tuple(1 + (longest - 1) * i // (batch - 1) for i in range(batch))


#: The full-attention layer as one rank of a pure attention-TP member holds it: the
#: 24 q / 2 kv heads split two ways (tp2, tep2) and four ways (tp4, tep4), each rank
#: keeping one kv head. Same pool sizes as `QWEN4_DENSE`.
_RANK_HALF = dict(QWEN4_DENSE, num_heads=12, num_kv_heads=1)
_RANK_QUARTER = dict(QWEN4_DENSE, num_heads=6, num_kv_heads=1)


def _batch(base: dict, requests: int, longest: int, p: int) -> dict:
    """A generation step of `requests` sequences, one block-offset row each, with
    histories `_spread` from 1 to `longest` tokens and P = `p` q rows apiece, over a
    pool of exactly the pages they reach plus 32 never written."""
    cached = _spread(requests, longest)
    pages = sum(-(-(c + p) // base["tokens_per_block"]) for c in cached)
    return dict(
        base,
        max_num_requests=requests,
        num_blocks=pages + 32,
        cached=cached,
        new=(p,) * requests,
        num_contexts=0,
        predicted_tokens_per_seq=p,
    )


def _prefill_chunk(base: dict) -> dict:
    """max_num_tokens of fresh prefill: four 2048-token contexts, 8192 q rows."""
    return dict(
        base,
        num_blocks=4 * 2048 // base["tokens_per_block"] + 32,
        cached=(0,) * 4,
        new=(2048,) * 4,
        num_contexts=4,
        predicted_tokens_per_seq=1,
    )


# The shapes this target calls the entry with. The first block is the qwen4_exp
# full-attention cell at the checkpoint's 24 q / 2 kv heads as the catalog test
# certifies it for the family -- the calls this target's attention-DP members and
# tp1 make at a few requests; the second is what its members add.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified": (
        Cell(
            why="qwen4_exp prefill: GQA 24q/2kv d256, bf16 pool, paged context; "
            "a fresh 70-row sequence beside 33 rows over a 48-token cached prefix",
            spec=dict(
                QWEN4_DENSE,
                cached=(0, 48),
                new=(70, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp decode P=1: one row per generation sequence over "
            "page-crossing histories (70, 35)",
            spec=dict(
                QWEN4_DENSE,
                cached=(70, 35),
                new=(1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp MTP verify P=2 (max_draft_len 1): draft row t attends "
            "[0, L-P+t], bottom-right causal within the block",
            spec=dict(
                QWEN4_DENSE,
                cached=(70, 35),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp MTP verify P=3 (max_draft_len 2): a separate decode "
            "compile, P reaches the kernel as maxSeqLenQ",
            spec=dict(
                QWEN4_DENSE,
                cached=(70, 35),
                new=(3, 3),
                num_contexts=0,
                predicted_tokens_per_seq=3,
            ),
        ),
        Cell(
            why="qwen4_exp MTP verify P=4 (max_draft_len 3)",
            spec=dict(
                QWEN4_DENSE,
                cached=(70, 35),
                new=(4, 4),
                num_contexts=0,
                predicted_tokens_per_seq=4,
            ),
        ),
        *(
            Cell(
                why=f"qwen4_exp MTP verify P={p} (max_draft_len {p - 1})"
                + (": the target's ceiling" if p == 8 else ""),
                spec=dict(
                    QWEN4_DENSE,
                    cached=(70, 35),
                    new=(p, p),
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p in (5, 6, 7, 8)
        ),
        Cell(
            why="qwen4_exp P=8 G=1: one draft block from row 95, across the page edge at 96",
            spec=dict(
                QWEN4_DENSE,
                cached=(95,),
                new=(8,),
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        # The trtllm-gen launch picks its generation kernel by batch size at P 5..8, so each P
        # is also driven at 32 requests and at one batch in 17..20, with histories of every
        # page shape in the step.
        *(
            Cell(
                why=f"qwen4_exp bf16 batch P={p} G=32: 32 blocks over histories 1..218, "
                f"{32 * p} q rows",
                spec=dict(
                    _QWEN4_DENSE_G32,
                    cached=_spread(32, 218),
                    new=(p,) * 32,
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p in (5, 6, 7, 8)
        ),
        *(
            Cell(
                why=f"qwen4_exp bf16 batch P={p} G={g}: {g} blocks over histories 200..1, "
                f"{g * p} q rows",
                spec=dict(
                    _QWEN4_DENSE_G32,
                    cached=tuple(reversed(_spread(g, 200))),
                    new=(p,) * g,
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p, g in ((5, 17), (6, 18), (7, 19), (8, 20))
        ),
        Cell(
            why="qwen4_exp mixed step: a fresh 20-row context ahead of two P=2 "
            "draft blocks, generation rows token-major after the context rows",
            spec=dict(
                QWEN4_DENSE,
                cached=(0, 70, 35),
                new=(20, 2, 2),
                num_contexts=1,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp mixed step P=8: a fresh 20-row context ahead of two P=8 "
            "draft blocks over histories 70 and 35",
            spec=dict(
                QWEN4_DENSE,
                cached=(0, 70, 35),
                new=(20, 8, 8),
                num_contexts=1,
                predicted_tokens_per_seq=8,
            ),
        ),
        Cell(
            why="qwen4_exp hybrid pool: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11), P=2 verify",
            spec=dict(
                QWEN4_DENSE,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp P=8 hybrid pool: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11), P=8 verify",
            spec=dict(
                QWEN4_DENSE,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(8, 8),
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        # qwen4_exp's own geometry over an fp8 pool: the kv head count changes the compiled
        # decode kernel, so the 24/4/256 cells above do not cover it.
        Cell(
            why="qwen4_exp fp8 pool paged-context prefill, 24/2/256: a fresh 48-row "
            "sequence beside 33 rows over a 48-token e4m3 prefix",
            spec=dict(
                _QWEN4_FP8,
                cached=(0, 48),
                new=(48, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 pool decode P=1, 24/2/256: histories 200 and 48 span "
            "pages and KV tiles",
            spec=dict(
                _QWEN4_FP8,
                cached=(200, 48),
                new=(1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        *(
            Cell(
                why=f"qwen4_exp fp8 pool MTP verify P={p}, 24/2/256 (max_draft_len {p - 1})",
                spec=dict(
                    _QWEN4_FP8,
                    cached=(70, 35),
                    new=(p, p),
                    num_contexts=0,
                    predicted_tokens_per_seq=p,
                ),
            )
            for p in (2, 4, 5, 6, 7, 8)
        ),
        Cell(
            why="qwen4_exp fp8 pool mixed step P=3, 24/2/256: a 20-row context ahead "
            "of two P=3 blocks over histories 31 and 64",
            spec=dict(
                _QWEN4_FP8,
                cached=(0, 31, 64),
                new=(20, 3, 3),
                num_contexts=1,
                predicted_tokens_per_seq=3,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 pool mixed step P=8, 24/2/256: a 20-row context ahead "
            "of two P=8 blocks over histories 31 and 64",
            spec=dict(
                _QWEN4_FP8,
                cached=(0, 31, 64),
                new=(20, 8, 8),
                num_contexts=1,
                predicted_tokens_per_seq=8,
            ),
        ),
        # KV stays within one 128-key tile (histories to 119 plus 8 rows), where the e4m3
        # probability model of `reference` is the kernel's own at any batch.
        Cell(
            why="qwen4_exp fp8 batch P=8 G=32: 32 blocks over histories 1..119, 256 q rows",
            spec=dict(
                _QWEN4_FP8_G32,
                cached=_spread(32, 119),
                new=(8,) * 32,
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 hybrid pool P=2: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11) over e4m3 slabs",
            spec=dict(
                _QWEN4_FP8,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp fp8 hybrid pool P=8: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11) over e4m3 slabs",
            spec=dict(
                _QWEN4_FP8,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(8, 8),
                num_contexts=0,
                predicted_tokens_per_seq=8,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool paged-context prefill, 24/2/256: a fresh 48-row "
            "sequence beside 33 rows over a 48-token e2m1 prefix",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(0, 48),
                new=(48, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool decode P=1, 24/2/256: histories 200 and 48",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(200, 48),
                new=(1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool MTP verify P=2, 24/2/256 (max_draft_len 1)",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(70, 35),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool MTP verify P=4, 24/2/256 (max_draft_len 3)",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(70, 35),
                new=(4, 4),
                num_contexts=0,
                predicted_tokens_per_seq=4,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool mixed step P=1: a 20-row context ahead of two "
            "1-row sequences over histories 31 and 64",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(0, 31, 64),
                new=(20, 1, 1),
                num_contexts=1,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp nvfp4 pool mixed step P=4: a 20-row context ahead of two 4-row blocks",
            spec=dict(
                _QWEN4_NVFP4,
                cached=(0, 31, 64),
                new=(20, 4, 4),
                num_contexts=1,
                predicted_tokens_per_seq=4,
            ),
        ),
        # tp2 / tep2: 12 q / 1 kv heads per rank.
        Cell(
            why="tp2 prefill 12q/1kv at row 47 of the hybrid mapping: a fresh 70-row "
            "sequence beside 33 rows over a 48-token prefix",
            spec=dict(
                _RANK_HALF,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(0, 48),
                new=(70, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="tp2 decode 12q/1kv at the dense ceiling: one row over a 2047-token history",
            spec=dict(
                _RANK_HALF,
                cached=(2047,),
                new=(1,),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="tp2 decode 12q/1kv at 128 requests over histories 1..2047",
            spec=_batch(_RANK_HALF, 128, 2047, 1),
        ),
        Cell(
            why="tp2 mixed step 12q/1kv: a fresh 20-row context ahead of two decode rows "
            "over histories 70 and 35",
            spec=dict(
                _RANK_HALF,
                cached=(0, 70, 35),
                new=(20, 1, 1),
                num_contexts=1,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="tp2 prefill chunk 12q/1kv: four 2048-token contexts, 8192 rows",
            spec=_prefill_chunk(_RANK_HALF),
        ),
        # tp4 / tep4: 6 q / 1 kv heads per rank, drafting with MTP depth 1..7.
        Cell(
            why="tp4 prefill 6q/1kv at row 47 of the hybrid mapping: a fresh 70-row "
            "sequence beside 33 rows over a 48-token prefix",
            spec=dict(
                _RANK_QUARTER,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(0, 48),
                new=(70, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="tp4 decode 6q/1kv at 128 requests over histories 1..2047",
            spec=_batch(_RANK_QUARTER, 128, 2047, 1),
        ),
        Cell(
            why="tep4 MTP verify P=2 6q/1kv: 32 blocks over histories 1..218",
            spec=_batch(_RANK_QUARTER, 32, 218, 2),
        ),
        Cell(
            why="tep4 MTP3 verify P=4 6q/1kv: 128 blocks over histories 1..2040, 512 rows",
            spec=_batch(_RANK_QUARTER, 128, 2040, 4),
        ),
        Cell(
            why="tep4 MTP7 verify P=8 6q/1kv: 128 blocks over histories 1..2040, 1024 rows",
            spec=_batch(_RANK_QUARTER, 128, 2040, 8),
        ),
        Cell(
            why="tep4 mixed step P=4 6q/1kv: a fresh 20-row context ahead of two 4-row "
            "blocks over histories 70 and 35",
            spec=dict(
                _RANK_QUARTER,
                cached=(0, 70, 35),
                new=(20, 4, 4),
                num_contexts=1,
                predicted_tokens_per_seq=4,
            ),
        ),
        Cell(
            why="tep4 draft layer P=4 6q/1kv over the separate draft KV cache (the MTP "
            "layer alone): one block from row 95, across the page edge at 96",
            spec=dict(
                _RANK_QUARTER,
                cached=(95,),
                new=(4,),
                num_contexts=0,
                predicted_tokens_per_seq=4,
            ),
        ),
        Cell(
            why="tp4 prefill chunk 6q/1kv: four 2048-token contexts, 8192 rows",
            spec=_prefill_chunk(_RANK_QUARTER),
        ),
        # tp1 / dep2 / dep4: every head on the rank, 128 requests per rank.
        Cell(
            why="dep4 decode 24q/2kv at 128 requests per rank over histories 1..2047",
            spec=_batch(QWEN4_DENSE, 128, 2047, 1),
        ),
        Cell(
            why="dep4 MTP3 verify P=4 24q/2kv: 128 blocks over histories 1..2040, 512 rows",
            spec=_batch(QWEN4_DENSE, 128, 2040, 4),
        ),
        Cell(
            why="dep4 fp8 pool decode 24q/2kv at 128 requests over histories 1..2047",
            spec=_batch(_QWEN4_FP8, 128, 2047, 1),
        ),
        Cell(
            why="dep4 nvfp4 pool decode 24q/2kv at 128 requests over histories 1..2047",
            spec=_batch(_QWEN4_NVFP4, 128, 2047, 1),
        ),
        Cell(
            why="tp1 prefill chunk 24q/2kv: four 2048-token contexts, 8192 rows",
            spec=_prefill_chunk(QWEN4_DENSE),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    """Drive every configuration the entry and this target claim, each against
    the entry's own `reference`, with `certifying` armed: the run is also proof
    that the entry's own `__call__` and `reference` were the ones exercised."""
    spec = cell.spec
    if spec.get("kv_cache") == "none":
        _cell_encoder(cell, certifying)
    elif "pool_layout" in spec:
        _cell_layered(cell, certifying)
    else:
        _cell_paged(cell, certifying)


# ─── A quantized-pool decode past one KV tile: the partition gate's rivals ──
#
# Gating against the closest of `_E4M3_PARTITIONS` admits only rounding: every
# member is the same attention. What it must still reject is a call that read
# something else -- K/V the pool did not hold, unrounded probabilities, or a
# different history -- and these are measured at the longest decode the target
# serves over each quantized pool.

_QUANTIZED_DECODES = [
    ("fp8 pool, 128 requests over 1..2047", _batch(_QWEN4_FP8, 128, 2047, 1)),
    ("nvfp4 pool, 128 requests over 1..2047", _batch(_QWEN4_NVFP4, 128, 2047, 1)),
]


def _newest_key_repeated(env, rids: List[int]):
    """The env's histories with each sequence's newest K/V row replaced by the
    row before it: what a call attending one position short computes."""

    def shifted(history):
        rows = torch.cat(history).clone()
        rows[-1] = rows[-2]
        return [rows]

    return SimpleNamespace(
        num_heads=env.num_heads,
        num_kv_heads=env.num_kv_heads,
        head_dim=env.head_dim,
        kv_scale_orig_quant=env.kv_scale_orig_quant,
        k_history={rid: shifted(env.k_history[rid]) for rid in rids},
        v_history={rid: shifted(env.v_history[rid]) for rid in rids},
    )


@pytest.mark.parametrize("name, spec", _QUANTIZED_DECODES, ids=[c[0] for c in _QUANTIZED_DECODES])
def test_quantized_decode_partition_gate_rejects_its_rivals(name, spec) -> None:
    """The call inside `_compare` of its closest partition, and each rival outside
    `_compare` of the call: K and V read as the bf16 rows the call appended rather
    than as the pool holds them, the softmax probabilities left in fp32, and every
    sequence's newest key replaced by the one before it. The distances are printed;
    the fp32-probability rival is the closest, the e4m3 probability rounding being
    the whole of what it leaves out."""
    torch.manual_seed(zlib.crc32(name.encode()))
    run = _run_paged(spec)
    env, qkv, out = run.env, run.qkv, run.out
    assert run.e4m3 and run.causal
    best, label = _closest_partition(
        out,
        lambda partition: _op_reference(
            env, qkv, run.new, run.rids, e4m3=True, nvfp4=run.nvfp4, partition=partition
        ),
    )
    _report(name, out, best, label)
    _compare(out, best)
    rivals = {
        "K/V read unquantized": _op_reference(env, qkv, run.new, run.rids),
        "probabilities left in fp32": _op_reference(
            env, qkv, run.new, run.rids, e4m3=True, nvfp4=run.nvfp4, e4m3_probabilities=False
        ),
        "newest key repeated": _op_reference(
            _newest_key_repeated(env, run.rids),
            qkv,
            run.new,
            run.rids,
            e4m3=True,
            nvfp4=run.nvfp4,
        ),
    }
    for what, rival in rivals.items():
        _report(f"  rival: {what}", out, rival)
        with pytest.raises(AssertionError):
            _compare(out, rival)
