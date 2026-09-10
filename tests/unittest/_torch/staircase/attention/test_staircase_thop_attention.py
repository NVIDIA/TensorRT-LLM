# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the thop_attention catalog entry.

The binding is stateless on the Python side: every piece of state it reads
is an explicit argument. The test builds that state for real — a caller-
owned paged KV pool, pool pointer/mapping tensors, block-offset tables
(hand-filled for single-layer pools, produced by a real KVCacheManager for
the multi-layer case), and the int32 host/device length tensors — and
compares against fp32 torch references. The certified configurations
exercised:

1. Standard GQA/MHA: packed-QKV self-attention over a
   [num_blocks, 2, Hkv, tokens_per_block, D] bf16 pool (kv_factor 2), with
   the paged-cache append additionally checked bit-exactly. Head geometry is
   swept across the two shipped-target geometries (32q/8kv and 32q/4kv,
   d128) plus MQA, a non-power-of-2 GQA ratio, non-power-of-2 head counts,
   and head_size 64/128/256; a non-multiple (Hq, Hkv) pair is a negative
   test, since the op computes it silently wrong.
2. MLA (is_mla_enable=True): context prefill (separate q/k/v, in-kernel
   GPT-J RoPE, latent-cache append) and generation decode (latent MQA over
   the paged cache) over a [num_blocks, tokens_per_block, C+R] bf16 latent
   pool (kv_factor 1), split into context_only / generation_only calls.
   Every MLA case except the chunked one (3b) runs at three query-head
   counts: 32 (the deepseek-v3-lite tp1 layer shape), 16 and 8 (its tep4
   slice), same C/R/nope/v. All three MLA call flavors additionally run at
   latent-pool page size 32 (the engine default; 64 is the tuned value) on
   page-32-specific geometry, at 32 and 8 query heads — and at 16 too for
   the decode flavor, the one where the page size and the head count both
   reach the compiled kernel. Each flavor is finally
   swept over q_lora_rank (1536 = DeepSeek-V3, 0 = a checkpoint with no
   q-LoRA, 1536 again as a determinism control, 4096 as an out-of-range
   probe) on identical inputs, asserting every observable bitwise equal —
   the argument turns out never to be read on the MLA paths; passing None
   for it is a negative test. That sweep runs at both shipped head counts,
   32 (tp1) and 8 (tep4), because the two compile different decode kernels.
2b. The DeepSeek-R1-0528 cell, at page 32: all four MLA call flavors at
   128 query heads (128/128/192 context, 128/1/576 decode — attention DP
   replicates the head count whole, so a dep4 rank runs all of them), first
   on the baseline rope/scale to isolate the head count and then at the two
   further values R1 moves: a YaRN-scaled rotary_cos_sin table and
   q_scaling = 1/mscale^2 != 1. Both of those are also driven as axes of
   their own — q_scaling swept over four values in both phases with every
   run gated outside the other three's references, and the rope table pinned
   as the *only* rope input the op reads (a plain-torch rebuild of the YaRN
   formula drives the op; the seven scalar rope arguments and
   rotary_inv_freq are bitwise inert beside it, with a table swap as the
   control that the comparison can see a rope change at all).
2c. The DeepSeek-R1-0528 layer geometry over an **fp8-e4m3 latent pool**
   (quant_mode 128, H = 128, page 32, one C+R-byte row per token): fresh
   prefill and generation decode on the baseline rope/scale, at the
   production kv scale s = 1.0 and swept over 1.5 and 2.0. Three things the
   standard configuration's fp8
   section does not carry over, all measured here: the context FMHA runs on
   e4m3 operands rather than staying bf16 (pinned bitwise by a peaked-softmax
   V readout, and gated against the bf16-KV reference); the decode kernel
   takes its query from quant_q_buffer and its two scales from
   mla_bmm1_scale/mla_bmm2_scale, reading neither `q` nor the kv scale
   tensors; and only the append side honours kv_scale_orig_quant, so the
   context phase is correct at s = 1.0 alone while decode is correct at any s
   once the caller folds it into the bmm scales. The requests' pages are
   scattered out of order, which is what pins the e4m3-sized slab geometry.
2d. The **complete R1 cell over that fp8 pool** — YaRN table and
   q_scaling = 1/mscale^2 together with the fp8 pool, which is what every
   call of a DeepSeek-R1-0528 rank passes and what (2b) and (2c) cover only
   separately. All four call flavors: fresh prefill, generation decode, the
   mixed batch pairing them off one offsets table, and the no-append
   (latent_cache=None) context an engine with block reuse runs for every
   cached prefix — three of which had never been run over an fp8 pool at all.
   Each run is gated against a q_scaling = 1.0 reference, prefill also
   against an unscaled-rope-table one, and the appended rows carry the table
   bitwise (with a mirror-swap control proving that comparison separates the
   two tables at the positions in flight). The no-append flavor's K/V are
   built the way a target builds them — cached prefix written into the pool
   as e4m3(row/s), read back as bf16(float(byte) * s), pushed through a
   kv_b_proj-shaped matmul, only the new tokens fresh — and it turns out to
   quantize them to e4m3 all over again, settled bitwise by its own peaked-V
   readout and swept over s. quant_mode 1152 (| FP8_1x128_128x128) and 384
   (| FP8_QDQ), which a checkpoint's quant config produces where a bare 128
   does not, are bit-identical to 128 on every flavor. One test is a positive
   control rather than a certification: it replays this entry's own
   documented torn-append race (a prefill whose pages are aliased onto one
   physical page) and asserts the pool byte comparison every append check
   rests on can see it.
2e. The MLA **generation** call at predicted_tokens_per_seq > 1 — one
   speculative-decoding step, where a generation sequence contributes P query
   rows (its draft chain) instead of one. Swept over P = 1, 2, 3, 4 (the
   target's max_draft_len 0..3 plus one), at the R1 cell over the fp8 pool and
   again over a bf16 one. The question the section exists to answer is the
   within-block mask, and on sm_100 no mask tensor is involved: a linear-tree
   draft has is_spec_decoding_enabled forced off there, so the answer has to
   come from predicted_tokens_per_seq alone. It is read out directly — cache
   rows carrying one-hot compressed_kv turn the output into the attention
   weight vector — and the mask is bottom-right-aligned causal: row t attends
   to keys [0, L - P + t] and its later siblings' columns come back bitwise
   zero. mask_type 0 and 1 are bitwise identical here. Around that: the
   token-major row order, the batch-state tensors this call reads at P > 1
   (cu_q_seqlens / cu_kv_seqlens / context_lengths contents still inert,
   sequence_length still the KV extent, an all-zero
   host_past_key_value_lengths skipping the call outright), mixed batches,
   realistic-input accuracy against the causal fp32 reference with the
   full-mask model as the rival, and a control that the decode-side comparison
   can see a pool torn by the entry's own documented append race.
3. MLA context with latent_cache=None (no in-kernel RoPE or append):
   a) one-shot cached-KV context — q pre-rotated upstream, K/V covering the
      full [cached + new] range, causal mask bottom-right aligned;
   b) chunked partial passes — K/V covering one chunk slice per call under
      a padding mask with softmax_stats_tensor emitted, then a causal
      new-token pass, folded together by the downstream trtllm merge op and
      checked against a single-pass full-range fp32 reference.
4. Standard GQA over one paged pool shared by 4 layers: pool, layer->pool
   mapping, and block offsets produced by a real multi-layer KVCacheManager
   and consumed by the op as-is, with local_layer_idx 0-3 selecting mapping
   rows. Per-layer outputs, bit-exact per-layer appends, sibling-layer
   isolation, and a doctored-mapping call pinning the pool-base shift to
   the mapping row's layer-in-pool column.
5. Standard GQA (32q/8kv d128, tpb 32) over an fp8-e4m3 paged pool
   (quant_mode 128, fp32 [1] CUDA kv scale tensors): context prefill
   (bf16 FMHA + bitwise-checked e4m3 append), decode over the quantized
   cache (page-crossing, long-history, and mixed-batch), and non-1.0
   scale semantics (bit-exact at the power-of-2 scale, one-ulp-bounded at
   a non-power-of-2 scale, dequant-on-read pinned by scale-aware
   references).
6. The intersection of (4) and (5): one fp8-e4m3 paged pool shared by
   4 layers (quant_mode 128, s=1.0, GQA 32q/8kv d128, tpb 32), manager
   state consumed as-is. Per-layer context and page-crossing decode
   outputs, bit-exact per-layer e4m3 appends, and bitwise sibling-layer
   isolation across both the context and generation append paths.
7. attention_sinks over the gpt-oss-120b tp1 geometry (64q/8kv d64,
   causal, bf16 pool): the per-head sink as one extra softmax-denominator
   column dropped from the output, gated in the context phase, the
   generation phase, a mixed batch, and the multi-CTA-KV decode reduction;
   a constant-V probe reading the softmax row mass directly; per-q-head
   indexing; bitwise inertness of attention_sinks=None and of a sink that
   cannot contribute; and rejection of non-fp32 dtypes (by the op) and of
   wrong-size / strided buffers (by the wrapper).
8. The cyclic sliding window (attention_window_size < max_seq_len) with
   sinks active, on the same 64q/8kv d64 geometry with window 128 — the
   gpt-oss-120b sliding layer. A one-hot-V probe reads the attention
   weights out directly and pins the attended key set to
   [p - window + 1, p] with exact zeros outside it, in both phases; the
   realistic-input cases are gated against sink-ignored, sink-pre-scaled
   and window-ignored rivals in prefill, decode, a mixed batch, and a
   2000-token history. The caller-side contract is pinned too: the append
   still lands at the absolute token position, so a bounded ring of pages
   (absolute page j -> ring[j % P], P * tokens_per_block >= window) makes
   the pool genuinely wrap while every decode keeps its full window;
   fully-aged-out pages are bitwise unread and one page short is silently
   wrong; a prefill longer than the window is legal in one call and its
   output never reads the pool; sequence_length must stay global while
   context_lengths and max_seq_len are inert; and layers with different
   windows share one pool and one block-offset table in the same batch.
9. The paged-context execution path (use_paged_context_fmha=True), which an
   engine prepares as soon as KV-cache reuse or chunked prefill is on: the
   context FMHA sources K/V from the paged pool, so a context call may carry
   a cached prefix (KV range beyond the q range, bottom-right-aligned causal
   mask). Covered on the three shipped bf16 geometries: bitwise equivalence
   with the packed path when nothing is cached; cached prefixes across the
   page grid, chunked prefill loops, and mixed cached/fresh/generation
   batches; the same over a real 4-layer KVCacheManager pool, where the read
   now also goes through the layer base shift; and the gpt-oss cell — sinks
   plus a 128-token window over a cached prefix past the window, with the
   attended key set read out one-hot and the read page set measured page by
   page. Wrong batch states (the flag left off, context_lengths carrying the
   full KV length or 0, a replaced in-window cached page) are gated far
   outside the tolerance band.
"""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType, KVCacheManager
from tensorrt_llm._torch.staircase.catalog.attention.thop_attention import thop_attention
from tensorrt_llm.functional import RotaryScalingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

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

# fp8-e4m3 latent-pool MLA tolerance, both phases. Under quant_mode 128 both
# MLA phases run their FMHA on e4m3 operands: context quantizes q/k/v itself,
# generation reads an e4m3 query out of quant_q_buffer and e4m3 rows out of
# the pool. The references round the same operands through e4m3 and then
# accumulate in fp32, so what is left is the kernel's own fp8 handling —
# chiefly the softmax probabilities, which are an MMA operand too and carry
# e4m3's 2**-4 relative error into a weighted sum of |V| ~ 1 rows. That error
# does not shrink with the output element's own magnitude, which is why the
# floor is 2**-3 rather than the 2**-4 of the standard-configuration decode
# gate above. Measured on sm_100 at H = 128, page 32 over six seeds: at a
# 2**-4 floor the 129-token context case runs 0.72-1.08 of the allowance
# (one seed *over* the gate — a 129x128x128 output samples the same noise 16x
# more often than a two-row decode does), and at this 2**-3 floor it runs
# 0.40-0.60 while decode runs 0.14-0.20 (0.28-0.39 at a 2**-4 floor).
#
# Discrimination. This gate is deliberately not where the "is the math really
# fp8" question is settled: the difference between quantized and unquantized
# operands is the same size as the gate, so the bf16-KV context reference —
# cache quantization ignored, what the bf16 surface is certified against —
# only misses it by 1.3-1.6x (its mean abs error is 2.5x the e4m3 model's,
# and it sits 23.5-32.3x outside the *bf16* band, which is what the context
# test gates on). That question is settled bitwise instead, by the peaked
# attention case: with the softmax collapsed onto one key the output *is* the
# V row, and it comes back bit-exactly e4m3(V) while the unquantized V row is
# 63 bf16 ulps away. What this gate does separate: the s = 1.0 math against
# s = 1.5 / 2.0 context runs (21x / 48x), a zeroed mla_bmm1_scale[1] (4.2x),
# and a decode whose mla_bmm scales leave the kv scale out (3.6x at s = 2.0).
# The rival it does not resolve at all is q left unquantized in the decode
# reference (0.23-0.24x against the correct model's 0.17x — e4m3 rounding of
# the query is invisible in an output that averages hundreds of cached rows),
# so no claim here rests on it.
FP8_MLA_ATOL = 2**-3
FP8_MLA_RTOL = 2**-4


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

        # Single-layer, single-pool paged KV cache (HND layout).
        self.kv_cache = torch.zeros(
            num_blocks,
            2,
            num_kv_heads,
            tokens_per_block,
            head_dim,
            dtype=pool_dtype,
            device="cuda",
        )
        self.pool_pointers = torch.zeros(1, 2, dtype=torch.int64, device="cpu")
        self.pool_pointers[0, 0] = self.kv_cache.data_ptr()
        # KV-cache scaling factor s: dequant = quantized * s. Mirrors the
        # production construction (quant_orig = s, orig_quant = 1/s, both
        # fp32 [1] CUDA); None = unquantized pool, scale args passed as None.
        if kv_scaling_factor is None:
            self.kv_scale_quant_orig: Optional[torch.Tensor] = None
            self.kv_scale_orig_quant: Optional[torch.Tensor] = None
        else:
            factor = torch.full((1,), kv_scaling_factor, dtype=torch.float32, device="cuda")
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
                row[0, j] = 2 * p
                row[1, j] = 2 * p + 1
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
        thop_attention(
            q=qkv,
            k=None,  # packed QKV rides inside q
            v=None,
            output=output,
            output_sf=None,
            workspace_=self.workspace,
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
            predicted_tokens_per_seq=1,
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
            spec_decoding_position_offsets_for_cpp=None,
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
    ) -> torch.Tensor:
        """fp32 SDPA over each sequence's full K/V history (must be called
        after call_op so the history includes this call's K/V). q_transform /
        kv_transform, when given, map the bf16 q / K/V history first (e.g.
        the e4m3 round-trip a quantized cache imposes). window, when given,
        additionally drops keys older than the sliding window."""
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
                    mask[i, lo : cached + i + 1] = True
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
        bf16 K/V fed so far, or their scaled e4m3 casts for an fp8 pool)."""
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


def _run_and_check(
    env: _PagedAttnEnv,
    seq_lens: List[int],
    num_contexts: int,
    request_ids: List[int],
    mask_type: int = MASK_CAUSAL,
) -> None:
    cached_lens = [env.cached_len(rid) for rid in request_ids]
    qkv = env.random_qkv(sum(seq_lens))
    out = env.call_op(qkv, seq_lens, num_contexts, request_ids, mask_type)
    ref = env.reference(qkv, seq_lens, request_ids, cached_lens, mask_type)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def test_bf16_context_prefill_gqa_d128() -> None:
    """Prefill-like: pure-context batch, mixed lengths crossing a block
    boundary, GQA 8q/2kv, head_dim 128; cache append checked bit-exactly."""
    torch.manual_seed(0)
    env = _PagedAttnEnv(num_heads=8, num_kv_heads=2, head_dim=128)
    env.add_request(0, 48)
    env.add_request(1, 17)
    _run_and_check(env, [48, 17], 2, [0, 1])
    env.check_cache(0)
    env.check_cache(1)


def test_bf16_decode_reads_cached_kv_gqa_d128() -> None:
    """Decode-like: prefill writes the cache, then two gen-only steps read it.

    ctx len 64 exactly fills two 32-token blocks, so the first decode token
    lands in a freshly allocated block (block-boundary crossing).
    """
    torch.manual_seed(1)
    env = _PagedAttnEnv(num_heads=8, num_kv_heads=2, head_dim=128)
    env.add_request(0, 64)
    env.add_request(1, 17)
    _run_and_check(env, [64, 17], 2, [0, 1])
    for _ in range(2):  # two decode steps: one token per sequence each
        _run_and_check(env, [1, 1], 0, [0, 1])
    env.check_cache(0)
    env.check_cache(1)


def test_bf16_mixed_batch_gqa_d128() -> None:
    """One batch mixing a context-phase sequence and a generation-phase one.

    Context sequences must precede generation sequences in the batch.
    """
    torch.manual_seed(2)
    env = _PagedAttnEnv(num_heads=8, num_kv_heads=2, head_dim=128)
    env.add_request(0, 40)
    _run_and_check(env, [40], 1, [0])
    env.add_request(1, 23)
    _run_and_check(env, [23, 1], 1, [1, 0])
    env.check_cache(0)
    env.check_cache(1)


def test_bf16_padding_mask_context() -> None:
    """mask_type=padding: bidirectional attention over a context batch."""
    torch.manual_seed(3)
    env = _PagedAttnEnv(num_heads=8, num_kv_heads=2, head_dim=128)
    env.add_request(0, 31)
    env.add_request(1, 9)
    _run_and_check(env, [31, 9], 2, [0, 1], mask_type=MASK_PADDING)


def test_bf16_mha_head_dim64() -> None:
    """MHA (4q/4kv), head_dim 64: prefill then one decode step."""
    torch.manual_seed(4)
    env = _PagedAttnEnv(num_heads=4, num_kv_heads=4, head_dim=64)
    env.add_request(0, 50)
    env.add_request(1, 3)
    _run_and_check(env, [50, 3], 2, [0, 1])
    _run_and_check(env, [1, 1], 0, [0, 1])
    env.check_cache(0)
    env.check_cache(1)


def test_bf16_shipped_target_geometries() -> None:
    """The two head geometries the shipped bf16 targets run: 32q/8kv d128
    (GQA 4:1, qwen3-8b) and 32q/4kv d128 (GQA 8:1, qwen3-30b-a3b).

    Each gets the full standard-configuration cycle: a page-crossing prefill
    batch, two decode steps, and a mixed context+generation batch, with the
    paged append checked bit-exactly and no writes outside the token ranges.
    """
    for i, (num_heads, num_kv_heads) in enumerate([(32, 8), (32, 4)]):
        torch.manual_seed(30 + i)
        env = _PagedAttnEnv(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=128)
        env.add_request(0, 48)  # crosses a 32-token page boundary
        env.add_request(1, 17)
        _run_and_check(env, [48, 17], 2, [0, 1])
        for _ in range(2):  # two decode steps: one token per sequence each
            _run_and_check(env, [1, 1], 0, [0, 1])
        env.add_request(2, 23)  # context sequence joining a generation one
        _run_and_check(env, [23, 1], 1, [2, 0])
        for rid in (0, 1, 2):
            env.check_cache(rid)
        env.check_unwritten_pool_zero()


def test_bf16_head_count_axis() -> None:
    """Head counts are a free axis, not an enumerated list: any (Hq, Hkv)
    with Hq % Hkv == 0 works, and head_size selects the FMHA kernel.

    Covered here: MQA (16q/1kv), a non-power-of-2 GQA ratio (28q/4kv,
    ratio 7), non-power-of-2 head counts (12q/3kv), and head_size 256 —
    none of them a power-of-2 grouping the earlier cases already pinned.
    """
    geometries = [(16, 1, 128), (28, 4, 128), (12, 3, 128), (8, 2, 256)]
    for i, (num_heads, num_kv_heads, head_dim) in enumerate(geometries):
        torch.manual_seed(40 + i)
        env = _PagedAttnEnv(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
        env.add_request(0, 40)
        env.add_request(1, 7)
        _run_and_check(env, [40, 7], 2, [0, 1])
        _run_and_check(env, [1, 1], 0, [0, 1])
        env.check_cache(0)
        env.check_cache(1)


def test_rejects_non_divisible_head_counts() -> None:
    """Hq must be an integer multiple of Hkv, and the wrapper must be the
    one to say so: a context-only call at 6q/4kv d128 returns without
    raising, having computed only the first (6 // 4) * 4 = 4 head columns
    and left the other two all-zero (the decode path does raise, from
    xqaDispatcher: 'numQHeads should be multiple of numKVHeads')."""
    torch.manual_seed(50)
    env = _PagedAttnEnv(num_heads=6, num_kv_heads=4, head_dim=128)
    env.add_request(0, 16)
    try:
        env.call_op(env.random_qkv(16), [16], 1, [0], MASK_CAUSAL)
    except AssertionError:
        return
    raise AssertionError("6q/4kv geometry was not rejected by the wrapper")


# ─── Standard configuration, multi-layer shared pool ───────────────────


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise tensor equality (byte view), dtype-agnostic: float equality
    would miss NaN-payload or signed-zero byte changes in an fp8 pool."""
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


class _MultiLayerPagedAttnEnv:
    """Real multi-layer op state: a trtllm KVCacheManager hosting one paged
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

        self.mgr = KVCacheManager(
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
        self.mgr.impl.add_token(request_id)

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
        """
        ns = len(request_ids)
        kv_lens = [self.cached_len(layer_idx, rid) + new for rid, new in zip(request_ids, seq_lens)]
        if ctx_lens is None:
            ctx_lens = [self.prompt_lens[rid] for rid in request_ids]
        req_types = [0 if i < num_contexts else 1 for i in range(ns)]
        num_ctx_tokens = sum(seq_lens[:num_contexts])
        if pool_mapping is None:
            pool_mapping = self.pool_mapping

        output = torch.empty(
            sum(seq_lens),
            self.num_heads * self.head_dim,
            dtype=qkv.dtype,
            device=qkv.device,
        )
        thop_attention(
            q=qkv,
            k=None,  # packed QKV rides inside q
            v=None,
            output=output,
            output_sf=None,
            workspace_=self.workspace,
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
            kv_cache_block_offsets=self.block_offsets,
            host_kv_cache_pool_pointers=self.mgr.kv_cache_pool_pointers,
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
            predicted_tokens_per_seq=1,
            local_layer_idx=layer_idx,
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
            spec_decoding_position_offsets_for_cpp=None,
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

    def check_caches(self, request_ids: List[int]) -> None:
        """Every layer's slabs must hold exactly the expected rows, bitwise
        (the bf16 K/V fed to that layer, or their scaled e4m3 casts for an
        fp8 pool), at the pages the manager assigned (offsets are
        layer-agnostic and count slabs: one page = num_layers * 2 slabs)."""
        tpb = self.tokens_per_block
        slabs_per_page = self.num_layers * 2
        for layer_idx in range(self.num_layers):
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


def test_bf16_multilayer_shared_pool_gqa_d128() -> None:
    """One paged pool shared by 4 layers, addressed as a real multi-layer
    KVCacheManager lays it out: pages interleave every layer's K/V slabs,
    the block-offset table is layer-agnostic, and each call selects its
    layer via local_layer_idx -> pool-mapping row. Prefill plus two decode
    steps per layer (decode crosses a page boundary), GQA 8q/2kv d128:
    per-layer outputs vs fp32 references, per-layer appends bit-exact,
    sibling layers bitwise untouched by each call. A final doctored-mapping
    call pins the pool-base shift to the mapping row's layer-in-pool column
    (local_layer_idx only selects the row)."""
    torch.manual_seed(10)
    num_layers = 4
    env = _MultiLayerPagedAttnEnv(num_layers=num_layers, num_heads=8, num_kv_heads=2, head_dim=128)
    # The real manager maps layer l to (pool 0, layer-in-pool l): one
    # mapping row per layer, rows beyond 0 shifting the pool base.
    assert env.pool_mapping.tolist() == [[0, layer] for layer in range(num_layers)]

    env.add_request(0, 64)  # exactly two pages: decode crosses into a third
    env.add_request(1, 17)
    env.refresh_offsets([0, 1], num_contexts=2)
    for layer in range(num_layers):
        qkv = env.random_qkv(81)
        siblings_before = [env.layer_views[m].clone() for m in range(num_layers) if m != layer]
        out = env.call_op(layer, qkv, [64, 17], 2, [0, 1])
        ref = env.reference(layer, qkv, [64, 17], [0, 1], [0, 0])
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
        siblings_after = [env.layer_views[m] for m in range(num_layers) if m != layer]
        for before, after in zip(siblings_before, siblings_after):
            assert torch.equal(before, after)  # no cross-layer write
    env.check_caches([0, 1])

    for _ in range(2):  # decode: one token per sequence per layer per step
        env.add_decode_token(0)
        env.add_decode_token(1)
        env.refresh_offsets([0, 1], num_contexts=0)
        for layer in range(num_layers):
            cached = [env.cached_len(layer, 0), env.cached_len(layer, 1)]
            qkv = env.random_qkv(2)
            out = env.call_op(layer, qkv, [1, 1], 0, [0, 1])
            ref = env.reference(layer, qkv, [1, 1], [0, 1], cached)
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    env.check_caches([0, 1])

    # Doctored mapping: local_layer_idx=1 whose row claims layer-in-pool 3.
    # The append must land in layer 3's slabs — the shift is driven by the
    # mapping row's layer column, not by local_layer_idx itself. (Appends
    # one token past each sequence's history, inside already-allocated
    # pages; run after all correctness checks since it plants garbage.)
    doctored = env.pool_mapping.clone()
    doctored[1, 1] = 3
    views_before = [env.layer_views[m].clone() for m in range(num_layers)]
    env.call_op(1, env.random_qkv(2), [1, 1], 0, [0, 1], pool_mapping=doctored, record=False)
    assert torch.equal(env.layer_views[1], views_before[1])  # row idx not the shift
    assert not torch.equal(env.layer_views[3], views_before[3])
    assert torch.equal(env.layer_views[0], views_before[0])
    assert torch.equal(env.layer_views[2], views_before[2])


# ─── Standard configuration, fp8-e4m3 paged KV pool (quant_mode 128) ───


def _fp8_env(kv_scaling_factor: float = 1.0) -> _PagedAttnEnv:
    """GQA 32q/8kv d128 tpb 32 (qwen3-8b-like geometry), fp8-e4m3 pool."""
    return _PagedAttnEnv(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        pool_dtype=torch.float8_e4m3fn,
        quant_mode=QUANT_MODE_FP8_KV_CACHE,
        kv_scaling_factor=kv_scaling_factor,
    )


def _e4m3_roundtrip(env: "_PagedAttnEnv | _MultiLayerPagedAttnEnv"):
    """bf16 -> e4m3 (scaled by orig_quant) -> bf16 (scaled by quant_orig):
    what a value fed into the fp8 pool looks like when read back out."""
    orig_quant, quant_orig = env.kv_scale_orig_quant, env.kv_scale_quant_orig
    assert orig_quant is not None and quant_orig is not None

    def transform(t: torch.Tensor) -> torch.Tensor:
        q = (t.float() * orig_quant).to(torch.float8_e4m3fn)
        return (q.float() * quant_orig).to(t.dtype)

    return transform


def test_fp8_kv_context_prefill_gqa32_8_d128() -> None:
    """fp8 pool, context prefill: the context FMHA computes over the bf16
    packed QKV — accuracy identical to the bf16-pool surface (an fp8-KV
    reference is ~1.4e-1 off; the pool plays no part in context math) —
    while the in-op append writes e4m3(K * orig_quant) into the pool,
    bit-exactly, touching nothing else."""
    torch.manual_seed(20)
    env = _fp8_env()
    env.add_request(0, 48)  # crosses a 32-token page boundary
    env.add_request(1, 17)
    qkv = env.random_qkv(65)
    out = env.call_op(qkv, [48, 17], 2, [0, 1], MASK_CAUSAL)
    ref = env.reference(qkv, [48, 17], [0, 1], [0, 0], MASK_CAUSAL)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    env.check_cache(0)
    env.check_cache(1)
    env.check_unwritten_pool_zero()


def test_fp8_kv_decode_reads_fp8_cache() -> None:
    """Decode over the fp8 pool: prefill [64, 17] (64 fills exactly two
    pages, so the first decode token opens a fresh page), then two decode
    steps checked against the quantized-KV reference; appends stay bit-exact
    through decode. A 200-token-history decode covers prefill-like KV
    extents (observed err there is ~4x smaller than the short-history max)."""
    torch.manual_seed(21)
    env = _fp8_env()
    env.add_request(0, 64)
    env.add_request(1, 17)
    env.call_op(env.random_qkv(81), [64, 17], 2, [0, 1], MASK_CAUSAL)
    rt = _e4m3_roundtrip(env)
    for _ in range(2):
        cached = [env.cached_len(0), env.cached_len(1)]
        qkv_d = env.random_qkv(2)
        out = env.call_op(qkv_d, [1, 1], 0, [0, 1], MASK_CAUSAL)
        ref = env.reference(
            qkv_d, [1, 1], [0, 1], cached, MASK_CAUSAL, q_transform=rt, kv_transform=rt
        )
        torch.testing.assert_close(out, ref, rtol=FP8_DECODE_RTOL, atol=FP8_DECODE_ATOL)
    env.check_cache(0)
    env.check_cache(1)
    env.check_unwritten_pool_zero()

    torch.manual_seed(22)
    env = _fp8_env()
    env.add_request(0, 200)
    env.call_op(env.random_qkv(200), [200], 1, [0], MASK_CAUSAL)
    rt = _e4m3_roundtrip(env)
    qkv_d = env.random_qkv(1)
    out = env.call_op(qkv_d, [1], 0, [0], MASK_CAUSAL)
    ref = env.reference(qkv_d, [1], [0], [200], MASK_CAUSAL, q_transform=rt, kv_transform=rt)
    torch.testing.assert_close(out, ref, rtol=FP8_DECODE_RTOL, atol=FP8_DECODE_ATOL)


def test_fp8_kv_mixed_batch() -> None:
    """One call mixing a context and a generation sequence over the fp8
    pool: context rows match the bf16-KV reference (bf16 context FMHA), the
    generation row matches the quantized-KV reference (fp8 decode kernel)."""
    torch.manual_seed(23)
    env = _fp8_env()
    env.add_request(0, 40)
    env.call_op(env.random_qkv(40), [40], 1, [0], MASK_CAUSAL)
    env.add_request(1, 23)
    cached_gen = env.cached_len(0)
    qkv = env.random_qkv(24)
    out = env.call_op(qkv, [23, 1], 1, [1, 0], MASK_CAUSAL)
    ref_ctx = env.reference(qkv[:23], [23], [1], [0], MASK_CAUSAL)
    torch.testing.assert_close(out[:23], ref_ctx, rtol=RTOL, atol=ATOL)
    rt = _e4m3_roundtrip(env)
    ref_gen = env.reference(
        qkv[23:], [1], [0], [cached_gen], MASK_CAUSAL, q_transform=rt, kv_transform=rt
    )
    torch.testing.assert_close(out[23:], ref_gen, rtol=FP8_DECODE_RTOL, atol=FP8_DECODE_ATOL)
    env.check_cache(0)
    env.check_cache(1)


def test_fp8_kv_scale_semantics() -> None:
    """Non-1.0 kv scales. s=2.0 (a power of two: scaling is an exact
    exponent shift): the append stays bit-exact vs the e4m3(K * orig_quant)
    mirror — orig_quant is consumed on write — and decode matches the
    scale-aware reference — quant_orig is consumed on read (ignoring it
    shows as ~4.4e-1). s=1.5 (not a power of two, so the e4m3 round-trip
    genuinely depends on the scale value): the kernel's quantization
    arithmetic differs from fp32-multiply-then-round-to-nearest on a small
    fraction of elements (observed 0.6%), each within one e4m3 ulp; decode
    matches within the same fp8 tolerance."""
    torch.manual_seed(24)
    env = _fp8_env(kv_scaling_factor=2.0)
    env.add_request(0, 40)
    qkv = env.random_qkv(40)
    out = env.call_op(qkv, [40], 1, [0], MASK_CAUSAL)
    ref = env.reference(qkv, [40], [0], [0], MASK_CAUSAL)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)  # bf16 context
    env.check_cache(0)
    rt = _e4m3_roundtrip(env)
    qkv_d = env.random_qkv(1)
    out = env.call_op(qkv_d, [1], 0, [0], MASK_CAUSAL)
    ref = env.reference(qkv_d, [1], [0], [40], MASK_CAUSAL, q_transform=rt, kv_transform=rt)
    torch.testing.assert_close(out, ref, rtol=FP8_DECODE_RTOL, atol=FP8_DECODE_ATOL)

    torch.manual_seed(25)
    env = _fp8_env(kv_scaling_factor=1.5)
    env.add_request(0, 64)
    env.call_op(env.random_qkv(64), [64], 1, [0], MASK_CAUSAL)
    got_k, got_v, exp_k, exp_v = env.cache_pages_content(0)
    for got, exp in ((got_k, exp_k), (got_v, exp_v)):
        exact = (got.view(torch.uint8) == exp.view(torch.uint8)).float().mean().item()
        assert exact >= 0.99, f"append bit-exact fraction only {exact:.4f}"
        diff = (got.float() - exp.float()).abs()
        # One e4m3 ulp at the element's magnitude: 2^(floor(log2 |x|) - 3)
        # for normals, 2^-9 below the min normal 2^-6.
        mag = torch.maximum(got.float().abs(), exp.float().abs()).clamp(min=2**-6)
        ulp = 2.0 ** (torch.floor(torch.log2(mag)) - 3)
        assert bool((diff <= ulp).all()), "append off by more than one e4m3 ulp"
    rt = _e4m3_roundtrip(env)
    qkv_d = env.random_qkv(1)
    out = env.call_op(qkv_d, [1], 0, [0], MASK_CAUSAL)
    ref = env.reference(qkv_d, [1], [0], [64], MASK_CAUSAL, q_transform=rt, kv_transform=rt)
    torch.testing.assert_close(out, ref, rtol=FP8_DECODE_RTOL, atol=FP8_DECODE_ATOL)


def test_fp8_kv_multilayer_shared_pool_gqa32_8_d128() -> None:
    """One fp8-e4m3 paged pool shared by 4 layers (quant_mode 128, s=1.0),
    manager state consumed as-is — the production serving shape of the fp8
    cache, at the production GQA 32q/8kv d128 tpb 32 geometry. The op sizes
    slabs from quant_mode alone, so the layer-base shift must be computed in
    e4m3 slab units for the append to land where the fp8 manager laid the
    layer out. Prefill plus two decode steps per layer (decode crosses a
    page boundary): per-layer context outputs vs bf16 references (context
    FMHA reads the packed bf16 q rows, not the pool), per-layer decode
    outputs vs quantized-KV references over that layer's own history,
    per-layer e4m3 appends bit-exact, and sibling layers bitwise untouched
    by every call — context and generation append paths both."""
    torch.manual_seed(26)
    num_layers = 4
    env = _MultiLayerPagedAttnEnv(
        num_layers=num_layers,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        dtype=DataType.FP8,
        quant_mode=QUANT_MODE_FP8_KV_CACHE,
        kv_scaling_factor=1.0,
    )
    # A DataType.FP8 manager allocates a real e4m3 pool with identity
    # mapping rows, exactly like the bf16 one.
    assert env.layer_views[0].dtype == torch.float8_e4m3fn
    assert env.pool_mapping.tolist() == [[0, layer] for layer in range(num_layers)]

    def check_sibling_isolation(layer: int, fn) -> None:
        before = [env.layer_views[m].clone() for m in range(num_layers) if m != layer]
        fn()
        after = [env.layer_views[m] for m in range(num_layers) if m != layer]
        for b, a in zip(before, after):
            assert _bitwise_equal(b, a), f"call on layer {layer} wrote a sibling"

    env.add_request(0, 64)  # exactly two pages: decode crosses into a third
    env.add_request(1, 17)
    env.refresh_offsets([0, 1], num_contexts=2)
    for layer in range(num_layers):
        qkv = env.random_qkv(81)

        def ctx_call(layer=layer, qkv=qkv):
            out = env.call_op(layer, qkv, [64, 17], 2, [0, 1])
            ref = env.reference(layer, qkv, [64, 17], [0, 1], [0, 0])
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

        check_sibling_isolation(layer, ctx_call)
    env.check_caches([0, 1])

    rt = _e4m3_roundtrip(env)
    for _ in range(2):  # decode: one token per sequence per layer per step
        env.add_decode_token(0)
        env.add_decode_token(1)
        env.refresh_offsets([0, 1], num_contexts=0)
        for layer in range(num_layers):
            cached = [env.cached_len(layer, 0), env.cached_len(layer, 1)]
            qkv = env.random_qkv(2)

            def gen_call(layer=layer, qkv=qkv, cached=cached):
                out = env.call_op(layer, qkv, [1, 1], 0, [0, 1])
                ref = env.reference(
                    layer,
                    qkv,
                    [1, 1],
                    [0, 1],
                    cached,
                    q_transform=rt,
                    kv_transform=rt,
                )
                torch.testing.assert_close(out, ref, rtol=FP8_DECODE_RTOL, atol=FP8_DECODE_ATOL)

            check_sibling_isolation(layer, gen_call)
    env.check_caches([0, 1])


# ─── Attention sinks (standard configuration, bf16 pool) ───────────────
# gpt-oss-120b tp1 attention geometry: 64 q heads, 8 kv heads, head_size 64.
SINK_HQ, SINK_HKV, SINK_D = 64, 8, 64


def _sink_env(**kwargs) -> _PagedAttnEnv:
    return _PagedAttnEnv(num_heads=SINK_HQ, num_kv_heads=SINK_HKV, head_dim=SINK_D, **kwargs)


def _sinks(low: float, high: float, seed: int) -> torch.Tensor:
    """One fp32 sink logit per q head, spread over [low, high] so every head
    carries a different value (a kernel indexing sinks by kv head, or by a
    shifted head index, cannot match the reference)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.empty(SINK_HQ, dtype=torch.float32, device="cuda").uniform_(
        low, high, generator=gen
    )


def _sink_reference(
    env: _PagedAttnEnv,
    qkv: torch.Tensor,
    seq_lens: List[int],
    request_ids: List[int],
    cached_lens: List[int],
    sink: Optional[torch.Tensor],
    prescale: bool = False,
    window: Optional[int] = None,
) -> torch.Tensor:
    """fp32 causal attention with one extra per-q-head logit in the softmax
    denominator that is dropped from the numerator:

        out[i, h] = softmax([scores[i, h, :], sink[h]])[:-1] @ V

    scores are the scaled logits (QK^T / sqrt(D)); the rows of the resulting
    weight matrix therefore sum to less than 1. sink=None gives plain softmax.
    prescale=True is the rival hypothesis — the sink joins the *unscaled*
    score row and so gets multiplied by the softmax scale too. window, when
    given, keeps only the newest `window` keys per query row.

    Built from each sequence's full K/V history, so it must be called after
    call_op (same convention as _PagedAttnEnv.reference).
    """
    hq, hkv, dim = env.num_heads, env.num_kv_heads, env.head_dim
    scale = 1.0 / math.sqrt(dim)
    rep = hq // hkv
    q_w = hq * dim
    outs = []
    start = 0
    for rid, sl, cached in zip(request_ids, seq_lens, cached_lens):
        q = qkv[start : start + sl, :q_w].view(sl, hq, dim).float()
        k = torch.cat(env.k_history[rid]).float().repeat_interleave(rep, dim=1)
        v = torch.cat(env.v_history[rid]).float().repeat_interleave(rep, dim=1)
        s = torch.einsum("ihd,jhd->hij", q, k) * scale  # [hq, sl, kv]
        keep = torch.zeros(sl, k.shape[0], dtype=torch.bool, device=q.device)
        for i in range(sl):
            lo = 0 if window is None else max(0, cached + i + 1 - window)
            keep[i, lo : cached + i + 1] = True
        s = s.masked_fill(~keep.unsqueeze(0), float("-inf"))
        if sink is None:
            p = torch.softmax(s, dim=-1)
        else:
            sk = sink.float().view(hq, 1) * (scale if prescale else 1.0)
            m = torch.maximum(s.max(dim=-1).values, sk)  # [hq, sl]
            e = torch.exp(s - m.unsqueeze(-1))
            p = e / (e.sum(-1) + torch.exp(sk - m)).unsqueeze(-1)
        outs.append(torch.einsum("hij,jhd->ihd", p, v).reshape(sl, -1))
        start += sl
    return torch.cat(outs).to(qkv.dtype)


def _assert_sink_effect(out: torch.Tensor, rival: torch.Tensor, label: str) -> None:
    """The observed output must sit far outside the tolerance band around a
    rival hypothesis. Without this, matching the sink reference would prove
    nothing: a kernel that silently ignored the sink argument would pass the
    positive comparison too whenever the sink's contribution is small."""
    gap = (out.float() - rival.float()).abs().max().item()
    allowance = ATOL + RTOL * rival.float().abs().max().item()
    assert gap > 5.0 * allowance, (
        f"{label}: max|out - rival| = {gap:.4g} is not 5x outside the "
        f"{allowance:.4g} tolerance band — the sink's effect is unresolvable here"
    )


def _check_sink_case(
    env: _PagedAttnEnv,
    qkv: torch.Tensor,
    out: torch.Tensor,
    seq_lens: List[int],
    request_ids: List[int],
    cached_lens: List[int],
    sink: torch.Tensor,
    label: str,
    window: Optional[int] = None,
) -> None:
    """Positive gate against the sink reference plus the separations that make
    it meaningful: the sink is honoured at all, it is a logit in the
    *scaled*-score domain rather than a pre-scaling one, and — when a sliding
    window is active — the window is honoured too."""
    args = (env, qkv, seq_lens, request_ids, cached_lens)
    torch.testing.assert_close(
        out, _sink_reference(*args, sink, window=window), rtol=RTOL, atol=ATOL
    )
    _assert_sink_effect(out, _sink_reference(*args, None, window=window), f"{label}: sink ignored")
    _assert_sink_effect(
        out,
        _sink_reference(*args, sink, prescale=True, window=window),
        f"{label}: sink pre-scaled",
    )
    if window is not None:
        _assert_sink_effect(out, _sink_reference(*args, sink), f"{label}: window ignored")


def test_bf16_sinks_gpt_oss_geometry() -> None:
    """attention_sinks over the gpt-oss-120b tp1 geometry (64q/8kv d64, causal,
    packed QKV, bf16 pool): prefill, two decode steps, and a mixed
    context+generation batch — the three shapes attention_input_type=0 serves.

    Context and generation take different FMHA kernels, so each phase is gated
    separately; a sink honoured in one and dropped in the other would be
    silent. The cache append must be unaffected (bit-exact, no writes outside
    the token ranges).
    """
    torch.manual_seed(60)
    env = _sink_env()
    sink = _sinks(-2.0, 4.0, seed=60)

    env.add_request(0, 48)  # crosses a 32-token page boundary
    env.add_request(1, 17)
    qkv = env.random_qkv(65)
    out = env.call_op(qkv, [48, 17], 2, [0, 1], MASK_CAUSAL, attention_sinks=sink)
    # The hand-written softmax must agree with the file's SDPA-based reference
    # when the sink is absent, so the sink cases test the sink and not the math.
    torch.testing.assert_close(
        _sink_reference(env, qkv, [48, 17], [0, 1], [0, 0], None),
        env.reference(qkv, [48, 17], [0, 1], [0, 0], MASK_CAUSAL),
        rtol=RTOL,
        atol=ATOL,
    )
    _check_sink_case(env, qkv, out, [48, 17], [0, 1], [0, 0], sink, "context")

    for step in range(2):  # generation phase: one token per sequence per step
        cached = [env.cached_len(0), env.cached_len(1)]
        qkv_d = env.random_qkv(2)
        out_d = env.call_op(qkv_d, [1, 1], 0, [0, 1], MASK_CAUSAL, attention_sinks=sink)
        _check_sink_case(env, qkv_d, out_d, [1, 1], [0, 1], cached, sink, f"decode step {step}")

    env.add_request(2, 23)  # context sequence sharing a call with a decode one
    cached_gen = env.cached_len(0)
    qkv_m = env.random_qkv(24)
    out_m = env.call_op(qkv_m, [23, 1], 1, [2, 0], MASK_CAUSAL, attention_sinks=sink)
    _check_sink_case(env, qkv_m, out_m, [23, 1], [2, 0], [0, cached_gen], sink, "mixed batch")

    for rid in (0, 1, 2):
        env.check_cache(rid)
    env.check_unwritten_pool_zero()


def test_bf16_sinks_denominator_only() -> None:
    """The sink enters the denominator and never the numerator.

    With every V row set to 1, the output of head h at token i is exactly the
    softmax row mass, so the sink's effect is directly readable:

        mass[i, h] = sum_j exp(s_ij - m) / (sum_j exp(s_ij - m) + exp(sink_h - m))
                   = sigmoid(logsumexp_j(s_ij) - sink_h)

    with s the *scaled* logits. Without a sink the mass is exactly 1; with one
    it drops below 1 by an amount that depends on the sink alone. The rival
    pre-scaling hypothesis predicts sigmoid(lse - sink/sqrt(D)) and is gated out.
    """
    torch.manual_seed(61)
    env = _sink_env()
    env.add_request(0, 37)
    n = 37
    qkv = env.random_qkv(n)
    qkv[:, (SINK_HQ + SINK_HKV) * SINK_D :] = 1.0  # V slice: all ones
    sink = _sinks(-1.0, 3.0, seed=61)

    # Both runs are the same (idempotent) call over an empty-cache sequence, so
    # neither records: the second must see the first's exact batch state.
    out = env.call_op(qkv, [n], 1, [0], MASK_CAUSAL, attention_sinks=sink, record=False)
    out_ns = env.call_op(qkv, [n], 1, [0], MASK_CAUSAL, record=False)

    scale = 1.0 / math.sqrt(SINK_D)
    q_w, kv_w = SINK_HQ * SINK_D, SINK_HKV * SINK_D
    q = qkv[:, :q_w].view(n, SINK_HQ, SINK_D).float()
    k = (
        qkv[:, q_w : q_w + kv_w]
        .view(n, SINK_HKV, SINK_D)
        .float()
        .repeat_interleave(SINK_HQ // SINK_HKV, dim=1)
    )
    s = torch.einsum("ihd,jhd->hij", q, k) * scale
    s = s.masked_fill(
        ~torch.ones(n, n, dtype=torch.bool, device="cuda").tril().unsqueeze(0),
        float("-inf"),
    )
    lse = torch.logsumexp(s, dim=-1)  # [HQ, n]

    def mass_to_out(mass: torch.Tensor) -> torch.Tensor:
        return (
            mass.transpose(0, 1)
            .unsqueeze(-1)
            .expand(n, SINK_HQ, SINK_D)
            .reshape(n, -1)
            .to(torch.bfloat16)
        )

    torch.testing.assert_close(
        out, mass_to_out(torch.sigmoid(lse - sink.view(-1, 1))), rtol=RTOL, atol=ATOL
    )
    # Control: no sink => every row mass is exactly 1.
    torch.testing.assert_close(out_ns, torch.ones_like(out_ns), rtol=RTOL, atol=ATOL)
    _assert_sink_effect(out, out_ns, "denominator: sink ignored")
    _assert_sink_effect(
        out,
        mass_to_out(torch.sigmoid(lse - sink.view(-1, 1) * scale)),
        "denominator: sink pre-scaled",
    )
    # Every row mass strictly below 1: the dropped sink column is real.
    assert bool((out.float() < 1.0).all()), "some softmax row still sums to 1"


def test_bf16_sinks_per_head_indexing() -> None:
    """Sinks are indexed by *q* head. A checkerboard sink (+30 on even heads,
    -100 on odd ones) makes the indexing structurally visible: heads with the
    saturating sink must come back at ~0 (their whole softmax mass moves to the
    dropped column), heads with the inert sink must be bit-identical to the
    same call with attention_sinks=None. Any other indexing (kv head, an
    offset, a transpose) scrambles which columns are which."""
    torch.manual_seed(62)
    env = _sink_env()
    env.add_request(0, 33)
    env.add_request(1, 9)
    qkv = env.random_qkv(42)
    heads = torch.arange(SINK_HQ, device="cuda")
    sink = torch.where(
        heads % 2 == 0,
        torch.full_like(heads, 30, dtype=torch.float32),
        torch.full_like(heads, -100, dtype=torch.float32),
    )
    out = env.call_op(qkv, [33, 9], 2, [0, 1], MASK_CAUSAL, attention_sinks=sink, record=False)
    out_ns = env.call_op(qkv, [33, 9], 2, [0, 1], MASK_CAUSAL, record=False)
    per_head = out.view(-1, SINK_HQ, SINK_D)
    per_head_ns = out_ns.view(-1, SINK_HQ, SINK_D)
    # exp(max_logit - 30) <= exp(-20) at these score magnitudes: the surviving
    # mass is ~1e-9, far below one bf16 ulp of the ~1 no-sink output.
    assert per_head[:, 0::2].abs().max().item() < 1e-6, "saturated head not drained"
    assert _bitwise_equal(per_head[:, 1::2], per_head_ns[:, 1::2]), (
        "inert-sink head differs from the no-sink run"
    )
    assert per_head_ns[:, 0::2].abs().max().item() > 0.1, "no-sink control is degenerate"


def test_bf16_sinks_long_history_decode() -> None:
    """The generation kernel splits long KV ranges across CTAs and folds the
    partial softmax states in a separate reduction; the sink is applied in that
    reduction, so a short-history decode does not cover it. Histories of 600
    and 2000 tokens select the MultiCtasKvCga decode variant.

    Sinks are drawn near logsumexp(scaled scores) ~ log(kv_len) for such a
    range, so the sink column carries a resolvable share of the mass: a
    "realistic" small sink is genuinely negligible against 2000 keys and would
    make the separation gate vacuous rather than the semantics different.
    """
    for i, hist in enumerate((600, 2000)):
        torch.manual_seed(63 + i)
        env = _sink_env(num_blocks=160, max_blocks_per_seq=72)
        env.add_request(0, hist)
        env.call_op(env.random_qkv(hist), [hist], 1, [0], MASK_CAUSAL)
        sink = _sinks(math.log(hist) - 1.0, math.log(hist) + 2.0, seed=63 + i)
        qkv_d = env.random_qkv(1)
        out = env.call_op(qkv_d, [1], 0, [0], MASK_CAUSAL, attention_sinks=sink)
        _check_sink_case(env, qkv_d, out, [1], [0], [hist], sink, f"decode over {hist} cached")


def test_bf16_sinks_inert_path() -> None:
    """attention_sinks=None must keep reproducing the certified no-sink
    semantics, and a sink that cannot contribute must be indistinguishable
    from absent — bit for bit, in both phases.

    Repeating an identical call is safe: the append derives its position from
    sequence_length minus the new-token count, so it rewrites the same slots.
    """
    torch.manual_seed(64)
    env = _sink_env()
    env.add_request(0, 40)
    env.add_request(1, 41)
    qkv = env.random_qkv(81)
    ctx = env.call_op(qkv, [40, 41], 2, [0, 1], MASK_CAUSAL, record=False)
    assert _bitwise_equal(ctx, env.call_op(qkv, [40, 41], 2, [0, 1], MASK_CAUSAL, record=False)), (
        "context call is not bitwise reproducible"
    )
    for value in (-100.0, float("-inf")):
        inert = torch.full((SINK_HQ,), value, dtype=torch.float32, device="cuda")
        assert _bitwise_equal(
            ctx,
            env.call_op(
                qkv,
                [40, 41],
                2,
                [0, 1],
                MASK_CAUSAL,
                attention_sinks=inert,
                record=False,
            ),
        ), f"context: sink={value} is not bitwise inert"
    # Record the prefill once, then the same checks on the generation kernel.
    env.call_op(qkv, [40, 41], 2, [0, 1], MASK_CAUSAL)
    torch.testing.assert_close(
        ctx,
        env.reference(qkv, [40, 41], [0, 1], [0, 0], MASK_CAUSAL),
        rtol=RTOL,
        atol=ATOL,
    )
    qkv_d = env.random_qkv(2)
    gen = env.call_op(qkv_d, [1, 1], 0, [0, 1], MASK_CAUSAL, record=False)
    assert _bitwise_equal(gen, env.call_op(qkv_d, [1, 1], 0, [0, 1], MASK_CAUSAL, record=False)), (
        "generation call is not bitwise reproducible"
    )
    for value in (-100.0, float("-inf")):
        inert = torch.full((SINK_HQ,), value, dtype=torch.float32, device="cuda")
        assert _bitwise_equal(
            gen,
            env.call_op(
                qkv_d,
                [1, 1],
                0,
                [0, 1],
                MASK_CAUSAL,
                attention_sinks=inert,
                record=False,
            ),
        ), f"generation: sink={value} is not bitwise inert"
    env.call_op(qkv_d, [1, 1], 0, [0, 1], MASK_CAUSAL)  # record the decode step
    torch.testing.assert_close(
        gen,
        env.reference(qkv_d, [1, 1], [0, 1], [40, 41], MASK_CAUSAL),
        rtol=RTOL,
        atol=ATOL,
    )


def test_sinks_reject_bad_dtype_and_layout() -> None:
    """Only the dtype is checked by the op ('Expected attention_sinks to have
    float dtype'). Size and contiguity are not: the buffer is read as
    num_heads raw fp32 values from data_ptr(), so a short tensor is read past
    its end and a strided view is read as its underlying memory — both
    silently wrong. The wrapper is what must reject those."""
    torch.manual_seed(65)
    env = _sink_env()
    env.add_request(0, 16)
    qkv = env.random_qkv(16)
    good = _sinks(-2.0, 4.0, seed=65)
    other = _sinks(-2.0, 4.0, seed=66)

    for dtype in (torch.bfloat16, torch.float16, torch.float64):
        try:
            env.call_op(
                qkv,
                [16],
                1,
                [0],
                MASK_CAUSAL,
                attention_sinks=good.to(dtype),
                record=False,
            )
        except RuntimeError as exc:
            assert "float dtype" in str(exc), f"unexpected rejection for {dtype}: {exc}"
        else:
            raise AssertionError(f"{dtype} attention_sinks was not rejected")

    # Interleaving good/other and taking every second element yields a tensor
    # whose *values* are `good` but whose memory is not — the op would read the
    # interleaved bytes.
    interleaved = torch.stack([good, other], dim=1).reshape(-1)
    bad_layouts = {
        "stride-2 view": interleaved[::2],
        "too few elements": good[: SINK_HQ // 2].contiguous(),
        "too many elements": torch.cat([good, other]),
        "empty": torch.empty(0, dtype=torch.float32, device="cuda"),
    }
    for label, sinks in bad_layouts.items():
        try:
            env.call_op(qkv, [16], 1, [0], MASK_CAUSAL, attention_sinks=sinks, record=False)
        except AssertionError:
            continue
        raise AssertionError(f"{label} attention_sinks was not rejected by the wrapper")


# ─── Cyclic sliding window (attention_window_size < max_seq_len) + sinks ─
# gpt-oss-120b tp1 sliding layers: 64q/8kv d64, window 128, per-head sink.
SWA_W = 128
# Relative tolerance for the exact per-key attention weights read out by the
# one-hot probe below. Those weights are pure fp32 softmax constants rounded
# once to bf16: half an ulp is 2^-9 = 2.0e-3 relative, and the kernel's own
# fp32 softmax rounding lands on top. 6e-3 is ~1.5 bf16 ulp; the worst
# observed on sm_100 across the probe cases is 3.6e-3, 59% of it.
SWA_WEIGHT_RTOL = 6e-3


def _swa_env(**kwargs) -> _PagedAttnEnv:
    """Sink-geometry env with the cyclic sliding window switched on."""
    kwargs.setdefault("attention_window_size", SWA_W)
    return _sink_env(**kwargs)


def _swa_indicator_qkv(n_tokens: int, positions: List[int], t0: int) -> torch.Tensor:
    """Packed QKV that turns the op into an attention-weight read-out.

    Every K row is zero, so every unmasked logit is exactly 0 and the softmax
    is uniform over precisely the attended set. V is a one-hot indicator —
    V[t] = e_{t - t0} for t in [t0, t0 + head_dim) and 0 otherwise — so output
    column d of a query row IS the attention weight that row gives key t0 + d.
    A key outside the window must therefore come back as an exact zero.
    """
    q_w, kv_w = SINK_HQ * SINK_D, SINK_HKV * SINK_D
    qkv = torch.zeros(n_tokens, q_w + 2 * kv_w, dtype=torch.bfloat16, device="cuda")
    qkv[:, :q_w] = torch.randn(n_tokens, q_w, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros(n_tokens, SINK_HKV, SINK_D, dtype=torch.bfloat16, device="cuda")
    for i, t in enumerate(positions):
        d = t - t0
        if 0 <= d < SINK_D:
            v[i, :, d] = 1.0
    qkv[:, q_w + kv_w :] = v.reshape(n_tokens, kv_w)
    return qkv


def _assert_window_weights(
    out: torch.Tensor,
    row: int,
    pos: int,
    t0: int,
    sink: Optional[torch.Tensor],
    label: str,
    window: int = SWA_W,
) -> None:
    """Output row `row` belongs to the query at absolute position `pos`. Under
    the one-hot probe its column d holds the weight of key t0 + d, which must
    be exactly 0 outside [pos - window + 1, pos] and 1 / (n + exp(sink_h))
    inside it, with n = min(pos + 1, window) keys."""
    w = out.view(-1, SINK_HQ, SINK_D)[row].float()  # [Hq, D] weights
    lo, n = max(0, pos - window + 1), min(pos + 1, window)
    denom = float(n) + (
        torch.zeros(SINK_HQ, dtype=torch.float64, device="cuda")
        if sink is None
        else torch.exp(sink.double())
    )
    expect = (1.0 / denom).float()
    inside = [d for d in range(SINK_D) if lo <= t0 + d <= pos]
    outside = [d for d in range(SINK_D) if not (lo <= t0 + d <= pos)]
    if outside:
        assert bool((w[:, outside] == 0).all()), (
            f"{label}: keys outside [{lo}, {pos}] carry weight "
            f"{w[:, outside].abs().max().item():.4g}"
        )
    for d in inside:
        torch.testing.assert_close(
            w[:, d],
            expect,
            rtol=SWA_WEIGHT_RTOL,
            atol=0.0,
            msg=lambda m, d=d: f"{label}: key {t0 + d} weight off\n{m}",
        )


def test_bf16_swa_window_boundary_exact() -> None:
    """Which keys does a query at absolute position p actually attend to?

    The one-hot probe answers it without any tolerance on the boundary: an
    out-of-window key is an exact zero, an in-window key is the uniform weight
    1 / (n + exp(sink_h)). Two probe placements pin both ends —
    t0 = 0 catches the moment the window starts biting (row 128 must drop key
    0) and t0 = 66 catches the moving lower edge (row 199 must drop keys
    66..71, keeping 72). The result: the attended set is [p - W + 1, p],
    exactly W keys, never W + 1 — in the context phase and in the generation
    phase, with the sink active in both.
    """
    torch.manual_seed(70)
    sink = _sinks(-1.0, 3.0, seed=70)
    prefill = 200  # runs well past the 128 window
    for t0 in (0, 66):
        for sk in (sink, None):
            env = _swa_env()
            env.add_request(0, prefill)
            qkv = _swa_indicator_qkv(prefill, list(range(prefill)), t0)
            out = env.call_op(qkv, [prefill], 1, [0], MASK_CAUSAL, attention_sinks=sk)
            for row in (66, 100, 127, 128, 150, 199):
                _assert_window_weights(out, row, row, t0, sk, f"context t0={t0} row={row}")
            qd = _swa_indicator_qkv(1, [prefill], t0)
            out_d = env.call_op(qd, [1], 0, [0], MASK_CAUSAL, attention_sinks=sk)
            _assert_window_weights(out_d, 0, prefill, t0, sk, f"decode t0={t0}")
    # Sanity: the probe is not vacuous — with the window off, the same prefill
    # gives every key of row 199 a non-zero weight (1 / 200), so the zeros
    # above come from the window and not from the construction.
    env = _sink_env()
    env.add_request(0, prefill)
    qkv = _swa_indicator_qkv(prefill, list(range(prefill)), 66)
    out = env.call_op(qkv, [prefill], 1, [0], MASK_CAUSAL)
    w199 = out.view(-1, SINK_HQ, SINK_D)[199].float()
    torch.testing.assert_close(
        w199, torch.full_like(w199, 1.0 / prefill), rtol=SWA_WEIGHT_RTOL, atol=0.0
    )


def test_bf16_swa_window_values_not_page_aligned() -> None:
    """The window is a token count, not a page count. Windows that are not
    multiples of tokens_per_block (33, 100 against pages of 32) — including one
    shorter than a single page — put the boundary at exactly the same
    [p - window + 1, p], read out one-hot with the sink active."""
    torch.manual_seed(79)
    sink = _sinks(-1.0, 3.0, seed=79)
    prefill = 150
    for window in (33, 100):
        env = _sink_env(attention_window_size=window)
        env.add_request(0, prefill)
        t0 = prefill - 1 - window - 3  # straddles the lower edge of the last rows
        qkv = _swa_indicator_qkv(prefill, list(range(prefill)), t0)
        out = env.call_op(qkv, [prefill], 1, [0], MASK_CAUSAL, attention_sinks=sink)
        _assert_window_weights(
            out, prefill - 1, prefill - 1, t0, sink, f"context w={window}", window
        )
        qd = _swa_indicator_qkv(1, [prefill], t0)
        out_d = env.call_op(qd, [1], 0, [0], MASK_CAUSAL, attention_sinks=sink)
        _assert_window_weights(out_d, 0, prefill, t0, sink, f"decode w={window}", window)


def test_bf16_swa_sinks_context_and_decode() -> None:
    """The gpt-oss-120b sliding layer on realistic inputs: window 128 with a
    per-head sink, over a prefill that runs past the window plus three decode
    steps. Every case is gated against three rivals — sink ignored, sink
    pre-scaled, window ignored (full causal) — so neither mechanism can hide
    behind the other. The paged append is unaffected by the window: it writes
    each new token at its absolute position, so the pool still holds the whole
    history bit-exactly and nothing outside the token ranges is touched.

    Sinks are drawn near log(window) — the logsumexp scale of a 128-key
    softmax — so the sink column keeps a resolvable share of the mass once the
    window is full; a smaller sink is genuinely negligible there and would
    make the sink-ignored separation gate vacuous rather than the semantics
    different."""
    torch.manual_seed(71)
    env = _swa_env(max_blocks_per_seq=16, num_blocks=96)
    sink = _sinks(math.log(SWA_W) - 1.0, math.log(SWA_W) + 2.0, seed=71)
    env.add_request(0, 200)  # 200 > 128: the window bites inside the prefill
    env.add_request(1, 96)  # shorter than the window: plain causal
    qkv = env.random_qkv(296)
    out = env.call_op(qkv, [200, 96], 2, [0, 1], MASK_CAUSAL, attention_sinks=sink)
    # The hand-written windowed softmax must agree with the file's SDPA-based
    # reference when the sink is absent, so the sink cases test the sink.
    torch.testing.assert_close(
        _sink_reference(env, qkv, [200, 96], [0, 1], [0, 0], None, window=SWA_W),
        env.reference(qkv, [200, 96], [0, 1], [0, 0], MASK_CAUSAL, window=SWA_W),
        rtol=RTOL,
        atol=ATOL,
    )
    _check_sink_case(env, qkv, out, [200, 96], [0, 1], [0, 0], sink, "swa context", window=SWA_W)

    for step in range(3):
        cached = [env.cached_len(0), env.cached_len(1)]
        qkv_d = env.random_qkv(2)
        out_d = env.call_op(qkv_d, [1, 1], 0, [0, 1], MASK_CAUSAL, attention_sinks=sink)
        _check_sink_case(
            env,
            qkv_d,
            out_d,
            [1, 1],
            [0, 1],
            cached,
            sink,
            f"swa decode step {step}",
            window=SWA_W,
        )
    for rid in (0, 1):
        env.check_cache(rid)  # whole history, bit-exact, at absolute positions
    env.check_unwritten_pool_zero()


def test_bf16_swa_sinks_mixed_batch_and_long_history() -> None:
    """Two shapes the window has to survive besides the plain ones: a mixed
    context+generation call, and a decode over a history far longer than the
    window, which selects the multi-CTA-KV decode variant that folds partial
    softmax states across CTAs. Sinks are drawn near log(window) so the sink
    column carries a resolvable share of a 128-key softmax — a small sink is
    negligible there and would make the separation gate vacuous."""
    torch.manual_seed(72)
    env = _swa_env(max_blocks_per_seq=16, num_blocks=96)
    sink = _sinks(math.log(SWA_W) - 1.0, math.log(SWA_W) + 2.0, seed=72)
    env.add_request(0, 300)
    env.call_op(env.random_qkv(300), [300], 1, [0], MASK_CAUSAL, attention_sinks=sink)
    env.add_request(1, 210)  # context sequence sharing the call with a decode one
    cached_gen = env.cached_len(0)
    qkv_m = env.random_qkv(211)
    out_m = env.call_op(qkv_m, [210, 1], 1, [1, 0], MASK_CAUSAL, attention_sinks=sink)
    _check_sink_case(
        env,
        qkv_m,
        out_m,
        [210, 1],
        [1, 0],
        [0, cached_gen],
        sink,
        "swa mixed batch",
        window=SWA_W,
    )

    long_env = _swa_env(num_blocks=200, max_blocks_per_seq=72)
    long_env.add_request(0, 2000)
    long_env.call_op(long_env.random_qkv(2000), [2000], 1, [0], MASK_CAUSAL)
    qd = long_env.random_qkv(1)
    out_l = long_env.call_op(qd, [1], 0, [0], MASK_CAUSAL, attention_sinks=sink)
    _check_sink_case(
        long_env,
        qd,
        out_l,
        [1],
        [0],
        [2000],
        sink,
        "swa decode over 2000 cached",
        window=SWA_W,
    )


def test_bf16_swa_cyclic_pool_wraps() -> None:
    """The cache genuinely wraps, not merely gets masked.

    The caller gives each sequence a bounded ring of physical pages and maps
    absolute page index j to ring[j % P]. Because the op appends every token
    at its *absolute* position (page j = t // tokens_per_block, slot
    t % tokens_per_block), that mapping makes the pool hold exactly the last
    P * tokens_per_block tokens, overwriting aged-out slots in place. With
    P * tokens_per_block >= window every decode still sees its full window.
    Here the ring holds 192 tokens and the sequence runs to 400 — two full
    wraps — with the sink on and every step checked against fp32."""
    torch.manual_seed(73)
    tpb, ring = 32, 6
    env = _swa_env(
        tokens_per_block=tpb,
        num_blocks=16,
        max_blocks_per_seq=16,
        page_ring=ring,
    )
    cap = ring * tpb  # 192 token slots >= window 128
    sink = _sinks(math.log(SWA_W) - 1.0, math.log(SWA_W) + 2.0, seed=73)
    prefill = 150  # <= cap: one call's new tokens must map to distinct slots
    env.add_request(0, prefill)
    qkv = env.random_qkv(prefill)
    out = env.call_op(qkv, [prefill], 1, [0], MASK_CAUSAL, attention_sinks=sink)
    _check_sink_case(env, qkv, out, [prefill], [0], [0], sink, "ring context", window=SWA_W)

    wraps = 0
    for step in range(250):
        pos = env.cached_len(0)
        qkv_d = env.random_qkv(1)
        page, slot = (pos // tpb) % ring, pos % tpb
        before = env.kv_cache.clone()
        out_d = env.call_op(qkv_d, [1], 0, [0], MASK_CAUSAL, attention_sinks=sink)
        if pos >= cap:  # this slot already held token pos - cap
            wraps += 1
            evicted = torch.cat(env.k_history[0])[pos - cap]
            assert _bitwise_equal(before[env.rings[0][page], 0, :, slot, :], evicted), (
                f"step {pos}: ring slot did not hold the token about to age out"
            )
        # Exactly one (K, V) row pair moves per decode: the append overwrites
        # the evicted slot in place and touches nothing else in the pool.
        changed = {
            (int(b), int(s))
            for b, _kv, _h, s in (before != env.kv_cache).any(-1).nonzero().tolist()
        }
        assert changed == {(env.rings[0][page], slot)}, (
            f"step {pos}: decode wrote pool slots {sorted(changed)}"
        )
        if step % 25 == 0 or step == 249:  # full rival gating, periodically
            _check_sink_case(
                env,
                qkv_d,
                out_d,
                [1],
                [0],
                [pos],
                sink,
                f"ring decode p={pos}",
                window=SWA_W,
            )
        else:
            torch.testing.assert_close(
                out_d,
                _sink_reference(env, qkv_d, [1], [0], [pos], sink, window=SWA_W),
                rtol=RTOL,
                atol=ATOL,
            )
    assert wraps > 200, f"the ring never wrapped enough ({wraps} wrapping steps)"
    # The ring physically holds the last `cap` tokens, each at slot t % cap.
    history = torch.cat(env.k_history[0])
    total = history.shape[0]
    rows = torch.cat([env.kv_cache[p, 0].permute(1, 0, 2) for p in env.rings[0]]).contiguous()
    newest = {t % cap: t for t in range(total - cap, total)}
    assert _bitwise_equal(rows, torch.stack([history[newest[s]] for s in range(cap)])), (
        "the ring does not hold the last cap tokens at slot t % cap"
    )


def test_bf16_swa_aged_out_pages_and_ring_bound() -> None:
    """What the caller still owes the op once tokens have aged out.

    A page whose tokens all sit below the window is never read — pointing its
    block-offset entry at a loud decoy leaves the output bitwise unchanged, so
    those pages may be recycled. Every page holding at least one in-window
    token must be valid: the same substitution there changes the result. The
    ring bound follows: P * tokens_per_block >= window, and one page short is
    silently wrong rather than rejected."""
    torch.manual_seed(74)
    tpb = 32
    env = _swa_env(tokens_per_block=tpb, num_blocks=64, max_blocks_per_seq=16)
    env.add_request(0, 200)
    env.call_op(env.random_qkv(200), [200], 1, [0], MASK_CAUSAL)
    qd = env.random_qkv(1)
    # record=False throughout: every call below is the same decode (kv length
    # 201, window [73, 200]) with only the page table changing, and the append
    # is idempotent, so the outputs are directly comparable bit for bit.
    base = env.call_op(qd, [1], 0, [0], MASK_CAUSAL, record=False)
    decoy = env.kv_cache.shape[0] - 1
    env.kv_cache[decoy] = 7.0
    live_from = (201 - SWA_W) // tpb  # first page holding an in-window token
    pages = env.pages[0]
    for j in range(len(pages)):
        keep, pages[j] = pages[j], decoy
        out = env.call_op(qd, [1], 0, [0], MASK_CAUSAL, record=False)
        pages[j] = keep
        if j < live_from:
            assert _bitwise_equal(out, base), (
                f"page {j} is fully aged out but its content still reached the output"
            )
        else:
            assert not _bitwise_equal(out, base), (
                f"page {j} holds in-window tokens but the decoy changed nothing"
            )

    for ring, ok in ((SWA_W // tpb, True), (SWA_W // tpb - 1, False)):
        renv = _swa_env(tokens_per_block=tpb, num_blocks=16, max_blocks_per_seq=16, page_ring=ring)
        renv.add_request(0, 100)
        renv.call_op(renv.random_qkv(100), [100], 1, [0], MASK_CAUSAL)
        worst, allowance = 0.0, 0.0
        for _ in range(120):
            pos = renv.cached_len(0)
            qkv_d = renv.random_qkv(1)
            out_d = renv.call_op(qkv_d, [1], 0, [0], MASK_CAUSAL)
            ref = _sink_reference(renv, qkv_d, [1], [0], [pos], None, window=SWA_W)
            worst = max(worst, (out_d.float() - ref.float()).abs().max().item())
            allowance = max(allowance, ATOL + RTOL * ref.float().abs().max().item())
        if ok:
            assert worst <= allowance, (
                f"a ring of {ring} pages holds the whole {SWA_W}-token window but "
                f"the output is off by {worst:.4g}"
            )
        else:
            assert worst > 10 * allowance, (
                f"a ring of {ring} pages cannot hold the window, yet the output "
                f"is within {worst:.4g} of the reference"
            )


def test_bf16_swa_context_prefill_longer_than_window() -> None:
    """A single context prefill longer than the window is legal in one call:
    the output matches the windowed reference row by row. The context FMHA
    reads the packed QKV, never the pool — aliasing every page of the sequence
    onto one decoy page leaves the context output bitwise identical (the
    append still runs, so the pool itself becomes garbage; that call is last).
    """
    torch.manual_seed(75)
    env = _swa_env(num_blocks=64, max_blocks_per_seq=16)
    sink = _sinks(math.log(SWA_W) - 1.0, math.log(SWA_W) + 2.0, seed=75)
    prefill = 300  # 2.3x the window in one call
    env.add_request(0, prefill)
    qkv = env.random_qkv(prefill)
    out = env.call_op(qkv, [prefill], 1, [0], MASK_CAUSAL, attention_sinks=sink, record=False)
    # Same call with every page of the sequence aliased onto one loud decoy:
    # the pool the call reads back is nonsense, the output must not move.
    decoy = env.kv_cache.shape[0] - 1
    env.kv_cache[decoy] = 7.0
    keep, env.pages[0] = env.pages[0], [decoy] * len(env.pages[0])
    aliased = env.call_op(qkv, [prefill], 1, [0], MASK_CAUSAL, attention_sinks=sink, record=False)
    env.pages[0] = keep
    assert _bitwise_equal(aliased, out), (
        "context output depends on the pool content; it must read packed QKV only"
    )
    # Re-run over the real pages (the append is idempotent) and record it, so
    # the reference and the cache check see the true history.
    again = env.call_op(qkv, [prefill], 1, [0], MASK_CAUSAL, attention_sinks=sink)
    assert _bitwise_equal(again, out), "context call is not bitwise reproducible"
    _check_sink_case(env, qkv, out, [prefill], [0], [0], sink, "long prefill", window=SWA_W)
    env.check_cache(0)  # the prefill wrote every token at its absolute position


def test_bf16_swa_batch_state_requirements() -> None:
    """What the length tensors must carry once tokens have aged out.

    sequence_length stays the *global* cached+new count: it is what the window
    is measured back from and where the append lands, so capping it to the
    window both relocates the write and truncates the attended range.
    context_lengths on a generation row and max_seq_len are inert — the same
    decode comes back bitwise identical for every value tried."""
    torch.manual_seed(76)
    prefill = 200

    def prepared() -> Tuple[_PagedAttnEnv, torch.Tensor]:
        env = _swa_env(num_blocks=64, max_blocks_per_seq=16)
        env.add_request(0, prefill)
        torch.manual_seed(76)
        env.call_op(env.random_qkv(prefill), [prefill], 1, [0], MASK_CAUSAL)
        torch.manual_seed(77)
        return env, env.random_qkv(1)

    env, qd = prepared()
    base = env.call_op(qd, [1], 0, [0], MASK_CAUSAL)
    ref = _sink_reference(env, qd, [1], [0], [prefill], None, window=SWA_W)
    torch.testing.assert_close(base, ref, rtol=RTOL, atol=ATOL)

    capped_env, capped_q = prepared()
    capped = capped_env.call_op(
        capped_q, [1], 0, [0], MASK_CAUSAL, record=False, seq_lens_override=[SWA_W]
    )
    allowance = ATOL + RTOL * ref.float().abs().max().item()
    assert (capped.float() - base.float()).abs().max().item() > 10 * allowance, (
        "a window-capped sequence_length silently produced the same answer"
    )
    kv_w = SINK_HKV * SINK_D
    q_w = SINK_HQ * SINK_D
    relocated = capped_q[0, q_w : q_w + kv_w].view(SINK_HKV, SINK_D)
    got_k, _, _, _ = capped_env.cache_pages_content(0)
    assert _bitwise_equal(got_k[SWA_W - 1], relocated), (
        "sequence_length also drives the append position: capping it must have "
        "written the new token at window - 1"
    )

    for ctx_len in (0, 1, prefill):
        env_c, q_c = prepared()
        out_c = env_c.call_op(
            q_c, [1], 0, [0], MASK_CAUSAL, record=False, ctx_lens_override=[ctx_len]
        )
        assert _bitwise_equal(out_c, base), f"context_lengths={ctx_len} changed a decode"
    for msl in (SWA_W, prefill + 1, 4 * (prefill + 1)):
        env_m, q_m = prepared()
        out_m = env_m.call_op(q_m, [1], 0, [0], MASK_CAUSAL, record=False, max_seq_len_override=msl)
        assert _bitwise_equal(out_m, base), f"max_seq_len={msl} changed a decode"


def test_bf16_swa_per_layer_windows_shared_pool() -> None:
    """attention_window_size is a per-call scalar and cache addressing is by
    absolute position, so layers with different windows share one pool, one
    layer->pool mapping and one block-offset table inside the same batch —
    the alternation gpt-oss runs (sliding, full, sliding, full). Each layer is
    checked against its own window's reference and against the other layer's
    window, which must be far away."""
    torch.manual_seed(78)
    num_layers, prefill = 4, 200
    env = _MultiLayerPagedAttnEnv(
        num_layers=num_layers,
        num_heads=SINK_HQ,
        num_kv_heads=SINK_HKV,
        head_dim=SINK_D,
        max_seq_len=512,
    )
    windows = [SWA_W, None, SWA_W, None]  # None = full attention
    env.add_request(0, prefill)
    env.refresh_offsets([0], num_contexts=1)
    for layer, window in enumerate(windows):
        qkv = env.random_qkv(prefill)
        out = env.call_op(layer, qkv, [prefill], 1, [0], attention_window_size=window)
        ref = env.reference(layer, qkv, [prefill], [0], [0], window=window)
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
        other = env.reference(layer, qkv, [prefill], [0], [0], window=None if window else SWA_W)
        gap = (out.float() - other.float()).abs().max().item()
        assert gap > 5.0 * (ATOL + RTOL * other.float().abs().max().item()), (
            f"layer {layer}: the two windows are indistinguishable here"
        )
    for _ in range(2):
        env.add_decode_token(0)
        env.refresh_offsets([0], num_contexts=0)
        for layer, window in enumerate(windows):
            cached = env.cached_len(layer, 0)
            qkv_d = env.random_qkv(1)
            out_d = env.call_op(layer, qkv_d, [1], 0, [0], attention_window_size=window)
            ref_d = env.reference(layer, qkv_d, [1], [0], [cached], window=window)
            torch.testing.assert_close(out_d, ref_d, rtol=RTOL, atol=ATOL)
    env.check_caches([0])  # every layer's full history, bit-exact


# ─── Paged-context FMHA (use_paged_context_fmha=True) ──────────────────
# The execution path an engine prepares as soon as KV-cache reuse or chunked
# prefill is on — trtllm's default, so the path every bf16 target here runs.
# With the flag set the context FMHA sources K/V from the paged pool instead
# of the packed q rows, which is what lets a context call carry a cached
# prefix: its KV range exceeds its q range and the causal mask is
# bottom-right aligned.

# A rival batch state must land this far outside the tolerance band around
# the correct output. The wrong states measured below sit at 31x-318x on
# sm_100 (aliased pages 31x, the flag left off over a cached prefix 51x,
# context_lengths carrying the full KV length 101x or 0 53x, a decoy in
# place of an in-window cached page 248x-318x), so the gate is nowhere near
# any of them, while a correct call uses at most 66% of the same band.
PAGED_CTX_MIN_SEPARATION = 20.0


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


def _run_paged_ctx_and_check(
    env: _PagedAttnEnv,
    seq_lens: List[int],
    num_contexts: int,
    request_ids: List[int],
    window: Optional[int] = None,
) -> None:
    """Positive gate for one paged-context call: fp32 attention over each
    sequence's full [cached + new] history, causal mask bottom-right aligned
    (query i of a context row sits at absolute position cached + i)."""
    cached = [env.cached_len(rid) for rid in request_ids]
    qkv = env.random_qkv(sum(seq_lens))
    out = _paged_ctx_call(env, qkv, seq_lens, num_contexts, request_ids)
    ref = env.reference(qkv, seq_lens, request_ids, cached, MASK_CAUSAL, window=window)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


def _assert_far_outside(out: torch.Tensor, correct: torch.Tensor, label: str) -> float:
    """A wrong batch state must land far outside the tolerance band around the
    correct output. A non-finite result counts as far outside — some wrong
    states also corrupt the pool, and a later read of it can produce NaNs.
    Returns the observed separation in units of the allowance."""
    diff = (out.float() - correct.float()).abs()
    if not bool(torch.isfinite(diff).all()):
        return float("inf")
    allowance = ATOL + RTOL * correct.float().abs().max().item()
    ratio = diff.max().item() / allowance
    assert ratio > PAGED_CTX_MIN_SEPARATION, (
        f"{label}: only {ratio:.4g}x outside the tolerance band"
    )
    return ratio


def test_bf16_paged_context_no_cached_tokens_matches_packed_path() -> None:
    """With nothing cached, the flag changes no observable.

    Every fresh prefill of a target running with cache reuse takes this path,
    so it has to reproduce the packed-QKV one exactly. A page-crossing prefill
    batch and a decode step, run from identical state at both flag values,
    come back bitwise identical in output and in pool content — on all three
    shipped bf16 geometries. The K/V source does move (see the aliasing case
    below); it is the append running first that leaves the pool holding
    exactly the packed rows the other path reads.
    """
    for i, (num_heads, num_kv_heads, head_dim) in enumerate(
        [(32, 8, 128), (32, 4, 128), (64, 8, 64)]
    ):
        runs = []
        for use_paged in (False, True):
            torch.manual_seed(80 + i)
            env = _PagedAttnEnv(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim)
            env.add_request(0, 48)  # crosses a 32-token page boundary
            env.add_request(1, 17)
            qkv = env.random_qkv(65)
            out = env.call_op(
                qkv,
                [48, 17],
                2,
                [0, 1],
                MASK_CAUSAL,
                use_paged_context_fmha=use_paged,
            )
            torch.testing.assert_close(
                out,
                env.reference(qkv, [48, 17], [0, 1], [0, 0], MASK_CAUSAL),
                rtol=RTOL,
                atol=ATOL,
            )
            cached = [env.cached_len(0), env.cached_len(1)]
            qkv_d = env.random_qkv(2)
            out_d = env.call_op(
                qkv_d, [1, 1], 0, [0, 1], MASK_CAUSAL, use_paged_context_fmha=use_paged
            )
            torch.testing.assert_close(
                out_d,
                env.reference(qkv_d, [1, 1], [0, 1], cached, MASK_CAUSAL),
                rtol=RTOL,
                atol=ATOL,
            )
            env.check_cache(0)
            env.check_cache(1)
            env.check_unwritten_pool_zero()
            runs.append((out, out_d, env.kv_cache))
        packed_run, paged_run = runs
        for what, packed, paged in zip(
            ("prefill output", "decode output", "pool"), packed_run, paged_run
        ):
            assert _bitwise_equal(packed, paged), (
                f"{num_heads}/{num_kv_heads}/{head_dim}: {what} differs between "
                f"the packed and paged context paths with nothing cached"
            )


def test_bf16_paged_context_cached_prefix_shipped_geometries() -> None:
    """The default path of the two plain-causal shipped targets: 32q/8kv d128
    (qwen3-8b) and 32q/4kv d128 (qwen3-30b-a3b).

    One full cycle per geometry: a fresh prefill, a context call over the
    cached prefix that crosses a page boundary, a second one landing exactly
    on a page end, a decode step, a second sequence, and finally a mixed batch
    pairing a cached-prefix context row with a fresh context row and a
    generation row — the batch shape a server with prefix reuse produces. The
    appends stay bit-exact and nothing outside the token ranges is written.
    """
    for i, (num_heads, num_kv_heads) in enumerate([(32, 8), (32, 4)]):
        torch.manual_seed(83 + i)
        env = _PagedAttnEnv(num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=128)
        env.add_request(0, 96)  # served by three context calls: 48 + 33 + 15
        _run_paged_ctx_and_check(env, [48], 1, [0])  # fresh, crosses page 1
        _run_paged_ctx_and_check(env, [33], 1, [0])  # 48 -> 81, crosses page 2
        _run_paged_ctx_and_check(env, [15], 1, [0])  # 81 -> 96, exact page end
        _run_paged_ctx_and_check(env, [1], 0, [0])  # decode over the same pool
        env.add_request(1, 23)
        env.add_request(2, 57)
        _run_paged_ctx_and_check(env, [40], 1, [2])  # a second cached prefix
        _run_paged_ctx_and_check(env, [23, 17, 1], 2, [1, 2, 0])
        for rid in (0, 1, 2):
            env.check_cache(rid)
        env.check_unwritten_pool_zero()


def test_bf16_paged_context_prefix_page_geometry() -> None:
    """Cached prefixes against the 32-token page grid, six of them advanced in
    one call so the batch mixes them: a one-token prefix, one ending mid-page
    (31), ones ending exactly on a page boundary (32 and 64), one that has
    just opened a page (33), and a multi-page one (200). Each sequence's new
    tokens land its KV range on a different side of a boundary."""
    torch.manual_seed(85)
    prefixes = [1, 31, 32, 33, 64, 200]
    new_tokens = [1, 2, 32, 31, 40, 40]
    env = _PagedAttnEnv(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_blocks=96,
        max_batch=len(prefixes),
        max_blocks_per_seq=16,
    )
    for rid, prefix in enumerate(prefixes):
        env.add_request(rid, prefix)
        _run_paged_ctx_and_check(env, [prefix], 1, [rid])  # fresh prefill
    _run_paged_ctx_and_check(env, new_tokens, len(new_tokens), list(range(len(prefixes))))
    for rid in range(len(prefixes)):
        env.check_cache(rid)
    env.check_unwritten_pool_zero()


def test_bf16_paged_context_chunked_prefill_loop() -> None:
    """Chunked prefill: two sequences advanced together over three calls, one
    on page-aligned chunks (32/32/17) and one on ragged ones (20/45/16), so
    every call is a context call over whatever each sequence has cached so
    far. Every chunk is gated against the full-history reference and the pool
    ends up holding both prompts bit-exactly."""
    torch.manual_seed(86)
    env = _PagedAttnEnv(
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_blocks=96,
        max_blocks_per_seq=16,
    )
    env.add_request(0, 81)
    env.add_request(1, 81)
    for chunk_a, chunk_b in ((32, 20), (32, 45), (17, 16)):
        _run_paged_ctx_and_check(env, [chunk_a, chunk_b], 2, [0, 1])
    env.check_cache(0)
    env.check_cache(1)
    env.check_unwritten_pool_zero()


def test_bf16_paged_context_reads_the_pool_and_batch_state() -> None:
    """What the flag actually moves, and what the caller then owes it.

    1. The K/V source moves to the paged pool. Aliasing every page of a
       *fresh* prefill onto one loud decoy leaves the packed path's output
       bitwise identical — it never reads the pool — and moves the paged
       path's far outside the band.
    2. A cached prefix is read from the pool: swapping a page holding it for
       the decoy changes the output, while the page holding only this call's
       own new tokens is invisible (the append lands wherever the offsets
       point and is read straight back from there).
    3. use_paged_context_fmha=False on a context call with a cached prefix is
       silently wrong — no exception, an answer far off the reference.
    4. context_lengths on a context row must be this call's new-token count.
       The sequence's full KV length and 0 are both far off.

    The wrong batch states of (3) and (4) also corrupt the pool, so every
    rival runs against its own freshly prepared copy of one fixed state:
    request 0 with 64 cached tokens, advanced by a 20-token context call.
    """
    # (1) which memory the context FMHA reads
    for use_paged in (False, True):
        torch.manual_seed(87)
        env = _PagedAttnEnv(num_heads=32, num_kv_heads=8, head_dim=128)
        env.add_request(0, 40)
        qkv = env.random_qkv(40)
        base = env.call_op(
            qkv,
            [40],
            1,
            [0],
            MASK_CAUSAL,
            record=False,
            use_paged_context_fmha=use_paged,
        )
        decoy = env.kv_cache.shape[0] - 1
        env.kv_cache[decoy] = 7.0
        keep, env.pages[0] = env.pages[0], [decoy] * len(env.pages[0])
        aliased = env.call_op(
            qkv,
            [40],
            1,
            [0],
            MASK_CAUSAL,
            record=False,
            use_paged_context_fmha=use_paged,
        )
        env.pages[0] = keep
        if use_paged:
            _assert_far_outside(aliased, base, "paged context over aliased pages")
        else:
            assert _bitwise_equal(aliased, base), "the packed path must not read the pool at all"

    cached, new = 64, 20

    def prepared() -> Tuple[_PagedAttnEnv, torch.Tensor]:
        torch.manual_seed(88)
        env = _PagedAttnEnv(num_heads=32, num_kv_heads=8, head_dim=128)
        env.add_request(0, cached + new)
        env.call_op(
            env.random_qkv(cached),
            [cached],
            1,
            [0],
            MASK_CAUSAL,
            ctx_lens_override=[cached],
            use_paged_context_fmha=True,
        )
        return env, env.random_qkv(new)

    # The baseline is the correct answer, gated against fp32 here so every
    # separation below is measured against a result that is known right.
    env, qkv = prepared()
    base = _paged_ctx_call(env, qkv, [new], 1, [0])
    torch.testing.assert_close(
        base,
        env.reference(qkv, [new], [0], [cached], MASK_CAUSAL),
        rtol=RTOL,
        atol=ATOL,
    )
    env.check_cache(0)
    env.check_unwritten_pool_zero()

    # (3) the flag itself, and (4) the context length
    penv, pq = prepared()
    assert _bitwise_equal(pq, qkv), "the prepared state is not reproducible"
    _assert_far_outside(
        penv.call_op(
            pq,
            [new],
            1,
            [0],
            MASK_CAUSAL,
            record=False,
            ctx_lens_override=[new],
            use_paged_context_fmha=False,
        ),
        base,
        "cached-prefix context at use_paged_context_fmha=False",
    )
    for ctx_len in (cached + new, 0):
        cenv, cq = prepared()
        _assert_far_outside(
            cenv.call_op(
                cq,
                [new],
                1,
                [0],
                MASK_CAUSAL,
                record=False,
                ctx_lens_override=[ctx_len],
                use_paged_context_fmha=True,
            ),
            base,
            f"context_lengths={ctx_len}",
        )

    # (2) page by page. Pages 0-1 hold the cached prefix and must be read;
    # page 2 holds only tokens 64..83, which this call appends itself.
    last_cached_page = (cached - 1) // env.tokens_per_block
    denv, dq = prepared()
    decoy = denv.kv_cache.shape[0] - 1
    for page in range(len(denv.pages[0])):
        denv.kv_cache[decoy] = 7.0
        keep, denv.pages[0][page] = denv.pages[0][page], decoy
        swapped = _paged_ctx_call(denv, dq, [new], 1, [0], record=False)
        denv.pages[0][page] = keep
        if page <= last_cached_page:
            _assert_far_outside(swapped, base, f"cached prefix page {page} replaced")
        else:
            assert _bitwise_equal(swapped, base), (
                f"page {page} holds only this call's own new tokens, yet "
                f"redirecting it changed the output"
            )


def test_bf16_paged_context_multilayer_shared_pool() -> None:
    """The paged-context read goes through the same layer base shift the
    append does.

    One pool shared by 4 layers — real KVCacheManager state, GQA 32q/8kv d128
    — with every layer served a fresh context chunk, then a context chunk over
    its own cached prefix, then a decode step. Each layer's output is gated
    against that layer's own history, so a read landing in a sibling's slabs
    cannot pass; the appends stay bit-exact and no call touches another
    layer's slabs."""
    torch.manual_seed(89)
    num_layers = 4
    env = _MultiLayerPagedAttnEnv(
        num_layers=num_layers,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        num_blocks=96,
        max_seq_len=512,
    )
    env.add_request(0, 96)  # the manager allocates the whole prompt up front
    env.add_request(1, 40)
    env.refresh_offsets([0, 1], num_contexts=2)
    for chunk in ([64, 25], [32, 15]):
        for layer in range(num_layers):
            cached = [env.cached_len(layer, 0), env.cached_len(layer, 1)]
            qkv = env.random_qkv(sum(chunk))
            siblings_before = [env.layer_views[m].clone() for m in range(num_layers) if m != layer]
            out = env.call_op(
                layer,
                qkv,
                chunk,
                2,
                [0, 1],
                use_paged_context_fmha=True,
                ctx_lens=chunk,
            )
            ref = env.reference(layer, qkv, chunk, [0, 1], cached)
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
            siblings_after = [env.layer_views[m] for m in range(num_layers) if m != layer]
            for before, after in zip(siblings_before, siblings_after):
                assert torch.equal(before, after)  # no cross-layer write
    env.check_caches([0, 1])

    env.add_decode_token(0)
    env.add_decode_token(1)
    env.refresh_offsets([0, 1], num_contexts=0)
    for layer in range(num_layers):
        cached = [env.cached_len(layer, 0), env.cached_len(layer, 1)]
        qkv_d = env.random_qkv(2)
        out_d = env.call_op(layer, qkv_d, [1, 1], 0, [0, 1], use_paged_context_fmha=True)
        ref_d = env.reference(layer, qkv_d, [1, 1], [0, 1], cached)
        torch.testing.assert_close(out_d, ref_d, rtol=RTOL, atol=ATOL)
    env.check_caches([0, 1])


def test_bf16_paged_context_gpt_oss_sinks_and_window() -> None:
    """The gpt-oss-120b tp1 cell on this path: 64q/8kv d64 with per-head
    sinks, first as a full-attention layer, then as a sliding one (window 128)
    whose cached prefix runs well past the window.

    Both mechanisms are already certified on the packed path; what is new is
    that the keys they weigh now come out of the pool and the mask is
    bottom-right aligned. Every case is gated against the sink-ignored and
    sink-pre-scaled rivals, the windowed ones additionally against a
    window-ignored (full causal) rival."""
    torch.manual_seed(90)
    sink = _sinks(-2.0, 4.0, seed=90)
    env = _sink_env()
    env.add_request(0, 81)
    qkv = env.random_qkv(48)
    out = _paged_ctx_call(env, qkv, [48], 1, [0], attention_sinks=sink)
    _check_sink_case(env, qkv, out, [48], [0], [0], sink, "paged fresh context")
    qkv2 = env.random_qkv(33)
    out2 = _paged_ctx_call(env, qkv2, [33], 1, [0], attention_sinks=sink)
    _check_sink_case(env, qkv2, out2, [33], [0], [48], sink, "paged cached context")
    env.check_cache(0)
    env.check_unwritten_pool_zero()

    torch.manual_seed(91)
    # Sinks near log(window), as on the packed sliding-window cases: against a
    # full 128-key softmax a smaller sink is negligible and the sink-ignored
    # separation gate would be vacuous rather than the semantics different.
    swa_sink = _sinks(math.log(SWA_W) - 1.0, math.log(SWA_W) + 2.0, seed=91)
    swa = _swa_env(num_blocks=96, max_blocks_per_seq=16)
    swa.add_request(0, 240)
    q_pre = swa.random_qkv(200)  # 200 > 128: the window bites inside the prefill
    out_pre = _paged_ctx_call(swa, q_pre, [200], 1, [0], attention_sinks=swa_sink)
    _check_sink_case(
        swa,
        q_pre,
        out_pre,
        [200],
        [0],
        [0],
        swa_sink,
        "paged swa prefill",
        window=SWA_W,
    )
    q_new = swa.random_qkv(40)
    out_new = _paged_ctx_call(swa, q_new, [40], 1, [0], attention_sinks=swa_sink)
    _check_sink_case(
        swa,
        q_new,
        out_new,
        [40],
        [0],
        [200],
        swa_sink,
        "paged swa cached context",
        window=SWA_W,
    )
    swa.check_cache(0)


def test_bf16_paged_context_window_boundary_and_read_set() -> None:
    """Which keys a cached-prefix context row attends to, read out exactly,
    and which pages the paged read touches.

    The one-hot probe (all-zero K, one-hot V, both cached and new) turns the
    output into the attention weights: for a query row at absolute position p
    every key outside [p - W + 1, p] comes back bitwise zero and the ones
    inside all carry 1 / (n + exp(sink_h)) with n = min(p + 1, W). That pins
    the mask as bottom-right aligned over the cached prefix, and as a token
    count rather than a page count.

    The read set follows, measured page by page against a decoy: a page
    changes the output exactly when it holds an in-window key this call does
    not write itself, i.e. an in-window *cached* token. Pages fully below the
    window are unread (recyclable), and so are pages holding only this call's
    new tokens, whose append lands wherever the offsets point. Under the
    packed path a context call read no page at all."""
    torch.manual_seed(92)
    sink = _sinks(-1.0, 3.0, seed=92)
    prefill, new, t0 = 150, 40, 66
    env = _swa_env(num_blocks=96, max_blocks_per_seq=16)
    env.add_request(0, prefill + new)
    _paged_ctx_call(
        env,
        _swa_indicator_qkv(prefill, list(range(prefill)), t0),
        [prefill],
        1,
        [0],
        attention_sinks=sink,
    )
    probe = _swa_indicator_qkv(new, list(range(prefill, prefill + new)), t0)
    out = _paged_ctx_call(env, probe, [new], 1, [0], attention_sinks=sink)
    for row in (0, 10, new - 1):
        _assert_window_weights(out, row, prefill + row, t0, sink, f"paged cached context row={row}")

    torch.manual_seed(93)
    tpb, cached = 32, 200
    renv = _swa_env(tokens_per_block=tpb, num_blocks=96, max_blocks_per_seq=16)
    renv.add_request(0, cached + 40)
    _run_paged_ctx_and_check(renv, [cached], 1, [0], window=SWA_W)
    qkv = renv.random_qkv(40)
    base = _paged_ctx_call(renv, qkv, [40], 1, [0], record=False)
    decoy = renv.kv_cache.shape[0] - 1
    # This call's oldest in-window key belongs to its first query row, at
    # absolute position `cached`; its last cached key is `cached - 1`.
    first_read = max(0, cached + 1 - SWA_W) // tpb
    last_cached = (cached - 1) // tpb
    for page in range(len(renv.pages[0])):
        renv.kv_cache[decoy] = 7.0
        keep, renv.pages[0][page] = renv.pages[0][page], decoy
        got = _paged_ctx_call(renv, qkv, [40], 1, [0], record=False)
        renv.pages[0][page] = keep
        if first_read <= page <= last_cached:
            _assert_far_outside(got, base, f"in-window cached page {page} replaced")
        else:
            assert _bitwise_equal(got, base), (
                f"page {page} holds no in-window cached token, yet the decoy reached the output"
            )
    assert _bitwise_equal(_paged_ctx_call(renv, qkv, [40], 1, [0], record=False), base), (
        "the page-swap probe left the pool in a different state"
    )


# ─── MLA configuration (is_mla_enable=True) ────────────────────────────
# DeepSeek MLA head geometry at four query-head counts: 32, the
# deepseek-v3-lite tp1 layer shape; 16, its tp2 slice; 8, its tep4 slice
# (32 checkpoint heads split over 4 tensor-parallel ranks); and 128, the
# DeepSeek-R1-0528 layer shape, which attention DP replicates whole onto
# every rank. C/R/nope/v are identical for all four — only the head count
# moves, so every case below runs at each count.
MLA_NUM_HEADS = 16
MLA_NUM_HEADS_H32 = 32
MLA_NUM_HEADS_H8 = 8
MLA_NUM_HEADS_H128 = 128
KV_LORA_RANK = 512  # C
QK_ROPE_HEAD_DIM = 64  # R
QK_NOPE_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192, context head size
V_HEAD_DIM = 128  # context v head dim
LATENT_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576, generation head size
# Latent-pool page size. 32 is the engine default (KvCacheConfig
# .tokens_per_block), 64 the value a tuned MLA target opts into. Both are
# covered; the page count per sequence is scaled so max_seq_len stays 1024
# either way, leaving the page size as the only moving axis.
MLA_TOKENS_PER_BLOCK = 64
MLA_PAGE32 = 32
MLA_MAX_SEQ_LEN = 1024
POSITION_EMBEDDING_TYPE_YARN = 8  # PositionEmbeddingType.yarn (MLA models)
# q-LoRA rank. DeepSeek-V3 down-projects q through a rank-1536 q_a_proj;
# checkpoints with "q_lora_rank": null (deepseek-v3-lite) have no q-LoRA at
# all and pass 0. Both are covered — see the inertness cases at the end.
Q_LORA_RANK_DSV3 = 1536
Q_LORA_RANK_ZERO = 0

# Both MLA phases scale QK^T by 1 / (q_scaling * sqrt(QK_HEAD_DIM)) — the
# generation phase too, despite its head_size being C+R.
MLA_SOFTMAX_SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)

ROTARY_SCALING_TYPE_NONE = 0  # RotaryScalingType.none
ROTARY_SCALING_TYPE_YARN = 5  # RotaryScalingType.yarn

# DeepSeek-R1-0528's rope configuration, from its config.json: theta 10000,
# YaRN over an original 4096-position window with factor 40, beta_fast 32 /
# beta_slow 1, both mscales 1.0, 163840 trained positions.
R1_ROPE_THETA = 10000.0
R1_ROPE_FACTOR = 40.0
R1_ROPE_ORIGINAL_MAX_POSITIONS = 4096
R1_ROPE_BETA_FAST = 32
R1_ROPE_BETA_SLOW = 1
R1_ROPE_MSCALE = 1.0
R1_ROPE_MSCALE_ALL_DIM = 1.0
R1_MAX_POSITION_EMBEDDINGS = 163840

# YaRN's attention-temperature term: mscale = 0.1 * mscale_cfg * ln(factor) + 1
# at factor > 1. R1 folds mscale^2 into the softmax scale rather than into the
# rope table (the table's own amplitude factor is
# yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim) = 1
# when the two config mscales are equal), so the model wants
# 1 / (q_scaling * sqrt(192)) = mscale^2 / sqrt(192).
R1_MSCALE = 0.1 * R1_ROPE_MSCALE * math.log(R1_ROPE_FACTOR) + 1.0
R1_Q_SCALING = 1.0 / (R1_MSCALE * R1_MSCALE)


class _RopeScalars(NamedTuple):
    """The op's seven scalar rope arguments, carried as one group.

    They arrive alongside the rotary_cos_sin table, which already encodes
    theta and any scaling in its content. Whether the op reads them at all on
    the MLA path is what the scalar sweep at the end of the MLA section
    settles; every MLA case here passes a set explicitly so the answer is
    never assumed.
    """

    rope_base: float = 10000.0
    rope_scale_type: int = ROTARY_SCALING_TYPE_NONE
    rope_scale: float = 1.0
    rope_short_m_scale: float = 1.0
    rope_long_m_scale: float = 1.0
    rope_max_positions: int = MLA_MAX_SEQ_LEN
    rope_original_max_positions: int = MLA_MAX_SEQ_LEN


# What a TrtllmAttention module forwards for a DeepSeek-R1-0528 layer:
# rope_params.theta / scale_type / scale / max_positions /
# original_max_positions straight off the checkpoint's rope_scaling block.
R1_ROPE_SCALARS = _RopeScalars(
    rope_base=R1_ROPE_THETA,
    rope_scale_type=ROTARY_SCALING_TYPE_YARN,
    rope_scale=R1_ROPE_FACTOR,
    rope_max_positions=R1_MAX_POSITION_EMBEDDINGS,
    rope_original_max_positions=R1_ROPE_ORIGINAL_MAX_POSITIONS,
)


def _r1_rope_params(max_positions: int) -> RopeParams:
    """The rope construction a DeepSeek-R1-0528 config produces, truncated to
    max_positions rows. Row content is independent of the row count (position
    p contributes t = p against a position-independent inv_freq), so this is
    the leading slice of the 163840-row table an engine builds."""
    return RopeParams(
        dim=QK_ROPE_HEAD_DIM,
        theta=R1_ROPE_THETA,
        scale_type=RotaryScalingType.yarn,
        scale=R1_ROPE_FACTOR,
        max_positions=max_positions,
        original_max_positions=R1_ROPE_ORIGINAL_MAX_POSITIONS,
        beta_fast=R1_ROPE_BETA_FAST,
        beta_slow=R1_ROPE_BETA_SLOW,
        mscale=R1_ROPE_MSCALE,
        mscale_all_dim=R1_ROPE_MSCALE_ALL_DIM,
        duplicate_data=True,
    )


# Gate for the appended latent rows' k_pe half. The kernel rotates in fp32 and
# stores bf16; the reference rotates in fp32 with torch's evaluation order. The
# two fp32 results differ by one fp32 ulp on values whose correctly-rounded
# fp32 form lands exactly on a bf16 rounding midpoint, which flips the stored
# bf16 by one ulp. Measured on sm_100 over 60 prefill runs at page sizes 32 and
# 64 (495360 roped elements): 8 such elements, 1.6e-5 of the total, never more
# than one ulp, and the same count at both page sizes — so it is an arithmetic
# tie, not an addressing effect. rtol 2**-7 covers exactly +/-1 bf16 ulp at any
# magnitude; the bit-exact-fraction floor keeps the claim "essentially bitwise"
# rather than merely "within an ulp". Wrong variants measured against this gate
# on one 96-token page-32 request (6144 roped elements): an append shifted one
# slot 6.1e5 ulp / 99.9% of elements differing, the pool read with
# tokens_per_block=64 instead of 32 2.6e5 ulp / 33%, rope applied at position
# i+1 5.1e5 ulp / 69%, k_pe left un-roped 4.4e5 ulp / 93%, another request's
# rows 4.1e5 ulp / 99.9% — every one of them >=2.6e5x past the ulp gate and
# >=330x past the fraction gate, and all but the rope-position variants also
# break the strictly bitwise compressed_kv half.
LATENT_ROPE_ULP_RTOL = 2**-7
LATENT_ROPE_MAX_INEXACT_FRACTION = 1e-3


def _assert_latent_rope(got: torch.Tensor, expected: torch.Tensor, request_id: int) -> None:
    """Dual gate on the roped k_pe half of a request's appended latent rows."""
    a = got[:, KV_LORA_RANK:].float()
    b = expected[:, KV_LORA_RANK:].float()
    torch.testing.assert_close(a, b, rtol=LATENT_ROPE_ULP_RTOL, atol=0.0)
    inexact = int((a != b).sum())
    allowed = max(1, int(LATENT_ROPE_MAX_INEXACT_FRACTION * a.numel()))
    assert inexact <= allowed, (
        f"latent cache rope not bit-exact enough: request {request_id}, "
        f"{inexact}/{a.numel()} elements differ (allowed {allowed})"
    )


# The same gate one e4m3 ulp wide, for an fp8 pool. e4m3 keeps 3 mantissa
# bits, so one ulp is 2**-3 relative — coarse enough to swallow the fp32
# evaluation-order difference that costs the bf16 pool its 8-in-495360
# elements: every appended row measured on sm_100 under quant_mode 128 (both
# halves, scales 1.0 / 1.5 / 2.0, H = 16 and 128, page-crossing sequences) was
# bit-exact against e4m3(row * kv_scale_orig_quant), so this gate has never
# been approached. It stays a gate rather than a bitwise assert because the
# rounding tie it covers is arithmetic, not addressing: a mis-addressed append
# lands orders of magnitude outside it (see LATENT_ROPE_ULP_RTOL's wrong-variant
# figures, which the fp8 append shares — the addressing is the same code).
LATENT_ROPE_E4M3_ULP_RTOL = 2**-3


def _assert_latent_rope_e4m3(got: torch.Tensor, expected: torch.Tensor, request_id: int) -> None:
    """One-e4m3-ulp gate plus a bit-exact-fraction floor on the roped k_pe half
    of a request's appended latent rows in an fp8 pool."""
    a = got[:, KV_LORA_RANK:].float()
    b = expected[:, KV_LORA_RANK:].float()
    torch.testing.assert_close(a, b, rtol=LATENT_ROPE_E4M3_ULP_RTOL, atol=0.0)
    inexact = int(
        (
            got[:, KV_LORA_RANK:].view(torch.uint8) != expected[:, KV_LORA_RANK:].view(torch.uint8)
        ).sum()
    )
    allowed = max(1, int(LATENT_ROPE_MAX_INEXACT_FRACTION * a.numel()))
    assert inexact <= allowed, (
        f"fp8 latent cache rope not bit-exact enough: request {request_id}, "
        f"{inexact}/{a.numel()} elements differ (allowed {allowed})"
    )


class _MlaPagedEnv:
    """Real MLA op state: caller-owned paged latent pool (kv_factor 1) +
    explicit metadata tensors + the duplicated-layout MLA RoPE table.

    Mirrors the expected latent-cache rows per request (computed with test-
    side torch math, never read back from the op), so decode references can
    be built from the true cache content and the pool can be checked
    bitwise against it.
    """

    def __init__(
        self,
        num_heads: int = MLA_NUM_HEADS,
        num_blocks: int = 64,
        max_batch: int = 4,
        max_blocks_per_seq: int = 16,
        tokens_per_block: int = MLA_TOKENS_PER_BLOCK,
        q_lora_rank: Optional[int] = Q_LORA_RANK_DSV3,
        q_scaling: float = 1.0,
        rope: Optional[RopeParams] = None,
        rope_scalars: Optional[_RopeScalars] = None,
        pool_dtype: torch.dtype = torch.bfloat16,
        quant_mode: int = 0,
        kv_scaling_factor: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.max_batch = max_batch
        self.tokens_per_block = tokens_per_block
        self.q_lora_rank = q_lora_rank
        self.max_seq_len = max_blocks_per_seq * tokens_per_block
        self.q_scaling = q_scaling
        self.quant_mode = quant_mode
        # KV-cache scaling factor s (fp8 pool only): a latent row lands in the
        # pool as e4m3(row * kv_scale_orig_quant) with orig_quant = 1/s, and
        # dequantizes as value * kv_scale_quant_orig with quant_orig = s. None
        # leaves both tensors unpassed, which the op reads as s = 1.0.
        self.kv_scale = 1.0 if kv_scaling_factor is None else kv_scaling_factor
        if kv_scaling_factor is None:
            self.kv_scale_quant_orig: Optional[torch.Tensor] = None
            self.kv_scale_orig_quant: Optional[torch.Tensor] = None
        else:
            factor = torch.full((1,), kv_scaling_factor, dtype=torch.float32, device="cuda")
            self.kv_scale_quant_orig = factor
            self.kv_scale_orig_quant = 1.0 / factor
        # Both MLA phases scale QK^T by 1 / (q_scaling * sqrt(nope + R)).
        self.softmax_scale = 1.0 / (q_scaling * math.sqrt(QK_HEAD_DIM))

        # Single-layer, single-pool paged MLA latent cache: kv_factor 1, one
        # kv head, row width C+R. Page p occupies pool[p] — one slab of
        # tokens_per_block * (C+R) elements, one byte each under quant_mode
        # 128 (e4m3) and two under quant_mode 0 (bf16).
        self.pool = torch.zeros(
            num_blocks,
            tokens_per_block,
            LATENT_DIM,
            dtype=pool_dtype,
            device="cuda",
        )
        self.pool_pointers = torch.zeros(1, 2, dtype=torch.int64, device="cpu")
        self.pool_pointers[0, 0] = self.pool.data_ptr()
        self.pool_mapping = torch.zeros(1, 2, dtype=torch.int32, device="cpu")
        self.block_offsets = torch.zeros(
            1, max_batch, 2, max_blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        # Auto-resized in place by the op on first call.
        self.workspace = torch.empty(0, dtype=torch.int8, device="cuda")

        # Duplicated-layout fp32 (cos, sin) table, as the MLA backend builds
        # it (RopeParams.from_config sets duplicate_data=True for MLA models).
        # Default: the unscaled theta-10000 table. rope, when given, supplies
        # another construction — the YaRN one a DeepSeek-R1 config produces.
        rope = rope or RopeParams(
            dim=QK_ROPE_HEAD_DIM,
            theta=10000.0,
            max_positions=self.max_seq_len,
            duplicate_data=True,
        )
        self.rotary_inv_freq, self.rotary_cos_sin = rope.create_rope_const_params()
        self.rope_scalars = rope_scalars or _RopeScalars(
            rope_max_positions=self.max_seq_len,
            rope_original_max_positions=self.max_seq_len,
        )

        self._next_free_page = 0
        self.pages: Dict[int, List[int]] = {}
        self.prompt_lens: Dict[int, int] = {}
        # Expected cache rows per request, [n, C+R] each, test-computed.
        self.latent_rows: Dict[int, List[torch.Tensor]] = {}

    def add_request(self, request_id: int, prompt_len: int) -> None:
        self.pages[request_id] = []
        self.prompt_lens[request_id] = prompt_len
        self.latent_rows[request_id] = []

    def cached_len(self, request_id: int) -> int:
        return sum(t.shape[0] for t in self.latent_rows[request_id])

    @property
    def fp8_pool(self) -> bool:
        return self.pool.dtype == torch.float8_e4m3fn

    def quantize_e4m3(self, x: torch.Tensor) -> torch.Tensor:
        """e4m3(x * kv_scale_orig_quant) — the op's write-side quantization,
        and the same one the caller applies to the decode query. One
        expression so the mirror, the reference and the buffer the op reads
        cannot disagree in the last bit."""
        return (x.float() * (1.0 / self.kv_scale)).to(torch.float8_e4m3fn)

    def to_pool(self, rows: torch.Tensor) -> torch.Tensor:
        """The bytes a latent row lands as in the pool: e4m3(row * orig_quant)
        for an fp8 pool, the bf16 row itself otherwise."""
        return self.quantize_e4m3(rows) if self.fp8_pool else rows

    def e4m3_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """fp32 value -> e4m3 at the pool's write scale -> fp32 again: what a
        value looks like once it has been through the fp8 pool (or through the
        caller-side q quantization, which uses the same scale)."""
        return self.quantize_e4m3(x).float() * self.kv_scale

    def fp8_decode_buffers(
        self, fused_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three buffers an fp8-pool MLA generation call requires, filled
        the way the production generation-preprocessing step fills them (values
        read back from torch.ops.trtllm.mla_rope_generation in a probe, rebuilt
        here in plain torch):

        - quant_q_buffer: e4m3(fused_q * kv_scale_orig_quant) — the query the
          decode kernel actually reads (`q` itself is not read);
        - mla_bmm1_scale: [x, x * log2(e)] with x = softmax_scale * s**2, the
          s**2 undoing the 1/s the query and the pool rows were quantized by.
          Only element [1] (the log2-domain copy) is read by the kernel;
        - mla_bmm2_scale: [s], undoing the 1/s of the V (= pool row) side.
        """
        num_gen = fused_q.shape[0]
        quant_q = self.quantize_e4m3(fused_q.view(num_gen, self.num_heads, LATENT_DIM)).contiguous()
        bmm1 = self.softmax_scale * self.kv_scale * self.kv_scale
        return (
            quant_q,
            torch.tensor([bmm1, bmm1 * math.log2(math.e)], dtype=torch.float32, device="cuda"),
            torch.tensor([self.kv_scale], dtype=torch.float32, device="cuda"),
        )

    def _ensure_pages(self, request_id: int, total_tokens: int) -> None:
        tpb = self.tokens_per_block
        needed = (total_tokens + tpb - 1) // tpb
        pages = self.pages[request_id]
        while len(pages) < needed:
            pages.append(self._next_free_page)
            self._next_free_page += 1

    def rope_ref(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """GPT-J interleaved rotation of the last dim, fp32 math, bf16 result."""
        half = QK_ROPE_HEAD_DIM // 2
        table = self.rotary_cos_sin.view(-1, QK_ROPE_HEAD_DIM, 2)
        cos = table[position, :half, 0]
        sin = table[position, :half, 1]
        pairs = x.float().reshape(*x.shape[:-1], half, 2)
        out = torch.empty_like(pairs)
        out[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        out[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        return out.reshape(x.shape).to(x.dtype)

    def append_decode_latent(self, request_id: int, row: torch.Tensor) -> None:
        """Write one decode token's latent row into the pool, standing in for
        the generation-preprocessing append the op itself does not perform."""
        pos = self.cached_len(request_id)
        self._ensure_pages(request_id, pos + 1)
        tpb = self.tokens_per_block
        page = self.pages[request_id][pos // tpb]
        self.pool[page, pos % tpb] = self.to_pool(row.unsqueeze(0))[0]
        self.latent_rows[request_id].append(row.unsqueeze(0))

    def _fill_offsets(self, request_ids: List[int]) -> None:
        # kv_factor-1 pool: raw block id in both the K and the V row.
        for s, rid in enumerate(request_ids):
            row = self.block_offsets[0, s]
            for j, p in enumerate(self.pages[rid]):
                row[0, j] = p
                row[1, j] = p

    def _call_op(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        kv_lens: List[int],
        ctx_lens: List[int],
        num_contexts: int,
        num_ctx_tokens: int,
        attention_input_type: int,
        head_size: int,
        num_kv_heads: int,
        v_head_dim: int,
        latent_cache: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        cu_q_seqlens: Optional[torch.Tensor],
        cu_kv_seqlens: Optional[torch.Tensor],
        fmha_scheduler_counter: Optional[torch.Tensor],
        mask_type: int = MASK_CAUSAL,
        softmax_stats_tensor: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        predicted_tokens_per_seq: int = 1,
        host_past_lens: Optional[List[int]] = None,
    ) -> None:
        ns = len(kv_lens)
        req_types = [0 if i < num_contexts else 1 for i in range(ns)]
        thop_attention(
            q=q,
            k=k,
            v=v,
            output=output,
            output_sf=None,
            workspace_=self.workspace,
            sequence_length=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
            host_past_key_value_lengths=torch.tensor(
                kv_lens if host_past_lens is None else host_past_lens,
                dtype=torch.int32,
            ),
            host_total_kv_lens=torch.tensor(
                [sum(kv_lens[:num_contexts]), sum(kv_lens[num_contexts:])],
                dtype=torch.int32,
            ),
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
            rotary_inv_freq=self.rotary_inv_freq,
            rotary_cos_sin=self.rotary_cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=None,
            attention_sinks=None,
            is_fused_qkv=k is None,
            update_kv_cache=True,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            local_layer_idx=0,
            num_heads=self.num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tokens_per_block=self.tokens_per_block,
            max_num_requests=self.max_batch,
            max_context_length=self.max_seq_len,
            max_seq_len=self.max_seq_len,
            attention_window_size=self.max_seq_len,
            beam_width=1,
            mask_type=mask_type,
            quant_mode=self.quant_mode,
            q_scaling=self.q_scaling,
            position_embedding_type=POSITION_EMBEDDING_TYPE_YARN,
            rope_dim=QK_ROPE_HEAD_DIM,
            rope_base=self.rope_scalars.rope_base,
            rope_scale_type=self.rope_scalars.rope_scale_type,
            rope_scale=self.rope_scalars.rope_scale,
            rope_short_m_scale=self.rope_scalars.rope_short_m_scale,
            rope_long_m_scale=self.rope_scalars.rope_long_m_scale,
            rope_max_positions=self.rope_scalars.rope_max_positions,
            rope_original_max_positions=self.rope_scalars.rope_original_max_positions,
            use_paged_context_fmha=False,
            attention_input_type=attention_input_type,
            is_mla_enable=True,
            chunked_prefill_buffer_batch_size=1,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=v_head_dim,
            rope_append=True,
            mrope_rotary_cos_sin=None,
            mrope_position_deltas=None,
            helix_position_offsets=None,
            helix_is_inactive_rank=None,
            attention_chunk_size=None,
            softmax_stats_tensor=softmax_stats_tensor,
            is_spec_decoding_enabled=False,
            use_spec_decoding=False,
            is_spec_dec_tree=False,
            spec_decoding_generation_lengths=None,
            spec_decoding_position_offsets_for_cpp=None,
            spec_decoding_packed_mask=None,
            spec_decoding_bl_tree_mask_offset=None,
            spec_decoding_bl_tree_mask=None,
            spec_bl_tree_first_sparse_mask_offset_kv=None,
            sparse_kv_indices=None,
            sparse_kv_offsets=None,
            sparse_attn_indices=None,
            sparse_attn_offsets=None,
            sparse_attn_indices_block_size=0,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
        )
        torch.cuda.synchronize()

    def call_context(
        self,
        request_ids: List[int],
        seq_lens: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        latent_cache: torch.Tensor,
        gen_request_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """context_only call over fresh context sequences (positions 0..len-1).

        gen_request_ids, if given, are generation-phase sequences sharing the
        batch metadata (their q rows belong to a separate generation_only
        call). Records the expected latent rows [ckv | rope(k_pe)] per
        context request.
        """
        gen_request_ids = gen_request_ids or []
        num_contexts = len(request_ids)
        num_ctx_tokens = sum(seq_lens)
        for rid, ln in zip(request_ids, seq_lens):
            assert self.cached_len(rid) == 0, "certified MLA context starts empty"
            self._ensure_pages(rid, ln)
        all_ids = request_ids + gen_request_ids
        self._fill_offsets(all_ids)
        kv_lens = list(seq_lens) + [self.cached_len(r) for r in gen_request_ids]
        ctx_lens = list(seq_lens) + [self.prompt_lens[r] for r in gen_request_ids]

        output = torch.empty(
            num_ctx_tokens,
            self.num_heads * V_HEAD_DIM,
            dtype=q.dtype,
            device=q.device,
        )
        self._call_op(
            q=q,
            k=k,
            v=v,
            output=output,
            kv_lens=kv_lens,
            ctx_lens=ctx_lens,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            attention_input_type=1,  # context_only
            head_size=QK_HEAD_DIM,
            num_kv_heads=self.num_heads,  # context runs as MHA
            v_head_dim=V_HEAD_DIM,
            latent_cache=latent_cache,
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
        )

        # Record the expected appended rows: [ckv | rope_pos(k_pe)].
        start = 0
        for rid, ln in zip(request_ids, seq_lens):
            rows = latent_cache[start : start + ln].clone()
            for i in range(ln):
                rows[i, KV_LORA_RANK:] = self.rope_ref(rows[i, KV_LORA_RANK:], i)
            self.latent_rows[rid].append(rows)
            start += ln
        return output

    def reserve_cache_pages(self, request_id: int, total_tokens: int) -> None:
        """Pre-allocate page capacity for a request's full [cached + new] KV
        range. The latent_cache=None context calls never touch the pool, but
        production always runs them with the pages already allocated."""
        self._ensure_pages(request_id, total_tokens)

    def call_context_no_append(
        self,
        request_ids: List[int],
        new_lens: List[int],
        pass_kv_lens: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_type: int = MASK_CAUSAL,
        softmax_stats_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """context_only call with latent_cache=None (cached-KV / chunked
        context): no in-kernel RoPE and no cache append — q arrives
        pre-rotated, K/V arrive explicitly with pass_kv_lens[s] rows per
        sequence. sequence_length carries pass_kv_lens while context_lengths
        stays at the new-token (q) counts, so KV and q lengths differ."""
        num_contexts = len(request_ids)
        num_ctx_tokens = sum(new_lens)
        self._fill_offsets(request_ids)
        output = torch.empty(
            num_ctx_tokens,
            self.num_heads * V_HEAD_DIM,
            dtype=q.dtype,
            device=q.device,
        )
        self._call_op(
            q=q,
            k=k,
            v=v,
            output=output,
            kv_lens=list(pass_kv_lens),
            ctx_lens=list(new_lens),
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            attention_input_type=1,  # context_only
            head_size=QK_HEAD_DIM,
            num_kv_heads=self.num_heads,  # context runs as MHA
            v_head_dim=V_HEAD_DIM,
            latent_cache=None,  # selects the no-RoPE / no-append context path
            q_pe=None,
            cu_q_seqlens=None,
            cu_kv_seqlens=None,
            fmha_scheduler_counter=None,
            mask_type=mask_type,
            softmax_stats_tensor=softmax_stats_tensor,
        )
        return output

    def call_generation(
        self,
        request_ids: List[int],
        fused_q: torch.Tensor,
        ctx_request_ids: Optional[List[int]] = None,
        ctx_seq_lens: Optional[List[int]] = None,
        fp8_buffers: Optional[
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
        ] = None,
        predicted_tokens_per_seq: int = 1,
        mask_type: int = MASK_CAUSAL,
        cu_q_override: Optional[torch.Tensor] = None,
        cu_kv_override: Optional[torch.Tensor] = None,
        host_past_override: Optional[List[int]] = None,
        ctx_lens_override: Optional[List[int]] = None,
        output_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """generation_only call: latent MQA over the paged cache.

        The new tokens' latent rows must already be in the pool
        (append_decode_latent). latent_cache/q_pe are presence-validated by
        the op but their data is not consumed: filled with garbage here on
        purpose. ctx_request_ids/ctx_seq_lens describe leading context-phase
        sequences sharing the batch metadata (mixed batch, separate call).

        predicted_tokens_per_seq = P gives each generation sequence P query
        rows instead of one (speculative decoding: a draft chain verified in
        one step). The rows are token-major within a sequence, so sequence g
        owns rows [g*P, (g+1)*P), and every per-row tensor is P times taller.

        Over an fp8 pool the call additionally needs the
        (quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale) triple, built by
        fp8_decode_buffers() from fused_q unless fp8_buffers overrides it —
        which is how the probes that pin each buffer's role feed deliberately
        wrong or missing ones. The *_override arguments feed deliberately
        wrong batch state the same way.
        """
        ctx_request_ids = ctx_request_ids or []
        ctx_seq_lens = ctx_seq_lens or []
        num_contexts = len(ctx_request_ids)
        num_gen = len(request_ids)
        p = predicted_tokens_per_seq
        rows = num_gen * p
        all_ids = ctx_request_ids + request_ids
        self._fill_offsets(all_ids)
        gen_kv_lens = [self.cached_len(r) for r in request_ids]
        kv_lens = list(ctx_seq_lens) + gen_kv_lens
        ctx_lens = list(ctx_seq_lens) + [self.prompt_lens[r] for r in request_ids]
        if ctx_lens_override is not None:
            ctx_lens = list(ctx_lens_override)

        # Decode-FMHA scheduler buffers, filled as the generation-phase
        # preprocessing op fills them: q rows / kv tokens cumulated over
        # generation sequences only, counter zeroed. At P > 1 a generation
        # sequence contributes num_heads * P q rows.
        cu_q = torch.arange(num_gen + 1, dtype=torch.int32) * (self.num_heads * p)
        cu_kv = torch.zeros(num_gen + 1, dtype=torch.int32)
        cu_kv[1:] = torch.tensor(gen_kv_lens, dtype=torch.int32).cumsum(0)
        if cu_q_override is not None:
            cu_q = cu_q_override
        if cu_kv_override is not None:
            cu_kv = cu_kv_override
        counter = torch.zeros(1, dtype=torch.uint32, device="cuda")

        garbage_latent = torch.randn(rows, LATENT_DIM, dtype=fused_q.dtype, device="cuda")
        garbage_q_pe = torch.randn(
            rows,
            self.num_heads,
            QK_ROPE_HEAD_DIM,
            dtype=fused_q.dtype,
            device="cuda",
        )
        output = (
            torch.empty(
                rows,
                self.num_heads * KV_LORA_RANK,
                dtype=fused_q.dtype,
                device="cuda",
            )
            if output_buffer is None
            else output_buffer
        )
        if fp8_buffers is None and self.fp8_pool:
            fp8_buffers = self.fp8_decode_buffers(fused_q)
        quant_q, bmm1, bmm2 = fp8_buffers or (None, None, None)
        self._call_op(
            q=fused_q,
            k=None,
            v=None,
            output=output,
            kv_lens=kv_lens,
            ctx_lens=ctx_lens,
            num_contexts=num_contexts,
            num_ctx_tokens=sum(ctx_seq_lens),
            attention_input_type=2,  # generation_only
            head_size=LATENT_DIM,
            num_kv_heads=1,  # latent MQA
            v_head_dim=KV_LORA_RANK,
            latent_cache=garbage_latent,
            q_pe=garbage_q_pe,
            cu_q_seqlens=cu_q.cuda(),
            cu_kv_seqlens=cu_kv.cuda(),
            fmha_scheduler_counter=counter,
            mask_type=mask_type,
            mla_bmm1_scale=bmm1,
            mla_bmm2_scale=bmm2,
            quant_q_buffer=quant_q,
            predicted_tokens_per_seq=p,
            host_past_lens=host_past_override,
        )
        return output

    def context_reference(
        self,
        seq_lens: List[int],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        latent_cache: torch.Tensor,
        softmax_scale: Optional[float] = None,
        e4m3_inputs: bool = False,
        kv_scale_factors: bool = False,
    ) -> torch.Tensor:
        """fp32 causal MHA over per-position-roped q/k, headDimV=V_HEAD_DIM.

        Must be given the pre-call q/k/latent contents (the op clobbers the
        q/k rope slices in place). softmax_scale overrides the env's own
        scale, which is what lets a q_scaling sweep build rival references.

        e4m3_inputs rounds q, k and v through e4m3 at scale 1.0, which is what
        the context FMHA quantizes them to under quant_mode 128 (the pool plays
        no part in the context math; it is written, not read). kv_scale_factors
        adds the two factors that path applies on top — softmax scale * s**2
        and output * s — which cancel only at s = 1.0.
        """
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        out_scale = 1.0
        if kv_scale_factors:
            scale *= self.kv_scale * self.kv_scale
            out_scale = self.kv_scale
        outs = []
        start = 0
        h = self.num_heads
        for ln in seq_lens:
            q_bf = q[start : start + ln].view(ln, h, QK_HEAD_DIM)
            q_seq = q_bf.float()
            k_seq = k[start : start + ln].view(ln, h, QK_HEAD_DIM).float()
            v_seq = v[start : start + ln].view(ln, h, V_HEAD_DIM).float()
            latent = latent_cache[start : start + ln]
            for i in range(ln):
                q_seq[i, :, QK_NOPE_HEAD_DIM:] = self.rope_ref(
                    q_bf[i, :, QK_NOPE_HEAD_DIM:], i
                ).float()
                # k_pe comes from latent_cache, roped and broadcast per head.
                k_seq[i, :, QK_NOPE_HEAD_DIM:] = self.rope_ref(latent[i, KV_LORA_RANK:], i).float()
            if e4m3_inputs:
                q_seq = q_seq.to(torch.float8_e4m3fn).float()
                k_seq = k_seq.to(torch.float8_e4m3fn).float()
                v_seq = v_seq.to(torch.float8_e4m3fn).float()
            scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq) * scale
            mask = torch.tril(torch.ones(ln, ln, dtype=torch.bool, device="cuda"))
            probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
            outs.append(
                (torch.einsum("hij,jhd->ihd", probs, v_seq) * out_scale)
                .reshape(ln, -1)
                .to(torch.bfloat16)
            )
            start += ln
        return torch.cat(outs)

    def generation_reference(
        self,
        request_ids: List[int],
        fused_q: torch.Tensor,
        softmax_scale: Optional[float] = None,
        predicted_tokens_per_seq: int = 1,
        full_mask: bool = False,
    ) -> torch.Tensor:
        """fp32 latent MQA over each sequence's mirrored cache rows:
        K = rows [L, C+R], V = K[:, :C], scale 1/(q_scaling*sqrt(nope+rope)).

        At P = predicted_tokens_per_seq > 1 a sequence contributes P query
        rows, token-major, and row t is the draft token at absolute position
        L - P + t, so it attends to keys [0, L - P + t] — the bottom-right
        aligned causal mask measured in the MTP section below. full_mask=True
        builds the rival model instead (every row sees all L cached rows,
        including the sibling draft tokens that do not exist yet at row t).

        Over an fp8 pool both operands go through the e4m3 round trip the op
        sees: K rows because that is how they were written, and q because the
        caller quantizes it into quant_q_buffer by the same scale (the decode
        kernel reads no bf16 query at all). The two kv-scale factors the caller
        folds into mla_bmm1_scale / mla_bmm2_scale cancel the 1/s exactly, so
        the scale here stays the plain softmax scale.
        """
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        p = predicted_tokens_per_seq
        outs = []
        for g, rid in enumerate(request_ids):
            k_all = torch.cat(self.latent_rows[rid]).float()  # [L, C+R]
            total = k_all.shape[0]
            if self.fp8_pool:
                k_all = self.e4m3_roundtrip(k_all)
            for t in range(p):
                k_seq = k_all if full_mask else k_all[: total - p + t + 1]
                q_g = fused_q[g * p + t].view(self.num_heads, LATENT_DIM).float()
                if self.fp8_pool:
                    q_g = self.e4m3_roundtrip(q_g)
                probs = torch.softmax(q_g @ k_seq.T * scale, dim=-1)
                outs.append((probs @ k_seq[:, :KV_LORA_RANK]).to(torch.bfloat16))
        return torch.stack(outs).view(len(request_ids) * p, -1)

    def check_cache(self, request_id: int) -> None:
        """The pool must hold the mirrored latent rows, page by page.

        compressed_kv is a dtype-preserving copy and is gated bitwise per
        page; the roped k_pe half is gated by _assert_latent_rope over the
        request's whole row range. Over an fp8 pool the mirror is the same
        rows put through e4m3(row * kv_scale_orig_quant) and both halves are
        compared on their raw bytes.
        """
        expected = self.to_pool(torch.cat(self.latent_rows[request_id]))
        total = expected.shape[0]
        tpb = self.tokens_per_block
        pages_read = []
        for j, p in enumerate(self.pages[request_id]):
            n = min(tpb, total - j * tpb)
            if n <= 0:
                break
            page = self.pool[p, :n]
            ref = expected[j * tpb : j * tpb + n]
            if self.fp8_pool:
                same = torch.equal(
                    page[:, :KV_LORA_RANK].view(torch.uint8),
                    ref[:, :KV_LORA_RANK].view(torch.uint8),
                )
            else:
                same = torch.equal(page[:, :KV_LORA_RANK], ref[:, :KV_LORA_RANK])
            assert same, f"latent cache compressed_kv mismatch: request {request_id}, page {j}"
            pages_read.append(page)
        gathered = torch.cat(pages_read)
        assert gathered.shape[0] == total
        if self.fp8_pool:
            _assert_latent_rope_e4m3(gathered, expected, request_id)
        else:
            _assert_latent_rope(gathered, expected, request_id)

    def check_unwritten_pool_zero(self, request_ids: List[int]) -> None:
        """Every page outside the given requests' page sets is still all-zero.

        This is what pins the slab geometry: the op sizes a page from
        tokens_per_block, C+R and quant_mode alone, so an element-width or
        page-stride mistake writes into pages nobody reserved.
        """
        used = set()
        for rid in request_ids:
            used.update(self.pages[rid])
        rest = [p for p in range(self.pool.shape[0]) if p not in used]
        assert bool((self.pool[rest].float() == 0).all()), (
            "the append wrote outside the requests' pages"
        )


def _random_context_inputs(
    num_tokens: int,
    num_heads: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production-shaped MLA context inputs: contiguous q and k (the k rope
    slice deliberately garbage — the op fills it from latent_cache), v a
    strided split view of a packed kv_b_proj-style buffer, latent
    [ckv | k_pe]."""
    q = torch.randn(num_tokens, num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(
        num_tokens,
        num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_nope, v = kv.split([num_heads * QK_NOPE_HEAD_DIM, num_heads * V_HEAD_DIM], dim=-1)
    k = torch.empty_like(q).view(num_tokens, num_heads, QK_HEAD_DIM)
    k[..., :QK_NOPE_HEAD_DIM] = k_nope.view(num_tokens, num_heads, QK_NOPE_HEAD_DIM)
    k[..., QK_NOPE_HEAD_DIM:] = torch.randn(
        num_tokens, num_heads, QK_ROPE_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    k = k.view(num_tokens, num_heads * QK_HEAD_DIM)
    latent = torch.randn(num_tokens, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    return q, k, v, latent


def _mla_context_prefill_case(
    num_heads: int,
    seed: int,
    tokens_per_block: int,
    seq_lens: List[int],
    q_scaling: float = 1.0,
    rope: Optional[RopeParams] = None,
    rope_scalars: Optional[_RopeScalars] = None,
) -> None:
    """MLA context_only prefill: two fresh sequences, at least one crossing
    a page boundary. Verifies the FMHA output against a rope-aware fp32
    reference, the latent-cache append page by page (check_cache), that
    latent_cache is read-only, and that only the q/k rope slices are
    clobbered."""
    torch.manual_seed(seed)
    env = _MlaPagedEnv(
        num_heads=num_heads,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // tokens_per_block,
        tokens_per_block=tokens_per_block,
        q_scaling=q_scaling,
        rope=rope,
        rope_scalars=rope_scalars,
    )
    num_tokens = sum(seq_lens)
    for rid, ln in enumerate(seq_lens):
        env.add_request(rid, ln)
    q, k, v, latent = _random_context_inputs(num_tokens, num_heads)
    q_orig, k_orig, latent_orig = q.clone(), k.clone(), latent.clone()

    rids = list(range(len(seq_lens)))
    out = env.call_context(rids, seq_lens, q, k, v, latent)
    ref = env.context_reference(seq_lens, q_orig, k_orig, v, latent_orig)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    # The append this case checks must really walk more than one page.
    assert max(len(env.pages[rid]) for rid in rids) > 1
    for rid in rids:
        env.check_cache(rid)
    # latent_cache is an input only.
    assert torch.equal(latent, latent_orig)
    # The op ropes q_pe/k_pe in place: nope slices intact, rope slices clobbered.
    q3 = q.view(num_tokens, num_heads, QK_HEAD_DIM)
    k3 = k.view(num_tokens, num_heads, QK_HEAD_DIM)
    q3_orig = q_orig.view(num_tokens, num_heads, QK_HEAD_DIM)
    k3_orig = k_orig.view(num_tokens, num_heads, QK_HEAD_DIM)
    assert torch.equal(q3[..., :QK_NOPE_HEAD_DIM], q3_orig[..., :QK_NOPE_HEAD_DIM])
    assert torch.equal(k3[..., :QK_NOPE_HEAD_DIM], k3_orig[..., :QK_NOPE_HEAD_DIM])
    assert not torch.equal(q3[..., QK_NOPE_HEAD_DIM:], q3_orig[..., QK_NOPE_HEAD_DIM:])
    assert not torch.equal(k3[..., QK_NOPE_HEAD_DIM:], k3_orig[..., QK_NOPE_HEAD_DIM:])


def test_bf16_mla_context_prefill() -> None:
    """Fresh-prefill MLA context at 16 query heads, page size 64 (100 > 64
    crosses a boundary)."""
    _mla_context_prefill_case(MLA_NUM_HEADS, 5, MLA_TOKENS_PER_BLOCK, [100, 17])


def test_bf16_mla_context_prefill_h32() -> None:
    """The same fresh-prefill case at 32 query heads (32/32/192)."""
    _mla_context_prefill_case(MLA_NUM_HEADS_H32, 105, MLA_TOKENS_PER_BLOCK, [100, 17])


def test_bf16_mla_context_prefill_h8() -> None:
    """The same fresh-prefill case at 8 query heads (8/8/192) — the tep4
    slice of a 32-head checkpoint."""
    _mla_context_prefill_case(MLA_NUM_HEADS_H8, 805, MLA_TOKENS_PER_BLOCK, [100, 17])


def test_bf16_mla_page32_context_prefill_h32() -> None:
    """Fresh-prefill MLA context at page size 32 (the engine default), 32
    query heads. 96 fills three 32-token pages exactly — a 64-token page
    never ends there — and 33 crosses into a second page by one token, so
    the append's page/slot arithmetic is exercised at both a page-aligned
    end and a one-token spill."""
    _mla_context_prefill_case(MLA_NUM_HEADS_H32, 205, MLA_PAGE32, [96, 33])


def test_bf16_mla_page32_context_prefill_h8() -> None:
    """The same page-32 fresh-prefill case at 8 query heads."""
    _mla_context_prefill_case(MLA_NUM_HEADS_H8, 815, MLA_PAGE32, [96, 33])


def test_bf16_mla_page32_context_prefill_h128() -> None:
    """The same page-32 fresh-prefill case at 128 query heads (128/128/192) —
    the DeepSeek-R1-0528 layer shape, which attention DP replicates whole onto
    every rank rather than slicing. Baseline rope/scale so this case isolates
    the head count; the R1 cell adds the other two axes further down."""
    _mla_context_prefill_case(MLA_NUM_HEADS_H128, 905, MLA_PAGE32, [96, 33])


def _mla_generation_decode_case(
    num_heads: int,
    seed: int,
    tokens_per_block: int,
    prefill_lens: List[int],
    q_scaling: float = 1.0,
    rope: Optional[RopeParams] = None,
    rope_scalars: Optional[_RopeScalars] = None,
) -> None:
    """MLA generation_only decode over cache written by the context call.

    The first prefill length is an exact multiple of the page size, so the
    first decode token lands in a fresh page. Two decode steps; each step
    the test appends the new latent row (the op does not append in
    generation) and checks the FMHA output against an fp32 latent-MQA
    reference. The garbage latent_cache/q_pe arguments plus cache/fused_q
    invariance pin the fusion boundary: the generation call only reads the
    paged pool.
    """
    torch.manual_seed(seed)
    env = _MlaPagedEnv(
        num_heads=num_heads,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // tokens_per_block,
        tokens_per_block=tokens_per_block,
        q_scaling=q_scaling,
        rope=rope,
        rope_scalars=rope_scalars,
    )
    assert prefill_lens[0] % tokens_per_block == 0
    rids = list(range(len(prefill_lens)))
    for rid, ln in zip(rids, prefill_lens):
        env.add_request(rid, ln)
    q, k, v, latent = _random_context_inputs(sum(prefill_lens), num_heads)
    env.call_context(rids, prefill_lens, q, k, v, latent)
    for rid in rids:
        env.check_cache(rid)
    pages_after_prefill = {rid: len(env.pages[rid]) for rid in rids}

    for _ in range(2):
        for rid in rids:
            env.append_decode_latent(
                rid, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda")
            )
        fused_q = torch.randn(
            len(rids), num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda"
        )
        fused_q_orig = fused_q.clone()
        pool_before = env.pool.clone()
        out = env.call_generation(rids, fused_q)
        ref = env.generation_reference(rids, fused_q_orig)
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
        assert torch.equal(fused_q, fused_q_orig)
        assert torch.equal(env.pool, pool_before)  # generation never writes
    # The decode steps must really have opened a new page on some sequence,
    # i.e. the read range crossed a page boundary the prefill had not.
    assert any(len(env.pages[rid]) > pages_after_prefill[rid] for rid in rids)


def test_bf16_mla_generation_decode() -> None:
    """Latent-MQA MLA decode at 16 query heads, page size 64."""
    _mla_generation_decode_case(MLA_NUM_HEADS, 6, MLA_TOKENS_PER_BLOCK, [64, 30])


def test_bf16_mla_generation_decode_h32() -> None:
    """The same decode case at 32 query heads (32/1/576)."""
    _mla_generation_decode_case(MLA_NUM_HEADS_H32, 106, MLA_TOKENS_PER_BLOCK, [64, 30])


def test_bf16_mla_generation_decode_h8() -> None:
    """The same decode case at 8 query heads (8/1/576). Unlike 16 and 32,
    which share the ...VarSeqQ16... decode kernel, 8 heads map to a
    ...VarSeqQ8... one — a differently named compiled variant, so this is the
    flavor where the head count actually changes the kernel."""
    _mla_generation_decode_case(MLA_NUM_HEADS_H8, 806, MLA_TOKENS_PER_BLOCK, [64, 30])


def test_bf16_mla_page32_generation_decode_h32() -> None:
    """Latent-MQA MLA decode at page size 32 (the engine default), 32 query
    heads — the decode path is where the page size selects a different
    compiled trtllm-gen kernel (...PagedKvDenseP32... rather than ...P64...).
    Sequence 0's 64-token prefill fills two pages exactly, so its first
    decode token opens page 2; sequence 1's 31-token prefill leaves its
    first decode token on page 0's last slot and its second one opens page
    1, so a decode read range crosses a boundary mid-case."""
    _mla_generation_decode_case(MLA_NUM_HEADS_H32, 206, MLA_PAGE32, [64, 31])


def test_bf16_mla_page32_generation_decode_h16() -> None:
    """The same page-32 decode case at 16 query heads. The decode kernel is
    JIT-compiled once per head count even though its name does not carry the
    count, so the page-32 variant is exercised at both certified counts."""
    _mla_generation_decode_case(MLA_NUM_HEADS, 216, MLA_PAGE32, [64, 31])


def test_bf16_mla_page32_generation_decode_h8() -> None:
    """The same page-32 decode case at 8 query heads. Both axes that reach
    the compiled decode kernel move here at once: the page size is in the
    kernel name (...P32...) and 8 heads take the ...VarSeqQ8... q-tile."""
    _mla_generation_decode_case(MLA_NUM_HEADS_H8, 816, MLA_PAGE32, [64, 31])


def test_bf16_mla_page32_generation_decode_h128() -> None:
    """The same page-32 decode case at 128 query heads (128/1/576). Decode is
    the phase where the head count reaches kernel selection: 128 reports the
    same ...P32VarSeqQ16Kv128... name 16 and 32 do, yet pays its own compile
    (the cache is keyed more finely than the name)."""
    _mla_generation_decode_case(MLA_NUM_HEADS_H128, 906, MLA_PAGE32, [64, 31])


def _mla_mixed_batch_case(
    num_heads: int,
    seed: int,
    tokens_per_block: int,
    first_len: int,
    second_len: int,
    q_scaling: float = 1.0,
    rope: Optional[RopeParams] = None,
    rope_scalars: Optional[_RopeScalars] = None,
) -> None:
    """A mixed batch is two calls sharing full-batch metadata: context_only
    over the leading context rows, generation_only over the trailing
    generation rows (indexed from num_contexts)."""
    torch.manual_seed(seed)
    env = _MlaPagedEnv(
        num_heads=num_heads,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // tokens_per_block,
        tokens_per_block=tokens_per_block,
        q_scaling=q_scaling,
        rope=rope,
        rope_scalars=rope_scalars,
    )
    # Prefill request 0 alone, then decode it alongside a new context request.
    env.add_request(0, first_len)
    q, k, v, latent = _random_context_inputs(first_len, num_heads)
    env.call_context([0], [first_len], q, k, v, latent)

    env.add_request(1, second_len)
    env.append_decode_latent(0, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda"))
    q, k, v, latent = _random_context_inputs(second_len, num_heads)
    q_orig, k_orig, latent_orig = q.clone(), k.clone(), latent.clone()
    out_ctx = env.call_context([1], [second_len], q, k, v, latent, gen_request_ids=[0])
    ref_ctx = env.context_reference([second_len], q_orig, k_orig, v, latent_orig)
    torch.testing.assert_close(out_ctx, ref_ctx, rtol=RTOL, atol=ATOL)

    fused_q = torch.randn(1, num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    out_gen = env.call_generation([0], fused_q, ctx_request_ids=[1], ctx_seq_lens=[second_len])
    ref_gen = env.generation_reference([0], fused_q)
    torch.testing.assert_close(out_gen, ref_gen, rtol=RTOL, atol=ATOL)
    env.check_cache(0)
    env.check_cache(1)


def test_bf16_mla_mixed_batch_split_calls() -> None:
    """Mixed-batch MLA (two phase calls) at 16 query heads, page size 64."""
    _mla_mixed_batch_case(MLA_NUM_HEADS, 7, MLA_TOKENS_PER_BLOCK, 40, 23)


def test_bf16_mla_mixed_batch_split_calls_h32() -> None:
    """The same mixed-batch case at 32 query heads."""
    _mla_mixed_batch_case(MLA_NUM_HEADS_H32, 107, MLA_TOKENS_PER_BLOCK, 40, 23)


def test_bf16_mla_mixed_batch_split_calls_h8() -> None:
    """The same mixed-batch case at 8 query heads: the two phase calls of one
    batch run at 8/8/192 and 8/1/576 off the same full-batch state tensors."""
    _mla_mixed_batch_case(MLA_NUM_HEADS_H8, 807, MLA_TOKENS_PER_BLOCK, 40, 23)


def test_bf16_mla_page32_mixed_batch_split_calls_h32() -> None:
    """Mixed-batch MLA at page size 32, 32 query heads: the decoding
    sequence's 64-token history fills two pages exactly, so its appended
    token opens page 2 and the decode call reads across three pages, while
    the context sequence sharing the batch spans two."""
    _mla_mixed_batch_case(MLA_NUM_HEADS_H32, 207, MLA_PAGE32, 64, 33)


def test_bf16_mla_page32_mixed_batch_split_calls_h8() -> None:
    """The same page-32 mixed-batch case at 8 query heads."""
    _mla_mixed_batch_case(MLA_NUM_HEADS_H8, 817, MLA_PAGE32, 64, 33)


def test_bf16_mla_page32_mixed_batch_split_calls_h128() -> None:
    """The same page-32 mixed-batch case at 128 query heads: the two phase
    calls run at 128/128/192 and 128/1/576 off one set of batch tensors."""
    _mla_mixed_batch_case(MLA_NUM_HEADS_H128, 907, MLA_PAGE32, 64, 33)


# ─── MLA context with latent_cache=None (cached KV / chunked prefill) ───


def _random_explicit_kv(num_tokens: int, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Production-shaped explicit K/V sources for the latent_cache=None
    context calls: a contiguous K and a packed [T, H*(nope+v)]
    kv_b_proj-style buffer whose _v_split_view is the V argument."""
    k = torch.randn(num_tokens, num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    packed_kv = torch.randn(
        num_tokens,
        num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return k, packed_kv


def _v_split_view(packed_kv: torch.Tensor, num_heads: int) -> torch.Tensor:
    """The [.., H*nope:] split view of a packed [T, H*(nope+v)] buffer. The
    context FMHA hard-codes V's row stride to H*(nope+v_head_dim) elements,
    so V must keep this stride — a contiguous [T, H*v] V is misread."""
    return packed_kv.split([num_heads * QK_NOPE_HEAD_DIM, num_heads * V_HEAD_DIM], dim=-1)[1]


def _explicit_kv_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: List[int],
    kv_lens: List[int],
    mask_type: int,
    num_heads: int,
    softmax_scale: float = MLA_SOFTMAX_SCALE,
    e4m3_inputs: bool = False,
    out_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 attention over explicit per-sequence K/V ranges plus softmax
    stats: per (token, head), m = row max of the scaled logits and
    sigma = sum(exp(logit - m)) over this pass's KV range. Causal is
    bottom-right aligned (query i of a sequence sits at absolute position
    kv_s - q_s + i). Rows of zero-KV sequences stay NaN — the op leaves them
    undefined and downstream merge plans skip them. Returns
    (bf16 output [Tq, H*v], fp32 stats [Tq, H, 2]).

    e4m3_inputs rounds q, k and v through e4m3 at scale 1.0, which is what this
    flavor quantizes them to under quant_mode 128; out_scale multiplies the
    output, the second of the two dequantization factors that path applies (the
    first, s**2, rides in softmax_scale). Both cancel only at s = 1.0. The
    stats stay in the un-scaled domain the op writes them in and are not
    certified on the fp8 path."""
    total_q = sum(q_lens)
    out = torch.full((total_q, num_heads * V_HEAD_DIM), float("nan"), device="cuda")
    stats = torch.full((total_q, num_heads, 2), float("nan"), device="cuda")
    q_start = kv_start = 0
    for q_len, kv_len in zip(q_lens, kv_lens):
        if kv_len == 0:
            q_start += q_len
            continue
        q_seq = q[q_start : q_start + q_len].view(q_len, num_heads, QK_HEAD_DIM).float()
        k_seq = k[kv_start : kv_start + kv_len].view(kv_len, num_heads, QK_HEAD_DIM).float()
        v_seq = v[kv_start : kv_start + kv_len].view(kv_len, num_heads, V_HEAD_DIM).float()
        if e4m3_inputs:
            q_seq = q_seq.to(torch.float8_e4m3fn).float()
            k_seq = k_seq.to(torch.float8_e4m3fn).float()
            v_seq = v_seq.to(torch.float8_e4m3fn).float()
        scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq) * softmax_scale
        if mask_type == MASK_CAUSAL:
            offset = kv_len - q_len
            mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device="cuda")
            for i in range(q_len):
                mask[i, : offset + i + 1] = True
            scores = scores.masked_fill(~mask, float("-inf"))
        row_max = scores.max(dim=-1).values  # [H, q_len]
        row_sum = torch.exp(scores - row_max.unsqueeze(-1)).sum(dim=-1)
        probs = torch.softmax(scores, dim=-1)
        out[q_start : q_start + q_len] = (
            torch.einsum("hij,jhd->ihd", probs, v_seq).reshape(q_len, -1) * out_scale
        )
        stats[q_start : q_start + q_len, :, 0] = row_max.transpose(0, 1)
        stats[q_start : q_start + q_len, :, 1] = row_sum.transpose(0, 1)
        q_start += q_len
        kv_start += kv_len
    return out.to(torch.bfloat16), stats


def _assert_valid_rows_close(
    actual: torch.Tensor, ref: torch.Tensor, ref_stats: torch.Tensor
) -> None:
    """Compare only rows of sequences that had KV in this pass (non-NaN in
    the reference); zero-KV rows are undefined op output."""
    valid = ~torch.isnan(ref_stats[:, 0, 0])
    torch.testing.assert_close(actual[valid], ref[valid], rtol=RTOL, atol=ATOL)


def _assert_stats_close(actual: torch.Tensor, ref_stats: torch.Tensor) -> None:
    valid = ~torch.isnan(ref_stats[:, 0, 0])
    # Both sides are fp32 reductions over the same bf16-rounded q/k, so they
    # differ only by accumulation order: observed max abs err 2e-6 (max stat)
    # and rel err 1.2e-6 (sum stat) on sm_100, on stat magnitudes ~1-40.
    # 1e-4 gives ~50x margin while still catching wrong-domain (log2) or
    # unscaled-logit stats outright.
    torch.testing.assert_close(actual[valid], ref_stats[valid], rtol=1e-4, atol=1e-4)


def _mla_context_cached_kv_case(
    num_heads: int,
    seed: int,
    tokens_per_block: int,
    cached_lens: List[int],
    new_lens: List[int],
    q_scaling: float = 1.0,
    rope: Optional[RopeParams] = None,
    rope_scalars: Optional[_RopeScalars] = None,
) -> None:
    """MLA context over a cached KV prefix: latent_cache=None, q pre-rotated
    upstream, K/V supplied for the full [cached + new] range so KV length
    exceeds q length. Causal masking is bottom-right aligned. Verifies the
    output against an fp32 reference and that the call mutates nothing but
    output: q, k, v, and the paged pool stay bitwise intact (no in-kernel
    RoPE, no append)."""
    torch.manual_seed(seed)
    env = _MlaPagedEnv(
        num_heads=num_heads,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // tokens_per_block,
        tokens_per_block=tokens_per_block,
        q_scaling=q_scaling,
        rope=rope,
        rope_scalars=rope_scalars,
    )
    kv_lens = [c + n for c, n in zip(cached_lens, new_lens)]
    for rid, total in enumerate(kv_lens):
        env.add_request(rid, new_lens[rid])
        env.reserve_cache_pages(rid, total)
    env.pool.normal_()  # op must not read or write the pool in this mode
    pool_before = env.pool.clone()

    q = torch.randn(sum(new_lens), num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k, packed_kv = _random_explicit_kv(sum(kv_lens), num_heads)
    v = _v_split_view(packed_kv, num_heads)
    q_orig, k_orig, v_orig = q.clone(), k.clone(), v.clone()

    out = env.call_context_no_append([0, 1, 2], new_lens, kv_lens, q, k, v)
    ref, _ = _explicit_kv_reference(
        q, k, v, new_lens, kv_lens, MASK_CAUSAL, num_heads, env.softmax_scale
    )
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    # Fusion boundary: everything is an input; only output rows are written.
    assert torch.equal(q, q_orig)
    assert torch.equal(k, k_orig)
    assert torch.equal(v, v_orig)
    assert torch.equal(env.pool, pool_before)


def test_bf16_mla_context_cached_kv_no_append() -> None:
    """Cached-KV (no-append) MLA context at 16 query heads, page size 64.
    Prefixes: page-crossing, mid-page, and empty."""
    _mla_context_cached_kv_case(MLA_NUM_HEADS, 8, MLA_TOKENS_PER_BLOCK, [80, 33, 0], [40, 7, 25])


def test_bf16_mla_context_cached_kv_no_append_h32() -> None:
    """The same cached-KV context case at 32 query heads."""
    _mla_context_cached_kv_case(
        MLA_NUM_HEADS_H32, 108, MLA_TOKENS_PER_BLOCK, [80, 33, 0], [40, 7, 25]
    )


def test_bf16_mla_context_cached_kv_no_append_h8() -> None:
    """The same cached-KV context case at 8 query heads."""
    _mla_context_cached_kv_case(
        MLA_NUM_HEADS_H8, 808, MLA_TOKENS_PER_BLOCK, [80, 33, 0], [40, 7, 25]
    )


def test_bf16_mla_page32_context_cached_kv_no_append_h32() -> None:
    """Cached-KV (no-append) MLA context at page size 32, 32 query heads.
    The pages are reserved as production does even though this flavor never
    touches the pool, so the batch carries a page-32 offsets table: prefixes
    of 96 (three exact pages), 31 (one short of a page) and 0, reaching KV
    lengths of 128 (four exact pages), 40 and 25."""
    _mla_context_cached_kv_case(MLA_NUM_HEADS_H32, 208, MLA_PAGE32, [96, 31, 0], [32, 9, 25])


def test_bf16_mla_page32_context_cached_kv_no_append_h8() -> None:
    """The same page-32 cached-KV context case at 8 query heads."""
    _mla_context_cached_kv_case(MLA_NUM_HEADS_H8, 818, MLA_PAGE32, [96, 31, 0], [32, 9, 25])


def test_bf16_mla_page32_context_cached_kv_no_append_h128() -> None:
    """The same page-32 cached-KV context case at 128 query heads. This is the
    flavor an engine with block reuse on runs for every context request that
    hits a cached prefix, so it carries the head count as much as the fresh
    one does."""
    _mla_context_cached_kv_case(MLA_NUM_HEADS_H128, 908, MLA_PAGE32, [96, 31, 0], [32, 9, 25])


def test_bf16_mla_context_chunked_prefill_with_merge() -> None:
    """MLA chunked context: one padding-masked partial pass per cached-KV
    chunk with softmax_stats_tensor emitted, a final causal pass over the
    new tokens, each folded into the running output by the downstream
    trtllm merge op (the production consumer of the emitted stats — the
    reference is still built from torch alone). Every pass's output and
    stats are checked against fp32 partial-attention references, and the
    fully merged output/stats against a single-pass full-range reference."""
    torch.manual_seed(9)
    env = _MlaPagedEnv()
    cached_lens = [96, 48, 0]
    new_lens = [32, 17, 23]
    kv_lens = [c + n for c, n in zip(cached_lens, new_lens)]
    num_ctx_tokens = sum(new_lens)
    for rid, total in enumerate(kv_lens):
        env.add_request(rid, new_lens[rid])
        env.reserve_cache_pages(rid, total)

    # Per-sequence full KV timeline; chunk passes slice the cached region.
    # V buffers are sliced and concatenated as packed rows so every pass's V
    # keeps the required H*(nope+v) row stride.
    k_full = []
    packed_full = []
    for total in kv_lens:
        k_seq, packed_seq = _random_explicit_kv(total, MLA_NUM_HEADS)
        k_full.append(k_seq)
        packed_full.append(packed_seq)
    q = torch.randn(
        num_ctx_tokens, MLA_NUM_HEADS * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )

    # Production chunk plan (greedy, 64-token buffer over the cached
    # regions 96/48/0): loop 0 takes 64 of seq 0; loop 1 the remaining 32 of
    # seq 0 plus 32 of seq 1; loop 2 the remaining 16 of seq 1. Merge ops:
    # 2 = copy on a sequence's first pass, 1 = merge, 0 = skip; the final
    # new-token pass merges (or copies for the cache-less seq 2).
    chunk_lens = [[64, 0, 0], [32, 32, 0], [0, 16, 0]]
    chunk_offsets = [[0, 0, 0], [64, 0, 0], [96, 32, 0]]
    merge_ops = [[2, 0, 0], [1, 2, 0], [0, 1, 0], [1, 1, 2]]

    merged = torch.empty(
        num_ctx_tokens, MLA_NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    merged_stats = torch.empty(num_ctx_tokens, MLA_NUM_HEADS, 2, dtype=torch.float32, device="cuda")
    temp_stats = torch.empty_like(merged_stats)
    cu_q = torch.tensor(
        [0, *torch.tensor(new_lens).cumsum(0).tolist()],
        dtype=torch.int64,
        device="cuda",
    )

    def run_pass(
        pass_kv_lens: List[int],
        offsets: List[int],
        mask_type: int,
        ops: List[int],
    ) -> None:
        slices = list(enumerate(zip(offsets, pass_kv_lens)))
        k_buf = torch.cat([k_full[s][o : o + n] for s, (o, n) in slices])
        v_buf = _v_split_view(
            torch.cat([packed_full[s][o : o + n] for s, (o, n) in slices]),
            MLA_NUM_HEADS,
        )
        temp_out = env.call_context_no_append(
            [0, 1, 2],
            new_lens,
            pass_kv_lens,
            q,
            k_buf,
            v_buf,
            mask_type=mask_type,
            softmax_stats_tensor=temp_stats,
        )
        ref_out, ref_stats = _explicit_kv_reference(
            q, k_buf, v_buf, new_lens, pass_kv_lens, mask_type, MLA_NUM_HEADS
        )
        _assert_valid_rows_close(temp_out, ref_out, ref_stats)
        _assert_stats_close(temp_stats, ref_stats)
        # The downstream merge op consumes the emitted stats directly.
        torch.ops.trtllm.merge_chunked_attention_for_mla(
            merged,
            temp_out,
            merged_stats,
            temp_stats,
            len(new_lens),
            cu_q,
            max(new_lens),
            torch.tensor(ops, dtype=torch.int64, device="cuda"),
            MLA_NUM_HEADS,
            V_HEAD_DIM,
        )
        torch.cuda.synchronize()

    for loop_idx, (lens, offs) in enumerate(zip(chunk_lens, chunk_offsets)):
        run_pass(lens, offs, MASK_PADDING, merge_ops[loop_idx])
    # Final pass: causal attention of the new tokens over themselves only.
    run_pass(new_lens, cached_lens, MASK_CAUSAL, merge_ops[-1])

    # The merged result must equal single-pass attention over the full
    # [cached + new] range (bottom-right-aligned causal).
    k_all = torch.cat(k_full)
    v_all = _v_split_view(torch.cat(packed_full), MLA_NUM_HEADS)
    ref_full, ref_full_stats = _explicit_kv_reference(
        q, k_all, v_all, new_lens, kv_lens, MASK_CAUSAL, MLA_NUM_HEADS
    )
    torch.testing.assert_close(merged, ref_full, rtol=RTOL, atol=ATOL)
    _assert_stats_close(merged_stats, ref_full_stats)


# ─── MLA at q_lora_rank = 0 (checkpoints with no q-LoRA) ───

# The values swept per MLA call flavor. Index 0 is the reference run and
# index 2 repeats it — the run-to-run determinism control that makes the
# bitwise comparisons meaningful. 4096 is larger than any real q_a_proj rank
# and larger than C, so any arithmetic actually reading the argument would
# move something.
Q_LORA_RANK_SWEEP = [Q_LORA_RANK_DSV3, Q_LORA_RANK_ZERO, Q_LORA_RANK_DSV3, 4096]

# KV-cache scaling factors swept over the fp8 latent pool: 1.0 is the
# production value (DeepSeek-R1-0528-FP4 declares kv_cache_quant_algo FP8
# with per-layer k_scale/v_scale both 1.0), 1.5 is not a power of two, so the
# e4m3 grid genuinely depends on it, and 2.0 is.
KV_SCALE_SWEEP = [1.0, 1.5, 2.0]


def _mla_page32_env(num_heads: int, q_lora_rank: int) -> _MlaPagedEnv:
    """A page-32 MLA env at one head count and one q_lora_rank. Head count and
    page size are certified axes of their own; fixing both within a sweep
    leaves q_lora_rank the only variable. The two counts swept are the shipped
    ones: 32 (deepseek-v3-lite tp1) and 8 (its tep4 slice), which is the count
    that takes its own compiled decode kernel (...VarSeqQ8...)."""
    return _MlaPagedEnv(
        num_heads=num_heads,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
        tokens_per_block=MLA_PAGE32,
        q_lora_rank=q_lora_rank,
    )


def _assert_q_lora_rank_inert(
    flavor: str,
    runs: List[Tuple[int, Dict[str, torch.Tensor], int]],
) -> None:
    """Every observable of the reference run — outputs, the whole paged pool,
    the in-place-written inputs, and the size the op grew the workspace to —
    must come back bitwise identical at every other q_lora_rank."""
    base_rank, base, base_ws = runs[0]
    for rank, cur, ws in runs[1:]:
        for key, expected in base.items():
            got = cur[key]
            assert torch.equal(expected, got), (
                f"{flavor}: {key} differs at q_lora_rank={rank} vs "
                f"{base_rank}: {int((expected != got).sum())}/{expected.numel()} "
                f"elements, max abs "
                f"{(expected.float() - got.float()).abs().max().item():.3e}"
            )
        assert ws == base_ws, (
            f"{flavor}: workspace grew to {ws} bytes at q_lora_rank={rank}, "
            f"{base_ws} at {base_rank}"
        )


def _mla_qlora_context_prefill_case(num_heads: int, seed: int) -> None:
    """Fresh-prefill MLA context swept over q_lora_rank (page 32).

    This is the flavor with the most machinery behind the MLA meta params —
    in-kernel GPT-J RoPE of q_pe/k_pe plus the paged latent append — so it is
    where a q_lora_rank-dependent layout would surface. The zero run is
    checked against the fp32 reference and its append against the mirrored
    latent rows; the whole sweep is then checked bitwise against the
    q_lora_rank=1536 run on identical inputs.
    """
    seq_lens = [96, 33]
    num_tokens = sum(seq_lens)
    h = num_heads
    torch.manual_seed(seed)
    q_src, k_src, v, latent_src = _random_context_inputs(num_tokens, h)
    v_src = v.clone()  # v is shared across the sweep: it must stay an input

    runs: List[Tuple[int, Dict[str, torch.Tensor], int]] = []
    for rank in Q_LORA_RANK_SWEEP:
        env = _mla_page32_env(h, rank)
        for rid, ln in enumerate(seq_lens):
            env.add_request(rid, ln)
        q, k, latent = q_src.clone(), k_src.clone(), latent_src.clone()
        out = env.call_context([0, 1], seq_lens, q, k, v, latent)

        if rank == Q_LORA_RANK_ZERO:
            # The surface under test has to be right, not merely reproducible.
            ref = env.context_reference(seq_lens, q_src, k_src, v, latent_src)
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
            assert max(len(env.pages[rid]) for rid in (0, 1)) > 1
            for rid in (0, 1):
                env.check_cache(rid)
            assert torch.equal(latent, latent_src)  # latent_cache is read-only
            q3 = q.view(num_tokens, h, QK_HEAD_DIM)
            k3 = k.view(num_tokens, h, QK_HEAD_DIM)
            q3_src = q_src.view(num_tokens, h, QK_HEAD_DIM)
            k3_src = k_src.view(num_tokens, h, QK_HEAD_DIM)
            nope = QK_NOPE_HEAD_DIM
            assert torch.equal(q3[..., :nope], q3_src[..., :nope])
            assert torch.equal(k3[..., :nope], k3_src[..., :nope])
            assert not torch.equal(q3[..., nope:], q3_src[..., nope:])
            assert not torch.equal(k3[..., nope:], k3_src[..., nope:])

        runs.append(
            (
                rank,
                {
                    "output": out,
                    "pool": env.pool.clone(),
                    "q_after_call": q,
                    "k_after_call": k,
                },
                env.workspace.numel(),
            )
        )
    assert torch.equal(v, v_src), "the sweep must have run on identical inputs"
    _assert_q_lora_rank_inert(f"fresh-prefill context H={h}", runs)


def test_bf16_mla_qlora0_context_prefill_h32() -> None:
    _mla_qlora_context_prefill_case(MLA_NUM_HEADS_H32, 305)


def test_bf16_mla_qlora0_context_prefill_h8() -> None:
    _mla_qlora_context_prefill_case(MLA_NUM_HEADS_H8, 315)


def _mla_qlora_generation_decode_case(num_heads: int, seed: int) -> None:
    """Latent-MQA MLA decode swept over q_lora_rank (page 32).

    The generation phase is the one that JIT-compiles its FMHA kernel, so
    this is where q_lora_rank would show up as a kernel-selection axis: the
    whole sweep runs in one process against a single compiled decode kernel
    (the compile cache is keyed by head count and page size, both fixed
    here). Each variant builds its own history through its own context call,
    so a rank that changed the append would already separate the pools.
    """
    prefill_lens = [64, 31]
    h = num_heads
    torch.manual_seed(seed)
    q_src, k_src, v, latent_src = _random_context_inputs(sum(prefill_lens), h)
    v_src = v.clone()
    # Decode rows and fused q are drawn once and replayed per variant.
    decode_rows = [
        [torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") for _ in prefill_lens]
        for _ in range(2)
    ]
    fused_qs = [
        torch.randn(len(prefill_lens), h * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    ]

    runs: List[Tuple[int, Dict[str, torch.Tensor], int]] = []
    for rank in Q_LORA_RANK_SWEEP:
        env = _mla_page32_env(h, rank)
        for rid, ln in enumerate(prefill_lens):
            env.add_request(rid, ln)
        env.call_context([0, 1], prefill_lens, q_src.clone(), k_src.clone(), v, latent_src.clone())
        observed = {"pool_after_prefill": env.pool.clone()}
        for step in range(2):
            for rid in range(len(prefill_lens)):
                env.append_decode_latent(rid, decode_rows[step][rid])
            # call_generation draws the (unconsumed) latent_cache/q_pe
            # garbage from the global RNG: reseed so every variant is fed the
            # same garbage and q_lora_rank stays the only difference.
            torch.manual_seed(seed * 10 + step)
            fused_q = fused_qs[step].clone()
            out = env.call_generation([0, 1], fused_q)
            if rank == Q_LORA_RANK_ZERO:
                ref = env.generation_reference([0, 1], fused_qs[step])
                torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
                assert torch.equal(fused_q, fused_qs[step])
            observed[f"decode_out_{step}"] = out
            observed[f"pool_after_decode_{step}"] = env.pool.clone()
        runs.append((rank, observed, env.workspace.numel()))
    assert torch.equal(v, v_src), "the sweep must have run on identical inputs"
    _assert_q_lora_rank_inert(f"generation decode H={h}", runs)


def test_bf16_mla_qlora0_generation_decode_h32() -> None:
    _mla_qlora_generation_decode_case(MLA_NUM_HEADS_H32, 306)


def test_bf16_mla_qlora0_generation_decode_h8() -> None:
    """H=8 decode compiles ...VarSeqQ8Kv128... where 16 and 32 both take
    ...VarSeqQ16...: a genuinely different kernel, so the H=32 sweep says
    nothing about it. This is the tep4 deepseek-v3-lite cell, whose config
    carries "q_lora_rank": null."""
    _mla_qlora_generation_decode_case(MLA_NUM_HEADS_H8, 316)


def _mla_qlora_context_cached_kv_no_append_case(num_heads: int, seed: int) -> None:
    """Cached-KV (no-append) MLA context swept over q_lora_rank (page 32).

    latent_cache=None skips the in-kernel RoPE and the append, so this flavor
    reaches a different context kernel from the fresh-prefill one; the paged
    pool is pre-filled with noise and must come back bitwise untouched at
    every rank.
    """
    cached_lens, new_lens = [96, 31, 0], [32, 9, 25]
    kv_lens = [c + n for c, n in zip(cached_lens, new_lens)]
    h = num_heads
    torch.manual_seed(seed)
    q_src = torch.randn(sum(new_lens), h * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k_src, packed_kv = _random_explicit_kv(sum(kv_lens), h)
    v = _v_split_view(packed_kv, h)
    v_src = v.clone()
    pool_fill: Optional[torch.Tensor] = None

    runs: List[Tuple[int, Dict[str, torch.Tensor], int]] = []
    for rank in Q_LORA_RANK_SWEEP:
        env = _mla_page32_env(h, rank)
        for rid, total in enumerate(kv_lens):
            env.add_request(rid, new_lens[rid])
            env.reserve_cache_pages(rid, total)
        if pool_fill is None:
            pool_fill = torch.randn_like(env.pool)
        env.pool.copy_(pool_fill)  # the op must neither read nor write it
        q, k = q_src.clone(), k_src.clone()
        out = env.call_context_no_append([0, 1, 2], new_lens, kv_lens, q, k, v)

        if rank == Q_LORA_RANK_ZERO:
            ref, _ = _explicit_kv_reference(q, k, v, new_lens, kv_lens, MASK_CAUSAL, h)
            torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
            assert torch.equal(q, q_src)
            assert torch.equal(k, k_src)
            assert torch.equal(env.pool, pool_fill)

        runs.append(
            (
                rank,
                {"output": out, "pool": env.pool.clone()},
                env.workspace.numel(),
            )
        )
    assert torch.equal(v, v_src), "the sweep must have run on identical inputs"
    _assert_q_lora_rank_inert(f"no-append context H={h}", runs)


def test_bf16_mla_qlora0_context_cached_kv_no_append_h32() -> None:
    _mla_qlora_context_cached_kv_no_append_case(MLA_NUM_HEADS_H32, 307)


def test_bf16_mla_qlora0_context_cached_kv_no_append_h8() -> None:
    _mla_qlora_context_cached_kv_no_append_case(MLA_NUM_HEADS_H8, 317)


# ─── The DeepSeek-R1-0528 cell: H=128 + YaRN rope table + q_scaling != 1 ───
#
# Three axes move at once relative to the deepseek-v3-lite cells above: 128
# query heads (attention DP replicates them whole onto every rank), a
# YaRN-scaled rotary_cos_sin table, and a softmax scale that folds YaRN's
# mscale^2 in through q_scaling. They are certified together — the four call
# flavors below at exactly the values a DeepSeek-R1-0528 layer passes — and
# separately, by the two axis sweeps after them.


def _assert_outside_band(
    out: torch.Tensor, rival: torch.Tensor, threshold: float, label: str
) -> float:
    """A rival hypothesis must land this far outside the tolerance band around
    the observed output. Without it, matching the positive reference proves
    little: an op that silently ignored the argument under test would pass the
    positive comparison too wherever the argument's effect happens to be
    small. Returns the separation in units of the allowance."""
    diff = (out.float() - rival.float()).abs().max().item()
    allowance = ATOL + RTOL * rival.float().abs().max().item()
    ratio = diff / allowance
    assert ratio > threshold, f"{label}: only {ratio:.4g}x outside the tolerance band"
    return ratio


def test_bf16_mla_r1_cell_context_prefill_h128() -> None:
    """Fresh-prefill MLA context at the full R1 cell: 128/128/192, page 32,
    the checkpoint's YaRN table, q_scaling = 1/mscale^2."""
    _mla_context_prefill_case(
        MLA_NUM_HEADS_H128,
        925,
        MLA_PAGE32,
        [96, 33],
        q_scaling=R1_Q_SCALING,
        rope=_r1_rope_params(MLA_MAX_SEQ_LEN),
        rope_scalars=R1_ROPE_SCALARS,
    )


def test_bf16_mla_r1_cell_generation_decode_h128() -> None:
    """Latent-MQA decode at the full R1 cell: 128/1/576, page 32. The rope
    table is passed but nothing is rotated here — the scale is what carries."""
    _mla_generation_decode_case(
        MLA_NUM_HEADS_H128,
        926,
        MLA_PAGE32,
        [64, 31],
        q_scaling=R1_Q_SCALING,
        rope=_r1_rope_params(MLA_MAX_SEQ_LEN),
        rope_scalars=R1_ROPE_SCALARS,
    )


def test_bf16_mla_r1_cell_mixed_batch_h128() -> None:
    """Mixed batch at the full R1 cell: the two phase calls share one set of
    batch tensors, one YaRN table and one q_scaling."""
    _mla_mixed_batch_case(
        MLA_NUM_HEADS_H128,
        927,
        MLA_PAGE32,
        64,
        33,
        q_scaling=R1_Q_SCALING,
        rope=_r1_rope_params(MLA_MAX_SEQ_LEN),
        rope_scalars=R1_ROPE_SCALARS,
    )


def test_bf16_mla_r1_cell_context_cached_kv_no_append_h128() -> None:
    """Cached-KV (no-append) context at the full R1 cell. This flavor applies
    no rope at all — q arrives pre-rotated — so it is where q_scaling is the
    only one of the three op-side axes with an effect."""
    _mla_context_cached_kv_case(
        MLA_NUM_HEADS_H128,
        928,
        MLA_PAGE32,
        [96, 31, 0],
        [32, 9, 25],
        q_scaling=R1_Q_SCALING,
        rope=_r1_rope_params(MLA_MAX_SEQ_LEN),
        rope_scalars=R1_ROPE_SCALARS,
    )


# q_scaling values swept in both phases. 1.0 is the value certified before
# this run, R1_Q_SCALING (~0.53366) is what the checkpoint's YaRN config
# wants, and 2.0 / 0.25 bracket it from both sides so the argument is
# exercised as an axis rather than pinned at one number.
Q_SCALING_SWEEP = [1.0, R1_Q_SCALING, 2.0, 0.25]

# A run at one q_scaling must land this far outside the tolerance band of a
# reference built at any other swept value. Measured over the full 4x4 matrix
# on sm_100 at H=128, page 32: every cross pair sits between 20.1x and 178x
# (the tightest is 1.0 against 2.0 in the context phase; the "q_scaling
# silently ignored" hypothesis — a reference at 1.0 for a run at any other
# value — spans 20.1x-70.5x), while a matching reference uses at most 0.23 of
# the same band. The 5x gate sits between those two populations by ~22x on
# one side and ~4x on the other. It is deliberately not tight enough to
# resolve two adjacent values: 0.5 against 0.53366, a 6.3% change of scale,
# separates by only 2.95x-3.61x, so the swept values are kept well apart.
Q_SCALING_MIN_SEPARATION = 5.0


def test_bf16_mla_q_scaling_axis_h128() -> None:
    """q_scaling is the softmax-scale axis of both MLA phases: QK^T is scaled
    by 1 / (q_scaling * sqrt(nope + R)) in the context call and in the
    generation call alike, the latter despite its head_size being C + R.

    Four values on identical inputs. Each run is checked against a reference
    built at its own value and gated outside the references of the other
    three, and the paged latent append is asserted bitwise identical across
    the sweep — q_scaling moves the softmax scale and nothing else.
    """
    h = MLA_NUM_HEADS_H128
    prefill_lens = [96, 33]
    torch.manual_seed(935)
    q_src, k_src, v, latent_src = _random_context_inputs(sum(prefill_lens), h)
    decode_rows = [
        torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") for _ in prefill_lens
    ]
    fused_q = torch.randn(len(prefill_lens), h * LATENT_DIM, dtype=torch.bfloat16, device="cuda")

    runs: List[Tuple[float, _MlaPagedEnv, torch.Tensor, torch.Tensor]] = []
    for q_scaling in Q_SCALING_SWEEP:
        env = _MlaPagedEnv(
            num_heads=h,
            max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
            tokens_per_block=MLA_PAGE32,
            q_scaling=q_scaling,
            rope=_r1_rope_params(MLA_MAX_SEQ_LEN),
            rope_scalars=R1_ROPE_SCALARS,
        )
        for rid, ln in enumerate(prefill_lens):
            env.add_request(rid, ln)
        out_ctx = env.call_context(
            [0, 1], prefill_lens, q_src.clone(), k_src.clone(), v, latent_src.clone()
        )
        for rid, row in enumerate(decode_rows):
            env.append_decode_latent(rid, row)
        out_gen = env.call_generation([0, 1], fused_q)
        runs.append((q_scaling, env, out_ctx, out_gen))

    for q_scaling, env, out_ctx, out_gen in runs:
        for rival in Q_SCALING_SWEEP:
            scale = 1.0 / (rival * math.sqrt(QK_HEAD_DIM))
            ref_ctx = env.context_reference(
                prefill_lens, q_src, k_src, v, latent_src, softmax_scale=scale
            )
            ref_gen = env.generation_reference([0, 1], fused_q, softmax_scale=scale)
            if rival == q_scaling:
                torch.testing.assert_close(out_ctx, ref_ctx, rtol=RTOL, atol=ATOL)
                torch.testing.assert_close(out_gen, ref_gen, rtol=RTOL, atol=ATOL)
            else:
                _assert_outside_band(
                    out_ctx,
                    ref_ctx,
                    Q_SCALING_MIN_SEPARATION,
                    f"context at q_scaling={q_scaling} vs a reference at {rival}",
                )
                _assert_outside_band(
                    out_gen,
                    ref_gen,
                    Q_SCALING_MIN_SEPARATION,
                    f"decode at q_scaling={q_scaling} vs a reference at {rival}",
                )

    base_pool = runs[0][1].pool
    for q_scaling, env, _, _ in runs[1:]:
        assert torch.equal(env.pool, base_pool), (
            f"the paged latent pool moved at q_scaling={q_scaling}"
        )


def _yarn_cos_sin_table(
    num_positions: int,
    dim: int,
    theta: float,
    factor: float,
    original_max_positions: int,
    beta_fast: int,
    beta_slow: int,
    mscale: float,
    mscale_all_dim: float,
) -> torch.Tensor:
    """The duplicated-layout fp32 (cos, sin) table a YaRN rope config
    produces, built from the published YaRN formula with plain torch.

    A caller that cannot reach TensorRT-LLM's own table builder has to
    reproduce this; asserting it against the builder's output is what makes
    the formula usable as a contract statement rather than a description.
    Layout: dim (cos, sin) pairs per position, the second dim/2 duplicating
    the first, flattened to [1, num_positions * dim * 2].
    """
    half = dim // 2

    def correction_dim(rotations: float) -> float:
        return (
            dim
            * math.log(original_max_positions / (rotations * 2 * math.pi))
            / (2 * math.log(theta))
        )

    def attention_mscale(cfg_mscale: float) -> float:
        return 1.0 if factor <= 1 else 0.1 * cfg_mscale * math.log(factor) + 1.0

    low = max(0, math.floor(correction_dim(beta_fast)))
    high = min(dim - 1, math.ceil(correction_dim(beta_slow)))
    # The table's own amplitude factor — 1.0 whenever the two config mscales
    # agree, which is where a model folds mscale into the softmax scale
    # instead (see R1_Q_SCALING).
    amplitude = attention_mscale(mscale) / attention_mscale(mscale_all_dim)

    pos_freqs = theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    ramp = torch.clamp(
        (torch.arange(half, dtype=torch.float32) - low) / max(high - low, 0.001), 0, 1
    )
    # Interpolated (position-stretched) frequencies below the ramp, the
    # original ones above it, blended across it.
    inv_freq = ramp / (factor * pos_freqs) + (1 - ramp) / pos_freqs
    angles = torch.outer(torch.arange(num_positions, dtype=torch.float32), inv_freq)
    angles = torch.cat([angles, angles], dim=-1)  # duplicate_data layout
    table = torch.stack([torch.cos(angles) * amplitude, torch.sin(angles) * amplitude], dim=-1)
    return table.reshape(1, -1).cuda()


def _unscaled_cos_sin_table() -> torch.Tensor:
    """The plain theta-10000 table every MLA case above this section uses."""
    return RopeParams(
        dim=QK_ROPE_HEAD_DIM,
        theta=10000.0,
        max_positions=MLA_MAX_SEQ_LEN,
        duplicate_data=True,
    ).create_rope_const_params()[1]


# A YaRN-table run must land this far outside the tolerance band of a
# reference built from the unscaled theta-10000 table. The separation grows
# with position, because YaRN only rescales the low-frequency half of the
# spectrum and those angles barely differ near position 0: measured on sm_100
# at H=128, page 32, it is 8.6x over a 96-token sequence, 16.9x at 256, 27.0x
# at 512 and 31.7x at 960, against a matching reference's 0.23. The case
# below therefore prefills 512 tokens — 96 would not clear this gate, which
# is the point of choosing a length rather than loosening the gate.
YARN_TABLE_MIN_SEPARATION = 10.0


def test_bf16_mla_yarn_rope_table_h128() -> None:
    """The in-kernel RoPE of the fresh-prefill context call reads the
    rotary_cos_sin table's *content*.

    Pinned here: the YaRN table a DeepSeek-R1-0528 config produces matches a
    plain-torch build of the published formula; truncating it to max_seq_len
    rows is the leading slice of the 163840-row table an engine allocates;
    and the op driven from the torch-built table reproduces a table-aware
    fp32 reference while landing far outside a reference built from the
    unscaled theta-10000 table (and vice versa), so the table content is
    established as load-bearing rather than assumed.
    """
    trtllm_inv_freq, trtllm_table = _r1_rope_params(MLA_MAX_SEQ_LEN).create_rope_const_params()
    torch_table = _yarn_cos_sin_table(
        MLA_MAX_SEQ_LEN,
        QK_ROPE_HEAD_DIM,
        R1_ROPE_THETA,
        R1_ROPE_FACTOR,
        R1_ROPE_ORIGINAL_MAX_POSITIONS,
        R1_ROPE_BETA_FAST,
        R1_ROPE_BETA_SLOW,
        R1_ROPE_MSCALE,
        R1_ROPE_MSCALE_ALL_DIM,
    )
    # Not bitwise: the two evaluate the same blend in different fp32 orders
    # ((1 - (1 - ramp)) against ramp). Observed max abs diff 1.9e-6 on cos/sin
    # values in [-1, 1] — a few fp32 ulp, 19% of this gate — while formula
    # slips land 4-5 orders of magnitude past it: dropping the YaRN
    # interpolation entirely (the unscaled table) or moving beta_fast 32 -> 64
    # both differ by 2.0, and taking the amplitude as mscale rather than the
    # mscale/mscale_all_dim ratio differs by 3.7e-1.
    torch.testing.assert_close(torch_table, trtllm_table, rtol=0.0, atol=1e-5)
    assert trtllm_inv_freq is not None and trtllm_inv_freq.numel() == (QK_ROPE_HEAD_DIM // 2)

    # The unscaled table every MLA case above this section uses is the same
    # construction at factor = 1: the ramp drops out and inv_freq(d) becomes
    # 1/theta^(2d/R). Gated here so the formula is certified for both
    # contents rather than only for the YaRN one (observed 3.8e-6, 38% of
    # the gate).
    unscaled_table = _unscaled_cos_sin_table()
    torch.testing.assert_close(
        _yarn_cos_sin_table(
            MLA_MAX_SEQ_LEN,
            QK_ROPE_HEAD_DIM,
            R1_ROPE_THETA,
            1.0,
            R1_ROPE_ORIGINAL_MAX_POSITIONS,
            R1_ROPE_BETA_FAST,
            R1_ROPE_BETA_SLOW,
            R1_ROPE_MSCALE,
            R1_ROPE_MSCALE_ALL_DIM,
        ),
        unscaled_table,
        rtol=0.0,
        atol=1e-5,
    )

    # Row content does not depend on the row count, so a table truncated to
    # max_seq_len is the production table's leading slice.
    _, full_table = _r1_rope_params(R1_MAX_POSITION_EMBEDDINGS).create_rope_const_params()
    row = QK_ROPE_HEAD_DIM * 2
    assert torch.equal(full_table.view(-1, row)[:MLA_MAX_SEQ_LEN], trtllm_table.view(-1, row))
    del full_table
    torch.cuda.empty_cache()

    h = MLA_NUM_HEADS_H128
    seq_lens = [512, 33]  # far enough past the ramp for YaRN to bite

    def prefill(table: torch.Tensor) -> Tuple[_MlaPagedEnv, torch.Tensor, tuple]:
        torch.manual_seed(936)
        env = _MlaPagedEnv(
            num_heads=h,
            max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
            tokens_per_block=MLA_PAGE32,
            q_scaling=R1_Q_SCALING,
            rope=_r1_rope_params(MLA_MAX_SEQ_LEN),
            rope_scalars=R1_ROPE_SCALARS,
        )
        env.rotary_cos_sin = table  # op and reference both read it here
        for rid, ln in enumerate(seq_lens):
            env.add_request(rid, ln)
        q, k, v, latent = _random_context_inputs(sum(seq_lens), h)
        pre = (list(seq_lens), q.clone(), k.clone(), v, latent.clone())
        out = env.call_context([0, 1], seq_lens, q, k, v, latent)
        return env, out, pre

    for driven, rival, label in (
        (torch_table, unscaled_table, "YaRN table"),
        (unscaled_table, torch_table, "unscaled table"),
    ):
        env, out, pre = prefill(driven)
        torch.testing.assert_close(out, env.context_reference(*pre), rtol=RTOL, atol=ATOL)
        env.rotary_cos_sin = rival
        _assert_outside_band(
            out,
            env.context_reference(*pre),
            YARN_TABLE_MIN_SEPARATION,
            f"{label} run against a reference built from the other table",
        )
        env.rotary_cos_sin = driven
        for rid in (0, 1):
            env.check_cache(rid)  # rope(k_pe) in the append follows the table


def _r1_prefill_observables(
    rope_scalars: _RopeScalars,
    cos_sin: Optional[torch.Tensor] = None,
    inv_freq_mode: str = "keep",
) -> Dict[str, torch.Tensor]:
    """One R1-cell fresh-prefill context call on fixed inputs, returning every
    observable a rope argument could move: the output rows, the whole paged
    latent pool (which receives rope(k_pe)), and the in-place-roped q/k."""
    seq_lens = [96, 33]
    torch.manual_seed(936)
    env = _MlaPagedEnv(
        num_heads=MLA_NUM_HEADS_H128,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
        tokens_per_block=MLA_PAGE32,
        q_scaling=R1_Q_SCALING,
        rope=_r1_rope_params(MLA_MAX_SEQ_LEN),
        rope_scalars=rope_scalars,
    )
    if cos_sin is not None:
        env.rotary_cos_sin = cos_sin
    if inv_freq_mode == "zeros":
        env.rotary_inv_freq = torch.zeros_like(env.rotary_inv_freq)
    elif inv_freq_mode == "none":
        env.rotary_inv_freq = None
    for rid, ln in enumerate(seq_lens):
        env.add_request(rid, ln)
    q, k, v, latent = _random_context_inputs(sum(seq_lens), MLA_NUM_HEADS_H128)
    out = env.call_context([0, 1], seq_lens, q, k, v, latent)
    return {
        "output": out,
        "pool": env.pool.clone(),
        "q_after_call": q,
        "k_after_call": k,
    }


def test_bf16_mla_rope_scalars_inert_h128() -> None:
    """With the table held fixed, the seven scalar rope arguments and
    rotary_inv_freq move no observable of an MLA call.

    That is what makes the YaRN certification portable: a caller supplies the
    scaled table and may pass whatever scalars its config carries. The last
    check is the control that gives the bitwise comparisons meaning — the one
    rope input the op does read (the table) is swapped, and the same
    comparison has to see it.
    """
    base = _r1_prefill_observables(R1_ROPE_SCALARS)
    variants = {
        # The set every MLA case above this section passes.
        "unscaled-config scalars": _r1_prefill_observables(
            _RopeScalars(
                rope_max_positions=MLA_MAX_SEQ_LEN,
                rope_original_max_positions=MLA_MAX_SEQ_LEN,
            )
        ),
        # Run-to-run determinism control for the comparisons around it.
        "R1 scalars again": _r1_prefill_observables(R1_ROPE_SCALARS),
        # Nothing a config would produce: a different theta, a different
        # scaling family, m-scales far from 1, and position windows shorter
        # than the sequences in flight.
        "out-of-range scalars": _r1_prefill_observables(
            _RopeScalars(
                rope_base=500000.0,
                rope_scale_type=3,
                rope_scale=7.5,
                rope_short_m_scale=3.0,
                rope_long_m_scale=9.0,
                rope_max_positions=77,
                rope_original_max_positions=13,
            )
        ),
        "rotary_inv_freq zeroed": _r1_prefill_observables(R1_ROPE_SCALARS, inv_freq_mode="zeros"),
        "rotary_inv_freq=None": _r1_prefill_observables(R1_ROPE_SCALARS, inv_freq_mode="none"),
    }
    for label, observed in variants.items():
        for key, expected in base.items():
            got = observed[key]
            assert torch.equal(expected, got), (
                f"{label}: {key} is not bitwise identical — "
                f"{int((expected != got).sum())}/{expected.numel()} elements, "
                f"max abs {(expected.float() - got.float()).abs().max().item():.3e}"
            )

    swapped = _r1_prefill_observables(R1_ROPE_SCALARS, cos_sin=_unscaled_cos_sin_table())
    for key in ("output", "pool", "q_after_call", "k_after_call"):
        assert not torch.equal(base[key], swapped[key]), (
            f"swapping the rope table left {key} bitwise unchanged — the "
            f"inertness comparisons above cannot see a rope change at all"
        )


# ─── MLA over an fp8-e4m3 latent pool (quant_mode 128) ─────────────────
# The DeepSeek-R1-0528-FP4 cell: H = 128, page 32, C/R/nope/v =
# 512/64/128/128, q_lora_rank 1536, one latent row of C+R e4m3 bytes per
# token. The checkpoint's per-layer k_scale/v_scale are both 1.0, so s = 1.0
# is the production scale; 1.5 and 2.0 are swept beside it because the
# scale tensors are the caller's only defence (None is read as 1.0).


def _fp8_mla_env(
    num_heads: int = MLA_NUM_HEADS_H128,
    kv_scaling_factor: Optional[float] = 1.0,
    q_scaling: float = 1.0,
    rope: Optional[RopeParams] = None,
    rope_scalars: Optional[_RopeScalars] = None,
    quant_mode: int = QUANT_MODE_FP8_KV_CACHE,
) -> _MlaPagedEnv:
    """Page-32 MLA env over an fp8-e4m3 latent pool at the R1 head geometry.

    kv_scaling_factor=None leaves both kv scale tensors unpassed, which is the
    silent-1.0 case a caller hits by forgetting them. q_scaling/rope/
    rope_scalars default to the baseline cell (unscaled theta-10000 table,
    scale 1.0); _fp8_r1_env below fills the DeepSeek-R1-0528 values.
    """
    return _MlaPagedEnv(
        num_heads=num_heads,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
        tokens_per_block=MLA_PAGE32,
        q_lora_rank=Q_LORA_RANK_DSV3,
        q_scaling=q_scaling,
        rope=rope,
        rope_scalars=rope_scalars,
        pool_dtype=torch.float8_e4m3fn,
        quant_mode=quant_mode,
        kv_scaling_factor=kv_scaling_factor,
    )


def _fp8_mla_prefill(
    env: _MlaPagedEnv, seq_lens: List[int], seed: int
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """One fresh-prefill context call over the fp8 pool, returning the output
    and the pre-call (q, k, v, latent) a reference has to be built from (the
    call clobbers the q/k rope slices in place)."""
    torch.manual_seed(seed)
    q, k, v, latent = _random_context_inputs(sum(seq_lens), env.num_heads)
    pre = (q.clone(), k.clone(), v, latent.clone())
    for rid, ln in enumerate(seq_lens):
        env.add_request(rid, ln)
    out = env.call_context(list(range(len(seq_lens))), seq_lens, q, k, v, latent)
    return out, pre


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


def test_fp8_mla_context_prefill_h128() -> None:
    """MLA fresh-prefill context over an fp8-e4m3 latent pool, s = 1.0.

    Two things the standard configuration's fp8 section does NOT carry over,
    both measured here rather than inherited:

    1. The context FMHA does *not* stay in bf16. Under quant_mode 128 the MLA
       context path quantizes q, k and v to e4m3 (at scale 1.0 — the pool's
       write scale is not applied to them) and runs the FMHA on those, so
       prefill accuracy *is* affected by cache quantization. The bf16-KV
       reference — the one the bf16 pool is certified against, and what a
       caller would assume from the standard-configuration section — sits far
       outside the bf16 band (observed 26x), while the e4m3-input reference
       matches within the fp8 band (observed 49% of it). The sharp form of
       this claim is the peaked-attention readout in the next case.
    2. The appended latent row lands as e4m3(row * kv_scale_orig_quant) on
       *both* halves — compressed_kv bitwise, and the in-kernel-roped k_pe
       half within the e4m3-ulp gate (observed bitwise too).

    The requests' pages are deliberately scattered and out of order, so a page
    stride computed with the wrong element width (2-byte bf16 rather than
    1-byte e4m3) lands in a page this test then finds non-zero.
    """
    env = _fp8_mla_env()
    lens = [96, 33]  # three exact 32-slot pages, and a one-token spill
    torch.manual_seed(400)
    env.add_request(0, lens[0])
    env.add_request(1, lens[1])
    # Hand-assigned, complete page sets (3 pages for 96 tokens, 2 for 33), so
    # the env allocates none of its own: scattered and non-monotonic, sharing
    # no page between the two requests.
    env.pages[0] = [5, 1, 9]
    env.pages[1] = [4, 7]
    q, k, v, latent = _random_context_inputs(sum(lens), env.num_heads)
    pre = (q.clone(), k.clone(), v, latent.clone())
    out = env.call_context([0, 1], lens, q, k, v, latent)

    ref = env.context_reference(lens, *pre, e4m3_inputs=True)
    torch.testing.assert_close(out, ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
    # Rival: the bf16-KV reference, i.e. "cache quantization does not reach
    # the context math". Gated at 10x the bf16 band; observed 32x.
    _assert_far_outside_band(
        out,
        env.context_reference(lens, *pre),
        10.0,
        "fp8 MLA context against the bf16-KV reference",
        ATOL,
        RTOL,
    )
    for rid in (0, 1):
        env.check_cache(rid)
    env.check_unwritten_pool_zero([0, 1])


def test_fp8_mla_context_v_operand_is_e4m3_h128() -> None:
    """The context FMHA's V operand really is e4m3, read out bitwise.

    The realistic-input case above compares two models that differ by the size
    of the tolerance itself, so it cannot settle "is the context math fp8?" on
    its own. This one can: give key 0 a score every other key cannot approach
    (matching nope halves at magnitude 4, k_pe zeroed so the roped tails
    contribute nothing) and every query row's softmax collapses onto it, so the
    output row *is* V row 0. It comes back as e4m3(V[0]) — bit-exact, max abs
    deviation 0, with only signed zeros differing in raw bytes — while the
    unquantized V[0] a bf16-pool call would return is up to 63 bf16 ulps away
    and matches in only 52% of its bytes.
    """
    env = _fp8_mla_env()
    seq_len = 40
    torch.manual_seed(500)
    h = env.num_heads
    q = torch.zeros(seq_len, h, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    q[:, :, :QK_NOPE_HEAD_DIM] = 4.0
    k = torch.zeros(seq_len, h, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k[0, :, :QK_NOPE_HEAD_DIM] = 4.0
    packed = torch.zeros(
        seq_len,
        h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
    )
    packed[:, : h * QK_NOPE_HEAD_DIM] = k[:, :, :QK_NOPE_HEAD_DIM].reshape(
        seq_len, h * QK_NOPE_HEAD_DIM
    )
    v = _v_split_view(packed, h)
    v.view(seq_len, h, V_HEAD_DIM).copy_(
        torch.randn(seq_len, h, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    )
    latent = torch.randn(seq_len, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    latent[:, KV_LORA_RANK:] = 0.0
    v0 = v.view(seq_len, h, V_HEAD_DIM)[0].clone()

    env.add_request(0, seq_len)
    rows = env.call_context(
        [0],
        [seq_len],
        q.view(seq_len, h * QK_HEAD_DIM),
        k.view(seq_len, h * QK_HEAD_DIM),
        v,
        latent,
    ).view(seq_len, h, V_HEAD_DIM)

    quantized = v0.float().to(torch.float8_e4m3fn).float().to(torch.bfloat16)
    expected = quantized.unsqueeze(0).expand_as(rows)
    torch.testing.assert_close(rows, expected, rtol=2**-8, atol=0.0)  # one bf16 ulp
    exact = float(
        (rows.reshape(-1).view(torch.uint8) == expected.reshape(-1).view(torch.uint8))
        .float()
        .mean()
    )
    assert exact > 0.99, f"only {exact:.4f} of the readout is bit-exact e4m3(V[0])"
    rival = v0.float().unsqueeze(0)
    ulps = float(((rows.float() - rival).abs() / (2**-8 * rival.abs().clamp(min=2**-8))).max())
    assert ulps > 8.0, (
        f"the unquantized V row is only {ulps:.3g} bf16 ulps away — this probe "
        f"cannot tell an e4m3 V operand from a bf16 one"
    )


def test_fp8_mla_generation_decode_h128() -> None:
    """MLA generation decode over the fp8-e4m3 latent pool, s = 1.0.

    Histories of 64 (two exact pages, so the first decode token opens page 2)
    and 31 (first decode token takes page 0's last slot, second opens page 1),
    two steps. The decode kernel reads its query from quant_q_buffer, never
    from `q`, and applies no kv scale of its own — the caller's
    mla_bmm1_scale/mla_bmm2_scale carry the dequantization (see
    fp8_decode_buffers). What the call then computes is latent MQA over the
    e4m3 round trip of both operands, which is what generation_reference
    builds. The pool must come back bitwise unchanged: the generation phase
    reads it and nothing else.
    """
    env = _fp8_mla_env()
    prefill = [64, 31]
    _fp8_mla_prefill(env, prefill, 401)
    for rid in (0, 1):
        env.check_cache(rid)
    for step in range(2):
        for rid in (0, 1):
            env.append_decode_latent(
                rid,
                torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5,
            )
        pool_before = env.pool.clone()
        fused_q = (
            torch.randn(2, env.num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
        )
        out = env.call_generation([0, 1], fused_q)
        ref = env.generation_reference([0, 1], fused_q)
        torch.testing.assert_close(out, ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
        assert _bitwise_equal(pool_before, env.pool), (
            "the MLA generation call wrote to the fp8 pool"
        )


def test_fp8_mla_decode_buffer_roles_h128() -> None:
    """The three buffers an fp8-pool MLA generation call requires, one role at
    a time. All three are presence-checked by the op; what each one carries is
    unchecked, so a caller gets these wrong silently.

    - quant_q_buffer holds the query: replacing `q` with garbage leaves the
      output bitwise identical, so `q` is not read on this path at all.
    - mla_bmm1_scale[1] (the log2-domain copy) is the softmax scale the kernel
      applies; element [0] is inert.
    - mla_bmm2_scale[0] multiplies the output.
    """
    env = _fp8_mla_env()
    _fp8_mla_prefill(env, [40], 402)
    torch.manual_seed(403)
    env.append_decode_latent(0, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5)
    fused_q = torch.randn(1, env.num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
    quant_q, bmm1, bmm2 = env.fp8_decode_buffers(fused_q)
    ref = env.generation_reference([0], fused_q)

    base = env.call_generation([0], fused_q, fp8_buffers=(quant_q, bmm1, bmm2))
    torch.testing.assert_close(base, ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)

    for label, buffers, message in (
        ("quant_q_buffer", (None, bmm1, bmm2), "quant_q_buf is nullptr"),
        ("mla_bmm1_scale", (quant_q, None, bmm2), "bmm1_scale is nullptr"),
        ("mla_bmm2_scale", (quant_q, bmm1, None), "bmm2_scale is nullptr"),
    ):
        try:
            env.call_generation([0], fused_q, fp8_buffers=buffers)
        except RuntimeError as exc:
            assert message in str(exc), f"{label}: unexpected message: {exc}"
        else:
            raise AssertionError(f"{label}=None was accepted under quant_mode 128")

    garbage_q = torch.randn_like(fused_q) * 4
    with_garbage = env.call_generation([0], garbage_q, fp8_buffers=(quant_q, bmm1, bmm2))
    assert _bitwise_equal(base, with_garbage), (
        "replacing q moved the output — the decode does read the bf16 query"
    )
    # The buffer is consumed as raw bytes through a pointer, so the same bytes
    # handed over as uint8 are the same query: nothing checks its dtype.
    assert _bitwise_equal(
        base,
        env.call_generation([0], fused_q, fp8_buffers=(quant_q.view(torch.uint8), bmm1, bmm2)),
    ), "quant_q_buffer's dtype changed the result"

    inert_head = torch.tensor([0.0, float(bmm1[1])], dtype=torch.float32, device="cuda")
    assert _bitwise_equal(
        base, env.call_generation([0], fused_q, fp8_buffers=(quant_q, inert_head, bmm2))
    ), "mla_bmm1_scale[0] moved the output"
    dead_log2 = torch.tensor([float(bmm1[0]), 0.0], dtype=torch.float32, device="cuda")
    # scale 0 flattens the softmax to a uniform average of the cached rows.
    _assert_far_outside_band(
        env.call_generation([0], fused_q, fp8_buffers=(quant_q, dead_log2, bmm2)),
        ref,
        2.0,
        "mla_bmm1_scale[1] zeroed",
        FP8_MLA_ATOL,
        FP8_MLA_RTOL,
    )

    doubled = torch.tensor([2.0 * env.kv_scale], dtype=torch.float32, device="cuda")
    out2 = env.call_generation([0], fused_q, fp8_buffers=(quant_q, bmm1, doubled))
    torch.testing.assert_close(
        out2, (ref.float() * 2.0).to(ref.dtype), rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL
    )


def test_fp8_mla_kv_scale_semantics_h128() -> None:
    """What the two kv scale tensors do on the MLA path, per phase.

    Write side (both phases' appends): kv_scale_orig_quant is applied, and the
    row lands bit-exactly as e4m3(row * orig_quant) at every scale.

    Context read side: the FMHA quantizes q/k/v at scale 1.0 but still applies
    the dequantization factors s**2 (softmax scale) and s (output), so at
    s != 1.0 the context result is silently wrong — the s = 1.0 math sits
    21x-48x outside the fp8 band, while the distorted model reproduces the run.

    Generation read side: neither scale tensor is read at all. The dequant has
    to arrive folded into mla_bmm1_scale/mla_bmm2_scale (as the production
    generation-preprocessing step writes them), and once it does, decode is
    correct at any scale — verified at s = 2.0, where dropping the fold lands
    3.6x outside the band.

    Passing None for both is bitwise identical to passing 1.0 tensors, in the
    output and in the pool: a forgotten scale is a wrong-number bug at any
    other s, not a crash.
    """
    lens = [48, 17]
    for scale in KV_SCALE_SWEEP:
        env = _fp8_mla_env(kv_scaling_factor=scale)
        out, pre = _fp8_mla_prefill(env, lens, 404)
        for rid in (0, 1):
            env.check_cache(rid)  # e4m3(row * orig_quant), both halves
        env.check_unwritten_pool_zero([0, 1])
        true_ref = env.context_reference(lens, *pre, e4m3_inputs=True)
        distorted = env.context_reference(lens, *pre, e4m3_inputs=True, kv_scale_factors=True)
        if scale == 1.0:
            torch.testing.assert_close(out, true_ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
        else:
            _assert_far_outside_band(
                out,
                true_ref,
                10.0,
                f"fp8 MLA context at s={scale} against the s=1.0 math",
                FP8_MLA_ATOL,
                FP8_MLA_RTOL,
            )
            # The distortion is exactly the two kv-scale factors: the model
            # lands inside the fp8 band itself (observed 0.58x at s=1.5 and
            # 0.84x at s=2.0, against 21x / 48x for the true-value hypothesis),
            # so it is gated at 1.5x rather than 1.0x only to leave the same
            # headroom the certified s=1.0 case has.
            assert _band_fraction(out, distorted, FP8_MLA_ATOL, FP8_MLA_RTOL) < 1.5, (
                f"the s**2 / s model does not explain the s={scale} run"
            )

    # None in both slots == 1.0 tensors, bitwise, in the output and the pool.
    env_none = _fp8_mla_env(kv_scaling_factor=None)
    out_none, _ = _fp8_mla_prefill(env_none, lens, 404)
    env_one = _fp8_mla_env(kv_scaling_factor=1.0)
    out_one, _ = _fp8_mla_prefill(env_one, lens, 404)
    assert _bitwise_equal(out_none, out_one) and _bitwise_equal(env_none.pool, env_one.pool), (
        "kv_scale_* = None is not the s = 1.0 path"
    )

    # Generation at s != 1.0: correct with the folded scales — at a
    # non-power-of-two scale too, where the e4m3 grid genuinely depends on the
    # value — and blind to the scale tensors themselves.
    for scale in (1.5, 2.0):
        env = _fp8_mla_env(kv_scaling_factor=scale)
        _fp8_mla_prefill(env, [40], 405)
        torch.manual_seed(406)
        env.append_decode_latent(
            0, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
        )
        fused_q = (
            torch.randn(1, env.num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
        )
        quant_q, bmm1, bmm2 = env.fp8_decode_buffers(fused_q)
        out = env.call_generation([0], fused_q, fp8_buffers=(quant_q, bmm1, bmm2))
        ref = env.generation_reference([0], fused_q)
        torch.testing.assert_close(out, ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
        # Unfolded scales — what a caller gets by reading the standard
        # configuration's fp8 section and assuming kv_scale_quant_orig is
        # applied on read.
        unfolded_1 = torch.tensor(
            [env.softmax_scale, env.softmax_scale * math.log2(math.e)],
            dtype=torch.float32,
            device="cuda",
        )
        unfolded_2 = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        _assert_far_outside_band(
            env.call_generation([0], fused_q, fp8_buffers=(quant_q, unfolded_1, unfolded_2)),
            ref,
            2.0,
            f"fp8 MLA decode at s={scale} with the kv scale left out of the bmm scales",
            FP8_MLA_ATOL,
            FP8_MLA_RTOL,
        )
        # The scale tensors themselves reach nothing: dropping both is bitwise
        # identical at a scale where a read-side dequant would be visible.
        env.kv_scale_orig_quant = None
        env.kv_scale_quant_orig = None
        assert _bitwise_equal(
            out, env.call_generation([0], fused_q, fp8_buffers=(quant_q, bmm1, bmm2))
        ), "the MLA decode path does read the kv scale tensors"


# ─── The complete DeepSeek-R1-0528 cell over the fp8-e4m3 latent pool ──
#
# The fp8 cases above run the unscaled theta-10000 table at q_scaling = 1.0;
# the R1-cell cases further up run the YaRN table and q_scaling = 1/mscale^2
# over a *bf16* pool. A DeepSeek-R1-0528 rank passes all three in every call,
# and certifying axes separately does not certify their combination, so the
# whole cell runs here: fp8 pool + YaRN table + q_scaling, H = 128, page 32,
# C/R/nope/v = 512/64/128/128, q_lora_rank 1536, s = 1.0.


def _fp8_r1_env(**kwargs) -> _MlaPagedEnv:
    """_fp8_mla_env at the complete R1 cell: the checkpoint's YaRN rope table,
    its scalar rope set, and q_scaling = 1/mscale^2."""
    kwargs.setdefault("q_scaling", R1_Q_SCALING)
    kwargs.setdefault("rope", _r1_rope_params(MLA_MAX_SEQ_LEN))
    kwargs.setdefault("rope_scalars", R1_ROPE_SCALARS)
    return _fp8_mla_env(**kwargs)


def _unscaled_table_env(**kwargs) -> _MlaPagedEnv:
    """The same env on the *unscaled* theta-10000 table — the rival whose
    references and append mirrors show that the table content is what the
    kernel reads. Only its rope table and the mirrors built from it are used;
    its pool is never called into."""
    kwargs.setdefault("q_scaling", R1_Q_SCALING)
    return _fp8_mla_env(**kwargs)


def _rope_rows(env: _MlaPagedEnv, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
    """The latent rows a fresh-prefill append is expected to leave in the pool
    for one sequence, roped position by position with env's table."""
    rows = latent[:seq_len].clone()
    for i in range(seq_len):
        rows[i, KV_LORA_RANK:] = env.rope_ref(latent[i, KV_LORA_RANK:], i)
    return rows


def _assert_table_pins_the_append(
    env: _MlaPagedEnv,
    rival_env: _MlaPagedEnv,
    request_id: int,
    latent: torch.Tensor,
    seq_len: int,
) -> None:
    """check_cache's bit-exact comparison must *fail* against a mirror roped by
    the unscaled table.

    The appended k_pe half is the sharpest place the rope table shows up: it is
    gated bitwise rather than by tolerance. But a bit-exact match proves nothing
    about which table was read unless the same comparison can tell the two
    tables apart at the positions in flight, and YaRN rescales only the
    low-frequency half of the spectrum, so the two agree closely at small
    positions — exactly where a 96-token prefill lives. Measured here at
    L = 96, page 32: 2336 of the 6144 e4m3 bytes of the roped half differ
    between the two mirrors (38.0%), against an allowance of 6, so the
    comparison separates them by ~389x.
    """
    correct = env.latent_rows[request_id]
    env.latent_rows[request_id] = [_rope_rows(rival_env, latent, seq_len)]
    try:
        env.check_cache(request_id)
    except AssertionError:
        pass
    else:
        raise AssertionError(
            "the appended rows match a mirror roped by the unscaled table too — "
            "this comparison cannot tell the two rope tables apart here"
        )
    finally:
        env.latent_rows[request_id] = correct


def test_fp8_mla_r1_cell_context_prefill_h128() -> None:
    """Fresh-prefill MLA context at the complete R1 cell over an fp8-e4m3
    latent pool: 128/128/192, page 32, YaRN table, q_scaling = 1/mscale^2.

    All three axes are gated, not just run: the output matches an fp32
    reference over e4m3-rounded q/k/v built from this table and this scale
    (observed 40% of the fp8 band), and sits far outside three rivals — the
    bf16-KV reference the bf16 pool is certified against (49.8x the bf16 band),
    a reference at q_scaling = 1.0 (12.7x the fp8 band), and one built from the
    unscaled theta-10000 table (3.7x). The bit-exact append carries the table
    a second time and much harder, with its own rival control.
    """
    env = _fp8_r1_env()
    rival = _unscaled_table_env()
    lens = [96, 33]  # three exact 32-slot pages, and a one-token spill
    torch.manual_seed(950)
    for rid, ln in enumerate(lens):
        env.add_request(rid, ln)
    q, k, v, latent = _random_context_inputs(sum(lens), env.num_heads)
    pre = (q.clone(), k.clone(), v, latent.clone())
    out = env.call_context([0, 1], lens, q, k, v, latent)

    torch.testing.assert_close(
        out,
        env.context_reference(lens, *pre, e4m3_inputs=True),
        rtol=FP8_MLA_RTOL,
        atol=FP8_MLA_ATOL,
    )
    _assert_far_outside_band(
        out,
        env.context_reference(lens, *pre),
        10.0,
        "fp8 R1-cell MLA context against the bf16-KV reference",
        ATOL,
        RTOL,
    )
    _assert_far_outside_band(
        out,
        env.context_reference(
            lens,
            *pre,
            e4m3_inputs=True,
            softmax_scale=1.0 / math.sqrt(QK_HEAD_DIM),
        ),
        4.0,
        "fp8 R1-cell MLA context against a q_scaling = 1.0 reference",
        FP8_MLA_ATOL,
        FP8_MLA_RTOL,
    )
    _assert_far_outside_band(
        out,
        rival.context_reference(lens, *pre, e4m3_inputs=True),
        2.0,
        "fp8 R1-cell MLA context against an unscaled-rope-table reference",
        FP8_MLA_ATOL,
        FP8_MLA_RTOL,
    )

    for rid in (0, 1):
        env.check_cache(rid)
    env.check_unwritten_pool_zero([0, 1])
    _assert_table_pins_the_append(env, rival, 0, pre[3], lens[0])


def test_fp8_mla_r1_cell_generation_decode_h128() -> None:
    """Latent-MQA decode at the complete R1 cell over the fp8-e4m3 pool.

    Nothing is rotated in this phase, so of the cell's three axes only
    q_scaling has an effect here — it reaches the kernel through the caller's
    mla_bmm1_scale, which fp8_decode_buffers builds from the env's softmax
    scale. Histories of 64 (two exact pages, first decode token opens page 2)
    and 31 (first decode token takes page 0's last slot, second opens page 1),
    two steps. Gated against a reference at q_scaling = 1.0 (observed 4.2x the
    fp8 band, where the matching reference uses 0.20).
    """
    env = _fp8_r1_env()
    prefill = [64, 31]
    _fp8_mla_prefill(env, prefill, 951)
    for rid in (0, 1):
        env.check_cache(rid)
    for _ in range(2):
        for rid in (0, 1):
            env.append_decode_latent(
                rid,
                torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5,
            )
        pool_before = env.pool.clone()
        fused_q = (
            torch.randn(2, env.num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
        )
        out = env.call_generation([0, 1], fused_q)
        torch.testing.assert_close(
            out,
            env.generation_reference([0, 1], fused_q),
            rtol=FP8_MLA_RTOL,
            atol=FP8_MLA_ATOL,
        )
        _assert_far_outside_band(
            out,
            env.generation_reference([0, 1], fused_q, softmax_scale=1.0 / math.sqrt(QK_HEAD_DIM)),
            2.0,
            "fp8 R1-cell MLA decode against a q_scaling = 1.0 reference",
            FP8_MLA_ATOL,
            FP8_MLA_RTOL,
        )
        assert _bitwise_equal(pool_before, env.pool), (
            "the MLA generation call wrote to the fp8 pool"
        )


def test_fp8_mla_r1_cell_mixed_batch_h128() -> None:
    """A mixed batch over the fp8-e4m3 pool at the complete R1 cell: a 64-token
    history decoded next to a fresh 33-token context sequence, the two phase
    calls sharing one set of batch tensors and one page-32 offsets table.

    This is the pairing dispatch 02 did not run under fp8: the context call
    carries a trailing generation sequence in its metadata (and must ignore its
    rows), and the generation call carries a leading context sequence (and must
    index its own from num_contexts). Both appends are checked bit-exactly
    afterwards, so neither call may have written into the other's pages.
    """
    env = _fp8_r1_env()
    first, second = 64, 33
    torch.manual_seed(952)
    env.add_request(0, first)
    q, k, v, latent = _random_context_inputs(first, env.num_heads)
    env.call_context([0], [first], q, k, v, latent)

    env.add_request(1, second)
    env.append_decode_latent(0, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5)
    q, k, v, latent = _random_context_inputs(second, env.num_heads)
    pre = (q.clone(), k.clone(), v, latent.clone())
    out_ctx = env.call_context([1], [second], q, k, v, latent, gen_request_ids=[0])
    torch.testing.assert_close(
        out_ctx,
        env.context_reference([second], *pre, e4m3_inputs=True),
        rtol=FP8_MLA_RTOL,
        atol=FP8_MLA_ATOL,
    )
    _assert_far_outside_band(
        out_ctx,
        env.context_reference([second], *pre),
        10.0,
        "fp8 R1-cell mixed-batch context against the bf16-KV reference",
        ATOL,
        RTOL,
    )

    fused_q = torch.randn(1, env.num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
    out_gen = env.call_generation([0], fused_q, ctx_request_ids=[1], ctx_seq_lens=[second])
    torch.testing.assert_close(
        out_gen,
        env.generation_reference([0], fused_q),
        rtol=FP8_MLA_RTOL,
        atol=FP8_MLA_ATOL,
    )
    env.check_cache(0)
    env.check_cache(1)
    env.check_unwritten_pool_zero([0, 1])


def _fp8_kv_b_proj(num_heads: int) -> torch.Tensor:
    """A fixed absorbed-KV projection weight `[C, H*(nope+v)]`, scaled by
    1/sqrt(C) so its output keeps the unit magnitude every other operand in
    this file has. Stands in for the checkpoint's kv_b_proj, which is what
    turns a gathered compressed_kv row into K's nope half and V."""
    g = torch.Generator(device="cuda").manual_seed(9001)
    w = torch.randn(
        KV_LORA_RANK,
        num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
        generator=g,
        dtype=torch.float32,
        device="cuda",
    )
    return (w / math.sqrt(KV_LORA_RANK)).to(torch.bfloat16)


def _fp8_pool_backed_kv(
    env: _MlaPagedEnv, cached_lens: List[int], new_lens: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a no-append context call's K/V the way a target on an fp8 latent
    pool builds them, and write the cached prefixes into the pool.

    Cached rows are stored exactly as the op's own append stores them,
    `e4m3(row * kv_scale_orig_quant)`, and read back with the fp8 latent
    gather's formula — `bf16(float(cache_byte) * kv_scale_quant_orig)`,
    mirrored in plain torch here so the operands depend on no other entry. The
    gathered compressed_kv goes through _fp8_kv_b_proj into the packed
    `[nope | v]` buffer the context FMHA wants; the gathered k_pe is already
    roped (that is how the pool holds it) and becomes k's per-head tail. The
    new tokens never touch the pool: their latent is fresh and their k_pe is
    roped here at its absolute position, which is what the caller does.

    Returns `(k, v)` with every sequence's `[cached | new]` range concatenated
    in batch order, `v` carrying the required `H * (nope + v)` row stride.
    """
    h = env.num_heads
    w = _fp8_kv_b_proj(h).float()
    k_parts: List[torch.Tensor] = []
    packed_parts: List[torch.Tensor] = []
    for rid, (c_len, n_len) in enumerate(zip(cached_lens, new_lens)):
        cached = torch.randn(c_len, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        for i in range(c_len):
            cached[i, KV_LORA_RANK:] = env.rope_ref(cached[i, KV_LORA_RANK:], i)
        stored = env.to_pool(cached)
        tpb = env.tokens_per_block
        for i in range(c_len):
            env.pool[env.pages[rid][i // tpb], i % tpb] = stored[i]
        gathered = (stored.float() * env.kv_scale).to(torch.bfloat16)

        fresh = torch.randn(n_len, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
        for i in range(n_len):
            fresh[i, KV_LORA_RANK:] = env.rope_ref(fresh[i, KV_LORA_RANK:], c_len + i)

        rows = torch.cat([gathered, fresh])
        total = c_len + n_len
        packed = (rows[:, :KV_LORA_RANK].float() @ w).to(torch.bfloat16)
        k = torch.empty(total, h, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k[..., :QK_NOPE_HEAD_DIM] = packed[:, : h * QK_NOPE_HEAD_DIM].view(
            total, h, QK_NOPE_HEAD_DIM
        )
        k[..., QK_NOPE_HEAD_DIM:] = rows[:, KV_LORA_RANK:].unsqueeze(1).expand(-1, h, -1)
        k_parts.append(k.reshape(total, h * QK_HEAD_DIM))
        packed_parts.append(packed)
    return torch.cat(k_parts), _v_split_view(torch.cat(packed_parts), h)


def _fp8_no_append_reference(
    env: _MlaPagedEnv,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    new_lens: List[int],
    kv_lens: List[int],
    e4m3_inputs: bool = True,
    kv_scale_factors: bool = False,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """The no-append context flavor's fp8 reference — the counterpart of
    _MlaPagedEnv.context_reference for this flavor's explicit K/V.

    e4m3_inputs (the default here) rounds q, k and v through e4m3 at scale 1.0,
    which is what the call quantizes them to under quant_mode 128; setting it
    False builds the bf16-KV rival. kv_scale_factors adds the two factors that
    path applies on top — s**2 on the softmax scale and s on the output —
    which cancel only at s = 1.0. softmax_scale overrides the env's own scale,
    which is what builds a q_scaling rival.
    """
    scale = env.softmax_scale if softmax_scale is None else softmax_scale
    s = env.kv_scale if kv_scale_factors else 1.0
    return _explicit_kv_reference(
        q,
        k,
        v,
        new_lens,
        kv_lens,
        MASK_CAUSAL,
        env.num_heads,
        scale * s * s,
        e4m3_inputs=e4m3_inputs,
        out_scale=s,
    )[0]


def test_fp8_mla_r1_cell_context_cached_kv_no_append_h128() -> None:
    """Cached-KV (no-append) MLA context over an fp8-e4m3 latent pool, at the
    complete R1 cell. This is the flavor an engine with block reuse on runs for
    every context request that hits a cached prefix, and it had never been run
    over an fp8 pool.

    Its K/V come from where production's come from: the cached prefix is real
    e4m3 pool content read back through the fp8 latent gather's formula and put
    through a kv_b_proj-shaped matmul; only the new tokens are fresh. Prefixes
    of 96 / 31 / 0 reach KV lengths of 128 / 40 / 25.

    The call quantizes those operands to e4m3 itself, exactly as the
    fresh-prefill flavor does (settled bitwise in the next test) — so the
    reference rounds q/k/v through e4m3, observed at 41% of the fp8 band, with
    the bf16-KV model 2.2x outside it and a q_scaling = 1.0 reference 10.8x.
    Nothing is mutated: q, k, v and the whole pool come back bitwise intact.
    """
    env = _fp8_r1_env()
    cached_lens = [96, 31, 0]
    new_lens = [32, 9, 25]
    kv_lens = [c + n for c, n in zip(cached_lens, new_lens)]
    torch.manual_seed(953)
    for rid, total in enumerate(kv_lens):
        env.add_request(rid, new_lens[rid])
        env.reserve_cache_pages(rid, total)
    k, v = _fp8_pool_backed_kv(env, cached_lens, new_lens)
    q = torch.randn(sum(new_lens), env.num_heads * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    q_orig, k_orig, v_orig = q.clone(), k.clone(), v.clone()
    pool_before = env.pool.clone()

    out = env.call_context_no_append([0, 1, 2], new_lens, kv_lens, q, k, v)
    operands = (q, k, v, new_lens, kv_lens)

    torch.testing.assert_close(
        out,
        _fp8_no_append_reference(env, *operands),
        rtol=FP8_MLA_RTOL,
        atol=FP8_MLA_ATOL,
    )
    _assert_far_outside_band(
        out,
        _fp8_no_append_reference(env, *operands, e4m3_inputs=False),
        1.5,
        "fp8 no-append MLA context against the bf16-KV reference",
        FP8_MLA_ATOL,
        FP8_MLA_RTOL,
    )
    _assert_far_outside_band(
        out,
        _fp8_no_append_reference(env, *operands, softmax_scale=1.0 / math.sqrt(QK_HEAD_DIM)),
        4.0,
        "fp8 no-append MLA context against a q_scaling = 1.0 reference",
        FP8_MLA_ATOL,
        FP8_MLA_RTOL,
    )

    assert torch.equal(q, q_orig)
    assert torch.equal(k, k_orig)
    assert torch.equal(v, v_orig)
    assert _bitwise_equal(env.pool, pool_before), (
        "the no-append MLA context call touched the fp8 pool"
    )


def test_fp8_mla_no_append_operands_and_scale_h128() -> None:
    """What quant_mode 128 does to the no-append context flavor, settled the
    two ways the realistic case cannot settle on its own.

    1. The operands really are e4m3 — the same fact the fresh-prefill flavor
       has, and it could not be assumed here, because this flavor's K/V arrive
       already dequantized from an fp8 pool and it appends nothing. Peaked
       readout: give key 0 a score no other key approaches, and every output
       row *is* V row 0. It comes back as e4m3(V[0]) with max abs deviation 0
       (99.98% of the raw bytes equal, the rest signed zeros), while the
       unquantized V[0] a bf16-pool call would return is 60.5 bf16 ulps away.
    2. The two kv-scale factors land on it exactly as they land on the fresh
       flavor: quantization at 1.0, then s**2 on the softmax scale and s on the
       output. At s = 2.0 the peaked readout returns 2 * e4m3(V[0]) bit for
       bit, and on realistic inputs the true-value model sits 48x outside the
       band while the distorted one explains the run at 0.86 of it. So this
       flavor is correct at s = 1.0 alone, like the other context flavor.
    """
    h = MLA_NUM_HEADS_H128
    seq_len = 40
    for scale in KV_SCALE_SWEEP:
        env = _fp8_r1_env(kv_scaling_factor=scale)
        torch.manual_seed(700)
        env.add_request(0, seq_len)
        env.reserve_cache_pages(0, seq_len)
        # Matching nope halves at magnitude 4 on key 0 only; k_pe zeroed so the
        # roped tails contribute nothing to any score.
        q = torch.zeros(seq_len, h, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        q[:, :, :QK_NOPE_HEAD_DIM] = 4.0
        k = torch.zeros(seq_len, h, QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k[0, :, :QK_NOPE_HEAD_DIM] = 4.0
        packed = torch.zeros(
            seq_len,
            h * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        packed[:, : h * QK_NOPE_HEAD_DIM] = k[:, :, :QK_NOPE_HEAD_DIM].reshape(
            seq_len, h * QK_NOPE_HEAD_DIM
        )
        v = _v_split_view(packed, h)
        v.view(seq_len, h, V_HEAD_DIM).copy_(
            torch.randn(seq_len, h, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        )
        v0 = v.view(seq_len, h, V_HEAD_DIM)[0].clone()

        rows = env.call_context_no_append(
            [0],
            [seq_len],
            [seq_len],
            q.view(seq_len, h * QK_HEAD_DIM),
            k.view(seq_len, h * QK_HEAD_DIM),
            v,
        ).view(seq_len, h, V_HEAD_DIM)

        expected = (
            (v0.float().to(torch.float8_e4m3fn).float() * scale)
            .to(torch.bfloat16)
            .unsqueeze(0)
            .expand_as(rows)
        )
        torch.testing.assert_close(rows, expected, rtol=2**-8, atol=0.0)  # one bf16 ulp
        exact = float(
            (rows.reshape(-1).view(torch.uint8) == expected.reshape(-1).view(torch.uint8))
            .float()
            .mean()
        )
        assert exact > 0.99, (
            f"only {exact:.4f} of the readout is bit-exact {scale} * e4m3(V[0]) at s={scale}"
        )
        rival = (v0.float() * scale).unsqueeze(0)
        ulps = float(((rows.float() - rival).abs() / (2**-8 * rival.abs().clamp(min=2**-8))).max())
        assert ulps > 8.0, (
            f"the unquantized V row is only {ulps:.3g} bf16 ulps away at "
            f"s={scale} — this probe cannot tell an e4m3 V operand from a bf16 one"
        )

    # Realistic inputs: which of the two models explains a run at s != 1.0.
    new_lens = [32, 9]
    kv_lens = [64, 24]
    for scale in KV_SCALE_SWEEP:
        env = _fp8_r1_env(kv_scaling_factor=scale)
        torch.manual_seed(954)
        for rid, total in enumerate(kv_lens):
            env.add_request(rid, new_lens[rid])
            env.reserve_cache_pages(rid, total)
        q = torch.randn(sum(new_lens), h * QK_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        k, packed = _random_explicit_kv(sum(kv_lens), h)
        v = _v_split_view(packed, h)
        out = env.call_context_no_append([0, 1], new_lens, kv_lens, q, k, v)
        operands = (q, k, v, new_lens, kv_lens)

        true_ref = _fp8_no_append_reference(env, *operands)
        if scale == 1.0:
            torch.testing.assert_close(out, true_ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
        else:
            _assert_far_outside_band(
                out,
                true_ref,
                10.0,
                f"fp8 no-append MLA context at s={scale} against the s=1.0 math",
                FP8_MLA_ATOL,
                FP8_MLA_RTOL,
            )
            # Same 1.5x allowance the fresh-prefill scale case uses: the
            # distorted model lands inside the band (observed 0.86x at s=2.0).
            distorted = _fp8_no_append_reference(env, *operands, kv_scale_factors=True)
            assert _band_fraction(out, distorted, FP8_MLA_ATOL, FP8_MLA_RTOL) < 1.5, (
                f"the s**2 / s model does not explain the s={scale} run"
            )


def test_fp8_mla_quant_mode_extra_bits_h128() -> None:
    """Quantization bits outside the KV-cache group ride along unread.

    A target derives quant_mode from its checkpoint's quant config and will not
    get a bare 128: DeepSeek-R1-0528-FP4 with an fp8 KV cache produces
    1152 (FP8_KV_CACHE | FP8_1x128_128x128), and a checkpoint with fp8 QDQ
    weights produces 384 (| FP8_QDQ). Both are bit-identical to 128 here — in
    the output *and* in the pool — across all three MLA call flavors, so the
    op reads only the KV-cache bit. Nothing checks the other bits either way;
    this test is what says they are inert rather than assumed to be.
    """
    lens = [96, 33]
    prefill = [64]
    new_lens, kv_lens = [32, 9], [64, 24]
    baseline: Dict[str, torch.Tensor] = {}
    for quant_mode in (QUANT_MODE_FP8_KV_CACHE, 1152, 384):
        env = _fp8_r1_env(quant_mode=quant_mode)
        torch.manual_seed(960)
        for rid, ln in enumerate(lens):
            env.add_request(rid, ln)
        q, k, v, latent = _random_context_inputs(sum(lens), env.num_heads)
        results = {
            "context": env.call_context([0, 1], lens, q, k, v, latent).clone(),
            "pool": env.pool.clone(),
        }

        gen_env = _fp8_r1_env(quant_mode=quant_mode)
        _fp8_mla_prefill(gen_env, prefill, 961)
        torch.manual_seed(962)
        gen_env.append_decode_latent(
            0, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
        )
        fused_q = (
            torch.randn(1, gen_env.num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda")
            * 0.3
        )
        results["generation"] = gen_env.call_generation([0], fused_q).clone()

        na_env = _fp8_r1_env(quant_mode=quant_mode)
        torch.manual_seed(963)
        for rid, total in enumerate(kv_lens):
            na_env.add_request(rid, new_lens[rid])
            na_env.reserve_cache_pages(rid, total)
        q_na = torch.randn(
            sum(new_lens),
            na_env.num_heads * QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        k_na, packed_na = _random_explicit_kv(sum(kv_lens), na_env.num_heads)
        results["no_append"] = na_env.call_context_no_append(
            [0, 1],
            new_lens,
            kv_lens,
            q_na,
            k_na,
            _v_split_view(packed_na, na_env.num_heads),
        ).clone()

        if quant_mode == QUANT_MODE_FP8_KV_CACHE:
            baseline = results
            continue
        for key, value in results.items():
            assert _bitwise_equal(value, baseline[key]), (
                f"quant_mode={quant_mode} changed the {key} result against a "
                f"bare {QUANT_MODE_FP8_KV_CACHE}"
            )


def test_fp8_mla_append_race_control_h128() -> None:
    """Positive control on the pool byte comparison every append check above
    rests on: it must be able to see a *racing* append.

    This entry documents one intermittent defect on its own write path — when
    a call's new tokens do not map to distinct physical slots, the writes race
    and leave torn cache rows, nondeterministically and with no error. That is
    the arming sequence replayed here on the fp8 latent pool: a 96-token
    prefill whose three absolute pages are all mapped onto one physical page,
    so three tokens contend for every slot. Four repeats of the identical
    seeded call must not all agree — measured on sm_100, every one of five
    repeats differed from the first, by 11 177 to 12 513 of the pool's bytes.

    Then the same comparison over the certified geometry (distinct pages, the
    R1-cell prefill), repeated in the same process, must be bitwise stable —
    which is what makes "the append is bit-exact" a result rather than the
    absence of a look.
    """
    lens = [96]
    armed: List[torch.Tensor] = []
    for _ in range(4):
        env = _fp8_r1_env()
        torch.manual_seed(964)
        env.add_request(0, lens[0])
        env.pages[0] = [2, 2, 2]  # every absolute page aliased onto one page
        q, k, v, latent = _random_context_inputs(sum(lens), env.num_heads)
        env.call_context([0], lens, q, k, v, latent)
        armed.append(env.pool.view(torch.uint8).clone())
    assert not all(_bitwise_equal(p, armed[0]) for p in armed[1:]), (
        "the aliased-page append came back identical in four repeats — this "
        "pool comparison cannot see a racing append at all"
    )

    stable: List[torch.Tensor] = []
    for _ in range(3):
        env = _fp8_r1_env()
        torch.manual_seed(965)
        env.add_request(0, lens[0])
        q, k, v, latent = _random_context_inputs(sum(lens), env.num_heads)
        env.call_context([0], lens, q, k, v, latent)
        env.check_cache(0)
        stable.append(env.pool.clone())
    for i, pool in enumerate(stable[1:], start=1):
        assert _bitwise_equal(pool, stable[0]), (
            f"the certified fp8 R1-cell append is not reproducible: repeat {i} "
            f"differs from repeat 0"
        )


# ---------------------------------------------------------------------------
# MLA generation at predicted_tokens_per_seq > 1
# ---------------------------------------------------------------------------

# The draft-chain lengths a DeepSeek-R1-0528 MTP target sweeps: max_draft_len
# 0/1/2/3, passed as predicted_tokens_per_seq = max_draft_len + 1. P = 1 is the
# regression check — it drives the code path the entry was already certified on.
MTP_SWEEP = [1, 2, 3, 4]

# Per-key score values for the mask readout below, and per-draft-row query
# values. All of them are exact in bf16 and in e4m3 (3 mantissa bits), so the
# logits the kernel forms are the ones the fp32 reference forms and the only
# inexactness left in the readout is the kernel's own e4m3 handling of the
# softmax probabilities.
MTP_KEY_SCORES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
MTP_Q_SCORES = [2.0, 4.0, 6.0, 8.0]

# Gate on the readout's in-set weights. Its output elements *are* the softmax
# probabilities, so against an exact fp32 softmax what is left is the kernel's
# own e4m3 handling of those probabilities before BMM2 — a relative error,
# hence a pure rtol with atol 0. Measured on sm_100 at H = 128, page 32 over
# P = 1..4, weights spanning 0.0061 to 0.0690: max relative deviation 5.5%
# against e4m3's 2**-4 = 6.25% single-rounding bound, so a gate at 2**-4 would
# sit at 88% of its allowance and be fragile, while 2**-3 (two e4m3 ulps) puts
# it at 44%. Over a bf16 pool the same readout deviates by at most 0.53%, so
# 2**-6 puts that at 34%.
#
# The entry's FP8_MLA_ATOL of 2**-3 must NOT be used here: every weight in this
# readout is smaller than that floor, so an assert_close carrying it is
# vacuous — measured, by replacing the expected weights with the rival
# full-mask model and watching the fp8 cases still pass.
#
# This gate is also deliberately not what separates the two candidate
# within-block masks: a full mask moves the in-set weights by at most 0.0042
# here against a 0.0024 residual (1.8x), which no tolerance resolves. The mask
# is separated by the *excluded* columns, gated bitwise zero.
MTP_READOUT_FP8_RTOL = 2**-3
MTP_READOUT_BF16_RTOL = 2**-6


def _mtp_readout_env(lens: List[int], offsets: List[int], fp8: bool = True) -> _MlaPagedEnv:
    """An MLA env whose cached latent rows turn the generation call's output
    into a direct readout of the attention weights it used.

    Cache row j of sequence g carries compressed_kv = one-hot at column
    offsets[g] + j and exactly one non-zero k_pe element. A query whose
    compressed-kv half is zero therefore scores only through k_pe, while the
    decode's V is K[:, :C] — the one-hot — so

        output[row, h, offsets[g] + j] = the weight `row` put on key j

    with an exact zero on every key the mask excluded. The sequences' key
    blocks are disjoint column ranges, so a row's non-zero columns also say
    which sequence the kernel assigned it to.
    """
    for ln, off in zip(lens, offsets):
        assert off + ln <= KV_LORA_RANK, "readout columns must fit in C"
    env = _MlaPagedEnv(
        num_heads=MLA_NUM_HEADS_H128,
        max_batch=max(4, len(lens)),
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
        tokens_per_block=MLA_PAGE32,
        q_lora_rank=Q_LORA_RANK_DSV3,
        pool_dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
        quant_mode=QUANT_MODE_FP8_KV_CACHE if fp8 else 0,
        kv_scaling_factor=1.0 if fp8 else None,
    )
    for g, ln in enumerate(lens):
        env.add_request(g, ln)
        for j in range(ln):
            row = torch.zeros(LATENT_DIM, dtype=torch.bfloat16, device="cuda")
            row[offsets[g] + j] = 1.0
            row[KV_LORA_RANK] = MTP_KEY_SCORES[j % len(MTP_KEY_SCORES)]
            env.append_decode_latent(g, row)
    return env


def _mtp_readout_q(rows: int, num_heads: int) -> torch.Tensor:
    """Query rows for the readout env: the compressed-kv half is zero, so the
    one-hot keys never enter a score, and one q_pe element carries a per-row
    value, so the P rows of a draft block do not share one softmax."""
    q = torch.zeros(rows, num_heads, LATENT_DIM, dtype=torch.bfloat16, device="cuda")
    for n in range(rows):
        q[n, :, KV_LORA_RANK] = MTP_Q_SCORES[n % len(MTP_Q_SCORES)]
    return q.view(rows, num_heads * LATENT_DIM)


def _mtp_expected_weights(env: _MlaPagedEnv, row: int, num_keys: int) -> torch.Tensor:
    """The exact fp32 softmax the readout env's query row puts on num_keys."""
    z = torch.tensor(
        [MTP_KEY_SCORES[j % len(MTP_KEY_SCORES)] for j in range(num_keys)],
        dtype=torch.float32,
        device="cuda",
    )
    return torch.softmax(MTP_Q_SCORES[row % len(MTP_Q_SCORES)] * z * env.softmax_scale, dim=0)


def _assert_mtp_causal_readout(
    env: _MlaPagedEnv,
    out: torch.Tensor,
    lens: List[int],
    offsets: List[int],
    p: int,
    label: str,
    rtol: float = MTP_READOUT_FP8_RTOL,
) -> None:
    """Read every draft row's attended key set out of `out`, all heads.

    Three assertions per row, and the rival within-block mask — row t sees its
    later siblings too — breaks all three:

    - every column outside [0, L - P + t] of the sequence's own key block is
      *bitwise zero*, in particular the columns of draft tokens t+1..P-1, which
      do not exist yet when row t is verified;
    - so is every column of the other sequences' key blocks, which is what pins
      the token-major row order;
    - the weights inside the set match the exact fp32 softmax, at the pure-rtol
      gate above.

    The gate on the excluded columns is exact zero rather than a tolerance, so
    the separation from the rival is absolute; the assert at the end of the
    loop keeps that meaningful by checking the rival would in fact have put
    non-trivial mass there.
    """
    heads = env.num_heads
    for n in range(len(lens) * p):
        g, t = n // p, n % p
        ln = lens[g]
        last = ln - p + t  # inclusive index of the newest key row n may see
        weights = out[n].view(heads, KV_LORA_RANK).float()
        expected = _mtp_expected_weights(env, n, last + 1)
        torch.testing.assert_close(
            weights[:, offsets[g] : offsets[g] + last + 1],
            expected.unsqueeze(0).expand(heads, -1),
            rtol=rtol,
            atol=0.0,
        )
        outside = torch.ones(KV_LORA_RANK, dtype=torch.bool, device="cuda")
        outside[offsets[g] : offsets[g] + last + 1] = False
        stray = int((weights[:, outside] != 0).sum())
        assert stray == 0, (
            f"{label}: row {n} (sequence {g}, draft token {t} of {p}) put "
            f"non-zero weight on {stray} key column(s) it must not see"
        )
        if t + 1 < p:
            rival = _mtp_expected_weights(env, n, ln)
            diverted = float(rival[last + 1 :].sum())
            assert diverted > 0.01, (
                f"{label}: row {n}'s future siblings would carry only "
                f"{diverted:.4g} of the mass under a full within-block mask, "
                f"so this readout could not separate the two masks here"
            )


def test_fp8_mla_mtp_mask_is_bottom_right_causal_h128() -> None:
    """What a draft row of an MLA generation call attends to at P > 1.

    This is the question the whole MTP surface turns on. At P = 1 there is no
    within-block mask to get wrong, and mask_type = 1 simply means "attend to
    everything cached". At P > 1 the P query rows of one sequence are its draft
    chain: row t sits at absolute position L - P + t and must attend to the
    cache up to and including its own position, and *not* to rows t+1..P-1 of
    its own block, which are tokens that do not exist yet. If the kernel let
    row t see row t+1, every verification past the first would be computed
    against a future token; nothing raises, and rejection sampling would still
    emit correct text with the drafts always rejected, so no downstream
    accuracy gate could see it either.

    On sm_100 no mask tensor does this job — a linear-tree draft has
    is_spec_decoding_enabled forced off there, so spec_decoding_packed_mask and
    its siblings are all None. Whatever masking happens comes from
    predicted_tokens_per_seq alone.

    Measured, at P = 1, 2, 3 and 4 over the fp8-e4m3 latent pool at the R1
    layer geometry, by reading the attention weights straight out of the
    output: the mask **is** causal and bottom-right aligned against the
    sequence's own KV length. Row t weights keys [0, L - P + t] and every other
    column comes back bitwise zero, including the 1 to 3 future-sibling columns
    that a full within-block mask would have weighted at 0.023-0.039 each.

    Two sequences of different lengths (33 and 50 cached rows, neither a
    multiple of the 32-slot page, so the causal cut crosses a page boundary)
    with disjoint readout column blocks, which also pins the row order: rows
    [g*P, (g+1)*P) belong to sequence g, token-major.
    """
    lens = [33, 50]
    offsets = [0, KV_LORA_RANK // 2]
    for p in MTP_SWEEP:
        torch.manual_seed(1100 + p)
        env = _mtp_readout_env(lens, offsets)
        fused_q = _mtp_readout_q(len(lens) * p, env.num_heads)
        pool_before = env.pool.clone()
        out = env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p)
        _assert_mtp_causal_readout(env, out, lens, offsets, p, f"fp8 P={p}")
        assert _bitwise_equal(pool_before, env.pool), (
            "the MLA generation call wrote to the fp8 pool"
        )


def test_fp8_mla_mtp_mask_type_and_page_alignment_h128() -> None:
    """Two follow-ups to the mask readout, on the same fp8 R1 geometry.

    1. mask_type = 0 (padding) does **not** differ from mask_type = 1 here at
       P > 1: both return the same bottom-right-causal attended sets, bitwise.
       The entry certifies mask_type = 0 for context flavors only, and the
       obvious guess — that padding would drop the within-block mask and let a
       draft row see its later siblings — is wrong. mask_type does not reach
       this path.
    2. The causal cut lands at L - P + t, so it walks across a 32-slot page
       boundary as L moves. Swept at P = 4 over L = 30..36, which puts the cut
       one slot before a boundary, exactly on it, and one slot after, and also
       covers L = P + small (a sequence whose whole cached history is barely
       longer than this step's own draft chain).
    """
    lens = [33, 50]
    offsets = [0, KV_LORA_RANK // 2]
    for p in (2, 4):
        torch.manual_seed(1200 + p)
        env = _mtp_readout_env(lens, offsets)
        fused_q = _mtp_readout_q(len(lens) * p, env.num_heads)
        causal = env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p)
        padding = env.call_generation(
            [0, 1], fused_q, predicted_tokens_per_seq=p, mask_type=MASK_PADDING
        )
        _assert_mtp_causal_readout(env, padding, lens, offsets, p, f"fp8 mask_type=0 P={p}")
        assert _bitwise_equal(causal, padding), (
            f"mask_type changed the MLA generation result at P={p} — the two "
            f"attended sets agree but not bitwise"
        )

    p = 4
    for ln in range(30, 37):
        torch.manual_seed(1300 + ln)
        env = _mtp_readout_env([ln], [0])
        fused_q = _mtp_readout_q(p, env.num_heads)
        out = env.call_generation([0], fused_q, predicted_tokens_per_seq=p)
        _assert_mtp_causal_readout(env, out, [ln], [0], p, f"fp8 page alignment L={ln} P={p}")


def test_fp8_mla_mtp_r1_cell_decode_h128() -> None:
    """Realistic-input MLA decode at the complete DeepSeek-R1-0528 cell over
    the fp8-e4m3 latent pool, swept over P = 1, 2, 3, 4.

    Random latent history and random fused q, two consecutive steps per P, two
    sequences whose KV lengths sit at different page-32 alignments. The gate is
    the fp32 latent-MQA reference under the measured bottom-right-causal mask,
    at the fp8 band.

    The rival full-mask model is gated *relatively* rather than absolutely: at
    these shapes it adds only 1-3 keys to a set of 30-70, and the fp8 band is
    wide enough that the rival sometimes stays inside it (measured 0.92-1.57 of
    the allowance over the seeds here, against 0.21-0.24 for the matching
    model), so "the rival fails assert_close" would not be a stable statement.
    What is stable is that the rival uses several times more of the allowance —
    4.4-6.9x measured, gated at 3x. The absolute mask evidence is the
    bitwise-zero readout above and the bf16-pool case below (17-23x outside its
    own band); what this case gates is the arithmetic of a production-shaped
    decode.
    """
    for p in MTP_SWEEP:
        torch.manual_seed(1400 + p)
        env = _fp8_r1_env()
        prefill = [64, 31]
        _fp8_mla_prefill(env, prefill, 1400 + p)
        for rid in (0, 1):
            env.check_cache(rid)
        for _ in range(2):
            for rid in (0, 1):
                for _ in range(p):
                    env.append_decode_latent(
                        rid,
                        torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5,
                    )
            pool_before = env.pool.clone()
            fused_q = (
                torch.randn(
                    2 * p,
                    env.num_heads * LATENT_DIM,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                * 0.3
            )
            out = env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p)
            ref = env.generation_reference([0, 1], fused_q, predicted_tokens_per_seq=p)
            torch.testing.assert_close(out, ref, rtol=FP8_MLA_RTOL, atol=FP8_MLA_ATOL)
            assert _bitwise_equal(pool_before, env.pool), (
                "the MLA generation call wrote to the fp8 pool"
            )
            if p > 1:
                matched = _band_fraction(out, ref, FP8_MLA_ATOL, FP8_MLA_RTOL)
                rival = _band_fraction(
                    out,
                    env.generation_reference(
                        [0, 1], fused_q, predicted_tokens_per_seq=p, full_mask=True
                    ),
                    FP8_MLA_ATOL,
                    FP8_MLA_RTOL,
                )
                assert rival > 3.0 * matched, (
                    f"fp8 R1-cell MTP decode at P={p}: the full-mask rival uses "
                    f"{rival:.3g} of the allowance against the matching model's "
                    f"{matched:.3g} — only {rival / matched:.3g}x apart"
                )


def test_fp8_mla_mtp_batch_state_h128() -> None:
    """Which batch-state tensors the MLA generation call reads at P > 1.

    The entry records, at P = 1, that cu_q_seqlens is required by presence but
    bitwise inert in its contents, that cu_kv_seqlens is inert outright, and
    that the per-sequence KV length comes from sequence_length. A taller query
    block is exactly the change that could have made the scheduler buffers
    start mattering, so all of it is re-measured at P = 3.

    Measured: cu_q_seqlens contents stay inert — zeroed, left in the P = 1
    i*H form, given i*P without the head factor, inflated 1000x and made
    non-monotonic all return bitwise-identical output. So do cu_kv_seqlens and
    both context_lengths copies. sequence_length remains the KV extent: a
    one-token change moves every draft row's attended set by one, read out
    one-hot. host_past_key_value_lengths is inert as long as it is non-zero
    (sequence_length - 1, - P and all-ones are all bitwise identical), but an
    all-zero vector makes the call skip its work entirely and return without
    writing a single element of `output` — not a P > 1 effect, it reproduces
    at P = 1.

    Also gated: the call writes exactly the first G*P rows of `output` and
    leaves a taller buffer's tail untouched.
    """
    lens = [33, 50]
    p = 3
    offsets = [0, KV_LORA_RANK // 2]
    torch.manual_seed(1500)
    env = _fp8_r1_env()
    for rid, ln in enumerate(lens):
        env.add_request(rid, ln - p)
        for _ in range(ln):
            env.append_decode_latent(
                rid, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
            )
    fused_q = (
        torch.randn(
            len(lens) * p,
            env.num_heads * LATENT_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.3
    )
    base = env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p)
    assert _bitwise_equal(base, env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p)), (
        "the MLA generation call is not run-to-run deterministic here"
    )

    h, g = env.num_heads, len(lens)
    for label, cu_q in (
        ("zeroed", torch.zeros(g + 1, dtype=torch.int32)),
        ("the P=1 form i*H", torch.arange(g + 1, dtype=torch.int32) * h),
        ("i*P without the head factor", torch.arange(g + 1, dtype=torch.int32) * p),
        ("1000x", torch.arange(g + 1, dtype=torch.int32) * (h * p) * 1000),
        ("non-monotonic", torch.tensor([0, 7 * h * p, 2 * h * p], dtype=torch.int32)),
    ):
        assert _bitwise_equal(
            base,
            env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p, cu_q_override=cu_q),
        ), f"cu_q_seqlens {label} moved the output at P={p}"

    for label, cu_kv in (
        ("zeroed", torch.zeros(g + 1, dtype=torch.int32)),
        ("1000x", torch.tensor([0, 33000, 83000], dtype=torch.int32)),
        ("non-monotonic", torch.tensor([0, 90, 5], dtype=torch.int32)),
    ):
        assert _bitwise_equal(
            base,
            env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p, cu_kv_override=cu_kv),
        ), f"cu_kv_seqlens {label} moved the output at P={p}"

    for label, host_past in (
        ("sequence_length - 1", [lens[0] - 1, lens[1] - 1]),
        ("sequence_length - P", [lens[0] - p, lens[1] - p]),
        ("all ones", [1, 1]),
    ):
        assert _bitwise_equal(
            base,
            env.call_generation(
                [0, 1],
                fused_q,
                predicted_tokens_per_seq=p,
                host_past_override=host_past,
            ),
        ), f"host_past_key_value_lengths = {label} moved the output at P={p}"

    for label, ctx_lens in (("zeroed", [0, 0]), ("the full KV length", list(lens))):
        assert _bitwise_equal(
            base,
            env.call_generation(
                [0, 1], fused_q, predicted_tokens_per_seq=p, ctx_lens_override=ctx_lens
            ),
        ), f"context_lengths {label} moved the output at P={p}"

    # An all-zero host_past_key_value_lengths is the one non-inert perturbation
    # found, and what it does is skip the work entirely: the call returns
    # without writing a single element of `output`, whatever that buffer held.
    # Stated against a sentinel fill rather than against zero, so it is a claim
    # about "not written" and not about a value that happens to be zero.
    sentinel_fill = torch.full(
        (len(lens) * p, env.num_heads * KV_LORA_RANK),
        7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    untouched = sentinel_fill.clone()
    env.call_generation(
        [0, 1],
        fused_q,
        predicted_tokens_per_seq=p,
        host_past_override=[0, 0],
        output_buffer=untouched,
    )
    assert _bitwise_equal(untouched, sentinel_fill), (
        "an all-zero host_past_key_value_lengths no longer skips the call — "
        "this control has stopped measuring what it claims"
    )
    written = sentinel_fill.clone()
    env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p, output_buffer=written)
    assert _bitwise_equal(written, base), (
        "the same call with a true host_past_key_value_lengths did not write the certified result"
    )

    # sequence_length is the KV extent: shift it down and every row's attended
    # set shifts with it, keeping the L - P + t shape.
    for delta in (0, -1, -2):
        shifted = _mtp_readout_env([lens[0] + delta, lens[1] + delta], offsets)
        out = shifted.call_generation(
            [0, 1],
            _mtp_readout_q(len(lens) * p, shifted.num_heads),
            predicted_tokens_per_seq=p,
        )
        _assert_mtp_causal_readout(
            shifted,
            out,
            [lens[0] + delta, lens[1] + delta],
            offsets,
            p,
            f"sequence_length shifted by {delta}",
        )

    tall = torch.full(
        (len(lens) * p + 4, env.num_heads * KV_LORA_RANK),
        7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    sentinel = tall[len(lens) * p :].clone()
    env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p, output_buffer=tall)
    assert _bitwise_equal(tall[len(lens) * p :], sentinel), (
        "the MLA generation call wrote past row G*P of `output`"
    )
    assert _bitwise_equal(tall[: len(lens) * p], base), (
        "writing into a taller output buffer changed the result"
    )


def test_fp8_mla_mtp_mixed_batch_h128() -> None:
    """A mixed batch whose generation sequences each carry P draft tokens.

    One leading context sequence plus one generation sequence, two phase calls
    sharing the full-batch state tensors and one page-32 offsets table, at the
    complete R1 cell over the fp8 pool. The context call's token accounting is
    unchanged by P — it owns num_ctx_tokens rows — while the generation call
    owns G*P rows indexed from num_contexts, and the two must still agree.
    Swept over P = 2, 3, 4.
    """
    for p in (2, 3, 4):
        torch.manual_seed(1600 + p)
        env = _fp8_r1_env()
        first_len, second_len = 40, 33
        env.add_request(0, first_len)
        q, k, v, latent = _random_context_inputs(first_len, env.num_heads)
        env.call_context([0], [first_len], q, k, v, latent)

        env.add_request(1, second_len)
        for _ in range(p):
            env.append_decode_latent(
                0, torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
            )
        q, k, v, latent = _random_context_inputs(second_len, env.num_heads)
        pre = (q.clone(), k.clone(), v, latent.clone())
        out_ctx = env.call_context([1], [second_len], q, k, v, latent, gen_request_ids=[0])
        torch.testing.assert_close(
            out_ctx,
            env.context_reference([second_len], *pre, e4m3_inputs=True),
            rtol=FP8_MLA_RTOL,
            atol=FP8_MLA_ATOL,
        )

        fused_q = (
            torch.randn(p, env.num_heads * LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
        )
        out_gen = env.call_generation(
            [0],
            fused_q,
            ctx_request_ids=[1],
            ctx_seq_lens=[second_len],
            predicted_tokens_per_seq=p,
        )
        torch.testing.assert_close(
            out_gen,
            env.generation_reference([0], fused_q, predicted_tokens_per_seq=p),
            rtol=FP8_MLA_RTOL,
            atol=FP8_MLA_ATOL,
        )
        env.check_cache(0)
        env.check_cache(1)
        env.check_unwritten_pool_zero([0, 1])


def test_bf16_mla_mtp_decode_h128() -> None:
    """The same MTP generation surface over a bf16 latent pool.

    Two things this adds over the fp8 cases. The bf16 band is 8x tighter, so
    the full-mask rival separates properly here — observed 17-23x outside,
    against 0.22 for the matching model, which is what makes "the within-block
    mask is causal" a gated result on realistic inputs and not only a readout.
    And it pins that P > 1 is a property of the MLA decode kernel rather than
    of the fp8 decode path, which is the only one the R1 target runs.
    """
    lens = [33, 50]
    offsets = [0, KV_LORA_RANK // 2]
    for p in (2, 4):
        torch.manual_seed(1700 + p)
        env = _mtp_readout_env(lens, offsets, fp8=False)
        fused_q = _mtp_readout_q(len(lens) * p, env.num_heads)
        out = env.call_generation([0, 1], fused_q, predicted_tokens_per_seq=p)
        _assert_mtp_causal_readout(
            env, out, lens, offsets, p, f"bf16 P={p}", rtol=MTP_READOUT_BF16_RTOL
        )

        torch.manual_seed(1750 + p)
        real = _MlaPagedEnv(
            num_heads=MLA_NUM_HEADS_H128,
            max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
            tokens_per_block=MLA_PAGE32,
            q_lora_rank=Q_LORA_RANK_DSV3,
        )
        for rid, ln in enumerate(lens):
            real.add_request(rid, ln - p)
            for _ in range(ln):
                real.append_decode_latent(
                    rid,
                    torch.randn(LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.5,
                )
        fq = (
            torch.randn(
                len(lens) * p,
                real.num_heads * LATENT_DIM,
                dtype=torch.bfloat16,
                device="cuda",
            )
            * 0.3
        )
        pool_before = real.pool.clone()
        got = real.call_generation([0, 1], fq, predicted_tokens_per_seq=p)
        torch.testing.assert_close(
            got,
            real.generation_reference([0, 1], fq, predicted_tokens_per_seq=p),
            rtol=RTOL,
            atol=ATOL,
        )
        _assert_far_outside_band(
            got,
            real.generation_reference([0, 1], fq, predicted_tokens_per_seq=p, full_mask=True),
            5.0,
            f"bf16 MTP decode at P={p} against a full-mask rival",
            ATOL,
            RTOL,
        )
        assert _bitwise_equal(pool_before, real.pool), (
            "the MLA generation call wrote to the bf16 pool"
        )


def test_fp8_mla_mtp_decode_sees_a_torn_pool_h128() -> None:
    """Positive control on the decode-side comparison every claim above rests
    on: it must be able to see a pool whose content moved under it.

    Every MTP result here reads a cache the caller wrote, and this entry
    documents one intermittent defect on that write path — a call whose new
    tokens do not map to distinct physical slots races and leaves torn rows,
    nondeterministically and with no error. At P > 1 the caller's
    generation-phase preprocessing writes P rows per sequence per step instead
    of one, so that arming sequence is newly reachable from a decode step.

    Armed here the way the append control arms it: a 96-token prefill whose
    three absolute pages are mapped onto one physical page. Four repeats must
    not all leave the same pool — and, the part this control adds, the P = 3
    decode run over those differing pools must not all return the same output,
    which is what makes a clean decode result a measurement rather than the
    absence of a look. Then the certified geometry, in the same process:
    distinct pages, identical seeds, pool and decode output both bitwise
    stable across repeats.
    """
    p = 3
    lens = [96]
    torch.manual_seed(1800)
    probe_q = (
        torch.randn(p, MLA_NUM_HEADS_H128 * LATENT_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
    )

    armed_pools: List[torch.Tensor] = []
    armed_outs: List[torch.Tensor] = []
    for _ in range(4):
        env = _fp8_r1_env()
        torch.manual_seed(1801)
        env.add_request(0, lens[0])
        env.pages[0] = [2, 2, 2]  # every absolute page aliased onto one page
        q, k, v, latent = _random_context_inputs(sum(lens), env.num_heads)
        env.call_context([0], lens, q, k, v, latent)
        armed_pools.append(env.pool.view(torch.uint8).clone())
        armed_outs.append(env.call_generation([0], probe_q, predicted_tokens_per_seq=p).clone())
    assert not all(_bitwise_equal(x, armed_pools[0]) for x in armed_pools[1:]), (
        "the aliased-page append came back identical in four repeats — this "
        "control cannot arm the defect it is meant to arm"
    )
    assert not all(_bitwise_equal(x, armed_outs[0]) for x in armed_outs[1:]), (
        "the P>1 decode returned the same output over four demonstrably "
        "different pools — the decode-side comparison is blind to pool content"
    )

    stable_pools: List[torch.Tensor] = []
    stable_outs: List[torch.Tensor] = []
    for _ in range(3):
        env = _fp8_r1_env()
        torch.manual_seed(1802)
        env.add_request(0, lens[0])
        q, k, v, latent = _random_context_inputs(sum(lens), env.num_heads)
        env.call_context([0], lens, q, k, v, latent)
        env.check_cache(0)
        stable_pools.append(env.pool.clone())
        stable_outs.append(env.call_generation([0], probe_q, predicted_tokens_per_seq=p).clone())
    for i in range(1, 3):
        assert _bitwise_equal(stable_pools[i], stable_pools[0]), (
            f"the certified fp8 R1-cell append is not reproducible: repeat {i}"
        )
        assert _bitwise_equal(stable_outs[i], stable_outs[0]), (
            f"the certified P>1 decode is not reproducible: repeat {i}"
        )


def test_mla_rejects_null_q_lora_rank() -> None:
    """q_lora_rank must be an int on the MLA path: the C++ unwraps the
    optional unconditionally, so None raises rather than defaulting."""
    torch.manual_seed(309)
    h = MLA_NUM_HEADS_H32
    q, k, v, latent = _random_context_inputs(32, h)
    env = _MlaPagedEnv(
        num_heads=h,
        max_blocks_per_seq=MLA_MAX_SEQ_LEN // MLA_PAGE32,
        tokens_per_block=MLA_PAGE32,
        q_lora_rank=None,
    )
    env.add_request(0, 32)
    try:
        env.call_context([0], [32], q, k, v, latent)
    except RuntimeError as exc:
        assert "bad optional access" in str(exc), f"unexpected message: {exc}"
    else:
        raise AssertionError("q_lora_rank=None was accepted on the MLA path")
