# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused attention core with fully explicit state: paged KV-cache append + masked FMHA.

Wraps ``FallbackFmha.attention``, which dispatches to the native
``AttentionOp`` context and generation methods. Every piece of batch state
and layer config arrives as an explicit argument; native runners are cached
by static configuration, device, and thread.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

from .._op import Arch, Cell, OpWrapper

#: attention_input_type. 0 is the standard append-and-attend path; 1 and 2 are
#: MLA's context and generation phases.
STANDARD, MLA_CONTEXT, MLA_GENERATION = 0, 1, 2

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128

#: qwen4_exp's full-attention layer -- the twelve trunk layers and the MTP
#: module alike: packed-QKV causal GQA over a bf16 pool, rope applied outside,
#: the paged-context path on (the target serves with block reuse and chunked
#: prefill). Pool sizes are chosen, not dumped: 128 pages of 32 tokens and 4
#: block-offset rows, larger than any cell's batch.
_QWEN4_DENSE = dict(
    input_type=STANDARD,
    num_heads=24,
    num_kv_heads=2,
    head_size=256,
    v_head_dim=256,
    sinks=False,
    window=None,
    quant_mode=0,
    mask_type=1,
    tokens_per_block=32,
    num_blocks=128,
    max_num_requests=4,
    use_paged_context_fmha=True,
)

#: qwen4_exp's vision tower: the no-KV-cache mode, one context sequence per
#: frame, MHA 16/16 at head_size 72 (hidden 1152 over 16 heads), padding mask.
#: `max_num_requests` is the segment count, as the target passes it.
_QWEN4_VISION = dict(
    input_type=STANDARD,
    kv_cache="none",
    num_heads=16,
    num_kv_heads=16,
    head_size=72,
    v_head_dim=72,
    sinks=False,
    window=None,
    quant_mode=0,
    mask_type=0,
)

#: A GQA 24/4/256 full-attention layer (Qwen3.8-27B) over an fp8-e4m3 pool at
#: the production kv scale 1.0. Same chosen pool sizes as `_QWEN4_DENSE`.
_FP8_24_4_256 = dict(
    input_type=STANDARD,
    num_heads=24,
    num_kv_heads=4,
    head_size=256,
    v_head_dim=256,
    sinks=False,
    window=None,
    quant_mode=QM_FP8_KV_CACHE,
    kv_scale=1.0,
    mask_type=1,
    tokens_per_block=32,
    num_blocks=128,
    max_num_requests=4,
)


class ThopAttention(OpWrapper):
    """Append this step's K/V to the paged cache, then attend over it.

    The shipped targets use the native phased ``AttentionOp`` through
    ``FallbackFmha.attention``: gpt-oss uses attention sinks and an alternating
    sliding window, deepseek uses MLA's two phases, and qwen4_exp uses packed
    GQA for its full-attention and MTP layers and uncached MHA for vision.

    Writes rows `[:num_tokens]` of `output` and returns `output` -- the same
    buffer, mutated and handed back so a call site can show what it touched
    without allocating.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # gpt-oss alternates its two window classes inside one engine, so both of
    # its cells are live every step; deepseek runs context and generation
    # against the same pool. qwen4_exp's cells give `cached` (tokens already in
    # the pool) and `new` (this call's q rows) per sequence, contexts first.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="gpt-oss sliding layer: GQA 64q/8kv d64, sinks, window 128",
            spec=dict(
                input_type=STANDARD,
                num_heads=64,
                num_kv_heads=8,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=128,
                quant_mode=0,
            ),
        ),
        Cell(
            why="gpt-oss full layer: the same heads with no window",
            spec=dict(
                input_type=STANDARD,
                num_heads=64,
                num_kv_heads=8,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=None,
                quant_mode=0,
            ),
        ),
        # The per-rank head counts the gpt-oss default target runs under
        # attention TP: the same code path as the two cells above, at the
        # head counts a rank holds after the split.
        Cell(
            why="gpt-oss default at attention TP8: 8q/1kv per rank, 128-token window",
            spec=dict(
                input_type=STANDARD,
                num_heads=8,
                num_kv_heads=1,
                head_size=64,
                v_head_dim=64,
                sinks=True,
                window=128,
                quant_mode=0,
            ),
        ),
        Cell(
            why="deepseek MLA context: explicit k/v, qk 576 -> v 128, fp8 pool",
            spec=dict(
                input_type=MLA_CONTEXT,
                num_heads=128,
                num_kv_heads=128,
                head_size=576,
                v_head_dim=128,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="deepseek MLA generation: one kv head, latent 576 -> 512, fp8 pool",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=128,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        # The per-rank head counts the deepseek default target runs under
        # attention TP: the same two MLA code paths as the cells above, at the
        # 128 heads split 2, 4 or 8 ways.
        Cell(
            why="deepseek default at attention TP8: MLA context, 16 heads per rank",
            spec=dict(
                input_type=MLA_CONTEXT,
                num_heads=16,
                num_kv_heads=16,
                head_size=576,
                v_head_dim=128,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="deepseek default at attention TP2: MLA generation, 64 heads per rank",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=64,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        Cell(
            why="deepseek default at attention TP8: MLA generation, 16 heads per rank",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=16,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
            ),
        ),
        # MTP3 under attention TP: the verify step's taller query block (draft
        # length 3 + 1 rows per sequence) at a split head count. The P > 1 mask
        # itself is pinned at 128 heads by the MTP cases in the test.
        Cell(
            why="deepseek default MTP3 at attention TP8: MLA generation, 16 heads, P=4",
            spec=dict(
                input_type=MLA_GENERATION,
                num_heads=16,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                sinks=False,
                window=None,
                quant_mode=QM_FP8_KV_CACHE,
                predicted_tokens_per_seq=4,
            ),
        ),
        Cell(
            why="qwen4_exp prefill: GQA 24q/2kv d256, bf16 pool, paged context; "
            "a fresh 70-row sequence beside 33 rows over a 48-token cached prefix",
            spec=dict(
                _QWEN4_DENSE,
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
                _QWEN4_DENSE,
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
                _QWEN4_DENSE,
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
                _QWEN4_DENSE,
                cached=(70, 35),
                new=(3, 3),
                num_contexts=0,
                predicted_tokens_per_seq=3,
            ),
        ),
        Cell(
            why="qwen4_exp MTP verify P=4 (max_draft_len 3): the target's ceiling",
            spec=dict(
                _QWEN4_DENSE,
                cached=(70, 35),
                new=(4, 4),
                num_contexts=0,
                predicted_tokens_per_seq=4,
            ),
        ),
        Cell(
            why="qwen4_exp mixed step: a fresh 20-row context ahead of two P=2 "
            "draft blocks, generation rows token-major after the context rows",
            spec=dict(
                _QWEN4_DENSE,
                cached=(0, 70, 35),
                new=(20, 2, 2),
                num_contexts=1,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="qwen4_exp hybrid pool: row 47 of the 48-row two-pool mapping "
            "(pool 0, layer-in-pool 11), P=2 verify",
            spec=dict(
                _QWEN4_DENSE,
                pool_layout="hybrid48",
                local_layer_idx=47,
                cached=(64, 25),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="48-layer single pool, identity mapping: row 47 = layer-in-pool 47, "
            "24 rows over a 40-token cached prefix",
            spec=dict(
                _QWEN4_DENSE,
                pool_layout="identity48",
                local_layer_idx=47,
                cached=(40,),
                new=(24,),
                num_contexts=1,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="qwen4_exp vision tower: no KV cache, MHA 16/16 d72, padding mask, "
            "frames of 27/9/41 rows each attending only to itself",
            spec=dict(_QWEN4_VISION, seg_lens=(27, 9, 41)),
        ),
        Cell(
            why="encoder-only, d72 causal: measured, not what the tower passes",
            spec=dict(_QWEN4_VISION, mask_type=1, seg_lens=(27, 9, 41)),
        ),
        Cell(
            why="encoder-only, d128 padding: the tower's other admitted width",
            spec=dict(_QWEN4_VISION, head_size=128, v_head_dim=128, seg_lens=(27, 9, 41)),
        ),
        Cell(
            why="encoder-only, d128 causal",
            spec=dict(
                _QWEN4_VISION, head_size=128, v_head_dim=128, mask_type=1, seg_lens=(27, 9, 41)
            ),
        ),
        Cell(
            why="encoder-only, a single one-token frame",
            spec=dict(_QWEN4_VISION, seg_lens=(1,)),
        ),
        Cell(
            why="mmmu serving shape: one 5580-row frame, the 62x90 patch grid",
            spec=dict(_QWEN4_VISION, seg_lens=(5580,)),
        ),
        Cell(
            why="mmmu serving shape: one 8160-row frame, the largest single-segment call",
            spec=dict(_QWEN4_VISION, seg_lens=(8160,)),
        ),
        Cell(
            why="mmmu serving shape: 2204 rows in four frames",
            spec=dict(_QWEN4_VISION, seg_lens=(728, 728, 460, 288)),
        ),
        Cell(
            why="mmmu serving shape: 1892 rows in five frames, the most segments per call",
            spec=dict(_QWEN4_VISION, seg_lens=(480, 480, 400, 352, 180)),
        ),
        Cell(
            why="mmmu serving shape: 9008 rows in two frames",
            spec=dict(_QWEN4_VISION, seg_lens=(8000, 1008)),
        ),
        Cell(
            why="mmmu serving shape: 22144 rows in two frames, the largest call; its "
            "12544-row frame is the longest segment",
            spec=dict(_QWEN4_VISION, seg_lens=(12544, 9600)),
        ),
        Cell(
            why="fp8 pool packed prefill, GQA 24q/4kv d256: the FMHA reads the "
            "bf16 q rows, not the pool",
            spec=dict(
                _FP8_24_4_256,
                use_paged_context_fmha=False,
                cached=(0, 0),
                new=(48, 17),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="fp8 pool padding-mask prefill, 24/4/256, context-only batch",
            spec=dict(
                _FP8_24_4_256,
                mask_type=0,
                use_paged_context_fmha=False,
                cached=(0, 0),
                new=(31, 9),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="fp8 pool decode, 24/4/256: q, K, V and the softmax probabilities "
            "are e4m3 operands; histories 200 and 48 span pages and KV tiles",
            spec=dict(
                _FP8_24_4_256,
                use_paged_context_fmha=False,
                cached=(200, 48),
                new=(1, 1),
                num_contexts=0,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="fp8 pool paged-context prefill, 24/4/256: the FMHA reads the "
            "pool, so its operands are e4m3",
            spec=dict(
                _FP8_24_4_256,
                use_paged_context_fmha=True,
                cached=(0, 48),
                new=(48, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="fp8 pool paged-context prefill at GQA 32q/8kv d128",
            spec=dict(
                _FP8_24_4_256,
                num_heads=32,
                num_kv_heads=8,
                head_size=128,
                v_head_dim=128,
                use_paged_context_fmha=True,
                cached=(0, 48),
                new=(48, 33),
                num_contexts=2,
                predicted_tokens_per_seq=1,
            ),
        ),
        Cell(
            why="fp8 pool MTP verify P=2, 24/4/256",
            spec=dict(
                _FP8_24_4_256,
                use_paged_context_fmha=True,
                cached=(70, 35),
                new=(2, 2),
                num_contexts=0,
                predicted_tokens_per_seq=2,
            ),
        ),
        Cell(
            why="fp8 pool MTP verify P=3, 24/4/256",
            spec=dict(
                _FP8_24_4_256,
                use_paged_context_fmha=True,
                cached=(70, 35),
                new=(3, 3),
                num_contexts=0,
                predicted_tokens_per_seq=3,
            ),
        ),
        Cell(
            why="fp8 pool MTP verify P=4, 24/4/256",
            spec=dict(
                _FP8_24_4_256,
                use_paged_context_fmha=True,
                cached=(70, 35),
                new=(4, 4),
                num_contexts=0,
                predicted_tokens_per_seq=4,
            ),
        ),
        Cell(
            why="fp8 pool mixed step, 24/4/256: a 20-row context ahead of two P=3 "
            "blocks over histories 31 and 64",
            spec=dict(
                _FP8_24_4_256,
                use_paged_context_fmha=True,
                cached=(0, 31, 64),
                new=(20, 3, 3),
                num_contexts=1,
                predicted_tokens_per_seq=3,
            ),
        ),
    )

    note = """
    KV addressing over one pool and two is certified. gpt-oss alternates
    sliding-window and full attention, and KVCacheManagerV2 gives each window
    class its own layer group, so its mapping carries two pool ids; the op reads
    the pool column itself and this entry passes the mapping through untouched.
    A third pool is outside what was measured. qwen4_exp's hybrid tower is the
    other two-pool shape: a 48-row mapping whose full-attention rows 3, 7, ...,
    47 sit on pool 0 at layer-in-pool 0..11 and whose 36 recurrent rows name
    pool 1. `local_layer_idx` is the mapping *row* -- the layer's
    `layer_offsets` entry -- and the pool-base shift comes from the row's
    layer column; an index computed by counting attention layers addresses a
    sibling's slabs and returns plausible output.

    `reference` covers the masked-softmax half. The append half -- that this
    step's K/V land in the right pool slots -- is what the paged-cache entries
    beside this one certify, and what the matrix here checks by reading the pool
    back rather than by a reference.

    The softmax statistics the MLA chunked-prefill path writes are a second
    output; `softmax_stats` states them separately so `reference` stays one
    thing.

    predicted_tokens_per_seq (P) is the q-row count of every generation
    sequence, token-major within it; the append writes all P and
    `sequence_length` must count them. P alone makes the within-block mask
    bottom-right causal: every `spec_decoding_*` tensor stays None and
    `is_spec_decoding_enabled` False, which is what sm_103 passes for a linear
    draft, and mask_type 0 and 1 are bitwise identical at P > 1. Nothing checks
    q's row count against P. P reaches the decode kernel as `maxSeqLenQ`: one
    JIT compile per P.

    No KV cache (an encoder): pass `kv_cache_block_offsets`,
    `host_kv_cache_pool_pointers` and `host_kv_cache_pool_mapping` all None,
    with `update_kv_cache=True` -- False is rejected ("KV cache update cannot
    be disabled now"). One context sequence per frame; mask_type 0 confines a
    frame to itself. `tokens_per_block` and `use_paged_context_fmha` are
    bitwise inert there. Passing only some of the three None is unmeasured.
    head_size 72 exists in the context phase only: a generation call raises
    "unsupported head_size 72", and the process stays usable.

    Over an fp8-e4m3 pool the packed context path reads the bf16 q rows and
    never the pool, while `use_paged_context_fmha=True` reads the pool, so its
    operands are e4m3: the flag is output-inert over a bf16 pool only. Every
    fp8 cell runs kv scale 1.0; at 1.5 `reference` does not reproduce the
    kernel's probability rounding.
    quant_mode 384, 1152, 4224 and 4480 are bit-identical to 128 at 24/4/256;
    4352 (the same bits without FP8_KV_CACHE) reads e4m3 slabs as bf16.

    The sparse / skip-softmax arguments are bitwise inert in the standard
    configuration (nine off-default settings at 24/2/256): passing them does
    not buy sparse attention.

    The kv head count moves the compiled decode kernel at fixed Hq and D
    (24/2/256 runs a VarSeqQ16 variant, 24/4/256 VarSeqQ8), so a geometry that
    differs from a cell only in `num_kv_heads` is not covered by it.

    Generation kernels JIT through NVRTC and find their headers with `pip show
    tensorrt_llm`: a `pip` on PATH that cannot see the install fails every
    generation call, and the first failure poisons the process ("Jitify fatal
    error: Uninitialized").
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        output_sf: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_total_kv_lens: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        host_request_types: torch.Tensor,
        max_context_q_len_override: Optional[int],
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        cache_indirection: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        rotary_inv_freq: Optional[torch.Tensor],
        rotary_cos_sin: Optional[torch.Tensor],
        latent_cache: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        attention_sinks: Optional[torch.Tensor],
        is_fused_qkv: bool,
        update_kv_cache: bool,
        predicted_tokens_per_seq: int,
        local_layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: Optional[int],
        max_num_requests: int,
        max_context_length: int,
        max_seq_len: int,
        attention_window_size: int,
        beam_width: int,
        mask_type: int,
        quant_mode: int,
        q_scaling: float,
        position_embedding_type: int,
        rope_dim: int,
        rope_base: float,
        rope_scale_type: int,
        rope_scale: float,
        rope_short_m_scale: float,
        rope_long_m_scale: float,
        rope_max_positions: int,
        rope_original_max_positions: int,
        use_paged_context_fmha: bool,
        attention_input_type: Optional[int],
        is_mla_enable: bool,
        chunked_prefill_buffer_batch_size: Optional[int],
        q_lora_rank: Optional[int],
        kv_lora_rank: Optional[int],
        qk_nope_head_dim: Optional[int],
        qk_rope_head_dim: Optional[int],
        v_head_dim: Optional[int],
        rope_append: Optional[bool],
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        helix_position_offsets: Optional[torch.Tensor],
        helix_is_inactive_rank: Optional[torch.Tensor],
        attention_chunk_size: Optional[int],
        softmax_stats_tensor: Optional[torch.Tensor],
        is_spec_decoding_enabled: bool,
        use_spec_decoding: bool,
        is_spec_dec_tree: bool,
        spec_decoding_generation_lengths: Optional[torch.Tensor],
        spec_decoding_position_offsets: Optional[torch.Tensor],
        spec_decoding_packed_mask: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask: Optional[torch.Tensor],
        spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor],
        sparse_kv_indices: Optional[torch.Tensor],
        sparse_kv_offsets: Optional[torch.Tensor],
        sparse_attn_indices: Optional[torch.Tensor],
        sparse_attn_offsets: Optional[torch.Tensor],
        sparse_attn_indices_block_size: int,
        num_sparse_topk: Optional[int] = None,
        sparse_attn_kv_lens: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor_prefill: Optional[float] = None,
        skip_softmax_threshold_scale_factor_decode: Optional[float] = None,
        skip_softmax_stat: Optional[torch.Tensor] = None,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None,
        flash_mla_num_splits: Optional[torch.Tensor] = None,
        sage_attn_num_elts_per_blk_q: int = 0,
        sage_attn_num_elts_per_blk_k: int = 0,
        sage_attn_num_elts_per_blk_v: int = 0,
        sage_attn_qk_int8: bool = False,
        num_contexts: int = 0,
        num_ctx_tokens: int = 0,
        trtllm_gen_jit_warmup: bool = False,
        aux_kv_cache_pool_ptr: Optional[int] = None,
        is_cross: bool = False,
        cross_kv: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        spec_decoding_target_max_draft_tokens: Optional[int] = None,
        quant_scale_qkv: Optional[torch.Tensor] = None,
        dsv4_inv_rope_cos_sin_cache: Optional[torch.Tensor] = None,
        enable_dsv4_epilogue_fusion: bool = False,
        # Defaults match what the in-tree caller
        # (attention/backends/fmha/fallback.py) passes on a dense, non-sparse,
        # non-folded path -- which is the path both migrated targets are on.
        max_num_sequences: Optional[int] = None,
        force_prepare_spec_dec_tree_mask: bool = False,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        skip_correction_threshold: float = 0.0,
    ) -> torch.Tensor:
        """Writes `output` in place and returns it -- the op destroys its
        prior contents in rows `[:num_tokens]`, so a caller that wants them
        has to have kept its own copy beforehand."""
        if is_mla_enable and q_lora_rank is None:
            raise ValueError("q_lora_rank must be an int for MLA; use 0 when there is no q-LoRA.")
        FallbackFmha.attention(
            q=q,
            k=k,
            v=v,
            output=output,
            output_sf=output_sf,
            workspace=workspace,
            sequence_length=sequence_length,
            host_past_key_value_lengths=host_past_key_value_lengths,
            host_total_kv_lens=host_total_kv_lens,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            host_request_types=host_request_types,
            max_context_q_len_override=max_context_q_len_override,
            kv_cache_block_offsets=kv_cache_block_offsets,
            host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            cache_indirection=cache_indirection,
            kv_scale_orig_quant=kv_scale_orig_quant,
            kv_scale_quant_orig=kv_scale_quant_orig,
            out_scale=out_scale,
            rotary_inv_freq=rotary_inv_freq,
            rotary_cos_sin=rotary_cos_sin,
            latent_cache=latent_cache,
            q_pe=q_pe,
            block_ids_per_seq=block_ids_per_seq,
            attention_sinks=attention_sinks,
            is_fused_qkv=is_fused_qkv,
            update_kv_cache=update_kv_cache,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            local_layer_idx=local_layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            tokens_per_block=tokens_per_block,
            max_num_requests=max_num_requests,
            max_context_length=max_context_length,
            max_seq_len=max_seq_len,
            attention_window_size=attention_window_size,
            beam_width=beam_width,
            mask_type=mask_type,
            quant_mode=quant_mode,
            q_scaling=q_scaling,
            position_embedding_type=position_embedding_type,
            rope_dim=rope_dim,
            rope_base=rope_base,
            rope_scale_type=rope_scale_type,
            rope_scale=rope_scale,
            rope_short_m_scale=rope_short_m_scale,
            rope_long_m_scale=rope_long_m_scale,
            rope_max_positions=rope_max_positions,
            rope_original_max_positions=rope_original_max_positions,
            use_paged_context_fmha=use_paged_context_fmha,
            attention_input_type=attention_input_type,
            is_mla_enable=is_mla_enable,
            chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_append=rope_append,
            mrope_rotary_cos_sin=mrope_rotary_cos_sin,
            mrope_position_deltas=mrope_position_deltas,
            helix_position_offsets=helix_position_offsets,
            helix_is_inactive_rank=helix_is_inactive_rank,
            attention_chunk_size=attention_chunk_size,
            softmax_stats_tensor=softmax_stats_tensor,
            is_spec_decoding_enabled=is_spec_decoding_enabled,
            use_spec_decoding=use_spec_decoding,
            is_spec_dec_tree=is_spec_dec_tree,
            spec_decoding_generation_lengths=spec_decoding_generation_lengths,
            spec_decoding_position_offsets=spec_decoding_position_offsets,
            spec_decoding_packed_mask=spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=spec_decoding_bl_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=spec_bl_tree_first_sparse_mask_offset_kv,
            sparse_kv_indices=sparse_kv_indices,
            sparse_kv_offsets=sparse_kv_offsets,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
            sparse_attn_indices_block_size=sparse_attn_indices_block_size,
            num_sparse_topk=num_sparse_topk,
            sparse_attn_kv_lens=sparse_attn_kv_lens,
            skip_softmax_threshold_scale_factor_prefill=skip_softmax_threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode=skip_softmax_threshold_scale_factor_decode,
            skip_softmax_stat=skip_softmax_stat,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
            flash_mla_tile_scheduler_metadata=flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=flash_mla_num_splits,
            sage_attn_num_elts_per_blk_q=sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=sage_attn_qk_int8,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            trtllm_gen_jit_warmup=trtllm_gen_jit_warmup,
            aux_kv_cache_pool_ptr=aux_kv_cache_pool_ptr,
            is_cross=is_cross,
            cross_kv=cross_kv,
            relative_attention_bias=relative_attention_bias,
            relative_attention_max_distance=relative_attention_max_distance,
            spec_decoding_target_max_draft_tokens=spec_decoding_target_max_draft_tokens,
            quant_scale_qkv=quant_scale_qkv,
            dsv4_inv_rope_cos_sin_cache=dsv4_inv_rope_cos_sin_cache,
            enable_dsv4_epilogue_fusion=enable_dsv4_epilogue_fusion,
            max_num_sequences=max_num_sequences,
            force_prepare_spec_dec_tree_mask=force_prepare_spec_dec_tree_mask,
            kv_norm_weight=kv_norm_weight,
            kv_norm_eps=kv_norm_eps,
            skip_correction_threshold=skip_correction_threshold,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_lens: Sequence[int],
        kv_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        softmax_scale: float,
        causal: bool = True,
        attention_sinks: Optional[torch.Tensor] = None,
        window: Optional[int] = None,
        e4m3_inputs: bool = False,
        out_scale: float = 1.0,
        e4m3_probabilities: bool = False,
    ) -> torch.Tensor:
        """Masked softmax attention over explicit per-sequence K/V, in fp32.

        The paged-cache exception to mirroring `__call__`, and the largest one:
        this op's output is a function of what the pool holds, and its 115
        arguments mostly say where to look. `k` and `v` are the gathered
        histories the call should have attended over; the test reads them out of
        the environment, which is the only place they exist.

        Causal is bottom-right aligned -- query `i` of a sequence sits at
        absolute position `kv_len - q_len + i`, which is what makes a decode
        step's single query attend to everything before it, and a P-token draft
        block's row `t` attend `[0, kv_len - P + t]`. With no KV cache, `k` and
        `v` are the call's own packed rows and `q_lens == kv_lens`.

        `attention_sinks` adds one per-head logit to the softmax *denominator*
        and drops it from the numerator, so the attention weights sum to less
        than one. It is not scaled by `softmax_scale`: the rival hypothesis that
        it is was measured and rejected, and the test still drives both so the
        distinction cannot quietly stop holding.

        `window` keeps only the newest `window` keys per query row. It is a pure
        mask -- the pool is shared with the full-attention layers and nothing is
        evicted -- which is why one cell can differ from another by this alone.

        `e4m3_inputs` rounds q, k and v through e4m3, which is what the fp8 path
        quantizes them to; `out_scale` is the second of that path's two
        dequantization factors, the first riding in `softmax_scale`.

        `e4m3_probabilities` rounds the softmax probabilities too, the value
        product's other operand on the standard configuration's fp8 kernels:
        `exp(score - row max)` scaled by e4m3's 448, rounded, unscaled, over a
        row sum kept in fp32. Measured as what those kernels compute: 0.22 bf16
        ulp RMS where the KV range fits one tile, 2.4 on a 200-token decode
        (the kernel rounds against each tile's running max, this against the
        row's), against 5-6 with the probabilities left in fp32 and 8 with them
        rounded unscaled. At kv scale 2.0 it is bitwise exact on a one-tile
        decode; at 1.5 nothing tried fits (4.2 ulp RMS at best). Not modelled
        with sinks.
        """
        rep = num_heads // num_kv_heads
        total_q = sum(q_lens)
        out = torch.full(
            (total_q, num_heads * v_head_dim), float("nan"), dtype=torch.float32, device=q.device
        )
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lens, kv_lens):
            if kv_len == 0:
                # The op leaves these rows undefined and the merge plans skip
                # them; saying so here keeps a caller from reading them.
                q_start += q_len
                continue
            q_seq = q[q_start : q_start + q_len].reshape(q_len, num_heads, qk_head_dim).float()
            k_seq = k[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, qk_head_dim)
            v_seq = v[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, v_head_dim)
            k_seq, v_seq = k_seq.float(), v_seq.float()
            if rep > 1:
                k_seq = k_seq.repeat_interleave(rep, dim=1)
                v_seq = v_seq.repeat_interleave(rep, dim=1)
            if e4m3_inputs:
                q_seq = q_seq.to(torch.float8_e4m3fn).float()
                k_seq = k_seq.to(torch.float8_e4m3fn).float()
                v_seq = v_seq.to(torch.float8_e4m3fn).float()

            scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq) * softmax_scale
            keep = self._mask(q_len, kv_len, causal, window, q.device)
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))

            if e4m3_probabilities:
                assert attention_sinks is None, "e4m3 probabilities are not modelled with sinks"
                weights = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
                e4m3_max = torch.finfo(torch.float8_e4m3fn).max
                rounded = (weights * e4m3_max).to(torch.float8_e4m3fn).float() / e4m3_max
                probs = rounded / weights.sum(dim=-1, keepdim=True)
            elif attention_sinks is None:
                probs = torch.softmax(scores, dim=-1)
            else:
                sink = attention_sinks.float().view(num_heads, 1)
                peak = torch.maximum(scores.max(dim=-1).values, sink)
                weights = torch.exp(scores - peak.unsqueeze(-1))
                probs = weights / (weights.sum(-1) + torch.exp(sink - peak)).unsqueeze(-1)

            out[q_start : q_start + q_len] = (
                torch.einsum("hij,jhd->ihd", probs, v_seq).reshape(q_len, -1) * out_scale
            )
            q_start += q_len
            kv_start += kv_len
        return out.to(q.dtype)

    def softmax_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_lens: Sequence[int],
        kv_lens: Sequence[int],
        num_heads: int,
        num_kv_heads: int,
        qk_head_dim: int,
        softmax_scale: float,
        causal: bool = True,
        window: Optional[int] = None,
    ) -> torch.Tensor:
        """The second output the MLA chunked-prefill path writes, `[Tq, H, 2]`.

        Per (token, head): the row max of the scaled logits and the sum of
        `exp(logit - max)` over *this pass's* KV range. Separate from
        `reference` because it is one path's extra product, not part of what
        attention computes, and folding it in would make every other cell carry
        a return value it has no claim about.

        Rows of zero-KV sequences stay NaN, as the op leaves them.
        """
        rep = num_heads // num_kv_heads
        total_q = sum(q_lens)
        stats = torch.full(
            (total_q, num_heads, 2), float("nan"), dtype=torch.float32, device=q.device
        )
        q_start = kv_start = 0
        for q_len, kv_len in zip(q_lens, kv_lens):
            if kv_len == 0:
                q_start += q_len
                continue
            q_seq = q[q_start : q_start + q_len].reshape(q_len, num_heads, qk_head_dim).float()
            k_seq = k[kv_start : kv_start + kv_len].reshape(kv_len, num_kv_heads, qk_head_dim)
            k_seq = k_seq.float()
            if rep > 1:
                k_seq = k_seq.repeat_interleave(rep, dim=1)
            scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq) * softmax_scale
            keep = self._mask(q_len, kv_len, causal, window, q.device)
            scores = scores.masked_fill(~keep.unsqueeze(0), float("-inf"))
            row_max = scores.max(dim=-1).values
            row_sum = torch.exp(scores - row_max.unsqueeze(-1)).sum(dim=-1)
            stats[q_start : q_start + q_len, :, 0] = row_max.transpose(0, 1)
            stats[q_start : q_start + q_len, :, 1] = row_sum.transpose(0, 1)
            q_start += q_len
            kv_start += kv_len
        return stats

    @staticmethod
    def _mask(q_len: int, kv_len: int, causal: bool, window: Optional[int], device) -> torch.Tensor:
        """`[q_len, kv_len]` keep-mask, bottom-right aligned."""
        if not causal and window is None:
            return torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        keep = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
        offset = kv_len - q_len
        for i in range(q_len):
            high = offset + i + 1 if causal else kv_len
            low = 0 if window is None else max(0, high - window)
            keep[i, low:high] = True
        return keep

    def is_valid(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        attention_sinks: Optional[torch.Tensor] = None,
        **unused_kwargs: object,
    ) -> None:
        """Keyword-only, because every call site passes keywords.

        The one entry that does not mirror `__call__`. This op takes 115
        arguments; naming all of them here would restate the signature without
        adding a claim, and `reference` cannot mirror it either -- it is the
        paged-cache exception and takes gathered K/V -- so there is no shared
        argument list left for the mirroring to preserve. `**unused_kwargs`
        absorbs the rest.
        """
        # A context-only call with a non-multiple returns without raising:
        # measured at 6q/4kv d128, only the first
        # `(num_heads // num_kv_heads) * num_kv_heads` head columns of output are
        # computed and the rest come back all-zero. The geometry cannot be left
        # for the op to reject.
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
        # attention_sinks is consumed as a raw buffer of exactly num_heads fp32
        # values read from data_ptr(); only its dtype is validated by the op.
        # Measured: a stride-2 view whose values were correct produced silently
        # wrong output (the underlying memory is what gets read), a 32-element
        # tensor was read 64 elements deep past its end, a 128-element one had
        # its tail silently ignored, and an empty tensor silently disabled the
        # sink.
        if attention_sinks is not None:
            assert attention_sinks.is_contiguous() and attention_sinks.numel() == num_heads, (
                f"attention_sinks must be contiguous with exactly num_heads "
                f"({num_heads}) elements, got shape "
                f"{tuple(attention_sinks.shape)} contiguous="
                f"{attention_sinks.is_contiguous()}"
            )


thop_attention = ThopAttention()
