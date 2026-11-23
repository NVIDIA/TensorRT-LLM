# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Flash's sliding-window MLA layer: paged latent append + windowed FMHA.

A local variant of ``catalog/attention/thop_attention.py`` for the
``hyv4_flash_for_conditional_generation`` target at its ``mla_windowed`` call
site: the thirty sliding-window layers of the checkpoint's ``swa_pattern``,
which run MLA's two phases with 64 q heads on one rank (16 under tp4, 32 under
tp2), latent rows ``[kv_lora 512 | rope 64]``, context qk ``[nope 192 | rope
64]``, context v 256, a per-head fp32 sink, a 512-token window and an
interleaved-mRoPE table built outside the op.

The same entry is the sparse layers' own decode (``MlaLayer._attend``): one
generation row per request at the sparse layers' 32 q heads unsharded (16
under tp2, 8 under tp4), latent MQA over the pool with the indexer's top-k
selection as ``sparse_attn_indices``.

The same entry is the DFly2 drafter's block attention (``Hyv4FlashDraft``
``_block_attention``): a generation-only call over the target sparse layer's
pool, eight q heads on one rank (the sparse layers' 32 under tp4), no window,
and this row's selection -- the anchor's compacted top-``dfly_target_index_topk``
picks followed by the block's own causal prefix -- in ``sparse_attn_indices``,
one row per block row rather than one latent row per request.

Wraps ``FallbackFmha.attention``, which dispatches to the native
``AttentionOp`` context and generation methods. Every piece of batch state
and layer config arrives as an explicit argument; native runners are cached
by static configuration, device, and thread.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: attention_input_type. 0 is the standard append-and-attend path; 1 and 2 are
#: MLA's context and generation phases.
STANDARD, MLA_CONTEXT, MLA_GENERATION = 0, 1, 2

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128

#: quant_mode bit for an NVFP4 KV cache: e2m1 values in the pool, one e4m3 scale
#: per 16 values in a second pool, fp32 [3] kv scales (q, K, V).
QM_NVFP4_KV_CACHE = 8192

#: Hy4-Flash's sliding-window MLA layer -- the `mla_windowed` call site. 64 q
#: heads on one rank (16 under tp4, 32 under tp2), latent rows `[kv_lora 512 | rope 64]`,
#: context qk `[nope 192 | rope 64]`, context v 256, a per-head fp32 sink, a
#: 512-token window and page 32. The checkpoint initialises its learned sinks
#: at 0.0; the cells drive a per-head range around it, because a served sink is
#: any fp32 value. `prefill`/`cached`/`new`/`decode` are this test's scenario
#: fields: token counts per request for the context call and decode rows after
#: it. Pool sizes are chosen, not dumped: each request gets the window's
#: sixteen pages and the block-offset table is sized for its longest history.
_HY4_SWA = dict(
    num_heads=64,
    num_kv_heads=64,
    head_size=576,
    v_head_dim=256,
    sinks=True,
    window=512,
    quant_mode=0,
    tokens_per_block=32,
    max_num_requests=64,
    use_paged_context_fmha=False,
)


#: The DFly2 drafter's block call at this site. The sparse layers' 32 q heads on
#: one rank are 8 under tp4, the pool is the bf16 latent pool the sparse and
#: windowed layers share, the window is the pool's max (a sparse row is bounded
#: by its own selection, not by a window), and `predicted_tokens_per_seq` is the
#: block width: `block_width` rows per request, each reading its own
#: `sparse_attn_indices` row of `dfly_target_index_topk` picks plus the block's
#: causal prefix, with `num_sparse_topk` that row's width.
_HY4_DFLY_BLOCK = dict(
    block=True,
    num_heads=8,
    num_kv_heads=1,
    head_size=576,
    v_head_dim=512,
    predicted_tokens_per_seq=6,
    tokens_per_block=32,
    quant_mode=0,
)


class ThopAttention(OpWrapper):
    """Append this step's K/V to the paged cache, then attend over it.

    The shipped targets use the native phased ``AttentionOp`` through
    ``FallbackFmha.attention``: gpt-oss uses attention sinks and an alternating
    sliding window, deepseek uses MLA's two phases, and qwen4_exp uses packed
    GQA over a bf16 or an fp8-e4m3 pool for its full-attention and MTP layers
    and uncached MHA for vision. This variant is Hy4-Flash's ``mla_windowed``
    site: MLA's two phases with a 512-token window and a per-head sink, which
    the checkpoint's sliding-window layers run over the same latent-pool
    geometry as its twelve indexed sparse layers.

    Writes rows `[:num_tokens]` of `output` and returns `output` -- the same
    buffer, mutated and handed back so a call site can show what it touched
    without allocating.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The Hy4 cells are in this variant's test. Each runs both of MLA's phases
    # against one real paged latent pool: a context call (fresh append, or a
    # chunk over a cached prefix on the paged-context path), then a decode row
    # appended at the history's end and one generation call. `window` is 512 in
    # every cell because it is the checkpoint's `sliding_window`; `prefill` 600
    # in the one-request cell is the smallest history that makes the window
    # clip, and the envelope cell's 8192 rows are the largest single-call
    # context row count this variant certifies.
    #
    # The sparse cell next to them is the target sparse layers' own decode at
    # the context deployment's shard: a generation-only call, one row per
    # request, whose `sparse_attn_indices` is the indexer's selection as the
    # core writes it, over histories both longer and shorter than the
    # selection's width.
    #
    # The last four cells are the DFly2 drafter's block call: a generation-only
    # call at `predicted_tokens_per_seq` 6 whose `sparse_attn_indices` is a k29
    # block selection (the anchor's top-2048 picks, then the block's causal
    # prefix), one row per block row, so the op is driven at the shape its
    # drafter uses and no other cell covers. The first two drive that shape at
    # the short `DRAFT_HISTORY`; the last two at the step the engine really
    # takes there -- the batch's device KV length
    # `spec_worker._write_block_kv_lens` leaves in `md.kv_lens_cuda_runtime`
    # (the request's `anchors + width`, 5134 at the gate's served length) with
    # each generating request's bound the writer's count through its block's
    # last row, which is what `draft.Hyv4FlashDraft._block_attention` hands.
    # Each of the last two also drives the device length once as a refused
    # control, once alone and once inside a mixed batch with context rows ahead
    # of the generation rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="Hy4 SWA MLA, one request over a bf16 pool: a 600-token prefill "
            "whose late rows see only the 512-token window, then one decode over "
            "601 keys",
            spec=dict(_HY4_SWA, max_num_requests=1, prefill=600, decode=1),
        ),
        Cell(
            why="64 requests over a bf16 pool: a 64-token prefill each, then one "
            "decode each over 65 keys",
            spec=dict(_HY4_SWA, prefill=64, decode=1),
        ),
        Cell(
            why="one request over a bf16 pool on the paged-context path: 32 new "
            "rows over a 512-token cached prefix, then one decode over 545 keys",
            spec=dict(
                _HY4_SWA,
                max_num_requests=1,
                use_paged_context_fmha=True,
                cached=512,
                new=32,
                decode=1,
            ),
        ),
        Cell(
            why="64 requests over an fp8-e4m3 pool: a 64-token prefill each, then "
            "one decode each over 65 keys",
            spec=dict(_HY4_SWA, quant_mode=QM_FP8_KV_CACHE, prefill=64, decode=1),
        ),
        Cell(
            why="64 requests over a bf16 pool on the paged-context path: 32 new "
            "rows over a 64-token cached prefix each, then one decode each",
            spec=dict(_HY4_SWA, use_paged_context_fmha=True, cached=64, new=32, decode=1),
        ),
        Cell(
            why="the tp4 shard, 16 q heads: 64 requests over a bf16 pool, a "
            "64-token prefill each, then one decode each",
            spec=dict(_HY4_SWA, num_heads=16, num_kv_heads=16, prefill=64, decode=1),
        ),
        Cell(
            why="the tp4 shard, 16 q heads, on the paged-context path: 32 new "
            "rows over a 64-token cached prefix each, then one decode each",
            spec=dict(
                _HY4_SWA,
                num_heads=16,
                num_kv_heads=16,
                use_paged_context_fmha=True,
                cached=64,
                new=32,
                decode=1,
            ),
        ),
        Cell(
            why="the ctx engine's SWA layers at attn_tp 2: 32 q / 32 kv heads, "
            "64 requests over a bf16 pool, a 64-token prefill then a decode "
            "each",
            spec=dict(_HY4_SWA, num_heads=32, num_kv_heads=32, prefill=64, decode=1),
        ),
        Cell(
            why="the envelope's largest context row count: 8192 rows (16 "
            "sequences of 512) in one paged-context call over a bf16 pool, then "
            "one decode each over 513 keys",
            spec=dict(
                _HY4_SWA,
                max_num_requests=16,
                use_paged_context_fmha=True,
                prefill=512,
                decode=1,
                rows=8192,
            ),
        ),
        Cell(
            why="the ctx engine's sparse layers at attn_tp 2: 16 q / 1 kv "
            "head, one decode each over the top-k index selection passed as "
            "sparse_attn_indices",
            spec=dict(
                block=True,
                num_heads=16,
                num_kv_heads=1,
                head_size=576,
                v_head_dim=512,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                max_num_requests=64,
            ),
        ),
        Cell(
            why="the DFly2 draft block at the tp4 shard: 4 requests x 6 rows over "
            "the sparse layers' bf16 pool, with the k29 block selection as "
            "sparse_attn_indices",
            spec=dict(_HY4_DFLY_BLOCK, max_num_requests=4),
        ),
        Cell(
            why="the engine's largest block batch: 64 requests x 6 rows, 384 "
            "generation rows in one call over the sparse layers' bf16 pool, each "
            "row reading its own selection",
            spec=dict(_HY4_DFLY_BLOCK, max_num_requests=64),
        ),
        Cell(
            why="the block at the engine's largest batch and the gate's served "
            "length: 64 requests x 6 rows over the sparse layers' bf16 pool, "
            "each row's selection 2176 wide, the batch's device KV length 5134 "
            "and the entry's bound the writer's count through the block's last "
            "row (2054); the device length is the refused control",
            spec=dict(
                _HY4_DFLY_BLOCK,
                max_num_requests=64,
                num_contexts=0,
                kv_len=5134,
                num_sparse_topk=2176,
            ),
        ),
        Cell(
            why="the same block in a mixed batch: 8 context rows ahead of 56 "
            "generation rows, the engine's largest batch, every sequence's "
            "device KV length 5134 and each generation row's selection 2176 "
            "wide, the generation rows bound by the writer's count through the "
            "block's last row; the device length is the refused control",
            spec=dict(
                _HY4_DFLY_BLOCK,
                max_num_requests=64,
                num_contexts=8,
                kv_len=5134,
                num_sparse_topk=2176,
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
    JIT compile per P. P is driven at 1..8 (max_draft_len 1..7) at 24/2/256 over
    bf16 and fp8 pools. From P = 5 the launch picks its generation kernel by
    batch size, so P 5..8 is also driven at 32 requests and at one batch in
    17..20 (bf16), and P = 8 at 32 requests over the fp8 pool.

    No KV cache (an encoder): pass `kv_cache_block_offsets`,
    `host_kv_cache_pool_pointers` and `host_kv_cache_pool_mapping` all None,
    with `update_kv_cache=True` -- False is rejected ("KV cache update cannot
    be disabled now"). One context sequence per frame; mask_type 0 confines a
    frame to itself. `tokens_per_block` and `use_paged_context_fmha` are
    bitwise inert there. Passing only some of the three None is unmeasured. An
    image is one frame and a video one frame per frame group; the cells run up to
    8192 frames and 32768 rows per call at 16/16/72 under the padding mask.
    `FusedMHARunnerV2` picks a different fmha_v2 kernel for a call whose longest
    frame is at most 64 rows than for a longer one, so a frame's last bits can
    move with the other frames it shares a call with, within `reference`'s band.
    head_size 72 exists in the context phase only: a generation call raises
    "unsupported head_size 72", and the process stays usable.

    Over an fp8-e4m3 pool the packed context path reads the bf16 q rows and
    never the pool, while `use_paged_context_fmha=True` reads the pool, so its
    operands are e4m3: the flag is output-inert over a bf16 pool only. Every
    fp8 cell runs kv scale 1.0 through fp32 [1] scale tensors; at 1.5 `reference`
    does not reproduce the kernel's probability rounding.
    quant_mode 384, 1152, 4224 and 4480 are bit-identical to 128 at 24/4/256;
    4352 (the same bits without FP8_KV_CACHE) reads e4m3 slabs as bf16. At
    24/2/256 only 128 was driven, with KV up to 200 keys at P = 1 and up to
    127 at P 2..8: the fp8 kernels round the softmax probabilities against a
    per-tile max, which the row-max model of
    `reference(e4m3_probabilities=True)` reproduces within one 128-key tile
    and approximates up to 200 keys. The fp8 cells' references pass it with
    the cells' sinks: the sink branch rounds its weights the same way and adds
    `exp(sink - peak)` to the unrounded row sum (1.83 bf16 ulp RMS on the
    64-request cell's context, 0.01 on its decode).

    NVFP4 pool (quant_mode 8192): K and V are stored as e2m1 values, two per
    byte (low nibble = even element), per head page `[tpb, D/2]`, with one e4m3
    scale per 16 values in a second pool whose pointer is the third axis of
    `host_kv_cache_pool_pointers` (`[pools, 2, 2]`: primary/secondary x
    data/scales); K's scales are token-major per head page, V's in four-token
    groups `[tpb/4, D/16, 4]`. The append quantizes with `kv_scale_orig_quant[1]`
    / `[2]` as second-level scales (fp32 `[3]`, `[0]` is q's e4m3 scale). One
    block-offset table addresses both pools, so they must hold the same pages
    per slot. The kernels are trtllm-gen cubins with e4m3 q and e2m1 K/V and
    exist for the paged-context path only, which needs the op to enable FP8
    context FMHA for an FP4 KV cache. Every NVFP4 cell runs the paged-context
    path at kv scale 1.0 and reads K and V through e4m3, so `reference` runs
    over the NVFP4 values with `e4m3_inputs` and `e4m3_probabilities`.

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

    At Hy4's `mla_windowed` site the window is a pure mask and the pool is a
    ring: the engine gives a 512-token windowed layer sixteen pages per
    request, the append writes every token at its absolute page and slot, and
    a history longer than the window overwrites the aged-out slots in place.
    A decode is meant to see the last `min(history, 512)` latent rows, which is
    what the variant's test builds its reference from. The context FMHA
    receives the window as a mask, but the MLA generation path does not carry
    it: `MLA::enqueueGeneration` hard-codes `mLeftSlidingWindow` and
    `mRightSlidingWindow` to -1 under a dense mask, so a decode whose history
    is longer than the window also reads the aged-out keys (measured at 513
    keys: the newest key is read twice and 0.3% of the output leaves the bf16
    band). `attention_sinks` is a per-head fp32 tensor the checkpoint learns
    from 0.0; the cells drive a range around zero, so the initial value is
    inside what they cover.
    Interleaved mRoPE is built outside this op: the MLA path reads only
    `rotary_cos_sin`, position-indexed, so `mrope_rotary_cos_sin` and
    `mrope_position_deltas` are carried but not read, and for text-only
    positions the interleaved table equals the plain duplicated one.

    A sparse layer's own generation call is this same MLA generation path with
    the indexer's selection in `sparse_attn_indices`: 32 q heads unsharded (16
    at tp2, 8 at tp4), latent MQA (`num_kv_heads` 1, absorbed `head_size` 576,
    `v_head_dim` 512), the window at the pool's maximum, `num_sparse_topk` and
    the host KV bound both the selection's width (`index_topk` 2048, as the
    core's `_generation_args` reads it off the buffer), and the device lengths
    the requests' own. The kernel's per-row key count comes from the device
    length, so a row shorter than the width reads exactly its live prefix --
    the `-1` tail the selection's writer leaves past it is never gathered --
    and a row longer than the width reads the width it selected. Measured at
    (16, 1) on sm_103 with 64 requests and one decode row each: 48 requests
    over 2561-key histories, whose lists are their newest 2048 positions, and
    16 over 1025, whose lists are their whole history followed by `-1`. Both
    regimes are inside the band, long rows first: worst element 2.00 / 1.88
    bf16 ulp against the 8.0 gate and relative RMS 0.70 / 0.72 against the 4.0
    one, the short rows gated against a reference that reads only the live
    prefix.

    That agreement is the identity's own, not the kernel's: at that call the
    count the kernel derives and the row's live prefix are the same function
    of the request's length (`min(length, index_topk)`). The drafter's block
    call ends its selection past the picks -- the block's own slots follow
    them -- so there the two part. A row's live prefix is
    `min(anchors + 1, index_topk) + row + 1` (2049..2054 at the gate's served
    length), while the count off the step's device length is
    `min(anchors + row + 1, num_sparse_topk)` with `num_sparse_topk` the
    buffer's own width, `index_width` 2176. They are equal while
    `anchors <= index_topk`, and past it the device length is ahead of the
    live prefix by `anchors - index_topk`, capped at the buffer's width -- 127
    columns in a row to 122 in the last at `anchors` 5128 -- so the writer's
    `PAD_SLOT_ID` tail enters the softmax when the kernel counts it. The call
    site hands the writer's own count instead:
    `draft.py::Hyv4FlashDraft._block_attention` overwrites each generating
    request's `sequence_length` with the writer's count through its block's
    *last* row (`DflyBlockSelection.lengths[width - 1 :: width]`), the one row
    where the block's `width` and the selection's live prefix agree, and keeps
    the step's device lengths on the context rows and on the rope prologue,
    which are what place the block's rows. At that bound the kernel's
    `bound - width + 1 + row` is the row's live prefix at every anchors, and
    the two block cells are green: 64 requests x 6 rows measures worst
    element 1.94 and the mixed batch 1.71, both at relative RMS 0.48, against
    the 8.0 and 4.0 gates. Each cell also drives the step's own device length
    (`spec_worker._write_block_kv_lens` writes `anchors + width` into
    `md.kv_lens_cuda_runtime`) as the bound over the same operands and pool,
    and asserts it leaves the band -- worst element 6.61-6.69, relative RMS
    4.93-4.95 -- which is what counting the device length costs once the
    picks fill the selection; the gate it leaves is the relative-RMS one, its
    worst element still inside the element gate. The entry's own route to the same counts,
    `sparse_attn_kv_lens` (`AttentionOp` takes `mSparseTopK` and
    `ptrSparseMlaTopKLens` from it and selects `SparseType::DynamicTokenSparse`
    when it is set), is not taken at this site: no dynamic-token-sparse kernel
    exists on this target's arch, so a call with it set refuses in the kernel
    lookup rather than running, and the count travels in the bound.

    The context phase runs at this geometry: `MLA::initContext` takes V's
    head width from `v_head_dim` (`fmhaParams.headSizeV = mMLAParams.v_head_dim`),
    so a 192/64/256 MLA context call builds a 256/256 BMM2-N and is dispatched
    (it used to read 192 from `qk_nope_head_dim` and be rejected --
    "Unsupported HeadDim for BMM2-N 192"). `head_size` itself is not K's row
    width -- K/V widths come from `qk_nope_head_dim + qk_rope_head_dim` and
    `v_head_dim`, and the argument only tells a generation MLA op (the
    absorbed latent width) from a context one. What still stops cells at this
    site is outside the entry, measured on this target's geometry: (1) a
    context call whose kv length exceeds the window selects trtllm-gen's
    SlidingOrChunkedCausal mask (`selectTrtllmGenMask`) and no such
    `SeparateQkv` context cubin exists at headDimQk/headDimV 256 on sm_103 or
    sm_100, so the call asserts "Trtllm-gen kernels not found: ...,
    maskType=2, ..." in `fmhaKernels.h` and the MLA context path has no
    unfused fallback (`FmhaDispatcher::isSupported()` probes the fixed causal
    mask at kv length 2 and cannot see the per-call window); (2) the
    generation path's missing window above; (3) over an fp8-e4m3 pool the
    non-absorption context quantizer `invokeMLAContextFp8Quantize` instantiates
    `quantizeCopyInputToFp8Kernel<T, 384, 128, 64, 128, false>` -- DeepSeek's
    128/64/128 head widths -- so this checkpoint's 192/64/256 operands are
    quantized with the wrong per-head layout and the context output is wrong
    (65.7% of elements outside the fp8 band at 64 requests x 64 tokens).

    An NVFP4 pool (`quant_mode` 8192) is uncertified at this site, not
    refused: its kernels are trtllm-gen cubins that exist for the paged-context
    path only, and its append needs a second e4m3 scale pool addressed by the
    third axis of `host_kv_cache_pool_pointers` with `kv_scale_orig_quant` as
    fp32 [3] second-level scales. The cells here drive the one-pool bf16 and
    fp8 shapes this target's layers run; the catalog entry certifies the
    NVFP4 paged-context shape. Values beyond the cells -- more than 64
    requests, a context call above 8192 rows, a window other than 512, head
    counts other than 64, 32 and 16 on the windowed path and 16 and 8 on the
    sparse one, a selection width other than 2048 (`index_topk`) on the
    target's own sparse decode, a block call at any other device KV length
    than the served 5134 or at a `num_contexts` other than the cells' 0 and 8,
    and a q-LoRA rank other than 1024 -- are uncertified here.
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
        decode; at 1.5 nothing tried fits (4.2 ulp RMS at best). With sinks the
        same rounding applies to the sink branch's weights, and the sink term
        stays in the denominator beside the unrounded row sum -- what the
        kernels compute with a sink: 1.83 bf16 ulp RMS on the 64-request fp8
        cell's context and 0.01 on its decode, against 4.45 and 4.26 with the
        probabilities left in fp32.
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
                e4m3_max = torch.finfo(torch.float8_e4m3fn).max
                peak = scores.max(dim=-1, keepdim=True).values
                if attention_sinks is not None:
                    sink = attention_sinks.float().view(num_heads, 1)
                    peak = torch.maximum(peak.squeeze(-1), sink).unsqueeze(-1)
                weights = torch.exp(scores - peak)
                rounded = (weights * e4m3_max).to(torch.float8_e4m3fn).float() / e4m3_max
                den = weights.sum(dim=-1, keepdim=True)
                if attention_sinks is not None:
                    den = den + torch.exp(sink - peak.squeeze(-1)).unsqueeze(-1)
                probs = rounded / den
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
