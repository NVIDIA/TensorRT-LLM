# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA generation-phase RoPE + latent KV-cache append + FMHA scheduler-buffer fill.

Hy4-Flash's decode prologue: the catalog entry ``attention/mla_rope_generation``,
copied into this target and certified over the checkpoint's geometry, its two
RoPE bases, its interleaved three-section mRoPE table and the pools it runs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: quant_mode bit for an fp8 KV cache.
QM_FP8_KV_CACHE = 128

#: Hy4-Flash's interleaved mRoPE: the 32 rope pairs are split [12, 10, 10]
#: across the temporal / height / width position streams (config
#: `mrope_section`), and each pair's angle comes from its own stream's
#: position. `MROPE_OFFSETS` makes the three streams differ by a fixed amount,
#: which is what a token whose streams have advanced differently sees; at zero
#: offsets the table is the plain single-position one.
MROPE_SECTION = (12, 10, 10)
MROPE_OFFSETS = (0, 7, 19)

#: The checkpoint's MLA geometry: a 512-wide latent plus a 64-wide rope half, a
#: 192-wide nope half, q_lora 1024 and a 256-wide value. `qk_nope_head_dim`
#: reaches only `mla_bmm1_scale` and `v_head_dim` nothing on this path, but they
#: are the checkpoint's values so a cell cannot drift from it.
_HY4 = dict(
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    q_lora_rank=1024,
    v_head_dim=256,
    q_scaling=0.0625,
    attention_window_size=512,
    mrope_section=MROPE_SECTION,
    mrope_offsets=MROPE_OFFSETS,
)

#: Hy4-Flash's two RoPE bases: the sparse layers' 1e7 and the SWA layers' 1e4.
THETA_SPARSE = 10_000_000.0
THETA_SWA = 10_000.0

#: The primary pool the drafter's block appends to, at tep4: the checkpoint's 42
#: text layers, one pool row per decoder layer (`MlaLayer._pool_row` returns the
#: decoder index), and row 25 -- the first of the sparse layers DFly2 reuses as
#: its KV source (`dims.target_kv_layers` = [25, 29, 33, 37, 41]), well above 0.
#: A pool row above 0 is what a single-row pool cannot show, so these cells name
#: the pool too.
NUM_TEXT_LAYERS = 42
DRAFT_TARGET_LAYER = 25
#: The pool's served length at the gate: DFly2 logged "served length 5134" (the
#: 5120 `max_seq_len` grown by the draft block). A sparse layer has no window of
#: its own -- `MlaLayer._window` falls through to `_pool_max` -- so the drafter's
#: call passes this length as `attention_window_size`, at rather than below the
#: KV length.
DRAFT_POOL_LEN = 5134
#: The token budget the drafter's cells size that 42-row pool with. The batch
#: holds ~9.4k tokens over its 64 sequences (the longest ending its block at the
#: pool's served length); 24576 tokens over 42 rows is the allocation.
DRAFT_POOL_TOKENS = 24576

#: The drafter's geometry: `_HY4` with the one value its call changes -- the
#: window, which for a reused sparse layer is the pool's served length rather
#: than the SWA layers' 512 tokens. A cell cannot pass that override beside
#: `**_HY4` (the same keyword twice), so the merged dict is named here.
_HY4_DRAFT = {**_HY4, "attention_window_size": DRAFT_POOL_LEN}


class MlaRopeGeneration(OpWrapper):
    """One decode step's MLA preprocessing, over the batch's generation
    sequences only.

    Each generation sequence carries `P = predicted_tokens_per_seq` new query
    tokens -- 1 is an ordinary decode, `P > 1` the MTP shape. The per-token
    tensors hold `G * P` rows token-major within a sequence, and row `(g, t)`
    sits at absolute position `sequence_length[num_contexts + g] - P + t`, so
    each row is rotated at its own angle.

    Over the bf16 latent pool this target runs, the call writes the roped
    `q_pe` into `fused_q`'s rope tail, appends `[compressed_kv | roped k_pe]`
    to the pool, and fills the two scheduler buffers. Over a quantized pool it
    writes the query into `quant_q_buffer` and the quantized row to the pool
    instead, and also fills `mla_bmm1_scale` / `mla_bmm2_scale`. Either way it
    returns `fused_q`, and the cells below read the buffers and the pool for
    the effect.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Hy4-Flash's generation shapes. The geometry is the checkpoint's, so the
    # cells vary what a decode step varies: which of the two head counts (the
    # 64-head SWA layers and the 32-head sparse ones, their TP4 shards at 16 and
    # 8, and the disaggregated context engine's attn_tp-2 shards -- 32 for SWA
    # and 16 for sparse, the same numbers the *other* layer kind carries at full
    # width and at tp4), how many new tokens (1 for decode, 6 for DFly2's five
    # drafts plus the anchor), the page size and the RoPE base. Every cell runs
    # the checkpoint's interleaved three-section mRoPE table (`MROPE_SECTION`);
    # the ten single-row cells run it under a 512-token window *below* their KV
    # length, the drafter's two at the pool's served length instead -- a sparse
    # layer has no window of its own (`MlaLayer._window` falls through to
    # `_pool_max`), so its window sits at the KV length rather than below it.
    #
    # The last two are the drafter's own call at tep4, and they add the three
    # axes a single-row pool over a short KV cannot show: the pool row the call
    # names (25, the first sparse layer DFly2 reuses as its KV source), the 42
    # rows that pool holds (one per text layer), and a KV length at the pool's
    # served length instead of a short one. The second of them leads the batch
    # with a context row, so the op resolves its generation tail at
    # `num_contexts + g` rather than from 0.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="SWA decode: 64 heads, P=1, 32-token pages, bf16 pool, mRoPE at 1e4",
            spec=dict(
                num_heads=64,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="SWA decode over 64-token pages: 64 heads, P=1, bf16 pool",
            spec=dict(
                num_heads=64,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=64,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="the DFly2 verify step: 64 heads, P=6, bf16 pool, mRoPE at 1e4",
            spec=dict(
                num_heads=64,
                num_seqs=4,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="sparse-layer decode: 32 heads, P=1, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=32,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="sparse-layer verify: 32 heads, P=6, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=32,
                num_seqs=4,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="a TP4 shard of the SWA layers: 16 heads, P=1, bf16 pool",
            spec=dict(
                num_heads=16,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="a TP4 shard of the sparse layers: 8 heads, P=1, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=8,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="the ctx engine's SWA decode at attn_tp 2: 32 heads, P=1, 32-token pages, bf16 pool, mRoPE at 1e4",
            spec=dict(
                num_heads=32,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        Cell(
            why="the ctx engine's sparse decode at attn_tp 2: 16 heads, P=1, 32-token pages, bf16 pool, mRoPE at 1e7",
            spec=dict(
                num_heads=16,
                num_seqs=4,
                predicted_tokens_per_seq=1,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                **_HY4,
            ),
        ),
        Cell(
            why="the envelope's largest row: 64 sequences at P=6, 384 q rows, 64 heads, bf16",
            spec=dict(
                num_heads=64,
                num_seqs=64,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SWA,
                **_HY4,
            ),
        ),
        # The drafter's own call. `**_HY4_DRAFT` is `_HY4` with the window at
        # the pool's served length (`MlaLayer._window` -> `_pool_max`) instead
        # of the 512-token SWA window; the pool is the checkpoint's 42 rows
        # rather than one, which the cells name.
        Cell(
            why=(
                "the drafter's block prologue at tep4: 8 q heads, P=6, 32-token pages, bf16 pool, "
                "mRoPE at 1e7, layer_idx 25 the target sparse layer's pool row, over a KV past 1024 positions"
            ),
            spec=dict(
                **_HY4_DRAFT,
                num_heads=8,
                num_seqs=64,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                layer_idx=DRAFT_TARGET_LAYER,
                num_layers=NUM_TEXT_LAYERS,
                num_contexts=0,
                kv_len=DRAFT_POOL_LEN,
                max_tokens=DRAFT_POOL_TOKENS,
            ),
        ),
        Cell(
            why="the same call in a mixed batch: context rows ahead of the generation requests, num_contexts above 0",
            spec=dict(
                **_HY4_DRAFT,
                num_heads=8,
                num_seqs=64,
                predicted_tokens_per_seq=6,
                tokens_per_block=32,
                quant_mode=0,
                rope_theta=THETA_SPARSE,
                layer_idx=DRAFT_TARGET_LAYER,
                num_layers=NUM_TEXT_LAYERS,
                num_contexts=1,
                kv_len=DRAFT_POOL_LEN,
                max_tokens=DRAFT_POOL_TOKENS,
            ),
        ),
    )

    note = """
    Certified on sm_103 over the checkpoint's MLA geometry: a 512-wide latent
    plus a 64-wide rope half, q_lora 1024, and both of its head counts (64 for
    the SWA layers, 32 for the sparse ones) with their TP4 shards at 16 and 8 and
    the disaggregated context engine's attn_tp-2 shards -- 32 for the SWA layers
    at 1e4 and 16 for the sparse ones at 1e7. Those two counts are the *other*
    layer kind's full count and the SWA tp4 shard, so each ctx cell names its
    layer kind's base as well as the count. The pools the model runs are
    driven: a bf16 latent pool (`quant_mode=0`) at every head count, and for
    the drafter's cells the checkpoint's 42-row primary pool with the call
    naming row 25.

    The cells run `P` up to 6 -- DFly2's five drafts plus the anchor, which is
    the largest `predicted_tokens_per_seq` the model asks for -- over an
    interleaved three-section mRoPE table (32 rope pairs split [12, 10, 10]
    across the temporal / height / width streams, the config's `mrope_section`)
    whose three position streams differ, which is the table the model builds
    rather than the plain single-position one. Both RoPE bases are driven: 1e4
    for the SWA layers and 1e7 for the sparse ones. Ten cells run under a
    512-token attention window *below* their KV length; the drafter's two run at
    the length its pool serves, which is the window the model itself passes for
    a layer with none of its own (`MlaLayer._window` falls through to
    `_pool_max`).

    The table is the engine's duplicated layout, not one copy per position:
    `2 * qk_rope_head_dim` floats per position, the 32 `(cos, sin)` pairs
    repeated, which is what `cos_sin_cache + ROPE_DIM * position_id` in
    `float2` units strides over. A table built without
    `RopeParams(duplicate_data=True)` is half as long and the kernel reads past
    its end -- measured here as an illegal memory access that killed the CUDA
    context and aborted the process in a later test, with nothing pointing at
    the call. `is_valid` refuses a table shorter than the positions
    `sequence_length` names; it cannot see a table of the right length built
    split-half instead of duplicated, which is silently wrong.

    The bf16 path is the one driven. The roped `q_pe` lands in `fused_q`'s
    rope tail, the absorbed-q head is untouched, the pool row is the nope half
    copied verbatim from `latent_cache` followed by its roped rope half,
    `quant_q_buffer` is unused, and `mla_bmm1_scale` / `mla_bmm2_scale` are
    left alone -- they belong to the quantized paths, so `reference_bmm_scales`
    states values no cell here drives.

    The write is not windowed. `attention_window_size` sizes the reader's cyclic
    remap (`KVBlockArray::getKVTokenIdx`) and the block array's window fields,
    while this append addresses the pool at the token's absolute position
    through `KVBlockArray::getKBlockPtr`. A test asserts that rather than
    assuming it: the same call with the window raised past the KV length is
    byte-identical, so a window below it loses nothing at this step -- which is
    why the drafter's cells can pass the length its pool serves, the window the
    model passes for a layer with none of its own.

    The drafter's call names a pool row rather than row 0, and a row is not a
    pool. Nothing in the coalescing key holds a layer id (`createStorageConfig`
    keys on `(coalescingGroup, size, sparsity)`, `makeLifeCycle` on
    `(slidingWindow, sinks, tokensPerBlock, policy)`), so the checkpoint's 42
    uniform MLA layers land in one pool interleaved page by page: the pool's
    `index_scales` is the row count 42, and `kv_cache_pool_mapping` row `L` is
    `[0, L]` -- the byte displacement the op adds for the pool row it resolved
    `layer_idx` to. Measured on that pool: `pool_pointers[0][0] + L * 36864`
    (one 32-token page) equals `get_buffers(L).data_ptr()` for rows 0, 25 and
    41, and a slot's page index is the same number for every row (`42 * slot`),
    so a `layer_idx` off by one writes a *different row's* page rather than its
    own page at a wrong offset. The test reads the pool back through
    `get_batch_cache_indices(..., layer_idx=L)`, which applies the same
    conversion the op's block-offset table is built with
    (`base * index_scale + kv_offset`), and asserts per cell that the call moved
    no pool page outside the ones its row names.

    The schema annotation is wrong in both directions: it marks `fused_q` and
    `q_pe` mutable, but `q_pe` is never modified and on the quantized path
    neither is `fused_q`, while `cu_q_seqlens`, `cu_kv_seqlens`,
    `fmha_scheduler_counter`, `quant_q_buffer` and both scale buffers -- all
    filled by the call -- carry no annotation at all. Do not rely on its alias
    info under functionalization.

    Re-running a prepared step is idempotent rather than double-appending: the
    positions come from `sequence_length`, so the same slots are rewritten.

    Measured: nine of the twelve cells agree with `reference` bit for bit on
    every roped element; the three that do not differ on 1/99840, 22/1597440
    and 4/221184 elements, the worst by 8.17e-4 of the tensor's largest
    magnitude -- 5% of the 4-ulp band the test allows (0.0156 of the scale).
    The drafter's own cell (42-row pool, row 25, KV at the pool's served
    length) is among the bit-exact ones; its mixed-batch twin is among the
    three, at 4/221184 and 4.46e-4. The mRoPE control, the same call with the
    position streams mirrored, moves 42.2% of the `k_pe` bytes with a worst
    element at 0.88 of the scale, so the table is doing the work rather than
    the comparison being loose.

    Not certified: the op's quantized pool paths at this geometry -- the
    fp8-e4m3 one (`quant_mode=128`, which the catalog entry certifies at
    DeepSeek-R1's 128-head geometry instead) and the NVFP4 one
    (`quant_mode=8192`) -- so neither `quant_q_buffer` nor either bmm scale is
    driven here; `P` above 6; a KV length above the pool's served length
    (5134); more than one context row leading the batch (`num_contexts` 1
    -- the smallest mixed step -- is driven; the gate's own batches ran at
    concurrency 1, so its logs hold no wider one); the ragged token-major
    presentation (`presented_token_major`, taken when
    `token_major_gen_view()` is not None, which presents a uniform block's rows
    at `seq_len == 1` instead of one row per sequence); pool rows other than
    25, which share its arithmetic (measured for rows 0, 1, 25 and 41) but are
    not driven by a cell; `beam_width > 1`; `rope_append=False`; the
    `precomputed_*` flags; `kv_only` / `kv_done_elsewhere`; `quant_scale_qkv`;
    `block_ids_per_seq` (the flash-MLA layout); helix; and `out_scale`.
    """

    def __call__(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: Optional[torch.Tensor],
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        kv_cache_scale_orig_quant: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        residual_dim: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch.ops.trtllm.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            rotary_cos_sin,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            sequence_length,
            host_past_key_value_lengths,
            host_context_lengths,
            num_contexts,
            kv_cache_block_offsets,
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            kv_scale_orig_quant,
            kv_scale_quant_orig,
            kv_cache_scale_orig_quant,
            out_scale,
            block_ids_per_seq,
            helix_tensor_params,
            predicted_tokens_per_seq,
            layer_idx,
            num_heads,
            num_kv_heads,
            head_size,
            residual_dim,
            tokens_per_block,
            attention_window_size,
            beam_width,
            quant_mode,
            q_scaling,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_append,
            kv_norm_weight,
            kv_norm_eps,
            precomputed_cu_seqlens,
            precomputed_fmha_scheduler,
            kv_only,
            kv_done_elsewhere,
            quant_scale_qkv,
        )
        return fused_q

    def reference(
        self,
        row: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rope_size: int,
        kv_scale_orig_quant: Optional[torch.Tensor] = None,
        store_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> torch.Tensor:
        """One quantized `[.. , C+R]` row this op is supposed to write.

        The same documented exception to mirroring as the other paged-cache
        entries: the op's effect is on buffers and its arguments only say where.
        One function covers both quantized outputs, because they are the same
        computation on different inputs -- `[fused_q[..., :C] | q_pe]` for a
        `quant_q_buffer` row, `[compressed_kv | k_pe]` for an appended cache
        row. `cos` / `sin` are the rope table's halves already selected for the
        row's absolute position; the test reads them out of the environment,
        which is the only place they exist.

        Two details decide whether a mirror agrees with the kernel:

        The pairing is **GPT-J interleaved** -- consecutive elements form a pair
        -- not the split-half neox form. The two are indistinguishable by shape
        and give different answers.

        The rotation is rounded to the activation dtype **before** quantizing,
        which is what the kernel does. Quantizing the fp32 rotation directly
        disagrees on ~3% of the roped elements: two roundings, not one.

        The passthrough half is not re-rounded -- it arrives in `row`'s dtype
        and is only scaled. The scale is applied once, so the stored bytes are a
        function of the row rather than of the order the kernel visited it in.
        """
        pe = row[..., row.shape[-1] - rope_size :]
        lora = row[..., : row.shape[-1] - rope_size]
        pairs = pe.float().reshape(*pe.shape[:-1], rope_size // 2, 2)
        roped = torch.empty_like(pairs)
        roped[..., 0] = pairs[..., 0] * cos - pairs[..., 1] * sin
        roped[..., 1] = pairs[..., 0] * sin + pairs[..., 1] * cos
        # Round to the activation dtype first: the kernel's own intermediate.
        out = torch.cat([lora, roped.reshape(pe.shape).to(row.dtype)], dim=-1).float()
        if kv_scale_orig_quant is not None:
            out = out * kv_scale_orig_quant[0]
        return out.to(store_dtype)

    def reference_bmm_scales(
        self,
        kv_scale_quant_orig: Optional[torch.Tensor],
        q_scaling: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The `(mla_bmm1_scale, mla_bmm2_scale)` this call is supposed to fill.

        A per-batch scalar pair, not per sequence, and `P` does not enter it.
        The read-side factor `r` appears squared in one and plain in the other
        because the consuming fp8 FMHA dequantizes both operands of BMM1 and one
        of BMM2 -- see `is_valid` for what that costs when `r != 1/w`.

        Computed in double and rounded once, so the comparison measures the
        kernel: every way of getting this scale wrong misses by a factor rather
        than by ulps.
        """
        r = 1.0 if kv_scale_quant_orig is None else float(kv_scale_quant_orig[0])
        x = r * r / (q_scaling * math.sqrt(qk_nope_head_dim + qk_rope_head_dim))
        return (
            torch.tensor([x, x * math.log2(math.e)], dtype=torch.float32),
            torch.tensor([r], dtype=torch.float32),
        )

    def reference_scheduler_buffers(
        self,
        kv_lens: List[int],
        num_heads: int,
        predicted_tokens_per_seq: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The `(cu_q_seqlens, cu_kv_seqlens)` this call is supposed to fill.

        `kv_lens` is the generation sequences' total KV lengths, in batch order,
        already including all `P` of this step's tokens. The q buffer is in
        units of q rows -- `num_heads` rows per generation *token*, so
        `num_heads * P` per sequence -- and the kv buffer in tokens. Exactly
        `len(kv_lens) + 1` entries are written whatever `P` is; anything past
        that is left alone.
        """
        p = predicted_tokens_per_seq
        cu_q = [i * num_heads * p for i in range(len(kv_lens) + 1)]
        cu_kv = [0]
        for length in kv_lens:
            cu_kv.append(cu_kv[-1] + length)
        return (
            torch.tensor(cu_q, dtype=torch.int32),
            torch.tensor(cu_kv, dtype=torch.int32),
        )

    def is_valid(
        self,
        fused_q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        rotary_cos_sin: Optional[torch.Tensor],
        cu_q_seqlens: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        fmha_scheduler_counter: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        quant_q_buffer: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        num_contexts: int,
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        kv_cache_scale_orig_quant: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        helix_tensor_params: List[Optional[torch.Tensor]],
        predicted_tokens_per_seq: int,
        layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        residual_dim: int,
        tokens_per_block: int,
        attention_window_size: int,
        beam_width: int,
        quant_mode: int,
        q_scaling: float,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_append: bool,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        precomputed_cu_seqlens: bool = False,
        precomputed_fmha_scheduler: bool = False,
        kv_only: bool = False,
        kv_done_elsewhere: bool = False,
        quant_scale_qkv: Optional[torch.Tensor] = None,
    ) -> None:
        """The inputs this op takes and answers wrongly.

        The signature mirrors `__call__` position for position because
        `validating` forwards a call's arguments verbatim and every caller here
        passes them positionally: a narrower signature would collect
        `residual_dim` and `quant_mode` into a catch-all and check nothing,
        silently. Only `rotary_cos_sin`, `sequence_length`, `quant_q_buffer`,
        `kv_scale_orig_quant`, `kv_scale_quant_orig`, `residual_dim`,
        `quant_mode` and `kv_norm_weight` are read.
        """
        # `residual_dim` must be 0: this entry certifies the plain-row layout,
        # where every token's data row is `kv_lora_rank + qk_rope_head_dim`
        # elements and the block scales live in their own pool. The inline-scale
        # record (`residual_dim == 2 * qk_rope_head_dim`, block scales in the
        # data row) is the DSA PrimTS layout, and a value in between is the
        # residual layout; no call here makes either, and against a plain row
        # they read bytes that are not there.
        assert residual_dim == 0, (
            f"residual_dim must be 0 for the plain MLA row layout; got {residual_dim}. "
            "Only the DSA PrimTS record and residual layouts use a non-zero value"
        )

        # The kernel strides the rope table `2 * qk_rope_head_dim` floats per
        # position -- the duplicated pair layout -- and reads it at each row's
        # absolute position, up to `max(sequence_length) - 1`. A table with one
        # copy per position is half as long, and the read past its end is not
        # checked: it lands somewhere plausible and the process later dies of an
        # illegal memory access with nothing pointing here. Measured, with a
        # `duplicate_data=False` table at 700 positions.
        if rotary_cos_sin is not None and sequence_length is not None:
            need = 2 * qk_rope_head_dim * int(sequence_length.max())
            assert rotary_cos_sin.numel() >= need, (
                f"rotary_cos_sin holds {rotary_cos_sin.numel()} floats, short of the "
                f"{need} a table of {2 * qk_rope_head_dim} floats per position needs for "
                f"position {int(sequence_length.max()) - 1}: the kernel strides "
                f"{2 * qk_rope_head_dim} floats per position (the duplicated pair layout, "
                "RopeParams(duplicate_data=True)) and reads past the end"
            )

        # Over an fp8 pool the quantized query has nowhere to go without this
        # buffer, and its absence is not checked: the kernel launches anyway,
        # the run dies with an illegal memory access, the CUDA context is lost
        # and the process aborts with nothing pointing here. The NVFP4 path
        # needs the same buffer but checks it loudly in the op, so the guard
        # does not duplicate it.
        if quant_mode & QM_FP8_KV_CACHE:
            assert quant_q_buffer is not None, (
                "an fp8 KV cache (quant_mode & 128) requires quant_q_buffer; the op does "
                "not presence-check it and launches into a null pointer, killing the "
                "CUDA context"
            )

        # A non-None weight folds the kv_a_layernorm into this kernel, which
        # then reads latent_cache RAW. A caller that already normalized -- every
        # caller here -- would be normalizing twice, and the result is plausible.
        assert kv_norm_weight is None, (
            "kv_norm_weight folds kv_a_layernorm into this kernel, which then reads "
            "latent_cache unnormalized; a caller that already normalized would "
            "normalize twice"
        )

        # The two scales are used independently and are not derived from each
        # other: `w` quantizes what this call writes, `r` is folded into the
        # scales the consuming fp8 FMHA dequantizes with. That cancellation is
        # only correct when `r = 1/w`, and nothing downstream can detect a
        # non-reciprocal pair -- the decode is silently mis-scaled. Driving them
        # apart on purpose is a characterization of the op, not a call a target
        # makes, so such a case belongs outside `validating()`.
        if kv_scale_orig_quant is not None or kv_scale_quant_orig is not None:
            w = 1.0 if kv_scale_orig_quant is None else float(kv_scale_orig_quant[0])
            r = 1.0 if kv_scale_quant_orig is None else float(kv_scale_quant_orig[0])
            assert w != 0.0 and math.isclose(r, 1.0 / w, rel_tol=1e-6), (
                f"kv_scale_quant_orig ({r}) must be the reciprocal of kv_scale_orig_quant "
                f"({w}): the consuming FMHA undoes the write scale with it, and a "
                "non-reciprocal pair mis-scales the decode with nothing to show for it"
            )


mla_rope_generation = MlaRopeGeneration()
