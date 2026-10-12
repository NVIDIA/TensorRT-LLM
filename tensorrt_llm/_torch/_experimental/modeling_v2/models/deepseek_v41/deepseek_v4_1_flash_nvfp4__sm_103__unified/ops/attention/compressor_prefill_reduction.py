# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill-side KV compressor: per-channel softmax pooling of each window of
`compress_ratio` tokens, with the per-token compressor state persisted to a
paged pool so a later call (next chunk, or decode) can finish an open window.

The op is `torch.ops.trtllm.compressor_prefill_reduction`, registered in
`cpp/tensorrt_llm/thop/compressorOp.cpp`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_ELEMENT_TYPES = (torch.float32, torch.bfloat16)

#: One-shot batches (every start_pos 0): odd and even lengths, page edges, up to
#: the 16384-token step.
_ONE_SHOT = (
    (1,),
    (2,),
    (3,),
    (7, 8),
    (5, 128, 255, 256, 1000),
    (8191,),
    (8192,),
    (16383,),
    (16384,),
)

#: Five calls over five sequences (301, 130, 1027, 5 and 8192 tokens): one row per
#: sequence, its chunk size in each call, so members at different start parities
#: share every call, 1-token chunks only park a pending token (zero outputs for
#: that member), and boundaries fall at odd positions.
_CHUNKED = (
    (1, 2, 97, 1, 200),
    (33, 31, 1, 1, 64),
    (128, 129, 256, 257, 257),
    (1, 1, 1, 1, 1),
    (4095, 1, 1, 4094, 1),
)


def _spec(plan: tuple, **overrides) -> dict:
    spec = dict(plan=plan, page_size=128, out_dtype=torch.float32, ape="zero")
    spec.update(overrides)
    return spec


def _one_shot(lengths: tuple) -> tuple:
    return tuple((n,) for n in lengths)


class CompressorPrefillReduction(OpWrapper):
    """Pool each complete window of `R = compress_ratio` tokens that gains a new
    token in this call into one output row, and store every new token's
    (kv, score + ape) in the paged compressor state.

    For context sequence `b < batch_size`, with `sp = start_pos[b]` and
    `kv_len = kv_lens[b]`, the new positions `[sp, kv_len)` are rows
    `cu_seq_lens[b] ..` of `kv_score` (`[kv || score]`, `S` wide each)::

        s'[p] = score[p] + ape[p % R]
        # every window k in [sp // R, kv_len // R)
        output[cu_kv_comp[b] + k - sp // R, d] = sum_p kv[p, d] * softmax_p(s'[p, d])
        paged_kv   [block_table_kv   [b, p // page_size], p % page_size] = kv[p]     # p in [sp, kv_len)
        paged_score[block_table_score[b, p // page_size], p % page_size] = s'[p]

    New positions are pooled straight from `kv_score`; only a position below
    `sp` (the first window, when `sp % R != 0`) is read back from the pools,
    whose score already holds `+ ape`. A trailing incomplete window is only
    stored. bf16 `output` gets the round-to-nearest-even of the fp32 value. A
    zero-row request (`kv_len == sp`) gets no row and touches nothing.

    Fusion boundary: the window softmax, the weighted sum and the state writes
    are inside the call. The caller owns the `wkv || wgate` projection that
    makes `kv_score`, the RMSNorm, RoPE, quantization and the compressed-KV
    cache write, and decode-time pooling (`compressor_paged_kv_compress`).

    Out-variant: writes `output`, `paged_kv` and `paged_score` in place and
    returns `(output, paged_kv, paged_score)` in that order (the same tensor
    twice when the two pools are one tensor, as in DeepSeek-V4.1).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"one-shot page={page} L={'-'.join(map(str, lengths))}: start_pos 0",
                spec=_spec(_one_shot(lengths), page_size=page),
            )
            for page in (128, 256)
            for lengths in _ONE_SHOT
        )
        + tuple(
            Cell(
                why=f"chunked page={page}: 5 calls, mixed parities, pending token carried",
                spec=_spec(_CHUNKED, page_size=page),
            )
            for page in (128, 256)
        )
        + (
            Cell(
                why="bf16 output: RNE of the fp32 pooled value, one-shot",
                spec=_spec(_one_shot((5, 128, 255, 256, 1000, 8191)), out_dtype=torch.bfloat16),
            ),
            Cell(
                why="nonzero ape, chunked: baked into the stored score",
                spec=_spec(_CHUNKED, ape="random"),
            ),
        )
    )

    note = """
    Cells. DeepSeek-V4.1's ratio-2 compressor of the KV-source layers 2/8/14
    at prefill: `compress_ratio` 2, `head_dim` 512, fp32 `kv_score`, fp32
    state (kv and score pools on randomly permuted, distinct pages, a trap page
    behind every unused table column), zero `ape` (one cell nonzero), fp32 or
    bf16 `output`, page sizes 128 and 256. Kv ~ N(0, 2^2), score ~ N(0, 4^2),
    plus one exact-tie channel and one whose pair gap of 240 underflows `exp`.

    Metadata consumed: none. The target passes the step's whole fp32
    `[T, 1024]` `kv_score` (context rows first; the generation rows after them
    are not read), the pools `get_buffers(owner, COMPRESSOR_KV / SCORE)` (one
    fp32 `[pages, 128, 512]` tensor for both roles and owners 2, 8, 14), the
    tables `sliding_block_tables[layer_offsets[owner], COMPRESSOR_KV / SCORE][:nc]`,
    a zero-filled `[num_total_compressed_tokens[2], 512]` output (bf16; rows from
    `cu_new_comp_kv_cuda[2][nc]` on are the decode op's and are not written),
    `kv_lens_cuda_runtime[:nc]`, `cached_token_lens_cuda[:nc]`,
    `cu_seq_lens_cuda[:nc + 1]`, `cu_new_comp_kv_cuda[2][:nc + 1]`,
    `num_contexts`, `max_ctx_compressed_tokens[2]` and `tokens_per_block`.
    Under encoder SWA bounded replay `kv_score` is the projection of the compute
    rows, `start_pos` is `v41_compute_start_pos_cuda[:nc]` and `cu_seq_lens`
    `v41_compute_cu_seq_lens_cuda[:nc + 1]`; a first chunk that holds only its
    replay rows is then a zero-row request.

    Preconditions: `kv_len - sp` is the sequence's row count in `kv_score`
    (the op never reads `cu_seq_lens[b + 1]`); `cu_kv_comp` is the prefix sum
    of `kv_len // R - sp // R`; `max_outputs >= max_b(kv_len // R - sp // R)`
    -- it sizes the grid, and an undersized value silently drops the rows past
    it and the trailing pending token's state (not guarded); the tables map
    every page covering `[sp, kv_len)` and, for `sp % R != 0`, the page of
    `sp - 1`, which holds that token's kv and score + ape; both tables have
    the same width (`block_table_kv.size(1)` is the row stride); `ape` fp32
    `[R, S]`. Pages behind the window may alias: only `sp - 1` is read back.
    No table entry is validated. The launcher sends any `head_dim` other than
    512 to the 128 instantiation unchecked.

    Hazards (is_valid): an fp16 `kv_score` or `output` is read or written as
    bf16 bits, and a strided view is walked with the dense row stride, all
    without an error. The pools are not guarded.

    Rejected by the op: `compress_ratio` other than 2, 4 or 128 (V4.1's
    ratio-1 layer cannot route here), bf16 `ape`, int64 `kv_lens`, pools of
    different element types.

    Precision: built with `--use_fast_math`, an online fp32 softmax that
    divides once, so not bitwise torch's softmax-then-sum.

    The test also certifies, on the engine's own pools and managers: every
    row count 1..16384, the target's call (dumped steps, and steps on
    managers of the envelope below; shared pool), offset hazards and pools up
    to 247806 pages, the multi-request envelope (1..256 contexts, T 1..16384,
    `max_outputs` 0..8192, tables 132 .. 8196 columns, kv lengths to
    1,048,590), every dumped prefill step, and zero-row requests anywhere in
    the batch.

    Not certified: CUDA-graph capture (prefill runs eagerly); bf16 `kv_score`
    or state, `head_dim` 128, ratios 4 and 128, other page sizes; more than
    16384 tokens or 256 contexts per call; kv lengths past 1,048,590 or tables
    wider than 8196 columns; pools above 247806 pages; page sharing under block
    reuse; a zero-row request at `sp` 0 or at an odd `sp` other than 7; a call
    whose `kv_score` has no row at all; performance.
    """

    def raw_call(
        self,
        kv_score: torch.Tensor,
        ape: torch.Tensor,
        paged_kv: torch.Tensor,
        paged_score: torch.Tensor,
        block_table_kv: torch.Tensor,
        block_table_score: torch.Tensor,
        output: torch.Tensor,
        kv_lens: torch.Tensor,
        start_pos: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        cu_kv_comp: torch.Tensor,
        batch_size: int,
        page_size: int,
        head_dim: int,
        compress_ratio: int,
        max_outputs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.compressor_prefill_reduction(
            kv_score,
            ape,
            paged_kv,
            paged_score,
            block_table_kv,
            block_table_score,
            output,
            kv_lens,
            start_pos,
            cu_seq_lens,
            cu_kv_comp,
            batch_size,
            page_size,
            head_dim,
            compress_ratio,
            max_outputs,
        )
        return output, paged_kv, paged_score

    def is_valid(
        self,
        kv_score: torch.Tensor,
        ape: torch.Tensor,
        paged_kv: torch.Tensor,
        paged_score: torch.Tensor,
        block_table_kv: torch.Tensor,
        block_table_score: torch.Tensor,
        output: torch.Tensor,
        kv_lens: torch.Tensor,
        start_pos: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        cu_kv_comp: torch.Tensor,
        batch_size: int,
        page_size: int,
        head_dim: int,
        compress_ratio: int,
        max_outputs: int,
    ) -> None:
        # These four are read through raw pointers sized only by element width and
        # by an assumed dense row stride: an fp16 tensor is read/written as bf16 bits
        # and a strided view is walked as if contiguous, with no error either way.
        assert kv_score.dtype in _ELEMENT_TYPES and output.dtype in _ELEMENT_TYPES, (
            f"kv_score/output must be float32 or bfloat16, got {kv_score.dtype}/{output.dtype}"
        )
        assert kv_score.is_contiguous() and output.is_contiguous(), (
            "kv_score and output must be contiguous (the op assumes dense rows)"
        )

    def reference(
        self,
        kv_score: torch.Tensor,
        ape: torch.Tensor,
        paged_kv: torch.Tensor,
        paged_score: torch.Tensor,
        block_table_kv: torch.Tensor,
        block_table_score: torch.Tensor,
        output: torch.Tensor,
        kv_lens: torch.Tensor,
        start_pos: torch.Tensor,
        cu_seq_lens: torch.Tensor,
        cu_kv_comp: torch.Tensor,
        batch_size: int,
        page_size: int,
        head_dim: int,
        compress_ratio: int,
        max_outputs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The whole of `(output, paged_kv, paged_score)` after the call, from
        their contents before it.

        A documented exception, like `load_paged_kv_cache_for_mla.reference`'s
        `stored`: when `sp % R != 0` the first pooled row is a function of the
        token(s) an earlier call parked below `sp`, which only the pools hold.
        This reads them from `paged_kv` / `paged_score` as passed, so it must
        see them as they were before the call -- call it before the op, which
        overwrites the chunk's slots in place. Nothing is mutated: the expected
        tensors are copies (one copy when the two pools are one tensor).

        The checkpoint's `Compressor.forward` pooling, `sum(kv * softmax_over_the_window(score + ape))`
        per channel in fp32, rounded once to `output`'s dtype; new positions
        from `kv_score`, parked ones from the pools. `head_dim` is the state
        width (the non-overlap ratios 2 and 128); the ratio-4 overlap mode is not
        modeled. `max_outputs` only sizes the launch grid and is ignored: this
        is the result under its precondition.
        """
        if compress_ratio == 4:
            raise NotImplementedError("the ratio-4 overlap mode is not modeled")
        R, S = compress_ratio, head_dim
        dev = kv_score.device
        old_kv, old_score = paged_kv.view(-1, S), paged_score.view(-1, S)
        exp_kv = paged_kv.clone()
        shared = paged_score.data_ptr() == paged_kv.data_ptr()
        exp_score = exp_kv if shared else paged_score.clone()
        exp_out = output.clone()
        kv_rows, score_rows = exp_kv.view(-1, S), exp_score.view(-1, S)
        ape32 = ape.float().reshape(R, S)
        lens = kv_lens[:batch_size].tolist()
        starts = start_pos[:batch_size].tolist()
        first_row = cu_seq_lens[:batch_size].tolist()
        first_out = cu_kv_comp[:batch_size].tolist()

        def slots(table: torch.Tensor, b: int, pos: torch.Tensor) -> torch.Tensor:
            return table[b].long()[pos // page_size] * page_size + pos % page_size

        for b in range(batch_size):
            sp, kv_len = starts[b], lens[b]
            if kv_len <= sp:
                continue
            pos = torch.arange(sp, kv_len, device=dev)
            rows = kv_score[first_row[b] : first_row[b] + kv_len - sp].float()
            kv_new, score_new = rows[:, :S], rows[:, S:] + ape32[pos % R]
            w0, w1 = sp // R, kv_len // R
            if w1 > w0:
                parked = torch.arange(w0 * R, sp, device=dev)
                k = torch.cat([old_kv[slots(block_table_kv, b, parked)].float(), kv_new])
                s = torch.cat([old_score[slots(block_table_score, b, parked)].float(), score_new])
                n = (w1 - w0) * R
                k, s = k[:n].unflatten(0, (w1 - w0, R)), s[:n].unflatten(0, (w1 - w0, R))
                pooled = (k * s.softmax(dim=1)).sum(dim=1)
                exp_out[first_out[b] : first_out[b] + w1 - w0] = pooled.to(output.dtype)
            kv_rows[slots(block_table_kv, b, pos)] = kv_new.to(kv_rows.dtype)
            score_rows[slots(block_table_score, b, pos)] = score_new.to(score_rows.dtype)
        return exp_out, exp_kv, exp_score

    def compare(
        self,
        out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Pools bitwise; fp32 rows at torch's default fp32 band; bf16 rows at
        >= 0.999 bit-identical.

        The state rows are copies (`kv`, and `score + ape` in fp32), so any
        differing pool byte is a wrong slot or a write the op must not make.

        The kernel is built with `--use_fast_math` and runs an online softmax
        that divides once at the end, so it is not bit-identical to torch's
        softmax-then-sum: over the five-sequence chunked batch with the cells'
        inputs the largest abs error measured on sm_103 is 1.907e-06, inside the
        default fp32 band (rtol 1.3e-6, atol 1e-5). A bf16 element can then
        differ only where the fp32 value straddles a bf16 rounding midpoint:
        0.999985 of 2516992 elements measured bit-identical; 0.999 is the gate,
        which a bf16-operand control (0.719) misses. Rows the op must leave
        alone are compared bit for bit (the test pre-fills them with NaN, so the
        bf16 fraction is over the finite, written rows).
        """
        (o, kv, sc), (ro, rkv, rsc) = out, ref
        for name, got, exp in (("paged_kv", kv, rkv), ("paged_score", sc, rsc)):
            assert got.dtype == exp.dtype and got.shape == exp.shape, (name, got.dtype, exp.dtype)
            bad = (got.view(torch.uint8) != exp.view(torch.uint8)).sum().item()
            assert bad == 0, f"{name}: {bad} of {got.numel() * got.element_size()} bytes differ"
        assert o.dtype == ro.dtype, (o.dtype, ro.dtype)
        assert o.shape == ro.shape, (o.shape, ro.shape)
        if o.dtype == torch.float32:
            torch.testing.assert_close(o, ro, equal_nan=True)
            return
        written = torch.isfinite(ro)
        same = o.view(torch.int16) == ro.view(torch.int16)
        assert same[~written].all(), "output written outside the rows the op owns"
        torch.testing.assert_close(o, ro, equal_nan=True)
        if written.any():
            exact = same[written].float().mean().item()
            assert exact >= 0.999, f"bf16 output bit-exact fraction {exact}"


compressor_prefill_reduction = CompressorPrefillReduction()
