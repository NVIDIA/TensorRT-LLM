# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-side KV compressor: every generation request appends `next_n` tokens
to its paged compressor state, and each window of `compress_ratio` tokens that
the step completes is pooled (per-channel softmax over the window) into one
output row.

The op is `torch.ops.trtllm.compressor_paged_kv_compress`, registered in
`cpp/tensorrt_llm/thop/compressorOp.cpp`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_ELEMENT_TYPES = (torch.float32, torch.bfloat16)

#: Histories (tokens already stored) that put a step's first new token at both
#: parities and on both sides of a 128- and a 256-token page boundary.
_CACHED = (1, 2, 127, 128, 255, 256, 1001, 8191)


def _spec(next_n: int, steps: int, **overrides) -> dict:
    spec = dict(
        cached=_CACHED,
        steps=steps,
        next_n=next_n,
        page_size=128,
        out_dtype=torch.float32,
        shared_pool=True,
        poison_history=False,
        lead_rows=0,
        lead_out=0,
        ape="zero",
    )
    spec.update(overrides)
    return spec


class CompressorPagedKvCompress(OpWrapper):
    """Store each generation request's `next_n` new tokens in the paged
    compressor state, then pool every window of `R = compress_ratio` positions
    the step completes into one output row.

    For request `b < batch_size`, with `kv_len = kv_lens[b]` its length after
    the step and `sp = kv_len - next_n`, the new positions `[sp, kv_len)` come
    from rows `cu_seq_lens[b] ..` of `kv_score` (`[kv || score]`, `S` wide each)::

        paged_kv   [block_table_kv   [b, p // page_size], p % page_size] = kv[p]
        paged_score[block_table_score[b, p // page_size], p % page_size] = score[p] + ape[p % R]

        # every window k in [sp // R, kv_len // R), both positions read back from the pools
        output[cu_kv_comp[b] + k - sp // R, d] = sum_p paged_kv[p, d] * softmax_p(paged_score[p, d])

    so a window that started in an earlier step (odd `sp` at `R = 2`) reads the
    token that step parked at `sp - 1`. bf16 `output` gets the round-to-nearest
    -even of the fp32 pooled value.

    Fusion boundary: the state writes, the per-channel window softmax and the
    weighted sum are inside the call. The caller owns the `wkv || wgate`
    projection that makes `kv_score`, the RMSNorm, RoPE, quantization and the
    compressed-KV cache write, and prefill pooling (`compressor_prefill_reduction`,
    which leaves the state this op reads).

    Out-variant: writes `output`, `paged_kv` and `paged_score` in place and
    returns `(output, paged_kv, paged_score)` in that order (the same tensor
    twice when the two pools are one tensor, as in DeepSeek-V4.1).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"decode next_n={n} page={page}: 5 steps over 8 histories",
                spec=_spec(n, 5, page_size=page),
            )
            for n in range(1, 9)
            for page in (128, 256)
        )
        + tuple(
            Cell(
                why=f"poisoned history next_n={n}: only start-1 is read back",
                spec=_spec(n, 4, poison_history=True),
            )
            for n in (1, 6)
        )
        + (
            Cell(
                why="separate kv and score pools: two tensors, next_n 6",
                spec=_spec(6, 4, shared_pool=False),
            ),
            Cell(
                why="context lead 1537 rows / 700 outputs: mixed-step slices",
                spec=_spec(6, 3, lead_rows=1537, lead_out=700),
            ),
        )
        + tuple(
            Cell(
                why=f"bf16 output next_n={n}: RNE of the fp32 pooled value",
                spec=_spec(n, 6, out_dtype=torch.bfloat16),
            )
            for n in (1, 6)
        )
        + tuple(
            Cell(
                why=f"nonzero ape next_n={n}: baked into the stored score",
                spec=_spec(n, 4, ape="random"),
            )
            for n in (1, 6)
        )
    )

    note = """
    Cells. DeepSeek-V4.1's ratio-2 compressor of the KV-source layers 2/8/14:
    `compress_ratio` 2, `head_dim` 512, fp32 `kv_score`, fp32 state with kv and
    score on distinct pages of one tensor, zero `ape`, fp32 or bf16 `output`;
    plain decode (`next_n` 1), DSpark verify (6), and 1..8 in all, at page
    sizes 128 and 256. Every cell runs several steps, so later steps read the
    token the op itself parked. Kv ~ N(0, 2^2), score ~ N(0, 4^2), plus one
    exact-tie channel and one channel whose pair gap of 240 underflows `exp`.

    Metadata consumed: none. The target passes, per owner layer, the pools
    `get_buffers(owner, COMPRESSOR_KV / COMPRESSOR_SCORE)` (one fp32
    `[pages, 128, 512]` base for both roles), the tables
    `sliding_block_tables[layer_offsets[owner], COMPRESSOR_KV / SCORE][nc:nc + B]`,
    `kv_lens_cuda_runtime[nc:nc + B]`, `cu_seq_lens_cuda[nc:nc + B + 1]`,
    `cu_new_comp_kv_cuda[2][nc:nc + B + 1]`, `num_gen_tokens_per_seq` and
    `tokens_per_block`, with the step's whole `kv_score` and the whole
    `num_total_compressed_tokens[2]`-row output (context rows ahead are neither
    read nor written).

    Preconditions: every request feeds exactly `next_n` tokens (the op never
    reads `cu_seq_lens[b + 1]`); `cu_kv_comp[b + 1] - cu_kv_comp[b]` is
    `kv_len // R - sp // R`; for an odd `sp` the state at `sp - 1` holds that
    token's kv and score + ape (left by the prefill op or an earlier call); the
    tables map every page covering `[sp - sp % R, kv_len)`, kv and score to
    different pages, and have the same width (`block_table_kv.size(1)` is the
    row stride of both); `page_size` even; `ape` fp32 `[R, S]`. No table entry
    is validated.

    Hazards (is_valid): an fp16 `kv_score` or `output` is read or written as
    bf16 bits, and a strided view is walked with the dense row stride, all
    without an error. The pools are not guarded.

    Rejected by the op: `compress_ratio` other than 2, 4 or 128 (V4.1's
    ratio-1 layer, which pools nothing, cannot route here), `next_n` 0, a
    `head_dim` with no instantiation, bf16 `ape`, int64 `kv_lens`, pools of
    different element types. The op's optional `new_tokens_per_seq` (ragged
    verify) is not passed by this entry.

    Precision: built with `--use_fast_math`, an online fp32 softmax that
    divides once, so not bitwise torch's softmax-then-sum. DSpark: a verify
    step emits rows for every pair its tokens complete, drafts included; the
    next step starting at `sp + 1 + accepted` rewrites rejected positions and,
    when it starts odd, re-emits the pair `(sp' - 1, sp')`, so the caller's
    compressed-cache write must overwrite, not append.

    No allocation, no host synchronization: graph-capturable. The test also
    certifies, on the engine's own pools and managers: every batch 1..256 at
    table widths 132 .. 8196, every context lead up to a 16384-token step, the
    capture batch set at every width, state pools up to 247806 pages (element
    offsets past 2^33), positions up to 1,049,087, and the DSpark tail at
    max_seq_len 16384 (positions up to 16397 on 132-column tables).

    Not certified: bf16 `kv_score` or state, `head_dim` 128, ratios 4 and 128,
    other page sizes; `next_n` above 8 (under capture, other than 1 and 6); more
    than 256 requests; tables wider than 8196 columns; pools above 247806
    pages; unmapped or negative table entries; block reuse and pages released
    behind the window on the real manager; performance.
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
        cu_seq_lens: torch.Tensor,
        cu_kv_comp: torch.Tensor,
        batch_size: int,
        page_size: int,
        head_dim: int,
        compress_ratio: int,
        next_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.ops.trtllm.compressor_paged_kv_compress(
            kv_score,
            ape,
            paged_kv,
            paged_score,
            block_table_kv,
            block_table_score,
            output,
            kv_lens,
            cu_seq_lens,
            cu_kv_comp,
            batch_size,
            page_size,
            head_dim,
            compress_ratio,
            next_n,
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
        cu_seq_lens: torch.Tensor,
        cu_kv_comp: torch.Tensor,
        batch_size: int,
        page_size: int,
        head_dim: int,
        compress_ratio: int,
        next_n: int,
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
        cu_seq_lens: torch.Tensor,
        cu_kv_comp: torch.Tensor,
        batch_size: int,
        page_size: int,
        head_dim: int,
        compress_ratio: int,
        next_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The whole of `(output, paged_kv, paged_score)` after the call, from
        their contents before it.

        A documented exception, like `load_paged_kv_cache_for_mla.reference`'s
        `stored`: for an odd start the pooled row is a function of the token an
        earlier call parked at `sp - 1`, which only the pools hold. This reads it
        from `paged_kv` / `paged_score` as passed, so it must see them as they
        were before the call -- call it before the op, which overwrites the
        step's slots in place. Nothing is mutated: the expected tensors are
        copies (one copy when the two pools are one tensor).

        The checkpoint's decode rule (`Compressor.forward` at `start_pos > 0`)
        applied to the step's tokens in order: store kv and `score + ape[p % R]`
        in the state's element type, then pool each completed window from the
        stored values, `sum(kv * softmax_over_the_window(score))` per channel in
        fp32, rounded once to `output`'s dtype. `head_dim` is the state width
        (the non-overlap ratios 2 and 128); the ratio-4 overlap mode is not
        modeled.
        """
        if compress_ratio == 4:
            raise NotImplementedError("the ratio-4 overlap mode is not modeled")
        R, S = compress_ratio, head_dim
        dev = kv_score.device
        exp_kv = paged_kv.clone()
        shared = paged_score.data_ptr() == paged_kv.data_ptr()
        exp_score = exp_kv if shared else paged_score.clone()
        exp_out = output.clone()
        kv_rows, score_rows = exp_kv.view(-1, S), exp_score.view(-1, S)
        ape32 = ape.float().reshape(R, S)
        lens = kv_lens[:batch_size].tolist()
        first_row = cu_seq_lens[:batch_size].tolist()
        first_out = cu_kv_comp[:batch_size].tolist()

        def slots(table: torch.Tensor, b: int, pos: torch.Tensor) -> torch.Tensor:
            return table[b].long()[pos // page_size] * page_size + pos % page_size

        for b in range(batch_size):
            kv_len = lens[b]
            sp = kv_len - next_n
            pos = torch.arange(sp, kv_len, device=dev)
            new = kv_score[first_row[b] : first_row[b] + next_n].float()
            kv_rows[slots(block_table_kv, b, pos)] = new[:, :S].to(kv_rows.dtype)
            score_rows[slots(block_table_score, b, pos)] = (new[:, S:] + ape32[pos % R]).to(
                score_rows.dtype
            )
            w0, w1 = sp // R, kv_len // R
            if w1 == w0:
                continue
            p = torch.arange(w0 * R, w1 * R, device=dev)
            k = kv_rows[slots(block_table_kv, b, p)].float().unflatten(0, (w1 - w0, R))
            s = score_rows[slots(block_table_score, b, p)].float().unflatten(0, (w1 - w0, R))
            pooled = (k * s.softmax(dim=1)).sum(dim=1)
            exp_out[first_out[b] : first_out[b] + w1 - w0] = pooled.to(output.dtype)
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
        softmax-then-sum: over the 168 rows of next_n 1 and 6 with the cells'
        inputs the largest abs error measured on sm_103 is 1.431e-06, inside
        the default fp32 band (rtol 1.3e-6, atol 1e-5). A bf16 element can then
        differ only where the fp32 value straddles a bf16 rounding midpoint:
        1.000000 of 86016 elements measured bit-identical; 0.999 is the gate,
        which a bf16-operand control (0.717) misses. Rows the op must leave
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


compressor_paged_kv_compress = CompressorPagedKvCompress()
