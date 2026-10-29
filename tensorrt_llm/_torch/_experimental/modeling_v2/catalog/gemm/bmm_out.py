# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul writing into a caller-owned output, via the trtllm bmm_out op.

Mistral Large 4 calls it on its absorbed-MLA views; its cells are the ones built by
`_ml4` (TEP4, 16 heads per rank), on `_ML4_64` (DEP4, all 64 heads on every rank),
on `_ML4_16_FQ64` (16 heads per rank into a 64-head buffer, on the fp8_ds_mla latent
KV) and on `_ML4_8` (TEP8 and TP8, 8 heads per rank; the `_ML4_8_FQ64` cells are
their K-B BMM into heads 0..7 of the 64-head fused query FlashMLA reads on the
fp8_ds_mla latent KV).
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

# Mistral Large 4's two absorbed BMMs per layer, batch = the H heads a rank holds:
# this rank's 16 at TEP4 and TP4, all 64 at DEP4 (attention data parallel),
# this rank's 8 at TEP8 and TP8. `a_row` / `out_row` are the token-major
# buffers' per-head widths.
_ML4_CALLS = {
    # K-B on the 54 RoPE layers: the 64-wide nope half of q [T, H, 128]
    # (nope | rope) into the first 512 columns of the fused query [T, H, 576].
    "k_b_rope": dict(k=64, n=512, a_row=128, out_row=576, b_layout="kn"),
    # K-B on the 18 NoPE layers: all 128 columns of q are nope.
    "k_b_nope": dict(k=128, n=512, a_row=128, out_row=576, b_layout="kn"),
    # V-B: the latent attention output [T, H, 512] into the [T, H, 128] view
    # of the [T, H x 128] attention output; `b` is the transposed view of the
    # V-B rows of the regrouped `wkv_b`, as `weights.py` lays it out.
    "v_b": dict(k=512, n=128, a_row=512, out_row=128, b_layout="nk_t"),
}


#: max_num_tokens of the two Mistral Large 4 targets: the FP8 target serves 9216,
#: the NVFP4 target 32768. Neither chunks prefill, so one step carries up to
#: that many rows through both BMMs.
_FP8_ROWS = 9216
_NVFP4_ROWS = 32768


def _ml4(call: str, **overrides) -> dict:
    """One Mistral Large 4 call at the FP8 target's max_num_tokens 9216, unless `m`
    is overridden."""
    return dict(dict(batch=16, m=_FP8_ROWS, dtype=torch.bfloat16, **_ML4_CALLS[call]), **overrides)


#: The NVFP4 checkpoint at DEP4: batch = all 64 heads, over the rank's own
#: tokens, up to the same max_num_tokens 32768 per rank. Each cell spreads one
#: `_ML4_CALLS` entry over this.
_ML4_64 = dict(batch=64, m=_NVFP4_ROWS, dtype=torch.bfloat16)

#: The NVFP4 checkpoint at TEP8 and TP8 (the two shard attention alike): batch = this
#: rank's 8 of the 64 heads, over every token of the step (attention DP off), up to
#: max_num_tokens 32768.
#: Each cell spreads one `_ML4_CALLS` entry over this.
_ML4_8 = dict(batch=8, m=_NVFP4_ROWS, dtype=torch.bfloat16)

#: The same at TEP8 and TP8 on the fp8_ds_mla latent KV: FlashMLA's sparse kernels take 64
#: query heads, so the K-B BMM writes the rank's 8 heads into heads 0..7 of a `[T, 64, 576]`
#: fused query (`out_heads`; the token stride is 64 x 576, the head stride 576) whose heads
#: 8..63 stay zero.
_ML4_8_FQ64 = dict(_ML4_8, out_heads=64)

#: The NVFP4 checkpoint at 16 heads per rank (TEP4, TP4) on the fp8_ds_mla latent KV: FlashMLA's
#: sparse kernels take 64 query heads, so the K-B BMM writes the rank's 16 heads into heads 0..15
#: of a `[T, 64, 576]` fused query (`out_heads`; the token stride is 64 x 576 = 36864, the head
#: stride 576) whose heads 16..63 stay zero for the step. Chunked prefill keeps a step's rows at
#: max_num_tokens 32768 or fewer.
_ML4_16_FQ64 = dict(batch=16, m=_NVFP4_ROWS, dtype=torch.bfloat16, out_heads=64)


class BmmOut(OpWrapper):
    """`out[i] = a[i] @ b[i]` for every batch index, written into `out`, which
    is returned.

    The point of the op over `torch.bmm` is the caller-owned `out`: every
    shipped call site passes a transposed view of a buffer the rest of the
    forward already holds, so the product lands where it is needed with no
    copy. That is also the hazard -- see `is_valid`.

    Fusion boundary: the product only. The weight regroup and transposition,
    and the buffers `a` and `out` are views of, are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # A `b_layout` of "nk_t" is the transposed view the targets pass for V-B;
    # the other cells' `b` is contiguous, which the targets pass for K-B.
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"DSpark absorbed {call}, {heads} heads, graph rows 1..64",
                spec=dict(
                    batch=heads,
                    m=64,
                    dtype=torch.bfloat16,
                    **_ML4_CALLS[call],
                    graph_m=tuple(range(1, 65)),
                ),
            )
            for heads in (8, 16, 64)
            for call in ("k_b_rope", "k_b_nope", "v_b")
        ),
        *(
            Cell(
                why=f"DSpark verify {call}, 8 into 64 heads, graph rows 1..64",
                spec=dict(_ML4_8_FQ64, **_ML4_CALLS[call], graph_m=tuple(range(1, 65))),
            )
            for call in ("k_b_rope", "k_b_nope")
        ),
        Cell(
            why="Mistral Large 4 K-B RoPE (k 64), every m 1..9216: each m's rows bitwise "
            "the leading rows of the 9216-row run, which is gated (~3 s)",
            spec=_ml4("k_b_rope", m_range=(1, 9216)),
        ),
        Cell(
            why="Mistral Large 4 K-B NoPE (k 128), every m 1..9216: each m's rows bitwise "
            "the leading rows of the 9216-row run, which is gated (~3 s)",
            spec=_ml4("k_b_nope", m_range=(1, 9216)),
        ),
        Cell(
            why="Mistral Large 4 V-B (transposed-view b), every m 1..9216: each m's rows "
            "bitwise the leading rows of the 9216-row run, which is gated (~1.5 s)",
            spec=_ml4("v_b", m_range=(1, 9216)),
        ),
        Cell(
            why="Mistral Large 4 K-B RoPE (k 64), CUDA graphs at the captured, padded "
            "decode batches 1/2/4/8/16: replays bitwise equal to eager",
            spec=_ml4("k_b_rope", graph_m=(1, 2, 4, 8, 16)),
        ),
        Cell(
            why="Mistral Large 4 K-B NoPE (k 128), CUDA graphs at the captured, padded "
            "decode batches 1/2/4/8/16: replays bitwise equal to eager",
            spec=_ml4("k_b_nope", graph_m=(1, 2, 4, 8, 16)),
        ),
        Cell(
            why="Mistral Large 4 V-B (transposed-view b), CUDA graphs at the captured, "
            "padded decode batches 1/2/4/8/16: replays bitwise equal to eager",
            spec=_ml4("v_b", graph_m=(1, 2, 4, 8, 16)),
        ),
        Cell(
            why="Mistral Large 4 K-B RoPE (k 64), a mixed step's 15 generation rows: "
            "rows 9201..9215 of 9216-row buffers, a nonzero storage offset",
            spec=_ml4("k_b_rope", tail_from=9201),
        ),
        Cell(
            why="Mistral Large 4 K-B NoPE (k 128), a mixed step's 15 generation rows: "
            "rows 9201..9215 of 9216-row buffers, a nonzero storage offset",
            spec=_ml4("k_b_nope", tail_from=9201),
        ),
        Cell(
            why="Mistral Large 4 V-B (transposed-view b), a mixed step's 15 generation "
            "rows: rows 9201..9215 of 9216-row buffers, a nonzero storage offset",
            spec=_ml4("v_b", tail_from=9201),
        ),
        # The NVFP4 target's envelope: max_num_tokens 32768 without chunked
        # prefill, so one context step passes both BMMs up to 32768 rows (one
        # request, or up to 16 packed), and the generation rows of a mixed step
        # sit behind up to 32753 context rows (max_batch_size 16). The captured
        # decode batches are 1/2/4/8/16 again, which the graph cells above hold.
        Cell(
            why="Mistral Large 4 32768 K-B RoPE (k 64), every m 1..32768, the NVFP4 "
            "target's max_num_tokens: each m's rows bitwise the leading rows of the "
            "32768-row run, which is gated (~7 s)",
            spec=_ml4("k_b_rope", m=_NVFP4_ROWS, m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="Mistral Large 4 32768 K-B NoPE (k 128), every m 1..32768, the NVFP4 "
            "target's max_num_tokens: each m's rows bitwise the leading rows of the "
            "32768-row run, which is gated (~7 s)",
            spec=_ml4("k_b_nope", m=_NVFP4_ROWS, m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="Mistral Large 4 32768 V-B (transposed-view b), every m 1..32768, the "
            "NVFP4 target's max_num_tokens: each m's rows bitwise the leading rows "
            "of the 32768-row run, which is gated (~4 s)",
            spec=_ml4("v_b", m=_NVFP4_ROWS, m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="Mistral Large 4 32768 K-B RoPE (k 64), a mixed step's 15 generation "
            "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=_ml4("k_b_rope", m=_NVFP4_ROWS, tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="Mistral Large 4 32768 K-B NoPE (k 128), a mixed step's 15 generation "
            "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=_ml4("k_b_nope", m=_NVFP4_ROWS, tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="Mistral Large 4 32768 V-B (transposed-view b), a mixed step's 15 "
            "generation rows: rows 32753..32767 of 32768-row buffers, a nonzero "
            "storage offset",
            spec=_ml4("v_b", m=_NVFP4_ROWS, tail_from=_NVFP4_ROWS - 15),
        ),
        # DEP4: the same three calls at batch 64, every rank holding all heads
        # and running only its own requests, so m is the rank's own token count:
        # up to 32768 in a context step (no chunked prefill), 1/2/4/8/16 in the
        # captured decode batches, and a mixed step's generation rows behind up
        # to 32753 context rows. Every m 1..32768 is four times the 16-head
        # sweep's bytes and past a cell's time, so those three cells are
        # `large`; pre-merge drives every m where cuBLAS's choice changes.
        Cell(
            why="dep4 64 heads K-B RoPE (k 64), every m 1..32768: each m's rows "
            "bitwise the leading rows of the 32768-row run, which is gated "
            "(large: ~27 s)",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS), large=True),
        ),
        Cell(
            why="dep4 64 heads K-B NoPE (k 128), every m 1..32768: each m's rows "
            "bitwise the leading rows of the 32768-row run, which is gated "
            "(large: ~27 s)",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS), large=True),
        ),
        Cell(
            why="dep4 64 heads V-B (transposed-view b), every m 1..32768: each m's "
            "rows bitwise the leading rows of the 32768-row run, which is gated "
            "(large: ~12 s)",
            spec=dict(_ML4_64, **_ML4_CALLS["v_b"], m_range=(1, _NVFP4_ROWS), large=True),
        ),
        Cell(
            why="dep4 64 heads K-B RoPE (k 64), edges of every 16-row window to "
            "32768 and every m 1..16, the only places its kernel and grid change: "
            "bitwise the leading rows of the gated 32768-row run (~4 s)",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
        ),
        Cell(
            why="dep4 64 heads K-B NoPE (k 128), edges of every 16-row window to "
            "32768 and every m 1..16, the only places its kernel and grid change: "
            "bitwise the leading rows of the gated 32768-row run (~4 s)",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
        ),
        Cell(
            why="dep4 64 heads V-B (transposed-view b), edges of every 16-row window "
            "to 32768 and every m 1..16, the only places its kernel and grid change: "
            "bitwise the leading rows of the gated 32768-row run (~2 s)",
            spec=dict(_ML4_64, **_ML4_CALLS["v_b"], m_window_edges=(1, _NVFP4_ROWS, 16)),
        ),
        Cell(
            why="dep4 64 heads K-B RoPE (k 64), CUDA graphs at the captured, padded "
            "decode batches 1/2/4/8/16: replays bitwise equal to eager",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], graph_m=(1, 2, 4, 8, 16)),
        ),
        Cell(
            why="dep4 64 heads K-B NoPE (k 128), CUDA graphs at the captured, padded "
            "decode batches 1/2/4/8/16: replays bitwise equal to eager",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], graph_m=(1, 2, 4, 8, 16)),
        ),
        Cell(
            why="dep4 64 heads V-B (transposed-view b), CUDA graphs at the captured, "
            "padded decode batches 1/2/4/8/16: replays bitwise equal to eager",
            spec=dict(_ML4_64, **_ML4_CALLS["v_b"], graph_m=(1, 2, 4, 8, 16)),
        ),
        Cell(
            why="dep4 64 heads K-B RoPE (k 64), a mixed step's 15 generation rows: "
            "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="dep4 64 heads K-B NoPE (k 128), a mixed step's 15 generation rows: "
            "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=dict(_ML4_64, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="dep4 64 heads V-B (transposed-view b), a mixed step's 15 generation "
            "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=dict(_ML4_64, **_ML4_CALLS["v_b"], tail_from=_NVFP4_ROWS - 15),
        ),
        # TEP8 and TP8 (the "tep8" cells are both targets' calls): the same three
        # calls at batch 8, every rank holding 8 of the 64 heads and running every
        # token of the step (attention DP off), so m is the step's token count: up
        # to 32768 in a context step (no chunked prefill), every batch 1..16 in a
        # captured decode step (the gates' padded 1/2/4/8/16 and the LLM API
        # default's unpadded batches), and a mixed step's generation rows behind
        # up to 32753 context rows. At half the 16-head bytes the every-m sweeps
        # fit a cell's time.
        Cell(
            why="tep8 8 heads K-B RoPE (k 64), every m 1..32768: each m's rows "
            "bitwise the leading rows of the 32768-row run, which is gated (~5 s)",
            spec=dict(_ML4_8, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="tep8 8 heads K-B NoPE (k 128), every m 1..32768: each m's rows "
            "bitwise the leading rows of the 32768-row run, which is gated (~5 s)",
            spec=dict(_ML4_8, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="tep8 8 heads V-B (transposed-view b), every m 1..32768: each m's "
            "rows bitwise the leading rows of the 32768-row run, which is gated (~4 s)",
            spec=dict(_ML4_8, **_ML4_CALLS["v_b"], m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="tep8 8 heads K-B RoPE (k 64), CUDA graphs at every decode batch "
            "1..16, padded and unpadded: replays bitwise equal to eager",
            spec=dict(_ML4_8, **_ML4_CALLS["k_b_rope"], graph_m=tuple(range(1, 17))),
        ),
        Cell(
            why="tep8 8 heads K-B NoPE (k 128), CUDA graphs at every decode batch "
            "1..16, padded and unpadded: replays bitwise equal to eager",
            spec=dict(_ML4_8, **_ML4_CALLS["k_b_nope"], graph_m=tuple(range(1, 17))),
        ),
        Cell(
            why="tep8 8 heads V-B (transposed-view b), CUDA graphs at every decode "
            "batch 1..16, padded and unpadded: replays bitwise equal to eager",
            spec=dict(_ML4_8, **_ML4_CALLS["v_b"], graph_m=tuple(range(1, 17))),
        ),
        Cell(
            why="tep8 8 heads K-B RoPE (k 64), a mixed step's 15 generation rows: "
            "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=dict(_ML4_8, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="tep8 8 heads K-B NoPE (k 128), a mixed step's 15 generation rows: "
            "rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=dict(_ML4_8, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="tep8 8 heads V-B (transposed-view b), a mixed step's 15 generation "
            "rows: rows 32753..32767 of 32768-row buffers, a nonzero storage offset",
            spec=dict(_ML4_8, **_ML4_CALLS["v_b"], tail_from=_NVFP4_ROWS - 15),
        ),
        # TEP8 and TP8 on the fp8_ds_mla latent KV (PrefillTarget._attention and
        # DecodeTarget._attention, one call over every row of the step): the two K-B
        # calls into heads 0..7 of the 64-head fused query, so the out token stride
        # is 36864 where the `_ML4_8` cells' is 4608 (V-B reads the compact latent
        # output, as above). Row counts at both edges of every 16-row window (a
        # 64-head row is eight times the 8-head one, so a sweep resetting whole rows
        # at every m would not fit a cell's time), CUDA graphs at every decode batch
        # 1..16 and a mixed step's tail:
        Cell(
            why="tep8 K-B RoPE (k 64) into heads 0..7 of the 64-head fused query, m 1..16 "
            "and both edges of every 16-row window to 32768: each m's rows bitwise the "
            "leading rows of the 32768-row run, which is gated; heads 8..63 untouched",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
        ),
        Cell(
            why="tep8 K-B NoPE (k 128) into heads 0..7 of the 64-head fused query, m 1..16 "
            "and both edges of every 16-row window to 32768: each m's rows bitwise the "
            "leading rows of the 32768-row run, which is gated; heads 8..63 untouched",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], m_window_edges=(1, _NVFP4_ROWS, 16)),
        ),
        Cell(
            why="tep8 K-B RoPE (k 64) into the 64-head fused query, CUDA graphs at every "
            "decode batch 1..16: replays bitwise equal to eager, heads 8..63 untouched",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], graph_m=tuple(range(1, 17))),
        ),
        Cell(
            why="tep8 K-B NoPE (k 128) into the 64-head fused query, CUDA graphs at every "
            "decode batch 1..16: replays bitwise equal to eager, heads 8..63 untouched",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], graph_m=tuple(range(1, 17))),
        ),
        Cell(
            why="tep8 K-B RoPE (k 64) into the 64-head fused query, a mixed step's 15 "
            "generation rows: rows 32753..32767 of 32768-row buffers",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="tep8 K-B NoPE (k 128) into the 64-head fused query, a mixed step's 15 "
            "generation rows: rows 32753..32767 of 32768-row buffers",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
        ),
        # ... and every row count 1..32768: at max_seq_len 1048576 with chunked
        # prefill (max_num_tokens 32768) a step is any mix of chunk rows -- a
        # request's last chunk is its prompt length mod 32768 -- and up to 16 decode
        # rows, so every m to max_num_tokens occurs. The sweep resets and compares
        # only the call's 8 heads of each row, which keeps it within a cell's time.
        Cell(
            why="tep8 fused q64 K-B RoPE (k 64), heads 0..7 of [T, 64, 576], every m "
            "1..32768: each m's rows bitwise the leading rows of the 32768-row run, "
            "which is gated; heads 8..63 never written (~5 s)",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="tep8 fused q64 K-B NoPE (k 128), heads 0..7 of [T, 64, 576], every m "
            "1..32768: each m's rows bitwise the leading rows of the 32768-row run, "
            "which is gated; heads 8..63 never written (~5 s)",
            spec=dict(_ML4_8_FQ64, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS)),
        ),
        # TEP4 and TP4 on the fp8_ds_mla latent KV (PrefillTarget._attention and
        # DecodeTarget._attention, one call over every row of the step): the two K-B
        # calls write the rank's 16 heads into heads 0..15 of the 64-head fused query,
        # so the out token stride is 36864 where the `_ml4` cells' is 9216. V-B reads
        # the compact `[T, 16, 512]` latent output, which the `_ml4` V-B cells above
        # hold. m is a step's rows: every count 1..32768 eager (chunked prefill keeps
        # a step at max_num_tokens), the padded decode batches 1/2/4/8/16 captured
        # (the LLM API's default list reaches every batch 1..16), and a mixed step's
        # generation rows at the tail.
        Cell(
            why="fused q64 K-B RoPE (k 64), 16 heads into heads 0..15 of [T, 64, 576], "
            "every m 1..32768: each m's rows bitwise the leading rows of the 32768-row "
            "run, which is gated; heads 16..63 never written (~9 s)",
            spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_rope"], m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="fused q64 K-B NoPE (k 128), 16 heads into heads 0..15 of [T, 64, 576], "
            "every m 1..32768: each m's rows bitwise the leading rows of the 32768-row "
            "run, which is gated; heads 16..63 never written (~9 s)",
            spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_nope"], m_range=(1, _NVFP4_ROWS)),
        ),
        Cell(
            why="fused q64 K-B RoPE (k 64), 16 heads, CUDA graphs at every decode batch "
            "1..16: replays bitwise equal to eager, heads 16..63 never written",
            spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_rope"], graph_m=tuple(range(1, 17))),
        ),
        Cell(
            why="fused q64 K-B NoPE (k 128), 16 heads, CUDA graphs at every decode batch "
            "1..16: replays bitwise equal to eager, heads 16..63 never written",
            spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_nope"], graph_m=tuple(range(1, 17))),
        ),
        Cell(
            why="fused q64 K-B RoPE (k 64), 16 heads, a mixed step's 15 generation rows: "
            "rows 32753..32767 of 32768-row buffers, heads 16..63 never written",
            spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_rope"], tail_from=_NVFP4_ROWS - 15),
        ),
        Cell(
            why="fused q64 K-B NoPE (k 128), 16 heads, a mixed step's 15 generation rows: "
            "rows 32753..32767 of 32768-row buffers, heads 16..63 never written",
            spec=dict(_ML4_16_FQ64, **_ML4_CALLS["k_b_nope"], tail_from=_NVFP4_ROWS - 15),
        ),
    )

    note = """
    DSpark graph cells exercise rows 1..64 at 8/16/64 heads, for both latent
    query absorption and value expansion. These row counts are below the
    eager envelope and integer-width boundaries recorded below.

    What runs: `trtllm::bmm_out` is a `torch.library.custom_op` registered in
    `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` whose body is
    `torch.bmm(a, b, out=out)`, so one call is one cuBLAS strided-batched GEMM.
    At Mistral Large 4's views the profiler shows a single `nvjet_sm103_tst_*`
    kernel per call and no copy, at each of 34 values of m from 1 to 32768 and
    on the trailing 15 rows of 32768-row buffers: the product is written
    straight into the caller's view. cuBLAS picks the kernel by size -- over
    those 34 values it named 14, 13 and 15 distinct kernels for K-B RoPE, K-B
    NoPE and V-B, 5, 4 and 5 of them at the 13 values above 9216 -- and every
    m's rows are still bitwise the leading rows of the full-size run, which
    the `_ml4` sweep cells hold at 9216 and at 32768 rows. A transposed-view
    `b` and the same values made contiguous give bitwise the same product. At
    those three shapes and 32768 rows the product is within one bf16 rounding
    of an fp64 product. The cells run under torch's matmul defaults
    (`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` True).

    At DEP4's 64 heads the profiler shows one `nvjet_sm103_tst_*` launch per
    call and no copy, at every m from 1 to 32768: 36, 42 and 67
    distinct kernels for K-B RoPE, K-B NoPE and V-B, in 770, 634 and 1936 runs
    of m with one kernel and grid, every change between m = 8 and 9 or
    between 16k and 16k + 1. Every m's rows are bitwise the leading
    rows of the 32768-row run: the `large` cells drive each m, and pre-merge
    drives every m to 16 and both edges of every 16-row window, which holds
    both edges of every run. A transposed-view `b` and the same values made
    contiguous give bitwise the same product here too. At these shapes and
    32768 rows the product is within one bf16 rounding of an fp64 product.

    At TEP8's and TP8's 8 heads the profiler shows one `nvjet_sm103_tst_*`
    launch per call and no copy, at every m from 1 to 32768: 58, 59 and 86
    distinct kernels for K-B RoPE, K-B NoPE and V-B, in 615, 658 and 1563 runs
    of m with one kernel and grid, every change between m = 8 and 9 or between
    16k and 16k + 1. Every m's rows are bitwise the leading rows of the
    32768-row run: the `_ML4_8` sweep cells drive each m in pre-merge. A
    transposed-view `b` and the same values made contiguous give bitwise the
    same product here too. At these shapes and 32768 rows the product is
    within one bf16 rounding of an fp64 product.

    On the fp8_ds_mla latent KV a tep8 or tp8 rank's K-B output is heads 0..7
    of the `[T, 64, 576]` fused query FlashMLA takes (`_ML4_8_FQ64`): the same
    batch of 8, an output token stride of 64 x 576 = 36864 instead of 8 x 576,
    and heads 8..63 left as the step zeroed them. Those cells hold the product
    there at every m 1..32768, each m's rows bitwise the 32768-row run's (and
    at m 1..16 and both edges of every 16-row window on their own), in CUDA
    graphs at every batch 1..16 and on a mixed step's 15-row tail, with heads
    8..63 never written. The profiler shows one `nvjet_sm103_tst_*` launch
    per call and no copy at each of 40 values of m from 1 to 32768 and on the
    15-row tail, at every one of them the kernel cuBLAS picks for the compact
    8-head layout: 18 distinct for K-B RoPE, 14 for K-B NoPE (tep8's
    profile). At 32768 rows the product is within one bf16 rounding of an
    fp64 product here too.

    On the fp8_ds_mla latent KV a tep4 (or tp4) rank's K-B output is heads
    0..15 of the `[T, 64, 576]` fused query FlashMLA takes (`_ML4_16_FQ64`): the
    same batch of 16, an output token stride of 64 x 576 = 36864 instead of
    16 x 576, and heads 16..63 left as the step zeroed them. Those cells hold
    the product there at every m 1..32768, each m's rows bitwise the
    32768-row run's, in CUDA graphs at every batch 1..16 and on a mixed
    step's 15-row tail, with heads 16..63 never written. The profiler shows
    one `nvjet_sm103_tst_*` launch per call and no copy at each of 40 values
    of m from 1 to 32768 and on the 15-row tail, at every one of them the
    kernel cuBLAS picks for the compact 16-head layout: 16 distinct for K-B
    RoPE, 14 for K-B NoPE (tep4's profile; tp4's, at 32 values of m and the
    tail, found the same at every one, 13 distinct and 11). At 32768 rows the
    product is within one bf16 rounding of an fp64 product here too.

    `out` is written, not accumulated into: its prior contents are ignored.
    m = 0 returns at once and writes nothing. Capturable: a captured call
    replays bitwise equal to an eager one. Metadata consumed: none.

    Refused loudly by the op itself, each kind held by the test: `a` or `b`
    not 3-D (`batch1 must be a 3D tensor`), a batch or K mismatch, and a mix of
    dtypes -- every pairing of bf16, fp16 and fp32 with `a` and `b` differing,
    under each `out` dtype, and the two `out` dtypes other than the operands'
    that were driven, bf16 into fp32 and fp16 into bf16 (`expected scalar type
    BFloat16 but found Half`, or `Expected out tensor to have dtype ...`).
    Refused by `is_valid`: an `out` of the wrong shape, which the op resizes
    in place with only a deprecation warning. A standalone `out` is
    re-allocated, so a buffer the caller aliased with it never sees the
    product. A Mistral Large 4 K-B view one token short is re-strided contiguous
    over the fused query's storage: the product lands there scrambled for the
    buffer's layout, and over its rope halves.

    Not in any cell: fp16 or fp32 operands (accepted), a contiguous `a` or
    `out` (accepted), an `out` whose last dimension is not unit-stride, `a`
    overlapping `out`, batch counts other than 8, 16 and 64, a batch written
    into a wider head buffer other than 16 of 64 and 8 of 64, and m above
    32768 at 64 heads, at 16 of 64 or at 8 of 64 (four probes, below).

    Hard limits (sm_103). The NVFP4 target's m <= 32768 reaches at most
    1,207,959,552 elements into the fused query at 64 heads (56% of 2^31) --
    the 64-head fused query a tep4 or tp4 rank writes 16 heads of, and a tep8
    rank 8 heads of, is that size too, the last element written 1,207,931,839
    and 1,207,927,231 -- 301,989,888 at 16 and 150,994,944 at 8; the FP8
    target's m <= 9216 at most 84,934,656.
      row offset  m x token stride   first m with an offset at 2^31:
                  64 heads           K-B out (36864) 58,255 -- probed at m 58260
                                     on K-B RoPE and NoPE: passed, every row
                                     within the band, nothing written past n,
                                     calls on the rows from 58247, 58254, 58255
                                     and 58257 bitwise the full call's rows and
                                     nothing written ahead of them;
                                     V-B a (32768) 65,537 -- probed at m 65540:
                                     passed, every row within the band, calls
                                     on the rows from 65529, 65536 and 65537
                                     the same; K-B a and V-B out (8192)
                                     262,145, computed
                  16 heads           K-B out (9216) 233,017; V-B a (8192)
                                     262,145; K-B a and V-B out (2048)
                                     1,048,577, computed
                  8 heads            K-B out (4608) 466,034; V-B a (4096)
                                     524,289; K-B a and V-B out (1024)
                                     2,097,153, computed
                  16 of 64 heads     K-B out (36864) 58,255 -- probed at m 58260
                                     on K-B RoPE and NoPE, heads 0..15 of a
                                     64-head buffer (last element written
                                     2,147,668,927): passed, every row within
                                     the band, nothing written past n or into
                                     heads 16..63, calls on the rows from
                                     58247, 58254, 58255 and 58257 (storage
                                     offsets to 2,147,586,048) bitwise the
                                     full call's rows and nothing written
                                     ahead of them
                  8 of 64 heads      K-B out (36864) 58,255 -- probed at m 58260
                                     on K-B RoPE and NoPE, heads 0..7 of a
                                     64-head buffer (last element written
                                     2,147,664,319): passed, every row within
                                     the band, nothing written past n or into
                                     heads 8..63, calls on the rows from
                                     58247, 58254, 58255 and 58257 (storage
                                     offsets to 2,147,586,048) bitwise the
                                     full call's rows and nothing written
                                     ahead of them
      m           the row count      2^31, computed
    No index passes through fp32. The op touches no KV pool, only the step's
    activations, so the pool's size sets no limit on it.
    """

    def __call__(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.ops.trtllm.bmm_out(a, b, out)
        return out

    def reference(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """The product the op writes into `out`, fp32-accumulated.

        `out` is named to mirror `__call__`: the op takes a destination,
        the reference returns a value, and hiding the argument would hide that
        the op mutates one of its inputs.

        TF32 off: torch's default fp32 matmul on this hardware is a TF32
        product, ~1e-3 relative, which would make the reference the inaccurate
        side of the comparison.
        """
        with true_fp32_matmul():
            return (a.float() @ b.float()).to(a.dtype)

    def is_valid(self, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
        # The op refuses on its own an `a` or `b` that is not 3-D, a batch or K
        # mismatch and any mix of dtypes; only `out` is taken and answered
        # wrongly. A wrong-shaped `out` is resized in place: a standalone one
        # is re-allocated, a view is re-strided over its parent's storage
        # (seen where the parent had room). Either way the product lands
        # somewhere the caller does not read, and the forward continues with
        # stale data rather than failing.
        if a.dim() == 3 and b.dim() == 3:
            assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
                "out must be [B, M, N] matching a [B, M, K] and b [B, K, N]"
            )


bmm_out = BmmOut()
