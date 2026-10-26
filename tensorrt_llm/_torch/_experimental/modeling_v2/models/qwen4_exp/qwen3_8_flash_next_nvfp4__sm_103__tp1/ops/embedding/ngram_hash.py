# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram hash: a token-id history becomes one global embedding row per head.

Tier-3 composed from native torch integer ops (~30 small launches); exact, so
gated bitwise. Nothing registered under ``torch.ops.trtllm`` hashes n-grams.
``EngramHashProvider.compute_hashes`` in
``tensorrt_llm/_torch/modules/engram/engram.py`` is the same family and a
different function on three axes: it hashes *compressed* ids (a tokenizer
table this checkpoint does not ship), derives its multipliers from a seed
(six ``(seed, layer)`` combinations reproduced none of this checkpoint's
stored ``layer_multipliers``), and pads instead of resetting at every EOS. Its
moduli rule does agree. The built-in Qwen4-Exp implementation has its own,
not wrapped here.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's `ple_embedding` buffers: 3-gram, 8 heads per order.
CKPT_MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)
CKPT_HEAD_VOCAB_SIZES = (
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
)  # fmt: skip
CKPT_HEAD_OFFSETS = (
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275,
)  # fmt: skip
CKPT_EOS_TOKEN_ID = 248044
CKPT_VOCAB_SIZE = 248320
#: `ngram_size - 1`: the cached ids ahead of the new tokens.
CKPT_CONTEXT_LEN = len(CKPT_MULTIPLIERS) - 1


def _rows(batch: int, length: int, query: int, why: str, **spec) -> Cell:
    return Cell(why=why, spec=dict(batch=batch, length=length, num_query_tokens=query, **spec))


class NgramHash(OpWrapper):
    """For a query token at history position `t` and head `h` of the `n`-gram
    block (`m = layer_multipliers`, `^` int64 XOR, `%` non-negative)::

        mixed[t] = (c[t] * m[0]) ^ (c[t - 1] * m[1]) ^ ... ^ (c[t - n + 1] * m[n - 1])
        row[t, h] = mixed[t] % head_vocab_sizes[h] + head_offsets[h]

    `c[t-k]` is `token_history[t-k]` when `t - k` lies in `t`'s segment and
    `eos_token_id` otherwise; a segment starts one past the most recent EOS
    **strictly before** `t` (an EOS closes its segment, so the token after it
    hashes as a sequence's first). Shifts before position 0 read EOS too.
    Heads run `[2-gram block | ... | ngram_size-gram block]`, `heads_per_ngram
    = head_vocab_sizes.numel() // (ngram_size - 1)` each, `ngram_size =
    layer_multipliers.numel()`.

    Returns a fresh contiguous int64 `[batch, num_query_tokens, ngram_heads]`
    for the **last** `num_query_tokens` positions; the leading ones only give
    the first query token its context. Indices are **global** -- the offset is
    applied, one concatenated table is addressed. No input is modified.

    Fusion boundary: segment scan, shifted streams, mix, modulo and offset.
    The id history (here built by `ngram_token_history`) and the table gather
    are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: batch, length, num_query_tokens; history is uniform random int64 ids
    # with the first `min(CONTEXT_LEN, length)` set to EOS when `lead_eos`
    # (default True), `eos_positions` forced to EOS, and `eos_density` of the
    # remaining ids replaced by EOS.
    #
    # Every cell is the call site's: one `[T, ngram_size]` window per packed
    # token from `ngram_token_history`, num_query_tokens 1.
    CELLS: tuple[Cell, ...] = (
        _rows(4, 3, 1, "decode row whose cached context is EOS", eos_positions=(0, 1)),
        _rows(
            32, 3, 1, "call site: 32 windows, decode / MTP verify", lead_eos=False, eos_density=0.2
        ),
        _rows(
            8192, 3, 1, "call site: 8192 windows, largest prefill", lead_eos=False, eos_density=0.05
        ),
    )

    note = """
    Metadata consumed: none; rows are independent, and within a row the only
    structure seen is `eos_token_id` -- packing two sequences into one row is
    safe only when the first ends in EOS. Call site: `ngram_hash(window,
    multipliers, head_vocab, head_offsets, eos_token_id, 1)` on the
    `[T, 3]` int64 window, reshaped to `[T, 16]` by the caller.

    Preconditions: `head_vocab_sizes`, `head_offsets` on `token_history`'s
    device (raises otherwise; a cpu `layer_multipliers` is accepted and
    gives the identical result). Certified on an int64 `token_history`, the
    window `ngram_token_history` returns. Negative ids are accepted and hashed
    as ordinary ids, not padding. Longer histories and `num_query_tokens > 1`
    (a chunk's tail) are not certified.
    A single multiplier raises `ZeroDivisionError`. The four silent mistakes
    are `is_valid`'s.

    A wrong hash is a valid row: an extra shift, no EOS reset, an inclusive
    segment start, reversed multipliers, no offsets, swapped blocks all stay
    in range and inside the table. The tail slice is taken before the mix
    (bitwise equal to slicing after), and chunked prefill with the
    `ngram_size - 1` carried ids reproduces one-shot bitwise.

    CUDA-graph capture certified, bitwise equal to eager. Cost, not gated:
    0.387 ms for 4096 x 16, 0.370 ms for an 8-row decode -- launch-bound.

    Hard limits
      token id  max_id * max(layer_multipliers) < 2^63  boundary 389114 for
        this checkpoint's multipliers  probed above it: accepted, no error,
        indices still inside the table (int64 wraps silently). The largest
        legal id 248319 gives 5.886e18. Not guarded: data, would sync.
    """

    def __call__(
        self,
        token_history: torch.Tensor,
        layer_multipliers: torch.Tensor,
        head_vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        eos_token_id: int,
        num_query_tokens: int,
    ) -> torch.Tensor:
        ids = token_history.long()
        batch_size, history_len = ids.shape
        ngram_size = layer_multipliers.numel()
        total_heads = head_vocab_sizes.numel()
        heads_per_ngram = total_heads // (ngram_size - 1)

        # Segment bookkeeping over the FULL history: the most recent EOS
        # strictly before each position starts that position's segment.
        positions = torch.arange(history_len, device=ids.device, dtype=torch.long)
        minus_one = positions.new_full((), -1)
        eos_positions = torch.where(ids == eos_token_id, positions, minus_one)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]],
            dim=1,
        )
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)

        # Elementwise along the token axis from here on, so slicing to the
        # query tail now is bitwise equal to slicing after the modulo.
        keep_from = history_len - num_query_tokens
        eos_fill = ids.new_full((), eos_token_id)
        shifted = [ids[:, keep_from:]]
        for shift in range(1, ngram_size):
            source_positions = positions - shift
            gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
            gathered = ids.gather(1, gather_positions)
            valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
            shifted.append(torch.where(valid, gathered, eos_fill)[:, keep_from:])

        blocks = []
        for ngram in range(2, ngram_size + 1):
            low = (ngram - 2) * heads_per_ngram
            high = low + heads_per_ngram
            mixed = shifted[0] * layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[position] * layer_multipliers[position])
            blocks.append(
                torch.remainder(mixed.unsqueeze(-1), head_vocab_sizes[low:high].view(1, 1, -1))
                + head_offsets[low:high].view(1, 1, -1)
            )
        return torch.cat(blocks, dim=-1)

    def reference(
        self,
        token_history: torch.Tensor,
        layer_multipliers: torch.Tensor,
        head_vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        eos_token_id: int,
        num_query_tokens: int,
    ) -> torch.Tensor:
        """Scalar Python-int arithmetic, one token and one head at a time, the
        segment start tracked by walking the row; no torch op computes it."""
        multipliers = layer_multipliers.tolist()
        sizes = head_vocab_sizes.tolist()
        offsets = head_offsets.tolist()
        ngram_size = len(multipliers)
        heads_per_ngram = len(sizes) // (ngram_size - 1)
        out = []
        for row in token_history.tolist():
            segment_start, last_eos = [], -1
            for position, token in enumerate(row):
                segment_start.append(last_eos + 1)
                if token == eos_token_id:
                    last_eos = position
            row_out = []
            for position in range(len(row) - num_query_tokens, len(row)):
                context = [
                    row[position - shift]
                    if position - segment_start[position] >= shift and position - shift >= 0
                    else eos_token_id
                    for shift in range(ngram_size)
                ]
                heads = []
                for ngram in range(2, ngram_size + 1):
                    mixed = context[0] * multipliers[0]
                    for index in range(1, ngram):
                        mixed ^= context[index] * multipliers[index]
                    low = (ngram - 2) * heads_per_ngram
                    heads.extend(
                        mixed % sizes[low + h] + offsets[low + h] for h in range(heads_per_ngram)
                    )
                row_out.append(heads)
            out.append(row_out)
        return torch.tensor(out, dtype=torch.long, device=token_history.device).reshape(
            token_history.shape[0], num_query_tokens, len(sizes)
        )

    def is_valid(
        self,
        token_history: torch.Tensor,
        layer_multipliers: torch.Tensor,
        head_vocab_sizes: torch.Tensor,
        head_offsets: torch.Tensor,
        eos_token_id: int,
        num_query_tokens: int,
    ) -> None:
        # Each measured to be accepted and to return plausible in-range rows.
        history_len = token_history.shape[-1]
        ngram_size = layer_multipliers.numel()
        total_heads = head_vocab_sizes.numel()
        assert 0 <= num_query_tokens <= history_len, (
            f"num_query_tokens must be in [0, history_len]; got {num_query_tokens} "
            f"against a history of {history_len} -- a value outside that range is "
            f"silently truncated to a shorter output rather than rejected"
        )
        assert layer_multipliers.dtype == torch.int64, (
            f"layer_multipliers must be int64; got {layer_multipliers.dtype} -- a "
            f"narrower dtype is accepted and silently hashes with truncated "
            f"multipliers"
        )
        assert ngram_size < 2 or total_heads % (ngram_size - 1) == 0, (
            f"head_vocab_sizes.numel()={total_heads} must be a multiple of "
            f"ngram_size-1={ngram_size - 1} -- the blocks are equal width, and a "
            f"remainder is silently dropped rather than rejected"
        )
        assert head_offsets.numel() == total_heads, (
            f"head_offsets and head_vocab_sizes must have the same length; got "
            f"{head_offsets.numel()} and {total_heads} -- an over-long head_offsets "
            f"is silently read for its first {total_heads} entries only"
        )


ngram_hash = NgramHash()
