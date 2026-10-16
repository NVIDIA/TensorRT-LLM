# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ngram_hash entry.

Exact integer arithmetic, gated bitwise against the entry's scalar reference.
A wrong hash is still a valid row index, so beyond the cells: six near-miss
derivations must each differ, the per-head bands, the EOS reset semantics,
row independence, the tail slice and chunked prefill, and the four guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.embedding.ngram_hash import (  # noqa: E501
    CKPT_CONTEXT_LEN,
    CKPT_EOS_TOKEN_ID,
    CKPT_HEAD_OFFSETS,
    CKPT_HEAD_VOCAB_SIZES,
    CKPT_MULTIPLIERS,
    CKPT_VOCAB_SIZE,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.embedding.ngram_hash import (  # noqa: E501
    ngram_hash as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "ngram_hash requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise: exact integer arithmetic, no tolerance to loosen."""
    assert out.dtype == ref.dtype == torch.int64, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    assert torch.equal(out, ref), f"{int((out != ref).sum())} indices differ"


DEV = "cuda"
EOS = CKPT_EOS_TOKEN_ID
NGRAM_SIZE = len(CKPT_MULTIPLIERS)
NGRAM_HEADS = len(CKPT_HEAD_VOCAB_SIZES)
PADDED_TABLE_ROWS = 320001536  # sum(head_vocab_sizes) rounded up to a multiple of 128


def _constants():
    return (
        torch.tensor(CKPT_MULTIPLIERS, dtype=torch.long, device=DEV),
        torch.tensor(CKPT_HEAD_VOCAB_SIZES, dtype=torch.long, device=DEV),
        torch.tensor(CKPT_HEAD_OFFSETS, dtype=torch.long, device=DEV),
    )


def _tokens(batch, length, seed, eos_positions=(), lead_eos=True, eos_density=0.0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, CKPT_VOCAB_SIZE, (batch, length), generator=g, dtype=torch.long)
    if eos_density > 0.0:
        ids[torch.rand(batch, length, generator=g) < eos_density] = EOS
    if lead_eos:
        ids[:, : min(CKPT_CONTEXT_LEN, length)] = EOS
    for position in eos_positions:
        ids[:, position] = EOS
    return ids.to(DEV)


def _hash(ids, num_query_tokens):
    return op(ids, *_constants(), EOS, num_query_tokens)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    ids = _tokens(
        spec["batch"],
        spec["length"],
        seed=zlib.crc32(cell.why.encode()),
        eos_positions=spec.get("eos_positions", ()),
        lead_eos=spec.get("lead_eos", True),
        eos_density=spec.get("eos_density", 0.0),
    ).to(spec.get("ids_dtype", torch.int64))
    before = ids.clone()
    args = (ids, *_constants(), EOS, spec["num_query_tokens"])
    out = op(*args)
    assert out.shape == (spec["batch"], spec["num_query_tokens"], NGRAM_HEADS)
    assert out.data_ptr() != ids.data_ptr()
    assert torch.equal(ids, before), "the history is read-only"
    _compare(out, op.reference(*args))


def _variant(rows, num_query_tokens, variant):
    """The scalar formula with one specific mistake in it."""
    multipliers = CKPT_MULTIPLIERS[::-1] if variant == "multipliers_reversed" else CKPT_MULTIPLIERS
    heads_per_ngram = NGRAM_HEADS // (NGRAM_SIZE - 1)
    out = []
    for row in rows:
        segment_start, last_eos = [], -1
        for position, token in enumerate(row):
            if variant == "inclusive_segment" and token == EOS:
                last_eos = position
            segment_start.append(last_eos + 1)
            if variant != "inclusive_segment" and token == EOS:
                last_eos = position
        row_out = []
        for position in range(len(row) - num_query_tokens, len(row)):
            context = []
            for shift in range(NGRAM_SIZE):
                source = position - shift - (1 if variant == "extra_shift" and shift else 0)
                if variant == "no_eos_reset":
                    inside = source >= 0
                else:
                    inside = position - segment_start[position] >= shift and source >= 0
                context.append(row[source] if inside else EOS)
            heads = []
            for ngram in range(2, NGRAM_SIZE + 1):
                mixed = context[0] * multipliers[0]
                for index in range(1, ngram):
                    mixed ^= context[index] * multipliers[index]
                block = ngram - 2
                if variant == "blocks_swapped":
                    block = NGRAM_SIZE - 2 - block
                low = block * heads_per_ngram
                for head in range(heads_per_ngram):
                    offset = 0 if variant == "no_offsets" else CKPT_HEAD_OFFSETS[low + head]
                    heads.append(mixed % CKPT_HEAD_VOCAB_SIZES[low + head] + offset)
            row_out.append(heads)
        out.append(row_out)
    return torch.tensor(out, dtype=torch.long, device=DEV)


def test_wrong_variants_are_separated() -> None:
    """Six near misses: each in range, inside the table, and different."""
    ids = _tokens(2, CKPT_CONTEXT_LEN + 512, seed=5, eos_positions=(30, 31, 200, 400))
    got = _hash(ids, 512)
    _compare(got, op.reference(ids, *_constants(), EOS, 512))
    for variant in (
        "extra_shift",
        "no_eos_reset",
        "inclusive_segment",
        "multipliers_reversed",
        "no_offsets",
        "blocks_swapped",
    ):
        wrong = _variant(ids.tolist(), 512, variant)
        differing = int((got != wrong).sum())
        print(f"  {variant:<22} {differing} indices differ")
        assert differing > 0, f"{variant} is indistinguishable from the correct result"
        assert 0 <= int(wrong.min()) and int(wrong.max()) < PADDED_TABLE_ROWS


def test_every_head_lands_in_its_own_band() -> None:
    got = _hash(_tokens(2, CKPT_CONTEXT_LEN + 512, seed=9, eos_positions=(100,)), 512)
    for head in range(NGRAM_HEADS):
        column = got[:, :, head]
        assert int(column.min()) >= CKPT_HEAD_OFFSETS[head]
        assert int(column.max()) < CKPT_HEAD_OFFSETS[head] + CKPT_HEAD_VOCAB_SIZES[head]


def test_eos_restarts_the_window() -> None:
    """One past an EOS hashes as a sequence's first token, and only there."""
    token = 4242
    fresh = torch.tensor([[EOS] * CKPT_CONTEXT_LEN + [token]], device=DEV)
    after_eos = torch.tensor([[11, 22, 33, EOS, token]], device=DEV)
    no_eos = torch.tensor([[11, 22, 33, 44, token]], device=DEV)
    assert torch.equal(_hash(fresh, 1), _hash(after_eos, 1))
    assert not torch.equal(_hash(fresh, 1), _hash(no_eos, 1))


def test_eos_token_itself_ends_its_segment() -> None:
    """An EOS hashes with the segment it closes, not as a fresh start."""
    closing = torch.tensor([[11, 22, 33, EOS]], device=DEV)
    opening = torch.tensor([[EOS] * (CKPT_CONTEXT_LEN + 1)], device=DEV)
    assert not torch.equal(_hash(closing, 1), _hash(opening, 1))


def test_rows_are_independent() -> None:
    ids = _tokens(4, CKPT_CONTEXT_LEN + 64, seed=13, eos_positions=(20,))
    batched = _hash(ids, 64)
    for row in range(ids.shape[0]):
        assert torch.equal(batched[row : row + 1], _hash(ids[row : row + 1], 64))


def test_tail_slice_is_exact() -> None:
    """Asking for Q rows equals asking for all and keeping the last Q."""
    ids = _tokens(2, 200, seed=17, eos_positions=(50, 51))
    whole = _hash(ids, 200)
    for num_query_tokens in (1, 3, 64, 199):
        assert torch.equal(_hash(ids, num_query_tokens), whole[:, -num_query_tokens:])


def test_chunked_prefill_agrees_with_one_shot() -> None:
    """Two chunks, the second carrying the `ngram_size - 1` cached ids."""
    ids = _tokens(2, CKPT_CONTEXT_LEN + 96, seed=19, eos_positions=(30, 60))
    split = CKPT_CONTEXT_LEN + 48
    first = _hash(ids[:, :split], 48)
    second = _hash(ids[:, split - CKPT_CONTEXT_LEN :], 48)
    assert torch.equal(torch.cat([first, second], dim=1), _hash(ids, 96))


# The four guards: each shows the unguarded op answering plausibly, then the
# refusal under `validating`.


def test_guard_refuses_num_query_tokens_beyond_the_history() -> None:
    ids = _tokens(1, CKPT_CONTEXT_LEN, seed=31)
    unguarded = op(ids, *_constants(), EOS, CKPT_CONTEXT_LEN + 1)
    assert unguarded.shape == (1, CKPT_CONTEXT_LEN - 1, NGRAM_HEADS), (
        "the hazard this guard covers is gone: it used to return fewer rows than asked"
    )
    for bad in (CKPT_CONTEXT_LEN + 1, -1):
        with pytest.raises(AssertionError, match="num_query_tokens"):
            with validating(op):
                op(ids, *_constants(), EOS, bad)


def test_guard_refuses_a_ragged_head_count() -> None:
    ids = _tokens(1, CKPT_CONTEXT_LEN + 4, seed=37)
    multipliers, sizes, offsets = _constants()
    unguarded = op(ids, multipliers, sizes[:15], offsets[:15], EOS, 4)
    assert unguarded.shape[-1] == 14, "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="multiple of"):
        with validating(op):
            op(ids, multipliers, sizes[:15], offsets[:15], EOS, 4)


def test_guard_refuses_an_over_long_offsets_tensor() -> None:
    ids = _tokens(1, CKPT_CONTEXT_LEN + 4, seed=41)
    multipliers, sizes, offsets = _constants()
    long_offsets = torch.cat([offsets, offsets])
    unguarded = op(ids, multipliers, sizes, long_offsets, EOS, 4)
    assert torch.equal(unguarded, _hash(ids, 4)), "the hazard this guard covers is gone"
    with pytest.raises(AssertionError, match="same length"):
        with validating(op):
            op(ids, multipliers, sizes, long_offsets, EOS, 4)


def test_guard_refuses_narrow_multipliers() -> None:
    ids = _tokens(1, CKPT_CONTEXT_LEN + 4, seed=47)
    multipliers, sizes, offsets = _constants()
    narrow = multipliers.to(torch.int32)
    unguarded = op(ids, narrow, sizes, offsets, EOS, 4)
    assert not torch.equal(unguarded, _hash(ids, 4)), "the hazard this guard covers is gone"
    assert int(unguarded.min()) >= 0 and int(unguarded.max()) < PADDED_TABLE_ROWS
    with pytest.raises(AssertionError, match="int64"):
        with validating(op):
            op(ids, narrow, sizes, offsets, EOS, 4)


def test_single_multiplier_raises_on_its_own() -> None:
    ids = _tokens(1, CKPT_CONTEXT_LEN + 4, seed=43)
    multipliers, sizes, offsets = _constants()
    with pytest.raises(ZeroDivisionError):
        op(ids, multipliers[:1], sizes, offsets, EOS, 4)


def test_cuda_graph_capture_and_replay() -> None:
    ids = _tokens(8, CKPT_CONTEXT_LEN + 1, seed=53, lead_eos=False, eos_density=0.2)
    constants = _constants()
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(ids, *constants, EOS, 1)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(ids, *constants, EOS, 1)

    fresh = _tokens(8, CKPT_CONTEXT_LEN + 1, seed=54, lead_eos=False, eos_density=0.2)
    ids.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op.reference(fresh, *constants, EOS, 1))
