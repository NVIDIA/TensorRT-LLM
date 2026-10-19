# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ngram_token_history entry.

Integer bookkeeping, gated bitwise -- window and the whole published pool --
against the entry's scalar reference, which builds each sequence's stream as
a list and shares no structure with the masked per-shift select under test.
A wrong window is a window of valid ids, so beyond the cells: six near-miss
assemblies must each differ, hand-computed windows pin the lifecycle, chunked
prefill is checked step by step, and the two guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.ngram_token_history import (  # noqa: E501
    CKPT_EOS_TOKEN_ID,
    CKPT_NGRAM_SIZE,
    CKPT_VOCAB_SIZE,
    ENGINE_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.ngram_token_history import (  # noqa: E501
    ngram_token_history as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "ngram_token_history requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, torch.Tensor], ref: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """`(window, context_pool)` -- the returned window with the pool as the
    call left it -- against `reference`'s pair, both bitwise: integer
    bookkeeping, no tolerance. The whole pool is compared, which is the
    lock on every slot the batch did not name."""
    (window, pool), (ref_window, ref_pool) = out, ref
    assert window.dtype == ref_window.dtype == torch.int64, (window.dtype, ref_window.dtype)
    assert window.shape == ref_window.shape, (window.shape, ref_window.shape)
    assert window.is_contiguous()
    assert torch.equal(window, ref_window), "window disagrees with the reference"
    assert torch.equal(pool, ref_pool), "published context pool disagrees with the reference"


DEV = "cuda"
EOS = CKPT_EOS_TOKEN_ID
NGRAM = CKPT_NGRAM_SIZE
CONTEXT_LEN = NGRAM - 1


def _qsl(lengths, dtype, qsl_from="literal"):
    if qsl_from == "seq_lens":
        # The call site under MTP verify: an int32 prefix of the metadata's
        # seq_lens buffer, scanned into an int64 vector with a leading zero.
        buf = torch.full((len(lengths) + 9,), 99, dtype=torch.int32, device=DEV)
        buf[: len(lengths)] = torch.tensor(lengths, dtype=torch.int32, device=DEV)
        seq_lens = buf[: len(lengths)]
        out = torch.zeros(len(lengths) + 1, dtype=torch.int64, device=DEV)
        torch.cumsum(seq_lens, dim=0, dtype=torch.int64, out=out[1:])
        return out
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    return torch.tensor(starts, dtype=dtype, device=DEV)


def _build(spec, seed):
    """`(args, backing_pool)`: the op's positional arguments, and the tensor
    the pool argument views (itself, unless `layout == "pool_view"`)."""
    g = torch.Generator().manual_seed(seed)
    ngram_size = spec.get("ngram_size", NGRAM)
    lengths = spec["lengths"]
    total = sum(lengths)
    ids = torch.randint(0, CKPT_VOCAB_SIZE, (total,), generator=g, dtype=torch.long)
    density = spec.get("eos_density", 0.0)
    if density > 0.0:
        ids[torch.rand(total, generator=g) < density] = EOS
    layout = spec.get("layout", "contiguous")
    pool_rows = spec["pool_slots"] * (2 if layout == "pool_view" else 1)
    backing = torch.randint(
        0, CKPT_VOCAB_SIZE, (pool_rows, ngram_size - 1), generator=g, dtype=torch.long
    ).to(DEV)
    pool = backing[::2] if layout == "pool_view" else backing
    input_ids = ids.to(DEV, spec.get("ids_dtype", torch.int64))
    if layout == "strided_ids":
        wide = torch.zeros(2 * total, dtype=input_ids.dtype, device=DEV)
        wide[::2] = input_ids
        input_ids = wide[::2]
        assert input_ids.stride(0) == 2
    index_dtype = spec.get("index_dtype", torch.int32)
    args = (
        input_ids,
        pool,
        torch.tensor(spec["slots"], dtype=index_dtype, device=DEV),
        torch.tensor(spec["carries"], dtype=torch.bool, device=DEV),
        _qsl(lengths, spec.get("qsl_dtype", torch.int32), spec.get("qsl_from", "literal")),
        EOS,
        ngram_size,
    )
    return args, backing


def _expect(args):
    """The reference, driven from a snapshot of the pool before the call."""
    return op.reference(args[0], args[1].clone(), *args[2:])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args, backing = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    read_only = [a.clone() for a in (args[0], args[2], args[3], args[4])]
    backing_before = backing.clone()
    pool_before = args[1].clone()
    expected = _expect(args)
    window = op(*args)
    _compare((window, args[1]), expected)
    for a, b in zip((args[0], args[2], args[3], args[4]), read_only):
        assert torch.equal(a, b), "only context_pool may be written"
    # The whole-pool comparison locks the unnamed slots; this keeps it from
    # being vacuous -- every cell has a sequence that publishes new ids.
    assert not torch.equal(args[1], pool_before), "no slot moved"
    if cell.spec.get("layout") == "pool_view":
        assert torch.equal(backing[1::2], backing_before[1::2]), "the view wrote outside itself"


# ----------------------------------------------------------- multi-step


def test_chunked_prefill_carries_context_across_steps() -> None:
    """One slot over three forward steps (96, 24, 7 tokens), each against the
    reference from the pool the previous step left."""
    pool = torch.randint(0, CKPT_VOCAB_SIZE, (5, CONTEXT_LEN), device=DEV)
    g = torch.Generator().manual_seed(4001)
    for step, (length, carry) in enumerate(((96, False), (24, True), (7, True))):
        ids = torch.randint(0, CKPT_VOCAB_SIZE, (length,), generator=g).to(DEV)
        if step == 0:
            ids[torch.rand(length, generator=g) < 0.04] = EOS
        args = (
            ids,
            pool,
            torch.tensor([2], dtype=torch.int32, device=DEV),
            torch.tensor([carry], device=DEV),
            torch.tensor([0, length], dtype=torch.int32, device=DEV),
            EOS,
            NGRAM,
        )
        expected = _expect(args)
        _compare((op(*args), pool), expected)


def test_chunked_prefill_matches_one_shot() -> None:
    """96 tokens split at 41 over two calls: window and pool bitwise one-shot."""
    g = torch.Generator().manual_seed(5001)
    ids = torch.randint(0, CKPT_VOCAB_SIZE, (96,), generator=g).to(DEV)
    start = torch.randint(0, CKPT_VOCAB_SIZE, (3, CONTEXT_LEN), generator=g).to(DEV)
    slot = torch.tensor([1], dtype=torch.int32, device=DEV)

    def call(pool, chunk, carry):
        flags = torch.tensor([carry], device=DEV)
        qsl = torch.tensor([0, chunk.shape[0]], dtype=torch.int32, device=DEV)
        return op(chunk, pool, slot, flags, qsl, EOS, NGRAM)

    one_shot_pool = start.clone()
    one_shot = call(one_shot_pool, ids, False)
    pool = start.clone()
    split = torch.cat([call(pool, ids[:41], False), call(pool, ids[41:], True)])
    assert torch.equal(split, one_shot)
    assert torch.equal(pool, one_shot_pool)


# ----------------------------------------------------------- hand-computed


def _literal(ids, pool, slots, carries, starts):
    window = op(
        torch.tensor(ids, device=DEV),
        pool,
        torch.tensor(slots, dtype=torch.int32, device=DEV),
        torch.tensor(carries, device=DEV),
        torch.tensor(starts, dtype=torch.int32, device=DEV),
        EOS,
        NGRAM,
    )
    return window.tolist()


def test_fresh_sequence_seeds_with_eos_not_the_pool() -> None:
    pool = torch.full((4, CONTEXT_LEN), 1234, device=DEV)
    window = _literal([11, 22, 33], pool, [2], [False], [0, 3])
    assert window == [[EOS, EOS, 11], [EOS, 11, 22], [11, 22, 33]]
    assert pool[2].tolist() == [22, 33]


def test_carry_row_is_oldest_first_and_publishes_its_tail() -> None:
    pool = torch.tensor([[70, 80]], device=DEV)
    assert _literal([90, 100], pool, [0], [True], [0, 2]) == [[70, 80, 90], [80, 90, 100]]
    assert pool[0].tolist() == [90, 100]
    pool = torch.tensor([[70, 80]], device=DEV)
    assert _literal([90], pool, [0], [True], [0, 1]) == [[70, 80, 90]]
    assert pool[0].tolist() == [80, 90], "part carry, part new"


def test_zero_token_rows_keep_or_reseed() -> None:
    pool = torch.tensor([[70, 80], [11, 12], [13, 14]], device=DEV)
    window = _literal([90, 91], pool, [0, 1, 2], [True, True, False], [0, 2, 2, 2])
    assert window == [[70, 80, 90], [80, 90, 91]]
    assert pool.tolist() == [[90, 91], [11, 12], [EOS, EOS]]


def test_sequence_boundary_is_not_crossed() -> None:
    pool = torch.tensor([[70, 80], [170, 180]], device=DEV)
    window = _literal([90, 91, 92, 190, 191], pool, [0, 1], [True, True], [0, 3, 5])
    assert window[3:] == [[170, 180, 190], [180, 190, 191]]


def test_in_stream_eos_is_passed_through_raw() -> None:
    pool = torch.tensor([[70, 80]], device=DEV)
    window = _literal([EOS, 91, 92], pool, [0], [True], [0, 3])
    assert window == [[70, 80, EOS], [80, EOS, 91], [EOS, 91, 92]]


# ----------------------------------------------------------- discrimination


def test_wrong_variants_are_separated() -> None:
    """Six mis-assemblies, each in-vocabulary; each must differ in window or pool."""
    spec = dict(
        lengths=(17, 1, 5, 1),
        slots=(3, 0, 2, 1),
        carries=(False, True, True, True),
        pool_slots=6,
        eos_density=0.08,
    )
    args, _ = _build(spec, seed=9001)
    ids, pool, slots, carries, qsl = args[:5]
    start = pool.clone()
    window = op(*args)
    published = pool

    def ref(**override):
        kw = dict(
            input_ids=ids,
            context_pool=start.clone(),
            state_indices=slots,
            has_initial_states=carries,
            query_start_loc=qsl,
            eos_token_id=EOS,
            ngram_size=NGRAM,
        )
        kw.update(override)
        return op.reference(**kw)

    right_window, right_pool = ref()

    def publishes_the_head():
        head_pool = start.clone()
        starts = qsl.tolist()
        for s, slot in enumerate(slots.tolist()):
            if starts[s + 1] > starts[s]:
                head_pool[slot] = right_window[starts[s + 1] - 1, :-1]
        return right_window, head_pool

    variants = {
        "newest_first": lambda: (right_window.flip(1), right_pool),
        "zero_seed": lambda: ref(eos_token_id=0),
        "ignores_carry_flags": lambda: ref(has_initial_states=torch.ones_like(carries)),
        # One flat stream: each sequence's first tokens read its neighbour's tail.
        "packs_across_boundaries": lambda: (
            ref(
                state_indices=slots[:1],
                has_initial_states=carries[:1],
                query_start_loc=qsl[[0, -1]],
            )[0],
            right_pool,
        ),
        "rotated_pool_read": lambda: ref(context_pool=start.roll(1, dims=1)),
        "publishes_the_head": publishes_the_head,
    }
    for name, build in variants.items():
        wrong_window, wrong_pool = build()
        same = torch.equal(window, wrong_window) and torch.equal(published, wrong_pool)
        assert not same, f"variant {name} is indistinguishable"


# ----------------------------------------------------------- guards


def _two_sequences():
    pool = torch.randint(0, CKPT_VOCAB_SIZE, (4, CONTEXT_LEN), device=DEV)
    return (
        torch.tensor([11, 12, 13, 21], device=DEV),
        pool,
        torch.tensor([0, 1], dtype=torch.int32, device=DEV),
        torch.tensor([True, False], device=DEV),
        torch.tensor([0, 3, 4], dtype=torch.int32, device=DEV),
        EOS,
        NGRAM,
    )


def _misses(args, bad_index, bad_value):
    """Run the op unguarded with one argument swapped; True if it returned
    and its answer differs from the reference on the intended arguments."""
    expected = _expect(args)
    bad = list(args)
    bad[1] = args[1].clone()
    bad[bad_index] = bad_value
    window = op(*bad)
    return not (torch.equal(window, expected[0]) and torch.equal(bad[1], expected[1]))


def test_guard_refuses_carry_flags_of_the_wrong_length() -> None:
    """Unguarded, one flag broadcasts: sequence 1 reads its stale slot, not the seed."""
    args = _two_sequences()
    one_flag = torch.ones(1, dtype=torch.bool, device=DEV)
    assert _misses(args, 3, one_flag), "the hazard this guard covers is gone"
    bad = list(args)
    bad[3] = one_flag
    with pytest.raises(AssertionError, match="one flag per sequence"):
        with validating(op):
            op(*bad)


def test_guard_refuses_a_short_query_start_loc() -> None:
    """Unguarded, a 2-entry vector for 2 sequences misattributes the tail."""
    args = _two_sequences()
    short = torch.tensor([0, 3], dtype=torch.int32, device=DEV)
    assert _misses(args, 4, short), "the hazard this guard covers is gone"
    bad = list(args)
    bad[4] = short
    with pytest.raises(AssertionError, match=r"query_start_loc must be \[num_seqs"):
        with validating(op):
            op(*bad)


@pytest.mark.parametrize(
    "index,value,match",
    [
        (1, lambda a: torch.zeros(4, CONTEXT_LEN + 1, dtype=torch.long, device=DEV), "must match"),
        (1, lambda a: a[1].to(torch.int32), "Index put requires the source and destination"),
        (3, lambda a: a[3].to(torch.int32), "where expected condition to be a boolean"),
        (2, lambda a: a[2].reshape(-1, 1), "Index is supposed to be an empty tensor or a vector"),
        (6, lambda a: 1, "must match the size of tensor"),
    ],
    ids=["pool_too_wide", "int32_pool", "int_flags", "2d_state_indices", "ngram_size_1"],
)
def test_loud_refusals_are_not_guarded(index, value, match) -> None:
    args = list(_two_sequences())
    args[index] = value(args)
    with pytest.raises(RuntimeError, match=match):
        op(*args)


def test_two_dimensional_input_ids_is_refused() -> None:
    with pytest.raises(RuntimeError, match="must match the size of tensor"):
        op(
            torch.zeros(1, 4, dtype=torch.long, device=DEV),
            torch.zeros(2, CONTEXT_LEN, dtype=torch.long, device=DEV),
            torch.zeros(1, dtype=torch.int32, device=DEV),
            torch.ones(1, dtype=torch.bool, device=DEV),
            torch.tensor([0, 1], dtype=torch.int32, device=DEV),
            EOS,
            NGRAM,
        )


# ----------------------------------------------------------- capture


def test_cuda_graph_capture_and_replay() -> None:
    """The captured decode shape, 8 sequences at the engine pool width."""
    spec = dict(
        lengths=(1,) * 8,
        slots=(0, ENGINE_SLOTS - 1, 4, 17, 30, 43, 56, 69),
        carries=(True,) * 8,
        pool_slots=ENGINE_SLOTS,
        ids_dtype=torch.int32,
        qsl_dtype=torch.int64,
    )
    args, _ = _build(spec, seed=11001)
    start = args[1].clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op(*args)
    torch.cuda.current_stream().wait_stream(side)

    args[1].copy_(start)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(*args)

    replay_ids = torch.randint(0, CKPT_VOCAB_SIZE, (8,), dtype=torch.int32, device=DEV)
    args[0].copy_(replay_ids)
    args[1].copy_(start)
    expected = op.reference(replay_ids, start.clone(), *args[2:])
    graph.replay()
    torch.cuda.synchronize()
    _compare((captured, args[1]), expected)
