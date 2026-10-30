# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ngram_token_history entry.

Integer bookkeeping, gated bitwise -- window and the whole published pool --
against the entry's scalar reference, which builds each sequence's stream as
a list and shares no structure with the masked per-shift select under test.
Beyond the cells: chunked prefill against one-shot (the cross-step carry
the forward relies on), the `query_start_loc` length guard and CUDA-graph
replay.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ple.ngram_token_history__r14 import (  # noqa: E501
    CKPT_EOS_TOKEN_ID,
    CKPT_NGRAM_SIZE,
    CKPT_VOCAB_SIZE,
    ENGINE_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.ple.ngram_token_history__r14 import (  # noqa: E501
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
    """The op's positional arguments for one cell."""
    g = torch.Generator().manual_seed(seed)
    lengths = spec["lengths"]
    total = sum(lengths)
    ids = torch.randint(0, CKPT_VOCAB_SIZE, (total,), generator=g, dtype=torch.long)
    density = spec.get("eos_density", 0.0)
    if density > 0.0:
        ids[torch.rand(total, generator=g) < density] = EOS
    pool = torch.randint(
        0, CKPT_VOCAB_SIZE, (spec["pool_slots"], CONTEXT_LEN), generator=g, dtype=torch.long
    ).to(DEV)
    index_dtype = spec.get("index_dtype", torch.int32)
    return (
        ids.to(DEV, spec.get("ids_dtype", torch.int64)),
        pool,
        torch.tensor(spec["slots"], dtype=index_dtype, device=DEV),
        torch.tensor(spec["carries"], dtype=torch.bool, device=DEV),
        _qsl(lengths, spec.get("qsl_dtype", torch.int32), spec.get("qsl_from", "literal")),
        EOS,
        NGRAM,
    )


def _expect(args):
    """The reference, driven from a snapshot of the pool before the call."""
    return op.reference(args[0], args[1].clone(), *args[2:])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    read_only = [a.clone() for a in (args[0], args[2], args[3], args[4])]
    pool_before = args[1].clone()
    expected = _expect(args)
    window = op(*args)
    _compare((window, args[1]), expected)
    for a, b in zip((args[0], args[2], args[3], args[4]), read_only):
        assert torch.equal(a, b), "only context_pool may be written"
    # The whole-pool comparison locks the unnamed slots; this keeps it from
    # being vacuous -- every cell has a sequence that publishes new ids.
    assert not torch.equal(args[1], pool_before), "no slot moved"


# ----------------------------------------------------------- multi-step


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
    args = _build(spec, seed=11001)
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
