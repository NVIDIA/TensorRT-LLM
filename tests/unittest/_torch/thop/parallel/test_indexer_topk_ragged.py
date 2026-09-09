# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ragged (per-request) query lengths for ``indexer_topk_decode``.

The uniform kernel reconstructs two things from the scalar ``next_n``: which
request a logits row belongs to (``row // next_n``) and how far back that row
may attend (``seq_len - next_n + row % next_n + 1``). Neither holds once DSpark's
confidence scheduler gives each request its own verify window, so the kernel
takes an optional ``row_kv_lens`` carrying the extent per row.

The dangerous failure is silent, not loud: a ragged batch whose padded row count
happens to be divisible by the request count satisfies the old
``seq_lens.size(0) * next_n == num_rows`` check, and the kernel then attributes
rows to the wrong requests and masks them at the wrong depth. Every test here is
built to catch that rather than a crash.
"""

import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for indexer_topk tests", allow_module_level=True)


def _register_ops() -> None:
    """Make ``torch.ops.trtllm.indexer_topk_decode`` available.

    Importing ``tensorrt_llm`` is the usual way, but it drags in the whole model
    zoo -- including optional CuTe-DSL code paths that are absent in some
    containers. This file tests one C++ kernel, so it falls back to loading the
    op library directly rather than being unrunnable whenever an unrelated
    optional dependency is missing.
    """
    try:
        import tensorrt_llm  # noqa: F401

        return
    except ImportError:
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, *[os.pardir] * 5))
    for candidate in (
        os.path.join(repo, "cpp", "build", "tensorrt_llm", "thop", "libth_common.so"),
        os.path.join(repo, "tensorrt_llm", "libs", "libth_common.so"),
    ):
        if os.path.exists(candidate):
            torch.ops.load_library(candidate)
            return
    pytest.skip("neither tensorrt_llm nor libth_common.so is importable", allow_module_level=True)


_register_ops()

INDEX_TOPK = 512
NUM_COLUMNS = 8192

# Every KV length is kept above ``INDEX_TOPK * kMaxBlocksPerRow`` so that each
# block of the split-work path still has at least ``INDEX_TOPK`` candidates to
# rank. Below that a block cannot fill its output slice, the merge pass reads
# whatever was in the scratch buffer, and two runs differ purely by allocator
# luck -- which looks exactly like a kernel bug.
KV_LENS = [2048, 4096, 3000, 2560]


def _assert_same_topk(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare top-k *sets* per row, not the raw index arrays.

    The kernel does not promise an output order: the radix path's split-work
    merge visits blocks in whatever order they finish, so two identical calls
    routinely return the same indices permuted differently. Only the selected
    set is part of the contract, and it is what a wrong causal extent would
    change.
    """
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    actual_sorted = actual.sort(dim=-1).values
    expected_sorted = expected.sort(dim=-1).values
    torch.testing.assert_close(actual_sorted, expected_sorted, atol=0, rtol=0)


def _make_logits(num_rows: int, num_columns: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(
        num_rows, num_columns, dtype=torch.float32, device="cuda", generator=generator
    )


def _row_kv_lens(kv_lens: list[int], verify_lens: list[int]) -> torch.Tensor:
    """The extent every row may attend to, matching the kernel's uniform formula.

    Request ``r`` verifying ``v`` positions contributes rows ``o = 0..v-1``, and
    row ``o`` sees ``kv_len - v + o + 1`` tokens. Substituting ``v = next_n``
    recovers exactly what the uniform path computes internally, which is what
    makes the equivalence test below meaningful.
    """
    rows: list[int] = []
    for kv_len, verify_len in zip(kv_lens, verify_lens):
        rows.extend(kv_len - verify_len + offset + 1 for offset in range(verify_len))
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def _run(
    logits: torch.Tensor,
    seq_lens: list[int],
    next_n: int,
    row_kv_lens: torch.Tensor = None,
    compress_ratio: int = 1,
) -> torch.Tensor:
    num_rows = logits.shape[0]
    indices = torch.empty(num_rows, INDEX_TOPK, dtype=torch.int32, device="cuda")
    # The op refuses to allocate its own split-work scratch (stale pointers are
    # a CUDA-graph hazard), so the caller owns it. kMaxBlocksPerRow is small;
    # over-allocating is free here and keeps the test independent of the
    # dispatcher's blocks-per-row heuristic.
    aux_elems = num_rows * 8 * INDEX_TOPK
    # Zeroed, not empty: the merge pass reads every slot its blocks could have
    # written, so uninitialized scratch makes the result allocator-dependent.
    aux_indices = torch.zeros(aux_elems, dtype=torch.int32, device="cuda")
    aux_logits = torch.zeros(aux_elems, dtype=torch.float32, device="cuda")
    torch.ops.trtllm.indexer_topk_decode(
        logits,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        indices,
        next_n,
        INDEX_TOPK,
        compress_ratio=compress_ratio,
        radix_aux_indices=aux_indices,
        radix_aux_logits=aux_logits,
        row_kv_lens=row_kv_lens,
    )
    return indices


@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2, 6])
def test_ragged_with_equal_lens_matches_uniform(next_n, compress_ratio):
    """``row_kv_lens`` spelled out by hand must reproduce the uniform result.

    This is the load-bearing test. If the explicit per-row extent and the
    kernel's internal ``next_n`` arithmetic ever disagree, every ragged result
    is wrong in a way no shape check can see -- so pin them to be bit-identical
    on a batch where both are defined.
    """
    batch_size = 4
    kv_lens = list(KV_LENS)
    logits = _make_logits(batch_size * next_n, NUM_COLUMNS, seed=1234)

    uniform = _run(logits, kv_lens, next_n, compress_ratio=compress_ratio)
    ragged = _run(
        logits,
        kv_lens,
        next_n=1,
        row_kv_lens=_row_kv_lens(kv_lens, [next_n] * batch_size),
        compress_ratio=compress_ratio,
    )

    _assert_same_topk(uniform, ragged)


def test_genuinely_ragged_matches_per_request_reference():
    """Each request's rows must match running that request on its own.

    A per-request reference is the only check that catches cross-request
    misattribution: if row ``i`` were scored against the wrong request's
    sequence length, a whole-batch self-consistency check would still pass.
    """
    kv_lens = list(KV_LENS)
    verify_lens = [6, 3, 1, 4]
    total_rows = sum(verify_lens)
    logits = _make_logits(total_rows, NUM_COLUMNS, seed=99)

    ragged = _run(logits, kv_lens, next_n=1, row_kv_lens=_row_kv_lens(kv_lens, verify_lens))

    offset = 0
    for kv_len, verify_len in zip(kv_lens, verify_lens):
        rows = logits[offset : offset + verify_len].contiguous()
        expected = _run(rows, [kv_len], verify_len)
        _assert_same_topk(ragged[offset : offset + verify_len], expected)
        offset += verify_len


def test_divisible_ragged_batch_is_not_silently_accepted():
    """The shape that used to pass the old check by coincidence.

    ``verify_lens = [6, 4, 1, 1]`` totals 12 rows for 4 requests, so
    ``seq_lens.size(0) * next_n == num_rows`` holds for ``next_n == 3`` even
    though no request verifies 3 positions. Without ``row_kv_lens`` the kernel
    accepts it and computes nonsense; with it, the answer must match the
    per-request reference.
    """
    kv_lens = list(KV_LENS)
    verify_lens = [6, 4, 1, 1]
    assert sum(verify_lens) % len(verify_lens) == 0, "test premise"
    logits = _make_logits(sum(verify_lens), NUM_COLUMNS, seed=7)

    ragged = _run(logits, kv_lens, next_n=1, row_kv_lens=_row_kv_lens(kv_lens, verify_lens))

    bogus_next_n = sum(verify_lens) // len(verify_lens)
    accidental = _run(logits, kv_lens, bogus_next_n)
    assert not torch.equal(ragged.sort(dim=-1).values, accidental.sort(dim=-1).values), (
        "the ragged result coincided with the misattributed uniform one; the "
        "test can no longer detect the bug it exists for"
    )


def test_row_kv_lens_length_is_validated():
    """A short ``row_kv_lens`` must raise rather than read out of bounds."""
    logits = _make_logits(8, NUM_COLUMNS, seed=3)
    with pytest.raises(RuntimeError, match="one entry per logits row"):
        _run(
            logits,
            [1024, 2048],
            next_n=1,
            row_kv_lens=torch.tensor([1024] * 4, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.parametrize(
    ("compress_ratio", "invalid_extent"),
    [
        (1, -1),
        (1, NUM_COLUMNS + 1),
        (4, (NUM_COLUMNS + 1) * 4),
    ],
)
def test_invalid_row_kv_extent_fails_closed(compress_ratio: int, invalid_extent: int) -> None:
    """Device-selected extents must never address outside a logits row.

    Two rows of width 8192 select four blocks per row on supported GPUs, so
    this also pins the fail-closed behavior through the split/merge path.
    """
    logits = _make_logits(2, NUM_COLUMNS, seed=5)
    row_kv_lens = torch.tensor([KV_LENS[0], invalid_extent], dtype=torch.int32, device="cuda")

    actual = _run(
        logits,
        KV_LENS[:2],
        next_n=1,
        row_kv_lens=row_kv_lens,
        compress_ratio=compress_ratio,
    )

    assert torch.all(actual[1] == -1)


@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_full_width_row_kv_extent_is_valid(compress_ratio: int) -> None:
    """The exact logical row width is valid, not an out-of-range extent."""
    logits = _make_logits(2, NUM_COLUMNS, seed=6)
    full_width_extent = NUM_COLUMNS * compress_ratio
    seq_lens = [full_width_extent, full_width_extent]

    uniform = _run(logits, seq_lens, next_n=1, compress_ratio=compress_ratio)
    ragged = _run(
        logits,
        seq_lens,
        next_n=1,
        row_kv_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        compress_ratio=compress_ratio,
    )

    _assert_same_topk(ragged, uniform)


def test_cuda_graph_capture_replays_with_new_lens():
    """The ragged path must be capturable and must honor rewritten lengths.

    Capture is the whole point of landing on a token bucket: a ragged step that
    cannot replay costs far more than the tokens it trims. Replay with different
    contents in the same buffers is what a real step does.
    """
    kv_lens = list(KV_LENS)
    verify_lens = [6, 3, 1, 4]
    total_rows = sum(verify_lens)
    logits = _make_logits(total_rows, NUM_COLUMNS, seed=11)

    seq_lens_dev = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    row_kv_lens_dev = _row_kv_lens(kv_lens, verify_lens)
    indices = torch.empty(total_rows, INDEX_TOPK, dtype=torch.int32, device="cuda")
    # Capture requires caller-owned scratch: the op refuses to allocate it
    # itself precisely because a graph would bake in a stale pointer.
    aux_elems = total_rows * 8 * INDEX_TOPK
    aux_indices = torch.zeros(aux_elems, dtype=torch.int32, device="cuda")
    aux_logits = torch.zeros(aux_elems, dtype=torch.float32, device="cuda")

    def launch():
        torch.ops.trtllm.indexer_topk_decode(
            logits,
            seq_lens_dev,
            indices,
            1,
            INDEX_TOPK,
            compress_ratio=1,
            radix_aux_indices=aux_indices,
            radix_aux_logits=aux_logits,
            row_kv_lens=row_kv_lens_dev,
        )

    # Warm up outside the graph; capture would otherwise record allocator work.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    # A different split, written in place, must produce the eager answer.
    new_verify_lens = [4, 4, 2, 4]
    assert sum(new_verify_lens) == total_rows, "replay must keep the shape"
    row_kv_lens_dev.copy_(_row_kv_lens(kv_lens, new_verify_lens))
    graph.replay()
    torch.cuda.synchronize()
    replayed = indices.clone()

    eager = _run(logits, kv_lens, next_n=1, row_kv_lens=_row_kv_lens(kv_lens, new_verify_lens))
    _assert_same_topk(replayed, eager)
