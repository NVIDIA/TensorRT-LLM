# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_compute_probs_split entry.

Every row must be bit for bit its own one-row sampler call (the multi-CTA family): chunks one after another, on
parallel streams, and on parallel streams under CUDA-graph replay, at 33-96 rows with mixed per-row filters and
temperatures. Outside the split band (97-128 rows included) the entry is the one sampler call.

The sampler's top-p cut is not run-to-run reproducible for a token sitting on it (its radix histogram sums masses
with float atomics), in either family. Random logits put a top-p 0.95 cut deep in a dense tail, so there the
certified cells compare top-p rows within one boundary token (row L1) and every other row bit for bit; the
row-matching test uses logits whose cuts fall between well-separated tokens, where every row compares bit for bit.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.fused_compute_probs_split import (  # noqa: E501
    MAX_CHUNKS,
    SPLIT_MAX_ROWS,
    SplitStreams,
    split_row_chunks,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.fused_compute_probs_split import (  # noqa: E501
    fused_compute_probs_split as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    VOCAB,
    sampling_spec,
)
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

__extra_import_path__ = ["../../../../..", "../../../ops/spec"]
from _k31_spec_inputs import (  # noqa: E402
    DISABLE_TOP_K,
    GREEDY_SENTINEL_T,
    on_sm103,
    row_l1,
    sampling_inputs,
    seed_of,
)
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103

#: One top-p boundary token of a random-logit row (see the fused_compute_probs_from_logits test).
_ROW_L1 = 5e-5


def _bits_differ(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``[rows]`` bool: rows whose float32 words differ anywhere."""
    return (a.view(torch.int32) != b.view(torch.int32)).any(-1)


def _three_ways(args):
    """(chunks in order on one stream, chunks on parallel streams, the same under CUDA-graph replay)."""
    streams = SplitStreams()
    serial = op(*args)
    with with_multi_stream(True):
        parallel = op(*args, streams)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(*args, streams)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = op(*args, streams)
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    return serial, parallel, replayed


def _separated_inputs(rows: int, seed: int):
    """Mixed per-row sampling over logits whose filter cuts fall between well-separated tokens.

    Each row: 24 hot tokens on a geometric ladder (each ~0.8x the one above at any temperature used), a tail
    ~1e-11 below, distinct tail values so a top-k cut in the tail is a clean rank. Per row: temperature in
    [0.4, 1.5], greedy-sentinel rows, and top-k 50 / top-p 0.9 / min-p 0.02 each on about half the rows.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, VOCAB, generator=gen, device="cuda") * 0.5 - 30.0
    hot = torch.stack(
        [torch.randperm(VOCAB, generator=gen, device="cuda")[:24] for _ in range(rows)]
    )
    ladder = 8.0 - 0.35 * torch.arange(24, device="cuda", dtype=torch.float32)
    logits.scatter_(1, hot, ladder.expand(rows, 24).contiguous())
    temperatures = 0.4 + 1.1 * torch.rand(rows, generator=gen, device="cuda")
    greedy = torch.randperm(rows, generator=gen, device="cuda")[: max(1, rows // 8)]
    temperatures[greedy] = GREEDY_SENTINEL_T

    def half(on, off, dtype):
        pick = torch.rand(rows, generator=gen, device="cuda") < 0.5
        return torch.where(
            pick,
            torch.full((rows,), on, dtype=dtype, device="cuda"),
            torch.full((rows,), off, dtype=dtype, device="cuda"),
        )

    top_ks = half(50, DISABLE_TOP_K, torch.int32)
    top_ps = half(0.9, 1.0, torch.float32)
    min_ps = half(0.02, 0.0, torch.float32)
    return logits.contiguous(), temperatures.contiguous(), top_ks, top_ps, min_ps


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = sampling_inputs(cell.spec, seed_of(cell), fused=True)
    serial, parallel, replayed = _three_ways(args)
    expected = op.reference(*args)
    assert expected.dtype == torch.float32 and expected.shape == args[0].shape
    top_p_rows = args[3][: args[0].shape[0]] < 1.0
    for name, got in (("serial", serial), ("parallel", parallel), ("graph", replayed)):
        differ = _bits_differ(got, expected)
        strict = (differ & ~top_p_rows).nonzero().flatten()
        assert strict.numel() == 0, (
            f"{name}: rows {strict.tolist()} differ from their own one-row call"
        )
        l1 = row_l1(got, expected)[differ & top_p_rows]
        assert l1.numel() == 0 or l1.max().item() <= _ROW_L1, (
            f"{name}: top-p row L1 {l1.max().item():.3e}"
        )
    torch.testing.assert_close(
        serial.sum(-1), torch.ones(serial.shape[0], device="cuda"), atol=1e-5, rtol=0
    )


@pytest.mark.parametrize("rows", [33, 34, 40, 47, 63, 64, 65, 72, 80, 95, 96])
def test_rows_match_their_own_small_call(rows) -> None:
    """Each row equals its own one-row call and the same row in any other <= 32-row chunking, bit for bit."""
    args = _separated_inputs(rows, 7919 * rows)
    serial, parallel, replayed = _three_ways(args)
    assert len(split_row_chunks(rows)) <= MAX_CHUNKS
    expected = op.reference(*args)
    for name, got in (("serial", serial), ("parallel", parallel), ("graph", replayed)):
        bad = _bits_differ(got, expected).nonzero().flatten()
        assert bad.numel() == 0, f"{name}: rows {bad.tolist()} differ from their own one-row call"
    engine = torch.ops.trtllm.fused_compute_probs_from_logits
    third = -(-rows // 3)
    other = torch.cat(
        [engine(*(t[b : min(b + third, rows)] for t in args)) for b in range(0, rows, third)]
    )
    assert third <= SPLIT_MAX_ROWS
    assert not _bits_differ(serial, other).any(), (
        "a row's values must not depend on the rows beside it"
    )


@pytest.mark.parametrize("rows", [1, 32, MAX_CHUNKS * SPLIT_MAX_ROWS + 1, 128])
def test_outside_the_band_is_one_call(rows) -> None:
    # No top-p: a top-p cut on random logits is not run-to-run reproducible in the one-CTA family.
    spec = sampling_spec(rows, temperature=0.7, top_k=50, min_p=0.02, greedy_rows=1, mixed=True)
    args = sampling_inputs(spec, 31 * rows, fused=True)
    with with_multi_stream(True):
        got = op(*args, SplitStreams())
    want = torch.ops.trtllm.fused_compute_probs_from_logits(*args)
    assert not _bits_differ(got, want).any()
