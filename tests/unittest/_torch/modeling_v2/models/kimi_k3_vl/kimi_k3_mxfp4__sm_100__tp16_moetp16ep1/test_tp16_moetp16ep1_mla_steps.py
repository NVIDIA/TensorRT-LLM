# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp16_moetp16ep1 target's MLA layer (``mla.py``) against itself on one GPU, at one TP16 rank's shape
(``_kimi_k3_mla_steps``): each step run two ways from the same cache contents.

* Captured and replayed (breakable CUDA graph for prefill steps, CUDA graph for decode steps) against eager: the same
  kernels in the same order, so the output and the cache are bit for bit equal.
* Ways that run other kernels or another order, within the bf16 band (``BAND``, relative to the largest
  output): chunked prefill against the whole prefix in one piece (the partials merged on softmax statistics), a mixed
  step against its context and generation requests run apart (FlashInfer's MLA decode against the CuTe DSL decode,
  and the projections over other row counts), and the decode branch against the generic path (the K3 decode kernels
  against the CuTe DSL decode). The cache rows are bit for bit equal where the same kernels produce and append them
  (chunked prefill against the whole prefix), else within the same band.
"""

import pytest
import torch

__extra_import_path__ = [".."]
import _kimi_k3_mla_steps as ms  # noqa: E402
from _kimi_k3_parity import assert_bitwise, assert_within  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp16ep1 import (  # noqa: E402,E501
    mla,
    step,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="the Kimi K3 MLA kernels are certified on sm_100",
)

#: Bound for two ways that differ in kernel or order: max |a - b| <= BAND * max |b| (bf16 rounding of the partials).
BAND = 2e-2


@pytest.fixture
def layer():
    layer = ms.MlaLayer(mla)
    try:
        yield layer
    finally:
        layer.shutdown()


def _both(layer, first, second):
    """Run ``first`` and ``second`` (callables returning an output) from the same cache contents; returns their
    outputs and the cache after each."""
    start = layer.snapshot()
    out_a = first()
    pool_a = layer.snapshot()
    layer.restore(start)
    out_b = second()
    pool_b = layer.snapshot()
    return out_a, pool_a, out_b, pool_b


def test_breakable_graph_matches_eager(layer):
    """Fresh prefill, chunked prefill over a cached prefix and a mixed step, each captured in a breakable CUDA graph
    and replayed: the output and the cache are the eager step's, bit for bit."""
    lens = [0] * ms.NUM_REQUESTS
    for make in (ms.fresh_prefill, lambda _: ms.prefix_step(True), ms.mixed):
        s = make(lens) if make is not ms.fresh_prefill else make()
        hidden = layer.hidden(s)
        eager, pool_e, graph, pool_g = _both(
            layer, lambda: layer.run(s, hidden), lambda: layer.run_breakable_graph(s, hidden)
        )
        assert_bitwise(graph, eager, f"{s.name}: breakable graph output")
        assert_bitwise(pool_g, pool_e, f"{s.name}: breakable graph cache")
        lens = ms.advance(lens, s)


@pytest.mark.parametrize("ids,tokens,branch", [(list(range(12)), 1, False), ([0, 1, 2, 3], 1, True), ([4, 5], 8, True)],
                         ids=["generic 12x1", "decode branch 4x1", "decode branch 2x8"])  # fmt: skip
def test_cuda_graph_matches_eager(layer, ids, tokens, branch):
    """A decode step captured in a CUDA graph and replayed (the generic path at 12 requests, the decode branch at
    4 x 1 and 2 x 8): the output and the cache are the eager step's, bit for bit."""
    prefill = ms.fresh_prefill()
    layer.run(prefill, layer.hidden(prefill))
    s = ms.generation(ids, tokens, ms.advance([0] * ms.NUM_REQUESTS, prefill))
    dstep = step.DecodeStep(len(ids) * tokens, len(ids), tokens) if branch else None
    hidden = layer.hidden(s)
    eager, pool_e, graph, pool_g = _both(
        layer, lambda: layer.run(s, hidden, dstep), lambda: layer.run_cuda_graph(s, hidden, dstep)
    )
    assert_bitwise(graph, eager, f"{s.name}: CUDA graph output")
    assert_bitwise(pool_g, pool_e, f"{s.name}: CUDA graph cache")


def test_chunked_prefill_matches_whole_prefix(layer):
    """A context step over a cached prefix with chunked prefill (the prefix in 5 chunks, the partials merged on
    softmax statistics) against the same step reading the prefix in one piece: the output within BAND; the appended
    rows, from the same kernel, bit for bit."""
    prefill = ms.fresh_prefill()
    layer.run(prefill, layer.hidden(prefill))
    chunked, whole = ms.prefix_step(True), ms.prefix_step(False)
    hidden = layer.hidden(chunked)
    out_c, pool_c, out_w, pool_w = _both(
        layer, lambda: layer.run(chunked, hidden), lambda: layer.run(whole, hidden)
    )
    assert_within(out_c, out_w, BAND, "chunked prefill vs whole prefix")
    assert_bitwise(pool_c, pool_w, "chunked prefill vs whole prefix: cache")


def test_mixed_step_matches_separate_steps(layer):
    """A mixed step (one context request, ten generation requests) against its context request and then its
    generation requests run as two steps: each part, and the cache rows written, within BAND (the fused
    [q_a | kv_a | k_pe] projection runs over 30 rows against 20 and 10, and cuBLAS picks its kernel by row count, so
    the latent rows can differ in the last bf16 bit)."""
    prefill = ms.fresh_prefill()
    layer.run(prefill, layer.hidden(prefill))
    lens = ms.advance([0] * ms.NUM_REQUESTS, prefill)
    mix = ms.mixed(lens)
    ctx = ms.Step("context", [12], [20], [0], 1, [20])
    gen = ms.generation(list(range(10)), 1, lens)
    hidden = layer.hidden(mix)
    parts = {}

    def separate():
        parts["ctx"] = layer.run(ctx, hidden[:20].clone())
        parts["gen"] = layer.run(gen, hidden[20:].clone())
        return None

    out_m, pool_m, _, pool_s = _both(layer, lambda: layer.run(mix, hidden), separate)
    assert_within(out_m[:20], parts["ctx"], BAND, "mixed vs separate: context rows")
    assert_within(out_m[20:], parts["gen"], BAND, "mixed vs separate: generation rows")
    assert_within(pool_m, pool_s, BAND, "mixed vs separate: cache")


def test_decode_branch_matches_generic_path(layer):
    """A decode step of 4 requests on the decode branch (the K3 decode kernels) against the generic path (the CuTe
    DSL decode): the output and the appended rows within BAND."""
    prefill = ms.fresh_prefill()
    layer.run(prefill, layer.hidden(prefill))
    lens = ms.advance([0] * ms.NUM_REQUESTS, prefill)
    s = ms.generation([0, 1, 2, 3], 1, lens)
    dstep = step.DecodeStep(4, 4, 1)
    hidden = layer.hidden(s)
    out_d, pool_d, out_g, pool_g = _both(
        layer, lambda: layer.run(s, hidden, dstep), lambda: layer.run(s, hidden)
    )
    assert_within(out_d, out_g, BAND, "decode branch vs generic path")
    new_lens = [lens[i] + 1 for i in s.request_ids]
    layer.restore(pool_d)
    rows_d = ms.rows_of(layer, s.request_ids, new_lens)
    layer.restore(pool_g)
    rows_g = ms.rows_of(layer, s.request_ids, new_lens)
    assert_within(rows_d, rows_g, BAND, "decode branch vs generic path: cache rows")
