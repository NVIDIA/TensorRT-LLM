# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stock producer -> shared_indexer_absorbed_query -> native DSA attention, through the target's
attention (the absorbed-query packet of absorbed_query, draft_absorbed_query and
shared_indexer_absorbed_query).

Uses the absorbed_query integration harness
(``absorbed_query/test_absorbed_query_attention_integration .py``): real weights of the full-indexer
layer 6 and the shared layer 7 at TP8 rank 0, six requests natively prefilled with 2,600 tokens, FP8
KV cache, generation steps in the 6 x 1 and the 1 x 6 (MTP verify) layouts, both call
paths, every step preceded by ``on_update_kv_lens()`` as in the runner.

With the fast paths ``absorbed_query,shared_indexer_absorbed_query`` running, layer 7 takes
shared_indexer_absorbed_query (native kv_a GEMM + the shared_indexer_absorbed_query
kernel, preabsorbed packet, no native BMM) and layer 6 takes absorbed_query.
Checks against the same step with every fast path off:

* engagement: shared_indexer_absorbed_query serves layer 7 only; absorbed_query serves layer 6 only;
* layer 7 (shared_indexer_absorbed_query): its latent cache is **byte-identical** to native, and the
  attention inputs recorded at the native ``mla_rope_generation`` (absorbed
  query, FP8 ``quant_q_buffer``, BMM scales) and the converted sparse
  indices are bit-identical / select the same keys;
* layer 6 (absorbed_query): bit-identical caches, attention and indexer inputs and
  equal top-k key sets (the absorbed_query harness's comparison);
* outputs within the native-noise bound; a repeated shared_indexer_absorbed_query step and 20 more
  steps reproduce the shared_indexer_absorbed_query caches and recorded inputs bit for bit;
* CUDA graphs (shared_indexer_absorbed_query alone, native alone, both in one pool): replays 1, 1
  and 20 times with fresh hidden states reproduce the eager shared_indexer_absorbed_query step bit
  for bit and compare with native as above; counters are zero after each trial;
* feature gating through the target's ``_absorbed_query_packet``: ``""``, ``absorbed_query``,
  ``shared_indexer_absorbed_query``, ``absorbed_query,shared_indexer_absorbed_query``,
  ``absorbed_query,draft_absorbed_query,shared_indexer_absorbed_query``;
* negative control: a shared_indexer_absorbed_query packet whose absorbed query is zeroed is
  flagged.
"""

import math
import os
import sys

import absorbed_query_fixture
import pytest
import test_absorbed_query_attention_integration as aq_integration
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import plan_testlib  # noqa: E402

LAYERS = aq_integration.LAYERS
SHARED = absorbed_query_fixture.SHARED_LAYER
FULL = absorbed_query_fixture.FULL_LAYER
LAYOUTS = aq_integration.LAYOUTS

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]


def _set_features(features: str, form: str = "auto"):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    query_proj.SERVED_FORM = form
    plan_testlib.select(dispatch, absorbed_query_fixture.with_base_features(features))


def _reset_runners(h):
    for layer in LAYERS:
        h.attn[layer]._absorbed_query = None
        h.attn[layer]._shared_indexer_query = None


def _runners(h):
    return {
        layer: (
            getattr(h.attn[layer], "_absorbed_query", None),
            getattr(h.attn[layer], "_shared_indexer_query", None),
        )
        for layer in LAYERS
    }


def _assert_engaged(h):
    r = _runners(h)
    assert r[SHARED][1] not in (None, False), "shared_indexer_absorbed_query did not engage on L7"
    assert r[SHARED][0] is None, (
        "absorbed_query must not run on the shared_indexer_absorbed_query layer"
    )
    assert r[FULL][1] is False, "shared_indexer_absorbed_query must decline the full layer"
    assert r[FULL][0] not in (None, False), "absorbed_query did not engage on L6"


@pytest.fixture(scope="module")
def harness():
    with torch.inference_mode():
        h = aq_integration._Harness()
    yield h
    _set_features("")


def _compare_shared_query_to_native(snap, native, cand, n_new, bounds, report, tag):
    """absorbed_query harness comparison (bitwise on both layers) plus an explicit check
    of the shared_indexer_absorbed_query layer's latent cache."""
    fails = aq_integration._compare_absorbed_query_to_native(
        snap, native, cand, n_new, bounds, report, tag
    )
    n_diff = report.get(f"{tag}:latent{SHARED}_bytes_differing")
    if n_diff != 0:
        fails.append(
            f"{tag}: shared_indexer_absorbed_query layer latent cache differs from native in "
            f"{n_diff} bytes"
        )
    return fails


@pytest.mark.parametrize("path", ["custom_op", "forward_impl"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_shared_query_into_native_attention(harness, layout, path):
    h = harness
    n_new = math.prod(LAYOUTS[layout])
    hidden, pos = h.gen_inputs(layout, seed=12)
    snap = h.snapshot()
    try:
        native_outs, native_recs, native_cache = aq_integration._native_runs(
            h, layout, path, hidden, pos, snap
        )
        bounds = aq_integration._output_bounds(native_outs)

        _set_features("absorbed_query,shared_indexer_absorbed_query")
        _reset_runners(h)
        shared_query_out, shared_query_rec = h.recorded_step(layout, path, hidden, pos)
        _assert_engaged(h)
        shared_query_cache = h.snapshot()
        rope_served = {layer: "rope_append_rope" in shared_query_rec[layer] for layer in LAYERS}
        print(f"rope_kv_append rope served: {rope_served}", flush=True)
        if path == "custom_op" and "rope_kv_append" in absorbed_query_fixture.with_base_features(
            ""
        ).split(","):
            assert rope_served[SHARED], (
                "rope_kv_append did not serve the shared_indexer_absorbed_query layer's rope"
            )

        report, fails = {}, []
        fails += _compare_shared_query_to_native(
            snap,
            (native_outs[0], native_recs[0], native_cache),
            (shared_query_out, shared_query_rec, shared_query_cache),
            n_new,
            bounds,
            report,
            "shared_indexer_absorbed_query",
        )

        h.restore(snap)
        _, rec_again = h.recorded_step(layout, path, hidden, pos)
        fails += aq_integration._compare_caches(
            snap, shared_query_cache, h.snapshot(), n_new, report, "shared_query_again"
        )
        fails += aq_integration._compare_records(
            shared_query_rec, rec_again, report, "shared_query_again"
        )
        for _ in range(20):
            again = h.gen_step(layout, path, hidden, pos)
        fails += aq_integration._compare_caches(
            snap, shared_query_cache, h.snapshot(), n_new, report, "shared_query_repeat20"
        )
        fails += aq_integration._compare_outputs(
            native_outs[0], again, bounds, report, "shared_query_repeat20"
        )
        assert h.attn[SHARED]._shared_indexer_query.counters_quiescent()
        print(f"layout={layout} path={path} {report}")
        assert not fails, fails
    finally:
        h.restore(snap)
        _set_features("")
        _reset_runners(h)


@pytest.mark.parametrize(
    "features,expect",
    [
        ("", {FULL: (None, None), SHARED: (None, None)}),
        ("absorbed_query", {FULL: ("absorbed_query", None), SHARED: ("absorbed_query", None)}),
        ("shared_indexer_absorbed_query", {FULL: (None, None), SHARED: (None, None)}),
        (
            "absorbed_query,shared_indexer_absorbed_query",
            {FULL: ("absorbed_query", False), SHARED: (None, "shared_indexer_absorbed_query")},
        ),
        (
            "absorbed_query,draft_absorbed_query,shared_indexer_absorbed_query",
            {FULL: ("absorbed_query", False), SHARED: (None, "shared_indexer_absorbed_query")},
        ),
    ],
)
@torch.inference_mode()
def test_feature_gating_through_adapter(harness, features, expect):
    """Which runner serves which layer for each feature selection (the draft_absorbed_query
    entry point declines target layers, so ``draft_absorbed_query`` changes nothing here).
    ``shared_indexer_absorbed_query`` without ``absorbed_query`` is refused by the selection itself
    (``_set_features``), before any runner is attached, when the installed selector has the
    dependency rule (``REQUIRED_FEATURES``); without that rule it runs native."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
        shared_indexer,
    )

    h = harness
    layout, path = "1req_x6tok", "custom_op"
    hidden, pos = h.gen_inputs(layout, seed=21)
    snap = h.snapshot()
    ids = features.split(",")
    rejected = (
        "shared_indexer_absorbed_query" in ids
        and "absorbed_query" not in ids
        and "shared_indexer_absorbed_query" in getattr(dispatch, "REQUIRED_FEATURES", {})
    )
    try:
        if rejected:
            _reset_runners(h)
            with pytest.raises(ValueError, match="which they extend"):
                _set_features(features)
            assert _runners(h) == {layer: (None, None) for layer in LAYERS}, _runners(h)
            return
        _set_features(features)
        _reset_runners(h)
        h.gen_step(layout, path, hidden, pos)
        got = _runners(h)
        for layer, (want_absorbed_query, want_shared_query) in expect.items():
            absorbed_query_r, shared_query_r = got[layer]
            if want_absorbed_query == "absorbed_query":
                assert isinstance(absorbed_query_r, query_proj.AbsorbedQueryProjection), (
                    features,
                    layer,
                    got,
                )
            else:
                assert absorbed_query_r is want_absorbed_query, (features, layer, got)
            if want_shared_query == "shared_indexer_absorbed_query":
                assert isinstance(shared_query_r, shared_indexer.SharedIndexerProjection), (
                    features,
                    layer,
                    got,
                )
            else:
                assert shared_query_r is want_shared_query, (features, layer, got)
    finally:
        h.restore(snap)
        _set_features("")
        _reset_runners(h)


GRAPH_SETS = {
    "w10only": ("shared_indexer_absorbed_query",),
    "nativeonly": ("native",),
    "both": ("shared_indexer_absorbed_query", "native"),
}


@pytest.mark.parametrize("graph_set", list(GRAPH_SETS))
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_shared_query_cuda_graph_replay_into_native_attention(harness, layout, graph_set):
    """Captured shared_indexer_absorbed_query (+ absorbed_query on layer 6) + native attention + native cache
    append, replayed once and 20 times with fresh inputs (the absorbed_query harness's
    graph protocol: warm-up pass, capture into one shared pool, the runner's
    on_update_kv_lens() inside every step)."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    h = harness
    n_new = math.prod(LAYOUTS[layout])
    path = "custom_op"
    snap = h.snapshot()
    graphs = {}
    wanted = GRAPH_SETS[graph_set]
    configs = [
        (
            name,
            "absorbed_query,shared_indexer_absorbed_query"
            if name == "shared_indexer_absorbed_query"
            else "",
        )
        for name in ("shared_indexer_absorbed_query", "native")
        if name in wanted
    ]
    try:
        warm = {}
        for name, features in configs:
            _set_features(features)
            _reset_runners(h)
            meta = h.gen_metadata(layout, cuda_graph=True)
            static_hidden, static_pos = h.gen_inputs(layout, seed=200)
            with with_multi_stream(True):
                for _ in range(2):
                    h.step(meta, static_pos, static_hidden, path)
            torch.cuda.synchronize()
            if name == "shared_indexer_absorbed_query":
                _assert_engaged(h)
            warm[name] = (meta, static_hidden, static_pos, _runners(h))
        h.restore(snap)
        pool = None
        for name, features in configs:
            meta, static_hidden, static_pos, runners = warm[name]
            _set_features(features)
            for layer, (absorbed_query_r, shared_query_r) in runners.items():
                h.attn[layer]._absorbed_query = absorbed_query_r
                h.attn[layer]._shared_indexer_query = shared_query_r
            graph = torch.cuda.CUDAGraph()
            with with_multi_stream(True), aq_integration._Recorder(h.attn) as rec:
                with torch.cuda.graph(graph, pool=pool):
                    outs = h.step(meta, static_pos, static_hidden, path)
            pool = graph.pool()
            graphs[name] = (graph, meta, static_hidden, outs, rec, runners)
            torch.cuda.synchronize()
        _reset_runners(h)

        report, fails = {}, []
        for trial, (seed, replays) in enumerate([(301, 1), (302, 1), (303, 20)]):
            hidden, pos = h.gen_inputs(layout, seed=seed)
            native_outs, native_recs, native_cache = aq_integration._native_runs(
                h, layout, path, hidden, pos, snap
            )
            bounds = aq_integration._output_bounds(native_outs)
            _set_features("absorbed_query,shared_indexer_absorbed_query")
            h.restore(snap)
            shared_query_eager = h.recorded_step(layout, path, hidden, pos)
            shared_query_eager_cache = h.snapshot()
            _set_features("")
            replayed = {}
            for name, (graph, meta, static_hidden, outs, rec, runners) in graphs.items():
                h.restore(snap)
                meta.prepare()
                static_hidden.copy_(hidden)
                for _ in range(replays):
                    graph.replay()
                torch.cuda.synchronize()
                replayed[name] = (
                    {layer: outs[layer].clone() for layer in LAYERS},
                    rec.snapshot(),
                    h.snapshot(),
                )
            tag = f"t{trial}x{replays}"
            if "shared_indexer_absorbed_query" in replayed:
                g_out, g_rec, g_cache = replayed["shared_indexer_absorbed_query"]
                fails += aq_integration._compare_caches(
                    snap,
                    shared_query_eager_cache,
                    g_cache,
                    n_new,
                    report,
                    f"{tag}:shared_query_graph_vs_eager",
                )
                fails += aq_integration._compare_records(
                    shared_query_eager[1], g_rec, report, f"{tag}:shared_query_graph_vs_eager"
                )
                fails += _compare_shared_query_to_native(
                    snap,
                    (native_outs[0], native_recs[0], native_cache),
                    (g_out, g_rec, g_cache),
                    n_new,
                    bounds,
                    report,
                    f"{tag}:shared_query_graph_vs_native",
                )
            if "native" in replayed:
                g_out, g_rec, g_cache = replayed["native"]
                fails += aq_integration._compare_caches(
                    snap, native_cache, g_cache, n_new, report, f"{tag}:native_graph_vs_eager"
                )
                fails += aq_integration._compare_records(
                    native_recs[0], g_rec, report, f"{tag}:native_graph_vs_eager"
                )
            for _, (graph, meta, sh, outs, rec, runners) in graphs.items():
                for absorbed_query_r, shared_query_r in runners.values():
                    if absorbed_query_r not in (None, False):
                        assert int(absorbed_query_r.sync.abs().sum()) == 0
                    if shared_query_r not in (None, False):
                        assert shared_query_r.counters_quiescent()
        print(f"graph layout={layout} graphs={wanted} {report}")
        assert not fails, fails
    except BaseException as exc:
        print(f"graph test failure: {exc!r}", flush=True)
        raise
    finally:
        graphs.clear()
        _reset_runners(h)
        h.restore(snap)
        _set_features("")


@torch.inference_mode()
def test_checks_detect_a_wrong_shared_query_absorbed_query(harness, monkeypatch):
    """Negative control: zeroing the shared_indexer_absorbed_query absorbed query must be flagged by
    the absorbed-query and quant_q checks of layer 7."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        shared_indexer,
    )

    h = harness
    layout, path = "1req_x6tok", "custom_op"
    hidden, pos = h.gen_inputs(layout, seed=12)
    snap = h.snapshot()
    orig = shared_indexer.SharedIndexerProjection.project

    def broken(self, hidden_states):
        packet = orig(self, hidden_states)
        packet.fused_q[..., : self.mla.kv_lora_rank].zero_()
        return packet

    try:
        native_outs, native_recs, _ = aq_integration._native_runs(
            h, layout, path, hidden, pos, snap
        )
        monkeypatch.setattr(shared_indexer.SharedIndexerProjection, "project", broken)
        _set_features("absorbed_query,shared_indexer_absorbed_query")
        _reset_runners(h)
        _, bad_rec = h.recorded_step(layout, path, hidden, pos)
        report = {}
        fails = aq_integration._compare_records(native_recs[0], bad_rec, report, "bad")
        print(f"negative control {report}")
        assert any(f"layer {SHARED} absorbed_q" in f for f in fails), fails
        assert any(f"layer {SHARED} quant_q" in f for f in fails), fails
        assert not any(f"layer {FULL} absorbed_q" in f for f in fails), fails
    finally:
        monkeypatch.undo()
        h.restore(snap)
        _set_features("")
        _reset_runners(h)
