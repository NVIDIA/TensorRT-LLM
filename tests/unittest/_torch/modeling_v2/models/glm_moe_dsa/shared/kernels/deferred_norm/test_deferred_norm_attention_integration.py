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
"""deferred_norm packet -> layer L+1 DSA attention through the target's adopted attention
(absorbed_query, draft_absorbed_query, shared_indexer_absorbed_query, post_moe_hop and
deferred_norm; single GPU; generation steps run as decode steps).

Uses the absorbed_query integration harness (real weights of the full-indexer layer 6
and the shared layer 7 at TP8 rank 0, six requests natively prefilled with
2,600 tokens, FP8 KV cache, layouts 6 x 1 and 1 x 6 (MTP verify), both call
paths, ``on_update_kv_lens()`` before every step). Each layer gets its own
input: residual-like rows ``x`` and that layer's real ``input_layernorm``
gain; the deferred_norm packet carries ``a = bf16(x * gamma)`` and the hop-order
square-sum partials, the native input is ``h = materialize(packet)`` (the post_moe_hop
hop's output bit for bit). The target's deferred-norm scope is ``shared``: the hop defers
the norm only into shared-indexer layers, so a deferred_norm step gives layer 7 the packet
and layer 6 the normalized ``h`` (``SERVED_LAYERS``, ``_served``).

* deviation (report + sanity bound): relative RMS vs the native
  step of the attention inputs recorded at ``mla_rope_generation`` and the
  indexer, top-k set agreement per row (identical rows, mean Jaccard), and
  the attention output. Reported next to two native-only references: a
  rerun of the native step (run-to-run noise) and the *rounding control*,
  the native step on ``bf16(bf16(x * gamma) * r)`` (the normalized hidden
  state with the gain applied first, a one-rounding perturbation of the
  native input in native code only). Bound: output relRMS <= max(2**-7,
  2 * control) and mean Jaccard >= 0.98 on every layer (2**-7 alone is
  below the control: the attention output amplifies rounding-level input
  differences through the FP8 query and the sparse softmax);
* fallback exactness: with the deferred_norm projection declined (``_deferred_norm_projection =
  False``) the packet is materialized and every cache, recorded input and
  output equals the native step bit for bit;
* engagement: the deferred_norm projection serves layer 7 and is never asked on layer 6,
  shared_indexer_absorbed_query serves layer 7 under
  ``absorbed_query,shared_indexer_absorbed_query,deferred_norm`` and absorbed_query serves layer 6;
* CUDA graph: a captured deferred_norm step replays (1, 1, 20 times with fresh
  inputs) bit-identically to the eager deferred_norm step;
* a projection compile error on layer 7 propagates; on layer 6 the projection declines
  before any compile;
* negative control: a packet whose ``a_frag`` lacks the gain (``bf16(x)``
  instead of ``bf16(x * gamma)``) violates the bound.

Fallback and graph comparisons of attention outputs whose inputs and caches
are bitwise equal allow 4 bf16 ulps of the output scale: the native sparse
attention is not bitwise run-to-run stable.

Every arm (native reference included) also runs the fast paths in
``GLM_MOE_DSA_TEST_BASE_FEATURES`` (e.g. ``rope_kv_append``), as the absorbed_query harness
does.
"""

import os
import sys

import absorbed_query_fixture
import deferred_norm_utils as U
import pytest
import test_absorbed_query_attention_integration as aq_integration
import torch

LAYERS = aq_integration.LAYERS
SHARED = absorbed_query_fixture.SHARED_LAYER
FULL = absorbed_query_fixture.FULL_LAYER
LAYOUTS = aq_integration.LAYOUTS
# The layers the deferred_norm projection serves: the shared-indexer ones.
SERVED_LAYERS = (SHARED,)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import plan_testlib  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]


def _set_features(features: str):
    """Run exactly the fast paths ``features`` plus ``GLM_MOE_DSA_TEST_BASE_FEATURES``.

    The latter are the neighbour paths every arm runs on, e.g.
    ``rope_kv_append`` on an integration image (absorbed_query harness rule).
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.absorbed_query import (  # noqa: E501
        query_proj,
    )

    query_proj.SERVED_FORM = "auto"
    plan_testlib.select(dispatch, absorbed_query_fixture.with_base_features(features))


def _served(packets):
    """The inputs of a deferred_norm step: the packet on the layers the projection serves, the
    normalized hidden state (the post-MoE hop's output) on the others."""
    return {layer: p if layer in SERVED_LAYERS else p.materialize() for layer, p in packets.items()}


def _reset_runners(h):
    for layer in LAYERS:
        for name in ("_absorbed_query", "_shared_indexer_query", "_deferred_norm_projection"):
            setattr(h.attn[layer], name, None)


class _DeferredNormHarness(aq_integration._Harness):
    """absorbed_query harness whose layers take per-layer inputs (tensor or deferred_norm packet)."""

    def __init__(self):
        super().__init__()
        self.gammas = {
            layer: absorbed_query_fixture.load_input_layernorm(layer).cuda() for layer in LAYERS
        }

    def forward(self, layer, positions, hidden, meta, path):
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
            is_deferred,
        )

        view = hidden.residual if is_deferred(hidden) else hidden
        attn = self.attn[layer]
        if path == "forward_impl":
            out = attn._create_outputs(view, meta)
            attn.forward_impl(positions, hidden, meta, attn_output=out)
            return out[0]
        import weakref

        from tensorrt_llm._torch.utils import model_extra_attrs

        extra = self.model_config.extra_attrs
        extra["attention_metadata"] = weakref.ref(meta)
        with model_extra_attrs(extra):
            out = [torch.ops.trtllm.create_mla_outputs(view, attn.layer_idx_str)]
            attn._forward_custom_op(hidden, positions, out, None)
        return out[0]

    def step(self, meta, positions, hidden, path):
        """Per-layer inputs when ``hidden`` is a dict (layer -> tensor or
        packet); otherwise the same tensor for every layer (native prefill). A
        generation-only step runs as a decode step."""
        from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
            dispatch,
        )

        phase = dispatch.Phase.DECODE if meta.num_contexts == 0 else dispatch.Phase.PREFILL
        meta.on_update_kv_lens()
        with dispatch.step(phase):
            return {
                layer: self.forward(
                    layer,
                    positions,
                    hidden[layer] if isinstance(hidden, dict) else hidden,
                    meta,
                    path,
                )
                for layer in LAYERS
            }

    def packets(self, layout, seed, scale=1.0):
        nreq, ntok = LAYOUTS[layout]
        out = {}
        for layer in LAYERS:
            g = torch.Generator(device="cuda").manual_seed(seed * 10 + layer)
            x = torch.randn(nreq * ntok, U.H, generator=g, device="cuda") * 2.0
            x = (x * scale).bfloat16()
            out[layer] = U.make_frag_packet(x, self.gammas[layer])
        pos = (
            (aq_integration.CTX + torch.arange(ntok, dtype=torch.int32, device="cuda"))
            .repeat(nreq)
            .unsqueeze(0)
        )
        return out, pos


@pytest.fixture(scope="module")
def harness():
    with torch.inference_mode():
        h = _DeferredNormHarness()
    yield h
    _set_features("")


def _run(h, features, layout, path, inputs, pos, snap):
    _set_features(features)
    h.restore(snap)
    out, rec = h.recorded_step(layout, path, inputs, pos)
    cache = h.snapshot()
    return out, rec, cache


def _jaccard_rows(a, b):
    rows = []
    for ra, rb in zip(a.tolist(), b.tolist()):
        sa = {v for v in ra if v >= 0}
        sb = {v for v in rb if v >= 0}
        union = len(sa | sb)
        rows.append(1.0 if union == 0 else len(sa & sb) / union)
    return rows


def _output_stats(native, cand, layout):
    """Deviation statistics per layer."""
    n_out, n_rec, _ = native
    c_out, c_rec, _ = cand
    stats = {}
    for layer in LAYERS:
        s = {"out_rel_rms": aq_integration._rel_rms(c_out[layer], n_out[layer])}
        nr, cr = n_rec[layer], c_rec[layer]
        if "absorbed_q" in nr:
            s["absorbed_q_rel_rms"] = aq_integration._rel_rms(cr["absorbed_q"], nr["absorbed_q"])
        if "quant_q" in nr:
            s["quant_q_bytes_diff_frac"] = float(
                (cr["quant_q"].view(torch.uint8) != nr["quant_q"].view(torch.uint8)).float().mean()
            )
        if "indexer_weights" in nr:
            s["indexer_weights_rel_rms"] = aq_integration._rel_rms(
                cr["indexer_weights"], nr["indexer_weights"]
            )
            s["indexer_q_fp8_bytes_diff_frac"] = float(
                (cr["indexer_q_fp8"].view(torch.uint8) != nr["indexer_q_fp8"].view(torch.uint8))
                .float()
                .mean()
            )
        if "topk" in nr:
            jac = _jaccard_rows(cr["topk"], nr["topk"])
            s["topk_rows_identical"] = sum(1 for j in jac if j == 1.0) / len(jac)
            s["topk_mean_jaccard"] = sum(jac) / len(jac)
        stats[layer] = s
    return stats


def _control_inputs(packets):
    """Rounding control: ``bf16(bf16(x * gamma) * r)`` per layer (native code
    consumes it; ``r`` in torch from the hop-order square sum)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        rows_from_frag,
    )

    out = {}
    for layer, p in packets.items():
        rows, hidden = p.residual.shape
        a = rows_from_frag(p.a_frag, rows, hidden)
        r = torch.rsqrt(U.square_sum_in_order(p.sumsq) / hidden + p.eps)
        out[layer] = (a.float() * r.unsqueeze(-1)).bfloat16()
    return out


def _check_output_bound(stats, control):
    fails = []
    for layer, s in stats.items():
        bound = max(2.0**-7, 2.0 * control[layer]["out_rel_rms"])
        if s["out_rel_rms"] > bound:
            fails.append(f"L{layer} output relRMS {s['out_rel_rms']:.3e} > {bound:.3e}")
        if s.get("topk_mean_jaccard", 1.0) < 0.98:
            fails.append(f"L{layer} top-k mean Jaccard {s['topk_mean_jaccard']:.4f} < 0.98")
    return fails


@pytest.mark.parametrize(
    "features",
    [
        "absorbed_query,post_moe_hop,deferred_norm",
        "absorbed_query,post_moe_hop,shared_indexer_absorbed_query,deferred_norm",
    ],
)
@pytest.mark.parametrize("path", ["custom_op", "forward_impl"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_deferred_norm_packet_into_attention(harness, layout, path, features):
    h = harness
    packets, pos = h.packets(layout, seed=3)
    native_in = {layer: p.materialize() for layer, p in packets.items()}
    snap = h.snapshot()
    try:
        base = ",".join(f for f in features.split(",") if f != "deferred_norm")
        _reset_runners(h)
        native = _run(h, base, layout, path, native_in, pos, snap)
        _reset_runners(h)
        cand = _run(h, features, layout, path, _served(packets), pos, snap)
        for layer in LAYERS:
            engaged = h.attn[layer]._deferred_norm_projection not in (None, False)
            assert engaged == (layer in SERVED_LAYERS), f"deferred_norm on L{layer}: {engaged}"
        if "shared_indexer_absorbed_query" in features:
            assert h.attn[SHARED]._shared_indexer_query not in (None, False)
        assert h.attn[FULL]._absorbed_query not in (None, False)
        _reset_runners(h)
        rerun = _run(h, base, layout, path, native_in, pos, snap)
        _reset_runners(h)
        control = _run(h, base, layout, path, _control_inputs(packets), pos, snap)
        stats = _output_stats(native, cand, layout)
        noise = _output_stats(native, rerun, layout)
        ctrl = _output_stats(native, control, layout)
        tag = f"layout={layout} path={path} features={features}"
        print(f"GLM_MOE_DSA_DEFERRED_NORM_OUTPUT_BOUND {tag} {stats}", flush=True)
        print(f"GLM_MOE_DSA_DEFERRED_NORM_OUTPUT_NOISE {tag} {noise}", flush=True)
        print(f"GLM_MOE_DSA_DEFERRED_NORM_OUTPUT_CONTROL {tag} {ctrl}", flush=True)
        fails = _check_output_bound(stats, ctrl)
        assert not fails, fails
    finally:
        h.restore(snap)
        _set_features("")


@pytest.mark.parametrize("path", ["custom_op", "forward_impl"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_fallback_materialize_is_exact(harness, layout, path):
    """deferred_norm projection declined -> materialized packet -> the absorbed_query path, bit for
    bit equal to the native-input step."""
    h = harness
    packets, pos = h.packets(layout, seed=5)
    native_in = {layer: p.materialize() for layer, p in packets.items()}
    snap = h.snapshot()
    try:
        _reset_runners(h)
        n_out, n_rec, n_cache = _run(h, "absorbed_query", layout, path, native_in, pos, snap)
        _reset_runners(h)
        for layer in LAYERS:
            h.attn[layer]._deferred_norm_projection = False
        _set_features("absorbed_query,post_moe_hop,deferred_norm")
        h.restore(snap)
        c_out, c_rec = h.recorded_step(layout, path, packets, pos)
        c_cache = h.snapshot()
        for key in n_cache:
            assert torch.equal(c_cache[key], n_cache[key]), key
        for layer in LAYERS:
            for key, val in n_rec[layer].items():
                if key in ("topk", "sparse_indices"):
                    continue  # native top-k order is not run-to-run stable
                assert torch.equal(c_rec[layer][key], val), (layer, key)
            if "topk" in n_rec[layer]:
                assert min(_jaccard_rows(c_rec[layer]["topk"], n_rec[layer]["topk"])) == 1.0
        noise_runs = [n_out]
        _reset_runners(h)
        noise_runs.append(_run(h, "absorbed_query", layout, path, native_in, pos, snap)[0])
        for layer in LAYERS:
            diff = (c_out[layer].float() - n_out[layer].float()).abs().max().item()
            noise = (noise_runs[1][layer].float() - n_out[layer].float()).abs().max().item()
            ulp = aq_integration._bf16_ulp(n_out[layer].float().abs().max().item())
            assert diff <= max(2 * noise, 4 * ulp), (layer, diff, noise, ulp)
    finally:
        h.restore(snap)
        _set_features("")


@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_cuda_graph_replay(harness, layout):
    h = harness
    snap = h.snapshot()
    try:
        _set_features("absorbed_query,post_moe_hop,shared_indexer_absorbed_query,deferred_norm")
        _reset_runners(h)
        packets, pos = h.packets(layout, seed=7)
        static = {}
        for layer, p in _served(packets).items():
            if layer in SERVED_LAYERS:
                st = U.empty_like_packet(p)
                st.a_frag.copy_(p.a_frag)
                st.sumsq.copy_(p.sumsq)
            else:
                st = p.clone()
            static[layer] = st
        static_pos = pos.clone()
        meta = h.gen_metadata(layout, cuda_graph=True)
        h.restore(snap)
        h.step(meta, static_pos, static, "custom_op")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        h.restore(snap)
        with torch.cuda.graph(graph):
            g_out = h.step(meta, static_pos, static, "custom_op")
        for trial, replays in enumerate((1, 1, 20)):
            fresh, _ = h.packets(layout, seed=100 + trial)
            fresh = _served(fresh)
            for layer in LAYERS:
                if layer in SERVED_LAYERS:
                    static[layer].a_frag.copy_(fresh[layer].a_frag)
                    static[layer].sumsq.copy_(fresh[layer].sumsq)
                else:
                    static[layer].copy_(fresh[layer])
            for _ in range(replays):
                h.restore(snap)
                graph.replay()
            torch.cuda.synchronize()
            got = {layer: g_out[layer].clone() for layer in LAYERS}
            got_cache = h.snapshot()
            h.restore(snap)
            eager = h.gen_step(
                layout, "custom_op", fresh, pos, meta=h.gen_metadata(layout, cuda_graph=True)
            )
            eager_cache = h.snapshot()
            for key in eager_cache:
                assert torch.equal(got_cache[key], eager_cache[key]), (trial, key)
            for layer in LAYERS:
                diff = (got[layer].float() - eager[layer].float()).abs().max().item()
                ref = eager[layer].float().abs().max().item()
                # Native sparse attention is not bitwise run-to-run stable:
                # bound by 4 bf16 ulps of the output scale.
                assert diff <= 4 * aq_integration._bf16_ulp(ref), (trial, layer, diff)
    finally:
        h.restore(snap)
        _set_features("")


@torch.inference_mode()
def test_projection_compile_error_is_not_masked(harness, monkeypatch):
    """A projection compile error in the consumer propagates (it invalidates
    the run) instead of a rank-local decline to the materialized path."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        projection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.query_entry import (  # noqa: E501
        deferred_norm_dsa_proj,
    )

    h = harness
    packets, pos = h.packets("1req_x6tok", seed=11)

    def _fail(self):
        raise RuntimeError("injected projection compile failure")

    monkeypatch.setattr(projection.DeferredNormProjection, "precompile", _fail)
    try:
        _set_features("absorbed_query,post_moe_hop,deferred_norm")
        _reset_runners(h)
        for layer in SERVED_LAYERS:
            with pytest.raises(RuntimeError, match="injected projection compile failure"):
                deferred_norm_dsa_proj(h.attn[layer], pos, packets[layer], None)
            assert h.attn[layer]._deferred_norm_projection is None, (
                "a failed runner must not be cached"
            )
        # A full-indexer layer keeps its native projection: declined before any compile.
        assert deferred_norm_dsa_proj(h.attn[FULL], pos, packets[FULL], None) is None
        assert h.attn[FULL]._deferred_norm_projection is False
    finally:
        _reset_runners(h)
        _set_features("")


@torch.inference_mode()
def test_negative_control_missing_gain(harness):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHidden,
        materialize,
    )

    h = harness
    layout = "1req_x6tok"
    packets, pos = h.packets(layout, seed=9)
    native_in = {layer: p.materialize() for layer, p in packets.items()}
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        frag_from_rows,
    )

    bad = {
        layer: DeferredNormHidden(
            frag_from_rows(p.residual), p.sumsq, p.residual, p.gamma, U.EPS, materialize
        )
        for layer, p in packets.items()
    }
    snap = h.snapshot()
    try:
        _reset_runners(h)
        native = _run(h, "absorbed_query", layout, "custom_op", native_in, pos, snap)
        _reset_runners(h)
        cand = _run(
            h, "absorbed_query,post_moe_hop,deferred_norm", layout, "custom_op", bad, pos, snap
        )
        _reset_runners(h)
        control = _run(
            h, "absorbed_query", layout, "custom_op", _control_inputs(packets), pos, snap
        )
        stats = _output_stats(native, cand, layout)
        ctrl = _output_stats(native, control, layout)
        print(f"GLM_MOE_DSA_DEFERRED_NORM_OUTPUT_NEGATIVE {stats}", flush=True)
        print(f"GLM_MOE_DSA_DEFERRED_NORM_OUTPUT_NEGATIVE_CONTROL {ctrl}", flush=True)
        assert _check_output_bound(stats, ctrl), (
            "the output bound must flag a packet without the gain"
        )
    finally:
        h.restore(snap)
        _set_features("")
