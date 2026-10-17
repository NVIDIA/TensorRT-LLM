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
"""fused_kv_indexer_projection through the target's absorbed-query packet into the native DSA
attention (single GPU; the target's adopted attention, generation steps as decode steps).

Uses the absorbed_query integration harness (real weights of the full-indexer layer 6
and the shared layer 7 at TP8 rank 0, six requests natively prefilled with
2,600 tokens, FP8 KV cache, layouts 6 x 1 and 1 x 6 (MTP verify), both call
paths, ``on_update_kv_lens()`` before every step). The same native
normalized hidden state goes through ``absorbed_query,shared_indexer_absorbed_query``
(absorbed_query / shared_indexer_absorbed_query with their native GEMM bridges) and
``absorbed_query,shared_indexer_absorbed_query,fused_kv_indexer_projection``
(fused_kv_indexer_projection serves both layers: ``entry.DEFAULT_LAYERS`` set to
``full,shared``).

Checks (the fused_kv_indexer_projection contract and the attention-level sanity bound
declared with it):

* engagement: fused_kv_indexer_projection serves both layers;
* every latent-cache byte and every recorded tensor derived only from
  ``kva`` (absorbed / FP8 query, BMM scales, indexer FP8 query) is
  bit-identical; the indexer-K cache, indexer weights and top-k sets are
  bit-identical when ``kw`` is, else reported and bounded (top-k mean
  Jaccard >= 0.98, output relRMS <= 2**-7);
* attention outputs: within ``max(2 * native rerun noise, 4 bf16 ulps)``
  (the native sparse attention is not bitwise run-to-run stable);
* CUDA graph: a captured fused_kv_indexer_projection step replays (1, 1, 20 times, fresh inputs)
  equal to the eager fused_kv_indexer_projection step;
* with no layer kind selected, ``fused_kv_indexer_projection`` selected runs the absorbed_query /
  shared_indexer_absorbed_query path bit for bit.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "absorbed_query"))
sys.path.insert(0, os.path.join(_HERE, ".."))

import absorbed_query_fixture  # noqa: E402
import plan_testlib  # noqa: E402
import test_absorbed_query_attention_integration as aq_integration  # noqa: E402

LAYERS = aq_integration.LAYERS
LAYOUTS = aq_integration.LAYOUTS
KW_EXACT = os.environ.get("GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_KW_EXACT", "0") == "1"
KW_RECORDS = ("indexer_weights",)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(
        not absorbed_query_fixture.checkpoint_available(), reason="needs the GLM-5.2 checkpoint"
    ),
]


def _set_features(features: str, fused_kv_layers: str = "full,shared"):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        entry,
    )

    entry.DEFAULT_LAYERS = tuple(fused_kv_layers.split(","))
    plan_testlib.select(dispatch, absorbed_query_fixture.with_base_features(features))


def _reset_runners(h):
    for layer in LAYERS:
        for name in ("_absorbed_query", "_shared_indexer_query", "_fused_kv_projection"):
            setattr(h.attn[layer], name, None)


@pytest.fixture(scope="module")
def harness():
    with torch.inference_mode():
        h = aq_integration._Harness()
    yield h
    _set_features("")


def _run(h, features, layout, path, hidden, pos, snap, fused_kv_layers="full,shared"):
    _set_features(features, fused_kv_layers)
    _reset_runners(h)
    h.restore(snap)
    out, rec = h.recorded_step(layout, path, hidden, pos)
    return out, rec, h.snapshot()


def _jaccard_rows(a, b):
    rows = []
    for ra, rb in zip(a.tolist(), b.tolist()):
        sa = {v for v in ra if v >= 0}
        sb = {v for v in rb if v >= 0}
        union = len(sa | sb)
        rows.append(1.0 if union == 0 else len(sa & sb) / union)
    return rows


def _compare(native, rerun, cand, report):
    n_out, n_rec, n_cache = native
    c_out, c_rec, c_cache = cand
    fails = []
    for key in n_cache:
        same = torch.equal(c_cache[key], n_cache[key])
        report[f"cache_{key}_identical"] = same
        if not same and (key.startswith("latent") or KW_EXACT):
            fails.append(f"cache {key} differs")
    for layer in LAYERS:
        for key, val in n_rec[layer].items():
            if key in ("topk", "sparse_indices"):
                continue
            same = torch.equal(c_rec[layer][key], val)
            report[f"L{layer}_{key}_identical"] = same
            if not same and (key not in KW_RECORDS or KW_EXACT):
                fails.append(f"L{layer} record {key} differs")
        if "topk" in n_rec[layer]:
            jac = _jaccard_rows(c_rec[layer]["topk"], n_rec[layer]["topk"])
            report[f"L{layer}_topk_mean_jaccard"] = sum(jac) / len(jac)
            if min(jac) < 1.0 and KW_EXACT:
                fails.append(f"L{layer} top-k sets differ")
            if sum(jac) / len(jac) < 0.98:
                fails.append(f"L{layer} top-k mean Jaccard {sum(jac) / len(jac):.4f} < 0.98")
        diff = (c_out[layer].float() - n_out[layer].float()).abs().max().item()
        noise = (rerun[0][layer].float() - n_out[layer].float()).abs().max().item()
        ulp = aq_integration._bf16_ulp(n_out[layer].float().abs().max().item())
        rel = aq_integration._rel_rms(c_out[layer], n_out[layer])
        report[f"L{layer}_out_maxdiff"] = diff
        report[f"L{layer}_out_noise"] = noise
        report[f"L{layer}_out_rel_rms"] = rel
        if KW_EXACT or LAYERS.index(layer) == 1:
            # kva-only layer (shared) or exact kw: native run-to-run bound.
            if diff > max(2 * noise, 4 * ulp):
                fails.append(f"L{layer} output diff {diff:.3e} > max(2*{noise:.3e}, 4*{ulp:.3e})")
        elif rel > 2.0**-7:
            fails.append(f"L{layer} output relRMS {rel:.3e} > 2**-7")
    return fails


@pytest.mark.parametrize("path", ["custom_op", "forward_impl"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_fused_kv_into_attention(harness, layout, path):
    h = harness
    hidden, pos = h.gen_inputs(layout, seed=3)
    snap = h.snapshot()
    try:
        native = _run(
            h, "absorbed_query,shared_indexer_absorbed_query", layout, path, hidden, pos, snap
        )
        cand = _run(
            h,
            "absorbed_query,shared_indexer_absorbed_query,fused_kv_indexer_projection",
            layout,
            path,
            hidden,
            pos,
            snap,
        )
        for layer in LAYERS:
            assert h.attn[layer]._fused_kv_projection not in (None, False), (
                f"fused_kv_indexer_projection not engaged on L{layer}"
            )
        rerun = _run(
            h, "absorbed_query,shared_indexer_absorbed_query", layout, path, hidden, pos, snap
        )
        report = {}
        fails = _compare(native, rerun, cand, report)
        print(
            f"GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_ATTN layout={layout} path={path} {report}",
            flush=True,
        )
        assert not fails, fails
    finally:
        h.restore(snap)
        _set_features("")


@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_no_layer_kind_selected_is_native_path(harness, layout):
    h = harness
    hidden, pos = h.gen_inputs(layout, seed=5)
    snap = h.snapshot()
    try:
        n_out, n_rec, n_cache = _run(
            h,
            "absorbed_query,shared_indexer_absorbed_query",
            layout,
            "custom_op",
            hidden,
            pos,
            snap,
        )
        c_out, c_rec, c_cache = _run(
            h,
            "absorbed_query,shared_indexer_absorbed_query,fused_kv_indexer_projection",
            layout,
            "custom_op",
            hidden,
            pos,
            snap,
            fused_kv_layers="",
        )
        for layer in LAYERS:
            assert h.attn[layer]._fused_kv_projection is False
        for key in n_cache:
            assert torch.equal(c_cache[key], n_cache[key]), key
        for layer in LAYERS:
            for key, val in n_rec[layer].items():
                if key in ("topk", "sparse_indices"):
                    continue
                assert torch.equal(c_rec[layer][key], val), (layer, key)
    finally:
        h.restore(snap)
        _set_features("")


@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_fused_kv_cuda_graph_replay(harness, layout):
    h = harness
    snap = h.snapshot()
    try:
        _set_features("absorbed_query,shared_indexer_absorbed_query,fused_kv_indexer_projection")
        _reset_runners(h)
        hidden, pos = h.gen_inputs(layout, seed=7)
        static = hidden.clone()
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
            fresh, _ = h.gen_inputs(layout, seed=100 + trial)
            static.copy_(fresh)
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
                assert diff <= 4 * aq_integration._bf16_ulp(ref), (trial, layer, diff)
    finally:
        h.restore(snap)
        _set_features("")
