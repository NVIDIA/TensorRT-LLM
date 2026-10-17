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
"""Diagnosis of deferred_norm's attention-output deviation next to native-only controls.

Same single-GPU harness and metrics as ``test_deferred_norm_attention_integration.py``
(real layer-6 / layer-7 weights at TP8 rank 0, six requests natively
prefilled with 2,600 tokens, FP8 KV cache, layouts 6 x 1 and 1 x 6, both
call paths, ``absorbed_query`` and ``absorbed_query,shared_indexer_absorbed_query``). Every arm runs
the same generation step from the same cache snapshot and is compared with the native reference step
on ``h = materialize(packet)`` (the post_moe_hop's normalized output, bit for bit).

Arms:

* ``deferred_norm``: the deferred_norm packet through the deferred_norm projection (the candidate)
  on the layers the projection serves (``dn_integration.SERVED_LAYERS``: the target's deferred-norm
  scope is ``shared``), the normalized ``h`` on the full-indexer layer, as a served step
  gives them.
* ``rerun``: the native step again (run-to-run noise of native attention).
* ``n1_rmsnorm``: native code on ``h'`` = TRT-LLM's ``RMSNorm`` module
  (FlashInfer RMSNorm) applied to the same ``x`` and gain: a second in-tree
  implementation of the same normalization.
* ``n2_cublas``: native code with the ``kv_a_proj_with_mqa`` GEMM through
  ``Linear.apply_linear`` (``cublas_mm``), the path ``DeepseekV3Linear``
  takes for more than 16 tokens instead of ``dsv3_fused_a_gemm_op``; the
  input ``h`` is unchanged.
* ``n3_round``: native code on ``bf16(bf16(x * gamma) * r)`` (a single
  extra rounding of the native input).
* ``n4_dither1`` / ``n4_dither50``: native code on ``h`` with a random
  +-1 bf16 ulp change in 1 % / 50 % of its elements.
* ``ref``: native code whose projection outputs (``kva`` and, on the
  full-indexer layer, the fused ``wk|weights_proj`` output ``kw``) are the
  correctly rounded FP64 oracle ``bf16(r* * sum_k x_k gamma_k W_jk)``; every
  arm is also reported as its distance to this step.
* ``neg``: a packet without the gain through deferred_norm on the served layers (negative
  control: a real bug).

Reported per case, per arm and layer: the deviation metrics vs the native step
(attention-output relRMS, absorbed-query relRMS, FP8 query bytes differing,
indexer weights relRMS, FP8 indexer-query bytes differing, top-k rows
identical, mean top-k Jaccard), the attention-output relRMS vs ``ref``, and
whether the predeclared deviation bound (output relRMS <=
2**-7 and mean Jaccard >= 0.98) holds. Per layout, per arm and served layer, the
projection-level error ``e = ||y - y*|| / ||y*||`` of ``kva``.

The numbers are evidence for the deviation bound, not a pass
criterion: the assertions only check that every arm ran
the intended path (deferred_norm engaged on the served layers only, hooks fired once, finite
outputs) and that the experiment can see a real bug (on every served layer the negative
control's output deviation exceeds every native-only control's).

Fixture label: weights real (checkpoint), activations synthetic (Gaussian
residual rows; the ``hot`` variant multiplies 12 channels by 30, as
residual streams with outlier channels have).
"""

import contextlib
import json
import os

import absorbed_query_fixture
import deferred_norm_utils as U
import pytest
import test_absorbed_query_attention_integration as aq_integration
import test_deferred_norm_attention_integration as dn_integration
import torch

LAYERS = aq_integration.LAYERS
FULL = absorbed_query_fixture.FULL_LAYER
LAYOUTS = aq_integration.LAYOUTS
OUTPUT_BOUND = 2.0**-7
NATIVE_CONTROLS = ("n1_rmsnorm", "n2_cublas", "n3_round", "n4_dither1", "n4_dither50")

pytestmark = dn_integration.pytestmark


@pytest.fixture(scope="module")
def harness():
    with torch.inference_mode():
        h = dn_integration._DeferredNormHarness()
    yield h
    dn_integration._set_features("")


def _packets(h, layout, seed, hot):
    """Per-layer packets of residual-like rows ``x`` (Gaussian x 2, optionally
    12 channels x 30) with the layer's real gain."""
    nreq, ntok = LAYOUTS[layout]
    out = {}
    for layer in LAYERS:
        g = torch.Generator(device="cuda").manual_seed(seed * 10 + layer)
        x = torch.randn(nreq * ntok, U.H, generator=g, device="cuda") * 2.0
        if hot:
            idx = torch.randint(0, U.H, (12,), generator=g, device="cuda")
            x[:, idx] *= 30.0
        out[layer] = U.make_frag_packet(x.bfloat16(), h.gammas[layer])
    pos = (
        (aq_integration.CTX + torch.arange(ntok, dtype=torch.int32, device="cuda"))
        .repeat(nreq)
        .unsqueeze(0)
    )
    return out, pos


def _rmsnorm_inputs(packets):
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    out = {}
    for layer, p in packets.items():
        norm = RMSNorm(hidden_size=U.H, eps=p.eps, dtype=torch.bfloat16, device="cuda")
        norm.weight.data.copy_(p.gamma)
        out[layer] = norm(p.residual)
    return out


def _dither(h, frac, seed):
    """``h`` with a random +-1 ulp step in a fraction ``frac`` of its nonzero
    elements (the bf16 bit pattern is sign-magnitude, so +-1 on the integer
    view moves the magnitude by one ulp)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    bits = h.contiguous().view(torch.int16)
    pick = torch.rand(h.shape, generator=g, device="cuda") < frac
    step = torch.where(torch.rand(h.shape, generator=g, device="cuda") < 0.5, -1, 1)
    nonzero = (bits.to(torch.int32) & 0x7FFF) != 0
    out = torch.where(pick & nonzero, bits.to(torch.int32) + step, bits.to(torch.int32))
    return out.to(torch.int16).view(torch.bfloat16)


@contextlib.contextmanager
def _cublas_kv_a():
    """``DeepseekV3Linear`` without its <= 16-token ``dsv3_fused_a_gemm_op``
    branch: ``Linear.apply_linear`` (``cublas_mm``), as for larger batches."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Linear
    from tensorrt_llm._torch.modules.linear import Linear

    calls = [0]
    own = DeepseekV3Linear.__dict__["apply_linear"]

    def apply_linear(self, input, bias, lora_params=None, layer_idx=None):
        calls[0] += 1
        return Linear.apply_linear(self, input, bias, lora_params, layer_idx)

    DeepseekV3Linear.apply_linear = apply_linear
    try:
        yield calls
    finally:
        DeepseekV3Linear.apply_linear = own


def _oracle(p, weight):
    return U.fp64_oracle(p.residual, p.gamma, weight, p.eps).to(torch.bfloat16)


@contextlib.contextmanager
def _oracle_projection(h, packets):
    """The native step with ``kva`` / ``kw`` replaced by the correctly rounded
    FP64 oracle of each layer's packet."""
    f = torch.nn.functional
    orig_linear = f.linear
    kva = {
        layer: _oracle(packets[layer], h.attn[layer].kv_a_proj_with_mqa.weight) for layer in LAYERS
    }
    kw = {}
    for layer in LAYERS:
        indexer = h.attn[layer].mqa.indexer
        if indexer is not None:
            w = indexer._fused_wk_wp_weight
            kw[id(w)] = (layer, _oracle(packets[layer], w))
    torch.cuda.synchronize()
    calls = {f"kva{layer}": 0 for layer in LAYERS}
    calls.update({f"kw{layer}": 0 for layer, _ in kw.values()})

    def make_forward(layer):
        def forward(input, *args, **kwargs):
            ref = kva[layer]
            assert input.shape[0] == ref.shape[0], (input.shape, ref.shape)
            calls[f"kva{layer}"] += 1
            return ref.clone()

        return forward

    def linear(input, weight, bias=None):
        hit = kw.get(id(weight))
        if hit is not None and bias is None:
            layer, ref = hit
            assert input.shape[0] == ref.shape[0], (input.shape, ref.shape)
            calls[f"kw{layer}"] += 1
            return ref.clone()
        return orig_linear(input, weight, bias)

    for layer in LAYERS:
        h.attn[layer].kv_a_proj_with_mqa.forward = make_forward(layer)
    f.linear = linear
    try:
        yield calls
    finally:
        f.linear = orig_linear
        for layer in LAYERS:
            del h.attn[layer].kv_a_proj_with_mqa.forward


def _projection_errors(h, layer, p, inputs):
    """``e`` of ``kva`` / ``kw`` per arm on one layer (projection level, vs
    the FP64 oracle). ``inputs``: arm -> that arm's normalized hidden ``h``."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        projection_eligibility,
    )
    from tensorrt_llm._torch.modules.linear import Linear

    attn = h.attn[layer]
    lin = attn.kv_a_proj_with_mqa
    indexer = attn.mqa.indexer
    w_kw = indexer._fused_wk_wp_weight if indexer is not None else None
    oracle = {"kva": U.fp64_oracle(p.residual, p.gamma, lin.weight, p.eps)}
    if w_kw is not None:
        oracle["kw"] = U.fp64_oracle(p.residual, p.gamma, w_kw, p.eps)
    ok, reason, kw_weight = projection_eligibility(attn)
    assert ok, reason
    proj = DeferredNormProjection(attn, kw_weight=kw_weight)
    assert proj.supported() is None, proj.supported()
    kva_w, kw_w = proj.project(p)
    arms = {"deferred_norm": {"kva": kva_w, "kw": kw_w}}
    for arm, hid in inputs.items():
        arms[arm] = {"kva": lin(hid)}
        if w_kw is not None:
            arms[arm]["kw"] = torch.nn.functional.linear(hid, w_kw)
    arms["n2_cublas"] = {"kva": Linear.apply_linear(lin, inputs["native"], None)}
    if w_kw is not None:
        arms["n2_cublas"]["kw"] = torch.nn.functional.linear(inputs["native"], w_kw)
    arms["ref"] = {k: v.to(torch.bfloat16) for k, v in oracle.items()}
    torch.cuda.synchronize()
    return {
        arm: {
            name: U.norm_rms_err(y, oracle[name])
            for name, y in ys.items()
            if name in oracle and y is not None
        }
        for arm, ys in arms.items()
    }


def _input_stats(inputs):
    """Fraction of elements of each arm's ``h`` that differ from native and
    their max bf16 ulp distance."""
    out = {}
    for arm, per_layer in inputs.items():
        if arm == "native":
            continue
        out[arm] = {}
        for layer, hid in per_layer.items():
            ulp = U.bf16_ulp_distance(hid, inputs["native"][layer])
            out[arm][layer] = {
                "frac_diff": float((ulp > 0).float().mean()),
                "max_ulp": int(ulp.max()),
            }
    return out


def _vs_ref(arm_out, ref_out):
    return {layer: aq_integration._rel_rms(arm_out[layer], ref_out[layer]) for layer in LAYERS}


@pytest.mark.parametrize("hot", [False, True], ids=["gauss", "hot"])
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
def test_output_bound_with_native_controls(harness, layout, path, features, hot):
    h = harness
    base = ",".join(f for f in features.split(",") if f != "deferred_norm")
    seed = 3 if not hot else 13
    packets, pos = _packets(h, layout, seed, hot)
    native_in = {layer: p.materialize() for layer, p in packets.items()}
    inputs = {
        "native": native_in,
        "n1_rmsnorm": _rmsnorm_inputs(packets),
        "n3_round": dn_integration._control_inputs(packets),
        "n4_dither1": {layer: _dither(v, 0.01, 1000 + layer) for layer, v in native_in.items()},
        "n4_dither50": {layer: _dither(v, 0.5, 2000 + layer) for layer, v in native_in.items()},
    }
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormHidden,
        materialize,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        frag_from_rows,
    )

    neg = {
        layer: DeferredNormHidden(
            frag_from_rows(p.residual), p.sumsq, p.residual, p.gamma, p.eps, materialize
        )
        if layer in dn_integration.SERVED_LAYERS
        else native_in[layer]
        for layer, p in packets.items()
    }
    snap = h.snapshot()
    runs = {}
    try:

        def native_run(hidden):
            dn_integration._reset_runners(h)
            return dn_integration._run(h, base, layout, path, hidden, pos, snap)

        runs["native"] = native_run(native_in)
        dn_integration._reset_runners(h)
        runs["deferred_norm"] = dn_integration._run(
            h, features, layout, path, dn_integration._served(packets), pos, snap
        )
        for layer in LAYERS:
            engaged = h.attn[layer]._deferred_norm_projection not in (None, False)
            assert engaged == (layer in dn_integration.SERVED_LAYERS), f"deferred_norm on L{layer}"
        runs["rerun"] = native_run(native_in)
        for arm in ("n1_rmsnorm", "n3_round", "n4_dither1", "n4_dither50"):
            runs[arm] = native_run(inputs[arm])
        with _cublas_kv_a() as calls:
            runs["n2_cublas"] = native_run(native_in)
        assert calls[0] == len(LAYERS), f"cuBLAS kv_a path ran {calls[0]} times"
        with _oracle_projection(h, packets) as calls:
            runs["ref"] = native_run(native_in)
        assert all(v == 1 for v in calls.values()), calls
        dn_integration._reset_runners(h)
        runs["neg"] = dn_integration._run(h, features, layout, path, neg, pos, snap)

        report = {
            "layout": layout,
            "path": path,
            "features": features,
            "data": "hot" if hot else "gauss",
            "arms": {},
        }
        ref_out = runs["ref"][0]
        for arm, run in runs.items():
            for layer in LAYERS:
                assert torch.isfinite(run[0][layer].float()).all(), (arm, layer)
            if arm == "native":
                continue
            stats = dn_integration._output_stats(runs["native"], run, layout)
            for layer, d in _vs_ref(run[0], ref_out).items():
                stats[layer]["out_rel_rms_vs_ref"] = d
                stats[layer]["original_output_bound_ok"] = bool(
                    stats[layer]["out_rel_rms"] <= OUTPUT_BOUND
                    and stats[layer].get("topk_mean_jaccard", 1.0) >= 0.98
                )
            report["arms"][arm] = stats
        report["arms"]["native"] = {
            layer: {"out_rel_rms_vs_ref": d}
            for layer, d in _vs_ref(runs["native"][0], ref_out).items()
        }
        report["inputs"] = _input_stats(inputs)
        print(
            "GLM_MOE_DSA_DEFERRED_NORM_CONTROLS " + json.dumps(report, sort_keys=True), flush=True
        )
        out_dir = os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_CONTROLS_OUT")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            name = f"controls-{layout}-{path}-{features.replace(',', '_')}-{report['data']}.json"
            with open(os.path.join(out_dir, name), "w") as fh:
                json.dump(report, fh, indent=1, sort_keys=True)

        for layer in dn_integration.SERVED_LAYERS:
            native_max = max(report["arms"][a][layer]["out_rel_rms"] for a in NATIVE_CONTROLS)
            assert report["arms"]["neg"][layer]["out_rel_rms"] > native_max, (
                f"L{layer}: the negative control is not separated from the native controls"
            )
    finally:
        h.restore(snap)
        dn_integration._set_features("")


@pytest.mark.parametrize("hot", [False, True], ids=["gauss", "hot"])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@torch.inference_mode()
def test_projection_error_of_every_arm(harness, layout, hot):
    """The projection error (``e`` vs the FP64 oracle) of deferred_norm and of every
    native-only input / GEMM variant of the diagnosis."""
    h = harness
    packets, _ = _packets(h, layout, 3 if not hot else 13, hot)
    native_in = {layer: p.materialize() for layer, p in packets.items()}
    per_arm_inputs = {
        "native": native_in,
        "n1_rmsnorm": _rmsnorm_inputs(packets),
        "n3_round": dn_integration._control_inputs(packets),
        "n4_dither1": {layer: _dither(v, 0.01, 1000 + layer) for layer, v in native_in.items()},
        "n4_dither50": {layer: _dither(v, 0.5, 2000 + layer) for layer, v in native_in.items()},
    }
    errs = {
        layer: _projection_errors(
            h, layer, packets[layer], {arm: v[layer] for arm, v in per_arm_inputs.items()}
        )
        for layer in dn_integration.SERVED_LAYERS
    }
    report = {
        "layout": layout,
        "data": "hot" if hot else "gauss",
        "e": errs,
        "inputs": _input_stats(per_arm_inputs),
    }
    print(
        "GLM_MOE_DSA_DEFERRED_NORM_CONTROLS_PROJ " + json.dumps(report, sort_keys=True), flush=True
    )
    for layer, arms in errs.items():
        for arm, d in arms.items():
            for name, e in d.items():
                assert e == e and e < 0.05, (layer, arm, name, e)
