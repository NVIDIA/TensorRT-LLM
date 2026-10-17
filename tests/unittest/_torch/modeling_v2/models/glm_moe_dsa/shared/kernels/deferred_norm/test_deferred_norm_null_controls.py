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
"""Null distribution of deferred_norm's attention-output distances among correct
rounding variants of the same projection.

Same harness, cases and inputs as ``test_deferred_norm_controls.py`` (real layer-6 /
layer-7 weights at TP8 rank 0, six requests natively prefilled with 2,600
tokens, FP8 KV cache, layouts 6 x 1 and 1 x 6, both call paths, ``absorbed_query`` and
``absorbed_query,shared_indexer_absorbed_query``, Gaussian and hot-channel residual rows).

The null family ``S_k`` (``k = 1..K``) models every correct implementation
of "RMSNorm, then the projection" with one bf16 rounding of the normalized
input and one of the output: native code whose projection outputs (``kva``
and, on the full-indexer layer, ``kw``) are ``bf16(W . h_k)`` with ``h_k =
h* + u_k * ulp(h*)``, where ``h* = x * gamma * r*`` is the exact normalized
hidden state (FP64), ``ulp`` the bf16 ulp at ``h*`` and ``u_k`` i.i.d.
uniform on [-1/2, 1/2]. Native (``h = bf16((x * r) * gamma)``, then the
GEMM) and deferred_norm (``a = bf16(x * gamma)``, the GEMM, then ``r``) are both of
this form; their rounding errors are deterministic draws, the family's are
random draws.

Per case, all on the same cache snapshot:

* ``native``, ``rerun``, ``deferred_norm``: as in ``test_deferred_norm_controls.py`` (the
  deferred_norm projection on the layers it serves, ``dn_integration.SERVED_LAYERS``: the target's
  deferred-norm scope is ``shared``).
* ``ref``: native code on the correctly rounded FP64 oracle projection.
* ``swap``: native code whose projection outputs on the served layers are deferred_norm's
  projection kernel outputs (``DeferredNormProjection.project`` on the same packet). It
  differs from the ``deferred_norm`` arm only in the code path around the projection.
* ``S_1 .. S_K``.

Reported per case, arm and layer: attention-output relRMS vs ``ref`` and vs
``native`` (overall and per token row), FP8 query bytes and mean top-k
Jaccard vs ``ref``; the null family's order statistics and the counts of
draws at or above ``deferred_norm`` / ``native``.

``test_projection_null`` reports, per data set, served layer and projection output, the
projection-level error ``e = ||y - y*|| / ||y*||`` of native, deferred_norm and
``K_PROJ`` draws of the family; per-segment errors; the bias and the
largest error in units of the bf16 ulp at ``y*``; and each kernel against
its own arithmetic specification evaluated in FP64 (native: ``bf16(W .
h)``; deferred_norm: ``bf16(r . (W . a))`` with the hop-order ``r``).

Assertions are harness facts only (hooks fired once per projection and run,
deferred_norm engaged on the served layers only, finite outputs). The decision rule that uses these numbers was
written before the data.
"""

import contextlib
import json
import os

import deferred_norm_utils as U
import pytest
import test_absorbed_query_attention_integration as aq_integration
import test_deferred_norm_attention_integration as dn_integration
import test_deferred_norm_controls as controls
import torch

LAYERS = aq_integration.LAYERS
LAYOUTS = aq_integration.LAYOUTS
K_ATTN = int(os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS_K", "48"))
K_PROJ = int(os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS_K_PROJ", "1000"))
SEED_BASE = 7000

pytestmark = dn_integration.pytestmark


@pytest.fixture(scope="module")
def harness():
    with torch.inference_mode():
        h = dn_integration._DeferredNormHarness()
    yield h
    dn_integration._set_features("")


def _kw_weight(h, layer):
    indexer = h.attn[layer].mqa.indexer
    return None if indexer is None else indexer._fused_wk_wp_weight


def _weights64(h):
    """Per layer: name -> FP64 copy of the projection weight."""
    out = {}
    for layer in LAYERS:
        out[layer] = {"kva": h.attn[layer].kv_a_proj_with_mqa.weight.double()}
        w = _kw_weight(h, layer)
        if w is not None:
            out[layer]["kw"] = w.double()
    return out


def _h_star(p):
    """Exact normalized hidden state ``x * gamma * r*`` (FP64)."""
    xd = p.residual.double()
    r = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + p.eps)
    return xd * p.gamma.double() * r


def _bf16_ulp64(t):
    """bf16 ulp at ``|t|`` (FP64; 0 where ``t == 0``)."""
    _, e = torch.frexp(t)
    ulp = torch.exp2((e - 8).to(torch.float64))
    return torch.where(t == 0, torch.zeros_like(t), ulp)


def _draw(hstar, ulp, w64, seed):
    """One member of the null family: name -> ``bf16(W . (h* + u ulp))``."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    u = torch.rand(hstar.shape, generator=g, device="cuda", dtype=torch.float64) - 0.5
    hk = hstar + u * ulp
    return {name: (hk @ w.t()).to(torch.bfloat16) for name, w in w64.items()}


def _seed(k, layer):
    return SEED_BASE + 97 * k + layer


@contextlib.contextmanager
def _projection_override(h, kva, kw):
    """Native step with ``kv_a_proj_with_mqa`` returning ``kva[layer]`` and the
    indexer's fused ``wk|weights_proj`` GEMM returning ``kw[layer]``."""
    f = torch.nn.functional
    orig_linear = f.linear
    by_weight = {id(_kw_weight(h, layer)): (layer, t) for layer, t in kw.items()}
    calls = {f"kva{layer}": 0 for layer in kva}
    calls.update({f"kw{layer}": 0 for layer in kw})

    def make_forward(layer):
        def forward(input, *args, **kwargs):
            t = kva[layer]
            assert input.shape[0] == t.shape[0], (input.shape, t.shape)
            calls[f"kva{layer}"] += 1
            return t.clone()

        return forward

    def linear(input, weight, bias=None):
        hit = by_weight.get(id(weight))
        if hit is not None and bias is None:
            layer, t = hit
            assert input.shape[0] == t.shape[0], (input.shape, t.shape)
            calls[f"kw{layer}"] += 1
            return t.clone()
        return orig_linear(input, weight, bias)

    for layer in kva:
        h.attn[layer].kv_a_proj_with_mqa.forward = make_forward(layer)
    f.linear = linear
    try:
        yield calls
    finally:
        f.linear = orig_linear
        for layer in kva:
            del h.attn[layer].kv_a_proj_with_mqa.forward


def _split(per_layer):
    kva = {layer: d["kva"] for layer, d in per_layer.items()}
    kw = {layer: d["kw"] for layer, d in per_layer.items() if d.get("kw") is not None}
    return kva, kw


def _deferred_norm_projection_outputs(h, packets):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        projection_eligibility,
    )

    out = {}
    for layer in dn_integration.SERVED_LAYERS:
        attn = h.attn[layer]
        ok, reason, kw_weight = projection_eligibility(attn)
        assert ok, reason
        proj = DeferredNormProjection(attn, kw_weight=kw_weight)
        assert proj.supported() is None, proj.supported()
        kva, kw = proj.project(packets[layer])
        out[layer] = {"kva": kva, "kw": kw}
    torch.cuda.synchronize()
    return out


def _row_rel(x, ref):
    xd = x.double().reshape(x.shape[0], -1)
    rd = ref.double().reshape(ref.shape[0], -1)
    return ((xd - rd).pow(2).mean(1).sqrt() / rd.pow(2).mean(1).sqrt().clamp_min(1e-30)).tolist()


def _order_stats(v):
    s = sorted(v)
    n = len(s)
    return {
        "min": s[0],
        "p10": s[int(0.1 * (n - 1))],
        "median": s[(n - 1) // 2],
        "p90": s[int(0.9 * (n - 1))],
        "max": s[-1],
        "mean": sum(s) / n,
    }


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
def test_attention_null(harness, layout, path, features, hot):
    h = harness
    base = ",".join(f for f in features.split(",") if f != "deferred_norm")
    packets, pos = controls._packets(h, layout, 3 if not hot else 13, hot)
    native_in = {layer: p.materialize() for layer, p in packets.items()}
    w64 = _weights64(h)
    hstar = {layer: _h_star(p) for layer, p in packets.items()}
    ulp = {layer: _bf16_ulp64(v) for layer, v in hstar.items()}
    oracle = {
        layer: {
            name: U.fp64_oracle(p.residual, p.gamma, w, p.eps).to(torch.bfloat16)
            for name, w in w64[layer].items()
        }
        for layer, p in packets.items()
    }
    swap = _deferred_norm_projection_outputs(h, packets)
    snap = h.snapshot()
    runs = {}
    try:

        def native_run(hidden=native_in):
            dn_integration._reset_runners(h)
            return dn_integration._run(h, base, layout, path, hidden, pos, snap)

        def override_run(per_layer):
            kva, kw = _split(per_layer)
            with _projection_override(h, kva, kw) as calls:
                run = native_run()
            assert all(v == 1 for v in calls.values()), calls
            return run

        runs["native"] = native_run()
        dn_integration._reset_runners(h)
        runs["deferred_norm"] = dn_integration._run(
            h, features, layout, path, dn_integration._served(packets), pos, snap
        )
        for layer in LAYERS:
            engaged = h.attn[layer]._deferred_norm_projection not in (None, False)
            assert engaged == (layer in dn_integration.SERVED_LAYERS), f"deferred_norm on L{layer}"
        runs["rerun"] = native_run()
        runs["ref"] = override_run(oracle)
        runs["swap"] = override_run(swap)
        null = []
        for k in range(1, K_ATTN + 1):
            draw = {
                layer: _draw(hstar[layer], ulp[layer], w64[layer], _seed(k, layer))
                for layer in LAYERS
            }
            null.append(override_run(draw))

        ref_out, nat_out = runs["ref"][0], runs["native"][0]
        report = {
            "layout": layout,
            "path": path,
            "features": features,
            "data": "hot" if hot else "gauss",
            "k": K_ATTN,
            "layers": {},
        }
        for layer in LAYERS:
            for run in list(runs.values()) + null:
                assert torch.isfinite(run[0][layer].float()).all(), layer
            arms = {}
            for arm, run in runs.items():
                vs_ref = dn_integration._output_stats(runs["ref"], run, layout)[layer]
                arms[arm] = {
                    "d_ref": aq_integration._rel_rms(run[0][layer], ref_out[layer]),
                    "d_nat": aq_integration._rel_rms(run[0][layer], nat_out[layer]),
                    "rows_d_ref": _row_rel(run[0][layer], ref_out[layer]),
                    "quant_q_bytes_diff_vs_ref": vs_ref.get("quant_q_bytes_diff_frac"),
                    "topk_jaccard_vs_ref": vs_ref.get("topk_mean_jaccard"),
                }
            d_ref = [aq_integration._rel_rms(r[0][layer], ref_out[layer]) for r in null]
            d_nat = [aq_integration._rel_rms(r[0][layer], nat_out[layer]) for r in null]
            rows = [_row_rel(r[0][layer], ref_out[layer]) for r in null]
            vs_ref = [dn_integration._output_stats(runs["ref"], r, layout)[layer] for r in null]
            q_bytes = [s.get("quant_q_bytes_diff_frac") for s in vs_ref]
            jac = [s.get("topk_mean_jaccard") for s in vs_ref if "topk_mean_jaccard" in s]
            nrows = len(rows[0])
            null_stats = {
                "d_ref": _order_stats(d_ref),
                "d_nat": _order_stats(d_nat),
                "d_ref_values": d_ref,
                "d_nat_values": d_nat,
                "rows_d_ref": [_order_stats([r[i] for r in rows]) for i in range(nrows)],
                "quant_q_bytes_diff_vs_ref": _order_stats(q_bytes),
            }
            if jac:
                null_stats["topk_jaccard_vs_ref"] = _order_stats(jac)
            for arm in ("deferred_norm", "native", "swap"):
                arms[arm]["null_ge_d_ref"] = sum(1 for v in d_ref if v >= arms[arm]["d_ref"])
                arms[arm]["null_ge_d_nat"] = sum(1 for v in d_nat if v >= arms[arm]["d_nat"])
            arms["deferred_norm"]["d_swap"] = aq_integration._rel_rms(
                runs["deferred_norm"][0][layer], runs["swap"][0][layer]
            )
            report["layers"][str(layer)] = {"arms": arms, "null": null_stats}
        print(
            "GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS " + json.dumps(report, sort_keys=True),
            flush=True,
        )
        out_dir = os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS_OUT")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            name = (
                f"null-controls-{layout}-{path}-{features.replace(',', '_')}-{report['data']}.json"
            )
            with open(os.path.join(out_dir, name), "w") as fh:
                json.dump(report, fh, indent=1, sort_keys=True)
    finally:
        h.restore(snap)
        dn_integration._set_features("")


def _segments(h, layer, name, n):
    attn = h.attn[layer]
    if name == "kva":
        kv = attn.kv_lora_rank
        rope = getattr(attn, "qk_rope_head_dim", 64)
        q = n - kv - rope
        return {"q_a": (0, q), "kv_a": (q, q + kv), "k_rope": (q + kv, n)}
    wk = attn.mqa.indexer.head_dim
    return {"wk": (0, wk), "weights_proj": (wk, n)}


def _audit(y, ystar, segments):
    """Error statistics of ``y`` vs the FP64 oracle ``y*``."""
    yd = y.double()
    ulp = _bf16_ulp64(ystar)
    err_ulp = torch.where(ulp > 0, (yd - ystar) / ulp.clamp_min(1e-300), torch.zeros_like(yd))
    nearest = ystar.to(torch.bfloat16)
    out = {
        "e": U.norm_rms_err(y, ystar),
        "frac_not_nearest": float((y != nearest).float().mean()),
        "max_err_ulp": float(err_ulp.abs().max()),
        "mean_err_ulp": float(err_ulp.mean()),
        "frac_err_ge_1ulp": float((err_ulp.abs() >= 1.0).float().mean()),
        "seg_e": {s: U.norm_rms_err(y[:, a:b], ystar[:, a:b]) for s, (a, b) in segments.items()},
        "row_e": [U.norm_rms_err(y[i], ystar[i]) for i in range(y.shape[0])],
    }
    return out


def _vs_spec(y, spec):
    ulp = U.bf16_ulp_distance(y, spec)
    return {"max_ulp": int(ulp.max()), "frac_diff": float((ulp > 0).float().mean())}


@pytest.mark.parametrize("hot", [False, True], ids=["gauss", "hot"])
@torch.inference_mode()
def test_projection_null(harness, hot):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        rows_from_frag,
    )

    h = harness
    packets, _ = controls._packets(h, "6req_x1tok", 3 if not hot else 13, hot)
    w64 = _weights64(h)
    swap = _deferred_norm_projection_outputs(h, packets)
    report = {"data": "hot" if hot else "gauss", "k": K_PROJ, "layers": {}}
    for layer in dn_integration.SERVED_LAYERS:
        p = packets[layer]
        rows, hidden = p.residual.shape
        hstar = _h_star(p)
        ulp = _bf16_ulp64(hstar)
        h_nat = p.materialize()
        a = rows_from_frag(p.a_frag, rows, hidden)
        r_w = torch.rsqrt(U.square_sum_in_order(p.sumsq) / hidden + p.eps)
        layer_rep = {}
        for name, w in w64[layer].items():
            ystar = U.fp64_oracle(p.residual, p.gamma, w, p.eps)
            segs = _segments(h, layer, name, w.shape[0])
            if name == "kva":
                y_nat = h.attn[layer].kv_a_proj_with_mqa(h_nat)
            else:
                y_nat = torch.nn.functional.linear(h_nat, _kw_weight(h, layer))
            y_deferred_norm = swap[layer][name]
            spec_nat = (h_nat.double() @ w.t()).to(torch.bfloat16)
            spec_deferred_norm = (r_w.double().unsqueeze(-1) * (a.double() @ w.t())).to(
                torch.bfloat16
            )
            e_null = []
            for k in range(1, K_PROJ + 1):
                g = torch.Generator(device="cuda").manual_seed(_seed(k, layer))
                u = torch.rand(hstar.shape, generator=g, device="cuda", dtype=torch.float64) - 0.5
                y = ((hstar + u * ulp) @ w.t()).to(torch.bfloat16)
                e_null.append(U.norm_rms_err(y, ystar))
            draw1 = _draw(hstar, ulp, {name: w}, _seed(1, layer))[name]
            e_deferred_norm = U.norm_rms_err(y_deferred_norm, ystar)
            e_nat = U.norm_rms_err(y_nat, ystar)
            layer_rep[name] = {
                "native": _audit(y_nat, ystar, segs),
                "deferred_norm": _audit(y_deferred_norm, ystar, segs),
                "ref": _audit(ystar.to(torch.bfloat16), ystar, segs),
                "null_draw1": _audit(draw1, ystar, segs),
                "null_e": _order_stats(e_null),
                "null_ge_e_deferred_norm": sum(1 for v in e_null if v >= e_deferred_norm),
                "null_ge_e_native": sum(1 for v in e_null if v >= e_nat),
                "native_vs_spec": _vs_spec(y_nat, spec_nat),
                "deferred_norm_vs_spec": _vs_spec(y_deferred_norm, spec_deferred_norm),
                "deferred_norm_vs_native": _vs_spec(y_deferred_norm, y_nat),
            }
        report["layers"][str(layer)] = layer_rep
    torch.cuda.synchronize()
    print(
        "GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS_PROJ " + json.dumps(report, sort_keys=True),
        flush=True,
    )
    out_dir = os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"null-controls-proj-{report['data']}.json"), "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
    for layer, d in report["layers"].items():
        for name, s in d.items():
            assert s["native"]["e"] == s["native"]["e"] and s["null_e"]["max"] < 0.05, (layer, name)


def _fp32_reference(op, w64, scale=None):
    """The same specification through an FP32 cuBLAS GEMM (TF32 off): a
    third accumulation order of the same products."""
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        y = op.float() @ w64.float().t()
        if scale is not None:
            y = y * scale.unsqueeze(-1)
        return y.to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _accumulation_audit(y, s, t_abs, ystar, max_list=16):
    """``y`` (bf16) against its exact pre-rounding specification ``s`` (FP64).

    ``rho`` is the error beyond the output rounding (half a bf16 ulp at
    ``max(|s|, |y|)``) and beyond a ``2**-21 |s|`` allowance for the scale
    ``r`` (``rsqrt.approx``), in units of ``2**-24 * sum_k |t_jk|``: the
    scale of FP32 accumulation error of the dot product's terms ``t_jk``.
    """
    yd = y.double()
    err = (yd - s).abs()
    half = 0.5 * _bf16_ulp64(torch.maximum(s.abs(), yd.abs()))
    excess = (err - half - 2.0**-21 * s.abs()).clamp_min(0.0)
    rho = excess / (2.0**-24 * t_abs).clamp_min(1e-300)
    ulp = U.bf16_ulp_distance(y, s.to(torch.bfloat16))
    big = (ulp > 1).nonzero().tolist()
    flat = rho.flatten()
    k999 = max(1, int(0.999 * flat.numel()))
    out = {
        "frac_diff": float((ulp > 0).float().mean()),
        "max_ulp": int(ulp.max()),
        "n_gt1ulp": len(big),
        "rho_max": float(rho.max()),
        "rho_p999": float(flat.kthvalue(k999).values),
        "gt1ulp": [
            {
                "row": i,
                "col": j,
                "ulp": int(ulp[i, j]),
                "cancel": float(s[i, j].abs() / t_abs[i, j]),
                "rho": float(rho[i, j]),
            }
            for i, j in big[:max_list]
        ],
        "bias_rel": float((yd - ystar).mean() / ystar.pow(2).mean().sqrt()),
        "frac_above_oracle": float(
            ((yd - ystar) > 0).double().sum() / ((yd - ystar) != 0).double().sum().clamp_min(1)
        ),
    }
    return out


@pytest.mark.parametrize("hot", [False, True], ids=["gauss", "hot"])
@torch.inference_mode()
def test_projection_spec_accumulation(harness, hot):
    """Each projection kernel (deferred_norm's and native's) against its own exact
    arithmetic specification, with the error measured in units of the FP32
    accumulation scale of its terms; next to an FP32 cuBLAS reference of the
    same specification."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.packet import (  # noqa: E501
        rows_from_frag,
    )

    h = harness
    packets, _ = controls._packets(h, "6req_x1tok", 3 if not hot else 13, hot)
    w64 = _weights64(h)
    deferred_norm_out = _deferred_norm_projection_outputs(h, packets)
    report = {"data": "hot" if hot else "gauss", "layers": {}}
    for layer in dn_integration.SERVED_LAYERS:
        p = packets[layer]
        rows, hidden = p.residual.shape
        h_nat = p.materialize()
        a = rows_from_frag(p.a_frag, rows, hidden)
        r_w = torch.rsqrt(U.square_sum_in_order(p.sumsq) / hidden + p.eps)
        layer_rep = {}
        for name, w in w64[layer].items():
            ystar = U.fp64_oracle(p.residual, p.gamma, w, p.eps)
            if name == "kva":
                y_nat = h.attn[layer].kv_a_proj_with_mqa(h_nat)
            else:
                y_nat = torch.nn.functional.linear(h_nat, _kw_weight(h, layer))
            s_deferred_norm = r_w.double().unsqueeze(-1) * (a.double() @ w.t())
            t_deferred_norm = r_w.double().unsqueeze(-1) * (a.double().abs() @ w.abs().t())
            s_nat = h_nat.double() @ w.t()
            t_nat = h_nat.double().abs() @ w.abs().t()
            layer_rep[name] = {
                "deferred_norm_kernel": _accumulation_audit(
                    deferred_norm_out[layer][name], s_deferred_norm, t_deferred_norm, ystar
                ),
                "deferred_norm_fp32_ref": _accumulation_audit(
                    _fp32_reference(a, w, r_w), s_deferred_norm, t_deferred_norm, ystar
                ),
                "native_kernel": _accumulation_audit(y_nat, s_nat, t_nat, ystar),
                "native_fp32_ref": _accumulation_audit(
                    _fp32_reference(h_nat, w), s_nat, t_nat, ystar
                ),
            }
        report["layers"][str(layer)] = layer_rep
    torch.cuda.synchronize()
    print(
        "GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS_SPEC " + json.dumps(report, sort_keys=True),
        flush=True,
    )
    out_dir = os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_NULL_CONTROLS_OUT")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"null-controls-spec-{report['data']}.json"), "w") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
    for layer, d in report["layers"].items():
        for name, s in d.items():
            for impl, v in s.items():
                assert v["rho_max"] == v["rho_max"], (layer, name, impl)
