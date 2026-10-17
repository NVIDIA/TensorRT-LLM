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
"""Single-GPU tests of the deferred_norm staged projection (real GLM-5.2 weights).

Native chain: ``h = bf16((x * r) * gamma)`` (the post_moe_hop's output, produced
here by the deferred_norm materialization kernel, which runs post_moe_hop's arithmetic) followed
by the native ``kv_a_proj_with_mqa`` (``dsv3_fused_a_gemm_op``).
deferred_norm: the staged projection on ``a = bf16(x * gamma)`` and the hop's
square-sum partials (emulated bit-exactly in torch).

The target's deferred-norm scope is ``shared``: the projection serves the shared-indexer
layers (``LAYERS``); a full-indexer layer (``FULL_INDEXER_LAYERS``) keeps its native
projection, and the projection's eligibility check declines it.

Frozen error budget: per output tensor
``allclose(y_deferred_norm, y_native, rtol=0.1, atol=1e-2)`` and normalized RMS error
vs the FP64 oracle ``e_deferred_norm <= 1.25 * e_native + 2**-12``.

Fixture label: weights real (checkpoint), activations synthetic.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "absorbed_query"))

import absorbed_query_fixture  # noqa: E402
import deferred_norm_utils as U  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not absorbed_query_fixture.checkpoint_available(),
    reason="needs a GPU and the GLM-5.2 checkpoint",
)

LAYERS = [int(x) for x in os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_TEST_LAYERS", "7").split(",")]
FULL_INDEXER_LAYERS = [
    int(x) for x in os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_TEST_FULL_LAYERS", "6").split(",")
]
ROWS = [int(x) for x in os.environ.get("GLM_MOE_DSA_DEFERRED_NORM_TEST_ROWS", "1,6,8").split(",")]
_MODS = {}


def _module(layer):
    if layer not in _MODS:
        with torch.inference_mode():
            cfg = absorbed_query_fixture.build_model_config(0)
            _MODS[layer] = (
                absorbed_query_fixture.build_native_attention(layer, 0, cfg),
                absorbed_query_fixture.load_input_layernorm(layer).cuda(),
            )
    return _MODS[layer]


def make_residual(rows, seed, scale=1.0):
    """Residual-stream-like rows: unit Gaussian plus a few large channels."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, U.H, generator=g, device="cuda") * 2.0
    hot = torch.randint(0, U.H, (12,), generator=g, device="cuda")
    x[:, hot] *= 30.0
    return (x * scale).bfloat16()


def make_packet(x, gamma):
    return U.make_frag_packet(x, gamma)


def _projection(attn):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        projection_eligibility,
    )

    ok, reason, kw = projection_eligibility(attn)
    assert ok, reason
    proj = DeferredNormProjection(attn, kw_weight=kw)
    assert proj.supported() is None, proj.supported()
    return proj


def _native(attn, h):
    kva = attn.kv_a_proj_with_mqa(h)
    kw = None
    if attn.mqa.indexer is not None:
        kw = torch.nn.functional.linear(h, attn.mqa.indexer._fused_wk_wp_weight)
    return kva, kw


@pytest.mark.parametrize("layer", LAYERS)
def test_served_configs_cover_the_consumer(layer):
    """``precompile_served`` (run by the producing hop before its cross-rank
    agreement) compiles exactly the kernel the consumer launches on this
    layer kind, so the consumer's ``precompile`` is a cache hit."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.proj_kernel import (  # noqa: E501
        is_compiled,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        precompile_served,
        served_kernel_configs,
    )

    device = torch.cuda.current_device()
    assert precompile_served(device) is None
    proj = _projection(_module(layer)[0])
    assert proj.served_config() in served_kernel_configs(device)
    assert is_compiled(**proj.served_config())


@pytest.mark.parametrize("scale", [1.0, 1e-3, 30.0])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("layer", LAYERS)
def test_projection_within_frozen_budget(layer, rows, scale):
    attn, gamma = _module(layer)
    proj = _projection(attn)
    with torch.inference_mode():
        x = make_residual(rows, seed=100 + rows + layer, scale=scale)
        packet = make_packet(x, gamma)
        h = packet.materialize()
        kva_n, kw_n = _native(attn, h)
        kva_w, kw_w = proj.project(packet)
        torch.cuda.synchronize()
        outs = [("kva", kva_w, kva_n, attn.kv_a_proj_with_mqa.weight)]
        if kw_n is not None:
            outs.append(("kw", kw_w, kw_n, attn.mqa.indexer._fused_wk_wp_weight))
        for name, cand, nat, w in outs:
            oracle = U.fp64_oracle(x, gamma, w)
            stats = U.compare(cand, nat, oracle)
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_BUDGET layer={layer} rows={rows} scale={scale} out={name} {stats}"
            )
            assert U.within_budget(stats), (name, stats)


@pytest.mark.parametrize("layer", LAYERS)
def test_native_input_mode(layer):
    """Same kernel on the normalized hidden with r = 1 (attribution arm)."""
    attn, gamma = _module(layer)
    proj = _projection(attn)
    with torch.inference_mode():
        x = make_residual(6, seed=7)
        h = make_packet(x, gamma).materialize()
        kva_n, kw_n = _native(attn, h)
        kva_w, kw_w = proj.run(h, None, U.EPS)
        torch.cuda.synchronize()
        for name, cand, nat in (("kva", kva_w, kva_n), ("kw", kw_w, kw_n)):
            if nat is None:
                continue
            ulp = U.bf16_ulp_distance(cand, nat)
            print(
                f"GLM_MOE_DSA_DEFERRED_NORM_NATIVE_INPUT layer={layer} out={name} max_ulp={int(ulp.max())} "
                f"frac_diff={float((ulp > 0).float().mean()):.3e}"
            )
            assert torch.allclose(cand.float(), nat.float(), rtol=0.1, atol=1e-2)


@pytest.mark.parametrize("layer", LAYERS)
def test_rows_match_single_row_runs(layer):
    """Every token row equals a launch of that row alone (no cross-row mixing)."""
    attn, gamma = _module(layer)
    proj = _projection(attn)
    with torch.inference_mode():
        x = make_residual(8, seed=11)
        kva, kw = proj.project(make_packet(x, gamma))
        for t in range(8):
            kva1, kw1 = proj.project(make_packet(x[t : t + 1].contiguous(), gamma))
            assert torch.equal(kva1[0], kva[t])
            if kw is not None:
                assert torch.equal(kw1[0], kw[t])


@pytest.mark.parametrize("layer", LAYERS)
def test_graph_replay_matches_eager(layer):
    attn, gamma = _module(layer)
    proj = _projection(attn)
    with torch.inference_mode():
        x0 = make_residual(6, seed=21)
        static = U.empty_like_packet(make_packet(x0, gamma))
        ref0 = make_packet(x0, gamma)
        static.a_frag.copy_(ref0.a_frag)
        static.sumsq.copy_(ref0.sumsq)
        proj.project(static)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            g_kva, g_kw = proj.project(static)
        for trial, replays in enumerate((1, 1, 50)):
            x = make_residual(6, seed=300 + trial)
            ref = make_packet(x, gamma)
            static.a_frag.copy_(ref.a_frag)
            static.sumsq.copy_(ref.sumsq)
            for _ in range(replays):
                graph.replay()
            e_kva, e_kw = proj.project(ref)
            torch.cuda.synchronize()
            assert torch.equal(g_kva, e_kva)
            if e_kw is not None:
                assert torch.equal(g_kw, e_kw)


@pytest.mark.parametrize("stage_before_wait", [True, False])
@pytest.mark.parametrize("delay_ns", [0, 5000, 50000])
@pytest.mark.parametrize("layer", LAYERS)
def test_pdl_delayed_producer(layer, delay_ns, stage_before_wait):
    """The producer releases its dependents at entry and writes the packet only
    after ``delay_ns``; the projection (launched with PDL, staging its weights
    before its dependency wait) must still read the final packet."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormProjConfig,
        DeferredNormProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        projection_eligibility,
    )

    attn, gamma = _module(layer)
    _, _, kwt = projection_eligibility(attn)
    proj = DeferredNormProjection(
        attn, DeferredNormProjConfig(stage_before_wait=stage_before_wait), kw_weight=kwt
    )
    if not proj.use_pdl:
        pytest.skip("PDL disabled (TRTLLM_ENABLE_PDL=0)")
    with torch.inference_mode():
        for trial in range(4):
            x = make_residual(6, seed=500 + trial)
            src = make_packet(x, gamma)
            dst = U.empty_like_packet(src)
            dst.a_frag.fill_(float("nan"))
            U.sumsq_full(dst).fill_(float("nan"))
            torch.cuda.synchronize()
            U.proxy_producer(
                src.a_frag, dst.a_frag, U.sumsq_full(src), U.sumsq_full(dst), delay_ns, rows=8
            )
            kva, kw = proj.project(dst)
            ref_kva, ref_kw = proj.project(src)
            torch.cuda.synchronize()
            assert torch.equal(kva, ref_kva), f"trial {trial}"
            if kw is not None:
                assert torch.equal(kw, ref_kw)


def test_materialize_close_to_torch():
    """The fallback's normalized output vs a torch RMSNorm of the same x
    (bit-exactness against the post_moe_hop is asserted by the MPI tests)."""
    _, gamma = _module(LAYERS[0])
    with torch.inference_mode():
        x = make_residual(6, seed=3)
        h = make_packet(x, gamma).materialize()
        s = U.square_sum_in_order(U.hop_sumsq_partials(x))
        r = torch.rsqrt(s / U.H + U.EPS).unsqueeze(-1)
        ref = ((x.float() * r) * gamma.float()).bfloat16()
        assert int(U.bf16_ulp_distance(h, ref).max()) <= 1


@pytest.mark.parametrize("layer", FULL_INDEXER_LAYERS)
def test_full_indexer_layer_keeps_its_native_projection(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.projection import (  # noqa: E501
        projection_eligibility,
    )

    attn, _ = _module(layer)
    assert attn.mqa.indexer is not None
    ok, reason, kw = projection_eligibility(attn)
    assert not ok and kw is None
    assert reason.endswith("=shared: full-indexer layer keeps its native projection"), reason


def test_eligibility_rejects_out_of_domain():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm.proj_kernel import (  # noqa: E501
        projection_supported,
    )

    assert projection_supported(6144, 2624, 152, 512, 18, 4) is None
    assert projection_supported(6144, 2784, 152, 512, 18, 4) is None
    assert projection_supported(6144, 2624, 64, 512, 18, 4) is not None  # 41 rows per CTA
    assert projection_supported(6144, 2784, 152, 512, 17, 4) is not None  # 2 rows over staged
