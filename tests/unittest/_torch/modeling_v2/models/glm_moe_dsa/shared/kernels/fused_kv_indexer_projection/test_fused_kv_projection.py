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
"""Single-GPU tests of the fused_kv_indexer_projection merged projection (real GLM-5.2 weights).

Oracle: the native ``kv_a_proj_with_mqa`` call (``dsv3_fused_a_gemm_op``)
and ``F.linear`` on the indexer's fused ``wk | weights_proj`` weight, on the
same normalized hidden state.

Contract (frozen before measurement): ``kva`` is
bit-identical to native for every R = 1..8; ``kw`` is bit-identical or, as
the named variant, within the frozen budget ``allclose(rtol=0.1,
atol=1e-2)`` and ``e_fused_kv <= 1.25 e_native + 2^-12`` against the FP64 oracle.

Fixture labels: weights real (checkpoint); activations: native RMSNorm of
synthetic residual rows with the real gain, scaled copies, plain Gaussian
rows, and (when mounted) real normalized hidden rows of the captured fixture.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "absorbed_query"))

import absorbed_query_fixture  # noqa: E402
import fused_kv_utils as U  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not absorbed_query_fixture.checkpoint_available(),
    reason="needs a GPU and the GLM-5.2 checkpoint",
)

LAYERS = [
    int(x)
    for x in os.environ.get(
        "GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_TEST_LAYERS", "0,1,2,3,78"
    ).split(",")
]
ROWS = [
    int(x)
    for x in os.environ.get(
        "GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_TEST_ROWS", "1,2,3,4,5,6,7,8"
    ).split(",")
]
# Whether kw must be bit-identical (set once the order probe found the native order).
KW_EXACT = os.environ.get("GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_KW_EXACT", "0") == "1"
INPUTS = ("norm", "norm_x1e-3", "norm_x30", "gauss", "real")


def _projection(attn, config=None):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        FusedKvProjection,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.projection import (  # noqa: E501
        projection_eligibility,
    )

    ok, reason, kw = projection_eligibility(attn)
    assert ok, reason
    proj = FusedKvProjection(attn, config, kw_weight=kw)
    assert proj.supported() is None, proj.supported()
    return proj


def _parse_cfg(text: str, **extra):
    from dataclasses import replace

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection import (  # noqa: E501
        FusedKvProjConfig,
    )

    return replace(FusedKvProjConfig.parse(text), **extra)


def _input(kind: str, rows: int, seed: int, gamma: torch.Tensor):
    if kind == "norm":
        return U.native_hidden(rows, seed, gamma)
    if kind == "norm_x1e-3":
        return U.native_hidden(rows, seed, gamma, scale=1e-3)
    if kind == "norm_x30":
        return U.native_hidden(rows, seed, gamma, scale=30.0)
    if kind == "gauss":
        g = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(rows, U.H, generator=g, device="cuda").bfloat16()
    return U.real_hidden(rows, offset=97 * seed % 1024)


def _check_outputs(attn, h, kva, kw, tag):
    kva_n, kw_n = U.native(attn, h)
    torch.cuda.synchronize()
    assert kva.shape == kva_n.shape and kva.dtype == kva_n.dtype and kva.stride() == kva_n.stride()
    ok = torch.equal(kva.view(torch.int16), kva_n.view(torch.int16))
    if not ok:
        stats = U.compare(kva, kva_n, U.fp64_oracle(h, attn.kv_a_proj_with_mqa.weight))
        pytest.fail(f"kva not bit-identical {tag}: {stats}")
    if kw_n is None:
        assert kw is None
        return None
    assert kw.shape == kw_n.shape and kw.dtype == kw_n.dtype and kw.stride() == kw_n.stride()
    stats = U.compare(kw, kw_n, U.fp64_oracle(h, attn.mqa.indexer._fused_wk_wp_weight))
    print(f"GLM_MOE_DSA_FUSED_KV_INDEXER_PROJECTION_KW {tag} {stats}")
    if KW_EXACT:
        assert stats["bit_identical"], (tag, stats)
    else:
        assert U.within_budget(stats), (tag, stats)
    return stats


@pytest.mark.parametrize("layer", LAYERS)
def test_served_kernel_supported_and_compiled(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.proj_kernel import (  # noqa: E501
        is_compiled,
    )

    attn, _ = U.module(layer)
    proj = _projection(attn)
    proj.precompile()
    assert is_compiled(**proj.kernel_config())
    assert proj.n1 == (0 if attn.mqa.indexer is None else 160)


@pytest.mark.parametrize("kind", INPUTS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("layer", LAYERS)
def test_kva_bit_identical_kw_contract(layer, rows, kind):
    attn, gamma = U.module(layer)
    proj = _projection(attn)
    with torch.inference_mode():
        h = _input(kind, rows, seed=1000 + 10 * layer + rows, gamma=gamma)
        if h is None:
            pytest.skip("real hidden-state fixture not mounted")
        kva, kw = proj.run(h)
        _check_outputs(attn, h, kva, kw, f"layer={layer} rows={rows} input={kind}")


@pytest.mark.parametrize("layer", LAYERS)
def test_rows_match_single_row_runs(layer):
    """Every token row equals a launch of that row alone (padded or other
    rows of the batch never change a live row)."""
    attn, gamma = U.module(layer)
    proj = _projection(attn)
    with torch.inference_mode():
        h = U.native_hidden(8, seed=11, gamma=gamma)
        kva, kw = proj.run(h)
        for t in range(8):
            kva1, kw1 = proj.run(h[t : t + 1].contiguous())
            assert torch.equal(kva1[0], kva[t])
            if kw is not None:
                assert torch.equal(kw1[0], kw[t])


@pytest.mark.parametrize("layer", LAYERS[:2] + LAYERS[-1:])
def test_graph_replay_changing_rows(layer):
    """Graphs captured at R = 6 and R = 1 (the draft-loop shapes) replayed
    interleaved with new inputs equal eager launches."""
    attn, gamma = U.module(layer)
    proj = _projection(attn)
    with torch.inference_mode():
        static_inputs, graphs, outs = {}, {}, {}
        for rows in (6, 1):
            static_inputs[rows] = U.native_hidden(rows, seed=rows, gamma=gamma)
            proj.run(static_inputs[rows])
            torch.cuda.synchronize()
            graphs[rows] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[rows]):
                outs[rows] = proj.run(static_inputs[rows])
        for trial in range(6):
            rows = (6, 1)[trial % 2]
            static_inputs[rows].copy_(U.native_hidden(rows, seed=300 + trial, gamma=gamma))
            for _ in range(1 + 20 * (trial == 4)):
                graphs[rows].replay()
            e_kva, e_kw = proj.run(static_inputs[rows])
            torch.cuda.synchronize()
            g_kva, g_kw = outs[rows]
            assert torch.equal(g_kva, e_kva)
            if e_kw is not None:
                assert torch.equal(g_kw, e_kw)
            _check_outputs(
                attn, static_inputs[rows], g_kva, g_kw, f"graph layer={layer} rows={rows}"
            )


@pytest.mark.parametrize(
    "cfg", ["b_load=cg", "b_load=ca", "copy_method=tma,b_load=ca", "prewait_chunks=1"]
)
@pytest.mark.parametrize("stage_before_wait", [True, False])
@pytest.mark.parametrize("delay_ns", [0, 5000, 50000])
@pytest.mark.parametrize("layer", LAYERS[:1] + LAYERS[-1:])
def test_pdl_delayed_producer(layer, delay_ns, stage_before_wait, cfg):
    """The producer releases its dependents at entry and writes the
    activation only after ``delay_ns``; the projection (PDL, staging weights
    before its dependency wait) must read the final activation. The same
    activation buffer is rewritten every trial, so an activation line kept in
    L1 from the previous trial (``b_load=ca``) would show up as a mismatch."""
    attn, gamma = U.module(layer)
    proj = _projection(attn, _parse_cfg(cfg, stage_before_wait=stage_before_wait))
    if not proj.use_pdl:
        pytest.skip("PDL disabled (TRTLLM_ENABLE_PDL=0)")
    with torch.inference_mode():
        dst = torch.empty(6, U.H, dtype=torch.bfloat16, device="cuda")
        for trial in range(6):
            src = U.native_hidden(6, seed=500 + trial, gamma=gamma)
            dst.fill_(float("nan") if trial % 2 else 0.0)
            torch.cuda.synchronize()
            U.proxy_producer(src, dst, delay_ns)
            kva, kw = proj.run(dst)
            ref_kva, ref_kw = U.native(attn, src)
            torch.cuda.synchronize()
            assert torch.equal(kva, ref_kva), f"trial {trial}"
            if kw is not None:
                r_kva, r_kw = proj.run(src)
                torch.cuda.synchronize()
                assert torch.equal(kw, r_kw), f"trial {trial}"


@pytest.mark.parametrize(
    "cfg",
    [
        "stage_before_wait=0",
        "trigger=end",
        "trigger=none",
        "nsub=8",
        "nsub=2",
        "b_dist=4",
        "b_dist=24",
        "b_dist=48",
        "a_dist=4",
        "ctas=148",
        "ctas=150",
        "l2_prefetch=0",
        "producer_warp=0",
        "copy_method=cpasync,producer_warps=4",
        "copy_method=cpasync,producer_warps=2,nsub=8",
        "b_load=ca",
        "chunks=48-24-12-12",
        "chunks=40-32-16-8,b_load=ca",
        "chunks=64-32,copy_method=cpasync,producer_warps=4",
        "copy_method=tma",
        "copy_method=tma,b_load=ca,nsub=8",
        "copy_method=tma,nsub=24",
        "copy_method=tma,tma_groups=3,chunks=48-24-12-12",
        "copy_method=tma,producer_warp=0",
        "copy_method=tma,stage_before_wait=0,b_load=ca",
        "copy_method=tma,ctas=148",
        "b_load=cg",
        "nsub=4,b_load=cg",
        "prewait_chunks=1",
        "prewait_chunks=2,copy_method=tma",
        "prewait_chunks=0",
        "prewait_chunks=3,copy_method=cpasync,producer_warps=4",
    ],
)
@pytest.mark.parametrize("layer", LAYERS[:1] + LAYERS[3:4])
def test_launch_configs_are_numerically_identical(cfg, layer):
    """Launch-configuration knobs change the schedule only, never a bit."""
    attn, gamma = U.module(layer)
    proj = _projection(attn, _parse_cfg(cfg))
    ref = _projection(attn)
    with torch.inference_mode():
        for rows in (1, 6, 8):
            h = U.native_hidden(rows, seed=900 + rows, gamma=gamma)
            a = proj.run(h)
            b = ref.run(h)
            torch.cuda.synchronize()
            assert torch.equal(a[0], b[0])
            if b[1] is not None:
                assert torch.equal(a[1], b[1])


def test_corunning_kernels_progress():
    """The projection on one stream while another stream keeps native GEMMs
    of the same weights in flight: every launch completes and is correct (no
    inter-CTA dependency, no co-residency assumption)."""
    attn, gamma = U.module(LAYERS[0])
    proj = _projection(attn)
    side = torch.cuda.Stream()
    with torch.inference_mode():
        hs = [U.native_hidden(6, seed=700 + i, gamma=gamma) for i in range(8)]
        refs = [U.native(attn, h) for h in hs]
        torch.cuda.synchronize()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(40):
                for h in hs:
                    U.native(attn, h)
        outs = [proj.run(h) for h in hs for _ in range(5)]
        torch.cuda.synchronize()
        for i, (kva, kw) in enumerate(outs):
            r_kva, r_kw = refs[i // 5]
            assert torch.equal(kva, r_kva)
            if r_kw is not None:
                assert U.within_budget(
                    U.compare(
                        kw, r_kw, U.fp64_oracle(hs[i // 5], attn.mqa.indexer._fused_wk_wp_weight)
                    )
                )


def test_supported_rejects_out_of_domain():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.fused_kv_indexer_projection.proj_kernel import (  # noqa: E501
        projection_supported,
    )

    assert projection_supported(6144, 2784, 152, 4, 18, 4, 4) is None
    assert projection_supported(6144, 2624, 152, 4, 18, 4, 4) is None
    assert projection_supported(6144, 2784, 152, 4, 17, 4, 4) is not None  # 2 direct rows
    assert projection_supported(6144, 2784, 152, 4, 18, 4, 4, (48, 24, 12, 12)) is None
    assert projection_supported(6144, 2784, 152, 4, 18, 4, 4, (48, 24, 12)) is not None
    assert projection_supported(6144, 2784, 152, 4, 18, 4, 4, (), 160, "tma") is None
    assert projection_supported(6144, 2624, 152, 4, 18, 4, 4, (), 0, "tma") is None
    assert projection_supported(6144, 2784, 152, 4, 18, 4, 24, (), 160, "tma", 3) is not None
    assert projection_supported(6144, 2784, 152, 4, 18, 4, 8, (), 160, "tma", 3) is None
    assert projection_supported(6144, 2784, 100, 4, 18, 4, 4, (), 160, "tma") is not None
    assert projection_supported(6144, 2784, 100, 4, 18, 4, 4) is not None  # 28 rows per CTA
    assert projection_supported(6144, 2784, 152, 4, 18, 3, 4) is not None  # slices vs warps
