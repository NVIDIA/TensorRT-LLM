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
"""``head_linear``: the vocabulary shard GEMM of the target head and the drafter's block.

At the served shape ([rows, 7168] x [10240, 7168]^T), every row count the device's table covers
runs its kernel (not cuBLAS) and matches the FP64 product within BF16 rounding of an FP32 sum; row
counts above the table are ``bf16_linear`` bit for bit. Every Rubin candidate tile is checked the
same way on this device, so a Rubin row picked from them is a numerically checked kernel. A CUDA
graph of the call, replayed on new inputs, gives the eager call's bits.

The selector gate (``selector_gate_linear``, [7, 7168] x [256, 7168]^T) keeps cuBLAS bit for bit
without a table row, and every kernel a Rubin row may take is checked the same way. The DSA
in-projection takes explicit warp split-K tactics (rings past vLLM's 16-stage check included):
ring depth and B-loader warps give the rule tactic's bits, a 128-wide K tile stays within BF16
rounding of the FP64 product.

The drafter's side-chain projections (the stacked history kv_a and fc, ``draft_side_choice``) keep
the plan / cuBLAS bit for bit without a model config field or a Rubin row; every kernel a Rubin row
may take is checked against the FP64 product, and a CUDA graph of it replays the eager bits.
"""

import importlib

import pytest
import torch

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
)
N, K = 10240, 7168
#: Tiles a Rubin row may take: warp split-K with 32- or 16-row output tiles, token tile 8 or 16,
#: ring depth 4 or 8; cluster split-K on 160 CTAs.
RUBIN_CANDIDATES = (
    *(
        ("warp", tile, tokens, 128, stages, 2)
        for tile in (32, 16)
        for tokens in (8, 16)
        for stages in (4, 8)
    ),
    ("splitk", 128, 8, 2, 6),
    ("splitk", 128, 16, 2, 6),
    ("splitk", 64, 8, 1, 12),
    ("splitk", 64, 16, 1, 11),
)


def _sm100_family():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


pytestmark = pytest.mark.skipif(not _sm100_family(), reason="needs an SM 10.x GPU")


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(_PKG + ".modeling")


@pytest.fixture(scope="module")
def weight():
    g = torch.Generator(device="cuda").manual_seed(3)
    return (0.02 * torch.randn(N, K, generator=g, device="cuda")).bfloat16()


def _x(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, K, generator=g, device="cuda").bfloat16()


def _close(got, x, weight):
    ref = x.double() @ weight.double().t()
    err = (got.double() - ref).abs().max() / ref.abs().max()
    return err.item() < 1e-2


@pytest.mark.parametrize("rows", [1, 7, 8, 9, 14, 16, 17, 32])
def test_table_rows_run_their_kernel(mod, weight, rows):
    x = _x(rows, rows)
    choice = mod._head_gemm_choice(x, weight)
    bounds = mod._HEAD_GEMM_ROWS_RUBIN if mod._rubin_rows(x.device) else mod._HEAD_GEMM_ROWS
    covered = bool(bounds) and rows <= bounds[-1][0]
    assert (choice is not None) == covered
    got = mod.head_linear(x, weight)
    assert got.shape == (rows, N) and got.dtype == torch.bfloat16
    if covered:
        direct = mod._head_gemm(x, weight, choice)
        assert torch.equal(got.view(torch.int16), direct.view(torch.int16))
        assert _close(got, x, weight)
    else:
        assert torch.equal(got.view(torch.int16), mod.bf16_linear(x, weight).view(torch.int16))


@pytest.mark.parametrize("choice", RUBIN_CANDIDATES, ids=str)
@pytest.mark.parametrize("rows", [7, 8, 14, 16])
def test_rubin_candidates(mod, weight, choice, rows):
    if choice[0] == "splitk" and choice[2] < rows:
        pytest.skip("the split-K token tile covers the rows in one tile")
    x = _x(rows, 100 + rows)
    got = mod._head_gemm(x, weight, choice)
    assert _close(got, x, weight)


def test_strided_or_unaligned_rows_keep_cublas(mod, weight):
    x = _x(9, 5)[1:]  # 8 rows at a 14336 B offset: aligned and packed, takes the table
    assert mod._head_gemm_choice(x, weight) is not None or not mod._HEAD_GEMM_ROWS
    y = _x(8, 6)[:, :K]
    z = torch.empty(8, K + 8, dtype=torch.bfloat16, device="cuda")[:, :K]
    z.copy_(y)
    assert mod._head_gemm_choice(z, weight) is None  # not packed rows


@pytest.mark.parametrize("rows", [7, 8, 16])
def test_graph_replay_matches_eager(mod, weight, rows):
    x = _x(rows, 7)
    mod.head_linear(x, weight)  # prepares the engine outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mod.head_linear(x, weight)
    for seed in (11, 12, 13):
        x.copy_(_x(rows, seed))
        graph.replay()
        torch.cuda.synchronize()
        eager = mod.head_linear(x, weight)
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


# ---- the selector gate (``selector_gate_linear``) and the DSA in-projection tactics ----------

SEL_N = 256
#: Kernels a Rubin selector-gate row may take at 7 rows ([7, 7168] x [256, 7168]^T).
SELECTOR_CANDIDATES = (
    "direct:128:1",
    "direct:128:2",
    "direct:64:2",
    "direct:128:4",
    "splitk:64:8:4:12",
    "splitk:64:8:2:12",
    "warp:16:8:256:16:2",
    "warp:16:8:256:8:2",
    "warp:16:8:128:16:2",
)
#: The DSA in-projection's Rubin rule tactic at 8 rows, and the candidates a Rubin row may take.
#: Ring depth and B-loader warps do not change an output's K order (each compute warp owns the
#: same K quarter of every K tile), so those give the rule's bits; a 128-wide K tile does not.
INPROJ_RULE = "16:8:256:16:2"
INPROJ_BITWISE = (
    "16:8:256:12:2",
    "16:8:256:14:2",
    "16:8:256:17:2",
    "16:8:256:18:2",
    "16:8:256:16:1",
)
INPROJ_REORDERED = ("16:8:128:32:2",)


@pytest.fixture(scope="module")
def features():
    return importlib.import_module(
        "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.common.rubin_ull_features"
    )


@pytest.fixture(scope="module")
def sel_weight():
    g = torch.Generator(device="cuda").manual_seed(4)
    return (0.02 * torch.randn(SEL_N, K, generator=g, device="cuda")).bfloat16()


def test_selector_gate_rows_use_tested_candidates(mod):
    """Every table row is one of ``SELECTOR_CANDIDATES`` (the choices this file checks against
    FP64, in graph replay and against cuBLAS), with increasing row bounds inside the gate's range."""
    tested = {mod.parse_gemm_choice(spec) for spec in SELECTOR_CANDIDATES}
    for table in (mod._SELECTOR_GATE_ROWS, mod._SELECTOR_GATE_ROWS_RUBIN):
        bounds = [bound for bound, _ in table]
        assert bounds == sorted(set(bounds)) and all(1 <= b <= 32 for b in bounds)
        assert all(choice in tested for _, choice in table)


def test_selector_gate_keeps_cublas_without_a_row(mod, sel_weight):
    x = _x(7, 21).view(1, 7, K)
    if not mod._SELECTOR_GATE_ROWS_RUBIN and not mod._SELECTOR_GATE_ROWS:
        assert mod.selector_gate_choice(x.view(7, K), sel_weight) is None
    got = mod.selector_gate_linear(x, sel_weight)
    want = torch.nn.functional.linear(x, sel_weight)
    if mod.selector_gate_choice(x.view(7, K), sel_weight) is None:
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("spec", SELECTOR_CANDIDATES)
def test_selector_gate_candidates(mod, features, sel_weight, spec):
    x = _x(7, 22).view(1, 7, K)
    with features.override(selector_gate=spec):
        assert mod.selector_gate_choice(x.view(7, K), sel_weight) == mod.parse_gemm_choice(spec)
        got = mod.selector_gate_linear(x, sel_weight)
    assert got.shape == (1, 7, SEL_N)
    assert _close(got.view(7, SEL_N), x.view(7, K), sel_weight)


@pytest.mark.parametrize("spec", ["direct:128:2", "warp:16:8:256:16:2"])
def test_selector_gate_graph_replay_matches_eager(mod, features, sel_weight, spec):
    x = _x(7, 23).view(1, 7, K)
    with features.override(selector_gate=spec):
        mod.selector_gate_linear(x, sel_weight)  # prepares a warp engine outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = mod.selector_gate_linear(x, sel_weight)
        for seed in (24, 25):
            x.copy_(_x(7, seed).view(1, 7, K))
            graph.replay()
            torch.cuda.synchronize()
            eager = mod.selector_gate_linear(x, sel_weight)
            assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


@pytest.fixture(scope="module")
def inproj_weight(mod):
    g = torch.Generator(device="cuda").manual_seed(5)
    return (0.02 * torch.randn(mod.PITCH, mod.HIDDEN, generator=g, device="cuda")).bfloat16()


def _inproj(mod, weight, spec, x):
    ip = mod.DsaInProjection(weight, {8: mod.parse_warp_tactic(spec)})
    ip.prepare([8])
    assert ip.engine.tactics[8] == mod.parse_warp_tactic(spec)
    assert ip.ready(x)
    return ip(x)


@pytest.mark.parametrize("spec", INPROJ_BITWISE + INPROJ_REORDERED)
def test_inproj_tactics(mod, inproj_weight, spec):
    x = _x(8, 31)
    rule = _inproj(mod, inproj_weight, INPROJ_RULE, x).clone()
    got = _inproj(mod, inproj_weight, spec, x)
    assert _close(got, x, inproj_weight)
    if spec in INPROJ_BITWISE:
        assert torch.equal(got.view(torch.int16), rule.view(torch.int16))


def test_inproj_tactic_from_the_model_config(mod, features, inproj_weight):
    with features.override(dsa_inproj_tactic="16:8:256:18:2"):
        tactics = mod.dsa_inproj_tactics(inproj_weight.device)
    assert sorted(tactics) == [3, 4, 5, 6, 7, 8]
    assert all(t.stages == 18 for t in tactics.values())
    if not mod._DSA_INPROJ_TACTICS_RUBIN:
        assert mod.dsa_inproj_tactics(inproj_weight.device) == {}
    with pytest.raises(ValueError):
        features.K31Features(dsa_inproj_tactic="16,8,256,18,2")
    with pytest.raises(ValueError):
        mod._check_ring_tactic(mod.parse_warp_tactic("16:8:256:19:2"), mod.PITCH, mod.HIDDEN)


# ---- the drafter's side-chain projections (``draft_side_choice``) ------------------------------

#: (weight shape, model config field, the kernels a Rubin row may take). kv_stack: the plan's
#: split-K family with split 4 (one wave of 180 CTAs on Rubin) and MR8's projection at the same
#: tile; fc: MR8's projection at split 8 (FlashInfer's split-K takes split 1-4) and its neighbours.
SIDE_CANDIDATES = {
    "kv_stack": (
        (2880, 7168),
        "draft_kv_stack_gemm",
        ("splitk:64:8:4:8", "splitk:64:8:4:12", "proj:64:8:4:8", "proj:64:8:2:12"),
    ),
    "fc": (
        (448, 43008),
        "draft_fc_gemm",
        ("proj:64:8:8:11", "proj:64:8:4:11", "proj:64:8:12:11", "splitk:64:8:4:12"),
    ),
}


@pytest.fixture(scope="module")
def side_weights():
    g = torch.Generator(device="cuda").manual_seed(6)
    return {
        site: (0.02 * torch.randn(*shape, generator=g, device="cuda")).bfloat16()
        for site, (shape, _, _) in SIDE_CANDIDATES.items()
    }


def _side_x(k, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, k, generator=g, device="cuda").bfloat16()


def test_side_rows_use_tested_candidates(mod):
    tested = {
        (shape, mod.parse_gemm_choice(spec))
        for shape, _, specs in SIDE_CANDIDATES.values()
        for spec in specs
    }
    assert all((shape, choice) in tested for shape, choice in mod._DRAFT_SIDE_GEMM_RUBIN.items())


@pytest.mark.parametrize("site", sorted(SIDE_CANDIDATES))
def test_side_projection_keeps_its_choice_without_a_row(mod, side_weights, site):
    """No model config and no Rubin row: ``draft_linear`` is the plan / cuBLAS, bit for bit."""
    shape, _, _ = SIDE_CANDIDATES[site]
    x = _side_x(shape[1], 8, 41)
    if not mod._DRAFT_SIDE_GEMM_RUBIN:
        assert mod.draft_side_choice(x, side_weights[site]) is None
    if mod.draft_side_choice(x, side_weights[site]) is not None:
        pytest.skip("this device has a Rubin row for the shape")
    got = mod.draft_linear(x, side_weights[site])
    policies = mod._DRAFT_GEMM_PLAN.get(shape, ())
    if policies:
        want = torch.empty_like(got)
        mod.bf16_low_m_gemm(x, side_weights[site], want, policies[0][1], True)
    else:
        want = mod.rubin_linear(x, side_weights[site], False)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("rows", [1, 7, 8])
@pytest.mark.parametrize(
    "site,spec", [(s, c) for s, (_, _, cs) in sorted(SIDE_CANDIDATES.items()) for c in cs]
)
def test_side_candidates(mod, features, side_weights, site, spec, rows):
    shape, field, _ = SIDE_CANDIDATES[site]
    x = _side_x(shape[1], rows, 42 + rows)
    with features.override(**{field: spec}):
        assert mod.draft_side_choice(x, side_weights[site]) == mod.parse_gemm_choice(spec)
        got = mod.draft_linear(x, side_weights[site])
    assert got.shape == (rows, shape[0])
    assert _close(got, x, side_weights[site])


@pytest.mark.parametrize("site,spec", [("kv_stack", "splitk:64:8:4:8"), ("fc", "proj:64:8:8:11")])
def test_side_graph_replay_matches_eager(mod, features, side_weights, site, spec):
    shape, field, _ = SIDE_CANDIDATES[site]
    x = _side_x(shape[1], 8, 51)
    with features.override(**{field: spec}):
        mod.draft_linear(x, side_weights[site])  # prepares the projection outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = mod.draft_linear(x, side_weights[site])
        for seed in (52, 53):
            x.copy_(_side_x(shape[1], 8, seed))
            graph.replay()
            torch.cuda.synchronize()
            eager = mod.draft_linear(x, side_weights[site])
            assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


def test_side_choice_from_the_model_config(mod, features, side_weights):
    x = _side_x(7168, 8, 61)
    w = side_weights["kv_stack"]
    with features.override(draft_kv_stack_gemm="cublas"):
        assert mod.draft_side_choice(x, w) == ("cublas",)
        got = mod.draft_linear(x, w)
    assert torch.equal(got.view(torch.int16), mod.rubin_linear(x, w, False).view(torch.int16))
    with features.override(draft_kv_stack_gemm="proj:64:8:4:8"):
        assert mod.draft_side_choice(_side_x(7168, 9, 62), w) is None  # past one request's block
    for bad in ("proj:64:8:4", "warp:16:8:256:16:2", "splitk,64,8,4,8"):
        with pytest.raises(ValueError):
            features.K31Features(draft_fc_gemm=bad)


def test_side_choice_only_takes_its_shapes(mod, features):
    """Every other drafter projection keeps ``draft_linear``'s choice under both fields."""
    with features.override(draft_kv_stack_gemm="proj:64:8:4:8", draft_fc_gemm="proj:64:8:8:11"):
        for shape in set(mod._DRAFT_GEMM_PLAN) | set(mod._DRAFT_WARP_RUBIN):
            if shape in mod._DRAFT_SIDE_FIELDS:
                continue
            w = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
            assert mod.draft_side_choice(_side_x(shape[1], 8, 64), w) is None, shape
