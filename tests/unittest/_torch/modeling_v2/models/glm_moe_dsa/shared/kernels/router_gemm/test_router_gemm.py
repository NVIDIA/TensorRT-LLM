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
"""Single-GPU correctness of the router_gemm against the native oracle.

The oracle is the native ``torch.ops.trtllm.dsv3_router_gemm_op`` that
``DeepseekV3Gate`` runs for this model. The router_gemm kernel reproduces its
arithmetic, so every comparison below is bitwise (FP32 bit patterns), not a
tolerance.
"""

import ast
import json
import os
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

import tensorrt_llm  # noqa: F401  (loads the native torch ops)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.router_gemm import (  # noqa: E501
    hop,
    router_gemm,
    router_gemm_supported,
    router_logits,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not 100
    <= torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
    < 110,
    reason="needs an SM100-family GPU",
)

HIDDEN = 6144
EXPERTS = 256
TOP_K = 8
ROUTED_SCALING = 2.5

# GLM-5.2 NVFP4 checkpoint directory. Router (mlp.gate) and
# post_attention_layernorm weights are BF16 there and are read tensor by tensor;
# tests that need them skip when it is absent. Activations are always synthetic
# (real per-layer activations need a captured fixture).
CKPT_ENV = "GLM52_NVFP4_CKPT"


@pytest.fixture(scope="module")
def real_layer_weights():
    """Return ``fn(layer) -> dict`` of real gate/bias/norm weights, or skip."""
    root = llm_models_root()
    default = os.path.join(str(root), "GLM-5.2-NVFP4") if root is not None else ""
    ckpt = os.environ.get(CKPT_ENV, default)
    index = os.path.join(ckpt or "", "model.safetensors.index.json")
    if not ckpt or not os.path.exists(index):
        pytest.skip("GLM-5.2 checkpoint not available")
    from safetensors import safe_open

    with open(index) as f:
        weight_map = json.load(f)["weight_map"]
    cache = {}

    def load(name):
        with safe_open(os.path.join(ckpt, weight_map[name]), framework="pt") as f:
            return f.get_tensor(name)

    def get(layer):
        if layer not in cache:
            pre = f"model.layers.{layer}"
            cache[layer] = {
                "gate": load(f"{pre}.mlp.gate.weight"),
                "bias": load(f"{pre}.mlp.gate.e_score_correction_bias"),
                "post_norm": load(f"{pre}.post_attention_layernorm.weight"),
            }
        return cache[layer]

    return get


def native_logits(hidden, weight):
    return torch.ops.trtllm.dsv3_router_gemm_op(
        hidden, weight.t(), bias=None, out_dtype=torch.float32
    )


def assert_bitwise(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    a_bits = actual.contiguous().view(torch.int32)
    e_bits = expected.contiguous().view(torch.int32)
    mismatch = (a_bits != e_bits).sum().item()
    if mismatch:
        diff = (actual.double() - expected.double()).abs().max().item()
        raise AssertionError(
            f"{mismatch}/{actual.numel()} logits differ in bits (max abs diff {diff:.3e})"
        )


def normed_activation(rows, hidden, gen, norm_weight=None, scale=1.0):
    """Synthetic post-RMSNorm activation, BF16 (as the AR/norm emits)."""
    x = torch.randn((rows, hidden), generator=gen, device="cuda") * scale
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    if norm_weight is not None:
        x = x * norm_weight.float()
    return x.to(torch.bfloat16)


def random_weight(hidden, gen, scale=0.02):
    return (torch.randn((EXPERTS, hidden), generator=gen, device="cuda") * scale).to(torch.bfloat16)


@pytest.fixture
def gen():
    g = torch.Generator(device="cuda")
    g.manual_seed(1234)
    return g


@pytest.mark.parametrize("hidden_size", [6144, 7168, 4096])
@pytest.mark.parametrize("rows", list(range(1, 17)))
def test_bitwise_matches_native_random(rows, hidden_size, gen):
    weight = random_weight(hidden_size, gen)
    for scale in (1.0, 37.0):
        hidden = normed_activation(rows, hidden_size, gen, scale=scale)
        ok, reason = router_gemm_supported(hidden, weight)
        if not ok:
            # Only the staged activation can exceed shared memory (K=7168).
            assert hidden_size == 7168 and "shared memory" in reason
            assert router_logits(hidden, weight) is None
            continue
        assert_bitwise(router_gemm(hidden, weight), native_logits(hidden, weight))


@pytest.mark.parametrize("experts_per_cta", [1, 2, 4, 8])
@pytest.mark.parametrize("use_pdl", [True, False])
@pytest.mark.parametrize("prefetch_weights", [True, False])
@pytest.mark.parametrize("paired_fma", [True, False])
@pytest.mark.parametrize("rows", [5, 6])
def test_bitwise_schedule_variants(
    experts_per_cta, use_pdl, prefetch_weights, paired_fma, rows, gen
):
    weight = random_weight(HIDDEN, gen)
    hidden = normed_activation(rows, HIDDEN, gen)
    out = router_gemm(
        hidden,
        weight,
        experts_per_cta=experts_per_cta,
        use_pdl=use_pdl,
        prefetch_weights=prefetch_weights,
        paired_fma=paired_fma,
    )
    assert_bitwise(out, native_logits(hidden, weight))


@pytest.mark.parametrize("layer", [3, 4, 40, 77])
@pytest.mark.parametrize("rows", [1, 6, 16])
def test_bitwise_matches_native_real_weights(layer, rows, real_layer_weights, gen):
    """Real GLM-5.2 router/norm weights; synthetic activations."""
    w = real_layer_weights(layer)
    weight = w["gate"].cuda()
    norm_weight = w["post_norm"].cuda()
    hidden = normed_activation(rows, HIDDEN, gen, norm_weight=norm_weight)
    assert_bitwise(router_gemm(hidden, weight), native_logits(hidden, weight))


def test_bitwise_special_values(gen):
    """Zeros, signed zeros, cancellation, wide range and underflowing products."""
    weight = random_weight(HIDDEN, gen)
    rows = []
    rows.append(torch.zeros(HIDDEN, device="cuda"))
    rows.append(-torch.zeros(HIDDEN, device="cuda"))
    alt = torch.ones(HIDDEN, device="cuda")
    alt[1::2] = -1
    rows.append(alt * 3.0)
    wide = torch.randn(HIDDEN, generator=gen, device="cuda")
    wide = wide * torch.exp2(torch.randint(-60, 60, (HIDDEN,), generator=gen, device="cuda"))
    rows.append(wide)
    rows.append(torch.randn(HIDDEN, generator=gen, device="cuda") * 1e-30)  # tiny
    rows.append(torch.randn(HIDDEN, generator=gen, device="cuda") * 1e30)
    hidden = torch.stack(rows).to(torch.bfloat16)
    weight_tiny = weight.clone()
    weight_tiny[:16] = (weight_tiny[:16].float() * 1e-20).to(torch.bfloat16)
    for w in (weight, weight_tiny):
        for epc, paired in ((1, True), (1, False), (2, True)):
            assert_bitwise(
                router_gemm(hidden, w, experts_per_cta=epc, paired_fma=paired),
                native_logits(hidden, w),
            )
    neg_zero = torch.full((1, HIDDEN), -0.0, dtype=torch.bfloat16, device="cuda")
    assert_bitwise(router_gemm(neg_zero, -weight.abs()), native_logits(neg_zero, -weight.abs()))


def native_route(logits, bias):
    return torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, ROUTED_SCALING)


@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_routing_decisions_identical_with_ties(bias_dtype, gen):
    """Exact logit ties (duplicated expert rows) route identically."""
    weight = random_weight(HIDDEN, gen)
    weight[17] = weight[200]
    weight[33] = weight[34]
    bias = torch.zeros(EXPERTS, dtype=bias_dtype, device="cuda")
    hidden = normed_activation(16, HIDDEN, gen)
    ours = router_gemm(hidden, weight)
    ref = native_logits(hidden, weight)
    assert_bitwise(ours, ref)
    assert torch.equal(ours[:, 17].view(torch.int32), ours[:, 200].view(torch.int32))
    v_ours, i_ours = native_route(ours, bias)
    v_ref, i_ref = native_route(ref, bias)
    assert torch.equal(i_ours, i_ref)
    assert torch.equal(v_ours.view(torch.int32), v_ref.view(torch.int32))


def test_routing_decisions_real_weights(real_layer_weights, gen):
    w = real_layer_weights(40)
    weight = w["gate"].cuda()
    bias = w["bias"].cuda().to(torch.bfloat16)
    norm_weight = w["post_norm"].cuda()
    for _ in range(8):
        hidden = normed_activation(6, HIDDEN, gen, norm_weight=norm_weight)
        ours = router_gemm(hidden, weight)
        ref = native_logits(hidden, weight)
        assert_bitwise(ours, ref)
        assert torch.equal(native_route(ours, bias)[1], native_route(ref, bias)[1])


def test_cuda_graph_replay(gen):
    """Producer -> router inside one graph; replay with changing inputs."""
    weight = random_weight(HIDDEN, gen)
    source = normed_activation(6, HIDDEN, gen)
    hidden = torch.empty_like(source)
    out_native = torch.empty((6, EXPERTS), dtype=torch.float32, device="cuda")
    router_gemm(source, weight)  # compile outside capture
    native_logits(source, weight)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            hidden.copy_(source)
            out_ours = router_gemm(hidden, weight)
            out_native.copy_(native_logits(hidden, weight))
    torch.cuda.current_stream().wait_stream(stream)
    for step in range(20):
        source.copy_(normed_activation(6, HIDDEN, gen, scale=1.0 + step))
        graph.replay()
        torch.cuda.synchronize()
        assert_bitwise(out_ours, out_native)
        assert_bitwise(out_ours, native_logits(source, weight))


def test_padded_rows_do_not_affect_live_rows(gen):
    weight = random_weight(HIDDEN, gen)
    hidden = normed_activation(6, HIDDEN, gen)
    live = 2
    base = router_gemm(hidden, weight).clone()
    hidden[live:] = normed_activation(6 - live, HIDDEN, gen, scale=1e3)
    changed = router_gemm(hidden, weight)
    assert_bitwise(changed[:live], base[:live])
    hidden[live:] = 0
    zero_rows = router_gemm(hidden, weight)[live:]
    assert torch.all(zero_rows == 0)
    assert_bitwise(router_gemm(hidden, weight), native_logits(hidden, weight))


def test_ineligible_calls_return_none(gen):
    weight = random_weight(HIDDEN, gen)
    hidden = normed_activation(6, HIDDEN, gen)
    cases = {
        "fp16": (hidden.half(), weight.half()),
        "rows_17": (normed_activation(17, HIDDEN, gen), weight),
        "rows_0": (hidden[:0], weight),
        "experts_128": (hidden, weight[:128]),
        "hidden_5120": (hidden[:, :5120].contiguous(), weight[:, :5120].contiguous()),
        "non_contiguous": (hidden.t().contiguous().t(), weight),
        "misaligned": (
            torch.empty(6 * HIDDEN + 1, dtype=torch.bfloat16, device="cuda")[1:].view(6, HIDDEN),
            weight,
        ),
        "cpu": (hidden.cpu(), weight.cpu()),
    }
    before = hop.engagement_counts()["native_fallback"]
    for name, (h, w) in cases.items():
        ok, reason = router_gemm_supported(h, w)
        assert not ok and reason, name
        assert router_logits(h, w) is None, name
    assert hop.engagement_counts()["native_fallback"] == before + len(cases)
    assert router_logits(hidden, weight) is not None


def _fake_model_config(tp_size=8, attention_dp=False, layers=78):
    return SimpleNamespace(
        mapping=SimpleNamespace(enable_attention_dp=attention_dp, tp_size=tp_size),
        pretrained_config=SimpleNamespace(num_hidden_layers=layers),
    )


def _gate(use_cute_dsl_bf16_gemm=False):
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate

    gate = DeepseekV3Gate(
        HIDDEN,
        EXPERTS,
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=ROUTED_SCALING,
        dtype=torch.bfloat16,
        moe_backend="TRTLLM",
        use_cute_dsl_bf16_gemm=use_cute_dsl_bf16_gemm,
    )
    return gate.cuda()


def test_make_moe_router_domain(gen):
    cfg = _fake_model_config()
    gate = _gate()
    with torch.no_grad():
        gate.weight.copy_(random_weight(HIDDEN, gen))
    route = hop.make_moe_router(gate, 3, cfg)
    assert route is not None
    hidden = normed_activation(6, HIDDEN, gen)
    assert_bitwise(route(hidden), gate(hidden))
    assert route(hidden.half()) is None
    assert hop.make_moe_router(gate, 78, cfg) is None  # MTP layer
    assert hop.make_moe_router(gate, None, cfg) is None
    assert hop.make_moe_router(gate, 3, _fake_model_config(attention_dp=True)) is None
    assert hop.make_moe_router(gate, 3, _fake_model_config(tp_size=1)) is None
    assert hop.make_moe_router(_gate(use_cute_dsl_bf16_gemm=True), 3, cfg) is None


def test_registered_entry_points():
    for name in ("router_logits", "pre_moe_hop", "make_moe_router"):
        assert callable(dispatch.get_impl("router_gemm", name))


def _enclosing_conditions(func, target):
    """Conjuncts of the ``if`` tests whose body contains ``target``.

    A ``target`` inside an ``else`` branch yields an ``<else>`` entry, so it
    never matches a required condition.
    """

    def visit(node, conditions):
        if node is target:
            return conditions
        if isinstance(node, ast.If):
            test = node.test
            parts = (
                test.values
                if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
                else [test]
            )
            for branch, extra in (
                (node.body, [ast.unparse(p) for p in parts]),
                (node.orelse, ["<else>"]),
            ):
                for child in branch:
                    found = visit(child, conditions + extra)
                    if found is not None:
                        return found
            return None
        for child in ast.iter_child_nodes(node):
            found = visit(child, conditions)
            if found is not None:
                return found
        return None

    return visit(func, [])


def _check_router_call_site(source, class_name):
    """The call-site semantics of ``<class_name>.compute_routed_output``.

    One router_gemm router call, guarded by the router's conditions, before one native
    gate call that runs only when no logits exist (the fallback). The call site also
    takes the post_attention_hop's logits as the ``router_logits`` argument; router_gemm
    must then not overwrite them.
    """
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    func = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_routed_output"
    )
    calls = {}
    for callee in ("self.router_gemm", "self.gate"):
        found = [
            node
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
        ]
        assert len(found) == 1, (callee, len(found))
        calls[callee] = found[0]
    router_gemm_call, gate_call = calls["self.router_gemm"], calls["self.gate"]
    assert router_gemm_call.lineno < gate_call.lineno

    router_guard = {"self.router_gemm is not None", "not use_dp_padding"}
    router_gemm_guard = set(_enclosing_conditions(func, router_gemm_call))
    assert router_guard <= router_gemm_guard <= router_guard | {"router_logits is None"}, (
        router_gemm_guard
    )
    if "router_logits" in [arg.arg for arg in func.args.args]:
        assert "router_logits is None" in router_gemm_guard, router_gemm_guard
    gate_guard = _enclosing_conditions(func, gate_call)
    assert gate_guard == ["router_logits is None"], gate_guard

    assigned = {
        ast.unparse(node.value)
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "router_logits" for t in node.targets)
    }
    assert assigned <= {"None", ast.unparse(router_gemm_call), ast.unparse(gate_call)}, assigned


def test_router_call_site_in_target():
    """The target MoE's router call site: one router_gemm selection (in ``GlmMoE.adopt``) and
    the call-site semantics in ``GlmMoE.compute_routed_output``.
    """
    from pathlib import Path

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import layers

    source = Path(layers.__file__).read_text()
    compile(source, "layers.py", "exec")
    assert source.count('dispatch.is_enabled("router_gemm")') == 1
    assert source.count("router_logits = self.gate(hidden_states)") == 1
    _check_router_call_site(source, "GlmMoE")
