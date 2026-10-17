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
"""GPU tests of package dense_mlp (small-M BF16 dense MLP).

Oracle: the native BF16 GatedMLP compute (cuBLAS ``F.linear`` -> TRT-LLM
``swiglu`` -> ``F.linear``) and an FP64 reference with native BF16 rounding.
Tolerance against native: the existing TRT-LLM small-M BF16 GEMM tolerance
(``tests/unittest/_torch/modules/test_low_m_gemm.py``: rtol 1e-2, atol 5e-3)
for each GEMM stage. Error budget against the FP64 oracle, frozen before
tuning: candidate normalized RMS error <= 1.25 x native + 1e-4.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
from utils.llm_data import llm_models_root

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

NRMS_FACTOR = 1.25
NRMS_SLACK = 1e-4


@pytest.fixture(scope="module", autouse=True)
def dense_mlp_op():
    import dense_mlp_utils as wu

    return wu.ensure_op()


@pytest.fixture(autouse=True)
def _seed_inputs():
    """Seed the global generator so each case draws the same inputs in any order."""
    torch.manual_seed(0)


def _random_weights(hidden, inter, scale=0.02, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = (scale * torch.randn(2 * inter, hidden, device="cuda", generator=g)).to(
        torch.bfloat16
    )
    down = (scale * torch.randn(hidden, inter, device="cuda", generator=g)).to(torch.bfloat16)
    return gate_up, down


def assert_within_bf16_ulps(actual, expected, ulps, abs_floor_rel=1e-5):
    """|actual - expected| <= ulps BF16 ulps of the larger magnitude + floor.

    The absolute floor (``abs_floor_rel`` x RMS of ``expected``) covers values
    near zero, where a different FP32 summation order of a cancelling sum
    moves the result by far more than its own (tiny) ulp.
    """
    a = actual.float()
    e = expected.float()
    mag = torch.maximum(a.abs(), e.abs()).clamp_min(2.0**-126)
    ulp = torch.exp2(torch.floor(torch.log2(mag)) - 7)
    floor = abs_floor_rel * e.pow(2).mean().sqrt().item()
    excess = ((a - e).abs() - ulps * ulp - floor).max().item()
    assert excess <= 0, f"difference exceeds {ulps} BF16 ulps by {excess}"


# SwiGLU of BF16-rounded gate and up, rounded to BF16: when another FP32
# summation order flips the rounding of gate and of up by one ulp each, the
# product moves by up to two ulps, plus one ulp from rounding h.
SWIGLU_ULPS = 3


def _check_against_native_and_oracle(x, gate_up, down, variants=(0, 0)):
    """Per-stage native comparison plus the end-to-end FP64-oracle budget.

    The down stage (one BF16 rounding) uses the existing small-M BF16 GEMM
    tolerance against the native down GEMM on the candidate's own SwiGLU
    output. The gate/up + SwiGLU stage has three rounding points (gate, up,
    h) and is bounded by SWIGLU_ULPS against native. End to end, the frozen
    FP64-oracle error budget applies.
    """
    import dense_mlp_utils as wu

    h = torch.ops.trtllm.glm_moe_dsa_dense_mlp_gate_up_swiglu(x, gate_up, variants[0])
    assert_within_bf16_ulps(h, wu.native_intermediate(x, gate_up), SWIGLU_ULPS)
    out = torch.ops.trtllm.glm_moe_dsa_dense_mlp_down(h, down, variants[1])
    torch.testing.assert_close(out, F.linear(h, down), rtol=1e-2, atol=5e-3)
    cand = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x, gate_up, down, *variants)
    assert torch.equal(cand, out)
    nat = wu.native_dense_mlp(x, gate_up, down)
    _, ref = wu.oracle_dense_mlp(x, gate_up, down)
    cand_err = wu.error_stats(cand, ref)["nrms"]
    nat_err = wu.error_stats(nat, ref)["nrms"]
    assert cand_err <= NRMS_FACTOR * nat_err + NRMS_SLACK, (cand_err, nat_err)
    return cand


# Contraction multiple (warps x 32 x steps) of each variant in
# denseMlp.cu.
VARIANT_K_MULTIPLE = [4 * 32 * 8, 8 * 32 * 4, 8 * 32 * 4, 4 * 32 * 8, 8 * 32 * 8]

# (gate/up variant, down variant) pairs exercised by the shape-generic tests.
VARIANT_PAIRS = [(0, 0), (1, 0), (2, 1), (3, 3), (4, 3)]


def _supported(variants, hidden, inter):
    return (
        hidden % VARIANT_K_MULTIPLE[variants[0]] == 0
        and inter % VARIANT_K_MULTIPLE[variants[1]] == 0
    )


@pytest.mark.parametrize("num_rows", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("hidden,inter", [(6144, 3072), (1024, 1024), (2048, 4096)])
@pytest.mark.parametrize("variants", VARIANT_PAIRS)
def test_matches_native_random(num_rows, hidden, inter, variants):
    if not _supported(variants, hidden, inter):
        pytest.skip("tiling does not divide the shape")
    gate_up, down = _random_weights(hidden, inter)
    x = torch.randn(num_rows, hidden, device="cuda", dtype=torch.bfloat16)
    _check_against_native_and_oracle(x, gate_up, down, variants)


@pytest.mark.parametrize("num_rows", [1, 6, 8])
def test_intermediate_rounding_matches_native(num_rows):
    """gate/up/h rounding follows native; h differs only by summation order."""
    import dense_mlp_utils as wu

    gate_up, down = _random_weights(6144, 3072, seed=3)
    x = torch.randn(num_rows, 6144, device="cuda", dtype=torch.bfloat16)
    h = torch.ops.trtllm.glm_moe_dsa_dense_mlp_gate_up_swiglu(x, gate_up)
    nat_h = wu.native_intermediate(x, gate_up)
    ref_h, _ = wu.oracle_dense_mlp(x, gate_up, down)
    assert_within_bf16_ulps(h, nat_h, SWIGLU_ULPS)
    # Values that native reproduces exactly from the FP64 oracle must mostly
    # match exactly here too (only BF16 rounding-boundary cases may differ).
    assert (h == nat_h).float().mean().item() > 0.97
    assert wu.error_stats(h, ref_h)["nrms"] <= (
        NRMS_FACTOR * wu.error_stats(nat_h, ref_h)["nrms"] + NRMS_SLACK
    )


@pytest.mark.parametrize("variant", range(len(VARIANT_K_MULTIPLE)))
def test_all_variants(variant):
    gate_up, down = _random_weights(6144, 3072, seed=5)
    x = torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16)
    ref = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x, gate_up, down, 0, 0)
    h = torch.ops.trtllm.glm_moe_dsa_dense_mlp_gate_up_swiglu(x, gate_up, variant)
    h0 = torch.ops.trtllm.glm_moe_dsa_dense_mlp_gate_up_swiglu(x, gate_up, 0)
    assert_within_bf16_ulps(h, h0, SWIGLU_ULPS)
    if 3072 % VARIANT_K_MULTIPLE[variant] != 0:
        with pytest.raises(RuntimeError):
            torch.ops.trtllm.glm_moe_dsa_dense_mlp_down(h0, down, variant)
        return
    out = torch.ops.trtllm.glm_moe_dsa_dense_mlp_down(h0, down, variant)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=5e-3)


@pytest.mark.parametrize("variants", VARIANT_PAIRS)
def test_deterministic(variants):
    gate_up, down = _random_weights(6144, 3072, seed=7)
    x = torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16)
    first = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x, gate_up, down, *variants)
    for _ in range(5):
        assert torch.equal(
            torch.ops.trtllm.glm_moe_dsa_dense_mlp(x, gate_up, down, *variants), first
        )


@pytest.mark.parametrize("variants", VARIANT_PAIRS)
def test_cuda_graph_replay_tracks_inputs(variants):
    gate_up, down = _random_weights(6144, 3072, seed=11)
    x_static = torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops.trtllm.glm_moe_dsa_dense_mlp(x_static, gate_up, down, *variants)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x_static, gate_up, down, *variants)
    for seed in range(4):
        torch.manual_seed(100 + seed)
        new_x = torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16)
        x_static.copy_(new_x)
        graph.replay()
        eager = torch.ops.trtllm.glm_moe_dsa_dense_mlp(new_x, gate_up, down, *variants)
        torch.cuda.synchronize()
        assert torch.equal(out_static, eager)


def test_rejects_unsupported_inputs():
    gate_up, down = _random_weights(6144, 3072)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.glm_moe_dsa_dense_mlp(
            torch.randn(9, 6144, device="cuda", dtype=torch.bfloat16), gate_up, down
        )
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.glm_moe_dsa_dense_mlp(
            torch.randn(6, 6144, device="cuda", dtype=torch.float16), gate_up, down
        )
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.glm_moe_dsa_dense_mlp(
            torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16), gate_up, down, 99, 0
        )


def _make_gated_mlp(hidden, inter, tp_size=1, overridden_tp_size=None):
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=tp_size, rank=0, tp_size=tp_size, gpus_per_node=max(tp_size, 1))
    mlp = GatedMLP(
        hidden_size=hidden,
        intermediate_size=inter,
        bias=False,
        dtype=torch.bfloat16,
        config=ModelConfig(mapping=mapping),
        overridden_tp_size=overridden_tp_size,
        reduce_output=tp_size > 1,
    ).cuda()
    gate_up, down = _random_weights(hidden, inter // (overridden_tp_size or tp_size), seed=13)
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(gate_up)
        mlp.down_proj.weight.copy_(down)
    return mlp


@pytest.mark.parametrize("num_rows", [1, 6, 8])
def test_forward_matches_gated_mlp_module(num_rows):
    import dense_mlp_utils as wu

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.dense_mlp import (  # noqa: E501
        dense_mlp,
        dense_mlp_forward,
    )

    mlp = _make_gated_mlp(2048, 4096)
    x = torch.randn(num_rows, 2048, device="cuda", dtype=torch.bfloat16)
    cand = dense_mlp_forward(mlp, x)
    assert cand is not None
    # The entry point runs exactly the op with the module's weights and the
    # package's default tilings; the module's native forward is the
    # reference the op is checked against.
    gate_up, down = mlp.gate_up_proj.weight, mlp.down_proj.weight
    variants = (dense_mlp.GATE_UP_VARIANT, dense_mlp.DOWN_VARIANT)
    assert torch.equal(cand, _check_against_native_and_oracle(x, gate_up, down, variants))
    _, ref = wu.oracle_dense_mlp(x, gate_up, down)
    assert wu.error_stats(cand, ref)["nrms"] <= (
        NRMS_FACTOR * wu.error_stats(mlp(x), ref)["nrms"] + NRMS_SLACK
    )


def test_eligibility_falls_back_before_device_work():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.dense_mlp import (  # noqa: E501
        check_eligibility,
        dense_mlp_forward,
    )
    from tensorrt_llm._torch.distributed import AllReduceParams
    from tensorrt_llm._torch.utils import Fp4QuantizedTensor

    mlp = _make_gated_mlp(2048, 4096)
    x = torch.randn(6, 2048, device="cuda", dtype=torch.bfloat16)
    assert check_eligibility(mlp, x)[0]
    cases = {
        "rows": torch.randn(9, 2048, device="cuda", dtype=torch.bfloat16),
        "dtype": x.float(),
        "noncontig": torch.randn(2048, 6, device="cuda", dtype=torch.bfloat16).t(),
        "3d": x.view(1, 6, 2048),
        "fp4": Fp4QuantizedTensor(x, x),
    }
    for name, bad in cases.items():
        eligible, reason = check_eligibility(mlp, bad)
        assert not eligible, name
        assert dense_mlp_forward(mlp, bad) is None, name
    assert not check_eligibility(mlp, x, lora_params={"x": 1})[0]
    mlp.swiglu_limit = 7.0
    assert not check_eligibility(mlp, x)[0]
    mlp.swiglu_limit = None
    # Output all-reduce requested on a TP>1 down_proj -> native path. The
    # mapping is swapped in place so no communicator is needed.
    from tensorrt_llm.mapping import Mapping

    mlp.down_proj.reduce_output = True
    mlp.down_proj.mapping = Mapping(world_size=2, rank=0, tp_size=2, gpus_per_node=2)
    assert not check_eligibility(mlp, x)[0]
    assert not check_eligibility(mlp, x, AllReduceParams(enable_allreduce=True))[0]
    assert check_eligibility(mlp, x, AllReduceParams(enable_allreduce=False))[0]
    mlp.activation = F.gelu
    assert not check_eligibility(mlp, x, AllReduceParams(enable_allreduce=False))[0]


def test_attention_dp_takes_native_path():
    """Attention DP runs the dense MLP unsharded: outside dense_mlp's TP domain."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.dense_mlp import (  # noqa: E501
        check_eligibility,
        dense_mlp_forward,
    )
    from tensorrt_llm.mapping import Mapping

    mlp = _make_gated_mlp(2048, 4096)
    x = torch.randn(6, 2048, device="cuda", dtype=torch.bfloat16)
    assert check_eligibility(mlp, x)[0]
    mlp.mapping = Mapping(
        world_size=1, rank=0, tp_size=1, gpus_per_node=1, enable_attention_dp=True
    )
    assert check_eligibility(mlp, x) == (False, "attention data parallelism")
    assert dense_mlp_forward(mlp, x) is None


def test_registered_with_dispatch():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.dense_mlp import (  # noqa: E501
        dense_mlp_forward,
    )

    assert dispatch.get_impl("dense_mlp", "dense_mlp_forward") is dense_mlp_forward


@pytest.mark.skipif(
    not os.path.isdir(os.path.join(str(llm_models_root()), "GLM-5.2-NVFP4")),
    reason="GLM-5.2 checkpoint not mounted",
)
@pytest.mark.parametrize("layer", [0, 1, 2])
def test_real_weights_real_hidden(layer):
    import dense_mlp_utils as wu

    gate_up, down = wu.load_dense_shard(layer, shard=layer % 4)
    real = wu.load_real_hidden()
    rows = [real[i : i + 6] for i in (0, 777, 1500)] if real is not None else []
    rows.append(torch.randn(6, 6144, dtype=torch.bfloat16))
    for x in rows:
        for variants in VARIANT_PAIRS:
            _check_against_native_and_oracle(x.cuda().contiguous(), gate_up, down, variants)


def _pdl_delay_module():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get("GLM_MOE_DSA_DENSE_MLP_JIT_BUILD_DIR")
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)
    return load(
        name="dense_mlp_pdl_delay",
        sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdl_delay.cu")],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_103a,code=sm_103a"]
        if torch.cuda.get_device_capability() == (10, 3)
        else ["-O3"],
        build_directory=build_dir,
        verbose=False,
    )


@pytest.mark.skipif(
    os.environ.get("TRTLLM_ENABLE_PDL") != "1",
    reason="the dense_mlp kernels launch with PDL only when TRTLLM_ENABLE_PDL=1",
)
@pytest.mark.parametrize("variants", [(1, 0), (0, 0)])
@pytest.mark.parametrize("delay_ns", [0, 20_000, 200_000])
def test_pdl_delayed_producer(variants, delay_ns):
    """Activations written late by a PDL predecessor are the ones consumed.

    The predecessor releases its dependents immediately and writes x only
    after a delay; the dense_mlp kernels issue weight loads before
    griddepcontrol.wait but must read x after it, in eager mode and in a
    replayed CUDA graph.
    """
    mod = _pdl_delay_module()
    gate_up, down = _random_weights(6144, 3072, seed=17)
    x_old = torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16)
    x_new = torch.randn(6, 6144, device="cuda", dtype=torch.bfloat16)
    expected = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x_new, gate_up, down, *variants)
    stale = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x_old, gate_up, down, *variants)
    assert not torch.equal(expected, stale)
    x = x_old.clone()
    mod.delayed_copy(x_new, x, delay_ns)
    out = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x, gate_up, down, *variants)
    torch.cuda.synchronize()
    assert torch.equal(out, expected)

    x.copy_(x_old)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.delayed_copy(x_new, x, delay_ns)
        torch.ops.trtllm.glm_moe_dsa_dense_mlp(x, gate_up, down, *variants)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        mod.delayed_copy(x_new, x, delay_ns)
        out_g = torch.ops.trtllm.glm_moe_dsa_dense_mlp(x, gate_up, down, *variants)
    for _ in range(3):
        x.copy_(x_old)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_g, expected)
