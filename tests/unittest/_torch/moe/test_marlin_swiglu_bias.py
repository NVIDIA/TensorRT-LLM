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
"""Marlin's SwigluBias epilogue: the activation MiniMax-M3 needs on Hopper.

Which layers Marlin admits is covered in the Marlin section of test_moe_impl.py.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import (
    convert_compressed_tensors_nvfp4_weights,
)
from tensorrt_llm._torch.moe.fused_moe import (
    MiniMaxM3MoeRoutingMethod,
    SwigluActivation,
    SwigluBiasActivation,
    create_moe,
)
from tensorrt_llm._torch.moe.fused_moe.activation import (
    materialize_activation_params,
    resolve_activation_support,
)
from tensorrt_llm._torch.moe.fused_moe.marlin import MarlinCudaNvfp4Impl
from tensorrt_llm._torch.utils import ActivationType, is_nvfp4_marlin_supported_sm
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
SWIGLU_BIAS = SwigluBiasActivation(gate_sigmoid_scale=ALPHA, linear_offset=BETA, clamp=LIMIT)
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _swiglu_bias(gate: torch.Tensor, up: torch.Tensor, limit: float | None = LIMIT) -> torch.Tensor:
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return gate * torch.sigmoid(ALPHA * gate) * (up + BETA)


def _swiglu_bias_unit(limit: float | None) -> MarlinCudaNvfp4Impl:
    module = MarlinCudaNvfp4Impl.__new__(MarlinCudaNvfp4Impl)
    module.activation_type = ActivationType.SwigluBias
    module.intermediate_size_per_partition = 4
    module.act_alpha, module.act_beta, module.act_clamp = ALPHA, BETA, limit
    return module


UP = torch.tensor([[-9.0, -1.0, 2.0, 9.0], [0.0, 0.5, -0.5, 8.0]])
GATE = torch.tensor([[-9.0, 1.0, 3.0, 9.0], [0.5, -0.5, -8.0, 0.0]])


@pytest.mark.cpu_only
def test_swiglu_bias_reads_the_cutlass_fc1_layout() -> None:
    """FC1 is packed [w3; w1], so the gate is the second half; the clamps are asymmetric."""
    out = _swiglu_bias_unit(LIMIT)._apply_activation(torch.cat([UP, GATE], dim=1))

    assert torch.equal(out, _swiglu_bias(GATE, UP))
    # A large negative gate stays unclamped and a large positive one is capped.
    assert out[0, 0] == -9.0 * torch.sigmoid(torch.tensor(ALPHA * -9.0)) * (-7.0 + BETA)
    assert out[0, 3] == 7.0 * torch.sigmoid(torch.tensor(ALPHA * 7.0)) * (7.0 + BETA)


@pytest.mark.cpu_only
def test_swiglu_bias_without_a_clamp_clamps_nothing() -> None:
    out = _swiglu_bias_unit(None)._apply_activation(torch.cat([UP, GATE], dim=1))

    assert torch.equal(out, _swiglu_bias(GATE, UP, limit=None))
    assert out[0, 3] == 9.0 * torch.sigmoid(torch.tensor(ALPHA * 9.0)) * (9.0 + BETA)


@pytest.mark.cpu_only
def test_swiglu_bias_rounds_bf16_once() -> None:
    """The epilogue runs in FP32 and casts the result back once."""
    gemm1_out = torch.cat([UP, GATE], dim=1).to(torch.bfloat16) * 0.37
    out = _swiglu_bias_unit(LIMIT)._apply_activation(gemm1_out)

    up, gate = gemm1_out.float().chunk(2, dim=1)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, _swiglu_bias(gate, up).to(torch.bfloat16))


@pytest.mark.cpu_only
def test_a_unit_built_directly_refuses_a_clamp_it_would_not_apply() -> None:
    """Plain SwiGLU runs through ``F.silu``, so its clamp must fail at construction."""
    activation = SwigluActivation(clamp=LIMIT)
    module = MarlinCudaNvfp4Impl.__new__(MarlinCudaNvfp4Impl)
    module.activation = activation
    with pytest.raises(ValueError, match="MarlinCudaNvfp4Impl kernels take no activation clamp"):
        materialize_activation_params(
            activation,
            resolve_activation_support(module),
            num_local_experts=2,
            owner="MarlinCudaNvfp4Impl",
        )


@pytest.mark.cpu_only
def test_swiglu_bias_keeps_its_constants_at_construction() -> None:
    module = MarlinCudaNvfp4Impl.__new__(MarlinCudaNvfp4Impl)
    module.activation = SWIGLU_BIAS
    params = materialize_activation_params(
        SWIGLU_BIAS,
        resolve_activation_support(module),
        num_local_experts=2,
        owner="MarlinCudaNvfp4Impl",
    )
    assert (params.alpha, params.beta, params.clamp) == (ALPHA, BETA, LIMIT)


@pytest.mark.gpu
@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_nvfp4_marlin_supported_sm()),
    reason="Marlin runs on SM89-SM99 only",
)
@pytest.mark.parametrize("tokens", [1, 24])
@pytest.mark.parametrize("gain", [1.0, 4.0], ids=["unclamped", "clamped"])
def test_marlin_swiglu_bias_matches_reference(tokens: int, gain: float) -> None:
    """Packed NVFP4 experts through the Marlin GEMMs vs an FP32 reference.

    Each expert has its own global scales, so a dropped scale or a wrong expert
    assignment shows up in the output, as does swapping the gate and up halves.
    """
    torch.manual_seed(0)
    num_experts, top_k, hidden, inter = 8, 2, 256, 256
    prefix = "language_model.model.layers.3.block_sparse_moe.experts"
    generator = torch.Generator().manual_seed(1)
    checkpoint = {}
    for e in range(num_experts):
        for proj, rows, cols, global_scale in (
            ("w1", inter, hidden, 50.0 + 7.0 * e),
            ("w3", inter, hidden, 50.0 + 7.0 * e),
            ("w2", hidden, inter, 60.0 + 5.0 * e),
        ):
            name = f"{prefix}.{e}.{proj}"
            checkpoint[f"{name}.weight_packed"] = torch.randint(
                0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator
            )
            checkpoint[f"{name}.weight_scale"] = (
                torch.rand(rows, cols // 16, generator=generator) * 2 + 0.25
            ).to(torch.float8_e4m3fn)
            checkpoint[f"{name}.weight_global_scale"] = torch.tensor(global_scale)
            checkpoint[f"{name}.input_global_scale"] = torch.tensor(1.0)

    def decode(expert: int, proj: str) -> torch.Tensor:
        name = f"{prefix}.{expert}.{proj}"
        packed = checkpoint[f"{name}.weight_packed"].cuda()
        table = torch.tensor(E2M1, device="cuda")
        low, high = (packed & 15).long(), (packed >> 4).long()
        values = torch.stack((table[low], table[high]), dim=-1).flatten(-2)
        scale = checkpoint[f"{name}.weight_scale"].cuda().float().repeat_interleave(16, dim=1)
        return values * scale / checkpoint[f"{name}.weight_global_scale"].cuda()

    experts_quant = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    model_config = ModelConfig(
        pretrained_config=SimpleNamespace(
            hidden_size=hidden,
            intermediate_size=inter,
            num_local_experts=num_experts,
            num_experts_per_tok=top_k,
            num_hidden_layers=4,
            num_attention_heads=8,
            torch_dtype=torch.bfloat16,
        ),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict={"model.layers.3.block_sparse_moe.experts": experts_quant},
        moe_backend="MARLIN",
        max_num_tokens=256,
    )
    score_correction_bias = torch.randn(num_experts, device="cuda") * 0.1
    moe = create_moe(
        routing_method=MiniMaxM3MoeRoutingMethod(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: score_correction_bias,
            routed_scaling_factor=2.0,
        ),
        num_experts=num_experts,
        model_config=model_config,
        layer_idx=3,
        override_quant_config=experts_quant,
        activation=SWIGLU_BIAS,
        allow_backend_degradation=False,
    )
    assert isinstance(moe.backend, MarlinCudaNvfp4Impl)
    translated = convert_compressed_tensors_nvfp4_weights(dict(checkpoint))
    with torch.device("cuda"):
        moe.create_weights()
        moe.load_weights([{key[len(prefix) + 1 :]: value for key, value in translated.items()}])
        moe.post_load_weights()

    x = (torch.randn(tokens, hidden, device="cuda") * gain).to(torch.bfloat16)
    logits = torch.randn(tokens, num_experts, device="cuda")
    with torch.inference_mode():
        out = moe(x, logits, all_rank_num_tokens=[tokens], use_dp_padding=False)
        ids, route = moe.routing_method.apply(logits)

    ids = ids.long()
    reference = torch.zeros(tokens, hidden, device="cuda")
    swapped = torch.zeros(tokens, hidden, device="cuda")
    for e in ids.unique().tolist():
        rows, slot = (ids == e).nonzero(as_tuple=True)
        gate, up = x[rows].float() @ decode(e, "w1").T, x[rows].float() @ decode(e, "w3").T
        weight = route[rows, slot, None].float()
        reference[rows] += weight * (_swiglu_bias(gate, up) @ decode(e, "w2").T)
        swapped[rows] += weight * (_swiglu_bias(up, gate) @ decode(e, "w2").T)

    out = out.float()
    error = ((out - reference).norm() / reference.norm()).item()
    swapped_error = ((out - swapped).norm() / reference.norm()).item()
    assert error < 0.01, f"relative L2 error {error:.4f} vs the FP32 reference"
    assert swapped_error > 0.1, f"gate/up swapped is only {swapped_error:.4f} away"
