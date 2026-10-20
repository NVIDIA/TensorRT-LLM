# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 generation steps of 9..16 rows: the TF32 tensor-core router's NVFP4 activation as the routed experts' input
(TRTLLM_M3_ROUTER_TC_FP4_MOE), through the real dispatch (create_moe with the TRTLLM backend, MiniMaxM3MoeRoutingMethod,
SwigluBiasActivation, 128 experts at the M3 hidden size).

router_tc's payload must equal fp4_quantize(x, fc31_input_scale, 16, False, False) bit for bit, and the experts fed that
payload as an unswizzled Fp4QuantizedTensor must equal the experts fed bf16 x bit for bit, so handing the payload over
removes the experts' own quantize without changing the MoE output."""

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType
from _torch.moe.quantize_utils import get_test_quant_params
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_minimaxm3 as mm
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3Gate
from tensorrt_llm._torch.moe.fused_moe.activation import SwigluBiasActivation
from tensorrt_llm._torch.moe.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.moe.fused_moe.routing import MiniMaxM3MoeRoutingMethod
from tensorrt_llm._torch.utils import ActivationType, Fp4QuantizedTensor
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

H, INTER, E, TOP_K = 6144, 1536, 128, 4
ACT = SwigluBiasActivation(gate_sigmoid_scale=1.702, linear_offset=1.0, clamp=7.0)

pytestmark = pytest.mark.skipif(get_sm_version() != 107, reason="SM107 (Rubin) only")


def _build(tokens):
    AutoTuner.get().setup_distributed_state(Mapping())
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn((tokens, H), device="cuda", generator=g).to(torch.bfloat16)
    bias = (0.05 * torch.randn(E, device="cuda", generator=g)).float()
    routing = MiniMaxM3MoeRoutingMethod(top_k=TOP_K, num_experts=E, callable_e_score_correction_bias=lambda: bias,
                                        routed_scaling_factor=2.0)  # fmt: skip
    torch.manual_seed(0)
    qcls, qcfg, qkw = get_test_quant_params(QuantAlgo.NVFP4, x, MoeBackendType.TRTLLM)
    qu = qcls(num_experts=E, dtype=torch.bfloat16, intermediate_size=INTER, hidden_size=H, quant_config=qcfg,
              bias=False, swiglu_gptoss_style=True, swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=7.0,
              activation_type=ActivationType.Swiglu)  # fmt: skip
    qkw.pop("ref_cls", None)
    weights = qu.create_weights(**qkw)
    pc = PretrainedConfig()
    pc.torch_dtype = torch.bfloat16
    mc = ModelConfig(pretrained_config=pc, mapping=Mapping(world_size=1, rank=0, tp_size=1), moe_backend="TRTLLM",
                     quant_config=qcfg, moe_max_num_tokens=8192)  # fmt: skip
    moe = create_moe(routing_method=routing, num_experts=E, hidden_size=H, intermediate_size=INTER,
                     dtype=torch.bfloat16, model_config=mc, activation=ACT, allow_backend_degradation=False,
                     layer_idx=3)  # fmt: skip
    moe.load_weights([weights])
    moe.post_load_weights()
    moe.cuda()
    gate = MiniMaxM3Gate(H, E, TOP_K, 1.0).cuda()
    gate.weight.data.copy_((torch.randn(E, H, device="cuda", generator=g) / H**0.5).float())
    return moe, gate, x


@pytest.mark.parametrize("tokens", [9, 12, 16])
def test_router_tc_payload_feeds_experts_bitwise(tokens, monkeypatch) -> None:
    if not torch.ops.trtllm.minimax_m3_router_tc_cluster16(H):
        pytest.skip("this device cannot co-schedule 16-CTA clusters (router_tc not used)")
    monkeypatch.setattr(
        mm, "_ROUTER_TC_GEN", True
    )  # default OFF until its gates pass: the gate below must take router_tc
    with torch.device("cuda"), torch.inference_mode():
        moe, gate, x = _build(tokens)
        backend = moe.backend
        assert type(backend).__name__ == "TrtllmTrtllmGenNvfp4Impl", type(backend)
        scale = backend.fc31_input_scale
        logits, q, sf = gate.router_tc_with_fp4(x, scale)
        assert torch.equal(logits, gate(x, generation_only=True)), (
            "payload launch changed the logits"
        )
        ref_q, ref_sf = torch.ops.trtllm.fp4_quantize(x, scale, 16, False, False)
        assert torch.equal(q.view(torch.uint8).flatten(), ref_q.view(torch.uint8).flatten()), (
            "packed e2m1 differs"
        )
        assert torch.equal(sf.view(torch.uint8).flatten(), ref_sf.view(torch.uint8).flatten()), (
            "block scales differ"
        )
        stock = moe(x, logits)
        handed = moe(
            Fp4QuantizedTensor(q, sf, is_sf_swizzled=False, unquantized_hidden_states=x),
            logits,
            output_dtype=torch.bfloat16,
        )
        assert torch.equal(stock.view(torch.int16), handed.view(torch.int16)), "MoE output differs"
