from itertools import product
from typing import Dict, List, Optional

import pytest
import torch
import torch.nn as nn
from utils.util import skip_pre_blackwell, skip_pre_hopper

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (BaseMoeRoutingMethod,
                                                   DefaultMoeRoutingMethod,
                                                   FusedMoE,
                                                   RenormalizeMoeRoutingMethod)
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize(
    "dtype, experts, RoutingMethodCls",
    product([torch.float16, torch.bfloat16], [3, 8, 512],
            [DefaultMoeRoutingMethod, RenormalizeMoeRoutingMethod]))
def test_fused_moe(dtype, experts, RoutingMethodCls):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = experts
    TOP_K = 2
    routing_method = RoutingMethodCls(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda()
        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        weights[f"{expert_id}.w1.weight"] = w1_weight
        weights[f"{expert_id}.w2.weight"] = w2_weight
        weights[f"{expert_id}.w3.weight"] = w3_weight
    fused_moe = FusedMoE(num_experts=NUM_EXPERTS,
                         routing_method=routing_method,
                         hidden_size=HIDDEN_SIZE,
                         intermediate_size=INTERMEDIATE_SIZE,
                         dtype=dtype,
                         reduce_results=False,
                         model_config=ModelConfig())
    fused_moe.load_weights([weights])
    fused_moe.cuda()

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        fused_moe.forward(x, router_logits)

    ref_fused_moe = RefGatedMLPFusedMoE(num_experts=NUM_EXPERTS,
                                        routing_method=routing_method,
                                        hidden_size=HIDDEN_SIZE,
                                        intermediate_size=INTERMEDIATE_SIZE,
                                        dtype=dtype,
                                        model_config=ModelConfig())
    ref_fused_moe.load_weights([weights])
    ref_fused_moe.cuda()

    # Evaluate the outputs on a variant sequence length to cover all possible keys in Autotuner cache
    m = SEQ_LEN
    while m >= 2:
        x = torch.randn((m, HIDDEN_SIZE), dtype=dtype).cuda()
        router_logits = torch.randn((m, NUM_EXPERTS), dtype=dtype).cuda()

        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
            ref_output = ref_fused_moe.forward(x, router_logits)

        # Evaluate outputs
        torch.cuda.synchronize()
        torch.testing.assert_close(output, ref_output, rtol=0.2, atol=0.2)
        m //= 2


@skip_pre_hopper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8(dtype):
    SEQ_LEN = 4
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = 3
    TOP_K = 2
    routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    _, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    x_scale = x_scale.float().squeeze()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda()
        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()

        w1_weight_fp8, w1_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            w1_weight)
        w1_weight_fp8 = w1_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w2_weight_fp8, w2_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            w2_weight)
        w2_weight_fp8 = w2_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w3_weight_fp8, w3_weight_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
            w3_weight)
        w3_weight_fp8 = w3_weight_fp8.view(torch.float8_e4m3fn).cuda()

        w1_input_scale = x_scale.cuda()
        w2_input_scale = x_scale.cuda()
        w3_input_scale = x_scale.cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_fp8
        weights[f"{expert_id}.w2.weight"] = w2_weight_fp8
        weights[f"{expert_id}.w3.weight"] = w3_weight_fp8
        weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale.float()
        weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale.float()
        weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale.float()
        weights[f"{expert_id}.w1.input_scale"] = w1_input_scale
        weights[f"{expert_id}.w2.input_scale"] = w2_input_scale
        weights[f"{expert_id}.w3.input_scale"] = w3_input_scale

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    fused_moe = FusedMoE(num_experts=NUM_EXPERTS,
                         routing_method=routing_method,
                         hidden_size=HIDDEN_SIZE,
                         intermediate_size=INTERMEDIATE_SIZE,
                         dtype=dtype,
                         reduce_results=False,
                         model_config=ModelConfig(quant_config=quant_config))
    fused_moe.cuda()
    fused_moe.load_weights([weights])

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        fused_moe.forward(x, router_logits)

    ref_fused_moe = RefGatedMLPFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        model_config=ModelConfig(quant_config=quant_config))
    ref_fused_moe.load_weights([weights])
    ref_fused_moe.cuda()
    with torch.inference_mode():
        output = fused_moe.forward(x, router_logits)
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


@skip_pre_blackwell
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_nvfp4(dtype):
    SCALING_VECTOR_SIZE = 16

    SEQ_LEN = 4
    HIDDEN_SIZE = 128
    INTERMEDIATE_SIZE = 128
    NUM_EXPERTS = 3
    TOP_K = 2
    routing_method = DefaultMoeRoutingMethod(top_k=TOP_K)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()
    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

    weights = {}
    for expert_id in range(NUM_EXPERTS):
        w1_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w1_sf_global = (448 * 6) / w1_weight.abs().max().float()

        w2_weight = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda()
        w2_sf_global = (448 * 6) / w2_weight.abs().max().float()

        w3_weight = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w3_sf_global = (448 * 6) / w3_weight.abs().max().float()

        w3_w1_global = min(
            w1_sf_global,
            w3_sf_global)  # w3 global and w1 global must be the same

        w1_weight_nvfp4, w1_sf_block = torch.ops.trtllm.fp4_quantize(
            w1_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
        w1_sf_block_unswizzled = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave_reverse(
            w1_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

        w2_weight_nvfp4, w2_sf_block = torch.ops.trtllm.fp4_quantize(
            w2_weight, w2_sf_global, SCALING_VECTOR_SIZE, False)
        w2_sf_block_unswizzled = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave_reverse(
            w2_sf_block.cpu().view(HIDDEN_SIZE, -1))

        w3_weight_nvfp4, w3_sf_block = torch.ops.trtllm.fp4_quantize(
            w3_weight, w3_w1_global, SCALING_VECTOR_SIZE, False)
        w3_sf_block_unswizzled = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave_reverse(
            w3_sf_block.cpu().view(INTERMEDIATE_SIZE, -1))

        w1_input_scale = x_sf_global.cuda()
        w2_input_scale = x_sf_global.cuda()
        w3_input_scale = x_sf_global.cuda()

        weights[f"{expert_id}.w1.weight"] = w1_weight_nvfp4
        weights[f"{expert_id}.w2.weight"] = w2_weight_nvfp4
        weights[f"{expert_id}.w3.weight"] = w3_weight_nvfp4
        weights[f"{expert_id}.w1.weight_scale"] = w1_sf_block_unswizzled.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w2.weight_scale"] = w2_sf_block_unswizzled.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w3.weight_scale"] = w3_sf_block_unswizzled.view(
            torch.float8_e4m3fn).cuda()
        weights[f"{expert_id}.w1.input_scale"] = 1.0 / w1_input_scale
        weights[f"{expert_id}.w2.input_scale"] = 1.0 / w2_input_scale
        weights[f"{expert_id}.w3.input_scale"] = 1.0 / w3_input_scale
        weights[f"{expert_id}.w1.weight_scale_2"] = 1.0 / w3_w1_global
        weights[f"{expert_id}.w2.weight_scale_2"] = 1.0 / w2_sf_global
        weights[f"{expert_id}.w3.weight_scale_2"] = 1.0 / w3_w1_global

    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    fused_moe = FusedMoE(num_experts=NUM_EXPERTS,
                         routing_method=routing_method,
                         hidden_size=HIDDEN_SIZE,
                         intermediate_size=INTERMEDIATE_SIZE,
                         dtype=dtype,
                         reduce_results=False,
                         model_config=ModelConfig(quant_config=quant_config))
    fused_moe.load_weights([weights])
    fused_moe.cuda()

    # Evaluate the outputs on a variant sequence length to cover all possible keys in Autotuner cache
    ref_fused_moe = RefGatedMLPFusedMoE(
        num_experts=NUM_EXPERTS,
        routing_method=routing_method,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        dtype=dtype,
        model_config=ModelConfig(quant_config=quant_config))
    ref_fused_moe.load_weights([weights])
    ref_fused_moe.cuda()

    AutoTuner.get().clear_cache()
    with torch.inference_mode(), autotune():
        fused_moe.forward(x, router_logits)

    with torch.inference_mode():
        output = fused_moe.forward(x, router_logits)
        ref_output = ref_fused_moe.forward(x, router_logits)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.1)


class RefGatedMLPFusedMoE(nn.Module):

    def __init__(self,
                 num_experts: int,
                 routing_method: BaseMoeRoutingMethod,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig()):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.dtype = dtype
        self.quant_config = model_config.quant_config

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=False,
                dtype=self.dtype,
                config=model_config,
            ) for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_size
        hidden_states = hidden_states.view(-1, self.hidden_size)

        selected_experts, routing_weights = self.routing_method.apply(
            router_logits)

        final_hidden_states = torch.zeros(hidden_states.shape,
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)

        for expert_id in range(self.num_experts):
            if not torch.any(selected_experts == expert_id):
                continue
            batch_idx, nth_expert = torch.where(selected_experts == expert_id)
            expert_inputs = hidden_states[batch_idx]

            output = self.experts[expert_id](expert_inputs)
            final_hidden_states[batch_idx] += routing_weights[
                batch_idx, nth_expert, None] * output.float()

        final_hidden_states = final_hidden_states.reshape(hidden_states.shape)
        return final_hidden_states

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1
        weights = weights[0]

        for expert in range(self.num_experts):
            gate_up_proj_weights = [{}, {}]
            down_proj_weights = [{}]

            gate_up_proj_weights[0]['weight'] = weights[f"{expert}.w1.weight"]
            gate_up_proj_weights[1]['weight'] = weights[f"{expert}.w3.weight"]
            down_proj_weights[0]['weight'] = weights[f"{expert}.w2.weight"]

            if self.quant_config and self.quant_config.quant_algo == QuantAlgo.FP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
            elif self.quant_config and self.quant_config.quant_algo == QuantAlgo.NVFP4:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]
                gate_up_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w1.input_scale"]
                gate_up_proj_weights[1]['input_scale'] = weights[
                    f"{expert}.w3.input_scale"]
                down_proj_weights[0]['input_scale'] = weights[
                    f"{expert}.w2.input_scale"]
                gate_up_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w1.weight_scale_2"]
                gate_up_proj_weights[1]['weight_scale_2'] = weights[
                    f"{expert}.w3.weight_scale_2"]
                down_proj_weights[0]['weight_scale_2'] = weights[
                    f"{expert}.w2.weight_scale_2"]

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)
