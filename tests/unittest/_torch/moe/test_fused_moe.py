from typing import Dict, List, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.util import check_accuracy, skip_no_hopper

from tensorrt_llm._torch.model_config import ModelConfig

# isort and yapf will fight against each other here, so we disable isort
# isort: off
from tensorrt_llm._torch.moe.fused_moe import (BaseMoeRoutingMethod,
                                               RenormalizeMoeRoutingMethod,
                                               TritonFusedMoE)
from tensorrt_llm._torch.moe.fused_moe.quantization import \
    NVFP4CutlassFusedMoEMethod
# isort: on
from tensorrt_llm._torch.modules.gated_mlp import GatedMLP
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# NOTE: This file is what is left after the deprecated, permanently-skipped MoE
# tests were removed; the unified MoE test framework in
# tests/unittest/_torch/moe/test_moe_backend.py and test_moe_module.py
# covers them. Add new MoE tests there, not here.


@skip_no_hopper
@pytest.mark.parametrize("experts", [8, 128])
@pytest.mark.parametrize(
    "hidden_size, intermediate_size",
    [
        (2880, 2880),
        (2880, 1440),
        (2880, 720),
        (2880, 360),
    ],
)
@pytest.mark.parametrize("fp8_activation", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dynamic_quant", [True, False])
def test_fused_moe_triton_mxfp4(experts, hidden_size, intermediate_size,
                                fp8_activation, bias, dynamic_quant):
    if fp8_activation:
        pytest.skip("Latest Triton requires BF16 activation on Hopper")

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f'cuda:{mapping.rank}'):
        dtype = torch.bfloat16
        SEQ_LEN = 8
        HIDDEN_SIZE = hidden_size
        INTERMEDIATE_SIZE = intermediate_size
        NUM_EXPERTS = experts
        TOP_K = 4
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
        router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=dtype).cuda()

        w1_weight = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w2_weight = torch.randn((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                                dtype=dtype).cuda()
        w3_weight = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE),
                                dtype=dtype).cuda()
        w1_bias = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE),
                              dtype=dtype).cuda()
        w2_bias = torch.randn((NUM_EXPERTS, HIDDEN_SIZE), dtype=dtype).cuda()
        w3_bias = torch.randn((NUM_EXPERTS, INTERMEDIATE_SIZE),
                              dtype=dtype).cuda()

        # The fast conversion kernels require a 32-aligned quantization axis.
        # Pad and slice to cover unaligned intermediate sizes.
        from triton_kernels.numerics_details.mxfp import (downcast_to_mxfp,
                                                          upcast_from_mxfp)

        def _pad_quant_axis(tensor, k):
            padded_k = (k + 31) // 32 * 32
            if padded_k != k:
                tensor = torch.nn.functional.pad(tensor,
                                                 (0, 0, 0, padded_k - k))
            return tensor

        def fp32_to_mxfp4(tensor):
            tensor = tensor.transpose(1, 2).contiguous()
            k = tensor.shape[1]
            # MXFP4 packs two values per byte along the quantization axis.
            assert k % 2 == 0, f"quantization axis must be even, got {k}"
            tensor_fp4, tensor_scales = downcast_to_mxfp(_pad_quant_axis(
                tensor, k),
                                                         torch.uint8,
                                                         axis=1)
            # Slice the packed values back to the unpadded logical size; the
            # scale count (ceil(k / 32)) is unchanged by the padding.
            tensor_fp4 = tensor_fp4[:, :k // 2]
            tensor_fp4 = tensor_fp4.transpose(1, 2).contiguous()
            tensor_scales = tensor_scales.transpose(1, 2).contiguous()
            return tensor_fp4, tensor_scales

        def mxfp4_to_fp32(tensor, scales):
            tensor = tensor.transpose(1, 2).contiguous()
            scales = scales.transpose(1, 2).contiguous()
            k = tensor.shape[1] * 2
            # Zero-pad the packed values so the logical size matches the
            # scale blocks (scales.shape[1] * 32); zero nibbles decode to
            # 0.0 and are sliced away below.
            padded_packed = scales.shape[1] * 16
            if padded_packed != tensor.shape[1]:
                tensor = torch.nn.functional.pad(
                    tensor, (0, 0, 0, padded_packed - tensor.shape[1]))
            tensor = upcast_from_mxfp(tensor, scales, torch.float32, axis=1)
            return tensor[:, :k].transpose(1, 2).contiguous()

        w1_weight_fp4, w1_weight_scale = fp32_to_mxfp4(w1_weight)
        w2_weight_fp4, w2_weight_scale = fp32_to_mxfp4(w2_weight)
        w3_weight_fp4, w3_weight_scale = fp32_to_mxfp4(w3_weight)
        w1_weight_qdq = mxfp4_to_fp32(w1_weight_fp4, w1_weight_scale)
        w2_weight_qdq = mxfp4_to_fp32(w2_weight_fp4, w2_weight_scale)
        w3_weight_qdq = mxfp4_to_fp32(w3_weight_fp4, w3_weight_scale)

        # Since we don't have mxfp4 reference, we run the ref in bf16 after q-dq
        weights = {}
        for expert_id in range(NUM_EXPERTS):
            weights[f"{expert_id}.w1.weight"] = w1_weight_qdq[expert_id]
            weights[f"{expert_id}.w2.weight"] = w2_weight_qdq[expert_id]
            weights[f"{expert_id}.w3.weight"] = w3_weight_qdq[expert_id]
            if bias:
                weights[f"{expert_id}.w1.bias"] = w1_bias[expert_id]
                weights[f"{expert_id}.w2.bias"] = w2_bias[expert_id]
                weights[f"{expert_id}.w3.bias"] = w3_bias[expert_id]

        ref_fused_moe = RefGatedMLPFusedMoE(num_experts=NUM_EXPERTS,
                                            routing_method=routing_method,
                                            hidden_size=HIDDEN_SIZE,
                                            intermediate_size=INTERMEDIATE_SIZE,
                                            dtype=dtype,
                                            model_config=ModelConfig(),
                                            bias=bias)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)
        torch.cuda.synchronize()

        # Now we run the TritonFusedMoE with MXFP4 weights
        weights = {}

        for expert_id in range(NUM_EXPERTS):
            if dynamic_quant:
                weights[f"{expert_id}.w1.weight"] = w1_weight_qdq[expert_id]
                weights[f"{expert_id}.w2.weight"] = w2_weight_qdq[expert_id]
                weights[f"{expert_id}.w3.weight"] = w3_weight_qdq[expert_id]
            else:
                weights[f"{expert_id}.w1.weight"] = w1_weight_fp4[expert_id]
                weights[f"{expert_id}.w2.weight"] = w2_weight_fp4[expert_id]
                weights[f"{expert_id}.w3.weight"] = w3_weight_fp4[expert_id]
                weights[f"{expert_id}.w1.weight_scale"] = w1_weight_scale[
                    expert_id]
                weights[f"{expert_id}.w2.weight_scale"] = w2_weight_scale[
                    expert_id]
                weights[f"{expert_id}.w3.weight_scale"] = w3_weight_scale[
                    expert_id]
            if bias:
                weights[f"{expert_id}.w1.bias"] = w1_bias[expert_id]
                weights[f"{expert_id}.w2.bias"] = w2_bias[expert_id]
                weights[f"{expert_id}.w3.bias"] = w3_bias[expert_id]

        quant_algo = QuantAlgo.W4A8_MXFP4_FP8 if fp8_activation else QuantAlgo.W4A16_MXFP4
        quant_config = QuantConfig(quant_algo=quant_algo)
        fused_moe = TritonFusedMoE(num_experts=NUM_EXPERTS,
                                   routing_method=routing_method,
                                   hidden_size=HIDDEN_SIZE,
                                   intermediate_size=INTERMEDIATE_SIZE,
                                   dtype=dtype,
                                   reduce_results=True,
                                   bias=bias,
                                   model_config=ModelConfig(
                                       quant_config=quant_config,
                                       mapping=mapping))
        fused_moe.load_weights([weights])
        fused_moe.cuda()

        with torch.inference_mode():
            output = fused_moe.forward(x, router_logits)
        torch.cuda.synchronize()

        # Evaluate outputs

        # There can be one off mismatch in the outputs due to different kernel implementations
        # Here we check certain percent of the outputs are within the tolerance
        check_accuracy(output, ref_output, rtol=0.6, atol=0.6, percent=0.945)


class RefGatedMLPFusedMoE(nn.Module):

    def __init__(self,
                 num_experts: int,
                 routing_method: BaseMoeRoutingMethod,
                 hidden_size: int,
                 intermediate_size: int,
                 dtype: Optional[torch.dtype] = None,
                 model_config: ModelConfig = ModelConfig(),
                 use_cute_dsl_blockscaling_mm: bool = False,
                 bias=False,
                 swiglu_alpha: Optional[float] = None,
                 swiglu_beta: Optional[float] = None,
                 swiglu_limit: Optional[float] = None):
        super().__init__()
        self.num_experts = num_experts
        self.routing_method = routing_method
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.bias = bias

        self.dtype = dtype
        self.quant_config = model_config.quant_config

        def custom_swiglu(x):
            gate, value = x.chunk(2, dim=-1)
            if swiglu_limit is not None and swiglu_limit != float("inf"):
                gate = gate.clamp(max=swiglu_limit)
                value = value.clamp(min=-swiglu_limit, max=swiglu_limit)

            alpha = swiglu_alpha if swiglu_alpha is not None else 1.0
            gate_act = gate * torch.sigmoid(gate * alpha)

            beta = swiglu_beta if swiglu_beta is not None else 0.0

            return gate_act * (value + beta)

        self.experts = nn.ModuleList([
            GatedMLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                bias=bias,
                dtype=self.dtype,
                config=model_config,
                use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
                activation=custom_swiglu
                if swiglu_alpha is not None else F.silu,
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
            if self.bias:
                gate_up_proj_weights[0]['bias'] = weights[f"{expert}.w1.bias"]
                gate_up_proj_weights[1]['bias'] = weights[f"{expert}.w3.bias"]
                down_proj_weights[0]['bias'] = weights[f"{expert}.w2.bias"]

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
            elif self.quant_config and self.quant_config.quant_algo in (
                    QuantAlgo.NVFP4, QuantAlgo.W4A8_NVFP4_FP8):
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
            elif (self.quant_config and self.quant_config.quant_algo
                  == QuantAlgo.FP8_BLOCK_SCALES):
                gate_up_proj_weights[0]["weight_scale"] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]["weight_scale"] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]["weight_scale"] = weights[
                    f"{expert}.w2.weight_scale"]
            elif self.quant_config and self.quant_config.quant_algo == QuantAlgo.W4A8_MXFP4_MXFP8:
                gate_up_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w1.weight_scale"]
                gate_up_proj_weights[1]['weight_scale'] = weights[
                    f"{expert}.w3.weight_scale"]
                down_proj_weights[0]['weight_scale'] = weights[
                    f"{expert}.w2.weight_scale"]

            self.experts[expert].gate_up_proj.load_weights(gate_up_proj_weights)
            self.experts[expert].down_proj.load_weights(down_proj_weights)

    def post_load_weights(self):
        for expert in self.experts:
            expert.gate_up_proj.post_load_weights()
            expert.down_proj.post_load_weights()


# Create a mock module with required attributes for NVFP4CutlassFusedMoEMethod.get_weights_shapes test.
class MockModule:

    def __init__(self, hidden_size, intermediate_size, expand_ratio,
                 expert_size, bias):
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size
        self.intermediate_size_expand_ratio = expand_ratio
        self.expand_intermediate_size_per_partition = intermediate_size * self.intermediate_size_expand_ratio
        self.expert_size_per_partition = expert_size
        self.bias = bias
        # Constants for NVFP4.
        self.scaling_vector_size = 16  # Standard for NVFP4
        self.weight_vec_size = 16  # 16 fp4 values packed into int64
        self.block_scales_vec_size = 4  # 4 fp8 values packed into int32


def test_nvfp4_cutlass_get_weights_shapes_error_cases():
    """Test NVFP4CutlassFusedMoEMethod.get_weights_shapes for error cases."""
    method = NVFP4CutlassFusedMoEMethod()
    module = MockModule(hidden_size=13,
                        intermediate_size=16,
                        expand_ratio=1,
                        expert_size=4,
                        bias=False)
    with pytest.raises(ValueError,
                       match="hidden_size 13 must be divisible by 4"):
        method.get_weights_shapes(module, module.weight_vec_size,
                                  module.block_scales_vec_size)


@pytest.mark.parametrize(
    "hidden_size, intermediate_size, expand_ratio, expert_size, bias", [
        (512, 1024, 1, 32, True),
        (512, 1024, 2, 32, True),
        (256, 512, 1, 16, False),
        (256, 512, 2, 16, False),
        (128, 120, 1, 8, False),
        (128, 120, 2, 8, False),
        (128, 120, 1, 8, True),
        (128, 120, 2, 8, True),
    ])
def test_nvfp4_cutlass_get_weights_shapes(hidden_size, intermediate_size,
                                          expand_ratio, expert_size, bias):
    """Test NVFP4CutlassFusedMoEMethod.get_weights_shapes for alignment requirements."""
    module = MockModule(hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        expand_ratio=expand_ratio,
                        expert_size=expert_size,
                        bias=bias)
    method = NVFP4CutlassFusedMoEMethod()
    NVFP4_ROW_ALIGNMENT = method.NVFP4_ROW_ALIGNMENT

    # Get weight shapes
    (w3_w1_weight_shape, w2_weight_shape, w3_w1_bias_shape, w2_bias_shape,
     w3_w1_weight_scale_shape,
     w2_weight_scale_shape) = method.get_weights_shapes(
         module, module.weight_vec_size, module.block_scales_vec_size)

    # Calculate expected aligned sizes
    intermediate_size_expand = intermediate_size * module.intermediate_size_expand_ratio
    intermediate_size_expand_aligned = (
        (intermediate_size_expand + NVFP4_ROW_ALIGNMENT - 1) //
        NVFP4_ROW_ALIGNMENT * NVFP4_ROW_ALIGNMENT)
    hidden_size_aligned = hidden_size

    expected_w3_w1_weight_shape = (expert_size,
                                   intermediate_size_expand_aligned,
                                   hidden_size_aligned //
                                   module.weight_vec_size)
    assert w3_w1_weight_shape == expected_w3_w1_weight_shape, (
        f"w3_w1_weight_shape mismatch: got {w3_w1_weight_shape}, "
        f"expected {expected_w3_w1_weight_shape}")

    expected_w2_weight_shape = (expert_size, hidden_size_aligned,
                                intermediate_size_expand_aligned //
                                module.intermediate_size_expand_ratio //
                                module.weight_vec_size)
    assert w2_weight_shape == expected_w2_weight_shape, (
        f"w2_weight_shape mismatch: got {w2_weight_shape}, "
        f"expected {expected_w2_weight_shape}")

    expected_w3_w1_weight_scale_shape = (expert_size,
                                         intermediate_size_expand_aligned,
                                         hidden_size_aligned //
                                         module.scaling_vector_size //
                                         module.block_scales_vec_size)
    assert w3_w1_weight_scale_shape == expected_w3_w1_weight_scale_shape, (
        f"w3_w1_weight_scale_shape mismatch: got {w3_w1_weight_scale_shape}, "
        f"expected {expected_w3_w1_weight_scale_shape}")

    expected_w2_weight_scale_shape = (expert_size, hidden_size_aligned,
                                      intermediate_size_expand_aligned //
                                      module.intermediate_size_expand_ratio //
                                      module.scaling_vector_size //
                                      module.block_scales_vec_size)
    assert w2_weight_scale_shape == expected_w2_weight_scale_shape, (
        f"w2_weight_scale_shape mismatch: got {w2_weight_scale_shape}, "
        f"expected {expected_w2_weight_scale_shape}")

    # Verify bias shapes
    if bias:
        expected_w3_w1_bias_shape = (expert_size,
                                     intermediate_size_expand_aligned)
        expected_w2_bias_shape = (expert_size, hidden_size_aligned)
        assert w3_w1_bias_shape == expected_w3_w1_bias_shape, (
            f"w3_w1_bias_shape mismatch: got {w3_w1_bias_shape}, "
            f"expected {expected_w3_w1_bias_shape}")
        assert w2_bias_shape == expected_w2_bias_shape, (
            f"w2_bias_shape mismatch: got {w2_bias_shape}, "
            f"expected {expected_w2_bias_shape}")
    else:
        assert w3_w1_bias_shape is None, f"Expected None for w3_w1_bias_shape, got {w3_w1_bias_shape}"
        assert w2_bias_shape is None, f"Expected None for w2_bias_shape, got {w2_bias_shape}"

    assert intermediate_size_expand_aligned % NVFP4_ROW_ALIGNMENT == 0, (
        f"intermediate_size_expand_aligned {intermediate_size_expand_aligned} "
        f"not aligned to {NVFP4_ROW_ALIGNMENT}")
