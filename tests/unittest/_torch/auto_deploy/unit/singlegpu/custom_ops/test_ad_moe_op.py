import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import reference_bmm_moe_torch, reference_moe_torch
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.enums import (
    ActivationFunction,
    MLPStyle,
    WeightsFormat,
    WeightsFusion,
)
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
from tensorrt_llm._torch.modules.fused_moe import MoE  # noqa: F401


def setup_moe_test(dtype, num_experts):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = num_experts
    TOP_K = 2

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)  # seed=0 will fail
    x = torch.rand(SEQ_LEN, HIDDEN_SIZE, dtype=dtype).cuda() * 0.1

    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=torch.float32).cuda()
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    final_scales, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
    final_scales = final_scales / final_scales.sum(dim=-1, keepdim=True)
    final_scales = final_scales.to(x.dtype)

    w1_weight, w2_weight, w3_weight = [], [], []
    weights = {}
    fused_w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype
    ).cuda()
    fused_w2_weight = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda()

    for expert_id in range(NUM_EXPERTS):
        w1 = torch.rand(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=dtype).cuda() * 0.1
        w2 = torch.rand(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=dtype).cuda() * 0.1
        w3 = torch.rand(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=dtype).cuda() * 0.1

        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3

        w1_weight.append(w1)
        w2_weight.append(w2)
        w3_weight.append(w3)

        fused_w3_w1_stacked_weight.data[expert_id].copy_(torch.cat([w3, w1], dim=-2))
        fused_w2_weight.data[expert_id].copy_(w2)

    return (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    )


def setup_bmm_moe_test(dtype, num_experts):
    """Setup for stacked MoE with topk=1 in TRT-LLM format."""
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = num_experts
    TOP_K = 1  # Llama4 stacked pattern requires topk=1

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    x = torch.rand(SEQ_LEN, HIDDEN_SIZE, dtype=dtype).cuda() * 0.1

    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=torch.float32).cuda()
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    final_scales, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
    final_scales = final_scales / final_scales.sum(dim=-1, keepdim=True)
    final_scales = final_scales.to(x.dtype)

    # TRT-LLM format: gate_up is (2*I, H), down is (H, I)
    w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype
    ).cuda()
    w2_stacked_weight = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype
    ).cuda()

    for expert_id in range(NUM_EXPERTS):
        w31 = torch.rand(INTERMEDIATE_SIZE * 2, HIDDEN_SIZE, dtype=dtype).cuda() * 0.1
        w2 = torch.rand(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=dtype).cuda() * 0.1
        # TRT-LLM format: concat w3 and w1 along intermediate dim
        w3_w1_stacked_weight.data[expert_id].copy_(w31)  # (2*I, H)
        w2_stacked_weight.data[expert_id].copy_(w2)  # (H, I)

    return (
        x,
        selected_experts,
        final_scales,
        w3_w1_stacked_weight,
        w2_stacked_weight,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_op_run(dtype):
    num_experts = 3
    (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    ) = setup_moe_test(dtype, num_experts)

    with torch.inference_mode():
        output_torch_moe = torch.ops.auto_deploy.torch_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
            weights_format=WeightsFormat.PER_EXPERT.value,
            weights_fusion=WeightsFusion.GATE_UP_DOWN.value,
            mlp_style=MLPStyle.GATED_MLP.value,
            act_fn=ActivationFunction.SILU.value,
        )
        output_torch_fused_moe = torch.ops.auto_deploy.torch_moe_fused(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )
        output_trt_fused_moe = torch.ops.auto_deploy.trtllm_moe_fused(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )
        ref_output = reference_moe_torch(x, selected_experts, final_scales, num_experts, weights)

    torch.cuda.synchronize()
    torch.testing.assert_close(output_trt_fused_moe, output_torch_fused_moe, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_trt_fused_moe, ref_output, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_torch_fused_moe, ref_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output_torch_moe, ref_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_based_moe_op_run(dtype):
    num_experts = 3
    (
        x,
        selected_experts,
        final_scales,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    ) = setup_bmm_moe_test(dtype, num_experts)

    with torch.inference_mode():
        x = final_scales * x
        selected_experts = torch.ones_like(selected_experts)
        # Use torch_moe with stacked+fused tensor format
        output_torch_moe = torch.ops.auto_deploy.torch_moe(
            x,
            selected_experts,
            final_scales,
            [fused_w3_w1_stacked_weight],  # weights_1
            [fused_w2_weight],  # weights_2
            [],  # weights_3
            weights_format=WeightsFormat.STACKED.value,
            weights_fusion=WeightsFusion.UPGATE_DOWN.value,
            mlp_style=MLPStyle.GATED_MLP.value,
            act_fn=ActivationFunction.SILU.value,
            apply_routing_on_input=True,
        )
        output_torch_fused_moe = torch.ops.auto_deploy.torch_moe_fused(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )
        output_trt_fused_moe = torch.ops.auto_deploy.trtllm_moe_fused(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )
        ref_output = reference_bmm_moe_torch(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
            apply_routing_on_input=True,
        )

    torch.cuda.synchronize()
    torch.testing.assert_close(output_trt_fused_moe, output_torch_fused_moe, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_trt_fused_moe, ref_output, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_torch_fused_moe, ref_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output_torch_moe, ref_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fp8_moe_op_run(dtype):
    num_experts = 3
    (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    ) = setup_moe_test(dtype, num_experts)

    with torch.inference_mode():
        output_torch_moe = torch.ops.auto_deploy.torch_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
            weights_format=WeightsFormat.PER_EXPERT.value,
            weights_fusion=WeightsFusion.GATE_UP_DOWN.value,
            mlp_style=MLPStyle.GATED_MLP.value,
            act_fn=ActivationFunction.SILU.value,
        )

    w1_input_scale, w2_input_scale, w3_input_scale = [], [], []
    w1_weight_scale, w2_weight_scale, w3_weight_scale = [], [], []
    for i in range(num_experts):
        inp_scale_val = torch.tensor(1.0).float().cuda()
        wt_scale_factor = 448 if dtype == torch.bfloat16 else 432  # float16 overflow with 448
        wt_scale_val = (torch.max(torch.abs(w1_weight[i])) / wt_scale_factor).float().to("cuda")
        w1_input_scale.append(inp_scale_val)
        w2_input_scale.append(inp_scale_val)
        w3_input_scale.append(inp_scale_val)
        w1_weight_scale.append(wt_scale_val)
        w2_weight_scale.append(wt_scale_val)
        w3_weight_scale.append(wt_scale_val)
        # Cast the expert weight tensors and fused weights to FP8.
        w1_weight[i] = (w1_weight[i] / w1_weight_scale[i]).to(torch.float8_e4m3fn)
        w2_weight[i] = (w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)
        w3_weight[i] = (w3_weight[i] / w3_weight_scale[i]).to(torch.float8_e4m3fn)
        fused_w3_w1_stacked_weight[i] = (fused_w3_w1_stacked_weight[i] / w1_weight_scale[i]).to(
            torch.float8_e4m3fn
        )
        fused_w2_weight[i] = (fused_w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)

    with torch.inference_mode():
        output_torch_fp8_moe = torch.ops.auto_deploy.torch_quant_fp8_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
            w1_input_scale,
            w2_input_scale,
            w3_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
        )
        ref_output = reference_moe_torch(x, selected_experts, final_scales, num_experts, weights)

    torch.cuda.synchronize()
    rtol = 0.5 if dtype == torch.bfloat16 else 1.5
    atol = 0.8 if dtype == torch.bfloat16 else 1
    torch.testing.assert_close(output_torch_fp8_moe, output_torch_moe, rtol=rtol, atol=atol)
    torch.testing.assert_close(output_torch_fp8_moe, ref_output, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not fp4_compatible() or not trtllm_ops_available(),
    reason="Requires fp4 and trtllm support",
)
def test_fp4_moe_op_run(dtype):
    num_experts = 3
    (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        _,
        _,
    ) = setup_moe_test(dtype, num_experts)

    with torch.inference_mode():
        output_torch_moe = torch.ops.auto_deploy.torch_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
            weights_format=WeightsFormat.PER_EXPERT.value,
            weights_fusion=WeightsFusion.GATE_UP_DOWN.value,
            mlp_style=MLPStyle.GATED_MLP.value,
            act_fn=ActivationFunction.SILU.value,
        )

    # prepare FP4 scales and quantized weights
    w1_input_scale, w2_input_scale, w3_input_scale = [], [], []
    w1_weight_scale, w2_weight_scale, w3_weight_scale = [], [], []
    w1_alpha, w2_alpha, w3_alpha = [], [], []
    scaling_vector_size = 16

    for i in range(num_experts):
        inp_scale = fp4_global_scale(x)
        wt_scale_2_w1 = fp4_global_scale(w1_weight[i])
        wt_scale_2_w2 = fp4_global_scale(w2_weight[i])
        wt_scale_2_w3 = fp4_global_scale(w3_weight[i])

        # quantize weights
        w1_fp4, w1_scale = torch.ops.trtllm.fp4_quantize(
            w1_weight[i], wt_scale_2_w1, scaling_vector_size, False
        )
        w2_fp4, w2_scale = torch.ops.trtllm.fp4_quantize(
            w2_weight[i], wt_scale_2_w2, scaling_vector_size, False
        )
        w3_fp4, w3_scale = torch.ops.trtllm.fp4_quantize(
            w3_weight[i], wt_scale_2_w3, scaling_vector_size, False
        )
        w1_weight[i] = w1_fp4
        w2_weight[i] = w2_fp4
        w3_weight[i] = w3_fp4

        # record scales and alpha
        w1_input_scale.append(inp_scale)
        w2_input_scale.append(inp_scale)
        w3_input_scale.append(inp_scale)
        w1_weight_scale.append(w1_scale)
        w2_weight_scale.append(w2_scale)
        w3_weight_scale.append(w3_scale)
        w1_alpha.append(1 / (inp_scale * wt_scale_2_w1))
        w2_alpha.append(1 / (inp_scale * wt_scale_2_w2))
        w3_alpha.append(1 / (inp_scale * wt_scale_2_w3))

    # run FP4 MoE op
    with torch.inference_mode():
        output_torch_fp4_moe = torch.ops.auto_deploy.torch_quant_nvfp4_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
            w1_input_scale,
            w2_input_scale,
            w3_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
            w1_alpha,
            w2_alpha,
            w3_alpha,
        )
        ref_output = reference_moe_torch(x, selected_experts, final_scales, num_experts, weights)

    torch.cuda.synchronize()
    rtol, atol = 1.5, 1.0
    torch.testing.assert_close(output_torch_fp4_moe, output_torch_moe, rtol=rtol, atol=atol)
    torch.testing.assert_close(output_torch_fp4_moe, ref_output, rtol=rtol, atol=atol)


# ============================================================================
# Negative Tests for MoE Enum-Based API Configuration Validation
# ============================================================================


class TestEnumStringConversion:
    """Test that invalid enum string values are rejected."""

    def test_invalid_mlp_style(self):
        from tensorrt_llm._torch.auto_deploy.enums import mlp_style_from_str

        with pytest.raises(ValueError, match="Unknown mlp_style.*invalid_style"):
            mlp_style_from_str("invalid_style")

    def test_invalid_activation_function(self):
        from tensorrt_llm._torch.auto_deploy.enums import act_fn_from_str

        with pytest.raises(ValueError, match="Unknown act_fn.*invalid_act"):
            act_fn_from_str("invalid_act")

    def test_invalid_weights_format(self):
        from tensorrt_llm._torch.auto_deploy.enums import weights_format_from_str

        with pytest.raises(ValueError, match="Unknown weights_format.*invalid_format"):
            weights_format_from_str("invalid_format")

    def test_invalid_weights_fusion(self):
        from tensorrt_llm._torch.auto_deploy.enums import weights_fusion_from_str

        with pytest.raises(ValueError, match="Unknown weights_fusion.*invalid_fusion"):
            weights_fusion_from_str("invalid_fusion")


class TestTorchMoeConfigValidation:
    """Negative tests for torch_moe parameter validation."""

    @pytest.fixture
    def base_inputs(self):
        """Create base input tensors for testing."""
        batch_size, hidden_size = 4, 64
        intermediate_size = 128
        num_experts = 8
        top_k = 2

        return {
            "x": torch.randn(batch_size, hidden_size).cuda(),
            "selected_experts": torch.randint(0, num_experts, (batch_size, top_k)).cuda(),
            "routing_weights": torch.rand(batch_size, top_k).cuda(),
            "num_experts": num_experts,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        }

    def test_fusion_not_applicable_to_mlp_style(self, base_inputs):
        """Test that fusion parameter is rejected for mlp style."""

        weights_1 = [
            torch.randn(base_inputs["intermediate_size"], base_inputs["hidden_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]
        weights_2 = [
            torch.randn(base_inputs["hidden_size"], base_inputs["intermediate_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]

        with pytest.raises(ValueError, match="weights_fusion.*only applies to gated_mlp"):
            torch.ops.auto_deploy.torch_moe(
                base_inputs["x"],
                base_inputs["selected_experts"],
                base_inputs["routing_weights"],
                weights_1=weights_1,
                weights_2=weights_2,
                weights_3=[],
                weights_format=WeightsFormat.PER_EXPERT.value,
                weights_fusion=WeightsFusion.UPGATE_DOWN.value,
                mlp_style=MLPStyle.MLP.value,
                act_fn=ActivationFunction.RELU2.value,
            )

    def test_per_expert_separate_missing_weights_3(self, base_inputs):
        """Test that per_expert+separate+gated_mlp requires weights_3."""
        weights_1 = [
            torch.randn(base_inputs["intermediate_size"], base_inputs["hidden_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]
        weights_2 = [
            torch.randn(base_inputs["hidden_size"], base_inputs["intermediate_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]

        with pytest.raises(
            ValueError, match="per_expert.*w1_w2_w3_separate.*gated_mlp.*weights_3 must have"
        ):
            torch.ops.auto_deploy.torch_moe(
                base_inputs["x"],
                base_inputs["selected_experts"],
                base_inputs["routing_weights"],
                weights_1=weights_1,
                weights_2=weights_2,
                weights_3=[],
                weights_format=WeightsFormat.PER_EXPERT.value,
                weights_fusion=WeightsFusion.GATE_UP_DOWN.value,
                mlp_style=MLPStyle.GATED_MLP.value,
                act_fn=ActivationFunction.SILU.value,
            )

    def test_per_expert_fused_has_weights_3(self, base_inputs):
        """Test that per_expert+fused rejects non-empty weights_3."""
        weights_1 = [
            torch.randn(2 * base_inputs["intermediate_size"], base_inputs["hidden_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]
        weights_2 = [
            torch.randn(base_inputs["hidden_size"], base_inputs["intermediate_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]
        weights_3 = [
            torch.randn(base_inputs["intermediate_size"], base_inputs["hidden_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]

        with pytest.raises(ValueError, match="per_expert.*w3w1_w2.*weights_3 must be empty"):
            torch.ops.auto_deploy.torch_moe(
                base_inputs["x"],
                base_inputs["selected_experts"],
                base_inputs["routing_weights"],
                weights_1=weights_1,
                weights_2=weights_2,
                weights_3=weights_3,
                weights_format=WeightsFormat.PER_EXPERT.value,
                weights_fusion=WeightsFusion.UPGATE_DOWN.value,
                mlp_style=MLPStyle.GATED_MLP.value,
                act_fn=ActivationFunction.SILU.value,
            )

    def test_mismatched_expert_counts(self, base_inputs):
        """Test that mismatched weight list lengths are rejected."""
        weights_1 = [
            torch.randn(base_inputs["intermediate_size"], base_inputs["hidden_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]
        weights_2 = [
            torch.randn(base_inputs["hidden_size"], base_inputs["intermediate_size"]).cuda()
            for _ in range(base_inputs["num_experts"] + 2)
        ]
        weights_3 = [
            torch.randn(base_inputs["intermediate_size"], base_inputs["hidden_size"]).cuda()
            for _ in range(base_inputs["num_experts"])
        ]

        with pytest.raises(ValueError, match="weights_1 and weights_2 must have same length"):
            torch.ops.auto_deploy.torch_moe(
                base_inputs["x"],
                base_inputs["selected_experts"],
                base_inputs["routing_weights"],
                weights_1=weights_1,
                weights_2=weights_2,
                weights_3=weights_3,
                weights_format=WeightsFormat.PER_EXPERT.value,
                weights_fusion=WeightsFusion.GATE_UP_DOWN.value,
                mlp_style=MLPStyle.GATED_MLP.value,
                act_fn=ActivationFunction.SILU.value,
            )

    def test_stacked_expert_count_mismatch(self, base_inputs):
        """Test that stacked weights must have matching expert counts."""
        weights_1 = [
            torch.randn(
                base_inputs["num_experts"],
                2 * base_inputs["intermediate_size"],
                base_inputs["hidden_size"],
            ).cuda()
        ]
        weights_2 = [
            torch.randn(
                base_inputs["num_experts"] + 2,
                base_inputs["hidden_size"],
                base_inputs["intermediate_size"],
            ).cuda()
        ]

        with pytest.raises(ValueError, match="Expert count mismatch"):
            torch.ops.auto_deploy.torch_moe(
                base_inputs["x"],
                base_inputs["selected_experts"],
                base_inputs["routing_weights"],
                weights_1=weights_1,
                weights_2=weights_2,
                weights_3=[],
                weights_format=WeightsFormat.STACKED.value,
                weights_fusion=WeightsFusion.UPGATE_DOWN.value,
                mlp_style=MLPStyle.GATED_MLP.value,
                act_fn=ActivationFunction.SILU.value,
            )

    def test_empty_weights_1(self, base_inputs):
        """Test that empty weights_1 is rejected for per_expert."""
        with pytest.raises(ValueError, match="per_expert format.*weights_1 cannot be empty"):
            torch.ops.auto_deploy.torch_moe(
                base_inputs["x"],
                base_inputs["selected_experts"],
                base_inputs["routing_weights"],
                weights_1=[],
                weights_2=[],
                weights_3=[],
                weights_format=WeightsFormat.PER_EXPERT.value,
                weights_fusion=WeightsFusion.GATE_UP_DOWN.value,
                mlp_style=MLPStyle.MLP.value,
                act_fn=ActivationFunction.RELU2.value,
            )


class TestTRTLLMMoeEnumValidation:
    """Test TRT-LLM MoE enum-based validation."""

    def test_unsupported_gated_mlp_relu2_combination(self):
        """Test that gated_mlp + relu2 is rejected."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.trtllm_moe import (
            _validate_mlp_style_and_act_fn,
        )
        from tensorrt_llm._torch.auto_deploy.enums import ActivationFunction, MLPStyle

        with pytest.raises(ValueError, match="Unsupported combination.*gated_mlp.*relu2"):
            _validate_mlp_style_and_act_fn(MLPStyle.GATED_MLP, ActivationFunction.RELU2)

    def test_unsupported_mlp_silu_combination(self):
        """Test that mlp + silu is rejected."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.trtllm_moe import (
            _validate_mlp_style_and_act_fn,
        )
        from tensorrt_llm._torch.auto_deploy.enums import ActivationFunction, MLPStyle

        with pytest.raises(ValueError, match="Unsupported combination.*mlp.*silu"):
            _validate_mlp_style_and_act_fn(MLPStyle.MLP, ActivationFunction.SILU)
