import pytest
import torch
import torch.nn.functional as F
from _torch.helpers import reference_moe_torch
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
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


def _prepare_nvfp4_moe_fused_args(x, selected_experts, final_scales, w1_weight, w2_weight,
                                   w3_weight, num_experts):
    """Quantize per-expert weights and prepare stacked args for trtllm_quant_nvfp4_moe_fused.

    Returns the fused kernel arguments and the per-expert reference op output.
    """
    from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.quant import (
        TRTLLM_NVFP4_SCALING_VECTOR_SIZE,
    )

    scaling_vector_size = TRTLLM_NVFP4_SCALING_VECTOR_SIZE

    w1_fp4_list, w2_fp4_list, w3_fp4_list = [], [], []
    w1_is, w2_is, w3_is = [], [], []
    w1_ws, w2_ws, w3_ws = [], [], []
    w1_al, w2_al, w3_al = [], [], []

    for i in range(num_experts):
        inp_scale = fp4_global_scale(x)
        wt_s2_w1 = fp4_global_scale(w1_weight[i])
        wt_s2_w2 = fp4_global_scale(w2_weight[i])
        wt_s2_w3 = fp4_global_scale(w3_weight[i])

        w1_fp4, w1_sc = torch.ops.trtllm.fp4_quantize(w1_weight[i], wt_s2_w1, scaling_vector_size, False)
        w2_fp4, w2_sc = torch.ops.trtllm.fp4_quantize(w2_weight[i], wt_s2_w2, scaling_vector_size, False)
        w3_fp4, w3_sc = torch.ops.trtllm.fp4_quantize(w3_weight[i], wt_s2_w3, scaling_vector_size, False)

        w1_fp4_list.append(w1_fp4)
        w2_fp4_list.append(w2_fp4)
        w3_fp4_list.append(w3_fp4)
        w1_is.append(inp_scale)
        w2_is.append(inp_scale)
        w3_is.append(inp_scale)
        w1_ws.append(w1_sc)
        w2_ws.append(w2_sc)
        w3_ws.append(w3_sc)
        w1_al.append(1 / (inp_scale * wt_s2_w1))
        w2_al.append(1 / (inp_scale * wt_s2_w2))
        w3_al.append(1 / (inp_scale * wt_s2_w3))

    # Get reference output from the per-expert op (known correct)
    with torch.inference_mode():
        ref_output = torch.ops.auto_deploy.torch_quant_nvfp4_moe(
            x, selected_experts, final_scales,
            w1_fp4_list, w2_fp4_list, w3_fp4_list,
            w1_is, w2_is, w3_is,
            w1_ws, w2_ws, w3_ws,
            w1_al, w2_al, w3_al,
        )

    # Prepare stacked tensors for the fused kernel (mimics _stack_nvfp4_moe_weights)
    w1_stacked = torch.stack(w1_fp4_list, dim=0)
    w2_stacked = torch.stack(w2_fp4_list, dim=0)
    w3_stacked = torch.stack(w3_fp4_list, dim=0)

    w1_is_t = torch.stack(w1_is)
    w2_is_t = torch.stack(w2_is)
    w3_is_t = torch.stack(w3_is)

    w1_ws_t = torch.stack(w1_ws).view(torch.float8_e4m3fn)
    w2_ws_t = torch.stack(w2_ws).view(torch.float8_e4m3fn)
    w3_ws_t = torch.stack(w3_ws).view(torch.float8_e4m3fn)

    w1_al_t = torch.stack(w1_al)
    w2_al_t = torch.stack(w2_al)
    w3_al_t = torch.stack(w3_al)

    # Check if w1/w3 alphas differ and adjust w3 block scales if needed
    alpha_ratio = w3_al_t / w1_al_t
    if not torch.allclose(alpha_ratio, torch.ones_like(alpha_ratio), rtol=1e-3, atol=1e-6):
        # Unswizzle → adjust → reswizzle (same as _adjust_w3_blockscales_for_alpha_diff)
        w3_bs_uint8 = w3_ws_t.view(torch.uint8)
        w3_bs_unswizzled = torch.ops.trtllm.block_scale_interleave_reverse(w3_bs_uint8)
        w3_bs_float = w3_bs_unswizzled.view(torch.float8_e4m3fn).float()
        w3_bs_float = w3_bs_float * alpha_ratio.view(-1, *([1] * (w3_bs_float.ndim - 1)))
        w3_bs_fp8 = w3_bs_float.to(torch.float8_e4m3fn)
        w3_ws_t = torch.ops.trtllm.block_scale_interleave(
            w3_bs_fp8.view(torch.uint8)
        ).view(torch.float8_e4m3fn).reshape(w3_ws_t.shape)

    # Concatenate for gated MLP: FC1 = [w3, w1]
    fc1_weights = torch.cat([w3_stacked, w1_stacked], dim=1).contiguous()
    fc1_blockscale = torch.cat([w3_ws_t, w1_ws_t], dim=1).contiguous()

    # FC1 uses a single input scale (all experts share input)
    fc1_act_scale = w1_is_t[0]

    # FC1 alpha: use w1's alpha (w3 block scales already compensated)
    fc1_alpha = w1_al_t

    # FC2
    fc2_weights = w2_stacked
    fc2_act_scale = w2_is_t
    fc2_alpha = w2_al_t
    fc2_blockscale = w2_ws_t

    return (
        fc1_weights, fc2_weights, fc1_blockscale, fc2_blockscale,
        fc1_act_scale, fc2_act_scale, fc1_alpha, fc2_alpha,
        ref_output,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    not fp4_compatible() or not trtllm_ops_available(),
    reason="Requires fp4 and trtllm support",
)
def test_fp4_fused_moe_with_different_weight_scales(dtype):
    """Test that the fused NVFP4 MoE kernel produces correct output when
    w1 and w3 have different per-expert weight global scales (weight_scale_2).

    This is a regression test for the bug where _stack_nvfp4_moe_weights
    ignored w3_alpha, causing incorrect dequantization of the w3 (up-projection)
    part of FC1 in the fused CUTLASS kernel.
    """
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

    # Deliberately scale w3 weights to create different weight_scale_2 per expert
    # This simulates checkpoints where per-expert weight scales are not synced
    for i in range(num_experts):
        w3_weight[i] = w3_weight[i] * (2.0 + i * 0.5)

    (
        fc1_weights, fc2_weights, fc1_blockscale, fc2_blockscale,
        fc1_act_scale, fc2_act_scale, fc1_alpha, fc2_alpha,
        ref_output,
    ) = _prepare_nvfp4_moe_fused_args(
        x, selected_experts, final_scales, w1_weight, w2_weight, w3_weight, num_experts
    )

    with torch.inference_mode():
        fused_output = torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused(
            x,
            selected_experts,
            final_scales,
            fc1_weights,
            fc2_weights,
            fc1_blockscale,
            fc2_blockscale,
            fc1_act_scale,
            fc2_act_scale,
            fc1_alpha,
            fc2_alpha,
        )

    torch.cuda.synchronize()
    # Tolerance is higher for fused kernel due to FP8 block scale rounding from adjustment
    rtol, atol = 2.0, 1.5
    torch.testing.assert_close(fused_output, ref_output, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    not fp4_compatible() or not trtllm_ops_available(),
    reason="Requires fp4 and trtllm support",
)
def test_fp4_fused_moe_with_same_weight_scales(dtype):
    """Test that the fused NVFP4 MoE kernel produces correct output when
    w1 and w3 have the same weight scales (common case, no adjustment needed).
    """
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

    (
        fc1_weights, fc2_weights, fc1_blockscale, fc2_blockscale,
        fc1_act_scale, fc2_act_scale, fc1_alpha, fc2_alpha,
        ref_output,
    ) = _prepare_nvfp4_moe_fused_args(
        x, selected_experts, final_scales, w1_weight, w2_weight, w3_weight, num_experts
    )

    with torch.inference_mode():
        fused_output = torch.ops.auto_deploy.trtllm_quant_nvfp4_moe_fused(
            x,
            selected_experts,
            final_scales,
            fc1_weights,
            fc2_weights,
            fc1_blockscale,
            fc2_blockscale,
            fc1_act_scale,
            fc2_act_scale,
            fc1_alpha,
            fc2_alpha,
        )

    torch.cuda.synchronize()
    rtol, atol = 1.5, 1.0
    torch.testing.assert_close(fused_output, ref_output, rtol=rtol, atol=atol)
