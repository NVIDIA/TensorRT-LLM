import pytest
import torch
from utils.util import woq_assert_near_eq, woq_groupwise_gt_matmul

import tensorrt_llm
from tensorrt_llm._torch.custom_ops.torch_custom_ops import W4A16GemmRunner
from tensorrt_llm._utils import get_sm_version


@pytest.mark.parametrize(
    "m, n, k, group_size, activation_dtype, has_pre_quant, has_zero, has_bias",
    [
        (3, 1024, 64, 64, torch.bfloat16, True, False, True),
        (128, 1024, 256, 64, torch.bfloat16, True, False, True),
        (192, 2048, 384, 64, torch.bfloat16, True, False, True),
        (256, 2048, 1024, 64, torch.bfloat16, True, False, True),
        (4, 1024, 128, 128, torch.bfloat16, True, False, True),
        (64, 1024, 256, 128, torch.bfloat16, True, False, True),
        (384, 2048, 384, 128, torch.bfloat16, True, False, True),
        (512, 2048, 1024, 128, torch.bfloat16, True, False, True),
        (4, 1024, 128, 128, torch.bfloat16, True, True, True),
        (64, 1024, 256, 128, torch.bfloat16, True, True, True),
        (384, 2048, 384, 128, torch.bfloat16, True, True, True),
        (512, 2048, 1024, 128, torch.bfloat16, True, True, False),
        (3, 1024, 64, 64, torch.float16, True, False, True),
        (128, 1024, 256, 64, torch.float16, True, False, True),
        (192, 2048, 384, 64, torch.float16, True, False, True),
        (256, 2048, 1024, 64, torch.float16, True, False, True),
        (4, 1024, 128, 128, torch.float16, True, False, True),
        (64, 1024, 256, 128, torch.float16, True, False, True),
        (384, 2048, 384, 128, torch.float16, True, False, True),
        (512, 2048, 1024, 128, torch.float16, True, False, True),
        (4, 1024, 128, 128, torch.float16, True, True, True),
        (64, 1024, 256, 128, torch.float16, True, True, True),
        (384, 2048, 384, 128, torch.float16, True, True, True),
        (512, 2048, 1024, 128, torch.float16, True, True, False),
    ])
def test_matmul_activation_int4_input(m, n, k, group_size, activation_dtype,
                                      has_pre_quant, has_zero, has_bias):
    torch.manual_seed(0)
    device = "cuda"

    if get_sm_version() > W4A16GemmRunner.MAX_SUPPORTED_SM_VERSION:
        pytest.skip(f"W4A16 not supported for SM version {get_sm_version()}")

    total_groups = (k + group_size - 1) // group_size
    activation = torch.randn(m, k, dtype=activation_dtype, device=device)
    scale = torch.rand(total_groups, n, dtype=activation_dtype, device=device)
    zero = torch.randn(total_groups, n, dtype=activation_dtype,
                       device=device) if has_zero else None
    pre_quant_scale = torch.rand(1, k, dtype=activation_dtype, device=device)
    bias = torch.randn(1, n, dtype=activation_dtype,
                       device=device) if has_bias else None

    num_weights_in_32_bits = 8  # for torch.quint4x2
    unprocessed_int_weight = torch.randint(-2**31,
                                           2**31,
                                           (k, n // num_weights_in_32_bits),
                                           dtype=torch.int32,
                                           device=device)
    unprocessed_weight = unprocessed_int_weight.view(torch.int8)

    # Ref quantized weights
    unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
    ref_q_weight = unpacker(unprocessed_weight.cpu()).contiguous().cuda()

    cuda_q_weight = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm(
        unprocessed_weight.cpu(), torch.quint4x2,
        activation_dtype).cuda().contiguous()

    scale_ref = scale.repeat_interleave(group_size, dim=0)[:k, :]
    ref_th_weight = ref_q_weight.to(activation_dtype) * scale_ref

    if has_zero:
        zero_ref = zero.repeat_interleave(group_size, dim=0)[:k, :]
        ref_th_weight += zero_ref

    if has_pre_quant:
        pre_quant_scale = pre_quant_scale.repeat(m, 1)
        activation = torch.mul(activation, pre_quant_scale)

    output = torch.ops.trtllm.w4a16_gemm(
        activation.contiguous(),
        cuda_q_weight,
        scale.contiguous(),
        group_size,
        has_zero,
        bias.contiguous() if has_bias else None,
        zeros=zero)

    ref = woq_groupwise_gt_matmul(activation,
                                  ref_th_weight.to(activation_dtype), bias)

    woq_assert_near_eq(ref, output, 2)
