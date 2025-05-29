import os
import sys
import unittest

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from parameterized import parameterized

import tensorrt_llm
import tensorrt_llm.quantization.functional
from tests.unittest.trt.quantization import _utils
from tests.unittest.utils.util import (skip_neither_ada_nor_hopper_unittest,
                                       unittest_name_func)


class TestWeightOnlyGroupWiseQuantMatmul(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.device = 'cuda'

    def _run_w4a16_gemm(self,
                        m,
                        n,
                        k,
                        group_size,
                        activation_dtype,
                        quantized_weight_dtype,
                        has_pre_quant,
                        has_zero,
                        has_bias,
                        use_w4a8_awq=False):

        total_groups = (k + group_size - 1) // group_size
        scale_zero_dtype = torch.float16 if use_w4a8_awq else activation_dtype
        activation = torch.randn(m,
                                 k,
                                 dtype=activation_dtype,
                                 device=self.device)

        scale = torch.rand(total_groups,
                           n,
                           dtype=scale_zero_dtype,
                           device=self.device)
        zero = torch.randn(
            total_groups, n, dtype=scale_zero_dtype,
            device=self.device) if has_zero else None

        pre_quant_scale = torch.rand(1,
                                     k,
                                     dtype=activation_dtype,
                                     device=self.device)
        bias = torch.randn(1, n, dtype=activation_dtype,
                           device=self.device) if has_bias else None

        fp8_alpha = torch.rand(1, dtype=torch.float32,
                               device="cuda") if use_w4a8_awq else None

        num_weights_in_32_bits = 0
        if quantized_weight_dtype == torch.int8:
            num_weights_in_32_bits = 4
        elif quantized_weight_dtype == torch.quint4x2:
            num_weights_in_32_bits = 8
        else:
            assert False, "Unsupported weight dtype."

        unprocessed_int_weight = torch.randint(-2**31,
                                               2**31,
                                               (k, n // num_weights_in_32_bits),
                                               dtype=torch.int32,
                                               device=self.device)

        unprocessed_weight = unprocessed_int_weight.view(torch.int8)

        if use_w4a8_awq:
            activation_type = torch.float8_e4m3fn
        else:
            activation_type = torch.float16

        if quantized_weight_dtype == torch.int8:
            ref_q_weight = unprocessed_weight
        elif quantized_weight_dtype == torch.quint4x2:
            # Weights must be a CPU Tensor
            unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
            ref_q_weight = unpacker(
                unprocessed_weight.cpu()).contiguous().cuda()

        cuda_q_weight = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm(
            unprocessed_weight.cpu(), quantized_weight_dtype, activation_type)

        scale_ref = scale.repeat_interleave(group_size, dim=0)[:k, :]
        ref_th_weight = ref_q_weight.to(activation_dtype) * scale_ref

        if has_zero:
            zero_ref = zero.repeat_interleave(group_size, dim=0)[:k, :]
            ref_th_weight += zero_ref

        if has_pre_quant:
            pre_quant_scale = pre_quant_scale.repeat(m, 1)
            activation = torch.mul(activation, pre_quant_scale)

        scale = scale.contiguous()
        bias = bias.contiguous() if has_bias else None
        cuda_q_weight = cuda_q_weight.cuda().contiguous()

        output = torch.ops.trtllm.w4a16_gemm(
            input=activation.to(activation_type).contiguous()
            if use_w4a8_awq else activation.contiguous(),  # todo check
            weight=cuda_q_weight,
            scales=scale,
            group_size=group_size,
            has_zero_point=has_zero,
            output_dtype=
            activation_dtype,  # NOTE: output_dtype nneds to match activation dtype for W4A16.
            #in W4A8 output dtype is float16/bfloat16 where activation dtype is float8_e4m3fn
            alpha=fp8_alpha.item() if use_w4a8_awq else None,
            bias=bias,
            zeros=zero)

        if use_w4a8_awq:
            activation *= fp8_alpha

        ref = _utils.woq_groupwise_gt_matmul(activation,
                                             ref_th_weight.to(activation_dtype),
                                             bias)

        _utils.woq_assert_near_eq(ref, output, 2)

    @parameterized.expand([(3, 1024, 64, 64, True, False, True),
                           (128, 1024, 256, 64, True, False, True),
                           (192, 2048, 384, 64, True, False, True),
                           (256, 2048, 1024, 64, True, False, True),
                           (4, 1024, 128, 128, True, False, True),
                           (64, 1024, 256, 128, True, False, True),
                           (384, 2048, 384, 128, True, False, True),
                           (512, 2048, 1024, 128, True, False, True),
                           (4, 1024, 128, 128, True, True, True),
                           (64, 1024, 256, 128, True, True, True),
                           (384, 2048, 384, 128, True, True, True),
                           (512, 2048, 1024, 128, True, True, False)])
    def test_matmul_fp16_int4_input(self, m, n, k, group_size, has_pre_quant,
                                    has_zero, has_bias):
        self._run_w4a16_gemm(m,
                             n,
                             k,
                             group_size,
                             torch.float16,
                             torch.quint4x2,
                             has_pre_quant=has_pre_quant,
                             has_zero=has_zero,
                             has_bias=has_bias,
                             use_w4a8_awq=False)

    @parameterized.expand([(3, 1024, 64, 64, True, False, True),
                           (128, 1024, 256, 64, True, False, True),
                           (192, 2048, 384, 64, True, False, True),
                           (256, 2048, 1024, 64, True, False, True),
                           (4, 1024, 128, 128, True, False, True),
                           (64, 1024, 256, 128, True, False, True),
                           (384, 2048, 384, 128, True, False, True),
                           (512, 2048, 1024, 128, True, False, True),
                           (4, 1024, 128, 128, True, True, True),
                           (64, 1024, 256, 128, True, True, True),
                           (384, 2048, 384, 128, True, True, True),
                           (512, 2048, 1024, 128, True, True, False)])
    def test_matmul_bf16_int4_input(self, m, n, k, group_size, has_pre_quant,
                                    has_zero, has_bias):
        self._run_w4a16_gemm(m,
                             n,
                             k,
                             group_size,
                             torch.bfloat16,
                             torch.quint4x2,
                             has_pre_quant=has_pre_quant,
                             has_zero=has_zero,
                             has_bias=has_bias,
                             use_w4a8_awq=False)

    @parameterized.expand(
        [(1, 1024, 128, torch.float16, True, True, True, 128, True),
         (4, 1024, 512, torch.bfloat16, True, True, True, 128, True),
         (4, 1024, 512, torch.float16, True, True, True, 128, True),
         (16, 1024, 256, torch.float16, True, True, False, 128, True),
         (32, 1024, 384, torch.bfloat16, True, True, True, 128, True),
         (64, 1024, 256, torch.float16, True, True, False, 128, True),
         (128, 2048, 384, torch.float16, True, False, True, 128, True),
         (256, 2048, 1024, torch.float16, True, False, False, 128, True)],
        name_func=unittest_name_func)
    @skip_neither_ada_nor_hopper_unittest
    def test_prequant_matmul_fp8_int4_input(self, m, n, k, dtype, has_pre_quant,
                                            has_zero, has_bias, group_size,
                                            use_w4a8_awq):
        self._run_w4a16_gemm(m=m,
                             n=n,
                             k=k,
                             activation_dtype=dtype,
                             quantized_weight_dtype=torch.quint4x2,
                             has_pre_quant=has_pre_quant,
                             has_zero=has_zero,
                             has_bias=has_bias,
                             group_size=group_size,
                             use_w4a8_awq=use_w4a8_awq)


if __name__ == "__main__":
    unittest.main()
