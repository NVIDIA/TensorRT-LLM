import os
import sys
import unittest

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from parameterized import parameterized

from tests.unittest.trt.quantization import _utils


class TestW4A16Gemm(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.device = 'cuda'

    def _run_w4a16_gemm(self, m, n, k, group_size, activation_dtype,
                        has_pre_quant):
        total_groups = (k + group_size - 1) // group_size

        activation = torch.randn(m,
                                 k,
                                 dtype=activation_dtype,
                                 device=self.device)
        scale = torch.rand(total_groups,
                           n,
                           dtype=activation_dtype,
                           device=self.device)
        pre_quant_scale = torch.rand(1,
                                     k,
                                     dtype=activation_dtype,
                                     device=self.device)
        bias = torch.randn(1, n, dtype=activation_dtype, device=self.device)

        unprocessed_int_weight = torch.randint(-2**31,
                                               2**31, (k, n // 8),
                                               dtype=torch.int32,
                                               device=self.device)

        unprocessed_weight = unprocessed_int_weight.view(torch.int8)

        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        ref_q_weight = unpacker(unprocessed_weight.cpu()).contiguous().cuda()

        preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
        cuda_q_weight = preprocessor(unprocessed_weight.cpu(), torch.quint4x2,
                                     activation_dtype)

        scale_ref = scale.repeat_interleave(group_size, dim=0)[:k, :]
        ref_th_weight = ref_q_weight.to(activation_dtype) * scale_ref

        if has_pre_quant:
            pre_quant_scale = pre_quant_scale.repeat(m, 1)
            activation = torch.mul(activation, pre_quant_scale)

        activation = activation.contiguous()
        scale = scale.contiguous()
        bias = bias.contiguous()
        cuda_q_weight = cuda_q_weight.cuda().contiguous()

        output = torch.ops.tensorrt_llm.w4a16_gemm(activation, cuda_q_weight,
                                                   scale, bias, group_size)

        ref = _utils.woq_groupwise_gt_matmul(activation,
                                             ref_th_weight.to(activation_dtype),
                                             bias)

        _utils.woq_assert_near_eq(ref, output, 2)

    @parameterized.expand([(3, 1024, 64, 64), (128, 1024, 256, 64),
                           (192, 2048, 384, 64), (256, 2048, 1024, 64),
                           (4, 1024, 128, 128), (64, 1024, 256, 128),
                           (384, 2048, 384, 128), (512, 2048, 1024, 128)])
    def test_w4a16_gemm(self, m, n, k, group_size):
        self._run_w4a16_gemm(m, n, k, group_size, torch.float16, True)
