# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest

import pytest

# isort: off
import torch
# isort: on

from parameterized import parameterized
from utils.util import (create_session, run_session,
                        skip_neither_ada_nor_hopper_unittest,
                        unittest_name_func)

import tensorrt_llm
import tensorrt_llm.quantization.functional
from tensorrt_llm import Tensor
from tensorrt_llm._utils import (get_sm_version, str_dtype_to_trt,
                                 torch_to_numpy, trt_dtype_to_str)
from tensorrt_llm.layers.moe import MoeConfig
from tensorrt_llm.quantization import QuantMode

from . import _utils


class TestMoEWeightOnlyQuantMatmul(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        tensorrt_llm.logger.set_level('error')

    def create_trt_session(
        self,
        str_dtype,
        act,
        router,
        fc1_prequant_scale,
        fc2_prequant_scale,
        fc1_weights,
        fc2_weights,
        weight_scaling_factor_1,
        weight_scaling_factor_2,
        zero_1,
        zero_2,
        alpha_1,
        alpha_2,
        top_k,
        has_pre_quant,
        has_zero,
        has_alpha,
        group_size,
    ):
        builder = tensorrt_llm.Builder()
        network = builder.create_network()
        dtype = str_dtype_to_trt(str_dtype)
        norm_mode = MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
        k = act.shape[1]
        if has_pre_quant:
            n = fc2_prequant_scale.shape[-1]
        else:
            n = weight_scaling_factor_1.shape[-1] // 2
        num_experts = weight_scaling_factor_1.shape[0]

        use_int8 = True if self.quant_mode.is_int8_weight_only() else False

        with tensorrt_llm.net_guard(network):
            trt_key = Tensor(name='input_hidden_states',
                             shape=act.shape,
                             dim_range=None,
                             dtype=dtype)

            network.plugin_config.moe_plugin = trt_dtype_to_str(dtype)

            moe_config = MoeConfig(num_experts=num_experts,
                                   top_k=top_k,
                                   normalization_mode=norm_mode)

            moe = tensorrt_llm.layers.MOE(moe_config=moe_config,
                                          hidden_size=k,
                                          ffn_hidden_size=n,
                                          hidden_act="swiglu",
                                          bias=False,
                                          dtype=dtype,
                                          quant_mode=self.quant_mode,
                                          pre_quant_scale=has_pre_quant,
                                          zero=has_zero,
                                          use_w4a8_awq=has_alpha,
                                          use_int8_weight=use_int8,
                                          group_size=group_size)
            moe.router.weight.value = torch_to_numpy(router.cpu())
            moe.fc.weight.value = torch_to_numpy(fc1_weights.cpu())
            moe.proj.weight.value = torch_to_numpy(fc2_weights.cpu())
            if group_size != -1:
                moe.fc.weights_scaling_factor.value = torch_to_numpy(
                    weight_scaling_factor_1.cpu())
                moe.proj.weights_scaling_factor.value = torch_to_numpy(
                    weight_scaling_factor_2.cpu())
            else:
                moe.fc.per_channel_scale.value = torch_to_numpy(
                    weight_scaling_factor_1.cpu())
                moe.proj.per_channel_scale.value = torch_to_numpy(
                    weight_scaling_factor_2.cpu())
            if has_pre_quant:
                moe.fc.prequant_scaling_factor.value = torch_to_numpy(
                    fc1_prequant_scale.cpu())
                moe.proj.prequant_scaling_factor.value = torch_to_numpy(
                    fc2_prequant_scale.cpu())
            if has_zero:
                moe.fc.zero.value = torch_to_numpy(zero_1.cpu())
                moe.proj.zero.value = torch_to_numpy(zero_2.cpu())
            if has_alpha:
                moe.fc.alpha.value = torch_to_numpy(alpha_1.cpu())
                moe.proj.alpha.value = torch_to_numpy(alpha_2.cpu())

            output = moe(trt_key, lora_layer_params=None)
            output.mark_output("output", dtype)
        # trt run
        session = create_session(builder,
                                 network,
                                 precision=trt_dtype_to_str(dtype),
                                 int8=use_int8,
                                 quant_mode=self.quant_mode)
        return session

    def _woq_moe_groupwise_matmul(self,
                                  m,
                                  n,
                                  k,
                                  num_experts,
                                  activation_dtype_str,
                                  quantized_weight_dtype,
                                  has_pre_quant,
                                  has_zero,
                                  has_alpha,
                                  top_k=2,
                                  group_size=128):

        activation_dtype = tensorrt_llm._utils.str_dtype_to_torch(
            activation_dtype_str)
        activation = torch.randn(m, k, dtype=activation_dtype, device="cuda")
        router = torch.randn((num_experts, k),
                             dtype=torch.float32,
                             device="cuda")

        num_weights_in_32_bits = 8

        assert n % num_weights_in_32_bits == 0, f"n must be a multiple of {num_weights_in_32_bits}"
        unprocessed_int_weight_1 = torch.randint(
            -2**31,
            2**31, (num_experts, k, n * 2 // num_weights_in_32_bits),
            dtype=torch.int32,
            device="cuda")
        unprocessed_int_weight_2 = torch.randint(
            -2**31,
            2**31, (num_experts, n, k // num_weights_in_32_bits),
            dtype=torch.int32,
            device="cuda")
        pre_quant_scale_1 = torch.randn(1,
                                        k,
                                        dtype=activation_dtype,
                                        device="cuda")
        pre_quant_scale_2 = torch.randn(1,
                                        n,
                                        dtype=activation_dtype,
                                        device="cuda")
        scale_1 = torch.randn(num_experts,
                              k // group_size,
                              n * 2,
                              dtype=activation_dtype,
                              device="cuda") * 0.01
        scale_2 = torch.randn(num_experts,
                              n // group_size,
                              k,
                              dtype=activation_dtype,
                              device="cuda") * 0.01
        zero_1 = torch.randn(num_experts,
                             k // group_size,
                             n * 2,
                             dtype=activation_dtype,
                             device="cuda") * 0.01
        zero_2 = torch.randn(num_experts,
                             n // group_size,
                             k,
                             dtype=activation_dtype,
                             device="cuda") * 0.01
        alpha_1 = torch.randn(
            num_experts, 1, dtype=torch.float32, device="cuda") * 0.1
        alpha_2 = torch.randn(
            num_experts, 1, dtype=torch.float32, device="cuda") * 0.1

        preprocessor = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm
        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8

        unprocessed_weight_1 = unprocessed_int_weight_1.view(torch.int8)
        unprocessed_weight_2 = unprocessed_int_weight_2.view(torch.int8)
        ref_q_weight_1 = unpacker(unprocessed_weight_1.cpu()).cuda()
        ref_q_weight_2 = unpacker(unprocessed_weight_2.cpu()).cuda()
        ref_weight_1 = ref_q_weight_1 * scale_1.repeat_interleave(group_size,
                                                                  dim=1)
        ref_weight_2 = ref_q_weight_2 * scale_2.repeat_interleave(group_size,
                                                                  dim=1)
        if has_zero:
            ref_weight_1 += zero_1.repeat_interleave(group_size, dim=1)
            ref_weight_2 += zero_2.repeat_interleave(group_size, dim=1)
        activation_type = torch.float8_e4m3fn if has_alpha else activation_dtype
        # Hopper w4a8 does not interleave weight
        do_weight_interleave = get_sm_version() != 90 or not has_alpha
        cuda_q_weight_1 = preprocessor(
            unprocessed_weight_1.cpu(),
            quantized_weight_dtype,
            activation_type,
            do_weight_interleave=do_weight_interleave).view(
                activation_dtype).cpu()
        cuda_q_weight_2 = preprocessor(
            unprocessed_weight_2.cpu(),
            quantized_weight_dtype,
            activation_type,
            do_weight_interleave=do_weight_interleave).view(
                activation_dtype).cpu()
        if get_sm_version() == 89 and has_alpha:
            scale_1 = scale_1.to(torch.float16).view(activation_dtype)
            scale_2 = scale_2.to(torch.float16).view(activation_dtype)
            zero_1 = zero_1.to(torch.float16).view(activation_dtype)
            zero_2 = zero_2.to(torch.float16).view(activation_dtype)

        if get_sm_version() == 90 and has_alpha:
            if has_zero:
                pytest.skip(
                    "has_zero is not supported in Hopper with WINT4AFP8.")

            def interleave_scales(scales: torch.Tensor, interleave_dim: int):
                # [num_experts, num_groups, num_cols] --> [num_experts, num_groups // interleave, num_cols * interleave]
                # Note: num_groups = num_rows // group_size
                E, G, C = scales.shape
                I = tensorrt_llm.quantization.functional.get_weight_scale_interleave_factor(
                    interleave_dim, group_size)
                assert G % I == 0, f"Group dimension ({G}) must be divisible by interleave factor ({I})."
                scales_interleaved = scales.reshape(E, G // I, I, C)
                scales_interleaved = scales_interleaved.permute(0, 1, 3, 2)
                scales_interleaved = scales_interleaved.reshape(
                    E, G // I, C * I)
                return scales_interleaved.contiguous()

            scale_1 = scale_1.to(torch.bfloat16).view(activation_dtype)
            scale_2 = scale_2.to(torch.bfloat16).view(activation_dtype)
            scale_1 = interleave_scales(scale_1, k)
            scale_2 = interleave_scales(scale_2, n)
            zero_1, zero_2 = None, None

        session = self.create_trt_session(
            activation_dtype_str, activation, router, pre_quant_scale_1,
            pre_quant_scale_2, cuda_q_weight_1, cuda_q_weight_2, scale_1,
            scale_2, zero_1, zero_2, alpha_1, alpha_2, top_k, has_pre_quant,
            has_zero, has_alpha, group_size)

        inputs = {"input_hidden_states": activation}
        outputs = run_session(session, inputs)
        out = outputs['output'].float()

        # ref
        inputs = activation.cuda().float()
        inputs_merged = inputs.view(-1, inputs.shape[-1])
        routing = torch.matmul(inputs_merged, router.T.float())
        router_probs = torch.softmax(routing, 1, dtype=inputs.dtype)
        topk = torch.topk(router_probs, top_k)
        results = torch.zeros_like(inputs_merged)
        for i, (scales, experts) in enumerate(zip(topk.values, topk.indices)):
            scales /= sum(scales)
            input = inputs_merged[i, :]
            for scale, expert in zip(scales, experts):
                input = inputs_merged[i, :]
                fc1_qd = ref_weight_1[expert].cuda().float()
                if has_pre_quant:
                    input = input * pre_quant_scale_1.squeeze()
                if has_alpha:
                    input[input > 448.0] = 448.0
                    input = input.to(torch.float8_e4m3fn).float()
                    fc1_qd = fc1_qd.to(torch.float8_e4m3fn).float()
                    fc1 = torch.matmul(input, fc1_qd) * alpha_1[expert]
                else:
                    fc1 = torch.matmul(input, fc1_qd)
                fc1, gate = fc1.chunk(2, dim=-1)
                fc1 = fc1 * torch.nn.functional.silu(gate)
                fc2_qd = ref_weight_2[expert].cuda().float()
                if has_pre_quant:
                    fc1 = fc1 * pre_quant_scale_2.squeeze()
                if has_alpha:
                    fc1[fc1 > 448.0] = 448.0
                    fc1 = fc1.to(torch.float8_e4m3fn).float()
                    fc2_qd = fc2_qd.to(torch.float8_e4m3fn).float()
                    final = torch.matmul(fc1, fc2_qd) * alpha_2[expert]
                else:
                    final = torch.matmul(fc1, fc2_qd)
                results[i] += scale * final
        ref = results.view(*inputs.shape)
        _utils.woq_assert_near_eq(ref, out, 2)

    def _woq_moe_matmul_per_channel(self,
                                    m,
                                    n,
                                    k,
                                    num_experts,
                                    activation_dtype_str,
                                    quantized_weight_dtype,
                                    top_k=2):

        activation_dtype = tensorrt_llm._utils.str_dtype_to_torch(
            activation_dtype_str)
        activation = torch.randn(m, k, dtype=activation_dtype, device="cuda")
        router = torch.randn((num_experts, k),
                             dtype=torch.float32,
                             device="cuda")

        num_weights_in_32_bits = 4

        assert n % num_weights_in_32_bits == 0, f"n must be a multiple of {num_weights_in_32_bits}"
        unprocessed_int_weight_1 = torch.randint(
            -2**31,
            2**31, (num_experts, k, n * 2 // num_weights_in_32_bits),
            dtype=torch.int32,
            device="cuda")
        unprocessed_int_weight_2 = torch.randint(
            -2**31,
            2**31, (num_experts, n, k // num_weights_in_32_bits),
            dtype=torch.int32,
            device="cuda")
        unprocessed_weight_1 = unprocessed_int_weight_1.view(torch.int8)
        unprocessed_weight_2 = unprocessed_int_weight_2.view(torch.int8)

        scale_1 = torch.randn(
            num_experts, 1, n * 2, dtype=activation_dtype, device="cuda") / k
        scale_2 = torch.randn(
            num_experts, 1, k, dtype=activation_dtype, device="cuda") / n

        ref_weight_1 = unprocessed_weight_1 * scale_1
        ref_weight_2 = unprocessed_weight_2 * scale_2
        scale_1 = scale_1.squeeze(1)
        scale_2 = scale_2.squeeze(1)

        preprocessor = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm

        cuda_q_weight_1 = preprocessor(unprocessed_weight_1.cpu(),
                                       quantized_weight_dtype,
                                       activation_dtype).cpu()
        cuda_q_weight_2 = preprocessor(unprocessed_weight_2.cpu(),
                                       quantized_weight_dtype,
                                       activation_dtype).cpu()

        session = self.create_trt_session(activation_dtype_str, activation,
                                          router, None, None, cuda_q_weight_1,
                                          cuda_q_weight_2, scale_1, scale_2,
                                          None, None, None, None, top_k, False,
                                          False, False, -1)

        inputs = {"input_hidden_states": activation}
        outputs = run_session(session, inputs)
        out = outputs['output'].float()

        # ref
        inputs = activation.cuda().float()
        inputs_merged = inputs.view(-1, inputs.shape[-1])
        routing = torch.matmul(inputs_merged, router.T.float())
        router_probs = torch.softmax(routing, 1, dtype=inputs.dtype)
        topk = torch.topk(router_probs, top_k)
        results = torch.zeros_like(inputs_merged)
        for i, (scales, experts) in enumerate(zip(topk.values, topk.indices)):
            scales /= sum(scales)
            input = inputs_merged[i, :]
            for scale, expert in zip(scales, experts):
                input = inputs_merged[i, :]
                fc1_qd = ref_weight_1[expert].cuda().float()
                fc1 = torch.matmul(input, fc1_qd)
                fc1, gate = fc1.chunk(2, dim=-1)
                fc1 = fc1 * torch.nn.functional.silu(gate)

                fc2_qd = ref_weight_2[expert].cuda().float()
                final = torch.matmul(fc1, fc2_qd)
                results[i] += scale * final
        ref = results.view(*inputs.shape)
        _utils.woq_assert_near_eq(ref, out, 1)

    @parameterized.expand([(1, 14336, 4096, 8, "float16"),
                           (1, 14336, 4096, 8, "bfloat16")],
                          name_func=unittest_name_func)
    def test_moe_w8a16(self, m, n, k, experts, dtype):
        self.quant_mode = QuantMode.use_weight_only(False, False)
        self._woq_moe_matmul_per_channel(m, n, k, experts, dtype, torch.int8)

    @parameterized.expand([(1, 14336, 4096, 8, "float16", True, True),
                           (1, 14336, 4096, 8, "float16", True, False),
                           (1, 14336, 4096, 8, "float16", False, True),
                           (1, 14336, 4096, 8, "bfloat16", True, True),
                           (1, 14336, 4096, 8, "bfloat16", True, False),
                           (1, 14336, 4096, 8, "bfloat16", False, True)],
                          name_func=unittest_name_func)
    def test_moe_w4a16_groupwise(self, m, n, k, experts, dtype, has_pre_quant,
                                 has_zero):
        self.quant_mode = QuantMode.use_weight_only(True, True)
        self._woq_moe_groupwise_matmul(m, n, k, experts, dtype, torch.quint4x2,
                                       has_pre_quant, has_zero, False)

    @parameterized.expand([(1, 14336, 4096, 8, "float16", True, False),
                           (1, 14336, 4096, 8, "float16", True, True),
                           (1, 14336, 4096, 8, "bfloat16", True, False),
                           (1, 14336, 4096, 8, "bfloat16", True, True)],
                          name_func=unittest_name_func)
    @skip_neither_ada_nor_hopper_unittest
    def test_moe_w4a8_groupwise(self, m, n, k, experts, dtype, has_pre_quant,
                                has_zero):
        self.quant_mode = QuantMode.use_weight_only(True, True)
        self._woq_moe_groupwise_matmul(m, n, k, experts, dtype, torch.quint4x2,
                                       has_pre_quant, has_zero, True)


if __name__ == '__main__':
    unittest.main()
