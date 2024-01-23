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
import math
import os
import sys
import unittest

import numpy as np
import pytest

# isort: off
import torch
import tensorrt as trt
# isort: on
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import torch_to_numpy, trt_dtype_to_torch
from tensorrt_llm.layers.moe import MoeConfig
from tensorrt_llm.quantization import QuantMode

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion

default_actfn = 'gelu'


def make_tuple(num_experts=4,
               topk=1,
               hidden_size=8,
               num_sequences=5,
               sequence_length=4,
               actfn=default_actfn,
               bias=True,
               dtype='float32',
               weight_dtype=None,
               norm_mode=MoeConfig.ExpertScaleNormalizationMode.NONE):
    if weight_dtype is None:
        weight_dtype = dtype
    return (num_experts, topk, hidden_size, num_sequences, sequence_length,
            actfn, bias, dtype, weight_dtype, norm_mode)


def gen_uniform_weights(*args, **kwargs):
    return (torch.rand(*args, **kwargs) * 2 - 1).contiguous()


def quant_dequant(weights, quant_mode):
    if not quant_mode.is_weight_only():
        return weights
    # use the test version `_symmetric_...` to get the non-interleaved weights
    type = torch.quint4x2 if quant_mode.is_int4_weight_only() else torch.int8
    quant_weights, _, torch_weight_scales = torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(
        weights.T.cpu().contiguous(), type)

    # Unpack the int4s int int8s
    if quant_mode.is_int4_weight_only():
        upper = (quant_weights >> 4)
        lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
        quant_weights = torch.stack((lower, upper), dim=2).view(weights.T.shape)

    quant_weights = quant_weights.to(dtype=weights.dtype)
    result = torch.multiply(quant_weights,
                            torch_weight_scales.unsqueeze(0)).T.contiguous()
    return result.to(device=weights.device)


GATED_TO_ACT = {
    'swiglu': 'silu',
    'geglu': 'gelu',
}


def is_gated_activation(actfn):
    return actfn in GATED_TO_ACT


def gated2act(actfn):
    if is_gated_activation(actfn):
        return GATED_TO_ACT[actfn]
    return actfn


def doact(input, actfn):
    assert not is_gated_activation(actfn)
    if actfn == 'gelu':
        return torch.nn.functional.gelu(input)
    if actfn == 'relu':
        return torch.nn.functional.relu(input)
    if actfn == 'silu':
        return torch.nn.functional.silu(input)
    assert actfn == "identity"
    return input  # Identity


def gated_matmul(input, weights, bias, actfn):
    assert is_gated_activation(actfn)
    fc1 = torch.matmul(input, weights.T) + bias
    fc1, gate = fc1.chunk(2, dim=-1)
    return fc1 * doact(gate, gated2act(actfn))


class TestFunctional(unittest.TestCase):

    def setUp(self):
        # There is a known precision issues where the topk may select different experts when the routing probabilities are similar.
        #  This causes a completely different output for the affected tokens. So we set the seed to prevent sporadic failures
        #  This shouldn't be a problem for most practical applications as it means the experts are equally good choices
        torch.manual_seed(0x766E)

    def eye(self, shape, dtype, device='cuda'):
        """ Utility function for creating expert weights as an identity matrix for easy debugging """
        eye = torch.eye(shape[-2], m=shape[-1], dtype=dtype, device=device)
        eye = eye.repeat(*shape[:-2], 1, 1)
        return eye

    @staticmethod
    def get_params():
        params = []
        # Some default values to use for most test cases
        for experts in [1, 4, 42, 1024]:
            for topk in [1, 2, 3]:
                if topk < experts:
                    params += [
                        make_tuple(num_experts=experts,
                                   topk=topk,
                                   dtype='float16')
                    ]

        for num_tokens in [1, 42, 100]:
            for sequence_length in [1, 3, 42]:
                num_sequences = math.ceil(num_tokens / sequence_length)
                params += [
                    make_tuple(num_sequences=num_sequences,
                               sequence_length=sequence_length,
                               dtype='float16')
                ]

        # Add a test for float32
        params += [
            make_tuple(dtype='float32'),
            # Try 5 because non-power 2 use a different topk kernel
            make_tuple(num_experts=5, dtype='float32')
        ]

        # Add a test for bfloat16
        if getSMVersion() >= 80:
            params += [
                make_tuple(dtype='bfloat16'),
                # Try 5 because non-power 2 use a different topk kernel
                make_tuple(num_experts=5, dtype='bfloat16')
            ]

        # Add some cases for quantized dtype
        for dtype in ('int8', 'int4'):
            params += [
                make_tuple(dtype='float16', hidden_size=64, weight_dtype=dtype),
                make_tuple(dtype='float16',
                           hidden_size=64,
                           num_experts=5,
                           weight_dtype=dtype),
            ]
            if getSMVersion() >= 80:
                params += [
                    make_tuple(dtype='bfloat16',
                               hidden_size=64,
                               weight_dtype=dtype)
                ]

        # Test all activation functions with float16
        for actfn in ('relu', 'silu', 'gelu', 'swiglu', 'geglu', 'identity'):
            if actfn == default_actfn:
                continue  # Dont need to retest the one every other case uses
            params += [make_tuple(actfn=actfn, dtype='float16')]

        # Test gated with all data types as it has a different path
        for actfn in ('swiglu', 'geglu'):
            if actfn == default_actfn:
                continue  # Dont need to retest the one every other case uses
            params += [
                make_tuple(actfn=actfn, dtype='float32'),
                make_tuple(actfn=actfn,
                           hidden_size=64,
                           dtype='float16',
                           weight_dtype='int8'),
            ]
            if getSMVersion() >= 80:
                params += [make_tuple(actfn=actfn, dtype='bfloat16')]

        # Test different k values for gated activations
        params += [
            make_tuple(actfn='geglu', topk=2, dtype='float16'),
            make_tuple(actfn='geglu', topk=2, bias=False, dtype='float16')
        ]

        # Test no bias
        params += [
            make_tuple(bias=False, dtype='float32'),
            make_tuple(bias=False, dtype='float16'),
            make_tuple(dtype='float16',
                       hidden_size=64,
                       weight_dtype='int8',
                       bias=False),
            make_tuple(dtype='float16',
                       hidden_size=64,
                       weight_dtype='int4',
                       bias=False)
        ]

        # Test renormalization
        params += [
            make_tuple(
                topk=2,
                dtype='float32',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE),
            make_tuple(
                topk=2,
                dtype='float16',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE),
            make_tuple(
                dtype='float16',
                topk=2,
                hidden_size=64,
                weight_dtype='int8',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE),
            make_tuple(
                dtype='float16',
                topk=2,
                hidden_size=128,
                weight_dtype='int4',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE),
            # Renorm affects the final accumulate, so sanity check with no bias too
            make_tuple(
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                topk=2,
                dtype='float16',
                bias=False),
        ]
        if getSMVersion() >= 80:
            params += [
                make_tuple(dtype='bfloat16',
                           topk=2,
                           norm_mode=MoeConfig.ExpertScaleNormalizationMode.
                           RENORMALIZE)
            ]

        return params

    def custom_name_func(testcase_func, param_num, param):
        return "%s_%s" % (
            testcase_func.__name__,
            parameterized.to_safe_name("_".join(str(x) for x in param.args)),
        )

    def create_weights(self, num_experts, hidden_size, ffn_hidden_size, bias,
                       dtype, weight_dtype, is_gated):
        self.router_weights = torch.randn((num_experts, hidden_size),
                                          dtype=trt_dtype_to_torch(dtype),
                                          device="cuda")
        # Use a uniform scale for int8 so the quantization has a well-behaved dynamic range
        genfn = gen_uniform_weights if weight_dtype == trt.int8 else torch.randn

        fc1_out_size = ffn_hidden_size * 2 if is_gated else ffn_hidden_size
        self.fc1_weights = genfn((num_experts, fc1_out_size, hidden_size),
                                 dtype=trt_dtype_to_torch(dtype),
                                 device="cuda")

        self.fc2_weights = genfn((num_experts, hidden_size, ffn_hidden_size),
                                 dtype=trt_dtype_to_torch(dtype),
                                 device="cuda")

        bias_tensor_func = genfn if bias else torch.zeros
        self.fc1_bias = bias_tensor_func((num_experts, fc1_out_size),
                                         dtype=trt_dtype_to_torch(dtype),
                                         device="cuda")

        self.fc2_bias = bias_tensor_func((num_experts, hidden_size),
                                         dtype=trt_dtype_to_torch(dtype),
                                         device="cuda")

    @parameterized.expand(get_params(), name_func=custom_name_func)
    def test_mixture_of_experts(self, num_experts, top_k, hidden_size,
                                num_sequences, sequence_lengths, actfn, bias,
                                dtype_str, weight_dtype_str, norm_mode):
        """ This test compares the MOE plugin result to a simple reference implementation using torch """
        dtype = tensorrt_llm.str_dtype_to_trt(dtype_str)
        use_int4_weights = weight_dtype_str == 'int4'
        weight_dtype = trt.int8 if use_int4_weights else tensorrt_llm.str_dtype_to_trt(
            weight_dtype_str)

        quant_mode = QuantMode(0)
        if weight_dtype != dtype:
            quant_mode = QuantMode.use_weight_only(
                use_int4_weights=use_int4_weights)

        ffn_hidden_size = 4 * hidden_size
        self.create_weights(num_experts,
                            hidden_size,
                            ffn_hidden_size,
                            bias,
                            dtype,
                            weight_dtype,
                            is_gated=is_gated_activation(actfn))

        input_data = gen_uniform_weights(
            (num_sequences, sequence_lengths, hidden_size),
            dtype=trt_dtype_to_torch(dtype))

        # construct trt network
        trt_res = self.trtImpl(input_data,
                               num_experts,
                               top_k,
                               hidden_size,
                               ffn_hidden_size,
                               actfn,
                               bias,
                               dtype,
                               weight_dtype=weight_dtype,
                               quant_mode=quant_mode,
                               norm_mode=norm_mode)['output'].float()

        ref = self.referenceImpl(input_data, top_k, actfn, weight_dtype,
                                 quant_mode, norm_mode).cpu().float()

        tolerances = {
            'float32': 1e-2,
            'float16': 5e-2,
            'bfloat16': 5e-2,
            'int8': 2e-1,
            'int4': 2e-1,
        }
        # NOTE: There is a known issue where similar routing values result in selecting a different expert to the reference
        #   This shouldn't cause issues in production, but will cause large deviations in the test results
        np.testing.assert_allclose(trt_res,
                                   ref,
                                   rtol=tolerances[weight_dtype_str],
                                   atol=tolerances[weight_dtype_str])

    @staticmethod
    def get_mlp_params():
        params = []
        for actfn in ('gelu', 'geglu'):
            params += [('float32', actfn), ('float16', actfn),
                       ('bfloat16', actfn), ('int8', actfn), ('int4', actfn)]
        return params

    @parameterized.expand(get_mlp_params(), name_func=custom_name_func)
    def test_mlp_comparison(self, dtype_str, actfn):
        """ This test uses one expert and compares the result to a plain MLP """
        if getSMVersion() < 80 and dtype_str == 'bfloat16':
            pytest.skip("Skip bf16 tests on arch < sm80")

        use_int4_weights = dtype_str == 'int4'
        weight_dtype = trt.int8 if use_int4_weights else tensorrt_llm.str_dtype_to_trt(
            dtype_str)

        dtype = weight_dtype
        quant_mode = QuantMode(0)
        hidden_size = 8
        if dtype_str == 'int8' or dtype_str == 'int4':
            dtype = tensorrt_llm.str_dtype_to_trt("float16")
            hidden_size = 64
            quant_mode = QuantMode.use_weight_only(
                use_int4_weights=use_int4_weights)

        num_sequences = 5
        sequence_lengths = 4
        num_experts = 1
        top_k = 1
        bias = True
        ffn_hidden_size = 4 * hidden_size
        self.create_weights(num_experts,
                            hidden_size,
                            ffn_hidden_size,
                            bias,
                            dtype,
                            weight_dtype,
                            is_gated=is_gated_activation(actfn))

        input_data = gen_uniform_weights(
            (num_sequences, sequence_lengths, hidden_size),
            dtype=trt_dtype_to_torch(dtype))

        def MLP(network, trt_key, _):
            mlp_type = tensorrt_llm.layers.GatedMLP if is_gated_activation(
                actfn) else tensorrt_llm.layers.MLP
            mlp = mlp_type(hidden_size=hidden_size,
                           ffn_hidden_size=ffn_hidden_size,
                           hidden_act=gated2act(actfn),
                           bias=bias,
                           quant_mode=quant_mode,
                           dtype=dtype)
            # Quantize the weights manually so the results are comparable
            fc1_qd = quant_dequant(self.fc1_weights[0].cpu(), quant_mode)
            if is_gated_activation(actfn):
                # Note that the MLP uses the opposite convention to the GLU paper for naming,
                #  the gate is the matrix the activations are NOT applied to
                gate, fc1_qd = fc1_qd.chunk(2, dim=0)
                mlp.gate.weight.value = np.ascontiguousarray(
                    torch_to_numpy(gate))

            mlp.fc.weight.value = np.ascontiguousarray(torch_to_numpy(fc1_qd))
            fc2_qd = quant_dequant(self.fc2_weights[0].cpu(), quant_mode)
            mlp.proj.weight.value = np.ascontiguousarray(torch_to_numpy(fc2_qd))
            if bias:
                fc1_bias = self.fc1_bias[0].cpu()

                if is_gated_activation(actfn):
                    gate, fc1_bias = fc1_bias.chunk(2, dim=0)
                    mlp.gate.bias.value = np.ascontiguousarray(
                        torch_to_numpy(gate))

                mlp.fc.bias.value = np.ascontiguousarray(
                    torch_to_numpy(fc1_bias))
                mlp.proj.bias.value = np.ascontiguousarray(
                    torch_to_numpy(self.fc2_bias[0].cpu()))

            output = mlp(trt_key).trt_tensor
            output.name = 'mlp_output'
            network.mark_output(output)
            output.dtype = dtype

        res = self.trtImpl(input_data,
                           num_experts,
                           top_k,
                           hidden_size,
                           ffn_hidden_size,
                           actfn,
                           bias,
                           dtype,
                           weight_dtype=weight_dtype,
                           quant_mode=quant_mode,
                           custom_network=MLP)

        tolerances = {
            'float32': 1e-2,
            'float16': 1e-2
            if getSMVersion() >= 75 else 1e-1,  # Some issues for geglu on volta
            'bfloat16': 1e-1,
            'int8': 2e-1,
            'int4': 2e-1,
        }
        np.testing.assert_allclose(res['output'].float(),
                                   res['mlp_output'].float(),
                                   rtol=tolerances[dtype_str],
                                   atol=tolerances[dtype_str])

    def set_weight_layer(self, input_weights, weight, scale, quant_mode):
        if quant_mode.is_weight_only():
            torch_transpose = torch.transpose(input_weights, 1,
                                              2).contiguous().cpu()
            type = torch.quint4x2 if quant_mode.is_int4_weight_only(
            ) else torch.int8
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch_transpose, type)
            # Change the shape to what moe expects without touching the underlying format
            weight.value = np.ascontiguousarray(
                torch_to_numpy(processed_torch_weights))
            scale.value = np.ascontiguousarray(
                torch_to_numpy(torch_weight_scales))
        else:
            weight.value = np.ascontiguousarray(torch_to_numpy(input_weights))

    def trtImpl(self,
                input_data,
                num_experts,
                top_k,
                hidden_size,
                ffn_hidden_size,
                actfn,
                bias,
                dtype: trt.DataType,
                weight_dtype: trt.DataType = None,
                quant_mode=QuantMode(0),
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.NONE,
                finished=None,
                custom_network=None):
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            trt_key = Tensor(name='input_hidden_states',
                             shape=tuple(input_data.shape),
                             dtype=dtype)
            trt_finished = Tensor(name='input_finished',
                                  shape=tuple(finished.shape),
                                  dtype=tensorrt_llm.str_dtype_to_trt(
                                      'bool')) if finished is not None else None

            moe = tensorrt_llm.layers.MOE(moe_config=MoeConfig(
                num_experts=num_experts,
                top_k=top_k,
                normalization_mode=norm_mode),
                                          hidden_size=hidden_size,
                                          ffn_hidden_size=ffn_hidden_size,
                                          hidden_act=actfn,
                                          bias=bias,
                                          dtype=dtype,
                                          quant_mode=quant_mode)
            moe.router.weight.value = torch_to_numpy(self.router_weights.cpu())
            self.set_weight_layer(self.fc1_weights, moe.experts_weight_1,
                                  moe.experts_scale_1, quant_mode)
            self.set_weight_layer(self.fc2_weights, moe.experts_weight_2,
                                  moe.experts_scale_2, quant_mode)
            if bias:
                moe.experts_bias_1.value = torch_to_numpy(self.fc1_bias.cpu())
                moe.experts_bias_2.value = torch_to_numpy(self.fc2_bias.cpu())

            if custom_network:
                custom_network(network, trt_key, trt_finished)

            output = moe(trt_key, trt_finished).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = dtype

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == trt.float16),
                                bf16=(dtype == trt.bfloat16),
                                int8=(weight_dtype == trt.int8),
                                precision_constraints='obey',
                                builder_optimization_level=4))
        assert build_engine is not None
        with TrtRunner(build_engine) as runner:
            feed_dict = {
                'input_hidden_states': input_data,
            }
            if finished is not None:
                feed_dict['input_finished'] = finished
            outputs = runner.infer(feed_dict=feed_dict)
        return outputs

    def referenceImpl(self, inputs, k, actfn, weight_dtype, quant_mode,
                      norm_mode):
        # Always run the ref implementation at full precision TODO is this a good choice?
        inputs = inputs.cuda().float()
        inputs_merged = inputs.view(-1, inputs.shape[-1])
        routing = torch.matmul(inputs_merged, self.router_weights.T.float())
        assert routing.shape == (inputs_merged.shape[0],
                                 self.router_weights.shape[0])
        router_probs = torch.softmax(routing, 1, dtype=inputs.dtype)
        assert routing.shape == router_probs.shape

        topk = torch.topk(router_probs, k)
        assert topk.indices.shape == (router_probs.shape[0], k)
        results = torch.zeros_like(inputs_merged)
        for i, (scales, experts) in enumerate(zip(topk.values, topk.indices)):
            if norm_mode == MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE:
                scales /= sum(scales)
            input = inputs_merged[i, :]
            for scale, expert in zip(scales, experts):
                fc1_qd = quant_dequant(self.fc1_weights[expert], quant_mode)
                if is_gated_activation(actfn):
                    fc1 = gated_matmul(input, fc1_qd.float(),
                                       self.fc1_bias[expert].float(), actfn)
                else:
                    fc1 = torch.matmul(
                        input,
                        fc1_qd.T.float()) + self.fc1_bias[expert].float()
                    fc1 = doact(fc1, actfn)
                fc2_qd = quant_dequant(self.fc2_weights[expert], quant_mode)
                final = torch.matmul(
                    fc1, fc2_qd.T.float()) + self.fc2_bias[expert].float()
                assert final.shape == (inputs.shape[-1], )
                results[i] += scale * final
        return results.view(*inputs.shape)


if __name__ == "__main__":
    unittest.main()
