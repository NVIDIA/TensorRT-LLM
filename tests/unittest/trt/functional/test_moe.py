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
import unittest
from collections import OrderedDict
from itertools import product

import numpy as np
import pytest

# isort: off
import torch
import tensorrt as trt
# isort: on

from parameterized import parameterized
from utils.util import (create_session, getSMVersion, run_session,
                        unittest_name_func)

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import (torch_to_numpy, trt_dtype_to_str,
                                 trt_dtype_to_torch)
from tensorrt_llm.layers.lora import Lora, LoraParams
from tensorrt_llm.layers.moe import MoeConfig, MoeOOTB
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo, QuantMode
from tensorrt_llm.quantization.quantize import fp4_quantize

default_actfn = 'gelu'
default_hidden_size = {
    'float32': 8,
    'float16': 8,
    'bfloat16': 8,
    'int8': 64,
    'int4': 64,
    'fp8': 16,
}


def make_tuple(num_experts=4,
               topk=1,
               hidden_size=None,
               actfn=default_actfn,
               bias=True,
               dtype='float16',
               weight_dtype=None,
               norm_mode=MoeConfig.ExpertScaleNormalizationMode.NONE,
               use_plugin=True,
               device_limited_n_group=0,
               device_limited_topk_group=0,
               device_limited_routed_scaling_factor=1.0):
    if weight_dtype is None:
        weight_dtype = dtype
    if hidden_size is None:
        hidden_size = default_hidden_size[weight_dtype]
    return (num_experts, topk, hidden_size, actfn, bias, dtype, weight_dtype,
            norm_mode, use_plugin, device_limited_n_group,
            device_limited_topk_group, device_limited_routed_scaling_factor)


def config_is_allowed(config):
    # TODO: Support ootb path with getSMVersion() < 90:
    enable_ootb = getSMVersion() >= 90 and getSMVersion() < 100
    enable_fp8 = getSMVersion() >= 89

    WEIGHT_TYPE_INDEX = 6
    USE_PLUGIN_INDEX = 8
    if not enable_fp8 and config[WEIGHT_TYPE_INDEX] == 'fp8':
        return False
    if not enable_ootb and not config[USE_PLUGIN_INDEX]:
        return False
    return True


def gen_uniform_weights(*args, **kwargs):
    return (torch.rand(*args, **kwargs) * 2 - 1).contiguous().cuda()


def quant_dequant_int(weights, quant_mode):
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


def quant_fp4(weights):
    num_experts = 1 if len(weights.shape) == 2 else weights.shape[0]
    from modelopt.torch.quantization.qtensor import NVFP4QTensor
    if num_experts == 1:
        quant_weights, scale, _ = NVFP4QTensor.quantize(
            weights,
            block_size=16,
            weights_scaling_factor_2=torch.tensor([1.0],
                                                  dtype=torch.float32,
                                                  device=weights.device))
        quant_weights = quant_weights._quantized_data
    else:
        quant_weight_list = []
        scale_list = []
        for i in range(num_experts):
            quant_weights, scale, _ = NVFP4QTensor.quantize(
                weights[i],
                block_size=16,
                weights_scaling_factor_2=torch.tensor([1.0],
                                                      dtype=torch.float32,
                                                      device=weights.device))
            quant_weights = quant_weights._quantized_data
            quant_weight_list.append(quant_weights)
            scale_list.append(scale)
        quant_weights = torch.stack(quant_weight_list, dim=0)
        scale = torch.stack(scale_list, dim=0)
    # quant_weights, scale = torch.ops.tensorrt_llm.half_to_e2m1_and_ufp8sf_scale(
    #     weights.cuda(),
    #     torch.FloatTensor([1.0] * num_experts).cuda(), 16, 1)
    shape_prefix = weights.shape[:-1]
    quant_weights = quant_weights.view(torch.int64).view(*shape_prefix,
                                                         -1).cpu()
    scale = scale.view(torch.int32).view(*shape_prefix, -1).cpu()
    return quant_weights, scale


def quant_dequant(weights, quant_mode):
    if quant_mode.is_weight_only():
        return quant_dequant_int(weights, quant_mode)

    return weights


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


class TestMoE(unittest.TestCase):

    def setUp(self):
        # There is a known precision issues where the topk may select different experts when the routing probabilities are similar.
        #  This causes a completely different output for the affected tokens. So we set the seed to prevent sporadic failures
        #  This shouldn't be a problem for most practical applications as it means the experts are equally good choices
        torch.manual_seed(0x766E)
        tensorrt_llm.logger.set_level('error')

    def eye(self, shape, dtype, device='cuda'):
        """ Utility function for creating expert weights as an identity matrix for easy debugging """
        eye = torch.eye(shape[-2], m=shape[-1], dtype=dtype, device=device)
        eye = eye.repeat(*shape[:-2], 1, 1)
        return eye

    @staticmethod
    def get_params():
        params = []
        params += [
            make_tuple(num_experts=1, topk=1, dtype='float16'),
            make_tuple(num_experts=4, topk=2, dtype='float16'),
            # Non-powers of two have special handling for plugin softmax
            make_tuple(num_experts=42, topk=3, dtype='float16'),
            # Experts > 256 have special handling for plugin softmax
            make_tuple(num_experts=1024, topk=3, dtype='float16'),
        ]
        # OOTB test
        params += [
            make_tuple(num_experts=1, topk=1, dtype='float16',
                       use_plugin=False),
            make_tuple(num_experts=4, topk=2, dtype='float16',
                       use_plugin=False),
            make_tuple(num_experts=42,
                       topk=3,
                       dtype='float16',
                       use_plugin=False),
        ]

        # Hidden size
        params += [
            make_tuple(hidden_size=128, dtype='float16'),
        ]

        # Add a test for float32
        params += [
            make_tuple(dtype='float32'),
            make_tuple(dtype='float32', use_plugin=False),
        ]

        # Add a test for bfloat16
        params += [
            make_tuple(dtype='bfloat16'),
        ]

        # Add some cases for quantized dtype
        for dtype in ('int8', 'int4'):
            params += [
                make_tuple(dtype='float16', hidden_size=64, weight_dtype=dtype),
            ]
            params += [
                make_tuple(dtype='bfloat16', hidden_size=64, weight_dtype=dtype)
            ]

        # fp8 tests
        params += [
            make_tuple(weight_dtype='fp8', bias=False),
            make_tuple(dtype='bfloat16', weight_dtype='fp8', bias=False),
            make_tuple(topk=2, weight_dtype='fp8', bias=False),
            make_tuple(num_experts=5, topk=2, weight_dtype='fp8', bias=False),
        ]

        # Test all activation functions with float16
        for actfn in ('relu', 'silu', 'gelu', 'swiglu', 'geglu', 'identity'):
            if actfn == default_actfn:
                continue  # Dont need to retest the activation function every other case uses

            params += [
                make_tuple(actfn=actfn, dtype='float16'),
                make_tuple(actfn=actfn, dtype='float16', use_plugin=False)
            ]

        # Test gated with all data types as it has a different path
        for actfn in ('swiglu', 'geglu'):
            if actfn == default_actfn:
                continue  # Dont need to retest the one every other case uses
            params += [
                make_tuple(actfn=actfn, dtype='float32'),
                make_tuple(actfn=actfn, dtype='float16', weight_dtype='int8'),
                make_tuple(actfn=actfn, dtype='bfloat16'),
                make_tuple(actfn='geglu',
                           dtype='float16',
                           weight_dtype='fp8',
                           bias=False)
            ]

        # Test different k values for gated activations (regression case)
        params += [
            make_tuple(actfn='geglu', topk=2, dtype='float16'),
        ]

        # Test no bias
        params += [
            make_tuple(bias=False, dtype='float32'),
            make_tuple(bias=False, dtype='float16'),
            make_tuple(dtype='float16', weight_dtype='int8', bias=False),
            make_tuple(dtype='float16', weight_dtype='int4', bias=False),
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
                dtype='bfloat16',
                topk=2,
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE),
            make_tuple(
                weight_dtype='fp8',
                topk=2,
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                bias=False),
            # Renorm affects the final accumulate, so sanity check with no bias too
            make_tuple(
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                topk=2,
                dtype='float16',
                bias=False),
        ]

        # Test OOTB renormalization
        params += [
            make_tuple(
                topk=2,
                dtype='float32',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                use_plugin=False),
            make_tuple(
                topk=2,
                dtype='float16',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                use_plugin=False),
            make_tuple(
                topk=2,
                dtype='bfloat16',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                use_plugin=False),
        ]

        # Test device-limited routing
        params += [
            make_tuple(
                num_experts=80,
                topk=3,
                hidden_size=1280,
                actfn='swiglu',
                bias=True,
                dtype='float16',
                weight_dtype='float16',
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED,
                use_plugin=True,
                device_limited_n_group=8,
                device_limited_topk_group=3,
                device_limited_routed_scaling_factor=16.0),
            make_tuple(num_experts=80,
                       topk=3,
                       hidden_size=1280,
                       actfn='swiglu',
                       bias=True,
                       dtype='float16',
                       weight_dtype='float16',
                       norm_mode=MoeConfig.ExpertScaleNormalizationMode.
                       DEVICE_LIMITED_RENORM,
                       use_plugin=True,
                       device_limited_n_group=8,
                       device_limited_topk_group=3,
                       device_limited_routed_scaling_factor=16.0),
        ]

        # Default configuration for mixtral
        params += [
            make_tuple(
                num_experts=8,
                topk=2,
                norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                hidden_size=2048,
                dtype='bfloat16',
                actfn='swiglu')
        ]

        filtered_params = []
        for p in params:
            if config_is_allowed(p):
                filtered_params.append(p)

        return filtered_params

    def create_weights(self, num_experts, hidden_size, ffn_hidden_size, bias,
                       dtype, weight_dtype, is_gated):
        self.router_weights = torch.randn((num_experts, hidden_size),
                                          dtype=torch.float32,
                                          device="cuda")
        # Use a uniform scale for int8 so the quantization has a well-behaved dynamic range
        genfn = gen_uniform_weights if weight_dtype == trt.int8 else torch.randn

        # Rescale the weights if we are using gated so the results are in a similar range
        # This is 'about right' to keep the variance the same based on some napkin maths
        fc1_weight_rescale = 1 / math.sqrt(2) if is_gated else 1
        fc2_weight_rescale = 1
        if genfn == torch.randn:
            fc1_weight_rescale *= math.sqrt(2.0 / ffn_hidden_size)
            fc2_weight_rescale *= math.sqrt(2.0 / hidden_size)

        fc1_out_size = ffn_hidden_size * 2 if is_gated else ffn_hidden_size
        self.fc1_weights = genfn((num_experts, fc1_out_size, hidden_size),
                                 dtype=trt_dtype_to_torch(dtype),
                                 device="cuda") * fc1_weight_rescale

        self.fc2_weights = genfn((num_experts, hidden_size, ffn_hidden_size),
                                 dtype=trt_dtype_to_torch(dtype),
                                 device="cuda") * fc2_weight_rescale

        bias_tensor_func = genfn if bias else torch.zeros
        self.fc1_bias = bias_tensor_func((num_experts, fc1_out_size),
                                         dtype=trt_dtype_to_torch(dtype),
                                         device="cuda")

        self.fc2_bias = bias_tensor_func((num_experts, hidden_size),
                                         dtype=trt_dtype_to_torch(dtype),
                                         device="cuda")

        # Set later
        self.weight_scaling_factor_1 = None
        self.weight_scaling_factor_2 = None
        self.activation_scaling_factor_1 = None
        self.activation_scaling_factor_2 = None

    def create_lora_weights(self, num_experts, hidden_size, ffn_hidden_size,
                            dtype, num_reqs, lora_rank):
        genfn = torch.randn

        self.lora_rank = lora_rank

        fc1_weight_rescale_1 = math.sqrt(2.0 / lora_rank)
        fc1_weight_rescale_2 = math.sqrt(2.0 / ffn_hidden_size)
        fc2_weight_rescale_1 = math.sqrt(2.0 / lora_rank)
        fc2_weight_rescale_2 = math.sqrt(2.0 / hidden_size)

        self.lora_fc1_weights_1 = (genfn(
            (num_experts, lora_rank, hidden_size),
            dtype=trt_dtype_to_torch(dtype),
            device="cuda",
        ) * fc1_weight_rescale_1)
        self.lora_fc1_weights_2 = (genfn(
            (num_experts, ffn_hidden_size, lora_rank),
            dtype=trt_dtype_to_torch(dtype),
            device="cuda",
        ) * fc1_weight_rescale_2)

        self.lora_fc1_weights_ptrs = torch.tensor(
            (
                self.lora_fc1_weights_1.data_ptr(),
                self.lora_fc1_weights_2.data_ptr(),
                # null DoRA scales ptr
                0),
            dtype=torch.int64,
        ).repeat(num_reqs, 1)
        self.lora_fc1_ranks = torch.tensor((lora_rank, ),
                                           dtype=torch.int32).repeat(num_reqs)

        self.lora_gated_weights_1 = (genfn(
            (num_experts, lora_rank, hidden_size),
            dtype=trt_dtype_to_torch(dtype),
            device="cuda",
        ) * fc1_weight_rescale_1)
        self.lora_gated_weights_2 = (genfn(
            (num_experts, ffn_hidden_size, lora_rank),
            dtype=trt_dtype_to_torch(dtype),
            device="cuda",
        ) * fc1_weight_rescale_2)

        self.lora_gated_weights_ptrs = torch.tensor(
            (
                self.lora_gated_weights_1.data_ptr(),
                self.lora_gated_weights_2.data_ptr(),
                # null DoRA scales ptr
                0),
            dtype=torch.int64,
        ).repeat(num_reqs, 1)
        self.lora_gated_ranks = torch.tensor((lora_rank, ),
                                             dtype=torch.int32).repeat(num_reqs)

        self.lora_fc2_weights_1 = (genfn(
            (num_experts, lora_rank, ffn_hidden_size),
            dtype=trt_dtype_to_torch(dtype),
            device="cuda",
        ) * fc2_weight_rescale_1)
        self.lora_fc2_weights_2 = (genfn(
            (num_experts, hidden_size, lora_rank),
            dtype=trt_dtype_to_torch(dtype),
            device="cuda",
        ) * fc2_weight_rescale_2)

        self.lora_fc2_weights_ptrs = torch.tensor(
            (
                self.lora_fc2_weights_1.data_ptr(),
                self.lora_fc2_weights_2.data_ptr(),
                # null DoRA scales ptr
                0),
            dtype=torch.int64,
        ).repeat(num_reqs, 1)
        self.lora_fc2_ranks = torch.tensor((lora_rank, ),
                                           dtype=torch.int32).repeat(num_reqs)

    def create_lora_params(self, num_reqs):

        moe_h_to_4h_weights_pointers = Tensor(
            shape=(num_reqs, 3),
            dtype=tensorrt_llm.str_dtype_to_trt("int64"),
            name="moe_h_to_4h_weights_pointers",
        )
        moe_h_to_4h_lora_ranks = Tensor(
            shape=(num_reqs, ),
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            name="moe_h_to_4h_lora_ranks",
        )
        moe_4h_to_h_weights_pointers = Tensor(
            shape=(num_reqs, 3),
            dtype=tensorrt_llm.str_dtype_to_trt("int64"),
            name="moe_4h_to_h_weights_pointers",
        )
        moe_4h_to_h_lora_ranks = Tensor(
            shape=(num_reqs, ),
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            name="moe_4h_to_h_lora_ranks",
        )
        moe_gate_weights_pointers = Tensor(
            shape=(num_reqs, 3),
            dtype=tensorrt_llm.str_dtype_to_trt("int64"),
            name="moe_gate_weights_pointers",
        )
        moe_gate_lora_ranks = Tensor(
            shape=(num_reqs, ),
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            name="moe_gate_lora_ranks",
        )
        host_context_lengths = Tensor(
            shape=(num_reqs, ),
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            name="host_context_lengths",
        )
        host_request_types = Tensor(
            shape=(num_reqs, ),
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            name="host_request_types",
        )

        self.lora_params = LoraParams(
            lora_ranks=[{
                "moe_h_to_4h_lora_ranks": moe_h_to_4h_lora_ranks,
                "moe_4h_to_h_lora_ranks": moe_4h_to_h_lora_ranks,
                "moe_gate_lora_ranks": moe_gate_lora_ranks,
                "mlp_h_to_4h_lora_ranks": moe_h_to_4h_lora_ranks,
                "mlp_4h_to_h_lora_ranks": moe_4h_to_h_lora_ranks,
                "mlp_gate_lora_ranks": moe_gate_lora_ranks,
            }],
            lora_weights_pointers=[{
                "moe_h_to_4h_lora_weights_pointers":
                moe_h_to_4h_weights_pointers,
                "moe_4h_to_h_lora_weights_pointers":
                moe_4h_to_h_weights_pointers,
                "moe_gate_lora_weights_pointers":
                moe_gate_weights_pointers,
                "mlp_h_to_4h_lora_weights_pointers":
                moe_h_to_4h_weights_pointers,
                "mlp_4h_to_h_lora_weights_pointers":
                moe_4h_to_h_weights_pointers,
                "mlp_gate_lora_weights_pointers":
                moe_gate_weights_pointers,
            }],
            host_context_lengths=host_context_lengths,
            host_request_types=host_request_types,
            weight_index=0,
        )

    @staticmethod
    def max_abs_tensor(tensor):
        return torch.max(torch.abs(tensor.view(-1, np.prod(tensor.shape[-2:]))),
                         dim=1,
                         keepdim=True)[0].float()

    def create_fp8_scaling_factors(self, max_act1, max_act2):
        self.activation_scaling_factor_1 = torch.tensor([max_act1
                                                         ]).float() / 440.
        self.activation_scaling_factor_2 = torch.tensor([max_act2
                                                         ]).float() / 440.

        self.weight_scaling_factor_1 = TestMoE.max_abs_tensor(
            self.fc1_weights) / 440.
        self.weight_scaling_factor_2 = TestMoE.max_abs_tensor(
            self.fc2_weights) / 440.

    @parameterized.expand(get_params(), name_func=unittest_name_func)
    def test_mixture_of_experts(self, num_experts, top_k, hidden_size, actfn,
                                bias, dtype_str, weight_dtype_str, norm_mode,
                                use_plugin, device_limited_n_group,
                                device_limited_topk_group,
                                device_limited_routed_scaling_factor):
        """ This test compares the MOE result to a simple reference implementation using torch """

        # Build time is also proportional to the size of these (more plugin profiler runs) so dont make them too big
        # TODO Increasing these also cause some failures (observed on Hopper), not sure if this is a problem or not
        torch.random.manual_seed(42)

        max_num_seq = 10
        max_seq_len = 4

        dtype = tensorrt_llm.str_dtype_to_trt(dtype_str)

        use_fp8_qdq = weight_dtype_str == 'fp8'
        use_int4_weights = weight_dtype_str == 'int4'
        weight_dtype = trt.int8 if use_int4_weights else tensorrt_llm.str_dtype_to_trt(
            weight_dtype_str)

        quant_mode = QuantMode(0)
        if use_fp8_qdq:
            quant_mode = quant_mode.set_fp8_qdq()
        elif weight_dtype != dtype:
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

        sequence_sizes = [(1, 1), (max_num_seq, max_seq_len)]
        inputs = [gen_uniform_weights((num_seq, seq_len, hidden_size), dtype=trt_dtype_to_torch(dtype)) \
                  for num_seq, seq_len in sequence_sizes]
        reference_values = []

        act_1_quant = max(*[torch.max(torch.abs(v)).item() for v in inputs])
        act_2_quant = 0.0

        for i, input in enumerate(inputs):
            result, act2_quant_values = self.generate_reference(
                input, top_k, actfn, weight_dtype, quant_mode, norm_mode,
                device_limited_n_group, device_limited_topk_group,
                device_limited_routed_scaling_factor)
            reference_values.append(result)
            act_2_quant = max(act_2_quant, act2_quant_values)

        self.create_fp8_scaling_factors(act_1_quant, act_2_quant)

        # build trt engine
        session = self.create_trt_session(
            (-1, -1, hidden_size),
            num_experts,
            top_k,
            hidden_size,
            ffn_hidden_size,
            actfn,
            bias,
            dtype,
            weight_dtype=weight_dtype,
            quant_mode=quant_mode,
            norm_mode=norm_mode,
            use_plugin=use_plugin,
            max_sizes=[max_num_seq, max_seq_len, hidden_size],
            device_limited_n_group=device_limited_n_group,
            device_limited_topk_group=device_limited_topk_group,
            device_limited_routed_scaling_factor=
            device_limited_routed_scaling_factor)

        for input, ref in zip(inputs, reference_values):
            # run trt output
            inputs = {"input_hidden_states": input}
            outputs = run_session(session, inputs)

            tight_tolerances = {
                'float32': 1e-2,
                'float16': 1e-2,
                'bfloat16': 1e-2,
                'fp4': 2e-2,
                'fp8': 5e-2,
                'int8': 5e-2,
                'int4': 5e-2,
            }

            assert torch.sum(
                torch.isclose(outputs['output'].float(),
                              ref.float(),
                              atol=tight_tolerances[weight_dtype_str],
                              rtol=tight_tolerances[weight_dtype_str])).item(
                              ) >= math.floor(torch.numel(input) * 0.95)

            tolerances = {
                'float32': 1e-2,
                'float16': 5e-2,
                'bfloat16': 5e-2,
                'fp4': 5e-2,
                'fp8': 2e-1,
                'int8': 2e-1,
                'int4': 2e-1,
            }
            tolerance = tolerances[weight_dtype_str]

            # Bit of a hack to allow bigger tolerance for the Mixtral tests
            if hidden_size > 1024:
                # Set a higher tolerance because we hit a small fraction of outlier cases (<<1%)
                tolerance = 0.3

            torch.testing.assert_close(outputs['output'].float(),
                                       ref.float(),
                                       rtol=tolerance,
                                       atol=tolerance)

    @staticmethod
    def get_mlp_params():
        params = []
        for actfn in ('gelu', 'geglu'):
            params += [('float32', actfn, True), ('float16', actfn, True),
                       ('bfloat16', actfn, True), ('int8', actfn, True),
                       ('int4', actfn, True)]
            # OOTB tests
            # TODO: Support ootb path with getSMVersion() < 90, quantization:
            if getSMVersion() >= 90 and getSMVersion() < 100:
                params += [('float32', actfn, False), ('float16', actfn, False),
                           ('bfloat16', actfn, False)]
            if getSMVersion() >= 100:
                params += [('fp4', actfn, True)]
        return params

    @parameterized.expand(get_mlp_params(), name_func=unittest_name_func)
    def test_mlp_comparison(self, dtype_str, actfn, use_plugin):
        """ This test uses one expert and compares the result to a plain MLP """
        torch.random.manual_seed(42)

        use_int4_weights = dtype_str == 'int4'
        custom_map = {
            "int4": trt.int8,
            "fp4": trt.float16,
        }
        weight_dtype = custom_map[
            dtype_str] if dtype_str in custom_map else tensorrt_llm.str_dtype_to_trt(
                dtype_str)

        dtype = weight_dtype
        quant_mode = QuantMode(0)
        hidden_size = 8
        if dtype_str == 'int8' or dtype_str == 'int4':
            dtype = tensorrt_llm.str_dtype_to_trt("float16")
            hidden_size = 64
            quant_mode = QuantMode.use_weight_only(
                use_int4_weights=use_int4_weights)
        elif dtype_str == "fp4":
            quant_mode = QuantMode.NVFP4
            hidden_size = 256  # At least vector_size * 16 to make padding simple

        num_sequences = 5
        sequence_lengths = 4
        num_experts = 1  # 4 # TODO Ampere fails to build the TRT network with multiple experts
        top_k = num_experts  # All tokens to all experts to make the comparison trivial
        bias = not quant_mode.has_nvfp4()
        ffn_hidden_size = 4 * hidden_size
        self.create_weights(num_experts,
                            hidden_size,
                            ffn_hidden_size,
                            bias,
                            dtype,
                            weight_dtype,
                            is_gated=is_gated_activation(actfn))

        # Override the router to ensure all values have the same scale
        for i in range(num_experts):
            self.router_weights[i] = torch.squeeze(
                torch.eye(hidden_size, m=1, dtype=torch.float32, device="cuda"))

        input_data = gen_uniform_weights(
            (num_sequences, sequence_lengths, hidden_size),
            dtype=trt_dtype_to_torch(dtype))

        def MLP(network, trt_key):
            output = trt_key * 0.0
            for i in range(num_experts):
                mlp_type = tensorrt_llm.layers.GatedMLP if is_gated_activation(
                    actfn) else tensorrt_llm.layers.MLP
                mlp = mlp_type(hidden_size=hidden_size,
                               ffn_hidden_size=ffn_hidden_size,
                               hidden_act=gated2act(actfn),
                               bias=bias,
                               quant_mode=quant_mode,
                               dtype=dtype)

                if quant_mode.has_nvfp4():
                    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
                    mlp = fp4_quantize(mlp, quant_config)
                    mlp.quant_mode = quant_mode

                # Quantize the weights manually so the results are comparable
                fc1_qd = quant_dequant(self.fc1_weights[i].cpu(), quant_mode)

                if is_gated_activation(actfn):
                    # Note that the MLP uses the opposite convention to the GLU paper for naming,
                    #  the gate is the matrix the activations are NOT applied to
                    gate, fc1_qd = fc1_qd.chunk(2, dim=0)

                    if quant_mode.has_nvfp4():
                        gate, block_scale = quant_fp4(gate)
                        self.set_fp4_scales(mlp.gate, block_scale, 1)

                    mlp.gate.weight.value = np.ascontiguousarray(
                        torch_to_numpy(gate))

                if quant_mode.has_nvfp4():
                    fc1_qd, block_scale = quant_fp4(fc1_qd)
                    self.set_fp4_scales(mlp.fc, block_scale, 1)

                mlp.fc.weight.value = np.ascontiguousarray(
                    torch_to_numpy(fc1_qd))

                fc2_qd = quant_dequant(self.fc2_weights[i].cpu(), quant_mode)
                if quant_mode.has_nvfp4():
                    fc2_qd, block_scale = quant_fp4(fc2_qd)
                    self.set_fp4_scales(mlp.proj, block_scale, 1)

                mlp.proj.weight.value = np.ascontiguousarray(
                    torch_to_numpy(fc2_qd))
                if bias:
                    fc1_bias = self.fc1_bias[i].cpu()

                    if is_gated_activation(actfn):
                        gate, fc1_bias = fc1_bias.chunk(2, dim=0)
                        mlp.gate.bias.value = np.ascontiguousarray(
                            torch_to_numpy(gate))

                    mlp.fc.bias.value = np.ascontiguousarray(
                        torch_to_numpy(fc1_bias))
                    mlp.proj.bias.value = np.ascontiguousarray(
                        torch_to_numpy(self.fc2_bias[i].cpu()))

                output += mlp(trt_key) / num_experts
            output.mark_output('mlp_output', dtype)

        session = self.create_trt_session(
            tuple(input_data.shape),
            num_experts,
            top_k,
            hidden_size,
            ffn_hidden_size,
            actfn,
            bias,
            dtype,
            weight_dtype,
            quant_mode,
            norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
            custom_network=MLP,
            use_plugin=use_plugin)

        inputs = {"input_hidden_states": input_data}
        outputs = run_session(session, inputs)

        tight_tolerances = {
            'float32': 1e-2,
            'float16': 1e-2,
            'bfloat16': 1e-2,
            'fp4': 1.5e-2,
            'int8': 5e-2,
            'int4': 5e-2,
        }

        result = torch.sum(
            torch.isclose(outputs["output"],
                          outputs["mlp_output"],
                          atol=tight_tolerances[dtype_str],
                          rtol=tight_tolerances[dtype_str])).item()
        assert result >= math.floor(torch.numel(input_data) * 0.95)

        loose_tolerances = {
            'float32': 1e-2,
            'float16': 2e-2,
            'bfloat16': 1e-1,
            'int8': 2e-1,
            'int4': 2e-1,
            'fp4': 6e-2,
        }

        torch.testing.assert_close(
            outputs["output"],
            outputs["mlp_output"],
            rtol=loose_tolerances[dtype_str],
            atol=loose_tolerances[dtype_str],
        )

    @staticmethod
    def get_ootb_comp_params():
        params = []
        for actfn in ('gelu', 'geglu'):
            for experts, k in [(8, 2), (10, 3)]:
                dtypes = ['float16', 'bfloat16', 'fp8']
                if getSMVersion() >= 100:
                    dtypes += ['fp4']
                for dtype in dtypes:
                    params.append((dtype, experts, k, actfn))
        return params

    @parameterized.expand(get_ootb_comp_params(), name_func=unittest_name_func)
    def test_ootb_comparison(self, dtype_str, num_experts, top_k, actfn):
        """ This test uses one expert and compares the result to a plain MLP """
        if getSMVersion() != 90:
            pytest.skip("OOTB tests disabled on pre-Hopper architectures")

        torch.random.manual_seed(42)

        use_int4_weights = dtype_str == 'int4'
        custom_map = {
            "int4": trt.int8,
            "fp4": trt.float16,
        }
        weight_dtype = custom_map[
            dtype_str] if dtype_str in custom_map else tensorrt_llm.str_dtype_to_trt(
                dtype_str)

        dtype = weight_dtype
        quant_mode = QuantMode(0)
        hidden_size = 8
        if dtype_str == "fp8":
            dtype = tensorrt_llm.str_dtype_to_trt("bfloat16")
            quant_mode = QuantMode.FP8_QDQ
            hidden_size = 64
        elif dtype_str == "fp4":
            quant_mode = QuantMode.NVFP4
            hidden_size = 256  # At least vector_size * 16 to make padding simple

        num_sequences = 5
        sequence_lengths = 4
        bias = not quant_mode.has_nvfp4() and not quant_mode.has_fp8_qdq()
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

        if quant_mode.has_fp8_qdq():
            max_act = TestMoE.max_abs_tensor(input_data.view(-1, hidden_size))
            max_weight = TestMoE.max_abs_tensor(
                self.fc1_weights.view(-1, hidden_size))
            self.create_fp8_scaling_factors(max_act, max_act * max_weight)

        session_moe = self.create_trt_session(
            tuple(input_data.shape),
            num_experts,
            top_k,
            hidden_size,
            ffn_hidden_size,
            actfn,
            bias,
            dtype,
            weight_dtype,
            quant_mode,
            norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
            use_plugin=True)
        session_ootb = self.create_trt_session(
            tuple(input_data.shape),
            num_experts,
            top_k,
            hidden_size,
            ffn_hidden_size,
            actfn,
            bias,
            dtype,
            weight_dtype,
            quant_mode,
            norm_mode=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
            use_plugin=False)

        inputs = {"input_hidden_states": input_data}
        outputs_moe = run_session(session_moe, inputs)
        outputs_ootb = run_session(session_ootb, inputs)

        tight_tolerances = {
            'float32': 5e-2,
            'float16': 5e-2,
            'bfloat16': 5e-2,
            'fp4': 5e-2,
            'fp8': 5e-2,
            'int8': 5e-2,
            'int4': 5e-2,
        }

        assert torch.sum(
            torch.isclose(
                outputs_moe["output"],
                outputs_ootb["output"],
                atol=tight_tolerances[dtype_str],
                rtol=tight_tolerances[dtype_str])).item() >= math.floor(
                    torch.numel(input_data) * 0.95)

        tolerances = {
            'float32': 1e-2,
            'float16': 1e-2,
            'bfloat16': 2e-2,
            'fp4': 5e-2,
            'fp8': 7e-2,
            'int8': 1e-1,
            'int4': 1e-1,
        }
        torch.testing.assert_close(
            outputs_moe["output"],
            outputs_ootb["output"],
            rtol=tolerances[dtype_str],
            atol=tolerances[dtype_str],
        )

    @parameterized.expand(list(
        product(["float16", "bfloat16", "int4", "int8"], ["gelu", "geglu"],
                [True], [32, 64])),
                          name_func=unittest_name_func)
    def test_mlp_lora_comparison(self, dtype_str, actfn, use_plugin, lora_rank):
        """This test uses one expert and compares the result to a plain MLP"""
        torch.random.manual_seed(42)

        use_int4_weights = dtype_str == "int4"
        weight_dtype = (trt.int8 if use_int4_weights else
                        tensorrt_llm.str_dtype_to_trt(dtype_str))

        dtype = weight_dtype
        quant_mode = QuantMode(0)
        hidden_size = 8
        if dtype_str == "int8" or dtype_str == "int4":
            dtype = tensorrt_llm.str_dtype_to_trt("float16")
            hidden_size = 64
            quant_mode = QuantMode.use_weight_only(
                use_int4_weights=use_int4_weights)

        num_sequences = 4
        sequence_lengths = 4
        num_experts = 1
        top_k = 1
        bias = False
        ffn_hidden_size = 4 * hidden_size
        self.create_weights(
            num_experts,
            hidden_size,
            ffn_hidden_size,
            bias,
            dtype,
            weight_dtype,
            is_gated=is_gated_activation(actfn),
        )

        self.create_lora_weights(
            num_experts,
            hidden_size,
            ffn_hidden_size,
            dtype,
            num_sequences,
            lora_rank,
        )

        input_data = gen_uniform_weights(
            (num_sequences, sequence_lengths, hidden_size),
            dtype=trt_dtype_to_torch(dtype),
        )

        def MLP(network, trt_key, lora_params):
            mlp_type = (tensorrt_llm.layers.GatedMLP if
                        is_gated_activation(actfn) else tensorrt_llm.layers.MLP)
            mlp = mlp_type(
                hidden_size=hidden_size,
                ffn_hidden_size=ffn_hidden_size,
                hidden_act=gated2act(actfn),
                bias=bias,
                quant_mode=quant_mode,
                dtype=dtype,
            )

            mlp.fc.lora = Lora(
                in_hidden_size=hidden_size,
                out_hidden_sizes=[ffn_hidden_size],
                max_low_rank=lora_rank,
            )

            mlp.proj.lora = Lora(
                in_hidden_size=ffn_hidden_size,
                out_hidden_sizes=[hidden_size],
                max_low_rank=lora_rank,
            )

            if is_gated_activation(actfn):
                mlp.gate.lora = Lora(
                    in_hidden_size=hidden_size,
                    out_hidden_sizes=[ffn_hidden_size],
                    max_low_rank=lora_rank,
                )
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

            output = mlp(trt_key, lora_params)
            output.mark_output("mlp_output", dtype)

        session = self.create_trt_session(
            tuple(input_data.shape),
            num_experts,
            top_k,
            hidden_size,
            ffn_hidden_size,
            actfn,
            bias,
            dtype,
            weight_dtype,
            quant_mode,
            norm_mode=MoeConfig.ExpertScaleNormalizationMode.NONE,
            custom_network=MLP,
            use_plugin=use_plugin,
            use_lora=True,
        )

        inputs = {
            "input_hidden_states":
            input_data,
            "moe_h_to_4h_weights_pointers":
            self.lora_fc1_weights_ptrs,
            "moe_h_to_4h_lora_ranks":
            self.lora_fc1_ranks,
            "moe_4h_to_h_weights_pointers":
            self.lora_fc2_weights_ptrs,
            "moe_4h_to_h_lora_ranks":
            self.lora_fc2_ranks,
            "moe_gate_weights_pointers":
            self.lora_gated_weights_ptrs,
            "moe_gate_lora_ranks":
            self.lora_gated_ranks,
            "host_context_lengths":
            torch.tensor((sequence_lengths, ),
                         dtype=torch.int32).repeat(num_sequences),
            "host_request_types":
            torch.tensor((0, ), dtype=torch.int32).repeat(num_sequences),
        }
        outputs = run_session(session, inputs)

        tolerances = {
            "float32": 1e-2,
            "float16": 2e-2,
            "bfloat16": 1e-1,
            "int8": 2e-1,
            "int4": 2e-1,
        }
        torch.testing.assert_close(
            outputs["output"],
            outputs["mlp_output"],
            rtol=tolerances[dtype_str],
            atol=tolerances[dtype_str],
        )

    def set_fp4_scales(self, moe_weight_wrapper, scale_factor: Tensor,
                       num_experts):
        moe_weight_wrapper.weights_block_scaling_factor.value = np.ascontiguousarray(
            torch_to_numpy(scale_factor.view(torch.float8_e4m3fn)))
        moe_weight_wrapper.weights_block_scaling_factor_interleaved.value = (
            np.ascontiguousarray(
                torch_to_numpy(
                    torch.ops.trtllm.block_scale_interleave(
                        scale_factor.view(torch.uint8).contiguous()).view(
                            scale_factor.dtype).reshape(
                                scale_factor.shape).view(torch.uint8))))
        moe_weight_wrapper.activation_global_scaling_factor.value = np.array(
            [1.], dtype=np.float32)
        moe_weight_wrapper.alpha.value = np.array([1.] * num_experts,
                                                  dtype=np.float32)

    def set_weight_layer(self,
                         input_weights,
                         moe_weight_wrapper,
                         quant_mode,
                         fp8_scalar=None):
        if quant_mode.is_weight_only():
            torch_transpose = torch.transpose(input_weights, 1,
                                              2).contiguous().cpu()
            type = torch.quint4x2 if quant_mode.is_int4_weight_only(
            ) else torch.int8
            processed_torch_weights, torch_weight_scales = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                torch_transpose, type)
            # Change the shape to what moe expects without touching the underlying format
            moe_weight_wrapper.weight.value = np.ascontiguousarray(
                torch_to_numpy(processed_torch_weights))
            moe_weight_wrapper.per_channel_scale.value = np.ascontiguousarray(
                torch_to_numpy(torch_weight_scales))
        elif quant_mode.has_fp8_qdq():
            processed_torch_weights = (input_weights /
                                       fp8_scalar.unsqueeze(-1)).to(
                                           torch.float8_e4m3fn)
            moe_weight_wrapper.weight.value = np.ascontiguousarray(
                torch_to_numpy(processed_torch_weights))
            moe_weight_wrapper.weights_scaling_factor.value = np.ascontiguousarray(
                torch_to_numpy(fp8_scalar))
        elif quant_mode.has_nvfp4():
            processed_torch_weights, torch_weight_scales = quant_fp4(
                input_weights)
            self.set_fp4_scales(moe_weight_wrapper, torch_weight_scales,
                                input_weights.shape[0])
            moe_weight_wrapper.weight.value = np.ascontiguousarray(
                torch_to_numpy(processed_torch_weights))
        else:
            moe_weight_wrapper.weight.value = np.ascontiguousarray(
                torch_to_numpy(input_weights))

    def create_trt_session(
        self,
        input_shape,
        num_experts,
        top_k,
        hidden_size,
        ffn_hidden_size,
        actfn,
        bias,
        dtype: trt.DataType,
        weight_dtype: trt.DataType,
        quant_mode,
        norm_mode,
        custom_network=None,
        use_plugin=True,
        max_sizes=None,
        use_lora=False,
        device_limited_n_group=0,
        device_limited_topk_group=0,
        device_limited_routed_scaling_factor=1.0,
    ):
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):
            if max_sizes:
                dim_range = OrderedDict([("max_num_seq", [[1, 1,
                                                           max_sizes[0]]]),
                                         ("max_seq_len", [[1, 1,
                                                           max_sizes[1]]]),
                                         ("hidden_size", [hidden_size])])
            else:
                dim_range = None

            trt_key = Tensor(name='input_hidden_states',
                             shape=input_shape,
                             dim_range=dim_range,
                             dtype=dtype)

            network.plugin_config.moe_plugin = trt_dtype_to_str(dtype)

            lora_params = None
            if use_lora:
                network.plugin_config.lora_plugin = trt_dtype_to_str(dtype)
                network.plugin_config.remove_input_padding = False
                self.create_lora_params(input_shape[0])
                lora_params = self.lora_params

            moe_config = MoeConfig(
                num_experts=num_experts,
                top_k=top_k,
                normalization_mode=norm_mode,
                device_limited_n_group=device_limited_n_group,
                device_limited_topk_group=device_limited_topk_group,
                device_limited_routed_scaling_factor=
                device_limited_routed_scaling_factor)

            moe = tensorrt_llm.layers.MOE(moe_config=moe_config,
                                          hidden_size=hidden_size,
                                          ffn_hidden_size=ffn_hidden_size,
                                          hidden_act=actfn,
                                          bias=bias,
                                          dtype=dtype,
                                          quant_mode=quant_mode)
            moe.router.weight.value = torch_to_numpy(self.router_weights.cpu())

            if use_lora:
                moe.max_low_rank = self.lora_rank

            self.set_weight_layer(self.fc1_weights, moe.fc, quant_mode,
                                  self.weight_scaling_factor_1)
            self.set_weight_layer(self.fc2_weights, moe.proj, quant_mode,
                                  self.weight_scaling_factor_2)

            if quant_mode.has_fp8_qdq():
                moe.fc.activation_scaling_factor.value = torch_to_numpy(
                    self.activation_scaling_factor_1)
                moe.proj.activation_scaling_factor.value = torch_to_numpy(
                    self.activation_scaling_factor_2)
                moe.fc.weights_scaling_factor.value = torch_to_numpy(
                    self.weight_scaling_factor_1)
                moe.proj.weights_scaling_factor.value = torch_to_numpy(
                    self.weight_scaling_factor_2)

            if bias:
                moe.fc.bias.value = torch_to_numpy(self.fc1_bias.cpu())
                moe.proj.bias.value = torch_to_numpy(self.fc2_bias.cpu())
            if quant_mode.has_nvfp4():
                network.plugin_config.gemm_plugin = 'nvfp4' if use_plugin else None
            if custom_network:
                if use_lora:
                    custom_network(network, trt_key, lora_params)
                else:
                    custom_network(network, trt_key)

            if not use_plugin:
                quant_config = None
                if quant_mode.has_fp8_qdq():
                    quant_config = QuantConfig(
                        quant_algo=QuantAlgo.FP8,
                        kv_cache_quant_algo=QuantAlgo.FP8)
                elif quant_mode.has_nvfp4():
                    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
                moe = moe.to(MoeOOTB, quant_config=quant_config)

            output = moe(trt_key, lora_layer_params=lora_params)
            output.mark_output("output", dtype)

            for k, v in moe.named_network_outputs():
                v.mark_output(k, v.dtype)

        # trt run
        session = create_session(builder,
                                 network,
                                 precision=trt_dtype_to_str(dtype),
                                 int8=weight_dtype == trt.int8,
                                 quant_mode=quant_mode)
        return session

    def generate_reference(self, inputs, k, actfn, weight_dtype, quant_mode,
                           norm_mode, n_group, topk_group,
                           routed_scaling_factor):
        # Always run the ref implementation at full precision TODO is this a good choice?
        inputs = inputs.cuda().float()
        inputs_merged = inputs.view(-1, inputs.shape[-1])
        routing = torch.matmul(inputs_merged, self.router_weights.T.float())
        assert routing.shape == (inputs_merged.shape[0],
                                 self.router_weights.shape[0])
        router_probs = torch.softmax(routing, 1, dtype=inputs.dtype)
        assert routing.shape == router_probs.shape

        if norm_mode not in [
                MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED,
                MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED_RENORM
        ]:
            topk_values, topk_indices = torch.topk(router_probs, k)
        else:
            scores = router_probs
            group_scores = (scores.view(scores.shape[0], n_group,
                                        -1).max(dim=-1).values)  # [n, n_group]
            group_idx = torch.topk(group_scores,
                                   k=topk_group,
                                   dim=-1,
                                   sorted=False)[1]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (group_mask.unsqueeze(-1).expand(
                group_mask.shape[0], n_group,
                self.router_weights.shape[0] // n_group).reshape(
                    group_mask.shape[0], -1))  # [n, e]
            scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_values, topk_indices = torch.topk(scores,
                                                   k=k,
                                                   dim=-1,
                                                   sorted=False)

            if k > 1 and norm_mode == MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED_RENORM:
                denominator = topk_values.sum(dim=-1, keepdim=True) + 1e-20
                topk_values = topk_values / denominator
            else:
                topk_values = topk_values * routed_scaling_factor

        assert topk_indices.shape == (router_probs.shape[0], k)
        max_act_2 = 0.0
        results = torch.zeros_like(inputs_merged)
        for i, (scales, experts) in enumerate(zip(topk_values, topk_indices)):
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

                max_act_2 = max(max_act_2, torch.max(torch.abs(fc1)).item())

                fc2_qd = quant_dequant(self.fc2_weights[expert], quant_mode)
                final = torch.matmul(
                    fc1, fc2_qd.T.float()) + self.fc2_bias[expert].float()
                assert final.shape == (inputs.shape[-1], )
                results[i] += scale * final
        return results.view(*inputs.shape), max_act_2
