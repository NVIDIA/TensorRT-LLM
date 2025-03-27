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
from itertools import product

import numpy as np
import pytest
import torch
from parameterized import parameterized
from utils.torch_ref import mamba_conv1d_ref
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_torch


class TestFunctional(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand(
        list(
            product([2048], [4], ['context', 'generation'],
                    ['float16', 'float32', 'bfloat16'], [5], [16], [0, 64],
                    [False, True], [False, True])) +
        # long sequence tests to cover the int overflow issue
        list(
            product([5376], [4], ['context'], ['float16', 'bfloat16'], [2],
                    [131072], [10240], [False, True], [False, True])),
        name_func=unittest_name_func)
    def test_mamba_conv1d(self, dim, dconv, req_type, dtype, batch_size,
                          max_seq_len, stride_size, remove_padding, apply_silu):
        if max_seq_len == 131072:
            total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
            if total_gpu_mem <= 33 * 1024**3:
                pytest.skip(
                    "The long sequence test needs at least 33GB memory, skipping"
                )

        device = "cuda"
        seq_len = max_seq_len if req_type == 'context' else 1
        with_stride = stride_size > 0
        pre_stride = stride_size
        post_stride = 64 if with_stride else 0
        mean = 0.0
        std_dev = 1.0 if dtype == "float32" else 0.5

        # test data
        last_token_ids_trt = None
        torch.random.manual_seed(0)
        if remove_padding and req_type == 'context':
            last_token_ids = torch.randint(1,
                                           seq_len + 1, (batch_size, ),
                                           dtype=torch.int32)
            last_token_ids[0] = seq_len
            host_context_length = last_token_ids.detach().clone().cpu()
            last_token_ids_trt = torch.cumsum(last_token_ids,
                                              dim=0,
                                              dtype=torch.int32).to(device)
        else:
            last_token_ids = torch.ones(
                (batch_size, ), dtype=torch.int32, device=device) * seq_len
            host_context_length = last_token_ids.detach().clone().cpu()
            last_token_ids_trt = last_token_ids
        if req_type == 'context':
            past_conv_state = torch.zeros([batch_size, dim, dconv - 1],
                                          dtype=str_dtype_to_torch(dtype),
                                          device=device)
        else:
            past_conv_state = torch.randn(batch_size,
                                          dim,
                                          dconv - 1,
                                          dtype=str_dtype_to_torch(dtype),
                                          device=device)
            past_conv_state.normal_(mean, std_dev)

        conv_weight = torch.randn([dim, 1, dconv],
                                  dtype=str_dtype_to_torch(dtype),
                                  device=device)
        conv_bias = torch.randn([dim],
                                dtype=str_dtype_to_torch(dtype),
                                device=device)

        host_request_types = torch.tensor([0 if req_type == 'context' else 1] *
                                          batch_size,
                                          dtype=torch.int32)
        x = torch.empty(batch_size,
                        dim,
                        seq_len,
                        device=device,
                        dtype=str_dtype_to_torch(dtype))
        x.normal_(mean, std_dev)

        x_trt = x.detach().permute(0, 2, 1).contiguous()
        if remove_padding and req_type == 'context':
            x_batches = []
            for b in range(batch_size):
                x_batches.append(x_trt[b, :last_token_ids[b], :])
            x_trt = torch.cat(x_batches, dim=0)
        past_conv_state_trt = past_conv_state.permute(0, 2, 1).contiguous()
        conv_weight_trt = conv_weight.permute(1, 2, 0).contiguous()

        if remove_padding and req_type == "generation":
            output_trt = torch.zeros_like(x_trt.squeeze(1))
        else:
            output_trt = torch.zeros_like(x_trt)

        present_conv_state_trt = torch.zeros_like(past_conv_state_trt)
        if with_stride:
            base_shape = [x_trt.shape[i] for i in range(len(x_trt.shape) - 1)]
            pad_pre_shape = base_shape + [pre_stride]
            pad_post_shape = base_shape + [post_stride]
            pad_pre = torch.randn(pad_pre_shape,
                                  device=device,
                                  dtype=str_dtype_to_torch(dtype))
            pad_post = torch.randn(pad_post_shape,
                                   device=device,
                                   dtype=str_dtype_to_torch(dtype))
            x_trt = torch.cat([pad_pre, x_trt, pad_post], dim=-1).contiguous()

        if remove_padding and req_type == "generation":
            x_trt = x_trt.squeeze(1)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        if remove_padding:
            net.plugin_config.remove_input_padding = True
        else:
            net.plugin_config.remove_input_padding = False
        net.plugin_config.paged_state = False
        with tensorrt_llm.net_guard(net):
            x_tensor = Tensor(name='input',
                              shape=x_trt.shape,
                              dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            conv_weight_tensor = Tensor(
                name='conv_weight',
                shape=conv_weight_trt.shape,
                dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            conv_bias_tensor = Tensor(
                name='conv_bias',
                shape=conv_bias.shape,
                dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            past_conv_state_tensor = Tensor(
                name='past_conv_state',
                shape=past_conv_state_trt.shape,
                dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            host_request_types_tensor = Tensor(
                name='host_request_types',
                shape=host_request_types.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            last_token_ids_tensor = Tensor(
                name='last_token_ids',
                shape=last_token_ids.shape,
                dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            host_context_length_tensor = None
            if remove_padding:
                host_context_length_tensor = Tensor(
                    name='host_context_lengths',
                    shape=host_context_length.shape,
                    dtype=tensorrt_llm.str_dtype_to_trt('int32'))
            outputs = tensorrt_llm.functional.mamba_conv1d(
                x_tensor,
                past_conv_state_tensor,
                conv_weight_tensor,
                conv_bias_tensor,
                host_request_types_tensor,
                last_token_ids_tensor,
                dim,
                dconv,
                dtype,
                pre_stride=pre_stride,
                post_stride=post_stride,
                host_context_lengths=host_context_length_tensor,
                apply_silu=apply_silu)
            net._mark_output(outputs[0],
                             'output',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            net._mark_output(outputs[1],
                             'present_conv_state',
                             dtype=tensorrt_llm.str_dtype_to_trt(dtype))

        inputs = {
            'input': x_trt,
            'conv_weight': conv_weight_trt,
            'conv_bias': conv_bias,
            'past_conv_state': past_conv_state_trt,
            'host_request_types': host_request_types,
            'last_token_ids': last_token_ids_trt,
        }
        if remove_padding:
            inputs['host_context_lengths'] = host_context_length
        outputs = {
            'output': output_trt,
            'present_conv_state': present_conv_state_trt
        }
        stream = torch.cuda.current_stream()
        builder_config = builder.create_builder_config(precision=dtype, )
        engine = builder.build_engine(net, builder_config)
        session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
        session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        torch.cuda.synchronize()

        out_ref = torch.zeros_like(x)
        present_conv_state_ref = torch.zeros_like(past_conv_state)

        for b in range(batch_size):
            out_ref[b:b + 1, :, :host_context_length[b].item(
            )], present_conv_state_ref[b:b + 1, :, :] = mamba_conv1d_ref(
                x[b:b + 1, :, :host_context_length[b].item()],
                past_conv_state[b:b + 1, :, :], conv_weight, conv_bias,
                apply_silu)
        present_conv_state_ref = present_conv_state_ref.permute(0, 2,
                                                                1).contiguous()
        out_ref = out_ref.permute(0, 2, 1).contiguous()

        if remove_padding and req_type == 'context':
            out_ref_batches = []
            for b in range(batch_size):
                out_ref_batches.append(out_ref[b, :host_context_length[b], :])
            out_ref = torch.cat(out_ref_batches, dim=0)

        if remove_padding and req_type == "generation":
            out_ref = out_ref.squeeze(1)

        dtype_atol = {"float16": 1e-2, "float32": 2e-3, "bfloat16": 1e-1}

        np.testing.assert_allclose(out_ref.to(torch.float32).cpu().numpy(),
                                   output_trt.to(torch.float32).cpu().numpy(),
                                   atol=dtype_atol[dtype])
        np.testing.assert_allclose(
            present_conv_state_ref.to(torch.float32).cpu().numpy(),
            present_conv_state_trt.to(torch.float32).cpu().numpy(),
            atol=dtype_atol[dtype])
