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
import contextlib
import os
import sys
import unittest

import _utils
import numpy as np
import pytest

# isort: off
import torch
import tensorrt as trt
# isort: on
from parameterized import parameterized
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner
from transformers import GPT2Config

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.quantization import QuantMode

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class GPT2AttentionSmoothQuant(torch.nn.Module):
    """ initially copied from transformers.models.gpt2.modeling_gpt2
        with modifications to run "smoothquant" GEMMs (i.e. i8xi8->i32->fp16)
    """

    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions),
                           dtype=torch.uint8)).view(1, 1, max_positions,
                                                    max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # we can't register linear layer with pytorch in int32. Use functional
        # and define registered buffers instead
        self.register_buffer("c_attn_weight",
                             torch.empty((3 * self.embed_dim, self.embed_dim)))
        self.register_buffer("c_attn_bias", torch.empty(3 * self.embed_dim))
        self.register_buffer("c_proj_weight",
                             torch.empty((self.embed_dim, self.embed_dim)))
        self.register_buffer("c_proj_bias", torch.empty(self.embed_dim))

        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1))**0.5)

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length -
                                query_length:key_length, :key_length].bool()
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device)
        attn_weights = torch.where(causal_mask,
                                   attn_weights.to(attn_weights.dtype),
                                   mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1,
                              3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size, )
        return tensor.view(new_shape)

    def forward(
            self,
            hidden_states,
            dtype,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
            quant_mode=QuantMode(0),
            c_attn_dyn_scaling_factor=None,
    ):
        if not quant_mode.has_act_and_weight_quant():
            raise ValueError("quant_mode has to have some quantization")

        qkv = _utils.gt_matmul_smooth_quant(hidden_states, self.c_attn_weight,
                                            self.scale_attn_out,
                                            self.scale_attn_w, dtype)
        qkv = (qkv + self.c_attn_bias).to(
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        query, key, value = qkv.split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value,
                                               attention_mask, head_mask)

        def to_i8(x):
            return x.round().clip(-128, 127).to(dtype=torch.int8)

        attn_output = self._merge_heads(attn_output, self.num_heads,
                                        self.head_dim)

        scales = self.scale_proj_out
        if quant_mode.has_act_static_scaling():
            attn_output = to_i8(attn_output * self.scale_proj_in.cuda())
        else:
            attn_output, scales = _utils.gt_quantize_per_token(attn_output)

        attn_output = _utils.gt_matmul_smooth_quant(attn_output,
                                                    self.c_proj_weight, scales,
                                                    self.scale_proj_w, dtype)
        attn_output = (attn_output + self.c_proj_bias).to(
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights, )

        return outputs  # a, present, (attentions)


class TestSmoothQuant(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('error')

    @parameterized.expand([
        ('float16', False, False, False,
         tensorrt_llm.quantization.layers.SmoothQuantLinear),
        ('float16', False, True, False,
         tensorrt_llm.quantization.layers.SmoothQuantLinear),
        ('float16', True, False, False,
         tensorrt_llm.quantization.layers.SmoothQuantLinear),
        ('float16', True, True, False,
         tensorrt_llm.quantization.layers.SmoothQuantLinear),
        ('float32', True, True, False,
         tensorrt_llm.quantization.layers.SmoothQuantLinear),
        ('int32', True, True, False,
         tensorrt_llm.quantization.layers.SmoothQuantLinear),
        ('float32', False, False, True,
         tensorrt_llm.quantization.layers.SmoothQuantLinear),
        ('float32', False, False, True,
         tensorrt_llm.quantization.layers.SmoothQuantRowLinear)
    ])
    @pytest.mark.skipif(
        getSMVersion() < 80,
        reason="INT8 TC is not supported in pre-ampere architecture"
    )  # Skip tests that are not supported in pre-ampere architecture
    def test_linear_smooth_quant(self, dtype, per_token_scaling,
                                 per_channel_scaling, bias, linear_cls):
        # test data
        d_h = 32
        ffn_h = 64
        test_shape = [2, 3, 5, d_h]

        # Init operands for multiplication in int8
        x_data = torch.randint(-128, 128, test_shape, dtype=torch.int8)
        fc1 = torch.randint(-128, 128, (ffn_h, d_h), dtype=torch.int8)

        bias_data = None
        if bias:
            bias_data = torch.randint(
                -5,
                5, (ffn_h, ),
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)) * 0.1

        m = test_shape[0] * test_shape[1] * test_shape[2]
        test_shape[3]
        c_1 = ffn_h

        quant_mode = QuantMode.from_description(True, True, per_token_scaling,
                                                per_channel_scaling)

        def init_scales(n):
            scale_a_shape = (m, 1) if per_token_scaling else (1, 1)
            scale_a = torch.ones(scale_a_shape, dtype=torch.float32) * 1e-2
            scale_a *= torch.randint(1, 10, scale_a_shape, dtype=torch.float32)
            scale_b_shape = (1, n) if per_channel_scaling else (1, 1)
            scale_b = torch.ones(scale_b_shape, dtype=torch.float32) * 1e-2
            scale_b *= torch.randint(1, 10, scale_b_shape, dtype=torch.float32)
            return scale_a, scale_b

        scale_fc1_out, scale_fc1_w = init_scales(c_1)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        # Allow SQ plugin of dtype type
        net.plugin_config.set_smooth_quant_gemm_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt('int8'))

            args = {}
            if linear_cls == tensorrt_llm.quantization.layers.SmoothQuantLinear:
                args['gather_output'] = False
            gm = linear_cls(d_h,
                            ffn_h,
                            bias=bias,
                            quant_mode=quant_mode,
                            **args)

            # TensorRT-LLM's Linear uses Parameter class which as a 'value' setter
            gm.weight.value = fc1.cpu().numpy()
            gm.per_channel_scale.value = scale_fc1_w.cpu().numpy()
            if bias:
                gm.bias.value = bias_data.cpu().numpy()
            # Set activation scaling factors if needed
            if quant_mode.has_act_static_scaling():
                gm.act_scale.value = scale_fc1_out.cpu().numpy()

            input = x
            # If we have dynamic scaling, Linear expects Tuple input:
            # (auntized tensor, scales per token)
            if quant_mode.has_per_token_dynamic_scaling():
                scale_dynamic = Tensor(
                    name='scale_dynamic',
                    shape=scale_fc1_out.shape,
                    dtype=tensorrt_llm.str_dtype_to_trt('float32'))

                input = (x, scale_dynamic)

            output = gm.forward(input).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                int8=True,
                fp16=(dtype == "float16"),
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

        with TrtRunner(build_engine) as runner:
            feed_dict = {'x': x_data.numpy()}
            if quant_mode.has_per_token_dynamic_scaling():
                feed_dict['scale_dynamic'] = scale_fc1_out.numpy()
            outputs = runner.infer(feed_dict=feed_dict)

        # pytorch run
        with torch.no_grad():
            ref = _utils.gt_matmul_smooth_quant(x_data, fc1, scale_fc1_out,
                                                scale_fc1_w, dtype, bias_data)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

    @parameterized.expand([
        (tensorrt_llm.quantization.layers.SmoothQuantLinear),
        (tensorrt_llm.quantization.layers.SmoothQuantRowLinear)
    ])
    def test_linear_smooth_quant_no_quant(self, linear_cls):
        # Weight only quant for SmoothQuant
        quant_mode = QuantMode.from_description(quantize_weights=True,
                                                quantize_activations=False,
                                                per_token=False,
                                                per_channel=False)

        args = {}
        if linear_cls == tensorrt_llm.quantization.layers.SmoothQuantLinear:
            args['gather_output'] = False

        # Create builder
        builder = tensorrt_llm.Builder()
        # Create empty network
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            tensorrt_llm.default_trtnet()
            # Get output tensor for SQ Linear
            with self.assertRaisesRegex(
                    ValueError,
                    "SmoothQuant Linear has to have act\+weight quantization mode set"
            ):
                linear_cls(32, 64, bias=False, quant_mode=quant_mode, **args)

    @parameterized.expand([('float16', False, False, 'gelu'),
                           ('float16', False, True, 'gelu'),
                           ('float16', True, False, 'gelu'),
                           ('float16', True, True, 'gelu'),
                           ('float32', True, True, 'gelu'),
                           ('float32', True, True, 'elu')])
    @pytest.mark.skipif(
        getSMVersion() < 80,
        reason="INT8 TC is not supported in pre-ampere architecture"
    )  # Skip tests that are not supported in pre-ampere architecture
    def test_mlp_smooth_quant(self, dtype, per_token_scaling,
                              per_channel_scaling, hidden_act):
        # test data
        d_h = 16
        ffn_h = 32
        test_shape = [2, 3, 5, d_h]

        torch.manual_seed(42)

        # Init operands for multiplication in int8
        x_data = torch.randint(-8, 8, test_shape, dtype=torch.int8)
        fc1 = torch.randint(-16, 16, (ffn_h, d_h), dtype=torch.int8)
        fc2 = torch.randint(-16, 16, (d_h, ffn_h), dtype=torch.int8)

        m = test_shape[0] * test_shape[1] * test_shape[2]
        c_1 = ffn_h
        c_2 = d_h

        quant_mode = QuantMode.from_description(True, True, per_token_scaling,
                                                per_channel_scaling)

        def init_scales(n):
            scale_a_shape = (m, 1) if per_token_scaling else (1, 1)
            scale_a = torch.ones(scale_a_shape, dtype=torch.float32) * 1e-2
            scale_a *= torch.randint(1, 10, scale_a_shape, dtype=torch.float32)
            scale_b_shape = (1, n) if per_channel_scaling else (1, 1)
            scale_b = torch.ones(scale_b_shape, dtype=torch.float32) * 1e-2
            scale_b *= torch.randint(1, 10, scale_b_shape, dtype=torch.float32)
            return scale_a, scale_b

        scale_fc1_out, scale_fc1_w = init_scales(c_1)
        scale_fc2_out, scale_fc2_w = init_scales(c_2)
        scale_fc2_in = torch.randint(3, 7, (1, ), dtype=torch.float32) * 0.1

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        # Allow SQ plugin of dtype type
        net.plugin_config.set_smooth_quant_gemm_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt('int8'))

            if hidden_act == 'elu':
                context = self.assertRaisesRegex(
                    ValueError, "unsupported activation function: *")
            else:
                context = contextlib.nullcontext()

            with context:
                gm = tensorrt_llm.quantization.layers.SmoothQuantMLP(
                    d_h,
                    ffn_h,
                    hidden_act=hidden_act,
                    bias=False,
                    quant_mode=quant_mode)

            if hidden_act != 'gelu':
                return

            # TensorRT-LLM's MLP uses Parameter class which as a 'value' setter
            gm.fc.weight.value = fc1.cpu().numpy()
            gm.fc.per_channel_scale.value = scale_fc1_w.cpu().numpy()
            gm.proj.weight.value = fc2.cpu().numpy()
            gm.proj.per_channel_scale.value = scale_fc2_w.cpu().numpy()
            gm.proj.smoother.value = np.ones([1, fc2.shape[1]],
                                             dtype=np.float32)

            # Set activation scaling factors if needed
            if quant_mode.has_act_static_scaling():
                gm.quantization_scaling_factor.value = scale_fc2_in.cpu().numpy(
                )
                gm.fc.act_scale.value = scale_fc1_out.cpu().numpy()
                gm.proj.act_scale.value = scale_fc2_out.cpu().numpy()

            input = x
            if quant_mode.has_per_token_dynamic_scaling():
                scale_dynamic = Tensor(
                    name='scale_dynamic',
                    shape=scale_fc1_out.shape,
                    dtype=tensorrt_llm.str_dtype_to_trt('float32'))

                input = (x, scale_dynamic)

            output = gm.forward(input).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                int8=True,
                fp16=(dtype == "float16"),
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

        with TrtRunner(build_engine) as runner:
            feed_dict = {'x': x_data.numpy()}
            if quant_mode.has_per_token_dynamic_scaling():
                feed_dict['scale_dynamic'] = scale_fc1_out.numpy()
            outputs = runner.infer(feed_dict=feed_dict)

        # pytorch run
        with torch.no_grad():
            gelu = torch.nn.GELU()
            # FC 1
            hidden = _utils.gt_matmul_smooth_quant(x_data, fc1, scale_fc1_out,
                                                   scale_fc1_w, dtype)
            # ACT
            hidden = gelu(hidden)

            # Dynamic/static quantization
            scale_act = scale_fc2_out
            if quant_mode.has_per_token_dynamic_scaling():
                hidden, scale_act = _utils.gt_quantize_per_token(hidden)
            else:
                hidden = (hidden * scale_fc2_in.cuda()).round().clip(
                    -128, 127).to(dtype=torch.int8)

            # FC 2
            ref = _utils.gt_matmul_smooth_quant(hidden, fc2, scale_act,
                                                scale_fc2_w, dtype)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=5e-2)

    @parameterized.expand([('float16', True, True), ('float16', True, False)])
    def test_smooth_quant_layer_norm_layer(self, dtype, per_token_scaling,
                                           elementwise_affine):
        torch.manual_seed(1997)
        # test data
        hidden_size = 1024
        x_data = torch.randn(
            (8, 128, hidden_size),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)).cuda()
        eps = 1e-5

        m = torch.nn.LayerNorm(
            hidden_size,
            eps=eps,
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            elementwise_affine=elementwise_affine).cuda()

        # Scale to int
        scale_data = torch.randint(2, 32, (1, ), dtype=torch.float32).cuda()
        scale_to_int_data = torch.ones((1, ), dtype=torch.float32).cuda()

        quant_mode = QuantMode.from_description(quantize_weights=True,
                                                quantize_activations=True,
                                                per_token=per_token_scaling,
                                                per_channel=False)

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_layernorm_quantization_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            ln = tensorrt_llm.quantization.layers.SmoothQuantLayerNorm(
                hidden_size,
                quant_mode=quant_mode,
                elementwise_affine=elementwise_affine,
                dtype=dtype)

            ln.scale_to_int.value = scale_to_int_data.detach().cpu().numpy()

            if elementwise_affine:
                gamma_data = m.weight.detach().cpu()
                beta_data = m.bias.detach().cpu()
                ln.weight.value = gamma_data.cpu().numpy()
                ln.bias.value = beta_data.cpu().numpy()

            output = ln.forward(x)

            if per_token_scaling:
                output, dynamic_scales = output
                net._mark_output(dynamic_scales, 'dynamic_scales', trt.float32)
            net._mark_output(output, 'output', trt.int8)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(int8=True,
                                fp16=(dtype == 'float16'),
                                precision_constraints="obey"))
        assert build_engine is not None, "Build engine failed"
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.cpu().numpy()})

        def cast_to_int8_with_sat(tensor):
            return tensor.round().clip(-128, 127).to(dtype=torch.int8)

        # pytorch run
        with torch.no_grad():
            ref = m(x_data).to(dtype=torch.float32)
            if per_token_scaling:
                abs_max_f, _ = ref.abs().max(dim=-1, keepdim=True)
                dynamic_scale = abs_max_f / 127.0
                ref_quantized = cast_to_int8_with_sat(ref * (127.0 / abs_max_f))
            else:
                ref_quantized = cast_to_int8_with_sat(ref * scale_data)

        # compare diff of quantized output
        # Set absolute tolerance to 1 to mitigate some rounding error
        np.testing.assert_allclose(ref_quantized.cpu().numpy(),
                                   outputs['output'],
                                   atol=1,
                                   rtol=0)

        # compare diff of dynamic activation scales
        if per_token_scaling:
            np.testing.assert_allclose(dynamic_scale.cpu().numpy(),
                                       outputs['dynamic_scales'],
                                       atol=1e-2)

    @parameterized.expand([
        ('float16', 1, False,
         tensorrt_llm.quantization.layers.WeightOnlyQuantLinear),
        ('float16', 2, False,
         tensorrt_llm.quantization.layers.WeightOnlyQuantLinear),
        ('float16', 1, True,
         tensorrt_llm.quantization.layers.WeightOnlyQuantLinear),
        ('float16', 1, True,
         tensorrt_llm.quantization.layers.WeightOnlyQuantRowLinear)
    ])
    @pytest.mark.skipif(
        getSMVersion() < 80,
        reason="weight only groupwise contains bug on pre-ampere"
    )  # Skip tests that are not supported in pre-ampere architecture
    def test_linear_weight_only_linear(self, dtype, wTypeId, bias, linear_cls):
        # test data
        m = 1
        n = 1024
        k = 4096

        # Init operands for multiplication in int32
        mat1 = _utils.woq_gen_weights(m, k, dtype) * 200.0
        weight = _utils.woq_gen_weights(k, n, dtype)

        ref_torch_weights, processed_torch_weights, torch_weight_scales = _utils.woq_conversion(
            weight, wTypeId)
        if wTypeId == 2:
            ref_torch_weights = torch.ops.fastertransformer.unpack_int4_packed_tensor_to_int8(
                ref_torch_weights)

        bias_data = None
        if bias:
            bias_data = torch.randint(
                -5,
                5, (n, ),
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)) * 0.1

        quant_mode = QuantMode.from_description(quantize_weights=True,
                                                quantize_activations=False,
                                                per_token=False,
                                                per_channel=False,
                                                use_int4_weights=(wTypeId == 2))

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        net.plugin_config.set_weight_only_quant_matmul_plugin(dtype)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=mat1.shape,
                       dtype=tensorrt_llm._utils.str_dtype_to_trt(dtype))

            args = {}
            if linear_cls == tensorrt_llm.quantization.layers.WeightOnlyQuantLinear:
                args['gather_output'] = False
            gm = linear_cls(k, n, bias=bias, quant_mode=quant_mode, **args)

            # TensorRT-LLM's Linear uses Parameter class which as a 'value' setter
            gm.weight.value = processed_torch_weights.cpu().numpy()
            gm.per_channel_scale.value = torch_weight_scales.cpu().numpy()
            if bias:
                gm.bias.value = bias_data.cpu().numpy()

            input = x

            output = gm.forward(input).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(
                int8=True,
                fp16=(dtype == "float16"),
                memory_pool_limits={trt.MemoryPoolType.WORKSPACE: 33554432}))

        with TrtRunner(build_engine) as runner:
            feed_dict = {'x': mat1.numpy()}
            output = torch.tensor(runner.infer(feed_dict=feed_dict)["output"])

        # pytorch run
        with torch.no_grad():
            ref = _utils.woq_gt_matmul(m, mat1, ref_torch_weights,
                                       torch_weight_scales, dtype, bias_data)

        # compare diff
        _utils.woq_assert_colwise_near_eq(ref, output, wTypeId)

    # @parameterized.expand([
    #     ('float16', QuantMode.PER_TENSOR),
    #     # ('float16', QuantMode.PER_CHANNEL),
    #     # ('float16', QuantMode.PER_TOKEN),
    #     # ('float16', QuantMode.PER_TOKEN_PER_CHANNEL),
    # ])
    @unittest.skip("Attention contains a bug and will be resolved in later MRs")
    def test_gpt_attention_smoothquant(self,
                                       dtype="float16",
                                       quant_mode=QuantMode.from_description(
                                           True, True, False, False)):

        def _construct_execution():
            builder = tensorrt_llm.Builder()
            net = builder.create_network()
            net.plugin_config.set_smooth_quant_gemm_plugin(dtype)
            net.plugin_config.set_gpt_attention_plugin(dtype)
            with tensorrt_llm.net_guard(net):
                hidden_states_tensor = Tensor(
                    name='hidden_states',
                    shape=tuple(input.shape),
                    dtype=tensorrt_llm.str_dtype_to_trt('int8'))

                input_tensor = hidden_states_tensor
                if quant_mode.has_per_token_dynamic_scaling():
                    scale_dynamic_tensor = Tensor(
                        name='scale_dynamic',
                        shape=tuple(scale_attn_out.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt('float32'))

                    input_tensor = (hidden_states_tensor, scale_dynamic_tensor)

                past_key_value = None
                if use_past_key_value or use_gpt_attention_plugin:
                    past_key_tensor = Tensor(
                        name='past_key',
                        shape=tuple(present_key.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt(dtype))
                    past_value_tensor = Tensor(
                        name='past_value',
                        shape=tuple(present_value.shape),
                        dtype=tensorrt_llm.str_dtype_to_trt(dtype))

                    past_key_value = (past_key_tensor, past_value_tensor)

                sequence_length_tensor = None
                past_key_value_length_tensor = None
                input_lengths_tensor = None
                cache_indirection_tensor = None
                if use_gpt_attention_plugin:
                    sequence_length_tensor = Tensor(name='sequence_length',
                                                    dtype=trt.int32,
                                                    shape=tuple(
                                                        sequence_length.shape))
                    past_key_value_length_tensor = Tensor(
                        name='past_key_value_length',
                        dtype=trt.int32,
                        shape=tuple(past_key_value_length.shape))

                    input_lengths_tensor = Tensor(name='input_lengths',
                                                  dtype=trt.int32,
                                                  shape=tuple(
                                                      input_lengths.shape))

                    cache_indirection_tensor = Tensor(
                        name='cache_indirection',
                        dtype=trt.int32,
                        shape=tuple(cache_indirection.shape))

                attention = tensorrt_llm_gpt

                attention.qkv.weight.value = weight_qkv.cpu().numpy()
                attention.qkv.bias.value = bias_qkv.cpu().numpy()
                attention.qkv.per_channel_scale.value = scale_attn_w.cpu(
                ).numpy()

                attention.dense.weight.value = weight_proj.cpu().numpy()
                attention.dense.bias.value = bias_proj.cpu().numpy()
                attention.dense.per_channel_scale.value = scale_proj_w.cpu(
                ).numpy()
                attention.dense.smoother.value = np.ones(
                    [1, weight_proj.shape[1] * 4], dtype=np.float32)

                # Set activation scaling factors if needed
                if quant_mode.has_act_static_scaling():
                    attention.quantization_scaling_factor.value = scale_proj_in.cpu(
                    ).numpy()
                    attention.qkv.act_scale.value = scale_attn_out.cpu().numpy()
                    attention.dense.act_scale.value = scale_proj_out.cpu(
                    ).numpy()

                outputs = attention(
                    input_tensor,
                    attention_mask=None,
                    past_key_value=past_key_value,
                    sequence_length=sequence_length_tensor,
                    past_key_value_length=past_key_value_length_tensor,
                    use_cache=True,
                    input_lengths=input_lengths_tensor,
                    cache_indirection=cache_indirection_tensor)

                trt_dtype = tensorrt_llm.str_dtype_to_trt(dtype)
                net._mark_output(outputs[0], 'output', trt_dtype)
                net._mark_output(outputs[1][0], 'present_key', trt_dtype)
                net._mark_output(outputs[1][1], 'present_value', trt_dtype)

            # Apply dynamic range to int8 connections
            for l in net.trt_network:
                for ii in range(l.num_inputs):
                    if l.get_input(
                            ii).dtype == tensorrt_llm._utils.str_dtype_to_trt(
                                "int8"):
                        l.get_input(ii).set_dynamic_range(-127, 127)
                for oi in range(l.num_outputs):
                    if l.get_output(
                            oi).dtype == tensorrt_llm._utils.str_dtype_to_trt(
                                "int8"):
                        l.get_output(oi).set_dynamic_range(-127, 127)

            # trt build engine
            build_engine = EngineFromNetwork(
                (builder.trt_builder, net.trt_network),
                config=CreateConfig(int8=True,
                                    fp16=(dtype == "float16"),
                                    precision_constraints="obey",
                                    memory_pool_limits={
                                        trt.MemoryPoolType.WORKSPACE:
                                        48 * (2**30)
                                    }))

            # Infer engine
            with TrtRunner(build_engine) as runner:
                feed_dict = {'hidden_states': input.cpu().numpy()}

                if use_past_key_value or use_gpt_attention_plugin:
                    feed_dict['past_key'] = present_key if isinstance(
                        present_key, np.ndarray) else present_key.cpu().numpy()
                    feed_dict['past_value'] = present_value if isinstance(
                        present_value,
                        np.ndarray) else present_value.cpu().numpy()

                if use_gpt_attention_plugin:
                    feed_dict['sequence_length'] = sequence_length.cpu().numpy()
                    feed_dict[
                        'past_key_value_length'] = past_key_value_length.cpu(
                        ).numpy()
                    feed_dict['input_lengths'] = input_lengths.cpu().numpy()

                if quant_mode.has_per_token_dynamic_scaling():
                    feed_dict['scale_dynamic'] = scale_attn_out.cpu().numpy()

                feed_dict['cache_indirection'] = cache_indirection.cpu().numpy()

                outputs = runner.infer(feed_dict=feed_dict)

            return outputs

        batch_size = 4
        in_len = 128
        out_len = 8
        max_seq_len = 148
        hidden_size = 1024
        num_heads = 16
        head_size = hidden_size // num_heads
        shape_dict = {
            'weight': (hidden_size * 3, hidden_size),
            'bias': (hidden_size * 3, ),
            'past_key': (batch_size, num_heads, max_seq_len, head_size),
            'past_value': (batch_size, num_heads, max_seq_len, head_size),
            'present_key': (batch_size, num_heads, max_seq_len, head_size),
            'present_value': (batch_size, num_heads, max_seq_len, head_size),
            'sequence_length': (batch_size, ),
            'input_lengths': (batch_size, ),
        }

        weight_qkv = torch.randint(-10,
                                   10,
                                   shape_dict['weight'],
                                   dtype=torch.int8)
        bias_qkv = torch.randn(
            shape_dict['bias'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda') * 1e-1

        weight_proj = torch.eye(hidden_size, dtype=torch.int8)
        bias_proj = torch.zeros(
            (hidden_size, ),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda')

        input_lengths = torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda') * in_len
        cache_indirection = torch.full((
            batch_size,
            1,
            max_seq_len,
        ),
                                       0,
                                       dtype=torch.int32,
                                       device='cuda')

        present_key = torch.zeros(
            shape_dict['present_key'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda')
        present_value = torch.zeros(
            shape_dict['present_value'],
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
            device='cuda')
        torch_present = None

        per_token_scaling = quant_mode.has_per_token_dynamic_scaling()
        per_channel_scaling = quant_mode.has_per_channel_scaling()

        def init_scales(m, n, token_scaling, channel_scaling):
            scale_a_shape = (m, 1) if token_scaling else (1, 1)
            scale_a = torch.ones(scale_a_shape, dtype=torch.float32) * 1e-2
            scale_a *= torch.randint(1, 10, scale_a_shape, dtype=torch.float32)
            scale_b_shape = (1, n) if channel_scaling else (1, 1)
            scale_b = torch.ones(scale_b_shape, dtype=torch.float32) * 1e-2
            scale_b *= torch.randint(1, 10, scale_b_shape, dtype=torch.float32)
            return scale_a, scale_b

        # We always do per channel scaling for QKV
        scale_attn_out, scale_attn_w = init_scales(batch_size * in_len,
                                                   3 * hidden_size,
                                                   per_token_scaling, True)
        scale_proj_out, scale_proj_w = init_scales(batch_size * in_len,
                                                   hidden_size,
                                                   per_token_scaling,
                                                   per_channel_scaling)
        scale_proj_in = torch.randint(3, 7, (1, ), dtype=torch.float32) * 0.1

        # instantiate pytorch equivalent of attention SQ
        configuration = GPT2Config(
            hidden_size=hidden_size,
            n_layer=1,
            n_head=num_heads,
            vocab_size=51200,
            use_cache=True,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            hidden_act='gelu',
            torch_dtype=dtype,
        )
        n_positions = configuration.n_positions

        gt_attention = GPT2AttentionSmoothQuant(configuration).cuda().eval()
        gt_attention.c_attn_weight = torch.nn.parameter.Parameter(
            data=weight_qkv.clone().detach(), requires_grad=False)
        gt_attention.c_attn_bias = torch.nn.parameter.Parameter(
            data=bias_qkv.clone().detach(), requires_grad=False)
        gt_attention.c_proj_weight = torch.nn.parameter.Parameter(
            data=weight_proj, requires_grad=False)
        gt_attention.c_proj_bias = torch.nn.parameter.Parameter(
            data=bias_proj, requires_grad=False)
        gt_attention.scale_attn_out, gt_attention.scale_proj_out = scale_attn_out, scale_proj_out
        gt_attention.scale_attn_w, gt_attention.scale_proj_w = scale_attn_w, scale_proj_w
        gt_attention.scale_proj_in = scale_proj_in

        # instantiate full gpt model before isolating its attention module
        tensorrt_llm_gpt = tensorrt_llm.quantization.layers.SmoothQuantAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            max_position_embeddings=n_positions,
            num_layers=1,
            attention_mask_type=tensorrt_llm.layers.AttentionMaskType.causal,
            dtype=dtype,
            quant_mode=quant_mode)

        for step in range(out_len):
            sequence_length = torch.ones(
                (batch_size, ), dtype=torch.int32,
                device='cuda') * (in_len + step)
            if step == 0:
                # Context stage
                shape_dict['hidden_states'] = (batch_size, in_len, hidden_size)
                shape_dict['output'] = shape_dict['hidden_states']
                past_key_value_length = torch.tensor([step], dtype=torch.int32)

                input = torch.randint(-16,
                                      16,
                                      shape_dict['hidden_states'],
                                      dtype=torch.int8)

                # torch execution
                torch_output, torch_present = gt_attention(
                    input,
                    dtype,
                    layer_past=None,
                    use_cache=True,
                    quant_mode=quant_mode)

                use_past_key_value = False
                use_gpt_attention_plugin = True
                outputs = _construct_execution()
                output = outputs["output"]
                present_key = outputs["present_key"]
                present_value = outputs["present_value"]

                print(output, torch_output)

                np.testing.assert_allclose(output,
                                           torch_output.cpu().numpy(),
                                           atol=1e-2)

            else:
                # Generation stage
                shape_dict['hidden_states'] = (batch_size, 1, hidden_size)
                shape_dict['output'] = shape_dict['hidden_states']
                past_key_value_length = torch.tensor([in_len + step - 1],
                                                     dtype=torch.int32)
                input = torch.randint(-16,
                                      16,
                                      shape_dict['hidden_states'],
                                      dtype=torch.int8)

                # torch execution
                torch_output, torch_present = gt_attention(
                    input,
                    dtype,
                    layer_past=torch_present,
                    use_cache=True,
                    quant_mode=quant_mode)

                use_past_key_value = True
                use_gpt_attention_plugin = True
                outputs = _construct_execution()
                output = outputs["output"]
                present_key = outputs["present_key"]
                present_value = outputs["present_value"]

                print(output, torch_output)

                np.testing.assert_allclose(output,
                                           torch_output.cpu().numpy(),
                                           atol=1e-2)

    def test_quantize_per_tensor(self):
        dtype = 'float32'
        x_data = torch.randn((1, 2, 2, 4), dtype=torch.float32)
        scaling_factor_data = torch.tensor(0.4, dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        with tensorrt_llm.net_guard(net):
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            q_layer = tensorrt_llm.quantization.layers.Quantize('int8')
            q_layer.scaling_factor.value = scaling_factor_data.numpy()
            output = q_layer.forward(x)
            net._mark_output(output, 'output', trt.int8)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = torch.quantize_per_tensor(x_data, scaling_factor_data, 0,
                                        torch.qint8)

        np.testing.assert_allclose(ref.int_repr().cpu().numpy(),
                                   outputs['output'])

    def test_quantize_per_channel(self):
        dtype = 'float32'
        x_data = torch.randn((2, 4, 4, 8), dtype=torch.float32)
        scaling_factor_data = torch.tensor((0.4, 0.1, 0.3, 0.2),
                                           dtype=torch.float32)
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            axis = 1
            q_layer = tensorrt_llm.quantization.layers.Quantize(
                'int8', 'float32', x_data.shape[axis], axis)
            q_layer.scaling_factor.value = scaling_factor_data.detach().cpu(
            ).numpy()
            output = q_layer.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = torch.quantize_per_channel(x_data, scaling_factor_data,
                                         torch.tensor([0, 0, 0, 0]), 1,
                                         torch.qint8)

        np.testing.assert_allclose(ref.int_repr().cpu().numpy(),
                                   outputs['output'])

    @parameterized.expand([('float16'), ('float32')])
    def test_quantize_per_token(self, dtype):
        x_data = torch.randn(
            (2, 4, 4, 8), dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))

        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            q_layer = tensorrt_llm.quantization.layers.QuantizePerToken()
            output, scale = q_layer.forward(x)

            output = output.trt_tensor
            output.name = 'output'
            network.mark_output(output)

            scale = scale.trt_tensor
            scale.name = 'scale'
            network.mark_output(scale)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)

        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref, ref_scale = _utils.gt_quantize_per_token(x_data)
        ref = ref.reshape(outputs['output'].shape)
        ref_scale = ref_scale.reshape(outputs['scale'].shape)

        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])

        np.testing.assert_allclose(ref_scale.cpu().numpy(),
                                   outputs['scale'],
                                   atol=1e-2)

    def test_dequantize(self):
        dtype = 'int8'
        quantized_torch_tensor = torch.quantize_per_tensor(
            torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float32), 0.1, 0,
            torch.qint8)
        quantized_data = quantized_torch_tensor.int_repr()
        scaling_factor_data = torch.tensor(0.1, dtype=torch.float32)

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        config = builder.trt_builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=quantized_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            dq_layer = tensorrt_llm.quantization.layers.Dequantize()

            dq_layer.scaling_factor.value = scaling_factor_data.numpy()
            output = dq_layer.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network),
                                         config)
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={'x': quantized_data.cpu().numpy()})
        ref = torch.dequantize(quantized_torch_tensor)

        np.testing.assert_allclose(ref.cpu().numpy(), outputs['output'])


if __name__ == '__main__':
    unittest.main()
