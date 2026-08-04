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
import tempfile
import unittest
from collections import OrderedDict

import numpy as np
import pytest
import torch
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, TrtRunner

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.builder import Builder
from tensorrt_llm.models.unet.attention import Transformer2DModel
from tensorrt_llm.models.unet.embeddings import TimestepEmbedding, Timesteps
from tensorrt_llm.models.unet.resnet import (Downsample2D, ResnetBlock2D,
                                             Upsample2D)

# isort: off
from tensorrt_llm.models.unet.unet_2d_blocks import (
    CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UNetMidBlock2D,
    UNetMidBlock2DCrossAttn, UpBlock2D)
# isort: on
from tensorrt_llm.models.unet.unet_2d_condition import UNet2DConditionModel
from tensorrt_llm.models.unet.weights import *
from tensorrt_llm.network import net_guard


@pytest.mark.skip(reason="Skip for saving CI pipeline time")
class TestUNet(unittest.TestCase):

    def test_transformer_2d_model_float32(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((2, 4, 16, 16))
        # FIXME: If we import diffusers at the top, and run unit test with
        # pytest --forked, a strange error throws:
        # [TRT] [W] Unable to determine GPU memory usage
        # [TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 91, GPU 0 (MiB)
        # [TRT] [W] CUDA initialization failure with error: 3. Please check your CUDA installation
        from diffusers.models.attention import Transformer2DModel as dt
        m = dt(in_channels=4,
               num_attention_heads=8,
               attention_head_dim=8,
               norm_num_groups=2).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            gm = Transformer2DModel(in_channels=4,
                                    num_attention_heads=8,
                                    attention_head_dim=8,
                                    norm_num_groups=2)

            update_transformer_2d_model_weight(gm, m)

            output = gm.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        # pytorch run
        with torch.no_grad():
            ref = m(x_data)
        ref = ref.sample

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_transformer_2d_model_float16(self):
        # test data
        dtype = 'float16'
        d_head = 64
        in_channels = 32
        n_heads = 8
        cross_attention_dim = 1280
        x_data = torch.randn(
            (2, in_channels, 16, 16),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)).cuda()
        y_data = torch.randn(
            (2, 128, cross_attention_dim),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)).cuda()
        # FIXME: If we import diffusers at the top, and run unit test with
        # pytest --forked, a strange error throws:
        # [TRT] [W] Unable to determine GPU memory usage
        # [TRT] [I] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 91, GPU 0 (MiB)
        # [TRT] [W] CUDA initialization failure with error: 3. Please check your CUDA installation
        from diffusers.models.attention import Transformer2DModel as dt
        m = dt(in_channels=in_channels,
               num_attention_heads=n_heads,
               attention_head_dim=d_head,
               cross_attention_dim=cross_attention_dim).cuda().half().eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network

        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=y_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            gm = Transformer2DModel(in_channels=in_channels,
                                    num_attention_heads=n_heads,
                                    attention_head_dim=d_head,
                                    cross_attention_dim=cross_attention_dim)

            update_transformer_2d_model_weight(gm, m)

            output = gm.forward(x, y).trt_tensor
            output.name = 'output'
            network.mark_output(output)
            output.dtype = tensorrt_llm.str_dtype_to_trt(dtype)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16')))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.cpu().numpy(),
                'y': y_data.cpu().numpy()
            })
        res = outputs['output']

        # pytorch run
        with torch.no_grad():
            ref = m(x_data, y_data)
        ref = ref.sample

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res,
                                   rtol=1e-05,
                                   atol=1e-02)

    def test_up_sample(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((2, 4, 8, 16))

        m = torch.nn.Upsample(scale_factor=2, mode='nearest')
        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            us = Upsample2D(4)

            output = us.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_down_sample(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((2, 4, 8, 16))

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            us = Downsample2D(4)

            output = us.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})

        ref = torch.nn.functional.avg_pool2d(x_data, kernel_size=2, stride=2)

        # compare diff
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_resnet_block_float32(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((16, 32, 64, 64))
        temb = torch.randn((16, 512))
        from diffusers.models.resnet import ResnetBlock2D as rb2d
        m = rb2d(in_channels=32).eval()
        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()
        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=temb.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            rb = ResnetBlock2D(in_channels=32)
            update_resnet_block_weight(m, rb)

            output = rb.forward(x, y).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
                'y': temb.numpy()
            })

        # pytorch run
        with torch.no_grad():
            ref = m(x_data, temb)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_upblock_2d(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((16, 32, 64, 64))
        r_data_1 = torch.randn((16, 32, 64, 64))
        r_data_2 = torch.randn((16, 32, 64, 64))

        res_data = (r_data_1, r_data_2)
        torch.randn((16, 512))
        from diffusers.models.unet_2d_blocks import UpBlock2D as ub2d
        m = ub2d(in_channels=32,
                 prev_output_channel=32,
                 out_channels=96,
                 temb_channels=1).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=r_data_1.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            z = Tensor(name='z',
                       shape=r_data_2.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            rb = UpBlock2D(in_channels=32,
                           prev_output_channel=32,
                           out_channels=96,
                           temb_channels=1)
            update_upblock_2d_weight(m, rb)

            output = rb.forward(x, [y, z]).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
                'y': r_data_1.numpy(),
                'z': r_data_2.numpy()
            })

        # pytorch run
        with torch.no_grad():
            ref = m(x_data, res_data)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_downblock_2d(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((8, 32, 32, 16))

        from diffusers.models.unet_2d_blocks import DownBlock2D as db2d
        m = db2d(in_channels=32, out_channels=32, temb_channels=1).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            rb = DownBlock2D(in_channels=32, out_channels=32, temb_channels=1)
            update_downblock_2d_weight(m, rb)
            _, outputs = rb.forward(x)
            hidden_states, output_states = outputs[0].trt_tensor, outputs[
                1].trt_tensor
            hidden_states.name = 'hidden_states'
            network.mark_output(hidden_states)
            output_states.name = 'output_states'
            network.mark_output(output_states)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})
        # pytorch run
        with torch.no_grad():
            _, ref_output = m(x_data)

        np.testing.assert_allclose(ref_output[0].cpu().numpy(),
                                   outputs['hidden_states'],
                                   atol=1e-03)
        np.testing.assert_allclose(ref_output[1].cpu().numpy(),
                                   outputs['output_states'],
                                   atol=1e-03)

    def test_mid_block_2d(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((2, 32, 8, 8))

        from diffusers.models.unet_2d_blocks import UNetMidBlock2D as umb2d
        m = umb2d(in_channels=32, temb_channels=1).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            rb = UNetMidBlock2D(in_channels=32, temb_channels=1)
            update_unetmidblock_2d_weight(m, rb)

            output = rb.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})
        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_mid_block_2d_float16(self):
        # test data
        dtype = 'float16'
        in_channels = 32
        attn_num_head_channels = 16
        x_data = torch.randn(
            (2, in_channels, 8, 8),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype)).cuda()

        from diffusers.models.unet_2d_blocks import UNetMidBlock2D as umb2d
        m = umb2d(
            in_channels=in_channels,
            temb_channels=1,
            attn_num_head_channels=attn_num_head_channels).cuda().half().eval()

        tensorrt_llm.logger.set_level('error')

        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            rb = UNetMidBlock2D(in_channels=in_channels,
                                temb_channels=1,
                                attn_num_head_channels=attn_num_head_channels)
            update_unetmidblock_2d_weight(m, rb)

            output = rb.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork(
            (builder.trt_builder, net.trt_network),
            config=CreateConfig(fp16=(dtype == 'float16')))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.cpu().numpy()})
        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   rtol=1e-05,
                                   atol=1e-02)

    def test_cross_attn_up_block_2d(self):
        dtype = 'float32'
        x_data = torch.randn((16, 32, 16, 16))

        r_data_1 = torch.randn((16, 32, 16, 16))
        r_data_2 = torch.randn((16, 32, 16, 16))
        res_data = (r_data_1, r_data_2)
        from diffusers.models.unet_2d_blocks import CrossAttnUpBlock2D as caub2d
        m = caub2d(in_channels=32,
                   prev_output_channel=32,
                   out_channels=64,
                   cross_attention_dim=64,
                   temb_channels=1).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=r_data_1.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            z = Tensor(name='z',
                       shape=r_data_2.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            rb = CrossAttnUpBlock2D(in_channels=32,
                                    out_channels=64,
                                    prev_output_channel=32,
                                    cross_attention_dim=64,
                                    temb_channels=1)
            update_crossattn_upblock_2d_weight(m, rb)

            output = rb.forward(x, [y, z]).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
                'y': r_data_1.numpy(),
                'z': r_data_2.numpy()
            })

        # pytorch run
        with torch.no_grad():
            ref = m(x_data, res_data)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_cross_attn_down_block_2d(self):
        dtype = 'float32'
        cross_attention_dim = 32
        x_data = torch.randn((16, 32, 16, 16))
        temb = torch.randn((16, 1))
        from diffusers.models.unet_2d_blocks import \
            CrossAttnDownBlock2D as cadb2d
        m = cadb2d(in_channels=32,
                   out_channels=32,
                   cross_attention_dim=cross_attention_dim,
                   temb_channels=1).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=temb.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            rb = CrossAttnDownBlock2D(in_channels=32,
                                      out_channels=32,
                                      cross_attention_dim=cross_attention_dim,
                                      temb_channels=1)
            update_crossattn_downblock_2d_weight(m, rb)

            _, outputs = rb.forward(x, y)

            hidden_states, output_states = outputs[0].trt_tensor, outputs[
                1].trt_tensor
            hidden_states.name = 'hidden_states'
            network.mark_output(hidden_states)
            output_states.name = 'output_states'
            network.mark_output(output_states)
        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={
                'x': x_data.numpy(),
                'y': temb.numpy()
            })

        # pytorch run
        with torch.no_grad():
            _, ref_output = m(x_data, temb)

        np.testing.assert_allclose(ref_output[0].cpu().numpy(),
                                   outputs['hidden_states'],
                                   atol=1e-03)
        np.testing.assert_allclose(ref_output[1].cpu().numpy(),
                                   outputs['output_states'],
                                   atol=1e-03)

    def test_unet_mid_block_2d_cross_attn(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((2, 32, 8, 8))

        from diffusers.models.unet_2d_blocks import \
            UNetMidBlock2DCrossAttn as umb2dca
        m = umb2dca(in_channels=32, cross_attention_dim=32,
                    temb_channels=1).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            rb = UNetMidBlock2DCrossAttn(in_channels=32,
                                         cross_attention_dim=32,
                                         temb_channels=1)
            update_unet_mid_block_2d_weight(m, rb)

            output = rb.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})
        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_timesteps(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((128, ))

        from diffusers.models.embeddings import Timesteps as ts
        m = ts(num_channels=16, flip_sin_to_cos=True,
               downscale_freq_shift=1).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            rb = Timesteps(num_channels=16,
                           flip_sin_to_cos=True,
                           downscale_freq_shift=1)

            output = rb.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})
        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_timesteps_embedding(self):
        # test data
        dtype = 'float32'
        x_data = torch.randn((128, 16))

        from diffusers.models.embeddings import TimestepEmbedding as tse
        m = tse(16, 128).eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=x_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            tseb = TimestepEmbedding(16, 128)
            update_timestep_weight(m, tseb)

            output = tseb.forward(x).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(feed_dict={'x': x_data.numpy()})
        # pytorch run
        with torch.no_grad():
            ref = m(x_data)

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    def test_unet_2d_conditional_model(self):
        # test data
        dtype = 'float32'
        sample = torch.randn((2, 4, 8, 8))
        timestep_data = torch.randn((2, ))
        ehs_data = torch.randn((2, 32, 1280))
        from diffusers.models.unet_2d_condition import \
            UNet2DConditionModel as u2dcm
        m = u2dcm().eval()

        tensorrt_llm.logger.set_level('error')
        # construct trt network
        builder = tensorrt_llm.Builder()
        net = builder.create_network()

        with tensorrt_llm.net_guard(net):
            network = tensorrt_llm.default_trtnet()
            x = Tensor(name='x',
                       shape=sample.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            y = Tensor(name='y',
                       shape=timestep_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))
            z = Tensor(name='z',
                       shape=ehs_data.shape,
                       dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            tseb = UNet2DConditionModel()
            update_unet_2d_condition_model_weights(m, tseb)

            output = tseb.forward(x, y, z).trt_tensor
            output.name = 'output'
            network.mark_output(output)

        # trt run
        build_engine = EngineFromNetwork((builder.trt_builder, net.trt_network))
        with TrtRunner(build_engine) as runner:
            outputs = runner.infer(
                feed_dict={
                    'x': sample.numpy(),
                    'y': timestep_data.numpy(),
                    'z': ehs_data.numpy()
                })
        # pytorch run
        with torch.no_grad():
            ref = m(sample, timestep_data, ehs_data, return_dict=False)

        np.testing.assert_allclose(ref[0].cpu().numpy(),
                                   outputs['output'],
                                   atol=1e-03)

    @pytest.mark.skip(reason="Skip for TRT 8.6 release temporially.")
    def test_unet_runtime_float16(self):
        # test data
        dtype = 'float16'
        cross_attention_dim = 768
        sample_data = torch.randn(
            (2, 4, 64, 64), dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        timestep_data = torch.randn(
            (2, ), dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        ehs_data = torch.randn(
            (2, 77, cross_attention_dim),
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        from diffusers.models.unet_2d_condition import \
            UNet2DConditionModel as u2dcm

        tensorrt_llm.logger.set_level('error')
        tensorrt_llm.set_default_dtype(dtype)

        # Initialize Module
        hf_unet = u2dcm(layers_per_block=2,
                        cross_attention_dim=cross_attention_dim).half().eval()

        tensorrt_llm_unet = UNet2DConditionModel(
            cross_attention_dim=cross_attention_dim)
        load_from_hf_unet(hf_unet, tensorrt_llm_unet)

        # Module -> Network
        builder = Builder()
        network = builder.create_network()
        with net_guard(network):
            # Prepare
            network.set_named_parameters(tensorrt_llm_unet.named_parameters())

            # Forward
            sample = tensorrt_llm.Tensor(name='sample',
                                         dtype=str_dtype_to_trt(dtype),
                                         shape=[2, 4, 64, 64],
                                         dim_range=OrderedDict([
                                             ('batch_size', [2]),
                                             ('channel', [4]),
                                             ('height', [64]),
                                             ('width', [64]),
                                         ]))
            timestep = tensorrt_llm.Tensor(name='timestep',
                                           dtype=str_dtype_to_trt(dtype),
                                           shape=[2],
                                           dim_range=OrderedDict([
                                               ('batch_size', [2]),
                                           ]))
            ehs = tensorrt_llm.Tensor(name='ehs',
                                      dtype=str_dtype_to_trt(dtype),
                                      shape=[2, 77, 768],
                                      dim_range=OrderedDict([
                                          ('batch_size', [2]),
                                          ('length', [77]),
                                          ('hidden_size', [768]),
                                      ]))
            sample_out = tensorrt_llm_unet(sample, timestep, ehs)

            # Mark outputs
            sample_out.mark_output('sample_out', str_dtype_to_trt(dtype))

        model = 'stable_diffusion'
        session = None
        stream = torch.cuda.current_stream().cuda_stream

        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(name=model, )
            builder_config.trt_builder_config.builder_optimization_level = 0
            engine_buffer = builder.build_engine(network, builder_config)
            session = tensorrt_llm.runtime.Session.from_serialized_engine(
                engine_buffer)
        inputs = {
            'sample': sample_data.cuda(),
            'timestep': timestep_data.cuda(),
            'ehs': ehs_data.cuda()
        }
        for i in inputs:
            session.context.set_input_shape(i, inputs[i].shape)
        trt_outputs = {
            "sample_out":
            torch.empty(tuple(session.context.get_tensor_shape('sample_out')),
                        dtype=trt_dtype_to_torch(
                            session.engine.get_tensor_dtype('sample_out')),
                        device='cuda')
        }
        ok = session.run(inputs, trt_outputs, stream)
        assert ok, "Execution failed"
        torch.cuda.synchronize()
        res = trt_outputs['sample_out']

        hf_unet.cuda()
        with torch.no_grad():
            ref = hf_unet(sample_data.cuda(),
                          timestep_data.cuda(),
                          ehs_data.cuda(),
                          return_dict=False)
        torch.cuda.synchronize()
        np.testing.assert_allclose(ref[0].cpu().numpy(),
                                   res.cpu().numpy(),
                                   rtol=1e-01,
                                   atol=1e-01)


if __name__ == '__main__':
    unittest.main()
