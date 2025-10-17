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
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import AutoModelForCausalLM
from utils.llm_data import llm_models_root
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.models.mamba.convert import (convert_from_hf_checkpoint,
                                               convert_hf_mamba)
from tensorrt_llm.network import net_guard


class TestMamba(unittest.TestCase):

    def _gen_tensorrt_llm_mamba(self, hf_config, hf_path, hf_mamba, load_mode,
                                dtype):
        vocab_size = hf_config.vocab_size
        pad_vocab_size_multiple = hf_config.pad_vocab_size_multiple
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size %
                                                     pad_vocab_size_multiple)
        config = {
            'architecture': 'MambaForCausalLM',
            'dtype': dtype,
            'logits_dtype': 'float32',
            'hidden_size': hf_config.hidden_size,
            'num_hidden_layers': hf_config.num_hidden_layers,
            'layer_types': ['recurrent'],
            'vocab_size': vocab_size,
            'rms_norm': hf_config.rms_norm,
            'residual_in_fp32': hf_config.residual_in_fp32,
            'pad_vocab_size_multiple': hf_config.pad_vocab_size_multiple,
            'hidden_act': 'silu',
            'num_attention_heads': 1,
            'rnn_hidden_size': hf_config.intermediate_size,
            'rnn_conv_dim_size': hf_config.intermediate_size,
            'state_size': hf_config.state_size,
            'conv_kernel': hf_config.conv_kernel,
            'use_bias': hf_config.use_bias,
            'mamba_version': 'Mamba1',
            'mapping': {
                'world_size': 1,
                'tp_size': 1,
                'pp_size': 1
            },
        }
        config = tensorrt_llm.models.PretrainedConfig.from_dict(config)
        if load_mode == 'from_checkpoint':
            weights = convert_from_hf_checkpoint(mamba_config=config,
                                                 model_dir=hf_path)
        else:
            weights = convert_hf_mamba(hf_mamba, dtype=dtype)
        tensorrt_llm_mamba = tensorrt_llm.models.MambaForCausalLM(config)
        tensorrt_llm_mamba.load(weights)
        return tensorrt_llm_mamba

    def _gen_tensorrt_llm_network(self, network, hf_config, hf_path, hf_mamba,
                                  load_mode, batch_size, input_len, output_len,
                                  dtype):
        tensorrt_llm_mamba = self._gen_tensorrt_llm_mamba(
            hf_config, hf_path, hf_mamba, load_mode, dtype)
        with net_guard(network):
            network.set_named_parameters(tensorrt_llm_mamba.named_parameters())
            inputs = tensorrt_llm_mamba.prepare_inputs(
                batch_size,
                input_len,
                input_len + output_len,
                max_num_tokens=batch_size * input_len,
                use_cache=False)
            # Prepare
            tensorrt_llm_mamba(**inputs)
        return network

    def _gen_tensorrt_llm_engine(self, model_name, gemm_plugin,
                                 mamba_conv1d_plugin, hf_config, hf_path,
                                 hf_mamba, load_mode, batch_size, input_len,
                                 output_len, dtype, remove_padding):
        builder = Builder()
        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name=model_name,
                precision=dtype,
                timing_cache='model.cache',
            )
            network = builder.create_network()
            network.plugin_config.to_legacy_setting()
            network.plugin_config.remove_input_padding = remove_padding
            network.plugin_config.paged_state = False
            if gemm_plugin:
                network.plugin_config.gemm_plugin = dtype
            if mamba_conv1d_plugin:
                network.plugin_config.mamba_conv1d_plugin = dtype
            else:
                network.plugin_config.mamba_conv1d_plugin = None

            self._gen_tensorrt_llm_network(network, hf_config, hf_path,
                                           hf_mamba, load_mode, batch_size,
                                           input_len, output_len, dtype)

            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(self, log_level, model_name, gemm_plugin,
                                  mamba_conv1d_plugin, hf_config, hf_path,
                                  hf_mamba, load_mode, batch_size, input_len,
                                  output_len, dtype, remove_padding):
        tensorrt_llm.logger.set_level(log_level)
        mapping = tensorrt_llm.Mapping()
        engine_buffer = self._gen_tensorrt_llm_engine(
            model_name, gemm_plugin, mamba_conv1d_plugin, hf_config, hf_path,
            hf_mamba, load_mode, batch_size, input_len, output_len, dtype,
            remove_padding)
        runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)
        return runtime, engine_buffer

    @parameterized.expand([
        (True, True, 'float16', False),
        (False, True, 'float16', False),
        (True, True, 'bfloat16', False),
        (False, True, 'bfloat16', False),
        (True, False, 'float16', False),
        (False, False, 'float16', False),
        (True, False, 'bfloat16', False),
        (False, False, 'bfloat16', False),
        (True, True, 'float16', True),
        (False, True, 'float16', True),
        (True, True, 'bfloat16', True),
        (False, True, 'bfloat16', True),
    ],
                          name_func=unittest_name_func)
    def test_mamba(self, gemm_plugin, mamba_conv1d_plugin, dtype,
                   remove_padding):
        from transformers import MambaConfig

        RANDOM_SEEDS = [1, 4, 5, 8]
        seed_idx = random.randint(0, len(RANDOM_SEEDS) - 1)
        torch.manual_seed(RANDOM_SEEDS[seed_idx])

        model_name = 'mamba'
        log_level = 'error'
        batch_size = 4
        input_len = 16
        output_len = 2
        load_mode = 'from_model'
        hf_path = ''
        hidden_size = 128
        hf_config = MambaConfig(hidden_size=hidden_size,
                                num_hidden_layers=2,
                                pad_vocab_size_multiple=8,
                                vocab_size=128,
                                rms_norm=True,
                                dtype=str_dtype_to_torch(dtype))

        # get hf mamba
        hf_mamba = AutoModelForCausalLM.from_config(
            hf_config, dtype=str_dtype_to_torch(dtype)).cuda().eval()

        # inputs
        if remove_padding:
            ctx_last_token_ids = torch.randint(1,
                                               input_len + 1, (batch_size, ),
                                               dtype=torch.int32)
            ctx_last_token_ids = torch.cumsum(ctx_last_token_ids,
                                              dim=0,
                                              dtype=torch.int32).to('cuda')
            total_num_tokens = ctx_last_token_ids[batch_size - 1]
        else:
            ctx_last_token_ids = input_len * torch.ones(
                (batch_size, ), dtype=torch.int32, device='cuda')
            total_num_tokens = batch_size * input_len
        ctx_ids = torch.randint(100, (total_num_tokens, )).int().cuda()
        step1_id = torch.randint(100, (batch_size, )).int().cuda()
        if not remove_padding:
            ctx_ids = ctx_ids.view(-1, input_len)
            step1_id = step1_id.view(-1, 1)

        # get ref outputs
        with torch.no_grad():
            if remove_padding:
                ref = torch.empty(batch_size, hidden_size)
                gen_ref = torch.empty(batch_size, hidden_size)
                for i in range(batch_size):
                    # ctx
                    start_id = 0 if i == 0 else ctx_last_token_ids[i - 1]
                    end_id = ctx_last_token_ids[i]
                    part_ctx_ids = torch.unsqueeze(ctx_ids[start_id:end_id],
                                                   dim=0)
                    part_hf_outputs = hf_mamba(part_ctx_ids)
                    torch.cuda.synchronize()
                    ref[i][:] = part_hf_outputs.logits[0, -1, :]
                    part_cache_params = part_hf_outputs.cache_params
                    # gen
                    part_step1_id = step1_id[i].view(1, 1)
                    part_hf_gen_outputs = hf_mamba.forward(
                        part_step1_id,
                        cache_params=part_cache_params,
                        cache_position=torch.arange(
                            hf_config.conv_kernel - 1,
                            hf_config.conv_kernel,
                            device=part_step1_id.device))
                    torch.cuda.synchronize()
                    gen_ref[i][:] = part_hf_gen_outputs.logits[0, -1, :]
            else:
                # ctx
                hf_outputs = hf_mamba.forward(ctx_ids)
                ref = hf_outputs.logits[:, -1, :]
                torch.cuda.synchronize()
                cache_params = hf_outputs.cache_params
                # gen
                hf_outputs = hf_mamba.forward(step1_id,
                                              cache_params=cache_params,
                                              use_cache=True,
                                              cache_position=torch.arange(
                                                  hf_config.conv_kernel - 1,
                                                  hf_config.conv_kernel,
                                                  device=step1_id.device))
                gen_ref = hf_outputs.logits[:, -1, :]

        # get tensorrt llm mamba runtime
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, model_name, gemm_plugin, mamba_conv1d_plugin, hf_config,
            hf_path, hf_mamba, load_mode, batch_size, input_len, output_len,
            dtype, remove_padding)

        # prepare buffers
        intermediate_size = hf_mamba.backbone.layers[0].mixer.intermediate_size
        conv_kernel = hf_mamba.backbone.layers[0].mixer.conv_kernel_size
        state_size = hf_mamba.backbone.layers[0].mixer.ssm_state_size
        if mamba_conv1d_plugin:
            conv_state_shape = (
                batch_size,
                conv_kernel - 1,
                intermediate_size,
            )
        else:
            conv_state_shape = (
                batch_size,
                intermediate_size,
                conv_kernel - 1,
            )

        rnn_state_shape = (
            batch_size,
            state_size,
            intermediate_size,
        )
        present_conv_states = []
        present_conv_states_1 = []
        present_rnn_states = []
        for _ in range(hf_config.num_hidden_layers):
            present_conv_states.append(
                torch.zeros(conv_state_shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'))
            present_conv_states_1.append(
                torch.empty(conv_state_shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'))
            present_rnn_states.append(
                torch.empty(rnn_state_shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'))

        # compare context
        if remove_padding:
            host_ctx_lengths = ctx_last_token_ids.detach().clone().cpu()
        else:
            host_ctx_lengths = input_len * torch.ones(
                (batch_size, ), dtype=torch.int32)
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32)
        ctx_buffer = {
            'input_ids': ctx_ids,
            'last_token_ids': ctx_last_token_ids,
            'host_request_types': ctx_host_request_types,
            'host_context_lengths': host_ctx_lengths,
        }
        for idx in range(hf_config.num_hidden_layers):
            ctx_buffer[f'past_conv_state_{idx}'] = present_conv_states[idx]
            ctx_buffer[f'present_conv_state_{idx}'] = present_conv_states_1[idx]
            ctx_buffer[f'past_rnn_state_{idx}'] = present_rnn_states[idx]
            ctx_buffer[f'present_rnn_state_{idx}'] = present_rnn_states[idx]
        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=0.1)

        # compare generation
        gen_last_token_ids = torch.ones((batch_size, ),
                                        dtype=torch.int32,
                                        device='cuda')
        if remove_padding:
            gen_last_token_ids = torch.cumsum(gen_last_token_ids,
                                              dim=0,
                                              dtype=torch.int32).to('cuda')
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32)
        step1_buffer = {
            'input_ids': step1_id,
            'last_token_ids': gen_last_token_ids,
            'host_request_types': gen_host_request_types,
            'host_context_lengths': host_ctx_lengths,
        }
        for idx in range(hf_config.num_hidden_layers):
            step1_buffer[f'past_conv_state_{idx}'] = present_conv_states_1[idx]
            step1_buffer[f'present_conv_state_{idx}'] = present_conv_states[idx]
            step1_buffer[f'past_rnn_state_{idx}'] = present_rnn_states[idx]
            step1_buffer[f'present_rnn_state_{idx}'] = present_rnn_states[idx]
        step1_shape = {k: v.shape for k, v in step1_buffer.items()}

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        np.testing.assert_allclose(gen_ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=0.1)

    @parameterized.expand([
        ('mamba-130m-hf', 'from_checkpoint'),
        ('mamba-130m-hf', 'from_model'),
    ],
                          name_func=unittest_name_func)
    def test_loaders(self, path, load_mode):
        from transformers import MambaConfig
        model_root = llm_models_root()
        if model_root is None:
            pytest.skip('Skipping since real weights are unavailable.')
        hf_path = Path(model_root, 'mamba', path)
        if not hf_path.exists():
            pytest.skip(f'Skipping since the path {hf_path} does not exist.')
        dtype = 'float16'

        if path == 'mamba-130m-hf' and load_mode == 'from_checkpoint':
            pytest.skip(
                f'Skipping since it is a known issue. Will be fixed in the near future.'
            )

        # get hf mamba
        hf_mamba = AutoModelForCausalLM.from_pretrained(
            hf_path, device_map='cpu', dtype=str_dtype_to_torch(dtype))

        # get tensort llm mamba
        hf_config = MambaConfig.from_pretrained(hf_path)
        tensorrt_llm_mamba = self._gen_tensorrt_llm_mamba(
            hf_config, hf_path, hf_mamba, load_mode, dtype)

        def has_bias(torch_layer):
            return hasattr(torch_layer, 'bias') and torch_layer.bias is not None

        # token embedding
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.vocab_embedding.weight.raw_value,
            hf_mamba.backbone.embeddings.weight.cpu().detach(),
            atol=1e-3)
        # output
        np.testing.assert_allclose(tensorrt_llm_mamba.lm_head.weight.raw_value,
                                   hf_mamba.lm_head.weight.cpu().detach(),
                                   atol=1e-3)
        # norm
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.ln_f.weight.raw_value,
            hf_mamba.backbone.norm_f.weight.cpu().detach(),
            atol=1e-3)
        if has_bias(hf_mamba.backbone.norm_f):
            np.testing.assert_allclose(
                tensorrt_llm_mamba.backbone.ln_f.bias.raw_value,
                hf_mamba.backbone.norm_f.bias.cpu().detach(),
                atol=1e-3)
        # Checking all of the layers takes too much time, just check one random layer
        l = np.random.randint(0, tensorrt_llm_mamba.config.num_hidden_layers)
        print(f"Checking Layer-{l} weights ...", flush=True)
        # layer{l}.input_layernorm
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].input_layernorm.weight.
            raw_value,
            hf_mamba.backbone.layers[l].norm.weight.cpu().detach(),
            atol=1e-3)
        if has_bias(hf_mamba.backbone.layers[l]):
            np.testing.assert_allclose(
                tensorrt_llm_mamba.backbone.layers[l].input_layernorm.bias.
                raw_value,
                hf_mamba.backbone.layers[l].norm.bias.cpu().detach(),
                atol=1e-3)
        # layer{l}.ssm.A
        A_hf = -torch.exp(hf_mamba.backbone.layers[l].mixer.A_log.float())
        A_hf_permute = A_hf.cpu().detach().permute([1, 0]).contiguous()
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.A.raw_value,
            A_hf_permute,
            atol=1e-3)
        # layer{l}.ssm.D
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.D.raw_value,
            hf_mamba.backbone.layers[l].mixer.D.float().cpu().detach(),
            atol=1e-3)
        # layer{l}.ssm.dt_bias
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.dt_bias.raw_value,
            hf_mamba.backbone.layers[l].mixer.dt_proj.bias.cpu().to(
                torch.float32).detach(),
            atol=1e-3)
        # layer{l}.ssm.in_proj
        d_inner = tensorrt_llm_mamba.backbone.layers[
            l].ssm.in_proj_x.weight.raw_value.shape[0]
        in_proj_x_hf = hf_mamba.backbone.layers[l].mixer.in_proj.weight[
            0:d_inner,
        ]
        in_proj_z_hf = hf_mamba.backbone.layers[l].mixer.in_proj.weight[
            d_inner:,
        ]
        np.testing.assert_allclose(tensorrt_llm_mamba.backbone.layers[l].ssm.
                                   in_proj_x.weight.raw_value,
                                   in_proj_x_hf.cpu().detach(),
                                   atol=1e-3)
        np.testing.assert_allclose(tensorrt_llm_mamba.backbone.layers[l].ssm.
                                   in_proj_z.weight.raw_value,
                                   in_proj_z_hf.cpu().detach(),
                                   atol=1e-3)
        if has_bias(hf_mamba.backbone.layers[l].mixer.in_proj):
            in_proj_bias_x_hf = hf_mamba.backbone.layers[l].mixer.in_proj.bias[
                0:d_inner]
            in_proj_bias_z_hf = hf_mamba.backbone.layers[l].mixer.in_proj.bias[
                d_inner:]
            np.testing.assert_allclose(tensorrt_llm_mamba.backbone.layers[l].
                                       ssm.in_proj_x.bias.raw_value,
                                       in_proj_bias_x_hf.cpu().detach(),
                                       atol=1e-3)
            np.testing.assert_allclose(tensorrt_llm_mamba.backbone.layers[l].
                                       ssm.in_proj_z.bias.raw_value,
                                       in_proj_bias_z_hf.cpu().detach(),
                                       atol=1e-3)
        # layer{l}.ssm.conv1d
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.conv1d.weight.raw_value,
            hf_mamba.backbone.layers[l].mixer.conv1d.weight.unsqueeze(
                3).cpu().detach(),
            atol=1e-3)
        if has_bias(hf_mamba.backbone.layers[l].mixer.conv1d):
            np.testing.assert_allclose(
                tensorrt_llm_mamba.backbone.layers[l].ssm.conv1d.bias.raw_value,
                hf_mamba.backbone.layers[l].mixer.conv1d.bias.cpu().detach(),
                atol=1e-3)
        # layer{l}.ssm.x_proj
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.x_proj.weight.raw_value,
            hf_mamba.backbone.layers[l].mixer.x_proj.weight.cpu().detach(),
            atol=1e-3)
        # layer{l}.ssm.dt_proj
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.dt_proj.weight.raw_value,
            hf_mamba.backbone.layers[l].mixer.dt_proj.weight.cpu().detach(),
            atol=1e-3)
        # layer{l}.ssm.out_proj
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.out_proj.weight.raw_value,
            hf_mamba.backbone.layers[l].mixer.out_proj.weight.cpu().detach(),
            atol=1e-3)
        if has_bias(hf_mamba.backbone.layers[l].mixer.out_proj):
            np.testing.assert_allclose(
                tensorrt_llm_mamba.backbone.layers[l].ssm.out_proj.bias.
                raw_value,
                hf_mamba.backbone.layers[l].mixer.out_proj.bias.cpu().detach(),
                atol=1e-3)
        return
