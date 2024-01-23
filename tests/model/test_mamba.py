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
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.utils.hf import load_config_hf
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.layers.ssm import MambaParameters
from tensorrt_llm.network import net_guard

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.mamba.convert_checkpoint import (convert_from_hf_checkpoint,
                                               convert_hf_mamba)
from tests.llm_data import llm_models_root
from tests.utils.util import getSMVersion


class TestMamba(unittest.TestCase):

    def _gen_tensorrt_llm_mamba(self, hf_config, hf_path, hf_mamba, load_mode,
                                dtype):
        config = {
            'architecture': 'MambaLMHeadModel',
            'dtype': dtype,
            'logits_dtype': 'float32',
            'hidden_size': hf_config.d_model,
            'num_hidden_layers': hf_config.n_layer,
            'vocab_size': hf_config.vocab_size,
            'ssm_cfg': MambaParameters(**hf_config.ssm_cfg).__dict__,
            'rms_norm': hf_config.rms_norm,
            'residual_in_fp32': hf_config.residual_in_fp32,
            'pad_vocab_size_multiple': hf_config.pad_vocab_size_multiple,
            'hidden_act': 'silu',
            'num_attention_heads': 1,
        }
        config = tensorrt_llm.models.PretrainedConfig.from_dict(config)
        if load_mode == 'from_checkpoint':
            weights = convert_from_hf_checkpoint(model_dir=hf_path, dtype=dtype)
        else:
            weights = convert_hf_mamba(hf_mamba, rank=0, dtype=dtype)

        tensorrt_llm_mamba = tensorrt_llm.models.MambaLMHeadModel(config)
        tensorrt_llm_mamba.load(weights)
        return tensorrt_llm_mamba

    def _gen_tensorrt_llm_network(self, network, hf_config, hf_path, hf_mamba,
                                  load_mode, batch_size, input_len, output_len,
                                  dtype):
        tensorrt_llm_mamba = self._gen_tensorrt_llm_mamba(
            hf_config, hf_path, hf_mamba, load_mode, dtype)
        with net_guard(network):
            network.set_named_parameters(tensorrt_llm_mamba.named_parameters())
            inputs = tensorrt_llm_mamba.prepare_inputs(batch_size,
                                                       input_len,
                                                       output_len,
                                                       use_cache=False)
            # Prepare
            tensorrt_llm_mamba(**inputs)
        return network

    def _gen_tensorrt_llm_engine(self, model_name, gemm_plugin, hf_config,
                                 hf_path, hf_mamba, load_mode, batch_size,
                                 input_len, output_len, dtype):
        builder = Builder()
        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name=model_name,
                precision=dtype,
                timing_cache='model.cache',
            )
            network = builder.create_network()
            network.plugin_config.set_selective_scan_plugin(dtype)
            if gemm_plugin:
                network.plugin_config.set_gemm_plugin(dtype)

            self._gen_tensorrt_llm_network(network, hf_config, hf_path,
                                           hf_mamba, load_mode, batch_size,
                                           input_len, output_len, dtype)

            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(self, log_level, model_name, gemm_plugin,
                                  hf_config, hf_path, hf_mamba, load_mode,
                                  batch_size, input_len, output_len, dtype):
        tensorrt_llm.logger.set_level(log_level)
        mapping = tensorrt_llm.Mapping()
        engine_buffer = self._gen_tensorrt_llm_engine(model_name, gemm_plugin,
                                                      hf_config, hf_path,
                                                      hf_mamba, load_mode,
                                                      batch_size, input_len,
                                                      output_len, dtype)
        runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)
        return runtime, engine_buffer

    @parameterized.expand([
        (True, 'float16'),
        (False, 'float16'),
        (True, 'bfloat16'),
        (False, 'bfloat16'),
    ])
    def test_mamba(self, gemm_plugin, dtype):
        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if dtype == 'bfloat16':
                pytest.skip(
                    "bfloat16 is not supported in pre-ampere architecture")

        RANDOM_SEEDS = [1, 4, 5, 8]
        seed_idx = random.randint(0, len(RANDOM_SEEDS) - 1)
        torch.manual_seed(RANDOM_SEEDS[seed_idx])

        model_name = 'mamba'
        log_level = 'error'
        batch_size = 4
        input_len = 4
        output_len = 2
        load_mode = 'from_model'
        hf_path = ''
        hf_config = MambaConfig(d_model=128, n_layer=2, vocab_size=128)

        # get hf mamba
        hf_mamba = MambaLMHeadModel(hf_config,
                                    device='cuda',
                                    dtype=str_dtype_to_torch(dtype))

        # get tensorrt llm mamba rumtime
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, model_name, gemm_plugin, hf_config, hf_path, hf_mamba,
            load_mode, batch_size, input_len, output_len, dtype)

        # prepare buffers
        mamba_d_inner = hf_mamba.backbone.layers[0].mixer.d_inner
        mamba_d_conv = hf_mamba.backbone.layers[0].mixer.d_conv
        mamba_d_state = hf_mamba.backbone.layers[0].mixer.d_state
        conv_state_shape = (
            batch_size,
            mamba_d_inner,
            mamba_d_conv - 1,
        )

        ssm_state_shape = (
            batch_size,
            mamba_d_inner,
            mamba_d_state,
        )
        present_conv_states = []
        present_conv_states_1 = []
        present_ssm_states = []
        for _ in range(hf_config.n_layer):
            present_conv_states.append(
                torch.zeros(conv_state_shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'))
            present_conv_states_1.append(
                torch.empty(conv_state_shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'))
            present_ssm_states.append(
                torch.empty(ssm_state_shape, dtype=torch.float32,
                            device='cuda'))

        # compare context
        ctx_ids = torch.randint(100, (batch_size, input_len)).int().cuda()
        ctx_last_token_ids = input_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32)
        infer_params = InferenceParams(max_seqlen=input_len + output_len,
                                       max_batch_size=batch_size)

        with torch.no_grad():
            hf_outputs = hf_mamba.forward(ctx_ids,
                                          inference_params=infer_params)
            infer_params.seqlen_offset += ctx_ids.shape[1]
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        ctx_buffer = {
            'input_ids': ctx_ids,
            'last_token_ids': ctx_last_token_ids,
            'host_request_types': ctx_host_request_types,
        }
        for idx in range(hf_config.n_layer):
            ctx_buffer[f'past_conv_state_{idx}'] = present_conv_states[idx]
            ctx_buffer[f'present_conv_state_{idx}'] = present_conv_states_1[idx]
            ctx_buffer[f'past_ssm_state_{idx}'] = present_ssm_states[idx]
            ctx_buffer[f'present_ssm_state_{idx}'] = present_ssm_states[idx]
        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=1e-2)

        # compare generation
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_last_token_ids = torch.zeros((batch_size),
                                         dtype=torch.int32,
                                         device='cuda')
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32)
        with torch.no_grad():
            hf_outputs = hf_mamba.forward(step1_id,
                                          inference_params=infer_params)
            infer_params.seqlen_offset += step1_id.shape[1]
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]
        step1_buffer = {
            'input_ids': step1_id,
            'last_token_ids': gen_last_token_ids,
            'host_request_types': gen_host_request_types,
        }
        for idx in range(hf_config.n_layer):
            step1_buffer[f'past_conv_state_{idx}'] = present_conv_states_1[idx]
            step1_buffer[f'present_conv_state_{idx}'] = present_conv_states[idx]
            step1_buffer[f'past_ssm_state_{idx}'] = present_ssm_states[idx]
            step1_buffer[f'present_ssm_state_{idx}'] = present_ssm_states[idx]
        step1_shape = {k: v.shape for k, v in step1_buffer.items()}

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=1e-2)

    @parameterized.expand([
        ('mamba-130m', 'from_checkpoint'),
        ('mamba-130m', 'from_model'),
    ])
    def test_loaders(self, path, load_mode):
        model_root = llm_models_root()
        if model_root is None:
            pytest.skip('Skipping since real weights are unavailable.')
        hf_path = Path(model_root, path)
        if not hf_path.exists():
            pytest.skip(f'Skipping since the path {hf_path} does not exist.')
        dtype = 'float16'

        # get hf mamba
        hf_mamba = MambaLMHeadModel.from_pretrained(
            hf_path, device='cpu', dtype=str_dtype_to_torch(dtype))

        # get tensort llm mamba
        config_data = load_config_hf(hf_path)
        hf_config = MambaConfig(**config_data)
        tensorrt_llm_mamba = self._gen_tensorrt_llm_mamba(
            hf_config, hf_path, hf_mamba, load_mode, dtype)

        def has_bias(torch_layer):
            return hasattr(torch_layer, 'bias') and torch_layer.bias is not None

        # token embedding
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.vocab_embedding.weight.raw_value,
            hf_mamba.backbone.embedding.weight.cpu().detach(),
            atol=1e-3)
        # output
        np.testing.assert_allclose(tensorrt_llm_mamba.lm_head.weight.raw_value,
                                   hf_mamba.lm_head.weight.cpu().detach(),
                                   atol=1e-3)
        # norm
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.norm_f.weight.raw_value,
            hf_mamba.backbone.norm_f.weight.cpu().detach(),
            atol=1e-3)
        if has_bias(hf_mamba.backbone.norm_f):
            np.testing.assert_allclose(
                tensorrt_llm_mamba.backbone.norm_f.bias.raw_value,
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
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.A.raw_value,
            A_hf.cpu().detach(),
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
        np.testing.assert_allclose(
            tensorrt_llm_mamba.backbone.layers[l].ssm.in_proj.weight.raw_value,
            hf_mamba.backbone.layers[l].mixer.in_proj.weight.cpu().detach(),
            atol=1e-3)
        if has_bias(hf_mamba.backbone.layers[l].mixer.in_proj):
            np.testing.assert_allclose(
                tensorrt_llm_mamba.backbone.layers[l].ssm.in_proj.bias.
                raw_value,
                hf_mamba.backbone.layers[l].mixer.in_proj.bias.cpu().detach(),
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
