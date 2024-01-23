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
import sys
import tempfile
import unittest
from itertools import product

import numpy as np
import pytest

# isort: off
import torch
# isort: on
from parameterized import parameterized
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.gptneox.convert_checkpoint import convert_hf_gptneox

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


def compare_max_abs_error(ref, res, str):
    # calculate max abs error
    compare_HF = ref.cpu().numpy().flatten()
    compare_TRT_LLM = res.cpu().numpy().flatten()
    max_abs_error = np.max(abs(compare_TRT_LLM - compare_HF))
    print(str, "max abs error = ", max_abs_error)


class TestGPTNeoX(unittest.TestCase):

    def _gen_hf_gpt_neox(self, hidden_act, n_layer, max_length, dtype):
        gpt_config = GPTNeoXConfig(hidden_act=hidden_act,
                                   num_hidden_layers=n_layer,
                                   max_length=max_length,
                                   torch_dtype=dtype,
                                   hidden_size=4096,
                                   intermediate_size=4096 * 4,
                                   num_attention_heads=64,
                                   rotary_pct=0.25)

        hf_gpt = GPTNeoXForCausalLM(gpt_config).cuda().to(
            tensorrt_llm._utils.str_dtype_to_torch(dtype)).eval()
        return gpt_config, hf_gpt

    def _gen_tensorrt_llm_network(self, network, builder, hf_gpt, gpt_config,
                                  batch_size, beam_width, input_len, output_len,
                                  fp16, gpt_attention_plugin, rank,
                                  tensor_parallel,
                                  apply_query_key_layer_scaling):
        num_layers = gpt_config.num_hidden_layers
        num_heads = gpt_config.num_attention_heads
        hidden_size = gpt_config.hidden_size
        vocab_size = gpt_config.vocab_size
        hidden_act = gpt_config.hidden_act
        max_position_embeddings = gpt_config.max_position_embeddings

        list(range(tensor_parallel))

        dtype = 'float16' if fp16 else 'float32'
        config = {
            'architecture': 'GPTNeoXForCausalLM',
            'dtype': dtype,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "position_embedding_type": "learned_absolute",
            "max_position_embeddings": max_position_embeddings,
            "rotary_emb_base": 10000,
            "rotary_pct": gpt_config.rotary_pct,
            "hidden_act": hidden_act,
            "quantization": {
                "use_weight_only": False,
                "weight_only_precision": "int8"
            },
            "mapping": {
                "world_size": tensor_parallel,
                "tp_size": tensor_parallel
            },
            "use_parallel_embedding": False,
            "embedding_sharding_dim": 0,
            "share_embedding_table": False
        }
        config = tensorrt_llm.models.PretrainedConfig.from_dict(config)
        weights = convert_hf_gptneox(hf_gpt,
                                     mapping=Mapping(rank=rank,
                                                     world_size=tensor_parallel,
                                                     tp_size=tensor_parallel),
                                     dtype=dtype)
        tensorrt_llm_gptneox = tensorrt_llm.models.GPTNeoXForCausalLM(config)
        tensorrt_llm_gptneox.load(weights)

        with net_guard(network):
            network.set_named_parameters(
                tensorrt_llm_gptneox.named_parameters())
            inputs = tensorrt_llm_gptneox.prepare_inputs(
                max_batch_size=batch_size,
                max_input_len=input_len,
                max_seq_len=input_len + output_len,
                use_cache=True,
                max_beam_width=beam_width)
            # Prepare
            tensorrt_llm_gptneox(**inputs)

        return network

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  gpt_config,
                                  hf_gpt,
                                  model,
                                  use_attention_plugin,
                                  batch_size,
                                  beam_width,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  use_ln_gemm_plugin,
                                  apply_query_key_layer_scaling,
                                  context_fmha_flag=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False):
        tensorrt_llm.logger.set_level('error')
        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)

        runtime = None
        builder = Builder()
        fp16 = (dtype == 'float16')

        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name='gptneox',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
                strongly_typed=fp16,
            )
            network = builder.create_network()
            if use_attention_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
            if use_ln_gemm_plugin:
                network.plugin_config.set_gemm_plugin(dtype)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network, builder, hf_gpt, gpt_config,
                                           batch_size, beam_width, input_len,
                                           output_len, fp16,
                                           use_attention_plugin, rank,
                                           world_size,
                                           apply_query_key_layer_scaling)

            engine_buffer = builder.build_engine(network, builder_config)
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)

            ok = builder.save_timing_cache(builder_config, 'model.cache')
            assert ok, "Failed to save timing cache."

        return runtime, engine_buffer

    def load_test_cases():
        test_cases = product([
            ContextFMHAType.disabled, ContextFMHAType.enabled,
            ContextFMHAType.enabled_with_fp32_acc
        ], [False, True])
        return test_cases

    @parameterized.expand(load_test_cases)
    def test_gptneox_plugin(self, context_fmha_flag,
                            enable_remove_input_padding):

        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if context_fmha_flag == ContextFMHAType.enabled_with_fp32_acc:
                pytest.skip(
                    "ContextFMHAType with fp32 acc is not supported in pre-ampere architecture"
                )

        torch.random.manual_seed(0)
        use_refit = False
        apply_query_key_layer_scaling = False
        model = 'gptneox'

        log_level = 'error'
        dtype = 'float16'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 6
        max_length = 128
        batch_size = 1
        beam_width = 1
        seq_len = 128
        total_seq_len = max_length + seq_len
        use_attention_plugin = True
        use_ln_gemm_plugin = True

        gpt_config, hf_gpt = self._gen_hf_gpt_neox(hidden_act, n_layer,
                                                   seq_len + max_length, dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
            use_attention_plugin, batch_size, beam_width, seq_len, max_length,
            use_refit, use_ln_gemm_plugin, apply_query_key_layer_scaling,
            context_fmha_flag, enable_remove_input_padding)
        key_value_cache_buffers = []
        head_size = gpt_config.hidden_size // gpt_config.num_attention_heads
        for i in range(gpt_config.num_hidden_layers):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    gpt_config.num_attention_heads,
                    total_seq_len,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))

        # compare context
        step = 0
        ctx_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(ctx_ids, use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        ctx_context_lengths = seq_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32)
        ctx_position_ids = torch.tensor(range(seq_len),
                                        dtype=torch.int32).reshape([
                                            1, seq_len
                                        ]).expand([batch_size, seq_len]).cuda()
        ctx_last_token_ids = ctx_context_lengths.clone()

        # We need sequence_lengths start as context_lengths for step 0,
        # and it will be added one after each step.
        sequence_length_buffer = ctx_context_lengths.detach().clone()

        if enable_remove_input_padding:
            ctx_ids = ctx_ids.view([batch_size * seq_len])
            ctx_position_ids = ctx_position_ids.view([batch_size * seq_len])
            ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                total_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                total_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers
        ctx_buffer = {
            'input_ids': ctx_ids,
            'context_lengths': ctx_context_lengths,
            'host_request_types': ctx_host_request_types,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'cache_indirection': cache_indirections[0],
        }
        if enable_remove_input_padding:
            ctx_buffer['host_context_lengths'] = ctx_context_lengths.cpu()
        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}
        shape = (batch_size, 2, gpt_config.num_attention_heads, total_seq_len,
                 gpt_config.hidden_size // gpt_config.num_attention_heads)
        for i in range(gpt_config.num_hidden_layers):
            ctx_shape[f'past_key_value_{i}'] = shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'host_max_attention_window_size_{i}'] = torch.tensor(
                [total_seq_len], dtype=torch.int32)
            ctx_shape[f'host_max_attention_window_size_{i}'] = (1, )
        ctx_buffer['sequence_length'] = sequence_length_buffer
        sequence_length_buffer = torch.add(sequence_length_buffer, step)
        ctx_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
        ctx_buffer['host_past_key_value_lengths'] = ctx_context_lengths.cpu()
        ctx_shape['host_past_key_value_lengths'] = ctx_buffer[
            'host_past_key_value_lengths'].shape
        ctx_buffer['host_sink_token_length'] = torch.tensor([0],
                                                            dtype=torch.int32)
        ctx_shape['host_sink_token_length'] = (1, )

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)

        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-1)

        compare_max_abs_error(ref, res, "context logits")

        v_inner = 16 // (2 if dtype == 'float16' else 4)
        for i in range(gpt_config.num_hidden_layers):
            res_present_key_value = ctx_buffer[f'present_key_value_{i}']

            past_key_value_tensor = res_present_key_value.permute(1, 0, 2, 3, 4)
            key, value = past_key_value_tensor.chunk(2)

            # TRT-LLM has the same cache layout for key and value:
            # [bs, n_head, max_seq_len, head_size]
            key = key.reshape(batch_size, gpt_config.num_attention_heads,
                              total_seq_len, head_size)

            value = value.reshape(batch_size, gpt_config.num_attention_heads,
                                  total_seq_len, head_size)

            ref_present_key, ref_present_value = hf_outputs.past_key_values[i]

            np.testing.assert_allclose(ref_present_key.cpu().numpy(),
                                       key[:, :, :seq_len, :].cpu().numpy(),
                                       atol=1e-1)

            compare_max_abs_error(ref_present_key, key[:, :, :seq_len, :],
                                  "ref_present_key")

            np.testing.assert_allclose(ref_present_value.cpu().numpy(),
                                       value[:, :, :seq_len, :].cpu().numpy(),
                                       atol=1e-1)

            compare_max_abs_error(ref_present_value, value[:, :, :seq_len, :],
                                  "ref_present_value")

        # compare generation
        step = 1
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_position_ids = torch.ones_like(step1_id).int().cuda() * seq_len
        gen_context_lengths = ctx_context_lengths.clone()
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32)
        gen_last_token_ids = torch.zeros_like(gen_context_lengths).int().cuda()

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(
                step1_id,
                past_key_values=hf_outputs.past_key_values,
                position_ids=gen_position_ids,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            step1_id = step1_id.view([batch_size])
            gen_position_ids = gen_position_ids.view([batch_size])
            gen_last_token_ids = torch.ones_like(
                gen_context_lengths).int().cuda()
            gen_last_token_ids = torch.cumsum(gen_last_token_ids, dim=0).int()

        step1_buffer = {
            'input_ids': step1_id,
            'context_lengths': gen_context_lengths,
            'host_request_types': gen_host_request_types,
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'cache_indirection': cache_indirections[1],
        }
        if enable_remove_input_padding:
            step1_buffer['host_context_lengths'] = gen_context_lengths.cpu()
        step1_shape = {k: v.shape for k, v in step1_buffer.items()}
        for i in range(gpt_config.num_hidden_layers):
            step1_shape[f'past_key_value_{i}'] = shape
            step1_shape[f'host_max_attention_window_size_{i}'] = (1, )
        step1_shape['sequence_length'] = (batch_size, )
        step1_shape['host_past_key_value_lengths'] = (batch_size, )
        step1_shape['host_sink_token_length'] = (1, )
        for i in range(gpt_config.num_hidden_layers):
            step1_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            step1_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
            step1_buffer[f'host_max_attention_window_size_{i}'] = torch.tensor(
                [total_seq_len], dtype=torch.int32)
        # For step 1, the sequence_lengths = context_lengths + 1.
        sequence_length_buffer = torch.add(sequence_length_buffer, step)
        step1_buffer['sequence_length'] = sequence_length_buffer
        step1_buffer['host_past_key_value_lengths'] = torch.tensor(
            [seq_len + step - 1] * batch_size, dtype=torch.int32)
        step1_buffer['host_sink_token_length'] = torch.tensor([0],
                                                              dtype=torch.int32)

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-1)

        compare_max_abs_error(ref, res, "generation logits")


if __name__ == '__main__':
    unittest.main()
