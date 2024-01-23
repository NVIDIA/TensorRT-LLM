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
from transformers import BloomConfig, BloomForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.generation import _prepare_attention_mask

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.bloom.convert_checkpoint import convert_hf_bloom

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestBloom(unittest.TestCase):

    def _gen_hf_bloom(self, hidden_act, n_layer, max_length, dtype):
        bloom_config = BloomConfig(
            hidden_act=hidden_act,
            n_layer=n_layer,
            max_length=max_length,
            torch_dtype=dtype,
        )

        hf_bloom = BloomForCausalLM(bloom_config).cuda().eval()
        return bloom_config, hf_bloom

    def _gen_tensorrt_llm_network(self, network, builder, hf_bloom,
                                  bloom_config, batch_size, input_len,
                                  output_len, fp16, gpt_attention_plugin,
                                  tensor_parallel,
                                  apply_query_key_layer_scaling):
        dtype = 'float16' if fp16 else 'float32'
        config = {
            'architecture': 'BloomForCausalLM',
            'dtype': dtype,
            'num_hidden_layers': bloom_config.n_layer,
            'num_attention_heads': bloom_config.n_head,
            'hidden_size': bloom_config.hidden_size,
            'vocab_size': bloom_config.vocab_size,
            'position_embedding_type': 'alibi',
            'max_position_embeddings': input_len + output_len,
            'hidden_act': 'gelu',
            'mapping': {
                'world_size': tensor_parallel,
                'tp_size': tensor_parallel
            },
            'use_parallel_embedding': False,
            'embedding_sharding_dim': 0,
            'share_embedding_table': False,
        }
        config = tensorrt_llm.models.PretrainedConfig.from_dict(config)
        # config.set_rank(rank)
        weights = convert_hf_bloom(hf_bloom,
                                   tensor_parallel=tensor_parallel,
                                   dtype=dtype)
        tensorrt_llm_bloom = tensorrt_llm.models.BloomForCausalLM(config)
        tensorrt_llm_bloom.load(weights)

        with net_guard(network):
            network.set_named_parameters(tensorrt_llm_bloom.named_parameters())
            inputs = tensorrt_llm_bloom.prepare_inputs(
                max_batch_size=batch_size,
                max_input_len=input_len,
                max_seq_len=input_len + output_len,
                use_cache=True,
                max_beam_width=1)
            # Prepare
            tensorrt_llm_bloom(**inputs)

        return network

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  bloom_config,
                                  hf_bloom,
                                  model,
                                  use_plugin,
                                  batch_size,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  fast_building=False,
                                  apply_query_key_layer_scaling=False,
                                  context_fmha_type=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False):
        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)

        runtime = None
        builder = Builder()
        fp16 = (dtype == 'float16')

        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name='bloom',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
                strongly_typed=fp16,
            )
            network = builder.create_network()
            if use_plugin:
                network.plugin_config.set_gpt_attention_plugin(dtype)
            if fast_building:
                network.plugin_config.set_gemm_plugin(dtype)
            network.plugin_config.set_context_fmha(context_fmha_type)
            if enable_remove_input_padding:
                network.plugin_config.enable_remove_input_padding()

            self._gen_tensorrt_llm_network(network, builder, hf_bloom,
                                           bloom_config, batch_size, input_len,
                                           output_len, fp16, use_plugin,
                                           world_size,
                                           apply_query_key_layer_scaling)

            engine_buffer = builder.build_engine(network, builder_config)
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)
        return runtime, engine_buffer

    def load_test_cases():
        test_cases = list(
            product([False, True], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], [False], ['float16', 'float32']))
        return test_cases

    @parameterized.expand(load_test_cases())
    def test_bloom(self, use_gpt_attention_plugin, context_fmha_type,
                   enable_remove_input_padding, dtype):
        model = 'bloom'
        log_level = 'error'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 2
        max_length = 2
        batch_size = 4
        beam_width = 1
        seq_len = 128
        total_length = seq_len + max_length

        bloom_config, hf_bloom = self._gen_hf_bloom(hidden_act, n_layer,
                                                    max_length, dtype)
        if bloom_config.hidden_size // bloom_config.n_head < 32 and use_gpt_attention_plugin:
            pytest.skip("unsupported head_size")
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level,
            dtype,
            world_size,
            rank,
            bloom_config,
            hf_bloom,
            model,
            use_gpt_attention_plugin,
            batch_size,
            seq_len,
            max_length,
            use_refit=False,
            fast_building=True,
            context_fmha_type=context_fmha_type,
            enable_remove_input_padding=enable_remove_input_padding)

        # compare context
        pad_token_id = 3
        ctx_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        ctx_ids[0][-1] = pad_token_id
        ctx_ids[1][-3:] = pad_token_id
        ctx_ids[2][-5:] = pad_token_id
        ctx_context_lengths = seq_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_position_ids = torch.tensor(range(seq_len),
                                        dtype=torch.int32).reshape([
                                            1, seq_len
                                        ]).expand([batch_size, seq_len]).cuda()
        ctx_last_token_ids = ctx_context_lengths.clone()
        ctx_attention_mask = _prepare_attention_mask(ctx_ids)
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32)
        ctx_sequence_length = torch.tensor([seq_len] * batch_size,
                                           dtype=torch.int32).cuda()
        ctx_host_past_key_value_lengths = torch.tensor([0] * batch_size,
                                                       dtype=torch.int32)
        host_max_attention_window_sizes = torch.tensor([total_length],
                                                       dtype=torch.int32)
        host_sink_token_length = torch.tensor([0], dtype=torch.int32)

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                total_length,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                total_length,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers

        ctx_buffer = {
            'input_ids': ctx_ids,
            'position_ids': ctx_position_ids,
            'context_lengths': ctx_context_lengths,
            'last_token_ids': ctx_last_token_ids,
            'host_request_types': ctx_host_request_types,
            'cache_indirection': cache_indirections[0],
        }

        ctx_host_context_lengths = None
        if use_gpt_attention_plugin:
            ctx_buffer['sequence_length'] = ctx_sequence_length
            ctx_buffer[
                'host_past_key_value_lengths'] = ctx_host_past_key_value_lengths
            ctx_buffer['host_sink_token_length'] = host_sink_token_length
            if enable_remove_input_padding:
                ctx_host_context_lengths = ctx_context_lengths.cpu()
                ctx_buffer["host_context_lengths"] = ctx_host_context_lengths
        else:
            ctx_buffer['attention_mask'] = ctx_attention_mask

        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

        for i in range(bloom_config.n_layer):
            shape = (batch_size, 2, bloom_config.n_head, 0,
                     bloom_config.hidden_size // bloom_config.n_head)
            past_buffer = torch.zeros((1, ),
                                      dtype=str_dtype_to_torch(dtype),
                                      device='cuda')
            ctx_shape.update({
                f'past_key_value_{i}': shape,
                f'host_max_attention_window_size_{i}': (1, ),
            })
            shape = (batch_size, 2, bloom_config.n_head, seq_len,
                     bloom_config.hidden_size // bloom_config.n_head)
            ctx_buffer.update({
                f'past_key_value_{i}':
                past_buffer,
                f'present_key_value_{i}':
                torch.zeros(shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'),
                f'host_max_attention_window_size_{i}':
                host_max_attention_window_sizes,
            })

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        with torch.no_grad():
            hf_outputs = hf_bloom.forward(ctx_ids,
                                          attention_mask=ctx_attention_mask)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-2)

        # compare generation
        step = 1
        gen_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_context_lengths = ctx_context_lengths.clone()
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32)
        gen_position_ids = torch.ones_like(gen_id).cuda() * seq_len
        gen_last_token_ids = torch.zeros_like(gen_context_lengths).cuda()
        gen_attention_mask = torch.cat([
            ctx_attention_mask,
            ctx_attention_mask.new_ones((ctx_attention_mask.shape[0], 1))
        ],
                                       dim=-1)
        gen_sequence_length = torch.tensor([seq_len + step] * batch_size,
                                           dtype=torch.int32).cuda()
        gen_host_past_key_value_lengths = torch.tensor([seq_len + step - 1] *
                                                       batch_size,
                                                       dtype=torch.int32)
        gen_host_max_attention_window_sizes = torch.tensor([total_length],
                                                           dtype=torch.int32)
        gen_host_sink_token_length = torch.tensor([0], dtype=torch.int32)
        step1_buffer = {
            'input_ids': gen_id,
            'context_lengths': gen_context_lengths.contiguous(),
            'position_ids': gen_position_ids.contiguous(),
            'last_token_ids': gen_last_token_ids.contiguous(),
            'host_request_types': gen_host_request_types.contiguous(),
            'cache_indirection': cache_indirections[1].contiguous(),
        }

        gen_host_context_lengths = None
        if use_gpt_attention_plugin:
            step1_buffer['sequence_length'] = gen_sequence_length
            step1_buffer[
                'host_past_key_value_lengths'] = gen_host_past_key_value_lengths
            gen_host_context_lengths = gen_context_lengths.cpu()
            step1_buffer['host_context_lengths'] = gen_host_context_lengths
            step1_buffer['host_sink_token_length'] = gen_host_sink_token_length
        else:
            step1_buffer['attention_mask'] = gen_attention_mask

        step1_shape = {k: v.shape for k, v in step1_buffer.items()}

        for i in range(bloom_config.n_layer):
            shape = (batch_size, 2, bloom_config.n_head, seq_len,
                     bloom_config.hidden_size // bloom_config.n_head)
            step1_shape.update({
                f'past_key_value_{i}': shape,
                f'host_max_attention_window_size_{i}': (1, ),
            })
            step1_buffer.update({
                f'past_key_value_{i}':
                ctx_buffer[f'present_key_value_{i}'],
                f'host_max_attention_window_size_{i}':
                host_max_attention_window_sizes,
            })

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        with torch.no_grad():
            hf_outputs = hf_bloom.forward(
                gen_id,
                attention_mask=gen_attention_mask,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-1)

    @parameterized.expand(load_test_cases())
    def test_greedy_search(self, use_gpt_attention_plugin, context_fmha_type,
                           enable_remove_input_padding, dtype):

        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
                pytest.skip(
                    "ContextFMHAType with fp32 acc is not supported in pre-ampere architecture"
                )

        model = 'bloom'
        log_level = 'error'
        world_size = 1
        rank = 0

        hidden_act = 'gelu'
        n_layer = 2
        max_new_tokens = 1
        batch_size = 4
        seq_len = 128

        do_sample = False
        early_stoppping = False
        num_beams = 1
        num_beam_groups = 1
        temperature = 1
        top_k = 0
        top_p = 0.0
        length_penalty = 1
        repetition_penalty = 1

        bloom_config, hf_bloom = self._gen_hf_bloom(hidden_act, n_layer,
                                                    max_new_tokens, dtype)
        runtime, engine_buffer = self._gen_tensorrt_llm_runtime(
            log_level,
            dtype,
            world_size,
            rank,
            bloom_config,
            hf_bloom,
            model,
            use_gpt_attention_plugin,
            batch_size,
            seq_len,
            max_new_tokens,
            use_refit=False,
            fast_building=True,
            context_fmha_type=context_fmha_type,
            enable_remove_input_padding=enable_remove_input_padding)

        model_config = ModelConfig(
            vocab_size=bloom_config.vocab_size,
            num_layers=bloom_config.n_layer,
            num_heads=bloom_config.n_head,
            num_kv_heads=bloom_config.n_head,
            hidden_size=bloom_config.hidden_size,
            gpt_attention_plugin=use_gpt_attention_plugin,
            remove_input_padding=enable_remove_input_padding,
            dtype=dtype)

        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)
        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, mapping)
        pad_token_id = 3
        eos_token_id = 2
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p,
                                         length_penalty=length_penalty,
                                         repetition_penalty=repetition_penalty)
        input_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        input_ids[0][-1] = pad_token_id
        input_ids[1][-3:] = pad_token_id
        input_ids[2][-5:] = pad_token_id

        context_lengths = torch.ones(
            (batch_size)).type(torch.int32).cuda() * seq_len

        decoder.setup(batch_size,
                      max_context_length=seq_len,
                      max_new_tokens=max_new_tokens,
                      beam_width=num_beams)

        output_ids = decoder.decode(input_ids, context_lengths, sampling_config)
        # TODO: change to actual ragged tensor after BLOOM plugin supports it
        output_ids_x = decoder.decode(input_ids, context_lengths,
                                      sampling_config)

        torch.cuda.synchronize()
        torch.testing.assert_close(output_ids, output_ids_x)

        res = output_ids.squeeze()
        res = res[:, -max_new_tokens:]

        ref_output_ids = hf_bloom.generate(
            input_ids,
            do_sample=do_sample,
            early_stopping=early_stoppping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_beam_groups=num_beam_groups,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        torch.cuda.synchronize()
        ref = ref_output_ids[:, -max_new_tokens:]

        np.testing.assert_allclose(ref.cpu().numpy(), res.cpu().numpy())


if __name__ == '__main__':
    unittest.main()
