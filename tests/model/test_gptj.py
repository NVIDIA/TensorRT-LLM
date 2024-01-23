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
import tensorrt as trt
# isort: on
from parameterized import parameterized
from transformers import GPTJConfig, GPTJForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.gptj.weight import load_from_hf_gpt_j

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import getSMVersion


class TestGPTJ(unittest.TestCase):

    def _gen_hf_gpt_j(self, hidden_act, n_layer, max_length, dtype):
        gpt_config = GPTJConfig(activation_function=hidden_act,
                                n_layer=n_layer,
                                max_length=max_length,
                                torch_dtype=dtype,
                                n_embd=4096,
                                n_head=16,
                                rotary_dim=64)
        hf_gpt = GPTJForCausalLM(gpt_config).cuda().to(
            tensorrt_llm._utils.str_dtype_to_torch(dtype)).eval()
        return gpt_config, hf_gpt

    def _gen_tensorrt_llm_network(self, network, hf_gpt, gpt_config, batch_size,
                                  beam_width, input_len, output_len, fp16,
                                  tensor_parallel):
        num_layers = gpt_config.n_layer
        num_heads = gpt_config.n_head
        hidden_size = gpt_config.n_embd
        vocab_size = gpt_config.vocab_size
        hidden_act = gpt_config.activation_function
        n_positions = gpt_config.n_positions
        rotary_dim = gpt_config.rotary_dim

        with net_guard(network):
            kv_dtype = trt.float16 if fp16 else trt.float32
            # Initialize model
            tensorrt_llm_gpt = tensorrt_llm.models.GPTJForCausalLM(
                num_layers=num_layers,
                num_heads=num_heads,
                hidden_size=hidden_size,
                vocab_size=vocab_size,
                hidden_act=hidden_act,
                max_position_embeddings=n_positions,
                rotary_dim=rotary_dim,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(world_size=tensor_parallel,
                                             tp_size=tensor_parallel),
            )
            inputs = tensorrt_llm_gpt.prepare_inputs(max_batch_size=batch_size,
                                                     max_input_len=input_len,
                                                     max_seq_len=input_len +
                                                     output_len,
                                                     use_cache=True,
                                                     max_beam_width=beam_width)
            load_from_hf_gpt_j(tensorrt_llm_gpt, hf_gpt, fp16=fp16)

            # Prepare
            network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

            tensorrt_llm_gpt(*inputs)

        return network

    def _gen_tensorrt_llm_runtime(self,
                                  dtype,
                                  world_size,
                                  rank,
                                  gpt_config,
                                  hf_gpt,
                                  use_attention_plugin,
                                  batch_size,
                                  beam_width,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  use_ln_gemm_plugin,
                                  context_fmha_flag=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False):
        tensorrt_llm.logger.set_level('error')
        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)

        runtime = None
        builder = Builder()
        fp16 = (dtype == 'float16')

        with tempfile.TemporaryDirectory() as tmpdirname:

            builder_config = builder.create_builder_config(
                name='gptj',
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

            self._gen_tensorrt_llm_network(network, hf_gpt, gpt_config,
                                           batch_size, beam_width, input_len,
                                           output_len, fp16, world_size)

            engine_buffer = builder.build_engine(network, builder_config)
            assert engine_buffer is not None
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)

            ok = builder.save_timing_cache(builder_config, 'model.cache')
            assert ok, "Failed to save timing cache."

        return runtime, engine_buffer

    def load_test_cases():
        test_cases = list(
            product([
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], [False, True]))

        return test_cases

    @parameterized.expand(load_test_cases)
    def test_gptj_plugin(self, context_fmha_flag, enable_remove_input_padding):

        # Skip tests that are not supported in pre-ampere architecture
        if getSMVersion() < 80:
            if context_fmha_flag == ContextFMHAType.enabled_with_fp32_acc:
                pytest.skip(
                    "ContextFMHAType with fp32 acc is not supported in pre-ampere architecture"
                )

        torch.random.manual_seed(0)
        use_refit = False
        dtype = 'float16'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 2
        max_length = 2
        batch_size = 1
        beam_width = 1
        seq_len = 12
        total_seq_len = max_length + seq_len
        use_attention_plugin = True
        use_ln_gemm_plugin = True

        gpt_config, hf_gpt = self._gen_hf_gpt_j(hidden_act, n_layer,
                                                seq_len + max_length, dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            dtype,
            world_size,
            rank,
            gpt_config,
            hf_gpt,
            use_attention_plugin,
            batch_size,
            beam_width,
            seq_len,
            max_length,
            use_refit,
            use_ln_gemm_plugin,
            context_fmha_flag,
            enable_remove_input_padding=enable_remove_input_padding)

        key_value_cache_buffers = []
        head_size = gpt_config.n_embd // gpt_config.n_head
        for i in range(gpt_config.n_layer):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    gpt_config.n_head,
                    total_seq_len,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))

        def run_engine(context,
                       input_ids,
                       context_lengths,
                       host_request_types,
                       position_ids,
                       last_token_ids,
                       cache_indirection,
                       host_past_key_value_lengths,
                       host_max_attention_window_sizes,
                       host_sink_token_length,
                       sequence_length,
                       host_context_lengths=None):

            ctx_buffer = {
                'input_ids': input_ids,
                'context_lengths': context_lengths,
                'host_request_types': host_request_types,
                'position_ids': position_ids,
                'last_token_ids': last_token_ids,
                'cache_indirection': cache_indirection,
                'host_past_key_value_lengths': host_past_key_value_lengths,
                'sequence_length': sequence_length,
                'host_sink_token_length': host_sink_token_length,
            }
            for i in range(gpt_config.n_layer):
                ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
                ctx_buffer[
                    f'host_max_attention_window_size_{i}'] = host_max_attention_window_sizes
                ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[
                    i]

            if enable_remove_input_padding:
                assert host_context_lengths is not None, "host_context_lengths is required for ragged input"
                ctx_buffer['host_context_lengths'] = host_context_lengths

            ctx_shape = {
                key: buffer.shape
                for key, buffer in ctx_buffer.items()
            }

            runtime._set_shape(context, ctx_shape)
            runtime._set_buffer(context, ctx_buffer)

            runtime._run(context)
            torch.cuda.synchronize()
            res = ctx_buffer['logits']
            return res

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

        ctx_context_lengths = seq_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')

        # We need sequence_lengths start as context_lengths, and are added one in each step.
        sequence_length_buffer = ctx_context_lengths.detach().clone()

        hf_outputs = None

        def compare_context():
            ctx_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()

            with torch.no_grad():
                nonlocal hf_outputs
                hf_outputs = hf_gpt.forward(ctx_ids, use_cache=True)
            torch.cuda.synchronize()
            ref = hf_outputs.logits[:, -1, :]

            ctx_position_ids = torch.tensor(range(seq_len),
                                            dtype=torch.int32).reshape([
                                                1, seq_len
                                            ]).expand([batch_size,
                                                       seq_len]).cuda()
            ctx_last_token_ids = ctx_context_lengths.clone()

            if enable_remove_input_padding:
                ctx_ids = ctx_ids.view([batch_size * seq_len])
                ctx_position_ids = ctx_position_ids.view([batch_size * seq_len])
                ctx_last_token_ids = torch.cumsum(ctx_last_token_ids,
                                                  dim=0).int()

            host_request_types = torch.tensor([0 for i in range(batch_size)],
                                              dtype=torch.int32).cpu()
            host_past_key_value_lengths = torch.tensor([0] * batch_size,
                                                       dtype=torch.int32)
            host_max_attention_window_sizes = torch.tensor([total_seq_len],
                                                           dtype=torch.int32)
            host_sink_token_length = torch.tensor([0], dtype=torch.int32)

            host_context_lengths = ctx_context_lengths.cpu(
            ) if enable_remove_input_padding else None

            res = run_engine(
                context=runtime.ctx_context,
                input_ids=ctx_ids,
                context_lengths=ctx_context_lengths,
                position_ids=ctx_position_ids,
                last_token_ids=ctx_last_token_ids,
                cache_indirection=cache_indirections[0],
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                sequence_length=sequence_length_buffer,
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types)

            np.testing.assert_allclose(ref.cpu().numpy(),
                                       res.cpu().numpy(),
                                       atol=1e-1)

            v_inner = 16 // (2 if dtype == 'float16' else 4)
            for i in range(gpt_config.n_layer):
                res_present_key_value = key_value_cache_buffers[i]

                past_key_value_tensor = res_present_key_value.permute(
                    1, 0, 2, 3, 4)
                key, value = past_key_value_tensor.chunk(2)

                # TRT-LLM has the same cache layout for key and value:
                # [bs, n_head, max_seq_len, head_size]
                head_size = gpt_config.n_embd // gpt_config.n_head
                key = key.reshape(batch_size, gpt_config.n_head, total_seq_len,
                                  head_size)

                value = value.reshape(batch_size, gpt_config.n_head,
                                      total_seq_len, head_size)

                ref_present_key, ref_present_value = hf_outputs.past_key_values[
                    i]

                np.testing.assert_allclose(ref_present_key.cpu().numpy(),
                                           key[:, :, :seq_len, :].cpu().numpy(),
                                           atol=1e-1)
                np.testing.assert_allclose(
                    ref_present_value.cpu().numpy(),
                    value[:, :, :seq_len, :].cpu().numpy(),
                    atol=1e-1)

        def compare_generation():
            step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
            gen_position_ids = torch.ones_like(step1_id).int().cuda() * seq_len
            gen_context_lengths = ctx_context_lengths.clone()
            gen_last_token_ids = torch.zeros_like(
                gen_context_lengths).int().cuda()

            with torch.no_grad():
                nonlocal hf_outputs
                hf_outputs = hf_gpt.forward(
                    step1_id,
                    past_key_values=hf_outputs.past_key_values,
                    position_ids=gen_position_ids.to(torch.int64),
                    use_cache=True)
            torch.cuda.synchronize()
            ref = hf_outputs.logits[:, -1, :]

            if enable_remove_input_padding:
                step1_id = step1_id.view([batch_size])
                gen_position_ids = gen_position_ids.view([batch_size])
                gen_last_token_ids = torch.ones_like(
                    gen_context_lengths).int().cuda()
                gen_last_token_ids = torch.cumsum(gen_last_token_ids,
                                                  dim=0).int()

            host_past_key_value_lengths = torch.tensor([seq_len] * batch_size,
                                                       dtype=torch.int32)

            host_max_attention_window_sizes = torch.tensor([total_seq_len],
                                                           dtype=torch.int32)

            host_sink_token_length = torch.tensor([0], dtype=torch.int32)

            host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32).cpu()
            host_context_lengths = gen_context_lengths.cpu(
            ) if enable_remove_input_padding else None

            # For step 1, the sequence_lengths = context_lengths + 1.
            sequence_length_buffer = torch.add(ctx_context_lengths, 1)

            res = run_engine(
                context=runtime.context_1,
                input_ids=step1_id,
                # note we should pass context length for generation phase.
                context_lengths=ctx_context_lengths,
                position_ids=gen_position_ids,
                last_token_ids=gen_last_token_ids,
                cache_indirection=cache_indirections[1],
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                sequence_length=sequence_length_buffer,
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types)

            np.testing.assert_allclose(ref.cpu().numpy(),
                                       res.cpu().numpy(),
                                       atol=1e-1)

        compare_context()
        compare_generation()

    def test_gptj_noplugin_supported(self):

        use_refit = False

        dtype = 'float16'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 1
        max_length = 2
        batch_size = 4
        seq_len = 128
        use_attention_plugin = False
        use_ln_gemm_plugin = True
        beam_width = 1

        gpt_config, hf_gpt = self._gen_hf_gpt_j(hidden_act, n_layer,
                                                seq_len + max_length, dtype)

        runtime, _ = self._gen_tensorrt_llm_runtime(
            dtype, world_size, rank, gpt_config, hf_gpt, use_attention_plugin,
            batch_size, beam_width, seq_len, max_length, use_refit,
            use_ln_gemm_plugin)

        use_ln_gemm_plugin = False
        if trt.__version__[:3] == '8.6':
            with self.assertRaisesRegex(
                    AssertionError,
                    "You need to enable the LayerNorm plugin for GPT-J with TensorRT"
            ):
                runtime, _ = self._gen_tensorrt_llm_runtime(
                    dtype, world_size, rank, gpt_config, hf_gpt,
                    use_attention_plugin, batch_size, beam_width, seq_len,
                    max_length, use_refit, use_ln_gemm_plugin)


if __name__ == '__main__':
    unittest.main()
