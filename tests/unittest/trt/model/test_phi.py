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
from itertools import product

import numpy as np
import torch
from parameterized import parameterized
from transformers import PhiConfig, PhiForCausalLM
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm.models.phi.convert import load_weights_from_hf_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType


def compare_max_abs_error(ref, res, str):
    # calculate max abs error
    compare_HF = ref.cpu().numpy().flatten()
    compare_TRT_LLM = res.cpu().numpy().flatten()
    max_abs_error = np.max(abs(compare_TRT_LLM - compare_HF))
    print(str, "max abs error = ", max_abs_error)


class TestPhi(unittest.TestCase):

    def setUp(self):
        super().setUp()
        # Fix random seed for the reproducibility.
        torch.random.manual_seed(1773)

    def generate_hf_model(self, dtype: str):
        phi_config = PhiConfig(num_hidden_layers=2)
        model = PhiForCausalLM(phi_config).cuda().to(
            tensorrt_llm._utils.str_dtype_to_torch(dtype)).eval()
        return phi_config, model

    def initialize_network(self, network: tensorrt_llm.Network, hf_model,
                           hf_config, dtype: str, batch_size: int,
                           beam_width: int, input_len: int, output_len: int,
                           tensor_parallel: int, rank: int):
        config = {
            'architecture': 'PhiForCausalLM',
            'dtype': dtype,
            'num_hidden_layers': hf_config.num_hidden_layers,
            'num_attention_heads': hf_config.num_key_value_heads,
            'rotary_pct': hf_config.partial_rotary_factor,
            'position_embedding_type': 'rope_gpt_neox',
            'rope_theta': hf_config.rope_theta,
            'hidden_size': hf_config.hidden_size,
            'intermediate_size': hf_config.intermediate_size,
            'vocab_size': hf_config.vocab_size,
            'max_position_embeddings': hf_config.max_position_embeddings,
            'hidden_act': hf_config.hidden_act,
            'mapping': {
                'world_size': tensor_parallel,
                'tp_size': tensor_parallel,
                'world_size': tensor_parallel,
            },
            'use_parallel_embedding': False,
            'embedding_sharding_dim': 0,
        }
        config = tensorrt_llm.models.PretrainedConfig.from_dict(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(hf_model, config)
        trtllm_model = tensorrt_llm.models.PhiForCausalLM(config)
        trtllm_model.load(weights)

        with net_guard(network):
            # Initialize model
            network.set_named_parameters(trtllm_model.named_parameters())
            inputs = trtllm_model.prepare_inputs(batch_size,
                                                 input_len,
                                                 input_len + output_len,
                                                 max_num_tokens=batch_size *
                                                 input_len,
                                                 use_cache=True,
                                                 max_beam_width=beam_width)
            # Prepare
            trtllm_model(**inputs)

    def generate_trtllm_runtime(self,
                                log_level,
                                dtype,
                                world_size,
                                rank,
                                hf_config,
                                hf_model,
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

        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name='phi',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
                strongly_typed=True,
            )
            network = builder.create_network()
            network.plugin_config.to_legacy_setting()
            if use_attention_plugin:
                network.plugin_config.gpt_attention_plugin = dtype
            if use_ln_gemm_plugin:
                network.plugin_config.gemm_plugin = dtype
            if enable_remove_input_padding:
                network.plugin_config.remove_input_padding = True
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self.initialize_network(network, hf_model, hf_config, dtype,
                                    batch_size, beam_width, input_len,
                                    output_len, world_size, rank)

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

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_phi(self, context_fmha_flag, enable_remove_input_padding):

        torch.random.manual_seed(0)
        use_refit = False
        apply_query_key_layer_scaling = False
        model = 'phi'

        log_level = 'error'
        dtype = 'float16'
        world_size = 1
        rank = 0
        max_length = 128
        batch_size = 1
        beam_width = 1
        seq_len = 128
        total_seq_len = max_length + seq_len
        use_attention_plugin = True
        use_ln_gemm_plugin = True

        gpt_config, hf_gpt = self.generate_hf_model(dtype)
        runtime, _ = self.generate_trtllm_runtime(
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

        perf_knob_tensor_size = 16
        context_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                                  dtype=torch.int64)
        if context_fmha_flag == ContextFMHAType.enabled_with_fp32_acc:
            context_runtime_perf_knobs[1] = 1  # enable_context_fmha_fp32_acc
        host_context_progress = torch.tensor([0], dtype=torch.int64)

        ctx_buffer = {
            'input_ids': ctx_ids,
            'context_lengths': ctx_context_lengths,
            'host_request_types': ctx_host_request_types,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'cache_indirection': cache_indirections[0],
            'host_runtime_perf_knobs': context_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        if enable_remove_input_padding:
            ctx_buffer['host_context_lengths'] = ctx_context_lengths.cpu()
        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}
        shape = (batch_size, 2, gpt_config.num_attention_heads, total_seq_len,
                 gpt_config.hidden_size // gpt_config.num_attention_heads)
        ctx_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
            [total_seq_len] * gpt_config.num_hidden_layers, dtype=torch.int32)
        ctx_shape[f'host_max_attention_window_sizes'] = (
            gpt_config.num_hidden_layers, )
        for i in range(gpt_config.num_hidden_layers):
            ctx_shape[f'past_key_value_{i}'] = shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
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

        # compare generation
        step = 1
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_position_ids = torch.ones_like(step1_id).int().cuda() * seq_len
        gen_context_lengths = ctx_context_lengths.clone()
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32)
        gen_last_token_ids = torch.zeros_like(gen_context_lengths).int().cuda()

        with torch.no_grad():
            hf_input_ids = torch.cat((ctx_ids.reshape(1, seq_len), step1_id), 1)
            hf_outputs = hf_gpt.forward(hf_input_ids, use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            step1_id = step1_id.view([batch_size])
            gen_position_ids = gen_position_ids.view([batch_size])
            gen_last_token_ids = torch.ones_like(
                gen_context_lengths).int().cuda()
            gen_last_token_ids = torch.cumsum(gen_last_token_ids, dim=0).int()

        gen_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                              dtype=torch.int64)
        if context_fmha_flag == ContextFMHAType.enabled_with_fp32_acc:
            gen_runtime_perf_knobs[1] = 1  # enable_context_fmha_fp32_acc

        step1_buffer = {
            'input_ids': step1_id,
            'context_lengths': gen_context_lengths,
            'host_request_types': gen_host_request_types,
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'cache_indirection': cache_indirections[1],
            'host_runtime_perf_knobs': gen_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        if enable_remove_input_padding:
            step1_buffer['host_context_lengths'] = gen_context_lengths.cpu()
        step1_shape = {k: v.shape for k, v in step1_buffer.items()}
        step1_shape[f'host_max_attention_window_sizes'] = (
            gpt_config.num_hidden_layers, )
        for i in range(gpt_config.num_hidden_layers):
            step1_shape[f'past_key_value_{i}'] = shape
        step1_shape['sequence_length'] = (batch_size, )
        step1_shape['host_past_key_value_lengths'] = (batch_size, )
        step1_shape['host_sink_token_length'] = (1, )
        step1_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
            [total_seq_len] * gpt_config.num_hidden_layers, dtype=torch.int32)
        for i in range(gpt_config.num_hidden_layers):
            step1_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            step1_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
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
