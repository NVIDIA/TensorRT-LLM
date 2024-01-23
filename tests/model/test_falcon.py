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
import unittest
from typing import Optional

import torch
from parameterized import parameterized
from transformers import FalconConfig, FalconForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.generation import _prepare_attention_mask

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from examples.falcon.convert_checkpoint import convert_hf_falcon  # isort:skip


class TestFalcon(unittest.TestCase):

    HFModelConfig = FalconConfig
    HFModel = FalconForCausalLM

    query_types = ['MHA', 'MQA', 'GQA']

    def setUp(self):
        super().setUp()
        # Fix random seed for the reproducibility.
        torch.random.manual_seed(1773)

    def generate_hf_model(self,
                          output_len: int,
                          dtype: str,
                          query_type: str,
                          num_kv_heads: Optional[int] = None,
                          use_alibi: bool = True,
                          parallel_attention: bool = False,
                          new_decoder_architecture: bool = False):
        if isinstance(dtype, str):
            dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
        assert query_type in self.query_types
        num_heads = 4
        multi_query = False
        if query_type == 'MQA':
            num_kv_heads = 1
            multi_query = True
        elif query_type == 'GQA':
            num_kv_heads == num_heads // 2
        else:
            num_kv_heads = None  # query_type = 'MHA'

        config = self.HFModelConfig(
            num_hidden_layers=2,
            vocab_size=128,
            hidden_size=128,
            num_attention_heads=num_heads,
            bias=True,
            max_length=output_len,
            torch_dtype=dtype,
            alibi=use_alibi,
            new_decoder_architecture=new_decoder_architecture,
            multi_query=multi_query,
            parallel_attn=parallel_attention,
            num_kv_heads=num_kv_heads,
            pad_token_id=1,
            eos_token_id=0,
        )
        model = FalconForCausalLM(config).cuda().to(dtype).eval()
        return config, model

    def initialize_network(self, network: tensorrt_llm.Network,
                           hf_model: HFModel, hf_config: HFModelConfig,
                           dtype: str, batch_size: int, beam_width: int,
                           input_len: int, output_len: int,
                           tensor_parallel: int, rank: int):
        config = {
            'architecture': 'FalconForCausalLM',
            'dtype': dtype,
            'num_hidden_layers': hf_config.num_hidden_layers,
            'num_attention_heads': hf_config.num_attention_heads,
            'num_key_value_heads': hf_config.num_kv_heads,
            'hidden_size': hf_config.hidden_size,
            'vocab_size': hf_config.vocab_size,
            'position_embedding_type':
            'alibi_with_scale' if hf_config.alibi else 'rope_gpt_neox',
            'max_position_embeddings': input_len + output_len,
            'hidden_act': 'gelu',
            'mapping': {
                'world_size': tensor_parallel,
                'tp_size': tensor_parallel
            },
            'bias': hf_config.bias,
            'parallel_attention': hf_config.parallel_attn,
            'new_decoder_architecture': hf_config.new_decoder_architecture,
        }
        config = tensorrt_llm.models.PretrainedConfig.from_dict(config)
        config.set_rank(rank)
        weights = convert_hf_falcon(hf_model,
                                    hf_config,
                                    config.mapping,
                                    dtype=dtype)
        trtllm_model = tensorrt_llm.models.FalconForCausalLM(config)
        trtllm_model.load(weights)

        with net_guard(network):
            # Initialize model
            network.set_named_parameters(trtllm_model.named_parameters())
            inputs = trtllm_model.prepare_inputs(max_batch_size=batch_size,
                                                 max_input_len=input_len,
                                                 max_seq_len=input_len +
                                                 output_len,
                                                 use_cache=True,
                                                 max_beam_width=beam_width)
            # Prepare
            trtllm_model(**inputs)

    def generate_trtllm_runtime(self,
                                model_name: str,
                                hf_config: HFModelConfig,
                                hf_model: HFModel,
                                dtype: str,
                                world_size: int = 1,
                                rank: int = 0,
                                batch_size: int = 4,
                                beam_width: int = 1,
                                input_len: int = 128,
                                output_len: int = 2,
                                use_refit=False,
                                use_gpt_attengion_plugin=False,
                                use_gemm_plugin=False,
                                enable_remove_input_padding=False,
                                context_fmha_type=ContextFMHAType.disabled,
                                log_level: str = 'error'):
        tensorrt_llm.logger.set_level(log_level)
        mapping = tensorrt_llm.Mapping(world_size, rank)
        builder = Builder()

        builder_config = builder.create_builder_config(
            name=model_name,
            precision=dtype,
            timing_cache='model.cache',
            tensor_parallel=world_size,
            use_alibi=hf_config.alibi,
            parallel_attention=hf_config.parallel_attn,
            use_refit=use_refit,
            strongly_typed=(dtype == "float16"),
        )

        network = builder.create_network()
        if use_gpt_attengion_plugin:
            network.plugin_config.set_gpt_attention_plugin(dtype)
        if use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype)
        if enable_remove_input_padding:
            network.plugin_config.enable_remove_input_padding()
        if world_size > 1:
            network.plugin_config.set_nccl_plugin(dtype)
        network.plugin_config.set_context_fmha(context_fmha_type)

        self.initialize_network(network=network,
                                hf_model=hf_model,
                                hf_config=hf_config,
                                dtype=dtype,
                                batch_size=batch_size,
                                beam_width=beam_width,
                                input_len=input_len,
                                output_len=output_len,
                                tensor_parallel=world_size,
                                rank=rank)

        engine_buffer = builder.build_engine(network, builder_config)
        runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)

        ok = builder.save_timing_cache(builder_config, 'model.cache')
        assert ok, "Failed to save timing cache."

        return runtime, engine_buffer

    def load_test_cases():
        test_cases = [
            # TC for Falcon-1B arch: MHA + ALiBi
            ('MHA', True, False, False, False, False, True, False,
             ContextFMHAType.disabled, 'float16'),
            ('MHA', True, False, False, False, False, True, False,
             ContextFMHAType.disabled, 'float32'),
            # TC for Falcon-7B arch: MQA + RoPE + parallel_attention
            ('MQA', False, True, False, False, True, True, False,
             ContextFMHAType.disabled, 'float16'),
            ('MQA', False, True, False, False, True, True, False,
             ContextFMHAType.disabled, 'float32'),
            # TC for Falcon-40B arch: GQA + RoPE + new_decoder_architecture
            ('GQA', False, False, True, False, True, True, False,
             ContextFMHAType.disabled, 'float16'),
            ('GQA', False, False, True, False, True, True, False,
             ContextFMHAType.disabled, 'float32'),
        ]
        return test_cases

    @staticmethod
    def convert_to_left_padding(token_ids, pad_id):
        converted = token_ids.clone()
        for i, tokens in enumerate(token_ids):
            assert pad_id is not None
            vals, cnts = tokens.unique_consecutive(return_counts=True)
            # Check if the last consecutive elements are pad tokens.
            if vals[-1] == pad_id:
                converted[i, :] = tokens.roll(cnts[-1].item())
        return converted

    @staticmethod
    def prepare_input_token_ids(batch_size,
                                input_len,
                                vocab_size,
                                pad_id=None,
                                remove_input_padding=False,
                                device=None):
        input_ids = torch.randint(vocab_size, (batch_size, input_len),
                                  dtype=torch.int32,
                                  device=device)
        context_lengths = input_ids.new_full((batch_size, ), input_len)
        if pad_id is not None:
            for i in range(1, batch_size):
                input_ids[i, -i:] = pad_id
                context_lengths[i] = input_len - i
        last_token_ids = context_lengths.clone()
        if remove_input_padding:
            last_token_ids = torch.cumsum(last_token_ids, dim=0)
        return input_ids, context_lengths, last_token_ids

    def skip_test_case(self, query_type, use_alibi, parallel_attention,
                       new_decoder_architecture, use_refit,
                       use_gpt_attengion_plugin, use_gemm_plugin,
                       remove_input_padding, context_fmha_type, dtype):
        print(' Test Case Parameters')
        print(' - query_type', query_type)
        print(' - use_alibi', use_alibi)
        print(' - parallel_attention', parallel_attention)
        print(' - new_decoder_architecture', new_decoder_architecture)
        print(' - use_refit', use_refit)
        print(' - use_gpt_attengion_plugin', use_gpt_attengion_plugin)
        print(' - use_gemm_plugin', use_gemm_plugin)
        print(' - remove_input_padding', remove_input_padding)
        print(' - context_fmha_type', context_fmha_type)
        print(' - dtype', dtype)

        # Skip unsupported cases.
        if use_alibi and use_gpt_attengion_plugin:
            self.skipTest('ALiBi needs use_gpt_attengion_plugin = False')
        if not use_alibi and not use_gpt_attengion_plugin:
            self.skipTest('RoPE needs use_gpt_attengion_plugin = True')

    @parameterized.expand(load_test_cases())
    def test_falcon(self, query_type, use_alibi, parallel_attention,
                    new_decoder_architecture, use_refit,
                    use_gpt_attengion_plugin, use_gemm_plugin,
                    remove_input_padding, context_fmha_type, dtype):
        self.skip_test_case(query_type, use_alibi, parallel_attention,
                            new_decoder_architecture, use_refit,
                            use_gpt_attengion_plugin, use_gemm_plugin,
                            remove_input_padding, context_fmha_type, dtype)
        world_size = 1
        rank = 0
        batch_size = 3
        beam_width = 1
        input_len = 7
        output_len = 2
        total_length = input_len + output_len
        log_level = 'error'

        hf_config, hf_model = self.generate_hf_model(
            output_len,
            dtype,
            use_alibi=use_alibi,
            parallel_attention=parallel_attention,
            new_decoder_architecture=new_decoder_architecture,
            query_type=query_type)
        runtime, _ = self.generate_trtllm_runtime(
            model_name='falcon',
            hf_config=hf_config,
            hf_model=hf_model,
            dtype=dtype,
            world_size=world_size,
            rank=rank,
            batch_size=batch_size,
            beam_width=beam_width,
            input_len=input_len,
            output_len=output_len,
            use_refit=use_refit,
            use_gpt_attengion_plugin=use_gpt_attengion_plugin,
            use_gemm_plugin=use_gemm_plugin,
            enable_remove_input_padding=remove_input_padding,
            context_fmha_type=context_fmha_type,
            log_level=log_level)

        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        num_kv_heads = hf_config.num_kv_heads
        kv_dtype = dtype
        device = hf_model.device
        pad_id = hf_config.pad_token_id

        # 1. Check the correctness of context computation.

        # Prepare context inputs.
        ctx_input_ids, ctx_context_lengths, ctx_last_token_ids = \
            self.prepare_input_token_ids(
                batch_size, input_len,
                vocab_size=hf_config.vocab_size,
                # Skip testing padded inputs due to bugs in HF Falcon.
                # Will enable when those are fixed.
                pad_id=None,
                remove_input_padding=remove_input_padding,
                device=device)
        ctx_position_ids = torch.arange(0,
                                        input_len,
                                        dtype=torch.int32,
                                        device=device).expand([batch_size, -1])
        ctx_attention_mask = _prepare_attention_mask(ctx_input_ids, pad_id)
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32)

        # ping-pong buffers
        cache_indirections = [
            torch.zeros((batch_size, beam_width, total_length),
                        dtype=torch.int32,
                        device=device),
            torch.zeros((batch_size, beam_width, total_length),
                        dtype=torch.int32,
                        device=device)
        ]

        # We need sequence_lengths start as context_lengths for step 0 (context),
        # and it will be added one after each step.
        sequence_length = ctx_context_lengths.detach().clone()
        # past kv length: (length, is_context)
        host_past_key_value_lengths = torch.tensor([0] * batch_size,
                                                   dtype=torch.int32)
        host_max_attention_window_sizes = torch.tensor([total_length],
                                                       dtype=torch.int32)
        host_sink_token_length = torch.tensor([0], dtype=torch.int32)

        ctx_buffer = {
            'input_ids': ctx_input_ids.contiguous(),
            'position_ids': ctx_position_ids.contiguous(),
            'context_lengths': ctx_context_lengths.contiguous(),
            'last_token_ids': ctx_last_token_ids.contiguous(),
            'attention_mask': ctx_attention_mask.contiguous(),
            'host_request_types': ctx_host_request_types.contiguous(),
            'cache_indirection': cache_indirections[0],
            'sequence_length': sequence_length.contiguous(),
            'host_past_key_value_lengths':
            host_past_key_value_lengths.contiguous(),
            'host_sink_token_length': host_sink_token_length,
        }
        if remove_input_padding:
            ctx_buffer['host_context_lengths'] = ctx_context_lengths.cpu()
        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

        if use_gpt_attengion_plugin:
            kv_shape = (batch_size, 2, num_kv_heads, total_length, head_dim)
            past_kv_shape = kv_shape
            present_kv_shape = kv_shape
        else:
            past_kv_shape = (batch_size, 2, num_kv_heads, 0, head_dim)
            present_kv_shape = (batch_size, 2, num_kv_heads, input_len,
                                head_dim)
        for i in range(hf_config.num_hidden_layers):
            ctx_shape[f'past_key_value_{i}'] = past_kv_shape
            ctx_shape[f'host_max_attention_window_size_{i}'] = (1, )
            ctx_buffer[f'present_key_value_{i}'] = torch.zeros(
                present_kv_shape,
                dtype=str_dtype_to_torch(kv_dtype),
                device=device)
            if use_gpt_attengion_plugin:
                ctx_buffer[f'past_key_value_{i}'] = ctx_buffer[
                    f'present_key_value_{i}']
                ctx_buffer[
                    f'host_max_attention_window_size_{i}'] = host_max_attention_window_sizes
            else:
                ctx_buffer[f'past_key_value_{i}'] = torch.zeros(
                    (1, ), dtype=str_dtype_to_torch(kv_dtype), device=device)

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits'].float()

        with torch.no_grad():
            # A decoder-only model of HF requires left padding.
            hf_ctx_input_ids = self.convert_to_left_padding(
                ctx_input_ids, pad_id)
            hf_ctx_attn_mask = _prepare_attention_mask(hf_ctx_input_ids,
                                                       pad_id=pad_id)
            hf_outputs = hf_model.forward(hf_ctx_input_ids,
                                          attention_mask=hf_ctx_attn_mask)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :].float()

        # Compare logits.
        torch.testing.assert_close(ref, res, atol=1e-2, rtol=1e-1)

        # 2. Check the correctness of generation step.

        gen_id = torch.randint(100, (batch_size, 1)).int().to(device)
        gen_context_lengths = ctx_context_lengths.clone()
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32)
        gen_position_ids = torch.full_like(gen_id, input_len)
        if remove_input_padding:
            gen_last_token_ids = torch.arange(1,
                                              1 + batch_size).int().to(device)
        else:
            gen_last_token_ids = torch.zeros_like(gen_context_lengths)
        gen_attention_mask = torch.cat([
            ctx_attention_mask,
            ctx_attention_mask.new_ones((ctx_attention_mask.shape[0], 1))
        ],
                                       dim=-1)

        # past kv length: sequence_length of last step
        host_past_key_value_lengths = sequence_length.cpu()

        # For step 1, the sequence_lengths = context_lengths + 1.
        sequence_length = torch.add(sequence_length, 1)

        step1_buffer = {
            'input_ids': gen_id,
            'context_lengths': gen_context_lengths.contiguous(),
            'position_ids': gen_position_ids.contiguous(),
            'last_token_ids': gen_last_token_ids.contiguous(),
            'attention_mask': gen_attention_mask.contiguous(),
            'host_request_types': gen_host_request_types.contiguous(),
            'cache_indirection': cache_indirections[1],
            'sequence_length': sequence_length.contiguous(),
            'host_past_key_value_lengths':
            host_past_key_value_lengths.contiguous(),
            'host_sink_token_length': host_sink_token_length,
        }
        if remove_input_padding:
            step1_buffer['host_context_lengths'] = gen_context_lengths.cpu()
        for i in range(hf_config.num_hidden_layers):
            kv_cache = ctx_buffer[f'present_key_value_{i}']
            step1_buffer[f'past_key_value_{i}'] = kv_cache
            if use_gpt_attengion_plugin:
                # gpt_attention_plugin shares past/present cache.
                step1_buffer[f'present_key_value_{i}'] = kv_cache
                step1_buffer[
                    f'host_max_attention_window_size_{i}'] = host_max_attention_window_sizes
        step1_shape = {k: v.shape for k, v in step1_buffer.items()}

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits'].float()

        with torch.no_grad():
            hf_gen_attn_mask = torch.cat([
                hf_ctx_attn_mask,
                hf_ctx_attn_mask.new_ones((hf_ctx_attn_mask.shape[0], 1))
            ],
                                         dim=-1)
            hf_outputs = hf_model.forward(
                gen_id,
                attention_mask=hf_gen_attn_mask,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :].float()

        torch.testing.assert_close(ref, res, atol=1e-2, rtol=1e-1)

    @parameterized.expand(load_test_cases())
    def test_greedy_search(self, query_type, use_alibi, parallel_attention,
                           new_decoder_architecture, use_refit,
                           use_gpt_attengion_plugin, use_gemm_plugin,
                           remove_input_padding, context_fmha_type, dtype):

        self.skip_test_case(query_type, use_alibi, parallel_attention,
                            new_decoder_architecture, use_refit,
                            use_gpt_attengion_plugin, use_gemm_plugin,
                            remove_input_padding, context_fmha_type, dtype)

        model_name = 'falcon'
        world_size = 1
        rank = 0
        batch_size = 3
        beam_width = 1
        input_len = 7
        output_len = 4
        log_level = 'error'

        hf_config, hf_model = self.generate_hf_model(
            output_len=output_len,
            dtype=dtype,
            query_type=query_type,
            use_alibi=use_alibi,
            parallel_attention=parallel_attention,
            new_decoder_architecture=new_decoder_architecture)
        _, engine_buffer = self.generate_trtllm_runtime(
            model_name=model_name,
            hf_config=hf_config,
            hf_model=hf_model,
            dtype=dtype,
            world_size=world_size,
            rank=rank,
            batch_size=batch_size,
            beam_width=beam_width,
            input_len=input_len,
            output_len=output_len,
            use_refit=use_refit,
            use_gpt_attengion_plugin=use_gpt_attengion_plugin,
            use_gemm_plugin=use_gemm_plugin,
            enable_remove_input_padding=remove_input_padding,
            context_fmha_type=context_fmha_type,
            log_level=log_level)
        device = hf_model.device

        model_config = ModelConfig(
            model_name=model_name,
            vocab_size=hf_config.vocab_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_kv_heads,
            hidden_size=hf_config.hidden_size,
            gpt_attention_plugin=use_gpt_attengion_plugin,
            dtype=dtype)

        sampling_config = SamplingConfig(end_id=hf_config.eos_token_id,
                                         pad_id=hf_config.pad_token_id,
                                         num_beams=1,
                                         temperature=1.0,
                                         top_k=1,
                                         top_p=0.0,
                                         length_penalty=1.0,
                                         repetition_penalty=1.0)

        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)
        decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                         engine_buffer,
                                                         mapping,
                                                         debug_mode=True)

        input_ids, context_lengths, _ = self.prepare_input_token_ids(
            batch_size,
            input_len,
            vocab_size=hf_config.vocab_size,
            # Skip testing padded inputs due to bugs in HF Falcon.
            # Will enable when those are fixed.
            pad_id=None,
            remove_input_padding=remove_input_padding,
            device=device)

        decoder.setup(batch_size,
                      max_context_length=input_len,
                      max_new_tokens=output_len)

        output_ids = decoder.decode(input_ids, context_lengths, sampling_config)
        # TODO: change to actual ragged tensor after the plugin supports it
        output_ids_x = decoder.decode(input_ids, context_lengths,
                                      sampling_config)
        torch.cuda.synchronize()
        torch.testing.assert_close(output_ids, output_ids_x)

        # Convert to left padding to match with HF's padding policy.
        res = self.convert_to_left_padding(output_ids[:, 0, :],
                                           sampling_config.end_id)

        ref_output_ids = hf_model.generate(
            self.convert_to_left_padding(input_ids, sampling_config.pad_id),
            do_sample=False,
            early_stopping=False,
            num_beams=sampling_config.num_beams,
            temperature=sampling_config.temperature,
            top_k=sampling_config.top_k,
            top_p=sampling_config.top_p,
            max_new_tokens=output_len,
            length_penalty=sampling_config.length_penalty,
            repetition_penalty=sampling_config.repetition_penalty,
            pad_token_id=sampling_config.pad_id,
            eos_token_id=sampling_config.end_id)
        torch.cuda.synchronize()
        ref = ref_output_ids.int()

        torch.testing.assert_close(res[:, -output_len:], ref[:, -output_len:])


if __name__ == '__main__':
    unittest.main()
