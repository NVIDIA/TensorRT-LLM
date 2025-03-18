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

import math
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from parameterized import parameterized
from transformers import AutoConfig, AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm import Builder, Mapping
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt
from tensorrt_llm.models.deepseek_v2.convert import (convert_deepseekv2,
                                                     create_trt_config_from_hf)
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime.kv_cache_manager import GenerationSequence
from tensorrt_llm.runtime.memory_pools.memory_pools_allocator import \
    MemoryPoolsAllocator
from tensorrt_llm.runtime.memory_pools.pools_kv_cache_manager import \
    PoolsKVCacheManager

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import unittest_name_func


def compare_abs_error(ref, res, str):
    # calculate max abs error
    compare_HF = ref.cpu().numpy().flatten()
    compare_TRT_LLM = res.cpu().numpy().flatten()
    max_abs_error = np.max(abs(compare_TRT_LLM - compare_HF))
    min_abs_error = np.min(abs(compare_TRT_LLM - compare_HF))
    print(str, "max abs error = ", max_abs_error, "min abs error = ",
          min_abs_error)


class TestDeepSeek(unittest.TestCase):

    def setUp(self):
        super().setUp()
        # Fix random seed for the reproducibility.
        torch.random.manual_seed(1773)

    def _gen_tensorrt_llm_deepseek(self, hf_path, hf_deepseek, num_layers,
                                   dtype, mapping):

        tensorrt_llm.models.deepseek_v2.convert.OVERRIDE_HIDDEN_LAYERS = num_layers
        config = create_trt_config_from_hf(str(hf_path), dtype, mapping)
        pretrained_config = tensorrt_llm.models.PretrainedConfig.from_dict(
            config)
        tensorrt_llm_deepseek = tensorrt_llm.models.DeepseekV2ForCausalLM.from_config(
            pretrained_config)
        weights = convert_deepseekv2(hf_deepseek, config, mapping, dtype)
        tensorrt_llm_deepseek.load(weights)
        return tensorrt_llm_deepseek

    # TODO: merge `_gen_tensorrt_llm_deepseek` and `_gen_tensorrt_llm_network` to one function,
    # separate for debug purpose
    def _gen_tensorrt_llm_network(self, network, hf_path, hf_deepseek,
                                  num_layers, dtype, mapping, batch_size,
                                  input_len, output_len):

        tensorrt_llm_deepseek = self._gen_tensorrt_llm_deepseek(
            hf_path, hf_deepseek, num_layers, dtype, mapping)
        with net_guard(network):
            str_dtype_to_trt(dtype)
            network.set_named_parameters(
                tensorrt_llm_deepseek.named_parameters())
            inputs = tensorrt_llm_deepseek.prepare_inputs(
                max_batch_size=batch_size,
                max_input_len=input_len,
                max_seq_len=input_len + output_len,
                max_num_tokens=batch_size * input_len,
                use_cache=True)
            # Prepare
            tensorrt_llm_deepseek(**inputs)
        return network

    # TODO: merge `_gen_tensorrt_llm_engine` and `_gen_tensorrt_llm_network` to one function,
    # separate for debug purpose
    def _gen_tensorrt_llm_engine(self, model_name, hf_path, hf_deepseek,
                                 num_layers, dtype, mapping, batch_size,
                                 input_len, output_len):
        builder = Builder()
        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name=model_name,
                precision=dtype,
                timing_cache='model.cache',
            )
            network = builder.create_network()
            network.plugin_config.to_legacy_setting()
            network.plugin_config.use_paged_context_fmha = True
            network.plugin_config.gemm_plugin = dtype
            network.plugin_config.gpt_attention_plugin = dtype
            network.plugin_config.remove_input_padding = True
            network.plugin_config.paged_kv_cache = True
            network.plugin_config.context_fmha = True
            # trtllm v0.16 no longer supports enable_xqa config
            # network.plugin_config.enable_xqa = True
            network.plugin_config.use_fused_mlp = True

            self._gen_tensorrt_llm_network(network, hf_path, hf_deepseek,
                                           num_layers, dtype, mapping,
                                           batch_size, input_len, output_len)

            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(self, log_level, model_name, hf_path,
                                  hf_deepseek, num_layers, dtype, mapping,
                                  batch_size, input_len, output_len):
        tensorrt_llm.logger.set_level(log_level)
        engine_buffer = self._gen_tensorrt_llm_engine(model_name, hf_path,
                                                      hf_deepseek, num_layers,
                                                      dtype, mapping,
                                                      batch_size, input_len,
                                                      output_len)
        runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)
        return runtime, engine_buffer

    @parameterized.expand(['DeepSeek-V2'], name_func=unittest_name_func)
    def test_deepseek_v2(self, path):
        # Use local model root path for testing instead of trtllm pytest env
        model_root = llm_models_root()
        # local path for testing
        # model_root = Path("/scratch/model/")
        if model_root is None:
            pytest.skip("Skipping since real weights are unavailable.")
        hf_path = Path(model_root, path)
        if not hf_path.exists():
            pytest.skip(f"Skipping since the path {hf_path} does not exist.")

        torch.manual_seed(0)
        random.seed(0)

        dtype = 'bfloat16'
        model_name = 'deepseek-v2'
        log_level = 'error'
        num_layers = 1
        batch_size = 4
        max_len = 2  # output_len
        seq_len = 128  # input_len
        total_len = seq_len + max_len  # 130
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        beam_width = 1
        tokens_per_block = 64  # fixed at 64 for now, 128 causes illegal memory access
        # for deepseek-v2
        rope_dim = 64
        c_k_dim = 512

        # get hf model
        hf_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
        hf_config.num_hidden_layers = num_layers
        # print(f"hf_config: {hf_config}")
        hf_deepseek = AutoModelForCausalLM.from_pretrained(
            hf_path,
            config=hf_config,
            device_map='auto',
            torch_dtype=str_dtype_to_torch(dtype),
            trust_remote_code=True)

        # get tensorrt-llm deepseek runtime
        runtime, engine_buffer = self._gen_tensorrt_llm_runtime(
            log_level, model_name, hf_path, hf_deepseek, num_layers, dtype,
            mapping, batch_size, seq_len, max_len)

        # compare context
        # inputs:
        # generate random context ids with shape [4, 128] and values in range [0, 100)
        ctx_ids = torch.randint(100, (batch_size, seq_len),
                                dtype=torch.int32,
                                device='cuda')

        # ref: run hf model forward with ctx_ids
        # hf_outputs.logits output shape: [4, 128, 102400], 102400 is the vocab size
        # -1 means the last token, so shape -> [4, 102400]
        with torch.no_grad():
            hf_outputs = hf_deepseek.forward(ctx_ids)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        # res: run tensorrt llm runtime forward with ctx_ids
        # generate position ids with shape [4, 16], values from 0 to 15 at each row
        ctx_position_ids = torch.tensor(range(seq_len),
                                        dtype=torch.int32).reshape([
                                            1, seq_len
                                        ]).expand([batch_size, seq_len]).cuda()
        # generate context lengths with shape [4], value [128, 128, 128, 128]
        ctx_context_lengths = seq_len * torch.ones(
            batch_size, dtype=torch.int32, device='cuda')
        ctx_last_token_ids = ctx_context_lengths.clone()

        # remove input padding
        # ctx_ids shape: [4, 128] -> [512]
        ctx_ids = ctx_ids.view([batch_size * seq_len])
        ctx_position_ids = ctx_position_ids.view([batch_size * seq_len])
        # ctx_last_token_ids shape: [4], value [128, 256, 384, 512]
        ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()
        # host_max_attention_window_sizes shape: [1], value [130]
        host_max_attention_window_sizes = torch.tensor(
            [total_len] * hf_config.num_hidden_layers, dtype=torch.int32)
        # host_sink_token_length shape: [1], value [0]
        host_sink_token_length = torch.tensor([0], dtype=torch.int32)
        host_context_lengths = ctx_context_lengths.cpu()
        # host_request_types shape: [4], value [0, 0, 0, 0]
        host_request_types = torch.tensor([0 for i in range(batch_size)],
                                          dtype=torch.int32).cpu()
        host_past_key_value_lengths = ctx_context_lengths.detach().clone().cpu()
        sequence_length = ctx_context_lengths.detach().clone()
        # context_runtime_perf_knobs shape: [16],
        # value [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        perf_knob_tensor_size = 16
        context_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                                  dtype=torch.int64)
        # host_context_progress shape: [1], value [0]
        host_context_progress = torch.tensor([0], dtype=torch.int64)

        cache_indirections = [
            torch.zeros((batch_size, beam_width, total_len),
                        dtype=torch.int32,
                        device='cuda'),
            torch.zeros((batch_size, beam_width, total_len),
                        dtype=torch.int32,
                        device='cuda')
        ]  # ping-pong buffers

        max_blocks_per_seq = math.ceil(total_len / tokens_per_block)
        num_blocks = batch_size * beam_width * max_blocks_per_seq
        ctx_context_lengths.detach().clone()
        # for deepseek-v2, head_size = c_k_dim + rope_dim
        memory_pools_allocator = MemoryPoolsAllocator(
            num_blocks=num_blocks,
            tokens_per_block=tokens_per_block,
            head_size=c_k_dim + rope_dim)
        # for deepseek-v2, num_kv_heads_per_layer = 1
        num_kv_heads_per_layer = MemoryPoolsAllocator.prepare_num_kv_heads_per_layer(
            1, hf_config.num_hidden_layers)
        memory_pools_allocator.allocate(dtype, num_kv_heads_per_layer)
        # for deepseek-v2, head_size = c_k_dim + rope_dim
        pools_kv_cache_manager = PoolsKVCacheManager(
            memory_pools_allocator.pools_metadata,
            max_blocks_per_seq,
            num_blocks,
            tokens_per_block,
            c_k_dim + rope_dim,
            max_attention_window_size=total_len,
            beam_width=beam_width,
            sink_token_len=0)
        host_kv_cache_pool_pointers = memory_pools_allocator.get_kv_cache_pool_pointers(
        )

        # Add sequences to the manager
        for bi in range(batch_size):
            generation_sequence = GenerationSequence(seq_idx=bi, batch_idx=bi)
            pools_kv_cache_manager.add_sequence(generation_sequence, seq_len)
        pools_kv_cache_manager.step([False] * batch_size)

        # ctx_buffer: dict
        ctx_buffer = {
            'input_ids': ctx_ids,
            'context_lengths': ctx_context_lengths,
            'host_request_types': host_request_types,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'cache_indirection': cache_indirections[0],
            'host_past_key_value_lengths': host_past_key_value_lengths,
            'sequence_length': sequence_length,
            'host_sink_token_length': host_sink_token_length,
            'host_runtime_perf_knobs': context_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
            'host_context_lengths': host_context_lengths,
        }

        assert beam_width == 1
        # for deepseek-v2
        # TODO: check if this is correct and why use get_single_kv_cache_manager()
        host_kv_manager = pools_kv_cache_manager.get_single_kv_cache_manager()
        host_kv_cache_block_offsets = host_kv_manager.get_block_offsets(
            beam_width=1)
        # host_kv_cache_block_offsets = pools_kv_cache_manager.get_block_offsets(
        #       beam_width=1)
        kv_cache_block_offsets = host_kv_cache_block_offsets.to('cuda')
        # new shape
        kv_cache_block_offsets = kv_cache_block_offsets.reshape(
            1, batch_size, 2, max_blocks_per_seq)
        shape = kv_cache_block_offsets.shape
        ctx_buffer['kv_cache_block_offsets'] = kv_cache_block_offsets.reshape(
            shape).contiguous()
        ctx_buffer[
            'host_kv_cache_block_offsets'] = host_kv_cache_block_offsets.reshape(
                shape).contiguous()
        ctx_buffer[
            'host_kv_cache_pool_pointers'] = host_kv_cache_pool_pointers.contiguous(
            )
        ctx_buffer[
            'host_kv_cache_pool_mapping'] = memory_pools_allocator.pool_mapping.contiguous(
            )
        ctx_buffer[
            'host_max_attention_window_sizes'] = host_max_attention_window_sizes
        # ctx_shape: dict
        ctx_shape = {key: buffer.shape for key, buffer in ctx_buffer.items()}

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-1)

        compare_abs_error(ref, res, "context logits")

        # compare generation
        # hf_outputs = None
        step = 1
        # gen_ids = [[64], [30], [43], [74]], shape: [4, 1]
        gen_ids = torch.randint(100, (batch_size, 1),
                                dtype=torch.int32,
                                device='cuda')
        # gen_context_lengths = [128, 128, 128, 128], shape: [4]
        gen_context_lengths = seq_len * torch.ones(
            batch_size, dtype=torch.int32, device='cuda')
        # gen_position_ids = [[128], [128], [128], [128]], shape: [4, 1]
        gen_position_ids = torch.ones_like(gen_ids).int().cuda() * seq_len
        # gen_last_token_ids = [0, 0, 0, 0], shape: [4]
        # gen_last_token_ids = torch.zeros_like(gen_context_lengths).int().cuda()

        # deepseek-v2 attention mask
        # deepseek-v2 attention mask shape: [4, 128]
        deepseek_v2_attention_mask = torch.ones((batch_size, seq_len),
                                                dtype=torch.int32,
                                                device='cuda')
        for i in range(batch_size):
            deepseek_v2_attention_mask[i, gen_context_lengths[i]:seq_len] = 0

        deepseek_v2_attention_mask = torch.cat(
            (deepseek_v2_attention_mask,
             deepseek_v2_attention_mask.new_ones((batch_size, 1))),
            dim=-1).contiguous()
        from transformers.modeling_attn_mask_utils import (
            AttentionMaskConverter, _prepare_4d_attention_mask)
        attention_mask = _prepare_4d_attention_mask(
            deepseek_v2_attention_mask,
            dtype=str_dtype_to_torch(dtype),
            tgt_len=1)

        attention_mask = attention_mask + AttentionMaskConverter._make_causal_mask(
            gen_ids.shape,
            dtype=str_dtype_to_torch(dtype),
            device='cuda',
            past_key_values_length=seq_len + step - 1)

        with torch.no_grad():
            hf_outputs = hf_deepseek.forward(
                gen_ids,
                attention_mask=attention_mask,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        # logits have shape [batch_size, seq_len, vocab_size]
        ref_gen = hf_outputs.logits[:, -1, :]

        # remove input padding
        # gen_ids shape: [4, 1] -> [4], value [64, 30, 43, 74]
        gen_ids = gen_ids.view([batch_size])
        # gen_position_ids shape: [4, 1] -> [4], value [128, 128, 128, 128]
        gen_position_ids = gen_position_ids.view([batch_size])
        # gen_last_token_ids shape: [4], value [1, 1, 1, 1] -> [1, 2, 3, 4]
        gen_last_token_ids = torch.ones_like(gen_context_lengths).int().cuda()
        gen_last_token_ids = torch.cumsum(gen_last_token_ids, dim=0).int()
        # host_past_key_value_lengths shape: [4], value [128, 128, 128, 128]
        host_past_key_value_lengths = torch.tensor([seq_len + step - 1] *
                                                   batch_size,
                                                   dtype=torch.int32)
        # host_max_attention_window_sizes shape: [1], value [129]
        host_max_attention_window_sizes = torch.tensor(
            [seq_len + step] * hf_config.num_hidden_layers, dtype=torch.int32)
        # host_sink_token_length shape: [1], value [0]
        host_sink_token_length = torch.tensor([0], dtype=torch.int32)
        # host_context_lengths shape: [4], value [128, 128, 128, 128]
        host_context_lengths = gen_context_lengths.cpu()
        # host_request_types shape: [4], value [1, 1, 1, 1]
        host_request_types = torch.tensor([1 for i in range(batch_size)],
                                          dtype=torch.int32).cpu()
        # sequence_length shape: [4], value [129, 129, 129, 129]
        sequence_length = torch.add(gen_context_lengths.detach().clone(), 1)
        # gen_runtime_perf_knobs shape: [16], value [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        perf_knob_tensor_size = 16
        gen_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                              dtype=torch.int64)
        # host_context_progress shape: [1], value [0]
        host_context_progress = torch.tensor([0], dtype=torch.int64)

        gen_buffer = {
            'input_ids': gen_ids,
            'context_lengths': gen_context_lengths,
            'host_request_types': host_request_types,
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'cache_indirection': cache_indirections[1],
            'host_past_key_value_lengths': host_past_key_value_lengths,
            'sequence_length': sequence_length,
            'host_sink_token_length': host_sink_token_length,
            'host_runtime_perf_knobs': gen_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
            'host_context_lengths': host_context_lengths,
        }

        gen_buffer['kv_cache_block_offsets'] = kv_cache_block_offsets.reshape(
            shape).contiguous()
        gen_buffer[
            'host_kv_cache_block_offsets'] = host_kv_cache_block_offsets.reshape(
                shape).contiguous()
        gen_buffer[
            'host_kv_cache_pool_pointers'] = host_kv_cache_pool_pointers.contiguous(
            )
        gen_buffer[
            'host_kv_cache_pool_mapping'] = memory_pools_allocator.pool_mapping.contiguous(
            )
        gen_buffer[
            'host_max_attention_window_sizes'] = host_max_attention_window_sizes

        # add key_value_cache_buffers
        key_value_cache_buffers = []
        plugin_kv_num_heads = 1
        cache_shape = (num_blocks, 2, plugin_kv_num_heads, tokens_per_block,
                       c_k_dim + rope_dim)
        for _ in range(hf_config.num_hidden_layers):
            key_value_cache_buffers.append(
                torch.zeros(cache_shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'))

        for i in range(hf_config.num_hidden_layers):
            gen_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            gen_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        # gen_shape: dict
        gen_shape = {key: buffer.shape for key, buffer in gen_buffer.items()}

        context = runtime.context_1
        runtime._set_shape(context, gen_shape)
        runtime._set_buffer(context, gen_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res_gen = gen_buffer['logits']

        # TRT-LLM engine logits has larger variance
        compare_abs_error(ref_gen, res_gen, "generation logits")

        # compare softmax and argmax
        # HF
        ref_softmax = F.softmax(ref_gen, dim=-1)
        ref_next_token = torch.argmax(ref_softmax, dim=-1)

        # TRT-LLM
        res_softmax = F.softmax(res_gen, dim=-1)
        res_next_token = torch.argmax(res_softmax, dim=-1)

        compare_abs_error(ref_next_token, res_next_token,
                          "generation next token")

        np.testing.assert_allclose(ref_next_token.cpu().numpy(),
                                   res_next_token.cpu().numpy(),
                                   atol=1e-1)


if __name__ == '__main__':
    unittest.main()
