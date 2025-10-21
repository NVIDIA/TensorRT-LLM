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
import random
import tempfile
import unittest
from argparse import Namespace
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

# isort: off
import torch
# isort: on
from gpt.convert_checkpoint import convert_and_save_hf
from parameterized import parameterized
from transformers import GPT2Config, GPT2LMHeadModel
from utils.llm_data import llm_models_root
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.functional import RotaryScalingType
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.models.gpt.convert import load_weights_from_hf_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.generation import _prepare_attention_mask
from tensorrt_llm.runtime.kv_cache_manager import GenerationSequence
from tensorrt_llm.runtime.memory_pools.memory_pools_allocator import \
    MemoryPoolsAllocator
from tensorrt_llm.runtime.memory_pools.pools_kv_cache_manager import \
    PoolsKVCacheManager


class TestGPT(unittest.TestCase):

    def _gen_hf_gpt(self, hidden_act, n_layer, max_length, dtype):
        gpt_config = GPT2Config(
            activation_function=hidden_act,
            n_layer=n_layer,
            max_length=max_length,
            dtype=dtype,
        )
        gpt_config.n_kv_head = gpt_config.n_head
        hf_gpt = GPT2LMHeadModel(gpt_config).cuda().eval()
        return gpt_config, hf_gpt

    def _gen_tensorrt_llm_network(self, network, builder, hf_gpt, gpt_config,
                                  batch_size, input_len, output_len, dtype,
                                  gpt_attention_plugin, tensor_parallel,
                                  apply_query_key_layer_scaling,
                                  gather_context_logits):
        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': dtype,
            'num_hidden_layers': gpt_config.n_layer,
            'num_attention_heads': gpt_config.n_head,
            'num_key_value_heads': gpt_config.n_head,
            'hidden_size': gpt_config.n_embd,
            'intermediate_size': gpt_config.n_embd * 4,
            'norm_epsilon': 1e-5,
            'vocab_size': gpt_config.vocab_size,
            'position_embedding_type': 'learned_absolute',
            'max_position_embeddings': gpt_config.n_positions,
            'hidden_act': gpt_config.activation_function,
            'mapping': {
                'world_size': tensor_parallel,
                'tp_size': tensor_parallel,
            },
            'bias': getattr(gpt_config, 'bias', True),
            'apply_query_key_layer_scaling': apply_query_key_layer_scaling,
        }
        config = tensorrt_llm.models.GPTConfig.from_dict(config)
        weights = load_weights_from_hf_model(hf_gpt, config)

        tensorrt_llm_gpt = tensorrt_llm.models.GPTForCausalLM(config)
        tensorrt_llm_gpt.load(weights)

        with net_guard(network):
            # Initialize model
            network.set_named_parameters(tensorrt_llm_gpt.named_parameters())
            inputs = tensorrt_llm_gpt.prepare_inputs(
                max_batch_size=batch_size,
                max_input_len=input_len,
                max_seq_len=input_len + output_len,
                max_num_tokens=batch_size * input_len,
                use_cache=True,
                max_beam_width=1,
                gather_context_logits=gather_context_logits)

            # Prepare
            tensorrt_llm_gpt(**inputs)

        return network

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  gpt_config,
                                  hf_gpt,
                                  model,
                                  use_plugin,
                                  batch_size,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  fast_building=False,
                                  apply_query_key_layer_scaling=False,
                                  context_fmha_type=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False,
                                  enable_paged_kv_cache=False,
                                  tokens_per_block=128,
                                  gather_context_logits=False,
                                  gather_generation_logits=False):
        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)

        runtime = None
        builder = Builder()

        with tempfile.TemporaryDirectory() as tmpdirname:

            builder_config = builder.create_builder_config(
                name='gpt',
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
                gather_context_logits=gather_context_logits,
                gather_generation_logits=gather_generation_logits,
                strongly_typed=True,
            )
            network = builder.create_network()
            network.plugin_config.to_legacy_setting()
            if use_plugin:
                network.plugin_config.gpt_attention_plugin = dtype
            if fast_building:
                network.plugin_config.gemm_plugin = dtype
            network.plugin_config.set_context_fmha(context_fmha_type)
            if enable_remove_input_padding:
                network.plugin_config.remove_input_padding = True
            if enable_paged_kv_cache:
                network.plugin_config.enable_paged_kv_cache(tokens_per_block)

            self._gen_tensorrt_llm_network(network, builder, hf_gpt, gpt_config,
                                           batch_size, input_len, output_len,
                                           dtype, use_plugin, world_size,
                                           apply_query_key_layer_scaling,
                                           gather_context_logits)

            engine_buffer = builder.build_engine(network, builder_config)
            runtime = tensorrt_llm.runtime.generation._Runtime(
                engine_buffer, mapping)
        return runtime, engine_buffer

    @parameterized.expand([("other", False)], name_func=unittest_name_func)
    def test_gpt_float32(self, test_partition, use_refit):
        torch.manual_seed(42)

        model = 'gpt'
        log_level = 'error'
        dtype = 'float32'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 2
        max_length = 2
        batch_size = 4
        beam_width = 1
        seq_len = 128
        total_length = seq_len + max_length
        use_plugin = False

        gpt_config, hf_gpt = self._gen_hf_gpt(hidden_act, n_layer, max_length,
                                              dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
            use_plugin, batch_size, seq_len, max_length, use_refit)

        # compare context
        pad_token_id = 50256
        ctx_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        ctx_ids[0][-1] = pad_token_id
        ctx_ids[1][-3:] = pad_token_id
        ctx_ids[2][-5:] = pad_token_id
        ctx_context_lengths = seq_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_host_context_lengths = ctx_context_lengths.cpu()
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32,
                                              device='cpu')
        ctx_position_ids = torch.tensor(range(seq_len),
                                        dtype=torch.int32).reshape([
                                            1, seq_len
                                        ]).expand([batch_size, seq_len]).cuda()
        ctx_last_token_ids = ctx_context_lengths.clone()
        ctx_attention_mask = _prepare_attention_mask(ctx_ids)

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

        perf_knob_tensor_size = 16
        context_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                                  dtype=torch.int64)
        host_context_progress = torch.tensor([0], dtype=torch.int64)

        ctx_buffer = {
            'input_ids': ctx_ids,
            'position_ids': ctx_position_ids,
            'context_lengths': ctx_context_lengths,
            'host_context_lengths': ctx_host_context_lengths,
            'last_token_ids': ctx_last_token_ids,
            'attention_mask': ctx_attention_mask,
            'host_request_types': ctx_host_request_types,
            'cache_indirection': cache_indirections[0],
            'host_runtime_perf_knobs': context_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}
        for i in range(gpt_config.n_layer):
            shape = (batch_size, 2, gpt_config.n_head, 0,
                     gpt_config.n_embd // gpt_config.n_head)
            past_buffer = torch.zeros((1, ),
                                      dtype=str_dtype_to_torch(dtype),
                                      device='cuda')
            ctx_shape.update({
                f'past_key_value_{i}': shape,
            })
            shape = (batch_size, 2, gpt_config.n_head, seq_len,
                     gpt_config.n_embd // gpt_config.n_head)
            ctx_buffer.update({
                f'past_key_value_{i}':
                past_buffer,
                f'present_key_value_{i}':
                torch.zeros(shape,
                            dtype=str_dtype_to_torch(dtype),
                            device='cuda'),
            })

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(ctx_ids,
                                        attention_mask=ctx_attention_mask)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]
        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-2)

        for i in range(gpt_config.n_layer):
            res_present_key_value = ctx_buffer[f'present_key_value_{i}']
            ref_present_key, ref_present_value = hf_outputs.past_key_values[i]

            past_key_value_tensor = res_present_key_value.permute(1, 0, 2, 3, 4)
            key, value = past_key_value_tensor.chunk(2)

            head_size = gpt_config.n_embd // gpt_config.n_head
            key = key.to(torch.float32).reshape(batch_size, gpt_config.n_head,
                                                seq_len, head_size)
            value = value.reshape(batch_size, gpt_config.n_head, seq_len,
                                  head_size)

            np.testing.assert_allclose(ref_present_key.cpu().numpy(),
                                       key.cpu().numpy(),
                                       atol=1e-2)

            np.testing.assert_allclose(ref_present_value.cpu().numpy(),
                                       value.cpu().numpy(),
                                       atol=1e-2)

        # compare generation
        gen_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_context_lengths = ctx_context_lengths.clone()
        gen_host_context_lengths = ctx_host_context_lengths.clone()
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32,
                                              device='cpu')
        gen_position_ids = torch.ones_like(gen_id).cuda() * seq_len
        gen_last_token_ids = torch.zeros_like(gen_context_lengths).cuda()
        gen_attention_mask = torch.cat([
            ctx_attention_mask,
            ctx_attention_mask.new_ones((ctx_attention_mask.shape[0], 1))
        ],
                                       dim=-1)

        gen_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                              dtype=torch.int64)
        step1_shape = {
            'input_ids': gen_id.shape,
            'context_lengths': gen_context_lengths.shape,
            'host_context_lengths': gen_host_context_lengths.shape,
            'host_request_types': gen_host_request_types.shape,
            'position_ids': gen_position_ids.shape,
            'last_token_ids': gen_last_token_ids.shape,
            'attention_mask': gen_attention_mask.shape,
            'cache_indirection': cache_indirections[1].shape,
            'host_runtime_perf_knobs': gen_runtime_perf_knobs.shape,
            'host_context_progress': host_context_progress.shape,
        }
        step1_buffer = {
            'input_ids': gen_id,
            'context_lengths': gen_context_lengths.contiguous(),
            'host_context_lengths': gen_host_context_lengths.contiguous(),
            'host_request_types': gen_host_request_types.contiguous(),
            'position_ids': gen_position_ids.contiguous(),
            'last_token_ids': gen_last_token_ids.contiguous(),
            'attention_mask': gen_attention_mask.contiguous(),
            'cache_indirection': cache_indirections[1].contiguous(),
            'host_runtime_perf_knobs': gen_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        for i in range(gpt_config.n_layer):
            shape = (batch_size, 2, gpt_config.n_head, seq_len,
                     gpt_config.n_embd // gpt_config.n_head)
            step1_shape.update({
                f'past_key_value_{i}': shape,
            })
            step1_buffer.update({
                f'past_key_value_{i}':
                ctx_buffer[f'present_key_value_{i}'],
            })

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        with torch.no_grad():
            hf_outputs = hf_gpt.forward(
                gen_id,
                attention_mask=gen_attention_mask,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        np.testing.assert_allclose(ref.cpu().numpy(),
                                   res.cpu().numpy(),
                                   atol=1e-2)

        for i in range(gpt_config.n_layer):
            res_present_key_value = step1_buffer[f'present_key_value_{i}']

            ref_present_key, ref_present_value = hf_outputs.past_key_values[i]

            past_key_value_tensor = res_present_key_value.permute(1, 0, 2, 3, 4)
            key, value = past_key_value_tensor.chunk(2)

            head_size = gpt_config.n_embd // gpt_config.n_head
            key = key.reshape(batch_size, gpt_config.n_head, seq_len + 1,
                              head_size)
            value = value.reshape(batch_size, gpt_config.n_head, seq_len + 1,
                                  head_size)

            np.testing.assert_allclose(ref_present_key.cpu().numpy(),
                                       key.cpu().numpy(),
                                       atol=1e-2)

            np.testing.assert_allclose(ref_present_value.cpu().numpy(),
                                       value.cpu().numpy(),
                                       atol=1e-2)

    def load_test_cases():
        test_cases = list(
            product([False, True], [False, True], [False, True], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], [False, True], [False, True], [False, True], [False, True]))
        # split test cases into 4 partitions
        test_cases = [(f"partition{int(i % 4)}", ) + case
                      for i, case in enumerate(test_cases)]

        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_gpt_plugin(self, test_partition, use_refit, fast_building,
                        apply_query_key_layer_scaling, context_fmha_type,
                        enable_remove_input_padding, enable_paged_kv_cache,
                        gather_context_logits, gather_generation_logits):
        # inflight batching mode only works with remove_input_padding and paged_kv_cache
        use_in_flight_batching = enable_remove_input_padding and enable_paged_kv_cache and not (
            gather_context_logits or gather_generation_logits)

        torch.manual_seed(0)
        random.seed(0)

        model = 'gpt'
        log_level = 'error'
        dtype = 'float16'
        world_size = 1
        rank = 0
        hidden_act = 'gelu'
        n_layer = 1
        max_length = 2
        batch_size = 4
        beam_width = 1
        seq_len = 128
        total_length = seq_len + max_length
        use_plugin = True
        tokens_per_block = 128
        gpt_config, hf_gpt = self._gen_hf_gpt(hidden_act, n_layer,
                                              seq_len + max_length, dtype)
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
            use_plugin, batch_size, seq_len, max_length, use_refit,
            fast_building, apply_query_key_layer_scaling, context_fmha_type,
            enable_remove_input_padding, enable_paged_kv_cache,
            tokens_per_block, gather_context_logits, gather_generation_logits)
        key_value_cache_buffers = []
        value_cache_buffers = []
        head_size = gpt_config.n_embd // gpt_config.n_head

        if enable_paged_kv_cache:
            num_blocks = batch_size * beam_width * math.ceil(
                total_length / tokens_per_block)
            cache_shape = (
                num_blocks,
                gpt_config.n_layer,
                2,
                gpt_config.n_head,
                tokens_per_block,
                head_size,
            )
            key_value_cache_buffers.append(
                torch.zeros(cache_shape,
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))
        else:
            cache_shape = (
                batch_size,
                2,
                gpt_config.n_head,
                total_length,
                head_size,
            )
            for _ in range(gpt_config.n_layer):
                key_value_cache_buffers.append(
                    torch.zeros(
                        cache_shape,
                        dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                        device='cuda'))

        for _ in range(gpt_config.n_layer):
            value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    gpt_config.n_head,
                    total_length,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))

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

        if enable_paged_kv_cache:
            max_blocks_per_seq = math.ceil(total_length / tokens_per_block)
            num_blocks = batch_size * beam_width * max_blocks_per_seq

            memory_pools_allocator = MemoryPoolsAllocator(
                num_blocks=num_blocks,
                tokens_per_block=tokens_per_block,
                head_size=head_size)
            num_kv_heads_per_layer = MemoryPoolsAllocator.prepare_num_kv_heads_per_layer(
                gpt_config.n_head, gpt_config.n_layer)
            memory_pools_allocator.allocate(dtype, num_kv_heads_per_layer)
            pools_kv_cache_manager = PoolsKVCacheManager(
                memory_pools_allocator.pools_metadata,
                max_blocks_per_seq,
                num_blocks,
                tokens_per_block,
                head_size,
                max_attention_window_size=total_length,
                beam_width=beam_width,
                sink_token_len=0)

            host_kv_cache_pool_pointers = memory_pools_allocator.get_kv_cache_pool_pointers(
            )
            host_kv_cache_pool_mapping = memory_pools_allocator.pool_mapping

            # block_size = gpt_config.n_head * tokens_per_block * head_size
            # kv_cache_manager = KVCacheManager(
            #     num_layers=gpt_config.n_layer,
            #     num_blocks=num_blocks,
            #     block_size=block_size,
            #     tokens_per_block=tokens_per_block,
            #     max_blocks_per_seq=max_blocks_per_seq,
            #     max_attention_window_size=total_length,
            #     sink_token_len=0,
            #     beam_width=beam_width)
            # host_kv_cache_pool_pointers = torch.tensor(
            # [key_value_cache_buffers[0].data_ptr(), 0], dtype=torch.int64)

            # Add sequences to the manager
            for bi in range(batch_size):
                generation_sequence = GenerationSequence(seq_idx=bi,
                                                         batch_idx=bi)
                pools_kv_cache_manager.add_sequence(generation_sequence,
                                                    seq_len)

            # Pre allocate the kv cache for the generated tokens.
            pools_kv_cache_manager.step([False] * batch_size)

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
                       host_runtime_perf_knobs,
                       host_context_progress,
                       sequence_length=None,
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
                'host_runtime_perf_knobs': host_runtime_perf_knobs,
                'host_context_progress': host_context_progress,
            }

            assert host_request_types is not None
            if enable_remove_input_padding:
                assert host_context_lengths is not None, "host_context_lengths is required for ragged input"
                ctx_buffer['host_context_lengths'] = host_context_lengths

            if enable_paged_kv_cache:
                assert beam_width == 1
                # for beam_width > 1 the argument must be '1' in ctx phase and 'beam_width' in gen phase
                host_kv_cache_block_offsets = pools_kv_cache_manager.get_block_offsets(
                    beam_width=1)
                kv_cache_block_offsets = host_kv_cache_block_offsets.to('cuda')

                shape = kv_cache_block_offsets.shape
                shape = [shape[0], shape[1] * shape[2], *shape[3:]]
                ctx_buffer[
                    f'kv_cache_block_offsets'] = kv_cache_block_offsets.reshape(
                        shape).contiguous()
                ctx_buffer[
                    f'host_kv_cache_block_offsets'] = host_kv_cache_block_offsets.reshape(
                        shape).contiguous()
                ctx_buffer[
                    f'host_kv_cache_pool_pointers'] = host_kv_cache_pool_pointers.contiguous(
                    )
                ctx_buffer[
                    f'host_kv_cache_pool_mapping'] = memory_pools_allocator.pool_mapping.contiguous(
                    )

                ctx_buffer[
                    f'host_max_attention_window_sizes'] = host_max_attention_window_sizes
            else:
                for i in range(gpt_config.n_layer):
                    ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[
                        i]
                    ctx_buffer[
                        f'present_key_value_{i}'] = key_value_cache_buffers[i]
                ctx_buffer[
                    f'host_max_attention_window_sizes'] = host_max_attention_window_sizes

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

        hf_outputs = None
        step0_ids = None
        step1_ids = None

        def compare_context(run_ref_only=False):
            nonlocal step0_ids
            step0_ids = torch.randint(
                100, (batch_size,
                      seq_len)).int().cuda() if step0_ids is None else step0_ids
            ctx_ids = step0_ids.clone()

            ctx_context_lengths = seq_len * torch.ones(
                (batch_size), dtype=torch.int32, device='cuda')
            ctx_position_ids = torch.tensor(range(seq_len),
                                            dtype=torch.int32).reshape([
                                                1, seq_len
                                            ]).expand([batch_size,
                                                       seq_len]).cuda()
            ctx_last_token_ids = ctx_context_lengths.clone()

            nonlocal hf_outputs
            with torch.no_grad():
                hf_outputs = hf_gpt.forward(ctx_ids)
            torch.cuda.synchronize()
            ref = hf_outputs.logits
            if run_ref_only:
                return ref[:, -1, :]

            if enable_remove_input_padding:
                ctx_ids = ctx_ids.view([batch_size * seq_len])
                ctx_position_ids = ctx_position_ids.view([batch_size * seq_len])
                ctx_last_token_ids = torch.cumsum(ctx_last_token_ids,
                                                  dim=0).int()

            host_max_attention_window_sizes = torch.tensor([total_length] *
                                                           gpt_config.n_layer,
                                                           dtype=torch.int32)
            host_sink_token_length = torch.tensor([0], dtype=torch.int32)

            host_context_lengths = ctx_context_lengths.cpu(
            ) if enable_remove_input_padding else None
            host_request_types = torch.tensor([0 for i in range(batch_size)],
                                              dtype=torch.int32).cpu()

            host_past_key_value_lengths = ctx_context_lengths.detach().clone(
            ).cpu()
            # We need sequence_lengths start as context_lengths for step 0 (context),
            # and it will be added one after each step.
            sequence_length = ctx_context_lengths.detach().clone()

            perf_knob_tensor_size = 16
            ctx_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                                  dtype=torch.int64)
            if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
                ctx_runtime_perf_knobs[1] = 1  # enable_context_fmha_fp32_acc

            host_context_progress = torch.tensor([0], dtype=torch.int64)

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
                sequence_length=sequence_length,
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types,
                host_runtime_perf_knobs=ctx_runtime_perf_knobs,
                host_context_progress=host_context_progress)

            if gather_context_logits:
                np.testing.assert_allclose(ref.cpu().numpy().flatten(),
                                           res.cpu().numpy().flatten(),
                                           atol=1e-1)
            else:
                np.testing.assert_allclose(ref[:, -1, :].cpu().numpy(),
                                           res.cpu().numpy(),
                                           atol=1e-1)

        def compare_generation(run_ref_only=False):
            step = 1
            nonlocal step1_ids
            step1_ids = torch.randint(
                100, (batch_size,
                      1)).int().cuda() if step1_ids is None else step1_ids

            gen_ids = step1_ids.clone()

            gen_context_lengths = seq_len * torch.ones(
                (batch_size), dtype=torch.int32, device='cuda')
            gen_position_ids = torch.ones_like(gen_ids).int().cuda() * seq_len
            gen_last_token_ids = torch.zeros_like(
                gen_context_lengths).int().cuda()

            nonlocal hf_outputs
            with torch.no_grad():
                hf_outputs = hf_gpt.forward(
                    gen_ids,
                    past_key_values=hf_outputs.past_key_values,
                    use_cache=True)
            torch.cuda.synchronize()
            ref = hf_outputs.logits[:, -1, :]
            if run_ref_only:
                return ref

            if enable_remove_input_padding:
                gen_ids = gen_ids.view([batch_size])
                gen_position_ids = gen_position_ids.view([batch_size])
                gen_last_token_ids = torch.ones_like(
                    gen_context_lengths).int().cuda()
                gen_last_token_ids = torch.cumsum(gen_last_token_ids,
                                                  dim=0).int()

            host_past_key_value_lengths = torch.tensor([seq_len + step - 1] *
                                                       batch_size,
                                                       dtype=torch.int32)
            host_max_attention_window_sizes = torch.tensor([seq_len + step] *
                                                           gpt_config.n_layer,
                                                           dtype=torch.int32)
            host_sink_token_length = torch.tensor([0], dtype=torch.int32)

            host_context_lengths = gen_context_lengths.cpu(
            ) if enable_remove_input_padding else None
            host_request_types = torch.tensor([1 for i in range(batch_size)],
                                              dtype=torch.int32).cpu()

            # For step 1, the sequence_lengths = context_lengths + 1.
            sequence_length = torch.add(gen_context_lengths.detach().clone(), 1)

            perf_knob_tensor_size = 16
            gen_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                                  dtype=torch.int64)
            if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
                gen_runtime_perf_knobs[1] = 1  # enable_context_fmha_fp32_acc

            host_context_progress = torch.tensor([0], dtype=torch.int64)

            res = run_engine(
                context=runtime.context_1,
                input_ids=gen_ids,
                context_lengths=gen_context_lengths,
                position_ids=gen_position_ids,
                last_token_ids=gen_last_token_ids,
                cache_indirection=cache_indirections[1],
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                sequence_length=sequence_length,
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types,
                host_runtime_perf_knobs=gen_runtime_perf_knobs,
                host_context_progress=host_context_progress)

            np.testing.assert_allclose(ref.cpu().numpy().flatten(),
                                       res.cpu().numpy().flatten(),
                                       atol=1e-1)

        def compare_mixing_context_and_generation_phases():

            num_context_input = 2
            assert batch_size >= num_context_input
            num_generation_input = batch_size - num_context_input

            # retrieve the reference output
            ref_ctx_out = compare_context(True)[:num_context_input, :]
            ref_gen_out = compare_generation(True)[num_context_input:, :]
            ref_out = torch.cat([ref_ctx_out, ref_gen_out], dim=0)

            ref_ctx_out = None
            ref_gen_out = None

            # compare_context()

            # prepare the inputs for plugin-based gpt
            assert step0_ids is not None and step1_ids is not None
            input_ids = torch.cat([
                step0_ids[:num_context_input, :].view(
                    (-1, )), step1_ids[num_context_input:].view((-1, ))
            ],
                                  dim=0)

            input_ids = input_ids.view((-1, ))

            ctx_position_ids = torch.tensor(
                range(seq_len), dtype=torch.int32).reshape(
                    (1, seq_len)).expand([num_generation_input,
                                          seq_len]).cuda()
            gen_position_ids = torch.ones_like(
                step1_ids[num_context_input:].view(
                    (-1, ))).int().cuda() * seq_len
            position_ids = torch.cat(
                [ctx_position_ids.view((-1, )), gen_position_ids], dim=0).view(
                    (-1, ))

            input_lengths = torch.tensor([seq_len] * num_context_input +
                                         [1] * num_generation_input,
                                         dtype=torch.int32).cuda()
            gen_last_token_ids = torch.cumsum(input_lengths, dim=0).int().cuda()

            # scalar of max_key_value_length for in-flight batching case
            host_past_key_value_lengths = torch.tensor(
                [0] * num_context_input + [seq_len] * num_generation_input,
                dtype=torch.int32)

            host_max_attention_window_sizes = torch.tensor([total_length] *
                                                           gpt_config.n_layer,
                                                           dtype=torch.int32)

            host_sink_token_length = torch.tensor([0], dtype=torch.int32)

            context_lengths = torch.tensor([seq_len] * batch_size,
                                           dtype=torch.int32).cuda()
            if enable_remove_input_padding:
                host_context_lengths = context_lengths.cpu()

            host_request_types = torch.tensor([0] * num_context_input +
                                              [1] * num_generation_input,
                                              dtype=torch.int32).cpu()

            # The sequence_lengths = context_lengths + step for generation stage.
            sequence_length = torch.tensor([seq_len] * num_context_input +
                                           [seq_len + 1] * num_generation_input,
                                           dtype=torch.int32).cuda()
            perf_knob_tensor_size = 16
            runtime_perf_knobs_tensor = torch.tensor([-1] *
                                                     perf_knob_tensor_size,
                                                     dtype=torch.int64)
            if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
                runtime_perf_knobs_tensor[1] = 1  # enable_context_fmha_fp32_acc

            host_context_progress = torch.tensor([0], dtype=torch.int64)

            res = run_engine(
                context=runtime.context_1,
                input_ids=input_ids,
                context_lengths=context_lengths,
                position_ids=position_ids,
                last_token_ids=gen_last_token_ids,
                cache_indirection=cache_indirections[0],
                host_past_key_value_lengths=host_past_key_value_lengths,
                host_max_attention_window_sizes=host_max_attention_window_sizes,
                host_sink_token_length=host_sink_token_length,
                sequence_length=sequence_length,
                host_context_lengths=host_context_lengths,
                host_request_types=host_request_types,
                host_runtime_perf_knobs=runtime_perf_knobs_tensor,
                host_context_progress=host_context_progress)

            np.testing.assert_allclose(ref_out.cpu().numpy(),
                                       res.cpu().numpy(),
                                       atol=1e-1)

        # Main logics
        compare_context()
        compare_generation()

        # Only inflight batching mode could accept the mixture of requests from both context and generation phases
        if use_in_flight_batching:
            compare_mixing_context_and_generation_phases()

    @parameterized.expand([("other", False, False), ("other", False, True)],
                          name_func=unittest_name_func)
    def test_greedy_search_float32(self, test_partition, use_refit, streaming):
        torch.manual_seed(42)

        model = 'gpt'
        log_level = 'error'
        dtype = 'float32'
        world_size = 1
        rank = 0

        hidden_act = 'gelu'
        n_layer = 2
        max_new_tokens = 1
        batch_size = 4
        seq_len = 128
        use_plugin = False

        do_sample = False
        early_stoppping = False
        num_beams = 1
        num_beam_groups = 1
        temperature = 1
        top_k = 0
        top_p = 0.0
        random_seed = 0
        length_penalty = 1
        repetition_penalty = 1

        gpt_config, hf_gpt = self._gen_hf_gpt(hidden_act, n_layer,
                                              max_new_tokens, dtype)
        runtime, engine_buffer = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, gpt_config, hf_gpt, model,
            use_plugin, batch_size, seq_len, max_new_tokens, use_refit)

        model_config = ModelConfig(max_batch_size=batch_size,
                                   max_beam_width=num_beams,
                                   vocab_size=gpt_config.vocab_size,
                                   num_layers=gpt_config.n_layer,
                                   num_heads=gpt_config.n_head,
                                   num_kv_heads=gpt_config.n_head,
                                   hidden_size=gpt_config.n_embd,
                                   gpt_attention_plugin=False,
                                   dtype=dtype)

        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)
        decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, mapping)
        pad_token_id = 50256
        eos_token_id = 50257
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         temperature=temperature,
                                         top_k=top_k,
                                         top_p=top_p,
                                         random_seed=random_seed,
                                         length_penalty=length_penalty,
                                         repetition_penalty=repetition_penalty)
        input_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        input_ids[0][-1] = pad_token_id
        input_ids[1][-3:] = pad_token_id
        input_ids[2][-5:] = pad_token_id

        input_lengths = torch.ones(
            (batch_size)).type(torch.int32).cuda() * seq_len

        decoder.setup(batch_size,
                      max_context_length=seq_len,
                      max_new_tokens=max_new_tokens,
                      beam_width=num_beams)
        if streaming:
            output_ids_gen = decoder.decode(input_ids,
                                            input_lengths,
                                            sampling_config,
                                            streaming=True)
            for output_ids in output_ids_gen:
                pass
        else:
            output_ids = decoder.decode(input_ids, input_lengths,
                                        sampling_config)
        #TODO: change to actual ragged tensor after GPT plugin supports it
        output_ids_x = decoder.decode(input_ids, input_lengths, sampling_config)

        # works because all requests in the batch has same
        # TODO: enable this when GPT Plugin attention works
        # output_ids_y = decoder.decode_batch([t[:input_lengths[i]] for i, t in enumerate(torch.split(input_ids, 1, dim=0))], sampling_config)

        torch.cuda.synchronize()
        torch.testing.assert_close(output_ids, output_ids_x)

        res = output_ids.squeeze()
        res = res[:, -max_new_tokens:]

        ref_output_ids = hf_gpt.generate(input_ids,
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

    @parameterized.expand(["other"], name_func=unittest_name_func)
    def test_rope_scaling_is_set_in_attention(self, test_partition):
        num_layers = 2
        position_embedding_type = 'rope_gpt_neox'
        rotary_embedding_percentage = 0.3
        rotary_base = 99999.1
        rotary_scaling = {"type": "linear", "factor": 2.72}

        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': 'float16',
            'num_hidden_layers': num_layers,
            'num_attention_heads': 4,
            'hidden_size': 128,
            'vocab_size': 256,
            'max_position_embeddings': 1024,
            'hidden_act': 'gelu',
            'position_embedding_type': position_embedding_type,
            'rotary_pct': rotary_embedding_percentage,
            'rotary_base': rotary_base,
            'rotary_scaling': rotary_scaling,
        }
        config = tensorrt_llm.models.PretrainedConfig.from_dict(config)
        tensorrt_llm_gpt = tensorrt_llm.models.GPTForCausalLM(config)

        for layer_i in range(num_layers):
            assert tensorrt_llm_gpt.transformer.layers[
                layer_i].attention.rotary_embedding_base == rotary_base
            assert tensorrt_llm_gpt.transformer.layers[
                layer_i].attention.rotary_embedding_scale == rotary_scaling[
                    "factor"]
            assert tensorrt_llm_gpt.transformer.layers[
                layer_i].attention.rotary_embedding_scale_type == RotaryScalingType.linear
            assert tensorrt_llm_gpt.transformer.layers[
                layer_i].attention.position_embedding_type == PositionEmbeddingType.rope_gpt_neox

    @parameterized.expand(["other"], name_func=unittest_name_func)
    def test_gpt_variant_is_overridden(self, test_partition):
        model_root = llm_models_root()
        if model_root is None:
            pytest.skip("Skipping since real weights are unavailable.")

        with tempfile.TemporaryDirectory() as tempdir:
            cli_args = Namespace(tp_size=1,
                                 pp_size=1,
                                 model_dir=f"{model_root}/starcoder2-3b",
                                 output_dir=tempdir,
                                 gpt_variant="starcoder2",
                                 dtype="float16",
                                 load_model_on_cpu=False,
                                 use_parallel_embedding=False,
                                 embedding_sharding_dim=0,
                                 use_weight_only=False,
                                 int8_kv_cache=False,
                                 smoothquant=None,
                                 workers=1)

            def check_gpt_variant(*args, **kwargs):
                self.assertEqual(kwargs.get("gpt_variant", ""),
                                 cli_args.gpt_variant)
                return from_hugging_face(*args, **kwargs)

            from_hugging_face = tensorrt_llm.models.GPTConfig.from_hugging_face

            with patch('tensorrt_llm.models.GPTConfig.from_hugging_face',
                       side_effect=check_gpt_variant):
                convert_and_save_hf(cli_args)


if __name__ == '__main__':
    unittest.main()
