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
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from parameterized import parameterized
from transformers import LlamaConfig, LlamaForCausalLM
from utils.llm_data import llm_models_root
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm import Builder
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_str
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models.llama.convert import (load_weights_from_hf_model,
                                               load_weights_from_meta_ckpt)
from tensorrt_llm.models.modeling_utils import optimize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType


class TestLLaMA(unittest.TestCase):
    EOS_TOKEN = 2
    PAD_TOKEN = 2

    def _gen_tensorrt_llm_network(self, network, hf_llama,
                                  llama_config: LlamaConfig, batch_size,
                                  beam_width, input_len, output_len, dtype,
                                  rank, tensor_parallel, **opt_flags):
        list(range(tensor_parallel))

        with net_guard(network):
            str_dtype_to_trt(dtype)

            config = {
                'architecture': "LlamaForCausalLM",
                'dtype': dtype,
                'logits_dtype': 'float32',
                'num_hidden_layers': llama_config.num_hidden_layers,
                'num_attention_heads': llama_config.num_attention_heads,
                'hidden_size': llama_config.hidden_size,
                'intermediate_size': llama_config.intermediate_size,
                'num_key_value_heads': llama_config.num_key_value_heads,
                'vocab_size': llama_config.vocab_size,
                'position_embedding_type': 'rope_gpt_neox',
                'max_position_embeddings': llama_config.max_position_embeddings,
                'hidden_act': llama_config.hidden_act,
                'rotary_base': getattr(llama_config, 'rotary_base', 10000.0),
                'rotary_scaling': getattr(llama_config, 'rotary_scaling', None),
                'norm_epsilon': llama_config.rms_norm_eps,
                'mapping': {
                    'world_size': tensor_parallel,
                    'tp_size': tensor_parallel,
                    'rank': rank,
                },
                "moe": {
                    "num_experts": 0,
                    "normalization_mode": 1
                },
                'use_parallel_embedding': False,
                'embedding_sharding_dim': 0,
            }

            # Initialize model
            config = tensorrt_llm.models.LLaMAConfig.from_dict(config)
            weights = load_weights_from_hf_model(hf_llama, config)

            tensorrt_llm_llama = tensorrt_llm.models.LLaMAForCausalLM(config)
            tensorrt_llm_llama.load(weights)
            optimize_model(tensorrt_llm_llama, **opt_flags)

            # Prepare
            network.set_named_parameters(tensorrt_llm_llama.named_parameters())
            inputs = tensorrt_llm_llama.prepare_inputs(
                max_batch_size=batch_size,
                max_input_len=input_len,
                max_seq_len=input_len + output_len,
                max_num_tokens=batch_size * input_len,
                use_cache=True,
                max_beam_width=beam_width)
            # Forward
            tensorrt_llm_llama(**inputs)

        return network

    def _gen_tensorrt_llm_engine(self,
                                 dtype,
                                 rank,
                                 world_size,
                                 llama_config,
                                 hf_llama,
                                 model_name,
                                 use_plugin,
                                 batch_size,
                                 beam_width,
                                 input_len,
                                 output_len,
                                 use_refit,
                                 fast_building=False,
                                 context_fmha_flag=ContextFMHAType.disabled,
                                 enable_remove_input_padding=False,
                                 **opt_flags):

        builder = Builder()

        with tempfile.TemporaryDirectory() as tmpdirname:
            builder_config = builder.create_builder_config(
                name=model_name,
                precision=dtype,
                timing_cache='model.cache',
                tensor_parallel=world_size,  # TP only
                use_refit=use_refit,
                strongly_typed=True,
            )
            network = builder.create_network()
            network.plugin_config.to_legacy_setting()
            if use_plugin:
                network.plugin_config.gpt_attention_plugin = dtype
            if fast_building:
                network.plugin_config.gemm_plugin = dtype
            if enable_remove_input_padding:
                network.plugin_config.remove_input_padding = True
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network, hf_llama, llama_config,
                                           batch_size, beam_width, input_len,
                                           output_len, dtype, rank, world_size,
                                           **opt_flags)

            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(self,
                                  log_level,
                                  dtype,
                                  world_size,
                                  rank,
                                  llama_config,
                                  hf_llama,
                                  model_name,
                                  use_plugin,
                                  batch_size,
                                  beam_width,
                                  input_len,
                                  output_len,
                                  use_refit,
                                  fast_building=False,
                                  context_fmha_flag=ContextFMHAType.disabled,
                                  enable_remove_input_padding=False,
                                  **opt_flags):
        tensorrt_llm.logger.set_level(log_level)
        mapping = tensorrt_llm.Mapping(world_size, rank, tp_size=world_size)
        engine_buffer = self._gen_tensorrt_llm_engine(
            dtype, rank, world_size, llama_config, hf_llama, model_name,
            use_plugin, batch_size, beam_width, input_len, output_len,
            use_refit, fast_building, context_fmha_flag,
            enable_remove_input_padding, **opt_flags)
        runtime = tensorrt_llm.runtime.generation._Runtime(
            engine_buffer, mapping)
        return runtime, engine_buffer

    def load_test_cases():
        test_cases = list(
            product([False], [False, True], [
                ContextFMHAType.disabled, ContextFMHAType.enabled,
                ContextFMHAType.enabled_with_fp32_acc
            ], [False, True], ['float16'], [0], ['silu'],
                    [{
                        "use_fused_mlp": True
                    }, {
                        "use_fused_mlp": False
                    }]))
        test_cases.append((False, True, ContextFMHAType.disabled, False,
                           'bfloat16', 0, 'silu', dict()))
        test_cases.append(
            (False, True, ContextFMHAType.enabled, False, 'float16', 1, 'silu',
             dict()))  # MQA
        test_cases.append((False, True, ContextFMHAType.disabled, False,
                           'bfloat16', 2, 'silu', dict()))  # GQA
        test_cases.append(
            (False, True, ContextFMHAType.enabled, False, 'float16', 2, 'silu',
             dict()))  # GQA
        test_cases.append((False, True, ContextFMHAType.enabled_with_fp32_acc,
                           False, 'float16', 4, 'silu', dict()))  # GQA
        return test_cases

    @parameterized.expand(load_test_cases, name_func=unittest_name_func)
    def test_llama(self, use_refit, fast_building, context_fmha_flag,
                   enable_remove_input_padding, dtype, num_kv_heads, hidden_act,
                   opt_flags):

        PRECHECKED_GOOD_RANDOM_SEEDS = [1, 4, 5, 8]
        model = 'llama'
        log_level = 'error'
        use_plugin = True  # gpt plugin
        batch_size = 4
        beam_width = 1
        input_len = 4
        output_len = 2
        max_seq_len = input_len + output_len
        world_size = 1
        head_size = 32
        rank = 0
        llama_config = LlamaConfig()
        llama_config.hidden_act = hidden_act
        llama_config.num_hidden_layers = 2
        llama_config.max_position_embeddings = 64
        llama_config.vocab_size = 128
        llama_config.num_attention_heads = 2 if num_kv_heads <= 1 else 2 * num_kv_heads
        llama_config.hidden_size = llama_config.num_attention_heads * head_size
        llama_config.head_dim = head_size
        llama_config.intermediate_size = ((
            (llama_config.hidden_size * 4 * 2 // 3) + head_size - 1) //
                                          head_size) * head_size
        if hasattr(llama_config, "num_key_value_heads"):
            llama_config.num_key_value_heads = num_kv_heads if num_kv_heads != 0 else llama_config.num_attention_heads
            print(llama_config.num_key_value_heads)
            assert (llama_config.num_attention_heads %
                    llama_config.num_key_value_heads) == 0
        llama_config.pad_token_id = self.PAD_TOKEN
        llama_config.eos_token_id = self.EOS_TOKEN
        seed_idx = random.randint(0, len(PRECHECKED_GOOD_RANDOM_SEEDS) - 1)
        torch.manual_seed(PRECHECKED_GOOD_RANDOM_SEEDS[seed_idx])
        hf_llama = LlamaForCausalLM(llama_config).cuda().eval()
        runtime, _ = self._gen_tensorrt_llm_runtime(
            log_level, dtype, world_size, rank, llama_config, hf_llama, model,
            use_plugin, batch_size, beam_width, input_len, output_len,
            use_refit, fast_building, context_fmha_flag,
            enable_remove_input_padding, **opt_flags)
        key_value_cache_buffers = []
        head_size = llama_config.hidden_size // llama_config.num_attention_heads
        for i in range(llama_config.num_hidden_layers):
            key_value_cache_buffers.append(
                torch.zeros((
                    batch_size,
                    2,
                    llama_config.num_key_value_heads,
                    max_seq_len,
                    head_size,
                ),
                            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype),
                            device='cuda'))

        # compare context
        step = 0
        ctx_ids = torch.randint(100, (batch_size, input_len)).int().cuda()
        ctx_context_lengths = input_len * torch.ones(
            (batch_size), dtype=torch.int32, device='cuda')
        ctx_position_ids = torch.tensor(range(input_len),
                                        dtype=torch.int32).reshape([
                                            1, input_len
                                        ]).expand([batch_size,
                                                   input_len]).cuda()
        ctx_last_token_ids = ctx_context_lengths.clone()
        ctx_host_request_types = torch.tensor([0] * batch_size,
                                              dtype=torch.int32)

        # We need sequence_lengths start as context_lengths for step 0,
        # and it will be added one after each step.
        sequence_length_buffer = ctx_context_lengths.detach().clone()

        with torch.no_grad():
            hf_outputs = hf_llama.forward(ctx_ids)
        torch.cuda.synchronize()
        ref = hf_outputs.logits[:, -1, :]

        if enable_remove_input_padding:
            ctx_ids = ctx_ids.view([batch_size * input_len])
            ctx_position_ids = ctx_position_ids.view([batch_size * input_len])
            ctx_last_token_ids = torch.cumsum(ctx_last_token_ids, dim=0).int()

        cache_indirections = [
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda'),
            torch.full((
                batch_size,
                beam_width,
                max_seq_len,
            ),
                       0,
                       dtype=torch.int32,
                       device='cuda')
        ]  # ping-pong buffers

        perf_knob_tensor_size = 16
        # runtime_perf_knobs is not used in context phase
        context_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                                  dtype=torch.int64)
        if context_fmha_flag == ContextFMHAType.enabled_with_fp32_acc:
            context_runtime_perf_knobs[1] = 1  # enable_context_fmha_fp32_acc

        host_context_progress = torch.tensor([0], dtype=torch.int64)

        ctx_buffer = {
            'input_ids': ctx_ids,
            'context_lengths': ctx_context_lengths,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'cache_indirection': cache_indirections[0],
            'host_request_types': ctx_host_request_types,
            'host_runtime_perf_knobs': context_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        if enable_remove_input_padding:
            ctx_buffer['host_context_lengths'] = ctx_context_lengths.cpu()

        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

        kv_shape = (batch_size, 2, llama_config.num_key_value_heads,
                    max_seq_len, head_size)
        ctx_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
            [max_seq_len] * llama_config.num_hidden_layers, dtype=torch.int32)
        ctx_shape[f'host_max_attention_window_sizes'] = (
            llama_config.num_hidden_layers, )
        for i in range(llama_config.num_hidden_layers):
            ctx_shape[f'past_key_value_{i}'] = kv_shape
            ctx_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            ctx_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        ctx_buffer['sequence_length'] = sequence_length_buffer
        ctx_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
        ctx_shape['host_past_key_value_lengths'] = (batch_size, )
        ctx_buffer['host_past_key_value_lengths'] = torch.tensor(
            [0] * batch_size, dtype=torch.int32)
        ctx_shape['host_sink_token_length'] = (1, )
        ctx_buffer['host_sink_token_length'] = torch.tensor([0],
                                                            dtype=torch.int32)

        context = runtime.ctx_context
        runtime._set_shape(context, ctx_shape)
        runtime._set_buffer(context, ctx_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = ctx_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=0.12)

        # compare generation
        step = 1
        step1_id = torch.randint(100, (batch_size, 1)).int().cuda()
        gen_context_lengths = ctx_context_lengths.clone()
        gen_position_ids = torch.ones_like(step1_id).int().cuda() * input_len
        gen_last_token_ids = torch.zeros_like(gen_context_lengths).int().cuda()
        gen_host_request_types = torch.tensor([1] * batch_size,
                                              dtype=torch.int32)

        with torch.no_grad():
            hf_outputs = hf_llama.forward(
                step1_id,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True)
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
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'host_request_types': gen_host_request_types,
            'cache_indirection': cache_indirections[1],
            'host_runtime_perf_knobs': gen_runtime_perf_knobs,
            'host_context_progress': host_context_progress,
        }
        if enable_remove_input_padding:
            step1_buffer['host_context_lengths'] = gen_context_lengths.cpu()

        step1_shape = {k: v.shape for k, v in step1_buffer.items()}

        step1_shape[f'host_max_attention_window_sizes'] = (
            llama_config.num_hidden_layers, )
        for i in range(llama_config.num_hidden_layers):
            step1_shape[f'past_key_value_{i}'] = kv_shape
        step1_shape['sequence_length'] = (batch_size, )
        step1_shape['host_past_key_value_lengths'] = (batch_size, )
        step1_shape['host_sink_token_length'] = (1, )
        step1_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
            [max_seq_len] * llama_config.num_hidden_layers, dtype=torch.int32)
        for i in range(llama_config.num_hidden_layers):
            step1_buffer[f'past_key_value_{i}'] = key_value_cache_buffers[i]
            step1_buffer[f'present_key_value_{i}'] = key_value_cache_buffers[i]
        step1_buffer[
            'host_past_key_value_lengths'] = sequence_length_buffer.cpu()
        sequence_length_buffer = torch.add(sequence_length_buffer, step)
        step1_buffer['sequence_length'] = sequence_length_buffer
        step1_buffer['host_sink_token_length'] = torch.tensor([0],
                                                              dtype=torch.int32)

        context = runtime.context_1
        runtime._set_shape(context, step1_shape)
        runtime._set_buffer(context, step1_buffer)
        runtime._run(context)
        torch.cuda.synchronize()
        res = step1_buffer['logits']

        np.testing.assert_allclose(ref.to(torch.float32).cpu().numpy(),
                                   res.to(torch.float32).cpu().numpy(),
                                   atol=0.12)

    def get_loader_test_cases():
        test_cases = []
        test_cases.extend(
            list(
                product([
                    ("llama-7b-hf", "7B"),
                ], [
                    (1, 0),
                    (2, 0),
                    (2, 1),
                ], [
                    -1,
                    0,
                    1,
                ])))
        test_cases.extend(
            list(
                product([
                    ("llama-7b-4gqa-hf", "7B-4GQA"),
                ], [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1)], [-1, 0, 1])))
        test_cases.extend(
            list(
                product([
                    ("llama-7b-4gqa-hf", "7B-4GQA"),
                ], [(8, 0), (8, 7)], [-1, 0, 1])))
        return test_cases

    @parameterized.expand(get_loader_test_cases, name_func=unittest_name_func)
    def test_loaders(self, paths, tp_info, emb_sharding_dim):
        model_root = llm_models_root()
        if model_root is None:
            pytest.skip("Skipping since real weights are unavailable.")
        model_root = Path(
            model_root,
            "llama-synthetic" if paths[0].find("gqa") >= 0 else "llama-models")
        hf_path = Path(model_root, paths[0])
        meta_path = Path(model_root, paths[1])
        if not hf_path.exists():
            pytest.skip(f"Skipping since the path {hf_path} does not exist.")
        if not meta_path.exists():
            pytest.skip(f"Skipping since the path {meta_path} does not exist.")

        def print_corner(name, t: np.ndarray):
            if len(t.shape) == 1:
                tl = t[:2]
                br = t[-2:]
            elif len(t.shape) == 2:
                tl = t[:2, :2]
                br = t[-2:, -2:]
            print(name, np.concatenate([tl, br]).flatten())

        def print_layers(m: tensorrt_llm.models.LLaMAForCausalLM):
            print_corner("vocab", m.vocab_embedding.weight.raw_value)
            print_corner("lm_head", m.lm_head.weight.raw_value)
            print_corner("ln_f", m.ln_f.weight.raw_value)
            print_corner("qkv", m.layers[0].attention.qkv.weight.raw_value)
            print_corner("gate", m.layers[0].mlp.gate.weight.raw_value)
            print_corner("inorm", m.layers[0].input_layernorm.weight.raw_value)
            print(flush=True)
            return

        import tensorrt as trt

        tp_size = tp_info[0]
        rank = tp_info[1]
        use_parallel_embedding = (emb_sharding_dim >= 0)
        embedding_sharding_dim = abs(emb_sharding_dim)
        hf_llama = LlamaForCausalLM.from_pretrained(
            hf_path,
            device_map={
                "model": "cpu",
                "lm_head": "cpu"
            },  # Load to CPU memory
            dtype="auto")
        assert hf_llama.config.torch_dtype == torch.float16
        kv_dtype = trt.float16 if hf_llama.config.torch_dtype == torch.float16 else trt.float32
        max_context_length = 128  # for loader tests this value does not matter
        config = {
            'architecture': "LlamaForCausalLM",
            'dtype': trt_dtype_to_str(kv_dtype),
            'logits_dtype': 'float32',
            'num_hidden_layers': hf_llama.config.num_hidden_layers,
            'num_attention_heads': hf_llama.config.num_attention_heads,
            'hidden_size': hf_llama.config.hidden_size,
            'intermediate_size': hf_llama.config.intermediate_size,
            'num_key_value_heads': hf_llama.config.num_key_value_heads,
            'vocab_size': hf_llama.config.vocab_size,
            'position_embedding_type': 'rope_gpt_neox',
            'max_position_embeddings': hf_llama.config.max_position_embeddings,
            'hidden_act': hf_llama.config.hidden_act,
            'rotary_base': getattr(hf_llama.config, 'rotary_base', 10000.0),
            'rotary_scaling': getattr(hf_llama.config, 'rotary_scaling', None),
            'norm_epsilon': hf_llama.config.rms_norm_eps,
            'mapping': {
                'world_size': tp_size,
                'tp_size': tp_size,
                'rank': rank,
            },
            "moe": {
                "num_experts": 0,
                "top_k": 0,
                "normalization_mode": 1,
            },
            'use_parallel_embedding': use_parallel_embedding,
            'embedding_sharding_dim': embedding_sharding_dim,
        }

        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_llama_wHF = tensorrt_llm.models.LLaMAForCausalLM(config)
        # print_layers(tensorrt_llm_llama_wHF)
        weights_wHF = load_weights_from_hf_model(hf_llama, config)
        tensorrt_llm_llama_wHF.load(weights_wHF)
        # print_layers(tensorrt_llm_llama_wHF)

        tensorrt_llm_llama_wMETA = tensorrt_llm.models.LLaMAForCausalLM(config)
        # print_layers(tensorrt_llm_llama_wMETA)
        weights_wMETA = load_weights_from_meta_ckpt(meta_path, config)
        tensorrt_llm_llama_wMETA.load(weights_wMETA)
        # print_layers(tensorrt_llm_llama_wMETA)
        # token embedding

        np.testing.assert_allclose(
            weights_wHF['transformer.vocab_embedding.weight'],
            weights_wMETA['transformer.vocab_embedding.weight'],
            atol=1e-3)
        # output
        np.testing.assert_allclose(weights_wHF['lm_head.weight'],
                                   weights_wMETA['lm_head.weight'],
                                   atol=1e-3)
        # norm
        np.testing.assert_allclose(weights_wHF['transformer.ln_f.weight'],
                                   weights_wMETA['transformer.ln_f.weight'],
                                   atol=1e-3)
        # Checking all of the layers takes too much time, just check one random layer
        l = np.random.randint(0, hf_llama.config.num_hidden_layers)
        # for l in range(tensorrt_llm_llama_wHF.num_layers):
        tllm_prefix = 'transformer.layers.{}.'.format(l)
        if l >= 0:
            print(f"Checking Layer-{l} weights ...", flush=True)
            # layer{l}.input_layernorm
            np.testing.assert_allclose(
                weights_wHF[tllm_prefix + 'input_layernorm.weight'],
                weights_wMETA[tllm_prefix + 'input_layernorm.weight'],
                atol=1e-3)
            # layer{l}.post_layernorm
            np.testing.assert_allclose(
                weights_wHF[tllm_prefix + 'post_layernorm.weight'],
                weights_wMETA[tllm_prefix + 'post_layernorm.weight'],
                atol=1e-3)
            # layer{l}.mlp.gate
            np.testing.assert_allclose(
                weights_wHF[tllm_prefix + 'mlp.gate.weight'],
                weights_wMETA[tllm_prefix + 'mlp.gate.weight'],
                atol=1e-3)
            # layer{l}.mlp.proj
            np.testing.assert_allclose(
                weights_wHF[tllm_prefix + 'mlp.proj.weight'],
                weights_wMETA[tllm_prefix + 'mlp.proj.weight'],
                atol=1e-3)
            # layer{l}.mlp.fc
            np.testing.assert_allclose(
                weights_wHF[tllm_prefix + 'mlp.fc.weight'],
                weights_wMETA[tllm_prefix + 'mlp.fc.weight'],
                atol=1e-3)
            # layer{l}.dense
            np.testing.assert_allclose(
                weights_wHF[tllm_prefix + 'attention.dense.weight'],
                weights_wMETA[tllm_prefix + 'attention.dense.weight'],
                atol=1e-3)
            # layer{l}.qkv
            np.testing.assert_allclose(
                weights_wHF[tllm_prefix + 'attention.qkv.weight'],
                weights_wMETA[tllm_prefix + 'attention.qkv.weight'],
                atol=1e-3)
        return


if __name__ == '__main__':
    unittest.main()
