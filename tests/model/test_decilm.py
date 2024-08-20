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
import itertools
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorrt as trt
import torch
import transformers
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.builder import Builder
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.deci.config import DeciConfig, DeciLayerConfig
from tensorrt_llm.models.deci.convert import _ffn_mult_to_intermediate_size
from tensorrt_llm.models.deci.layer_config import (AttentionImplementation,
                                                   FFNImplementation)
from tensorrt_llm.models.deci.model import DeciLMForCausalLM
from tensorrt_llm.network import Network, net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime.generation import _Runtime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import unittest_name_func


class TestDeciLM(unittest.TestCase):

    def _make_decilm_config(self,
                            layer_configs: List[Union[DeciLayerConfig,
                                                      Dict[str, Dict[str,
                                                                     Any]]]],
                            dtype: str = 'bfloat16',
                            num_attention_heads: int = 32,
                            num_key_value_heads: Optional[int] = None,
                            hidden_size: int = 4096,
                            intermediate_size: int = 16384,
                            vocab_size: int = 32128,
                            max_positions_embedding: int = 1024,
                            norm_epsilon: float = 1e-05) -> DeciConfig:
        config = {
            'architecture': 'DeciLMForCausalLM',
            'num_hidden_layers': len(layer_configs),
            'num_attention_heads': num_attention_heads,
            'num_key_value_heads': num_key_value_heads,
            'dtype': dtype,
            'logits_dtype': dtype,
            'hidden_size': hidden_size,
            'intermediate_size': intermediate_size,
            'vocab_size': vocab_size,
            'position_embedding_type': 'rope_gpt_neox',
            'max_position_embeddings': max_positions_embedding,
            'hidden_act': 'silu',
            'norm_epsilon': norm_epsilon,
            'layer_configs': layer_configs
        }

        config = DeciConfig.from_dict(config)
        return config

    def _gen_tensorrt_llm_network(self, network: Network,
                                  decilm: DeciLMForCausalLM, batch_size: int,
                                  beam_width: int, input_len: int,
                                  output_len: int, rank: int,
                                  tensor_parallel: int, **opt_flags):
        list(range(tensor_parallel))

        with net_guard(network):
            # optimize_model(decilm, **opt_flags)
            # Prepare
            network.set_named_parameters(decilm.named_parameters())
            inputs = decilm.prepare_inputs(max_batch_size=batch_size,
                                           max_input_len=input_len,
                                           max_seq_len=input_len + output_len,
                                           max_num_tokens=batch_size *
                                           input_len,
                                           use_cache=True,
                                           max_beam_width=beam_width)
            # Forward
            decilm(**inputs)
        return network

    def _gen_tensorrt_llm_engine(
            self,
            rank: int,
            world_size: int,
            decilm: DeciLMForCausalLM,
            model_name: str,
            use_plugin: bool,
            batch_size: int,
            beam_width: int,
            input_len: int,
            output_len: int,
            use_refit: bool,
            use_gemm: bool = False,
            context_fmha_flag: ContextFMHAType = ContextFMHAType.disabled,
            enable_remove_input_padding: bool = False,
            **opt_flags) -> trt.IHostMemory:

        builder = Builder()
        dtype = decilm.config.dtype

        with tempfile.TemporaryDirectory():
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
            if use_gemm:
                network.plugin_config.gemm_plugin = dtype
            if enable_remove_input_padding:
                network.plugin_config.remove_input_padding = True
            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network=network,
                                           decilm=decilm,
                                           batch_size=batch_size,
                                           beam_width=beam_width,
                                           input_len=input_len,
                                           output_len=output_len,
                                           rank=rank,
                                           tensor_parallel=world_size,
                                           **opt_flags)
            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _gen_tensorrt_llm_runtime(
            self,
            log_level: str,
            world_size: int,
            rank: int,
            decilm: DeciLMForCausalLM,
            model_name: str,
            use_plugin: bool,
            batch_size: int,
            beam_width: int,
            input_len: int,
            output_len: int,
            use_refit: bool,
            use_gemm: bool = False,
            context_fmha_flag: ContextFMHAType = ContextFMHAType.disabled,
            enable_remove_input_padding: bool = False,
            **opt_flags) -> Tuple[_Runtime, trt.IHostMemory]:
        logger.set_level(log_level)
        mapping = Mapping(world_size, rank, tp_size=world_size)
        engine_buffer = self._gen_tensorrt_llm_engine(
            rank=rank,
            world_size=world_size,
            decilm=decilm,
            model_name=model_name,
            use_plugin=use_plugin,
            batch_size=batch_size,
            beam_width=beam_width,
            input_len=input_len,
            output_len=output_len,
            use_refit=use_refit,
            use_gemm=use_gemm,
            context_fmha_flag=context_fmha_flag,
            enable_remove_input_padding=enable_remove_input_padding,
            **opt_flags)
        runtime = _Runtime(engine_buffer, mapping)
        return runtime, engine_buffer

    def test_config_to_from_dict(self) -> None:
        config = self._make_decilm_config(layer_configs=[{
            "attention": {
                "num_key_value_heads": 4
            },
            "ffn": {}
        }, {
            "attention": {
                "num_key_value_heads": 2
            },
            "ffn": {
                "impl": "no_op"
            }
        }, {
            "attention": {
                "impl": "no_op"
            },
            "ffn": {
                "intermediate_size": 8192
            }
        }])

        config2 = DeciConfig.from_dict(config.to_dict())
        self.assertListEqual(config.layer_configs, config2.layer_configs)

    def test_save_load_config(self) -> None:
        config = self._make_decilm_config(layer_configs=[{
            "attention": {
                "num_key_value_heads": 4
            },
            "ffn": {}
        }, {
            "attention": {
                "num_key_value_heads": 2
            },
            "ffn": {
                "impl": "no_op"
            }
        }, {
            "attention": {
                "impl": "no_op"
            },
            "ffn": {
                "intermediate_size": 8192
            }
        }])

        with tempfile.TemporaryDirectory(
                prefix="test_save_load_checkpoint") as ckpt_dir:
            config_file = f"{ckpt_dir}/config.json"
            config.to_json_file(config_file)
            config2 = DeciConfig.from_json_file(config_file)

        self.assertDictEqual(config.to_dict(), config2.to_dict())
        self.assertListEqual(config.layer_configs, config2.layer_configs)

    def get_loader_test_cases():
        model_root = llm_models_root(check=True)
        test_models_base_path = Path(model_root, "nvsmall/tests")

        models_path = [
            os.path.join(test_models_base_path, x)
            for x in os.listdir(test_models_base_path)
        ]
        test_cases = list(
            itertools.product(models_path, ["bfloat16", "float16"]))

        return test_cases

    @parameterized.expand(get_loader_test_cases, name_func=unittest_name_func)
    def test_allclose_to_hf(self, hf_model_dir, dtype):
        if hf_model_dir is None:
            self.skipTest(
                f"Missing nvsmall checkpoint, define a valid checkpoint path with the NVSMALL_CKPT environment variable"
            )

        dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)

        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_dir, trust_remote_code=True, torch_dtype=dtype).cuda()
        decilm = DeciLMForCausalLM.from_hugging_face(hf_model)
        config = decilm.config

        log_level = "warning"
        batch_size = 1
        beam_width = 1
        input_len = 4
        output_len = 2
        max_seq_len = input_len + output_len
        dtype = config.dtype
        enable_remove_input_padding = False
        use_gpt_plugin = True
        use_gemm = True

        runtime, engine_buffer = self._gen_tensorrt_llm_runtime(
            log_level=log_level,
            decilm=decilm,
            batch_size=batch_size,
            beam_width=beam_width,
            input_len=input_len,
            output_len=output_len,
            rank=0,
            world_size=1,
            model_name="decilm",
            use_gemm=use_gemm,
            use_plugin=use_gpt_plugin,
            use_refit=False)

        key_value_cache_buffers = []
        head_size = config.hidden_size // config.num_attention_heads

        attn_layer_idx = [
            i for i in range(config.num_hidden_layers)
            if config.get_layer_config(i).attention.needs_kv_cache
        ]
        for layer_idx in attn_layer_idx:
            layer_config = config.get_layer_config(layer_idx)
            new_cache = torch.zeros((
                batch_size,
                2,
                layer_config.attention.num_key_value_heads,
                max_seq_len,
                head_size,
            ),
                                    dtype=str_dtype_to_torch(dtype),
                                    device='cuda')
            key_value_cache_buffers.append(new_cache)

        # compare context
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
            hf_outputs = hf_model.forward(ctx_ids,
                                          output_hidden_states=True,
                                          output_attentions=True)

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

        ctx_buffer = {
            'input_ids': ctx_ids,
            'context_lengths': ctx_context_lengths,
            'position_ids': ctx_position_ids,
            'last_token_ids': ctx_last_token_ids,
            'cache_indirection': cache_indirections[0],
            'host_request_types': ctx_host_request_types,
            'host_runtime_perf_knobs': context_runtime_perf_knobs,
        }
        if enable_remove_input_padding:
            ctx_buffer['host_context_lengths'] = ctx_context_lengths.cpu()

        ctx_shape = {k: v.shape for k, v in ctx_buffer.items()}

        ctx_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
            [max_seq_len] * len(attn_layer_idx), dtype=torch.int32)
        ctx_shape[f'host_max_attention_window_sizes'] = (len(attn_layer_idx), )
        for layer_idx, buf in zip(attn_layer_idx, key_value_cache_buffers):
            layer_config = config.get_layer_config(layer_idx)
            kv_shape = (batch_size, 2,
                        layer_config.attention.num_key_value_heads, max_seq_len,
                        head_size)
            ctx_shape[f'past_key_value_{layer_idx}'] = kv_shape
            ctx_buffer[f'past_key_value_{layer_idx}'] = buf
            ctx_buffer[f'present_key_value_{layer_idx}'] = buf

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
        gen_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                              dtype=torch.int64)

        with torch.no_grad():
            hf_outputs = hf_model.forward(
                step1_id,
                past_key_values=hf_outputs.past_key_values,
                use_cache=True,
                output_hidden_states=True)

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
            'position_ids': gen_position_ids,
            'last_token_ids': gen_last_token_ids,
            'host_request_types': gen_host_request_types,
            'cache_indirection': cache_indirections[1],
            'host_runtime_perf_knobs': gen_runtime_perf_knobs,
        }
        if enable_remove_input_padding:
            step1_buffer['host_context_lengths'] = gen_context_lengths.cpu()

        step1_shape = {k: v.shape for k, v in step1_buffer.items()}

        sequence_length_buffer = torch.add(sequence_length_buffer, step)
        step1_buffer[f'host_max_attention_window_sizes'] = torch.tensor(
            [max_seq_len] * len(attn_layer_idx), dtype=torch.int32)
        step1_shape[f'host_max_attention_window_sizes'] = (
            len(attn_layer_idx), )
        for layer_idx, buf in zip(attn_layer_idx, key_value_cache_buffers):
            layer_config = config.get_layer_config(layer_idx)
            kv_shape = (batch_size, 2,
                        layer_config.attention.num_key_value_heads, max_seq_len,
                        head_size)
            step1_shape[f"past_key_value_{layer_idx}"] = kv_shape
            step1_buffer[f"past_key_value_{layer_idx}"] = buf
            step1_buffer[f"present_key_value_{layer_idx}"] = buf

        step1_buffer['sequence_length'] = sequence_length_buffer
        step1_shape['sequence_length'] = ctx_buffer['sequence_length'].shape
        step1_shape['sequence_length'] = (batch_size, )
        step1_shape['host_past_key_value_lengths'] = (batch_size, )
        step1_buffer[
            'host_past_key_value_lengths'] = sequence_length_buffer.cpu()
        step1_shape['host_sink_token_length'] = (1, )
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

    @parameterized.expand(
        itertools.product(
            (os.getenv("NVSMALL_CKPT"), ),  # "deci/decilm-7b"),
            (True, False),
            (1, 2),
            (1, 2),
            ("auto", "float16", "bfloat16")))
    def test_convert_config_from_hf(self, ckpt_path: Optional[str],
                                    preloaded: bool, tp_size: int, pp_size: int,
                                    dtype: str) -> None:
        if ckpt_path is None:
            self.skipTest(
                f"Missing nvsmall checkpoint, define a valid checkpoint path with the NVSMALL_CKPT environment variable"
            )

        hf_config = transformers.AutoConfig.from_pretrained(
            ckpt_path, trust_remote_code=True)

        mapping = Mapping(world_size=(tp_size * pp_size),
                          rank=0,
                          gpus_per_node=1,
                          tp_size=tp_size,
                          pp_size=pp_size)

        config = DeciConfig.from_hugging_face(
            hf_config if preloaded else ckpt_path,
            dtype=dtype,
            mapping=mapping,
            trust_remote_code=not preloaded)

        if getattr(hf_config, "num_key_value_heads_per_layer",
                   None) is not None:
            # verify layers for old config
            for layer_idx, num_kv_heads in enumerate(
                    hf_config.num_key_value_heads_per_layer):
                layer_config = config.get_layer_config(layer_idx)
                self.assertEqual(layer_config.attention.impl,
                                 AttentionImplementation.ATTENTION)
                self.assertEqual(num_kv_heads,
                                 layer_config.attention.num_key_value_heads)
                self.assertEqual(layer_config.ffn.impl, FFNImplementation.MLP)
                self.assertEqual(layer_config.ffn.intermediate_size,
                                 config.intermediate_size)

        elif getattr(hf_config, "block_configs", None) is not None:
            # verify layers for new config
            for layer_idx, block_config in enumerate(hf_config.block_configs):
                layer_config = config.get_layer_config(layer_idx)
                if layer_config.attention.impl == AttentionImplementation.ATTENTION:
                    self.assertFalse(block_config.attention.no_op)
                    self.assertFalse(block_config.attention.replace_with_linear)
                    self.assertEqual(
                        config.num_attention_heads //
                        block_config.attention.n_heads_in_group,
                        layer_config.attention.num_key_value_heads)
                elif layer_config.attention.impl == AttentionImplementation.NO_OP:
                    self.assertTrue(block_config.attention.no_op)
                elif layer_config.attention.impl == AttentionImplementation.LINEAR:
                    self.assertTrue(block_config.attention.replace_with_linear)

                if layer_config.ffn.impl == FFNImplementation.MLP:
                    self.assertFalse(block_config.ffn.no_op)
                    self.assertFalse(block_config.ffn.replace_with_linear)
                    self.assertEqual(
                        _ffn_mult_to_intermediate_size(
                            block_config.ffn.ffn_mult, config.hidden_size),
                        layer_config.ffn.intermediate_size)
                elif layer_config.ffn.impl == FFNImplementation.NO_OP:
                    self.assertTrue(block_config.ffn.no_op)
                elif layer_config.ffn.impl == FFNImplementation.LINEAR:
                    self.assertTrue(block_config.ffn.replace_with_linear)

        # verify config is valid enough for model creation
        DeciLMForCausalLM(config)

    @parameterized.expand(
        itertools.product(
            (os.getenv("NVSMALL_CKPT"), ),  # "deci/decilm-7b"),
            (True, False),
            (1, 2),
            (1, 2),
            ("auto", "float16", "bfloat16")))
    def test_convert_model_from_hf(self, ckpt_path: Optional[str],
                                   preloaded: bool, tp_size: int, pp_size: int,
                                   dtype: str) -> None:
        if ckpt_path is None:
            self.skipTest(
                f"Missing nvsmall checkpoint, define a valid checkpoint path with the NVSMALL_CKPT environment variable"
            )

        if preloaded:
            hf_model_or_dir = transformers.AutoModelForCausalLM.from_pretrained(
                ckpt_path, trust_remote_code=True)
        else:
            hf_model_or_dir = ckpt_path

        mapping = Mapping(world_size=(tp_size * pp_size),
                          rank=0,
                          gpus_per_node=1,
                          tp_size=tp_size,
                          pp_size=pp_size)

        DeciLMForCausalLM.from_hugging_face(hf_model_or_dir=hf_model_or_dir,
                                            dtype=dtype,
                                            mapping=mapping,
                                            trust_remote_code=not preloaded)
