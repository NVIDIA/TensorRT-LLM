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
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import tensorrt as trt
import torch
import transformers
from parameterized import parameterized
from transformers import AutoTokenizer
from typing_extensions import Literal
from utils.llm_data import llm_datasets_root, llm_models_root
from utils.util import get_project_root, unittest_name_func

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.builder import Builder, Engine, EngineConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import PretrainedConfig
from tensorrt_llm.models.nemotron_nas.config import DeciConfig, DeciLayerConfig
from tensorrt_llm.models.nemotron_nas.convert import (
    _ffn_mult_to_intermediate_size, load_weights_from_hf_safetensors)
from tensorrt_llm.models.nemotron_nas.layer_config import (
    AttentionImplementation, FFNImplementation)
from tensorrt_llm.models.nemotron_nas.model import DeciLMForCausalLM
from tensorrt_llm.network import Network, net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime.generation import _Runtime
from tensorrt_llm.runtime.kv_cache_manager import GenerationSequence
from tensorrt_llm.runtime.memory_pools.memory_pools_allocator import \
    MemoryPoolsAllocator
from tensorrt_llm.runtime.memory_pools.pools_kv_cache_manager import \
    PoolsKVCacheManager
from tensorrt_llm.runtime.model_runner import ModelRunner
from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp


@dataclass(kw_only=True, frozen=True)
class TestParams:
    enable_paged_kv_cache: bool
    enable_remove_input_padding: bool
    dtype: Literal["float16", "bfloat16"]

    batch_size: int = 1
    beam_width: int = 1
    seq_len: int = 128
    total_length: int = seq_len + 2
    tokens_per_block: int = 128

    @property
    def output_len(self):
        return self.total_length - self.seq_len

    def __str__(self) -> str:
        """tests/utils/util.py#L143 - > `str(x)`: parameterized test name"""
        properties_without_default = (self.enable_paged_kv_cache,
                                      self.enable_remove_input_padding,
                                      self.dtype)
        return "_".join((parameterized.to_safe_name(prop).lower()
                         for prop in properties_without_default))

    @property
    def mapping(self) -> Mapping:
        return Mapping(world_size=1, rank=0, tp_size=1)


@dataclass
class RuntimeHandle:
    """Deleting `Runtime().runtime` will **definitively** deallocate the weights."""
    runtime: _Runtime


QUANTIZED_DIR_PREFIX = "/tmp/nemotron/quantized"


@cache
def quantized(model_dir: str) -> str:
    root = get_project_root(__file__)
    quantize_path = str(root / "examples/quantization/quantize.py")

    quantize_dir = f"{QUANTIZED_DIR_PREFIX}/{Path(model_dir).stem}"

    quantize = [
        sys.executable,
        quantize_path,
        f"--model_dir={model_dir}",
        f"--output_dir={quantize_dir}",
        "--dtype=bfloat16",
        "--kv_cache_dtype=fp8",
        "--qformat=fp8",
        f"--calib_dataset={llm_datasets_root()}/cnn_dailymail",
        "--calib_size=1",  # It's a test, so calibration won't really be useful. So keep it short.
    ]
    print(f"Running quantize: {quantize}")
    subprocess.run(quantize, check=True)

    return quantize_dir


class TestNemotronNas(unittest.TestCase):

    def _make_config(self,
                     layer_configs: List[Union[DeciLayerConfig,
                                               Dict[str, Dict[str, Any]]]],
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
                                  model: DeciLMForCausalLM, batch_size: int,
                                  beam_width: int, input_len: int,
                                  output_len: int, rank: int,
                                  tensor_parallel: int, **opt_flags):
        list(range(tensor_parallel))

        with net_guard(network):
            # Prepare
            network.set_named_parameters(model.named_parameters())
            inputs = model.prepare_inputs(max_batch_size=batch_size,
                                          max_input_len=input_len,
                                          max_seq_len=input_len + output_len,
                                          max_num_tokens=batch_size * input_len,
                                          use_cache=True,
                                          max_beam_width=beam_width)
            # Forward
            model(**inputs)
        return network

    def _gen_tensorrt_llm_engine(
            self,
            rank: int,
            world_size: int,
            model: DeciLMForCausalLM,
            model_name: str,
            use_plugin: bool,
            batch_size: int,
            beam_width: int,
            input_len: int,
            output_len: int,
            tokens_per_block: int,
            use_refit: bool,
            use_gemm: bool = False,
            context_fmha_flag: ContextFMHAType = ContextFMHAType.disabled,
            enable_remove_input_padding: bool = False,
            enable_paged_kv_cache: bool = False,
            **opt_flags) -> trt.IHostMemory:

        builder = Builder()
        dtype = model.config.dtype

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
            if enable_paged_kv_cache:
                network.plugin_config.enable_paged_kv_cache(tokens_per_block)

            network.plugin_config.set_context_fmha(context_fmha_flag)

            self._gen_tensorrt_llm_network(network=network,
                                           model=model,
                                           batch_size=batch_size,
                                           beam_width=beam_width,
                                           input_len=input_len,
                                           output_len=output_len,
                                           rank=rank,
                                           tensor_parallel=world_size,
                                           **opt_flags)
            engine_buffer = builder.build_engine(network, builder_config)
            return engine_buffer

    def _from_hf_model(
            self,
            hf_model: transformers.AutoModelForCausalLM,
            params: TestParams,
            *,
            model_name: str = "nemotron-nas",
            use_plugin: bool = True,
            use_refit: bool = False,
            use_gemm: bool = True,
            context_fmha_flag: ContextFMHAType = ContextFMHAType.disabled,
            **opt_flags) -> Tuple[RuntimeHandle, PretrainedConfig]:
        model = DeciLMForCausalLM.from_hugging_face(hf_model)
        logger.set_level("warning")
        mapping = params.mapping
        engine_buffer = self._gen_tensorrt_llm_engine(
            rank=mapping.rank,
            world_size=mapping.world_size,
            model=model,
            model_name=model_name,
            use_plugin=use_plugin,
            batch_size=params.batch_size,
            beam_width=params.beam_width,
            input_len=params.seq_len,
            output_len=params.output_len,
            use_refit=use_refit,
            use_gemm=use_gemm,
            context_fmha_flag=context_fmha_flag,
            enable_remove_input_padding=params.enable_remove_input_padding,
            tokens_per_block=params.tokens_per_block,
            enable_paged_kv_cache=params.enable_paged_kv_cache,
            **opt_flags)
        runtime = RuntimeHandle(_Runtime(engine_buffer, mapping))
        return runtime, model.config

    @classmethod
    def tearDownClass(cls) -> None:
        if Path(QUANTIZED_DIR_PREFIX).is_dir():
            shutil.rmtree(QUANTIZED_DIR_PREFIX)

    def _from_fp8_quantized_engine(
            self, *, model_dir: str,
            params: TestParams) -> Tuple[RuntimeHandle, PretrainedConfig]:
        quantize_dir = quantized(model_dir)

        engine_path = f"{quantize_dir}/engine"
        build = [
            "trtllm-build",
            f"--checkpoint_dir={quantize_dir}",
            f"--output_dir={engine_path}",
            f"--max_input_len={params.seq_len}",
            f"--max_batch_size={params.batch_size}",
            f"--remove_input_padding={'enable' if params.enable_remove_input_padding else 'disable'}",
            f"--kv_cache_type={'paged' if params.enable_paged_kv_cache else 'continuous'}",
            "--gemm_plugin=auto",
            "--gpt_attention_plugin=auto",
        ]

        if params.enable_paged_kv_cache:
            build.append(f"--tokens_per_block={params.tokens_per_block}")

        print(f"Running trtllm-build: {build}")
        subprocess.run(build, check=True)

        engine = Engine.from_dir(engine_path)
        runtime = RuntimeHandle(_Runtime(engine.engine, params.mapping))
        config = EngineConfig.from_json_file(f"{engine_path}/config.json")

        return runtime, config.pretrained_config

    def test_config_to_from_dict(self) -> None:
        config = self._make_config(layer_configs=[{
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
        config = self._make_config(layer_configs=[{
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

        params_product = [
            TestParams(
                enable_paged_kv_cache=paged,
                enable_remove_input_padding=padded,
                dtype=dtype,
            ) for paged, padded, dtype in itertools.product(
                [True, False],
                [True, False],
                ["bfloat16", "float16"],
            )
        ]
        test_cases = list(itertools.product(models_path, params_product))

        return test_cases

    @parameterized.expand(get_loader_test_cases, name_func=unittest_name_func)
    def test_allclose_to_hf(self, hf_model_dir: str, params: TestParams):
        self.skipTest(f"https://nvbugs/5444611")
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_dir,
            trust_remote_code=True,
            dtype=tensorrt_llm._utils.str_dtype_to_torch(params.dtype),
        ).cuda()
        runtime, config = self._from_hf_model(hf_model, params)
        self.allclose(
            runtime,
            config=config,
            params=params,
            obtain_hf_model=lambda: hf_model,
        )

    def allclose(
        self,
        runtime_handle: RuntimeHandle,
        *,
        config: PretrainedConfig,
        params: TestParams,
        obtain_hf_model: Callable[[], transformers.AutoModelForCausalLM],
        atol: int = 1e-1,
    ):
        batch_size = params.batch_size
        beam_width = params.beam_width
        seq_len = params.seq_len
        total_length = params.total_length
        dtype = config.dtype
        tokens_per_block = params.tokens_per_block
        enable_remove_input_padding = params.enable_remove_input_padding
        enable_paged_kv_cache = params.enable_paged_kv_cache

        key_value_cache_buffers = []
        head_size = config.hidden_size // config.num_attention_heads
        attn_layer_idx = [
            i for i in range(config.num_hidden_layers)
            if config.get_layer_config(i).attention.needs_kv_cache
        ]

        if enable_paged_kv_cache:
            num_blocks = batch_size * beam_width * math.ceil(
                total_length / tokens_per_block)

            memory_pools_allocator = MemoryPoolsAllocator(
                num_blocks=num_blocks,
                tokens_per_block=tokens_per_block,
                head_size=head_size)
            if config.num_kv_heads_per_layer is None:
                num_kv_heads = config.get_layer_config(
                    attn_layer_idx[0]).attention.num_key_value_heads
                num_kv_heads_per_layer = MemoryPoolsAllocator.prepare_num_kv_heads_per_layer(
                    num_kv_heads, len(attn_layer_idx))
            else:
                num_kv_heads_per_layer = config.num_kv_heads_per_layer

            memory_pools_allocator.allocate(dtype, num_kv_heads_per_layer)
            max_blocks_per_seq = math.ceil(total_length / tokens_per_block)
            num_blocks = batch_size * beam_width * max_blocks_per_seq

            pools_kv_cache_manager = PoolsKVCacheManager(
                memory_pools_allocator.pools_metadata,
                max_blocks_per_seq,
                num_blocks,
                tokens_per_block,
                head_size,
                max_attention_window_size=total_length,
                beam_width=beam_width,
                sink_token_len=0)
            # Add sequences to the manager
            for bi in range(batch_size):
                generation_sequence = GenerationSequence(seq_idx=bi,
                                                         batch_idx=bi)
                pools_kv_cache_manager.add_sequence(generation_sequence,
                                                    seq_len)

            # Pre allocate the kv cache for the generated tokens.
            pools_kv_cache_manager.step([False] * batch_size)

        else:
            for layer_idx in attn_layer_idx:
                layer_config = config.get_layer_config(layer_idx)
                new_cache = torch.zeros((
                    batch_size,
                    2,
                    layer_config.attention.num_key_value_heads,
                    total_length,
                    head_size,
                ),
                                        dtype=str_dtype_to_torch(dtype),
                                        device='cuda')
                key_value_cache_buffers.append(new_cache)

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
                target_shape = [shape[0], shape[1] * shape[2], *shape[3:]]
                ctx_buffer[
                    f'kv_cache_block_offsets'] = kv_cache_block_offsets.reshape(
                        target_shape)
                ctx_buffer[
                    f'host_kv_cache_block_offsets'] = host_kv_cache_block_offsets.reshape(
                        target_shape)
                ctx_buffer[
                    f'host_kv_cache_pool_pointers'] = memory_pools_allocator.get_kv_cache_pool_pointers(
                    ).contiguous()
                ctx_buffer[
                    f'host_kv_cache_pool_mapping'] = memory_pools_allocator.pool_mapping.contiguous(
                    )
                ctx_buffer[
                    f'host_max_attention_window_sizes'] = host_max_attention_window_sizes
            else:
                for layer_idx, buf in zip(attn_layer_idx,
                                          key_value_cache_buffers):
                    ctx_buffer[f'past_key_value_{layer_idx}'] = buf
                    ctx_buffer[f'present_key_value_{layer_idx}'] = buf
                ctx_buffer[
                    f'host_max_attention_window_sizes'] = host_max_attention_window_sizes

            ctx_shape = {
                key: buffer.shape
                for key, buffer in ctx_buffer.items()
            }

            runtime_handle.runtime._set_shape(context, ctx_shape)
            runtime_handle.runtime._set_buffer(context, ctx_buffer)
            runtime_handle.runtime._run(context)
            torch.cuda.synchronize()
            res = ctx_buffer['logits']
            return res

        step0_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        step1_ids = torch.randint(100, (batch_size, 1)).int().cuda()

        def tllm() -> Tuple[np.ndarray, np.ndarray]:
            ctx_ids = step0_ids.clone()

            ctx_context_lengths = seq_len * torch.ones(
                (batch_size), dtype=torch.int32, device='cuda')
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

            host_max_attention_window_sizes = torch.tensor([total_length] *
                                                           len(attn_layer_idx),
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
            host_context_progress = torch.tensor([0], dtype=torch.int64)

            step0 = run_engine(
                context=runtime_handle.runtime.ctx_context,
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

            step = 1
            gen_ids = step1_ids.clone()

            gen_context_lengths = seq_len * torch.ones(
                (batch_size), dtype=torch.int32, device='cuda')
            gen_position_ids = torch.ones_like(gen_ids).int().cuda() * seq_len
            gen_last_token_ids = torch.zeros_like(
                gen_context_lengths).int().cuda()

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
                                                           len(attn_layer_idx),
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

            step1 = run_engine(
                context=runtime_handle.runtime.context_1,
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

            return step0, step1

        def hf() -> Tuple[np.ndarray, np.ndarray]:
            with torch.no_grad():
                hf_model = obtain_hf_model()
                step0_outputs = hf_model.forward(step0_ids.clone())
                torch.cuda.synchronize()
                step0 = step0_outputs.logits[:, -1, :]
                step1_outputs = hf_model.forward(
                    step1_ids.clone(),
                    past_key_values=step0_outputs.past_key_values,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                step1 = step1_outputs.logits[:, -1, :]

            return step0, step1

        res_step0, res_step1 = tllm()
        del runtime_handle.runtime
        ref_step0, ref_step1 = hf()
        np.testing.assert_allclose(ref_step0.cpu().numpy().flatten(),
                                   res_step0.cpu().numpy().flatten(),
                                   atol=atol)
        np.testing.assert_allclose(ref_step1.cpu().numpy().flatten(),
                                   res_step1.cpu().numpy().flatten(),
                                   atol=atol)

    @parameterized.expand(get_loader_test_cases, name_func=unittest_name_func)
    def test_allclose_to_hf_fp8(self, hf_model_dir: str, params: TestParams):
        runtime, config = self._from_fp8_quantized_engine(
            model_dir=hf_model_dir, params=params)
        self.allclose(
            runtime,
            config=config,
            params=params,
            obtain_hf_model=lambda: transformers.AutoModelForCausalLM.
            from_pretrained(
                hf_model_dir,
                trust_remote_code=True,
                dtype=tensorrt_llm._utils.str_dtype_to_torch(params.dtype),
            ).cuda(),
            atol=
            0.92,  # We've observed that on a real checkpoint with the current code, fp8 MMLU is on par with BF16, and this is the observed threshold, though it may seem high.
        )

    @pytest.mark.skipif(
        os.environ.get("NEMOTRON_NAS_CKPT") is None,
        reason="You must define NEMOTRON_NAS_CKPT",
    )
    def test_allclose_to_hf_fp8_accelerate(self):
        hf_model_dir = os.environ["NEMOTRON_NAS_CKPT"]
        params = TestParams(enable_paged_kv_cache=True,
                            enable_remove_input_padding=True,
                            dtype="float16",
                            seq_len=2048)
        runtime, config = self._from_fp8_quantized_engine(
            model_dir=hf_model_dir, params=params)
        self.allclose(
            runtime,
            config=config,
            params=params,
            obtain_hf_model=lambda: transformers.AutoModelForCausalLM.
            from_pretrained(
                hf_model_dir,
                trust_remote_code=True,
                dtype=tensorrt_llm._utils.str_dtype_to_torch(params.dtype),
                device_map="auto",
            ),
        )

    @parameterized.expand(
        itertools.product(("nvidia/Llama-3_1-Nemotron-51B-Instruct", ),
                          (True, False), (1, 2), (1, 2),
                          ("auto", "float16", "bfloat16")))
    def test_convert_config_from_hf(self, ckpt_path: Optional[str],
                                    preloaded: bool, tp_size: int, pp_size: int,
                                    dtype: str) -> None:
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
            os.listdir(
                Path(llm_models_root(check=True), "nvsmall/tests").as_posix()),
            (True, False), (1, 2), (1, 2), ("auto", "float16", "bfloat16")))
    def test_convert_model_from_hf(self, model_dir: Optional[str],
                                   preloaded: bool, tp_size: int, pp_size: int,
                                   dtype: str) -> None:
        self.skipTest(f"https://nvbugs/5444611")
        ckpt_path = Path(llm_models_root(check=True), "nvsmall/tests",
                         model_dir)

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

    @parameterized.expand(
        itertools.product(
            os.listdir(
                Path(llm_models_root(check=True), "nvsmall/tests").as_posix()),
            (1, 2, 4)))
    def test_weights_loader(self, model_dir: str, tp_size: int) -> None:

        ckpt_path = Path(llm_models_root(check=True), "nvsmall/tests",
                         model_dir)
        config = DeciConfig.from_hugging_face(ckpt_path, trust_remote_code=True)
        weights = load_weights_from_hf_safetensors(ckpt_path, config)

        shard_configs = [
            DeciConfig.from_hugging_face(ckpt_path,
                                         trust_remote_code=True,
                                         mapping=Mapping(world_size=tp_size,
                                                         tp_size=tp_size,
                                                         rank=rank))
            for rank in range(tp_size)
        ]
        shard_weights = [
            load_weights_from_hf_safetensors(ckpt_path, shard_config)
            for shard_config in shard_configs
        ]

        for name, param in weights.items():
            shards = [shard[name] for shard in shard_weights]

            if name.endswith("attention.weight"):
                # linear attention
                combined = torch.cat(shards, dim=0)
                torch.testing.assert_close(combined, param, atol=0, rtol=0)
            elif name.endswith("attention.qkv.weight"):
                # proper attention
                layer_idx = int(
                    re.match("transformer.layers.(\\d+).", name).groups()[0])
                layer_config = config.layer_configs[layer_idx]
                num_kv_heads = int(layer_config.attention.num_key_value_heads)
                num_kv_heads_tp = (num_kv_heads + tp_size - 1) // tp_size
                dups = tp_size // num_kv_heads or 1
                q, k, v = torch.split(param, [
                    config.num_attention_heads * config.head_size,
                    num_kv_heads * config.head_size,
                    num_kv_heads * config.head_size
                ])

                q_shards, k_shards, v_shards = [], [], []
                for rank, shard in enumerate(shards):
                    qt, kt, vt = torch.split(
                        shard,
                        [(config.num_attention_heads // tp_size) *
                         config.head_size, num_kv_heads_tp * config.head_size,
                         num_kv_heads_tp * config.head_size])
                    q_shards.append(qt)
                    if rank % dups == 0:
                        k_shards.append(kt)
                        v_shards.append(vt)

                combined_q = torch.cat(q_shards, dim=0)
                combined_k = torch.cat(k_shards, dim=0)
                combined_v = torch.cat(v_shards, dim=0)

                torch.testing.assert_close(combined_q, q, atol=0, rtol=0)
                torch.testing.assert_close(combined_k, k, atol=0, rtol=0)
                torch.testing.assert_close(combined_v, v, atol=0, rtol=0)

    @parameterized.expand(itertools.product([True, False],
                                            ["float16", "bfloat16"], [None],
                                            [None]),
                          name_func=unittest_name_func)
    def test_vgqa_model_runner_allclose(self, use_py_session, dtype, engine_dir,
                                        hf_model_dir):
        input_text = "Born in north-east France, Soyer trained as a"
        tokenizer_dir = hf_model_dir

        if engine_dir is None or not Path(engine_dir).exists:
            self.skipTest(f"Engine dir is either None or doesn't exist")
        if hf_model_dir is None or not Path(hf_model_dir).exists:
            self.skipTest(
                f"Missing HF checkpoint, define a valid checkpoint path with the NEMOTRON_NAS_CKPT environment variable"
            )

        dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)

        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_dir, trust_remote_code=True, dtype=dtype).cuda()

        batch_size = 1
        max_seq_len = 30

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                  padding_side="left",
                                                  truncation_side="left",
                                                  trust_remote_code=True,
                                                  use_fast=True)
        batch_input_ids = [
            torch.tensor(tokenizer.encode(input_text,
                                          add_special_tokens=True,
                                          truncation=True),
                         dtype=torch.int32)
        ]

        hf_batch_ids = batch_input_ids[0].unsqueeze(0).repeat(batch_size,
                                                              1).cuda()
        in_tokens = batch_input_ids[0].shape[0]

        with torch.no_grad():
            hf_outputs = hf_model.generate(hf_batch_ids, max_length=max_seq_len)

        torch.cuda.synchronize()

        if use_py_session:
            runner = ModelRunner.from_dir(engine_dir=engine_dir,
                                          rank=0,
                                          debug_mode=False)

        else:
            runner = ModelRunnerCpp.from_dir(engine_dir=engine_dir,
                                             rank=0,
                                             debug_mode=False)

        pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            runner_outputs = runner.generate(batch_input_ids=batch_input_ids,
                                             max_new_tokens=max_seq_len -
                                             in_tokens,
                                             end_id=tokenizer.eos_token_id,
                                             pad_id=pad_token_id,
                                             output_sequence_lengths=True,
                                             return_dict=False)

        torch.cuda.synchronize()

        del runner

        if not use_py_session:
            np.testing.assert_allclose(
                runner_outputs[0][0][:max_seq_len].cpu().numpy(),
                hf_outputs[0].cpu().numpy())
        else:
            np.testing.assert_allclose(runner_outputs[0].cpu().numpy(),
                                       hf_outputs.cpu().numpy())
