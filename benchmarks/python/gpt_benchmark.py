# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
from math import ceil

import torch
from allowed_configs import get_build_config, get_model_family
from base_benchmark import BaseBenchmark, get_engine_name, serialize_engine

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.models import (fp8_quantize, smooth_quantize,
                                 weight_only_quantize)
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode


class GPTBenchmark(BaseBenchmark):

    def __init__(self,
                 engine_dir,
                 model_name,
                 mode,
                 batch_sizes,
                 in_out_lens,
                 dtype,
                 refit,
                 num_beams,
                 top_k,
                 top_p,
                 output_dir,
                 n_positions=None,
                 max_input_len=None,
                 max_output_len=None,
                 max_batch_size=None,
                 enable_custom_all_reduce=None,
                 **kwargs):
        super().__init__(engine_dir, model_name, dtype, output_dir)
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.refit = refit
        self.num_beams = num_beams
        self.build_time = 0
        self.mode = mode  # plugin or ootb
        self.fuse_bias = True

        self.cuda_graph_mode = kwargs.get('enable_cuda_graph', False)
        self.enable_custom_all_reduce = enable_custom_all_reduce

        if engine_dir is not None:
            # Get build configs from engine directory is done in base class
            # Deserialize engine from engine directory
            self.serialize_path = os.path.join(engine_dir, self.engine_name)
            with open(self.serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            # Build engine
            self.world_size = tensorrt_llm.mpi_world_size()
            self.apply_query_key_layer_scaling = False
            self.use_smooth_quant = False
            # this attribute is not stored in allowed_config
            self.enable_fp8 = kwargs.get('enable_fp8', False)
            self.fp8_kv_cache = kwargs.get('fp8_kv_cache', False)

            self.use_weight_only = False
            self.weight_only_precision = 'int8'
            self.per_token = False
            self.per_channel = False

            is_plugin_mode = mode == 'plugin'
            plg_dtype = dtype if is_plugin_mode else False
            self.use_gpt_attention_plugin = plg_dtype
            self.use_gemm_plugin = plg_dtype
            # Starting TRT9.1 OOTB norm layer sees improvement over plugin norm layer
            self.use_layernorm_plugin = False
            self.use_rmsnorm_plugin = False
            self.use_lookup_plugin = plg_dtype
            self.enable_context_fmha = True
            self.quant_mode = QuantMode(0)
            self.remove_input_padding = is_plugin_mode

            for key, value in get_build_config(model_name).items():
                setattr(self, key, value)

            # Override the n_position/max_input_len/max_output_len/max_batch_size to value from cmd line if that's specified.
            if n_positions is not None:
                assert isinstance(
                    n_positions, int
                ) and n_positions > 0, f"n_positions should be a valid int number, got {n_positions}"
                self.n_positions = n_positions
            if max_input_len is not None:
                assert isinstance(
                    max_input_len, int
                ) and max_input_len > 0, f"max_input_len should be a valid int number, got {max_input_len}"
                self.max_input_len = max_input_len
            if max_output_len is not None:
                assert isinstance(
                    max_output_len, int
                ) and max_output_len > 0, f"max_output_len should be a valid int number, got {max_output_len}"
                self.max_output_len = max_output_len
            if max_batch_size is not None:
                assert isinstance(
                    max_batch_size, int
                ) and max_batch_size > 0, f"max_batch_size should be a valid int number, got {max_batch_size}"
                self.max_batch_size = max_batch_size
            if self.num_kv_heads is None:
                self.num_kv_heads = self.num_heads
            if kwargs.get('force_num_layer_1', False):
                self.num_layers = 1

            if self.use_smooth_quant:
                self.quant_mode = QuantMode.use_smooth_quant(
                    self.per_token, self.per_channel)
            elif self.use_weight_only:
                self.quant_mode = QuantMode.use_weight_only(
                    self.weight_only_precision == 'int4')

            if self.enable_fp8:
                self.quant_mode = self.quant_mode.set_fp8_qdq()

            if self.fp8_kv_cache:
                # Watch out, enable_fp8 and fp8_kv_cache are not exclusive
                assert self.use_gpt_attention_plugin, "GPT attention plugin needed"
                self.quant_mode = self.quant_mode.set_fp8_kv_cache()

            engine_buffer = self.build()

        assert engine_buffer is not None

        model_config = tensorrt_llm.runtime.ModelConfig(
            num_heads=self.num_heads // self.world_size,
            num_kv_heads=ceil(self.num_kv_heads / self.world_size),
            hidden_size=self.hidden_size // self.world_size,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            gpt_attention_plugin=self.use_gpt_attention_plugin,
            remove_input_padding=self.remove_input_padding,
            quant_mode=self.quant_mode,
            use_custom_all_reduce=self.enable_custom_all_reduce,
        )
        if model_name == 'chatglm_6b':
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=130005,
                pad_id=3,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p)
            self.decoder = tensorrt_llm.runtime.ChatGLM6BHeadModelGenerationSession(
                model_config, engine_buffer, self.runtime_mapping)
        else:
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=50256,
                pad_id=50256,
                num_beams=num_beams,
                top_k=top_k,
                top_p=top_p)
            self.decoder = tensorrt_llm.runtime.GenerationSession(
                model_config,
                engine_buffer,
                self.runtime_mapping,
                cuda_graph_mode=self.cuda_graph_mode)

    def get_config(self):
        for inlen, outlen in self.in_out_lens:
            if inlen > self.max_input_len or outlen > self.max_output_len:
                print(
                    f'[WARNING] check inlen({inlen}) <= max_inlen({self.max_input_len}) and '
                    f'outlen({outlen}) <= max_outlen({self.max_output_len}) failed, skipping.'
                )
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    print(
                        f'[WARNING] check batch_size({batch_size}) '
                        f'<= max_batch_size({self.max_batch_size}) failed, skipping.'
                    )
                    continue
                yield (batch_size, inlen, outlen)

    def prepare_inputs(self, config):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        input_ids = torch.randint(100, (batch_size, inlen)).int().cuda()
        input_lengths = torch.tensor([inlen
                                      for _ in range(batch_size)]).int().cuda()

        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        return (input_ids, input_lengths)

    def build(self):
        builder = Builder()
        builder_config = builder.create_builder_config(
            name=self.model_name,
            precision=self.dtype,
            timing_cache=None,
            tensor_parallel=self.world_size,  # TP only
            parallel_build=True,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.n_positions,
            apply_query_key_layer_scaling=self.apply_query_key_layer_scaling,
            max_batch_size=self.max_batch_size,
            max_input_len=self.max_input_len,
            max_output_len=self.max_output_len,
            int8=self.quant_mode.has_act_and_weight_quant(),
            fp8=self.quant_mode.has_fp8_qdq(),
            quant_mode=self.quant_mode,
            use_refit=self.refit,
            opt_level=self.builder_opt)
        engine_name = get_engine_name(self.model_name, self.dtype,
                                      self.world_size, self.runtime_rank)

        kv_dtype = str_dtype_to_trt(self.dtype)

        # Initialize Module
        family = get_model_family(self.model_name)
        if family == "gpt":
            tensorrt_llm_model = tensorrt_llm.models.GPTLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                position_embedding_type=PositionEmbeddingType.learned_absolute
                if self.position_embedding_type is None else
                self.position_embedding_type,
                rotary_embedding_percentage=self.rotary_pct,
                quant_mode=self.quant_mode,
                bias=self.bias)
        elif family == "opt":
            tensorrt_llm_model = tensorrt_llm.models.OPTLMHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                pre_norm=self.pre_norm,
                do_layer_norm_before=self.do_layer_norm_before)
        elif family == "llama":
            tensorrt_llm_model = tensorrt_llm.models.LLaMAForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mlp_hidden_size=self.inter_size,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                quant_mode=self.quant_mode)
        elif family == "gptj":
            tensorrt_llm_model = tensorrt_llm.models.GPTJForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                rotary_dim=self.rotary_dim,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                quant_mode=self.quant_mode)
        elif family == "gptneox":
            tensorrt_llm_model = tensorrt_llm.models.GPTNeoXForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                rotary_dim=self.rotary_dim,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling)
        elif family == "chatglm":
            tensorrt_llm_model = tensorrt_llm.models.ChatGLM6BHeadModel(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                hidden_act=self.hidden_act,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                apply_query_key_layer_scaling=builder_config.
                apply_query_key_layer_scaling,
                quant_mode=self.quant_mode)
        elif family == "bloom":
            tensorrt_llm_model = tensorrt_llm.models.BloomForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                mapping=tensorrt_llm.Mapping(
                    world_size=self.world_size,
                    tp_size=self.world_size),  # TP only
                use_parallel_embedding=(self.model_name == 'bloom_176b'))
        elif family == "falcon":
            tensorrt_llm_model = tensorrt_llm.models.FalconForCausalLM(
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_position_embeddings=self.n_positions,
                dtype=kv_dtype,
                bias=self.bias,
                use_alibi=self.use_alibi,
                new_decoder_architecture=self.new_decoder_architecture,
                parallel_attention=self.parallel_attention,
                mapping=tensorrt_llm.Mapping(world_size=self.world_size,
                                             tp_size=self.world_size))
        else:
            raise Exception(f'Unexpected model: {self.model_name}')

        if self.use_smooth_quant:
            tensorrt_llm_model = smooth_quantize(tensorrt_llm_model,
                                                 self.quant_mode)
        elif self.use_weight_only and self.weight_only_precision == 'int8':
            tensorrt_llm_model = weight_only_quantize(
                tensorrt_llm_model, QuantMode.use_weight_only())
        elif self.use_weight_only and self.weight_only_precision == 'int4':
            tensorrt_llm_model = weight_only_quantize(
                tensorrt_llm_model,
                QuantMode.use_weight_only(use_int4_weights=True))
        elif self.enable_fp8 or self.fp8_kv_cache:
            tensorrt_llm_model = fp8_quantize(tensorrt_llm_model,
                                              self.quant_mode)

        # Module -> Network
        network = builder.create_network()
        network.trt_network.name = engine_name
        if self.use_gpt_attention_plugin:
            network.plugin_config.set_gpt_attention_plugin(
                dtype=self.use_gpt_attention_plugin)
        if self.use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype=self.use_gemm_plugin)
        if self.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(
                dtype=self.use_layernorm_plugin)
        if self.use_rmsnorm_plugin:
            network.plugin_config.set_rmsnorm_plugin(
                dtype=self.use_rmsnorm_plugin)
        if self.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if self.remove_input_padding:
            network.plugin_config.enable_remove_input_padding()

        # Quantization plugins.
        if self.use_smooth_quant:
            network.plugin_config.set_smooth_quant_gemm_plugin(dtype=self.dtype)
            network.plugin_config.set_layernorm_quantization_plugin(
                dtype=self.dtype)
            network.plugin_config.set_quantize_tensor_plugin()
            network.plugin_config.set_quantize_per_token_plugin()
        elif self.use_weight_only:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype=self.dtype)

        # RMS norm plugin for SmoothQuant
        if self.quant_mode.has_act_and_weight_quant(
        ) and 'llama' in self.model_name:
            network.plugin_config.set_rmsnorm_quantization_plugin()

        if self.world_size > 1:
            network.plugin_config.set_nccl_plugin(self.dtype,
                                                  self.enable_custom_all_reduce)

        # Use the plugin for the embedding parallism and sharing
        network.plugin_config.set_lookup_plugin(dtype=self.use_lookup_plugin)

        with net_guard(network):
            # Prepare
            network.set_named_parameters(tensorrt_llm_model.named_parameters())

            # Forward
            inputs = tensorrt_llm_model.prepare_inputs(self.max_batch_size,
                                                       self.max_input_len,
                                                       self.max_output_len,
                                                       True, self.num_beams)
            tensorrt_llm_model(*inputs)

        if self.fuse_bias:
            tensorrt_llm.graph_rewriting.optimize(network)

        # Network -> Engine
        start = time.time()
        engine = builder.build_engine(network, builder_config)
        end = time.time()
        self.build_time = round(end - start, 2)

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.serialize_path = os.path.join(self.output_dir,
                                               self.engine_name)
            serialize_engine(engine, self.serialize_path)
            if self.runtime_rank == 0:
                config_path = os.path.join(self.output_dir, 'config.json')
                builder_config.plugin_config = network.plugin_config
                builder.save_config(builder_config, config_path)
        return engine

    def run(self, inputs, config):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        if self.remove_input_padding:
            self.decoder.decode_batch(inputs[0], self.sampling_config)
        else:
            self.decoder.decode(inputs[0], inputs[1], self.sampling_config)
        torch.cuda.synchronize()

    def report(self, config, latency, percentile95, percentile99, peak_gpu_used,
               csv):
        report_dict = super().get_report_dict()
        batch_size, inlen, outlen = config[0], config[1], config[2]
        tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
        report_dict["num_heads"] = self.num_heads
        report_dict["num_kv_heads"] = self.num_kv_heads
        report_dict["num_layers"] = self.num_layers
        report_dict["hidden_size"] = self.hidden_size
        report_dict["vocab_size"] = self.vocab_size
        report_dict["batch_size"] = batch_size
        report_dict["input_length"] = inlen
        report_dict["output_length"] = outlen
        report_dict["latency(ms)"] = latency
        report_dict["build_time(s)"] = self.build_time
        report_dict["tokens_per_sec"] = tokens_per_sec
        report_dict["percentile95(ms)"] = percentile95
        report_dict["percentile99(ms)"] = percentile99
        report_dict["gpu_peak_mem(gb)"] = peak_gpu_used
        if self.runtime_rank == 0:
            if csv:
                line = ",".join([str(v) for v in report_dict.values()])
                print(line)
                with open(self.get_csv_filename(), "a") as file:
                    file.write(line + "\n")
            else:
                kv_pairs = [f"{k} {v}" for k, v in report_dict.items()]
                line = '[BENCHMARK] ' + " ".join(kv_pairs)
                print(line)
