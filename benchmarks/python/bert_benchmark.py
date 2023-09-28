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
from collections import OrderedDict

import tensorrt as trt
import torch
from allowed_configs import get_build_config
from base_benchmark import BaseBenchmark, serialize_engine

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.runtime import TensorInfo


class BERTBenchmark(BaseBenchmark):

    def __init__(self,
                 engine_dir,
                 model_name,
                 mode,
                 batch_sizes,
                 in_lens,
                 dtype,
                 output_dir,
                 n_positions=None,
                 max_input_len=None,
                 max_output_len=None,
                 max_batch_size=None,
                 **kwargs):
        super().__init__(engine_dir, model_name, dtype, output_dir)
        self.batch_sizes = batch_sizes
        self.in_lens = in_lens
        self.build_time = 0

        if engine_dir is not None:
            # Deserialize engine from engine directory
            self.serialize_path = os.path.join(engine_dir, self.engine_name)
            with open(self.serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            # Build engine
            self.use_bert_attention_plugin = False
            self.use_gemm_plugin = False
            self.use_layernorm_plugin = False
            self.enable_qk_half_accum = False
            self.enable_context_fmha = False
            if mode == 'plugin':
                self.use_bert_attention_plugin = dtype
                self.use_gemm_plugin = dtype
                self.use_layernorm_plugin = dtype
            for key, value in get_build_config(model_name).items():
                setattr(self, key, value)
            # Override the n_positions/max_input_len/max_output_len/max_batch_size to value from cmd line if that's specified.
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
            if kwargs.get('force_num_layer_1', False):
                self.num_layers = 1

            engine_buffer = self.build()

        assert engine_buffer is not None

        self.session = tensorrt_llm.runtime.Session.from_serialized_engine(
            engine_buffer)

    def get_config(self):
        for inlen in self.in_lens:
            if inlen > self.max_input_len:
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    continue
                yield (batch_size, inlen)

    def prepare_inputs(self, config):
        batch_size, inlen = config[0], config[1]
        input_ids = torch.randint(100, (batch_size, inlen)).int().cuda()
        input_lengths = inlen * torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda')
        inputs = {'input_ids': input_ids, 'input_lengths': input_lengths}
        output_info = self.session.infer_shapes([
            TensorInfo('input_ids', trt.DataType.INT32, input_ids.shape),
            TensorInfo('input_lengths', trt.DataType.INT32, input_lengths.shape)
        ])
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        stream = torch.cuda.current_stream().cuda_stream
        return (inputs, outputs, stream)

    def build(self):
        bs_range = [1, (self.max_batch_size + 1) // 2, self.max_batch_size]
        inlen_range = [1, (self.max_input_len + 1) // 2, self.max_input_len]

        builder = Builder()
        builder_config = builder.create_builder_config(
            name=self.model_name,
            precision=self.dtype,
            timing_cache=None,
            tensor_parallel=self.world_size,  # TP only
            parallel_build=True,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.n_positions,
            max_batch_size=self.max_batch_size,
            max_input_len=self.max_input_len,
            opt_level=self.builder_opt)
        # Initialize model
        tensorrt_llm_bert = tensorrt_llm.models.BertModel(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.n_positions,
            type_vocab_size=self.type_vocab_size,
            mapping=tensorrt_llm.Mapping(world_size=self.world_size,
                                         tp_size=self.world_size))

        # Module -> Network
        network = builder.create_network()
        if self.use_bert_attention_plugin:
            network.plugin_config.set_bert_attention_plugin(
                dtype=self.use_bert_attention_plugin)
        if self.use_gemm_plugin:
            network.plugin_config.set_gemm_plugin(dtype=self.use_gemm_plugin)
        if self.use_layernorm_plugin:
            network.plugin_config.set_layernorm_plugin(
                dtype=self.use_layernorm_plugin)
        if self.enable_qk_half_accum:
            network.plugin_config.enable_qk_half_accum()
        if self.enable_context_fmha:
            network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        if self.world_size > 1:
            network.plugin_config.set_nccl_plugin(self.dtype)
        with net_guard(network):
            # Prepare
            network.set_named_parameters(tensorrt_llm_bert.named_parameters())

            # Forward
            input_ids = tensorrt_llm.Tensor(
                name='input_ids',
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict([('batch_size', [bs_range]),
                                       ('input_len', [inlen_range])]),
            )
            input_lengths = tensorrt_llm.Tensor(name='input_lengths',
                                                dtype=trt.int32,
                                                shape=[-1],
                                                dim_range=OrderedDict([
                                                    ('batch_size', [bs_range])
                                                ]))
            hidden_states = tensorrt_llm_bert(input_ids=input_ids,
                                              input_lengths=input_lengths)

            # Mark outputs
            hidden_states_dtype = str_dtype_to_trt(self.dtype)
            hidden_states.mark_output('hidden_states', hidden_states_dtype)

        # Network -> Engine
        start = time.time()
        engine = builder.build_engine(network, builder_config)
        end = time.time()
        self.build_time = round(end - start, 2)

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.serialize_path = os.path.join(self.output_dir,
                                               self.engine_name)
            serialize_engine(engine, self.serialize_path)
            if self.runtime_rank == 0:
                config_path = os.path.join(self.output_dir, 'config.json')
                builder_config.plugin_config = network.plugin_config
                builder.save_config(builder_config, config_path)
        return engine

    def run(self, inputs, config):
        ok = self.session.run(*inputs)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()

    def report(self, config, latency, percentile95, percentile99,
               peak_gpu_used):
        if self.runtime_rank == 0:
            line = '[BENCHMARK] ' + (
                f'model_name {self.model_name} world_size {self.world_size} precision {self.dtype} '
                f'batch_size {config[0]} input_length {config[1]} gpu_peak_mem(gb) {peak_gpu_used} '
                f'build_time(s) {self.build_time} percentile95(ms) {percentile95} '
                f'percentile99(ms) {percentile99} latency(ms) {latency}')
            print(line)

    def report(self, config, latency, percentile95, percentile99, peak_gpu_used,
               csv):
        report_dict = super().get_report_dict()
        batch_size, inlen = config[0], config[1]
        report_dict["num_heads"] = self.num_heads
        report_dict["num_kv_heads"] = self.num_heads
        report_dict["num_layers"] = self.num_layers
        report_dict["hidden_size"] = self.hidden_size
        report_dict["vocab_size"] = self.vocab_size
        report_dict["batch_size"] = batch_size
        report_dict["input_length"] = inlen
        report_dict["output_length"] = "n/a"
        report_dict["latency(ms)"] = latency
        report_dict["build_time(s)"] = self.build_time
        report_dict["tokens_per_sec"] = "n/a"
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
