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
import json
from math import ceil

import pandas as pd
import tensorrt as trt
import torch

import tensorrt_llm
from tensorrt_llm.bindings import KVCacheType
from tensorrt_llm.builder import Engine
from tensorrt_llm.runtime import (ChatGLMGenerationSession, GenerationSession,
                                  SamplingConfig)

from base_benchmark import BaseBenchmark  # isort:skip


def element_size(dtype: str):
    str_to_size_in_bytes = dict(float16=2,
                                float32=4,
                                int64=8,
                                int32=4,
                                int8=1,
                                bool=1,
                                bfloat16=2,
                                fp8=1)
    return str_to_size_in_bytes[dtype]


class GPTBenchmark(BaseBenchmark):

    def __init__(self, args, batch_sizes, in_out_lens, gpu_weights_percents,
                 rank, world_size):
        super().__init__(args.engine_dir, args.model, args.dtype, rank,
                         world_size)
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.gpu_weights_percents = gpu_weights_percents
        self.num_beams = args.num_beams
        self.cuda_graph_mode = args.enable_cuda_graph
        self.dump_layer_info = args.dump_layer_info

        # Get build configs from engine directory is done in base class
        # Deserialize engine from engine directory
        engine = Engine.from_dir(args.engine_dir, rank)
        engine_buffer = engine.engine
        assert engine_buffer is not None
        pretrained_config = engine.config.pretrained_config
        if pretrained_config.architecture == 'ChatGLMForCausalLM' and pretrained_config.chatglm_version in [
                'glm', 'chatglm'
        ]:
            session_cls = ChatGLMGenerationSession
        else:
            session_cls = GenerationSession

        if not hasattr(self, 'num_kv_heads') or self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads

        rnn_config_items = [
            'conv_kernel', 'layer_types', 'rnn_hidden_size', 'state_size',
            'state_dtype', 'rnn_head_size', 'rnn_conv_dim_size'
        ]
        rnn_configs_kwargs = {}
        for item in rnn_config_items:
            if hasattr(self, item):
                rnn_configs_kwargs[item] = getattr(self, item)

        kv_cache_type = KVCacheType.CONTINUOUS
        if hasattr(self, 'kv_cache_type'):
            kv_cache_type = self.kv_cache_type
        else:
            if hasattr(self, 'paged_kv_cache'):
                kv_cache_type = KVCacheType.PAGED if self.paged_kv_cache == True else KVCacheType.CONTINUOUS

        model_config = tensorrt_llm.runtime.ModelConfig(
            max_batch_size=self.max_batch_size,
            max_beam_width=self.num_beams,
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads // self.world_size,
            num_kv_heads=ceil(self.num_kv_heads / self.world_size),
            hidden_size=self.hidden_size // self.world_size,
            gpt_attention_plugin=self.use_gpt_attention_plugin,
            kv_cache_type=kv_cache_type,
            paged_state=self.paged_state
            if hasattr(self, 'paged_state') else False,
            dtype=self.dtype,
            remove_input_padding=self.remove_input_padding,
            quant_mode=self.quant_mode,
            tokens_per_block=self.tokens_per_block if hasattr(
                self, 'tokens_per_block') else 64,
            mamba_conv1d_plugin=self.use_mamba_conv1d_plugin,
            gpu_weights_percent=list(sorted(gpu_weights_percents))[0],
            **rnn_configs_kwargs,
        )
        self.sampling_config = SamplingConfig(end_id=2, pad_id=0)
        self.decoder = session_cls(model_config,
                                   engine_buffer,
                                   self.runtime_mapping,
                                   cuda_graph_mode=self.cuda_graph_mode)

        # Print context memory size for CI/CD to track.
        context_mem_size = self.decoder.context_mem_size
        print(
            f"Allocated {context_mem_size / 1048576.0:.2f} MiB for execution context memory."
        )

    def get_config(self):
        for inlen, outlen in self.in_out_lens:
            if inlen > self.max_input_len or inlen + outlen > self.max_seq_len:
                print(
                    f'[WARNING] check inlen({inlen}) <= max_inlen({self.max_input_len}) or '
                    f'seqlen({inlen + outlen}) <= max_seq_len({self.max_seq_len}) failed, skipping.'
                )
                continue
            for batch_size in self.batch_sizes:
                if batch_size > self.max_batch_size:
                    print(
                        f'[WARNING] check batch_size({batch_size}) '
                        f'<= max_batch_size({self.max_batch_size}) failed, skipping.'
                    )
                    continue
                for gpu_weights_percent in self.gpu_weights_percents:
                    yield (batch_size, inlen, outlen, gpu_weights_percent)

    def set_weight_streaming(self, config):
        gpu_weights_percent = config[3]
        self.decoder.runtime._set_weight_streaming(gpu_weights_percent)

    def prepare_inputs(self, config):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        input_ids = torch.randint(100, (batch_size, inlen)).int().cuda()
        input_lengths = torch.tensor([inlen
                                      for _ in range(batch_size)]).int().cuda()

        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        return (input_ids, input_lengths)

    def get_report_dict(self, benchmark_profiler=None):
        report_dict = super().get_report_dict(
            benchmark_profiler=benchmark_profiler)
        if benchmark_profiler is not None:
            report_dict["generation_time(ms)"] = None
            report_dict["total_generated_tokens"] = None
            report_dict["generation_tokens_per_second"] = None
        return report_dict

    def run(self, inputs, config, benchmark_profiler=None):
        batch_size, inlen, outlen = config[0], config[1], config[2]
        self.decoder.setup(batch_size, inlen, outlen, beam_width=self.num_beams)
        if self.remove_input_padding:
            self.decoder.decode_batch(inputs[0],
                                      self.sampling_config,
                                      benchmark_profiler=benchmark_profiler)
        else:
            self.decoder.decode(inputs[0],
                                inputs[1],
                                self.sampling_config,
                                benchmark_profiler=benchmark_profiler)
        torch.cuda.synchronize()

    def report(self,
               config,
               latency,
               percentile95,
               percentile99,
               peak_gpu_used,
               csv,
               benchmark_profiler=None):
        report_dict = super().get_report_dict()
        batch_size, inlen, outlen, gpu_weights_percent = config[0], config[
            1], config[2], config[3]
        tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
        report_dict["num_heads"] = self.num_heads
        report_dict["num_kv_heads"] = self.num_kv_heads
        report_dict["num_layers"] = self.num_layers
        report_dict["hidden_size"] = self.hidden_size
        report_dict["vocab_size"] = self.vocab_size
        report_dict["batch_size"] = batch_size
        report_dict["gpu_weights_percent"] = gpu_weights_percent
        report_dict["input_length"] = inlen
        report_dict["output_length"] = outlen
        report_dict["latency(ms)"] = latency
        report_dict["tokens_per_sec"] = tokens_per_sec
        report_dict["percentile95(ms)"] = percentile95
        report_dict["percentile99(ms)"] = percentile99
        report_dict["gpu_peak_mem(gb)"] = peak_gpu_used
        if benchmark_profiler is not None:
            iter_count = benchmark_profiler.get_aux_info('iter_count')
            generation_time_ms = benchmark_profiler.get_timer_value(
                'generation_time')
            generation_step_count = benchmark_profiler.get_aux_info(
                'generation_step_count')
            token_per_step = batch_size * self.num_beams
            total_tokens = generation_step_count * token_per_step
            report_dict["generation_time(ms)"] = round(
                generation_time_ms / iter_count, 3)
            report_dict["total_generated_tokens"] = total_tokens / iter_count
            tokens_per_second = round(
                total_tokens * 1000.0 / generation_time_ms, 3)
            report_dict["generation_tokens_per_second"] = tokens_per_second

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

        if self.dump_layer_info:
            engine_inspector = self.decoder.engine_inspector
            inspector_result = engine_inspector.get_engine_information(
                trt.LayerInformationFormat.JSON)
            json_result = json.loads(inspector_result)
            layers = json_result["Layers"]
            for layer_idx, _ in enumerate(layers):
                layer_info = engine_inspector.get_layer_information(
                    layer_idx, trt.LayerInformationFormat.ONELINE)
                print(layer_info)

    def report_profiler(self, benchmark_profiler=None):
        if benchmark_profiler is not None and benchmark_profiler.is_recording_perf_profile:
            perf_profile_data = self.decoder.profiler.results
            if not perf_profile_data:
                tensorrt_llm.logger.error("profiler data is empty")
                return

            ctx_layers = list()
            generation_layers = list()
            start = 0
            ctx_iter_cnt = 0
            generation_iter_cnt = 0

            # split context/generations layer information
            for idx, layer_info in enumerate(perf_profile_data):
                if layer_info[0] == "step":
                    if layer_info[1] == 0:
                        ctx_layers.extend(perf_profile_data[start:idx])
                        ctx_iter_cnt += 1
                    else:
                        generation_layers.extend(perf_profile_data[start:idx])
                        generation_iter_cnt += 1
                        start = idx + 1

            # Reduce all data
            def reduce_layer_data(layers):
                layer_infos = dict()
                for layer in layers:
                    if layer[0] in layer_infos:
                        layer_infos[layer[0]] += layer[1]
                    else:
                        layer_infos[layer[0]] = layer[1]
                return layer_infos

            # Dump kernel data
            def dump_kernel_profile_table(name: str, profile_data: list,
                                          iter_cnt: int):
                table = pd.DataFrame(
                    [['{:0.3f}'.format(v), k]
                     for k, v in profile_data.items() if v != 0.0],
                    columns=['times (ms)', '{} Phase LayerName'.format(name)])

                def ljust(s):
                    s = s.astype(str).str.strip()
                    return s.str.ljust(s.str.len().max())

                print(table.apply(ljust).to_string(index=False, justify='left'))
                print("{} phase step iter: {}".format(name, iter_cnt))

            ctx_layer_infos = reduce_layer_data(ctx_layers)
            generation_layer_infos = reduce_layer_data(generation_layers)
            dump_kernel_profile_table("Context", ctx_layer_infos, ctx_iter_cnt)
            dump_kernel_profile_table("Generation", generation_layer_infos,
                                      generation_iter_cnt)
