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
from math import ceil

import torch

import tensorrt_llm

from allowed_configs import get_build_config  # isort:skip
from base_benchmark import BaseBenchmark  # isort:skip
from build import build_gpt, get_quant_mode  # isort:skip


class GPTBenchmark(BaseBenchmark):

    def __init__(self, args, batch_sizes, in_out_lens, rank, world_size):
        super().__init__(args.engine_dir, args.model, args.dtype, rank,
                         world_size, args.serial_build)
        self.batch_sizes = batch_sizes
        self.in_out_lens = in_out_lens
        self.num_beams = args.num_beams
        self.mode = args.mode
        self.build_time = 0

        self.cuda_graph_mode = args.enable_cuda_graph

        if args.engine_dir is not None:
            # Get build configs from engine directory is done in base class
            # Deserialize engine from engine directory
            self.serialize_path = os.path.join(args.engine_dir,
                                               self.engine_name)
            with open(self.serialize_path, 'rb') as f:
                engine_buffer = f.read()
        else:
            for key, value in get_build_config(args.model).items():
                setattr(self, key, value)
            if args.force_num_layer_1:
                self.num_layers = 1
            if args.max_batch_size is not None:
                self.max_batch_size = args.max_batch_size
            if args.max_input_len is not None:
                self.max_input_len = args.max_input_len
            if args.max_output_len is not None:
                self.max_output_len = args.max_output_len

            self.quant_mode, _, _ = get_quant_mode(args.quantization)
            self.enable_fp8 = self.quant_mode.has_fp8_qdq()
            self.fp8_kv_cache = self.quant_mode.has_fp8_kv_cache()

            # Plugins
            self.use_gpt_attention_plugin = False
            self.remove_input_padding = False
            if args.mode == 'plugin':
                self.use_gpt_attention_plugin = True
                self.remove_input_padding = True
            elif args.mode == 'ootb-except-mha':
                self.use_gpt_attention_plugin = True

            engine_buffer, build_time = build_gpt(args)
            self.build_time = build_time

        assert engine_buffer is not None
        if args.build_only:
            return

        if not hasattr(self, 'num_kv_heads') or self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        model_config = tensorrt_llm.runtime.ModelConfig(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads // self.world_size,
            num_kv_heads=ceil(self.num_kv_heads / self.world_size),
            hidden_size=self.hidden_size // self.world_size,
            gpt_attention_plugin=self.use_gpt_attention_plugin,
            remove_input_padding=self.remove_input_padding,
            quant_mode=self.quant_mode,
            use_custom_all_reduce=self.use_custom_all_reduce,
        )
        if args.model == 'chatglm_6b':
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=130005,
                pad_id=3,
                num_beams=self.num_beams,
                top_k=args.top_k,
                top_p=args.top_p)
            self.decoder = tensorrt_llm.runtime.ChatGLMGenerationSession(
                model_config, engine_buffer, self.runtime_mapping)
        elif args.model in ['chatglm2_6b', 'chatglm3_6b']:
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=2,
                pad_id=0,
                num_beams=self.num_beams,
                top_k=args.top_k,
                top_p=args.top_p)
            self.decoder = tensorrt_llm.runtime.GenerationSession(
                model_config, engine_buffer, self.runtime_mapping)
        else:
            end_id = 50256
            pad_id = 50256
            if "llama" in args.model:
                end_id = 2
                pad_id = 0
            self.sampling_config = tensorrt_llm.runtime.SamplingConfig(
                end_id=end_id,
                pad_id=pad_id,
                num_beams=self.num_beams,
                top_k=args.top_k,
                top_p=args.top_p)
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
