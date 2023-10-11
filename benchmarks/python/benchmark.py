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
import argparse
from multiprocessing import Process, Queue
from time import time

import torch
from allowed_configs import get_allowed_models
from bert_benchmark import BERTBenchmark
from gpt_benchmark import GPTBenchmark
from mem_monitor import mem_monitor

from tensorrt_llm.logger import logger


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Benchmark TensorRT-LLM models.')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default="gpt_350m",
                        choices=get_allowed_models(),
                        help='Specify model you want to benchmark.')
    parser.add_argument(
        '--mode',
        type=str,
        default="plugin",
        choices=['ootb', 'plugin'],
        help=
        ('Choose mode between ootb/plugin. '
         '\"ootb\" means the engines will be built without any plugins, '
         'while \"plugin\" means the engines will be built with tuned recipe of using plugins.'
         ))

    parser.add_argument('--batch_size',
                        type=str,
                        default="8",
                        help=('Specify batch size(s) you want to benchmark. '
                              'Multiple batch sizes can be separated by \";\", '
                              'example: \"1;8;64\".'))
    parser.add_argument(
        '--input_len',
        type=str,
        default="128",
        help=('Specify input length(s) you want to benchmark, '
              'this option is mainly for BERT. '
              'Multiple input lengths can be separated by \";\", '
              'example: \"20;60;128\".'))
    parser.add_argument(
        '--input_output_len',
        type=str,
        default="128,20",
        help=('Specify input-output length(s) you want to benchmark, '
              'this option is mainly for GPT and GPT-like models. '
              'Multiple input lengths can be separated by \";\", '
              'example: \"60,20;128,20\".'))
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Choose data type between float16/bfloat16/float32.')
    parser.add_argument(
        '--refit',
        default=False,
        action="store_true",
        help=
        'If this option is specified, a refit flag is added to TensorRT engines.'
    )

    parser.add_argument('--num_beams',
                        type=int,
                        default="1",
                        help=('Specify number of beams you want to benchmark.'))
    parser.add_argument('--top_k',
                        type=int,
                        default="1",
                        help=('Specify Top-K value of decoding.'))
    parser.add_argument('--top_p',
                        type=float,
                        default="0",
                        help=('Specify Top-P value of decoding.'))

    parser.add_argument(
        '--log_level',
        type=str,
        default="error",
        choices=['verbose', 'info', 'warning', 'error', 'internal_error'],
        help=
        'Choose log level between verbose/info/warning/error/internal_error.')
    parser.add_argument(
        '--warm_up',
        type=int,
        default=2,
        help='Specify warm up iterations before benchmark starts.')
    parser.add_argument(
        '--num_runs',
        type=int,
        default=10,
        help='Minimal number of iterations to run during benchmarking.')
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Minimal duration of iterations to measure in seconds.')

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help=
        'If this option is specified, TensorRT engines will be saved to engine_dir.'
    )
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=None,
        help=
        ('If this option is specified, instead of building engines on-air before benchmarking, '
         'the engines contained in the engine_dir will be used.'))
    parser.add_argument(
        '--n_positions',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the n_positions of TRT engines to the specified value instead of using pre-defined one'
         'By default when this option is not used, it will use pre-defined n_positions'
         ))
    parser.add_argument(
        '--max_input_len',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max input len of TRT engines to the specified value instead of using pre-defined one'
         'By default when this option is not used, it will use pre-defined max input len'
         ))
    parser.add_argument(
        '--max_output_len',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max output len of TRT engines to the specified value instead of using pre-defined one'
         'By default when this option is not used, it will use pre-defined max output len'
         ))
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max batch size of TRT engines to the specified value instead of using pre-defined one'
         'By default when this option is not used, it will use pre-defined max batch size'
         ))
    parser.add_argument(
        '--force_num_layer_1',
        default=False,
        action='store_true',
        help=
        'Quick sanity check with num_layer=1; will be silently ignored if --engine_dir is specified.'
    )
    parser.add_argument(
        '--enable_fp8',
        default=False,
        action='store_true',
        help='Use FP8 Linear layer for LMHead, Attention QKV/Dense, and MLP.')
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV'
    )
    parser.add_argument('--csv',
                        default=False,
                        action="store_true",
                        help='Output in CSV format.')
    parser.add_argument('--enable_cuda_graph',
                        default=False,
                        action='store_true',
                        help='Execute GPT session with CUDA graph.')
    parser.add_argument(
        '--enable_custom_all_reduce',
        default=False,
        action='store_true',
        help=
        'Use latency-optimized all-reduce for tensor parallelism. Gives better performance with NVLink.'
    )

    return parser.parse_args()


def main(args):
    logger.set_level(args.log_level)

    # Batch size
    batch_size_options = args.batch_size.split(';')
    batch_size_options = [int(i) for i in batch_size_options]
    # Input length (for BERT-like models)
    input_len_options = args.input_len.split(';')
    input_len_options = [int(i) for i in input_len_options]
    # Input-output length combination (for GPT-like models)
    in_out_len_options = args.input_output_len.split(';')
    in_out_len_options = [[int(i) for i in io.split(',')]
                          for io in in_out_len_options]

    if args.model in get_allowed_models(benchmark_type="gpt"):
        benchmarker = GPTBenchmark(
            args.engine_dir,
            args.model,
            args.mode,
            batch_size_options,
            in_out_len_options,
            args.dtype,
            args.refit,
            args.num_beams,
            args.top_k,
            args.top_p,
            args.output_dir,
            args.n_positions,
            args.max_input_len,
            args.max_output_len,
            args.max_batch_size,
            force_num_layer_1=args.force_num_layer_1,
            enable_fp8=args.enable_fp8,
            fp8_kv_cache=args.fp8_kv_cache,
            enable_cuda_graph=args.enable_cuda_graph,
            enable_custom_all_reduce=args.enable_custom_all_reduce)
    elif args.model in get_allowed_models(benchmark_type="bert"):
        benchmarker = BERTBenchmark(args.engine_dir,
                                    args.model,
                                    args.mode,
                                    batch_size_options,
                                    input_len_options,
                                    args.dtype,
                                    args.output_dir,
                                    args.n_positions,
                                    args.max_input_len,
                                    args.max_output_len,
                                    args.max_batch_size,
                                    force_num_layer_1=args.force_num_layer_1)
    else:
        raise Exception(f'Unexpected model: {args.model}')

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    benchmarker.print_report_header(args.csv)
    for config in benchmarker.get_config():
        try:
            inputs = benchmarker.prepare_inputs(config)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                f'Exception {e} caught while allocating memory; skipping {config}'
            )
            continue

        torch.cuda.empty_cache()
        latencies = []

        # Launch a subprocess to monitor memory usage
        q1 = Queue()  # q1 is used for sending signal to subprocess
        q2 = Queue()  # q2 is used for receiving results from subprocess
        p = Process(target=mem_monitor, args=(q1, q2))
        p.start()

        iter_idx = 0
        try:
            # Warm up
            for _ in range(args.warm_up):
                benchmarker.run(inputs, config)
            logger.info('Warm up done. Start benchmarking.')

            cur_duration = 0
            start_time = time()
            while iter_idx < args.num_runs or cur_duration < args.duration:
                start.record()
                benchmarker.run(inputs, config)
                end.record()

                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))

                iter_idx += 1
                cur_duration = round(time() - start_time, 3)
            logger.info(
                f'Benchmarking done. Iteration: {iter_idx}, duration: {cur_duration} sec.'
            )

        except Exception as e:
            p.kill()
            raise e

        q1.put(1)
        peak_gpu_used = q2.get()
        p.join()

        latency = round(sum(latencies) / iter_idx, 3)
        latencies.sort()
        percentile95 = round(latencies[int(iter_idx * 0.95)], 3)
        percentile99 = round(latencies[int(iter_idx * 0.99)], 3)
        benchmarker.report(config,
                           latency,
                           percentile95,
                           percentile99,
                           peak_gpu_used,
                           csv=args.csv)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
