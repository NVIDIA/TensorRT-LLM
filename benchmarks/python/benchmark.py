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
import argparse
import multiprocessing as mp
from time import time

import torch


def parse_arguments():
    from allowed_configs import get_allowed_models
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
        choices=['ootb', 'plugin', 'ootb-except-mha'],
        help=
        ('Choose mode between ootb/plugin. '
         '\"ootb\" means the engines will be built without any plugins, '
         '\"plugin\" means the engines will be built with tuned recipe of using plugins.'
         '\"ootb-except-mha\" means the engines will be built with only attention plugins.'
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
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
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
        'If this option is specified, TensorRT engines will be saved to the specified path.'
    )
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=None,
        help=
        ('If this option is specified, instead of building engines on-air before benchmarking, '
         'the engines contained in the engine_dir will be used.'))
    parser.add_argument(
        '--max_beam_width',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max beam width of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument(
        '--max_input_len',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max input len of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument(
        '--max_encoder_input_len',
        type=int,
        default=None,
        help=
        ('This argument is only for encoder-decoder models'
         'If this option is specified, it will override the max encoder input len of TRT engines to the specified value instead of using pre-defined one'
         'By default when this option is not used, it will use pre-defined max encoder input len'
         ))
    parser.add_argument(
        '--max_decoder_input_len',
        type=int,
        default=None,
        help=
        ('This argument is only for encoder-decoder models'
         'If this option is specified, it will override the max decoder input len of TRT engines to the specified value instead of using pre-defined one'
         'By default when this option is not used, it will use pre-defined max decoder input len'
         ))
    parser.add_argument(
        '--max_output_len',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max output len of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max batch size of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument(
        '--force_num_layer_1',
        default=False,
        action='store_true',
        help=
        'Quick sanity check with num_layer=1; will be silently ignored if --engine_dir is specified.'
    )
    parser.add_argument('--strongly_typed',
                        default=False,
                        action='store_true',
                        help='This option will reduce the building time.')

    parser.add_argument('--csv',
                        default=False,
                        action="store_true",
                        help='Output in CSV format.')
    parser.add_argument('--enable_cuda_graph',
                        default=False,
                        action='store_true',
                        help='Execute GPT session with CUDA graph.')
    parser.add_argument(
        '--quantization',
        type=str,
        default=None,
        choices=[
            'fp8', 'fp8_gemm', 'fp8_kv_cache', 'int8_sq_per_tensor',
            'int8_sq_per_token_channel', 'int8_weight_only', 'int4_weight_only',
            'int4_weight_only_awq', 'int4_weight_only_gptq'
        ],
        help="Optimize the model with specified quantization recipe")
    parser.add_argument(
        '--build_only',
        default=False,
        action='store_true',
        help=
        "Build engine only and skip inference, this can help to benchmark the build time on single gpu node for multi GPU model, where the inference is not possible"
    )

    parser.add_argument('--serial_build',
                        default=False,
                        action='store_true',
                        help="Build engines serially")

    parser.add_argument(
        '--rank',
        type=int,
        default=None,
        help=
        "The rank of the model to be built, only used when --build_only and --serial_build is specified"
    )
    parser.add_argument(
        '--world_size',
        type=int,
        default=None,
        help=
        "The number of gpus to be used for inference, only used when --build_only and --serial_build is specified"
    )

    return parser.parse_args()


def main(args):
    # We import tensorrt_llm here because MPI is initialized when
    # tensorrt_llm is imported, but mpi4py does not work well with
    # the start method `spawn` of Python multiprocessing,
    # so we set the start method first, then initialize MPI.
    from allowed_configs import get_allowed_models
    from benchmark_profiler import BenchmarkProfiler
    from bert_benchmark import BERTBenchmark
    from enc_dec_benchmark import EncDecBenchmark
    from gpt_benchmark import GPTBenchmark

    import tensorrt_llm
    from tensorrt_llm.logger import logger

    logger.set_level(args.log_level)

    # Batch size
    batch_size_options = args.batch_size.split(';')
    batch_size_options = [int(i) for i in batch_size_options]
    # Input length (for BERT-like models)
    input_len_options = args.input_len.split(';')
    input_len_options = [int(i) for i in input_len_options]
    # Input-output length combination (for GPT-like models and enc_dec models)
    in_out_len_options = args.input_output_len.split(';')
    in_out_len_options = [[int(i) for i in io.split(',')]
                          for io in in_out_len_options]

    if args.serial_build and not args.build_only:
        raise Exception(
            f"--serial_build must be used with --build_only, always need to parallel build to do inference in the same process"
        )

    if args.build_only and args.serial_build and args.rank is not None and args.world_size is not None:
        rank = args.rank
        world_size = args.world_size
    else:
        rank = tensorrt_llm.mpi_rank()
        world_size = tensorrt_llm.mpi_world_size()

    # TODO: Re-enable memory monitor for multi-gpu benchmarks.
    # Current Mem Monitor will cause benchmark script hang
    # because MPI does not work well with multiprocessing.
    disable_mem_monitor = world_size > 1
    if not disable_mem_monitor:
        from mem_monitor import MemoryMonitor

    benchmark_profiler = None
    if args.model in get_allowed_models(benchmark_type="gpt"):
        benchmark_profiler = BenchmarkProfiler()
        benchmarker = GPTBenchmark(args, batch_size_options, in_out_len_options,
                                   rank, world_size)
    elif args.model in get_allowed_models(benchmark_type="bert"):
        benchmarker = BERTBenchmark(args, batch_size_options, input_len_options,
                                    rank, world_size)
    elif args.model in get_allowed_models(benchmark_type="enc_dec"):
        benchmarker = EncDecBenchmark(args, batch_size_options,
                                      in_out_len_options, rank, world_size)
    else:
        raise Exception(f'Unexpected model: {args.model}')

    if args.build_only:
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    benchmarker.print_report_header(args.csv,
                                    benchmark_profiler=benchmark_profiler)
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

        if not disable_mem_monitor:
            memory_monitor = MemoryMonitor()
            memory_monitor.start()

        iter_idx = 0
        try:
            # Warm up
            for _ in range(args.warm_up):
                benchmarker.run(inputs, config)
            logger.info('Warm up done. Start benchmarking.')
            if benchmark_profiler is not None:
                benchmark_profiler.clean()
                benchmark_profiler.start()
            cur_duration = 0
            start_time = time()
            while iter_idx < args.num_runs or cur_duration < args.duration:
                start.record()
                benchmarker.run(inputs,
                                config,
                                benchmark_profiler=benchmark_profiler)
                end.record()

                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))

                iter_idx += 1
                cur_duration = round(time() - start_time, 3)
            logger.info(
                f'Benchmarking done. Iteration: {iter_idx}, duration: {cur_duration} sec.'
            )

        except Exception as e:
            print("Found exception during benchmarking", e.with_traceback())
            if not disable_mem_monitor:
                memory_monitor.kill()
            raise e

        if not disable_mem_monitor:
            memory_monitor.stop()
            _, peak_gpu_used = memory_monitor.get_peak_memory_usage("GiB")
            peak_gpu_used = round(peak_gpu_used, 3)
        else:
            peak_gpu_used = 0.0

        if benchmark_profiler is not None:
            benchmark_profiler.add_aux_info('iter_count', iter_idx)
            benchmark_profiler.stop()

        latency = round(sum(latencies) / iter_idx, 3)
        latencies.sort()
        percentile95 = round(latencies[int(iter_idx * 0.95)], 3)
        percentile99 = round(latencies[int(iter_idx * 0.99)], 3)
        benchmarker.report(config,
                           latency,
                           percentile95,
                           percentile99,
                           peak_gpu_used,
                           csv=args.csv,
                           benchmark_profiler=benchmark_profiler)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parse_arguments()
    main(args)
