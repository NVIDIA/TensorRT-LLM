#!/usr/bin/env python3

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

import yaml


class TRTLLMBenchmark:

    def __init__(self, args):
        # Model configuration
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.tp_size = args.tp_size
        self.ep_size = args.ep_size

        # Output and data configuration
        self.output_dir = Path(args.output_dir)
        self.data_gen_path = Path(args.data_gen_path)
        self.input_length = args.input_length
        self.output_length = args.output_length

        # Benchmark configuration
        print(args)
        assert args.max_batch_size is not None
        assert args.max_num_tokens is not None
        assert args.concurrency is not None
        assert args.num_requests is not None

        self.num_requests = args.num_requests
        self.max_batch_size = args.max_batch_size
        self.max_num_tokens = args.max_num_tokens
        self.concurrency = args.concurrency

        # Ray executor configuration
        self.use_ray_executor = args.use_ray_executor

        # Attention DP configuration
        self.enable_attention_dp = args.enable_attention_dp

        self.setup_logging()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, exiting...")
        sys.exit(0)

    def create_directories(self):
        self.logger.info("Creating output directories...")
        for directory in ['configs', 'logs', 'reports']:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)

    def create_cuda_graph_configs(self):
        self.logger.info("Creating CUDA graph configuration files...")

        config = {
            'print_iter_log': False,
            'cuda_graph_config': {
                'enable_padding': True,
                'batch_sizes': [128],
            },
            'kv_cache_config': {
                'dtype': 'auto',
                'free_gpu_memory_fraction': 0.2
            },
            'enable_attention_dp': self.enable_attention_dp
        }

        if self.use_ray_executor:
            config['executor_type'] = 'ray'
            self.logger.info("Ray executor configuration enabled")

        config_path = self.output_dir / 'configs' / 'bench_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Created config: {config_path}")

    def generate_dataset(self) -> Path:
        dataset_path = self.output_dir / f"dataset_{self.input_length}_{self.output_length}_{self.num_requests}.txt"
        if not dataset_path.exists() or os.path.getsize(dataset_path) == 0:
            self.logger.info(
                f"Generating dataset: ISL={self.input_length}, OSL={self.output_length}, requests={self.num_requests}"
            )
            tokenizer_path = self.model_path if self.model_path is not None else self.model_name
            assert tokenizer_path is not None

            cmd = [
                'python',
                str(self.data_gen_path / 'prepare_dataset.py'), '--stdout',
                '--tokenizer',
                str(tokenizer_path), 'token-norm-dist', '--num-requests',
                str(self.num_requests), '--input-mean',
                str(self.input_length), '--input-stdev', '0', '--output-mean',
                str(self.output_length), '--output-stdev', '0'
            ]

            with open(dataset_path, 'w') as f:
                result = subprocess.run(cmd,
                                        stdout=f,
                                        stderr=subprocess.PIPE,
                                        text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Dataset generation failed: {result.stderr}")

            self.logger.info(f"Dataset generated: {dataset_path}")
        else:
            self.logger.info(f"Using existing dataset: {dataset_path}")

        return dataset_path

    def run_benchmark(self, config_name: str, config_path: Path,
                      dataset_path: Path) -> bool:
        benchmark_log = self.output_dir / 'logs' / f'benchmark_{config_name}.log'
        # report_json = self.output_dir / 'reports' / f'report_{config_name}.json'
        # iteration_log = self.output_dir / 'logs' / f'iteration_{config_name}.log'

        self.logger.info(f"Running benchmark: {config_name}")

        cmd = [
            'trtllm-bench',
            f'--model={self.model_name}'
            if self.model_name is not None else '--model=dummy_model',
            f'--model_path={self.model_path}'
            if self.model_path is not None else '',
            'throughput',
            f'--dataset={dataset_path}',
            '--backend=pytorch',
            f'--max_batch_size={self.max_batch_size}',
            f'--max_num_tokens={self.max_num_tokens}',
            f'--concurrency={self.concurrency}',
            f'--num_requests={self.num_requests}',
            f'--extra_llm_api_options={config_path}',
            #f'--report_json={report_json}',
            #f'--iteration_log={iteration_log}'
        ]

        cmd = [arg for arg in cmd if (arg != '' or arg is not None)]

        if self.tp_size > 1:
            cmd.append(f'--tp={self.tp_size}')
        if self.ep_size > 1:
            cmd.append(f'--ep={self.ep_size}')

        self.logger.info(f"Executing: {' '.join(cmd)}")

        try:
            with open(benchmark_log, 'w') as f:
                result = subprocess.run(cmd,
                                        stdout=f,
                                        stderr=subprocess.STDOUT,
                                        text=True)

            success = result.returncode == 0
            if success:
                self.logger.info(
                    f"Benchmark completed successfully for {config_name}")
            else:
                self.logger.error(
                    f"Benchmark failed for {config_name}. Check {benchmark_log}"
                )

        except Exception as e:
            self.logger.error(f"Exception during benchmark: {e}")
            success = False
        finally:
            time.sleep(2)

        return success

    def run(self):
        self.logger.info("Starting trtllm benchmark")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(
            f"Executor type: {'Ray' if self.use_ray_executor else 'MPI (default)'}"
        )

        try:
            self.create_directories()
            self.create_cuda_graph_configs()

            dataset_path = self.generate_dataset()
            config_path = self.output_dir / "configs" / "bench_config.yaml"

            if self.run_benchmark("bench_config", config_path, dataset_path):
                self.logger.info("✓ Benchmark completed successfully")
            else:
                self.logger.error("✗ Benchmark failed")

            self.logger.info("trtllm benchmark completed!")
            self.logger.info(f"Results available in: {self.output_dir}")

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise


def get_gpu_defaults(gpu_type: str, gpu_count: int) -> Dict[str, any]:
    """Get default values based on GPU type and count."""
    gpu_type = gpu_type.upper()

    defaults = {
        "GENERIC": {
            1: {
                'tp_size': 1,
                'num_requests': 64,
                'max_batch_size': 64,
                'max_num_tokens': 1024,
                'concurrency': 64
            },
            8: {
                'tp_size': 8,
                'num_requests': 2048,
                'max_batch_size': 64,
                'max_num_tokens': 1024,
                'concurrency': 512
            }
        },
        "H200": {
            1: {
                'tp_size': 1,
                'num_requests': 64,
                'max_batch_size': 64,
                'max_num_tokens': 640,
                'concurrency': 64
            },
            8: {
                'tp_size': 8,
                'num_requests': 128 * 8,
                'max_batch_size': 128,
                'max_num_tokens': 1151,
                'concurrency': 1024
            }
        },
        "B200": {
            8: {
                'tp_size': 8,
                'num_requests': 49152,
                'max_batch_size': 384,
                'max_num_tokens': 1536,
                'concurrency': 3072
            }
        },
        "A100": {
            1: {
                'tp_size': 1,
                'num_requests': 64,
                'max_batch_size': 64,
                'max_num_tokens': 1024,
                'concurrency': 64
            },
            8: {
                'tp_size': 8,
                'num_requests': 2048,
                'max_batch_size': 64,
                'max_num_tokens': 1024,
                'concurrency': 512
            }
        }
    }

    if gpu_type not in defaults:
        raise ValueError(
            f"Unsupported GPU type: {gpu_type}. Supported: generic, H200, B200, A100"
        )

    if gpu_count not in defaults[gpu_type]:
        print(f"Using default GPU count: {gpu_count}")
        gpu_count = 8

    # Ensure ep_size always matches tp_size in the returned defaults
    gpu_defaults = dict(defaults[gpu_type][gpu_count])
    gpu_defaults['ep_size'] = gpu_defaults.get('tp_size', 1)
    return gpu_defaults


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='TensorRT-LLM Benchmark Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GPU configuration
    parser.add_argument('--gpu-type',
                        choices=['generic', 'H200', 'B200', 'A100'],
                        default='generic',
                        help='Type of GPU (generic, H200, B200, A100)')
    parser.add_argument('--gpu-count',
                        type=int,
                        default=1,
                        help='Number of GPUs')

    # Model configuration
    parser.add_argument(
        '--model-path',
        help=
        'Path to the model directory (auto-configured based on GPU type/count)')
    parser.add_argument(
        '--model-name',
        help='Name of the model (auto-configured based on GPU type/count)')
    parser.add_argument(
        '--tp-size',
        default=1,
        type=int,
        help='Tensor parallelism size (auto-configured based on GPU count)')
    parser.add_argument(
        '--ep-size',
        default=1,
        type=int,
        help='Expert parallelism size (auto-configured; defaults to 1)')

    # Output and data configuration
    parser.add_argument(
        '--output-dir',
        help=
        'Output directory for logs and reports (auto-configured based on GPU type/count)'
    )
    parser.add_argument('--data-gen-path',
                        default='bencmark_output',
                        help='Path to data generation scripts')
    parser.add_argument('--input-length',
                        type=int,
                        default=512,
                        help='Input sequence length')
    parser.add_argument('--output-length',
                        type=int,
                        default=512,
                        help='Output sequence length')

    # Benchmark configuration
    parser.add_argument(
        '--num-requests',
        type=int,
        help=
        'Number of requests for benchmarking (auto-configured based on GPU type/count)'
    )
    parser.add_argument(
        '--max-batch-size',
        type=int,
        help='Maximum batch size (auto-configured based on GPU type/count)')
    parser.add_argument(
        '--max-num-tokens',
        type=int,
        help='Maximum number of tokens (auto-configured based on GPU type/count)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        help='Concurrency level (auto-configured based on GPU type/count)')

    # Ray executor configuration
    parser.add_argument('--use-ray-executor',
                        action='store_true',
                        help='Use Ray executor instead of MPI (default)')

    # Attention DP configuration
    parser.add_argument('--enable-attention-dp',
                        action='store_true',
                        help='Enable attention DP (default: False)')

    args = vars(parser.parse_args())

    # Get GPU-specific defaults
    try:
        gpu_defaults = get_gpu_defaults(args['gpu_type'], args['gpu_count'])

        # Set defaults for arguments that weren't provided
        for attr, default_value in gpu_defaults.items():
            if args[attr] is None:
                print("Overriding", attr, default_value)
                args[attr] = default_value

        if args['output_dir'] is None:
            mpi_or_ray = 'mpi' if not args['use_ray_executor'] else 'ray'
            args[
                'output_dir'] = f'trtllm_bench_{mpi_or_ray}_{args["gpu_count"]}gpu_{args["gpu_type"].lower()}_logs'

    except ValueError as e:
        parser.error(str(e))

    return argparse.Namespace(**args)


def main():
    args = parse_arguments()
    benchmark = TRTLLMBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    main()
