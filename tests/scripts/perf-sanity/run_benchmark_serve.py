#!/usr/bin/env python3
"""
Script to run benchmarks from YAML configuration file
Usage: python run_benchmark_serve.py --output_folder <output_folder> --config_file <config_file> [--skip <skip_pattern>] [--select <select_pattern>]
Skip pattern format: "2,4-1" means skip test case 2 and test case 4's 1st concurrency
Select pattern format: "1,3,5" means only run test cases 1, 3, and 5
Select pattern format: "1-1,2-3" means only run test case 1's 1st concurrency and test case 2's 3rd concurrency
If select_pattern is empty, all test cases are selected
If skip_pattern is empty, no test cases are skipped
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import requests
import yaml


class BenchmarkRunner:

    def __init__(self,
                 output_folder: str,
                 config_file: str,
                 skip_pattern: str = None,
                 select_pattern: str = None):
        self.output_folder = Path(output_folder)
        self.config_file = Path(config_file)

        # Treat empty or "default" values as None (default behavior)
        self.skip_pattern = None if not skip_pattern or skip_pattern.lower(
        ) == "default" else skip_pattern
        self.select_pattern = None if not select_pattern or select_pattern.lower(
        ) == "default" else select_pattern

        self.skip_test_cases: Set[int] = set()
        self.skip_concurrencies: Dict[int, Set[int]] = {}
        self.select_test_cases: Set[int] = set()
        self.select_concurrencies: Dict[int, Set[int]] = {}

        if self.skip_pattern:
            self.parse_skip_pattern(self.skip_pattern)

        if self.select_pattern:
            self.parse_select_pattern(self.select_pattern)

        # Execution plan: {test_case_id: [concurrency_indices]}
        self.execution_plan: Dict[int, List[int]] = {}

        # Model path mapping
        self.model_paths = {
            "70B-FP4":
            "/home/scratch.omniml_data_2/HF_model_hub/Llama-3.3-70B-Instruct-FP4",
            "70B-FP8":
            "/home/scratch.omniml_data_2/HF_model_hub/Llama-3.3-70B-Instruct-FP8",
            "Scout-FP4":
            "/home/scratch.trt_llm_data/llm-models/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP4",
            "Scout-FP8":
            "/home/scratch.trt_llm_data/llm-models/llama4-models/Llama-4-Scout-17B-16E-Instruct-FP8",
            "R1-FP8":
            "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1/",
            "R1-FP4":
            "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-0528-FP4"
        }

        # Set environment variables
        os.environ['TQDM_MININTERVAL'] = '1000'
        os.environ['PRINT_ITER_LOG'] = 'false'

        # Change to output directory
        os.chdir(self.output_folder)

    def parse_skip_pattern(self, skip_pattern: str) -> None:
        """Parse skip pattern like '2,4-1' to determine what to skip"""
        if not skip_pattern:
            return

        parts = skip_pattern.split(',')
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue

            if '-' in part:
                # Format: "test_case-concurrency_index" (1-based)
                try:
                    test_case_str, concurrency_str = part.split('-')
                    test_case_id = int(test_case_str)
                    concurrency_index = int(
                        concurrency_str) - 1  # Convert to 0-based

                    if test_case_id not in self.skip_concurrencies:
                        self.skip_concurrencies[test_case_id] = set()
                    self.skip_concurrencies[test_case_id].add(concurrency_index)
                except ValueError:
                    raise ValueError(
                        f"Invalid skip pattern '{part}'. Expected format: 'test_case-concurrency_index' (e.g., '2-1')"
                    )
            else:
                # Format: "test_case" - skip entire test case
                try:
                    test_case_id = int(part)
                    self.skip_test_cases.add(test_case_id)
                except ValueError:
                    raise ValueError(
                        f"Invalid test case ID '{part}' in skip pattern. Must be a valid integer."
                    )

        print(f"Skipping test cases: {sorted(self.skip_test_cases)}")
        print(f"Skipping concurrencies: {self.skip_concurrencies}")

    def parse_select_pattern(self, select_pattern: str) -> None:
        """Parse select pattern like '1,3,5' or '1-1,2-3' to determine which test cases/concurrencies to run"""
        if not select_pattern:
            return

        self.select_concurrencies: Dict[int, Set[int]] = {}

        parts = select_pattern.split(',')
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue

            if '-' in part:
                # Format: "test_case-concurrency_index" (1-based)
                try:
                    test_case_str, concurrency_str = part.split('-')
                    test_case_id = int(test_case_str)
                    concurrency_index = int(
                        concurrency_str) - 1  # Convert to 0-based

                    if test_case_id not in self.select_concurrencies:
                        self.select_concurrencies[test_case_id] = set()
                    self.select_concurrencies[test_case_id].add(
                        concurrency_index)
                except ValueError:
                    raise ValueError(
                        f"Invalid select pattern '{part}'. Expected format: 'test_case-concurrency_index' (e.g., '2-1')"
                    )
            else:
                # Format: "test_case" - select entire test case
                try:
                    test_case_id = int(part)
                    self.select_test_cases.add(test_case_id)
                except ValueError:
                    raise ValueError(
                        f"Invalid test case ID '{part}' in select pattern. Must be a valid integer."
                    )

        print(f"Selected test cases: {sorted(self.select_test_cases)}")
        print(f"Selected concurrencies: {self.select_concurrencies}")

    def build_execution_plan(self, test_cases: List[Dict[str, Any]]) -> None:
        """Build execution plan by analyzing config file, skip_pattern, and select_pattern"""
        self.execution_plan.clear()

        # Step 1: Initialize execution plan based on select_pattern
        if not self.select_pattern:
            # If select_pattern is empty or default, include all test cases with all concurrencies
            for test_case in test_cases:
                test_case_id = test_case['id']
                all_concurrencies = list(
                    range(len(test_case['concurrency_iterations'])))
                self.execution_plan[test_case_id] = all_concurrencies
        else:
            # If select_pattern is specified, only include selected test cases and concurrencies
            for test_case in test_cases:
                test_case_id = test_case['id']

                # Check if this test case is selected
                if test_case_id in self.select_test_cases:
                    # Test case is selected - include all concurrencies
                    all_concurrencies = list(
                        range(len(test_case['concurrency_iterations'])))
                    self.execution_plan[test_case_id] = all_concurrencies
                elif test_case_id in self.select_concurrencies:
                    # Specific concurrencies are selected for this test case
                    selected_concurrencies = list(
                        self.select_concurrencies[test_case_id])
                    # Validate that selected concurrencies exist in config
                    max_concurrency_index = len(
                        test_case['concurrency_iterations']) - 1
                    valid_concurrencies = [
                        c for c in selected_concurrencies
                        if 0 <= c <= max_concurrency_index
                    ]
                    if valid_concurrencies:
                        self.execution_plan[test_case_id] = valid_concurrencies

        # Step 2: Apply skip_pattern to remove test cases and concurrencies
        # Remove entire test cases that are in skip_test_cases
        for test_case_id in self.skip_test_cases:
            if test_case_id in self.execution_plan:
                del self.execution_plan[test_case_id]

        # Remove specific concurrencies that are in skip_concurrencies
        for test_case_id, skip_concurrency_indices in self.skip_concurrencies.items(
        ):
            if test_case_id in self.execution_plan:
                # Remove skipped concurrencies from the list
                remaining_concurrencies = [
                    c for c in self.execution_plan[test_case_id]
                    if c not in skip_concurrency_indices
                ]
                if remaining_concurrencies:
                    self.execution_plan[test_case_id] = remaining_concurrencies
                else:
                    # If no concurrencies remain, remove the entire test case
                    del self.execution_plan[test_case_id]

        # Step 3: Clean up - remove test cases with empty concurrency lists
        # (This should not happen with the above logic, but just to be safe)
        test_cases_to_remove = []
        for test_case_id, concurrencies in self.execution_plan.items():
            if not concurrencies:
                test_cases_to_remove.append(test_case_id)

        for test_case_id in test_cases_to_remove:
            del self.execution_plan[test_case_id]

    def print_execution_plan(self, test_cases: List[Dict[str, Any]]) -> None:
        """Print which test cases and concurrencies will be executed"""
        print("\n" + "=" * 80)
        print("EXECUTION PLAN")
        print("=" * 80)

        total_test_cases = 0
        total_concurrencies = 0

        for test_case in test_cases:
            test_case_id = test_case['id']
            model_label = test_case['model']

            # Check if this test case is in execution plan
            if test_case_id not in self.execution_plan:
                print(f"Test Case {test_case_id}: {model_label} - SKIPPED")
                continue

            total_test_cases += 1
            print(f"\nTest Case {test_case_id}: {model_label}")
            print(
                f"  Config: GPUs={test_case['gpus']}, TP={test_case['tp']}, EP={test_case['ep']}, attn_backend={test_case['attn_backend']}, moe_backend={test_case['moe_backend']}"
            )

            # Get concurrencies from execution plan
            concurrencies_to_run = []
            for concurrency_index in self.execution_plan[test_case_id]:
                concurrency, iteration = test_case['concurrency_iterations'][
                    concurrency_index]
                concurrencies_to_run.append(
                    (concurrency_index + 1, concurrency,
                     iteration))  # +1 for 1-based display
                total_concurrencies += 1

            print(
                f"  Concurrencies to run ({len(concurrencies_to_run)}/{len(test_case['concurrency_iterations'])}):"
            )
            for concurrency_num, concurrency, iteration in concurrencies_to_run:
                print(
                    f"    {concurrency_num}. Concurrency={concurrency}, Iteration={iteration}"
                )

        print("\n" + "=" * 80)
        print(
            f"SUMMARY: {total_test_cases} test cases, {total_concurrencies} concurrencies will be executed"
        )
        print("=" * 80 + "\n")

    def generate_extra_llm_api_config(self, test_case: Dict[str, Any]) -> str:
        """Generate extra-llm-api-config.yml content"""
        config_lines = [
            "print_iter_log: true",
            f"enable_attention_dp: {str(test_case['enable_attention_dp']).lower()}",
            "disable_overlap_scheduler: false",
            "stream_interval: 10",
            f"attn_backend: {test_case['attn_backend']}",
            "cuda_graph_config:",
            "  enable_padding: true",
            f"  max_batch_size: {test_case['max_batch_size']}",
            "kv_cache_config:",
            "  dtype: fp8",
            f"  free_gpu_memory_fraction: {test_case['free_gpu_mem_fraction']}",
            "  enable_block_reuse: false",
        ]

        # Add moe_config if moe_backend is specified
        if test_case['moe_backend']:
            config_lines.append("moe_config:")
            config_lines.append(f"  backend: {test_case['moe_backend']}")

            if test_case['moe_max_num_tokens']:
                config_lines.append(
                    f"  max_num_tokens: {test_case['moe_max_num_tokens']}")

        return "\n".join(config_lines)

    def wait_for_server(self,
                        server_pid: int,
                        server_log_filename: str,
                        max_attempts: int = 360) -> bool:
        """Wait for server to be ready"""
        print("Waiting for trtllm-serve to be ready...")

        for attempt in range(1, max_attempts + 1):
            # Check if server is still running
            try:
                os.kill(server_pid, 0)  # Check if process exists
            except OSError:
                print("Error: Server process has died")
                return False

            # Check server log for runtime errors
            if self.check_for_runtime_error(server_log_filename):
                print(
                    f"RuntimeError detected in server log: {server_log_filename}"
                )
                print("Killing server process due to runtime error")
                try:
                    subprocess.run(f"kill -9 {server_pid}",
                                   shell=True,
                                   check=False)
                    subprocess.run(f"wait {server_pid} 2>/dev/null || true",
                                   shell=True,
                                   check=False)
                except Exception as e:
                    print(f"Warning: Error killing server process: {e}")
                return False

            # Try to connect to server
            try:
                response = requests.get("http://localhost:8000/v1/models",
                                        timeout=5)
                if response.status_code == 200:
                    print(
                        f"Server is ready! HTTP status: {response.status_code}")
                    return True
            except requests.RequestException:
                pass

            print(
                f"Attempt {attempt}/{max_attempts}: Server not ready yet, waiting..."
            )
            time.sleep(10)

        print(
            f"Error: Server did not become ready after {max_attempts} attempts")
        return False

    def check_for_runtime_error(self, log_file_path: str) -> bool:
        """Check if RuntimeError exists in log file"""
        try:
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as f:
                    content = f.read()
                    if "RuntimeError" in content or "runtime error" in content or "illegal memory access" in content or "terminate called" in content:
                        return True
        except Exception as e:
            print(f"Warning: Could not read log file {log_file_path}: {e}")
        return False

    def run_benchmark(self, test_case: Dict[str, Any], concurrency: int,
                      iteration: int, model_path: str,
                      server_log_filename: str) -> bool:
        """Run a single benchmark with monitoring. Returns True if successful, False if should skip test case"""
        num_prompts = concurrency * iteration

        print(
            f'Running benchmark with concurrency: {concurrency}, iteration: {iteration}, num-prompts: {num_prompts}'
        )

        # Build benchmark command
        benchmark_cmd = [
            "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model", model_path, "--dataset-name", "random", "--random-ids",
            "--num-prompts",
            str(num_prompts), "--random-input-len",
            str(test_case['isl']), "--random-output-len",
            str(test_case['osl']), "--random-range-ratio", "0.0",
            "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--max-concurrency",
            str(concurrency)
        ]

        print(f'Running benchmark with command:')
        print(' '.join(benchmark_cmd))
        print()

        # Prepare log filename
        benchmark_log_filename = f"serve.{test_case['model']}.tp{test_case['tp']}.ep{test_case['ep']}.attn{test_case['attn_backend']}.moe{test_case['moe_backend']}.gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}.isl{test_case['isl']}.osl{test_case['osl']}.tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}.concurrency{concurrency}.iter{iteration}.log"

        try:
            # Start benchmark as subprocess
            with open(benchmark_log_filename, 'w') as log_file:
                benchmark_process = subprocess.Popen(benchmark_cmd,
                                                     stdout=log_file,
                                                     stderr=subprocess.STDOUT)

            # Monitor logs every 60 seconds
            print(
                f"Starting log monitoring for benchmark process (PID: {benchmark_process.pid})"
            )
            while benchmark_process.poll() is None:  # Process is still running
                time.sleep(60)  # Wait 60 seconds

                print(
                    f"Checking logs for RuntimeError... (benchmark PID: {benchmark_process.pid})"
                )

                # Check server log for RuntimeError
                if self.check_for_runtime_error(server_log_filename):
                    print(
                        f"RuntimeError found in server log: {server_log_filename}"
                    )
                    print(
                        "Killing benchmark process and skipping this test case")
                    try:
                        subprocess.run(f"kill -9 {benchmark_process.pid}",
                                       shell=True,
                                       check=False)
                        benchmark_process.wait(timeout=10)
                    except Exception as e:
                        print(f"Warning: Error killing benchmark process: {e}")
                    return False  # Signal to skip test case

                # Check benchmark log for RuntimeError
                if self.check_for_runtime_error(benchmark_log_filename):
                    print(
                        f"RuntimeError found in benchmark log: {benchmark_log_filename}"
                    )
                    print(
                        "Killing benchmark process and skipping this test case")
                    try:
                        subprocess.run(f"kill -9 {benchmark_process.pid}",
                                       shell=True,
                                       check=False)
                        benchmark_process.wait(timeout=10)
                    except Exception as e:
                        print(f"Warning: Error killing benchmark process: {e}")
                    return False  # Signal to skip test case

            # Process completed, check final return code
            return_code = benchmark_process.returncode
            if return_code != 0:
                print(
                    f"Benchmark process completed with error code: {return_code}"
                )

                # Read and display error output
                try:
                    with open(benchmark_log_filename, 'r') as f:
                        error_content = f.read()
                        print(
                            f"Benchmark error output:\n{error_content[-1000:]}"
                        )  # Last 1000 chars
                except Exception as e:
                    print(f"Could not read benchmark log: {e}")

                print(
                    f"Skipping this concurrency level and continuing with next one..."
                )
                print("-----------------------------------------")
                return True  # Continue with next concurrency, don't skip test case

            # Success case
            print(
                f"Benchmark completed successfully (PID: {benchmark_process.pid})"
            )

            # Add configuration summary to log file
            config_summary = f"Completed benchmark with Configuration: model_label={test_case['model']}, GPUs={test_case['gpus']}, TP={test_case['tp']}, EP={test_case['ep']}, attn_backend={test_case['attn_backend']}, moe_backend={test_case['moe_backend']}, enable_attention_dp={test_case['enable_attention_dp']}, free_gpu_mem_fraction={test_case['free_gpu_mem_fraction']}, max_batch_size={test_case['max_batch_size']}, ISL={test_case['isl']}, OSL={test_case['osl']}, max_num_tokens={test_case['max_num_tokens']}, moe_max_num_tokens={test_case['moe_max_num_tokens']}, Concurrency={concurrency}"
            print(config_summary)

            with open(benchmark_log_filename, 'a') as f:
                f.write(f"\n{config_summary}\n")

            print("-----------------------------------------")
            return True  # Continue with next concurrency

        except Exception as e:
            print(
                f"Error running benchmark with concurrency {concurrency}: {e}")
            print(
                f"Skipping this concurrency level and continuing with next one..."
            )
            print("-----------------------------------------")
            return True  # Continue with next concurrency, don't skip test case

    def run_test_case(self, test_case: Dict[str, Any]) -> None:
        """Run a test case using the execution plan"""
        model_label = test_case['model']
        test_case_id = test_case['id']

        # Get model path
        model_path = self.model_paths.get(model_label)
        if not model_path:
            print(f"Error: No model path found for {model_label}")
            return

        # Use local path if it exists, otherwise use model name
        if os.path.exists(model_path):
            MODEL = model_path
        else:
            MODEL = model_label

        # Generate extra-llm-api-config.yml
        config_content = self.generate_extra_llm_api_config(test_case)
        config_path = "/tmp/extra-llm-api-config.yml"

        with open(config_path, 'w') as f:
            f.write(config_content)

        print("extra-llm-api-config.yml:")
        print(config_content)

        # Build trtllm-serve command
        serve_cmd = [
            "trtllm-serve", MODEL, "--backend", "pytorch", "--tp_size",
            str(test_case['tp']), "--ep_size",
            str(test_case['ep']), "--max_batch_size",
            str(test_case['max_batch_size']), "--max_num_tokens",
            str(test_case['max_num_tokens']),
            "--kv_cache_free_gpu_memory_fraction",
            str(test_case['free_gpu_mem_fraction']), "--extra_llm_api_options",
            config_path
        ]

        print("Starting trtllm-serve with command:")
        print(' '.join(serve_cmd))
        print()

        # Start server
        server_log_filename = f"trtllm-serve.{model_label}.tp{test_case['tp']}.ep{test_case['ep']}.attn{test_case['attn_backend']}.moe{test_case['moe_backend']}.gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}.isl{test_case['isl']}.osl{test_case['osl']}.tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}.log"

        try:
            with open(server_log_filename, 'w') as log_file:
                server_process = subprocess.Popen(serve_cmd,
                                                  stdout=log_file,
                                                  stderr=subprocess.STDOUT)

            # Wait for server to be ready
            if not self.wait_for_server(server_process.pid,
                                        server_log_filename):
                print(
                    "Failed to start server, killing process and skipping this test case"
                )
                try:
                    subprocess.run(f"kill -9 {server_process.pid}",
                                   shell=True,
                                   check=False)
                    subprocess.run(
                        f"wait {server_process.pid} 2>/dev/null || true",
                        shell=True,
                        check=False)
                except Exception as e:
                    print(f"Warning: Error during server cleanup: {e}")
                return

            # Run benchmarks based on execution plan
            for concurrency_index in self.execution_plan[test_case_id]:
                concurrency, iteration = test_case['concurrency_iterations'][
                    concurrency_index]
                should_continue = self.run_benchmark(test_case, concurrency,
                                                     iteration, MODEL,
                                                     server_log_filename)

                # If run_benchmark returns False, skip the entire test case
                if not should_continue:
                    print(
                        f"RuntimeError detected - skipping remaining concurrencies for test case {test_case_id}"
                    )
                    break

        finally:
            # Cleanup: Kill server process using shell commands like in the original bash script
            print(f"Stopping server for {model_label}")
            try:
                # Use shell commands for more reliable process killing
                subprocess.run(f"kill -9 {server_process.pid}",
                               shell=True,
                               check=False)
                subprocess.run(f"wait {server_process.pid} 2>/dev/null || true",
                               shell=True,
                               check=False)
            except Exception as e:
                print(f"Warning: Error during server cleanup: {e}")

            time.sleep(5)  # Give it time to clean up resources
            print(f"Benchmark completed for {model_label}")
            print()

    def run_benchmarks(self) -> None:
        """Main function to run all benchmarks from config file"""
        print(f"Using config file: {self.config_file}")
        if self.select_pattern:
            print(f"Select pattern: {self.select_pattern}")
        else:
            print("Select pattern: default (all test cases)")
        if self.skip_pattern:
            print(f"Skip pattern: {self.skip_pattern}")
        else:
            print("Skip pattern: default (no skipping)")

        # Load configuration
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        test_cases = config['test_cases']

        # Build execution plan
        self.build_execution_plan(test_cases)

        # Print execution plan before starting benchmarks
        self.print_execution_plan(test_cases)

        # Run each test case based on execution plan
        for i, test_case in enumerate(test_cases, 1):
            test_case_id = test_case['id']

            if test_case_id not in self.execution_plan:
                print("=" * 57)
                print(
                    f"Test case {i}/{len(test_cases)} (ID: {test_case_id}): {test_case['model']} - SKIPPED"
                )
                print("=" * 57)
                continue

            print("=" * 57)
            print(
                f"Test case {i}/{len(test_cases)} (ID: {test_case_id}): {test_case['model']}"
            )
            print(
                f"Config: GPUs={test_case['gpus']}, TP={test_case['tp']}, EP={test_case['ep']}, attn_backend={test_case['attn_backend']}, moe_backend={test_case['moe_backend']}"
            )
            print("=" * 57)

            self.run_test_case(test_case)

        print("All benchmarks completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Run benchmarks from YAML configuration file')
    parser.add_argument('--output_folder',
                        required=True,
                        help='Output folder for benchmark results')
    parser.add_argument('--config_file',
                        required=True,
                        help='Path to YAML configuration file')
    parser.add_argument(
        '--skip',
        help=
        'Skip pattern: "2,4-1" means skip test case 2 and test case 4\'s 1st concurrency'
    )
    parser.add_argument(
        '--select',
        help=
        'Select pattern: "1,3,5" means only run test cases 1, 3, and 5; "1-1,2-3" means only run test case 1\'s 1st concurrency and test case 2\'s 3rd concurrency'
    )

    args = parser.parse_args()

    try:
        subprocess.run(f'echo "TRT-LLM GIT COMMIT": $TRT_LLM_GIT_COMMIT',
                       shell=True,
                       check=True)
    except subprocess.CalledProcessError:
        print("Warning: Could not echo TRT-LLM GIT COMMIT")

    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' does not exist")
        sys.exit(1)

    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder '{args.output_folder}' does not exist")
        sys.exit(1)

    try:
        runner = BenchmarkRunner(args.output_folder, args.config_file,
                                 args.skip, args.select)
        runner.run_benchmarks()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
