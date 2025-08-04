#!/usr/bin/env python3
"""
Script to run benchmarks from YAML configuration file
Usage: python run_benchmark_serve.py --output_folder <output_folder> --commit <commit> --config_file <config_file> [--skip <skip_pattern>] [--select <select_pattern>]
Skip pattern format: "2,4-1" means skip test case 2 and test case 4's 1st concurrency
Select pattern format: "1,3,5" means only run test cases 1, 3, and 5
If select_pattern is empty, all test cases are selected
If skip_pattern is empty, no test cases are skipped
"""

import yaml
import subprocess
import sys
import os
import time
import signal
import requests
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set


class BenchmarkRunner:
    def __init__(self, output_folder: str, commit: str, config_file: str, skip_pattern: str = None, select_pattern: str = None):
        self.output_folder = Path(output_folder)
        self.commit = commit
        self.config_file = Path(config_file)
        self.skip_pattern = skip_pattern
        self.select_pattern = select_pattern
        self.skip_test_cases: Set[int] = set()
        self.skip_concurrencies: Dict[int, Set[int]] = {}
        self.select_test_cases: Set[int] = set()
        
        if skip_pattern:
            self.parse_skip_pattern(skip_pattern)
        
        if select_pattern:
            self.parse_select_pattern(select_pattern)
        
        # Model path mapping
        self.model_paths = {
            "70B-FP4": "/home/scratch.omniml_data_2/HF_model_hub/Llama-3.3-70B-Instruct-FP4",
            "70B-FP8": "/home/scratch.omniml_data_2/HF_model_hub/Llama-3.3-70B-Instruct-FP8",
            "Scout-FP4": "/home/scratch.omniml_data_2/HF_model_hub/Llama-4-Scout-17B-16E-Instruct-FP4",
            "Scout-FP8": "/home/scratch.omniml_data_2/HF_model_hub/Llama-4-Scout-17B-16E-Instruct-FP8",
            "R1-FP8": "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1/",
            "R1-FP4": "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-0528-FP4"
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
            if '-' in part:
                # Format: "test_case-concurrency_index" (1-based)
                test_case_str, concurrency_str = part.split('-')
                test_case_id = int(test_case_str)
                concurrency_index = int(concurrency_str) - 1  # Convert to 0-based
                
                if test_case_id not in self.skip_concurrencies:
                    self.skip_concurrencies[test_case_id] = set()
                self.skip_concurrencies[test_case_id].add(concurrency_index)
            else:
                # Format: "test_case" - skip entire test case
                test_case_id = int(part)
                self.skip_test_cases.add(test_case_id)
        
        print(f"Skipping test cases: {sorted(self.skip_test_cases)}")
        print(f"Skipping concurrencies: {self.skip_concurrencies}")
    
    def parse_select_pattern(self, select_pattern: str) -> None:
        """Parse select pattern like '1,3,5' to determine which test cases to run"""
        if not select_pattern:
            return
            
        parts = select_pattern.split(',')
        for part in parts:
            part = part.strip()
            if part:  # Skip empty parts
                test_case_id = int(part)
                self.select_test_cases.add(test_case_id)
        
        print(f"Selected test cases: {sorted(self.select_test_cases)}")
    
    def should_skip_test_case(self, test_case_id: int) -> bool:
        """Check if a test case should be skipped"""
        # First check if test case is in selected set (if select_pattern is specified)
        if self.select_test_cases and test_case_id not in self.select_test_cases:
            return True
        
        # Then check if test case is in skip set
        return test_case_id in self.skip_test_cases
    
    def should_skip_concurrency(self, test_case_id: int, concurrency_index: int) -> bool:
        """Check if a specific concurrency should be skipped"""
        return (test_case_id in self.skip_concurrencies and 
                concurrency_index in self.skip_concurrencies[test_case_id])
        
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
                config_lines.append(f"  max_num_tokens: {test_case['moe_max_num_tokens']}")
        
        return "\n".join(config_lines)
    
    def wait_for_server(self, server_pid: int, max_attempts: int = 360) -> bool:
        """Wait for server to be ready"""
        print("Waiting for trtllm-serve to be ready...")
        
        for attempt in range(1, max_attempts + 1):
            # Check if server is still running
            try:
                os.kill(server_pid, 0)  # Check if process exists
            except OSError:
                print("Error: Server process has died")
                return False
            
            # Try to connect to server
            try:
                response = requests.get("http://localhost:8000/v1/models", timeout=5)
                if response.status_code == 200:
                    print(f"Server is ready! HTTP status: {response.status_code}")
                    return True
            except requests.RequestException:
                pass
            
            print(f"Attempt {attempt}/{max_attempts}: Server not ready yet, waiting...")
            time.sleep(10)
        
        print(f"Error: Server did not become ready after {max_attempts} attempts")
        return False
    
    def run_benchmark(self, test_case: Dict[str, Any], concurrency: int, iteration: int, model_path: str) -> None:
        """Run a single benchmark"""
        num_prompts = concurrency * iteration
        
        print(f'Running benchmark with concurrency: {concurrency}, iteration: {iteration}, num-prompts: {num_prompts}')
        
        # Build benchmark command
        benchmark_cmd = [
            "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving",
            "--model", model_path,
            "--dataset-name", "random",
            "--random-ids",
            "--num-prompts", str(num_prompts),
            "--random-input-len", str(test_case['isl']),
            "--random-output-len", str(test_case['osl']),
            "--random-range-ratio", "0.0",
            "--ignore-eos",
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--max-concurrency", str(concurrency)
        ]
        
        print(f'Running benchmark with command:')
        print(' '.join(benchmark_cmd))
        print()
        
        # Run benchmark and capture output
        log_filename = f"serve.{test_case['model']}.tp{test_case['tp']}.ep{test_case['ep']}.attn{test_case['attn_backend']}.moe{test_case['moe_backend']}.gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}.isl{test_case['isl']}.osl{test_case['osl']}.tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}.concurrency{concurrency}.iter{iteration}.log"
        
        try:
            # Run benchmark and capture output
            result = subprocess.run(benchmark_cmd, capture_output=True, text=True, check=True)
            
            # Write output to log file
            with open(log_filename, 'w') as f:
                f.write(result.stdout)
                f.write(result.stderr)
            
            # Add configuration summary to log file
            config_summary = f"Completed benchmark with Configuration: model_label={test_case['model']}, GPUs={test_case['gpus']}, TP={test_case['tp']}, EP={test_case['ep']}, attn_backend={test_case['attn_backend']}, moe_backend={test_case['moe_backend']}, enable_attention_dp={test_case['enable_attention_dp']}, free_gpu_mem_fraction={test_case['free_gpu_mem_fraction']}, max_batch_size={test_case['max_batch_size']}, ISL={test_case['isl']}, OSL={test_case['osl']}, max_num_tokens={test_case['max_num_tokens']}, moe_max_num_tokens={test_case['moe_max_num_tokens']}, Concurrency={concurrency}"
            print(config_summary)
            
            with open(log_filename, 'a') as f:
                f.write(f"\n{config_summary}\n")
            
            print("-----------------------------------------")
            
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark with concurrency {concurrency}: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            print(f"Skipping this concurrency level and continuing with next one...")
            print("-----------------------------------------")

    def run_test_case(self, test_case: Dict[str, Any]) -> None:
        """Run a complete test case with multiple concurrency/iteration combinations"""
        model_label = test_case['model']
        
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
            "trtllm-serve", MODEL,
            "--backend", "pytorch",
            "--tp_size", str(test_case['tp']),
            "--ep_size", str(test_case['ep']),
            "--max_batch_size", str(test_case['max_batch_size']),
            "--max_num_tokens", str(test_case['max_num_tokens']),
            "--kv_cache_free_gpu_memory_fraction", str(test_case['free_gpu_mem_fraction']),
            "--extra_llm_api_options", config_path
        ]
        
        print("Starting trtllm-serve with command:")
        print(' '.join(serve_cmd))
        print()
        
        # Start server
        server_log_filename = f"trtllm-serve.{model_label}.tp{test_case['tp']}.ep{test_case['ep']}.attn{test_case['attn_backend']}.moe{test_case['moe_backend']}.gpu{test_case['free_gpu_mem_fraction']}.batch{test_case['max_batch_size']}.isl{test_case['isl']}.osl{test_case['osl']}.tokens{test_case['max_num_tokens']}.moetokens{test_case['moe_max_num_tokens']}.log"
        
        try:
            with open(server_log_filename, 'w') as log_file:
                server_process = subprocess.Popen(
                    serve_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
            
            # Wait for server to be ready
            if not self.wait_for_server(server_process.pid):
                print("Failed to start server, killing process and skipping this test case")
                try:
                    subprocess.run(f"kill -9 {server_process.pid}", shell=True, check=False)
                    subprocess.run(f"wait {server_process.pid} 2>/dev/null || true", shell=True, check=False)
                except Exception as e:
                    print(f"Warning: Error during server cleanup: {e}")
                return
            
            # Run all benchmarks for this test case
            for concurrency_index, (concurrency, iteration) in enumerate(test_case['concurrency_iterations']):
                if self.should_skip_concurrency(test_case['id'], concurrency_index):
                    print(f"Skipping concurrency {concurrency} (index {concurrency_index}) for test case {test_case['id']}")
                    continue
                self.run_benchmark(test_case, concurrency, iteration, MODEL)
            
        finally:
            # Cleanup: Kill server process using shell commands like in the original bash script
            print(f"Stopping server for {model_label}")
            try:
                # Use shell commands for more reliable process killing
                subprocess.run(f"kill -9 {server_process.pid}", shell=True, check=False)
                subprocess.run(f"wait {server_process.pid} 2>/dev/null || true", shell=True, check=False)
            except Exception as e:
                print(f"Warning: Error during server cleanup: {e}")
            
            time.sleep(5)  # Give it time to clean up resources
            print(f"Benchmark completed for {model_label}")
            print()
    
    def run_benchmarks(self) -> None:
        """Main function to run all benchmarks from config file"""
        print(f"TRT-LLM GIT COMMIT: {self.commit}")
        print(f"Using config file: {self.config_file}")
        if self.select_pattern:
            print(f"Select pattern: {self.select_pattern}")
        if self.skip_pattern:
            print(f"Skip pattern: {self.skip_pattern}")
        
        # Load configuration
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        test_cases = config['test_cases']
        
        # Run nvidia-smi to show GPU status
        try:
            subprocess.run(["nvidia-smi"], check=True)
        except subprocess.CalledProcessError:
            print("Warning: nvidia-smi failed")
        
        # Run each test case
        for i, test_case in enumerate(test_cases, 1):
            test_case_id = test_case['id']
            
            if self.should_skip_test_case(test_case_id):
                print("=" * 57)
                print(f"Test case {i}/{len(test_cases)} (ID: {test_case_id}): {test_case['model']} - SKIPPED")
                print("=" * 57)
                continue
            
            print("=" * 57)
            print(f"Test case {i}/{len(test_cases)} (ID: {test_case_id}): {test_case['model']}")
            print(f"Config: GPUs={test_case['gpus']}, TP={test_case['tp']}, EP={test_case['ep']}, attn_backend={test_case['attn_backend']}, moe_backend={test_case['moe_backend']}")
            print("=" * 57)
            
            self.run_test_case(test_case)
        
        print("All benchmarks completed!")


def main():
    parser = argparse.ArgumentParser(description='Run benchmarks from YAML configuration file')
    parser.add_argument('--output_folder', required=True, help='Output folder for benchmark results')
    parser.add_argument('--commit', required=True, help='Git commit ID')
    parser.add_argument('--config_file', required=True, help='Path to YAML configuration file')
    parser.add_argument('--skip', help='Skip pattern: "2,4-1" means skip test case 2 and test case 4\'s 1st concurrency')
    parser.add_argument('--select', help='Select pattern: "1,3,5" means only run test cases 1, 3, and 5')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_file):
        print(f"Error: Config file '{args.config_file}' does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder '{args.output_folder}' does not exist")
        sys.exit(1)
    
    try:
        runner = BenchmarkRunner(args.output_folder, args.commit, args.config_file, args.skip, args.select)
        runner.run_benchmarks()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 