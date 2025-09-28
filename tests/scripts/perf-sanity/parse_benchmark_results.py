#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
# Import ServerConfig, ClientConfig, and parse_config_file from run_benchmark_serve.py
from run_benchmark_serve import (CLIENT_CONFIG_METRICS, SERVER_CONFIG_METRICS,
                                 ClientConfig, ServerConfig, parse_config_file)


class PerfMetrics:
    """Class to store and parse performance metrics from benchmark logs"""

    def __init__(self):
        # Basic metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.benchmark_duration = 0.0
        self.total_input_tokens = 0
        self.total_generated_tokens = 0
        self.request_throughput = 0.0
        self.output_token_throughput = 0.0
        self.total_token_throughput = 0.0
        self.user_throughput = 0.0
        self.avg_decoded_tokens_per_iter = 0.0

        # Time to First Token (TTFT)
        self.mean_ttft_ms = 0.0
        self.median_ttft_ms = 0.0
        self.p99_ttft_ms = 0.0

        # Time per Output Token (TPOT)
        self.mean_tpot_ms = 0.0
        self.median_tpot_ms = 0.0
        self.p99_tpot_ms = 0.0

        # Inter-token Latency (ITL)
        self.mean_itl_ms = 0.0
        self.median_itl_ms = 0.0
        self.p99_itl_ms = 0.0

        # End-to-end Latency (E2EL)
        self.mean_e2el_ms = 0.0
        self.median_e2el_ms = 0.0
        self.p99_e2el_ms = 0.0

    def to_str(self) -> str:
        return f"Total Requests: {self.total_requests}, Successful Requests: {self.successful_requests}, Failed Requests: {self.failed_requests}, Benchmark Duration (s): {self.benchmark_duration}, Total Input Tokens: {self.total_input_tokens}, Total Generated Tokens: {self.total_generated_tokens}, Request Throughput (req/s): {self.request_throughput}, Output Token Throughput (tok/s): {self.output_token_throughput}, Total Token Throughput (tok/s): {self.total_token_throughput}, User Throughput (tok/s): {self.user_throughput}, Avg Decoded Tokens per Iter: {self.avg_decoded_tokens_per_iter}, Mean TTFT (ms): {self.mean_ttft_ms}, Median TTFT (ms): {self.median_ttft_ms}, P99 TTFT (ms): {self.p99_ttft_ms}, Mean TPOT (ms): {self.mean_tpot_ms}, Median TPOT (ms): {self.median_tpot_ms}, P99 TPOT (ms): {self.p99_tpot_ms}, Mean ITL (ms): {self.mean_itl_ms}, Median ITL (ms): {self.median_itl_ms}, P99 ITL (ms): {self.p99_itl_ms}, Mean E2EL (ms): {self.mean_e2el_ms}, Median E2EL (ms): {self.median_e2el_ms}, P99 E2EL (ms): {self.p99_e2el_ms}"

    @classmethod
    def from_log_content(cls, log_content: str) -> 'PerfMetrics':
        """Parse performance metrics from log content"""
        metrics = cls()

        # Define patterns for each metric
        patterns = {
            'total_requests': r'Total requests:\s+(\d+)',
            'successful_requests': r'Successful requests:\s+(\d+)',
            'failed_requests': r'Failed requests:\s+(\d+)',
            'benchmark_duration': r'Benchmark duration \(s\):\s+([\d.]+)',
            'total_input_tokens': r'Total input tokens:\s+(\d+)',
            'total_generated_tokens': r'Total generated tokens:\s+(\d+)',
            'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
            'output_token_throughput':
            r'Output token throughput \(tok/s\):\s+([\d.]+)',
            'total_token_throughput':
            r'Total Token throughput \(tok/s\):\s+([\d.]+)',
            'user_throughput': r'User throughput \(tok/s\):\s+([\d.]+)',
            'avg_decoded_tokens_per_iter':
            r'Avg Decoded Tokens per Iter:\s+([\d.]+)',
            'mean_ttft_ms': r'Mean TTFT \(ms\):\s+([\d.]+)',
            'median_ttft_ms': r'Median TTFT \(ms\):\s+([\d.]+)',
            'p99_ttft_ms': r'P99 TTFT \(ms\):\s+([\d.]+)',
            'mean_tpot_ms': r'Mean TPOT \(ms\):\s+([\d.]+)',
            'median_tpot_ms': r'Median TPOT \(ms\):\s+([\d.]+)',
            'p99_tpot_ms': r'P99 TPOT \(ms\):\s+([\d.]+)',
            'mean_itl_ms': r'Mean ITL \(ms\):\s+([\d.]+)',
            'median_itl_ms': r'Median ITL \(ms\):\s+([\d.]+)',
            'p99_itl_ms': r'P99 ITL \(ms\):\s+([\d.]+)',
            'mean_e2el_ms': r'Mean E2EL \(ms\):\s+([\d.]+)',
            'median_e2el_ms': r'Median E2EL \(ms\):\s+([\d.]+)',
            'p99_e2el_ms': r'P99 E2EL \(ms\):\s+([\d.]+)',
        }

        # Parse each metric
        for attr_name, pattern in patterns.items():
            match = re.search(pattern, log_content)
            if match:
                value = match.group(1)
                try:
                    if '.' in value:
                        setattr(metrics, attr_name, float(value))
                    else:
                        setattr(metrics, attr_name, int(value))
                except ValueError:
                    # Keep default value if parsing fails
                    pass

        return metrics


def extract_server_and_client_config_from_log(log_file):
    """
    Extract ServerConfig and ClientConfig from log file content using the new format:
    [Perf Sanity Test] Server Config: ... and [Perf Sanity Test] Client Config: ...
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()

            # Extract ServerConfig
            server_config = extract_server_config_from_content(content)
            if server_config is None:
                print(
                    f"Warning: Could not extract ServerConfig from {log_file}")
                return None, None, None

            # Extract ClientConfig
            client_config = extract_client_config_from_content(content)
            if client_config is None:
                print(
                    f"Warning: Could not extract ClientConfig from {log_file}")
                return None, None, None

            # Extract PerfMetrics
            perf_metrics = PerfMetrics.from_log_content(content)

            return server_config, client_config, perf_metrics

    except Exception as e:
        print(f"Warning: Could not read {log_file}: {e}")
        return None, None, None


def extract_server_config_from_content(content):
    """
    Extract ServerConfig from log content
    """
    # Find the Server Config section
    server_config_match = re.search(
        r'\[Perf Sanity Test\] Server Config:\s*\n(.*?)(?=\[Perf Sanity Test\] Client Config:|$)',
        content, re.DOTALL)
    if not server_config_match:
        return None

    config_text = server_config_match.group(1)

    # Parse line by line to avoid regex issues with empty values
    config_values = {}
    lines = config_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in SERVER_CONFIG_METRICS.keys() and value != "":
                config_values[key] = SERVER_CONFIG_METRICS[key][1](value)

    # Check for required fields
    for key in SERVER_CONFIG_METRICS.keys():
        if not SERVER_CONFIG_METRICS[key][0] and (key not in config_values
                                                  or config_values[key] == ""):
            print(f"Warning: Missing required field '{key}' in ServerConfig")
            return None

    return ServerConfig(**config_values)


def extract_client_config_from_content(content):
    """
    Extract ClientConfig from log content
    """
    # Find the Client Config section
    client_config_match = re.search(
        r'\[Perf Sanity Test\] Client Config:\s*\n(.*?)(?=\[|$)', content,
        re.DOTALL)
    if not client_config_match:
        return None

    config_text = client_config_match.group(1)

    # Parse line by line
    config_values = {}
    lines = config_text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key in CLIENT_CONFIG_METRICS.keys() and value != "":
                config_values[key] = CLIENT_CONFIG_METRICS[key][1](value)

    # Check for required fields
    for key in CLIENT_CONFIG_METRICS.keys():
        if not CLIENT_CONFIG_METRICS[key][0] and (key not in config_values
                                                  or config_values[key] == ""):
            print(f"Warning: Missing required field '{key}' in ClientConfig")
            return None

    return ClientConfig(**config_values)


def match_log_to_server_and_client_config(log_server_config, log_client_config,
                                          server_configs, client_configs):
    """
    Check if log_server_config and log_client_config match a server_config and client_config
    """
    if not log_server_config or not log_client_config:
        return False

    # Check if log_server_config and log_client_config match a server_config and client_config
    for server_config_id, server_config in server_configs:
        if log_server_config.to_str() == server_config.to_str():
            for client_config in client_configs[server_config_id]:
                if log_client_config.to_str() == client_config.to_str():
                    return True
    return False


def parse_benchmark_results(input_folder, output_csv, config_file, print_perf,
                            generate_csv):
    """
    Parse benchmark results and generate CSV table
    """
    input_folder = Path(input_folder)
    config_file = Path(config_file)

    # Validate inputs
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return

    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        return

    if not config_file.exists():
        print(f"Error: Config file '{config_file}' does not exist")
        return

    server_configs, client_configs = parse_config_file(config_file)

    # Find all trtllm-benchmark.*.log files
    log_files = list(input_folder.glob("trtllm-benchmark.*.log"))
    print(f"Found {len(log_files)} log files to process")

    # Process each log file
    matched_count = 0
    parsed_results = []  # Store all parsed results for summary printing

    for log_file in log_files:
        # Extract ServerConfig, ClientConfig, and PerfMetrics from log
        server_config, client_config, perf_metrics = extract_server_and_client_config_from_log(
            log_file)
        if not server_config or not client_config or not perf_metrics:
            print(f"  Skipped - could not parse configuration or metrics")
            continue

        # Store parsed results for summary printing
        parsed_results.append((server_config, client_config, perf_metrics))

        # Match log to test case
        matched = match_log_to_server_and_client_config(server_config,
                                                        client_config,
                                                        server_configs,
                                                        client_configs)
        if matched:
            matched_count += 1
        else:
            print(
                f"  Skipped - no matching test case found for server_config={server_config.model_name}, client_config={client_config.concurrency}"
            )

    print(f"Successfully matched {matched_count} log files to test cases")

    # Print summary of all parsed results if requested
    if print_perf and parsed_results:
        print_all_configs_summary(parsed_results)
    # Generate CSV file of results if requested
    if generate_csv:
        generate_csv_summary(server_configs, client_configs, parsed_results,
                             output_csv)


def print_all_configs_summary(parsed_results):
    """
    Print summary of all ServerConfigs and their ClientConfigs with performance metrics
    """
    print("=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    # Group results by ServerConfig

    for server_config, client_config, perf_metrics in parsed_results:
        # Create a key for ServerConfig comparison
        print("ServerConfig: ")
        print(server_config.to_str())
        print("ClientConfig: ")
        print(client_config.to_str())
        print("PerfMetrics: ")
        print(perf_metrics.to_str())
        print("-" * 100)


def generate_csv_summary(server_configs, client_configs, parsed_results,
                         output_csv):
    """
    Generate a CSV file of results with specified columns
    """
    # Create a mapping of (server_config, client_config) -> perf_metrics for quick lookup
    parsed_results_map = {}
    for server_config, client_config, perf_metrics in parsed_results:
        # Create a key using the string representation for matching
        server_key = server_config.to_str()
        client_key = client_config.to_str()
        parsed_results_map[(server_key, client_key)] = perf_metrics

    # Define CSV columns
    columns = [
        'ServerConfig', 'TP', 'EP', 'Concurrency', 'Iterations',
        'Total Token Throughput', 'User Throughput', 'Median TTFT',
        'Benchmark Duration'
    ]

    # Generate CSV rows
    csv_rows = []

    for server_config_id, server_config in server_configs:
        # Add empty row before each new server config (except the first one)
        if csv_rows:
            empty_row = {col: '' for col in columns}
            csv_rows.append(empty_row)

        # Process each client config for this server config
        for client_config in client_configs[server_config_id]:
            # Create row for this test case
            row = {
                'ServerConfig': server_config.to_str(),
                'TP': server_config.tp,
                'EP': server_config.ep,
                'Concurrency': client_config.concurrency,
                'Iterations': client_config.iterations,
                'Total Token Throughput': '',
                'User Throughput': '',
                'Median TTFT': '',
                'Benchmark Duration': ''
            }

            # Try to find matching performance metrics
            server_key = server_config.to_str()
            client_key = client_config.to_str()

            if (server_key, client_key) in parsed_results_map:
                perf_metrics = parsed_results_map[(server_key, client_key)]
                row['Total Token Throughput'] = perf_metrics.total_token_throughput
                row['User Throughput'] = perf_metrics.user_throughput
                row['Median TTFT'] = perf_metrics.median_ttft_ms
                row['Benchmark Duration'] = perf_metrics.benchmark_duration

            csv_rows.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_rows)

    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"Table summary saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=
        "Script to parse benchmark metrics from a specified folder and generate CSV file",
        epilog=
        "Example: python parse_benchmark_results.py ./benchmark_logs results.csv ./benchmark_config.yaml"
    )
    parser.add_argument(
        "--input_folder",
        help="Folder containing benchmark log files (serve.*.log)")
    parser.add_argument("--config_file",
                        help="Path to benchmark_config.yaml file")
    parser.add_argument("--print_perf",
                        action="store_true",
                        help="Print performance summary for each test case")
    parser.add_argument("--generate_csv",
                        action="store_true",
                        help="Generate the CSV file of results")
    parser.add_argument("--output_csv",
                        default="",
                        help="Output CSV filename for the results")

    args = parser.parse_args()

    if not args.print_perf and not args.generate_csv:
        print("Error: Either --print_perf or --generate_csv must be specified")
        sys.exit(1)

    if args.generate_csv and not args.output_csv:
        print(
            "Error: --output_csv must be specified when --generate_csv is specified"
        )
        sys.exit(1)

    # Validate inputs
    input_folder_path = Path(args.input_folder)
    config_file_path = Path(args.config_file)

    if not input_folder_path.exists():
        print(f"Error: Input folder '{args.input_folder}' not found.")
        sys.exit(1)
    if not input_folder_path.is_dir():
        print(f"Error: '{args.input_folder}' is not a directory.")
        sys.exit(1)
    if not config_file_path.exists():
        print(f"Error: Config file '{args.config_file}' not found.")
        sys.exit(1)

    parse_benchmark_results(args.input_folder, args.output_csv,
                            args.config_file, args.print_perf,
                            args.generate_csv)


if __name__ == "__main__":
    main()
