#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path


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


def extract_server_and_client_name_from_log(log_file):
    """
    Extract server name, client name, and performance metrics from log file.
    Looks for pattern: Server-Config: <server_name>-<client_name>
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()

            # Look for Server-Config pattern
            server_config_match = re.search(r'Server-Config:\s*(\S+)', content)
            if not server_config_match:
                print(
                    f"Warning: Could not find 'Server-Config:' pattern in {log_file}"
                )
                return None, None, None

            # Extract the full config name
            config_name = server_config_match.group(1)

            # Split on the last '-' to separate server and client names
            # Format: <server_name>-<client_name>
            parts = config_name.rsplit('-', 1)
            if len(parts) != 2:
                print(
                    f"Warning: Invalid Server-Config format in {log_file}: {config_name}"
                )
                return None, None, None

            server_name = parts[0]
            client_name = parts[1]

            # Extract PerfMetrics
            perf_metrics = PerfMetrics.from_log_content(content)

            return server_name, client_name, perf_metrics

    except Exception as e:
        print(f"Warning: Could not read {log_file}: {e}")
        return None, None, None


def parse_benchmark_results(log_folder):
    """
    Parse benchmark results from log files and print grouped by server and client names
    """
    log_folder = Path(log_folder)

    # Validate inputs
    if not log_folder.exists():
        print(f"Error: Input folder '{log_folder}' does not exist")
        return

    if not log_folder.is_dir():
        print(f"Error: '{log_folder}' is not a directory")
        return

    # Find all trtllm-benchmark.*.log files
    log_files = list(log_folder.glob("trtllm-benchmark.*.log"))
    print(f"Found {len(log_files)} log files to process")

    # Dictionary to group results by server name and client name
    # Structure: {server_name: {client_name: perf_metrics}}
    results_by_server = {}

    # Process each log file
    parsed_count = 0
    for log_file in log_files:
        # Extract server name, client name, and PerfMetrics from log
        server_name, client_name, perf_metrics = extract_server_and_client_name_from_log(
            log_file)
        if not server_name or not client_name or not perf_metrics:
            continue

        parsed_count += 1

        # Group results by server name and client name
        if server_name not in results_by_server:
            results_by_server[server_name] = {}

        results_by_server[server_name][client_name] = perf_metrics

    print(f"Successfully parsed {parsed_count} log files\n")

    # Print grouped results
    print_grouped_results(results_by_server)


def print_grouped_results(results_by_server):
    """
    Print benchmark results grouped by server name and client name
    """
    print("=" * 100)

    # Sort server names for consistent output
    for server_name in sorted(results_by_server.keys()):
        print(f"Server Name: {server_name}")

        # Sort client names for consistent output
        for client_name in sorted(results_by_server[server_name].keys()):
            perf_metrics = results_by_server[server_name][client_name]

            print(f"Client Name: {client_name}")
            print(
                f"Benchmark duration (s): {perf_metrics.benchmark_duration:.2f} "
                f"Request throughput (req/s): {perf_metrics.request_throughput:.2f} "
                f"Output token throughput (tok/s): {perf_metrics.output_token_throughput:.2f} "
                f"Total Token throughput (tok/s): {perf_metrics.total_token_throughput:.2f} "
                f"User throughput (tok/s): {perf_metrics.user_throughput:.2f} "
                f"Mean TTFT (ms): {perf_metrics.mean_ttft_ms:.2f} "
                f"Median TTFT (ms): {perf_metrics.median_ttft_ms:.2f} "
                f"P99 TTFT (ms): {perf_metrics.p99_ttft_ms:.2f}")

        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description=
        "Script to parse benchmark metrics from log files and print grouped by server and client names",
        epilog=
        "Example: python parse_benchmark_results.py --log_folder ./benchmark_logs"
    )
    parser.add_argument(
        "--log_folder",
        required=True,
        help="Folder containing benchmark log files (trtllm-benchmark.*.log)")

    args = parser.parse_args()

    # Validate inputs
    log_folder_path = Path(args.log_folder)

    if not log_folder_path.exists():
        print(f"Error: Input folder '{args.log_folder}' not found.")
        sys.exit(1)
    if not log_folder_path.is_dir():
        print(f"Error: '{args.log_folder}' is not a directory.")
        sys.exit(1)

    parse_benchmark_results(args.log_folder)


if __name__ == "__main__":
    main()
