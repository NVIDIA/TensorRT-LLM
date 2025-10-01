# Adopted from
# https://github.com/vllm-project/vllm/blob/200bbf92e8861e2458a6f90bca73f40cc3b1ad1f/benchmarks/benchmark_utils.py
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo
import requests
from tqdm.asyncio import tqdm


def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = extra_info["tensor_parallel_size"]

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)


def download_and_cache_file(url: str, path: Optional[str], name: str,
                            timeout: int) -> str:
    # Adapted from
    # https://github.com/sgl-project/sglang/blob/58f10679e1850fdc86046057c23bac5193156de9/python/sglang/bench_serving.py#L586
    """Read and cache a file from a url."""

    # Check if the path is valid and if the file exists
    if path is None or not os.path.exists(path):
        raise ValueError("download_path is not provided or does not exist")
    filename = os.path.join(path, name)

    if is_file_valid_json(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def is_file_valid_json(path) -> bool:
    # Adapted from
    # https://github.com/sgl-project/sglang/blob/58f10679e1850fdc86046057c23bac5193156de9/python/sglang/bench_serving.py#L620
    if not os.path.isfile(path):
        return False

    # TODO can fuse into the real file open later
    try:
        with open(path) as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(
            f"{path} exists but json loading fails ({e=}), thus treat as invalid file"
        )
        return False


@dataclass
class TimingMetric:
    """Configuration for a timing metric segment."""
    name: str
    display_name: str
    color: str
    description: str
    start_field: str
    end_field: str
    server_type: Optional[
        str] = None  # 'ctx', 'gen', 'disagg', or None for direct calculation

    def calculate_duration(self, timing_data: Dict[str, float]) -> float:
        """Calculate the duration for this metric from timing data."""
        start_time = timing_data.get(self.start_field, 0)
        end_time = timing_data.get(self.end_field, 0)

        # If either timestamp is 0 (not available), return 0 duration
        if start_time == 0 or end_time == 0:
            return 0.0

        return max(0, end_time - start_time)


class TimingMetricsConfig:
    """Configuration class that defines all available timing metrics."""

    def __init__(self):
        self.metrics = [
            TimingMetric(
                name='disagg_preprocessing',
                display_name='Disagg Preprocessing',
                color='lightgray',
                description=
                'Time duration from the disagg server receives the request to a context server receives it',
                start_field='disagg_server_arrival_time',
                end_field='ctx_server_arrival_time',
                server_type='disagg'),
            TimingMetric(
                name='ctx_preprocessing',
                display_name='Context Preprocessing',
                color='lightgreen',
                description=
                'Time duration from a context server receives the request to a LLM worker queues it',
                start_field='ctx_server_arrival_time',
                end_field='ctx_arrival_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_queue',
                display_name='Context Queue',
                color='lightblue',
                description=
                'Time duration from the request is queued to first scheduled',
                start_field='ctx_arrival_time',
                end_field='ctx_first_scheduled_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_processing',
                display_name='Context Processing',
                color='blue',
                description=
                'Time duration from first scheduled to first token generated on a LLM worker',
                start_field='ctx_first_scheduled_time',
                end_field='ctx_first_token_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_postprocessing',
                display_name='Context Postprocessing',
                color='purple',
                description=
                'Time duration from first token generated on a LLM worker to the first token response sent by the context server',
                start_field='ctx_first_token_time',
                end_field='ctx_server_first_token_time',
                server_type='ctx'),
            TimingMetric(
                name='gen_preprocessing',
                display_name='Generation Preprocessing',
                color='lightyellow',
                description=
                'Time duration from a generation server receives the request to a LLM worker receives it',
                start_field='gen_server_arrival_time',
                end_field='gen_arrival_time',
                server_type='gen'),
            TimingMetric(
                name='gen_queue',
                display_name='Generation Queue',
                color='lightcoral',
                description=
                'Time duration from the request is queued to first scheduled',
                start_field='gen_arrival_time',
                end_field='gen_first_scheduled_time',
                server_type='gen'),
            TimingMetric(
                name='gen_postprocessing',
                display_name='Generation Postprocessing',
                color='grey',
                description=
                'Time duration from first scheduled to the first token response sent by the generation server',
                start_field='gen_first_scheduled_time',
                end_field='gen_server_first_token_time',
                server_type='gen'),
            TimingMetric(
                name='disagg_postprocessing',
                display_name='Disagg Postprocessing',
                color='gray',
                description=
                'Time duration from the first token response sent by the generation server to sent by the disagg server',
                start_field='gen_server_first_token_time',
                end_field='disagg_server_first_token_time',
                server_type='disagg')
        ]

    def get_metric_by_name(self, name: str) -> Optional[TimingMetric]:
        """Get a metric by its name."""
        return next((m for m in self.metrics if m.name == name), None)

    def get_metrics_by_server(self, server_type: str) -> List[TimingMetric]:
        """Get all metrics for a specific server type."""
        return [m for m in self.metrics if m.server_type == server_type]

    def add_metric(self, metric: TimingMetric):
        """Add a new timing metric."""
        self.metrics.append(metric)

    def remove_metric(self, name: str):
        """Remove a timing metric by name."""
        self.metrics = [m for m in self.metrics if m.name != name]


class RequestDataParser:
    """Parser for disaggregated format with ctx_perf_metrics and gen_perf_metrics."""

    def parse_request(self, request_data: Dict,
                      request_index: int) -> Dict[str, Any]:
        is_disaggregated = 'ctx_perf_metrics' in request_data and 'gen_perf_metrics' in request_data

        ctx_metrics = {}
        gen_metrics = {}
        if is_disaggregated:
            ctx_metrics = request_data.get('ctx_perf_metrics', {}).get(
                'perf_metrics', {}).get('timing_metrics', {})
            gen_metrics = request_data.get('gen_perf_metrics', {}).get(
                'perf_metrics', {}).get('timing_metrics', {})
        else:
            ctx_metrics = request_data.get('perf_metrics',
                                           {}).get('timing_metrics', {})

        ctx_arrival_time = ctx_metrics.get('arrival_time', 0)
        ctx_first_scheduled_time = ctx_metrics.get('first_scheduled_time', 0)
        ctx_first_token_time = ctx_metrics.get('first_token_time', 0)
        ctx_server_arrival_time = ctx_metrics.get('server_arrival_time', 0)
        ctx_server_first_token_time = ctx_metrics.get('server_first_token_time',
                                                      0)

        gen_server_first_token_time = gen_metrics.get('server_first_token_time',
                                                      0)
        gen_server_arrival_time = gen_metrics.get('server_arrival_time', 0)
        gen_arrival_time = gen_metrics.get('arrival_time', 0)
        gen_first_token_time = gen_metrics.get('first_token_time', 0)
        gen_first_scheduled_time = gen_metrics.get('first_scheduled_time', 0)

        disagg_server_arrival_time = 0
        disagg_server_first_token_time = 0
        if is_disaggregated:
            disagg_server_arrival_time = request_data.get(
                'disagg_server_arrival_time', 0)
            disagg_server_first_token_time = request_data.get(
                'disagg_server_first_token_time', 0)

        # Get request ID
        if is_disaggregated:
            request_id = request_data.get('ctx_perf_metrics',
                                          {}).get('request_id', request_index)
        else:
            request_id = request_data.get('request_id', request_index)

        return {
            'request_index': request_id,
            'ctx_server_arrival_time': ctx_server_arrival_time,
            'ctx_arrival_time': ctx_arrival_time,
            'ctx_first_scheduled_time': ctx_first_scheduled_time,
            'ctx_first_token_time': ctx_first_token_time,
            'ctx_server_first_token_time': ctx_server_first_token_time,
            'gen_server_arrival_time': gen_server_arrival_time,
            'gen_arrival_time': gen_arrival_time,
            'gen_first_scheduled_time': gen_first_scheduled_time,
            'gen_first_token_time': gen_first_token_time,
            'gen_server_first_token_time': gen_server_first_token_time,
            'disagg_server_arrival_time': disagg_server_arrival_time,
            'disagg_server_first_token_time': disagg_server_first_token_time,
        }


class RequestTimingBreakdown:
    """Main class for analyzing request timing breakdown."""

    def __init__(self, config: Optional[TimingMetricsConfig] = None):
        self.config = config or TimingMetricsConfig()
        self.parser = RequestDataParser()

    def parse_json_file(self, json_file_path: str) -> List[Dict]:
        """Parse JSON performance metrics file and extract timing information."""
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{json_file_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file '{json_file_path}': {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file '{json_file_path}': {e}")
            sys.exit(1)

        timing_data = []

        for i, request in enumerate(data):
            parsed_data = self.parser.parse_request(request, i)

            # Calculate durations for each metric
            for metric in self.config.metrics:
                duration = metric.calculate_duration(parsed_data)
                parsed_data[f'{metric.name}_time'] = duration

            timing_data.append(parsed_data)

        if timing_data:
            has_gen_metrics = any(entry['gen_server_arrival_time'] > 0
                                  for entry in timing_data)
            format_type = "disaggregated " if has_gen_metrics else "aggregated"
            print(
                f"Parsed timing data for {len(timing_data)} requests from {json_file_path} ({format_type} format)"
            )
        else:
            print(f"Parsed timing data for 0 requests from {json_file_path}")

        return timing_data

    def create_timing_diagram(self,
                              timing_data: List[Dict],
                              output_file: str = None):
        """Create an interactive HTML stacked bar chart showing time breakdown."""
        if not timing_data:
            print("No timing data to visualize.")
            return

        # Extract data for plotting
        request_indices = [data['request_index'] for data in timing_data]

        # Create the interactive plot
        fig = go.Figure()

        # Add traces for each metric
        for metric in self.config.metrics:
            times_ms = [
                data.get(f'{metric.name}_time', 0) * 1000
                for data in timing_data
            ]

            # Only add trace if there's some non-zero data
            if any(t > 0 for t in times_ms):
                fig.add_trace(
                    go.Bar(
                        x=request_indices,
                        y=times_ms,
                        name=metric.display_name,
                        marker_color=metric.color,
                        hovertemplate=
                        f'<b>Request %{{x}}</b><br>{metric.display_name}: %{{y:.2f}} ms<br>{metric.description}<extra></extra>'
                    ))

        # Update layout
        fig.update_layout(
            barmode='stack',
            title={
                'text':
                'Request Processing Time Breakdown<br><sub>Time Spent in Each Segment (Interactive)</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {
                    'size': 16
                }
            },
            xaxis_title='Request Index',
            yaxis_title='Time (milliseconds)',
            hovermode='x unified',
            legend=dict(orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02),
            width=1200,
            height=700,
            margin=dict(r=200))

        # Calculate and add statistics
        self._add_statistics_annotation(fig, timing_data)

        # Set default output filename if not provided
        if not output_file:
            output_file = 'timing_breakdown.html'
        elif not output_file.endswith('.html'):
            output_file += '.html'

        # Save as HTML
        pyo.plot(fig, filename=output_file, auto_open=False)
        print(f"Interactive timing breakdown diagram saved to: {output_file}")
        print(f"Open the file in your web browser to interact with the chart!")

    def _add_statistics_annotation(self, fig, timing_data: List[Dict]):
        """Add statistics annotation to the plot."""
        # Calculate median times for each metric
        stats_lines = ['<b>Median Times (ms):</b>']
        total_times = []

        for metric in self.config.metrics:
            times = [
                data.get(f'{metric.name}_time', 0) * 1000
                for data in timing_data
            ]
            if any(t > 0 for t in times):
                median_time = np.median(times)
                stats_lines.append(f'{metric.display_name}: {median_time:.2f}')

        # Calculate total time per request
        for data in timing_data:
            total = sum(
                data.get(f'{metric.name}_time', 0) * 1000
                for metric in self.config.metrics)
            total_times.append(total)

        if total_times:
            median_total = np.median(total_times)
            stats_lines.append(f'<b>Total per Request: {median_total:.2f}</b>')

        stats_lines.append(f'<b>Requests: {len(timing_data)}</b>')

        stats_text = '<br>'.join(stats_lines)

        fig.add_annotation(x=0.98,
                           y=0.98,
                           xref='paper',
                           yref='paper',
                           text=stats_text,
                           showarrow=False,
                           align='right',
                           bgcolor='rgba(255, 255, 255, 0.8)',
                           bordercolor='black',
                           borderwidth=1,
                           font=dict(size=10))

    def show_statistics(self, timing_data: List[Dict]):
        """Show detailed statistics about the timing data."""
        if not timing_data:
            print("No timing data to analyze.")
            return

        print("\n=== Timing Statistics ===")
        print(f"Total requests: {len(timing_data)}")

        for metric in self.config.metrics:
            times = [data.get(f'{metric.name}_time', 0) for data in timing_data]
            if any(t > 0 for t in times):
                print(f"\n{metric.display_name} Times (seconds):")
                print(f"  Range: {min(times):.3f} to {max(times):.3f}")
                print(f"  Median: {np.median(times):.3f}")
                print(f"  Description: {metric.description}")
