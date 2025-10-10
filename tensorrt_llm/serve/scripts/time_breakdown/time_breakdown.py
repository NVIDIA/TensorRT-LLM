#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Time Breakdown Analysis Tool

This module provides tools for analyzing and visualizing request time breakdown
from TensorRT-LLM server performance metrics. It can be used both as a library
and as a standalone CLI tool.

Usage as CLI:
    python time_breakdown.py <json_file> [options]

Usage as library:
    from time_breakdown import RequestTimeBreakdown
    analyzer = RequestTimeBreakdown()
    timing_data = analyzer.parse_json_file("perf_metrics.json")
    analyzer.create_timing_diagram(timing_data, "output.html")
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.offline as pyo


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
        start_time = timing_data.get(self.start_field, float('nan'))
        end_time = timing_data.get(self.end_field, float('nan'))

        # If either timestamp is NaN (not available), return NaN duration
        if math.isnan(start_time) or math.isnan(end_time):
            print(f"Warning: {self.name} has NaN start or end time")
            return 0

        if start_time > end_time:
            print(f"Warning: {self.name} has start time after end time")
            return 0

        return end_time - start_time


class TimingMetricsConfig:
    """Configuration class that defines all available timing metrics."""

    def __init__(self):
        self.metrics = [
            TimingMetric(
                name='disagg_preprocessing',
                display_name='Disagg Preprocessing',
                color='#B8B8B8',  # Light gray
                description=
                'Time duration from the disagg server receives the request to a context server receives it',
                start_field='disagg_server_arrival_time',
                end_field='ctx_server_arrival_time',
                server_type='disagg'),
            TimingMetric(
                name='ctx_preprocessing',
                display_name='Context Preprocessing',
                color='#90EE90',  # Light green
                description=
                'Time duration from a context server receives the request to a LLM worker queues it',
                start_field='ctx_server_arrival_time',
                end_field='ctx_arrival_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_queue',
                display_name='Context Queue',
                color='#FFB347',  # Light orange
                description=
                'Time duration from the request is queued to first scheduled',
                start_field='ctx_arrival_time',
                end_field='ctx_first_scheduled_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_processing',
                display_name='Context Processing',
                color='#6495ED',  # Cornflower blue
                description=
                'Time duration from first scheduled to first token generated on a LLM worker',
                start_field='ctx_first_scheduled_time',
                end_field='ctx_first_token_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_postprocessing',
                display_name='Context Postprocessing',
                color='#DDA0DD',  # Plum
                description=
                'Time duration from first token generated on a LLM worker to the first token response sent by the context server',
                start_field='ctx_first_token_time',
                end_field='ctx_server_first_token_time',
                server_type='ctx'),
            TimingMetric(
                name='gen_preprocessing',
                display_name='Generation Preprocessing',
                color='#FFE66D',  # Bright yellow
                description=
                'Time duration from a generation server receives the request to a LLM worker receives it',
                start_field='gen_server_arrival_time',
                end_field='gen_arrival_time',
                server_type='gen'),
            TimingMetric(
                name='gen_queue',
                display_name='Generation Queue',
                color='#FF6B6B',  # Coral red
                description=
                'Time duration from the request is queued to first scheduled',
                start_field='gen_arrival_time',
                end_field='gen_first_scheduled_time',
                server_type='gen'),
            TimingMetric(
                name='gen_postprocessing',
                display_name='Generation Postprocessing',
                color='#95E1D3',  # Mint/teal
                description=
                'Time duration from first scheduled to the first token response sent by the generation server',
                start_field='gen_first_scheduled_time',
                end_field='gen_server_first_token_time',
                server_type='gen'),
            TimingMetric(
                name='disagg_postprocessing',
                display_name='Disagg Postprocessing',
                color='#A9A9A9',  # Dark gray
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


class RequestTimeBreakdown:
    """Main class for analyzing request time breakdown."""

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
                              output_file: Optional[str] = None):
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
                        f'<b>Request %{{x}}</b><br>{metric.display_name}: %{{y:.2f}} ms<extra></extra>'
                    ))

        # Update layout
        fig.update_layout(
            barmode='stack',
            title={
                'text':
                'Request Time Breakdown<br><sub>Time Spent in Each Segment (Interactive)</sub>',
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
            output_file = 'time_breakdown.html'
        elif not output_file.endswith('.html'):
            output_file += '.html'

        # Generate the plotly div
        plot_div = pyo.plot(fig,
                            output_type='div',
                            include_plotlyjs='cdn',
                            auto_open=False)

        # Generate descriptions HTML
        descriptions_html = self._generate_descriptions_html(timing_data)

        # Combine into full HTML
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Request Timing Breakdown</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .descriptions {{
            margin-top: 30px;
            padding: 20px;
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            max-width: 1200px;
        }}
        .descriptions h2 {{
            margin-top: 0;
            color: #333;
        }}
        .metric-desc {{
            margin-bottom: 15px;
            line-height: 1.6;
        }}
        .metric-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    {plot_div}
    {descriptions_html}
</body>
</html>
"""

        # Write to file
        with open(output_file, 'w') as f:
            f.write(full_html)

        print(f"Interactive time breakdown diagram saved to: {output_file}")
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

    def _generate_descriptions_html(self, timing_data: List[Dict]) -> str:
        """Generate HTML for metric descriptions section."""
        desc_items = []

        for metric in self.config.metrics:
            times = [
                data.get(f'{metric.name}_time', 0) * 1000
                for data in timing_data
            ]
            # Only include metrics that have non-zero data
            if any(t > 0 for t in times):
                desc_items.append(
                    f'<div class="metric-desc">'
                    f'<span class="metric-name">{metric.display_name}:</span> '
                    f'{metric.description}'
                    f'</div>')

        if not desc_items:
            return ''

        descriptions_html = f"""
    <div class="descriptions">
        <h2>Metric Descriptions</h2>
        {''.join(desc_items)}
        Reference: https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/serve/scripts/time_breakdown/README.md
    </div>
"""
        return descriptions_html

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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize TensorRT-LLM server time breakdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze performance metrics and create timing diagram
  python time_breakdown.py perf_metrics.json

  # Specify custom output file
  python time_breakdown.py perf_metrics.json -o my_timing.html

  # Show statistics only (no diagram)
  python time_breakdown.py perf_metrics.json --stats-only

  # Create diagram and show statistics
  python time_breakdown.py perf_metrics.json --show-stats
        """)

    parser.add_argument('json_file',
                        type=str,
                        help='Path to the JSON performance metrics file')

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default=None,
        help='Output HTML file path (default: time_breakdown.html)')

    parser.add_argument('--stats-only',
                        action='store_true',
                        help='Show statistics only without creating diagram')

    parser.add_argument('--show-stats',
                        action='store_true',
                        help='Show statistics in addition to creating diagram')

    args = parser.parse_args()

    # Create analyzer
    analyzer = RequestTimeBreakdown()

    # Parse the JSON file
    print(f"Parsing timing data from: {args.json_file}")
    timing_data = analyzer.parse_json_file(args.json_file)

    if not timing_data:
        print("No timing data found in the file.")
        sys.exit(1)

    # Show statistics if requested
    if args.stats_only or args.show_stats:
        analyzer.show_statistics(timing_data)

    # Create diagram unless stats-only mode
    if not args.stats_only:
        analyzer.create_timing_diagram(timing_data, args.output)


if __name__ == '__main__':
    main()
