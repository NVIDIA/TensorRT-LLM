#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Time Breakdown Analysis Tool

This module provides tools for analyzing and visualizing request time breakdown
from TensorRT-LLM server performance metrics. It can be used both as a library
and as a standalone CLI tool.

Features:
- Per-step generation metrics visualization
- CPU and GPU timeline separation
- Scrollable horizontal timeline view
- Hover to show individual segment details

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


@dataclass
class TimingMetric:
    """Configuration for a timing metric segment."""
    name: str
    display_name: str
    color: str
    description: str
    start_field: str
    end_field: str
    server_type: Optional[str] = None

    def calculate_duration(self, timing_data: Dict[str, float]) -> float:
        """Calculate the duration for this metric from timing data."""
        start_time = timing_data.get(self.start_field)
        end_time = timing_data.get(self.end_field)

        if start_time is None or end_time is None:
            return 0
        if math.isnan(start_time) or math.isnan(end_time):
            return 0
        if start_time > end_time:
            return 0
        return end_time - start_time


class TimingMetricsConfig:
    """Configuration class that defines all available timing metrics."""

    def __init__(self):
        self.metrics = [
            TimingMetric(
                name='disagg_preprocessing',
                display_name='Disagg Preprocessing',
                color='#B8B8B8',
                description=
                'Disagg orchestrator overhead: request routing and load balancing from disagg server to context server (includes network transfer)',
                start_field='disagg_server_arrival_time',
                end_field='ctx_server_arrival_time',
                server_type='disagg'),
            TimingMetric(
                name='ctx_preprocessing',
                display_name='Context Preprocessing',
                color='#90EE90',
                description=
                'Context server overhead: tokenization, request validation, and IPC from OpenAI server to LLM executor',
                start_field='ctx_server_arrival_time',
                end_field='ctx_arrival_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_queue',
                display_name='Context Queue',
                color='#FFB347',
                description=
                'Scheduler queue wait: time request waits in executor queue before being scheduled for context/prefill processing',
                start_field='ctx_arrival_time',
                end_field='ctx_first_scheduled_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_processing',
                display_name='Context Processing',
                color='#6495ED',
                description=
                'Context/Prefill execution: GPU forward pass for input tokens, KV cache population, and first token generation',
                start_field='ctx_first_scheduled_time',
                end_field='ctx_first_token_time',
                server_type='ctx'),
            TimingMetric(
                name='ctx_postprocessing',
                display_name='Context Postprocessing',
                color='#DDA0DD',
                description=
                'First token delivery: detokenization, streaming response creation, and IPC from LLM executor to OpenAI server',
                start_field='ctx_first_token_time',
                end_field='ctx_server_first_token_time',
                server_type='ctx'),
            TimingMetric(
                name='disagg_relay',
                display_name='Disagg Relay',
                color='#000000',
                description=
                'Context-to-Generation handoff: network transfer from context server to generation server via disagg orchestrator',
                start_field='ctx_server_first_token_time',
                end_field='gen_server_arrival_time',
                server_type='disagg'),
            TimingMetric(
                name='gen_preprocessing',
                display_name='Generation Preprocessing',
                color='#FFE66D',
                description=
                'Generation server overhead: request parsing and IPC from OpenAI server to LLM executor on generation server',
                start_field='gen_server_arrival_time',
                end_field='gen_arrival_time',
                server_type='gen'),
            TimingMetric(
                name='gen_queue_wait',
                display_name='Generation Queue Wait',
                color='#FF6B6B',
                description=
                'Generation queue wait: time request waits in generation executor queue before KV cache transfer begins',
                start_field='gen_arrival_time',
                end_field='gen_kv_cache_transfer_start',
                server_type='gen'),
            TimingMetric(
                name='gen_kv_transfer',
                display_name='Generation KV Transfer',
                color='#4ECDC4',
                description=
                'KV cache transfer: GPU-to-GPU transfer of cached key-value tensors from context server to generation server',
                start_field='gen_kv_cache_transfer_start',
                end_field='gen_kv_cache_transfer_end',
                server_type='gen'),
            TimingMetric(
                name='gen_post_transfer',
                display_name='Generation Post Transfer',
                color='#FF9F43',
                description=
                'Post-transfer wait: time from KV transfer completion to request being scheduled for generation',
                start_field='gen_kv_cache_transfer_end',
                end_field='gen_first_scheduled_time',
                server_type='gen'),
            TimingMetric(
                name='gen_postprocessing',
                display_name='Generation Postprocessing',
                color='#95E1D3',
                description=
                'Generation execution: all decode steps including forward, sampling, detokenization, and streaming responses',
                start_field='gen_first_scheduled_time',
                end_field='gen_server_first_token_time',
                server_type='gen'),
            TimingMetric(
                name='disagg_postprocessing',
                display_name='Disagg Postprocessing',
                color='#A9A9A9',
                description=
                'Disagg response relay: network transfer and aggregation of responses from generation server through disagg orchestrator to client',
                start_field='gen_server_first_token_time',
                end_field='disagg_server_first_token_time',
                server_type='disagg')
        ]

    def get_metric_by_name(self, name: str) -> Optional[TimingMetric]:
        return next((m for m in self.metrics if m.name == name), None)


class RequestDataParser:
    """Parser for disaggregated format with ctx_perf_metrics and gen_perf_metrics."""

    def parse_request(self, request_data: Dict,
                      request_index: int) -> Dict[str, Any]:
        # Check if both ctx_perf_metrics and gen_perf_metrics exist and are not None
        ctx_perf = request_data.get('ctx_perf_metrics')
        gen_perf = request_data.get('gen_perf_metrics')
        is_disaggregated = ctx_perf is not None and gen_perf is not None

        ctx_metrics = {}
        gen_metrics = {}
        if is_disaggregated:
            ctx_perf = ctx_perf or {}
            gen_perf = gen_perf or {}
            ctx_perf_metrics = ctx_perf.get('perf_metrics') or {}
            gen_perf_metrics = gen_perf.get('perf_metrics') or {}
            ctx_metrics = ctx_perf_metrics.get('timing_metrics') or {}
            gen_metrics = gen_perf_metrics.get('timing_metrics') or {}
        else:
            perf_metrics = request_data.get('perf_metrics') or {}
            ctx_metrics = perf_metrics.get('timing_metrics') or {}

        # Context timing
        ctx_arrival_time = ctx_metrics.get('arrival_time', float('nan'))
        ctx_first_scheduled_time = ctx_metrics.get('first_scheduled_time',
                                                   float('nan'))
        ctx_first_token_time = ctx_metrics.get('first_token_time', float('nan'))
        ctx_server_arrival_time = ctx_metrics.get('server_arrival_time',
                                                  float('nan'))
        ctx_server_first_token_time = ctx_metrics.get('server_first_token_time',
                                                      float('nan'))

        # Generation timing
        gen_server_first_token_time = gen_metrics.get('server_first_token_time',
                                                      float('nan'))
        gen_server_arrival_time = gen_metrics.get('server_arrival_time',
                                                  float('nan'))
        gen_arrival_time = gen_metrics.get('arrival_time', float('nan'))
        gen_first_token_time = gen_metrics.get('first_token_time', float('nan'))
        gen_first_scheduled_time = gen_metrics.get('first_scheduled_time',
                                                   float('nan'))
        gen_kv_cache_transfer_start = gen_metrics.get('kv_cache_transfer_start',
                                                      float('nan'))
        gen_kv_cache_transfer_end = gen_metrics.get('kv_cache_transfer_end',
                                                    float('nan'))

        # Disagg timing
        disagg_server_arrival_time = float('nan')
        disagg_server_first_token_time = float('nan')
        if is_disaggregated:
            disagg_server_arrival_time = request_data.get(
                'disagg_server_arrival_time', float('nan'))
            disagg_server_first_token_time = request_data.get(
                'disagg_server_first_token_time', float('nan'))

        # Request ID
        if is_disaggregated:
            request_id = (ctx_perf or {}).get('request_id', request_index)
        else:
            request_id = request_data.get('request_id', request_index)

        # Time breakdown metrics - check new unified structure first, then fall back to legacy
        step_metrics = None
        ctx_gpu_forward_time = None
        ctx_gpu_sample_time = None
        ctx_chunk_metrics = None

        # Try new unified time_breakdown_metrics structure
        if is_disaggregated:
            # time_breakdown_metrics is at gen_perf_metrics top level, not inside perf_metrics
            time_breakdown = (gen_perf or {}).get('time_breakdown_metrics')
            if time_breakdown:
                step_metrics = time_breakdown.get('step_metrics')
                ctx_gpu_forward_time = time_breakdown.get(
                    'ctx_gpu_forward_time')
                ctx_gpu_sample_time = time_breakdown.get('ctx_gpu_sample_time')
                ctx_chunk_metrics = time_breakdown.get('ctx_chunk_metrics')
            else:
                # Legacy: step_metrics inside perf_metrics
                gen_perf_data = (gen_perf or {}).get('perf_metrics') or {}
                step_metrics = gen_perf_data.get('step_metrics')
            # ctx GPU timing / chunk metrics from ctx_perf
            if ctx_gpu_forward_time is None:
                ctx_time_breakdown = (ctx_perf
                                      or {}).get('time_breakdown_metrics')
                if ctx_time_breakdown:
                    ctx_gpu_forward_time = ctx_time_breakdown.get(
                        'ctx_gpu_forward_time')
                    ctx_gpu_sample_time = ctx_time_breakdown.get(
                        'ctx_gpu_sample_time')
                    if not ctx_chunk_metrics:
                        ctx_chunk_metrics = ctx_time_breakdown.get(
                            'ctx_chunk_metrics')
                else:
                    ctx_gpu_forward_time = (ctx_perf
                                            or {}).get('ctx_gpu_forward_time')
                    ctx_gpu_sample_time = (ctx_perf
                                           or {}).get('ctx_gpu_sample_time')
        else:
            # Try time_breakdown_metrics at top level first (new structure)
            time_breakdown = request_data.get('time_breakdown_metrics')
            if time_breakdown:
                step_metrics = time_breakdown.get('step_metrics')
                ctx_gpu_forward_time = time_breakdown.get(
                    'ctx_gpu_forward_time')
                ctx_gpu_sample_time = time_breakdown.get('ctx_gpu_sample_time')
                ctx_chunk_metrics = time_breakdown.get('ctx_chunk_metrics')
            else:
                # Fall back to perf_metrics (legacy structure)
                perf_data = request_data.get('perf_metrics') or {}
                time_breakdown = perf_data.get('time_breakdown_metrics')
                if time_breakdown:
                    step_metrics = time_breakdown.get('step_metrics')
                    ctx_gpu_forward_time = time_breakdown.get(
                        'ctx_gpu_forward_time')
                    ctx_gpu_sample_time = time_breakdown.get(
                        'ctx_gpu_sample_time')
                    ctx_chunk_metrics = time_breakdown.get('ctx_chunk_metrics')
                else:
                    step_metrics = perf_data.get('step_metrics')
                    ctx_gpu_forward_time = perf_data.get('ctx_gpu_forward_time')
                    ctx_gpu_sample_time = perf_data.get('ctx_gpu_sample_time')

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
            'gen_kv_cache_transfer_start': gen_kv_cache_transfer_start,
            'gen_kv_cache_transfer_end': gen_kv_cache_transfer_end,
            'disagg_server_arrival_time': disagg_server_arrival_time,
            'disagg_server_first_token_time': disagg_server_first_token_time,
            'step_metrics': step_metrics,
            'ctx_chunk_metrics': ctx_chunk_metrics,
            'ctx_gpu_forward_time': ctx_gpu_forward_time,
            'ctx_gpu_sample_time': ctx_gpu_sample_time,
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

        timing_data = []
        for i, request in enumerate(data):
            parsed_data = self.parser.parse_request(request, i)

            # Calculate durations for each metric
            for metric in self.config.metrics:
                duration = metric.calculate_duration(parsed_data)
                parsed_data[f'{metric.name}_time'] = duration

            timing_data.append(parsed_data)

        if timing_data:
            has_gen_metrics = any(not math.isnan(
                entry.get('gen_server_arrival_time', float('nan')))
                                  for entry in timing_data)
            format_type = "disaggregated" if has_gen_metrics else "aggregated"
            print(f"Parsed {len(timing_data)} requests ({format_type} format)")

        return timing_data

    def create_timing_diagram(self,
                              timing_data: List[Dict],
                              output_file: Optional[str] = None,
                              max_requests: Optional[int] = None,
                              sort_by: str = 'arrival'):
        """Create an interactive HTML timeline chart showing time breakdown with CPU/GPU separation.

        Each request's timeline is aligned to t=0 at the start of disagg_preprocessing (or ctx_server_arrival).
        GPU timeline is offset to align with actual GPU execution start time within the request.
        All steps use unified colors for easy comparison.

        Args:
            timing_data: List of parsed timing data for each request.
            output_file: Output HTML file path.
            max_requests: Maximum number of requests to display. None means no limit.
            sort_by: Sort order for requests. Options:
                - 'arrival': Sort by arrival time (earliest first, default)
                - 'e2e': Sort by end-to-end latency (longest first)
        """
        if not timing_data:
            print("No timing data to visualize.")
            return

        # Calculate e2e time for each request
        def get_e2e_time(data):
            """Calculate end-to-end time from arrival to last token."""
            # Get arrival time
            arrival = data.get('ctx_server_arrival_time', float('nan'))
            if math.isnan(arrival):
                arrival = data.get('ctx_arrival_time', float('nan'))
            if math.isnan(arrival):
                arrival = data.get('disagg_server_arrival_time', float('nan'))

            # Get last token time from step_metrics
            step_metrics = data.get('step_metrics', []) or []
            if step_metrics:
                last_step = step_metrics[-1]
                last_token = last_step.get('token_time', 0)
                if last_token and not math.isnan(arrival):
                    return (last_token - arrival) * 1000  # Convert to ms
            return 0

        def get_arrival_time(data):
            arrival = data.get('ctx_server_arrival_time', float('nan'))
            if math.isnan(arrival):
                arrival = data.get('ctx_arrival_time', float('nan'))
            if math.isnan(arrival):
                arrival = data.get('disagg_server_arrival_time', float('inf'))
            return arrival

        # Sort based on sort_by parameter
        if sort_by == 'e2e':
            # Sort by e2e time, longest first
            timing_data = sorted(timing_data, key=get_e2e_time, reverse=True)
            print(
                f"Sorted {len(timing_data)} requests by E2E latency (longest first)"
            )
        else:
            # Default: sort by arrival time (earliest first)
            timing_data = sorted(timing_data, key=get_arrival_time)

        # Limit number of requests if specified
        if max_requests is not None and max_requests > 0 and len(
                timing_data) > max_requests:
            original_count = len(timing_data)
            timing_data = timing_data[:max_requests]
            print(
                f"Limiting display to {max_requests} requests (out of {original_count} total)"
            )

        # Calculate total data points to determine if pagination is needed
        PAGINATION_THRESHOLD = 10000
        total_steps = sum(
            len(data.get('step_metrics', []) or []) for data in timing_data)
        num_requests = len(timing_data)
        total_data_points = total_steps + num_requests  # steps + context phases

        needs_pagination = total_data_points > PAGINATION_THRESHOLD

        if needs_pagination:
            # Calculate requests per page to stay under threshold
            avg_steps_per_request = total_steps / num_requests if num_requests > 0 else 1
            requests_per_page = max(
                1, int(PAGINATION_THRESHOLD / (avg_steps_per_request + 1)))
            # Round to nice numbers (multiples of 10 or 50)
            if requests_per_page >= 50:
                requests_per_page = (requests_per_page // 50) * 50
            elif requests_per_page >= 10:
                requests_per_page = (requests_per_page // 10) * 10

            num_pages = (num_requests + requests_per_page -
                         1) // requests_per_page
            print(
                f"Data too large ({total_data_points} points > {PAGINATION_THRESHOLD}). "
                f"Creating paginated view: {num_pages} pages, {requests_per_page} requests per page."
            )
        else:
            requests_per_page = num_requests
            num_pages = 1

        # For pagination, we'll render the first page and let JavaScript handle page switching
        self._create_paginated_diagram(timing_data, output_file,
                                       requests_per_page, num_pages,
                                       needs_pagination)

    def _create_paginated_diagram(self, timing_data: List[Dict],
                                  output_file: Optional[str],
                                  requests_per_page: int, num_pages: int,
                                  needs_pagination: bool):
        """Internal method to create the diagram with optional pagination support."""

        fig = go.Figure()

        # Unified colors for all steps (same color regardless of step number)
        # Overlap mode CPU timeline colors (showing batch N processing):
        STEP_CPU_PREPROC_COLOR = '#DDA0DD'  # Plum - Preprocessing (first_token, speculative, etc)
        STEP_CPU_FWD_COLOR = '#87CEEB'  # Light blue - Forward(N) call
        STEP_CPU_OVERLAP_COLOR = '#FFA500'  # Orange - Update(N-1) + SendKV(N-1)
        STEP_CPU_SMP_COLOR = '#98FB98'  # Pale green - Sample(N) call
        STEP_CPU_POST_COLOR = '#FFB6C1'  # Light pink - Postprocessing (_handle_responses)
        # GPU timeline colors:
        STEP_GPU_FWD_COLOR = '#4169E1'  # Royal blue - GPU forward
        STEP_GPU_SMP_COLOR = '#32CD32'  # Lime green - GPU sample
        CTX_GPU_FWD_COLOR = '#1E90FF'  # Dodger blue - Context GPU forward
        CTX_GPU_SMP_COLOR = '#228B22'  # Forest green - Context GPU sample
        # Postprocessing colors:
        POST_GEN_COLOR = '#2E8B57'  # Sea green
        POST_DISAGG_COLOR = '#8B4513'  # Saddle brown

        # Batch data collectors for each trace type (to reduce HTML size)
        # Each collector stores: y, x, base, text, step, duration
        # Note: Plotly's %{x} shows base+x (end position) not x (width),
        # so we store duration separately for accurate hover display
        batch_data = {
            'step_preproc': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'step_cpu_fwd': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'step_overlap': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': [],
                'handled_step': []
            },
            'step_cpu_smp': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'step_cpu_post': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': [],
                'handled_step': []
            },
            'step_cpu_proc': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'ctx_gpu_fwd': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'ctx_gpu_smp': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'step_gpu_fwd': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'step_gpu_smp': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'gen_postproc': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'disagg_postproc': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
            'ctx_postproc': {
                'y': [],
                'x': [],
                'base': [],
                'text': [],
                'step': [],
                'duration': []
            },
        }

        # Standard metrics collectors (keyed by metric name)
        std_metric_data = {}

        # Build traces
        traces = []
        y_labels = []
        max_timeline = 0

        for req_idx, data in enumerate(timing_data):
            request_id = data['request_index']

            # Determine the reference time (t=0) for this request
            if not math.isnan(
                    data.get('disagg_server_arrival_time', float('nan'))):
                ref_time = data['disagg_server_arrival_time']
            elif not math.isnan(
                    data.get('ctx_server_arrival_time', float('nan'))):
                ref_time = data['ctx_server_arrival_time']
            else:
                ref_time = data.get('ctx_arrival_time', 0)

            cpu_label = f"Req {request_id} (CPU)"
            gpu_label = f"Req {request_id} (GPU)"
            y_labels.extend([cpu_label, gpu_label])

            # ============ CPU Timeline (using actual timestamps) ============
            # All CPU metrics use actual timestamps for accurate alignment

            # Track positions for alignment
            cpu_pos = 0  # Track max position for timeline width calculation
            cpu_ctx_processing_start = 0
            cpu_step_start = 0

            # Add standard CPU metrics (excluding postprocessing which goes on separate row)
            postprocessing_metrics = {
                'gen_postprocessing', 'disagg_postprocessing'
            }
            step_metrics = data.get('step_metrics', [])
            ctx_chunk_metrics_cpu = data.get('ctx_chunk_metrics') or []
            has_step_metrics = bool(step_metrics)
            has_ctx_chunks = len(
                ctx_chunk_metrics_cpu
            ) >= 1  # Has ctx chunk data (single or multiple)

            # Check if this is disagg mode (has gen_server_arrival_time)
            is_disagg = not math.isnan(
                data.get('gen_server_arrival_time', float('nan')))

            # Determine if first step_metric serves as context (non-chunked, non-disagg, no explicit ctx GPU)
            ctx_gpu_fwd_val = data.get('ctx_gpu_forward_time', 0) or 0
            used_first_step_as_ctx = (ctx_gpu_fwd_val == 0
                                      and not has_ctx_chunks and not is_disagg
                                      and has_step_metrics)

            # Short labels for bar text display
            short_labels = {
                'disagg_preprocessing': 'Disagg Pre',
                'ctx_preprocessing': 'Ctx Pre',
                'ctx_queue': 'Ctx Queue',
                'ctx_processing': 'Ctx Proc',
                'ctx_postprocessing': 'Ctx Post',
                'disagg_relay': 'Disagg Relay',
                'gen_preprocessing': 'Gen Pre',
                'gen_queue_wait': 'Gen Queue',
                'gen_kv_transfer': 'Gen KV',
                'gen_post_transfer': 'Gen Post',
                'gen_postprocessing': 'Gen Postproc',
                'disagg_postprocessing': 'Disagg Post',
            }

            for metric in self.config.metrics:
                # Skip metrics drawn separately: postprocessing as overlay
                if metric.name in postprocessing_metrics:
                    continue
                # Skip ctx_processing when per-chunk or per-step detail is available
                # In disagg mode, ctx_processing shows context server work (keep unless chunked)
                if metric.name == 'ctx_processing' and (has_ctx_chunks or
                                                        (has_step_metrics
                                                         and not is_disagg)):
                    continue
                # In disagg mode, ctx_postprocessing is part of sequential timeline (after ctx_processing)
                # In non-disagg mode, it's drawn as overlay since it overlaps with generation
                if metric.name == 'ctx_postprocessing' and not is_disagg:
                    continue

                # Use actual timestamps for positioning
                start_time = data.get(metric.start_field)
                end_time = data.get(metric.end_field)

                # Disagg ctx_postprocessing: extend start to last chunk's sample_end
                # to cover executor-side postprocessing (GPU sync, KV send, handle_responses)
                if metric.name == 'ctx_postprocessing' and has_ctx_chunks and ctx_chunk_metrics_cpu:
                    last_c = ctx_chunk_metrics_cpu[-1]
                    chunk_smp_end = last_c.get('sample_end_time')
                    if chunk_smp_end and chunk_smp_end > 0 and (
                            not start_time or chunk_smp_end < start_time):
                        start_time = chunk_smp_end

                # Skip if timestamps are invalid (None, NaN, or not positive)
                if start_time is None or end_time is None:
                    continue
                if isinstance(start_time, float) and math.isnan(start_time):
                    continue
                if isinstance(end_time, float) and math.isnan(end_time):
                    continue
                if start_time <= 0 or end_time <= 0:
                    continue
                if start_time >= end_time:
                    continue

                # Calculate position and duration using actual timestamps
                start_pos = (start_time - ref_time) * 1000
                end_pos = (end_time - ref_time) * 1000
                duration_ms = end_pos - start_pos

                if duration_ms > 0:
                    # Track context processing start position
                    if metric.name == 'ctx_processing':
                        cpu_ctx_processing_start = start_pos

                    text_label = short_labels.get(metric.name,
                                                  '') if duration_ms > 2 else ''

                    # Collect data for batch processing
                    if metric.name not in std_metric_data:
                        std_metric_data[metric.name] = {
                            'y': [],
                            'x': [],
                            'base': [],
                            'duration': [],
                            'text': [],
                            'display_name': metric.display_name,
                            'color': metric.color,
                            'description': metric.description
                        }
                    std_metric_data[metric.name]['y'].append(cpu_label)
                    std_metric_data[metric.name]['x'].append(
                        round(duration_ms, 2))
                    std_metric_data[metric.name]['base'].append(
                        round(start_pos, 2))
                    std_metric_data[metric.name]['duration'].append(
                        round(duration_ms, 2))
                    std_metric_data[metric.name]['text'].append(text_label)

                    # Track end position for step_start and max timeline calculation
                    cpu_step_start = max(cpu_step_start, end_pos)
                    cpu_pos = max(cpu_pos, end_pos)

            # Add per-chunk CPU metrics for chunked prefill context
            # Initialize prev_chunk_end_time to first_scheduled_time so C1 Pre is drawn
            ctx_first_scheduled_ts = data.get('ctx_first_scheduled_time')
            prev_chunk_end_time = ctx_first_scheduled_ts if (
                ctx_first_scheduled_ts
                and not math.isnan(ctx_first_scheduled_ts)
                and ctx_first_scheduled_ts > 0) else None
            prev_chunk_token_time = None
            if has_ctx_chunks:
                for chunk_idx, chunk in enumerate(ctx_chunk_metrics_cpu):
                    chunk_fwd_start = chunk.get('forward_start_time', 0)
                    chunk_fwd_end = chunk.get('forward_end_time', 0)
                    chunk_smp_start = chunk.get('sample_start_time', 0)
                    chunk_smp_end = chunk.get('sample_end_time', 0)
                    chunk_num = chunk_idx + 1
                    num_chunks = len(ctx_chunk_metrics_cpu)
                    chunk_prefix = f"Ctx C{chunk_num}" if num_chunks > 1 else "Ctx"

                    if not chunk_fwd_start or math.isnan(ref_time):
                        continue

                    fwd_start_pos = (chunk_fwd_start - ref_time) * 1000

                    # Inter-chunk gap: split into postprocessing + scheduling
                    prev_chunk_label = f"Ctx C{chunk_idx}" if num_chunks > 1 else "Ctx"
                    if prev_chunk_end_time is not None:
                        prev_end_pos = (prev_chunk_end_time - ref_time) * 1000
                        prev_token_time = prev_chunk_token_time
                        if prev_token_time and prev_token_time > prev_chunk_end_time:
                            # Postprocessing: sample_end → token_time
                            post_end_pos = (prev_token_time - ref_time) * 1000
                            post_ms = post_end_pos - prev_end_pos
                            if post_ms > 0.05:
                                text_label = f"{prev_chunk_label} Post" if post_ms > 2 else ""
                                batch_data['step_cpu_post']['y'].append(
                                    cpu_label)
                                batch_data['step_cpu_post']['x'].append(
                                    round(post_ms, 2))
                                batch_data['step_cpu_post']['base'].append(
                                    round(prev_end_pos, 2))
                                batch_data['step_cpu_post']['text'].append(
                                    text_label)
                                batch_data['step_cpu_post']['step'].append(
                                    prev_chunk_label)
                                batch_data['step_cpu_post']['duration'].append(
                                    round(post_ms, 2))
                                batch_data['step_cpu_post'][
                                    'handled_step'].append(prev_chunk_label)

                            # Scheduling: token_time → next forward_start
                            sched_ms = fwd_start_pos - post_end_pos
                            if sched_ms > 0.05:
                                text_label = f"{chunk_prefix} Pre" if sched_ms > 2 else ""
                                batch_data['step_preproc']['y'].append(
                                    cpu_label)
                                batch_data['step_preproc']['x'].append(
                                    round(sched_ms, 2))
                                batch_data['step_preproc']['base'].append(
                                    round(post_end_pos, 2))
                                batch_data['step_preproc']['text'].append(
                                    text_label)
                                batch_data['step_preproc']['step'].append(
                                    chunk_prefix)
                                batch_data['step_preproc']['duration'].append(
                                    round(sched_ms, 2))
                        else:
                            # No token_time: show entire gap as preprocessing
                            gap_ms = fwd_start_pos - prev_end_pos
                            if gap_ms > 0.05:
                                text_label = f"{chunk_prefix} Pre" if gap_ms > 2 else ""
                                batch_data['step_preproc']['y'].append(
                                    cpu_label)
                                batch_data['step_preproc']['x'].append(
                                    round(gap_ms, 2))
                                batch_data['step_preproc']['base'].append(
                                    round(prev_end_pos, 2))
                                batch_data['step_preproc']['text'].append(
                                    text_label)
                                batch_data['step_preproc']['step'].append(
                                    chunk_prefix)
                                batch_data['step_preproc']['duration'].append(
                                    round(gap_ms, 2))

                    # Chunk Forward
                    if chunk_fwd_end:
                        fwd_end_pos = (chunk_fwd_end - ref_time) * 1000
                        chunk_fwd_ms = fwd_end_pos - fwd_start_pos
                        if chunk_fwd_ms > 0:
                            text_label = f"{chunk_prefix} Fwd" if chunk_fwd_ms > 2 else ""
                            batch_data['step_cpu_fwd']['y'].append(cpu_label)
                            batch_data['step_cpu_fwd']['x'].append(
                                round(chunk_fwd_ms, 2))
                            batch_data['step_cpu_fwd']['base'].append(
                                round(fwd_start_pos, 2))
                            batch_data['step_cpu_fwd']['text'].append(
                                text_label)
                            batch_data['step_cpu_fwd']['step'].append(
                                chunk_prefix)
                            batch_data['step_cpu_fwd']['duration'].append(
                                round(chunk_fwd_ms, 2))

                        # Chunk Update (forward_end → sample_start)
                        if chunk_smp_start:
                            smp_start_pos = (chunk_smp_start - ref_time) * 1000
                            upd_ms = smp_start_pos - fwd_end_pos
                            if upd_ms > 0.05:
                                text_label = f"{chunk_prefix} Upd" if upd_ms > 2 else ""
                                batch_data['step_overlap']['y'].append(
                                    cpu_label)
                                batch_data['step_overlap']['x'].append(
                                    round(upd_ms, 2))
                                batch_data['step_overlap']['base'].append(
                                    round(fwd_end_pos, 2))
                                batch_data['step_overlap']['text'].append(
                                    text_label)
                                batch_data['step_overlap']['step'].append(
                                    chunk_prefix)
                                batch_data['step_overlap']['duration'].append(
                                    round(upd_ms, 2))
                                batch_data['step_overlap'][
                                    'handled_step'].append(chunk_prefix)

                            # Chunk Sample
                            if chunk_smp_end:
                                smp_end_pos = (chunk_smp_end - ref_time) * 1000
                                smp_ms = smp_end_pos - smp_start_pos
                                if smp_ms > 0:
                                    text_label = f"{chunk_prefix} Smp" if smp_ms > 2 else ""
                                    batch_data['step_cpu_smp']['y'].append(
                                        cpu_label)
                                    batch_data['step_cpu_smp']['x'].append(
                                        round(smp_ms, 2))
                                    batch_data['step_cpu_smp']['base'].append(
                                        round(smp_start_pos, 2))
                                    batch_data['step_cpu_smp']['text'].append(
                                        text_label)
                                    batch_data['step_cpu_smp']['step'].append(
                                        chunk_prefix)
                                    batch_data['step_cpu_smp'][
                                        'duration'].append(round(smp_ms, 2))
                                prev_chunk_end_time = chunk_smp_end
                                prev_chunk_token_time = chunk.get(
                                    'token_time', 0) or None
                                cpu_pos = max(cpu_pos, smp_end_pos)
                            else:
                                prev_chunk_end_time = chunk_smp_start
                                prev_chunk_token_time = None
                                cpu_pos = max(cpu_pos, smp_start_pos)
                        else:
                            prev_chunk_end_time = chunk_fwd_end
                            prev_chunk_token_time = None
                            cpu_pos = max(cpu_pos, fwd_end_pos)
                    else:
                        cpu_pos = max(cpu_pos, fwd_start_pos)

                # Last chunk post: in disagg mode this is absorbed into ctx_postprocessing metric.
                # In IFB non-overlap mode, draw if token_time is before S1 fwd_start.
                if not is_disagg and prev_chunk_end_time and prev_chunk_token_time and prev_chunk_token_time > prev_chunk_end_time:
                    s1_fwd_start = step_metrics[0].get('forward_start_time',
                                                       0) if step_metrics else 0
                    if not s1_fwd_start or prev_chunk_token_time <= s1_fwd_start:
                        num_c = len(ctx_chunk_metrics_cpu)
                        post_label = f"Ctx C{num_c}" if num_c > 1 else "Ctx"
                        post_start_pos = (prev_chunk_end_time - ref_time) * 1000
                        post_end_pos = (prev_chunk_token_time - ref_time) * 1000
                        post_ms = post_end_pos - post_start_pos
                        if post_ms > 0.05:
                            text_label = f"{post_label} Post" if post_ms > 2 else ""
                            batch_data['step_cpu_post']['y'].append(cpu_label)
                            batch_data['step_cpu_post']['x'].append(
                                round(post_ms, 2))
                            batch_data['step_cpu_post']['base'].append(
                                round(post_start_pos, 2))
                            batch_data['step_cpu_post']['text'].append(
                                text_label)
                            batch_data['step_cpu_post']['step'].append(
                                post_label)
                            batch_data['step_cpu_post']['duration'].append(
                                round(post_ms, 2))
                            batch_data['step_cpu_post']['handled_step'].append(
                                post_label)
                            cpu_pos = max(cpu_pos, post_end_pos)

            # Add per-step CPU metrics (generation phase)
            # In overlap mode, each step is split into:
            #   1. Preprocessing (prev_token_time → forward_start_time) - schedule + prepare + handle_responses(N-2)
            #   2. Forward call (forward_start_time → forward_end_time)
            #   3. Update(N-1) (forward_end_time → sample_start_time) - sync + update previous batch
            #   4. Sample call (sample_start_time → sample_end_time)
            #   5. Postprocessing (sample_end → token_time) - handle_responses
            step_metrics = data.get('step_metrics', [])
            prev_step_token_time = None
            if step_metrics:
                for step_idx, step in enumerate(step_metrics):
                    forward_start_time = step.get('forward_start_time', 0)
                    forward_end_time = step.get('forward_end_time', 0)
                    sample_start_time = step.get('sample_start_time', 0)
                    token_time = step.get('token_time', 0)
                    # Legacy support: old data may have scheduled_time instead of forward_start_time
                    scheduled_time = step.get('scheduled_time', 0)

                    # Check if we have valid timing data
                    has_valid_timing = (
                        not math.isnan(ref_time)
                        and (forward_start_time or scheduled_time)
                        and (token_time or
                             (forward_end_time and sample_start_time)))
                    if has_valid_timing:
                        step_num = step_idx + 1
                        prev_step_num = step_idx  # N-1
                        # Determine label: "Ctx" only when first step IS context (non-chunked, non-disagg)
                        is_ctx_step = (step_num == 1 and used_first_step_as_ctx)
                        step_label = "Ctx" if is_ctx_step else f"S{step_num}"
                        prev_step_label = "Ctx" if (
                            prev_step_num == 1 and used_first_step_as_ctx
                        ) else (
                            f"S{prev_step_num}" if prev_step_num > 0 else "Ctx")

                        # Determine step start position
                        # Use forward_start_time if available, otherwise fall back to scheduled_time
                        fwd_start = forward_start_time or scheduled_time
                        fwd_start_pos = (fwd_start - ref_time) * 1000

                        # Check if we have detailed overlap timing
                        has_overlap_timing = forward_end_time and sample_start_time

                        if has_overlap_timing:
                            # 1. Preprocessing (N) - from previous step's token_time to forward_start
                            if forward_start_time:
                                if step_idx == 0:
                                    # First gen step: determine preprocessing start
                                    gen_first_scheduled = data.get(
                                        'gen_first_scheduled_time')
                                    ctx_first_scheduled = data.get(
                                        'ctx_first_scheduled_time')
                                    if is_disagg and gen_first_scheduled and not math.isnan(
                                            gen_first_scheduled):
                                        # Disagg mode: from gen_first_scheduled_time (gen server timeline)
                                        preproc_start = (gen_first_scheduled -
                                                         ref_time) * 1000
                                    elif has_ctx_chunks and ctx_chunk_metrics_cpu:
                                        # IFB: from last chunk's sample_end (not token_time,
                                        # which may be recorded after S1 fwd_start in overlap mode)
                                        last_c = ctx_chunk_metrics_cpu[-1]
                                        last_chunk_end = last_c.get(
                                            'sample_end_time') or last_c.get(
                                                'token_time')
                                        if last_chunk_end and last_chunk_end > 0:
                                            preproc_start = (last_chunk_end -
                                                             ref_time) * 1000
                                        elif ctx_first_scheduled and not math.isnan(
                                                ctx_first_scheduled):
                                            preproc_start = (ctx_first_scheduled
                                                             - ref_time) * 1000
                                        else:
                                            preproc_start = cpu_step_start
                                    elif ctx_first_scheduled and not math.isnan(
                                            ctx_first_scheduled):
                                        # IFB non-chunked: from ctx_first_scheduled_time
                                        preproc_start = (ctx_first_scheduled -
                                                         ref_time) * 1000
                                    else:
                                        preproc_start = cpu_step_start
                                elif prev_step_token_time:
                                    preproc_start = (prev_step_token_time -
                                                     ref_time) * 1000
                                else:
                                    preproc_start = None

                                if preproc_start is not None:
                                    preproc_ms = fwd_start_pos - preproc_start
                                    if preproc_ms > 0.05:
                                        text_label = f"{step_label} Pre" if preproc_ms > 2 else ""
                                        batch_data['step_preproc']['y'].append(
                                            cpu_label)
                                        batch_data['step_preproc']['x'].append(
                                            round(preproc_ms, 2))
                                        batch_data['step_preproc'][
                                            'base'].append(
                                                round(preproc_start, 2))
                                        batch_data['step_preproc'][
                                            'text'].append(text_label)
                                        batch_data['step_preproc'][
                                            'step'].append(step_label)
                                        batch_data['step_preproc'][
                                            'duration'].append(
                                                round(preproc_ms, 2))

                            # 2. Forward(N) call - from forward_start to forward_end
                            fwd_end = (forward_end_time - ref_time) * 1000
                            cpu_fwd_ms = fwd_end - fwd_start_pos
                            if cpu_fwd_ms > 0:
                                text_label = f"{step_label} Fwd" if cpu_fwd_ms > 2 else ""
                                batch_data['step_cpu_fwd']['y'].append(
                                    cpu_label)
                                batch_data['step_cpu_fwd']['x'].append(
                                    round(cpu_fwd_ms, 2))
                                batch_data['step_cpu_fwd']['base'].append(
                                    round(fwd_start_pos, 2))
                                batch_data['step_cpu_fwd']['text'].append(
                                    text_label)
                                batch_data['step_cpu_fwd']['step'].append(
                                    step_label)
                                batch_data['step_cpu_fwd']['duration'].append(
                                    round(cpu_fwd_ms, 2))

                            # 2. Update - overlap with GPU forward
                            smp_start = (sample_start_time - ref_time) * 1000
                            overlap_ms = smp_start - fwd_end
                            if overlap_ms > 0.05:
                                text_label = f"{step_label} Upd" if overlap_ms > 2 else ""
                                batch_data['step_overlap']['y'].append(
                                    cpu_label)
                                batch_data['step_overlap']['x'].append(
                                    round(overlap_ms, 2))
                                batch_data['step_overlap']['base'].append(
                                    round(fwd_end, 2))
                                batch_data['step_overlap']['text'].append(
                                    text_label)
                                batch_data['step_overlap']['step'].append(
                                    step_label)
                                batch_data['step_overlap']['duration'].append(
                                    round(overlap_ms, 2))
                                batch_data['step_overlap'][
                                    'handled_step'].append(prev_step_label)

                            # 3. Sample(N) call - from sample_start to sample_end
                            sample_end_time = step.get('sample_end_time', 0)
                            step_token_time = step.get('token_time', 0)
                            effective_token_time = step_token_time

                            if sample_end_time:
                                smp_end = (sample_end_time - ref_time) * 1000
                                cpu_smp_ms = smp_end - smp_start
                                if cpu_smp_ms > 0:
                                    text_label = f"{step_label} Smp" if cpu_smp_ms > 2 else ""
                                    batch_data['step_cpu_smp']['y'].append(
                                        cpu_label)
                                    batch_data['step_cpu_smp']['x'].append(
                                        round(cpu_smp_ms, 2))
                                    batch_data['step_cpu_smp']['base'].append(
                                        round(smp_start, 2))
                                    batch_data['step_cpu_smp']['text'].append(
                                        text_label)
                                    batch_data['step_cpu_smp']['step'].append(
                                        step_label)
                                    batch_data['step_cpu_smp'][
                                        'duration'].append(round(cpu_smp_ms, 2))

                                # 4. Postprocessing - from sample_end to token_time
                                if effective_token_time:
                                    token_end = (effective_token_time -
                                                 ref_time) * 1000
                                    cpu_post_ms = token_end - smp_end
                                    if cpu_post_ms > 0:
                                        text_label = f"{step_label} Post" if cpu_post_ms > 2 else ""
                                        batch_data['step_cpu_post']['y'].append(
                                            cpu_label)
                                        batch_data['step_cpu_post']['x'].append(
                                            round(cpu_post_ms, 2))
                                        batch_data['step_cpu_post'][
                                            'base'].append(round(smp_end, 2))
                                        batch_data['step_cpu_post'][
                                            'text'].append(text_label)
                                        batch_data['step_cpu_post'][
                                            'step'].append(step_label)
                                        batch_data['step_cpu_post'][
                                            'duration'].append(
                                                round(cpu_post_ms, 2))
                                        batch_data['step_cpu_post'][
                                            'handled_step'].append(step_label)
                                    cpu_pos = max(cpu_pos, token_end)
                                    prev_step_token_time = effective_token_time
                                else:
                                    cpu_pos = max(cpu_pos, smp_end)
                            else:
                                # Legacy: no sample_end_time, use token_time as sample end
                                if token_time:
                                    token_end = (token_time - ref_time) * 1000
                                    cpu_smp_ms = token_end - smp_start
                                    if cpu_smp_ms > 0:
                                        text_label = f"{step_label} Smp" if cpu_smp_ms > 2 else ""
                                        batch_data['step_cpu_smp']['y'].append(
                                            cpu_label)
                                        batch_data['step_cpu_smp']['x'].append(
                                            round(cpu_smp_ms, 2))
                                        batch_data['step_cpu_smp'][
                                            'base'].append(round(smp_start, 2))
                                        batch_data['step_cpu_smp'][
                                            'text'].append(text_label)
                                        batch_data['step_cpu_smp'][
                                            'step'].append(step_label)
                                        batch_data['step_cpu_smp'][
                                            'duration'].append(
                                                round(cpu_smp_ms, 2))
                                    cpu_pos = max(cpu_pos, token_end)
                                else:
                                    cpu_pos = max(cpu_pos, smp_start)
                        else:
                            # Non-overlap mode or old data: show single segment
                            # Use forward_start or scheduled_time as the start point
                            if token_time:
                                token_end = (token_time - ref_time) * 1000
                                cpu_proc_ms = token_end - fwd_start_pos
                                if cpu_proc_ms > 0:
                                    batch_data['step_cpu_proc']['y'].append(
                                        cpu_label)
                                    batch_data['step_cpu_proc']['x'].append(
                                        round(cpu_proc_ms, 2))
                                    batch_data['step_cpu_proc']['base'].append(
                                        round(fwd_start_pos, 2))
                                    batch_data['step_cpu_proc']['text'].append(
                                        "")
                                    batch_data['step_cpu_proc']['step'].append(
                                        step_label)
                                    batch_data['step_cpu_proc'][
                                        'duration'].append(round(
                                            cpu_proc_ms, 2))
                                    cpu_pos = max(cpu_pos, token_end)

            max_timeline = max(max_timeline, cpu_pos)

            # ============ GPU Timeline (aligned to CPU timeline) ============
            ctx_chunk_metrics = data.get('ctx_chunk_metrics') or []
            ctx_gpu_fwd = data.get('ctx_gpu_forward_time', 0) or 0
            ctx_gpu_smp = data.get('ctx_gpu_sample_time', 0) or 0
            has_chunk_detail = len(ctx_chunk_metrics) > 0

            # Non-disagg fallback: use first step's GPU times as context GPU
            if used_first_step_as_ctx and step_metrics:
                first_step = step_metrics[0]
                ctx_gpu_fwd = first_step.get('gpu_forward_time', 0) or 0
                ctx_gpu_smp = first_step.get('gpu_sample_time', 0) or 0

            # Align GPU timeline start with actual CPU processing start
            if has_chunk_detail:
                # Chunked prefill: align with first chunk's forward_start_time
                first_chunk_fwd_start = ctx_chunk_metrics[0].get(
                    'forward_start_time')
                if first_chunk_fwd_start and not math.isnan(ref_time):
                    gpu_pos = (first_chunk_fwd_start - ref_time) * 1000
                else:
                    gpu_pos = cpu_ctx_processing_start
            elif used_first_step_as_ctx and step_metrics:
                # Non-disagg non-chunked: align with first step's forward_start_time
                first_fwd_start = step_metrics[0].get('forward_start_time')
                if first_fwd_start and not math.isnan(ref_time):
                    gpu_pos = (first_fwd_start - ref_time) * 1000
                else:
                    gpu_pos = cpu_ctx_processing_start
            else:
                # Disagg non-chunked: align with ctx_first_scheduled_time
                ctx_first_scheduled = data.get('ctx_first_scheduled_time')
                if ctx_first_scheduled and not math.isnan(
                        ref_time) and not math.isnan(ctx_first_scheduled):
                    gpu_pos = (ctx_first_scheduled - ref_time) * 1000
                else:
                    gpu_pos = cpu_ctx_processing_start

            # Draw context GPU chunks (per-chunk or total)
            if has_chunk_detail:
                # Chunked prefill: draw each chunk separately
                for chunk_idx, chunk in enumerate(ctx_chunk_metrics):
                    chunk_fwd = chunk.get('gpu_forward_time', 0) or 0
                    chunk_smp = chunk.get('gpu_sample_time', 0) or 0
                    chunk_num = chunk_idx + 1
                    num_chunks = len(ctx_chunk_metrics)
                    if num_chunks > 1:
                        label_prefix = f"Ctx C{chunk_num}"
                    else:
                        label_prefix = "Ctx"

                    if chunk_fwd > 0:
                        text_label = f"{label_prefix} Fwd" if chunk_fwd > 2 else ""
                        batch_data['ctx_gpu_fwd']['y'].append(gpu_label)
                        batch_data['ctx_gpu_fwd']['x'].append(
                            round(chunk_fwd, 2))
                        batch_data['ctx_gpu_fwd']['base'].append(
                            round(gpu_pos, 2))
                        batch_data['ctx_gpu_fwd']['text'].append(text_label)
                        batch_data['ctx_gpu_fwd']['step'].append(label_prefix)
                        batch_data['ctx_gpu_fwd']['duration'].append(
                            round(chunk_fwd, 2))
                        gpu_pos += chunk_fwd

                    if chunk_smp > 0:
                        text_label = f"{label_prefix} Smp" if chunk_smp > 2 else ""
                        batch_data['ctx_gpu_smp']['y'].append(gpu_label)
                        batch_data['ctx_gpu_smp']['x'].append(
                            round(chunk_smp, 2))
                        batch_data['ctx_gpu_smp']['base'].append(
                            round(gpu_pos, 2))
                        batch_data['ctx_gpu_smp']['text'].append(text_label)
                        batch_data['ctx_gpu_smp']['step'].append(label_prefix)
                        batch_data['ctx_gpu_smp']['duration'].append(
                            round(chunk_smp, 2))
                        gpu_pos += chunk_smp
            else:
                # Non-chunked: draw single context GPU block
                if ctx_gpu_fwd > 0:
                    text_label = "Ctx Fwd" if ctx_gpu_fwd > 2 else ""
                    batch_data['ctx_gpu_fwd']['y'].append(gpu_label)
                    batch_data['ctx_gpu_fwd']['x'].append(round(ctx_gpu_fwd, 2))
                    batch_data['ctx_gpu_fwd']['base'].append(round(gpu_pos, 2))
                    batch_data['ctx_gpu_fwd']['text'].append(text_label)
                    batch_data['ctx_gpu_fwd']['step'].append("Ctx")
                    batch_data['ctx_gpu_fwd']['duration'].append(
                        round(ctx_gpu_fwd, 2))
                    gpu_pos += ctx_gpu_fwd

                if ctx_gpu_smp > 0:
                    text_label = "Ctx Smp" if ctx_gpu_smp > 2 else ""
                    batch_data['ctx_gpu_smp']['y'].append(gpu_label)
                    batch_data['ctx_gpu_smp']['x'].append(round(ctx_gpu_smp, 2))
                    batch_data['ctx_gpu_smp']['base'].append(round(gpu_pos, 2))
                    batch_data['ctx_gpu_smp']['text'].append(text_label)
                    batch_data['ctx_gpu_smp']['step'].append("Ctx")
                    batch_data['ctx_gpu_smp']['duration'].append(
                        round(ctx_gpu_smp, 2))
                    gpu_pos += ctx_gpu_smp

            # Reset gpu_pos to align generation GPU with CPU step1 forward
            # For disagg or chunked prefill, context GPU may end at a different time than gen GPU starts
            if not used_first_step_as_ctx:
                first_step = step_metrics[0] if step_metrics else {}
                first_fwd_start = first_step.get(
                    'forward_start_time') or first_step.get('scheduled_time')
                if first_fwd_start and not math.isnan(ref_time):
                    gpu_pos = (first_fwd_start - ref_time) * 1000
                else:
                    gpu_pos = cpu_step_start

            # Per-step GPU timing - sequential stacking
            if step_metrics:
                for step_idx, step in enumerate(step_metrics):
                    if step_idx == 0 and used_first_step_as_ctx:
                        continue  # Already shown as context GPU

                    gpu_fwd = step.get('gpu_forward_time', 0) or 0
                    gpu_smp = step.get('gpu_sample_time', 0) or 0
                    step_num = step_idx + 1
                    gpu_step_label = f"S{step_num}"
                    step_gpu_start = gpu_pos

                    if gpu_fwd > 0:
                        text_label = f"{gpu_step_label} Fwd" if gpu_fwd > 2 else ""
                        batch_data['step_gpu_fwd']['y'].append(gpu_label)
                        batch_data['step_gpu_fwd']['x'].append(round(
                            gpu_fwd, 2))
                        batch_data['step_gpu_fwd']['base'].append(
                            round(step_gpu_start, 2))
                        batch_data['step_gpu_fwd']['text'].append(text_label)
                        batch_data['step_gpu_fwd']['step'].append(
                            gpu_step_label)
                        batch_data['step_gpu_fwd']['duration'].append(
                            round(gpu_fwd, 2))
                        gpu_pos += gpu_fwd

                    if gpu_smp > 0:
                        text_label = f"{gpu_step_label} Smp" if gpu_smp > 2 else ""
                        batch_data['step_gpu_smp']['y'].append(gpu_label)
                        batch_data['step_gpu_smp']['x'].append(round(
                            gpu_smp, 2))
                        batch_data['step_gpu_smp']['base'].append(
                            round(gpu_pos, 2))
                        batch_data['step_gpu_smp']['text'].append(text_label)
                        batch_data['step_gpu_smp']['step'].append(
                            gpu_step_label)
                        batch_data['step_gpu_smp']['duration'].append(
                            round(gpu_smp, 2))
                        gpu_pos += gpu_smp

            # ============ Postprocessing (overlaid on CPU timeline as thin bar at top) ============
            # These run in parallel with generation steps
            # Get timestamps for postprocessing (use absolute times, convert to ms relative to ref_time)
            ctx_first_token = data.get('ctx_first_token_time', 0)
            ctx_server_first_token = data.get('ctx_server_first_token_time', 0)
            gen_first_scheduled = data.get('gen_first_scheduled_time', 0)
            gen_server_first_token = data.get('gen_server_first_token_time', 0)
            disagg_server_first_token = data.get(
                'disagg_server_first_token_time', 0)

            # Postprocessing overlays (detokenize + IPC)
            # In disagg mode, ctx_postproc is already drawn as standard metric, so skip overlay
            # In non-disagg mode, draw as overlay since it runs in parallel with generation
            if not is_disagg and ctx_first_token and ctx_server_first_token and not math.isnan(
                    ref_time):
                post_start = (ctx_first_token - ref_time) * 1000
                post_end = (ctx_server_first_token - ref_time) * 1000
                post_duration = post_end - post_start
                if post_duration > 0:
                    batch_data['ctx_postproc']['y'].append(cpu_label)
                    batch_data['ctx_postproc']['x'].append(
                        round(post_duration, 2))
                    batch_data['ctx_postproc']['base'].append(
                        round(post_start, 2))
                    batch_data['ctx_postproc']['text'].append("")
                    batch_data['ctx_postproc']['step'].append("Ctx Post")
                    batch_data['ctx_postproc']['duration'].append(
                        round(post_duration, 2))

            if gen_first_scheduled and gen_server_first_token and not math.isnan(
                    ref_time):
                post_start = (gen_first_scheduled - ref_time) * 1000
                post_end = (gen_server_first_token - ref_time) * 1000
                post_duration = post_end - post_start
                if post_duration > 0:
                    batch_data['gen_postproc']['y'].append(cpu_label)
                    batch_data['gen_postproc']['x'].append(
                        round(post_duration, 2))
                    batch_data['gen_postproc']['base'].append(
                        round(post_start, 2))
                    batch_data['gen_postproc']['text'].append("")
                    batch_data['gen_postproc']['step'].append("Gen Post")
                    batch_data['gen_postproc']['duration'].append(
                        round(post_duration, 2))
            if gen_server_first_token and disagg_server_first_token and not math.isnan(
                    ref_time):
                post_start = (gen_server_first_token - ref_time) * 1000
                post_end = (disagg_server_first_token - ref_time) * 1000
                post_duration = post_end - post_start

                if post_duration > 0:
                    batch_data['disagg_postproc']['y'].append(cpu_label)
                    batch_data['disagg_postproc']['x'].append(
                        round(post_duration, 2))
                    batch_data['disagg_postproc']['base'].append(
                        round(post_start, 2))
                    batch_data['disagg_postproc']['text'].append("")
                    batch_data['disagg_postproc']['step'].append("Disagg Post")
                    batch_data['disagg_postproc']['duration'].append(
                        round(post_duration, 2))

            max_timeline = max(max_timeline, gpu_pos)

        # Create batched traces from collected data
        # This significantly reduces HTML size by combining many bars into single traces
        # Use simplified hovertemplate with %{base} and %{x} to avoid storing hover strings

        # Standard metrics traces
        for metric_name, metric_data in std_metric_data.items():
            if metric_data['y']:
                fig.add_trace(
                    go.Bar(
                        y=metric_data['y'],
                        x=metric_data['x'],
                        base=metric_data['base'],
                        customdata=metric_data[
                            'duration'],  # Store actual duration for hover
                        orientation='h',
                        name=metric_data['display_name'],
                        marker_color=metric_data['color'],
                        legendgroup=metric_name,
                        showlegend=True,
                        text=metric_data.get('text'),
                        textposition='inside',
                        textfont=dict(size=9, color='black'),
                        hovertemplate=(
                            f"<b>{metric_data['display_name']}</b><br>"
                            "Duration: %{customdata:.2f} ms<br>"
                            "Start: %{base:.2f} ms<extra></extra>")))

        # Step metrics traces - define trace configs with hover templates
        # Use %{customdata[0]} for step number, %{customdata[1]} for duration, %{base} for start
        # For step_overlap and step_cpu_post: %{customdata[2]} shows the step being handled
        # Note: Plotly's %{x} shows base+x (end position) when base is set, not the actual x value
        trace_configs = [
            ('step_preproc', 'Preproc(N)', STEP_CPU_PREPROC_COLOR, 'black',
             None,
             "<b>%{customdata[0]}: Preprocessing</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('step_cpu_fwd', 'Forward(N)', STEP_CPU_FWD_COLOR, 'black', None,
             "<b>%{customdata[0]}: Forward</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('step_overlap', 'Update(N)', STEP_CPU_OVERLAP_COLOR, 'white', None,
             "<b>%{customdata[0]}: Update</b><br>(handling %{customdata[2]})<br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('step_cpu_smp', 'Sample(N)', STEP_CPU_SMP_COLOR, 'black', None,
             "<b>%{customdata[0]}: Sample</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('step_cpu_post', 'Postproc(N)', STEP_CPU_POST_COLOR, 'black', None,
             "<b>%{customdata[0]}: Postprocessing</b><br>(handling %{customdata[2]})<br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('step_cpu_proc', 'Step CPU Proc', STEP_CPU_FWD_COLOR, 'black',
             None,
             "<b>%{customdata[0]}: CPU Processing</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('ctx_gpu_fwd', 'Ctx GPU Forward', CTX_GPU_FWD_COLOR, 'white', None,
             "<b>%{customdata[0]}: GPU Forward</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('ctx_gpu_smp', 'Ctx GPU Sample', CTX_GPU_SMP_COLOR, 'white', None,
             "<b>%{customdata[0]}: GPU Sample</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('step_gpu_fwd', 'Step GPU Fwd', STEP_GPU_FWD_COLOR, 'white', None,
             "<b>%{customdata[0]}: GPU Forward</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('step_gpu_smp', 'Step GPU Smp', STEP_GPU_SMP_COLOR, 'black', None,
             "<b>%{customdata[0]}: GPU Sample</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('ctx_postproc', 'Ctx Postproc (Detokenize+IPC)', '#DDA0DD',
             'black', 0.3,
             "<b>Context Postprocessing</b><br>(Detokenize + IPC to OpenAI Server)<br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('gen_postproc', 'Gen Postproc (Ctx Server)', POST_GEN_COLOR,
             'white', 0.3,
             "<b>Generation Postprocessing</b><br>(Runs on context/disagg server)<br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
            ('disagg_postproc', 'Disagg Postproc', POST_DISAGG_COLOR, 'white',
             0.3,
             "<b>Disagg Postprocessing</b><br>Duration: %{customdata[1]:.2f} ms<br>Start: %{base:.2f} ms<extra></extra>"
             ),
        ]

        for key, name, color, text_color, width, hover_template in trace_configs:
            data = batch_data[key]
            if data['y']:
                # Create customdata - include handled_step for step_overlap and step_cpu_post
                if 'handled_step' in data and data['handled_step']:
                    customdata = list(
                        zip(data['step'], data['duration'],
                            data['handled_step']))
                else:
                    customdata = list(zip(data['step'], data['duration']))
                bar_kwargs = dict(y=data['y'],
                                  x=data['x'],
                                  base=data['base'],
                                  customdata=customdata,
                                  orientation='h',
                                  name=name,
                                  marker_color=color,
                                  legendgroup=key,
                                  showlegend=True,
                                  text=data['text'] if data['text'] else None,
                                  textposition='inside',
                                  textfont=dict(size=9, color=text_color),
                                  hovertemplate=hover_template)
                if width:
                    bar_kwargs['width'] = width
                    bar_kwargs['offset'] = 0.25
                fig.add_trace(go.Bar(**bar_kwargs))

        # Add individual traces (for items not batched)
        for trace in traces:
            fig.add_trace(trace)

        # Calculate chart dimensions
        num_rows = len(y_labels)
        chart_width = max(1400, int(max_timeline * 2))
        chart_height = max(600, num_rows * 25 + 150)

        # Update layout - no internal title, fixed range
        fig.update_layout(
            barmode='overlay',
            xaxis=dict(
                title='Time (milliseconds)',
                side='bottom',
                range=[0, max_timeline * 1.05],
                fixedrange=True,
            ),
            yaxis=dict(
                title='',
                categoryorder='array',
                categoryarray=list(reversed(y_labels)),
                fixedrange=True,
            ),
            hovermode='closest',
            legend=dict(orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02,
                        font=dict(size=9)),
            width=chart_width,
            height=chart_height,
            margin=dict(l=150, r=250, t=50, b=80),
            showlegend=True,
        )

        # Add statistics annotation
        self._add_statistics_annotation(fig, timing_data)

        # Set output file
        if not output_file:
            output_file = 'time_breakdown.html'
        elif not output_file.endswith('.html'):
            output_file += '.html'

        # Generate HTML with scrollable container
        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={
                'scrollZoom':
                False,
                'displayModeBar':
                True,
                'modeBarButtonsToRemove':
                ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            })

        # Calculate viewport dimensions
        viewport_width = min(1400, chart_width)
        viewport_height = min(700, chart_height)

        # Generate request list for dropdown
        request_options = ''.join([
            f'<option value="{data["request_index"]}">Request {data["request_index"]}</option>'
            for data in timing_data
        ])

        # Generate pagination options if needed
        pagination_html = ""
        pagination_js = ""
        if needs_pagination:
            page_options = []
            for page in range(num_pages):
                start_idx = page * requests_per_page
                end_idx = min(
                    (page + 1) * requests_per_page, len(timing_data)) - 1
                start_req = timing_data[start_idx]['request_index']
                end_req = timing_data[end_idx]['request_index']
                page_options.append(
                    f'<option value="{page}">Requests {start_req} - {end_req}</option>'
                )
            pagination_options = ''.join(page_options)
            pagination_html = f'''
        <label for="page-select">Page:</label>
        <select id="page-select">
            {pagination_options}
        </select>'''
            pagination_js = f'''
        var requestsPerPage = {requests_per_page};
        var numPages = {num_pages};
        var allRequestIds = {json.dumps([d['request_index'] for d in timing_data])};
        var currentPage = 0;
        var hasPagination = true;

        document.getElementById('page-select').addEventListener('change', function() {{
            var pageNum = parseInt(this.value);
            filterToPage(pageNum);
        }});

        function filterToPage(pageNum) {{
            if (!plotDiv || !originalData) return;
            currentPage = pageNum;

            var startIdx = pageNum * requestsPerPage;
            var endIdx = Math.min((pageNum + 1) * requestsPerPage, allRequestIds.length);
            var pageRequestIds = allRequestIds.slice(startIdx, endIdx);

            // Build set of valid labels for this page
            var validLabels = new Set();
            pageRequestIds.forEach(function(reqId) {{
                validLabels.add('Req ' + reqId + ' (CPU)');
                validLabels.add('Req ' + reqId + ' (GPU)');
            }});

            // Filter each trace
            var newData = originalData.map(function(trace) {{
                if (!trace.y) return trace;

                var indices = [];
                trace.y.forEach(function(y, i) {{
                    if (validLabels.has(y)) {{
                        indices.push(i);
                    }}
                }});

                if (indices.length === 0) return null;

                var newTrace = JSON.parse(JSON.stringify(trace));
                newTrace.y = indices.map(function(i) {{ return trace.y[i]; }});
                newTrace.x = indices.map(function(i) {{ return trace.x[i]; }});
                newTrace.base = indices.map(function(i) {{ return trace.base[i]; }});
                if (trace.customdata) {{
                    newTrace.customdata = indices.map(function(i) {{ return trace.customdata[i]; }});
                }}
                if (trace.text) {{
                    newTrace.text = indices.map(function(i) {{ return trace.text[i]; }});
                }}
                return newTrace;
            }}).filter(function(t) {{ return t !== null; }});

            // Build y-axis labels for this page
            var pageYLabels = [];
            pageRequestIds.slice().reverse().forEach(function(reqId) {{
                pageYLabels.push('Req ' + reqId + ' (GPU)');
                pageYLabels.push('Req ' + reqId + ' (CPU)');
            }});

            // Calculate x-axis range
            var minX = Infinity, maxX = 0;
            newData.forEach(function(trace) {{
                if (trace.base && trace.x) {{
                    trace.base.forEach(function(b, i) {{
                        minX = Math.min(minX, b);
                        maxX = Math.max(maxX, b + trace.x[i]);
                    }});
                }}
            }});

            var pageHeight = Math.max(600, pageRequestIds.length * 2 * 25 + 150);

            Plotly.react(plotDiv, newData, {{
                barmode: 'overlay',
                xaxis: {{
                    title: 'Time (milliseconds)',
                    side: 'bottom',
                    range: [0, maxX * 1.05],
                }},
                yaxis: {{
                    title: '',
                    categoryorder: 'array',
                    categoryarray: pageYLabels,
                }},
                hovermode: 'closest',
                legend: plotDiv.layout.legend,
                width: {chart_width},
                height: pageHeight,
                margin: {{l: 150, r: 250, t: 50, b: 80}},
                showlegend: true,
            }});

            // Update request dropdown to show only this page's requests
            var requestSelect = document.getElementById('request-select');
            requestSelect.innerHTML = '<option value="all">All (this page)</option>';
            pageRequestIds.forEach(function(reqId) {{
                var option = document.createElement('option');
                option.value = reqId;
                option.textContent = 'Request ' + reqId;
                requestSelect.appendChild(option);
            }});

            document.getElementById('selection-info').textContent =
                'Page ' + (pageNum + 1) + ' of ' + numPages + ' (' + pageRequestIds.length + ' requests)';
        }}

        // Initialize to first page after DOM and Plotly are ready
        document.addEventListener('DOMContentLoaded', function() {{
            // Wait a bit for Plotly to fully initialize
            setTimeout(function() {{
                if (typeof filterToPage === 'function') {{
                    filterToPage(0);
                }}
            }}, 100);
        }});
'''

        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Request Timeline Breakdown</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 5px;
            max-width: {viewport_width}px;
            margin-left: auto;
            margin-right: auto;
            flex-wrap: wrap;
        }}
        .controls label {{
            font-weight: bold;
            color: #333;
        }}
        .controls select {{
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
            min-width: 150px;
        }}
        .current-selection {{
            font-size: 14px;
            color: #666;
        }}
        .chart-container {{
            width: {viewport_width}px;
            height: {viewport_height}px;
            overflow: scroll;
            border: 1px solid #ccc;
            background-color: white;
            margin: 0 auto;
        }}
        .descriptions {{
            margin-top: 30px;
            padding: 20px;
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            max-width: {viewport_width}px;
            margin-left: auto;
            margin-right: auto;
        }}
        .descriptions h2 {{
            margin-top: 0;
            color: #333;
        }}
        .metric-desc {{
            margin-bottom: 10px;
            line-height: 1.5;
        }}
        .metric-name {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .stats-box {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            max-width: {viewport_width}px;
            margin-left: auto;
            margin-right: auto;
        }}
        .hint {{
            font-size: 12px;
            color: #888;
            text-align: center;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1>Request Timeline Breakdown</h1>
    <div class="controls">{pagination_html}
        <label for="request-select">Select Request:</label>
        <select id="request-select">
            <option value="all">All Requests</option>
            {request_options}
        </select>
        <span class="current-selection" id="selection-info">Showing all requests</span>
    </div>
    <p class="hint">💡 Tip: Click on any bar in the chart to focus on that request, or use dropdown to select</p>
    <div class="chart-container" id="chart-container">
        {fig_html}
    </div>
    {self._generate_descriptions_html(timing_data)}

    <script>
        // Store original data for reset
        var originalData = null;
        var plotDiv = null;
        var allYLabels = {json.dumps(y_labels)};

        // Initialize after page load
        document.addEventListener('DOMContentLoaded', function() {{
            plotDiv = document.querySelector('.js-plotly-plot');
            if (plotDiv) {{
                originalData = JSON.parse(JSON.stringify(plotDiv.data));

                // Add click handler to the plot
                plotDiv.on('plotly_click', function(data) {{
                    if (data.points && data.points.length > 0) {{
                        var clickedY = data.points[0].y;
                        // Extract request ID from y label (format: "Req X (CPU)" or "Req X (GPU)")
                        var match = clickedY.match(/Req\\s+(\\S+)\\s+\\(/);
                        if (match) {{
                            var reqId = match[1];
                            document.getElementById('request-select').value = reqId;
                            filterToRequest(reqId);
                        }}
                    }}
                }});
            }}
        }});

        // Handle dropdown change
        document.getElementById('request-select').addEventListener('change', function() {{
            var selectedValue = this.value;
            if (selectedValue === 'all') {{
                // If pagination is enabled, show current page; otherwise show all
                if (typeof hasPagination !== 'undefined' && hasPagination) {{
                    filterToPage(currentPage);
                }} else {{
                    resetView();
                }}
            }} else {{
                filterToRequest(selectedValue);
            }}
        }});

        function filterToRequest(reqId) {{
            if (!plotDiv || !originalData) return;

            var cpuLabel = 'Req ' + reqId + ' (CPU)';
            var gpuLabel = 'Req ' + reqId + ' (GPU)';

            // Filter each trace to only show data for selected request
            var newData = originalData.map(function(trace) {{
                if (!trace.y) return trace;

                var indices = [];
                trace.y.forEach(function(y, i) {{
                    if (y === cpuLabel || y === gpuLabel) {{
                        indices.push(i);
                    }}
                }});

                if (indices.length === 0) return null;

                var newTrace = JSON.parse(JSON.stringify(trace));
                newTrace.y = indices.map(function(i) {{ return trace.y[i]; }});
                newTrace.x = indices.map(function(i) {{ return trace.x[i]; }});
                newTrace.base = indices.map(function(i) {{ return trace.base[i]; }});
                if (trace.customdata) {{
                    newTrace.customdata = indices.map(function(i) {{ return trace.customdata[i]; }});
                }}
                if (trace.text) {{
                    newTrace.text = indices.map(function(i) {{ return trace.text[i]; }});
                }}
                return newTrace;
            }}).filter(function(t) {{ return t !== null; }});

            // Calculate x-axis range for this request
            var minX = Infinity, maxX = 0;
            newData.forEach(function(trace) {{
                if (trace.base && trace.x) {{
                    trace.base.forEach(function(b, i) {{
                        minX = Math.min(minX, b);
                        maxX = Math.max(maxX, b + trace.x[i]);
                    }});
                }}
            }});

            // Update the plot
            Plotly.react(plotDiv, newData, {{
                barmode: 'overlay',
                xaxis: {{
                    title: 'Time (milliseconds)',
                    side: 'bottom',
                    range: [Math.max(0, minX - 5), maxX + 10],
                }},
                yaxis: {{
                    title: '',
                    categoryorder: 'array',
                    categoryarray: [gpuLabel, cpuLabel],
                }},
                hovermode: 'closest',
                legend: plotDiv.layout.legend,
                width: {chart_width},
                height: 300,
                margin: {{l: 150, r: 250, t: 50, b: 80}},
                showlegend: true,
            }});

            document.getElementById('selection-info').textContent = 'Showing Request ' + reqId;
        }}

        function resetView() {{
            if (!plotDiv || !originalData) return;

            document.getElementById('request-select').value = 'all';

            Plotly.react(plotDiv, originalData, {{
                barmode: 'overlay',
                xaxis: {{
                    title: 'Time (milliseconds)',
                    side: 'bottom',
                    range: [0, {max_timeline} * 1.05],
                    fixedrange: true,
                }},
                yaxis: {{
                    title: '',
                    categoryorder: 'array',
                    categoryarray: {json.dumps(list(reversed(y_labels)))},
                    fixedrange: true,
                }},
                hovermode: 'closest',
                legend: plotDiv.layout.legend,
                width: {chart_width},
                height: {chart_height},
                margin: {{l: 150, r: 250, t: 50, b: 80}},
                showlegend: true,
            }});

            document.getElementById('selection-info').textContent = 'Showing all requests';
        }}
        {pagination_js}
    </script>
</body>
</html>
"""  # nosec B608 - HTML template, not SQL
        with open(output_file, 'w') as f:
            f.write(full_html)

        print(f"Timeline diagram saved to: {output_file}")

    def _add_statistics_annotation(self, fig, timing_data: List[Dict]):
        """Add statistics annotation to the plot."""
        stats_lines = ['<b>Median Times (ms):</b>']
        total_times = []

        for metric in self.config.metrics:
            times = [
                data.get(f'{metric.name}_time', 0) * 1000
                for data in timing_data
            ]
            if any(t > 0 for t in times):
                median_time = np.median([t for t in times if t > 0])
                stats_lines.append(f'{metric.display_name}: {median_time:.2f}')

        for data in timing_data:
            total = sum(
                data.get(f'{metric.name}_time', 0) * 1000
                for metric in self.config.metrics)
            total_times.append(total)

        if total_times:
            stats_lines.append(f'<b>Total: {np.median(total_times):.2f}</b>')

        stats_lines.append(f'<b>Requests: {len(timing_data)}</b>')

        fig.add_annotation(x=0.98,
                           y=0.98,
                           xref='paper',
                           yref='paper',
                           text='<br>'.join(stats_lines),
                           showarrow=False,
                           align='right',
                           bgcolor='rgba(255, 255, 255, 0.9)',
                           bordercolor='black',
                           borderwidth=1,
                           font=dict(size=9))

    def _generate_stats_html(self, timing_data: List[Dict]) -> str:
        """Generate HTML for step metrics statistics."""
        all_cpu_times = []
        all_gpu_fwd_times = []
        all_gpu_smp_times = []

        for data in timing_data:
            step_metrics = data.get('step_metrics', [])
            for step in step_metrics or []:
                scheduled = step.get('scheduled_time', 0)
                token = step.get('token_time', 0)
                if scheduled and token:
                    all_cpu_times.append((token - scheduled) * 1000)

                gpu_fwd = step.get('gpu_forward_time', 0)
                gpu_smp = step.get('gpu_sample_time', 0)
                if gpu_fwd:
                    all_gpu_fwd_times.append(gpu_fwd)
                if gpu_smp:
                    all_gpu_smp_times.append(gpu_smp)

        if not all_cpu_times and not all_gpu_fwd_times:
            return ""

        stats_html = '<div class="stats-box"><h3>Step Metrics Statistics</h3><table>'
        stats_html += '<tr><th>Metric</th><th>Avg</th><th>P50</th><th>P99</th><th>Count</th></tr>'

        if all_cpu_times:
            stats_html += f'''<tr>
                <td>CPU Step Time (ms)</td>
                <td>{np.mean(all_cpu_times):.2f}</td>
                <td>{np.percentile(all_cpu_times, 50):.2f}</td>
                <td>{np.percentile(all_cpu_times, 99):.2f}</td>
                <td>{len(all_cpu_times)}</td>
            </tr>'''

        if all_gpu_fwd_times:
            stats_html += f'''<tr>
                <td>GPU Forward Time (ms)</td>
                <td>{np.mean(all_gpu_fwd_times):.2f}</td>
                <td>{np.percentile(all_gpu_fwd_times, 50):.2f}</td>
                <td>{np.percentile(all_gpu_fwd_times, 99):.2f}</td>
                <td>{len(all_gpu_fwd_times)}</td>
            </tr>'''

        if all_gpu_smp_times:
            stats_html += f'''<tr>
                <td>GPU Sample Time (ms)</td>
                <td>{np.mean(all_gpu_smp_times):.2f}</td>
                <td>{np.percentile(all_gpu_smp_times, 50):.2f}</td>
                <td>{np.percentile(all_gpu_smp_times, 99):.2f}</td>
                <td>{len(all_gpu_smp_times)}</td>
            </tr>'''

        stats_html += '</table></div>'
        return stats_html

    def _generate_descriptions_html(self, timing_data: List[Dict]) -> str:
        """Generate HTML for metric descriptions section."""
        desc_items = []

        # Standard metrics from config
        for metric in self.config.metrics:
            times = [
                data.get(f'{metric.name}_time', 0) * 1000
                for data in timing_data
            ]
            if any(t > 0 for t in times):
                desc_items.append(
                    f'<div class="metric-desc">'
                    f'<span class="metric-name">{metric.display_name}:</span> '
                    f'{metric.description}'
                    f'</div>')

        # Check if step metrics are present
        has_step_metrics = any(data.get('step_metrics') for data in timing_data)

        if has_step_metrics:
            # Add step metrics descriptions
            step_metric_descs = [
                ('Step Preprocessing',
                 'Per-step scheduling overhead: batch scheduling, request preparation, and handling responses from previous steps'
                 ),
                ('Step Forward',
                 'CPU time for model forward pass: launching GPU kernels and waiting for completion'
                 ),
                ('Step Update',
                 'GPU synchronization and batch update: sync GPU results, update KV cache, and prepare for sampling'
                 ),
                ('Step Sample',
                 'CPU time for sampling operation: token selection from logits using sampling strategy'
                 ),
                ('Step Postprocessing',
                 'Response handling: update request states, process canceled requests, create streaming responses'
                 ),
                ('GPU Forward',
                 'GPU execution time for model forward pass: attention, MLP, and all transformer layers'
                 ),
                ('GPU Sample',
                 'GPU execution time for sampling: logits processing and token selection on GPU'
                 ),
            ]
            for name, desc in step_metric_descs:
                desc_items.append(f'<div class="metric-desc">'
                                  f'<span class="metric-name">{name}:</span> '
                                  f'{desc}'
                                  f'</div>')

        if not desc_items:
            return ''

        return f'''
    <div class="descriptions">
        <h2>Metric Descriptions</h2>
        {''.join(desc_items)}
    </div>
'''

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
                valid_times = [t for t in times if t > 0]
                print(f"\n{metric.display_name} (seconds):")
                print(
                    f"  Range: {min(valid_times):.3f} to {max(valid_times):.3f}"
                )
                print(f"  Median: {np.median(valid_times):.3f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize TensorRT-LLM server time breakdown',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python time_breakdown.py perf_metrics.json
  python time_breakdown.py perf_metrics.json -o my_timing.html
  python time_breakdown.py perf_metrics.json --stats-only
  python time_breakdown.py perf_metrics.json --max-requests 50 --sort-by e2e
  python time_breakdown.py perf_metrics.json --max-requests 100 --sort-by arrival
        """)

    parser.add_argument('json_file',
                        type=str,
                        help='Path to JSON performance metrics file')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default=None,
                        help='Output HTML file path')
    parser.add_argument('--stats-only',
                        action='store_true',
                        help='Show statistics only')
    parser.add_argument('--show-stats',
                        action='store_true',
                        help='Show statistics with diagram')
    parser.add_argument(
        '--max-requests',
        type=int,
        default=None,
        help='Maximum number of requests to display (default: no limit)')
    parser.add_argument(
        '--sort-by',
        type=str,
        choices=['arrival', 'e2e'],
        default='arrival',
        help=
        'Sort order: arrival (by arrival time, default), e2e (by E2E latency, longest first)'
    )

    args = parser.parse_args()

    analyzer = RequestTimeBreakdown()
    print(f"Parsing: {args.json_file}")
    timing_data = analyzer.parse_json_file(args.json_file)

    if not timing_data:
        print("No timing data found.")
        sys.exit(1)

    if args.stats_only or args.show_stats:
        analyzer.show_statistics(timing_data)

    if not args.stats_only:
        analyzer.create_timing_diagram(timing_data,
                                       args.output,
                                       max_requests=args.max_requests,
                                       sort_by=args.sort_by)


if __name__ == '__main__':
    main()
