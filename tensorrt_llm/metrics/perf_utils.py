# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pure-Python utilities for computing per-request performance metrics.

Kept in tensorrt_llm/metrics/ so that this logic has no GPU / heavy
dependencies and can be unit-tested without a full TensorRT-LLM install.
"""

from typing import Optional

from .enums import MetricNames, RequestEventTiming


def process_req_perf_metrics(
        req_perf_metrics_dict: Optional[dict],
        output_length: int,
        is_multiple_response: bool = False) -> dict:
    """Compute derived per-request latency and token-count metrics.

    Args:
        req_perf_metrics_dict: Raw timing dict from the executor, keyed by
            ``RequestEventTiming`` enum members.  May be ``None`` or empty.
        output_length: Number of output tokens generated for this request.
        is_multiple_response: True when ``sampling_params.n > 1``; token
            counts and TPOT are suppressed in this case to avoid double-
            counting across response candidates.

    Returns:
        Dict mapping ``MetricNames`` enum members to float/int values.
        Keys with value <= 0 are filtered out.
    """
    stat: dict = {}
    if not req_perf_metrics_dict:
        return stat

    arrival = req_perf_metrics_dict.get(RequestEventTiming.ARRIVAL_TIME, 0)
    first_scheduled = req_perf_metrics_dict.get(
        RequestEventTiming.FIRST_SCHEDULED_TIME, 0)
    first_token = req_perf_metrics_dict.get(
        RequestEventTiming.FIRST_TOKEN_TIME, 0)
    last_token = req_perf_metrics_dict.get(RequestEventTiming.LAST_TOKEN_TIME,
                                           0)

    stat: dict = {}

    # Base latency metrics — only compute when all required timestamps are
    # present (> 0).  Absent timestamps default to 0, so a difference that
    # would be negative or zero indicates the timestamp was missing.
    if first_token > 0 and arrival > 0:
        stat[MetricNames.TTFT] = first_token - arrival
    if last_token > 0 and arrival > 0:
        stat[MetricNames.E2E] = last_token - arrival
    # REQUEST_QUEUE_TIME is >= 0 for normally scheduled requests; zero is a
    # valid value (immediate scheduling) so we include it when both timestamps
    # are present.
    if first_scheduled > 0 and arrival > 0:
        stat[MetricNames.REQUEST_QUEUE_TIME] = first_scheduled - arrival

    # Phase latency metrics — require all three anchor timestamps to be valid.
    # PREFILL_TIME = time from first scheduling to first generated token.
    if first_token > 0 and first_scheduled > 0:
        stat[MetricNames.PREFILL_TIME] = first_token - first_scheduled
    # DECODE_TIME = time from first token to last token (generation phase).
    if last_token > 0 and first_token > 0:
        stat[MetricNames.DECODE_TIME] = last_token - first_token
    # INFERENCE_TIME = first_scheduled → last_token (total execution time).
    if last_token > 0 and first_scheduled > 0:
        stat[MetricNames.INFERENCE_TIME] = last_token - first_scheduled

    # Token counts — suppressed for multi-response (n>1) to avoid double-
    # counting across response candidates.
    if output_length > 0 and not is_multiple_response:
        stat[MetricNames.GENERATION_TOKENS] = output_length

    # TPOT = decode duration per output token.  Requires at least 2 tokens
    # (denominator would be 0 for a single-token output).
    if output_length > 1 and not is_multiple_response:
        stat[MetricNames.TPOT] = (last_token - first_token) / (output_length -
                                                                1)

    # Filter out non-positive values: negatives indicate clock-skew anomalies
    # and should not be reported; absent timestamps produce 0 which is filtered
    # here except for REQUEST_QUEUE_TIME (which is re-added below if valid).
    result = {k: v for k, v in stat.items() if v > 0}
    # Restore REQUEST_QUEUE_TIME=0 if it was explicitly computed (zero queue
    # time is a valid, meaningful observation).
    if MetricNames.REQUEST_QUEUE_TIME in stat and stat[
            MetricNames.REQUEST_QUEUE_TIME] >= 0:
        result[MetricNames.REQUEST_QUEUE_TIME] = stat[
            MetricNames.REQUEST_QUEUE_TIME]
    return result
