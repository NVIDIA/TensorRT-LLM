# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared performance-metrics enablement for serving frontends."""

from dataclasses import dataclass

from pydantic import TypeAdapter

from tensorrt_llm.llmapi.llm_args import BaseLlmArgs

_RETURN_PERF_METRICS = TypeAdapter(BaseLlmArgs.model_fields["return_perf_metrics"].annotation)


@dataclass(frozen=True)
class PerfMetricsConfig:
    expose_response_metrics: bool
    collect_request_metrics: bool


def resolve_perf_metrics_config(
    return_perf_metrics: object = False,
    perf_metrics_output_dir: str | None = None,
) -> PerfMetricsConfig:
    """Validate the opt-in and distinguish response exposure from collection.

    File output enables request-performance collection and Prometheus, while
    response headers still require the original return_perf_metrics opt-in.
    No Prometheus imports occur here: directory preparation must come first.
    """
    expose = _RETURN_PERF_METRICS.validate_python(return_perf_metrics)
    return PerfMetricsConfig(
        expose_response_metrics=expose,
        collect_request_metrics=expose or perf_metrics_output_dir is not None,
    )
