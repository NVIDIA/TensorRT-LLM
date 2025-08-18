# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

__all__ = [
    'SpanAttributes', 'SpanKind', 'contains_trace_headers',
    'extract_trace_context', 'extract_trace_headers', 'get_span_exporter',
    'global_otlp_tracer', 'init_tracer', 'insufficient_request_metrics_warning',
    'is_otel_available', 'is_tracing_enabled', 'log_tracing_disabled_warning',
    'set_global_otlp_tracer'
]

import os
from collections.abc import Mapping
from typing import Optional

from tensorrt_llm._utils import run_once
from tensorrt_llm.logger import logger

# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0rc1/vllm/tracing.py#L11
TRACE_HEADERS = ["traceparent", "tracestate"]

_global_tracer_ = None
_is_otel_imported = False
otel_import_error_traceback: Optional[str] = None

try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import \
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import \
        TraceContextTextMapPropagator

    _is_otel_imported = True
except ImportError:
    import traceback

    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(instrumenting_module_name: str,
                otlp_traces_endpoint: str) -> Optional[Tracer]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}")
    trace_provider = TracerProvider()
    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)
    tracer = trace_provider.get_tracer(instrumenting_module_name)
    set_global_otlp_tracer(tracer)
    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
            OTLPSpanExporter
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import \
            OTLPSpanExporter  # type: ignore
    else:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' is configured")
    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(
        headers: Optional[Mapping[str, str]]) -> Optional[Context]:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    # Return only recognized trace headers with normalized lowercase keys
    lower_map = {k.lower(): v for k, v in headers.items()}
    return {h: lower_map[h] for h in TRACE_HEADERS if h in lower_map}


def global_otlp_tracer() -> Tracer:
    """Get the global OTLP instance in the current process."""
    return _global_tracer_


def set_global_otlp_tracer(tracer: Tracer):
    """Set the global OTLP Tracer instance in the current process."""
    global _global_tracer_
    assert _global_tracer_ is None
    _global_tracer_ = tracer


def is_tracing_enabled() -> bool:
    return _global_tracer_ is not None


class SpanAttributes:
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    lower_keys = {k.lower() for k in headers.keys()}
    return any(h in lower_keys for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning(
        "Received a request with trace context but tracing is disabled")


@run_once
def insufficient_request_metrics_warning() -> None:
    logger.warning(
        "Insufficient request metrics available; trace generation aborted.")
