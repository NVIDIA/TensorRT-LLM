import logging
import os
import tempfile
import threading
from collections.abc import Iterable
from concurrent import futures
from typing import Callable, Generator, Literal

import openai
import pytest
import yaml
from llmapi.apps.openai_server import RemoteOpenAIServer
from llmapi.test_llm import get_model_path
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import \
    ExportTraceServiceResponse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer, add_TraceServiceServicer_to_server)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.sdk.environment_variables import \
    OTEL_EXPORTER_OTLP_TRACES_INSECURE

from tensorrt_llm.llmapi.tracing import SpanAttributes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAKE_TRACE_SERVER_ADDRESS = "localhost:4317"


class FakeTraceService(TraceServiceServicer):

    def __init__(self):
        self.request = None
        self.evt = threading.Event()

    def Export(self, request, context):
        self.request = request
        self.evt.set()
        return ExportTraceServiceResponse()


@pytest.fixture(scope="function")
def trace_service() -> Generator[FakeTraceService, None, None]:
    """Fixture to set up a fake gRPC trace service"""
    executor = futures.ThreadPoolExecutor(max_workers=1)
    import grpc
    server = grpc.server(executor)
    service = FakeTraceService()
    add_TraceServiceServicer_to_server(service, server)
    server.add_insecure_port(FAKE_TRACE_SERVER_ADDRESS)
    server.start()

    yield service

    server.stop(None)
    executor.shutdown(wait=True)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module", params=["pytorch", "trt"])
def backend(request):
    return request.param


@pytest.fixture(scope="module", params=[0], ids=["disable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "enable_chunked_prefill": False,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "max_tokens": 40000
            },
            "return_perf_metrics": True,
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, temp_extra_llm_api_options_file: str,
           num_postprocess_workers: int):
    model_path = get_model_path(model_name)
    args = ["--backend", f"{backend}"]
    if backend == "trt":
        args.extend(["--max_beam_width", "4"])
    args.extend(["--extra_llm_api_options", temp_extra_llm_api_options_file])
    args.extend(["--num_postprocess_workers", f"{num_postprocess_workers}"])
    args.extend(["--otlp_traces_endpoint", FAKE_TRACE_SERVER_ADDRESS])

    os.environ[OTEL_EXPORTER_OTLP_TRACES_INSECURE] = "true"

    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


FieldName = Literal["bool_value", "string_value", "int_value", "double_value",
                    "array_value"]


def decode_value(value: AnyValue):
    field_decoders: dict[FieldName, Callable] = {
        "bool_value": (lambda v: v.bool_value),
        "string_value": (lambda v: v.string_value),
        "int_value": (lambda v: v.int_value),
        "double_value": (lambda v: v.double_value),
        "array_value":
        (lambda v: [decode_value(item) for item in v.array_value.values]),
    }
    for field, decoder in field_decoders.items():
        if value.HasField(field):
            return decoder(value)
    raise ValueError(f"Couldn't decode value: {value}")


def decode_attributes(attributes: Iterable[KeyValue]):
    return {kv.key: decode_value(kv.value) for kv in attributes}


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.threadleak(enabled=False)
def test_tracing(client: openai.OpenAI, model_name: str,
                 trace_service: FakeTraceService):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    temperature = 0.9
    top_p = 0.9
    max_completion_tokens = 10

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        top_p=top_p,
        logprobs=False,
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test finish_reason
    finish_reason = chat_completion.choices[0].finish_reason
    completion_tokens = chat_completion.usage.completion_tokens
    if finish_reason == "length":
        assert completion_tokens == 10
    elif finish_reason == "stop":
        assert completion_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")

    timeout = 10
    if not trace_service.evt.wait(timeout):
        raise TimeoutError(
            f"The fake trace service didn't receive a trace within "
            f"the {timeout} seconds timeout")

    request = trace_service.request
    assert len(
        request.resource_spans) == 1, (f"Expected 1 resource span, "
                                       f"but got {len(request.resource_spans)}")
    assert len(request.resource_spans[0].scope_spans) == 1, (
        f"Expected 1 scope span, "
        f"but got {len(request.resource_spans[0].scope_spans)}")
    assert len(request.resource_spans[0].scope_spans[0].spans) == 1, (
        f"Expected 1 span, "
        f"but got {len(request.resource_spans[0].scope_spans[0].spans)}")

    attributes = decode_attributes(
        request.resource_spans[0].scope_spans[0].spans[0].attributes)

    assert (attributes.get(SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS) ==
            chat_completion.usage.completion_tokens)
    assert (attributes.get(SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS) ==
            chat_completion.usage.prompt_tokens)
    assert (attributes.get(
        SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS) == max_completion_tokens)
    assert attributes.get(SpanAttributes.GEN_AI_REQUEST_TOP_P) == top_p
    assert attributes.get(
        SpanAttributes.GEN_AI_REQUEST_TEMPERATURE) == temperature
    assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN) > 0
    assert attributes.get(SpanAttributes.GEN_AI_LATENCY_E2E) > 0
    assert attributes.get(SpanAttributes.GEN_AI_LATENCY_TIME_IN_QUEUE) > 0
    assert len(attributes.get(
        SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS)) > 0
