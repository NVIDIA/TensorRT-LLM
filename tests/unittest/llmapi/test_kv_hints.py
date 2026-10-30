# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV hint transport through real request paths with inference boundaries faked."""

import asyncio
import json
import pickle
from contextlib import nullcontext
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import JsonValue, ValidationError

from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.kv_hints import KvHintsEnvelope
from tensorrt_llm.kv_hints.protocol import MAX_KV_HINTS_BYTES, admit_kv_hints, snapshot_kv_hints
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


class _StubModelConfig:
    """Model identity needed by request parsing, without loading a model."""

    model_type = "llama"
    vocab_size = 1024


def _wire() -> dict:
    """Return the producer's kv.fetch fixture plus an opaque future action."""
    return {
        "protocol_version": "0.1",
        "message_id": "msg-123",
        "actions": [
            {
                "action_id": "a1",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://127.0.0.1:23280",
                    "block_hashes": [11, 22],
                },
            },
            {
                "action_id": "future",
                "action_type": "kv.future_action",
                "action_version": "7.3",
                "payload": {"nested": {"enabled": True}, "items": [1, None, "x"]},
            },
        ],
    }


def _hints() -> KvHintsEnvelope:
    """Return a valid envelope without interpreting its actions."""
    return KvHintsEnvelope.model_validate(_wire())


def _request(hints: KvHintsEnvelope | None = None) -> GenerationRequest:
    """Build a real generation request without an engine."""
    return GenerationRequest([1, 2, 3], SamplingParams(max_tokens=4), kv_hints=hints)


def test_wire_round_trip_preserves_unknown_actions_and_json() -> None:
    """Action order, unknown versions and opaque nested values survive JSON."""
    envelope = KvHintsEnvelope.model_validate_json(json.dumps(_wire()))
    assert json.loads(envelope.model_dump_json()) == _wire()


def test_optional_metadata_survives_admission_without_a_version_change() -> None:
    """Unknown envelope and action metadata remain JSON-valid and immutable."""
    wire = _wire()
    wire["producer_metadata"] = {"trace": ["router-a", 7]}
    wire["actions"][0]["optional_metadata"] = {"future": True}
    hints = KvHintsEnvelope.model_validate(wire)
    request = _request(hints)
    assert request.kv_hints.envelope().model_dump() == wire
    hints.__pydantic_extra__["producer_metadata"]["trace"].append("mutated")
    assert request.kv_hints.envelope().model_dump() == wire


@pytest.mark.parametrize("action_level", [False, True])
@pytest.mark.parametrize("invalid", [float("nan"), object()])
def test_optional_metadata_must_remain_json(action_level: bool, invalid: object) -> None:
    """Forward-compatible metadata cannot carry non-JSON Python objects."""
    wire = _wire()
    target = wire["actions"][0] if action_level else wire
    target["optional_metadata"] = {"nested": invalid}
    with pytest.raises(ValidationError):
        KvHintsEnvelope.model_validate(wire)
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        assert _request(wire).kv_hints is None
    count.assert_called_once_with("malformed")


@pytest.mark.parametrize(
    ("field", "value"),
    [("protocol_version", ""), ("protocol_version", 7), ("message_id", ""), ("actions", None)],
)
def test_malformed_envelope_rejected(field: str, value: object) -> None:
    """Construction rejects malformed envelopes; admission discards optional hints."""
    wire = _wire()
    wire[field] = value
    with pytest.raises(ValidationError):
        KvHintsEnvelope.model_validate(wire)
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        assert _request(wire).kv_hints is None
    count.assert_called_once_with("malformed")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_id", ""),
        ("action_type", 7),
        ("action_version", None),
        ("payload", []),
        ("payload", {"value": float("nan")}),
        ("payload", {"value": b"not-json"}),
    ],
)
def test_malformed_action_rejected(field: str, value: object) -> None:
    """Opaque payloads still have to be valid JSON objects."""
    wire = _wire()
    wire["actions"][0][field] = value
    with pytest.raises(ValidationError):
        KvHintsEnvelope.model_validate(wire)
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        assert _request(wire).kv_hints is None
    count.assert_called_once_with("malformed")


@pytest.mark.parametrize("hints", [b"x", "not-an-object", 7, ["hint"], object()])
def test_non_object_hints_are_ignored_before_admission(hints: object) -> None:
    """Invalid optional field types cannot reject an otherwise valid request."""
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        request = _request(hints)
        assert request.kv_hints is None
        assert request.prompt_token_ids == [1, 2, 3]
        assert pickle.loads(pickle.dumps(request)).kv_hints is None
    count.assert_called_once_with("malformed")


@pytest.mark.parametrize("invalid_json", ["circular", "unpaired_surrogate"])
def test_non_json_raw_hints_are_ignored_before_admission(invalid_json: str) -> None:
    """Invalid raw objects or UTF-8 strings cannot become admitted hint metadata."""
    wire = _wire()
    wire["actions"][0]["payload"]["invalid"] = wire if invalid_json == "circular" else "\ud800"
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        assert _request(wire).kv_hints is None
    count.assert_called_once_with("malformed")


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("location", ["payload", "envelope_extra", "action_extra"])
@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        (-(2**63) - 1, False),
        (-(2**63), True),
        (2**63 - 1, True),
        (2**63, True),
        (2**64 - 1, True),
        (2**64, False),
    ],
)
def test_admission_enforces_msgpack_integer_bounds(
    typed: bool, location: str, value: int, accepted: bool
) -> None:
    """Admitted opaque JSON values must survive the proxy's MessagePack transport."""
    import msgspec

    wire = _wire()
    targets = {
        "payload": wire["actions"][0]["payload"],
        "envelope_extra": wire,
        "action_extra": wire["actions"][0],
    }
    targets[location]["nested"] = {"items": [value]}
    hints = KvHintsEnvelope.model_validate(wire) if typed else wire
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        request = _request(hints)
        assert request.prompt_token_ids == [1, 2, 3]
        if accepted:
            assert request.kv_hints is not None
            decoded = request.kv_hints.envelope().model_dump(mode="json")
            assert msgspec.msgpack.decode(msgspec.msgpack.encode(decoded)) == wire
            count.assert_not_called()
        else:
            assert request.kv_hints is None
            count.assert_called_once_with("malformed")


@pytest.mark.parametrize("request_kind", ["completion", "chat", "responses"])
@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_proxy_forwarding_discards_non_msgpack_hints(request_kind: str, value: int) -> None:
    """Invalid optional hints cannot stop a valid proxy request from being encoded."""
    import msgspec

    from tensorrt_llm.serve.openai_client import _encode_request_body
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionRequest,
        CompletionRequest,
        ResponsesRequest,
    )

    request_types = {
        "completion": (CompletionRequest, {"prompt": [1, 2]}),
        "chat": (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]}),
        "responses": (ResponsesRequest, {"input": "hi"}),
    }
    request_type, inputs = request_types[request_kind]
    hints = _wire()
    hints["actions"][0]["payload"]["nested"] = {"items": [value]}
    handoff = {"request_type": "context_only", "disagg_request_id": 123}
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        request = request_type.model_validate(
            {"model": "test", **inputs, "kv_hints": hints, "disaggregated_params": handoff}
        )
        forwarded = msgspec.msgpack.decode(_encode_request_body(request))
    assert request.kv_hints is None
    assert forwarded["kv_hints"] is None
    assert forwarded["disaggregated_params"]["disagg_request_id"] == 123
    count.assert_called_once_with("malformed")


@pytest.mark.parametrize("request_kind", ["completion", "chat", "responses"])
@pytest.mark.parametrize("typed", [False, True])
def test_http_request_repr_omits_hints_without_changing_transport(
    request_kind: str, typed: bool
) -> None:
    """Request logging omits hints while JSON and proxy forwarding retain them."""
    import msgspec

    from tensorrt_llm.serve.openai_client import _encode_request_body
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionRequest,
        CompletionRequest,
        ResponsesRequest,
    )

    request_types = {
        "completion": (CompletionRequest, {"prompt": [1, 2]}),
        "chat": (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]}),
        "responses": (ResponsesRequest, {"input": "hi"}),
    }
    request_type, inputs = request_types[request_kind]
    wire = _wire()
    sensitive_value = "private-hint-payload-marker"
    wire["actions"][0]["payload"]["opaque_credential"] = sensitive_value
    hints = KvHintsEnvelope.model_validate(wire) if typed else wire
    request = request_type.model_validate({"model": "test", **inputs, "kv_hints": hints})

    for representation in (str(request), repr(request)):
        assert "kv_hints" not in representation
        assert sensitive_value not in representation
    assert request.model_dump(mode="json")["kv_hints"] == wire
    assert json.loads(request.model_dump_json())["kv_hints"] == wire
    assert msgspec.msgpack.decode(_encode_request_body(request))["kv_hints"] == wire


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("container_kind", ["dict", "list"])
@pytest.mark.parametrize("location", ["payload", "envelope_extra", "action_extra"])
@pytest.mark.parametrize(("depth", "accepted"), [(32, True), (200, False)])
def test_admission_requires_decodable_nested_json(
    typed: bool, container_kind: str, location: str, depth: int, accepted: bool
) -> None:
    """Python-valid metadata must also fit the snapshot's JSON decoder."""
    nested: JsonValue = "leaf"
    for _ in range(depth):
        nested = {"child": nested} if container_kind == "dict" else [nested]
    wire = _wire()
    targets = {
        "payload": wire["actions"][0]["payload"],
        "envelope_extra": wire,
        "action_extra": wire["actions"][0],
    }
    targets[location]["nested"] = nested
    envelope = KvHintsEnvelope.model_validate(wire)
    wire_json = envelope.model_dump_json().encode("utf-8")
    assert len(wire_json) < MAX_KV_HINTS_BYTES
    hints = envelope if typed else wire

    if not accepted:
        with pytest.raises(ValidationError, match="recursion limit exceeded"):
            KvHintsEnvelope.model_validate_json(wire_json)

    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        admitted = admit_kv_hints(hints)
        if accepted:
            assert admitted is not None
            assert admitted.model_dump() == wire
            count.assert_not_called()
        else:
            assert admitted is None
            count.assert_called_once_with("malformed")

    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        request = _request(hints)
        assert request.prompt_token_ids == [1, 2, 3]
        if accepted:
            assert request.kv_hints is not None
            assert request.kv_hints.wire_json == wire_json
            assert request.kv_hints.envelope().model_dump() == wire
            count.assert_not_called()
        else:
            assert request.kv_hints is None
            count.assert_called_once_with("malformed")


def test_invalid_request_input_is_not_hidden_by_ignored_hints() -> None:
    """Hint fallback does not relax the generation request's own input contract."""
    with pytest.raises(TypeError, match="prompt_token_ids"):
        GenerationRequest("not-token-ids", SamplingParams(max_tokens=4), kv_hints={})


def test_admission_snapshot_is_deeply_immutable() -> None:
    """Caller and consumer mutations cannot change a queued snapshot."""
    hints = _hints()
    request = _request(hints)
    hints.actions[0].payload["block_hashes"][0] = 99
    hints.actions.append(hints.actions[0])
    assert json.loads(request.kv_hints.wire_json) == _wire()

    decoded = request.kv_hints.envelope()
    decoded.actions[1].payload["nested"]["enabled"] = False
    assert request.kv_hints.envelope().model_dump() == _wire()
    with pytest.raises(FrozenInstanceError):
        request.kv_hints.wire_json = b"{}"


@pytest.mark.parametrize(
    "mutation",
    ["non_json_payload", "invalid_field_type", "missing_protocol_version", "unpaired_surrogate"],
)
def test_admission_revalidates_mutated_envelope(mutation: str) -> None:
    """Admission discards a typed envelope that became invalid after construction."""
    hints = _hints()
    if mutation == "non_json_payload":
        hints.actions[0].payload["bad"] = object()
    elif mutation == "invalid_field_type":
        hints.actions = "invalid-actions"
    elif mutation == "missing_protocol_version":
        del hints.protocol_version
    else:
        hints.actions[0].payload["bad"] = "\ud800"
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        assert _request(hints).kv_hints is None
    count.assert_called_once_with("malformed")


def test_generation_request_pickle_preserves_snapshot() -> None:
    """The existing request pickle path retains Python hint metadata."""
    request = _request(_hints())
    clone = pickle.loads(pickle.dumps(request))
    assert clone.kv_hints == request.kv_hints
    assert clone.kv_hints.envelope().model_dump() == _wire()


def test_no_hint_and_legacy_request_state_default_to_none() -> None:
    """Old serialized requests remain valid without a kv_hints field."""
    request = _request()
    assert request.kv_hints is None
    state = request.__getstate__()
    del state["kv_hints"]
    restored = GenerationRequest.__new__(GenerationRequest)
    restored.__setstate__(state)
    assert restored.kv_hints is None


def test_executor_admits_hint_snapshot() -> None:
    """Executor admission passes hints into the real GenerationRequest."""
    from tensorrt_llm.executor.executor import GenerationExecutor

    executor = MagicMock()
    GenerationExecutor.generate_async(
        executor, [1, 2], SamplingParams(max_tokens=4), kv_hints=_hints()
    )
    submitted = executor.submit.call_args.args[0]
    assert submitted.kv_hints.envelope().model_dump() == _wire()


@pytest.mark.parametrize("batched", [False, True])
def test_llm_generate_forwards_hint_metadata(batched: bool) -> None:
    """Sync generation forwards a scalar envelope or one value per prompt."""
    from tensorrt_llm.llmapi.llm import BaseLLM

    llm = object.__new__(BaseLLM)
    llm.generate_async = MagicMock()
    hints = _hints()
    values = [hints, None] if batched else hints
    BaseLLM.generate(llm, [[1, 2], [3, 4]], kv_hints=values, use_tqdm=False)
    calls = llm.generate_async.call_args_list
    assert calls[0].kwargs["kv_hints"] is hints
    assert calls[1].kwargs["kv_hints"] is (None if batched else hints)


def test_llm_rejects_hint_batch_length_before_submitting() -> None:
    """A misaligned hint batch cannot partially submit requests."""
    from tensorrt_llm.llmapi.llm import BaseLLM

    llm = object.__new__(BaseLLM)
    llm.generate_async = MagicMock()
    with pytest.raises(ValueError, match="kv_hints list length"):
        BaseLLM.generate(llm, [[1, 2], [3, 4]], kv_hints=[_hints()], use_tqdm=False)
    llm.generate_async.assert_not_called()


def test_llm_async_forwards_hints_to_executor() -> None:
    """The async public method retains metadata after prompt preprocessing."""
    from tensorrt_llm.llmapi.llm import BaseLLM, RequestOutput

    llm = MagicMock()
    llm._encode_only = False
    llm._executor.is_shutdown.return_value = False
    llm._prepare_sampling_params.return_value = SamplingParams(max_tokens=4)
    llm._preprocess.return_value = ([1, 2], None, None, None)
    llm.args.return_perf_metrics = False
    hints = _hints()
    with patch.object(RequestOutput, "_from_generation_result"):
        BaseLLM.generate_async(llm, [1, 2], kv_hints=hints)
    assert llm._executor.generate_async.call_args.kwargs["kv_hints"] is hints


@pytest.mark.parametrize("with_hints", [False, True])
def test_worker_and_llm_request_conversion(with_hints: bool) -> None:
    """Real native request construction carries the Python side-channel data."""
    from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
    from tensorrt_llm.executor.base_worker import BaseWorker

    request = _request(_hints() if with_hints else None)
    request.set_id(42)
    worker = MagicMock()
    worker._is_pytorch_backend = True
    worker.max_seq_len = None
    worker.engine.enqueue_request.return_value = 42
    BaseWorker._enqueue_request(worker, request)
    native_request = worker.engine.enqueue_request.call_args.args[0]
    assert native_request.py_kv_hints == request.kv_hints

    llm_request = executor_request_to_llm_request(
        42, native_request, child_req_ids=[], exclude_last_generation_logits=False
    )
    assert llm_request.py_kv_hints == request.kv_hints


def test_child_requests_inherit_hint_snapshot() -> None:
    """Child cloning preserves the admitted snapshot independently of request IDs."""
    from tensorrt_llm._torch.pyexecutor.llm_request import executor_request_to_llm_request
    from tensorrt_llm.bindings import executor as bindings

    request = bindings.Request(
        input_token_ids=[1, 2, 3],
        max_tokens=4,
        sampling_config=bindings.SamplingConfig(num_return_sequences=2),
    )
    request.py_kv_hints = snapshot_kv_hints(_hints())
    llm_request = executor_request_to_llm_request(
        42, request, child_req_ids=[43], exclude_last_generation_logits=False
    )
    assert llm_request.child_requests[0].py_kv_hints == request.py_kv_hints


@pytest.mark.parametrize("broadcast_kind", ["single_round", "tensor_parallel", "pipeline_parallel"])
def test_request_broadcast_transports_hints(broadcast_kind: str) -> None:
    """Rank transport preserves immutable hints alongside other Python metadata."""
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem
    from tensorrt_llm._torch.pyexecutor.request_utils import (
        RequestBroadcaster,
        attach_py_objects_to_requests,
    )

    hints = snapshot_kv_hints(_hints())
    items = [
        RequestQueueItem(
            1,
            SimpleNamespace(
                py_kv_hints=hints,
                py_conversation_params=None,
                py_lora_ckpt_source="hf",
            ),
        ),
        RequestQueueItem(2, SimpleNamespace(py_kv_hints=None, py_conversation_params=None)),
    ]
    pipeline_parallel = broadcast_kind == "pipeline_parallel"
    transport = MagicMock()
    transport.world_size = 2
    transport.rank = 0
    transport.has_pp = pipeline_parallel
    transport.supports_single_round_broadcast = broadcast_kind == "single_round"
    transport.is_first_pp_rank = True
    transport.is_last_pp_rank = False
    transport.pp_size = 2
    transport.broadcast.side_effect = lambda value, root: pickle.loads(pickle.dumps(value))
    transport.tp_cp_broadcast.side_effect = transport.broadcast.side_effect
    transport.broadcast_or_none.side_effect = transport.broadcast.side_effect
    hang_detector = SimpleNamespace(pause=nullcontext)
    broadcaster = RequestBroadcaster(transport, hang_detector)
    collected = broadcaster._collect_py_objects(items)
    with patch.dict("os.environ", {"TRTLLM_PP_REQ_SEND_ASYNC": "0"}):
        if broadcast_kind == "single_round":
            original_requests, _ = broadcaster.broadcast(items)
            assert original_requests is items
            sent = transport.broadcast_or_none.call_args.args[0]
            receiver = MagicMock()
            receiver.world_size = 2
            receiver.rank = 1
            receiver.has_pp = False
            receiver.supports_single_round_broadcast = True
            receiver.broadcast_or_none.return_value = pickle.loads(pickle.dumps(sent))
            _, received = RequestBroadcaster(receiver, hang_detector).broadcast([])
            transport.broadcast_or_none.assert_called_once()
            receiver.broadcast_or_none.assert_called_once_with(None, root=0)
            transport.broadcast_int64.assert_not_called()
            receiver.broadcast_int64.assert_not_called()
        else:
            _, received = broadcaster._broadcast_requests(items, collected)
    if pipeline_parallel:
        transport.tp_cp_broadcast.assert_called_once()
        transport.send_object.assert_called_once()
        sent = transport.send_object.call_args.args[0]
        receiver = MagicMock()
        receiver.world_size = 2
        receiver.has_pp = True
        receiver.is_first_pp_rank = False
        receiver.is_last_pp_rank = True
        receiver.pp_size = 2
        receiver.recv_object.return_value = pickle.loads(pickle.dumps(sent))
        _, received = RequestBroadcaster(receiver, hang_detector)._broadcast_requests([], None)
        receiver.recv_object.assert_called_once()
    elif broadcast_kind == "tensor_parallel":
        transport.broadcast.assert_called_once()

    targets = [RequestQueueItem(1, SimpleNamespace()), RequestQueueItem(2, SimpleNamespace())]
    attach_py_objects_to_requests(targets, received)
    transported_hints = targets[0].request.py_kv_hints
    assert transported_hints == hints
    assert targets[0].request.py_lora_ckpt_source == "hf"
    assert getattr(targets[1].request, "py_kv_hints", None) is None
    transported_hints.envelope().actions[0].payload["block_hashes"].append(99)
    assert transported_hints.envelope().model_dump() == _wire()
    with pytest.raises(FrozenInstanceError):
        transported_hints.wire_json = b"{}"


@pytest.mark.parametrize("request_kind", ["completion", "chat", "responses"])
def test_http_request_models_validate_hints(request_kind: str) -> None:
    """All supported HTTP request schemas share the same envelope validation."""
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionRequest,
        CompletionRequest,
        ResponsesRequest,
    )

    constructors = {
        "completion": (CompletionRequest, {"prompt": [1, 2]}),
        "chat": (ChatCompletionRequest, {"messages": [{"role": "user", "content": "hi"}]}),
        "responses": (ResponsesRequest, {"input": "hi"}),
    }
    request_type, fields = constructors[request_kind]
    request = request_type(model="test", kv_hints=_wire(), **fields)
    assert request.kv_hints.model_dump() == _wire()
    assert request_type(model="test", **fields).kv_hints is None
    invalid = _wire()
    invalid["protocol_version"] = "unsupported"
    assert request_type(model="test", kv_hints=invalid, **fields).kv_hints is None


@pytest.mark.parametrize("request_kind", ["completion", "chat", "harmony", "responses"])
@pytest.mark.parametrize("with_hints", [False, True])
def test_http_handlers_forward_hints(request_kind: str, with_hints: bool) -> None:
    """Run each real HTTP handler until its model-generation boundary.

    The engine deliberately raises after recording the call. Tokenized inputs
    avoid loading a tokenizer; Responses preprocessing is the only stubbed
    adapter because it normally resolves model and conversation state.
    """
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionRequest,
        CompletionRequest,
        ResponsesRequest,
    )
    from tensorrt_llm.serve.openai_server import OpenAIServer

    server = object.__new__(OpenAIServer)
    server.model = "test"
    server.allow_request_chat_template = False
    server.model_config = _StubModelConfig()
    server.processor = None
    server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=1024))
    server.chat_template = None
    server.tool_parser = None
    server.tool_call_id_type = "random"
    server.multimodal_server_config = None
    server.use_harmony = request_kind == "harmony"
    server.harmony_adapter = SimpleNamespace(get_stop_tokens=lambda: [2])
    server.enable_store = False
    server.conversation_store = None
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            gather_generation_logits=False,
            reasoning_parser=None,
            backend="pytorch",
            guided_decoding_backend=None,
            num_postprocess_workers=0,
        ),
        generate_async=MagicMock(side_effect=ValueError("captured at engine boundary")),
    )
    hints = _hints() if with_hints else None
    if request_kind == "completion":
        request = CompletionRequest(model="test", prompt=[1, 2], max_tokens=4, kv_hints=hints)
        handler = server.openai_completion
    elif request_kind == "responses":
        request = ResponsesRequest(model="test", input="hi", max_output_tokens=4, kv_hints=hints)
        handler = server.openai_responses
    else:
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            prompt_token_ids=[1, 2],
            max_tokens=4,
            kv_hints=hints,
        )
        handler = server.chat_harmony if request_kind == "harmony" else server.openai_chat

    with patch(
        "tensorrt_llm.serve.openai_server.responses_api_request_preprocess",
        new=AsyncMock(return_value=([1, 2], SamplingParams(max_tokens=4))),
    ):
        response = asyncio.run(handler(request, None))
    server.generator.generate_async.assert_called_once()
    assert server.generator.generate_async.call_args.kwargs["kv_hints"] == hints
    assert b"captured at engine boundary" in response.body


@pytest.mark.parametrize("typed", [False, True])
def test_unknown_version_is_ignored_before_admission(typed: bool) -> None:
    """Raw future envelopes do not have to conform to the current action schema."""
    wire = _wire()
    wire["protocol_version"] = "future"
    hints = (
        KvHintsEnvelope.model_validate(wire)
        if typed
        else {"protocol_version": "future", "future_payload": [7], "actions": "different format"}
    )
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        request = _request(hints)
        assert request.kv_hints is None
        pickle.loads(pickle.dumps(request))
    count.assert_called_once_with("unknown_version")


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_snapshot_bounds_utf8_bytes(typed: bool, extra_bytes: int) -> None:
    """The byte bound includes multibyte payloads and accepts the exact limit."""
    wire = _wire()
    wire["actions"][0]["payload"] = {"padding": "雪"}
    size = len(KvHintsEnvelope.model_validate(wire).model_dump_json().encode("utf-8"))
    wire["actions"][0]["payload"]["padding"] += "x" * (MAX_KV_HINTS_BYTES - size + extra_bytes)
    hints = KvHintsEnvelope.model_validate(wire) if typed else wire
    with patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count:
        request = _request(hints)
    if extra_bytes:
        assert request.kv_hints is None
        count.assert_called_once_with("oversized")
    else:
        assert len(request.kv_hints.wire_json) == MAX_KV_HINTS_BYTES
        count.assert_not_called()


def test_hint_types_are_exported_with_request_parameters() -> None:
    """Both public import paths identify the same request parameter types."""
    from tensorrt_llm import llmapi
    from tensorrt_llm.kv_hints import KvHintAction, KvHintsInput

    for name, value in (
        ("KvHintAction", KvHintAction),
        ("KvHintsEnvelope", KvHintsEnvelope),
        ("KvHintsInput", KvHintsInput),
    ):
        assert name in llmapi.__all__
        assert getattr(llmapi, name) is value


@pytest.mark.parametrize(
    "route_kind", ["worker_json", "worker_msgpack", "proxy", "proxy_responses"]
)
@pytest.mark.parametrize("trusted", [False, True])
@pytest.mark.parametrize(
    "hint_kind", ["valid", "unknown", "oversized", "malformed", "malformed_action", "non_object"]
)
def test_http_ingress_policy_before_nested_validation(
    route_kind: str,
    trusted: bool,
    hint_kind: str,
) -> None:
    """Real worker/proxy routes strip before parsing and never alter P/D fields."""
    import msgspec
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tensorrt_llm.serve.kv_hints import KvHintsRoute
    from tensorrt_llm.serve.openai_protocol import CompletionRequest, ResponsesRequest
    from tensorrt_llm.serve.openai_server import _MsgspecRoute

    hints = _wire()
    if hint_kind == "unknown":
        hints = {"protocol_version": "future", "actions": "different schema"}
    elif hint_kind == "oversized":
        hints["actions"][0]["payload"] = {"large": "雪" * MAX_KV_HINTS_BYTES}
    elif hint_kind == "malformed":
        hints.pop("message_id")
    elif hint_kind == "malformed_action":
        hints["actions"][0]["action_type"] = 7
    elif hint_kind == "non_object":
        hints = 7
    app = FastAPI()
    app.state.trust_kv_hints = trusted
    app.router.route_class = KvHintsRoute if route_kind.startswith("proxy") else _MsgspecRoute

    def capture(request: CompletionRequest | ResponsesRequest) -> dict:
        """Return admitted hints and required handoff fields without inference."""
        admitted = _request(request.kv_hints)
        return {
            "kv_hints": None
            if admitted.kv_hints is None
            else admitted.kv_hints.envelope().model_dump(),
            "disaggregated_params": request.disaggregated_params.model_dump(),
        }

    async def capture_completions(request: CompletionRequest) -> dict:
        """Validate completion inputs before recording their admitted metadata."""
        return capture(request)

    async def capture_responses(request: ResponsesRequest) -> dict:
        """Validate Responses inputs before recording their admitted metadata."""
        return capture(request)

    is_responses = route_kind == "proxy_responses"
    route_path = "/v1/responses" if is_responses else "/v1/completions"
    endpoint = capture_responses if is_responses else capture_completions
    app.add_api_route(route_path, endpoint, methods=["POST"])
    body = {
        "model": "test",
        **({"input": "hi"} if is_responses else {"prompt": [1, 2]}),
        "kv_hints": hints,
        "disaggregated_params": {"request_type": "context_only", "disagg_request_id": 123},
    }
    with (
        TestClient(app) as client,
        patch("tensorrt_llm.serve.kv_hints.count_ignored_kv_hints") as trust_count,
        patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as admission_count,
    ):
        if route_kind == "worker_msgpack":
            response = client.post(
                route_path,
                content=msgspec.msgpack.encode(body),
                headers={"content-type": "application/json", "x-trtllm-msgpack": "1"},
            )
        else:
            response = client.post(route_path, json=body)
        assert response.status_code == 200
        output = response.json()
        assert output["kv_hints"] == (_wire() if trusted and hint_kind == "valid" else None)
        assert output["disaggregated_params"]["disagg_request_id"] == 123
        if not trusted:
            trust_count.assert_called_once_with("untrusted")
            admission_count.assert_not_called()
        else:
            trust_count.assert_not_called()
            reasons = {
                "unknown": "unknown_version",
                "oversized": "oversized",
                "malformed": "malformed",
                "malformed_action": "malformed",
                "non_object": "malformed",
            }
            if hint_kind == "valid":
                admission_count.assert_not_called()
            else:
                admission_count.assert_called_once_with(reasons[hint_kind])


@pytest.mark.parametrize("route_kind", ["worker_json", "worker_msgpack", "proxy"])
@pytest.mark.parametrize("invalid_field", ["prompt", "handoff_request_type"])
def test_ignored_hints_do_not_relax_http_request_validation(
    route_kind: str,
    invalid_field: str,
) -> None:
    """Ordinary request and required handoff fields remain strict after hint fallback."""
    import msgspec
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tensorrt_llm.serve.kv_hints import KvHintsRoute
    from tensorrt_llm.serve.openai_protocol import CompletionRequest
    from tensorrt_llm.serve.openai_server import _MsgspecRoute

    app = FastAPI()
    app.state.trust_kv_hints = True
    app.router.route_class = KvHintsRoute if route_kind == "proxy" else _MsgspecRoute

    def capture(request: CompletionRequest) -> dict:
        """Fail if invalid request fields reach the endpoint."""
        raise AssertionError("Invalid request fields must fail before endpoint dispatch")

    app.add_api_route("/v1/completions", capture, methods=["POST"])
    body = {
        "model": "test",
        "prompt": [1, 2],
        "kv_hints": {"protocol_version": "0.1", "actions": "malformed"},
        "disaggregated_params": {"request_type": "context_only", "disagg_request_id": 123},
    }
    if invalid_field == "prompt":
        body.pop("prompt")
        error_location = ["body", "prompt"]
    else:
        body["disaggregated_params"].pop("request_type")
        error_location = ["body", "disaggregated_params", "request_type"]

    with (
        TestClient(app) as client,
        patch("tensorrt_llm.kv_hints.protocol.count_ignored_kv_hints") as count,
    ):
        if route_kind == "worker_msgpack":
            response = client.post(
                "/v1/completions",
                content=msgspec.msgpack.encode(body),
                headers={"content-type": "application/json", "x-trtllm-msgpack": "1"},
            )
        else:
            response = client.post("/v1/completions", json=body)
    count.assert_called_once_with("malformed")
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any(error["loc"] == error_location for error in errors)
    assert all("kv_hints" not in error["loc"] for error in errors)
