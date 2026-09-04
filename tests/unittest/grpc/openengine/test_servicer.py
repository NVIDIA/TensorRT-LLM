# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OpenEngine Generate adapter."""

import asyncio
import base64
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip(
    "openengine",
    reason='OpenEngine dependency not installed (pip install "tensorrt_llm[openengine]")',
)

import grpc  # noqa: E402
from openengine.v1 import generation_pb2  # noqa: E402

import tensorrt_llm.grpc.openengine.servicer as openengine_servicer  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import (  # noqa: E402
    OpenEngineInferenceServicer,
    sampling_params_from_request,
)

# Runs on the CPU stage: the engine is stubbed, so nothing here needs a GPU.
pytestmark = pytest.mark.cpu_only


class _FakeTokenizer:
    def convert_ids_to_tokens(
        self, token_ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str]:
        del skip_special_tokens
        return [f"token-{token_id}" for token_id in token_ids]

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(f"token-{token_id}" for token_id in token_ids)


class _FakeResultHandle:
    def __init__(self, results: list[Any]) -> None:
        self._results = results
        self.aborted = False

    async def __aiter__(self) -> AsyncIterator[Any]:
        for result in self._results:
            yield result

    def abort(self) -> None:
        self.aborted = True


class _FakeLlm:
    def __init__(self, results: list[Any]) -> None:
        self.tokenizer = _FakeTokenizer()
        self.result_handle = _FakeResultHandle(results)
        self.generate_kwargs = None

    def generate_async(self, **kwargs: Any) -> _FakeResultHandle:
        self.generate_kwargs = kwargs
        kwargs["sampling_params"]._validate()
        if kwargs["sampling_params"].stop == ["AB"]:
            kwargs["sampling_params"]._stop_word_ids = [[10, 11]]
        return self.result_handle


class _FakeContext:
    def __init__(self, metadata: Sequence[tuple[str, str]] = ()) -> None:
        self._metadata = [SimpleNamespace(key=key, value=value) for key, value in metadata]
        self.abort_code = None
        self.abort_details = None
        self.done_callbacks = []

    def invocation_metadata(self) -> list[SimpleNamespace]:
        return self._metadata

    def cancelled(self) -> bool:
        return False

    def add_done_callback(self, callback: Any) -> None:
        self.done_callbacks.append(callback)

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.abort_code = code
        self.abort_details = details
        raise _AbortError


class _AbortError(Exception):
    pass


def _logprob(value: float, rank: int) -> SimpleNamespace:
    return SimpleNamespace(logprob=value, rank=rank)


def test_sampling_params_from_request_preserves_portable_options() -> None:
    """OpenEngine optional scalars and stop conditions retain their semantics."""
    request = generation_pb2.GenerateRequest(
        sampling=generation_pb2.SamplingParams(
            temperature=0.0,
            top_p=0.9,
            top_k=20,
            min_p=0.1,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            repetition_penalty=1.1,
            seed=42,
            num_sequences=1,
        ),
        stopping=generation_pb2.StoppingOptions(
            max_tokens=64,
            min_tokens=4,
            conditions=[
                generation_pb2.StopCondition(stop_text="done"),
                generation_pb2.StopCondition(stop_token_id=7),
            ],
            ignore_eos=False,
            include_stop_in_output=True,
        ),
        response=generation_pb2.ResponseOptions(
            return_prompt_logprobs=True,
            prompt_candidates=generation_pb2.CandidateTokenSelection(top_n=3),
            return_output_logprobs=True,
            output_candidates=generation_pb2.CandidateTokenSelection(top_n=4),
        ),
        guided=generation_pb2.GuidedDecoding(regex="[a-z]+"),
    )

    params = sampling_params_from_request(request, "xgrammar")

    assert params.temperature == 0.0
    assert params.top_p == pytest.approx(0.9)
    assert params.top_k == 20
    assert params.min_p == pytest.approx(0.1)
    assert params.frequency_penalty == pytest.approx(0.2)
    assert params.presence_penalty == pytest.approx(0.3)
    assert params.repetition_penalty == pytest.approx(1.1)
    assert params.seed == 42
    assert params.n == 1
    assert params.max_tokens == 64
    assert params.min_tokens == 4
    assert params.stop == ["done"]
    assert params.stop_token_ids == [7]
    assert params.ignore_eos is False
    assert params.include_stop_str_in_output is True
    assert params.prompt_logprobs == 3
    assert params.logprobs == 4
    assert params.guided_decoding.regex == "[a-z]+"


def test_sampling_params_default_truncation_matches_openai() -> None:
    """Unspecified top_p/top_k default to the /v1/completions values (1.0 / 0).

    This keeps a sampling request behaving identically across the HTTP and
    OpenEngine transports.
    """
    request = generation_pb2.GenerateRequest(
        sampling=generation_pb2.SamplingParams(temperature=0.7),
    )

    params = sampling_params_from_request(request)

    assert params.temperature == pytest.approx(0.7)
    assert params.top_p == pytest.approx(1.0)
    assert params.top_k == 0


def test_omitted_max_tokens_is_unbounded_like_openai() -> None:
    """An omitted stopping.max_tokens means "unbounded", not SamplingParams' 32.

    /v1/completions models an absent max_tokens as None and the engine deduces
    the budget from max_seq_len. Inheriting the dataclass default instead would
    silently truncate every OpenEngine request at 32 tokens.
    """
    assert sampling_params_from_request(generation_pb2.GenerateRequest()).max_tokens is None

    with_stopping = generation_pb2.GenerateRequest(
        stopping=generation_pb2.StoppingOptions(min_tokens=1),
    )
    assert sampling_params_from_request(with_stopping).max_tokens is None

    explicit = generation_pb2.GenerateRequest(
        stopping=generation_pb2.StoppingOptions(max_tokens=7),
    )
    assert sampling_params_from_request(explicit).max_tokens == 7


def test_greedy_decoding_with_multiple_sequences_is_rejected() -> None:
    """n/best_of go through SamplingParams._validate(), as on /v1/completions.

    _validate runs only from __post_init__, so assigning n after construction
    would skip the guard that rejects multiple returns under greedy decoding and
    hand the client duplicate sequences reported as distinct samples.
    """
    request = generation_pb2.GenerateRequest(
        sampling=generation_pb2.SamplingParams(temperature=0.0, num_sequences=3),
    )

    with pytest.raises(ValueError, match="[Gg]reedy"):
        sampling_params_from_request(request)


def test_multiple_sequences_are_accepted_when_sampling() -> None:
    """The guard is specific to greedy decoding; n > 1 still works otherwise."""
    request = generation_pb2.GenerateRequest(
        sampling=generation_pb2.SamplingParams(temperature=0.8, num_sequences=3),
    )

    params = sampling_params_from_request(request)

    assert params.n == 3
    assert params.best_of == 3


def test_sampling_params_default_truncation_without_sampling_message() -> None:
    """The truncation defaults are applied even with no sampling message."""
    params = sampling_params_from_request(generation_pb2.GenerateRequest())

    assert params.top_p == pytest.approx(1.0)
    assert params.top_k == 0


def test_sampling_params_explicit_truncation_is_preserved() -> None:
    """Client-provided top_p/top_k are not overridden by the defaults."""
    request = generation_pb2.GenerateRequest(
        sampling=generation_pb2.SamplingParams(temperature=0.7, top_p=0.5, top_k=10),
    )

    params = sampling_params_from_request(request)

    assert params.top_p == pytest.approx(0.5)
    assert params.top_k == 10


def test_token_holdback_from_stop_word_ids() -> None:
    """Holdback withholds (longest stop-word length - 1) trailing tokens."""
    params = openengine_servicer.SamplingParams(include_stop_str_in_output=False)
    params._stop_word_ids = [[10, 11], [12]]
    assert openengine_servicer._token_holdback(params) == 1


def test_token_holdback_zero_when_including_stop() -> None:
    """No holdback is needed when the stop string is kept in the output."""
    params = openengine_servicer.SamplingParams(include_stop_str_in_output=True)
    params._stop_word_ids = [[10, 11]]
    assert openengine_servicer._token_holdback(params) == 0


def test_token_holdback_zero_without_stop_words() -> None:
    """Missing/empty _stop_word_ids degrades to no holdback rather than failing."""
    params = openengine_servicer.SamplingParams(include_stop_str_in_output=False)
    assert openengine_servicer._token_holdback(params) == 0


def test_disaggregated_params_none_for_normal_request() -> None:
    """A request with no phase marker and no kv.session is aggregated (None)."""
    assert (
        openengine_servicer._disaggregated_params_from_request(generation_pb2.GenerateRequest())
        is None
    )


def test_disaggregated_params_context_only_from_extra() -> None:
    """extra.request_type selects the disaggregation phase."""
    request = generation_pb2.GenerateRequest()
    request.extra.update({"request_type": "context_only"})
    params = openengine_servicer._disaggregated_params_from_request(request)
    assert params is not None
    assert params.request_type == "context_only"


def test_disaggregated_params_invalid_request_type() -> None:
    request = generation_pb2.GenerateRequest()
    request.extra.update({"request_type": "bogus"})
    with pytest.raises(ValueError):
        openengine_servicer._disaggregated_params_from_request(request)


def test_disaggregated_params_generation_only_from_session() -> None:
    """A kv.session handle decodes back into generation_only ctx params."""
    request = generation_pb2.GenerateRequest()
    session = request.kv.session
    session.session_id = "42"
    session.dp_rank = 1
    session.endpoints.add(host="10.0.0.5", port=9100, protocol="UCX")
    session.attributes_struct.update(
        {
            "opaque_state": base64.b64encode(b"kvstate").decode(),
            "first_gen_tokens": [7, 8],
        }
    )

    params = openengine_servicer._disaggregated_params_from_request(request)

    assert params.request_type == "generation_only"
    assert params.ctx_request_id == 42
    assert params.ctx_dp_rank == 1
    assert params.ctx_info_endpoint == "10.0.0.5:9100"
    assert params.opaque_state == b"kvstate"
    assert params.first_gen_tokens == [7, 8]


def test_disaggregated_params_session_id_must_be_int() -> None:
    request = generation_pb2.GenerateRequest()
    request.kv.session.session_id = "not-an-int"
    with pytest.raises(ValueError):
        openengine_servicer._disaggregated_params_from_request(request)


def test_prefill_ready_round_trip() -> None:
    """A context handoff packs into PrefillReady and decodes back identically."""
    params = openengine_servicer.DisaggregatedParams(
        request_type="context_only",
        ctx_request_id=99,
        ctx_dp_rank=2,
        ctx_info_endpoint="1.2.3.4:5555",
        opaque_state=b"state-bytes",
        first_gen_tokens=[11, 12],
    )

    resp = openengine_servicer._prefill_ready_response("req-1", params, transfer_backend="NIXL")

    assert resp.WhichOneof("event") == "prefill_ready"
    session = resp.prefill_ready.kv_session
    assert session.session_id == "99"
    assert session.dp_rank == 2
    assert session.transfer_backend == "NIXL"
    assert session.endpoints[0].host == "1.2.3.4"
    assert session.endpoints[0].port == 5555

    follow_up = generation_pb2.GenerateRequest()
    follow_up.kv.session.CopyFrom(session)
    decoded = openengine_servicer._disaggregated_params_from_request(follow_up)
    assert decoded.request_type == "generation_only"
    assert decoded.ctx_request_id == 99
    assert decoded.ctx_dp_rank == 2
    assert decoded.opaque_state == b"state-bytes"
    assert decoded.first_gen_tokens == [11, 12]


def test_prefill_ready_round_trip_full_handoff() -> None:
    """The full disagg handoff (draft tokens, ids, schedule style, usage, dp_rank=0) round-trips."""
    from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

    params = openengine_servicer.DisaggregatedParams(
        request_type="context_only",
        ctx_request_id=7,
        ctx_dp_rank=0,  # explicit 0 must not collapse to "unset"
        opaque_state=b"s",
        first_gen_tokens=[1],
        draft_tokens=[2, 3, 4],
        disagg_request_id=123456789012345,
        schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
        ctx_usage={"prompt_tokens": 5, "total_tokens": 6},
    )

    resp = openengine_servicer._prefill_ready_response("r", params, transfer_backend="")
    follow_up = generation_pb2.GenerateRequest()
    follow_up.kv.session.CopyFrom(resp.prefill_ready.kv_session)
    decoded = openengine_servicer._disaggregated_params_from_request(follow_up)

    assert decoded.draft_tokens == [2, 3, 4]
    assert decoded.disagg_request_id == 123456789012345
    assert decoded.ctx_dp_rank == 0
    assert decoded.schedule_style == DisaggScheduleStyle.GENERATION_FIRST
    assert decoded.ctx_usage == {"prompt_tokens": 5, "total_tokens": 6}


def test_disaggregated_params_dp_rank_none_preserved() -> None:
    """An unset ctx_dp_rank round-trips as None, not 0."""
    params = openengine_servicer.DisaggregatedParams(
        request_type="context_only", ctx_request_id=1, opaque_state=b"x"
    )
    resp = openengine_servicer._prefill_ready_response("r", params, transfer_backend="")
    follow_up = generation_pb2.GenerateRequest()
    follow_up.kv.session.CopyFrom(resp.prefill_ready.kv_session)
    decoded = openengine_servicer._disaggregated_params_from_request(follow_up)
    assert decoded.ctx_dp_rank is None


def test_prefill_ready_carries_first_gen_log_probs() -> None:
    """Verbose logprobs (dict[int, Logprob] per position) survive the handoff."""
    from tensorrt_llm.executor.result import Logprob

    params = openengine_servicer.DisaggregatedParams(
        request_type="context_only",
        ctx_request_id=1,
        opaque_state=b"x",
        first_gen_log_probs=[
            {3681: Logprob(logprob=-0.68, rank=1), 5982: Logprob(logprob=-1.8, rank=2)},
        ],
    )
    resp = openengine_servicer._prefill_ready_response("r", params, transfer_backend="")
    follow_up = generation_pb2.GenerateRequest()
    follow_up.kv.session.CopyFrom(resp.prefill_ready.kv_session)
    decoded = openengine_servicer._disaggregated_params_from_request(follow_up)

    fglp = decoded.first_gen_log_probs
    assert isinstance(fglp, list) and len(fglp) == 1
    entry = fglp[0]
    assert set(entry) == {3681, 5982}
    assert entry[3681].logprob == pytest.approx(-0.68)
    assert entry[3681].rank == 1
    assert entry[5982].rank == 2


def test_first_gen_log_probs_simple_format_round_trip() -> None:
    """Simple logprobs (per-token float) survive the serialize/deserialize."""
    serialized = openengine_servicer._serialize_first_gen_log_probs([-0.1, -0.2])
    restored = openengine_servicer._deserialize_first_gen_log_probs(serialized)
    assert restored == [pytest.approx(-0.1), pytest.approx(-0.2)]


def test_sampling_params_records_only_provided_fields() -> None:
    """Only client-supplied sampling fields are recorded, not the materialized defaults."""
    request = generation_pb2.GenerateRequest(sampling=generation_pb2.SamplingParams(top_p=0.8))
    params = sampling_params_from_request(request)
    provided = params._request_provided_fields
    assert provided is not None
    assert "top_p" in provided
    # temperature/top_k were materialized as defaults, not client-provided.
    assert "temperature" not in provided
    assert "top_k" not in provided


def test_generate_streams_incremental_events_and_final_usage() -> None:
    """Generate emits non-cumulative deltas and one terminal event with usage."""
    prompt_logprobs = [
        {2: _logprob(-0.2, 1), 3: _logprob(-1.2, 2)},
        {10: _logprob(-0.3, 1), 12: _logprob(-1.3, 2)},
    ]
    first_output = SimpleNamespace(
        index=0,
        token_ids=[10],
        text="A",
        logprobs=[
            {
                10: _logprob(-2.3, 3),
                12: _logprob(-0.3, 1),
                14: _logprob(-1.3, 2),
            }
        ],
        prompt_logprobs=prompt_logprobs,
        finish_reason=None,
        stop_reason=None,
    )
    final_output = SimpleNamespace(
        index=0,
        token_ids=[10, 11],
        text="AB",
        logprobs=[
            {
                10: _logprob(-2.3, 3),
                12: _logprob(-0.3, 1),
                14: _logprob(-1.3, 2),
            },
            {11: _logprob(-0.4, 1), 13: _logprob(-1.4, 2)},
        ],
        prompt_logprobs=prompt_logprobs,
        finish_reason="length",
        stop_reason=None,
    )
    results = [
        SimpleNamespace(
            prompt_token_ids=[1, 2], outputs=[first_output], cached_tokens=1, error=None
        ),
        SimpleNamespace(
            prompt_token_ids=[1, 2], outputs=[final_output], cached_tokens=1, error=None
        ),
    ]
    llm = _FakeLlm(results)
    servicer = OpenEngineInferenceServicer(llm, model="test-model")
    context = _FakeContext(
        metadata=(
            ("traceparent", "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"),
            ("tracestate", "vendor=value"),
        )
    )
    request = generation_pb2.GenerateRequest(
        request_id="request-1",
        model="test-model",
        prompt="hello",
        sampling=generation_pb2.SamplingParams(temperature=0.7),
        stopping=generation_pb2.StoppingOptions(max_tokens=2),
        response=generation_pb2.ResponseOptions(
            return_prompt_logprobs=True,
            prompt_candidates=generation_pb2.CandidateTokenSelection(top_n=2),
            return_output_logprobs=True,
            output_candidates=generation_pb2.CandidateTokenSelection(top_n=2),
        ),
        kv=generation_pb2.KvOptions(cache_salt="tenant-a"),
    )

    async def collect_responses() -> list[generation_pb2.GenerateResponse]:
        return [response async for response in servicer.Generate(request, context)]

    responses = asyncio.run(collect_responses())

    assert [response.WhichOneof("event") for response in responses] == [
        "prompt",
        "token",
        "token",
        "finished",
    ]
    assert [token.token_id for token in responses[1].token.tokens] == [10]
    assert responses[1].token.text == "A"
    assert responses[1].token.tokens[0].rank == 3
    assert [candidate.token_id for candidate in responses[1].token.tokens[0].candidates] == [
        12,
        14,
    ]
    assert [token.token_id for token in responses[2].token.tokens] == [11]
    assert responses[2].token.text == "B"
    assert responses[3].finished.output_index == 0
    assert responses[3].finished.reason == generation_pb2.FINISH_REASON_LENGTH
    assert responses[3].usage.prompt_tokens == 2
    assert responses[3].usage.completion_tokens == 2
    assert responses[3].usage.total_tokens == 4
    assert responses[3].usage.cached_prompt_tokens == 1
    assert not responses[0].prompt.tokens[0].HasField("logprob")
    assert responses[0].prompt.tokens[1].logprob == pytest.approx(-0.2)
    assert [candidate.token_id for candidate in responses[0].prompt.tokens[1].candidates] == [
        2,
        3,
    ]

    assert llm.generate_kwargs["inputs"] == "hello"
    assert llm.generate_kwargs["streaming"] is True
    assert llm.generate_kwargs["cache_salt"] == "tenant-a"
    assert llm.generate_kwargs["trace_headers"] == {
        "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
        "tracestate": "vendor=value",
    }
    assert llm.generate_kwargs["sampling_params"].temperature == pytest.approx(0.7)
    assert llm.generate_kwargs["sampling_params"].max_tokens == 2


def test_generate_sets_usage_only_on_last_simultaneous_finish() -> None:
    """Simultaneous n>1 completions attach cumulative usage only to the final event."""
    outputs = [
        SimpleNamespace(
            index=index,
            token_ids=[10 + index],
            text=str(index),
            logprobs=[],
            prompt_logprobs=[],
            finish_reason="length",
            stop_reason=None,
        )
        for index in range(2)
    ]
    result = SimpleNamespace(prompt_token_ids=[1], outputs=outputs, cached_tokens=0, error=None)
    llm = _FakeLlm([result])
    servicer = OpenEngineInferenceServicer(llm, model="test-model")
    request = generation_pb2.GenerateRequest(
        request_id="request-2",
        model="test-model",
        token_ids=generation_pb2.TokenIds(ids=[1]),
        sampling=generation_pb2.SamplingParams(num_sequences=2),
    )

    async def collect_responses() -> list[generation_pb2.GenerateResponse]:
        return [response async for response in servicer.Generate(request, _FakeContext())]

    responses = asyncio.run(collect_responses())
    finished = [response for response in responses if response.HasField("finished")]

    assert [response.finished.output_index for response in finished] == [0, 1]
    assert not finished[0].HasField("usage")
    assert finished[1].HasField("usage")
    assert finished[1].usage.completion_tokens == 2


def test_generate_does_not_stream_excluded_stop_prefix() -> None:
    """A partial multi-token stop is withheld until the terminal update removes it."""
    partial_stop = SimpleNamespace(
        index=0,
        token_ids=[10],
        text="A",
        logprobs=[],
        prompt_logprobs=[],
        finish_reason=None,
        stop_reason=None,
    )
    stopped = SimpleNamespace(
        index=0,
        token_ids=[],
        text="",
        logprobs=[],
        prompt_logprobs=[],
        finish_reason="stop",
        stop_reason="AB",
    )
    results = [
        SimpleNamespace(prompt_token_ids=[1], outputs=[partial_stop], cached_tokens=0, error=None),
        SimpleNamespace(prompt_token_ids=[1], outputs=[stopped], cached_tokens=0, error=None),
    ]
    servicer = OpenEngineInferenceServicer(_FakeLlm(results), model="test-model")
    request = generation_pb2.GenerateRequest(
        request_id="request-3",
        model="test-model",
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(
            conditions=[generation_pb2.StopCondition(stop_text="AB")]
        ),
    )

    async def collect_responses() -> list[generation_pb2.GenerateResponse]:
        return [response async for response in servicer.Generate(request, _FakeContext())]

    responses = asyncio.run(collect_responses())

    assert [response.WhichOneof("event") for response in responses] == ["finished"]
    assert responses[0].finished.stop_match.stop_text == "AB"


def test_generate_aborts_when_response_stream_closes() -> None:
    """Closing an accepted response stream aborts unfinished engine work."""
    output = SimpleNamespace(
        index=0,
        token_ids=[10],
        text="A",
        logprobs=[],
        prompt_logprobs=[],
        finish_reason=None,
        stop_reason=None,
    )
    result = SimpleNamespace(prompt_token_ids=[1], outputs=[output], cached_tokens=0, error=None)
    llm = _FakeLlm([result])
    servicer = OpenEngineInferenceServicer(llm, model="test-model")
    request = generation_pb2.GenerateRequest(
        request_id="request-4",
        model="test-model",
        prompt="hello",
    )

    async def close_after_first_response() -> None:
        responses = servicer.Generate(request, _FakeContext())
        response = await responses.__anext__()
        assert response.HasField("token")
        await responses.aclose()

    asyncio.run(close_after_first_response())

    assert llm.result_handle.aborted


def test_generate_aborts_stalled_response_consumer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stalled response consumer cannot leave engine output buffering indefinitely."""
    output = SimpleNamespace(
        index=0,
        token_ids=[10],
        text="A",
        logprobs=[],
        prompt_logprobs=[],
        finish_reason=None,
        stop_reason=None,
    )
    result = SimpleNamespace(prompt_token_ids=[1], outputs=[output], cached_tokens=0, error=None)
    llm = _FakeLlm([result])
    servicer = OpenEngineInferenceServicer(llm, model="test-model")
    request = generation_pb2.GenerateRequest(
        request_id="request-5",
        model="test-model",
        prompt="hello",
    )
    monkeypatch.setattr(openengine_servicer, "_RESPONSE_STALL_TIMEOUT_SECONDS", 0.0)

    async def stall_after_first_response() -> generation_pb2.GenerateResponse:
        responses = servicer.Generate(request, _FakeContext())
        first_response = await responses.__anext__()
        assert first_response.HasField("token")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        stalled_response = await responses.__anext__()
        await responses.aclose()
        return stalled_response

    response = asyncio.run(stall_after_first_response())

    assert llm.result_handle.aborted
    assert response.error.code == openengine_servicer.error_pb2.ERROR_CODE_OVERLOADED
    assert response.error.retryable


def test_generate_does_not_abort_slow_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow engine (long gap before output) is not misread as a stalled consumer."""
    output = SimpleNamespace(
        index=0,
        token_ids=[10],
        text="A",
        logprobs=[],
        prompt_logprobs=[],
        finish_reason="stop",
        stop_reason=None,
    )
    result = SimpleNamespace(
        prompt_token_ids=[1], outputs=[output], cached_tokens=0, error=None, finished=True
    )

    class _SlowEngineHandle:
        def __init__(self) -> None:
            self.aborted = False

        async def __aiter__(self):
            # No output for longer than the stall timeout; nothing is pending
            # delivery, so a correct watchdog must not fire.
            await asyncio.sleep(0.05)
            yield result

        def abort(self) -> None:
            self.aborted = True

    handle = _SlowEngineHandle()

    def _generate_async(**kwargs):
        kwargs["sampling_params"]._validate()
        return handle

    llm = SimpleNamespace(
        tokenizer=_FakeTokenizer(), result_handle=handle, generate_async=_generate_async
    )
    servicer = OpenEngineInferenceServicer(llm, model="test-model")
    request = generation_pb2.GenerateRequest(
        request_id="slow-1", model="test-model", prompt="hello"
    )
    monkeypatch.setattr(openengine_servicer, "_RESPONSE_STALL_TIMEOUT_SECONDS", 0.02)

    async def collect() -> list:
        return [r async for r in servicer.Generate(request, _FakeContext())]

    responses = asyncio.run(collect())

    assert not handle.aborted
    assert [r.WhichOneof("event") for r in responses] == ["token", "finished"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("openengine-priority", "not-an-integer"),
        ("openengine-target-dp-rank", "-1"),
    ],
)
def test_generate_rejects_malformed_numeric_metadata(key: str, value: str) -> None:
    """Malformed OpenEngine numeric metadata returns INVALID_ARGUMENT."""
    servicer = OpenEngineInferenceServicer(_FakeLlm([]), model="test-model")
    context = _FakeContext(metadata=((key, value),))
    request = generation_pb2.GenerateRequest(
        request_id="request-6",
        model="test-model",
        prompt="hello",
    )

    async def collect_responses() -> None:
        async for _ in servicer.Generate(request, context):
            pass

    with pytest.raises(_AbortError):
        asyncio.run(collect_responses())

    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT


def test_generate_context_only_ends_at_prefill_ready() -> None:
    """A context request terminates with PrefillReady and no `finished` event.

    The handoff is the terminal event for the context leg. A `finished` after it
    reads as "the request is complete" to a client, which drops the handoff so
    the decode leg never runs.
    """
    disagg = openengine_servicer.DisaggregatedParams(
        request_type="context_only",
        ctx_request_id=99,
        ctx_info_endpoint="tcp://1.2.3.4:5555",
        first_gen_tokens=[10],
    )
    streaming_output = SimpleNamespace(
        index=0,
        token_ids=[10],
        text="A",
        logprobs=[],
        finish_reason=None,
        stop_reason=None,
        disaggregated_params=None,
    )
    final_output = SimpleNamespace(
        index=0,
        token_ids=[10],
        text="A",
        logprobs=[],
        finish_reason="length",
        stop_reason=None,
        disaggregated_params=disagg,
    )
    results = [
        SimpleNamespace(
            prompt_token_ids=[1, 2], outputs=[streaming_output], cached_tokens=0, error=None
        ),
        SimpleNamespace(
            prompt_token_ids=[1, 2], outputs=[final_output], cached_tokens=0, error=None
        ),
    ]
    servicer = OpenEngineInferenceServicer(_FakeLlm(results), model="test-model")
    request = generation_pb2.GenerateRequest(
        request_id="request-ctx",
        model="test-model",
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(max_tokens=1),
    )
    request.extra.update({"request_type": "context_only"})

    async def collect_responses() -> list[generation_pb2.GenerateResponse]:
        return [response async for response in servicer.Generate(request, _FakeContext())]

    responses = asyncio.run(collect_responses())
    events = [response.WhichOneof("event") for response in responses]

    assert events.count("prefill_ready") == 1
    assert "finished" not in events
    assert events[-1] == "prefill_ready"
    assert responses[-1].prefill_ready.kv_session.session_id == "99"


def _set_priority_metadata(request: generation_pb2.GenerateRequest) -> None:
    del request


def _set_lora_name(request: generation_pb2.GenerateRequest) -> None:
    request.lora_name = "adapter-a"


def _set_media(request: generation_pb2.GenerateRequest) -> None:
    request.media.add(url="http://example.invalid/img.png")


def _set_bypass_prefix_cache(request: generation_pb2.GenerateRequest) -> None:
    request.kv.bypass_prefix_cache = True


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("priority", _set_priority_metadata),
        ("lora_name", _set_lora_name),
        ("media", _set_media),
        ("bypass_prefix_cache", _set_bypass_prefix_cache),
    ],
)
def test_generate_rejects_unsupported_features_as_unimplemented(field, mutate) -> None:
    """A well-formed request for a feature this adapter cannot map is UNIMPLEMENTED.

    INVALID_ARGUMENT would tell the client its request is malformed and not worth
    retrying elsewhere; UNIMPLEMENTED says this engine cannot serve it, which is
    what lets a router try another worker. UnsupportedFeatureError subclasses
    ValueError, so the two map to the same status if the except arms are ordered
    wrongly.
    """
    servicer = OpenEngineInferenceServicer(_FakeLlm([]), model="test-model")
    # Priority arrives as metadata rather than a request field.
    metadata = (("openengine-priority", "5"),) if field == "priority" else ()
    context = _FakeContext(metadata=metadata)
    request = generation_pb2.GenerateRequest(
        request_id="request-unsupported",
        model="test-model",
        prompt="hello",
    )
    mutate(request)

    async def collect_responses() -> None:
        async for _ in servicer.Generate(request, context):
            pass

    with pytest.raises(_AbortError):
        asyncio.run(collect_responses())

    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED


def test_first_gen_log_probs_clamp_masked_candidates() -> None:
    """Masked candidates carry -inf, which a protobuf Struct cannot hold.

    Guided decoding masks disallowed tokens to -inf. The handoff packs the first
    generated position's logprobs into a Struct, so an unclamped -inf makes the
    whole PrefillReady unserializable and the disaggregated request fails.
    """
    positions = [{7: _logprob(float("-inf"), 2), 9: _logprob(-0.5, 1)}, float("-inf")]

    packed = openengine_servicer._serialize_first_gen_log_probs(positions)

    assert packed[0][0][1] == openengine_servicer._MIN_LOGPROB
    assert packed[1] == openengine_servicer._MIN_LOGPROB

    params = openengine_servicer.DisaggregatedParams(
        request_type="context_only",
        ctx_request_id=99,
        ctx_info_endpoint="tcp://1.2.3.4:5555",
        first_gen_log_probs=positions,
    )
    resp = openengine_servicer._prefill_ready_response("req-1", params, transfer_backend="NIXL")
    stored = resp.prefill_ready.kv_session.attributes_struct["first_gen_log_probs"]
    assert stored[1] == openengine_servicer._MIN_LOGPROB

    restored = openengine_servicer._deserialize_first_gen_log_probs(packed)
    assert restored[0][7].logprob == openengine_servicer._MIN_LOGPROB
    assert restored[0][9].logprob == -0.5


def test_stalled_stream_cleanup_does_not_untrack_a_resubmission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled stream's cleanup must not untrack a newer request reusing its id.

    The watchdog drops the id so the client can retry, but the stalled generator
    is still suspended and its `finally` runs later. If that cleanup is not
    identity-guarded it removes the *replacement*, so Control.Abort reports the
    live request as already finished and shutdown skips it.
    """

    def make_result() -> SimpleNamespace:
        output = SimpleNamespace(
            index=0,
            token_ids=[10],
            text="A",
            logprobs=[],
            prompt_logprobs=[],
            finish_reason=None,
            stop_reason=None,
        )
        return SimpleNamespace(prompt_token_ids=[1], outputs=[output], cached_tokens=0, error=None)

    class _PerCallHandleLlm:
        """Unlike _FakeLlm, hands out a distinct handle per call."""

        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()
            self.handles: list[_FakeResultHandle] = []

        def generate_async(self, **kwargs: Any) -> _FakeResultHandle:
            kwargs["sampling_params"]._validate()
            handle = _FakeResultHandle([make_result()])
            self.handles.append(handle)
            return handle

    llm = _PerCallHandleLlm()
    servicer = OpenEngineInferenceServicer(llm, model="test-model")
    request = generation_pb2.GenerateRequest(request_id="dup", model="test-model", prompt="hello")

    async def scenario() -> int:
        monkeypatch.setattr(openengine_servicer, "_RESPONSE_STALL_TIMEOUT_SECONDS", 0.0)
        first = servicer.Generate(request, _FakeContext())
        assert (await first.__anext__()).HasField("token")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # The watchdog released the id so the client may retry it.
        assert servicer.active_request_count() == 0

        # Large timeout so the replacement's own watchdog cannot fire here.
        monkeypatch.setattr(openengine_servicer, "_RESPONSE_STALL_TIMEOUT_SECONDS", 3600.0)
        second = servicer.Generate(request, _FakeContext())
        assert (await second.__anext__()).HasField("token")
        assert servicer.active_request_count() == 1

        await first.aclose()
        tracked_after_cleanup = servicer.active_request_count()
        aborted = servicer.abort_request_by_id("dup")
        await second.aclose()
        assert aborted, "the replacement must still be abortable"
        return tracked_after_cleanup

    assert asyncio.run(scenario()) == 1
    # The replacement's handle was aborted, not the stalled one's.
    assert llm.handles[1].aborted
