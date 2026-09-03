# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agreement between the capabilities Control advertises and what Generate does.

`GetModelInfo` tells a client what it may send. If it advertises something
`Generate` rejects, capability discovery becomes a per-request failure; if it
denies something `Generate` implements, a conformant client silently gives up a
working feature. Neither shows up in an ordinary end-to-end test, because those
only exercise the paths the client already believes in.

These run against a **real** ``LLM``. A stubbed engine cannot answer the question
being asked here: a stub accepts whatever its fake ``generate_async`` accepts, so
it will happily agree with an advertisement the real engine would reject --
which is exactly the class of bug these tests exist to catch. Only the gRPC
context is faked.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, Sequence

import pytest

pytest.importorskip(
    "openengine",
    reason='OpenEngine dependency not installed (pip install "tensorrt_llm[openengine]")',
)

import grpc  # noqa: E402
import torch  # noqa: E402
from openengine.v1 import generation_pb2, lifecycle_pb2, model_pb2  # noqa: E402
from utils.llm_data import llm_models_root  # noqa: E402

from tensorrt_llm import LLM  # noqa: E402
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import (  # noqa: E402
    GUIDE_SUPPORT_BY_BACKEND,
    OpenEngineInferenceServicer,
    supported_guides,
)
from tensorrt_llm.llmapi import KvCacheConfig  # noqa: E402
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs  # noqa: E402
from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS  # noqa: E402

MODEL_NAME = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
GUIDED_BACKEND = "xgrammar"

# The GPU requirement belongs to the engine fixture, not the module: the
# pure-function tests below are the only coverage for the guide table and the
# endpoint round trip, and they must still run on a CPU runner.

_MODE_BY_GUIDE_FIELD = {
    "json_schema": model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA,
    "regex": model_pb2.GUIDED_DECODING_MODE_REGEX,
    "ebnf_grammar": model_pb2.GUIDED_DECODING_MODE_EBNF_GRAMMAR,
    "json_object": model_pb2.GUIDED_DECODING_MODE_JSON_OBJECT,
    "structural_tag": model_pb2.GUIDED_DECODING_MODE_STRUCTURAL_TAG,
    "choice": model_pb2.GUIDED_DECODING_MODE_CHOICE,
}


class _AbortError(Exception):
    """Stands in for the abort grpc.aio raises out of a servicer."""


class _FakeContext:
    """Only the gRPC context is faked; the engine behind it is real."""

    def __init__(self, metadata: Sequence[tuple[str, str]] = ()) -> None:
        self._metadata = [SimpleNamespace(key=k, value=v) for k, v in metadata]
        self.abort_code = None
        self.abort_details = None

    def invocation_metadata(self) -> list[SimpleNamespace]:
        return self._metadata

    def cancelled(self) -> bool:
        return False

    def add_done_callback(self, callback: Any) -> None:
        pass

    async def abort(self, code: grpc.StatusCode, details: str) -> None:
        self.abort_code = code
        self.abort_details = details
        raise _AbortError


@pytest.fixture(scope="module")
def llm():
    """One real engine for the module; building it dominates the runtime."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU to run a real engine")
    model_path = llm_models_root(check=True) / MODEL_NAME
    engine = LLM(
        model=str(model_path),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        guided_decoding_backend=GUIDED_BACKEND,
        max_batch_size=4,
        max_seq_len=1024,
    )
    try:
        yield engine
    finally:
        engine.shutdown()


@pytest.fixture(scope="module")
def model_info(llm) -> model_pb2.ModelInfo:
    control = OpenEngineControlServicer(
        llm, MODEL_NAME, OpenEngineInferenceServicer(llm, MODEL_NAME)
    )
    return asyncio.run(
        control.GetModelInfo(model_pb2.GetModelInfoRequest(model=MODEL_NAME), _FakeContext())
    )


_counter = iter(range(10_000))


def _request(**kw: Any) -> generation_pb2.GenerateRequest:
    base = dict(
        request_id=f"cap-{next(_counter)}",
        model=MODEL_NAME,
        token_ids=generation_pb2.TokenIds(ids=[1, 15043, 29892]),
        stopping=generation_pb2.StoppingOptions(max_tokens=4),
    )
    base.update(kw)
    return generation_pb2.GenerateRequest(**base)


def _accepted(llm, request, context: _FakeContext | None = None) -> bool:
    """Drive the real Generate against the real engine.

    Returns True when the stream completed, False when the servicer aborted it
    or the engine reported the request as failed. Because the engine is real,
    "accepted" means it actually honored the option rather than that a stub
    tolerated it.
    """
    context = context or _FakeContext()
    servicer = OpenEngineInferenceServicer(llm, MODEL_NAME)

    async def drive() -> bool:
        saw_error = False
        async for response in servicer.Generate(request, context):
            if response.HasField("error"):
                saw_error = True
        return not saw_error

    try:
        return asyncio.run(drive())
    except _AbortError:
        return False


# ---------------------------------------------------------------------------
# Request-path capabilities
# ---------------------------------------------------------------------------


def test_priority_capability_matches_generate(llm, model_info):
    """`openengine-priority` is the only portable way to express priority."""
    advertised = model_info.generation.supports_priority
    context = _FakeContext(metadata=(("openengine-priority", "5"),))
    accepted = _accepted(llm, _request(), context)

    assert advertised == accepted, (
        f"supports_priority={advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} openengine-priority "
        f"({context.abort_code})"
    )


def test_lora_capability_matches_generate(llm, model_info):
    advertised = model_info.supports_lora
    accepted = _accepted(llm, _request(lora_name="adapter-a"))

    assert advertised == accepted, (
        f"supports_lora={advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} lora_name"
    )


def test_cache_salt_capability_matches_generate(llm, model_info):
    advertised = model_info.generation.supports_cache_salt
    accepted = _accepted(llm, _request(kv=generation_pb2.KvOptions(cache_salt="tenant-a")))

    assert advertised == accepted, (
        f"supports_cache_salt={advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} kv.cache_salt"
    )


def test_prefix_cache_bypass_capability_matches_generate(llm, model_info):
    advertised = model_info.generation.supports_prefix_cache_bypass
    accepted = _accepted(llm, _request(kv=generation_pb2.KvOptions(bypass_prefix_cache=True)))

    assert advertised == accepted, (
        f"supports_prefix_cache_bypass={advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} kv.bypass_prefix_cache"
    )


def test_multimodal_capability_matches_generate(llm, model_info):
    advertised = model_info.supports_multimodal
    accepted = _accepted(
        llm,
        _request(
            media=[
                generation_pb2.MediaItem(
                    modality=generation_pb2.MODALITY_IMAGE, url="https://example.invalid/a.png"
                )
            ]
        ),
    )

    assert advertised == accepted, (
        f"supports_multimodal={advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} media"
    )


def test_text_input_capability_matches_generate(llm, model_info):
    """A tokenizer-less engine rejects string prompts; the claim must track it."""
    advertised = model_info.supports_text_input
    accepted = _accepted(
        llm,
        generation_pb2.GenerateRequest(
            request_id="cap-text",
            model=MODEL_NAME,
            prompt="Hello",
            stopping=generation_pb2.StoppingOptions(max_tokens=4),
        ),
    )

    assert advertised == accepted, (
        f"supports_text_input={advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} a string prompt"
    )


def test_token_ids_input_capability_matches_generate(llm, model_info):
    advertised = model_info.supports_token_ids_input
    accepted = _accepted(llm, _request())

    assert advertised == accepted, (
        f"supports_token_ids_input={advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} token ids"
    )


# ---------------------------------------------------------------------------
# Sampling and response-option limits
# ---------------------------------------------------------------------------


def test_logprob_ceiling_is_the_limit_the_engine_enforces(llm, model_info):
    """The advertised ceiling must be the one the engine actually rejects past."""
    caps = model_info.generation.output_logprobs
    assert caps.HasField("max_top_n"), "max_top_n unset; a client cannot discover the limit"
    limit = caps.max_top_n

    def candidates(top_n: int):
        return _request(
            response=generation_pb2.ResponseOptions(
                return_output_logprobs=True,
                output_candidates=generation_pb2.CandidateTokenSelection(top_n=top_n),
            )
        )

    assert _accepted(
        llm, candidates(limit)
    ), f"advertised max_top_n={limit} but the engine rejected it"
    assert not _accepted(
        llm, candidates(limit + 1)
    ), f"engine accepted top_n={limit + 1}, above the advertised ceiling {limit}"


def test_logprob_ceiling_matches_the_enforced_constant(model_info):
    assert model_info.generation.output_logprobs.max_top_n == MAX_TOP_LOGPROBS


def test_multi_sequence_support_matches_max_num_sequences(llm, model_info):
    """N > 1 is served by sampling, so a beam-width-derived limit would be wrong."""
    generation = model_info.generation
    accepted = _accepted(
        llm,
        _request(sampling=generation_pb2.SamplingParams(temperature=1.0, num_sequences=2)),
    )

    if generation.HasField("max_num_sequences") and generation.max_num_sequences < 2:
        assert not accepted, (
            f"advertised max_num_sequences={generation.max_num_sequences} "
            "but the engine accepted n=2"
        )
    else:
        assert accepted, "no limit advertised but the engine rejected n=2"


def test_prompt_logprobs_match_their_advertised_support(llm, model_info):
    caps = model_info.generation.prompt_logprobs
    accepted = _accepted(
        llm,
        _request(
            response=generation_pb2.ResponseOptions(
                return_prompt_logprobs=True,
                prompt_candidates=generation_pb2.CandidateTokenSelection(top_n=1),
            )
        ),
    )
    assert caps.supported == accepted, (
        f"prompt_logprobs.supported={caps.supported} but the engine "
        f"{'accepted' if accepted else 'rejected'} the request"
    )


# ---------------------------------------------------------------------------
# Guided decoding
# ---------------------------------------------------------------------------

_GUIDE_PROBES = {
    "json_schema": dict(json_schema='{"type":"object","properties":{"n":{"type":"integer"}}}'),
    "regex": dict(regex="[0-9]+"),
    "ebnf_grammar": dict(ebnf_grammar='root ::= "a"'),
    "json_object": dict(json_object=generation_pb2.JsonObjectConstraint()),
    "structural_tag": dict(
        structural_tag='{"type":"structural_tag","format":{"type":"json_schema",'
        '"json_schema":{"type":"object","properties":{"n":{"type":"integer"}}}}}'
    ),
    "choice": dict(choice=generation_pb2.ChoiceConstraint(choices=["a", "b"])),
}


@pytest.mark.parametrize("guide", sorted(_GUIDE_PROBES))
def test_each_guided_mode_is_accepted_exactly_when_advertised(llm, model_info, guide):
    """The real grammar backend decides; the advertisement must agree with it."""
    advertised = set(model_info.generation.guided_decoding.modes)
    is_advertised = _MODE_BY_GUIDE_FIELD[guide] in advertised

    accepted = _accepted(
        llm,
        _request(
            guided=generation_pb2.GuidedDecoding(**_GUIDE_PROBES[guide]),
            stopping=generation_pb2.StoppingOptions(max_tokens=8),
        ),
    )

    assert accepted == is_advertised, (
        f"{guide}: advertised={is_advertised} but the engine "
        f"{'accepted' if accepted else 'rejected'} it on {GUIDED_BACKEND}"
    )


def test_advertised_guided_modes_come_from_the_backend_table(model_info):
    """Control must advertise exactly what Generate enforces for this backend."""
    expected = {
        _MODE_BY_GUIDE_FIELD[field]
        for field in supported_guides(GUIDED_BACKEND)
        if field in _MODE_BY_GUIDE_FIELD
    }
    assert set(model_info.generation.guided_decoding.modes) == expected


@pytest.mark.parametrize("backend", sorted(GUIDE_SUPPORT_BY_BACKEND))
def test_guide_support_table_matches_each_backend(backend: str):
    """Pure-table check, so it also covers the backend this module cannot deploy.

    llguidance's matcher factory has no structural-tag case.
    """
    expected = {"json_schema", "regex", "ebnf_grammar", "json_object"}
    if backend == "xgrammar":
        expected |= {"structural_tag"}
    assert supported_guides(backend) == expected


# ---------------------------------------------------------------------------
# Engine metadata and readiness
# ---------------------------------------------------------------------------


def test_context_window_is_the_engine_sequence_limit(llm, model_info):
    assert model_info.max_context_length == llm.args.max_seq_len, (
        "max_context_length must be the engine's real input+output window; a "
        "client sizes its default max_tokens against it"
    )


def test_health_reports_ready_for_a_live_engine(llm):
    """Pins the readiness predicate against a genuinely healthy engine.

    A predicate that reads the wrong attribute would report NOT_READY here even
    though the engine is serving.
    """
    control = OpenEngineControlServicer(
        llm, MODEL_NAME, OpenEngineInferenceServicer(llm, MODEL_NAME)
    )
    response = asyncio.run(control.Health(lifecycle_pb2.HealthRequest(), _FakeContext()))

    assert response.state == lifecycle_pb2.HEALTH_STATE_READY
    model_check = next(c for c in response.checks if c.name == "model")
    assert model_check.state == lifecycle_pb2.HEALTH_STATE_READY


def test_health_inference_probe_runs_on_a_real_engine(llm):
    control = OpenEngineControlServicer(
        llm, MODEL_NAME, OpenEngineInferenceServicer(llm, MODEL_NAME)
    )
    response = asyncio.run(
        control.Health(lifecycle_pb2.HealthRequest(include_inference_probe=True), _FakeContext())
    )

    probe = next(c for c in response.checks if c.name == "inference_probe")
    assert probe.state == lifecycle_pb2.HEALTH_STATE_READY, probe.message


def test_control_reads_only_engine_arguments_that_exist():
    """Guard against silent drift when an LlmArgs field is renamed upstream.

    Control reads these through `getattr(..., None)`, so a rename does not raise
    -- it quietly turns the reported field into "unknown". Pin the names here so
    the break is visible at test time instead of in a deployment.
    """
    read_by_control = {
        "max_seq_len",
        "max_batch_size",
        "max_num_tokens",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "guided_decoding_backend",
        "reasoning_parser",
        "kv_cache_config",
    }
    missing = sorted(read_by_control - set(TorchLlmArgs.model_fields))
    assert not missing, f"Control reads LlmArgs fields that no longer exist: {missing}"


def test_guided_requests_are_rejected_without_a_grammar_backend():
    """An engine with no grammar backend never builds the grammar.

    Accepting the request would stream unconstrained text as a success, so the
    client would receive schema-violating output believing it was constrained.
    """
    assert supported_guides(None) == frozenset()
    assert supported_guides("") == frozenset()


def test_an_unrecognised_grammar_backend_grants_no_guides():
    """Failing closed keeps a future backend from inheriting xgrammar's set."""
    assert supported_guides("some-future-backend") == frozenset()


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://10.0.0.7:38693",
        "10.0.0.7:38693",
        "tcp://[fe80::1]:38693",
    ],
)
def test_ctx_info_endpoint_survives_the_handoff_round_trip(endpoint):
    """The generation worker connects a ZMQ socket to this string verbatim.

    Splitting it on the first colon takes the scheme for the host and drops the
    rest, which reaches ZMQ as `addr='tcp'`.
    """
    from tensorrt_llm.grpc.openengine.servicer import (
        _disaggregated_params_from_request,
        _prefill_ready_response,
    )

    ctx_params = SimpleNamespace(
        ctx_request_id=7,
        ctx_dp_rank=0,
        ctx_info_endpoint=endpoint,
        opaque_state=None,
        first_gen_tokens=[1],
        draft_tokens=None,
        disagg_request_id=None,
        schedule_style=None,
        ctx_usage=None,
        first_gen_log_probs=None,
    )
    ready = _prefill_ready_response("req-1", ctx_params, "NIXL")

    decoded = _disaggregated_params_from_request(
        generation_pb2.GenerateRequest(
            request_id="req-2",
            model=MODEL_NAME,
            token_ids=generation_pb2.TokenIds(ids=[1]),
            kv=generation_pb2.KvOptions(session=ready.prefill_ready.kv_session),
        )
    )
    assert decoded.ctx_info_endpoint == endpoint


def test_an_unsupported_load_bearing_handoff_field_fails_loudly():
    """Dropping a load-bearing field would decode against missing state.

    Only fields that come and go across releases may be dropped; anything the
    handoff depends on must fail rather than silently produce a wrong answer.
    """
    from tensorrt_llm.grpc.openengine.servicer import _set_disagg_attr

    class _NoEndpoint:
        __dataclass_fields__ = {"request_type": None}

    params = _NoEndpoint()
    with pytest.raises(ValueError, match="ctx_info_endpoint"):
        _set_disagg_attr(params, "ctx_info_endpoint", "tcp://h:1")

    # Cross-release metadata is dropped instead.
    _set_disagg_attr(params, "conversation_id", "abc", optional=True)


# ---------------------------------------------------------------------------
# Disaggregation handoff safety
# ---------------------------------------------------------------------------


def _session(**kw):
    from google.protobuf import struct_pb2
    from openengine.v1 import kv_pb2

    attrs = struct_pb2.Struct()
    attrs.update(kw.pop("attributes", {}))
    return kv_pb2.KvSessionRef(attributes_struct=attrs, **kw)


def _decode_request(session):
    return generation_pb2.GenerateRequest(
        request_id="gen-1",
        model=MODEL_NAME,
        token_ids=generation_pb2.TokenIds(ids=[1]),
        kv=generation_pb2.KvOptions(session=session),
    )


@pytest.mark.parametrize(
    ("session", "reason"),
    [
        (_session(session_id="7"), "neither a ctx_info_endpoint nor an opaque_state"),
        (
            _session(session_id="7", attributes={"ctx_info_endpoint": "tcp://nohost"}),
            "unusable address",
        ),
        (
            _session(attributes={"ctx_info_endpoint": "tcp://h:1"}),
            "neither a session_id nor a disagg_request_id",
        ),
    ],
)
def test_an_unusable_kv_session_is_rejected_before_the_engine(session, reason):
    """The engine consumes the handoff with no timeout on its executor loop.

    A session naming an unreachable peer blocks that loop indefinitely, so every
    later request on the engine stalls too. Rejecting here turns an engine-wide
    hang into one INVALID_ARGUMENT.
    """
    from tensorrt_llm.grpc.openengine.servicer import _disaggregated_params_from_request

    with pytest.raises(ValueError):
        _disaggregated_params_from_request(_decode_request(session))


def test_a_usable_kv_session_is_accepted():
    from tensorrt_llm.grpc.openengine.servicer import _disaggregated_params_from_request

    params = _disaggregated_params_from_request(
        _decode_request(
            _session(session_id="7", attributes={"ctx_info_endpoint": "tcp://10.0.0.7:1"})
        )
    )
    assert params.request_type == "generation_only"
    assert params.ctx_info_endpoint == "tcp://10.0.0.7:1"


def test_logprobs_stay_aligned_when_the_peer_omitted_the_first_one():
    """The engine tolerates being one logprob short and only warns.

    Slicing positionally against that attributes every logprob to the token
    after the one it describes -- a wrong answer rather than an error.
    """
    from tensorrt_llm.grpc.openengine.servicer import _aligned_logprobs

    output = SimpleNamespace(token_ids=[10, 11, 12], logprobs=["b", "c"])
    assert _aligned_logprobs(output) == [None, "b", "c"]

    aligned = SimpleNamespace(token_ids=[10, 11], logprobs=["a", "b"])
    assert _aligned_logprobs(aligned) == ["a", "b"]

    # No logprobs requested at all stays empty rather than becoming all-None.
    assert _aligned_logprobs(SimpleNamespace(token_ids=[10], logprobs=[])) == []


def test_a_context_request_reports_its_usage():
    """PrefillReady is the context request's final response, so usage rides on it.

    Without this the context phase reports no usage at all -- its terminal event
    is suppressed -- and a caller has nothing to attribute the prompt tokens to.
    """
    from tensorrt_llm.grpc.openengine.servicer import _prefill_ready_response, _usage

    result = SimpleNamespace(
        prompt_token_ids=[1, 2, 3, 4],
        outputs=[SimpleNamespace(token_ids=[9])],
        cached_tokens=2,
    )
    params = SimpleNamespace(
        ctx_request_id=7,
        ctx_dp_rank=0,
        ctx_info_endpoint="tcp://10.0.0.7:1",
        opaque_state=None,
        first_gen_tokens=[9],
        draft_tokens=None,
        disagg_request_id=None,
        schedule_style=None,
        ctx_usage=None,
        first_gen_log_probs=None,
    )

    response = _prefill_ready_response("r", params, "NIXL", usage=_usage(result))

    assert response.usage.prompt_tokens == 4
    assert response.usage.cached_prompt_tokens == 2
