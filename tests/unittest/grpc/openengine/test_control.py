# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the OpenEngine Control service."""

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "openengine",
    reason='OpenEngine dependency not installed (pip install "tensorrt_llm[openengine]")',
)

import grpc  # noqa: E402
from conftest import AbortError, FakeServicerContext  # noqa: E402
from openengine.v1 import kv_pb2, lifecycle_pb2, lora_pb2, model_pb2, server_pb2  # noqa: E402

from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402

# Runs on the CPU stage: the engine is stubbed, so nothing here needs a GPU.
pytestmark = pytest.mark.cpu_only

MODEL = "test-model"


class _FakeHandle:
    def __init__(self, fail: bool = False) -> None:
        self.aborted = False
        self._fail = fail

    def abort(self) -> None:
        if self._fail:
            raise RuntimeError("engine refused the abort")
        self.aborted = True


def _inference(active: dict | None = None) -> OpenEngineInferenceServicer:
    """A real inference servicer with a seeded in-flight table.

    Control is deliberately given the production servicer rather than a stand-in,
    so `active_request_count` / `abort_request_by_id` / `abort_all_requests` are
    the code actually under test -- a duplicate would pass by construction and
    would not catch a regression in the real methods.
    """
    servicer = OpenEngineInferenceServicer(_llm(), MODEL)
    servicer._active_requests.update(active or {})
    return servicer


def _llm(**overrides):
    args = SimpleNamespace(
        max_seq_len=4096,
        max_batch_size=16,
        max_num_tokens=8192,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        enable_lora=False,
        guided_decoding_backend="xgrammar",
        max_beam_width=1,
        reasoning_parser=None,
        kv_cache_config=SimpleNamespace(tokens_per_block=32),
    )
    for key, value in overrides.items():
        if key.startswith("_"):
            continue
        setattr(args, key, value)
    shutdown = overrides.get("_executor_shutdown", False)
    unhealthy = overrides.get("_executor_unhealthy", False)
    executor = SimpleNamespace(
        is_shutdown=lambda: shutdown,
        check_health=lambda: not (shutdown or unhealthy),
    )
    return SimpleNamespace(
        args=args,
        llm_id="instance-1",
        tokenizer=overrides.get("_tokenizer", object()),
        _executor=executor,
        _check_health=lambda: executor.check_health(),
    )


def _servicer(inference=None, kv_transfer_backend="NIXL", **overrides):
    return OpenEngineControlServicer(
        _llm(**overrides),
        MODEL,
        inference if inference is not None else _inference(),
        kv_transfer_backend=kv_transfer_backend,
    )


@pytest.mark.asyncio
async def test_server_info_reports_identity_parallelism_and_capacity():
    info = await _servicer().GetServerInfo(server_pb2.GetServerInfoRequest(), FakeServicerContext())

    assert info.engine_name == "tensorrt_llm"
    assert info.instance_id == "instance-1"
    assert info.supported_models == [MODEL]
    assert info.schema_revision > 0
    assert info.parallelism.tensor_parallel_size == 2
    assert info.capacity.max_running_requests == 16
    assert info.capacity.max_batched_tokens == 8192
    assert info.capacity.kv_block_size == 32
    assert info.kv_connector.enabled is True
    assert info.kv_connector.transfer_backend == "NIXL"
    # Abort(kv_session) is UNIMPLEMENTED, so cleanup must not be advertised: a
    # router that believed it would drop a prefill whose blocks stay pinned.
    assert info.kv_connector.supports_abort_cleanup is False


def test_kv_transfer_backend_detects_an_unset_backend_field():
    """cache_transceiver_config.backend is Optional and defaults to None.

    Leaving it unset is the documented way to take the default transceiver, so
    presence must be decided by the config object. Keying on the field made a
    correctly configured disagg worker advertise kv_connector.enabled=False.
    """
    from tensorrt_llm.grpc.openengine.server import _kv_transfer_backend

    unset = SimpleNamespace(
        args=SimpleNamespace(cache_transceiver_config=SimpleNamespace(backend=None))
    )
    assert _kv_transfer_backend(unset) == "DEFAULT"

    named = SimpleNamespace(
        args=SimpleNamespace(cache_transceiver_config=SimpleNamespace(backend="NIXL"))
    )
    assert _kv_transfer_backend(named) == "NIXL"

    absent = SimpleNamespace(args=SimpleNamespace(cache_transceiver_config=None))
    assert _kv_transfer_backend(absent) == ""


@pytest.mark.asyncio
async def test_server_info_omits_kv_connector_without_a_transceiver():
    info = await _servicer(kv_transfer_backend="").GetServerInfo(
        server_pb2.GetServerInfoRequest(), FakeServicerContext()
    )
    assert info.kv_connector.enabled is False


@pytest.mark.asyncio
async def test_model_info_reports_the_context_window_and_capabilities():
    info = await _servicer().GetModelInfo(
        model_pb2.GetModelInfoRequest(model=MODEL), FakeServicerContext()
    )

    assert info.model_id == MODEL
    # A client sizes its default max_tokens against this, so it must be the
    # engine's real input+output window.
    assert info.max_context_length == 4096
    assert info.generation.guided_decoding.supported is True
    assert model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA in info.generation.guided_decoding.modes
    assert info.generation.output_logprobs.supported is True
    assert info.supports_token_ids_input is True


@pytest.mark.asyncio
async def test_model_info_leaves_the_context_window_unset_when_unknown():
    """An unset limit means "unknown"; advertising zero would be a real claim."""
    info = await _servicer(max_seq_len=None).GetModelInfo(
        model_pb2.GetModelInfoRequest(), FakeServicerContext()
    )
    assert not info.HasField("max_context_length")


@pytest.mark.asyncio
async def test_model_info_reports_guided_decoding_unsupported_without_a_backend():
    info = await _servicer(guided_decoding_backend=None).GetModelInfo(
        model_pb2.GetModelInfoRequest(), FakeServicerContext()
    )
    assert info.generation.guided_decoding.supported is False
    assert list(info.generation.guided_decoding.modes) == []


@pytest.mark.asyncio
async def test_get_load_counts_in_flight_requests():
    inference = _inference({"a": _FakeHandle(), "b": _FakeHandle()})
    load = await _servicer(inference).GetLoad(server_pb2.GetLoadRequest(), FakeServicerContext())

    assert load.running_requests == 2
    assert load.instance_id == "instance-1"
    assert load.timestamp_unix_nanos > 0


@pytest.mark.asyncio
async def test_health_reports_ready_with_per_component_checks():
    response = await _servicer().Health(lifecycle_pb2.HealthRequest(), FakeServicerContext())

    assert response.state == lifecycle_pb2.HEALTH_STATE_READY
    names = {check.name for check in response.checks}
    assert {"grpc", "model", "kv_connector"} <= names
    assert all(check.state == lifecycle_pb2.HEALTH_STATE_READY for check in response.checks)


@pytest.mark.asyncio
async def test_abort_by_request_id_aborts_the_handle():
    handle = _FakeHandle()
    inference = _inference({"req-1": handle})

    response = await _servicer(inference).Abort(
        lifecycle_pb2.AbortRequest(request_id="req-1"), FakeServicerContext()
    )

    assert handle.aborted is True
    assert response.status == lifecycle_pb2.ABORT_STATUS_ABORTED


@pytest.mark.asyncio
async def test_abort_of_an_unknown_request_reports_already_finished():
    """The caller's intent already holds, so this is not an error."""
    response = await _servicer().Abort(
        lifecycle_pb2.AbortRequest(request_id="gone"), FakeServicerContext()
    )
    assert response.status == lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED


@pytest.mark.asyncio
async def test_abort_all_aborts_every_in_flight_request():
    handles = {"a": _FakeHandle(), "b": _FakeHandle()}
    inference = _inference(dict(handles))

    response = await _servicer(inference).Abort(
        lifecycle_pb2.AbortRequest(all_requests=lifecycle_pb2.AllRequests()), FakeServicerContext()
    )

    assert all(handle.aborted for handle in handles.values())
    assert response.status == lifecycle_pb2.ABORT_STATUS_ABORTED
    assert "2" in response.message


@pytest.mark.asyncio
async def test_abort_by_kv_session_is_unimplemented():
    """session_id is the engine's context request id, not a Generate request_id."""
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await _servicer().Abort(
            lifecycle_pb2.AbortRequest(kv_session=kv_pb2.KvSessionRef(session_id="7")), context
        )
    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.asyncio
async def test_abort_without_a_target_is_invalid():
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await _servicer().Abort(lifecycle_pb2.AbortRequest(), context)
    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rpc, request_message",
    [
        ("LoadLora", lora_pb2.LoadLoraRequest()),
        ("UnloadLora", lora_pb2.UnloadLoraRequest()),
        ("ListLoras", lora_pb2.ListLorasRequest()),
        ("GetKvEventSources", kv_pb2.GetKvEventSourcesRequest()),
    ],
)
async def test_unsupported_rpcs_report_unimplemented(rpc, request_message):
    """Explicit UNIMPLEMENTED, so a client can tell "not supported" from "empty"."""
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await getattr(_servicer(), rpc)(request_message, context)
    assert context.abort_code == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_lora", [False, True])
async def test_model_info_never_advertises_lora_that_generate_rejects(enable_lora):
    """Generate rejects `lora_name` unconditionally, even on an enable_lora build."""
    info = await _servicer(enable_lora=enable_lora).GetModelInfo(
        model_pb2.GetModelInfoRequest(), FakeServicerContext()
    )
    assert info.supports_lora is False


@pytest.mark.asyncio
async def test_model_info_capabilities_match_the_generate_path():
    """Advertising a capability Generate rejects turns discovery into failures."""
    info = await _servicer().GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext())

    # `openengine-priority` is rejected by Generate.
    assert info.generation.supports_priority is False
    # `kv.bypass_prefix_cache` is rejected by Generate.
    assert info.generation.supports_prefix_cache_bypass is False
    # `kv.cache_salt` is read and forwarded to generate_async.
    assert info.generation.supports_cache_salt is True
    # n > 1 is served by sampling, not beam search, so max_beam_width is not the
    # limit and no limit is claimed.
    assert not info.generation.HasField("max_num_sequences")


@pytest.mark.asyncio
async def test_abort_leaves_cleanup_to_the_generate_stream():
    """The id must stay in the table until Generate's finally removes it.

    Popping it here would let a duplicate request_id slip past the ALREADY_EXISTS
    check while the original stream is still draining.
    """
    handle = _FakeHandle()
    inference = _inference({"req-1": handle})

    await _servicer(inference).Abort(
        lifecycle_pb2.AbortRequest(request_id="req-1"), FakeServicerContext()
    )

    assert handle.aborted is True
    assert inference.active_request_count() == 1
    assert "req-1" in inference._active_requests


@pytest.mark.asyncio
async def test_abort_fails_the_rpc_when_the_engine_refuses():
    """A refused abort must not be reported as ALREADY_FINISHED.

    AbortStatus has no failure value, and the request is still running and still
    holding KV blocks -- a caller told it had finished would stop tracking it.
    """
    handle = _FakeHandle(fail=True)
    inference = _inference({"req-1": handle})
    context = FakeServicerContext()

    with pytest.raises(AbortError):
        await _servicer(inference).Abort(lifecycle_pb2.AbortRequest(request_id="req-1"), context)

    assert handle.aborted is False
    assert context.abort_code == grpc.StatusCode.INTERNAL


@pytest.mark.asyncio
async def test_abort_all_fails_the_rpc_when_any_abort_is_refused():
    """A partial failure is not a success: the refused requests keep running."""
    good, bad = _FakeHandle(), _FakeHandle(fail=True)
    inference = _inference({"good": good, "bad": bad})
    context = FakeServicerContext()

    with pytest.raises(AbortError):
        await _servicer(inference).Abort(
            lifecycle_pb2.AbortRequest(all_requests=lifecycle_pb2.AllRequests()), context
        )

    assert good.aborted is True
    assert context.abort_code == grpc.StatusCode.INTERNAL
    assert "still running" in context.abort_details


@pytest.mark.asyncio
async def test_health_inference_probe_is_bounded_and_aborts_a_stuck_request():
    """A wedged engine must yield DEGRADED, not a hung RPC or a leaked request."""
    import asyncio

    import tensorrt_llm.grpc.openengine.control as control_module

    handle = _FakeHandle()

    async def _never() -> None:
        await asyncio.sleep(3600)

    handle.aresult = _never
    llm = _llm()
    llm.generate_async = lambda *a, **k: handle

    servicer = OpenEngineControlServicer(llm, MODEL, _inference(), kv_transfer_backend="")
    original = control_module._INFERENCE_PROBE_TIMEOUT_SECONDS
    control_module._INFERENCE_PROBE_TIMEOUT_SECONDS = 0.05
    try:
        response = await servicer.Health(
            lifecycle_pb2.HealthRequest(include_inference_probe=True), FakeServicerContext()
        )
    finally:
        control_module._INFERENCE_PROBE_TIMEOUT_SECONDS = original

    probe = next(c for c in response.checks if c.name == "inference_probe")
    assert probe.state == lifecycle_pb2.HEALTH_STATE_DEGRADED
    assert response.state == lifecycle_pb2.HEALTH_STATE_DEGRADED
    # The probe request must not be left running on the engine.
    assert handle.aborted is True


@pytest.mark.asyncio
async def test_health_reports_not_ready_when_the_executor_is_shut_down():
    """A dead engine must not answer READY, or a probe keeps routing traffic in."""
    response = await _servicer(_executor_shutdown=True).Health(
        lifecycle_pb2.HealthRequest(), FakeServicerContext()
    )

    model_check = next(c for c in response.checks if c.name == "model")
    assert model_check.state == lifecycle_pb2.HEALTH_STATE_NOT_READY
    assert response.state == lifecycle_pb2.HEALTH_STATE_NOT_READY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend, expects_structural_tag",
    [("xgrammar", True), ("llguidance", False)],
)
async def test_guided_modes_follow_the_configured_backend(backend, expects_structural_tag):
    """Only xgrammar builds structural tags; llguidance rejects them."""
    info = await _servicer(guided_decoding_backend=backend).GetModelInfo(
        model_pb2.GetModelInfoRequest(), FakeServicerContext()
    )

    modes = set(info.generation.guided_decoding.modes)
    assert info.generation.guided_decoding.supported is True
    assert model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA in modes
    assert (model_pb2.GUIDED_DECODING_MODE_STRUCTURAL_TAG in modes) is expects_structural_tag


@pytest.mark.asyncio
async def test_model_info_reports_the_logprob_candidate_ceiling():
    """The limit is a known enforced constant, so "unknown" would be dishonest."""
    from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS

    info = await _servicer().GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext())

    assert info.generation.output_logprobs.max_top_n == MAX_TOP_LOGPROBS
    assert info.generation.prompt_logprobs.max_top_n == MAX_TOP_LOGPROBS


@pytest.mark.asyncio
async def test_health_reports_not_ready_for_a_crashed_engine_that_is_not_shut_down():
    """`is_shutdown()` is too weak a readiness signal.

    It only covers `doing_shutdown or _fatal_error`; a dead MPI worker whose
    error is still queued leaves it False. The LLM's own health contract drains
    that queue, so readiness must come from there or a probe keeps routing
    traffic into an engine where every request fails.
    """
    response = await _servicer(_executor_unhealthy=True).Health(
        lifecycle_pb2.HealthRequest(), FakeServicerContext()
    )

    model_check = next(c for c in response.checks if c.name == "model")
    assert model_check.state == lifecycle_pb2.HEALTH_STATE_NOT_READY
    assert response.state == lifecycle_pb2.HEALTH_STATE_NOT_READY
    # The executor here is running but unhealthy, so the message must not
    # claim it shut down.
    assert "shut down" not in model_check.message


@pytest.mark.asyncio
async def test_health_survives_a_raising_health_check():
    """A probe reports state; it must not turn an engine fault into an RPC error."""
    llm = _llm()
    llm._check_health = lambda: (_ for _ in ()).throw(RuntimeError("engine gone"))
    servicer = OpenEngineControlServicer(llm, MODEL, _inference(), kv_transfer_backend="")

    response = await servicer.Health(lifecycle_pb2.HealthRequest(), FakeServicerContext())

    model_check = next(c for c in response.checks if c.name == "model")
    assert model_check.state == lifecycle_pb2.HEALTH_STATE_NOT_READY


@pytest.mark.asyncio
async def test_text_input_is_not_advertised_without_a_tokenizer():
    """A skip_tokenizer_init engine rejects every string prompt."""
    info = await _servicer(_tokenizer=None).GetModelInfo(
        model_pb2.GetModelInfoRequest(), FakeServicerContext()
    )
    assert info.supports_text_input is False
    # Token ids remain usable without a tokenizer.
    assert info.supports_token_ids_input is True


@pytest.mark.asyncio
async def test_text_input_is_advertised_when_a_tokenizer_exists():
    info = await _servicer().GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext())
    assert info.supports_text_input is True


@pytest.mark.asyncio
async def test_health_probe_is_counted_by_get_load_and_released():
    """A probe occupies a scheduler slot, so GetLoad must report it.

    A router sizing itself on running_requests would otherwise send more traffic
    to an engine whose probes are already queued behind real work.
    """
    import asyncio

    started = asyncio.Event()
    release = asyncio.Event()
    handle = _FakeHandle()

    async def _blocking() -> None:
        started.set()
        await release.wait()

    handle.aresult = _blocking
    llm = _llm()
    llm.generate_async = lambda *a, **k: handle

    inference = _inference()
    servicer = OpenEngineControlServicer(llm, MODEL, inference, kv_transfer_backend="")

    probe = asyncio.ensure_future(
        servicer.Health(
            lifecycle_pb2.HealthRequest(include_inference_probe=True), FakeServicerContext()
        )
    )
    await started.wait()
    assert inference.active_request_count() == 1

    release.set()
    await probe
    assert inference.active_request_count() == 0


@pytest.mark.asyncio
async def test_concurrent_health_probes_share_one_engine_request():
    """A readiness loop must not stack one engine request per poll."""
    import asyncio

    release = asyncio.Event()
    handle = _FakeHandle()
    calls = 0

    async def _blocking() -> None:
        await release.wait()

    handle.aresult = _blocking
    llm = _llm()

    def _generate(*a, **k):
        nonlocal calls
        calls += 1
        return handle

    llm.generate_async = _generate

    servicer = OpenEngineControlServicer(llm, MODEL, _inference(), kv_transfer_backend="")
    request = lifecycle_pb2.HealthRequest(include_inference_probe=True)
    probes = [
        asyncio.ensure_future(servicer.Health(request, FakeServicerContext())) for _ in range(5)
    ]
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(*probes)

    assert calls == 1
