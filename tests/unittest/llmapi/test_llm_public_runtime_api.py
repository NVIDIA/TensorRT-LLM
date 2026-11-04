# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production tests of the public runtime API the serving frontends use.

Covers the health members (BaseLLM.check_health(), fatal_error,
shutdown_started, engine_dead and the executor-side equivalents), the runtime
accessors the frontends read (record_fatal_error, multi_frontend_attach_info,
multi_frontend_ipc_dir, engine_in_worker_process, hf_model_dir,
hf_model_config, resource_governor_queue, encode_limits, get_kv_cache_load,
get_openengine_discovery) and the EngineClient protocol conformance of LLM,
against the real classes (no mirrors; the LLM objects are built with
object.__new__ plus exactly the attributes BaseLLM.__init__ sets).
"""

import inspect
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy, GenerationExecutorProxy
from tensorrt_llm.llmapi.engine_client import EngineClient
from tensorrt_llm.llmapi.llm import LLM, EncodeLimits

pytestmark = pytest.mark.cpu_only


class _NoOpExecutor(GenerationExecutor):
    """Minimal concrete GenerationExecutor; health logic comes from the base."""

    def submit(self, request):
        raise NotImplementedError("health tests never submit requests")

    def abort_request(self, request_id):
        pass

    def shutdown(self):
        self.doing_shutdown = True


def _llm(executor, encode_only=False, encoder_executor=None):
    """An LLM with exactly the attributes BaseLLM.__init__ sets for health."""
    llm = object.__new__(LLM)
    llm._executor = executor
    llm._encode_only = encode_only
    llm._encoder_executor = encoder_executor
    return llm


def _proxy(engine_dead=False, fatal_error=None):
    """A proxy with exactly the two fields its engine_dead property reads."""
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy._engine_dead = engine_dead
    proxy._fatal_error = fatal_error
    return proxy


# ---------------------------------------------------------------------------
# BaseLLM.check_health()
# ---------------------------------------------------------------------------
def test_check_health_encode_only_with_encoder_executor():
    assert _llm(None, encode_only=True, encoder_executor=_NoOpExecutor()).check_health() is True


def test_check_health_encode_only_without_encoder_executor():
    assert _llm(None, encode_only=True, encoder_executor=None).check_health() is False


def test_check_health_healthy_executor():
    assert _llm(_NoOpExecutor()).check_health() is True


def test_check_health_fatal_error():
    executor = _NoOpExecutor()
    executor._set_fatal_error(RuntimeError("CUDA OOM"))
    assert _llm(executor).check_health() is False


def test_check_health_doing_shutdown():
    executor = _NoOpExecutor()
    executor.doing_shutdown = True
    assert _llm(executor).check_health() is False


def test_check_health_without_executor():
    assert _llm(None).check_health() is False


def test_check_health_drains_queued_fatal_error():
    """A fatal error still in the error queue is promoted by check_health()."""
    executor = _NoOpExecutor()
    queued = RuntimeError("engine died in a background thread")
    executor._error_queue.put(queued)

    llm = _llm(executor)
    assert llm.check_health() is False
    assert llm.fatal_error is queued
    assert llm.shutdown_started is True


def test_check_health_private_alias_is_the_same_function():
    """Kept for one release for out-of-tree callers that look the name up."""
    assert LLM._check_health is LLM.check_health


# ---------------------------------------------------------------------------
# BaseLLM.fatal_error
# ---------------------------------------------------------------------------
def test_fatal_error_none_without_executor():
    assert _llm(None).fatal_error is None


def test_fatal_error_none_on_healthy_executor():
    assert _llm(_NoOpExecutor()).fatal_error is None


def test_fatal_error_returns_recorded_exception_first_one_wins():
    executor = _NoOpExecutor()
    first = RuntimeError("first fatal error")
    executor._set_fatal_error(first)
    executor._set_fatal_error(RuntimeError("second fatal error"))
    assert _llm(executor).fatal_error is first


# ---------------------------------------------------------------------------
# BaseLLM.shutdown_started
# ---------------------------------------------------------------------------
def test_shutdown_started_follows_doing_shutdown():
    executor = _NoOpExecutor()
    llm = _llm(executor)
    assert llm.shutdown_started is False
    executor.doing_shutdown = True
    assert llm.shutdown_started is True


def test_shutdown_started_true_without_executor():
    assert _llm(None).shutdown_started is True


def test_shutdown_started_false_for_encode_only():
    assert _llm(None, encode_only=True, encoder_executor=_NoOpExecutor()).shutdown_started is False


# ---------------------------------------------------------------------------
# BaseLLM.engine_dead
# ---------------------------------------------------------------------------
def test_engine_dead_false_without_executor():
    assert _llm(None).engine_dead is False


def test_engine_dead_follows_the_executor():
    executor = _NoOpExecutor()
    llm = _llm(executor)
    assert llm.engine_dead is False
    executor._set_fatal_error(RuntimeError("worker died"))
    assert llm.engine_dead is True


# ---------------------------------------------------------------------------
# GenerationExecutor.fatal_error and engine_dead
# ---------------------------------------------------------------------------
def test_executor_fatal_error_property():
    executor = _NoOpExecutor()
    assert executor.fatal_error is None
    error = RuntimeError("fatal")
    executor._set_fatal_error(error)
    assert executor.fatal_error is error


def test_executor_engine_dead_property():
    executor = _NoOpExecutor()
    assert executor.engine_dead is False
    executor._set_fatal_error(RuntimeError("fatal"))
    assert executor.engine_dead is True


@pytest.mark.parametrize(
    "engine_dead,fatal_error,expected",
    [
        (False, None, False),
        (True, None, True),
        (False, RuntimeError("fatal"), True),
        (True, RuntimeError("fatal"), True),
    ],
)
def test_proxy_engine_dead_reads_both_fields(engine_dead, fatal_error, expected):
    """_mark_engine_dead can set the sticky flag before the error is recorded."""
    assert _proxy(engine_dead=engine_dead, fatal_error=fatal_error).engine_dead is expected


# ---------------------------------------------------------------------------
# BaseLLM.record_fatal_error
# ---------------------------------------------------------------------------
def test_record_fatal_error_without_executor_is_a_no_op():
    assert _llm(None).record_fatal_error(RuntimeError("lost")) is False


def test_record_fatal_error_records_on_the_executor_first_one_wins():
    llm = _llm(_NoOpExecutor())
    first = RuntimeError("first fatal error")
    assert llm.record_fatal_error(first) is True
    assert llm.record_fatal_error(RuntimeError("second fatal error")) is True
    assert llm.fatal_error is first


# ---------------------------------------------------------------------------
# BaseLLM.multi_frontend_attach_info and multi_frontend_ipc_dir
# ---------------------------------------------------------------------------
def test_multi_frontend_attach_info_returns_the_proxy_payload():
    proxy = _proxy()
    payload = {"mode": "classic"}
    proxy.multi_frontend_attach_info = lambda: payload
    assert _llm(proxy).multi_frontend_attach_info() is payload


def test_multi_frontend_attach_info_raises_for_a_single_frontend_proxy():
    """The real proxy method returns None outside multi-frontend mode."""
    proxy = _proxy()
    proxy._num_frontends = 1
    with pytest.raises(
        ValueError,
        match=(
            "num_serve_frontends > 1 requires the classic IPC executor "
            "proxy in multi-frontend mode, got GenerationExecutorProxy"
        ),
    ):
        _llm(proxy).multi_frontend_attach_info()


def test_multi_frontend_attach_info_raises_for_a_non_proxy_executor():
    with pytest.raises(ValueError, match="got _NoOpExecutor"):
        _llm(_NoOpExecutor()).multi_frontend_attach_info()


def test_multi_frontend_attach_info_raises_without_an_executor():
    with pytest.raises(ValueError, match="got NoneType"):
        _llm(None).multi_frontend_attach_info()


def test_multi_frontend_ipc_dir_reads_the_proxy_directory():
    proxy = _proxy()
    proxy._multi_frontend_ipc_dir = "/tmp/trtllm_frontends_x"
    assert _llm(proxy).multi_frontend_ipc_dir == "/tmp/trtllm_frontends_x"


def test_multi_frontend_ipc_dir_none_outside_multi_frontend_mode():
    assert _llm(None).multi_frontend_ipc_dir is None
    assert _llm(_NoOpExecutor()).multi_frontend_ipc_dir is None
    # An attached frontend skips GenerationExecutorProxy.__init__, so the
    # private field can be missing entirely; the property stays tolerant.
    frontend = object.__new__(GenerationExecutorFrontendProxy)
    assert _llm(frontend).multi_frontend_ipc_dir is None


# ---------------------------------------------------------------------------
# BaseLLM.engine_in_worker_process
# ---------------------------------------------------------------------------
def test_engine_in_worker_process_true_for_the_ipc_proxies():
    assert _llm(_proxy()).engine_in_worker_process is True
    frontend = object.__new__(GenerationExecutorFrontendProxy)
    assert _llm(frontend).engine_in_worker_process is True


def test_engine_in_worker_process_false_for_in_process_executors():
    assert _llm(_NoOpExecutor()).engine_in_worker_process is False
    assert _llm(None).engine_in_worker_process is False


# ---------------------------------------------------------------------------
# BaseLLM.hf_model_dir and hf_model_config
# ---------------------------------------------------------------------------
def test_hf_model_dir_returns_the_resolved_checkpoint_directory():
    llm = _llm(None)
    llm._hf_model_dir = "/models/resolved"
    llm.args = SimpleNamespace(model="org/model")
    assert llm.hf_model_dir == "/models/resolved"


def test_hf_model_dir_falls_back_to_args_model_until_resolved():
    llm = _llm(None)
    llm._hf_model_dir = None
    llm.args = SimpleNamespace(model="org/model")
    assert llm.hf_model_dir == "org/model"


def test_hf_model_config_returns_the_loaded_config():
    llm = _llm(None)
    llm._hf_model_config = None
    assert llm.hf_model_config is None
    config = SimpleNamespace(model_type="llama")
    llm._hf_model_config = config
    assert llm.hf_model_config is config


# ---------------------------------------------------------------------------
# BaseLLM.resource_governor_queue
# ---------------------------------------------------------------------------
def test_resource_governor_queue_forwards_the_executor_value():
    assert _llm(_NoOpExecutor()).resource_governor_queue is None
    proxy = _proxy()
    queue = object()
    proxy._resource_governor_queue = queue
    assert _llm(proxy).resource_governor_queue is queue


def test_resource_governor_queue_raises_without_an_executor():
    with pytest.raises(RuntimeError, match="resource_governor_queue is not available"):
        _llm(None).resource_governor_queue


def test_resource_governor_queue_attribute_error_on_a_bare_instance():
    """AttributeError (not RuntimeError) on an attribute-less instance.

    The api-stability snapshotter inspects such instances and tolerates only
    AttributeError.
    """
    with pytest.raises(AttributeError):
        object.__new__(LLM).resource_governor_queue


# ---------------------------------------------------------------------------
# BaseLLM.encode_limits and encode()
# ---------------------------------------------------------------------------
def _encoder(batch_size=4, max_num_tokens=512, max_seq_len=128):
    return SimpleNamespace(
        model_engine=SimpleNamespace(
            batch_size=batch_size, max_num_tokens=max_num_tokens, max_seq_len=max_seq_len
        )
    )


def test_encode_limits_reads_the_encoder_engine():
    llm = _llm(None, encode_only=True, encoder_executor=_encoder())
    assert llm.encode_limits == EncodeLimits(max_batch_size=4, max_num_tokens=512, max_seq_len=128)


def test_encode_limits_none_outside_encode_only_and_after_shutdown():
    assert _llm(_NoOpExecutor()).encode_limits is None
    assert _llm(None, encode_only=True, encoder_executor=None).encode_limits is None


def test_encode_rejects_a_batch_above_encode_limits():
    llm = _llm(None, encode_only=True, encoder_executor=_encoder(batch_size=2))
    with pytest.raises(ValueError, match=r"Batch size \(3\) exceeds max_batch_size"):
        llm.encode([[1], [2], [3]])


# ---------------------------------------------------------------------------
# BaseLLM.get_kv_cache_load and get_openengine_discovery
# ---------------------------------------------------------------------------
def test_kv_load_and_discovery_forward_the_executor_values():
    class _ReportingExecutor(_NoOpExecutor):
        def get_kv_cache_load(self):
            return {"usedNumBlocks": 1}

        def get_openengine_discovery(self):
            return {"node": {"engine_id": "x"}}

    llm = _llm(_ReportingExecutor())
    assert llm.get_kv_cache_load() == {"usedNumBlocks": 1}
    assert llm.get_openengine_discovery() == {"node": {"engine_id": "x"}}


def test_kv_load_and_discovery_empty_without_an_executor():
    assert _llm(None).get_kv_cache_load() == {}
    assert _llm(None).get_openengine_discovery() == {}
    # The base executor reports nothing by default.
    assert _llm(_NoOpExecutor()).get_kv_cache_load() == {}
    assert _llm(_NoOpExecutor()).get_openengine_discovery() == {}


# ---------------------------------------------------------------------------
# EngineClient conformance
# ---------------------------------------------------------------------------
_CONFORMANCE_HINT = (
    "LLM no longer satisfies EngineClient: {member}; keep "
    "tensorrt_llm/llmapi/engine_client.py and LLM in step "
    "(API review committee approves both)."
)


def test_llm_satisfies_engine_client():
    """Every public EngineClient member exists on LLM with the same kind."""
    for name, member in vars(EngineClient).items():
        if name.startswith("_"):
            continue
        llm_member = inspect.getattr_static(LLM, name, None)
        assert llm_member is not None, _CONFORMANCE_HINT.format(member=name)
        if isinstance(member, property):
            assert isinstance(llm_member, property), _CONFORMANCE_HINT.format(member=name)
        else:
            assert callable(llm_member), _CONFORMANCE_HINT.format(member=name)


def test_llm_generate_async_accepts_every_engine_client_parameter():
    """Each protocol parameter exists on LLM.generate_async with the same default.

    A protocol default of `...` is a placeholder for an implementation-defined
    value and is not compared.
    """
    protocol_params = inspect.signature(EngineClient.generate_async).parameters
    llm_params = inspect.signature(LLM.generate_async).parameters
    for name, parameter in protocol_params.items():
        assert name in llm_params, _CONFORMANCE_HINT.format(member=f"generate_async({name})")
        if parameter.default in (inspect.Parameter.empty, Ellipsis):
            continue
        assert llm_params[name].default == parameter.default, _CONFORMANCE_HINT.format(
            member=f"generate_async({name}) default"
        )
