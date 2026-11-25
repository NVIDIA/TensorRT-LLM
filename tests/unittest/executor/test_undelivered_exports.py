# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exports of a response that is dropped on its way to the client.

A multimodal encoder's response carries its MRoPE positions by value. Its CPU
embeddings live in ``/dev/shm`` until a consumer or ``release_dict`` opens them.
"""

import asyncio
import base64
import gc
from contextlib import closing
from pathlib import Path
from queue import Queue
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse, LlmResult, PyResult
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.base_worker import (
    AwaitResponseHelper,
    BaseWorker,
    _drop_refused_response,
)
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.postproc_worker import PostprocWorker
from tensorrt_llm.executor.proxy import GenerationExecutorProxy, _RequestSender
from tensorrt_llm.executor.result import GenerationResult, ResponseWrapper
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.utils import ErrorResponse, RequestError
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only

CLIENT_ID = 7


def _exporting_response() -> tuple[LlmResponse, set[Path]]:
    py_result = PyResult(prompt_len=1, max_new_tokens=1)
    py_result.append_mm_embeddings(torch.ones(2, 1024), [1, 1])
    py_result.set_mrope_position(torch.zeros(3, 1024), torch.zeros(1024))
    files = {
        Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
        for handle in py_result.mm_embedding_handles
    }
    response = LlmResponse(
        request_id=CLIENT_ID,
        result=LlmResult(b"", py_result, is_final=True),
        client_id=CLIENT_ID,
    )
    return response, files


def _drop_at_worker(response):
    assert _drop_refused_response(SimpleNamespace(_refused_client_ids={CLIENT_ID}), response)


def _replace_at_worker_on_background_error(response):
    errors = Queue()
    errors.put(RuntimeError("background error"))
    sent = []
    worker = SimpleNamespace(
        enable_postprocess_parallel=False,
        _refused_client_ids=set(),
        _error_queue=errors,
        frontend_result_queues=None,
        result_queue=SimpleNamespace(put=sent.extend),
        _pop_result=lambda client_id: None,
    )
    worker._has_background_error = MethodType(BaseWorker._has_background_error, worker)
    worker._create_error_response = MethodType(BaseWorker._create_error_response, worker)
    AwaitResponseHelper(worker).handle_for_ipc_batched([response])
    assert sent == [ErrorResponse(CLIENT_ID, "background error", CLIENT_ID)]


class _ResultQueue:
    def __init__(self, item):
        self._item = item

    def get(self):
        return self._item


def _drop_at_proxy(response):
    # Without __init__, which spawns workers; the attributes keep teardown a no-op.
    proxy = object.__new__(GenerationExecutorProxy)
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    proxy._results = {}
    proxy.result_queue = _ResultQueue(ResponseWrapper(response))
    assert proxy.dispatch_result_task()


class _RpcProxy(RpcExecutorMixin, GenerationExecutor):
    pass


_RpcProxy.__abstractmethods__ = frozenset()


def _drop_at_rpc_proxy(response):
    proxy = _RpcProxy.__new__(_RpcProxy)
    proxy._results = {}
    proxy.handle_responses([[response]])


@pytest.mark.parametrize(
    "drop",
    [_drop_at_worker, _replace_at_worker_on_background_error, _drop_at_proxy, _drop_at_rpc_proxy],
    ids=["refused_at_worker", "errored_at_worker", "late_at_proxy", "unknown_at_rpc_proxy"],
)
def test_a_dropped_response_releases_its_exports(drop):
    response, files = _exporting_response()
    gc.collect()
    assert all(path.exists() for path in files)

    drop(response)
    gc.collect()

    assert not any(path.exists() for path in files)


def test_partial_encoder_export_is_retired(monkeypatch):
    py_result = PyResult(prompt_len=2, max_new_tokens=1)
    files = []
    dump = SharedTensorContainer.dump_to_dict

    def fail_second(container):
        if files:
            raise ValueError("second item failed")
        handle = dump(container)
        files.append(
            Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
        )
        return handle

    monkeypatch.setattr(SharedTensorContainer, "dump_to_dict", fail_second)
    with pytest.raises(ValueError, match="second item failed"):
        py_result.append_mm_embeddings(torch.ones(2, 8), [1, 1])
    assert files[0].exists()
    py_result.release_exports()
    gc.collect()
    assert not files[0].exists()


def test_postprocessor_releases_encoder_exports_on_error():
    response, files = _exporting_response()
    worker = object.__new__(PostprocWorker)
    worker._records = {}
    worker._to_stop = asyncio.Event()
    worker._pull_pipe = SimpleNamespace(
        get_async=AsyncMock(return_value=PostprocWorker.Input(response))
    )
    worker._handle_input = AsyncMock(side_effect=ValueError("invalid completion"))

    result = worker._mainloop()
    with closing(asyncio.new_event_loop()) as loop:
        try:
            [error] = loop.run_until_complete(result.__anext__())
            assert "invalid completion" in error.error_msg
        finally:
            loop.run_until_complete(result.aclose())
    gc.collect()
    assert not any(path.exists() for path in files)


def test_proxy_releases_postprocessed_encoder_exports_for_a_missing_client():
    response, files = _exporting_response()
    params = DisaggregatedParams(multimodal_embedding_handles=response.result.mm_embedding_handles)
    output = PostprocWorker.Output(
        client_id=CLIENT_ID, res=None, is_final=True, disaggregated_params=params
    )

    _drop_at_proxy(output)
    gc.collect()
    assert not any(path.exists() for path in files)


def test_prefill_import_failure_releases_remaining_encoder_exports():
    handles = [SharedTensorContainer.from_tensor(torch.ones(1, 8)).dump_to_dict() for _ in range(3)]
    files = [
        Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
        for handle in handles
    ]
    # The second storage remains valid, but its view is malformed.
    del handles[1]["tensor_stride"]
    request = SimpleNamespace(
        id=1, disaggregated_params=DisaggregatedParams(multimodal_embedding_handles=handles)
    )
    with pytest.raises(RequestError, match="tensor_stride"):
        BaseWorker._enqueue_request(SimpleNamespace(), request)
    gc.collect()
    assert not any(path.exists() for path in files)


def test_failed_frontend_completion_releases_its_exports():
    response, files = _exporting_response()
    params = DisaggregatedParams(multimodal_embedding_handles=response.result.mm_embedding_handles)

    def fail(_):
        raise ValueError("formatter failed")

    frontend = SimpleNamespace(_handle_response=fail, _disaggregated_params=params)
    with pytest.raises(ValueError, match="formatter failed"):
        GenerationResult._consume_response(frontend, response)
    gc.collect()
    assert not any(path.exists() for path in files)
    assert params.multimodal_embedding_handles is None


@pytest.mark.parametrize(
    "failure", ["bad_request", "closed_sender", "failed_send", "worker_rejection"]
)
def test_prefill_rejection_releases_trusted_exports(failure):
    response, files = _exporting_response()
    handles = response.result.mm_embedding_handles
    handoff = DisaggregatedParams(request_type="context_only", multimodal_embedding_handles=handles)
    errors = Queue()

    def closed_socket(_):
        raise RuntimeError("closed")

    sender = _RequestSender(SimpleNamespace(put=closed_socket), errors)

    def submit(request):
        request.set_id(42)
        if failure == "worker_rejection":
            worker = SimpleNamespace(
                _lora_manager=None, engine=SimpleNamespace(kv_cache_transceiver=None)
            )
            return BaseWorker._enqueue_request(worker, request)
        if failure == "closed_sender":
            sender.close()
        return GenerationExecutorProxy._send_request(
            SimpleNamespace(_request_sender=sender), request
        )

    proxy = SimpleNamespace(_maybe_initialize_iteration_results=lambda: None, submit=submit)
    kwargs = dict(
        prompt_token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
        disaggregated_params=handoff,
        multimodal_params=MultimodalParams(multimodal_data={"multimodal_embedding": handles}),
        cache_salt=1 if failure == "bad_request" else None,
    )
    try:
        if failure == "failed_send":
            GenerationExecutor.generate_async(proxy, **kwargs)
            sender.close(timeout=10)
            assert str(errors.get_nowait()) == "closed"
        else:
            with pytest.raises((TypeError, RuntimeError, AssertionError)):
                GenerationExecutor.generate_async(proxy, **kwargs)
    finally:
        sender.close(timeout=10)
        # The in-process worker may already own views of these storages.
        kwargs["multimodal_params"].multimodal_data.clear()
    gc.collect()
    assert not any(path.exists() for path in files)
