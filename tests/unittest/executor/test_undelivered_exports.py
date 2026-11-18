# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exports of a response that is dropped on its way to the client.

A multimodal encoder's response carries its embeddings and MRoPE positions as exported
handles, which only a consumer that opens them, or ``release_dict``, consumes. The
exports here are CPU tensors: each lives in a ``/dev/shm`` file until its handles are
consumed.
"""

import base64
import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmResponse, LlmResult, PyResult
from tensorrt_llm.executor.base_worker import _drop_refused_response
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.result import ResponseWrapper
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin

pytestmark = pytest.mark.cpu_only

CLIENT_ID = 7


def _exporting_response() -> tuple[LlmResponse, set[Path]]:
    py_result = PyResult(prompt_len=1, max_new_tokens=1)
    py_result.append_mm_embeddings(torch.ones(2, 1024), [1, 1])
    py_result.set_mrope_position(torch.zeros(3, 1024), torch.zeros(1024))
    handles = [
        *py_result.mm_embedding_handles,
        py_result.mrope_position_ids_handle,
        py_result.mrope_position_deltas_handle,
    ]
    files = {
        Path("/dev/shm") / base64.b64decode(handle["storage_handle"]).decode().lstrip("/")
        for handle in handles
    }
    response = LlmResponse(
        request_id=CLIENT_ID,
        result=LlmResult(b"", py_result, is_final=True),
        client_id=CLIENT_ID,
    )
    return response, files


def _drop_at_worker(response):
    assert _drop_refused_response(SimpleNamespace(_refused_client_ids={CLIENT_ID}), response)


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
    [_drop_at_worker, _drop_at_proxy, _drop_at_rpc_proxy],
    ids=["refused_at_worker", "late_at_proxy", "unknown_at_rpc_proxy"],
)
def test_a_dropped_response_releases_its_exports(drop):
    response, files = _exporting_response()
    gc.collect()
    assert all(path.exists() for path in files)

    drop(response)
    gc.collect()

    assert not any(path.exists() for path in files)
