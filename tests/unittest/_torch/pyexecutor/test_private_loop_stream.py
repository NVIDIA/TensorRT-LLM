# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The executor loop leaves the legacy default stream when the KV connector asks.

A transfer engine that host-waits on its caller's per-thread default stream
(Mooncake's NVLink transport) otherwise waits for every forward pass in flight,
because the loop makes its current stream wait for the execution stream and
per-thread default streams are ordered behind the legacy one.
"""

import threading
import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _executor(wants, *, armed=False, connector=True):
    worker = types.SimpleNamespace(wants_private_loop_stream=wants)
    ex = types.SimpleNamespace(
        kv_connector_manager=types.SimpleNamespace(worker=worker) if connector else None,
        _execution_stream_loop_armed=armed,
        drafter=None,
        draft_model_engine=None,
        guided_decoder=None,
        is_encoder_decoder=False,
        _mm_encoder_item_scheduling_enabled=False,
        execution_stream=torch.cuda.Stream() if torch.cuda.is_available() else None,
        _loop_on_execution_stream=False,
    )
    ex._private_loop_stream_scope = lambda: PyExecutor._private_loop_stream_scope(ex)
    return ex


def _scope(ex, overlap):
    if overlap:
        return PyExecutor._overlap_loop_stream_scope(ex)
    return PyExecutor._private_loop_stream_scope(ex)


@cuda
@pytest.mark.parametrize("overlap", [True, False])
def test_connector_opt_in_moves_the_loop_to_a_private_stream(overlap):
    outer = torch.cuda.current_stream()
    ex = _executor(True)
    with _scope(ex, overlap):
        inner = torch.cuda.current_stream()
        assert inner != outer
        assert inner != torch.cuda.default_stream()
        assert inner != ex.execution_stream
        # Every forward keeps both explicit waits.
        assert ex._loop_on_execution_stream is False
    assert torch.cuda.current_stream() == outer


@cuda
@pytest.mark.parametrize(
    "wants, connector", [(False, True), (True, False)], ids=["no_opt_in", "no_connector"]
)
@pytest.mark.parametrize("overlap", [True, False])
def test_loop_stays_on_the_current_stream_without_opt_in(wants, connector, overlap):
    outer = torch.cuda.current_stream()
    with _scope(_executor(wants, connector=connector), overlap):
        assert torch.cuda.current_stream() == outer


@cuda
def test_execution_stream_loop_still_wins_without_a_connector():
    ex = _executor(False, armed=True, connector=False)
    with _scope(ex, True):
        assert torch.cuda.current_stream() == ex.execution_stream
        assert ex._loop_on_execution_stream is True
    assert ex._loop_on_execution_stream is False


@cuda
def test_private_loop_stream_is_ordered_both_ways_with_the_outer_stream():
    """Work queued before the loop runs first; work queued after it runs last."""
    x = torch.zeros(1 << 20, device="cuda")
    torch.cuda._sleep(50_000_000)
    x.add_(1)
    with _scope(_executor(True), False):
        x.mul_(3)
        torch.cuda._sleep(50_000_000)
        x.add_(2)
    x.mul_(5)
    torch.cuda.synchronize()
    assert torch.all(x == 25)


def _per_thread_stream_blocked(forward_done, wait_s=0.3):
    """Whether an event on another thread's per-thread default stream waits for this device's work.

    None when the forward (``forward_done``) had already finished when the
    event was queried, so the answer says nothing about ordering.
    """
    from cuda.bindings import runtime as cudart

    result = {}
    device = torch.cuda.current_device()  # the current device is per thread

    def probe():
        torch.cuda.set_device(device)
        err, event = cudart.cudaEventCreateWithFlags(cudart.cudaEventDisableTiming)
        assert err == cudart.cudaError_t.cudaSuccess
        per_thread = cudart.cudaStream_t(2)  # cudaStreamPerThread
        (err,) = cudart.cudaEventRecord(event, per_thread)
        assert err == cudart.cudaError_t.cudaSuccess
        threading.Event().wait(wait_s)
        (err,) = cudart.cudaEventQuery(event)
        blocked = err == cudart.cudaError_t.cudaErrorNotReady
        result["blocked"] = None if forward_done.query() else blocked
        cudart.cudaEventSynchronize(event)
        cudart.cudaEventDestroy(event)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return result["blocked"]


def _forward_then_wait(ex):
    """One loop step as _forward_step queues it: a long forward, then current waits for it.

    Returns an event that completes with the forward.
    """
    ex.execution_stream.wait_stream(torch.cuda.current_stream())
    done = torch.cuda.Event()
    with torch.cuda.stream(ex.execution_stream):
        torch.cuda._sleep(2_000_000_000)
        done.record()
    torch.cuda.current_stream().wait_stream(ex.execution_stream)
    return done


@cuda
@pytest.mark.parametrize("wants", [False, True], ids=["legacy_loop", "private_loop"])
def test_per_thread_default_stream_waits_for_the_forward_only_on_a_legacy_loop(wants):
    pytest.importorskip("cuda.bindings")
    ex = _executor(wants)
    torch.cuda.synchronize()
    with _scope(ex, True):
        blocked = _per_thread_stream_blocked(_forward_then_wait(ex))
    torch.cuda.synchronize()
    if blocked is None:
        pytest.skip("the forward finished before the probe looked")
    assert blocked is (not wants)
