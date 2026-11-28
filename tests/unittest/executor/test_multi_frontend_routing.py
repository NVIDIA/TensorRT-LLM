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
"""CPU-only tests for classic-path multi-frontend serving.

Multi-frontend serving (num_serve_frontends) on the classic IPC
executor path. Covers client-id namespacing, worker-side response-lane routing, and the
attached-frontend proxy lifecycle over real ipc:// sockets.
"""

import os
import pickle
import threading
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.utils import (
    FRONTEND_COUNTER_MASK,
    MAX_NUM_FRONTENDS,
    bucket_responses_by_frontend,
    frontend_lane_index,
    get_frontend_id,
    namespace_client_id,
)

pytestmark = pytest.mark.cpu_only


class TestClientIdNamespacing:
    def test_frontend_zero_keeps_legacy_ids(self):
        for client_id in (1, 42, FRONTEND_COUNTER_MASK):
            assert namespace_client_id(0, client_id) == client_id

    def test_roundtrip(self):
        for frontend_id in (0, 1, 7, MAX_NUM_FRONTENDS - 1):
            for counter in (1, 12345, FRONTEND_COUNTER_MASK):
                client_id = namespace_client_id(frontend_id, counter)
                assert get_frontend_id(client_id) == frontend_id
                assert client_id & FRONTEND_COUNTER_MASK == counter
                # Bit 63 stays clear: ids remain positive as signed int64.
                assert 0 < client_id < (1 << 63)

    def test_knob_cap_matches_encoding(self):
        from tensorrt_llm.llmapi.llm_args import BaseLlmArgs

        meta = BaseLlmArgs.model_fields["num_serve_frontends"].metadata
        assert any(getattr(m, "le", None) == MAX_NUM_FRONTENDS for m in meta)

    def test_counter_wraparound_stays_in_namespace(self):
        # A counter overflowing its field must not leak into the frontend bits.
        client_id = namespace_client_id(3, FRONTEND_COUNTER_MASK + 5)
        assert get_frontend_id(client_id) == 3
        assert client_id & FRONTEND_COUNTER_MASK == 4

    def test_non_int_client_id_routes_to_launcher(self):
        assert get_frontend_id(None) == 0

    def test_lane_index_clamps_to_launcher(self):
        assert frontend_lane_index(namespace_client_id(2, 7), 3) == 2
        assert frontend_lane_index(namespace_client_id(7, 1), 2) == 0
        assert frontend_lane_index(None, 4) == 0


def _response(client_id):
    return SimpleNamespace(client_id=client_id)


class TestClassicResponseBucketing:
    """The classic IPC path routes batches with bucket_responses_by_frontend."""

    def test_buckets_by_namespace(self):
        responses = [
            _response(namespace_client_id(0, 1)),
            _response(namespace_client_id(1, 1)),
            _response(namespace_client_id(1, 2)),
            _response(namespace_client_id(2, 1)),
        ]
        buckets = bucket_responses_by_frontend(responses, 3)
        assert [len(b) for b in buckets] == [1, 2, 1]
        assert all(get_frontend_id(r.client_id) == 1 for r in buckets[1])

    def test_none_client_id_routes_to_launcher(self):
        # ADP dummy responses carry client_id=None: they must land in the
        # launcher's lane (which silently discards them), not raise.
        buckets = bucket_responses_by_frontend([_response(None)], 4)
        assert len(buckets[0]) == 1
        assert all(not b for b in buckets[1:])

    def test_out_of_range_frontend_routes_to_launcher(self):
        buckets = bucket_responses_by_frontend([_response(namespace_client_id(7, 1))], 2)
        assert len(buckets[0]) == 1 and not buckets[1]


class _LaneStub:
    def __init__(self):
        self.items = []

    def put(self, obj):
        self.items.append(obj)


class TestClassicSendRspLaneRouting:
    """_send_rsp selects the origin frontend's lane on the non-postproc path."""

    @staticmethod
    def _fake_worker(num_lanes):
        pops = []
        return SimpleNamespace(
            result_queue=None,
            postproc_queues=None,
            frontend_result_queues=[_LaneStub() for _ in range(num_lanes)],
            _pop_result=pops.append,
        ), pops

    def test_error_response_routes_to_origin_lane(self):
        from tensorrt_llm.executor.base_worker import _send_rsp
        from tensorrt_llm.executor.utils import ErrorResponse

        worker, pops = self._fake_worker(3)
        client_id = namespace_client_id(2, 7)
        _send_rsp(worker, ErrorResponse(client_id, "boom", 1))
        assert [len(q.items) for q in worker.frontend_result_queues] == [0, 0, 1]
        assert pops == [client_id]

    def test_none_client_id_routes_to_launcher_lane(self):
        from tensorrt_llm.executor.base_worker import _send_rsp
        from tensorrt_llm.executor.utils import ErrorResponse

        worker, _ = self._fake_worker(2)
        _send_rsp(worker, ErrorResponse(None, "adp dummy", 1))
        assert len(worker.frontend_result_queues[0].items) == 1
        assert not worker.frontend_result_queues[1].items

    def test_rsp_batch_defers_lane_selection(self):
        from tensorrt_llm.executor.base_worker import _send_rsp
        from tensorrt_llm.executor.utils import ErrorResponse

        worker, _ = self._fake_worker(2)
        rsp_batch = []
        _send_rsp(worker, ErrorResponse(namespace_client_id(1, 3), "x", 1), rsp_batch=rsp_batch)
        assert len(rsp_batch) == 1
        assert all(not q.items for q in worker.frontend_result_queues)


class TestClassicFrontendProxyEndToEnd:
    """GenerationExecutorFrontendProxy against a fake rank0 worker.

    Real ipc:// sockets: namespaced submit, cancel-on-shutdown, and --
    critically -- that an attached frontend NEVER emits the None
    engine-shutdown sentinel.
    """

    @staticmethod
    def _make_proxy_and_fake_worker(tmpdir, frontend_id=1, num_frontends=2):
        import zmq

        from tensorrt_llm.executor.ipc import IpcQueue
        from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy

        hmac_key = os.urandom(32)
        request_addr = f"ipc://{os.path.join(tmpdir, 'request.sock')}"
        result_addrs = [
            f"ipc://{os.path.join(tmpdir, f'result_{i}.sock')}" for i in range(num_frontends)
        ]
        # The fake rank0 worker binds the request ingress (PULL), exactly
        # like worker_main does in multi-frontend mode.
        worker_ingress = IpcQueue(
            (request_addr, hmac_key),
            is_server=True,
            socket_type=zmq.PULL,
            name="fake_worker_request_queue",
        )
        proxy = GenerationExecutorFrontendProxy(
            {
                "mode": "classic",
                "request_addr": request_addr,
                "result_addrs": result_addrs,
                "hmac_key": hmac_key.hex(),
            },
            frontend_id=frontend_id,
        )
        return proxy, worker_ingress, hmac_key, result_addrs

    def test_submit_namespaces_and_shutdown_never_sends_sentinel(
        self, tmp_path, monkeypatch
    ) -> None:
        from tensorrt_llm.executor.request import CancellingRequest, GenerationRequest
        from tensorrt_llm.executor.utils import RequestError
        from tensorrt_llm.sampling_params import SamplingParams

        proxy, worker_ingress, _, _ = self._make_proxy_and_fake_worker(tmp_path)
        receiving = threading.Event()
        real_recv = proxy.result_queue.queue._recv_data

        def recv() -> object:
            receiving.set()
            return real_recv()

        monkeypatch.setattr(proxy.result_queue.queue, "_recv_data", recv)
        try:
            # OpenAIServer reads this property when it initializes a frontend.
            assert proxy.resource_governor_queue is None
            result = proxy.submit(GenerationRequest([1, 2, 3], SamplingParams()))
            assert get_frontend_id(result.request_id) == 1
            assert worker_ingress.poll(5)
            received = worker_ingress.get()
            assert isinstance(received, GenerationRequest)
            assert received.id == result.request_id

            # No worker result or sentinel is sent to this lane on detach.
            assert receiving.wait(5)
            proxy.shutdown()
            assert not proxy.dispatch_result_thread.is_alive()
            assert proxy.result_queue.queue.socket is None
            assert proxy.request_queue.socket is None
            assert not proxy._results
            with pytest.raises(RequestError, match="shut down"):
                result.result(timeout=1)
            assert not proxy._engine_dead
            assert worker_ingress.poll(5)
            cancel = worker_ingress.get()
            assert isinstance(cancel, CancellingRequest)
            assert cancel.id == result.request_id
            proxy.shutdown()  # Idempotent; no duplicate cancel or global quit.
            assert not worker_ingress.poll(0.1), (
                "an attached frontend must never send the engine-shutdown sentinel"
            )
            with pytest.raises(RuntimeError, match="shutting down|shut down"):
                proxy.submit(GenerationRequest([4], SamplingParams()))
        finally:
            with ExitStack() as cleanup:
                cleanup.callback(worker_ingress.close)
                proxy.shutdown()

    @pytest.mark.parametrize("batching", [False, True])
    def test_check_health_and_submit_reflect_fatal_error(
        self, tmp_path, monkeypatch, batching
    ) -> None:
        from tensorrt_llm.executor.request import CancellingRequest, GenerationRequest
        from tensorrt_llm.executor.utils import EngineDeadError
        from tensorrt_llm.sampling_params import SamplingParams

        monkeypatch.setenv("TRTLLM_PROXY_BATCH_REQUESTS", "1" if batching else "0")
        proxy, worker_ingress, _, _ = self._make_proxy_and_fake_worker(tmp_path)
        try:
            assert proxy.check_health()
            assert proxy.model_world_size == 1
            assert proxy.dispatch_result_thread is None
            results = [proxy.submit(GenerationRequest([i], SamplingParams())) for i in (1, 2)]
            received = []
            while len(received) < len(results):
                assert worker_ingress.poll(5)
                message = worker_ingress.get()
                received.extend([pickle.loads(p) for p in message] if batching else [message])
            assert [r.id for r in received] == [r.request_id for r in results]
            assert all(isinstance(r, GenerationRequest) for r in received)

            # A local fatal error must not leave work running on the shared engine.
            proxy._set_fatal_error(RuntimeError("frontend dispatcher failed"))
            assert not proxy.check_health()
            with pytest.raises(EngineDeadError):
                proxy.submit(GenerationRequest([3], SamplingParams()))
            proxy.shutdown()
            assert not proxy.dispatch_result_thread.is_alive()
            assert not proxy._results
            cancels = []
            while len(cancels) < len(results):
                assert worker_ingress.poll(5)
                message = worker_ingress.get()
                cancels.extend([pickle.loads(p) for p in message] if batching else [message])
            assert all(isinstance(r, CancellingRequest) for r in cancels)
            assert [r.id for r in cancels] == [r.request_id for r in results]
            for result in results:
                with pytest.raises(EngineDeadError):
                    result.result(timeout=1)
            proxy.shutdown()
            assert not worker_ingress.poll(0.1), "no duplicate cancel or global engine quit"
        finally:
            with ExitStack() as cleanup:
                cleanup.callback(worker_ingress.close)
                proxy.shutdown()

    def test_dispatch_routes_own_lane_responses(self, tmp_path) -> None:
        import zmq

        from tensorrt_llm.executor.ipc import FusedIpcQueue
        from tensorrt_llm.executor.request import GenerationRequest
        from tensorrt_llm.executor.utils import ErrorResponse, RequestError
        from tensorrt_llm.sampling_params import SamplingParams

        proxy, worker_ingress, hmac_key, result_addrs = self._make_proxy_and_fake_worker(tmp_path)
        completed = threading.Event()

        class _ResultRegistry(dict):
            def pop(self, key: object, default: object = None) -> object:
                value = super().pop(key, default)
                completed.set()
                return value

        proxy._results = _ResultRegistry(proxy._results)
        worker_lane = FusedIpcQueue(
            (result_addrs[1], hmac_key),
            is_server=False,
            fuse_message=False,
            socket_type=zmq.PUSH,
            name="fake_worker_result_lane",
        )
        try:
            result = proxy.submit(GenerationRequest([1, 2, 3], SamplingParams()))
            client_id = result.request_id
            assert client_id in proxy._results
            assert worker_ingress.poll(5)
            worker_ingress.get()
            worker_lane.put(ErrorResponse(client_id, "boom", 1))
            with pytest.raises(RequestError, match="boom"):
                result.result(timeout=5)
            # Queue delivery precedes the dispatcher's final registry bookkeeping.
            assert completed.wait(5)
            assert client_id not in proxy._results
            proxy.shutdown()
            assert not proxy.dispatch_result_thread.is_alive()
            assert not proxy._results
            assert not worker_ingress.poll(0.1)
        finally:
            with ExitStack() as cleanup:
                cleanup.callback(worker_ingress.close)
                cleanup.callback(worker_lane.close)
                proxy.shutdown()
