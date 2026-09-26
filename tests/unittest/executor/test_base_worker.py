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
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

# isort: off
from utils.llm_data import llm_models_root
from utils.util import skip_single_gpu
# isort: on

import tensorrt_llm.executor.base_worker as base_worker_module
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest, LoRARequest
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.sampling_params import SamplingParams

default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
model_path = llm_models_root() / default_model_name


@pytest.mark.cpu_only
def test_enqueue_request_wraps_lora_load_error():

    class LoraManager:

        def is_adapter_in_cpu_cache(self, adapter_id):
            return False

    def raise_load_error(lora_request):
        raise RuntimeError("bad adapter")

    worker = object.__new__(BaseWorker)
    # GC-time __del__ -> shutdown() reads this; __init__ is bypassed here, so
    # seed it to keep teardown a clean no-op.
    worker.doing_shutdown = False
    worker._lora_manager = LoraManager()
    worker._load_lora_adapter = raise_load_error
    request = type(
        "Request", (), {
            "id": 1,
            "lora_request": type("LoraRequest", (), {"adapter_id": 999})(),
        })()

    with pytest.raises(RequestError, match="Failed to load LoRA adapter"):
        worker._enqueue_request(request)


def test_lora_request_does_not_probe_filesystem_on_init(tmp_path):
    missing_path = str(tmp_path / "private-lora-path")

    request = LoRARequest("missing", 1, missing_path)

    assert request.path == missing_path


@pytest.mark.cpu_only
def test_setup_engine_failure_releases_cpu_affinity(monkeypatch):
    release = Mock()
    worker = object.__new__(BaseWorker)
    worker._engine = "unused"
    worker.rank = 0
    worker.llm_args = SimpleNamespace(backend="unsupported")
    worker._backend = "unsupported"
    worker._cpu_affinity_lease = None
    worker.doing_shutdown = True

    def acquire_then_fail():
        worker._cpu_affinity_lease = 7
        return [], []

    worker._get_comm_ranks_device_id = acquire_then_fail
    monkeypatch.setattr(base_worker_module, "release_cpu_affinity", release)

    with pytest.raises(ValueError, match="Unsupported backend config"):
        worker.setup_engine()

    release.assert_called_once_with(7)
    assert worker._cpu_affinity_lease is None


@pytest.mark.cpu_only
def test_shutdown_failure_releases_cpu_affinity(monkeypatch):
    release = Mock()
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = False
    worker._cpu_affinity_lease = 11
    worker.engine = Mock()
    worker.engine.can_shutdown.return_value = True
    worker.engine.shutdown.side_effect = RuntimeError("shutdown failed")
    monkeypatch.setattr(base_worker_module, "release_cpu_affinity", release)

    with pytest.raises(RuntimeError, match="shutdown failed"):
        worker.shutdown()

    release.assert_called_once_with(11)
    assert worker._cpu_affinity_lease is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("can_shutdown", [False, RuntimeError("status failed")])
def test_shutdown_releases_cpu_affinity_without_engine_shutdown(
        monkeypatch, can_shutdown):
    release = Mock()
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = False
    worker._cpu_affinity_lease = 13
    worker.engine = Mock()
    if isinstance(can_shutdown, Exception):
        worker.engine.can_shutdown.side_effect = can_shutdown
    else:
        worker.engine.can_shutdown.return_value = can_shutdown
    monkeypatch.setattr(base_worker_module, "release_cpu_affinity", release)

    if isinstance(can_shutdown, Exception):
        with pytest.raises(RuntimeError, match="status failed"):
            worker.shutdown()
    else:
        worker.shutdown()

    worker.engine.shutdown.assert_not_called()
    release.assert_called_once_with(13)
    assert worker._cpu_affinity_lease is None


def create_fake_llm_args(engine_path, tp_size: int = 1):
    """Create TorchLlmArgs for testing.

    Args:
        engine_path: Path to the model
        tp_size: Tensor parallel size

    Returns:
        TorchLlmArgs
    """
    llm_args = TorchLlmArgs(
        model=engine_path,
        tensor_parallel_size=tp_size,
        backend='pytorch',
        enable_iter_perf_stats=True,
        max_seq_len=2048,  # Set reasonable max sequence length
        max_batch_size=8,  # Set reasonable batch size for tests
        max_num_tokens=2048,  # Set reasonable max tokens
    )
    return llm_args


class FakeWorker(BaseWorker):

    def __init__(self, engine: str, tp_size: int = 1):
        llm_args = TorchLlmArgs(
            model=model_path,
            tensor_parallel_size=tp_size,
            backend='pytorch',
            enable_iter_perf_stats=True,
        )
        super().__init__(
            engine=engine,
            llm_args=llm_args,
            hf_model_dir=engine,
        )
        # Note: BaseWorker doesn't call setup_engine() automatically,
        # unlike GenerationExecutorWorker, so we need to call it manually
        self.setup_engine()
        self._started = False

    def start(self):
        """Override start to mark as started - no background threads needed for test."""
        if not self._started:
            self._started = True
            # For testing, we don't need background threads
            # The engine's await_responses will handle the mock responses

    def shutdown(self):
        self._started = False
        try:
            if self.engine is not None:
                self.engine.shutdown()
                self.engine = None
        finally:
            self._release_cpu_affinity()


class TestWorkerBase:

    def test_create_engine(self):
        with FakeWorker(engine=model_path) as worker:
            print(f"Created engine: {worker.engine}")

    def test_submit_request(self):
        sampling_params = SamplingParams(max_tokens=10)
        request = GenerationRequest(prompt_token_ids=[3, 4, 5],
                                    sampling_params=sampling_params)
        with FakeWorker(engine=model_path) as worker:
            print(f"Created engine: {worker.engine}")
            result = worker.submit(request)

            # For PyTorch backend, the engine handles requests internally
            # We just need to give it some time to process
            timeout = 15.0  # 15 seconds timeout
            start_time = time.time()

            while not result.finished and (time.time() - start_time) < timeout:
                # Call await_responses with timeout to prevent hanging
                responses = worker.await_responses(timeout=0.5)
                time.sleep(0.1)

            if not result.finished:
                print(f"Request did not complete within {timeout} seconds")
            else:
                print(f"Request completed successfully")
                print(f"Result: {result}")

    def test_fetch_stats(self):
        request = GenerationRequest(
            prompt_token_ids=[3, 4, 5],
            sampling_params=SamplingParams(max_tokens=10))
        with FakeWorker(engine=model_path) as worker:
            result = worker.submit(request)

            # Give the engine time to start processing
            time.sleep(1)

            # Fetch stats while request is processing
            stats = worker.fetch_stats()
            print(f"Stats: {stats}")

            # Continue processing until completion or timeout
            timeout = 10.0
            start_time = time.time()
            while not result.finished and (time.time() - start_time) < timeout:
                worker.await_responses(timeout=0.5)
                time.sleep(0.1)

    @pytest.mark.parametrize("timeout", [0.1, 0.2, 1])
    def test_fetch_responses_timeout(self, timeout: float):
        with FakeWorker(engine=model_path) as worker:
            # Not submit any request, and let the await_responses timeout.
            start_time = time.time()
            results = worker.await_responses(timeout=timeout)
            elapsed = time.time() - start_time
            print(f"await_responses latency: {elapsed:.3f} seconds")
            assert timeout / 2 <= elapsed <= timeout * 2, f"Latency out of expected range: {elapsed}"


class TestRpcWorkerBaseTP2:

    def setup_method(self):
        # Use TorchLlmArgs for PyTorch backend with TP2
        self.llm_args = TorchLlmArgs(model=model_path,
                                     tensor_parallel_size=2,
                                     backend='pytorch')
        self.session = self.create_worker_session()

    def create_worker_session(self):
        # wait_shutdown: block shutdown until the workers exited, so a test
        # handed a live pool right after this one cannot race the GPU release.
        session = MpiPoolSession(n_workers=2, wait_shutdown=True)
        return session

    @pytest.mark.gpu2
    @skip_single_gpu
    def test_create_executor(self):
        futures = self.session.submit(
            TestRpcWorkerBaseTP2.create_executor,
            engine=model_path,
            llm_args=self.llm_args,
        )
        # Wait for completion
        for future in futures:
            future.result()

        self.session.shutdown()

    @staticmethod
    def create_executor(engine, llm_args):
        rank = mpi_rank()
        world_size = mpi_world_size()
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

        print(f"[Test] Rank {rank}/{world_size} using device {device_id}")

        # Synchronize all workers before creating executor
        mpi_comm().barrier()

        print(f"[Test] Rank {rank} creating FakeWorker...")
        executor = FakeWorker(engine=engine, tp_size=2)

        # Note: setup_engine is already called in FakeWorker.__init__
        print(
            f"[Test] Rank {rank} FakeWorker created and setup_engine completed successfully"
        )

        executor.shutdown()


if __name__ == "__main__":
    test_worker_base = TestWorkerBase()
    test_worker_base.test_submit_request()
