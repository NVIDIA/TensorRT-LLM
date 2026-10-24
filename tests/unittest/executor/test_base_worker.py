# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

# isort: off
from utils.llm_data import llm_models_root
from utils.util import skip_single_gpu
# isort: on

from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest, LoRARequest
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.sampling_params import SamplingParams

default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
model_path = llm_models_root() / default_model_name


@pytest.mark.cpu_only
def test_embedding_bias_sparse_to_dense():
    vocab_size = 123
    embedding_bias = ((1, -0.5), (42, -2.0))
    dense_bias = BaseWorker._embedding_bias_sparse_to_dense(
        embedding_bias,
        vocab_size=vocab_size,
    )
    assert dense_bias.shape == (vocab_size, )
    for idx, val in embedding_bias:
        assert dense_bias[idx] == val
    idx_nz = set(idx for idx, _ in embedding_bias)
    for idx in range(vocab_size):
        if idx in idx_nz:
            continue
        assert dense_bias[idx] == 0


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


@pytest.fixture
def lora_identity_worker() -> BaseWorker:
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = False
    worker._is_pytorch_backend = True
    worker._runtime_model_config = None
    worker._lora_model_config = None
    worker._lora_adapter_identities = {}
    worker._lora_adapter_id_locks = {}
    worker._lora_adapter_id_locks_guard = Lock()
    worker._lora_manager = Mock()
    worker._lora_manager.cpp_lora_weights = {}
    worker._lora_manager.cpp_lora_config = {}
    worker._lora_manager.load_from_ckpt.return_value = ["7"]
    worker.engine = Mock()
    return worker


@pytest.mark.cpu_only
@pytest.mark.parametrize("change_source", [False, True])
def test_lora_identity_rejects_conflicting_request(
        lora_identity_worker: BaseWorker, tmp_path: Path,
        change_source: bool) -> None:
    worker = lora_identity_worker
    path = str(tmp_path / "adapter")
    assert worker._load_lora_adapter(LoRARequest("first", 7, path))
    conflict = LoRARequest("second", 7,
                           path if change_source else str(tmp_path / "other"),
                           "nemo" if change_source else "hf")
    with patch("tensorrt_llm.executor.base_worker.logger.warning") as warning:
        with pytest.raises(RequestError, match="ID 7 is already bound"):
            worker._enqueue_request(SimpleNamespace(id=1,
                                                    lora_request=conflict))
    assert warning.call_args.args[1:3] == ("7", (path, "hf"))
    assert worker._lora_manager.load_from_ckpt.call_count == 1
    worker.engine.enqueue_request.assert_not_called()


@pytest.mark.cpu_only
def test_lora_identity_preserves_reuse(lora_identity_worker: BaseWorker,
                                       tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    worker = lora_identity_worker
    path = tmp_path / "adapter"
    path.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(path, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    assert worker._load_lora_adapter(LoRARequest("first", 7, str(path)))
    worker._lora_manager.load_from_ckpt.return_value = []
    with patch.object(Path,
                      "resolve",
                      side_effect=AssertionError("unexpected resolve")):
        assert not worker._load_lora_adapter(
            LoRARequest("renamed", 7, str(path)))
    for alternate in ("adapter", str(alias), ""):
        assert not worker._load_lora_adapter(
            LoRARequest("renamed", 7, alternate))
    assert worker._lora_adapter_identities == {"7": (str(path), "hf")}
    worker._lora_manager.load_from_ckpt.assert_called_with([""],
                                                           model_config=None,
                                                           uids=["7"],
                                                           ckpt_source="hf")
    worker._lora_manager.load_from_ckpt.return_value = ["8"]
    assert worker._load_lora_adapter(
        LoRARequest("independent", 8, str(tmp_path / "other")))


@pytest.mark.cpu_only
def test_lora_identity_preserves_id_only_first_use(
        lora_identity_worker: BaseWorker, tmp_path: Path) -> None:
    worker = lora_identity_worker
    worker._lora_manager.load_from_ckpt.return_value = []
    assert not worker._load_lora_adapter(LoRARequest("cached", 7))
    assert worker._lora_adapter_identities == {"7": ("", "hf")}
    with pytest.raises(ValueError, match="already bound"):
        worker._load_lora_adapter(LoRARequest("new", 7, str(tmp_path)))
    assert worker._lora_manager.load_from_ckpt.call_count == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("empty_path", [False, True])
def test_lora_identity_failed_lookup_is_retryable(
        lora_identity_worker: BaseWorker, tmp_path: Path,
        empty_path: bool) -> None:
    worker = lora_identity_worker
    worker._lora_manager.load_from_ckpt.side_effect = FileNotFoundError(
        "missing")
    with pytest.raises(FileNotFoundError, match="missing"):
        worker._load_lora_adapter(
            LoRARequest("missing", 7,
                        "" if empty_path else str(tmp_path / "typo")))
    assert worker._lora_adapter_identities == {}
    worker._lora_manager.load_from_ckpt.side_effect = None
    assert worker._load_lora_adapter(
        LoRARequest("corrected", 7, str(tmp_path / "adapter")))
    assert worker._lora_adapter_identities == {
        "7": (str(tmp_path / "adapter"), "hf")
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("published_cache",
                         ["cpp_lora_weights", "cpp_lora_config"])
def test_lora_identity_retains_partial_binding(lora_identity_worker: BaseWorker,
                                               tmp_path: Path,
                                               published_cache: str) -> None:
    worker = lora_identity_worker

    def fail_after_publish(*args, **kwargs) -> None:
        getattr(worker._lora_manager, published_cache)["7"] = []
        raise RuntimeError("load failed")

    worker._lora_manager.load_from_ckpt.side_effect = fail_after_publish
    with pytest.raises(RuntimeError, match="load failed"):
        worker._load_lora_adapter(LoRARequest("first", 7, str(tmp_path / "a")))
    with pytest.raises(ValueError, match="already bound"):
        worker._load_lora_adapter(LoRARequest("second", 7, str(tmp_path / "b")))
    assert worker._lora_manager.load_from_ckpt.call_count == 1


@pytest.mark.cpu_only
def test_lora_identity_serializes_registration(lora_identity_worker: BaseWorker,
                                               tmp_path: Path) -> None:
    worker = lora_identity_worker
    load_started = Event()
    release_load = Event()

    class SignalingLock:

        def __init__(self) -> None:
            self._lock = Lock()
            self._attempts_lock = Lock()
            self._attempts = 0
            self.second_attempted = Event()

        def __enter__(self):
            with self._attempts_lock:
                self._attempts += 1
                if self._attempts == 2:
                    self.second_attempted.set()
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            self._lock.release()

    def load_adapter(*args, **kwargs):
        load_started.set()
        assert release_load.wait(timeout=5)
        return ["7"]

    adapter_lock = SignalingLock()
    worker._lora_adapter_id_locks["7"] = adapter_lock
    worker._lora_manager.load_from_ckpt.side_effect = load_adapter
    requests = [
        LoRARequest("adapter", 7, str(tmp_path / name)) for name in ("a", "b")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker._load_lora_adapter, requests[0])
        assert load_started.wait(timeout=5)
        second = pool.submit(worker._load_lora_adapter, requests[1])
        try:
            assert adapter_lock.second_attempted.wait(timeout=5)
        finally:
            release_load.set()
        futures = [first, second]
    assert sum(future.exception() is None for future in futures) == 1
    errors = [future.exception() for future in futures if future.exception()]
    assert isinstance(errors[0], ValueError)
    assert "already bound" in str(errors[0])
    assert worker._lora_manager.load_from_ckpt.call_count == 1


@pytest.mark.cpu_only
def test_lora_identity_allows_independent_registration(
        lora_identity_worker: BaseWorker, tmp_path: Path) -> None:
    worker = lora_identity_worker
    first_started = Event()
    release_first = Event()
    second_loaded = Event()

    def load_adapter(*args, uids, **kwargs):
        adapter_id = uids[0]
        if adapter_id == "7":
            first_started.set()
            assert release_first.wait(timeout=5)
        else:
            second_loaded.set()
        return [adapter_id]

    worker._lora_manager.load_from_ckpt.side_effect = load_adapter
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker._load_lora_adapter,
                            LoRARequest("first", 7, str(tmp_path / "a")))
        assert first_started.wait(timeout=5)
        second = pool.submit(worker._load_lora_adapter,
                             LoRARequest("second", 8, str(tmp_path / "b")))
        try:
            assert second_loaded.wait(timeout=5)
            assert second.result(timeout=5)
        finally:
            release_first.set()
        assert first.result(timeout=5)


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
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None


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
