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

from tensorrt_llm.executor import base_worker
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import GenerationRequest, LoRARequest
from tensorrt_llm.executor.result import LogProbsResult
from tensorrt_llm.executor.utils import RequestError
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.sampling_params import LogprobParams, SamplingParams

default_model_name = "Qwen3/Qwen3-0.6B"
model_path = llm_models_root() / default_model_name


@pytest.mark.cpu_only
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("drop_context_logits", [False, True])
@pytest.mark.parametrize("simple", [False, True])
def test_prompt_logprobs_computation_and_streaming_cache(
        monkeypatch, streaming, drop_context_logits, simple):
    prompt_logprobs = [-0.5, -0.25, -0.125] if simple else [{11: "logprob"}]
    logits = object()
    calls = []

    def compute(k, context_logits, prompt_token_ids, *, simple_prompt_logprobs):
        calls.append(
            (k, context_logits, prompt_token_ids, simple_prompt_logprobs))
        return LogProbsResult(prompt=prompt_logprobs)

    monkeypatch.setattr(base_worker, "compute_prompt_logprobs", compute)
    generation_result = SimpleNamespace(
        _logprob_params=LogprobParams(prompt_logprobs=0,
                                      drop_context_logits=drop_context_logits,
                                      prompt_logprobs_simple_format=simple),
        _streaming=streaming,
        _generation_request=SimpleNamespace(prompt_token_ids=[10, 11, 12]))
    generation_result.clear_logprob_params = lambda: delattr(
        generation_result, "_logprob_params")
    worker = SimpleNamespace(_results={1: generation_result})

    def make_response(context_logits):
        result = SimpleNamespace(
            context_logits=context_logits,
            is_final=False,
            get_result=lambda: SimpleNamespace(output_token_ids=[[13]]))
        return SimpleNamespace(client_id=1,
                               result=result,
                               clear_context_logits=lambda: setattr(
                                   result, "context_logits", None))

    response = make_response(logits)
    assert base_worker._get_logprobs(worker, response).prompt == prompt_logprobs
    assert calls == [(0, logits, [11, 12, 13], simple)]
    assert response.result.context_logits is (None if drop_context_logits else
                                              logits)

    if streaming:
        # Later streaming chunks do not carry the original context logits.
        response = make_response(None)
        assert base_worker._get_logprobs(worker,
                                         response).prompt == prompt_logprobs
        assert len(calls) == 1

    if not streaming:
        response = make_response(logits)
    response.result.is_final = True
    assert base_worker._get_logprobs(worker, response).prompt == prompt_logprobs
    assert not hasattr(generation_result, "_logprob_params")


@pytest.mark.cpu_only
def test_generation_logprobs_do_not_recompute_logits(monkeypatch):

    def unexpected_compute(*args, **kwargs):
        pytest.fail("Generation logprobs must come from the sampler")

    monkeypatch.setattr(base_worker, "compute_prompt_logprobs",
                        unexpected_compute)
    generation_result = SimpleNamespace(
        _logprob_params=None,
        clear_logprob_params=lambda: delattr(generation_result,
                                             "_logprob_params"))
    worker = SimpleNamespace(_results={1: generation_result})
    response = SimpleNamespace(client_id=1,
                               result=SimpleNamespace(is_final=True))
    assert base_worker._get_logprobs(worker, response) is None
    assert not hasattr(generation_result, "_logprob_params")


@pytest.mark.cpu_only
@pytest.mark.parametrize("logprobs", [None, 0, 2])
@pytest.mark.parametrize("prompt_logprobs", [None, 0, 2])
@pytest.mark.parametrize("return_context_logits", [False, True])
def test_output_config_preserves_sampler_and_prompt_logprobs(
        logprobs, prompt_logprobs, return_context_logits):
    params = SamplingParams(logprobs=logprobs,
                            prompt_logprobs=prompt_logprobs,
                            return_context_logits=return_context_logits)
    config = params._get_output_config()
    assert config.return_log_probs == (logprobs is not None)
    assert config.return_context_logits == (return_context_logits
                                            or prompt_logprobs is not None)


@pytest.mark.cpu_only
def test_get_startup_metrics_promotes_model_engine_stages() -> None:
    """Verify startup metrics promote saved engine stages while retaining executor metrics."""
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = True
    worker.engine = SimpleNamespace(
        metrics={
            "worker_start_seconds": 0.5,
            "initial_model_engine": {
                "total_warmup_seconds": 2.5
            },
            "final_model_engine": {
                "total_warmup_seconds": 3.5
            },
        },
        model_engine=SimpleNamespace(
            metrics={"must_not_be_reported": 1.0},
            model_loader=SimpleNamespace(
                metrics={"total_model_loading_seconds": 1.5}),
        ),
        draft_model_engine=SimpleNamespace(
            metrics={"must_not_be_reported": 1.0},
            model_loader=SimpleNamespace(
                metrics={"total_model_loading_seconds": 0.75}),
        ),
    )

    assert worker.get_startup_metrics() == {
        "initial_model_engine": {
            "total_warmup_seconds": 2.5
        },
        "final_model_engine": {
            "total_warmup_seconds": 3.5
        },
        "py_executor": {
            "worker_start_seconds": 0.5
        },
        "model_loader": {
            "total_model_loading_seconds": 1.5
        },
        "draft_model_loader": {
            "total_model_loading_seconds": 0.75
        },
    }


@pytest.mark.cpu_only
def test_enqueue_request_wraps_lora_load_error():

    def raise_load_error(lora_request):
        raise RuntimeError("bad adapter")

    worker = object.__new__(BaseWorker)
    # GC-time __del__ -> shutdown() reads this; __init__ is bypassed here, so
    # seed it to keep teardown a clean no-op.
    worker.doing_shutdown = False
    worker.engine = None
    worker._lora_manager = object()
    worker._load_lora_adapter = raise_load_error
    request = type(
        "Request", (), {
            "id": 1,
            "lora_request": type("LoraRequest", (), {"adapter_id": 999})(),
        })()

    with pytest.raises(RequestError,
                       match="Failed to load LoRA adapter: bad adapter"):
        worker._enqueue_request(request)


def test_lora_request_does_not_probe_filesystem_on_init(tmp_path):
    missing_path = str(tmp_path / "private-lora-path")

    request = LoRARequest("missing", 1, missing_path)

    assert request.path == missing_path


@pytest.fixture
def lora_identity_worker() -> BaseWorker:
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = False
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
    worker._load_lora_adapter(LoRARequest("first", 7, path))
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
    worker._load_lora_adapter(LoRARequest("first", 7, str(path)))
    worker._lora_manager.load_from_ckpt.return_value = []
    with patch.object(Path,
                      "resolve",
                      side_effect=AssertionError("unexpected resolve")):
        worker._load_lora_adapter(LoRARequest("renamed", 7, str(path)))
    for alternate in ("adapter", str(alias), ""):
        worker._load_lora_adapter(LoRARequest("renamed", 7, alternate))
    assert worker._lora_adapter_identities == {"7": (str(path), "hf")}
    worker._lora_manager.load_from_ckpt.assert_called_with([""],
                                                           model_config=None,
                                                           uids=["7"],
                                                           ckpt_source="hf")
    worker._lora_manager.load_from_ckpt.return_value = ["8"]
    worker._load_lora_adapter(
        LoRARequest("independent", 8, str(tmp_path / "other")))


@pytest.mark.cpu_only
def test_lora_identity_preserves_id_only_first_use(
        lora_identity_worker: BaseWorker, tmp_path: Path) -> None:
    worker = lora_identity_worker
    worker._lora_manager.load_from_ckpt.return_value = []
    worker._load_lora_adapter(LoRARequest("cached", 7))
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
    worker._load_lora_adapter(
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
            assert second.result(timeout=5) is None
        finally:
            release_first.set()
        assert first.result(timeout=5) is None
    assert set(worker._lora_adapter_identities) == {"7", "8"}


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


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("drop_context_logits", [False, True])
@pytest.mark.parametrize("top_k,simple", [(0, False), (0, True), (2, False)])
def test_prompt_logprobs_preserve_token_alignment_and_streaming_cache(
        streaming, drop_context_logits, top_k, simple):
    import math

    from tensorrt_llm.executor.base_worker import _get_logprobs
    from tensorrt_llm.sampling_params import LogprobParams

    generation_result = SimpleNamespace(
        _streaming=streaming,
        _generation_request=SimpleNamespace(prompt_token_ids=[0, 1, 2]),
        _logprob_params=LogprobParams(prompt_logprobs=top_k,
                                      drop_context_logits=drop_context_logits,
                                      prompt_logprobs_simple_format=simple),
    )
    worker = SimpleNamespace(_results={7: generation_result})
    response_result = SimpleNamespace(
        is_final=False,
        context_logits=torch.tensor([[0., 1., 2., 3.]] * 3),
        get_result=lambda: SimpleNamespace(output_token_ids=[[0]]),
    )

    def clear_context_logits():
        response_result.context_logits = None

    response = SimpleNamespace(client_id=7,
                               result=response_result,
                               clear_context_logits=clear_context_logits)
    result = _get_logprobs(worker, response)

    normalizer = math.log(sum(math.exp(i) for i in range(4)))
    # Context logits predict the next prompt token; the last row predicts
    # the first generated token. The initial prompt token has no logprob.
    expected_tokens = [1, 2, 0]
    for entry, token in zip(result.prompt, expected_tokens):
        value = entry if simple else entry[token].logprob
        assert value == pytest.approx(token - normalizer)
        if not simple:
            assert entry[token].rank == 4 - token
    assert len(result.prompt) == len(expected_tokens)
    assert (response_result.context_logits is None) == drop_context_logits

    if streaming:
        # Later streaming responses can omit the already-consumed logits.
        response_result.context_logits = None
        cached_result = _get_logprobs(worker, response)
        assert cached_result.prompt is result.prompt


@pytest.mark.cpu_only
def test_generation_logprobs_need_no_worker_recomputation():
    from tensorrt_llm.executor.base_worker import _get_logprobs

    worker = SimpleNamespace(
        _results={7: SimpleNamespace(_logprob_params=None)})
    # Generation logprobs come directly from the sampler's response tensors.
    response = SimpleNamespace(client_id=7,
                               result=SimpleNamespace(is_final=False))
    assert _get_logprobs(worker, response) is None
