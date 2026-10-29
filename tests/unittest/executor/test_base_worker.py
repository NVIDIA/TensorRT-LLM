# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib
import importlib.util
import sys
import time
import warnings
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock

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
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
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
            "prompt_adapter_request": None,
            "lora_request": type("LoraRequest", (), {"adapter_id": 999})(),
        })()

    with pytest.raises(RequestError,
                       match="Failed to load LoRA adapter: bad adapter"):
        worker._enqueue_request(request)


def test_lora_request_does_not_probe_filesystem_on_init(tmp_path):
    missing_path = str(tmp_path / "private-lora-path")

    request = LoRARequest("missing", 1, missing_path)

    assert request.path == missing_path


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


@pytest.mark.cpu_only
def test_prompt_adapter_rejected_before_loading_or_changing_tokens() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        GenerationRequest(prompt_token_ids=[10, 11],
                          sampling_params=SamplingParams(max_tokens=1))
        GenerationRequest(prompt_token_ids=[10, 11],
                          sampling_params=SamplingParams(max_tokens=1),
                          prompt_adapter_request=None)
    assert not recorded

    with pytest.warns(DeprecationWarning,
                      match="prompt_adapter_request is deprecated"):
        request = GenerationRequest(
            prompt_token_ids=[10, 11],
            sampling_params=SamplingParams(max_tokens=1),
            prompt_adapter_request=object())
    request.id = 1
    # An unsupported adapter must fail before accessing any worker resources.
    with pytest.raises(RequestError, match="Prompt adapters are not supported"):
        BaseWorker._enqueue_request(SimpleNamespace(), request)

    worker = MagicMock(rank=0,
                       _results={},
                       _client_id_to_request_id={},
                       _handle_background_error=None)
    worker._get_logprob_params.return_value = None
    worker._enqueue_request.side_effect = lambda req: BaseWorker._enqueue_request(
        worker, req)
    with pytest.raises(RequestError, match="Prompt adapters are not supported"):
        BaseWorker.submit(worker, request)
    assert worker._results == {}
    assert worker._client_id_to_request_id == {}
    assert request.prompt_token_ids == [10, 11]


@pytest.mark.cpu_only
@pytest.mark.parametrize("error_type",
                         [None, RequestError, ValueError, RuntimeError])
def test_submit_cleans_up_failed_enqueue(
        error_type: type[Exception] | None) -> None:
    request = GenerationRequest(prompt_token_ids=[10, 11],
                                sampling_params=SamplingParams(max_tokens=1))
    request.set_id(7)
    peer_result = object()
    worker = MagicMock(rank=0,
                       _results={8: peer_result},
                       _client_id_to_request_id={8: 80})
    worker._handle_background_error = MethodType(lambda self: None, worker)
    worker._get_logprob_params.return_value = None
    error = error_type("enqueue failed") if error_type else None

    def enqueue(req: GenerationRequest) -> int:
        assert worker._results[req.id]._generation_request is req
        if error is not None:
            raise error
        return 70

    worker._enqueue_request.side_effect = enqueue
    if error is not None:
        with pytest.raises(error_type, match="enqueue failed") as exc_info:
            BaseWorker.submit(worker, request)
        assert exc_info.value is error
        assert worker._results == {8: peer_result}
        assert worker._client_id_to_request_id == {8: 80}
    else:
        result = BaseWorker.submit(worker, request)
        assert worker._results == {7: result, 8: peer_result}
        assert worker._client_id_to_request_id == {7: 70, 8: 80}


def _load_ray_executor(monkeypatch: pytest.MonkeyPatch) -> type:
    """Load the real Ray executor with cluster-only imports stubbed."""
    importlib.import_module("tensorrt_llm.executor.ray.utils")
    dependencies = {
        name: ModuleType(name)
        for name in (
            "ray",
            "ray.util",
            "ray.util.placement_group",
            "ray.util.scheduling_strategies",
            "tensorrt_llm.executor.ray.gpu_worker",
        )
    }
    placement = dependencies["ray.util.placement_group"]
    placement.get_current_placement_group = MagicMock()
    placement.placement_group = MagicMock()
    dependencies[
        "ray.util.scheduling_strategies"].PlacementGroupSchedulingStrategy = MagicMock(
        )
    gpu_worker = dependencies["tensorrt_llm.executor.ray.gpu_worker"]
    gpu_worker.RayGPUWorker = MagicMock()
    gpu_worker.RayWorkerWrapper = MagicMock()
    for name, module in dependencies.items():
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(base_worker.__file__).parent / "ray" / "executor.py"
    spec = importlib.util.spec_from_file_location(
        "tensorrt_llm.executor.ray._submit_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RayExecutor


@pytest.mark.cpu_only
@pytest.mark.parametrize("executor_kind", ["rpc", "ray"])
@pytest.mark.parametrize("unsupported_adapter", [False, True])
def test_rpc_submit_rejects_prompt_adapters_before_dispatch(
        monkeypatch: pytest.MonkeyPatch, executor_kind: str,
        unsupported_adapter: bool) -> None:
    executor_class = RpcExecutorMixin if executor_kind == "rpc" else _load_ray_executor(
        monkeypatch)
    if unsupported_adapter:
        with pytest.warns(DeprecationWarning,
                          match="prompt_adapter_request is deprecated"):
            request = GenerationRequest(
                prompt_token_ids=[10, 11],
                sampling_params=SamplingParams(max_tokens=1),
                prompt_adapter_request=object(),
            )
    else:
        request = GenerationRequest(
            prompt_token_ids=[10, 11],
            sampling_params=SamplingParams(max_tokens=1))
    peer_result = object()
    proxy = MagicMock(_results={8: peer_result}, _handle_background_error=None)
    proxy._get_next_client_id.return_value = 7
    proxy._get_logprob_params.return_value = None

    if unsupported_adapter:
        with pytest.raises(RequestError,
                           match="Prompt adapters are not supported"):
            executor_class.submit(proxy, request)
        proxy.rpc_client.submit.assert_not_called()
        proxy._get_next_client_id.assert_not_called()
        assert request.id is None
        assert proxy._results == {8: peer_result}
    else:
        result = executor_class.submit(proxy, request)
        proxy.rpc_client.submit.assert_called_once_with(request)
        proxy.rpc_client.submit.return_value.remote.assert_called_once_with(
            need_response=False)
        assert request.id == 7
        assert proxy._results == {7: result, 8: peer_result}
