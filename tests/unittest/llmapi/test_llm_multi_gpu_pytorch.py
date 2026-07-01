import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.sampling_params import SamplingParams

from .lora_test_utils import (
    check_llama_7b_multi_lora_from_request_test_harness,
    check_phi3_lora_fused_modules_output_tp2_identical_to_tp1,
    test_lora_with_and_without_cuda_graph)
from .test_llm import (_test_llm_capture_request_error, llama_model_path,
                       llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness,
                       llm_return_logprobs_test_harness,
                       tinyllama_logits_processor_test_harness)
from .test_llm_pytorch import llama_7b_lora_from_dir_test_harness

global_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


def _shared_mpi_session(n_workers: int):
    if mpi_disabled():
        yield None
        return

    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    mpi_session = MpiPoolSession(n_workers=n_workers)
    try:
        yield mpi_session
    finally:
        mpi_session.shutdown()


@pytest.fixture(scope="module")
def shared_mpi_session_2gpu():
    yield from _shared_mpi_session(2)


@pytest.fixture(scope="module")
def shared_mpi_session_4gpu():
    yield from _shared_mpi_session(4)


def _mpi_session_kwargs(mpi_session) -> dict:
    return {"_mpi_session": mpi_session} if mpi_session is not None else {}


def _make_shared_llm(mpi_session):
    """Return an LLM factory that transparently injects a shared MPI session.

    Tests that build the LLM directly can request the ``shared_llm_*`` fixture
    and call it exactly like ``LLM(...)`` -- the shared session is passed through
    without the test having to know it exists. Harness-based tests inject the
    session via ``**_mpi_session_kwargs(...)`` instead, since the harness owns
    LLM construction.
    """

    def shared_llm(*args, **kwargs):
        return LLM(*args, **kwargs, **_mpi_session_kwargs(mpi_session))

    return shared_llm


@pytest.fixture(scope="module")
def shared_llm_2gpu(shared_mpi_session_2gpu):
    return _make_shared_llm(shared_mpi_session_2gpu)


@pytest.mark.gpu2
def test_llm_capture_request_error(shared_mpi_session_2gpu):
    _test_llm_capture_request_error(
        pytorch_backend=True,
        tp_size=2,
        **_mpi_session_kwargs(shared_mpi_session_2gpu))


@pytest.mark.gpu4
def test_tinyllama_logits_processor_tp2pp2(shared_mpi_session_4gpu):
    tinyllama_logits_processor_test_harness(
        backend="pytorch",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        **_mpi_session_kwargs(shared_mpi_session_4gpu))


@pytest.mark.gpu2
@pytest.mark.part0
@pytest.mark.parametrize("tp_size, pp_size", [(1, 2), (2, 1)])
def test_tinyllama_logits_processor_2gpu(tp_size: int, pp_size: int,
                                         shared_mpi_session_2gpu):
    tinyllama_logits_processor_test_harness(
        backend="pytorch",
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        **_mpi_session_kwargs(shared_mpi_session_2gpu))


@pytest.mark.gpu2
def test_llama_7b_lora_tp2(shared_mpi_session_2gpu):
    llama_7b_lora_from_dir_test_harness(
        tensor_parallel_size=2,
        kv_cache_config=global_kv_cache_config,
        **_mpi_session_kwargs(shared_mpi_session_2gpu))


@pytest.mark.gpu4
@skip_ray  # https://nvbugs/5682551
@test_lora_with_and_without_cuda_graph
def test_llama_7b_multi_lora_tp4(cuda_graph_config, shared_mpi_session_4gpu):
    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=8,
                             max_loras=1,
                             max_cpu_loras=8)
    check_llama_7b_multi_lora_from_request_test_harness(
        LLM,
        lora_config=lora_config,
        tensor_parallel_size=4,
        kv_cache_config=global_kv_cache_config,
        cuda_graph_config=cuda_graph_config,
        **_mpi_session_kwargs(shared_mpi_session_4gpu))


@skip_ray  # https://nvbugs/5727075
@pytest.mark.gpu2
@test_lora_with_and_without_cuda_graph
def test_phi3_lora_fused_modules_output_on_tp2_identical_to_tp1(
        cuda_graph_config) -> None:
    check_phi3_lora_fused_modules_output_tp2_identical_to_tp1(
        LLM, cuda_graph_config=cuda_graph_config)


@skip_ray
@pytest.mark.gpu2
def test_llm_rpc_tp2(shared_llm_2gpu):
    _run_llm_rpc_tp2(shared_llm_2gpu)


@skip_ray
@pytest.mark.gpu2
@pytest.mark.asyncio
async def test_llm_rpc_streaming_tp2(shared_llm_2gpu):
    await _run_llm_rpc_streaming_tp2(shared_llm_2gpu)


def _run_llm_rpc_tp2(make_llm=LLM):
    with make_llm(model=llama_model_path,
                  kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
                  orchestrator_type="rpc",
                  tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        res = llm.generate("Tell me a joke",
                           sampling_params=SamplingParams(max_tokens=10,
                                                          end_id=-1))
        print(f"get result: {res}")

        assert len(res.outputs) == 1
        assert len(res.outputs[0].token_ids) == 10


async def _run_llm_rpc_streaming_tp2(make_llm=LLM):
    with make_llm(model=llama_model_path,
                  kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
                  orchestrator_type="rpc",
                  tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        async for output in llm.generate_async("Tell me a joke",
                                               sampling_params=SamplingParams(
                                                   max_tokens=10, end_id=-1)):
            print(f"get result: {output}")


@skip_ray
@pytest.mark.gpu2
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits",
    [
        (None, 1, False,
         False),  # generation logprobs only (top-1, PyTorch limit)
    ])
def test_llm_return_logprobs_streaming_tp2(prompt_logprobs, logprobs,
                                           return_context_logits,
                                           return_generation_logits,
                                           shared_mpi_session_2gpu):
    llm_return_logprobs_test_harness(
        prompt_logprobs,
        logprobs,
        return_context_logits,
        return_generation_logits,
        streaming=True,
        backend="pytorch",
        tp_size=2,
        **_mpi_session_kwargs(shared_mpi_session_2gpu))


@skip_ray
@pytest.mark.gpu2
@pytest.mark.parametrize(
    "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, True),
        (False, True, True),
    ],
)
def test_llm_get_stats_pp2(return_context_logits, enable_chunked_prefill,
                           enable_iter_req_stats, shared_mpi_session_2gpu):
    llm_get_stats_test_harness(
        tp_size=1,
        pp_size=2,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
        **_mpi_session_kwargs(shared_mpi_session_2gpu),
    )


@skip_ray
@pytest.mark.gpu4
@pytest.mark.parametrize(
    "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, True),
        (False, True, True),
    ],
)
def test_llm_get_stats_pp4(return_context_logits, enable_chunked_prefill,
                           enable_iter_req_stats, shared_mpi_session_4gpu):
    llm_get_stats_test_harness(
        tp_size=1,
        pp_size=4,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
        **_mpi_session_kwargs(shared_mpi_session_4gpu),
    )


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_tp2(shared_mpi_session_2gpu):
    llm_get_stats_test_harness(tp_size=2,
                               pytorch_backend=True,
                               **_mpi_session_kwargs(shared_mpi_session_2gpu))


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_async_tp2(shared_mpi_session_2gpu):
    llm_get_stats_async_test_harness(
        tp_size=2,
        pytorch_backend=True,
        **_mpi_session_kwargs(shared_mpi_session_2gpu))


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_async_pp2(shared_mpi_session_2gpu):
    llm_get_stats_async_test_harness(
        pp_size=2,
        pytorch_backend=True,
        **_mpi_session_kwargs(shared_mpi_session_2gpu))
