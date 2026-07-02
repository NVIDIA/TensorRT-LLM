import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi._grouped_test_utils import \
    make_shared_llm as _make_shared_llm
from tensorrt_llm.llmapi._grouped_test_utils import \
    mpi_session_kwargs as _mpi_session_kwargs
from tensorrt_llm.llmapi._grouped_test_utils import \
    shared_mpi_session as _shared_mpi_session
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


class _MultiGpuLlmTests:
    """Base for the multi-GPU LLM API test classes.

    Subclasses set ``n_gpus``; the class-scoped fixtures below build one shared
    MpiPoolSession (of that many workers) reused across the subclass's tests and
    torn down when the subclass finishes -- so the 2-GPU and 4-GPU pools never
    coexist. Fixtures are lazy, so tests that opt out (e.g. constructing a bare
    ``LLM``) simply don't request them and pay no session cost.
    """
    n_gpus: int

    @pytest.fixture(scope="class")
    def mpi_session(self):
        yield from _shared_mpi_session(self.n_gpus)

    @pytest.fixture(scope="class")
    def mpi_kwargs(self, mpi_session):
        return _mpi_session_kwargs(mpi_session)

    @pytest.fixture(scope="class")
    def shared_llm(self, mpi_session):
        return _make_shared_llm(mpi_session)


@pytest.mark.gpu2
class TestLlmMultiGpu2gpu(_MultiGpuLlmTests):
    """2-GPU multi-GPU LLM API tests (shared session from _MultiGpuLlmTests)."""
    n_gpus = 2

    def test_llm_capture_request_error(self, mpi_kwargs):
        _test_llm_capture_request_error(pytorch_backend=True,
                                        tp_size=2,
                                        **mpi_kwargs)

    @pytest.mark.part0
    @pytest.mark.parametrize("tp_size, pp_size", [(1, 2), (2, 1)])
    def test_tinyllama_logits_processor_2gpu(self, tp_size, pp_size,
                                             mpi_kwargs):
        tinyllama_logits_processor_test_harness(backend="pytorch",
                                                tensor_parallel_size=tp_size,
                                                pipeline_parallel_size=pp_size,
                                                **mpi_kwargs)

    def test_llama_7b_lora_tp2(self, mpi_kwargs):
        llama_7b_lora_from_dir_test_harness(
            tensor_parallel_size=2,
            kv_cache_config=global_kv_cache_config,
            **mpi_kwargs)

    @skip_ray  # https://nvbugs/5727075
    @test_lora_with_and_without_cuda_graph
    def test_phi3_lora_fused_modules_output_on_tp2_identical_to_tp1(
            self, cuda_graph_config) -> None:
        check_phi3_lora_fused_modules_output_tp2_identical_to_tp1(
            LLM, cuda_graph_config=cuda_graph_config)

    @skip_ray
    def test_llm_rpc_tp2(self, shared_llm):
        _run_llm_rpc_tp2(shared_llm)

    @skip_ray
    @pytest.mark.asyncio
    async def test_llm_rpc_streaming_tp2(self, shared_llm):
        await _run_llm_rpc_streaming_tp2(shared_llm)

    @skip_ray
    @pytest.mark.parametrize(
        "prompt_logprobs, logprobs, return_context_logits, return_generation_logits",
        [
            (None, 1, False,
             False),  # generation logprobs only (top-1, PyTorch limit)
        ])
    def test_llm_return_logprobs_streaming_tp2(self, prompt_logprobs, logprobs,
                                               return_context_logits,
                                               return_generation_logits,
                                               mpi_kwargs):
        llm_return_logprobs_test_harness(prompt_logprobs,
                                         logprobs,
                                         return_context_logits,
                                         return_generation_logits,
                                         streaming=True,
                                         backend="pytorch",
                                         tp_size=2,
                                         **mpi_kwargs)

    @skip_ray
    @pytest.mark.parametrize(
        "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
        [
            (False, False, True),
            (False, True, True),
        ],
    )
    def test_llm_get_stats_pp2(self, return_context_logits,
                               enable_chunked_prefill, enable_iter_req_stats,
                               mpi_kwargs):
        llm_get_stats_test_harness(
            tp_size=1,
            pp_size=2,
            return_context_logits=return_context_logits,
            pytorch_backend=True,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_iter_req_stats=enable_iter_req_stats,
            **mpi_kwargs,
        )

    @skip_ray
    def test_llm_get_stats_tp2(self, mpi_kwargs):
        llm_get_stats_test_harness(tp_size=2,
                                   pytorch_backend=True,
                                   **mpi_kwargs)

    @skip_ray
    def test_llm_get_stats_async_tp2(self, mpi_kwargs):
        llm_get_stats_async_test_harness(tp_size=2,
                                         pytorch_backend=True,
                                         **mpi_kwargs)

    @skip_ray
    def test_llm_get_stats_async_pp2(self, mpi_kwargs):
        llm_get_stats_async_test_harness(pp_size=2,
                                         pytorch_backend=True,
                                         **mpi_kwargs)


@pytest.mark.gpu4
class TestLlmMultiGpu4gpu(_MultiGpuLlmTests):
    """4-GPU multi-GPU LLM API tests (shared session from _MultiGpuLlmTests)."""
    n_gpus = 4

    def test_tinyllama_logits_processor_tp2pp2(self, mpi_kwargs):
        tinyllama_logits_processor_test_harness(backend="pytorch",
                                                tensor_parallel_size=2,
                                                pipeline_parallel_size=2,
                                                **mpi_kwargs)

    @skip_ray  # https://nvbugs/5682551
    @test_lora_with_and_without_cuda_graph
    def test_llama_7b_multi_lora_tp4(self, cuda_graph_config, mpi_kwargs):
        # For LoRA checkpoints without finetuned embedding and lm_head, we can
        # either: (1) specify lora_target_modules, or (2) provide a lora_dir to
        # infer the lora_target_modules.
        lora_config = LoraConfig(
            lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=8)
        check_llama_7b_multi_lora_from_request_test_harness(
            LLM,
            lora_config=lora_config,
            tensor_parallel_size=4,
            kv_cache_config=global_kv_cache_config,
            cuda_graph_config=cuda_graph_config,
            **mpi_kwargs)

    @skip_ray
    @pytest.mark.parametrize(
        "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
        [
            (False, False, True),
            (False, True, True),
        ],
    )
    def test_llm_get_stats_pp4(self, return_context_logits,
                               enable_chunked_prefill, enable_iter_req_stats,
                               mpi_kwargs):
        llm_get_stats_test_harness(
            tp_size=1,
            pp_size=4,
            return_context_logits=return_context_logits,
            pytorch_backend=True,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_iter_req_stats=enable_iter_req_stats,
            **mpi_kwargs,
        )
