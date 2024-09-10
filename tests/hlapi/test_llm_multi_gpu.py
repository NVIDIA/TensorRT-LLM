import asyncio
import os
import subprocess  # nosec B404
import sys
import tempfile
import time

import pytest
import torch
from parameterized import parameterized

from tensorrt_llm.executor import (ExecutorBindingsProxy, GenerationRequest,
                                   GenerationResult)
from tensorrt_llm.hlapi.llm import LLM, SamplingParams
from tensorrt_llm.hlapi.llm_utils import KvCacheConfig
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.utils import get_total_gpu_memory
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.llama.model import LLaMAForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.util import skip_single_gpu, unittest_name_func

try:
    from .test_llm import (_test_llm_generate_async, default_model_name,
                           get_model_path, llama_7b_multi_lora_test_harness,
                           llama_model_path, llama_v2_13b_lora_test_harness,
                           llm_check_output, llm_test_harness,
                           mixtral_model_name, prompts)
except ImportError:
    from test_llm import (_test_llm_generate_async, default_model_name,
                          get_model_path, llama_7b_multi_lora_test_harness,
                          llama_model_path, llama_v2_13b_lora_test_harness,
                          llm_check_output, llm_test_harness,
                          mixtral_model_name, prompts)


@pytest.fixture(scope="module")
def engine_from_checkpoint() -> tempfile.TemporaryDirectory:
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    assert tokenizer is not None
    tp_size = 2
    with tempfile.TemporaryDirectory() as ckpt_dir:
        for rank in range(tp_size):
            mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=rank)
            llama = LLaMAForCausalLM.from_hugging_face(llama_model_path,
                                                       mapping=mapping)
            llama.save_checkpoint(ckpt_dir, save_config=(rank == 0))
            del llama

        llm = LLM(
            ckpt_dir,
            tokenizer=tokenizer,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        )
        assert llm.args.parallel_config.tp_size == tp_size

    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)

    return tmpdir


@skip_single_gpu
def test_llm_loading_from_ckpt_for_tp2(
        engine_from_checkpoint: tempfile.TemporaryDirectory):
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    llm_test_harness(engine_from_checkpoint.name,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=8),
                     tokenizer=tokenizer)


@skip_single_gpu
def test_llm_generate_tp2(engine_from_checkpoint):
    model_dir = engine_from_checkpoint.name
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)

    llm_test_harness(
        model_dir,
        prompts, ["D E F G H I J K"],
        sampling_params=SamplingParams(max_tokens=8),
        tokenizer=tokenizer,
        tensor_parallel_size=2,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))


@pytest.mark.parametrize("use_auto_parallel", [True, False],
                         ids=["enable_auto_parallel", "disable_auto_parallel"])
@pytest.mark.parametrize("from_ckpt", [True, False],
                         ids=["from_ckpt", "from_hf"])
@skip_single_gpu
def test_llm_generate_async_tp2(
        engine_from_checkpoint: tempfile.TemporaryDirectory,
        use_auto_parallel: bool, from_ckpt: bool):
    if use_auto_parallel and from_ckpt:
        pytest.skip("Skip auto parallel for TP2 checkpoint")
    model_dir = engine_from_checkpoint.name if from_ckpt else get_model_path(
        llama_model_path)
    tokenizer_dir = get_model_path(llama_model_path)
    tokenizer = TransformersTokenizer.from_pretrained(tokenizer_dir)
    _test_llm_generate_async(
        model_dir,
        tp_size=2,
        use_auto_parallel=use_auto_parallel,
        tokenizer=tokenizer,
    )


# TODO[chunweiy]: Move mixtral test to the e2e test
def is_memory_enough_for_mixtral():
    if torch.cuda.device_count() < 2:
        return False
    try:
        total_memory = get_total_gpu_memory(0) + get_total_gpu_memory(1)
        if total_memory >= 160 * 1024**3:
            return True
    except:
        return False


# NOTE: This is not activated in CI due to resource constraints
@skip_single_gpu
@pytest.mark.skipif(not is_memory_enough_for_mixtral(),
                    reason="The test needs at least 160GB memory, skipping")
def test_llm_generate_mixtral_for_tp2():
    llm = LLM(
        get_model_path(mixtral_model_name),
        tensor_parallel_size=2,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    for output in llm.generate(prompts):
        print(output)


def test_llm_pp2():
    llm_test_harness(
        llama_model_path,
        prompts, ["D E F G H I J K"],
        sampling_params=SamplingParams(max_tokens=8),
        pipeline_parallel_size=2,
        auto_parallel=False,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))


def llm_end2end_tp2_cases():
    yield ({}, )  # Default options
    yield ({'embedding_parallel_mode': 'NONE'}, )
    yield ({'embedding_parallel_mode': 'SHARDING_ALONG_HIDDEN'}, )
    yield ({
        'embedding_parallel_mode': 'SHARDING_ALONG_VOCAB',
        'share_embedding_table': True
    }, )


@parameterized.expand(llm_end2end_tp2_cases(), name_func=unittest_name_func)
@skip_single_gpu
def test_llm_end2end_tp2(llm_additional_options):
    model_path = get_model_path(default_model_name)

    llm = LLM(model_path, tensor_parallel_size=2, **llm_additional_options)
    assert llm.args._convert_checkpoint_options

    embedding_parallel_mode = llm_additional_options.pop(
        'embedding_parallel_mode', 'SHARDING_ALONG_VOCAB')
    if embedding_parallel_mode == 'NONE':
        assert llm.args._convert_checkpoint_options[
            'use_parallel_embedding'] is False
    elif embedding_parallel_mode == 'SHARDING_ALONG_VOCAB':
        assert llm.args._convert_checkpoint_options[
            'use_parallel_embedding'] is True
        assert llm.args._convert_checkpoint_options[
            'embedding_sharding_dim'] == 0
    elif embedding_parallel_mode == 'SHARDING_ALONG_HIDDEN':
        assert llm.args._convert_checkpoint_options[
            'use_parallel_embedding'] is True
        assert llm.args._convert_checkpoint_options[
            'embedding_sharding_dim'] == 1

    if 'share_embedding_table' in llm_additional_options:
        assert llm.args._convert_checkpoint_options[
            'share_embedding_table'] == llm_additional_options.pop(
                'share_embedding_table')
    else:
        assert llm.args._convert_checkpoint_options[
            'share_embedding_table'] is False

    assert not llm_additional_options

    llm_check_output(llm,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=8))


@skip_single_gpu
def test_llama_v2_13b_lora_tp2():
    llama_v2_13b_lora_test_harness(tensor_parallel_size=2)


@skip_single_gpu
def test_llama_7b_multi_lora_tp2():
    llama_7b_multi_lora_test_harness(tensor_parallel_size=2,
                                     max_loras=1,
                                     max_cpu_loras=8)


@skip_single_gpu
def _test_llm_multi_node(engine_from_checkpoint: tempfile.TemporaryDirectory):
    # TODO[chunweiy]: reactivate this later
    nworkers = 2
    test_case_file = os.path.join(os.path.dirname(__file__), "run_llm.py")
    os.path.join(os.path.dirname(__file__), "launch.py")
    command = f"mpirun --allow-run-as-root -n {nworkers} trtllm-hlapi-launch python3 {test_case_file} --model_dir {engine_from_checkpoint.name} --tp_size {nworkers}"
    subprocess.run(command, shell=True, check=True,
                   env=os.environ)  # nosec B603


@skip_single_gpu
def test_executor_results_cleanup():
    llm = LLM(model=llama_model_path,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
              tensor_parallel_size=2)
    sampling_params = SamplingParams(max_new_tokens=6)
    for i in range(20):
        llm.generate(prompts, sampling_params=sampling_params)

    num_remaining_results = len(llm._executor._results)
    assert num_remaining_results == 0


class DummyExecutor:

    @staticmethod
    def create(
        engine,
        executor_config,
        model_world_size: int = 1,
        world_size: int = 0,
        mpi_session=None,
        reuse_mpi_comm: bool = False,
    ):
        worker_kwargs = {
            "engine": engine,
            "executor_config": executor_config,
        }
        return DummyExecutorProxy(worker_kwargs,
                                  model_world_size=model_world_size,
                                  mpi_session=mpi_session)


class DummyExecutorProxy(ExecutorBindingsProxy):

    def __init__(
        self,
        workers_kwargs,
        model_world_size: int = 1,
        mpi_session=None,
    ) -> None:
        super().__init__(workers_kwargs, model_world_size, mpi_session)

    # This is copied from the ExecutorBindsProxy.submit method with minor modification
    def submit(self, request: GenerationRequest) -> GenerationResult:
        if not self.workers_started:
            self.start()

        self.request_queue.put(request)

        req_id = self.request_id_queue.get()
        request.set_id(req_id)

        result = GenerationResult(
            request, background_error_handler=self._handle_background_error)

        # Force the responses to be delayed
        time.sleep(1)

        print(f"number of pending responses: {len(self._pending_responses)}")
        assert self._pending_responses

        self._results[req_id] = result

        assert self._cleanup_pending_responses()

        return result


def test_executor_pending_requests():
    llm = LLM(model=llama_model_path,
              executor_cls=DummyExecutor,
              tensor_parallel_size=2)
    # The dummy executor will delay the responses
    sampling_params = SamplingParams(max_tokens=6)

    def test_nonstreaming():
        for output in llm.generate(prompts, sampling_params=sampling_params):
            print(output)

    def test_streaming():

        async def task():
            async for output in llm.generate_async(
                    prompts[0], streaming=True,
                    sampling_params=sampling_params):
                print(output)

        asyncio.run(task())

    test_nonstreaming()

    test_streaming()


class DummyExecutor2:

    @staticmethod
    def create(
        engine,
        executor_config,
        model_world_size: int = 1,
        world_size: int = 0,
        mpi_session=None,
        reuse_mpi_comm: bool = False,
    ):
        worker_kwargs = {
            "engine": engine,
            "executor_config": executor_config,
        }
        return DummyExecutorProxy2(worker_kwargs,
                                   model_world_size=model_world_size,
                                   mpi_session=mpi_session)


class DummyExecutorProxy2(ExecutorBindingsProxy):

    def __init__(
        self,
        workers_kwargs,
        model_world_size: int = 1,
        mpi_session=None,
    ) -> None:
        super().__init__(workers_kwargs, model_world_size, mpi_session)
        self.counter = 0

    def dispatch_result_task(self) -> bool:
        self.counter += 1

        if self.counter == 2:
            raise ValueError("Test error")

        return super().dispatch_result_task()


# TODO[chunweiy]: This test is not stable, need to investigate
def _test_executor_process_background_error():

    llm = LLM(model=llama_model_path, executor_cls=DummyExecutor2)
    # The dummy executor will delay the responses
    sampling_params = SamplingParams(max_tokens=6)

    # test in streaming mode
    async def task():
        with pytest.raises(ValueError):
            async for output in llm.generate_async(
                    prompts[0], streaming=True,
                    sampling_params=sampling_params):
                print(output)

    asyncio.run(task())


if __name__ == '__main__':
    test_llm_pp2()
    test_executor_pending_requests()
