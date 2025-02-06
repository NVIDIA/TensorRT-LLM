import asyncio
import json
import os
import subprocess  # nosec B404
import tempfile

import pytest
from parameterized import parameterized

from tensorrt_llm.executor import ExecutorBindingsProxy
from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models.llama.model import LLaMAForCausalLM

# isort: off
from test_llm import (
    DummyError, DummyExecutorWorker3, _test_llm_capture_request_error,
    _test_llm_generate_async, check_llm_return_context_logits,
    check_llm_return_generation_logits, default_model_name, get_model_path,
    llama_7b_multi_lora_test_harness, llama_model_path,
    llama_v2_7b_prompt_adapter_test_harness, llama_v2_13b_lora_test_harness,
    llm_check_output, llm_get_stats_async_test_harness,
    llm_get_stats_test_harness, llm_test_harness, mixtral_model_name, prompts,
    tinyllama_guided_decoding_test_harness,
    tinyllama_logits_processor_test_harness, run_llm_with_postprocess_parallel,
    run_llm_with_postprocess_parallel_and_result_handler)
from utils.util import (skip_gpu_memory_less_than, skip_num_gpus_less_than,
                        skip_single_gpu, unittest_name_func)
# isort: on


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


# shrink the kv_cache_config to avoid OOM in CI
global_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

# python api does not seem to support extra tokens needed for prompt tuning + reuse.
# disable block reuse for those tests.
# TODO: Add extra tokens to prompt tuning unit tests.
global_kv_cache_config_no_reuse = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                                enable_block_reuse=False)


@skip_single_gpu
def test_llm_loading_from_ckpt_for_tp2(
        engine_from_checkpoint: tempfile.TemporaryDirectory):
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    llm_test_harness(engine_from_checkpoint.name,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=8),
                     tokenizer=tokenizer,
                     kv_cache_config=global_kv_cache_config)


@skip_single_gpu
def test_llm_generate_tp2():
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=8),
                     tensor_parallel_size=2,
                     kv_cache_config=global_kv_cache_config)


def test_llm_explicit_shutdown():
    # with-statement will invoke _shutdown() explicitly
    with LLM(model=llama_model_path,
             tensor_parallel_size=2,
             kv_cache_config=global_kv_cache_config,
             fast_build=True) as llm:
        llm_check_output(llm,
                         prompts, ["D E F G H I J K"],
                         sampling_params=SamplingParams(max_tokens=8))


@skip_single_gpu
def test_llm_return_context_logits_tp2():
    check_llm_return_context_logits(tp_size=2)


@skip_single_gpu
def test_llm_return_generation_logits_tp2():
    check_llm_return_generation_logits(tp_size=2)


@pytest.mark.parametrize("use_auto_parallel", [True, False],
                         ids=["enable_auto_parallel", "disable_auto_parallel"])
@pytest.mark.parametrize("from_ckpt", [True, False],
                         ids=["from_ckpt", "from_hf"])
@skip_single_gpu
def test_llm_generate_async_tp2(
        engine_from_checkpoint: tempfile.TemporaryDirectory, from_ckpt: bool,
        use_auto_parallel: bool):
    if use_auto_parallel and from_ckpt:
        pytest.skip("Skip auto parallel for TP2 checkpoint")
    model_dir = engine_from_checkpoint.name if from_ckpt else get_model_path(
        llama_model_path)
    tokenizer_dir = get_model_path(llama_model_path)
    tokenizer = TransformersTokenizer.from_pretrained(tokenizer_dir)
    _test_llm_generate_async(model_dir,
                             tp_size=2,
                             use_auto_parallel=use_auto_parallel,
                             tokenizer=tokenizer)


@skip_single_gpu
@skip_gpu_memory_less_than(70 * 1024**3)
def test_llm_generate_mixtral_for_tp2():
    llm = LLM(get_model_path(mixtral_model_name),
              tensor_parallel_size=2,
              kv_cache_config=global_kv_cache_config)
    for output in llm.generate(prompts):
        print(output)


@skip_single_gpu
@skip_gpu_memory_less_than(70 * 1024**3)
def test_llm_generate_mixtral_for_ep2():
    llm = LLM(get_model_path(mixtral_model_name),
              tensor_parallel_size=2,
              moe_expert_parallel_size=2,
              moe_tensor_parallel_size=1,
              kv_cache_config=global_kv_cache_config)
    for output in llm.generate(prompts):
        print(output)

    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)

    with open(os.path.join(tmpdir.name, "config.json"), "r") as f:
        # read the build_config and check if the parameters are correctly saved
        engine_config = json.load(f)

        pretrained_config = PretrainedConfig.from_dict(
            engine_config["pretrained_config"])

        assert pretrained_config.mapping.moe_tp_size == 1
        assert pretrained_config.mapping.moe_ep_size == 2


def test_llm_pp2():
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=8),
                     pipeline_parallel_size=2,
                     auto_parallel=False,
                     kv_cache_config=global_kv_cache_config)


def llm_end2end_tp2_cases():
    yield ({}, )  # Default options
    yield ({'embedding_parallel_mode': 'NONE'}, )
    yield ({'embedding_parallel_mode': 'SHARDING_ALONG_HIDDEN'}, )


@parameterized.expand(llm_end2end_tp2_cases(), name_func=unittest_name_func)
@skip_single_gpu
def test_llm_end2end_tp2(llm_additional_options):
    model_path = get_model_path(default_model_name)

    llm = LLM(model_path,
              tensor_parallel_size=2,
              **llm_additional_options,
              kv_cache_config=global_kv_cache_config)
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

    assert not llm_additional_options

    llm_check_output(llm,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=8))


@skip_num_gpus_less_than(4)
def test_tinyllama_logits_processor_tp2pp2():
    tinyllama_logits_processor_test_harness(tensor_parallel_size=2,
                                            pipeline_parallel_size=2)


@skip_num_gpus_less_than(4)
def test_tinyllama_guided_decoding_tp2pp2():
    tinyllama_guided_decoding_test_harness(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        kv_cache_config=global_kv_cache_config)


@skip_single_gpu
def test_llama_v2_13b_lora_tp2():
    llama_v2_13b_lora_test_harness(tensor_parallel_size=2,
                                   kv_cache_config=global_kv_cache_config)


@skip_single_gpu
def test_llama_7b_multi_lora_tp2():
    llama_7b_multi_lora_test_harness(tensor_parallel_size=2,
                                     max_loras=1,
                                     max_cpu_loras=8,
                                     kv_cache_config=global_kv_cache_config)


@skip_single_gpu
def test_llama_v2_7b_prompt_adapter_tp2():
    llama_v2_7b_prompt_adapter_test_harness(
        tensor_parallel_size=2, kv_cache_config=global_kv_cache_config_no_reuse)


@skip_single_gpu
def _test_llm_multi_node(engine_from_checkpoint: tempfile.TemporaryDirectory):
    # TODO[chunweiy]: reactivate this later
    nworkers = 2
    test_case_file = os.path.join(os.path.dirname(__file__), "run_llm.py")
    os.path.join(os.path.dirname(__file__), "launch.py")
    command = f"mpirun --allow-run-as-root -n {nworkers} trtllm-llmapi-launch python3 {test_case_file} --model_dir {engine_from_checkpoint.name} --tp_size {nworkers}"
    subprocess.run(command, shell=True, check=True,
                   env=os.environ)  # nosec B603


@skip_single_gpu
def test_executor_results_cleanup():
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kv_cache_config,
              tensor_parallel_size=2)
    sampling_params = SamplingParams(max_new_tokens=6)
    for i in range(20):
        llm.generate(prompts, sampling_params=sampling_params)

    num_remaining_results = len(llm._executor._results)
    assert num_remaining_results == 0


class DummyExecutorMeta(type):

    def __new__(cls, name, bases, dic, proxy_class):
        new_cls = super().__new__(cls, name, bases, dic)

        @staticmethod
        def create(engine,
                   executor_config,
                   model_world_size: int = 1,
                   world_size: int = 0,
                   mpi_session=None,
                   reuse_mpi_comm: bool = False):
            worker_kwargs = {
                "engine": engine,
                "executor_config": executor_config,
            }
            return proxy_class(worker_kwargs,
                               model_world_size=model_world_size,
                               mpi_session=mpi_session)

        new_cls.create = create
        return new_cls


class DummyExecutorProxy2(ExecutorBindingsProxy):
    ''' This is for testing the error occur in the thread in the Proxy. '''

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

        # This will raise error in dispatch_result_thread in the main process, it will be captured by ManagedThread and
        # redirect to the error_queue
        if self.counter == 2:
            raise DummyError("Test error")

        return super().dispatch_result_task()


DummyExecutor2 = DummyExecutorMeta("DummyExecutor2", (), {},
                                   proxy_class=DummyExecutorProxy2)


# TODO[chunweiy]: This test is not stable, need to investigate
def _test_executor_handle_background_error_in_dispatch_result_thread():
    llm = LLM(model=llama_model_path,
              executor_cls=DummyExecutor2,
              kv_cache_config=global_kv_cache_config,
              fast_build=True)
    # The dummy executor will delay the responses
    sampling_params = SamplingParams(max_tokens=6)

    # test in streaming mode
    async def task():
        with pytest.raises(DummyError):
            with llm:
                async for output in llm.generate_async(
                        prompts[0], streaming=True,
                        sampling_params=sampling_params):
                    print(output)

    asyncio.run(task())


class DummyExecutorProxy3(ExecutorBindingsProxy):
    ''' This is for testing the error occur in a Worker process in the Proxy. '''

    def __init__(
        self,
        workers_kwargs,
        model_world_size: int = 1,
        mpi_session=None,
    ) -> None:
        super().__init__(
            workers_kwargs,
            model_world_size,
            mpi_session,
            # The worker process will raise error, and be captured by mpi4py done handler, and redirect to
            # the global error queue.
            worker_cls=DummyExecutorWorker3)


DummyExecutor3 = DummyExecutorMeta("DummyExecutor3", (), {},
                                   proxy_class=DummyExecutorProxy3)


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/4955607")
@skip_single_gpu
def test_llm_get_stats_tp2():
    llm_get_stats_test_harness(tp_size=2)


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/4955607")
@skip_single_gpu
def test_llm_get_stats_async_tp2():
    llm_get_stats_async_test_harness(tp_size=2)


def test_llm_capture_request_error():
    _test_llm_capture_request_error(tp_size=2)


def test_llm_with_postprocess_parallel_tp2():
    run_llm_with_postprocess_parallel(tp_size=2)


def test_llm_with_postprocess_parallel_and_result_handler_tp2():
    run_llm_with_postprocess_parallel_and_result_handler(tp_size=2)


if __name__ == '__main__':

    #test_llm_capture_request_error()

    test_llm_generate_tp2()
