import os
import subprocess  # nosec B404
import sys
import tempfile

import pytest
import torch
from parameterized import parameterized

from tensorrt_llm.hlapi.llm import LLM, SamplingParams
from tensorrt_llm.hlapi.llm_utils import KvCacheConfig
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.util import unittest_name_func

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.llama.model import LLaMAForCausalLM

try:
    from .test_llm import (_test_llm_generate_async, default_model_name,
                           get_model_path, llama_model_path, mixtral_model_name,
                           prompts)
except ImportError:
    from test_llm import (_test_llm_generate_async, default_model_name,
                          get_model_path, llama_model_path, mixtral_model_name,
                          prompts)

skip_single_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="The test needs at least 2 GPUs, skipping")


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


@pytest.fixture(scope="module")
@skip_single_gpu
@pytest.mark.parametrize("enable_executor", [True, False])
def test_llm_loading_from_ckpt_for_tp2(
        engine_from_checkpoint: tempfile.TemporaryDirectory,
        enable_executor: bool):
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    llm = LLM(engine_from_checkpoint.name,
              tokenizer=tokenizer,
              enable_executor=enable_executor)

    sampling_params = SamplingParams(max_new_tokens=8)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I J K"


@skip_single_gpu
def test_llm_generate_tp2(engine_from_checkpoint):
    model_dir = engine_from_checkpoint.name
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)

    llm = LLM(
        model_dir,
        tensor_parallel_size=2,
        tokenizer=tokenizer,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    for output in llm.generate(prompts):
        print(output)


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
    llm = LLM(
        llama_model_path,
        pipeline_parallel_size=2,
        auto_parallel=False,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    sampling_params = SamplingParams(max_new_tokens=8, beam_width=1)
    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I J K"


def llm_end2end_tp2_cases():
    yield ({}, )  # Default options
    yield ({'embedding_parallel_mode': 'NONE'}, )
    yield ({'embedding_parallel_mode': 'SHARDING_ALONG_HIDDEN'}, )
    yield ({
        'embedding_parallel_mode': 'SHARDING_ALONG_VOCAB',
        'share_embedding_table': True
    }, )


@skip_single_gpu
@parameterized.expand(llm_end2end_tp2_cases(), name_func=unittest_name_func)
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

    assert len(llm_additional_options) == 0

    sampling_params = SamplingParams(max_new_tokens=8)
    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I J K"


@skip_single_gpu
def test_llm_multi_node(engine_from_checkpoint: tempfile.TemporaryDirectory):
    nworkers = 2
    test_case_file = os.path.join(os.path.dirname(__file__), "run_llm.py")
    os.path.join(os.path.dirname(__file__), "launch.py")
    command = f"mpirun --allow-run-as-root -n {nworkers} trtllm-hlapi-launch python3 {test_case_file} --model_dir {engine_from_checkpoint.name} --tp_size {nworkers}"
    subprocess.run(command, shell=True, check=True,
                   env=os.environ)  # nosec B603


if __name__ == '__main__':
    test_llm_pp2()
