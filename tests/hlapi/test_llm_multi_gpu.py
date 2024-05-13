import os
import sys
import tempfile

import pytest
import torch

from tensorrt_llm.hlapi.llm import LLM, KvCacheConfig, ModelConfig
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.utils import get_total_gpu_memory

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.llama.model import LLaMAForCausalLM

try:
    from .test_llm import (default_model_name, get_model_path, llama_model_path,
                           mixtral_model_name, skip_single_gpu,
                           test_llm_generate_async)
except ImportError:
    from test_llm import (default_model_name, get_model_path, llama_model_path,
                          mixtral_model_name, skip_single_gpu,
                          test_llm_generate_async)

prompts = ["A B C"]


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

        config = ModelConfig(ckpt_dir)
        assert config.parallel_config.tp_size == tp_size
        llm = LLM(
            config,
            tokenizer=tokenizer,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        )

    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)

    return tmpdir


@pytest.fixture(scope="module")
@skip_single_gpu
def test_llm_loading_from_ckpt_for_tp2(
        engine_from_checkpoint: tempfile.TemporaryDirectory):
    config = ModelConfig(engine_from_checkpoint.name)
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    llm = LLM(config, tokenizer=tokenizer)

    sampling_config = llm.get_default_sampling_config()
    assert sampling_config is not None

    for output in llm.generate(prompts):
        print(output)
        assert output.text == "<s> A B C D E F G H I J K L M N O P Q R S T U V W X Y Z\nA B C D E F G H"


@skip_single_gpu
def test_llm_generate_tp2(engine_from_checkpoint):
    model_dir = engine_from_checkpoint.name
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    config = ModelConfig(model_dir)
    config.parallel_config.tp_size = 2

    llm = LLM(
        config,
        tokenizer=tokenizer,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    for output in llm.generate(prompts):
        print(output)


@skip_single_gpu
@pytest.mark.parametrize("use_auto_parallel", [True, False],
                         ids=["enable_auto_parallel", "disable_auto_parallel"])
def test_llm_generate_async_tp2(
        use_auto_parallel, engine_from_checkpoint: tempfile.TemporaryDirectory):
    model_dir = engine_from_checkpoint.name if not use_auto_parallel else default_model_name
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    test_llm_generate_async(
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
    config = ModelConfig(get_model_path(mixtral_model_name))
    config.parallel_config.tp_size = 2
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    for output in llm.generate(prompts):
        print(output)


def test_llm_pp2():
    config = ModelConfig(llama_model_path)
    config.parallel_config.pp_size = 2
    config.parallel_config.auto_parallel = False
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    for output in llm.generate(prompts):
        assert output.text == "<s> A B C D E F G H I J K L M N O P Q R S T U V W X Y Z\nA B C D E F G H"


if __name__ == '__main__':
    test_llm_generate_async_tp2(use_auto_parallel=True)
    test_llm_generate_async_tp2(use_auto_parallel=False)
    test_llm_pp2()
