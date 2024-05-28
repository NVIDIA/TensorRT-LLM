import asyncio
import json
import os
import pickle
import sys
import tempfile
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm.hlapi.llm import (LLM, BuildConfig, KvCacheConfig,
                                    ModelConfig, OutputConfig, ParallelConfig,
                                    PretrainedConfig, SamplingConfig,
                                    StreamingLLMParam, TokenizerBase)
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.utils import get_total_gpu_memory

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar

from tensorrt_llm.models.llama.model import LLaMAForCausalLM

# The unittests are based on the tiny-llama, which is fast to build and run.
# There are other tests based on llama-7B model, such as the end-to-end tests in test_e2e.py, and parallel tests in test_llm_multi_gpu.py.


def get_model_path(model_name):
    engine_dir = os.environ.get('LLM_ENGINE_DIR', None)
    if engine_dir:
        return engine_dir
    return str(llm_models_root() / model_name)


default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
mixtral_model_name = "Mixtral-8x7B-v0.1"

llama_model_path = get_model_path(default_model_name)
llm_engine_dir = os.environ.get('LLM_ENGINE_DIR', './tmp.engine')
prompts = ["A B C"]


@force_ampere
def test_ModelConfig_build_config():
    config = ModelConfig(llama_model_path)
    assert config.build_config is not None

    # change some building parameters
    config.build_config.max_batch_size = 129
    config.build_config.max_beam_width = 4
    config.build_config.builder_opt = 3
    config.build_config.max_num_tokens = 888
    config.build_config.strongly_typed = True
    config.build_config.max_output_len = 333

    llm = LLM(config,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))
    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)

    with open(os.path.join(tmpdir.name, "config.json"), "r") as f:
        # read the build_config and check if the parameters are correctly saved
        engine_config = json.load(f)

        pretrained_config = PretrainedConfig.from_dict(
            engine_config["pretrained_config"])
        build_config = BuildConfig.from_dict(engine_config["build_config"])

        # Know issue: this will be converted to None after save engine for single-gpu
        build_config.plugin_config.nccl_plugin = 'float16'
        assert build_config.max_batch_size == config.build_config.max_batch_size
        assert build_config.max_beam_width == config.build_config.max_beam_width
        assert build_config.builder_opt == config.build_config.builder_opt
        assert build_config.max_num_tokens == config.build_config.max_num_tokens
        assert build_config.strongly_typed == config.build_config.strongly_typed
        assert build_config.max_output_len == config.build_config.max_output_len


def test_llm_loading_from_hf():
    config = ModelConfig(llama_model_path)
    # The performance-related flags are turned on eagerly to check the functionality

    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        enable_chunked_context=False,
    )

    sampling_config = SamplingConfig()
    assert sampling_config is not None
    sampling_config.max_new_tokens = 8

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)
        assert output.text == "D E F G H I J K"


@force_ampere
def test_llm_loading_from_ckpt():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    assert tokenizer is not None
    with tempfile.TemporaryDirectory() as ckpt_dir:
        llama = LLaMAForCausalLM.from_hugging_face(llama_model_path)
        llama.save_checkpoint(ckpt_dir)
        del llama

        config = ModelConfig(ckpt_dir)
        llm = LLM(
            config,
            tokenizer=tokenizer,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        )

        sampling_config = SamplingConfig()
        assert sampling_config is not None
        sampling_config.max_new_tokens = 8

        for output in llm.generate(prompts, sampling_config=sampling_config):
            print(output)
            assert output.text == "D E F G H I J K"


class MyTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir,
                                                  **kwargs)
        return MyTokenizer(tokenizer)

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_encode_plus(self, texts: List[str], **kwargs) -> dict:
        return self.tokenizer.batch_encode_plus(texts, **kwargs)


def test_llm_with_customized_tokenizer():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(config.model_dir),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    for output in llm.generate(prompts):
        print(output)


def test_llm_without_tokenizer():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_config = SamplingConfig(end_id=2, pad_id=2, max_new_tokens=8)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_config=sampling_config):
        assert not output.text, "The output should be empty since the tokenizer is missing"
        print(output)


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


def test_llm_generate_async():
    _test_llm_generate_async()


def _test_llm_generate_async(model_name=default_model_name,
                             tp_size: int = 1,
                             use_auto_parallel: bool = False,
                             tokenizer=None):
    if "Mixtral" in model_name and use_auto_parallel:
        pytest.skip("Auto parallel is not supported for Mixtral models")

    config = ModelConfig(llama_model_path)
    if use_auto_parallel:
        config.parallel_config.auto_parallel = True
        config.parallel_config.world_size = tp_size
    else:
        config.parallel_config.tp_size = tp_size

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    devices = config.parallel_config.devices
    if torch.cuda.get_device_properties(devices[0]).major >= 8:
        kv_cache_config.enable_block_reuse = True

    llm = LLM(
        config,
        tokenizer=tokenizer,
        kv_cache_config=kv_cache_config,
    )
    llm.save("./tmp.engine.8")  # DEBUG

    sampling_config = SamplingConfig()
    sampling_config.max_new_tokens = 6

    def test_async(streaming: bool):

        async def task(prompt: str):
            outputs = []
            async for output in llm.generate_async(
                    prompt, streaming=streaming,
                    sampling_config=sampling_config):
                print('output', output)
                outputs.append(output.text)
            print(' '.join(outputs))

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    def test_wait(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt,
                                        streaming=streaming,
                                        sampling_config=sampling_config)
            for output in future:
                print('wait', output)

    def test_non_streaming_usage_wait():
        for prompt in prompts:
            output = llm.generate_async(prompt,
                                        streaming=False,
                                        sampling_config=sampling_config)
            print(output.text)

    def test_future(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt,
                                        streaming=streaming,
                                        sampling_config=sampling_config)
            if streaming is True:
                for output in future:
                    # Do something else and then wait for the result if needed
                    output = output.result(timeout=10)
                    print('future', output.text)
            else:
                # Do something else and then wait for the result if needed
                output = future.result(timeout=10)
                print('future', output.text)

    def test_future_async():

        async def task(prompt: str):
            future = llm.generate_async(prompt,
                                        streaming=False,
                                        sampling_config=sampling_config)
            output = await future.aresult()
            print('future', output.text)

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    test_async(streaming=True)
    test_async(streaming=False)
    test_wait(streaming=True)
    test_wait(streaming=False)
    test_future(streaming=True)
    test_future(streaming=False)
    test_future_async()
    test_non_streaming_usage_wait()


@pytest.fixture(scope="module")
def llm_for_sampling_config() -> LLM:
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    return llm


@force_ampere
def test_generate_with_sampling_config_per_prompt(llm_for_sampling_config: LLM):
    llm = llm_for_sampling_config
    sampling_configs = [SamplingConfig() for _ in range(2)]
    sampling_configs[0].max_new_tokens = 4
    sampling_configs[1].max_new_tokens = 8
    for sc in sampling_configs:
        sc.end_id = -1
        sc.pad_id = -1

    for off, output in enumerate(
            llm.generate(prompts, sampling_config=sampling_configs)):
        output_len = len(output.token_ids)
        print(f"output_len: {output_len}")
        assert output_len <= sampling_configs[off].max_new_tokens


@force_ampere
@pytest.fixture(scope="module")
@pytest.mark.parametrize(
    "sampling_config",
    [
        # temperature
        SamplingConfig(
            max_new_tokens=6, temperature=0.5, beam_search_diversity_rate=0.5),
        # topK
        SamplingConfig(max_new_tokens=6, top_k=10, top_p=0.92),
        # topP
        SamplingConfig(max_new_tokens=6, top_p=0.92),
        # penalty
        SamplingConfig(max_new_tokens=8,
                       length_penalty=1.0,
                       presence_penalty=0.0,
                       repetition_penalty=1.0,
                       min_length=5),
        # early stopping
        SamplingConfig(max_new_tokens=6, early_stopping=5),
    ])
def test_generate_with_SamplingConfig(llm_for_sampling_config: LLM,
                                      sampling_config: SamplingConfig):
    llm = llm_for_sampling_config

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)


@force_ampere
def test_generate_with_beam_search():
    config = ModelConfig(llama_model_path)
    config.build_config.max_beam_width = 2
    config.build_config.max_num_tokens = 20

    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_config = SamplingConfig()
    sampling_config.max_new_tokens = 6
    sampling_config.beam_width = 2

    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)
        assert len(output.text) == 2
        assert len(output.token_ids) == 2
        assert similar(output.text[0], "D E F G H I")
        assert similar(output.text[1], "D E F G I J")


@force_ampere
def test_generate_with_streaming_llm():
    config = ModelConfig(llama_model_path)
    # TODO[chunweiy]: Test with larger size when the underlying support is ready
    llm = LLM(config, streaming_llm=StreamingLLMParam(64, 4))

    # Check the plugin config is correctly set
    assert config.build_config.plugin_config.streamingllm is True
    assert config.build_config.plugin_config.use_paged_context_fmha is False

    sampling_config = llm.get_default_sampling_config()
    assert sampling_config
    sampling_config.max_new_tokens = 4

    for output in llm.generate(prompts, sampling_config=sampling_config):
        assert output.text == "D E F G"
        print(output)


def test_sampling_config():
    sc = SamplingConfig()
    sc.max_new_tokens = 1024

    sc0 = pickle.loads(pickle.dumps(sc))
    assert sc0.max_new_tokens == 1024


def test_parallel_config():
    config = ParallelConfig()
    config.tp_size = 2
    config.pp_size = 2
    assert config.world_size == 4
    config.world_size = 4  # should not raise exception

    with pytest.raises(ValueError):
        config.world_size = 5


def test_OutputConfig_pickle():
    output_config = OutputConfig()
    output_config.exclude_input_from_output = True
    output_config.return_context_logits = True
    output_config.return_generation_logits = True

    output_config0 = pickle.loads(pickle.dumps(output_config))

    assert output_config == output_config0


def test_SamplingConfig_pickle():
    sc = SamplingConfig()
    sc.max_new_tokens = 1024
    sc.end_id = 2
    sc.pad_id = 2
    sc.beam_width = 2
    sc.temperature = 0.5
    sc.top_k = 10
    sc.top_p = 0.92
    sc.length_penalty = 1.0
    sc.presence_penalty = None
    sc.repetition_penalty = 1.0
    sc.min_length = 5
    sc.early_stopping = 5

    sc0 = pickle.loads(pickle.dumps(sc))

    assert sc == sc0


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("return_log_probs", [True])  # prune space
def test_generate_with_OutputConfig(gather_context_logits: bool,
                                    gather_generation_logits: bool,
                                    return_log_probs: bool):
    if not (gather_context_logits or gather_generation_logits):  # prune space
        return

    config = ModelConfig(llama_model_path)
    config.build_config.gather_context_logits = gather_context_logits
    config.build_config.gather_generation_logits = gather_generation_logits
    config.build_config.return_log_probs = return_log_probs

    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    output_config = OutputConfig(
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        return_log_probs=return_log_probs)
    sampling_config = SamplingConfig(max_new_tokens=8)

    for output in llm.generate(prompts,
                               output_config=output_config,
                               sampling_config=sampling_config):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(prompts[0].split()) + 1 == output.context_logits.shape[0]
        if gather_generation_logits:
            assert output.generation_logits is not None
            assert sampling_config.max_new_tokens == output.generation_logits.shape[
                1]
        if return_log_probs:
            assert output.log_probs is not None

        print(output)


@force_ampere
def test_generate_with_stop_words():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_config = SamplingConfig()
    sampling_config.max_new_tokens = 6

    for output in llm.generate(prompts,
                               sampling_config=sampling_config,
                               stop_words=[[11]]):
        print(output)
        assert output.text == "D E F G H I"


@force_ampere
def test_generate_with_bad_words():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_config = SamplingConfig()
    sampling_config.max_new_tokens = 6

    tokenizer = AutoTokenizer.from_pretrained(llama_model_path,
                                              add_prefix_space=False)

    # TODO[chunweiy]: Consider to make the generate api accept bad_words as a list of strings
    bad_words = tokenizer(["H", "I"]).input_ids
    bad_words = [row[1] for row in tokenizer(["H", "I"]).input_ids]
    bad_words = [bad_words]
    print('bad_words:', bad_words)

    for output in llm.generate(prompts,
                               sampling_config=sampling_config,
                               bad_words=bad_words):
        print(output)
        assert output.text == "D E F G H J"


@force_ampere
def test_generate_block_reuse():
    config = ModelConfig(llama_model_path)
    llm = LLM(
        config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4,
                                      enable_block_reuse=True),
    )

    # Check the configurations are correctly set
    assert config.build_config.plugin_config.use_paged_context_fmha is True
    assert config.build_config.plugin_config.paged_kv_cache is True

    sampling_config = SamplingConfig(max_new_tokens=6)

    prompts = ["A B C", "A B C D"]
    for output in llm.generate(prompts, sampling_config=sampling_config):
        print(output)


# TODO[chunweiy]: Add test for loading inmemory model

if __name__ == '__main__':
    test_llm_loading_from_hf()
