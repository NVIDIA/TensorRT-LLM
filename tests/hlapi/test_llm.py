import asyncio
import json
import os
import sys
import tempfile
from typing import List

import pytest
import torch
from transformers import AutoTokenizer

from tensorrt_llm.hlapi import LLM, KvCacheConfig, SamplingParams, TokenizerBase
from tensorrt_llm.hlapi.llm_utils import BuildConfig, _ParallelConfig
from tensorrt_llm.hlapi.tokenizer import TransformersTokenizer
from tensorrt_llm.hlapi.utils import get_total_gpu_memory
from tensorrt_llm.models import PretrainedConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar

from tensorrt_llm.models.llama.model import LLaMAForCausalLM

# The unittests are based on the tiny-llama, which is fast to build and run.
# There are other tests based on llama-7B model, such as the end-to-end tests in test_e2e.py, and parallel tests in
# test_llm_multi_gpu.py.


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
def test_llm_build_config():
    build_config = BuildConfig()
    # change some building parameters
    build_config.max_batch_size = 129
    build_config.max_beam_width = 4
    build_config.builder_opt = 3
    build_config.max_num_tokens = 888
    build_config.strongly_typed = True
    build_config.max_seq_len = 333

    llm = LLM(model=llama_model_path,
              build_config=build_config,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))
    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)

    with open(os.path.join(tmpdir.name, "config.json"), "r") as f:
        # read the build_config and check if the parameters are correctly saved
        engine_config = json.load(f)

        pretrained_config = PretrainedConfig.from_dict(
            engine_config["pretrained_config"])
        build_config1 = BuildConfig.from_dict(engine_config["build_config"])

        # Know issue: this will be converted to None after save engine for single-gpu
        build_config1.plugin_config.nccl_plugin = 'float16'
        assert build_config1.max_batch_size == build_config.max_batch_size
        assert build_config1.max_beam_width == build_config.max_beam_width
        assert build_config1.builder_opt == build_config.builder_opt
        assert build_config1.max_num_tokens == build_config.max_num_tokens
        assert build_config1.strongly_typed == build_config.strongly_typed
        assert build_config1.max_seq_len == build_config.max_seq_len


def test_llm_loading_from_hf():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_params = SamplingParams(max_new_tokens=8)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I J K"


@force_ampere
def test_llm_loading_from_ckpt():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    assert tokenizer is not None
    with tempfile.TemporaryDirectory() as ckpt_dir:
        llama = LLaMAForCausalLM.from_hugging_face(llama_model_path)
        llama.save_checkpoint(ckpt_dir)
        del llama

        llm = LLM(
            model=ckpt_dir,
            tokenizer=tokenizer,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        )

        sampling_params = SamplingParams(max_new_tokens=8)

        for output in llm.generate(prompts, sampling_params=sampling_params):
            print(output)
            assert output.outputs[0].text == "D E F G H I J K"


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
    llm = LLM(
        model=llama_model_path,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(llama_model_path),
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    for output in llm.generate(prompts):
        print(output)


def test_llm_without_tokenizer():
    llm = LLM(
        model=llama_model_path,
        skip_tokenizer_init=True,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_params = SamplingParams(end_id=2, pad_id=2, max_new_tokens=8)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert not output.outputs[0].text, \
            "The output should be empty since the tokenizer is missing"
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

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

    tp_size = tp_size if not use_auto_parallel else 1
    world_size = tp_size if use_auto_parallel else None

    llm = LLM(
        model=get_model_path(model_name),
        tokenizer=tokenizer,
        kv_cache_config=kv_cache_config,
        tensor_parallel_size=tp_size,
        auto_parallel=use_auto_parallel,
        world_size=world_size,
    )

    sampling_params = SamplingParams(max_new_tokens=6)

    def test_async(streaming: bool):

        async def task(prompt: str):
            outputs = []
            async for output in llm.generate_async(
                    prompt, streaming=streaming,
                    sampling_params=sampling_params):
                print('output', output)
                outputs.append(output.outputs[0].text)
            print(' '.join(outputs))

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    def test_wait(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt,
                                        streaming=streaming,
                                        sampling_params=sampling_params)
            for output in future:
                print('wait', output)

    def test_non_streaming_usage_wait():
        for prompt in prompts:
            output = llm.generate_async(prompt,
                                        streaming=False,
                                        sampling_params=sampling_params)
            print(output.outputs[0].text)

    def test_future(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(prompt,
                                        streaming=streaming,
                                        sampling_params=sampling_params)
            if streaming is True:
                for output in future:
                    # Do something else and then wait for the result if needed
                    output = output.result(timeout=10)
                    print('future', output.outputs[0].text)
            else:
                # Do something else and then wait for the result if needed
                output = future.result(timeout=10)
                print('future', output.outputs[0].text)

    def test_future_async():

        async def task(prompt: str):
            future = llm.generate_async(prompt,
                                        streaming=False,
                                        sampling_params=sampling_params)
            output = await future.aresult()
            print('future', output.outputs[0].text)

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
def llm_for_sampling_params() -> LLM:
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    return llm


@force_ampere
def test_generate_with_sampling_params_per_prompt(llm_for_sampling_params: LLM):
    llm = llm_for_sampling_params
    sampling_params_list = [
        SamplingParams(end_id=-1, pad_id=-1) for _ in range(2)
    ]
    sampling_params_list[0].max_new_tokens = 4
    sampling_params_list[1].max_new_tokens = 8

    for i, output in enumerate(
            llm.generate(prompts, sampling_params=sampling_params_list)):
        output_len = len(output.outputs[0].token_ids)
        print(f"output_len: {output_len}")
        assert output_len <= sampling_params_list[i].max_new_tokens


@force_ampere
@pytest.fixture(scope="module")
@pytest.mark.parametrize(
    "sampling_params",
    [
        # temperature
        SamplingParams(
            max_new_tokens=6, temperature=0.5, beam_search_diversity_rate=0.5),
        # topK
        SamplingParams(max_new_tokens=6, top_k=10, top_p=0.92),
        # topP
        SamplingParams(max_new_tokens=6, top_p=0.92),
        # penalty
        SamplingParams(max_new_tokens=8,
                       length_penalty=1.0,
                       presence_penalty=0.0,
                       repetition_penalty=1.0,
                       min_length=5),
        # early stopping
        SamplingParams(max_new_tokens=6, early_stopping=5),
    ])
def test_generate_with_SamplingConfig(llm_for_sampling_params: LLM,
                                      sampling_params: SamplingParams):
    llm = llm_for_sampling_params

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)


@force_ampere
def test_generate_with_beam_search():
    build_config = BuildConfig()
    build_config.max_beam_width = 2
    build_config.max_num_tokens = 20

    llm = LLM(
        model=llama_model_path,
        build_config=build_config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_params = SamplingParams(max_new_tokens=6, beam_width=2)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert len(output.outputs) == 2
        assert similar(output.outputs[0].text, "D E F G H I")
        assert similar(output.outputs[1].text, "D E F G I J")


@force_ampere
def test_generate_with_streaming_llm():
    # TODO[chunweiy]: Test with larger size when the underlying support is ready
    build_config = BuildConfig()
    build_config.plugin_config.streamingllm = True
    kv_cache_config = KvCacheConfig(max_attention_window=64,
                                    sink_token_length=4)

    llm = LLM(model=llama_model_path,
              kv_cache_config=kv_cache_config,
              build_config=build_config)

    # Check the plugin config is correctly set
    assert build_config.plugin_config.streamingllm is True
    #assert build_config.plugin_config.use_paged_context_fmha is False

    sampling_params = SamplingParams(max_new_tokens=4)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert output.outputs[0].text == "D E F G"
        print(output)


def test_parallel_config():
    config = _ParallelConfig()
    config.tp_size = 2
    config.pp_size = 2
    assert config.world_size == 4
    config.world_size = 4  # should not raise exception

    with pytest.raises(ValueError):
        config.world_size = 5


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("gather_context_logits", [True, False])
@pytest.mark.parametrize("gather_generation_logits", [True, False])
@pytest.mark.parametrize("return_log_probs", [True])  # prune space
def test_generate_with_OutputConfig(gather_context_logits: bool,
                                    gather_generation_logits: bool,
                                    return_log_probs: bool):
    if not (gather_context_logits or gather_generation_logits):  # prune space
        return

    build_config = BuildConfig()
    build_config.gather_context_logits = gather_context_logits
    build_config.gather_generation_logits = gather_generation_logits

    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        build_config=build_config,
    )
    sampling_params = SamplingParams(
        max_new_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        return_log_probs=return_log_probs)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(prompts[0].split()) + 1 == output.context_logits.shape[0]
        if gather_generation_logits:
            assert output.outputs[0].generation_logits is not None
            assert sampling_params.max_new_tokens == output.outputs[
                0].generation_logits.shape[0]
        if return_log_probs:
            assert output.outputs[0].logprobs is not None

        print(output)


@force_ampere
def test_generate_with_stop_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_params = SamplingParams(max_new_tokens=6, stop_words=[[11]])

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G H I"


@force_ampere
def test_generate_with_bad_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    # TODO[chunweiy]: Consider to make the generate api accept bad_words as a list of strings
    bad_words = [llm.tokenizer.encode("H I", add_special_tokens=False)]
    print('bad_words:', bad_words)

    sampling_params = SamplingParams(max_new_tokens=6, bad_words=bad_words)
    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "D E F G HI"


@force_ampere
def test_generate_with_embedding_bias():
    llm = LLM(
        llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    biased_word_id = llm.tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[biased_word_id] = torch.finfo(torch.float32).max

    sampling_params = SamplingParams(max_new_tokens=6,
                                     embedding_bias=embedding_bias)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "Z Z Z Z Z Z"


@force_ampere
def test_generate_with_logits_post_processor():
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]

    def logits_post_processor(req_id: int, logits: torch.Tensor,
                              ids: List[List[int]], stream_ptr: int):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits[:] = float("-inf")
            logits[..., biased_word_id] = 0

    llm = LLM(llama_model_path,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
              logits_post_processor_map={"my_logits_pp": logits_post_processor})

    sampling_params = SamplingParams(max_new_tokens=6,
                                     logits_post_processor_name="my_logits_pp")

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "Z Z Z Z Z Z"


@force_ampere
def test_generate_block_reuse():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4,
                                      enable_block_reuse=True),
    )

    # Check the configurations are correctly set
    assert llm.args.build_config.plugin_config.use_paged_context_fmha is True
    assert llm.args.build_config.plugin_config.paged_kv_cache is True

    sampling_params = SamplingParams(max_new_tokens=6)

    prompts = ["A B C", "A B C D"]
    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)


# TODO[chunweiy]: Add test for loading inmemory model

if __name__ == '__main__':
    test_llm_loading_from_hf()
