import asyncio
import contextlib
import datetime
import json
import os
import random
import sys
import time
from typing import List, Optional, Union

import datasets
import pytest
import torch
import transformers

from tensorrt_llm import LLM
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor import GenerationResultBase, RequestError
from tensorrt_llm.llmapi import (KvCacheConfig, KvCacheRetentionConfig,
                                 LookaheadDecodingConfig, RequestOutput)
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.llmapi.llm_args import DynamicBatchConfig, SchedulerConfig
from tensorrt_llm.llmapi.llm_utils import _ParallelConfig
from tensorrt_llm.llmapi.tokenizer import (TokenizerBase, TransformersTokenizer,
                                           load_hf_tokenizer)
from tensorrt_llm.sampling_params import LogitsProcessor, SamplingParams
from tensorrt_llm.serve.openai_protocol import CompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.postprocess_handlers import (ChatPostprocArgs,
                                                     chat_stream_post_processor)

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from gc_utils import assert_resource_freed
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar, altered_env

# isort: on

# The unittests are based on the tiny-llama, which is fast to build and run.
# There are other tests based on llama-7B model, such as the end-to-end tests in test_e2e.py, and parallel tests in
# test_llm_multi_gpu.py.

pytestmark = pytest.mark.threadleak(enabled=False)


def get_model_path(model_name):
    engine_dir = os.environ.get('LLM_ENGINE_DIR', None)
    if engine_dir:
        return engine_dir
    return str(llm_models_root() / model_name)


def check_output(outputs: List[RequestOutput],
                 references: Union[List[str], List[List[str]]],
                 *,
                 similar_threshold: float = 0.8,
                 finish_reasons: Optional[List[str]] = None,
                 stop_reasons: Optional[List[Union[int, str]]] = None):
    assert len(outputs) == len(references)

    for i, (output, reference) in enumerate(zip(outputs, references)):
        if isinstance(reference, list):
            # N output
            assert len(output.outputs) == len(reference)
            for j, (out, ref) in enumerate(zip(output.outputs, reference)):
                assert similar(out.text, ref, threshold=similar_threshold)
                if finish_reasons is not None:
                    assert out.finish_reason == finish_reasons[i][j]
                if stop_reasons is not None:
                    assert out.stop_reason == stop_reasons[i][j]
        else:
            out = output.outputs[0]
            assert similar(out.text, reference, threshold=similar_threshold)
            if finish_reasons is not None:
                assert out.finish_reason == finish_reasons[i]
            if stop_reasons is not None:
                assert out.stop_reason == stop_reasons[i]


def llm_test_harness(model_dir: str,
                     inputs: List[str],
                     references: List[str],
                     *,
                     sampling_params: Optional[SamplingParams] = None,
                     similar_threshold: float = 0.8,
                     **llm_kwargs):

    tp_size = llm_kwargs.get('tensor_parallel_size', 1)
    pp_size = llm_kwargs.get('pipeline_parallel_size', 1)
    world_size = tp_size * pp_size
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"world_size ({world_size}) is greater than available GPUs ({torch.cuda.device_count()})"
        )

    tokenizer = llm_kwargs.pop('tokenizer', None)
    if tokenizer is None:
        tokenizer = model_dir

    with assert_resource_freed(LLM, model_dir, tokenizer, **llm_kwargs) as llm:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        print(outputs)
        check_output(outputs, references, similar_threshold=similar_threshold)


def llm_check_output(llm: LLM,
                     inputs: List[str],
                     references: List[str],
                     *,
                     sampling_params: Optional[SamplingParams] = None,
                     similar_threshold: float = 0.8,
                     finish_reasons: Optional[List[str]] = None,
                     stop_reasons: Optional[List[Union[int, str]]] = None,
                     **gen_kwargs):
    outputs = llm.generate(inputs,
                           sampling_params=sampling_params,
                           **gen_kwargs)
    print(outputs)
    check_output(outputs,
                 references,
                 similar_threshold=similar_threshold,
                 finish_reasons=finish_reasons,
                 stop_reasons=stop_reasons)


default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
mixtral_model_name = "Mixtral-8x7B-v0.1"

llama_model_path = get_model_path(default_model_name)

cnn_dailymail_path = str(llm_models_root() / "datasets" / "cnn_dailymail")
alpaca_chinese_path = str(llm_models_root() / "datasets" / "silk-road" /
                          "alpaca-data-gpt4-chinese")

prompts = ["A B C"]
global_kvcache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

# python api does not seem to support extra tokens needed for prompt tuning + reuse.
# disable block reuse for those tests.
# TODO: Add extra tokens to prompt tuning unit tests.
global_kvcache_config_no_reuse = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                               enable_block_reuse=False)


@pytest.mark.part0
def test_llm_loading_from_hf():
    sampling_params = SamplingParams(max_tokens=8)
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=sampling_params,
                     kv_cache_config=global_kvcache_config)


class MyTokenizer(TokenizerBase):
    ''' A wrapper for the Transformers' tokenizer.
    This is the default tokenizer for LLM. '''

    @classmethod
    def from_pretrained(cls, pretrained_model_dir: str, **kwargs):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_dir, **kwargs)
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
        return self.tokenizer(texts, **kwargs)


@pytest.mark.part0
def test_llm_with_customized_tokenizer():
    llm = LLM(
        model=llama_model_path,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(llama_model_path),
        kv_cache_config=global_kvcache_config,
    )

    for output in llm.generate(prompts):
        print(output)


@pytest.mark.part0
def test_llm_without_tokenizer():
    llm = LLM(
        model=llama_model_path,
        skip_tokenizer_init=True,
        kv_cache_config=global_kvcache_config,
    )

    sampling_params = SamplingParams(end_id=2, pad_id=2, max_tokens=8)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert not output.outputs[0].text, \
            "The output should be empty since the tokenizer is missing"
        print(output)


@pytest.mark.part0
def test_llm_with_kv_cache_retention_config():
    kv_cache_retention_config = KvCacheRetentionConfig([
        KvCacheRetentionConfig.TokenRangeRetentionConfig(
            0, 2, 30, datetime.timedelta(seconds=30))
    ], 80, None, tllm.KvCacheTransferMode.DRAM, "test_dir")

    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)

    for output in llm.generate(
            prompts, kv_cache_retention_config=kv_cache_retention_config):
        print(output)


@pytest.mark.parametrize('backend', ["HF", "TRTLLM"])
@pytest.mark.parametrize(
    'tokenizer_dir, clean_up_tokenization_spaces, threshold',
    [
        (get_model_path('gpt2'), False, 0.95),  # BPE
        (get_model_path('bert/bert-base-uncased'), True, 0.95),  # WordPiece
        (get_model_path('t5-small'), True, 0.95),  # SentencePiece
        (get_model_path('starcoder2-3b'), False, 0.95),
        (get_model_path('falcon-7b-instruct'), False, 0.95),
        (get_model_path('llama-models-v2/llama-v2-7b-hf'), False, 0.95),
        (get_model_path('codellama/CodeLlama-7b-Instruct-hf'), False, 0.95),
        (llama_model_path, False, 0.95),
        (get_model_path(mixtral_model_name), False, 0.95),
        (get_model_path('llama-3.1-model/Meta-Llama-3.1-8B'), False, 0.95),
        (get_model_path('DeepSeek-R1/DeepSeek-R1'), False, 0.95)
    ])
@pytest.mark.part0
def test_tokenizer_decode_incrementally(tokenizer_dir: str,
                                        clean_up_tokenization_spaces: bool,
                                        threshold: float, backend: str, mocker):
    import tensorrt_llm.llmapi.tokenizer
    mocker.patch.object(tensorrt_llm.llmapi.tokenizer,
                        "TLLM_INCREMENTAL_DETOKENIZATION_BACKEND", backend)
    assert tensorrt_llm.llmapi.tokenizer.TLLM_INCREMENTAL_DETOKENIZATION_BACKEND == backend

    random.seed(42)

    num_samples = 100
    cnn_dailymail = datasets.load_dataset(cnn_dailymail_path,
                                          name='3.0.0',
                                          split='train',
                                          trust_remote_code=True)
    alpaca_chinese = datasets.load_dataset(alpaca_chinese_path,
                                           split='train',
                                           trust_remote_code=True)
    dataset = cnn_dailymail['article'][:num_samples // 2] + alpaca_chinese[
        'output_zh'][:num_samples // 2]

    tokenizer = TransformersTokenizer.from_pretrained(tokenizer_dir,
                                                      legacy=False,
                                                      padding_side='left',
                                                      truncation_side='left',
                                                      trust_remote_code=True,
                                                      use_fast=True)

    num_perfect = 0
    for text in dataset:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        seq_len = len(token_ids)
        prompt_len = random.randint(1, seq_len // 2)
        decoded_text, states = tokenizer.decode_incrementally(
            token_ids[:prompt_len])
        for i in range(prompt_len, len(token_ids)):
            decoded_text, states = tokenizer.decode_incrementally(
                [token_ids[i]], decoded_text, states)

        if clean_up_tokenization_spaces and tokenizer.clean_up_tokenization_spaces:
            decoded_text = tokenizer.clean_up_tokenization(decoded_text)
        reference = tokenizer.decode(token_ids)
        if decoded_text == reference:
            num_perfect += 1
        else:
            # For non-perfect matching cases, decoded_text should also be very similar to the reference
            assert similar(decoded_text, reference, 0.99)
    print(f"Perfect matching ratio: {num_perfect / num_samples * 100}%")
    assert num_perfect / num_samples >= threshold


@pytest.mark.part0
def test_llm_generate_async():
    _test_llm_generate_async()


def _test_llm_generate_async(model_name=default_model_name,
                             tp_size: int = 1,
                             tokenizer=None):
    llm = LLM(
        model=get_model_path(model_name),
        tokenizer=tokenizer,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
    )

    sampling_params = SamplingParams(max_tokens=6)

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


@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.part0
@pytest.mark.mpi_ray_parity
def test_llm_generate_async_with_stream_interval(chunked):
    model_path = get_model_path('llama-models-v2/llama-v2-7b-hf')
    max_num_tokens = 256
    with LLM(model_path,
             max_num_tokens=max_num_tokens,
             stream_interval=4,
             enable_chunked_prefill=chunked) as llm:
        sampling_params = SamplingParams(max_tokens=13,
                                         ignore_eos=True,
                                         detokenize=False)
        step = 0
        last_step_len = 0
        prompt = "The capital of France is "
        if chunked:
            prompt = prompt * max_num_tokens
        for output in llm.generate_async(prompt,
                                         sampling_params=sampling_params,
                                         streaming=True):
            current_step_len = len(output.outputs[0].token_ids)
            # The output lens of each step need to be [1, 3, 4, 4, 1]
            if step == 0:
                assert current_step_len == 1
            elif step == 1:
                assert current_step_len - last_step_len == 3
            elif step == 2 or step == 3:
                assert current_step_len - last_step_len == 4
            else:
                assert current_step_len - last_step_len == 1
            step += 1
            last_step_len = current_step_len


@pytest.mark.part0
def test_parallel_config():
    config = _ParallelConfig()
    config.tp_size = 2
    config.pp_size = 2
    assert config.world_size == 4
    config.world_size = 4  # should not raise exception

    with pytest.raises(ValueError):
        config.world_size = 5


@force_ampere
@pytest.mark.part0
def test_generate_with_stop_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    )
    stop_id = llm.tokenizer.encode("N", add_special_tokens=False)[-1]

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M"],
                     sampling_params=SamplingParams(end_id=stop_id),
                     finish_reasons=['stop'],
                     stop_reasons=[None])

    llm_check_output(llm,
                     prompts, ["D E F G H"],
                     sampling_params=SamplingParams(max_tokens=5),
                     finish_reasons=['length'],
                     stop_reasons=[None])

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M"],
                     sampling_params=SamplingParams(stop_token_ids=[stop_id]),
                     finish_reasons=['stop'],
                     stop_reasons=[stop_id])

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M N"],
                     sampling_params=SamplingParams(
                         stop_token_ids=[stop_id],
                         include_stop_str_in_output=True),
                     finish_reasons=['stop'],
                     stop_reasons=[stop_id])

    llm_check_output(llm,
                     prompts, ["D E F G H"],
                     sampling_params=SamplingParams(stop="I J"),
                     finish_reasons=['stop'],
                     stop_reasons=["I J"])

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M"],
                     sampling_params=SamplingParams(stop="I E", max_tokens=10),
                     finish_reasons=['length'],
                     stop_reasons=[None])

    llm_check_output(llm,
                     prompts, ["D E F G H I J"],
                     sampling_params=SamplingParams(
                         stop="I J", include_stop_str_in_output=True),
                     finish_reasons=['stop'],
                     stop_reasons=["I J"])

    llm_check_output(llm,
                     prompts, ["D E F G H"],
                     sampling_params=SamplingParams(stop=["F E", "I J"],
                                                    stop_token_ids=[stop_id]),
                     finish_reasons=['stop'],
                     stop_reasons=["I J"])


@force_ampere
@pytest.mark.part0
@pytest.mark.parametrize("model_path", [
    get_model_path('gemma/gemma-3-1b-it'),
])
def test_generate_with_detokenization_stop_words(model_path):
    llm = LLM(
        model=model_path,
        kv_cache_config=global_kvcache_config,
    )

    # Format the prompt using chat template
    messages = [{
        "role": "user",
        "content": "Say exactly: Hello there! How can I help"
    }]

    formatted_prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    detokenization_prompts = [formatted_prompt]

    # Test case 1: Stop word "How" should be detected after detokenization
    llm_check_output(llm,
                     detokenization_prompts, ["Hello there!"],
                     sampling_params=SamplingParams(stop="How", max_tokens=10),
                     finish_reasons=['stop'],
                     stop_reasons=["How"])

    # Test case 2: Stop word "there" should be detected after detokenization
    llm_check_output(llm,
                     detokenization_prompts, ["Hello"],
                     sampling_params=SamplingParams(stop="there",
                                                    max_tokens=10),
                     finish_reasons=['stop'],
                     stop_reasons=["there"])

    # Test case 3: Stop word that should not be found after detokenization
    llm_check_output(llm,
                     detokenization_prompts, ["Hello there! How can I help"],
                     sampling_params=SamplingParams(stop="XYZ", max_tokens=10),
                     finish_reasons=['length'],
                     stop_reasons=[None])

    # Test case 4: Multiple stop words, one should be found after detokenization
    llm_check_output(llm,
                     detokenization_prompts, ["Hello"],
                     sampling_params=SamplingParams(stop=["XYZ", "there"],
                                                    max_tokens=10),
                     finish_reasons=['stop'],
                     stop_reasons=["there"])


@force_ampere
@pytest.mark.part0
@pytest.mark.parametrize("model_path", [
    get_model_path('gemma/gemma-3-1b-it'),
])
def test_generate_with_detokenization_stop_words_streaming(model_path):
    llm = LLM(
        model=model_path,
        kv_cache_config=global_kvcache_config,
    )

    # Format the prompt using chat template
    messages = [{
        "role": "user",
        "content": "Say exactly: Hello there! How can I help"
    }]

    formatted_prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(stop="How", max_tokens=10)

    for output in llm.generate_async(formatted_prompt,
                                     sampling_params=sampling_params,
                                     streaming=True):
        if output.outputs[0].finish_reason == 'stop':
            assert output.outputs[0].stop_reason == "How"
            break
        elif output.outputs[0].finish_reason == 'length':
            assert False, f"Expected to find stop word 'How' but reached max_tokens. Generated: {output.outputs[0].text}"


@force_ampere
@pytest.mark.part0
def test_generate_with_bad_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    )

    bad_id = llm.tokenizer.encode("N", add_special_tokens=False)[-1]

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M\n\nI hope this"],
                     sampling_params=SamplingParams(max_tokens=15,
                                                    bad_token_ids=[bad_id]))

    llm_check_output(llm,
                     prompts, ["D E F G H I K L M N O P Q R S"],
                     sampling_params=SamplingParams(max_tokens=15, bad="I J"))

    llm_check_output(llm,
                     prompts, ["D E F G H I K L M N O P Q R S"],
                     sampling_params=SamplingParams(max_tokens=15,
                                                    bad=["F E", "I J"]))


class MyLogitsProcessor(LogitsProcessor):

    def __init__(self, biased_word_id):
        self.biased_word_id = biased_word_id

    def __call__(self, req_id: int, logits: torch.Tensor, ids: List[List[int]],
                 stream_ptr: int, client_id: Optional[int]):
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(
            stream_ptr)
        with torch.cuda.stream(stream):
            logits[:] = float("-inf")
            logits[..., self.biased_word_id] = 0


def tinyllama_logits_processor_test_harness(backend=None, **llm_kwargs):
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    sampling_params = SamplingParams(
        max_tokens=6, logits_processor=MyLogitsProcessor(biased_word_id))

    prompts = ["A B C"]
    if llm_kwargs.get('enable_chunked_prefill', None):
        prompts[0] = prompts[0] * 256
        llm_kwargs["max_num_tokens"] = 256

    llm_test_harness(
        llama_model_path,
        prompts, ["Z Z Z Z Z Z"],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        backend=backend,
        **llm_kwargs)


@force_ampere
def test_executor_lookahead_decoding_config():
    lookahead_config = LookaheadDecodingConfig(max_window_size=10,
                                               max_ngram_size=9,
                                               max_verification_set_size=8)
    sampling_params = SamplingParams(max_tokens=3,
                                     lookahead_config=lookahead_config)

    assert sampling_params.lookahead_config.max_window_size == 10
    assert sampling_params.lookahead_config.max_ngram_size == 9
    assert sampling_params.lookahead_config.max_verification_set_size == 8


def test_executor_results_cleanup():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    sampling_params = SamplingParams(max_tokens=6)
    for i in range(20):
        llm.generate(prompts, sampling_params=sampling_params)

    num_remaining_results = len(llm._executor._results)
    print(f"result.size: {num_remaining_results}")
    assert num_remaining_results == 0


def llm_return_logprobs_test_harness(prompt_logprobs: Optional[int],
                                     logprobs: Optional[int],
                                     return_context_logits: bool,
                                     return_generation_logits: bool,
                                     tp_size=1,
                                     streaming=False,
                                     backend=None):
    kv_cache_args_extra = {}
    if streaming:
        # need this so that context_logits / prompt_logprobs are not dropped
        # in the 2nd reuse of llm.generate() in streaming mode
        kv_cache_args_extra["enable_block_reuse"] = False

    llm = LLM(
        llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4,
                                      **kv_cache_args_extra),
        tensor_parallel_size=tp_size,
        gather_generation_logits=True,
    )

    prompts = ["A B C D E F G H I J K"]
    sampling_params = SamplingParams(
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        return_context_logits=return_context_logits,
        return_generation_logits=return_generation_logits)

    for output in llm.generate(prompts, sampling_params):
        context_logits = output.context_logits
        generation_logits = output.outputs[0].generation_logits
        logprobs_result = output.outputs[0].logprobs
        prompt_logprobs_result = output.outputs[0].prompt_logprobs
        token_ids = output.outputs[0].token_ids

        # ensure logits are dropped unless users specify return_context_logits=True
        if prompt_logprobs and not return_context_logits:
            assert context_logits is None

        if logprobs and not return_generation_logits:
            assert generation_logits is None

        if return_context_logits:
            assert isinstance(context_logits, torch.Tensor)

        if return_generation_logits:
            assert isinstance(generation_logits, torch.Tensor)

        if prompt_logprobs:
            assert prompt_logprobs_result and len(
                prompt_logprobs_result[0].keys()) == prompt_logprobs
            print("prompt_logprobs[0]: ", prompt_logprobs_result[0])

        if logprobs:
            assert logprobs_result and len(
                logprobs_result[0].keys()) in {logprobs, logprobs + 1}
            # Most contain log prob of the sample token, even if it's not within K
            assert token_ids[0] in logprobs_result[0].keys()
            for step_logprobs in logprobs_result:
                assert len(step_logprobs) == logprobs
                logprob_items = [(logprob_obj.logprob, logprob_obj.rank)
                                 for logprob_obj in step_logprobs.values()]
                sorted_by_rank = sorted(logprob_items, key=lambda x: x[1])

                for i in range(logprobs - 1):
                    current_logprob, current_rank = sorted_by_rank[i]
                    next_logprob, next_rank = sorted_by_rank[i + 1]
                    assert current_logprob >= next_logprob
                    assert current_rank == i + 1
                    assert next_rank == current_rank + 1
            print("logprobs[0]: ", logprobs_result[0])

    if streaming:

        async def task(id: int, prompt: str):
            logprobs_result_streaming = []
            async for output in llm.generate_async(prompt,
                                                   sampling_params,
                                                   streaming=True):
                logprobs_result_streaming += output.outputs[0].logprobs_diff

            # comparing streaming logprobs result to non-streaming
            assert logprobs_result_streaming == logprobs_result
            assert output.outputs[0].prompt_logprobs == prompt_logprobs_result

        async def main():
            tasks = [task(id, prompt) for id, prompt in enumerate(prompts)]
            await asyncio.gather(*tasks)

        asyncio.run(main())


def validate_stats(
    *,
    results,
    pytorch_backend,
    max_tokens,
    pp_size=1,
    use_overlap=False,
    enable_chunked_prefill=False,
    enable_iter_req_stats=False,
):
    assert results
    for iter, result in enumerate(results):
        ifbStats = result["inflightBatchingStats"]
        print(f"iter: {iter}, ifbStats: {ifbStats}")

    # Filter out the results where no requests are scheduled
    results = [
        r for r in results
        if r["inflightBatchingStats"]["numScheduledRequests"] > 0
    ]

    context_iterations = 2 if enable_chunked_prefill else 1
    generation_iterations = max_tokens - 1
    assert len(results) == context_iterations + generation_iterations

    microbatch_id = 0
    for iter, result in enumerate(results):
        ifbStats = result["inflightBatchingStats"]

        if iter < context_iterations:
            assert ifbStats["numScheduledRequests"] == 1, f"iter: {iter}"
            assert ifbStats["numContextRequests"] == 1, f"iter: {iter}"
            assert ifbStats["numGenRequests"] == 0, f"iter: {iter}"
            assert result["numActiveRequests"] == 1, f"iter: {iter}"
            assert ifbStats["microBatchId"] == microbatch_id, f"iter: {iter}"
        elif iter < (context_iterations + generation_iterations):
            assert ifbStats["numScheduledRequests"] == 1, f"iter: {iter}"
            assert ifbStats["numContextRequests"] == 0, f"iter: {iter}"
            assert ifbStats["numGenRequests"] == 1, f"iter: {iter}"
            assert result["numActiveRequests"] == 1, f"iter: {iter}"
            assert ifbStats["microBatchId"] == microbatch_id, f"iter: {iter}"

        # In pipeline parallel mode, increment microbatch_id for each context iteration except the last one,
        # since the context chunks can be scheduled in each iteration.
        if pp_size > 1 and iter < context_iterations - 1:
            microbatch_id += 1

        if enable_iter_req_stats:
            assert "requestStats" in result, f"iter: {iter}"
            req_stats = result["requestStats"]
            assert len(req_stats) == 1, f"iter: {iter}"
            req_stat = req_stats[0]
            if iter < (context_iterations - 1):
                # If use_overlap, the stats are one iteration ahead
                assert req_stat[
                    "stage"] == "GENERATION_IN_PROGRESS" if use_overlap else "CONTEXT_IN_PROGRESS", f"iter: {iter}"
                assert req_stat[
                    "contextPrefillPosition"] == 54 if use_overlap else 32, f"iter: {iter}"
                assert req_stat["numGeneratedTokens"] == 0, f"iter: {iter}"
            elif iter < (context_iterations - 1 + generation_iterations):
                assert req_stat[
                    "stage"] == "GENERATION_IN_PROGRESS", f"iter: {iter}"
                assert req_stat["contextPrefillPosition"] == 54, f"iter: {iter}"
                assert req_stat["numGeneratedTokens"] == iter - (
                    context_iterations - 1) + 1, f"iter: {iter}"
            else:
                assert req_stat[
                    "stage"] == "GENERATION_COMPLETE", f"iter: {iter}"
                assert req_stat["contextPrefillPosition"] == 54, f"iter: {iter}"
                assert req_stat[
                    "numGeneratedTokens"] == max_tokens, f"iter: {iter}"
            assert req_stat["scheduled"] == True, f"iter: {iter}"

        expected_num_completed = 1 if iter == len(results) - 1 else 0

        #TODO: For some reason, with stats_async and TRT backend, numCompleted is 0 at first iteration
        if pytorch_backend:
            assert result["numCompletedRequests"] == expected_num_completed

            # Per-iteration request-aggregate fields populated by
            # PyExecutor._update_iter_stats inside inflightBatchingStats.
            # Assert presence (a missing key indicates a serializer or
            # RPC-path regression) and sane per-iteration values (a
            # zero-under-load value indicates a mis-wired populate block).
            new_aggregate_keys = (
                "numCtxKvTokens",
                "numGenKvTokens",
                "numQueuedContextRequests",
                "numQueuedCtxTokens",
                "numQueuedGenRequests",
                "numQueuedGenKvTokens",
                "numPausedKvTokens",
            )
            for k in new_aggregate_keys:
                assert k in ifbStats, f"iter {iter}: missing ifbStats key {k}"
                assert isinstance(
                    ifbStats[k],
                    int), (f"iter {iter}: ifbStats key {k} not int "
                           f"(got {type(ifbStats[k])})")
                assert ifbStats[
                    k] >= 0, f"iter {iter}: ifbStats key {k} negative"

            if iter < context_iterations:
                # Prefill iteration: at least one scheduled context request
                # and nonzero numCtxTokens. numCtxTokens is sourced from
                # model_engine.iter_states after _forward_step for this
                # batch, so it is overlap-safe under every scheduler
                # configuration.
                assert ifbStats["numContextRequests"] >= 1, f"iter: {iter}"
                assert ifbStats["numGenRequests"] == 0, f"iter: {iter}"
                assert ifbStats["numCtxTokens"] > 0, f"iter: {iter}"
            else:
                # Generation iteration: at least one decode request with
                # nonzero total KV context length.
                assert ifbStats["numGenRequests"] >= 1, f"iter: {iter}"
                assert ifbStats["numGenKvTokens"] > 0, f"iter: {iter}"
                assert ifbStats["numContextRequests"] == 0, f"iter: {iter}"


def llm_get_stats_test_harness(tp_size: int = 1,
                               pp_size: int = 1,
                               return_context_logits: bool = False,
                               pytorch_backend: bool = False,
                               use_overlap: bool = False,
                               enable_chunked_prefill: bool = False,
                               enable_iter_req_stats: bool = False):

    if return_context_logits and pytorch_backend:
        pytest.skip("pytorch backend does not support context logits")

    if enable_iter_req_stats and not pytorch_backend:
        pytest.skip(
            "enable_iter_req_stats not supported yet without pytorch backend")

    print("-------------")
    print("return_context_logits: ", return_context_logits)
    print("pytorch_backend: ", pytorch_backend)
    print("use_overlap: ", use_overlap)
    print("enable_chunked_prefill: ", enable_chunked_prefill)
    print("enable_iter_req_stats: ", enable_iter_req_stats)
    print("-------------")

    llm_args_extra = {}
    sampling_args_extra = {}
    if return_context_logits:
        llm_args_extra["gather_generation_logits"] = True
        sampling_args_extra["return_context_logits"] = True

    if enable_chunked_prefill:
        llm_args_extra["enable_chunked_prefill"] = True
        llm_args_extra["max_num_tokens"] = 32

    if pytorch_backend:
        llm_args_extra.update(
            dict(enable_iter_perf_stats=True,
                 enable_iter_req_stats=enable_iter_req_stats,
                 disable_overlap_scheduler=not use_overlap))

    # Since we need to check pp's internal states, we disable the async broadcast
    # to get a deterministic behavior.
    env_ctx = altered_env(TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE="0") \
        if pp_size > 1 else contextlib.nullcontext()

    with env_ctx, LLM(model=llama_model_path,
                      kv_cache_config=global_kvcache_config,
                      tensor_parallel_size=tp_size,
                      pipeline_parallel_size=pp_size,
                      **llm_args_extra) as llm:

        max_tokens = 5
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         **sampling_args_extra)

        long_prompts = [
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " * 2
        ]

        for output in llm.generate(long_prompts,
                                   sampling_params=sampling_params):
            print(output)

        time.sleep(2)
        results = llm.get_stats(2)

        validate_stats(results=results,
                       pp_size=pp_size,
                       pytorch_backend=pytorch_backend,
                       max_tokens=max_tokens,
                       use_overlap=use_overlap,
                       enable_chunked_prefill=enable_chunked_prefill,
                       enable_iter_req_stats=enable_iter_req_stats)

        assert not llm.get_stats(0.5)

        # test that IterationResult()._done is properly set
        _ = llm.generate(prompts, sampling_params=sampling_params)
        assert llm.get_stats(2)


def test_llm_get_queued_stats():
    enable_iter_req_stats = True
    use_overlap = False
    tp_size = 1

    num_requests = 10
    repeated_prompts = ["A B C D E F G H I J K L M"] * num_requests

    llm_args_extra = {}
    sampling_args_extra = {}

    llm_args_extra.update(
        dict(enable_iter_perf_stats=True,
             enable_iter_req_stats=enable_iter_req_stats,
             disable_overlap_scheduler=not use_overlap))

    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              tensor_parallel_size=tp_size,
              max_batch_size=1,
              **llm_args_extra)

    max_tokens = 10
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     **sampling_args_extra)

    max_tries = 10
    has_queue_requests = False

    while not has_queue_requests and max_tries > 0:
        max_tries -= 1
        # Generate outputs, which will queue requests
        for output in llm.generate(repeated_prompts,
                                   sampling_params=sampling_params):
            print(output)

        results = llm.get_stats(2)

        for index, result in enumerate(results):
            if "requestStats" in result:
                for requestStat in result["requestStats"]:
                    if requestStat["stage"] == "QUEUED":
                        has_queue_requests = True
                        assert requestStat["numGeneratedTokens"] == 0

        if not has_queue_requests:
            print("No queued requests found, retrying...")
            asyncio.sleep(1)
        else:
            print("Found queued requests, breaking out of the loop.")

    assert has_queue_requests


def llm_get_stats_async_test_harness(tp_size: int = 1,
                                     pp_size: int = 1,
                                     return_context_logits: bool = False,
                                     pytorch_backend: bool = False,
                                     use_overlap: bool = False,
                                     enable_chunked_prefill: bool = False,
                                     enable_iter_req_stats: bool = False):

    if return_context_logits and pytorch_backend:
        pytest.skip("pytorch backend does not support context logits")

    if enable_iter_req_stats and not pytorch_backend:
        pytest.skip(
            "enable_iter_req_stats not supported yet without pytorch backend")

    print("-------------")
    print("return_context_logits: ", return_context_logits)
    print("pytorch_backend: ", pytorch_backend)
    print("use_overlap: ", use_overlap)
    print("enable_chunked_prefill: ", enable_chunked_prefill)
    print("enable_iter_req_stats: ", enable_iter_req_stats)
    print("-------------")

    llm_args_extra = {}
    sampling_args_extra = {}
    if return_context_logits:
        sampling_args_extra["return_context_logits"] = True

    if enable_chunked_prefill:
        llm_args_extra["enable_chunked_prefill"] = True
        llm_args_extra["max_num_tokens"] = 32

    if pytorch_backend:
        llm_args_extra.update(
            dict(enable_iter_perf_stats=True,
                 enable_iter_req_stats=enable_iter_req_stats,
                 disable_overlap_scheduler=not use_overlap))

    with LLM(model=llama_model_path,
             kv_cache_config=global_kvcache_config,
             tensor_parallel_size=tp_size,
             pipeline_parallel_size=pp_size,
             **llm_args_extra) as llm:

        max_tokens = 6
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         **sampling_args_extra)

        long_prompts = [
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " * 2
        ]

        async def task0():
            async for output in llm.generate_async(
                    long_prompts[0],
                    streaming=True,
                    sampling_params=sampling_params):
                print(output)

        async def task1(repetition_index: int):
            results = []
            await asyncio.sleep(
                4)  # ensure there's stats to collect for the assertion
            async for stats in llm.get_stats_async(
                    10):  # it will return immediately
                results.append(stats)

            assert results
            if not use_overlap:
                validate_stats(
                    results=results,
                    pp_size=pp_size,
                    pytorch_backend=pytorch_backend,
                    max_tokens=max_tokens,
                    use_overlap=use_overlap,
                    # After the first repetition, context will be reused and there will be no chunking.
                    enable_chunked_prefill=enable_chunked_prefill
                    if repetition_index == 0 else False,
                    enable_iter_req_stats=enable_iter_req_stats)

        async def main():
            for repetition_index in range(2):  # test recurrent usage
                await asyncio.gather(task0(), task1(repetition_index))

        asyncio.run(main())


def _test_llm_capture_request_error(pytorch_backend: bool, tp_size: int = 1):
    llm = LLM(
        model=llama_model_path,
        max_num_tokens=64,
        tensor_parallel_size=tp_size,
    )

    prompt = 'A ' * 65  # the minimum max_num_tokens is 64
    # Both backends now consistently raise RequestError for max_num_tokens validation
    with pytest.raises(RequestError):
        llm.generate(prompt)


def test_llm_shutdown_executor():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    )

    llm.generate("A")
    llm.shutdown()

    with pytest.raises(RuntimeError):
        llm.generate("A")


def test_llm_api_jupyter_scenario():

    with LLM(
            model=llama_model_path,
            kv_cache_config=global_kvcache_config,
    ) as llm:

        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        async def task():
            return llm.generate(["A", "B", "C", "D"], sampling_params)

        output = asyncio.run(task())
        for token in output:
            print(token)


def test_llm_dynamic_batch_config():
    scheduler_config = SchedulerConfig(dynamic_batch_config=DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=True,
        dynamic_batch_moving_average_window=128))
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=9),
                     scheduler_config=scheduler_config)


def run_llm_with_postprocess_parallel(tp_size: int = 1):
    sampling_params = SamplingParams(max_tokens=6)

    postproc_settings = dict(num_postprocess_workers=2,
                             postprocess_tokenizer_dir=llama_model_path)

    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=sampling_params,
                     kv_cache_config=global_kvcache_config,
                     tensor_parallel_size=tp_size,
                     **postproc_settings)


def test_llm_with_postprocess_parallel():
    run_llm_with_postprocess_parallel(tp_size=1)


def _stream_payloads_from_chunks(chunks):
    payloads = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        for line in chunk.splitlines():
            if line.startswith("data: "):
                data = line[len("data: "):].strip()
                if data != "[DONE]":
                    payloads.append(json.loads(data))
    return payloads


def test_chat_stream_post_processor_reuses_stream_metadata() -> None:
    result = GenerationResultBase(123, SamplingParams())
    output = result._outputs[0]
    output.text = "x"
    output.token_ids = [1]

    args = ChatPostprocArgs(role="assistant", model="test-model")
    chunks = chat_stream_post_processor(result, args)

    output._last_text_len = len(output.text)
    output._last_token_ids_len = len(output.token_ids)
    output.text = "xy"
    output.token_ids.append(2)
    chunks += chat_stream_post_processor(result, args)

    payloads = _stream_payloads_from_chunks(chunks)

    assert {payload["id"] for payload in payloads} == {"chatcmpl-123"}
    assert len({payload["created"] for payload in payloads}) == 1
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[-1]["choices"][0]["delta"]["content"] == "y"


class _FakeCompletionGeneratorArgs:
    backend = "pytorch"
    gather_generation_logits = False
    num_postprocess_workers = 0
    return_perf_metrics = False


class _FakeModelConfig:
    vocab_size = 32000


class _FakeCompletionStreamResult(GenerationResultBase):

    @property
    def finished(self):
        return self._done

    @property
    def request_id(self):
        return self.id


class _FakeCompletionPromise:

    def __init__(self, result, prompt_token_ids):
        self._result = result
        self._yielded = False
        self.prompt_token_ids = prompt_token_ids
        self.aborted = False

    @property
    def finished(self):
        return True

    @property
    def request_id(self):
        return self._result.id

    def abort(self):
        self.aborted = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return self._result


class _FakeCompletionGenerator:

    def __init__(self):
        self.args = _FakeCompletionGeneratorArgs()
        self.postproc_args = []

    def input_processor(self, prompt, _sampling_params):
        token_id = ord(prompt["prompt"])
        return [token_id], {}

    def generate_async(self, inputs, sampling_params, _postproc_params,
                       streaming, **_kwargs):
        assert streaming
        result_id = 100 + len(self.postproc_args)
        result = _FakeCompletionStreamResult(result_id, sampling_params)
        result._done = True

        output = result._outputs[0]
        output.text = f"text-{result_id}"
        output.token_ids = [result_id]
        output.finish_reason = "stop"

        self.postproc_args.append(_postproc_params.postproc_args)
        return _FakeCompletionPromise(result, inputs["prompt_token_ids"])


class _FakeRawRequestState:
    pass


class _FakeRawRequest:

    def __init__(self):
        self.headers = {}
        self.state = _FakeRawRequestState()
        self.client = "test-client"

    async def is_disconnected(self):
        return True


def test_startup_metrics_are_cached_and_reported_by_server_info() -> None:

    class FakeExecutor:

        def __init__(self):
            self.calls = 0

        def get_startup_metrics(self):
            self.calls += 1
            return {"model_loader": {"total_model_loading_seconds": 1.5}}

    executor = FakeExecutor()
    generator = object.__new__(BaseLLM)
    generator._executor = executor
    generator._startup_metrics = None
    generator._disaggregated_params = {}

    server = object.__new__(OpenAIServer)
    server.generator = generator
    first_response = asyncio.run(server.get_server_info())
    second_response = asyncio.run(server.get_server_info())

    expected = {
        "disaggregated_params": {},
        "startup_metrics": {
            "model_loader": {
                "total_model_loading_seconds": 1.5
            }
        },
    }
    assert json.loads(first_response.body) == expected
    assert json.loads(second_response.body) == expected
    assert executor.calls == 1


def test_openai_completion_list_prompt_stream_reuses_stream_metadata() -> None:

    async def run_request():
        generator = _FakeCompletionGenerator()
        server = object.__new__(OpenAIServer)
        server.generator = generator
        server.model = "test-model"
        server.model_config = _FakeModelConfig()
        server.tokenizer = None
        server.metrics_collector = None
        server.perf_metrics = None

        request = CompletionRequest(model="test-model",
                                    prompt=["A", "B"],
                                    stream=True)
        response = await server.openai_completion(request, _FakeRawRequest())
        chunks = [chunk async for chunk in response.body_iterator]
        return generator, _stream_payloads_from_chunks(chunks)

    generator, payloads = asyncio.run(run_request())

    ids = {payload["id"] for payload in payloads}
    created = {payload["created"] for payload in payloads}
    choice_indexes = {
        payload["choices"][0]["index"]
        for payload in payloads if payload["choices"]
    }

    assert len(payloads) == 2
    assert len(ids) == 1
    assert ids.isdisjoint({"cmpl-100", "cmpl-101"})
    assert len(created) == 1
    assert choice_indexes == {0, 1}
    assert {args.stream_response_id for args in generator.postproc_args} == ids
    assert {args.stream_created for args in generator.postproc_args} == created


def run_llm_with_postprocess_parallel_and_result_handler(
        streaming, backend, tp_size: int = 1):
    # avoid import error when running in CI
    from tensorrt_llm.executor.postproc_worker import PostprocParams
    from tensorrt_llm.serve.postprocess_handlers import (
        ChatPostprocArgs, chat_stream_post_processor)

    from .run_llm_with_postproc import get_concatenated_content

    sampling_params = SamplingParams(max_tokens=6)
    tokenizer = load_hf_tokenizer(llama_model_path)
    post_proc_args = ChatPostprocArgs(tokenizer=tokenizer,
                                      role="assistant",
                                      model=llama_model_path)
    post_proc_params = PostprocParams(post_processor=chat_stream_post_processor,
                                      postproc_args=post_proc_args)

    llm = LLM(model=llama_model_path,
              backend=backend,
              kv_cache_config=global_kvcache_config,
              tensor_parallel_size=tp_size,
              num_postprocess_workers=2,
              postprocess_tokenizer_dir=llama_model_path)
    golden_result = "D E F G H I"
    outputs = []
    for output in llm.generate_async(prompts[0],
                                     sampling_params=sampling_params,
                                     _postproc_params=post_proc_params,
                                     streaming=streaming):
        outputs.append(output.outputs[0]._postprocess_result)
    actual_result = get_concatenated_content(outputs)
    assert actual_result == golden_result, \
        f"Expected: {golden_result}, Actual: {actual_result}"


def run_llm_abort_request(llm: LLM, sampling_params: SamplingParams):
    # to make sure LLM run slower for canceling the request to be actually performed
    sampling_params.max_tokens = 100
    sampling_params.end_id = -1  # let it run for a while

    async def task():
        result = llm.generate_async(prompts[0],
                                    sampling_params=sampling_params,
                                    streaming=True)
        print(f"to abort")
        result.abort()

        print(f"waiting for the result")
        # Before it actually abort, we should see some outputs
        outputs = []
        async for output in result:
            print(f"get output: {output}")
            outputs.append(output)
        print(f"get {len(outputs)} remaining outputs")
        print(f"outputs: {outputs}")
        print(f"finish_reason: {outputs[-1].outputs[0].finish_reason}")
        assert 1 <= len(
            outputs) < 1000  # It should be aborted before the completion
        # NOTE: known issue: only the last output is finished and got the finish_reason
        assert outputs[-1].outputs[-1].finish_reason == "cancelled"

    asyncio.run(task())


sampling_params_for_aborting_request = [
    SamplingParams(),
    # n-returns
    SamplingParams(n=2, top_k=2),
    SamplingParams(n=2, top_k=2, best_of=3),
    SamplingParams(n=3, use_beam_search=True),
    SamplingParams(n=2, best_of=3, use_beam_search=True),
]
