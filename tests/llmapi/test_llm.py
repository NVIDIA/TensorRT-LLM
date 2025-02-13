import asyncio
import datetime
import gc
import json
import os
import random
import shutil
import sys
import tempfile
from typing import List, Optional, Union

import datasets
import pytest
import torch
import transformers
from pydantic import BaseModel
from utils.util import skip_single_gpu

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.executor import (ExecutorBindingsWorker, LoRARequest,
                                   PromptAdapterRequest, RequestError)
from tensorrt_llm.llmapi import (LLM, BuildCacheConfig, EagleDecodingConfig,
                                 GuidedDecodingParams, KvCacheConfig,
                                 KvCacheRetentionConfig,
                                 LookaheadDecodingConfig, MedusaDecodingConfig,
                                 NoStatsAvailable, SamplingParams)
from tensorrt_llm.llmapi._perf_evaluator import perform_faked_oai_postprocess
from tensorrt_llm.llmapi.llm_utils import (BuildConfig, LlmArgs, QuantAlgo,
                                           QuantConfig, _ParallelConfig)
from tensorrt_llm.llmapi.tokenizer import TokenizerBase, TransformersTokenizer
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.models.automodel import AutoConfig, AutoModelForCausalLM
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode

# isort: off
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar, skip_gpu_memory_less_than_40gb, skip_pre_hopper
# isort: on

# The unittests are based on the tiny-llama, which is fast to build and run.
# There are other tests based on llama-7B model, such as the end-to-end tests in test_e2e.py, and parallel tests in
# test_llm_multi_gpu.py.


def get_model_path(model_name):
    engine_dir = os.environ.get('LLM_ENGINE_DIR', None)
    if engine_dir:
        return engine_dir
    return str(llm_models_root() / model_name)


def get_reference_count(obj):
    '''
    Get the reference count.
    '''
    return sys.getrefcount(obj) - 1


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

    llm = LLM(model_dir, tokenizer=tokenizer, **llm_kwargs)
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(outputs)
    for out, ref in zip(outputs, references):
        if isinstance(ref, list):
            # N output
            assert len(out.outputs) == len(ref)
            for o, r in zip(out.outputs, ref):
                assert similar(o.text, r, threshold=similar_threshold)
        else:
            assert similar(out.outputs[0].text,
                           ref,
                           threshold=similar_threshold)

    assert gc.is_tracked(llm)
    assert len(
        gc.get_referrers(llm)) == 0, f"the references: {gc.get_referrers(llm)}"


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
    assert len(outputs) == len(references)

    for i, (output, target_output) in enumerate(zip(outputs, references)):
        if isinstance(target_output, list):
            # N output
            assert len(output.outputs) == len(target_output)
            for j, (out, ref) in enumerate(zip(output.outputs, target_output)):
                assert similar(out.text, ref, threshold=similar_threshold)
                if finish_reasons is not None:
                    assert out.finish_reason == finish_reasons[i][j]
                if stop_reasons is not None:
                    assert out.stop_reason == stop_reasons[i][j]
        else:
            out = output.outputs[0]
            assert similar(out.text, target_output, threshold=similar_threshold)
            if finish_reasons is not None:
                assert out.finish_reason == finish_reasons[i]
            if stop_reasons is not None:
                assert out.stop_reason == stop_reasons[i]


default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
mixtral_model_name = "Mixtral-8x7B-v0.1"

llama_model_path = get_model_path(default_model_name)
llm_engine_dir = os.environ.get('LLM_ENGINE_DIR', './tmp.engine')

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


@force_ampere
def test_llm_build_config():
    build_config = BuildConfig()
    # change some building parameters
    build_config.max_batch_size = 129
    build_config.max_beam_width = 4
    build_config.max_num_tokens = 888
    build_config.strongly_typed = True
    build_config.max_seq_len = 333

    llm = LLM(model=llama_model_path,
              build_config=build_config,
              kv_cache_config=global_kvcache_config)
    tmpdir = tempfile.TemporaryDirectory()
    llm.save(tmpdir.name)

    with open(os.path.join(tmpdir.name, "config.json"), "r") as f:
        # read the build_config and check if the parameters are correctly saved
        engine_config = json.load(f)

        build_config1 = BuildConfig.from_dict(engine_config["build_config"])

        # Know issue: this will be converted to None after save engine for single-gpu
        build_config1.plugin_config.nccl_plugin = 'float16'
        assert build_config1.max_batch_size == build_config.max_batch_size
        assert build_config1.max_beam_width == build_config.max_beam_width
        assert build_config1.max_num_tokens == build_config.max_num_tokens
        assert build_config1.strongly_typed == build_config.strongly_typed
        assert build_config1.max_seq_len == build_config.max_seq_len


def test_llm_args_invalid_usage():
    runtime_max_batch_size = 3
    runtime_max_num_tokens = 2

    # Update build_config with warning msg if runtime arguments are passed.
    llm_args = LlmArgs.from_kwargs(model='test-model',
                                   max_batch_size=runtime_max_batch_size,
                                   max_num_tokens=runtime_max_num_tokens)
    assert llm_args.build_config.max_batch_size == runtime_max_batch_size
    assert llm_args.build_config.max_num_tokens == runtime_max_num_tokens

    # Conflict between build_config and runtime_params
    build_config = BuildConfig(max_batch_size=5, max_num_tokens=7)
    llm_args = LlmArgs.from_kwargs(model='test-model',
                                   build_config=build_config,
                                   max_batch_size=runtime_max_batch_size,
                                   max_num_tokens=runtime_max_num_tokens)
    assert llm_args.build_config.max_batch_size == build_config.max_batch_size
    assert llm_args.build_config.max_num_tokens == build_config.max_num_tokens


def test_llm_loading_from_hf():
    sampling_params = SamplingParams(max_tokens=8)
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=sampling_params,
                     kv_cache_config=global_kvcache_config)


@force_ampere
def test_llm_loading_from_ckpt():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    assert tokenizer is not None

    ckpt_dir = tempfile.TemporaryDirectory()
    llama = AutoModelForCausalLM.from_hugging_face(llama_model_path)
    llama.save_checkpoint(ckpt_dir.name)
    del llama

    llm_test_harness(ckpt_dir.name,
                     prompts, ["D E F G H I J K"],
                     tokenizer=tokenizer,
                     kv_cache_config=global_kvcache_config,
                     sampling_params=SamplingParams(max_tokens=8))


@pytest.mark.parametrize('model_format', ['hf', 'ckpt'])
def test_llm_with_dummy_weights(model_format):
    # dummy_dir contains config.json and tokenizer files only
    # the test fails if load_format != 'dummy'
    dummy_dir = tempfile.TemporaryDirectory()
    if model_format == 'hf':
        hf_config = transformers.AutoConfig.from_pretrained(llama_model_path)
        hf_config.save_pretrained(dummy_dir.name)
    else:
        config = AutoConfig.from_hugging_face(llama_model_path, dtype='float16')
        config.to_json_file(os.path.join(dummy_dir.name, 'config.json'))
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    tokenizer.save_pretrained(dummy_dir.name)

    sampling_params = SamplingParams(max_tokens=8)
    llm_test_harness(dummy_dir.name,
                     prompts,
                     ["A placeholder reference for dummy-weight engine."],
                     sampling_params=sampling_params,
                     similar_threshold=0.0,
                     load_format='dummy',
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
        return self.tokenizer.batch_encode_plus(texts, **kwargs)


def test_llm_with_customized_tokenizer():
    llm = LLM(
        model=llama_model_path,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(llama_model_path),
        kv_cache_config=global_kvcache_config)

    for output in llm.generate(prompts):
        print(output)


def test_llm_without_tokenizer():
    llm = LLM(model=llama_model_path,
              skip_tokenizer_init=True,
              kv_cache_config=global_kvcache_config)

    sampling_params = SamplingParams(end_id=2, pad_id=2, max_tokens=8)

    prompts = [[23, 14, 3]]

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert not output.outputs[0].text, \
            "The output should be empty since the tokenizer is missing"
        print(output)


def test_llm_with_kv_cache_retention_config():
    kv_cache_retention_config = KvCacheRetentionConfig([
        KvCacheRetentionConfig.TokenRangeRetentionConfig(
            0, 2, 30, datetime.timedelta(seconds=30))
    ], 80)

    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)

    for output in llm.generate(
            prompts, kv_cache_retention_config=kv_cache_retention_config):
        print(output)


@pytest.mark.parametrize(
    'tokenizer_dir, threshold',
    [
        (get_model_path('gpt2'), 0.95),  # BPE
        (get_model_path('bert/bert-base-uncased'), 0.95),  # WordPiece
        (get_model_path('t5-small'), 0.95),  # SentencePiece
        (get_model_path('opt-125m'), 0.95),
        (get_model_path('starcoder2-3b'), 0.95),
        (get_model_path('gpt-j-6b'), 0.95),
        (get_model_path('bloom-560m'), 0.95),
        (get_model_path('mpt-7b'), 0.95),
        (get_model_path('falcon-7b-instruct'), 0.95),
        (get_model_path('llama-models-v2/llama-v2-7b-hf'), 0.95),
        (get_model_path('codellama/CodeLlama-7b-Instruct-hf'), 0.95),
        (llama_model_path, 0.95),
        (get_model_path(mixtral_model_name), 0.95)
    ])
def test_tokenizer_decode_incrementally(tokenizer_dir: str, threshold: float):
    random.seed(42)

    num_samples = 100
    cnn_dailymail = datasets.load_dataset(cnn_dailymail_path,
                                          name='3.0.0',
                                          split='train')
    alpaca_chinese = datasets.load_dataset(alpaca_chinese_path, split='train')
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

        if tokenizer_dir.endswith(
                'bert-base-uncased') and tokenizer.clean_up_tokenization_spaces:
            decoded_text = tokenizer.clean_up_tokenization(decoded_text)
        reference = tokenizer.decode(token_ids)
        if decoded_text == reference:
            num_perfect += 1
        else:
            # For non-perfect matching cases, decoded_text should also be very similar to the reference
            assert similar(decoded_text, reference, 0.99)
    print(f"Perfect matching ratio: {num_perfect / num_samples * 100}%")
    assert num_perfect / num_samples >= threshold


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

    tp_size = tp_size if not use_auto_parallel else 1
    world_size = tp_size if use_auto_parallel else None

    llm = LLM(
        model=get_model_path(model_name),
        tokenizer=tokenizer,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
        auto_parallel=use_auto_parallel,
        auto_parallel_world_size=world_size,
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


@pytest.fixture(scope="module")
def llm_for_sampling_params() -> LLM:
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    )
    return llm


def test_user_specify_workspace():
    user_specified_ws_path = '/tmp/specified_workspace'
    shutil.rmtree(user_specified_ws_path, ignore_errors=True)
    os.mkdir(user_specified_ws_path)
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              workspace=user_specified_ws_path)
    pre_built_engine_cfg = llm.args.model / 'config.json'
    assert pre_built_engine_cfg.exists()
    del llm
    gc.collect()
    assert not pre_built_engine_cfg.exists()


@force_ampere
def test_generate_with_sampling_params_per_prompt(llm_for_sampling_params: LLM):
    llm = llm_for_sampling_params
    sampling_params_list = [
        SamplingParams(end_id=-1, pad_id=-1) for _ in range(2)
    ]
    sampling_params_list[0].max_tokens = 4
    sampling_params_list[1].max_tokens = 8

    for i, output in enumerate(
            llm.generate(prompts, sampling_params=sampling_params_list)):
        output_len = len(output.outputs[0].token_ids)
        print(f"output_len: {output_len}")
        assert output_len <= sampling_params_list[i].max_tokens


@force_ampere
@pytest.fixture(scope="module")
@pytest.mark.parametrize(
    "sampling_params",
    [
        # temperature
        SamplingParams(
            max_tokens=6, temperature=0.5, beam_search_diversity_rate=0.5),
        # topK
        SamplingParams(max_tokens=6, top_k=10, top_p=0.92),
        # topP
        SamplingParams(max_tokens=6, top_p=0.92),
        # penalty
        SamplingParams(max_tokens=8,
                       length_penalty=1.0,
                       presence_penalty=0.0,
                       repetition_penalty=1.0,
                       min_tokens=5),
        # early stopping
        SamplingParams(max_tokens=6, early_stopping=5),
        # n-returns
        SamplingParams(max_tokens=6, n=2, top_k=2),
        SamplingParams(max_tokens=6, n=2, top_k=2, best_of=3),
        SamplingParams(max_tokens=6, n=3, use_beam_search=True),
        SamplingParams(max_tokens=6, n=2, best_of=3, use_beam_search=True),
    ])
def test_generate_with_SamplingConfig(llm_for_sampling_params: LLM,
                                      sampling_params: SamplingParams):
    llm = llm_for_sampling_params

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert len(output.outputs) == sampling_params.n


@force_ampere
def test_generate_with_beam_search():
    build_config = BuildConfig()
    build_config.max_beam_width = 2
    build_config.max_num_tokens = 20

    llm_test_harness(
        llama_model_path,
        prompts, [["D E F G H I", "D E F G I J"]],
        build_config=build_config,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        sampling_params=SamplingParams(max_tokens=6, beam_width=2))


@force_ampere
def test_generate_with_streaming_llm():
    # TODO[chunweiy]: Test with larger size when the underlying support is ready
    build_config = BuildConfig()
    build_config.plugin_config.streamingllm = True
    build_config.max_batch_size = 8
    build_config.max_seq_len = 512
    kv_cache_config = KvCacheConfig(max_attention_window=[64],
                                    sink_token_length=4)

    # Check the plugin config is correctly set
    assert build_config.plugin_config.streamingllm is True

    sampling_params = SamplingParams(max_tokens=4)

    llm_test_harness(llama_model_path,
                     prompts, ["D E F G"],
                     sampling_params=sampling_params,
                     build_config=build_config,
                     kv_cache_config=kv_cache_config)


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
    build_config.max_batch_size = 128  # reduce buffer sizes, specially for generation logits
    build_config.gather_context_logits = gather_context_logits

    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        build_config=build_config,
        gather_generation_logits=gather_generation_logits,
    )
    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        return_log_probs=return_log_probs)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(prompts[0].split()) + \
                1 == output.context_logits.shape[0]
        if gather_generation_logits:
            assert output.outputs[0].generation_logits is not None
            assert sampling_params.max_tokens == output.outputs[
                0].generation_logits.shape[0]
        if return_log_probs:
            assert output.outputs[0].logprobs is not None

        print(output)


@force_ampere
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


@force_ampere
def test_generate_with_sampling_params_misc():
    llm = LLM(
        model=llama_model_path,
        tokenizer_mode='slow',
        kv_cache_config=global_kvcache_config,
    )

    fake_end_id = llm.tokenizer.encode("N", add_special_tokens=False)[-1]

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M"],
                     sampling_params=SamplingParams(max_tokens=15,
                                                    end_id=fake_end_id))

    llm_check_output(llm,
                     prompts, ["D E F G H I K L M N O P Q R S"],
                     sampling_params=SamplingParams(max_tokens=15,
                                                    end_id=fake_end_id,
                                                    ignore_eos=True))

    llm_check_output(llm,
                     prompts, [""],
                     sampling_params=SamplingParams(max_tokens=15,
                                                    end_id=fake_end_id,
                                                    detokenize=False))

    outputs = llm.generate(prompts)
    assert outputs[0].prompt_token_ids == [1, 319, 350, 315]

    outputs = llm.generate(prompts, SamplingParams(add_special_tokens=False))
    assert outputs[0].prompt_token_ids == [319, 350, 315]

    outputs = llm.generate(prompts, SamplingParams(truncate_prompt_tokens=2))
    assert outputs[0].prompt_token_ids == [1, 315]

    # Use embedding bias to force the output tokens to be special tokens
    unk_id = llm.tokenizer.encode('<unk>', add_special_tokens=False)[-1]
    vocab_size_padded = 32000
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[unk_id] = torch.finfo(torch.float32).max

    outputs = llm.generate(
        prompts, SamplingParams(max_tokens=5, embedding_bias=embedding_bias))
    assert outputs[0].outputs[0].text == ""

    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=5,
                       embedding_bias=embedding_bias,
                       skip_special_tokens=False,
                       spaces_between_special_tokens=False))
    assert outputs[0].outputs[0].text == "<unk><unk><unk><unk><unk>"

    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=5,
                       embedding_bias=embedding_bias,
                       skip_special_tokens=False,
                       spaces_between_special_tokens=True))
    assert outputs[0].outputs[0].text == "<unk> <unk> <unk> <unk> <unk>"


@force_ampere
def test_generate_with_embedding_bias():
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[biased_word_id] = torch.finfo(torch.float32).max

    sampling_params = SamplingParams(max_tokens=6,
                                     embedding_bias=embedding_bias)

    llm_test_harness(
        llama_model_path,
        prompts, ["Z Z Z Z Z Z"],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))


@force_ampere
def test_invalid_embedding_bias():
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000

    # Should raise "Embedding bias data type must be same as model logits type"
    embedding_bias = torch.zeros(vocab_size_padded, dtype=torch.float16)
    embedding_bias[biased_word_id] = torch.finfo(torch.float16).max

    llm = LLM(llama_model_path)
    sampling_params = SamplingParams(max_tokens=6,
                                     embedding_bias=embedding_bias)

    try:
        llm.generate(["A B C"], sampling_params=sampling_params)
    except RequestError:
        return

    assert (0)


@skip_pre_hopper
def test_generate_with_embedding_bias_fp8():
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    assert quant_config.quant_mode.has_any_quant()

    llm = LLM(llama_model_path, quant_config=quant_config)

    # FP32 embedding bias input (will be converted to FP16)
    embedding_bias = torch.zeros(vocab_size_padded)
    embedding_bias[biased_word_id] = torch.finfo(torch.float32).max
    sampling_params = SamplingParams(max_tokens=6,
                                     embedding_bias=embedding_bias)

    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "Z Z Z Z Z Z"

    # FP16 embedding bias input
    embedding_bias = torch.zeros(vocab_size_padded, dtype=torch.float16)
    embedding_bias[biased_word_id] = torch.finfo(torch.float16).max

    sampling_params = SamplingParams(max_tokens=6,
                                     embedding_bias=embedding_bias)

    for output in llm.generate(["A B C"], sampling_params=sampling_params):
        print(output)
        assert output.outputs[0].text == "Z Z Z Z Z Z"


@skip_pre_hopper
def test_invalid_embedding_bias_fp8():
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    assert quant_config.quant_mode.has_any_quant()

    llm = LLM(llama_model_path, quant_config=quant_config)

    # Should raise "Embedding bias tensor needs to be in CPU memory for casting"
    embedding_bias = torch.zeros(vocab_size_padded, device='cuda')
    embedding_bias[biased_word_id] = torch.finfo(torch.float32).max
    sampling_params = SamplingParams(max_tokens=6,
                                     embedding_bias=embedding_bias)

    try:
        llm.generate(["A B C"], sampling_params=sampling_params)
    except RequestError:
        return

    assert (0)


class MyLogitsPostProcessor:

    def __init__(self, biased_word_id):
        self.biased_word_id = biased_word_id

    def __call__(self, req_id: int, logits: torch.Tensor, ids: List[List[int]],
                 stream_ptr: int, client_id: Optional[int]):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits[:] = float("-inf")
            logits[..., self.biased_word_id] = 0


def tinyllama_logits_processor_test_harness(**llm_kwargs):
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    sampling_params = SamplingParams(max_tokens=6,
                                     logits_post_processor_name="my_logits_pp")

    llm_test_harness(
        llama_model_path,
        prompts, ["Z Z Z Z Z Z"],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        logits_post_processor_map={
            "my_logits_pp": MyLogitsPostProcessor(biased_word_id)
        },
        **llm_kwargs)


@force_ampere
def test_tinyllama_logits_processor():
    tinyllama_logits_processor_test_harness()


class MyBatchedLogitsPostProcessor:

    def __init__(self, biased_word_id):
        self.biased_word_id = biased_word_id

    def __call__(self, req_ids_batch: List[int],
                 logits_batch: List[torch.Tensor],
                 token_ids_batch: List[List[List[int]]], stream_ptr: int,
                 client_ids_batch: List[Optional[int]]):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            for logits in logits_batch:
                logits[:] = float("-inf")
                logits[..., self.biased_word_id] = 0


def tinyllama_logits_processor_batched_test_harness(**llm_kwargs):
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    sampling_params = SamplingParams(
        max_tokens=6,
        logits_post_processor_name=SamplingParams.BATCHED_POST_PROCESSOR_NAME)

    llm_test_harness(
        llama_model_path,
        prompts, ["Z Z Z Z Z Z"],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        logits_post_processor_map={
            SamplingParams.BATCHED_POST_PROCESSOR_NAME:
            MyBatchedLogitsPostProcessor(biased_word_id)
        },
        **llm_kwargs)


@force_ampere
def test_tinyllama_logits_processor_batched():
    tinyllama_logits_processor_batched_test_harness()


def tinyllama_guided_decoding_test_harness(**llm_kwargs):
    prompts = [
        "What is 1+1? Answer formatted in a dict in json format: ",
        "What is the year after 2024? Answer: ",
    ]

    class Answer(BaseModel):
        answer: int

    json_schema = json.dumps(Answer.model_json_schema())
    regex = r"\d+"
    ebnf_grammar = "root ::= [0-9]+"

    sampling_params = [
        SamplingParams(max_tokens=10),
        SamplingParams(max_tokens=10,
                       guided_decoding=GuidedDecodingParams(json_object=True)),
        SamplingParams(max_tokens=10,
                       guided_decoding=GuidedDecodingParams(json=json_schema)),
        SamplingParams(max_tokens=10,
                       guided_decoding=GuidedDecodingParams(regex=regex)),
        SamplingParams(
            max_tokens=10,
            guided_decoding=GuidedDecodingParams(grammar=ebnf_grammar)),
    ]

    num_prompts, num_sampling_params = len(prompts), len(sampling_params)
    prompts = [p for p in prompts for _ in range(num_sampling_params)]
    sampling_params = [sp for _ in range(num_prompts) for sp in sampling_params]
    references = [
        '\n\n```\n{\n    "1":',
        '{"1": "1", "2": "',
        '{"answer": 1}',
        '1',
        '1',
        '2025\n\nQuestion 3:',
        '[2025]',
        '{"answer": 202',
        '2025',
        '2025',
    ]
    llm_test_harness(llama_model_path,
                     prompts,
                     references,
                     sampling_params=sampling_params,
                     guided_decoding_backend='xgrammar',
                     similar_threshold=0.7,
                     **llm_kwargs)


@force_ampere
def test_tinyllama_guided_decoding():
    tinyllama_guided_decoding_test_harness()


def test_llm_api_medusa():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    build_config = BuildConfig(
        max_batch_size=1,
        max_seq_len=1024,
        max_draft_len=63,
        speculative_decoding_mode=SpeculativeDecodingMode.MEDUSA)

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = MedusaDecodingConfig(num_medusa_heads=4,
                            medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                             [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
      )
    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              speculative_model=get_model_path("medusa-vicuna-7b-v1.3"),
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@skip_single_gpu
def test_llm_api_medusa_tp2():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    build_config = BuildConfig(
        max_batch_size=1,
        max_seq_len=1024,
        max_draft_len=63,
        speculative_decoding_mode=SpeculativeDecodingMode.MEDUSA)

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = MedusaDecodingConfig(num_medusa_heads=4,
                            medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                             [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
      )
    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              speculative_model=get_model_path("medusa-vicuna-7b-v1.3"),
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              tensor_parallel_size=2)

    outputs = llm.generate(prompts, sampling_params, tensor_parallel_size=2)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def test_llm_api_eagle():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    build_config = BuildConfig(
        max_batch_size=1,
        max_seq_len=1024,
        max_draft_len=63,
        speculative_decoding_mode=SpeculativeDecodingMode.EAGLE)

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = EagleDecodingConfig(
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
                            eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    )
    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              speculative_model=get_model_path("EAGLE-Vicuna-7B-v1.3"),
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@skip_single_gpu
def test_llm_api_eagle_tp2():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    build_config = BuildConfig(
        max_batch_size=1,
        max_seq_len=1024,
        max_draft_len=63,
        speculative_decoding_mode=SpeculativeDecodingMode.EAGLE)

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = EagleDecodingConfig(
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
                            eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    )
    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              speculative_model=get_model_path("EAGLE-Vicuna-7B-v1.3"),
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              tensor_parallel_size=2)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def tinyllama_lookahead_decoding_test_harness(**llm_kwargs):
    prompts = [
        "A B C",
    ]
    lookahead_config = LookaheadDecodingConfig(max_window_size=3,
                                               max_ngram_size=3,
                                               max_verification_set_size=3)

    build_config = BuildConfig(max_batch_size=8,
                               max_num_tokens=128,
                               max_input_len=32,
                               max_seq_len=64)

    sampling_params = [
        SamplingParams(max_tokens=8, lookahead_config=lookahead_config),
    ]

    num_prompts, num_sampling_params = len(prompts), len(sampling_params)
    prompts = [p for p in prompts for _ in range(num_sampling_params)]
    sampling_params = [sp for _ in range(num_prompts) for sp in sampling_params]
    references = [
        'D E F G H I J K',
    ]
    llm_test_harness(llama_model_path,
                     prompts,
                     references,
                     sampling_params=sampling_params,
                     speculative_config=lookahead_config,
                     build_config=build_config,
                     kv_cache_config=global_kvcache_config,
                     **llm_kwargs)


@force_ampere
def test_tinyllama_lookahead_decoding():
    tinyllama_lookahead_decoding_test_harness()


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


def llama_v2_13b_lora_test_harness(**llm_kwargs):
    hf_model_dir = get_model_path("llama-models-v2/llama-v2-13b-hf")
    hf_lora_dir = get_model_path("llama-models-v2/chinese-llama-2-lora-13b")

    # For LoRA checkpoints with finetuned embedding and lm_head, lora_dir must be provided at build time.
    build_config = BuildConfig(lora_config=LoraConfig(lora_dir=[hf_lora_dir]))
    llm = LLM(hf_model_dir,
              tokenizer=hf_lora_dir,
              enable_lora=True,
              max_lora_rank=64,
              build_config=build_config,
              **llm_kwargs)

    prompts = [
        "今天天气很好，我到公园的时候，",
        "今天天气很好，我到公园的时候，",
    ]
    references = [
        "看见好多人们都看书，看书书看书书，看书书看书书书书书书",
        "发现公园里到处都是人，有的在跑步，有的在打羽毛球，还有的",
    ]
    lora_req = LoRARequest("Chinese", 1, hf_lora_dir)
    sampling_params = SamplingParams(max_tokens=20, add_special_tokens=False)
    outputs = llm.generate(prompts,
                           sampling_params,
                           lora_request=[None, lora_req])
    for output, ref in zip(outputs, references):
        assert similar(output.outputs[0].text, ref)


def llama_7b_multi_lora_test_harness(**llm_kwargs):
    hf_model_dir = get_model_path("llama-models/llama-7b-hf")
    hf_lora_dir1 = get_model_path("llama-models/luotuo-lora-7b-0.1")
    hf_lora_dir2 = get_model_path("llama-models/Japanese-Alpaca-LoRA-7b-v0")

    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    build_config = BuildConfig(lora_config=LoraConfig(
        lora_target_modules=['attn_q', 'attn_k', 'attn_v']))
    llm = LLM(hf_model_dir,
              enable_lora=True,
              max_lora_rank=8,
              build_config=build_config,
              **llm_kwargs)

    prompts = [
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "美国的首都在哪里? \n答案:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
        "アメリカ合衆国の首都はどこですか? \n答え:",
    ]
    references = [
        "沃尔玛\n\n## 新闻\n\n* ",
        "美国的首都是华盛顿。\n\n美国的",
        "纽约\n\n### カンファレンスの",
        "Washington, D.C.\nWashington, D.C. is the capital of the United",
        "华盛顿。\n\n英国の首都是什",
        "ワシントン\nQ1. アメリカ合衆国",
    ]
    lora_req1 = LoRARequest("luotuo", 1, hf_lora_dir1)
    lora_req2 = LoRARequest("Japanese", 2, hf_lora_dir2)
    sampling_params = SamplingParams(max_tokens=20)
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=[None, lora_req1, lora_req2, None, lora_req1, lora_req2])
    for output, ref in zip(outputs, references):
        assert similar(output.outputs[0].text, ref)


@skip_gpu_memory_less_than_40gb
def test_llama_v2_13b_lora():
    llama_v2_13b_lora_test_harness()


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora():
    llama_7b_multi_lora_test_harness(max_loras=1, max_cpu_loras=8)


def llama_v2_7b_prompt_adapter_test_harness(**llm_kwargs):
    hf_model_dir = get_model_path("llama-models-v2/llama-v2-7b-hf")
    hf_prompt_adapter_dir = get_model_path("llama-models-v2/llama_tweet_ptune")
    llm = LLM(hf_model_dir,
              enable_prompt_adapter=True,
              max_prompt_adapter_token=8,
              **llm_kwargs)

    prompts = [
        "Born in north-east France, Soyer trained as a",
        "Born in north-east France, Soyer trained as a",
        "Tweet text: I have complaints! Label: ",
        "Tweet text: I have complaints! Label: ",
        "Tweet text: I have no problems Label: ",
        "Tweet text: I have no problems Label: ",
    ]
    references = [
        "painter at the École des Beaux-Arts in Paris. He was a member of the",
        "chef and has worked in the restaurant industry for 15 years.Ћ\nBorn in north",
        "1999.\nTweet text: I have complaints! Label: 19",
        "no complaint",
        "100%\nI have no problems Label: 100%\nI have no",
        "no complaint",
    ]
    pa_req = PromptAdapterRequest('tweet', 1, hf_prompt_adapter_dir)
    sampling_params = SamplingParams(max_tokens=20)
    outputs = llm.generate(
        prompts,
        sampling_params,
        prompt_adapter_request=[None, pa_req, None, pa_req, None, pa_req])
    for output, ref in zip(outputs, references):
        assert similar(output.outputs[0].text, ref)


@skip_gpu_memory_less_than_40gb
def test_llama_v2_7b_prompt_adapter():
    llama_v2_7b_prompt_adapter_test_harness(
        kv_cache_config=global_kvcache_config_no_reuse)


@force_ampere
def test_generate_block_reuse():
    build_config = BuildConfig()
    build_config.plugin_config._use_paged_context_fmha = True
    build_config.plugin_config._paged_kv_cache = True
    llm = LLM(model=llama_model_path,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4,
                                            enable_block_reuse=True),
              build_config=build_config)

    sampling_params = SamplingParams(max_tokens=6)

    prompts = ["A B C", "A B C D"]
    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)


def test_executor_results_cleanup():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    sampling_params = SamplingParams(max_tokens=6)
    for i in range(20):
        llm.generate(prompts, sampling_params=sampling_params)

    num_remaining_results = len(llm._executor._results)
    print(f"result.size: {num_remaining_results}")
    assert num_remaining_results == 0


@pytest.mark.parametrize("trust_remote_code", [True, False])
def _test_llm_trust_remote_code(trust_remote_code: bool):
    # OOM when tested with other cases
    # TODO[chunweiy]: Enable this later

    if trust_remote_code:
        internlm_model_path = get_model_path("internlm-chat-7b")
        llm = LLM(model=internlm_model_path,
                  trust_remote_code=trust_remote_code,
                  tokenizer=TransformersTokenizer.from_pretrained(
                      internlm_model_path, trust_remote_code=trust_remote_code),
                  kv_cache_config=global_kvcache_config)
        sampling_params = SamplingParams(max_tokens=6,
                                         temperature=0.8,
                                         top_p=0.95)
        prompts = [
            "The future of AI is",
        ]

        for output in llm.generate(prompts, sampling_params=sampling_params):
            print(output)
    else:
        with pytest.raises(ValueError):
            llm = LLM(model="internlm/internlm-chat-7b",
                      trust_remote_code=trust_remote_code,
                      tokenizer="internlm/internlm-chat-7b",
                      kv_cache_config=global_kvcache_config)


def test_llm_build_cache():
    # Activate the build-cache
    cache_config = BuildCacheConfig(max_records=1, max_cache_storage_gb=10)
    sampling_params = SamplingParams(max_tokens=6)

    def first_run():
        llm = LLM(model=llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  enable_build_cache=cache_config)
        llm_check_output(llm,
                         prompts, ["D E F G H I J K"],
                         sampling_params=sampling_params)

    def second_run():
        llm = LLM(model=llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  enable_build_cache=cache_config)
        llm_check_output(llm,
                         prompts, ["D E F G H I J K"],
                         sampling_params=sampling_params)

        # the cache should be hit
        assert llm.llm_build_stats.cache_hitted, llm.llm_build_stats.cache_info

    first_run()
    second_run()


class DummyError(Exception):
    pass


class DummyExecutorMeta(type):

    def __new__(cls, name, bases, dic, worker_cls):
        new_cls = super().__new__(cls, name, bases, dic)

        @classmethod
        def create(cls, engine, executor_config, *args, **kwargs):
            return worker_cls(engine=engine, executor_config=executor_config)

        new_cls.create = create
        return new_cls


def test_llm_apidocs():
    doc = LLM.__doc__
    assert doc
    assert doc.find('pipeline_parallel_size') != -1
    assert doc.find('tensor_parallel_size') != -1
    assert doc.find('auto_parallel') != -1


def check_llm_return_context_logits(tp_size=1):
    build_config = BuildConfig(gather_context_logits=True)

    llm = LLM(llama_model_path,
              tensor_parallel_size=tp_size,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
              build_config=build_config)

    sampling_params = SamplingParams(max_tokens=8, return_context_logits=True)

    prompts = ["A B C D E F G H I J K"] * 8

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert isinstance(output.context_logits, torch.Tensor)
        print(output)

    # Check the WAR for returning logits performance
    if tp_size == 1:
        assert isinstance(llm._executor, ExecutorBindingsWorker)


def check_llm_return_generation_logits(tp_size=1):

    llm = LLM(
        llama_model_path,
        tensor_parallel_size=tp_size,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        gather_generation_logits=True,
    )

    sampling_params = SamplingParams(max_tokens=8,
                                     return_generation_logits=True)

    prompts = ["A B C D E F G H I J K"] * 8

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert isinstance(output.outputs[0].generation_logits, torch.Tensor)
        print(output)

    # Check the WAR for returning logits performance
    if tp_size == 1:
        assert isinstance(llm._executor, ExecutorBindingsWorker)


def test_llm_return_context_logits():
    check_llm_return_context_logits(tp_size=1)


def test_llm_return_generation_logits():
    check_llm_return_generation_logits(tp_size=1)


class DummyExecutorWorker3(ExecutorBindingsWorker):
    should_raise_error = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.counter = 0
        self.failed_requests = set()

    def _engine_response_callback(self, response: tllm.Response):
        if response.client_id in self.failed_requests:
            return response
        # Making the first response failed, and the subsequent responses successful
        if DummyExecutorWorker3.should_raise_error:
            DummyExecutorWorker3.should_raise_error = False
            print(f"Raise error for {response.client_id}")
            self.failed_requests.add(response.client_id)
            return tllm.Response(
                request_id=0,  # dummy value
                client_id=response.client_id,
                error_msg="Test error")
        else:
            return response


DummyExecutor3 = DummyExecutorMeta("DummyExecutor3", (), {},
                                   worker_cls=DummyExecutorWorker3)


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5063025")
def test_llm_handling_per_requeust_error():
    llm = LLM(model=llama_model_path,
              executor_cls=DummyExecutor3,
              kv_cache_config=global_kvcache_config)
    # The dummy executor will delay the responses
    sampling_params = SamplingParams(max_tokens=6)

    def batch_task():
        DummyExecutorWorker3.should_raise_error = True
        with pytest.raises(RequestError):
            for output in llm.generate(prompts,
                                       sampling_params=sampling_params):
                print(output)

        for output in llm.generate(prompts, sampling_params=sampling_params):
            print(output)

    batch_task()


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5063025")
def test_llm_handling_per_requeust_error_async():
    llm = LLM(model=llama_model_path,
              executor_cls=DummyExecutor3,
              kv_cache_config=global_kvcache_config)
    # The dummy executor will delay the responses
    sampling_params = SamplingParams(max_tokens=6)

    # test in streaming mode
    async def task():
        # 10 requests, each request will get error, while the whole LLM instance is still alive
        with pytest.raises(RequestError):
            DummyExecutorWorker3.should_raise_error = True
            async for output in llm.generate_async(
                    prompts[0], streaming=True,
                    sampling_params=sampling_params):
                print(output)

        DummyExecutorWorker3.should_raise_error = False
        async for output in llm.generate_async(prompts[0],
                                               streaming=True,
                                               sampling_params=sampling_params):
            print(output)

    asyncio.run(task())


def llm_get_stats_test_harness(tp_size: int = 1):
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              tensor_parallel_size=tp_size,
              fast_build=True)
    sampling_params = SamplingParams(max_tokens=100)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)

    while True:
        try:
            stats = llm._get_stats(2)
            print(stats)
        except NoStatsAvailable:
            break


# The LLM._get_stats/_async is temporary APIs, and we don't plan to have a public one in the short run.
# TODO Introduce some dedicated DS similar to executor.GenerationResult, that should be more stable
@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5000903")
def test_llm_get_stats():
    llm_get_stats_test_harness(tp_size=1)


def llm_get_stats_async_test_harness(tp_size: int = 1):
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              tensor_parallel_size=tp_size,
              fast_build=True)
    sampling_params = SamplingParams(max_tokens=6)

    async def task0():
        async for output in llm.generate_async(prompts[0],
                                               streaming=True,
                                               sampling_params=sampling_params):
            print(output)

    async def task1():
        while True:
            try:
                stats = await llm._get_stats_async(2)
                print(stats)
            except NoStatsAvailable:
                break

    async def main():
        await asyncio.gather(task0(), task1())

    asyncio.run(main())


@pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5000903")
def test_llm_get_stats_async():
    llm_get_stats_async_test_harness(tp_size=1)


def test_llm_chunked_prefill():
    sampling_params = SamplingParams(max_tokens=8)
    build_config = BuildConfig()
    build_config.plugin_config.use_paged_context_fmha = True
    build_config.max_num_tokens = 64
    new_tokens = 8
    build_config.max_seq_len = build_config.max_num_tokens + new_tokens

    def fail_path():
        sampling_params = SamplingParams(max_tokens=8)
        llm = LLM(model=llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  build_config=build_config,
                  enable_chunked_prefill=False)

        with pytest.raises(ValueError):
            output = llm.generate_async(
                "A " * build_config.max_num_tokens,
                sampling_params=sampling_params,
            ).result()

    def success_path():
        llm = LLM(model=llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  build_config=build_config,
                  enable_chunked_prefill=True)

        output = llm.generate_async(
            "A " * build_config.max_num_tokens,
            sampling_params=sampling_params,
        ).result()

    fail_path()
    success_path()


def _test_llm_capture_request_error(tp_size: int = 1):
    build_config = BuildConfig()
    build_config.max_num_tokens = 64

    llm = LLM(
        model=llama_model_path,
        build_config=build_config,
    )

    prompt = 'A ' * 65  # the minimum max_num_tokens is 64

    with pytest.raises(RequestError):
        llm.generate(prompt)


def test_llm_capture_request_error():
    _test_llm_capture_request_error(tp_size=1)


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
    scheduler_config = tllm.SchedulerConfig(
        dynamic_batch_config=tllm.DynamicBatchConfig(
            enable_batch_size_tuning=True,
            enable_max_num_tokens_tuning=True,
            dynamic_batch_moving_average_window=128))
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=SamplingParams(max_tokens=9),
                     scheduler_config=scheduler_config)


def run_llm_with_postprocess_parallel(tp_size: int = 1):
    sampling_params = SamplingParams(max_tokens=6)

    postproc_settings = dict(_num_postprocess_workers=2,
                             _postprocess_tokenizer_dir=llama_model_path)

    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=sampling_params,
                     kv_cache_config=global_kvcache_config,
                     tensor_parallel_size=tp_size,
                     **postproc_settings)


def test_llm_with_postprocess_parallel():
    run_llm_with_postprocess_parallel(tp_size=1)


def run_llm_with_postprocess_parallel_and_result_handler(tp_size: int = 1):
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              tensor_parallel_size=tp_size,
              _num_postprocess_workers=2,
              _postprocess_tokenizer_dir=llama_model_path,
              _postprocess_result_handler=perform_faked_oai_postprocess)
    sampling_params = SamplingParams(max_tokens=6)

    golden_result = "DEFGHI"
    for i, output in enumerate(
            llm.generate_async(prompts[0],
                               sampling_params=sampling_params,
                               streaming=True)):

        if i < len(golden_result) - 1:
            assert golden_result[i] in output.outputs[0]._postprocess_result[-1]
        else:
            assert golden_result[i] in output.outputs[0]._postprocess_result[
                -2]  # EOS


def test_llm_with_postprocess_parallel_and_result_handler():
    run_llm_with_postprocess_parallel_and_result_handler(tp_size=1)


if __name__ == '__main__':
    test_llm_capture_request_error()
