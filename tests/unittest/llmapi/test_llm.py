import asyncio
import datetime
import gc
import json
import os
import sys
import time

# Required for test_generate_with_seed to pass.
# See the discussion in https://github.com/NVIDIA/TensorRT-LLM/pull/4264#issuecomment-2943269891
# The following line must be ahead of any tensorrt_llm imports,
# since currently env util functions like getEnvForceDeterministic are implemented using static variables,
# which means they are only initialized once the CPP translation unit is loaded (should be refactored to be non static later).
os.environ['TRTLLM_FORCE_XQA'] = '1'
# Note that we cannot use os.environ['FORCE_DETERMINISTIC'] = '1' here,
# since it will disable KV cache reuse and make test_llm_api_draft_target fail.

import random
import shutil
import sys
import tempfile
from typing import List, Optional, Union

import datasets
import pytest
import torch
import transformers

from tensorrt_llm import LLM as LLM_torch
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor import (GenerationExecutorWorker, GenerationRequest,
                                   GenerationResult, LoRARequest,
                                   PromptAdapterRequest, RequestError)
from tensorrt_llm.llmapi import (BuildCacheConfig, CacheTransceiverConfig,
                                 EagleDecodingConfig, KvCacheConfig,
                                 KvCacheRetentionConfig,
                                 LookaheadDecodingConfig, MedusaDecodingConfig,
                                 RequestOutput)
from tensorrt_llm.llmapi import TrtLlmArgs as LlmArgs
from tensorrt_llm.llmapi.llm_args import (DynamicBatchConfig, PeftCacheConfig,
                                          SchedulerConfig)
from tensorrt_llm.llmapi.llm_utils import (BuildConfig, QuantAlgo, QuantConfig,
                                           _ParallelConfig)
from tensorrt_llm.llmapi.tokenizer import (TokenizerBase, TransformersTokenizer,
                                           load_hf_tokenizer)
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.models.automodel import AutoConfig, AutoModelForCausalLM
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
from tensorrt_llm.sampling_params import (BatchedLogitsProcessor,
                                          LogitsProcessor, SamplingParams)

# isort: off
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from gc_utils import assert_resource_freed
from llmapi.lora_test_utils import (
    check_llama_7b_multi_lora_from_request_test_harness,
    check_llama_7b_multi_unique_lora_adapters_from_request)
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar, skip_gpu_memory_less_than_40gb, skip_pre_hopper, skip_single_gpu
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


def get_reference_count(obj):
    '''
    Get the reference count.
    '''
    return sys.getrefcount(obj) - 1


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
    backend = llm_kwargs.get('backend', None)
    world_size = tp_size * pp_size
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"world_size ({world_size}) is greater than available GPUs ({torch.cuda.device_count()})"
        )

    tokenizer = llm_kwargs.pop('tokenizer', None)
    if tokenizer is None:
        tokenizer = model_dir

    llm_cls = LLM_torch if backend == "pytorch" else LLM

    with assert_resource_freed(llm_cls, model_dir, tokenizer,
                               **llm_kwargs) as llm:
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


@pytest.mark.part0
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
              kv_cache_config=global_kvcache_config,
              fast_build=True)
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


@pytest.mark.part0
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


@pytest.mark.part0
def test_llm_loading_from_hf():
    sampling_params = SamplingParams(max_tokens=8)
    llm_test_harness(llama_model_path,
                     prompts, ["D E F G H I J K"],
                     sampling_params=sampling_params,
                     kv_cache_config=global_kvcache_config)


@force_ampere
@pytest.mark.part0
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


@pytest.mark.parametrize('model_format', [
    'hf',
    'ckpt',
])
@pytest.mark.part0
def test_llm_with_dummy_weights(model_format):
    # dummy_dir contains config.json and tokenizer files only
    # the test fails if load_format != 'dummy'
    dummy_dir = tempfile.TemporaryDirectory()
    if model_format == 'hf':
        hf_config = transformers.AutoConfig.from_pretrained(llama_model_path)
        hf_config.save_pretrained(dummy_dir.name)
    else:
        config = AutoConfig.from_hugging_face(llama_model_path,
                                              dtype='float16',
                                              trust_remote_code=True)
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


@pytest.mark.part0
def test_llm_with_customized_tokenizer():
    llm = LLM(
        model=llama_model_path,
        # a customized tokenizer is passed to override the default one
        tokenizer=MyTokenizer.from_pretrained(llama_model_path),
        kv_cache_config=global_kvcache_config,
        fast_build=True,
    )

    for output in llm.generate(prompts):
        print(output)


@pytest.mark.part0
def test_llm_without_tokenizer():
    llm = LLM(
        model=llama_model_path,
        skip_tokenizer_init=True,
        kv_cache_config=global_kvcache_config,
        fast_build=True,
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

    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              fast_build=True)

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
        fast_build=True,
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
    with LLM_torch(model_path,
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


@pytest.fixture(scope="module")
def llm_for_sampling_params():
    build_config = BuildConfig(max_beam_width=3)
    llm = LLM(
        model=llama_model_path,
        build_config=build_config,
        kv_cache_config=global_kvcache_config,
        fast_build=True,
    )
    yield llm
    llm.shutdown()


@pytest.mark.skip(reason="https://nvbugs/5504095")
@pytest.mark.part0
def test_user_specify_workspace():
    user_specified_ws_path = '/tmp/specified_workspace'
    shutil.rmtree(user_specified_ws_path, ignore_errors=True)
    os.mkdir(user_specified_ws_path)
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              workspace=user_specified_ws_path,
              fast_build=True)
    pre_built_engine_cfg = llm.args.model / 'config.json'
    assert pre_built_engine_cfg.exists()
    del llm
    gc.collect()
    assert not pre_built_engine_cfg.exists()


@force_ampere
@pytest.mark.part0
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
@pytest.mark.part0
def test_generate_with_SamplingConfig(llm_for_sampling_params: LLM,
                                      sampling_params: SamplingParams):
    llm = llm_for_sampling_params

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)
        assert len(output.outputs) == sampling_params.n


@force_ampere
@pytest.mark.part0
def test_generate_with_seed(llm_for_sampling_params: LLM):
    prompts = ["The capital of France is"] * 10
    # Use a high temperature and large max_tokens to increase the diversity
    sampling_params = [
        SamplingParams(temperature=100, top_k=100, max_tokens=100)
        for _ in range(10)
    ]
    # Fix the seed for the second 5 prompts
    for i in range(5, 10):
        sampling_params[i].seed = 515

    llm = llm_for_sampling_params
    generated_texts = []
    for output in llm.generate(prompts, sampling_params):
        generated_texts.append(output.outputs[0].text)
    for output in llm.generate(prompts, sampling_params):
        generated_texts.append(output.outputs[0].text)

    assert len(generated_texts) == 20
    assert len(set(generated_texts)) == 11


@force_ampere
@pytest.mark.part0
def test_generate_with_beam_search(llm_for_sampling_params: LLM):
    llm = llm_for_sampling_params
    references = [["D E F G H I", "D E F G I J"]]
    sampling_params = SamplingParams(max_tokens=6, n=2, use_beam_search=True)

    # Non-streaming mode
    outputs = llm.generate(prompts, sampling_params)
    print(outputs)
    check_output(outputs, references)

    # Streaming mode
    outputs = [
        llm.generate_async(prompt, sampling_params, streaming=True)
        for prompt in prompts
    ]
    outputs = [output.result() for output in outputs]
    print(outputs)
    check_output(outputs, references)


@pytest.mark.skip(reason="https://nvbugs/5435714")
@force_ampere
@pytest.mark.part0
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


@pytest.mark.part0
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
@pytest.mark.part0
def test_generate_with_OutputConfig(
    gather_context_logits: bool,
    gather_generation_logits: bool,
):
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
        fast_build=True,
    )
    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(prompts[0].split()) + \
                1 == output.context_logits.shape[0]
        if gather_generation_logits:
            assert output.outputs[0].generation_logits is not None
            assert sampling_params.max_tokens == output.outputs[
                0].generation_logits.shape[0]

        print(output)


@force_ampere
@pytest.mark.part0
def test_generate_with_stop_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        fast_build=True,
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
        fast_build=True,
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
        fast_build=True,
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
        fast_build=True,
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


@pytest.mark.skip(reason="https://nvbugs/5370718")
@force_ampere
@pytest.mark.part0
def test_generate_with_sampling_params_misc():
    llm = LLM(
        model=llama_model_path,
        tokenizer_mode='slow',
        kv_cache_config=global_kvcache_config,
        fast_build=True,
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
@pytest.mark.part0
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
@pytest.mark.part0
def test_invalid_embedding_bias():
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000

    # Should raise "Embedding bias data type must be same as model logits type"
    embedding_bias = torch.zeros(vocab_size_padded, dtype=torch.float16)
    embedding_bias[biased_word_id] = torch.finfo(torch.float16).max

    llm = LLM(llama_model_path, fast_build=True)
    sampling_params = SamplingParams(max_tokens=6,
                                     embedding_bias=embedding_bias)

    try:
        llm.generate(["A B C"], sampling_params=sampling_params)
    except RequestError:
        return

    assert (0)


@skip_pre_hopper
@pytest.mark.part0
def test_generate_with_embedding_bias_fp8():
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    assert quant_config.quant_mode.has_any_quant()

    llm = LLM(llama_model_path, quant_config=quant_config, fast_build=True)

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
@pytest.mark.part0
def test_invalid_embedding_bias_fp8():
    tokenizer = transformers.AutoTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    vocab_size_padded = 32000

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)
    assert quant_config.quant_mode.has_any_quant()

    llm = LLM(llama_model_path, quant_config=quant_config, fast_build=True)

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
@pytest.mark.part0
def test_tinyllama_logits_processor():
    tinyllama_logits_processor_test_harness()


class MyBatchedLogitsProcessor(BatchedLogitsProcessor):

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
    sampling_params = SamplingParams(max_tokens=6,
                                     apply_batched_logits_processor=True)

    llm_test_harness(
        llama_model_path,
        prompts, ["Z Z Z Z Z Z"],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        batched_logits_processor=MyBatchedLogitsProcessor(biased_word_id),
        **llm_kwargs)


@force_ampere
@pytest.mark.part0
def test_tinyllama_logits_processor_batched():
    tinyllama_logits_processor_batched_test_harness()


@pytest.mark.part0
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
    )

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = MedusaDecodingConfig(num_medusa_heads=4,
            max_draft_len=63,
            speculative_model_dir=get_model_path("medusa-vicuna-7b-v1.3"),
            medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                             [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
      )
    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              fast_build=True)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@skip_single_gpu
@pytest.mark.part0
def test_llm_api_medusa_tp2():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    build_config = BuildConfig(max_batch_size=1, max_seq_len=1024)

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = MedusaDecodingConfig(num_medusa_heads=4,
            max_draft_len=63,
              speculative_model_dir=get_model_path("medusa-vicuna-7b-v1.3"),
                            medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                             [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
      )
    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              build_config=build_config,
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              tensor_parallel_size=2,
              fast_build=True)

    outputs = llm.generate(prompts, sampling_params, tensor_parallel_size=2)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.part0
def test_llm_api_eagle(**llm_kwargs):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = EagleDecodingConfig(
        max_draft_len=63,
        speculative_model_dir=get_model_path("EAGLE-Vicuna-7B-v1.3"),
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
                            eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], \
                                            [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], \
                                            [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], \
                                            [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], \
                                            [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], \
                                            [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    )

    # in test_llm_multi_gpu, kv_cache_config is passed as a kwarg
    if "kv_cache_config" in llm_kwargs:
        kv_cache_config = llm_kwargs["kv_cache_config"]
        del llm_kwargs["kv_cache_config"]

    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              max_batch_size=1,
              max_seq_len=1024,
              fast_build=True,
              **llm_kwargs)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


@pytest.mark.part0
def test_llm_api_eagle2(**llm_kwargs):
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    kv_cache_config = KvCacheConfig(enable_block_reuse=True)

    speculative_config = EagleDecodingConfig(
        max_draft_len=63,
        speculative_model_dir=get_model_path("EAGLE-Vicuna-7B-v1.3"),
        num_eagle_layers=4,
        max_non_leaves_per_layer=10,
        use_dynamic_tree=True,
        dynamic_tree_max_topK=10)

    # in test_llm_multi_gpu, kv_cache_config is passed as a kwarg
    if "kv_cache_config" in llm_kwargs:
        kv_cache_config = llm_kwargs["kv_cache_config"]
        del llm_kwargs["kv_cache_config"]

    llm = LLM(model=get_model_path("vicuna-7b-v1.3"),
              kv_cache_config=kv_cache_config,
              speculative_config=speculative_config,
              max_batch_size=1,
              max_seq_len=1024,
              fast_build=True,
              **llm_kwargs)

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


def llama_v2_13b_lora_from_dir_test_harness(**llm_kwargs):
    # Shahar- perhaps disable build config
    hf_model_dir = get_model_path("llama-models-v2/llama-v2-13b-hf")
    hf_lora_dir = get_model_path("llama-models-v2/chinese-llama-2-lora-13b")

    # For LoRA checkpoints with finetuned embedding and lm_head, lora_dir must be provided at build time.
    build_config = BuildConfig(lora_config=LoraConfig(
        lora_dir=[hf_lora_dir], max_lora_rank=64, max_loras=2, max_cpu_loras=2))
    llm = LLM(hf_model_dir,
              tokenizer=hf_lora_dir,
              enable_lora=True,
              build_config=build_config,
              fast_build=True,
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


def _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call: list[int], max_loras: int,
        max_cpu_loras: int, repeat_calls: int, repeats_per_call: int):
    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    build_config = BuildConfig(lora_config=LoraConfig(
        lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
        max_lora_rank=8,
        max_loras=max_loras,
        max_cpu_loras=max_cpu_loras))
    check_llama_7b_multi_unique_lora_adapters_from_request(
        lora_adapter_count_per_call,
        repeat_calls,
        repeats_per_call,
        LLM,
        enable_lora=True,
        build_config=build_config,
        fast_build=True)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_evict_and_reload_lora_gpu_cache():
    """Test eviction and re-loading a previously evicted adapter from the LoRA GPU cache, within a single
    llm.generate call, that's repeated twice.
    """  # noqa: D205
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[2],
        max_loras=1,
        max_cpu_loras=2,
        repeat_calls=2,
        repeats_per_call=3)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_evict_and_load_new_adapters_in_cpu_and_gpu_cache():
    """Test eviction and loading of new adapters in the evicted space, over several llm.generate calls, with LoRA GPU
    cache size < LoRA CPU cache size.
    """  # noqa: D205
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[2, 2, 2],
        max_loras=1,
        max_cpu_loras=3,
        repeat_calls=1,
        repeats_per_call=1)


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora_read_from_cache_after_insert():
    """Test that loading and then using the same adapters loaded in cache works."""
    _check_llama_7b_multi_lora_evict_load_new_adapters(
        lora_adapter_count_per_call=[3],
        max_loras=3,
        max_cpu_loras=3,
        repeat_calls=2,
        repeats_per_call=1)


def test_llama_7b_peft_cache_config_affects_peft_cache_size():
    """Tests that LLM arg of peft_cache_config affects the peft cache sizes.

    NOTE: The caller can't get the actual LoRA cache sizes, so we instead we
    test that it fails when configured with a value too small to contain a
    single adapter.
    """
    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config_no_cache_size_values = LoraConfig(
        lora_target_modules=['attn_q', 'attn_k', 'attn_v'], max_lora_rank=8)
    build_config = BuildConfig(lora_config=lora_config_no_cache_size_values)

    # Test that too small PeftCacheConfig.host_cache_size causes failure
    with pytest.raises(RuntimeError):
        check_llama_7b_multi_lora_from_request_test_harness(
            LLM,
            enable_lora=True,
            build_config=build_config,
            fast_build=True,
            lora_config=lora_config_no_cache_size_values,
            peft_cache_config=PeftCacheConfig(
                host_cache_size=1))  # size in bytes

    # Test that too small PeftCacheConfig.device_cache_percent causes failure
    with pytest.raises(RuntimeError):
        check_llama_7b_multi_lora_from_request_test_harness(
            LLM,
            enable_lora=True,
            build_config=build_config,
            fast_build=True,
            lora_config=lora_config_no_cache_size_values,
            peft_cache_config=PeftCacheConfig(device_cache_percent=0.0000001))


def test_llama_7b_lora_config_overrides_peft_cache_config():
    """Tests that cache size args in lora_config LLM arg override the cache size
    parameters in peft_cache_config LLM arg.
    """    # noqa: D205
    build_config = BuildConfig(lora_config=LoraConfig(
        lora_target_modules=['attn_q', 'attn_k', 'attn_v'], max_lora_rank=8))
    check_llama_7b_multi_lora_from_request_test_harness(
        LLM,
        enable_lora=True,
        build_config=build_config,
        fast_build=True,
        lora_config=LoraConfig(
            lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
            max_lora_rank=8,
            max_loras=2,
            max_cpu_loras=2),
        peft_cache_config=PeftCacheConfig(
            host_cache_size=1,  # size in bytes
            device_cache_percent=0.0000001))


@skip_gpu_memory_less_than_40gb
def test_llama_v2_13b_lora():
    llama_v2_13b_lora_from_dir_test_harness()


def llama_v2_7b_prompt_adapter_test_harness(**llm_kwargs):
    hf_model_dir = get_model_path("llama-models-v2/llama-v2-7b-hf")
    hf_prompt_adapter_dir = get_model_path("llama-models-v2/llama_tweet_ptune")
    llm = LLM(hf_model_dir,
              enable_prompt_adapter=True,
              max_prompt_adapter_token=8,
              fast_build=True,
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
        [
            "painter at the École des Beaux-Arts in Paris. He was a member of the"
        ],
        [
            "chef and has worked in the restaurant industry for 15 years.Ћ\nBorn in north"
        ],
        ["1999.\nTweet text: I have complaints! Label: 19"],
        ["no complaint"],
        [
            "100%\nI have no problems Label: 100%\nI have no",
            "1999\nLabel: 1999 (1999)\nT"
        ],
        ["no complaint"],
    ]
    pa_req = PromptAdapterRequest('tweet', 1, hf_prompt_adapter_dir)
    sampling_params = SamplingParams(max_tokens=20)
    outputs = llm.generate(
        prompts,
        sampling_params,
        prompt_adapter_request=[None, pa_req, None, pa_req, None, pa_req])
    for output, ref in zip(outputs, references):
        # Currently, the 5th request may have non-deterministic outputs.
        # Let the test pass if the generation output matches any of the candidate references.
        assert any(similar(output.outputs[0].text, r) for r in ref)


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
              build_config=build_config,
              fast_build=True)

    sampling_params = SamplingParams(max_tokens=6)

    prompts = ["A B C", "A B C D"]
    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)


def test_executor_results_cleanup():
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              fast_build=True)
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
                  kv_cache_config=global_kvcache_config,
                  fast_build=True)
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
                      kv_cache_config=global_kvcache_config,
                      fast_build=True)


def test_llm_build_cache():
    # Activate the build-cache
    cache_config = BuildCacheConfig(max_records=1, max_cache_storage_gb=10)
    sampling_params = SamplingParams(max_tokens=6)

    def first_run():
        llm = LLM(model=llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  enable_build_cache=cache_config,
                  fast_build=True)
        llm_check_output(llm,
                         prompts, ["D E F G H I J K"],
                         sampling_params=sampling_params)

    def second_run():
        llm = LLM(model=llama_model_path,
                  kv_cache_config=global_kvcache_config,
                  enable_build_cache=cache_config,
                  fast_build=True)
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


def check_llm_return_context_logits(tp_size=1):
    build_config = BuildConfig(gather_context_logits=True)

    llm = LLM(
        llama_model_path,
        tensor_parallel_size=tp_size,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        build_config=build_config,
        fast_build=True,
    )

    sampling_params = SamplingParams(max_tokens=8, return_context_logits=True)

    prompts = ["A B C D E F G H I J K"] * 8

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert isinstance(output.context_logits, torch.Tensor)
        print(output)

    # Check the WAR for returning logits performance
    if tp_size == 1:
        assert isinstance(llm._executor, GenerationExecutorWorker)


def check_llm_return_generation_logits(tp_size=1):

    llm = LLM(
        llama_model_path,
        tensor_parallel_size=tp_size,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        gather_generation_logits=True,
        fast_build=True,
    )

    sampling_params = SamplingParams(max_tokens=8,
                                     return_generation_logits=True)

    prompts = ["A B C D E F G H I J K"] * 8

    for output in llm.generate(prompts, sampling_params=sampling_params):
        assert isinstance(output.outputs[0].generation_logits, torch.Tensor)
        print(output)

    # Check the WAR for returning logits performance
    if tp_size == 1:
        assert isinstance(llm._executor, GenerationExecutorWorker)


def test_llm_return_context_logits():
    check_llm_return_context_logits(tp_size=1)


def test_llm_return_generation_logits():
    check_llm_return_generation_logits(tp_size=1)


def llm_return_logprobs_test_harness(prompt_logprobs: Optional[int],
                                     logprobs: Optional[int],
                                     return_context_logits: bool,
                                     return_generation_logits: bool,
                                     tp_size=1,
                                     streaming=False,
                                     backend=None):
    LLM_CLASS = LLM
    llm_args_extra = {}
    kv_cache_args_extra = {}
    if backend in ["pytorch", "autodeploy"]:
        LLM_CLASS = LLM_torch
        if streaming:
            # need this so that context_logits / prompt_logprobs are not dropped
            # in the 2nd reuse of llm.generate() in streaming mode
            kv_cache_args_extra["enable_block_reuse"] = False
    else:
        llm_args_extra["fast_build"] = True

    llm = LLM_CLASS(
        llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4,
                                      **kv_cache_args_extra),
        build_config=BuildConfig(gather_context_logits=True),
        tensor_parallel_size=tp_size,
        gather_generation_logits=True,
        **llm_args_extra,
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


@force_ampere
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits, backend",
    [
        # TRT backend test cases
        (2, None, True, False, "trt"),  # prompt_logprobs with context_logits
        (None, 2, False, False, "trt"),  # generation logprobs only (top-2)
        (2, None, False, False,
         "trt"),  # prompt_logprobs without context_logits
        (None, None, False, False, "trt"),  # no logprobs at all
    ])
def test_llm_return_logprobs(prompt_logprobs: Optional[int],
                             logprobs: Optional[int],
                             return_context_logits: bool,
                             return_generation_logits: bool, backend: str):
    llm_return_logprobs_test_harness(prompt_logprobs,
                                     logprobs,
                                     return_context_logits,
                                     return_generation_logits,
                                     backend=backend)


@force_ampere
def test_llm_return_logprobs_streaming():
    llm_return_logprobs_test_harness(2, 2, False, True, streaming=True)


class DummyExecutorWorker3(GenerationExecutorWorker):
    should_raise_error = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.counter = 0
        self.failed_requests = set()

    def _engine_response_callback(self, response: tllm.Response):
        client_id = response.client_id
        if client_id in self.failed_requests:
            return response
        # Making the first response failed, and the subsequent responses successful
        if DummyExecutorWorker3.should_raise_error:
            DummyExecutorWorker3.should_raise_error = False
            print(f"Raise error for {client_id}")
            self.failed_requests.add(client_id)
            if not response.result.is_final:
                self.abort_request(client_id)
            return tllm.Response(
                request_id=self._client_id_to_request_id[client_id],
                client_id=client_id,
                error_msg="Test error")
        else:
            return response

    def _pop_result(self, client_id: int):
        # The actual worker didn't error, so it may continue generating result,
        # until the abort message reached it.
        # So we avoid removing the result queue.
        if client_id in self.failed_requests:
            return
        super()._pop_result(client_id)


DummyExecutor3 = DummyExecutorMeta("DummyExecutor3", (), {},
                                   worker_cls=DummyExecutorWorker3)


def test_llm_handling_per_request_error():
    llm = LLM(
        model=llama_model_path,
        executor_cls=DummyExecutor3,
        kv_cache_config=global_kvcache_config,
        fast_build=True,
    )
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


def test_llm_handling_per_request_error_async():
    llm = LLM(
        model=llama_model_path,
        executor_cls=DummyExecutor3,
        kv_cache_config=global_kvcache_config,
        fast_build=True,
    )
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


class DummyExecutorWorker4(GenerationExecutorWorker):
    should_raise_error = True

    def submit(self, request: GenerationRequest) -> GenerationResult:
        # Making the first response failed, and the subsequent responses successful
        if DummyExecutorWorker4.should_raise_error:
            DummyExecutorWorker4.should_raise_error = False
            raise RequestError("Test error")

        return super().submit(request)


DummyExecutor4 = DummyExecutorMeta("DummyExecutor4", (), {},
                                   worker_cls=DummyExecutorWorker4)


def test_llm_handling_per_request_submit_error():
    llm = LLM(
        model=llama_model_path,
        executor_cls=DummyExecutor4,
        kv_cache_config=global_kvcache_config,
        fast_build=True,
    )
    # The dummy executor will delay the responses
    sampling_params = SamplingParams(max_tokens=6)

    def batch_task():
        DummyExecutorWorker4.should_raise_error = True
        with pytest.raises(RequestError):
            for output in llm.generate(prompts,
                                       sampling_params=sampling_params):
                print(output)

        for output in llm.generate(prompts, sampling_params=sampling_params):
            print(output)

    batch_task()


def validate_stats(results,
                   pytorch_backend,
                   max_tokens,
                   enable_iter_req_stats=False):
    assert results
    assert len(results) == max_tokens if pytorch_backend else max_tokens + 1
    for iter, result in enumerate(results):
        ifbStats = result["inflightBatchingStats"]
        expected_num_scheduled = 1 if (iter < max_tokens) else 0
        assert ifbStats["numScheduledRequests"] == expected_num_scheduled
        if iter == 0:
            assert ifbStats["numContextRequests"] == 1
            assert ifbStats["numGenRequests"] == 0
            assert result["numActiveRequests"] == 1
        elif iter == max_tokens:
            assert ifbStats["numContextRequests"] == 0
            assert ifbStats["numGenRequests"] == 0
            assert result["numActiveRequests"] == 0
        else:
            assert ifbStats["numContextRequests"] == 0
            assert ifbStats["numGenRequests"] == 1
            assert result["numActiveRequests"] == 1

        if enable_iter_req_stats:
            assert "requestStats" in result
            req_stats = result["requestStats"]
            assert len(req_stats) == 1
            req_stat = req_stats[0]
            assert req_stat["numGeneratedTokens"] == iter + 1
            assert req_stat["scheduled"] == True
            assert req_stat[
                "stage"] == "GENERATION_IN_PROGRESS" if iter + 1 < max_tokens else "GENERATION_COMPLETE"
            assert req_stat["contextPrefillPosition"] == 4

        expected_num_completed = 1 if iter == len(results) - 1 else 0

        #TODO: For some reason, with stats_async and TRT backend, numCompleted is 0 at first iteration
        if pytorch_backend:
            assert result["numCompletedRequests"] == expected_num_completed


def llm_get_stats_test_harness(tp_size: int = 1,
                               return_context_logits: bool = False,
                               pytorch_backend: bool = False,
                               use_overlap: bool = False,
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
    print("enable_iter_req_stats: ", enable_iter_req_stats)
    print("-------------")

    llm_args_extra = {}
    sampling_args_extra = {}
    if return_context_logits:
        llm_args_extra["build_config"] = BuildConfig(gather_context_logits=True)
        llm_args_extra["gather_generation_logits"] = True
        sampling_args_extra["return_context_logits"] = True

    if pytorch_backend:
        llm_args_extra.update(
            dict(enable_iter_perf_stats=True,
                 enable_iter_req_stats=enable_iter_req_stats,
                 disable_overlap_scheduler=not use_overlap))
        LLM_CLASS = LLM_torch
    else:
        LLM_CLASS = LLM

    if not pytorch_backend:
        llm_args_extra["fast_build"] = True

    llm = LLM_CLASS(model=llama_model_path,
                    kv_cache_config=global_kvcache_config,
                    tensor_parallel_size=tp_size,
                    **llm_args_extra)

    max_tokens = 5
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     **sampling_args_extra)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(output)

    results = llm.get_stats(2)

    validate_stats(results, pytorch_backend, max_tokens, enable_iter_req_stats)

    assert not llm.get_stats(2)

    # test that IterationResult()._done is properly set
    _ = llm.generate(prompts, sampling_params=sampling_params)
    assert llm.get_stats(2)


@pytest.mark.parametrize("return_context_logits", [True, False])
@pytest.mark.parametrize("enable_iter_req_stats", [True, False])
def test_llm_get_stats(return_context_logits, enable_iter_req_stats):
    llm_get_stats_test_harness(tp_size=1,
                               return_context_logits=return_context_logits,
                               pytorch_backend=False,
                               enable_iter_req_stats=enable_iter_req_stats)


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
    LLM_CLASS = LLM_torch

    llm = LLM_CLASS(model=llama_model_path,
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
                                     return_context_logits: bool = False,
                                     pytorch_backend: bool = False,
                                     use_overlap: bool = False,
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
    print("enable_iter_req_stats: ", enable_iter_req_stats)
    print("-------------")

    llm_args_extra = {}
    sampling_args_extra = {}
    if return_context_logits:
        llm_args_extra["build_config"] = BuildConfig(gather_context_logits=True)
        sampling_args_extra["return_context_logits"] = True

    if pytorch_backend:
        llm_args_extra.update(
            dict(enable_iter_perf_stats=True,
                 enable_iter_req_stats=enable_iter_req_stats,
                 disable_overlap_scheduler=not use_overlap))
        LLM_CLASS = LLM_torch
    else:
        LLM_CLASS = LLM
        llm_args_extra["fast_build"] = True

    llm = LLM_CLASS(model=llama_model_path,
                    kv_cache_config=global_kvcache_config,
                    tensor_parallel_size=tp_size,
                    **llm_args_extra)

    max_tokens = 6
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     **sampling_args_extra)

    async def task0():
        async for output in llm.generate_async(prompts[0],
                                               streaming=True,
                                               sampling_params=sampling_params):
            print(output)

    async def task1():
        results = []
        await asyncio.sleep(
            3)  # ensure there's stats to collect for the assertion
        async for stats in llm.get_stats_async(timeout=2):
            results.append(stats)

        assert results
        if not use_overlap:
            validate_stats(results, pytorch_backend, max_tokens,
                           enable_iter_req_stats)

    async def main():
        for i in range(2):  # test recurrent usage
            await asyncio.gather(task0(), task1())

    asyncio.run(main())


@pytest.mark.parametrize("return_context_logits", [True, False])
@pytest.mark.parametrize("enable_iter_req_stats", [True, False])
def test_llm_get_stats_async(return_context_logits, enable_iter_req_stats):
    llm_get_stats_async_test_harness(
        tp_size=1,
        return_context_logits=return_context_logits,
        pytorch_backend=False,
        enable_iter_req_stats=enable_iter_req_stats)


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
                  enable_chunked_prefill=False,
                  fast_build=True)

        with pytest.raises(ValueError):
            output = llm.generate_async(
                "A " * build_config.max_num_tokens,
                sampling_params=sampling_params,
            ).result()

    def success_path():
        llm = LLM(
            model=llama_model_path,
            kv_cache_config=global_kvcache_config,
            build_config=build_config,
            enable_chunked_prefill=True,
            fast_build=True,
        )

        output = llm.generate_async(
            "A " * build_config.max_num_tokens,
            sampling_params=sampling_params,
        ).result()

    fail_path()
    success_path()


def _test_llm_capture_request_error(pytorch_backend: bool, tp_size: int = 1):
    llm_args_extra = {}
    if pytorch_backend:
        LLM_CLASS = LLM_torch
        llm_args_extra["max_num_tokens"] = 64
    else:
        LLM_CLASS = LLM
        build_config = BuildConfig()
        build_config.max_num_tokens = 64
        llm_args_extra["fast_build"] = True
        llm_args_extra["build_config"] = build_config

    llm = LLM_CLASS(
        model=llama_model_path,
        tensor_parallel_size=tp_size,
        **llm_args_extra,
    )

    prompt = 'A ' * 65  # the minimum max_num_tokens is 64
    if pytorch_backend:
        # pytorch backend will raise ValueError for max_num_tokens
        with pytest.raises(ValueError):
            llm.generate(prompt)
    else:
        with pytest.raises(RequestError):
            llm.generate(prompt)


def test_llm_capture_request_error():
    _test_llm_capture_request_error(pytorch_backend=False, tp_size=1)


def test_llm_shutdown_executor():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
        fast_build=True,
    )

    llm.generate("A")
    llm.shutdown()

    with pytest.raises(RuntimeError):
        llm.generate("A")


def test_llm_api_jupyter_scenario():

    with LLM(
            model=llama_model_path,
            kv_cache_config=global_kvcache_config,
            fast_build=True,
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
    kwargs = {}
    if backend not in ["pytorch", "autodeploy"]:
        kwargs["fast_build"] = True
        LLM_CLASS = LLM
    else:
        LLM_CLASS = LLM_torch

    llm = LLM_CLASS(model=llama_model_path,
                    backend=backend,
                    kv_cache_config=global_kvcache_config,
                    tensor_parallel_size=tp_size,
                    num_postprocess_workers=2,
                    postprocess_tokenizer_dir=llama_model_path,
                    **kwargs)
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


@pytest.mark.parametrize("streaming", [True, False])
def test_llm_with_postprocess_parallel_and_result_handler(streaming):
    run_llm_with_postprocess_parallel_and_result_handler(streaming,
                                                         backend=None,
                                                         tp_size=1)


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


@force_ampere
@pytest.mark.parametrize("sampling_params",
                         sampling_params_for_aborting_request)
def test_llm_abort_request(llm_for_sampling_params,
                           sampling_params: SamplingParams):
    run_llm_abort_request(llm=llm_for_sampling_params,
                          sampling_params=sampling_params)


def test_llm_sampling_params_n_lt_max_batch_size():
    sampling_params = SamplingParams(n=2, top_p=0.95)
    build_config = BuildConfig(max_batch_size=1, max_seq_len=1024)
    llm = LLM(model=llama_model_path,
              kv_cache_config=global_kvcache_config,
              build_config=build_config,
              fast_build=True)

    with pytest.raises(ValueError):
        llm.generate_async(prompts[0], sampling_params=sampling_params)


def test_llm_api_draft_target():
    sampling_params = SamplingParams(max_tokens=4)

    build_config = BuildConfig(
        speculative_decoding_mode=SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL,
        max_draft_len=4,
        max_batch_size=2,
        max_beam_width=1,
        max_seq_len=128,
        max_num_tokens=64)

    llm = LLM(llama_model_path,
              build_config=build_config,
              kv_cache_config=global_kvcache_config,
              fast_build=True)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def test_llm_context_only_timed_out():
    tp_size = 1
    use_overlap = False
    enable_iter_req_stats = False

    llm_args_extra = {}

    llm_args_extra.update(
        dict(enable_iter_perf_stats=True,
             enable_iter_req_stats=enable_iter_req_stats,
             disable_overlap_scheduler=not use_overlap))
    LLM_CLASS = LLM_torch

    llm = LLM_CLASS(model=llama_model_path,
                    kv_cache_config=global_kvcache_config,
                    tensor_parallel_size=tp_size,
                    cache_transceiver_config=CacheTransceiverConfig(
                        backend="DEFAULT", kv_transfer_timeout_ms=1000),
                    **llm_args_extra)

    max_tokens = 1
    sampling_params = SamplingParams(max_tokens=max_tokens)

    disaggregated_params = DisaggregatedParams(request_type="context_only")

    prompts0 = [
        "What is your name?",
    ]
    prompts1 = [
        "Nvidia is awesome because",
    ]

    # Send context-only request
    for output in llm.generate(prompts1,
                               sampling_params=sampling_params,
                               disaggregated_params=disaggregated_params):
        print(output)

    results = llm.get_stats(2)
    assert len(results) == 1
    context_only_used_num_blocks = results[0]["kvCacheStats"]["usedNumBlocks"]
    print(f"Context only used num blocks: {context_only_used_num_blocks}")

    # Sleep 5 seconds to allow context only request to time out
    time.sleep(5)

    # Send regular request
    for output in llm.generate(prompts0, sampling_params=sampling_params):
        print(output)

    # Get number of allocated blocks
    results = llm.get_stats(2)
    assert len(results) == 1
    final_used_num_blocks = results[0]["kvCacheStats"]["usedNumBlocks"]

    assert final_used_num_blocks == 0
