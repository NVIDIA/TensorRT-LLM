import asyncio
import json
import os
import sys
import tempfile
import time
from typing import List, Optional

import pytest
import torch
import transformers

from tensorrt_llm.executor import (ExecutorBindingsWorker, GenerationRequest,
                                   GenerationResult, LoRARequest)
from tensorrt_llm.hlapi import (LLM, BuildCacheConfig, KvCacheConfig,
                                SamplingParams)
from tensorrt_llm.hlapi.llm_utils import BuildConfig, _ParallelConfig
from tensorrt_llm.hlapi.tokenizer import TokenizerBase, TransformersTokenizer
from tensorrt_llm.hlapi.utils import get_total_gpu_memory
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.models import PretrainedConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar, skip_less_than_40gb_memory

from tensorrt_llm.models.automodel import AutoConfig, AutoModelForCausalLM

# The unittests are based on the tiny-llama, which is fast to build and run.
# There are other tests based on llama-7B model, such as the end-to-end tests in test_e2e.py, and parallel tests in
# test_llm_multi_gpu.py.


def get_model_path(model_name):
    engine_dir = os.environ.get('LLM_ENGINE_DIR', None)
    if engine_dir:
        return engine_dir
    return str(llm_models_root() / model_name)


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


def llm_check_output(llm: LLM,
                     inputs: List[str],
                     references: List[str],
                     similar_threshold: float = 0.8,
                     *gen_args,
                     **gen_kwargs):
    outputs = llm.generate(inputs, *gen_args, **gen_kwargs)
    assert len(outputs) == len(references)

    for output, target_output in zip(outputs, references):
        if isinstance(target_output, list):
            # N output
            assert len(output.outputs) == len(target_output)
            for out, ref in zip(output.outputs, target_output):
                assert similar(out.text, ref, threshold=similar_threshold)
        else:
            assert similar(output.outputs[0].text, target_output)


default_model_name = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
mixtral_model_name = "Mixtral-8x7B-v0.1"

llama_model_path = get_model_path(default_model_name)
llm_engine_dir = os.environ.get('LLM_ENGINE_DIR', './tmp.engine')

cnn_dailymail_path = str(llm_models_root() / "datasets" / "cnn_dailymail")

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
    sampling_params = SamplingParams(max_tokens=8)
    llm_test_harness(
        llama_model_path,
        prompts, ["D E F G H I J K"],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))


@force_ampere
def test_llm_loading_from_ckpt():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    assert tokenizer is not None

    ckpt_dir = tempfile.TemporaryDirectory()
    llama = AutoModelForCausalLM.from_hugging_face(llama_model_path)
    llama.save_checkpoint(ckpt_dir.name)
    del llama

    llm_test_harness(
        ckpt_dir.name,
        prompts, ["D E F G H I J K"],
        tokenizer=tokenizer,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
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
    llm_test_harness(
        dummy_dir.name,
        prompts, ["A placeholder reference for dummy-weight engine."],
        sampling_params=sampling_params,
        similar_threshold=0.0,
        load_format='dummy',
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))


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

    sampling_params = SamplingParams(end_id=2, pad_id=2, max_tokens=8)

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
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    return llm


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
    build_config.gather_context_logits = gather_context_logits
    build_config.gather_generation_logits = gather_generation_logits

    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        build_config=build_config,
    )
    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        return_log_probs=return_log_probs)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        if gather_context_logits:
            assert output.context_logits is not None
            assert len(prompts[0].split()) + 1 == output.context_logits.shape[0]
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
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )
    stop_id = llm.tokenizer.encode("N", add_special_tokens=False)[-1]

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M"],
                     sampling_params=SamplingParams(stop_token_ids=[stop_id]))

    llm_check_output(llm,
                     prompts, ["D E F G H I J K L M N"],
                     sampling_params=SamplingParams(
                         stop_token_ids=[stop_id],
                         include_stop_str_in_output=True))

    llm_check_output(llm,
                     prompts, ["D E F G H"],
                     sampling_params=SamplingParams(stop="I J"))

    llm_check_output(llm,
                     prompts, ["D E F G H I J"],
                     sampling_params=SamplingParams(
                         stop="I J", include_stop_str_in_output=True))

    llm_check_output(llm,
                     prompts, ["D E F G H"],
                     sampling_params=SamplingParams(stop=["F E", "I J"],
                                                    stop_token_ids=[stop_id]))


@force_ampere
def test_generate_with_bad_words():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
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
def test_generate_with_logits_post_processor():
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]

    def logits_post_processor(req_id: int, logits: torch.Tensor,
                              ids: List[List[int]], stream_ptr: int,
                              client_id: Optional[int]):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits[:] = float("-inf")
            logits[..., biased_word_id] = 0

    sampling_params = SamplingParams(max_tokens=6,
                                     logits_post_processor_name="my_logits_pp")

    llm_test_harness(
        llama_model_path,
        prompts, ["Z Z Z Z Z Z"],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        logits_post_processor_map={"my_logits_pp": logits_post_processor})


def llama_v2_13b_lora_test_harness(**llm_kwargs):
    hf_model_dir = get_model_path("llama-models-v2/llama-v2-13b-hf")
    hf_lora_dir = get_model_path("llama-models-v2/chinese-llama-2-lora-13b")

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

    build_config = BuildConfig(lora_config=LoraConfig(
        lora_dir=[hf_lora_dir1, hf_lora_dir2],
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


@skip_less_than_40gb_memory
def test_llama_v2_13b_lora():
    llama_v2_13b_lora_test_harness()


@skip_less_than_40gb_memory
def test_llama_7b_multi_lora():
    llama_7b_multi_lora_test_harness(max_loras=1, max_cpu_loras=8)


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
    llm = LLM(model=llama_model_path,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))
    sampling_params = SamplingParams(max_tokens=6)
    for i in range(20):
        llm.generate(prompts, sampling_params=sampling_params)

    num_remaining_results = len(llm._executor._results)
    print(f"result.size: {num_remaining_results}")
    assert num_remaining_results == 0


def test_llm_build_cache():
    # Activate the build-cache
    cache_config = BuildCacheConfig(max_records=1, max_cache_storage_gb=10)
    sampling_params = SamplingParams(max_tokens=6)

    def first_run():
        llm = LLM(model=llama_model_path,
                  kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
                  enable_build_cache=cache_config)
        llm_check_output(llm,
                         prompts, ["D E F G H I J K"],
                         sampling_params=sampling_params)

    def second_run():
        llm = LLM(model=llama_model_path,
                  kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
                  enable_build_cache=cache_config)
        llm_check_output(llm,
                         prompts, ["D E F G H I J K"],
                         sampling_params=sampling_params)

        # the cache should be hitted
        assert llm.llm_build_stats.cache_hitted

    first_run()
    second_run()


def test_executor_catching_exception():

    class DummyError(Exception):
        pass

    llm = LLM(model=llama_model_path,
              kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4))

    sampling_params = SamplingParams(max_tokens=6)

    def test_before_generation():
        # Since we cannot alter the executor's behavior, we put a dummy error in the error queue
        llm._executor._error_queue.put(DummyError("Test exception"))
        # The dummy error should be caught and raised in the main thread.
        with pytest.raises(DummyError):
            for output in llm.generate(prompts,
                                       sampling_params=sampling_params):
                pass

    def test_during_generation():
        with pytest.raises(DummyError):
            prompts = ["A B C"] * 10
            futures = []
            for no, prompt in enumerate(prompts):
                futures.append(
                    llm.generate_async(prompt, sampling_params=sampling_params))
                if no == 3:
                    # This exception should be caught and raised in the main thread. Before the 4-th output is generated.
                    llm._executor._error_queue.put(DummyError("Test exception"))

            for no, future in enumerate(futures):
                print(future.result())

    test_before_generation()
    test_during_generation()


class DummyExecutor:

    @classmethod
    def create(cls, engine, executor_config, *args, **kwargs):
        return DummyExecutorWorker(engine=engine,
                                   executor_config=executor_config)


class DummyExecutorWorker(ExecutorBindingsWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def submit(self, request: GenerationRequest) -> GenerationResult:
        # This is copied from the ExecutorBindingsWorker.submit method with minor modification

        self.start()

        req_id = self._enqueue_request(request)
        request.set_id(req_id)

        result = GenerationResult(request)

        # Force the responses to be delayed
        time.sleep(1)

        print(f"number of pending responses: {len(self._pending_responses)}")
        assert self._pending_responses

        self._results[req_id] = result

        assert self._cleanup_pending_responses()

        return result


def test_executor_pending_requests():
    llm = LLM(model=llama_model_path, executor_cls=DummyExecutor)
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

    @classmethod
    def create(cls, engine, executor_config, *args, **kwargs):
        return DummyExecutorWorker2(engine=engine,
                                    executor_config=executor_config)


class DummyExecutorWorker2(ExecutorBindingsWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.counter = 0

    def await_response_task(self) -> bool:
        self.counter += 1

        if self.counter == 2:  # raise exception on the third token
            print(f"To raise exception")
            raise ValueError("Test exception")

        return super().await_response_task()


def test_executor_process_background_error():
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


# TODO[chunweiy]: Add test for loading inmemory model

if __name__ == '__main__':
    #test_executor_results_cleanup()
    #test_llm_loading_from_hf()
    #test_executor_catching_exception()
    test_executor_pending_requests()
    #test_executor_process_background_error()
