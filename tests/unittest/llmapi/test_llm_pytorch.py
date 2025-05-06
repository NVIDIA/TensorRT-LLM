import pytest

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .test_llm import (get_model_path, global_kvcache_config, llama_model_path,
                       llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness, prompts,
                       run_llm_abort_request,
                       run_llm_with_postprocess_parallel_and_result_handler,
                       tinyllama_guided_decoding_test_harness,
                       tinyllama_logits_processor_test_harness)
from utils.util import force_ampere
# isort: on


@force_ampere
def test_tinyllama_guided_decoding():
    pytest.skip(reason="https://nvbugs/5240350")
    tinyllama_guided_decoding_test_harness(backend="pytorch")


@force_ampere
def test_tinyllama_logits_processor():
    tinyllama_logits_processor_test_harness(backend="pytorch")


@pytest.mark.parametrize("return_context_logits, use_overlap", [
    (False, False),
    (False, True),
])
def test_llm_get_stats(return_context_logits, use_overlap):
    llm_get_stats_test_harness(tp_size=1,
                               return_context_logits=return_context_logits,
                               pytorch_backend=True,
                               use_overlap=use_overlap)


@pytest.mark.parametrize("return_context_logits, use_overlap", [
    (False, False),
    (False, True),
])
def test_llm_get_stats_async(return_context_logits, use_overlap):
    llm_get_stats_async_test_harness(
        tp_size=1,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        use_overlap=use_overlap)


@force_ampere
@pytest.mark.parametrize(
    "sampling_params",
    [
        SamplingParams()  # pytorch only supports n=1
    ])
def test_llm_abort_request(sampling_params):
    from tensorrt_llm._torch import LLM as LLM_torch
    llm = LLM_torch(model=llama_model_path,
                    kv_cache_config=global_kvcache_config)
    run_llm_abort_request(llm=llm, sampling_params=sampling_params)


def test_llm_reward_model():
    rm_model_path = get_model_path("Qwen2.5-Math-PRM-7B")
    tokenizer = TransformersTokenizer.from_pretrained(rm_model_path)
    tokenized_input = tokenizer(prompts, return_tensors="pt")["input_ids"]

    from tensorrt_llm._torch import LLM as LLM_torch
    from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
    llm = LLM_torch(
        model=rm_model_path,
        pytorch_backend_config=PyTorchConfig(attn_backend="VANILLA"))

    sampling_params = SamplingParams(return_context_logits=True)

    outputs = llm.generate(prompts, sampling_params)
    scores = outputs[0].context_logits

    print(scores)

    assert scores.shape == (tokenized_input.shape[1], 2)
    assert not outputs[0].outputs[0].text


@pytest.mark.parametrize("streaming", [True, False])
def test_llm_with_postprocess_parallel_and_result_handler(streaming):
    run_llm_with_postprocess_parallel_and_result_handler(streaming,
                                                         "pytorch",
                                                         tp_size=1)
