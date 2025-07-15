from typing import OrderedDict, Type

from utils.llm_data import llm_models_root
from utils.util import duplicate_list_to_length, flatten_list, similar

from tensorrt_llm import SamplingParams
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi.llm import BaseLLM


def check_llama_7b_multi_unique_lora_adapters_from_request(
        lora_adapter_count_per_call: list[int], repeat_calls: int,
        repeats_per_call: int, llm_class: Type[BaseLLM], **llm_kwargs):
    """Calls llm.generate s.t. for each C in lora_adapter_count_per_call, llm.generate is called with C requests
    repeated 'repeats_per_call' times, where each request is configured with a unique LoRA adapter ID.
    This entire process is done in a loop 'repeats_per_call' times with the same requests.
    Asserts the output of each llm.generate call is similar to the expected.
    """  # noqa: D205
    total_lora_adapters = sum(lora_adapter_count_per_call)
    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"
    hf_lora_dirs = [
        f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1",
        f"{llm_models_root()}/llama-models/Japanese-Alpaca-LoRA-7b-v0"
    ]
    # Each prompt should have a reference for every LoRA adapter dir (in the same order as in hf_lora_dirs)
    prompt_to_references = OrderedDict({
        "美国的首都在哪里? \n答案:": [
            "美国的首都是华盛顿。\n\n美国的",
            "纽约\n\n### カンファレンスの",
        ],
        "アメリカ合衆国の首都はどこですか? \n答え:": [
            "华盛顿。\n\n英国の首都是什",
            "ワシントン\nQ1. アメリカ合衆国",
        ],
    })

    prompts_to_generate = duplicate_list_to_length(
        flatten_list([[prompt] * len(hf_lora_dirs)
                      for prompt in prompt_to_references.keys()]),
        total_lora_adapters)
    references = duplicate_list_to_length(
        flatten_list(list(prompt_to_references.values())), total_lora_adapters)
    lora_requests = [
        LoRARequest(str(i), i, hf_lora_dirs[i % len(hf_lora_dirs)])
        for i in range(total_lora_adapters)
    ]
    llm = llm_class(hf_model_dir, **llm_kwargs)

    # Perform repeats of the same requests to test reuse and reload of adapters previously unloaded from cache
    try:
        for _ in range(repeat_calls):
            last_idx = 0
            for adapter_count in lora_adapter_count_per_call:
                sampling_params = SamplingParams(max_tokens=20)
                outputs = llm.generate(
                    prompts_to_generate[last_idx:last_idx + adapter_count] *
                    repeats_per_call,
                    sampling_params,
                    lora_request=lora_requests[last_idx:last_idx +
                                               adapter_count] *
                    repeats_per_call)
                for output, ref in zip(
                        outputs, references[last_idx:last_idx + adapter_count] *
                        repeats_per_call):
                    assert similar(output.outputs[0].text, ref)
                last_idx += adapter_count
    finally:
        llm.shutdown()


def check_llama_7b_multi_lora_from_request_test_harness(
        llm_class: Type[BaseLLM], **llm_kwargs) -> None:
    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"
    hf_lora_dir1 = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"
    hf_lora_dir2 = f"{llm_models_root()}/llama-models/Japanese-Alpaca-LoRA-7b-v0"
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
    key_words = [
        "沃尔玛",
        "华盛顿",
        "纽约",
        "Washington",
        "华盛顿",
        "ワシントン",
    ]
    lora_req1 = LoRARequest("luotuo", 1, hf_lora_dir1)
    lora_req2 = LoRARequest("Japanese", 2, hf_lora_dir2)
    sampling_params = SamplingParams(max_tokens=20)

    llm = llm_class(hf_model_dir, **llm_kwargs)
    try:
        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=[
                                   None, lora_req1, lora_req2, None, lora_req1,
                                   lora_req2
                               ])
    finally:
        llm.shutdown()
    for output, ref, key_word in zip(outputs, references, key_words):
        assert similar(output.outputs[0].text,
                       ref) or key_word in output.outputs[0].txt
