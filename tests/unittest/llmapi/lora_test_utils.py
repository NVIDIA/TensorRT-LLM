from typing import OrderedDict

from utils.llm_data import llm_models_root
from utils.util import duplicate_list_to_length, flatten_list, similar

from tensorrt_llm import SamplingParams
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi.llm import BaseLLM, _TorchLLM, _TrtLLM
from tensorrt_llm.llmapi.llm_utils import BuildConfig
from tensorrt_llm.lora_manager import LoraConfig


def check_multi_unique_lora_adapters_from_request(
        llm: BaseLLM, hf_lora_dirs: list[str],
        lora_adapter_count_per_call: list[int], repeats: int):
    """Calls llm.generate s.t. for each c in lora_adapter_count_per_call, llm.generate is called with c requests.
    All requests sent to llm.generate over all calls (in a single repeats iteration) are configured to each use a unique
    LoRA adapter. This entire process is done in a loop (with the same requests) 'repeats' times with the same requests.
    Asserts the output of each llm.generate call is similar to the expected.
    """  # noqa: D205
    total_lora_adapters = sum(lora_adapter_count_per_call)

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

    # Perform repeats of the same requests to test reuse and reload of adapters previously unloaded from cache
    try:
        for _ in range(repeats):
            last_idx = 0
            for adapter_count in lora_adapter_count_per_call:
                sampling_params = SamplingParams(max_tokens=20)
                outputs = llm.generate(
                    prompts_to_generate[last_idx:last_idx + adapter_count],
                    sampling_params,
                    lora_request=lora_requests[last_idx:last_idx + adapter_count])
                for output, ref in zip(
                        outputs, references[last_idx:last_idx + adapter_count]):
                    assert similar(output.outputs[0].text, ref)
                last_idx += adapter_count
    finally:
        llm.shutdown()


def check_pytorch_llama_7b_multi_lora_from_request_test_harness(
        max_lora_rank: int = 8, **llm_kwargs) -> None:
    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"

    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=max_lora_rank)

    llm = _TorchLLM(hf_model_dir, lora_config=lora_config, **llm_kwargs)
    _check_llama_7b_multi_lora_from_request_test_harness(llm)


def check_trt_python_llama_7b_multi_lora_from_request_test_harness(
        max_lora_rank: int = 8, **llm_kwargs):
    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"

    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    build_config = BuildConfig(lora_config=LoraConfig(
        lora_target_modules=['attn_q', 'attn_k', 'attn_v']))
    llm = _TrtLLM(hf_model_dir,
                  enable_lora=True,
                  max_lora_rank=max_lora_rank,
                  build_config=build_config,
                  fast_build=True,
                  **llm_kwargs)
    _check_llama_7b_multi_lora_from_request_test_harness(llm)


def _check_llama_7b_multi_lora_from_request_test_harness(llm: BaseLLM) -> None:
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
    try:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=[None, lora_req1, lora_req2, None, lora_req1, lora_req2])
    finally:
        llm.shutdown()
    for output, ref, key_word in zip(outputs, references, key_words):
        assert similar(output.outputs[0].text,
                       ref) or key_word in output.outputs[0].txt
