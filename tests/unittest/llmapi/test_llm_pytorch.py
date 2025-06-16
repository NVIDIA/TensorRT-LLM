import pytest

from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from .test_llm import (get_model_path, global_kvcache_config, llama_model_path,
                       llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness, prompts,
                       run_llm_abort_request,
                       run_llm_with_postprocess_parallel_and_result_handler,
                       tinyllama_logits_processor_test_harness)
from utils.util import force_ampere, similar, skip_gpu_memory_less_than_40gb, skip_gpu_memory_less_than_80gb, skip_gpu_memory_less_than_138gb
from utils.llm_data import llm_models_root
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
import tempfile

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM

# isort: on


@force_ampere
def test_tinyllama_logits_processor():
    tinyllama_logits_processor_test_harness(backend="pytorch")


@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_iter_req_stats", [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
    ])
def test_llm_get_stats(return_context_logits, use_overlap,
                       enable_iter_req_stats):
    llm_get_stats_test_harness(tp_size=1,
                               return_context_logits=return_context_logits,
                               pytorch_backend=True,
                               use_overlap=use_overlap,
                               enable_iter_req_stats=enable_iter_req_stats)


@pytest.mark.parametrize(
    "return_context_logits, use_overlap, enable_iter_req_stats", [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
    ])
def test_llm_get_stats_async(return_context_logits, use_overlap,
                             enable_iter_req_stats):
    llm_get_stats_async_test_harness(
        tp_size=1,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        use_overlap=use_overlap,
        enable_iter_req_stats=enable_iter_req_stats)


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
    llm = LLM_torch(model=rm_model_path,
                    attn_backend="VANILLA",
                    disable_overlap_scheduler=True)

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


def llama_v2_13b_lora_test_harness(**llm_kwargs) -> None:
    from tensorrt_llm._torch.llm import LLM

    lora_config = LoraConfig(lora_dir=[
        f"{llm_models_root()}/llama-models-v2/chinese-llama-2-lora-13b"
    ],
                             max_lora_rank=64)
    llm = LLM(model=f"{llm_models_root()}/llama-models-v2/llama-v2-13b-hf",
              lora_config=lora_config,
              **llm_kwargs)

    prompts = [
        "今天天气很好，我到公园的时候，",
    ]
    references = [
        "发现公园里到处都是人，有的在跑步，有的在打羽毛球，还有的",
    ]
    sampling_params = SamplingParams(max_tokens=20, add_special_tokens=False)
    lora_req = LoRARequest(
        "task-0", 0,
        f"{llm_models_root()}/llama-models-v2/chinese-llama-2-lora-13b")
    lora_request = [lora_req]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    assert similar(outputs[0].outputs[0].text, references[0])


def llama_7b_multi_lora_test_harness(**llm_kwargs) -> None:
    from tensorrt_llm._torch.llm import LLM

    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"
    hf_lora_dir1 = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"
    hf_lora_dir2 = f"{llm_models_root()}/llama-models/Japanese-Alpaca-LoRA-7b-v0"

    # For LoRA checkpoints without finetuned embedding and lm_head, we can either:
    # (1) specify lora_target_modules, or
    # (2) provide a lora_dir to infer the lora_target_modules.
    lora_config = LoraConfig(lora_target_modules=['attn_q', 'attn_k', 'attn_v'],
                             max_lora_rank=8)
    llm = LLM(hf_model_dir, lora_config=lora_config, **llm_kwargs)

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
def test_llama_7b_lora_default_modules() -> None:
    from tensorrt_llm._torch.llm import LLM

    lora_config = LoraConfig(max_lora_rank=64)

    hf_model_dir = f"{llm_models_root()}/llama-models/llama-7b-hf"

    llm = LLM(model=hf_model_dir, lora_config=lora_config)

    hf_lora_dir = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"
    prompts = [
        "美国的首都在哪里? \n答案:",
    ]
    references = [
        "美国的首都是华盛顿。\n\n美国的",
    ]
    sampling_params = SamplingParams(max_tokens=20, add_special_tokens=False)
    lora_req = LoRARequest("luotuo", 1, hf_lora_dir)
    lora_request = [lora_req]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    assert similar(outputs[0].outputs[0].text, references[0])


@skip_gpu_memory_less_than_40gb
def test_llama_7b_multi_lora():
    llama_7b_multi_lora_test_harness()


# TODO smor: currently Nemotron-Super-49B-v1 with LoRA memory consumption is overly high
# https://jirasw.nvidia.com/browse/TRTLLM-5045
@skip_gpu_memory_less_than_138gb
def test_nemotron_nas_lora() -> None:
    from tensorrt_llm._torch.llm import LLM

    lora_config = LoraConfig(lora_dir=[
        f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_r64"
    ],
                             max_lora_rank=64,
                             max_loras=1,
                             max_cpu_loras=1)

    llm = LLM(
        model=
        f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1",
        lora_config=lora_config,
    )

    prompts = [
        "Hello, how are you?",
        "Hello, how are you?",
    ]

    sampling_params = SamplingParams(max_tokens=10, add_special_tokens=False)
    lora_req = LoRARequest(
        "task-0", 0,
        f"{llm_models_root()}/nemotron-nas/Llama-3_3-Nemotron-Super-49B-v1-lora-adapter_r64"
    )
    lora_request = [lora_req, None]

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    assert similar(outputs[0].outputs[0].text, outputs[1].outputs[0].text)


@skip_gpu_memory_less_than_80gb
def test_codellama_fp8_with_bf16_lora() -> None:
    from tensorrt_llm._torch.llm import LLM

    model_dir = f"{llm_models_root()}/codellama/CodeLlama-7b-Instruct-hf/"
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)

    target_modules = ['attn_q', 'attn_k', 'attn_v']

    # Set up temporary directory for LoRA adapters
    with tempfile.TemporaryDirectory() as lora_dir:
        print("Creating dummy LoRAs...")

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        hf_modules = ["q_proj", "k_proj", "v_proj"]

        lora_config = PeftLoraConfig(r=8,
                                     target_modules=hf_modules,
                                     bias="none",
                                     task_type="CAUSAL_LM")

        lora_paths = []
        for i in range(2):
            lora_model = get_peft_model(model, lora_config)
            for param in lora_model.parameters():
                param.data.zero_()
            lora_path = f"{lora_dir}/lora_{i}"
            lora_model.save_pretrained(lora_path)
            lora_paths.append(lora_path)

        lora_config = LoraConfig(lora_dir=lora_paths,
                                 lora_target_modules=target_modules,
                                 max_lora_rank=8)

        llm = LLM(model_dir, quant_config=quant_config, lora_config=lora_config)

        prompts = [
            "Write a function that calculates the Fibonacci sequence.",
            "Convert this C++ code to Python: int x = 0; x++;",
        ]

        lora_req1 = LoRARequest("lora-1", 0, lora_paths[0])
        lora_req2 = LoRARequest("lora-2", 1, lora_paths[1])
        lora_requests = [lora_req1, lora_req2]
        sampling_params = SamplingParams(max_tokens=200)

        outputs = llm.generate(prompts,
                               sampling_params,
                               lora_request=lora_requests)

        assert len(outputs) == 2
