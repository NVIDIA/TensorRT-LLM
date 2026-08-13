import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import skip_fp8_pre_ada, skip_gpu_memory_less_than

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm import RequestOutput
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, LoadFormat
from tensorrt_llm.sampling_params import SamplingParams


def get_logprobs(token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    raw_probs = torch.softmax(logits, dim=-1)
    index = token_ids.unsqueeze(1)
    assert index.device == raw_probs.device, f"index and raw_probs should be on the same device, but got index location: {index.device}, raw_probs location: {raw_probs.device}"
    token_probs = torch.gather(raw_probs, dim=1, index=index).squeeze(-1)
    return torch.log(token_probs)


def extract_prefill_logprobs(result: RequestOutput) -> torch.Tensor:
    token_ids = torch.tensor(result.prompt_token_ids[1:])
    logits = result.context_logits[:-1, :]
    return get_logprobs(token_ids.cuda(), logits)


def extract_decode_logprobs(result: RequestOutput,
                            gen_idx: int = 0) -> torch.Tensor:
    token_ids = torch.tensor(result.outputs[gen_idx].token_ids)
    logits = result.outputs[gen_idx].generation_logits
    return get_logprobs(token_ids, logits)


def create_nemotron_h_llm(model_folder,
                          use_cuda_graph,
                          disable_overlap_scheduler,
                          max_batch_size,
                          mamba_ssm_cache_dtype=None,
                          enable_chunked_prefill=False,
                          max_num_tokens=8192,
                          load_format=None):
    """Create LLM with specific overlap scheduler setting"""
    model_dir = f"{llm_models_root(check=True)}/{model_folder}"
    kwargs = {}
    if max_num_tokens is not None:
        kwargs["max_num_tokens"] = max_num_tokens
    if load_format is not None:
        kwargs["load_format"] = load_format

    return LLM(
        model=model_dir,
        tensor_parallel_size=1,
        max_batch_size=max_batch_size,
        cuda_graph_config=CudaGraphConfig() if use_cuda_graph else None,
        disable_overlap_scheduler=disable_overlap_scheduler,
        kv_cache_config=KvCacheConfig(
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype)
        if mamba_ssm_cache_dtype is not None else KvCacheConfig(),
        enable_chunked_prefill=enable_chunked_prefill,
        **kwargs,
    )


@pytest.mark.parametrize("mamba_ssm_cache_dtype", [None, "float32"],
                         ids=lambda n: f"mamba_ssm_cache_dtype:{n}")
@pytest.mark.parametrize("model_folder", [
    pytest.param("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                 marks=skip_gpu_memory_less_than((2 * 30 + 1) * 2**30)),
    pytest.param("NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                 marks=skip_gpu_memory_less_than((30 + 1) * 2**30)),
])
def test_nemotron_h_sanity(mamba_ssm_cache_dtype, model_folder):
    # Skip test if FP8 is not supported on the current architecture.
    use_fp8 = model_folder == "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    skip_fp8_pre_ada(use_fp8)

    torch.cuda.empty_cache()

    text_prompts = [
        "The future of AI is",
        "The president of the United States is",
    ]
    num_prompts = len(text_prompts)

    with create_nemotron_h_llm(
            model_folder=model_folder,
            use_cuda_graph=False,
            disable_overlap_scheduler=False,
            max_batch_size=num_prompts,
            mamba_ssm_cache_dtype=mamba_ssm_cache_dtype,
            load_format=LoadFormat.DUMMY,
    ) as nemotron_h:
        sampling_params = SamplingParams(max_tokens=9,
                                         temperature=0.0,
                                         add_special_tokens=False,
                                         return_context_logits=True,
                                         return_generation_logits=True)

        # Non-batching prefill sanity check.
        _ = [
            nemotron_h.generate(text_prompt, sampling_params)
            for text_prompt in text_prompts
        ]

        # Batching prefill sanity check.
        results_batching = nemotron_h.generate(text_prompts, sampling_params)
        completions_batching = [
            result.outputs[0].text for result in results_batching
        ]

        # Decoding sanity check.
        text_prompts_with_completions = [
            f"{text_prompts[i]}{completions_batching[i]}"
            for i in range(num_prompts)
        ]
        sampling_params.max_tokens = 1
        nemotron_h.generate(text_prompts_with_completions, sampling_params)
