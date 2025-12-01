### :title Speculative Decoding
### :order 5
### :section Customization
from typing import Optional

import click

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (EagleDecodingConfig, KvCacheConfig,
                                 MTPDecodingConfig, NGramDecodingConfig)

prompts = [
    "What is the capital of France?",
    "What is the future of AI?",
]


def run_MTP(model: Optional[str] = None):
    spec_config = MTPDecodingConfig(num_nextn_predict_layers=1,
                                    use_relaxed_acceptance_for_thinking=True,
                                    relaxed_topk=10,
                                    relaxed_delta=0.01)

    llm = LLM(
        # You can change this to a local model path if you have the model downloaded
        model=model or "nvidia/DeepSeek-R1-FP4",
        speculative_config=spec_config,
    )

    for prompt in prompts:
        response = llm.generate(prompt, SamplingParams(max_tokens=10))
        print(response.outputs[0].text)


def run_Eagle3():
    spec_config = EagleDecodingConfig(
        max_draft_len=3,
        speculative_model_dir="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        eagle3_one_model=True)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        speculative_config=spec_config,
        kv_cache_config=kv_cache_config,
    )

    for prompt in prompts:
        response = llm.generate(prompt, SamplingParams(max_tokens=10))
        print(response.outputs[0].text)


def run_ngram():
    spec_config = NGramDecodingConfig(
        max_draft_len=3,
        max_matching_ngram_size=3,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        speculative_config=spec_config,
        # ngram doesn't work with overlap_scheduler
        disable_overlap_scheduler=True,
    )

    for prompt in prompts:
        response = llm.generate(prompt, SamplingParams(max_tokens=10))
        print(response.outputs[0].text)


@click.command()
@click.argument("algo",
                type=click.Choice(["MTP", "EAGLE3", "DRAFT_TARGET", "NGRAM"]))
@click.option("--model",
              type=str,
              default=None,
              help="Path to the model or model name.")
def main(algo: str, model: Optional[str] = None):
    algo = algo.upper()
    if algo == "MTP":
        run_MTP(model)
    elif algo == "EAGLE3":
        run_Eagle3()
    elif algo == "NGRAM":
        run_ngram()
    else:
        raise ValueError(f"Invalid algorithm: {algo}")


if __name__ == "__main__":
    main()
