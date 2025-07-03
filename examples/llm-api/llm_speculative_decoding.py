import click

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import MTPDecodingConfig

prompts = [
    "What is the capital of France?",
    "What is the future of AI?",
]


def run_MTP():
    spec_config = MTPDecodingConfig(num_nextn_predict_layers=1,
                                    use_relaxed_acceptance_for_thinking=True,
                                    relaxed_topk=10,
                                    relaxed_delta=0.01)

    llm = LLM(
        model_dir="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        speculative_config=spec_config,
    )

    for prompt in prompts:
        response = llm.generate(prompt, SamplingParams(max_tokens=10))
        print(response.text)


def run_Eagle3():
    spec_config = EagleDecodingConfig(
        max_draft_len=3,
        pytorch_weights_path="models/eagle3-8b-instruct",
        eagle3_one_model=True)

    llm = LLM(
        model_dir="models/llama-3.1-8b-instruct",
        speculative_config=spec_config,
    )

    for prompt in prompts:
        response = llm.generate(prompt, SamplingParams(max_tokens=10))
        print(response.text)


def run_draft_target():
    spec_config = DraftTargetDecodingConfig(
        max_draft_len=3,
        pytorch_weights_path="models/draft-target-8b-instruct",
    )

    llm = LLM(
        model_dir="models/llama-3.1-8b-instruct",
        speculative_config=spec_config,
    )

    for prompt in prompts:
        response = llm.generate(prompt, SamplingParams(max_tokens=10))
        print(response.text)


def run_ngram():
    spec_config = NGramDecodingConfig(
        prompt_lookup_num_tokens=3,
        max_matching_ngram_size=3,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=True,
    )

    llm = LLM(
        model_dir="models/llama-3.1-8b-instruct",
        speculative_config=spec_config,
    )

    for prompt in prompts:
        response = llm.generate(prompt, SamplingParams(max_tokens=10))
        print(response.text)


@click.command()
@click.argument("algo",
                type=click.Choice(["MTP", "EAGLE3", "DRAFT_TARGET", "NGRAM"]))
def main(algo: str):
    algo = algo.upper()
    if algo == "MTP":
        run_MTP()
    elif algo == "EAGLE3":
        run_Eagle3()
    elif algo == "DRAFT_TARGET":
        run_draft_target()
    elif algo == "NGRAM":
        run_ngram()
    else:
        raise ValueError(f"Invalid algorithm: {algo}")


if __name__ == "__main__":
    main()
