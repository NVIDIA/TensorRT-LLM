from tensorrt_llm import LLM, SamplingParams


def main():
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        gather_generation_logits=True  # Required. TODO: Acutal API TBD.
    )

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Current behavior:
    # - With return_generation_logits=True: Returns ONLY the sampled token's logprob
    # - Without return_generation_logits=True: Returns top-K tokens (sampled token NOT guaranteed)
    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=0.7,
        top_p=0.95,
        logprobs=1,
        return_generation_logits=True,
    )

    for output in llm.generate(prompts, sampling_params):
        print(f"\n{'='*80}")
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print(f"Generated token IDs: {output.outputs[0].token_ids}")

        if output.outputs[0].logprobs:
            print(f"\nLogprobs for each generated token:")
            for i, (token_id, token_logprobs) in enumerate(
                zip(output.outputs[0].token_ids, output.outputs[0].logprobs)
            ):
                print(f"\n  Token {i}: ID={token_id}, Text={llm.tokenizer.decode([token_id])!r}")

                # TODO. move to proper test
                assert len(token_logprobs) == 1
                assert token_id in token_logprobs, f"Sampled token {token_id} not in logprobs dict."

                for tid, logprob_obj in token_logprobs.items():
                    token_text = llm.tokenizer.decode([tid])
                    is_sampled = "← SAMPLED" if tid == token_id else ""
                    print(f"    • Token {tid:5d} ({token_text:15s}): "
                          f"logprob={logprob_obj.logprob:8.4f}, "
                          f"rank={logprob_obj.rank} {is_sampled}")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
