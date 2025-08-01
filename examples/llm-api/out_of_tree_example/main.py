import modeling_opt  # noqa

from tensorrt_llm import LLM


def main():
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    llm = LLM(model='facebook/opt-125m')
    outputs = llm.generate(prompts)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
