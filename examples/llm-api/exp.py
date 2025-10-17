from tensorrt_llm import LLM, SamplingParams


def main():

    # Llama-3.2-1B-FP8
    llm = LLM(model="/llm-models/llama-3.2-models/Llama-3.2-1B-FP8", cuda_graph_config=None)

    prompt = "Hello, my name is"

    sampling_params = SamplingParams(max_tokens=3)

    output = llm.generate(prompt, sampling_params)
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")

if __name__ == '__main__':
    main()