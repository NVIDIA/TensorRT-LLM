### Generate Text in Streaming

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM


def main():

    # Set your own model path here.
    model = "microsoft/Phi-4-mini-instruct"

    llm_api_args = {
        "model": model,
        "tokenizer": model,
        "tokenizer_mode": "auto",
        "skip_tokenizer_init": False,
        "trust_remote_code": True,
        # Below are quite important for TP2.
        "tensor_parallel_size": 2,
        "embedding_parallel_mode": "None",
    }

    llm = LLM(**llm_api_args)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.01, n=1, max_tokens=32)

    # Async based on Python coroutines
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()