import os
from pathlib import Path

from tensorrt_llm import LLM, SamplingParams

if __name__ == '__main__':
    prompts = [
        "Hello, my name is",
        "The president of the China is",
        "The future of chip is",
    ]
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

    model_path = Path(os.environ.get(
        "LLM_MODELS_ROOT")) / "llama-models-v2/Qwen3-Next-80B-A3B-Instruct"
    print(f'model_path: {model_path}')

    with LLM(model=str(model_path)) as llm:

        outputs = llm.generate(prompts, sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
