import os
from pathlib import Path

from tensorrt_llm import LLM, SamplingParams

if __name__ == '__main__':
    prompts = [
        "Hello, my name is",
        "The president of the China is",
        "The future of chip is",
    ]
    sampling_params = SamplingParams(temperature=1.0, top_p=0.95)

    models_root = os.environ.get("LLM_MODELS_ROOT")
    if not models_root:
        raise RuntimeError(
            "LLM_MODELS_ROOT is not set; please export it to the model root directory."
        )
    model_path = Path(models_root) / "llama-models-v2/llama-v2-70b-chat-hf"
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    print(f'model_path: {model_path}')

    with LLM(model=str(model_path)) as llm:

        outputs = llm.generate(prompts, sampling_params)

        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
