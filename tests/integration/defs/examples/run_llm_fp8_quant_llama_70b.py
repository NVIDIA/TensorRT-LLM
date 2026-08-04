import os
from pathlib import Path

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import QuantAlgo, QuantConfig

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model_path = Path(
    os.environ.get("LLM_MODELS_ROOT")) / "llama-models-v2/llama-v2-70b-chat-hf"
print(f'model_path: {model_path}')
print(f'gpus: {os.environ.get("CUDA_VISIBLE_DEVICES")}')


def main():

    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8,
                               kv_cache_quant_algo=QuantAlgo.FP8)

    llm = LLM(model=str(model_path),
              quant_config=quant_config,
              tensor_parallel_size=2)

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
