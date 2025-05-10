from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM

import torch


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=32, return_context_logits=True)

    model_path = "/code/tensorrt_llm/custom_bert_classifier"
    llm = LLM(model=model_path)
    outputs = llm.generate(prompts, sampling_params)

    tllm_logits = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        logits = output.context_logits.cpu()
        print(f"[{i}] Prompt: {prompt!r}, logits: {logits}")
        tllm_logits += [logits]

    # stack logits
    tllm_logits = torch.stack(tllm_logits)
    print(f"tllm_logits: {tllm_logits}")



if __name__ == '__main__':
    main()
