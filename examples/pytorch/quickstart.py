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

    llm = LLM(model='./converted-classification-model/')
    outputs = llm.generate(prompts, sampling_params)

    tllm_logits = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        tllm_logit = output.context_logits.cpu()
        print(f"Prompt: {prompt!r}, Context logits: {tllm_logit}")
        tllm_logits += [tllm_logit]
    # Stack the output
    tllm_logits = torch.stack(tllm_logits)
    print(tllm_logits)
    print(f"shape: {tllm_logits.shape}")


if __name__ == '__main__':
    main()
