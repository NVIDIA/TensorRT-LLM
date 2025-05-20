from tensorrt_llm import SamplingParams
from tensorrt_llm._torch import LLM
import torch
from sentence_transformers import SentenceTransformer


def main():
    prompts = [
        "The capital of France is",
    ]
    sbert_model_path = "./my_custom_model"
    sbert = SentenceTransformer(sbert_model_path)
    sbert_embeddings = sbert.encode(prompts)
    print(sbert_embeddings)
    print(f"shape: {sbert_embeddings.shape}")

    print("=======")

    sampling_params = SamplingParams(max_tokens=32, return_context_logits=True)
    llm = LLM(model='./converted-classification-model/')
    outputs = llm.generate(prompts, sampling_params)

    tllm_logits = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        tllm_logit = output.context_logits.cpu()[0, :]
        print(f"Prompt: {prompt!r}, Context logits: {tllm_logit}")
        tllm_logits += [tllm_logit]
    # Stack the output
    tllm_logits = torch.stack(tllm_logits)
    print(tllm_logits)
    print(f"shape: {tllm_logits.shape}")


if __name__ == '__main__':
    main()
