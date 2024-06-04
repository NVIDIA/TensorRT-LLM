from tensorrt_llm.hlapi import LLM, ModelConfig
from tensorrt_llm.hlapi.utils import download_hf_model

prompts = ["A B C"]


def test_download_hf_model():
    dir = download_hf_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    assert dir.exists()
    print(f"Downloaded model to {dir}")


def test_llm_with_model_downloaded():
    config = ModelConfig(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    llm = LLM(config)
    for output in llm.generate(prompts):
        print(output)
