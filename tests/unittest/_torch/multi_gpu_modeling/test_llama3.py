from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig


def test_llama_3_3():
    model_dir = llm_models_root(
    ) / "llama-3.3-models" / "Llama-3.3-70B-Instruct"
    tp = 2
    pp = 2

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.5, )
    llm = LLM(model_dir,
              tensor_parallel_size=tp,
              pipeline_parallel_size=pp,
              kv_cache_config=kv_cache_config)
    prompts = [
        "The capital of France is",
        "The president of the United States is",
    ]

    outputs = llm.generate(prompts)

    expected_outputs = [
        " a city of romance, art, fashion, and cuisine. Paris, also known as the City of Light, is a must-visit destination for anyone interested in",
        " the head of state and head of government of the United States. The president is also the commander-in-chief of the armed forces. The president is elected by the",
    ]
    for i, output in enumerate(outputs):
        assert similar(output.outputs[0].text, expected_outputs[i])
