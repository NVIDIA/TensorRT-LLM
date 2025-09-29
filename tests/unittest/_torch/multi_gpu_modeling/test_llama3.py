from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import LLM


def test_llama_3_3():
    model_dir = llm_models_root(
    ) / "llama-3.3-models" / "Llama-3.3-70B-Instruct-FP8"
    tp = 2
    pp = 2

    llm = LLM(model_dir, tensor_parallel_size=tp, pipeline_parallel_size=pp)
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
