import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.llmapi.utils import print_colored

# isort: off
sys.path.append(os.path.join(cur_dir, '..'))
from utils.llm_data import llm_models_root
# isort: on

model_path = llm_models_root() / "llama-models-v2" / "TinyLlama-1.1B-Chat-v1.0"


def run_llm_tp2():
    with LLM(model=model_path, tensor_parallel_size=2) as llm:
        sampling_params = SamplingParams(max_tokens=10, end_id=-1)
        for output in llm.generate(["Hello, my name is"], sampling_params):
            print(output)


def run_multi_llm_tasks():
    for i in range(3):
        print_colored(f"Running LLM task {i}\n", "green")
        run_llm_tp2()
        print_colored(f"LLM task {i} completed\n", "green")


if __name__ == "__main__":
    run_multi_llm_tasks()
