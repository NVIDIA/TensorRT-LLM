"""
# for llama
python simple_ray_single_node.py

# for DeepSeek
python simple_ray_single_node.py  --model_dir=/scratch/llm-models/DeepSeek-V3-Lite/bf16

"""
import argparse
from typing import List

import ray
import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig


class HFModel:

    def __init__(self, model_name: str):
        self.is_deepseek = True if "deepseek" in model_name.lower() else False
        if self.is_deepseek:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
                torch_dtype='auto',
                trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(
                "cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompts: List[str]):
        generated_texts = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt,
                                    return_tensors="pt").to(self.model.device)
            # use_cache: https://github.com/huggingface/transformers/issues/36071
            ret = self.model.generate(**inputs,
                                      max_new_tokens=50,
                                      use_cache=not self.is_deepseek)
            generated_texts.append(
                self.tokenizer.decode(ret[0], skip_special_tokens=True))
        return generated_texts


def run_llm_with_config(config: dict, prompts: List[str]):
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)
    config["kv_cache_config"] = kv_cache_config

    print(f"Running LLM with config: {config}")
    llm = LLM(**config)
    llm_ret = llm.generate(prompts)
    del llm
    ray.shutdown()
    outputs = []
    for index, r in enumerate(llm_ret):
        outputs.append(prompts[index] + " " + r.outputs[0].text)
    return outputs


def run_hf_model(model_name: str, prompts: List[str]):
    hf_model = HFModel(model_name)
    hf_ret = hf_model.generate(prompts)
    del hf_model
    torch.cuda.empty_cache()
    return hf_ret


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LLM models with the PyTorch workflow.")

    parser.add_argument('--model_dir',
                        type=str,
                        required=False,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model checkpoint directory.")
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    parser.add_argument('--moe_tp_size', type=int, default=-1)

    args = parser.parse_args()
    return args


TEST_CONFIGS = {
    "llama": [
        {
            "name": "llm_tp2",
            "tensor_parallel_size": 2
        },
        {
            "name": "llm_pp2",
            "pipeline_parallel_size": 2
        },
        {
            "name": "llm_tp2pp2",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2
        },
    ],
    "deepseek": [
        {
            "name": "llm_tp2",
            "tensor_parallel_size": 2
        },
        {
            "name": "llm_tp4ep4",
            "tensor_parallel_size": 4,
            "moe_expert_parallel_size": 4
        },
    ]
}

if __name__ == "__main__":
    args = parse_arguments()
    model_name = args.model_dir

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    if "llama" in model_name.lower() or 'qwen' in model_name.lower():
        test_configs = TEST_CONFIGS["llama"]
    elif "deepseek" in model_name.lower():
        test_configs = TEST_CONFIGS["deepseek"]
    else:
        # TODO: change to use args
        raise ValueError(
            f"Manually add your parallel config for {model_name} in script for now."
        )

    print("Running HF reference")
    generated_text = {}
    generated_text["hf"] = run_hf_model(model_name, prompts)

    gpu_count = torch.cuda.device_count()
    for base_config in test_configs:
        tp = base_config.get("tensor_parallel_size", 1)
        pp = base_config.get("pipeline_parallel_size", 1)
        ep = base_config.get("moe_expert_parallel_size", None)
        required_gpus = tp if ep is not None else tp * pp

        if gpu_count < required_gpus:
            print(
                f"Skipping {base_config['name']} as it needs {required_gpus} GPUs but you only have {gpu_count}."
            )
            continue

        config = base_config.copy()
        config["model"] = model_name
        config["executor_type"] = "ray"
        del config["name"]
        generated_text[base_config["name"]] = run_llm_with_config(
            config, prompts)

    for index, prompt in enumerate(prompts):
        table = [[k] for k in generated_text.keys()]
        header = [f'Prompt {index + 1}', prompt]
        for i, value in enumerate(generated_text.values()):
            table[i].append(value[index])
        print(
            tabulate(table,
                     headers=header,
                     tablefmt="grid",
                     maxcolwidths=[None, 128]))
