# Refer to https://github.com/NVIDIA/TensorRT-LLM/blob/main/tests/unittest/_torch/modeling/test_modeling_deepseek.py#L82
import json
import logging
import os
import shutil
from pathlib import Path
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hf_gen_dir = "/tmp/hf_gen"


class HFModel:

    def __init__(self, model_name: str):
        self.default_device = torch.get_default_device()
        torch.set_default_device("cuda")
        config = AutoConfig.from_pretrained(model_name)
        torch.manual_seed(13)
        self.model = AutoModelForCausalLM.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.model.save_pretrained(hf_gen_dir)
        self.copy_safetensors(hf_gen_dir, model_name)

    def copy_safetensors(self, src_dir, dest_dir):
        for file in os.listdir(src_dir):
            if file.endswith(
                    ".safetensors") or file == 'model.safetensors.index.json':
                shutil.copy2(os.path.join(src_dir, file),
                             os.path.join(dest_dir, file))

    def generate(self, prompts: List[str]):
        generated_texts = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            ret = self.model.generate(**inputs, max_new_tokens=50)
            generated_texts.append(
                self.tokenizer.decode(ret[0], skip_special_tokens=True))
        return generated_texts

    def __del__(self):
        print(f"del hf model, restore default device to {self.default_device}")
        torch.set_default_device(self.default_device)


def process_and_copy_folder(src_folder,
                            dst_folder,
                            exclude_mpt_from_quant=False):
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)
    os.makedirs(dst_folder)

    for root, dirs, files in os.walk(src_folder):
        rel_path = os.path.relpath(root, src_folder)
        dest_dir = os.path.join(dst_folder, rel_path)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        num_hidden_layers = 4
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, file)
            if 'safetensor' in file:
                continue

            if file == 'config.json':
                with open(src_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                config['num_hidden_layers'] = num_hidden_layers
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            elif file == 'hf_quant_config.json':
                with open(src_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if exclude_mpt_from_quant:
                        config['exclude_modules'] = ['model.layers.4*']
                with open(dest_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                shutil.copy2(src_path, dest_path)


def prepare_model(model_name):
    llm_models_root = '/home/scratch.trt_llm_data/llm-models'
    model_dir = str(llm_models_root / Path(f"DeepSeek-R1/{model_name}"))
    assert Path(model_dir).exists()
    tmp_model_dir = f"/tmp/{model_name}"
    exclude_mpt_from_quant = model_name == "DeepSeek-R1-FP4"
    process_and_copy_folder(model_dir, tmp_model_dir, exclude_mpt_from_quant)
    return tmp_model_dir


def test_deepseek(model_dir, prompts):
    spec_config = MTPDecodingConfig(num_nextn_predict_layers=1)

    pytorch_config = dict(
        disable_overlap_scheduler=True,
        kv_cache_dtype="auto",
        load_format="auto",
    )

    llm = LLM(model=model_dir,
              tensor_parallel_size=4,
              enable_chunked_prefill=False,
              **pytorch_config,
              moe_expert_parallel_size=4,
              moe_tensor_parallel_size=-1,
              enable_attention_dp=False,
              speculative_config=spec_config,
              executor_type="ray",
              kv_cache_config=KvCacheConfig(enable_block_reuse=False,
                                            free_gpu_memory_fraction=0.4))

    sampling_params = SamplingParams(max_tokens=20)

    try:
        output = llm.generate(prompts, sampling_params)
        for o in output:
            print(o.outputs[0].text)
    except Exception as e:
        raise e


def run_hf_model(model_dir, prompts):
    hf_model = HFModel(model_dir)
    hf_output = hf_model.generate(prompts)
    for o in hf_output:
        print(o)


if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    model_dir = prepare_model("DeepSeek-R1")
    run_hf_model(model_dir, ["The president of the United States is"])
    test_deepseek(model_dir, prompts)
