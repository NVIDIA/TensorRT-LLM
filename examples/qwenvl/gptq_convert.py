import argparse
import logging

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="Qwen-VL-Chat",
    )
    parser.add_argument(
        "--quantized_model_dir",
        type=str,
        default="Qwen-VL-Chat-4bit",
    )

    args = parser.parse_args()
    return args


args = parse_arguments()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")

tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_dir,
                                          use_fast=True,
                                          trust_remote_code=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.",
        return_tensors="pt").to(device)
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=
    False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(args.pretrained_model_dir,
                                            quantize_config,
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=True,
                                            device_map=device,
                                            fp16=True)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model using safetensors
model.save_quantized(args.quantized_model_dir, use_safetensors=True)
