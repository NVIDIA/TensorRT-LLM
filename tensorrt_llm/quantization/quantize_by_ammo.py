# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from examples/quantization/hf_ptq.py
"""

import copy
import json
import random
import sys
import time

import numpy as np
import safetensors
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

EMPTY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "enable": False,
        },
        "*input_quantizer": {
            "enable": False
        },
        "*lm_head*": {
            "enable": False
        },
        "*output_layer*": {
            "enable": False
        },
        "default": {
            "enable": False
        },
    },
    "algorithm": "max",
}

KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.Wqkv.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.W_pack.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.c_attn.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
}


def quant_cfg_choices():
    import ammo.torch.quantization as atq
    QUANT_CFG_CHOICES = {
        "int8_sq": atq.INT8_SMOOTHQUANT_CFG,
        "fp8": atq.FP8_DEFAULT_CFG,
        "int4_awq": atq.INT4_AWQ_CFG,
        "w4a8_awq": atq.W4A8_AWQ_BETA_CFG,
        "int8_wo": EMPTY_CFG,
        "int4_wo": EMPTY_CFG,
        "full_prec": EMPTY_CFG,
    }
    return QUANT_CFG_CHOICES


MODEL_NAME_PATTERN_MAP = {
    "GPT2": "gpt2",
    "Xverse": "llama",
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "QWen": "qwen",
    "Gemma": "gemma",
}


def get_tokenizer(ckpt_path, max_seq_length, model_type=None):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_length,
        padding_side="left",
        trust_remote_code=True,
    )
    if model_type and model_type == "qwen":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16" or dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "fp16" or dtype == "float16":
        dtype = torch.float16
    elif dtype == "fp32" or dtype == "float32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    # Note: VILA model is not in public HF model zoo yet. We need to explicitly import from the git repo
    if "vila" in ckpt_path:
        sys.path.append(args.model_dir + "/../VILA")
        from llava.model import LlavaConfig, LlavaLlamaForCausalLM
        AutoConfig.register("llava_llama", LlavaConfig)
        AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

    model_kwargs = {"torch_dtype": "auto"}
    model = AutoModelForCausalLM.from_pretrained(ckpt_path,
                                                 device_map="auto",
                                                 **model_kwargs,
                                                 trust_remote_code=True)
    model.eval()

    model_dtype = next(model.parameters()).dtype
    if dtype != model_dtype:
        print(
            f"[TensorRT-LLM][WARNING] The manually set model data type is {dtype}, "
            f"but the data type of the HuggingFace model is {model_dtype}.")

    return model


def get_model_type(model):
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_calib_dataloader(data="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512,
                         device=None):
    print("Loading calibration dataset")
    if data == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train")
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=block_size)
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded,
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def quantize_model(model, quant_cfg, calib_dataloader=None):
    import ammo.torch.quantization as atq

    def calibrate_loop():
        if calib_dataloader is None:
            return
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            print(f"Calibrating batch {idx}")
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print("Quantization done. Total time used: {:.2f} s.".format(end_time -
                                                                 start_time))

    return model


def quantize_and_export(*, model_dir, dtype, device, qformat, kv_cache_dtype,
                        calib_size, batch_size, awq_block_size, output_dir,
                        tp_size, pp_size, seed, max_seq_length):
    '''
        Load model from the model_dir, call AMMO to quantize the model, and then export
        the quantized model as TRT-LLM checkpoint
    '''
    from ammo.torch.export import export_model_config
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(seed)
    np.random.seed(seed)

    model = get_model(model_dir, dtype, device)
    model_type = get_model_type(model)
    tokenizer = get_tokenizer(model_dir,
                              max_seq_length=max_seq_length,
                              model_type=model_type)

    if qformat in ["full_prec", "int8_wo", "int4_wo"
                   ] and kv_cache_dtype is None:
        print(f"No quantization applied, export {dtype} model")
    else:
        if "awq" in qformat:
            if calib_size > 32:
                print(
                    f"AWQ calibration could take longer with calib_size = {calib_size}, Using"
                    " calib_size=32 instead")
                calib_size = 32
            print(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument --batch_size <batch_size> to the command line.\n"
            )

        calib_dataloader = get_calib_dataloader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            calib_size=calib_size,
            device=device,
        )

        if qformat in quant_cfg_choices():
            quant_cfg = quant_cfg_choices()[qformat]
        else:
            raise ValueError(f"Unsupported quantization format: {qformat}")

        if "awq" in qformat:
            quant_cfg = copy.deepcopy(quant_cfg_choices()[qformat])
            weight_quantizer = quant_cfg["quant_cfg"][
                "*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = awq_block_size

        if kv_cache_dtype is not None:
            if kv_cache_dtype == "fp8":
                for value in KV_CACHE_CFG.values():
                    value.update({"num_bits": (4, 3)})  # type: ignore
            quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

        print(quant_cfg)

        model = quantize_model(model, quant_cfg, calib_dataloader)

    with torch.inference_mode():
        if model_type is None:
            print(
                f"Unknown model type {type(model).__name__}. Continue exporting..."
            )
            model_type = f"unknown:{type(model).__name__}"

        export_path = output_dir
        start_time = time.time()

        if qformat == "int4_awq" and model_type == "qwen":
            torch.save(model.state_dict(), export_path)
        else:
            export_npz = (model_type not in [
                'gptj', 'falcon', 'chatglm', 'mpt', 'llama', 'baichuan', 'gemma'
            ])
            export_model_config(model,
                                model_type,
                                getattr(torch, dtype),
                                export_dir=export_path,
                                inference_tensor_parallel=tp_size,
                                inference_pipeline_parallel=pp_size,
                                export_tensorrt_llm_config=(not export_npz),
                                export_npz=export_npz)

            # Workaround for wo quantization
            if qformat in ["int8_wo", "int4_wo", "full_prec"]:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                if qformat == "int8_wo":
                    tensorrt_llm_config["quantization"]["quant_algo"] = 'W8A16'
                elif qformat == "int4_wo":
                    tensorrt_llm_config["quantization"]["quant_algo"] = 'W4A16'
                else:
                    tensorrt_llm_config["quantization"]["quant_algo"] = None
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Workaround for share_embedding_table
            if pp_size == 1:
                with safetensors.safe_open(f"{export_path}/rank0.safetensors",
                                           framework='pt',
                                           device='cpu') as f:
                    share_embedding_table = 'lm_head.weight' not in f.keys()
                if share_embedding_table:
                    with open(f"{export_path}/config.json", "r") as f:
                        tensorrt_llm_config = json.load(f)
                    tensorrt_llm_config["share_embedding_table"] = True
                    with open(f"{export_path}/config.json", "w") as f:
                        json.dump(tensorrt_llm_config, f, indent=4)

        end_time = time.time()
        print(
            "Quantized model exported to {} \nTotal time used {:.2f} s.".format(
                export_path, end_time - start_time))
