# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import random

import ammo.torch.quantization as atq
import numpy as np
import torch
from ammo.torch.export import export_model_config
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

RAND_SEED = 1234
MAX_SEQ_LEN = 2048


def get_calib_dataloader(data="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512):
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

    batch_encoded = tokenizer.batch_encode_plus(
        dataset,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=block_size)["input_ids"]

    calib_dataloader = DataLoader(batch_encoded,
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,
                                              model_max_length=max_seq_len,
                                              padding_side="left",
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")
    model_kwargs = {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(ckpt_path,
                                                 device_map="auto",
                                                 **model_kwargs)
    model.eval()
    model = model.to(memory_format=torch.channels_last)

    return model


def quantize_model(model, qformat, calib_dataloader=None):
    if qformat == "fp8":
        quant_cfg = atq.FP8_DEFAULT_CFG
    elif qformat == "int8_sq":
        quant_cfg = atq.INT8_SMOOTHQUANT_CFG
    else:
        raise ValueError(f"Unsupported quantization format: {qformat}")

    def calibrate_loop():
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            print(f"Calibrating batch {idx}")
            model(data)

    print("Starting quantization...")
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    print("Quantization done")
    return model


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    tokenizer = get_tokenizer(args.pyt_ckpt_path)
    model = get_model(args.pyt_ckpt_path, args.dtype, args.device)

    if "Llama" in type(model).__name__:
        model_type = "llama"
    elif "GPTJ" in type(model).__name__:
        model_type = "gptj"
    else:
        raise NotImplementedError(f"Unknown model type {type(model).__name__}")

    if args.qformat in ["fp8", "int8_sq"]:
        calib_dataloader = get_calib_dataloader(tokenizer=tokenizer,
                                                calib_size=args.calib_size)
        model = quantize_model(model, args.qformat, calib_dataloader)
    else:
        print(f"No quantization applied, export {args.dtype} model")

    with torch.inference_mode():
        export_model_config(
            model,
            model_type,
            torch.float16,
            quantization=(args.qformat),
            export_dir=args.export_path,
        )
    print(f"Quantized model exported to :{args.export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyt_ckpt_path",
                        help="Specify where the PyTorch checkpoint path is",
                        required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", help="Model data type.", default="fp16")
    parser.add_argument("--qformat", help="Quantization format.", default="fp8")
    parser.add_argument("--calib_size",
                        help="Number of samples for calibration.",
                        type=int,
                        default=512)
    parser.add_argument("--export_path", default="exported_model")
    args = parser.parse_args()

    main(args)
