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
"""
Adapted from examples/llama/quantize.py
"""

import random
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..._utils import str_dtype_to_torch
from ...logger import logger
from ...models.quantized.ammo import quantize_and_export


def get_calib_dataloader(data="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512,
                         cache_dir=None):
    print("Loading calibration dataset")
    if data == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train",
            cache_dir=cache_dir)
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail",
                               name="3.0.0",
                               split="train",
                               cache_dir=cache_dir)
        dataset = dataset["article"][:calib_size]
    else:
        raise NotImplementedError

    dataset_input_ids = tokenizer(dataset,
                                  return_tensors="pt",
                                  padding=True,
                                  truncation=True,
                                  max_length=block_size).input_ids.cuda()

    calib_dataloader = DataLoader(dataset_input_ids,
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def get_tokenizer(ckpt_path, **kwargs):
    logger.info(f"Loading tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path,
                                              padding_side="left",
                                              **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(ckpt_path, dtype="float16", cache_dir=None):
    logger.info(f"Loading model from {ckpt_path}")
    torch_dtype = str_dtype_to_torch(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    return model


def quantize_llama_and_export(hf_model_dir,
                              export_path,
                              qformat: str = 'fp8',
                              dtype: Optional[str] = 'float16',
                              calib_size: Optional[int] = 512,
                              hf_cache_dir: Optional[str] = None,
                              seed: Optional[int] = None,
                              quantize_lm_head=False):
    '''
        Quantize a llama model from HF model dir and save it as export_path.
        Parameters:
            hf_model_dir: huggingface model directory
            export_path: a path to save the quantized weights and scales tensors
            qformat: quantization format, currently 'int4_awq' and 'fp8' are supported
            dtype: the datatype to run the HF/pytorch model forward during quantization
            calib_size: Number of samples for calibration.
            seed: the seed to be used in the random and np.random package during quantization

        Return: None, raises exception if the quantization failed due to any reason.
    '''
    assert qformat in ['int4_awq', 'fp8'
                       ], "More quantization format supported in future release"
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    tokenizer = get_tokenizer(hf_model_dir, cache_dir=hf_cache_dir)
    model = get_model(hf_model_dir, dtype, cache_dir=hf_cache_dir)

    calib_dataloader = get_calib_dataloader(tokenizer=tokenizer,
                                            calib_size=calib_size,
                                            cache_dir=hf_cache_dir)
    quant_cfg_dict = {}
    if quantize_lm_head:
        quant_cfg_dict.update({
            "*lm_head*": {
                "enable": True
            },
        })

    model = quantize_and_export(model,
                                qformat=qformat,
                                calib_dataloader=calib_dataloader,
                                export_path=export_path,
                                quant_cfg_dict=quant_cfg_dict)
