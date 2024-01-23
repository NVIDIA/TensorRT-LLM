#! /usr/bin/env python3
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
import argparse
import datetime
#from utils.convert import cpu_map_location
#from utils.nemo import unpack_nemo_ckpt
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.runtime.lora_manager import LoraConfig

log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
logging.basicConfig(format=log_format)
LOGGER = logging.getLogger(__name__)


def save_val(val, dir, key, tp_num=None, write_npy=False):
    ext = "npy" if write_npy else "bin"
    suffix = ext if tp_num is None else f"{tp_num}.{ext}"
    if write_npy:
        np.save(dir / f"model.{key}.{suffix}", val)
    else:
        val.tofile(dir / f"model.{key}.{suffix}")


def get_all_lora_weights(lora_weights):
    all_weights = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(
        r'.*\.layers\.([0-9]+)\.(self_attn|mlp)\.([a-z_]+)\.lora_(A|B)\.weight.*'
    )
    for key, weights in lora_weights.items():
        m = pattern.match(key)
        if not m:
            print(f"no match {key}")
            continue
        layer_idx = int(m.group(1))
        hf_module = m.group(3)
        inout = "in" if m.group(4) == "A" else "out"
        all_weights[layer_idx][hf_module][inout] = weights
    return all_weights


hf_modules_to_trtllm_modules = {
    "q_proj": "attn_q",
    "v_proj": "attn_v",
    "k_proj": "attn_k",
    "o_proj": "attn_dense",
    "gate_proj": "mlp_h_to_4h",
    "down_proj": "mlp_4h_to_h",
    "up_proj": "mlp_gate"
}  # lora modules on llama
hf_modules_to_module_id = {
    k: LoraConfig.LORA_MODULE_IDS[v]
    for k, v in hf_modules_to_trtllm_modules.items()
}


def convert_hf_model(model_dir, dtype, out_dir):
    saved_dir = Path(out_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{model_dir}/adapter_config.json", "r") as f:
        config = json.load(f)
        config["r"]
    lora_model = torch.load(f"{model_dir}/adapter_model.bin")
    all_weights = get_all_lora_weights(lora_model)
    converted_weights = []
    converted_config = []
    for layer_idx, layer_weights in all_weights.items():
        for hf_module, module_weights in layer_weights.items():
            in_weights = module_weights['in']
            out_weights = module_weights['out']
            in_out_weights = []
            adapter_size = 0
            for w, inout in ((in_weights, "in"), (out_weights, "out")):
                assert len(w.shape) == 2
                # assume that the hidden dim is the larger of the 2
                dim0 = w.shape[0]
                dim1 = w.shape[1]
                adapter_size = min(dim0, dim1)
                # in_weights should have shape [adaper_size, hidden]
                if dim1 < dim0 and inout == "in":
                    adapter_size = dim1
                    w = w.transpose(1, 0)
                # out_weights should have shape [hidden, adapter_size]
                elif dim0 < dim1 and inout == "out":
                    adapter_size = dim0
                    w = w.transpose(1, 0)

                w = w.contiguous().flatten().to(dtype=str_dtype_to_torch(dtype))
                in_out_weights.append(w)
            in_out_weights = torch.concatenate(in_out_weights).flatten()
            converted_weights.append(in_out_weights)
            converted_config.append(
                [hf_modules_to_module_id[hf_module], layer_idx, adapter_size])
    max_row_size = 0
    for t in converted_weights:
        max_row_size = max(max_row_size, t.shape[0])
    for i in range(len(converted_weights)):
        converted_weights[i] = torch.nn.functional.pad(
            converted_weights[i],
            (0, max_row_size - converted_weights[i].shape[0])).unsqueeze(0)
    converted_weights = torch.concatenate(
        converted_weights,
        dim=0).unsqueeze(0).to(dtype=str_dtype_to_torch(dtype)).cpu().numpy()
    converted_config = torch.tensor(converted_config,
                                    dtype=torch.int32,
                                    device='cpu').unsqueeze(0).numpy()

    save_val(converted_weights,
             saved_dir,
             "lora_weights",
             tp_num=None,
             write_npy=True)
    save_val(converted_config,
             saved_dir,
             "lora_config",
             tp_num=None,
             write_npy=True)


def main(args):
    start_time = datetime.datetime.now()
    convert_hf_model(args.in_file, args.storage_type, args.out_dir)

    LOGGER.info("Spent %s (h:m:s) to convert the prompt model",
                datetime.datetime.now() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-dir',
        '-o',
        type=Path,
        help='path to output embedding table file in the .npy format',
        required=True)
    parser.add_argument('--in-file',
                        '-i',
                        type=Path,
                        help='path to input lora checkpoint file',
                        required=True)
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    parser.add_argument("--storage-type",
                        type=str,
                        default="float16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    main(args)
