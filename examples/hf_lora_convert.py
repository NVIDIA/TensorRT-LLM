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
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.lora_manager import LoraManager
from tensorrt_llm.models.convert_utils import get_model_path, load_state_dict

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
        r'(.*\.layers\.([0-9]+)\.(self_attn|mlp)\.([a-z_]+))\.(?:lora_(?:(A|B)\.weight|(magnitude)_vector)|weight_(m_wdecomp).weight).*'
    )
    moe_pattern = re.compile(
        r'(.*\.layers\.([0-9]+)\.(block_sparse_moe)\.((experts)\.([0-9]+)\.|)([a-zA-Z0-9_]+))\.(?:lora_(?:(A|B)\.weight|(magnitude)_vector)|weight_(m_wdecomp).weight).*'
    )
    for key, weights in lora_weights.items():
        m = pattern.match(key)
        m_moe = moe_pattern.match(key)
        if m:
            layer_idx = int(m.group(2))
            hf_module = m.group(4)
            inout = m.group(5)
            dora_magnitude = m.group(6) or m.group(7)

            if inout:
                inout = "in" if inout == "A" else "out"
                all_weights[layer_idx][hf_module][inout] = weights
            elif dora_magnitude:
                LOGGER.warning(
                    "Detected DoRA magnitude vector, make sure it was preprocessed and normalized using the proper base model weights"
                )
                all_weights[layer_idx][hf_module]["magnitude"] = weights.view(
                    -1)

        elif m_moe:
            layer_idx = int(m_moe.group(2))
            hf_module = m_moe.group(7)
            inout = m_moe.group(8)
            dora_magnitude = m_moe.group(9) or m.group(10)

            if inout:
                inout = "in" if inout == "A" else "out"
                all_weights[layer_idx][hf_module][inout] = weights
            elif dora_magnitude:
                LOGGER.warning(
                    "Detected DoRA magnitude vector, make sure it was preprocessed and normalized using the proper base model weights"
                )
                all_weights[layer_idx][hf_module]["magnitude"] = weights.view(
                    -1)
        else:
            print(f"no match {key}")
            continue
    return all_weights


def preprocess_lora_weights(lora_model):
    # Swap weights of gate_up_proj
    for key, value in lora_model.items():
        if "gate_up_proj.lora_B.weight" in key:
            print("Swap {}".format(key))
            original_weights = value.contiguous().clone()
            half_split = original_weights.shape[0] // 2
            first_half = original_weights[:half_split, :]
            second_half = original_weights[half_split:, :]
            value = torch.cat((second_half, first_half), dim=0)
            lora_model[key] = value
    return lora_model


hf_modules_to_trtllm_modules = {
    "q_proj": "attn_q",
    "v_proj": "attn_v",
    "k_proj": "attn_k",
    "qkv_proj": "attn_qkv",
    "query_key_value": "attn_qkv",
    "o_proj": "attn_dense",
    "dense": "attn_dense",
    "gate_proj": "mlp_h_to_4h",
    "down_proj": "mlp_4h_to_h",
    "up_proj": "mlp_gate",
    "gate_up_proj": "mlp_h_to_4h",
    "c_fc": "mlp_h_to_4h",
    "c_proj": "mlp_4h_to_h",
    "w1": "moe_h_to_4h",
    "w2": "moe_4h_to_h",
    "w3": "moe_gate",
    "gate": "moe_router",
}  # lora modules on llama
hf_modules_to_module_id = {
    k: LoraManager.LORA_MODULE_IDS[v]
    for k, v in hf_modules_to_trtllm_modules.items()
}


def convert_hf_model(model_dir, dtype, out_dir):
    saved_dir = Path(out_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{model_dir}/adapter_config.json", "r") as f:
        config = json.load(f)

    alpha = config.get("lora_alpha")
    use_rslora = config.get("use_rslora", False)

    lora_model = load_state_dict(get_model_path(model_dir, "adapter_model"))
    lora_model = preprocess_lora_weights(lora_model)
    all_weights = get_all_lora_weights(lora_model)
    converted_weights = []
    converted_config = []

    def derive_adapter_size(inout_weight: torch.Tensor) -> int:
        assert len(inout_weight.shape) == 2
        dim0, dim1 = inout_weight.shape
        # assume the hidden dim is the larger of the 2
        adapter_size = min(dim0, dim1)
        return adapter_size

    def derive_weights_scale(adapter_size: int, alpha: float,
                             use_rslora: bool) -> float:
        if use_rslora:
            return alpha / np.sqrt(adapter_size)
        return alpha / adapter_size

    for layer_idx, layer_weights in all_weights.items():
        for hf_module, module_weights in layer_weights.items():
            in_weights = module_weights['in']
            out_weights = module_weights['out']
            magnitude = module_weights.get("magnitude", None)
            is_dora = magnitude is not None

            processed_weights = []

            assert len(in_weights.shape) == 2
            assert len(out_weights.shape) == 2
            assert not is_dora or len(magnitude.shape) == 1

            adapter_size = derive_adapter_size(in_weights)
            assert adapter_size == derive_adapter_size(
                out_weights), "adapter size of A mismatches adapter size of B"
            scale = derive_weights_scale(adapter_size, alpha, use_rslora)

            for w, inout in ((in_weights, "in"), (out_weights, "out")):
                dim0 = w.shape[0]
                dim1 = w.shape[1]
                # in_weights should have shape [adaper_size, hidden]
                if dim1 < dim0 and inout == "in":
                    w = w.transpose(1, 0)
                # out_weights should have shape [hidden, adapter_size]
                elif dim0 < dim1 and inout == "out":
                    w = w.transpose(1, 0)
                if inout == "out":
                    w = w * scale
                w = w.contiguous().flatten().to(dtype=str_dtype_to_torch(dtype))
                processed_weights.append(w)

            if is_dora:
                processed_weights.append(magnitude.contiguous().flatten().to(
                    dtype=str_dtype_to_torch(dtype)))

            processed_weights = torch.concatenate(processed_weights).flatten()
            converted_weights.append(processed_weights)
            converted_config.append([
                hf_modules_to_module_id[hf_module], layer_idx, adapter_size,
                1 if is_dora else 0
            ])
    max_row_size = 0
    for t in converted_weights:
        max_row_size = max(max_row_size, t.shape[0])
    for i in range(len(converted_weights)):
        converted_weights[i] = torch.nn.functional.pad(
            converted_weights[i],
            (0, max_row_size - converted_weights[i].shape[0])).unsqueeze(0)
    converted_weights = torch_to_numpy(
        torch.concatenate(
            converted_weights,
            dim=0).unsqueeze(0).to(dtype=str_dtype_to_torch(dtype)).cpu())
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
