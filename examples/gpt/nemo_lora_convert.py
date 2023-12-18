#! /usr/bin/env python3
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
import datetime
import logging
import tempfile
from pathlib import Path

import torch
import yaml
from utils.convert import cpu_map_location
from utils.nemo import unpack_nemo_ckpt

from tensorrt_llm._utils import to_json_file, torch_to_numpy

log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
logging.basicConfig(format=log_format)
LOGGER = logging.getLogger(__name__)


def get_lora_keys(layer_id):
    in_key = f'model.language_model.encoder.layers.{layer_id}.self_attention.adapter_layer.lora_kqv_adapter.linear_in.weight'
    out_key = f'model.language_model.encoder.layers.{layer_id}.self_attention.adapter_layer.lora_kqv_adapter.linear_out.weight'
    return in_key, out_key


def save_val(val, dir, key, tp_num=None):
    suffix = "bin" if tp_num is None else f"{tp_num}.bin"
    val.tofile(dir / f"model.{key}.{suffix}")


def lora_convert(out_dir, lora_config, lora_weights, customization_id):
    saved_dir = Path(out_dir)
    saved_dir.mkdir(parents=True, exist_ok=True)
    num_layers = int(lora_config["num_layers"])
    config = {"lora_config": {"lora_kqv_adapter": {}}}
    for layer_id in range(num_layers):
        in_key, out_key = get_lora_keys(layer_id)
        config["lora_config"]["lora_kqv_adapter"]["0"] = {
            "key": f"{customization_id}",
            "low_rank": f"{lora_weights[in_key].shape[0]}",
        }

        linear_in_weight = lora_weights[in_key]
        linear_out_weight = lora_weights[out_key]

        save_val(torch_to_numpy(linear_in_weight.transpose(1, 0).contiguous()),
                 saved_dir,
                 in_key.replace("lora_kqv_adapter", f"lora_kqv_adapter.{0}"))
        save_val(torch_to_numpy(linear_out_weight.transpose(1, 0).contiguous()),
                 saved_dir,
                 out_key.replace("lora_kqv_adapter", f"lora_kqv_adapter.{0}"))

        to_json_file(config, saved_dir / "lora_weights.json")


def main(args):
    start_time = datetime.datetime.now()
    with tempfile.TemporaryDirectory() as prompt_out_dir:
        prompt_out_dir = Path(prompt_out_dir)
        unpack_nemo_ckpt(args.in_file, prompt_out_dir)
        LOGGER.info("Spent %s (h:m:s) to unpack NeMo prompt archive",
                    datetime.datetime.now() - start_time)

        model_weights_ckpt = "model_weights.ckpt"
        with open(prompt_out_dir / "model_config.yaml") as f:
            prompt_config = yaml.full_load(f)
        LOGGER.debug(prompt_config)

        start_time = datetime.datetime.now()
        weight_path = prompt_out_dir / model_weights_ckpt

        prompt_weights = torch.load(
            weight_path,
            map_location=cpu_map_location,
        )
    lora_convert(args.out_dir, prompt_config, prompt_weights,
                 args.customization_id)

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
                        help='path to input prompt-tuning checkpoint file',
                        required=True)
    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    parser.add_argument("--customization-id", type=str, default="lora")
    args = parser.parse_args()

    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    main(args)
