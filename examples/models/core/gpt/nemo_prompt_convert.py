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
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.models.gpt.convert import cpu_map_location, unpack_nemo_ckpt

log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
logging.basicConfig(format=log_format)
LOGGER = logging.getLogger(__name__)


def prompt_convert(out_file, prompt_config, prompt_weights):
    nemo_type = "peft_tuning" if "peft" in prompt_config else "prompt_learning"

    vtokens_embeddings = []
    vtokens_len = []

    if nemo_type == "peft_tuning":
        ptuning_key = "model.embedding.adapter_layer.ptuning_adapter.inference_table"
        if ptuning_key not in prompt_weights:
            key_match = "adapter_layer.ptuning_adapter"
            for k in prompt_weights.keys():
                if key_match in k:
                    ptuning_key = k
                    break
            else:
                raise ValueError(
                    "Could not find a suitable ptuning key in Nemo dict."
                    f" Tried {ptuning_key} or any key matching *{key_match}*")
        prompt_task_weights = prompt_weights[ptuning_key]

        if 'hidden_size' in prompt_config:
            assert prompt_config['hidden_size'] == prompt_task_weights.shape[
                1], "P-Tuning hidden size does not match the model's."

        vtokens_embeddings.append(prompt_task_weights)
        vtokens_len.append(prompt_task_weights.shape[0])
    else:
        prompt_templates = prompt_config["task_templates"]
        actual_task_id = 0
        for task_name_id, prompt_task in enumerate(prompt_templates):
            prompt_task_name = prompt_task["taskname"]
            LOGGER.info(f"Task {actual_task_id}: {prompt_task['taskname']}")
            prompt_task_weights = prompt_weights["prompt_table"].get(
                f"prompt_table.{prompt_task_name}.prompt_embeddings.weight")
            if prompt_task_weights is None:
                continue
            vtokens_embeddings.append(prompt_task_weights)
            vtokens_len.append(prompt_task_weights.shape[0])
            actual_task_id += 1

    max_vtoken_len = max(vtokens_len)
    embedding_dim = vtokens_embeddings[0].shape[1]

    # pad tasks to longest task embedding table
    for i, vtoken_emb_table in enumerate(vtokens_embeddings):
        padded_table = torch.zeros((max_vtoken_len, embedding_dim))
        padded_table[:vtoken_emb_table.shape[0], :] = vtoken_emb_table
        vtokens_embeddings[i] = padded_table

    vtokens_embeddings = torch.stack(vtokens_embeddings)
    np.save(out_file, torch_to_numpy(vtokens_embeddings))


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
        if not weight_path.exists():
            weight_path = prompt_out_dir / "mp_rank_00" / model_weights_ckpt

        prompt_weights = torch.load(
            weight_path,
            map_location=cpu_map_location,
        )
    prompt_convert(args.out_file, prompt_config, prompt_weights)

    LOGGER.info("Spent %s (h:m:s) to convert the prompt model",
                datetime.datetime.now() - start_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-file',
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
                        choices=["fp32", "fp16"])
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    args = parser.parse_args()

    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    print("\n=============== Argument ===============")
    for key in vars(args):
        print(f"{key}: {vars(args)[key]}")
    print("========================================")

    main(args)
