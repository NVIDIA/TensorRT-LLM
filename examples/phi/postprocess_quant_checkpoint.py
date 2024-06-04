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
import json
import time

import safetensors
from safetensors.torch import save_file

import tensorrt_llm
from tensorrt_llm.models.phi3.phi3small.convert import shuffle_qkv_weights


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    tensorrt_llm.logger.set_level('info')

    tik = time.time()
    with open(f"{args.checkpoint_dir}/config.json", "r") as f:
        config = json.load(f)

    weights = {}
    with safetensors.safe_open(f"{args.checkpoint_dir}/rank0.safetensors",
                               framework="pt") as f:
        for k in f.keys():
            weights[k] = f.get_tensor(k)

    # Transform QKV weights from custom Phi3Small format to TRT-LLM format
    num_total_heads = config[
        'num_attention_heads'] + 2 * config['num_key_value_heads']
    for key, value in weights.items():
        if "qkv." in key:
            if 'scaling_factor' in key and value.shape[0] % num_total_heads != 0:
                continue
            weights[key] = shuffle_qkv_weights(value, config)

    save_file(weights, f'{args.checkpoint_dir}/rank0.safetensors')

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
