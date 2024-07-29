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
'''
Script that prunes TRT-LLM checkpoints.
'''
import argparse
import json
import os
from pathlib import Path
from typing import Dict

import safetensors
import torch
from safetensors.torch import save_file

from tensorrt_llm.logger import logger
from tensorrt_llm.models import MODEL_MAP, PretrainedConfig

SUPPORTED_MODELS = list(MODEL_MAP.keys())
PRUNABLE_WEIGHTS = [
    'attention.qkv.weight',
    'attention.proj.weight',
    'mlp.fc.weight',
    'mlp.proj.weight',
    'mlp.gate.weight',
]


def can_prune(key: str) -> bool:
    for w in PRUNABLE_WEIGHTS:
        if w in key:
            return True
    return False


def load_config(config_path: Path) -> Dict[str, any]:
    if not config_path.exists():
        return {}

    with open(str(config_path), 'r') as f:
        return json.load(f)


def prune_and_save(ckpt_dir: str, out_dir: str, prune_all: bool):
    logger.info(f'Checkpoint Dir: {ckpt_dir}, Out Dir: {out_dir}')
    model_config = PretrainedConfig.from_json_file(
        os.path.join(ckpt_dir, 'config.json'))

    architecture = model_config.architecture
    if architecture not in MODEL_MAP:
        raise RuntimeError(f'Unsupported model architecture: {architecture}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for rank in range(model_config.mapping.world_size):
        pruned_weights = {}
        with safetensors.safe_open(os.path.join(ckpt_dir,
                                                f'rank{rank}.safetensors'),
                                   framework='pt',
                                   device='cpu') as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if prune_all or can_prune(key):
                    pruned_weights[key] = torch.tensor([], dtype=tensor.dtype)
                else:
                    pruned_weights[key] = tensor

        save_file(pruned_weights,
                  os.path.join(out_dir, f'rank{rank}.safetensors'))

    config_path = Path(ckpt_dir, 'config.json')
    with open(str(Path(out_dir, 'config.json')), 'w') as f:
        config = load_config(config_path)
        config['is_pruned'] = True
        json.dump(config, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--prune_all',
                        default=False,
                        action='store_true',
                        help='Remove all weights in the checkpoint')
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help=
        'Path to write pruned checkpoint. Defaults to the same directory append with `.pruned`'
    )
    args = parser.parse_args()

    if args.checkpoint_dir is None:
        raise RuntimeError(
            "No `--checkpoint_dir` supplied to checkpoint pruner.")

    if args.out_dir is None:
        ckpt_path = Path(args.checkpoint_dir)
        ckpt_name = ckpt_path.name
        args.out_dir = str(
            Path(args.checkpoint_dir).with_name(ckpt_name + '.pruned'))

    prune_and_save(os.path.abspath(args.checkpoint_dir),
                   os.path.abspath(args.out_dir), args.prune_all)


if __name__ == '__main__':
    main()
