#!/usr/bin/env python3
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
from pathlib import Path


def generate_dataset(num_samples, output_filename, long_prompt=False):
    resources_dir = Path(__file__).parent.resolve().parent
    data_dir = resources_dir / 'data'
    dummy_cnn_dataset = data_dir / output_filename

    input = ' '.join(['test' for _ in range(10)])
    output = ' '.join(['test' for _ in range(10)])

    instruction = "Summarize the following news article:"
    if long_prompt:
        instruction = (
            "TensorRT-LLM provides users with an easy-to-use Python "
            "API to define Large Language Models (LLMs) and build TensorRT engines "
            "that contain state-of-the-art optimizations to perform inference "
            "efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to "
            "create Python and C++ runtimes that execute those TensorRT engines. "
            "It also includes a backend for integration with the NVIDIA Triton Inference "
            "Server; a production-quality system to serve LLMs. Models built with "
            "TensorRT-LLM can be executed on a wide range of configurations going from "
            "a single GPU to multiple nodes with multiple GPUs (using Tensor Parallelism "
            "and/or Pipeline Parallelism). The Python API of TensorRT-LLM is architectured "
            "to look similar to the PyTorch API. It provides users with a functional module "
            "containing functions like einsum, softmax, matmul or view. The layers module "
            "bundles useful building blocks to assemble LLMs; like an Attention block, a MLP "
            "or the entire Transformer layer. Model-specific components, like GPTAttention "
            "or BertAttention, can be found in the models module.")

    samples = []
    for _ in range(num_samples):
        samples.append({
            "input": input,
            "instruction": instruction,
            "output": output
        })

    with open(dummy_cnn_dataset, 'w') as f:
        json.dump(samples, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--long_prompt',
                        default=False,
                        action='store_true',
                        help='Using long prompts to generate tokens.')
    parser.add_argument('--output_filename',
                        type=str,
                        default='dummy_cnn.json',
                        help=('The name of the json output file.'))
    FLAGS = parser.parse_args()
    generate_dataset(num_samples=50,
                     output_filename=FLAGS.output_filename,
                     long_prompt=FLAGS.long_prompt)
