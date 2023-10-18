#!/usr/bin/env python3
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
import json
import os
import pathlib as _pl
import re
import sys

import numpy as np
import torch
import transformers

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

resources_dir = _pl.Path(
    __file__).parent.parent.parent.parent.parent / "examples/chatglm6b"
sys.path.insert(0, str(resources_dir))

from build import get_engine_name

END_ID = 130005
PAD_ID = 3


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir',
                        type=str,
                        default=str(resources_dir) + '/trtModel')
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='*',
        default=["Hello!", "Could you introduce NVIDIA Corporation for me?"])
    parser.add_argument(
        '--input_tokens',
        type=str,
        help='CSV file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=str(resources_dir) + '/pyTorchModel',
                        help='Directory containing the tokenizer model.')
    return parser.parse_args()


def process_response(responseList):
    for i, response in enumerate(responseList):
        response = response.strip()
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0],
                              r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0],
                              r"%s\1" % item[1], response)

        responseList[i] = response
    return responseList


def generate(batch_size, beam_width):

    print("generate expected chatglm6b output BatchSize=%d, BeamWidth=%d" %
          (batch_size, beam_width))
    args = parse_arguments()
    if batch_size == 1:
        args.input_text = args.input_text[:1]
    elif batch_size > 2:
        args.input_text += args.input_text[0] * (batch_size - 2)
    args.beam_width = beam_width

    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('chatglm6b', dtype, world_size, runtime_rank)
    serialize_path = os.path.join(args.engine_dir, engine_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True)
    input_text = args.input_text
    tokenized = tokenizer(input_text,
                          return_tensors="pt",
                          padding=True,
                          return_length=True)
    input_ids = tokenized['input_ids'].int().contiguous().cuda()
    input_lengths = tokenized['length'].int().contiguous().cuda()

    input_ids_padding_right = torch.zeros_like(input_ids) + END_ID
    for i, sample in enumerate(input_ids):
        nPadding = 0
        for token in sample:
            if token == PAD_ID:
                nPadding += 1
            else:
                break
        input_ids_padding_right[i, :len(sample[nPadding:])] = sample[nPadding:]
    input_ids = input_ids_padding_right

    model_config = ModelConfig(model_name="chatglm6b",
                               num_heads=num_heads,
                               num_kv_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin)
    sampling_config = SamplingConfig(
        end_id=END_ID,
        pad_id=PAD_ID,
        top_k=1,
        top_p=1.0,
        num_beams=args.beam_width,
    )
    sampling_config.random_seed = 1

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.ChatGLM6BHeadModelGenerationSession(
        model_config, engine_buffer, runtime_mapping)
    decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len,
                  args.beam_width)
    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    data_path = _pl.Path(__file__).parent.parent / "data/chatglm6b"
    if not os.path.exists(str(data_path)):
        os.mkdir(data_path)
    nBS, nBM = input_ids.size(0), args.beam_width
    np.save(
        str(data_path) + "/inputId-BS%d-BM%d.npy" % (nBS, nBM),
        input_ids.detach().cpu().numpy())
    outputId = output_ids.detach().cpu().numpy()

    nMaxOutputLength = 0
    for single_output in outputId.reshape(nBS * nBM, -1):
        nMaxOutputLength = max(nMaxOutputLength,
                               np.min(np.where(single_output == END_ID)))
    np.save(
        str(data_path) + "/outputId-BS%d-BM%d.npy" % (nBS, nBM),
        outputId[:, :, :(nMaxOutputLength + 1)])


if __name__ == '__main__':
    generate(batch_size=1, beam_width=1)
    generate(batch_size=2, beam_width=1)
    generate(batch_size=1, beam_width=2)
    print("Finished!")
