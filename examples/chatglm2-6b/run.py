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
import re
from pathlib import Path

import torch
import transformers

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import GenerationSession, ModelConfig, SamplingConfig

from build import find_engines  # isort:skip

MODEL_NAME = "chatglm2-6b"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='trtModel')
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='*',
        default=[
            "What's new between ChatGLM2-6B and ChatGLM-6B?",
            "Could you introduce NVIDIA Corporation for me?"
        ],
    )
    parser.add_argument(
        '--input_tokens',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None,
    )
    parser.add_argument(
        '--tokenizer_dir',
        type=str,
        default='pyTorchModel',
        help='Directory containing the tokenizer model.',
    )
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--random_seed', type=int, default=1)
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


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    assert (config['builder_config']['name'] == MODEL_NAME)
    dtype = config['builder_config']['precision']
    end_id = config['builder_config']['eos_token_id']
    pad_id = config['builder_config']['pad_token_id']
    max_batch_size = config['builder_config']['max_batch_size']
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(
    ), f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = find_engines(Path(args.engine_dir),
                                  dtype=dtype,
                                  tp_size=world_size,
                                  rank=runtime_rank)[0]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True)
    input_ids = None
    input_text = None
    if args.input_tokens is None:
        input_text = args.input_text[:max_batch_size]
        tokenized = tokenizer(input_text,
                              return_tensors="pt",
                              padding=True,
                              return_length=True)
        input_ids = tokenized['input_ids'].int().contiguous().cuda()
        input_lengths = tokenized['length'].int().contiguous().cuda()
    else:
        input_ids = []
        with open(args.input_tokens) as f_in:
            for line in f_in:
                for e in line.strip().split(','):
                    input_ids.append(int(e))
        input_text = "<ids from file>"
        input_ids = torch.tensor(input_ids,
                                 dtype=torch.int32).cuda().unsqueeze(0)

    if use_gpt_attention_plugin:
        # when using gpt attention plugin, inputs needs to align at the head
        input_ids_padding_right = torch.zeros_like(input_ids) + end_id
        for i, sample in enumerate(input_ids):
            nPadding = 0
            for token in sample:
                if token == pad_id:
                    nPadding += 1
                else:
                    break
            input_ids_padding_right[
                i, :len(sample[nPadding:])] = sample[nPadding:]
        input_ids = input_ids_padding_right

    model_config = ModelConfig(
        vocab_size=config['builder_config']['vocab_size'],
        num_layers=config['builder_config']['num_layers'],
        num_heads=config['builder_config']['num_heads'] // world_size,
        num_kv_heads=config['builder_config']['num_kv_heads'] // world_size,
        hidden_size=config['builder_config']['hidden_size'] // world_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=config['builder_config']['remove_input_padding'],
        model_name=MODEL_NAME,
        paged_kv_cache=config['builder_config']['paged_kv_cache'],
        quant_mode=QuantMode(config['builder_config']['quant_mode']),
        dtype=dtype,
    )

    sampling_config = SamplingConfig(
        end_id=end_id,
        pad_id=pad_id,
        num_beams=args.beam_width,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    sampling_config.random_seed = args.random_seed

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = GenerationSession(
        model_config,
        engine_buffer,
        runtime_mapping,
    )
    decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len,
                  args.beam_width)
    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    for i in range(len(output_ids.tolist())):
        output_beams_list = [
            tokenizer.batch_decode(output_ids[batch_idx, :,
                                              input_lengths[batch_idx]:],
                                   skip_special_tokens=True)
            for batch_idx in range(input_ids.size(0))
        ]
        output_text = process_response(output_beams_list[i])
        end = torch.where(input_ids[i] == end_id)[0]
        inputLength = int(end[0]) if len(end) > 0 else input_ids.shape[1]
        print("\nInput  %2d ---> len=%d\n%s" % (i, inputLength, input_text[i]))
        print("\nOutput %2d --->" % i)
        for j, simple_output in enumerate(output_text):
            end = torch.where(output_ids[i, j, input_lengths[i]:] == end_id)[0]
            outputLength = int(end[0]) if len(end) > 0 else args.max_output_len
            print("  Beam %2d ---> len=%d\n%s" %
                  (j, outputLength, simple_output))

    print("Finished!")
