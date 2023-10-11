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
import csv
import json
from pathlib import Path

import numpy as np
import torch
from utils import token_encoder

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

# GPT3 Related variables
# Reference : https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/gpt_sample.py
MERGES_FILE = "merges.txt"
VOCAB_FILE = "vocab.json"

PAD_ID = 50256
START_ID = 50256
END_ID = 50256


def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    dtype = config['builder_config']['precision']

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               quant_mode=quant_mode,
                               dtype=dtype)

    max_input_len = config['builder_config']['max_input_len']

    return model_config, world_size, dtype, max_input_len


def parse_input(input_text: str, input_file: str, tokenizer, pad_id: int,
                remove_input_padding: bool):
    input_tokens = []
    if input_file is None:
        input_tokens.append(tokenizer.encode(input_text))
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_tokens.append(np.array(line, dtype='int32'))
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                row = row[row != pad_id]
                input_tokens.append(row)
        else:
            print('Input file format not supported.')
            raise SystemExit

    input_ids = None
    input_lengths = torch.tensor([len(x) for x in input_tokens],
                                 dtype=torch.int32,
                                 device='cuda')
    if remove_input_padding:
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                 device='cuda').unsqueeze(0)
    else:
        input_ids = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
            pad_id).cuda()

    return input_ids, input_lengths


def print_output(output_ids, cum_log_probs, input_lengths, sequence_lengths,
                 tokenizer, output_csv, output_npy):

    num_beams = output_ids.size(1)
    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = output_ids[b][0][:input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input {b}: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = sequence_lengths[b][beam]
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                if num_beams > 1:
                    cum_log_prob = cum_log_probs[b][beam]
                    print(
                        f'Output {b}, beam {beam}: \"{output_text}\" (cum_log_prob: {cum_log_prob})'
                    )
                else:
                    print(f'Output {b}: \"{output_text}\"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='gpt_outputs')
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument(
        '--input_tokens',
        dest='input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--hf_model_location',
        type=str,
        default="gptj",
        help=
        'The hugging face model location stores the merges.txt and vocab.json to create tokenizer'
    )
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'gpt_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    input_file: str = None,
    output_csv: str = None,
    output_npy: str = None,
    hf_model_location: str = 'gptj',
    num_beams: int = 1,
    min_length: int = 1,
):
    tensorrt_llm.logger.set_level(log_level)

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    model_config, world_size, dtype, max_input_len = read_config(config_path)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    vocab_file = Path(hf_model_location) / VOCAB_FILE
    merges_file = Path(hf_model_location) / MERGES_FILE
    assert vocab_file.is_file(), f"{vocab_file} does not exist"
    assert merges_file.is_file(), f"{merges_file} does not exist"
    tokenizer = token_encoder.get_encoder(vocab_file, merges_file)

    sampling_config = SamplingConfig(end_id=END_ID,
                                     pad_id=PAD_ID,
                                     num_beams=num_beams,
                                     min_length=min_length)

    engine_name = get_engine_name('gptj', dtype, world_size, runtime_rank)
    serialize_path = Path(engine_dir) / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    input_ids, input_lengths = parse_input(input_text, input_file, tokenizer,
                                           PAD_ID,
                                           model_config.remove_input_padding)

    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0),
                  max_input_length,
                  max_output_len,
                  beam_width=num_beams)

    outputs = decoder.decode(input_ids,
                             input_lengths,
                             sampling_config,
                             output_sequence_lengths=True,
                             return_dict=True)
    output_ids = outputs['output_ids']
    sequence_lengths = outputs['sequence_lengths']
    torch.cuda.synchronize()

    cum_log_probs = decoder.cum_log_probs if num_beams > 1 else None

    if runtime_rank == 0:
        print_output(output_ids, cum_log_probs, input_lengths, sequence_lengths,
                     tokenizer, output_csv, output_npy)


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
