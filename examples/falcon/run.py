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
from transformers import PreTrainedTokenizerFast

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='falcon_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="tiiuae/falcon-rw-1b",
                        help="Tokenizer path or name.")
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
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def read_config(config_path: Path):
    with config_path.open('r') as f:
        config = json.load(f)

    builder_config = config['builder_config']
    dtype = builder_config['precision']
    tp_size = builder_config['tensor_parallel']
    pp_size = builder_config['pipeline_parallel']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size '\
        f'({tensorrt_llm.mpi_world_size()})'

    num_heads = builder_config['num_heads'] // tp_size
    num_kv_heads = builder_config.get('num_kv_heads', num_heads)
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    hidden_size = builder_config['hidden_size'] // tp_size

    vocab_size = builder_config['vocab_size']
    num_layers = builder_config['num_layers']
    quant_mode = QuantMode(builder_config['quant_mode'])

    plugin_config = config['plugin_config']
    use_gpt_attention_plugin = plugin_config['gpt_attention_plugin']
    paged_kv_cache = plugin_config['paged_kv_cache']
    tokens_per_block = plugin_config['tokens_per_block']
    remove_input_padding = plugin_config['remove_input_padding']
    use_custom_all_reduce = plugin_config.get('use_custom_all_reduce', False)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               quant_mode=quant_mode,
                               dtype=dtype,
                               use_custom_all_reduce=use_custom_all_reduce)

    return model_config, tp_size, pp_size, world_size, dtype


def parse_input(input_text: str, input_file: str, tokenizer, pad_id: int,
                remove_input_padding: bool):
    input_tokens = []
    if input_file is None:
        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))
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


def print_output(output_ids, input_lengths, max_output_len, tokenizer,
                 output_csv, output_npy):
    num_beams = output_ids.size(1)
    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = output_ids[b][0][:input_lengths[b]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[b]
                output_end = input_lengths[b] + max_output_len
                outputs = output_ids[b][beam][output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                print(f'Output: \"{output_text}\"')

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


def main():
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    engine_dir = Path(args.engine_dir)
    model_config, tp_size, pp_size, world_size, dtype = read_config(
        engine_dir / 'config.json')

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name('falcon', dtype, tp_size, pp_size,
                                  runtime_rank)
    serialize_path = engine_dir / engine_name

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    input_ids, input_lengths = parse_input(args.input_text, args.input_file,
                                           tokenizer, tokenizer.eos_token_id,
                                           model_config.remove_input_padding)

    sampling_config = SamplingConfig(end_id=tokenizer.eos_token_id,
                                     pad_id=tokenizer.pad_token_id,
                                     num_beams=args.num_beams)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=args.debug)
    decoder.setup(input_ids.size(0),
                  max_context_length=input_ids.size(1),
                  max_new_tokens=args.max_output_len,
                  beam_width=args.num_beams)
    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    if runtime_rank == 0:
        print_output(output_ids, input_lengths, args.max_output_len, tokenizer,
                     args.output_csv, args.output_npy)


if __name__ == '__main__':
    main()
