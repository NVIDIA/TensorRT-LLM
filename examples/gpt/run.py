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
from transformers import AutoTokenizer, T5Tokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip


def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config'].get('pipeline_parallel', 1)
    world_size = tp_size * pp_size
    assert tp_size * pp_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({tp_size} * {pp_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    assert (config['builder_config']['num_heads'] %
            tp_size) == 0, f"The number of heads must be a multiple of tp_size"
    num_heads = config['builder_config']['num_heads'] // tp_size
    num_kv_heads = (config['builder_config']['num_kv_heads'] + tp_size -
                    1) // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    use_prompt_tuning = config['builder_config']['use_prompt_tuning']
    dtype = config['builder_config']['precision']
    gather_all_token_logits = config['builder_config'][
        'gather_all_token_logits']
    use_custom_all_reduce = config['plugin_config']['use_custom_all_reduce']

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               remove_input_padding=remove_input_padding,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               use_prompt_tuning=use_prompt_tuning,
                               dtype=dtype,
                               gather_all_token_logits=gather_all_token_logits,
                               use_custom_all_reduce=use_custom_all_reduce)

    dtype = config['builder_config']['precision']
    max_input_len = config['builder_config']['max_input_len']

    return model_config, world_size, dtype, max_input_len


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


def ptuning_setup(prompt_table, dtype, hidden_size, tasks, input_ids,
                  input_lengths, remove_input_padding):
    if prompt_table is not None:
        prompt_table = torch.from_numpy(np.load(prompt_table))
        task_vocab_size = torch.tensor([prompt_table.shape[1]],
                                       dtype=torch.int32,
                                       device="cuda")
        prompt_table = prompt_table.view(
            (prompt_table.shape[0] * prompt_table.shape[1],
             prompt_table.shape[2]))
        prompt_table = prompt_table.cuda().to(
            dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
    else:
        prompt_table = torch.empty([1, hidden_size]).cuda()
        task_vocab_size = torch.zeros([1]).cuda()

    num_sequences = input_lengths.size(
        0) if remove_input_padding else input_ids.size(0)

    if tasks is not None:
        tasks = torch.tensor([int(t) for t in tasks.split(',')],
                             dtype=torch.int32,
                             device="cuda")
        assert tasks.shape[
            0] == num_sequences, "Number of supplied tasks must match input batch size"
    else:
        tasks = torch.zeros([num_sequences]).cuda()

    return [prompt_table, tasks, task_vocab_size]


def print_output(output_ids, input_lengths, sequence_lengths, tokenizer,
                 output_csv, output_npy):

    num_beams = output_ids.size(1)
    if output_csv is None and output_npy is None:
        for batch_idx in range(input_lengths.size(0)):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            print(f'Input: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='gpt_outputs')
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
    parser.add_argument('--tokenizer',
                        dest='tokenizer_path',
                        help="HF tokenizer config path",
                        default='gpt2')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument(
        '--prompt_table',
        type=Path,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--tasks',
        help="Comma-separated list of tasks for prompt tuning: ex 0,3,1,0")
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'gpt_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    input_file: str = None,
    output_csv: str = None,
    output_npy: str = None,
    tokenizer_path: str = 'gpt2',
    vocab_file=None,
    num_beams: int = 1,
    prompt_table: Path = None,
    tasks: str = None,
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

    if vocab_file is not None:
        tokenizer = T5Tokenizer(vocab_file=vocab_file)
        EOS_TOKEN = 50256
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        EOS_TOKEN = tokenizer.eos_token_id

    # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
    # stop_words_list = [[" London"], ["eventually became"]]
    # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
    # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
    stop_words_list = None

    # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
    # bad_words_list = [[" chef"], [" eventually, chef before"]]
    # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
    # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
    bad_words_list = None

    sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                     pad_id=EOS_TOKEN,
                                     num_beams=num_beams)

    engine_name = get_engine_name('gpt', dtype, world_size, runtime_rank)
    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=False)

    input_ids, input_lengths = parse_input(input_text, input_file, tokenizer,
                                           EOS_TOKEN,
                                           model_config.remove_input_padding)

    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0),
                  max_input_length,
                  max_output_len,
                  beam_width=num_beams)

    ptuning_args = [] if not model_config.use_prompt_tuning else ptuning_setup(
        prompt_table, dtype, model_config.hidden_size, tasks, input_ids,
        input_lengths, model_config.remove_input_padding)

    outputs = decoder.decode(input_ids,
                             input_lengths,
                             sampling_config,
                             *ptuning_args,
                             output_sequence_lengths=True,
                             return_dict=True,
                             stop_words_list=stop_words_list,
                             bad_words_list=bad_words_list)
    output_ids = outputs['output_ids']
    sequence_lengths = outputs['sequence_lengths']
    torch.cuda.synchronize()
    if runtime_rank == 0:
        print_output(output_ids, input_lengths, sequence_lengths, tokenizer,
                     output_csv, output_npy)
        if model_config.gather_all_token_logits:
            print(outputs['context_logits'])


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
