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
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer

import tensorrt_llm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='warning')
    parser.add_argument('--model_dir', type=str, default='gpt2')
    parser.add_argument('--data_type',
                        type=str,
                        choices=['fp32', 'fp16'],
                        default='fp32')
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
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'error',
    model_dir: str = 'gpt2',
    data_type: str = 'fp32',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    input_file: str = None,
    output_csv: str = None,
    output_npy: str = None,
    tokenizer_path='gpt2',
    vocab_file=None,
):
    tensorrt_llm.logger.set_level(log_level)

    model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                 trust_remote_code=True)
    model.cuda()
    if data_type == 'fp16':
        model.half()

    if vocab_file is not None:
        tokenizer = T5Tokenizer(vocab_file=vocab_file)
        END_ID = 50256
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        END_ID = tokenizer.eos_token_id

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
                row = row[row != END_ID]
                input_tokens.append(row)
        else:
            print('Input file format not supported.')
            raise SystemExit

    input_ids = None
    input_lengths = None
    if input_file is None:
        input_ids = torch.tensor(input_tokens, dtype=torch.int32, device='cuda')
        input_lengths = torch.tensor([input_ids.size(1)],
                                     dtype=torch.int32,
                                     device='cuda')
        max_input_length = torch.max(input_lengths).item()
    else:
        input_lengths = torch.tensor([len(x) for x in input_tokens],
                                     dtype=torch.int32,
                                     device='cuda')
        max_input_length = torch.max(input_lengths).item()
        input_ids = np.full((len(input_lengths), max_input_length), END_ID)
        for i in range(len(input_lengths)):
            input_ids[i][-len(input_tokens[i]):] = input_tokens[i]
        input_ids = torch.tensor(input_ids, dtype=torch.int32, device='cuda')

    top_k = 1
    temperature = 1
    output_ids = model.generate(input_ids,
                                max_length=max_input_length + max_output_len,
                                top_k=top_k,
                                temperature=temperature,
                                eos_token_id=END_ID,
                                pad_token_id=END_ID)
    torch.cuda.synchronize()

    if output_csv is None and output_npy is None:
        for b in range(input_lengths.size(0)):
            inputs = input_tokens[b]
            input_text = tokenizer.decode(inputs)
            print(f'Input: {input_text}')
            outputs = output_ids[b][max_input_length:].tolist()
            output_text = tokenizer.decode(outputs)
            print(f'Output: {output_text}')

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids[:, max_input_length:].tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids[:, max_input_length:].tolist()
        np.save(output_file, np.array(outputs, dtype='int32'))


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
