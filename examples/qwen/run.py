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
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import GenerationSession, ModelConfig, SamplingConfig
from tensorrt_llm.runtime.generation import Mapping

from build import get_engine_name  # isort:skip

now_dir = os.path.dirname(os.path.abspath(__file__))

MAX_INPUT_LEN = 2048
MAX_SEQ_LEN = 4096


class QWenForCausalLMGenerationSession(GenerationSession):

    def __init__(
        self,
        model_config: ModelConfig,
        engine_buffer,
        mapping: Mapping,
        debug_mode=False,
        debug_tensors_to_save=None,
        cuda_graph_mode=False,
        stream: torch.cuda.Stream = None,
        global_max_input_length=MAX_INPUT_LEN,
        global_max_output_length=MAX_SEQ_LEN,
    ):
        super().__init__(model_config,
                         engine_buffer,
                         mapping,
                         debug_mode,
                         debug_tensors_to_save=debug_tensors_to_save,
                         cuda_graph_mode=cuda_graph_mode,
                         stream=stream)
        self.global_max_input_length = global_max_input_length
        self.global_max_output_length = global_max_output_length

    def generate(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        sampling_config: SamplingConfig,
        max_new_tokens: int,
        runtime_rank: int = 0,
    ):
        max_input_length = torch.max(input_lengths).item()
        max_new_tokens = min(max_new_tokens,
                             self.global_max_output_length - max_input_length)
        # setup batch_size, max_input_length, max_output_len
        self.setup(batch_size=input_lengths.size(0),
                   max_context_length=max_input_length,
                   max_new_tokens=max_new_tokens)
        output_ids = self.decode(input_ids, input_lengths, sampling_config)
        with torch.no_grad():
            torch.cuda.synchronize()
            if runtime_rank == 0:
                outputs = output_ids[:, 0, :]
                return outputs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument(
        '--engine_dir',
        type=str,
        default="qwen_outputs",
    )
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    default_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好，请问你叫什么？<|im_end|>\n<|im_start|>assistant\n"
    parser.add_argument('--input_text', type=str, default=default_text)
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
    return parser.parse_args()


def get_model(tokenizer_dir, engine_dir, log_level='error'):
    # --load the tokenizer and engine #
    tensorrt_llm.logger.set_level(log_level)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        legacy=False,
        trust_remote_code=True,
    )
    config_path = os.path.join(engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    gen_config_path = os.path.join(tokenizer_dir, 'generation_config.json')
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    top_k = gen_config['top_k']
    top_p = gen_config['top_p']
    chat_format = gen_config['chat_format']
    if chat_format == "raw":
        eos_token_id = gen_config['eos_token_id']
        pad_token_id = gen_config['pad_token_id']
    elif chat_format == "chatml":
        pad_token_id = eos_token_id = tokenizer.im_end_id
    else:
        raise Exception("unknown chat format ", chat_format)

    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size=world_size,
                                           rank=runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               dtype=dtype,
                               quant_mode=quant_mode,
                               use_custom_all_reduce=use_custom_all_reduce)
    sampling_config = SamplingConfig(
        end_id=eos_token_id,
        pad_id=pad_token_id,
        num_beams=1,
        top_k=top_k,
        top_p=top_p,
    )

    engine_name = get_engine_name('qwen', dtype, tp_size, pp_size, runtime_rank)
    serialize_path = os.path.join(engine_dir, engine_name)
    print(f'Loading engine from {serialize_path}')
    return (model_config, sampling_config, runtime_mapping, runtime_rank,
            serialize_path, remove_input_padding, tokenizer, eos_token_id,
            pad_token_id)


def generate(
    max_new_tokens: int,
    log_level: str = 'error',
    engine_dir: str = 'qwen_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    input_file: str = None,
    output_csv: str = None,
    output_npy: str = None,
    tokenizer_dir: str = None,
    num_beams: int = 1,
):
    (model_config, sampling_config, runtime_mapping, runtime_rank,
     serialize_path, remove_input_padding, tokenizer, eos_token_id,
     pad_token_id) = get_model(tokenizer_dir, engine_dir, log_level)
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = QWenForCausalLMGenerationSession(
        model_config,
        engine_buffer,
        runtime_mapping,
    )

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
                row = row[row != eos_token_id]
                input_tokens.append(row)
        else:
            print('Input file format not supported.')
            raise SystemExit

    input_ids = None
    input_lengths = None
    if input_file is None:
        input_ids = torch.tensor(input_tokens, device="cuda", dtype=torch.int32)
        input_lengths = torch.tensor([input_ids.size(1)],
                                     device="cuda",
                                     dtype=torch.int32)
    else:
        input_lengths = torch.tensor([len(x) for x in input_tokens],
                                     device="cuda",
                                     dtype=torch.int32)
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(input_ids,
                                     device="cuda",
                                     dtype=torch.int32).unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                eos_token_id).cuda()

    max_input_length = torch.max(input_lengths).item()
    max_new_tokens = min(max_new_tokens, MAX_SEQ_LEN - max_input_length)
    decoder.setup(batch_size=input_lengths.size(0),
                  max_context_length=max_input_length,
                  max_new_tokens=max_new_tokens)

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    if runtime_rank == 0:
        if output_csv is None and output_npy is None:
            for b in range(input_lengths.size(0)):
                inputs = input_tokens[b]
                input_text = tokenizer.decode(inputs)
                print(f'Input: \"{input_text}\"')
                if num_beams <= 1:
                    outputs = output_ids[b][0, len(inputs):].tolist()
                    output_text = tokenizer.decode(outputs,
                                                   skip_special_tokens=True)
                    print(f'Output: \"{output_text}\"')
                else:
                    for beam in range(num_beams):
                        outputs = output_ids[b][beam, len(inputs):].tolist()
                        output_text = tokenizer.decode(outputs,
                                                       skip_special_tokens=True)
                        print(f'Output(beam: {beam}): \"{output_text}\"')

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
    return


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
