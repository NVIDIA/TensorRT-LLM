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

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import transformers

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (ChatGLMGenerationSession, GenerationSession,
                                  ModelConfig, SamplingConfig)

resources_dir = Path(
    __file__).parent.parent.parent.parent.parent / "examples/chatglm"
sys.path.insert(0, str(resources_dir))

from run import parse_arguments  # isort:skip

from build import find_engines  # isort:skip


def generate(model_name, batch_size, beam_width):

    model_name_dict = OrderedDict([
        ["chatglm-6b", "1"],
        ["chatglm2-6b", "2"],
        ["chatglm3-6b", "3"],
    ])

    print("generate expected %s output BatchSize=%d, BeamWidth=%d" %
          (model_name, batch_size, beam_width))

    args = parse_arguments()
    if batch_size == 1:
        args.input_text = args.input_text[:1]
    elif batch_size > 2:
        args.input_text += args.input_text[0] * (batch_size - 2)
    args.model_version = model_name_dict[model_name]
    args.beam_width = beam_width
    args.tokenizer_dir = resources_dir / model_name
    args.engine_dir = Path(__file__).parent.parent / "models/rt_engine/chatglm"

    tensorrt_llm.logger.set_level(args.log_level)

    config_path = Path(args.engine_dir) / (model_name + '-config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    assert (config['builder_config']['name'] == model_name)
    dtype = config['builder_config']['precision']
    end_id = config['builder_config']['eos_token_id']
    pad_id = config['builder_config']['pad_token_id']
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(
    ), f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = find_engines(
        Path(args.engine_dir),
        model_name=model_name,
        dtype=dtype,
        tp_size=world_size,
        rank=runtime_rank,
    )[0]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True)
    input_text = args.input_text
    tokenized = tokenizer(input_text,
                          return_tensors="pt",
                          padding=True,
                          return_length=True)
    input_ids = tokenized['input_ids'].int().contiguous().cuda()
    input_lengths = tokenized['length'].int().contiguous().cuda()

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
        model_name=model_name,
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
    if model_name == 'chatglm-6b':
        decoder = ChatGLMGenerationSession(
            model_config,
            engine_buffer,
            runtime_mapping,
        )
    else:
        decoder = GenerationSession(
            model_config,
            engine_buffer,
            runtime_mapping,
        )
    decoder.setup(input_ids.size(0), input_ids.size(1), args.max_output_len,
                  args.beam_width)
    output_ids = decoder.decode(input_ids, input_lengths, sampling_config)
    torch.cuda.synchronize()

    data_path = Path(__file__).parent.parent / "data" / model_name
    data_path.mkdir(parents=True, exist_ok=True)
    nBS, nBM = input_ids.size(0), args.beam_width
    np.save(
        str(data_path) + "/inputId-BS%d-BM%d.npy" % (nBS, nBM),
        input_ids.detach().cpu().numpy())
    outputId = output_ids.detach().cpu().numpy()

    nMaxOutputLength = 0
    for single_output in outputId.reshape(nBS * nBM, -1):
        if end_id in single_output:
            nMaxOutputLength = max(nMaxOutputLength,
                                   np.min(np.where(single_output == end_id)))
        else:
            nMaxOutputLength = len(single_output)
    np.save(
        str(data_path) + "/outputId-BS%d-BM%d.npy" % (nBS, nBM),
        outputId[:, :, :(nMaxOutputLength + 1)])


if __name__ == '__main__':
    generate("chatglm-6b", batch_size=1, beam_width=1)
    generate("chatglm-6b", batch_size=2, beam_width=1)
    generate("chatglm2-6b", batch_size=1, beam_width=1)
    generate("chatglm2-6b", batch_size=2, beam_width=1)
    generate("chatglm2-6b", batch_size=1, beam_width=2)
    generate("chatglm3-6b", batch_size=1, beam_width=1)
    generate("chatglm3-6b", batch_size=2, beam_width=1)
    generate("chatglm3-6b", batch_size=1, beam_width=2)
    print("Done.")
