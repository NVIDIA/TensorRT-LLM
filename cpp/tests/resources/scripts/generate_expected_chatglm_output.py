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

import json
import shutil as _shutil
from pathlib import Path as _Path

import numpy as np
import torch
import transformers

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import (ChatGLMGenerationSession, GenerationSession,
                                  ModelConfig, SamplingConfig)
from tensorrt_llm.runtime.model_runner import get_engine_name

import run  # isort:skip

resources_dir = _Path(
    __file__).parent.parent.parent.parent.parent / "examples/chatglm"


def generate(model_name, batch_size, beam_width):

    print("generate expected %s output BatchSize=%d, BeamWidth=%d" %
          (model_name, batch_size, beam_width))

    engine_dir = _Path(
        __file__).parent.parent / f"models/rt_engine/{model_name}"
    args = run.parse_arguments([
        '--engine_dir',
        str(engine_dir),
        '--tokenizer_dir',
        str(resources_dir / model_name),
        '--max_output_len',
        str(1024),
        '--num_beams',
        str(beam_width),
        '--input_text',
        "What's new between ChatGLM3-6B and ChatGLM2-6B?",
        "Could you introduce NVIDIA Corporation for me?",
    ])
    args.random_seed = 1

    tensorrt_llm.logger.set_level(args.log_level)

    if batch_size == 1:
        args.input_text = args.input_text[:1]
    else:
        args.input_text += args.input_text[0] * (batch_size - 2)

    config_path = _Path(args.engine_dir) / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    assert (config['builder_config']['name'] == model_name)
    dtype = config['builder_config']['precision']
    config['builder_config']['max_batch_size']
    max_input_len = config['builder_config']['max_input_len']
    max_output_len = config['builder_config']['max_output_len']
    config['builder_config']['max_beam_width']
    remove_input_padding = config['builder_config']['remove_input_padding']
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(
    ), f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(
        world_size,
        runtime_rank,
        tp_size=world_size,
    )
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name(model_name,
                                  dtype=dtype,
                                  tp_size=world_size,
                                  pp_size=1,
                                  rank=runtime_rank)
    serialize_path = _Path(args.engine_dir) / engine_name

    # fix remained error in chatglm_6b, hope to remove this in the future
    if model_name == "chatglm_6b":
        _shutil.copy(resources_dir / "tokenization_chatglm.py",
                     args.tokenizer_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_dir, trust_remote_code=True)
    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if model_name in ["glm_10b"]:
        sop_id = tokenizer.sop_token_id
        eop_id = tokenizer.eop_token_id
    tokenized = tokenizer(args.input_text,
                          return_tensors="pt",
                          padding=True,
                          return_length=True)
    input_ids = tokenized['input_ids'].int()
    input_lengths = tokenized['length'].int()
    max_input_len_real = torch.max(input_lengths)
    if max_input_len_real > max_input_len:
        print("Truncate input_length as %d" % max_input_len)
        input_ids = input_ids[:, :max_input_len]
        input_lengths = torch.where(input_lengths > max_input_len,
                                    max_input_len, input_lengths)
    else:
        max_input_len = max_input_len_real
    if model_name in ["glm_10b"]:
        input_ids = torch.cat(
            (input_ids, input_ids.new_full((batch_size, 1), sop_id)),
            dim=-1,
        )
        input_lengths += 1
        max_input_len_real += 1

    if remove_input_padding:
        input_ids_no_padding = torch.zeros(torch.sum(input_lengths),
                                           dtype=torch.int32)
        lengths_acc = torch.cumsum(
            torch.cat([torch.IntTensor([0]), input_lengths]),
            dim=0,
        )
        for i in range(len(input_ids)):
            input_ids_no_padding[
                lengths_acc[i]:lengths_acc[i + 1]] = torch.IntTensor(
                    input_ids[i,
                              max_input_len - input_lengths[i]:max_input_len])

        input_ids = input_ids_no_padding

    elif use_gpt_attention_plugin:
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
        end_id=eop_id if model_name in ["glm_10b"] else end_id,
        pad_id=pad_id,
        num_beams=beam_width,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    sampling_config.random_seed = args.random_seed

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()

    if model_name in ["chatglm_6b", "glm_10b"]:
        session = ChatGLMGenerationSession
    else:
        session = GenerationSession
    decoder = session(
        model_config,
        engine_buffer,
        runtime_mapping,
    )

    decoder.setup(
        len(args.input_text),
        max_input_len,
        max_output_len,
        beam_width,
    )
    output = decoder.decode(
        input_ids.contiguous().cuda(),
        input_lengths.contiguous().cuda(),
        sampling_config,
        output_sequence_lengths=True,
        return_dict=True,
    )
    torch.cuda.synchronize()

    output_ids = output["output_ids"]
    output["sequence_lengths"]

    data_path = _Path(__file__).parent.parent / "data" / model_name
    data_path.mkdir(parents=True, exist_ok=True)
    nBS, nBM = input_ids.size(0), beam_width
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

    generate("chatglm_6b", batch_size=1, beam_width=1)
    generate("chatglm2_6b", batch_size=1, beam_width=1)
    generate("chatglm2_6b", batch_size=2, beam_width=1)
    generate("chatglm2_6b", batch_size=1, beam_width=2)
    generate("chatglm3_6b", batch_size=1, beam_width=1)
    generate("chatglm3_6b", batch_size=2, beam_width=1)
    generate("chatglm3_6b", batch_size=1, beam_width=2)

    print("Done")
