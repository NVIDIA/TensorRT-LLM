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
import os

# isort: off
import torch
import tensorrt as trt
# isort: on

import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.models import BertForQuestionAnswering, BertModel
from tensorrt_llm.runtime import Session, TensorInfo

from build import get_engine_name  # isort:skip


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='bert_outputs')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    model_name = config['builder_config']['name']
    runtime_rank = tensorrt_llm.mpi_rank() if world_size > 1 else 0

    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = get_engine_name(model_name, dtype, world_size,
                                     runtime_rank)
    serialize_path = os.path.join(args.engine_dir, serialize_path)

    stream = torch.cuda.current_stream().cuda_stream
    logger.info(f'Loading engine from {serialize_path}')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine')
    session = Session.from_serialized_engine(engine_buffer)

    for i in range(3):
        batch_size = (i + 1) * 4
        seq_len = (i + 1) * 32
        input_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()
        input_lengths = seq_len * torch.ones(
            (batch_size, ), dtype=torch.int32, device='cuda')
        token_type_ids = torch.randint(100, (batch_size, seq_len)).int().cuda()

        inputs = {
            'input_ids': input_ids,
            'input_lengths': input_lengths,
            'token_type_ids': token_type_ids
        }
        output_info = session.infer_shapes([
            TensorInfo('input_ids', trt.DataType.INT32, input_ids.shape),
            TensorInfo('input_lengths', trt.DataType.INT32,
                       input_lengths.shape),
            TensorInfo('token_type_ids', trt.DataType.INT32,
                       token_type_ids.shape),
        ])
        outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device='cuda')
            for t in output_info
        }
        if (model_name == BertModel.__name__):
            output_name = 'hidden_states'
        elif (model_name == BertForQuestionAnswering.__name__):
            output_name = 'logits'
        else:
            assert False, f"Unknown BERT model {model_name}"

        assert output_name in outputs, f'{output_name} not found in outputs, check if build.py set the name correctly'

        ok = session.run(inputs, outputs, stream)
        assert ok, "Runtime execution failed"
        torch.cuda.synchronize()
        res = outputs[output_name]
