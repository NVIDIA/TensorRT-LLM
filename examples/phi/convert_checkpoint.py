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
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import safetensors
import torch
from transformers import AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_torch


def torch_split(v, tensor_parallel, idx, dim=0):
    if tensor_parallel == 1:
        return v
    else:
        return (torch.split(v, v.shape[dim] // tensor_parallel,
                            dim=dim)[idx]).contiguous()


def convert_hf_phi(hf_model,
                   rank=0,
                   tensor_parallel=1,
                   dtype='float32',
                   use_parallel_embedding=False,
                   sharding_dim=0):

    hf_model_phi_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "self_attn.dense.weight",
        "self_attn.dense.bias",
        "mlp.fc1.weight",
        "mlp.fc1.bias",
        "mlp.fc2.weight",
        "mlp.fc2.bias",
    ]

    tensorrt_llm_model_phi_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "attention.dense.weight",
        "attention.dense.bias",
        "mlp.fc.weight",
        "mlp.fc.bias",
        "mlp.proj.weight",
        "mlp.proj.bias",
    ]

    weights = {}
    torch_dtype = str_dtype_to_torch(dtype)
    hf_phi_state_dict = hf_model.state_dict()

    # Embedding
    # [vocab_size, hidden_size]
    v = hf_phi_state_dict.get('model.embed_tokens.weight').to(torch_dtype).cpu()
    if use_parallel_embedding:
        v = torch_split(v, tensor_parallel, rank, sharding_dim)
    weights['transformer.vocab_embedding.weight'] = v

    # Decoder Layers
    n_layer = hf_model.config.num_hidden_layers
    for layer_idx in range(n_layer):
        hf_prefix = f"model.layers.{layer_idx}."
        tllm_prex = f'transformer.layers.{layer_idx}.'

        # MLPs
        for idx, hf_attr in enumerate(hf_model_phi_block_names):
            v = hf_phi_state_dict.get(hf_prefix + hf_attr).to(torch_dtype).cpu()

            if tensor_parallel > 1:
                if 'self_attn.dense.weight' in hf_attr:
                    # [n=hidden_size, k=hidden_size] ->
                    # [n=hidden_size, k=hidden_size // tensor_parallel]
                    v = torch_split(v, tensor_parallel, rank, dim=1)
                elif 'mlp.fc1.weight' in hf_attr:
                    # [hidden_size * 4, hidden_size] ->
                    # [hidden_size * 4 // tensor_parallel, hidden_size]
                    v = torch_split(v, tensor_parallel, rank, dim=0)
                elif 'mlp.fc1.bias' in hf_attr:
                    # [hidden_size * 4] -> [hidden_size * 4 // tensor_parallel]
                    v = torch_split(v, tensor_parallel, rank, dim=0)
                elif 'mlp.fc2.weight' in hf_attr:
                    # [hidden_size, hidden_size * 4] ->
                    # [hidden_size, hidden_size * 4 // tensor_parallel]
                    v = torch_split(v, tensor_parallel, rank, dim=1)

            tllm_attr = tensorrt_llm_model_phi_block_names[idx]
            weights[f'{tllm_prex}{tllm_attr}'] = v

        # Attention QKV Linear
        num_heads = hf_model.config.num_attention_heads
        hidden_size = hf_model.config.hidden_size
        hidden_size // num_heads

        # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
        q_weights = hf_phi_state_dict.get(hf_prefix + "self_attn.q_proj.weight")
        k_weights = hf_phi_state_dict.get(hf_prefix + "self_attn.k_proj.weight")
        v_weights = hf_phi_state_dict.get(hf_prefix + "self_attn.v_proj.weight")
        q_bias = hf_phi_state_dict.get(hf_prefix + "self_attn.q_proj.bias")
        k_bias = hf_phi_state_dict.get(hf_prefix + "self_attn.k_proj.bias")
        v_bias = hf_phi_state_dict.get(hf_prefix + "self_attn.v_proj.bias")
        qkv_weights = torch.cat((q_weights, k_weights, v_weights), dim=0)
        qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)

        qkv_weights = qkv_weights.reshape([hidden_size * 3, hidden_size])
        qkv_bias = qkv_bias.reshape([hidden_size * 3])

        if tensor_parallel > 1:
            qkv_weights = qkv_weights.reshape(
                3, hidden_size, hidden_size).to(torch_dtype).cpu()
            qkv_weights = torch_split(qkv_weights, tensor_parallel, rank,
                                      dim=1).reshape(
                                          3 * (hidden_size // tensor_parallel),
                                          hidden_size)

            qkv_bias = qkv_bias.reshape(3, hidden_size).to(torch_dtype).cpu()
            qkv_bias = torch_split(qkv_bias, tensor_parallel, rank,
                                   dim=1).reshape(
                                       3 * (hidden_size // tensor_parallel))

            weights[
                f"{tllm_prex}attention.qkv.weight"] = qkv_weights.contiguous()
            weights[f"{tllm_prex}attention.qkv.bias"] = qkv_bias.contiguous()
        else:
            weights[f"{tllm_prex}attention.qkv.weight"] = qkv_weights.to(
                torch_dtype).cpu()
            weights[f"{tllm_prex}attention.qkv.bias"] = qkv_bias.to(
                torch_dtype).cpu()

    # Final Layer Norm
    v = hf_phi_state_dict.get('model.final_layernorm.weight')
    weights["transformer.ln_f.weight"] = v.to(torch_dtype).cpu()

    v = hf_phi_state_dict.get('model.final_layernorm.bias')
    weights["transformer.ln_f.bias"] = v.to(torch_dtype).cpu()

    # LM Head
    v = hf_phi_state_dict.get('lm_head.weight').to(torch_dtype).cpu()
    if tensor_parallel > 1:
        # [vocab_size, hidden_size] ->
        # [vocab_size // tensor_parallel, hidden_size]
        if v.shape[0] % tensor_parallel != 0:
            # padding
            vocab_size_padded = pad_vocab_size(v.shape[0], tensor_parallel)
            pad_width = vocab_size_padded - v.shape[0]
            v = np.pad(v, ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0)

        v = torch_split(v, tensor_parallel, rank, dim=0)
    weights["lm_head.weight"] = v

    v = hf_phi_state_dict.get('lm_head.bias').to(torch_dtype).cpu()
    if tensor_parallel > 1:
        v = torch_split(v, tensor_parallel, rank, dim=0)
    weights["lm_head.bias"] = v

    return weights


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size
    assert args.pp_size == 1, "Pipeline parallelism is not supported."

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                    torch_dtype="auto",
                                                    trust_remote_code=True)
    hf_config = hf_model.config
    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_key_value_heads,
        'partial_rotary_factor': hf_config.partial_rotary_factor,
        'rope_theta': hf_config.rope_theta,
        'hidden_size': hf_config.hidden_size,
        'intermediate_size': hf_config.intermediate_size,
        'vocab_size': hf_config.vocab_size,
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': hf_config.hidden_act,
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': False,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': False,
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def covert_and_save(rank):
        weights = convert_hf_phi(
            hf_model,
            rank,
            world_size,
            dtype=args.dtype,
            use_parallel_embedding=args.use_parallel_embedding,
            sharding_dim=args.embedding_sharding_dim)
        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:
        for rank in range(world_size):
            covert_and_save(rank)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(covert_and_save, rank) for rank in range(world_size)
            ]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
