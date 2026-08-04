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

import time

import torch
from transformers import AutoModelForCausalLM

from ..._utils import pad_vocab_size, release_gc


## Get HF model
def load_hf_deepseek(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                 device_map='auto',
                                                 dtype='auto',
                                                 trust_remote_code=True)
    return model


## Prepare weights for TP
def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def get_weight(config, prefix, dtype, postfix='.weight'):
    if config[prefix + postfix].dtype != dtype:
        config[prefix + postfix].data = config[prefix + postfix].to(dtype)
    return config[prefix + postfix].detach().cpu()


def get_trtllm_linear_weight(weight, prefix, postfix='weight'):
    results = {}
    results[prefix + postfix] = weight

    return results


def convert_deepseek(hf_model,
                     config,
                     mapping,
                     dtype='float32',
                     use_parallel_embedding=False,
                     sharding_dim=0):

    weights = {}
    tik = time.time()
    mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    moe_config = config.moe
    layers_range = mapping.pp_layers(config.num_hidden_layers)

    def convert_layer(l):
        prefix = f'model.layers.{l}.'
        print(prefix)
        trtllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        q_weight = get_weight(model_params, prefix + 'self_attn.q_proj', dtype)
        k_weight = get_weight(model_params, prefix + 'self_attn.k_proj', dtype)
        v_weight = get_weight(model_params, prefix + 'self_attn.v_proj', dtype)

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

        split_v = split_qkv_tp(qkv_weight, config.num_attention_heads,
                               config.hidden_size, mapping.tp_size,
                               mapping.tp_rank)

        weights.update(
            get_trtllm_linear_weight(split_v, trtllm_prex + 'attention.qkv.'))

        attn_dense_weight = get_weight(model_params,
                                       prefix + 'self_attn.o_proj', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)

        weights.update(
            get_trtllm_linear_weight(split_v, trtllm_prex + 'attention.dense.'))

        if moe_config.has_moe() and l > 0:
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)
            for suffix in ["gate_proj", "down_proj", "up_proj"]:
                model_params[f'model.layers.{l}.mlp.experts.{suffix}.weight'] = \
                torch.stack([model_params[f'model.layers.{l}.mlp.experts.{expert}.{suffix}.weight'].detach().cpu()
                            for expert in rank_experts])

            gate_proj = model_params[
                f'model.layers.{l}.mlp.experts.gate_proj.weight']
            down_proj = model_params[
                f'model.layers.{l}.mlp.experts.down_proj.weight']
            up_proj = model_params[
                f'model.layers.{l}.mlp.experts.up_proj.weight']
            if mapping.has_moe_tp():
                gate_proj = split(gate_proj,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)
                down_proj = split(down_proj,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=2)
                up_proj = split(up_proj,
                                mapping.tp_size,
                                mapping.tp_rank,
                                dim=1)

            model_params[
                f'model.layers.{l}.mlp.experts.up_gate_proj.weight'] = torch.concat(
                    [up_proj, gate_proj], dim=-2)
            model_params[
                f'model.layers.{l}.mlp.experts.down_proj.weight'] = down_proj

            ## mlp.experts.down_proj.weight
            moe_experts_down_proj_weights = get_weight(
                model_params, prefix + 'mlp.experts.down_proj', dtype)
            weights.update(
                get_trtllm_linear_weight(moe_experts_down_proj_weights,
                                         trtllm_prex + 'mlp.proj.'))
            ##mlp.experts.up_gate.weight
            moe_experts_up_gate_proj_weights = get_weight(
                model_params, prefix + 'mlp.experts.up_gate_proj', dtype)
            weights.update(
                get_trtllm_linear_weight(moe_experts_up_gate_proj_weights,
                                         trtllm_prex + 'mlp.fc.'))
            ## MOE hardcoded routing_input into trt.float32, please refer to moe.py line 397
            moe_experts_gate_weights = get_weight(model_params,
                                                  prefix + 'mlp.gate',
                                                  torch.float32)
            weights.update(
                get_trtllm_linear_weight(moe_experts_gate_weights,
                                         trtllm_prex + 'mlp.router.'))

            if moe_config.shared_expert_intermediate_size > 0:
                shared_moe_up_proj_weights = get_weight(
                    model_params, prefix + 'mlp.shared_experts.up_proj', dtype)
                shared_moe_up_proj_weights = split_matrix_tp(
                    shared_moe_up_proj_weights,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=0)
                shared_moe_down_proj_weights = get_weight(
                    model_params, prefix + 'mlp.shared_experts.down_proj',
                    dtype)
                shared_moe_down_proj_weights = split_matrix_tp(
                    shared_moe_down_proj_weights,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=1)
                shared_moe_gate_proj_weights = get_weight(
                    model_params, prefix + 'mlp.shared_experts.gate_proj',
                    dtype)
                shared_moe_gate_proj_weights = split_matrix_tp(
                    shared_moe_gate_proj_weights,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=0)
                shared_moe_gate_up_proj_weights = torch.concat(
                    [shared_moe_up_proj_weights, shared_moe_gate_proj_weights],
                    dim=-2)

                ## mlp.shared_experts.gate_up_proj.weight
                weights.update(
                    get_trtllm_linear_weight(
                        shared_moe_gate_up_proj_weights,
                        trtllm_prex + 'mlp.shared_expert.fc.'))

                ## mlp.shared_experts.down_proj.weight
                weights.update(
                    get_trtllm_linear_weight(
                        shared_moe_down_proj_weights,
                        trtllm_prex + 'mlp.shared_expert.proj.'))

        else:
            ## Current deepseek model has one MLP layer only, if it goes large consider to do fuse
            mlp_gate_weight = get_weight(model_params, prefix + 'mlp.up_proj',
                                         dtype)
            split_gate = split_matrix_tp(mlp_gate_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=0)
            weights.update(
                get_trtllm_linear_weight(split_gate, trtllm_prex + 'mlp.gate.'))

            mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                       dtype)
            split_fc = split_matrix_tp(mlp_fc_weight,
                                       mapping.tp_size,
                                       mapping.tp_rank,
                                       dim=0)
            weights.update(
                get_trtllm_linear_weight(split_fc, trtllm_prex + 'mlp.fc.'))

            mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                         dtype)
            split_proj = split_matrix_tp(mlp_proj_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=1)
            weights.update(
                get_trtllm_linear_weight(split_proj, trtllm_prex + 'mlp.proj.'))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + 'input_layernorm',
                                     dtype)
        weights[trtllm_prex + 'input_layernorm.weight'] = input_ln_weight
        post_ln_weight = get_weight(model_params,
                                    prefix + 'post_attention_layernorm', dtype)
        weights[trtllm_prex + 'post_layernorm.weight'] = post_ln_weight

    for l in layers_range:
        convert_layer(l)
        release_gc()

    v = get_weight(model_params, 'model.embed_tokens', dtype)
    if hf_model.config.tie_word_embeddings:
        # lm_head.weight has the same weights as embedding
        if mapping.is_last_pp_rank():
            if config.vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(config.vocab_size,
                                                   mapping.tp_size)
                pad_width = vocab_size_padded - config.vocab_size
                v = torch.nn.functional.pad(v, (0, 0, 0, pad_width), 'constant',
                                            0)
            weights['lm_head.weight'] = split(v, mapping.tp_size,
                                              mapping.tp_rank)
    if use_parallel_embedding:
        v = split_matrix_tp(v,
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=config.embedding_sharding_dim)
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v
    lm_head_weights = get_weight(model_params, 'lm_head', dtype)

    if mapping.is_last_pp_rank():
        if config.vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(config.vocab_size,
                                               mapping.tp_size)
            pad_width = vocab_size_padded - config.vocab_size
            lm_head_weights = torch.nn.functional.pad(lm_head_weights,
                                                      (0, 0, 0, pad_width),
                                                      'constant',
                                                      value=0)
        weights['lm_head.weight'] = split_matrix_tp(lm_head_weights,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=0)
    ln_f_w = get_weight(model_params, 'model.norm', dtype)
    weights['transformer.ln_f.weight'] = ln_f_w
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    #print(set(weights.keys()))
    return weights
