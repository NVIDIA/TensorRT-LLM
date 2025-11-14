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
from transformers import AutoConfig, AutoModelForCausalLM

from tensorrt_llm.layers import MoeConfig

from ..._utils import pad_vocab_size, release_gc, str_dtype_to_torch
from ...logger import logger
from ...mapping import Mapping
from ..convert_utils import get_tllm_linear_weight

# `Override num_hidden_layers` used for reduce number of hidden layers in DeepseekV2ForCausalLM for debug purpose
OVERRIDE_HIDDEN_LAYERS = None  # 2


# Convert config parameters to dict TODO: remove this function and change CI test
def create_trt_config_from_hf(model_dir,
                              dtype,
                              mapping: Mapping,
                              override_fields: dict = {}):
    config = {}
    assert isinstance(model_dir, str)
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # Override num_hidden_layers
    if OVERRIDE_HIDDEN_LAYERS is not None:
        hf_config.num_hidden_layers = OVERRIDE_HIDDEN_LAYERS
        print(
            f'Override hidden layers to {hf_config.num_hidden_layers} for DeepseekV2ForCausalLM'
        )
    dtype = dtype
    n_layer = hf_config.num_hidden_layers
    n_head = hf_config.num_attention_heads
    n_embd = hf_config.hidden_size
    inter_size = hf_config.intermediate_size
    n_kv_head = hf_config.num_key_value_heads
    vocab_size = hf_config.vocab_size
    n_positions = hf_config.max_position_embeddings
    hidden_act = 'swiglu'  # TRT-LLM request make gated activation explicit for MOE implementation
    rotary_base = hf_config.rope_theta
    rms_norm_eps = hf_config.rms_norm_eps
    rotary_scaling_beta_fast = hf_config.rope_scaling['beta_fast']
    rotary_scaling_beta_slow = hf_config.rope_scaling['beta_slow']
    rotary_scaling_factor = hf_config.rope_scaling['factor']
    rotary_scaling_mscale = hf_config.rope_scaling['mscale']
    rotary_scaling_mscale_all_dim = hf_config.rope_scaling['mscale_all_dim']
    rotary_scaling_original_max_position_embeddings = hf_config.rope_scaling[
        'original_max_position_embeddings']
    rotary_scaling_type = 'yarn'
    kv_lora_rank = hf_config.kv_lora_rank
    q_lora_rank = hf_config.q_lora_rank
    qk_nope_head_dim = hf_config.qk_nope_head_dim
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    v_head_dim = hf_config.v_head_dim
    moe_num_experts = hf_config.n_routed_experts
    moe_inter_size = hf_config.moe_intermediate_size
    moe_num_shared_experts = hf_config.n_shared_experts
    moe_top_k = hf_config.num_experts_per_tok
    moe_n_group = hf_config.n_group
    moe_topk_group = hf_config.topk_group
    moe_routed_scaling_factor = hf_config.routed_scaling_factor
    assert moe_routed_scaling_factor > 0, 'routed_scaling_factor should be greater than 0'
    if hf_config.topk_method == 'group_limited_greedy':
        if moe_top_k > 1 and hf_config.norm_topk_prob:
            moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED_RENORM
        else:
            moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.DEVICE_LIMITED
    elif hf_config.topk_method == 'greedy':
        assert moe_routed_scaling_factor == 1.0, 'The combination of topk_method == greedy and routed_scaling_factor != 1.0 is not supported'
        if moe_top_k > 1 and hf_config.norm_topk_prob:
            moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
        else:
            moe_renorm_mode = MoeConfig.ExpertScaleNormalizationMode.NONE
    else:
        raise AssertionError(
            'Unsupported topk_method in hf_config: {hf_config.topk_method}')

    config = {
        'architecture': 'DeepseekV2ForCausalLM',
        'dtype': dtype,
        'logits_type': 'float32',
        'num_hidden_layers': n_layer,
        'num_attention_heads': n_head,
        'hidden_size': n_embd,
        'intermediate_size': inter_size,
        'num_key_value_heads': n_kv_head,
        'vocab_size': vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': n_positions,
        'hidden_act': hidden_act,
        'rotary_base': rotary_base,
        'norm_epsilon': rms_norm_eps,
        'rotary_scaling': {
            'beta_fast': rotary_scaling_beta_fast,
            'beta_slow': rotary_scaling_beta_slow,
            'factor': rotary_scaling_factor,
            'mscale': rotary_scaling_mscale,
            'mscale_all_dim': rotary_scaling_mscale_all_dim,
            'original_max_position_embeddings':
            rotary_scaling_original_max_position_embeddings,
            'type': rotary_scaling_type,
        },
        'mapping': {
            'world_size': mapping.tp_size * mapping.pp_size,
            'tp_size': mapping.tp_size,
            'pp_size': mapping.pp_size,
            'moe_tp_size': mapping.moe_tp_size,
            'moe_ep_size': mapping.moe_ep_size,
        },
        'kv_lora_rank': kv_lora_rank,
        'q_lora_rank': q_lora_rank,
        'qk_nope_head_dim': qk_nope_head_dim,
        'qk_rope_head_dim': qk_rope_head_dim,
        'v_head_dim': v_head_dim,
        'moe_num_experts': moe_num_experts,
        'moe_inter_size': moe_inter_size,
        'moe_num_shared_experts': moe_num_shared_experts,
        'moe_top_k': moe_top_k,
        'moe_renorm_mode': moe_renorm_mode,
        'moe_n_group': moe_n_group,
        'moe_topk_group': moe_topk_group,
        'moe_routed_scaling_factor': moe_routed_scaling_factor,
    }

    config.update(override_fields)

    moe_config = MoeConfig(
        num_experts=config['moe_num_experts'],
        shared_expert_intermediate_size=config['moe_num_shared_experts'] *
        config['moe_inter_size'],
        top_k=config['moe_top_k'],
        normalization_mode=config['moe_renorm_mode'],
        device_limited_n_group=config['moe_n_group'],
        device_limited_topk_group=config['moe_topk_group'],
        device_limited_routed_scaling_factor=config['moe_routed_scaling_factor']
    )
    moe_config.validate()

    return config


# Get HF model
def load_hf_deepseek(model_dir, load_model_on_cpu=False):
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if OVERRIDE_HIDDEN_LAYERS is not None:
        hf_config.num_hidden_layers = OVERRIDE_HIDDEN_LAYERS
        print(
            f'Override hidden layers to {hf_config.num_hidden_layers} for DeepseekV2ForCausalLM'
        )

    if load_model_on_cpu:
        # Skip setting max_memory when loading on CPU, you might have OOM.
        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     config=hf_config,
                                                     device_map='cpu',
                                                     dtype='auto',
                                                     trust_remote_code=True)
    else:
        # Deepseek-v2 236B parameters with FP16 dtype need at least 472G GPU memory
        # (official suggest at least 8x80G GPUs, see https://huggingface.co/deepseek-ai/DeepSeek-V2)

        max_memory = None
        device_map = 'auto'

        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_count = torch.cuda.device_count()

        if gpu_memory < 90_000_000_000 and gpu_count == 8:
            # WAR OOM loading on 8*80G GPUs
            max_memory = {i: "76GB" for i in range(8)}
            device_map = 'sequential'
        elif gpu_memory < 180_000_000_000 and gpu_count == 4:
            # WAR OOM loading on 4*141G GPUs
            max_memory = {i: "128GB" for i in range(4)}
            device_map = 'sequential'
        elif gpu_memory < 180_000_000_000 and gpu_count == 8:
            # WAR OOM loading on 8*141G GPUs
            max_memory = {i: "128GB" for i in range(8)}
            device_map = 'sequential'

        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     config=hf_config,
                                                     device_map=device_map,
                                                     max_memory=max_memory,
                                                     dtype='auto',
                                                     trust_remote_code=True)

    return model


# Prepare weights for TP
def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def get_weight(config, prefix, dtype, postfix='.weight'):
    if config[prefix + postfix].dtype != dtype:
        config[prefix + postfix].data = config[prefix + postfix].to(dtype)
    return config[prefix + postfix].detach().cpu()


def get_param_weight(weight, prefix):
    results = {}
    results[prefix] = weight

    return results


def convert_deepseekv2(hf_model,
                       config,
                       mapping,
                       dtype='float32',
                       use_parallel_embedding=False,
                       sharding_dim=0):

    weights = {}
    tik = time.time()
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    moe_config = config.moe

    layers_range = mapping.pp_layers(config.num_hidden_layers)

    def convert_layer(l):
        prefix = f'model.layers.{l}.'
        trtllm_prefix = f'transformer.layers.{l - layers_range[0]}.'
        # Fuse matrices for compression
        # Split matrices for decompression
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        num_heads = config.num_attention_heads
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        hidden_size = config.hidden_size

        if q_lora_rank is not None:
            q_a_proj_weight = get_weight(model_params,
                                         prefix + 'self_attn.q_a_proj', dtype)
            # Layer normalization
            q_a_layernorm_weight = get_weight(
                model_params,
                prefix + 'self_attn.q_a_layernorm',
                dtype,
            )

        kv_a_proj_with_mqa_weight = get_weight(
            model_params, prefix + 'self_attn.kv_a_proj_with_mqa', dtype)

        kv_a_layernorm_weight = get_weight(
            model_params,
            prefix + 'self_attn.kv_a_layernorm',
            dtype,
        )

        if q_lora_rank is not None:
            fused_a_weight = torch.cat(
                [q_a_proj_weight, kv_a_proj_with_mqa_weight],
                dim=0,
            )

            q_b_proj_weight = get_weight(
                model_params, prefix + 'self_attn.q_b_proj', dtype).unflatten(
                    0,
                    [
                        num_heads,
                        qk_nope_head_dim + qk_rope_head_dim,
                    ],
                )
        else:
            fused_a_weight = kv_a_proj_with_mqa_weight

            q_b_proj_weight = get_weight(
                model_params, prefix + 'self_attn.q_proj', dtype).unflatten(
                    0,
                    [
                        num_heads,
                        qk_nope_head_dim + qk_rope_head_dim,
                    ],
                )

        kv_b_proj_weight = get_weight(model_params,
                                      prefix + 'self_attn.kv_b_proj',
                                      dtype).unflatten(
                                          0,
                                          [
                                              num_heads,
                                              qk_nope_head_dim + v_head_dim,
                                          ],
                                      )

        o_proj_weight = get_weight(model_params, prefix + 'self_attn.o_proj',
                                   dtype)

        q_b_proj_weight = split_matrix_tp(
            q_b_proj_weight,
            mapping.tp_size,
            mapping.tp_rank,
            dim=0,
        )
        kv_b_proj_weight = split_matrix_tp(
            kv_b_proj_weight,
            mapping.tp_size,
            mapping.tp_rank,
            dim=0,
        )
        o_proj_weight = split_matrix_tp(
            o_proj_weight,
            mapping.tp_size,
            mapping.tp_rank,
            dim=1,
        )

        q_nope_weight, q_pe_weight = q_b_proj_weight.split(
            [qk_nope_head_dim, qk_rope_head_dim],
            dim=1,
        )
        k_nope_weight, v_weight = kv_b_proj_weight.split(
            [qk_nope_head_dim, v_head_dim],
            dim=1,
        )

        if q_lora_rank is None:
            q_b_proj_weight = q_b_proj_weight.reshape(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim) //
                mapping.tp_size, hidden_size)
        else:
            q_b_proj_weight = q_b_proj_weight.reshape(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim) //
                mapping.tp_size, q_lora_rank)

        kv_b_proj_weight = torch.concat([
            k_nope_weight.reshape(
                num_heads * qk_nope_head_dim // mapping.tp_size, kv_lora_rank),
            v_weight.reshape(num_heads * v_head_dim // mapping.tp_size,
                             kv_lora_rank)
        ],
                                        dim=0)

        # Fuse matrices for decompression
        fused_q_nope_weight = torch.einsum(
            'hdq,hdk->hkq',
            q_nope_weight,
            k_nope_weight,
        )
        fused_q_weight = torch.cat(
            [fused_q_nope_weight, q_pe_weight],
            dim=1,
        ).flatten(start_dim=0, end_dim=1)

        weights.update(
            get_tllm_linear_weight(fused_a_weight,
                                   trtllm_prefix + 'attention.fused_a.'))
        weights.update(
            get_tllm_linear_weight(kv_a_layernorm_weight,
                                   trtllm_prefix + 'attention.kv_a_layernorm.'))
        weights.update(
            get_param_weight(fused_q_weight,
                             trtllm_prefix + 'attention.fused_q_proj'))
        weights.update(
            get_param_weight(q_b_proj_weight,
                             trtllm_prefix + 'attention.q_b_proj'))
        weights.update(
            get_param_weight(kv_b_proj_weight,
                             trtllm_prefix + 'attention.kv_b_proj'))
        weights.update(
            get_tllm_linear_weight(o_proj_weight,
                                   trtllm_prefix + 'attention.dense.'))

        if q_lora_rank is not None:
            weights.update(
                get_tllm_linear_weight(
                    q_a_layernorm_weight,
                    trtllm_prefix + 'attention.q_a_layernorm.'))

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

            # mlp.experts.down_proj.weight
            moe_experts_down_proj_weights = get_weight(
                model_params, prefix + 'mlp.experts.down_proj', dtype)
            weights.update(
                get_tllm_linear_weight(moe_experts_down_proj_weights,
                                       trtllm_prefix + 'mlp.proj.'))
            # mlp.experts.up_gate.weight
            moe_experts_up_gate_proj_weights = get_weight(
                model_params, prefix + 'mlp.experts.up_gate_proj', dtype)
            weights.update(
                get_tllm_linear_weight(moe_experts_up_gate_proj_weights,
                                       trtllm_prefix + 'mlp.fc.'))
            # MOE hardcoded routing_input into trt.float32, please refer to moe.py line 397
            moe_experts_gate_weights = get_weight(model_params,
                                                  prefix + 'mlp.gate',
                                                  torch.float32)
            weights.update(
                get_tllm_linear_weight(moe_experts_gate_weights,
                                       trtllm_prefix + 'mlp.router.'))

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

                # mlp.shared_experts.gate_up_proj.weight
                weights.update(
                    get_tllm_linear_weight(
                        shared_moe_gate_up_proj_weights,
                        trtllm_prefix + 'mlp.shared_expert.fc.'))

                # mlp.shared_experts.down_proj.weight
                weights.update(
                    get_tllm_linear_weight(
                        shared_moe_down_proj_weights,
                        trtllm_prefix + 'mlp.shared_expert.proj.'))

        else:
            # Current MLP layer is only one, if it goes large consider to do fuse
            mlp_gate_weight = get_weight(model_params, prefix + 'mlp.up_proj',
                                         dtype)
            split_gate = split_matrix_tp(mlp_gate_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=0)
            weights.update(
                get_tllm_linear_weight(split_gate, trtllm_prefix + 'mlp.gate.'))

            mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                       dtype)
            split_fc = split_matrix_tp(mlp_fc_weight,
                                       mapping.tp_size,
                                       mapping.tp_rank,
                                       dim=0)
            weights.update(
                get_tllm_linear_weight(split_fc, trtllm_prefix + 'mlp.fc.'))

            mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                         dtype)
            split_proj = split_matrix_tp(mlp_proj_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=1)
            weights.update(
                get_tllm_linear_weight(split_proj, trtllm_prefix + 'mlp.proj.'))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + 'input_layernorm',
                                     dtype)
        weights[trtllm_prefix + 'input_layernorm.weight'] = input_ln_weight
        post_ln_weight = get_weight(model_params,
                                    prefix + 'post_attention_layernorm', dtype)
        weights[trtllm_prefix + 'post_layernorm.weight'] = post_ln_weight

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


def load_weights_from_hf_safetensors(model_dir,
                                     config,
                                     mapping,
                                     use_parallel_embedding=False,
                                     sharding_dim=0):
    logger.info('Loading weights from Huggingface safetensors...')
    weights = {}
    tik = time.time()
    import json
    import os

    import safetensors

    model_dir = model_dir if model_dir.endswith("/") else model_dir + "/"
    safetensors_map = {}
    has_safetensor_index_json = True
    try:
        with open(model_dir + "model.safetensors.index.json", 'r') as fr:
            sharding_map = json.load(fr)
        # safetensors_map structure:
        # key: e.g., model.layers.0.self_attn.q_a_proj.weight
        # value: e.g., model-00001-of-000055.safetensors
        # int(value[6:11]) -> safetensors_idx
        for k, v in sharding_map['weight_map'].items():
            safetensors_map[k] = int(v[6:11]) - 1
    except FileNotFoundError:
        has_safetensor_index_json = False
    # shard_files purpose to verify all .safetensors is aligned with .index.json
    shard_files = []
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            if has_safetensor_index_json and name not in sharding_map[
                    'weight_map'].values():
                continue
            shard_files.append(name)
    shard_files.sort()
    # Create all .safetensors memory address
    safetensors_ptrs = [
        safetensors.safe_open(model_dir + shard_file,
                              framework="pt",
                              device="cpu") for shard_file in shard_files
    ]

    moe_config = MoeConfig(
        num_experts=config.moe.num_experts,
        shared_expert_intermediate_size=config.moe.
        shared_expert_intermediate_size,
        top_k=config.moe.top_k,
        normalization_mode=config.moe.normalization_mode,
        device_limited_n_group=config.moe.device_limited_n_group,
        device_limited_topk_group=config.moe.device_limited_topk_group,
        device_limited_routed_scaling_factor=config.moe.
        device_limited_routed_scaling_factor)

    torch_dtype = str_dtype_to_torch(config.dtype)

    def load_weights(key, dtype):
        assert key in safetensors_map, f"'{key}' not found in safetensors_map"
        ptr_idx = safetensors_map[key]

        assert key in safetensors_ptrs[ptr_idx].keys(
        ), f"'{key}' not found in safetensors file {ptr_idx}"
        tensor_slice = safetensors_ptrs[ptr_idx].get_slice(key)
        tensor_slice.get_shape()
        res = tensor_slice[:]
        if res.dtype != dtype:
            res = res.to(dtype)
        return res.contiguous().detach().cpu()

    layers_range = mapping.pp_layers(config.num_hidden_layers)

    def convert_layer(l):
        prefix = f'model.layers.{l}.'
        trtllm_prefix = f'transformer.layers.{l - layers_range[0]}.'
        # Fuse matrices for compression
        # Split matrices for decompression
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        num_heads = config.num_attention_heads
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        hidden_size = config.hidden_size

        # MLA:
        # q_lora_rank used for different deepseek-v2 and deepseek-v2-lite
        if q_lora_rank is not None:
            q_a_proj_weight = load_weights(prefix + 'self_attn.q_a_proj.weight',
                                           torch_dtype)
            q_a_layernorm_weight = load_weights(
                prefix + 'self_attn.q_a_layernorm.weight', torch_dtype)

        kv_a_proj_with_mqa_weight = load_weights(
            prefix + 'self_attn.kv_a_proj_with_mqa.weight', torch_dtype)
        kv_a_layernorm_weight = load_weights(
            prefix + 'self_attn.kv_a_layernorm.weight', torch_dtype)

        if q_lora_rank is not None:
            fused_a_weight = torch.cat(
                [q_a_proj_weight, kv_a_proj_with_mqa_weight], dim=0)
            q_b_proj_weight = load_weights(
                prefix + 'self_attn.q_b_proj.weight', torch_dtype).unflatten(
                    0, [num_heads, qk_nope_head_dim + qk_rope_head_dim])
        else:
            fused_a_weight = kv_a_proj_with_mqa_weight
            q_b_proj_weight = load_weights(
                prefix + 'self_attn.q_proj.weight', torch_dtype).unflatten(
                    0, [num_heads, qk_nope_head_dim + qk_rope_head_dim])

        kv_b_proj_weight = load_weights(
            prefix + 'self_attn.kv_b_proj.weight',
            torch_dtype).unflatten(0,
                                   [num_heads, qk_nope_head_dim + v_head_dim])
        o_proj_weight = load_weights(prefix + 'self_attn.o_proj.weight',
                                     torch_dtype)

        q_b_proj_weight = split_matrix_tp(q_b_proj_weight,
                                          mapping.tp_size,
                                          mapping.tp_rank,
                                          dim=0)
        kv_b_proj_weight = split_matrix_tp(kv_b_proj_weight,
                                           mapping.tp_size,
                                           mapping.tp_rank,
                                           dim=0)
        o_proj_weight = split_matrix_tp(o_proj_weight,
                                        mapping.tp_size,
                                        mapping.tp_rank,
                                        dim=1)

        q_nope_weight, q_pe_weight = q_b_proj_weight.split(
            [qk_nope_head_dim, qk_rope_head_dim], dim=1)
        k_nope_weight, v_weight = kv_b_proj_weight.split(
            [qk_nope_head_dim, v_head_dim], dim=1)

        if q_lora_rank is None:
            q_b_proj_weight = q_b_proj_weight.reshape(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim) //
                mapping.tp_size, hidden_size)
        else:
            q_b_proj_weight = q_b_proj_weight.reshape(
                num_heads * (qk_nope_head_dim + qk_rope_head_dim) //
                mapping.tp_size, q_lora_rank)

        kv_b_proj_weight = torch.concat([
            k_nope_weight.reshape(
                num_heads * qk_nope_head_dim // mapping.tp_size, kv_lora_rank),
            v_weight.reshape(num_heads * v_head_dim // mapping.tp_size,
                             kv_lora_rank)
        ],
                                        dim=0)

        # Fuse matrices for decompression
        fused_q_nope_weight = torch.einsum('hdq,hdk->hkq', q_nope_weight,
                                           k_nope_weight)
        fused_q_weight = torch.cat([fused_q_nope_weight, q_pe_weight],
                                   dim=1).flatten(start_dim=0, end_dim=1)

        weights.update(
            get_tllm_linear_weight(fused_a_weight,
                                   trtllm_prefix + 'attention.fused_a.'))
        weights.update(
            get_tllm_linear_weight(kv_a_layernorm_weight,
                                   trtllm_prefix + 'attention.kv_a_layernorm.'))
        weights.update(
            get_param_weight(fused_q_weight,
                             trtllm_prefix + 'attention.fused_q_proj'))
        weights.update(
            get_param_weight(q_b_proj_weight,
                             trtllm_prefix + 'attention.q_b_proj'))
        weights.update(
            get_param_weight(kv_b_proj_weight,
                             trtllm_prefix + 'attention.kv_b_proj'))
        weights.update(
            get_tllm_linear_weight(o_proj_weight,
                                   trtllm_prefix + 'attention.dense.'))

        if q_lora_rank is not None:
            weights.update(
                get_tllm_linear_weight(
                    q_a_layernorm_weight,
                    trtllm_prefix + 'attention.q_a_layernorm.'))

        # MOE:
        if moe_config.has_moe() and l > 0:
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)
            expert_weight_dict = {}
            for suffix in ["gate_proj", "down_proj", "up_proj"]:
                expert_weight_dict[f'model.layers.{l}.mlp.experts.{suffix}.weight'] = \
                torch.stack([load_weights(prefix + f'mlp.experts.{expert}.{suffix}.weight', torch_dtype) for expert in rank_experts])

            gate_proj = expert_weight_dict[
                f'model.layers.{l}.mlp.experts.gate_proj.weight']
            down_proj = expert_weight_dict[
                f'model.layers.{l}.mlp.experts.down_proj.weight']
            up_proj = expert_weight_dict[
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

            expert_weight_dict[
                f'model.layers.{l}.mlp.experts.up_gate_proj.weight'] = torch.concat(
                    [up_proj, gate_proj], dim=-2)
            expert_weight_dict[
                f'model.layers.{l}.mlp.experts.down_proj.weight'] = down_proj

            # mlp.experts.down_proj.weight
            moe_experts_down_proj_weights = expert_weight_dict[
                f'model.layers.{l}.mlp.experts.down_proj.weight']
            weights.update(
                get_tllm_linear_weight(moe_experts_down_proj_weights,
                                       trtllm_prefix + 'mlp.proj.'))

            # mlp.experts.up_gate.weight
            moe_experts_up_gate_proj_weights = expert_weight_dict[
                f'model.layers.{l}.mlp.experts.up_gate_proj.weight']
            weights.update(
                get_tllm_linear_weight(moe_experts_up_gate_proj_weights,
                                       trtllm_prefix + 'mlp.fc.'))

            # MOE hardcoded routing_input into trt.float32, please refer to moe.py line 397
            moe_experts_gate_weights = load_weights(prefix + 'mlp.gate.weight',
                                                    torch.float32)
            weights.update(
                get_tllm_linear_weight(moe_experts_gate_weights,
                                       trtllm_prefix + 'mlp.router.'))

            if moe_config.shared_expert_intermediate_size > 0:
                shared_moe_up_proj_weights = load_weights(
                    prefix + 'mlp.shared_experts.up_proj.weight', torch_dtype)
                shared_moe_up_proj_weights = split_matrix_tp(
                    shared_moe_up_proj_weights,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=0)
                shared_moe_down_proj_weights = load_weights(
                    prefix + 'mlp.shared_experts.down_proj.weight', torch_dtype)
                shared_moe_down_proj_weights = split_matrix_tp(
                    shared_moe_down_proj_weights,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=1)
                shared_moe_gate_proj_weights = load_weights(
                    prefix + 'mlp.shared_experts.gate_proj.weight', torch_dtype)
                shared_moe_gate_proj_weights = split_matrix_tp(
                    shared_moe_gate_proj_weights,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=0)
                shared_moe_gate_up_proj_weights = torch.concat(
                    [shared_moe_up_proj_weights, shared_moe_gate_proj_weights],
                    dim=-2)

                # mlp.shared_experts.gate_up_proj.weight
                weights.update(
                    get_tllm_linear_weight(
                        shared_moe_gate_up_proj_weights,
                        trtllm_prefix + 'mlp.shared_expert.fc.'))

                # mlp.shared_experts.down_proj.weight
                weights.update(
                    get_tllm_linear_weight(
                        shared_moe_down_proj_weights,
                        trtllm_prefix + 'mlp.shared_expert.proj.'))

        else:
            # Current MLP layer is only one (layer 0), if it goes large consider to do fuse
            mlp_gate_weight = load_weights(prefix + 'mlp.up_proj.weight',
                                           torch_dtype)
            split_gate = split_matrix_tp(mlp_gate_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=0)
            weights.update(
                get_tllm_linear_weight(split_gate, trtllm_prefix + 'mlp.gate.'))

            mlp_fc_weight = load_weights(prefix + 'mlp.gate_proj.weight',
                                         torch_dtype)
            split_fc = split_matrix_tp(mlp_fc_weight,
                                       mapping.tp_size,
                                       mapping.tp_rank,
                                       dim=0)
            weights.update(
                get_tllm_linear_weight(split_fc, trtllm_prefix + 'mlp.fc.'))

            mlp_proj_weight = load_weights(prefix + 'mlp.down_proj.weight',
                                           torch_dtype)
            split_proj = split_matrix_tp(mlp_proj_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=1)
            weights.update(
                get_tllm_linear_weight(split_proj, trtllm_prefix + 'mlp.proj.'))

        # Layer norms do not use tensor parallelism
        input_ln_weight = load_weights(prefix + 'input_layernorm.weight',
                                       torch_dtype)
        weights[trtllm_prefix + 'input_layernorm.weight'] = input_ln_weight
        post_ln_weight = load_weights(
            prefix + 'post_attention_layernorm.weight', torch_dtype)
        weights[trtllm_prefix + 'post_layernorm.weight'] = post_ln_weight

    for l in layers_range:
        convert_layer(l)
        release_gc()

    v = load_weights('model.embed_tokens.weight', torch_dtype)
    if use_parallel_embedding:
        v = split_matrix_tp(v,
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=config.embedding_sharding_dim)
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v
    lm_head_weights = load_weights('lm_head.weight', torch_dtype)

    if mapping.is_last_pp_rank():
        if config.vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(config.vocab_size,
                                               mapping.tp_size)
            pad_width = vocab_size_padded - config.vocab_size
            lm_head_weights = torch.nn.functional.pad(lm_head_weights,
                                                      (0, 0, 0, pad_width),
                                                      'constant', 0)
        weights['lm_head.weight'] = split_matrix_tp(lm_head_weights,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=0)
    ln_f_w = load_weights('model.norm.weight', torch_dtype)
    weights['transformer.ln_f.weight'] = ln_f_w
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights
