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

from ..._utils import pad_vocab_size, release_gc
from ...mapping import Mapping
from ..convert_utils import get_tllm_linear_weight

# `Override num_hidden_layers` used for reduce number of hidden layers in DeepseekV2ForCausalLM for debug purpose
OVERRIDE_HIDDEN_LAYERS = None  # 2


## Convert config parameters to dict
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
        moe_intermediate_size=config['moe_inter_size'],
        num_shared_experts=config['moe_num_shared_experts'],
        top_k=config['moe_top_k'],
        normalization_mode=config['moe_renorm_mode'],
        device_limited_n_group=config['moe_n_group'],
        device_limited_topk_group=config['moe_topk_group'],
        device_limited_routed_scaling_factor=config['moe_routed_scaling_factor']
    )
    moe_config.validate()

    return config


## Get HF model
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
                                                     torch_dtype='auto',
                                                     trust_remote_code=True)
    else:
        #Deepseek-v2 236B parameters with FP16 dtype need at least 472G GPU memory
        #(official suggest at least 8x80G GPUs, see https://huggingface.co/deepseek-ai/DeepSeek-V2)
        #'max_memory' should be set based on memory property of GPUs
        if torch.cuda.get_device_properties(
                0
        ).total_memory > 80000000000 and torch.cuda.get_device_properties(
                0).total_memory < 90000000000:
            max_memory = {i: "76GB" for i in range(8)}
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                config=hf_config,
                device_map='sequential',
                max_memory=max_memory,
                torch_dtype='auto',
                trust_remote_code=True)
        elif torch.cuda.get_device_properties(0).total_memory >= 90000000000:
            model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                         config=hf_config,
                                                         device_map='auto',
                                                         torch_dtype='auto',
                                                         trust_remote_code=True)
        else:
            assert torch.cuda.get_device_properties(
                0
            ).total_memory >= 80000000000, "deepseek v2 loading requires per GPU memory above 80G"

    return model


## Prepare weights for TP
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
                       sharding_dim=0,
                       share_embedding_table=False):

    weights = {}
    tik = time.time()
    mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    moe_config = MoeConfig(
        num_experts=config['moe_num_experts'],
        moe_intermediate_size=config['moe_inter_size'],
        num_shared_experts=config['moe_num_shared_experts'],
        top_k=config['moe_top_k'],
        normalization_mode=config['moe_renorm_mode'],
        device_limited_n_group=config['moe_n_group'],
        device_limited_topk_group=config['moe_topk_group'],
        device_limited_routed_scaling_factor=config['moe_routed_scaling_factor']
    )

    layers_range = mapping.pp_layers(config['num_hidden_layers'])

    def convert_layer(l):
        prefix = f'model.layers.{l}.'
        print(prefix)
        trtllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        # Fuse matrices for compression
        q_a_proj_weight = get_weight(model_params,
                                     prefix + 'self_attn.q_a_proj', dtype)
        kv_a_proj_with_mqa_weight = get_weight(
            model_params, prefix + 'self_attn.kv_a_proj_with_mqa', dtype)

        fused_a_weight = torch.cat(
            [q_a_proj_weight, kv_a_proj_with_mqa_weight],
            dim=0,
        )

        # Layer normalization
        q_a_layernorm_weight = get_weight(
            model_params,
            prefix + 'self_attn.q_a_layernorm',
            dtype,
        )
        kv_a_layernorm_weight = get_weight(
            model_params,
            prefix + 'self_attn.kv_a_layernorm',
            dtype,
        )

        # Split matrices for decompression
        q_lora_rank = config['q_lora_rank']
        kv_lora_rank = config['kv_lora_rank']
        num_heads = config['num_attention_heads']
        qk_nope_head_dim = config['qk_nope_head_dim']
        qk_rope_head_dim = config['qk_rope_head_dim']
        v_head_dim = config['v_head_dim']

        q_b_proj_weight = get_weight(
            model_params, prefix + 'self_attn.q_b_proj', dtype).unflatten(
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
                                   trtllm_prex + 'attention.fused_a.'))
        weights.update(
            get_tllm_linear_weight(q_a_layernorm_weight,
                                   trtllm_prex + 'attention.q_a_layernorm.'))
        weights.update(
            get_tllm_linear_weight(kv_a_layernorm_weight,
                                   trtllm_prex + 'attention.kv_a_layernorm.'))
        weights.update(
            get_param_weight(fused_q_weight,
                             trtllm_prex + 'attention.fused_q_proj'))
        weights.update(
            get_param_weight(q_b_proj_weight,
                             trtllm_prex + 'attention.q_b_proj'))
        weights.update(
            get_param_weight(kv_b_proj_weight,
                             trtllm_prex + 'attention.kv_b_proj'))
        weights.update(
            get_tllm_linear_weight(o_proj_weight,
                                   trtllm_prex + 'attention.dense.'))

        # #OPT MLA
        # head_num = config["num_attention_heads"]  #128
        # head_size = config["qk_nope_head_dim"]  #128
        # c_k = config["kv_lora_rank"]  #512
        # c_q = config["q_lora_rank"]  #1536

        # #kv_b:[32768,512]=[128,128*2=256,512]
        # kv_b_proj_weight = kv_b_proj_weight.view(head_num, head_size * 2, c_k)
        # #kv_b_proj_weight:[32768,512], k_proj: [128,128,512], v_proj: [128,128,512]
        # k_proj, v_proj = torch.split(kv_b_proj_weight, [head_size, head_size],
        #                              dim=1)
        # #k_proj:[128*128=16384,512], v_proj:[128*128=16384,512]
        # k_proj = k_proj.reshape(head_num * head_size, c_k)
        # v_proj = v_proj.reshape(head_num * head_size, c_k)
        # #kv_b:[16384+16384=32768,512]
        # # kv_b = torch.concat([k_proj, v_proj], dim=0)

        # #q_b_proj_weight:[24576,1536],split_q:[128,128,1536], split_rope:[128,64,1526]
        # split_q, split_rope = torch.split(q_b_proj_weight.view(
        #     head_num, head_size + rope_dim, c_q), [head_size, rope_dim],
        #                                   dim=1)
        # #k_proj:[128,128,512]=[128,512,128]
        # k_trans = k_proj.view(head_num, head_size, c_k).transpose(1, 2)
        # #k_trans:[128,512,128],split_q:[128,128,1536],absorbed_gemm:[128,512,1536]
        # absorbed_gemm = torch.bmm(k_trans, split_q)
        # #absorbed_gemm:[128,512,1536],split_rope:[128,64,1536],final_w:[128*(512+64)]
        # final_w = torch.concat([absorbed_gemm, split_rope],
        #                        dim=1).reshape(head_num * (c_k + rope_dim), c_q)

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
                get_tllm_linear_weight(moe_experts_down_proj_weights,
                                       trtllm_prex + 'mlp.moe.proj.'))
            ## mlp.experts.up_gate.weight
            moe_experts_up_gate_proj_weights = get_weight(
                model_params, prefix + 'mlp.experts.up_gate_proj', dtype)
            weights.update(
                get_tllm_linear_weight(moe_experts_up_gate_proj_weights,
                                       trtllm_prex + 'mlp.moe.fc.'))
            ## MOE hardcoded routing_input into trt.float32, please refer to moe.py line 397
            moe_experts_gate_weights = get_weight(model_params,
                                                  prefix + 'mlp.gate',
                                                  torch.float32)
            weights.update(
                get_tllm_linear_weight(moe_experts_gate_weights,
                                       trtllm_prex + 'mlp.moe.router.'))

            if moe_config.num_shared_experts > 0:
                ## mlp.shared_experts.gate_proj.weight
                shared_moe_gate_proj_weights = get_weight(
                    model_params, prefix + 'mlp.shared_experts.gate_proj',
                    dtype)
                split_v = split_matrix_tp(shared_moe_gate_proj_weights,
                                          mapping.tp_size,
                                          mapping.tp_rank,
                                          dim=0)
                weights.update(
                    get_tllm_linear_weight(
                        split_v, trtllm_prex + 'mlp.shared_experts.fc.'))

                # mlp.shared_experts.down_proj.weight
                shared_moe_down_proj_weights = get_weight(
                    model_params, prefix + 'mlp.shared_experts.down_proj',
                    dtype)
                split_down = split_matrix_tp(shared_moe_down_proj_weights,
                                             mapping.tp_size,
                                             mapping.tp_rank,
                                             dim=1)
                weights.update(
                    get_tllm_linear_weight(
                        split_down, trtllm_prex + 'mlp.shared_experts.proj.'))

                ## mlp.shared_experts.up_proj.weight
                shared_moe_up_gate_proj_weights = get_weight(
                    model_params, prefix + 'mlp.shared_experts.up_proj', dtype)
                split_up_gate = split_matrix_tp(shared_moe_up_gate_proj_weights,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=0)
                weights.update(
                    get_tllm_linear_weight(
                        split_up_gate,
                        trtllm_prex + 'mlp.shared_experts.gate.'))

        else:
            ## Current MLP layer is only one, if it goes large consider to do fuse
            mlp_gate_weight = get_weight(model_params, prefix + 'mlp.up_proj',
                                         dtype)
            split_gate = split_matrix_tp(mlp_gate_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=0)
            weights.update(
                get_tllm_linear_weight(split_gate, trtllm_prex + 'mlp.gate.'))

            mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                       dtype)
            split_fc = split_matrix_tp(mlp_fc_weight,
                                       mapping.tp_size,
                                       mapping.tp_rank,
                                       dim=0)
            weights.update(
                get_tllm_linear_weight(split_fc, trtllm_prex + 'mlp.fc.'))

            mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                         dtype)
            split_proj = split_matrix_tp(mlp_proj_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=1)
            weights.update(
                get_tllm_linear_weight(split_proj, trtllm_prex + 'mlp.proj.'))

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
            if config['vocab_size'] % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(config['vocab_size'],
                                                   mapping.tp_size)
                pad_width = vocab_size_padded - config['vocab_size']
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
        if config['vocab_size'] % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(config['vocab_size'],
                                               mapping.tp_size)
            pad_width = vocab_size_padded - config['vocab_size']
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
