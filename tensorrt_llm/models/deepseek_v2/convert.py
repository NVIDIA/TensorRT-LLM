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

from ..._utils import pad_vocab_size, release_gc
from ...layers import MoeConfig
from ...quantization import QuantAlgo
from ..convert_utils import get_tllm_linear_weight
from .config import DeepSeekV2Config

# `Override num_hidden_layers` used for reduce number of hidden layers in DeepseekV2ForCausalLM for debug purpose
OVERRIDE_HIDDEN_LAYERS = None  # 2


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

        model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                     config=hf_config,
                                                     device_map=device_map,
                                                     max_memory=max_memory,
                                                     torch_dtype='auto',
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


def load_weights_from_hf_model(hf_model,
                               config: DeepSeekV2Config,
                               use_parallel_embedding=False,
                               sharding_dim=0):
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None
    use_gemm_woq_plugin = True

    weights = {}
    tik = time.time()
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)

    mapping = config.mapping
    moe_config = config.moe
    first_k_dense_replace = config.first_k_dense_replace
    layers_range = mapping.pp_layers(config.num_hidden_layers)

    def convert_layer(l):
        prefix = f'model.layers.{l}.'
        trtllm_prex = f'transformer.layers.{l - layers_range[0]}.'
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
                                   trtllm_prex + 'attention.fused_a.'))
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
            get_tllm_linear_weight(
                o_proj_weight,
                trtllm_prex + 'attention.dense.',
                use_weight_only=use_weight_only,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                use_gemm_woq_plugin=use_gemm_woq_plugin))

        if q_lora_rank is not None:
            weights.update(
                get_tllm_linear_weight(q_a_layernorm_weight, trtllm_prex +
                                       'attention.q_a_layernorm.'))

        if moe_config.has_moe() and l >= first_k_dense_replace:
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
                get_tllm_linear_weight(
                    moe_experts_down_proj_weights,
                    trtllm_prex + 'mlp.proj.',
                    use_weight_only=use_weight_only,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                    use_gemm_woq_plugin=use_gemm_woq_plugin))
            ## mlp.experts.up_gate.weight
            moe_experts_up_gate_proj_weights = get_weight(
                model_params, prefix + 'mlp.experts.up_gate_proj', dtype)
            weights.update(
                get_tllm_linear_weight(
                    moe_experts_up_gate_proj_weights,
                    trtllm_prex + 'mlp.fc.',
                    use_weight_only=use_weight_only,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                    use_gemm_woq_plugin=use_gemm_woq_plugin))
            ## MOE hardcoded routing_input into trt.float32, please refer to moe.py line 397
            moe_experts_gate_weights = get_weight(model_params,
                                                  prefix + 'mlp.gate',
                                                  torch.float32)
            weights.update(
                get_tllm_linear_weight(moe_experts_gate_weights,
                                       trtllm_prex + 'mlp.router.'))

            if moe_config.topk_method == MoeConfig.TopKMethod.NOAUX_TC:
                e_score_correction_bias = get_weight(
                    model_params, prefix + 'mlp.gate.e_score_correction_bias',
                    torch.float32, '')
                weights.update(
                    get_param_weight(
                        e_score_correction_bias,
                        trtllm_prex + 'mlp.e_score_correction_bias'))

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
                    get_tllm_linear_weight(
                        shared_moe_gate_up_proj_weights,
                        trtllm_prex + 'mlp.shared_expert.fc.',
                        use_weight_only=use_weight_only,
                        plugin_weight_only_quant_type=
                        plugin_weight_only_quant_type,
                        use_gemm_woq_plugin=use_gemm_woq_plugin))

                ## mlp.shared_experts.down_proj.weight
                weights.update(
                    get_tllm_linear_weight(
                        shared_moe_down_proj_weights,
                        trtllm_prex + 'mlp.shared_expert.proj.',
                        use_weight_only=use_weight_only,
                        plugin_weight_only_quant_type=
                        plugin_weight_only_quant_type,
                        use_gemm_woq_plugin=use_gemm_woq_plugin))

        else:
            ## Current MLP layer is only one, if it goes large consider to do fuse
            mlp_gate_weight = get_weight(model_params, prefix + 'mlp.up_proj',
                                         dtype)
            split_gate = split_matrix_tp(mlp_gate_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=0)
            weights.update(
                get_tllm_linear_weight(
                    split_gate,
                    trtllm_prex + 'mlp.gate.',
                    use_weight_only=use_weight_only,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                    use_gemm_woq_plugin=use_gemm_woq_plugin))

            mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                       dtype)
            split_fc = split_matrix_tp(mlp_fc_weight,
                                       mapping.tp_size,
                                       mapping.tp_rank,
                                       dim=0)
            weights.update(
                get_tllm_linear_weight(
                    split_fc,
                    trtllm_prex + 'mlp.fc.',
                    use_weight_only=use_weight_only,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                    use_gemm_woq_plugin=use_gemm_woq_plugin))

            mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                         dtype)
            split_proj = split_matrix_tp(mlp_proj_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=1)
            weights.update(
                get_tllm_linear_weight(
                    split_proj,
                    trtllm_prex + 'mlp.proj.',
                    use_weight_only=use_weight_only,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                    use_gemm_woq_plugin=use_gemm_woq_plugin))

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
