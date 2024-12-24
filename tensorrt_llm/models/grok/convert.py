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
from pathlib import Path
from typing import Optional

import jax
import numpy as np
import torch
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

from ..._utils import pad_vocab_size, release_gc
from ...layers import MoeConfig
from ...quantization import QuantAlgo
from ..convert_utils import split
from ..modeling_utils import PretrainedConfig, QuantConfig, optimize_model


def get_jax_weight(config, prefix, dtype, postfix='.weight', key_name='scale'):

    return torch.as_tensor((config[prefix + postfix][key_name])._value,
                           dtype=dtype).T


def get_jax_weight_scale_tp(params, key, rank):
    jax_obj = params[key]['w']
    jax_scales = jax.device_put(jax_obj.scales, device=jax.devices('gpu')[rank])
    torch_scales = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(jax_scales))
    return torch.as_tensor(
        np.asarray(jax_obj.weight.addressable_shards[rank].data)), torch_scales


def get_jax_weight_scale(params, key):
    jax_obj = params[key]['w']

    jax_scales = jax.device_put(jax_obj.scales, device=jax.devices('cpu')[0])

    torch_scales = torch_dlpack.from_dlpack(
        jax_dlpack.to_dlpack(jax_scales, copy=False))
    return torch.as_tensor(np.asarray(jax_obj.weight),
                           dtype=torch.int8), torch_scales


def get_tllm_linear_weight(
    weight,
    torch_weight_scales,
    prefix,
    plugin_weight_only_quant_type=torch.int8,
    postfix='weight',
):
    results = {}
    processed_weight = torch.ops.trtllm.preprocess_weights_for_mixed_gemm(
        weight if weight.is_contiguous() else weight.contiguous(),
        plugin_weight_only_quant_type, torch.bfloat16)
    results[prefix + postfix] = processed_weight

    results[prefix + 'per_channel_scale'] = torch_weight_scales.contiguous()

    return results


def convert_grok(hf_model,
                 config,
                 mapping,
                 vocab_size=32000,
                 dtype='float32',
                 use_parallel_embedding=False,
                 sharding_dim=0,
                 use_weight_only=False,
                 use_gemm_woq_plugin=False,
                 plugin_weight_only_quant_type=torch.int8,
                 moe_config=None):

    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = hf_model
    dtype = getattr(torch, dtype)

    config['num_attention_heads']
    config['hidden_size']

    layers_range = mapping.pp_layers(config['num_hidden_layers'])

    def convert_layer(l):
        prefix = f'transformer/decoder_layer_{l}/'
        print(prefix)
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'

        wq, q_scale = get_jax_weight_scale_tp(
            model_params, prefix + 'multi_head_attention/query',
            mapping.tp_rank)
        wk, k_scale = get_jax_weight_scale_tp(
            model_params, prefix + 'multi_head_attention/key', mapping.tp_rank)
        wv, v_scale = get_jax_weight_scale_tp(
            model_params, prefix + 'multi_head_attention/value',
            mapping.tp_rank)

        qs = split(q_scale, mapping.tp_size, mapping.tp_rank, dim=1)
        ks = split(k_scale, mapping.tp_size, mapping.tp_rank, dim=1)
        vs = split(v_scale, mapping.tp_size, mapping.tp_rank, dim=1)
        split_v = torch.concat((wq, wk, wv), dim=1)
        scale_v = torch.concat((qs, ks, vs), dim=1)

        weights.update(
            get_tllm_linear_weight(split_v, scale_v.squeeze(),
                                   tllm_prex + 'attention.qkv.',
                                   plugin_weight_only_quant_type))

        attn_dense_weight, attn_dense_scales = get_jax_weight_scale_tp(
            model_params, prefix + 'multi_head_attention/linear',
            mapping.tp_rank)

        split_scales = split(attn_dense_scales,
                             tensor_parallel,
                             mapping.tp_rank,
                             dim=0)

        weights.update(
            get_tllm_linear_weight(attn_dense_weight, split_scales.squeeze(),
                                   tllm_prex + 'attention.dense.',
                                   plugin_weight_only_quant_type))
        if mapping.moe_ep_size > 1:
            w3, s3 = get_jax_weight_scale(
                model_params, f'transformer/decoder_layer_{l}/moe/linear_v')

            w2, s2 = get_jax_weight_scale(
                model_params, f'transformer/decoder_layer_{l}/moe/linear_1')

            w1, s1 = get_jax_weight_scale(
                model_params, f'transformer/decoder_layer_{l}/moe/linear')

            # moe expert parallel
            w3_split = split(w3,
                             mapping.moe_ep_size,
                             mapping.moe_ep_rank,
                             dim=0)
            w2_split = split(w2,
                             mapping.moe_ep_size,
                             mapping.moe_ep_rank,
                             dim=0)
            w1_split = split(w1,
                             mapping.moe_ep_size,
                             mapping.moe_ep_rank,
                             dim=0)

            s3_split = split(s3,
                             mapping.moe_ep_size,
                             mapping.moe_ep_rank,
                             dim=0)
            s2_split = split(s2,
                             mapping.moe_ep_size,
                             mapping.moe_ep_rank,
                             dim=0)
            s1_split = split(s1,
                             mapping.moe_ep_size,
                             mapping.moe_ep_rank,
                             dim=0)

            # moe tensor parallel
            w3_split = split(w3_split,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=2)
            w2_split = split(w2_split,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=1)
            w1_split = split(w1_split,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=2)

            s3_split = split(s3_split,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=2)
            s2_split = split(s2_split,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=1)
            s1_split = split(s1_split,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=2)
        else:
            w3_split, s3 = get_jax_weight_scale_tp(
                model_params, f'transformer/decoder_layer_{l}/moe/linear_v',
                mapping.tp_rank)

            w2_split, s2 = get_jax_weight_scale_tp(
                model_params, f'transformer/decoder_layer_{l}/moe/linear_1',
                mapping.tp_rank)

            w1_split, s1 = get_jax_weight_scale_tp(
                model_params, f'transformer/decoder_layer_{l}/moe/linear',
                mapping.tp_rank)

            s3_split = split(s3,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=2)
            s2_split = split(s2,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=1)
            s1_split = split(s1,
                             mapping.moe_tp_size,
                             mapping.moe_tp_rank,
                             dim=2)
        weights.update(
            get_tllm_linear_weight(w2_split,
                                   s2_split.reshape(moe_config.num_experts, -1),
                                   tllm_prex + 'mlp.proj.',
                                   plugin_weight_only_quant_type))

        weights.update(
            get_tllm_linear_weight(
                torch.concat([w3_split, w1_split], dim=-1),
                torch.concat([s3_split, s1_split],
                             dim=-1).reshape(moe_config.num_experts, -1),
                tllm_prex + 'mlp.fc.',
                plugin_weight_only_quant_type,
            ))

        moe_experts_gate_weights = get_jax_weight(model_params,
                                                  prefix + 'router',
                                                  torch.float32,
                                                  postfix='',
                                                  key_name='w').contiguous()

        weights[tllm_prex + 'mlp.router.weight'] = moe_experts_gate_weights

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_jax_weight(model_params,
                                         prefix + 'rms_norm',
                                         dtype,
                                         postfix='')
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_attn_weight = get_jax_weight(model_params,
                                          prefix + 'rms_norm_1',
                                          dtype,
                                          postfix='')
        weights[tllm_prex + 'post_attn_layernorm.weight'] = post_attn_weight

        post_ln_weight = get_jax_weight(model_params,
                                        prefix + 'rms_norm_2',
                                        dtype,
                                        postfix='')
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

        post_mlp_weight = get_jax_weight(model_params,
                                         prefix + 'rms_norm_3',
                                         dtype,
                                         postfix='')
        weights[tllm_prex + 'post_mlp_layernorm.weight'] = post_mlp_weight

    for l in layers_range:
        convert_layer(l)
        release_gc()

    v = get_jax_weight(model_params,
                       'language_model/in_out_embed',
                       dtype,
                       postfix='',
                       key_name='embeddings').T
    tie_word_embeddings = config['tie_word_embeddings']
    if tie_word_embeddings:
        # lm_head.weight has the same weights as embedding
        if mapping.is_last_pp_rank():
            if vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size

                v = torch.nn.functional.pad(v, (0, pad_width, 0, 0), 'constant',
                                            0)
            weights['lm_head.weight'] = split(v, mapping.tp_size,
                                              mapping.tp_rank)

    if use_parallel_embedding:
        v = split(v, mapping.tp_size, mapping.tp_rank, dim=sharding_dim)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v

    ln_f_w = get_jax_weight(model_params,
                            'language_model/rms_norm',
                            dtype,
                            postfix='')
    weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def create_config_from_xai(dtype,
                           mapping,
                           quantization: QuantConfig = None,
                           override_fields: dict = {}):
    config = {}
    hf_config = {
        "architectures": ["Grok1ModelForCausalLM"],
        "vocab_size": 131072,
        "hidden_size": 6144,
        "intermediate_size": 32768,
        "num_hidden_layers": 64,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "attn_output_multiplier": 0.08838834764831845,
        "embedding_multiplier_scale": 78.38367176906169,
        "output_multiplier_scale": 0.5773502691896257,
        "max_attn_value": 30.0,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": True,
        "num_experts_per_tok": 2,
        "num_experts": 8,
        "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.35.0"
    }
    # same for from_meta and from_cli_args
    n_head = hf_config['num_attention_heads']
    inter_size = hf_config['intermediate_size']
    n_layer = hf_config['num_hidden_layers']
    n_embd = hf_config['hidden_size']
    n_kv_head = hf_config['num_key_value_heads']
    rms_norm_eps = hf_config['rms_norm_eps']
    vocab_size = hf_config['vocab_size']
    n_positions = hf_config['max_position_embeddings']
    hidden_act = 'geglu'
    config['rotary_scaling'] = None
    rotary_base = 10000.0

    config[
        'moe_normalization_mode'] = MoeConfig.ExpertScaleNormalizationMode.NONE

    moe_num_experts = hf_config['num_experts']

    moe_top_k = hf_config['num_experts_per_tok']

    attn_output_multiplier = hf_config['attn_output_multiplier']
    embedding_multiplier_scale = hf_config['embedding_multiplier_scale']

    output_multiplier_scale = hf_config['output_multiplier_scale']
    max_attn_value = hf_config['max_attn_value']

    architecture = hf_config['architectures'][0]

    attn_bias = False

    config.update({
        'architecture':
        architecture,
        'dtype':
        dtype,
        'logits_dtype':
        'float32',
        'num_hidden_layers':
        n_layer,
        'num_attention_heads':
        n_head,
        'hidden_size':
        n_embd,
        'intermediate_size':
        inter_size,
        'num_key_value_heads':
        n_kv_head,
        'vocab_size':
        vocab_size,
        'position_embedding_type':
        'rope_gpt_neox',
        'max_position_embeddings':
        n_positions,
        'hidden_act':
        hidden_act,
        'rotary_base':
        rotary_base,
        'norm_epsilon':
        rms_norm_eps,
        'moe_num_experts':
        moe_num_experts,
        'moe_top_k':
        moe_top_k,
        'moe_normalization_mode':
        MoeConfig.ExpertScaleNormalizationMode.NONE,
        #TODO: should have directly map from the Mapping object to the TRT-LLM checkpoint fields
        'mapping': {
            'world_size': mapping.tp_size * mapping.pp_size,
            'tp_size': mapping.tp_size,
            'pp_size': mapping.pp_size,
            'moe_tp_size': mapping.moe_tp_size,
            'moe_ep_size': mapping.moe_ep_size,
        },
        'attn_bias':
        attn_bias,
        "attn_output_multiplier":
        attn_output_multiplier,
        "embedding_multiplier_scale":
        embedding_multiplier_scale,
        "output_multiplier_scale":
        output_multiplier_scale,
        "max_attn_value":
        max_attn_value,
        "tie_word_embeddings":
        True,
    })

    config['quantization'] = quantization.to_dict()
    config.update(override_fields)

    return config


def from_hugging_face(cls,
                      model_dir,
                      dtype,
                      *,
                      mapping,
                      quantization: QuantConfig = None,
                      override_fields={},
                      skip_loading_weights=False,
                      preloaded_model=None):
    ''' Create a LLaMAForCausalLM object from give parameters
    '''
    assert model_dir is not None
    if isinstance(model_dir, Path):  # some code relies on this as string
        model_dir = str(model_dir)

    config = create_config_from_xai(dtype,
                                    mapping,
                                    quantization,
                                    override_fields=override_fields)

    pretrained_config = PretrainedConfig.from_dict(config)
    pretrained_config.set_rank(mapping.rank)  # TODO:remove this hack

    grok = cls.from_config(pretrained_config)
    grok = optimize_model(
        grok,
        use_parallel_embedding=pretrained_config.use_parallel_embedding,
    )

    if skip_loading_weights:
        return grok

    weights = load_weights_from_xai(config=config,
                                    mapping=mapping,
                                    model=preloaded_model)

    grok.load(weights)
    return grok


def quantize(dtype,
             model_dir,
             output_dir,
             mapping,
             quantization: QuantConfig,
             *,
             override_fields,
             dataset_cache_dir: Optional[str] = None):
    '''
        Quantize the save the model as TRT-LLM checkpoint to output_dir
    '''
    pass  #The official grok-1 model is published under int8 wo format, we don't need to quantize again.


def load_weights_from_xai(*, config, mapping, model):
    assert model is not None
    plugin_weight_only_quant_type = None  # the value does not matter when use_weight_only is False
    quant_algo = config['quantization']['quant_algo']
    assert quant_algo == QuantAlgo.W8A16
    plugin_weight_only_quant_type = torch.int8

    moe_config = MoeConfig(
        num_experts=config['moe_num_experts'],
        top_k=config['moe_top_k'],
        normalization_mode=config['moe_normalization_mode']).validate()

    use_weight_only = quant_algo in [QuantAlgo.W8A16]

    weights = convert_grok(
        model,
        config,
        mapping,
        vocab_size=config['vocab_size'],
        dtype=config['dtype'],
        use_weight_only=use_weight_only,
        use_gemm_woq_plugin=not config.get('disable_weight_only_quant_plugin',
                                           False),
        plugin_weight_only_quant_type=plugin_weight_only_quant_type,
        use_parallel_embedding=config.get('use_parallel_embedding', False),
        sharding_dim=config.get('embedding_sharding_dim', 0),
        moe_config=moe_config)
    return weights
