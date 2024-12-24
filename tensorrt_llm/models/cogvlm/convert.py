import time

import numpy as np
import torch

from tensorrt_llm.logger import logger

from ..._utils import pad_vocab_size
from ..llama.convert import (get_tllm_linear_weight, get_weight, split,
                             split_matrix_tp, split_qkv_tp)


def convert_hf_cogvlm(hf_model,
                      mapping,
                      vocab_size=32000,
                      dtype='float32',
                      use_parallel_embedding=False,
                      sharding_dim=0,
                      use_weight_only=False,
                      use_gemm_woq_plugin=False,
                      plugin_weight_only_quant_type=torch.int8,
                      use_smooth_quant=False,
                      per_channel=False,
                      per_token=False,
                      int8_kv_cache=False,
                      act_range=[],
                      qkv_para=[],
                      smoother=[]):

    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    if hasattr(hf_model.config, "num_key_value_heads"):
        num_key_value_heads = hf_model.config.num_key_value_heads
    else:
        num_key_value_heads = num_attention_heads
    mha_mode = (num_key_value_heads == num_attention_heads)
    layers_range = mapping.pp_layers(hf_model.config.num_hidden_layers)
    assert mha_mode, "CogVLM only supports mha mode"
    assert not use_smooth_quant, "CogVLM currently doesn't support smooth quant"
    assert not int8_kv_cache, "CogVLM currently doesn't support int8 kv cache"

    for l in layers_range:
        prefix = f'model.layers.{l}.'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'

        qkv_weight = get_weight(
            model_params, prefix + 'self_attn.language_expert_query_key_value',
            dtype)
        split_v = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                               tensor_parallel, mapping.tp_rank)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.qkv.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        vis_qkv_weight = get_weight(
            model_params, prefix + 'self_attn.vision_expert_query_key_value',
            dtype)
        split_v = split_qkv_tp(vis_qkv_weight, num_attention_heads, hidden_size,
                               tensor_parallel, mapping.tp_rank)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.vis_qkv.',
                                   None, use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        attn_dense_weight = get_weight(
            model_params, prefix + 'self_attn.language_expert_dense', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.dense.',
                                   None, use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        attn_vision_dense_weight = get_weight(
            model_params, prefix + 'self_attn.vision_expert_dense', dtype)
        split_v = split_matrix_tp(attn_vision_dense_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.vis_dense.',
                                   None, use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        mlp_gate_weight = get_weight(model_params,
                                     prefix + 'mlp.language_mlp.up_proj', dtype)
        split_v = split_matrix_tp(mlp_gate_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=0)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.gate.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        vision_mlp_gate_weight = get_weight(model_params,
                                            prefix + 'mlp.vision_mlp.up_proj',
                                            dtype)
        split_v = split_matrix_tp(vision_mlp_gate_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=0)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'vis_mlp.gate.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        mlp_fc_weight = get_weight(model_params,
                                   prefix + 'mlp.language_mlp.gate_proj', dtype)
        split_v = split_matrix_tp(mlp_fc_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=0)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.fc.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        vision_mlp_fc_weight = get_weight(model_params,
                                          prefix + 'mlp.vision_mlp.gate_proj',
                                          dtype)
        split_v = split_matrix_tp(vision_mlp_fc_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=0)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'vis_mlp.fc.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        mlp_proj_weight = get_weight(model_params,
                                     prefix + 'mlp.language_mlp.down_proj',
                                     dtype)
        split_v = split_matrix_tp(mlp_proj_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        vision_mlp_proj_weight = get_weight(model_params,
                                            prefix + 'mlp.vision_mlp.down_proj',
                                            dtype)
        split_v = split_matrix_tp(vision_mlp_proj_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'vis_mlp.proj.', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type, dtype,
                                   use_gemm_woq_plugin))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + 'input_layernorm',
                                     dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_ln_weight = get_weight(model_params,
                                    prefix + 'post_attention_layernorm', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight
        cur_block_weights = [
            weight_name for weight_name in model_params
            if weight_name.find(prefix) != -1
        ]
        for weight_name in cur_block_weights:
            model_params[weight_name] = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    v = get_weight(model_params, 'model.embed_tokens', dtype)
    if hf_model.config.tie_word_embeddings:
        # lm_head.weight has the same weights as embedding
        if mapping.is_last_pp_rank():
            if vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size

                v = torch.from_numpy(
                    np.pad(v.detach().cpu().numpy(), ((0, pad_width), (0, 0)),
                           'constant',
                           constant_values=0))
            weights['lm_head.weight'] = split(v, mapping.tp_size,
                                              mapping.tp_rank)

    if use_parallel_embedding:
        v = split_matrix_tp(v,
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=sharding_dim)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v

    lm_head_weights = get_weight(model_params, 'lm_head', dtype)

    if mapping.is_last_pp_rank():
        if vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size

            lm_head_weights = torch.from_numpy(
                np.pad(lm_head_weights.detach().cpu().numpy(),
                       ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0))
        weights['lm_head.weight'] = split_matrix_tp(lm_head_weights,
                                                    tensor_parallel,
                                                    mapping.tp_rank,
                                                    dim=0)
        ln_f_w = get_weight(model_params, 'model.norm', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights
