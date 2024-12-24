import time
from typing import Dict, Optional

import torch

from ...quantization import QuantAlgo
from ..convert_utils import (get_weight, get_weight_and_bias,
                             iterate_shard_files, load_state_dict,
                             retrieved_layer_index_from_name)
from .config import FalconConfig


def split(weight: torch.Tensor,
          tp_size: int,
          rank: int = 0,
          dim: int = 0) -> torch.Tensor:
    if tp_size == 1:
        return weight
    elif weight.ndim == 1:
        return torch.chunk(weight, tp_size)[rank].clone()
    else:
        return torch.chunk(weight, tp_size, dim=dim)[rank].clone()


def reorder_qkv_weight_or_bias(weight: torch.Tensor,
                               head_dim: int,
                               num_heads: int,
                               num_kv_heads: Optional[int] = None,
                               tp_size: int = 1,
                               is_bias: bool = False) -> torch.Tensor:
    """ Reorder the qkv weight for TRT-LLM use.

    The shape of the fused QKV weights in HF is different from the shape that
    TRT-LLM requires. In particular, the weight of HF consists of interleaved
    q, k, v head weights, while that of TRT-LLM is contiguous.
        HF     : [q1, k1, v1, ..., qh, kh, vh]
        TRT-LLM: [q1, ..., qh, k1, ..., kh, v1, vh]
    where qi, vi, ki are weight vectors corresponding to attention head i.
    It's similar to multi/grouped query attention cases.

    We reorder and split the weight of an attention layer to fit into TRT-LLM.
    The reordered weight and bias will be
        weight: (T, Qh * D + 2 * KVh * D, H)
        bias  : (T, Qh * D + 2 * KVh * D)
    where T=tp_size, Qh=local_num_q_heads, KVh=local_num_kv_heads, D=head_dim,
    H=hidden_dim. In the multi/grouped query attention, the number of K/V
    attention heads are less than that of Q attention, so that K/V attention
    heads may be shared across different ranks if necessary.

    For tensor parallelism, we use the first dimension to select the
    corresponding weights.
    """

    # Query types and expected kv heads.
    #  - Conventional MHA: num_heads = num_kv_heads
    #  - Multi-Query Attention: num_kv_heads = 1
    #  - Grouped-Query Attention: num_heads % num_kv_heads = 0
    num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    assert num_heads % num_kv_heads == 0, \
        f'num_heads({num_heads}) must be divisible by ' \
        f'num_kv_heads({num_kv_heads})).'

    # The number of attention heads per group: N q head + 1 k head + 1 v head.
    num_group_heads = num_heads // num_kv_heads + 2
    assert weight.shape[0] == num_kv_heads * num_group_heads * head_dim, \
        f'{weight.shape[0]} != {num_kv_heads} * {num_group_heads} * {head_dim}'

    qkv_in = num_heads * head_dim if not is_bias else 1

    # Split Q/K/V weights
    weight = weight.reshape(num_kv_heads, num_heads // num_kv_heads + 2,
                            head_dim, qkv_in)
    q_w = weight[:, :-2, ...]  # (nKV, num_heads // nKV, head_dim, qkv_in)
    k_w = weight[:, -2:-1, ...]  # (nKV, 1, head_dim, qkv_in)
    v_w = weight[:, -1:, ...]  # (nKV, 1, head_dim, qkv_in)

    if num_kv_heads < num_heads and num_kv_heads < tp_size:
        # Duplicate K/V heads to make sure that each rank has at least one
        # K/V heads. For instance, num_heads=8, num_kv_heads=2, tp_size=4,
        # we will make the qkv weight as below.
        #   Orig: [q0 q1 q2 q3 k0 v0 q4 q5 q6 q7 k1 v0 v1]
        #   >>>>  [[q0 q1 k0 v0], [q2 q3 k0 v0], [q4 q5 k1 v1], [q6 q7 k1 v1]]
        assert tp_size % num_kv_heads == 0
        num_dups = tp_size // num_kv_heads

        # k_w and v_w have the same shape.
        new_shape = (num_kv_heads, num_dups) + k_w.shape[2:]
        k_w = torch.broadcast_to(k_w, size=new_shape)
        v_w = torch.broadcast_to(v_w, size=new_shape)

        # Update the number of kv heads.
        num_kv_heads = tp_size

    reordered = torch.concat(
        [
            q_w.reshape(tp_size, num_heads // tp_size, head_dim, qkv_in),
            k_w.reshape(tp_size, num_kv_heads // tp_size, head_dim, qkv_in),
            v_w.reshape(tp_size, num_kv_heads // tp_size, head_dim, qkv_in),
        ],
        dim=1,
    )

    qkv_out = (num_heads + 2 * num_kv_heads) // tp_size * head_dim
    return reordered.reshape((tp_size, qkv_out, -1))


def split_qkv_weight(weight: torch.Tensor,
                     hidden_size: int,
                     num_heads: int,
                     tp_size: int,
                     rank: int,
                     is_bias: bool,
                     num_kv_heads: Optional[int] = None) -> torch.Tensor:
    """ Splits the QKV matrix according to tensor parallelism """
    head_dim = hidden_size // num_heads
    weight = reorder_qkv_weight_or_bias(weight,
                                        head_dim=head_dim,
                                        num_heads=num_heads,
                                        num_kv_heads=num_kv_heads,
                                        tp_size=tp_size,
                                        is_bias=is_bias)

    # Copy a sliced tensor to prevent memory leak. A sliced tensor shares the
    # memory buffer of the original tensor. So, returning without copying makes
    # the buffer of a loaded "qkv" be referenced, resulting GC can't release
    # those weights until the whole process ends.
    if not is_bias:
        return weight[rank, ...].clone()
    else:
        return weight[rank, ...].ravel().clone()


def split_matrix(weight: torch.Tensor, tp_size: int, rank: int,
                 dim: int) -> torch.Tensor:
    return split(weight, tp_size, rank, dim=dim)


def get_tllm_linear_weight(
    weight: torch.Tensor,
    prefix: str,
    bias: Optional[torch.Tensor] = None,
    use_weight_only: bool = False,
    plugin_weight_only_quant_type: torch.dtype = torch.int8
) -> Dict[str, torch.Tensor]:
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[f'{prefix}.weight'] = processed_torch_weights
        results[f'{prefix}.per_channel_scale'] = torch_weight_scales
    else:
        results[f'{prefix}.weight'] = weight

    if bias is not None:
        results[f'{prefix}.bias'] = bias

    return results


def get_tllm_param(
    param: torch.Tensor,
    name: str,
    use_weight_only: bool = False,
    plugin_weight_only_quant_type: torch.dtype = torch.int8
) -> Dict[str, torch.Tensor]:
    results = {}
    if name.endswith('.weight') and use_weight_only:
        v = param.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[name] = processed_torch_weights
        results[name.replace('weight',
                             'per_channel_scale')] = torch_weight_scales
    else:
        results[name] = param

    return results


def load_weights_from_hf_model(model, config: FalconConfig):
    weights = {}
    tik = time.time()

    model_params = dict(model.named_parameters())
    dtype = getattr(torch, config.dtype)
    mapping = config.mapping
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_kv_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    num_hidden_layers = config.num_hidden_layers
    parallel_attention = config.parallel_attention
    new_decoder_architecture = config.new_decoder_architecture
    use_parallel_embedding = config.use_parallel_embedding
    sharding_dim = config.embedding_sharding_dim
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'transformer.h.{l}'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}'
        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, f'{prefix}.self_attention.query_key_value', dtype)
        qkv_w = split_qkv_weight(qkv_weight,
                                 hidden_size,
                                 num_attention_heads,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 is_bias=False,
                                 num_kv_heads=num_kv_heads)
        if qkv_bias is None:
            qkv_b = None
        else:
            qkv_b = split_qkv_weight(qkv_bias,
                                     hidden_size,
                                     num_attention_heads,
                                     mapping.tp_size,
                                     mapping.tp_rank,
                                     is_bias=True,
                                     num_kv_heads=num_kv_heads)
        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv', qkv_b,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, f'{prefix}.self_attention.dense', dtype)
        attn_dense_w = split_matrix(attn_dense_weight,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=1)
        weights.update(
            get_tllm_linear_weight(attn_dense_w, f'{tllm_prex}.attention.dense',
                                   attn_dense_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_h_to_4h', dtype)
        mlp_fc_w = split_matrix(mlp_fc_weight,
                                mapping.tp_size,
                                mapping.tp_rank,
                                dim=0)
        if mlp_fc_bias is None:
            mlp_fc_b = None
        else:
            mlp_fc_b = split_matrix(mlp_fc_bias,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=0)
        weights.update(
            get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc', mlp_fc_b,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_4h_to_h', dtype)
        mlp_proj_w = split_matrix(mlp_proj_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                   mlp_proj_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        if new_decoder_architecture:
            input_ln_weight, input_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.ln_attn', dtype)
            if input_ln_weight is None:
                input_ln_weight, input_ln_bias = get_weight_and_bias(
                    model_params, f'{prefix}.input_layernorm', dtype)
            weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight
            if input_ln_bias is not None:
                weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_bias

            mlp_ln_weight, mlp_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.ln_mlp', dtype)
            if mlp_ln_weight is not None:
                weights[f'{tllm_prex}.mlp_layernorm.weight'] = mlp_ln_weight
            if mlp_ln_bias is not None:
                weights[f'{tllm_prex}.mlp_layernorm.bias'] = mlp_ln_bias
        else:
            input_ln_weight, input_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.input_layernorm', dtype)
            weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight
            if input_ln_bias is not None:
                weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_bias

            if not parallel_attention:
                post_ln_weight, post_ln_bias = get_weight_and_bias(
                    model_params, f'{prefix}.post_attention_layernorm', dtype)
                if post_ln_weight is not None:
                    weights[
                        f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight
                if post_ln_bias is not None:
                    weights[f'{tllm_prex}.post_layernorm.bias'] = post_ln_bias

    embed_w = get_weight(model_params, 'transformer.word_embeddings', dtype)
    if mapping.is_first_pp_rank():
        if not use_parallel_embedding:
            weights['transformer.vocab_embedding.weight'] = embed_w
        else:
            if sharding_dim == 0:
                assert vocab_size % mapping.tp_size == 0
            else:
                assert hidden_size % mapping.tp_size == 0
            weights['transformer.vocab_embedding.weight'] = split_matrix(
                embed_w, mapping.tp_size, mapping.tp_rank, sharding_dim)

    if mapping.is_last_pp_rank():
        lm_head = get_weight(model_params, 'lm_head', dtype)
        if lm_head is None:
            # No lm_head in the checkpoint, cloning word_embedding.
            lm_head = embed_w.clone()
        weights['lm_head.weight'] = split_matrix(lm_head,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 dim=0)
        ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'transformer.ln_f',
                                             dtype)
        weights['transformer.ln_f.weight'] = ln_f_w
        if ln_f_b is not None:
            weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def load_weights_from_hf_by_shard(model_dir: str, config: FalconConfig):
    weights = {}
    tik = time.time()

    dtype = getattr(torch, config.dtype)
    mapping = config.mapping
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_kv_heads = getattr(config, 'num_key_value_heads', num_attention_heads)
    num_hidden_layers = config.num_hidden_layers
    use_weight_only = config.quantization.quant_algo in [
        QuantAlgo.W8A16, QuantAlgo.W4A16
    ]
    if config.quantization.quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif config.quantization.quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    layers_range = mapping.pp_layers(num_hidden_layers)
    for model_file in iterate_shard_files(model_dir, mapping.tp_rank):
        state_dict = load_state_dict(model_file, dtype)
        for name, param in state_dict.items():
            l = retrieved_layer_index_from_name(name)
            if l is not None:
                if l not in layers_range:
                    continue
                prefix = f'transformer.layers.{l-layers_range[0]}'

                if 'self_attention.query_key_value' in name:
                    if name.endswith('weight'):
                        qkv_w = split_qkv_weight(param,
                                                 hidden_size,
                                                 num_attention_heads,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 is_bias=False,
                                                 num_kv_heads=num_kv_heads)
                        weights.update(
                            get_tllm_param(qkv_w,
                                           f'{prefix}.attention.qkv.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        qkv_b = split_qkv_weight(param,
                                                 hidden_size,
                                                 num_attention_heads,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 is_bias=True,
                                                 num_kv_heads=num_kv_heads)
                        weights.update(
                            get_tllm_param(qkv_b,
                                           f'{prefix}.attention.qkv.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'self_attention.dense' in name:
                    if name.endswith('weight'):
                        attn_dense_w = split_matrix(param,
                                                    mapping.tp_size,
                                                    mapping.tp_rank,
                                                    dim=1)
                        weights.update(
                            get_tllm_param(attn_dense_w,
                                           f'{prefix}.attention.dense.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        weights.update(
                            get_tllm_param(param,
                                           f'{prefix}.attention.dense.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'mlp.dense_h_to_4h' in name:
                    if name.endswith('weight'):
                        mlp_fc_w = split_matrix(param,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=0)
                        weights.update(
                            get_tllm_param(mlp_fc_w, f'{prefix}.mlp.fc.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        mlp_fc_b = split_matrix(param,
                                                mapping.tp_size,
                                                mapping.tp_rank,
                                                dim=0)
                        weights.update(
                            get_tllm_param(mlp_fc_b, f'{prefix}.mlp.fc.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'mlp.dense_4h_to_h' in name:
                    if name.endswith('weight'):
                        mlp_proj_w = split_matrix(param,
                                                  mapping.tp_size,
                                                  mapping.tp_rank,
                                                  dim=1)
                        weights.update(
                            get_tllm_param(mlp_proj_w,
                                           f'{prefix}.mlp.proj.weight',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))
                    else:
                        weights.update(
                            get_tllm_param(param, f'{prefix}.mlp.proj.bias',
                                           use_weight_only,
                                           plugin_weight_only_quant_type))

                elif 'ln_attn' in name or 'input_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.input_layernorm.weight'] = param
                    else:
                        weights[f'{prefix}.input_layernorm.bias'] = param
                elif 'ln_mlp' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.mlp_layernorm.weight'] = param
                    else:
                        weights[f'{prefix}.mlp_layernorm.bias'] = param
                elif 'post_attention_layernorm' in name:
                    if name.endswith('weight'):
                        weights[f'{prefix}.post_layernorm.weight'] = param
                    else:
                        weights[f'{prefix}.post_layernorm.bias'] = param
            elif 'word_embeddings' in name:
                if mapping.is_first_pp_rank():
                    if not config.use_parallel_embedding:
                        weights['transformer.vocab_embedding.weight'] = param
                    else:
                        if config.embedding_sharding_dim == 0:
                            assert vocab_size % mapping.tp_size == 0
                        else:
                            assert hidden_size % mapping.tp_size == 0
                        weights[
                            'transformer.vocab_embedding.weight'] = split_matrix(
                                param, mapping.tp_size, mapping.tp_rank,
                                config.embedding_sharding_dim)
                if mapping.is_last_pp_rank():
                    weights['lm_head.weight'] = split_matrix(param,
                                                             mapping.tp_size,
                                                             mapping.tp_rank,
                                                             dim=0)
            elif 'ln_f' in name:
                if mapping.is_last_pp_rank():
                    if name.endswith('weight'):
                        weights['transformer.ln_f.weight'] = param
                    else:
                        weights['transformer.ln_f.bias'] = param
        del state_dict

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights
