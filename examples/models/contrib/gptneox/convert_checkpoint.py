import argparse
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import safetensors
import safetensors.torch
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import (get_weight, get_weight_and_bias,
                                               split_matrix_tp)
from tensorrt_llm.quantization import QuantAlgo


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
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4', 'int4_gptq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument('--quant_ckpt_path',
                        type=str,
                        default=None,
                        help='Path of a quantized model checkpoint')
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
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


# TODO: Seems all convert checkpoints may use following utility functions.
#       Maybe in one common version.
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


def get_gptq_gptneox_group_size(quant_ckpt_path, hf_config):
    gptq_model = safe_open(quant_ckpt_path, framework="pt", device=0)
    gptq_prefix = "gpt_neox."
    split_sym = "."

    def load(key, no_prefix=0):
        if no_prefix:
            return gptq_model.get_tensor(key).cpu()
        else:
            return gptq_model.get_tensor(gptq_prefix + key).cpu()

    hidden_size = hf_config.hidden_size
    prefix = "layers" + split_sym + "0" + split_sym

    scales_fp16 = load(prefix + 'attention.query_key_value.scales')
    return hidden_size // scales_fp16.shape[0]


def load_from_gptq_gptneox(quant_ckpt_path,
                           hf_config=None,
                           use_parallel_embedding=False,
                           sharding_dim=0,
                           mapping=Mapping(),
                           dtype='float16'):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise GPTQ LLaMA safetensors...')
    weights = {}
    tik = time.time()

    gptq_model = safe_open(quant_ckpt_path, framework="pt", device=0)
    gptq_prefix = "gpt_neox."
    gptq_suffix_list = [".qweight", ".qzeros", ".scales"]
    split_sym = "."

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    def load(key, no_prefix=0):
        if no_prefix:
            return gptq_model.get_tensor(key).cpu()
        else:
            return gptq_model.get_tensor(gptq_prefix + key).cpu()

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank].contiguous()

    def unpack_int32_into_int8(w_packed):
        # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8,
                                 device=w_packed.device)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def process_and_assign_weight(v: List[torch.Tensor],
                                  tllm_prex: str,
                                  tp_dim: int = -1):
        if tp_dim == -1:
            qweight_int32, qzeros_int32, scales_fp16 = [
                item.cpu() for item in v
            ]
        else:
            qweight_int32, qzeros_int32, scales_fp16 = [
                torch_split(item, tp_dim).cpu() for item in v
            ]

        USE_UINT4_INPUT = 1  # Set to true if checkpoint store UINT4 weights
        USE_GPTQ_FOR_LLAMA = 1  # GPTQ-for-LLaMA added 1 to zeros

        qweight_unpacked_int8 = unpack_int32_into_int8(
            qweight_int32.T).T.contiguous() - 8
        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2,
                                           torch.float16).view(torch.float16)
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)
        if not USE_UINT4_INPUT:
            # Correcting UINT4 values back to INT4 order
            mask_negative = qzeros_unpacked_int32[qzeros_unpacked_int32 < 0]
            mask_positive = qzeros_unpacked_int32[qzeros_unpacked_int32 >= 0]
            qzeros_unpacked_int32 = qzeros_unpacked_int32 + 16 * mask_negative - 16 * mask_positive
        zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8 * USE_UINT4_INPUT -
                               USE_GPTQ_FOR_LLAMA) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        results = {
            f'{tllm_prex}.weight': qweight_interleaved,
            f'{tllm_prex}.weights_scaling_factor': scales_fp16,
            f'{tllm_prex}.zero': zeros_x_scales_fp16,
        }
        return results

    def preprocess_groupwise_weight_params(qweight_unpacked_int8, scales_fp16,
                                           qzeros_unpacked_int8):
        UINT4_TO_INT4_FLAG = 1
        GPTQ_FLAG = 1

        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2,
                                           torch.float16).view(torch.float16)

        # zeros = zeros * scales
        zeros_x_scales_fp16 = (-qzeros_unpacked_int8 + 8 * UINT4_TO_INT4_FLAG -
                               GPTQ_FLAG) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        # return processed interleaved weight, original scales and zeros * scales
        return qweight_interleaved.contiguous(), scales_fp16.contiguous(
        ), zeros_x_scales_fp16.contiguous()

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load('embed_in.weight')
    if mapping.is_first_pp_rank():
        if not use_parallel_embedding:
            weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)
        else:
            assert hf_config.vocab_size % mapping.tp_size == 0
            weights['transformer.vocab_embedding.weight'] = torch_split(
                v, sharding_dim).to(torch_dtype)
    # 2. lm_head
    if mapping.is_last_pp_rank():
        v = load('embed_out.weight', no_prefix=1)
        weights['lm_head.weight'] = torch_split(v, 0).to(torch_dtype)

    # 3. ln_f
    v = load('final_layer_norm.weight')
    b = load('final_layer_norm.bias')
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)
        weights['transformer.ln_f.bias'] = b.to(torch_dtype)
    # 4. Weights inside each layer
    num_hidden_layers = hf_config.num_hidden_layers
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        layer_idx = l - layers_range[0]
        prefix = "layers" + split_sym + str(l) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        # layer = tensorrt_llm_llama.layers[layer_idx]
        tllm_prex = f'transformer.layers.{l - layers_range[0]}'
        # 4.1 attention.qkv
        num_heads = hf_config.num_attention_heads
        hidden_size = hf_config.hidden_size
        head_size = hidden_size // num_heads

        qweight_int32 = load(prefix + 'attention.query_key_value.qweight')
        scales_fp16 = load(prefix + 'attention.query_key_value.scales')
        qzeros_int32 = load(prefix + 'attention.query_key_value.qzeros')
        biases_fp16 = load(prefix + 'attention.query_key_value.bias')
        GROUP_SIZE = hidden_size // scales_fp16.shape[0]

        # [hidden_size // 8, hidden_size * 3] -> [hidden_size * 3, hidden_size]
        qweight_unpacked_int8 = unpack_int32_into_int8(
            qweight_int32.T).contiguous() - 8
        # [hidden_size // GROUP_SIZE, hidden_size * 3 // 8] ->
        # [hidden_size // GROUP_SIZE, hidden_size * 3]
        qzeros_unpacked_int8 = unpack_int32_into_int8(qzeros_int32)

        # qkv_weights [num_heads x (q|k|v), hidden_size] ->
        # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
        new_qkv_weight_shape = torch.Size(
            [num_heads, 3, head_size * qweight_unpacked_int8.size()[-1]])
        # [hidden_size * 3, hidden_size]
        qweight_unpacked_int8 = qweight_unpacked_int8.view(
            new_qkv_weight_shape).permute(1, 0, 2).reshape(
                [hidden_size * 3, hidden_size]).contiguous()

        new_qkv_scale_shape = torch.Size(
            [num_heads, 3, head_size * (hidden_size // GROUP_SIZE)])
        # [hidden_size * 3, hidden_size // GROUP_SIZE]
        scales_fp16 = scales_fp16.T.contiguous().view(
            new_qkv_scale_shape).permute(1, 0, 2).reshape(
                [hidden_size * 3, hidden_size // GROUP_SIZE]).contiguous()

        new_qkv_zero_shape = torch.Size(
            [num_heads, 3, head_size * (hidden_size // GROUP_SIZE)])
        # [hidden_size * 3, hidden_size // GROUP_SIZE]
        qzeros_unpacked_int8 = qzeros_unpacked_int8.T.contiguous().view(
            new_qkv_zero_shape).permute(1, 0, 2).reshape(
                [hidden_size * 3, hidden_size // GROUP_SIZE]).contiguous()

        new_qkv_bias_shape = torch.Size([num_heads, 3, head_size])
        biases_fp16 = biases_fp16.view(new_qkv_bias_shape).permute(
            1, 0, 2).reshape([hidden_size * 3])

        tp_size = mapping.tp_size

        if tp_size > 1:
            qweight_unpacked_int8 = qweight_unpacked_int8.reshape(
                [3, hidden_size, hidden_size])
            qweight_unpacked_int8 = torch_split(qweight_unpacked_int8, dim=1)
            qweight_unpacked_int8 = qweight_unpacked_int8.reshape(
                [3 * hidden_size // tp_size, hidden_size])

            scales_fp16 = scales_fp16.reshape(
                [3, hidden_size, hidden_size // GROUP_SIZE])
            scales_fp16 = torch_split(scales_fp16, dim=1)
            scales_fp16 = scales_fp16.reshape(
                [3 * hidden_size // tp_size, hidden_size // GROUP_SIZE])

            qzeros_unpacked_int8 = qzeros_unpacked_int8.reshape(
                [3, hidden_size, hidden_size // GROUP_SIZE])
            qzeros_unpacked_int8 = torch_split(qzeros_unpacked_int8, dim=1)
            qzeros_unpacked_int8 = qzeros_unpacked_int8.reshape(
                [3 * hidden_size // tp_size, hidden_size // GROUP_SIZE])

            biases_fp16 = biases_fp16.reshape([3, hidden_size])
            biases_fp16 = torch_split(biases_fp16, dim=1)
            biases_fp16 = biases_fp16.reshape([3 * hidden_size // tp_size])

        qweight_fp32, scales_fp16, zeros_fp16 = preprocess_groupwise_weight_params(
            qweight_unpacked_int8.T.contiguous(), scales_fp16.T.contiguous(),
            qzeros_unpacked_int8.T.contiguous())
        weights.update({
            f'{tllm_prex}.attention.qkv.weight': qweight_fp32,
            f'{tllm_prex}.attention.qkv.weights_scaling_factor': scales_fp16,
            f'{tllm_prex}.attention.qkv.zero': zeros_fp16,
            f'{tllm_prex}.attention.qkv.bias': biases_fp16,
        })

        # 4.2 attention.dense
        v = [load(prefix + 'attention.dense' + suf) for suf in gptq_suffix_list]
        # pre scaling down for duplicated bias add between different tp ranks
        b = load(prefix + 'attention.dense.bias') / mapping.tp_size

        weights.update(
            process_and_assign_weight(v,
                                      f'{tllm_prex}.attention.dense',
                                      tp_dim=0))
        weights.update({f'{tllm_prex}.attention.dense.bias': b.to(torch_dtype)})
        # 4.3 mlp.fc
        v = [
            load(prefix + 'mlp.dense_h_to_4h' + suf) for suf in gptq_suffix_list
        ]
        b = load(prefix + 'mlp.dense_h_to_4h.bias')
        weights.update(
            process_and_assign_weight(v, f'{tllm_prex}.mlp.fc', tp_dim=1))
        weights.update(
            {f'{tllm_prex}.mlp.fc.bias': torch_split(b, dim=0).to(torch_dtype)})
        # 4.4 mlp.proj
        v = [
            load(prefix + 'mlp.dense_4h_to_h' + suf) for suf in gptq_suffix_list
        ]
        # pre scaling down for duplicated bias add between different tp ranks
        b = load(prefix + 'mlp.dense_4h_to_h.bias') / mapping.tp_size

        weights.update(
            process_and_assign_weight(v, f'{tllm_prex}.mlp.proj', tp_dim=0))
        weights.update({f'{tllm_prex}.mlp.proj.bias': b.to(torch_dtype)})
        # 4.5 input_layernorm
        v = load(prefix + 'input_layernorm.weight')
        b = load(prefix + 'input_layernorm.bias')
        weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)
        weights[f'{tllm_prex}.input_layernorm.bias'] = b.to(torch_dtype)

        # 4.6 post_layernorm
        v = load(prefix + 'post_attention_layernorm.weight')
        b = load(prefix + 'post_attention_layernorm.bias')
        weights[f'{tllm_prex}.post_attention_layernorm.weight'] = v.to(
            torch_dtype)
        weights[f'{tllm_prex}.post_attention_layernorm.bias'] = b.to(
            torch_dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')

    return weights


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
        return weight[rank, ...].clone().contiguous()
    else:
        return weight[rank, ...].ravel().clone().contiguous()


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8):
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[prefix + 'weight'] = processed_torch_weights
        results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + 'weight'] = weight.contiguous()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def convert_hf_gptneox(hf_model,
                       mapping: Mapping,
                       dtype='float32',
                       use_parallel_embedding=False,
                       sharding_dim=0,
                       use_weight_only=False,
                       plugin_weight_only_quant_type=torch.int8):
    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    tensor_parallel = mapping.tp_size
    rank = mapping.rank

    for l in range(hf_model.config.num_hidden_layers):
        prefix = f'gpt_neox.layers.{l}.'
        tllm_prex = f'transformer.layers.{l}.'

        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, prefix + 'attention.query_key_value', dtype)
        qkv_w = split_qkv_weight(qkv_weight,
                                 hidden_size,
                                 num_attention_heads,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 is_bias=False,
                                 num_kv_heads=num_attention_heads)
        if qkv_bias is None:
            qkv_b = None
        else:
            qkv_b = split_qkv_weight(qkv_bias,
                                     hidden_size,
                                     num_attention_heads,
                                     mapping.tp_size,
                                     mapping.tp_rank,
                                     is_bias=True,
                                     num_kv_heads=num_attention_heads)
        weights.update(
            get_tllm_linear_weight(qkv_w, tllm_prex + 'attention.qkv.', qkv_b,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, prefix + 'attention.dense', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'attention.dense.',
                                   attn_dense_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, prefix + 'mlp.dense_h_to_4h', dtype)
        split_v = split_matrix_tp(mlp_fc_weight, tensor_parallel, rank, dim=0)
        bias = split_matrix_tp(mlp_fc_bias, tensor_parallel, rank, dim=0)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.fc.', bias,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))

        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, prefix + 'mlp.dense_4h_to_h', dtype)
        split_v = split_matrix_tp(mlp_proj_weight, tensor_parallel, rank, dim=1)
        weights.update(
            get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.',
                                   mlp_proj_bias, use_weight_only,
                                   plugin_weight_only_quant_type))

        # Layer norms do not use tensor parallelism
        input_ln_weight, input_ln_bias = get_weight_and_bias(
            model_params, prefix + 'input_layernorm', dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight
        weights[tllm_prex + 'input_layernorm.bias'] = input_ln_bias

        post_ln_weight, post_ln_bias = get_weight_and_bias(
            model_params, prefix + 'post_attention_layernorm', dtype)
        weights[tllm_prex + 'post_attention_layernorm.weight'] = post_ln_weight
        weights[tllm_prex + 'post_attention_layernorm.bias'] = post_ln_bias

    embed_w = get_weight(model_params, 'gpt_neox.embed_in', dtype)
    lm_head_w = get_weight(model_params, 'embed_out', dtype)

    weights['lm_head.weight'] = split_matrix_tp(lm_head_w,
                                                tensor_parallel,
                                                rank,
                                                dim=0)

    if not use_parallel_embedding:
        weights['transformer.vocab_embedding.weight'] = embed_w
    else:
        assert hf_model.config.vocab_size % tensor_parallel == 0
        weights['transformer.vocab_embedding.weight'] = split_matrix_tp(
            embed_w, tensor_parallel, rank, dim=sharding_dim)

    ln_f_w, ln_f_b = get_weight_and_bias(model_params,
                                         'gpt_neox.final_layer_norm', dtype)
    weights['transformer.ln_f.weight'] = ln_f_w
    weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size
    assert args.pp_size == 1, "Pipeline parallelism is not supported."

    tensorrt_llm.logger.set_level('info')
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2
        quant_algo = QuantAlgo.W4A16
    elif args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        quant_algo = QuantAlgo.W4A16_GPTQ

    hf_config = AutoConfig.from_pretrained(args.model_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                    dtype="auto")

    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'hidden_size': hf_config.hidden_size,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': hf_config.max_position_embeddings,
        'rotary_emb_base': hf_config.rotary_emb_base,
        'rotary_pct': hf_config.rotary_pct,
        'hidden_act': hf_config.hidden_act,
        'quantization': {
            'quant_algo': quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
    }
    if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        config['quantization'].update({
            'has_zero_point':
            True,
            'group_size':
            get_gptq_gptneox_group_size(args.quant_ckpt_path, hf_config)
        })

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            weights = load_from_gptq_gptneox(
                args.quant_ckpt_path,
                hf_config,
                use_parallel_embedding=args.use_parallel_embedding,
                sharding_dim=args.embedding_sharding_dim,
                mapping=mapping,
                dtype=args.dtype)
        else:
            weights = convert_hf_gptneox(
                hf_model,
                mapping,
                dtype=args.dtype,
                use_weight_only=args.use_weight_only,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                use_parallel_embedding=args.use_parallel_embedding,
                sharding_dim=args.embedding_sharding_dim)
        safe_save_path = os.path.join(args.output_dir,
                                      f'rank{rank}.safetensors')
        tensorrt_llm.logger.info(f'Saving safetensors to: {safe_save_path}')
        safetensors.torch.save_file(weights, safe_save_path)
        tensorrt_llm.logger.info(f'Saved safetensors to: {safe_save_path}')
        return True

    if args.workers == 1:
        for rank in range(world_size):
            passed = covert_and_save(rank)
            assert passed, "Convert checkpoint failed"
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
