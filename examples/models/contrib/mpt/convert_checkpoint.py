import argparse
import copy
import functools
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, MptConfig, MptForCausalLM
from transformers.pytorch_utils import Conv1D

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import (generate_int8, get_weight,
                                               load_calib_dataset, smooth_gemm,
                                               split)
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
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
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

    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument(
        "--calibrate_kv_cache",
        "-kv",
        action="store_true",
        help=
        "Generate scaling factors for KV cache. Used for storing KV cache in int8."
    )
    parser.add_argument(
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument("--dataset_cache_dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
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
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
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


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=1,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token = tokenizer.eos_token

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(
                1e-8, None).max(dim=1)[0].float()

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        datapoint = dataset[i:i + 1]
        line = copy.copy(datapoint)
        line[0] = line[0] + ' TL;DR: '
        line[0] = line[0].strip()
        line[0] = line[0].replace(" n't", "n't")
        input_ids = tokenizer(line,
                              return_tensors="pt",
                              max_length=seq_len,
                              padding=True,
                              truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def smooth_mpt_model(model, scales, alpha, mpt_qkv_para, mpt_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, type(model.transformer.blocks[0])):
            continue
        # qkv_proj
        layer_name_qkv = name + ".attn.Wqkv"
        weight = module.attn.Wqkv.weight
        smoother = smooth_gemm(weight, scales[layer_name_qkv]["x"],
                               module.norm_1.weight, module.norm_1.bias, alpha)
        scales[layer_name_qkv]["x"] = scales[layer_name_qkv]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        # see transpose_weights function
        mpt_qkv_para[layer_name_qkv] = weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".attn.out_proj"
        smoother = smooth_gemm(module.attn.out_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        mpt_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.out_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        fc1_layer_name = name + ".ffn.up_proj"

        smoother = smooth_gemm(module.ffn.up_proj.weight,
                               scales[fc1_layer_name]["x"],
                               module.norm_2.weight, module.norm_2.bias, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.ffn.up_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        layer_name = name + ".ffn.down_proj"
        smoother = smooth_gemm(module.ffn.down_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        mpt_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.ffn.down_proj.weight.abs().max(
            dim=1)[0]


def get_tllm_linear_sq_weight(vals,
                              prefix,
                              shape,
                              tensor_parallel,
                              is_qkv=False,
                              per_token=False,
                              per_channel=False,
                              last_prefix=None,
                              bias=None,
                              smoother_value=None,
                              smoother_shape=None,
                              rank=0,
                              cat_dim=0,
                              multi_query_mode=False):
    results = {}

    def multi_query_split(data, local_dim, head_size, tp_size, cur_rank):
        q, k, v = np.split(data, [local_dim, local_dim + head_size], axis=-1)
        q_split = np.split(q, tp_size, axis=-1)
        k_split = np.split(k, tp_size, axis=-1)
        v_split = np.split(v, tp_size, axis=-1)
        return [
            np.concatenate((q_split[ii], k_split[ii], v_split[ii]), axis=-1)
            for ii in range(tp_size)
        ][cur_rank]

    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
        if per_channel:
            original_weights = np.array(vals["weight.int8.col"])
        else:
            original_weights = np.array(vals["weight.int8"])
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(original_weights, local_dim,
                                            head_size, tensor_parallel, rank)
        else:
            cur_weights = np.split(original_weights,
                                   tensor_parallel,
                                   axis=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(
                np.array([1.0], dtype=np.float32))

        if per_channel:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
            if smoother_value is None:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig.col"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_w_quant_orig.col"],
                        tensor_parallel,
                        axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig"]
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(vals["scale_w_quant_orig"],
                                                     tensor_parallel,
                                                     axis=cat_dim)[rank]

        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array(cur_per_channel_value,
                     dtype=np.float32).reshape(col_shape)).contiguous()
    else:
        if per_channel:
            original_weights = np.array(vals["weight.int8.col"])
        else:
            original_weights = np.array(vals["weight.int8"])
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(original_weights, local_dim,
                                            head_size, tensor_parallel, rank)
        else:
            cur_weights = np.split(original_weights,
                                   tensor_parallel,
                                   axis=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()

        if per_channel:
            cur_per_channel_value = vals["scale_y_accum_quant.col"]
            if smoother_value is None:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant.col"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant.col"],
                        tensor_parallel,
                        axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_y_accum_quant"]
            # QKV is always per_channel
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant"],
                        tensor_parallel,
                        axis=cat_dim)[rank]

        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array([cur_per_channel_value],
                     dtype=np.float32).reshape(col_shape)).contiguous()

        results[last_prefix] = torch.from_numpy(
            np.array([vals['scale_x_orig_quant']],
                     dtype=np.float32)).contiguous()

        results[prefix + 'act_scale'] = torch.from_numpy(
            np.array([[vals["scale_y_quant_orig"]]],
                     dtype=np.float32)).contiguous()

    if smoother_value is not None:
        cur_smoother_value = np.split(smoother_value,
                                      tensor_parallel,
                                      axis=cat_dim)[rank]
        results[prefix + 'smoother'] = cur_smoother_value.reshape(
            smoother_shape).contiguous().to(torch.float32)

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def split_qkv_tp(qkv, n_head, n_kv_heads, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    kv_head_size = n_kv_heads * (n_hidden // n_head)
    q, k, v = torch.split(qkv, [n_hidden, kv_head_size, kv_head_size], dim=0)
    q = split(q, tensor_parallel, rank, dim=0)
    k = split(k, tensor_parallel, rank, dim=0)
    v = split(v, tensor_parallel, rank, dim=0)
    return torch.concatenate([q, k, v], dim=0).contiguous()


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
        results[f'{prefix}.weight'] = weight.contiguous()

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


def convert_hf_mpt_legacy(hf_model,
                          hf_config,
                          mapping,
                          rank=0,
                          dtype='float32',
                          use_parallel_embedding: bool = False,
                          sharding_dim: int = 0,
                          use_weight_only=False,
                          plugin_weight_only_quant_type='int8',
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
    num_attention_heads = hf_model.config.n_heads
    hidden_size = hf_model.config.d_model
    vocab_size = hf_model.config.vocab_size
    num_key_value_heads = hf_config.attn_config['kv_n_heads'] if 'kv_n_heads' in hf_config.attn_config \
        else hf_config.n_heads
    multi_query_mode = (num_key_value_heads != num_attention_heads)

    for l in range(hf_model.config.n_layers):
        prefix = f'transformer.blocks.{l}.'
        tllm_prex = f'transformer.layers.{l}.'

        # attn.Wqkv -> attention.qkv
        qkv_weight = get_weight(model_params, prefix + 'attn.Wqkv', dtype)

        if use_smooth_quant:
            qkv_out_dim = qkv_weight.shape[0]
            qkv_weight = qkv_weight.t().numpy()
            if not multi_query_mode:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)
            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix + 'attn.Wqkv'),
                                         is_qkv=True,
                                         multi_query_mode=multi_query_mode)
            weights.update(
                get_tllm_linear_sq_weight(int8_weights,
                                          tllm_prex + 'attention.qkv.',
                                          [1, qkv_out_dim // tensor_parallel],
                                          tensor_parallel,
                                          is_qkv=True,
                                          per_token=per_token,
                                          per_channel=per_channel,
                                          last_prefix=tllm_prex +
                                          'input_layernorm.scale_to_int',
                                          smoother_value=None,
                                          smoother_shape=None,
                                          rank=rank,
                                          cat_dim=-1,
                                          multi_query_mode=multi_query_mode))
        else:
            qkv_weight = split_qkv_tp(qkv_weight, num_attention_heads,
                                      num_key_value_heads, hidden_size,
                                      mapping.tp_size, mapping.tp_rank)
            weights.update(
                get_tllm_linear_weight(qkv_weight, tllm_prex + 'attention.qkv',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

        if int8_kv_cache:
            qkv_weight = get_weight(model_params, prefix + 'attn.Wqkv', dtype)
            qkv_weight = qkv_weight.t().numpy()
            if not multi_query_mode:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)
            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix + 'attn.Wqkv'),
                                         is_qkv=True,
                                         multi_query_mode=multi_query_mode)
            weights[tllm_prex +
                    'attention.kv_cache_scaling_factor'] = torch.from_numpy(
                        np.array([int8_weights['scale_y_quant_orig']],
                                 dtype=np.float32)).contiguous()

        # attn.out_proj -> attention.dense
        attn_dense_weight = get_weight(model_params, prefix + 'attn.out_proj',
                                       dtype)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t().numpy()
            int8_weights = generate_int8(
                attn_dense_weight, act_range.get(prefix + 'attn.out_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.dense.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex +
                    'attention.quantization_scaling_factor',
                    smoother_value=smoother[(prefix + 'attn.out_proj')],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=rank,
                    cat_dim=0))
        else:
            attn_dense_w = split_matrix(attn_dense_weight,
                                        mapping.tp_size,
                                        mapping.tp_rank,
                                        dim=1)
            weights.update(
                get_tllm_linear_weight(attn_dense_w,
                                       tllm_prex + 'attention.dense', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type))

        # ffn.up_proj -> mlp.fc
        mlp_fc_weight = get_weight(model_params, prefix + 'ffn.up_proj', dtype)
        if use_smooth_quant:
            mlp_fc_weight = mlp_fc_weight.t().numpy()
            int8_weights = generate_int8(mlp_fc_weight,
                                         act_range.get(prefix + 'ffn.up_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'mlp.fc.',
                    [1, 4 * hidden_size // tensor_parallel],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                    smoother_value=None,
                    smoother_shape=None,
                    rank=rank,
                    cat_dim=-1))
        else:
            mlp_fc_weight = split_matrix(mlp_fc_weight,
                                         mapping.tp_size,
                                         mapping.tp_rank,
                                         dim=0)
            weights.update(
                get_tllm_linear_weight(mlp_fc_weight, tllm_prex + 'mlp.fc',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

        # ffn.down_proj -> mlp.proj
        mlp_proj_weight = get_weight(model_params, prefix + 'ffn.down_proj',
                                     dtype)
        if use_smooth_quant:
            mlp_proj_weight = mlp_proj_weight.t().numpy()
            int8_weights = generate_int8(
                mlp_proj_weight, act_range.get(prefix + 'ffn.down_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'mlp.proj.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'mlp.quantization_scaling_factor',
                    smoother_value=smoother[prefix + 'ffn.down_proj'],
                    smoother_shape=[1, 4 * hidden_size // tensor_parallel],
                    rank=rank,
                    cat_dim=0))
        else:
            mlp_proj_weight = split_matrix(mlp_proj_weight,
                                           mapping.tp_size,
                                           mapping.tp_rank,
                                           dim=1)
            weights.update(
                get_tllm_linear_weight(mlp_proj_weight, tllm_prex + 'mlp.proj',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type))

        # input layer_norm
        input_ln_weight = get_weight(model_params, prefix + 'norm_1', dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        # post layer_norm
        post_ln_weight = get_weight(model_params, prefix + 'norm_2', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

    embed_w = get_weight(model_params, 'transformer.wte', dtype)
    if mapping.is_first_pp_rank():
        # Embedding
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
        # lm_head weight and bias
        weights['lm_head.weight'] = split_matrix(embed_w.clone(),
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 dim=0)
        ln_f_w = get_weight(model_params, 'transformer.norm_f', dtype)
        # ln_f weight and bias
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def convert_hf_mpt(hf_model: MptForCausalLM,
                   hf_config: MptConfig,
                   mapping: Mapping,
                   dtype: str = 'float32',
                   use_parallel_embedding: bool = False,
                   sharding_dim: int = 0,
                   use_weight_only: bool = False,
                   plugin_weight_only_quant_type: torch.dtype = torch.int8):

    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_hidden_layers = hf_config.n_layers
    num_head = hf_config.n_heads
    num_kv_heads = getattr(hf_config.attn_config, 'kv_n_heads',
                           hf_config.n_heads)
    hidden_size = hf_config.d_model
    vocab_size = hf_config.vocab_size

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f'transformer.blocks.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # Attention QKV (no bias)
        qkv_w = get_weight(model_params, f'{prefix}.attn.Wqkv', dtype)
        qkv_w = split_qkv_tp(qkv_w, num_head, num_kv_heads, hidden_size,
                             mapping.tp_size, mapping.tp_rank)
        weights.update(
            get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))
        # Attention dense (no bias)
        attn_dense_weight = get_weight(model_params, f'{prefix}.attn.out_proj',
                                       dtype)
        attn_dense_w = split_matrix(attn_dense_weight,
                                    mapping.tp_size,
                                    mapping.tp_rank,
                                    dim=1)
        weights.update(
            get_tllm_linear_weight(attn_dense_w, f'{tllm_prex}.attention.dense',
                                   None, use_weight_only,
                                   plugin_weight_only_quant_type))
        # MLP fc_in (no bias)
        mlp_fc_weight = get_weight(model_params, f'{prefix}.ffn.up_proj', dtype)
        mlp_fc_w = split_matrix(mlp_fc_weight,
                                mapping.tp_size,
                                mapping.tp_rank,
                                dim=0)
        weights.update(
            get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))
        # MLP fc_out (no bias)
        mlp_proj_weight = get_weight(model_params, f'{prefix}.ffn.down_proj',
                                     dtype)
        mlp_proj_w = split_matrix(mlp_proj_weight,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  dim=1)
        weights.update(
            get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj', None,
                                   use_weight_only,
                                   plugin_weight_only_quant_type))
        # input layer_norm
        input_ln_weight = get_weight(model_params, f'{prefix}.norm_1', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight

        # post layer_norm
        post_ln_weight = get_weight(model_params, f'{prefix}.norm_2', dtype)
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight

    embed_w = get_weight(model_params, 'transformer.wte', dtype)
    if mapping.is_first_pp_rank():
        # Embedding
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
        # lm_head weight and bias
        weights['lm_head.weight'] = split_matrix(embed_w.clone(),
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 dim=0)
        ln_f_w = get_weight(model_params, 'transformer.norm_f', dtype)
        # ln_f weight and bias
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    world_size = args.tp_size * args.pp_size
    quant_algo = None
    plugin_weight_only_quant_type = None
    if args.use_weight_only and args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
        quant_algo = QuantAlgo.W8A16
    elif args.use_weight_only and args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2
        quant_algo = QuantAlgo.W4A16

    if args.smoothquant:
        if args.per_token and args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
        elif not args.per_token and not args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        elif not args.per_token and args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        elif args.per_token and not args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN

    if args.calibrate_kv_cache:
        kv_cache_quant_algo = QuantAlgo.INT8
    else:
        kv_cache_quant_algo = None

    hf_config = MptConfig.from_pretrained(args.model_dir,
                                          trust_remote_code=True)
    num_kv_heads = getattr(hf_config.attn_config, 'kv_n_heads',
                           hf_config.n_heads)
    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'logits_dtype': args.logits_dtype,
        'vocab_size': hf_config.vocab_size,
        'hidden_size': hf_config.d_model,
        'intermediate_size': hf_config.d_model * 4,
        'num_hidden_layers': hf_config.n_layers,
        'num_attention_heads': hf_config.n_heads,
        'num_key_value_heads': num_kv_heads,
        'position_embedding_type': 'alibi',
        'hidden_act': 'gelu',
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'quantization': {
            'quant_algo': quant_algo,
            'kv_cache_quant_algo': kv_cache_quant_algo,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'bias': (not hf_config.no_bias),
        'clip_qkv': hf_config.attn_config.clip_qkv,
        'alibi_bias_max': hf_config.attn_config.alibi_bias_max
    }

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    hf_model = MptForCausalLM.from_pretrained(args.model_dir,
                                              device_map="auto",
                                              dtype=getattr(torch, args.dtype))

    act_range = {}
    mpt_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    mpt_smoother = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                                  padding_side='left')
        dataset = load_calib_dataset(args.calib_dataset,
                                     cache_dir=args.dataset_cache_dir)

        act_range = capture_activation_range(hf_model, tokenizer, dataset)
        if args.smoothquant is not None:
            smooth_mpt_model(hf_model, act_range, args.smoothquant,
                             mpt_qkv_para, mpt_smoother)

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        if args.smoothquant is not None or args.calibrate_kv_cache:
            weights = convert_hf_mpt_legacy(
                hf_model,
                hf_config,
                mapping,
                rank,
                dtype=args.dtype,
                use_parallel_embedding=args.use_parallel_embedding,
                sharding_dim=args.embedding_sharding_dim,
                use_weight_only=args.use_weight_only,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                use_smooth_quant=(args.smoothquant is not None),
                per_channel=args.per_channel,
                per_token=args.per_token,
                int8_kv_cache=args.calibrate_kv_cache,
                act_range=act_range,
                qkv_para=mpt_qkv_para,
                smoother=mpt_smoother)
        else:
            weights = convert_hf_mpt(
                hf_model,
                hf_config,
                mapping,
                dtype=args.dtype,
                use_parallel_embedding=args.use_parallel_embedding,
                sharding_dim=args.embedding_sharding_dim,
                use_weight_only=args.use_weight_only,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)

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

    del hf_model
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')
