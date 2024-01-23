import argparse
import functools
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BloomConfig, BloomForCausalLM, BloomTokenizerFast
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.pytorch_utils import Conv1D

# isort: off
import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.models.llama.utils import iterate_shard_files, load_state_dict  #TODO: move the utils to common dir shared by models
# isort: on


@torch.no_grad()
def apply_smoothing(scales,
                    gemm_weights,
                    layernorm_weights=None,
                    layernorm_bias=None,
                    dtype=torch.float32,
                    layernorm_1p=False):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if layernorm_weights is not None:
        assert layernorm_weights.numel() == scales.numel()
        layernorm_weights.div_(scales).to(dtype)
    if layernorm_bias is not None:
        assert layernorm_bias.numel() == scales.numel()
        layernorm_bias.div_(scales).to(dtype)
    if layernorm_1p:
        layernorm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(gemm_weights,
                act_scales,
                layernorm_weights=None,
                layernorm_bias=None,
                alpha=0.5,
                weight_scales=None):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]
    orig_dtype = gemm_weights[0].dtype

    for gemm in gemm_weights:
        # gemm_weights are expected to be transposed
        assert gemm.shape[1] == act_scales.numel()

    if weight_scales is None:
        weight_scales = torch.cat(
            [gemm.abs().max(dim=0, keepdim=True)[0] for gemm in gemm_weights],
            dim=0)
        weight_scales = weight_scales.max(dim=0)[0]
    weight_scales.to(float).clamp(min=1e-5)
    scales = (act_scales.to(gemm_weights[0].device).to(float).pow(alpha) /
              weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    apply_smoothing(scales, gemm_weights, layernorm_weights, layernorm_bias,
                    orig_dtype)

    return scales


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

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
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        input_ids = tokenizer(dataset[i]["text"],
                              return_tensors="pt",
                              max_length=seq_len,
                              truncation=True).input_ids.to(device)
        model(input_ids)

    for h in hooks:
        h.remove()

    return act_scales


def reorder_torch_qkv_weight_or_bias(v, model, is_bias=False):
    """ Reorder the qkv weight.

    Note that the shape of the fused QKV weights in HF is different from the
    shape that TRT-LLM requires.
       HF: (num_heads x 3 x head_dim, hidden_size)
       TRT-LLM: (3 x num_heads x head_dim, hidden_size)
    This is unlike to the other models in HF e.g. GPT where they have the
    same shape with TRT-LLM, i.e., (3 x num_heads x head_dim, hidden_size). We reshape the qkv
        weight: (3 x num_heads x head_dim, hidden).
        bias  : (3 x num_heads x head_dim).
    """

    n_head = model.transformer.num_heads
    hidden_size = model.transformer.embed_dim
    head_dim = hidden_size // n_head

    # (3 x hidden, ...) view as (num_heads, 3, head_dim, ...)
    v = v.reshape(n_head, 3, head_dim, -1)
    # permute to (3, num_heads, head_dim, ...)
    v = v.permute((1, 0, 2, 3))
    # final shape: weight=(3 x hidden, hidden) or bias=(3 x hidden)
    if is_bias:
        return v.reshape(3 * hidden_size)
    return v.reshape(3 * hidden_size, hidden_size)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, default=None)
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
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
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
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
        '--use_embedding_sharing',
        action="store_true",
        default=False,
        help=
        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.')
    parser.add_argument('--output_dir',
                        type=Path,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        "--smoothquant",
        "-sq",
        type=float,
        default=None,
        help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
        " to Smoothquant the model, and output int8 weights."
        " A good first try is 0.5. Must be in [0, 1]")
    parser.add_argument(
        '--per_channel',
        action="store_true",
        default=False,
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        action="store_true",
        default=False,
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers to convert checkpoint in parallel')
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()

    return args


def generate_int8(weights, act_range, is_qkv=False, multi_query_mode=False):
    """
     This function has two purposes:
      - compute quantized weights, scaled either per-tensor or per-column
      - compute scaling factors

      Depending on the GEMM API (CUTLASS/CUBLAS) the required scaling factors differ.
      CUTLASS uses two sets of scaling factors. One for the activation X, one for the weight W.
      CUBLAS only has one (we can't do per-row scaling). So we must provide pre-multiplied scaling factor.

      Here is the list of what we need (T means per-tensor, C per-column):
        - scale_x_orig_quant puts fp activation into the quantized range (i.e. [-128, 127], for int8). Used before the GEMM. (T)
        - scale_y_quant_orig puts quantized activation into the fp range. Used if the GEMM outputs int8. (T)
        - scale_w_quant_orig puts weights from quant range to fp range (used with CUTLASS) (T, C)
        - scale_y_accum_quant puts the GEMM result (XW) from accumulation range (int32)
          to quant range (int8) (used for CUBLAS) (T, C)

      Note that we don't do anything special about row-parallel GEMM. Theoretically, we could have per-GPU scaling factors too,
      but then the model would change depending on the number of GPUs used.

      For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection, we consider it
      as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
      For our GEMM implementation to respect this behavior, we use per-column mode and replicate values along columns.
    """

    # compute weight scaling factors for fp->int8 and int8->fp
    weights = weights.detach().cpu().numpy()

    if is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
            dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].reshape(3,
                                                             -1).cpu().numpy()

    elif is_qkv and multi_query_mode:
        hidden_dim = weights.shape[0]
        local_dim = act_range["w"].shape[0]
        kv_dim = (local_dim - hidden_dim) // 2
        scale_w_q = act_range["w"][0:hidden_dim]
        scale_w_k = act_range["w"][hidden_dim:hidden_dim + kv_dim]
        scale_w_v = act_range["w"][-kv_dim:]

        scale_w_qkv_t = torch.concat([
            scale_w_q.max(dim=0, keepdim=True)[0],
            scale_w_k.max(dim=0, keepdim=True)[0],
            scale_w_v.max(dim=0, keepdim=True)[0]
        ])

        scale_w_orig_quant_t = 127. / scale_w_qkv_t.cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    else:
        scale_w_orig_quant_t = 127. / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127. / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127. / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.)
    scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_t)
    scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                    scale_w_orig_quant_c)
    if is_qkv and not multi_query_mode:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t,
                                                scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t,
                                               scale_w_orig_quant_c.shape)
    if is_qkv and multi_query_mode:
        scale_q_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[0],
                                            scale_w_q.shape)
        scale_k_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[1],
                                            scale_w_k.shape)
        scale_v_y_accum_t = np.broadcast_to(scale_y_accum_quant_t[2],
                                            scale_w_v.shape)
        scale_y_accum_quant_t = np.concatenate(
            [scale_q_y_accum_t, scale_k_y_accum_t, scale_v_y_accum_t])
        scale_w_quant_orig_t = np.concatenate([
            np.broadcast_to(scale_w_quant_orig_t[0], scale_w_q.shape),
            np.broadcast_to(scale_w_quant_orig_t[1], scale_w_k.shape),
            np.broadcast_to(scale_w_quant_orig_t[2], scale_w_v.shape)
        ])

    to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)

    if is_qkv and multi_query_mode:
        scale_w_quant_orig_t_expand = np.ones([weights.shape[-1]])
        scale_w_quant_orig_t_expand[:hidden_dim] = scale_w_quant_orig_t[0]
        scale_w_quant_orig_t_expand[hidden_dim:hidden_dim +
                                    kv_dim] = scale_w_quant_orig_t[1]
        scale_w_quant_orig_t_expand[-kv_dim:] = scale_w_quant_orig_t[2]
        weight_int8 = to_i8(weights * scale_w_quant_orig_t_expand)
        weight_int8_col = to_i8(weights * scale_w_orig_quant_c)
    elif is_qkv and not multi_query_mode:
        hidden_dim = weights.shape[0]
        weight_int8 = to_i8(weights * scale_w_orig_quant_t)
        weight_int8 = weight_int8
        weight_int8_col = to_i8(weights * scale_w_orig_quant_c)
    else:
        weight_int8 = to_i8(weights * scale_w_orig_quant_t)
        weight_int8_col = to_i8(weights * scale_w_orig_quant_c)
    return {
        "weight.int8": weight_int8,
        "weight.int8.col": weight_int8_col,
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].clone()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].clone()


def reorder_qkv_weight_or_bias(v, n_head, n_hidden, is_bias=False):
    """ Reorder the qkv weight.

    Note that the shape of the fused QKV weights in HF is different from the
    shape that TRT-LLM requires.
       HF: (num_heads x 3 x head_dim, hidden_size)
       TRT-LLM: (3 x num_heads x head_dim, hidden_size)
    This is unlike to the other models in HF e.g. GPT where they have the
    same shape with TRT-LLM, i.e., (3 x num_heads x head_dim, hidden_size). Also,
    to split across attention heads in tensor parallel, we reshape the qkv
        weight: (3, num_heads x head_dim, hidden).
        bias  : (3, num_heads x head_dim).
    """

    head_dim = n_hidden // n_head

    # (3 x hidden, ...) view as (num_heads, 3, head_dim, ...)
    v = v.reshape(n_head, 3, head_dim, -1)
    # permute to (3, num_heads, head_dim, ...)
    v = v.transpose(0, 1)
    # final shape: weight=(3, hidden, hidden) or bias=(3, hidden)
    if is_bias:
        return v.reshape(3, n_hidden)
    return v.reshape(3, n_hidden, n_hidden)


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = reorder_qkv_weight_or_bias(v, n_head, n_hidden, is_bias=False)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.contiguous()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = reorder_qkv_weight_or_bias(v, n_head, n_hidden, is_bias=True)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def get_weight(config, prefix, dtype):
    return config[prefix + '.weight'].to(dtype).detach()


def get_bias(config, prefix, dtype):
    return config[prefix + '.bias'].to(dtype).detach()


def get_weight_and_bias(config, prefix, dtype):
    return get_weight(config, prefix, dtype), get_bias(config, prefix, dtype)


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8):
    results = {}
    if use_weight_only:
        v = weight.cpu().t().contiguous()
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


def add_tllm_weight(
        weights: Dict[str, torch.Tensor],
        name: str,
        param: torch.Tensor,
        quant_mode: QuantMode = QuantMode(0),
):
    assert name not in weights, f'{name} is already added.'

    if name.endswith('.weight') and quant_mode.is_weight_only():
        if quant_mode.is_int8_weight_only():
            quant_dtype = torch.int8
        elif quant_mode.is_int4_weight_only():
            quant_dtype = torch.quint4x2
        else:
            raise ValueError(
                f'Invalid configuration, got quant_mode={quant_mode}')
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                param.t().contiguous(), quant_dtype)
        weights[name] = processed_torch_weights
        scale_name = name.replace('.weight', '.per_channel_scale')
        weights[scale_name] = torch_weight_scales
    else:
        weights[name] = param.contiguous()


@torch.no_grad()
def smooth_bloom_model(model, scales, alpha, bloom_qkv_param, bloom_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(module, BloomBlock):
            continue

        # reorder qkv weight/bias and scales
        param = module.self_attention.query_key_value.weight
        param = reorder_torch_qkv_weight_or_bias(param, model, is_bias=False)

        layer_name = name + ".self_attention.query_key_value"
        act_range_qkv = scales.get(layer_name)
        # (n_head x 3 x head_dim) -> (3 x n_head x head_dim)
        act_range_qkv['w'] = reorder_torch_qkv_weight_or_bias(
            act_range_qkv['w'], model, is_bias=True)
        act_range_qkv['y'] = reorder_torch_qkv_weight_or_bias(
            act_range_qkv['y'], model, is_bias=True)
        scales[layer_name] = act_range_qkv

        # qkv_proj
        smoother = smooth_gemm(param, scales[layer_name]["x"],
                               module.input_layernorm.weight,
                               module.input_layernorm.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = param.abs().max(dim=1)[0]
        bloom_qkv_param[layer_name] = param

        # dense
        # enabled for better accuracy with perf overhead of quantiztion
        layer_name = name + ".self_attention.dense"
        smoother = smooth_gemm(module.self_attention.dense.weight,
                               scales[layer_name]["x"], None, None, alpha)
        bloom_smoother[layer_name] = smoother

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attention.dense.weight.abs().max(
            dim=1)[0]

        # fc1
        layer_name = name + ".mlp.dense_h_to_4h"
        smoother = smooth_gemm(module.mlp.dense_h_to_4h.weight,
                               scales[layer_name]["x"],
                               module.post_attention_layernorm.weight,
                               module.post_attention_layernorm.bias, alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.dense_h_to_4h.weight.abs().max(
            dim=1)[0]

        # fc2
        # enabled for better accuracy with perf overhead of quantiztion
        layer_name = name + ".mlp.dense_4h_to_h"
        smoother = smooth_gemm(module.mlp.dense_4h_to_h.weight,
                               scales[layer_name]["x"], None, None, alpha)
        bloom_smoother[layer_name] = smoother
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.dense_4h_to_h.weight.abs().max(
            dim=1)[0]


def get_tllm_linear_sq_weight(
    vals,
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
):
    results = {}

    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:

        original_weights = vals["weight.int8.col"]
        cur_weights = np.split(original_weights, tensor_parallel,
                               axis=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(
                np.array([1.0], dtype=np.float32))

        if smoother_value is None:
            cur_per_channel_value = np.split(vals["scale_w_quant_orig.col"],
                                             tensor_parallel,
                                             axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array(cur_per_channel_value,
                     dtype=np.float32).reshape(col_shape)).contiguous()
    else:
        original_weights = np.array(vals["weight.int8"])
        cur_weights = np.split(original_weights, tensor_parallel,
                               axis=cat_dim)[rank]

        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()

        cur_per_channel_value = vals["scale_y_accum_quant"]

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


def convert_hf_bloom(hf_bloom,
                     rank=0,
                     tensor_parallel=1,
                     dtype='float32',
                     use_parallel_embedding=False,
                     sharding_dim=0,
                     share_embedding_table=False,
                     use_weight_only=False,
                     plugin_weight_only_quant_type=torch.int8,
                     use_smooth_quant=False,
                     bloom_qkv_param={},
                     act_range=None,
                     smoother=None,
                     per_channel=False,
                     per_token=False,
                     int8_kv_cache=False):

    weights = {}
    tik = time.time()

    model_params = dict(hf_bloom.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_bloom.config.n_head
    hidden_size = hf_bloom.config.hidden_size

    for l in range(hf_bloom.config.num_hidden_layers):
        prefix = f'transformer.h.{l}.'
        tllm_prex = f'transformer.layers.{l}.'
        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, prefix + 'self_attention.query_key_value', dtype)
        split_v = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                               tensor_parallel, rank)
        bias = split_qkv_bias_tp(qkv_bias, num_attention_heads, hidden_size,
                                 tensor_parallel, rank)

        if use_smooth_quant:
            qkv_weight = bloom_qkv_param[prefix +
                                         'self_attention.query_key_value'].t()

            qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)
            int8_weights = generate_int8(
                qkv_weight,
                act_range.get(
                    (tllm_prex + 'self_attention.query_key_value').replace(
                        ".layers.", ".h.")),
                is_qkv=True)

            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.qkv.',
                    [1, 3 * hidden_size // tensor_parallel],
                    tensor_parallel,
                    is_qkv=True,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'input_layernorm.scale_to_int',
                    bias=bias,
                    smoother_value=None,
                    smoother_shape=None,
                    rank=rank,
                    cat_dim=-1))
        else:
            split_v = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                                   tensor_parallel, rank)
            bias = split_qkv_bias_tp(qkv_bias, num_attention_heads, hidden_size,
                                     tensor_parallel, rank)
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'attention.qkv.',
                                       bias, use_weight_only,
                                       plugin_weight_only_quant_type))
        if int8_kv_cache:
            qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)

            int8_weights = generate_int8(
                qkv_weight,
                act_range.get(
                    (tllm_prex + 'self_attention.query_key_value').replace(
                        ".layers.", ".h.")),
                is_qkv=True)

            kv_cache_weights = {}

            kv_cache_weights[
                tllm_prex +
                'attention.kv_cache_scaling_factor'] = torch.from_numpy(
                    np.array([int8_weights['scale_y_quant_orig']],
                             dtype=np.float32)).contiguous()
            weights.update(kv_cache_weights)
        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, prefix + 'self_attention.dense', dtype)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(
                attn_dense_weight,
                act_range.get((tllm_prex + 'self_attention.dense').replace(
                    ".layers.", ".h.")))

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
                    bias=attn_dense_bias,
                    smoother_value=smoother[(tllm_prex +
                                             'self_attention.dense').replace(
                                                 ".layers.", ".h.")],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=rank,
                    cat_dim=0))
        else:

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
        bias = split_matrix_tp(mlp_fc_bias, tensor_parallel, rank, dim=0)

        if use_smooth_quant:
            mlp_fc_weight = mlp_fc_weight.t()
            int8_weights = generate_int8(
                mlp_fc_weight,
                act_range.get((tllm_prex + 'mlp.dense_h_to_4h').replace(
                    ".layers.", ".h.")))

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
                    bias=bias,
                    smoother_value=None,
                    smoother_shape=None,
                    rank=rank,
                    cat_dim=-1))
        else:
            split_v = split_matrix_tp(mlp_fc_weight,
                                      tensor_parallel,
                                      rank,
                                      dim=0)
            bias = split_matrix_tp(mlp_fc_bias, tensor_parallel, rank, dim=0)
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'mlp.fc.', bias,
                                       use_weight_only,
                                       plugin_weight_only_quant_type))
        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, prefix + 'mlp.dense_4h_to_h', dtype)
        if use_smooth_quant:
            mlp_proj_weight = mlp_proj_weight.t()
            int8_weights = generate_int8(
                mlp_proj_weight,
                act_range.get((tllm_prex + 'mlp.dense_4h_to_h').replace(
                    ".layers.", ".h.")))

            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'mlp.proj.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'mlp.quantization_scaling_factor',
                    bias=mlp_proj_bias,
                    smoother_value=smoother[(tllm_prex +
                                             'mlp.dense_4h_to_h').replace(
                                                 ".layers.", ".h.")],
                    smoother_shape=[1, 4 * hidden_size // tensor_parallel],
                    rank=rank,
                    cat_dim=0))
        else:
            split_v = split_matrix_tp(mlp_proj_weight,
                                      tensor_parallel,
                                      rank,
                                      dim=1)
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
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight
        weights[tllm_prex + 'post_layernorm.bias'] = post_ln_bias

    embed_w = get_weight(model_params, 'transformer.word_embeddings', dtype)
    if not share_embedding_table:
        weights['lm_head.weight'] = split_matrix_tp(embed_w.clone(),
                                                    tensor_parallel,
                                                    rank,
                                                    dim=0)

    if not use_parallel_embedding:
        weights['transformer.vocab_embedding.weight'] = embed_w
    else:
        assert hf_bloom.config.vocab_size % tensor_parallel == 0
        weights['transformer.vocab_embedding.weight'] = split_matrix_tp(
            embed_w, tensor_parallel, rank, dim=sharding_dim)

    embed_f_w, embed_f_b = get_weight_and_bias(
        model_params, 'transformer.word_embeddings_layernorm', dtype)
    weights['transformer.ln_embed.weight'] = embed_f_w
    weights['transformer.ln_embed.bias'] = embed_f_b

    ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'transformer.ln_f',
                                         dtype)
    weights['transformer.ln_f.weight'] = ln_f_w
    weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def rename_hf_to_tllm(name: str):
    """ Rename a HF parameter name by the corresponding TRT-LLM style name. """
    if 'word_embeddings_layernorm.' in name:
        name = name.replace('word_embeddings_layernorm', 'ln_embed')
        if not name.startswith('transformer.'):
            name = f'transformer.{name}'
    elif 'word_embeddings.' in name:
        name = name.replace('word_embeddings', 'vocab_embedding')
    if name.startswith(('ln_embed.', 'vocab_embedding.', 'ln_f.')):
        name = f'transformer.{name}'

    # Parameter names in layers
    if name.startswith(('transformer.h.', 'h.')):
        import re
        name = re.sub(r'^(transformer.h.|h.)', 'transformer.layers.', name, 1)
    if 'post_attention_layernorm' in name:
        name = name.replace('post_attention_layernorm', 'post_layernorm')
    elif 'self_attention.query_key_value' in name:
        name = name.replace('self_attention.query_key_value', 'attention.qkv')
    elif 'self_attention.dense' in name:
        name = name.replace('self_attention.dense', 'attention.dense')
    elif 'mlp.dense_h_to_4h' in name:
        name = name.replace('mlp.dense_h_to_4h', 'mlp.fc')
    elif 'mlp.dense_4h_to_h' in name:
        name = name.replace('mlp.dense_4h_to_h', 'mlp.proj')
    return name


def contain_any(name: str, words: Iterable[str]):
    for word in words:
        if word in name:
            return True
    return False


def convert_from_hf_checkpoint(
    model_dir: Union[str, Path],
    rank=0,
    tensor_parallel=1,
    dtype: Union[str, torch.dtype] = torch.float32,
    use_parallel_embedding: bool = False,
    sharding_dim: int = 0,
    share_embedding_table: bool = False,
    use_weight_only: bool = False,
    plugin_weight_only_quant_type: torch.dtype = torch.int8,
    use_smooth_quant: bool = False,
    bloom_qkv_param: Optional[Dict] = None,
    act_range: Optional[Any] = None,
    smoother: Optional[Any] = None,
    per_channel: bool = False,
    per_token: bool = False,
    int8_kv_cache: bool = False,
):
    logger.info('Loading weights from HF BLOOM...')
    tik = time.time()

    weights = {}
    hf_config = BloomConfig.from_pretrained(model_dir)
    num_heads = hf_config.n_head
    hidden_size = hf_config.hidden_size
    if isinstance(dtype, str):
        dtype = tensorrt_llm.str_dtype_to_torch(dtype)
    tp_rank = rank
    tp_size = tensor_parallel

    if use_smooth_quant:
        quant_mode = QuantMode.use_smooth_quant(per_token, per_channel)
    elif use_weight_only:
        quant_mode = QuantMode.from_description(
            quantize_weights=True,
            quantize_activations=False,
            per_token=False,
            per_channel=False,
            use_int8_kv_cache=int8_kv_cache,
            use_int4_weights=plugin_weight_only_quant_type == torch.quint4x2)
    else:
        quant_mode = QuantMode(0)

    def is_bias(_name):
        return 'bias' in _name

    for model_file in iterate_shard_files(model_dir, tp_rank):
        logger.debug(f'Loading file {str(model_file)}...')
        model_params = load_state_dict(model_file, dtype=dtype)
        for name, param in model_params.items():
            logger.debug(f'Converting weight {name}...')
            tllm_name = rename_hf_to_tllm(name)
            param = param.detach().cpu()

            # TODO: Support SmmothQuant.

            if 'self_attention.query_key_value' in name:
                if not is_bias(name):
                    param = split_qkv_tp(param, num_heads, hidden_size, tp_size,
                                         tp_rank)
                    # TODO: Add KV scalers when quantizing KV cache.
                else:
                    param = split_qkv_bias_tp(param, num_heads, hidden_size,
                                              tp_size, tp_rank)
                add_tllm_weight(weights, tllm_name, param, quant_mode)
            elif 'self_attention.dense' in name:
                if not is_bias(name):
                    param = split_matrix_tp(param, tp_size, tp_rank, dim=1)
                add_tllm_weight(weights, tllm_name, param, quant_mode)
            elif 'mlp.dense_h_to_4h' in name:
                if not is_bias(name):
                    param = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                else:
                    param = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                add_tllm_weight(weights, tllm_name, param, quant_mode)
            elif 'mlp.dense_4h_to_h' in name:
                if not is_bias(name):
                    param = split_matrix_tp(param, tp_size, tp_rank, dim=1)
                add_tllm_weight(weights, tllm_name, param, quant_mode)
            elif 'word_embeddings.' in name:
                if not share_embedding_table:
                    # TODO: safetensor doesn't allow to save a shared tensor.
                    # Currently, we clone the weight but to save the disk, it
                    # would be better to skip saving lm_head weights and
                    # handle it at the loading phase.
                    lm_head = split_matrix_tp(param, tp_size, tp_rank, dim=0)
                    weights['lm_head.weight'] = lm_head.clone()
                if not use_parallel_embedding:
                    weights[tllm_name] = param
                else:
                    assert hf_config.vocab_size % tp_size == 0
                    weights[tllm_name] = split_matrix_tp(param,
                                                         tp_size,
                                                         tp_rank,
                                                         dim=sharding_dim)
            elif contain_any(name,
                             ('input_layernorm', 'post_attention_layernorm',
                              'word_embeddings_layernorm.', 'ln_f.')):
                weights[tllm_name] = param
        del model_params

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return weights


def do_convert_from_ckpt(args):
    return (args.model_dir.exists() and args.smoothquant is None
            and not args.use_weight_only and not args.int8_kv_cache)


def convert(worker_rank, args, convert_args):
    convert_from_ckpt = do_convert_from_ckpt(args)
    for rank in range(worker_rank, args.world_size, args.workers):
        if convert_from_ckpt:
            weights = convert_from_hf_checkpoint(rank=rank, **convert_args)
        else:
            weights = convert_hf_bloom(rank=rank, **convert_args)
        safetensors.torch.save_file(weights,
                                    args.output_dir / f'rank{rank}.safetensors')


def main():
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)

    args = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()

    args.output_dir.mkdir(exist_ok=True, parents=True)

    hf_config = BloomConfig.from_pretrained(args.model_dir)
    config = {
        'architecture': hf_config.architectures[0],
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'hidden_size': hf_config.hidden_size,
        'norm_epsilon': hf_config.layer_norm_epsilon,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type': 'alibi',
        'hidden_act': 'gelu',
        'intermediate_size': hf_config.hidden_size * 4,
        'quantization': {
            'use_weight_only': args.use_weight_only,
            'weight_only_precision': args.weight_only_precision,
            'int8_kv_cache': args.int8_kv_cache,
            'use_smooth_quant': args.smoothquant is not None,
            'per_channel': args.smoothquant is not None and args.per_channel,
            'per_token': args.smoothquant is not None and args.per_token,
        },
        'mapping': {
            'world_size': args.world_size,
            'tp_size': args.world_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
    }

    with (args.output_dir / 'config.json').open('w') as f:
        json.dump(config, f, indent=4)

    # TODO: convert_from_hf_checkpoint is memory efficient but has not
    # supported quantization yet. Will enable once implemented.
    convert_from_ckpt = do_convert_from_ckpt(args)
    if not convert_from_ckpt:
        logger.info(f'Convert by using model')
        hf_bloom = BloomForCausalLM.from_pretrained(args.model_dir,
                                                    torch_dtype="auto",
                                                    device_map="auto",
                                                    trust_remote_code=True)
    else:
        logger.info(f'Convert by using checkpoint')
        hf_bloom = None

    act_range = {}
    bloom_qkv_param = {}
    bloom_smoother = {}

    if args.smoothquant is not None or args.int8_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset
        dataset = load_dataset("lambada", split="validation", cache_dir=None)
        act_range = capture_activation_range(
            hf_bloom, BloomTokenizerFast.from_pretrained(args.model_dir),
            dataset)
        if args.smoothquant is not None:
            smooth_bloom_model(hf_bloom, act_range, args.smoothquant,
                               bloom_qkv_param, bloom_smoother)

    if args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
    elif args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    convert_args = dict(
        tensor_parallel=args.world_size,
        dtype=args.dtype,
        use_weight_only=args.use_weight_only,
        plugin_weight_only_quant_type=plugin_weight_only_quant_type,
        use_parallel_embedding=args.use_parallel_embedding,
        sharding_dim=args.embedding_sharding_dim,
        share_embedding_table=args.use_embedding_sharing,
        use_smooth_quant=args.smoothquant,
        act_range=act_range,
        bloom_qkv_param=bloom_qkv_param,
        smoother=bloom_smoother,
        per_channel=args.per_channel,
        per_token=args.per_token,
        int8_kv_cache=args.int8_kv_cache,
    )
    if convert_from_ckpt:
        convert_args['model_dir'] = args.model_dir
    else:
        convert_args['hf_bloom'] = hf_bloom

    if args.workers == 1:
        convert(0, args, convert_args)
    else:
        if args.workers > args.world_size:
            args.workers = args.world_size
        logger.info(f'Convert checkpoint using {args.workers} workers.')
        import torch.multiprocessing as mp
        mp.spawn(convert, nprocs=args.workers, args=(args, convert_args))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
