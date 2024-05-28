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
import argparse
import copy
import functools
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

import tensorrt_llm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import (load_calib_dataset,
                                               weight_only_quantize_dict)
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--model_version',
                        type=str,
                        default='v1_13b',
                        choices=['v1_7b', 'v1_13b', 'v2_7b', 'v2_13b'])
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='ccdv/cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
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
    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ/AWQ quantization.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
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
    else:
        weight_int8 = to_i8(weights * scale_w_orig_quant_t)
    return {
        "weight.int8": weight_int8,
        "weight.int8.col": to_i8(weights * scale_w_orig_quant_c),
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


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
def smooth_gemm_fc1_gate(fc1_weights,
                         gate_weights,
                         act_scales,
                         layernorm_weights=None,
                         layernorm_bias=None,
                         alpha=0.5,
                         weight_scales=None):
    gemm_weights = []
    if not isinstance(fc1_weights, list):
        fc1_weights = [fc1_weights]
    if not isinstance(gate_weights, list):
        gate_weights = [gate_weights]

    for i in range(len(fc1_weights)):
        gemm_weight = torch.cat([fc1_weights[i], gate_weights[i]], dim=0)
        gemm_weights.append(gemm_weight)

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

    apply_smoothing(scales, fc1_weights + gate_weights, layernorm_weights,
                    layernorm_bias, orig_dtype)

    return scales


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    test_token_num = 923
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
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=1)[0]

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
        line_encoded = tokenizer(line,
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True)["input_ids"].type(torch.int64)
        line_encoded = line_encoded[:, -test_token_num:]
        line_encoded = line_encoded.cuda()
        model(line_encoded)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def smooth_baichuan_model(model, scales, alpha, baichuan_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if not 'Layer' in class_name:
            continue
        print(f'smoothing module: {name}, class_name: {class_name}')
        # qkv_proj
        layer_name_qkv = name + ".self_attn.W_pack"

        smoother = smooth_gemm(module.self_attn.W_pack.weight,
                               scales[layer_name_qkv]["x"],
                               module.input_layernorm.weight, None, alpha)

        scales[layer_name_qkv]["x"] = scales[layer_name_qkv]["x"] / smoother
        scales[layer_name_qkv]["w"] = module.self_attn.W_pack.weight.abs().max(
            dim=1)[0].float()

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        baichuan_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(
            dim=1)[0].float()

        # ==================================================================
        fc1_layer_name = name + ".mlp.gate_proj"
        gate_layer_name = name + ".mlp.up_proj"

        smoother = smooth_gemm_fc1_gate(module.mlp.gate_proj.weight,
                                        module.mlp.up_proj.weight,
                                        scales[fc1_layer_name]["x"],
                                        module.post_attention_layernorm.weight,
                                        None, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.gate_proj.weight.abs().max(
            dim=1)[0].float()

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.up_proj.weight.abs().max(
            dim=1)[0].float()

        # ==================================================================
        layer_name = name + ".mlp.down_proj"
        smoother = smooth_gemm(module.mlp.down_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        baichuan_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(
            dim=1)[0].float()


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


def split(weight: torch.Tensor,
          tp_size: int,
          rank: int = 0,
          dim: int = 0) -> torch.Tensor:
    if tp_size == 1:
        return weight
    elif weight.ndim == 1:
        return torch.chunk(weight, tp_size)[rank].contiguous()
    else:
        return torch.chunk(weight, tp_size, dim=dim)[rank].contiguous()


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


def get_weight(params: Dict[str, torch.Tensor], prefix: str,
               dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.weight' not in params:
        return None
    return params[f'{prefix}.weight'].to(dtype).detach().cpu()


def get_bias(params: Dict[str, torch.Tensor], prefix: str,
             dtype: torch.dtype) -> torch.Tensor:
    if f'{prefix}.bias' not in params:
        return None
    return params[f'{prefix}.bias'].to(dtype).detach().cpu()


def get_weight_and_bias(params: Dict[str, torch.Tensor], prefix: str,
                        dtype: torch.dtype) -> Tuple[torch.Tensor]:
    return get_weight(params, prefix, dtype), get_bias(params, prefix, dtype)


def get_tllm_linear_weight(
    weight: torch.Tensor,
    prefix: str,
    bias: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    results = {}
    results[f'{prefix}.weight'] = weight.contiguous()
    if bias is not None:
        results[f'{prefix}.bias'] = bias
    return results


def load_baichuan_config(model_dir: str) -> AutoConfig:
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    return config


def convert_hf_baichuan_sq(hf_model,
                           mapping,
                           rank=0,
                           dtype='float32',
                           per_channel=False,
                           per_token=False,
                           int8_kv_cache=False,
                           act_range=[],
                           smoother=[]):
    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    inter_size = hf_model.config.intermediate_size
    num_key_value_heads = hf_model.config.num_attention_heads
    multi_query_mode = (num_key_value_heads != num_attention_heads)

    for l in range(hf_model.config.num_hidden_layers):
        prefix = f'model.layers.{l}.'
        tllm_prex = f'transformer.layers.{l}.'

        # self_attn.W_pack -> attention.qkv
        qkv_weight = get_weight(model_params, prefix + 'self_attn.W_pack',
                                dtype)
        qkv_weight = qkv_weight.t().numpy()
        qkv_out_dim = qkv_weight.shape[1]
        if not multi_query_mode:
            qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)
        int8_weights = generate_int8(qkv_weight,
                                     act_range.get(prefix + 'self_attn.W_pack'),
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

        if int8_kv_cache:
            qkv_weight = get_weight(model_params, prefix + 'self_attn.W_pack',
                                    dtype)
            qkv_weight = qkv_weight.t().numpy()
            if not multi_query_mode:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)
            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix +
                                                       'self_attn.W_pack'),
                                         is_qkv=True,
                                         multi_query_mode=multi_query_mode)
            weights[tllm_prex +
                    'attention.kv_cache_scaling_factor'] = torch.from_numpy(
                        np.array([int8_weights['scale_y_quant_orig']],
                                 dtype=np.float32)).contiguous()

        # attn.out_proj -> attention.dense
        attn_dense_weight = get_weight(model_params,
                                       prefix + 'self_attn.o_proj', dtype)
        attn_dense_weight = attn_dense_weight.t().numpy()
        int8_weights = generate_int8(attn_dense_weight,
                                     act_range.get(prefix + 'self_attn.o_proj'))
        weights.update(
            get_tllm_linear_sq_weight(
                int8_weights,
                tllm_prex + 'attention.dense.', [1, hidden_size],
                tensor_parallel,
                is_qkv=False,
                per_token=per_token,
                per_channel=per_channel,
                last_prefix=tllm_prex + 'attention.quantization_scaling_factor',
                smoother_value=smoother[(prefix + 'self_attn.o_proj')],
                smoother_shape=[1, hidden_size // tensor_parallel],
                rank=rank,
                cat_dim=0))

        # mlp.gate_proj -> mlp.fc
        mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                   dtype)
        mlp_fc_weight = mlp_fc_weight.t().numpy()
        int8_weights = generate_int8(mlp_fc_weight,
                                     act_range.get(prefix + 'mlp.gate_proj'))
        weights.update(
            get_tllm_linear_sq_weight(
                int8_weights,
                tllm_prex + 'mlp.fc.', [1, inter_size // tensor_parallel],
                tensor_parallel,
                is_qkv=False,
                per_token=per_token,
                per_channel=per_channel,
                last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                smoother_value=None,
                smoother_shape=None,
                rank=rank,
                cat_dim=-1))

        # mlp.down_proj -> mlp.proj
        mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                     dtype)
        mlp_proj_weight = mlp_proj_weight.t().numpy()
        int8_weights = generate_int8(mlp_proj_weight,
                                     act_range.get(prefix + 'mlp.down_proj'))
        weights.update(
            get_tllm_linear_sq_weight(
                int8_weights,
                tllm_prex + 'mlp.proj.', [1, hidden_size],
                tensor_parallel,
                is_qkv=False,
                per_token=per_token,
                per_channel=per_channel,
                last_prefix=tllm_prex + 'mlp.quantization_scaling_factor',
                smoother_value=smoother[prefix + 'mlp.down_proj'],
                smoother_shape=[1, inter_size // tensor_parallel],
                rank=rank,
                cat_dim=0))

        # mlp.up_proj -> mlp.gate
        mlp_gate_weight = get_weight(model_params, prefix + 'mlp.up_proj',
                                     dtype)
        mlp_gate_weight = mlp_gate_weight.t().numpy()
        int8_weights = generate_int8(mlp_gate_weight,
                                     act_range.get(prefix + 'mlp.up_proj'))
        weights.update(
            get_tllm_linear_sq_weight(
                int8_weights,
                tllm_prex + 'mlp.gate.', [1, inter_size // tensor_parallel],
                tensor_parallel,
                is_qkv=False,
                per_token=per_token,
                per_channel=per_channel,
                last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                smoother_value=None,
                smoother_shape=None,
                rank=rank,
                cat_dim=-1))

        # input layer_norm
        input_ln_weight = get_weight(model_params, prefix + 'input_layernorm',
                                     dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        # post layer_norm
        post_ln_weight = get_weight(model_params,
                                    prefix + 'post_attention_layernorm', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

    embed_w = get_weight(model_params, 'model.embed_tokens', dtype)
    if mapping.is_first_pp_rank():
        # Embedding
        weights['transformer.vocab_embedding.weight'] = embed_w
    lm_head_w = get_weight(model_params, 'lm_head', dtype)
    if mapping.is_last_pp_rank():
        # lm_head weight and bias
        weights['lm_head.weight'] = split_matrix(lm_head_w.clone(),
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 dim=0)
        ln_f_w = get_weight(model_params, 'model.norm', dtype)
        # ln_f weight and bias
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def convert_hf_baichuan(hf_model: AutoModelForCausalLM,
                        hf_config: AutoConfig,
                        model_version: str,
                        mapping: Mapping,
                        dtype: str = 'float32',
                        quant_algo: str = None):

    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_hidden_layers = hf_config.num_hidden_layers
    hf_key = [
        "model.embed_tokens.weight",  # vocab_embedding
        "lm_head.weight",  # lm_head
        "model.norm.weight",  # ln_f
        "self_attn.W_pack.weight",  # attention.qkv
        "self_attn.o_proj.weight",  # attention.dense
        "mlp.up_proj.weight",  # mlp.gate
        "mlp.down_proj.weight",  # mlp.proj
        "mlp.gate_proj.weight",  # mlp.fc
        "input_layernorm.weight",  # input_layernorm
        "post_attention_layernorm.weight",  # post_layernorm
    ]

    def load(key_id, layer_idx=-1, tp_dim=-1, quant=False):
        prefix = "" if layer_idx == -1 else f"model.layers.{layer_idx}."
        v = model_params[prefix + hf_key[key_id]]
        if key_id == 3:
            q_emb = v.shape[0] // 3
            model_emb = v.shape[1]
            v = v.reshape(3, q_emb, model_emb)
            if v.shape[1] % mapping.tp_size != 0:
                tensorrt_llm.logger.error(
                    "Current weight shape is invalid for mapping.tp_size=" +
                    str(mapping.tp_size))
            v = v.split(v.shape[1] // mapping.tp_size, dim=1)[mapping.tp_rank]
            v = v.reshape(3 * (q_emb // mapping.tp_size), model_emb)
        if tp_dim >= 0:
            if v.shape[tp_dim] % mapping.tp_size != 0:
                tensorrt_llm.logger.error(
                    "Current weight shape is invalid for mapping.tp_size=" +
                    str(mapping.tp_size))
            v = v.split(v.shape[tp_dim] // mapping.tp_size,
                        dim=tp_dim)[mapping.tp_rank]
        v = v.to(dtype).contiguous().detach().cpu()
        return v

    # Convert vocab_embedding
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = load(0)

    # Convert lm_head
    v = load(1, -1, 0)
    if model_version.startswith('v2'):
        v = torch.nn.functional.normalize(v)
    if mapping.is_last_pp_rank():
        weights['lm_head.weight'] = v

    # Convert ln_f
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = load(2)

    # Convert layers
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        prefix = f"transformer.layers.{l}."
        weights[prefix + 'attention.qkv.weight'] = load(3, l)
        weights[prefix + 'attention.dense.weight'] = load(4, l, 1)
        weights[prefix + 'mlp.gate.weight'] = load(5, l, 0)
        weights[prefix + 'mlp.proj.weight'] = load(6, l, 1)
        weights[prefix + 'mlp.fc.weight'] = load(7, l, 0)
        weights[prefix + 'input_layernorm.weight'] = load(8, l)
        weights[prefix + 'post_layernorm.weight'] = load(9, l)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')

    return weight_only_quantize_dict(weights,
                                     quant_algo=quant_algo,
                                     plugin=True)


def convert_baichuan_gptq(hf_config: AutoConfig,
                          quant_ckpt_path: str,
                          model_version: str,
                          mapping=Mapping(),
                          dtype="float16"):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise GPTQ Baichuan safetensors...')
    weights = {}
    tik = time.time()

    gptq_baichuan = safetensors.safe_open(quant_ckpt_path,
                                          framework="pt",
                                          device=0)
    gptq_prefix = "model."
    gptq_suffix_list = [".qweight", ".qzeros", ".scales"]
    gptq_key_list = [
        "embed_tokens.weight",  # vocab_embedding
        "lm_head.weight",  # lm_head
        "norm.weight",  # ln_f
        "self_attn.W_pack",  # attention.qkv
        "_proj",  #
        "self_attn.o_proj",  # attention.dense
        "mlp.up_proj",  # mlp.gate
        "mlp.down_proj",  # mlp.proj
        "mlp.gate_proj",  # mlp.fc
        "input_layernorm.weight",  # input_layernorm
        "post_attention_layernorm.weight",  # post_layernorm
    ]

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = getattr(torch, dtype)

    def load(key, no_prefix=0):
        if no_prefix:
            return gptq_baichuan.get_tensor(key)
        else:
            return gptq_baichuan.get_tensor(gptq_prefix + key)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def unpack_int32_into_int8(w_packed):
        # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def process_and_assign_weight(prefix, v, tp_dim=-1):
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

        # return processed interleaved weight, original scales and zeros * scales
        weights[prefix + ".weight"] = qweight_interleaved
        weights[prefix + ".weights_scaling_factor"] = scales_fp16
        weights[prefix + ".zero"] = zeros_x_scales_fp16

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = load(gptq_key_list[0])
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)

    # 2. lm_head
    original_v = load(gptq_key_list[1], "no_prefix")
    if model_version.startswith('v2'):
        # baichuan v2 models use NormHead
        tensorrt_llm.logger.info(
            f'Normalizing lm_head.weight for {model_version}')
        v = torch_split(torch.nn.functional.normalize(original_v), 0)
    else:
        v = torch_split(original_v, 0)
    if mapping.is_last_pp_rank():
        weights['lm_head.weight'] = v.to(torch_dtype)

    # 3. ln_f
    v = load(gptq_key_list[2])
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)

    # 4. Weights inside each layer
    num_hidden_layers = hf_config.num_hidden_layers
    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        layer_idx = l - layers_range[0]
        prefix = f"layers.{l}."
        tllm_prefix = f"transformer.layers.{l}."
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')

        # 4.1 attention.qkv
        qkv_weight_list = []
        for suf in gptq_suffix_list:
            qkv_list = []
            comp_part = load(prefix + gptq_key_list[3] + suf)
            qkv = torch.chunk(comp_part, 3, 1)
            for i in range(3):
                comp_part = qkv[i]
                comp_part = torch_split(comp_part, 1)
                qkv_list.append(comp_part)
            qkv_weight_list.append(torch.cat(qkv_list, dim=1))

        process_and_assign_weight(tllm_prefix + "attention.qkv",
                                  qkv_weight_list)

        # 4.2 attention.dense
        v = [load(prefix + gptq_key_list[5] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(tllm_prefix + "attention.dense", v, 0)

        # 4.3 mlp.gate
        v = [load(prefix + gptq_key_list[6] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(tllm_prefix + "mlp.gate", v, 1)

        # 4.4 mlp.proj
        v = [load(prefix + gptq_key_list[7] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(tllm_prefix + "mlp.proj", v, 0)

        # 4.5 mlp.fc
        v = [load(prefix + gptq_key_list[8] + suf) for suf in gptq_suffix_list]
        process_and_assign_weight(tllm_prefix + "mlp.fc", v, 1)

        # 4.6 input_layernorm
        v = load(prefix + gptq_key_list[9])
        weights[tllm_prefix + 'input_layernorm.weight'] = v.to(torch_dtype)

        # 4.7 pst_layernorm
        v = load(prefix + gptq_key_list[10])
        weights[tllm_prefix + 'post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
    return weights


if __name__ == '__main__':
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
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

    if args.smoothquant:
        if args.per_token and args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
        elif not args.per_token and not args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        elif not args.per_token and args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        elif args.per_token and not args.per_channel:
            quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN

    if args.int8_kv_cache:
        kv_cache_quant_algo = QuantAlgo.INT8
    else:
        kv_cache_quant_algo = None

    if args.model_version == 'v1_7b' or args.model_version == 'v2_7b':
        position_embedding_type = 'rope_gpt_neox'
    else:
        position_embedding_type = 'alibi'

    hf_config = load_baichuan_config(args.model_dir)
    if args.model_version == 'v1_7b' or args.model_version == 'v2_7b':
        max_position_embeddings = hf_config.max_position_embeddings
    else:
        max_position_embeddings = hf_config.model_max_length

    if args.weight_only_precision == 'int4_gptq':
        hf_config.vocab_size = int((hf_config.vocab_size + 63) / 64) * 64

    world_size = args.tp_size * args.pp_size
    config = {
        'architecture': 'BaichuanForCausalLM',
        'dtype': args.dtype,
        'logits_dtype': args.logits_dtype,
        'vocab_size': hf_config.vocab_size,
        'max_position_embeddings': max_position_embeddings,
        'hidden_size': hf_config.hidden_size,
        'num_hidden_layers': hf_config.num_hidden_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': hf_config.num_attention_heads,
        'hidden_act': hf_config.hidden_act,
        'intermediate_size': hf_config.intermediate_size,
        'norm_epsilon': hf_config.rms_norm_eps,
        'position_embedding_type': position_embedding_type,
        'quantization': {
            'quant_algo': quant_algo,
            'kv_cache_quant_algo': kv_cache_quant_algo,
            'group_size': args.group_size,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
    }
    if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
        config['quantization'].update({
            'has_zero_point': True,
        })

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    hf_model = AutoModelForCausalLM.from_pretrained(args.model_dir,
                                                    trust_remote_code=True,
                                                    torch_dtype="auto")

    if args.smoothquant is not None or args.int8_kv_cache:
        act_range = {}
        baichuan_smoother = {}
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir,
                                                  use_fast=False,
                                                  trust_remote_code=True)
        dataset = load_calib_dataset(args.calib_dataset)
        act_range = capture_activation_range(hf_model.cuda(), tokenizer,
                                             dataset)
        if args.smoothquant is not None:
            smooth_baichuan_model(hf_model, act_range, args.smoothquant,
                                  baichuan_smoother)

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        if args.smoothquant is not None or args.int8_kv_cache:
            weights = convert_hf_baichuan_sq(hf_model, mapping, rank,
                                             args.dtype, args.per_channel,
                                             args.per_token, args.int8_kv_cache,
                                             act_range, baichuan_smoother)
        elif args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            weights = convert_baichuan_gptq(hf_config,
                                            args.quant_ckpt_path,
                                            args.model_version,
                                            mapping,
                                            dtype=args.dtype)
        else:
            weights = convert_hf_baichuan(hf_model,
                                          hf_config,
                                          args.model_version,
                                          mapping,
                                          dtype=args.dtype,
                                          quant_algo=quant_algo)

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
