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
import copy
import functools
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from ...logger import logger
from ...quantization import QuantAlgo, QuantMode
from ..convert_utils import (load_calib_dataset, smooth_gemm,
                             smooth_gemm_fc1_gate, weight_only_quantize_dict)
from .config import BaichuanConfig


def generate_int8(
    weights: torch.Tensor,
    act_range: Dict[str, torch.Tensor],
    is_qkv: bool = False,
    multi_query_mode: bool = False,
):
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
def capture_activation_range(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    num_samples: int = 512,
):
    model.eval()
    device = next(model.parameters()).device
    act_scales: defaultdict[Any, Dict[str,
                                      torch.Tensor]] = defaultdict(lambda: {
                                          "x": None,
                                          "y": None,
                                          "w": None
                                      })

    test_token_num = 923
    tokenizer.pad_token = tokenizer.eos_token

    def stat_tensor(
        name: str,
        tensor: torch.Tensor,
        act_scales: defaultdict[Any, Dict[str, torch.Tensor]],
        key: str,
    ):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(
        m: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        name: str,
    ):
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
                                 max_length=test_token_num,
                                 padding=True,
                                 truncation=True).input_ids.to(device)
        model(line_encoded)

    for h in hooks:
        h.remove()

    return act_scales


@torch.no_grad()
def smooth_baichuan_model(
    model: AutoModelForCausalLM,
    scales: Dict[Any, Dict[str, torch.Tensor]],
    alpha: float,
    baichuan_smoother: Dict[str, torch.Tensor],
):
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


def get_tllm_linear_sq_weight(
    vals: Dict[str, np.ndarray],
    prefix: str,
    shape: List[int],
    tensor_parallel: int,
    quant_mode: QuantMode,
    is_qkv: bool = False,
    last_prefix: Optional[str] = None,
    bias: Optional[torch.Tensor] = None,
    smoother_value=None,
    smoother_shape=None,
    rank: int = 0,
    cat_dim: int = 0,
    multi_query_mode: bool = False,
):
    per_token = quant_mode.has_per_token_dynamic_scaling()
    per_channel = quant_mode.has_per_channel_scaling()
    results: Dict[str, torch.Tensor] = {}

    def multi_query_split(
        data: np.array,
        start: int,
        size: int,
        tp_size: int,
        cur_rank: int,
    ):
        q, k, v = np.split(data, [start, start + size], axis=-1)
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


def split_matrix(weight: torch.Tensor, tp_size: int, rank: int,
                 dim: int) -> torch.Tensor:
    return split(weight, tp_size, rank, dim=dim)


def get_weight(params: Dict[str, torch.Tensor], prefix: str,
               dtype: torch.dtype) -> Optional[torch.Tensor]:
    if f'{prefix}.weight' not in params:
        return None
    return params[f'{prefix}.weight'].to(dtype).detach().cpu()


def quantize(hf_model_dir: str,
             output_dir: str,
             config: BaichuanConfig,
             device: str = 'cuda',
             calib_dataset: str = 'ccdv/cnn_dailymail',
             trust_remote_code: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    config.to_json_file(os.path.join(output_dir, 'config.json'))

    mapping = config.mapping
    assert mapping.rank == 0, "quantize should be called at rank 0 only"

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        device_map='auto' if device != 'cpu' else 'cpu',
        dtype='auto'
        if not config.quantization._use_plugin_sq else torch.float16,
        trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_dir, use_fast=False, trust_remote_code=trust_remote_code)
    dataset = load_calib_dataset(calib_dataset)

    act_range = capture_activation_range(hf_model, tokenizer, dataset)
    smoother = {}
    if config.quantization._use_plugin_sq:
        smooth_baichuan_model(hf_model, act_range,
                              config.quantization.smoothquant_val, smoother)

    quant_mode = config.quantization.quant_mode
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    inter_size = config.intermediate_size
    num_key_value_heads = config.num_key_value_heads
    multi_query_mode = (num_key_value_heads != num_attention_heads)

    for rank in range(config.mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = {}
        tensor_parallel = config.mapping.tp_size

        for l in config.mapping.pp_layers(config.num_hidden_layers):
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
                                         act_range.get(prefix +
                                                       'self_attn.W_pack'),
                                         is_qkv=True,
                                         multi_query_mode=multi_query_mode)
            weights.update(
                get_tllm_linear_sq_weight(int8_weights,
                                          tllm_prex + 'attention.qkv.',
                                          [1, qkv_out_dim // tensor_parallel],
                                          tensor_parallel,
                                          quant_mode,
                                          is_qkv=True,
                                          last_prefix=tllm_prex +
                                          'input_layernorm.scale_to_int',
                                          smoother_value=None,
                                          smoother_shape=None,
                                          rank=rank,
                                          cat_dim=-1,
                                          multi_query_mode=multi_query_mode))

            if config.quantization.kv_cache_quant_algo == QuantAlgo.INT8:
                qkv_weight = get_weight(model_params,
                                        prefix + 'self_attn.W_pack', dtype)
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
            int8_weights = generate_int8(
                attn_dense_weight, act_range.get(prefix + 'self_attn.o_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.dense.', [1, hidden_size],
                    tensor_parallel,
                    quant_mode,
                    is_qkv=False,
                    last_prefix=tllm_prex +
                    'attention.quantization_scaling_factor',
                    smoother_value=smoother[(prefix + 'self_attn.o_proj')],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=rank,
                    cat_dim=0))

            # mlp.gate_proj -> mlp.fc
            mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                       dtype)
            mlp_fc_weight = mlp_fc_weight.t().numpy()
            int8_weights = generate_int8(
                mlp_fc_weight, act_range.get(prefix + 'mlp.gate_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'mlp.fc.', [1, inter_size // tensor_parallel],
                    tensor_parallel,
                    quant_mode,
                    is_qkv=False,
                    last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                    smoother_value=None,
                    smoother_shape=None,
                    rank=rank,
                    cat_dim=-1))

            # mlp.down_proj -> mlp.proj
            mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                         dtype)
            mlp_proj_weight = mlp_proj_weight.t().numpy()
            int8_weights = generate_int8(
                mlp_proj_weight, act_range.get(prefix + 'mlp.down_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'mlp.proj.', [1, hidden_size],
                    tensor_parallel,
                    quant_mode,
                    is_qkv=False,
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
                    quant_mode,
                    is_qkv=False,
                    last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                    smoother_value=None,
                    smoother_shape=None,
                    rank=rank,
                    cat_dim=-1))

            # input layer_norm
            input_ln_weight = get_weight(model_params,
                                         prefix + 'input_layernorm', dtype)
            weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

            # post layer_norm
            post_ln_weight = get_weight(model_params,
                                        prefix + 'post_attention_layernorm',
                                        dtype)
            weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

        embed_w = get_weight(model_params, 'model.embed_tokens', dtype)
        if config.mapping.is_first_pp_rank():
            # Embedding
            if config.use_parallel_embedding:
                embed_w = split_matrix(embed_w,
                                       config.mapping.tp_size,
                                       config.mapping.tp_rank,
                                       dim=config.embedding_sharding_dim)
            weights['transformer.vocab_embedding.weight'] = embed_w

        lm_head_w = get_weight(model_params, 'lm_head', dtype)
        if config.mapping.is_last_pp_rank():
            # lm_head weight and bias
            weights['lm_head.weight'] = split_matrix(lm_head_w.clone(),
                                                     config.mapping.tp_size,
                                                     config.mapping.tp_rank,
                                                     dim=0)
            ln_f_w = get_weight(model_params, 'model.norm', dtype)
            # ln_f weight and bias
            weights['transformer.ln_f.weight'] = ln_f_w

        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights


def load_weights_from_hf_model(hf_model: AutoModelForCausalLM,
                               config: BaichuanConfig):
    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, config.dtype)
    num_hidden_layers = config.num_hidden_layers
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

    def load(key_id: int, layer_idx: int = -1, tp_dim: int = -1):
        layer_prefix = "" if layer_idx == -1 else f"model.layers.{layer_idx}."
        v: torch.Tensor = model_params[layer_prefix + hf_key[key_id]]
        if key_id == 3:
            q_emb = v.shape[0] // 3
            model_emb = v.shape[1]
            v = v.reshape(3, q_emb, model_emb)
            if v.shape[1] % config.mapping.tp_size != 0:
                logger.error(
                    "Current weight shape is invalid for mapping.tp_size=" +
                    str(config.mapping.tp_size))
            v = v.split(v.shape[1] // config.mapping.tp_size,
                        dim=1)[config.mapping.tp_rank]
            v = v.reshape(3 * (q_emb // config.mapping.tp_size), model_emb)
        if tp_dim >= 0:
            if v.shape[tp_dim] % config.mapping.tp_size != 0:
                logger.error(
                    "Current weight shape is invalid for mapping.tp_size=" +
                    str(config.mapping.tp_size))
            v = v.split(v.shape[tp_dim] // config.mapping.tp_size,
                        dim=tp_dim)[config.mapping.tp_rank]
        v = v.to(dtype).contiguous().detach().cpu()
        return v

    # Convert vocab_embedding
    if config.mapping.is_first_pp_rank():
        embed_w = load(0)
        if config.use_parallel_embedding:
            embed_w = split_matrix(embed_w,
                                   config.mapping.tp_size,
                                   config.mapping.tp_rank,
                                   dim=config.embedding_sharding_dim)
        weights['transformer.vocab_embedding.weight'] = embed_w

    # Convert lm_head
    v = load(1, -1, 0)
    if config.model_version.startswith('v2'):
        v = torch.nn.functional.normalize(v)
    if config.mapping.is_last_pp_rank():
        weights['lm_head.weight'] = v

    # Convert ln_f
    if config.mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = load(2)

    # Convert layers
    layers_range = config.mapping.pp_layers(num_hidden_layers)
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
                                     quant_algo=config.quantization.quant_algo,
                                     plugin=True)


def load_weights_from_gptq(config: BaichuanConfig, quant_ckpt_path: str):
    logger.info('Loading weights from groupwise GPTQ Baichuan safetensors...')
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
    torch_dtype = getattr(torch, config.dtype)

    def load(key: str, no_prefix: bool = False):
        if no_prefix:
            return gptq_baichuan.get_tensor(key)
        else:
            return gptq_baichuan.get_tensor(gptq_prefix + key)

    def torch_split(tensor: torch.Tensor, dim: int):
        if tensor.shape[dim] % config.mapping.tp_size != 0:
            logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(config.mapping.tp_size))
            assert False, "Invalid TP size"
        return tensor.split(tensor.shape[dim] // config.mapping.tp_size,
                            dim=dim)[config.mapping.tp_rank]

    def unpack_int32_into_int8(w_packed: torch.Tensor):
        # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def process_and_assign_weight(prefix: str,
                                  tensors: List[torch.Tensor],
                                  tp_dim: int = -1):
        if tp_dim == -1:
            qweight_int32, qzeros_int32, scales_fp16 = [
                item.cpu() for item in tensors
            ]
        else:
            qweight_int32, qzeros_int32, scales_fp16 = [
                torch_split(item, tp_dim).cpu() for item in tensors
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
    if config.mapping.is_first_pp_rank():
        if config.use_parallel_embedding:
            v = split_matrix(v,
                             config.mapping.tp_size,
                             config.mapping.tp_rank,
                             dim=config.embedding_sharding_dim)
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)

    # 2. lm_head
    original_v = load(gptq_key_list[1], True)
    if config.model_version.startswith('v2'):
        # baichuan v2 models use NormHead
        logger.info(f'Normalizing lm_head.weight for {config.model_version}')
        v = torch_split(torch.nn.functional.normalize(original_v), 0)
    else:
        v = torch_split(original_v, 0)
    if config.mapping.is_last_pp_rank():
        weights['lm_head.weight'] = v.to(torch_dtype)

    # 3. ln_f
    v = load(gptq_key_list[2])
    if config.mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)

    # 4. Weights inside each layer
    num_hidden_layers = config.num_hidden_layers
    layers_range = config.mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        layer_idx = l - layers_range[0]
        hf_prefix = f"layers.{l}."
        tllm_prefix = f"transformer.layers.{l}."
        logger.info(f'Process weights in layer: {layer_idx}')

        # 4.1 attention.qkv
        qkv_weight_list = []
        for suf in gptq_suffix_list:
            qkv_list = []
            comp_part = load(hf_prefix + gptq_key_list[3] + suf)
            qkv = torch.chunk(comp_part, 3, 1)
            for i in range(3):
                comp_part = qkv[i]
                comp_part = torch_split(comp_part, 1)
                qkv_list.append(comp_part)
            qkv_weight_list.append(torch.cat(qkv_list, dim=1))

        process_and_assign_weight(tllm_prefix + "attention.qkv",
                                  qkv_weight_list)

        # 4.2 attention.dense
        v = [
            load(hf_prefix + gptq_key_list[5] + suf) for suf in gptq_suffix_list
        ]
        process_and_assign_weight(tllm_prefix + "attention.dense", v, 0)

        # 4.3 mlp.gate
        v = [
            load(hf_prefix + gptq_key_list[6] + suf) for suf in gptq_suffix_list
        ]
        process_and_assign_weight(tllm_prefix + "mlp.gate", v, 1)

        # 4.4 mlp.proj
        v = [
            load(hf_prefix + gptq_key_list[7] + suf) for suf in gptq_suffix_list
        ]
        process_and_assign_weight(tllm_prefix + "mlp.proj", v, 0)

        # 4.5 mlp.fc
        v = [
            load(hf_prefix + gptq_key_list[8] + suf) for suf in gptq_suffix_list
        ]
        process_and_assign_weight(tllm_prefix + "mlp.fc", v, 1)

        # 4.6 input_layernorm
        v = load(hf_prefix + gptq_key_list[9])
        weights[tllm_prefix + 'input_layernorm.weight'] = v.to(torch_dtype)

        # 4.7 pst_layernorm
        v = load(hf_prefix + gptq_key_list[10])
        weights[tllm_prefix + 'post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights
