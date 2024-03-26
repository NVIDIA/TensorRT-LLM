import copy
import functools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.pytorch_utils import Conv1D

from ..._utils import pad_vocab_size, release_gc
from ...layers import MoeConfig
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..modeling_utils import PretrainedConfig, QuantConfig
from .weight import load_from_hf_checkpoint

try:
    from transformers import LlavaConfig, LlavaForConditionalGeneration
except ImportError:
    pass

try:
    pass
except ImportError:
    pass


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
    weights = weights.detach().cpu().numpy()

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

    scale_w_orig_quant_c = scale_w_orig_quant_c.astype(np.float32)
    scale_w_orig_quant_t = scale_w_orig_quant_t.astype(np.float32)

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
        weight_int8 = to_i8(weights / scale_w_quant_orig_t)
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
def smooth_llama_model(model, scales, alpha, llama_qkv_para, llama_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not isinstance(
                module,
                LlamaDecoderLayer) and not module.__class__.__name__ in [
                    "InternLMDecoderLayer", "MistralDecoderLayer"
                ]:
            continue
        # qkv_proj
        layer_name_q = name + ".self_attn.q_proj"
        layer_name_k = name + ".self_attn.k_proj"
        layer_name_v = name + ".self_attn.v_proj"
        layer_name_qkv = name + ".self_attn.qkv_proj"

        weight = torch.cat([
            module.self_attn.q_proj.weight, module.self_attn.k_proj.weight,
            module.self_attn.v_proj.weight
        ],
                           dim=0)

        smoother = smooth_gemm(weight, scales[layer_name_q]["x"],
                               module.input_layernorm.weight, None, alpha)

        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat([
            scales[layer_name_q]["y"], scales[layer_name_k]["y"],
            scales[layer_name_v]["y"]
        ],
                                                dim=0)

        # see transpose_weights function
        llama_qkv_para[layer_name_qkv] = weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        llama_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(
            dim=1)[0]

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
            dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.up_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.down_proj"
        smoother = smooth_gemm(module.mlp.down_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        llama_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(
            dim=1)[0]


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             num_samples=512,
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
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        datapoint = dataset['train'][i:i + 1]
        line = copy.copy(datapoint['article'])
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


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].clone()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.clone()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = v.reshape(3, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.clone()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def get_weight(config, prefix, dtype):
    if config[prefix + '.weight'].dtype != dtype:
        config[prefix + '.weight'].data = config[prefix + '.weight'].to(dtype)
    return config[prefix + '.weight'].detach().cpu()


def get_bias(config, prefix, dtype):
    if config[prefix + '.bias'].dtype != dtype:
        config[prefix + '.bias'].data = config[prefix + '.bias'].to(dtype)
    return config[prefix + '.bias'].detach().cpu()


def get_weight_and_bias(config, prefix, dtype):
    return get_weight(config, prefix, dtype), get_bias(config, prefix, dtype)


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8,
                           dtype='float32',
                           use_gemm_woq_plugin=True,
                           postfix='weight',
                           quant_scale_name=None):
    results = {}
    if use_weight_only:
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous().clone()
        else:
            v = weight.t().contiguous().clone()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        if not use_gemm_woq_plugin:
            results[prefix + postfix] = v.to(dtype)
        else:
            results[prefix + postfix] = processed_torch_weights
        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.clone()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def dup_kv_weight(v, num_head, tp_size):
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    head_size = v.shape[0] // num_head
    v = v.reshape(num_head, head_size,
                  -1)[:, None, :, :].expand(num_head, reps, head_size,
                                            v.shape[1])
    return v.reshape(num_head * reps * head_size, -1).clone().detach()


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
        results[prefix + 'weight'] = torch.from_numpy(
            cur_weights).t().clone().contiguous()
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
        results[prefix + 'weight'] = torch.from_numpy(
            cur_weights).t().clone().contiguous()

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


def convert_hf_llama(hf_model,
                     mapping,
                     vocab_size=32000,
                     dtype='float32',
                     use_parallel_embedding=False,
                     sharding_dim=0,
                     use_weight_only=False,
                     share_embedding_table=False,
                     use_gemm_woq_plugin=False,
                     plugin_weight_only_quant_type=torch.int8,
                     use_smooth_quant=False,
                     per_channel=False,
                     per_token=False,
                     int8_kv_cache=False,
                     act_range=[],
                     qkv_para=[],
                     smoother=[],
                     moe_config=None):

    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    head_size = hidden_size // num_attention_heads
    intermediate_size = hf_model.config.intermediate_size
    num_key_value_heads = getattr(hf_model.config, 'num_key_value_heads',
                                  num_attention_heads)
    mha_mode = (num_key_value_heads == num_attention_heads)
    layers_range = mapping.pp_layers(hf_model.config.num_hidden_layers)

    def convert_layer(l):
        prefix = f'model.layers.{l}.'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        q_weight = get_weight(model_params, prefix + 'self_attn.q_proj', dtype)
        k_weight = get_weight(model_params, prefix + 'self_attn.k_proj', dtype)
        v_weight = get_weight(model_params, prefix + 'self_attn.v_proj', dtype)

        if not mha_mode:
            if num_key_value_heads < tensor_parallel:
                # duplicate the KV heads up to tensor_parallel
                k_weight = dup_kv_weight(k_weight, num_key_value_heads,
                                         tensor_parallel)
                v_weight = dup_kv_weight(v_weight, num_key_value_heads,
                                         tensor_parallel)
            assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
            assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0

            wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
            wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
            wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

            split_v = torch.concat((wq, wk, wv))

        else:
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

            split_v = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                                   tensor_parallel, mapping.tp_rank)

        if prefix + 'self_attn.q_proj.bias' in model_params:
            # only used in Internlm 7B models
            q_bias = get_bias(model_params, prefix + 'self_attn.q_proj', dtype)
            k_bias = get_bias(model_params, prefix + 'self_attn.k_proj', dtype)
            v_bias = get_bias(model_params, prefix + 'self_attn.v_proj', dtype)
            qkv_bias = torch.cat((q_bias, k_bias, v_bias))
            split_bias_v = split_qkv_bias_tp(qkv_bias, num_attention_heads,
                                             hidden_size, tensor_parallel,
                                             mapping.tp_rank)
        else:
            split_bias_v = None

        if use_smooth_quant:
            qkv_weight = qkv_para[prefix + 'self_attn.qkv_proj']
            qkv_out_dim = qkv_weight.shape[1]

            if not mha_mode:
                local_dim = qkv_weight.shape[0]
                kv_hidden_size = (qkv_weight.shape[-1] - local_dim) // 2
                qkv_weight = qkv_weight.reshape(local_dim,
                                                local_dim + 2 * kv_hidden_size)
            else:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)

            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix +
                                                       'self_attn.qkv_proj'),
                                         is_qkv=True,
                                         multi_query_mode=bool(not mha_mode))

            weights.update(
                get_tllm_linear_sq_weight(int8_weights,
                                          tllm_prex + 'attention.qkv.',
                                          [1, qkv_out_dim // tensor_parallel],
                                          tensor_parallel,
                                          is_qkv=True,
                                          bias=split_bias_v,
                                          per_token=per_token,
                                          per_channel=per_channel,
                                          last_prefix=tllm_prex +
                                          'input_layernorm.scale_to_int',
                                          smoother_value=None,
                                          smoother_shape=None,
                                          rank=mapping.tp_rank,
                                          cat_dim=-1,
                                          multi_query_mode=bool(not mha_mode)))
        else:
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'attention.qkv.',
                                       split_bias_v, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        if int8_kv_cache:
            qkv_y = torch.cat([
                act_range.get(prefix + 'self_attn.q_proj')["y"],
                act_range.get(prefix + 'self_attn.k_proj')["y"],
                act_range.get(prefix + 'self_attn.v_proj')["y"]
            ],
                              dim=0)

            int8_kv_scales = qkv_y.max() / 127.

            kv_cache_weights = {}

            kv_cache_weights[
                tllm_prex +
                'attention.kv_cache_scaling_factor'] = int8_kv_scales.reshape(
                    [1])

            weights.update(kv_cache_weights)

        attn_dense_weight = get_weight(model_params,
                                       prefix + 'self_attn.o_proj', dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)

        if prefix + 'self_attn.o_proj.bias' in model_params:
            attn_dense_bias = get_bias(model_params,
                                       prefix + 'self_attn.o_proj', dtype)
        else:
            attn_dense_bias = None
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(
                attn_dense_weight, act_range.get(prefix + 'self_attn.o_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.dense.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    bias=attn_dense_bias,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex +
                    'attention.quantization_scaling_factor',
                    smoother_value=smoother[(prefix + 'self_attn.o_proj')],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'attention.dense.',
                                       attn_dense_bias, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        if moe_config and moe_config.has_moe():

            rank_experts = list(range(moe_config.num_experts))
            if moe_config.tp_mode == moe_config.ParallelismMode.EXPERT_PARALLEL:
                rank_experts = mapping.ep_experts(moe_config.num_experts)
            for suffix in ["w1", "w2", "w3"]:
                model_params[f'model.layers.{l}.block_sparse_moe.experts.{suffix}.weight'] = \
                            torch.stack([model_params[f'model.layers.{l}.block_sparse_moe.experts.{expert}.{suffix}.weight'].detach()
                                        for expert in rank_experts])
            w3 = model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w3.weight']
            w2 = model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w2.weight']
            w1 = model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w1.weight']
            if moe_config.tp_mode == moe_config.ParallelismMode.TENSOR_PARALLEL:
                w3 = split(w3, mapping.tp_size, mapping.tp_rank, dim=1)
                w2 = split(w2, mapping.tp_size, mapping.tp_rank, dim=2)
                w1 = split(w1, mapping.tp_size, mapping.tp_rank, dim=1)

            model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w3w1.weight'] = torch.concat(
                    [w3, w1], dim=-2)

            model_params[
                f'model.layers.{l}.block_sparse_moe.experts.w2.weight'] = w2

            ## block_sparse_moe.experts.w2.weight
            moe_experts_w2_weights = get_weight(
                model_params, prefix + 'block_sparse_moe.experts.w2', dtype)
            weights.update(
                get_tllm_linear_weight(moe_experts_w2_weights,
                                       tllm_prex + 'mlp.experts_weight_2',
                                       None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type,
                                       dtype,
                                       use_gemm_woq_plugin,
                                       postfix='',
                                       quant_scale_name=tllm_prex +
                                       'mlp.experts_scale_2'))
            ##block_sparse_moe.experts.w3w1.weight
            moe_experts_w3w1_weights = get_weight(
                model_params, prefix + 'block_sparse_moe.experts.w3w1', dtype)
            weights.update(
                get_tllm_linear_weight(moe_experts_w3w1_weights,
                                       tllm_prex + 'mlp.experts_weight_1',
                                       None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type,
                                       dtype,
                                       use_gemm_woq_plugin,
                                       postfix='',
                                       quant_scale_name=tllm_prex +
                                       'mlp.experts_scale_1'))

            moe_experts_gate_weights = get_weight(
                model_params, prefix + 'block_sparse_moe.gate', torch.float32)
            weights.update(
                get_tllm_linear_weight(
                    moe_experts_gate_weights.to(torch.float32),
                    tllm_prex + 'mlp.router.',
                    None,
                    False,  # Router should never be quantized
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin))
        else:
            mlp_gate_weight = get_weight(model_params, prefix + 'mlp.up_proj',
                                         dtype)
            split_v = split_matrix_tp(mlp_gate_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=0)
            if use_smooth_quant:
                mlp_gate_weight = mlp_gate_weight.t()
                int8_weights = generate_int8(
                    mlp_gate_weight, act_range.get(prefix + 'mlp.up_proj'))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.gate.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.gate.',
                                           None, use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

            mlp_fc_weight = get_weight(model_params, prefix + 'mlp.gate_proj',
                                       dtype)
            split_v = split_matrix_tp(mlp_fc_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=0)

            if use_smooth_quant:
                mlp_fc_weight = mlp_fc_weight.t()  #verified
                int8_weights = generate_int8(
                    mlp_fc_weight, act_range.get(prefix + 'mlp.gate_proj'))
                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.fc.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.fc.', None,
                                           use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

            mlp_proj_weight = get_weight(model_params, prefix + 'mlp.down_proj',
                                         dtype)
            split_v = split_matrix_tp(mlp_proj_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=1)

            if use_smooth_quant:
                mlp_proj_weight = mlp_proj_weight.t()
                int8_weights = generate_int8(
                    mlp_proj_weight, act_range.get(prefix + 'mlp.down_proj'))
                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.proj.', [1, hidden_size],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex +
                        'mlp.quantization_scaling_factor',
                        smoother_value=smoother[prefix + 'mlp.down_proj'],
                        smoother_shape=[
                            1, intermediate_size // tensor_parallel
                        ],
                        rank=mapping.tp_rank,
                        cat_dim=0))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.',
                                           None, use_weight_only,
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

    for l in layers_range:
        convert_layer(l)
        release_gc()

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

    # if not use_parallel_embedding:
    #     weights['transformer.vocab_embedding.weight'] = embed_w
    # else:
    #     assert hf_model.config.vocab_size % tensor_parallel == 0
    #     weights['transformer.vocab_embedding.weight'] = split_matrix_tp(
    #         embed_w, tensor_parallel, rank

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
    print(f'Weights loaded. Total time: {t}')
    return weights


def smooth_quant(model,
                 model_dir,
                 dataset_cache_dir,
                 smoothquant: Optional[float] = None):
    assert model is not None
    act_range = {}
    llama_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    llama_smoother = {}

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    dataset = load_dataset("ccdv/cnn_dailymail",
                           '3.0.0',
                           cache_dir=dataset_cache_dir)

    act_range = capture_activation_range(
        model,
        AutoTokenizer.from_pretrained(model_dir,
                                      trust_remote_code=True,
                                      use_fast=False,
                                      padding_side='left'), dataset)
    if smoothquant is not None:
        smooth_llama_model(model, act_range, smoothquant, llama_qkv_para,
                           llama_smoother)
    return act_range, llama_qkv_para, llama_smoother


def create_config_from_hugging_face(hf_model,
                                    dtype,
                                    mapping,
                                    quantization: QuantConfig = None,
                                    override_fields: dict = {}):
    config = {}
    hf_config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
    if hf_config.model_type == "llava":
        # LLaVA = Vision model + Llama LLM
        # We load a llava config and use its' text config as llama config
        hf_config = LlavaConfig.from_pretrained(hf_model).text_config
    # TODO: directly assign the hf_config fields to the config dict w/o creating these local vars
    # same for from_meta and from_cli_args
    n_head = hf_config.num_attention_heads
    inter_size = hf_config.intermediate_size
    n_layer = hf_config.num_hidden_layers
    n_embd = hf_config.hidden_size
    n_kv_head = getattr(hf_config, "num_key_value_heads", n_head)
    rms_norm_eps = hf_config.rms_norm_eps
    vocab_size = hf_config.vocab_size
    n_positions = hf_config.max_position_embeddings
    hidden_act = hf_config.hidden_act
    config['rotary_scaling'] = getattr(hf_config, "rope_scaling", None)
    rotary_base = getattr(hf_config, "rope_theta", 10000.0)
    if hf_config.model_type == "mixtral":
        # HF LLaMA-type models are implicitly using gated activation.
        # With our MoE implementation, we must make it explicit
        hidden_act = "swiglu"
        config[
            'moe_normalization_mode'] = MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE
    else:
        config['moe_normalization_mode'] = None
    moe_num_experts = getattr(hf_config, "num_local_experts", 0)
    moe_top_k = getattr(hf_config, "num_experts_per_tok", 0)
    moe_tp_mode = MoeConfig.ParallelismMode.TENSOR_PARALLEL
    architecture = hf_config.architectures[0]
    # VILA model, force to use llama config
    if hf_config.model_type == "llava_llama":
        architecture = "LlamaForCausalLM"
    attn_bias = getattr(hf_config, 'bias', False) or getattr(
        hf_config, 'attention_bias', False)

    config.update({
        'architecture': architecture,
        'dtype': dtype,
        'logits_dtype': 'float32',
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
        'moe_num_experts': moe_num_experts,
        'moe_top_k': moe_top_k,
        'moe_tp_mode': moe_tp_mode,
        #TODO: should have directly map from the Mapping object to the TRT-LLM checkpoint fields
        'mapping': {
            'world_size': mapping.tp_size * mapping.pp_size,
            'tp_size': mapping.tp_size,
            'pp_size': mapping.pp_size
        },
        'attn_bias': attn_bias,
    })
    config['quantization'] = quantization.asdict()
    config.update(override_fields)

    moe_config = MoeConfig(config['moe_num_experts'], config['moe_top_k'],
                           config['moe_tp_mode'],
                           config['moe_normalization_mode']).validate()
    use_weight_only = config['quantization']['quant_algo'] in [
        QuantAlgo.W8A16, QuantAlgo.W4A16
    ]
    if use_weight_only and moe_config.has_moe():
        config['quantization']['exclude_modules'].append('router')

    return config


def from_hugging_face(cls,
                      model_dir,
                      dtype,
                      *,
                      mapping,
                      quantization: QuantConfig = None,
                      load_by_shard=False,
                      load_model_on_cpu=False,
                      override_fields={},
                      skip_loading_weights=False,
                      preloaded_model=None):
    ''' Create a LLaMAForCausalLM object from give parameters
    '''
    assert model_dir is not None
    if isinstance(model_dir, Path):  # some code relies on this as string
        model_dir = str(model_dir)

    # register VILA model
    if "vila" in model_dir:
        sys.path.append(model_dir + "/../VILA")
        from llava.model import LlavaConfig, LlavaLlamaForCausalLM
        AutoConfig.register("llava_llama", LlavaConfig)
        AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)

    config = create_config_from_hugging_face(model_dir,
                                             dtype,
                                             mapping,
                                             quantization,
                                             override_fields=override_fields)

    pretrained_config = PretrainedConfig.from_dict(config)
    pretrained_config.set_rank(mapping.rank)  # TODO:remove this hack

    llama = cls.from_config(pretrained_config)
    if skip_loading_weights:
        return llama

    model = preloaded_model
    if model is None and not load_by_shard:  # when load by shard, no need to create complete hf model
        hf_config = AutoConfig.from_pretrained(model_dir,
                                               trust_remote_code=True)
        if hf_config.model_type == "llava":
            hf_llava = LlavaForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype="auto")
            model = hf_llava.language_model
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map='auto' if not load_model_on_cpu else 'cpu',
                torch_dtype='auto',
                trust_remote_code=True,
            )

    if load_by_shard:
        weights = load_from_hf_checkpoint(model_dir, mapping, pretrained_config)
    else:
        weights = load_weights_from_hf(config=config,
                                       mapping=mapping,
                                       model=model)

    llama.load(weights)
    return llama


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
    #TODO: currently only smooth quant and kv cache quantization are supported, needs to support mode quant algorithm calling ammo
    config = create_config_from_hugging_face(model_dir,
                                             dtype,
                                             mapping,
                                             quantization,
                                             override_fields=override_fields)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    assert mapping.rank == -1, "You shall call quantize only once in one rank, assert rank==-1 for precaution"
    act_range = {}
    llama_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    llama_smoother = {}
    model = None
    assert config['quantization']['quant_algo'] == quantization.quant_algo
    int8_kv_cache = quantization.kv_cache_quant_algo == QuantAlgo.INT8
    use_smooth_quant = quantization.quant_algo is not None and quantization.quant_algo.startswith(
        'W8A8_SQ')

    assert use_smooth_quant or int8_kv_cache, "Call from_hugging_face when there is no quantization"
    if use_smooth_quant:
        assert quantization.smoothquant_val is not None, "A smooth value must be specified when using smooth quant"

    assert model_dir is not None
    ## only load and call smooth quant routine once for all ranks
    hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    assert "llava" not in hf_config.model_type, "Smooth quant llava/vila is not supported yet"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map='auto',
        torch_dtype='auto' if not use_smooth_quant else torch.float16,
        trust_remote_code=True)
    act_range, llama_qkv_para, llama_smoother = smooth_quant(
        model, model_dir, dataset_cache_dir, quantization.smoothquant_val)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        ranked_mapping = Mapping(world_size=mapping.world_size,
                                 rank=rank,
                                 tp_size=mapping.tp_size,
                                 pp_size=mapping.pp_size)
        weights = load_weights_from_hf(
            config=config,
            mapping=ranked_mapping,
            model=model,
            # for smooth quant only
            act_range=act_range,
            llama_qkv_para=llama_qkv_para,
            llama_smoother=llama_smoother,
        )
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights
        release_gc()


def load_weights_from_hf(*,
                         config,
                         mapping,
                         model,
                         act_range={},
                         llama_qkv_para={},
                         llama_smoother={}):
    #TODO: simplify the parameters here

    assert model is not None
    plugin_weight_only_quant_type = None  # the value does not matter when use_weight_only is False
    quant_algo = config['quantization']['quant_algo']
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2

    moe_config = MoeConfig(config['moe_num_experts'], config['moe_top_k'],
                           config['moe_tp_mode'],
                           config['moe_normalization_mode']).validate()

    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    use_smooth_quant = quant_algo is not None and quant_algo.startswith(
        'W8A8_SQ')
    per_channel_sq = use_smooth_quant and 'PER_CHANNEL' in quant_algo
    per_token_sq = use_smooth_quant and 'PER_TOKEN' in quant_algo
    use_int8_kv_cache = config['quantization'][
        'kv_cache_quant_algo'] == QuantAlgo.INT8
    weights = convert_hf_llama(
        model,
        mapping,
        vocab_size=config['vocab_size'],
        dtype=config['dtype'],
        use_weight_only=use_weight_only,
        use_gemm_woq_plugin=not config.get('disable_weight_only_quant_plugin',
                                           False),
        plugin_weight_only_quant_type=plugin_weight_only_quant_type,
        use_parallel_embedding=config.get('use_parallel_embedding', False),
        sharding_dim=config.get('embedding_sharding_dim', 0),
        share_embedding_table=config.get('share_embedding_table', False),
        use_smooth_quant=use_smooth_quant,
        per_channel=per_channel_sq,
        per_token=per_token_sq,
        int8_kv_cache=use_int8_kv_cache,
        act_range=act_range,
        qkv_para=llama_qkv_para,
        smoother=llama_smoother,
        moe_config=moe_config)
    return weights
