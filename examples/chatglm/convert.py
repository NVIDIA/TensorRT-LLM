# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
    Utilities for exporting a model to our custom format.
"""
import numpy as np
import torch

from tensorrt_llm.logger import logger


def save_val(val, dir, key, tp_num=None):
    suffix = "bin" if tp_num is None else f"{tp_num}.bin"
    val.tofile(dir / f"{key}.{suffix}")


def save_split(split_vals, dir, key, i, factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * factor + j)


def generate_int8(weights, act_range, is_qkv=False, multi_query_mode=True):
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

    # For ChatGLM2/3-6B models (num_kv_head == 2), we regard multi_query_mode == True to reuse code from gpt and baichuan examples.
    if is_qkv and multi_query_mode:
        hidden_dim, local_dim = weights.shape

        kv_dim = (local_dim - hidden_dim) // 2
        scale_w_q = act_range["w"][0:hidden_dim]
        scale_w_k = act_range["w"][hidden_dim:hidden_dim + kv_dim]
        scale_w_v = act_range["w"][-kv_dim:]

        scale_w_qkv_t = torch.concat([
            scale_w_q.max(dim=0, keepdim=True)[0],
            scale_w_k.max(dim=0, keepdim=True)[0],
            scale_w_v.max(dim=0, keepdim=True)[0]
        ])
        scale_w_orig_quant_t = 127. / scale_w_qkv_t.cpu().numpy().astype(
            np.float32)
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy().astype(
            np.float32)
    elif is_qkv and not multi_query_mode:
        scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
            dim=-1, keepdims=True)[0].cpu().numpy().astype(np.float32)
        scale_w_orig_quant_c = 127. / act_range["w"].reshape(
            3, -1).cpu().numpy().astype(np.float32)

    else:
        scale_w_orig_quant_t = 127. / act_range["w"].max().cpu().numpy().astype(
            np.float32)
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy().astype(
            np.float32)
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127. / act_range["x"].max().item()).astype(
        np.float32)
    scale_y_orig_quant_t = np.array(127. / act_range["y"].max().item()).astype(
        np.float32)
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.).astype(
        np.float32)
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
        scale_w_orig_quant_t_expand = np.ones([weights.shape[-1]])
        scale_w_orig_quant_t_expand[:hidden_dim] = scale_w_orig_quant_t[0]
        scale_w_orig_quant_t_expand[hidden_dim:hidden_dim +
                                    kv_dim] = scale_w_orig_quant_t[1]
        scale_w_orig_quant_t_expand[-kv_dim:] = scale_w_orig_quant_t[2]
        weight_int8 = to_i8(weights * scale_w_orig_quant_t_expand)
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


def save_multi_query_mode_qkv_int8(val, dir, base_key, saved_key, factor, rank,
                                   local_dim, head_size):
    q, k, v = np.split(val, [local_dim, local_dim + head_size], axis=-1)
    q_split = np.split(q, factor, axis=-1)
    k_split = np.split(k, factor, axis=-1)
    v_split = np.split(v, factor, axis=-1)
    split_vals = [
        np.concatenate((q_split[ii], k_split[ii], v_split[ii]), axis=-1)
        for ii in range(factor)
    ]
    save_split(split_vals, dir, f"{base_key}.{saved_key}", rank, factor)


def write_int8(vals,
               dir,
               base_key,
               split_dim,
               i,
               factor,
               is_qkv=False,
               multi_query_mode=False):
    saved_keys_once = [
        "scale_x_orig_quant", "scale_w_quant_orig", "scale_y_accum_quant",
        "scale_y_quant_orig"
    ]

    if is_qkv and multi_query_mode:
        assert split_dim == -1
        local_dim = vals["weight.int8"].shape[0]
        head_size = (vals["weight.int8"].shape[1] - local_dim) // 2

        save_multi_query_mode_qkv_int8(vals["weight.int8"], dir, base_key,
                                       "weight.int8", factor, i, local_dim,
                                       head_size)
        save_multi_query_mode_qkv_int8(vals["weight.int8.col"], dir, base_key,
                                       "weight.int8.col", factor, i, local_dim,
                                       head_size)
        save_multi_query_mode_qkv_int8(vals["scale_w_quant_orig.col"], dir,
                                       base_key, "scale_w_quant_orig.col",
                                       factor, i, local_dim, head_size)
        save_multi_query_mode_qkv_int8(vals["scale_y_accum_quant.col"], dir,
                                       base_key, "scale_y_accum_quant.col",
                                       factor, i, local_dim, head_size)
    else:
        save_split(np.split(vals["weight.int8"], factor, axis=split_dim), dir,
                   f"{base_key}.weight.int8", i, factor)
        save_split(np.split(vals["weight.int8.col"], factor, axis=split_dim),
                   dir, f"{base_key}.weight.int8.col", i, factor)

        if split_dim == -1:
            save_split(
                np.split(vals["scale_w_quant_orig.col"], factor,
                         axis=split_dim),
                dir,
                f"{base_key}.scale_w_quant_orig.col",
                i,
                factor,
            )
            save_split(
                np.split(vals["scale_y_accum_quant.col"],
                         factor,
                         axis=split_dim),
                dir,
                f"{base_key}.scale_y_accum_quant.col",
                i,
                factor,
            )
        else:
            saved_keys_once += [
                "scale_w_quant_orig.col", "scale_y_accum_quant.col"
            ]

    if i == 0:
        for save_key in saved_keys_once:
            save_val(vals[save_key], dir, f"{base_key}.{save_key}")


def str_to_np_dtype(type_str):
    convert_dict = {
        "floatp32": np.float32,
        "float16": np.float16,
    }
    if type_str in convert_dict.keys():
        return convert_dict[type_str]
    raise ValueError(f"{type_str} is an invalid storage type")


def split_and_save_weight(
    i,
    output_dir,
    factor,
    key,
    val,
    act_range,
    config,
):

    logger.debug(f"Convert HF name {key}")

    int8_outputs = config.get("int8_outputs", None)
    multi_query_mode = config.get("multi_query_mode", False)
    save_int8 = int8_outputs in ["all", "kv_cache_only"]
    weights_normalization = [
        "pre_norm.weight",
        "pre_norm.bias",
        "post_norm.weight",
        "post_norm.bias",
        "final_norm.weight",
        "final_norm.bias",
    ]

    if i == 0 and any([k in key for k in weights_normalization]):
        save_val(val, output_dir, key)

    elif "attention.query_key_value.weight" in key:
        w_q, w_k, w_v = np.split(val, [4096, 4096 + 256], axis=-1)  # hard code
        w_q_split = np.split(w_q, factor, axis=-1)
        w_k_split = np.split(w_k, factor, axis=-1)
        w_v_split = np.split(w_v, factor, axis=-1)
        split_vals = [
            np.concatenate([w_q_split[ii], w_k_split[ii], w_v_split[ii]],
                           axis=-1) for ii in range(factor)
        ]
        save_split(split_vals, output_dir, key, i, factor)
        if save_int8:
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(
                val,
                act_range,
                is_qkv=True,
                multi_query_mode=multi_query_mode,
            )
            write_int8(
                vals_i8,
                output_dir,
                base_key,
                -1,
                i,
                factor,
                is_qkv=True,
                multi_query_mode=multi_query_mode,
            )

    elif "attention.query_key_value.bias" in key:
        w_q, w_k, w_v = np.split(val, [4096, 4096 + 256], axis=-1)  # hard code
        w_q_split = np.split(w_q, factor, axis=-1)
        w_k_split = np.split(w_k, factor, axis=-1)
        w_v_split = np.split(w_v, factor, axis=-1)
        split_vals = [
            np.concatenate((w_q_split[ii], w_k_split[ii], w_v_split[ii]),
                           axis=-1) for ii in range(factor)
        ]
        save_split(split_vals, output_dir, key, i, factor)

    elif "attention.dense.weight" in key or \
        "mlp.fc.weight" in key or "mlp.proj.weight" in key:
        split_dim = -1 if "mlp.fc.weight" in key else 0
        split_vals = np.split(val, factor, axis=split_dim)
        save_split(split_vals, output_dir, key, i, factor)
        if act_range is not None and int8_outputs == "all":
            vals_i8 = generate_int8(val, act_range)
            base_key = key.replace(".weight", "")
            write_int8(vals_i8, output_dir, base_key, split_dim, i, factor)

    elif "attention.dense.smoother" in key or "mlp.proj.smoother" in key:
        split_vals = np.split(val, factor, axis=0)
        save_split(split_vals, output_dir, key, i, factor)

    else:
        logger.warning(f"{key} not handled by converter")
