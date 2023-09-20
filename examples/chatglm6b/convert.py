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


def save_val(val, dir, key, tp_num=None):
    suffix = "bin" if tp_num is None else f"{tp_num}.bin"
    val.tofile(dir / f"model.{key}.{suffix}")


def save_split(split_vals, dir, key, i, factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * factor + j)


def generate_int8(weights, act_range, is_qkv=False):
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

      Note that we don't do anything special about row-parallel GEMM. Theorically, we could have per-GPU scaling factors too,
      but then the model would change depending on the number of GPUs used.

      For QKV projection, the behavior is special. Even if we have a single matrix to perform QKV projection, we consider it
      as three different matrices: Q, K, and V. So per-tensor actually means one scaling factor for each Q, K and V.
    """

    # compute weight scaling factors for fp->int8 and int8->fp
    if is_qkv:
        scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
            dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].reshape(3,
                                                             -1).cpu().numpy()
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
    if is_qkv:
        scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t,
                                                scale_w_orig_quant_c.shape)
        scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t,
                                               scale_w_orig_quant_c.shape)

    to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)
    return {
        "weight.int8": to_i8(weights * scale_w_orig_quant_t),
        "weight.int8.col": to_i8(weights * scale_w_orig_quant_c),
        "scale_x_orig_quant": scale_x_orig_quant_t.astype(np.float32),
        "scale_w_quant_orig": scale_w_quant_orig_t.astype(np.float32),
        "scale_w_quant_orig.col": scale_w_quant_orig_c.astype(np.float32),
        "scale_y_accum_quant": scale_y_accum_quant_t.astype(np.float32),
        "scale_y_accum_quant.col": scale_y_accum_quant_c.astype(np.float32),
        "scale_y_quant_orig": scale_y_quant_orig_t.astype(np.float32),
    }


def write_int8(vals, dir, base_key, split_dim, i, factor):
    save_split(np.split(vals["weight.int8"], factor, axis=split_dim), dir,
               f"{base_key}.weight.int8", i, factor)
    save_split(np.split(vals["weight.int8.col"], factor, axis=split_dim), dir,
               f"{base_key}.weight.int8.col", i, factor)

    saved_keys_once = [
        "scale_x_orig_quant", "scale_w_quant_orig", "scale_y_accum_quant",
        "scale_y_quant_orig"
    ]
    # per-column scaling factors are loaded per-gpu for ColumnParallel GEMMs (QKV, FC1)
    if split_dim == -1:
        save_split(
            np.split(vals["scale_w_quant_orig.col"], factor, axis=split_dim),
            dir, f"{base_key}.scale_w_quant_orig.col", i, factor)
        save_split(
            np.split(vals["scale_y_accum_quant.col"], factor, axis=split_dim),
            dir, f"{base_key}.scale_y_accum_quant.col", i, factor)
    else:
        saved_keys_once += ["scale_w_quant_orig.col", "scale_y_accum_quant.col"]

    if i == 0:
        for save_key in saved_keys_once:
            save_val(vals[save_key], dir, f"{base_key}.{save_key}")


def str_to_np_dtype(type_str):
    convert_dict = {
        "fp32": np.float32,
        "fp16": np.float16,
    }
    dtype = convert_dict.get(type_str)
    if dtype is None:
        raise ValueError(f"{type_str} is an invalid storage type")
    return dtype


def split_and_save_weight(i, saved_dir, factor, key, args, val, act_range):
    save_int8 = act_range is not None

    if "input_layernorm.weight" in key or "input_layernorm.bias" in key or \
        "attention.dense.bias" in key or "post_attention_layernorm.weight" in key or \
        "post_attention_layernorm.bias" in key or "mlp.dense_4h_to_h.bias" in key or \
        "final_layernorm.weight" in key or "final_layernorm.bias" in key:

        # shared weights, only need to convert the weights of rank 0
        if i == 0:
            save_val(val, saved_dir, key)

    elif "attention.dense.weight" in key or "mlp.dense_4h_to_h.weight" in key:
        split_dim = 0
        split_vals = np.split(val, factor, axis=split_dim)
        save_split(split_vals, saved_dir, key, i, factor)
        if save_int8:
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range)
            write_int8(vals_i8, saved_dir, base_key, split_dim, i, factor)

    elif "mlp.dense_h_to_4h.weight" in key:
        split_dim = -1
        split_vals = np.split(val, factor, axis=split_dim)
        save_split(split_vals, saved_dir, key, i, factor)
        if save_int8:
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range)
            write_int8(vals_i8, saved_dir, base_key, split_dim, i, factor)

    elif "mlp.dense_h_to_4h.bias" in key:
        split_vals = np.split(val, factor, axis=-1)
        save_split(split_vals, saved_dir, key, i, factor)

    elif "attention.query_key_value.bias" in key:
        local_dim = val.shape[-1] // 3

        val = val.reshape(3, local_dim)
        split_vals = np.split(val, factor, axis=-1)
        save_split(split_vals, saved_dir, key, i, factor)

    elif "attention.query_key_value.weight" in key:
        hidden_dim = val.shape[0] // 3
        local_dim = val.shape[-1]

        val = val.reshape(3, hidden_dim, local_dim)
        split_dim = -1
        split_vals = np.split(val, factor, axis=split_dim)
        save_split(split_vals, saved_dir, key, i, factor)
        if save_int8:
            base_key = key.replace(".weight", "")
            vals_i8 = generate_int8(val, act_range, is_qkv=True)
            write_int8(vals_i8, saved_dir, base_key, split_dim, i, factor)

    else:
        print(f"[WARNING] {key} not handled by converter")
