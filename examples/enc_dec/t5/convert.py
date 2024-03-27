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
import configparser
import logging
import multiprocessing
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

dir_path = os.path.dirname(os.path.realpath(__file__))

from datasets import load_dataset
import numpy as np
import torch  # pytype: disable=import-error
from transformers import T5ForConditionalGeneration
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import (
    T5Block, T5LayerCrossAttention, T5LayerSelfAttention, T5LayerFF
)

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from smoothquant import capture_activation_range, smooth_gemm, smooth_gemm_fc1_gate

LOGGER = logging.getLogger(__name__)

extra_configs = {
    "structure": {
        "t5_with_bias": "false",
        "use_gated_activation": "false",
        "position_embedding_type": "relative",
        'model_type': 't5'
    }
}

def save_val(val, dir, key, tp_num=None):
    suffix = "bin" if tp_num is None else f"{tp_num}.bin"
    val.tofile(dir / f"{key}.{suffix}")

def save_split(split_vals, dir, key, i, factor):
    for j, val in enumerate(split_vals):
        save_val(val, dir, key, i * factor + j)

def fuse_qkv(model, factor, saved_dir, act_range: dict, int8_outputs: Optional[str]):

    def get_attn_module(component, block, layer, attn_type):
        m = getattr(model, component)
        m = m.block[int(block)].layer[int(layer)]
        m = getattr(m, attn_type)
        return m
    
    save_int8 = int8_outputs == "all"

    for name, param in model.named_parameters():
        if 'Attention.q' in name:
            q = param
            component, _, block_idx, _, layer_idx, attn_type, *_ = name.split(
                '.')
            attn_mdl = get_attn_module(component, block_idx, layer_idx,
                                       attn_type)
            shape = q.shape  # (d_out, d_in)
            qkv = torch.cat([q, attn_mdl.k.weight, attn_mdl.v.weight],
                            dim=0).reshape([3, shape[0],
                                            shape[1]])  # (3, d_out, d_in)
            qkv = torch_to_numpy(qkv)
            # embed_dim --> hidden_dim qkv projection weights, [3, hidden_dim, embed_dim] split dim=1
            # ColumnLinear projection weights W=[3*d_out/TP, d_in], split dim=0 or dim=1 in [3, d_out/TP, d_in]
            split_dim = 1
            split_vals = np.split(qkv, factor, axis=split_dim)
            key = f"{component}.block.{block_idx}.layer.{layer_idx}.{attn_type}.qkv.weight"
            component = "encoder" if name.startswith("encoder") else "decoder"
            save_split(split_vals, saved_dir, key, 0, factor)

            scales = act_range.get(name.replace(".weight", "").replace("Attention.q", "Attention.qkv"))
            if save_int8 and scales:
                vals_i8 = generate_int8(qkv,
                                        scales,
                                        key,
                                        is_qkv=True)
                write_int8(vals_i8,
                        saved_dir,
                        key,
                        split_dim,
                        0,
                        factor,
                        is_qkv=True)

def generate_int8(weights, act_range, key: str, is_qkv=False):
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
    if is_qkv:
        scale_w_orig_quant_t = 127. / act_range["w"].reshape(3, -1).max(
            dim=-1, keepdims=True)[0].cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].reshape(3,
                                                             -1).cpu().numpy()
    else:
        scale_w_orig_quant_t = 127. / act_range["w"].max().cpu().numpy()
        scale_w_orig_quant_c = 127. / act_range["w"].cpu().numpy()
    
    scale_w_orig_quant_t = np.expand_dims(scale_w_orig_quant_t, -1)
    scale_w_orig_quant_c = np.expand_dims(scale_w_orig_quant_c, -1)
    scale_w_quant_orig_t = 1.0 / scale_w_orig_quant_t
    scale_w_quant_orig_c = 1.0 / scale_w_orig_quant_c

    # compute the rest of needed scaling factors
    scale_x_orig_quant_t = np.array(127. / act_range["x"].max().item())
    scale_y_orig_quant_t = np.array(127. / act_range["y"].max().item())
    scale_y_quant_orig_t = np.array(act_range["y"].max().item() / 127.)
    try:
        scale_y_accum_quant_t = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                        scale_w_orig_quant_t)
        scale_y_accum_quant_c = scale_y_orig_quant_t / (scale_x_orig_quant_t *
                                                        scale_w_orig_quant_c)
        if is_qkv:
            scale_y_accum_quant_t = np.broadcast_to(scale_y_accum_quant_t,
                                                    scale_w_orig_quant_c.shape)
            scale_w_quant_orig_t = np.broadcast_to(scale_w_quant_orig_t,
                                               scale_w_orig_quant_c.shape)
    except:
        print(f"---------------{key}-----------------------")
        print(f"scale_w_orig_quant_t = {scale_w_orig_quant_t}")
        print(f"scale_w_orig_quant_c = {scale_w_orig_quant_c}")
        print(f"scale_x_orig_quant_t = {scale_x_orig_quant_t}")
        print(f"scale_y_orig_quant_t = {scale_y_orig_quant_t}")

    to_i8 = lambda x: x.round().clip(-127, 127).astype(np.int8)

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

def write_int8(vals,
               dir,
               key,
               split_dim,
               i,
               factor,
               is_qkv=False):

    base_key = key.replace(".weight", "")
    saved_keys_once = [
        "scale_x_orig_quant", "scale_w_quant_orig", "scale_y_accum_quant",
        "scale_y_quant_orig"
    ]

    save_split(np.split(vals["weight.int8"], factor, axis=split_dim), dir,
                f"{base_key}.weight.int8", i, factor)
    save_split(np.split(vals["weight.int8.col"], factor, axis=split_dim),
                dir, f"{base_key}.weight.int8.col", i, factor)

    if split_dim != -1:
        save_split(
            np.split(vals["scale_w_quant_orig.col"], factor,
                        axis=split_dim), dir,
            f"{base_key}.scale_w_quant_orig.col", i, factor)
        save_split(
            np.split(vals["scale_y_accum_quant.col"],
                        factor,
                        axis=split_dim), dir,
            f"{base_key}.scale_y_accum_quant.col", i, factor)
        if is_qkv:
            save_split(
                np.split(vals["scale_y_accum_quant"],
                            factor,
                            axis=split_dim), dir,
                f"{base_key}.scale_y_accum_quant", i, factor)
            save_split(
                np.split(vals["scale_w_quant_orig"], factor,
                            axis=split_dim), dir,
                f"{base_key}.scale_w_quant_orig", i, factor)
            saved_keys_once = ["scale_x_orig_quant", "scale_y_quant_orig"]
    else:
        saved_keys_once += [
            "scale_w_quant_orig.col", "scale_y_accum_quant.col"
        ]

    if i == 0:
        for save_key in saved_keys_once:
            save_val(vals[save_key], dir, f"{base_key}.{save_key}")

@torch.no_grad()
def smooth_t5_model(
    model: T5ForConditionalGeneration,
    scales: defaultdict,
    alpha: float,
    t5_smoother: dict,
) -> None:
    # Smooth the activation and weights with smoother = $\diag{s}$

    def set_attention_scales_and_smoother(name, submodules, attention_type: Literal["self_attn", "cross_attn"]):
        layer_idx, submodule = submodules[attention_type]
        attn_field_name = "SelfAttention" if attention_type == "self_attn" else "EncDecAttention"
        attention_layer = getattr(submodule, attn_field_name)
        attn_qkv_weight = torch.cat([
            attention_layer.q.weight,
            attention_layer.k.weight,
            attention_layer.v.weight, 
        ], dim=0)
        
        # QKV
        layer_name_q = name + f".layer.{layer_idx}.{attn_field_name}.q"
        layer_name_k = name + f".layer.{layer_idx}.{attn_field_name}.k"
        layer_name_v = name + f".layer.{layer_idx}.{attn_field_name}.v"
        layer_name_qkv = name + f".layer.{layer_idx}.{attn_field_name}.qkv"

        smoother = smooth_gemm(attn_qkv_weight, scales[layer_name_q]["x"],
                               submodule.layer_norm.weight, None, alpha)

        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = attn_qkv_weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat([
            scales[layer_name_q]["y"], scales[layer_name_k]["y"],
            scales[layer_name_v]["y"]
        ], dim=0)

        # Attention O
        layer_name = name + f".layer.{layer_idx}.{attn_field_name}.o"
        smoother = smooth_gemm(attention_layer.o.weight,
                               scales[layer_name]["x"], None, None, alpha)
        t5_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = attention_layer.o.weight.abs().max(
            dim=1)[0]

    
    for name, module in model.named_modules():
        # TODO: remove this logic when we add SmoothQuantBertAttention
        # For now, only quantize the weights for the decoder.
        if name.startswith("encoder"):
            continue
        if not isinstance(module, T5Block):
            continue
        
        submodules = {}
        for i, submodule in enumerate(module.layer):
            # Each T5 Block has SelfAttention, LayerFF, and maybe CrossAttention (decoder only).
            if isinstance(submodule, T5LayerSelfAttention):
                submodules["self_attn"] = (i, submodule)
            elif isinstance(submodule, T5LayerCrossAttention):
                submodules["cross_attn"] = (i, submodule)
            elif isinstance(submodule, T5LayerFF):
                submodules["dense_act"] = (i, submodule)

        # ======================= Scale and Smooth Attention =============================
        set_attention_scales_and_smoother(name, submodules, "self_attn")
        if "cross_attn" in submodules:
            set_attention_scales_and_smoother(name, submodules, "cross_attn")

        # ======================= Scale and Smooth MLP =============================
        layer_idx, submodule = submodules["dense_act"]
        fc1_layer_name = name + f".layer.{layer_idx}.DenseReluDense.wi"
        
        if model.config.is_gated_act:
            fc1_layer_name = name + f".layer.{layer_idx}.DenseReluDense.wi_0"
            gate_layer_name = name + f".layer.{layer_idx}.DenseReluDense.wi_1"
            smoother = smooth_gemm_fc1_gate(
                submodule.DenseReluDense.wi_0.weight,
                submodule.DenseReluDense.wi_1.weight,
                scales[fc1_layer_name]["x"],
                submodule.layer_norm.weight,
                None, alpha)
            scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
            scales[gate_layer_name]["w"] = submodule.DenseReluDense.wi_1.weight.abs().max(
                dim=1)[0]
            scales[fc1_layer_name]["w"] = submodule.DenseReluDense.wi_0.weight.abs().max(
            dim=1)[0]
        else:
            smoother = smooth_gemm(
                submodule.DenseReluDense.wi.weight,
                scales[fc1_layer_name]["x"],
                submodule.layer_norm.weight,
                None, alpha)
            scales[fc1_layer_name]["w"] = submodule.DenseReluDense.wi.weight.abs().max(
            dim=1)[0]

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        
        
        mlp_proj_layer_name = name + f".layer.{layer_idx}.DenseReluDense.wo"
        smoother = smooth_gemm(submodule.DenseReluDense.wo.weight,
                               scales[mlp_proj_layer_name]["x"], None, None, alpha)
        t5_smoother[mlp_proj_layer_name] = smoother.float()
        scales[mlp_proj_layer_name]["x"] = scales[mlp_proj_layer_name]["x"] / smoother
        scales[mlp_proj_layer_name]["w"] = submodule.DenseReluDense.wo.weight.abs().max(
            dim=1)[0]

def split_and_convert_process(
    key,
    val,
    factor,
    saved_dir,
    act_range: Optional[dict] = None,
    int8_outputs: Optional[str] = None
) -> None:
    # The split_factor indicates the number of ranks to implement
    # distributed GEMMs. For Tensor Parallelism, each rank/GPU works
    # on split_hidden_dim // split_factor channels.

    saved_key = key
    LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

    if "shared.weight" in key or "layer_norm.weight" in key:
        # embedding table / layernorm weight, no split
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

    elif "relative_attention_bias" in key:
        # relative attention table, transpose [num_buckets, num_heads] -> [num_heads, num_buckets]
        # and split on num_heads // split_factor dim
        split_dim = 0
        val = np.ascontiguousarray(val.transpose(1, 0))
        if val.shape[0] % factor != 0:
            LOGGER.error(
                f"[ERROR] Relative attention table, number of heads {val.shape[0]} is not divisible by TP size {factor}!"
            )  # assert doesn't work
        split_vals = np.split(val, factor, axis=split_dim)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif ("SelfAttention.o.weight" in key or "EncDecAttention.o.weight" in key
          or "DenseReluDense.wo.weight" in key):
        # RowLinear projection weight W=[d_out, d_in/TP], split dim=-1
        split_dim = -1
        split_vals = np.split(val, factor, axis=split_dim)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
        
        if act_range is not None and int8_outputs == "all":
            vals_i8 = generate_int8(val, act_range, key)
            write_int8(vals_i8, saved_dir, key, split_dim, 0, factor)

    elif ("lm_head.weight" in key or "DenseReluDense.wi.weight" in key
          or "DenseReluDense.wi_0.weight" in key
          or "DenseReluDense.wi_1.weight" in key):
        # ColumnLinear projection weights W=[d_out/TP, d_in], split dim=0
        split_dim = 0
        if "DenseReluDense.wi_0.weight" in key:
            saved_key = key.replace("wi_0", "wi")
        elif "DenseReluDense.wi_1.weight" in key:
            saved_key = key.replace("wi_1", "wi2")
        split_vals = np.split(val, factor, axis=split_dim)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())
        
        if act_range is not None and int8_outputs == "all":
            vals_i8 = generate_int8(val, act_range, key)
            write_int8(vals_i8, saved_dir, saved_key, split_dim, 0, factor)

    elif (("encoder" in key and
           ("SelfAttention.q.weight" in key or "SelfAttention.k.weight" in key
            or "SelfAttention.v.weight" in key)) or
          ("decoder" in key and
           ("SelfAttention.q.weight" in key or "SelfAttention.k.weight" in key
            or "SelfAttention.v.weight" in key or "EncDecAttention.q.weight"
            in key or "EncDecAttention.k.weight" in key
            or "EncDecAttention.v.weight" in key))):
        # weight needs to be fused, handled by fuse_qkv()
        pass

    elif "encoder.embed_tokens.weight" in key or "decoder.embed_tokens.weight" in key:
        LOGGER.warning(f"Not save {key}, using shared.weight directly.")
    elif ".smoother" in key:
        split_vals = np.split(val, factor, axis=0)
        save_split(split_vals, saved_dir, key, 0, factor)

    else:
        LOGGER.warning(f"cannot find key '{key}' with shape {val.shape}")


def convert_checkpoint(args):
    saved_dir = Path(args.output_dir) / f"tp{args.inference_tensor_para_size}"
    saved_dir.mkdir(parents=True, exist_ok=True)

    t5_model = T5ForConditionalGeneration.from_pretrained(
        args.input_dir,
        device_map="auto" if not args.load_model_on_cpu else "cpu",
        torch_dtype=str_dtype_to_torch(args.weight_data_type),
    )

    config = configparser.ConfigParser()
    extra_configs["structure"]["use_gated_activation"] = str(
        t5_model.encoder.config.is_gated_act)

    config["encoder"] = {}
    for key, val in t5_model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = args.weight_data_type

    act_range = {}
    t5_smoother = {}  # smoother for inputs of SelfAttention.o and DenseReluDense.wo

    if args.smoothquant is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        # TODO: Provide ability to pass in custom dataset
        dataset = load_dataset("ccdv/cnn_dailymail",
                               '3.0.0',
                               cache_dir=args.dataset_cache_dir)
        act_range = capture_activation_range(
            t5_model,
            T5Tokenizer.from_pretrained(args.input_dir),
            dataset,
            num_samples=args.calibration_set_num_samples)
        if args.smoothquant is not None:
            smooth_t5_model(t5_model, act_range, args.smoothquant,
                               t5_smoother)

    # manually set q_scaling to offset attention scaling's effect.
    # TODO: modify kernels to control whether to disable attention scaling
    def get_offset_q_scaling(config) -> str:
        d_model = config.d_model
        num_heads = config.num_heads
        head_size = d_model / num_heads
        scaling = 1 / head_size**.5
        return str(scaling)

    config["encoder"]["q_scaling"] = get_offset_q_scaling(
        t5_model.encoder.config)

    config["decoder"] = {}
    for key, val in t5_model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type

    config["decoder"]["q_scaling"] = get_offset_q_scaling(
        t5_model.decoder.config)

    for key, val in extra_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val
    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)

    i_gpu_num = args.inference_tensor_para_size

    int8_outputs = None
    if args.smoothquant is not None:
        int8_outputs = "all"

    pool = multiprocessing.Pool(args.processes)
    starmap_args = []
    for name, param in t5_model.state_dict().items():
        if name.replace(".weight", "") in t5_smoother.keys():
            smoother = t5_smoother[name.replace(".weight", "")]
            smoother = smoother.detach().cpu().numpy()
            starmap_args.append(
                (
                    f"{name}.smoother".replace(".weight", ""),
                    smoother,
                    i_gpu_num,
                    saved_dir,
                    None,
                    int8_outputs
                )
            )
        starmap_args.append(
            (
                name,
                torch_to_numpy(param),
                i_gpu_num,
                saved_dir,
                act_range.get(name.replace(".weight", "")),
                int8_outputs
            )
        )
    pool.starmap_async(
        split_and_convert_process, starmap_args
    )

    pool.close()
    pool.join()

    fuse_qkv(t5_model, i_gpu_num, saved_dir, act_range, int8_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_dir",
                        "-i",
                        type=str,
                        help="Path to the framework checkpoint file",
                        required=True)
    parser.add_argument("--output_dir",
                        "-o",
                        type=str,
                        help="Path to the converted TRT-LLM model weight file",
                        required=True)
    parser.add_argument("--inference_tensor_para_size",
                        "-i_g",
                        type=int,
                        help="How many gpus for inference",
                        required=True)
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4)
    parser.add_argument("--weight_data_type",
                        type=str,
                        default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
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
    parser.add_argument("--dataset-cache-dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument(
        "--calibration_set_num_samples",
        type=int,
        default=512,
        help="Number of samples used for capture_activation_range. Only used for SmoothQuant."
    )
    parser.add_argument(
        "--load_model_on_cpu",
        action="store_true",
        default=False,
        help="Whether to load the model on CPU."
    )
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
