import argparse
import functools
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import safetensors
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size, str_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


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
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--inter_size', type=int, default=22016)
    parser.add_argument('--rms_norm_eps', type=float, default=1e-06)

    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')
    parser.add_argument(
        '--disable_weight_only_quant_plugin',
        default=False,
        action="store_true",
        help=
        'By default, using plugin implementation for weight quantization. Enabling disable_weight_only_quant_plugin flag will use ootb implementation instead of plugin.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
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
        '--ammo_quant_ckpt_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .npz format')

    parser.add_argument(
        '--per_group',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.')

    parser.add_argument('--hidden_act', type=str, default='silu')

    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)

    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ/AWQ quantization.')

    parser.add_argument("--storage-type",
                        "-t",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"])
    parser.add_argument("--dataset-cache-dir",
                        type=str,
                        default=None,
                        help="cache dir to load the hugging face dataset")
    parser.add_argument("--load_model_on_cpu", action="store_true")

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
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument(
        '--dense_context_fmha',
        default=False,
        action='store_true',
        help=
        'Enable dense fmha in context phase, otherwise sliding window attention.'
        'If dense_context_fmha=False, the sliding window size is the max attention window size.'
    )
    args = parser.parse_args()
    return args


def load_from_gptq_qwen(
        model,
        num_hidden_layers=None,
        mapping=Mapping(),
        dtype="float16",
):
    tensorrt_llm.logger.info(
        "loading weights from groupwise GPTQ QWen safetensors...")
    weights = {}
    tik = time.time()

    model_params = {k: v for k, v in model.state_dict().items()}
    torch.cuda.empty_cache()

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def unpack_int32_into_int8(w_packed):
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
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
                                           torch.quint4x2).view(torch.float16)
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

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = model_params['transformer.wte.weight']
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)

    # 2. ln_f
    v = model_params['transformer.ln_f.weight']
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)

    # 3. lm_head
    v = model_params['lm_head.weight']
    if mapping.is_last_pp_rank():
        weights['lm_head.weight'] = torch_split(v, 0).to(torch_dtype)

    # 4. Weights inside each layer
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
    suffixs = ["qweight", "qzeros", "scales"]

    for l in tqdm(layers_range, desc="loading weight in each layer..."):
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "transformer.h." + str(layer_idx) + "."
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # 4.1 attention.qkv
        qkv_weight_list = []
        for suf in suffixs:
            qkv_part = model_params[prefix + "attn.c_attn." + suf]
            qkv_weight_list.append(qkv_part)
        weights.update(
            process_and_assign_weight(qkv_weight_list,
                                      f'{tllm_prex}.attention.qkv'))
        # 4.2 attention.bias
        qkv_bias = model_params[prefix + "attn.c_attn.bias"].to(
            torch_dtype).cpu().contiguous()
        q_emb = qkv_bias.shape[0] // 3
        qkv_bias = qkv_bias.reshape(3, q_emb)
        split_v = split(qkv_bias, mapping.tp_size, mapping.rank, dim=1)
        split_v = split_v.reshape(3 * (q_emb // mapping.tp_size))
        weights[tllm_prex + ".attention.qkv.bias"] = split_v
        # 4.3 attention.dense
        qkv_dense_list = []
        for suf in suffixs:
            qkv_dense_part = model_params[prefix + "attn.c_proj." + suf]
            qkv_dense_list.append(qkv_dense_part)
        weights.update(
            process_and_assign_weight(qkv_dense_list,
                                      f'{tllm_prex}.attention.dense',
                                      tp_dim=0))
        # 4.4 mlp.gate
        mlp_gate_list = []
        for suf in suffixs:
            mlp_gate_part = model_params[prefix + "mlp.w1." + suf]
            mlp_gate_list.append(mlp_gate_part)
        weights.update(
            process_and_assign_weight(mlp_gate_list,
                                      f'{tllm_prex}.mlp.gate',
                                      tp_dim=1))
        # 4.5 mlp.proj
        mlp_proj_list = []
        for suf in suffixs:
            mlp_proj_part = model_params[prefix + "mlp.c_proj." + suf]
            mlp_proj_list.append(mlp_proj_part)
        weights.update(
            process_and_assign_weight(mlp_proj_list,
                                      f'{tllm_prex}.mlp.proj',
                                      tp_dim=0))
        # 4.6 mlp.fc
        mlp_fc_list = []
        for suf in suffixs:
            mlp_fc_part = model_params[prefix + "mlp.w2." + suf]
            mlp_fc_list.append(mlp_fc_part)
        weights.update(
            process_and_assign_weight(mlp_fc_list,
                                      f'{tllm_prex}.mlp.fc',
                                      tp_dim=1))
        # 4.7 input_layernorm
        v = model_params[prefix + "ln_1.weight"]
        weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)
        # 4.8 post_layernorm
        v = model_params[prefix + "ln_2.weight"]
        weights[f'{tllm_prex}.post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime("%h:%m:%s", time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f"weights loaded. total time: {t}")

    return weights


def make_context(
    tokenizer,
    query,
    history,
    system,
    max_input_length,
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return (f"{role}\n{content}",
                    tokenizer.encode(
                        role,
                        allowed_special=set(),
                    ) + nl_tokens + tokenizer.encode(
                        content,
                        allowed_special=set(),
                    ))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens

            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response)
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens
            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (len(system_tokens) +
                                    len(next_context_tokens) +
                                    len(context_tokens))
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (nl_tokens + im_start_tokens +
                           _tokenize_str("user", query)[1] + im_end_tokens +
                           nl_tokens + im_start_tokens +
                           tokenizer.encode("assistant") + nl_tokens)
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    # truncate to max_input_length, truncate from the front
    return raw_text, context_tokens[-max_input_length:]


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
def smooth_qwen_model(model, scales, alpha, qwen_qkv_para, qwen_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not module._get_name() == "QWenBlock":
            continue
        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight,
                               scales[layer_name]["x"], module.ln_1.weight,
                               None, alpha)

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=1)[0]

        # see transpose_weights function
        qwen_qkv_para[layer_name] = module.attn.c_attn.weight.transpose(0, 1)

        # =================================================================
        layer_name = name + ".attn.c_proj"
        smoother = smooth_gemm(
            module.attn.c_proj.weight,
            scales[layer_name]["x"],
            None,
            None,
            alpha=alpha,
        )
        qwen_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_proj.weight.abs().max(dim=1)[0]
        # ==================================================================
        fc1_layer_name = name + ".mlp.w1"
        gate_layer_name = name + ".mlp.w2"

        smoother = smooth_gemm_fc1_gate(module.mlp.w1.weight,
                                        module.mlp.w2.weight,
                                        scales[fc1_layer_name]["x"],
                                        module.ln_2.weight, None, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.w1.weight.abs().max(dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.w2.weight.abs().max(dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.c_proj"
        smoother = smooth_gemm(module.mlp.c_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_proj.weight.abs().max(dim=1)[0]


@torch.no_grad()
def capture_activation_range(model,
                             tokenizer,
                             dataset,
                             system_prompt,
                             chat_format,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    tokenizer.pad_token_id = tokenizer.im_end_id

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
        line = dataset['train'][i]["article"]
        line = line + ' TL;DR: '
        line = line.strip()
        line = line.replace(" n't", "n't")
        _, input_id_list = make_context(tokenizer=tokenizer,
                                        query=line,
                                        history=[],
                                        system=system_prompt,
                                        chat_format=chat_format,
                                        max_input_length=seq_len)
        line_encoded = torch.from_numpy(np.array(
            input_id_list, dtype=np.int32)).type(torch.int32).unsqueeze(0)
        line_encoded = line_encoded.to(device)
        model(line_encoded)
    for h in hooks:
        h.remove()
    return act_scales


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def split_qkv_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV matrix according to tensor parallelism
    """
    v = v.reshape(3, n_hidden, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel), n_hidden)
    return split_v.contiguous()


def split_qkv_bias_tp(v, n_head, n_hidden, tensor_parallel, rank):
    """
    Splits the QKV bias according to tensor parallelism
    """
    v = v.reshape(3, n_hidden)
    split_v = split(v, tensor_parallel, rank, dim=1)
    split_v = split_v.reshape(3 * (n_hidden // tensor_parallel))
    return split_v.contiguous()


def split_matrix_tp(v, tensor_parallel, rank, dim):
    return split(v, tensor_parallel, rank, dim=dim)


def get_weight(config, prefix, dtype):
    if config[prefix + '.weight'].dtype != dtype:
        config[prefix + '.weight'].data = config[prefix + '.weight'].to(dtype)
    return config[prefix + '.weight']


def get_bias(config, prefix, dtype):
    if config[prefix + '.bias'].dtype != dtype:
        config[prefix + '.bias'].data = config[prefix + '.bias'].to(dtype)
    return config[prefix + '.bias']


def get_weight_and_bias(config, prefix, dtype):
    return get_weight(config, prefix, dtype), get_bias(config, prefix, dtype)


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8,
                           dtype='float32',
                           use_gemm_woq_plugin=True,
                           postfix='weight'):
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        if not use_gemm_woq_plugin:
            results[prefix + postfix] = v.to(dtype)
        else:
            results[prefix + postfix] = processed_torch_weights
        results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.contiguous()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


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
        original_weights = vals["weight.int8.col"]

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

        if smoother_value is None:
            if multi_query_mode:
                cur_per_channel_value = multi_query_split(
                    vals["scale_w_quant_orig.col"], local_dim, head_size,
                    tensor_parallel, rank)
            else:
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
        # 'weight'] = torch.from_numpy(cur_weights).t().contiguous()

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


def convert_hf_qwen(hf_model,
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
                    smoother=[]):
    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_model.config.num_attention_heads
    hidden_size = hf_model.config.hidden_size
    intermediate_size = hf_model.config.intermediate_size // 2  # Qwen's actual intermediate_size is one half of what's in hf_config
    num_key_value_heads = hf_model.config.num_key_value_heads if hasattr(
        hf_model.config, "num_key_value_heads") else num_attention_heads
    mha_mode = (num_key_value_heads == num_attention_heads)
    assert mha_mode == True, "QWen uses MHA."
    layers_range = mapping.pp_layers(hf_model.config.num_hidden_layers)

    for l in layers_range:
        prefix = f'transformer.h.{l}.'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        qkv_weight, qkv_bias = get_weight_and_bias(model_params,
                                                   prefix + 'attn.c_attn',
                                                   dtype)
        qkv_w = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                             tensor_parallel, mapping.tp_rank)
        qkv_b = split_qkv_bias_tp(qkv_bias, num_attention_heads, hidden_size,
                                  tensor_parallel, mapping.tp_rank)

        if use_smooth_quant:
            qkv_weight = qkv_para[prefix + 'attn.c_attn']
            qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)

            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix + 'attn.c_attn'),
                                         is_qkv=True,
                                         multi_query_mode=bool(not mha_mode))

            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.qkv.', [
                        1, 3 * hidden_size // tensor_parallel
                        if mha_mode else hidden_size // tensor_parallel +
                        (hidden_size // num_key_value_heads) //
                        tensor_parallel * 2
                    ],
                    tensor_parallel,
                    is_qkv=True,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'input_layernorm.scale_to_int',
                    bias=qkv_bias,
                    smoother_value=None,
                    smoother_shape=None,
                    rank=mapping.tp_rank,
                    cat_dim=-1,
                    multi_query_mode=bool(not mha_mode)))
        else:
            weights.update(
                get_tllm_linear_weight(qkv_w, tllm_prex + 'attention.qkv.',
                                       qkv_b, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        if int8_kv_cache:
            qkv_y = act_range.get(prefix + 'attn.c_attn')["y"]

            int8_kv_scales = qkv_y.max() / 127.

            kv_cache_weights = {}

            kv_cache_weights[
                tllm_prex +
                'attention.kv_cache_scaling_factor'] = int8_kv_scales.reshape(
                    [1])

            weights.update(kv_cache_weights)

        attn_dense_weight = get_weight(model_params, prefix + 'attn.c_proj',
                                       dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(attn_dense_weight,
                                         act_range.get(prefix + 'attn.c_proj'))
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
                    smoother_value=smoother[(prefix + 'attn.c_proj')],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'attention.dense.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        mlp_gate_weight = get_weight(model_params, prefix + 'mlp.w1', dtype)
        split_v = split_matrix_tp(mlp_gate_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=0)
        if use_smooth_quant:
            mlp_gate_weight = mlp_gate_weight.t()
            int8_weights = generate_int8(mlp_gate_weight,
                                         act_range.get(prefix + 'mlp.w1'))

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
                get_tllm_linear_weight(split_v, tllm_prex + 'mlp.gate.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        mlp_fc_weight = get_weight(model_params, prefix + 'mlp.w2', dtype)
        split_v = split_matrix_tp(mlp_fc_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=0)

        if use_smooth_quant:
            mlp_fc_weight = mlp_fc_weight.t()  #verified
            int8_weights = generate_int8(mlp_fc_weight,
                                         act_range.get(prefix + 'mlp.w2'))
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

        mlp_proj_weight = get_weight(model_params, prefix + 'mlp.c_proj', dtype)
        split_v = split_matrix_tp(mlp_proj_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)

        if use_smooth_quant:
            mlp_proj_weight = mlp_proj_weight.t()
            int8_weights = generate_int8(mlp_proj_weight,
                                         act_range.get(prefix + 'mlp.c_proj'))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'mlp.proj.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex + 'mlp.quantization_scaling_factor',
                    smoother_value=smoother[prefix + 'mlp.c_proj'],
                    smoother_shape=[1, intermediate_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + 'ln_1', dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_ln_weight = get_weight(model_params, prefix + 'ln_2', dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

    v = get_weight(model_params, 'transformer.wte', dtype)

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
        ln_f_w = get_weight(model_params, 'transformer.ln_f', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def main():
    # TODO(qijun): Currently, the convert script depends on a torch op:
    # torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix,
    # which is included in tensorrt_llm Python package. Otherwise, the convert
    # script does not need to import tensorrt_llm. Will remove it after reimplementing
    # the op with PyTorch.
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    hf_config = None
    if args.model_dir is not None:
        hf_config = AutoConfig.from_pretrained(args.model_dir,
                                               trust_remote_code=True)
        args.model_type = hf_config.model_type
        args.n_head = hf_config.num_attention_heads
        args.inter_size = hf_config.intermediate_size
        args.n_layer = hf_config.num_hidden_layers
        args.n_embd = hf_config.hidden_size
        if hasattr(hf_config, "num_key_value_heads"):
            args.n_kv_head = hf_config.num_key_value_heads
        args.rms_norm_eps = hf_config.layer_norm_epsilon
        args.vocab_size = hf_config.vocab_size
        args.n_positions = hf_config.max_position_embeddings
        args.rotary_base = hf_config.rotary_emb_base
    args.n_kv_head = args.n_kv_head or args.n_head

    if args.rotary_scaling is not None:
        # assert args.use_gpt_attention_plugin, "RoPE scaling is only supported through GPT attention plugin."
        rotary_scaling = {
            "type": args.rotary_scaling[0],
            "factor": float(args.rotary_scaling[1])
        }
        assert rotary_scaling["type"] in ["linear", "dynamic"]
        assert rotary_scaling["factor"] > 1.0
        args.rotary_scaling = rotary_scaling

    config = {
        'architecture': "QWenForCausalLM",
        'dtype': args.dtype,
        'logits_dtype': 'float32',
        'num_hidden_layers': args.n_layer,
        'num_attention_heads': args.n_head,
        'hidden_size': args.n_embd,
        'intermediate_size': args.inter_size,
        'num_key_value_heads': args.n_kv_head,
        'vocab_size': args.vocab_size,
        'position_embedding_type': 'rope_gpt_neox',
        'max_position_embeddings': args.n_positions,
        'hidden_act': args.hidden_act,
        'rotary_base': args.rotary_base,
        'rotary_scaling': args.rotary_scaling,
        'norm_epsilon': args.rms_norm_eps,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'dense_context_fmha': args.dense_context_fmha,
        'disable_weight_only_quant_plugin':
        args.disable_weight_only_quant_plugin
    }

    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            config['quantization']['quant_algo'] = 'W8A16'
        elif args.weight_only_precision == 'int4':
            config['quantization']['quant_algo'] = 'W4A16'
    elif args.smoothquant:
        if args.per_channel:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN'
            else:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN'
        else:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN'
            else:
                config['quantization'][
                    'quant_algo'] = 'W8A8_SQ_PER_TENSOR_PLUGIN'

    if args.int8_kv_cache:
        config['quantization']['kv_cache_quant_algo'] = 'INT8'

    if args.weight_only_precision == 'int4_gptq':
        config['quantization'].update({
            "group_size": args.group_size,
            "has_zero_point": True,
            "pre_quant_scale": False,
            'quant_algo': 'W4A16_GPTQ'
        })

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.model_dir is None:
        return

    if args.weight_only_precision == 'int8':
        plugin_weight_only_quant_type = torch.int8
    elif args.weight_only_precision == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2

    act_range = {}
    qwen_qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    qwen_smoother = {}
    model = None
    if args.model_dir is not None:
        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir, device_map="auto",
                trust_remote_code=True).eval().cpu()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                device_map='auto' if not args.load_model_on_cpu else 'cpu',
                torch_dtype='auto' if not args.smoothquant else torch.float16,
                trust_remote_code=True,
            ).half()

        if args.smoothquant is not None or args.int8_kv_cache:
            os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
                "TOKENIZERS_PARALLELISM", "false")
            if args.load_model_on_cpu:
                logger.warning(
                    "Note that running capture_activation_range on cpu would be very small."
                )
            dataset = load_dataset("ccdv/cnn_dailymail",
                                   '3.0.0',
                                   cache_dir=args.dataset_cache_dir)
            system_prompt = "You are a useful assistant, please directly output the corresponding summary according to the article entered by the user."
            gen_config_path = os.path.join(args.model_dir,
                                           'generation_config.json')
            with open(gen_config_path, 'r') as f:
                gen_config = json.load(f)
            chat_format = gen_config['chat_format']
            act_range = capture_activation_range(
                model,
                AutoTokenizer.from_pretrained(args.model_dir,
                                              trust_remote_code=True,
                                              use_fast=False,
                                              padding_side='left'), dataset,
                system_prompt, chat_format)
            if args.smoothquant is not None:
                smooth_qwen_model(model, act_range, args.smoothquant,
                                  qwen_qkv_para, qwen_smoother)
    convert_args = {
        'hf_model': model,
        'act_range': act_range,
        'qwen_qkv_para': qwen_qkv_para,
        'qwen_smoother': qwen_smoother,
    }

    def covert_and_save(rank, convert_args):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        if args.use_weight_only and args.weight_only_precision == 'int4_gptq':
            weights = load_from_gptq_qwen(convert_args['hf_model'],
                                          args.n_layer,
                                          mapping,
                                          dtype=args.dtype)

        else:
            weights = convert_hf_qwen(
                convert_args['hf_model'],
                mapping,
                vocab_size=args.vocab_size,
                dtype=args.dtype,
                use_weight_only=args.use_weight_only,
                use_gemm_woq_plugin=not args.disable_weight_only_quant_plugin,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type,
                use_parallel_embedding=args.use_parallel_embedding,
                sharding_dim=args.embedding_sharding_dim,
                share_embedding_table=args.use_embedding_sharing,
                use_smooth_quant=args.smoothquant,
                per_channel=args.per_channel,
                per_token=args.per_token,
                int8_kv_cache=args.int8_kv_cache,
                act_range=convert_args['act_range'],
                qkv_para=convert_args['qwen_qkv_para'],
                smoother=convert_args['qwen_smoother'])

        safetensors.torch.save_file(
            weights, os.path.join(args.output_dir, f'rank{rank}.safetensors'))

    if args.workers == 1:

        for rank in range(world_size):
            covert_and_save(rank, convert_args)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            futures = [
                p.submit(covert_and_save, rank, convert_args)
                for rank in range(world_size)
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


if __name__ == '__main__':
    main()
