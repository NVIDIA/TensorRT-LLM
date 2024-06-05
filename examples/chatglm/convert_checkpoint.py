import argparse
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
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import load_calib_dataset
from tensorrt_llm.quantization import QuantAlgo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument(
        '--chatglm_version',
        default=None,
        choices=[None, 'glm', 'chatglm', 'chatglm2', 'chatglm3'],
        help=
        "By default the script will try to infer the chatglm_version from model_dir. "
        "Or users may overwrite chatglm_version by explicitly passing the version."
    )
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
        '--use_embedding_sharing',
        action="store_true",
        default=False,
        help=
        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.')

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
        '--calib_dataset',
        type=str,
        default='cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
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

    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT-LLM checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    args = parser.parse_args()

    return args


def load_chatglm_config(model_dir: str,
                        chatglm_version: Optional[str] = None) -> AutoConfig:
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)

    if chatglm_version is None:
        print("Inferring chatglm version from path...")
        for v in ['chatglm3', 'chatglm2', 'chatglm', 'glm']:
            if v in config._name_or_path:
                chatglm_version = v
                break
    assert chatglm_version in ['glm', 'chatglm', 'chatglm2', 'chatglm3']
    print(f"Chatglm version: {chatglm_version}")
    if chatglm_version == 'glm':
        config.num_kv_heads = config.num_attention_heads
        config.ffn_hidden_size = config.hidden_size * 4
        config.hidden_act = 'gelu'
        config.layernorm_epsilon = 1e-5
        config.max_position_embeddings = config.max_sequence_length
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.apply_query_key_layer_scaling = False
        config.apply_residual_connection_post_layernorm = False
        config.rmsnorm = False
        config.rope_ratio = 1.0
    elif chatglm_version == 'chatglm':
        config.num_kv_heads = config.num_attention_heads
        config.ffn_hidden_size = config.inner_hidden_size
        config.hidden_act = 'gelu'
        config.max_position_embeddings = config.max_sequence_length
        config.add_bias_linear = True
        config.add_qkv_bias = True
        config.apply_query_key_layer_scaling = False
        config.apply_residual_connection_post_layernorm = False
        config.rmsnorm = False
        config.rope_ratio = 1.0
    else:
        config.vocab_size = config.padded_vocab_size
        config.num_kv_heads = config.multi_query_group_num
        config.hidden_act = 'swiglu'
        config.max_position_embeddings = config.seq_length
        config.rmsnorm = getattr(config, 'rmsnorm', 1.0)
        config.rope_ratio = getattr(config, 'rope_ratio', 1.0)

    return config, chatglm_version


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


def tile_kv_weight_bias(v: torch.Tensor, kv_num_head: int, tp_size: int):
    head_size = v.shape[0] // kv_num_head
    reps = tp_size // kv_num_head
    if v.ndim == 1:
        v = v.reshape(kv_num_head, head_size)[:, None, :]
        v = v.expand(kv_num_head, reps, head_size).reshape(-1).clone()
    else:
        hidden_size = v.shape[1]
        v = v.reshape(kv_num_head, head_size, hidden_size)[:, None, :, :]
        v = v.expand(kv_num_head, reps, head_size,
                     hidden_size).reshape(-1, hidden_size).clone()
    return v


def split_qkv(v: torch.Tensor, tp_size: int, rank: int, hidden_size: int,
              num_heads: int, num_kv_heads: int):
    head_size = hidden_size // num_heads
    if tp_size == 1:
        return v

    assert v.shape[0] == hidden_size + head_size * num_kv_heads * 2
    query = v[:hidden_size]
    key = v[hidden_size:hidden_size + head_size * num_kv_heads]
    value = v[hidden_size + head_size * num_kv_heads:hidden_size +
              head_size * num_kv_heads * 2]

    if num_kv_heads < tp_size:
        key = tile_kv_weight_bias(key, num_kv_heads, tp_size)
        value = tile_kv_weight_bias(value, num_kv_heads, tp_size)
    assert (key.shape[0] % (tp_size * head_size)) == 0
    assert (value.shape[0] % (tp_size * head_size)) == 0

    q_tmp = torch.chunk(query, tp_size, dim=0)[rank]
    k_tmp = torch.chunk(key, tp_size, dim=0)[rank]
    v_tmp = torch.chunk(value, tp_size, dim=0)[rank]
    return torch.concatenate([q_tmp, k_tmp, v_tmp], dim=0).contiguous()


def split_embedding(
    param: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    use_parallel_embedding: bool = False,
    sharding_dim: int = 0,
) -> torch.Tensor:
    if param is None:
        return None
    if not use_parallel_embedding:
        return param

    vocab_size, hidden_size = param.size()
    if sharding_dim == 0:
        if vocab_size % tp_size != 0:
            vocab_size_padded = pad_vocab_size(vocab_size, tp_size)
            pad_width = vocab_size_padded - vocab_size
            param = torch.nn.functional.pad(param, (0, 0, 0, pad_width),
                                            value=0)
        else:
            assert hidden_size % tp_size == 0
    return split(param, tp_size, tp_rank, dim=sharding_dim)


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


def swap_and_split_mlp(weight: torch.Tensor,
                       tp_size: int,
                       tp_rank: int,
                       dim: int = 0) -> torch.Tensor:
    """Swap the positions of gate and fc weights, and split weights for tensor parallel.
    """
    gate_weight, fc_weight = torch.chunk(weight, 2, dim=dim)
    fc_w = split(fc_weight, tp_size, tp_rank, dim=dim)
    gate_w = split(gate_weight, tp_size, tp_rank, dim=dim)
    return torch.cat([fc_w, gate_w], dim=dim).contiguous()


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


@torch.no_grad()
def apply_smoothing(
    scales,
    gemm_weights,
    norm_weights=None,
    norm_bias=None,
    dtype=torch.float32,
    norm_1p=False,
):
    if not isinstance(gemm_weights, list):
        gemm_weights = [gemm_weights]

    if norm_weights is not None:
        assert norm_weights.numel() == scales.numel()
        norm_weights.div_(scales).to(dtype)
    if norm_bias is not None:
        assert norm_bias.numel() == scales.numel()
        norm_bias.div_(scales).to(dtype)
    if norm_1p:
        norm_weights += (1 / scales) - 1

    for gemm in gemm_weights:
        gemm.mul_(scales.view(1, -1)).to(dtype)


@torch.no_grad()
def smooth_gemm(
    gemm_weights,
    act_scales,
    norm_weights=None,
    norm_bias=None,
    alpha=0.5,
    weight_scales=None,
):
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

    apply_smoothing(scales, gemm_weights, norm_weights, norm_bias, orig_dtype)

    return scales


@torch.no_grad()
def capture_activation_range(
    model,
    tokenizer,
    dataset,
    num_samples=64,
    seq_len=512,
):

    model.eval()
    device = next(model.parameters()).device
    scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    def stat_tensor(name, tensor, key):
        tensor = tensor.view(-1, tensor.shape[-1]).detach()
        comming_max = tensor.abs().max(dim=0)[0].float()
        if scales[name][key] is None:
            scales[name][key] = comming_max
        else:
            scales[name][key] = torch.max(scales[name][key], comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, "x")
        stat_tensor(name, y, "y")
        if scales[name]["w"] is None:
            scales[name]["w"] = m.weight.abs().clip(1e-8, None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="Calibration"):
        input_ids = tokenizer(
            dataset[i],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        )
        model(input_ids.input_ids.to(device))

    for h in hooks:
        h.remove()

    return scales


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


@torch.no_grad()
def smooth_chatglm_model(
    model,
    act_range,
    alpha,
    model_smoother,
):
    for name, module in model.named_modules():
        if not module._get_name() == "GLMBlock":
            continue

        # QKV multiplication weight
        layer_name = name + '.self_attention.query_key_value'
        print(f'Smoothing module: {layer_name}')
        weight = module.self_attention.query_key_value.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            module.input_layernorm.weight,
            None,
            alpha,
        )
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Dense multiplication weight
        layer_name = name + ".self_attention.dense"
        print(f'Smoothing module: {layer_name}')
        weight = module.self_attention.dense.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            None,
            None,
            alpha,
        )
        model_smoother[layer_name] = smoother.float()
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Multilayer perceptron h -> 4h weight
        layer_name = name + ".mlp.dense_h_to_4h"
        print(f'Smoothing module: {layer_name}')
        weight = module.mlp.dense_h_to_4h.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            module.post_attention_layernorm.weight,
            None,
            alpha,
        )
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]

        # Multilayer perceptron 4h -> h weight
        layer_name = name + ".mlp.dense_4h_to_h"
        print(f'Smoothing module: {layer_name}')
        weight = module.mlp.dense_4h_to_h.weight
        smoother = smooth_gemm(
            weight,
            act_range[layer_name]["x"],
            None,
            None,
            alpha,
        )
        model_smoother[layer_name] = smoother.float()
        act_range[layer_name]["x"] = act_range[layer_name]["x"] / smoother
        act_range[layer_name]["w"] = weight.abs().max(dim=1)[0]


def get_tllm_linear_sq_weight(vals,
                              prefix,
                              shape,
                              is_qkv=False,
                              per_token=False,
                              per_channel=False,
                              last_prefix=None,
                              smoother_value=None,
                              smoother_shape=None):
    results = {}
    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
        if per_channel:
            original_weights = np.array(vals["weight.int8.col"])
        else:
            original_weights = np.array(vals["weight.int8"])

        cur_weights = original_weights
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(
                np.array([1.0], dtype=np.float16))

        if per_channel:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig"]

        results[prefix + 'per_channel_scale'] = torch.from_numpy(
            np.array(cur_per_channel_value,
                     dtype=np.float32).reshape(col_shape)).contiguous()
    else:
        if per_channel:
            original_weights = np.array(vals["weight.int8.col"])
        else:
            original_weights = np.array(vals["weight.int8"])
        cur_weights = original_weights

        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix +
                'weight'] = torch.from_numpy(cur_weights).t().contiguous()

        if per_channel:
            cur_per_channel_value = vals["scale_y_accum_quant.col"]
        else:
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
        results[prefix + 'smoother'] = smoother_value.reshape(
            smoother_shape).contiguous().to(torch.float32)

    return results


def convert_hf_chatglm(hf_model: AutoModel,
                       hf_config: AutoConfig,
                       chatglm_version: str,
                       mapping: Mapping,
                       dtype: str = 'float32',
                       use_parallel_embedding: bool = False,
                       sharding_dim: int = 0,
                       share_embedding_table: bool = False,
                       use_weight_only: bool = False,
                       plugin_weight_only_quant_type: str = 'int8',
                       use_smooth_quant: bool = False,
                       per_channel=False,
                       per_token=False,
                       int8_kv_cache=False,
                       act_range=None,
                       smoother=None):
    weights = {}
    tik = time.time()

    model_params = dict(hf_model.named_parameters())
    dtype = getattr(torch, dtype)
    num_attention_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    hf_config.vocab_size
    num_kv_heads = getattr(hf_config, 'num_kv_heads', num_attention_heads)
    num_hidden_layers = hf_config.num_layers

    layers_range = mapping.pp_layers(num_hidden_layers)
    for l in layers_range:
        if chatglm_version in ['glm', 'chatglm']:
            prefix = f'transformer.layers.{l}'
        elif chatglm_version in ['chatglm2', 'chatglm3']:
            prefix = f'transformer.encoder.layers.{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        # Attention QKV
        if chatglm_version in ['glm', 'chatglm']:
            qkv_weight, qkv_bias = get_weight_and_bias(
                model_params, f'{prefix}.attention.query_key_value', dtype)
            qkv_act_range = act_range.get(f'{prefix}.attention.query_key_value')
        elif chatglm_version in ['chatglm2', 'chatglm3']:
            qkv_weight, qkv_bias = get_weight_and_bias(
                model_params, f'{prefix}.self_attention.query_key_value', dtype)
            qkv_act_range = act_range.get(
                f'{prefix}.self_attention.query_key_value')

        if use_smooth_quant:
            qkv_vals_int8 = generate_int8(qkv_weight.t().numpy(),
                                          qkv_act_range,
                                          is_qkv=True,
                                          multi_query_mode=True)
            weights.update(
                get_tllm_linear_sq_weight(
                    vals=qkv_vals_int8,
                    prefix=f'{tllm_prex}.attention.qkv.',
                    shape=[1, qkv_weight.size(0)],
                    is_qkv=True,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=f'{tllm_prex}.input_layernorm.scale_to_int',
                    smoother_value=None,
                    smoother_shape=None))
            if qkv_bias is not None:
                qkv_b = split_qkv(qkv_bias,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  hidden_size,
                                  num_attention_heads,
                                  num_kv_heads=num_kv_heads)
                weights[f'{tllm_prex}.attention.qkv.bias'] = qkv_b
        else:
            qkv_w = split_qkv(qkv_weight,
                              mapping.tp_size,
                              mapping.tp_rank,
                              hidden_size,
                              num_attention_heads,
                              num_kv_heads=num_kv_heads)
            if qkv_bias is None:
                qkv_b = None
            else:
                qkv_b = split_qkv(qkv_bias,
                                  mapping.tp_size,
                                  mapping.tp_rank,
                                  hidden_size,
                                  num_attention_heads,
                                  num_kv_heads=num_kv_heads)

            weights.update(
                get_tllm_linear_weight(qkv_w, f'{tllm_prex}.attention.qkv',
                                       qkv_b, use_weight_only,
                                       plugin_weight_only_quant_type))

        if int8_kv_cache:
            qkv_vals_int8 = generate_int8(qkv_weight.t().numpy(),
                                          qkv_act_range,
                                          is_qkv=True,
                                          multi_query_mode=True)
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = torch.from_numpy(
                    np.array([qkv_vals_int8['scale_y_quant_orig']],
                             dtype=np.float32)).contiguous()

        # Attention dense
        if chatglm_version in ['glm', 'chatglm']:
            attn_dense_weight, attn_dense_bias = get_weight_and_bias(
                model_params, f'{prefix}.attention.dense', dtype)
            dense_act_range = act_range.get(f'{prefix}.attention.dense')
            dense_smoother = smoother.get(f'{prefix}.attention.dense')
        else:
            attn_dense_weight, attn_dense_bias = get_weight_and_bias(
                model_params, f'{prefix}.self_attention.dense', dtype)
            dense_act_range = act_range.get(f'{prefix}.self_attention.dense')
            dense_smoother = smoother.get(f'{prefix}.self_attention.dense')

        if use_smooth_quant:
            dense_vals_int8 = generate_int8(attn_dense_weight.t().numpy(),
                                            dense_act_range,
                                            is_qkv=False,
                                            multi_query_mode=True)
            weights.update(
                get_tllm_linear_sq_weight(
                    vals=dense_vals_int8,
                    prefix=f'{tllm_prex}.attention.dense.',
                    shape=[1, hidden_size],
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=
                    f'{tllm_prex}.attention.quantization_scaling_factor',
                    smoother_value=dense_smoother,
                    smoother_shape=[1, hidden_size]))
            if attn_dense_bias is not None:
                weights[f'{tllm_prex}.attention.dense.bias'] = attn_dense_bias
        else:
            attn_dense_w = split(attn_dense_weight,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 dim=1)
            weights.update(
                get_tllm_linear_weight(attn_dense_w,
                                       f'{tllm_prex}.attention.dense',
                                       attn_dense_bias, use_weight_only,
                                       plugin_weight_only_quant_type))

        # MLP FC
        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_h_to_4h', dtype)

        if use_smooth_quant:
            fc_act_range = act_range.get(f'{prefix}.mlp.dense_h_to_4h')
            fc_vals_int8 = generate_int8(mlp_fc_weight.t().numpy(),
                                         fc_act_range,
                                         is_qkv=False,
                                         multi_query_mode=True)
            cur_weights = get_tllm_linear_sq_weight(
                vals=fc_vals_int8,
                prefix=f'{tllm_prex}.mlp.fc.',
                shape=[1, mlp_fc_weight.size(0)],
                is_qkv=False,
                per_token=per_token,
                per_channel=per_channel,
                last_prefix=f'{tllm_prex}.post_layernorm.scale_to_int',
                smoother_value=None,
                smoother_shape=None,
            )
            cur_weights[f'{tllm_prex}.mlp.fc.weight'] = swap_and_split_mlp(
                cur_weights[f'{tllm_prex}.mlp.fc.weight'],
                mapping.tp_size,
                mapping.tp_rank,
                dim=0,
            )
            if per_channel:
                cur_weights[
                    f'{tllm_prex}.mlp.fc.per_channel_scale'] = swap_and_split_mlp(
                        cur_weights[f'{tllm_prex}.mlp.fc.per_channel_scale'],
                        mapping.tp_size,
                        mapping.tp_rank,
                        dim=1,
                    )
            weights.update(cur_weights)
            if chatglm_version in ['glm', 'chatglm']:
                if mlp_fc_bias is not None:
                    mlp_fc_b = split(mlp_fc_bias,
                                     mapping.tp_size,
                                     mapping.tp_rank,
                                     dim=0)
                    weights[f'{tllm_prex}.mlp.fc.bias'] = mlp_fc_b
            elif chatglm_version in ['chatglm2', 'chatglm3']:
                if mlp_fc_bias is not None:
                    mlp_fc_b = swap_and_split_mlp(mlp_fc_bias, mapping.tp_size,
                                                  mapping.tp_rank)
                    weights[f'{tllm_prex}.mlp.fc.bias'] = mlp_fc_b
        else:
            if chatglm_version in ['glm', 'chatglm']:
                mlp_fc_w = split(mlp_fc_weight,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 dim=0)
                if mlp_fc_bias is None:
                    mlp_fc_b = None
                else:
                    mlp_fc_b = split(mlp_fc_bias,
                                     mapping.tp_size,
                                     mapping.tp_rank,
                                     dim=0)
            elif chatglm_version in ['chatglm2', 'chatglm3']:
                mlp_fc_w = swap_and_split_mlp(mlp_fc_weight, mapping.tp_size,
                                              mapping.tp_rank)
                if mlp_fc_bias is None:
                    mlp_fc_b = None
                else:
                    mlp_fc_b = swap_and_split_mlp(mlp_fc_bias, mapping.tp_size,
                                                  mapping.tp_rank)
            weights.update(
                get_tllm_linear_weight(mlp_fc_w, f'{tllm_prex}.mlp.fc',
                                       mlp_fc_b, use_weight_only,
                                       plugin_weight_only_quant_type))

        # MLP Proj
        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_4h_to_h', dtype)

        if use_smooth_quant:
            proj_act_range = act_range.get(f'{prefix}.mlp.dense_4h_to_h')
            proj_smoother = smoother.get(f'{prefix}.mlp.dense_4h_to_h')
            proj_vals_int8 = generate_int8(mlp_proj_weight.t().numpy(),
                                           proj_act_range,
                                           is_qkv=False,
                                           multi_query_mode=True)
            weights.update(
                get_tllm_linear_sq_weight(
                    vals=proj_vals_int8,
                    prefix=f'{tllm_prex}.mlp.proj.',
                    shape=[1, hidden_size],
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=f'{tllm_prex}.mlp.quantization_scaling_factor',
                    smoother_value=proj_smoother,
                    smoother_shape=[1, hf_config.ffn_hidden_size]))
            if mlp_proj_bias is not None:
                weights[f'{tllm_prex}.mlp.proj.bias'] = mlp_proj_bias
        else:
            mlp_proj_w = split(mlp_proj_weight,
                               mapping.tp_size,
                               mapping.tp_rank,
                               dim=1)
            weights.update(
                get_tllm_linear_weight(mlp_proj_w, f'{tllm_prex}.mlp.proj',
                                       mlp_proj_bias, use_weight_only,
                                       plugin_weight_only_quant_type))

        input_ln_weight, input_ln_bias = get_weight_and_bias(
            model_params, f'{prefix}.input_layernorm', dtype)
        weights[f'{tllm_prex}.input_layernorm.weight'] = input_ln_weight
        if input_ln_bias is not None:
            weights[f'{tllm_prex}.input_layernorm.bias'] = input_ln_bias

        post_ln_weight, post_ln_bias = get_weight_and_bias(
            model_params, f'{prefix}.post_attention_layernorm', dtype)
        weights[f'{tllm_prex}.post_layernorm.weight'] = post_ln_weight
        if post_ln_bias is not None:
            weights[f'{tllm_prex}.post_layernorm.bias'] = post_ln_bias

    if mapping.is_first_pp_rank():
        if chatglm_version == 'glm':
            embed_w = get_weight(model_params, 'word_embeddings', dtype)
            pos_embed_w = get_weight(model_params,
                                     'transformer.position_embeddings', dtype)
            weights['transformer.position_embedding.weight'] = split_embedding(
                pos_embed_w,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                use_parallel_embedding=use_parallel_embedding,
                sharding_dim=sharding_dim)
            block_embed_w = get_weight(model_params,
                                       'transformer.block_position_embeddings',
                                       dtype)
            weights['transformer.block_embedding.weight'] = split_embedding(
                block_embed_w,
                tp_size=mapping.tp_size,
                tp_rank=mapping.tp_rank,
                use_parallel_embedding=use_parallel_embedding,
                sharding_dim=sharding_dim)
        elif chatglm_version == 'chatglm':
            embed_w = get_weight(model_params, 'transformer.word_embeddings',
                                 dtype)
        elif chatglm_version in ['chatglm2', 'chatglm3']:
            embed_w = get_weight(model_params,
                                 'transformer.embedding.word_embeddings', dtype)

        weights['transformer.vocab_embedding.weight'] = split_embedding(
            embed_w,
            tp_size=mapping.tp_size,
            tp_rank=mapping.tp_rank,
            use_parallel_embedding=use_parallel_embedding,
            sharding_dim=sharding_dim)

    if mapping.is_last_pp_rank():
        if chatglm_version == 'glm':
            lm_head_weight = get_weight(model_params, 'word_embeddings',
                                        dtype).clone()
        elif chatglm_version == 'chatglm':
            lm_head_weight = get_weight(model_params,
                                        'transformer.word_embeddings',
                                        dtype).clone()
        elif chatglm_version in ['chatglm2', 'chatglm3']:
            lm_head_weight = get_weight(model_params,
                                        'transformer.output_layer', dtype)
            assert not share_embedding_table

        if not share_embedding_table:
            weights['lm_head.weight'] = split(lm_head_weight,
                                              mapping.tp_size,
                                              mapping.tp_rank,
                                              dim=0)

        if chatglm_version in ['glm', 'chatglm']:
            ln_f_w, ln_f_b = get_weight_and_bias(model_params,
                                                 'transformer.final_layernorm',
                                                 dtype)
        elif chatglm_version in ['chatglm2', 'chatglm3']:
            ln_f_w, ln_f_b = get_weight_and_bias(
                model_params, 'transformer.encoder.final_layernorm', dtype)
        weights['transformer.ln_f.weight'] = ln_f_w
        if ln_f_b is not None:
            weights['transformer.ln_f.bias'] = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


if __name__ == '__main__':
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    world_size = args.tp_size * args.pp_size
    assert args.pp_size == 1, "Pipeline parallelism is not supported."

    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    hf_config, chatglm_version = load_chatglm_config(args.model_dir,
                                                     args.chatglm_version)

    if chatglm_version == 'glm':
        position_embedding_type = 'learned_absolute'
    elif chatglm_version == 'chatglm':
        position_embedding_type = 'chatglm'
    elif chatglm_version in ['chatglm2', 'chatglm3']:
        position_embedding_type = 'rope_gptj'

    config = {
        'architecture': 'ChatGLMForCausalLM',
        'dtype': args.dtype,
        'logits_dtype': args.logits_dtype,
        'num_hidden_layers': hf_config.num_layers,
        'num_attention_heads': hf_config.num_attention_heads,
        'num_key_value_heads': hf_config.num_kv_heads,
        'hidden_size': hf_config.hidden_size,
        'intermediate_size': hf_config.ffn_hidden_size,
        'norm_epsilon': hf_config.layernorm_epsilon,
        'vocab_size': hf_config.vocab_size,
        'position_embedding_type': position_embedding_type,
        'max_position_embeddings': hf_config.max_position_embeddings,
        'hidden_act': hf_config.hidden_act,
        'use_parallel_embedding': args.use_parallel_embedding,
        'embedding_sharding_dim': args.embedding_sharding_dim,
        'share_embedding_table': args.use_embedding_sharing,
        'quantization': {
            'quant_algo': None,
            'kv_cache_quant_algo': None,
        },
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'pp_size': args.pp_size,
        },
        'chatglm_version': chatglm_version,
        'add_bias_linear': hf_config.add_bias_linear,
        'add_qkv_bias': hf_config.add_qkv_bias,
        'apply_query_key_layer_scaling': False,
        'apply_residual_connection_post_layernorm':
        hf_config.apply_residual_connection_post_layernorm,
        'rmsnorm': hf_config.rmsnorm,
        'rope_ratio': hf_config.rope_ratio,
    }

    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            config['quantization']['quant_algo'] = QuantAlgo.W8A16
        elif args.weight_only_precision == 'int4':
            config['quantization']['quant_algo'] = QuantAlgo.W4A16
    elif args.smoothquant:
        if args.per_channel:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
        else:
            if args.per_token:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
            else:
                config['quantization'][
                    'quant_algo'] = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN

    if args.int8_kv_cache:
        config['quantization']['kv_cache_quant_algo'] = QuantAlgo.INT8

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    if args.use_weight_only:
        if args.weight_only_precision == 'int8':
            plugin_weight_only_quant_type = torch.int8
        elif args.weight_only_precision == 'int4':
            plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None

    hf_model = AutoModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype='auto' if chatglm_version != 'glm' else getattr(
            torch, args.dtype),
        device_map='auto' if chatglm_version != 'glm' else 'cuda')

    act_range = {}
    # smoother for query_key_value.dense and mlp.proj
    model_smoother = {}
    if args.smoothquant is not None or args.int8_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
        )
        dataset = load_calib_dataset(args.calib_dataset)
        act_range = capture_activation_range(hf_model,
                                             tokenizer,
                                             dataset,
                                             num_samples=64)
        if args.smoothquant is not None:
            smooth_chatglm_model(hf_model, act_range, args.smoothquant,
                                 model_smoother)

    def covert_and_save(rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          pp_size=args.pp_size)

        weights = convert_hf_chatglm(
            hf_model,
            hf_config,
            chatglm_version,
            mapping,
            dtype=args.dtype,
            use_parallel_embedding=args.use_parallel_embedding,
            sharding_dim=args.embedding_sharding_dim,
            share_embedding_table=args.use_embedding_sharing,
            use_weight_only=args.use_weight_only,
            plugin_weight_only_quant_type=plugin_weight_only_quant_type,
            use_smooth_quant=args.smoothquant is not None,
            per_channel=args.per_channel,
            per_token=args.per_token,
            int8_kv_cache=args.int8_kv_cache,
            act_range=act_range,
            smoother=model_smoother,
        )

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
