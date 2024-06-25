from pathlib import Path

import torch

from tensorrt_llm import logger
from tensorrt_llm._utils import str_dtype_to_torch
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.convert_utils import split


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8,
                           postfix='weight'):
    results = {}
    if use_weight_only:
        v = weight.t().contiguous().cpu()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[prefix + postfix] = processed_torch_weights
        results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.contiguous()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def load_medusa_hf(medusa_path: str,
                   num_medusa_heads: int,
                   num_medusa_layers: int,
                   mapping=Mapping(),
                   dtype='float32',
                   use_weight_only=False,
                   plugin_weight_only_quant_type=None):
    logger.info("Loading Medusa heads' weights ...")
    is_ckpt_safetensors = False

    ckpt_file = Path(medusa_path) / "medusa_lm_head.pt"
    if not ckpt_file.exists():
        ckpt_file = Path(medusa_path) / "medusa_lm_head.safetensors"
        is_ckpt_safetensors = True

    if is_ckpt_safetensors:
        logger.info("Safetensors Found ...")
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_file)
    else:
        state_dict = torch.load(ckpt_file, map_location="cpu")

    torch_dtype = str_dtype_to_torch(dtype)
    weights = {}

    for h in range(num_medusa_heads):
        for l in range(num_medusa_layers):
            w = state_dict[f"{h}.{l}.linear.weight"].clone().to(torch_dtype)

            split_v = split(w, mapping.tp_size, mapping.tp_rank)
            weights.update(
                get_tllm_linear_weight(
                    split_v, f'medusa_heads.{h}.medusa_layers.{l}.linear.',
                    None, use_weight_only, plugin_weight_only_quant_type))

            b = state_dict[f"{h}.{l}.linear.bias"].clone().to(torch_dtype)

            weights['medusa_heads.{}.medusa_layers.{}.linear.bias'.format(
                h, l)] = split(b, mapping.tp_size, mapping.tp_rank)

        lm = state_dict[f"{h}.{num_medusa_layers}.weight"].clone().to(
            torch_dtype)  # LM Head

        weights['medusa_heads.{}.lm_head.weight'.format(h)] = split(
            lm, mapping.tp_size, mapping.tp_rank)

    return weights
