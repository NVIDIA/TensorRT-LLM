from typing import Dict

import torch

from ..quantization import QuantAlgo


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


def weight_only_quantize(weight: torch.Tensor,
                         quant_algo: str,
                         plugin: bool = True):
    assert quant_algo in [QuantAlgo.W4A16, QuantAlgo.W8A16
                          ], f'unsupported quant algo: {quant_algo}'
    if quant_algo == QuantAlgo.W4A16:
        assert plugin, 'W4A16 is only supported with plugin'
    if weight.dim() > 2:
        v = weight.transpose(-1, -2)
    else:
        v = weight.t()
    t = torch.quint4x2 if quant_algo == QuantAlgo.W4A16 else torch.int8
    processed_torch_weights, torch_weight_scales = \
        torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
        v.contiguous(), t)
    if plugin:
        return processed_torch_weights, torch_weight_scales
    else:
        return v, torch_weight_scales


def weight_only_quantize_dict(weights: Dict[str, torch.Tensor],
                              quant_algo: str,
                              quant_weights=[
                                  'qkv.weight', 'dense.weight', 'fc.weight',
                                  'proj.weight', 'gate.weight'
                              ],
                              plugin: bool = True):
    if quant_algo not in [QuantAlgo.W4A16, QuantAlgo.W8A16]:
        return weights
    for name in list(weights):
        if any([_name in name for _name in quant_weights
                ]) and weights[name].dtype != torch.int8:
            quant_weight, quant_scale = weight_only_quantize(
                weight=weights[name], quant_algo=quant_algo, plugin=plugin)
            weights[name] = quant_weight
            weights[name.replace('.weight', '.per_channel_scale')] = quant_scale
    return weights
