import math
import typing
from typing import Union

import numpy as np
import torch  # pytype: disable=import-error

from tensorrt_llm._utils import str_dtype_to_torch


def split(v: Union[np.ndarray, torch.Tensor],
          tp_size: int,
          tp_rank: int,
          dim=0):
    if tp_size == 1:
        if isinstance(v, np.ndarray):
            return np.ascontiguousarray(v.copy())
        else:
            return v.clone().detach()
    assert len(v.shape) > 1 or dim == 0
    if isinstance(v, np.ndarray):
        return np.ascontiguousarray(
            np.split(v, tp_size, axis=dim)[tp_rank].copy())
    else:
        assert v.shape[dim] % tp_size == 0, \
            'Unable to split: shape={v.shape} (dim={dim}) tp_size={tp_size}.'
        split_size = v.shape[dim] // tp_size
        return v.split(split_size, dim=dim)[tp_rank].clone().detach()


def reshape(v: torch.Tensor, shape=None):
    if shape is None:
        return v.contiguous()
    else:
        return v.reshape(shape).contiguous()


def fuse_qkv_one_layer(params, attn_module_name, trtllm_layer_name, tp_size,
                       tp_rank, model_type, weight_shape, bias_shape):

    qkv_module_names = get_qkv_module_name(model_type)

    weight = {}

    # fuse weights of q, k, v
    q_w = params[f'{attn_module_name}.{qkv_module_names["q"]}.weight']
    k_w = params[f'{attn_module_name}.{qkv_module_names["k"]}.weight']
    v_w = params[f'{attn_module_name}.{qkv_module_names["v"]}.weight']

    # fuse qkv weight
    shape = q_w.shape  # (do, din)
    qkv_w = torch.cat([q_w, k_w, v_w],
                      dim=0).reshape([3, shape[0], shape[1]])  # (3, do, din)
    qkv_w = split(qkv_w, tp_size, tp_rank, dim=1)
    weight[f'{trtllm_layer_name}.qkv.weight'] = reshape(qkv_w,
                                                        shape=weight_shape)

    # fuse qkv biases if present
    if f'{attn_module_name}.{qkv_module_names["q"]}.bias' in params.keys(
    ) and params[f'{attn_module_name}.{qkv_module_names["q"]}.bias'] is not None:
        q_b = params[f'{attn_module_name}.{qkv_module_names["q"]}.bias']
        k_b = params[f'{attn_module_name}.{qkv_module_names["k"]}.bias']
        v_b = params[f'{attn_module_name}.{qkv_module_names["v"]}.bias']
        shape = q_b.shape[0]  # (do,)
        qkv_b = torch.cat([q_b, k_b, v_b], dim=0).reshape([3, shape])  # (3, do)
        qkv_b = split(qkv_b, tp_size, tp_rank, dim=1)
        weight[f'{trtllm_layer_name}.qkv.bias'] = reshape(qkv_b,
                                                          shape=bias_shape)
    return weight


def get_qkv_module_name(model_type):
    if model_type in ["t5", "blip2"]:
        q = "q"
        k = "k"
        v = "v"
    elif model_type == "bart" or model_type == "nmt" or model_type == "language_adapter":
        q = "q_proj"
        k = "k_proj"
        v = "v_proj"
    elif model_type == "pix2struct":
        q = "query"
        k = "key"
        v = "value"
    return {"q": q, "k": k, "v": v}


def convert_weight_to_dtype(params: typing.Dict[str, torch.Tensor],
                            dtype: typing.Optional[np.dtype] = None):
    if dtype is not None:
        assert isinstance(dtype,
                          str), f"dtype must be str, but get type {type(dtype)}"
        for name in params.keys():
            params[name] = params[name].to(str_dtype_to_torch(dtype))


def fairseq_sin_pos_embedding(num_embeddings: int, embedding_dim: int):
    '''
    generate fairseq specific sinusoidal position embedding [sin, sin, ... cos, cos...]
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py
    '''
    padding_offset = 2
    half_dim = embedding_dim // 2.0
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float16) * -emb)
    emb = torch.arange(num_embeddings + padding_offset,
                       dtype=torch.float16).unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                    dim=1).view(num_embeddings + padding_offset, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat(
            [emb, torch.zeros(num_embeddings + padding_offset, 1)],
            dim=1,
            dtype=torch.float16)
    '''
    remove first 2 column to match position_id setup difference between fairseq & trt
    fairseq position_id starts with 2, ex: [2, 3, 4 ..]
    trt position_id starts with 0 [0, 1, 2]
    '''
    return emb[padding_offset:, :]
