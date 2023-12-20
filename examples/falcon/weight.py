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
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode


def split(weight: np.ndarray, tp_size: int, rank: int = 0, dim: int = 0):
    if tp_size == 1:
        return weight
    elif weight.ndim == 1:
        return np.ascontiguousarray(np.split(weight, tp_size)[rank].copy())
    return np.ascontiguousarray(
        np.split(weight, tp_size, axis=dim)[rank].copy())


def reorder_qkv_weight_or_bias(weight: np.ndarray,
                               head_dim: int,
                               num_heads: int,
                               num_kv_heads: Optional[int] = None,
                               tp_size: int = 1,
                               is_bias: bool = False):
    """ Reorder the qkv weight for TRT-LLM use.

    The shape of the fused QKV weights in HF is different from the shape that
    TRT-LLM requires. In particular, the weight of HF consists of interleaved
    q, k, v head weights, while that of TRT-LLM is contiguous.
        HF     : [q1, k1, v1, ..., qh, kh, vh]
        TRT-LLM: [q1, ..., qh, k1, ..., kh, v1, vh]
    where qi, vi, ki are weight vectors corresponding to attention head i.
    It's similar to multi/grouped query attention cases.

    We reorder and split the weight of an attention layer to fit into TRT-LLM.
    The reordered weight and bias will be
        weight: (T, Qh * D + 2 * KVh * D, H)
        bias  : (T, Qh * D + 2 * KVh * D)
    where T=tp_size, Qh=local_num_q_heads, KVh=local_num_kv_heads, D=head_dim,
    H=hidden_dim. In the multi/grouped query attention, the number of K/V
    attention heads are less than that of Q attention, so that K/V attention
    heads may be shared across different ranks if necessary.

    For tensor parallelism, we use the first dimension to select the
    corresponding weights.
    """

    # Query types and expected kv heads.
    #  - Conventional MHA: num_heads = num_kv_heads
    #  - Multi-Query Attention: num_kv_heads = 1
    #  - Grouped-Query Attention: num_heads % num_kv_heads = 0
    num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    assert num_heads % num_kv_heads == 0, \
        f'num_heads({num_heads}) must be divisible by '\
        f'num_kv_heads({num_kv_heads})).'

    # The number of attention heads per group: N q head + 1 k head + 1 v head.
    num_group_heads = num_heads // num_kv_heads + 2
    assert weight.shape[0] == num_kv_heads * num_group_heads * head_dim, \
        f'{weight.shape[0]} != {num_kv_heads} * {num_group_heads} * {head_dim}'

    qkv_in = num_heads * head_dim if not is_bias else 1

    # Split Q/K/V weights
    weight = weight.reshape(num_kv_heads, num_heads // num_kv_heads + 2,
                            head_dim, qkv_in)
    q_w = weight[:, :-2, ...]  # (nKV, num_heads // nKV, head_dim, qkv_in)
    k_w = weight[:, -2:-1, ...]  # (nKV, 1, head_dim, qkv_in)
    v_w = weight[:, -1:, ...]  # (nKV, 1, head_dim, qkv_in)

    if num_kv_heads < num_heads and num_kv_heads < tp_size:
        # Duplicate K/V heads to make sure that each rank has at least one
        # K/V heads. For instance, num_heads=8, num_kv_heads=2, tp_size=4,
        # we will make the qkv weight as below.
        #   Orig: [q0 q1 q2 q3 k0 v0 q4 q5 q6 q7 k1 v0 v1]
        #   >>>>  [[q0 q1 k0 v0], [q2 q3 k0 v0], [q4 q5 k1 v1], [q6 q7 k1 v1]]
        assert tp_size % num_kv_heads == 0
        num_dups = tp_size // num_kv_heads

        # k_w and v_w have the same shape.
        new_shape = (num_kv_heads, num_dups) + k_w.shape[2:]
        k_w = np.broadcast_to(k_w, shape=new_shape)
        v_w = np.broadcast_to(v_w, shape=new_shape)

        # Update the number of kv heads.
        num_kv_heads = tp_size

    reordered = np.concatenate(
        [
            q_w.reshape(tp_size, num_heads // tp_size, head_dim, qkv_in),
            k_w.reshape(tp_size, num_kv_heads // tp_size, head_dim, qkv_in),
            v_w.reshape(tp_size, num_kv_heads // tp_size, head_dim, qkv_in),
        ],
        axis=1,
    )

    qkv_out = (num_heads + 2 * num_kv_heads) // tp_size * head_dim
    return reordered.reshape((tp_size, qkv_out, -1))


def split_qkv_weight(trtllm_falcon: tensorrt_llm.models.FalconModel,
                     weight: np.ndarray,
                     tp_size: int,
                     rank: int,
                     is_bias: bool,
                     num_kv_heads: Optional[int] = None):
    """ Splits the QKV matrix according to tensor parallelism """
    n_heads = trtllm_falcon.num_heads
    hidden_size = trtllm_falcon.hidden_size
    head_dim = hidden_size // n_heads
    weight = reorder_qkv_weight_or_bias(weight,
                                        head_dim=head_dim,
                                        num_heads=n_heads,
                                        num_kv_heads=num_kv_heads,
                                        tp_size=tp_size,
                                        is_bias=is_bias)

    # Copy a sliced tensor to prevent memory leak. A sliced tensor shares the
    # memory buffer of the original tensor. So, returning without copying makes
    # the buffer of a loaded "qkv" be referenced, resulting GC can't release
    # those weights until the whole process ends.
    if not is_bias:
        return np.ascontiguousarray(weight[rank, ...].copy())
    else:
        return weight[rank, ...].ravel().copy()


def split_matrix(weight: np.ndarray, tp_size: int, rank: int, dim: int):
    return np.ascontiguousarray(split(weight, tp_size, rank, dim=dim))


def get_weight(params: Dict, prefix: str, dtype: torch.dtype):
    if f'{prefix}.weight' not in params:
        return None
    param = params[f'{prefix}.weight'].to(dtype).detach().cpu()
    return tensorrt_llm._utils.torch_to_numpy(param)


def get_bias(params: Dict, prefix: str, dtype: torch.dtype):
    if f'{prefix}.bias' not in params:
        return None
    param = params[f'{prefix}.bias'].to(dtype).detach().cpu()
    return tensorrt_llm._utils.torch_to_numpy(param)


def get_weight_and_bias(params: Dict, prefix: str, dtype: torch.dtype):
    return get_weight(params, prefix, dtype), get_bias(params, prefix, dtype)


def load_from_hf_falcon(trtllm_falcon: tensorrt_llm.models.FalconForCausalLM,
                        hf_falcon,
                        mapping=Mapping(),
                        dtype: Union[str, torch.dtype] = torch.float32):
    logger.info('Loading weights from HF Falcon...')
    tik = time.time()

    model_params = dict(hf_falcon.named_parameters())
    if isinstance(dtype, str):
        dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)
    num_kv_heads = trtllm_falcon.num_kv_heads

    layers_range = trtllm_falcon.mapping.pp_layers(trtllm_falcon.num_layers)
    for i in layers_range:
        prefix = f'transformer.h.{i}'
        layer = trtllm_falcon.layers[i - layers_range[0]]
        qkv_weight, qkv_bias = get_weight_and_bias(
            model_params, f'{prefix}.self_attention.query_key_value', dtype)
        qkv_w = split_qkv_weight(trtllm_falcon,
                                 qkv_weight,
                                 mapping.tp_size,
                                 mapping.tp_rank,
                                 is_bias=False,
                                 num_kv_heads=num_kv_heads)
        layer.attention.qkv.weight.value = qkv_w
        if qkv_bias is not None:
            layer.attention.qkv.bias.value = split_qkv_weight(
                trtllm_falcon,
                qkv_bias,
                mapping.tp_size,
                mapping.tp_rank,
                is_bias=True,
                num_kv_heads=num_kv_heads)

        logger.debug(f'Layer {i}: Loading attention Dense weights...')
        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            model_params, f'{prefix}.self_attention.dense', dtype)
        layer.attention.dense.weight.value = split_matrix(attn_dense_weight,
                                                          mapping.tp_size,
                                                          mapping.tp_rank,
                                                          dim=1)
        if attn_dense_bias is not None:
            layer.attention.dense.bias.value = attn_dense_bias

        logger.debug(f'Layer {i}: Loading MLP FC weights...')
        mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_h_to_4h', dtype)
        layer.mlp.fc.weight.value = split_matrix(mlp_fc_weight,
                                                 mapping.tp_size,
                                                 mapping.tp_rank,
                                                 dim=0)
        if mlp_fc_bias is not None:
            layer.mlp.fc.bias.value = split_matrix(mlp_fc_bias,
                                                   mapping.tp_size,
                                                   mapping.tp_rank,
                                                   dim=0)

        logger.debug(f'Layer {i}: Loading MLP Proj weights...')
        mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
            model_params, f'{prefix}.mlp.dense_4h_to_h', dtype)
        layer.mlp.proj.weight.value = split_matrix(mlp_proj_weight,
                                                   mapping.tp_size,
                                                   mapping.tp_rank,
                                                   dim=1)
        if mlp_proj_bias is not None:
            layer.mlp.proj.bias.value = mlp_proj_bias

        if trtllm_falcon.new_decoder_architecture:
            input_ln_weight, input_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.ln_attn', dtype)
            layer.input_layernorm.weight.value = input_ln_weight
            if input_ln_bias is not None:
                layer.input_layernorm.bias.value = input_ln_bias

            mlp_ln_weight, mlp_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.ln_mlp', dtype)
            layer.mlp_layernorm.weight.value = mlp_ln_weight
            if mlp_ln_bias is not None:
                layer.mlp_layernorm.bias.value = mlp_ln_bias
        else:
            # Layer norms do not use tensor parallelism
            logger.debug(f'Layer {i}: Loading normalization weights...')
            input_ln_weight, input_ln_bias = get_weight_and_bias(
                model_params, f'{prefix}.input_layernorm', dtype)
            layer.input_layernorm.weight.value = input_ln_weight
            if input_ln_bias is not None:
                layer.input_layernorm.bias.value = input_ln_bias

            if not trtllm_falcon.parallel_attention:
                post_ln_weight, post_ln_bias = get_weight_and_bias(
                    model_params, f'{prefix}.post_attention_layernorm', dtype)
                if post_ln_weight is not None:
                    layer.post_layernorm.weight.value = post_ln_weight
                if post_ln_bias is not None:
                    layer.post_layernorm.bias.value = post_ln_bias

    embed_w = get_weight(model_params, 'transformer.word_embeddings', dtype)
    if mapping.is_first_pp_rank():
        trtllm_falcon.embedding.weight.value = embed_w.copy()
    if mapping.is_last_pp_rank():
        trtllm_falcon.lm_head.weight.value = split_matrix(embed_w,
                                                          mapping.tp_size,
                                                          mapping.tp_rank,
                                                          dim=0)

        ln_f_w, ln_f_b = get_weight_and_bias(model_params, 'transformer.ln_f',
                                             dtype)
        trtllm_falcon.ln_f.weight.value = ln_f_w
        if ln_f_b is not None:
            trtllm_falcon.ln_f.bias.value = ln_f_b

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')


def load_state_dict(file_path: Path,
                    dtype: torch.dtype) -> Dict[str, np.ndarray]:
    """ Load weights from model file

    `safetensors` or `pytorch binary` is supported

    # Args.
        file_path: model file path, ends with .bin or .safetensors.
        dtype: torch.dtype, data type.
    # Returns.
        Dict[str, torch.Tensor]
    """

    state_dict = {}
    if file_path.suffix == '.safetensors':
        # load from safetensors file
        from safetensors import safe_open
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for name in f.keys():
                param = f.get_tensor(name).to(dtype)
                state_dict[name] = tensorrt_llm._utils.torch_to_numpy(param)
    elif file_path.suffix == '.bin':
        # load from pytorch bin file
        state_dict = torch.load(file_path, map_location='cpu')
        for name in state_dict:
            param = state_dict[name].to(dtype)
            state_dict[name] = tensorrt_llm._utils.torch_to_numpy(param)
    else:
        raise NotImplementedError(
            f'Support .safetensors or .bin files, but got {str(file_path)}')
    return state_dict


def retrieved_layer_index_from_name(name: str) -> Optional[int]:
    res = re.search(r'\d+', name)
    return int(res.group()) if res is not None else res


def iterate_shard_files(model_dir: Path, rank: int):
    import tqdm

    shard_files = list(model_dir.glob('*.bin')) + list(
        model_dir.glob('*.safetensors'))
    desc = f'Rank [{rank}] Loading weights'
    for shard_file in tqdm.tqdm(shard_files, desc=desc, position=rank):
        yield shard_file


def load_from_hf_checkpoint(
        trtllm_falcon: tensorrt_llm.models.FalconForCausalLM,
        model_dir: Union[str, Path],
        mapping=Mapping(),
        dtype: Union[str, torch.dtype] = torch.float32,
):
    logger.info('Loading weights from HF Falcon...')
    tik = time.time()

    model_dir = Path(model_dir)
    if isinstance(dtype, str):
        dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)

    def is_bias(_name):
        return 'bias' in _name

    layers_range = trtllm_falcon.mapping.pp_layers(trtllm_falcon.num_layers)
    for model_file in iterate_shard_files(model_dir, mapping.tp_rank):
        logger.debug(f'Loading file {str(model_file)}...')
        state_dict = load_state_dict(model_file, dtype)
        for name, param in state_dict.items():
            logger.debug(f'Converting weight {name}...')
            i = retrieved_layer_index_from_name(name)
            if i is None:
                layer = None
            else:
                if i not in layers_range:
                    continue
                layer = trtllm_falcon.layers[i - layers_range[0]]

            if 'self_attention.query_key_value' in name:
                if not is_bias(name):
                    layer.attention.qkv.weight.value = split_qkv_weight(
                        trtllm_falcon,
                        param,
                        mapping.tp_size,
                        mapping.tp_rank,
                        is_bias=False,
                        num_kv_heads=trtllm_falcon.num_kv_heads)
                else:
                    layer.attention.qkv.bias.value = split_qkv_weight(
                        trtllm_falcon,
                        param,
                        mapping.tp_size,
                        mapping.tp_rank,
                        is_bias=True,
                        num_kv_heads=trtllm_falcon.num_kv_heads)
            elif 'self_attention.dense' in name:
                if not is_bias(name):
                    layer.attention.dense.weight.value = split_matrix(
                        param, mapping.tp_size, mapping.tp_rank, dim=1)
                else:
                    layer.attention.dense.bias.value = param
            elif 'mlp.dense_h_to_4h' in name:
                if not is_bias(name):
                    layer.mlp.fc.weight.value = split_matrix(param,
                                                             mapping.tp_size,
                                                             mapping.tp_rank,
                                                             dim=0)
                else:
                    layer.mlp.fc.bias.value = split_matrix(param,
                                                           mapping.tp_size,
                                                           mapping.tp_rank,
                                                           dim=0)
            elif 'mlp.dense_4h_to_h' in name:
                if not is_bias(name):
                    layer.mlp.proj.weight.value = split_matrix(param,
                                                               mapping.tp_size,
                                                               mapping.tp_rank,
                                                               dim=1)
                else:
                    layer.mlp.proj.bias.value = param
            elif 'ln_attn' in name or 'input_layernorm' in name:
                if not is_bias(name):
                    layer.input_layernorm.weight.value = param
                else:
                    layer.input_layernorm.bias.value = param
            elif 'ln_mlp' in name:
                assert layer.mlp_layernorm is not None
                if not is_bias(name):
                    layer.mlp_layernorm.weight.value = param
                else:
                    layer.mlp_layernorm.bias.value = param
            elif 'post_attention_layernorm' in name:
                assert layer.post_layernorm is not None
                if not is_bias(name):
                    layer.post_layernorm.weight.value = param
                else:
                    layer.post_layernorm.bias.value = param
            elif 'word_embeddings' in name:
                if mapping.is_first_pp_rank():
                    trtllm_falcon.embedding.weight.value = param.copy()
                if mapping.is_last_pp_rank():
                    trtllm_falcon.lm_head.weight.value = split_matrix(
                        param, mapping.tp_size, mapping.tp_rank, dim=0)
            elif 'ln_f' in name:
                if mapping.is_last_pp_rank():
                    if not is_bias(name):
                        trtllm_falcon.ln_f.weight.value = param
                    else:
                        trtllm_falcon.ln_f.bias.value = param
        del state_dict

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for Falcon model

    Returns a dictionary of scaling factors for the selected layers of the
    Falcon model.

    Args:
        model_path (str): Path to the quantized Falcon model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        Falcon model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_out' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
            'proj_act': proj_act_scale,
            'proj_weights': proj_weights_scale,
        }
    """

    if model_path is None:
        logger.warning(f"--quantized_fp8_model_path not specified. "
                       f"Initialize quantization scales automatically.")
        return get_dummy_quant_scales(num_layers)
    weight_dict = np.load(model_path)

    # yapf: disable
    scaling_factor = {
        'qkv_act': [],
        'qkv_weights': [],
        'qkv_output': [],
        'dense_act': [],
        'dense_weights': [],
        'fc_act': [],
        'fc_weights': [],
        'proj_act': [],
        'proj_weights': [],
    }

    for layer in range(num_layers):
        scaling_factor['qkv_act'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:activation_scaling_factor'].item()
            ))
        scaling_factor['qkv_weights'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:weights_scaling_factor'].item()
            ))
        if quant_mode is not None and quant_mode.has_fp8_kv_cache():
            # Not calibrarting KV cache.
            scaling_factor['qkv_output'].append(1.0)
        scaling_factor['dense_act'].append(weight_dict[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
        scaling_factor['dense_weights'].append(weight_dict[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
        scaling_factor['fc_act'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:activation_scaling_factor'].item())
        scaling_factor['fc_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:weights_scaling_factor'].item())
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
        f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor


def load_from_awq_falcon(
        tensorrt_llm_falcon: tensorrt_llm.models.FalconForCausalLM,
        quant_ckpt_path,
        mapping=Mapping(),
        dtype="float16"):
    tensorrt_llm.logger.info(
        'Loading weights from groupwise AWQ Falcon checkpoint...')
    tik = time.time()

    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm
    torch_dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype)

    if quant_ckpt_path.endswith(".npz"):
        awq_falcon = np.load(quant_ckpt_path)
        awq_prefix = "_np:"
        awq_suffix_list = [
            ":weight",
            ":weights_scaling_factor",
            ":prequant_scaling_factor",
        ]
        awq_key_list = [
            "vocab_embedding:weight",  # embedding
            "lm_head",  # lm_head
            "final_layernorm",  # ln_f
            "attention:qkv:",  # attention.qkv
            "attention:dense",  # attention.dense
            "mlp:proj",  # mlp.proj
            "mlp:fc",  # mlp.fc
            "input_layernorm",  # input_layernorm.weight
            "mlp_layernorm",  # mlp_layernorm.weight
        ]
        split_sym = ":"
        AMMO_WEIGHT_SCALING_FACTOR_COEFF = 7

        def load(key):
            v = torch.from_numpy(awq_falcon[awq_prefix + key]).to(torch_dtype)
            if "weights_scaling_factor" in key:
                v *= AMMO_WEIGHT_SCALING_FACTOR_COEFF  # For AMMO *.npz checkpoints
            return v

        group_size = load("layers:0:attention:dense:weight").numel() // load(
            "layers:0:attention:dense:weights_scaling_factor").numel()
    else:
        raise ValueError("Unsupported AWQ quantized checkpoint format")

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            tensorrt_llm.logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            raise ValueError("Invalid TP size")
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def AWQ_quantize_pack_preprocess(weight, scale):
        weight /= scale.repeat_interleave(group_size, dim=0)
        qweight_int8 = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        int4_weight = preprocessor(packer(qweight_int8.cpu()), torch.quint4x2)
        return int4_weight.view(torch.int8)

    def process_and_assign_weight(mOp, v, tp_dim=0):
        weight = v[0].T.contiguous()
        [k, n] = weight.shape
        weight = torch_split(weight, tp_dim)
        amax = v[1].reshape((n, k // group_size)).T.contiguous()
        amax = torch_split(amax, tp_dim)
        pre_quant_scale = v[2].reshape((1, k))
        if tp_dim == 0:
            pre_quant_scale = torch_split(pre_quant_scale, 1)
        scale = amax / 8.0
        mOp.qweight.value = AWQ_quantize_pack_preprocess(weight, scale)
        mOp.scale.value = scale.to(torch_dtype)
        mOp.pre_quant_scale.value = pre_quant_scale.to(torch_dtype)

    def get_scale(weight):
        [k, n] = weight.shape
        weight_t = weight.T.contiguous()
        weight_t = weight_t.reshape(n, k // group_size, group_size)
        weight_t = torch.abs(weight_t.reshape(-1, group_size))
        amax, idx = weight_t.max(1)
        amax = amax.reshape(n, k // group_size).T.contiguous()
        scale = amax / 8
        return scale

    def process_and_assign_qkv_weight(prefix, mOp):
        q_weight = load(prefix + "q" + awq_suffix_list[0])
        k_weight = load(prefix + "k" + awq_suffix_list[0])
        v_weight = load(prefix + "v" + awq_suffix_list[0])
        dim_k = q_weight.shape[0]
        q_weight = torch_split(q_weight, 1)
        k_weight = torch_split(k_weight, 1)
        v_weight = torch_split(v_weight, 1)
        qkv_pre_quant_scale = load(prefix + "q" + awq_suffix_list[2]).reshape(
            (1, dim_k))
        qkv_weights = torch.cat((q_weight, k_weight, v_weight), dim=1)
        qkv_scale = get_scale(qkv_weights)

        mOp.pre_quant_scale.value = qkv_pre_quant_scale.to(torch_dtype)
        mOp.qweight.value = AWQ_quantize_pack_preprocess(qkv_weights, qkv_scale)
        mOp.scale.value = qkv_scale.to(torch_dtype)

    # Load weights from AWQ checkpoint into TRT-LLM module
    # 1. embedding
    v = load(awq_key_list[0])
    # TRT-LLM requires vocab_size to be multiple of 64 for successful GEMM
    if v.shape[0] % 64 != 0:
        v = torch.nn.functional.pad(v, [0, 0, 0, 64 - v.shape[0] % 64])
    if mapping.is_first_pp_rank():
        tensorrt_llm_falcon.embedding.weight.value = v.to(torch_dtype)

    # 2. lm_head
    v = [load(awq_key_list[1] + suf) for suf in awq_suffix_list]
    if v[0].shape[0] % 64 != 0:
        v[0] = torch.nn.functional.pad(v[0], [0, 0, 0, 64 - v[0].shape[0] % 64])
        v[1] = torch.nn.functional.pad(v[1], [0, 0, 0, 64 - v[1].shape[0] % 64],
                                       value=1)
    if mapping.is_last_pp_rank():
        process_and_assign_weight(tensorrt_llm_falcon.lm_head, v, 1)

    # 3. ln_f
    v_weight = load(awq_key_list[2] + split_sym + "weight")
    v_bias = load(awq_key_list[2] + split_sym + "bias")
    if mapping.is_last_pp_rank():
        tensorrt_llm_falcon.ln_f.weight.value = v_weight.to(torch_dtype)
        tensorrt_llm_falcon.ln_f.bias.value = v_bias.to(torch_dtype)

    # 4. Weights inside each layer
    num_hidden_layers = tensorrt_llm_falcon.num_layers
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))

    for l in layers_range:
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "layers" + split_sym + str(layer_idx) + split_sym
        tensorrt_llm.logger.info(f'Process weights in layer: {layer_idx}')
        layer = tensorrt_llm_falcon.layers[layer_idx]

        # 4.1 attention.qkv
        process_and_assign_qkv_weight(prefix + awq_key_list[3],
                                      layer.attention.qkv)

        # 4.2 attention.dense
        v = [load(prefix + awq_key_list[4] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.attention.dense, v, 0)

        # 4.3 mlp.proj
        v = [load(prefix + awq_key_list[5] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.proj, v, 0)

        # 4.4 mlp.fc
        v = [load(prefix + awq_key_list[6] + suf) for suf in awq_suffix_list]
        process_and_assign_weight(layer.mlp.fc, v, 1)

        # 4.5 input_layernorm
        v = load(prefix + awq_key_list[7] + split_sym + "weight")
        layer.input_layernorm.weight.value = v.to(torch_dtype)
        v = load(prefix + awq_key_list[7] + split_sym + "bias")
        layer.input_layernorm.bias.value = v.to(torch_dtype)

        # 4.6 mlp_layernorm
        v = load(prefix + awq_key_list[8] + split_sym + "weight")
        layer.mlp_layernorm.weight.value = v.to(torch_dtype)
        v = load(prefix + awq_key_list[8] + split_sym + "bias")
        layer.mlp_layernorm.bias.value = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Elapsed time: {t}')
