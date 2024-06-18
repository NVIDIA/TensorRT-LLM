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

import time
from typing import List

import torch
from tqdm import tqdm

from ..._utils import str_dtype_to_torch
from ...logger import logger
from ...mapping import Mapping
from .utils import get_qwen_key_list


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return torch.chunk(v, tp_size)[idx].contiguous()
    else:
        return torch.chunk(v, tp_size, dim=dim)[idx].contiguous()


def load_from_gptq_qwen(
        model,
        qwen_type,
        num_hidden_layers=None,
        mapping=Mapping(),
        dtype="float16",
):
    logger.info("loading weights from groupwise GPTQ QWen safetensors...")
    weights = {}
    tik = time.time()

    model_params = {k: v for k, v in model.state_dict().items()}
    torch.cuda.empty_cache()
    assert qwen_type in [
        'qwen', 'qwen2'
    ], "Currently, only qwen and qwen2 support gptq. qwen2_moe is not supported yet."
    layer_prefix = "transformer.h." if qwen_type == 'qwen' else "model.layers."
    key_list = get_qwen_key_list(qwen_type)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            logger.error(
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
                                           torch.quint4x2,
                                           torch.float16).view(torch.float16)
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
    v = model_params[key_list[7] + '.weight']
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)

    # 2. ln_f
    v = model_params[key_list[8] + '.weight']
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
    suffixs = [".qweight", ".qzeros", ".scales"]

    for l in tqdm(layers_range, desc="loading weight in each layer..."):
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = layer_prefix + str(layer_idx) + "."
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # 4.1 attention.qkv
        qkv_weight_list = []
        if qwen_type == 'qwen':
            for suf in suffixs:
                qkv_part = model_params[prefix + key_list[0] + suf]
                q_emb = qkv_part.shape[1] // 3
                model_emb = qkv_part.shape[0]
                qkv_part = qkv_part.reshape(model_emb, 3, q_emb)
                qkv_part = torch_split(qkv_part, 2)
                qkv_part = qkv_part.reshape(model_emb,
                                            3 * (q_emb // mapping.tp_size))
                qkv_weight_list.append(qkv_part)
        else:
            for suf in suffixs:
                qkv_list = []
                for comp in ["q_proj", "k_proj", "v_proj"]:
                    comp_part = model_params[prefix + key_list[0] + comp + suf]
                    comp_part = torch_split(comp_part, 1)
                    qkv_list.append(comp_part)
                qkv_weight_list.append(torch.cat(qkv_list, dim=1))
        weights.update(
            process_and_assign_weight(qkv_weight_list,
                                      f'{tllm_prex}.attention.qkv'))
        # 4.2 attention.bias
        suf = ".bias"
        if qwen_type == 'qwen':
            qkv_bias = model_params[prefix + key_list[0] +
                                    suf].to(torch_dtype).cpu().contiguous()
            q_emb = qkv_bias.shape[0] // 3
            qkv_bias = qkv_bias.reshape(3, q_emb)
            split_v = split(qkv_bias, mapping.tp_size, mapping.rank, dim=1)
            qkv_bias = split_v.reshape(3 * (q_emb // mapping.tp_size))
        else:
            qkv_bias_list = []
            for comp in ["q_proj", "k_proj", "v_proj"]:
                comp_part = model_params[prefix + key_list[0] + comp + suf].to(
                    torch_dtype).cpu().contiguous()
                comp_part = torch_split(comp_part, dim=0)
                qkv_bias_list.append(comp_part)
            qkv_bias = torch.cat(qkv_bias_list, dim=0)
        weights[tllm_prex + ".attention.qkv.bias"] = qkv_bias
        # 4.3 attention.dense
        qkv_dense_list = []
        for suf in suffixs:
            qkv_dense_part = model_params[prefix + key_list[1] + suf]
            qkv_dense_list.append(qkv_dense_part)
        weights.update(
            process_and_assign_weight(qkv_dense_list,
                                      f'{tllm_prex}.attention.dense',
                                      tp_dim=0))
        # 4.4 mlp.gate
        mlp_gate_list = []
        for suf in suffixs:
            mlp_gate_part = model_params[prefix + key_list[2] + suf]
            mlp_gate_list.append(mlp_gate_part)
        weights.update(
            process_and_assign_weight(mlp_gate_list,
                                      f'{tllm_prex}.mlp.gate',
                                      tp_dim=1))
        # 4.5 mlp.fc
        mlp_fc_list = []
        for suf in suffixs:
            mlp_fc_part = model_params[prefix + key_list[3] + suf]
            mlp_fc_list.append(mlp_fc_part)
        weights.update(
            process_and_assign_weight(mlp_fc_list,
                                      f'{tllm_prex}.mlp.fc',
                                      tp_dim=1))
        # 4.6 mlp.proj
        mlp_proj_list = []
        for suf in suffixs:
            mlp_proj_part = model_params[prefix + key_list[4] + suf]
            mlp_proj_list.append(mlp_proj_part)
        weights.update(
            process_and_assign_weight(mlp_proj_list,
                                      f'{tllm_prex}.mlp.proj',
                                      tp_dim=0))
        # 4.7 input_layernorm
        v = model_params[prefix + key_list[5] + '.weight']
        weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)
        # 4.8 post_layernorm
        v = model_params[prefix + key_list[6] + '.weight']
        weights[f'{tllm_prex}.post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"weights loaded. total time: {t}")

    return weights
