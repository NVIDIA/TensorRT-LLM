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
import json
import math
import os
import re
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Dict, Optional, Union

import h5py
import numpy
import numpy as np
import sentencepiece as sp
import torch

import tensorrt_llm
import tensorrt_llm.models.modeling_utils as tllm_utils
from tensorrt_llm._utils import (np_bfloat16, numpy_to_torch,
                                 str_dtype_to_torch, torch_to_numpy)
from tensorrt_llm.logger import logger
from tensorrt_llm.models.convert_utils import load_calib_dataset
from tensorrt_llm.models.gemma.config import GemmaConfig
from tensorrt_llm.models.gemma.smoothquant import (capture_activation_range,
                                                   convert_hf_model,
                                                   create_model_from_config,
                                                   smooth_model)
from tensorrt_llm.models.gemma.utils import params as gemma_params
from tensorrt_llm.models.gemma.weight import (dummy_weights_awq,
                                              load_from_fp8_gemma,
                                              quantize_fp8_weights)
from tensorrt_llm.quantization.mode import QuantAlgo

Weights = Dict[str, torch.Tensor]
Flattened = Dict[str, np.ndarray]

if TYPE_CHECKING:
    import transformers


class JAXParser:

    def load_parameters(self,
                        checkpoint_path: Path,
                        load_model_on_cpu: bool = False) -> Weights:
        checkpoint_path = checkpoint_path.absolute()
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        return gemma_params.nest_params(
            gemma_params.param_remapper(
                gemma_params.load_params(checkpoint_path)))

    def embedding_weights(self, ckpt_params) -> np.ndarray:
        return ckpt_params["transformer"]["embedder"]["input_embedding"]

    def get_config(self, checkpoint_path, ckpt_params,
                   num_embed) -> SimpleNamespace:
        from tensorrt_llm.models.gemma.utils.transformer import \
            TransformerConfig

        return TransformerConfig.from_params(ckpt_params, num_embed=num_embed)

    def rename_to_trt_llm(self, name: str) -> Optional[str]:
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix, name = name.split(".", maxsplit=1)
        assert prefix == "transformer"
        sub_patterns = (
            (r"embedder.input_embedding", r"vocab_embedding.weight"),
            (r"layer_(\d+).pre_attention_norm.scale",
             r"layers.\1.input_layernorm.weight"),
            (r"layer_(\d+).attn.q_einsum.w", r"layers.\1.attention.qkv.weight"),
            (r"layer_(\d+).attn.kv_einsum.w",
             None),  # drop as kv will be concatenated with q
            (r"layer_(\d+).attn.qkv_einsum.w",
             r"layers.\1.attention.qkv.weight"),
            (r"layer_(\d+).attn.attn_vec_einsum.w",
             r"layers.\1.attention.dense.weight"),
            (r"layer_(\d+).mlp.gating_einsum", r"layers.\1.mlp.fc.weight"),
            (r"layer_(\d+).mlp.linear", r"layers.\1.mlp.proj.weight"),
            (r"layer_(\d+).pre_ffw_norm.scale",
             r"layers.\1.post_layernorm.weight"),
            (r"final_norm.scale", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {prefix}.{name}")

    def flatten_params(self, params) -> Flattened:
        import flax.traverse_util

        new_params = flax.traverse_util.flatten_dict(params, sep=".")
        # if the dtype is bfloat16, cast to float32
        for k in new_params:
            if new_params[k].dtype != np.float32 and new_params[
                    k].dtype != np.float16:
                new_params[k] = new_params[k].astype(np.float32)
        return new_params


class KerasParser:

    def load_parameters(self,
                        checkpoint_path: Path,
                        load_model_on_cpu: bool = False) -> h5py.File:
        checkpoint_path = checkpoint_path.absolute()
        config_file = "config.json"
        weights_file = json.load(open(checkpoint_path / config_file))["weights"]
        h5_path = checkpoint_path / weights_file
        return h5py.File(h5_path, "r")

    def embedding_weights(self, ckpt_params) -> np.ndarray:
        return np.array(ckpt_params["layers/reversible_embedding/vars/0"])

    def get_config(self, checkpoint_path, ckpt_params,
                   num_embed) -> SimpleNamespace:
        checkpoint_path = checkpoint_path.absolute()
        config_file = "config.json"
        config_old = json.load(open(checkpoint_path / config_file))["config"]
        config_new = SimpleNamespace()
        config_new.num_layers = config_old["num_layers"]
        config_new.num_embed = config_old["vocabulary_size"]
        config_new.embed_dim = config_old["hidden_dim"]
        config_new.hidden_dim = config_old["intermediate_dim"] // 2
        config_new.num_heads = config_old["num_query_heads"]
        config_new.head_dim = config_old["head_dim"]
        config_new.num_kv_heads = config_old["num_key_value_heads"]
        return config_new

    def rename_to_trt_llm(self, name: str) -> Optional[str]:
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix = "transformer"
        name = name.replace("/gemma_decoder_block/", "/gemma_decoder_block_0/")
        sub_patterns = (
            (r"layers/reversible_embedding/vars/0", r"vocab_embedding.weight"),
            (r"layers/gemma_decoder_block_(\d+)/pre_attention_norm/vars/0",
             r"layers.\1.input_layernorm.weight"),
            (r"layers/gemma_decoder_block_(\d+)/attention/query_dense/vars/0",
             r"layers.\1.attention.qkv.weight"),
            (r"layers/gemma_decoder_block_(\d+)/attention/key_dense/vars/0",
             None),  # drop as k will be concatenated with q
            (r"layers/gemma_decoder_block_(\d+)/attention/value_dense/vars/0",
             None),  # drop as v will be concatenated with q
            (r"layers/gemma_decoder_block_(\d+)/attention/output_dense/vars/0",
             r"layers.\1.attention.dense.weight"),
            (r"layers/gemma_decoder_block_(\d+)/gating_ffw/vars/0",
             r"layers.\1.mlp.fc.weight"),
            (r"layers/gemma_decoder_block_(\d+)/gating_ffw_2/vars/0",
             None),  # merged with above
            (r"layers/gemma_decoder_block_(\d+)/ffw_linear/vars/0",
             r"layers.\1.mlp.proj.weight"),
            (r"layers/gemma_decoder_block_(\d+)/pre_ffw_norm/vars/0",
             r"layers.\1.post_layernorm.weight"),
            (r"layers/rms_normalization/vars/0", r"ln_f.weight"),
            (r"optimizer/vars/(\d+)", None),  # Not used
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {prefix}.{name}")

    def flatten_params(self, params) -> Flattened:
        f_params = {}

        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.dtype == "|V2":
                    # bfloat16 case
                    f_params[name] = torch_to_numpy(
                        numpy_to_torch(np.array(obj).astype(np_bfloat16)).to(
                            torch.float32))
                else:
                    f_params[name] = np.array(obj)

        params.visititems(walk)
        return f_params


class TorchParser:

    def load_parameters(self,
                        checkpoint_path: Path,
                        load_model_on_cpu: bool = False) -> Weights:
        ckpt_path = list(checkpoint_path.glob('*.ckpt'))[0]
        model_params = torch.load(ckpt_path)['model_state_dict']
        model_params.pop('freqs_cis')
        return model_params

    def embedding_weights(self, ckpt_params) -> np.ndarray:
        return ckpt_params["embedder.weight"]

    def get_config(self, checkpoint_path, ckpt_params,
                   num_embed) -> SimpleNamespace:
        checkpoint_path = checkpoint_path.absolute()
        config_file = "config.json"
        with open(checkpoint_path / config_file, 'r') as f:
            json_str = f.read()
            json_str = json_str.replace("'", "\"")
            json_str = json_str.replace(",\n}", "\n}")
            config_old = json.loads(json_str)
        config_new = SimpleNamespace()
        config_new.num_layers = config_old["num_hidden_layers"]
        config_new.num_embed = config_old["vocab_size"]
        config_new.embed_dim = config_old["hidden_size"]
        config_new.hidden_dim = config_old["intermediate_size"]
        config_new.num_heads = config_old["num_attention_heads"]
        config_new.head_dim = config_old["head_dim"]
        config_new.num_kv_heads = config_old["num_key_value_heads"]
        return config_new

    def rename_to_trt_llm(self, name: str) -> Optional[str]:
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix = "transformer"
        sub_patterns = (
            (r"embedder.weight", r"vocab_embedding.weight"),
            (r"model.layers.(\d+).input_layernorm.weight",
             r"layers.\1.input_layernorm.weight"),
            (r"model.layers.(\d+).self_attn.qkv_proj.weight",
             r"layers.\1.attention.qkv.weight"),
            (r"model.layers.(\d+).self_attn.o_proj.weight",
             r"layers.\1.attention.dense.weight"),
            (r"model.layers.(\d+).mlp.gate_proj.weight",
             r"layers.\1.mlp.fc.weight"),
            (r"model.layers.(\d+).mlp.up_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).mlp.down_proj.weight",
             r"layers.\1.mlp.proj.weight"),
            (r"model.layers.(\d+).post_attention_layernorm.weight",
             r"layers.\1.post_layernorm.weight"),
            (r"model.norm.weight", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {name}")

    def flatten_params(self, params) -> Flattened:
        f_params = {}
        for k, v in params.items():
            if v.dtype == torch.bfloat16:
                v = v.float()
            f_params[k] = torch_to_numpy(v)
        return f_params


class HfParser:

    def load_parameters(self,
                        checkpoint_path: Path,
                        load_model_on_cpu: bool = False) -> Weights:
        """`AutoModelForCausalLM.from_pretrained` will parse the correct gemma, whether Gemma or Gemma2 or future versions."""
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="cpu" if load_model_on_cpu else "auto",
            dtype='auto',
            trust_remote_code=True,
        )
        model_params = dict(hf_model.named_parameters())
        return model_params

    def embedding_weights(self, ckpt_params) -> np.ndarray:
        raise RuntimeError(
            "This method shouldn't be called - `GemmaConfig.from_hugging_face` takes care of this."
        )

    def get_config(self, checkpoint_path, ckpt_params,
                   num_embed) -> SimpleNamespace:
        raise RuntimeError(
            "This method shouldn't be called - `GemmaConfig.from_hugging_face` takes care of this."
        )

    def rename_to_trt_llm(self, name: str) -> Optional[str]:
        """Rename a gemma parameter name by the corresponding TRT-LLM style name."""
        prefix = "transformer"
        sub_patterns = (
            (r"model.embed_tokens.weight", r"vocab_embedding.weight"),
            (r"model.layers.(\d+).input_layernorm.weight",
             r"layers.\1.input_layernorm.weight"),
            (r"model.layers.(\d+).self_attn.q_proj.weight",
             r"layers.\1.attention.qkv.weight"),
            (r"model.layers.(\d+).self_attn.k_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).self_attn.v_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).self_attn.o_proj.weight",
             r"layers.\1.attention.dense.weight"),
            (r"model.layers.(\d+).self_attn.q_norm.weight",
             r"layers.\1.attention.q_layernorm.weight"),
            (r"model.layers.(\d+).self_attn.k_norm.weight",
             r"layers.\1.attention.k_layernorm.weight"),
            (r"model.layers.(\d+).mlp.gate_proj.weight",
             r"layers.\1.mlp.fc.weight"),
            (r"model.layers.(\d+).mlp.up_proj.weight",
             None),  # merged with above
            (r"model.layers.(\d+).mlp.down_proj.weight",
             r"layers.\1.mlp.proj.weight"),
            (r"model.layers.(\d+).post_attention_layernorm.weight",
             r"layers.\1.post_layernorm.weight"),
            (r"model.layers.(\d+).pre_feedforward_layernorm.weight",
             r"layers.\1.pre_feedforward_layernorm.weight"),
            (r"model.layers.(\d+).post_feedforward_layernorm.weight",
             r"layers.\1.post_feedforward_layernorm.weight"),
            (r"model.norm.weight", r"ln_f.weight"),
        )

        for source, target in sub_patterns:
            if re.match(source, name):
                if target is None:
                    return target
                else:
                    name = re.sub(source, target, name)
                    return ".".join((prefix, name))
        else:
            raise ValueError(f"Don't know how to rename {prefix}.{name}")

    def flatten_params(self, params: Weights) -> Dict[str, numpy.ndarray]:
        f_params = {}
        for k, v in params.items():
            if v.dtype == torch.bfloat16:
                v = v.float()
            f_params[k] = torch_to_numpy(v)
        return f_params


def split(v: np.ndarray, tp_size: int, idx: int, dim: int = 0) -> np.ndarray:
    if tp_size == 1:
        return v
    return np.split(v, tp_size, axis=dim)[idx]


def split_matrix_tp(v: np.ndarray, tensor_parallel: int, rank: int,
                    dim: int) -> np.ndarray:
    return split(v, tensor_parallel, rank, dim=dim)


def add_trt_llm_weight(weights: Flattened,
                       name: str,
                       param: np.ndarray,
                       dtype: Optional[np.dtype] = None):
    assert name not in weights, f"{name} is already added."
    param = numpy_to_torch(param)
    if dtype is not None:
        assert isinstance(dtype,
                          str), f"dtype must be str, but get type {type(dtype)}"
        param = param.to(str_dtype_to_torch(dtype))
    weights[name] = param.contiguous()


def quantize(param: np.ndarray,
             quant_mode: tensorrt_llm.quantization.QuantMode):
    if quant_mode.is_int8_weight_only():
        quant_dtype = torch.int8
    elif quant_mode.is_int4_weight_only():
        quant_dtype = torch.quint4x2
    else:
        raise ValueError(f"Invalid configuration got quant_mode={quant_mode}")

    param = numpy_to_torch(param)
    param = param.t().contiguous()

    # previously this fn was available in torch.ops.fastertransformer namespace
    (
        quantized_weights,
        scales,
    ) = torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
        param, quant_dtype)

    if scales.dtype == torch.bfloat16:
        scales = scales.to(torch.float32).numpy().astype("bfloat16")
    else:
        scales = scales.numpy()
    return quantized_weights.numpy(), scales


Parsers = Union[JAXParser, KerasParser, TorchParser, HfParser]


def load_gemma_weights(
    *,
    trt_llm_config: tllm_utils.PretrainedConfig,
    parameters_or_model_dir: "Union[Weights, Path]",
    ckpt_parser: Parsers,
    load_model_on_cpu: bool = True,
):
    print("Loading weights...")
    tik = time.time()

    tp_rank = trt_llm_config.mapping.tp_rank
    tp_size = trt_llm_config.mapping.tp_size
    hidden_size = trt_llm_config.hidden_size
    head_dim = trt_llm_config.head_size

    weights = {}

    if isinstance(parameters_or_model_dir, Path):
        logger.debug(f"Loading directory {str(parameters_or_model_dir)}...")
        model_params = ckpt_parser.load_parameters(
            parameters_or_model_dir,
            load_model_on_cpu=load_model_on_cpu,
        )
    else:
        model_params = parameters_or_model_dir

    model_params = ckpt_parser.flatten_params(model_params)

    for name, param in model_params.items():
        logger.debug(f"Converting weight {name}...")
        trt_llm_name = ckpt_parser.rename_to_trt_llm(name)
        if trt_llm_name is None:  # omit as used with other params
            continue

        if "attn.q_einsum" in name:
            gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads
            assert gqa_mode

            # initial shape: (num_q_heads, hidden_size, head_dim)
            q_param = param.transpose(1, 0, 2)
            q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=1)

            # initial shape: (2, num_kv_heads, hidden_size, head_dim)
            kv_name = name.replace("q_einsum", "kv_einsum")
            kv_param = model_params[kv_name]
            kv_param = kv_param.reshape(
                trt_llm_config.num_key_value_heads * 2,
                hidden_size,
                head_dim,
            ).transpose(1, 0, 2)

            # -> (hidden_size, num_q_heads / tp_size + 2, head_dim)
            qkv_param = np.concatenate([q_param, kv_param], axis=1)
            qkv_param = qkv_param.reshape(qkv_param.shape[0], -1)
            qkv_param = qkv_param.transpose(1, 0)

            # If int8 kv enabled, weight-only quantization will be done later.
            if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                not trt_llm_config.quant_mode.has_int8_kv_cache():
                qkv_param_quantized, qkv_param_scales = quantize(
                    qkv_param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, qkv_param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    qkv_param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                   trt_llm_config.dtype)
        elif "self_attn.qkv_proj" in name:
            q_param, k_param, v_param = np.split(param, [
                trt_llm_config.num_attention_heads * trt_llm_config.head_size,
                trt_llm_config.num_attention_heads * trt_llm_config.head_size +
                trt_llm_config.num_key_value_heads * trt_llm_config.head_size
            ],
                                                 axis=0)
            gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads

            q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=0)
            if not gqa_mode:
                k_param = split_matrix_tp(k_param, tp_size, tp_rank, dim=0)
                v_param = split_matrix_tp(v_param, tp_size, tp_rank, dim=0)

            qkv_param = np.concatenate([q_param, k_param, v_param], axis=0)
            if trt_llm_config.quant_mode.is_weight_only(
            ) and not trt_llm_config.quant_mode.has_per_group_scaling():
                qkv_param_quantized, qkv_param_scales = quantize(
                    qkv_param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, qkv_param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    qkv_param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                   trt_llm_config.dtype)
        elif "attn.qkv_einsum" in name:
            gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads
            assert not gqa_mode
            # initial shape: [3, num_heads, hidden_size, head_dim] -> [3, num_heads, head_dim, hidden_size]
            qkv_param = param.transpose(0, 1, 3, 2)
            qkv_param = qkv_param.reshape(qkv_param.shape[0], -1,
                                          qkv_param.shape[3])
            qkv_param = split_matrix_tp(qkv_param, tp_size, tp_rank, dim=1)
            qkv_param = qkv_param.reshape(-1, qkv_param.shape[2])
            if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() \
                and not trt_llm_config.quant_mode.has_int8_kv_cache():
                qkv_param_quantized, qkv_param_scales = quantize(
                    qkv_param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, qkv_param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    qkv_param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                   trt_llm_config.dtype)
        elif "attention/query_dense" in name:
            # Keras specific KQV convert
            gqa_mode = trt_llm_config.num_attention_heads != trt_llm_config.num_key_value_heads
            if gqa_mode:

                # initial shape: (num_q_heads, hidden_size, head_dim)
                q_param = param.transpose(1, 0, 2)
                q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=1)

                # initial shape: (2, num_kv_heads, hidden_size, head_dim)
                k_name = name.replace("query", "key")
                k_param = model_params[k_name]
                v_name = name.replace("query", "value")
                v_param = model_params[v_name]
                kv_param = np.stack((k_param, v_param), axis=0)

                kv_param = kv_param.reshape(
                    trt_llm_config.num_key_value_heads * 2,
                    hidden_size,
                    head_dim,
                ).transpose(1, 0, 2)

                # -> (hidden_size, num_q_heads / tp_size + 2, head_dim)
                qkv_param = np.concatenate([q_param, kv_param], axis=1)
                qkv_param = qkv_param.reshape(qkv_param.shape[0], -1)
                qkv_param = qkv_param.transpose(1, 0)

                if trt_llm_config.quant_mode.is_weight_only(
                ) and not trt_llm_config.quant_mode.has_int8_kv_cache():
                    qkv_param_quantized, qkv_param_scales = quantize(
                        qkv_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       qkv_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        qkv_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                       trt_llm_config.dtype)
            else:
                q_param = param
                k_name = name.replace("query", "key")
                k_param = model_params[k_name]
                v_name = name.replace("query", "value")
                v_param = model_params[v_name]
                # initial shape: [3, num_heads, hidden_size, head_dim] -> [3, num_heads, head_dim, hidden_size]
                qkv_param = np.stack((q_param, k_param, v_param), axis=0)
                qkv_param = qkv_param.transpose(0, 1, 3, 2)
                qkv_param = qkv_param.reshape(qkv_param.shape[0], -1,
                                              qkv_param.shape[3])
                qkv_param = split_matrix_tp(qkv_param, tp_size, tp_rank, dim=1)
                qkv_param = qkv_param.reshape(-1, qkv_param.shape[2])
                if trt_llm_config.quant_mode.is_weight_only(
                ) and not trt_llm_config.quant_mode.has_int8_kv_cache():
                    qkv_param_quantized, qkv_param_scales = quantize(
                        qkv_param, trt_llm_config.quant_mode)
                    add_trt_llm_weight(weights, trt_llm_name,
                                       qkv_param_quantized)
                    add_trt_llm_weight(
                        weights,
                        trt_llm_name.replace(".weight", ".per_channel_scale"),
                        qkv_param_scales,
                        trt_llm_config.dtype,
                    )
                else:
                    add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                       trt_llm_config.dtype)
        elif "q_proj" in name:
            mqa_mode = trt_llm_config.num_key_value_heads == 1

            if mqa_mode:
                # initial shape: (num_heads * head_dim, hidden_size)
                q_param = param
                q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=0)

                k_name = name.replace("q_proj", "k_proj")
                k_param = model_params[k_name]

                v_name = name.replace("q_proj", "v_proj")
                v_param = model_params[v_name]
            else:
                # initial shape: (num_heads * head_dim, hidden_size)
                q_param = param
                q_param = split_matrix_tp(q_param, tp_size, tp_rank, dim=0)

                k_name = name.replace("q_proj", "k_proj")
                k_param = model_params[k_name]
                k_param = split_matrix_tp(k_param, tp_size, tp_rank, dim=0)

                v_name = name.replace("q_proj", "v_proj")
                v_param = model_params[v_name]
                v_param = split_matrix_tp(v_param, tp_size, tp_rank, dim=0)

            qkv_param = np.concatenate([q_param, k_param, v_param], axis=0)
            qkv_param = qkv_param.reshape(qkv_param.shape[0], -1)

            # If int8 kv enabled, weight-only quantization will be done later.
            if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                not trt_llm_config.quant_mode.has_int8_kv_cache():
                qkv_param_quantized, qkv_param_scales = quantize(
                    qkv_param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, qkv_param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    qkv_param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, qkv_param,
                                   trt_llm_config.dtype)
        elif "attention.dense.weight" in trt_llm_name:
            # initial shape: (num_heads, head_dim, hidden_size)
            if len(param.shape) == 3:
                param = param.reshape(-1, param.shape[2])
                param = param.transpose(
                    1, 0)  # (hidden_size, num_heads * head_dum)
            param = split_matrix_tp(param, tp_size, tp_rank, dim=1)
            if trt_llm_config.quant_mode.is_weight_only(
            ) and not trt_llm_config.quant_mode.has_int8_kv_cache():
                param_quantized, param_scales = quantize(
                    param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
        elif "mlp.fc.weight" in trt_llm_name:
            if isinstance(ckpt_parser, KerasParser):
                # initial shape: (hidden_size, intermediate_size)
                fc_param, gate_param = param, model_params[name.replace(
                    "gating_ffw", "gating_ffw_2")]
            elif isinstance(ckpt_parser, TorchParser):
                # initial shape: (intermediate_size, hidden_size)
                fc_param, gate_param = param, model_params[name.replace(
                    "mlp.gate_proj", "mlp.up_proj")]
                fc_param = fc_param.transpose(1, 0)
                gate_param = gate_param.transpose(1, 0)
            elif isinstance(ckpt_parser, HfParser):
                # initial shape: (intermediate_size, hidden_size)
                fc_param, gate_param = param, model_params[name.replace(
                    "mlp.gate_proj", "mlp.up_proj")]
                fc_param = fc_param.transpose(1, 0)
                gate_param = gate_param.transpose(1, 0)
            else:
                # initial shape: (2, hidden_size, intermediate_size)
                fc_param, gate_param = param[0], param[1]
            fc_param = fc_param.transpose(1, 0)
            fc_param = split_matrix_tp(fc_param, tp_size, tp_rank, dim=0)
            if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                not trt_llm_config.quant_mode.has_int8_kv_cache():
                fc_param_quantized, fc_param_scales = quantize(
                    fc_param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, fc_param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    fc_param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, fc_param,
                                   trt_llm_config.dtype)

            gate_param = gate_param.transpose(1, 0)
            gate_param = split_matrix_tp(gate_param, tp_size, tp_rank, dim=0)
            trt_llm_name = trt_llm_name.replace("mlp.fc.weight",
                                                "mlp.gate.weight")
            if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                not trt_llm_config.quant_mode.has_int8_kv_cache():
                gate_param_quantized, gate_param_scales = quantize(
                    gate_param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, gate_param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    gate_param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, gate_param,
                                   trt_llm_config.dtype)
        elif "mlp.proj.weight" in trt_llm_name:
            if not isinstance(ckpt_parser, TorchParser) and not isinstance(
                    ckpt_parser, HfParser):
                # initial shape: (intermediate_size, hidden_size)
                param = param.transpose(1, 0)
            param = split_matrix_tp(param, tp_size, tp_rank, dim=1)
            if trt_llm_config.quant_mode.is_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                not trt_llm_config.quant_mode.has_int8_kv_cache():
                param_quantized, param_scales = quantize(
                    param, trt_llm_config.quant_mode)
                add_trt_llm_weight(weights, trt_llm_name, param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_channel_scale"),
                    param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
        elif "embedder.input_embedding" in name or "reversible_embedding" in name or "embedder.weight" in name \
                or "embed_tokens.weight" in name:
            # TODO: safetensor doesn't allow to save a shared tensor.
            # Currently, we clone the weight but to save the disk, it
            # would be better to skip saving lm_head weights and
            # handle it at the loading phase.
            lm_head = split_matrix_tp(param, tp_size, tp_rank, dim=0)
            add_trt_llm_weight(weights, "lm_head.weight", np.copy(lm_head),
                               trt_llm_config.dtype)

            if trt_llm_config.use_parallel_embedding:
                assert trt_llm_config.vocab_size % tp_size == 0
                param = split_matrix_tp(
                    param,
                    tp_size,
                    tp_rank,
                    dim=trt_llm_config.embedding_sharding_dim,
                )
            if trt_llm_config.quant_mode.is_int8_weight_only() and not trt_llm_config.quant_mode.has_per_group_scaling() and \
                not trt_llm_config.quant_mode.has_int8_kv_cache() and trt_llm_config.quantization.exclude_modules is not None:

                # shape of embedding table: [V, K], V: vocab size, K: embedding dim

                # quantize will do following work:
                # 1. transpose the input from [V, K] to [K, V]
                # 2. compute V scales across dimension K
                # 3. quantize the input to int8_weight
                # 4. transpose the int8_weight to [V, K], but there is a bug in 'quantize' that the claimed shape is [K, V]
                param_quantized, param_scales = quantize(
                    param, trt_llm_config.quant_mode)

                # Reshape the [K, V] to [V, K] to match the real data layout.
                param_quantized = np.ascontiguousarray(
                    param_quantized.reshape(
                        [param_quantized.shape[1], param_quantized.shape[0]]))

                add_trt_llm_weight(weights, trt_llm_name, param_quantized)
                add_trt_llm_weight(
                    weights,
                    trt_llm_name.replace(".weight", ".per_token_scale"),
                    param_scales,
                    trt_llm_config.dtype,
                )
            else:
                add_trt_llm_weight(weights, trt_llm_name, param,
                                   trt_llm_config.dtype)
        elif any(keyword in name for keyword in (
                "pre_attention_norm.scale",
                "pre_ffw_norm.scale",
                "final_norm.scale",
                "pre_attention_norm/vars/0",
                "pre_ffw_norm/vars/0",
                "rms_normalization/vars/0",
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "model.norm.weight",
                "q_norm.weight",
                "k_norm.weight",
        )):
            param = param + 1.0  # upcasted to float32 in case of bfloat16
            add_trt_llm_weight(weights, trt_llm_name, param,
                               trt_llm_config.dtype)
        else:
            raise RuntimeError(f"Unhandled {name} module weights")
    del model_params

    print(
        f"Weights loaded. Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - tik))}"
    )
    return weights


@dataclass(frozen=True, kw_only=True)
class QuantizeModifiers:
    """
    Bag of additional conversion parameters, sourced from argparse or from defaults.
    We may use this class to easily create wrappers of `convert_gemma`
    (Otherwise, we would need to repeat these params many times or spread `**kwargs` in an untypesafe manner)
    """

    use_weight_only_with_precision: Optional[str] = None
    per_channel: bool = False
    per_token: bool = False
    calib_dataset: str = "ccdv/cnn_dailymail"
    tokenizer_dir: Optional[str] = None
    enable_fp8: bool = False
    fp8_kv_cache: bool = False
    quant_ckpt_path: Optional[str] = None

    @classmethod
    def from_args(cls, args: Namespace) -> "QuantizeModifiers":
        return cls(**{
            k: v
            for k, v in vars(args).items() if k in cls.__annotations__
        })


def non_modelopt_quantize_if_needed(
        weights: Weights, *, model_dir: Path,
        quantize_modifiers: QuantizeModifiers,
        trt_llm_config: Union[GemmaConfig,
                              tllm_utils.PretrainedConfig]) -> Weights:
    tokenizer_dir = quantize_modifiers.tokenizer_dir or model_dir

    quant_cfg = trt_llm_config.quantization

    kv_cache_is_int8 = quant_cfg.kv_cache_quant_algo == QuantAlgo.INT8
    use_smooth_quant = quant_cfg._use_plugin_sq

    if use_smooth_quant or kv_cache_is_int8:
        qkv_para = {}
        smoother = {}
        dataset = load_calib_dataset(quantize_modifiers.calib_dataset)
        tokenizer = sp.SentencePieceProcessor(model_file=tokenizer_dir)
        if "transformer.vocab_embedding.weight" in weights:
            # To use the HF to do SmoothQuant, we need to scale the embedding.
            weights["transformer.vocab_embedding.weight"] = torch.multiply(
                weights["transformer.vocab_embedding.weight"].to(torch.float32),
                math.sqrt(trt_llm_config.hidden_size),
            )
        hf_model = create_model_from_config(trt_llm_config, weights)
        act_range = capture_activation_range(hf_model, tokenizer, dataset)
        if use_smooth_quant:
            smooth_model(hf_model, act_range, quant_cfg.smoothquant_val,
                         qkv_para, smoother)
        weights = convert_hf_model(
            hf_model=hf_model,
            mapping=trt_llm_config.mapping,
            vocab_size=trt_llm_config.vocab_size or 32000,
            dtype=trt_llm_config.dtype,
            use_parallel_embedding=trt_llm_config.use_parallel_embedding,
            sharding_dim=0,
            use_weight_only=quantize_modifiers.use_weight_only_with_precision
            is not None,
            plugin_weight_only_quant_type=torch.int8
            if quantize_modifiers.use_weight_only_with_precision == 'int8' else
            torch.quint4x2,
            use_smooth_quant=use_smooth_quant,
            per_channel=quantize_modifiers.per_channel,
            per_token=quantize_modifiers.per_token,
            int8_kv_cache=kv_cache_is_int8,
            act_range=act_range,
            qkv_para=qkv_para,
            smoother=smoother)
        if "transformer.vocab_embedding.weight" in weights:
            # Revert the scaling of embedding
            weights["transformer.vocab_embedding.weight"] = torch.divide(
                weights["transformer.vocab_embedding.weight"].to(torch.float32),
                math.sqrt(trt_llm_config.hidden_size),
            ).to(str_dtype_to_torch(trt_llm_config.dtype))
        if "lm_head.weight" in weights:  # Remove lm_head before saving it in unified checkpoint.
            del weights["lm_head.weight"]
        return weights
    if quantize_modifiers.use_weight_only_with_precision and quantize_modifiers.use_weight_only_with_precision.endswith(
            "awq"):
        weights = dummy_weights_awq(
            weights=weights,
            precision=quantize_modifiers.use_weight_only_with_precision,
            trt_llm_config=trt_llm_config,
            group_size=128)
    elif quantize_modifiers.enable_fp8 or quantize_modifiers.fp8_kv_cache:
        weight_scales = quantize_fp8_weights(weights,
                                             trt_llm_config.num_hidden_layers,
                                             trt_llm_config.mapping)
        scales = load_from_fp8_gemma(quantize_modifiers.quant_ckpt_path,
                                     trt_llm_config.num_hidden_layers,
                                     trt_llm_config.mapping,
                                     quantize_modifiers.fp8_kv_cache,
                                     weight_scales)
        weights.update(scales)

    return weights


def load_gemma_weights_from_hf_model(
        hf_model: "transformers.AutoModelForCausalLM",
        config: GemmaConfig) -> Weights:
    return load_gemma_weights(parameters_or_model_dir=dict(
        hf_model.named_parameters()),
                              trt_llm_config=config,
                              ckpt_parser=HfParser())
