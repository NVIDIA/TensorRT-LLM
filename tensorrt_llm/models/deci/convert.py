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
import enum
import json
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, TypedDict, Union

import safetensors
import torch

from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.logger import logger
from tensorrt_llm.models.convert_utils import dup_kv_weight, split
from tensorrt_llm.models.deci.layer_config import (AttentionConfig,
                                                   AttentionImplementation,
                                                   DeciLayerConfig, FFNConfig,
                                                   FFNImplementation)
from tensorrt_llm.quantization.mode import QuantAlgo


def _ffn_mult_to_intermediate_size(ffn_mult: float, n_embd: int) -> int:
    intermediate_size = int(2 * ffn_mult * n_embd / 3)
    return _find_multiple(intermediate_size, 256)


def _find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# BlockConfig is a custom class defined inside deci huggingface checkpoints, we can't import it
def hf_block_config_to_layer_config(block_config: "BlockConfig",
                                    num_attn_heads: int,
                                    hidden_size: int) -> DeciLayerConfig:
    attn = block_config.attention
    if attn.no_op:
        attn_impl = AttentionImplementation.NO_OP
        num_key_value_heads = None
    elif attn.replace_with_linear:
        attn_impl = AttentionImplementation.LINEAR
        num_key_value_heads = None
    elif attn.sparsify:
        raise NotImplementedError("Sparsification is not supported")
    else:
        attn_impl = AttentionImplementation.ATTENTION
        num_key_value_heads = num_attn_heads // attn.n_heads_in_group

    ffn = block_config.ffn
    if ffn.no_op:
        ffn_impl = FFNImplementation.NO_OP
        intermediate_size = None
    elif ffn.replace_with_linear:
        ffn_impl = FFNImplementation.LINEAR
        intermediate_size = None
    elif ffn.sparsify:
        raise NotImplementedError("Sparsification is not supported")
    else:
        ffn_impl = FFNImplementation.MLP
        intermediate_size = _ffn_mult_to_intermediate_size(
            ffn.ffn_mult, hidden_size)

    return DeciLayerConfig(attention=AttentionConfig(
        impl=attn_impl, num_key_value_heads=num_key_value_heads),
                           ffn=FFNConfig(impl=ffn_impl,
                                         intermediate_size=intermediate_size))


@contextmanager
def timed_loading() -> Iterator[None]:
    tik = time.time()
    yield

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')


class TpDim(enum.IntEnum):
    NO_TP = -1
    COLWISE = 0
    ROWWISE = 1


class SafetensorsIndex(TypedDict):
    metadata: Dict[str, Any]
    weight_map: Dict[str, str]


class WeightsLoader(ABC):

    @abstractmethod
    def get_weight(self,
                   name: str,
                   tp_dim: TpDim = TpDim.NO_TP,
                   tp_size: int = 1,
                   tp_rank: int = 0) -> torch.Tensor:
        ...


class HFModelWeightsLoader(WeightsLoader):

    def __init__(self, *, hf_model: "transformers.PreTrainedModel",
                 dtype: str) -> None:
        self.model_params = dict(hf_model.named_parameters())
        self.dtype = getattr(torch, dtype)

    def get_weight(self,
                   name: str,
                   tp_dim: TpDim = TpDim.NO_TP,
                   tp_size: int = 1,
                   tp_rank: int = 0) -> torch.Tensor:
        weight = self.model_params[name]
        if weight.dtype != self.dtype:
            weight = weight.to(self.dtype)
        weight = weight.detach()

        if tp_dim != TpDim.NO_TP:
            weight = split(weight, tp_size, tp_rank, dim=tp_dim)
        return weight


class SafetensorsWeightsLoader(WeightsLoader):

    def __init__(self, *, model_dir: Path, dtype: str) -> None:
        self.model_dir = model_dir
        self.dtype = getattr(torch, dtype)

        # the index has a weight map that maps weight names to the files they are found in
        safetensor_index_json = self.model_dir / "model.safetensors.index.json"
        has_safetensor_index_json = safetensor_index_json.is_file()
        if has_safetensor_index_json:
            with safetensor_index_json.open("r") as fr:
                self.sharding_map: SafetensorsIndex = json.load(fr)
        else:
            self.sharding_map = SafetensorsIndex(metadata={}, weight_map={})

        shard_files = {f.name for f in self.model_dir.glob("*.safetensors")}
        if has_safetensor_index_json:
            # only read the files that have weights according to the index
            shard_files &= set(self.sharding_map["weight_map"].values())
        self.shard_files = sorted(list(shard_files))

        self.safetensors_files = {
            shard_file: safetensors.safe_open(model_dir / shard_file,
                                              framework="pt",
                                              device="cpu")
            for shard_file in shard_files
        }

    def get_weight(self,
                   name: str,
                   tp_dim: TpDim = TpDim.NO_TP,
                   tp_size: int = 1,
                   tp_rank: int = 0) -> torch.Tensor:
        shard_filename = self.sharding_map['weight_map'].get(
            name, self.shard_files[0])
        if tp_dim == TpDim.NO_TP:
            res = self.safetensors_files[shard_filename].get_tensor(name)
        else:
            tensor_slice = self.safetensors_files[shard_filename].get_slice(
                name)
            tensor_shape = tensor_slice.get_shape()
            if len(tensor_shape) == 1:
                if tp_dim == TpDim.COLWISE:
                    slice_width = tensor_shape[0] // tp_size
                    res = tensor_slice[slice_width * tp_rank:slice_width *
                                       (tp_rank + 1)]
                else:  # row-wise, but 1-dimensional ==> no tp
                    res = tensor_slice[:]
            else:
                assert tensor_shape[
                    tp_dim] % tp_size == 0, f"Current weight shape is invalid for tp_size={tp_size}"
                slice_width = tensor_shape[tp_dim] // tp_size
                if tp_dim == TpDim.COLWISE:
                    res = tensor_slice[slice_width * tp_rank:slice_width *
                                       (tp_rank + 1), :]
                else:
                    res = tensor_slice[:, slice_width * tp_rank:slice_width *
                                       (tp_rank + 1)]

        return res.to(self.dtype).contiguous()


def load_model_weights(loader: WeightsLoader,
                       config: "DeciConfig") -> Dict[str, torch.Tensor]:
    mapping = config.mapping
    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    pad_vocab = vocab_size % mapping.tp_size != 0
    vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)

    weights = {}

    def load_weight(name: str, tp_dim: TpDim = TpDim.NO_TP) -> torch.Tensor:
        return loader.get_weight(name=name,
                                 tp_dim=tp_dim,
                                 tp_rank=mapping.tp_rank,
                                 tp_size=mapping.tp_size)

    with timed_loading():
        if mapping.is_first_pp_rank():
            weights['transformer.vocab_embedding.weight'] = load_weight(
                "model.embed_tokens.weight",
                TpDim(config.embedding_sharding_dim)
                if config.use_parallel_embedding else
                TpDim.NO_TP)  # vocab_embedding

        if mapping.is_last_pp_rank():
            v = load_weight("lm_head.weight",
                            TpDim.NO_TP) if pad_vocab else load_weight(
                                "lm_head.weight", TpDim.COLWISE)  # lm_head
            if pad_vocab:
                v = torch.nn.functional.pad(
                    v, (0, 0, 0, vocab_size_padded - vocab_size), 'constant', 0)
                v = split(v, mapping.tp_size, mapping.tp_rank)
            weights['lm_head.weight'] = v
            weights['transformer.ln_f.weight'] = load_weight(
                "model.norm.weight")  # ln_f

        layers_range = mapping.pp_layers(num_hidden_layers)
        for l in layers_range:
            layer_config = config.get_layer_config(l)
            layer_idx = l - layers_range[0]
            tllm_prex = f'transformer.layers.{layer_idx}'

            # Attention
            if layer_config.is_attention_layer:
                weights[f'{tllm_prex}.input_layernorm.weight'] = load_weight(
                    f"model.layers.{l}.input_layernorm.weight"
                )  # input_layernorm

                qkv = {}
                for comp in ["q", "k", "v"]:
                    weight_part = load_weight(
                        f"model.layers.{l}.self_attn.{comp}_proj.weight",
                        TpDim.COLWISE)
                    qkv[comp] = weight_part

                if layer_config.attention.num_key_value_heads < mapping.tp_size:
                    # duplicate the KV heads up to tensor_parallel
                    qkv["k"] = dup_kv_weight(
                        qkv["k"], layer_config.attention.num_key_value_heads,
                        mapping.tp_size)
                    qkv["v"] = dup_kv_weight(
                        qkv["v"], layer_config.attention.num_key_value_heads,
                        mapping.tp_size)

                weights[f'{tllm_prex}.attention.qkv.weight'] = torch.cat(
                    [qkv["q"], qkv["k"], qkv["v"]], 0)
                weights[f'{tllm_prex}.attention.dense.weight'] = load_weight(
                    f"model.layers.{l}.self_attn.o_proj.weight",
                    TpDim.ROWWISE)  # attention.dense

            elif layer_config.is_linear_attention_layer:
                weights[f'{tllm_prex}.input_layernorm.weight'] = load_weight(
                    f"model.layers.{l}.input_layernorm.weight"
                )  # input_layernorm

                weights[f'{tllm_prex}.attention.weight'] = load_weight(
                    f"model.layers.{l}.self_attn.linear_attn.weight",
                    TpDim.COLWISE)

            elif not layer_config.is_noop_attention_layer:
                raise NotImplementedError(
                    f"Loading weights for layer with attention of type {layer_config.attention.impl} is not supported"
                )

            # MLP
            if layer_config.is_mlp_layer:
                weights[f'{tllm_prex}.post_layernorm.weight'] = load_weight(
                    f"model.layers.{l}.post_attention_layernorm.weight"
                )  # post_layernorm

                weights[f'{tllm_prex}.ffn.gate.weight'] = load_weight(
                    f"model.layers.{l}.mlp.up_proj.weight",
                    TpDim.COLWISE)  # mlp.gate
                weights[f'{tllm_prex}.ffn.proj.weight'] = load_weight(
                    f"model.layers.{l}.mlp.down_proj.weight",
                    TpDim.ROWWISE)  # mlp.proj
                weights[f'{tllm_prex}.ffn.fc.weight'] = load_weight(
                    f"model.layers.{l}.mlp.gate_proj.weight",
                    TpDim.COLWISE)  # mlp.fc

            elif layer_config.is_linear_ffn_layer:
                weights[f'{tllm_prex}.post_layernorm.weight'] = load_weight(
                    f"model.layers.{l}.post_attention_layernorm.weight"
                )  # post_layernorm

                weights[f'{tllm_prex}.ffn.weight'] = load_weight(
                    f"model.layers.{l}.mlp.linear_mlp.weight", TpDim.COLWISE)

            elif not layer_config.is_noop_ffn_layer:
                raise NotImplementedError(
                    f"Loading weights for a layer with FFN of type {layer_config.ffn.impl} is not implemented yet"
                )

        return weights


def load_weights_from_hf_model(
        hf_model: "transformers.PreTrainedModel",
        config: "DeciConfig",
        act_range: Optional[dict] = None,
        qkv_para: Optional[dict] = None,
        smoother: Optional[dict] = None) -> Dict[str, torch.Tensor]:
    quant_algo = config.quantization.quant_algo
    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    if quant_algo == QuantAlgo.W8A16:
        torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        torch.quint4x2
    else:
        pass

    use_smooth_quant = config.quantization.use_plugin_sq
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8
    if use_smooth_quant or int8_kv_cache:
        assert act_range is not None
        assert qkv_para is not None
        assert smoother is not None

    # TODO(oargov): add support for these quants
    assert not use_weight_only, "WOQ is not supported yet"
    assert not use_smooth_quant, "SmoothQuant is not supported yet"
    assert not int8_kv_cache, "INT8 KV cache is not supported yet"

    # TODO(oargov): support moe
    moe_config = getattr(config, "moe", None)
    assert moe_config is None, "MoE is not supported yet"

    # TODO(oargov): implement resisdual mlp
    residual_mlp = getattr(config, "residual_mlp", None)
    assert not residual_mlp, "Residual MLP is not supported yet"

    loader = HFModelWeightsLoader(hf_model=hf_model, dtype=config.dtype)
    logger.info('Converting weights from Huggingface model...')
    return load_model_weights(loader=loader, config=config)


def load_weights_from_hf_safetensors(
        model_dir: Union[str, Path],
        config: "DeciConfig") -> Dict[str, torch.Tensor]:

    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    loader = SafetensorsWeightsLoader(model_dir=model_dir, dtype=config.dtype)
    logger.info('Loading weights from Huggingface safetensors...')
    return load_model_weights(loader=loader, config=config)
