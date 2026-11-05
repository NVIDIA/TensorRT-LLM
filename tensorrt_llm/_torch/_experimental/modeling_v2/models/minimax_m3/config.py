# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration interpretation shared by the SM107 MiniMax-M3 targets."""

from __future__ import annotations

import copy
import dataclasses
from typing import List

from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig


def _is_vl_config(pretrained_config: PretrainedConfig) -> bool:
    model_type = getattr(pretrained_config, "model_type", None)
    if model_type == "minimax_m3_vl":
        return True
    if model_type == "minimax_m3":
        return False
    return any(
        "ForConditionalGeneration" in a
        for a in (getattr(pretrained_config, "architectures", None) or [])
    )


def _text_model_config(model_config: ModelConfig) -> ModelConfig:
    """The model config with the language model's ``text_config`` as its pretrained config.

    The checkpoint declares the vision-language wrapper; its ``text_config`` arrives as a plain dict (no typed
    sub-config is registered), so it is materialized into a ``PretrainedConfig`` with the wrapper's dtype and
    embedding tie carried over. A text config is returned unchanged.
    """
    if not _is_vl_config(model_config.pretrained_config):
        return model_config
    cfg = copy.deepcopy(model_config)
    text = getattr(cfg.pretrained_config, "text_config", None)
    if text is None:
        raise ValueError("MiniMax-M3 VL config has no `text_config`")
    if isinstance(text, dict):
        out = PretrainedConfig()
        for k, v in text.items():
            setattr(out, k, v)
        text = out
    if (
        getattr(text, "torch_dtype", None) is None
        and getattr(cfg.pretrained_config, "torch_dtype", None) is not None
    ):
        text.torch_dtype = cfg.pretrained_config.torch_dtype
    if (
        getattr(text, "tie_word_embeddings", None) is None
        and getattr(cfg.pretrained_config, "tie_word_embeddings", None) is not None
    ):
        text.tie_word_embeddings = cfg.pretrained_config.tie_word_embeddings
    return dataclasses.replace(cfg, pretrained_config=text)


def _validate_runtime(model_config: ModelConfig) -> None:
    """The runtime this target is written for. It is set by engine arguments rather than the checkpoint, so routing
    sees only its KV cache algorithm; this check at construction refuses every other runtime.

    The MiniMax-M3 MSA backend (it owns the step metadata and the index-K cache), the fused five-way QKV/index
    projection or the separate QKV and index projections, an FP8 indexer and an FP8 or NVFP4 main KV cache.
    """
    sparse = model_config.sparse_attention_config
    if (
        sparse is None
        or getattr(sparse, "algorithm", None) != "minimax_m3"
        or sparse.implementation != "msa"
    ):
        raise ValueError(
            "this MiniMax-M3 target requires sparse_attention_config.algorithm='minimax_m3' with implementation='msa'"
        )
    if getattr(sparse, "indexer_kv_dtype", None) != "fp8":
        raise ValueError(
            "this MiniMax-M3 target requires sparse_attention_config.indexer_kv_dtype='fp8'"
        )
    mode = model_config.quant_config.quant_mode
    if mode is None or not (mode.has_fp8_kv_cache() or mode.has_fp4_kv_cache()):
        raise ValueError("this MiniMax-M3 target requires an FP8 or NVFP4 KV cache")


def _sparse_layer_ids(cfg: PretrainedConfig) -> List[int]:
    freq = cfg.sparse_attention_config["sparse_attention_freq"]
    return [i for i, f in enumerate(freq) if int(f) != 0]


def _moe_layer_ids(cfg: PretrainedConfig) -> List[int]:
    freq = getattr(cfg, "moe_layer_freq", None)
    if freq is None:
        return list(range(cfg.num_hidden_layers))
    return [i for i, f in enumerate(freq) if int(f) != 0]
