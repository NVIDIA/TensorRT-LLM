# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dense BF16 arithmetic and batch-shape helpers for LTX-2.3."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING, TypeGuard, TypeVar

import torch
from torch import nn

from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.utils import gelu_tanh

from ..ltx2.text_cache import TextCache
from ..ltx2.transformer_ltx2 import BasicAVTransformerBlock, TransformerConfig
from .ltx23_core.modality import LTX23Modality

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig


def _uses_reference_precision(
    config: DiffusionModelConfig | None,
) -> TypeGuard[DiffusionModelConfig]:
    """Select dense BF16 arithmetic without changing any quantized model path."""
    return (
        config is not None
        and config.torch_dtype == torch.bfloat16
        and config.quant_config.quant_algo is None
        and not config.quant_config_dict
        and not config.dynamic_weight_quant
        and config.attention.quant_attention_config is None
    )


class _DenseReferenceMLP(MLP):
    def _unquantized_gelu_fusion_eligible(self, _x: torch.Tensor) -> bool:
        """Keep the BF16 up-projection rounded before applying GELU."""
        return False


class _DenseReferenceRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        # A constant factory is also valid inside the loader's meta-init context.
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return nn.functional.rms_norm(value, self.weight.shape, self.weight, self.eps)


def _make_mlp(cfg: TransformerConfig, model_config: DiffusionModelConfig | None, idx: int) -> MLP:
    if not _uses_reference_precision(model_config):
        return BasicAVTransformerBlock._make_mlp(cfg, model_config, idx)
    return _DenseReferenceMLP(
        hidden_size=cfg.dim,
        intermediate_size=cfg.dim * 4,
        bias=True,
        activation=gelu_tanh,
        dtype=model_config.torch_dtype,
        config=model_config,
        layer_idx=idx,
    )


_BatchBundle = TypeVar("_BatchBundle", LTX23Modality, TextCache)


def _slice_batch_bundle(bundle: _BatchBundle, index: int, batch_size: int) -> _BatchBundle:
    """Slice batch tensors, keeping singleton and shared CFG positional data."""

    def part(value: object) -> object:
        if isinstance(value, torch.Tensor):
            if value.ndim and value.shape[0] == batch_size:
                return value[index : index + 1]
            # CFG branches can share an original, smaller batch of positions.
            if value.ndim >= 3 and 1 < value.shape[0] < batch_size:
                if batch_size % value.shape[0] == 0:
                    row = index % value.shape[0]
                    return value[row : row + 1]
            return value
        if isinstance(value, tuple):
            return tuple(part(item) for item in value)
        if isinstance(value, list):
            return [part(item) for item in value]
        return value

    return replace(
        bundle, **{field.name: part(getattr(bundle, field.name)) for field in fields(bundle)}
    )
