# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module and backend hooks for sparse attention algorithms.

Algorithms register typed module adapters from
``sparse/<algorithm>/module.py``. Backend prediction hooks use the backend
subclass directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Optional

from .params import SparseRuntimeParams

if TYPE_CHECKING:
    import torch

    from ...distributed import AllReduceParams
    from ...modules.attention import Attention
    from ...modules.mla import MLA
    from ..interface import AttentionForwardArgs, AttentionMask, AttentionMetadata
    from ..trtllm import TrtllmAttention

__all__ = [
    "AttentionSparseHooks",
    "MLASparseHooks",
    "get_sparse_attention_hooks",
    "get_sparse_mla_hooks",
    "prepare_sparse_attention_prediction",
    "register_attention_sparse_hooks",
    "register_mla_sparse_hooks",
]


class MLASparseHooks(ABC):
    """Typed module-layer adapter for a sparse MLA algorithm."""

    mqa_rope_append = True
    need_absorption = True
    need_dense_mha = True
    need_default_o_proj = True

    @abstractmethod
    def initialize(self, mla: "MLA") -> None:
        """Initialize algorithm-specific MLA state."""

    def get_mqa_aux_stream(self, mla: "MLA") -> Optional["torch.cuda.Stream"]:
        """Return the auxiliary stream used to initialize the MQA backend."""
        return mla.aux_stream

    def create_weights(self, mla: "MLA") -> None:
        """Create algorithm-specific weights."""

    def transform_weights(self, mla: "MLA") -> None:
        """Transform algorithm-specific weights."""

    def prepare_outputs(
        self,
        mla: "MLA",
        hidden_states: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
    ) -> Optional[list["torch.Tensor"]]:
        """Return algorithm-specific outputs, or ``None`` for the default."""
        return None

    @abstractmethod
    def forward(
        self,
        mla: "MLA",
        position_ids: Optional["torch.Tensor"],
        hidden_states: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        attn_output: list["torch.Tensor"],
    ) -> None:
        """Run the sparse MLA forward implementation."""

    def forward_custom_op(
        self,
        mla: "MLA",
        hidden_states: "torch.Tensor",
        position_ids: Optional["torch.Tensor"],
        attn_output: list["torch.Tensor"],
        latent_cache_gen: Optional["torch.Tensor"],
    ) -> bool:
        """Run a custom-op override and return whether handled."""
        return False

    def project_output(
        self,
        mla: "MLA",
        attn_output: list["torch.Tensor"],
        position_ids: Optional["torch.Tensor"],
        attn_metadata: "AttentionMetadata",
        all_reduce_params: Optional["AllReduceParams"],
    ) -> Optional["torch.Tensor"]:
        """Return an algorithm-specific projection, or ``None`` for the default."""
        return None


class AttentionSparseHooks:
    """Typed module-layer adapter for a sparse Attention algorithm."""

    def initialize(self, attention: "Attention") -> None:
        """Initialize algorithm-specific Attention state."""

    def forward(
        self,
        attention: "Attention",
        q: "torch.Tensor",
        k: Optional["torch.Tensor"],
        v: Optional["torch.Tensor"],
        attn_metadata: "AttentionMetadata",
        attention_mask: "AttentionMask",
        attention_window_size: Optional[int],
        attention_mask_data: Optional["torch.Tensor"],
        mrope_config: Optional[dict[str, object]],
        attention_sinks: Optional["torch.Tensor"],
        relative_attention_bias: Optional["torch.Tensor"],
        relative_attention_max_distance: int,
        has_lora: bool,
        **kwargs: object,
    ) -> Optional["torch.Tensor | tuple[torch.Tensor, torch.Tensor]"]:
        """Return an algorithm-specific forward result, or ``None`` for the default."""
        return None

    def project_output(
        self,
        attention: "Attention",
        attn_output: "torch.Tensor",
        attn_metadata: "AttentionMetadata",
        all_reduce_params: Optional["AllReduceParams"],
        lora_params: Optional[dict[str, object]],
    ) -> Optional["torch.Tensor"]:
        """Return an algorithm-specific projection, or ``None`` for the default."""
        return None


_MLA_HOOK_MODULE_PATHS = {
    "dsa": ".dsa.module",
    "deepseek_v4": ".deepseek_v4.module",
}
_ATTENTION_HOOK_MODULE_PATHS = {
    "rocket": ".rocket.module",
}
_MLA_HOOKS: dict[str, type[MLASparseHooks]] = {}
_ATTENTION_HOOKS: dict[str, type[AttentionSparseHooks]] = {}


def register_mla_sparse_hooks(algorithm: str, hooks: type[MLASparseHooks]) -> None:
    """Register the typed MLA adapter for ``algorithm``."""
    if algorithm in _MLA_HOOKS:
        raise ValueError(f"MLA sparse hooks are already registered for {algorithm!r}")
    _MLA_HOOKS[algorithm] = hooks


def register_attention_sparse_hooks(algorithm: str, hooks: type[AttentionSparseHooks]) -> None:
    """Register the typed Attention adapter for ``algorithm``."""
    if algorithm in _ATTENTION_HOOKS:
        raise ValueError(f"Attention sparse hooks are already registered for {algorithm!r}")
    _ATTENTION_HOOKS[algorithm] = hooks


def _get_sparse_mla_hooks_for_algorithm(
    algorithm: str,
) -> Optional[type[MLASparseHooks]]:
    hooks = _MLA_HOOKS.get(algorithm)
    if hooks is not None:
        return hooks

    module_name = _MLA_HOOK_MODULE_PATHS.get(algorithm)
    if module_name is None:
        return None
    import_module(module_name, package=__package__)
    hooks = _MLA_HOOKS.get(algorithm)
    if hooks is None:
        raise RuntimeError(f"{module_name} did not register MLA sparse hooks for {algorithm!r}")
    return hooks


def get_sparse_mla_hooks(mla: "MLA") -> Optional[MLASparseHooks]:
    """Return the MLA adapter selected by ``mla.sparse_params``."""
    algorithm = getattr(getattr(mla, "sparse_params", None), "algorithm", None)
    if algorithm is None:
        return None
    hooks = _get_sparse_mla_hooks_for_algorithm(algorithm)
    return None if hooks is None else hooks()


def _get_sparse_attention_hooks_for_algorithm(
    algorithm: str,
) -> Optional[type[AttentionSparseHooks]]:
    hooks = _ATTENTION_HOOKS.get(algorithm)
    if hooks is not None:
        return hooks

    module_name = _ATTENTION_HOOK_MODULE_PATHS.get(algorithm)
    if module_name is None:
        return None
    import_module(module_name, package=__package__)
    hooks = _ATTENTION_HOOKS.get(algorithm)
    if hooks is None:
        raise RuntimeError(
            f"{module_name} did not register Attention sparse hooks for {algorithm!r}"
        )
    return hooks


def get_sparse_attention_hooks(attention: "Attention") -> Optional[AttentionSparseHooks]:
    """Return the Attention adapter selected by ``attention.sparse_params``."""
    algorithm = getattr(getattr(attention, "sparse_params", None), "algorithm", None)
    if algorithm is None:
        return None
    hooks = _get_sparse_attention_hooks_for_algorithm(algorithm)
    return None if hooks is None else hooks()


def prepare_sparse_attention_prediction(
    backend: "TrtllmAttention",
    q: "torch.Tensor",
    k: Optional["torch.Tensor"],
    v: Optional["torch.Tensor"],
    metadata: "AttentionMetadata",
    forward_args: "AttentionForwardArgs",
) -> SparseRuntimeParams:
    """Return a precomputed prediction or invoke the backend predictor once."""
    runtime_params = forward_args.sparse_runtime_params
    if runtime_params is not None:
        return runtime_params
    return backend.predict_sparse_attention(q, k, v, metadata, forward_args)
