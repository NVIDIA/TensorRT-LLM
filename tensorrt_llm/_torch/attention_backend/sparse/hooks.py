# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module and backend hooks for sparse attention algorithms.

Algorithms implement only the hooks they need in
``sparse/<algorithm>/module.py``. This module validates those hooks and owns
module dispatch. Backend prediction hooks use the backend subclass directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import import_module
from inspect import Parameter, signature
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    import torch

    from ..interface import AttentionForwardArgs, AttentionMetadata
    from ..trtllm import TrtllmAttention
    from .params import SparseRuntimeParams

SparseAttnHook = Callable[..., object]

__all__ = [
    "SparseAttnHooks",
    "get_sparse_attn_hooks",
    "prepare_sparse_runtime_params",
]

_SPARSE_ATTN_HOOK_MODULE_PATHS = {
    "rocket": ".rocket.module",
    "dsa": ".dsa.module",
    "deepseek_v4": ".deepseek_v4.module",
}
_HOOK_PARAMETER_NAMES = {
    "initialize_sparse_attn": (
        "module",
        "config",
        "mapping",
        "mapping_o",
        "rms_norm_eps",
        "quant_config",
        "q_scaling",
        "bias",
        "dtype",
        "reduce_output",
        "aux_stream",
    ),
    "create_sparse_attn_weights": ("module",),
    "transform_sparse_attn_weights": ("module",),
    "prepare_sparse_attn_outputs": ("module", "hidden_states", "attn_metadata"),
    "forward_sparse_attn": (
        "module",
        "position_ids",
        "hidden_states",
        "attn_metadata",
        "attn_output",
    ),
    "forward_sparse_attn_custom_op": (
        "module",
        "hidden_states",
        "position_ids",
        "attn_output",
        "latent_cache_gen",
    ),
    "project_sparse_attn_output": (
        "module",
        "attn_output",
        "position_ids",
        "attn_metadata",
        "all_reduce_params",
    ),
}
_ALTERNATE_HOOK_PARAMETER_NAMES = {
    "forward_sparse_attn": (
        (
            "module",
            "q",
            "k",
            "v",
            "attn_metadata",
            "attention_mask",
            "attention_window_size",
            "attention_mask_data",
            "mrope_config",
            "attention_sinks",
            "relative_attention_bias",
            "relative_attention_max_distance",
            "has_lora",
            "kwargs",
        ),
    ),
    "project_sparse_attn_output": (
        (
            "module",
            "attn_output",
            "attn_metadata",
            "all_reduce_params",
            "lora_params",
        ),
    ),
}
_INITIALIZE_KEYWORD_ONLY_PARAMETERS = frozenset(_HOOK_PARAMETER_NAMES["initialize_sparse_attn"][1:])


def _get_hook(
    module: ModuleType,
    hook_name: str,
    *,
    algorithm: str,
) -> Optional[SparseAttnHook]:
    hook = getattr(module, hook_name, None)
    if hook is None:
        return None
    if not callable(hook):
        raise TypeError(
            f"Sparse attention hook {algorithm!r}.{hook_name} must be callable, "
            f"got {type(hook).__name__}"
        )

    parameters = tuple(signature(hook).parameters.values())
    parameter_names = tuple(
        "module" if index == 0 and parameter.name in ("self", "mla") else parameter.name
        for index, parameter in enumerate(parameters)
    )
    expected_names = (
        _HOOK_PARAMETER_NAMES[hook_name],
        *_ALTERNATE_HOOK_PARAMETER_NAMES.get(hook_name, ()),
    )
    if parameter_names not in expected_names:
        raise TypeError(
            f"Sparse attention hook {algorithm!r}.{hook_name} has parameters {parameter_names}; "
            f"expected one of {expected_names}"
        )
    if hook_name == "initialize_sparse_attn":
        invalid_keyword_only = [
            parameter.name
            for parameter in parameters
            if parameter.name in _INITIALIZE_KEYWORD_ONLY_PARAMETERS
            and parameter.kind != Parameter.KEYWORD_ONLY
        ]
        if invalid_keyword_only:
            raise TypeError(
                f"Sparse attention hook {algorithm!r}.{hook_name} must declare these parameters "
                f"as keyword-only: {', '.join(invalid_keyword_only)}"
            )
    if parameter_names and parameter_names[-1] == "kwargs":
        if parameters[-1].kind != Parameter.VAR_KEYWORD:
            raise TypeError(
                f"Sparse attention hook {algorithm!r}.{hook_name} must declare 'kwargs' as **kwargs"
            )
    return hook


@dataclass(frozen=True)
class SparseAttnHooks:
    """Validated module-layer hooks for one sparse attention algorithm."""

    algorithm: Optional[str] = None
    initialize_sparse_attn: Optional[SparseAttnHook] = None
    create_sparse_attn_weights: Optional[SparseAttnHook] = None
    transform_sparse_attn_weights: Optional[SparseAttnHook] = None
    prepare_sparse_attn_outputs: Optional[SparseAttnHook] = None
    forward_sparse_attn: Optional[SparseAttnHook] = None
    forward_sparse_attn_custom_op: Optional[SparseAttnHook] = None
    project_sparse_attn_output: Optional[SparseAttnHook] = None

    def __bool__(self) -> bool:
        """Return whether sparse attention is configured for the module."""
        return self.algorithm is not None

    def require(self, hook_name: str) -> SparseAttnHook:
        """Return an implemented hook required by the current module path."""
        if hook_name not in _HOOK_PARAMETER_NAMES:
            raise ValueError(f"Unknown sparse attention hook {hook_name!r}")
        hook = getattr(self, hook_name)
        if hook is None:
            raise NotImplementedError(
                f"Sparse attention algorithm {self.algorithm!r} does not implement "
                f"the {hook_name!r} hook required by this module path"
            )
        return hook

    @classmethod
    def from_module(cls, algorithm: str, module: ModuleType) -> "SparseAttnHooks":
        """Validate and adapt an algorithm ``module.py`` to the hook contract."""
        return cls(
            algorithm=algorithm,
            initialize_sparse_attn=_get_hook(module, "initialize_sparse_attn", algorithm=algorithm),
            create_sparse_attn_weights=_get_hook(
                module, "create_sparse_attn_weights", algorithm=algorithm
            ),
            transform_sparse_attn_weights=_get_hook(
                module, "transform_sparse_attn_weights", algorithm=algorithm
            ),
            prepare_sparse_attn_outputs=_get_hook(
                module, "prepare_sparse_attn_outputs", algorithm=algorithm
            ),
            forward_sparse_attn=_get_hook(module, "forward_sparse_attn", algorithm=algorithm),
            forward_sparse_attn_custom_op=_get_hook(
                module, "forward_sparse_attn_custom_op", algorithm=algorithm
            ),
            project_sparse_attn_output=_get_hook(
                module, "project_sparse_attn_output", algorithm=algorithm
            ),
        )


_EMPTY_SPARSE_ATTN_HOOKS = SparseAttnHooks()


@lru_cache(maxsize=None)
def _get_sparse_attn_hooks_for_algorithm(algorithm: str) -> SparseAttnHooks:
    module_name = _SPARSE_ATTN_HOOK_MODULE_PATHS.get(algorithm)
    if module_name is None:
        return SparseAttnHooks(algorithm=algorithm)
    module = import_module(module_name, package=__package__)
    return SparseAttnHooks.from_module(algorithm, module)


def get_sparse_attn_hooks(module) -> SparseAttnHooks:
    """Return hooks selected by ``module.sparse_params`` or an empty hook set."""
    algorithm = getattr(getattr(module, "sparse_params", None), "algorithm", None)
    if algorithm is None:
        return _EMPTY_SPARSE_ATTN_HOOKS
    return _get_sparse_attn_hooks_for_algorithm(algorithm)


def prepare_sparse_runtime_params(
    backend: "TrtllmAttention",
    q: "torch.Tensor",
    k: Optional["torch.Tensor"],
    metadata: "AttentionMetadata",
    forward_args: "AttentionForwardArgs",
) -> "SparseRuntimeParams":
    """Run backend prediction hooks and update attention-op parameters."""
    runtime_params = forward_args.sparse_runtime_params
    if backend.sparse_params is None:
        return runtime_params

    kv_indices, kv_offsets = backend.sparse_kv_predict(q, k, metadata, forward_args)
    attn_indices, attn_offsets = backend.sparse_attn_predict(q, k, metadata, forward_args)
    block_size = (
        backend.sparse_params.indices_block_size
        if attn_indices is not None or attn_offsets is not None
        else runtime_params.sparse_attn_indices_block_size
    )
    return replace(
        runtime_params,
        sparse_kv_indices=kv_indices,
        sparse_kv_offsets=kv_offsets,
        sparse_attn_indices=attn_indices,
        sparse_attn_offsets=attn_offsets,
        sparse_attn_indices_block_size=block_size,
    )
