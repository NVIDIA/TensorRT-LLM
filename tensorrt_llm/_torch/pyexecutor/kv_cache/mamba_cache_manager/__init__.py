# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable, lazily loaded exports for Mamba cache managers.

The former module is now a package. Lazy exports preserve its import surface
without making model-owned feature modules depend on manager import order.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import (
        BaseMambaCacheManager,
        MambaAuxCacheExtension,
        MambaCacheBuildContext,
        MambaCacheFeatures,
        MambaHybridCacheManager,
        MambaLayerCache,
        MambaRole,
        MambaStateUpdateBatch,
        MambaStateUpdateStrategy,
        ReplayStateUpdateMetadata,
        SpeculativeMambaLayerCache,
    )
    from .legacy import (
        CppMambaHybridCacheManager,
        MambaCacheManager,
        MixedMambaHybridCacheManager,
        PythonMambaCacheManager,
    )
    from .mamba_cache_manager_v2 import MambaHybridCacheManagerV2

_EXPORTS = {
    "BaseMambaCacheManager": ("common", "BaseMambaCacheManager"),
    "CppMambaHybridCacheManager": ("legacy", "CppMambaHybridCacheManager"),
    "MIN_REPLAY_HISTORY_SIZE": ("common", "MIN_REPLAY_HISTORY_SIZE"),
    "MambaAuxCacheExtension": ("common", "MambaAuxCacheExtension"),
    "MambaCacheBuildContext": ("common", "MambaCacheBuildContext"),
    "MambaCacheFeatures": ("common", "MambaCacheFeatures"),
    "MambaCacheManager": ("legacy", "MambaCacheManager"),
    "MambaHybridCacheManager": ("common", "MambaHybridCacheManager"),
    "MambaHybridCacheManagerV2": (
        "mamba_cache_manager_v2",
        "MambaHybridCacheManagerV2",
    ),
    "MambaLayerCache": ("common", "MambaLayerCache"),
    "MambaRole": ("common", "MambaRole"),
    "MambaStateUpdateBatch": ("common", "MambaStateUpdateBatch"),
    "MambaStateUpdateStrategy": ("common", "MambaStateUpdateStrategy"),
    "MixedMambaHybridCacheManager": ("legacy", "MixedMambaHybridCacheManager"),
    "PythonMambaCacheManager": ("legacy", "PythonMambaCacheManager"),
    "ReplayStateUpdateMetadata": ("common", "ReplayStateUpdateMetadata"),
    "SpeculativeMambaLayerCache": ("common", "SpeculativeMambaLayerCache"),
    "_advance_replay_state": ("common", "_advance_replay_state"),
    "_allocate_mamba_seed_buffer": ("common", "_allocate_mamba_seed_buffer"),
    "_compute_deterministic_mamba_seed": (
        "common",
        "_compute_deterministic_mamba_seed",
    ),
    "_get_local_mamba_cache_layout": ("common", "_get_local_mamba_cache_layout"),
    "_get_mamba_hybrid_pool_size": ("common", "_get_mamba_hybrid_pool_size"),
    "_get_num_cuda_graph_padding_dummy_slots": (
        "common",
        "_get_num_cuda_graph_padding_dummy_slots",
    ),
    "_mamba_effective_tp_size": ("common", "_mamba_effective_tp_size"),
    "_mamba_rank_offset": ("common", "_mamba_rank_offset"),
    "_mamba_snapshot_rule_counts": ("common", "_mamba_snapshot_rule_counts"),
    "_promote_mamba_state_triton": ("common", "_promote_mamba_state_triton"),
    "use_py_mamba_cache_manager": ("common", "use_py_mamba_cache_manager"),
}


def __getattr__(name: str):
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = export
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = [
    "BaseMambaCacheManager",
    "CppMambaHybridCacheManager",
    "MIN_REPLAY_HISTORY_SIZE",
    "MambaAuxCacheExtension",
    "MambaCacheBuildContext",
    "MambaCacheFeatures",
    "MambaCacheManager",
    "MambaHybridCacheManager",
    "MambaHybridCacheManagerV2",
    "MambaLayerCache",
    "MambaRole",
    "MambaStateUpdateBatch",
    "MambaStateUpdateStrategy",
    "MixedMambaHybridCacheManager",
    "PythonMambaCacheManager",
    "ReplayStateUpdateMetadata",
    "SpeculativeMambaLayerCache",
    "use_py_mamba_cache_manager",
]
