# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
"""Lazy QSA exports; importing geometry does not construct the cache stack."""

from importlib import import_module

_EXPORTS = {
    "QSATrtllmAttention": "backend",
    "QSAAttentionMetadata": "metadata",
    "QSASparseMetadataParams": "params",
    "QSASparseParams": "params",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
=======
from .backend import QSATrtllmAttention
from .cache_manager import QSAMambaHybridCacheManagerV2
from .metadata import QSAMetadata
from .params import QSASparseMetadataParams, QSASparseParams

__all__ = [
    "QSAMambaHybridCacheManagerV2",
    "QSAMetadata",
    "QSASparseMetadataParams",
    "QSASparseParams",
    "QSATrtllmAttention",
]
>>>>>>> e7e42e77dd ([None][feat] Attach feature and sparse metadata state as extensions)
