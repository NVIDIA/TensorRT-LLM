# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family DSA metadata tests' step objects and the family's index-key layout."""

import inspect
import types

#: The family's index keys: 128 values per key, one 4-byte scale word per key, 64-token pages.
HEAD_DIM = 128
PAGE = 64
SCALE = 4


def bare(cls, **attrs):
    """A ``cls`` metadata object without the engine's constructor. Read-only engine properties
    among ``attrs`` (step counts, ``seq_lens_cuda``, ...) become plain attributes of a subclass."""
    props = {
        k: v for k, v in attrs.items() if isinstance(inspect.getattr_static(cls, k, None), property)
    }
    if props:
        cls = type(f"_Step{cls.__name__}", (cls,), props)
    obj = object.__new__(cls)
    for key, value in attrs.items():
        if key not in props:
            setattr(obj, key, value)
    return obj


def index_key_manager(index_k: str, **extra):
    """The index-key facts of the family cache manager (``indexer_k_dtype`` and ``use_fp4``)."""
    return types.SimpleNamespace(
        index_head_dim=HEAD_DIM, indexer_k_dtype=index_k, use_fp4=index_k == "fp4", **extra
    )


def data_bytes(index_k: str) -> int:
    """Bytes of an index key's codes (FP4 packs two values per byte)."""
    return HEAD_DIM // 2 if index_k == "fp4" else HEAD_DIM


def index_key_slots(index_k: str, block: int, in_page: int):
    """The (codes, scale) byte offsets of a key at ``in_page`` of page ``block``."""
    data = data_bytes(index_k)
    stride = PAGE * (data + SCALE)
    return block * stride + in_page * data, block * stride + PAGE * data + in_page * SCALE
