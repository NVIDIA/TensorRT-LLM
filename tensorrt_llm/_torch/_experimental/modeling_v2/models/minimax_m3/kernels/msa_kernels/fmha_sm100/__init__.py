# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""FMHA varlen attention for SM100.

Provides fmha_sm100_plan() and fmha_sm100() interfaces
with per-variant lazy JIT compilation.
"""

from .api import fmha_sm100, fmha_sm100_plan, sparse_topk_select

# Public symbols of the CuTe-DSL sparse stack (implemented in fmha_sm100.sparse).
# Re-exported lazily so a bare ``import fmha_sm100`` does not pull in the
# nvidia-cutlass-dsl runtime; they are resolved on first attribute access.
_SPARSE_LAZY_EXPORTS = frozenset(
    {
        "sparse_atten_func",
        "sparse_atten_nvfp4_kv_func",
        "sparse_decode_atten_func",
        "SparseDecodePagedAttentionWrapper",
        "fp4_indexer_block_scores",
        "build_k2q_csr",
        "SparseK2qCsrBuilderSm100",
    }
)

__all__ = [
    "fmha_sm100_plan",
    "fmha_sm100",
    "sparse_topk_select",
    *sorted(_SPARSE_LAZY_EXPORTS),
]


def __getattr__(name):
    # PEP 562 module-level hook: resolve sparse symbols on first access by
    # importing the fmha_sm100.sparse shim (which loads the CuTe-DSL stack).
    if name in _SPARSE_LAZY_EXPORTS:
        from . import sparse as _sparse

        return getattr(_sparse, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_SPARSE_LAZY_EXPORTS})

# TensorRT-LLM: this copy registers no global TVM-FFI functions. The packaged top-level fmha_sm100
# registers the minfer.ops.* names, and a second registration in the same process would fail.
