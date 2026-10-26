# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Clean, importable surface for the CuTe-DSL sparse attention kernels.

The CuTe-DSL sparse implementation lives in the ``cute/`` directory next to
this file. Its top-level modules are loaded via ``sys.path.insert`` rather than
as a real Python subpackage, so historically every caller had to replicate that
``sys.path.insert`` hack (see ``sparse_fmha_adapter.py``) and then
``from interface import ...``.

This module encapsulates that mechanism **once** and re-exports the public
sparse API, so downstream consumers (for example an sglang attention backend)
can simply do::

    from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.fmha_sm100.sparse import (
        fp4_indexer_block_scores,      # block-score indexer (topk is caller-owned)
        build_k2q_csr,                 # q2k indices -> CSR + schedule
        SparseK2qCsrBuilderSm100,      # SM100 CSR builder (fused schedule)
        sparse_atten_func,             # block-sparse prefill
        sparse_atten_nvfp4_kv_func,    # block-sparse prefill, NVFP4 K/V
        sparse_decode_atten_func,      # block-sparse decode (functional)
        SparseDecodePagedAttentionWrapper,  # paged FP8 decode (plan/run)
    )

The convenience aliases are also re-exported lazily from the package root, so
``from tensorrt_llm._torch.attention.backends.fmha.msa_kernels.fmha_sm100 import sparse_atten_func`` works too (see ``__init__.py``).

Importing this module pulls in the CuTe-DSL stack (``nvidia-cutlass-dsl`` etc.)
and is SM100 specific, so keep it off the hot import path
and import it lazily where needed.

NOTE (packaging): the ``sys.path.insert`` below exposes the ``cute/``
top-level modules (``interface``, ``src``, ``quantize`` ...) as importable
top-level names, which can shadow same-named modules elsewhere. The long-term
fix is to make ``cute`` a real subpackage with relative imports;
that is a larger, runtime-validated change tracked separately. This shim keeps
the public import path stable in the meantime and matches the existing
behaviour of ``sparse_fmha_adapter.py``.
"""

from __future__ import annotations

import os
import sys

_MM_SPARSE_DIR = os.path.join(os.path.dirname(__file__), "cute")
if os.path.isdir(_MM_SPARSE_DIR):
    _abs = os.path.abspath(_MM_SPARSE_DIR)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)
else:  # pragma: no cover - only happens on a broken install
    raise ImportError(
        "cute sources not found next to fmha_sm100 "
        f"(looked in {_MM_SPARSE_DIR!r}). The CuTe-DSL sparse kernels ship as "
        "package data; reinstall fmha_sm100 (`pip install -e .`)."
    )

# Sparse attention forward / decode (cute/interface.py).
from interface import (  # noqa: E402
    SparseDecodePagedAttentionWrapper,
    sparse_atten_func,
    sparse_atten_nvfp4_kv_func,
    sparse_decode_atten_func,
)

# CSR + schedule construction (cute/sparse_index_utils.py).
from sparse_index_utils import build_k2q_csr  # noqa: E402

# SM100 fused CSR builder (cute/src/sm100/prepare_k2q_csr.py).
from src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100  # noqa: E402

# FP4 block-score indexer (cute/fp4_indexer_interface.py).
# Returns per-(Hq, kv_block, q) max scores; topK selection + q2k construction
# remain caller-owned downstream steps.
from fp4_indexer_interface import fp4_indexer_block_scores  # noqa: E402

# NVFP4 quantization helpers used to feed the FP4 indexer / NVFP4 attention
# (cute/quantize.py).
from quantize import (  # noqa: E402
    Nvfp4QuantizedTensor,
    dequantize_nvfp4_to_bf16,
    nvfp4_global_scale_from_amax,
    nvfp4_scale_linear_offset,
    nvfp4_scale_swizzle4x4_offset,
    pack_nvfp4_scale,
    quantize_bf16_to_nvfp4,
    quantize_kv_bf16_to_nvfp4,
)

__all__ = [
    # attention
    "sparse_atten_func",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "SparseDecodePagedAttentionWrapper",
    # indexing / CSR
    "fp4_indexer_block_scores",
    "build_k2q_csr",
    "SparseK2qCsrBuilderSm100",
    # nvfp4 quantization helpers
    "Nvfp4QuantizedTensor",
    "quantize_bf16_to_nvfp4",
    "quantize_kv_bf16_to_nvfp4",
    "dequantize_nvfp4_to_bf16",
    "pack_nvfp4_scale",
    "nvfp4_scale_linear_offset",
    "nvfp4_scale_swizzle4x4_offset",
    "nvfp4_global_scale_from_amax",
]
