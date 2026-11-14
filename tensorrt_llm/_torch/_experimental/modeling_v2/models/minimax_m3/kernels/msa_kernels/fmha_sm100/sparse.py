# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Clean, importable surface for the CuTe-DSL sparse attention kernels.

The CuTe-DSL sparse implementation lives in the ``cute/`` subpackage next to
this file. This module re-exports its public sparse API::

    from <this package>.sparse import (
        fp4_indexer_block_scores,      # block-score indexer (topk is caller-owned)
        build_k2q_csr,                 # q2k indices -> CSR + schedule
        SparseK2qCsrBuilderSm100,      # SM100 CSR builder (fused schedule)
        sparse_atten_func,             # block-sparse prefill
        sparse_atten_nvfp4_kv_func,    # block-sparse prefill, NVFP4 K/V
        sparse_decode_atten_func,      # block-sparse decode (functional)
        SparseDecodePagedAttentionWrapper,  # paged FP8 decode (plan/run)
    )

The convenience aliases are also re-exported lazily from the package root
(see ``__init__.py``).

Importing this module pulls in the CuTe-DSL stack (``nvidia-cutlass-dsl`` etc.)
and is SM100 specific, so keep it off the hot import path
and import it lazily where needed.

TensorRT-LLM: ``cute`` is a regular subpackage imported with relative imports,
so this copy never adds directories to ``sys.path`` and never shares top-level
module names (``interface``, ``src`` ...) with another copy of the sources.
"""

from __future__ import annotations

# Sparse attention forward / decode (cute/interface.py).
from .cute.interface import (
    SparseDecodePagedAttentionWrapper,
    sparse_atten_func,
    sparse_atten_nvfp4_kv_func,
    sparse_decode_atten_func,
)

# CSR + schedule construction (cute/sparse_index_utils.py).
from .cute.sparse_index_utils import build_k2q_csr

# SM100 fused CSR builder (cute/src/sm100/prepare_k2q_csr.py).
from .cute.src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100

# FP4 block-score indexer (cute/fp4_indexer_interface.py).
# Returns per-(Hq, kv_block, q) max scores; topK selection + q2k construction
# remain caller-owned downstream steps.
from .cute.fp4_indexer_interface import fp4_indexer_block_scores

# NVFP4 quantization helpers used to feed the FP4 indexer / NVFP4 attention
# (cute/quantize.py).
from .cute.quantize import (
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
