# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuteDSL MegaMoE NVFP4 kernel package.

Hosts the ported MegaMoE fused dispatch + FC1 + activation + FC2 + combine
CuteDSL kernel (flattened from the upstream ``moe_nvfp4_swapab/`` + ``src/``
split). The package is loaded
lazily by :mod:`tensorrt_llm._torch.modules.fused_moe.mega_moe.mega_moe_cute_dsl`
through :func:`import_kernel` so environments without a CUDA 13 Cutlass DSL
runtime can still import the backend file for capability probing.

NOTE: Top-level eager import of ``Sm100MegaMoEKernel`` would pull every
CuteDSL kernel symbol the moment any caller does
``from tensorrt_llm._torch.cute_dsl_kernels.mega_moe_nvfp4 import ...``.
The current tests only check capability (no kernel launch); keep the
public surface lazy so unit tests on non-SM100 boxes do not trigger
CuteDSL import side effects.
"""

from __future__ import annotations

from .blocked_scale import (SfPaddingBlock, cat_byte_reinterpretable_tensors,
                            ceil_div, from_blocked,
                            stack_byte_reinterpretable_tensors, to_blocked)
from .megamoe_constants import (Fp8E4M3FNMax, Nvfp4BlockSize, Nvfp4E2M1Max,
                                SupportedMmaTileM, SupportedMmaTileN,
                                TmaLeadingDimByteAlign)

__all__ = [
    "Fp8E4M3FNMax",
    "Nvfp4BlockSize",
    "Nvfp4E2M1Max",
    "SfPaddingBlock",
    "SupportedMmaTileM",
    "SupportedMmaTileN",
    "TmaLeadingDimByteAlign",
    "cat_byte_reinterpretable_tensors",
    "ceil_div",
    "from_blocked",
    "import_kernel",
    "import_sym_buffer_host",
    "import_topk_reduce",
    "stack_byte_reinterpretable_tensors",
    "to_blocked",
]


def import_kernel():
    """Lazily import the CuteDSL kernel class.

    Returns the ``Sm100MegaMoEKernel`` class. Raises ``ImportError`` with
    an actionable message if the active CUDA 13 Cutlass DSL runtime does
    not expose every symbol the kernel needs. Callers must catch the
    error and report capability negatively (e.g. ``can_implement`` /
    backend skip), not crash the import of the wrapper module.
    """
    from .megamoe_kernel import Sm100MegaMoEKernel

    return Sm100MegaMoEKernel


def import_sym_buffer_host():
    """Lazily import the symmetric-buffer host wrapper.

    Returns the ``SymBufferHost`` factory function from
    :mod:`.sym_buffer`. See its module docstring for the runtime
    contract; the caller is responsible for supplying a peer-pointer
    provider (NVSHMEM-equivalent) on multi-rank execution.
    """
    from . import sym_buffer

    # SymBufferHost lives at module scope as a factory; the upstream API
    # constructs the per-world-size variant inside sym_buffer.py.
    return sym_buffer


def import_topk_reduce():
    """Lazily import the standalone CuteDSL top-k reduce kernel API.

    Returns ``(compile_topk_reduce, launch_compiled_topk_reduce)`` from
    :mod:`.topk_reduce` (mirrors :func:`import_kernel`). The reduce kernel
    is only needed by the opt-in transformers graph
    (``apply_topk_in_fc1=False``); the deepgemm-default route reduces on
    the host via ``combine_output.sum(dim=1)`` and never imports it. Like
    ``import_kernel`` this stays lazy so non-SM100 / no-cutlass-dsl
    environments can import the backend for capability probing without
    pulling the heavyweight CuteDSL symbols.
    """
    from .topk_reduce import compile_topk_reduce, launch_compiled_topk_reduce

    return compile_topk_reduce, launch_compiled_topk_reduce
