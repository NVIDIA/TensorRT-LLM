# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode step's pre-layer chain in one launch.

Part of the ``layer0_qkv_table`` fast path (``kernels/tt_table/target_l0``, which builds its
table with this kernel and runs it beside the table's gather for the residual stream). The chain it
stands for is four kernels between the token ids and layer 0's QKV projection: the replicated
embedding gather (``ops/embedding/replicated_gather``), the weightless embedding RMSNorm, layer 0's
gemma pre-QKV RMSNorm of its BF16 output (the attention input) and the exact FP32 widening of that
output (the residual stream); ``DecodeTarget`` builds the same two outputs through
``model_boundary`` or its built-in chain. ``pf_pre.cu`` is the four in one kernel, one
CTA per row, with the same floating-point operations in the same order (its header has the
statements), so both outputs are the built-in chain's bits: value-identical by construction,
certified bitwise against the built-in kernels by the unit test.

The fused call applies only where the built-in chain runs exactly the kernels it reproduces
(``unsupported_reason``): the replicated table, a weightless ``RMSNorm`` and a gemma ``RMSNorm``
that reach flashinfer's CuTe-DSL ``RMSNormKernel``, ``m_emb == 1`` and an 8192-wide BF16 model.
Every rank answers alike (construction-time state, shapes and dtypes only).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import torch

#: Whether the fused pre stands in for the pre-layer chain (read by ``kernels/tt_table/target_l0``,
#: which the shell installs only with the ``layer0_qkv_table`` fast path).
FUSED_PRE = True
#: The fused kernel's variant (``pf_pre.cu``'s header): 0 the base kernel; 1 the norm
#: weights loaded before the dependency wait; 2 as 1 with each row split over a cluster of four
#: one-warp CTAs. Every variant computes the same values with the same operations.
PRE_VARIANT = 1
#: The row width the kernel is written for (``kHidden``).
HIDDEN = 8192
#: Largest call (ids) the fused kernel takes: the replicated embedding's limit.
MAX_ROWS = 576
LIBRARY_NAME = "musespark_open_pf_pre"
SOURCE = Path(__file__).resolve().parent / "pf_pre.cu"
NVCC_FLAGS = ("-O3", "-lineinfo")

_LOCK = threading.Lock()
_LOADED: Optional[str] = None


def arch_target() -> str:
    capability = torch.cuda.get_device_capability()
    if capability == (10, 3):
        return "103a"
    if capability == (10, 7):
        return "107a"
    raise RuntimeError(
        f"the pf_pre kernel is built for sm_103a / sm_107a, not sm_{capability[0]}{capability[1]}"
    )


def load_library() -> str:
    """Build (if needed) and load the library once per process; its path."""
    global _LOADED
    with _LOCK:
        if _LOADED is None:
            from torch.utils.cpp_extension import load

            target = arch_target()
            path = load(
                name=f"{LIBRARY_NAME}_sm{target}",
                sources=[str(SOURCE)],
                extra_cuda_cflags=list(NVCC_FLAGS)
                + [f"-gencode=arch=compute_{target},code=sm_{target}"],
                is_python_module=False,
                verbose=False,
            )
            _LOADED = str(path) if path else LIBRARY_NAME
        return _LOADED


def _norm_reason(norm, gemma: bool) -> Optional[str]:
    """Why ``norm`` does not run flashinfer's CuTe-DSL ``RMSNormKernel`` on an 8192-wide BF16 row
    (weight_bias 1 when ``gemma``, else 0), or None."""
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    if type(norm) is not RMSNorm:
        return f"{type(norm).__name__} is not the built-in RMSNorm"
    if bool(getattr(norm, "use_gemma", False)) != gemma:
        return f"use_gemma is {getattr(norm, 'use_gemma', None)}, expected {gemma}"
    if getattr(norm, "use_cuda_tile", False):
        return "the norm runs the cuda.tile kernel"
    if getattr(norm, "is_nvfp4", False) or getattr(norm, "return_hp_output", False):
        return "the norm has an NVFP4 epilogue"
    weight = getattr(norm, "weight", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dtype != torch.bfloat16
        or tuple(weight.shape) != (HIDDEN,)
        or not weight.is_contiguous()
    ):
        return "the norm weight is not a contiguous [8192] BF16 tensor"
    return None


def unsupported_reason(core) -> Optional[str]:
    """Why the fused pre cannot stand in for ``core``'s decode pre-layer chain, or None (rank-uniform)."""
    from ... import attention as _attention
    from ..replicated_embedding.embedding import table_of

    table = table_of(getattr(core, "embed_tokens", None))
    if table is None:
        return "the token embedding is not the replicated table"
    if table.dtype != torch.bfloat16 or table.dim() != 2 or table.shape[1] != HIDDEN:
        return "the table is not [vocab, 8192] BF16"
    if getattr(core, "dtype", None) != torch.bfloat16:
        return "the model dtype is not BF16"
    if getattr(core, "m_emb", None) != 1.0:
        return f"m_emb is {getattr(core, 'm_emb', None)}, not 1"
    reason = _norm_reason(getattr(core, "embed_norm", None), gemma=False)
    if reason is not None:
        return f"embed_norm: {reason}"
    reason = _norm_reason(getattr(core.layers[0], "pre_qkv_proj_norm", None), gemma=True)
    if reason is not None:
        return f"layer 0 pre_qkv_proj_norm: {reason}"
    reason = _attention._builtin_prep_kernel_mismatch()
    if reason is not None:
        return reason
    return None


def takes(input_ids: Optional[torch.Tensor]) -> bool:
    """Whether a call with these ids fits the kernel (rank-uniform: shape and dtype only)."""
    return (
        isinstance(input_ids, torch.Tensor)
        and input_ids.is_cuda
        and input_ids.dtype in (torch.int32, torch.int64)
        and 1 <= input_ids.numel() <= MAX_ROWS
    )


def fused_pre(
    core,
    input_ids: torch.Tensor,
    pdl: bool = True,
    variant: Optional[int] = None,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(layer0_input, hidden_states)``: layer 0's BF16 pre-QKV norm and the FP32 residual of
    the normalized embedding of ``input_ids``, as the pre-layer chain computes them.

    ``variant``: the kernel variant (``PRE_VARIANT`` when None). ``out``: the two outputs to write
    (contiguous ``[rows, 8192]`` BF16 and FP32), e.g. allocated on another stream than the launch's;
    fresh tensors when None."""
    from ..replicated_embedding.embedding import table_of

    load_library()
    table = table_of(core.embed_tokens)
    ids = input_ids.reshape(-1).contiguous()
    rows = ids.numel()
    if out is None:
        attn = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=ids.device)
        residual = torch.empty((rows, HIDDEN), dtype=torch.float32, device=ids.device)
    else:
        attn, residual = out
        if not (
            attn.dtype == torch.bfloat16
            and residual.dtype == torch.float32
            and tuple(attn.shape) == (rows, HIDDEN)
            and tuple(residual.shape) == (rows, HIDDEN)
            and attn.is_contiguous()
            and residual.is_contiguous()
        ):
            raise ValueError("pf_pre: out must be contiguous [rows, 8192] BF16 and FP32 tensors")
    qkv_norm = core.layers[0].pre_qkv_proj_norm
    variant = PRE_VARIANT if variant is None else int(variant)
    args = (
        table,
        ids,
        core.embed_norm.weight,
        float(core.embed_norm.variance_epsilon),
        0.0,
        qkv_norm.weight,
        float(qkv_norm.variance_epsilon),
        1.0,
        attn,
        residual,
        bool(pdl),
    )
    if variant == 0:
        torch.ops.musespark_open_pf_pre.pre(*args)
    else:
        torch.ops.musespark_open_pf_pre.pre_variant(*args, variant)
    return attn, residual
