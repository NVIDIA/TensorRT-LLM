# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel dispatch for the in-tree Kimi K3 Attention Residual fused op.

The optimized ``attn_res_fwd`` kernel (SM100 CuTe TMA warp-specialised
online-softmax + RMSNorm, Blackwell sm_100/sm_103 only) is source-integrated
into TensorRT-LLM as the ``trtllm::attn_res_fwd`` Torch op
(``cpp/tensorrt_llm/kernels/kimiK3AttnRes`` + ``cpp/tensorrt_llm/thop/attnResOp.cpp``).

Dispatch: on sm_100/sm_103 with the compiled Torch bindings loaded the
module uses the fused op; otherwise it falls back to the pure-torch chunked
reference in :mod:`kimi_k3_attn_res.kimi_k3_attn_res`.
"""

from __future__ import annotations

import torch

try:
    from tensorrt_llm._utils import get_sm_version as _tllm_get_sm_version
except ImportError:  # pragma: no cover — source-loader stub path
    _tllm_get_sm_version = None


def _default_get_sm_version() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return -1
    prop = torch.cuda.get_device_properties(0)
    return prop.major * 10 + prop.minor


def get_attn_res_sm_version() -> int:
    """Return the runtime SM version used for kernel selection.

    Prefers ``tensorrt_llm._utils.get_sm_version`` when the real package is
    importable so environment-side overrides propagate. Falls back to a
    plain CUDA-property probe otherwise.
    """
    if _tllm_get_sm_version is not None:
        try:
            return int(_tllm_get_sm_version())
        except RuntimeError:
            # torch raises RuntimeError when no CUDA device is usable;
            # the property probe below handles that case itself.
            return _default_get_sm_version()
    return _default_get_sm_version()


def is_attn_res_optimized_supported() -> bool:
    """The optimized ``attn_res_fwd`` kernel is Blackwell sm_100 only."""
    return get_attn_res_sm_version() in (100, 103)


def is_intree_attn_res_available() -> bool:
    """True when the in-tree ``trtllm::attn_res_fwd`` Torch op is registered.

    The op is registered when TensorRT-LLM's compiled Torch bindings
    (``libth_common``) are loaded, which happens on ``import tensorrt_llm``.
    Under the source-loader stub subtree the bindings may be absent — then
    this returns False and callers fall back to the reference (or the legacy
    external-loader) path.
    """
    try:
        torch.ops.trtllm.attn_res_fwd  # noqa: B018 — probe schema lookup
        return True
    except (AttributeError, RuntimeError):
        return False


def intree_attn_res_fwd(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
):
    """Run the in-tree fused op. Returns ``(output, rsigma, probs, logits)``."""
    return torch.ops.trtllm.attn_res_fwd(
        layer_residual,
        block_residual,
        res_weight.reshape(-1).contiguous(),
        rms_weight.contiguous(),
        float(rms_eps),
    )
