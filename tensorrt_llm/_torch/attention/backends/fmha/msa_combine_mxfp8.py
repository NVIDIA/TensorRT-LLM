# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call-owned MXFP8 output of the MiniMax-M3 NVFP4 sparse prefill combine."""

from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any, Optional

import torch

from tensorrt_llm._torch.utils import MxFp8QuantizedTensor
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from ..sparse.minimax_m3.common import MiniMaxM3BackendForwardArgs

ENABLED = os.environ.get("TRTLLM_CTX_COMBINE_MXFP8", "1") != "0"
SKIP_BF16 = os.environ.get("TRTLLM_CTX_COMBINE_MXFP8_SKIP_BF16", "1") != "0"
_takes: dict = {}


def _sparse_takes_o_fp8(sparse: Any) -> bool:
    func = sparse.sparse_atten_nvfp4_kv_func
    if func not in _takes:
        params = inspect.signature(func).parameters
        _takes[func] = all(name in params for name in ("o_fp8", "sf_out", "write_bf16"))
    return _takes[func]


def o_proj_accepts_mxfp8(o_proj: Any) -> bool:
    """Whether this projection consumes an MXFP8 carrier directly on the current path."""
    from tensorrt_llm._torch.modules import linear

    method = getattr(o_proj, "quant_method", None)
    return (
        isinstance(method, linear.MXFP8LinearMethod)
        and bool(getattr(method, "use_cutlass", False))
        and not linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
    )


def combine_kwargs(
    sparse: Any,
    out: torch.Tensor,
    metadata: Any,
    in_place: bool,
    request: Optional[MiniMaxM3BackendForwardArgs] = None,
) -> Optional[dict]:
    """Allocate MXFP8 outputs for an explicitly requested, complete eager context step."""
    if not ENABLED or request is None or not request.mxfp8_output:
        return None
    from tensorrt_llm._torch.modules.linear import ctx_eager_only

    if not ctx_eager_only() or not in_place or getattr(metadata, "num_generations", 1) != 0:
        return None
    if (
        out.dim() != 3
        or out.shape[-1] != 128
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or not _sparse_takes_o_fp8(sparse)
    ):
        return None
    rows, cols = out.shape[0], out.shape[1] * out.shape[2]
    padded_rows = (rows + 127) // 128 * 128
    o_fp8 = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=out.device)
    sf = torch.empty((padded_rows * cols // 32,), dtype=torch.uint8, device=out.device)
    if padded_rows != rows:
        tile = 128 * cols // 32
        sf[-tile:].zero_()
    return {
        "o_fp8": o_fp8,
        "sf_out": sf,
        "write_bf16": not (SKIP_BF16 and request.mxfp8_skip_bf16),
    }


def quantized_output(out: torch.Tensor, kwargs: dict) -> MxFp8QuantizedTensor:
    """Return the buffers this combine wrote, retaining BF16 only when it was produced."""
    act = kwargs["o_fp8"]
    return MxFp8QuantizedTensor(
        act,
        kwargs["sf_out"],
        is_sf_swizzled=True,
        unquantized_hidden_states=out.view(act.shape) if kwargs["write_bf16"] else None,
        producer_stream=torch.cuda.current_stream(act.device) if act.is_cuda else None,
    )


def log_consumed(input: torch.Tensor | MxFp8QuantizedTensor) -> None:
    """Log completed explicit consumption without retaining a request or its buffers."""
    if isinstance(input, MxFp8QuantizedTensor):
        logger.info_once(
            "[M3 CTX] combine MXFP8 o_proj input taken", key="m3_ctx_combine_mxfp8_taken"
        )
        if input.unquantized_hidden_states is None:
            logger.info_once(
                "[M3 CTX] combine MXFP8 o_proj input taken (bf16 store skipped)",
                key="m3_ctx_combine_mxfp8_skip_bf16_taken",
            )
