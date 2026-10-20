# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 CTX: the NVFP4 sparse prefill's combine also writes the o_proj input as MXFP8.

The combine's epilogue quantizes the final bf16 attention output it is about to store (fmha_sm100 combine
``o_fp8=`` / ``sf_out=``), bitwise ``mxfp8_quantize(o, True)``, and hands the pair to the o_proj through the MXFP8
prequant slot (modules.linear.set_prequantized_mxfp8), so the o_proj skips its own activation quantize.

It engages only where the slot is certainly consumed by an eager o_proj on exactly the combine's rows:
- the attention module armed it (``armed()``) around an eager attention core that is followed by its o_proj;
- a pure context step (no generation rows: the combine then covers every row of the o_proj input);
- the in-place SM107 combine path (the combine writes the o_proj input buffer itself) and an fmha_sm100 that
  accepts ``o_fp8``.
The bf16 output is still written (``write_bf16=True``), so a slot that is not consumed only costs the extra stores;
it is logged once as a miss (``after_o_proj``). Kill switch: TRTLLM_CTX_COMBINE_MXFP8=0.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from typing import Any, Optional

import torch

from tensorrt_llm.logger import logger

ENABLED = os.environ.get("TRTLLM_CTX_COMBINE_MXFP8", "1") != "0"
# Skip the bf16 store too (the o_proj reads only the MXFP8 pair). On by default (mega2; kill: =0). Only where
# o_proj_takes_slot() proved the o_proj consumes the slot; a pair it then does not take raises (its bf16 input was
# never written).
SKIP_BF16 = os.environ.get("TRTLLM_CTX_COMBINE_MXFP8_SKIP_BF16", "1") != "0"
_ENGAGEMENT = "combine MXFP8 o_proj input taken"
_ENGAGEMENT_SKIP = "combine MXFP8 o_proj input taken (bf16 store skipped)"

_armed: list = [None]  # the arming call site's tag (slot audit), None when not armed
_skip_ok: list = [False]  # the armed o_proj consumes the slot, so the bf16 store may be skipped
_pending: list = [None]  # the slot entry this module set and the o_proj has not consumed yet
_takes: dict = {}


def _sparse_takes_o_fp8(sparse: Any) -> bool:
    # Keyed on the function itself (held by the dict), not id(sparse): a freed object's id can be reused.
    func = sparse.sparse_atten_nvfp4_kv_func
    if func not in _takes:
        params = inspect.signature(func).parameters
        _takes[func] = all(name in params for name in ("o_fp8", "sf_out", "write_bf16"))
    return _takes[func]


def o_proj_takes_slot(o_proj: Any) -> bool:
    """Whether ``o_proj`` reads its input only through the MXFP8 slot when the slot holds it: an MXFP8 CUTLASS
    linear outside the decode-graph FlashInfer path (MXFP8LinearMethod.apply). Its LoRA layer (every attention
    o_proj has one) reads the input only when the call passes lora_params, which _core_then_o_proj never does."""
    from tensorrt_llm._torch.modules import linear

    method = getattr(o_proj, "quant_method", None)
    return (
        isinstance(method, linear.MXFP8LinearMethod)
        and bool(getattr(method, "use_cutlass", False))
        and not linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.get()
    )


@contextlib.contextmanager
def armed(site: str = "sparse", skip_bf16_ok: bool = False):
    """Arm the fold for the attention core run inside the block (its o_proj follows, eagerly)."""
    previous = _armed[0], _skip_ok[0]
    _armed[0] = site if ENABLED else None
    _skip_ok[0] = SKIP_BF16 and skip_bf16_ok
    try:
        yield
    finally:
        _armed[0], _skip_ok[0] = previous


def combine_kwargs(sparse: Any, out: torch.Tensor, metadata: Any, in_place: bool) -> Optional[dict]:
    """The fmha_sm100 combine's MXFP8 outputs for ``out`` [total_q, heads, 128] when the fold engages, else None."""
    site = _armed[0]
    if site is None:
        return None
    if not in_place:
        return _decline(site, "not_in_place")
    if getattr(metadata, "num_generations", 1) != 0:
        # mixed step: the generation rows of the o_proj input come from the decode kernel
        return _decline(site, "mixed_step")
    if (
        out.dim() != 3
        or out.shape[-1] != 128
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        return _decline(site, "layout")
    if not _sparse_takes_o_fp8(sparse):
        return _decline(site, "no_o_fp8")
    rows, cols = out.shape[0], out.shape[1] * out.shape[2]
    padded_rows = (rows + 127) // 128 * 128
    o_fp8 = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=out.device)
    sf = torch.empty((padded_rows * cols // 32,), dtype=torch.uint8, device=out.device)
    if padded_rows != rows:
        # The last 128-row scale tile holds the padding rows; the kernel writes every real row's scales in it.
        tile = 128 * cols // 32
        sf[-tile:].zero_()
    return {"o_fp8": o_fp8, "sf_out": sf, "write_bf16": not _skip_ok[0]}


def _decline(site: str, reason: str) -> None:
    """Count an armed call the fold declined (in the MXFP8 slot audit, as its own tag's set count)."""
    from tensorrt_llm._torch.modules import linear

    linear._prequantized_mxfp8_stats(f"combine_{site}_declined_{reason}")[0] += 1
    return None


def publish(out: torch.Tensor, kwargs: dict) -> None:
    """Hand the combine's MXFP8 pair to the next MXFP8 linear (the o_proj) for the input ``out`` rows."""
    from tensorrt_llm._torch.modules import linear

    source = out.view(out.shape[0], -1)
    linear.set_prequantized_mxfp8(
        source, kwargs["o_fp8"], kwargs["sf_out"], tag=f"combine_{_armed[0]}"
    )
    _pending[0] = (source, linear._PREQUANTIZED_MXFP8_TAKES[0], not kwargs["write_bf16"])


def after_o_proj(o_proj_input: Optional[torch.Tensor] = None) -> None:
    """The o_proj ran: log the engagement only if it took the pair this module published (a take on mismatched
    storage or shape drops the entry and quantizes the bf16 output instead)."""
    pending = _pending[0]
    if pending is None:
        return
    _pending[0] = None
    source, takes_before, bf16_skipped = pending
    from tensorrt_llm._torch.modules import linear

    entry = linear._PREQUANTIZED_MXFP8[0]
    if entry is not None and entry[0] is source:
        linear._PREQUANTIZED_MXFP8[0] = None
        linear._prequantized_mxfp8_stats(entry[3])[3] += 1
    if linear._PREQUANTIZED_MXFP8_TAKES[0] != takes_before:
        logger.info_once(f"[M3 CTX] {_ENGAGEMENT}", key="minimax_m3_ctx_combine_mxfp8_taken")
        if bf16_skipped:
            logger.info_once(
                f"[M3 CTX] {_ENGAGEMENT_SKIP}", key="minimax_m3_ctx_combine_mxfp8_skip_taken"
            )
        return
    if bf16_skipped:
        raise RuntimeError(
            "MiniMax-M3 CTX: the o_proj did not take the combine's MXFP8 pair and its bf16 input was not written "
            f"(published {tuple(source.shape)}; TRTLLM_CTX_COMBINE_MXFP8_SKIP_BF16=0 keeps the bf16 store)"
        )
    seen = (
        "none"
        if o_proj_input is None
        else f"{tuple(o_proj_input.shape)} @{o_proj_input.data_ptr():#x}"
    )
    logger.warning_once(
        "[M3 CTX] combine MXFP8 o_proj input not consumed by the o_proj (it quantized the bf16 output): "
        f"published {tuple(source.shape)} @{source.data_ptr():#x}, o_proj input {seen}",
        key="minimax_m3_ctx_combine_mxfp8_missed",
    )
