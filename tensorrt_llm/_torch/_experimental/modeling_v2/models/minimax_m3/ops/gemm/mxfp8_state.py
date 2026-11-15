# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP8 backend dispatch definitions; each model constructs its own mutable state."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from tensorrt_llm._torch.modules.linear import (
    MXFP8LinearMethod,
    flashinfer_mxfp8_autotune_active,
    flashinfer_mxfp8_decode_graph_capture_active,
)


class _Mxfp8GemmState(nn.Module):
    """The MXFP8 GEMM backend state the engine's warmup drives.

    The engine finds MXFP8 projections by a module's ``quant_method`` and, for every one, opts it into native GEMM
    tuning and (on decode CUDA graphs) the auto-dispatched FlashInfer GEMM, then marks each tuned. Every projection of
    this target dispatches identically, so one method object carries that state for all of them; the forward reads it
    at each call (``_mx_mode``) and passes the resolved backend to the GEMM entries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.quant_method = MXFP8LinearMethod()


@dataclass(frozen=True)
class MxMode:
    """The MXFP8 GEMM dispatch of one call, resolved from ``_Mxfp8GemmState`` and the engine's contexts."""

    decode_graph: bool  # inside a decode CUDA-graph warmup or capture
    autotune: bool  # inside the engine's FlashInfer MXFP8 tuning pass
    tuned_quantize_gemm: (
        bool  # per-bucket tuned quantize + GEMM (decode graphs, tune_decode_graph_backends)
    )
    flashinfer: bool  # FlashInfer's MXFP8 GEMM on the quantized activation
    native_tuning: (
        bool  # the native GEMM through the autotuner (warmup, before its tactics are registered)
    )


def _mx_mode(state: _Mxfp8GemmState) -> MxMode:
    method = state.quant_method
    decode_graph = flashinfer_mxfp8_decode_graph_capture_active()
    autotune = flashinfer_mxfp8_autotune_active()
    return MxMode(
        decode_graph=decode_graph,
        autotune=autotune,
        tuned_quantize_gemm=method.tune_decode_graph_backends and decode_graph,
        flashinfer=method.backend == "flashinfer"
        or (
            method.backend == "auto"
            and (autotune or (method._flashinfer_autotuned and decode_graph))
        ),
        native_tuning=method.needs_native_autotune,
    )
