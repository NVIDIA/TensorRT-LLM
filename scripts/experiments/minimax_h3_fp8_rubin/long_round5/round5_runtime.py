# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compose the published round-four runtime with exact conversion kernels."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from round4_runtime import Runtime as Round4Runtime
from round5_cast_runtime import Runtime as CastRuntime
from round5_kernels import prepare
from round5_quant_runtime import Runtime as QuantRuntime

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


class Runtime:
    """Opt-in single frozen pipeline experiment; graphs disabled, U1/TP1 only."""

    MODES = ("native", "g2", "long4", "quant", "candidate")

    def __init__(
        self, pipeline: BasePipeline, *, timer, cache: Path, base: Round4Runtime | None = None
    ) -> None:
        assert not pipeline.pipeline_config.cuda_graph.enable
        assert int(pipeline.transformer.sharder.size) == 1
        self.base = base if base is not None else Round4Runtime(pipeline)
        self.base.configure("native")
        scales = torch.cat(
            [
                block.attn.to_out[0].input_scale.detach().reshape(1)
                for block in pipeline.transformer.transformer_blocks
            ]
        )
        quant, cast, factors, self.proof = prepare(scales, cache)
        self.quant = QuantRuntime(pipeline, quant, factors, self.proof)
        self.cast = CastRuntime(pipeline, timer=timer, module=cast, proof=self.proof)
        self.mode = "native"

    def configure(self, mode: str, *, validate: bool = False) -> None:
        """Restore helper/timer owners before changing the underlying round-four mode."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown round-five mode: {mode}")
        self.cast.configure(False)
        if self.quant.enabled:
            self.quant.configure(False)
        base_mode = mode if mode in ("native", "g2") else "kf_cache_recip_attention"
        self.base.configure(base_mode)
        self.base.cache.validate = False
        if mode in ("quant", "candidate"):
            self.quant.configure(True, validate=validate)
        if mode == "candidate":
            self.cast.configure(True, validate=validate)
        self.mode = mode

    def report(self) -> dict:
        return dict(
            mode=self.mode,
            kernel_qualification=self.proof,
            quant=self.quant.report(),
            cast=self.cast.report(),
        )
