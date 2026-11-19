# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guarded static attention-output quantization for one frozen U1 experiment."""

from __future__ import annotations

import os
from collections import Counter
from types import ModuleType
from typing import TYPE_CHECKING

import round4_attention_runtime as attention
import torch
from round5_kernels import BLOCKS, THREADS, same

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


def version(value: torch.Tensor) -> int | None:
    try:
        return int(value._version)
    except RuntimeError:
        return None


def stamp(value: torch.Tensor) -> tuple:
    return (
        value.device.index,
        value.data_ptr(),
        str(value.dtype),
        tuple(value.shape),
        tuple(value.stride()),
        value.storage_offset(),
        version(value),
    )


class Runtime:
    """Keep original scales and GEMM; reject changed owners before enabling.

    Fast inputs are contiguous CUDA BF16 [M, 7168], M in {37776, 37820, 38169}; waiter-only HOST shape port.
    Scales are the original contiguous CUDA FP32 [1] attention-output scales.
    """

    def __init__(
        self, pipeline: BasePipeline, module: ModuleType, factors: torch.Tensor, proof: dict
    ) -> None:
        self.pipeline = pipeline
        self.blocks = pipeline.transformer.transformer_blocks
        assert len(self.blocks) == 50 and not pipeline.pipeline_config.cuda_graph.enable
        self.module, self.factors, self.proof = module, factors, proof
        assert proof["unary_qualified"] and factors.shape == (50,)
        self.original = attention.quantize
        assert (
            self.original.__module__ == attention.__name__ and self.original.__name__ == "quantize"
        )
        self.bound = self.quantize
        self.enabled = self.validate = False
        self.counts, self.reasons = Counter(), Counter()
        self.checks: list[dict] = []
        self.entries: dict[tuple, tuple] = {}
        self.pdl = os.environ.get("TRTLLM_ENABLE_PDL") != "0"
        assert int(self.pdl) == proof["pdl"]
        self.scales = [block.attn.to_out[0].input_scale for block in self.blocks]
        for index, scale in enumerate(self.scales):
            assert scale.is_cuda and scale.dtype == torch.float32 and scale.numel() == 1
            assert scale.is_contiguous() and version(scale) is not None
            key = (scale.device.index, scale.data_ptr())
            if key in self.entries:
                assert self.entries[key][1] == stamp(scale)
            else:
                self.entries[key] = (scale, stamp(scale), factors[index : index + 1])
        bits = torch.cat([s.detach().reshape(1) for s in self.scales]).cpu().view(torch.int32)
        assert [int(x) & 0xFFFFFFFF for x in bits.tolist()] == proof["scale_bits_u32"]
        self.owner = self.model_stamp()

    def model_stamp(self) -> tuple:
        return tuple(
            (
                id(block),
                id(block.attn.to_out[0]),
                id(block.attn.to_out[0].input_scale),
                stamp(block.attn.to_out[0].input_scale),
            )
            for block in self.blocks
        )

    def quantize(self, raw: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        self.counts["calls"] += 1
        entry = self.entries.get((scale.device.index, scale.data_ptr()))
        reason = None
        if not self.enabled:
            reason = "disabled"
        elif entry is None:
            reason = "unknown_scale"
        elif stamp(scale) != entry[1]:
            reason = "scale_changed"
        elif (
            not raw.is_cuda
            or raw.dtype != torch.bfloat16
            or raw.ndim < 2
            or raw.shape[-1] != 7168
            or raw.numel() // 7168 not in (37776, 37820, 38169)
            or not raw.is_contiguous()
            or raw.numel() == 0
            or raw.device != scale.device
            or raw.data_ptr() % 16
        ):
            reason = "unsupported_raw"
        if reason:
            self.counts["fallback_calls"] += 1
            self.reasons[reason] += 1
            return self.original(raw, scale)
        value = self.module.quantize_config(raw, scale, entry[2], 0, self.pdl, BLOCKS, THREADS)
        value = value.reshape(-1, raw.shape[-1])
        self.counts["fast_calls"] += 1
        if self.validate:
            equal = same(value, self.original(raw, scale))
            self.checks.append(
                dict(
                    index=len(self.checks),
                    byte_equal=equal,
                    raw_shape=list(raw.shape),
                    raw_stride=list(raw.stride()),
                    scale_pointer=scale.data_ptr(),
                )
            )
            self.counts["original_quant_checks"] += 1
            assert equal, self.checks[-1]
        return value

    def configure(self, enabled: bool, *, validate: bool = False) -> None:
        assert self.model_stamp() == self.owner, "Frozen scale/module owner changed"
        assert not self.pipeline.pipeline_config.cuda_graph.enable
        assert attention.quantize in (self.original, self.bound)
        self.validate = bool(validate)
        attention.quantize = self.bound if enabled else self.original
        self.enabled = bool(enabled)

    def report(self) -> dict:
        return dict(
            counts=dict(self.counts),
            reasons=dict(self.reasons),
            checks=list(self.checks),
            pdl=int(self.pdl),
            launch=dict(mode=0, threads=THREADS, blocks=BLOCKS),
            scope=(
                "One frozen U1 pipeline, graphs OFF, original static scales/GEMM; "
                "no concurrent model mutation or raw scale alias writes."
            ),
        )
