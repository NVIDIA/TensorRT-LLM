# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Immutable specialization keys for the PrimsTS dense GEMM family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

OperandFormat = Literal["fp8_e4m3", "nvfp4_e2m1"]
OutputFormat = Literal["bf16", "fp8_e4m3", "nvfp4_e2m1"]
Epilogue = Literal["linear", "swiglu", "qkv_qknorm_rope"]


@dataclass(frozen=True)
class PrimsTsGemmConfig:
    """Every compile-time decision captured by a CuTe program."""

    arch: int
    operand_format: OperandFormat
    output_format: OutputFormat
    epilogue: Epilogue
    has_bias: bool
    head_dim: Optional[int]
    is_neox: Optional[bool]
    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 128
    cluster_shape: tuple[int, int, int] = (2, 2, 1)
    scheduler: str = "clc_dynamic"
    shape_bucket: Optional[tuple[int, int, int]] = None
    has_qkv_scale: bool = False
    # 96 selects the SM103 NVFP4 instruction; 64 retains the reference path.
    nvfp4_mma_k: int = 64
    # None retains the epilogue default (8 for overlap/fused, otherwise 4).
    epilogue_warps: Optional[int] = None
    tmem_overlap: bool = False
    # None retains the derived pipeline depth (6 FP8, 5 NVFP4, 4 QKV).
    ab_stages: Optional[int] = None
    # False keeps the direct epilogue store. FP4 output has no TMA store path.
    use_tma_store: bool = False
    # Store Q, K, V as three contiguous [M, N/3] sections.
    separate_qkv_output: bool = False
