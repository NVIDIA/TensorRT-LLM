# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from defs.visual_gen import minimax_h3_quality_cases as release
from test_common.visual_gen.h3_rollout import h3_modeling_v1 as h3_modeling_v1

pytestmark = pytest.mark.usefixtures("h3_modeling_v1")


_full_cuda_memory_budget = release._full_cuda_memory_budget


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minimax_h3_released_transformer_loads_and_runs() -> None:
    release.check_released_transformer_loads_and_runs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES", "NVFP4"])
def test_minimax_h3_released_transformer_dynamic_quant_compile_smoke(quant_algo: str) -> None:
    release.check_released_transformer_dynamic_quant_compile_smoke(quant_algo)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minimax_h3_released_pipeline_smoke() -> None:
    release.check_released_pipeline_smoke()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minimax_h3_public_visual_gen_api_smoke() -> None:
    release.check_public_visual_gen_api_smoke()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("task", "quant_algo"),
    [("t2va", None), ("fl2va", None), ("t2va", "FP8_BLOCK_SCALES"), ("ref2va", None)],
    ids=["t2va", "fl2va", "t2va-fp8-blockwise", "ref2va"],
)
def test_minimax_h3_diffusers_lpips_and_audio_reference(
    task: str, quant_algo: str | None, tmp_path: Path, _full_cuda_memory_budget: None
) -> None:
    release.check_diffusers_lpips_and_audio_reference(
        task, quant_algo, tmp_path, _full_cuda_memory_budget
    )
