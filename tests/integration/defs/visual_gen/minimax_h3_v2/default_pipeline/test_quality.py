# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

from defs.visual_gen.minimax_h3_v2 import minimax_h3_quality_cases as release

_full_cuda_memory_budget = release._full_cuda_memory_budget


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
