# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from defs.visual_gen import minimax_h3_quality_cases as release


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
