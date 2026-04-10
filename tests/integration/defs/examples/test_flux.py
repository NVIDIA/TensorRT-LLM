# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration test for build_and_run_flux.py with multiple quantization formats."""

import importlib.util
import os

import pytest
import torch
from build_and_run_flux import clip_model as load_clip_model

# Check if CLIP is available
CLIP_AVAILABLE = importlib.util.find_spec("transformers") is not None


class FluxTestConfig:
    """Configuration for Flux integration test."""

    MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
    PROMPT = "a photo of an astronaut riding a horse on mars"
    MIN_CLIP_SIMILARITY = 0.25
    NUM_INFERENCE_STEPS = 20
    MAX_BATCH_SIZE = 1
    BACKEND = "torch-opt"

    # Checkpoint paths for different quantization formats
    # These can be set via environment variables or test parameters
    FP8_CHECKPOINT = os.environ.get("FLUX_FP8_CHECKPOINT")
    FP4_CHECKPOINT = os.environ.get("FLUX_FP4_CHECKPOINT")


@pytest.fixture(scope="module")
def clip_model():
    """Pytest fixture for loading CLIP model once per test module."""
    if not CLIP_AVAILABLE:
        pytest.skip("CLIP not available")
    return load_clip_model()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Flux model")
@pytest.mark.slow  # Mark as slow test
class TestFluxIntegration:
    """Integration tests for Flux model with different quantization formats."""
