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
from build_and_run_flux import compute_clip_similarity, main

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

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

    @pytest.mark.parametrize(
        "precision,checkpoint_path",
        [
            ("bf16", None),
            pytest.param(
                "fp8",
                FluxTestConfig.FP8_CHECKPOINT,
                marks=pytest.mark.skipif(
                    FluxTestConfig.FP8_CHECKPOINT is None
                    or not (
                        FluxTestConfig.FP8_CHECKPOINT
                        and os.path.exists(FluxTestConfig.FP8_CHECKPOINT)
                    ),
                    reason="FP8 checkpoint not available",
                ),
            ),
            pytest.param(
                "fp4",
                FluxTestConfig.FP4_CHECKPOINT,
                marks=pytest.mark.skipif(
                    FluxTestConfig.FP4_CHECKPOINT is None
                    or not (
                        FluxTestConfig.FP4_CHECKPOINT
                        and os.path.exists(FluxTestConfig.FP4_CHECKPOINT)
                    ),
                    reason="FP4 checkpoint not available",
                ),
            ),
        ],
    )
    def test_flux_e2e_with_clip_validation(self, precision, checkpoint_path, clip_model, tmp_path):
        """End-to-end test for Flux model with CLIP similarity validation.

        Tests:
        1. Call build_and_run_flux.py main function
        2. Validate generated image quality using CLIP similarity
        """
        output_image = tmp_path / f"flux_{precision}_output.png"

        # Build arguments for main function
        args = [
            "--model",
            FluxTestConfig.MODEL_ID,
            "--prompt",
            FluxTestConfig.PROMPT,
            "--image_path",
            str(output_image),
            "--max_batch_size",
            str(FluxTestConfig.MAX_BATCH_SIZE),
        ]

        # Add checkpoint if provided
        if checkpoint_path:
            args.extend(["--restore_from", checkpoint_path])

        ad_logger.info(f"Running main with args for {precision}: {' '.join(args)}")

        # Call main function directly with args
        main(args)

        # Verify image was generated
        assert output_image.exists(), f"Output image not found at {output_image}"

        # Compute CLIP similarity
        similarity = compute_clip_similarity(str(output_image), FluxTestConfig.PROMPT, clip_model)
        ad_logger.info(f"CLIP similarity score for {precision}: {similarity:.4f}")

        # Assert similarity is above threshold
        assert similarity >= FluxTestConfig.MIN_CLIP_SIMILARITY, (
            f"CLIP similarity {similarity:.4f} is below threshold {FluxTestConfig.MIN_CLIP_SIMILARITY}. "
            f"Image may not match prompt: '{FluxTestConfig.PROMPT}'"
        )

        ad_logger.info(f"✓ Test passed for {precision}: CLIP similarity = {similarity:.4f}")
