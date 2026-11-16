# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Pipeline-level tests for WanDMDPipeline.

Run:
    pytest tests/unittest/_torch/visual_gen/test_fastwan_pipeline.py -v -s
"""

import gc
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import AttentionConfig, TorchCompileConfig, VisualGenArgs


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


FASTWAN_SUBDIR = "FastWan2.2-TI2V-5B-FullAttn-Diffusers"


@pytest.fixture(scope="module")
def fastwan_pipeline():
    args = VisualGenArgs(
        model=get_checkpoint(FASTWAN_SUBDIR),
        torch_compile_config=TorchCompileConfig(enable=False),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFastWanForward:
    """Run the full forward() pass and validate the output video shape."""

    def test_single_prompt_shape(self, fastwan_pipeline):
        """Single prompt returns (B, T, H, W, C) with B=1."""
        result = fastwan_pipeline.forward(
            prompt="A red fox walking through snow",
            height=704,
            width=1280,
            num_frames=9,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 1 and H == 704 and W == 1280 and C == 3

    def test_batch_prompt_shape(self, fastwan_pipeline):
        """List of prompts returns (B, T, H, W, C) with B=2."""
        result = fastwan_pipeline.forward(
            prompt=["A red fox walking through snow", "A cat on a roof"],
            height=704,
            width=1280,
            num_frames=9,
            seed=42,
        )
        assert result.video.dim() == 5, f"Expected 5D (B,T,H,W,C), got {result.video.dim()}D"
        B, _T, H, W, C = result.video.shape
        assert B == 2 and H == 704 and W == 1280 and C == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFastWanFP8:
    """Confirm FP8 + TRTLLM attention runs end-to-end on the FastWan checkpoint.

    FastWan inherits FP8 support from WanPipeline.post_load_weights(), but that
    method is only tested against the TI2V-5B checkpoint in CI. This test confirms
    it also works with the FastWan distilled weights and the DMD denoising loop.
    """

    def test_fp8_trtllm(self):
        args = VisualGenArgs(
            model=get_checkpoint(FASTWAN_SUBDIR),
            torch_compile_config=TorchCompileConfig(enable=False),
            quant_config={"quant_algo": "FP8", "dynamic": True},
            attention_config=AttentionConfig(backend="TRTLLM"),
        )
        pipeline = PipelineLoader(args).load(skip_warmup=True)
        try:
            with torch.no_grad():
                result = pipeline.forward(
                    prompt="A red fox walking through snow",
                    height=704,
                    width=1280,
                    num_frames=9,
                    seed=42,
                )
            assert result.video.dim() == 5
            B, _T, H, W, C = result.video.shape
            assert B == 1 and H == 704 and W == 1280 and C == 3
        finally:
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
