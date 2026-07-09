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
"""Pipeline-level tests for FastWanPipeline requiring a real checkpoint.

TestFastWanComponentAccuracy
    Verifies that the text encoder and VAE decoder load correctly and produce
    numerically stable output. Reference tensors were generated once from this
    pipeline and committed to golden/component_accuracy/. Future runs compare
    against them to catch weight-loading bugs, quantization changes, or
    refactors that silently shift component outputs.

Run:
    DIFFUSION_MODEL_PATH_FASTWAN=/path/to/checkpoint \\
        pytest tests/unittest/_torch/visual_gen/test_fastwan_pipeline.py -v -s

Override checkpoint path via DIFFUSION_MODEL_PATH_FASTWAN env var or place
the checkpoint at $LLM_MODELS_ROOT/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
"""

import gc
import os
from pathlib import Path

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def _checkpoint_path() -> str:
    if "DIFFUSION_MODEL_PATH_FASTWAN" in os.environ:
        return os.environ["DIFFUSION_MODEL_PATH_FASTWAN"]
    root = os.environ.get(
        "LLM_MODELS_ROOT",
        "/home/scratch.trt_llm_data_ci/llm-models",
    )
    return os.path.join(root, "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers")


FASTWAN_PATH = _checkpoint_path()

GOLDEN_DIR = Path(__file__).parent / "golden" / "component_accuracy"

# Prompt used when generating the text encoder reference tensor.
_COMPONENT_PROMPT = "A cat sitting on a sunny windowsill watching birds outside."


@pytest.fixture(scope="module")
def fastwan_pipeline():
    if not os.path.exists(FASTWAN_PATH):
        pytest.skip(f"Checkpoint not found: {FASTWAN_PATH}")
    args = VisualGenArgs(
        model=FASTWAN_PATH,
        torch_compile_config=TorchCompileConfig(enable=False),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    yield pipeline
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestFastWanComponentAccuracy:
    """Verify text encoder and VAE decoder produce numerically stable output.

    Each test loads a reference tensor from golden/component_accuracy/, runs the
    component on a fixed input, and asserts torch.allclose against the reference.
    This catches weight-loading bugs and silent numerical regressions without
    running the full pipeline.
    """

    def test_text_encoder_reference(self, fastwan_pipeline):
        ref = torch.load(GOLDEN_DIR / "text_encoder_reference.pt", weights_only=True)
        with torch.no_grad():
            embeds, _ = fastwan_pipeline._encode_prompt([_COMPONENT_PROMPT], "", 512)
        embeds_cpu = embeds.cpu().float()
        ref_f = ref.float()
        assert torch.allclose(embeds_cpu, ref_f, atol=1e-3), (
            f"Text encoder output diverged from reference. "
            f"Max diff: {(embeds_cpu - ref_f).abs().max():.6f} "
            f"(shape {embeds_cpu.shape}, dtype {embeds.dtype})"
        )

    def test_vae_decoder_reference(self, fastwan_pipeline):
        latent = torch.load(GOLDEN_DIR / "vae_input_latent.pt", weights_only=True)
        ref = torch.load(GOLDEN_DIR / "vae_reference.pt", weights_only=True)
        with torch.no_grad():
            output = fastwan_pipeline._decode_latents(latent.to(fastwan_pipeline.device))
        output_cpu = output.cpu().float()
        ref_f = ref.float()
        assert torch.allclose(output_cpu, ref_f, atol=1.0), (
            f"VAE decoder output diverged from reference. "
            f"Max diff: {(output_cpu - ref_f).abs().max():.1f} pixels "
            f"(shape {output_cpu.shape})"
        )

