# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for WanTimeTextImageEmbedding against the HuggingFace reference.

Isolates condition_embedder from both implementations to verify each output
element independently (temb, temb_proj, projected text, projected image).

Tested configurations:
  - T2V (1.3B): timestep + text (1, 77, 4096) → (temb, temb_proj, text_out, None)
  - I2V (14B 480P): timestep + text (1, 512, 4096) + image (1, 257, 1280)
                    → (temb, temb_proj, text_out, img_out)

Run all:
    pytest tests/unittest/_torch/visual_gen/test_wan_condition_embedder.py -v -s

Run one:
    pytest tests/unittest/_torch/visual_gen/test_wan_condition_embedder.py -v -s -k t2v
    pytest tests/unittest/_torch/visual_gen/test_wan_condition_embedder.py -v -s -k i2v

Override checkpoint paths:
    DIFFUSION_MODEL_PATH_WAN21_1_3B=/path/to/Wan2.1-T2V-1.3B-Diffusers \\
    DIFFUSION_MODEL_PATH_WAN21_I2V_480P=/path/to/Wan2.1-I2V-14B-480P-Diffusers \\
        pytest tests/unittest/_torch/visual_gen/test_wan_condition_embedder.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionArgs, DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.wan.transformer_wan import WanTransformer3DModel


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Path helpers
# ============================================================================


def _llm_models_root() -> Path:
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    else:
        root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return root


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or str(_llm_models_root() / default_name)


WAN21_1_3B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_1_3B", "Wan2.1-T2V-1.3B-Diffusers")
WAN21_I2V_480P_PATH = os.environ.get("DIFFUSION_MODEL_PATH_WAN21_I2V_480P", "")

COS_SIM_THRESHOLD = 0.99
DEVICE = "cuda"
DTYPE = torch.bfloat16


# ============================================================================
# Helpers
# ============================================================================


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def _load_models(checkpoint_dir: str):
    """Load HF and TRTLLM transformers with identical weights.

    Returns (hf_model, our_model).  Caller accesses .condition_embedder on each.
    """
    from diffusers import WanTransformer3DModel as HFWanTransformer3DModel

    hf_model = (
        HFWanTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder="transformer",
            torch_dtype=DTYPE,
        )
        .to(DEVICE)
        .eval()
    )

    args = DiffusionArgs(
        checkpoint_path=checkpoint_dir,
        device=DEVICE,
        dtype="bfloat16",
    )
    model_config = DiffusionModelConfig.from_pretrained(checkpoint_dir, args=args)
    our_model = WanTransformer3DModel(model_config=model_config).to(DEVICE).eval()
    our_model.load_weights({k: v for k, v in hf_model.state_dict().items()})
    our_model.post_load_weights()

    return hf_model, our_model


def _assert_embedder_outputs(
    hf_ce,
    our_ce,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor = None,
) -> None:
    """Run both condition embedders and assert cosine similarity >= 0.99 per output."""
    with torch.no_grad():
        hf_out = hf_ce(timestep, encoder_hidden_states, encoder_hidden_states_image)
        our_out = our_ce(timestep, encoder_hidden_states, encoder_hidden_states_image)

    names = ["temb", "temb_proj", "text_embeds", "image_embeds"]
    for name, hf_tensor, our_tensor in zip(names, hf_out, our_out):
        if hf_tensor is None and our_tensor is None:
            continue
        assert hf_tensor is not None and our_tensor is not None, (
            f"{name}: one is None but the other is not (hf={hf_tensor}, ours={our_tensor})"
        )
        cos_sim = _cosine_similarity(our_tensor, hf_tensor)
        max_diff = (our_tensor.float() - hf_tensor.float()).abs().max().item()
        print(f"  {name}: cosine_similarity={cos_sim:.6f}  max_diff={max_diff:.6f}")
        assert cos_sim >= COS_SIM_THRESHOLD, (
            f"{name}: cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. "
            f"max_diff={max_diff:.6f}"
        )


# ============================================================================
# T2V — Wan2.1-T2V-1.3B
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanT2VConditionEmbedder:
    """Condition embedder outputs must match HF for T2V (no image input)."""

    @pytest.fixture(scope="class")
    def t2v_models(self):
        if not os.path.exists(WAN21_1_3B_PATH):
            pytest.skip(f"Checkpoint not found: {WAN21_1_3B_PATH}")
        hf_model, our_model = _load_models(WAN21_1_3B_PATH)
        yield hf_model, our_model
        del hf_model, our_model
        torch.cuda.empty_cache()

    def test_embedder_outputs(self, t2v_models):
        hf_model, our_model = t2v_models

        torch.manual_seed(42)
        timestep = torch.tensor([500.0], device=DEVICE, dtype=torch.float32)
        encoder_hidden_states = torch.randn(1, 77, 4096, device=DEVICE, dtype=DTYPE)

        print("\n  === T2V condition embedder ===")
        _assert_embedder_outputs(
            hf_model.condition_embedder,
            our_model.condition_embedder,
            timestep,
            encoder_hidden_states,
        )


# ============================================================================
# I2V — Wan2.1-I2V-14B-480P
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_i2v
class TestWanI2VConditionEmbedder:
    """Condition embedder outputs must match HF for I2V (with image input)."""

    @pytest.fixture(scope="class")
    def i2v_models(self):
        if not WAN21_I2V_480P_PATH or not os.path.exists(WAN21_I2V_480P_PATH):
            pytest.skip(
                "Checkpoint not found. "
                "Set DIFFUSION_MODEL_PATH_WAN21_I2V_480P=/path/to/Wan2.1-I2V-14B-480P-Diffusers"
            )
        hf_model, our_model = _load_models(WAN21_I2V_480P_PATH)
        yield hf_model, our_model
        del hf_model, our_model
        torch.cuda.empty_cache()

    def test_embedder_outputs(self, i2v_models):
        hf_model, our_model = i2v_models

        torch.manual_seed(42)
        timestep = torch.tensor([500.0], device=DEVICE, dtype=torch.float32)
        # text_seq_len=512: required by the I2V cross-attention hardcoded split
        encoder_hidden_states = torch.randn(1, 512, 4096, device=DEVICE, dtype=DTYPE)
        # CLIP ViT-H/14: 256 patches + 1 CLS token
        encoder_hidden_states_image = torch.randn(1, 257, 1280, device=DEVICE, dtype=DTYPE)

        print("\n  === I2V condition embedder ===")
        _assert_embedder_outputs(
            hf_model.condition_embedder,
            our_model.condition_embedder,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
        )
