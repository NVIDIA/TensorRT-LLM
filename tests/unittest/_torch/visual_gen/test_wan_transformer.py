# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for WanTransformer3DModel against the HuggingFace reference.

Compares our implementation against diffusers WanTransformer3DModel using
identical weights and inputs, asserting cosine similarity >= 0.99.

Two modes are tested:
  - T2V (1.3B): 480x832, no image conditioning
  - I2V (14B 480P): 480x832, CLIP image conditioning

Run all:
    pytest tests/unittest/_torch/visual_gen/test_wan_transformer.py -v -s

Run one:
    pytest tests/unittest/_torch/visual_gen/test_wan_transformer.py -v -s -k t2v
    pytest tests/unittest/_torch/visual_gen/test_wan_transformer.py -v -s -k i2v

Override checkpoint paths:
    DIFFUSION_MODEL_PATH_WAN21_1_3B=/path/to/Wan2.1-T2V-1.3B-Diffusers \\
    DIFFUSION_MODEL_PATH_WAN21_I2V_480P=/path/to/Wan2.1-I2V-14B-480P-Diffusers \\
        pytest tests/unittest/_torch/visual_gen/test_wan_transformer.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig, VisualGenArgs
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


def _llm_models_root() -> str:
    """Return LLM_MODELS_ROOT path if set in env, assert when it's set but not a valid path."""
    root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return str(root)


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or os.path.join(_llm_models_root(), default_name)


WAN21_1_3B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN21_1_3B", "Wan2.1-T2V-1.3B-Diffusers")
WAN21_I2V_480P_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_WAN21_I2V_480P", "Wan2.1-I2V-14B-480P-Diffusers"
)


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
    """Load both the HF reference transformer and our transformer from the same checkpoint.

    Both models receive identical weights (loaded from the HF state_dict).
    Returns (hf_model, our_model) in eval mode on DEVICE with DTYPE.
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

    args = VisualGenArgs(
        checkpoint_path=checkpoint_dir,
        device=DEVICE,
        dtype="bfloat16",
    )
    model_config = DiffusionModelConfig.from_pretrained(checkpoint_dir, args=args)
    our_model = WanTransformer3DModel(model_config=model_config).to(DEVICE).eval()

    # Initialize our model with the exact same weights as the HF model.
    our_model.load_weights({k: v for k, v in hf_model.state_dict().items()})
    # Cast non-Linear embedder submodules (time_embedder, text_embedder) to target dtype.
    our_model.post_load_weights()

    return hf_model, our_model


# ============================================================================
# T2V correctness test — Wan2.1-T2V-1.3B
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
class TestWanT2VTransformerCorrectness:
    """Output of our T2V transformer must match HF WanTransformer3DModel.

    Model: Wan2.1-T2V-1.3B-Diffusers
      hidden_size=1536, 30 layers, 12 heads, patch_size=(1,2,2)

    Input shape:
      latent  (1, 16, 1, 60, 104)  — 480/8=60, 832/8=104, 1 latent frame
      text    (1, 77, 4096)
      timestep (1,)

    Post-patch sequence length: 1 * (60/2) * (104/2) = 1560 tokens.
    """

    @pytest.fixture(scope="class")
    def t2v_models(self):
        if not os.path.exists(WAN21_1_3B_PATH):
            pytest.skip(f"Checkpoint not found: {WAN21_1_3B_PATH}")
        hf_model, our_model = _load_models(WAN21_1_3B_PATH)
        yield hf_model, our_model
        del hf_model, our_model
        torch.cuda.empty_cache()

    def test_cosine_similarity(self, t2v_models):
        hf_model, our_model = t2v_models

        torch.manual_seed(42)
        B, C, T, H, W = 1, 16, 1, 60, 104
        text_seq_len = 77

        hidden_states = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE)
        timestep = torch.tensor([500.0], device=DEVICE, dtype=torch.float32)
        encoder_hidden_states = torch.randn(B, text_seq_len, 4096, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            hf_out = hf_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )[0]

            our_out = our_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        cos_sim = _cosine_similarity(our_out, hf_out)
        max_diff = (our_out.float() - hf_out.float()).abs().max().item()
        print(f"\n  T2V 480x832 cosine_similarity={cos_sim:.6f}  max_diff={max_diff:.6f}")
        print(f"  our_out.shape={our_out.shape}  hf_out.shape={hf_out.shape}")

        assert cos_sim >= COS_SIM_THRESHOLD, (
            f"T2V cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. max_diff={max_diff:.6f}"
        )


# ============================================================================
# I2V correctness test — Wan2.1-I2V-14B-480P
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_i2v
class TestWanI2VTransformerCorrectness:
    """Output of our I2V transformer must match HF WanTransformer3DModel.

    Model: Wan2.1-I2V-14B-480P-Diffusers
      hidden_size=5120, 40 layers, 40 heads, patch_size=(1,2,2)
      in_channels=36 (16 video + 4 mask + 16 condition), add_k_proj present

    Input shape:
      latent        (1, 36, 1, 60, 104)  — 36ch I2V, 480/8=60, 832/8=104
      text          (1, 512, 4096)        — 512 required by hardcoded I2V split
      image_embeds  (1, 257, 1280)        — CLIP ViT-H/14 (256 patches + CLS)
      timestep      (1,)

    Post-patch sequence length: 1 * 30 * 52 = 1560 tokens.
    Cross-attention: image context = total_len - 512 = 257 tokens,
                     text context  = 512 tokens.
    """

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

    def test_cosine_similarity(self, i2v_models):
        hf_model, our_model = i2v_models

        torch.manual_seed(42)
        B, C, T, H, W = 1, 36, 1, 60, 104
        text_seq_len = 512  # hardcoded split in I2V cross-attention
        img_seq_len = 257  # CLIP ViT-H/14 tokens
        img_embed_dim = 1280  # CLIP ViT-H/14 embed dim

        hidden_states = torch.randn(B, C, T, H, W, device=DEVICE, dtype=DTYPE)
        timestep = torch.tensor([500.0], device=DEVICE, dtype=torch.float32)
        encoder_hidden_states = torch.randn(B, text_seq_len, 4096, device=DEVICE, dtype=DTYPE)
        image_embeds = torch.randn(B, img_seq_len, img_embed_dim, device=DEVICE, dtype=DTYPE)

        with torch.no_grad():
            hf_out = hf_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=image_embeds,
                return_dict=False,
            )[0]

            our_out = our_model(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=image_embeds,
            )

        cos_sim = _cosine_similarity(our_out, hf_out)
        max_diff = (our_out.float() - hf_out.float()).abs().max().item()
        print(f"\n  I2V 480x832 cosine_similarity={cos_sim:.6f}  max_diff={max_diff:.6f}")
        print(f"  our_out.shape={our_out.shape}  hf_out.shape={hf_out.shape}")

        assert cos_sim >= COS_SIM_THRESHOLD, (
            f"I2V cosine similarity {cos_sim:.6f} < {COS_SIM_THRESHOLD}. max_diff={max_diff:.6f}"
        )
