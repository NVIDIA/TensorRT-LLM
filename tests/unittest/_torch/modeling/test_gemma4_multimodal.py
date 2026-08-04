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
"""Unit + E2E tests for the Gemma4 multimodal model components.

Tier 0/1 (no LLM_MODELS_ROOT needed, GPU only):
  - ``TestGemma4VisionTower`` — refactored TRT-LLM ``Gemma4VisionModel``
    (Option B: ``Gemma4VisionAttention`` inherits ``Attention`` base, vision
    runs through the TRTLLM attention backend, weights load via fused
    ``qkv_proj`` + clamp-buffer remap).
  - ``TestGemma4MultimodalEmbedder`` — RMSNorm + projection parity vs HF.
  - ``TestGemma4ForConditionalGeneration`` — wrapper instantiation; verifies
    that ``vision_tower`` is the refactored class (not HF ``AutoModel``).

Tier 2 (LLM_MODELS_ROOT gated, real tokenizer/processor):
  - ``TestGemma4InputProcessor`` — image/audio/video pre-processing.
  - ``TestGemma4MultimodalE2E`` — full LLM+vision generate parity via the
    shared ``TestModelingMultimodal`` skeleton (image / multi-image modalities).

Audio-side tests (``TestGemma4InputProcessor.test_audio_*``) are intentionally
left unchanged — audio tower refactor (Conformer migration) is a follow-up.
"""

from __future__ import annotations

import os
import unittest
from copy import deepcopy
from typing import Dict, List, Type

import pytest
import torch

# Gemma4 requires transformers>=5.5.0 (native Gemma4 config/model classes).
pytest.importorskip(
    "transformers", minversion="5.5.0", reason="Gemma4 requires transformers>=5.5.0"
)

from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig  # noqa: E402
from transformers import Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4MultimodalEmbedder as HFGemma4MultimodalEmbedder,
)
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4VisionModel as HFGemma4VisionModel,
)

from tensorrt_llm._torch.model_config import ModelConfig  # noqa: E402
from tensorrt_llm._torch.models.modeling_gemma4_vision import Gemma4VisionModel  # noqa: E402
from tensorrt_llm._torch.models.modeling_gemma4mm import (  # noqa: E402
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from tensorrt_llm.mapping import Mapping  # noqa: E402

# ---------------------------------------------------------------------------
# Small configs for unit-level tests
# ---------------------------------------------------------------------------

# Real Gemma4 vision dims scaled down for speed. ``head_dim`` is the floor
# imposed by MMHA (kernel rejects head_dim < 32). Keep hidden_size = heads *
# head_dim so the fused ``qkv_proj`` shape stays consistent.
SMALL_VISION_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 1024,
    "use_clipped_linears": False,
    "standardize": True,
    "model_type": "gemma4_vision",
}

SMALL_TEXT_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "global_head_dim": 32,
    "num_global_key_value_heads": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": None,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
}


def _make_dummy_pixel_input(vision_cfg, B=1, side=6, dtype=torch.float32, device="cuda"):
    """Match the smoke ``gemma4_vision_smoke.py:make_dummy_input`` contract.

    Returns ``(pixel_values, pixel_position_ids, output_length)`` where
    ``output_length = side**2 // pooling_kernel_size**2``.
    """
    N = side * side
    C = vision_cfg.patch_size**2 * 3
    pixel_values = torch.randn(B, N, C, device=device, dtype=dtype)
    pos = (
        torch.stack(
            torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        .reshape(1, -1, 2)
        .expand(B, -1, -1)
        .contiguous()
    )
    output_length = N // (vision_cfg.pooling_kernel_size**2)
    return pixel_values, pos, output_length


def _build_trt_vision_tower(vision_cfg, dtype=torch.float32, device="cuda"):
    """Build the refactored TRT-LLM ``Gemma4VisionModel`` with the TRTLLM attn backend.

    ``get_sub_model_config`` in ``modeling_gemma4mm.py`` always picks
    ``attn_backend="TRTLLM"`` for the vision tower, so unit tests mirror that.
    """
    mc = ModelConfig(
        pretrained_config=vision_cfg,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        attn_backend="TRTLLM",
    )
    return Gemma4VisionModel(mc).to(device).to(dtype).eval()


# ---------------------------------------------------------------------------
# Tier 0/1 — refactored vision tower
# ---------------------------------------------------------------------------


class TestGemma4VisionTower(unittest.TestCase):
    """Refactored ``Gemma4VisionModel`` (Option B) — Attention base class path.

    Covers:
      - build via ``ModelConfig`` with TRTLLM backend
      - random-weight forward (shape + NaN check)
      - HF state_dict → ``load_weights`` weight fusion (q/k/v → qkv_proj)
      - HF ↔ TRT forward parity (no clamp)
    """

    def test_build_from_model_config(self):
        """``Gemma4VisionModel(ModelConfig)`` exposes the expected submodules."""
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        tower = _build_trt_vision_tower(vision_cfg, dtype=torch.bfloat16)

        self.assertIsNotNone(tower.patch_embedder)
        self.assertIsNotNone(tower.encoder)
        self.assertIsNotNone(tower.pooler)
        self.assertEqual(len(tower.encoder.layers), vision_cfg.num_hidden_layers)
        # Standardize buffers must register when standardize=True (HF parity).
        self.assertTrue(hasattr(tower, "std_bias"))
        self.assertTrue(hasattr(tower, "std_scale"))

    @torch.no_grad()
    def test_forward_random_weights(self):
        """Forward smoke test with HF-default-init weights.

        Production loads weights via ``load_weights(hf_state_dict)``, so
        seeding through an HF tower's ``state_dict`` is the right sanity
        check. PyTorch-default init on TRT-LLM ``Linear`` / fused ``qkv_proj``
        can saturate the TRTLLM attention kernel at this small head_dim and
        produce NaN — that is a kernel-input contract, not a model bug, and
        is already covered indirectly by the parity tests below.
        """
        torch.manual_seed(0)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        tower = _build_trt_vision_tower(vision_cfg, dtype=torch.float32)
        tower.load_weights(dict(hf_tower.state_dict()))

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg, dtype=torch.float32)
        with torch.inference_mode():
            out = tower(pv, pos, output_length=output_length).last_hidden_state

        self.assertEqual(out.shape, torch.Size([output_length, vision_cfg.hidden_size]))
        self.assertFalse(out.isnan().any(), "Vision tower output contains NaN")

    @torch.no_grad()
    def test_load_weights_fuses_qkv(self):
        """HF ``q_proj / k_proj / v_proj`` collapse into the fused ``qkv_proj``.

        Validates the ``_load_weights_impl`` ``params_map`` regex
        (``self_attn.q_proj.linear.weight`` → ``self_attn.q_proj.weight``) plus
        the built-in qkv fusion (``qkv_proj`` ← stack(q,k,v)).

        When ``head_dim`` is padded to an FMHA-supported size (e.g. 32 → 64,
        72 → 80), the fused ``qkv_proj.weight`` has zero-padded tail channels
        per head; comparison must apply the same zero-pad to the HF reference.
        """
        torch.manual_seed(0)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)

        trt_tower.load_weights(dict(hf_tower.state_dict()))

        # Spot-check: fused qkv_proj.weight == vstack(zero-pad(q,k,v)) from HF.
        hf_sd = hf_tower.state_dict()
        for i in range(vision_cfg.num_hidden_layers):
            attn = trt_tower.encoder.layers[i].self_attn
            nh = attn.num_heads
            nkv = attn.num_key_value_heads
            hf_hd = attn.hf_head_dim
            padded_hd = attn.head_dim

            def _pad_head_dim(t: torch.Tensor, heads: int) -> torch.Tensor:
                # t: (heads * hf_hd, hidden) → (heads * padded_hd, hidden)
                if padded_hd == hf_hd:
                    return t
                t = t.view(heads, hf_hd, -1)
                zeros = t.new_zeros(heads, padded_hd - hf_hd, t.shape[-1])
                return torch.cat([t, zeros], dim=1).reshape(heads * padded_hd, -1)

            q = _pad_head_dim(hf_sd[f"encoder.layers.{i}.self_attn.q_proj.linear.weight"], nh)
            k = _pad_head_dim(hf_sd[f"encoder.layers.{i}.self_attn.k_proj.linear.weight"], nkv)
            v = _pad_head_dim(hf_sd[f"encoder.layers.{i}.self_attn.v_proj.linear.weight"], nkv)
            expected = torch.cat([q, k, v], dim=0)
            got = trt_tower.encoder.layers[i].self_attn.qkv_proj.weight.data
            torch.testing.assert_close(got.float(), expected.float(), atol=0, rtol=0)

    @torch.no_grad()
    def test_forward_parity_no_clamp(self):
        """HF ↔ TRT forward parity with ``use_clipped_linears=False``.

        Same shape/tolerance contract as ``gemma4_vision_smoke.py``
        Tier 1 (atol/rtol=5e-2) — the smoke at real dims hit
        diff.mean≈0.009; small dims should stay well inside.
        """
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = False
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg)
        with torch.inference_mode():
            hf_out = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            trt_out = trt_tower(pv, pos, output_length=output_length).last_hidden_state

        hf_flat = hf_out.reshape(-1, vision_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        atol, rtol = 5e-2, 5e-2
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=atol, rtol=rtol),
            f"max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )

    @torch.no_grad()
    def test_forward_batched_matches_per_image_loop(self):
        """B>1 single-call equivalence to per-image looped calls.

        The caller ``_get_image_features`` was rewritten to feed all images
        in one batched ``vision_tower`` call (varlen ``cu_seqlens`` attention).
        This regression test guards that the batched path is numerically
        equivalent to looping ``B=1`` calls and concatenating. If a future
        change introduces cross-image leakage (e.g. shared attn_metadata
        state not properly reset, or pooler bleeding across images), the
        loop-vs-batched diff catches it.
        """
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = False
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        B = 3
        pv_batched, pos_batched, output_length = _make_dummy_pixel_input(vision_cfg, B=B)

        # Path 1 (legacy): per-image loop, B=1 each, then concat.
        per_image = []
        with torch.inference_mode():
            for i in range(B):
                out_i = trt_tower(
                    pv_batched[i : i + 1],
                    pos_batched[i : i + 1],
                    output_length=output_length,
                ).last_hidden_state
                per_image.append(out_i)
        looped = torch.cat(per_image, dim=0)

        # Path 2 (new): single batched call.
        with torch.inference_mode():
            batched = trt_tower(
                pv_batched, pos_batched, output_length=output_length
            ).last_hidden_state

        # Shape must match (both flat, N_valid_total = B * output_length when
        # all images have full validity, as in this dummy input).
        self.assertEqual(looped.shape, batched.shape)
        # Values should be byte-equivalent (deterministic FULL attention on
        # both paths, no cross-image bleed). Allow a small atol for tiny
        # kernel-order tile reshuffles, but no rtol slack.
        diff = (looped - batched).abs()
        self.assertTrue(
            torch.allclose(looped, batched, atol=1e-4, rtol=0),
            f"Cross-image batched call diverges from per-image loop: "
            f"max diff={diff.max().item():.6f} mean diff={diff.mean().item():.6f}",
        )

    @torch.no_grad()
    def test_forward_image_seq_lens_kwarg_matches_default(self):
        """``image_seq_lens`` kwarg path is byte-equivalent to the default.

        The tower derives per-image seq_lens entirely on CPU — no GPU
        reduction, no D2H sync. Two CPU branches:
          - ``image_seq_lens`` is provided (production: ``Gemma4InputProcessor``
            emits it alongside ``image_position_ids``)
          - ``image_seq_lens is None`` and the dummy/test input has no ``-1``
            padding → fall back to ``[N] * B`` (every patch valid)

        For a full-grid synthetic input the two branches must produce
        identical output, since the explicit list ``[N] * B`` is exactly
        what the default fallback computes.
        """
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = False
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        B = 3
        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg, B=B)
        full_seq_lens = [pos.shape[1]] * B

        with torch.inference_mode():
            default_path = trt_tower(pv, pos, output_length=output_length).last_hidden_state
            explicit_path = trt_tower(
                pv, pos, output_length=output_length, image_seq_lens=full_seq_lens
            ).last_hidden_state

        self.assertEqual(default_path.shape, explicit_path.shape)
        diff = (default_path - explicit_path).abs()
        self.assertTrue(
            torch.allclose(default_path, explicit_path, atol=1e-4, rtol=0),
            f"image_seq_lens kwarg path diverges from default-all-valid path: "
            f"max diff={diff.max().item():.6f}",
        )


class TestGemma4VisionTowerClamp(unittest.TestCase):
    """``use_clipped_linears=True`` — exercises the clamp-buffer remap path.

    HF lays out clamps as:
      ``encoder.layers.{i}.self_attn.{q,k,v}_proj.{input,output}_{min,max}``
      ``encoder.layers.{i}.self_attn.o_proj.{input,output}_{min,max}``
    TRT-LLM collapses to:
      ``encoder.layers.{i}.self_attn.qkv_input_{min,max}`` (shared, q/k/v must agree)
      ``encoder.layers.{i}.self_attn.{q,k,v}_output_{min,max}``
      ``encoder.layers.{i}.self_attn.o_{input,output}_{min,max}``
    """

    @torch.no_grad()
    def test_clamp_buffers_remap_and_parity(self):
        torch.manual_seed(0)
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = True
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)

        # Inject finite, non-default clamps into HF state_dict so the remap
        # path has something concrete to copy (default is ±inf — a no-op).
        # Same value on q/k/v so the fusion assertion in
        # ``_remap_clamp_buffers`` is satisfied.
        hf_sd = dict(hf_tower.state_dict())
        QKV_LO, QKV_HI = -1.5, 1.5
        Q_OUT_LO, Q_OUT_HI = -2.0, 2.0
        K_OUT_LO, K_OUT_HI = -2.5, 2.5
        V_OUT_LO, V_OUT_HI = -3.0, 3.0
        O_IN_LO, O_IN_HI = -4.0, 4.0
        O_OUT_LO, O_OUT_HI = -5.0, 5.0
        for i in range(vision_cfg.num_hidden_layers):
            prefix = f"encoder.layers.{i}.self_attn"
            for sec in ("q", "k", "v"):
                hf_sd[f"{prefix}.{sec}_proj.input_min"] = torch.tensor(QKV_LO)
                hf_sd[f"{prefix}.{sec}_proj.input_max"] = torch.tensor(QKV_HI)
            hf_sd[f"{prefix}.q_proj.output_min"] = torch.tensor(Q_OUT_LO)
            hf_sd[f"{prefix}.q_proj.output_max"] = torch.tensor(Q_OUT_HI)
            hf_sd[f"{prefix}.k_proj.output_min"] = torch.tensor(K_OUT_LO)
            hf_sd[f"{prefix}.k_proj.output_max"] = torch.tensor(K_OUT_HI)
            hf_sd[f"{prefix}.v_proj.output_min"] = torch.tensor(V_OUT_LO)
            hf_sd[f"{prefix}.v_proj.output_max"] = torch.tensor(V_OUT_HI)
            hf_sd[f"{prefix}.o_proj.input_min"] = torch.tensor(O_IN_LO)
            hf_sd[f"{prefix}.o_proj.input_max"] = torch.tensor(O_IN_HI)
            hf_sd[f"{prefix}.o_proj.output_min"] = torch.tensor(O_OUT_LO)
            hf_sd[f"{prefix}.o_proj.output_max"] = torch.tensor(O_OUT_HI)

        # Mirror the injected values into HF buffers so HF and TRT both clamp.
        hf_tower.load_state_dict(hf_sd, strict=False)

        trt_tower.load_weights(hf_sd)

        # Buffers must now be live (non-inf) on TRT side.
        for i in range(vision_cfg.num_hidden_layers):
            attn = trt_tower.encoder.layers[i].self_attn
            self.assertAlmostEqual(attn.qkv_input_min.item(), QKV_LO, places=5)
            self.assertAlmostEqual(attn.qkv_input_max.item(), QKV_HI, places=5)
            self.assertAlmostEqual(attn.q_output_min.item(), Q_OUT_LO, places=5)
            self.assertAlmostEqual(attn.q_output_max.item(), Q_OUT_HI, places=5)
            self.assertAlmostEqual(attn.k_output_min.item(), K_OUT_LO, places=5)
            self.assertAlmostEqual(attn.k_output_max.item(), K_OUT_HI, places=5)
            self.assertAlmostEqual(attn.v_output_min.item(), V_OUT_LO, places=5)
            self.assertAlmostEqual(attn.v_output_max.item(), V_OUT_HI, places=5)
            self.assertAlmostEqual(attn.o_input_min.item(), O_IN_LO, places=5)
            self.assertAlmostEqual(attn.o_input_max.item(), O_IN_HI, places=5)
            self.assertAlmostEqual(attn.o_output_min.item(), O_OUT_LO, places=5)
            self.assertAlmostEqual(attn.o_output_max.item(), O_OUT_HI, places=5)

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg)
        with torch.inference_mode():
            hf_out = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            trt_out = trt_tower(pv, pos, output_length=output_length).last_hidden_state

        hf_flat = hf_out.reshape(-1, vision_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        atol, rtol = 5e-2, 5e-2
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=atol, rtol=rtol),
            f"clamped parity max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )

    def test_clamp_qkv_input_mismatch_raises(self):
        """q/k/v input clamps must agree — collapse must not silently lose info."""
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["use_clipped_linears"] = True
        vision_cfg = Gemma4VisionConfig(**cfg_dict)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)

        hf_sd = dict(hf_tower.state_dict())
        # Set q/k/v input_min to *different* values for layer 0 — load must fail.
        hf_sd["encoder.layers.0.self_attn.q_proj.input_min"] = torch.tensor(-1.0)
        hf_sd["encoder.layers.0.self_attn.k_proj.input_min"] = torch.tensor(-1.1)
        hf_sd["encoder.layers.0.self_attn.v_proj.input_min"] = torch.tensor(-1.0)
        hf_sd["encoder.layers.0.self_attn.q_proj.input_max"] = torch.tensor(1.0)
        hf_sd["encoder.layers.0.self_attn.k_proj.input_max"] = torch.tensor(1.0)
        hf_sd["encoder.layers.0.self_attn.v_proj.input_max"] = torch.tensor(1.0)

        with self.assertRaises(ValueError):
            trt_tower.load_weights(hf_sd)


# ---------------------------------------------------------------------------
# Multimodal embedder (unchanged from skeleton — already covers the
# refactored ``Gemma4MultimodalEmbedder``).
# ---------------------------------------------------------------------------


class TestGemma4MultimodalEmbedder(unittest.TestCase):
    """Test the multimodal embedder projection layer."""

    def test_embedder_instantiation(self):
        """Embedder creates norm + projection with correct dimensions."""
        embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=64,
            text_hidden_size=128,
            eps=1e-6,
            dtype=torch.bfloat16,
        )
        self.assertEqual(embedder.embedding_projection.weight.shape, torch.Size([128, 64]))

    @torch.no_grad()
    def test_embedder_forward(self):
        """Embedder forward: norm → linear, output matches expected shape."""
        embedder = (
            Gemma4MultimodalEmbedder(
                mm_hidden_size=64,
                text_hidden_size=128,
                dtype=torch.bfloat16,
            )
            .to("cuda")
            .to(torch.bfloat16)
        )
        torch.nn.init.xavier_uniform_(embedder.embedding_projection.weight.data)

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = embedder(x)
        self.assertEqual(out.shape, torch.Size([4, 128]))
        self.assertFalse(out.isnan().any())

    @torch.no_grad()
    def test_embedder_matches_hf(self):
        """Compare embedder output with HF Gemma4MultimodalEmbedder."""
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        dtype = torch.bfloat16
        device = "cuda"

        hf_embedder = HFGemma4MultimodalEmbedder(vision_cfg, text_cfg).to(dtype).to(device).eval()

        trt_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_cfg.hidden_size,
            eps=vision_cfg.rms_norm_eps,
            dtype=dtype,
        ).to(device)

        trt_embedder.embedding_projection.weight.data.copy_(
            hf_embedder.embedding_projection.weight.data
        )

        x = torch.randn(4, vision_cfg.hidden_size, device=device, dtype=dtype)
        with torch.inference_mode():
            hf_out = hf_embedder(x)
            trt_out = trt_embedder(x)

        self.assertTrue(
            torch.allclose(hf_out, trt_out, atol=1e-2),
            f"Embedder max diff: {(hf_out - trt_out).abs().max()}",
        )


# ---------------------------------------------------------------------------
# Vision + embedder end-to-end (no LLM, no real model required)
# ---------------------------------------------------------------------------


class TestGemma4VisionPipeline(unittest.TestCase):
    """Full vision-pipeline parity: pixel_values → tower → embedder → text_hidden.

    Replaces the pre-refactor ``test_vision_pipeline_matches_hf`` which built
    the tower via ``AutoModel.from_config`` and the projection via
    ``load_state_dict``. Both sides now use the refactored TRT-LLM tower +
    embedder; weights flow through the production ``load_weights`` path.
    """

    @torch.no_grad()
    def test_pipeline_matches_hf(self):
        torch.manual_seed(0)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        dtype = torch.float32
        device = "cuda"

        # --- HF pipeline ---
        hf_tower = HFGemma4VisionModel(vision_cfg).to(device).to(dtype).eval()
        hf_embedder = HFGemma4MultimodalEmbedder(vision_cfg, text_cfg).to(device).to(dtype).eval()

        # --- TRT-LLM pipeline ---
        trt_tower = _build_trt_vision_tower(vision_cfg, dtype=dtype, device=device)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        trt_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_cfg.hidden_size,
            eps=vision_cfg.rms_norm_eps,
            dtype=dtype,
        ).to(device)
        trt_embedder.embedding_projection.weight.data.copy_(
            hf_embedder.embedding_projection.weight.data
        )

        pv, pos, output_length = _make_dummy_pixel_input(
            vision_cfg, B=1, side=6, dtype=dtype, device=device
        )

        with torch.inference_mode():
            hf_hidden = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            hf_out = hf_embedder(hf_hidden)
            trt_hidden = trt_tower(pv, pos, output_length=output_length).last_hidden_state
            # Production ``_get_image_features`` adds a batch dim around the
            # embedder call. Mirror that exactly so the projection's input
            # shape contract is what production sees.
            trt_out = trt_embedder(trt_hidden.unsqueeze(0)).squeeze(0)

        # HF returns (B, output_length, H); TRT returns (N_valid, H). For B=1
        # with no padding, N_valid == output_length, so the flattened shapes
        # must match.
        hf_flat = hf_out.reshape(-1, text_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=5e-2, rtol=5e-2),
            f"Pipeline max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )


# ---------------------------------------------------------------------------
# Multimodal wrapper (LLM + vision tower + embedder)
# ---------------------------------------------------------------------------


class TestGemma4ForConditionalGeneration(unittest.TestCase):
    """Test the multimodal VLM wrapper."""

    def test_instantiation_with_vision(self):
        """VLM wrapper creates LLM + vision tower + embedder.

        Asserts ``vision_tower`` is the refactored TRT-LLM ``Gemma4VisionModel``
        (Option B), *not* the HF ``AutoModel`` placeholder used pre-refactor.
        """
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        config = Gemma4Config(
            text_config=text_cfg,
            vision_config=vision_cfg,
            audio_config=None,
        )

        mc = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend="FLASHINFER",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIsNotNone(model.llm)
        self.assertIsNotNone(model.vision_tower)
        self.assertIsNotNone(model.embed_vision)
        self.assertIsNone(model.audio_tower)
        self.assertIsNone(model.embed_audio)
        # Regression guard for Option B: the vision tower must be the native
        # TRT-LLM class, not ``transformers.AutoModel`` output.
        self.assertIsInstance(model.vision_tower, Gemma4VisionModel)

    def test_instantiation_without_vision(self):
        """VLM wrapper works text-only when vision_config is None."""
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        config = Gemma4Config(
            text_config=text_cfg,
            vision_config=None,
            audio_config=None,
        )

        mc = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend="FLASHINFER",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIsNotNone(model.llm)
        self.assertIsNone(model.vision_tower)
        self.assertIsNone(model.embed_vision)


# ---------------------------------------------------------------------------
# Input processor (real tokenizer/processor — gated on LLM_MODELS_ROOT)
# ---------------------------------------------------------------------------


def _get_model_path():
    llm_models_root = os.environ.get("LLM_MODELS_ROOT")
    if llm_models_root:
        return os.path.join(llm_models_root, "gemma4/gemma-4-26B-A4B-it")
    return None


MODEL_26B_PATH = _get_model_path()


def _model_available():
    if MODEL_26B_PATH is None:
        return False
    return os.path.isfile(os.path.join(MODEL_26B_PATH, "config.json"))


class TestGemma4InputProcessor(unittest.TestCase):
    """Tests for Gemma4InputProcessor with real model tokenizer/processor files."""

    @classmethod
    def setUpClass(cls):
        if not _model_available():
            raise unittest.SkipTest(f"Gemma4 model not found at {MODEL_26B_PATH}")

        from transformers import AutoConfig, AutoTokenizer

        cls.config = AutoConfig.from_pretrained(MODEL_26B_PATH)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_26B_PATH)

    def _make_processor(self):
        from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4InputProcessor

        return Gemma4InputProcessor(
            model_path=MODEL_26B_PATH,
            config=self.config,
            tokenizer=self.tokenizer,
            trust_remote_code=True,
        )

    def test_instantiation(self):
        """Input processor initializes with AutoProcessor."""
        proc = self._make_processor()
        self.assertIsNotNone(proc._processor)
        self.assertIsNotNone(proc._image_processor)
        self.assertEqual(proc.model_path, MODEL_26B_PATH)

    def test_text_only_processing(self):
        """Text-only input returns input_ids without multimodal data."""
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        inputs = {"prompt": "Hello, how are you?"}
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsInstance(input_ids, list)
        self.assertGreater(len(input_ids), 0)
        self.assertIsNone(mm_data)

    def test_image_processing(self):
        """Image input returns input_ids + pixel_values in multimodal data."""
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsInstance(input_ids, list)
        self.assertGreater(len(input_ids), 0)
        self.assertIsNotNone(mm_data)
        self.assertIn("multimodal_data", mm_data)
        self.assertIn("image", mm_data["multimodal_data"])
        img_data = mm_data["multimodal_data"]["image"]
        self.assertIn("pixel_values", img_data)
        self.assertEqual(img_data["pixel_values"].dim(), 3)

    def test_image_token_expansion(self):
        """Chat template expands <image> placeholder to image tokens."""
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "What is this?"},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, _ = proc(inputs, sp)

        image_token_id = self.config.image_token_id
        image_count = sum(1 for t in input_ids if t == image_token_id)
        self.assertGreater(image_count, 0, "Expected image tokens in expanded input_ids")

    def test_multiple_images(self):
        """Multiple images are handled correctly."""
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        img1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "Compare these images."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img1, img2]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsNotNone(mm_data)
        pv = mm_data["multimodal_data"]["image"]["pixel_values"]
        self.assertEqual(pv.shape[0], 2)

    def test_torch_tensor_image(self):
        """Torch tensor images disable rescaling."""
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        img_tensor = torch.randn(3, 224, 224)

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": [img_tensor]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsNotNone(mm_data)
        self.assertIn("pixel_values", mm_data["multimodal_data"]["image"])

    def test_processor_properties(self):
        """Verify processor property accessors."""
        proc = self._make_processor()
        self.assertIs(proc.config, self.config)
        self.assertIs(proc.tokenizer, self.tokenizer)
        self.assertEqual(proc.model_path, MODEL_26B_PATH)
        self.assertIsNotNone(proc.processor)

    def test_audio_processing(self):
        """Audio input returns input_ids + audio_features in multimodal data."""
        import numpy as np

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        audio = np.random.randn(16000).astype(np.float32)

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": "Transcribe the audio."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsInstance(input_ids, list)
        self.assertIsNotNone(mm_data)
        self.assertIn("multimodal_data", mm_data)
        self.assertIn("audio", mm_data["multimodal_data"])
        audio_data = mm_data["multimodal_data"]["audio"]
        self.assertIn("audio_features", audio_data)
        self.assertEqual(audio_data["audio_features"].dim(), 3)

    def test_audio_token_expansion(self):
        """Audio placeholder expands to audio soft-token positions."""
        import numpy as np

        from tensorrt_llm.sampling_params import SamplingParams

        audio_token_id = getattr(self.config, "audio_token_id", None)
        if audio_token_id is None:
            self.skipTest("Model config has no audio_token_id")

        proc = self._make_processor()
        audio = np.random.randn(16000).astype(np.float32)

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio"},
                        {"type": "text", "text": "Transcribe the audio."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, _ = proc(inputs, sp)

        audio_count = sum(1 for t in input_ids if t == audio_token_id)
        self.assertGreater(audio_count, 0, "Expected audio soft tokens in expanded input_ids")

    def test_audio_input_normalization(self):
        """_normalize_audio_inputs downmixes stereo and resamples."""
        import numpy as np

        from tensorrt_llm._torch.models.modeling_gemma4mm import _normalize_audio_inputs

        sr_src = 48000
        dur_sec = 0.5
        stereo = np.random.randn(int(sr_src * dur_sec), 2).astype(np.float32)
        normalized = _normalize_audio_inputs([(stereo, sr_src)], target_sr=16000)
        self.assertEqual(len(normalized), 1)
        arr = normalized[0]
        self.assertEqual(arr.ndim, 1)
        self.assertEqual(arr.dtype, np.float32)
        self.assertAlmostEqual(arr.size, 8000, delta=4)

    def test_audio_input_torch_tensor(self):
        """_normalize_audio_inputs accepts torch tensors."""
        import numpy as np

        from tensorrt_llm._torch.models.modeling_gemma4mm import _normalize_audio_inputs

        tensor = torch.randn(8000)
        normalized = _normalize_audio_inputs([tensor], target_sr=16000)
        self.assertEqual(normalized[0].dtype, np.float32)
        self.assertEqual(normalized[0].ndim, 1)
        self.assertEqual(normalized[0].size, 8000)

    def test_unknown_mm_modality_rejected(self):
        """Passing an unknown modality key raises KeyError."""
        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        inputs = {
            "prompt": "<|placeholder|>",
            "multi_modal_data": {"point_cloud": object()},
        }
        sp = SamplingParams(max_tokens=10)
        with self.assertRaises(KeyError):
            proc(inputs, sp)

    def test_video_processing_returns_pixel_values(self):
        """Video input bypasses HF Gemma4Processor and yields per-frame
        pixel_values + expanded video soft tokens in input_ids.
        """
        import numpy as np
        from PIL import Image

        from tensorrt_llm.sampling_params import SamplingParams

        proc = self._make_processor()
        if not hasattr(proc._processor, "video_processor"):
            self.skipTest("Processor has no video_processor")

        frames = [
            Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(32)
        ]
        VideoData = type("VideoData", (), {})
        vd = VideoData()
        vd.frames = frames

        prompt = proc._processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"video": [vd]},
        }
        sp = SamplingParams(max_tokens=10)
        input_ids, mm_data = proc(inputs, sp)

        self.assertIsNotNone(mm_data)
        self.assertIn("video", mm_data["multimodal_data"])
        video_data = mm_data["multimodal_data"]["video"]
        self.assertIn("pixel_values", video_data)
        self.assertEqual(video_data["pixel_values"].dim(), 3)
        self.assertEqual(video_data["pixel_values"].shape[0], 32)

        video_token_id = self._video_token_id()
        if video_token_id is not None:
            self.assertGreater(
                sum(1 for t in input_ids if t == video_token_id),
                0,
                "Expected video soft tokens in expanded input_ids",
            )

    def _video_token_id(self):
        return getattr(self.config, "video_token_id", None)


# ---------------------------------------------------------------------------
# Tier 2 — Full LLM+vision E2E via shared TestModelingMultimodal skeleton.
# ---------------------------------------------------------------------------
#
# Reuses ``test_modeling_multimodal.TestModelingMultimodal`` (the abstract
# base used by qwen3vl / nemotron_nano_v2_vl / etc.). Mirrors the qwen3vl
# pattern: provide config + class hooks, gate on LLM_MODELS_ROOT.
#
# Audio scenarios are intentionally not included — audio tower refactor is
# follow-up work.


_GEMMA4_E2E_PATH = (
    os.path.join(os.environ["LLM_MODELS_ROOT"], "gemma4/gemma-4-26B-A4B-it")
    if os.environ.get("LLM_MODELS_ROOT")
    else None
)


def _gemma4_e2e_model_available() -> bool:
    if _GEMMA4_E2E_PATH is None or not os.path.isfile(
        os.path.join(_GEMMA4_E2E_PATH, "config.json")
    ):
        return False
    # The shared multimodal skeleton's default `get_raw_test_inputs` reads
    # ``{LLM_MODELS_ROOT}/multimodals/test_data/seashore.png``; gate on that
    # too so the test reports skipped instead of failing in CI environments
    # where multimodal test fixtures aren't mirrored.
    test_img = os.path.join(
        os.environ["LLM_MODELS_ROOT"],
        "multimodals",
        "test_data",
        "seashore.png",
    )
    return os.path.isfile(test_img)


# Reduced-layer text+vision config for E2E. Real ``_name_or_path`` so the
# input processor + tokenizer pick up the genuine Gemma4 processor files;
# attention head_dim follows the existing dummy-config recipe in
# ``test_gemma4_e2e_dummy.py`` (128 / 256 for FlashInfer compatibility).
GEMMA4_E2E_CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "model_type": "gemma4",
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    # HF Gemma4 (transformers 5.5.3) does not advertise SDPA support; pin
    # to eager so PreTrainedModel.__init__ does not raise on dispatch check.
    "_attn_implementation": "eager",
    "image_token_id": 262145,
    "audio_token_id": 262273,
    "video_token_id": 262146,
    "boi_token_index": 255999,
    "eoi_token_index": 256000,
    "boa_token_index": 256001,
    "eoa_token_index": 256002,
    "_name_or_path": _GEMMA4_E2E_PATH or "",
    "text_config": {
        "model_type": "gemma4_text",
        "vocab_size": 262400,
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "global_head_dim": 256,
        "num_global_key_value_heads": 4,
        "hidden_activation": "gelu_pytorch_tanh",
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-6,
        "sliding_window": 1024,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "num_kv_shared_layers": 0,
        "hidden_size_per_layer_input": 0,
        "use_double_wide_mlp": False,
        "final_logit_softcapping": None,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    },
    "vision_config": {
        "model_type": "gemma4_vision",
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 72,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 10240,
        "use_clipped_linears": False,
        "standardize": True,
        "torch_dtype": "bfloat16",
    },
}


@pytest.mark.skipif(
    not _gemma4_e2e_model_available(),
    reason=(
        "LLM_MODELS_ROOT not set or Gemma4 model not available — "
        "E2E generate parity requires real tokenizer/processor files."
    ),
)
class TestGemma4MultimodalE2E(unittest.TestCase):
    """Full LLM + vision parity via the shared multimodal skeleton.

    Implemented as a thin wrapper around ``TestModelingMultimodal`` so we
    reuse the standardized context+generation phase, KV cache manager
    setup, HF input loading, and tolerance/compare helpers.
    """

    # The actual abstract base + skeleton live in
    # ``test_modeling_multimodal``. Importing it pulls in helpers that depend
    # on ``utils.llm_data.llm_models_root`` — they only load when the gate
    # above passes.

    def test_image_scenarios(self):
        from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal

        from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import (
            Gemma4HfWeightMapper,
        )

        class _Gemma4E2EImpl(TestModelingMultimodal):
            def get_model_config(self) -> Dict:
                return deepcopy(GEMMA4_E2E_CONFIG)

            def get_trtllm_model_class(self) -> Type:
                return Gemma4ForConditionalGeneration

            def get_hf_model_class(self) -> Type:
                return HFGemma4ForConditionalGeneration

            def get_weight_mapper_class(self) -> Type:
                return Gemma4HfWeightMapper

            def get_model_type(self) -> str:
                return "gemma4"

            def get_model_config_class(self) -> Type:
                return Gemma4Config

            def get_scenarios(self) -> List[MultimodalScenario]:
                # Single sanity-image scenario for the in-PR test list.
                # CUDA-graph / chunked-prefill / kv-cache-reuse permutations
                # are tracked separately in the integration test list.
                return [
                    MultimodalScenario(
                        modality="image",
                        use_cuda_graph=False,
                        chunked_prefill=False,
                        kv_cache_reuse=False,
                    ),
                ]

        # TestModelingMultimodal is abstract — instantiate a dynamic
        # subclass and run only ``test_all`` to keep the surface minimal.
        runner = _Gemma4E2EImpl("test_all")
        runner.setUp()
        try:
            runner.test_all()
        finally:
            runner.tearDown()


class TestGemma4VisionHeadDimPadding(unittest.TestCase):
    """Head_dim 72 (26B/31B vision) gets zero-padded to 80 so the trtllm-gen
    FMHA dispatcher finds a matching cubin (sm100a ships H64/H80/H128/H256/H512;
    no H72). qkv_proj/o_proj weights pad along the head dim; norm + RoPE run
    on the unpadded HF channels; the padded last channels stay zero through
    the kernel and don't contribute to the output (zero × anything = 0 in
    both QK^T and softmax(QK^T)V).

    Without this padding, vision attention falls into the unfused MHA path
    (``Fall back to unfused MHA``) with O(B × T²) workspace, OOMing 26B/31B
    MMMU at LLM-scale ``max_num_requests``.
    """

    HEAD_DIM_72_CONFIG = {
        # nh=2, head_dim=72 → hidden_size=144. Small enough to run on CPU/GPU
        # quickly while exercising the 72→80 padding path.
        "hidden_size": 144,
        "intermediate_size": 288,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 72,
        "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "position_embedding_size": 1024,
        "use_clipped_linears": False,
        "standardize": True,
        "model_type": "gemma4_vision",
    }

    def test_attention_overrides_head_dim_to_80(self):
        """``Gemma4VisionAttention`` should pad HF head_dim=72 to kernel
        head_dim=80 at init; qkv_proj weight dim should reflect padded size."""
        vision_cfg = Gemma4VisionConfig(**self.HEAD_DIM_72_CONFIG)
        trt_tower = _build_trt_vision_tower(vision_cfg)
        attn = trt_tower.encoder.layers[0].self_attn

        self.assertEqual(attn.hf_head_dim, 72)
        self.assertEqual(attn.head_dim, 80)
        self.assertEqual(attn.head_dim_pad, 8)
        # qkv_proj fused: q_size + 2*kv_size = nh*80 + 2*nkv*80 = (2 + 2*2)*80 = 480.
        expected_qkv_out = (
            vision_cfg.num_attention_heads + 2 * vision_cfg.num_key_value_heads
        ) * attn.head_dim
        self.assertEqual(attn.qkv_proj.weight.shape[0], expected_qkv_out)
        # q_norm operates on unpadded HF head_dim (72).
        self.assertEqual(attn.q_norm.weight.shape[0], 72)

    def test_attention_no_pad_when_head_dim_supported(self):
        """Head_dim already in FMHA cubin set (64) → padding is a no-op."""
        cfg_dict = deepcopy(SMALL_VISION_CONFIG)
        cfg_dict["head_dim"] = 64
        cfg_dict["hidden_size"] = 64 * cfg_dict["num_attention_heads"]
        vision_cfg = Gemma4VisionConfig(**cfg_dict)
        trt_tower = _build_trt_vision_tower(vision_cfg)
        attn = trt_tower.encoder.layers[0].self_attn

        self.assertEqual(attn.hf_head_dim, 64)
        self.assertEqual(attn.head_dim, 64)
        self.assertEqual(attn.head_dim_pad, 0)

    @torch.no_grad()
    def test_load_weights_zero_pads_qkv_and_o_proj(self):
        """Weight load should pad q/k/v_proj (rows) and o_proj (cols) head dim
        72→80 with zeros; the appended slices must be exactly zero."""
        vision_cfg = Gemma4VisionConfig(**self.HEAD_DIM_72_CONFIG)
        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        nh = vision_cfg.num_attention_heads
        for layer in trt_tower.encoder.layers:
            qkv_w = layer.self_attn.qkv_proj.weight.data
            # qkv_proj.weight shape: ((nh + 2*nkv) * 80, hidden_size).
            # Reshape to (heads_total, 80, hidden); last 8 channels of each
            # head must be exactly zero (zero-pad from load_weights).
            heads_total = nh + 2 * vision_cfg.num_key_value_heads
            qkv_w_3d = qkv_w.view(heads_total, 80, -1)
            self.assertTrue(
                torch.all(qkv_w_3d[:, 72:, :] == 0),
                "qkv_proj.weight last 8 channels per head must be zero (padding)",
            )

            o_w = layer.self_attn.o_proj.weight.data
            # o_proj.weight shape: (hidden_size, nh * 80) — pad last 8 cols per head.
            o_w_3d = o_w.view(-1, nh, 80)
            self.assertTrue(
                torch.all(o_w_3d[:, :, 72:] == 0),
                "o_proj.weight last 8 cols per head must be zero (padding)",
            )

    @torch.no_grad()
    def test_forward_parity_head_dim_72(self):
        """HF ↔ TRT forward parity at head_dim=72 with padding-to-80 dance.

        The zero-pad math is mathematically equivalent to unpadded HF
        computation: zeros in q/k don't contribute to QK^T, zeros in v
        produce zero output channels, zeros in o_proj's last columns ignore
        those zero channels. Same fp32 tolerance as the head_dim=32 parity
        test (``test_forward_parity_no_clamp``).
        """
        vision_cfg = Gemma4VisionConfig(**self.HEAD_DIM_72_CONFIG)

        hf_tower = HFGemma4VisionModel(vision_cfg).to("cuda").to(torch.float32).eval()
        trt_tower = _build_trt_vision_tower(vision_cfg)
        trt_tower.load_weights(dict(hf_tower.state_dict()))

        pv, pos, output_length = _make_dummy_pixel_input(vision_cfg)
        with torch.inference_mode():
            hf_out = hf_tower(pv, pos, output_length=output_length).last_hidden_state
            trt_out = trt_tower(pv, pos, output_length=output_length).last_hidden_state

        hf_flat = hf_out.reshape(-1, vision_cfg.hidden_size)
        self.assertEqual(hf_flat.shape, trt_out.shape)
        diff = (hf_flat - trt_out).abs()
        atol, rtol = 5e-2, 5e-2
        self.assertTrue(
            torch.allclose(hf_flat, trt_out, atol=atol, rtol=rtol),
            f"head_dim=72 padded-to-80 diverges from HF: "
            f"max diff={diff.max().item():.4f} mean diff={diff.mean().item():.4f}",
        )


if __name__ == "__main__":
    unittest.main()
