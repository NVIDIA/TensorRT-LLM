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
"""Tests for the Step3p7 multimodal bring-up (``modeling_step3p7vl``).

The text decoder + MTP wiring is covered by ``test_modeling_step3p7.py``; this
module focuses on the Perception-Encoder vision tower and the VLM registration.

TestStep3p7VisionTower — vision tower component tests (no checkpoint, no GPU).
A shrunken synthetic ``vision_config`` (see ``_make_tiny_vision_config``) keeps
the real per-layer geometry (Conv2d patch embed -> pre-LN transformer blocks
with 2D RoPE + LayerScale -> two stride-2 Conv2d downsamplers -> linear
projector) while letting the tower build and run on CPU in well under a second:
  - 2D RoPE helpers (zero-frequency identity at grid position 0, freq-cache
    shape, dynamic sub-grid selection)
  - LayerScale / fused-QKV attention / MLP / pre-LN block component contracts
  - ``Step3p7VisionEncoder`` forward output shape, including the abs-posemb
    bilinear interpolation path for off-grid image sizes
  - ``Step3p7VisionTower`` projector geometry, ``vision_model.*`` /
    ``vit_large_projector.*`` weight-prefix splitting, and the per-request
    ``[patches... | full image]`` embedding flattening contract

TestStep3p7VLRegistration — registry verification:
  - ``Step3p7ForConditionalGeneration`` resolves to the VLM entry point
  - ``Step3p7VisionTower`` is registered as the architecture's vision encoder
  - the multimodal input processor is registered for ``model_type == step3p7``

TestStep3p7VLCheckpoint — checkpoint-backed ``vision_config`` geometry tests
(requires the Step-3.7-Flash checkpoints under ``LLM_MODELS_ROOT``). The FP8
block-scale, NVFP4, and BF16 reference checkpoints all ship the same
PerceptionEncoder vision config; only the routed-expert dtype/layout of the
text decoder differs.
"""

import json
import os
import unittest

import torch
from parameterized import parameterized
from PIL import Image
from transformers import PretrainedConfig
from utils.llm_data import llm_models_root

# The multimodal checkpoint is the same on disk as the text one; the FP8
# block-scale, NVFP4, and BF16 reference checkpoints all ship the same
# PerceptionEncoder vision tower (only the text decoder's routed-expert
# dtype/layout differs). See test_modeling_step3p7.py for the text-path tests.
STEP3P7_FP8_DIR = str(os.path.join(llm_models_root(), "Step-3.7-Flash-FP8"))
STEP3P7_NVFP4_DIR = str(os.path.join(llm_models_root(), "Step-3.7-Flash-NVFP4"))
STEP3P7_BF16_DIR = str(os.path.join(llm_models_root(), "Step-3.7-Flash"))


def _load_config(checkpoint_dir: str) -> dict:
    with open(os.path.join(checkpoint_dir, "config.json")) as f:
        return json.load(f)


def _make_tiny_vision_config(
    width: int = 64,
    heads: int = 4,
    layers: int = 2,
    patch_size: int = 8,
    image_size: int = 64,
    ls_init_value: float = 0.1,
    use_cls_token: bool = False,
    use_ln_post: bool = False,
    hidden_act: str = "quick_gelu",
) -> PretrainedConfig:
    """Tiny ``vision_config`` mirroring the Step-3.7-Flash PerceptionEncoder.

    Field names match the real checkpoint (``width`` / ``heads`` / ``layers``
    rather than ``hidden_size`` / ``num_heads`` / ``num_hidden_layers``) so the
    encoder reads them exactly as it would from ``config.json``. Sizes are
    shrunk so the tower builds and runs on CPU in well under a second; float32
    keeps the identity invariants below numerically exact.
    """
    cfg = PretrainedConfig()
    cfg.model_type = "perception_encoder"
    cfg.width = width
    cfg.heads = heads
    cfg.layers = layers
    cfg.patch_size = patch_size
    cfg.image_size = image_size
    cfg.hidden_act = hidden_act
    cfg.ls_init_value = ls_init_value
    cfg.use_cls_token = use_cls_token
    cfg.use_ln_post = use_ln_post
    cfg.torch_dtype = torch.float32
    return cfg


def _make_tiny_vision_model_config(text_hidden_size: int = 32, **vision_kwargs):
    """Wrap a tiny vision config in a ``ModelConfig`` the vision tower accepts.

    ``Step3p7VisionTower`` reads ``vision_config``, ``text_config.hidden_size``,
    ``torch_dtype``, ``image_token_id`` and ``projector_bias`` off the top-level
    pretrained config; supply just those.
    """
    from tensorrt_llm._torch.model_config import ModelConfig

    vision_cfg = _make_tiny_vision_config(**vision_kwargs)
    top = PretrainedConfig()
    top.torch_dtype = torch.float32
    top.vision_config = vision_cfg
    text_cfg = PretrainedConfig()
    text_cfg.hidden_size = text_hidden_size
    top.text_config = text_cfg
    top.image_token_id = 128001
    top.projector_bias = False
    return ModelConfig(pretrained_config=top)


class TestStep3p7VisionTower(unittest.TestCase):
    """Perception-Encoder vision tower tests — no checkpoint and no GPU."""

    @staticmethod
    def _downsampled_tokens(grid: int) -> int:
        """Token count after the two trailing stride-2 / kernel-3 / pad-1 convs.

        Each downsampler maps spatial ``s -> floor((s - 1) / 2) + 1``; applied
        twice this equals ``grid // 4`` for the grid sizes used here, matching
        the encoder's documented ``(Gh//4) * (Gw//4)`` output.
        """
        after1 = (grid - 1) // 2 + 1
        after2 = (after1 - 1) // 2 + 1
        return after2 * after2

    # ----- 2D RoPE -------------------------------------------------------

    def test_rope2d_freqs_cache_shape(self):
        """Cached frequencies are ``(1, 1, Gh*Gw, head_dim)``."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionRope2D

        head_dim, gh, gw = 16, 4, 4
        rope = Step3VisionRope2D(dim=head_dim, max_grid_height=gh, max_grid_width=gw)
        self.assertEqual(tuple(rope.freqs_cache.shape), (1, 1, gh * gw, head_dim))

    def test_rope2d_position_zero_is_identity(self):
        """At grid position (0, 0) the 2D frequencies are zero → cos=1, sin=0.

        The first sequence position must therefore pass through unchanged while
        a later position is actually rotated.
        """
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionRope2D

        head_dim, gh, gw = 16, 4, 4
        rope = Step3VisionRope2D(dim=head_dim, max_grid_height=gh, max_grid_width=gw)
        q = torch.randn(1, 2, gh * gw, head_dim)
        k = torch.randn(1, 2, gh * gw, head_dim)
        q_out, k_out = rope(q, k, grid_hw=(gh, gw))
        self.assertEqual(q_out.shape, q.shape)
        self.assertTrue(torch.allclose(q_out[..., 0, :], q[..., 0, :], atol=1e-6))
        self.assertTrue(torch.allclose(k_out[..., 0, :], k[..., 0, :], atol=1e-6))
        self.assertFalse(torch.allclose(q_out[..., -1, :], q[..., -1, :], atol=1e-4))

    def test_rope2d_dynamic_subgrid_selects_positions(self):
        """A grid smaller than the cached max grid hits the index-select path."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionRope2D

        head_dim = 16
        rope = Step3VisionRope2D(dim=head_dim, max_grid_height=8, max_grid_width=8)
        gh, gw = 4, 4
        q = torch.randn(1, 2, gh * gw, head_dim)
        k = torch.randn(1, 2, gh * gw, head_dim)
        q_out, k_out = rope(q, k, grid_hw=(gh, gw))
        self.assertEqual(q_out.shape, (1, 2, gh * gw, head_dim))
        self.assertEqual(k_out.shape, (1, 2, gh * gw, head_dim))

    # ----- components ----------------------------------------------------

    def test_layer_scale_scales_by_gamma(self):
        """LayerScale multiplies its input element-wise by the per-channel gamma."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionLayerScale

        ls = Step3VisionLayerScale(dim=4, init_value=2.0)
        x = torch.ones(1, 3, 4)
        self.assertTrue(torch.allclose(ls(x), x * 2.0))

    def test_vision_mlp_names_and_shape(self):
        """The FFN keeps the HF ``c_fc`` / ``c_proj`` parameter names."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionMLP

        mlp = Step3VisionMLP(hidden_size=8, intermediate_size=16, hidden_act="quick_gelu")
        self.assertTrue(hasattr(mlp, "c_fc"))
        self.assertTrue(hasattr(mlp, "c_proj"))
        out = mlp(torch.randn(2, 5, 8))
        self.assertEqual(out.shape, (2, 5, 8))

    def test_vision_attention_fused_qkv_layout_and_shape(self):
        """HF fused-QKV layout (``in_proj_weight`` / ``in_proj_bias``) and shape."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionAttention

        hidden, heads, gh, gw = 64, 4, 4, 4
        attn = Step3VisionAttention(
            hidden_size=hidden,
            num_heads=heads,
            max_grid_height=gh,
            max_grid_width=gw,
            use_cls_token=False,
            use_rope2d=True,
        )
        self.assertEqual(tuple(attn.in_proj_weight.shape), (3 * hidden, hidden))
        self.assertEqual(tuple(attn.in_proj_bias.shape), (3 * hidden,))
        out = attn(torch.randn(2, gh * gw, hidden), grid_hw=(gh, gw))
        self.assertEqual(out.shape, (2, gh * gw, hidden))

    def test_vision_attention_rejects_indivisible_heads(self):
        """``hidden_size`` not divisible by ``num_heads`` is rejected up front."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionAttention

        with self.assertRaises(ValueError):
            Step3VisionAttention(
                hidden_size=65,
                num_heads=4,
                max_grid_height=4,
                max_grid_width=4,
                use_cls_token=False,
                use_rope2d=False,
            )

    def test_vision_block_zero_layerscale_is_identity(self):
        """With ``ls_init_value=0`` both residual branches are scaled to zero, so
        a pre-LN block reduces to the identity."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionBlock

        hidden, heads, gh, gw = 64, 4, 4, 4
        block = Step3VisionBlock(
            hidden_size=hidden,
            num_heads=heads,
            mlp_ratio=2.0,
            hidden_act="quick_gelu",
            layer_norm_eps=1e-5,
            ls_init_value=0.0,
            max_grid_height=gh,
            max_grid_width=gw,
            use_cls_token=False,
            use_rope2d=True,
            rope_theta=10000.0,
            rope_theta_rescale_factor=1.0,
        )
        x = torch.randn(1, gh * gw, hidden)
        out = block(x, grid_hw=(gh, gw))
        self.assertTrue(torch.allclose(out, x, atol=1e-6))

    # ----- encoder -------------------------------------------------------

    def test_vision_encoder_forward_output_shape(self):
        """Encoder returns ``(B, (Gh//4)*(Gw//4), 4*width)`` post-downsample."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionEncoder

        width, patch, image = 64, 8, 64
        enc = Step3p7VisionEncoder(_make_tiny_vision_config(width=width), dtype=torch.float32)
        with torch.inference_mode():
            feats = enc(torch.randn(1, 3, image, image))
        grid = image // patch
        self.assertEqual(feats.shape, (1, self._downsampled_tokens(grid), 4 * width))

    def test_vision_encoder_smaller_image_interpolates_posemb(self):
        """An off-grid (smaller) image triggers bilinear abs-posemb interpolation
        and the RoPE sub-grid path, still yielding a well-formed feature map."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionEncoder

        width, patch, image = 64, 8, 128  # base grid 16
        enc = Step3p7VisionEncoder(
            _make_tiny_vision_config(width=width, patch_size=patch, image_size=image),
            dtype=torch.float32,
        )
        smaller = 64  # grid 8 < base grid 16
        with torch.inference_mode():
            feats = enc(torch.randn(1, 3, smaller, smaller))
        grid = smaller // patch
        self.assertEqual(feats.shape, (1, self._downsampled_tokens(grid), 4 * width))

    # ----- tower (encoder + projector) -----------------------------------

    def test_vision_tower_projector_geometry(self):
        """Projector maps ``4*width -> text hidden_size``; tower honours dtype."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower

        tower = Step3p7VisionTower(_make_tiny_vision_model_config(text_hidden_size=32, width=64))
        self.assertEqual(tower.vit_large_projector.in_features, 4 * 64)
        self.assertEqual(tower.vit_large_projector.out_features, 32)
        self.assertIsNone(tower.vit_large_projector.bias)  # projector_bias=False
        self.assertEqual(tower.dtype, torch.float32)

    def test_vision_tower_encode_projects_to_text_hidden(self):
        """``_encode`` runs the encoder + projector to the text hidden size."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower

        tower = Step3p7VisionTower(
            _make_tiny_vision_model_config(
                text_hidden_size=32, width=64, patch_size=8, image_size=64
            )
        )
        with torch.inference_mode():
            out = tower._encode(torch.randn(1, 3, 64, 64))
        self.assertEqual(out.shape, (1, self._downsampled_tokens(64 // 8), 32))

    def test_vision_tower_forward_flattens_patches_then_image(self):
        """``forward`` lays each request out as ``[patches... | full image]``.

        Matches the input processor's placeholder expansion: per image, the
        per-patch feature blocks come first (in order), then the full-image
        block, all flattened to ``(num_tokens, text_hidden)``.
        """
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower
        from tensorrt_llm.inputs.multimodal import MultimodalParams

        tower = Step3p7VisionTower(
            _make_tiny_vision_model_config(
                text_hidden_size=32, width=64, patch_size=8, image_size=64
            )
        )
        tokens_per_tile = self._downsampled_tokens(64 // 8)

        # One full image, no patches → just the full-image block.
        mm_image_only = MultimodalParams(
            multimodal_data={"image": {"pixel_values": torch.randn(1, 3, 64, 64)}}
        )
        out = tower.forward([mm_image_only])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (tokens_per_tile, 32))

        # One image with 2 patches → 2 patch blocks then the full-image block.
        mm_with_patches = MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": torch.randn(1, 3, 64, 64),
                    "patch_pixel_values": torch.randn(2, 3, 64, 64),
                    "num_patches": [2],
                }
            }
        )
        out = tower.forward([mm_with_patches])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (3 * tokens_per_tile, 32))

        # No image payload → no embeddings produced.
        self.assertEqual(tower.forward([MultimodalParams(multimodal_data={})]), [])

    def test_vision_tower_load_weights_splits_prefixes(self):
        """``load_weights`` routes ``vision_model.*`` to the encoder and
        ``vit_large_projector.*`` to the projector, ignoring unrelated keys."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower

        tower = Step3p7VisionTower(_make_tiny_vision_model_config(text_hidden_size=32, width=64))
        proj_w = torch.ones_like(tower.vit_large_projector.weight)
        conv_w = torch.ones_like(tower.vision_model.conv1.weight)
        weights = {
            "vit_large_projector.weight": proj_w,
            "vision_model.conv1.weight": conv_w,
            # A text-decoder key the tower must leave untouched.
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        }
        tower.load_weights(weights)
        self.assertTrue(torch.equal(tower.vit_large_projector.weight.detach(), proj_w))
        self.assertTrue(torch.equal(tower.vision_model.conv1.weight.detach(), conv_w))


class TestStep3p7VLRegistration(unittest.TestCase):
    """Verify the Step3p7 multimodal architecture and its encoder are registered."""

    def test_vlm_entry_point_and_vision_encoder_registered(self):
        from tensorrt_llm._torch.models.modeling_step3p7vl import (
            Step3p7VisionTower,
            Step3p7VLForConditionalGeneration,
        )
        from tensorrt_llm._torch.models.modeling_utils import (
            MODEL_CLASS_MAPPING,
            MODEL_CLASS_VISION_ENCODER_MAPPING,
        )

        self.assertIs(
            MODEL_CLASS_MAPPING.get("Step3p7ForConditionalGeneration"),
            Step3p7VLForConditionalGeneration,
        )
        entry = MODEL_CLASS_VISION_ENCODER_MAPPING.get("Step3p7ForConditionalGeneration")
        self.assertIsNotNone(entry, "vision encoder not registered for Step3p7")
        vision_cls, _vlm_base = entry
        self.assertIs(vision_cls, Step3p7VisionTower)

    def test_input_processor_registered_for_model_type(self):
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VLInputProcessor

        # ``register_input_processor`` stamps the model_type onto the processor
        # class; the placeholder is the OOV-rewritten ``<im_patch>`` token.
        self.assertEqual(Step3p7VLInputProcessor._registered_model_type, "step3p7")


class TestStep3p7VLCheckpoint(unittest.TestCase):
    """``vision_config`` geometry checks against the real Step-3.7-Flash checkpoints.

    The FP8 block-scale, NVFP4, and BF16 reference checkpoints all ship the same
    PerceptionEncoder vision tower; only the text decoder's routed-expert
    dtype/layout differs. The checkpoints are expected under ``LLM_MODELS_ROOT``
    (as in CI); a missing checkpoint surfaces as a test failure, not a skip.
    """

    def _check_vision_config(self, vision_cfg: dict):
        self.assertEqual(vision_cfg["model_type"], "perception_encoder")
        self.assertEqual(vision_cfg["width"], 1536)
        self.assertEqual(vision_cfg["heads"], 16)
        self.assertEqual(vision_cfg["layers"], 47)
        self.assertEqual(vision_cfg["patch_size"], 14)
        self.assertEqual(vision_cfg["image_size"], 728)
        self.assertEqual(vision_cfg["hidden_act"], "quick_gelu")
        self.assertEqual(vision_cfg["ls_init_value"], 0.1)
        self.assertIs(vision_cfg["use_cls_token"], False)
        # The vision MHA requires width divisible by the head count.
        self.assertEqual(vision_cfg["width"] % vision_cfg["heads"], 0)

    @parameterized.expand(
        [
            ("fp8", STEP3P7_FP8_DIR),
            ("nvfp4", STEP3P7_NVFP4_DIR),
            ("bf16", STEP3P7_BF16_DIR),
        ]
    )
    def test_vision_config_geometry(self, name, checkpoint_dir):
        """All three checkpoints carry the same PerceptionEncoder vision config
        plus the multimodal wiring (image token id, biasless projector)."""
        config_dict = _load_config(checkpoint_dir)
        self.assertIn("vision_config", config_dict)
        self._check_vision_config(config_dict["vision_config"])
        self.assertEqual(config_dict["image_token_id"], 128001)
        self.assertIs(config_dict.get("projector_bias", False), False)


class TestStep3p7VLInputProcessorHooks(unittest.TestCase):
    """Multimodal-hashing hooks on ``Step3p7VLInputProcessor``.

    Step3 image spans interleave structural framing tokens
    (``<im_start>``/``<im_end>``/``<patch_start>``/``<patch_end>``/
    ``<patch_newline>``) with the ``<im_patch>`` embed slots. These tests
    verify the processor exposes the per-image token count and framing-token
    ids the generic hashing path needs to keep each image one contiguous span
    (the KV-cache-reuse / chunked-prefill prerequisite). Requires the
    Step-3.7-Flash checkpoint under ``LLM_MODELS_ROOT``.
    """

    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoTokenizer

        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VLInputProcessor

        cls.config = AutoConfig.from_pretrained(STEP3P7_BF16_DIR, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(STEP3P7_BF16_DIR, trust_remote_code=True)
        cls.proc = Step3p7VLInputProcessor(
            STEP3P7_BF16_DIR, cls.config, tokenizer, trust_remote_code=True
        )

    def test_special_and_mm_token_ids(self):
        """All five framing tokens resolve; mm_token_ids = sentinel + framing."""
        special = self.proc.get_mm_special_token_ids()
        self.assertIsNotNone(special)
        self.assertEqual(special.numel(), 5)
        # Distinct and resolved (not collapsed to a single unk id).
        self.assertEqual(len(set(special.tolist())), 5)

        mm_ids = self.proc.get_mm_token_ids()
        self.assertIsNotNone(mm_ids)
        self.assertEqual(mm_ids[0].item(), self.proc._tllm_multimodal_token_id)
        self.assertEqual(sorted(mm_ids[1:].tolist()), sorted(special.tolist()))

    def test_num_tokens_per_image_matches_processor(self):
        """The hook delegates to the remote processor's span-length logic."""
        img = Image.new("RGB", (800, 600))
        n = self.proc.get_num_tokens_per_image(image=img)
        self.assertEqual(n, self.proc._processor.get_num_image_tokens(800, 600))
        self.assertGreater(n, 0)
        # CHW tensor (h, w) must agree with the PIL (w, h) path.
        n_tensor = self.proc.get_num_tokens_per_image(image=torch.zeros(3, 600, 800))
        self.assertEqual(n_tensor, n)

    def test_image_span_is_contiguous(self):
        """End-to-end: the hashing masks cover one contiguous image span whose
        length equals ``get_num_tokens_per_image``."""
        from tensorrt_llm.inputs.multimodal import _compute_mm_masks
        from tensorrt_llm.sampling_params import SamplingParams

        img = Image.new("RGB", (800, 600))
        inputs = {"prompt": "<im_patch>", "multi_modal_data": {"image": [img]}}
        token_ids, _ = self.proc(inputs, SamplingParams())
        ids = torch.tensor(token_ids)

        mm_mask, embed_mask, special_mask = _compute_mm_masks(
            ids,
            vocab_size=self.proc.get_vocab_size(),
            mm_token_ids=self.proc.get_mm_token_ids(),
            mm_special_token_ids=self.proc.get_mm_special_token_ids(),
        )

        expected = self.proc.get_num_tokens_per_image(image=img)
        self.assertEqual(int(mm_mask.sum()), expected)
        # Embed slots (sentinels) + framing tokens partition the span.
        self.assertEqual(int(embed_mask.sum()) + int(special_mask.sum()), expected)
        # The span is a single contiguous run.
        positions = mm_mask.nonzero().flatten()
        self.assertEqual(int(positions[-1] - positions[0]) + 1, positions.numel())
        # Embed slots are exactly the OOV sentinels.
        self.assertTrue(bool((ids[embed_mask] == self.proc._tllm_multimodal_token_id).all()))


if __name__ == "__main__":
    unittest.main()
