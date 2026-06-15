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

TestStep3p7VisionTower — vision tower component tests (no checkpoint).
A shrunken synthetic ``vision_config`` (see ``_make_tiny_vision_config``) keeps
the real per-layer geometry (Conv2d patch embed -> pre-LN transformer blocks
with 2D RoPE + LayerScale -> two stride-2 Conv2d downsamplers -> linear
projector) while letting the tower build and run in well under a second. The
2D-RoPE / LayerScale / MLP invariants stay on CPU (float32); the attention,
block, encoder and tower forwards dispatch through the TRT-LLM ``Attention``
(TRTLLM FMHA backend) and are therefore GPU-only:
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

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping

# The PerceptionEncoder head_dim is not in the FMHA cubin set, so the vision
# attention is dispatched through TRT-LLM's ``Attention`` (TRTLLM backend) with
# head_dim zero-padding — that path needs a GPU. Module-forward tests are
# therefore GPU-only; the pure-PyTorch component invariants (2D RoPE, LayerScale,
# MLP) stay on CPU and remain numerically exact in float32.
requires_gpu = unittest.skipUnless(
    torch.cuda.is_available(), "vision attention dispatch requires a GPU"
)

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


# The vision tower runs through the FMHA dispatch, which requires a bf16/fp16
# activation dtype; the GPU module-forward tests therefore build the tiny tower
# in bf16. The CPU component tests construct their modules directly in float32.
_GPU_DTYPE = torch.bfloat16


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
    torch_dtype: torch.dtype = _GPU_DTYPE,
) -> PretrainedConfig:
    """Tiny ``vision_config`` mirroring the Step-3.7-Flash PerceptionEncoder.

    Field names match the real checkpoint (``width`` / ``heads`` / ``layers``
    rather than ``hidden_size`` / ``num_heads`` / ``num_hidden_layers``) so the
    encoder reads them exactly as it would from ``config.json``. Sizes are
    shrunk so the tower builds and runs in well under a second. The shrunken
    head_dim (``width // heads`` = 16) is zero-padded to the FMHA-supported 64,
    exercising the same padding path the real 96 → 128 case uses.
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
    cfg.torch_dtype = torch_dtype
    return cfg


def _make_tiny_vision_model_config(
    text_hidden_size: int = 32, torch_dtype: torch.dtype = _GPU_DTYPE, **vision_kwargs
):
    """Wrap a tiny vision config in a ``ModelConfig`` the vision tower accepts.

    ``Step3p7VisionTower`` reads ``vision_config``, ``text_config.hidden_size``,
    ``torch_dtype``, ``image_token_id`` and ``projector_bias`` off the top-level
    pretrained config; supply just those.
    """
    vision_cfg = _make_tiny_vision_config(torch_dtype=torch_dtype, **vision_kwargs)
    top = PretrainedConfig()
    top.torch_dtype = torch_dtype
    top.vision_config = vision_cfg
    text_cfg = PretrainedConfig()
    text_cfg.hidden_size = text_hidden_size
    top.text_config = text_cfg
    top.image_token_id = 128001
    top.projector_bias = False
    return ModelConfig(pretrained_config=top)


def _vision_attention_model_config(vision_config: PretrainedConfig) -> ModelConfig:
    """Single-rank, quant-disabled ``ModelConfig`` for the vision submodules.

    Mirrors what ``Step3p7VisionTower`` builds internally so the attention /
    block / encoder can be constructed in isolation by the component tests.
    """
    return ModelConfig(
        pretrained_config=vision_config,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        attn_backend="TRTLLM",
        skip_create_weights_in_init=False,
    )


def _vision_attn_metadata(seq_lens):
    """Context-only (no KV cache) attention metadata for a flat varlen stream."""
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend

    md = get_attention_backend("TRTLLM").Metadata(
        max_num_requests=64,
        max_num_tokens=max(8192, int(sum(seq_lens))),
        kv_cache_manager=None,
    )
    md.num_contexts = len(seq_lens)
    md.request_ids = list(range(1, len(seq_lens) + 1))
    md.prompt_lens = list(seq_lens)
    md.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
    md.max_seq_len = max(seq_lens)
    md.prepare()
    return md


def _setup_encoder_attn_metadata(module, max_num_tokens: int = 8192):
    """Mirror the engine's ``_set_up_multimodal_encoder_attn_metadata`` walk.

    The encoder builds its ``AttentionMetadata`` via the engine-driven
    ``MultimodalEncoderMixin.setup_attn_metadata`` after model load; standalone
    tests that construct the encoder/tower directly must do the same before the
    encoder forward. Returns ``module`` for chaining.
    """
    from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin

    for m in module.modules():
        if isinstance(m, MultimodalEncoderMixin):
            m.setup_attn_metadata(max_num_requests=max_num_tokens, max_num_tokens=max_num_tokens)
    return module


@torch.no_grad()
def _init_finite_weights(module, std: float = 0.02):
    """Fill all parameters with small finite values.

    TRT-LLM ``Linear`` / ``Attention`` allocate their weights uninitialized;
    on GPU that surfaces as garbage (``inf``) until a checkpoint is loaded.
    Value-comparing tests seed deterministic finite weights instead."""
    for p in module.parameters():
        p.normal_(0.0, std)


class TestStep3p7VisionTower(unittest.TestCase):
    """Perception-Encoder vision tower tests (no checkpoint).

    The 2D-RoPE / LayerScale / MLP component invariants run on CPU in float32.
    Module-forward tests (attention, block, encoder, tower) dispatch through the
    TRT-LLM ``Attention`` (TRTLLM FMHA backend) and are GPU-only.
    """

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
        """The FFN reuses the TRT-LLM ``MLP`` module (``up_proj`` / ``down_proj``)."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionMLP

        mc = _vision_attention_model_config(_make_tiny_vision_config(width=8, heads=2))
        mlp = Step3VisionMLP(
            model_config=mc,
            hidden_size=8,
            intermediate_size=16,
            hidden_act="quick_gelu",
            dtype=torch.float32,
        )
        # HF ``c_fc`` / ``c_proj`` are remapped onto the base module's
        # ``up_proj`` / ``down_proj`` in ``_remap_vision_weights``.
        self.assertTrue(hasattr(mlp, "up_proj"))
        self.assertTrue(hasattr(mlp, "down_proj"))
        self.assertEqual(tuple(mlp.up_proj.weight.shape), (16, 8))
        self.assertEqual(tuple(mlp.down_proj.weight.shape), (8, 16))
        out = mlp(torch.randn(2, 5, 8))
        self.assertEqual(out.shape, (2, 5, 8))

    @requires_gpu
    def test_vision_attention_fused_qkv_layout_and_shape(self):
        """Ported attention exposes fused ``qkv_proj`` / ``o_proj`` with the
        FMHA-padded head_dim, and returns the flat ``(num_tokens, hidden)`` shape.
        """
        from tensorrt_llm._torch.models.modeling_step3p7vl import (
            Step3VisionAttention,
            Step3VisionRope2D,
        )

        hidden, heads, gh, gw = 64, 4, 4, 4
        mc = _vision_attention_model_config(_make_tiny_vision_config(width=hidden, heads=heads))
        attn = (
            Step3VisionAttention(
                mc, hidden_size=hidden, num_heads=heads, layer_idx=0, dtype=_GPU_DTYPE
            )
            .cuda()
            .eval()
        )
        # head_dim 16 is padded up to the FMHA-supported 64.
        self.assertEqual(attn.hf_head_dim, 16)
        self.assertEqual(attn.head_dim, 64)
        self.assertEqual(tuple(attn.qkv_proj.weight.shape), (3 * heads * 64, hidden))
        self.assertEqual(tuple(attn.o_proj.weight.shape), (hidden, heads * 64))

        seq = gh * gw
        x = torch.randn(2 * seq, hidden, device="cuda", dtype=_GPU_DTYPE)
        md = _vision_attn_metadata([seq, seq])
        rope = Step3VisionRope2D(dim=16, max_grid_height=gh, max_grid_width=gw).cuda()
        freqs = rope.freqs_for_grid((gh, gw), x.device).unsqueeze(1).repeat(2, 1, 1)
        out = attn(x, md, (freqs.cos(), freqs.sin()))
        self.assertEqual(out.shape, (2 * seq, hidden))

    def test_vision_attention_rejects_indivisible_heads(self):
        """``hidden_size`` not divisible by ``num_heads`` is rejected up front."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3VisionAttention

        mc = _vision_attention_model_config(_make_tiny_vision_config())
        with self.assertRaises(ValueError):
            Step3VisionAttention(mc, hidden_size=65, num_heads=4, layer_idx=0, dtype=torch.float32)

    @requires_gpu
    def test_vision_attention_matches_reference_sdpa(self):
        """The padded FMHA path is numerically equal to a raw-SDPA reference.

        Guards the head_dim zero-padding + ``q_scaling`` (softmax scale must be
        ``hf_head_dim ** -0.5``) + the FULL (bidirectional) mask + 2D RoPE on the
        real channels: a reference built from the attention's own real (unpadded)
        QKV/o_proj channels must match the kernel output.
        """
        import torch.nn.functional as F

        from tensorrt_llm._torch.models.modeling_step3p7vl import (
            Step3VisionAttention,
            Step3VisionRope2D,
            _apply_rotary_emb,
        )

        hidden, heads, gh, gw = 64, 4, 4, 4
        mc = _vision_attention_model_config(_make_tiny_vision_config(width=hidden, heads=heads))
        attn = (
            Step3VisionAttention(
                mc, hidden_size=hidden, num_heads=heads, layer_idx=0, dtype=_GPU_DTYPE
            )
            .cuda()
            .eval()
        )
        _init_finite_weights(attn)
        nh, pd, hf = attn.num_heads, attn.head_dim, attn.hf_head_dim
        seq = gh * gw
        x = torch.randn(seq, hidden, device="cuda", dtype=_GPU_DTYPE)
        rope = Step3VisionRope2D(dim=hf, max_grid_height=gh, max_grid_width=gw).cuda()
        freqs = rope.freqs_for_grid((gh, gw), x.device).unsqueeze(1)

        out = attn(x, _vision_attn_metadata([seq]), (freqs.cos(), freqs.sin()))

        # Reference: real (unpadded) channels + SDPA with scale = hf ** -0.5.
        qkv_w = attn.qkv_proj.weight.view(3, nh, pd, hidden)[:, :, :hf, :].reshape(
            3 * nh * hf, hidden
        )
        qkv_b = attn.qkv_proj.bias.view(3, nh, pd)[:, :, :hf].reshape(-1)
        q, k, v = F.linear(x, qkv_w, qkv_b).chunk(3, dim=-1)
        q = _apply_rotary_emb(freqs, q.view(seq, nh, hf))
        k = _apply_rotary_emb(freqs, k.view(seq, nh, hf))
        v = v.view(seq, nh, hf)
        ref = F.scaled_dot_product_attention(
            q.permute(1, 0, 2).unsqueeze(0),
            k.permute(1, 0, 2).unsqueeze(0),
            v.permute(1, 0, 2).unsqueeze(0),
            is_causal=False,
            scale=hf**-0.5,
        )
        ref = ref.squeeze(0).permute(1, 0, 2).reshape(seq, nh * hf)
        o_w = attn.o_proj.weight.view(hidden, nh, pd)[:, :, :hf].reshape(hidden, nh * hf)
        ref = F.linear(ref, o_w, attn.o_proj.bias)
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    @requires_gpu
    def test_vision_block_zero_layerscale_is_identity(self):
        """With ``ls_init_value=0`` both residual branches are scaled to zero, so
        a pre-LN block reduces to the identity."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import (
            Step3VisionBlock,
            Step3VisionRope2D,
        )

        hidden, heads, gh, gw = 64, 4, 4, 4
        mc = _vision_attention_model_config(_make_tiny_vision_config(width=hidden, heads=heads))
        block = (
            Step3VisionBlock(
                mc,
                layer_idx=0,
                hidden_size=hidden,
                num_heads=heads,
                mlp_ratio=2.0,
                hidden_act="quick_gelu",
                layer_norm_eps=1e-5,
                ls_init_value=0.0,
                dtype=_GPU_DTYPE,
            )
            # The encoder normally casts the whole tower to the model dtype;
            # do the same here so the directly-built LayerScale/MLP are bf16.
            .to(_GPU_DTYPE)
            .cuda()
            .eval()
        )
        # Finite attention weights (so 0 * attn is 0, not 0 * inf = NaN), but
        # keep the LayerScale gammas at their constructed 0 to test the identity.
        _init_finite_weights(block)
        block.ls_1.gamma.data.zero_()
        block.ls_2.gamma.data.zero_()
        seq = gh * gw
        x = torch.randn(seq, hidden, device="cuda", dtype=_GPU_DTYPE)
        rope = Step3VisionRope2D(dim=16, max_grid_height=gh, max_grid_width=gw).cuda()
        freqs = rope.freqs_for_grid((gh, gw), x.device).unsqueeze(1)
        out = block(x, _vision_attn_metadata([seq]), (freqs.cos(), freqs.sin()))
        # ls=0 ⇒ both residual branches contribute nothing ⇒ output == input.
        self.assertTrue(torch.allclose(out.float(), x.float(), atol=1e-6))

    # ----- encoder -------------------------------------------------------

    @requires_gpu
    def test_vision_encoder_forward_output_shape(self):
        """Encoder returns ``(B, (Gh//4)*(Gw//4), 4*width)`` post-downsample."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionEncoder

        width, patch, image = 64, 8, 64
        vision_cfg = _make_tiny_vision_config(width=width, patch_size=patch, image_size=image)
        enc = _setup_encoder_attn_metadata(
            Step3p7VisionEncoder(
                _vision_attention_model_config(vision_cfg), vision_cfg, dtype=_GPU_DTYPE
            )
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            feats = enc(torch.randn(1, 3, image, image, device="cuda", dtype=_GPU_DTYPE))
        grid = image // patch
        self.assertEqual(feats.shape, (1, self._downsampled_tokens(grid), 4 * width))

    @requires_gpu
    def test_vision_encoder_smaller_image_interpolates_posemb(self):
        """An off-grid (smaller) image triggers bilinear abs-posemb interpolation
        and the RoPE sub-grid path, still yielding a well-formed feature map."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionEncoder

        width, patch, image = 64, 8, 128  # base grid 16
        vision_cfg = _make_tiny_vision_config(width=width, patch_size=patch, image_size=image)
        enc = _setup_encoder_attn_metadata(
            Step3p7VisionEncoder(
                _vision_attention_model_config(vision_cfg), vision_cfg, dtype=_GPU_DTYPE
            )
            .cuda()
            .eval()
        )
        smaller = 64  # grid 8 < base grid 16
        with torch.inference_mode():
            feats = enc(torch.randn(1, 3, smaller, smaller, device="cuda", dtype=_GPU_DTYPE))
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
        self.assertEqual(tower.dtype, _GPU_DTYPE)

    @requires_gpu
    def test_vision_tower_encode_projects_to_text_hidden(self):
        """``_encode`` runs the encoder + projector to the text hidden size."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower

        tower = _setup_encoder_attn_metadata(
            Step3p7VisionTower(
                _make_tiny_vision_model_config(
                    text_hidden_size=32, width=64, patch_size=8, image_size=64
                )
            )
            .cuda()
            .eval()
        )
        with torch.inference_mode():
            out = tower._encode(torch.randn(1, 3, 64, 64, device="cuda", dtype=_GPU_DTYPE))
        self.assertEqual(out.shape, (1, self._downsampled_tokens(64 // 8), 32))

    @requires_gpu
    def test_vision_tower_forward_flattens_patches_then_image(self):
        """``forward`` lays each request out as ``[patches... | full image]``.

        Matches the input processor's placeholder expansion: per image, the
        per-patch feature blocks come first (in order), then the full-image
        block, all flattened to ``(num_tokens, text_hidden)``.
        """
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower
        from tensorrt_llm.inputs.multimodal import MultimodalParams

        tower = _setup_encoder_attn_metadata(
            Step3p7VisionTower(
                _make_tiny_vision_model_config(
                    text_hidden_size=32, width=64, patch_size=8, image_size=64
                )
            )
            .cuda()
            .eval()
        )
        tokens_per_tile = self._downsampled_tokens(64 // 8)

        def _img(*shape):
            return torch.randn(*shape, device="cuda", dtype=_GPU_DTYPE)

        # One full image, no patches → just the full-image block.
        mm_image_only = MultimodalParams(
            multimodal_data={"image": {"pixel_values": _img(1, 3, 64, 64)}}
        )
        out = tower.forward([mm_image_only])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (tokens_per_tile, 32))

        # One image with 2 patches → 2 patch blocks then the full-image block.
        mm_with_patches = MultimodalParams(
            multimodal_data={
                "image": {
                    "pixel_values": _img(1, 3, 64, 64),
                    "patch_pixel_values": _img(2, 3, 64, 64),
                    "num_patches": [2],
                }
            }
        )
        out = tower.forward([mm_with_patches])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].shape, (3 * tokens_per_tile, 32))

        # No image payload → no embeddings produced.
        self.assertEqual(tower.forward([MultimodalParams(multimodal_data={})]), [])

    @requires_gpu
    def test_vision_tower_batches_across_requests(self):
        """Batching images across requests yields the same per-request features
        as encoding each request alone (TODO #2 correctness)."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower
        from tensorrt_llm.inputs.multimodal import MultimodalParams

        tower = _setup_encoder_attn_metadata(
            Step3p7VisionTower(
                _make_tiny_vision_model_config(
                    text_hidden_size=32, width=64, patch_size=8, image_size=64
                )
            )
            .cuda()
            .eval()
        )
        _init_finite_weights(tower)
        img_a = torch.randn(1, 3, 64, 64, device="cuda", dtype=_GPU_DTYPE)
        img_b = torch.randn(1, 3, 64, 64, device="cuda", dtype=_GPU_DTYPE)
        mm_a = MultimodalParams(multimodal_data={"image": {"pixel_values": img_a}})
        mm_b = MultimodalParams(multimodal_data={"image": {"pixel_values": img_b}})

        with torch.inference_mode():
            batched = tower.forward([mm_a, mm_b])[0]
            only_a = tower.forward([mm_a])[0]
            only_b = tower.forward([mm_b])[0]

        half = batched.shape[0] // 2
        torch.testing.assert_close(batched[:half].float(), only_a.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(batched[half:].float(), only_b.float(), atol=2e-2, rtol=2e-2)

    def test_vision_tower_load_weights_remaps_and_routes(self):
        """``load_weights`` remaps HF fused ``in_proj`` / ``out_proj`` onto the
        ported ``qkv_proj`` / ``o_proj`` (with head_dim zero-padding) and HF
        ``mlp.c_fc`` / ``mlp.c_proj`` onto the ported ``up_proj`` / ``down_proj``,
        routes ``vision_model.*`` / ``vit_large_projector.*`` correctly, and
        ignores unrelated text-decoder keys."""
        from tensorrt_llm._torch.models.modeling_step3p7vl import Step3p7VisionTower

        tower = Step3p7VisionTower(_make_tiny_vision_model_config(text_hidden_size=32, width=64))
        # Seed finite weights so the inverted HF state is meaningful (freshly
        # allocated TRT-LLM weights are uninitialized and may contain NaN).
        _init_finite_weights(tower)
        nh = tower.vision_model.num_heads
        hidden = tower.vision_model.hidden_size
        hf = hidden // nh

        # Build an HF-style vision state (in_proj_*/out_proj) by inverting the
        # current ported state: drop the padded head channels.
        hf_vision = {}
        for key, val in tower.vision_model.state_dict().items():
            if key.endswith(".attn.qkv_proj.weight"):
                pref = key[: -len("qkv_proj.weight")]
                pd = val.shape[0] // (3 * nh)
                hf_vision[pref + "in_proj_weight"] = (
                    val.view(3, nh, pd, hidden)[:, :, :hf, :].reshape(3 * nh * hf, hidden).clone()
                )
            elif key.endswith(".attn.qkv_proj.bias"):
                pref = key[: -len("qkv_proj.bias")]
                pd = val.shape[0] // (3 * nh)
                hf_vision[pref + "in_proj_bias"] = (
                    val.view(3, nh, pd)[:, :, :hf].reshape(-1).clone()
                )
            elif key.endswith(".attn.o_proj.weight"):
                pref = key[: -len("o_proj.weight")]
                pd = val.shape[1] // nh
                hf_vision[pref + "out_proj.weight"] = (
                    val.view(hidden, nh, pd)[:, :, :hf].reshape(hidden, nh * hf).clone()
                )
            elif key.endswith(".attn.o_proj.bias"):
                hf_vision[key[: -len("o_proj.bias")] + "out_proj.bias"] = val.clone()
            elif key.endswith(".mlp.up_proj.weight"):
                hf_vision[key[: -len("up_proj.weight")] + "c_fc.weight"] = val.clone()
            elif key.endswith(".mlp.up_proj.bias"):
                hf_vision[key[: -len("up_proj.bias")] + "c_fc.bias"] = val.clone()
            elif key.endswith(".mlp.down_proj.weight"):
                hf_vision[key[: -len("down_proj.weight")] + "c_proj.weight"] = val.clone()
            elif key.endswith(".mlp.down_proj.bias"):
                hf_vision[key[: -len("down_proj.bias")] + "c_proj.bias"] = val.clone()
            else:
                hf_vision[key] = val.clone()

        weights = {f"vision_model.{k}": v for k, v in hf_vision.items()}
        weights.update(
            {
                f"vit_large_projector.{k}": v.clone()
                for k, v in tower.vit_large_projector.state_dict().items()
            }
        )
        proj_w = torch.ones_like(tower.vit_large_projector.weight)
        conv_w = torch.ones_like(tower.vision_model.conv1.weight)
        weights["vit_large_projector.weight"] = proj_w
        weights["vision_model.conv1.weight"] = conv_w
        # A text-decoder key the tower must leave untouched (and not route).
        weights["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(2, 2)

        tower.load_weights(weights)
        self.assertTrue(torch.equal(tower.vit_large_projector.weight.detach(), proj_w))
        self.assertTrue(torch.equal(tower.vision_model.conv1.weight.detach(), conv_w))

        # Remap correctness: the loaded fused qkv real channels equal the HF
        # in_proj, and the appended (padded) channels are exactly zero.
        blk0 = tower.vision_model.transformer.resblocks[0].attn
        pd = blk0.head_dim
        qkv = blk0.qkv_proj.weight.detach().view(3, nh, pd, hidden)
        self.assertTrue(
            torch.allclose(
                qkv[:, :, :hf, :].reshape(3 * nh * hf, hidden),
                hf_vision["transformer.resblocks.0.attn.in_proj_weight"],
            )
        )
        self.assertEqual(qkv[:, :, hf:, :].abs().max().item(), 0.0)

        # MLP remap: HF ``c_fc`` / ``c_proj`` land verbatim on ``up_proj`` /
        # ``down_proj`` (no padding -- the FFN dims are kernel-agnostic).
        blk0_mlp = tower.vision_model.transformer.resblocks[0].mlp
        self.assertTrue(
            torch.equal(
                blk0_mlp.up_proj.weight.detach(),
                hf_vision["transformer.resblocks.0.mlp.c_fc.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                blk0_mlp.down_proj.weight.detach(),
                hf_vision["transformer.resblocks.0.mlp.c_proj.weight"],
            )
        )


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
