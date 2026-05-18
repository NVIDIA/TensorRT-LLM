# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SD3 / SD3.5 family adapter for the auto path.

Targets Diffusers `SD3Transformer2DModel`. Architecture differs from Flux.1:

- **Input shape**: `hidden_states` is ``(B, C, H, W)`` raw image (not
  pre-patched) — patching happens inside the transformer via `PatchEmbed`.
- **No RoPE**: position information is in a *learned* `PatchEmbed`
  (additive), so there are no cos/sin tensors in the captured graph. The
  ``qk_rope_fusion`` matchers correctly skip these sites because their
  pattern keys on an ``aten.add.Tensor`` consuming two muls — no such
  node exists when RoPE is absent.
- **All joint attention**: every block is a `JointTransformerBlock` (no
  single stream). First 13 layers in SD3.5 also have a
  ``use_dual_attention=True`` extra self-attention on the hidden side
  (an additional single-stream site per affected block).
- **qk_norm**: SD3.5 uses per-head RMSNorm (same as Flux.1). SD3 (without
  .5) uses none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig

# Example dims for export tracing. H/W refer to LATENT dimensions (the shape
# the transformer actually sees — `hidden_states` is `(B, in_channels, H_lat,
# W_lat)` where the VAE has already downsampled the image by 8×).  For a
# 512×512 image, H_lat = W_lat = 64. Kept *static* at the actual latent
# resolution; broad dynamic H/W would trip many divisibility guards in
# SD3's `PatchEmbed`.
_EXAMPLE_BATCH = 2
_DEFAULT_LATENT_H = 64
_DEFAULT_LATENT_W = 64
_EXAMPLE_TXT_SEQ = 64
_EXAMPLE_TIMESTEP = 500


def _arch_dim(cfg: "DiffusionModelConfig", attr: str, default: int) -> int:
    pc = getattr(cfg, "pretrained_config", None)
    return int(getattr(pc, attr, default)) if pc is not None else default


def _cfg_dim(cfg: "DiffusionModelConfig", attr: str, default: int) -> int:
    """Read a value from cfg, with a default fallback (cfg attrs may not exist)."""
    return int(getattr(cfg, attr, default))


class SD3Adapter(VisGenFamilyAdapter):
    family = "SD3"
    diffusers_transformer_cls = "SD3Transformer2DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        in_channels = _arch_dim(cfg, "in_channels", 16)
        joint_dim = _arch_dim(cfg, "joint_attention_dim", 4096)
        pooled_dim = _arch_dim(cfg, "pooled_projection_dim", 2048)

        B = _EXAMPLE_BATCH
        H = _cfg_dim(cfg, "latent_height", _DEFAULT_LATENT_H)
        W = _cfg_dim(cfg, "latent_width", _DEFAULT_LATENT_W)
        S_txt = _EXAMPLE_TXT_SEQ

        kwargs: dict[str, Any] = {
            "hidden_states": torch.randn(B, in_channels, H, W, device=device, dtype=dtype),
            "encoder_hidden_states": torch.randn(B, S_txt, joint_dim, device=device, dtype=dtype),
            "pooled_projections": torch.randn(B, pooled_dim, device=device, dtype=dtype),
            # SD3 uses FlowMatchEulerDiscreteScheduler — timestep is a float
            # (sigma value), not a Long. The forward signature is annotated
            # `LongTensor` but Diffusers' SD3 pipeline passes FP32 (the
            # scheduler's default).  The transformer casts internally.
            "timestep": torch.full(
                (B,), float(_EXAMPLE_TIMESTEP), device=device, dtype=torch.float32
            ),
            "return_dict": False,
        }
        return (), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        # H/W are static (captured at example_inputs size). Making them
        # dynamic requires modeling the model's `pos_embed_max_size` bound
        # and internal divisibility constraints (H % 2, W % 2) — defer to a
        # follow-up. For now: capture per resolution; if you change H/W,
        # re-capture.
        B = torch.export.Dim("B", min=1, max=8)
        S_txt = torch.export.Dim("S_txt", min=1, max=512)
        return {
            "hidden_states": {0: B},
            "encoder_hidden_states": {0: B, 1: S_txt},
            "pooled_projections": {0: B},
            "timestep": {0: B},
            "return_dict": None,
        }

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        return RewritePolicy(attention_backend=cfg.attention.backend)

    @property
    def uses_internal_seq_shard(self) -> bool:
        # SD3's `hidden_states` is 4-D `(B, C, H, W)` like a video DiT —
        # the boundary-level dim-1 shard would slice channels, not the
        # post-pos_embed sequence. Use the in-model shard pattern (mirroring
        # WAN's): slice after `self.pos_embed(...)` produces the flat
        # `(B, S, hidden)` sequence, all-gather before `norm_out`.
        return True

    def pre_capture_patch(self, model, visual_gen_mapping) -> None:
        ulysses_size = getattr(visual_gen_mapping, "ulysses_size", 1) if visual_gen_mapping else 1
        if ulysses_size <= 1:
            return
        ulysses_rank = getattr(visual_gen_mapping, "ulysses_rank", 0)

        from types import MethodType

        def _sd3_ulysses_forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            pooled_projections=None,
            timestep=None,
            return_dict: bool = False,
            **_unused,  # joint_attention_kwargs, block_controlnet_hidden_states, skip_layers
        ):
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            height, width = hidden_states.shape[-2:]

            # pos_embed produces token-flat `(B, S, inner_dim)`.
            hidden_states = self.pos_embed(hidden_states)

            # --- Ulysses shard on the flat image-stream sequence ---
            P = ulysses_size
            R = ulysses_rank
            S = hidden_states.shape[1]
            chunk = S // P
            hidden_states = hidden_states[:, R * chunk : (R + 1) * chunk, :].contiguous()
            # NOTE: SD3 is MM-DiT — each block concatenates text+image inside
            # for joint attention. Text (`encoder_hidden_states`) is kept
            # full on every rank; only the image stream is sharded. The
            # UlyssesAttention inside `visgen_auto.sdpa` will all-to-all the
            # concatenated stream, redistributing across heads.
            # --------------------------------------------------------

            temb = self.time_text_embed(timestep, pooled_projections)
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=None,
                )

            # --- Gather image sequence back to full S ---
            hidden_states = torch.ops.visgen_auto.all_gather_seq(hidden_states, 1, ulysses_size)
            # --------------------------------------------

            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            patch_size = self.config.patch_size
            h2 = height // patch_size
            w2 = width // patch_size
            hidden_states = hidden_states.reshape(
                (hidden_states.shape[0], h2, w2, patch_size, patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                (hidden_states.shape[0], self.out_channels, h2 * patch_size, w2 * patch_size)
            )
            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        model.forward = MethodType(_sd3_ulysses_forward, model)
