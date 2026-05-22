# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PixArt family adapter (PixArt-Σ / PixArt-α).

Targets Diffusers `PixArtTransformer2DModel`. Architecture:
- `hidden_states` is 4-D `(B, C, H, W)` raw latents — patchified internally
  via `PatchEmbed`.
- Single-stream attention (no MM-DiT joint), with cross-attention to text
  embeddings (`encoder_hidden_states`) of shape `(B, S_text, caption_channels)`.
- `caption_channels` (default 4096) ≠ inner_dim (1152 for Σ-XL).
- Uses `AdaLayerNormSingle` (`norm_type="ada_norm_single"`) — needs
  `added_cond_kwargs={"resolution":..., "aspect_ratio":...}` only when
  `use_additional_conditions=True`. The Σ-XL config doesn't set this
  to True by default, but the pipeline passes `added_cond_kwargs` anyway.
- `qk_norm`: none (no per-head RMSNorm) → `fuse_qk_rope=False`. Also
  no RoPE — positions come from learned PatchEmbed, like SD3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig


_EX_B = 2  # CFG-doubled batch (PixArt uses guidance_scale > 1 by default)
_DEFAULT_LATENT_H = 128  # Σ-XL @ 1024² → 1024/8 = 128
_DEFAULT_LATENT_W = 128
_EX_TXT_SEQ = 300  # PixArt-Σ uses T5-XXL with default max_sequence_length=300


def _arch_dim(cfg: "DiffusionModelConfig", attr: str, default: int) -> int:
    pc = getattr(cfg, "pretrained_config", None)
    return int(getattr(pc, attr, default)) if pc is not None else default


class PixArtAdapter(VisGenFamilyAdapter):
    family = "PixArt"
    diffusers_transformer_cls = "PixArtTransformer2DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        in_channels = _arch_dim(cfg, "in_channels", 4)
        caption_channels = _arch_dim(cfg, "caption_channels", 4096)

        B = _EX_B
        H = _DEFAULT_LATENT_H
        W = _DEFAULT_LATENT_W
        S_txt = _EX_TXT_SEQ

        # PixArtSigmaPipeline calls the transformer with `hidden_states`
        # POSITIONAL (see diffusers/pipelines/pixart_alpha/pipeline_pixart_sigma.py).
        # Pass it as a positional in example_inputs so capture matches.
        hidden_states = torch.randn(B, in_channels, H, W, device=device, dtype=dtype)
        kwargs: dict[str, Any] = {
            "encoder_hidden_states": torch.randn(
                B, S_txt, caption_channels, device=device, dtype=dtype
            ),
            "timestep": torch.full((B,), 500, device=device, dtype=torch.long),
            "encoder_attention_mask": torch.ones(B, S_txt, device=device, dtype=torch.long),
            # PixArt-Σ XL-2 has `use_additional_conditions=False`; the
            # diffusers `PixArtSigmaPipeline` passes
            # `added_cond_kwargs = {"resolution": None, "aspect_ratio": None}`
            # (both values literally None). If we capture with tensors here,
            # the captured graph traces tensor `.size()` ops that crash on
            # None at runtime. Match the pipeline's call shape literally.
            "added_cond_kwargs": {"resolution": None, "aspect_ratio": None},
            "return_dict": False,
        }
        return (hidden_states,), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        # All-static for first-light. PixArt-Σ has internal divisibility
        # asserts on H/W (PatchEmbed) and on S_txt (caption_projection); the
        # cleanest start is to specialize the captured graph at the example
        # shape, which matches runtime CFG batch=2 + S_txt=64.
        # Flat dict keyed by forward-signature parameter names — torch.export
        # matches by name across (args, kwargs).
        return {
            "hidden_states": None,
            "encoder_hidden_states": None,
            "timestep": None,
            "encoder_attention_mask": None,
            "added_cond_kwargs": {
                "resolution": None,
                "aspect_ratio": None,
            },
            "return_dict": None,
        }

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        # PixArt has no QK-RMSNorm (just plain SDPA after Q/K projections).
        return RewritePolicy(
            attention_backend=cfg.attention.backend,
            fuse_qk_rope=False,
        )

    @property
    def uses_internal_seq_shard(self) -> bool:
        # 4-D `(B, C, H, W)` input → patchify inside via `self.pos_embed`.
        # Same shard pattern as SD3: slice the flat sequence *after*
        # `pos_embed`, all-gather before `norm_out`/`proj_out`.
        return True

    def pre_capture_patch(self, model, visual_gen_mapping) -> None:
        ulysses_size = getattr(visual_gen_mapping, "ulysses_size", 1) if visual_gen_mapping else 1
        if ulysses_size <= 1:
            return
        ulysses_rank = getattr(visual_gen_mapping, "ulysses_rank", 0)

        from types import MethodType

        def _pixart_ulysses_forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            timestep=None,
            added_cond_kwargs=None,
            cross_attention_kwargs=None,
            attention_mask=None,
            encoder_attention_mask=None,
            return_dict: bool = False,
        ):
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            # Attention mask → bias (matches diffusers' built-in conversion).
            if attention_mask is not None and attention_mask.ndim == 2:
                attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
                attention_mask = attention_mask.unsqueeze(1)
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            batch_size = hidden_states.shape[0]
            height = hidden_states.shape[-2] // self.config.patch_size
            width = hidden_states.shape[-1] // self.config.patch_size

            # 1. pos_embed produces token-flat `(B, S, inner_dim)`.
            hidden_states = self.pos_embed(hidden_states)

            # --- Ulysses shard on the flat image-stream sequence ---
            P = ulysses_size
            R = ulysses_rank
            S = hidden_states.shape[1]
            chunk = S // P
            hidden_states = hidden_states[:, R * chunk : (R + 1) * chunk, :].contiguous()
            # Text (`encoder_hidden_states`) is kept full on every rank;
            # the cross-attention in each block has full text K/V and
            # sharded image Q → output is sharded image, no gather needed.
            # ----------------------------------------------------------

            timestep, embedded_timestep = self.adaln_single(
                timestep,
                added_cond_kwargs,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )

            if self.caption_projection is not None:
                encoder_hidden_states = self.caption_projection(encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states.view(
                    batch_size, -1, hidden_states.shape[-1]
                )

            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )

            # --- Gather image sequence back to full S before output ---
            hidden_states = torch.ops.visgen_auto.all_gather_seq(hidden_states, 1, ulysses_size)
            # ----------------------------------------------------------

            shift, scale = (
                self.scale_shift_table[None]
                + embedded_timestep[:, None].to(self.scale_shift_table.device)
            ).chunk(2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(
                hidden_states.device
            )
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

            # Unpatchify.
            hidden_states = hidden_states.reshape(
                shape=(
                    -1,
                    height,
                    width,
                    self.config.patch_size,
                    self.config.patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    self.out_channels,
                    height * self.config.patch_size,
                    width * self.config.patch_size,
                )
            )
            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        model.forward = MethodType(_pixart_ulysses_forward, model)
