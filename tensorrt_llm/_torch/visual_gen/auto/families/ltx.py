# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LTX-Video family adapter for the auto path.

Targets Diffusers `LTXVideoTransformer3DModel` (LTX-Video, not the
audiovisual LTX-2). Forward signature is image-DiT-shaped:

- `hidden_states`: `(B, S_video, in_channels)` — already patched + flattened
  by Diffusers' `LTXVideoPipeline` before calling the transformer
- `encoder_hidden_states`: `(B, S_text, cross_attention_dim)`
- `timestep`: `(B,)`
- `encoder_attention_mask`: `(B, S_text)` (used as additive bias internally)
- plus optional `num_frames / height / width` ints, `video_coords`,
  `rope_interpolation_scale`, etc. — we leave at defaults for the spike

LTX-2 (`LTX2VideoTransformer3DModel`) is the audiovisual variant with 17+
kwargs including audio inputs and per-modality sigmas — out of scope here,
needs its own adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig


_EX_B = 2  # CFG-doubled runtime batch — Dim.AUTO with hint=1 specializes
_EX_VIDEO_SEQ = 256
_EX_TEXT_SEQ = 64
# Patched by the runner before capture so the int kwargs (num_frames/height/
# width) get baked into the captured graph at the runtime production shape
# (otherwise diffusers' default 25 frames @ 704×512 → static-int guard fail
# at any other shape).
_EX_NUM_FRAMES = 1
_EX_H = 16
_EX_W = 16


class LTXAdapter(VisGenFamilyAdapter):
    family = "LTX-Video"
    diffusers_transformer_cls = "LTXVideoTransformer3DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        pc = cfg.pretrained_config
        in_channels = getattr(pc, "in_channels", 128)
        # encoder_hidden_states comes from T5 (caption_channels), then
        # gets projected by `caption_projection` to `inner_dim` inside the
        # transformer. The kwarg's *input* dim is `caption_channels`, not
        # `cross_attention_dim`.
        caption_channels = getattr(pc, "caption_channels", 4096)

        # `num_frames`/`height`/`width` are baked into the captured graph as
        # static literals (Guard failed: num_frames == ... at runtime if they
        # change). The diffusers `LTXPipeline` passes these explicitly per
        # call. Read the production values from the adapter module so the
        # parity runner (which patches `_EX_NUM_FRAMES`/`_EX_H`/`_EX_W`
        # before capture) can specialize at runtime shape.
        kwargs: dict[str, Any] = {
            "hidden_states": torch.randn(
                _EX_B, _EX_VIDEO_SEQ, in_channels, device=device, dtype=dtype
            ),
            "encoder_hidden_states": torch.randn(
                _EX_B, _EX_TEXT_SEQ, caption_channels, device=device, dtype=dtype
            ),
            "timestep": torch.full((_EX_B,), 500, device=device, dtype=torch.long),
            "encoder_attention_mask": torch.ones(
                _EX_B, _EX_TEXT_SEQ, device=device, dtype=torch.long
            ),
            "num_frames": _EX_NUM_FRAMES,
            "height": _EX_H,
            "width": _EX_W,
            "return_dict": False,
        }
        return (), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        # Same story as WanAdapter — LTX's forward has shape-dependent
        # int reshapes; Dim.AUTO lets export specialize without erroring.
        AUTO = torch.export.Dim.AUTO
        return {
            "hidden_states": {0: AUTO, 1: AUTO},
            "encoder_hidden_states": {0: AUTO, 1: AUTO},
            "timestep": {0: AUTO},
            "encoder_attention_mask": {0: AUTO, 1: AUTO},
            "num_frames": None,
            "height": None,
            "width": None,
            "return_dict": None,
        }

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        # LTX uses `qk_norm='rms_norm_across_heads'` — RMSNorm operates on
        # the full `inner_dim` (across all heads). Our `fused_dit_qk_norm_rope`
        # kernel only supports per-head norm (head_dim ≤ 128); disable the
        # fusion here. Drift impact is sub-perceptual (ULP noise) and only
        # matters for pixel-equality tests anyway.
        return RewritePolicy(
            attention_backend=cfg.attention.backend,
            fuse_qk_rope=False,
        )

    @property
    def uses_internal_seq_shard(self) -> bool:
        # LTX-Video's `hidden_states` is already token-flat `(B, S, C)`
        # — but `self.rope(hidden_states, num_frames, height, width, ...)`
        # internally builds a per-token coords tensor sized to the full S,
        # so the pipeline-boundary shard would have the rope module see a
        # smaller hidden_states but still try to build full coords (guard
        # fail). Use the in-model shard pattern: compute rope on the full
        # sequence first, slice both `hidden_states` and the rope outputs
        # by ulysses_rank, run blocks, all-gather before `norm_out`.
        return True

    def pre_capture_patch(self, model, visual_gen_mapping) -> None:
        ulysses_size = getattr(visual_gen_mapping, "ulysses_size", 1) if visual_gen_mapping else 1
        if ulysses_size <= 1:
            return
        ulysses_rank = getattr(visual_gen_mapping, "ulysses_rank", 0)

        from types import MethodType

        def _ltx_ulysses_forward(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            num_frames=None,
            height=None,
            width=None,
            rope_interpolation_scale=None,
            video_coords=None,
            attention_kwargs=None,
            return_dict: bool = False,
        ):
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            # Compute rope on the full sequence first (it depends on num_frames
            # / height / width before any sharding).
            image_rotary_emb = self.rope(
                hidden_states,
                num_frames,
                height,
                width,
                rope_interpolation_scale,
                video_coords,
            )

            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

            batch_size = hidden_states.size(0)
            hidden_states = self.proj_in(hidden_states)

            # --- Ulysses shard: slice hidden_states and rope along seq dim ---
            P = ulysses_size
            R = ulysses_rank
            S = hidden_states.shape[1]
            chunk = S // P
            hidden_states = hidden_states[:, R * chunk : (R + 1) * chunk, :].contiguous()
            if isinstance(image_rotary_emb, (tuple, list)):
                # LTX-Video rope returns (cos, sin); both shape (B, S, D) or (B, H, S, D).
                def _shard_rope_leaf(t):
                    if not isinstance(t, torch.Tensor) or t.dim() < 2:
                        return t
                    # seq dim heuristic: same as handwritten LTX-2 shard.
                    if t.dim() == 4 and t.shape[2] == S:
                        return t[:, :, R * chunk : (R + 1) * chunk].contiguous()
                    if t.dim() == 3 and t.shape[1] == S:
                        return t[:, R * chunk : (R + 1) * chunk].contiguous()
                    return t

                image_rotary_emb = tuple(_shard_rope_leaf(t) for t in image_rotary_emb)
            # --------------------------------------------------------------------

            temb, embedded_timestep = self.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.size(-1)
            )

            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                )

            # --- Gather hidden_states back to full S before output ---
            hidden_states = torch.ops.visgen_auto.all_gather_seq(hidden_states, 1, ulysses_size)
            # ----------------------------------------------------------

            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            output = self.proj_out(hidden_states)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        model.forward = MethodType(_ltx_ulysses_forward, model)
