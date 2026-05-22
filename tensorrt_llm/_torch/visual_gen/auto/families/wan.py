# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WAN family adapter for the auto path.

Targets Diffusers `WanTransformer3DModel` (Wan2.1 / 2.2). Major differences
from the image-DiT families:

- 5-D `hidden_states` `(B, C, T, H, W)` — video latents pre-patch. Internal
  `Conv3d(patch_size=(1,2,2))` flattens to a sequence inside the model.
- `encoder_hidden_states` is text-only (T2V); the I2V variant adds a
  separate `encoder_hidden_states_image` arg which we skip (None).
- `qk_norm='rms_norm_across_heads'` — full-mode RMSNorm. PR #13614's new
  RMSNorm with `allreduce_variance=True` will be relevant for TP here.
- Much larger: 40 layers, FFN dim 13824, 14B params per transformer.

This adapter is a *spike* — intended to confirm the auto path's
infrastructure (capture, sensitivity heuristic, dispatcher) doesn't have
image-DiT-specific assumptions baked in. End-to-end inference will require
matching the Diffusers `WanPipeline` shell (dual transformers, scheduler
specifics), which is follow-up work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig


# Minimum-viable example shapes for capture. Real inference will use much
# larger spatial dims (e.g., 480×832 or 720×1280) and 81–161 frames.
_EX_B = 1
_EX_C = 16  # WAN T2V default in_channels
_EX_T = 4  # post-patch frames; with patch_size[0]=1 this is raw frame count
_EX_H = 32  # post-patch_embedding height (so input H = 32 * patch_size[1] = 64)
_EX_W = 32
_EX_TEXT_LEN = 32
_EX_TEXT_DIM = 4096


class WanAdapter(VisGenFamilyAdapter):
    family = "Wan"
    diffusers_transformer_cls = "WanTransformer3DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        pc = cfg.pretrained_config
        C = getattr(pc, "in_channels", _EX_C)
        text_dim = getattr(pc, "text_dim", _EX_TEXT_DIM)
        patch = getattr(pc, "patch_size", (1, 2, 2))

        # Input video latent is `(B, C, T, H, W)` *before* patch_embedding.
        # The model's `Conv3d(patch_size=patch)` produces post-patch
        # `(B, inner_dim, T/p_t, H/p_h, W/p_w)`, then flatten+transpose to seq.
        H_in = _EX_H * patch[1]
        W_in = _EX_W * patch[2]
        T_in = _EX_T * patch[0]

        # `encoder_hidden_states_image` is the I2V branch; Diffusers'
        # T2V `WanPipeline.__call__` doesn't pass it. Including it in the
        # captured-graph signature would force the pipeline to pass it too
        # (torch.export's kwarg-set is strict at runtime). Drop it for the
        # T2V capture; I2V is a follow-up adapter mode.
        kwargs: dict[str, Any] = {
            "hidden_states": torch.randn(_EX_B, C, T_in, H_in, W_in, device=device, dtype=dtype),
            "timestep": torch.full((_EX_B,), 500, device=device, dtype=torch.long),
            "encoder_hidden_states": torch.randn(
                _EX_B, _EX_TEXT_LEN, text_dim, device=device, dtype=dtype
            ),
            "return_dict": False,
            "attention_kwargs": None,
        }
        return (), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        # WAN's forward has shape-dependent control flow (frame counts, patch
        # math via `hidden_states.shape[...]` used as Python ints in reshapes).
        # Trying to mark these as Dim.DYNAMIC trips torch.export's specialization
        # detector. We fall back to AUTO for now — produces a static capture per
        # call, sufficient for the spike. Re-captures across resolution changes
        # land as follow-up (sharding + dynamic-shape recovery is per-family work).
        AUTO = torch.export.Dim.AUTO
        return {
            "hidden_states": {0: AUTO, 2: AUTO, 3: AUTO, 4: AUTO},
            "timestep": {0: AUTO},
            "encoder_hidden_states": {0: AUTO, 1: AUTO},
            "return_dict": None,
            "attention_kwargs": None,
        }

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        # WAN's `qk_norm` is `rms_norm_across_heads`: RMSNorm operates on the
        # full inner_dim (5120 = 40 heads × 128 head_dim) BEFORE the unflatten
        # to per-head shape — verified by walking the captured graph
        # (`rms_norm.args[1] == [5120]`, then `aten.unflatten.int(..., 2, [40,
        # -1])` downstream). The `fused_dit_qk_norm_rope` kernel only handles
        # per-head RMSNorm (head_dim ≤ 128), so the fusion isn't applicable —
        # disable explicitly to make the policy honest. Same constraint as
        # LTX-Video / LTX-2.
        return RewritePolicy(
            attention_backend=cfg.attention.backend,
            fuse_qk_rope=False,
        )

    @property
    def uses_internal_seq_shard(self) -> bool:
        # WAN's 5-D `(B, C, T, H, W)` input doesn't admit a clean axis-aligned
        # half (T=21 odd; H_p × W_p partial slices break the patch+flatten
        # row-major order). Handwritten visual_gen WAN shards the flat sequence
        # *inside* the model after `patch_embedding.flatten(2).transpose(1,2)`
        # and `all_gather`s before `norm_out`. We mirror that by monkey-patching
        # the forward in `pre_capture_patch`.
        return True

    def pre_capture_patch(self, model, visual_gen_mapping) -> None:
        ulysses_size = getattr(visual_gen_mapping, "ulysses_size", 1) if visual_gen_mapping else 1
        if ulysses_size <= 1:
            return  # single-GPU — no patch needed
        ulysses_rank = getattr(visual_gen_mapping, "ulysses_rank", 0)

        # Capture the originals so the wrapped forward can call into them.
        # Reuse the model's internals; we only inject slice + all_gather around
        # the existing block loop.
        from types import MethodType

        # `_wan_ulysses_forward` is bound to the model below; closes over
        # `ulysses_size` and `ulysses_rank` as Python constants so torch.export
        # bakes them into the captured graph.
        def _wan_ulysses_forward(
            self,
            hidden_states,
            timestep,
            encoder_hidden_states,
            return_dict: bool = False,
            attention_kwargs=None,
        ):
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            batch_size, _C, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.config.patch_size
            post_patch_num_frames = num_frames // p_t
            post_patch_height = height // p_h
            post_patch_width = width // p_w

            rotary_emb = self.rope(hidden_states)
            hidden_states = self.patch_embedding(hidden_states)
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
            # hidden_states: (B, S, inner_dim) — full sequence on every rank.

            # --- Ulysses shard along the flat sequence (handwritten WAN pattern) ---
            P = ulysses_size
            R = ulysses_rank
            S = hidden_states.shape[1]
            # S must be divisible by P; choose resolutions that satisfy this
            # (for p06 @ 81×720×1280 with patch (1,2,2) → S = 21×45×80 = 75600,
            # divisible by 2/4/6/...).
            chunk = S // P
            hidden_states = hidden_states[:, R * chunk : (R + 1) * chunk, :].contiguous()

            if isinstance(rotary_emb, (tuple, list)):
                rotary_emb = tuple(
                    (
                        t[:, R * chunk : (R + 1) * chunk].contiguous()
                        if isinstance(t, torch.Tensor)
                        else t
                    )
                    for t in rotary_emb
                )
            elif isinstance(rotary_emb, torch.Tensor):
                rotary_emb = rotary_emb[:, R * chunk : (R + 1) * chunk].contiguous()
            # ----------------------------------------------------------------

            temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
                self.condition_embedder(
                    timestep, encoder_hidden_states, None, timestep_seq_len=None
                )
            )
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

            for block in self.blocks:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )

            # --- Gather sequence back to full S before output projection ----
            hidden_states = torch.ops.visgen_auto.all_gather_seq(hidden_states, 1, ulysses_size)
            # ----------------------------------------------------------------

            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(
                2, dim=1
            )
            shift = shift.to(hidden_states.device)
            scale = scale.to(hidden_states.device)
            hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
                hidden_states
            )
            hidden_states = self.proj_out(hidden_states)

            hidden_states = hidden_states.reshape(
                batch_size,
                post_patch_num_frames,
                post_patch_height,
                post_patch_width,
                p_t,
                p_h,
                p_w,
                -1,
            )
            hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
            output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

        model.forward = MethodType(_wan_ulysses_forward, model)
