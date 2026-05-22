# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LTX-2 family adapter for the auto path.

Targets Diffusers `LTX2VideoTransformer3DModel` (audiovisual LTX-2.x).
Distinct from `LTXAdapter` which handles LTX-Video (no audio).

The forward sig is significantly bigger than LTX-Video — joint video+audio
streams, per-modality text embeddings, per-modality timesteps, and
pre-computed RoPE coords.  We match the authoritative runtime call in
`diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline.__call__` (see line
1224).

Notes:
- `timestep` at runtime is shape `(B,)` — the model broadcasts internally.
  The class docstring says `(B, S_video)` but the pipeline always passes
  `(B,)`.  We capture with `(B,)`.
- `sigma = timestep` in the pipeline call (LTX-2.3 cross-attn modulation).
- `video_coords` / `audio_coords` are pre-computed on the pipeline side
  from `num_frames`/`height`/`width`/`fps` — we provide them explicitly
  so export doesn't have to symbolically resolve int math inside
  `LTX2AudioVideoRotaryPosEmbed`.
- `qk_norm="rms_norm_across_heads"` (same as LTX-Video) → disable
  `fuse_qk_rope` (the kernel only supports per-head RMSNorm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..adapter import VisGenFamilyAdapter
from ..policy import RewritePolicy

if TYPE_CHECKING:
    from ...config import DiffusionModelConfig


_EX_B = 2  # match CFG-doubled runtime batch (positive + negative prompts)
_EX_VIDEO_SEQ = 512
_EX_AUDIO_SEQ = 128
_EX_TEXT_SEQ = 64
_EX_NUM_FRAMES = 8
_EX_LATENT_H = 8
_EX_LATENT_W = 8
_EX_AUDIO_NUM_FRAMES = 8
_EX_FPS = 24.0
_EX_TIMESTEP = 500


def _arch_dim(cfg: "DiffusionModelConfig", attr: str, default: int) -> int:
    pc = getattr(cfg, "pretrained_config", None)
    return int(getattr(pc, attr, default)) if pc is not None else default


class LTX2Adapter(VisGenFamilyAdapter):
    family = "LTX-2"
    diffusers_transformer_cls = "LTX2VideoTransformer3DModel"

    def example_inputs(
        self,
        cfg: "DiffusionModelConfig",
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> tuple[tuple, dict[str, Any]]:
        in_channels = _arch_dim(cfg, "in_channels", 128)
        audio_in_channels = _arch_dim(cfg, "audio_in_channels", 128)
        caption_channels = _arch_dim(cfg, "caption_channels", 3840)

        B = _EX_B
        S_video = _EX_VIDEO_SEQ
        S_audio = _EX_AUDIO_SEQ
        S_text = _EX_TEXT_SEQ

        kwargs: dict[str, Any] = {
            # Joint video+audio streams (post-patchify, flat sequences).
            "hidden_states": torch.randn(B, S_video, in_channels, device=device, dtype=dtype),
            "audio_hidden_states": torch.randn(
                B, S_audio, audio_in_channels, device=device, dtype=dtype
            ),
            # Per-modality text embeddings (Gemma3 → caption_channels).
            "encoder_hidden_states": torch.randn(
                B, S_text, caption_channels, device=device, dtype=dtype
            ),
            "audio_encoder_hidden_states": torch.randn(
                B, S_text, caption_channels, device=device, dtype=dtype
            ),
            # Timestep is (B,) at runtime — pipeline does `t.expand(B)`.
            # Pass Long to match scheduler outputs.
            "timestep": torch.full((B,), _EX_TIMESTEP, device=device, dtype=torch.long),
            # sigma == timestep at runtime (LTX-2.3 cross-attn modulation).
            "sigma": torch.full((B,), float(_EX_TIMESTEP), device=device, dtype=dtype),
            "encoder_attention_mask": torch.ones(B, S_text, device=device, dtype=torch.long),
            "audio_encoder_attention_mask": torch.ones(B, S_text, device=device, dtype=torch.long),
            # Pre-computed RoPE coords — supplying these bypasses internal int
            # math from num_frames/height/width (which would specialize hard
            # during export). The int kwargs num_frames/height/width/fps/
            # audio_num_frames are deliberately *omitted* — the transformer
            # only consults them when video_coords/audio_coords is None, and
            # leaving them out of the captured signature means the wrapper's
            # kwargs filter drops them at runtime instead of forcing a
            # specialization-failed guard.
            "video_coords": torch.zeros(B, 3, S_video, 2, device=device, dtype=dtype),
            "audio_coords": torch.zeros(B, 1, S_audio, 2, device=device, dtype=dtype),
            # Bool/None flags.
            "isolate_modalities": False,
            "spatio_temporal_guidance_blocks": None,
            "perturbation_mask": None,
            "use_cross_timestep": False,
            "attention_kwargs": None,
            "return_dict": False,
        }
        return (), kwargs

    def dynamic_shapes(self, cfg: "DiffusionModelConfig") -> dict[str, Any]:
        # LTX family has shape-dependent int reshapes inside RoPE / block
        # forwards. Dim.AUTO lets export specialize without erroring.
        AUTO = torch.export.Dim.AUTO
        return {
            "hidden_states": {0: AUTO, 1: AUTO},
            "audio_hidden_states": {0: AUTO, 1: AUTO},
            "encoder_hidden_states": {0: AUTO, 1: AUTO},
            "audio_encoder_hidden_states": {0: AUTO, 1: AUTO},
            "timestep": {0: AUTO},
            "sigma": {0: AUTO},
            "encoder_attention_mask": {0: AUTO, 1: AUTO},
            "audio_encoder_attention_mask": {0: AUTO, 1: AUTO},
            "video_coords": {0: AUTO, 2: AUTO},
            "audio_coords": {0: AUTO, 2: AUTO},
            # Static / not tensors (num_frames/height/width/fps/audio_num_frames
            # are filtered out of the runtime call by the wrapper's
            # _expected_kwargs check — they are not in example_inputs).
            "isolate_modalities": None,
            "spatio_temporal_guidance_blocks": None,
            "perturbation_mask": None,
            "use_cross_timestep": None,
            "attention_kwargs": None,
            "return_dict": None,
        }

    def rewrite_policy(self, cfg: "DiffusionModelConfig") -> RewritePolicy:
        # LTX-2 uses `qk_norm="rms_norm_across_heads"` — same constraint as
        # LTX-Video. The fused DiT QK-norm+RoPE kernel only handles per-head
        # RMSNorm (head_dim ≤ 128); disable it here.
        return RewritePolicy(
            attention_backend=cfg.attention.backend,
            fuse_qk_rope=False,
        )

    # No family-specific excludes needed: `auto/sensitivity.py`'s
    # "narrow-dim embedder, either direction" heuristic catches both the
    # in-side (proj_in, time_embed, ...) and out-side (proj_out → 128 VAE
    # latent channels, audio_proj_out → same) for LTX-2 automatically.

    @property
    def uses_internal_seq_shard(self) -> bool:
        # Joint audio+video transformer: video stream is sharded along its
        # sequence dim; audio and text are kept full on every rank. The
        # shard/gather hooks have to be injected *inside* `model.forward`
        # because the cross-modal v2a attention needs an explicit all-gather
        # of the video K/V (audio Q is full → must see all video tokens).
        return True

    def pre_capture_patch(self, model, visual_gen_mapping) -> None:
        ulysses_size = getattr(visual_gen_mapping, "ulysses_size", 1) if visual_gen_mapping else 1
        if ulysses_size <= 1:
            return
        ulysses_rank = getattr(visual_gen_mapping, "ulysses_rank", 0)

        from types import MethodType

        import torch
        from torch import nn

        # --- v2a wrapper -----------------------------------------------------
        # `block.video_to_audio_attn` is the only attention site where audio Q
        # (full) reads video K/V (sharded). Wrap it to all-gather both the
        # encoder_hidden_states (video stream) and the key_rotary_emb (video
        # rope for the K side) before delegating. Video rope is split-mode →
        # shape (B, H, S, D) with seq at dim 2; the wrapper handles either
        # 4-D split or 3-D interleaved by probing the tensor rank.
        class _V2ACrossAttnWrapper(nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def forward(
                self,
                hidden_states,  # audio Q, full
                encoder_hidden_states=None,  # video K/V, sharded
                query_rotary_emb=None,
                key_rotary_emb=None,
                attention_mask=None,
                **kwargs,
            ):
                if encoder_hidden_states is not None:
                    encoder_hidden_states = torch.ops.visgen_auto.all_gather_seq(
                        encoder_hidden_states, 1, ulysses_size
                    )
                if key_rotary_emb is not None:
                    cos, sin = key_rotary_emb
                    # split rope (B, H, S, D) → seq at dim 2;
                    # interleaved rope (B, S, D) → seq at dim 1.
                    seq_dim = 2 if cos.dim() == 4 else 1
                    cos = torch.ops.visgen_auto.all_gather_seq(cos, seq_dim, ulysses_size)
                    sin = torch.ops.visgen_auto.all_gather_seq(sin, seq_dim, ulysses_size)
                    key_rotary_emb = (cos, sin)
                return self.original(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    query_rotary_emb=query_rotary_emb,
                    key_rotary_emb=key_rotary_emb,
                    attention_mask=attention_mask,
                    **kwargs,
                )

        for block in model.transformer_blocks:
            block.video_to_audio_attn = _V2ACrossAttnWrapper(block.video_to_audio_attn)

        # --- monkey-patched forward -----------------------------------------
        # Copy of the diffusers LTX2VideoTransformer3DModel.forward with two
        # surgical edits:
        #   (a) after `proj_in(hidden_states)`, slice the video stream and the
        #       video rope tensors along their seq dim by ulysses_rank;
        #   (b) before `norm_out(hidden_states)`, `all_gather_seq` the video
        #       stream so the final scale/shift/proj_out runs on full S_video.
        # Audio path is untouched; encoder_hidden_states (text) kept full.
        P = ulysses_size
        R = ulysses_rank

        def _ltx2_ulysses_forward(
            self,
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            audio_encoder_hidden_states,
            timestep,
            audio_timestep=None,
            sigma=None,
            audio_sigma=None,
            encoder_attention_mask=None,
            audio_encoder_attention_mask=None,
            video_coords=None,
            audio_coords=None,
            isolate_modalities: bool = False,
            spatio_temporal_guidance_blocks=None,
            perturbation_mask=None,
            use_cross_timestep: bool = False,
            attention_kwargs=None,
            return_dict: bool = False,
        ):
            from diffusers.models.transformers.transformer_ltx2 import AudioVisualModelOutput

            audio_timestep = audio_timestep if audio_timestep is not None else timestep
            audio_sigma = audio_sigma if audio_sigma is not None else sigma

            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (
                    1 - encoder_attention_mask.to(hidden_states.dtype)
                ) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
                audio_encoder_attention_mask = (
                    1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)
                ) * -10000.0
                audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

            batch_size = hidden_states.size(0)

            # 1. RoPE — pipeline already supplied coords; our adapter declines
            # `num_frames`/`height`/`width`/`fps`/`audio_num_frames` so the
            # `if video_coords is None` recompute path is unreachable here.
            video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
            audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)
            video_cross_attn_rotary_emb = self.cross_attn_rope(
                video_coords[:, 0:1, :], device=hidden_states.device
            )
            audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
                audio_coords[:, 0:1, :], device=audio_hidden_states.device
            )

            # 2. Patchify
            hidden_states = self.proj_in(hidden_states)
            audio_hidden_states = self.audio_proj_in(audio_hidden_states)

            # --- Ulysses shard: slice video stream + video rope along seq ----
            S = hidden_states.shape[1]
            chunk = S // P

            def _shard_dim(t, dim):
                # only shard if dim has the expected video seq length
                if t is None or t.shape[dim] != S:
                    return t
                idx = [slice(None)] * t.dim()
                idx[dim] = slice(R * chunk, (R + 1) * chunk)
                return t[tuple(idx)].contiguous()

            def _shard_rope(rope):
                if rope is None:
                    return None
                cos, sin = rope
                # split rope: (B, H, S, D) → seq at dim 2
                # interleaved rope: (B, S, D) → seq at dim 1
                dim = 2 if cos.dim() == 4 else 1
                return (_shard_dim(cos, dim), _shard_dim(sin, dim))

            hidden_states = hidden_states[:, R * chunk : (R + 1) * chunk, :].contiguous()
            video_rotary_emb = _shard_rope(video_rotary_emb)
            video_cross_attn_rotary_emb = _shard_rope(video_cross_attn_rotary_emb)
            # ------------------------------------------------------------------

            # 3. Timestep embeddings — temb / embedded_timestep are typically
            # (B, 1, D) and broadcast against (B, S, D), so no shard needed.
            temb, embedded_timestep = self.time_embed(
                timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(batch_size, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))
            temb_audio, audio_embedded_timestep = self.audio_time_embed(
                audio_timestep.flatten(),
                batch_size=batch_size,
                hidden_dtype=audio_hidden_states.dtype,
            )
            temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
            audio_embedded_timestep = audio_embedded_timestep.view(
                batch_size, -1, audio_embedded_timestep.size(-1)
            )

            if self.prompt_modulation:
                temb_prompt, _ = self.prompt_adaln(
                    sigma.flatten(),
                    batch_size=batch_size,
                    hidden_dtype=hidden_states.dtype,
                )
                temb_prompt_audio, _ = self.audio_prompt_adaln(
                    audio_sigma.flatten(),
                    batch_size=batch_size,
                    hidden_dtype=audio_hidden_states.dtype,
                )
                temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
                temb_prompt_audio = temb_prompt_audio.view(
                    batch_size, -1, temb_prompt_audio.size(-1)
                )
            else:
                temb_prompt = temb_prompt_audio = None

            # 3.2. Cross attn modulation params
            from diffusers.models.transformers.transformer_ltx2 import (
                LTX2AdaLayerNormSingle,  # noqa: F401
            )

            ts_mult = (
                self.config.cross_attn_timestep_scale_multiplier
                / self.config.timestep_scale_multiplier
            )
            video_ca_ts = audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
            v_ca_ss, _ = self.av_cross_attn_video_scale_shift(
                video_ca_ts, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            v_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
                video_ca_ts * ts_mult, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
            v_ca_ss = v_ca_ss.view(batch_size, -1, v_ca_ss.shape[-1])
            v_a2v_gate = v_a2v_gate.view(batch_size, -1, v_a2v_gate.shape[-1])
            audio_ca_ts = sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
            a_ca_ss, _ = self.av_cross_attn_audio_scale_shift(
                audio_ca_ts, batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype
            )
            a_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
                audio_ca_ts * ts_mult, batch_size=batch_size, hidden_dtype=audio_hidden_states.dtype
            )
            a_ca_ss = a_ca_ss.view(batch_size, -1, a_ca_ss.shape[-1])
            a_v2a_gate = a_v2a_gate.view(batch_size, -1, a_v2a_gate.shape[-1])

            # 4. Prompt embeddings — kept full on every rank
            if self.config.use_prompt_embeddings:
                encoder_hidden_states = self.caption_projection(encoder_hidden_states)
                encoder_hidden_states = encoder_hidden_states.view(
                    batch_size, -1, hidden_states.size(-1)
                )
                audio_encoder_hidden_states = self.audio_caption_projection(
                    audio_encoder_hidden_states
                )
                audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                    batch_size, -1, audio_hidden_states.size(-1)
                )

            # 5. Blocks
            stg = set(spatio_temporal_guidance_blocks or [])
            all_pert = False
            if perturbation_mask is not None and perturbation_mask.ndim == 1:
                perturbation_mask = perturbation_mask[:, None, None]
            if perturbation_mask is not None:
                all_pert = torch.all(perturbation_mask == 0).item()

            for i, block in enumerate(self.transformer_blocks):
                bpm = perturbation_mask if i in stg else None
                bap = all_pert if i in stg else False
                hidden_states, audio_hidden_states = block(
                    hidden_states=hidden_states,
                    audio_hidden_states=audio_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    temb=temb,
                    temb_audio=temb_audio,
                    temb_ca_scale_shift=v_ca_ss,
                    temb_ca_audio_scale_shift=a_ca_ss,
                    temb_ca_gate=v_a2v_gate,
                    temb_ca_audio_gate=a_v2a_gate,
                    temb_prompt=temb_prompt,
                    temb_prompt_audio=temb_prompt_audio,
                    video_rotary_emb=video_rotary_emb,
                    audio_rotary_emb=audio_rotary_emb,
                    ca_video_rotary_emb=video_cross_attn_rotary_emb,
                    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                    encoder_attention_mask=encoder_attention_mask,
                    audio_encoder_attention_mask=audio_encoder_attention_mask,
                    self_attention_mask=None,
                    audio_self_attention_mask=None,
                    a2v_cross_attention_mask=None,
                    v2a_cross_attention_mask=None,
                    use_a2v_cross_attention=not isolate_modalities,
                    use_v2a_cross_attention=not isolate_modalities,
                    perturbation_mask=bpm,
                    all_perturbed=bap,
                )

            # --- Ulysses gather: bring video stream back to full S ----------
            hidden_states = torch.ops.visgen_auto.all_gather_seq(hidden_states, 1, ulysses_size)
            # -----------------------------------------------------------------

            # 6. Output
            scale_shift = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            output = self.proj_out(hidden_states)

            audio_scale_shift = (
                self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
            )
            audio_shift, audio_scale = audio_scale_shift[:, :, 0], audio_scale_shift[:, :, 1]
            audio_hidden_states = self.audio_norm_out(audio_hidden_states)
            audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
            audio_output = self.audio_proj_out(audio_hidden_states)

            if not return_dict:
                return (output, audio_output)
            return AudioVisualModelOutput(sample=output, audio_sample=audio_output)

        model.forward = MethodType(_ltx2_ulysses_forward, model)
