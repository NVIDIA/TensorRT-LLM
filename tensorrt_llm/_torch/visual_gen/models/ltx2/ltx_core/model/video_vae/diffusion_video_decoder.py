"""Diffusion (NATTEN) video VAE decoder."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import Iterator, List, Literal, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.devices import upload_from_host
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.disposable import Disposable
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae import decode_workspace, diffusion_tiling
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import (
    DecodeKeyframes,
    KeyframePlanes,
    KeyframeStream,
    keyframe_clip_times,
    planes_for_tile,
    remaining_time_strides,
    upsample_keyframe_planes,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.ops import (
    PerChannelStatistics,
    apply_rms_norm_uint32_safe,
    patchify_channels_last,
    unpatchify,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer import (
    AdaLNZero,
    ChannelLinear,
    CombinedDiffusionNABlock,
    LinearPixelShuffleUpsample,
    NABlock,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.rope_math import abs_rope_table_cache
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.stage5_tail import stage5_tail, stage5_tail_supported
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.video_vae import VideoDecoder, iter_decoded_single_frames
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling import (
    Tile,
    TilingConfig,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    scale_by_masks_1d,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.utils import to_velocity

logger: logging.Logger = logging.getLogger(__name__)

# Production CausalVideoAutoencoderL decoder layout: channel width per stage.
# Mirrors the (non-diffusion) NA decoder's stage spec.
_L_STAGE_CHANNELS: Tuple[int, ...] = (1024, 512, 256, 256, 128)
_L_STAGE_DEPTHS: Tuple[int, ...] = (4, 6, 4, 2, 2)
# (stride, out_channels_reduction_factor) per upsample, in stage order.
_L_UPSAMPLES: Tuple[Tuple[Tuple[int, int, int], int], ...] = (
    ((1, 2, 2), 2),  # compress_space x2
    ((2, 1, 1), 2),  # compress_time x2
    ((2, 2, 2), 1),  # compress_all x1 (channel-preserving)
    ((2, 2, 2), 2),  # compress_all x2
)
# Per-stage 3D neighborhood (K_t, K_h, K_w).
_L_STAGE_KERNELS: Tuple[Tuple[int, int, int], ...] = (
    (3, 7, 7),
    (3, 7, 7),
    (3, 5, 5),
    (3, 5, 5),
    (3, 3, 3),
)

# Stage-5 (diffusion stage) defaults: wider kernel + more blocks than the
# deterministic stages, since it carries the entire per-step diffusion compute.
_DIFF_STAGE5_KERNEL_DEFAULT: Tuple[int, int, int] = (3, 7, 7)
_DIFF_STAGE5_DEPTH_DEFAULT: int = 8
_DIFF_STAGE_DEPTHS_DEFAULT: Tuple[int, ...] = (*_L_STAGE_DEPTHS[:-1], _DIFF_STAGE5_DEPTH_DEFAULT)

# Rows of the stage-5 volume normalised per in-place ``norm_out`` chunk (unfused tail).
_NORM_OUT_CHUNK_ROWS: int = 1 << 20
# Pixel bytes per ``conv_out`` / unpatchify chunk (unfused tail) and per tile-blend chunk.
_TAIL_CHUNK_BYTES: int = 1 << 29


def _release_cached_blocks(device: torch.device) -> None:
    """Return the caching allocator's free segments, so one decode phase's churn is not reserved on top of
    the next one's."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _blend_tile_into(
    buffer: torch.Tensor,
    weights: torch.Tensor | None,
    local_coords: tuple,
    pixel_tile: torch.Tensor,
    masks: tuple,
) -> None:
    """The masked accumulate of ``pixel_tile`` into ``buffer`` (and ``weights``), one temporal chunk at a time.
    Per-element masks and accumulate, so chunking reproduces the whole-tile blend bit for bit.
    """
    frames = int(pixel_tile.shape[2])
    t_lo, t_hi, _ = local_coords[2].indices(buffer.shape[2])
    if t_hi - t_lo != frames:
        raise RuntimeError(f"tile covers {frames} frames but its buffer slice spans {t_hi - t_lo}")
    frame_bytes = pixel_tile[:, :, :1].numel() * 4
    chunk_t = max(1, min(frames, _TAIL_CHUNK_BYTES // max(frame_bytes, 1)))
    for f0 in range(0, frames, chunk_t):
        f1 = min(frames, f0 + chunk_t)
        coords = (local_coords[0], local_coords[1], slice(t_lo + f0, t_lo + f1), local_coords[3], local_coords[4])
        chunk_masks = (
            masks[0],
            masks[1],
            masks[2][f0:f1] if masks[2].numel() > 1 else masks[2],
            masks[3],
            masks[4],
        )
        chunk = pixel_tile[:, :, f0:f1].to(buffer.dtype)
        buffer[coords] += scale_by_masks_1d(chunk, chunk_masks)
        if weights is not None:
            strength = torch.ones(chunk.shape, device=buffer.device, dtype=buffer.dtype)
            weights[coords] += scale_by_masks_1d(strength, chunk_masks)


class DiffusionVideoDecoder(nn.Module, Disposable, VideoDecoder):
    """Diffusion-based video VAE decoder (Neighborhood-Attention backbone).
    Minimal port of the reference ``NADiffusionDecoder``.
    Stages 1-4 deterministically upsample the latent into a context volume
    (same NA-upsample path as the non-diffusion NA decoder). Stage 5 runs
    ``DiffusionNABlock``s that denoise the patchified noised pixels ``x_t``,
    guided by that context via AdaLN-Zero scale/shift (ungated residuals;
    legacy static gates are folded into Linear weights at load time).
    Last-frame NATTEN window-shift is mitigated by temporarily replicating the
    last latent frame ``(stage1_K_t // 2) * 2`` times through stages 1-4, then
    cropping that appendix from context before stage 5 - but only down to
    ``max(original_context_T, stage5_kernel[0])`` so undersized clips (e.g. a
    single latent frame) still satisfy NATTEN's kernel floor. Latents / tiles
    below ``stage_min_tile_sizes`` are edge-padded first via ``diffusion_tiling``;
    leftover pad is cropped from the final pixels.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        rope_dim_split: Tuple[int, int, int] | None = None,
        stage_channels: Tuple[int, ...] = _L_STAGE_CHANNELS,
        stage_depths: Tuple[int, ...] = _DIFF_STAGE_DEPTHS_DEFAULT,
        stage_kernels: Tuple[Tuple[int, int, int], ...] = _L_STAGE_KERNELS,
        upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...] = _L_UPSAMPLES,
        stage5_kernel: Tuple[int, int, int] = _DIFF_STAGE5_KERNEL_DEFAULT,
        stage5_channels: int | None = None,
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 2,
        timestep_scale_multiplier: float = 1.0,
        model_output_type: Literal["v", "x0"] = "v",
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(stage_depths) == len(stage_kernels)
        assert len(upsamples) == len(stage_channels) - 1
        for c in stage_channels:
            assert c % head_dim == 0, f"stage_channels {stage_channels} must each be a multiple of head_dim={head_dim}"

        self.patch_size = patch_size
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps, device="cpu"),
            persistent=False,
        )
        self.out_channels = out_channels
        self.stage_channels = stage_channels
        self.stage_depths = stage_depths
        self.base_channels = stage_channels[-1]
        self.causal = False
        self.timestep_conditioning = True
        self.video_downscale_factors = SpatioTemporalScaleFactors.default()
        self.stage5_kernel: Tuple[int, int, int] = tuple(stage5_kernel)  # type: ignore[assignment]
        # NATTEN last-frame border workaround: replicate last latent frame
        # ``(K_t // 2) * 2`` times through stages 1-4, then crop the appendix
        # off context before stage 5 down to at least ``stage5_kernel[0]``.
        self._natten_trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        # Encoder output is per-channel normalized; undo before conv_in (same as ConvVideoDecoder).
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        self.conv_in = ChannelLinear(in_channels, stage_channels[0], bias=True)
        # Keyframe-stream tag, added to un-normalized keyframe latents before the shared
        # ``conv_in`` and nowhere else. It is the only keyframe-specific weight in the
        # whole feature. Checkpoints predating the keyframe training have no such key, so
        # ``video_decoder_sd_ops_for_checkpoint`` synthesizes zeros -- a missing key would
        # otherwise leave the parameter on the meta device under ``strict=False`` load.
        self.type_emb = nn.Parameter(torch.zeros(in_channels))

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        n_det_stages = len(stage_channels) - 1
        for stage_i in range(n_det_stages):
            c = stage_channels[stage_i]
            depth = stage_depths[stage_i]
            kernel = stage_kernels[stage_i]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        NABlock(dim=c, kernel_size=kernel, head_dim=head_dim, rope_dim_split=rope_dim_split)
                        for _ in range(depth)
                    ]
                )
            )
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(
                LinearPixelShuffleUpsample(in_channels=c, stride=stride, out_channels_reduction_factor=reduction)
            )

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)

        c_ctx = stage_channels[-1]
        self.context_channels = c_ctx
        c5 = stage5_channels if stage5_channels is not None else c_ctx
        d5 = stage_depths[-1]
        assert c5 % head_dim == 0, f"stage5_channels {c5} must be a multiple of head_dim={head_dim}"
        noised_pixel_channels = out_channels * (patch_size**2)

        # Latent-grid floor so stages 1-3 (full volume) never undershoot NA.
        self.stage_min_tile_sizes: Tuple[int, int, int] = diffusion_tiling.all_stages_min_tile_size(
            stage_kernels, upsamples, stage5_kernel
        )
        # Stage-4-input tile floor / overlap halos (only stages 4-5 are tiled).
        up3_stride = upsamples[3][0]
        self.tile_min_sizes: Tuple[int, int, int] = diffusion_tiling.compute_tile_min_size(
            stage_kernels[3], stage5_kernel, up3_stride
        )
        self.tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = diffusion_tiling.compute_tile_halos(
            stage_kernels[3],
            stage_depths[3],
            stage5_kernel,
            stage_depths[-1],
            up3_stride,
        )
        self.conv_in_x_t = ChannelLinear(noised_pixel_channels, c5, bias=True)

        # Shared AdaLN-Zero (7-chunk for shape compat; gate slots unused in block).
        self.shared_adaln = AdaLNZero(dim=c5, t_emb_dim=t_emb_dim)

        self.diff_blocks = nn.ModuleList(
            [
                CombinedDiffusionNABlock(
                    dim=c5,
                    kernel_size=stage5_kernel,
                    context_channels=c_ctx,
                    head_dim=head_dim,
                    rope_dim_split=rope_dim_split,
                )
                for _ in range(d5)
            ]
        )

        self.norm_out = nn.RMSNorm(c5, eps=1e-6)
        self.conv_out = ChannelLinear(c5, noised_pixel_channels, bias=True)

        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        # Set True by ``compile_diffusion_decoder`` so decode marks T/H/W dynamic.
        self.mark_dynamic_shapes = False
        # When True, skip stage-4 upsample and inject via deferred sequential upsample+proj.
        # Default False = combined pathway (``CombinedDiffusionNABlock``). Chunked DiffVAE
        # modes flip this via ``apply_diffvae_config``.
        self.deferred_stage4_upsample = False
        # Remaining temporal upsampling per stage input, plus 1 for stage 5: the divisor in
        # ``keyframe_stage_times``. (8, 8, 4, 2, 1) for the production ladder.
        self._keyframe_time_strides: Tuple[int, ...] = remaining_time_strides(self.upsamples)

    def _run_det_stage(self, x: torch.Tensor, stage_i: int, drop_leading_frame: bool) -> torch.Tensor:
        """One deterministic stage: NA blocks + upsample."""
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
        for block in self.det_stages[stage_i]:
            x = block(x)
        return self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)

    def forward_stages_1_to_3(
        self,
        z_noisy: torch.Tensor,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        """Stages 1-3 on a full (or already ghost-padded) latent → stage-4 input feature.
        Output is channels-last ``(B, T, H, W, C)`` at stage-4 input resolution.
        Callers that want NATTEN trailing ghosting should pad the latent first via
        ``diffusion_tiling.pad_trailing_latent_for_natten_border``.
        """
        z_noisy = self.per_channel_statistics.un_normalize(z_noisy)
        x = z_noisy.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i in range(3):
            x = self._run_det_stage(x, stage_i, drop_leading_frame)
        return x

    def _keyframe_stream_from_latents(
        self,
        keyframes: DecodeKeyframes,
        *,
        valid: torch.Tensor | None = None,
    ) -> KeyframeStream:
        """Keyframe latents to a stage-1-input stream: un-normalize, tag, ``conv_in``, mask.
        Stages 1-3 always run on the whole volume, so times stay in the global stage-1
        origin. Tile-local rebasing happens later, at stage 4.
        Unlike upstream we un-normalize first, because our ``conv_in`` consumes
        un-normalized latents (``forward_stages_1_to_3`` does the same for video) while
        upstream un-normalizes above the decoder. ``type_emb`` still lands in exactly the
        same place: on the latents, immediately before the shared ``conv_in``.
        """
        latents = self.per_channel_statistics.un_normalize(keyframes.latents)
        x = latents.permute(0, 2, 3, 4, 1)
        x = x + self.type_emb.view(1, 1, 1, 1, -1)
        x = self.conv_in(x)
        planes = x.shape[1]
        if valid is None:
            valid = torch.ones(planes, dtype=torch.bool)
        times = keyframe_clip_times(
            keyframes.pixel_frame_indices,
            self._keyframe_time_strides[0],
            keyframes.clip_start_frame,
        )
        return KeyframeStream.from_host(x, times, valid).masked()

    def _run_det_stage_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
        stage_i: int,
        drop_leading_frame: bool,
        pixel_frame_indices: torch.Tensor,
        next_time_origin: float,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """One deterministic stage over both streams: joint NA blocks + upsample.
        The keyframe upsample is spatial-only and always drops its leading frame; the video
        stream's ``drop_leading_frame`` is a tiling property and must not leak into it.
        Times are rebuilt from the *next* stage's remaining stride after upsampling.
        ``next_time_origin`` is in the **next** stage's temporal units, because that is the
        scale the times it rebases are expressed in. Zero everywhere except the stage-4 hop of
        a tiled decode, whose next stage is 5 and whose origin is therefore a pixel frame.
        ``clip_start_frame`` is the first global pixel of *this* video latent (0 for a full
        clip; Dist's tile origin for a slice).
        """
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
        for block in self.det_stages[stage_i]:
            x, keyframes = block.forward_with_keyframes(x, keyframes)
        x = self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
        keyframe_x = upsample_keyframe_planes(self.upsamples[stage_i], keyframes.x)
        next_times = keyframe_clip_times(
            pixel_frame_indices,
            self._keyframe_time_strides[stage_i + 1],
            clip_start_frame,
            extra_origin=next_time_origin,
        )
        return x, dataclasses.replace(keyframes, x=keyframe_x).with_times(next_times).masked()

    def forward_stages_1_to_3_with_keyframes(
        self,
        z_noisy: torch.Tensor,
        keyframes: DecodeKeyframes,
        drop_leading_frame: bool = True,
        *,
        keyframe_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Dual-stream stages 1-3: video latent + keyframe planes to stage-4 inputs.
        Keyframe counterpart of :meth:`forward_stages_1_to_3`. ``z_noisy`` and
        ``keyframes.latents`` must already carry identical spatial padding -- the pad is
        applied symmetrically, so padding only one stream would offset every keyframe plane
        from the video by half the pad and read as ghosting rather than a failure.
        Times are relative to :attr:`DecodeKeyframes.clip_start_frame`. For a full-clip decode
        that is 0, so they match global ``t_s``. For a Dist slice they are ``t_s(index) -
        t_s(clip_start)`` from stage 1, because this path's "whole volume" *is* the slice.
        Additional in-volume tile origins are applied in :meth:`forward_stage_4_with_keyframes`.
        """
        keyframes.validate()
        if z_noisy.shape[-2:] != keyframes.latents.shape[-2:]:
            raise ValueError(
                f"keyframe latents must share the video latent's H/W (identical padding), got "
                f"{tuple(keyframes.latents.shape[-2:])} vs {tuple(z_noisy.shape[-2:])}"
            )
        stream = self._keyframe_stream_from_latents(keyframes, valid=keyframe_valid)
        x = self.per_channel_statistics.un_normalize(z_noisy).permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i in range(3):
            x, stream = self._run_det_stage_with_keyframes(
                x,
                stream,
                stage_i,
                drop_leading_frame,
                keyframes.pixel_frame_indices,
                0.0,
                clip_start_frame=keyframes.clip_start_frame,
            )
        return x, stream

    def forward_stage_4_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
        *,
        stage4_time_origin: float = 0.0,
        pixel_time_origin: float = 0.0,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Dual-stream stage 4 to stage-5 context. Keyframe counterpart of
        :meth:`forward_stage_4`.
        The ghost-pad crop applies to the video stream only: the trailing replicate is a
        temporal-border workaround and the keyframe planes have no temporal extent to pad.
        On the deferred (chunked) pathway neither stream is upsampled here: each stage-5
        block folds ``upsamples[3]`` into its own context inject. The returned stream's
        ``times`` are nonetheless the *stage-5* times, because that is where they are
        consumed -- the same asymmetry the video stream already has, whose returned
        ``x`` is a pre-upsample feature rather than stage-5 context.
        **Two origins, at two scales.** The video stream's RoPE is tile-local 0-based at every
        stage, so keyframe times must be rebased to whatever the tile's frame 0 is -- and this
        method spans two different temporal resolutions. ``stage4_time_origin`` is the tile's
        start in stage-4 input units (for the blocks); ``pixel_time_origin`` is its first
        global pixel frame (for stage 5). They are taken separately from the tile rather than
        derived from one another: ``drop_leading_frame`` and the causal first frame make
        ``pixel_origin == stride_t * stage4_origin`` an off-by-one trap, not an identity. Both
        are 0.0 for an untiled full-clip decode. ``clip_start_frame`` is subtracted in stage
        units as well, so a Dist slice whose first pixel is 56 still sees ``t_s(48) - t_s(56)``.
        """
        # Rebuild from global indices rather than trusting the caller's stream: stages 1-3 of a
        # full-clip decode are global, and Dist has already folded clip_start into clip times.
        keyframes = keyframes.with_times(
            keyframe_clip_times(
                pixel_frame_indices,
                self._keyframe_time_strides[3],
                clip_start_frame,
                extra_origin=stage4_time_origin,
            )
        )
        if self.deferred_stage4_upsample:
            return self._forward_stage_4_deferred_with_keyframes(
                x, keyframes, pixel_frame_indices, pad_trailing, pixel_time_origin, clip_start_frame
            )
        x, keyframes = self._run_det_stage_with_keyframes(
            x,
            keyframes,
            3,
            drop_leading_frame,
            pixel_frame_indices,
            pixel_time_origin,
            clip_start_frame=clip_start_frame,
        )
        if pad_trailing:
            x = diffusion_tiling.crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time,
                stage5_kernel_t=self.stage5_kernel[0],
            )
        return x, keyframes

    def _forward_stage_4_deferred_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        pad_trailing: bool,
        pixel_time_origin: float,
        clip_start_frame: int = 0,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Stage-4 blocks only, both streams, for the deferred (chunked) pathway.
        Mirrors :meth:`forward_stage_4`'s deferred branch: no ``upsamples[3]`` on either
        stream, ghost cropped at pre-upsample temporal resolution (video only).
        """
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
        for block in self.det_stages[3]:
            x, keyframes = block.forward_with_keyframes(x, keyframes)
        if pad_trailing:
            up_t = int(self.upsamples[3].stride[0])
            x = diffusion_tiling.crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time // up_t,
                stage5_kernel_t=max(1, -(-self.stage5_kernel[0] // up_t)),
            )
        stage5_times = keyframe_clip_times(
            pixel_frame_indices,
            self._keyframe_time_strides[4],
            clip_start_frame,
            extra_origin=pixel_time_origin,
        )
        return x, keyframes.with_times(stage5_times).masked()

    def forward_stage_4(
        self,
        x: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        """Stage 4 on a stage-4-input feature tile → stage-5 context (or pre-upsample feat).
        ``x`` is channels-last. When ``pad_trailing``, soft-crop the ghosting
        appendix before returning (ghost pad must already be present upstream).
        When ``deferred_stage4_upsample`` is set, runs NA blocks only (no
        ``upsamples[3]``) and crops ghost at pre-upsample temporal resolution.
        """
        if self.deferred_stage4_upsample:
            if self.mark_dynamic_shapes:
                for dim in (1, 2, 3):
                    torch._dynamo.mark_dynamic(x, dim)
            for block in self.det_stages[3]:
                x = block(x)
            if pad_trailing:
                up_t = int(self.upsamples[3].stride[0])
                x = diffusion_tiling.crop_trailing_context_natten_pad(
                    x,
                    n_latent_frames=self._natten_trailing_pad_latent_frames,
                    time_scale=self.video_downscale_factors.time // up_t,
                    stage5_kernel_t=max(1, -(-self.stage5_kernel[0] // up_t)),
                )
            return x

        x = self._run_det_stage(x, 3, drop_leading_frame)
        if pad_trailing:
            x = diffusion_tiling.crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time,
                stage5_kernel_t=self.stage5_kernel[0],
            )
        return x

    def _context_and_x_for_diff_step(self, context: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Build block-ready ``[context | conv_in_x_t(patched x)]`` for ``forward_diff_step``."""
        x = self.conv_in_x_t(patchify_channels_last(x_t, self.patch_size))
        return torch.cat([context, x], dim=-1)

    def _keyframe_context_and_x_for_diff_step(
        self,
        keyframe_context: torch.Tensor,
        keyframe_x_t: torch.Tensor,
        keyframe_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Keyframe ``[context | conv_in_x_t(patched x)]``, mask-zeroed.
        ``keyframe_x_t`` is ``(B, C_pix, P, H_pix, W_pix)`` -- the keyframe planes' own noised
        pixels, one pixel frame per plane, through the *shared* ``conv_in_x_t``.
        """
        x = self.conv_in_x_t(patchify_channels_last(keyframe_x_t, self.patch_size))
        x = x * keyframe_valid[None, :, None, None, None]
        return torch.cat([keyframe_context, x], dim=-1)

    def _x_for_diff_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """Conv-processed noised pixels only (deferred-context path)."""
        return self.conv_in_x_t(patchify_channels_last(x_t, self.patch_size))

    def _keyframe_x_for_diff_step(self, keyframe_x_t: torch.Tensor, keyframe_valid: torch.Tensor) -> torch.Tensor:
        """Mask-zeroed keyframe noised pixels only (deferred-context path).
        Contiguous by construction: the chunked pathway mutates this buffer in place. Masked in
        place: ``x`` is this call's own fresh projection, whose storage the fused block chain recycles.
        """
        x = self.conv_in_x_t(patchify_channels_last(keyframe_x_t, self.patch_size))
        return x.contiguous().mul_(keyframe_valid[None, :, None, None, None].to(x.dtype))

    def forward_diff_step(
        self,
        context_and_x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One stage-5 diffusion step. Returns the model prediction in pixel space.
        ``context_and_x`` is ``[latent_context | conv_in_x_t(x)]`` (channels-last), built
        at the call site via ``_context_and_x_for_diff_step``. That single buffer is
        reused across ``diff_blocks``: each block writes its output x-half back
        with ``copy_`` (no per-block ``cat``). One-tensor layout keeps Dynamo
        T/H/W symbols identical under ``mark_dynamic``.
        """
        x_half = context_and_x[..., self.context_channels :]
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x_half.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(context_and_x, dim)

        for block in self.diff_blocks:
            x_half.copy_(block.forward_combined(context_and_x, modulation))
        return self._pixels_from_stage5(x_half)

    def forward_diff_step_with_keyframes(
        self,
        context_and_x: torch.Tensor,
        keyframe_context_and_x: torch.Tensor,
        t: torch.Tensor,
        keyframe_planes: KeyframePlanes,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One dual-stream stage-5 step. Returns ``(video_pred, keyframe_pred)`` in pixel space.
        The keyframe stream at stage 5 is a genuine second *pixel* diffusion stream -- its own
        noised pixels through the shared ``conv_in_x_t``, its own per-block
        ``context_proj(keyframe_context)``, the same AdaLN modulation -- not a zero tensor and
        not the context. It is evolved through the same Euler loop as video so the hidden
        state the joint attention reads sits at the noise level it was trained to see, then
        discarded: callers use the video prediction only.
        Both buffers follow ``forward_diff_step``'s ``[context | x]`` layout and the same
        ``copy_``-into-a-view discipline. Video T/H/W stay dynamic; the keyframe plane axis is
        specialized, since ``keyframe_planes`` pin it.
        """
        x_half = context_and_x[..., self.context_channels :]
        keyframe_half = keyframe_context_and_x[..., self.context_channels :]
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x_half.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(context_and_x, dim)
            # Keyframe dim 1 is the plane count, not T, and keyframe_planes pin it.
            # Marking it dynamic and then specializing to P raises ConstraintViolationError.
            for dim in (2, 3):
                torch._dynamo.mark_dynamic(keyframe_context_and_x, dim)

        for block in self.diff_blocks:
            x_out, keyframe_out = block.forward_combined_with_keyframes(
                context_and_x,
                keyframe_context_and_x,
                modulation,
                keyframe_planes,
            )
            x_half.copy_(x_out)
            keyframe_half.copy_(keyframe_out)

        return self._pixels_from_stage5(x_half), self._pixels_from_stage5(keyframe_half)

    def _pixels_from_stage5(self, x: torch.Tensor) -> torch.Tensor:
        """Shared stage-5 tail: ``norm_out`` -> ``conv_out`` -> channels-first -> unpatchify.
        Without the fused tail, ``norm_out`` runs row chunk by row chunk into a contiguous ``x`` (the
        block chain's own output, read by nothing after this) and ``conv_out`` + unpatchify one
        temporal chunk at a time into the pixel volume: same values, no volume-sized transients.
        Each eager RMSNorm launch stays under 2**32 elements (torch < 2.13 CUDA kernels wrap
        flattened offsets).
        """
        if stage5_tail_supported(x, self.norm_out, self.conv_out, self.patch_size):
            return stage5_tail(x, self.norm_out, self.conv_out, self.patch_size)
        if x.is_contiguous():
            rows = x.view(-1, x.shape[-1])
            for r0 in range(0, rows.shape[0], _NORM_OUT_CHUNK_ROWS):
                chunk = rows[r0 : r0 + _NORM_OUT_CHUNK_ROWS]
                chunk.copy_(apply_rms_norm_uint32_safe(self.norm_out, chunk))
        else:
            x = apply_rms_norm_uint32_safe(self.norm_out, x)
        b, t, h5, w5, _ = x.shape
        p = self.patch_size
        pixels = torch.empty((b, self.out_channels, t, h5 * p, w5 * p), device=x.device, dtype=x.dtype)
        frame_bytes = b * h5 * w5 * self.out_channels * p * p * x.element_size()
        chunk_t = max(1, min(t, _TAIL_CHUNK_BYTES // max(frame_bytes, 1)))
        for t0 in range(0, t, chunk_t):
            t1 = min(t, t0 + chunk_t)
            y = self.conv_out(x[:, t0:t1]).permute(0, 4, 1, 2, 3).contiguous()
            pixels[:, :, t0:t1].copy_(unpatchify(y, patch_size_hw=self.patch_size, patch_size_t=1))
        return pixels

    def forward_diff_step_deferred(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        t: torch.Tensor,
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        """Stage-5 step with deferred context: only ``x`` + low-res ``stage4_feat``.
        Marks T/H/W dynamic on both tensors. CHUNKED blocks upsample then
        ``context_proj`` on the host before attn+mlp; BLACKWELL_DSL
        (``DSLDiffusionBlockChain``) folds that hop into the fused kernel and never
        materialises full-resolution context. ``drop_leading_frame`` must match the
        flag used for this tile's stage-4 path (origin tile vs non-origin).
        """
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
                torch._dynamo.mark_dynamic(stage4_feat, dim)

        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.dsl_kernels import DSLDiffusionBlockChain  # noqa: PLC0415

        if isinstance(self.diff_blocks, DSLDiffusionBlockChain):
            # Ping-pong fused launches; same deferred (x, stage4_feat) contract.
            x = self.diff_blocks(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)
        else:
            for block in self.diff_blocks:
                x = block.forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)

        return self._pixels_from_stage5(x)

    def forward_diff_step_deferred_with_keyframes(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        keyframe_x: torch.Tensor,
        keyframe_stage4_feat: torch.Tensor,
        t: torch.Tensor,
        keyframe_planes: KeyframePlanes,
        *,
        drop_leading_frame: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-stream stage-5 step with deferred context. Keyframe counterpart of
        :meth:`forward_diff_step_deferred`.
        Each stream carries its own pre-upsample stage-4 feature and injects it per block,
        so full-resolution context is never materialised for either. ``drop_leading_frame``
        is the video stream's tiling property; the keyframe inject always collapses its
        temporal stride.
        """
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
                torch._dynamo.mark_dynamic(stage4_feat, dim)
            # Plane count, not T -- see the note in the combined path above.
            for dim in (2, 3):
                torch._dynamo.mark_dynamic(keyframe_x, dim)
                torch._dynamo.mark_dynamic(keyframe_stage4_feat, dim)

        # The DSL chain drives itself, so both streams recycle output buffers instead of
        # allocating a volume per block per stream; the chunked blocks expose the same
        # ``forward_x_ctx_with_keyframes``, so the fallback loop drives those.
        from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.dsl_kernels import DSLDiffusionBlockChain  # noqa: PLC0415

        if isinstance(self.diff_blocks, DSLDiffusionBlockChain):
            x, keyframe_x = self.diff_blocks.forward_x_ctx_with_keyframes(
                x,
                stage4_feat,
                keyframe_x,
                keyframe_stage4_feat,
                modulation,
                keyframe_planes,
                drop_leading_frame=drop_leading_frame,
            )
        else:
            for block in self.diff_blocks:
                x, keyframe_x = block.forward_x_ctx_with_keyframes(
                    x,
                    stage4_feat,
                    keyframe_x,
                    keyframe_stage4_feat,
                    modulation,
                    keyframe_planes,
                    drop_leading_frame=drop_leading_frame,
                )

        return self._pixels_from_stage5(x), self._pixels_from_stage5(keyframe_x)

    def _euler_step(
        self, x_t: torch.Tensor, model_out: torch.Tensor, t_now: torch.Tensor, t_next: torch.Tensor
    ) -> torch.Tensor:
        """One reverse-diffusion Euler update: advance ``x_t`` from ``t_now`` to
        ``t_next`` given the model's prediction at ``t_now``.
        """
        compute_dtype = x_t.dtype
        dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1))).to(torch.float32)
        x_t_fp32 = x_t.to(torch.float32)
        v_pred = model_out if self.model_output_type == "v" else to_velocity(x_t_fp32, t_now, model_out)
        return (x_t_fp32 - dt * v_pred).to(compute_dtype)

    def _decode_one_tile(
        self,
        feat_tile: torch.Tensor,
        x_t_tile_init: torch.Tensor,
        *,
        is_origin: bool,
        timestep: torch.Tensor,
        pad_trailing: bool,
    ) -> torch.Tensor:
        """Run stage 4 + diffusion on one stage-4 feature tile (isolation)."""
        context_tile = self.forward_stage_4(
            feat_tile,
            drop_leading_frame=is_origin,
            pad_trailing=pad_trailing,
        )

        x_t = x_t_tile_init
        _, num_steps = timestep.shape
        for i in range(num_steps - 1):
            t_now = timestep[:, i]
            t_next = timestep[:, i + 1]
            if self.deferred_stage4_upsample:
                x = self._x_for_diff_step(x_t)
                model_out = self.forward_diff_step_deferred(x, context_tile, t_now, drop_leading_frame=is_origin).to(
                    torch.float32
                )
            else:
                context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
                model_out = self.forward_diff_step(context_and_x, t_now).to(torch.float32)
            x_t = self._euler_step(x_t, model_out, t_now, t_next)

        t_now = timestep[:, -1]
        if self.deferred_stage4_upsample:
            x = self._x_for_diff_step(x_t)
            model_out = self.forward_diff_step_deferred(x, context_tile, t_now, drop_leading_frame=is_origin)
        else:
            context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
            model_out = self.forward_diff_step(context_and_x, t_now)
        if self.model_output_type == "x0":
            return model_out
        return self._euler_step(x_t, model_out.to(torch.float32), t_now, torch.zeros_like(t_now))

    def _stage5_canvas_from_context(
        self,
        context_tile: torch.Tensor,
        *,
        drop_leading_frame: bool,
    ) -> tuple[int, int, int]:
        """``(T, H_pix, W_pix)`` of the stage-5 pixel canvas a context tile implies.
        On the combined pathway the context is already at stage-5 resolution and
        ``_context_and_x_for_diff_step`` patchifies ``x_t`` by ``patch_size``, so the canvas
        is just ``(T, H * patch_size, W * patch_size)``. On the deferred pathway the tile is
        still pre-upsample: each stage-5 block folds ``upsamples[3]`` into its inject, so
        apply that stride here -- including the leading-frame drop the fold performs when the
        temporal stride is 2.
        """
        t, h, w = context_tile.shape[1], context_tile.shape[2], context_tile.shape[3]
        if self.deferred_stage4_upsample:
            # Context is still pre-upsample, so this is the same geometry as a stage-4
            # input. Ghost crop already ran, so do not re-apply the kernel-T floor.
            return diffusion_tiling.stage5_pixel_shape_from_stage4(
                t,
                h,
                w,
                upsample_stride=tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
                patch_size=self.patch_size,
                stage5_kernel_t=self.stage5_kernel[0],
                drop_leading_frame=drop_leading_frame,
                pad_trailing=False,
            )
        return t, h * self.patch_size, w * self.patch_size

    def _decode_one_tile_with_keyframes(  # noqa: PLR0913
        self,
        feat_tile: torch.Tensor,
        keyframes: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        *,
        is_origin: bool,
        timestep: torch.Tensor,
        pad_trailing: bool,
        generator: torch.Generator | None,
        compute_dtype: torch.dtype,
        x_t_tile_init: torch.Tensor | None = None,
        stage4_time_origin: float = 0.0,
        pixel_time_origin: float = 0.0,
        clip_start_frame: int = 0,
    ) -> torch.Tensor:
        """Stage 4 + dual-stream diffusion on one stage-4 feature tile.
        Both streams are Euler-stepped together; only the video pixels are returned. The
        keyframe pixel stream exists so the hidden state the joint attention reads stays at
        the noise level it was trained on, and is discarded here (upstream exposes it only
        through its explicit per-step entry points, which the trainer uses).
        Noise is sized from the stage-5 context rather than from a re-derived tile geometry;
        see :meth:`_stage5_canvas_from_context` for the deferred-pathway correction. The
        keyframe canvas differs only in its frame count -- one pixel frame per plane, since
        keyframe upsampling collapses its temporal stride.
        ``x_t_tile_init`` lets a tiled decode share one global noise field across tiles (edge
        policy applied by the caller, as the plain path does); ``None`` draws fresh noise. The
        keyframe stream always draws its own -- its planes are not part of the video canvas.
        """
        context_tile, keyframes = self.forward_stage_4_with_keyframes(
            feat_tile,
            keyframes,
            pixel_frame_indices,
            drop_leading_frame=is_origin,
            pad_trailing=pad_trailing,
            stage4_time_origin=stage4_time_origin,
            pixel_time_origin=pixel_time_origin,
            clip_start_frame=clip_start_frame,
        )

        batch = context_tile.shape[0]
        canvas_t, canvas_h, canvas_w = self._stage5_canvas_from_context(context_tile, drop_leading_frame=is_origin)
        randn_device = generator.device if generator is not None else feat_tile.device

        def _noise(frames: int) -> torch.Tensor:
            return torch.randn(
                (batch, self.out_channels, frames, canvas_h, canvas_w),
                dtype=compute_dtype,
                generator=generator,
                device=randn_device,
            ).to(feat_tile.device)

        x_t = _noise(canvas_t) if x_t_tile_init is None else x_t_tile_init
        keyframe_x_t = _noise(keyframes.num_planes)

        def _step(t_now: torch.Tensor, *, final: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal x_t, keyframe_x_t
            if self.deferred_stage4_upsample:
                x_in = self._x_for_diff_step(x_t)
                keyframe_x_in = self._keyframe_x_for_diff_step(keyframe_x_t, keyframes.planes.valid.device_tensor)
                if final and self.model_output_type == "x0":
                    # An x0 model's last step never reads the noise again; release both canvases now.
                    x_t = keyframe_x_t = None  # type: ignore[assignment]
                return self.forward_diff_step_deferred_with_keyframes(
                    x_in,
                    context_tile,
                    keyframe_x_in,
                    keyframes.x,
                    t_now,
                    keyframes.planes,
                    drop_leading_frame=is_origin,
                )
            context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
            keyframe_context_and_x = self._keyframe_context_and_x_for_diff_step(
                keyframes.x, keyframe_x_t, keyframes.planes.valid.device_tensor
            )
            return self.forward_diff_step_with_keyframes(context_and_x, keyframe_context_and_x, t_now, keyframes.planes)

        _, num_steps = timestep.shape
        for i in range(num_steps - 1):
            t_now = timestep[:, i]
            t_next = timestep[:, i + 1]
            video_out, keyframe_out = _step(t_now)
            x_t = self._euler_step(x_t, video_out.to(torch.float32), t_now, t_next)
            keyframe_x_t = self._euler_step(keyframe_x_t, keyframe_out.to(torch.float32), t_now, t_next)

        t_now = timestep[:, -1]
        video_out, _ = _step(t_now, final=True)
        if self.model_output_type == "x0":
            return video_out
        return self._euler_step(x_t, video_out.to(torch.float32), t_now, torch.zeros_like(t_now))

    def _decode_temporal_group_isolated_with_keyframes(  # noqa: PLR0913
        self,
        tiles: List[Tile],
        feat_s4: torch.Tensor,
        stream: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        content_s4_frames: int,
        x_t_init: torch.Tensor | None,
        timestep: torch.Tensor,
        full_video_shape: VideoLatentShape,
        curr_temporal_slice: slice,
        generator: torch.Generator | None,
        *,
        complementary: bool,
        clip_start_frame: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Decode one temporal group's tiles with keyframes and blend into a group buffer.
        Keyframe counterpart of :meth:`_decode_temporal_group_isolated`. Each tile carries
        only the planes near it -- those inside its pixel-frame span plus the nearest plane on
        each *side* of it (:func:`planes_for_tile`) -- and the two origins that plane set has
        to be rebased against come from the tile, never from each other.
        Tile ``out_coords`` are local to this latent. Dist slices keep global
        ``pixel_frame_indices`` and pass ``clip_start_frame`` so selection lines up.
        """
        group_temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
        group_shape = full_video_shape._replace(frames=group_temporal_len)
        full_torch_shape = full_video_shape.to_torch_shape()
        accum_dtype = diffusion_tiling.accumulator_dtype_for(feat_s4.dtype)
        # One tile with all-ones masks covering the group *is* the blend (0 + x * 1).
        sole_tile = len(tiles) == 1 and complementary and all(bool((mask == 1).all()) for mask in tiles[0].masks_1d)
        buffer = (
            None if sole_tile else torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)
        )
        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
        local_temporal_slice = slice(0, group_temporal_len)

        compute_dtype = feat_s4.dtype
        up3_stride = tuple(self.upsamples[3].stride)

        for tile_index, tile in enumerate(tiles):
            feat_tile, is_origin, pad_trailing, content_thw = diffusion_tiling.slice_stage4_tile(
                feat_s4, tile, content_frames=content_s4_frames
            )
            # Two origins at two scales -- see forward_stage_4_with_keyframes.
            stage4_origin = tile.in_coords[1].indices(content_s4_frames)[0]
            pixel_lo, pixel_hi, _ = tile.out_coords[2].indices(full_torch_shape[2])

            # ``out_coords`` are local to this latent. Dist slices keep global indices and
            # ``clip_start_frame`` as the origin, so a local ``[0, 72)`` still has to select
            # global ``[56, 127]``.
            keep = planes_for_tile(pixel_frame_indices, pixel_lo, pixel_hi - 1, clip_start_frame=clip_start_frame)
            if not bool(keep.any()):
                raise RuntimeError(
                    f"tile covering pixel frames [{pixel_lo + clip_start_frame}, "
                    f"{pixel_hi - 1 + clip_start_frame}] selected no keyframe planes "
                    f"out of {int(pixel_frame_indices.shape[0])}; planes_for_tile always keeps at least one"
                )
            tile_stream = stream.select_planes(keep).crop_spatial(tile.in_coords[2], tile.in_coords[3])
            # Not debug-only: the decode below needs this tile's plane positions.
            tile_indices = pixel_frame_indices[keep.to(pixel_frame_indices.device)]
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "keyframe decode: tile %d/%d frames [%d, %d), stage-4 extent %dx%dx%d, %d of %d planes at %s",
                    tile_index + 1,
                    len(tiles),
                    pixel_lo,
                    pixel_hi,
                    feat_tile.shape[1],
                    feat_tile.shape[2],
                    feat_tile.shape[3],
                    tile_stream.num_planes,
                    int(pixel_frame_indices.shape[0]),
                    tile_indices.tolist(),
                )

            x_t_tile_init: torch.Tensor | None = None
            if x_t_init is not None:
                stage5_f, stage5_h, stage5_w = diffusion_tiling.stage5_pixel_shape_from_stage4(
                    content_thw[0],
                    content_thw[1],
                    content_thw[2],
                    upsample_stride=up3_stride,  # type: ignore[arg-type]
                    patch_size=self.patch_size,
                    stage5_kernel_t=self.stage5_kernel[0],
                    drop_leading_frame=is_origin,
                    pad_trailing=pad_trailing,
                )
                # Same edge policy as the plain path: expand/crop the shared noise field
                # rather than drawing fresh noise, since NA mixes padded values inward.
                x_t_tile_init = x_t_init[tile.out_coords]
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 2, stage5_f, mode="repeat_last")
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 3, stage5_h, mode="symmetric")
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 4, stage5_w, mode="symmetric")

            pixel_tile = self._decode_one_tile_with_keyframes(
                feat_tile,
                tile_stream,
                tile_indices,
                is_origin=is_origin,
                timestep=timestep,
                pad_trailing=pad_trailing,
                generator=generator,
                compute_dtype=compute_dtype,
                x_t_tile_init=x_t_tile_init,
                stage4_time_origin=float(stage4_origin),
                pixel_time_origin=float(pixel_lo),
                clip_start_frame=clip_start_frame,
            )
            # The tile's pixels are a fresh tensor; its leased stage-5 volumes go back to the workspace.
            decode_workspace.release_tile()
            content_pixel_shape = diffusion_tiling.pixel_tile_shape(full_torch_shape, tile.out_coords)
            pixel_tile = diffusion_tiling.crop_pixels_to_content(
                pixel_tile,
                content_pixel_shape[2],
                content_pixel_shape[3],
                content_pixel_shape[4],
            )
            if sole_tile and pixel_tile.shape == group_shape.to_torch_shape():
                pixel_tile = pixel_tile.to(accum_dtype)
                _release_cached_blocks(pixel_tile.device)
                return pixel_tile, None
            if buffer is None:
                buffer = torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)

            masks = tuple(upload_from_host(m.to(dtype=torch.float32), buffer.device) for m in tile.masks_1d)
            local_coords = (
                tile.out_coords[0],
                tile.out_coords[1],
                local_temporal_slice,
                tile.out_coords[3],
                tile.out_coords[4],
            )
            _blend_tile_into(buffer, weights, local_coords, pixel_tile, masks)

        # The tiles' transients are gone; hand their segments back before the caller emits the group.
        del pixel_tile
        _release_cached_blocks(buffer.device)
        return buffer, weights

    def _tile_workspace(self, tiles: List[Tile], feat_s4: torch.Tensor, content_s4_frames: int) -> object | None:
        """The per-decode buffer workspace the fused stage-5 blocks draw from; ``None`` without ``ltx-kernels``.
        Large-volume slots hold the biggest stage-5 volume any tile asks for, so they are allocated once.
        """
        st, sh, sw = (int(v) for v in self.upsamples[3].stride)
        rows = 0
        for tile in tiles:
            feat_tile, _, _, _ = diffusion_tiling.slice_stage4_tile(feat_s4, tile, content_frames=content_s4_frames)
            _, t, h, w, _ = feat_tile.shape
            rows = max(rows, t * st * h * sh * w * sw)
        if not decode_workspace.available():
            return None
        channels = int(self.conv_in_x_t.out_features)
        return decode_workspace.tile_workspace((rows + decode_workspace.out_slack_rows()) * channels)

    def _decode_groups_with_keyframes(  # noqa: PLR0913, PLR0915
        self,
        feat_s4: torch.Tensor,
        stream: KeyframeStream,
        pixel_frame_indices: torch.Tensor,
        latent: torch.Tensor,
        tiling_config: TilingConfig,
        timestep: torch.Tensor,
        generator: torch.Generator | None,
        *,
        content_pixel: VideoLatentShape,
        h_pad: diffusion_tiling.AxisPad | None,
        w_pad: diffusion_tiling.AxisPad | None,
        as_fhwc: bool,
        clip_start_frame: int = 0,
    ) -> Iterator[torch.Tensor]:
        """Tiled keyframe decode, streaming one temporal group at a time.
        Same shape as :meth:`_decode_pixels`: only the trailing overlap of the previous group
        is retained between iterations, and a group's exclusive frames are yielded before the
        next group decodes. That keeps residency at roughly two tile extents rather than a
        whole video, which matters more here than on the plain path -- a keyframe decode also
        carries a second pixel stream through stage 5. Every group's tiles draw their large
        stage-5 buffers from one workspace allocated for the decode.
        """
        full_video_shape = (
            VideoLatentShape.from_torch_shape(latent.shape)
            .upscale(self.video_downscale_factors)
            ._replace(channels=self.out_channels)
        )
        target_shape = full_video_shape.to_torch_shape()
        strides = [tuple(u.stride) for u in self.upsamples]
        s4_t, s4_h, s4_w = diffusion_tiling.stage4_thw_from_latent(
            strides, latent.shape[2], latent.shape[3], latent.shape[4], drop_leading_frame=True
        )
        tiles = diffusion_tiling.prepare_tile_schedule(
            torch.Size([latent.shape[0], latent.shape[1], s4_t, s4_h, s4_w]),
            tiling_config,
            upsample3_stride=tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
            patch_size=self.patch_size,
            min_tile_size=self.tile_min_sizes,
            tile_halos=self.tile_halos,
        )
        complementary = masks_are_complementary(tiles, target_shape)
        groups = group_tiles_by_temporal_slice(tiles)
        group_slices = [slice(*group[0].out_coords[2].indices(target_shape[2])[:2]) for group in groups]

        single_step_x0 = timestep.shape[1] == 1 and self.model_output_type == "x0"
        x_t_init: torch.Tensor | None = None
        if not single_step_x0:
            randn_device = generator.device if generator is not None else latent.device
            x_t_init = torch.randn(
                tuple(target_shape), dtype=latent.dtype, generator=generator, device=randn_device
            ).to(latent.device)

        logger.info(
            "keyframe decode: %d tile(s) in %d temporal group(s), %d frames at %dx%d, %d planes",
            len(tiles),
            len(groups),
            content_pixel.frames,
            content_pixel.height,
            content_pixel.width,
            int(pixel_frame_indices.shape[0]),
        )

        scaled_h_pad = diffusion_tiling.scale_axis_pad(h_pad, self.video_downscale_factors.height)
        scaled_w_pad = diffusion_tiling.scale_axis_pad(w_pad, self.video_downscale_factors.width)
        overlap_stub: torch.Tensor | None = None
        overlap_stub_weights: torch.Tensor | None = None
        workspace = self._tile_workspace(tiles, feat_s4, s4_t)

        def _emit(buf: torch.Tensor, wts: torch.Tensor | None, global_start: int) -> torch.Tensor | None:
            """Finalize, crop to content, and lay out one emitted run of frames."""
            if global_start >= content_pixel.frames or buf.shape[2] < 1:
                return None
            frames_keep = min(buf.shape[2], content_pixel.frames - global_start)
            if frames_keep < 1:
                return None
            chunk = buf[:, :, :frames_keep]
            if wts is not None:
                floor = diffusion_tiling._weight_floor(wts.dtype)
                chunk = chunk / wts[:, :, :frames_keep].clamp(min=floor)
            if as_fhwc:
                # Crop first, then cast and re-lay out in one copy: one frame-run transient instead of two.
                cropped = diffusion_tiling.crop_pixels_to_content(
                    chunk,
                    frames_keep,
                    content_pixel.height,
                    content_pixel.width,
                    h_pad=scaled_h_pad,
                    w_pad=scaled_w_pad,
                )[0]
                out = torch.empty(
                    (cropped.shape[1], cropped.shape[2], cropped.shape[3], cropped.shape[0]),
                    device=cropped.device,
                    dtype=latent.dtype,
                )
                out.copy_(cropped.permute(1, 2, 3, 0))
                return out
            return diffusion_tiling.crop_pixels_to_content(
                chunk.to(latent.dtype),
                frames_keep,
                content_pixel.height,
                content_pixel.width,
                h_pad=scaled_h_pad,
                w_pad=scaled_w_pad,
            )

        for group_index, group in enumerate(groups):
            curr_temporal_slice = group_slices[group_index]
            logger.info(
                "keyframe decode: group %d/%d, frames [%d, %d)",
                group_index + 1,
                len(groups),
                curr_temporal_slice.start,
                curr_temporal_slice.stop,
            )
            # Each group's tiles have their own transient sizes; hand back the previous group's free segments first.
            _release_cached_blocks(feat_s4.device)
            # Same-extent tiles and blocks build their abs-RoPE tables once per group.
            with decode_workspace.scope(workspace), abs_rope_table_cache(self):
                buffer, weights = self._decode_temporal_group_isolated_with_keyframes(
                    group,
                    feat_s4,
                    stream,
                    pixel_frame_indices,
                    s4_t,
                    x_t_init,
                    timestep,
                    full_video_shape,
                    curr_temporal_slice,
                    generator,
                    complementary=complementary,
                    clip_start_frame=clip_start_frame,
                )

            if overlap_stub is not None:
                overlap_len = int(overlap_stub.shape[2])
                if overlap_len > 0:
                    overlap_stub += buffer[:, :, :overlap_len]
                    buffer[:, :, :overlap_len] = overlap_stub
                    if not complementary:
                        assert overlap_stub_weights is not None
                        assert weights is not None
                        overlap_stub_weights += weights[:, :, :overlap_len]
                        weights[:, :, :overlap_len] = overlap_stub_weights
                overlap_stub = None
                overlap_stub_weights = None

            if group_index + 1 < len(groups):
                next_start = group_slices[group_index + 1].start
                exclusive_len = min(max(0, next_start - curr_temporal_slice.start), buffer.shape[2])
                emitted = _emit(
                    buffer[:, :, :exclusive_len],
                    None if weights is None else weights[:, :, :exclusive_len],
                    curr_temporal_slice.start,
                )
                if emitted is not None:
                    yield emitted
                del emitted
                # Retain only the trailing overlap for the next group's handoff.
                overlap_stub = buffer[:, :, exclusive_len:].clone()
                if not complementary:
                    assert weights is not None
                    overlap_stub_weights = weights[:, :, exclusive_len:].clone()
                del buffer, weights
            else:
                emitted = _emit(buffer, weights, curr_temporal_slice.start)
                if emitted is not None:
                    yield emitted

    def _decode_pixels_with_keyframes(
        self,
        latent: torch.Tensor,
        keyframes: DecodeKeyframes,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        as_fhwc: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Keyframe-aware decode, yielding one chunk of ``(B, C, F, H, W)`` in ``[-1, 1]``.
        Stages 1-3 run once on the whole volume for both streams -- so the keyframe stream
        reaches stage 4 with *global* times -- then stages 4-5 run per tile with pixel blend.
        With ``tiling_config=None`` that is a single tile and the whole thing is one pass.
        """
        content_shape = VideoLatentShape.from_torch_shape(latent.shape)
        content_pixel = content_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        keyframes.validate(num_frames=content_pixel.frames)

        latent, (_t_pad, h_pad, w_pad) = diffusion_tiling.ensure_min_latent_shape(latent, self.stage_min_tile_sizes)
        # Same spatial floor for the keyframe planes, with the plane axis pinned by a
        # temporal minimum of 1. The pad is symmetric, so padding only one stream would
        # offset every plane from the video by half of it.
        _min_t, min_h, min_w = self.stage_min_tile_sizes
        keyframe_latents, (_, keyframe_h_pad, keyframe_w_pad) = diffusion_tiling.ensure_min_latent_shape(
            keyframes.latents, (1, min_h, min_w)
        )
        if (keyframe_h_pad, keyframe_w_pad) != (h_pad, w_pad):
            raise RuntimeError(
                f"keyframe spatial pad {(keyframe_h_pad, keyframe_w_pad)} != video pad {(h_pad, w_pad)}; "
                "the two streams must share one spatial origin"
            )
        padded_keyframes = dataclasses.replace(keyframes, latents=keyframe_latents)

        # Ghost pad is a temporal-border workaround for the video stream; keyframe planes have
        # no temporal extent to pad. The appendix is cropped off context before stage 5.
        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, self._natten_trailing_pad_latent_frames
        )
        feat_s4, stream = self.forward_stages_1_to_3_with_keyframes(
            latent_padded, padded_keyframes, drop_leading_frame=True
        )

        batch = latent.shape[0]
        timestep = self.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
        if tiling_config is not None:
            yield from self._decode_groups_with_keyframes(
                feat_s4,
                stream,
                keyframes.pixel_frame_indices,
                latent,
                tiling_config,
                timestep,
                generator,
                content_pixel=content_pixel,
                h_pad=h_pad,
                w_pad=w_pad,
                as_fhwc=as_fhwc,
                clip_start_frame=keyframes.clip_start_frame,
            )
            return

        logger.info("keyframe decode: untiled, %d frames, %d planes", content_pixel.frames, stream.num_planes)
        pixels = self._decode_one_tile_with_keyframes(
            feat_s4,
            stream,
            keyframes.pixel_frame_indices,
            is_origin=True,
            timestep=timestep,
            pad_trailing=True,
            generator=generator,
            compute_dtype=latent.dtype,
            clip_start_frame=keyframes.clip_start_frame,
        )
        pixels = diffusion_tiling.crop_pixels_to_content(
            pixels,
            content_pixel.frames,
            content_pixel.height,
            content_pixel.width,
            h_pad=diffusion_tiling.scale_axis_pad(h_pad, self.video_downscale_factors.height),
            w_pad=diffusion_tiling.scale_axis_pad(w_pad, self.video_downscale_factors.width),
        ).to(latent.dtype)
        if as_fhwc:
            yield pixels[0].permute(1, 2, 3, 0).contiguous()
        else:
            yield pixels

    def _decode_video_with_keyframes(
        self,
        latent: torch.Tensor,
        keyframes: DecodeKeyframes,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Keyframe-aware decode, yielding float chunk(s) ``[f, h, w, c]`` in ``[0, 1]``.
        Implementation of :meth:`decode_video` when ``keyframes`` is set. ``keyframes`` carries
        single-frame latents plus their global pixel frame indices; every video token then
        attends to the nearest planes through a joint neighborhood-attention window.
        """

        def to_rgb(frames: torch.Tensor) -> torch.Tensor:
            return frames.add_(1).mul_(0.5).clamp_(0, 1)

        for chunk in self._decode_pixels_with_keyframes(
            latent, keyframes, tiling_config, generator=generator, as_fhwc=True
        ):
            yield to_rgb(chunk)

    def _decode_temporal_group_isolated(
        self,
        tiles: List[Tile],
        feat_s4: torch.Tensor,
        content_s4_frames: int,
        x_t_init: torch.Tensor | None,
        timestep: torch.Tensor,
        full_video_shape: VideoLatentShape,
        curr_temporal_slice: slice,
        generator: torch.Generator | None,
        *,
        complementary: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Decode every tile of one temporal group in isolation and blend."""
        group_temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
        group_shape = full_video_shape._replace(frames=group_temporal_len)
        full_torch_shape = full_video_shape.to_torch_shape()
        accum_dtype = diffusion_tiling.accumulator_dtype_for(feat_s4.dtype)
        # One tile with all-ones masks covering the group *is* the blend (0 + x * 1).
        sole_tile = len(tiles) == 1 and complementary and all(bool((mask == 1).all()) for mask in tiles[0].masks_1d)
        buffer = (
            None if sole_tile else torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)
        )
        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
        local_temporal_slice = slice(0, group_temporal_len)

        compute_dtype = feat_s4.dtype
        randn_device = generator.device if generator is not None else feat_s4.device
        up3_stride = tuple(self.upsamples[3].stride)

        for tile in tiles:
            feat_tile, is_origin, pad_trailing, content_thw = diffusion_tiling.slice_stage4_tile(
                feat_s4, tile, content_frames=content_s4_frames
            )
            content_pixel_shape = diffusion_tiling.pixel_tile_shape(full_torch_shape, tile.out_coords)
            stage5_f, stage5_h, stage5_w = diffusion_tiling.stage5_pixel_shape_from_stage4(
                content_thw[0],
                content_thw[1],
                content_thw[2],
                upsample_stride=up3_stride,  # type: ignore[arg-type]
                patch_size=self.patch_size,
                stage5_kernel_t=self.stage5_kernel[0],
                drop_leading_frame=is_origin,
                pad_trailing=pad_trailing,
            )

            if x_t_init is None:
                x_t_tile_init = torch.randn(
                    (content_pixel_shape[0], content_pixel_shape[1], stage5_f, stage5_h, stage5_w),
                    dtype=compute_dtype,
                    generator=generator,
                    device=randn_device,
                ).to(feat_s4.device)
            else:
                # Expand/crop to stage-5 canvas with the same edge policy as latent
                # size-floor / ghost pad (not fresh noise - NA mixes padded values
                # into kept pixels near the boundary).
                x_t_tile_init = x_t_init[tile.out_coords]
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 2, stage5_f, mode="repeat_last")
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 3, stage5_h, mode="symmetric")
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 4, stage5_w, mode="symmetric")

            pixel_tile = self._decode_one_tile(
                feat_tile,
                x_t_tile_init,
                is_origin=is_origin,
                timestep=timestep,
                pad_trailing=pad_trailing,
            )
            pixel_tile = diffusion_tiling.crop_pixels_to_content(
                pixel_tile,
                content_pixel_shape[2],
                content_pixel_shape[3],
                content_pixel_shape[4],
            ).to(accum_dtype)
            if sole_tile and pixel_tile.shape == group_shape.to_torch_shape():
                return pixel_tile, None
            if buffer is None:
                buffer = torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)

            masks = tuple(upload_from_host(m.to(dtype=torch.float32), buffer.device) for m in tile.masks_1d)
            local_coords = (
                tile.out_coords[0],
                tile.out_coords[1],
                local_temporal_slice,
                tile.out_coords[3],
                tile.out_coords[4],
            )
            buffer[local_coords] += scale_by_masks_1d(pixel_tile, masks)
            if weights is not None:
                strength = torch.ones(pixel_tile.shape, device=buffer.device, dtype=buffer.dtype)
                weights[local_coords] += scale_by_masks_1d(strength, masks)

        return buffer, weights

    def _decode_pixels(  # noqa: PLR0912, PLR0915
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        as_fhwc: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Decode latent to pixels, yielding temporal chunks.
        Default yields raw ``(B, C, F, H, W)`` in ``[-1, 1]``. With ``as_fhwc=True``
        (used by :meth:`decode_video`), each chunk is materialized once as
        contiguous ``[F, H, W, C]`` still in ``[-1, 1]`` - layout copy only;
        range mapping stays in ``to_rgb``.
        Stages 1-3 run once on the full volume; stages 4-5 run per tile with
        pixel blend (one tile / one group when untiled or no real split).
        Across temporal groups only the trailing overlap is retained between
        iterations; exclusive frames are yielded before the next group decodes.
        Peak residency is ~two tile extents (current buffer + still-live emit /
        overlap stub), not a single ``tile + overlap`` slab.
        """
        content_shape = VideoLatentShape.from_torch_shape(latent.shape)
        content_pixel = content_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)

        latent, (_t_pad, h_pad, w_pad) = diffusion_tiling.ensure_min_latent_shape(latent, self.stage_min_tile_sizes)
        spatial_scale = (self.video_downscale_factors.height, self.video_downscale_factors.width)
        work_shape = VideoLatentShape.from_torch_shape(latent.shape)
        full_video_shape = work_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target_shape = full_video_shape.to_torch_shape()

        strides = [tuple(u.stride) for u in self.upsamples]
        s4_t, s4_h, s4_w = diffusion_tiling.stage4_thw_from_latent(
            strides, latent.shape[2], latent.shape[3], latent.shape[4], drop_leading_frame=True
        )
        tiles = diffusion_tiling.prepare_tile_schedule(
            torch.Size([latent.shape[0], latent.shape[1], s4_t, s4_h, s4_w]),
            tiling_config,
            upsample3_stride=tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
            patch_size=self.patch_size,
            min_tile_size=self.tile_min_sizes,
            tile_halos=self.tile_halos,
        )

        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, self._natten_trailing_pad_latent_frames
        )
        if self.mark_dynamic_shapes:
            for dim in (2, 3, 4):
                torch._dynamo.mark_dynamic(latent_padded, dim)

        feat_s4 = self.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)

        batch = latent.shape[0]
        timestep = self.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
        single_step_x0 = timestep.shape[1] == 1 and self.model_output_type == "x0"

        x_t_init: torch.Tensor | None = None
        if not single_step_x0:
            compute_dtype = latent.dtype
            randn_device = generator.device if generator is not None else latent.device
            x_t_init = torch.randn(
                tuple(target_shape), dtype=compute_dtype, generator=generator, device=randn_device
            ).to(latent.device)

        complementary = masks_are_complementary(tiles, target_shape)
        groups = group_tiles_by_temporal_slice(tiles)
        group_slices = [slice(*group[0].out_coords[2].indices(target_shape[2])[:2]) for group in groups]

        # Keep only the trailing temporal overlap of the previous group (not the full
        # chunk). Exclusive frames are yielded before the next group is decoded; the
        # consumer may still hold that emit while the next buffer is live (~2x tile).
        overlap_stub: torch.Tensor | None = None
        overlap_stub_weights: torch.Tensor | None = None

        def _finalize(buf: torch.Tensor, wts: torch.Tensor | None) -> torch.Tensor:
            if complementary:
                return buf.to(latent.dtype)
            assert wts is not None
            wts = wts.clamp(min=diffusion_tiling._weight_floor(wts.dtype))
            return (buf / wts).to(latent.dtype)

        def _narrow_content_cfhw(t: torch.Tensor, frames_keep: int) -> torch.Tensor:
            """Spatial/temporal content crop as views (no ``.contiguous()``)."""
            x = t[:, :, :frames_keep]
            th, tw = content_pixel.height, content_pixel.width
            scale_h, scale_w = spatial_scale
            if h_pad is not None:
                before = diffusion_tiling.scale_axis_pad(h_pad, scale_h).before
                x = x.narrow(3, before, th)
            else:
                need = x.shape[3] - th
                if need > 0:
                    x = x.narrow(3, need // 2, th)
                elif need < 0:
                    x, _ = diffusion_tiling.resize_axis(x, 3, th, mode="symmetric")
            if w_pad is not None:
                before = diffusion_tiling.scale_axis_pad(w_pad, scale_w).before
                x = x.narrow(4, before, tw)
            else:
                need = x.shape[4] - tw
                if need > 0:
                    x = x.narrow(4, need // 2, tw)
                elif need < 0:
                    x, _ = diffusion_tiling.resize_axis(x, 4, tw, mode="symmetric")
            return x

        def _crop_emit(buf: torch.Tensor, wts: torch.Tensor | None, global_start: int) -> torch.Tensor | None:
            if global_start >= content_pixel.frames or buf.shape[2] < 1:
                return None
            frames_keep = min(buf.shape[2], content_pixel.frames - global_start)
            if frames_keep < 1:
                return None
            if not as_fhwc:
                chunk = _finalize(buf[:, :, :frames_keep], None if wts is None else wts[:, :, :frames_keep])
                return diffusion_tiling.crop_pixels_to_content(
                    chunk,
                    frames_keep,
                    content_pixel.height,
                    content_pixel.width,
                    h_pad=h_pad,
                    w_pad=w_pad,
                    spatial_scale=spatial_scale,
                )

            # One materialize: contiguous FHWC in latent.dtype, still [-1, 1].
            # CFHW→FHWC cannot be inplace; range mapping is left to to_rgb.
            cfhw = _narrow_content_cfhw(buf, frames_keep)
            src = cfhw[0]  # C, F, H, W (view into accumulator)
            video = torch.empty(
                src.shape[1],
                src.shape[2],
                src.shape[3],
                src.shape[0],
                dtype=latent.dtype,
                device=src.device,
            )
            video.copy_(src.permute(1, 2, 3, 0))
            if not complementary:
                assert wts is not None
                w_cfhw = _narrow_content_cfhw(wts, frames_keep)
                wview = w_cfhw[0].permute(1, 2, 3, 0)
                # Inplace floor on exclusive weight region only (discarded after emit).
                wview.clamp_min_(diffusion_tiling._weight_floor(w_cfhw.dtype))
                video.div_(wview)
            return video

        for gi, group in enumerate(groups):
            curr_temporal_slice = group_slices[gi]
            buffer, weights = self._decode_temporal_group_isolated(
                group,
                feat_s4,
                s4_t,
                x_t_init,
                timestep,
                full_video_shape,
                curr_temporal_slice,
                generator=generator,
                complementary=complementary,
            )

            if overlap_stub is not None:
                overlap_len = int(overlap_stub.shape[2])
                if overlap_len > 0:
                    # Stub is exactly the region overlapping this group (cloned when
                    # the previous group finished); blend then write back into buffer.
                    overlap_stub += buffer[:, :, :overlap_len]
                    if complementary:
                        buffer[:, :, :overlap_len] = overlap_stub
                    else:
                        assert overlap_stub_weights is not None
                        assert weights is not None
                        overlap_stub_weights += weights[:, :, :overlap_len]
                        buffer[:, :, :overlap_len] = overlap_stub
                        weights[:, :, :overlap_len] = overlap_stub_weights
                overlap_stub = None
                overlap_stub_weights = None

            if gi + 1 < len(groups):
                next_start = group_slices[gi + 1].start
                exclusive_len = min(max(0, next_start - curr_temporal_slice.start), buffer.shape[2])
                emitted = _crop_emit(
                    buffer[:, :, :exclusive_len],
                    None if weights is None else weights[:, :, :exclusive_len],
                    curr_temporal_slice.start,
                )
                if emitted is not None:
                    yield emitted
                # Retain only the trailing overlap for the next handoff.
                overlap_stub = buffer[:, :, exclusive_len:].clone()
                if not complementary:
                    assert weights is not None
                    overlap_stub_weights = weights[:, :, exclusive_len:].clone()
                del buffer, weights
            else:
                emitted = _crop_emit(buffer, weights, curr_temporal_slice.start)
                if emitted is not None:
                    yield emitted

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode via ``_decode_pixels`` with ``tiling_config=None`` (single full tile)."""
        return next(self._decode_pixels(sample, tiling_config=None, generator=generator))

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Tiled decode: stages 1-3 once, stages 4-5 per tile, pixel blend."""
        yield from self._decode_pixels(latent, tiling_config, generator=generator)

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        keyframes: DecodeKeyframes | None = None,
        **_kwargs: object,
    ) -> Iterator[torch.Tensor]:
        """Decode latent video, yielding float chunk(s) ``[f, h, w, c]`` in ``[0, 1]``.
        Untiled and tiled both go through ``_decode_pixels``. Tiled decode may yield
        multiple times when ``tiling_config.frames`` splits the video.
        With ``keyframes`` this is :meth:`_decode_video_with_keyframes`; the argument exists on
        every decoder so a caller can pass planes without first asking which VAE it holds.
        Layout is packed once to contiguous FHWC on emit; ``to_rgb`` only does
        inplace ``[-1, 1]→[0, 1]`` (no second realloc).
        """
        if keyframes is not None:
            yield from self._decode_video_with_keyframes(latent, keyframes, tiling_config, generator)
            return

        def to_rgb(frames: torch.Tensor) -> torch.Tensor:
            return frames.add_(1).mul_(0.5).clamp_(0, 1)

        for chunk in self._decode_pixels(latent, tiling_config, generator=generator, as_fhwc=True):
            yield to_rgb(chunk)

    def decode_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode each latent as its own one-frame clip, yielding one RGB tensor per latent."""
        yield from iter_decoded_single_frames(self, latents, generator)
