"""Convolutional video VAE decoder."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Iterator, List, Tuple

import torch
from einops import rearrange
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.common.normalization import PixelNorm
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.disposable import Disposable
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.attention import AttnBlock3D
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.convolution import make_conv_nd
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.enums import NormLayerType, PaddingModeType
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import DecodeKeyframes
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.ops import PerChannelStatistics, unpatchify
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.sampling import DepthToSpaceUpsample
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.video_vae import (
    VideoDecoder,
    iter_decoded_single_frames,
    map_spatial_slice,
    map_temporal_slice,
    to_mapping_operation,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    Tile,
    TilingConfig,
    create_tiles,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    scale_by_masks_1d,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

logger: logging.Logger = logging.getLogger(__name__)


def _decoder_bottleneck_channels(base_channels: int, decoder_blocks: List[Tuple[str, int | dict[str, Any]]]) -> int:
    """Channel count at the decoder bottleneck (the latent end, before any upsampling).
    The decoder mirrors the encoder: ``conv_in`` widens the latents to the bottleneck,
    then each upsampling block divides the channels by its multiplier as it grows the
    spatial/temporal size, ending at ``base_channels``. So the bottleneck is
    ``base_channels`` times the product of those multipliers -- 8x for the 32x32x8 VAE,
    4x for the 16x16x4 VAE.
    """
    channel_multiplier = 1
    for block_name, block_params in decoder_blocks:
        config = {"num_layers": block_params} if isinstance(block_params, int) else block_params
        if block_name in ("compress_time", "compress_space", "compress_all"):
            channel_multiplier *= config.get("multiplier", 1)
        elif block_name == "res_x_y":
            channel_multiplier *= config.get("multiplier", 2)
    return base_channels * channel_multiplier


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
    attention: AttentionFunction | AttentionCallable,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels // block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn":
        block = AttnBlock3D(in_channels=in_channels, attention=attention)
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


class ConvVideoDecoder(nn.Module, Disposable, VideoDecoder):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Decoder. Decodes latent representation into video frames.
    The decoder upsamples latents through a series of upsampling operations (inverse of encoder).
    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard LTX Video configuration.
    Upsampling blocks expand dimensions by 2x in specified dimensions:
        - "compress_time": temporal only
        - "compress_space": spatial only (H and W)
        - "compress_all": all dimensions (F, H, W)
        - "res_x" / "res_x_y" / "attn_res_x": no upsampling
    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.
        First frame removed after temporal upsampling in both modes. Output shape unchanged.
        Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512) for both modes.
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels (latent channels). Default is 128.
        out_channels: The number of output channels. For RGB images, this is 3.
        decoder_blocks: The list of blocks to construct the decoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: Final spatial expansion factor. For standard LTX Video, use 4 for 4x spatial expansion:
                    H -> Hx4, W -> Wx4. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal: Whether to use causal convolutions. For standard LTX Video, use False for symmetric padding.
                When True, uses causal padding (past/current frames only).
        timestep_conditioning: Whether to condition the decoder on timestep for denoising.
    """

    def __init__(  # noqa: PLR0913
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        base_channels: int = 128,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.PYTORCH,
    ):
        super().__init__()

        self.video_downscale_factors = SpatioTemporalScaleFactors.from_blocks(decoder_blocks, patch_size)

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        feature_channels = _decoder_bottleneck_channels(base_channels, decoder_blocks)

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
                attention=attention,
            )

            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        r"""
        Decode latent representation into video frames.
        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True). Uses default 0.05 if None.
            generator: Random generator for deterministic noise injection (if inject_noise=True in blocks).
        Returns:
            Decoded video (B, 3, F, H, W) where F = 8x(F'-1) + 1, H = 32xH', W = 32xW'.
            Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512).
            Note: First frame is removed after temporal upsampling regardless of causal mode.
            When causal=False, allows future frame dependencies in convolutions but maintains same output shape.
        """
        batch_size = sample.shape[0]
        output_dtype = sample.dtype
        weights_dtype = next(self.parameters()).dtype
        sample = sample.to(weights_dtype)

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )

            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for up_block in self.up_blocks:
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Final spatial expansion: reverse the initial patchify from encoder
        # Moves pixels from channels back to spatial dimensions
        # Example: (B, 48, F, 128, 128) -> (B, 3, F, 512, 512) with patch_size=4
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample.to(output_dtype)

    def _prepare_tiles(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> List[Tile]:
        splitters = [DEFAULT_SPLIT_OPERATION] * len(latent.shape)
        mappers = [DEFAULT_MAPPING_OPERATION] * len(latent.shape)
        if tiling_config is None:
            return create_tiles(latent.shape, splitters, mappers)

        t_split, h_split, w_split = tiling_config.to_splitters(self.video_downscale_factors)
        splitters[2] = t_split
        splitters[3] = h_split
        splitters[4] = w_split

        if tiling_config.height.is_tiled():
            mappers[3] = to_mapping_operation(map_spatial_slice, scale=self.video_downscale_factors.height)
        if tiling_config.width.is_tiled():
            mappers[4] = to_mapping_operation(map_spatial_slice, scale=self.video_downscale_factors.width)
        if tiling_config.frames.is_tiled():
            mappers[2] = to_mapping_operation(map_temporal_slice, scale=self.video_downscale_factors.time)

        return create_tiles(latent.shape, splitters, mappers)

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """
        Decode a latent tensor into video frames using tiled processing.
        Splits the latent tensor into tiles, decodes each tile individually,
        and yields video chunks as they become available.
        Args:
            latent: Input latent tensor (B, C, F', H', W').
            tiling_config: Tiling configuration for the latent tensor.
            timestep: Optional timestep for decoder conditioning.
            generator: Optional random generator for deterministic decoding.
        Yields:
            Video chunks (B, C, T, H, W) by temporal slices;
        """

        # Calculate full video shape from latent shape to get spatial dimensions
        full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(self.video_downscale_factors)
        tiles = self._prepare_tiles(latent, tiling_config)
        complementary = masks_are_complementary(tiles, full_video_shape.to_torch_shape())

        temporal_groups = group_tiles_by_temporal_slice(tiles)

        # State for temporal overlap handling
        previous_chunk = None
        previous_weights = None
        previous_temporal_slice = None

        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = temporal_group_tiles[0].out_coords[2]

            # Calculate the shape of the temporal buffer for this group of tiles.
            # The temporal length depends on whether this is the first tile (starts at 0) or not.
            # - First tile: (frames - 1) * scale + 1
            # - Subsequent tiles: frames * scale
            # This logic is handled by TemporalAxisMapping and reflected in out_coords.
            temporal_tile_buffer_shape = full_video_shape._replace(
                frames=curr_temporal_slice.stop - curr_temporal_slice.start,
            )

            buffer = torch.zeros(
                temporal_tile_buffer_shape.to_torch_shape(),
                device=latent.device,
                dtype=latent.dtype,
            )

            curr_weights = self._accumulate_temporal_group_into_buffer(
                group_tiles=temporal_group_tiles,
                buffer=buffer,
                latent=latent,
                timestep=timestep,
                generator=generator,
                complementary=complementary,
            )

            # Blend with previous temporal chunk if it exists
            if previous_chunk is not None:
                # Check if current temporal slice overlaps with previous temporal slice
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(curr_temporal_slice.start - previous_temporal_slice.start, None)

                    # The overlap is already masked before it reaches this step. Each tile is accumulated into buffer
                    # with its trapezoidal mask, and curr_weights accumulates the same mask. In the overlap blend we add
                    # the masked values (buffer[...]) and the corresponding weights (curr_weights[...]) into the
                    # previous buffers, then later normalize by weights.
                    previous_chunk[:, :, temporal_overlap_slice, :, :] += buffer[:, :, slice(0, overlap_len), :, :]
                    if not complementary:
                        previous_weights[:, :, temporal_overlap_slice, :, :] += curr_weights[
                            :, :, slice(0, overlap_len), :, :
                        ]
                        buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[:, :, temporal_overlap_slice, :, :]
                        curr_weights[:, :, slice(0, overlap_len), :, :] = previous_weights[
                            :, :, temporal_overlap_slice, :, :
                        ]
                    else:
                        buffer[:, :, slice(0, overlap_len), :, :] = previous_chunk[:, :, temporal_overlap_slice, :, :]

                # Yield the non-overlapping part of the previous chunk
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                if complementary:
                    yield previous_chunk[:, :, :yield_len, :, :]
                else:
                    previous_weights = previous_weights.clamp(min=1e-8)
                    yield (previous_chunk / previous_weights)[:, :, :yield_len, :, :]

            # Update state for next iteration
            previous_chunk = buffer
            previous_weights = curr_weights
            previous_temporal_slice = curr_temporal_slice

        # Yield any remaining chunk
        if previous_chunk is not None:
            if complementary:
                yield previous_chunk
            else:
                previous_weights = previous_weights.clamp(min=1e-8)
                yield previous_chunk / previous_weights

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        keyframes: DecodeKeyframes | None = None,
        **_kwargs: object,
    ) -> Iterator[torch.Tensor]:
        """Decode a video latent tensor, yielding float chunks ``[f, h, w, c]`` in ``[0, 1]``.
        Subclasses (e.g. ``DistributedVideoDecoder``) may override this to
        control eagerness or distribution across ranks.
        ``keyframes`` is accepted for signature parity with the diffusion decoder and ignored:
        anchoring on keyframe planes is a diffusion-decode feature. Ignoring is deliberate --
        raising would mean every caller has to branch on which VAE it was handed -- but it is
        logged, because a silent drop looks exactly like a keyframe decode that did nothing.
        """
        if keyframes is not None:
            logger.warning(
                "Conv video decoder ignores the %d keyframe plane(s) it was given; keyframe-anchored "
                "decoding needs a diffusion VAE. Decoding without them.",
                keyframes.num_planes,
            )

        def to_rgb(frames: torch.Tensor) -> torch.Tensor:
            video = rearrange(frames[0], "c f h w -> f h w c")
            return video.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)

        if tiling_config is not None:
            for frames in self.tiled_decode(latent, tiling_config, generator=generator):
                yield to_rgb(frames)
        else:
            decoded = self(latent, generator=generator)
            yield to_rgb(decoded)

    def decode_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode each latent as its own one-frame clip, yielding one RGB tensor per latent."""
        yield from iter_decoded_single_frames(self, latents, generator)

    def _accumulate_temporal_group_into_buffer(
        self,
        group_tiles: List[Tile],
        buffer: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor | None,
        generator: torch.Generator | None,
        *,
        complementary: bool,
    ) -> torch.Tensor | None:
        """
        Decode and accumulate all tiles of a temporal group into a local buffer.
        The buffer is local to the group and always starts at time 0; temporal coordinates
        are rebased by subtracting temporal_slice.start.
        Returns the weights buffer, or ``None`` when ``complementary`` (no divisor needed).
        """
        temporal_slice = group_tiles[0].out_coords[2]

        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)

        for tile in group_tiles:
            decoded_tile = self.forward(latent[tile.in_coords], timestep, generator)
            temporal_offset = tile.out_coords[2].start - temporal_slice.start
            # Use the tile's output coordinate length, not the decoded tile's length,
            # as the decoder may produce a different number of frames than expected
            expected_temporal_len = tile.out_coords[2].stop - tile.out_coords[2].start
            decoded_temporal_len = decoded_tile.shape[2]

            # Ensure we don't exceed the buffer or decoded tile bounds
            actual_temporal_len = min(expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset)

            chunk_coords = (
                slice(None),  # batch
                slice(None),  # channels
                slice(temporal_offset, temporal_offset + actual_temporal_len),
                tile.out_coords[3],  # height
                tile.out_coords[4],  # width
            )

            decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
            masks = tuple(m.to(device=buffer.device, dtype=torch.float32) for m in tile.masks_1d)
            # Slice the temporal mask to the frames we actually write (length-1 untiled stays length-1).
            masks = (*masks[:2], masks[2][:actual_temporal_len], *masks[3:])

            buffer[chunk_coords] += scale_by_masks_1d(decoded_slice, masks)
            if weights is not None:
                strength = torch.ones(decoded_slice.shape, device=buffer.device, dtype=torch.float32)
                weights[chunk_coords] += scale_by_masks_1d(strength, masks)

        return weights
