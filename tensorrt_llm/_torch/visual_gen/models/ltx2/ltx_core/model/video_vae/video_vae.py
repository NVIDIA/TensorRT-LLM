import itertools
import logging
from collections.abc import Sequence
from typing import Any, Callable, Iterator, List, Protocol, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.common.normalization import PixelNorm
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.disposable import Disposable
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.attention import AttnBlock3D
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.convolution import make_conv_nd, memory_format_of
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import DecodeKeyframes
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.ops import PerChannelStatistics, patchify
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.resnet import ResnetBlock3D, UNetMidBlock3D
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.sampling import SpaceToDepthDownsample
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    Tile,
    TileCountConfig,
    TileSizeConfig,
    TilingConfig,
    _validate_overlap,
    compute_rectangular_mask_1d,
    compute_trapezoidal_mask_1d,
    create_tiles,
    masks_are_complementary,
    scale_by_masks_1d,
    untiled_mask_1d,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors, VideoLatentShape

logger: logging.Logger = logging.getLogger(__name__)


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
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
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn":
        block = AttnBlock3D(in_channels=in_channels, attention=attention)
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module, Disposable):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Encoder. Encodes video frames into a latent representation.
    The encoder compresses the input video through a series of downsampling operations controlled by
    patch_size and encoder_blocks. The output is a normalized latent tensor with shape (B, 128, F', H', W').
    Compression Behavior:
        The total compression is determined by:
        1. Initial spatial compression via patchify: H -> H/4, W -> W/4 (patch_size=4)
        2. Sequential compression through encoder_blocks based on their stride patterns
        Compression blocks apply 2x compression in specified dimensions:
            - "compress_time" / "compress_time_res": temporal only
            - "compress_space" / "compress_space_res": spatial only (H and W)
            - "compress_all" / "compress_all_res": all dimensions (F, H, W)
            - "res_x" / "res_x_y": no compression
        Standard LTX Video configuration:
            - patch_size=4
            - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
            - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
            - Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16)
            - Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels. For RGB images, this is 3.
        out_channels: The number of output channels (latent channels). For latent channels, this is 128.
        encoder_blocks: The list of blocks to construct the encoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: The patch size for initial spatial compression. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var: The log variance mode. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, int]] | List[Tuple[str, dict[str, Any]]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
        attention: AttentionFunction | AttentionCallable = AttentionFunction.PYTORCH,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS
        # Spatiotemporal downscaling derived from the block list (see SpatioTemporalScaleFactors.from_blocks).
        self.video_scale_factors = SpatioTemporalScaleFactors.from_blocks(encoder_blocks, patch_size)

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
                attention=attention,
            )

            self.down_blocks.append(block)

        # out
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        r"""
        Encode video frames into normalized latent representation.
        Args:
            sample: Input video (B, C, F, H, W). F should be 1 + 8*k (e.g., 1, 9, 17, 25, 33...).
                If not, the encoder crops the last frames to the nearest valid length.
                Should be normalized to [-1, 1] range before encoding.
        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16).
        """
        # Validate frame count (crop to nearest valid length if needed)
        temporal_factor = self.video_scale_factors.time
        frames_count = sample.shape[2]
        if ((frames_count - 1) % temporal_factor) != 0:
            frames_to_crop = (frames_count - 1) % temporal_factor
            logger.warning(
                "Invalid number of frames %s for encode; cropping last %s frames to satisfy 1 + %s*k.",
                frames_count,
                frames_to_crop,
                temporal_factor,
            )
            sample = sample[:, :, :-frames_to_crop, ...]

        # Initial spatial compression: trade spatial resolution for channel depth
        # This reduces H,W by patch_size and increases channels, making convolutions more efficient
        # Example: (B, 3, F, 512, 512) -> (B, 48, F, 128, 128) with patch_size=4
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        # Activations follow the weights' layout: channels-last weights (CHANNELS_LAST_3D_WEIGHTS)
        # keep the whole encoder on NHWC conv kernels.
        sample = sample.contiguous(memory_format=memory_format_of(self.conv_in.weight))
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Uniform Variance: model outputs N means and 1 shared log-variance channel.
            # We need to expand the single logvar to match the number of means channels
            # to create a format compatible with PER_CHANNEL (means + logvar, each with N channels).
            # Sample shape: (B, N+1, ...) where N = latent_channels (e.g., 128 means + 1 logvar = 129)
            # Target shape: (B, 2*N, ...) where first N are means, last N are logvar

            if sample.shape[1] < 2:
                raise ValueError(
                    f"Invalid channel count for UNIFORM mode: expected at least 2 channels "
                    f"(N means + 1 logvar), got {sample.shape[1]}"
                )

            # Extract means (first N channels) and logvar (last 1 channel)
            means = sample[:, :-1, ...]  # (B, N, ...)
            logvar = sample[:, -1:, ...]  # (B, 1, ...)

            # Repeat logvar N times to match means channels
            # Use expand/repeat pattern that works for both 4D and 5D tensors
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)  # (B, N, ...)

            # Concatenate to create (B, 2*N, ...) format: [means, repeated_logvar]
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        # Split into means and logvar, then normalize means
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means).contiguous()

    def tiled_encode(
        self,
        video: torch.Tensor,
        tiling_config: TilingConfig | None = None,
    ) -> torch.Tensor:
        """Encode video to latent using tiled processing of the given video tensor.
        Device Handling:
            - Input video can be on CPU or GPU
            - Accumulation buffers are created on model's device
            - Each tile is automatically moved to model's device before encoding
            - Output latent is returned on model's device
        Args:
            video: Input video tensor (B, 3, F, H, W) in range [-1, 1]
            tiling_config: Tiling configuration for the video tensor
        Returns:
            Latent tensor (B, 128, F', H', W') on model's device
            where F' = 1 + (F-1)/8, H' = H/32, W' = W/32
        """
        # Detect model device and dtype
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        # Extract shape components
        batch, _, frames, height, width = video.shape

        # Check frame count and crop if needed
        if (frames - 1) % self.video_scale_factors.time != 0:
            frames_to_crop = (frames - 1) % self.video_scale_factors.time
            logger.warning(
                f"Number of frames {frames} of input video is not ({self.video_scale_factors.time} * k + 1), "
                f"last {frames_to_crop} frames will be cropped"
            )
            video = video[:, :, :-frames_to_crop, ...]
            # Update frames after cropping
            frames = video.shape[2]

        # Calculate output latent shape (inverse of upscale)
        latent_shape = VideoLatentShape(
            batch=batch,
            channels=self.latent_channels,  # 128 for standard VAE
            frames=(frames - 1) // self.video_scale_factors.time + 1,
            height=height // self.video_scale_factors.height,
            width=width // self.video_scale_factors.width,
        )

        # Prepare tiles (operates on VIDEO dimensions)
        tiles = prepare_tiles_for_encoding(video, tiling_config, scale_factors=self.video_scale_factors)
        complementary = masks_are_complementary(tiles, latent_shape.to_torch_shape())

        # Initialize accumulation buffers on model device
        latent_buffer = torch.zeros(
            latent_shape.to_torch_shape(),
            device=model_device,
            dtype=model_dtype,
        )
        weights_buffer: torch.Tensor | None = None if complementary else torch.zeros_like(latent_buffer)

        # Process each tile
        for tile in tiles:
            # Extract video tile from input (may be on CPU)
            video_tile = video[tile.in_coords]

            # Move tile to model device if needed
            if video_tile.device != model_device or video_tile.dtype != model_dtype:
                video_tile = video_tile.to(device=model_device, dtype=model_dtype)

            # Encode tile to latent (output on model device)
            latent_tile = self.forward(video_tile)

            masks = tuple(m.to(device=model_device, dtype=torch.float32) for m in tile.masks_1d)
            latent_buffer[tile.out_coords] += scale_by_masks_1d(latent_tile, masks)
            if weights_buffer is not None:
                strength = torch.ones(latent_tile.shape, device=model_device, dtype=torch.float32)
                weights_buffer[tile.out_coords] += scale_by_masks_1d(strength, masks)

            del latent_tile, video_tile

        if weights_buffer is None:
            return latent_buffer
        weights_buffer = weights_buffer.clamp(min=1e-8)
        return latent_buffer / weights_buffer


def prepare_tiles_for_encoding(
    video: torch.Tensor,
    tiling_config: TilingConfig | None = None,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> List[Tile]:
    """Prepare tiles for VAE encoding.
    Splits on the latent grid via ``tiling_config.to_splitters``, maps each
    interval to pixel ``in_coords`` (encode-specific) and latent ``out_coords``,
    and builds :class:`~ltx_core.tiling.Tile` values. Size- and count-based
    configs share this path.
    Args:
        video: Input video tensor (B, 3, F, H, W) in range [-1, 1]
        tiling_config: Size- or count-based tiling configuration
    Returns:
        List of tiles for the video tensor
    """
    if tiling_config is None:
        return create_tiles(
            video.shape,
            [DEFAULT_SPLIT_OPERATION] * len(video.shape),
            [DEFAULT_MAPPING_OPERATION] * len(video.shape),
        )

    # Conv-VAE encode needs enough overlap to discard symmetric-pad edge artifacts.
    _validate_overlap(tiling_config, min_overlap_frames=16, min_overlap_pixels=64)
    _, _, frames, height, width = video.shape
    latent_frames = (frames - 1) // scale_factors.time + 1
    latent_height = height // scale_factors.height
    latent_width = width // scale_factors.width
    t_split, h_split, w_split = tiling_config.to_splitters(scale_factors, causal_temporal=True)
    t_intervals = t_split(latent_frames).intervals
    h_intervals = h_split(latent_height).intervals
    w_intervals = w_split(latent_width).intervals

    tiles: List[Tile] = []
    for t_iv, h_iv, w_iv in itertools.product(t_intervals, h_intervals, w_intervals):
        t_in, _ = map_temporal_slice(t_iv.start, t_iv.end, t_iv.left_ramp, t_iv.right_ramp, scale_factors.time)
        h_in, _ = map_spatial_slice(h_iv.start, h_iv.end, h_iv.left_ramp, h_iv.right_ramp, scale_factors.height)
        w_in, _ = map_spatial_slice(w_iv.start, w_iv.end, w_iv.left_ramp, w_iv.right_ramp, scale_factors.width)
        tiles.append(
            Tile(
                in_coords=(slice(None), slice(None), t_in, h_in, w_in),
                out_coords=(
                    slice(None),
                    slice(None),
                    slice(t_iv.start, t_iv.end),
                    slice(h_iv.start, h_iv.end),
                    slice(w_iv.start, w_iv.end),
                ),
                masks_1d=(
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                    compute_trapezoidal_mask_1d(t_iv.end - t_iv.start, t_iv.left_ramp, t_iv.right_ramp, True),
                    compute_trapezoidal_mask_1d(h_iv.end - h_iv.start, h_iv.left_ramp, h_iv.right_ramp, False),
                    compute_trapezoidal_mask_1d(w_iv.end - w_iv.start, w_iv.left_ramp, w_iv.right_ramp, False),
                ),
            )
        )
    return tiles


def clip_generators(
    count: int,
    generator: torch.Generator | Sequence[torch.Generator | None] | None,
) -> Sequence[torch.Generator | None]:
    """One generator per clip: broadcast a single generator (or ``None``), or check a sequence."""
    if generator is None:
        return (None,) * count
    if isinstance(generator, torch.Generator):
        return (generator,) * count
    if len(generator) != count:
        raise ValueError(f"decode_single_frames got {count} latents and {len(generator)} generators")
    return generator


def validate_single_frame_latents(latents: Sequence[torch.Tensor]) -> None:
    """Raise unless every entry is a ``(B, C, 1, H, W)`` single-frame latent.
    Checked over the whole list before any of it is decoded, so a decode that splits the list
    across ranks raises on every rank instead of only on the one that owns the bad plane -- a
    lone raiser leaves the rest of the group waiting on a collective that never comes.
    """
    for index, latent in enumerate(latents):
        if latent.ndim != 5 or latent.shape[2] != 1:
            raise ValueError(
                f"decode_single_frames expects (B, C, 1, H, W) latents, got {tuple(latent.shape)} at index {index}"
            )


def iter_decoded_single_frames(
    decoder: "VideoDecoder",
    latents: Sequence[torch.Tensor],
    generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
) -> Iterator[torch.Tensor]:
    """Decode each latent through ``decode_video`` as its own clip.
    Dist must pass the inner SGPU decoder, not itself: ``decode_video`` on Dist splits the
    volume and workers return an empty iterator.
    """
    gens = clip_generators(len(latents), generator)
    validate_single_frame_latents(latents)
    for index, (latent, gen) in enumerate(zip(latents, gens, strict=True)):
        chunks = list(decoder.decode_video(latent, tiling_config=None, generator=gen))
        if not chunks:
            raise RuntimeError(f"Decoder returned no pixels for single-frame latent {index}")
        yield torch.cat(chunks, dim=0)


class VideoDecoder(Protocol):
    """Structural interface for video VAE decoders.
    Implementations decode a latent tensor into pixel-space video chunks
    (e.g. ``ConvVideoDecoder``, ``DiffusionVideoDecoder``,
    ``DistributedVideoDecoder``).
    """

    video_downscale_factors: SpatioTemporalScaleFactors

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TileSizeConfig | TileCountConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        keyframes: "DecodeKeyframes | None" = None,
        **kwargs: object,
    ) -> Iterator[torch.Tensor]:
        """Decode a video latent tensor, yielding float chunks ``[f, h, w, c]`` in ``[0, 1]``.
        ``keyframes`` anchors the decode on already-encoded single-frame planes. It is a hint,
        not a contract: a decoder that cannot use it says so once and decodes plainly, so a
        caller never has to ask which decoder it got before handing keyframes over.
        """
        ...

    def decode_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode each latent as its own one-frame clip, yielding one RGB tensor per latent.
        A causal VAE cannot decode stacked independent planes without bleeding neighbours.
        Dist implementations split the list across their ranks and gather the pixels back, so
        every rank still receives every frame, in the order the latents were given.
        """
        ...


def get_video_chunks_number(
    num_frames: int,
    tiling_config: TileSizeConfig | TileCountConfig | None = None,
) -> int:
    """
    Get the number of video chunks for a given number of frames and tiling configuration.
    Delegates to :meth:`TileSizeConfig.video_chunks_number` /
    :meth:`TileCountConfig.video_chunks_number`.
    Args:
        num_frames: Number of frames in the video.
        tiling_config: Tiling configuration.
    Returns:
        Number of video chunks.
    """
    if tiling_config is None:
        return 1
    return tiling_config.video_chunks_number(num_frames)


def to_mapping_operation(
    map_func: Callable[[int, int, int, int, int], Tuple[slice, torch.Tensor]],
    scale: int,
) -> MappingOperation:
    """Create a mapping operation over a set of tiling intervals.
    The given mapping function is applied to each interval in the input dimension. The result function is used for
    creating tiles in the output dimension.
    Args:
        map_func: Mapping function to create the mapping operation from
        scale: Scale factor for the transformation, used as an argument for the mapping function
    Returns:
        Mapping operation that takes a set of tiling intervals and returns a set of slices and masks in the output
        dimension.
    """

    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor] = []
        for interval in intervals.intervals:
            output_slice, mask_1d = map_func(
                interval.start, interval.end, interval.left_ramp, interval.right_ramp, scale
            )
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp = 0 if left_ramp == 0 else 1 + (left_ramp - 1) * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, True)


def map_temporal_interval_to_latent(
    begin: int, end: int, left_ramp: int, right_ramp: int | None = None, scale: int = 1
) -> Tuple[slice, torch.Tensor]:
    """
    Map temporal interval in video frame space to latent space.
    Args:
        begin: Start position in video frame space
        end: End position in video frame space
        left_ramp: Left ramp size in video frame space
        right_ramp: Right ramp size in video frame space
        scale: Scale factor for transformation
    Returns:
        Tuple of (output_slice, blend_mask)
    """
    start = begin // scale
    stop = (end - 1) // scale + 1

    left_ramp_latents = 0 if left_ramp == 0 else 1 + (left_ramp - 1) // scale
    right_ramp_latents = right_ramp // scale

    if right_ramp_latents != 0:
        raise ValueError("For tiled encoding, temporal tiles are expected to have a right ramp equal to 0")

    mask_1d = compute_rectangular_mask_1d(stop - start, left_ramp_latents, right_ramp_latents)

    return slice(start, stop), mask_1d


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    left_ramp = left_ramp * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, False)


def map_spatial_interval_to_latent(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> Tuple[slice, torch.Tensor]:
    """Map spatial interval in pixel space to latent space.
       Args:
        begin: Start position in pixel space
        end: End position in pixel space
        left_ramp: Left ramp size in pixel space
        right_ramp: Right ramp size in pixel space
        scale: Scale factor for transformation
    Returns:
        Tuple of (output_slice, blend_mask)
    """
    start = begin // scale
    stop = end // scale
    left_ramp = max(0, left_ramp // scale - 1)

    right_ramp = 0 if right_ramp == 0 else 1

    mask_1d = compute_rectangular_mask_1d(stop - start, left_ramp, right_ramp)
    return slice(start, stop), mask_1d



__all__ = [
    "ConvVideoDecoder",
    "DiffusionVideoDecoder",
    "VideoDecoder",
    "VideoEncoder",
    "get_video_chunks_number",
    "map_spatial_interval_to_latent",
    "map_spatial_slice",
    "map_temporal_interval_to_latent",
    "map_temporal_slice",
    "prepare_tiles_for_encoding",
    "to_mapping_operation",
]

# Defer optional decoder dependencies until the requested decoder is used.
import importlib

_LAZY_EXPORTS = {'ConvVideoDecoder': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.conv_video_decoder', 'ConvVideoDecoder'), 'DiffusionVideoDecoder': ('tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.diffusion_video_decoder', 'DiffusionVideoDecoder')}

def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module, symbol = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module), symbol)
    globals()[name] = value
    return value
