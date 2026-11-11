"""DiffVAE tiling helpers: schedule, pad/crop/size-floor, blend utilities.
Decode orchestration lives on ``DiffusionVideoDecoder``. This module owns the
geometry/schedule/mask pieces that tiling uses.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.color.yuv import frames_per_yuv_gemm
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.apply import resolve_attention_for_host
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.config import DiffVAEMode, NAttentionKind
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.tiling import (
    DEFAULT_SPLIT_OPERATION,
    DimensionInterval,
    DimensionSizeConfig,
    SplitOperation,
    Tile,
    TileSizeConfig,
    TilingConfig,
    _validate_overlap,
    compute_trapezoidal_mask_1d,
    split_by_size,
    untiled_mask_1d,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors

logger = logging.getLogger(__name__)

ResizeAxisMode = Literal["repeat_last", "symmetric"]

_GIB: int = 1 << 30

# Peak-activation heuristic (bytes):
#   hard (resident for whole tiled decode):
#     stage-4 input feature (stages 1-3 output): s4_txs4_hxs4_wxstage4_channelsxdtype.itemsize
#   per temporal group + stage-5 tile:
#     accumulator: H x W x (2 x tile_t) x out_channels x accum_dtype.itemsize
#       (full-frame spatial; ~current group buffer + retained overlap stub / still-
#        live emitted exclusive chunk heading to encode; RGB by default; accum_dtype
#        matches decode: fp16 when features are bf16, else feature dtype)
#     stage-5: stage5_tokens x stage5_channels x dtype.itemsize x coef
# where stage5_tokens = F x (H/patch) x (W/patch) and coef folds NA working-set multiplicity.


@dataclass(frozen=True, slots=True)
class _StageFiveBudget:
    """One mode's stage-5 multiplicity and withheld reserve, with and without keyframes."""

    coef: float
    coef_keyframes: float
    reserve_bytes: int


# Keyframe decode is *not* compiled: ``compile_diffusion_decoder`` wraps ``forward_attn_mlp`` /
# ``forward_combined``, while the keyframe pathway runs ``*_with_keyframes``. So with keyframes the
# diff blocks are eager whatever the mode asked for, and the budget follows the block shape rather
# than the compile setting:
#   - chunked: eager already, so keyframes change nothing (5) and CHUNKED_COMPILE's 7 -- which buys
#     its extra headroom from Inductor fusing the attn+mlp region -- falls back to 5.
#   - combined: keeps its combined ``context_and_x`` buffer and full-volume attention, but without
#     the fusion it holds several live full-volume intermediates instead of one Inductor buffer,
#     measured as 11 -> 15.
#   - BLACKWELL_DSL: the CuTe kernel has its own joint entry point, so the fused path serves
#     keyframes too, but the keyframe decode's eager det blocks and its keyframe planes raise the
#     measured working set to 4.0 (2.5 leaves a streamed multi-GPU decode out of memory warm).
_BUDGET_BY_MODE: dict[DiffVAEMode, _StageFiveBudget] = {
    DiffVAEMode.COMBINED_COMPILE: _StageFiveBudget(coef=11, coef_keyframes=15, reserve_bytes=2 * _GIB),
    DiffVAEMode.CHUNKED_COMPILE: _StageFiveBudget(coef=7, coef_keyframes=5, reserve_bytes=2 * _GIB),
    DiffVAEMode.CHUNKED_EAGER: _StageFiveBudget(coef=5, coef_keyframes=5, reserve_bytes=1 * _GIB),
    DiffVAEMode.BLACKWELL_DSL: _StageFiveBudget(coef=2.5, coef_keyframes=4.0, reserve_bytes=2 * _GIB),
}
_ACCUMULATOR_CHANNELS: int = 3  # RGB pixel blend buffer (decoder out_channels)
_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset({torch.bfloat16, torch.float16, torch.float32})
_MIN_MODEL_BYTES_FLOOR: dict[torch.dtype, int] = {
    torch.bfloat16: 1 * _GIB,
    torch.float16: 1 * _GIB,
    torch.float32: 2 * _GIB,
}
_BUDGET_SAFETY_BYTES_EAGER: int = 1 * _GIB
#: Keyframe decode runs eager blocks, so it takes the eager reserve -- unless the joint attention
#: ends up on torch's MATH SDPA kernel, which materializes its score block and wants one more GiB.
_BUDGET_SAFETY_BYTES_JOINT_MATERIALIZED: int = 2 * _GIB


def _falls_back_to_eager_na(mode: DiffVAEMode) -> bool:
    """True when this host has no natten and the mode's NA remaps to Triton/eager.
    For chunked modes that remap also switches ``compile_blocks`` off, so the peak is the eager
    one; :func:`resolve_attention_for_host` is the single owner of that decision.
    """
    resolved = resolve_attention_for_host(mode.resolve())
    return resolved.attention in (NAttentionKind.TRITON, NAttentionKind.EAGER_SDPA) and not resolved.compile_blocks


def stage5_mem_coef(mode: DiffVAEMode, *, keyframes: bool = False) -> float:
    """Stage-5 working-set multiplicity for auto tiling, after host NA resolve.
    Args:
        mode: the decode preset.
        keyframes: whether this decode carries a keyframe stream, which runs eager blocks.
    """
    try:
        budget = _BUDGET_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported DiffVAEMode for tiling budget: {mode!r}") from exc
    if keyframes:
        return budget.coef_keyframes
    if _falls_back_to_eager_na(mode):
        return _BUDGET_BY_MODE[DiffVAEMode.CHUNKED_EAGER].coef
    return budget.coef


# Peak of the SDR encode sink, per emitted pixel, on top of the chunk itself (which the
# accumulator term already covers). Measured at 11.50 B/px for bf16 RGB at 3840x2176.
# Two peaks compete, and which one wins depends on whether ``rgb_to_yuv`` converts the
# chunk in one GEMM or in write-back chunks (see ``color.yuv.frames_per_yuv_gemm``):
#   one GEMM (chunk <= limit)   a full new YUV tensor, kept alive afterwards by the
#                               ``y`` view into it
#   write-back (chunk > limit)  YUV overwrites the RGB storage the accumulator already
#                               charges, so only a GEMM-sized temporary is new
# On top of that, both paths allocate ``uv_full`` (2 ch, freed before the pack), the
# pooled 4:2:0 ``uv`` (0.5 ch), and then in ``pack_i420`` a concatenated float plane
# (1.5 ch) plus its uint8 copy (1.5 B/px, which does not scale with dtype).
# Halves are carried doubled so the arithmetic stays integral.
_CONVERT_PEAK_UV_CHANNELS_X2 = 5  # uv_full (2) + pooled uv (0.5)
_PACK_PEAK_UV_CHANNELS_X2 = 4  # pooled uv (0.5) + packed float (1.5)
_PACK_PEAK_UINT8_BYTES_X2 = 3  # 1.5 B/px


def max_emitted_frames(*, num_frames: int, tile_frames: int, overlap_frames: int) -> int:
    """Longest chunk the tiled decode yields, in pixel frames.
    ``_decode_groups_with_keyframes`` yields only a group's *exclusive* span -- one
    stride -- and keeps the trailing overlap as a stub for the next group. Only the
    final group yields its whole buffer. Charging ``tile_frames`` for every chunk
    therefore roughly doubles the estimate on long clips, which blocks layouts that
    would have fit. ``+1`` covers the causal shift, which moves each group after the
    first back by one frame.
    """
    if tile_frames >= num_frames:
        return num_frames
    stride = tile_frames - overlap_frames
    if stride <= 0:
        return tile_frames
    n_tiles = 1 + -(-(num_frames - tile_frames) // stride)
    last_group = num_frames - (n_tiles - 1) * stride
    return min(tile_frames, max(stride, last_group + 1))


def emit_convert_bytes(
    *,
    tile_frames: int,
    height: int,
    width: int,
    out_channels: int,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Downstream bytes a yielded chunk costs while the encoder consumes it.
    The decode budget alone is not enough to size a tile: whatever ``_emit`` yields is
    handed straight to the video encoder, and that peak overlaps the decode because the
    decoder is a generator -- it stays suspended holding stage-4 features and the
    accumulator while the consumer converts the chunk it just yielded. A layout that
    decodes comfortably can therefore still die in the encoder, and because the
    recommender spends spare VRAM on *larger* temporal tiles, more free memory used to
    make that failure more likely rather than less.
    ``out_channels`` is the width of the intermediate **YUV** tensor, not a second copy
    of the emitted RGB: the chunk itself is charged by the accumulator term. On the
    write-back path YUV lands in that same RGB storage, so it is not charged here at all
    and only the GEMM temporary is.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype for emit_convert_bytes: {dtype!r}")
    element_size = dtype.itemsize
    frames = int(tile_frames)
    gemm_frames = min(frames, frames_per_yuv_gemm(height, width))
    # Write-back reuses the RGB storage, so no full YUV tensor survives into the pack.
    resident_yuv_frames = frames if gemm_frames == frames else 0
    yuv_channels_x2 = 2 * int(out_channels)

    convert_peak_x2 = element_size * (yuv_channels_x2 * gemm_frames + _CONVERT_PEAK_UV_CHANNELS_X2 * frames)
    pack_peak_x2 = (
        element_size * (yuv_channels_x2 * resident_yuv_frames + _PACK_PEAK_UV_CHANNELS_X2 * frames)
        + _PACK_PEAK_UINT8_BYTES_X2 * frames
    )

    return int(height) * int(width) * max(convert_peak_x2, pack_peak_x2) // 2


def budget_safety_bytes(
    mode: DiffVAEMode,
    *,
    keyframes: bool = False,
    joint_sdpa_materializes: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Extra bytes withheld from the recommend budget, scaled by ``dtype``.
    Args:
        mode: the decode preset.
        keyframes: whether this decode carries a keyframe stream.
        joint_sdpa_materializes: whether the joint attention will run on torch's MATH SDPA kernel
            (see ``fallback_na.joint_eager.sdpa_materializes_scores``). Ignored without keyframes,
            and irrelevant when a fused joint kernel serves the decode.
        dtype: activation or accumulator dtype (defaults to ``torch.bfloat16``).
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype for tiling budget: {dtype!r}")
    try:
        budget = _BUDGET_BY_MODE[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported DiffVAEMode for tiling budget: {mode!r}") from exc
    if keyframes and mode is not DiffVAEMode.BLACKWELL_DSL:
        base_reserve = (
            _BUDGET_SAFETY_BYTES_JOINT_MATERIALIZED if joint_sdpa_materializes else _BUDGET_SAFETY_BYTES_EAGER
        )
    elif _falls_back_to_eager_na(mode):
        base_reserve = _BUDGET_SAFETY_BYTES_EAGER
    else:
        base_reserve = budget.reserve_bytes
    return (base_reserve * dtype.itemsize) // torch.float16.itemsize


def accumulator_dtype_for(feature_dtype: torch.dtype) -> torch.dtype:
    """Accumulator dtype for tiled pixel blend; mirrors ``_decode_temporal_group_isolated``.
    Returns ``torch.float16`` for ``torch.bfloat16`` features and passes through
    ``torch.float16`` and ``torch.float32``.
    """
    if feature_dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported feature_dtype for accumulator: {feature_dtype!r}")
    if feature_dtype is torch.bfloat16:
        return torch.float16
    return feature_dtype


def stage4_feature_bytes(
    *,
    height: int,
    width: int,
    num_frames: int,
    upsample_strides: Sequence[Tuple[int, int, int]],
    stage4_channels: int,
    dtype: torch.dtype = torch.bfloat16,
    natten_trailing_pad_latent_frames: int = 0,
) -> int:
    """Resident stages-1-3 output size (full volume tiled into stage 4).
    Matches ``DiffusionVideoDecoder.forward_stages_1_to_3`` after optional NATTEN
    trailing latent pad: channels-last ``(B, T, H, W, C)`` at stage-4 input resolution.
    """
    if stage4_channels < 1:
        raise ValueError(f"stage4_channels must be >= 1, got {stage4_channels}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype for stage4_feature_bytes: {dtype!r}")
    if len(upsample_strides) < 3:
        raise ValueError(f"need at least 3 upsample strides, got {len(upsample_strides)}")
    if natten_trailing_pad_latent_frames < 0:
        raise ValueError(f"natten_trailing_pad_latent_frames must be >= 0, got {natten_trailing_pad_latent_frames}")

    # Local import: types ↔ tiling cycle avoidance at module import time.
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.types import VIDEO_SCALE_FACTORS, VideoLatentShape, VideoPixelShape  # noqa: PLC0415

    latent = VideoLatentShape.from_pixel_shape(
        VideoPixelShape(batch=1, frames=int(num_frames), height=int(height), width=int(width), fps=24.0),
        scale_factors=VIDEO_SCALE_FACTORS,
    )
    s4_t, s4_h, s4_w = stage4_thw_from_latent(
        upsample_strides[:3],
        latent.frames + int(natten_trailing_pad_latent_frames),
        latent.height,
        latent.width,
        drop_leading_frame=True,
    )
    return int(s4_t) * int(s4_h) * int(s4_w) * int(stage4_channels) * dtype.itemsize


# ---------------------------------------------------------------------------
# Public entry points (pipeline recommend / decode schedule)
# ---------------------------------------------------------------------------


def recommended_decode_tiling_config(  # noqa: PLR0913
    *,
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
    min_tile_size_s4: Tuple[int, int, int],
    patch_size: int,
    height: int,
    width: int,
    num_frames: int,
    mode: DiffVAEMode,
    free_bytes: int,
    stage5_channels: int,
    stage4_channels: int,
    upsample_strides: Sequence[Tuple[int, int, int]],
    model_bytes: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    natten_trailing_pad_latent_frames: int = 0,
    out_channels: int = _ACCUMULATOR_CHANNELS,
    keyframes: bool = False,
    joint_sdpa_materializes: bool = False,
) -> TileSizeConfig:
    """Pick DiffVAE decode tiling from stage-4/5 halos and free VRAM.
    Always enables both spatial and temporal tiling (temporal-only full-frame slabs
    are unsafe on Hopper / some natten builds).
    Selection (size-grid, accumulator-aware):
      1. Enumerate legal tile **sizes** on the LCM of DiffVAE ``pixel_scale`` and
         :data:`~ltx_core.types.VIDEO_SCALE_FACTORS` (so configs also pass
         :class:`~ltx_core.tiling.TileSizeConfig` construction); derive tile
         counts from :func:`~ltx_core.tiling.split_by_size` (same as decode).
      2. Drop triples whose peak-bytes estimate exceeds ``usable`` bytes
         (``free - max(model, _MIN_MODEL_BYTES_FLOOR[dtype]) - safety - stage4_feature``;
         base safety is 1 GiB eager / 2 GiB compiled at 2-byte width and scales with
         ``dtype``, see :func:`budget_safety_bytes`).
         Stage-4 input features stay resident for the whole tiled decode.
      3. Among feasible triples, pick minimal :func:`volumetric_overlap_waste`.
    Peak-bytes estimate::
        stage4_feature_bytes(...)                         # hard, full volume
        + H * W * (2 * tile_t) * out_channels * accum_dtype.itemsize
        + stage5_tokens * stage5_channels * dtype.itemsize * coef
    Accumulator is full output HxW (not spatially tiled) with temporal extent
    ``2 * tile_t``: current group buffer plus the still-live previous exclusive
    emit / overlap stub during handoff (not merely ``tile_t + overlap_t``).
    RGBx``accum_dtype.itemsize`` by default: production bf16 features use fp16
    accumulators (2 bytes), matching :func:`accumulator_dtype_for`. Stage-5 uses
    ``dtype.itemsize`` x ``stage5_channels`` x ``coef``, which :func:`stage5_mem_coef`
    reads off the mode and ``keyframes`` (11 / 7 / 5 / 2.5 by mode; 15 / 5 / 5 / 4.0
    with a keyframe stream, which runs eager blocks).
    Args beyond the geometry:
        keyframes: this decode carries a keyframe stream (joint attention, eager blocks).
        joint_sdpa_materializes: the joint attention will run on torch's MATH SDPA kernel;
            costs one more GiB of reserve. See :func:`budget_safety_bytes`.
    """
    if height < 1 or width < 1 or num_frames < 1:
        raise ValueError(f"height/width/num_frames must be >= 1, got {height}x{width}x{num_frames}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if stage5_channels < 1:
        raise ValueError(f"stage5_channels must be >= 1, got {stage5_channels}")
    if out_channels < 1:
        raise ValueError(f"out_channels must be >= 1, got {out_channels}")
    accum_dtype = accumulator_dtype_for(dtype)

    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)

    ft, fh, fw = pixel_scale.time, pixel_scale.height, pixel_scale.width
    # Construction validates fixed 8/32/32; to_splitters uses pixel_scale - step both.
    step_t = math.lcm(ft, VIDEO_SCALE_FACTORS.time)
    step_h = math.lcm(fh, VIDEO_SCALE_FACTORS.height)
    step_w = math.lcm(fw, VIDEO_SCALE_FACTORS.width)
    min_t_px = _round_up(
        # ``2 * overlap`` so left+right ramps fit (else masks are not complementary and
        # decode allocates a full weights buffer ≈ another accumulator).
        max(2 * ft, 2 * overlap_t, _round_up(min_tile_size_s4[0] * ft, ft), 16),
        step_t,
    )
    min_h_px = _round_up(
        max(2 * fh, 2 * overlap_hw, _round_up(min_tile_size_s4[1] * fh, fh), 64),
        step_h,
    )
    min_w_px = _round_up(
        max(2 * fw, 2 * overlap_hw, _round_up(min_tile_size_s4[2] * fw, fw), 64),
        step_w,
    )

    model_cost = max(int(model_bytes), _MIN_MODEL_BYTES_FLOOR[dtype])
    coef = stage5_mem_coef(mode, keyframes=keyframes)
    s4_feat_bytes = stage4_feature_bytes(
        height=height,
        width=width,
        num_frames=num_frames,
        upsample_strides=upsample_strides,
        stage4_channels=stage4_channels,
        dtype=dtype,
        natten_trailing_pad_latent_frames=natten_trailing_pad_latent_frames,
    )
    reserve = budget_safety_bytes(
        mode,
        keyframes=keyframes,
        joint_sdpa_materializes=joint_sdpa_materializes,
        dtype=accum_dtype,
    )
    usable = max(0, int(free_bytes) - model_cost - reserve - s4_feat_bytes)
    s5_bytes_per_token = max(1.0, float(stage5_channels) * float(dtype.itemsize) * coef)
    acc_bytes_per_pixel = int(out_channels) * int(accum_dtype.itemsize)

    t_cands = _axis_candidates(num_frames, overlap_t, min_t_px, step_t)
    h_cands = _axis_candidates(height, overlap_hw, min_h_px, step_h)
    w_cands = _axis_candidates(width, overlap_hw, min_w_px, step_w)

    scored: list[tuple[float, int, int, int, int, int]] = []
    # (waste, -volume, n_t*n_h*n_w, tile_t, tile_h, tile_w) - minimize waste, then launches.
    for tile_t, n_t in t_cands:
        # Current group buffer + still-live emit/stub during temporal handoff.
        acc_frames = 2 * int(tile_t)
        acc_bytes = acc_frames * int(height) * int(width) * acc_bytes_per_pixel
        # The consumer converts each yielded chunk while this decode is suspended.
        downstream_bytes = emit_convert_bytes(
            tile_frames=max_emitted_frames(num_frames=num_frames, tile_frames=tile_t, overlap_frames=overlap_t),
            height=height,
            width=width,
            out_channels=out_channels,
            dtype=accum_dtype,
        )
        if acc_bytes + downstream_bytes >= usable:
            continue
        s5_budget_bytes = usable - acc_bytes - downstream_bytes
        max_s5_tokens = int(s5_budget_bytes // s5_bytes_per_token)
        for tile_h, n_h in h_cands:
            for tile_w, n_w in w_cands:
                if stage5_tokens_for_pixel_tile(tile_t, tile_h, tile_w, patch_size=patch_size) > max_s5_tokens:
                    continue
                waste = volumetric_overlap_waste(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    tile_frames=tile_t,
                    tile_height=tile_h,
                    tile_width=tile_w,
                    n_t=n_t,
                    n_h=n_h,
                    n_w=n_w,
                )
                scored.append((waste, -tile_t * tile_h * tile_w, n_t * n_h * n_w, tile_t, tile_h, tile_w))

    if not scored:
        raise ValueError(
            "Cannot fit a DiffVAE decode tile under the memory budget: "
            f"min tile ~{min_t_px}f x {min_h_px}x{min_w_px}px "
            f"(overlaps T={overlap_t}, HW={overlap_hw}), "
            f"mode={mode.value}, keyframes={keyframes}, coef={coef}, stage5_channels={stage5_channels}, "
            f"stage4_feature_bytes={s4_feat_bytes}, usable_bytes={usable}. "
            "Reduce resolution, reduce num_frames (stage-4 features and the per-chunk "
            "encode buffers both scale with it), or free GPU memory."
        )

    scored.sort()
    _waste, _vol, _ntiles, tile_t, tile_h, tile_w = scored[0]
    logger.info(
        "DiffVAE AUTO tiling: %df x %dx%d (overlap T=%d HW=%d) mode=%s keyframes=%s free_bytes=%d",
        tile_t,
        tile_h,
        tile_w,
        overlap_t,
        overlap_hw,
        mode.value,
        keyframes,
        int(free_bytes),
    )
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=tile_t, overlap=overlap_t),
        height=DimensionSizeConfig(tile_size=tile_h, overlap=overlap_hw),
        width=DimensionSizeConfig(tile_size=tile_w, overlap=overlap_hw),
    )


def prepare_tile_schedule(
    stage4_shape_bcthw: torch.Size,
    tiling_config: TilingConfig | None,
    *,
    upsample3_stride: Tuple[int, int, int],
    patch_size: int,
    min_tile_size: Tuple[int, int, int],
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
) -> List[Tile]:
    """Build pixel-blend tiles whose ``in_coords`` land on the stage-4 input grid.
    DiffVAE temporal tiling deliberately skips ConvVAE causal split/mask tricks
    (``split_temporal_causal``, ``left_starts_from_0``): pixel overlap already
    covers blend+halo, and interval propagation follows
    :class:`~ltx_core.model.video_vae.transformer.layers.LinearPixelShuffleUpsample`
    (``drop_leading_frame`` only on the origin tile) with *symmetric* trapezoid
    ramps so masks stay complementary without a weight buffer.
    """
    pixel_scale = stage4_to_pixel_scale_factors(upsample3_stride, patch_size)
    if tiling_config is None:
        return [
            Tile(
                in_coords=(slice(None), slice(None), slice(None), slice(None), slice(None)),
                out_coords=(slice(None), slice(None), slice(None), slice(None), slice(None)),
                masks_1d=(
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                    untiled_mask_1d(),
                ),
            )
        ]

    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)
    _validate_overlap(tiling_config, min_overlap_frames=overlap_t, min_overlap_pixels=overlap_hw)
    # Plain split (not split_temporal_causal): no start-1 / left_ramp+1 copycat of ConvVAE.
    t_split, h_split, w_split = tiling_config.to_splitters(
        pixel_scale, min_tile_size=min_tile_size, causal_temporal=False
    )
    st, sh, sw = upsample3_stride

    def axis_specs(
        split_op: SplitOperation,
        dim_len: int,
        stride_component: int,
        *,
        propagate_causal: bool,
        apply_patch: bool,
    ) -> list[tuple[slice, slice, torch.Tensor]]:
        if split_op is DEFAULT_SPLIT_OPERATION:
            return [(slice(None), slice(None), untiled_mask_1d())]
        intervals = split_op(dim_len).intervals
        specs = []
        for iv in intervals:
            stage5 = _propagate_interval_through_upsample_hops(iv, [stride_component], propagate_causal)
            if apply_patch:
                pixel = _propagate_interval_through_upsample_hops(stage5, [patch_size], causal=False)
            else:
                pixel = stage5
            # Symmetric ramps (left_starts_from_0=False) for partition-of-unity with
            # pixel-shuffle out_coords; ConvVAE sacrificial first-sample is not used.
            mask_pixel = compute_trapezoidal_mask_1d(
                pixel.end - pixel.start, pixel.left_ramp, pixel.right_ramp, left_starts_from_0=False
            )
            specs.append((slice(iv.start, iv.end), slice(pixel.start, pixel.end), mask_pixel))
        return specs

    # Temporal: pixel-shuffle propagate (drop-leading geometry); spatial: exact x stride.
    t_specs = axis_specs(t_split, stage4_shape_bcthw[2], st, propagate_causal=True, apply_patch=False)
    h_specs = axis_specs(h_split, stage4_shape_bcthw[3], sh, propagate_causal=False, apply_patch=True)
    w_specs = axis_specs(w_split, stage4_shape_bcthw[4], sw, propagate_causal=False, apply_patch=True)

    tiles: List[Tile] = []
    for t_spec, h_spec, w_spec in itertools.product(t_specs, h_specs, w_specs):
        t_s4, t_px, t_mask = t_spec
        h_s4, h_px, h_mask = h_spec
        w_s4, w_px, w_mask = w_spec
        tiles.append(
            Tile(
                in_coords=(slice(None), t_s4, h_s4, w_s4, slice(None)),
                out_coords=(slice(None), slice(None), t_px, h_px, w_px),
                masks_1d=(untiled_mask_1d(), untiled_mask_1d(), t_mask, h_mask, w_mask),
            )
        )
    return tiles


def slice_stage4_tile(
    feat_s4: torch.Tensor,
    tile: Tile,
    *,
    content_frames: int,
) -> tuple[torch.Tensor, bool, bool, tuple[int, int, int]]:
    """Slice a stage-4 feature tile, extending trailing tiles to include ghost frames."""
    is_origin = tile.in_coords[1].start in (0, None)
    _, stop, _ = tile.in_coords[1].indices(content_frames)
    pad_trailing = stop == content_frames
    _b, t_coord, h_coord, w_coord, _c = tile.in_coords
    t0, t1, _ = t_coord.indices(content_frames)
    h0, h1, _ = h_coord.indices(feat_s4.shape[2])
    w0, w1, _ = w_coord.indices(feat_s4.shape[3])
    content_thw = (t1 - t0, h1 - h0, w1 - w0)
    if pad_trailing:
        t1 = feat_s4.shape[1]
    feat_tile = feat_s4[:, t0:t1, h_coord, w_coord, :]
    return feat_tile, is_origin, pad_trailing, content_thw


# ---------------------------------------------------------------------------
# Common helpers (geometry, pad/crop, stage floors)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisPad:
    """How many elements were added (pad) or removed (crop) on each side of one axis."""

    before: int
    after: int


def resize_axis(
    x: torch.Tensor,
    dim: int,
    size: int,
    *,
    mode: ResizeAxisMode,
) -> tuple[torch.Tensor, AxisPad]:
    """Pad or crop axis ``dim`` so its length becomes ``size``.
    Pad (``len < size``):
      ``repeat_last`` - append copies of the last slice.
      ``symmetric`` - edge-replicate first/last; leftover goes to the end
      (``before = need // 2``, ``after = need - before``).
    Crop (``len > size``):
      ``repeat_last`` - drop from the end.
      ``symmetric`` - drop from both ends with the same split rule as pad.
    """
    if size < 1:
        raise ValueError(f"resize_axis target size must be >= 1, got {size}")
    if dim < 0:
        dim += x.ndim
    if not 0 <= dim < x.ndim:
        raise ValueError(f"dim {dim} out of range for rank-{x.ndim} tensor")

    length = x.shape[dim]
    if length == size:
        return x, AxisPad(0, 0)

    if length < size:
        need = size - length
        if mode == "repeat_last":
            last = x.narrow(dim, length - 1, 1)
            expand_shape = list(x.shape)
            expand_shape[dim] = need
            pad = last.expand(expand_shape)
            return torch.cat([x, pad], dim=dim), AxisPad(0, need)

        before = need // 2
        after = need - before
        first = x.narrow(dim, 0, 1)
        last = x.narrow(dim, length - 1, 1)
        parts: list[torch.Tensor] = []
        if before:
            expand_shape = list(x.shape)
            expand_shape[dim] = before
            parts.append(first.expand(expand_shape))
        parts.append(x)
        if after:
            expand_shape = list(x.shape)
            expand_shape[dim] = after
            parts.append(last.expand(expand_shape))
        return torch.cat(parts, dim=dim), AxisPad(before, after)

    need = length - size
    if mode == "repeat_last":
        return x.narrow(dim, 0, size).contiguous(), AxisPad(0, need)

    before = need // 2
    after = need - before
    return x.narrow(dim, before, size).contiguous(), AxisPad(before, after)


def ensure_min_latent_shape(
    latent: torch.Tensor,
    min_tile_sizes: Tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[AxisPad, AxisPad, AxisPad]]:
    """Pad latent ``(B, C, T, H, W)`` up to ``min_tile_sizes`` if needed."""
    min_t, min_h, min_w = min_tile_sizes
    t_pad = AxisPad(0, 0)
    h_pad = AxisPad(0, 0)
    w_pad = AxisPad(0, 0)
    x = latent
    if x.shape[2] < min_t:
        x, t_pad = resize_axis(x, 2, min_t, mode="repeat_last")
    if x.shape[3] < min_h:
        x, h_pad = resize_axis(x, 3, min_h, mode="symmetric")
    if x.shape[4] < min_w:
        x, w_pad = resize_axis(x, 4, min_w, mode="symmetric")
    return x, (t_pad, h_pad, w_pad)


def scale_axis_pad(pad: AxisPad, scale: int) -> AxisPad:
    """Scale a latent-grid ``AxisPad`` into pixel (or other) units."""
    return AxisPad(pad.before * scale, pad.after * scale)


def crop_pixels_to_content(
    pixels: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    *,
    h_pad: AxisPad | None = None,
    w_pad: AxisPad | None = None,
    spatial_scale: Tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """Crop padded decode output ``(B, C, F, H, W)`` back to the content shape.
    Temporal pad is always trailing (``repeat_last``), so T is cropped from the
    end. Spatial size-floor pads must pass the recorded ``h_pad`` / ``w_pad``
    (latent units) plus ``spatial_scale`` ``(H, W)`` so odd leftovers are not
    re-split by a center-crop after upscaling.
    """
    x, _ = resize_axis(pixels, 2, frames, mode="repeat_last")
    scale_h, scale_w = spatial_scale
    if h_pad is not None:
        before = scale_axis_pad(h_pad, scale_h).before
        if before + height > x.shape[3]:
            raise ValueError(f"H crop out of range: before={before}, height={height}, got {x.shape[3]}")
        x = x.narrow(3, before, height).contiguous()
    else:
        x, _ = resize_axis(x, 3, height, mode="symmetric")
    if w_pad is not None:
        before = scale_axis_pad(w_pad, scale_w).before
        if before + width > x.shape[4]:
            raise ValueError(f"W crop out of range: before={before}, width={width}, got {x.shape[4]}")
        x = x.narrow(4, before, width).contiguous()
    else:
        x, _ = resize_axis(x, 4, width, mode="symmetric")
    return x


def stage5_pixel_shape_from_stage4(
    stage4_t: int,
    stage4_h: int,
    stage4_w: int,
    *,
    upsample_stride: Tuple[int, int, int],
    patch_size: int,
    stage5_kernel_t: int,
    drop_leading_frame: bool,
    pad_trailing: bool,
) -> tuple[int, int, int]:
    """Pixel ``(F, H, W)`` for a stage-4-input extent (one remaining NA hop + patch)."""
    st, sh, sw = upsample_stride
    frames = stage4_t * st - 1 if drop_leading_frame and st == 2 else stage4_t * st
    if pad_trailing:
        frames = max(frames, stage5_kernel_t)
    return frames, stage4_h * sh * patch_size, stage4_w * sw * patch_size


def pad_trailing_latent_for_natten_border(latent: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Replicate the last latent frame ``n_frames`` times for NATTEN last-frame border."""
    if n_frames <= 0:
        return latent
    padded, _ = resize_axis(latent, 2, latent.shape[2] + n_frames, mode="repeat_last")
    return padded


def crop_trailing_context_natten_pad(
    context: torch.Tensor,
    *,
    n_latent_frames: int,
    time_scale: int,
    stage5_kernel_t: int,
) -> torch.Tensor:
    """Crop ghosting appendix before stage 5, leaving at least ``stage5_kernel_t``."""
    if n_latent_frames <= 0:
        return context
    ghost = n_latent_frames * time_scale
    content_t = max(context.shape[1] - ghost, 1)
    keep = min(context.shape[1], max(content_t, stage5_kernel_t))
    cropped, _ = resize_axis(context, 1, keep, mode="repeat_last")
    return cropped


def _weight_floor(dtype: torch.dtype) -> float:
    """Smallest divisor that safely guards ``buffer / weights`` in ``dtype``."""
    return max(1e-8, torch.finfo(dtype).tiny)


def stage4_thw_from_latent(
    upsample_strides: Sequence[Tuple[int, int, int]],
    latent_t: int,
    latent_h: int,
    latent_w: int,
    *,
    drop_leading_frame: bool = True,
) -> Tuple[int, int, int]:
    """Stage-4 input ``(T, H, W)`` after the first three upsample hops."""
    t, h, w = latent_t, latent_h, latent_w
    for st, sh, sw in upsample_strides[:3]:
        t, h, w = t * st, h * sh, w * sw
        if st == 2 and drop_leading_frame:
            t -= 1
    return t, h, w


def stage4_to_pixel_scale_factors(
    upsample_stride: Tuple[int, int, int],
    patch_size: int,
) -> SpatioTemporalScaleFactors:
    """Pixel/frame units per stage-4-input cell (last NA hop + unpatchify)."""
    st, sh, sw = upsample_stride
    return SpatioTemporalScaleFactors(time=st, height=sh * patch_size, width=sw * patch_size)


def compute_tile_min_size(
    stage4_kernel: Tuple[int, int, int],
    stage5_kernel: Tuple[int, int, int],
    upsample3_stride: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Min stage-4-input ``(T, H, W)`` so stages 4 and 5 each see ``>= kernel``."""
    return tuple(max(stage4_kernel[a], -(-stage5_kernel[a] // upsample3_stride[a])) for a in range(3))


def compute_tile_halos(
    stage4_kernel: Tuple[int, int, int],
    stage4_depth: int,
    stage5_kernel: Tuple[int, int, int],
    stage5_depth: int,
    upsample3_stride: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """One-sided halos in stage-4-input units for stages 4 and 5."""
    halo4 = tuple(stage4_depth * (stage4_kernel[a] // 2) for a in range(3))
    halo5 = tuple(-(-(stage5_depth * (stage5_kernel[a] // 2)) // upsample3_stride[a]) for a in range(3))
    return halo4, halo5  # type: ignore[return-value]


def _cumulative_upsample_strides(
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
) -> List[Tuple[int, int, int]]:
    """Per-axis product of hop strides for ``upsamples[:i]`` (``cumulative[0] = (1,1,1)``)."""
    cumulative = [(1, 1, 1)]
    t, h, w = 1, 1, 1
    for stride, _ in upsamples:
        t, h, w = t * stride[0], h * stride[1], w * stride[2]
        cumulative.append((t, h, w))
    return cumulative


def all_stages_min_tile_size(
    stage_kernels: Sequence[Tuple[int, int, int]],
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
    stage5_kernel: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Per-axis latent-grid floor so every stage's NA sees dims ``>= kernel_size``."""
    cumulative = _cumulative_upsample_strides(upsamples)
    mins = [1, 1, 1]
    for stage_i in range(len(upsamples)):
        strides = cumulative[stage_i]
        for axis in range(3):
            mins[axis] = max(mins[axis], -(-stage_kernels[stage_i][axis] // strides[axis]))
    strides5 = cumulative[len(upsamples)]
    for axis in range(3):
        mins[axis] = max(mins[axis], -(-stage5_kernel[axis] // strides5[axis]))
    return (mins[0], mins[1], mins[2])


def pixel_tile_shape(full_shape: tuple[int, ...], out_coords: tuple[slice, ...]) -> tuple[int, ...]:
    dims: list[int] = []
    for size, coord in zip(full_shape, out_coords, strict=True):
        start, stop, step = coord.indices(size)
        dims.append(len(range(start, stop, step)))
    return tuple(dims)


# ---------------------------------------------------------------------------
# Recommendation helpers
# ---------------------------------------------------------------------------


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def recommended_pixel_overlaps(
    tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
) -> Tuple[int, int]:
    """Stage-4/5-safe ``(temporal_overlap_frames, spatial_overlap_pixels)``.
    Shared by :func:`recommended_decode_tiling_config` (to *set* overlaps) and
    :func:`~ltx_core.tiling._validate_overlap` (to reject undersized configs).
    """

    def dominant(axis: int) -> int:
        return max(tile_halos[i][axis] for i in range(len(tile_halos)))

    overlap_t = _round_up(dominant(0) * pixel_scale.time, 8)
    halo_hw = max(dominant(1), dominant(2))
    overlap_hw = _round_up(halo_hw * pixel_scale.height, 32)
    return overlap_t, overlap_hw


def stage5_tokens_for_pixel_tile(
    tile_frames: int,
    tile_height: int,
    tile_width: int,
    *,
    patch_size: int,
) -> int:
    """Pre-unpatchify stage-5 token count for a pixel-space tile (NATTEN volume)."""
    h5 = max(1, tile_height // patch_size)
    w5 = max(1, tile_width // patch_size)
    return tile_frames * h5 * w5


def _axis_candidates(length: int, overlap: int, min_size: int, multiple: int) -> list[tuple[int, int]]:
    """``(tile_size, num_tiles)`` for every legal size on ``multiple``'s grid."""
    out: list[tuple[int, int]] = []
    max_size = max(_round_up(length, multiple), min_size)
    for size in range(min_size, max_size + multiple, multiple):
        if size <= overlap:
            continue
        n = len(split_by_size(size, overlap)(length).intervals)
        out.append((size, n))
    return out


def volumetric_overlap_waste(
    *,
    num_frames: int,
    height: int,
    width: int,
    tile_frames: int,
    tile_height: int,
    tile_width: int,
    n_t: int,
    n_h: int,
    n_w: int,
) -> float:
    """``processed_volume / unique_volume`` (>= 1). Lower means less overlap recompute."""
    processed = n_t * n_h * n_w * tile_frames * tile_height * tile_width
    unique = max(1, num_frames * height * width)
    return processed / unique


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------


def _propagate_interval_through_upsample_hops(
    interval: DimensionInterval,
    strides: Sequence[int],
    causal: bool,
) -> DimensionInterval:
    """Forward-propagate one interval through a sequence of upsample hops on one axis.
    Mirrors :class:`~ltx_core.model.video_vae.transformer.layers.LinearPixelShuffleUpsample`:
    multiply by ``stride``, and for the causal temporal axis when ``stride == 2`` apply
    the duplicate-frame drop (``end -= 1``; non-origin also ``start -= 1``).
    This is *not* :func:`~ltx_core.model.video_vae.video_vae.map_temporal_slice` (ConvVAE).
    DiffVAE non-origin tiles run with ``drop_leading_frame=False`` and must keep length
    ``tile_t * stride``; the ConvVAE ``1+(L-1)*stride`` mapping is one frame short and
    shifts non-origin ``out_coords``, which breaks tiled↔untiled temporal blend even
    when masks are complementary.
    """
    x = interval
    for stride in strides:
        if stride < 1:
            raise ValueError(f"upsample stride must be >= 1, got {stride}")
        start = x.start * stride
        end = x.end * stride
        left_ramp = x.left_ramp * stride
        right_ramp = x.right_ramp * stride
        if causal and stride == 2:
            end -= 1
            if x.start != 0:
                start -= 1
        x = DimensionInterval(start=start, end=end, left_ramp=left_ramp, right_ramp=right_ramp)
    return x
