"""GPU RGB→YUV420 packing for PyAV video encode (SDR 8-bit and HDR 10-bit).
Shared by ltx-pipelines and ltx-trainer. Runs between the VAE decoder (float RGB
chunks) and encode / HLG paths, bypassing pyav's CPU-side libswscale.
RGB→Y'CbCr matrices come from ``colour-science``. 4:2:0 subsampling and
I420 / planar packing stay here (not provided by colour).
"""

from __future__ import annotations

import enum
import functools

import colour
import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.devices import upload_from_host


class ColorSpace(enum.Enum):
    """YUV color space standard (FFmpeg ``AVCOL_SPC_*`` + colour-science weights)."""

    BT_709 = "bt709"
    BT_2020_NCL = "bt2020ncl"

    @property
    def av_colorspace(self) -> int:
        """FFmpeg ``AVCOL_SPC_*`` for ``codec_context.colorspace``."""
        match self:
            case ColorSpace.BT_709:
                return 1  # AVCOL_SPC_BT709
            case ColorSpace.BT_2020_NCL:
                return 9  # AVCOL_SPC_BT2020_NCL

    @property
    def _weights_name(self) -> str:
        match self:
            case ColorSpace.BT_709:
                return "ITU-R BT.709"
            case ColorSpace.BT_2020_NCL:
                return "ITU-R BT.2020"

    @property
    def rgb_to_ycbcr_matrix(self) -> torch.Tensor:
        """Float RGB→Y'CbCr (full range, no code levels), float32 3x3."""
        return _RGB_TO_YCBCR[self]


class ColorRange(enum.Enum):
    """YUV color range."""

    MPEG = "mpeg"
    JPEG = "jpeg"

    @property
    def av_color_range(self) -> int:
        """FFmpeg ``AVCOL_RANGE_*`` for ``codec_context.color_range``."""
        match self:
            case ColorRange.MPEG:
                return 1  # AVCOL_RANGE_MPEG (limited)
            case ColorRange.JPEG:
                return 2  # AVCOL_RANGE_JPEG (full)


class PixelFormat(enum.Enum):
    """Pixel format for video frames."""

    RGB24 = "rgb24"
    YUV420P = "yuv420p"

    @property
    def av_format(self) -> str:
        """PyAV format string for ``VideoFrame.from_ndarray``."""
        return self.value


def _rgb_to_ycbcr_matrix(weights_name: str) -> torch.Tensor:
    """Float RGB→Y'CbCr from colour-science (``matrix_YCbCr`` inverted)."""
    ycbcr_to_rgb = torch.as_tensor(
        colour.matrix_YCbCr(colour.WEIGHTS_YCBCR[weights_name], is_legal=False, is_int=False),
        dtype=torch.float64,
    )
    return torch.linalg.inv(ycbcr_to_rgb).to(torch.float32)


# Built once; ``ColorSpace.rgb_to_ycbcr_matrix`` reads from here.
_RGB_TO_YCBCR = {cs: _rgb_to_ycbcr_matrix(cs._weights_name) for cs in ColorSpace}


@functools.lru_cache(maxsize=16)
def _device_rgb_to_ycbcr_matrix(color_space: ColorSpace, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``color_space``'s matrix on ``device`` in ``dtype``, uploaded once without waiting for the device."""
    return upload_from_host(color_space.rgb_to_ycbcr_matrix.to(dtype=dtype), device)


# cuBLAS GEMM dims are int32. PyTorch folds ``(F, H*W, 3) @ (3, 3)`` into one
# ``mm`` of size ``F*H*W``, reported as ``n`` in column-major. ``n`` must be
# strictly less than this value (see ``at::cuda::blas::gemm``).
_CUBLAS_INT32_MAX = 2**31 - 1


def _require_rgb_chw(image: torch.Tensor) -> None:
    if image.ndim < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")


def _max_frames_per_gemm(height: int, width: int) -> int:
    """Largest ``F`` with ``F * H * W < 2**31 - 1``.
    ``int_max // H // W`` is that exclusive upper bound: a GEMM may use strictly
    fewer frames than this so the flattened ``n`` stays below the cuBLAS limit.
    """
    if height <= 0 or width <= 0:
        return _CUBLAS_INT32_MAX
    return _CUBLAS_INT32_MAX // height // width


def frames_per_yuv_gemm(height: int, width: int) -> int:
    """Frames :func:`rgb_to_yuv` converts per GEMM at ``height x width``.
    Above this many frames it converts in chunks and writes YUV back over the RGB
    storage, so only a chunk-sized temporary is newly allocated. Memory planners
    (see ``diffusion_tiling.emit_convert_bytes``) need that split to size a reserve.
    """
    limit = _max_frames_per_gemm(height, width)
    return 1 if limit <= 1 else limit - 1


def rgb_to_yuv(image: torch.Tensor, color_space: ColorSpace) -> torch.Tensor:
    """RGB ``[0, 1]`` ``(*, 3, H, W)`` → YUV ``(*, 3, H, W)`` (float, no code levels).
    Matrix from colour-science. Long clips are converted in frame chunks of size
    ``< (2**31 - 1) // H // W`` so each 3x3 GEMM stays inside cuBLAS int32.
    When chunking, YUV is written back over the RGB storage; otherwise chunks
    are concatenated. (``colour.RGB_to_YCbCr`` is NumPy-only and not used here.)
    """
    _require_rgb_chw(image)
    mat = _device_rgb_to_ycbcr_matrix(color_space, image.device, image.dtype)
    orig_shape = image.shape
    height, width = orig_shape[-2], orig_shape[-1]
    frames = image.reshape(-1, 3, height, width)
    n_frames = frames.shape[0]
    limit = _max_frames_per_gemm(height, width)
    if limit < 1:
        raise ValueError(
            f"Frame {height}x{width} exceeds the cuBLAS int32 GEMM limit "
            f"({_CUBLAS_INT32_MAX}); cannot convert even one frame."
        )
    # Strictly fewer than ``int_max // H // W`` so ``F*H*W < 2**31 - 1``.
    chunk_f = 1 if limit == 1 else limit - 1

    def _gemm(rgb: torch.Tensor) -> torch.Tensor:
        pixels = rgb.movedim(-3, -1)
        yuv = pixels.flatten(-3, -2) @ mat.T
        return yuv.unflatten(-2, (height, width)).movedim(-1, -3)

    if n_frames <= chunk_f:
        return _gemm(frames).reshape(orig_shape)

    # Overwrite RGB in place when ``reshape`` is a view; otherwise collect.
    writeback = frames.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()
    chunks: list[torch.Tensor] = []
    for start in range(0, n_frames, chunk_f):
        yuv = _gemm(frames[start : start + chunk_f])
        if writeback:
            frames[start : start + chunk_f].copy_(yuv)
        else:
            chunks.append(yuv)
    if writeback:
        return frames.reshape(orig_shape)
    return torch.cat(chunks, dim=0).reshape(orig_shape)


def apply_color_range_(
    y: torch.Tensor,
    uv: torch.Tensor,
    color_range: ColorRange,
    *,
    bits: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale float Y/UV to 8- or 10-bit code levels in-place.
    MPEG (limited): Y ``(219·E'+16)·2^(bits-8)``, Cb/Cr ``(224·E'+128)·2^(bits-8)``.
    JPEG (full): Y ``E'·(2^bits-1)``, Cb/Cr ``(E'+0.5)·(2^bits-1)``.
    """
    if bits not in (8, 10):
        raise ValueError(f"Unsupported bit depth: {bits}")
    factor = 1 << (bits - 8)
    match color_range:
        case ColorRange.MPEG:
            y.mul_(219 * factor).add_(16 * factor)
            uv.mul_(224 * factor).add_(128 * factor)
        case ColorRange.JPEG:
            max_code = (1 << bits) - 1
            y.mul_(max_code)
            uv.add_(0.5).mul_(max_code)
        case _:
            raise ValueError(f"Unsupported color range: {color_range}")
    return y, uv


def apply_color_range_10bit_(
    y: torch.Tensor, uv: torch.Tensor, color_range: ColorRange
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale Y/UV to 10-bit code levels in-place."""
    return apply_color_range_(y, uv, color_range, bits=10)


def _rgb_to_yuv420_unit(image: torch.Tensor, color_space: ColorSpace) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB ``[0,1]`` → Y ``[0,1]`` + UV centered at 0, 4:2:0 (no code levels)."""
    _require_rgb_chw(image)
    if image.shape[-2] % 2 != 0 or image.shape[-1] % 2 != 0:
        raise ValueError(f"Input H and W must be divisible by 2. Got {image.shape}")

    yuv = rgb_to_yuv(image, color_space)
    y = yuv[..., :1, :, :]
    uv_full = yuv[..., 1:3, :, :].contiguous()
    lead = uv_full.shape[:-3]
    uv_flat = uv_full.reshape(-1, 2, uv_full.shape[-2], uv_full.shape[-1])
    uv = torch.nn.functional.avg_pool2d(uv_flat, kernel_size=2, stride=2)
    return y, uv.reshape(*lead, 2, uv.shape[-2], uv.shape[-1])


def rgb_to_yuv420(
    image: torch.Tensor, color_space: ColorSpace, color_range: ColorRange
) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB ``(*, 3, H, W)`` → 8-bit-level Y / UV 4:2:0."""
    y, uv = _rgb_to_yuv420_unit(image, color_space)
    return apply_color_range_(y, uv, color_range)


def rgb_to_yuv420p10(
    image: torch.Tensor, color_space: ColorSpace, color_range: ColorRange
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB float ``[0, 1]`` ``(*, 3, H, W)`` → planar uint16 Y, U, V (yuv420p10)."""
    y, uv = _rgb_to_yuv420_unit(image, color_space)
    apply_color_range_(y, uv, color_range, bits=10)
    return yuv420_planes_u16(y, uv)


def rgba_to_yuva444p10(image: torch.Tensor, color_space: ColorSpace, color_range: ColorRange) -> torch.Tensor:
    """Straight RGBA float ``[0, 1]`` ``(*, 4, H, W)`` → planar uint16 Y, U, V, A ``(*, 4, H, W)``.
    Colour takes ``color_range`` 10-bit code levels; alpha is always full range (``0``-``1023``).
    """
    if image.ndim < 3 or image.shape[-3] != 4:
        raise ValueError(f"Expected (*, 4, H, W) RGBA, got shape {tuple(image.shape)}")
    yuv = rgb_to_yuv(image[..., :3, :, :].contiguous(), color_space)
    y, uv = apply_color_range_(yuv[..., :1, :, :], yuv[..., 1:, :, :], color_range, bits=10)
    alpha = image[..., 3:, :, :] * 1023
    return torch.cat([y, uv, alpha], dim=-3).round_().clamp_(0, 1023).to(torch.uint16)


def pack_i420(y: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Pack Y + UV into I420 ``(*, H*3//2, W)`` uint8 for PyAV."""
    y_plane = y[..., 0, :, :]
    uv_packed = uv.reshape(*uv.shape[:-3], uv.shape[-2], uv.shape[-1] * 2)
    return torch.cat([y_plane, uv_packed], dim=-2).clamp_(0, 255).to(torch.uint8)


def yuv420_planes_u16(y: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split 10-bit Y/UV code levels into planar ``uint16`` Y, U, V."""
    y16 = y[..., 0, :, :].round_().clamp_(0, 1023).to(torch.uint16)
    u16 = uv[..., 0, :, :].round_().clamp_(0, 1023).to(torch.uint16)
    v16 = uv[..., 1, :, :].round_().clamp_(0, 1023).to(torch.uint16)
    return y16, u16, v16


class FrameConverter(enum.Enum):
    """Converts ``[*, C, H, W]`` float ``[0, 1]`` frames for encode.
    Carries encoding metadata so the caller can tag the output stream.
    ``__call__`` **may mutate** its input (aliases keep the trailing-underscore
    convention: ``rgb_uint8_converter_``, ``yuv420p_bt709_converter_``).
    HLG / 10-bit packing stays in ``hlg`` — not a case here.
    """

    RGB24 = "rgb24"
    YUV420P_BT709 = "yuv420p_bt709"

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        match self:
            case FrameConverter.RGB24:
                return frames.clamp_(0.0, 1.0).mul_(255.0).to(torch.uint8).movedim(-3, -1)
            case FrameConverter.YUV420P_BT709:
                y, uv = rgb_to_yuv420(frames, ColorSpace.BT_709, ColorRange.MPEG)
                return pack_i420(y, uv)

    @property
    def pixel_format(self) -> PixelFormat:
        match self:
            case FrameConverter.RGB24:
                return PixelFormat.RGB24
            case FrameConverter.YUV420P_BT709:
                return PixelFormat.YUV420P

    @property
    def color_space(self) -> ColorSpace | None:
        match self:
            case FrameConverter.RGB24:
                return None
            case FrameConverter.YUV420P_BT709:
                return ColorSpace.BT_709

    @property
    def color_range(self) -> ColorRange | None:
        match self:
            case FrameConverter.RGB24:
                return None
            case FrameConverter.YUV420P_BT709:
                return ColorRange.MPEG


rgb_uint8_converter_ = FrameConverter.RGB24
yuv420p_bt709_converter_ = FrameConverter.YUV420P_BT709
