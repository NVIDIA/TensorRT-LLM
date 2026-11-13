# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""NVFP4 quantization helpers for the KVFP4 sparse attention kernel.

This file is intended as a customer-facing example for preparing KV tensors
for the KVFP4 attention kernel:
  - BF16/FP16 K/V input
  - packed E2M1 FP4 data (Transformer Engine, or a dependency-free PyTorch
    fallback)
  - E4M3 block scales in **vLLM's** scale-factor byte order
  - one FP32 tensor/global scale per tensor

Scale-factor layouts
--------------------
There is exactly one supported layout *pair*, matching what vLLM's NVFP4
KV-cache store kernel writes (``csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu``),
so MSA reads that cache with no repack pass:

===============  =========================================  ===============
layout           byte offset inside a scale region          used for
===============  =========================================  ===============
``"linear"``     ``t*S + s``                                K
``"swizzle4x4"`` ``(t//4)*(4*S) + 4*s + (t%4)``             V
===============  =========================================  ===============

``t`` is the token index inside the region, ``s`` the scale column, and
``S = head_dim // 16`` the number of scale columns.  ``S`` must be a multiple
of 4 and, for ``"swizzle4x4"``, the region must hold a whole number of token
quads.  The cuBLAS/cuDNN **128x4** layout this module used to emit is deleted:
MSA uses no block-scaled MMA, so it bought nothing, and it did not match vLLM.

Physical containers
-------------------
A *region* is exactly ``region_tokens * S`` bytes -- there is no row or column
padding.  The container returned by :func:`quantize_bf16_to_nvfp4` has the
input's shape with the last dimension replaced by ``S``:

* **paged** input ``[num_pages, Hkv, page_size, D]``
  -> container ``[num_pages, Hkv, page_size, S]``.
  One region per ``(page, head)``, ``region_tokens = page_size``.  This is
  byte-for-byte the view ``vllm.utils.torch_utils.nvfp4_split_data_scale``
  hands back (modulo its ``float8_e4m3fn`` dtype; see the note in
  ``interface.py``).

* **flat varlen** input ``[total_k, Hkv, D]`` (or ``[total_k, D]``, Hkv = 1)
  -> container ``[total_k, Hkv, S]``.  For ``"linear"`` one region per
  ``(token, head)`` (``region_tokens = 1``, so the container is also
  *logically* indexed by ``(token, head, s)``).  For ``"swizzle4x4"`` one
  region per ``(token_quad, head)``, ``region_tokens = 4``, regions ordered
  quad-major then head; the container is then only a **byte container** --
  element ``(t, h, s)`` of the tensor is not the scale of token ``t``.
  Byte address of the scale for ``(t, h, s)``, with
  ``page_stride = stride(0) = Hkv*S`` and ``head_stride = stride(1) = S``::

      linear      : t*page_stride + h*head_stride + s
      swizzle4x4  : (t & ~3)*page_stride + 4*h*head_stride + 4*s + (t & 3)

  The flat swizzle has no production producer (vLLM is paged only); it exists
  because flat test inputs are far cheaper to build than paged ones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch


NVFP4_BLOCK_SIZE = 16
NVFP4_FP4_MAX = 6.0
NVFP4_FP8_E4M3_MAX = 448.0

#: The only supported scale-factor layouts.  ``"linear"`` for K,
#: ``"swizzle4x4"`` for V -- see the module docstring.
NVFP4_SCALE_LAYOUTS: Tuple[str, ...] = ("linear", "swizzle4x4")


@dataclass(frozen=True)
class Nvfp4QuantizedTensor:
    """Packed NVFP4 tensor plus dequantization metadata.

    Attributes
    ----------
    data : torch.Tensor
        Packed E2M1 FP4 data.  The last dimension is half of the original
        logical last dimension because each byte stores two FP4 values.
    scale : torch.Tensor
        E4M3 block scales as uint8, in the physical container described in the
        module docstring (input shape with the last dim replaced by
        ``head_dim // 16``).
    global_scale : torch.Tensor
        FP32 tensor/global dequant scale.
    logical_scale_shape : tuple[int, int]
        Logical 2D scale shape ``(rows, cols)`` before packing, where ``rows``
        is ``prod(original_shape[:-1])``.
    original_shape : tuple[int, ...]
        Original BF16/FP16 tensor shape before quantization.
    layout : str
        Scale-factor layout of ``scale``: ``"linear"`` or ``"swizzle4x4"``.
    """

    data: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    logical_scale_shape: Tuple[int, int]
    original_shape: Tuple[int, ...]
    layout: str


def nvfp4_scale_linear_offset(t, s, S):
    """Byte offset of scale column ``s`` of token ``t`` in a linear region.

    ``off = t*S + s``.  Matches vLLM's K scale store
    (``nvfp4_kv_cache_kernels.cu:162``).

    ``t`` and ``s`` may be Python ints or integer tensors (broadcasting).
    """

    return t * S + s


def nvfp4_scale_swizzle4x4_offset(t, s, S):
    """Byte offset of scale column ``s`` of token ``t`` in a 4x4-swizzled region.

    ``off = (t//4)*(4*S) + 4*s + (t%4)`` -- i.e. a plain row-major
    ``[T/4][S][4]`` array indexed by ``(t//4, s, t%4)``.  Matches vLLM's V
    scale store (``nvfp4_kv_cache_kernels.cu:25-39``, ``:164-171``); the two
    forms are algebraically identical for every ``S`` divisible by 4.

    ``t`` and ``s`` may be Python ints or integer tensors (broadcasting).
    """

    return (t // 4) * (4 * S) + 4 * s + (t % 4)


_SCALE_OFFSET_FNS = {
    "linear": nvfp4_scale_linear_offset,
    "swizzle4x4": nvfp4_scale_swizzle4x4_offset,
}


def _check_layout(layout: str) -> str:
    layout = str(layout)
    if layout not in NVFP4_SCALE_LAYOUTS:
        raise ValueError(
            f"layout must be one of {list(NVFP4_SCALE_LAYOUTS)}, got {layout!r}"
        )
    return layout


def _region_offsets(page_size: int, scale_cols: int, layout: str, device) -> torch.Tensor:
    """Flat physical offsets for one region, in logical ``(t, s)`` row-major order."""

    t = torch.arange(page_size, device=device, dtype=torch.int64)[:, None]
    s = torch.arange(scale_cols, device=device, dtype=torch.int64)[None, :]
    return _SCALE_OFFSET_FNS[layout](t, s, scale_cols).reshape(-1)


def _validate_region(page_size: int, scale_cols: int, layout: str) -> None:
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if scale_cols <= 0 or scale_cols % 4 != 0:
        raise ValueError(
            "scale_cols must be a positive multiple of 4 (the 4x4 swizzle "
            f"derivation assumes S % 4 == 0), got {scale_cols}"
        )
    if layout == "swizzle4x4" and page_size % 4 != 0:
        raise ValueError(
            "layout='swizzle4x4' tiles tokens in quads, so the region must hold "
            f"a whole number of quads; got page_size={page_size}"
        )


def pack_nvfp4_scale(
    scale: torch.Tensor,
    *,
    page_size: int,
    scale_cols: int,
    layout: str,
) -> torch.Tensor:
    """Pack logical per-region block scales into physical scale-factor order.

    Parameters
    ----------
    scale : torch.Tensor
        Logical scales with shape ``[n_regions, page_size, scale_cols]``.
        Reinterpreted as uint8 if it is not already uint8.
    page_size : int
        Tokens per region.  Must be a multiple of 4 for ``"swizzle4x4"``.
    scale_cols : int
        Scale columns per token, ``S = head_dim // 16``.  Must be a multiple
        of 4.
    layout : str
        ``"linear"`` or ``"swizzle4x4"``.

    Returns
    -------
    torch.Tensor
        Contiguous uint8 tensor with the same shape, whose *bytes* are the
        packed region layout: region ``r`` occupies exactly
        ``page_size * scale_cols`` contiguous bytes (**no padding** -- unlike
        the deleted 128x4 layout, which padded to
        ``round_up(rows,128) x round_up(cols,4)``), and the scale of token
        ``t``, column ``s`` sits at byte ``offset_<layout>(t, s, scale_cols)``
        inside it.

    ``"linear"`` is the identity: a row-major ``[n_regions, page_size,
    scale_cols]`` array already *is* the linear layout.
    """

    layout = _check_layout(layout)
    page_size = int(page_size)
    scale_cols = int(scale_cols)
    _validate_region(page_size, scale_cols, layout)

    if scale.ndim != 3:
        raise ValueError(
            "scale must be rank-3 [n_regions, page_size, scale_cols], got shape "
            f"{tuple(scale.shape)}"
        )
    if int(scale.shape[1]) != page_size or int(scale.shape[2]) != scale_cols:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match "
            f"page_size={page_size}, scale_cols={scale_cols}"
        )
    if scale.dtype is not torch.uint8:
        scale = scale.view(torch.uint8)

    n_regions = int(scale.shape[0])
    src = scale.reshape(n_regions, page_size * scale_cols).contiguous()
    if layout == "linear":
        return src.reshape(n_regions, page_size, scale_cols)

    offsets = _region_offsets(page_size, scale_cols, layout, src.device)
    out = torch.empty_like(src)
    out[:, offsets] = src
    return out.reshape(n_regions, page_size, scale_cols)


def _unpack_nvfp4_scale(
    packed: torch.Tensor,
    *,
    page_size: int,
    scale_cols: int,
    layout: str,
) -> torch.Tensor:
    """Inverse of :func:`pack_nvfp4_scale`; returns logical ``(t, s)`` order."""

    layout = _check_layout(layout)
    page_size = int(page_size)
    scale_cols = int(scale_cols)
    _validate_region(page_size, scale_cols, layout)

    n_regions = int(packed.shape[0])
    src = packed.reshape(n_regions, page_size * scale_cols)
    if layout == "linear":
        return src.reshape(n_regions, page_size, scale_cols)
    offsets = _region_offsets(page_size, scale_cols, layout, src.device)
    return src[:, offsets].reshape(n_regions, page_size, scale_cols)


def _scale_region_grid(original_shape, layout: str) -> Tuple[int, int, bool]:
    """Region decomposition of a K/V tensor of shape ``original_shape``.

    Returns ``(n_regions, region_tokens, quad_regroup)``.  ``quad_regroup``
    marks the flat-varlen swizzle case, whose logical rows (token-major, then
    head) have to be regrouped into ``(token_quad, head)`` regions.
    """

    shape = tuple(int(v) for v in original_shape)
    if len(shape) == 4:
        # Paged: [num_pages, Hkv, page_size, D] -> one region per (page, head).
        return shape[0] * shape[1], shape[2], False
    if len(shape) not in (2, 3):
        raise ValueError(
            "NVFP4 K/V must be rank-4 [num_pages, Hkv, page_size, D], rank-3 "
            f"[total_k, Hkv, D] or rank-2 [total_k, D]; got rank {len(shape)}"
        )
    total_k = shape[0]
    heads = shape[1] if len(shape) == 3 else 1
    if layout == "linear":
        # One region per (token, head): a single token row of S bytes.
        return total_k * heads, 1, False
    if total_k % 4 != 0:
        raise ValueError(
            "flat-varlen layout='swizzle4x4' tiles the global token axis in "
            f"quads, so total_k must be a multiple of 4; got {total_k}"
        )
    return (total_k // 4) * heads, 4, True


def _pack_block_scale(
    scale: torch.Tensor,
    *,
    rows: int,
    scale_cols: int,
    original_shape,
    layout: str,
) -> torch.Tensor:
    """Pack logical ``[rows, scale_cols]`` block scales into the kernel container.

    ``scale`` may be larger than ``[rows, scale_cols]`` (Transformer Engine
    returns padded rowwise scales); the excess is dropped.
    """

    layout = _check_layout(layout)
    shape = tuple(int(v) for v in original_shape)
    rows = int(rows)
    scale_cols = int(scale_cols)
    if scale.dtype is not torch.uint8:
        scale = scale.view(torch.uint8)
    if scale.ndim != 2:
        raise ValueError(f"block scale must be 2D, got shape {tuple(scale.shape)}")
    if int(scale.shape[0]) < rows or int(scale.shape[1]) < scale_cols:
        raise ValueError(
            "block scale is smaller than the logical shape: got "
            f"{tuple(scale.shape)}, need at least {(rows, scale_cols)}"
        )
    logical = scale[:rows, :scale_cols].contiguous()

    n_regions, region_tokens, quad_regroup = _scale_region_grid(shape, layout)
    if quad_regroup:
        heads = shape[1] if len(shape) == 3 else 1
        regions = (
            logical.reshape(shape[0] // 4, 4, heads, scale_cols)
            .permute(0, 2, 1, 3)
            .reshape(n_regions, region_tokens, scale_cols)
        )
    else:
        regions = logical.reshape(n_regions, region_tokens, scale_cols)

    packed = pack_nvfp4_scale(
        regions, page_size=region_tokens, scale_cols=scale_cols, layout=layout
    )
    return packed.reshape(shape[:-1] + (scale_cols,)).contiguous()


def _unpack_block_scale(
    packed: torch.Tensor,
    *,
    rows: int,
    scale_cols: int,
    original_shape,
    layout: str,
) -> torch.Tensor:
    """Inverse of :func:`_pack_block_scale`; returns ``[rows, scale_cols]`` uint8."""

    layout = _check_layout(layout)
    shape = tuple(int(v) for v in original_shape)
    rows = int(rows)
    scale_cols = int(scale_cols)
    if packed.dtype is not torch.uint8:
        packed = packed.view(torch.uint8)
    if packed.numel() < rows * scale_cols:
        raise ValueError(
            f"packed scale holds {packed.numel()} bytes, need "
            f"{rows * scale_cols} for shape {shape}"
        )

    n_regions, region_tokens, quad_regroup = _scale_region_grid(shape, layout)
    flat = packed.contiguous().reshape(n_regions, region_tokens * scale_cols)
    logical = _unpack_nvfp4_scale(
        flat, page_size=region_tokens, scale_cols=scale_cols, layout=layout
    )
    if quad_regroup:
        heads = shape[1] if len(shape) == 3 else 1
        logical = (
            logical.reshape(shape[0] // 4, heads, 4, scale_cols)
            .permute(0, 2, 1, 3)
            .reshape(rows, scale_cols)
        )
    return logical.reshape(rows, scale_cols)


def nvfp4_global_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """Compute TE NVFP4 tensor/global dequant scale from rowwise amax.

    Parameters
    ----------
    amax : torch.Tensor
        Rowwise absolute maxima returned by Transformer Engine.

    Returns
    -------
    torch.Tensor
        FP32 global scale equal to ``amax / (448 * 6)``.
    """

    return amax.to(torch.float32) / (NVFP4_FP8_E4M3_MAX * NVFP4_FP4_MAX)


def _import_te_nvfp4_quantizer():
    try:
        from transformer_engine.pytorch.tensor import NVFP4Quantizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Transformer Engine NVFP4 quantization is unavailable. Install a "
            "Transformer Engine build with its PyTorch dependencies, including "
            "FlashAttention v3 when required by that TE build."
        ) from exc
    return NVFP4Quantizer


# ---------------------------------------------------------------------------
# Dependency-free (pure PyTorch) NVFP4 quantizer
# ---------------------------------------------------------------------------
#
# This is a drop-in replacement for the Transformer Engine path above.  It
# exists so the NVFP4 accuracy tests can run on machines without a working
# Transformer Engine build; it is an exact round-trip partner of
# ``dequantize_nvfp4_to_bf16`` below and uses the same physical layout the
# kernel reads (``_scale_offset_linear`` / ``_scale_offset_swizzle4x4`` in
# ``src/sm100/fwd/atten_fwd_nvfp4_kv.py``).

# FP4 E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit.  Codes 0..7 hold the
# non-negative magnitudes and bit 3 is the sign.  Same table as the LUT used by
# ``dequantize_nvfp4_to_bf16``.
_E2M1_MAGNITUDES: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# ``_E2M1_MIDPOINTS[i]`` is the tie point between code ``i`` and code ``i + 1``.
_E2M1_MIDPOINTS: Tuple[float, ...] = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _round_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 values to FP4 E2M1 codes, round-to-nearest-even.

    Ties resolve to the code whose mantissa bit is zero, i.e. the even code,
    matching ``cvt.rn.satfinite.e2m1x2.f32``.  Magnitudes above 6 saturate to
    code 7 (6.0); the sign is preserved in bit 3, including for negative zero.
    """

    xf = torch.nan_to_num(x.to(torch.float32), nan=0.0)
    sign = torch.signbit(xf)
    magnitude = xf.abs().contiguous()
    midpoints = torch.tensor(
        _E2M1_MIDPOINTS, dtype=torch.float32, device=xf.device
    )
    # ``low`` counts midpoints strictly below the magnitude (round down on a
    # tie); ``high`` counts midpoints at or below it (round up on a tie).  They
    # differ only exactly on a tie, where the tie sits at index ``low``.
    low = torch.searchsorted(midpoints, magnitude, right=False)
    high = torch.searchsorted(midpoints, magnitude, right=True)
    code = torch.where(low % 2 == 0, low, high).to(torch.uint8)
    return code | (sign.to(torch.uint8) << 3)


def _pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack pairs of E2M1 codes into bytes; even element in the low nibble."""

    return (codes[..., 0::2] & 0x0F) | (codes[..., 1::2] << 4)


def _quantize_bf16_to_nvfp4_torch(
    x: torch.Tensor,
    *,
    rows: int,
    scale_cols: int,
    layout: str,
) -> "Nvfp4QuantizedTensor":
    """Quantize to NVFP4 with plain PyTorch, no Transformer Engine.

    Follows the same recipe TE uses:
      global_scale = amax(|x|) / (448 * 6)
      block_scale  = to_e4m3(amax(|block|) / (6 * global_scale))    <= 448
      fp4          = to_e2m1(block / (block_scale * global_scale))
    so ``fp4 * block_scale * global_scale`` reconstructs the value exactly in
    FP32, which is what ``dequantize_nvfp4_to_bf16`` computes.

    Everything except the final :func:`_pack_block_scale` call is
    layout-agnostic: the global scale, the block amax, the E4M3 encode, the
    E2M1 rounding and the nibble packing do not depend on where the scale
    bytes end up.
    """

    flat = x.detach().to(torch.float32).contiguous().reshape(rows, -1)
    amax = flat.abs().amax()
    global_scale = nvfp4_global_scale_from_amax(amax.reshape(1))
    # An all-zero tensor would give a zero (and thus unusable) global scale.
    global_scale = torch.where(
        global_scale > 0, global_scale, torch.ones_like(global_scale)
    )

    blocks = flat.reshape(rows, scale_cols, NVFP4_BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)
    encoded = (block_amax / (NVFP4_FP4_MAX * global_scale)).clamp(
        min=0.0, max=NVFP4_FP8_E4M3_MAX
    )
    block_scale = encoded.to(torch.float8_e4m3fn)
    dequant_step = block_scale.to(torch.float32) * global_scale
    nonzero = dequant_step > 0
    safe_step = torch.where(nonzero, dequant_step, torch.ones_like(dequant_step))

    codes = _round_to_e2m1_codes(blocks / safe_step.unsqueeze(-1))
    codes = torch.where(nonzero.unsqueeze(-1), codes, torch.zeros_like(codes))
    data = _pack_e2m1_codes(codes.reshape(rows, -1)).reshape(
        *x.shape[:-1], x.shape[-1] // 2
    )

    scale = _pack_block_scale(
        block_scale.view(torch.uint8),
        rows=rows,
        scale_cols=scale_cols,
        original_shape=x.shape,
        layout=layout,
    )

    return Nvfp4QuantizedTensor(
        data=data.contiguous(),
        scale=scale.contiguous(),
        global_scale=global_scale.to(torch.float32).contiguous(),
        logical_scale_shape=(rows, scale_cols),
        original_shape=tuple(int(v) for v in x.shape),
        layout=layout,
    )


def te_nvfp4_quantizer_available() -> bool:
    """Return True when Transformer Engine's NVFP4 quantizer can be imported."""

    try:
        _import_te_nvfp4_quantizer()
    except RuntimeError:
        return False
    return True


def quantize_bf16_to_nvfp4(
    x: torch.Tensor,
    *,
    layout: str,
    backend: str = "auto",
) -> Nvfp4QuantizedTensor:
    """Quantize a BF16/FP16 tensor to NVFP4.

    TE returns rowwise scales in logical padded layout.  This helper returns
    the scales in the physical container the attention kernel reads --
    ``"linear"`` for K, ``"swizzle4x4"`` for V (see the module docstring).
    128x4 was this module's own choice and never a TE constraint, so the TE
    backend is unaffected by the layout change: it still hands back logical
    ``meta["rowwise_scale_inv"]`` and the packing happens here.

    Two backends produce the same layout:
      * ``"te"``    - Transformer Engine's ``NVFP4Quantizer``.
      * ``"torch"`` - a dependency-free PyTorch implementation of the same
        recipe.  Use it when Transformer Engine is not installed.
      * ``"auto"``  - TE if importable, otherwise ``"torch"``.  Override with
        the ``MSA_NVFP4_QUANTIZER_BACKEND`` environment variable.

    Parameters
    ----------
    x : torch.Tensor
        CUDA BF16 or FP16 tensor, shaped ``[num_pages, Hkv, page_size, D]``
        (paged), ``[total_k, Hkv, D]`` or ``[total_k, D]`` (flat varlen).  The
        last dimension must be divisible by 16, and the flattened row
        dimension ``prod(x.shape[:-1])`` must also be divisible by 16.
    layout : str
        ``"linear"`` (K) or ``"swizzle4x4"`` (V).  Required -- there is no
        default, because silently picking the wrong one for V produces
        plausible-looking but wrong numbers.
    backend : str, optional
        ``"auto"``, ``"te"`` or ``"torch"``.

    Returns
    -------
    Nvfp4QuantizedTensor
        Packed FP4 data, packed block scales, global scale, and shape metadata
        needed by the KVFP4 attention kernel or by reference dequantization.
    """

    layout = _check_layout(layout)
    if not x.is_cuda:
        raise ValueError("NVFP4 quantization requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"x must be bf16 or fp16, got {x.dtype}")
    if x.ndim not in (2, 3, 4):
        raise ValueError(
            "x must be rank-4 [num_pages, Hkv, page_size, D], rank-3 "
            f"[total_k, Hkv, D] or rank-2 [total_k, D], got rank {x.ndim}"
        )
    if x.shape[-1] % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"last dimension must be divisible by {NVFP4_BLOCK_SIZE}, got {x.shape[-1]}"
        )

    rows = 1
    for dim in x.shape[:-1]:
        rows *= int(dim)
    if rows % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            "flattened row dimension must be divisible by "
            f"{NVFP4_BLOCK_SIZE}, got {rows}"
        )

    scale_cols = int(x.shape[-1]) // NVFP4_BLOCK_SIZE
    # Fail here rather than after a full quantization pass.
    _scale_region_grid(x.shape, layout)

    # An explicit argument wins; the environment only supplies the default, so
    # a caller that asks for a specific backend always gets it (or an error).
    backend = backend.lower()
    if backend == "auto":
        backend = os.environ.get("MSA_NVFP4_QUANTIZER_BACKEND", "auto").lower()
    if backend not in ("auto", "te", "torch"):
        raise ValueError(
            f"backend must be 'auto', 'te' or 'torch', got {backend!r}"
        )
    if backend == "auto":
        backend = "te" if te_nvfp4_quantizer_available() else "torch"
    if backend == "torch":
        return _quantize_bf16_to_nvfp4_torch(
            x, rows=rows, scale_cols=scale_cols, layout=layout
        )

    NVFP4Quantizer = _import_te_nvfp4_quantizer()
    quantizer = NVFP4Quantizer(rowwise=True, columnwise=False)
    qx = quantizer.quantize(x.contiguous())
    meta = qx.get_metadata()

    data = meta["rowwise_data"]
    if data.dtype is not torch.uint8:
        data = data.view(torch.uint8)
    logical_scale = meta["rowwise_scale_inv"]
    amax = meta["amax_rowwise"]
    scale = _pack_block_scale(
        logical_scale,
        rows=rows,
        scale_cols=scale_cols,
        original_shape=x.shape,
        layout=layout,
    )
    global_scale = nvfp4_global_scale_from_amax(amax).contiguous()

    return Nvfp4QuantizedTensor(
        data=data,
        scale=scale,
        global_scale=global_scale,
        logical_scale_shape=(rows, scale_cols),
        original_shape=tuple(int(v) for v in x.shape),
        layout=layout,
    )


def quantize_kv_bf16_to_nvfp4(
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[Nvfp4QuantizedTensor, Nvfp4QuantizedTensor]:
    """Quantize BF16/FP16 K and V tensors independently for KVFP4 attention.

    K is packed ``"linear"`` and V ``"swizzle4x4"``, which is exactly what the
    attention kernel reads and what vLLM's NVFP4 KV-cache store kernel writes.

    Parameters
    ----------
    k : torch.Tensor
        CUDA BF16 or FP16 K tensor.
    v : torch.Tensor
        CUDA BF16 or FP16 V tensor.

    Returns
    -------
    tuple[Nvfp4QuantizedTensor, Nvfp4QuantizedTensor]
        Quantized K and V tensors with independent scales.
    """

    return (
        quantize_bf16_to_nvfp4(k, layout="linear"),
        quantize_bf16_to_nvfp4(v, layout="swizzle4x4"),
    )


def dequantize_nvfp4_to_bf16(
    qx: Nvfp4QuantizedTensor,
    *,
    include_global_scale: bool = True,
) -> torch.Tensor:
    """Reference dequantization for validation.

    This mirrors the kernel contract:
      x = e2m1 * E4M3_block_scale_1x16 * FP32_global_scale

    The scale gather uses ``qx.layout``, so the result is layout-independent:
    the same logical block scales give the same BF16 output whether they were
    packed ``"linear"`` or ``"swizzle4x4"``.

    Parameters
    ----------
    qx : Nvfp4QuantizedTensor
        Quantized tensor returned by ``quantize_bf16_to_nvfp4``.  ``qx.scale``
        must be a contiguous packed container.
    include_global_scale : bool, optional
        If True, multiply by ``qx.global_scale`` after applying per-block
        scales.

    Returns
    -------
    torch.Tensor
        BF16 tensor with shape ``qx.original_shape``.
    """

    layout = _check_layout(qx.layout)
    data = qx.data if qx.data.dtype is torch.uint8 else qx.data.view(torch.uint8)
    if data.shape[-1] * 2 != qx.original_shape[-1]:
        raise ValueError(
            "packed data last dimension does not match original shape: "
            f"{data.shape[-1]} packed vs {qx.original_shape[-1]} logical"
        )

    rows, scale_cols = qx.logical_scale_shape
    logical_dim = int(qx.original_shape[-1])
    if scale_cols * NVFP4_BLOCK_SIZE != logical_dim:
        raise ValueError(
            "logical scale columns do not match original last dimension: "
            f"{scale_cols} scale cols vs dim {logical_dim}"
        )

    fp4_lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=data.device,
    )
    packed = data.reshape(rows, logical_dim // 2)
    lo = packed & 0x0F
    hi = packed >> 4
    values = torch.empty((rows, logical_dim), dtype=torch.float32, device=data.device)
    values[:, 0::2] = fp4_lut[lo.long()]
    values[:, 1::2] = fp4_lut[hi.long()]

    scale_u8 = _unpack_block_scale(
        qx.scale,
        rows=rows,
        scale_cols=scale_cols,
        original_shape=qx.original_shape,
        layout=layout,
    )
    scale = scale_u8.view(torch.float8_e4m3fn).to(torch.float32)
    scale = scale.repeat_interleave(NVFP4_BLOCK_SIZE, dim=1)
    out = values * scale
    if include_global_scale:
        global_scale = qx.global_scale.reshape(-1)[0].to(torch.float32)
        out = out * global_scale
    return out.reshape(qx.original_shape).to(torch.bfloat16)


def _example() -> None:
    device = torch.device("cuda")
    k = torch.randn(128, 2, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_q, v_q = quantize_kv_bf16_to_nvfp4(k, v)
    print("K FP4 data:", tuple(k_q.data.shape), k_q.data.dtype)
    print("K scale (linear):", tuple(k_q.scale.shape), k_q.scale.dtype)
    print("K global scale:", tuple(k_q.global_scale.shape), k_q.global_scale.dtype)
    print("V FP4 data:", tuple(v_q.data.shape), v_q.data.dtype)
    print("V scale (swizzle4x4):", tuple(v_q.scale.shape), v_q.scale.dtype)
    print("V global scale:", tuple(v_q.global_scale.shape), v_q.global_scale.dtype)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("quantize.py requires CUDA")
    _example()
