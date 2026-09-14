# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kaiming-init + block-scaled quantization for batched-GEMM test data.

Fused-activation GEMM inputs are generated as fp32 Kaiming-uniform data and then
quantized to the kernel's input dtype, instead of drawing raw random storage +
random scale factors. Random scale factors produce wildly varying per-element
magnitudes whose products catastrophically cancel in the accumulator, which the
SwiGLU clamp then amplifies into sparse ref-check failures on large configs.
Kaiming-then-quantize keeps the accumulator well-behaved and in the kernel's
clamp regime.

Supports every block-scaled input dtype that flows through the reference's MMA
branches:

* element dtype: packed E2M1 (FP4) or E4M3 (FP8);
* scale-factor dtype: E4M3 (NVFP4) or UE8M0 (MX), the latter being a strict
  power of two.

Quantization mechanics live here so they can be reused across dtypes and stay
separate from the (large) reference-check driver.
"""

import math

import torch

from .batched_gemm_config import DType, SfLayout

# Max representable magnitudes / powers-of-two per element dtype.
MAX_E2M1_VAL = 6.0
MAX_E4M3_VAL = 448.0
MAX_E2M1_POW2 = 4.0
MAX_E4M3_POW2 = 256.0

_E2M1_KINDS = (int(DType.E2M1), int(DType.MXE2M1))


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def is_e2m1_kind(dtype_kind: int) -> bool:
    """True for packed 4-bit E2M1 element storage (NVFP4 / MXFP4)."""
    return dtype_kind in _E2M1_KINDS


def kaiming_uniform_tensor(
    shape: tuple[int, ...],
    *,
    fan_in: int,
    dtype: torch.dtype,
    device: str,
    relu: bool = False,
) -> torch.Tensor:
    """Kaiming-uniform init: symmetric bound based on fan-in (1/sqrt(fan_in))."""
    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive, got {fan_in}")
    bound = 1.0 / math.sqrt(fan_in)
    tensor = torch.empty(shape, dtype=torch.float32, device=device).uniform_(
        -bound, bound
    )
    if relu:
        tensor = torch.clamp_min(tensor, 0)
    return tensor.to(dtype)


def e2m1_round_to_codes(values: torch.Tensor) -> torch.Tensor:
    """Round fp32 ``values`` to the nearest E2M1 nibble code (uint8 in 0..15).

    Done via ``bucketize`` on magnitudes (O(N) memory) rather than a 16-way
    broadcast+argmin, which would materialize a ``[..., 16]`` tensor far larger
    than the input and risk OOM on full-size weights.
    """
    # Positive E2M1 magnitudes (codes 0..7); codes 8..15 are their negatives.
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=values.device,
    )
    midpoints = (magnitudes[1:] + magnitudes[:-1]) * 0.5
    idx = torch.bucketize(values.abs(), midpoints).to(torch.uint8)
    return torch.where(values < 0, idx + 8, idx)


def pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 nibble codes ``[.., K]`` into bytes ``[.., K/2]``.

    Even logical index -> low nibble, odd logical index -> high nibble (inverse
    of the reference's ``_unpack_fp4_e2m1``).
    """
    lo = codes[..., 0::2] & 0x0F
    hi = codes[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def quantize_block_scaled(
    data: torch.Tensor,
    *,
    dtype_kind: int,
    uses_mx: bool,
    sf_vec_size: int,
):
    """Block-scale-quantize fp32 ``data`` ``[rows, K]`` to a kernel input dtype.

    Returns ``(storage_uint8, sf_q)`` where ``storage_uint8`` is ``[rows, K/2]``
    for packed E2M1 or ``[rows, K]`` for E4M3, and ``sf_q`` is the per-block
    ``[rows, K/sf_vec_size]`` scale factor (``float8_e8m0fnu`` for MX, else
    ``float8_e4m3fn``).

    Per-block scale factor: NVFP4 uses ``sf = amax / 6``; MX uses
    ``sf = amaxPow2 / maxPow2`` with ``amaxPow2`` the largest power of two
    <= ``amax`` (so the UE8M0 scale is exact). The caller is expected to have
    already scaled ``data`` into the kernel's accumulator regime (see the data
    scale in the orchestrator); no extra global SF scale is applied, matching the
    reference dequant ``value = element * sf`` which carries no global divisor.
    """
    rows, k = data.shape
    nblk = k // sf_vec_size
    blocks = data.reshape(rows, nblk, sf_vec_size)
    amax = blocks.abs().amax(dim=-1)
    e2m1 = is_e2m1_kind(dtype_kind)

    if uses_mx:
        # UE8M0 scale: largest power of two <= amax, divided by the largest
        # power of two the element dtype can represent. Exact (no quant error).
        pow2_max = MAX_E2M1_POW2 if e2m1 else MAX_E4M3_POW2
        amax_pow2 = torch.where(
            amax > 0,
            torch.exp2(torch.floor(torch.log2(amax))),
            torch.zeros_like(amax),
        )
        sf_q = (amax_pow2 / pow2_max).to(torch.float8_e8m0fnu)
    else:
        elem_max = MAX_E2M1_VAL if e2m1 else MAX_E4M3_VAL
        sf_q = (amax / elem_max).to(torch.float8_e4m3fn)

    sf_back = sf_q.float()
    quant_scale = torch.where(sf_back > 0, 1.0 / sf_back, torch.zeros_like(sf_back))
    scaled = blocks * quant_scale.unsqueeze(-1)

    if e2m1:
        # e2m1_round_to_codes saturates large magnitudes to the top code (6).
        codes = e2m1_round_to_codes(scaled).reshape(rows, k)
        storage = pack_e2m1_codes(codes)
    else:
        # MX uses an E4M3 power-of-two max of 256, so a block's quantized
        # elements can reach 256*amax/amax_pow2 < 512 -- above E4M3's max of 448.
        # A saturating cast is required; torch's fp32->e4m3 maps >448 to NaN, so
        # clamp first.
        storage = (
            scaled.reshape(rows, k)
            .clamp(-MAX_E4M3_VAL, MAX_E4M3_VAL)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
    return storage, sf_q


def sf_offsets_2d(
    sf_rows: torch.Tensor,
    *,
    nblk: int,
    layout: int,
    data_blk_cols: int,
) -> torch.Tensor:
    """Vectorized scale-factor byte offsets for a set of SF row indices.

    ``sf_rows`` is a 1-D long tensor of SF row coordinates. Returns a
    ``[len(sf_rows), nblk]`` long tensor of byte offsets into a flat SF buffer,
    matching the per-row offsets used by the reference's dequant readers.
    """
    r = sf_rows.view(-1, 1)
    j = torch.arange(nblk, dtype=torch.long, device=sf_rows.device).view(1, -1)
    if layout == int(SfLayout.LINEAR):
        return r * data_blk_cols + j
    if layout == int(SfLayout.R8c4):
        rows_per_block, cols_per_block, bytes_per_block = 8, 4, 32
        sf_row = r % rows_per_block
    elif layout == int(SfLayout.R128c4):
        rows_per_block, cols_per_block, bytes_per_block = 128, 4, 512
        sf_row = (r % 32) * 4 + (r % rows_per_block) // 32
    else:
        raise ValueError(f"Unsupported SF layout for quantization: {layout}")
    sf_blk_row = r // rows_per_block
    sf_blk_col = j // cols_per_block
    sf_blk_idx = sf_blk_row * (data_blk_cols // cols_per_block) + sf_blk_col
    sf_col = j % cols_per_block
    return sf_blk_idx * bytes_per_block + sf_row * cols_per_block + sf_col


def build_sf_buffer(
    sf_q: torch.Tensor,
    *,
    total_bytes: int,
    layout: int,
    data_blk_cols: int,
    device: str,
    sf_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter per-row scale factors into a flat uint8 SF buffer.

    ``sf_q`` is ``[num_rows, nblk]`` (E4M3 or UE8M0). ``sf_rows`` maps each
    physical row to its SF row coordinate (e.g. expanded token index); it
    defaults to ``arange(num_rows)``.
    """
    num_rows, nblk = sf_q.shape
    if sf_rows is None:
        sf_rows = torch.arange(num_rows, dtype=torch.long, device=device)
    offsets = sf_offsets_2d(
        sf_rows,
        nblk=nblk,
        layout=layout,
        data_blk_cols=data_blk_cols,
    )
    buf = torch.zeros((total_bytes,), dtype=torch.uint8, device=device)
    buf[offsets.reshape(-1)] = sf_q.view(torch.uint8).reshape(-1)
    return buf
