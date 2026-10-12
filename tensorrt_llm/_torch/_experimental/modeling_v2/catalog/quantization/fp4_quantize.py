# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization: packed e2m1 data plus one e4m3 scale per 16 elements."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

_E4M3_MAX = 448.0
_E2M1_MAX = 6.0

#: Midpoints between consecutive e2m1 magnitudes. A value landing exactly on one
#: is a rounding tie; `_TIE_UP[i]` is True when rounding *up* yields the even
#: code, which is the direction round-to-nearest-even takes.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_TIE_UP = (False, True, False, True, False, True, False)

#: The kernel builds its per-block output scale through two `rcp.approx.ftz.f32`
#: reciprocals, each ~2^-23 relative. A value within a few fp32 ulps of a
#: midpoint can therefore round to either neighbour, and no reference can say
#: which. Everything further out than this must match bit for bit. Largest
#: deviation observed on this machine: 7.9e-8 relative, ~12x inside the window.
TIE_WINDOW = 2.0**-20

#: MXFP4 mode (`sf_vec_size=32`, `sf_use_ue8m0=True`): the reference floors each
#: 32-block's max at `6 * 2^-126`, so its smallest scale is 2^-126 (UE8M0 byte
#: 0x01). The op has no floor; see `reference`.
_MX_AMAX_FLOOR = 6 * 2.0**-126
#: The reference multiplies the block max by this fp32 constant and takes
#: ceil(log2(.)) from the product's IEEE bits.
_INV6_F32 = 1.0 / 6.0


#: DeepSeek-V4.1's routed-expert input: bf16 NVFP4 with a static calibrated `[1]` scale.
_V41_MOE = dict(dtype=torch.bfloat16, vec=16, gs="static")
#: DeepSeek-V4.1's index K / index Q quantizer: bf16 MXFP4, linear, no global scale, and
#: a power-of-two magnitude per 32-block so the UE8M0 scales spread.
_V41_INDEX = dict(dtype=torch.bfloat16, vec=32, swizzled=False, ue8m0=True, gs=None, block_mag=True)


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class Fp4Quantize(OpWrapper):
    """Quantize to FP4: `(packed e2m1 data [..., K/2] uint8, scale bytes, 1-D)`.

    One scale per `sf_vec_size` contiguous elements of the last dim. NVFP4 is
    `sf_vec_size=16` with `sf_use_ue8m0=False`, which makes the scale an e4m3
    byte derived from the block maximum and the caller's `global_scale`. MXFP4
    is `sf_vec_size=32` with `sf_use_ue8m0=True`: the scale is a UE8M0 power of
    two of the block maximum alone, and `global_scale` is not read. Any other
    pairing raises.

    `is_sf_swizzled_layout` selects the 128x4 swizzled scale order (True) or the
    row-major linear order (False); the buffer is 1-D either way, so the layout
    is something a consumer has to be told, not something its shape shows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The first five cells are deepseek-r1's. It reaches this op from both layouts: the
    # swizzled one feeding nvfp4_gemm in the dense and shared-expert MLPs, the
    # linear one feeding the routed-expert path. Widths are its hidden size and
    # its intermediate.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, hidden 7168, swizzled scales -- the dense MLP's input",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        Cell(
            why="prefill chunk, hidden 7168, swizzled",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        Cell(
            why="linear scales -- the routed-expert path asks for this order",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="the activation width, 2 x moe_intermediate, between the two GEMMs",
            spec=dict(shape=(512, 4096), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        Cell(
            why="a row count that is not a multiple of the swizzle's 128-row tile",
            spec=dict(shape=(37, 7168), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        # deepseek-v4.1 routed-expert input: NVFP4 at its hidden 5120, with the
        # layer's static `[1]` g1 = 1 / input_scale calibrated on another batch
        # (`gs="static"`). The target asks for linear scales; both layouts ran.
        # Rows 1..8192 are swept by test_v41_moe_input_k5120_every_row_count and
        # the prefix test; these cells are the sweep's ends and the grid edge.
        Cell(
            why="deepseek-v4.1 MoE in K5120 linear: 1 row, static g1 [1]",
            spec=dict(shape=(1, 5120), swizzled=False, **_V41_MOE),
        ),
        Cell(
            why="deepseek-v4.1 MoE in K5120 linear: 609 rows, past the launcher's 608-row grid cap "
            "(152 SMs x 4 blocks), so rows go through the grid-stride loop",
            spec=dict(shape=(609, 5120), swizzled=False, **_V41_MOE),
        ),
        Cell(
            why="deepseek-v4.1 MoE in K5120 linear: 8192 rows, the target's MoE chunk size",
            spec=dict(shape=(8192, 5120), swizzled=False, **_V41_MOE),
        ),
        Cell(
            why="deepseek-v4.1 MoE in K5120 swizzled: 1 row, 127 rows of padding",
            spec=dict(shape=(1, 5120), swizzled=True, **_V41_MOE),
        ),
        Cell(
            why="deepseek-v4.1 MoE in K5120 swizzled: 8192 rows",
            spec=dict(shape=(8192, 5120), swizzled=True, **_V41_MOE),
        ),
        # deepseek-v4.1 index K [T, 128] and index Q [T, 32, 128]: MXFP4, linear,
        # global_scale None -- the checkpoint's `fp4_act_quant(x, 32)`. Each
        # 32-block gets its own power-of-two magnitude in 2^-20..2^20
        # (`block_mag=True`) so the UE8M0 scales spread. Rows 1..32768 are swept by
        # the prefix test; these cells are the sweep's ends and the grid edges.
        Cell(
            why="deepseek-v4.1 index K MXFP4 [1, 128]",
            spec=dict(shape=(1, 128), **_V41_INDEX),
        ),
        Cell(
            why="deepseek-v4.1 index K MXFP4 [19457, 128]: past the launcher's 19456-row grid cap "
            "(152 SMs x 128 blocks) at K 128",
            spec=dict(shape=(19457, 128), **_V41_INDEX),
        ),
        Cell(
            why="deepseek-v4.1 index K MXFP4 [32768, 128]: the certified row maximum",
            spec=dict(shape=(32768, 128), **_V41_INDEX),
        ),
        Cell(
            why="deepseek-v4.1 index Q MXFP4 [1, 32, 128]: rank 3, leading dims collapse in sf",
            spec=dict(shape=(1, 32, 128), **_V41_INDEX),
        ),
        Cell(
            why="deepseek-v4.1 index Q MXFP4 [609, 32, 128]: 19488 rows, past the 19456-row grid cap",
            spec=dict(shape=(609, 32, 128), **_V41_INDEX),
        ),
        Cell(
            why="deepseek-v4.1 index Q MXFP4 [32768, 32, 128]: the certified token maximum",
            spec=dict(shape=(32768, 32, 128), **_V41_INDEX),
        ),
    )

    note = """
    `sf_use_ue8m0=True` selects power-of-two scales instead of e4m3. It is
    certified only in the DeepSeek-V4.1 form below: `sf_vec_size=32`, bf16,
    `K = 128`, linear layout.

    DeepSeek-V4.1: three call forms, no metadata. (1) The routed-expert input
    `[r, 5120]` bf16, NVFP4, linear, with the layer's static `[1]` fp32 scale
    `g1 = 1 / max_e max(w1.input_scale[e], w3.input_scale[e])` -- the reciprocal
    of the checkpoint's `input_scale`; passing `input_scale` itself is accepted
    and zeroes most scale bytes. The target splits the MoE rows into 8192-row
    chunks and hands each over as a contiguous row view at an offset of a
    multiple of 8192 rows. (2) Index K `[r, 128]` and (3) index Q `[r, 32, 128]`,
    `fp4_quantize(x, None, 32, True, False)`, bit for bit the checkpoint's
    `fp4_act_quant(x, 32)` with no tie window; `data` feeds
    `indexer_k_cache_scatter_op` as `k_data` and `sf.view(torch.int32).view(T, 1)`
    as `k_scale`. Certified eagerly at every row count 1..8192 for (1) (both
    layouts) and 1..32768 for (2) and (3), each prefix bitwise equal to the
    longest call, one kernel instantiation per mode at every probed row count;
    the 8192-row chunk views of a 65536-row input; and the three calls captured
    together in one CUDA graph at n in {1..32, 64, 96, 128, 1024, 8192}, and as a
    whole decode step at every graph batch G of {1..32, 40..128 step 8, 192, 256}
    with next_n 1 and 6 (rows G, 3G, 6G, 4G, 24G) -- a replay reads the inputs and
    `global_scale` from device memory and rewrites the same two output buffers.
    Hazards: V4.1's own g1 lands in the NVFP4 tie regime (0.19% near-ties,
    0.033% resolved the other way, one adjacent code each), and it crosses the
    `g * vecmax / 6 <= 448` bound on real inputs (column block 153 of the BOS and
    a few other rows in layers 20 and 28-37, up to 6.1x); the op clamps those
    blocks to 448 silently. In MXFP4 mode a block whose max is at or below
    `3 * 2^-126` (in practice an all-zero padding row) gets scale byte 0x00 where
    `fp4_act_quant` writes 0x01; both decode to zero. Not certified for V4.1:
    MXFP4 with float16 input, the swizzled layout, `K != 128`, non-finite input
    or zero rows; either mode above its row maximum in one call; CUDA-graph
    capture at row counts outside the sets above or of any other call form.
    """

    def raw_call(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fp4_quantize(
            input, global_scale, sf_vec_size, sf_use_ue8m0, is_sf_swizzled_layout
        )

    def reference(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native-torch NVFP4, with the scales in the requested layout.

        Per block: `vecmax = max|x|`, the scale is `e4m3(global_scale * vecmax /
        6)`, and the data is the input times `global_scale / scale` rounded to
        e2m1. torch's fp32 -> e4m3 cast emits NaN above 464 where the kernel's
        saturates, so the scale is clamped first -- between 448 and 464 both
        round to 448 anyway.

        With `sf_use_ue8m0=True` (MXFP4) the reference is DeepSeek-V4.1's
        `fp4_act_quant(x, 32)` instead, and `global_scale` is ignored: per
        32-block, `s = 2^ceil(log2(max(vecmax, 6 * 2^-126) * fp32(1/6)))`, the
        scale byte is `log2(s) + 127`, and the data is `x / s` (exact, `s` is a
        power of two) rounded to e2m1, ties to even. `data` keeps the input's
        leading dims. The op has no `6 * 2^-126` floor: a block whose max is at
        or below `3 * 2^-126` gets scale byte 0x00 and signed-zero data, where
        this writes 0x01 (and, for a max above 2^-128, nonzero codes). Those
        are the only blocks where the two differ.
        """
        if sf_use_ue8m0:
            packed, sf = self._mx_quantize(input, sf_vec_size)
        else:
            packed, sf, _ = self._quantize(input, global_scale, sf_vec_size)
        if not is_sf_swizzled_layout:
            return packed, sf.reshape(-1)
        return packed, self._swizzle(sf)

    def scaled(
        self, input: torch.Tensor, global_scale: Optional[torch.Tensor], sf_vec_size: int
    ) -> torch.Tensor:
        """The pre-rounding product, which is what `near_tie` reads.

        Exposed because a caller cannot reconstruct it from the op's outputs:
        rounding is the step that loses it.
        """
        return self._quantize(input, global_scale, sf_vec_size)[2]

    @staticmethod
    def near_tie(scaled: torch.Tensor) -> torch.Tensor:
        """Elements whose rounding this entry refuses to predict.

        Inside `TIE_WINDOW` of an e2m1 midpoint the kernel's approximate
        reciprocals decide the direction, so a code that differs there is not a
        defect. Outside it, a difference is.
        """
        mid = torch.tensor(_E2M1_MIDPOINTS, device=scaled.device, dtype=torch.float32)
        a = scaled.abs().unsqueeze(-1)
        return ((a - mid).abs() <= TIE_WINDOW * a).any(-1)

    def is_valid(
        self,
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor],
        sf_vec_size: int,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
    ) -> None:
        # The kernel loads one scalar from global_scale and ignores every
        # element past the first, so a per-token [num_tokens] tensor is silently
        # applied as global_scale[0] to every row -- observed to return a
        # valid-looking result, never to raise.
        assert global_scale is None or global_scale.numel() == 1, (
            "global_scale must hold exactly one element; extra elements are "
            "silently ignored and global_scale[0] is applied to every row"
        )
        assert input.shape[-1] % sf_vec_size == 0, (
            f"the quantized dim must be a whole number of {sf_vec_size}-element "
            f"blocks; got {input.shape[-1]}"
        )
        assert input.is_contiguous(), "input must be contiguous; the kernel reads it flat"

    # ── the arithmetic, shared by reference() and scaled() ──────────────────

    def _quantize(
        self, input: torch.Tensor, global_scale: Optional[torch.Tensor], vec: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gs = 1.0 if global_scale is None else global_scale.float().reshape(())
        k = input.shape[-1]
        rows = input.numel() // k
        flat = input.reshape(rows, k)
        vecmax = flat.abs().reshape(rows, k // vec, vec).amax(-1).float()
        sf = (gs * (vecmax / _E2M1_MAX)).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
        out_scale = torch.where(vecmax != 0, gs / sf.float(), torch.zeros_like(vecmax))
        scaled = (flat.float().reshape(rows, k // vec, vec) * out_scale.unsqueeze(-1)).reshape(
            rows, k
        )
        codes = self._e2m1_codes(scaled)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return packed, sf.view(torch.uint8), scaled

    @staticmethod
    def _mx_quantize(input: torch.Tensor, vec: int) -> tuple[torch.Tensor, torch.Tensor]:
        """MXFP4: (packed data [..., K/2], UE8M0 scale bytes [rows, K/vec])."""
        k = input.shape[-1]
        rows = input.numel() // k
        xf = input.reshape(rows, k // vec, vec).float()
        amax = xf.abs().amax(-1).clamp_min(_MX_AMAX_FLOOR)
        # ceil(log2(amax / 6)) from the fp32 product's bits: the exponent, plus one
        # unless the mantissa is zero (an exact power of two).
        bits = (amax * torch.tensor(_INV6_F32, dtype=torch.float32, device=amax.device)).view(
            torch.int32
        )
        e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        s = ((e + 127) << 23).view(torch.float32)
        q = (xf / s.unsqueeze(-1)).clamp(-_E2M1_MAX, _E2M1_MAX)
        # e2m1 steps are 0.5 below 2, 1 on [2, 4) and 2 on [4, 6]; torch.round is
        # half-to-even on each binade's grid.
        a = q.abs()
        r = torch.where(
            a < 2,
            torch.round(a * 2) / 2,
            torch.where(a < 4, torch.round(a), torch.round(a / 2) * 2),
        )
        # value -> code: {0, 0.5, 1, 1.5} -> 2r; {2, 3} -> r + 2; {4, 6} -> r/2 + 4
        code = torch.where(r < 2, r * 2, torch.where(r < 4, r + 2, r / 2 + 4)).to(torch.uint8)
        # The sign comes from the input, so -0.0 and a negative lane rounding to 0 are code 8.
        codes = (code | (torch.signbit(xf).to(torch.uint8) << 3)).reshape(rows, k)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return packed.reshape(*input.shape[:-1], k // 2), (e + 127).to(torch.uint8)

    @staticmethod
    def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
        """fp32 -> 4-bit e2m1 codes (bit 3 = sign), round-to-nearest-even."""
        mid = torch.tensor(_E2M1_MIDPOINTS, device=v.device, dtype=torch.float32)
        tie_up = torch.tensor(_TIE_UP, device=v.device, dtype=torch.bool)
        a = v.abs().unsqueeze(-1)
        greater = (a > mid).sum(-1)  # strict: ties provisionally round down
        up = ((a == mid) & tie_up).any(-1)  # ties whose even side is up
        code = (greater + up).to(torch.uint8)
        negative = (v < 0) | ((v == 0) & torch.signbit(v))
        return code | (negative.to(torch.uint8) << 3)

    @staticmethod
    def _swizzle(sf: torch.Tensor) -> torch.Tensor:
        """Scatter [rows, cols] scale bytes into the 128x4-swizzled buffer."""
        rows, cols = sf.shape
        padded_cols, padded_rows = _pad_up(cols, 4), _pad_up(rows, 128)
        r = torch.arange(rows, device=sf.device).view(-1, 1)
        c = torch.arange(cols, device=sf.device).view(1, -1)
        index = (
            (c % 4)
            + (c // 4) * (4 * 128)
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + (r // 128) * (128 * padded_cols)
        )
        out = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8, device=sf.device)
        out[index.reshape(-1)] = sf.reshape(-1)
        return out


fp4_quantize = Fp4Quantize()
