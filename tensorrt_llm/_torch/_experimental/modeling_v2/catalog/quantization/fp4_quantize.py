# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization: packed e2m1 data plus one e4m3 scale per 16 elements."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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


#: Step-5-Preview's NVFP4 MoE input regimes, read from the checkpoint: the
#: gate/up `input_scale` of the layer (identical across its 352 experts and
#: between gate and up), and the RMS of its `post_attention_layernorm` weight.
#: Layer 3 holds the smallest input_scale of layers 3-87 (the largest global
#: scale, 4327.85; calibrated amax 0.621), layer 81 the largest (global scale
#: 145.30; calibrated amax 18.5).
_STEP5_LAYER3 = dict(input_scale=0.00023106166918296367, sigma=0.2348, outlier=1.0, tiny=4e-6)
_STEP5_LAYER81 = dict(input_scale=0.006882440764456987, sigma=0.0248, outlier=8.0, tiny=1e-4)


def _pad_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class Fp4Quantize(OpWrapper):
    """Quantize to FP4: `(packed e2m1 data [..., K/2] uint8, scale bytes, 1-D)`.

    One scale per `sf_vec_size` contiguous elements of the last dim. NVFP4 --
    the only form any shipped target uses -- is `sf_vec_size=16` with
    `sf_use_ue8m0=False`, which makes the scale an e4m3 byte derived from the
    block maximum and the caller's `global_scale`.

    `is_sf_swizzled_layout` selects the 128x4 swizzled scale order (True) or the
    row-major linear order (False); the buffer is 1-D either way, so the layout
    is something a consumer has to be told, not something its shape shows.
    """

    # The Step-5 cells below were driven on sm_103 only; every other cell was
    # certified on both sm_100 and sm_103.
    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # deepseek-r1 reaches this op from both layouts: the swizzled one feeding
    # nvfp4_gemm in the dense and shared-expert MLPs, the linear one feeding the
    # routed-expert path, at its hidden size and its intermediate. qwen3.8
    # (qwen4_exp) calls it only for its routed experts: linear scales at hidden
    # 2560. A linear buffer has no row padding, so at a row count that is not a
    # multiple of 128 it is shorter than the swizzled one; at a multiple of 128
    # the two are the same length and only the cells' byte order tells them
    # apart.
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
        Cell(
            why="deepseek default at attention TP8: the shared expert's down input, 2048/8 wide",
            spec=dict(shape=(16, 256), dtype=torch.bfloat16, vec=16, swizzled=True),
        ),
        Cell(
            why="qwen3.8 decode: 1 token, hidden 2560, linear -- its routed-expert input",
            spec=dict(shape=(1, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="qwen3.8, 7 rows at hidden 2560, linear: several rows, no row padding",
            spec=dict(shape=(7, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        Cell(
            why="qwen3.8 prefill at max_num_tokens 8192, hidden 2560, linear",
            spec=dict(shape=(8192, 2560), dtype=torch.bfloat16, vec=16, swizzled=False),
        ),
        # Step-5-Preview quantizes the routed experts' input (hidden 4096) with
        # swizzled scales before the CUTLASS fused MoE, NVFP4 layers 3-87. The
        # global scale is the engine's `fc31_input_scale`: the reciprocal of the
        # checkpoint's per-layer gate/up `input_scale`, passed as a 0-dim fp32
        # tensor. Unlike the cells above, it is not derived from the input, so
        # blocks above the calibrated amax (2688 * input_scale) saturate the
        # e4m3 scale at 448. `sigma` is the layer's post-attention RMSNorm
        # weight RMS (the norm output has unit RMS before the weight), `outlier`
        # the std of 8 outlier channels that push blocks past saturation, and
        # `tiny` the std of one 16-channel block sized to land a subnormal
        # e4m3 scale.
        Cell(
            why="step5 decode, 1 row x 4096, layer-3 input scale, swizzled",
            spec=dict(
                shape=(1, 4096),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=True,
                **_STEP5_LAYER3,
            ),
        ),
        Cell(
            why="step5 decode batch, 128 rows x 4096 (max_batch_size), layer 3",
            spec=dict(
                shape=(128, 4096),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=True,
                **_STEP5_LAYER3,
            ),
        ),
        Cell(
            why="step5 prefill chunk, 8192 x 4096 (max_num_tokens), layer 3",
            spec=dict(
                shape=(8192, 4096),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=True,
                **_STEP5_LAYER3,
            ),
        ),
        Cell(
            why="step5 dep4 gathered decode, 512 x 4096, layer-81 scale (largest)",
            spec=dict(
                shape=(512, 4096),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=True,
                **_STEP5_LAYER81,
            ),
        ),
    )

    note = """
    `sf_use_ue8m0=True` selects power-of-two scales instead of e4m3. No shipped
    target uses it and this entry does not certify it.

    Step-5-Preview call site: the NVFP4 routed-expert input of layers 3-87,
    `fp4_quantize(x [rows, 4096] bf16, fc31_input_scale (0-dim fp32), 16, False,
    True)`, ahead of the CUTLASS fused MoE, in prefill (8192-row chunks) and in
    decode (b rows at tp4/tep4, 4b at dep4 after the all-gather; CUDA-graph
    captured). The global scale is 1 / input_scale from the checkpoint, so the
    op meets saturated block scales (block max above 2688 * input_scale: the
    e4m3 scale clamps to 448 and the codes to +-6) and, on near-zero channel
    blocks, subnormal e4m3 scales; the Step-5 cells assert both are reached and
    match the reference bit for bit away from the e2m1 tie window. Measured on
    sm_103: the layer-3 cells saturate 358689 of 2097152 blocks at 8192 rows
    and the layer-81 cell 102 of 131072, every cell has one subnormal scale per
    row, scales match exactly, 0.135% to 0.244% of elements sit in the tie
    window, and no code differs, inside it or outside. A block whose scale
    underflows to 0 dequantizes to 0 whatever its codes; no cell holds one.
    Capture and replay at 128 rows reproduce the eager result bit for bit.

    Uncertified for Step-5: every Step-5 cell on sm_100 (driven on sm_103
    only); linear (`is_sf_swizzled_layout=False`) scales at
    width 4096 (the trtllm-gen MoE backends ask for that order; the Step-5 MoE
    entry is CUTLASS), row counts other than 1, 128, 512 and 8192 at width
    4096 (a gathered prefill set larger than one 8192-row chunk included), and
    global scales between the layer-3 and layer-81 ends of the checkpoint's
    range.

    Hard limits: the input is addressed flat; at width 4096, 2^31 elements is
    524288 rows, 64x the envelope's largest row count (8192 per chunk, 512
    gathered decode), so no cell probes it. The swizzled scale buffer is
    pad128(rows) * K/16 bytes.
    """

    def __call__(
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
        """
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
