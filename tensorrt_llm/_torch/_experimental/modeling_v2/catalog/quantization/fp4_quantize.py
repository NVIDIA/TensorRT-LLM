# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization: packed e2m1 data plus one e4m3 scale per 16 elements.

The Mistral Large 4 NVFP4 targets quantize their routed experts' input with it, linear scales at
hidden 6144, under each layer's static checkpoint global scale: at tep4 each rank's own tokens,
at dep4 the token set gathered from the four ranks, at dep8 the token set gathered from the eight
ranks, a chunk of at most 32768 rows at a time. Sourcing tier 1:
``torch.ops.trtllm.fp4_quantize``, registered from C++
(``cpp/tensorrt_llm/thop/fp4Quantize.cpp``).
"""

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
#: deviation observed on this machine: 7.9e-8 relative, ~12x inside the window
#: (5.96e-8 across Mistral Large 4's 71 static global scales x every bf16 block max).
TIE_WINDOW = 2.0**-20

#: Mistral-Large-4-1T-A52B NVFP4's routed-expert `input_scale` of w1, MoE layers
#: 1..71 in order: one value per layer for all 128 experts and equal to w3's
#: (read on every expert). They are bf16 values stored as F32; the target's
#: static global scale is their fp32 reciprocal, 23.0112 (layer 65) to 9039.45
#: (layer 3).
_MISTRAL_LARGE4_INPUT_SCALES = tuple(
    float.fromhex(h)
    for h in (
        "0x1.d6p-13 0x1.36p-13 0x1.d0p-14 0x1.24p-11 0x1.3cp-10 0x1.b4p-13 0x1.0ap-11 0x1.34p-11 "
        "0x1.20p-12 0x1.b6p-10 0x1.b0p-11 0x1.eap-11 0x1.64p-8 0x1.eep-12 0x1.d2p-12 0x1.a8p-12 "
        "0x1.04p-11 0x1.70p-9 0x1.c4p-12 0x1.c6p-10 0x1.32p-10 0x1.0ep-11 0x1.1cp-11 0x1.46p-11 "
        "0x1.d0p-11 0x1.50p-9 0x1.3cp-9 0x1.0ep-7 0x1.40p-9 0x1.b0p-10 0x1.78p-9 0x1.fcp-9 "
        "0x1.aep-8 0x1.a8p-7 0x1.10p-8 0x1.6ep-8 0x1.80p-8 0x1.c4p-8 0x1.bap-8 0x1.e4p-8 "
        "0x1.18p-7 0x1.48p-7 0x1.4ap-7 0x1.68p-7 0x1.64p-7 0x1.4ep-7 0x1.3ep-7 0x1.52p-7 "
        "0x1.58p-7 0x1.80p-7 0x1.8ap-7 0x1.d0p-7 0x1.0cp-6 0x1.1cp-6 0x1.16p-6 0x1.2ap-6 "
        "0x1.36p-6 0x1.8cp-6 0x1.90p-6 0x1.12p-5 0x1.16p-5 0x1.4cp-5 0x1.38p-5 0x1.4ep-5 "
        "0x1.64p-5 0x1.10p-5 0x1.fcp-6 0x1.06p-5 0x1.cap-6 0x1.86p-6 0x1.54p-6"
    ).split()
)
_MISTRAL_LARGE4_LAYER3 = _MISTRAL_LARGE4_INPUT_SCALES[3 - 1]  # global scale 9039.45, the largest
_MISTRAL_LARGE4_LAYER65 = _MISTRAL_LARGE4_INPUT_SCALES[65 - 1]  # global scale 23.0112, the smallest

#: Mistral Large 4 dep4: four ranks, max_num_tokens 32768 each, so the gathered
#: token set is 4 x T_max rows, at most 131072.
_DEP4_RANKS = 4
_DEP4_GATHERED_MAX = _DEP4_RANKS * 32768
#: Mistral Large 4 dep8: eight ranks over two nodes, so the decode graphs gather
#: 8 x b rows for the captured per-rank batch b in 1..16.
_DEP8_RANKS = 8
#: Rows one pass of the grid covers at hidden 6144 on this GB300: 512-thread
#: CTAs, 4 per SM on every SM. Past 608 rows the launch no longer changes with
#: the row count; CTA b walks rows b, b + 608, ...
_GRID_ROWS = 4 * 152
#: Gathered row counts past 32768 driven in pre-merge: either side of every
#: grid wrap (each CTA's row count steps there) and of every rank segment
#: boundary, and every row count of the last grid period.
_DEP4_SWEEP_EDGES = tuple(
    sorted(
        {
            m
            for edge in range(_GRID_ROWS, _DEP4_GATHERED_MAX + 1, _GRID_ROWS)
            for m in (edge - 4, edge, edge + 4)
            if 32768 < m <= _DEP4_GATHERED_MAX
        }
        | {
            m
            for edge in range(32768, _DEP4_GATHERED_MAX + 1, 32768)
            for m in (edge - 4, edge, edge + 4)
            if 32768 < m <= _DEP4_GATHERED_MAX
        }
        | set(range(_DEP4_GATHERED_MAX - _GRID_ROWS + 1, _DEP4_GATHERED_MAX + 1))
    )
)


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

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1 reaches this op from both layouts: the swizzled one feeding
    # nvfp4_gemm in the dense and shared-expert MLPs, the linear one feeding the
    # routed-expert path, at its hidden size and its intermediate. qwen3.8
    # (qwen4_exp) calls it only for its routed experts: linear scales at hidden
    # 2560. A linear buffer has no row padding, so at a row count that is not a
    # multiple of 128 it is shorter than the swizzled one; at a multiple of 128
    # the two are the same length and only the cells' byte order tells them
    # apart. Mistral Large 4 (NVFP4, tep4) calls it for its routed experts'
    # input, linear scales at hidden 6144, under each layer's static global
    # scale 1 / input_scale(w1) rather than the canonical 448*6/amax of the
    # input: its cells carry `input_scale`, and their inputs are drawn against
    # the amax that scale was calibrated for (448*6 * input_scale), times
    # `input_amax_over_calibrated`.
    # Mistral Large 4 at dep4 (attention data parallel, experts over 4 ranks) calls
    # it once per MoE layer on the token set gathered from the four ranks: each
    # rank's rows zero-padded to T_max, the largest rank's token count, so
    # 4 x T_max rows, up to 4 x max_num_tokens 32768 = 131072 eager and
    # 4 x 1/2/4/8/16 = 4/8/16/32/64 in the decode graphs, where every rank is
    # padded to the captured batch with dummy rows. Its cells carry
    # `rank_rows`, the live rows of each rank's segment. Mistral Large 4 at dep8
    # (attention data parallel, experts over 8 ranks on two nodes) gathers
    # 8 x T_max rows, up to 262144, and calls it once per chunk of at most
    # 32768 of them (a contiguous row slice starting at a multiple of 32768),
    # so its eager calls are 8..32768 rows, which the 32768-row cell's sweep
    # drives; its decode graphs quantize 8 x 1/2/4/8/16 = 8/16/32/64/128 rows,
    # every 8 x 1..16 under the default capture list. Its graph cell carries
    # `ranks=8`.
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
        Cell(
            why="Mistral Large 4 decode, CUDA graphs at 1..16 rows (the gates' padded "
            "1/2/4/8/16 and the default capture's 1..16), layer 3's global scale "
            "9039.45: each captured, replayed on new rows bitwise equal to eager; "
            "padded replays (live rows r < b, a garbage tail) leave the live rows "
            "bitwise equal to the op on them alone",
            spec=dict(
                shape=(16, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                graph_rows=tuple(range(1, 17)),
                input_scale=_MISTRAL_LARGE4_LAYER3,
                input_amax_over_calibrated=1.0,
            ),
        ),
        Cell(
            why="Mistral Large 4 max_num_tokens 32768 (one unchunked context step), "
            "layer 65's global scale 23.0112: gated against reference, and every "
            "row count 1..32768 bitwise the leading rows of the 32768-row call "
            "(~5 s, ~16 GiB)",
            spec=dict(
                shape=(32768, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                sweep_rows=(1, 32768),
                input_scale=_MISTRAL_LARGE4_LAYER65,
                input_amax_over_calibrated=1.0,
            ),
        ),
        Cell(
            why="Mistral Large 4 live activations past the calibrated amax, 32768 rows "
            "at layer 3's global scale 9039.45: input amax twice 448*6/gs, so a "
            "share of blocks saturates its e4m3 scale at 448 and its codes at +-6",
            spec=dict(
                shape=(32768, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                input_scale=_MISTRAL_LARGE4_LAYER3,
                input_amax_over_calibrated=2.0,
            ),
        ),
        Cell(
            why="Mistral Large 4's 71 static global scales x every positive finite "
            "bf16 block max (32639 blocks and one zero block, 85 rows): the scale "
            "byte across the whole e4m3 range at every layer, underflow to 0 and "
            "saturation at 448 included",
            spec=dict(
                shape=(85, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                block_max="every positive finite bf16",
                input_scale=_MISTRAL_LARGE4_INPUT_SCALES,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep4 gathered maximum, 4 ranks x 32768 = 131072 rows, "
            "ranks zero-padded to T_max (live 32768/1/20000/32768), layer 65's "
            "global scale 23.0112: every row gated against reference; the row "
            "counts either side of each grid wrap and rank boundary past 32768, "
            "and all of 130465..131072, bitwise the leading rows (~1.5 s, ~18 GiB)",
            spec=dict(
                shape=(_DEP4_GATHERED_MAX, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                rank_rows=(32768, 1, 20000, 32768),
                sweep_counts=_DEP4_SWEEP_EDGES,
                input_scale=_MISTRAL_LARGE4_LAYER65,
                input_amax_over_calibrated=1.0,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep4 envelope step past the calibrated amax: four "
            "32K prefills gathered, 131072 live rows at layer 3's global scale "
            "9039.45, input amax twice 448*6/gs, so a share of blocks saturates "
            "its e4m3 scale at 448 and its codes at +-6",
            spec=dict(
                shape=(_DEP4_GATHERED_MAX, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                rank_rows=(32768,) * _DEP4_RANKS,
                input_scale=_MISTRAL_LARGE4_LAYER3,
                input_amax_over_calibrated=2.0,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep4 decode graphs, 4 ranks x 1..16 = 4..64 gathered "
            "rows (the gates' padded 1/2/4/8/16 per rank, 4/8/16/32/64 gathered, "
            "and the default capture's 1..16), layer 3's global scale 9039.45: "
            "each captured, replayed on new rows bitwise equal to eager; padded "
            "replays (each rank's segment some live rows, then dummy or zero "
            "rows) leave the live rows bitwise equal to the op on them alone",
            spec=dict(
                shape=(_DEP4_RANKS * 16, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                graph_rows=tuple(_DEP4_RANKS * b for b in range(1, 17)),
                ranks=_DEP4_RANKS,
                input_scale=_MISTRAL_LARGE4_LAYER3,
                input_amax_over_calibrated=1.0,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep8 decode graphs, 8 ranks x 1..16 = 8..128 gathered "
            "rows (the gates' padded 1/2/4/8/16 per rank, 8/16/32/64/128 gathered, "
            "and the default capture's 1..16), layer 3's global scale 9039.45: "
            "each captured, replayed on new rows bitwise equal to eager; padded "
            "replays (each rank's segment some live rows, then dummy or zero "
            "rows, and one live row with seven ranks idle) leave the live rows "
            "bitwise equal to the op on them alone",
            spec=dict(
                shape=(_DEP8_RANKS * 16, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                graph_rows=tuple(_DEP8_RANKS * b for b in range(1, 17)),
                ranks=_DEP8_RANKS,
                input_scale=_MISTRAL_LARGE4_LAYER3,
                input_amax_over_calibrated=1.0,
            ),
        ),
        Cell(
            why="Mistral Large 4 dep4 every gathered row count 4 x T_max, T_max "
            "8193..32768 (32772..131072 step 4), each bitwise the leading rows "
            "of the 131072-row call at layer 65's global scale, every row of "
            "which is gated against reference; large: 24576 calls, ~13 s",
            spec=dict(
                shape=(_DEP4_GATHERED_MAX, 6144),
                dtype=torch.bfloat16,
                vec=16,
                swizzled=False,
                rank_rows=(32768,) * _DEP4_RANKS,
                sweep_rows=(32772, _DEP4_GATHERED_MAX, _DEP4_RANKS),
                input_scale=_MISTRAL_LARGE4_LAYER65,
                input_amax_over_calibrated=1.0,
                large=True,
            ),
        ),
    )

    note = """
    Metadata consumed: none. Preconditions: `input` contiguous, its last dim a
    whole number of `sf_vec_size` blocks; with `sf_use_ue8m0=False`,
    `global_scale` a float32 device tensor holding one element (extra elements
    are what `is_valid` refuses).

    The path the cells drive is the bf16 one of `quantize_with_block_size`
    (`cpp/tensorrt_llm/kernels/quantization.cuh`, per block
    `cvt_warp_fp16_to_fp4`): scale byte = e4m3 (round to nearest, saturating)
    of `global_scale * max|x| / 6`; codes = e2m1 (round to nearest, saturating)
    of `x * global_scale / scale`. At hidden 6144 the launcher
    (`invokeFP4Quantization`) runs min(rows, 4 x SMs) CTAs of 512 threads, 608
    on this GB300, each walking rows at a stride of the grid, so past
    608 rows the launch is the same at every row count. Every row's output
    depends on that row and `global_scale` alone: at every row count 1..32768;
    at every dep4 gathered count 4 x T_max past it, up to 131072 (the `large`
    cell; pre-merge drives either side of every grid wrap and rank boundary,
    and all of the last grid period, 130465..131072); and in padded CUDA-graph
    replays, dep4's (4 x 1..16 rows) and dep8's (8 x 1..16 rows, one live row
    with seven ranks idle among them) with each rank's segment padded on its
    own. A probe, not a
    cell, drove every row count 32769..131072 (98304 calls, 51 s) against the
    131072-row call: each was bitwise its leading rows. A zero row, dep4's
    padding of a rank up to T_max, comes out as zero code bytes under scale
    byte 0.

    At dep8 each eager call is one chunk `o_all[lo:hi]` of the 8 x T_max
    gathered rows, `lo` a multiple of 32768: a contiguous slice at a nonzero
    storage offset. A probe, not a cell, quantized every chunk of a
    262144-row gathered set (byte offsets up to 2818572288) and the 8200-row
    last chunk of a 73736-row one: each was bitwise the op on a contiguous
    copy of the same rows, its scale bytes equal to `reference`. It also ran
    dep8's padding skew (rank 0 at 32768 rows, ranks 1..7 at one, T_max
    32768), whose chunks 1..7 each hold one live row and 32767 zero rows: zero
    codes under scale byte 0 there, the live row bitwise the op on it alone.

    `global_scale` states the range of the input. The canonical 448*6/amax of
    the input itself puts its largest block at scale 448. A static, calibrated
    scale (Mistral Large 4: 1 / input_scale(w1), one per MoE layer) states a range
    a live input can leave, in both directions, measured at every one of the
    checkpoint's 71 scales:
    - above it, a block whose `global_scale * max|x| / 6` reaches 448 gets
      scale byte 448 (0x7E) and its codes saturate at +-6; no scale byte is
      e4m3 NaN, even at bf16's largest finite block max;
    - below it, a block whose product rounds under the smallest e4m3 subnormal
      gets scale byte 0, so it dequantizes to 0 whatever its codes: at
      Mistral Large 4's scales every block whose max|x| is at most the bf16 value
      2.537e-4 (layer 65, 23.0112) down to 6.482e-7 (layer 3, 9039.45), the
      largest such block maxima found. Its nonzero inputs come out +-6,
      as in `reference`; an exact zero there makes a NaN product (0 x inf),
      which the kernel writes as code 7 (+6) and `reference` as 0. The test
      does not compare those codes: the block dequantizes to 0 either way.

    Cost on GB300 at hidden 6144: at 1..16 rows, and at dep8's gathered 8..128
    rows, a replay of a one-call CUDA graph is cheaper than an eager call.

    `sf_use_ue8m0=True` selects power-of-two scales instead of e4m3. No shipped
    target uses it and this entry does not certify it.

    Hard limits (sm_103)
      rows        the leading dims' product, passed to the launcher
                  (`invokeFP4Quantization`) as `int m`    boundary 2^31 rows
                  not probed: at least 2^35 elements, far past any envelope
                  (Mistral Large 4's largest call is dep4's 131072 gathered rows;
                  dep8's is one 32768-row chunk of its up to 262144)
      data        per 8-element vector, (row) * (K / 8) + col formed in int64
                  no int32 boundary below 2^31 rows    probed at 349526 x 6144 =
                  2147487744 elements (past 2^31), linear: passed -- the rows
                  straddling element 2^31, the middle rows and the last rows
                  bitwise equal to the op on those rows alone, their scale
                  bytes equal to `reference`
      scales      linear row * (K / 16) + col in int64; swizzled 128x4 tile
                  offsets in int64    no int32 boundary below 2^31 rows
                  linear: same probe, passed; swizzled: not probed
      the envelope's largest call, dep4's gathered 131072 x 6144 = 805306368
      elements (1610612736 input bytes, 402653184 code bytes, 50331648 scale
      bytes), is 37.5% of 2^31 elements; its input bytes would reach 2^31 at
      174763 rows, which the data row's int64 vector offsets never form. The
      probe above is 2.67x past it in rows, so it was not probed again.

    Not certified: `sf_use_ue8m0=True`; fp16 or e4m3 input; an input holding
    inf or NaN; a `global_scale` that is not finite and positive; Mistral Large 4's
    linear 6144 cells above 131072 rows (save the probe above), or with the
    swizzled layout; row counts 32769..130464 that are not multiples of 4 (no
    dep4 gather passes one; only the probe above drove them); CUDA-graph
    capture of any other cell; `torch.compile`.
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
        defect. Outside it, a difference is. An infinite product is no tie: it
        saturates at +-6 on both sides.
        """
        mid = torch.tensor(_E2M1_MIDPOINTS, device=scaled.device, dtype=torch.float32)
        a = scaled.abs().unsqueeze(-1)
        return ((a - mid).abs() <= TIE_WINDOW * a).any(-1) & torch.isfinite(scaled)

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
