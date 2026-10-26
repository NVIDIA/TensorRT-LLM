# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perception-encoder 2-D rotary embedding with adjacent-pair (interleaved)
rotation: the first half of every head turns by the patch's column, the second
half by its row.

Tier 4, composed from torch ops: no engine kernel computes this rotation.
Rejected at the cheaper rungs:

- `torch.ops.trtllm.flashinfer_apply_rope_with_cos_sin_cache_inplace` takes one
  position per token and views the query as `[tokens, -1, head_size]`, so the
  two halves of a head, which turn by different positions, cannot both be
  addressed in one call; and with `is_neox=False` it pairs `(2j, 2j + 1)` across
  the whole head against one angle table, not two per-axis tables.
- The qwen4_exp `vision_rope_2d` entry pairs NeoX-style `(i, i + D/2)`, puts rows
  before columns, computes its own positions in spatial-merge order from
  `grid_thw`, and is certified at 16 x 72 heads only.
- No trtllm op and no FlashInfer callable takes a per-token 2-D position.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: `vision_config`: width 1536 over 16 heads is a 96-wide head; the reference
#: encoder builds its rope with `dim=head_dim`, `theta` 1e4 and a rescale
#: factor of 1.0 (so `theta` is unchanged).
NUM_HEADS = 16
HEAD_DIM = 96
THETA = 10000.0

#: Tiles in patch units: the 728-px overview is 52 x 52 patches of 14, a
#: 504-px crop 36 x 36. A crop's patches index the reference's 52-wide table at
#: `row * 52 + col`, which holds the angles of `(row, col)`: the crop's own
#: coordinates, not offsets into the overview.
OVERVIEW = (52, 52)
CROP = (36, 36)


def _cell(why: str, tiles: tuple, layout: str = "packed_qkv") -> Cell:
    return Cell(why=why, spec=dict(tiles=tiles, layout=layout, dtype=torch.bfloat16))


class VisionRope2dInterleaved(OpWrapper):
    """Rotate every head of `q` and `k` by its patch's 2-D angles; return fresh
    `(q_out, k_out)`, contiguous, in the inputs' shapes and dtypes.

        D = head_dim;  half = D / 2;  n = D / 4 frequencies per axis
        inv_freq[i] = theta ** (-2i / half)                      i < n
        angle[t]    = [col(t) * inv_freq, row(t) * inv_freq]     length half
        out[.., 2j]     = x[2j] cos a_j - x[2j + 1] sin a_j      j < half
        out[.., 2j + 1] = x[2j + 1] cos a_j + x[2j] sin a_j

    At D = 96: dims 0..47 turn by the column, 48..95 by the row, 24 frequencies
    `1e4 ** (-2i / 48)` per axis. Pairs are adjacent `(2j, 2j + 1)` (GPT-J
    style), not NeoX `(i, i + half)`.

    `positions` is `[tokens, 2]` integer on `q`'s device, `(row, col)` of each
    token's patch, so tiles of different grids can share one call. The angles
    and their cosines and sines are computed in fp64 and rounded once to fp32;
    the rotation runs in fp32 and rounds once on the store.

    Fusion boundary: the angles and the rotation. The qkv projection, the
    positions, the unrotated `v`, the head padding and the attention are the
    caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: tiles -- the (rows, cols) grids packed back to back into one call,
    # in order; layout -- "packed_qkv" (q and k are [T, 16, 96] views of one
    # [T, 3 * 1536] projection, as the tower holds them) or "separate"
    # (contiguous q and k); dtype.
    CELLS: tuple[Cell, ...] = (
        _cell("call site: one overview tile, 52x52 = 2704 rows", (OVERVIEW,)),
        _cell("call site: one crop tile, 36x36 = 1296 rows", (CROP,)),
        _cell(
            "call site: merlion, 6 crops then the overview, 10480 rows",
            (CROP,) * 6 + (OVERVIEW,),
        ),
        _cell(
            "envelope: overview + 8 crops, 13072 rows, the largest vision call",
            (OVERVIEW,) + (CROP,) * 8,
        ),
        _cell("contiguous q and k: one overview tile, 2704 rows", (OVERVIEW,), "separate"),
    )

    note = """
    Metadata consumed: none. `positions` is read on the device (no host sync);
    `q` and `k` need last-dim stride 1 only and are not mutated. Out-of-place:
    two fresh contiguous `[T, H, 96]` tensors.

    Caller preconditions: `positions[t] = (row, col)` of token t's patch inside
    its own tile, row-major over the tile, as the patch embedding flattens it.
    A crop tile's coordinates run 0..35: the reference indexes its 52-wide
    table at `row * 52 + col`, which is the angle of `(row, col)` itself.

    Launches: 29 CUDA kernels per call (angle table, fp64 cos/sin, and per
    tensor a widen, four products, two sums, an interleave and a narrow).
    Measured on sm_103 with q/k as packed-qkv views: 194 us per call and
    57.6 MiB peak above the inputs at 2704 rows; 646 us and 277.9 MiB at
    13072 rows. The tower calls it once per block (47 per vision call). The
    tuning phase replaces it with one fused kernel.

    Numerics: the output is the fp64 rotation rounded once, except where the
    fp32 evaluation lands within its own error of a bf16 rounding midpoint
    (those elements are one bf16 ulp off) or where a pair cancels to far
    below its head's scale. Measured over the cells: 1.5e-5 to 1.8e-5 of
    elements differ from the rounded reference; the largest deviation is
    1.56e-2 (one ulp at magnitude 2..4). The test gates each element at one
    bf16 ulp plus 2^-20 of its head's largest magnitude and the off fraction
    at 1e-3 (derivation in the test). fp32 operands (measured, no cell):
    4.8e-7 from fp64. Wrong operators in fp32 land 8.3 to 10.3 away: NeoX
    pairing, rows before columns, `inv_freq` over the head (2i / 96), the
    column angle on both halves.

    StepFun's inference code (`EncoderRope2D.forward`) casts its angle table
    to the activation dtype before taking cos/sin, so in bf16 it turns by
    bf16-rounded angles (up to 0.125 rad off at angle 51) in bf16 arithmetic.
    This entry does not reproduce that rounding. Measured on unit-normal
    bf16 q at 52x52: that path lands 0.39 (max) from the fp64 rotation,
    relative RMS 9.5e-3, 37% of elements off; the band refuses it. A
    comparison against that implementation's bf16 tower output carries this
    difference in every block.

    Guards (`is_valid`): `k` or `positions` with a token count other than
    `q`'s (a one-row `k` or `positions` broadcasts to a plausible `[T, H, D]`
    with no error).

    Loud, not guarded: a head_dim that is 2 mod 4 (the angle table is one
    pair short and the broadcast raises); `positions` on the host (device
    mismatch raises).

    Hard limits
      positions are converted to fp64, exact far past any grid side.
      q/k numel: largest certified 13072 x 16 x 96 = 2.0e7, far below 2^31
      (2^31 / 1536 = 1398101 rows per call).

    Certified: bf16 q/k `[T, 16, 96]`, as split views of a packed
    `[T, 3 * 1536]` projection or contiguous, theta 1e4, positions int64 on
    the device, tiles 52x52 and 36x36 packed up to 13072 rows.

    Not certified: fp16/fp32 operands; head geometries other than 16 x 96;
    theta other than 1e4; grid sides past 52; int32 positions (measured equal
    to int64 in the test, no cell); CUDA-graph capture (the tower is eager;
    the call has no host sync); torch.compile.
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        half = q.shape[-1] // 2
        exponent = torch.arange(half // 2, dtype=torch.float64, device=q.device) * (-2.0 / half)
        inv_freq = torch.pow(theta, exponent)
        # [T, 2] (row, col) -> [T, half]: column angles first, then row angles.
        angles = (positions.flip(-1).to(torch.float64)[:, :, None] * inv_freq).flatten(1)
        cos = angles.cos().float()[:, None, :]
        sin = angles.sin().float()[:, None, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            pairs = x.float().unflatten(-1, (half, 2))
            even, odd = pairs[..., 0], pairs[..., 1]
            turned = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
            return turned.flatten(-2).to(x.dtype)

        return rotate(q), rotate(k)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-axis angles from Python-float `inv_freq` in fp64, the rotation as
        a complex128 multiply of each adjacent pair, rounded once to the input
        dtype."""
        half = q.shape[-1] // 2
        inverse = torch.tensor(
            [theta ** (-2.0 * i / half) for i in range(half // 2)],
            dtype=torch.float64,
            device=q.device,
        )
        coords = positions.to(torch.float64)
        row, col = coords[:, 0:1], coords[:, 1:2]
        angles = torch.cat((col * inverse, row * inverse), dim=-1)
        rotor = torch.polar(torch.ones_like(angles), angles)[:, None, :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            z = torch.view_as_complex(x.double().reshape(*x.shape[:-1], half, 2).contiguous())
            return torch.view_as_real(z * rotor).reshape(x.shape).to(x.dtype)

        return rotate(q), rotate(k)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
    ) -> None:
        # A one-row k or positions broadcasts against q's tokens and returns a
        # full-size result without an error.
        assert k.shape[0] == q.shape[0], (
            f"vision_rope_2d_interleaved: k carries {k.shape[0]} tokens, q {q.shape[0]}"
        )
        assert positions.shape == (q.shape[0], 2), (
            f"vision_rope_2d_interleaved: positions must be [{q.shape[0]}, 2] (row, col); "
            f"got {tuple(positions.shape)}"
        )


vision_rope_2d_interleaved = VisionRope2dInterleaved()
