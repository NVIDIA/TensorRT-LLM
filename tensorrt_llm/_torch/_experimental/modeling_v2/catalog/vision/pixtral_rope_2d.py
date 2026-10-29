# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower's 2-D rotary embedding: q and k of every patch rotated by
angles taken from the patch's (row, column) in its image.

Tier 4, composed by this run from native torch (frequency vector, angle
table, cos/sin rotor, one complex multiply per tensor); no engine kernel
computes it. Rejected at the cheaper rungs: every registered rope op that
takes positions (`mla_rope_inplace`,
`flashinfer_apply_rope_with_cos_sin_cache_inplace`, `fused_qk_norm_rope`,
`fused_dit_qk_norm_rope`) takes 1-D positions into one cos/sin table, so a
per-patch 2-D table would still have to be composed first; the one that takes
per-token cos/sin, `fused_dit_split_norm_rope`, fuses a full-width RMSNorm the
tower's q/k do not have. The in-tree Pixtral tower
(`_torch/models/modeling_pixtral.py`) calls transformers' Pixtral rotary
embedding, out of tree and in the HF half-split pairing over permuted
weights. FlashInfer's `rope` module has no 2-D variant. A single-kernel form
(for instance a per-patch table fed to one of the table-indexed ops above in
its non-NeoX mode) is the tuning phase's to supply.
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: params.json `vision_encoder`: hidden 2048 over 16 heads, rope_theta 1e4.
HEADS = 16
HEAD_DIM = 128
THETA = 10000.0


class PixtralRope2d(OpWrapper):
    """Rotate `q` [N, Hq, D] and `k` [N, Hk, D] of N patches by their 2-D
    angles; return fresh `(q_out, k_out)` in the inputs' shapes and dtypes.

        f_i         = theta ** (-2 i / D)                 i = 0 .. D/2 - 1
        angle[n, j] = row_n * f_{2j}                      j = 0 .. D/4 - 1
        angle[n, j] = col_n * f_{2(j - D/4) + 1}          j = D/4 .. D/2 - 1
        (x[2j], x[2j+1]) -> (x[2j] cos a_j - x[2j+1] sin a_j,
                             x[2j] sin a_j + x[2j+1] cos a_j)

    GPT-J interleaved pairs `(2j, 2j+1)` over the checkpoint's own (mistral
    native, unpermuted) wq/wk rows: row angles on the first D/4 pairs with the
    even frequencies, column angles on the last D/4 with the odd ones. fp32
    math throughout: the frequency vector `1 / theta ** (2i / D)` is computed
    in fp32 on the device, each angle is `fp32(position) * f` rounded once,
    the rotor is `torch.polar(1, angle)` (fp32 cos/sin), and each head is
    rotated as one complex64 multiply of `x.float()` viewed as D/2 complex
    pairs, rounded once to the input dtype on the way out.

    `positions` is [N, 2] integer (row, col) of each patch within its own
    image; patches of several images are concatenated along N, each image's
    positions starting at (0, 0). Order along N is the caller's (the
    reference walks each image row-major); the op is per patch.

    Fusion boundary: the angle table and the rotation. The qkv projection, the
    position grid of each image, the unrotated `v` and the attention are the
    caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `grid` one image's (rows, cols) or a tuple of them (several images
    # in one call); `heads` / `head_dim` for q and k; `theta`; `dtype` of q/k;
    # `layout` "packed_qkv" (q and k are head-reshaped column chunks of one
    # [N, 3 * heads * head_dim] projection, the tower's own layout) or
    # "separate" (contiguous q and k). Positions are int64 (row, col), each
    # image row-major from (0, 0).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one-patch image 1x1: angle 0, the rotor is exactly 1",
            spec=dict(
                grid=(1, 1),
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.bfloat16,
                layout="packed_qkv",
            ),
        ),
        Cell(
            why="32x32 image: 1024 patches",
            spec=dict(
                grid=(32, 32),
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.bfloat16,
                layout="packed_qkv",
            ),
        ),
        Cell(
            why="110x110 image (1540 px, the largest): 12100 patches, positions 0..109",
            spec=dict(
                grid=(110, 110),
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.bfloat16,
                layout="packed_qkv",
            ),
        ),
        Cell(
            why="two images in one call, 110x110 + 40x60 (non-square): 14500 patches",
            spec=dict(
                grid=((110, 110), (40, 60)),
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.bfloat16,
                layout="packed_qkv",
            ),
        ),
        Cell(
            why="envelope: two max-size images in one call, 24200 patches",
            spec=dict(
                grid=((110, 110), (110, 110)),
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.bfloat16,
                layout="packed_qkv",
            ),
        ),
        Cell(
            why="32K image envelope: ten maximum-size images, 121000 patches",
            spec=dict(
                grid=((110, 110),) * 10,
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.bfloat16,
                layout="packed_qkv",
            ),
        ),
        Cell(
            why="110x110 with q and k as separate contiguous tensors",
            spec=dict(
                grid=(110, 110),
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.bfloat16,
                layout="separate",
            ),
        ),
        Cell(
            why="110x110 fp32 q/k: the fp32 math's own error, not hidden by bf16",
            spec=dict(
                grid=(110, 110),
                heads=HEADS,
                head_dim=HEAD_DIM,
                theta=THETA,
                dtype=torch.float32,
                layout="packed_qkv",
            ),
        ),
    )

    note = """
    Metadata consumed: none. Out-of-place; `q`, `k` and `positions` are not
    mutated, and `q_out` / `k_out` are fresh contiguous tensors. `q` and `k`
    need last-dim stride 1 only (head-reshaped chunks of one qkv projection are
    taken as they are; `.float()` makes the fp32 copy the rotation runs on).
    `positions` sits on q's device. No host synchronization; capture under a
    CUDA graph was not measured (the call site runs eagerly).

    Weight layout: the pairing is GPT-J interleaved over the checkpoint's
    unpermuted wq/wk. The engine's mistral loader has `permute_qk`
    (`_torch/models/checkpoints/mistral/weight_mapper.py`), which reorders a
    vision tower's wq/wk into the HF half-split layout; a target that applies
    it to these weights needs the half-split pairing instead, and this entry
    is then silently wrong.

    Against the reference implementation: its mistral-format vision encoder
    computes the same arithmetic (fp32 frequencies, fp32 angle = position
    times frequency, complex64 rotor and multiply) but builds the frequency
    vector and the cos/sin table on the host and moves them to the device.
    Run in its own environment on the same bf16 inputs (not in CI), the
    entry's outputs fell inside this test's band at every cell grid, with 749
    of 49561600 elements per tensor differing at two 110x110 images: the
    device's fp32 pow, cos and sin sit a few ulp from the host's (4 of the 64
    frequencies differ by one ulp). The test's host-built-table alternative
    reproduces that recipe and lands inside the band.

    Accuracy (test-derived band, per element): within half a unit of the
    output dtype at both the result and the fp64 closed form, plus the fp32
    math's own budget times the element's pair magnitude; the cells spend at
    most 0.107 of that budget.

    Cost (eager, not gated): a call is a chain of small torch kernels; the
    tower calls it once per block.

    Guarded by `is_valid` (each accepted without an error and answered wrongly
    when unguarded): `positions` not [N, 2] with N the token count of q (a
    one-row `positions` broadcasts the first patch's angles over every
    patch); `k` with a different token count from `q` (a one-token `k`
    broadcasts to N tokens); `q` or `k` not 3-D (a 2-D [N, D] broadcasts to
    [N, N, D]).

    Certified at heads 16, head_dim 128, theta 1e4, int64 positions with rows
    and columns 0..109, bf16 q/k (fp32 at 110x110), up to 24200 patches per
    call. Uncertified: other head counts or head dims (head_dim must be a
    multiple of 4: the angle table's width is head_dim/2 only then), other
    theta, fp16 q/k, int32 positions, positions above 109, more than 24200
    patches per call, a q/k dtype mismatch.

    Hard limits (sm_103)
      positions  fp32(position), torch elementwise  boundary 2^24 + 1  not
                 reached: positions stop at 109 (a 1540-px side over 14-px
                 patches) -- unprobed
      numel      N * heads * head_dim, torch elementwise  boundary 2^31
                 not reached: the envelope's largest call is 24200 * 16 * 128
                 = 49561600 elements -- unprobed
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        head_dim = q.shape[-1]
        freqs = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
        )
        pos = positions.to(torch.float32)
        angles = torch.cat((pos[:, 0:1] * freqs[0::2], pos[:, 1:2] * freqs[1::2]), dim=-1)
        rotor = torch.polar(torch.ones_like(angles), angles).unsqueeze(-2)
        q_pairs = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_pairs = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_pairs * rotor).flatten(-2).to(q.dtype)
        k_out = torch.view_as_real(k_pairs * rotor).flatten(-2).to(k.dtype)
        return q_out, k_out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The closed form in fp64 from the pair index, with explicit real
        arithmetic: frequency, axis and angle per pair, then the 2x2 rotation,
        rounded once to the input dtype."""
        head_dim = q.shape[-1]
        pairs = head_dim // 2
        quarter = pairs // 2
        j = torch.arange(pairs, dtype=torch.float64, device=q.device)
        on_col = j >= quarter
        freq_index = torch.where(on_col, 2.0 * (j - quarter) + 1.0, 2.0 * j)
        freq = torch.pow(torch.tensor(theta, dtype=torch.float64), -2.0 * freq_index / head_dim)
        pos = positions.to(torch.float64)
        axis_pos = torch.where(on_col, pos[:, 1:2], pos[:, 0:1])
        angle = (axis_pos * freq.to(q.device)).unsqueeze(-2)
        cos, sin = torch.cos(angle), torch.sin(angle)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            x64 = x.double()
            even, odd = x64[..., 0::2], x64[..., 1::2]
            out = torch.empty_like(x64)
            out[..., 0::2] = even * cos - odd * sin
            out[..., 1::2] = even * sin + odd * cos
            return out.to(x.dtype)

        return rotate(q), rotate(k)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        theta: float,
    ) -> None:
        assert q.dim() == 3 and k.dim() == 3, (
            f"q and k must be [tokens, heads, head_dim]; got q {tuple(q.shape)}, k {tuple(k.shape)}"
        )
        assert k.shape[0] == q.shape[0], (
            f"q and k must carry the same tokens; got q {tuple(q.shape)}, k {tuple(k.shape)}"
        )
        assert positions.dim() == 2 and tuple(positions.shape) == (q.shape[0], 2), (
            f"positions must be [tokens, 2] (row, col) for the {q.shape[0]} tokens of q; "
            f"got {tuple(positions.shape)}"
        )


pixtral_rope_2d = PixtralRope2d()
