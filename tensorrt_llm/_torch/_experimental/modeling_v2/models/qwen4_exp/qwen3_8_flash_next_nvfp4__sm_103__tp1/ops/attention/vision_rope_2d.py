# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vision-tower 2-D axial rotary embedding (patch row / patch column), no KV
cache.

Not the text tower's mRoPE: that rotates 64 of a 256-wide head against three
interleaved position rows (`mrope_section`, `rope_theta` 1e7); this rotates the
whole 72-wide vision head against two spatial rows from the image grid, 18 row
angles then 18 column angles, `theta` 1e4. Neither reads the other's config.

Tier-3 composed from native torch (angle table, index gather, cos/sin, two
multiply-adds). Every registered rope op takes 1-D positions;
`flashinfer_apply_rope_with_cos_sin_cache_inplace` with an explicit per-token
table would sidestep that, but at `head_size=72` it raises "misaligned
address" and poisons the CUDA context.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint: 1152 / 16 heads = a 72-wide head, spatial_merge_size 2,
#: theta 1e4 (`model.visual.blocks.N.attn.qkv.weight` is [3456, 1152]).
NUM_HEADS = 16
HEAD_DIM = 72
MERGE = 2
THETA = 10000.0

#: (frames, height, width) in patch units, from the runtime-value dump of the
#: mmmu gate (58.33): 170 segment sizes from 256 to 12544 per frame, calls of up
#: to 22144 rows. 62x90 is a grid this processor emits.
SERVING_FRAME = ((1, 62, 90),)


def _rotate_half(x: torch.Tensor, half: int) -> torch.Tensor:
    """NeoX-style half rotation: `[x0, x1] -> [-x1, x0]` over the last dim."""
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _spatial_merge_positions(grid: Sequence[Sequence[int]], merge: int) -> List[Tuple[int, int]]:
    """Per-token `(row, col)` by four nested loops, in spatial-merge order."""
    positions: List[Tuple[int, int]] = []
    for frames, height, width in grid:
        frame = [
            (block_row * merge + r, block_col * merge + c)
            for block_row in range(height // merge)
            for block_col in range(width // merge)
            for r in range(merge)
            for c in range(merge)
        ]
        positions.extend(frame * frames)
    return positions


class VisionRope2d(OpWrapper):
    """Rotate every head of `q` and `k` by its patch's 2-D axial angles; return
    fresh `(q_out, k_out)` in the inputs' shapes and dtypes.

        rope_dim = head_dim // 2;  axis_dim = rope_dim // 2
        inv_freq[j] = theta ** (-2j / rope_dim)               j < axis_dim
        angle[t] = [row(t) * inv_freq, col(t) * inv_freq]      height half first
        out[.., i]            = x[i] cos a_i - x[i + rope_dim] sin a_i
        out[.., i + rope_dim] = x[i + rope_dim] cos a_i + x[i] sin a_i

    NeoX half-split pairing `(i, i + rope_dim)`, not GPT-J `(2j, 2j+1)`. Tokens
    are in spatial-merge order (whole `merge x merge` blocks row-major,
    row-major inside a block) -- what the patch merger consumes; a raster walk
    is silently wrong. A video's frames repeat the spatial grid: no temporal
    axis. The angle table is built in fp64 and rounded once to fp32; the
    rotation runs in fp32 and rounds once on the store.

    Fusion boundary: the angle table and the rotation. The qkv projection, the
    unrotated `v`, the per-frame attention segmentation and the attention are
    the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: grid, dtype; `heads` / `heads_k` / `head_dim` (default 16 / heads /
    # 72); `theta` (default 1e4); `grid_dtype` / `grid_device` for `grid_thw`
    # (default int32 on the host); `layout` "separate" (default) or
    # "packed_qkv" -- q and k as the `split` views of one [T, 3*heads, D]
    # projection, which is how the tower calls it.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="call site: q/k split views of packed qkv, 62x90",
            spec=dict(grid=SERVING_FRAME, dtype=torch.bfloat16, layout="packed_qkv"),
        ),
        Cell(
            why="call site: packed qkv views, video + image (two grid rows)",
            spec=dict(
                grid=((3, 8, 8), (1, 16, 24)),
                dtype=torch.bfloat16,
                layout="packed_qkv",
                grid_dtype=torch.int64,
            ),
        ),
        Cell(
            why="call site: packed qkv views, video at the 32768-row ceiling",
            spec=dict(
                grid=((64, 16, 32),),
                dtype=torch.bfloat16,
                layout="packed_qkv",
                grid_dtype=torch.int64,
            ),
        ),
    )

    note = """
    Metadata consumed: none. `grid_thw` is read with `.tolist()`: a device
    tensor forces a sync (int32/int64, host/device all bitwise equal). `q` and
    `k` need last-dim stride 1 only; they are not mutated.

    `head_dim` must be divisible by 4; otherwise the angle table's width
    mismatches the head and the broadcast raises (loud, so not guarded; not
    pinned by a test).

    Accuracy vs fp64 angles over the cells: bf16 <= 1.56e-2 (one ulp at
    unit-normal operands). Measured, not certified: fp32 <= 4.77e-7, flat in the
    grid side up to 6272 -- the fp64 table is why (an fp32 `inv_freq` times fp32
    position moved an fp32 output 1.475e-5 at a 64 side, outside the default
    gate); wrong operators at 1024 bf16 tokens land 8.5-9.5 away: GPT-J pairing,
    axes swapped, raster order, `inv_freq` over `head_dim`.

    Host-bound: ~353 us/call at 1024 and 4096 tokens (`.tolist()` + small
    launches), x27 blocks per image. The table depends only on the grid, and the
    result does not depend on the grid's placement (host or device) or integer
    dtype (int32 or int64).

    Not certified: float16 or float32; mixed q/k dtypes or head counts; head
    geometries other than 16 x 72; float8; `spatial_merge_size` != 2; `theta`
    other than 1e4; grid sides past 90 or calls past 32768 rows; CUDA-graph
    capture (it would bake in the grid).
    """

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        theta: float = 10000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens, _, head_dim = q.shape
        # Angles per token: half the head (the other half is each pair's
        # partner); each axis owns half of those, height first.
        rope_dim = head_dim // 2

        grid: List[List[int]] = grid_thw.tolist()

        device = q.device
        # fp64 then one rounding to fp32: the largest angle is (side - 1)
        # radians, and fp32 throughout rounds twice there.
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64, device=device) / rope_dim)
        )
        max_side = max(max(height, width) for _, height, width in grid)
        freq_table = torch.outer(
            torch.arange(max_side, dtype=torch.float64, device=device), inv_freq
        ).to(torch.float32)

        merge = spatial_merge_size
        intra = torch.arange(merge, device=device)
        pos_ids = torch.empty((num_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for frames, height, width in grid:
            blocks_h, blocks_w = height // merge, width // merge
            rows = (
                (
                    torch.arange(blocks_h, device=device)[:, None, None, None] * merge
                    + intra[None, None, :, None]
                )
                .expand(blocks_h, blocks_w, merge, merge)
                .reshape(-1)
            )
            cols = (
                (
                    torch.arange(blocks_w, device=device)[None, :, None, None] * merge
                    + intra[None, None, None, :]
                )
                .expand(blocks_h, blocks_w, merge, merge)
                .reshape(-1)
            )
            coords = torch.stack((rows, cols), dim=-1)
            if frames > 1:
                coords = coords.repeat(frames, 1)
            pos_ids[offset : offset + coords.shape[0]] = coords
            offset += coords.shape[0]

        # [tokens, 2, axis_dim] -> [tokens, rope_dim]: height angles, then width.
        angles = freq_table[pos_ids].flatten(1)
        duplicated = torch.cat((angles, angles), dim=-1)
        cos = duplicated.cos().unsqueeze(-2)
        sin = duplicated.sin().unsqueeze(-2)

        q_fp32, k_fp32 = q.float(), k.float()
        q_out = (q_fp32 * cos + _rotate_half(q_fp32, rope_dim) * sin).to(q.dtype)
        k_out = (k_fp32 * cos + _rotate_half(k_fp32, rope_dim) * sin).to(k.dtype)
        return q_out, k_out

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        theta: float = 10000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Positions from nested loops, angles from Python-float `inv_freq` times
        integer positions in fp64, rotation as a complex multiply in fp32."""
        rope_dim = q.shape[-1] // 2
        axis_dim = rope_dim // 2
        positions = _spatial_merge_positions(grid_thw.tolist(), spatial_merge_size)
        inverse = torch.tensor(
            [theta ** (-(2.0 * j) / rope_dim) for j in range(axis_dim)], dtype=torch.float64
        )
        pos = torch.tensor(positions, dtype=torch.float64).reshape(-1, 2)
        angles = (pos[:, :, None] * inverse).flatten(1).to(device=q.device, dtype=torch.float32)
        rotor = torch.polar(torch.ones_like(angles), angles).unsqueeze(-2)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            fp32 = x.float()
            rotated = torch.complex(fp32[..., :rope_dim], fp32[..., rope_dim:]) * rotor
            return torch.cat((rotated.real, rotated.imag), dim=-1).to(x.dtype)

        return rotate(q), rotate(k)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        theta: float = 10000.0,
    ) -> None:
        # Each of these returns a full-size result without an error: a side
        # that does not divide fills only the whole blocks' rows of the
        # position table and leaves the rest uninitialized (four tokens of a
        # 4x5 grid read back as unrotated); a grid describing fewer tokens than
        # q carries leaves the tail uninitialized the same way; a one-token k
        # broadcasts against every token's angles.
        assert q.dim() == 3 and k.dim() == 3, (
            f"q and k must be [tokens, heads, head_dim]; got q {tuple(q.shape)}, k {tuple(k.shape)}"
        )
        assert q.shape[0] == k.shape[0] and q.shape[-1] == k.shape[-1], (
            f"q and k must agree on tokens and head_dim; got q {tuple(q.shape)}, k {tuple(k.shape)}"
        )
        total = 0
        for frames, height, width in grid_thw.tolist():
            assert height % spatial_merge_size == 0 and width % spatial_merge_size == 0, (
                f"grid sides must be divisible by spatial_merge_size={spatial_merge_size}; "
                f"got height={height}, width={width}"
            )
            total += frames * height * width
        assert total == q.shape[0], f"grid_thw describes {total} tokens but q/k carry {q.shape[0]}"


vision_rope_2d = VisionRope2d()
