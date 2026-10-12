# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place RoPE on the trailing rope_dim channels of every head, from a caller-built cos/sin table.

The op is `torch.ops.trtllm.mla_rope_inplace`, registered by `tensorrt_llm._torch.custom_ops`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Rows of the largest table the target builds: `max_seq_len + 64` at the envelope's largest
#: per-request length 1048576 with DSpark on (+14). Every position below it is certified.
_TABLE_ROWS = 1048654
_RATIO0, _COMPRESSED = "ratio0_theta1e4", "compressed_theta160k_yarn"
#: (name, H, head_dim, nope_dim, inverse) of every tensor the checkpoint rotates with a recipe
#: table: q, window kv, index q and index k forward, the attention output inversely.
_TABLE_FORMS = (
    ("q_tp4", 16, 512, 448, False),
    ("q_dep4", 64, 512, 448, False),
    ("window_kv", 1, 512, 448, False),
    ("index_q_tp4", 8, 128, 64, False),
    ("index_q_32", 32, 128, 64, False),
    ("index_k", 1, 128, 64, False),
    ("attn_out_tp4", 16, 512, 448, True),
    ("attn_out_dep4", 64, 512, 448, True),
)
#: Row counts and how the test places their positions on the table.
_TABLE_T = {
    1: "the table's last row",
    6: "a consecutive run, as in a DSpark verify step",
    8192: "random over the whole table, 0 and the last row included",
}
#: Patch counts of the DeepSeek-V4.1 vision tower's per-request call: the prototype's
#: whole-tower graph sizes, the MMMU maximum 17037, and the two smallest.
_VIT_P = (1, 2, 1517, 4680, 5325, 8772, 17037)


class MlaRopeInplace(OpWrapper):
    """Rotate `data[:, :, nope_dim:]` in place by the table rows at `position_ids`.

    With `c, s = cos_sin_cache[position_ids[t], 0], cos_sin_cache[position_ids[t], 1]`
    (`s` negated when `inverse`), every head's trailing `rope_dim` channels are rotated pair by
    pair: adjacent channels `(2i, 2i+1)` when `is_neox=False` (GPT-J), the two halves
    `(i, i + rope_dim/2)` when `is_neox=True`. Computed in fp32, rounded once to `data`'s
    dtype. The leading `nope_dim` channels are not touched.

    Fusion boundary: one rotation of one tensor. The projection that produced `data`, the
    table (frequencies, YaRN, any mscale), the positions and everything after are the
    caller's. Writes `data` in place and returns `data`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=(
                    f"{name} {table.split('_')[0]} T={t}: [T,{h},{d}] nope {nope}"
                    f"{', inverse' if inverse else ''}, {where}"
                ),
                spec=dict(
                    form="table",
                    table=table,
                    rows=_TABLE_ROWS,
                    T=t,
                    H=h,
                    D=d,
                    nope=nope,
                    inverse=inverse,
                    is_neox=False,
                ),
            )
            for name, h, d, nope, inverse in _TABLE_FORMS
            for table in (_RATIO0, _COMPRESSED)
            for t, where in _TABLE_T.items()
        )
        + tuple(
            Cell(
                why=(
                    f"group-first {name} r={ratio}: [{_TABLE_ROWS // ratio},1,{d}] at positions"
                    " j*r, every group whose first position is in the compressed table"
                ),
                spec=dict(
                    form="group_first",
                    table=_COMPRESSED,
                    rows=_TABLE_ROWS,
                    ratio=ratio,
                    H=1,
                    D=d,
                    nope=d - 64,
                    inverse=False,
                    is_neox=False,
                ),
            )
            for ratio in (1, 2)
            for name, d in (("compressed_kv", 512), ("index_k", 128))
        )
        + tuple(
            Cell(
                why=(
                    f"neox full head T={t} {'inverse' if inverse else 'forward'}: [T,16,64]"
                    " nope 0, half-split pairs, random angles per row"
                ),
                spec=dict(
                    form="random_angles",
                    T=t,
                    H=16,
                    D=64,
                    nope=0,
                    inverse=inverse,
                    is_neox=True,
                ),
            )
            for t in (1296, 4096)
            for inverse in (False, True)
        )
        + tuple(
            Cell(
                why=(
                    f"vit packed qkv P={p}: [3P,16,64] NeoX, own [P+1] table, positions"
                    " (t, t, P) so v reads the identity row and comes back unchanged"
                ),
                spec=dict(form="vit_packed", P=p, H=16, D=64, nope=0, inverse=False, is_neox=True),
            )
            for p in _VIT_P
        )
    )

    note = """
    Measured on sm_103 against an fp32 torch rotation: at least 0.99997 of
    elements bit-exact, every difference inside bf16's default band, largest
    3.1e-2 on unit-normal inputs (consistent with FMA contraction changing the
    final rounding). `inverse=True` undoes a forward rotation to within two
    bf16 roundings.

    The table is the caller's: RoPE base, YaRN interpolation and any mscale
    live in it; the op neither normalizes nor derives frequencies, and a table
    scaled by an mscale scales the rotated channels. The certified tables are
    DeepSeek-V4.1's two `[rows, 2, 32]` fp32 recipes, no mscale: ratio-0 (base
    10000, no YaRN; decoder layers 0-1 and the DSpark stages) and compressed
    (base 160000, YaRN factor 16, original 65536, beta_fast 32, beta_slow 1;
    layers 2-39). The target builds them once per engine with
    `max_seq_len + 64` rows; a shorter table is bitwise the first rows of a
    longer one. The compressed latent and index k of a ratio-r layer rotate at
    their group's first position `j * r`. The ViT forms pass their own table.

    Metadata consumed: none. `position_ids[t]` is only a row index into the
    table; the op reads no batch layout or sequence boundary.

    Preconditions: every position in `[0, rows)`. There is no bounds check, so
    an out-of-range position (a negative padding position too) is an
    out-of-bounds read; the row offset is 32-bit, exact below position 2^25.
    Alignment is not checked: `nope_dim` and the head width must be multiples
    of 8 elements and `data` must start 16-byte aligned (a contiguous row
    slice of a contiguous buffer does), otherwise the kernel faults with a
    sticky misaligned-address error that leaves the CUDA context unusable.
    The op raises RuntimeError, leaving `data` untouched, on: non-contiguous
    `data` (a `q[..., 448:]` tail view), non-int32 positions, data that is not
    bf16/fp16, a non-fp32 table, a table not shaped `[P, 2, rope_dim / 2]`,
    non-3D `data`, `data.size(1) != num_heads`, `data.size(2) != nope_dim +
    rope_dim`, and a `rope_dim` that is odd or not a multiple of 4. `T = 0` is
    accepted.

    In place: a caller that needs the pre-rope tensor keeps its own copy (the
    DeepSeek-V4.1 indexer reads the compressed latent before its rotation).
    For a q + kv pair at the same positions, one FlashInfer
    `apply_rope_with_cos_sin_cache_inplace` call measured 1.4-2.4 us faster
    than two calls of this op.

    Beyond the cells, the test certifies: every row count 1..32768 for the 21
    core form/table calls (q and the output at H 8/16/32/64, window kv, index
    q at 32 heads, index k, compressed latent), the boot-gate row counts, one
    `mlaRoPEInplaceKernel` launch per call at every probed T, CUDA-graph
    capture of all 21 calls at the 126 row counts `m * G` (m in {1, 3, 5, 6},
    G in 1..32, 40..128 step 8, 192, 256), every envelope engine's own table
    with positions at its far end, and the packed ViT form at every row count
    1..51111 and under graph capture at T = 3P for its graph sizes.

    Not certified: fp16 numerics (accepted, not compared); `rope_dim` other
    than 64; `nope_dim` other than 0, 64 or 448; heads other than 1, 8, 16, 32
    or 64; positions at or past 1048654; T above 32768 at H > 1, except the
    packed ViT form (to 51111, tables up to 17038 rows); half-split pairs with
    `nope_dim > 0`; graph capture of the core forms at a T outside the
    126-value union.
    """

    def raw_call(
        self,
        data: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        num_heads: int,
        nope_dim: int,
        rope_dim: int,
        inverse: bool,
        is_neox: bool,
    ) -> torch.Tensor:
        torch.ops.trtllm.mla_rope_inplace(
            data,
            position_ids,
            cos_sin_cache,
            num_heads,
            nope_dim,
            rope_dim,
            inverse,
            is_neox,
        )
        return data

    def reference(
        self,
        data: torch.Tensor,
        position_ids: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        num_heads: int,
        nope_dim: int,
        rope_dim: int,
        inverse: bool,
        is_neox: bool,
    ) -> torch.Tensor:
        """`data` with its rope channels rotated, in fp32, rounded once to `data`'s dtype.

        `data` is not mutated: the result is a new tensor. `num_heads` is ignored (it is
        `data.size(1)`).
        """
        y = data.float().clone()
        cos = cos_sin_cache[position_ids.long(), 0][:, None, :]
        sin = cos_sin_cache[position_ids.long(), 1][:, None, :]
        if inverse:
            sin = -sin
        tail = y[..., nope_dim : nope_dim + rope_dim]
        half = rope_dim // 2
        if is_neox:
            x1, x2 = tail[..., :half].clone(), tail[..., half:].clone()
            tail[..., :half] = x1 * cos - x2 * sin
            tail[..., half:] = x1 * sin + x2 * cos
        else:
            x1, x2 = tail[..., 0::2].clone(), tail[..., 1::2].clone()
            tail[..., 0::2] = x1 * cos - x2 * sin
            tail[..., 1::2] = x1 * sin + x2 * cos
        return y.to(data.dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """torch's default band, plus at least 0.999 of elements bit-exact.

        Kernel and reference both rotate in fp32 and round once; they differ only where FMA
        contraction flips the final rounding, measured on sm_103 at 0.99997 or more of the
        elements bit-exact in every probed shape. The fraction gate is on top of the band, not
        a loosening of it. The wrong-variant controls (pairing, rotated channels, YaRN, mscale,
        table, direction, position off by one) put 0.04-0.24 of the elements outside the band.
        """
        assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        torch.testing.assert_close(out, ref)
        bitexact = (out == ref).float().mean().item()
        assert bitexact >= 0.999, f"bit-exact fraction {bitexact}"


mla_rope_inplace = MlaRopeInplace()
