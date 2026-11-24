# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer Q / K quantization to MXFP4: concatenate two halves of each 128-wide row and quantize
it to FP4 E2M1 with one UE8M0 scale per 32 elements (CUDA).

This is the FP4 index-key format the family's FP4 entries share: the index-key cache, its scatter,
and the paged and dense FP4 MQA logits. A row is 64 bytes of E2M1 codes, two per byte with the even
element in the low nibble, plus one int32 that holds the four block exponents: block ``g`` (elements
``32 g`` to ``32 g + 31``) is byte ``g``, little-endian. :func:`mxfp4_values` decodes a row.

The compiled op is looked up on the first call, so the entry (its reference, its guard and the
decoder) imports on a host without the built extension.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Elements per row (the indexer head dim) and per UE8M0 scale block.
ROW = 128
BLOCK = 32
#: The E2M1 magnitudes of codes 0..7; bit 3 is the sign.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: The kernel's 1 / 6 (E2M1's largest magnitude), rounded to fp32.
_INV_E2M1_MAX = torch.tensor(1.0 / 6.0, dtype=torch.float32).item()
_MIN_AMAX = 1.0e-12
#: Upper edges of the code intervals: a magnitude above edge ``i`` (strictly) takes code ``i + 1``,
#: so every midpoint rounds to the smaller magnitude.
_EDGES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


@functools.cache
def _trtllm_ops():
    """``torch.ops.trtllm`` with the compiled ops registered, on the first call."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

    return torch.ops.trtllm


def mxfp4_values(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 ``[..., 128]``: the values of MXFP4 rows.

    ``packed``: ``[..., 64]`` codes, any 1-byte dtype; ``scales``: ``[...]`` or ``[..., 1]`` four
    UE8M0 exponents per row, any 4-byte dtype. Every value is an E2M1 magnitude times a power of two,
    exact in fp32 for the exponents a quantized bf16 row takes.
    """
    data = packed.contiguous().view(torch.uint8)
    lead = data.shape[:-1]
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(*lead, ROW).long()
    table = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=data.device)
    values = table[codes & 7] * (1.0 - 2.0 * ((codes >> 3) & 1).float())
    exponents = scales.contiguous().view(torch.uint8).reshape(*lead, ROW // BLOCK)
    scale = torch.exp2(exponents.float() - 127.0)
    return (values.view(*lead, ROW // BLOCK, BLOCK) * scale[..., None]).reshape(*lead, ROW)


def _spec(tokens, heads, **overrides) -> dict:
    """``heads``: 32 for the indexer Q (rows = tokens x 32), 0 for the index key (rows = tokens)."""
    spec = dict(tokens=tokens, heads=heads, pe_dim=64, data="randn")
    spec.update(overrides)
    return spec


class RubinIdxFp4Quantize(OpWrapper):
    """Row ``m`` of ``x = cat([pe, nope], -1)`` (bf16, 128 wide) is quantized in four blocks of 32.

    Each block ``g`` gets the scale ``s = 2 ** ceil(log2(max(amax_g, 1e-12) * fp32(1 / 6)))``
    (its UE8M0 exponent is the scale's fp32 exponent byte). Each element gets the E2M1 code of
    ``min(|x / s|, 6)``, rounding every midpoint toward zero, with the sign bit set only on a
    nonzero code.

    Returns ``(packed, scales)``, newly allocated:

    * ``packed``: int8 ``[M, 64]``, two codes per byte, even element in the low nibble;
    * ``scales``: int32 ``[M, 1]``, the four exponent bytes, block 0 lowest.

    ``M`` is the row count of ``pe``, with all leading dims flattened. ``pe`` and ``nope`` may be
    strided views (unit innermost stride, ``pe``'s width a multiple of 4).
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"indexer Q of {tokens} tokens (32 heads, halves of one row-major tensor)",
                spec=_spec(tokens, 32),
            )
            for tokens in (1, 8, 96, 300, 2048)
        ),
        *(
            Cell(
                why=f"index keys of {tokens} tokens (halves of the normalized wk output)",
                spec=_spec(tokens, 0),
            )
            for tokens in (1, 8, 96, 3000, 16384)
        ),
        Cell(
            why="all-zero blocks: the 1e-12 amax floor, codes 0 and no negative zero",
            spec=_spec(8, 32, data="zeros_block"),
        ),
        Cell(
            why="large and tiny magnitudes: block exponents far from 127",
            spec=_spec(64, 32, data="wide"),
        ),
        Cell(
            why="values on the code midpoints and the clamp: the rounding rule",
            spec=_spec(64, 32, data="midpoints"),
        ),
    )
    note = """
    The Rubin default target calls this twice per DSA layer per step for an FP4 index-key cache:
    the indexer Q ([T, 32, 128] from wq_b) and the index keys ([T, 128] from the key LayerNorm),
    each passed as its two 64-wide halves (the indexer has no rope). The scales are powers of two,
    so x / s is exact and the codes equal the reference bit for bit; the query scales stay with the
    queries (the FP4 MQA logits apply them in the MMA), so the logits weights carry no Q scale. One
    warp per row; deterministic. Captures in a CUDA graph (the outputs come from the graph pool).
    """

    def __call__(self, pe: torch.Tensor, nope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _trtllm_ops().fused_cat_fp4(pe, nope)

    def reference(self, pe: torch.Tensor, nope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 evaluation with the kernel's scale and rounding rules."""
        x = torch.cat([pe, nope], dim=-1).reshape(-1, ROW).float()
        rows = x.shape[0]
        blocks = x.view(rows, ROW // BLOCK, BLOCK)
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(_MIN_AMAX)
        bits = (amax * _INV_E2M1_MAX).view(torch.int32)
        exponent = bits & 0x7F800000
        exponent = torch.where((bits & 0x007FFFFF) != 0, exponent + 0x00800000, exponent)
        scale = exponent.view(torch.float32)
        scaled = blocks / scale
        magnitude = scaled.abs().clamp_max(6.0)
        code = torch.zeros_like(magnitude, dtype=torch.int32)
        for edge in _EDGES:
            code += (magnitude > edge).to(torch.int32)
        code = code | (((scaled < 0) & (code != 0)).to(torch.int32) << 3)
        code = code.view(rows, ROW).to(torch.uint8)
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).view(torch.int8)
        exponent_bytes = (exponent.view(rows, ROW // BLOCK) >> 23).to(torch.uint8)
        scales = exponent_bytes.contiguous().view(torch.int32).view(rows, 1)
        return packed, scales

    def is_valid(self, pe: torch.Tensor, nope: torch.Tensor) -> None:
        # The kernel walks the flattened rows with one row stride, stride(-2): leading dims that
        # do not collapse onto it (a slice of heads, a transposed view) are read at the wrong
        # addresses with no error.
        for name, t in (("pe", pe), ("nope", nope)):
            for d in range(t.dim() - 2):
                assert t.stride(d) == t.stride(d + 1) * t.shape[d + 1], (
                    f"{name}'s leading dims must collapse onto stride(-2); got shape "
                    f"{tuple(t.shape)} strides {t.stride()}"
                )


rubin_idx_fp4_quantize = RubinIdxFp4Quantize()
