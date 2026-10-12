# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMS normalization over the last dim via the flashinfer rmsnorm kernel.

Everything a target author needs in order to decide whether this op covers
their case is below, and all of it is checked: `reference` is what CI
compares the kernel against, and `CELLS` is what CI drives. Read the cell
list for the certified range.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

# DeepSeek-V4.1's layouts, all bf16 at eps 1e-20:
# (id tag, width, parent row stride or None for contiguous, column offset, call site).
# The q / kv views slice one [T, 1792] qkv buffer; the DSpark drafter's ring
# write slices one [M, 1536] buffer per stage. Each layout is certified at
# every row count 1..16384 by the row sweep in the test; the cells hold the
# sweep's two ends, and at one row a column view takes the compact kernel.
_DSV41_LAYOUTS = (
    ("hidden 5120", 5120, None, 0, "attn / ffn / final norm"),
    ("q 1280@0/1792", 1280, 1792, 0, "q_norm on qkv[:, :1280]"),
    ("kv 512@1280/1792", 512, 1792, 1280, "kv_norm on qkv[:, 1280:]"),
    ("comp 512", 512, None, 0, "compressed-latent norm"),
    ("idx 128", 128, None, 0, "index-K norm"),
    ("ring0 512@0/1536", 512, 1536, 0, "DSpark ring kv_norm, stage 0"),
    ("ring1 512@512/1536", 512, 1536, 512, "DSpark ring kv_norm, stage 1"),
    ("ring2 512@1024/1536", 512, 1536, 1024, "DSpark ring kv_norm, stage 2"),
)
_DSV41_ROWS = (1, 16384)
_DSV41_VIT_ROWS = (1, 51111)

_DSV41_CELLS = tuple(
    Cell(
        why=f"deepseek-v4.1 {tag} rows={rows}: {site}, eps 1e-20, "
        f"{'one row' if rows == 1 else 'top'} of the row sweep 1..16384",
        spec=dict(
            shape=(rows, width),
            dtype=torch.bfloat16,
            eps=1e-20,
            row_stride=stride,
            col_offset=offset,
            data="eps_classes",
        ),
    )
    for tag, width, stride, offset, site in _DSV41_LAYOUTS
    for rows in _DSV41_ROWS
) + tuple(
    Cell(
        why=f"deepseek-v4.1 vision 1024 rows={rows}: vision tower norm1 / norm2 / "
        f"final norm, eps 1e-6, {'one row' if rows == 1 else 'top'} of the row sweep 1..51111",
        spec=dict(shape=(rows, 1024), dtype=torch.bfloat16, eps=1e-6, data="vit"),
    )
    for rows in _DSV41_VIT_ROWS
)


class FlashinferRmsnorm(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * weight`, one op call, new tensor out.

    Fusion boundary: normalization and weight scaling only. No residual add
    (`flashinfer_fused_add_rmsnorm`), no gemma `(1 + w)` form
    (`flashinfer_gemma_rmsnorm`), no quantization. `x` is not mutated.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The widths the shipped targets normalize over. gpt-oss reaches this op
    # only on its residual stream; deepseek reaches it there and on the two
    # lora ranks its MLA projects through, which is why the narrow cells are
    # here at all -- a 512-wide row is one 128-bit vector per lane and takes a
    # different path through the kernel than a 7168-wide one.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden 2880",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="decode-shaped, deepseek-r1 hidden 7168",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek q_lora rank 1536, the q_norm this op is called on",
            spec=dict(shape=(16, 1536), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek kv_lora rank 512, the narrowest row any target asks for",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
        # DeepSeek-V4.1: see `_DSV41_LAYOUTS`. Optional spec keys: `row_stride`
        # and `col_offset` make `x` a column view of a wider parent, `data`
        # names the row-magnitude pattern (the test's `_build` documents both).
        *_DSV41_CELLS,
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    PDL is controlled by `TRTLLM_ENABLE_PDL` (default on) inside the trtllm
    custom op; it changes scheduling, not results, and is not a cell axis.

    fp16 and fp32 inputs are accepted by the op and were measured to work,
    but no shipped target uses them, so they are not cells. TRT-LLM's own
    `RMSNorm` module routes here for fp16/bf16 only.

    `float8_e4m3fn` is the trap this entry cannot guard: the CuTe DSL path's
    dtype table maps it and returns a plausible fp8 rmsnorm rather than
    raising, so a caller who forgets to dequantize gets a result. `is_valid`
    rejects it below. The CUDA-JIT path
    (`FLASHINFER_USE_CUDA_NORM=1`) does raise, but this entry does not
    certify that path.

    DeepSeek-V4.1: bf16 at eps 1e-20 on the layouts in `_DSV41_LAYOUTS`,
    metadata none. Every row count 1..16384 is certified at each layout: the
    16384-row call meets the reference, every prefix of it is bitwise the op
    on that many rows, nothing is written past the last row, and no kernel
    variant compiles per row count. Rows mix unit, eps-scale and zero
    magnitudes, so a kernel that clamped or ignored the tiny eps fails the
    gate; zero rows come back exactly 0. Column views need a 16-byte-aligned
    row stride and start; a one-row view reports contiguous and takes the
    compact kernel, which needs a 128-byte-aligned start (column offset a
    multiple of 64 elements, as every certified site is). A misaligned view
    raises `ValueError`; it is never computed wrongly. A column view's output
    is a new contiguous `[rows, width]` tensor (at one row it keeps the view's
    row stride). Any width-512 view with a 16-byte-aligned row stride up to
    2048 and column offset was measured at 13 row counts up to 160. CUDA graphs
    are certified, after an eager call at the same shape, at the 126 per-rank
    row counts {G, 3G, 5G, 6G} over the 46 captured batch sizes up to 256 for
    the five core layouts, and at 473 counts up to 1280 for the ring views.
    The vision tower calls it at width 1024, eps 1e-6, contiguous `[P, 1024]`
    with +-2000 massive channels, every P 1..51111, captured at P = 1517, 4680,
    5325, 8772 and 17037. Not certified for V4.1: fp16 / fp32 at eps 1e-20;
    rows whose squares underflow fp32 (|x| below about 1e-19, flushed to zero);
    more than 16384 rows (51111 at width 1024); width 1024 at another eps or as
    a view; capture at other row counts, or before the shape's kernel variant
    has compiled; 3-D input at eps 1e-20.
    """

    def raw_call(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_rmsnorm(x, weight, eps)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """fp32-accumulated; the kernel reduces in fp32 and rounds once at the end."""
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        # Only the case the op takes and answers wrongly. Shape and dtype
        # mismatches it rejects itself, loudly, and duplicating that here
        # would only swap its error for a worse one.
        assert x.dtype is not torch.float8_e4m3fn, (
            "float8_e4m3fn is accepted by the kernel's dtype table and returns "
            "a plausible fp8 rmsnorm; dequantize before normalizing"
        )
        assert x.stride(-1) == 1, (
            f"the normalized dim must be contiguous; got stride {x.stride(-1)}"
        )


flashinfer_rmsnorm = FlashinferRmsnorm()
