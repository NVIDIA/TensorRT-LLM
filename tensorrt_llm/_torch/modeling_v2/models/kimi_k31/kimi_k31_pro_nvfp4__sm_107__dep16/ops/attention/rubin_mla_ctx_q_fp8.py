# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 latent columns of the sparse MLA prefill query at 128 heads (Triton).

The DEP16 context server's attention op reads its query as E4M3 ``[tokens, 128, 576]``. This entry writes the
latent columns of that buffer from the absorbed ``q_nope @ W_UK`` rows (BF16): the 96 real heads converted as the
op converts a BF16 query, the padding heads zero. The op's RoPE kernel writes the RoPE columns.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.rubin_mla_ctx_q_fp8 import E4M3_MAX, _ctx_q_fp8


class RubinMlaCtxQFp8(OpWrapper):
    """Fill ``dst`` ``[n, pad, D]`` (E4M3, last dim contiguous): rows ``h < heads`` from ``src`` ``[n, >= heads,
    D]`` (BF16, last dim contiguous) at unit scale, rounded to nearest even and saturated to +-448 (NaN kept), rows
    ``h >= heads`` with zeros. ``D`` must be a power of two. Returns ``dst``."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 prefill query, a 16k chunk: 96 heads of 512 latent columns into the 128-head "
            "FP8 buffer [n, 128, 576] (the RoPE columns stay)",
            spec=dict(n=16384, pad=128, heads=96, d=512, width=576),
        ),
        Cell(
            why="DEP16 prefill query, a ragged chunk",
            spec=dict(n=37, pad=128, heads=96, d=512, width=576),
        ),
        Cell(
            why="one token",
            spec=dict(n=1, pad=128, heads=96, d=512, width=576),
        ),
    )
    note = """
    Replaces, on the context call, the attention op's quantization of the whole BF16 query and the zeroing of
    its padding heads. The FP8 bytes are the ones the op would make of the same BF16 rows: the BMM's product
    is still written in BF16 and only converted here.
    """

    def __call__(self, dst: torch.Tensor, src: torch.Tensor, heads: int) -> torch.Tensor:
        _ctx_q_fp8(dst, src, heads)
        return dst

    def reference(self, dst: torch.Tensor, src: torch.Tensor, heads: int) -> torch.Tensor:
        """A new tensor: ``dst`` with its first ``heads`` heads converted from ``src`` and the others zero."""
        out = dst.clone()
        out[:, heads:] = 0
        out[:, :heads] = src[:, :heads].float().clamp(-E4M3_MAX, E4M3_MAX).to(out.dtype)
        return out

    def is_valid(self, dst: torch.Tensor, src: torch.Tensor, heads: int) -> None:
        # The kernel reads BF16 head rows and stores E4M3 ones with 16-byte accesses along the last dim.
        assert dst.dtype == torch.float8_e4m3fn and src.dtype == torch.bfloat16
        assert dst.stride(-1) == 1 and src.stride(-1) == 1, "last dims must be contiguous"
        assert dst.dim() == 3 and src.dim() == 3
        assert src.shape[0] == dst.shape[0] and src.shape[-1] == dst.shape[-1]
        assert 0 <= heads <= min(dst.shape[1], src.shape[1])
        d = dst.shape[-1]
        assert d > 0 and d & (d - 1) == 0, "the latent width must be a power of two"


rubin_mla_ctx_q_fp8 = RubinMlaCtxQFp8()
