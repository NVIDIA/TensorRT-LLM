# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W_UK with an E4M3 epilogue: the FP8 latent columns of the padded sparse MLA prefill query (CuTe DSL).

The context server's attention op reads its query as E4M3 ``[tokens, 128, 576]``. This entry writes the latent
columns of that buffer straight from ``q_nope`` and W_UK in one kernel (``rubin_mla_head_gemm``): per real head the
absorbed product rounded to BF16 and converted as the op converts a BF16 query, the padding heads zero. The op's
RoPE kernel writes the RoPE columns.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.attention.rubin_mla_head_gemm import wuk_fp8

E4M3_MAX = 448.0


def _cell(n: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(n=n, pad=128, heads=96, nope=128, rope=64, lora=512))


class RubinMlaCtxWukFp8(OpWrapper):
    """Fill ``dst`` ``[n, pad, lora]`` (E4M3, unit last-dim stride, 16-byte aligned rows and heads): head
    ``h < heads`` with ``q_nope[:, h] @ w_uk[h]`` (FP32 accumulation, rounded to BF16) at unit scale, rounded to
    nearest even and saturated to +-448 (NaN kept); heads ``>= heads`` with zeros. ``q_nope`` ``[n, heads, nope]``
    and ``w_uk`` ``[heads, nope, lora]`` are BF16 with unit last-dim strides; ``nope`` a multiple of 64 and
    ``lora`` of 256. Returns ``dst``."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        _cell(
            16384,
            "DEP16 / DEP12 / DEP8 prefill query, a 16k chunk: 96 heads into the 128-head buffer",
        ),
        _cell(2048, "cep8 prefill query, one rank's 2k rows of a 16k chunk"),
        _cell(4097, "a ragged chunk: one row past the last full 128-row tile"),
        _cell(37, "a short prompt"),
        _cell(1, "one token"),
    )
    note = """
    Replaces, on the context call, W_UK's BF16 product (``bmm_out`` into a BF16 latent buffer) and
    ``rubin_mla_ctx_q_fp8``'s conversion of it. Each FP32 sum is accumulated over the 128 NoPE columns in
    order, 16 products per tcgen05 MMA, as the BMM's kernel accumulates it, and rounded to BF16 before the
    conversion: the bytes are the two-pass form's, which the test checks bit for bit at the production shapes.
    """

    def __call__(self, dst: torch.Tensor, q_nope: torch.Tensor, w_uk: torch.Tensor) -> torch.Tensor:
        wuk_fp8(dst, q_nope, w_uk)
        return dst

    def reference(
        self, dst: torch.Tensor, q_nope: torch.Tensor, w_uk: torch.Tensor
    ) -> torch.Tensor:
        """A new tensor: ``dst`` with the converted products on the real heads and zeros on the others."""
        heads = w_uk.shape[0]
        with true_fp32_matmul():
            prod = torch.bmm(q_nope.transpose(0, 1).float(), w_uk.float()).to(torch.bfloat16)
        out = dst.clone()
        out[:, heads:] = 0
        out[:, :heads] = prod.transpose(0, 1).float().clamp(-E4M3_MAX, E4M3_MAX).to(out.dtype)
        return out

    def is_valid(self, dst: torch.Tensor, q_nope: torch.Tensor, w_uk: torch.Tensor) -> None:
        assert dst.dtype == torch.float8_e4m3fn
        assert q_nope.dtype == torch.bfloat16 and w_uk.dtype == torch.bfloat16
        assert dst.dim() == 3 and q_nope.dim() == 3 and w_uk.dim() == 3
        assert dst.stride(-1) == 1 and q_nope.stride(-1) == 1 and w_uk.stride(-1) == 1, (
            "last dims must be contiguous"
        )
        heads, nope, lora = w_uk.shape
        assert q_nope.shape == (dst.shape[0], heads, nope)
        assert dst.shape[1] >= heads and dst.shape[2] == lora
        assert nope % 64 == 0 and lora % 256 == 0, (
            "nope is loaded in 64-wide chunks, lora in 256-wide tiles"
        )
        assert dst.data_ptr() % 16 == 0 and dst.stride(0) % 16 == 0 and dst.stride(1) % 16 == 0, (
            "the FP8 rows are stored in 16-byte vectors"
        )
        assert q_nope.data_ptr() % 16 == 0 and w_uk.data_ptr() % 16 == 0
        for st in (q_nope.stride(0), q_nope.stride(1), w_uk.stride(0), w_uk.stride(1)):
            assert st % 8 == 0, "TMA strides must be multiples of 16 bytes"


rubin_mla_ctx_wuk_fp8 = RubinMlaCtxWukFp8()
