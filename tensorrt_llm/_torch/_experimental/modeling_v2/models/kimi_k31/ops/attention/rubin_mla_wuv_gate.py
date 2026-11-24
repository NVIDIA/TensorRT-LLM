# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W_UV with the MLA output gate in its epilogue: the gated ``o_proj`` input of the sparse MLA prefill (CuTe DSL).

The context rows' latent attention output goes to the value heads through W_UV and is gated with
``sigmoid(g_proj(x))`` before ``o_proj``. This entry writes the gated rows in one kernel (``rubin_mla_head_gemm``):
per head the BF16 product of W_UV, gated with the per-element PTX of ``k31_sigmoid_gate_mul``. Rows past the
attention's rows (padding) are gated zeros, as the two-pass form zeroes them first.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

from ...kernels.attention.rubin_mla_head_gemm import wuv_gate


def _cell(rows: int, rows_lat: int, why: str) -> Cell:
    return Cell(
        why=why, spec=dict(rows=rows, rows_lat=rows_lat, pad=128, heads=96, lora=512, v=128)
    )


class RubinMlaWuvGate(OpWrapper):
    """Fill ``out`` ``[rows, heads * v]`` (BF16, unit column stride): head ``h``'s columns with
    ``bf16(lat[:, h] @ w_uv[h]) * sigmoid(gate)`` as ``k31_sigmoid_gate_mul`` computes it, the attention output
    taken as +0 on rows ``>= lat.shape[0]``. ``lat`` ``[rows_lat, >= heads, lora]`` (unit last-dim stride),
    ``w_uv`` ``[heads, lora, v]`` (unit stride along ``lora`` or ``v``) and ``gate`` ``[rows, heads * v]`` (unit
    column stride) are BF16. Returns ``out``."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        _cell(16384, 16384, "DEP16 / DEP12 / DEP8 prefill, a 16k chunk: 96 of 128 latent heads"),
        _cell(2048, 2048, "cep8 prefill, one rank's 2k rows of a 16k chunk"),
        _cell(4097, 4097, "a ragged chunk: one row past the last full 128-row tile"),
        _cell(40, 37, "padding rows past the attention's rows: gated zeros"),
        _cell(1, 1, "one token"),
    )
    note = """
    Replaces, on the context rows, W_UV's BF16 product (``bmm_out`` into the head columns of ``out``), the
    zeroing of the padding rows and ``k31_sigmoid_gate_mul``. Each FP32 sum is accumulated over ``kv_lora`` in
    order, 16 products per tcgen05 MMA, as the BMM's kernel accumulates it, rounded to BF16 and gated with that
    kernel's per-element PTX (``gate_mul_pair``): ``o_proj`` reads the two-pass form's bytes, which the test
    checks bit for bit at the production shapes.
    """

    def __call__(
        self, out: torch.Tensor, lat: torch.Tensor, w_uv: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        wuv_gate(out, lat, w_uv, gate)
        return out

    def reference(
        self, out: torch.Tensor, lat: torch.Tensor, w_uv: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """A new tensor: the BF16 value heads (zero past ``lat``'s rows) times ``gate.sigmoid()`` in BF16."""
        heads, _, v = w_uv.shape
        attn = torch.zeros(out.shape[0], heads, v, dtype=torch.bfloat16, device=out.device)
        with true_fp32_matmul():
            prod = torch.bmm(lat[:, :heads].transpose(0, 1).float(), w_uv.float())
        attn[: lat.shape[0]] = prod.transpose(0, 1).to(torch.bfloat16)
        return attn.view(out.shape[0], heads * v) * gate.sigmoid()

    def is_valid(
        self, out: torch.Tensor, lat: torch.Tensor, w_uv: torch.Tensor, gate: torch.Tensor
    ) -> None:
        for t in (out, lat, w_uv, gate):
            assert t.dtype == torch.bfloat16
        assert out.dim() == 2 and gate.dim() == 2 and lat.dim() == 3 and w_uv.dim() == 3
        heads, lora, v = w_uv.shape
        assert out.shape == (out.shape[0], heads * v) and gate.shape == out.shape
        assert out.stride(1) == 1 and gate.stride(1) == 1 and lat.stride(-1) == 1, (
            "unit column strides"
        )
        assert w_uv.stride(1) == 1 or w_uv.stride(2) == 1, (
            "W_UV needs a unit stride along lora or v"
        )
        assert lat.shape[0] <= out.shape[0] and lat.shape[1] >= heads and lat.shape[2] == lora
        assert v % 128 == 0 and lora % 64 == 0, (
            "v is computed in 128-wide tiles, lora loaded in 64-wide chunks"
        )
        for t in (out, lat, w_uv, gate):
            assert t.data_ptr() % 16 == 0, "16-byte aligned operands"
        for st in (out.stride(0), gate.stride(0), lat.stride(0), lat.stride(1), w_uv.stride(0)):
            assert st % 8 == 0, "rows and heads 16-byte aligned"


rubin_mla_wuv_gate = RubinMlaWuvGate()
