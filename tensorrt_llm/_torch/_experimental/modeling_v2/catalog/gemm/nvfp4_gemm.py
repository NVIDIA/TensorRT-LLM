# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-scaled NVFP4 matmul via the trtllm nvfp4_gemm op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import ULP, Arch, Cell, OpWrapper, assert_within_ulp

_VEC = 16

#: e2m1 code -> magnitude. code = (exponent << 1) | mantissa, bit 3 is the sign.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _pad_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _swizzled_scale_numel(rows: int, k: int) -> int:
    """Bytes a 128x4 swizzled block-scale buffer holds for `rows` x `k`.

    The swizzle addresses the padded rectangle, not the real one, so the buffer
    is this size whatever `rows` and `k` are.
    """
    return _pad_up(rows, 128) * _pad_up(k // _VEC, 4)


def _swizzle_index(rows: int, cols: int, device) -> torch.Tensor:
    """[rows, cols] scale coordinates -> flat offsets in the swizzled buffer."""
    padded_cols = _pad_up(cols, 4)
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(cols, device=device).view(1, -1)
    return (
        (c % 4)
        + (c // 4) * (4 * 128)
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )


class _Nvfp4Gemm(OpWrapper):
    """`alpha * (act @ weight.T) (+ bias)` over block-scaled NVFP4 operands.

    Both operands are packed e2m1, two values to a byte, with one e4m3 scale per
    16 elements of K held in the 128x4 swizzled order. `alpha` folds the two
    global scales the quantization divided out back into the product.
    """

    ARCHS = frozenset({Arch.SM_103})

    # deepseek-r1's dense and shared-expert MLPs are the only callers: hidden
    # 7168 into 2 x intermediate, then back down. K and N here are the real
    # widths; M is the token count.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode, gate+up: [1, 7168] @ [4096, 7168].T",
            spec=dict(m=1, k=7168, n=4096, bias=False),
        ),
        Cell(
            why="decode, down: [1, 2048] @ [7168, 2048].T",
            spec=dict(m=1, k=2048, n=7168, bias=False),
        ),
        Cell(
            why="prefill chunk through the wider of the two",
            spec=dict(m=2048, k=7168, n=4096, bias=False),
        ),
        Cell(
            why="an M that is not a multiple of the swizzle's 128-row tile",
            spec=dict(m=37, k=7168, n=4096, bias=False),
        ),
    )

    note = """
    `bias`, `group` and `output_buffer_kind` exist on the op and no shipped
    target passes them, so none is a cell axis.

    `allowed_backends` selects among cutlass, cublaslt and cuda_core. The cells
    drive the default string, so which backend answers is the op's choice on the
    day -- that is deliberate, since a target gets the same choice.
    """

    def __call__(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        output_buffer_kind: int = 0,
        allowed_backends: str = "cutlass,cublaslt,cuda_core",
        group: list[int] | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.trtllm.nvfp4_gemm(
            act_fp4,
            weight,
            act_sf,
            weight_scale,
            alpha,
            output_dtype,
            output_buffer_kind,
            allowed_backends,
            group,
            bias,
        )

    def reference(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype,
        output_buffer_kind: int = 0,
        allowed_backends: str = "",
        group: list[int] | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequantize both operands, then one fp32 product.

        Takes the packed form rather than the values it was built from, so the
        reference covers the unpacking and the scale layout too -- a swizzle the
        kernel and this disagreed about would otherwise never show up.
        """
        a = self._dequantize(act_fp4, act_sf)
        b = self._dequantize(weight, weight_scale)
        out = alpha.float().reshape(()) * (a @ b.T)
        if bias is not None:
            out = out + bias.float()
        return out.to(output_dtype)

    @staticmethod
    def _dequantize(packed: torch.Tensor, swizzled_sf: torch.Tensor) -> torch.Tensor:
        """[R, K/2] packed e2m1 plus a swizzled scale buffer -> [R, K] fp32."""
        rows, half = packed.shape
        k = half * 2
        codes = torch.empty(rows, k, dtype=torch.uint8, device=packed.device)
        codes[:, 0::2] = packed & 0xF
        codes[:, 1::2] = packed >> 4
        table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
        value = table[(codes & 7).long()]
        value = torch.where((codes & 8).bool(), -value, value)

        # Gather the scales back out of the swizzle -- the inverse of the
        # scatter that put them there.
        cols = k // _VEC
        index = _swizzle_index(rows, cols, packed.device).reshape(-1)
        scale = swizzled_sf.reshape(-1)[index].reshape(rows, cols)
        scale = scale.view(torch.float8_e4m3fn).float()
        return value * scale.repeat_interleave(_VEC, dim=-1)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Two fp4 operands accumulated over K, so the band is the operands'.

        The values are exact on both sides -- e2m1 codes and e4m3 scales decode
        to the same numbers -- and the only difference is the order K is summed
        in. Over a K of 7168 a few outputs cancel toward zero, where torch's
        per-element rtol is meaningless. Scored in bf16 ulp because that is what
        the output is asked to hold; 8 ulp element-wise and 4 of relative RMS,
        the same band a two-GEMM chain gets, because the fp4 operands make each
        product coarser than a bf16 one.
        """
        assert_within_ulp(out, ref, element_ulp=8.0, rms_ulp=4.0, ulp=ULP[torch.bfloat16])

    def is_valid(
        self,
        act_fp4: torch.Tensor,
        weight: torch.Tensor,
        act_sf: torch.Tensor,
        weight_scale: torch.Tensor,
        alpha: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
        *unused_args: object,
        **unused_kwargs: object,
    ) -> None:
        # Every guard is pure metadata, and every guarded violation was observed
        # to be read wrongly *without raising* by at least one backend the
        # default `allowed_backends` string can select: cutlass checks
        # contiguity and raises, cublaslt does not, and none of the three
        # rejects an oversized `alpha`.
        assert act_fp4.is_contiguous() and weight.is_contiguous(), (
            "act_fp4 [M, K/2] and weight [N, K/2] must be contiguous; the cublaslt "
            "backend ignores strides and returns wrong results"
        )
        assert act_sf.is_contiguous() and weight_scale.is_contiguous(), (
            "act_sf and weight_scale must be contiguous; the cublaslt backend "
            "ignores strides and returns wrong results"
        )
        assert alpha.numel() == 1, (
            "alpha must hold exactly one element; extra elements are silently "
            "ignored (this build has no per-token alpha)"
        )
        # Contiguity says nothing about length, and a short scale buffer is read
        # past its end rather than rejected: the kernel indexes the *padded*
        # rectangle. Both operands are checked because M and N pad
        # independently. Only for 2-D operands and only for under-length --
        # wrong rank, zero rows and K disagreement the op rejects itself and
        # loudly, and preempting that would swap its error for a worse one.
        if act_fp4.dim() == 2 and weight.dim() == 2:
            k = act_fp4.shape[1] * 2  # two 4-bit values per byte
            floor_act = _swizzled_scale_numel(act_fp4.shape[0], k)
            floor_weight = _swizzled_scale_numel(weight.shape[0], k)
            assert act_sf.numel() >= floor_act, (
                f"act_sf must hold at least pad_up(M,128) * pad_up(K/16,4) = "
                f"{floor_act} bytes for M={act_fp4.shape[0]}, K={k}; got "
                f"{act_sf.numel()} -- the kernel would read past its end"
            )
            assert weight_scale.numel() >= floor_weight, (
                f"weight_scale must hold at least pad_up(N,128) * pad_up(K/16,4) = "
                f"{floor_weight} bytes for N={weight.shape[0]}, K={k}; got "
                f"{weight_scale.numel()} -- the kernel would read past its end"
            )


nvfp4_gemm = _Nvfp4Gemm()
