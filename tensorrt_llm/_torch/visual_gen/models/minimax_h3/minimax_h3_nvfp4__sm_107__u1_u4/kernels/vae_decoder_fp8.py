# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 per-tensor GEMMs for the MiniMax-H3 VAE ViT decoder (``vae_fp8=True`` by default).

The decoder's 36 transformer blocks run once per clip in fp16 autocast and are GEMM-bound
(fp16 ``nvjet`` GEMM kernels dominate the decode). Every ``nn.Linear`` under the
decoder blocks is replaced by :class:`Fp8Linear`: the weight is quantized once to e4m3 with a
static per-tensor scale, the activation is quantized per call with a dynamic per-tensor scale,
and ``torch._scaled_mm`` produces the fp16 output with the original bias. This changes the
decoded pixels (a format change, gated by the quality worker), not the latents.

``vae_fp8_tiled=1`` additionally routes the two largest activation quantizes of each
block (the SwiGLU output feeding the down projection, and the attention output feeding
``to_out``) through tiled Triton kernels exposed as custom ops instead of the
``torch.compile``-generated split reductions. The kernels reproduce Inductor's arithmetic
(SiLU as ``g / (1 + exp(-g))`` in fp32, fp32 product, the amax stored as fp16, the cast's scale
recomputed from the fp16 amax, the GEMM's scale from the fp32 amax), so the e4m3 codes, the
GEMM inputs and therefore the decoded pixels are bit-identical to the non-tiled path.

``vae_fp8_tile=BT,BC,WARPS`` / ``vae_fp8_qtile=BT,BH,WARPS`` pick the tile of the SwiGLU /
attention-output kernels (default 8,512,4 / 16,4,4). The amax is a max and the cast is
element-wise, so the tile changes the schedule only, never a code.

``vae_fused_norm=1`` (needs the tiled path) removes the redundant work the compiled block
does around its norms, with the same arithmetic in the same order:

* the QKV projections quantize their shared input once (one amax, one scale, one e4m3 buffer)
  instead of three identical times;
* the q / k RMSNorm + RoPE is written as one 64-lane expression (a ``where`` over the rotary and
  pass-through lanes with a gathered partner lane) so Inductor emits one kernel instead of a
  48-lane kernel plus two 16-lane copies; the products and sums keep Inductor's order;
* the tiled quantize ops return the GEMM scale beside the amax (the scale is a monotone function
  of the amax, so the per-tile atomic max of the scale is the scale of the max) and the cast
  rounds the amax to fp16 in-kernel, which removes the four one-element kernels per block.
"""

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ..config import SCENARIO

E4M3_MAX = 448.0
INV_E4M3_MAX = 0.002232142857142857  # Inductor's constant for "/ 448.0"
_TILED = SCENARIO.vae_fp8_tiled
_FUSED = _TILED and SCENARIO.vae_fused_norm


def _parse_tile(name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
    value = getattr(SCENARIO, name, None)
    if not value:
        return default
    parts = tuple(int(p) for p in value)
    if len(parts) != 3:
        raise ValueError(f"{name} must be BT,BC,WARPS, got {value!r}")
    return parts


class Fp8Linear(nn.Module):
    """``nn.Linear`` replacement: e4m3 weight (static per-tensor scale), e4m3 activation (dynamic
    per-tensor scale), fp32 accumulation, fp16 output via ``torch._scaled_mm``."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach().to(torch.float32)
        amax = weight.abs().amax().clamp(min=1e-12)
        w_scale = (amax / E4M3_MAX).to(torch.float32).reshape(())
        w_fp8 = (weight / w_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("weight_fp8", w_fp8.contiguous())  # (out, in) row-major
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(torch.float16).contiguous())
        else:
            self.bias = None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        x_scale = (x.abs().amax().to(torch.float32).clamp(min=1e-12) / E4M3_MAX).reshape(())
        x_fp8 = (x.to(torch.float32) / x_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # (in, out) column-major, as cuBLASLt FP8 requires
            scale_a=x_scale,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.float16,
        )
        return out.reshape(*shape[:-1], self.out_features)


class Fp8QKVShared(nn.Module):
    """The three attention input projections on one quantized copy of their shared input.

    Same amax, scale and e4m3 expression as :class:`Fp8Linear` (Inductor fuses them the same way),
    computed once instead of three times; the three ``_scaled_mm`` calls see the same codes and
    scale they saw before, so q / k / v are bit-identical.
    """

    def __init__(self, q: Fp8Linear, k: Fp8Linear, v: Fp8Linear) -> None:
        super().__init__()
        self.q, self.k, self.v = q, k, v

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        x_scale = (x.abs().amax().to(torch.float32).clamp(min=1e-12) / E4M3_MAX).reshape(())
        x_fp8 = (x.to(torch.float32) / x_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        outs = []
        for lin in (self.q, self.k, self.v):
            out = torch._scaled_mm(
                x_fp8,
                lin.weight_fp8.t(),
                scale_a=x_scale,
                scale_b=lin.w_scale,
                bias=lin.bias,
                out_dtype=torch.float16,
            )
            outs.append(out.reshape(*shape[:-1], lin.out_features))
        return outs[0], outs[1], outs[2]


# ---------------------------------------------------------------------------------------------
# Tiled quantize path (vae_fp8_tiled=1)
# ---------------------------------------------------------------------------------------------
if _TILED:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HEAD_DIM = 64
    _MLP_TILE = _parse_tile("vae_fp8_tile", (8, 512, 4))
    _Q_TILE = _parse_tile("vae_fp8_qtile", (16, 4, 4))
    _KN = dict(
        mlp_bt=_MLP_TILE[0],
        mlp_bc=_MLP_TILE[1],
        mlp_warps=_MLP_TILE[2],
        q_bt=_Q_TILE[0],
        q_bh=_Q_TILE[1],
        q_warps=_Q_TILE[2],
    )

    def set_tiles(
        mlp: tuple[int, int, int] | None = None, q: tuple[int, int, int] | None = None
    ) -> dict:
        """Change the kernel tiles at run time (the tile sweep of the V.1 fixture); returns the table."""
        if mlp is not None:
            _KN.update(mlp_bt=mlp[0], mlp_bc=mlp[1], mlp_warps=mlp[2])
        if q is not None:
            _KN.update(q_bt=q[0], q_bh=q[1], q_warps=q[2])
        return dict(_KN)

    @triton.jit
    def _silu_mul(x, g):
        # Inductor: g / (libdevice.exp(-g) + 1.0), then x * that, all fp32
        return x * (g / (libdevice.exp(-g) + 1.0))

    @triton.jit
    def _scale_from_amax16(amax16_ptr):
        # Inductor's cast kernel: stored fp16 amax -> fp32, max(., 1e-12), * (1/448)
        a = tl.load(amax16_ptr).to(tl.float32)
        return tl.maximum(a, 1e-12) * 0.002232142857142857

    @triton.jit
    def _scale_from_amax32_as16(amax32_ptr):
        # The same scale from the fp32 amax: round to fp16 (RNE, as .to(float16) does), back to fp32
        a = tl.load(amax32_ptr).to(tl.float16).to(tl.float32)
        return tl.maximum(a, 1e-12) * 0.002232142857142857

    @triton.jit
    def _silu_mul_amax_kernel(
        h_ptr, out_ptr, T, HALF, stride_t, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < T) & (cols[None, :] < HALF)
        off = rows[:, None] * stride_t + cols[None, :]
        x = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(h_ptr + off + HALF, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(_silu_mul(x, g)))
        tl.atomic_max(out_ptr, m)

    @triton.jit
    def _silu_mul_amax2_kernel(
        h_ptr, out_ptr, T, HALF, stride_t, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        # out[0] = amax (fp32), out[1] = the GEMM scale max(amax, 1e-12) * (1/448): the scale is a
        # non-decreasing function of the amax, so the max over tiles of the scale is the scale of the max
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < T) & (cols[None, :] < HALF)
        off = rows[:, None] * stride_t + cols[None, :]
        x = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(h_ptr + off + HALF, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(_silu_mul(x, g)))
        tl.atomic_max(out_ptr, m)
        tl.atomic_max(out_ptr + 1, tl.maximum(m, 1e-12) * 0.002232142857142857)

    @triton.jit
    def _silu_mul_cast_kernel(
        h_ptr, amax16_ptr, out_ptr, T, HALF, stride_t, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < T) & (cols[None, :] < HALF)
        off = rows[:, None] * stride_t + cols[None, :]
        x = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(h_ptr + off + HALF, mask=mask, other=0.0).to(tl.float32)
        q = _silu_mul(x, g) / _scale_from_amax16(amax16_ptr)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(out_ptr + rows[:, None] * HALF + cols[None, :], q.to(tl.float8e4nv), mask=mask)

    @triton.jit
    def _silu_mul_cast2_kernel(
        h_ptr, amax32_ptr, out_ptr, T, HALF, stride_t, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr
    ):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < T) & (cols[None, :] < HALF)
        off = rows[:, None] * stride_t + cols[None, :]
        x = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(h_ptr + off + HALF, mask=mask, other=0.0).to(tl.float32)
        q = _silu_mul(x, g) / _scale_from_amax32_as16(amax32_ptr)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(out_ptr + rows[:, None] * HALF + cols[None, :], q.to(tl.float8e4nv), mask=mask)

    @triton.jit
    def _amax3_kernel(
        x_ptr,
        out_ptr,
        N,
        T,
        H,
        stride_b,
        stride_t,
        stride_h,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # rows run over the flattened [B, T] leading dims: row = b * T + t
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ds = tl.arange(0, D)
        mask = (rows[:, None, None] < N) & (hs[None, :, None] < H) & (ds[None, None, :] < D)
        row_off = (rows // T) * stride_b + (rows % T) * stride_t
        off = row_off[:, None, None] + hs[None, :, None] * stride_h + ds[None, None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_max(out_ptr, tl.max(tl.abs(x)))

    @triton.jit
    def _amax3_2_kernel(
        x_ptr,
        out_ptr,
        N,
        T,
        H,
        stride_b,
        stride_t,
        stride_h,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ds = tl.arange(0, D)
        mask = (rows[:, None, None] < N) & (hs[None, :, None] < H) & (ds[None, None, :] < D)
        row_off = (rows // T) * stride_b + (rows % T) * stride_t
        off = row_off[:, None, None] + hs[None, :, None] * stride_h + ds[None, None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(x))
        tl.atomic_max(out_ptr, m)
        tl.atomic_max(out_ptr + 1, tl.maximum(m, 1e-12) * 0.002232142857142857)

    @triton.jit
    def _cast3_kernel(
        x_ptr,
        amax16_ptr,
        out_ptr,
        N,
        T,
        H,
        stride_b,
        stride_t,
        stride_h,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ds = tl.arange(0, D)
        mask = (rows[:, None, None] < N) & (hs[None, :, None] < H) & (ds[None, None, :] < D)
        row_off = (rows // T) * stride_b + (rows % T) * stride_t
        off = row_off[:, None, None] + hs[None, :, None] * stride_h + ds[None, None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        q = x / _scale_from_amax16(amax16_ptr)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        out_off = rows[:, None, None] * (H * D) + hs[None, :, None] * D + ds[None, None, :]
        tl.store(out_ptr + out_off, q.to(tl.float8e4nv), mask=mask)

    @triton.jit
    def _cast3_2_kernel(
        x_ptr,
        amax32_ptr,
        out_ptr,
        N,
        T,
        H,
        stride_b,
        stride_t,
        stride_h,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ds = tl.arange(0, D)
        mask = (rows[:, None, None] < N) & (hs[None, :, None] < H) & (ds[None, None, :] < D)
        row_off = (rows // T) * stride_b + (rows % T) * stride_t
        off = row_off[:, None, None] + hs[None, :, None] * stride_h + ds[None, None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        q = x / _scale_from_amax32_as16(amax32_ptr)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        out_off = rows[:, None, None] * (H * D) + hs[None, :, None] * D + ds[None, None, :]
        tl.store(out_ptr + out_off, q.to(tl.float8e4nv), mask=mask)

    def _mlp_grid(T, half):
        return (triton.cdiv(T, _KN["mlp_bt"]), triton.cdiv(half, _KN["mlp_bc"]))

    @torch.library.custom_op("h3vae::silu_mul_amax", mutates_args=())
    def _silu_mul_amax(h: torch.Tensor) -> torch.Tensor:
        h2 = h.reshape(-1, h.shape[-1])
        T, C = h2.shape
        half = C // 2
        out = torch.zeros((), dtype=torch.float32, device=h.device)
        _silu_mul_amax_kernel[_mlp_grid(T, half)](
            h2,
            out,
            T,
            half,
            h2.stride(0),
            BLOCK_T=_KN["mlp_bt"],
            BLOCK_C=_KN["mlp_bc"],
            num_warps=_KN["mlp_warps"],
        )
        return out

    @_silu_mul_amax.register_fake
    def _(h):
        return torch.empty((), dtype=torch.float32, device=h.device)

    @torch.library.custom_op("h3vae::silu_mul_amax2", mutates_args=())
    def _silu_mul_amax2(h: torch.Tensor) -> torch.Tensor:
        """[amax32, gemm_scale32] of silu(g) * x over the whole tensor, in one pass."""
        h2 = h.reshape(-1, h.shape[-1])
        T, C = h2.shape
        half = C // 2
        out = torch.zeros((2,), dtype=torch.float32, device=h.device)
        _silu_mul_amax2_kernel[_mlp_grid(T, half)](
            h2,
            out,
            T,
            half,
            h2.stride(0),
            BLOCK_T=_KN["mlp_bt"],
            BLOCK_C=_KN["mlp_bc"],
            num_warps=_KN["mlp_warps"],
        )
        return out

    @_silu_mul_amax2.register_fake
    def _(h):
        return torch.empty((2,), dtype=torch.float32, device=h.device)

    @torch.library.custom_op("h3vae::silu_mul_cast", mutates_args=())
    def _silu_mul_cast(h: torch.Tensor, amax16: torch.Tensor) -> torch.Tensor:
        h2 = h.reshape(-1, h.shape[-1])
        T, C = h2.shape
        half = C // 2
        out = torch.empty((T, half), dtype=torch.float8_e4m3fn, device=h.device)
        _silu_mul_cast_kernel[_mlp_grid(T, half)](
            h2,
            amax16,
            out,
            T,
            half,
            h2.stride(0),
            BLOCK_T=_KN["mlp_bt"],
            BLOCK_C=_KN["mlp_bc"],
            num_warps=_KN["mlp_warps"],
        )
        return out

    @_silu_mul_cast.register_fake
    def _(h, amax16):
        return torch.empty(
            (h.numel() // h.shape[-1], h.shape[-1] // 2), dtype=torch.float8_e4m3fn, device=h.device
        )

    @torch.library.custom_op("h3vae::silu_mul_cast2", mutates_args=())
    def _silu_mul_cast2(h: torch.Tensor, amax2: torch.Tensor) -> torch.Tensor:
        """The cast with the scale derived in-kernel from amax2[0] rounded to fp16 (what the fp16 amax was)."""
        h2 = h.reshape(-1, h.shape[-1])
        T, C = h2.shape
        half = C // 2
        out = torch.empty((T, half), dtype=torch.float8_e4m3fn, device=h.device)
        _silu_mul_cast2_kernel[_mlp_grid(T, half)](
            h2,
            amax2,
            out,
            T,
            half,
            h2.stride(0),
            BLOCK_T=_KN["mlp_bt"],
            BLOCK_C=_KN["mlp_bc"],
            num_warps=_KN["mlp_warps"],
        )
        return out

    @_silu_mul_cast2.register_fake
    def _(h, amax2):
        return torch.empty(
            (h.numel() // h.shape[-1], h.shape[-1] // 2), dtype=torch.float8_e4m3fn, device=h.device
        )

    def _as_bthd(x: torch.Tensor):
        """A [B, T, H, D] attention output (any B / T / H strides, D contiguous; the decoder
        batches up to 16 tiles so B is 1..16) or a contiguous [..., C] activation as
        (view, B, T, H, D, stride_b, stride_t, stride_h)."""
        if x.dim() == 4:
            B, T, H, D = x.shape
            if x.stride(3) != 1:
                raise ValueError("tiled quantize expects a [B, T, H, D] view with contiguous D")
            return x, B, T, H, D, x.stride(0), x.stride(1), x.stride(2)
        x2 = x.reshape(-1, x.shape[-1])
        T, C = x2.shape
        D = _HEAD_DIM
        return x2, 1, T, C // D, D, 0, x2.stride(0), D

    def _q_grid(B, T, H):
        return (triton.cdiv(B * T, _KN["q_bt"]), triton.cdiv(H, _KN["q_bh"]))

    @torch.library.custom_op("h3vae::amax", mutates_args=())
    def _amax(x: torch.Tensor) -> torch.Tensor:
        v, B, T, H, D, sb, st, sh = _as_bthd(x)
        out = torch.zeros((), dtype=torch.float32, device=x.device)
        _amax3_kernel[_q_grid(B, T, H)](
            v,
            out,
            B * T,
            T,
            H,
            sb,
            st,
            sh,
            D=D,
            BLOCK_T=_KN["q_bt"],
            BLOCK_H=_KN["q_bh"],
            num_warps=_KN["q_warps"],
        )
        return out

    @_amax.register_fake
    def _(x):
        return torch.empty((), dtype=torch.float32, device=x.device)

    @torch.library.custom_op("h3vae::amax2", mutates_args=())
    def _amax2(x: torch.Tensor) -> torch.Tensor:
        v, B, T, H, D, sb, st, sh = _as_bthd(x)
        out = torch.zeros((2,), dtype=torch.float32, device=x.device)
        _amax3_2_kernel[_q_grid(B, T, H)](
            v,
            out,
            B * T,
            T,
            H,
            sb,
            st,
            sh,
            D=D,
            BLOCK_T=_KN["q_bt"],
            BLOCK_H=_KN["q_bh"],
            num_warps=_KN["q_warps"],
        )
        return out

    @_amax2.register_fake
    def _(x):
        return torch.empty((2,), dtype=torch.float32, device=x.device)

    @torch.library.custom_op("h3vae::cast", mutates_args=())
    def _cast(x: torch.Tensor, amax16: torch.Tensor) -> torch.Tensor:
        v, B, T, H, D, sb, st, sh = _as_bthd(x)
        out = torch.empty((B * T, H * D), dtype=torch.float8_e4m3fn, device=x.device)
        _cast3_kernel[_q_grid(B, T, H)](
            v,
            amax16,
            out,
            B * T,
            T,
            H,
            sb,
            st,
            sh,
            D=D,
            BLOCK_T=_KN["q_bt"],
            BLOCK_H=_KN["q_bh"],
            num_warps=_KN["q_warps"],
        )
        return out

    @_cast.register_fake
    def _(x, amax16):
        if x.dim() == 4:
            B, T, H, D = x.shape
            return torch.empty((B * T, H * D), dtype=torch.float8_e4m3fn, device=x.device)
        return torch.empty(
            (x.numel() // x.shape[-1], x.shape[-1]), dtype=torch.float8_e4m3fn, device=x.device
        )

    @torch.library.custom_op("h3vae::cast2", mutates_args=())
    def _cast2(x: torch.Tensor, amax2: torch.Tensor) -> torch.Tensor:
        v, B, T, H, D, sb, st, sh = _as_bthd(x)
        out = torch.empty((B * T, H * D), dtype=torch.float8_e4m3fn, device=x.device)
        _cast3_2_kernel[_q_grid(B, T, H)](
            v,
            amax2,
            out,
            B * T,
            T,
            H,
            sb,
            st,
            sh,
            D=D,
            BLOCK_T=_KN["q_bt"],
            BLOCK_H=_KN["q_bh"],
            num_warps=_KN["q_warps"],
        )
        return out

    @_cast2.register_fake
    def _(x, amax2):
        if x.dim() == 4:
            B, T, H, D = x.shape
            return torch.empty((B * T, H * D), dtype=torch.float8_e4m3fn, device=x.device)
        return torch.empty(
            (x.numel() // x.shape[-1], x.shape[-1]), dtype=torch.float8_e4m3fn, device=x.device
        )

    def _gemm_scale(amax32: torch.Tensor) -> torch.Tensor:
        # Inductor computes the GEMM's scale_a from the fp32 amax held in registers
        return (torch.clamp(amax32, min=1e-12) * INV_E4M3_MAX).reshape(())

    class Fp8LinearTiled(nn.Module):
        """``Fp8Linear`` whose activation amax and cast run as the tiled kernels; accepts the
        [B, T, H, D] attention output directly (B = tiles per decoder call)."""

        def __init__(self, lin: Fp8Linear) -> None:
            super().__init__()
            self.lin = lin
            self.out_features = lin.out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lead = (
                x.shape[:2] if x.dim() == 4 else x.shape[:-1]
            )  # [B, T] rows of the [B*T, out] GEMM
            if _FUSED:
                amax2 = torch.ops.h3vae.amax2(x)
                codes = torch.ops.h3vae.cast2(x, amax2)
                scale_a = amax2[1]
            else:
                amax32 = torch.ops.h3vae.amax(x)
                codes = torch.ops.h3vae.cast(x, amax32.to(torch.float16))
                scale_a = _gemm_scale(amax32)
            out = torch._scaled_mm(
                codes,
                self.lin.weight_fp8.t(),
                scale_a=scale_a,
                scale_b=self.lin.w_scale,
                bias=self.lin.bias,
                out_dtype=torch.float16,
            )
            return out.reshape(*lead, self.out_features)

    class Fp8SwiGLUDown(nn.Module):
        """Replaces ``FeedForward.net`` = [SwiGLU(proj), Dropout, Fp8Linear(down)]: the proj GEMM, the
        fused silu*mul amax and cast, the down GEMM."""

        def __init__(self, proj: Fp8Linear, down: Fp8Linear) -> None:
            super().__init__()
            self.proj = proj
            self.down = down

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.proj(x)
            if _FUSED:
                amax2 = torch.ops.h3vae.silu_mul_amax2(h)
                codes = torch.ops.h3vae.silu_mul_cast2(h, amax2)
                scale_a = amax2[1]
            else:
                amax32 = torch.ops.h3vae.silu_mul_amax(h)
                codes = torch.ops.h3vae.silu_mul_cast(h, amax32.to(torch.float16))
                scale_a = _gemm_scale(amax32)
            out = torch._scaled_mm(
                codes,
                self.down.weight_fp8.t(),
                scale_a=scale_a,
                scale_b=self.down.w_scale,
                bias=self.down.bias,
                out_dtype=torch.float16,
            )
            return out.reshape(*x.shape[:-1], self.down.out_features)

    def _tiled_processor_type():
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
            MiniMaxH3VideoAttnProcessor,
        )

        class TiledAttnProcessor(MiniMaxH3VideoAttnProcessor):
            """The stock processor, except that the attention output reaches ``to_out`` as the strided
            ``[B, T, H, D]`` view (no flatten copy in front of the opaque quantize op)."""

            def __call__(self, attn, hidden_states, rotary_emb=None):
                query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
                key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
                value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
                query = attn.norm_q(query.float()).to(query.dtype)
                key = attn.norm_k(key.float()).to(key.dtype)
                if rotary_emb is not None:
                    cos, sin = rotary_emb
                    cos = cos.to(query.dtype)
                    sin = sin.to(query.dtype)
                    rotary_dim = cos.shape[-1]
                    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
                    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
                    query_first, query_second = query_rotary.chunk(2, dim=-1)
                    key_first, key_second = key_rotary.chunk(2, dim=-1)
                    query_rotated = torch.cat([-query_second, query_first], dim=-1)
                    key_rotated = torch.cat([-key_second, key_first], dim=-1)
                    query = torch.cat(
                        [query_rotary * cos + query_rotated * sin, query_pass], dim=-1
                    )
                    key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)
                hidden_states = dispatch_attention_fn(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
                return attn.to_out[0](hidden_states)

        return TiledAttnProcessor

    class PaddedRope(nn.Module):
        """Wraps the decoder's rotary embedding: cos / sin padded from the 48 rotary lanes to the 64 head
        lanes (zeros in the pass-through lanes, which the processor never reads), plus the partner-lane
        index (lane i pairs with i + 24 or i - 24), the sign of the rotated partner (-1 for the first
        half, +1 for the second) and the rotary-lane mask. Computed once per decoder call."""

        def __init__(self, rope: nn.Module, head_dim: int = 64) -> None:
            super().__init__()
            self.rope = rope
            self.head_dim = head_dim

        def forward(self, position_ids: torch.Tensor):
            cos, sin = self.rope(position_ids)  # [B, T, 1, R] fp32
            rotary_dim = cos.shape[-1]
            half = rotary_dim // 2
            pad = self.head_dim - rotary_dim
            cos = torch.nn.functional.pad(cos, (0, pad))
            sin = torch.nn.functional.pad(sin, (0, pad))
            lanes = torch.arange(self.head_dim, device=cos.device)
            perm = torch.where(
                lanes < half, lanes + half, torch.where(lanes < rotary_dim, lanes - half, lanes)
            )
            sign = torch.where(lanes < half, -1.0, 1.0).to(torch.float32)
            rot_mask = lanes < rotary_dim
            return cos, sin, perm, sign, rot_mask

    def _fused_processor_type():
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
            MiniMaxH3VideoAttnProcessor,
        )

        class FusedAttnProcessor(MiniMaxH3VideoAttnProcessor):
            """One quantize for q / k / v, the q / k norm + RoPE as one 64-lane expression (the rotary
            product and sum in the stock order, the pass-through lanes selected by ``where``), the
            attention output to ``to_out`` as the strided view."""

            def __call__(self, attn, hidden_states, rotary_emb=None):
                query, key, value = attn.qkv_shared(hidden_states)
                query = query.unflatten(2, (attn.heads, -1))
                key = key.unflatten(2, (attn.heads, -1))
                value = value.unflatten(2, (attn.heads, -1))
                query = attn.norm_q(query.float()).to(query.dtype)
                key = attn.norm_k(key.float()).to(key.dtype)
                if rotary_emb is not None:
                    cos, sin, perm, sign, rot_mask = rotary_emb
                    cos = cos.to(query.dtype)
                    sin = sin.to(query.dtype)
                    sign = sign.to(query.dtype)
                    query_rotated = query.index_select(-1, perm) * sign
                    key_rotated = key.index_select(-1, perm) * sign
                    query = torch.where(rot_mask, query * cos + query_rotated * sin, query)
                    key = torch.where(rot_mask, key * cos + key_rotated * sin, key)
                hidden_states = dispatch_attention_fn(
                    query,
                    key,
                    value,
                    attn_mask=None,
                    backend=self._attention_backend,
                    parallel_config=self._parallel_config,
                )
                return attn.to_out[0](hidden_states)

        return FusedAttnProcessor

    def _tile_block(block: nn.Module, processor_type: type) -> None:
        swiglu, down = block.ff.net[0], block.ff.net[2]
        block.ff.net = nn.ModuleList([Fp8SwiGLUDown(swiglu.proj, down)])
        block.attn.to_out[0] = Fp8LinearTiled(block.attn.to_out[0])
        if _FUSED:
            block.attn.qkv_shared = Fp8QKVShared(block.attn.to_q, block.attn.to_k, block.attn.to_v)
        block.attn.set_processor(processor_type())


def convert_decoder_linears_fp8(decoder: nn.Module) -> int:
    """Replace every ``nn.Linear`` under ``decoder.transformer_blocks`` with :class:`Fp8Linear`.

    ``proj_in`` (K = 24) and ``proj_out`` run once per decoder call and stay fp16. Returns the
    number of replaced modules. Call after ``_prepare_vae_decoder`` and before ``torch_compile``.
    With ``vae_fp8_tiled=1`` the SwiGLU-output and attention-output quantizes of every
    block then move to the tiled kernels (bit-identical codes).
    """
    processor_type = None
    if _TILED:
        # Create separate instances of one type so Dynamo can reuse guards across layers.
        processor_type = _fused_processor_type() if _FUSED else _tiled_processor_type()
    replaced = 0
    for block in decoder.transformer_blocks:
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Fp8Linear(child))
                    replaced += 1
        if processor_type is not None:
            _tile_block(block, processor_type)
    if _FUSED:
        decoder.rope = PaddedRope(decoder.rope, head_dim=_HEAD_DIM)
    logger.info(
        f"MiniMax-H3 VAE decoder: {replaced} Linears converted to FP8 per-tensor"
        + (", tiled SwiGLU / attention-output quantize" if _TILED else "")
        + (f", fused norm path (tiles mlp {_MLP_TILE} q {_Q_TILE})" if _FUSED else "")
    )
    return replaced
