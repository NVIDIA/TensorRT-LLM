# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2's NVFP4 routed experts on FlashInfer's CuTe DSL fused MoE (W4A4, SiTU).

One rank's experts `[offset, offset + E)` over every token of the step; the output is this
rank's partial sum, which an all-reduce over the EP group completes. Per token and chosen expert
`e` with weight `w`:

    x^   = nvfp4(x)                    per token, 4/6 block scales (E4M3 max 256)
    h    = situ(x^ @ gate_e^T, x^ @ up_e^T)
         = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)
    out += w * nvfp4(h) @ down_e^T

Weights are the checkpoint's NVFP4 tensors: E2M1 pairs (low nibble first), one E4M3 block scale
per 16 values along K, and a per-expert global scale `alpha = amax / outer_scale_denominator`
applied after the block-scaled product. In the layout the kernel reads:

* `w13` `[E, 2I, H / 2]`: rows in FlashInfer's gated FC1 order (`fc1_row_order`): in each group
  of 128 rows, up rows `64g .. 64g + 63` then gate rows `64g ..`. The checkpoint holds gate `j`
  at row `2j` and up `j` at row `2j + 1`.
* `w13_sf` `[E, 2I, H / 16]`, `w2_sf` `[E, H, I / 16]`: those rows' block scales, swizzled into
  128x4 atoms (`swizzle_block_scales`), as uint8.
* `w2` `[E, H, I / 2]` as stored.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Per-token activation global scale of the 4/6 recipe with an E4M3 max of 256: 1 / (6 * 256).
ACT_GLOBAL_SCALE = 1.0 / (6.0 * 256.0)
#: FlashInfer's FC1 interleave: groups of 64 up rows, then 64 gate rows.
FC1_GROUP = 64

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

#: Inkling-2-Small at EP4: 96 local experts of 384, hidden 4608, intermediate 2304.
_EP4 = dict(experts=384, local=96, hidden=4608, inter=2304, top_k=8)


def recipe():
    """FlashInfer's 4/6 recipe the checkpoint was quantized for."""
    from flashinfer.quantization.nvfp4_quantization_utils import NVFP44Over6Config

    return NVFP44Over6Config(e4m3_max=256, err_mode="MAE", err_use_fast_math=False)


def fc1_row_order(intermediate: int, device: torch.device) -> torch.Tensor:
    """Checkpoint row of each kernel FC1 row, for one expert's `[2I, ...]` w13 (see module doc)."""
    j = torch.arange(intermediate, device=device)
    halves = torch.stack([2 * j + 1, 2 * j])  # [up; gate]
    return halves.view(2, intermediate // FC1_GROUP, FC1_GROUP).transpose(0, 1).reshape(-1)


def swizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """`[E, M, K / 16]` uint8 linear block scales -> 128x4 atoms, the layout of FlashInfer's
    `block_scale_interleave`: row `m`, column `k` at `(m // 128, k // 4, m % 32, (m % 128) // 32,
    k % 4)`. Needs `M % 128 == 0` and `(K / 16) % 4 == 0`."""
    e, m, ks = scales.shape
    assert m % 128 == 0 and ks % 4 == 0, (m, ks)
    return (
        scales.reshape(e, m // 128, 4, 32, ks // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(e, m, ks)
    )


def unswizzle_block_scales(scales: torch.Tensor) -> torch.Tensor:
    """Inverse of `swizzle_block_scales`."""
    e, m, ks = scales.shape
    return (
        scales.reshape(e, m // 128, ks // 4, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .contiguous()
        .view(e, m, ks)
    )


def dequantize_nvfp4(values: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """`[..., K / 2]` packed E2M1 and `[..., K / 16]` E4M3 block scales (uint8 bits) -> fp32
    `[..., K]`, without any global scale."""
    lut = torch.tensor(_E2M1, device=values.device)
    v = values.view(torch.uint8)
    pairs = torch.stack([lut[(v & 0xF).long()], lut[(v >> 4).long()]], dim=-1).flatten(-2)
    scales = block_scales.view(torch.uint8).view(torch.float8_e4m3fn).float()
    return (pairs.unflatten(-1, (-1, 16)) * scales.unsqueeze(-1)).flatten(-2)


def quantize_activations(x: torch.Tensor):
    """Per-token 4/6 NVFP4 activations: (packed values, linear E4M3 scales as uint8, per-token
    decode scales), with `x ~= e2m1 * block_scale * per_token`."""
    from flashinfer import SfLayout, nvfp4_quantize

    return nvfp4_quantize(
        x,
        ACT_GLOBAL_SCALE,
        sfLayout=SfLayout.layout_linear,
        per_token_activation=True,
        backend="cute-dsl",
        nvfp4_4over6=recipe(),
    )


def fc2_input_scale(device: torch.device) -> torch.Tensor:
    """The GEMM2-input global scale the kernel takes, `[1]` fp32 `1 / (6 * 256)`.

    FlashInfer treats it as a weight-pack constant: it reads it to the host once per tensor and
    remembers it by storage and version, so a caller builds it once, outside inference mode,
    and passes the same tensor on every call.
    """
    with torch.inference_mode(False):
        return torch.full((1,), ACT_GLOBAL_SCALE, device=device, dtype=torch.float32)


def _situ(
    gate: torch.Tensor, up: torch.Tensor, gate_softcap: float, up_softcap: float
) -> torch.Tensor:
    return (
        gate_softcap
        * torch.tanh(gate / gate_softcap)
        * torch.sigmoid(gate)
        * up_softcap
        * torch.tanh(up / up_softcap)
    )


class Inkling2Nvfp4Experts(OpWrapper):
    """This rank's NVFP4 experts: `x` `[T, H]` bf16, `topk_ids` `[T, k]` int32 (global ids),
    `topk_weights` `[T, k]` fp32 -> `[T, H]` bf16, the sum over the chosen experts this rank holds.
    Weight layouts as in the module docstring; `w13_alpha`, `w2_alpha` `[E]` fp32; `fc2_scale` the
    persistent GEMM2-input scale from `fc2_input_scale`."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="EP4 rank 1, one decode token", spec=dict(_EP4, tokens=1, rank=1)),
        Cell(why="EP4 rank 0, 64 tokens", spec=dict(_EP4, tokens=64, rank=0)),
        Cell(why="EP4 rank 3, 300 tokens", spec=dict(_EP4, tokens=300, rank=3)),
    )

    note = """
    The SiTU caps are the model's mlp_gate_softcap (4) and mlp_up_softcap (25); the sigmoid
    sees the uncapped gate. FlashInfer requantizes the SiTU output per token with the same 4/6
    recipe before the down projection, from fp32; the reference requantizes the bf16-rounded
    intermediate with the same quantizer, which moves a few block choices. The kernel adds the
    experts' outputs with atomics, so repeated calls differ in the last bf16 bit. Swapping gate
    and up, or the 64-row interleave halves, gives a finite, plausibly scaled and wrong output.
    """

    def __call__(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w13_sf: torch.Tensor,
        w13_alpha: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        w2_alpha: torch.Tensor,
        fc2_scale: torch.Tensor,
        num_experts: int,
        local_expert_offset: int,
        gate_softcap: float,
        up_softcap: float,
    ) -> torch.Tensor:
        from flashinfer import cute_dsl_fused_moe
        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

        local, rows, packed_k = w13.shape
        hidden, inter = packed_k * 2, rows // 2
        values, scales, per_token = quantize_activations(x)
        return cute_dsl_fused_moe(
            values,
            scales,
            topk_ids,
            topk_weights,
            w13,
            convert_sf_to_mma_layout(w13_sf, m=rows, k=hidden, num_groups=local, sf_vec_size=16),
            w13_alpha,
            fc2_scale,
            w2,
            convert_sf_to_mma_layout(w2_sf, m=hidden, k=inter, num_groups=local, sf_vec_size=16),
            w2_alpha,
            num_experts,
            topk_ids.shape[1],
            num_local_experts=local,
            local_expert_offset=local_expert_offset,
            situ_beta=gate_softcap,
            situ_linear_beta=up_softcap,
            quant_mode="w4a4",
            per_token_scale=per_token,
            nvfp4_4over6=recipe(),
        )

    def reference(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w13_sf: torch.Tensor,
        w13_alpha: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        w2_alpha: torch.Tensor,
        fc2_scale: torch.Tensor,
        num_experts: int,
        local_expert_offset: int,
        gate_softcap: float,
        up_softcap: float,
    ) -> torch.Tensor:
        """Dense fp32 per expert from the dequantized weights; the activations and the SiTU output
        (rounded to bf16) go through the same per-token 4/6 quantizer. `num_experts` and `fc2_scale`
        (a constant) are unused."""
        local, rows, _ = w13.shape
        inter = rows // 2
        values, scales, per_token = quantize_activations(x)
        x_hat = dequantize_nvfp4(values, scales) * per_token.float().view(-1, 1)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in range(local):
            tokens, slots = torch.where(topk_ids == local_expert_offset + e)
            if tokens.numel() == 0:
                continue
            fc1 = (
                dequantize_nvfp4(w13[e], unswizzle_block_scales(w13_sf[e : e + 1])[0])
                * w13_alpha[e]
            )
            groups = fc1.view(inter // FC1_GROUP, 2, FC1_GROUP, -1)
            up, gate = groups[:, 0].reshape(inter, -1), groups[:, 1].reshape(inter, -1)
            down = (
                dequantize_nvfp4(w2[e], unswizzle_block_scales(w2_sf[e : e + 1])[0]) * w2_alpha[e]
            )
            with true_fp32_matmul():
                h = _situ(
                    x_hat[tokens] @ gate.t(), x_hat[tokens] @ up.t(), gate_softcap, up_softcap
                )
                h_values, h_scales, h_per_token = quantize_activations(h.to(torch.bfloat16))
                h_hat = dequantize_nvfp4(h_values, h_scales) * h_per_token.float().view(-1, 1)
                out.index_add_(
                    0, tokens, (h_hat @ down.t()) * topk_weights[tokens, slots].float().unsqueeze(1)
                )
        return out.to(x.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w13_sf: torch.Tensor,
        w13_alpha: torch.Tensor,
        w2: torch.Tensor,
        w2_sf: torch.Tensor,
        w2_alpha: torch.Tensor,
        fc2_scale: torch.Tensor,
        num_experts: int,
        local_expert_offset: int,
        gate_softcap: float,
        up_softcap: float,
    ) -> None:
        local, rows, packed_k = w13.shape
        assert x.dtype == torch.bfloat16 and x.shape[1] == 2 * packed_k, "x must be bf16 [T, H]"
        assert w13.dtype == w2.dtype == torch.uint8, "packed NVFP4 weights are uint8"
        assert w13_sf.shape == (local, rows, 2 * packed_k // 16), "w13_sf must be [E, 2I, H / 16]"
        assert w2.shape == (local, 2 * packed_k, rows // 4), "w2 must be [E, H, I / 2]"
        assert w2_sf.shape == (local, 2 * packed_k, rows // 32), "w2_sf must be [E, H, I / 16]"
        assert w13_alpha.dtype == w2_alpha.dtype == torch.float32 and w13_alpha.shape == (local,)
        assert topk_ids.dtype == torch.int32 and topk_weights.dtype == torch.float32
        assert 0 <= local_expert_offset and local_expert_offset + local <= num_experts
        assert fc2_scale.shape == (1,) and abs(float(fc2_scale[0]) / ACT_GLOBAL_SCALE - 1) < 1e-6, (
            "fc2_scale is 1 / (6 * 256)"
        )


inkling2_nvfp4_experts = Inkling2Nvfp4Experts()
