# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-weight / NVFP4-activation mixture-of-experts layer: caller-supplied
top-k + grouped FC1 GEMM + gated activation + NVFP4 requantization + grouped
FC2 GEMM + routing-weighted combine, in one trtllm-gen call."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: NVFP4 block size: one e4m3 scale per 16 elements along K.
SF_VEC_SIZE = 16

#: Largest e2m1 magnitude, and the largest finite e4m3 magnitude. Their product
#: is the global scale a checkpoint's `input_scale` encodes.
E2M1_MAX = 6.0
E4M3_MAX = 448.0

#: `act_type` for SwiGlu, the only activation this entry certifies.
ACT_TYPE_SWIGLU = 0

#: The widest unchunked call a stock engine issues: trtllm's default
#: `max_num_tokens`. A caller whose own token count exceeds it must chunk.
MAX_CERTIFIED_TOKENS = 8192

#: The 16 e2m1 code points in code order: sign bit 3, exponent bits 2:1,
#: mantissa bit 0.
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: One expert stack's `(codes, block scales)`, unshuffled and unswizzled.
Operand = Tuple[torch.Tensor, torch.Tensor]


def _e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    """Round to nearest even on the e2m1 grid, saturating at +-6.

    The grid step is 0.5 below 2, 1 below 4 and 2 above -- the binade step of
    e2m1's three exponents, the subnormal binade sharing the first normal one's.
    `torch.round` is banker's rounding, which is the format's ties-to-even-code
    rule: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4,
    5.0 -> 4.
    """
    step = torch.where(a.abs() < 2.0, 0.5, torch.where(a.abs() < 4.0, 1.0, 2.0))
    return torch.sign(a) * torch.clamp(torch.round(a.abs() / step) * step, max=E2M1_MAX)


def quantize_nvfp4(x: torch.Tensor, global_scale: float) -> torch.Tensor:
    """NVFP4 quantize-dequantize of `x` under `global_scale`.

    Per 16 consecutive columns: `sf = e4m3(g * blockmax / 6)` and
    `data = e2m1(x * g / sf)`, returning `data * sf` -- the value a downstream
    block-scaled MMA multiplies, which is `g` times the reconstruction of `x`.
    This models the FC1 epilogue, which the entry's test pins bit-exactly.
    """
    rows, cols = x.shape
    blocks = x.reshape(rows, cols // SF_VEC_SIZE, SF_VEC_SIZE)
    sf = (
        (global_scale * blocks.abs().amax(dim=-1, keepdim=True) / E2M1_MAX)
        .clamp(max=E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .float()
    )
    out_scale = torch.where(sf == 0, torch.zeros_like(sf), global_scale / sf)
    return (_e2m1_rne(blocks * out_scale) * sf).reshape(rows, cols)


def dequantize_nvfp4(codes: torch.Tensor, block_scales: torch.Tensor) -> torch.Tensor:
    """One expert's `[N, K]` fp32 weight from its codes and e4m3 block scales."""
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    return table[codes.long()] * block_scales.float().repeat_interleave(SF_VEC_SIZE, dim=1)


class _Fp4BlockScaleMoeRunner(OpWrapper):
    """One NVFP4-weight MoE layer over NVFP4 (e2m1 + e4m3 block-scale)
    activations, entered pre-routed.

    The caller owns top-k: `topk_ids` carries global expert ids and
    `topk_weights` the combine weights, neither renormalized here. This rank
    answers for `[local_expert_offset, local_expert_offset + local_num_experts)`
    and contributes zero for every other id, so an expert-parallel split is four
    calls whose outputs sum -- which is what the dep4 target does.

    With `do_finalize=True` the result is a one-element list holding the
    combined `[T, H]` bf16 rows, or an empty tensor when `output` was given.
    With `do_finalize=False` it is three tensors: per-slot expert rows, an
    unwritten scale buffer, and the expanded-index -> permuted-row map.
    """

    ARCHS = frozenset({Arch.SM_103})

    # DeepSeek-R1-0528's routed geometry. The dep4 target gathers four ranks'
    # tokens before the expert call and issues one call per 64-wide window, so
    # the window cells are the shape it runs and the full stack is the
    # single-rank one.
    _R1 = dict(hidden=7168, intermediate=2048, num_experts=256, top_k=8)

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4's own call: the 64-wide expert window at offset 0, R1 geometry",
            spec=dict(tokens=64, local_expert_offset=0, local_num_experts=64, **_R1),
        ),
        Cell(
            why="a window at a non-zero offset -- the ids this rank must answer for move with it",
            spec=dict(tokens=64, local_expert_offset=192, local_num_experts=64, **_R1),
        ),
        Cell(
            why="the single-rank shape: all 256 experts local, and one token, the decode row count",
            spec=dict(tokens=1, local_expert_offset=0, local_num_experts=256, **_R1),
        ),
    )

    note = """
    Certified pre-routed only: `routing_logits` is `None` at every cell, so the
    kernel's own routing is not exercised and `routing_method_type`, `n_group`,
    `topk_group` and `routed_scaling_factor` are inert -- they are passed
    because the signature takes them, not because they do anything here.

    `act_type` 0 (SwiGlu) only. `gemm1_bias`, `gemm1_alpha`, `gemm1_beta`,
    `gemm1_clamp_limit` and `gemm2_bias` are `None`.

    `tune_max_num_tokens` is an autotuner bucket hint and was measured inert;
    it is not a token count. `T` is `hidden_states.shape[0]`, and 8192 is the
    top of the certified column.

    The caller owns the weight layout entirely: FC1 and FC2 weights are
    pre-shuffled and their scales additionally 128x4 swizzled. Nothing here
    checks that, and getting it wrong is 300-3400 bf16 ulp with no error -- the
    entry's test builds the layout twice, once in torch and once through the
    trtllm preprocessing ops, and compares them before either reaches the op.
    """

    def __call__(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int = 0,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> list[torch.Tensor]:
        return torch.ops.trtllm.fp4_block_scale_moe_runner(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_bias,
            gemm1_alpha,
            gemm1_beta,
            gemm1_clamp_limit,
            gemm2_weights,
            gemm2_weights_scale,
            gemm2_bias,
            output1_scale_scalar,
            output1_scale_gate_scalar,
            output2_scale_scalar,
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            do_finalize,
            act_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp,
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        up: Operand,
        gate: Operand,
        down: Operand,
        fc2_global_scale: float,
        local_expert_offset: int = 0,
        local_num_experts: Optional[int] = None,
    ) -> torch.Tensor:
        """The layer, in plain torch over dequantized weights, fp32 throughout.

        The documented exception to mirroring: `__call__` takes the *prepared*
        operands -- packed e2m1 codes, shuffled rows, swizzled scales -- and a
        reference given those would be re-deriving the layout rather than the
        computation. So this takes the per-expert `(codes, scales)` pairs the
        layout was built from, which is where a target's checkpoint starts too,
        plus the routing this call was entered with.

        `up`, `gate` and `down` are each `(codes[E, M, K/2], scales[E, M, K/16])`
        in their unshuffled, unswizzled form. `topk_ids` carries *global* ids;
        this rank contributes only for the ids its window owns, and every other
        slot contributes zero -- which is what makes an expert-parallel split
        summable.

        The FC1 activation is requantized to NVFP4 under `fc2_global_scale`
        before FC2, because the kernel does: the FC1 epilogue emits `g2 * act`
        as NVFP4 and FC2's alpha divides it back out. Skipping that step is a
        reference that disagrees with a correct kernel by ~23 ulp RMS, so it is
        not an optional refinement.
        """
        num_tokens, hidden = hidden_states.shape
        up_c, up_s = up
        gt_c, gt_s = gate
        dn_c, dn_s = down
        local_num_experts = up_c.shape[0] if local_num_experts is None else local_num_experts
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        for local_e in range(local_num_experts):
            tok, slot = (topk_ids == local_expert_offset + local_e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = hidden_states[tok]
            u = xe @ dequantize_nvfp4(up_c[local_e], up_s[local_e]).t()
            g = xe @ dequantize_nvfp4(gt_c[local_e], gt_s[local_e]).t()
            act = u * g * torch.sigmoid(g)
            act = quantize_nvfp4(act, fc2_global_scale) / fc2_global_scale
            y = act @ dequantize_nvfp4(dn_c[local_e], dn_s[local_e]).t()
            out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
        return out

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """16 ulp per element against the row's own scale, 2 ulp relative RMS.

        Kernel and reference consume bit-identical NVFP4 weights and activations
        and model the same FC1-output requantization, so they differ only in
        accumulation order and in whether a marginal FC1 value rounds to the
        same e2m1 code. An e2m1 code is a coarse step -- two mantissa bits -- so
        one flipped intermediate moves the output row by a visible fraction, and
        torch's default band cannot express that: its bf16 `atol` of 1e-5 sits
        three orders of magnitude below one output ulp of a two-GEMM chain, and
        per-element `rtol` is meaningless where cancellation drives `|ref|` to
        zero.

        Both numbers are measured, not chosen. Worst over every configuration
        the entry's test covers: 9.97 ulp element-wise and 0.88 ulp RMS; the R1
        geometry lands at 9.62 / 0.74 for a 64-wide window and 4.68 / 0.70 for
        the 256-expert stack. Only the RMS gate is scale-free -- the element
        figure is a max over `T*H` elements, so its tail grows with `T` at
        constant accuracy and reaches 16.33 at the top of the certified token
        column. Size a caller's tolerance off the RMS number.

        Both were sized against deliberately wrong computations: a gate/up swap
        (263-278 ulp), a scalar-role swap (177/48), a missing scale swizzle
        (408/357), a missing row shuffle (508/432), and a reference skipping the
        FC1 requantization (43/28 element, 23/23 RMS). The smallest RMS distance
        any of them reaches is 39.8.
        """
        assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        ulp = 2.0**-8
        row = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
        torch.testing.assert_close(out.float() / row, ref.float() / row, rtol=0.0, atol=16 * ulp)
        rel_rms = (
            (out.float() - ref.float()).pow(2).mean().sqrt()
            / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
        ).item() / ulp
        assert rel_rms <= 2.0, f"relative RMS {rel_rms:.2f} ulp > 2 ulp"

    def is_valid(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_bias: Optional[torch.Tensor],
        gemm1_alpha: Optional[torch.Tensor],
        gemm1_beta: Optional[torch.Tensor],
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        gemm2_bias: Optional[torch.Tensor],
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        do_finalize: bool,
        act_type: int = 0,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> None:
        """The two ways this op takes an input and answers it wrongly.

        Everything else a caller can get wrong here the op rejects loudly: a
        mismatched `intermediate_size`, a `top_k` outside `(0, num_experts)`, a
        hidden width that disagrees between the two weight operands. Those are
        not restated -- swapping their errors for ours would only make them
        worse.
        """
        # The kernel takes raw data pointers and assumes a dense row-major
        # layout for every tensor. A strided view is accepted without complaint
        # and silently reads -- or writes -- the wrong elements. Observed on
        # this machine for hidden_states, topk_weights and the weight operands.
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("routing_bias", routing_bias),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm1_bias", gemm1_bias),
            ("gemm1_alpha", gemm1_alpha),
            ("gemm1_beta", gemm1_beta),
            ("gemm1_clamp_limit", gemm1_clamp_limit),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("gemm2_bias", gemm2_bias),
            ("output1_scale_scalar", output1_scale_scalar),
            ("output1_scale_gate_scalar", output1_scale_gate_scalar),
            ("output2_scale_scalar", output2_scale_scalar),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
            ("output", output),
        ):
            if tensor is not None:
                assert tensor.is_contiguous(), (
                    f"{name} must be contiguous; a strided view is read as if dense "
                    "and silently produces wrong results"
                )

        # With the block-scale layout this entry contracts -- row shuffle plus
        # 128x4 scale swizzle -- a hidden size that is only a multiple of 128
        # makes the kernel read the FC1 weight scales for the wrong blocks.
        # Measured wrong at hidden 384 / 640 / 896: no error, 300-3400 bf16 ulp.
        assert gemm1_weights.shape[-1] * 2 % 256 == 0, (
            "hidden size (gemm1_weights.shape[-1] * 2) must be a multiple of 256; "
            "the prepared block-scale layout is silently misread otherwise"
        )


fp4_block_scale_moe_runner = _Fp4BlockScaleMoeRunner()
