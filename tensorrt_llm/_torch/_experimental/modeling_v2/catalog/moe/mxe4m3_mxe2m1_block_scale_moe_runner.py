# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP4-weight / MXFP8-activation mixture-of-experts layer: routing (or given
top-k) + grouped FC1 GEMM + clamped gated activation + MXFP8 requantization +
grouped FC2 GEMM + routing-weighted combine, in one trtllm-gen call."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: routing_method_type values for which the kernel never reads routing_bias:
#: 0 Default, 1 Renormalize, 4 RenormalizeNaive, 6 SigmoidRenorm. Observed on
#: this machine: a +-1e3 bias leaves the result bitwise unchanged.
_ROUTING_BIAS_IGNORED = (0, 1, 4, 6)

#: routing_method_type the gpt-oss target passes, and `act_type` for SwiGlu.
ROUTING_METHOD_RENORMALIZE = 1
ACT_TYPE_SWIGLU = 0

#: MX block size: one e8m0 scale per 32 elements along K.
SF_VEC_SIZE = 32

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The 16 e2m1 code points in code order.
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

#: One expert stack's `(codes, e8m0 exponents, bias)`.
Operand = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

#: The FC1 epilogue's block-scale recipe is architecture-specific, and the
#: difference is bit-exact rather than a tolerance: trtllm-gen ships one cubin
#: per architecture. Measured on each with an identity down-projection reading
#: the intermediate out element by element.
#:
#:   sm_100   e8m0 = floor(log2(amax)) - 8      "OCP scale"
#:   sm_103   e8m0 = ceil(log2(amax / 448))     "round-up scale"
#:
#: The round-up form is the one ``torch.ops.trtllm.mxfp8_quantize`` has always
#: used, so sm_103 makes the MoE epilogue and the standalone quantizer agree.
#: Both are named so that a future cubin switching back cannot pass silently.
OCP_SCALE, ROUND_UP_SCALE = "ocp", "round_up"
SCALE_RECIPE_BY_SM = {(10, 0): OCP_SCALE, (10, 3): ROUND_UP_SCALE}


def scale_recipe() -> str:
    """The block-scale recipe this device's FC1 epilogue uses.

    Refuses an architecture nobody measured rather than guessing one of the
    two: the difference is bit-exact, so a wrong guess is a reference that
    disagrees with a correct kernel by ~10 ulp RMS and reads as a kernel bug.
    """
    sm = torch.cuda.get_device_capability()
    recipe = SCALE_RECIPE_BY_SM.get(sm)
    assert recipe is not None, (
        f"the FC1 epilogue's requantization recipe is not certified on sm_{sm[0]}{sm[1]}; "
        "run the identity-down-projection probe and record it before trusting this entry"
    )
    return recipe


def _block_scale(amax: torch.Tensor, recipe: str) -> torch.Tensor:
    """The per-32-column e8m0 scale, under the named recipe."""
    if recipe == OCP_SCALE:
        exp = torch.floor(torch.log2(amax)) - 8.0
    else:
        exp = torch.ceil(torch.log2(amax / E4M3_MAX))
    exp = torch.where(amax == 0, torch.full_like(amax, -127.0), exp)
    return torch.exp2(exp.clamp(-127.0, 127.0))


def quantize_mxfp8(act: torch.Tensor, recipe: Optional[str] = None) -> torch.Tensor:
    """The MXFP8 requantization the FC1 epilogue applies to its own output.

    Per 32 consecutive intermediate columns: this architecture's block scale,
    then round to nearest even into e4m3 saturating at +-448. Returns the
    dequantized fp32 value FC2 actually consumes, which is what a reference has
    to model -- skipping it lands ~10 ulp RMS away from a correct kernel.
    """
    rows, cols = act.shape
    blocks = act.reshape(rows, cols // SF_VEC_SIZE, SF_VEC_SIZE)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = _block_scale(amax, recipe or scale_recipe())
    q = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(rows, cols)


def dequantize_mxfp4(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """One expert's `[N, K]` fp32 weight from its e2m1 codes and e8m0 exponents."""
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    scale = torch.exp2(exponents.float() - 127.0).repeat_interleave(SF_VEC_SIZE, dim=1)
    return table[codes.long()] * scale


class _Mxe4m3Mxe2m1BlockScaleMoeRunner(OpWrapper):
    """One MXFP4-weight MoE layer over MXFP8 (e4m3 + UE8M0) activations, with
    gpt-oss's clamped gated activation.

    Unlike its NVFP4 sibling this op will route for you: given `routing_logits`
    it computes top-k itself, and given `topk_ids` / `topk_weights` it takes the
    caller's. The gpt-oss target passes logits and lets the kernel route.

    `valid_hidden_size` and `valid_intermediate_size` are the *unpadded* widths.
    Both GEMMs want their rows aligned -- 128 on each, and 512 more on FC1's K
    axis -- so the operands are padded and these two say how much of the result
    is real. The returned rows are `valid_hidden_size` wide.

    Returns a fresh `[num_tokens, valid_hidden_size]` bf16 tensor, or an empty
    `[0]` tensor when `output` was given.
    """

    ARCHS = frozenset({Arch.SM_103})

    # gpt-oss-120b's expert geometry: 128 experts, H = I = 2880, top-4, and the
    # clamped SwiGlu constants the checkpoint carries (alpha 1.702, beta 1.0,
    # clamp limit 7.0). The target runs tp1, so every expert is local.
    _GPT_OSS = dict(hidden=2880, intermediate=2880, num_experts=128, top_k=4)

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the gpt-oss target's own call: router entry point, all 128 experts local",
            spec=dict(
                tokens=64,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **_GPT_OSS,
            ),
        ),
        Cell(
            why="one token -- the decode row count, where the routing has a single row to place",
            spec=dict(
                tokens=1,
                routed_by_kernel=True,
                local_expert_offset=0,
                local_num_experts=128,
                **_GPT_OSS,
            ),
        ),
        Cell(
            why="pre-routed with an expert window: the caller owns top-k and this rank owns 32 ids",
            spec=dict(
                tokens=64,
                routed_by_kernel=False,
                local_expert_offset=32,
                local_num_experts=32,
                **_GPT_OSS,
            ),
        ),
    )

    note = """
    `act_type` 0 (SwiGlu) only, with the clamp: `gemm1_alpha`, `gemm1_beta` and
    `gemm1_clamp_limit` are per-local-expert fp32 vectors, and gpt-oss sets them
    to 1.702 / 1.0 / 7.0. Passing `None` for all three is the unclamped form,
    which this entry does not certify against the target.

    The FC1 epilogue's requantization recipe is per architecture and the
    difference is bit-exact, not a tolerance -- see `scale_recipe`. That is the
    reason `ARCHS` is load-bearing on this entry rather than bookkeeping.

    `tune_max_num_tokens` is an autotuner bucket hint and was measured inert; it
    is not a token count.

    The caller owns the weight layout: both GEMMs' weights are pre-shuffled and
    their scales 128x4 swizzled, padded up to the alignments above. Nothing here
    checks it, and the entry's test builds the layout twice -- once in torch,
    once through the trtllm preprocessing ops -- and compares them first.
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
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        valid_hidden_size: Optional[int],
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        act_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> torch.Tensor:
        """Run the layer. Returns a fresh `[num_tokens, valid_hidden_size]`
        bf16 tensor, or an empty `[0]` tensor when `output` was given."""
        return torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
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
            num_experts,
            top_k,
            n_group,
            topk_group,
            intermediate_size,
            valid_hidden_size,
            valid_intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
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
        alpha: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        clamp_limit: Optional[torch.Tensor] = None,
        local_expert_offset: int = 0,
        local_num_experts: Optional[int] = None,
    ) -> torch.Tensor:
        """The layer, in plain torch over dequantized weights, fp32 throughout.

        The documented exception to mirroring: `__call__` takes the *prepared*
        operands -- packed e2m1 codes, shuffled and padded rows, swizzled
        scales -- and a reference given those would be re-deriving the layout
        rather than the computation. So this takes the per-expert
        `(codes, e8m0 exponents, bias)` triples the layout was built from, at
        their valid (unpadded) widths, plus the routing the call resolved to.

        `topk_ids` carries *global* ids; this rank contributes only for the ids
        its window owns and zero elsewhere, which is what makes an
        expert-parallel split summable.

        The activation is gpt-oss's clamped GLU:
        `(up + beta) * gate * sigmoid(alpha * gate)`, with `gate` clamped above
        at `clamp_limit` and `up` clamped to both sides -- an asymmetry, not a
        typo. Its output is requantized to MXFP8 before FC2 because the kernel
        does; skipping that is ~10 ulp RMS away from a correct kernel.
        """
        num_tokens, hidden = hidden_states.shape
        up_c, up_s, up_b = up
        gt_c, gt_s, gt_b = gate
        dn_c, dn_s, dn_b = down
        if local_num_experts is None:
            local_num_experts = up_c.shape[0]
        xf = hidden_states.float()
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        for local_e in range(local_num_experts):
            tok, slot = (topk_ids == local_expert_offset + local_e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = xf[tok]
            u = xe @ dequantize_mxfp4(up_c[local_e], up_s[local_e]).t() + up_b[local_e]
            g = xe @ dequantize_mxfp4(gt_c[local_e], gt_s[local_e]).t() + gt_b[local_e]
            if clamp_limit is not None:
                lim = float(clamp_limit[local_e])
                g = g.clamp(max=lim)
                u = u.clamp(-lim, lim)
            a = 1.0 if alpha is None else float(alpha[local_e])
            b = 0.0 if beta is None else float(beta[local_e])
            act = quantize_mxfp8((u + b) * g * torch.sigmoid(a * g))
            y = act @ dequantize_mxfp4(dn_c[local_e], dn_s[local_e]).t() + dn_b[local_e]
            out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
        return out

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """8 ulp per element against the row's own scale, 4 ulp relative RMS.

        Kernel and reference consume bit-identical mxfp4 weights and MXFP8
        activations and model the same FC1-output requantization, so they differ
        only in accumulation order and in whether a marginal FC1 value rounds to
        the same e4m3 code. Torch's default band cannot express that: its bf16
        `atol` of 1e-5 sits three orders of magnitude below one output ulp of a
        two-GEMM chain, and per-element `rtol` is meaningless where cancellation
        drives `|ref|` to zero.

        Both numbers are measured. Worst over every configuration the entry's
        test covers: 2.0 ulp element-wise (128 experts, 8192 tokens, H=I=2880)
        and 0.87 ulp RMS -- so this gate sits at 4x and 4.6x the observed
        maxima. Both were sized against deliberately wrong computations: a
        reference skipping the FC1 requantization lands at 19.6 / 10.6 ulp, and
        a gate/up swap or an unshuffled operand past 100.
        """
        assert out.dtype == ref.dtype == torch.bfloat16, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        ulp = 2.0**-8
        row = ref.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-9)
        torch.testing.assert_close(out.float() / row, ref.float() / row, rtol=0.0, atol=8 * ulp)
        rel_rms = (
            (out.float() - ref.float()).pow(2).mean().sqrt()
            / ref.float().pow(2).mean().sqrt().clamp_min(1e-9)
        ).item() / ulp
        assert rel_rms <= 4.0, f"relative RMS {rel_rms:.2f} ulp > 4 ulp"

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
        num_experts: int,
        top_k: int,
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        valid_hidden_size: Optional[int],
        valid_intermediate_size: Optional[int],
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        act_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> None:
        """The two ways this op takes an input and answers it wrongly.

        Everything else a caller can get wrong here the op rejects loudly -- a
        mismatched `intermediate_size`, a `top_k` outside its range, a weight
        shape that disagrees with the declared valid widths. Restating those
        would only swap their errors for worse ones.
        """
        # Pure-metadata guard: the kernel takes raw data pointers and assumes a
        # dense row-major layout for every tensor. A strided view is accepted
        # without complaint and silently reads the wrong elements (observed on
        # this machine for every tensor argument listed here, hidden_states_scale
        # included).
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
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
            ("output", output),
        ):
            if tensor is not None:
                assert tensor.is_contiguous(), (
                    f"{name} must be contiguous; a strided view is read as if dense "
                    "and silently produces wrong results"
                )
        # Pure-metadata guard: routing_bias is a no-op for every non-grouped
        # routing method, and for any routing method once topk_ids/topk_weights
        # carry the routing. A caller expecting the bias to shift expert selection
        # gets a silently different model.
        if routing_bias is not None:
            assert topk_ids is None, (
                "routing_bias is ignored when topk_ids/topk_weights are given "
                "(routing has already happened); fold it into the router logits"
            )
            assert routing_method_type not in _ROUTING_BIAS_IGNORED, (
                f"routing_bias is silently ignored for routing_method_type="
                f"{routing_method_type}; add it to routing_logits before the call"
            )


mxe4m3_mxe2m1_block_scale_moe_runner = _Mxe4m3Mxe2m1BlockScaleMoeRunner()
