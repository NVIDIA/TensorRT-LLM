# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4-weight / NVFP4-activation mixture-of-experts layer: caller-supplied
top-k + grouped FC1 GEMM + gated (SwiGlu) or element-wise (Relu2) activation +
NVFP4 requantization + grouped FC2 GEMM + routing-weighted combine, in one
trtllm-gen call."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: NVFP4 block size: one e4m3 scale per 16 elements along K.
SF_VEC_SIZE = 16

#: Largest e2m1 magnitude, and the largest finite e4m3 magnitude. Their product
#: is the global scale a checkpoint's `input_scale` encodes.
E2M1_MAX = 6.0
E4M3_MAX = 448.0

#: `act_type` for SwiGlu: FC1 is the `[up; gate]` pair, 2 * I rows.
ACT_TYPE_SWIGLU = 0

#: `act_type` for Relu2, `relu(x) ** 2`: element-wise, so FC1 is the up
#: projection alone, I rows.
ACT_TYPE_RELU2 = 1

#: The trtllm-gen activations whose FC1 is a gated pair: SwiGlu (0) and SiTu
#: (3). Relu2 (1) and Silu (2) are element-wise. SiTu and Silu are not
#: certified here; they are listed so the FC1 row-count guard is right for them.
_GATED_ACT_TYPES = frozenset({0, 3})

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

#: One expert stack's NVFP4 projection as a checkpoint stores it, unshuffled
#: and unswizzled: `(codes [E, M, K] uint8, block scales [E, M, K/16] e4m3)`,
#: optionally followed by `weight_scale_2 [E]` fp32 -- each expert's global
#: dequantization scale, 1 when absent.
Operand = Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


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


def _project(x: torch.Tensor, operand: Operand, expert: int) -> torch.Tensor:
    """`x @ W_e^T` for one expert of an `Operand`, fp32.

    A per-expert `weight_scale_2` multiplies the product rather than the
    weight, which is where the kernel applies it too: folded into the
    accumulator's dequantization scalar.
    """
    y = x @ dequantize_nvfp4(operand[0][expert], operand[1][expert]).t()
    return y * operand[2][expert].float() if len(operand) > 2 else y


class Fp4BlockScaleMoeRunner(OpWrapper):
    """One NVFP4-weight MoE layer over NVFP4 (e2m1 + e4m3 block-scale)
    activations, entered pre-routed.

    The caller owns top-k: `topk_ids` carries global expert ids and
    `topk_weights` the combine weights, neither renormalized here. This rank
    answers for `[local_expert_offset, local_expert_offset + local_num_experts)`
    and contributes zero for every other id, so an expert-parallel split is N
    calls whose outputs sum -- which is what the deepseek dep4 target does;
    qwen4_exp makes one call over its whole expert stack.

    With `do_finalize=True` the result is a one-element list holding the
    combined `[T, H]` bf16 rows, or an empty tensor when `output` was given.
    With `do_finalize=False` it is three tensors: per-slot expert rows, an
    unwritten scale buffer, and the expanded-index -> permuted-row map.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    # DeepSeek-R1-0528's routed geometry. The dep4 target gathers four ranks'
    # tokens before the expert call and issues one call per 64-wide window, so
    # the window cells are the shape it runs and the full stack is the
    # single-rank one.
    _R1 = dict(hidden=7168, intermediate=2048, num_experts=256, top_k=8)

    # qwen4_exp's (Qwen3.8-Flash-Next) routed geometry: many small experts, a
    # top_k past 8, an intermediate width of 640, all 512 experts local. Its
    # router renormalizes the top-10 softmax weights to sum to 1 before the
    # call (`routing="renormalized"`, a test axis rather than an op argument;
    # the R1 cells draw uniform [0, 1) weights). Nothing is padded:
    # H % 256, 2I % 128, H/16 % 4 and I/16 % 4 are all zero.
    _QWEN4_EXP = dict(
        hidden=2560,
        intermediate=640,
        num_experts=512,
        top_k=10,
        local_expert_offset=0,
        local_num_experts=512,
        routing="renormalized",
    )

    # Nemotron-3 Ultra's routed experts, in its 2048-wide latent space, as its
    # tep4 target issues them: every rank routes the same tokens over all 512
    # experts and answers for its own 128-wide window, so the windows' outputs
    # sum across ranks. FC1 is the up projection alone (Relu2 is element-wise:
    # 5120 rows, no gate half). The slot weights are the router's sigmoid
    # scores of the top 22, renormalized and scaled to sum to 5
    # (`routing="sigmoid_x5"`). Every expert carries its own `weight_scale_2`
    # while both input scales are shared by the layer, so the scalars differ
    # per expert -- the test draws them that way at every Relu2 cell. Nothing
    # is padded: H % 256, I % 128, H/16 % 4 and I/16 % 4 are all zero.
    _ULTRA = dict(
        hidden=2048,
        intermediate=5120,
        num_experts=512,
        top_k=22,
        local_num_experts=128,
        act_type=ACT_TYPE_RELU2,
        routing="sigmoid_x5",
    )

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
        Cell(
            why="qwen4_exp decode: one token over 512 experts, top-10 renormalized, H 2560 / I 640",
            spec=dict(tokens=1, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp at 4 rows: the last count the pre-routed path's static-block "
            "permutation kernel takes",
            spec=dict(tokens=4, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp at 5 rows: the first count on the pre-routed path's "
            "dynamic-block permutation kernel (5 to 12 rows)",
            spec=dict(tokens=5, **_QWEN4_EXP),
        ),
        Cell(
            why="qwen4_exp prefill: 8192 rows, the single-cluster kernel's cap and the top "
            "of the certified column",
            spec=dict(tokens=8192, **_QWEN4_EXP),
        ),
        Cell(
            why="Ultra tep4 decode, 1 row, rank 0: relu2 FC1 of 5120 rows, top-22 of 512, "
            "a 128-wide window",
            spec=dict(tokens=1, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra decode, 1 row, rank 3: the last window, ids 384-511 -- the top of "
            "the expert range",
            spec=dict(tokens=1, local_expert_offset=384, **_ULTRA),
        ),
        Cell(
            why="Ultra at 8 rows, rank 1: the pre-routed path's dynamic-block permutation "
            "kernel (5 to 12 rows)",
            spec=dict(tokens=8, local_expert_offset=128, **_ULTRA),
        ),
        Cell(
            why="Ultra decode ceiling, 32 rows, rank 0: the single-cluster permutation kernel",
            spec=dict(tokens=32, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra decode ceiling, 32 rows, rank 3",
            spec=dict(tokens=32, local_expert_offset=384, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill chunk, 300 rows, rank 0: a mid-width FC1 token tile",
            spec=dict(tokens=300, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill chunk, 300 rows, rank 3",
            spec=dict(tokens=300, local_expert_offset=384, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill at 8192 rows, rank 0: the widest FC1 token tile and the top "
            "of the certified column",
            spec=dict(tokens=8192, local_expert_offset=0, **_ULTRA),
        ),
        Cell(
            why="Ultra prefill at 8192 rows, rank 3",
            spec=dict(tokens=8192, local_expert_offset=384, **_ULTRA),
        ),
    )

    note = """
    Certified pre-routed only: `routing_logits` is `None` at every cell, so the
    kernel's own routing is not exercised and `routing_method_type`, `n_group`,
    `topk_group` and `routed_scaling_factor` do not change the result -- they
    are passed because the signature takes them. `routing_method_type` 0, 2,
    4, 5, 6 were measured bitwise identical to 1 at 512 experts / top-10; at
    512 experts / top-22, 0, 1, 4 and 6, and 2 with `n_group`, `topk_group`
    and `routed_scaling_factor` either numbers or `None`, are all bitwise one
    result.

    They are read in one place: inside the autotuner's `autotune()` context the
    op builds a dummy routing from `routing_method_type` to profile with, and
    DeepSeekV3's (2) is constructed in Python from `n_group`, `topk_group` and
    `routed_scaling_factor` -- `None` for any of them raises `TypeError` there
    and only there, so a cold-cache test never sees it. A caller passing 2
    passes all three as numbers (a single group is `n_group = topk_group = 1`);
    1 (Renormalize) needs none of them.

    Which `(num_experts, top_k)` pairs the routing kernels are compiled for is
    a per-policy tier list, and a pair with no tier is a logged error, not a
    raise. The pre-routed path substitutes its own preprocess/postprocess pair
    (whose tiers include 512/16 and 512/22) before dispatch, and picks its
    permutation kernel by row count: a static-block kernel up to 4 rows, a
    dynamic-block one from 5 to 12 rows while the expert tier is at most 1024,
    a single-cluster one up to 8192 rows.

    `act_type` 0 (SwiGlu) and 1 (Relu2). Gated or not is read off FC1's row
    count (2 * I or I) while the kernel that runs is chosen by `act_type`; the
    two must agree, which `is_valid` checks. The FC1 epilogue, read out of the
    kernel bit for bit, is
        SwiGlu: C = output1_scale_scalar * up_acc * silu(output1_scale_gate_scalar * gate_acc)
        Relu2:  C = output1_scale_scalar * relu(output1_scale_gate_scalar * up_acc) ** 2
    followed by one NVFP4 quantization of C (e4m3 block scale
    `amax16(C) / 6`, saturating at 448). So `output1_scale_gate_scalar` is the
    FC1 dequantization and, for Relu2, `output1_scale_scalar` is the FC2-input
    quantization scale *alone* -- for SwiGlu it also carries the dequantization.
    From a ModelOpt NVFP4 checkpoint (per expert e; `input_scale` shared by the
    layer, `weight_scale_2` per expert):
        activation global scale (fp4_quantize)  = 1 / up.input_scale
        output1_scale_gate_scalar[e]            = up.input_scale * up.weight_scale_2[e]
        output1_scale_scalar[e]  (Relu2)        = 1 / down.input_scale
        output1_scale_scalar[e]  (SwiGlu)       = output1_scale_gate_scalar[e] / down.input_scale
        output2_scale_scalar[e]                 = down.input_scale * down.weight_scale_2[e]
    The SwiGlu form handed to a Relu2 call is accepted and wrong by the factor
    `output1_scale_gate_scalar`, which at checkpoint magnitudes underflows the
    intermediate's block scales: measured on a real checkpoint layer's scales,
    the output is exactly zero.

    `gemm1_bias`, `gemm1_alpha`, `gemm1_beta`, `gemm1_clamp_limit` and
    `gemm2_bias` are `None`. `hidden_states_scale` is the linear (unswizzled)
    e4m3 block-scale layout, one scale per 16 elements of each row.

    At 512 experts the SwiGlu cells hold the whole stack and the Relu2 cells
    128-wide windows; `do_finalize=False` and a caller-supplied `output` were
    measured at neither.

    `tune_max_num_tokens` is an autotuner bucket hint and was measured inert;
    it is not a token count. `T` is `hidden_states.shape[0]`, and 8192 is the
    top of the certified column. A warm autotuner cache was measured bitwise
    inert against a cold one at the R1 stack, its windows, the 512-expert
    stack and the Relu2 128-wide windows; `local_num_experts` separates the
    tuner's cache entries, so each layout warms its own.

    A Relu2 window call captures into a CUDA graph and replays bitwise equal
    to the eager call (decode ceiling, 32 rows); the op allocates its own
    intermediates per call, which capture takes from the graph's pool.

    The caller owns the weight layout entirely. Each expert's FC1 and FC2 rows
    are permuted in 32-row blocks (source row `4u + v` of a block lands at
    `8v + u`) and their e4m3 scales take the same row permutation and then the
    128x4 swizzle; SwiGlu's FC1 additionally interleaves its `[up; gate]`
    halves row by row before the block permutation, and Relu2's has nothing to
    interleave. Nothing here checks any of it, and getting it wrong is 300-3400
    bf16 ulp with no error -- the entry's test builds the layout in torch,
    compares it with trtllm's own preprocessing before either reaches the op,
    and shows each omission landing far outside its gate.
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
        gate: Optional[Operand],
        down: Operand,
        fc2_global_scale: float,
        local_expert_offset: int = 0,
        local_num_experts: Optional[int] = None,
        act_type: int = ACT_TYPE_SWIGLU,
    ) -> torch.Tensor:
        """The layer, in plain torch over dequantized weights, fp32 throughout.

        The documented exception to mirroring: `__call__` takes the *prepared*
        operands -- packed e2m1 codes, shuffled rows, swizzled scales -- and a
        reference given those would be re-deriving the layout rather than the
        computation. So this takes the per-expert operands the layout was built
        from, which is where a target's checkpoint starts too, plus the routing
        this call was entered with. `hidden_states` is the activation's exact
        value (codes times block scales over the global scale); `act_type`
        keeps `__call__`'s name.

        `up`, `gate` and `down` are each an `Operand`: `(codes[E, M, K],
        scales[E, M, K/16])` in their unshuffled, unswizzled form, optionally
        with each expert's `weight_scale_2` as a third element. `gate` is
        `None` for Relu2, whose FC1 is the up projection alone. `topk_ids`
        carries *global* ids; this rank contributes only for the ids its window
        owns, and every other slot contributes zero -- which is what makes an
        expert-parallel split summable.

        The FC1 activation is requantized to NVFP4 under `fc2_global_scale`
        before FC2, because the kernel does: the FC1 epilogue emits `g2 * act`
        as NVFP4 and FC2's alpha divides it back out. Skipping that step is a
        reference that disagrees with a correct kernel by ~23 ulp RMS, so it is
        not an optional refinement. That holds for Relu2 as for SwiGlu: the
        relu2 output is what is requantized, under the same global scale.
        """
        if act_type == ACT_TYPE_SWIGLU:
            assert gate is not None, "SwiGlu is gated: the gate operand is required"
        elif act_type == ACT_TYPE_RELU2:
            assert gate is None, "Relu2 is element-wise: FC1 is the up projection alone"
        else:
            raise NotImplementedError(f"act_type {act_type} is not certified by this entry")
        num_tokens, hidden = hidden_states.shape
        local_num_experts = up[0].shape[0] if local_num_experts is None else local_num_experts
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        for local_e in range(local_num_experts):
            tok, slot = (topk_ids == local_expert_offset + local_e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = hidden_states[tok]
            u = _project(xe, up, local_e)
            if act_type == ACT_TYPE_RELU2:
                r = torch.relu(u)
                act = r * r
            else:
                g = _project(xe, gate, local_e)
                act = u * g * torch.sigmoid(g)
            act = quantize_nvfp4(act, fc2_global_scale) / fc2_global_scale
            y = _project(act, down, local_e)
            out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
        return out

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
        """The three ways this op takes an input and answers it wrongly.

        Everything else a caller can get wrong here the op rejects loudly: an
        `intermediate_size` that is neither FC1's row count nor half of it, a
        `top_k` outside `(0, num_experts)`, a hidden width that disagrees
        between the two weight operands. Those are not restated -- swapping
        their errors for ours would only make them worse.
        """
        # Gated or not is read off FC1's row count -- 2 * I rows is a gated
        # [up; gate] pair, I rows a single projection -- while the kernel that
        # runs is chosen by act_type, and the op checks only that the rows are
        # one or the other. Relu2 handed a 2I-row SwiGlu stack walks it at
        # I-row strides and returns plausible rows from the wrong weights
        # (measured); SwiGlu handed an I-row stack would read 2I rows per
        # expert, past the end of the stack. A count that is neither the op
        # rejects itself.
        rows = gemm1_weights.shape[1]
        expected = (2 if act_type in _GATED_ACT_TYPES else 1) * intermediate_size
        assert rows not in (intermediate_size, 2 * intermediate_size) or rows == expected, (
            f"act_type {act_type} needs FC1 with {expected} rows (intermediate_size "
            f"{intermediate_size}), got {rows}: the op accepts either count and answers wrongly"
        )

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


fp4_block_scale_moe_runner = Fp4BlockScaleMoeRunner()
