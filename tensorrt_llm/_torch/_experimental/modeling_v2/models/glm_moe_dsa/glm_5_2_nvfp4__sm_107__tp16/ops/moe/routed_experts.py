# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's routed experts at MoE TP16: DeepSeek-V3 routing without groups
(sigmoid + bias, top-8 of 256, scaled sum normalization), grouped NVFP4 FC1 +
SwiGLU + NVFP4 requantization + grouped NVFP4 FC2, in one call of
`trtllm::glm_routed_experts`: the TRTLLM-Gen MoE kernels with GLM's own router
and FC2 kernels beside them.

On the decode step (up to 8 rows) the call runs the router GEMV inside the
routing kernel, lets the NVFP4 input still be in flight on another stream
until FC1 starts, and runs FC2 on the K = 128 kernel; it stops before the
finalize step, which the post-MoE all-reduce hop performs."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (  # noqa: E501
    dequantize_nvfp4,
    quantize_nvfp4,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
    FP4BlockScaleMoEInputs,
    FP4BlockScaleMoERunner,
    prepare_dummy_topk_and_hook,
)

__all__ = [
    "ACT_TYPE_SWIGLU",
    "FUSED_ROUTER_MAX_TOKENS",
    "FUSED_ROUTER_WORKSPACE_WORDS",
    "ROUTING_METHOD_DEEPSEEK_V3",
    "make_fused_router_workspace",
    "routed_experts",
]

#: `routing_method_type` of DeepSeek-V3 routing; with `n_group` 1 the op runs
#: it as sigmoid + bias, top-k, and the sum-normalized, scaled sigmoid weights.
ROUTING_METHOD_DEEPSEEK_V3 = 2

#: `act_type` for SwiGlu.
ACT_TYPE_SWIGLU = 0

#: The largest row count the fused router + routing kernel takes.
FUSED_ROUTER_MAX_TOKENS = 8

#: One expert stack's `(codes [E, N, K] uint8, block scales [E, N, K/16]
#: float8_e4m3fn)`, unshuffled and unswizzled.
Operand = Tuple[torch.Tensor, torch.Tensor]

_IDLE_BITS = -(2**31)  # -0.0f, the idle value of a fused-router exchange buffer

#: Words of a fused-router workspace (`router_workspace`): two exchange
#: buffers of [8 tokens][256 experts][4 slices] fp32, then the epoch word,
#: padded to 16 bytes (kFusedRouterWorkspaceWords, glmMoeFusedRouter.h).
FUSED_ROUTER_WORKSPACE_WORDS = 2 * 8 * 256 * 4 + 4


def make_fused_router_workspace(device: torch.device) -> torch.Tensor:
    """A fused-router workspace in its between-launch state: both exchange
    buffers idle (-0.0, bits 0x80000000) and epoch 0. Only the op writes it,
    and launches that share it must run in stream order."""
    workspace = torch.zeros(FUSED_ROUTER_WORKSPACE_WORDS, dtype=torch.int32, device=device)
    workspace[: FUSED_ROUTER_WORKSPACE_WORDS - 4].fill_(_IDLE_BITS)
    return workspace


class _RoutedExpertsTunableRunner(FP4BlockScaleMoERunner):
    """The autotuner's runner for `trtllm::glm_routed_experts`: the tactics
    (tile N, FC1 / FC2 config pair) and tuning buckets of the TRTLLM-Gen FP4
    MoE runner, run on `torch.classes.trtllm.GlmRoutedExpertsRunner`."""

    runner_dict = dict()

    def __init__(
        self,
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
        act_type: int,
        tune_max_num_tokens: int,
        use_dp: bool,
        prefer_gemm2_k128: bool,
    ):
        super().__init__(
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
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp,
        )
        self.prefer_gemm2_k128 = prefer_gemm2_k128

    def unique_id(self):
        # FC2 on the K = 128 kernel times the tile-8 tactics differently.
        key = super().unique_id()
        return key + ("gemm2_k128",) if self.prefer_gemm2_k128 else key

    def get_runner(self):
        key = (self.act_type,)
        if key not in _RoutedExpertsTunableRunner.runner_dict:
            _RoutedExpertsTunableRunner.runner_dict[key] = (
                torch.classes.trtllm.GlmRoutedExpertsRunner(self.act_type)
            )
        return _RoutedExpertsTunableRunner.runner_dict[key]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Optional[List[int]] = None,
        output: Optional[torch.Tensor] = None,
        hidden_states_ready_event: Optional[int] = None,
        router_hidden_states: Optional[torch.Tensor] = None,
        router_weight: Optional[torch.Tensor] = None,
        router_workspace: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        args = FP4BlockScaleMoEInputs(*inputs)
        return self.get_runner().run_moe(
            args.routing_logits,
            args.routing_bias,
            args.hidden_states,
            args.hidden_states_scale,
            args.gemm1_weights,
            args.gemm1_weights_scale,
            args.gemm1_bias,
            args.gemm1_alpha,
            args.gemm1_beta,
            args.gemm1_clamp_limit,
            args.gemm2_weights,
            args.gemm2_weights_scale,
            args.gemm2_bias,
            args.output1_scale_scalar,
            args.output1_scale_gate_scalar,
            args.output2_scale_scalar,
            self.num_experts,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.intermediate_size,
            self.local_expert_offset,
            self.local_num_experts,
            self.routed_scaling_factor,
            self.routing_method_type,
            self.do_finalize,
            [-1, -1] if tactic is None else tactic,
            args.topk_weights,
            args.topk_ids,
            output,
            hidden_states_ready_event,
            router_hidden_states,
            router_weight,
            router_workspace,
            self.prefer_gemm2_k128,
        )


@torch.library.custom_op("trtllm::glm_routed_experts", mutates_args=("output", "router_workspace"))
def _glm_routed_experts(
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
    act_type: int,
    topk_weights: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    tune_max_num_tokens: int,
    use_dp: bool,
    hidden_states_ready_event: Optional[int],
    router_hidden_states: Optional[torch.Tensor],
    router_weight: Optional[torch.Tensor],
    router_workspace: Optional[torch.Tensor],
    prefer_gemm2_k128: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The op behind `GlmRoutedExperts.__call__` (see there). The autotuner
    profiles it routed from logits, as it profiles the TRTLLM-Gen FP4 MoE op."""
    tuner = AutoTuner.get()
    kernel_runner = _RoutedExpertsTunableRunner(
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
        tune_max_num_tokens,
        use_dp,
        prefer_gemm2_k128,
    )
    routing_logits_for_tuner, topk_weights_for_tuner, topk_ids_for_tuner, tuning_config = (
        prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            hidden_states=hidden_states,
            routing_logits=routing_logits,
            base_tuning_config=kernel_runner.tuning_config,
            top_k=top_k,
            num_experts=num_experts,
            local_num_experts=local_num_experts,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=local_expert_offset,
            use_dp=use_dp,
        )
    )
    input_tensors = [
        routing_logits_for_tuner,
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
        topk_weights_for_tuner,
        topk_ids_for_tuner,
    ]
    kernel_runner, best_tactic = tuner.choose_one(
        "trtllm::glm_routed_experts", [kernel_runner], tuning_config, input_tensors
    )
    # The call itself runs on the caller's routing inputs, not the tuner's.
    input_tensors[0] = routing_logits
    input_tensors[-2] = topk_weights
    input_tensors[-1] = topk_ids
    result = kernel_runner(
        input_tensors,
        tactic=[-1, -1] if best_tactic == -1 else best_tactic,
        output=output,
        hidden_states_ready_event=hidden_states_ready_event,
        router_hidden_states=router_hidden_states,
        router_weight=router_weight,
        router_workspace=router_workspace,
    )
    if not do_finalize:
        return result[0], result[1], result[2]
    # Mutable custom ops need individual Tensor returns for auto-functionalization.
    # Finalized calls pad the internal tuple; the public wrapper drops the padding.
    # A caller-provided output is written in place and cannot alias a return.
    finalized = result[0] if output is None else output.new_empty(0)
    return finalized, finalized.new_empty(0), finalized.new_empty(0)


@_glm_routed_experts.register_fake
def _(
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
    topk_weights,
    topk_ids,
    output,
    tune_max_num_tokens,
    use_dp,
    hidden_states_ready_event,
    router_hidden_states,
    router_weight,
    router_workspace,
    prefer_gemm2_k128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if do_finalize:
        finalized = (
            hidden_states.new_empty(
                (hidden_states.shape[0], hidden_states.shape[1] * 2), dtype=torch.bfloat16
            )
            if output is None
            else output.new_empty(0)
        )
        return finalized, finalized.new_empty(0), finalized.new_empty(0)
    # The autotuned tile determines the padded row count. Only the map's rows
    # are initialized; the packed input's width is half the bf16 output width.
    padded_rows = torch.library.get_ctx().new_dynamic_size()
    return (
        hidden_states.new_empty((padded_rows, hidden_states.shape[1] * 2), dtype=torch.bfloat16),
        hidden_states.new_empty((hidden_states.shape[0], top_k), dtype=torch.bfloat16),
        hidden_states.new_empty((hidden_states.shape[0], top_k), dtype=torch.int32),
    )


def _dequantize_activation(hidden_states: torch.Tensor, hidden_states_scale: torch.Tensor):
    """`[T, H]` fp32 value of an NVFP4 activation in the op's input layout:
    packed e2m1 codes `[T, H/2]` (element 2i in the low nibble) and linear e4m3
    block scales, one per 16 elements of a row. The global scale stays in."""
    rows = hidden_states.shape[0]
    codes = torch.stack([hidden_states & 0x0F, hidden_states >> 4], dim=-1).reshape(rows, -1)
    scales = hidden_states_scale.view(torch.float8_e4m3fn).reshape(rows, -1)
    return dequantize_nvfp4(codes, scales)


class GlmRoutedExperts(OpWrapper):
    """One routed-expert MoE layer of GLM-5.2 at MoE TP16, over NVFP4 input.

    Routing happens inside the op, from one of two sources:

    * `routing_logits` `[T, 256]` -- the router GEMM's fp32 logits;
    * `router_hidden_states` `[T, 6144]` bf16 with `router_weight`
      `[256, 6144]` bf16 and `router_workspace` (`make_fused_router_workspace`):
      the routing kernel runs the router GEMV itself, bit for bit the logits of
      `dsv3_router_gemm_op`, at 1-8 rows. `routing_logits` then only gives the
      shape and is not read.

    `hidden_states_ready_event` is the raw `cudaEvent_t` of a `torch.cuda.Event`
    recorded on the stream that writes `hidden_states` / `hidden_states_scale`
    (`event.cuda_event`, after its first record): routing launches without
    waiting for it, FC1 waits. `prefer_gemm2_k128` runs FC2 on the persistent
    K = 128 kernel where it covers the call -- intermediate 128, tile N 8 and
    enough SMs, which an 8-row call meets on Rubin's SM count -- with the Gemm2
    cubin's rows, bitwise.

    `do_finalize=False` returns `[fc2 rows [P, H] bf16, expert weights [T, 8]
    bf16, expanded index -> permuted row [T, 8] int32]`: token `t`'s output is
    `sum_k weights[t, k] * rows[map[t, k]]`. `do_finalize=True` returns
    `[output [T, H] bf16]`.
    """

    ARCHS = frozenset({Arch.SM_107})

    # GLM-5.2's routed geometry at MoE TP16, ep 1: every rank holds all 256
    # experts with a 2048 / 16 = 128-wide intermediate slice.
    _GLM_TP16 = dict(
        hidden=6144,
        intermediate=128,
        num_experts=256,
        top_k=8,
        routed_scaling_factor=2.5,
        local_expert_offset=0,
        local_num_experts=256,
    )

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the c1 verify step: 8 rows routed by the fused router GEMV, the input joined "
            "right before FC1, FC2 at K = 128, unfinalized for the post-MoE hop",
            spec=dict(tokens=8, fused_router=True, late_join=True, gemm2_k128=True, **_GLM_TP16),
        ),
        Cell(
            why="one row, a step without MTP: the fused router's single-token instance and "
            "FC2 at K = 128 over 8 token tiles",
            spec=dict(tokens=1, fused_router=True, late_join=True, gemm2_k128=True, **_GLM_TP16),
        ),
        Cell(
            why="32 rows (4 requests x 8), past the fused router: routed from the router "
            "GEMM's logits, FC2 back on the Gemm2 cubin",
            spec=dict(tokens=32, fused_router=False, late_join=False, gemm2_k128=True, **_GLM_TP16),
        ),
    )

    note = """
    Certified unfinalized (`do_finalize=False`) only; the finalize step is the
    post-MoE hop's. No expert-parallel window (`local_expert_offset` 0, all 256
    experts local), no `output` buffer, `use_dp` False, `topk_ids` /
    `topk_weights` None. `gemm1_bias`, `gemm1_alpha`, `gemm1_beta`,
    `gemm1_clamp_limit` and `gemm2_bias` are None, `act_type` SwiGlu.

    `routing_bias` is GLM-5.2's `e_score_correction_bias` in bf16, the dtype
    the TRTLLM MoE backend loads it in; `n_group` / `topk_group` are 1.

    The tactic is the autotuner's: the cells run cold (the op's default tile,
    N 8 at these row counts). Under autotuning, a call routed by the fused
    router profiles from `routing_logits`, so the caller fills that buffer
    once (random logits) rather than leaving it uninitialized.

    A router workspace is per call site and its launches must run in stream
    order; the kernel leaves the exchange buffer of the next launch idle and
    advances the epoch word. `tune_max_num_tokens` is an autotuner bucket hint.

    The unfinalized output's padded row count depends on the autotuned tile.
    Dynamo fullgraph tracing of that mode requires
    `torch._dynamo.config.capture_dynamic_output_shape_ops = True`; finalized
    calls have static output shapes and do not require that setting.

    The caller owns the weight layout -- FC1 rows `[up ; gate]` interleaved and
    32-row block shuffled, FC2 rows block shuffled, both scale stacks 128x4
    swizzled -- as the catalog `fp4_block_scale_moe_runner` entry describes.
    """

    def __call__(
        self,
        routing_logits: torch.Tensor,
        routing_bias: torch.Tensor,
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
        routed_scaling_factor: float,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        output: Optional[torch.Tensor],
        tune_max_num_tokens: int,
        use_dp: bool,
        hidden_states_ready_event: Optional[int],
        router_hidden_states: Optional[torch.Tensor],
        router_weight: Optional[torch.Tensor],
        router_workspace: Optional[torch.Tensor],
        prefer_gemm2_k128: bool,
    ) -> list[torch.Tensor]:
        result = torch.ops.trtllm.glm_routed_experts(
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
            topk_weights,
            topk_ids,
            output,
            tune_max_num_tokens,
            use_dp,
            hidden_states_ready_event,
            router_hidden_states,
            router_weight,
            router_workspace,
            prefer_gemm2_k128,
        )
        return [result[0]] if do_finalize else list(result)

    def reference(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        up: Operand,
        gate: Operand,
        down: Operand,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        top_k: int,
        local_expert_offset: int,
        routed_scaling_factor: float,
        router_hidden_states: Optional[torch.Tensor] = None,
        router_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer, in plain torch over dequantized weights, fp32 throughout.

        Returns `(output [T, H] bf16, topk ids [T, top_k], topk weights [T,
        top_k] bf16)`: the finalized layer, rounded once, and the routing it
        used. A `do_finalize=False` call is compared after summing its rows with
        its own weights.

        The documented exception to mirroring: `__call__` takes the prepared
        weight operands -- shuffled rows, swizzled scales -- and a reference
        given those would re-derive the layout rather than the computation, so
        this takes the per-expert `(codes, block scales)` pairs `up`, `gate`
        and `down` the layout was built from. Everything else keeps the name it
        has in `__call__`.

        Routing: the logits are `routing_logits`, or the router GEMV
        `router_hidden_states @ router_weight.T` when those are given (the
        fused router). Scores are `sigmoid(logits)` -- in the routing kernel's
        `0.5 tanh(x / 2) + 0.5` form --, experts are chosen by score + bias,
        and an expert's weight is its score over the sum of the chosen scores,
        times `routed_scaling_factor`, rounded to bf16 as the kernel stores it.

        Experts: the op's own scalars carry the global scales. FC1's gate half
        is dequantized by `output1_scale_gate_scalar` (`a1`), its linear half
        by the same `a1`; the SwiGLU output is requantized to NVFP4 under the
        FC2-input global scale `g2 = output1_scale_scalar / a1`, which the FC1
        epilogue folds in, and FC2's accumulator is dequantized by
        `output2_scale_scalar`. Skipping that requantization is a reference
        some 20 ulp RMS away from a correct kernel.
        """
        x = _dequantize_activation(hidden_states, hidden_states_scale)
        num_tokens, hidden = x.shape
        with true_fp32_matmul():
            if router_hidden_states is not None:
                logits = router_hidden_states.float() @ router_weight.float().t()
            else:
                logits = routing_logits.float()
            scores = 0.5 * torch.tanh(0.5 * logits) + 0.5
            chosen = torch.topk(scores + routing_bias.float(), top_k, dim=-1).indices
            picked = scores.gather(1, chosen)
            weights = (picked * routed_scaling_factor / picked.sum(-1, keepdim=True)).to(
                torch.bfloat16
            )

            up_c, up_s = up
            gt_c, gt_s = gate
            dn_c, dn_s = down
            out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=x.device)
            for local_e in range(up_c.shape[0]):
                tok, slot = (chosen == local_expert_offset + local_e).nonzero(as_tuple=True)
                if tok.numel() == 0:
                    continue
                a1 = output1_scale_gate_scalar[local_e].float()
                g2 = output1_scale_scalar[local_e].float() / a1
                a2 = output2_scale_scalar[local_e].float()
                xe = x[tok]
                u = (xe @ dequantize_nvfp4(up_c[local_e], up_s[local_e]).t()) * a1
                g = (xe @ dequantize_nvfp4(gt_c[local_e], gt_s[local_e]).t()) * a1
                act = quantize_nvfp4(u * g * torch.sigmoid(g), float(g2))
                y = (act @ dequantize_nvfp4(dn_c[local_e], dn_s[local_e]).t()) * a2
                out.index_add_(0, tok, y * weights[tok, slot].float().unsqueeze(1))
        return out.to(torch.bfloat16), chosen.to(torch.int32), weights

    def is_valid(
        self,
        routing_logits: torch.Tensor,
        routing_bias: torch.Tensor,
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
        routed_scaling_factor: float,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: Optional[torch.Tensor],
        topk_ids: Optional[torch.Tensor],
        output: Optional[torch.Tensor],
        tune_max_num_tokens: int,
        use_dp: bool,
        hidden_states_ready_event: Optional[int],
        router_hidden_states: Optional[torch.Tensor],
        router_weight: Optional[torch.Tensor],
        router_workspace: Optional[torch.Tensor],
        prefer_gemm2_k128: bool,
    ) -> None:
        """The ways this op takes an input and answers it wrongly.

        Shapes, dtypes, the routing configuration the fused router covers and
        the workspace size are rejected loudly by the op and are not restated.
        """
        # The kernels take raw data pointers and assume dense row-major
        # tensors; a strided view is read as if dense.
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("routing_bias", routing_bias),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("output1_scale_scalar", output1_scale_scalar),
            ("output1_scale_gate_scalar", output1_scale_gate_scalar),
            ("output2_scale_scalar", output2_scale_scalar),
            ("router_hidden_states", router_hidden_states),
            ("router_weight", router_weight),
            ("router_workspace", router_workspace),
            ("output", output),
        ):
            if tensor is not None:
                assert tensor.is_contiguous(), (
                    f"{name} must be contiguous; a strided view is read as if dense "
                    "and silently produces wrong results"
                )
        # The prepared block-scale layout is misread at a hidden size that is
        # only a multiple of 128 (see the catalog fp4_block_scale_moe_runner).
        assert gemm1_weights.shape[-1] * 2 % 256 == 0, (
            "hidden size (gemm1_weights.shape[-1] * 2) must be a multiple of 256; "
            "the prepared block-scale layout is silently misread otherwise"
        )
        # A torch.cuda.Event has no cudaEvent_t before its first record, and a
        # null handle reads as "no event": FC1 would not wait for the input.
        assert hidden_states_ready_event is None or hidden_states_ready_event != 0, (
            "hidden_states_ready_event is a null cudaEvent_t: take event.cuda_event after "
            "the event is recorded, or FC1 reads hidden_states without waiting"
        )
        # The routing CTA takes any value but -0.0 in the exchange buffer of
        # this launch as a published slice sum. Unless that buffer is idle, it
        # routes on stale sums -- a zero-filled workspace routes on zero logits.
        if router_hidden_states is not None and router_workspace is not None:
            words = router_workspace.view(torch.int32)
            buffer_words = (FUSED_ROUTER_WORKSPACE_WORDS - 4) // 2
            epoch = int(words[2 * buffer_words].item())
            current = words[(epoch & 1) * buffer_words : ((epoch & 1) + 1) * buffer_words]
            assert bool((current == _IDLE_BITS).all()), (
                "router_workspace is not in its between-launch state (the exchange buffer "
                "of the next launch must hold -0.0 everywhere); build it with "
                "make_fused_router_workspace and let only this op write it"
            )


routed_experts = GlmRoutedExperts()
