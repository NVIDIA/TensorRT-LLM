# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-style FP8 block-scale mixture-of-experts layer, routed in the kernel: sigmoid +
correction-bias top-k routing, grouped FC1 GEMM, FP8 requantization, clamped SwiGLU, FP8
requantization, grouped FC2 GEMM and the routing-weighted combine, in one trtllm-gen call."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Block edge on every scale axis: 1x128 activations, 128x128 weights.
BLOCK = 128

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The activation kernel's floor on a block amax (`kDsActAmaxEpsilon` in
#: trtllmGen/DevKernel.cu). Only an all-zero block reaches it.
ACT_AMAX_FLOOR = 1.0e-10

#: `routing_method_type` values (`RoutingMethodType` in kernels/moe/trtllmGen/runner.h).
ROUTING_RENORMALIZE = 1
ROUTING_DEEPSEEK_V3 = 2
ROUTING_MINIMAX2 = 5

#: The routing methods that read `routing_bias`; every other one ignores it.
_BIAS_READING_ROUTING = frozenset({ROUTING_DEEPSEEK_V3, ROUTING_MINIMAX2})


def dequantize_block_weight(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """One expert's `[R, C]` fp32 weight from e4m3 codes and `[R/128, C/128]` scales."""
    r, c = codes.shape
    blocks = codes.float().reshape(r // BLOCK, BLOCK, c // BLOCK, BLOCK)
    return (blocks * scales.float().reshape(r // BLOCK, 1, c // BLOCK, 1)).reshape(r, c)


def requantize_1x128(x: torch.Tensor) -> torch.Tensor:
    """The e4m3 round trip the kernel applies to the FC1 output and to the activation.

    Per row and per 128 consecutive columns: `scale = max(amax, 1e-10) / 448`, then
    round-to-nearest-even into e4m3 saturating at +-448. The plain `amax / 448` scale, not a
    power of two.
    """
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR)
    # Tensor-by-tensor divisions: torch divides by a Python scalar as a multiply by its
    # rounded reciprocal.
    scale = amax / torch.full_like(amax, E4M3_MAX)
    codes = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(shape)


def route_deepseek_v3(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ungrouped DeepSeek-V3 routing as the kernel evaluates it: `(ids [T, k] int64, weights
    [T, k] bf16)`.

    `s = sigmoid(logit)` in its tanh form, `0.5 * tanh(x / 2) + 0.5`, in fp32. The experts are
    chosen by `s + bias`; each chosen expert's weight is its unbiased score, recovered as
    `(s + bias) - bias` the way the kernel recovers it, times `routed_scaling_factor` (1 when
    None) over the sum of the chosen scores, rounded to bf16 -- the dtype the combine reads.
    Ties go to the lower expert index here; the kernel's tie rule is not certified.
    """
    score = 0.5 * torch.tanh(0.5 * routing_logits.float()) + 0.5
    bias = routing_bias.float()
    choice = score + bias
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :top_k]
    picked = choice.gather(1, ids) - bias[ids]
    scale = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    weights = picked * scale / picked.sum(dim=-1, keepdim=True)
    return ids, weights.to(torch.bfloat16)


class Fp8BlockScaleMoeRunner(OpWrapper):
    """One FP8 block-scale MoE layer over e4m3 activations, routed from the router logits.

    Routing, per token `t` (when `topk_ids` is None, `routing_method_type` DeepSeek-V3 and
    `n_group` at most 1): `s = sigmoid(routing_logits[t])`, the `top_k` experts by `s +
    routing_bias`, and combine weights `s_e * routed_scaling_factor / sum(s_chosen)` rounded to
    bf16 -- see `route_deepseek_v3`. With `topk_ids` / `topk_weights` given instead, those are
    the routing, used as given.

    Then per token `t` and slot `j`, with `A[t]` the fp32 activations
    (`hidden_states[t, k] * hidden_states_scale[k // 128, t]`) and `W_*` the dequantized
    128x128-block weights of local expert `e = id[t, j] - local_expert_offset` (slots outside
    `[0, local_num_experts)` contribute nothing):

        up, gate = requant(A[t] @ W1[e].T).split(I)          # FC1 rows are [up ; gate]
        gate, up = min(gate, L), clamp(up, -L, L)            # L = gemm1_clamp_limit
        y        = bf16(requant(silu(gate) * up) @ W2[e].T)
        out[t]  += w[t, j] * y

    accumulated in fp32 and stored bf16, `requant` being `requantize_1x128`. FC1 rows are
    `[up ; gate]` -- up first, the checkpoint's `[w3 ; w1]` -- the opposite of a dense MLP's
    fused `[gate ; up]` weight.

    Fusion boundary: routing, permutation, both GEMMs with on-the-fly dequantization, both
    intermediate requantizations, the clamped SwiGLU and the weighted combine are inside.
    The router GEMM, quantizing the bf16 hidden states (`fp8_quantize_1x128`), the shared
    expert, any all-reduce and the residual are the caller's.

    `hidden_states_scale` is `[H/128, T]` -- K-block-major, exactly what `fp8_quantize_1x128`
    returns on this architecture. Pass it straight through: at `T == H/128` (32 rows at
    H 4096) its transpose has the same shape and passes every check, and the answer is wrong.

    Returns a fresh `[T, H]` bf16 tensor, or an empty `[0]` bf16 tensor when `output` is given
    (the result is then in `output`).
    """

    ARCHS = frozenset({Arch.SM_100})

    #: GLM-5.3-Flash's routed experts at TP4: hidden 4096, moe_intermediate_size 2048 split
    #: four ways (512), all 288 experts local, ungrouped DeepSeek-V3 top-8 routing scaled by
    #: 2.5, swiglu_limit 10.
    GLM53_FLASH_TP4 = dict(
        hidden=4096,
        inter=512,
        num_experts=288,
        top_k=8,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        gemm1_clamp_limit=10.0,
        local_expert_offset=0,
        local_num_experts=288,
    )

    # Only GLM-5.3-Flash calls this entry. Its own cells are the routing-kernel boundaries at
    # GLM's geometry -- 288 experts dispatch to the 384-expert tier, where 2 to 8 rows run the
    # small-token kernel, 9 to 256 the single-cluster kernel and more the multi-kernel path.
    # The row counts the target runs are in the test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="9 rows: the first count on the single-cluster routing kernel",
            spec=dict(tokens=9, **GLM53_FLASH_TP4),
        ),
        Cell(
            why="2,047 rows: past the single-cluster kernel's 256, the multi-kernel routing path",
            spec=dict(tokens=2047, **GLM53_FLASH_TP4),
        ),
    )

    note = """
    Metadata consumed: none. Two process-global caches sit behind the call: the autotuner's
    profiling cache and the C++ runner cache. Outside an `autotune()` context a cold cache
    runs a deterministic fallback tactic keyed by the row count; a warm, tuned tactic is not
    certified here.

    The op is a Python `torch.library.custom_op` (`trtllm::fp8_block_scale_moe_runner` in
    `_torch/custom_ops/trtllm_gen_custom_ops.py`) over the compiled-in C++ class
    `torch.classes.trtllm.FP8BlockScaleMoERunner` (thop/moe/trtllmGen/fp8BlockScaleMoe.cpp),
    which launches the routing kernels, trtllm-gen's precompiled batched-GEMM cubins, a
    separate activation kernel and a finalize kernel: nothing is JIT-compiled.

    Routing. DeepSeek-V3 with `n_group` at most 1 runs the custom routing kernels with a
    sigmoid + bias preprocess and a scaled sum-normalize postprocess (kernels/moe/trtllmGen/
    runner.cu); which kernel depends on the row count at the expert tier (288 experts use the
    384 tier): 1 row the cooperative block kernel, 2-8 the small-token kernel, 9-256 the
    single-cluster kernel, more the histogram + offsets path. The sigmoid is the tanh form and
    the weights are written as bf16, which `route_deepseek_v3` reproduces. Both
    `routing_logits` and `routing_bias` may be fp32 or bf16; GLM-5.3-Flash passes fp32 for
    both (rounding the correction bias to bf16 can reorder near-equal experts). The kernel's
    tie rule is not certified: the test's routing has no near-ties.

    Preconditions, each enforced loudly by the op: `hidden_states` `[T, H]` e4m3 and
    `hidden_states_scale` `[H/128, T]` fp32 (see the class docstring); weights in the plain
    per-expert layout, no shuffle and no swizzle: `gemm1_weights` `[E, 2I, H]` e4m3 rows
    `[up ; gate]`, `gemm1_weights_scale` `[E, 2I/128, H/128]` fp32, `gemm2_weights` `[E, H, I]`,
    `gemm2_weights_scale` `[E, H/128, I/128]`, `E == local_num_experts`; H and I multiples of
    128; `intermediate_size == gemm1_weights.shape[1] // 2`; `routing_logits` `[T, num_experts]`
    and `routing_bias` `[num_experts]`; `num_experts > top_k`.

    `gemm1_clamp_limit` is one float for every expert, applied after the FC1 requantization
    (gate on the upper side, up on both). Any value given is applied, 0 included -- unlike
    `silu_and_mul`, which drops a non-positive limit. None means no clamp.

    Silent on the raw op, refused by `is_valid`: any non-contiguous tensor (read as dense);
    `act_type != 0` (the Python runner forces SwiGLU); `routing_logits` or `routing_bias`
    alongside `topk_ids` (both ignored); a `routing_bias` with a routing method that does not
    read it; DeepSeek-V3 routing from logits without a `routing_bias` (the routing kernels read
    it for every expert with no null check -- read from the source, not driven); a
    `local_expert_offset` outside `[0, num_experts)`.

    The registered fake returns `[T, 2H]` (it assumes packed fp4 hidden states): an upstream
    defect; eager is unaffected.

    Certified (sm_100): GLM-5.3-Flash's routed experts at TP4 as listed above, routed in the
    kernel, at 5, 8, 9, 32, 256 and 2,047 rows.

    Not certified: the pre-routed path (`topk_ids` given), grouped routing (`n_group > 1`),
    other routing methods, bf16 logits or bias, `num_fused_shared_experts > 0`,
    expert-parallel windows, `output=`, a warm autotuner cache, CUDA-graph capture, other
    geometries, sm_103.
    """

    def __call__(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_fused_shared_experts: Optional[int],
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        act_type: int = 0,
        gemm1_clamp_limit: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> torch.Tensor:
        return torch.ops.trtllm.fp8_block_scale_moe_runner(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            num_experts,
            top_k,
            num_fused_shared_experts,
            n_group,
            topk_group,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            routing_method_type,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            act_type=act_type,
            gemm1_clamp_limit=gemm1_clamp_limit,
            output=output,
            tune_max_num_tokens=tune_max_num_tokens,
            use_dp=use_dp,
        )

    def reference(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_fused_shared_experts: Optional[int],
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        act_type: int = 0,
        gemm1_clamp_limit: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> torch.Tensor:
        """The layer over bit-identical operands: fp32 GEMMs, the kernel's two e4m3
        requantizations, its bf16 FC2 rows and bf16 combine weights, one bf16 store.

        Routes with `route_deepseek_v3` unless `topk_ids` / `topk_weights` are given. Not
        modeled, so refused: fused shared experts, grouped routing and any routing method but
        DeepSeek-V3 when routing from logits. Ignored: `num_experts` (the routing space is
        `routing_logits.shape[1]`), `topk_group`, `act_type`, `output`,
        `tune_max_num_tokens`, `use_dp`. Returns the `[T, H]` result even when `output` is
        given.
        """
        assert not num_fused_shared_experts, "the reference does not model fused shared experts"
        if topk_ids is None:
            assert routing_logits is not None and routing_bias is not None
            assert routing_method_type == ROUTING_DEEPSEEK_V3 and (n_group or 1) <= 1, (
                "the reference routes ungrouped DeepSeek-V3 only"
            )
            ids, weights = route_deepseek_v3(
                routing_logits, routing_bias, top_k, routed_scaling_factor
            )
        else:
            assert topk_weights is not None
            ids, weights = topk_ids.long(), topk_weights.to(torch.bfloat16)
        weights = weights.float()

        num_tokens, hidden = hidden_states.shape
        nb = hidden // BLOCK
        act_in = (
            hidden_states.float().reshape(num_tokens, nb, BLOCK)
            * hidden_states_scale.float().t().reshape(num_tokens, nb, 1)
        ).reshape(num_tokens, hidden)
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        local = ids - local_expert_offset
        live = local[(local >= 0) & (local < local_num_experts)]
        inter = intermediate_size
        with true_fp32_matmul():
            for e in torch.unique(live).tolist():
                tok, slot = (local == e).nonzero(as_tuple=True)
                w1 = dequantize_block_weight(gemm1_weights[e], gemm1_weights_scale[e])
                h1 = requantize_1x128(act_in[tok] @ w1.t())
                up, gate = h1[:, :inter], h1[:, inter:]
                if gemm1_clamp_limit is not None:
                    gate = gate.clamp(max=gemm1_clamp_limit)
                    up = up.clamp(-gemm1_clamp_limit, gemm1_clamp_limit)
                act = requantize_1x128(gate / (1.0 + torch.exp(-gate)) * up)
                w2 = dequantize_block_weight(gemm2_weights[e], gemm2_weights_scale[e])
                y = (act @ w2.t()).to(torch.bfloat16).float()
                # A row picks an expert at most once, so `tok` holds no duplicates.
                out[tok] += y * weights[tok, slot].unsqueeze(1)
        return out.to(torch.bfloat16)

    def is_valid(
        self,
        routing_logits: Optional[torch.Tensor],
        routing_bias: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_fused_shared_experts: Optional[int],
        n_group: Optional[int],
        topk_group: Optional[int],
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: Optional[float],
        routing_method_type: int,
        topk_weights: Optional[torch.Tensor] = None,
        topk_ids: Optional[torch.Tensor] = None,
        act_type: int = 0,
        gemm1_clamp_limit: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
        tune_max_num_tokens: int = 8192,
        use_dp: bool = False,
    ) -> None:
        """Inputs the op accepts and answers wrongly, or faults on.

        The transposed activation-scale buffer at `T == H/128` is one more, and no metadata
        check can see it.
        """
        # The kernels take raw data pointers: a strided view is read as dense.
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("routing_bias", routing_bias),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("gemm1_weights", gemm1_weights),
            ("gemm1_weights_scale", gemm1_weights_scale),
            ("gemm2_weights", gemm2_weights),
            ("gemm2_weights_scale", gemm2_weights_scale),
            ("topk_weights", topk_weights),
            ("topk_ids", topk_ids),
            ("output", output),
        ):
            if tensor is not None:
                assert tensor.is_contiguous(), (
                    f"{name} must be contiguous; a strided view is read as if dense "
                    "and silently produces wrong results"
                )
        # The Python runner overwrites the activation with SwiGLU.
        assert act_type == 0, (
            f"act_type={act_type} is silently ignored -- this op always computes SwiGLU"
        )
        if topk_ids is not None:
            # Once topk_ids carry the routing, nothing reads the logits or the bias.
            assert routing_logits is None and routing_bias is None, (
                "routing_logits and routing_bias are ignored when topk_ids/topk_weights are "
                "given (routing has already happened)"
            )
        else:
            # DeepSeek-V3 routing loads routing_bias[e] for every expert with no null check.
            if routing_method_type == ROUTING_DEEPSEEK_V3:
                assert routing_bias is not None, (
                    "DeepSeek-V3 routing from logits reads routing_bias for every expert and "
                    "faults on None; pass zeros for no correction"
                )
            # Only the DeepSeek-V3 and MiniMax-2 routing kernels are handed the bias.
            if routing_bias is not None:
                assert routing_method_type in _BIAS_READING_ROUTING, (
                    f"routing_method_type={routing_method_type} ignores routing_bias; only "
                    "DeepSeek-V3 (2) and MiniMax-2 (5) routing read it"
                )
        # `num_experts` is the routing space, not the local window.
        assert 0 <= local_expert_offset < num_experts, (
            f"local_expert_offset={local_expert_offset} is outside the routing space "
            f"[0, num_experts={num_experts}); num_experts is every routed expert, not the "
            "window's count"
        )


fp8_block_scale_moe_runner = Fp8BlockScaleMoeRunner()
