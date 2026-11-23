# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-style FP8 block-scale mixture-of-experts, one trtllm-gen call.

Owned by the Hy4-Flash unified target: `LatentMoe._routed` hands it the router
GEMM's logits and the checkpoint's fp32 bias and leaves `topk_ids` and
`topk_weights` None, so **the kernel routes in-op** over the local expert window
-- all 512 experts at tp4, one 128-expert window per rank at EP4, one 256-expert
window per rank at EP2 (the disaggregated deployment's context engine at tep2,
`dims.py`'s `expert_offset`/`local_experts`). The pre-routed path (ids and
weights given) is the same op call and is certified alongside it.

Adapted from the pre-routed entry of the same kernel at
`models/qwen4_exp/ops/moe/fp8_block_scale_moe_runner.py` (NVIDIA, Apache-2.0):
same op call, same reference arithmetic and same guards, with `CELLS` re-derived
from this checkpoint, the reference made faithful to the kernel's FC2 store
(each expert's FC2 row is bf16 in memory, as `finalizeDeepSeekKernel` reads it)
and extended to the kernel's own router (`in_op_routing`).
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Block edge on every scale axis: 1x128 activations, 128x128 weights.
BLOCK = 128

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The activation kernel's floor on a block amax (`kDsActAmaxEpsilon` in
#: `cpp/tensorrt_llm/kernels/moe/trtllmGen/DevKernel.cu`). Only an all-zero
#: block reaches it.
ACT_AMAX_FLOOR = 1.0e-10

#: `RoutingMethodType::DeepSeekV3` (`cpp/tensorrt_llm/kernels/moe/trtllmGen/runner.h`),
#: this checkpoint's router family. Live when the ids are absent: `runner.cu`
#: takes its `nGroup <= 1` and no-fused-shared-expert case to `routingCustom`
#: (SigmoidBias + ScaledSumNormalize), not to the grouped DeepSeek kernel. Inert
#: once `topk_ids` carry the routing.
ROUTING_DEEPSEEK_V3 = 2


def dequantize_block_weight(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """One expert's `[M, K]` fp32 weight from e4m3 codes and `[M/128, K/128]` scales."""
    m, k = codes.shape
    blocks = codes.float().reshape(m // BLOCK, BLOCK, k // BLOCK, BLOCK)
    return (blocks * scales.float().reshape(m // BLOCK, 1, k // BLOCK, 1)).reshape(m, k)


def requantize_1x128(x: torch.Tensor) -> torch.Tensor:
    """The e4m3 round trip the kernel applies to both intermediates.

    Per row and per 128 consecutive columns: `scale = max(amax, 1e-10) / 448`,
    then a divide rounded to nearest-even into e4m3 saturating at +-448. The
    plain `amax / 448` scale, not the round-up power of two the MX members use.
    """
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR) / E4M3_MAX
    codes = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(shape)


def in_op_routing(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ids and weights the kernel's own router produces, in its arithmetic.

    With `routing_method_type == 2` and no ids, `runner.cu` reads this
    checkpoint's `nGroup <= 1` and no-fused-shared-expert case as
    `routingCustom` (`routing/RoutingCustomKernels.cuh`) with the
    `SigmoidBias` preprocess and the `ScaledSumNormalize` postprocess
    (`routing/RoutingCustomPolicy.cuh`):

        sel[j] = sigmoid_accurate(logit[j]) + bias[j]     # all fp32
        top-k  = the k largest `sel[j]`, ties to the lower expert id
        w[j]   = (sel[j] - bias[j]) * scale / (sum of the (sel - bias) + 0)

    `sigmoid_accurate` is `0.5f * tanhf(0.5f * x) + 0.5f`, not `1/(1+exp(-x))`:
    the two agree to within an fp32 ulp in the body and only the tanh form
    saturates to exactly 1.0 / 0.0 in the tails. The bias is loaded per expert
    through `loadScalar` -- an fp32 bias as itself, a bf16 one widened -- and the
    postprocess recovers the sigmoid score as `sel - bias`, i.e. through the
    rounded selection score, never the pre-add value. The sum is a warp reduce
    with `mSumEpsilon = 0.0f` (never set on this path), and the weights are
    stored bf16 (`mDtypeOutput = Bfloat16`) for the MoE to consume.

    Selection ranks every expert in the routing space (`num_experts`), whatever
    the local window is; a window's own call still returns the whole top-k.

    Returns `(topk_ids [T, top_k] int32, topk_weights [T, top_k] bf16)`, both
    newly allocated and contiguous.
    """
    scores = 0.5 * torch.tanh(0.5 * routing_logits.float()) + 0.5
    choice = scores + routing_bias.float()
    ids = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :top_k]
    sigmoid = torch.gather(choice, 1, ids) - routing_bias.float()[ids]
    total = sigmoid.sum(dim=-1, keepdim=True)
    weights = (sigmoid * routed_scaling_factor / total).to(torch.bfloat16)
    return ids.to(torch.int32).contiguous(), weights.contiguous()


#: The routed experts at this checkpoint's geometry (vocabulary row "routed
#: experts"): 512 experts over the 1536-wide latent, top-16, clamp 10.0, scaling
#: 3.998. `local_num_experts` is 512 at tp4 (every expert on every rank).
_HY4 = dict(
    hidden=1536,
    inter=1536,
    num_experts=512,
    top_k=16,
    n_group=1,
    topk_group=1,
    local_expert_offset=0,
    local_num_experts=512,
    gemm1_clamp_limit=10.0,
    routed_scaling_factor=3.998,
)


def _window(offset: int, local: int = 128) -> dict:
    """One expert-parallel rank's window of the 512: the router still ranks all
    512, so `num_experts` stays 512 and only the window's weights are local.
    128 wide at EP4, 256 at EP2."""
    return {**_HY4, "local_expert_offset": offset, "local_num_experts": local}


class Fp8BlockScaleMoeRunner(OpWrapper):
    """One FP8 block-scale MoE layer over e4m3 activations.

    Per token `t` and slot `j`, with `A[t]` the fp32 activations
    (`hidden_states[t, k] * hidden_states_scale[k // 128, t]`) and `W_*` the
    dequantized 128x128-block weights of local expert `e = topk_ids[t, j] -
    local_expert_offset` (slots outside `[0, local_num_experts)` contribute
    nothing):

        up, gate = requant(A[t] @ [W_up; W_gate].T).split(I)
        gate, up = min(gate, lim), clamp(up, -lim, lim)    # gemm1_clamp_limit
        out[t]  += topk_weights[t, j] * bf16(requant(silu(gate) * up) @ W_down.T)

    accumulated in fp32 and stored bf16, `requant` being `requantize_1x128` and
    `bf16(...)` the store of each expert's FC2 row into the bf16 `gemm2_output`
    buffer. FC1 rows are `[up ; gate]` -- up first, the engine loader's
    `[w3; w1]` (the checkpoint's `gate_up_proj` is gate-first; the loader
    reorders it).

    Fusion boundary: permutation, both GEMMs with on-the-fly dequantization,
    both intermediate requantizations, the clamped SwiGLU and the weighted
    combine are inside, and so is the router when `topk_ids` is None: with
    `routing_logits` and `routing_bias` given and `routing_method_type == 2` the
    kernel selects top-k, renormalizes and applies `routed_scaling_factor`
    itself (`in_op_routing`), ranking all `num_experts` whatever the local window
    is. That is the path `LatentMoe._routed` takes -- `modeling.py` imports
    `ops/moe/noaux_tc_op__r17.py` and never calls it. On the pre-routed path the
    ids and weights are the caller's and the router GEMM
    (`cublas_mm(x, router.T)`) always is. Quantizing the bf16 hidden states
    (`fp8_quantize_1x128(x, use_ue8m0=False)`), the shared expert, the latent
    projections, any all-reduce and the residual are the caller's.

    `hidden_states_scale` is `[H/128, T]` -- transposed, exactly what
    `fp8_quantize_1x128` returns on this architecture. Pass it straight
    through: at `T == H/128` (12 tokens at H 1536) its transpose has the same
    shape and passes every check, and is silently wrong.

    Returns a fresh `[T, H]` bf16 tensor, or an empty `[0]` bf16 tensor when
    `output` is given (the result is then in `output`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # The token column at this checkpoint's geometry. tp4 keeps every expert
        # local, so the whole-stack cells are its call; the token count picks the
        # kernel's routing and tile tactics.
        Cell(
            why="decode: one token through all 512 experts (tp4's local stack)",
            spec=dict(tokens=1, **_HY4),
        ),
        Cell(
            why="12 rows == H/128: the activation scale buffer is square",
            spec=dict(tokens=12, **_HY4),
        ),
        Cell(
            why="64 rows: a short batch through all 512 experts",
            spec=dict(tokens=64, **_HY4),
        ),
        Cell(
            why="prefill chunk: 5120 rows through all 512 experts",
            spec=dict(tokens=5120, **_HY4),
        ),
        # EP4 runs one 128-expert window per rank, at the same row counts. One
        # cell per window offset, and the 5120-row chunk on a window.
        Cell(
            why="EP4 decode: one token, rank 0's 128-expert window",
            spec=dict(tokens=1, **_window(0)),
        ),
        Cell(
            why="EP4: 64 rows, rank 1's 128-expert window",
            spec=dict(tokens=64, **_window(128)),
        ),
        Cell(
            why="EP4 prefill chunk: 5120 rows, rank 2's 128-expert window",
            spec=dict(tokens=5120, **_window(256)),
        ),
        Cell(
            why="EP4 decode: one token, the last 128-expert window (384)",
            spec=dict(tokens=1, **_window(384)),
        ),
        # EP2 runs two 256-expert windows (the disaggregated deployment's
        # context engine at tep2, its decode and its 5120-token prefill chunk).
        Cell(
            why="the ctx engine's decode, one token, through EP2 rank 0's 256-expert window",
            spec=dict(tokens=1, **_window(0, 256)),
        ),
        Cell(
            why="a 64-row batch, rank 0's 256-expert window",
            spec=dict(tokens=64, **_window(0, 256)),
        ),
        Cell(
            why="a 5120-token prefill chunk, rank 1's 256-expert window",
            spec=dict(tokens=5120, **_window(256, 256)),
        ),
        Cell(
            why="the ctx engine's decode, one token, through EP2 rank 1's 256-expert window",
            spec=dict(tokens=1, **_window(256, 256)),
        ),
        # The path `LatentMoe._routed` takes: the router GEMM's logits and the
        # fp32 bias go in, `topk_ids`/`topk_weights` stay None, and the kernel
        # routes in-op. Its two ends: the decode step and the prefill chunk, the
        # whole stack at tp4 and one EP4 window.
        Cell(
            why="the engine's decode: one token through all 512 experts at tp4, routed in-op "
            "from the router logits and bias",
            spec=dict(
                entry="in-kernel-routing",
                tokens=1,
                routing_method_type=ROUTING_DEEPSEEK_V3,
                topk_ids=None,
                topk_weights=None,
                **_HY4,
            ),
        ),
        Cell(
            why="the engine's 5120-token prefill chunk and one EP4 window (128 experts at "
            "offset 128): the in-op router ranks all 512",
            spec=dict(
                entry="in-kernel-routing",
                tokens=5120,
                routing_method_type=ROUTING_DEEPSEEK_V3,
                topk_ids=None,
                topk_weights=None,
                **_window(128),
            ),
        ),
    )

    note = """
    Metadata consumed: none. Two process-global caches sit behind the call --
    the autotuner's profiling cache and the C++ runner cache. The autotuner's
    buckets stop at `tune_max_num_tokens` (default 8192, which the cells stay
    under); a tuned tactic is a reordering of the same arithmetic.

    The op is a Python `torch.library.custom_op`
    (`trtllm::fp8_block_scale_moe_runner` in
    `_torch/custom_ops/trtllm_gen_custom_ops.py`) over the compiled-in C++ class
    `torch.classes.trtllm.FP8BlockScaleMoERunner`
    (`cpp/tensorrt_llm/thop/moe/trtllmGen/fp8BlockScaleMoe.cpp`), which launches
    trtllm-gen's precompiled batched-GEMM, activation and finalize cubins:
    nothing is JIT-compiled. The thop checks `isSM100Family()` (sm 100..109), so
    sm_103 is the arch this target runs it on.

    Preconditions. `hidden_states` is `[T, H]` e4m3 and `hidden_states_scale`
    `[H/128, T]` fp32 (see the class docstring); weights are the checkpoint's
    per-expert tensors in plain row-major layout, no shuffle and no swizzle:
    `gemm1_weights` `[E, 2I, H]` e4m3 rows `[up ; gate]`, `gemm1_weights_scale`
    `[E, 2I/128, H/128]` fp32, `gemm2_weights` `[E, H, I]`,
    `gemm2_weights_scale` `[E, H/128, I/128]` fp32; `topk_ids` `[T, top_k]`
    int32 and `topk_weights` `[T, top_k]` bf16, used as given (nothing is
    renormalized). The checkpoint's `scale_inv` is bf16 `[512, 24, 12]` gate_up
    and `[512, 12, 12]` down; `DeepSeekFP8BlockScalesFusedMoEMethod`
    (`_torch/moe/fused_moe/quantization.py`) widens it to the fp32 scaling
    params this op takes and stores each expert's FC1 rows as `[w3(up); w1(gate)]`.
    H and I are multiples of 128, `intermediate_size == gemm1_weights.shape[1] // 2`,
    `0 < top_k < num_experts`, `top_k <= 32`. The op raises on each of those.

    The window: local slot i answers for global id `local_expert_offset + i`, and
    `num_experts` stays the routing space (512) on every window call. The
    weights and all four scale tensors hold the window's experts only: the op
    compares each one's dim 0 with `local_num_experts` and raises on the whole
    stack (the test holds all four). A token with no slot in the window comes
    back bitwise zero.

    Routing. One op call, two paths. `LatentMoe._routed` takes the in-op one:
    `routing_logits` (the gate GEMM `cublas_mm(x, router.T)`, bf16 in this
    checkpoint), `routing_bias` (fp32, `model.layers.{i}.mlp.gate.
    e_score_correction_bias`), `topk_ids`/`topk_weights` left None and
    `routing_method_type=2`. `runner.cu` reads that as `RoutingMethodType::
    DeepSeekV3` with `nGroup <= 1` and no fused shared expert and routes with
    `routingCustom` (SigmoidBias + ScaledSumNormalize), ranking all 512 experts
    whatever the local window is and applying `routed_scaling_factor` (3.998)
    inside -- so on this path the scale is live and the ids/weights the MoE
    consumes are bf16, rounded once from the fp32 router. `in_op_routing` is that
    router in plain torch. Two cells drive it: the decode step and the prefill
    chunk, the whole stack at tp4 (offset 0) and one EP4 window (offset 128).

    Inert on the *pre-routed* cells, bitwise: `routing_method_type` (the ids
    bypass the router), `routed_scaling_factor` -- the caller's router has
    already renormalized and multiplied by 3.998, and the runner must not apply
    it again -- `tune_max_num_tokens`, `use_dp`. `gemm1_clamp_limit` is live and
    one scalar for every expert, applied after the FC1 requantization; None is
    bitwise equal to omitting it.

    Silent on the raw op, refused by `is_valid` (the test holds the strided
    `hidden_states`, `act_type`, the bias-with-ids and the outside-the-routing-
    space refusals): any non-contiguous tensor (read as dense, 55-115 ulp RMS
    off at the op's other pre-routed entry); `act_type != 0` (the Python runner
    forces SwiGlu, so 1 and 2 are bitwise identical to 0); `routing_bias`
    alongside `topk_ids` (ignored; a +-1e3 bias leaves the output bitwise
    unchanged); a `local_expert_offset` outside `[0, num_experts)` (at small
    row counts the op answers all zeros; at 16 rows it reads out of bounds).
    `num_fused_shared_experts > 0` is rejected loudly on the pre-routed path.

    Certified on sm_103 (GB300), the whole test file green: the fourteen cells
    sit at most 4.09 bf16 ulp per element of the row max and 1.522 ulp RMS from
    `reference` (the test's 16 / 2.5 gate; the whole-stack cells stay at
    0.77-1.00 RMS and the 256-expert windows at 0.000-0.777, the one-token
    windows do not average -- two answer bitwise equal to the reference). The
    two in-op cells are the size of the pre-routed ones -- one token through the
    whole stack 1.19 element / 0.987 RMS, the 5120-token EP4 window 4.09 /
    0.714 -- so the kernel's own router agrees with `in_op_routing` to the same
    distance the tested router does, with no disagreement to report. The band is
    derived from measured variants, not tuned: this reference measures 0.686 ulp
    RMS where every alternative rounding model is worse -- an fp32 FC2 store
    instead of the kernel's bf16 one 0.829, no SwiGLU-product requantization
    5.77, no FC1 requantization 12.07, a ue8m0 FC1 scale 15.38. Wrong readings
    are far outside the gate: dropping the clamp 74.74 element / 46.83 RMS,
    reading the checkpoint's gate-first rows as `[up; gate]` 384.00 / 254.09,
    and routing the in-op cell's own logits with the bias zeroed -- which moves
    1012 of 1024 slots -- 438.32 / 309.71. The four EP4 windows' bf16 outputs
    summed in fp32 sit 1.90 / 0.690 from the 512-expert call at 64 rows, and the
    sum with any one window left out fails the gate; the two EP2 windows summed
    the same way sit 1.90 / 0.729, and either window left out fails it too.
    `routed_scaling_factor` and `routing_method_type` were measured
    bitwise-inert on a certified cell.

    The registered fake returns `[T, 2H]` (it assumes packed fp4 hidden
    states): an upstream defect, eager is unaffected.

    Hard limits
      T            whole stack certified at 1, 12, 64 and 5120 (the largest
                   prefill chunk); 128- and 256-expert windows at 1, 64 and 5120
      top_k        <= 32 (raises past it); certified 16, the checkpoint's
      E, H, I      certified at the checkpoint's 512, 1536, 1536
      window       the whole stack, 128 wide at offsets 0, 128, 256 and 384 of
                   512, and 256 wide at offsets 0 and 256 (the tep2 ctx engine)
      routing      in-op at `routing_method_type=2` (DeepSeekV3), `n_group <=
                   1`: at `n_group > 1` the runner takes its other kernel, the
                   grouped DeepSeek one (top_k <= 8), and `in_op_routing` no
                   longer describes it

    Not certified: windows of other widths or offsets, a window past the
    routing space's end, repeated or out-of-range ids, `output=`, the registered
    fake, the op's other loud rejections, CUDA-graph capture of the call (the
    cells run eager), in-op routing at a `routing_method_type` other than 2
    (SigmoidRenorm, MiniMax2, Llama4 and the rest are different kernels), an
    fp32 `routing_logits` (the thop takes one; the engine's gate GEMM is bf16),
    a bf16 `routing_bias` (the thop takes one), `routing_logits` without
    `routing_bias` (the in-op path loads the bias pointer for every expert; a
    null one is read), the router's tie-break (the cells' logits are continuous
    draws, so no two selection scores are equal), the checkpoint's own router
    logits and bias values (the cells draw them), multi-rank execution (every
    window here runs on one device; the cross-rank sum is the collective's),
    geometries other than the checkpoint's, sm_100 and sm_107.
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
        """The layer over bit-identical operands, fp32 GEMMs, bf16 stores.

        Both routing paths. With `routing_logits` and `routing_bias` given and
        the ids and weights None, the routing is `in_op_routing` -- the kernel's
        own router, in the kernel's arithmetic. With the ids and weights given,
        they are used as they stand; then ignored: `routing_bias`, `num_experts`,
        `top_k`, `routed_scaling_factor`, `routing_method_type`. Ignored on both:
        `num_fused_shared_experts`, `n_group`, `topk_group`, `act_type`, `output`,
        `tune_max_num_tokens`, `use_dp`. Returns the `[T, H]` result even when
        `output` is given.
        """
        if routing_logits is not None:
            assert topk_ids is None and topk_weights is None, (
                "pass either routing_logits or topk_ids/topk_weights: the kernel "
                "routes in-op only when both id and weight tensors are absent"
            )
            topk_ids, topk_weights = in_op_routing(
                routing_logits, routing_bias, top_k, routed_scaling_factor
            )
        else:
            assert topk_ids is not None and topk_weights is not None, (
                "the reference needs the routing either way: routing_logits and "
                "routing_bias, or topk_ids and topk_weights"
            )
        num_tokens, hidden = hidden_states.shape
        nb = hidden // BLOCK
        act_in = (
            hidden_states.float().reshape(num_tokens, nb, BLOCK)
            * hidden_states_scale.t().reshape(num_tokens, nb, 1).float()
        ).reshape(num_tokens, hidden)
        out = torch.zeros(num_tokens, hidden, dtype=torch.float32, device=hidden_states.device)
        local = topk_ids.long() - local_expert_offset
        live = local[(local >= 0) & (local < local_num_experts)]
        with true_fp32_matmul():
            for e in torch.unique(live).tolist():
                tok, slot = (local == e).nonzero(as_tuple=True)
                w1 = dequantize_block_weight(gemm1_weights[e], gemm1_weights_scale[e])
                y1 = requantize_1x128(act_in[tok] @ w1.t())
                up, gate = y1[:, :intermediate_size], y1[:, intermediate_size:]
                if gemm1_clamp_limit is not None:
                    gate = gate.clamp(max=gemm1_clamp_limit)
                    up = up.clamp(-gemm1_clamp_limit, gemm1_clamp_limit)
                act = requantize_1x128(torch.nn.functional.silu(gate) * up)
                w2 = dequantize_block_weight(gemm2_weights[e], gemm2_weights_scale[e])
                # The FC2 row lands in the bf16 `gemm2_output` buffer before the
                # combine; `finalizeDeepSeekKernel` reads it back as float.
                y = (act @ w2.t()).to(torch.bfloat16).float()
                out.index_add_(0, tok, y * topk_weights[tok, slot].float().unsqueeze(1))
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
        """Four inputs the op accepts and answers wrongly, each measured.

        The transposed activation-scale buffer at `T == H/128` is the fifth,
        and no metadata check can see it.
        """
        # The kernel takes raw data pointers: a strided view is read as dense.
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
        # The Python runner overwrites the activation with SwiGlu.
        assert act_type == 0, (
            f"act_type={act_type} is silently ignored -- this op always computes "
            "SwiGLU. Use a different runner for a non-gated activation"
        )
        # Once topk_ids carry the routing, nothing reads the bias.
        if routing_bias is not None:
            assert topk_ids is None, (
                "routing_bias is ignored when topk_ids/topk_weights are given "
                "(routing has already happened); fold it into the router logits"
            )
        # `num_experts` is the routing space, not the window: a window that
        # starts at or past it holds no routable id. Passing the window's own
        # count as `num_experts` puts ranks 1..3 there: the op answers all zeros
        # with no error at small row counts and reads out of bounds at 16 rows.
        assert 0 <= local_expert_offset < num_experts, (
            f"local_expert_offset={local_expert_offset} is outside the routing space "
            f"[0, num_experts={num_experts}); num_experts is every routed expert "
            "(512 for a 128-expert window of 512), not the window's count"
        )


fp8_block_scale_moe_runner = Fp8BlockScaleMoeRunner()
