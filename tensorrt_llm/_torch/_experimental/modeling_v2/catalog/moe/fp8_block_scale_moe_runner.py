# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-style FP8 block-scale mixture-of-experts layer: given top-k +
grouped FC1 GEMM + SwiGLU + FP8 requantization + grouped FC2 GEMM +
routing-weighted combine, in one trtllm-gen call.

The qwen4 unified target's variant (r40) of the qwen4_exp tp1 target's
`ops/moe/fp8_block_scale_moe_runner`, copied from it."""

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
#: trtllmGenKernels/blockScaleMoe/DevKernel.cu). Only an all-zero block reaches it.
ACT_AMAX_FLOOR = 1.0e-10


def dequantize_block_weight(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """One expert's `[M, K]` fp32 weight from e4m3 codes and `[M/128, K/128]` scales."""
    m, k = codes.shape
    blocks = codes.float().reshape(m // BLOCK, BLOCK, k // BLOCK, BLOCK)
    return (blocks * scales.float().reshape(m // BLOCK, 1, k // BLOCK, 1)).reshape(m, k)


def requantize_1x128(x: torch.Tensor) -> torch.Tensor:
    """The e4m3 round trip the kernel applies to both intermediates.

    Per row and per 128 consecutive columns: `scale = max(amax, 1e-10) / 448`,
    then round-to-nearest-even into e4m3 saturating at +-448. The plain
    `amax / 448` scale, not the round-up power of two the MX members use.
    """
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR) / E4M3_MAX
    codes = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(shape)


def _geom(hidden, inter, num_experts, top_k, offset=0, local=None):
    """A cell's MoE geometry; the window defaults to every expert local."""
    return dict(
        hidden=hidden,
        inter=inter,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=offset,
        local_num_experts=num_experts if local is None else local,
    )


#: The MTP draft layer's routed experts: H 2560, I 640, 512 experts, top-10,
#: every expert local (tp1).
_CKPT = _geom(2560, 640, 512, 10)


def _window(offset: int) -> dict:
    """One rank's 128-expert window of the 512 (the qwen4 unified target at tp4,
    tep4 and dep4): the router still ranks all 512, so `num_experts` stays 512."""
    return _geom(2560, 640, 512, 10, offset=offset, local=128)


class Fp8BlockScaleMoeRunner(OpWrapper):
    """One FP8 block-scale MoE layer over e4m3 activations, entered pre-routed.

    Per token `t` and slot `j`, with `A[t]` the fp32 activations
    (`hidden_states[t, k] * hidden_states_scale[k // 128, t]`) and `W_*` the
    dequantized 128x128-block weights of local expert `e = topk_ids[t, j] -
    local_expert_offset` (slots outside `[0, local_num_experts)` contribute
    nothing):

        up, gate = requant(A[t] @ [W_up; W_gate].T).split(I)
        gate, up = min(gate, lim), clamp(up, -lim, lim)    # gemm1_clamp_limit
        out[t]  += topk_weights[t, j] * (requant(silu(gate) * up) @ W_down.T)

    accumulated in fp32 and stored bf16, `requant` being `requantize_1x128`.
    FC1 rows are `[up ; gate]` -- up first, the engine loader's `[w3; w1]`.

    Fusion boundary: permutation, both GEMMs with on-the-fly dequantization,
    both intermediate requantizations, the clamped SwiGLU and the weighted
    combine are inside. The router GEMM and top-k, quantizing the bf16 hidden
    states (`fp8_quantize_1x128(x, use_ue8m0=False)`), the shared expert, any
    all-reduce and the residual are the caller's.

    `hidden_states_scale` is `[H/128, T]` -- transposed, exactly what
    `fp8_quantize_1x128` returns on this architecture. Pass it straight through:
    at `T == H/128` (20 tokens at H 2560) its transpose has the same shape and
    passes every check, and is 41-43 ulp RMS wrong.

    Returns a fresh `[T, H]` bf16 tensor, or an empty `[0]` bf16 tensor when
    `output` is given (the result is then in `output`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # The token column at the checkpoint geometry, the only one the draft layer
        # calls. The cold-cache fallback tactic is keyed on the token count, and rows
        # per active expert (T * 10 / 512) is what moves the kernel's tile choice.
        Cell(
            why="MTPLayer._moe decode: one token, 10 of 512 experts", spec=dict(tokens=1, **_CKPT)
        ),
        Cell(
            why="20 tokens == H/128: the activation scale buffer is square",
            spec=dict(tokens=20, **_CKPT),
        ),
        Cell(
            why="16384 rows: max_num_tokens of the tuned config; past the last tuning bucket",
            spec=dict(tokens=16384, **_CKPT),
        ),
        # The qwen4 unified target: every member runs the MTP experts as four
        # 128-wide windows, one per rank -- at tp4 too, where 640 / 4 = 160 is not
        # a multiple of 128 and the intermediate cannot be split. One cell per
        # window offset, each at a row count the MTP layer reaches.
        Cell(
            why="unified MTP drafter decode: one token, rank 0's 128-expert window",
            spec=dict(tokens=1, **_window(0)),
        ),
        Cell(
            why="unified: 128 drafter rows, the largest captured batch; rank 1's window",
            spec=dict(tokens=128, **_window(128)),
        ),
        Cell(
            why="unified: 1024 rows, the first draft over mtp7's 128 x 8 verify rows; window 2",
            spec=dict(tokens=1024, **_window(256)),
        ),
        Cell(
            why="unified: 8192 rows, a max_num_tokens context chunk; the last window (384)",
            spec=dict(tokens=8192, **_window(384)),
        ),
    )

    note = """
    Metadata consumed: none. Two process-global caches sit behind the call --
    the autotuner's profiling cache and the C++ runner cache -- and a warm
    tuned tactic is bitwise equal to the cold fallback (the test holds this at
    256 tokens, for the whole stack and for a 128-expert window, which tunes
    under its own `local_num_experts` key).

    The op is a Python `torch.library.custom_op`
    (`trtllm::fp8_block_scale_moe_runner` in
    `_torch/custom_ops/trtllm_gen_custom_ops.py`) over the compiled-in C++ class
    `torch.classes.trtllm.FP8BlockScaleMoERunner`
    (`cpp/tensorrt_llm/thop/moe/trtllmGen/fp8BlockScaleMoe.cpp`), which launches
    trtllm-gen's precompiled batched-GEMM cubins: nothing is JIT-compiled.

    Preconditions. `hidden_states` is `[T, H]` e4m3 and `hidden_states_scale`
    `[H/128, T]` fp32 (see the class docstring); weights are the checkpoint's
    per-expert tensors in plain row-major layout, no shuffle and no swizzle:
    `gemm1_weights` `[E, 2I, H]` e4m3 rows `[up ; gate]`, `gemm1_weights_scale`
    `[E, 2I/128, H/128]` fp32, `gemm2_weights` `[E, H, I]`, `gemm2_weights_scale`
    `[E, H/128, I/128]`; `topk_ids` `[T, top_k]` int32 and `topk_weights`
    `[T, top_k]` bf16, used as given (nothing is renormalized). H and I are
    multiples of 128, `intermediate_size == gemm1_weights.shape[1] // 2`,
    `0 < top_k < num_experts`, `top_k <= 32`. The op raises on each of those.
    Supply distinct ids per row: a repeated id counted once at 16 tokens and
    twice at 256 (the token count picks the routing kernel).

    The window: local slot i answers for global id `local_expert_offset + i`,
    and `num_experts` stays the routing space (512) on every window call. The
    weights and all four scale tensors hold the window's experts only: the op
    compares each one's dim 0 with `local_num_experts` and raises on the whole
    stack (the test holds all four). A token with no slot in the window comes
    back bitwise zero (every window cell holds this).

    The qwen4 unified target's four 128-wide windows of 512 are certified at the
    row counts its MTP layer reaches, one window offset per cell, on sm_103: 1
    to 8192 rows, at most 7.95 bf16 ulp per element of the row max and 0.227
    ulp RMS, inside the test's 16 / 0.75 gate. Each cell asserts some slot routes
    into its window (2 of 10 at one token, 20345 of 81920 at 8192 rows). The four
    windows' bf16 outputs added in fp32 -- what the target's all-reduce or
    reduce-scatter adds -- sit at most 4.41 / 0.297 ulp from the 512-expert
    reference at 1, 128 and 1024 rows, and the sum with any one window left out
    fails the gate. Not bitwise to the 512-expert call (each window rounds its
    partial to bf16): at most 1.97 / 0.187 ulp from it.

    `num_experts` passed as the window's own count (128) puts windows 1..3 past
    the routing space: at 1, 4 and 5 rows the op answers all zeros with no
    error, and at 16 rows it reads out of bounds (an illegal memory access) --
    ids `>= num_experts` are dropped only at small row counts. `is_valid`
    refuses a window that starts outside `[0, num_experts)`; window 0 of that
    mistake still passes it, and ids past `num_experts` cannot be seen without
    reading the ids. A window that starts inside the routing space and runs
    past its end ([448, 576) of 512) computes its in-range experts correctly
    (measured at 1 to 1024 rows).

    Inert on the certified path, bitwise: `routing_method_type` 0/1/2/4 (the
    pre-routed path forces the routing postprocess), `n_group`/`topk_group`
    (1, 1), `routed_scaling_factor` None/1.0/2.5, `tune_max_num_tokens`,
    `use_dp`. `gemm1_clamp_limit` is live and one scalar for every expert,
    applied after the FC1 requantization; None is bitwise equal to omitting it.

    Silent on the raw op, refused by `is_valid` (the test holds the strided
    `hidden_states` and the outside-the-routing-space refusals): any
    non-contiguous tensor (read as dense, 55-115 ulp RMS off); `act_type != 0`
    (the Python runner forces SwiGlu, so 1 and 2 are bitwise identical to 0);
    `routing_bias` alongside `topk_ids` (ignored; a +-1e3 bias leaves the output
    bitwise unchanged); a `local_expert_offset` outside `[0, num_experts)`.
    `num_fused_shared_experts > 0` is rejected loudly on the pre-routed path,
    which is the only path the C++ runner's small-tile tactic exclusion guards.

    The registered fake returns `[T, 2H]` (it assumes packed fp4 hidden
    states): an upstream defect, eager is unaffected.

    Hard limits
      T            whole stack certified at 1, 20 and 16384 (max_num_tokens of the
                   tuned config); 128-expert windows at 1, 128, 1024 and 8192
                   (the unified target's max_num_tokens chunk). Chunk above
                   them, not probed. Autotuner buckets stop at
                   `tune_max_num_tokens` (default 8192, which the target
                   leaves): past it a call runs the 8192 bucket's tactic; a
                   larger value makes the op append a bucket at that size
      top_k        <= 32 (raises past it); certified 10, the checkpoint's
      E, H, I      certified at the checkpoint's 512, 2560, 640
      window       the whole stack, and 128 wide at offsets 0, 128, 256 and 384 of 512

    Not certified (the statements above were measured at bring-up and are not
    re-driven by the test): windows of other widths (a 256-wide dep2 window,
    say) or offsets, a window past the routing space's end, repeated or
    out-of-range ids, `gemm1_clamp_limit`, the inert arguments, `output=`, the
    registered fake, the op's other loud rejections. Nor: CUDA-graph capture
    of the call (the cells run eager), the in-kernel routing path
    (`routing_logits` given), multi-rank execution (every window here runs on
    one device; the cross-rank sum is the collective's), geometries other than
    the checkpoint's, sm_100.
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
        """The layer over bit-identical operands, fp32 GEMMs, bf16 store.

        Pre-routed only: `routing_logits` must be None. Ignored: `routing_bias`,
        `num_experts`, `top_k`, `num_fused_shared_experts`, `n_group`,
        `topk_group`, `routed_scaling_factor`, `routing_method_type`,
        `act_type`, `output`, `tune_max_num_tokens`, `use_dp` -- each inert on
        that path. Returns the `[T, H]` result even when `output` is given.
        """
        assert routing_logits is None and topk_ids is not None and topk_weights is not None, (
            "the reference models the pre-routed path only"
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
                y = act @ w2.t()
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
        # count as `num_experts` puts ranks 1..3 there, and the op answers all
        # zeros with no error (measured at 1, 4 and 5 rows; at 16 rows the ids
        # past the routing space then read out of bounds).
        assert 0 <= local_expert_offset < num_experts, (
            f"local_expert_offset={local_expert_offset} is outside the routing space "
            f"[0, num_experts={num_experts}); num_experts is every routed expert "
            "(512 for a 128-expert window of 512), not the window's count"
        )


fp8_block_scale_moe_runner = Fp8BlockScaleMoeRunner()
