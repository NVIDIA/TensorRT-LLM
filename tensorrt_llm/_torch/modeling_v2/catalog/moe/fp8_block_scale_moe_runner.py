# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-style FP8 block-scale mixture-of-experts layer: given top-k +
grouped FC1 GEMM + SwiGLU + FP8 requantization + grouped FC2 GEMM +
routing-weighted combine, in one trtllm-gen call."""

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
#: cpp/tensorrt_llm/kernels/moe/trtllmGen/DevKernel.cu). Only an all-zero block reaches it.
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


def _tep4(offset, **kw):
    """A Mistral Large 4 routed-expert call on the tep4 rank whose window starts at `offset`.

    H 6144, I 6144, top-4 of 128 experts, 32 of them local. Operands follow the
    checkpoint and the engine: only the rank's 32 experts are drawn, their 128x128
    scales are powers of two (the checkpoint's UE8M0 qscales), the hidden states
    are quantized by the engine's 1x128 amax/448 recipe, the routing is a
    renormalized top-4 in bf16, and `tune_max_num_tokens` is max_num_tokens.
    """
    return dict(
        _geom(6144, 6144, 128, 4, offset=offset, local=32),
        weights="window",
        weight_scales="pow2",
        hidden_states="quantized",
        tune_max_num_tokens=9216,
        **kw,
    )


#: One eager sweep's token counts: every count to 256 (the routing kernel
#: changes at 5 and 13, the finalize kernel at 50, the GEMM tile at 72 and 136,
#: and at 48 == H/128 the activation scale buffer is square), each power of two
#: from 512 with both neighbours (the autotuner's buckets; the activation kernel
#: changes at 4096 and the routing kernel at 8193), and max_num_tokens 9216.
_TEP4_SWEEP = tuple(
    sorted({*range(1, 257), *(2**k + d for k in range(9, 14) for d in (-1, 0, 1)), 9215, 9216})
)


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
    all-reduce and the residual are the caller's. Under expert parallelism the
    result is this rank's partial: tokens with no slot in the window come back
    zero, and the caller sums the ranks.

    `hidden_states_scale` is `[H/128, T]` -- transposed, exactly what
    `fp8_quantize_1x128` returns on this architecture (contiguous, measured at
    T 1, 3, 5, 16, 20, 48 and 9216 for H 6144). Pass it straight through: at
    `T == H/128` (48 tokens at H 6144) its transpose has the same shape and
    passes every check, and is 16 ulp RMS wrong.

    Returns a fresh `[T, H]` bf16 tensor, or an empty `[0]` bf16 tensor when
    `output` is given (the result is then in `output`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # Mistral Large 4 tep4: every rank sees all T tokens (no attention DP) and
        # runs its 32-expert window; decode is captured at batches 1, 2, 4, 8,
        # 16 (padded) -- 1..16 unpadded under the default CudaGraphConfig --
        # and context steps are eager up to max_num_tokens 9216, unchunked.
        Cell(
            why="tep4 decode graphs at 1..16 tokens, rank 0 window (0, 32): "
            "each captured, then replayed on new tokens and new routing",
            spec=_tep4(0, graph=tuple(range(1, 17))),
        ),
        Cell(
            why="tep4 eager token counts to max_num_tokens 9216, rank 3 window (96, 32): "
            "every routing, tile, activation and finalize kernel edge",
            spec=_tep4(96, sweep=_TEP4_SWEEP),
        ),
        Cell(
            why="tep4 9216 tokens, rank 1 window (32, 32): the cooperative routing kernel",
            spec=_tep4(32, tokens=9216),
        ),
        Cell(
            why="tep4 9216 tokens, rank 2 window (64, 32)",
            spec=_tep4(64, tokens=9216),
        ),
        Cell(
            why="tep4 local experts 29..31 on every token: gemm1 rows past 2^31 bytes",
            spec=_tep4(0, sweep=(1, 16, 9216), far=True),
        ),
    )

    note = """
    Metadata consumed: none. Two process-global caches sit behind the call --
    the autotuner's profiling cache and the C++ runner cache. The tactic does
    not change the result: at the tep4 geometry every valid tactic (48 at 16
    tokens, 20 at 9216) is bitwise equal to the cold fallback, so the cells
    certify the tuned path an engine warms up as well as the cold one they run.

    Preconditions. `hidden_states` is `[T, H]` e4m3 and `hidden_states_scale`
    `[H/128, T]` fp32 (see the class docstring); weights are the checkpoint's
    per-expert tensors in plain row-major layout, no shuffle and no swizzle:
    `gemm1_weights` `[E, 2I, H]` e4m3 rows `[up ; gate]`, `gemm1_weights_scale`
    `[E, 2I/128, H/128]` fp32, `gemm2_weights` `[E, H, I]`, `gemm2_weights_scale`
    `[E, H/128, I/128]`; `topk_ids` `[T, top_k]` int32 and `topk_weights`
    `[T, top_k]` bf16, used as given (nothing is renormalized). H and I are
    multiples of 128, `intermediate_size == gemm1_weights.shape[1] // 2`,
    `0 < top_k < num_experts`, `top_k <= 32`. The op raises on each of those.
    Supply distinct ids per row: how often a repeated id counts depends on the
    routing kernel the token count picks, and is not certified (the test
    records the reading at 16 and 256 tokens and asserts neither).

    The window: local slot i answers for global id `local_expert_offset + i`.
    Nothing checks the window against `num_experts`; ids outside it (negative,
    `>= num_experts`) are dropped, a token with no slot in it comes back
    bitwise zero, and a window starting at `num_experts` returns all zeros.

    Kernels the tep4 cells drove, by token count (H 6144, I 6144, top-4 of 128,
    32 local, cold fallback tactic): routing -- block kernel to 4 tokens,
    dyn-block 5..12, cluster 13..8192, expert-count init plus the cooperative
    kernel 8193..9216; both GEMMs -- tile 128x32 to 71 tokens, 128x64 for
    72..135, 128x128 from 136; activation -- the expanded-space kernel below
    4096 tokens, the permuted-space wide kernel from 4096 (the switch is
    `T * top_k / num_experts >= tile`, so a tuned tactic with a narrower tile
    switches earlier); finalize -- scalar to 49 tokens, vector loads from 50.
    Rows are batch invariant: every m-row call for m in 1..9216 was measured
    bitwise equal to the first m rows of one 9216-row call, across all of those
    switches; the sweep cell re-proves it at every switch. A CUDA graph captured
    at 1..16 tokens replays bitwise equal to an eager call on new tokens and
    new routing (the routing kernel runs inside the graph). The cells run with
    `TRTLLM_ENABLE_PDL` unset, which enables PDL; 0 and 1 were measured bitwise
    equal to it at 1, 16, 4096 and 9216 tokens.

    Inert on the pre-routed path, bitwise -- measured, but not certified here:
    `routing_method_type` 0/1/2/4 (the pre-routed path forces the routing
    postprocess; measured at H 256, 16 tokens -- the tep4 cells pass 1),
    `n_group`/`topk_group` (1, 1), `routed_scaling_factor` None/1.0/2.5,
    `tune_max_num_tokens`, `use_dp`. `tune_max_num_tokens` only adds a
    profiling bucket on the warm path; pass max_num_tokens as the engine does
    (the tep4 cells pass 9216). At tep4, warming the autotuner at 9216 tokens
    took 6.0 s and wrote 15 cache entries. `gemm1_clamp_limit` is accepted by
    the op but not certified here; `reference` applies it as one scalar for
    every expert, after the FC1 requantization.

    Silent on the raw op, refused by `is_valid`: any non-contiguous tensor (read
    as dense, 55-115 ulp RMS off); `act_type != 0` (the Python runner forces
    SwiGlu, so 1 and 2 are bitwise identical to 0); `routing_bias` alongside
    `topk_ids` (ignored; a +-1e3 bias leaves the output bitwise unchanged).
    `num_fused_shared_experts > 0` is rejected loudly on the pre-routed path,
    which is the only path the C++ runner's small-tile tactic exclusion guards.

    The registered fake returns `[T, 2H]` (it assumes packed fp4 hidden
    states): an upstream defect, eager is unaffected.

    Hard limits (sm_103)
      gemm1_weights  local expert e starts at byte e * 2I * H; at tep4 (I = H
                     = 6144, 32 local) the tensor is 2,415,919,104 bytes and
                     experts 29..31 lie past 2^31 (28 straddles it)
                     probed: each of 29, 30, 31 gives the same bits moved to
                     slot 1 -- passed
      gemm1 output   permuted rows * 2I bytes, rows <= T * top_k + padding;
                     2^31 at ~43,690 tokens for top-4, 2I 12288 -- not reached
                     (9216), unprobed
      the rest       at tep4 and 9216 tokens: hidden_states 56,623,104
                     elements, gemm2_weights 1,207,959,552 bytes, T * top_k
                     36,864 expanded slots -- all below 2^31; no index passes
                     through fp32
      T              certified 1..9216 at tep4 (the cooperative routing kernel
                     from 8193); chunk above, not probed
      top_k          <= 32 (raises past it); certified 4
      E, H, I        certified at 128 (32 local), 6144, 6144

    Not certified: the in-kernel routing path (`routing_logits` given);
    multi-rank execution (each window ran in one process; the tep4 rank
    partials were not summed against a 128-expert call); more than 9216
    tokens at tep4; `output=` at the tep4 geometry (driven at H 256 only);
    `gemm1_clamp_limit`; a window overrunning `num_experts` and ids outside
    `[0, num_experts)` (both accepted by the op); sm_100.
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
        """Three inputs the op accepts and answers wrongly, each measured.

        The transposed activation-scale buffer at `T == H/128` is the fourth,
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


fp8_block_scale_moe_runner = Fp8BlockScaleMoeRunner()
