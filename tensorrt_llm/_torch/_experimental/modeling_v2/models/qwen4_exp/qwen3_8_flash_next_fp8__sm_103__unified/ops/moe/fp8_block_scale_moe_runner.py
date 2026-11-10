# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-style FP8 block-scale mixture-of-experts layer: top-k routed grouped FC1 GEMM +
SwiGLU + FP8 requantization + grouped FC2 GEMM + routing-weighted combine, in one trtllm-gen call.

This target's copy of the family entry `models/qwen4_exp/ops/moe/fp8_block_scale_moe_runner.py`,
with the logits-routed decode front end (`fp8_block_scale_moe_front`) and pinned decode tactics."""

from __future__ import annotations

from typing import Callable, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP8BlockScaleMoERunner

from ... import native_ops

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
    """The e4m3 round trip the kernel applies to both intermediates: per row and 128 columns,
    `scale = max(amax, 1e-10) / 448` (not a power of two), round-to-nearest-even saturating at 448.
    """
    shape = x.shape
    blocks = x.reshape(*shape[:-1], shape[-1] // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=ACT_AMAX_FLOOR) / E4M3_MAX
    codes = (blocks / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * scale).reshape(shape)


def _geom(hidden, inter, num_experts, top_k, offset=0, local=None, capture=False):
    """A cell's MoE geometry (every expert local by default; `capture`: a captured graph)."""
    return dict(
        hidden=hidden,
        inter=inter,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=offset,
        local_num_experts=num_experts if local is None else local,
        capture=capture,
    )


def _whole(capture: bool = False) -> dict:
    """The checkpoint's routed experts (H 2560, I 640, 512, top-10), all local (tp1)."""
    return _geom(2560, 640, 512, 10, capture=capture)


_CKPT = _whole()


def _window(offset: int, local: int = 128, capture: bool = False) -> dict:
    """One rank's window of the 512 (128 or 256 wide); `num_experts` stays 512."""
    return _geom(2560, 640, 512, 10, offset=offset, local=local, capture=capture)


_CUSTOM_OP = "trtllm::fp8_block_scale_moe_runner"

#: The geometry `DECODE_TACTICS` was measured on: (num_experts, top_k, intermediate, routing).
DECODE_TACTICS_GEOMETRY = (512, 10, 640, 1)

#: trtllm-gen tactic `(tileN, config index)` per (window width, rows) for logits-routed decode
#: calls (`routing_logits=`, Renormalize). The engine's tuning pass profiles pre-routed shapes
#: only, so the AutoTuner holds nothing for these and would hand them the slower cold fallback
#: `[32, 0]`. Every tactic gives the fallback's bits: the choice moves time only. A call not
#: listed takes the AutoTuner's tactic for the same call pre-routed (`_dispatch`), else the fallback.
DECODE_TACTICS: dict[tuple[int, int], tuple[int, int]] = {
    (128, 1): (8, 13),
    (128, 2): (8, 13),
    (128, 3): (8, 13),
    (128, 4): (8, 13),
    (128, 6): (8, 12),
    (128, 8): (8, 12),
}


#: The tactic logits-routed decode calls over a 256-expert window (tp2 / tep2) run from
#: `WIDE_DECODE_MIN_ROWS` rows on. The AutoTuner profiles near-uniform routing, where tileN 8 wins;
#: this checkpoint's skewed routing makes hot experts restream weight tiles at tileN 8, so tileN 32
#: is faster from 168 rows (GB300, recorded routing). Every tactic writes the same bits. Config 8
#: pairs the persistent-scheduler FC1 (`u2`) and FC2 cubins.
WIDE_DECODE_TACTIC: tuple[int, int] = (32, 8)
WIDE_DECODE_MIN_ROWS = 168
WIDE_DECODE_WIDTH = 256

#: The same calls below `WIDE_DECODE_MIN_ROWS`, by the row count each tactic starts at: what a
#: full-size tuning call picks. Pinned because a small first tuning call profiles every bucket on
#: a few experts (the tuner repeats its rows' `topk_ids`) and picks tiles slower under real routing.
WIDE_DECODE_SMALL_TACTICS: tuple[tuple[int, tuple[int, int]], ...] = (
    (1, (8, 13)),
    (4, (8, 12)),
    (8, (8, 8)),
)


#: `(tactic, min_rows)` of the 256-wide decode calls; None (tests set it) leaves them to the AutoTuner.
_WIDE_DECODE: Optional[tuple[tuple[int, int], int]] = (WIDE_DECODE_TACTIC, WIDE_DECODE_MIN_ROWS)


def _pinned_tactic(geometry: tuple, inputs: list, runner) -> Optional[list]:
    """`DECODE_TACTICS`'s tactic for a logits-routed call at its geometry, if listed, or for a
    256-wide window `WIDE_DECODE_TACTIC` from `WIDE_DECODE_MIN_ROWS` rows on and
    `WIDE_DECODE_SMALL_TACTICS` below -- either only if valid at the call's rows; else None."""
    experts, top_k, inter, _, local, routing = geometry[:6]
    if inputs[0] is None or inputs[9] is not None:
        return None
    if (experts, top_k, inter, routing) != DECODE_TACTICS_GEOMETRY:
        return None
    rows = inputs[2].shape[0]
    tactic = DECODE_TACTICS.get((local, rows))
    if tactic is None and _WIDE_DECODE is not None and local == WIDE_DECODE_WIDTH:
        if rows >= _WIDE_DECODE[1]:
            tactic = _WIDE_DECODE[0]
        else:
            tactic = [t for start, t in WIDE_DECODE_SMALL_TACTICS if rows >= start][-1]
    if tactic is None:
        return None
    hidden = inputs[2].shape[1]
    valid = runner.get_runner().get_valid_configs(top_k, None, hidden, inter, local, rows)
    if list(tactic) not in [[int(v) for v in t] for t in valid]:
        return None
    return list(tactic)


# The custom op's AutoTuner choice, keyed on everything its lookup reads (geometry, input shapes,
# the profiling cache's size: a lookup made before the tuning pass is not reused after it).
_DISPATCH: dict = {}


def _dispatch(geometry: tuple, inputs: list) -> Optional[tuple]:
    """`(runner, tactic)` as `torch.ops.trtllm.fp8_block_scale_moe_runner` would choose them,
    remembered; None while the AutoTuner tunes or captures (the custom op then runs as is)."""
    tuner = AutoTuner.get()
    if tuner.is_tuning_mode or tuner.active_capture is not None:
        return None
    shapes = tuple(None if t is None else tuple(t.shape) for t in inputs)
    key = (geometry, shapes, len(tuner.profiling_cache))
    hit = _DISPATCH.get(key)
    if hit is not None:
        return hit
    experts, top_k, inter, offset, local, routing, act, clamp, tune_max, use_dp = geometry
    runner = FP8BlockScaleMoERunner(
        experts,
        top_k,
        None,
        None,
        None,
        inter,
        offset,
        local,
        None,
        routing,
        act,
        tune_max_num_tokens=tune_max,
        use_dp=use_dp,
        gemm1_clamp_limit_value=clamp,
    )
    pinned = _pinned_tactic(geometry, inputs, runner)
    if pinned is not None:
        _DISPATCH[key] = (runner, pinned)
        return runner, pinned
    lookup = inputs
    if inputs[0] is not None and inputs[9] is None:
        # Routed from logits, which the tuning pass never profiles: use the tactic held for the
        # same call pre-routed (the GEMMs' shapes are the same).
        rows, device = inputs[2].shape[0], inputs[2].device
        lookup = list(inputs)
        lookup[0] = None
        lookup[8] = torch.empty((rows, top_k), dtype=torch.bfloat16, device=device)
        lookup[9] = torch.empty((rows, top_k), dtype=torch.int32, device=device)
    runner, tactic = tuner.choose_one(_CUSTOM_OP, [runner], runner.tuning_config, lookup)
    if tactic == -1:
        tactic = runner.get_fallback_tactic(inputs[2].shape[1], inputs[2].shape[0])
    _DISPATCH[key] = (runner, tactic)
    return runner, tactic


def fp8_block_scale_moe_front(
    routing_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routing_method_type: int,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
) -> Optional[Callable[[Optional[torch.Tensor]], torch.Tensor]]:
    """The logits-routed runner call through this target's one-launch decode routing
    (`native_ops.fp8_block_scale_moe_front`, Renormalize bitwise the runner's) at the tactic the
    custom op would choose. Returns a callable taking the bf16 hidden states (then also quantized
    into `hidden_states` / `hidden_states_scale`, bitwise `fp8_quantize_1x128`) or None; returns
    None while the AutoTuner tunes or captures (the caller makes the plain call)."""
    inputs = [
        routing_logits,
        None,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        None,
        None,
    ]
    geometry = (
        num_experts,
        top_k,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routing_method_type,
        0,
        None,
        tune_max_num_tokens,
        use_dp,
    )
    chosen = _dispatch(geometry, inputs)
    if chosen is None:
        return None
    runner, tactic = chosen
    tactic = [int(v) for v in tactic]
    if len(tactic) != 2 or tactic[0] <= 0 or tactic[1] < 0:
        return None

    def call(hidden_states_bf16: Optional[torch.Tensor]) -> torch.Tensor:
        return native_ops.fp8_block_scale_moe_front(
            routing_logits,
            hidden_states_bf16,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            num_experts,
            top_k,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routing_method_type,
            tactic[0],
            tactic[1],
        )

    return call


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
    FC1 rows are `[up ; gate]` (the loader's `[w3; w1]`). The router and top-k, the input
    quantization (`fp8_quantize_1x128(x, use_ue8m0=False)`), the shared expert and any
    collective are the caller's.

    `hidden_states_scale` is `[H/128, T]`, as `fp8_quantize_1x128` returns it: at `T == H/128`
    (20 tokens) its transpose has the same shape, passes every check and is wrong.

    Returns a fresh `[T, H]` bf16 tensor, or an empty `[0]` tensor when `output` is given (the
    result is then in `output`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # Each window width (512 / 256 / 128) at its smallest and largest rows, first and last
        # offsets, captured where the engine captures, and one non-power-of-two decode batch.
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
        Cell(
            why="tp1 captured decode, 128 rows: the whole 512-expert stack",
            spec=dict(tokens=128, **_whole(capture=True)),
        ),
        Cell(
            why="tep2 captured decode, 1 row: 256-expert window 0",
            spec=dict(tokens=1, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="dep2 captured decode, 2 x 128 gathered rows: 256-expert window 0",
            spec=dict(tokens=256, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="tep2 prefill, a 16384-row chunk (max_num_tokens 16384): 256-expert window 256",
            spec=dict(tokens=16384, **_window(256, 256)),
        ),
        Cell(
            why="tep4 captured decode, 1 row: 128-expert window 256",
            spec=dict(tokens=1, **_window(256, capture=True)),
        ),
        Cell(
            why="tp4 captured decode, 128 rows: 128-expert window 0",
            spec=dict(tokens=128, **_window(0, capture=True)),
        ),
        Cell(
            why="dep4 captured MTP3 verify, 4 x 128 x 4 gathered rows: 128-expert window 384",
            spec=dict(tokens=2048, **_window(384, capture=True)),
        ),
        Cell(
            why="tep2/tp2 captured decode, batch 40: 256-expert window 0",
            spec=dict(tokens=40, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="tep4/tp4 captured decode, batch 56: 128-expert window 128",
            spec=dict(tokens=56, **_window(128, capture=True)),
        ),
    )

    note = """
    Source: the custom op `trtllm::fp8_block_scale_moe_runner`
    (`_torch/custom_ops/trtllm_gen_custom_ops.py`) over `torch.classes.trtllm.FP8BlockScaleMoERunner`
    (`cpp/tensorrt_llm/thop/moe/trtllmGen/fp8BlockScaleMoe.cpp`): trtllm-gen's precompiled
    batched-GEMM cubins, nothing JIT-compiled. A warm tuned tactic is bitwise the cold fallback;
    the tactic is chosen on the host at capture and fixed in the graph, and the runner's
    workspaces come from the graph pool, so a replay is bitwise the eager call.

    Preconditions: `hidden_states` `[T, H]` e4m3, `hidden_states_scale` `[H/128, T]` fp32 (see
    the class docstring); plain row-major weights, no shuffle or swizzle: `gemm1_weights`
    `[E, 2I, H]` e4m3 rows `[up ; gate]`, `gemm2_weights` `[E, H, I]`, scales `[E, 2I/128, H/128]`
    and `[E, H/128, I/128]` fp32; `topk_ids` `[T, top_k]` int32 and `topk_weights` `[T, top_k]`
    bf16, used as given. H and I multiples of 128, `top_k <= 32` (the op raises otherwise). Ids
    must be distinct per row: a repeated id counts once or twice by the routing kernel the row
    count picks.

    Windows: local slot i is global id `local_expert_offset + i`; `num_experts` stays the routing
    space (512) and the weights hold the window's experts only; a token with no slot in the
    window gets zero. The window's own count passed as `num_experts` silently zeroes or reads out
    of bounds; `is_valid` refuses a window starting outside `[0, num_experts)`.
    `gemm1_clamp_limit` is live; `routing_method_type`, `n_group` / `topk_group`,
    `routed_scaling_factor`, `tune_max_num_tokens` and `use_dp` are inert on the pre-routed path.
    The registered fake returns `[T, 2H]` (upstream defect; eager is unaffected).

    Certified on GB300 with near-uniform random routing at the checkpoint's E 512, H 2560, I 640,
    top-10, within the test's 16 / 0.75 bf16-ulp gate: the whole stack, 256-wide windows at
    offsets 0 / 256 and 128-wide at 0 / 128 / 256 / 384, at the cells' rows (1 to 16384, captured
    where the cells say). Not certified: other windows or geometries, rows past 16384 (the target
    chunks at 8192 under attention DP and 16384 under TP), captured rows other than the cells',
    repeated or out-of-range ids, `gemm1_clamp_limit`, `output=`, skewed routing beyond the
    256-wide logits-routed calls the test holds, multi-rank execution, sm_100.
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
        # Outside a tuning pass, a pre-routed (or plain logits-routed) unfused call enters the
        # runner directly at the custom op's own choice (`_dispatch`), skipping its dispatch and
        # AutoTuner lookup (~0.1 ms of host time a call).
        if (
            (routing_logits is None) != (topk_ids is None)
            and routing_bias is None
            and num_fused_shared_experts is None
            and n_group is None
            and topk_group is None
            and routed_scaling_factor is None
        ):
            inputs = [
                routing_logits,
                None,
                hidden_states,
                hidden_states_scale,
                gemm1_weights,
                gemm1_weights_scale,
                gemm2_weights,
                gemm2_weights_scale,
                topk_weights,
                topk_ids,
            ]
            geometry = (
                num_experts,
                top_k,
                intermediate_size,
                local_expert_offset,
                local_num_experts,
                routing_method_type,
                act_type,
                gemm1_clamp_limit,
                tune_max_num_tokens,
                use_dp,
            )
            chosen = _dispatch(geometry, inputs)
            if chosen is not None:
                runner, tactic = chosen
                result = runner.forward(inputs, tactic=tactic, output=output)
                if output is not None:
                    return torch.empty(0, device=result.device, dtype=result.dtype)
                return result
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
        """The layer over bit-identical operands, fp32 GEMMs, bf16 store. Pre-routed only
        (`routing_logits` None); the arguments inert on that path are ignored, and the `[T, H]`
        result is returned even when `output` is given."""
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
        """Inputs the op accepts and answers wrongly."""
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
        # `num_experts` is the routing space, not the window: a window starting at or past it
        # holds no routable id (silent zeros, or an out-of-bounds read at 16 rows).
        assert 0 <= local_expert_offset < num_experts, (
            f"local_expert_offset={local_expert_offset} is outside the routing space "
            f"[0, num_experts={num_experts}); num_experts is every routed expert "
            "(512 for a 128-expert window of 512), not the window's count"
        )


fp8_block_scale_moe_runner = Fp8BlockScaleMoeRunner()
