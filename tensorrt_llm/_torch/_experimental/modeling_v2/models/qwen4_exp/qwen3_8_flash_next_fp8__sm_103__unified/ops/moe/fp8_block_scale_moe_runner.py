# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-style FP8 block-scale mixture-of-experts layer: given top-k +
grouped FC1 GEMM + SwiGLU + FP8 requantization + grouped FC2 GEMM +
routing-weighted combine, in one trtllm-gen call.

This target's copy of the qwen4_exp family entry
(`models/qwen4_exp/ops/moe/fp8_block_scale_moe_runner.py`), certified at the
cells this target calls it with: the routed experts of the trunk's 48 layers
(`_routed_experts`) and of the draft layer (`_mtp_moe_block`), over each
member's whole-expert window, eager and inside a captured CUDA graph."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

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


def _geom(hidden, inter, num_experts, top_k, offset=0, local=None, capture=False):
    """A cell's MoE geometry; the window defaults to every expert local.

    `capture` drives the call inside a captured CUDA graph, replayed over two
    steps' activations and routing, rather than eagerly."""
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
    """The checkpoint's routed experts, H 2560, I 640, 512 experts, top-10, every
    expert local (tp1)."""
    return _geom(2560, 640, 512, 10, capture=capture)


_CKPT = _whole()


def _window(offset: int, local: int = 128, capture: bool = False) -> dict:
    """One rank's whole-expert window of the 512 -- 128 wide at tp4, tep4 and dep4,
    256 at tp2, tep2 and dep2: the router still ranks all 512, so `num_experts`
    stays 512."""
    return _geom(2560, 640, 512, 10, offset=offset, local=local, capture=capture)


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
        # Kept from the family entry (the qwen4 NVFP4 default target's MTP layer):
        # four 128-wide windows, one per rank -- at tp4 too, where 640 / 4 = 160 is
        # not a multiple of 128 and the intermediate cannot be split. One cell per
        # window offset, each at a row count the MTP layer reaches.
        Cell(
            why="default MTP drafter decode: one token, rank 0's 128-expert window",
            spec=dict(tokens=1, **_window(0)),
        ),
        Cell(
            why="default: 128 drafter rows, the largest captured batch; rank 1's window",
            spec=dict(tokens=128, **_window(128)),
        ),
        Cell(
            why="default: 1024 rows, the first draft over mtp7's 128 x 8 verify rows; window 2",
            spec=dict(tokens=1024, **_window(256)),
        ),
        Cell(
            why="default: 8192 rows, a max_num_tokens context chunk; the last window (384)",
            spec=dict(tokens=8192, **_window(384)),
        ),
        # This target, every member: the trunk's `_routed_experts` and the draft's
        # `_mtp_moe_block` run each rank's whole-expert window -- the 512 at tp1, 256
        # at tp2 / tep2 / dep2, 128 at tp4 / tep4 / dep4; pure TP never splits the
        # 640 intermediate (320 and 160 are not multiples of 128) -- and complete it
        # with a plain sum. Decode and MTP verify run inside the captured CUDA graph,
        # prefill eagerly in chunks of at most 8192 rows (max_num_tokens, also the
        # chunk the attention-DP gather is run in). Gathered rows are the DP group's
        # (dep2 x2, dep4 x4); the 2 x 128 and 4 x 128 cells take a per-rank batch of 128,
        # past the dep2 (64) and dep4 (32) engines' measured largest, whose gathered
        # decode tops out at 128 rows.
        Cell(
            why="tp1 eager decode, 128 rows: the whole 512-expert stack",
            spec=dict(tokens=128, **_whole()),
        ),
        Cell(
            why="tp1 captured decode, 128 rows: the whole 512-expert stack",
            spec=dict(tokens=128, **_whole(capture=True)),
        ),
        Cell(
            why="tp1 prefill chunk, 8192 rows (max_num_tokens): the whole stack",
            spec=dict(tokens=8192, **_whole()),
        ),
        Cell(
            why="tep2 captured decode, 1 row: 256-expert window 0",
            spec=dict(tokens=1, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="tp2 captured decode, 128 rows: 256-expert window 256",
            spec=dict(tokens=128, **_window(256, 256, capture=True)),
        ),
        Cell(
            why="dep2 captured decode, 2 x 128 gathered rows: 256-expert window 0",
            spec=dict(tokens=256, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="dep2 prefill, an 8192-row chunk of the gathered rows: 256-expert window 256",
            spec=dict(tokens=8192, **_window(256, 256)),
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
            why="dep4 captured decode, 4 x 128 gathered rows (and tep4 MTP3 verify, 128 x 4): "
            "128-expert window 128",
            spec=dict(tokens=512, **_window(128, capture=True)),
        ),
        Cell(
            why="dep4 captured MTP3 verify, 4 x 128 x 4 gathered rows: 128-expert window 384",
            spec=dict(tokens=2048, **_window(384, capture=True)),
        ),
        Cell(
            why="tep4 captured MTP3 verify at batch 1, 4 rows: 128-expert window 384",
            spec=dict(tokens=4, **_window(384, capture=True)),
        ),
        # The decode batches the engines captured between those (measured: tp2, tep2,
        # tp4 and tep4 capture 1, 2, 4, 8, 16..128 in steps of 8; dep2 1..64 and dep4
        # 1..32 per rank, every rank padded to the captured batch, so the window runs
        # over 2x and 4x the batch). Each is a row count whose cold fallback tactic is
        # its own (keyed on the rows) and whose tuned tactic is its power-of-two bucket's
        # (40 and 56 tune at 32, 120 at 64).
        Cell(
            why="tep2/tp2 captured decode, batch 2 (dep2: 2 x 1 gathered): 256-expert window 0",
            spec=dict(tokens=2, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="tep2/tp2 captured decode, batch 8 (dep2: 2 x 4 gathered): 256-expert window 256",
            spec=dict(tokens=8, **_window(256, 256, capture=True)),
        ),
        Cell(
            why="tep2/tp2 captured decode, batch 40: 256-expert window 0",
            spec=dict(tokens=40, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="tep2/tp2 captured decode, batch 64 (dep2: 2 x 32 gathered): 256-expert window 256",
            spec=dict(tokens=64, **_window(256, 256, capture=True)),
        ),
        Cell(
            why="tep2/tp2 captured decode, batch 120: 256-expert window 0",
            spec=dict(tokens=120, **_window(0, 256, capture=True)),
        ),
        Cell(
            why="tep4/tp4 captured decode, batch 8 (dep4: 4 x 2 gathered): 128-expert window 0",
            spec=dict(tokens=8, **_window(0, capture=True)),
        ),
        Cell(
            why="tep4/tp4 captured decode, batch 56: 128-expert window 128",
            spec=dict(tokens=56, **_window(128, capture=True)),
        ),
        Cell(
            why="tep4/tp4 captured decode, batch 120: 128-expert window 384",
            spec=dict(tokens=120, **_window(384, capture=True)),
        ),
    )

    note = """
    Metadata consumed: none. Two process-global caches sit behind the call --
    the autotuner's profiling cache and the C++ runner cache -- and a warm
    tuned tactic is bitwise equal to the cold fallback, and captures and
    replays bitwise (the test holds this at 256 tokens, for the whole stack and
    for a 256- and a 128-expert window, each tuning under its own
    `local_num_experts` key, and at every captured cell's row count, where the
    tuned tactic is its power-of-two bucket's and the fallback keyed on the
    rows themselves).

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

    This target's windows, certified on sm_103 (GB300): the whole 512 at tp1,
    256 wide at offsets 0 and 256 (tp2, tep2, dep2), 128 wide at offsets 0,
    128, 256 and 384 (tp4, tep4, dep4), at the rows each member reaches -- 1 to
    16384 over the cells, at most 8.21 bf16 ulp per element of the row max
    (16384 rows; the max over T*H samples grows with T) and 0.297 ulp RMS (a
    2-row replay; 0.14-0.30 over every cell and replay), inside
    the test's 16 / 0.75 gate. Each window cell asserts some slot routes into
    its window. The cells draw near-uniform routing (random logits, top-10 of a
    softmax: about T * 10 / 512 rows per expert); the weights are not
    renormalized, which scales each row's output by one factor and leaves the
    row-normalized gate unchanged.

    The windows' bf16 outputs added in fp32 -- what the target's all-reduce or
    reduce-scatter adds -- sit inside the gate against the 512-expert
    reference, and the sum with any one window left out fails it: four 128-wide
    at 1, 128 and 1024 rows at most 3.71 / 0.518 ulp (0.518 RMS at one row),
    two 256-wide at 1 and 256 rows at most 2.93 / 0.277. Not bitwise to the
    512-expert call (each window rounds its partial to bf16): at most 2.00 /
    0.213 ulp from it.

    CUDA-graph capture, certified at the captured cells: the whole stack at
    128 rows; 256-expert windows at 1, 2, 8, 40, 64, 120, 128 and 256 rows
    (tp2 / tep2 decode batches, dep2's 2x gathered ones); 128-expert windows at
    1, 4, 8, 56, 120, 128, 512 and 2048 rows (tp4 / tep4 decode batches and
    MTP3 verify, dep4's 4x gathered ones). The call is warmed eagerly,
    captured over private copies of the step inputs (`hidden_states`,
    `hidden_states_scale`, `topk_weights`, `topk_ids`), and the replay is
    bitwise the eager call it captured; a second step's activations and routing
    (other ids) written into those buffers replay bitwise that step's eager
    call, inside the gate (worst 7.65 / 0.297 ulp over both replays; at the
    decode batches 2..120, 4.32 / 0.297). With a warm AutoTuner cache -- the
    engine tunes at warmup, then captures -- every captured cell's tactic gives
    bitwise the cold fallback's result and replays it bitwise. Nothing
    on the call path waits on the host: the C++ runner sizes its workspaces
    from the row count and allocates them with `empty_cuda` (from the graph's
    pool under capture), and outside tuning mode `prepare_dummy_topk_and_hook`
    passes the caller's routing through. The tactic is chosen on the host at
    capture -- the AutoTuner's if its cache is warm, otherwise the fallback,
    which is keyed on the row count -- and is fixed in the graph.

    `num_experts` passed as the window's own count (128) puts windows 1..3 past
    the routing space: at small row counts the op answers all zeros with no
    error, and at 16 rows it reads out of bounds (an illegal memory access) --
    ids `>= num_experts` are dropped only at small row counts. `is_valid`
    refuses a window that starts outside `[0, num_experts)`; window 0 of that
    mistake still passes it, and ids past `num_experts` cannot be seen without
    reading the ids. A window that starts inside the routing space and runs
    past its end ([448, 576) of 512) computes its in-range experts correctly.

    Inert on the certified path, bitwise: `routing_method_type` 0/1/2/4 (the
    pre-routed path forces the routing postprocess), `n_group`/`topk_group`
    (1, 1), `routed_scaling_factor` None/1.0/2.5, `tune_max_num_tokens`,
    `use_dp`. `gemm1_clamp_limit` is live and one scalar for every expert,
    applied after the FC1 requantization; None is bitwise equal to omitting it.

    Silent on the raw op, refused by `is_valid` (the test holds each refusal,
    and the silent answer of the first and third): any non-contiguous tensor
    (read as dense; a column-sliced `hidden_states` is 84.7 ulp RMS off at 64
    rows); `act_type != 0` (the Python runner forces SwiGlu, so 1 and 2 are
    bitwise identical to 0); `routing_bias` alongside `topk_ids` (ignored; a
    +-1e3 bias leaves the output bitwise unchanged); a `local_expert_offset`
    outside `[0, num_experts)` (refused before the launch; its out-of-bounds
    read is not re-driven).
    `num_fused_shared_experts > 0` is rejected loudly on the pre-routed path,
    which is the only path the C++ runner's small-tile tactic exclusion guards.

    The registered fake returns `[T, 2H]` (it assumes packed fp4 hidden
    states): an upstream defect, eager is unaffected.

    Hard limits
      T            whole stack certified at 1, 20, 128, 8192 and 16384;
                   256-expert windows at 1, 2, 8, 40, 64, 120, 128, 256 and
                   8192; 128-expert windows at 1, 4, 8, 56, 120, 128, 512, 1024,
                   2048 and 8192. The target runs at most 8192 rows a call
                   (max_num_tokens, and the chunk its attention-DP gather is
                   run in). Chunk above them, not probed. Captured at the
                   capture cells' rows only (listed above).
                   Autotuner buckets stop at
                   `tune_max_num_tokens` (default 8192, which the target
                   leaves): past it a call runs the 8192 bucket's tactic; a
                   larger value makes the op append a bucket at that size
      top_k        <= 32 (raises past it); certified 10, the checkpoint's
      E, H, I      certified at the checkpoint's 512, 2560, 640
      window       the whole stack, 256 wide at offsets 0 and 256, and 128 wide at
                   offsets 0, 128, 256 and 384 of 512

    Not certified (measured, but not re-driven by the test): windows of other
    widths or offsets, a window past the routing space's end, repeated or
    out-of-range ids, `gemm1_clamp_limit`, the inert arguments, `output=`, the
    registered fake, the op's other loud rejections. Nor: capture at the other
    captured decode rows -- 256-expert windows at 4, 16, 24, 32, 48, 56, 72..112;
    128-expert windows at 2, 16..48, 64..112 (the 16 tuning bucket, 16..31
    rows, is captured by no cell at either width) -- nor tep4 MTP7 verify's 1024
    or other MTP depths' verify rows; skewed routing (a hot expert holding far more
    than T * 10 / 512 rows), the in-kernel routing path (`routing_logits`
    given), multi-rank execution (every window here runs on one device; the
    cross-rank sum is the collective's), geometries other than the
    checkpoint's, sm_100.
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
        # count as `num_experts` puts ranks 1..3 there: the op answers all zeros
        # with no error at small row counts and reads out of bounds at 16 rows.
        assert 0 <= local_expert_offset < num_experts, (
            f"local_expert_offset={local_expert_offset} is outside the routing space "
            f"[0, num_experts={num_experts}); num_experts is every routed expert "
            "(512 for a 128-expert window of 512), not the window's count"
        )


fp8_block_scale_moe_runner = Fp8BlockScaleMoeRunner()
