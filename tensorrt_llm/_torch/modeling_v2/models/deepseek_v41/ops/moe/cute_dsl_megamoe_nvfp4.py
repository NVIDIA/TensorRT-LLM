# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE: expert dispatch over NVLink, grouped FC1, clamped SwiGLU, NVFP4 requantization, grouped
FC2 and the weighted combine back to the source rank, in one fused CuTe-DSL launch (optionally with
an FP8 shared expert as extra tiles)."""

from __future__ import annotations

import math
from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_COMBINE_FORMATS = ("bf16", "32e4m3xe8m0", "16e2m1xbf16")


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)


def _e2m1_rne(a: torch.Tensor) -> torch.Tensor:
    step = torch.where(a.abs() < 2.0, 0.5, torch.where(a.abs() < 4.0, 1.0, 2.0))
    return torch.sign(a) * torch.clamp(torch.round(a.abs() / step) * step, max=6.0)


def _nvfp4_roundtrip(a: torch.Tensor, g: float) -> torch.Tensor:
    """NVFP4 quantize + dequantize under global scale `g` (16-element e4m3 block scales)."""
    rows, k = a.shape
    b = a.reshape(rows, k // 16, 16)
    sf = (g * b.abs().amax(-1, keepdim=True) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    inv = torch.where(sf == 0, torch.zeros_like(sf), g / sf)
    return (_e2m1_rne(b * inv) * sf / g).reshape(rows, k)


def _mxfp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    rows, k = x.shape
    b = x.float().view(rows, k // 32, 32)
    bits = (b.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    e8 = torch.clamp((bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32), max=254)
    scale = torch.exp2(e8.float() - 127.0)[..., None]
    return ((b / scale).to(torch.float8_e4m3fn).float() * scale).view(rows, k)


_ROWS = [64, 17, 0, 1]  # rows per rank: a busy rank, a short one, an idle one, a single row


class CuteDslMegaMoeNvfp4(OpWrapper):
    """One MegaMoE layer on this rank's `num_tokens` rows; returns `[num_tokens, hidden]` bf16.

    NVFP4 routed experts over an expert-parallel group of `W` ranks (pure DEP: torch `WORLD` == EP) in one
    fused CuTe-DSL launch: NVLink dispatch of each rank's NVFP4 token rows to the rank holding each picked
    **slot**, grouped FC1, clamped SwiGLU, per-slot NVFP4 requantization, grouped FC2, weighted combine back to
    the source rank. Optionally an FP8 shared expert runs as extra tiles and its output is added unweighted.

    Per token `t` and pick `k` (slot `s`, weight `w`), with `x^` the exact dequantization of the NVFP4 input:

        gate, up = x^ @ W1_gate[s]^T, x^ @ W1_up[s]^T            # real units: acc * fc1_alpha[s]
        a        = silu(min(gate, L)) * clamp(up, -L, L) * w      # L = gate_up_clamp, in REAL units
        a^       = nvfp4_roundtrip(a, g = fc1_norm_const[s])      # 16-element e4m3 block scales
        out[t]  += a^ @ W2[s]^T                                   # real units: acc * fc2_alpha[s]

    The routing weight is applied before the FC2 requantization (DeepSeek-V4 style), not after FC2 as
    `fp4_block_scale_moe_runner` does. And the clamp unit differs from that runner: it clamps in accumulator
    units, this kernel clamps the dequantized values, so `gate_up_clamp` is the checkpoint's `swiglu_limit`
    as is (gate: upper bound only; up: symmetric).

    Operands (all contiguous; the wrapper views bytes as `float4_e2m1fn_x2` / `float8_e4m3fn`):

    - `activation` `[max_T, H/2]` u8 and `activation_sf` `[max_T, H/16]` u8: the bucket's **symmetric**
      regions (`megamoe_symm_regions`), written by `megamoe_stage_inputs`;
    - `topk_slots` `[max_T, K]` **int64**, local, rows past `num_tokens` are -1; `topk_weights` `[max_T, K]` f32,
      symmetric, rows past `num_tokens` are 0;
    - `fc1_weight` `[S, 2I, H/2]`, `fc2_weight` `[S, H, I/2]` u8 and their scale atoms (`megamoe_pack_experts`);
      `fc1_alpha`, `fc2_alpha`, `fc1_norm_const` `[S]` f32 (`megamoe_expert_scalars`);
    - `shared_workspace` (u8) and `peer_offsets` (`W` ints): the symmetric region and peer deltas of the bucket;
    - the fused shared expert (optional): six tensors, `shared_intermediate_size`, `shared_gate_up_clamp`
      (`megamoe_pack_shared_expert`, `megamoe_shared_activation_buffers`).

    `S` is the local slots (`num_slots / W`), `max_T` the bucket, `K <= 16`, `H % 32 == 0`, `I % 16 == 0`; the shared
    expert additionally needs `H % 512 == 0` and `2 * I % 512 == 0`. With `combine_output=None` the result is a fresh
    buffer (the certified path); a symmetric `combine_output` region is not needed.

    Collective and graph rules. Every rank launches every layer, also with `num_tokens == 0` (peers wait in
    the in-kernel barrier), with the same `max_tokens_per_rank`. The weights are rank-local reads; only
    activation / scales / top-k weights / workspace are peer-visible, and peers choose destination slots from
    their own slot tables, so a placement change must be applied on every rank (weights, scalars, table) with the
    device idle and a barrier before the next launch. The first launch per (local workspace, shared workspace)
    pairing is an eager collective (it synchronizes the stream and runs `dist.barrier()` on WORLD): capture
    raises on an unfenced pairing, and once a graph captured a pairing the op refuses a later eager first
    launch on that shared workspace. Warm every bucket eagerly before the first capture and do not mix
    kernel variants (shared on/off, combine format) over the same symmetric buffers after a capture. Replay is
    graph-safe. Hidden state behind the op: a process-wide local-workspace cache and a barrier-fence record,
    which the owning target feeds by keeping one workspace per bucket for its lifetime.

    Wire formats: `combine_format` `bf16` (default), `32e4m3xe8m0` (relative RMS 0.027 against 0.0023 for
    bf16), `16e2m1xbf16` (0.094). The fused shared expert is certified with `bf16` only. The engine's
    `use_low_precision_moe_combine` does not select these: MegaMoE reads the env var `MEGAMOE_COMBINE_FORMAT`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="EP4, one copy per expert (32 slots per rank), bf16 combine: busy, short, idle and single-row ranks",
            spec=dict(world=4, slots=32, bucket=256, rows=_ROWS, shared=False, combine="bf16"),
        ),
        Cell(
            why="EP4, replicated placement (34 slots per rank, hot experts have two copies), bf16 combine",
            spec=dict(world=4, slots=34, bucket=256, rows=_ROWS, shared=False, combine="bf16"),
        ),
        Cell(
            why="EP4, replicated placement with the FP8 shared expert fused in (the deployed layer)",
            spec=dict(world=4, slots=34, bucket=256, rows=_ROWS, shared=True, combine="bf16"),
        ),
        Cell(
            why="EP4, replicated placement, 32e4m3xe8m0 combine wire format",
            spec=dict(
                world=4, slots=34, bucket=256, rows=_ROWS, shared=False, combine="32e4m3xe8m0"
            ),
        ),
        Cell(
            why="EP4, replicated placement, 16e2m1xbf16 combine wire format",
            spec=dict(
                world=4, slots=34, bucket=256, rows=_ROWS, shared=False, combine="16e2m1xbf16"
            ),
        ),
    )
    note = """
    Certified at H=5120, I=2304, top-6, 128 experts over 4 ranks (the 4-rank matrix; see the test). EP wider
    than 4 (the deployed 8 and 32), `apply_topk_in_fc1=False`, SwiGLU alpha/beta, SiTU, `in_kernel_fc2_reduce`
    and autotuning are not driven. Results on sm_103: routed vs reference relative RMS 0.0023; routed + fused
    shared 0.017 (the shared expert is ~65% of the output RMS); replicated vs one-copy bit-equal on the test
    data; eager vs graph replay bit-equal.
    """

    def __call__(
        self,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_slots: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        shared_workspace: torch.Tensor,
        peer_offsets: List[int],
        *,
        world_size: int,
        local_rank: int,
        max_tokens_per_rank: int,
        num_tokens: int,
        gate_up_clamp: Optional[float],
        combine_format: str = "bf16",
        shared_fc1_weight: Optional[torch.Tensor] = None,
        shared_fc1_weight_sf: Optional[torch.Tensor] = None,
        shared_fc2_weight: Optional[torch.Tensor] = None,
        shared_fc2_weight_sf: Optional[torch.Tensor] = None,
        shared_activation: Optional[torch.Tensor] = None,
        shared_activation_sf: Optional[torch.Tensor] = None,
        shared_intermediate_size: Optional[int] = None,
        shared_gate_up_clamp: Optional[float] = None,
        combine_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_slots, two_i, h_bytes = fc1_weight.shape
        hidden, inter = h_bytes * 2, two_i // 2
        top_k = topk_slots.shape[1]
        if combine_output is None:
            # Rows keep the bucket's divisibility (a kernel ABI requirement); rows past num_tokens drop.
            rows = max(num_tokens, 1)
            step = math.gcd(max_tokens_per_rank, 128)
            rows = -(-rows // step) * step
            combine_output = torch.empty(
                (rows, 1, hidden), dtype=torch.bfloat16, device=activation.device
            )
        shared_kwargs = {}
        if shared_intermediate_size is not None:
            shared_kwargs = dict(
                shared_fc1_weight=shared_fc1_weight,
                shared_fc1_weight_sf=shared_fc1_weight_sf,
                shared_fc2_weight=shared_fc2_weight,
                shared_fc2_weight_sf=shared_fc2_weight_sf,
                shared_activation=shared_activation,
                shared_activation_sf=shared_activation_sf,
                shared_num_tokens=num_tokens,
                shared_intermediate_size=shared_intermediate_size,
                shared_gate_up_clamp=shared_gate_up_clamp,
            )
        torch.ops.trtllm.cute_dsl_megamoe_nvfp4_blackwell(
            activation=activation.view(torch.float4_e2m1fn_x2),
            activation_sf=activation_sf.view(torch.float8_e4m3fn),
            topk_idx=topk_slots,
            topk_weights=topk_weights,
            fc1_weight=fc1_weight.view(torch.float4_e2m1fn_x2),
            fc1_weight_sf=fc1_weight_sf.view(torch.float8_e4m3fn),
            fc2_weight=fc2_weight.view(torch.float4_e2m1fn_x2),
            fc2_weight_sf=fc2_weight_sf.view(torch.float8_e4m3fn),
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            combine_output=combine_output,
            shared_workspace=shared_workspace,
            world_size=world_size,
            local_rank=local_rank,
            num_topk=top_k,
            num_experts_per_rank=n_slots,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            expand_intermediate_size_per_partition=two_i,
            max_tokens_per_rank=max_tokens_per_rank,
            peer_offsets=peer_offsets,
            apply_topk_in_fc1=True,
            gate_up_clamp=gate_up_clamp,
            in_kernel_fc2_reduce=False,
            combine_format=combine_format,
            tactic_autotune=False,
            num_tokens=num_tokens,
            **shared_kwargs,
        )
        if num_tokens == 0:
            return combine_output[:0, 0]
        return combine_output[:num_tokens, 0]

    def reference(self, xhat, ids, weights, expert, clamp, shared=None) -> torch.Tensor:
        """Pure-torch fp32 reference of one rank's rows (no peers needed: dispatch and combine are exact).

        Narrower than `__call__`, as the collective it is: `xhat` `[n, H]` f32 is the exact dequantization of the
        NVFP4 input the kernel is fed, `ids` `[n, K]` are EXPERT ids (not slots), `weights` `[n, K]` f32.
        `expert(e)` returns the real-valued `(w_gate [I, H], w_up [I, H], w_down [H, I], g)` of expert `e` in f32
        (codes * block scale * weight_scale_2) and `g = fc1_norm_const` of the slot holding it (1 / its w2
        input_scale). `shared`: None, or `(w1, w3, w2, x, limit)`, the real f32 weights of the shared expert and
        the bf16 input rows `x` `[n, H]`, whose MXFP8 round trip is applied. Math per token and pick:
        `a = silu(min(gate, L)) * clamp(up, -L, L) * w`, `a^ = nvfp4_roundtrip(a, g)`, `out += a^ @ w_down^T`;
        plus the shared expert's `silu(min(g, Ls)) * clamp(u, -Ls, Ls)` FC2 (the MXFP8 intermediate in the
        kernel is NOT modelled: expect ~2% relative RMS there).

        Tolerance that held on sm_103 (H=5120, I=2304, K=6): routed relative RMS 0.0023; with the shared
        expert 0.018; elementwise |d| <= 0.15 + 0.1 |ref| for at least 92% of the elements.
        """
        out = torch.zeros(xhat.shape, dtype=torch.float32, device=xhat.device)
        for e in ids.unique().tolist():
            rows, picks = torch.where(ids == e)
            w_gate, w_up, w_down, g = expert(int(e))
            act = (
                _swiglu(xhat[rows] @ w_gate.t(), xhat[rows] @ w_up.t(), clamp)
                * weights[rows, picks, None]
            )
            out[rows] += _nvfp4_roundtrip(act, g) @ w_down.t()
        if shared is not None:
            w1, w3, w2, x, limit = shared
            xq = _mxfp8_roundtrip(x)
            out = out + _swiglu(xq @ w1.t(), xq @ w3.t(), limit) @ w2.t()
        return out

    def is_valid(
        self,
        activation: torch.Tensor,
        activation_sf: torch.Tensor,
        topk_slots: torch.Tensor,
        topk_weights: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        shared_workspace: torch.Tensor,
        peer_offsets: List[int],
        *,
        world_size: int,
        local_rank: int,
        max_tokens_per_rank: int,
        num_tokens: int,
        gate_up_clamp: Optional[float],
        combine_format: str = "bf16",
        shared_fc1_weight: Optional[torch.Tensor] = None,
        shared_fc1_weight_sf: Optional[torch.Tensor] = None,
        shared_fc2_weight: Optional[torch.Tensor] = None,
        shared_fc2_weight_sf: Optional[torch.Tensor] = None,
        shared_activation: Optional[torch.Tensor] = None,
        shared_activation_sf: Optional[torch.Tensor] = None,
        shared_intermediate_size: Optional[int] = None,
        shared_gate_up_clamp: Optional[float] = None,
        combine_output: Optional[torch.Tensor] = None,
    ) -> None:
        # Pure-metadata guards: the kernel reads every one of these as raw pointers and sizes.
        n_slots, two_i, h_bytes = fc1_weight.shape
        hidden, inter = h_bytes * 2, two_i // 2
        top_k = topk_slots.shape[1]
        assert combine_format in _COMBINE_FORMATS, (
            f"combine_format must be one of {_COMBINE_FORMATS}"
        )
        assert fc1_weight.dtype == torch.uint8 and fc2_weight.dtype == torch.uint8
        assert fc2_weight.shape == (n_slots, hidden, inter // 2), (
            "fc2_weight must be [slots, H, I/2]"
        )
        assert (
            activation.shape == (max_tokens_per_rank, h_bytes) and activation.dtype == torch.uint8
        )
        assert activation_sf.dtype == torch.uint8 and activation_sf.shape[0] == max_tokens_per_rank
        assert topk_slots.shape == (max_tokens_per_rank, top_k), (
            "topk_slots must be the bucket buffer"
        )
        assert topk_slots.dtype == torch.int64, "sm_103 reads int64 slot ids"
        assert (
            topk_weights.shape == (max_tokens_per_rank, top_k)
            and topk_weights.dtype == torch.float32
        )
        assert fc1_alpha.shape == fc2_alpha.shape == fc1_norm_const.shape == (n_slots,)
        for name, t in (
            ("fc1_alpha", fc1_alpha),
            ("fc2_alpha", fc2_alpha),
            ("fc1_norm_const", fc1_norm_const),
        ):
            assert t.dtype == torch.float32, f"{name} must be float32"
        assert len(peer_offsets) == world_size, "peer_offsets needs one entry per rank"
        assert 0 <= num_tokens <= max_tokens_per_rank
        assert inter % 16 == 0 and hidden % 32 == 0
        for name, t in (
            ("activation", activation),
            ("activation_sf", activation_sf),
            ("topk_slots", topk_slots),
            ("topk_weights", topk_weights),
            ("fc1_weight", fc1_weight),
            ("fc1_weight_sf", fc1_weight_sf),
            ("fc2_weight", fc2_weight),
            ("fc2_weight_sf", fc2_weight_sf),
            ("fc1_alpha", fc1_alpha),
            ("fc2_alpha", fc2_alpha),
            ("fc1_norm_const", fc1_norm_const),
            ("shared_workspace", shared_workspace),
        ):
            assert t.is_contiguous(), f"{name} must be contiguous"
        if shared_intermediate_size is not None:
            assert combine_format == "bf16", "the fused shared expert needs the bf16 combine"
            assert None not in (
                shared_fc1_weight,
                shared_fc1_weight_sf,
                shared_fc2_weight,
                shared_fc2_weight_sf,
                shared_activation,
                shared_activation_sf,
            ), "the fused shared expert needs all six shared tensors"
        if combine_output is not None:
            assert (
                combine_output.shape[1:] == (1, hidden) and combine_output.dtype == torch.bfloat16
            )


cute_dsl_megamoe_nvfp4 = CuteDslMegaMoeNvfp4()
