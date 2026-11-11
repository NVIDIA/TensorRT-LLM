# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FFN side: the three leading dense MLPs and the 288-expert MoE (top-8 plus one shared
expert), block-FP8 throughout, the intermediate split over the four ranks.

Every projection is a 128 x 128 block-FP8 weight with FP32 block scales read as the
checkpoint stores them (``weight_scale_inv``, a dequantization multiplier), and every input
of it is the 1 x 128 block-FP8 quantization of its rows: ``fp8_quantize_1x128`` then
``fp8_blockwise_gemm``. The SwiGLU is clamped: ``min(gate, 10)`` and ``clamp(up, +-10)``.

Layouts. A gated MLP (the dense layers and the shared expert) keeps its gate and up rows
stacked ``[gate; up]`` (the activation reads ``[gate | up]``). The routed experts are
``[E, 2I, H]`` with ``[up; gate]`` rows and ``[E, H, I]`` down weights, the plain layout the
TRTLLM-gen FP8 block-scale runner, ``decode_moe`` and ``prefill_moe`` all read; the decode
kernel also takes the shared expert in that layout, so the shared expert is kept in both.

The MoE takes one of three paths by row count:

* at most ``decode_moe.MAX_ROWS`` (4) rows: ``decode_moe``, routing, the routed experts and
  the shared expert (as a ninth slot) in one persistent launch;
* more rows in a generation step: the router GEMM, the TRTLLM-gen runner (routing in the
  kernel) and the shared expert's two GEMMs, summed;
* sequence-parallel prefill: ``prefill_moe`` on rows the caller quantized and gathered, with
  the shared expert's output as its addend.

Outputs are this rank's partials (each expert's K-split of its intermediate); the caller
reduces them once over the TP group.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.silu_and_mul import (
    silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.dsv3_router_gemm import (
    dsv3_router_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.fp8_blockwise_gemm import (
    fp8_blockwise_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp8_block_scale_moe_runner import (
    fp8_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp8_quantize_1x128 import (
    fp8_quantize_1x128,
)

from ..checkpoint import Checkpoint, cols, rows, share
from ..dims import Dims
from ..ops.moe.decode_moe import MAX_ROWS as DECODE_MOE_ROWS
from ..ops.moe.decode_moe import decode_moe
from ..ops.moe.prefill_moe import PrefillMoeWorkspace, prefill_moe

_BLOCK = 128
#: The TRTLLM-gen runner's routing method: DeepSeek-V3 (sigmoid scores, correction bias,
#: grouped top-k, normalized weights times the routed scaling factor).
_ROUTING_DEEPSEEK_V3 = 2


@dataclass
class GatedWeights:
    """One TP-split gated MLP (a dense layer, or the shared expert)."""

    gate_up: torch.Tensor  # [2 * I_local, H] e4m3, rows [gate; up]
    gate_up_scale: torch.Tensor  # [2 * I_local / 128, H / 128] fp32
    down: torch.Tensor  # [H, I_local] e4m3
    down_scale: torch.Tensor  # [H / 128, I_local / 128] fp32


@dataclass
class DenseMlpWeights:
    mlp: GatedWeights


@dataclass
class MoeWeights:
    router: torch.Tensor  # [E, H] bf16
    bias: torch.Tensor  # [E] fp32, e_score_correction_bias
    w31: torch.Tensor  # [E, 2 * I_local, H] e4m3, rows [up; gate]
    s31: torch.Tensor  # [E, 2 * I_local / 128, H / 128] fp32
    w2: torch.Tensor  # [E, H, I_local] e4m3
    s2: torch.Tensor  # [E, H / 128, I_local / 128] fp32
    shared: GatedWeights
    #: The shared expert in the routed experts' layout, decode_moe's ninth slot:
    #: (w31 [2 * I_local, H] [up; gate], s31, w2 [H, I_local], s2).
    shared_decode: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _load_gated(ck: Checkpoint, prefix: str, intermediate: int, dims: Dims) -> GatedWeights:
    """``prefix``'s gate / up / down projections at this rank's intermediate share: gate and up
    rows, down columns, and the matching 128-blocks of their scales."""
    start, end = share(intermediate, dims.tp_size, dims.tp_rank)
    sb, se = start // _BLOCK, end // _BLOCK
    gate = ck.read(f"{prefix}gate_proj.weight", cut=rows(start, end))
    gate_s = ck.read(f"{prefix}gate_proj.weight_scale_inv", cut=rows(sb, se), dtype=torch.float32)
    up = ck.read(f"{prefix}up_proj.weight", cut=rows(start, end))
    up_s = ck.read(f"{prefix}up_proj.weight_scale_inv", cut=rows(sb, se), dtype=torch.float32)
    down = ck.read(f"{prefix}down_proj.weight", cut=cols(start, end))
    down_s = ck.read(f"{prefix}down_proj.weight_scale_inv", cut=cols(sb, se), dtype=torch.float32)
    for t in (gate, up, down):
        assert t.dtype == torch.float8_e4m3fn, (
            f"{prefix}: expected block-FP8 weights, got {t.dtype}"
        )
    return GatedWeights(
        gate_up=torch.cat([gate, up]).contiguous(),
        gate_up_scale=torch.cat([gate_s, up_s]).contiguous(),
        down=down,
        down_scale=down_s,
    )


def load_dense_mlp(ck: Checkpoint, dims: Dims, layer: int) -> DenseMlpWeights:
    return DenseMlpWeights(
        _load_gated(ck, f"{dims.layer_prefix(layer)}mlp.", dims.intermediate, dims)
    )


def load_moe(ck: Checkpoint, dims: Dims, layer: int) -> MoeWeights:
    prefix = f"{dims.layer_prefix(layer)}mlp."
    num_experts, hidden = dims.num_experts, dims.hidden
    inter = dims.moe_intermediate_local
    start, end = share(dims.moe_intermediate, dims.tp_size, dims.tp_rank)
    sb, se = start // _BLOCK, end // _BLOCK
    device = ck.device
    w31 = torch.empty((num_experts, 2 * inter, hidden), dtype=torch.float8_e4m3fn, device=device)
    s31 = torch.empty(
        (num_experts, 2 * inter // _BLOCK, hidden // _BLOCK), dtype=torch.float32, device=device
    )
    w2 = torch.empty((num_experts, hidden, inter), dtype=torch.float8_e4m3fn, device=device)
    s2 = torch.empty(
        (num_experts, hidden // _BLOCK, inter // _BLOCK), dtype=torch.float32, device=device
    )
    half = inter // _BLOCK
    for e in range(num_experts):
        ep = f"{prefix}experts.{e}."
        w31[e, :inter] = ck.read(f"{ep}up_proj.weight", cut=rows(start, end))
        w31[e, inter:] = ck.read(f"{ep}gate_proj.weight", cut=rows(start, end))
        s31[e, :half] = ck.read(f"{ep}up_proj.weight_scale_inv", cut=rows(sb, se))
        s31[e, half:] = ck.read(f"{ep}gate_proj.weight_scale_inv", cut=rows(sb, se))
        w2[e] = ck.read(f"{ep}down_proj.weight", cut=cols(start, end))
        s2[e] = ck.read(f"{ep}down_proj.weight_scale_inv", cut=cols(sb, se))
    shared = _load_gated(ck, f"{prefix}shared_experts.", dims.moe_intermediate, dims)
    gate, up = shared.gate_up.chunk(2, dim=0)
    gate_s, up_s = shared.gate_up_scale.chunk(2, dim=0)
    return MoeWeights(
        router=ck.read(f"{prefix}gate.weight", dtype=torch.bfloat16),
        bias=ck.read(f"{prefix}gate.e_score_correction_bias", dtype=torch.float32),
        w31=w31,
        s31=s31,
        w2=w2,
        s2=s2,
        shared=shared,
        shared_decode=(
            torch.cat([up, gate]).contiguous(),
            torch.cat([up_s, gate_s]).contiguous(),
            shared.down,
            shared.down_scale,
        ),
    )


def _gated_from_fp8(
    w: GatedWeights, x_fp8: torch.Tensor, x_scale: torch.Tensor, limit: float
) -> torch.Tensor:
    """A gated MLP's partial on rows already quantized 1 x 128 (``x_scale`` K-block-major):
    gate_up, clamped SwiGLU, the activation's own 1 x 128 quantization, down."""
    h = fp8_blockwise_gemm(x_fp8, w.gate_up, x_scale, w.gate_up_scale)
    act = silu_and_mul(h, swiglu_limit=limit)
    a_fp8, a_scale = fp8_quantize_1x128(act)
    return fp8_blockwise_gemm(a_fp8, w.down, a_scale, w.down_scale)


class DenseMlp:
    def __init__(self, w: DenseMlpWeights, dims: Dims, layer_idx: int) -> None:
        self.w = w.mlp
        self.limit = dims.swiglu_limit
        self.hidden = dims.hidden
        self.layer_idx = layer_idx

    def partial(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, self.hidden)
        x_fp8, x_scale = fp8_quantize_1x128(flat)
        return _gated_from_fp8(self.w, x_fp8, x_scale, self.limit).view_as(x)

    def partial_from_fp8(self, x_fp8: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        return _gated_from_fp8(self.w, x_fp8, x_scale, self.limit)


class MoeLayer:
    def __init__(self, w: MoeWeights, dims: Dims, layer_idx: int) -> None:
        self.w = w
        self.hidden = dims.hidden
        self.limit = dims.swiglu_limit
        self.scaling = dims.routed_scaling
        self.num_experts = dims.num_experts
        self.top_k = dims.top_k
        self.inter_local = dims.moe_intermediate_local
        self.tune_max_num_tokens = dims.max_num_tokens
        self.layer_idx = layer_idx

    def router_logits(self, x: torch.Tensor) -> torch.Tensor:
        """FP32 ``[rows, E]`` router logits: the BF16 router GEMM with FP32 output."""
        return dsv3_router_gemm(x, self.w.router.t(), None, torch.float32)

    def partial(self, x: torch.Tensor) -> torch.Tensor:
        """Routed experts plus the shared expert, this rank's partial, for a generation step."""
        w = self.w
        flat = x.reshape(-1, self.hidden)
        logits = self.router_logits(flat)
        if flat.shape[0] <= DECODE_MOE_ROWS:
            out = decode_moe(
                flat.contiguous(),
                logits,
                w.bias,
                self.scaling,
                self.limit,
                w.w31,
                w.s31,
                w.w2,
                w.s2,
                *w.shared_decode,
            )
            return out.view_as(x)
        x_fp8, x_scale = fp8_quantize_1x128(flat)
        routed = fp8_block_scale_moe_runner(
            logits,
            w.bias,
            x_fp8,
            x_scale,
            w.w31,
            w.s31,
            w.w2,
            w.s2,
            self.num_experts,
            self.top_k,
            0,  # no shared expert fused into the routed weights
            1,  # n_group
            1,  # topk_group
            self.inter_local,
            0,  # local_expert_offset: every rank holds every expert
            self.num_experts,
            self.scaling,
            _ROUTING_DEEPSEEK_V3,
            gemm1_clamp_limit=self.limit,
            tune_max_num_tokens=self.tune_max_num_tokens,
        )
        shared = _gated_from_fp8(w.shared, x_fp8, x_scale, self.limit)
        return (routed + shared).view_as(x)

    def prefill_partial(
        self,
        x_fp8: torch.Tensor,
        x_scale: torch.Tensor,
        logits: torch.Tensor,
        workspace: PrefillMoeWorkspace,
    ) -> torch.Tensor:
        """Every row of a sequence-parallel prefill step, from the gathered quantized rows and
        router logits: the routed experts with the shared expert's output as the addend."""
        w = self.w
        shared = _gated_from_fp8(w.shared, x_fp8, x_scale, self.limit)
        return prefill_moe(
            x_fp8,
            x_scale,
            logits,
            w.bias,
            self.scaling,
            self.limit,
            w.w31,
            w.s31,
            w.w2,
            w.s2,
            shared,
            workspace,
        )
