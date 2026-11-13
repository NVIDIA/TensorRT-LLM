# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The latent MoE layer on operations: the router, the routed experts (MXFP4 weights, MXFP8 activations, SiTU, on
the catalog's trtllm-gen runner), their checkpoint layout, and the shared experts.

A MoE layer on the generic path (a step the decode path of ``decode_moe.py`` does not take) runs:

1. the router logits, ``gemm/dsv3_router_gemm`` (bf16 x bf16 -> fp32) on the replicated hidden state;
2. the latent down projection, ``gemm/dsv3_fused_a_gemm`` (7168 -> 3584);
3. the top-16 of 896 and the MXFP8 latent: ``moe/k3_route_quant`` up to 64 tokens, else the catalog's
   ``moe/noaux_tc_op`` and ``quantization/mxfp8_quantize``;
4. this rank's routed experts, the catalog's ``moe/mxe4m3_mxe2m1_block_scale_moe_runner`` on the given top-16
   (SiTU, act_type 3, with the two soft caps per local expert in ``gemm1_alpha`` / ``gemm1_beta``), a partial sum
   over this rank's experts and intermediate slice; above ``moe_max_num_tokens`` tokens in even chunks;
5. the latent all-reduce over the TP group, the latent RMSNorm and the latent up projection (3584 -> 7168);
6. beside 2-5, on the auxiliary stream while a CUDA graph is captured: the shared experts (gate_up GEMM,
   ``activation/situ_and_mul``, down GEMM and its all-reduce);
7. the sum of the two.

These are the kernels and the order of the built-in ``ConfigurableMoE`` + ``TRTLLMGenFusedMoE`` + ``GatedMLP`` block
this replaces, so the output is the same bits. The routed experts' buffers keep the layout that block's loader
produced (``pack_mxfp4_expert``), so the decode path's kernels read them unchanged.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.mxe4m3_mxe2m1_block_scale_moe_runner import (  # noqa: E501
    mxe4m3_mxe2m1_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    mxfp8_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    environment as _environment,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.activation.situ_and_mul import (
    situ_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
    K3AllReduce,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.dsv3_fused_a_gemm import (
    dsv3_fused_a_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.dsv3_router_gemm import (
    dsv3_router_gemm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.moe.k3_route_quant import (
    k3_route_quant,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

from . import decode_comm as _decode_comm
from . import decode_gemv as _decode_gemv
from . import decode_moe as _decode_moe
from .norm import RMSNorm, run_in_parallel
from .step import DecodeStep

# A/B escape hatch: restore nn.Linear for the K3 latent MoE projections
# instead of the min-latency fused GEMM op (read once at import).
_K3_DISABLE_MIN_LATENCY_LATENT_PROJ = _environment.disable_min_latency_latent_proj()

# A/B escape hatch: route up to 64 tokens with noaux_tc_op + mxfp8_quantize instead of the fused
# moe/k3_route_quant (read once at import).
_K3_DISABLE_FUSED_ROUTE_QUANT = _environment.disable_fused_route_quant()

# The geometry moe/k3_route_quant serves: 896 experts, top-16, latent 3584, up to 64 tokens.
_ROUTE_QUANT_SHAPE = (896, 16, 3584)
_ROUTE_QUANT_MAX_TOKENS = 64

# The runner's routing_method_type for DeepSeek-V3 routing (RoutingMethodType.DeepSeekV3); with the top-16 given,
# the kernel only reads the routing arguments, it does not route.
_ROUTING_METHOD_DEEPSEEK_V3 = 2
# The runner's act_type for SiTU (ActType_TrtllmGen.SiTu).
_ACT_TYPE_SITU = 3

# MXFP4 (group-32 E8M0 scales, two E2M1 values per byte) on the trtllm-gen cubins.
_SCALE_BLOCK = 32
# The cubins' row alignment of both GEMMs' intermediate axis, and FC1's K alignment.
_WEIGHT_ALIGN = 128
_FC1_K_ALIGN = 512
# The 32-row block shuffle of the trtllm-gen weight layout (epilogue tile 128): source row 4u + v of a block lands
# at destination row 8v + u.
_SHUFFLE_BLOCK = 32


class KimiK3MoEGate(nn.Module):
    """Kimi K3 gate weights: the router weight and the routing correction bias."""

    def __init__(
        self,
        config: Any,
        *,
        logits_gemm_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size

        assert self.moe_router_activation_func in ("sigmoid", "softmax"), (
            "K3 MoE gate supports sigmoid or softmax scoring only"
        )

        # The checkpoint stores the gate weight in bf16. Storing it in bf16
        # permits the single bf16xbf16 router GEMM while retaining fp32 output.
        weight_dtype = logits_gemm_dtype or torch.float32
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim), dtype=weight_dtype, device=device)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32, device=device)
        )

    def check_routing(self) -> None:
        """The routing the routed experts compute: DeepSeek-V3 sigmoid scores with renormalized top-k weights."""
        if self.moe_router_activation_func != "sigmoid":
            raise ValueError("Kimi K3 ConfigurableMoE routing requires sigmoid scores.")
        if not self.moe_renormalize:
            raise ValueError(
                "Kimi K3 ConfigurableMoE routing requires top-k weight renormalization."
            )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute fp32 routing logits shaped ``[num_tokens, num_experts]``."""
        hidden_2d = hidden_states.reshape(-1, self.gating_dim)
        if self.weight.dtype == torch.bfloat16 and hidden_2d.dtype == torch.bfloat16:
            return dsv3_router_gemm(hidden_2d.contiguous(), self.weight)
        return F.linear(hidden_2d.type(torch.float32), self.weight.type(torch.float32), None)


def _resolve_kimi_situ_betas(cfg: Any) -> tuple[float, float]:
    """Return the finite SiTu betas required by the routed-expert kernels."""
    config_situ_beta = getattr(cfg, "activation_situ_beta", None)
    situ_beta = 1.0 if config_situ_beta is None else config_situ_beta
    situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
    if situ_linear_beta is None:
        raise ValueError(
            "Kimi K3 routed SiTu experts require activation_situ_linear_beta; "
            "None means an identity linear branch that the fused kernels cannot represent."
        )
    if situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError(
            f"Kimi K3 SiTu betas must be positive; got {situ_beta} and {situ_linear_beta}."
        )
    return float(situ_beta), float(situ_linear_beta)


_K3_ROUTED_EXPERT_KEY_SUFFIXES = ("block_sparse_moe.experts", "mlp.experts")


# The routed experts' module path prefixes. ``exclude_modules`` matches with
# wildcards and walks ancestor prefixes, so an empty prefix would widen what
# matches instead of just missing.
_K3_ROUTED_EXPERT_MODULE_PREFIXES = ("language_model.model.", "model.")


# Routed-expert quantization used when the checkpoint declares nothing per
# layer. The original ``moonshotai/Kimi-K3`` ships a compressed-tensors
# ``mxfp4-pack-quantized`` config with no ModelOpt per-layer entries, and that
# checkpoint is what this default has always served.
_K3_DEFAULT_ROUTED_QUANT_ALGO = QuantAlgo.W4A8_MXFP4_MXFP8


# ----------------------------------------------------------------------------------------------------------------------
# The routed experts' checkpoint layout: one expert's packed MXFP4 tensors into its buffer slots.
# ----------------------------------------------------------------------------------------------------------------------


def _pad_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _weight_alignment(tp_size: int, shard_dim_size: int) -> int:
    """The row alignment a tensor is padded to before its MoE TP split: whole scale groups and whole 128-row tiles
    per rank. The built-in loader's ``_get_weight_alignment`` (weight alignment 128, scale group 32)."""
    alignment = math.lcm(_WEIGHT_ALIGN, _SCALE_BLOCK, tp_size)
    per_shard = _pad_up(shard_dim_size, alignment) // tp_size
    if per_shard % _WEIGHT_ALIGN != 0:
        alignment = _WEIGHT_ALIGN * math.ceil(per_shard / _WEIGHT_ALIGN) * tp_size
    return alignment


def _pad_2d(t: torch.Tensor, col_alignment: int, row_alignment: int) -> torch.Tensor:
    """Zero-pad the columns of ``t`` to a multiple of ``col_alignment`` and its rows to one of ``row_alignment``."""
    cols = (col_alignment - t.shape[-1]) % col_alignment
    rows = (row_alignment - t.shape[-2]) % row_alignment
    return F.pad(t, (0, cols, 0, rows)) if cols or rows else t


def _pad_to(t: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a 2-D ``t`` to ``[rows, cols]``."""
    assert t.shape[0] <= rows and t.shape[1] <= cols, (tuple(t.shape), rows, cols)
    if t.shape == (rows, cols):
        return t
    return F.pad(t, (0, cols - t.shape[1], 0, rows - t.shape[0]))


def _even_shard(t: torch.Tensor, tp_size: int, tp_rank: int, dim: int) -> torch.Tensor:
    """Rank ``tp_rank``'s slice of ``t`` along ``dim`` when split in ``tp_size`` ceil-sized parts."""
    if tp_size <= 1:
        return t
    width = math.ceil(t.shape[dim] / tp_size)
    start = tp_rank * width
    return t.narrow(dim, start, min(start + width, t.shape[dim]) - start)


def _shuffle_rows(rows: int, device: torch.device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle."""
    assert rows % _SHUFFLE_BLOCK == 0, rows
    src = torch.arange(_SHUFFLE_BLOCK)
    idx = torch.empty(_SHUFFLE_BLOCK, dtype=torch.long)
    idx[(src % 4) * 8 + src // 4] = src
    blocks = rows // _SHUFFLE_BLOCK
    return (
        idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(_SHUFFLE_BLOCK) * _SHUFFLE_BLOCK
    ).to(device)


def _fc1_rows(rows: int, device: torch.device) -> torch.Tensor:
    """Gather index of the FC1 rows: ``[up | gate]`` interleaved (up 0, gate 0, up 1, ...), then block-shuffled."""
    interleave = torch.empty(rows, dtype=torch.long)
    interleave[0::2] = torch.arange(0, rows // 2)
    interleave[1::2] = torch.arange(rows // 2, rows)
    return interleave.to(device)[_shuffle_rows(rows, device)]


def _swizzle_scales(s: torch.Tensor) -> torch.Tensor:
    """The 128x4 block-scale swizzle of a ``[M, C]`` scale tensor: the byte at ``(m, c)`` moves to flat offset
    ``(m//128)*512*(C//4) + (c//4)*512 + (m%32)*16 + ((m%128)//32)*4 + (c%4)``, viewed back as ``[M, C]``."""
    m, c = s.shape
    assert m % 128 == 0 and c % 4 == 0, (m, c)
    v = s.reshape(m // 128, 4, 32, c // 4, 4)  # (m/128, (m%128)/32, m%32, c/4, c%4)
    return v.permute(0, 3, 2, 1, 4).reshape(m, c).contiguous()


@dataclass(frozen=True)
class Mxfp4ExpertLayout:
    """Where one rank's share of an expert lands: ``hidden`` (the latent width the experts read), the checkpoint's
    ``intermediate`` width, and the rank's place in the MoE TP split of that intermediate.

    The intermediate split follows the built-in trtllm-gen loader: ``intermediate // moe_tp`` columns per rank,
    padded to whole 128-row tiles. When that is whole scale groups but not whole tiles (192 of 3072 at moe TP 16),
    each rank takes its logical slice and zero-pads it (192 -> 256); otherwise the full tensor is padded first and
    split evenly (768 of 3072 at moe TP 4: no padding)."""

    hidden: int
    intermediate: int
    moe_tp_size: int = 1
    moe_tp_rank: int = 0

    @property
    def per_rank(self) -> int:
        """The rank's logical share of the intermediate, before padding."""
        return self.intermediate // self.moe_tp_size

    @property
    def inter_pad(self) -> int:
        """The intermediate width of the rank's buffers."""
        return _pad_up(self.per_rank, _WEIGHT_ALIGN)

    @property
    def fc1_k_pad(self) -> int:
        return _pad_up(self.hidden, _FC1_K_ALIGN)

    @property
    def fc2_rows_pad(self) -> int:
        return _pad_up(self.hidden, _WEIGHT_ALIGN)

    @property
    def logical_slices(self) -> bool:
        """Whether each rank slices its logical share and pads it (rather than padding the whole, then splitting)."""
        per = self.per_rank
        return (
            self.moe_tp_size > 1
            and self.intermediate % self.moe_tp_size == 0
            and per % _SCALE_BLOCK == 0
            and per % _WEIGHT_ALIGN != 0
        )

    @property
    def valid_intermediate(self) -> int:
        """How many of the rank's ``inter_pad`` intermediate columns hold checkpoint values (the runner's
        ``valid_intermediate_size``)."""
        if self.logical_slices:
            return self.per_rank
        padded = _pad_up(self.intermediate, _weight_alignment(self.moe_tp_size, self.intermediate))
        width = math.ceil(padded / self.moe_tp_size) if self.moe_tp_size > 1 else padded
        start = self.moe_tp_rank * width if self.moe_tp_size > 1 else 0
        return max(0, min(self.intermediate, start + width) - start)

    def slot_shapes(self) -> Dict[str, Tuple[int, int]]:
        """Each buffer's per-expert shape (uint8)."""
        return {
            "w3_w1_weight": (2 * self.inter_pad, self.fc1_k_pad // 2),
            "w3_w1_weight_scale": (2 * self.inter_pad, self.fc1_k_pad // _SCALE_BLOCK),
            "w2_weight": (self.fc2_rows_pad, self.inter_pad // 2),
            "w2_weight_scale": (self.fc2_rows_pad, self.inter_pad // _SCALE_BLOCK),
        }


def _fc1_half(t: torch.Tensor, layout: Mxfp4ExpertLayout, values_per_byte: int) -> torch.Tensor:
    """The rank's ``[inter_pad, fc1_k_pad / values_per_byte]`` share of one FC1 half (``w1`` or ``w3``, weights with
    ``values_per_byte`` 2 or scales with 32)."""
    rows, cols = layout.inter_pad, layout.fc1_k_pad // values_per_byte
    if layout.logical_slices:
        per = layout.per_rank
        return _pad_to(t.narrow(0, layout.moe_tp_rank * per, per), rows, cols)
    alignment = _weight_alignment(layout.moe_tp_size, t.shape[0])
    t = _pad_2d(t, _FC1_K_ALIGN // values_per_byte, alignment)
    shard = _even_shard(t, layout.moe_tp_size, layout.moe_tp_rank, 0)
    assert shard.shape == (rows, cols), (tuple(shard.shape), rows, cols)
    return shard


def pack_mxfp4_expert(
    layout: Mxfp4ExpertLayout,
    w1_weight: torch.Tensor,
    w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w3_weight: torch.Tensor,
    w3_weight_scale: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """One checkpoint expert's packed MXFP4 tensors as this rank's buffer slots, in the trtllm-gen layout.

    Inputs are the checkpoint's uint8 tensors at full width, on any device (the outputs land on the same device):
    ``w1`` (gate) and ``w3`` (up) ``[intermediate, hidden / 2]`` with scales ``[intermediate, hidden / 32]``, and
    ``w2`` (down) ``[hidden, intermediate / 2]`` with scales ``[hidden, intermediate / 32]``.

    Returns new contiguous uint8 tensors, keyed by buffer name (``Mxfp4ExpertLayout.slot_shapes``): the rank's
    intermediate share of each (``Mxfp4ExpertLayout``), zero-padded; ``w3_w1_weight`` as ``[up | gate]`` rows
    interleaved then 32-row block-shuffled, ``w2_weight`` block-shuffled, and both scales row-permuted as their
    weights and 128x4 swizzled. The same bytes as the built-in ``load_packed_mxfp4_expert``.
    """
    for name, value in (
        ("w1_weight", w1_weight),
        ("w1_weight_scale", w1_weight_scale),
        ("w2_weight", w2_weight),
        ("w2_weight_scale", w2_weight_scale),
        ("w3_weight", w3_weight),
        ("w3_weight_scale", w3_weight_scale),
    ):
        if value.dtype != torch.uint8 or value.dim() != 2:
            raise TypeError(
                f"{name} must be 2-D packed MXFP4 uint8 data, got {value.dtype} {tuple(value.shape)}."
            )
    device = w1_weight.device
    tp, rank = layout.moe_tp_size, layout.moe_tp_rank

    w3_w1 = torch.cat([_fc1_half(w3_weight, layout, 2), _fc1_half(w1_weight, layout, 2)])
    w3_w1_scale = torch.cat(
        [
            _fc1_half(w3_weight_scale, layout, _SCALE_BLOCK),
            _fc1_half(w1_weight_scale, layout, _SCALE_BLOCK),
        ]
    )
    fc1_rows = _fc1_rows(w3_w1.shape[0], device)

    rows, inter_pad = layout.fc2_rows_pad, layout.inter_pad
    if layout.logical_slices:
        per = layout.per_rank
        w2 = _pad_to(w2_weight.narrow(1, rank * per // 2, per // 2), rows, inter_pad // 2)
        w2_scale = _pad_to(
            w2_weight_scale.narrow(1, rank * per // _SCALE_BLOCK, per // _SCALE_BLOCK),
            rows,
            inter_pad // _SCALE_BLOCK,
        )
    else:
        # The weight's split is sized off its logical width, the scale's off its own column count, as the built-in
        # loader does; both come out at the rank's padded width.
        alignment = _weight_alignment(tp, 2 * w2_weight.shape[1])
        w2 = _even_shard(_pad_2d(w2_weight, alignment // 2, _WEIGHT_ALIGN), tp, rank, 1)
        alignment = _weight_alignment(tp, w2_weight_scale.shape[1])
        w2_scale = _even_shard(
            _pad_2d(w2_weight_scale, alignment // _SCALE_BLOCK, _WEIGHT_ALIGN), tp, rank, 1
        )
        assert w2.shape == (rows, inter_pad // 2), (tuple(w2.shape), rows, inter_pad)
        assert w2_scale.shape == (rows, inter_pad // _SCALE_BLOCK), (tuple(w2_scale.shape), rows)
    fc2_rows = _shuffle_rows(rows, device)

    return {
        "w3_w1_weight": torch.index_select(w3_w1, 0, fc1_rows).contiguous(),
        "w3_w1_weight_scale": _swizzle_scales(torch.index_select(w3_w1_scale, 0, fc1_rows)),
        "w2_weight": torch.index_select(w2, 0, fc2_rows).contiguous(),
        "w2_weight_scale": _swizzle_scales(torch.index_select(w2_scale, 0, fc2_rows)),
    }


def _check_routed_quant(quant_algo: Optional[QuantAlgo]) -> None:
    """Reject a routed-expert quantization this module has no buffer layout for (only the MXFP4 checkpoint's
    W4A8_MXFP4_MXFP8; an unquantized expert set, from a checkpoint that excludes them, is not served)."""
    if quant_algo != _K3_DEFAULT_ROUTED_QUANT_ALGO:
        raise NotImplementedError(
            f"Kimi K3 routed experts are quantized as {quant_algo}, for which "
            "no per-expert checkpoint layout is known. Supported: "
            f"{[_K3_DEFAULT_ROUTED_QUANT_ALGO.name]}."
        )


# Streaming loaders run experts on a thread pool: the duplicate check and the slot claim are one step.
_SLOT_CLAIM_LOCK = threading.Lock()


class K3MxfpExpertWeights(nn.Module):
    """This rank's routed-expert buffers in the trtllm-gen W4A8_MXFP4_MXFP8 layout the decode kernels read:
    ``w3_w1_weight`` ``[E, 2 I_pad, H / 2]``, ``w3_w1_weight_scale`` ``[E, 2 I_pad, H / 32]``, ``w2_weight``
    ``[E, H, I_pad / 2]``, ``w2_weight_scale`` ``[E, H, I_pad / 32]`` (uint8, E local experts, I_pad the padded
    intermediate share; ``pack_mxfp4_expert``), and the SiTU soft caps per local expert (``act_alpha`` the gate's,
    ``act_beta`` the linear branch's, fp32), the runner's ``gemm1_alpha`` / ``gemm1_beta``."""

    def __init__(
        self,
        *,
        num_experts: int,
        moe_ep_size: int,
        moe_ep_rank: int,
        layout: Mxfp4ExpertLayout,
        situ_betas: Tuple[float, float],
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if num_experts % moe_ep_size:
            raise ValueError(f"{num_experts} experts do not split over moe_ep {moe_ep_size}")
        if layout.intermediate % layout.moe_tp_size or layout.per_rank % _SCALE_BLOCK:
            raise ValueError(
                "TRTLLM-Gen SiTu MoE TP requires intermediate_size "
                f"({layout.intermediate}) divisible by moe_tp_size "
                f"({layout.moe_tp_size}) with the per-rank shard a multiple of "
                f"{_SCALE_BLOCK}, got {layout.per_rank}."
            )
        self.num_experts = num_experts
        self.expert_size_per_partition = num_experts // moe_ep_size
        self.slot_start = moe_ep_rank * self.expert_size_per_partition
        self.initial_local_expert_ids = list(
            range(self.slot_start, self.slot_start + self.expert_size_per_partition)
        )
        self.layout = layout
        self.hidden_size = layout.hidden
        # The checkpoint columns each expert's intermediate holds on this rank; its buffers hold layout.inter_pad.
        self.intermediate_size_per_partition = layout.valid_intermediate
        self.situ_betas = situ_betas
        # The local slots written so far (``load_packed_mxfp4_expert``), checked by ``check_loaded``.
        self.loaded_slots: set[int] = set()
        for name, shape in layout.slot_shapes().items():
            self.register_parameter(
                name,
                nn.Parameter(
                    torch.empty(
                        (self.expert_size_per_partition, *shape), dtype=torch.uint8, device=device
                    ),
                    requires_grad=False,
                ),
            )
        self.register_buffer(
            "act_alpha",
            torch.empty(self.expert_size_per_partition, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "act_beta",
            torch.empty(self.expert_size_per_partition, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.cache_derived_state()

    def cache_derived_state(self) -> None:
        """Fill the SiTU soft caps (configuration, not checkpoint weights; a meta-initialized module has them
        unset until here)."""
        if self.act_alpha.device.type == "meta":
            return
        with torch.no_grad():
            self.act_alpha.fill_(self.situ_betas[0])
            self.act_beta.fill_(self.situ_betas[1])

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def load_packed_mxfp4_expert(
        self,
        *,
        global_expert_id: int,
        local_slot_id: int,
        w1_weight: torch.Tensor,
        w1_weight_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        w3_weight: torch.Tensor,
        w3_weight_scale: torch.Tensor,
    ) -> None:
        """Write one checkpoint expert (``pack_mxfp4_expert``'s inputs, on any device) into local slot
        ``local_slot_id``. Each slot is written once; ``loaded_slots`` records the filled slots. Thread safe across
        slots."""
        if not 0 <= local_slot_id < self.expert_size_per_partition:
            raise IndexError(
                f"local_slot_id={local_slot_id} is outside [0, {self.expert_size_per_partition})."
            )
        expected = self.initial_local_expert_ids[local_slot_id]
        if global_expert_id != expected:
            raise ValueError(
                f"local slot {local_slot_id} expects global expert {expected}, got {global_expert_id}."
            )
        # The slot is claimed before it is written: a failed write leaves it partly filled, and a retry must not
        # pass for a complete load.
        with _SLOT_CLAIM_LOCK:
            if local_slot_id in self.loaded_slots:
                raise ValueError(f"Packed MXFP4 local slot {local_slot_id} was loaded twice.")
            self.loaded_slots.add(local_slot_id)
        device = self.w3_w1_weight.device
        slots = pack_mxfp4_expert(
            self.layout,
            w1_weight.to(device),
            w1_weight_scale.to(device),
            w2_weight.to(device),
            w2_weight_scale.to(device),
            w3_weight.to(device),
            w3_weight_scale.to(device),
        )
        with torch.no_grad():
            for name, value in slots.items():
                getattr(self, name).data[local_slot_id].copy_(value)

    def check_loaded(self) -> None:
        """Raise unless every local slot has been written once."""
        missing = sorted(set(range(self.expert_size_per_partition)) - self.loaded_slots)
        if missing:
            raise RuntimeError(
                f"Kimi K3 load did not fill every routed-expert slot; missing {missing[:10]}"
            )


class K3RoutedExperts(nn.Module):
    """This rank's routed experts (``backend``: the buffers) and the latent all-reduce (``all_reduce``, None on one
    rank). ``forward`` routes the latent and returns this rank's partial sum over its experts and intermediate share,
    in chunks of at most ``moe_max_num_tokens`` tokens."""

    def __init__(
        self,
        backend: K3MxfpExpertWeights,
        *,
        top_k: int,
        n_group: int,
        topk_group: int,
        all_reduce: Optional[K3AllReduce],
        moe_max_num_tokens: int,
        tune_max_num_tokens: int,
        chunk_stream: Optional[torch.cuda.Stream],
    ):
        super().__init__()
        self.backend = backend
        self.all_reduce = all_reduce
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_max_num_tokens = moe_max_num_tokens
        self.tune_max_num_tokens = tune_max_num_tokens
        # Chunks alternate between this stream and the caller's when chunking is configured below the engine's token
        # budget, as the built-in MoE does; None runs every chunk on the caller's stream.
        self.chunk_stream = chunk_stream
        self.chunk_events = (
            (torch.cuda.Event(), torch.cuda.Event()) if chunk_stream is not None else None
        )
        self.route_quant_fused = (
            not _K3_DISABLE_FUSED_ROUTE_QUANT
            and (backend.num_experts, top_k, backend.hidden_size) == _ROUTE_QUANT_SHAPE
            and (n_group, topk_group) == (1, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> torch.Tensor:
        """``x``: the latent, bf16 ``[M, H]``; ``router_logits``: fp32 ``[M, num_experts]``. Returns bf16 ``[M, H]``."""
        rows = x.shape[0]
        if rows <= self.moe_max_num_tokens:
            return self._run(x, router_logits, e_score_correction_bias, routed_scaling_factor)
        num_chunks = math.ceil(rows / self.moe_max_num_tokens)
        div, mod = divmod(rows, num_chunks)
        sizes = [div + 1] * mod + [div] * (num_chunks - mod)
        xs, logits = x.split(sizes), router_logits.split(sizes)
        stream = self.chunk_stream
        if stream is not None:
            main, aux_done = self.chunk_events
            main.record()
            with torch.cuda.stream(stream):
                main.wait()
        outputs = []
        for i, (x_chunk, logits_chunk) in enumerate(zip(xs, logits)):
            if stream is not None and i % 2 == 0:
                with torch.cuda.stream(stream):
                    out = self._run(
                        x_chunk, logits_chunk, e_score_correction_bias, routed_scaling_factor
                    )
            else:
                out = self._run(
                    x_chunk, logits_chunk, e_score_correction_bias, routed_scaling_factor
                )
            outputs.append(out)
        if stream is not None:
            with torch.cuda.stream(stream):
                aux_done.record()
            aux_done.wait()
        return torch.cat(outputs)

    def _run(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        routed_scaling_factor: float,
    ) -> torch.Tensor:
        """One chunk: the top-k and the MXFP8 latent, then the experts on the runner."""
        rows = x.shape[0]
        if self.route_quant_fused and 0 < rows <= _ROUTE_QUANT_MAX_TOKENS:
            topk_ids, topk_weights, x_fp8, x_sf = k3_route_quant(
                router_logits, e_score_correction_bias, x, routed_scaling_factor
            )
        else:
            weights, ids = noaux_tc_op(
                router_logits,
                e_score_correction_bias,
                self.n_group,
                self.topk_group,
                self.top_k,
                routed_scaling_factor,
            )
            # The built-in routing's casts: int32 ids, fp32 then bf16 weights (the runner reads bf16).
            topk_ids = ids.to(torch.int32)
            topk_weights = weights.to(torch.float32).to(torch.bfloat16)
            x_fp8, x_sf = mxfp8_quantize(x, False, _FC1_K_ALIGN)
        b = self.backend
        layout = b.layout
        return mxe4m3_mxe2m1_block_scale_moe_runner(
            None,
            None,
            x_fp8,
            x_sf.reshape(-1),
            b.w3_w1_weight,
            b.w3_w1_weight_scale,
            None,
            b.act_alpha,
            b.act_beta,
            None,
            b.w2_weight,
            b.w2_weight_scale,
            None,
            b.num_experts,
            self.top_k,
            self.n_group,
            self.topk_group,
            layout.inter_pad,
            layout.hidden,
            layout.valid_intermediate,
            b.slot_start,
            b.expert_size_per_partition,
            routed_scaling_factor,
            _ROUTING_METHOD_DEEPSEEK_V3,
            _ACT_TYPE_SITU,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=None,
            tune_max_num_tokens=self.tune_max_num_tokens,
            use_dp=False,
        )


class K3RowParallelLinear(nn.Module):
    """``x @ weight^T`` over this rank's input columns, then the TP all-reduce (None on one rank). ``weight`` is
    ``[out, in / tp]`` bf16, no bias."""

    def __init__(
        self, weight_shape: Tuple[int, int], dtype: torch.dtype, all_reduce: Optional[K3AllReduce]
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(weight_shape, dtype=dtype))
        self.bias = None
        self.all_reduce = all_reduce

    def forward(
        self, x: torch.Tensor, *, all_reduce_params=None, layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """``all_reduce_params``: the stock all-reduce's parameters (``AllReduceParams(enable_allreduce=False)``
        returns this rank's partial); ``layer_idx`` is accepted for the built-in module's call signature."""
        del layer_idx
        out = F.linear(x, self.weight)
        if self.all_reduce is not None:
            out = self.all_reduce(out, all_reduce_params=all_reduce_params)
        return out


class KimiK3GatedMLP(nn.Module):
    """``down(SiTU(gate_up(x)))`` at this rank's intermediate share: ``gate_up_proj`` ``[2 I / tp, H]`` (``[gate |
    up]`` rows), ``down_proj`` a ``K3RowParallelLinear`` ``[H, I / tp]`` with the TP all-reduce."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        situ_betas: Tuple[float, float],
        mapping: Mapping,
        all_reduce: Optional[K3AllReduce],
        layer_idx: Optional[int],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if intermediate_size % mapping.tp_size:
            raise ValueError(
                f"the shared experts' intermediate {intermediate_size} does not split over TP {mapping.tp_size}"
            )
        local = intermediate_size // mapping.tp_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.situ_betas = situ_betas
        self.layer_idx = layer_idx
        self.gate_up_proj = nn.Linear(hidden_size, 2 * local, bias=False, dtype=dtype)
        self.down_proj = K3RowParallelLinear((hidden_size, local), dtype, all_reduce)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """SiTU on the ``[gate | up]`` halves of the last dimension."""
        out = situ_and_mul(x.reshape(-1, x.shape[-1]), *self.situ_betas)
        return out.reshape(*x.shape[:-1], x.shape[-1] // 2)

    def forward(self, x: torch.Tensor, final_all_reduce_params=None) -> torch.Tensor:
        return self.down_proj(
            self._apply_activation(self.gate_up_proj(x)),
            all_reduce_params=final_all_reduce_params,
            layer_idx=self.layer_idx,
        )


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block on operations (see the module docstring)."""

    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        """Build the routed experts and the shared expert for one MoE layer.

        ``cfg`` is the raw ``PretrainedConfig`` rather than anything derived:
        the SiTU soft-caps and the routed-expert geometry are Kimi K3 fields
        that ``ModelConfig`` does not carry.

        ``aux_stream_dict`` is shared across every layer of the model, so the
        streams reached through it are borrowed and must not be synchronized
        or reassigned here.
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        # ValueError (not assert): these guard unsupported checkpoint
        # configurations and must stay active under ``python -O``.
        if self.moe_hidden_size is None:
            raise ValueError("Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)")
        if not getattr(cfg, "latent_moe_use_norm", False):
            raise ValueError("Kimi K3 runtime expects latent_moe_use_norm=True")

        situ_beta, situ_linear_beta = _resolve_kimi_situ_betas(cfg)
        self._situ_betas = (situ_beta, situ_linear_beta)
        dtype = torch.bfloat16

        # Routing scores stay fp32; the gate GEMM runs bf16xbf16 with fp32
        # accumulate/output (checkpoint stores the gate weight in bf16; saves a
        # per-layer input cast + fp32 splitK pair on the bs1 decode path).
        # KIMI_K3_ROUTER_BF16=0 forces the upcast-to-fp32 GEMM.
        _router_bf16 = _environment.router_bf16()
        self.gate = KimiK3MoEGate(cfg, logits_gemm_dtype=torch.bfloat16 if _router_bf16 else None)
        self.gate.check_routing()

        if model_config.moe_backend != "TRTLLM":
            raise NoTarget(
                f"moe_config.backend {model_config.moe_backend}: the Kimi K3 targets run the routed experts on "
                "trtllm-gen (TRTLLM, or AUTO, which resolves to it)"
            )
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        _check_routed_quant(self._resolve_routed_quant_algo(model_config, layer_idx))

        mapping = model_config.mapping
        routed_mapping = self._routed_mapping(mapping)
        logger.info_once(
            f"Kimi K3 routed MoE parallelism: moe_tp={routed_mapping.moe_tp_size}, "
            f"moe_ep={routed_mapping.moe_ep_size} (tp_size={mapping.tp_size})",
            key="kimi_k3_moe_tp_ep_split",
        )
        # Direct MoE-TP leaves both branches as partials, each with its all-reduce.
        reduce = mapping.tp_size > 1
        if reduce and mapping.enable_attention_dp:
            raise RuntimeError(
                "Kimi K3 direct MoE tensor parallelism requires the latent all-reduce, which attention data "
                "parallelism does not build."
            )
        self._reduce_routed_output = reduce
        backend = K3MxfpExpertWeights(
            num_experts=self.num_experts,
            moe_ep_size=routed_mapping.moe_ep_size,
            moe_ep_rank=routed_mapping.moe_ep_rank,
            layout=Mxfp4ExpertLayout(
                hidden=self.moe_hidden_size,
                intermediate=cfg.moe_intermediate_size,
                moe_tp_size=routed_mapping.moe_tp_size,
                moe_tp_rank=routed_mapping.moe_tp_rank,
            ),
            situ_betas=self._situ_betas,
        )
        moe_max_num_tokens = model_config.moe_max_num_tokens
        chunked_below_budget = moe_max_num_tokens < model_config.max_num_tokens * mapping.dp_size
        self.routed_experts = K3RoutedExperts(
            backend,
            top_k=self.top_k,
            n_group=self.gate.num_expert_group,
            topk_group=self.gate.topk_group,
            all_reduce=(
                K3AllReduce(routed_mapping, model_config.allreduce_strategy, dtype)
                if reduce
                else None
            ),
            moe_max_num_tokens=moe_max_num_tokens,
            tune_max_num_tokens=model_config.max_num_tokens,
            chunk_stream=(
                aux_stream_dict[AuxStreamType.MoeChunkingOverlap] if chunked_below_budget else None
            ),
        )
        self.local_expert_ids = tuple(backend.initial_local_expert_ids)
        self.experts_per_rank = backend.expert_size_per_partition
        self.expert_lo = backend.slot_start
        self.expert_hi = self.expert_lo + self.experts_per_rank

        self.shared_experts = KimiK3GatedMLP(
            cfg.hidden_size,
            cfg.moe_intermediate_size * cfg.num_shared_experts,
            self._situ_betas,
            mapping,
            (K3AllReduce(mapping, model_config.allreduce_strategy, dtype) if reduce else None),
            layer_idx,
            dtype=dtype,
        )
        # Side stream (+ fork/join events) for overlapping shared-expert
        # compute with the routed chain. Only engaged while a CUDA graph is
        # captured; otherwise both run in order on the default stream.
        self.shared_expert_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
        self.routed_expert_down_proj = nn.Linear(
            cfg.hidden_size, self.moe_hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_up_proj = nn.Linear(
            self.moe_hidden_size, cfg.hidden_size, bias=False, dtype=dtype
        )
        self.routed_expert_norm = RMSNorm(self.moe_hidden_size, cfg.rms_norm_eps, dtype=dtype)
        # The decode path (decode_moe.py) and the decode GEMVs' state, set by the target's post_load_weights.
        self.decode_moe: Optional[_decode_moe.K3DecodeMoeLayer] = None
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

    @staticmethod
    def _routed_projection(hidden_states: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        """A latent projection on ``gemm/dsv3_fused_a_gemm`` (nn.Linear under the escape hatch)."""
        if _K3_DISABLE_MIN_LATENCY_LATENT_PROJ:
            return projection(hidden_states)
        return dsv3_fused_a_gemm(hidden_states, projection.weight)

    @staticmethod
    def _routed_mapping(mapping: Mapping) -> Mapping:
        """The routed experts' mapping: the engine's, with the user config's explicit ``moe_tensor_parallel_size`` /
        ``moe_expert_parallel_size`` (16 x 1, asserted at construction) and no MoE cluster."""
        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = mapping.moe_tp_size
        mapping_dict["moe_ep_size"] = mapping.moe_ep_size
        return Mapping.from_dict(mapping_dict)

    @staticmethod
    def _resolve_routed_quant_algo(
        model_config: ModelConfig, layer_idx: int
    ) -> Optional[QuantAlgo]:
        """Routed-expert quantization for ``layer_idx``: the MXFP4 checkpoint declares nothing per layer (asserted at
        routing), so the experts keep the ``W4A8_MXFP4_MXFP8`` default; None (unquantized) where the checkpoint
        excludes them, which no layout here serves. Matched as a pattern, so it is asked only about real module
        names."""
        quant_config = model_config.quant_config
        if quant_config is not None and any(
            quant_config.is_module_excluded_from_quantization(
                f"{prefix}layers.{layer_idx}.{suffix}"
            )
            for prefix in _K3_ROUTED_EXPERT_MODULE_PREFIXES
            for suffix in _K3_ROUTED_EXPERT_KEY_SUFFIXES
        ):
            return None
        return _K3_DEFAULT_ROUTED_QUANT_ALGO

    def tail_rp_eligible(self, hidden_states: torch.Tensor, step: DecodeStep) -> bool:
        """Whether this layer's forward on ``step`` can hand its output on as this rank's unreduced row-parallel
        share (``partial_tail``): the decode path takes the step with that tail."""
        return self.decode_moe is not None and self.decode_moe.takes(hidden_states, step, True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens=None,
        partial_tail: bool = False,
        *,
        step: DecodeStep,
    ) -> Union[torch.Tensor, _decode_comm.PendingTail]:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16. ``step``: the step's classification
        (``decode_step``); the decode path (``decode_moe.py``) runs the steps it takes. ``partial_tail`` (only where
        ``tail_rp_eligible`` holds): return this rank's unreduced share of the output instead, a ``PendingTail`` at
        most 8 tokens, a tensor on a wide decode step."""
        decode = self.decode_moe
        if decode is not None and decode.takes(hidden_states, step, partial_tail):
            return decode.forward(
                self, hidden_states, self.decode_gemvs, partial_tail, push=step.latent_push
            )
        if partial_tail:
            raise RuntimeError("the row-parallel MoE tail needs the decode path to take the step")
        return self.forward_generic(hidden_states, all_rank_num_tokens)

    def forward_generic(
        self, hidden_states: torch.Tensor, all_rank_num_tokens: Optional[List[int]] = None
    ) -> torch.Tensor:
        """The generic path (module docstring). ``all_rank_num_tokens`` is accepted for the layer's call signature;
        without attention data parallelism every rank holds the same tokens."""
        del all_rank_num_tokens
        identity = hidden_states
        router_logits = self.gate.compute_logits(hidden_states)

        def _routed_output():
            routed_in = self._routed_projection(hidden_states, self.routed_expert_down_proj)
            y = self.routed_experts(
                routed_in,
                router_logits,
                self.gate.e_score_correction_bias,
                self.gate.routed_scaling_factor,
            )
            if self._reduce_routed_output:
                return y
            # One rank: the routed result is complete.
            y = self.routed_expert_norm(y)
            return self._routed_projection(y, self.routed_expert_up_proj)

        # Shared experts depend only on the block input, so overlap their GEMMs
        # with the routed chain while a CUDA graph is captured. The shared
        # experts include their output all-reduce on the auxiliary stream. The
        # join below must precede the routed all-reduce: concurrent collectives
        # on different streams can corrupt SYMM_MEM all-reduce state.
        routed_out, shared_out = run_in_parallel(
            _routed_output,
            lambda: self.shared_experts(identity),
            self.moe_main_event,
            self.moe_shared_event,
            self.shared_expert_stream,
        )
        if self._reduce_routed_output:
            routed_latent = self.routed_experts.all_reduce(routed_out)
            routed_latent = self.routed_expert_norm(routed_latent)
            routed_out = self._routed_projection(routed_latent, self.routed_expert_up_proj)
        return routed_out + shared_out
