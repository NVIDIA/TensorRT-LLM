# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VC-Attention operand contract of the prims_ts context kernel.

VC-Attention (arXiv 2609.15810) adds two changes on the value side of
attention: a token permutation that groups similar value rows into the same
128-token K/V tile, and per-tile value smoothing. Each V tile is stored as E4M3
residuals around its tile mean; the kernel restores the mean inside the online
softmax as ``O += rowsum(P_tile) * mean_tile`` (bf16 K=16 UMMA steps), so the
residual quantization error no longer carries the value DC component. P is
E4M3 through the ExpCast path in both recipes of the Q/K side:

* VC-Attention-QK16 keeps Q and K in bf16 (plan ``q_dtype`` bf16);
* VC-Attention-QK8 (plan ``q_dtype`` E4M3) centres K by its per-head channel
  mean, rotates Q and K by the normalised 128-point Hadamard matrix and
  quantizes them to E4M3 with one scale per 128-token Q block and per
  128-token K tile, as the paper does.

This module holds what the kernel and the plan need: the tile constants, the
mean operand shapes, ``VCAttentionConfig`` and ``VCAttentionParams`` with
their validation, the repair tile arithmetic and ``vc_reference``, the fp32
attention over the dequantized operands. The quantizer that produces the
operands lives in ``quantization`` beside this module and its CuTe DSL
kernels in ``prepare``.

The kernel takes the means and the per-channel V scale through
``BatchPrefillTSWrapper.run`` (``vc=VCAttentionParams(...)``), which folds the
scale into ``output_scale``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .quantization import VCAttentionOperands


# The head dimension the preparation kernels and the attention plans support.
VC_HEAD_DIM = 128
# The K/V tile whose value mean the kernel restores, its only supported size.
VC_K_BLOCK_SIZE = 128
# Q tokens per E4M3 Q scale of VC-Attention-QK8, powers of two up to the Q tile.
# K of the bf16 UMMA mean step, one kind::f16 K step.
VC_MEAN_MMA_K = 16
# Mean UMMA steps issued back to back per group of tiles; each tile takes two of a
# step's 16 K slots (bf16 hi/lo row sum).
VC_MEAN_OPERANDS = 2
VC_MEAN_GROUP_TILES = VC_MEAN_OPERANDS * VC_MEAN_MMA_K // 2
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def vc_mean_group_shape(head_dim: int) -> tuple[int, int]:
    """Shape of the packed mean operands of one tile group: the ``VC_MEAN_OPERANDS``
    operands of ``vc_mean_operand_shape`` stacked along the row axis."""
    rows, cols = vc_mean_operand_shape(head_dim)
    return (VC_MEAN_OPERANDS * rows, cols)


def vc_mean_operand_shape(head_dim: int) -> tuple[int, int]:
    """Shape of one packed bf16 ``[head_dim x VC_MEAN_MMA_K]`` mean operand, in
    rows of 256 elements (the kernel's 512-byte TMA box row)."""
    return (head_dim * VC_MEAN_MMA_K // 256, 256)


@dataclass(frozen=True)
class VCAttentionConfig:
    """Compile-time VC-Attention recipe of one plan.

    With demean on, the default, the kernel restores the tile means. Each
    128-token K/V tile stores E4M3 residuals around its bf16 mean and the
    kernel adds the mean back. With demean off the means are zero and the
    kernel skips that step. This is the plan for the steps after the V-Smooth
    window. The demean flag changes per denoising step while a plan is
    compiled once, so a caller holds one plan of each kind.

    A positive repair tile count selects V repair and needs demean off. The
    preprocessor appends that many 128-row tiles of the largest quantization
    residuals after the sequence and the kernel adds them to the output. Each
    repair tile costs one extra K/V tile of attention work. Budgets of two to
    five percent of the sequence were used so far and the count cannot exceed
    the number of sequence tiles.

    Under VC-Attention-QK8, the plan with E4M3 Q and K, every 128-token Q
    block shares one E4M3 Q scale and every 128-token K/V tile one K scale.
    """

    demean: bool = True
    repair_tiles: int = 0

    def __post_init__(self) -> None:
        if self.repair_tiles < 0:
            raise ValueError(f"repair_tiles must be >= 0, got {self.repair_tiles}")
        if self.repair_tiles and self.demean:
            raise ValueError("repair_tiles requires demean=False")


def vc_repair_tiles(seq_len_kv: int, budget: float) -> int:
    """Repair tiles of a V repair run, whole 128-row tiles holding the selected
    tokens, ``budget`` times the sequence length rounded and at least one."""
    if not (0.0 < budget < 1.0):
        raise ValueError(f"budget must be in (0, 1), got {budget}")
    return _blocks(max(1, round(budget * seq_len_kv)), VC_K_BLOCK_SIZE)


def vc_repair_kv_len(seq_len_kv: int, repair_tiles: int) -> int:
    """K/V rows of a V repair run, the sequence zero-padded to whole tiles plus
    the repair tiles."""
    return (_blocks(seq_len_kv, VC_K_BLOCK_SIZE) + repair_tiles) * VC_K_BLOCK_SIZE


@dataclass(frozen=True, kw_only=True)
class VCAttentionParams:
    """Per-run VC-Attention operands beyond ``q``, the permuted ``k`` and the
    E4M3 ``v`` residuals.

    VC-Attention-QK8 plans take ``q_scale`` ``[Hq, B, ceil(Sq / 128)]`` and
    ``k_scale`` ``[Hkv, B, ceil(Skv / 128)]``, fp32, one scale per 128-token
    block. QK16 plans take neither. ``v_scale`` is the ``[B, Hkv, D]`` fp32
    per-channel E4M3 residual scale. ``tile_means`` is the packed bf16 mean
    operand ``[B, Hkv, ceil(num_kv_tiles / 16), 16, 256]`` the preparation
    writes, the means already divided by ``v_scale``. The scale must be positive
    and finite. The kernel does not check it. A plan with demean off takes no
    ``tile_means``.
    """

    q_scale: torch.Tensor | None = None
    k_scale: torch.Tensor | None = None
    v_scale: torch.Tensor
    tile_means: torch.Tensor | None


def vc_scale_shapes(
    config: VCAttentionConfig,
    *,
    batch_size: int,
    seq_len_kv: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len_q: int | None = None,
    num_qo_heads: int | None = None,
    qk_fp8: bool = False,
) -> dict[str, tuple[int, ...]]:
    """Return the shape of every ``VCAttentionParams`` tensor a plan consumes, by
    field name. ``qk_fp8`` adds the Q/K scales of VC-Attention-QK8, which need
    ``seq_len_q`` and ``num_qo_heads``."""
    num_kv_tiles = _blocks(seq_len_kv, VC_K_BLOCK_SIZE)
    if config.repair_tiles:
        # The repair tiles follow the whole tiles.
        seq_len_kv = vc_repair_kv_len(seq_len_kv, config.repair_tiles)
    shapes: dict[str, tuple[int, ...]] = {}
    if qk_fp8:
        if seq_len_q is None or num_qo_heads is None:
            raise ValueError("qk_fp8 scale shapes need seq_len_q and num_qo_heads")
        shapes["q_scale"] = (
            num_qo_heads,
            batch_size,
            _blocks(seq_len_q, VC_K_BLOCK_SIZE),
        )
        shapes["k_scale"] = (
            num_kv_heads,
            batch_size,
            _blocks(seq_len_kv, VC_K_BLOCK_SIZE),
        )
    shapes["v_scale"] = (batch_size, num_kv_heads, head_dim)
    if config.demean:
        shapes["tile_means"] = (
            batch_size,
            num_kv_heads,
            _blocks(num_kv_tiles, VC_MEAN_GROUP_TILES),
            *vc_mean_group_shape(head_dim),
        )
    return shapes


_VC_PARAM_DTYPES = {
    "q_scale": torch.float32,
    "k_scale": torch.float32,
    "v_scale": torch.float32,
    "tile_means": torch.bfloat16,
}


def validate_vc_params(
    params: VCAttentionParams,
    expected_shapes: dict[str, tuple[int, ...]],
    *,
    device: torch.device,
) -> None:
    """Validate the operands of one run against the plan's expected shapes.

    ``expected_shapes`` is the plan's ``vc_scale_shapes`` result.
    """
    if not isinstance(params, VCAttentionParams):
        raise TypeError("vc must be a VCAttentionParams instance")
    if "tile_means" not in expected_shapes and params.tile_means is not None:
        raise ValueError("tile_means must be None on a plan with demean off")
    for name, expected_shape in expected_shapes.items():
        tensor = getattr(params, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != _VC_PARAM_DTYPES[name]:
            raise ValueError(
                f"{name} must have dtype {_VC_PARAM_DTYPES[name]}, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} data pointer must be 16-byte aligned")
        if tensor.device != device:
            raise ValueError(f"{name} must be on device {device}, got {tensor.device}")


def _blocks(length: int, block: int) -> int:
    return (length + block - 1) // block


def _per_token_scales(scale: torch.Tensor, s: int, block: int) -> torch.Tensor:
    """Expand ``[B, H, nb]`` block scales to ``[B, S, H]``."""
    return scale.permute(0, 2, 1).repeat_interleave(block, dim=1)[:, :s]


@torch.no_grad()
def vc_reference(
    q: torch.Tensor, ops: VCAttentionOperands, *, sm_scale: float | None = None
) -> torch.Tensor:  # noqa: D401
    """fp32 attention of ``q`` over the dequantized VC operands (what the kernel
    computes, up to E4M3 P rounding), as ``[B, S_q, H, D]`` fp32. Under
    VC-Attention-QK8 the rotated E4M3 ``ops.q`` and ``ops.k`` are dequantized
    with their scales and ``q`` is unused."""
    b, s_k, h, d = ops.k.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    if ops.q_scale is not None:
        s_q = ops.q.shape[1]
        q_scale = ops.q_scale.permute(1, 0, 2)
        k_scale = ops.k_scale.permute(1, 0, 2)
        q = ops.q.float() * _per_token_scales(q_scale, s_q, VC_K_BLOCK_SIZE).unsqueeze(
            -1
        )
        k = ops.k.float() * _per_token_scales(k_scale, s_k, VC_K_BLOCK_SIZE).unsqueeze(
            -1
        )
    else:
        q = q.float()
        k = ops.k.float()
    v = ops.v.float() * ops.v_scale.unsqueeze(1)
    repairs_v = ops.repair_tiles > 0
    if repairs_v:
        # Repair rows add to the numerator only, the zero padding of the partial
        # last tile is masked, and the denominator runs over the original tokens.
        repair = ops.repair_tiles * VC_K_BLOCK_SIZE
        whole = ops.seq_len_kv // VC_K_BLOCK_SIZE * VC_K_BLOCK_SIZE
        end = ops.seq_len_kv + repair
        row = torch.arange(s_k, device=k.device)
        original = (row < whole) | ((row >= whole + repair) & (row < end))
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale
        scores = scores.masked_fill(row >= end, -torch.inf)
        weights = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        out = torch.einsum("bhqk,bkhd->bqhd", weights, v)
        denom = weights[..., original].sum(dim=-1).permute(0, 2, 1)  # [B, Sq, H]
        return out / denom.unsqueeze(-1)
    # The kernel restores bf16(mean / v_scale) * v_scale, per channel.
    vs = ops.v_scale.reshape(b, ops.k.shape[2], 1, d)  # [B, H, 1, D]
    mean = (ops.mean / vs).to(torch.bfloat16).float() * vs
    mean_tokens = mean.permute(0, 2, 1, 3).repeat_interleave(VC_K_BLOCK_SIZE, dim=1)[
        :, :s_k
    ]
    v = v + mean_tokens
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


__all__ = [
    "E4M3_MAX",
    "VC_HEAD_DIM",
    "VC_K_BLOCK_SIZE",
    "VC_MEAN_MMA_K",
    "VCAttentionConfig",
    "VCAttentionParams",
    "validate_vc_params",
    "vc_repair_kv_len",
    "vc_repair_tiles",
    "vc_mean_group_shape",
    "vc_mean_operand_shape",
    "VC_MEAN_GROUP_TILES",
    "VC_MEAN_OPERANDS",
    "vc_reference",
    "vc_scale_shapes",
]
