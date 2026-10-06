# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Token-sharded tensor parallelism (Megatron-style sequence parallelism) for VisualGen DiT blocks.

With plain tensor parallelism (TP) every row-parallel projection ends in an
all-reduce and every rank holds the full, replicated residual stream. With
``parallel_config.tp_layout='token_sharded'`` the residual stream between projections is
instead *token-sharded* across the TP group:

Layout
    The ``B`` samples of ``S`` tokens are padded per sample to ``S_pad`` tokens
    (``S_pad = S`` unless ``B * S`` is not divisible by ``tp``, see
    :class:`TokenShardPlan`) and flattened to ``[B * S_pad]`` rows. TP rank ``r``
    owns rows ``[r * m, (r + 1) * m)`` with ``m = B * S_pad / tp``.

The three boundaries of a DiT block
    Every all-reduce after a row-parallel projection (attention ``to_out``,
    cross-attention ``to_out``, MLP ``down_proj``) becomes::

        row-parallel GEMM (K-partials, [B*S_pad, N]) -> reduce-scatter -> [m, N]
        row-local residual (+gate), LayerNorm (+AdaLN) (+static NVFP4 quantize) on m rows
        all-gather (bf16, or NVFP4 payload + scaling factors) -> column-parallel GEMM on B*S rows

Invariant
    Only row-local ops (residual adds, norms, modulation, quantization) run on the
    shard. Token-mixing ops (self/cross attention, QK-norm, RoPE) always see the full
    ``[B, S]`` token set with heads sharded exactly as in plain TP, and every GEMM runs
    on all tokens (the row-parallel output projections see ``B * S_pad`` rows when a
    shape is padded).

Numerics
    Token-sharded TP differs from the all-reduce path in its collectives: the reduce-scatter may
    sum the K-partials in a different order *and with a different NCCL algorithm*. With
    NCCL's default tuning at ``tp >= 8`` on NVSwitch systems the all-reduce runs as NVLS
    while the reduce-scatter runs as a ring, which rounds to bf16 after each hop, so
    Token-sharded TP is then measurably less precise than all-reduce TP (``NCCL_ALGO=
    "ReduceScatter:NVLS"`` restores bitwise parity at a latency cost). Row-local norms
    reuse the model's own kernels (the fused op at ``D == 5120``, the model's LayerNorm
    module through ``RowNorm.module`` otherwise).

This is not Ulysses or Ring attention (``ulysses_size`` / ``ring_size`` /
``attn2d_size``): those shard the sequence *through* attention; this only shards it
*between* projections inside a TP group, and is exclusive with them for now.

Two API layers:

* primitives: :class:`TokenShardPlan`, ``shard`` / ``unshard``, modulation tables,
  ``reduce_scatter``, ``all_gather`` (bf16 or NVFP4 with scaling-factor regroup), and
  row-local ``residual`` / ``norm`` / :func:`quantize_nvfp4`;
* GEMM-owning boundary ops: ``column_linear``, ``row_linear``,
  ``row_linear_residual_norm``, ``mlp_residual``. These are the only call sites a
  communication/compute overlap or a fused GEMM+collective kernel has to replace.

See ``TOKEN_SHARDED_TP_DEVELOPER_GUIDE.md`` next to this file.
"""

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

from ...modules.linear import (
    Linear,
    NVFP4LinearMethod,
    TensorParallelMode,
    is_static_nvfp4_input_eligible,
)
from ...utils import Fp4QuantizedTensor, compute_swizzled_sf_shape
from ..modules.fused_norm_quant import (
    apply_fused_layernorm_adaln_quant,
    apply_fused_layernorm_affine_quant,
)

if TYPE_CHECKING:
    from ..config import DiffusionModelConfig

__all__ = [
    "NVFP4_SF_VEC_SIZE",
    "RowNorm",
    "TokenShardedTP",
    "TokenShardPlan",
    "apply_residual",
    "apply_row_norm",
    "quantize_nvfp4",
    "regroup_swizzled_sf",
    "static_nvfp4_input_scale",
    "swizzled_sf_numel",
]

NVFP4_SF_VEC_SIZE = 16
# Hidden size supported by torch.ops.trtllm.fused_adaptive_layernorm(_quant).
_FUSED_LN_HIDDEN_SIZE = 5120

# A row-parallel boundary input / activation: bf16 rows or a static-scale NVFP4 tensor.
Activation = torch.Tensor | Fp4QuantizedTensor
# A GEMM (column- or row-parallel): a TRT-LLM Linear or any callable with the same contract.
GemmFn = Callable[[Activation], torch.Tensor]


# =============================================================================
# Plan (pure Python ints: CPU-testable, torch.compile / CUDA-graph safe)
# =============================================================================


@dataclass(frozen=True)
class TokenShardPlan:
    """Static token-shard plan for one ``(batch_size, seq_len)`` on one TP rank.

    Python ints only, so it is compile/graph safe; cached per shape by
    :meth:`TokenShardedTP.begin`.

    Padding rule: with ``d = gcd(tp, B)`` and ``t' = tp / d`` every sample is padded at
    its end to ``S_pad = round_up(S, t')`` tokens. Then every aligned group of
    ``rows_per_entry = S_pad / t'`` local rows lies inside one sample, so a
    per-sample modulation table needs only ``B / d <= B`` entries on every rank
    (``entry_batch``), for every shape. Padding happens iff ``B * S % tp != 0``.
    """

    batch_size: int  # B
    seq_len: int  # S (real tokens per sample)
    padded_seq_len: int  # S_pad = round_up(S, t'), t' = tp // gcd(tp, B)
    tp_size: int
    tp_rank: int
    local_rows: int  # m = B * S_pad // tp
    row_start: int  # tp_rank * m, in flat [B * S_pad] order
    rows_per_entry: int  # g = S_pad // t' (the fused AdaLN op's seq_len_per_batch)
    entry_batch: tuple[int, ...]  # sample index of each modulation entry; len B // gcd(tp, B)

    @property
    def num_tokens(self) -> int:
        """Real tokens B * S (rows of a gathered activation)."""
        return self.batch_size * self.seq_len

    @property
    def padded_rows(self) -> int:
        """Rows of the padded stream, B * S_pad (rows of a reduce-scatter input)."""
        return self.batch_size * self.padded_seq_len

    @property
    def is_padded(self) -> bool:
        return self.padded_seq_len != self.seq_len

    @staticmethod
    def build(batch_size: int, seq_len: int, tp_size: int, tp_rank: int) -> "TokenShardPlan":
        if batch_size < 1 or seq_len < 1:
            raise ValueError(
                f"TokenShardPlan needs batch_size >= 1 and seq_len >= 1 "
                f"(got batch_size={batch_size}, seq_len={seq_len})."
            )
        if tp_size < 1 or not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"TokenShardPlan needs 0 <= tp_rank < tp_size (got tp_rank={tp_rank}, "
                f"tp_size={tp_size})."
            )
        d = math.gcd(tp_size, batch_size)
        t = tp_size // d
        s_pad = pad_up(seq_len, t)
        m = batch_size * s_pad // tp_size  # exact: tp | B * s_pad
        g = s_pad // t
        row_start = tp_rank * m
        entry_batch = tuple((row_start + j * g) // s_pad for j in range(m // g))
        return TokenShardPlan(
            batch_size=batch_size,
            seq_len=seq_len,
            padded_seq_len=s_pad,
            tp_size=tp_size,
            tp_rank=tp_rank,
            local_rows=m,
            row_start=row_start,
            rows_per_entry=g,
            entry_batch=entry_batch,
        )

    def local_segments(self) -> tuple[tuple[int, int, int], ...]:
        """``(b, s0, s1)`` pieces in padded coordinates covering this rank's rows, in order.

        Sample ``b``'s padded tokens ``[s0, s1)`` (``s1 <= S_pad``); tokens ``>= S`` are
        padding. At most ``B + 1`` pieces.
        """
        s_pad = self.padded_seq_len
        row, end = self.row_start, self.row_start + self.local_rows
        segments = []
        while row < end:
            b, s0 = divmod(row, s_pad)
            s1 = min(s_pad, s0 + end - row)
            segments.append((b, s0, s1))
            row += s1 - s0
        return tuple(segments)


# =============================================================================
# Pure row-local and layout functions (no process group)
# =============================================================================


def swizzled_sf_numel(rows: int, sf_cols: int) -> int:
    """Elements of a 128x4-swizzled NVFP4 scaling-factor buffer: pad128(rows) * pad4(sf_cols)."""
    padded_rows, padded_cols = compute_swizzled_sf_shape(rows, sf_cols)
    return padded_rows * padded_cols


def regroup_swizzled_sf(sf_cat: torch.Tensor, plan: TokenShardPlan, sf_cols: int) -> torch.Tensor:
    """Re-tile ``tp`` gathered per-rank swizzled SF buffers into one buffer for ``B * S`` rows.

    Each rank's buffer is the 128x4 layout for ``pad128(m)`` rows (pad rows
    uninitialized), where element ``(row, k)`` sits at
    ``[row // 128][k // 4][row % 32][(row % 128) // 32][k % 4]``. The result is the same
    layout for the ``B * S`` real rows: each rank's 128-row tile padding and the
    per-sample token padding are dropped.

    Zero-copy fast path (a slice of ``sf_cat``) when ``m % 128 == 0`` and the plan is
    unpadded or ``B == 1``: a row's offset does not depend on the total row count.
    Otherwise views/permutes/pads only (one gather-copy kernel under Inductor).
    """
    tp, m = plan.tp_size, plan.local_rows
    if m % 128 == 0 and (not plan.is_padded or plan.batch_size == 1):
        return sf_cat[: swizzled_sf_numel(plan.num_tokens, sf_cols)]
    k4 = pad_up(sf_cols, 4)
    kt = k4 // 4
    t_loc = pad_up(m, 128) // 128
    # [tp, mTile, kTile, m%32, (m%128)//32, k%4] -> linear [tp, t_loc*128, k4] rows
    rows = (
        sf_cat.view(tp, t_loc, kt, 32, 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(tp, t_loc * 128, k4)[:, :m]
    )
    b, s, s_pad = plan.batch_size, plan.seq_len, plan.padded_seq_len
    rows = rows.reshape(b, s_pad, k4)[:, :s].reshape(b * s, k4)  # drop per-sample padding
    rows = F.pad(rows, (0, 0, 0, pad_up(b * s, 128) - b * s))
    return rows.view(-1, 4, 32, kt, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def static_nvfp4_input_scale(linear: nn.Module | None) -> torch.Tensor | None:
    """The consumer's static (calibrated) NVFP4 ``input_scale``, or None.

    Non-None iff ``linear`` quantizes its input to NVFP4 with 16-element blocks from a
    calibrated scale (no AWQ ``pre_quant_scale``, no forced dynamic quantization), i.e.
    iff a row-local quantize with this scale yields exactly the bytes the Linear would
    produce itself, so the activation can be all-gathered as NVFP4. This is the rule
    ``TokenShardedTP`` users (Wan included) wire their ``RowNorm.quant_scale`` from.
    """
    if not is_static_nvfp4_input_eligible(linear):
        return None
    if getattr(linear, "scaling_vector_size", None) != NVFP4_SF_VEC_SIZE:
        return None
    return linear.input_scale


def quantize_nvfp4(h: torch.Tensor, input_scale: torch.Tensor) -> Fp4QuantizedTensor:
    """Static-scale NVFP4 quantize of ``h`` ([..., K] -> [rows, K/2] + swizzled SF).

    Uses the same quantize op as ``NVFP4LinearMethod._input_prepare`` so the bytes match
    what the consuming Linear would produce on the same rows.
    """
    h2 = h.reshape(-1, h.shape[-1]).contiguous()
    if NVFP4LinearMethod.use_tunable_quantize:
        fp4, sf = torch.ops.trtllm.tunable_fp4_quantize(h2, input_scale, NVFP4_SF_VEC_SIZE, False)
    else:
        fp4, sf = torch.ops.trtllm.fp4_quantize(h2, input_scale, NVFP4_SF_VEC_SIZE, False)
    return Fp4QuantizedTensor(fp4, sf, is_sf_swizzled=True)


def _check_table_rows(op: str, rows: int, entries: int) -> None:
    if entries < 1 or rows % entries != 0:
        raise ValueError(
            f"TokenShardedTP.{op}: a per-row-group table of {entries} entries does not "
            f"divide the {rows} local rows; build it with per_sample_table()/shard_rows()."
        )


def apply_residual(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None = None
) -> torch.Tensor:
    """Row-local residual on ``m`` rows: ``x + y``, or ``x + y * gate`` in fp32.

    Args:
        x: Residual rows ``[m, D]``.
        y: Rows to add ``[m, D]`` (same dtype as ``x``).
        gate: Optional ``[n, D]`` table; entry ``j`` applies to local rows
            ``[j * m / n, (j + 1) * m / n)`` (see ``TokenShardedTP.per_sample_table``).

    Returns:
        ``[m, D]`` in ``x.dtype``.
    """
    if gate is None:
        return x + y
    m, d = x.shape
    n = gate.shape[0]
    _check_table_rows("residual", m, n)
    g = m // n
    out = x.float().view(n, g, d) + y.float().view(n, g, d) * gate.float().unsqueeze(1)
    return out.to(x.dtype).view(m, d)


@dataclass(frozen=True, eq=False)
class RowNorm:
    """Row-local LayerNorm spec for a block boundary (see :func:`apply_row_norm`).

    ``y = LN(x) [* weight + bias] [* (1 + scale) + shift]``, then an optional static
    NVFP4 quantize. The fused op covers exactly one of affine (``weight``/``bias``) and
    AdaLN (``scale``/``shift``); a spec with both, or with neither, runs the eager path.

    Attributes:
        eps: LayerNorm epsilon.
        weight: Affine LayerNorm weight ``[D]`` (given together with ``bias``).
        bias: Affine LayerNorm bias ``[D]``.
        scale: AdaLN scale table ``[n, D]``; entry ``j`` applies to local rows
            ``[j * m / n, (j + 1) * m / n)``. Build it with ``per_sample_table`` /
            ``shard_rows``, never from a global ``[B, D]`` table.
        shift: AdaLN shift table ``[n, D]`` (given together with ``scale``).
        quant_scale: The consumer's static NVFP4 ``input_scale``
            (:func:`static_nvfp4_input_scale`); when set the norm returns an
            :class:`Fp4QuantizedTensor`.
        identity: Skip the norm (e.g. Wan ``cross_attn_norm=False``) but still quantize
            when ``quant_scale`` is set.
        module: Optional LayerNorm module (e.g. the model's own ``norm1``) that the eager
            path calls as ``module(x.float())`` instead of ``F.layer_norm``, so it runs the
            same kernel as the model's all-reduce path. It must compute the same function
            as ``eps`` / ``weight`` / ``bias`` (the fused path uses those).
    """

    eps: float = 1e-6
    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    shift: torch.Tensor | None = None
    quant_scale: torch.Tensor | None = None
    identity: bool = False
    module: nn.Module | None = None

    def __post_init__(self) -> None:
        if (self.scale is None) != (self.shift is None):
            raise ValueError("RowNorm: scale and shift must be given together.")
        if (self.weight is None) != (self.bias is None):
            raise ValueError("RowNorm: weight and bias must be given together.")
        if self.identity and (
            self.weight is not None or self.scale is not None or self.module is not None
        ):
            raise ValueError("RowNorm: identity=True excludes weight/bias/scale/shift/module.")


def apply_row_norm(x: torch.Tensor, spec: RowNorm) -> Activation:
    """Row-local LayerNorm (+AdaLN / affine) (+static NVFP4 quantize) on ``x`` [m, D].

    Uses the fused ``fused_adaptive_layernorm(_quant)`` op when it applies (D == 5120,
    bf16, CUDA, exactly one of modulation/affine); otherwise fp32 LayerNorm
    (``spec.module`` if set, else ``F.layer_norm``) with the same math as the Wan block,
    followed by :func:`quantize_nvfp4` when ``spec.quant_scale`` is set.

    Args:
        x: Local rows ``[m, D]``.
        spec: The norm to apply; its tables must have ``n`` entries with ``n | m``.

    Returns:
        ``[m, D]`` in ``x.dtype``, or an :class:`Fp4QuantizedTensor` (payload
        ``[m, D/2]``, 128x4-swizzled SF for ``m`` rows) when ``spec.quant_scale`` is set.
    """
    m, d = x.shape
    modulated = spec.scale is not None
    affine = spec.weight is not None
    if spec.identity:
        h = x
    elif (
        d == _FUSED_LN_HIDDEN_SIZE
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and modulated != affine
    ):
        if modulated:
            n = spec.scale.shape[0]
            _check_table_rows("norm", m, n)
            return apply_fused_layernorm_adaln_quant(
                x, spec.scale, spec.shift, m // n, spec.quant_scale, spec.eps
            )
        return apply_fused_layernorm_affine_quant(
            x, spec.weight, spec.bias, spec.quant_scale, spec.eps
        )
    else:
        if spec.module is not None:
            y = spec.module(x.float())
        else:
            y = F.layer_norm(
                x.float(),
                (d,),
                spec.weight.float() if affine else None,
                spec.bias.float() if affine else None,
                spec.eps,
            )
        if modulated:
            n = spec.scale.shape[0]
            _check_table_rows("norm", m, n)
            g = m // n
            y = y.view(n, g, d) * (
                1 + spec.scale.float().unsqueeze(1)
            ) + spec.shift.float().unsqueeze(1)
            y = y.view(m, d)
        h = y.to(x.dtype)
    return quantize_nvfp4(h, spec.quant_scale) if spec.quant_scale is not None else h


# =============================================================================
# Private collective seam (the only place a communication primitive is chosen)
# =============================================================================


def _wait(t: torch.Tensor) -> torch.Tensor:
    # Eager: a plain tensor (no AsyncCollectiveTensor leaking into trtllm ops).
    # Traced: an explicit wait op.
    return t.wait() if isinstance(t, funcol.AsyncCollectiveTensor) else funcol.wait_tensor(t)


def _all_gather_rows(x_loc: torch.Tensor, group_name: str) -> torch.Tensor:
    return _wait(funcol.all_gather_single(x_loc.contiguous(), 0, group_name))


def _reduce_scatter_rows(y: torch.Tensor, group_name: str) -> torch.Tensor:
    return _wait(funcol.reduce_scatter_single(y.contiguous(), "sum", 0, group_name))


# =============================================================================
# TokenShardedTP
# =============================================================================


class TokenShardedTP:
    """Megatron-style sequence parallelism inside a VisualGen TP group (see module docstring).

    Not an nn.Module (no parameters/state_dict; safe under MetaInitMode). One instance
    per transformer (per token stream); blocks hold a plain reference. The model forward
    calls ``begin(B, S)`` and ``shard`` eagerly before the blocks and ``unshard`` after
    them; the blocks read the cached :class:`TokenShardPlan` through ``plan``.

    Args:
        group: The TP process group (any ``torch.distributed`` group with >= 2 ranks).
            The group rank order is the token-shard order.
        tp_rank: Optional expected rank of this process in ``group``; a mismatch raises
            (catches a mapping whose TP rank differs from the group's rank order).
    """

    def __init__(self, group: dist.ProcessGroup | None, *, tp_rank: int | None = None) -> None:
        if group is None:
            raise ValueError(
                "TokenShardedTP needs a torch.distributed TP process group; got None "
                "(is the VisualGenMapping device mesh initialized?)."
            )
        tp_size = dist.get_world_size(group)
        if tp_size < 2:
            raise ValueError(
                f"TokenShardedTP needs a process group with at least 2 ranks (got {tp_size})."
            )
        actual_rank = dist.get_rank(group)
        if tp_rank is not None and tp_rank != actual_rank:
            raise ValueError(
                f"TokenShardedTP: tp_rank={tp_rank} does not match this process's rank in "
                f"the TP process group ({actual_rank}); a rank's token shard must follow the "
                "group rank order."
            )
        self.group = group
        self.tp_size: int = tp_size
        self.tp_rank: int = actual_rank
        # Resolved once here: compiled blocks pass the name string to the functional
        # collectives instead of looking up the group (a compiler-disabled mesh path).
        self.group_name: str = group.group_name
        self._plans: dict[tuple[int, int], TokenShardPlan] = {}
        self._plan: TokenShardPlan | None = None

    @classmethod
    def from_model_config(cls, model_config: "DiffusionModelConfig") -> "TokenShardedTP | None":
        """The helper for a VisualGen model, or None unless ``tp_layout`` is ``'token_sharded'``.

        Re-validates the mapping and cache backend for callers that build a
        ``DiffusionModelConfig`` directly (the ``VisualGenArgs`` validators cover the
        user-facing path).
        """
        parallel = getattr(model_config, "parallel", None)
        if not getattr(parallel, "token_sharded_tp", False):
            return None
        vgm = model_config.visual_gen_mapping
        if vgm is None:
            raise ValueError(
                "TokenShardedTP: tp_layout='token_sharded' needs a VisualGenMapping "
                "(model_config.visual_gen_mapping is None)."
            )
        if vgm.tp_size <= 1 or vgm.seq_size > 1:
            raise ValueError(
                "TokenShardedTP: tp_layout='token_sharded' needs a VisualGenMapping "
                f"with tp_size > 1 and seq_size == 1 (got tp_size={vgm.tp_size}, "
                f"seq_size={vgm.seq_size}: ulysses={vgm.ulysses_size}, ring={vgm.ring_size}, "
                f"attn2d={vgm.attn2d_row_size}x{vgm.attn2d_col_size})."
            )
        if model_config.cache_backend == "cache_dit":
            raise ValueError(
                "TokenShardedTP: tp_layout='token_sharded' does not support "
                "cache_backend='cache_dit' (per-block skip decisions would see "
                "token-sharded hidden states)."
            )
        return cls(vgm.tp_group_pg, tp_rank=vgm.tp_rank)

    # --- plan -----------------------------------------------------------------------

    def begin(self, batch_size: int, seq_len: int) -> TokenShardPlan:
        """Select (and cache) the plan for this forward's ``(B, S)``; call eagerly.

        The first use of a shape checks with one ``all_gather_object`` that all TP ranks
        run the same shape (a mismatch would otherwise hang or corrupt the NCCL
        collectives). Later calls with a cached shape skip the check, so a rank that
        reuses a cached shape while a peer runs a new one is not detected.

        Args:
            batch_size: ``B``, samples in this forward (e.g. 2 for batched CFG).
            seq_len: ``S``, tokens per sample.

        Returns:
            This rank's :class:`TokenShardPlan`, also available as ``self.plan``.
        """
        key = (batch_size, seq_len)
        plan = self._plans.get(key)
        if plan is None:
            plan = TokenShardPlan.build(batch_size, seq_len, self.tp_size, self.tp_rank)
            self._check_rank_agreement(batch_size, seq_len)
            if plan.is_padded:
                extra = batch_size * (plan.padded_seq_len - seq_len)
                logger.info_once(
                    f"Token-sharded TP: {batch_size}x{seq_len} tokens do not split evenly "
                    f"over tp_size={self.tp_size}; padding each sample to "
                    f"{plan.padded_seq_len} tokens ({extra} extra rows; adds copies at each "
                    "block boundary).",
                    key=("token_sharded_tp_padding", batch_size, seq_len, self.tp_size),
                )
            self._plans[key] = plan
        self._plan = plan
        return plan

    def _check_rank_agreement(self, batch_size: int, seq_len: int) -> None:
        shapes = [None] * self.tp_size
        dist.all_gather_object(shapes, (batch_size, seq_len), group=self.group)
        if any(tuple(s) != (batch_size, seq_len) for s in shapes):
            layout = [(rank, b, s) for rank, (b, s) in enumerate(shapes)]
            raise ValueError(
                f"TokenShardedTP: TP ranks disagree on the token layout {layout}; all "
                "ranks of a TP group must run the transformer on identically shaped inputs."
            )

    @property
    def plan(self) -> TokenShardPlan:
        if self._plan is None:
            raise RuntimeError(
                "TokenShardedTP.begin(batch_size, seq_len) must be called (in the model "
                "forward) before the first block runs."
            )
        return self._plan

    # --- shape checks -----------------------------------------------------------------

    def _plan_desc(self) -> str:
        p = self.plan
        return f"B={p.batch_size}, S={p.seq_len}, tp={p.tp_size}"

    def _rows_error(self, op: str, rows: int) -> ValueError:
        p = self.plan
        expected = str(p.num_tokens)
        if p.is_padded:
            expected += f" (B * S) or {p.padded_rows} (B * S_pad)"
        return ValueError(
            f"TokenShardedTP.{op}: expected {expected} rows for the current plan "
            f"({self._plan_desc()}); got {rows}."
        )

    def _check_local_rows(self, op: str, t: torch.Tensor) -> None:
        m = self.plan.local_rows
        if t.dim() != 2 or t.shape[0] != m:
            raise ValueError(
                f"TokenShardedTP.{op}: expected this rank's [{m}, K] rows for the current "
                f"plan ({self._plan_desc()}); got shape {tuple(t.shape)}. Did you forget "
                "shard(), or call begin() for another shape in between?"
            )

    def _check_table(self, op: str, name: str, table: torch.Tensor | None) -> None:
        # Python-int shape check (a compile-time guard under torch.compile).
        if table is None:
            return
        p = self.plan
        if table.shape[0] not in (len(p.entry_batch), p.local_rows):
            raise ValueError(
                f"TokenShardedTP.{op}: {name} has {table.shape[0]} entries; the current "
                f"plan ({self._plan_desc()}) expects this shard's per-sample table "
                f"({len(p.entry_batch)} entries, per_sample_table()) or per-row table "
                f"({p.local_rows} rows, shard_rows()). A global [B, ...] table would "
                "silently apply other samples' modulation on a shard."
            )

    # --- entry / exit / per-token metadata (once per forward) ---------------------------

    def _local_pieces(self, t: torch.Tensor, op: str) -> list[torch.Tensor]:
        p = self.plan
        if t.dim() < 2 or tuple(t.shape[:2]) != (p.batch_size, p.seq_len):
            raise ValueError(
                f"TokenShardedTP.{op}: expected a [B={p.batch_size}, S={p.seq_len}, ...] "
                f"tensor for the current plan; got shape {tuple(t.shape)}."
            )
        pieces = []
        for b, s0, s1 in p.local_segments():
            if s0 < p.seq_len:
                pieces.append(t[b, s0 : min(s1, p.seq_len)])
            n_pad = s1 - max(s0, p.seq_len)
            if n_pad > 0:
                pieces.append(t.new_zeros((n_pad, *t.shape[2:])))
        return pieces

    def shard(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, D]`` (any strides) -> this rank's contiguous ``[m, D]`` rows.

        Only the rank's ``m`` rows are copied; pad rows are zeros.
        """
        pieces = self._local_pieces(x, "shard")
        return pieces[0].contiguous() if len(pieces) == 1 else torch.cat(pieces)

    def unshard(self, x_loc: torch.Tensor) -> torch.Tensor:
        """``[m, D]`` -> ``[B, S, D]`` (all-gather, padding dropped). Once per forward."""
        p = self.plan
        self._check_local_rows("unshard", x_loc)
        out = self._drop_padding(_all_gather_rows(x_loc, self.group_name))
        return out.view(p.batch_size, p.seq_len, -1)

    def local_view(self, t: torch.Tensor) -> torch.Tensor:
        """This rank's ``[m, *rest]`` rows -> ``[n, g, *rest]`` sample groups (a view).

        ``n = len(plan.entry_batch)`` and ``g = plan.rows_per_entry``: every group lies inside
        one sample, so a per-sample ``[n, 1, D]`` table (:meth:`per_sample_table`) broadcasts
        over the shard as a ``[B, 1, D]`` table does over ``[B, S, D]``.
        """
        p = self.plan
        return t.view(len(p.entry_batch), p.rows_per_entry, *t.shape[1:])

    def gather_input(self, consumer: nn.Module | None, act: Activation) -> Activation:
        """A column projection's input: this rank's rows -> all ``B * S`` real rows.

        ``act`` is ``[n, g, K]`` or ``[m, K]``: bf16, or an :class:`Fp4QuantizedTensor`
        quantized with ``consumer``'s static scale. A bf16 input is quantized here when
        ``consumer`` has a static NVFP4 input scale (decided per call: the scale exists only
        after loading), so the all-gather moves NVFP4.
        """
        if isinstance(act, Fp4QuantizedTensor):
            payload = act.fp4_tensor
            act = dataclasses.replace(act, fp4_tensor=payload.reshape(-1, payload.shape[-1]))
        else:
            act = act.reshape(-1, act.shape[-1])
            scale = static_nvfp4_input_scale(consumer)
            if scale is not None:
                act = quantize_nvfp4(act, scale)
        self._check_fp4_consumer(consumer, act, "gather_input", "the consuming projection")
        return self.all_gather(act)

    def pad_row_input(self, act: torch.Tensor) -> torch.Tensor:
        """A row projection's input for all tokens -> ``[B * S_pad, K]`` (zero rows per sample).

        ``act`` is ``[B, S, K]`` or ``[B * S, K]``; the padded stream is not accepted.
        """
        p = self.plan
        if tuple(act.shape[:-1]) not in ((p.batch_size, p.seq_len), (p.num_tokens,)):
            raise ValueError(
                f"TokenShardedTP.pad_row_input: expected a [B={p.batch_size}, S={p.seq_len}, K] "
                f"or [B * S={p.num_tokens}, K] input for the current plan; got shape "
                f"{tuple(act.shape)}."
            )
        return self._add_padding(act)

    def shard_rows(self, t: torch.Tensor) -> torch.Tensor:
        """``[B, S, *rest]`` per-token metadata -> ``[m, *rest]`` (zeros for pad rows).

        Returns a view when the rank's rows lie in one sample and are unpadded.
        """
        pieces = self._local_pieces(t, "shard_rows")
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def per_sample_table(self, t: torch.Tensor) -> torch.Tensor:
        """``[B, *rest]`` per-sample table -> this shard's ``[n_entries, *rest]`` table.

        Entry ``j`` applies to local rows ``[j * g, (j + 1) * g)`` with
        ``g = plan.rows_per_entry``. Built with slices/cat only (no index tensor), so it is
        CUDA-graph capture safe. Never pass the global ``[B, ...]`` table to a row-local
        op on a shard.
        """
        eb = self.plan.entry_batch
        b0, n = eb[0], len(eb)
        if eb == tuple(range(b0, b0 + n)):
            return t[b0 : b0 + n]
        return torch.cat([t[b : b + 1] for b in eb])

    def _drop_padding(self, t2d: torch.Tensor) -> torch.Tensor:
        # [B * S_pad, K] -> [B * S, K]
        p = self.plan
        if not p.is_padded:
            return t2d
        if p.batch_size == 1:
            return t2d[: p.seq_len]  # contiguous prefix view
        return t2d.view(p.batch_size, p.padded_seq_len, -1)[:, : p.seq_len].reshape(
            p.num_tokens, -1
        )

    def _add_padding(self, t: torch.Tensor) -> torch.Tensor:
        # [B * S, K] or [B, S, K] -> [B * S_pad, K]
        p = self.plan
        if not p.is_padded:
            return t.reshape(p.num_tokens, -1)
        return F.pad(
            t.reshape(p.batch_size, p.seq_len, -1), (0, 0, 0, p.padded_seq_len - p.seq_len)
        ).reshape(p.padded_rows, -1)

    # --- primitives -------------------------------------------------------------------

    def reduce_scatter(self, partial: torch.Tensor) -> torch.Tensor:
        """K-partial sums -> this rank's reduced ``[m, N]`` rows.

        Args:
            partial: ``[B * S_pad, N]`` (padded stream) or ``[B * S, N]`` / ``[B, S, N]``
                (padded here) partial sums of a row-parallel GEMM.

        Returns:
            ``[m, N]``.
        """
        p = self.plan
        n = partial.shape[-1]
        rows = partial.numel() // n if n else 0
        if rows == p.padded_rows:
            partial = partial.reshape(rows, n)
        elif rows == p.num_tokens:
            partial = self._add_padding(partial)
        else:
            raise self._rows_error("reduce_scatter", rows)
        return _reduce_scatter_rows(partial, self.group_name)

    def all_gather(self, act_loc: Activation) -> Activation:
        """This rank's ``[m, K]`` rows -> all ``[B * S, K]`` rows (padding dropped).

        Args:
            act_loc: ``[m, K]`` bf16/fp32 rows, or an :class:`Fp4QuantizedTensor` from a
                static-scale quantize (payload ``[m, K/2]``, 128x4-swizzled SF for ``m``
                rows), which is gathered as NVFP4.

        Returns:
            ``[B * S, K]``, or an :class:`Fp4QuantizedTensor` with payload
            ``[B * S, K/2]`` and the SF regrouped for ``B * S`` rows.
        """
        p = self.plan
        if isinstance(act_loc, Fp4QuantizedTensor):
            payload, sf, k = self._check_fp4(act_loc)
            payload = self._drop_padding(_all_gather_rows(payload, self.group_name))
            sf = regroup_swizzled_sf(
                _all_gather_rows(sf, self.group_name), p, k // NVFP4_SF_VEC_SIZE
            )
            return Fp4QuantizedTensor(payload, sf, is_sf_swizzled=True)
        self._check_local_rows("all_gather", act_loc)
        return self._drop_padding(_all_gather_rows(act_loc, self.group_name))

    def _check_fp4(self, a: Fp4QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        m = self.plan.local_rows
        if a.reciprocal_scale is not None:
            raise ValueError(
                "TokenShardedTP.all_gather: cannot gather an Fp4QuantizedTensor with a "
                "per-rank dynamic scale (reciprocal_scale is set): each rank used a different "
                "global scale. Gather the BF16 activation, or quantize with the consumer's "
                "static input_scale."
            )
        if a.unquantized_hidden_states is not None:
            raise ValueError(
                "TokenShardedTP.all_gather: the Fp4QuantizedTensor carries a local "
                "unquantized_hidden_states side-car, which cannot be gathered with it; drop "
                "the side-car (gather payload + scaling factors only) or gather the BF16 "
                "activation."
            )
        payload = a.fp4_tensor
        if payload.dtype != torch.uint8 or payload.dim() != 2 or payload.shape[0] != m:
            raise ValueError(
                f"TokenShardedTP.all_gather: NVFP4 payload must be uint8 [{m}, K/2] for "
                f"the current plan ({self._plan_desc()}); got {payload.dtype} "
                f"{tuple(payload.shape)}."
            )
        k = payload.shape[-1] * 2
        expected = swizzled_sf_numel(m, k // NVFP4_SF_VEC_SIZE)
        if not a.is_sf_swizzled or a.scaling_factor.numel() != expected:
            raise ValueError(
                "TokenShardedTP.all_gather: NVFP4 scaling factors must be 128x4-swizzled "
                f"for {m} rows x {k // NVFP4_SF_VEC_SIZE} SF columns ({expected} bytes); got "
                f"{a.scaling_factor.numel()} bytes (is_sf_swizzled={a.is_sf_swizzled})."
            )
        return payload, a.scaling_factor.reshape(-1), k

    def residual(
        self, x_loc: torch.Tensor, y_loc: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Row-local residual ``x + y (* gate)``; see :func:`apply_residual`.

        ``gate`` must be this shard's table (``per_sample_table`` / ``shard_rows``).
        """
        self._check_table("residual", "gate", gate)
        return apply_residual(x_loc, y_loc, gate)

    def norm(self, x_loc: torch.Tensor, spec: RowNorm) -> Activation:
        """Row-local norm (+quantize); see :func:`apply_row_norm`.

        ``spec.scale`` / ``spec.shift`` must be this shard's tables.
        """
        self._check_table("norm", "RowNorm.scale", spec.scale)
        self._check_table("norm", "RowNorm.shift", spec.shift)
        return apply_row_norm(x_loc, spec)

    # --- GEMM-owning boundary ops (the seam later overlap / fused kernels replace) -----

    def _check_linear(self, linear: object, mode: TensorParallelMode, op: str, what: str) -> None:
        """TP mode / size / reduction checks for a TRT-LLM Linear (other callables pass)."""
        if not isinstance(linear, Linear):
            return
        if mode == TensorParallelMode.ROW:
            ok = linear.tp_mode == TensorParallelMode.ROW and not linear.reduce_output
            detail = (
                f"built with reduce_output=False (got tensor_parallel_mode={linear.tp_mode}, "
                f"reduce_output={linear.reduce_output}); an all-reduced output would be "
                "summed again by the reduce-scatter"
            )
        else:
            ok = linear.tp_mode == TensorParallelMode.COLUMN and not linear.gather_output
            detail = (
                f"without gather_output (got tensor_parallel_mode={linear.tp_mode}, "
                f"gather_output={linear.gather_output})"
            )
        if not ok:
            raise ValueError(
                f"TokenShardedTP.{op}: expected {what} to be a {mode.name.lower()}-parallel "
                f"Linear {detail}."
            )
        if linear.tp_size != self.tp_size:
            raise ValueError(
                f"TokenShardedTP.{op}: {what} is sharded for tp_size={linear.tp_size}, "
                f"but the helper's TP group has {self.tp_size} ranks."
            )

    @staticmethod
    def _check_fp4_consumer(consumer: object, act: Activation, op: str, what: str) -> None:
        # An NVFP4-gathered activation is only valid for a consumer that would quantize
        # with the same static input_scale (a Linear silently uses its own alpha).
        if (
            isinstance(act, Fp4QuantizedTensor)
            and isinstance(consumer, Linear)
            and static_nvfp4_input_scale(consumer) is None
        ):
            raise ValueError(
                f"TokenShardedTP.{op}: got an NVFP4 activation, but {what} has no static "
                "NVFP4 input_scale (static_nvfp4_input_scale() is None); gather BF16 for it."
            )

    def column_linear(self, linear: GemmFn, act_loc: Activation) -> torch.Tensor:
        """All-gather ``act_loc`` then run the column-parallel ``linear`` on ``B * S`` rows.

        Args:
            linear: A column-parallel TRT-LLM ``Linear`` (checked: COLUMN, no
                ``gather_output``, same tp_size) or any callable taking ``[B * S, K]``
                (or an :class:`Fp4QuantizedTensor`) and returning a contiguous
                ``[B * S, N_local]``.
            act_loc: This rank's ``[m, K]`` rows (bf16, or static-scale NVFP4 from
                ``norm(..., RowNorm(quant_scale=static_nvfp4_input_scale(linear)))``).

        Returns:
            ``[B, S, N_local]``.
        """
        self._check_linear(linear, TensorParallelMode.COLUMN, "column_linear", "linear")
        self._check_fp4_consumer(linear, act_loc, "column_linear", "linear")
        p = self.plan
        return linear(self.all_gather(act_loc)).view(p.batch_size, p.seq_len, -1)

    def _row_linear(self, linear: GemmFn, act: torch.Tensor, op: str) -> torch.Tensor:
        self._check_linear(linear, TensorParallelMode.ROW, op, "linear")
        p = self.plan
        if tuple(act.shape[:-1]) not in ((p.batch_size, p.seq_len), (p.num_tokens,)):
            raise ValueError(
                f"TokenShardedTP.{op}: expected a [B={p.batch_size}, S={p.seq_len}, K] or "
                f"[B * S={p.num_tokens}, K] input for the current plan; got shape "
                f"{tuple(act.shape)}."
            )
        return self.reduce_scatter(linear(self._add_padding(act)))

    def row_linear(self, linear: GemmFn, act: torch.Tensor) -> torch.Tensor:
        """Row-parallel ``linear`` on all tokens, then reduce-scatter -> ``[m, N]``.

        ``act`` is padded per sample before the GEMM (cheaper than padding the
        ``[B * S, N]`` partial), so the GEMM sees ``B * S_pad`` rows on padded plans; zero
        pad rows leave a dynamic amax unchanged and only carry the bias of tp_rank 0.

        Args:
            linear: A row-parallel TRT-LLM ``Linear`` built with ``reduce_output=False``
                (checked), or any callable returning K-partial sums ``[rows, N]`` with the
                bias added on exactly one rank.
            act: ``[B, S, K_local]`` or ``[B * S, K_local]`` (the rank's K slice of all
                tokens, e.g. an attention output before ``to_out``).

        Returns:
            ``[m, N]`` reduced rows.
        """
        return self._row_linear(linear, act, "row_linear")

    def row_linear_residual_norm(
        self,
        linear: GemmFn,
        act: torch.Tensor,
        residual: torch.Tensor,
        *,
        gate: torch.Tensor | None = None,
        norm: RowNorm | Callable[[torch.Tensor], Activation] | None = None,
    ) -> tuple[torch.Tensor, Activation | None]:
        """``x = residual + row_linear(linear, act) (* gate)``; ``h = norm(x)``.

        Args:
            linear: As for :meth:`row_linear`.
            act: As for :meth:`row_linear`.
            residual: This rank's ``[m, N]`` residual rows.
            gate: Optional gate table for this shard (``per_sample_table`` /
                ``shard_rows``), ``[n, N]``.
            norm: A :class:`RowNorm`, any row-local callable ``[m, N] -> [m, N]`` or
                :class:`Fp4QuantizedTensor` (e.g. a custom RMSNorm that calls
                :func:`quantize_nvfp4`), or None.

        Returns:
            ``(x, h)``: the new ``[m, N]`` residual rows and ``norm(x)`` (None without a
            norm).
        """
        x = self.residual(residual, self._row_linear(linear, act, "row_linear_residual_norm"), gate)
        if norm is None:
            return x, None
        h = self.norm(x, norm) if isinstance(norm, RowNorm) else norm(x)
        return x, h

    def mlp_residual(
        self,
        mlp: GemmFn,
        act_loc: Activation,
        residual: torch.Tensor,
        *,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """All-gather -> ``mlp`` (up, act, down K-partials) -> reduce-scatter -> residual.

        The MLP runs on the ``B * S`` real rows only (pad rows would be non-zero after a
        norm and perturb dynamic-amax quantization inside it); its output partial is
        padded per sample before the reduce-scatter.

        Args:
            mlp: A TRT-LLM ``MLP`` / ``GatedMLP`` (any module whose ``down_proj`` is a
                TRT-LLM ``Linear``: checked to be row-parallel with
                ``reduce_output=False``) or any callable returning K-partial sums
                ``[B * S, N]`` with the bias added on exactly one rank.
            act_loc: This rank's ``[m, K]`` rows (bf16, or static-scale NVFP4 for the
                MLP's input projection).
            residual: This rank's ``[m, N]`` residual rows.
            gate: Optional gate table for this shard, ``[n, N]``.

        Returns:
            The new ``[m, N]`` residual rows.
        """
        self._check_linear(
            getattr(mlp, "down_proj", None), TensorParallelMode.ROW, "mlp_residual", "mlp.down_proj"
        )
        up = getattr(mlp, "up_proj", None) or getattr(mlp, "gate_up_proj", None)
        self._check_fp4_consumer(up, act_loc, "mlp_residual", "the MLP's input projection")
        y = mlp(self.all_gather(act_loc))
        return self.residual(residual, self.reduce_scatter(y), gate)
