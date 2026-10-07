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
"""Token-sharded tensor parallelism for VisualGen DiT blocks (``tp_layout='token_sharded'``).

Megatron-style sequence parallelism inside the TP group: between projections each TP rank holds
only its rows of the residual stream (:class:`TokenShardPlan`). Each row-parallel all-reduce
becomes a reduce-scatter to those rows, and the next column-parallel projection all-gathers them
first (as NVFP4 when that projection has a static NVFP4 input scale, as FP8 plus its 1x128 block
scales when it is an FP8 block-scale Linear on the DeepGEMM path). Only row-local ops
(residual adds, norms, modulation, quantization) run on the shard; attention, QK-norm and RoPE
see all tokens, with heads sharded as in plain TP. This is not Ulysses / ring / attn2d, which
shard the sequence through attention, and cannot be combined with them yet.

Numerics: the reduce-scatter may sum the K-partials in another order and with another NCCL
algorithm than the all-reduce. At ``tp >= 8`` on NVSwitch systems NCCL's default runs the
all-reduce as NVLS but the reduce-scatter as a ring (bf16 rounding per hop), which is measurably
less precise; ``NCCL_ALGO="ReduceScatter:NVLS"`` restores bitwise parity at a latency cost.

A model opts in with ``_supports_token_sharded_tp = True`` and ``self._apply_tp_layout()`` at
the end of ``__init__``, which replaces its ``self.sharder`` with a
:class:`TokenShardedSequenceSharder` and converts the TP modules of its ``self.blocks``
(``token_sharded_modules``).
Its forward routes the residual stream through ``sharder.shard`` / ``gather``, per-token tables
through ``shard(..., expected_seq_len=S)`` and per-sample tables through ``shard_per_sample``,
while ``shard_rope`` leaves RoPE whole. Code between projections must be row-local.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.quantization.utils.fp4_utils import NVFP4_SF_VEC_SIZE

from ...modules.linear import FP8BlockScalesLinearMethod, Linear, is_static_nvfp4_input_eligible
from ...utils import Fp4QuantizedTensor, compute_swizzled_sf_shape
from ..utils import SequenceSharder

if TYPE_CHECKING:
    from ..config import DiffusionModelConfig

__all__ = [
    "FP8_BLOCK_SIZE",
    "Fp8BlockScaledActivation",
    "TokenShardPlan",
    "TokenShardedSequenceSharder",
    "TokenShardedTP",
    "fp8_block_scale_prequant_ok",
    "fp8_scale_cols",
    "fp8_scales_mn_major",
    "quantize_fp8_block",
    "quantize_nvfp4",
    "regroup_swizzled_sf",
    "static_nvfp4_input_scale",
    "swizzled_sf_numel",
]

FP8_BLOCK_SIZE = 128


class Fp8BlockScaledActivation(NamedTuple):
    """A 1x128 FP8 block-scale activation, in the ``(activation, scale)`` tuple form
    ``FP8BlockScalesLinearMethod.apply`` takes as a pre-quantized input.

    Attributes:
        fp8: ``[rows, K]`` float8_e4m3fn.
        scale: ``[rows, P]`` int32, ``P = fp8_scale_cols(K)`` (four UE8M0 scales packed per
            element, one per 128-wide K block; ``K/512`` when 512 divides ``K``), MN-major:
            strides ``(1, pad4(rows))`` as DeepGEMM reads them in eager. Under torch.compile
            Inductor materializes a gathered scale with strides ``(1, rows)``, which the GEMM
            runner re-strides with one small copy.
    """

    fp8: torch.Tensor
    scale: torch.Tensor


# A column projection's input: bf16 rows, a static-scale NVFP4 tensor or an FP8 block-scale
# activation.
Activation = torch.Tensor | Fp4QuantizedTensor | Fp8BlockScaledActivation


# =============================================================================
# Plan
# =============================================================================


@dataclass(frozen=True)
class TokenShardPlan:
    """Which rows of the flattened ``[B * S_pad]`` token stream one TP rank holds.

    Each sample is zero-padded at its end to ``S_pad = round_up(S, t' * row_align)`` with
    ``t' = tp / gcd(tp, B)`` so that every aligned group of ``g`` local rows lies inside one
    sample: the shard views as ``[n, g, D]`` and a per-sample table needs only
    ``n = B / gcd(tp, B)`` entries on every rank. With the default ``row_align = 1`` padding
    happens only when ``B * S % tp != 0``. Python ints only (compile / CUDA-graph safe); each
    new ``(B, S)`` specializes the compiled blocks (eager past Dynamo's ``cache_size_limit``),
    so warm up every served shape.
    """

    batch_size: int  # B
    seq_len: int  # S (real tokens per sample)
    padded_seq_len: int  # S_pad = round_up(S, t' * row_align), t' = tp // gcd(tp, B)
    tp_size: int
    tp_rank: int
    local_rows: int  # m = B * S_pad // tp
    row_start: int  # tp_rank * m, in flat [B * S_pad] order
    rows_per_entry: int  # g = S_pad // t' (the fused AdaLN op's seq_len_per_batch)
    entry_batch: tuple[int, ...]  # sample index of each modulation entry; len B // gcd(tp, B)
    row_align: int = 1  # m is a multiple of this

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
    def build(
        batch_size: int, seq_len: int, tp_size: int, tp_rank: int, row_align: int = 1
    ) -> "TokenShardPlan":
        """Build the plan for one rank.

        Args:
            batch_size: ``B``.
            seq_len: ``S``, real tokens per sample.
            tp_size: Ranks in the TP group.
            tp_rank: This rank.
            row_align: Pad every sample so that each rank's row count ``m`` is a multiple
                of it. Lets a deployment align the shards to a GEMM tile (e.g. 64 rows); at
                4 the FP8 block-scale quantizer's per-rank scale buffers carry no pad rows.
                The default 1 keeps every shape as it is without alignment.
        """
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
        if row_align < 1:
            raise ValueError(f"TokenShardPlan needs row_align >= 1 (got {row_align}).")
        d = math.gcd(tp_size, batch_size)
        t = tp_size // d
        s_pad = pad_up(seq_len, t * row_align)
        m = batch_size * s_pad // tp_size  # exact: tp | B * s_pad; row_align | m
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
            row_align=row_align,
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
# NVFP4 helpers
# =============================================================================


def swizzled_sf_numel(rows: int, sf_cols: int) -> int:
    """Elements of a 128x4-swizzled NVFP4 scaling-factor buffer: pad128(rows) * pad4(sf_cols)."""
    padded_rows, padded_cols = compute_swizzled_sf_shape(rows, sf_cols)
    return padded_rows * padded_cols


def regroup_swizzled_sf(sf_cat: torch.Tensor, plan: TokenShardPlan, sf_cols: int) -> torch.Tensor:
    """Re-tile ``tp`` gathered per-rank swizzled SF buffers into one buffer for ``B * S`` rows.

    Each rank's buffer is the 128x4 layout for ``pad128(m)`` rows, element ``(row, k)`` at
    ``[row // 128][k // 4][row % 32][(row % 128) // 32][k % 4]``; the per-rank tile padding and
    the per-sample token padding are dropped. A slice of ``sf_cat`` (zero-copy) when
    ``m % 128 == 0`` and the plan is unpadded or ``B == 1``, since a row's offset does not
    depend on the total row count.
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
    """``linear``'s static NVFP4 ``input_scale``, or None.

    Non-None iff quantizing each rank's rows with this scale gives exactly the bytes ``linear``
    would produce on all rows (calibrated scale, 16-element blocks, no AWQ ``pre_quant_scale``,
    no forced dynamic quantization), so its input can be all-gathered as NVFP4. Inputs of other
    consumers are gathered as BF16.
    """
    if not is_static_nvfp4_input_eligible(linear):
        return None
    if getattr(linear, "scaling_vector_size", None) != NVFP4_SF_VEC_SIZE:
        return None
    return linear.input_scale


def quantize_nvfp4(h: torch.Tensor, input_scale: torch.Tensor) -> Fp4QuantizedTensor:
    """Static-scale NVFP4 quantize of ``h`` (``[..., K]`` -> ``[rows, K/2]`` + swizzled SF).

    Pinned to ``trtllm::fp4_quantize`` even when VisualGen tunes the Linears' quantize: the
    tunable op may pick FlashInfer's kernel, which shuffles rows and SF in 128-row tiles, while
    the all-gather and :func:`regroup_swizzled_sf` need plain row order.
    """
    h2 = h.reshape(-1, h.shape[-1]).contiguous()
    fp4, sf = torch.ops.trtllm.fp4_quantize(h2, input_scale, NVFP4_SF_VEC_SIZE, False)
    return Fp4QuantizedTensor(fp4, sf, is_sf_swizzled=True)


# =============================================================================
# FP8 block-scale helpers
# =============================================================================


def fp8_block_scale_prequant_ok(linear: nn.Module | None) -> bool:
    """Whether ``linear`` consumes a bf16 input through the DeepGEMM FP8 block-scale path, so
    that a row-local 1x128 quantize yields exactly the bytes it would produce on all rows and
    the activation can be all-gathered as FP8 + scales (an :class:`Fp8BlockScaledActivation`).

    True iff ``linear`` is a ``Linear`` with the ``FP8BlockScalesLinearMethod`` (weights
    created), on an SM100-family GPU, with neither ``use_cute_dsl_blockscaling_mm`` nor
    ``disable_deep_gemm`` (those GEMMs quantize differently and reject the packed scales),
    and with ``in_features`` a multiple of 128 (whole 1x128 blocks; the packed quantize itself
    needs only 16). The column and MLP adapters quantize a bf16 input before their all-gather
    by this rule, decided per call (the quant method can change until the weights are loaded).
    A LoRA is not part of the rule: ``MLP`` attaches a ``LoraLayer`` to its projections whether
    or not adapters are loaded, and only an active ``lora_params`` at call time needs the dense
    input, which the adapters handle with ``gather_input(..., prequantize=False)``.
    """
    if not isinstance(linear, Linear) or not getattr(linear, "_weights_created", False):
        return False
    return (
        type(linear.quant_method) is FP8BlockScalesLinearMethod
        and is_sm_100f()
        and not linear.use_cute_dsl_blockscaling_mm
        and not linear.disable_deep_gemm
        and linear.in_features % FP8_BLOCK_SIZE == 0
    )


def fp8_scale_cols(k: int) -> int:
    """Packed int32 scale columns ``P`` of a 1x128 FP8 block-scale activation with ``K = k``."""
    return (pad_up(k, FP8_BLOCK_SIZE) // FP8_BLOCK_SIZE + 3) // 4


def quantize_fp8_block(h: torch.Tensor) -> Fp8BlockScaledActivation:
    """1x128 FP8 block-scale quantize of bf16 ``h`` (``[..., K]`` -> fp8 ``[rows, K]`` + packed
    UE8M0 scales ``[rows, P]`` int32 with ``P = fp8_scale_cols(K)``, MN-major).

    Pinned to ``trtllm::fp8_quantize_1x128_packed_ue8m0``, the CUDA quantizer of the Linear's
    own ``fp8_swap_ab_gemm`` path (its default, with ``TRTLLM_FUSED_FP8_QUANT_PACK=1``). That
    path autotunes its quantizer and may pick the Triton kernel for large row counts; the two
    follow the same scale rule but are not guaranteed bitwise, so the gathered bytes equal the
    Linear's own exactly for the CUDA tactic.
    """
    h2 = h.reshape(-1, h.shape[-1]).contiguous()
    fp8, scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(h2, False)
    return Fp8BlockScaledActivation(fp8, scale)


def fp8_scales_mn_major(rows: torch.Tensor) -> torch.Tensor:
    """Row-major ``[M, P]`` packed scales -> the layout DeepGEMM reads: ``(M, P)`` with strides
    ``(1, pad4(M))`` over a full ``[P, pad4(M)]`` buffer, as the quantize op allocates it. One
    transposing copy of ``M * P`` ints in eager; under torch.compile Inductor materializes the
    result with strides ``(1, M)`` instead, which the GEMM runner re-strides with one more tiny
    copy."""
    m, cols = rows.shape
    lead = pad_up(m, 4)
    out = torch.empty((cols, lead), dtype=rows.dtype, device=rows.device).t()[:m]
    out.copy_(rows)
    return out


# =============================================================================
# Collectives
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
    """The token plan and collectives of one transformer's TP group (see the module docstring).

    The model's :class:`TokenShardedSequenceSharder` calls :meth:`begin` and :meth:`shard`
    eagerly before the blocks and :meth:`unshard` after them; the adapters inside the blocks
    read :attr:`plan`. One instance per token stream: models with several streams (e.g. MMDiT
    text + image) are not supported. Not an ``nn.Module``, so the adapters hold it as a plain
    attribute.

    Args:
        group: The TP process group; its rank order is the token-shard order.
        tp_rank: Expected rank of this process in ``group``; a mismatch raises.
        row_align: Row alignment of every plan (see :meth:`TokenShardPlan.build`); the
            default 1 pads only what the rank count requires.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        *,
        tp_rank: int | None = None,
        row_align: int = 1,
    ) -> None:
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
        if row_align < 1:
            raise ValueError(f"TokenShardedTP needs row_align >= 1 (got {row_align}).")
        self.group = group
        self.tp_size: int = tp_size
        self.tp_rank: int = actual_rank
        self.row_align: int = row_align
        # Resolved once here: compiled blocks pass the name string to the functional
        # collectives instead of looking up the group (a compiler-disabled mesh path).
        self.group_name: str = group.group_name
        self._plans: dict[tuple[int, int], TokenShardPlan] = {}
        self._plan: TokenShardPlan | None = None

    @classmethod
    def from_model_config(cls, model_config: "DiffusionModelConfig") -> "TokenShardedTP":
        """The helper for the TP group of ``model_config.visual_gen_mapping``."""
        vgm = model_config.visual_gen_mapping
        if vgm is None:
            raise ValueError(
                "TokenShardedTP: tp_layout='token_sharded' needs a VisualGenMapping "
                "(model_config.visual_gen_mapping is None)."
            )
        return cls(vgm.tp_group_pg, tp_rank=vgm.tp_rank)

    # --- plan -----------------------------------------------------------------------

    def begin(self, batch_size: int, seq_len: int) -> TokenShardPlan:
        """Select and cache this rank's plan for ``(batch_size, seq_len)``.

        Call eagerly, and for a new shape before any CUDA-graph capture: the first use of a
        shape checks with an ``all_gather_object`` that all TP ranks run that shape (a mismatch
        would hang or corrupt the collectives). Later uses skip the check, so a rank that reuses
        a cached shape while a peer starts a new one is not detected.
        """
        key = (batch_size, seq_len)
        plan = self._plans.get(key)
        if plan is None:
            plan = TokenShardPlan.build(
                batch_size, seq_len, self.tp_size, self.tp_rank, self.row_align
            )
            self._check_rank_agreement(batch_size, seq_len)
            if plan.is_padded:
                extra = batch_size * (plan.padded_seq_len - seq_len)
                aligned = f" into {self.row_align}-row aligned shards" if self.row_align > 1 else ""
                logger.info_once(
                    f"Token-sharded TP: {batch_size}x{seq_len} tokens do not split evenly "
                    f"over tp_size={self.tp_size}{aligned}; padding each sample to "
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
                "TokenShardedTP.begin(batch_size, seq_len) must be called before the first block "
                "runs; shard the token stream with the model's sharder.shard() first."
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

    # --- shard / unshard / per-shard tables -------------------------------------------

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
        """This rank's ``[m, *rest]`` rows as ``[n, g, *rest]`` sample groups (a view).

        Every group lies inside one sample, so a per-sample ``[n, 1, D]`` table
        (:meth:`per_sample_table`) broadcasts over the shard as a ``[B, 1, D]`` table does over
        ``[B, S, D]``: block code written for ``[B, S, D]`` runs unchanged on the shard, and a
        fused AdaLN kernel reads ``seq_len_per_batch = g`` from the shapes.
        """
        p = self.plan
        return t.view(len(p.entry_batch), p.rows_per_entry, *t.shape[1:])

    def gather_input(
        self, consumer: nn.Module | None, act: Activation, *, prequantize: bool = True
    ) -> Activation:
        """A column projection's input: this rank's rows -> all ``B * S`` real rows.

        ``act`` is ``[n, g, K]`` or ``[m, K]``: bf16, an :class:`Fp4QuantizedTensor`
        quantized with ``consumer``'s static scale, or an :class:`Fp8BlockScaledActivation`.
        A bf16 input is quantized here (decided per call: the quant method is final only
        after loading) when ``consumer`` has a static NVFP4 input scale, or else when it
        takes a pre-quantized FP8 block-scale input (:func:`fp8_block_scale_prequant_ok`), so
        the all-gather moves NVFP4 or FP8 + scales instead of bf16. ``prequantize=False``
        keeps a bf16 input bf16 on the FP8 path, for a call whose consumer needs the dense
        input (an active LoRA).
        """
        if isinstance(act, Fp4QuantizedTensor):
            payload = act.fp4_tensor
            act = Fp4QuantizedTensor(  # the constructor, not dataclasses.replace: compiled code
                payload.reshape(-1, payload.shape[-1]),
                act.scaling_factor,
                act.is_sf_swizzled,
                act.unquantized_hidden_states,
                act.reciprocal_scale,
            )
        elif isinstance(act, Fp8BlockScaledActivation):
            act = Fp8BlockScaledActivation(act.fp8.reshape(-1, act.fp8.shape[-1]), act.scale)
        elif isinstance(act, torch.Tensor):
            act = act.reshape(-1, act.shape[-1])
            scale = static_nvfp4_input_scale(consumer)
            if scale is not None:
                act = quantize_nvfp4(act, scale)
            elif (
                prequantize
                and act.dtype == torch.bfloat16
                and fp8_block_scale_prequant_ok(consumer)
            ):
                act = quantize_fp8_block(act)
        else:
            raise self._unsupported_input("gather_input", act)
        self._check_fp4_consumer(consumer, act)
        self._check_fp8_consumer(consumer, act)
        return self.all_gather(act)

    @staticmethod
    def _check_fp4_consumer(consumer: object, act: Activation) -> None:
        # An NVFP4-gathered activation is only valid for a consumer that would quantize
        # with the same static input_scale (a Linear silently uses its own alpha).
        if (
            isinstance(act, Fp4QuantizedTensor)
            and isinstance(consumer, Linear)
            and static_nvfp4_input_scale(consumer) is None
        ):
            raise ValueError(
                "TokenShardedTP.gather_input: got an NVFP4 activation, but the consuming "
                "projection has no static NVFP4 input_scale; gather BF16 for it."
            )

    @staticmethod
    def _check_fp8_consumer(consumer: object, act: Activation) -> None:
        # Only the DeepGEMM FP8 block-scale Linear takes the (fp8, packed scales) pair; every
        # other Linear either rejects it or quantizes differently from the gathered bytes.
        if (
            isinstance(act, Fp8BlockScaledActivation)
            and isinstance(consumer, Linear)
            and not fp8_block_scale_prequant_ok(consumer)
        ):
            raise ValueError(
                "TokenShardedTP.gather_input: got an FP8 block-scale activation, but the "
                "consuming projection is not an FP8 block-scale Linear on the DeepGEMM path; "
                "gather BF16 for it."
            )

    @staticmethod
    def _unsupported_input(op: str, act: object) -> TypeError:
        # Linear takes bare (payload, scale) tuples for NVFP4 and FP8 alike; here the FP8 pair
        # must be typed so that neither is mistaken for the other.
        return TypeError(
            f"TokenShardedTP.{op}: expected a tensor, an Fp4QuantizedTensor or an "
            f"Fp8BlockScaledActivation; got {type(act).__name__}."
        )

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
        """Row-parallel K-partials -> this rank's reduced ``[m, N]`` rows.

        ``partial`` is the padded ``[B * S_pad, N]`` stream, or ``[B * S, N]`` / ``[B, S, N]``,
        which is padded here.
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
        """This rank's ``[m, K]`` rows -> all ``[B * S, K]`` rows, padding dropped.

        A static-scale :class:`Fp4QuantizedTensor` (payload ``[m, K/2]``, 128x4-swizzled SF for
        ``m`` rows) is gathered as NVFP4 and returned with its SF regrouped for ``B * S`` rows;
        an :class:`Fp8BlockScaledActivation` (``[m, K]`` fp8 + ``[m, P]`` int32 scales) is
        gathered as FP8 + scales and returned for ``B * S`` rows with its scales MN-major.
        """
        p = self.plan
        if isinstance(act_loc, Fp4QuantizedTensor):
            payload, sf, k = self._check_fp4(act_loc)
            payload = self._drop_padding(_all_gather_rows(payload, self.group_name))
            sf = regroup_swizzled_sf(
                _all_gather_rows(sf, self.group_name), p, k // NVFP4_SF_VEC_SIZE
            )
            return Fp4QuantizedTensor(payload, sf, is_sf_swizzled=True)
        if isinstance(act_loc, Fp8BlockScaledActivation):
            fp8, scale = self._check_fp8(act_loc)
            # Neither gloo nor the functional collectives take float8: move the bytes.
            payload = _all_gather_rows(fp8.view(torch.uint8), self.group_name)
            payload = self._drop_padding(payload).view(torch.float8_e4m3fn)
            # The per-rank scales are MN-major (P columns, tiny): gather them as rows,
            # drop the per-sample padding like the payload, then re-lay for B * S rows.
            scale = self._drop_padding(_all_gather_rows(scale, self.group_name))
            return Fp8BlockScaledActivation(payload, fp8_scales_mn_major(scale))
        if not isinstance(act_loc, torch.Tensor):
            raise self._unsupported_input("all_gather", act_loc)
        self._check_local_rows("all_gather", act_loc)
        return self._drop_padding(_all_gather_rows(act_loc, self.group_name))

    def _check_fp4(self, a: Fp4QuantizedTensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        m = self.plan.local_rows
        if a.reciprocal_scale is not None:
            raise ValueError(
                "TokenShardedTP.all_gather: cannot gather an Fp4QuantizedTensor with a per-rank "
                "dynamic scale (reciprocal_scale is set); gather BF16, or quantize with the "
                "consumer's static input_scale."
            )
        if a.unquantized_hidden_states is not None:
            raise ValueError(
                "TokenShardedTP.all_gather: cannot gather the unquantized_hidden_states side-car "
                "of an Fp4QuantizedTensor; drop it or gather BF16."
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

    def _check_fp8(self, a: Fp8BlockScaledActivation) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.plan.local_rows
        fp8, scale = a
        if fp8.dtype != torch.float8_e4m3fn or fp8.dim() != 2 or fp8.shape[0] != m:
            raise ValueError(
                f"TokenShardedTP.all_gather: FP8 block-scale payload must be float8_e4m3fn "
                f"[{m}, K] for the current plan ({self._plan_desc()}); got {fp8.dtype} "
                f"{tuple(fp8.shape)}."
            )
        cols = fp8_scale_cols(fp8.shape[1])
        if scale.dtype != torch.int32 or tuple(scale.shape) != (m, cols):
            raise ValueError(
                f"TokenShardedTP.all_gather: FP8 block scales must be packed UE8M0 int32 "
                f"[{m}, {cols}] for a [{m}, {fp8.shape[1]}] payload; got {scale.dtype} "
                f"{tuple(scale.shape)}."
            )
        return fp8, scale


# =============================================================================
# TokenShardedSequenceSharder
# =============================================================================


class TokenShardedSequenceSharder(SequenceSharder):
    """A model's ``SequenceSharder`` under token-sharded TP (set by ``_apply_tp_layout``).

    ``shard`` gives this TP rank its rows of a ``[B, S, ...]`` tensor as ``[n, g, ...]``
    sample groups, ``shard_per_sample`` slices a ``[B, ...]`` table to those groups and
    ``gather`` restores ``[B, S, ...]``. The sequence-parallel state (``is_active``, ``size``,
    ``group``) stays inactive, so RoPE stays whole and Ulysses-only code paths stay off.
    """

    token_sharded_tp = True

    def __init__(self, tp: TokenShardedTP):
        super().__init__(size=1, rank=0, group=None)
        self._tp = tp

    def shard(
        self,
        tensor: torch.Tensor | None,
        dim: int = 1,
        *,
        expected_seq_len: int | None = None,
        pad_to_multiple: bool = False,
    ) -> torch.Tensor | None:
        """As :meth:`SequenceSharder.shard`, to this rank's ``[n, g, ...]`` sample groups.

        The token stream (no ``expected_seq_len``) selects the forward's plan from its
        ``[B, S]``; per-token tables are sharded with that plan. ``pad_to_multiple`` is ignored:
        the plan pads each sample and ``gather`` drops the padding.
        """
        if tensor is None:
            return None
        if dim != 1:
            raise ValueError(
                "token-sharded TP shards the token dimension of [B, S, ...] tensors (dim=1); "
                f"got dim={dim}."
            )
        if expected_seq_len is None:
            self._tp.begin(tensor.shape[0], tensor.shape[1])
        elif tensor.shape[1] != expected_seq_len:
            return tensor
        return self._tp.local_view(self._tp.shard(tensor))

    def shard_per_sample(self, table: torch.Tensor | None) -> torch.Tensor | None:
        return None if table is None else self._tp.per_sample_table(table)

    def gather(
        self, tensor: torch.Tensor, dim: int = 1, *, unpad_to: int | None = None
    ) -> torch.Tensor:
        if dim != 1:
            raise ValueError(
                f"token-sharded TP gathers the token dimension (dim=1); got dim={dim}."
            )
        out = self._tp.unshard(tensor.reshape(self._tp.plan.local_rows, *tensor.shape[2:]))
        return out if unpad_to is None else out[:, :unpad_to]
