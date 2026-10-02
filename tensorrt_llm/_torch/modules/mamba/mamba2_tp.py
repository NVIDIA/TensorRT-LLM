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
"""Tensor-parallel sharding of Mamba2 mixers, including group replication.

Mamba2 splits its heads across TP ranks. The B/C projections are shared by
groups of heads (``n_groups``), so every rank has to hold the B/C columns of
each group its heads belong to:

* ``tp_size <= n_groups``: each rank owns ``n_groups // tp_size`` whole groups
  and no B/C column is stored twice. This is the historical layout and stays
  bit-identical.
* ``tp_size > n_groups``: one group spans ``tp_size // n_groups`` ranks. Each of
  them keeps a replica of that group's B/C rows of ``in_proj``, its ``conv1d``
  channels and its conv-state slice, and recomputes B/C redundantly. Heads are
  never replicated, so the row-parallel ``out_proj`` all-reduce stays exact;
  only the grouped gated RMSNorm has to exchange the per-token sum of squares
  across the replicas (:class:`ReplicatedGroupRMSNormGated`).

The padded "virtual full" sizes (``padded_*``) are what the column-parallel
``Linear`` layers are constructed with, so that the standard ``tp_size``-way
row split of a checkpoint tensor rearranged by
:meth:`Mamba2TpShard.rearrange_in_proj_rows` /
:meth:`Mamba2TpShard.rearrange_conv1d_rows` yields exactly the per-rank layout.
"""

from dataclasses import dataclass
from typing import List

import torch

from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

from ...utils import split
from .layernorm_gated import RMSNorm as RMSNormGated


def mamba2_valid_tp_sizes(nheads: int, n_groups: int) -> List[int]:
    """TP sizes whose heads split evenly and whose groups split or replicate evenly."""
    return [
        tp
        for tp in range(1, nheads + 1)
        if nheads % tp == 0 and (n_groups % tp == 0 or tp % n_groups == 0)
    ]


def validate_mamba2_tp(nheads: int, n_groups: int, tp_size: int) -> None:
    """Raise ``ValueError`` unless ``tp_size`` is a legal Mamba2 TP degree."""
    degenerate = nheads < 1 or n_groups < 1
    if (
        tp_size < 1
        or degenerate
        or nheads % tp_size != 0
        or not (n_groups % tp_size == 0 or tp_size % n_groups == 0)
    ):
        message = (
            "Mamba2 tensor parallelism needs tp_size to divide "
            f"mamba_num_heads={nheads} and to either divide or be a multiple "
            f"of n_groups={n_groups}; got tp_size={tp_size}."
        )
        # With a degenerate geometry the "valid" list is computed from bogus
        # inputs (n_groups=0 makes every tp_size look legal), so suppress it
        # rather than advertise TP degrees that cannot work.
        if not degenerate:
            message += f" Valid tp_size values: {mamba2_valid_tp_sizes(nheads, n_groups)}"
        raise ValueError(message)


@dataclass(frozen=True)
class Mamba2TpShard:
    """Per-rank Mamba2 dimensions for a given TP degree.

    All consumers (mixer, weight mapper, cache managers, memory estimates) must
    take their per-rank sizes from here so the layouts agree.
    """

    tp_size: int
    nheads: int
    n_groups: int
    head_dim: int
    d_state: int

    def __post_init__(self):
        validate_mamba2_tp(self.nheads, self.n_groups, self.tp_size)

    # ---- replication ------------------------------------------------------
    @property
    def replication(self) -> int:
        """Number of ranks sharing one group (1 when groups split evenly)."""
        return self.tp_size // self.n_groups if self.tp_size > self.n_groups else 1

    @property
    def replicated(self) -> bool:
        return self.replication > 1

    def group_index(self, tp_rank: int) -> int:
        """Index of the single group owned by ``tp_rank`` (replicated layouts)."""
        assert self.replicated, "group_index is only defined for replicated layouts"
        return tp_rank // self.replication

    # ---- per-rank sizes ---------------------------------------------------
    @property
    def d_inner(self) -> int:
        return self.nheads * self.head_dim

    @property
    def tp_nheads(self) -> int:
        return self.nheads // self.tp_size

    @property
    def tp_ngroups(self) -> int:
        return 1 if self.replicated else self.n_groups // self.tp_size

    @property
    def tp_d_inner(self) -> int:
        return self.tp_nheads * self.head_dim

    @property
    def tp_grouped_state_dim(self) -> int:
        """Rows of one B (or C) block on this rank: ``tp_ngroups * d_state``."""
        return self.tp_ngroups * self.d_state

    @property
    def tp_conv_dim(self) -> int:
        return self.tp_d_inner + 2 * self.tp_grouped_state_dim

    @property
    def tp_d_in_proj(self) -> int:
        return 2 * self.tp_d_inner + 2 * self.tp_grouped_state_dim + self.tp_nheads

    @property
    def group_size(self) -> int:
        """Width of one gated-RMSNorm group in the unsharded model."""
        return self.d_inner // self.n_groups

    # ---- padded "virtual full" sizes for column-parallel Linear -----------
    @property
    def padded_n_groups(self) -> int:
        return self.tp_ngroups * self.tp_size

    @property
    def padded_conv_dim(self) -> int:
        return self.tp_conv_dim * self.tp_size

    @property
    def padded_d_in_proj(self) -> int:
        return self.tp_d_in_proj * self.tp_size

    # ---- checkpoint row sharding -----------------------------------------
    def shard_heads(self, t: torch.Tensor, tp_rank: int) -> torch.Tensor:
        """Rows of a head-major tensor (``d_inner`` or ``nheads`` rows) for ``tp_rank``."""
        return split(t, self.tp_size, tp_rank)

    def shard_groups(self, t: torch.Tensor, tp_rank: int) -> torch.Tensor:
        """Rows of an ``[n_groups * d_state, ...]`` tensor held by ``tp_rank``."""
        if self.replicated:
            return t.narrow(0, self.group_index(tp_rank) * self.d_state, self.d_state)
        return split(t, self.tp_size, tp_rank)

    def rearrange_in_proj_rows(self, w: torch.Tensor) -> torch.Tensor:
        """Reorder ``in_proj`` rows ``[z | x | B | C | dt]`` (weight or per-row
        scale) into the concatenation over ranks of ``[z_r | x_r | B_r | C_r |
        dt_r]`` so the standard ``tp_size``-way split along dim 0 yields rank
        ``r``'s slice. The result has ``padded_d_in_proj`` rows."""
        grouped = self.n_groups * self.d_state
        z, x, b, c, dt = torch.split(
            w, [self.d_inner, self.d_inner, grouped, grouped, self.nheads], dim=0
        )
        return torch.cat(
            [
                torch.cat(
                    [
                        self.shard_heads(z, r),
                        self.shard_heads(x, r),
                        self.shard_groups(b, r),
                        self.shard_groups(c, r),
                        self.shard_heads(dt, r),
                    ]
                )
                for r in range(self.tp_size)
            ]
        ).contiguous()

    def rearrange_conv1d_rows(self, w: torch.Tensor) -> torch.Tensor:
        """Same as :meth:`rearrange_in_proj_rows` for ``conv1d`` rows ``[x | B | C]``."""
        grouped = self.n_groups * self.d_state
        x, b, c = torch.split(w, [self.d_inner, grouped, grouped], dim=0)
        return torch.cat(
            [
                torch.cat(
                    [
                        self.shard_heads(x, r),
                        self.shard_groups(b, r),
                        self.shard_groups(c, r),
                    ]
                )
                for r in range(self.tp_size)
            ]
        ).contiguous()


class ReplicatedGroupRMSNormGated(RMSNormGated):
    """Gated grouped RMSNorm for a rank holding ``1 / replication`` of one group.

    Mamba2 normalizes ``x * silu(z)`` over groups of ``group_size`` channels.
    When a group is spread over several ranks, every rank has only a partial
    sum of squares, so the per-token statistics are combined with one small
    all-reduce over the TP group: a ``[tokens, n_groups]`` fp32 buffer in which
    each rank fills the column of its own group. Only ``norm_before_gate=False``
    (the Mamba2 convention) is supported, and ``forward`` is a plain PyTorch
    implementation rather than a call into the fused norm+quant kernels.
    Skipping them costs nothing here: the fused FP8 output path fires only when
    ``fp8_scale`` is attached to the norm, which no Mamba2 code path does --
    ``gdn_mixer.py`` is the only place that attaches it -- so that path is
    unreachable for Mamba2 norms. ``out_proj`` quantizes its input itself.

    The inherited ``group_size`` attribute equals ``tp_d_inner``, this rank's
    LOCAL width; that is all the base (non-replicated) class needs, but it is
    not the true, unsharded group width. Normalization here instead divides by
    ``full_group_size`` (set in ``__init__``), so a future fused kernel that
    reads ``group_size`` off this class must use ``full_group_size`` instead,
    or it will normalize over the wrong number of channels.

    ``allreduce_strategy`` defaults to, and is only validated with, ``NCCL``:
    the exchanged buffer is a small fp32 ``[tokens, n_groups]`` tensor, not
    the 16-bit activation shapes the workspace/MNNVL strategies size their
    buffers for. The mixer always passes ``NCCL`` explicitly; the parameter
    is kept so a caller can opt into another strategy once it has been
    validated for this buffer shape.
    """

    def __init__(
        self,
        shard: Mamba2TpShard,
        tp_rank: int,
        mapping: Mapping,
        eps: float = 1e-5,
        dtype: torch.dtype | None = None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.NCCL,
    ):
        super().__init__(
            shard.tp_d_inner,
            eps=eps,
            group_size=shard.tp_d_inner,
            norm_before_gate=False,
            dtype=dtype,
            is_nvfp4=False,
        )
        # Deferred: this module is imported by config/estimation code and the
        # weight mapper, neither of which should pull in the distributed ops
        # at module import time.
        from ...distributed import AllReduce

        self.full_group_size = shard.group_size
        self.n_groups = shard.n_groups
        self.group_index = shard.group_index(tp_rank)
        self.all_reduce = AllReduce(
            mapping=mapping, strategy=allreduce_strategy, dtype=torch.float32
        )

    def _exchange(self, buf: torch.Tensor) -> torch.Tensor:
        """All-reduce the ``[tokens, n_groups]`` statistics buffer over TP."""
        return self.all_reduce(buf)

    def reduce_sum_of_squares(self, local_ss: torch.Tensor) -> torch.Tensor:
        """Turn this rank's partial per-token sum of squares into the group total."""
        buf = local_ss.new_zeros((local_ss.shape[0], self.n_groups))
        buf[:, self.group_index] = local_ss
        return self._exchange(buf)[:, self.group_index]

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None):
        if z is None:
            raise ValueError("ReplicatedGroupRMSNormGated requires the gate tensor z")
        x_shape = x.shape
        xz = x.reshape(-1, x_shape[-1]).float() * torch.nn.functional.silu(
            z.reshape(-1, x_shape[-1]).float()
        )
        total_ss = self.reduce_sum_of_squares(xz.square().sum(dim=-1))
        rstd = torch.rsqrt(total_ss / self.full_group_size + self.eps)
        y = xz * rstd.unsqueeze(-1) * self.weight.float()
        return y.to(x.dtype).reshape(x_shape)
