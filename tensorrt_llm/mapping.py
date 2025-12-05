# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from enum import IntEnum
from typing import List

import torch
from torch.distributed import ProcessGroup

from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl
from tensorrt_llm._utils import mpi_disabled


class CpType(IntEnum):
    # CP type for ulysses parallelism
    ULYSSES = 0
    # CP type for star attention
    STAR = 1
    # CP type for ring attention
    RING = 2
    # CP type for helix parallelism
    HELIX = 3


class MappingBase:
    """Base class for distributed mapping configurations"""

    tp_rank: int
    pp_rank: int
    cp_rank: int

    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            *,
            cp_size=1,
            cp_config=None,
            tp_size=1,
            pp_size=1,
            pp_partition=None,
            moe_cluster_size=-1,  # -1 means no moe
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1,  # -1 means no moe
            attn_tp_size=-1,
            attn_cp_size=-1,
            enable_attention_dp=False,
            enable_lm_head_tp_in_adp=False):
        # set default values for non-moe cases
        # or where only one MOE parallelism size is specified
        if moe_cluster_size == -1:
            moe_cluster_size = 1

        # Set default cp_type to ULYSSES.
        cp_type = CpType.ULYSSES

        # Convert cp_type to CpType enum if it is a string.
        if cp_config is not None:
            if "cp_type" in cp_config and isinstance(cp_config["cp_type"], str):
                try:
                    cp_config["cp_type"] = CpType[cp_config["cp_type"].upper()]
                except KeyError:
                    raise ValueError(f"Invalid cp_type: {cp_config['cp_type']}. " \
                                    f"Must be one of: {', '.join([t.name for t in CpType])}")
            cp_type = cp_config.get("cp_type", CpType.ULYSSES)

        moe_world_size = tp_size if cp_type == CpType.ULYSSES else tp_size * cp_size

        if moe_tp_size == -1 and moe_ep_size == -1:
            moe_tp_size = moe_world_size // moe_cluster_size
            moe_ep_size = 1

        elif moe_tp_size == -1:
            moe_tp_size = moe_world_size // (moe_ep_size * moe_cluster_size)

        elif moe_ep_size == -1:
            moe_ep_size = moe_world_size // (moe_tp_size * moe_cluster_size)

        if attn_tp_size == -1 and attn_cp_size == -1:
            if cp_type == CpType.ULYSSES:
                # fallback to ulysses
                attn_tp_size = tp_size * cp_size
                attn_cp_size = 1
            else:
                # fallback to helix
                attn_tp_size = tp_size
                attn_cp_size = cp_size

        elif attn_tp_size == -1:
            attn_tp_size = (tp_size * cp_size) // attn_cp_size

        elif attn_cp_size == -1:
            attn_cp_size = (tp_size * cp_size) // attn_tp_size

        if attn_cp_size != 1 and cp_type == CpType.ULYSSES:
            raise ValueError(
                f"attn_cp_size must be 1 for now for ulysses, but got {attn_tp_size}, {attn_cp_size}."
            )

        if tp_size * pp_size * cp_size != world_size:
            raise ValueError(
                "world_size must equal to tp_size * pp_size * cp_size, "
                f"but got {world_size} != {tp_size} * {pp_size} * {cp_size}.")

        moe_tp_ep_size = moe_tp_size * moe_ep_size
        self.moe_tp_cluster_ep_size = moe_tp_ep_size * moe_cluster_size
        if self.moe_tp_cluster_ep_size != moe_world_size:
            raise ValueError(
                "moe_tp_size * moe_ep_size * moe_cluster_size must equal to moe_world_size, "
                f"but got {self.moe_tp_cluster_ep_size} != {moe_world_size}")

        attn_tp_cp_size = attn_tp_size * attn_cp_size
        if attn_tp_cp_size != tp_size * cp_size:
            raise ValueError(
                "tp_size * cp_size must equal to attn_tp_size * attn_cp_size, "
                f"but got {tp_size} * {cp_size} != {attn_tp_size} * {attn_cp_size}"
            )

        if moe_ep_size != 1 and cp_size > 1 and cp_type != CpType.HELIX:
            raise NotImplementedError(
                f"CP {cp_type} doesn't support MoE tp/ep yet")

        if moe_cluster_size > 1:
            assert moe_ep_size == 1

        self.tp_size = tp_size
        self.cp_size = cp_size
        self.cp_config = cp_config if cp_config is not None else {}
        self.pp_size = pp_size
        self.pp_partition = pp_partition
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.moe_cluster_size = moe_cluster_size
        self.attn_tp_size = attn_tp_size
        self.attn_cp_size = attn_cp_size
        self.world_size = world_size
        self.enable_attention_dp = enable_attention_dp
        if enable_lm_head_tp_in_adp:
            assert enable_attention_dp, "enable_lm_head_tp_in_adp requires enable_attention_dp"
        self.enable_lm_head_tp_in_adp = enable_lm_head_tp_in_adp
        self.rank = rank
        self.gpus_per_node = gpus_per_node

        self.pp_groups = []
        self.cp_groups = []
        self.tp_groups = []
        self.moe_cluster_groups = []
        self.moe_tp_groups = []
        self.moe_ep_groups = []

    def __eq__(self, other):
        if not isinstance(other, MappingBase):
            return NotImplemented

        return (self.world_size == other.world_size and self.rank == other.rank
                and self.gpus_per_node == other.gpus_per_node
                and self.cp_size == other.cp_size
                and self.tp_size == other.tp_size
                and self.moe_cluster_size == other.moe_cluster_size
                and self.pp_size == other.pp_size
                and self.pp_partition == other.pp_partition
                and self.moe_tp_size == other.moe_tp_size
                and self.moe_ep_size == other.moe_ep_size
                and self.attn_tp_size == other.attn_tp_size
                and self.attn_cp_size == other.attn_cp_size
                and self.cp_config == other.cp_config)

    def __hash__(self):
        return hash((
            self.world_size,
            self.rank,
            self.gpus_per_node,
            self.cp_size,
            self.tp_size,
            self.pp_size,
            self.moe_tp_size,
            self.moe_cluster_size,
            self.moe_ep_size,
            self.attn_tp_size,
            self.attn_cp_size,
            # note: we do not allow updating cp_config after initialization
            tuple(sorted(self.cp_config.items())),
            tuple(self.pp_partition) if self.pp_partition is not None else (),
        ))

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        # TODO(qijun): skip check for enable_attention_dp temporarily, will support attention_dp_size
        if not self.enable_attention_dp:
            if not isinstance(rank, int) or rank < 0 or rank >= self.world_size:
                raise ValueError(
                    f"Rank should be an integer between 0 and {self.world_size-1}, but got {rank}."
                )
        self._rank = rank

    @property
    def moe_tp_rank(self):
        return self.tp_rank // (self.moe_ep_size * self.moe_cluster_size)

    @property
    def moe_cluster_rank(self):
        return self.tp_rank % self.moe_cluster_size

    @property
    def moe_ep_rank(self):
        return self.tp_rank % self.moe_ep_size

    @property
    def moe_cluster_group(self):
        return self.moe_cluster_groups[self.pp_rank * self.moe_tp_size +
                                       self.moe_tp_rank]

    @property
    def node_rank(self):
        return self.rank // self.gpus_per_node

    @property
    def local_rank(self):
        return self.rank % self.gpus_per_node

    @property
    def dp_size(self):
        return self.tp_size if self.enable_attention_dp else 1

    def has_cp_ulysses(self):
        return self.cp_size > 1 and self.cp_config.get(
            "cp_type") == CpType.ULYSSES

    def has_cp_helix(self):
        return self.cp_size > 1 and self.cp_config.get(
            "cp_type") == CpType.HELIX

    def get_helix_overridden_tp_rank(self) -> int:
        """Get the overridden TP rank when repurposing helix CP to TP.

        In helix parallelism, CP groups are structured differently than TP groups.
        For example, with tp_size=2, cp_size=2:
        - CP groups: [0, 2], [1, 3] (accumulated order: [0, 2, 1, 3])
        - When repurposed to TP: [0, 1, 2, 3]

        The helix accumulated order iterates through TP ranks, and for each TP rank
        iterates through CP ranks. So the position in helix order is:
            helix_position = tp_rank * cp_size + cp_rank

        This function computes the TP rank in the repurposed mapping, accounting
        for the reordering from helix accumulated order to standard TP order.

        Returns:
            The TP rank in the repurposed (tp_size * cp_size, cp_size=1) mapping.
        """
        return self.tp_rank * self.cp_size + self.cp_rank

    def get_node_rank(self, rank: int):
        return rank // self.gpus_per_node

    def get_local_rank(self, rank: int):
        return rank % self.gpus_per_node

    def is_multi_node(self):
        return self.world_size > self.gpus_per_node

    def has_tp(self):
        return self.tp_size > 1

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_second_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 2

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        p = self.rank - self.tp_size * self.cp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size * self.cp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

    def is_last_cp_rank(self):
        return self.cp_rank == self.cp_size - 1

    def is_first_cp_rank(self):
        return self.cp_rank == 0

    def has_cp(self):
        return self.cp_size > 1

    def prev_cp_rank(self):
        p = self.rank - self.tp_size
        if p // (self.tp_size * self.cp_size) < self.rank // (self.tp_size *
                                                              self.cp_size):
            return p + self.tp_size * self.cp_size
        return p

    def next_cp_rank(self):
        p = self.rank + self.tp_size
        if p // (self.tp_size * self.cp_size) > self.rank // (self.tp_size *
                                                              self.cp_size):
            return p - self.tp_size * self.cp_size
        return p

    def has_moe_cluster(self):
        return self.moe_cluster_size > 1

    def has_moe_tp(self):
        return self.moe_tp_size > 1

    def has_moe_ep(self):
        return self.moe_ep_size > 1

    def pp_layers(self, num_layers: int) -> List[int]:
        if self.pp_partition is not None:
            if len(self.pp_partition) != self.pp_size:
                raise ValueError(
                    f"{len(self.pp_partition)=} does not match {self.pp_size=}."
                )
            if sum(self.pp_partition) != num_layers:
                raise ValueError(
                    f"{sum(self.pp_partition)=} does not match {num_layers=}.")
            return torch.arange(num_layers).split(
                self.pp_partition)[self.pp_rank].tolist()
        else:
            # If num_layers % pp_size = n != 0, first n ranks get one extra layer
            return torch.tensor_split(torch.arange(num_layers),
                                      self.pp_size)[self.pp_rank].tolist()

    def ep_experts(self, num_experts: int) -> List[int]:
        assert self.cp_size == 1
        experts_per_rank = num_experts // self.moe_ep_size
        experts_range = range(self.moe_ep_rank * experts_per_rank,
                              (self.moe_ep_rank + 1) * experts_per_rank)
        return list(experts_range)

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(**mapping)

    def to_dict(self):
        return {
            'world_size': self.world_size,
            'rank': self.rank,
            'gpus_per_node': self.gpus_per_node,
            'cp_size': self.cp_size,
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'moe_tp_size': self.moe_tp_size,
            'moe_cluster_size': self.moe_cluster_size,
            'moe_ep_size': self.moe_ep_size,
            'attn_tp_size': self.attn_tp_size,
            'attn_cp_size': self.attn_cp_size,
            'cp_config': self.cp_config,
            'enable_attention_dp': self.enable_attention_dp,
            'enable_lm_head_tp_in_adp': self.enable_lm_head_tp_in_adp,
        }


class Mapping(MappingBase):
    """
    A node with 8 GPUs, tp_size = 4, cp_size = 1, pp_size = 2

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 pp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    A node with 8 GPUs, tp_size = 4, cp_size = 2, pp_size = 1

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 cp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    A node with 8 GPUs, moe_tp_size = 2, moe_ep_size = 4

    4 moe_tp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    2 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    2 nodes with 16 GPUs, moe_tp_size = 2, moe_ep_size = 4, pp_size = 2

    8 moe_tp groups:

    - [0 4]
    - [1 5]
    - [2 6]
    - [3 7]
    - [8 12]
    - [9 13]
    - [10 14]
    - [11 15]

    4 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]
    - [8, 9, 10, 11]
    - [12, 13, 14, 15]

    8 pp groups:

    - [0 8]
    - [1 9]
    - [2 10]
    - [3 11]
    - [4 12]
    - [5 13]
    - [6 14]
    - [7 15]

    2 nodes with 8 GPUs, tp_size 2, pp_size 2, cp_size 2

    4 tp groups:
    - [0, 1]
    - [2, 3]
    - [4, 5]
    - [6, 7]

    4 pp groups:
    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    4 cp groups:
    - [0, 2]
    - [1, 3]
    - [4, 6]
    - [5, 7]
    """

    def __new__(cls, *args, **kwargs):
        if mpi_disabled():
            return super().__new__(DeviceMeshTopology)
        else:
            return super().__new__(MpiTopology)

    # Intentionally repeated for type hints
    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            *,
            cp_size=1,
            cp_config=None,
            tp_size=1,
            pp_size=1,
            pp_partition=None,
            moe_cluster_size=-1,  # -1 means no moe
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1,  # -1 means no moe
            attn_tp_size=-1,
            attn_cp_size=-1,
            enable_attention_dp=False,
            enable_lm_head_tp_in_adp=False):
        super().__init__(world_size=world_size,
                         rank=rank,
                         gpus_per_node=gpus_per_node,
                         cp_size=cp_size,
                         cp_config=cp_config,
                         tp_size=tp_size,
                         pp_size=pp_size,
                         pp_partition=pp_partition,
                         moe_cluster_size=moe_cluster_size,
                         moe_tp_size=moe_tp_size,
                         moe_ep_size=moe_ep_size,
                         attn_tp_size=attn_tp_size,
                         attn_cp_size=attn_cp_size,
                         enable_attention_dp=enable_attention_dp,
                         enable_lm_head_tp_in_adp=enable_lm_head_tp_in_adp)

    def repurpose_helix_cp_to_tp(self):
        # In helix parallelism, CP is relevant only for the attention layer. These ranks are repurposed to TP
        # for FFN layers.
        assert self.has_cp_helix()
        return Mapping(
            world_size=self.world_size,
            rank=self.rank,
            gpus_per_node=self.gpus_per_node,
            cp_size=1,
            cp_config={},
            tp_size=self.tp_size * self.cp_size,
            pp_size=self.pp_size,
            pp_partition=self.pp_partition,
            moe_cluster_size=self.moe_cluster_size,
            moe_tp_size=self.moe_tp_size,
            moe_ep_size=self.moe_ep_size,
            # attn_tp_size, attn_cp_size shall be set in the constructor of Mapping.
            enable_attention_dp=self.enable_attention_dp,
            enable_lm_head_tp_in_adp=self.enable_lm_head_tp_in_adp)

    # DeviceMesh specific methods
    @property
    def tp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("tp_group_pg is not implemented.")

    @property
    def pp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("pp_group_pg is not implemented.")

    @property
    def cp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("cp_group_pg is not implemented.")

    @property
    def moe_tp_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("moe_tp_group_pg is not implemented.")

    @property
    def moe_ep_group_pg(self) -> ProcessGroup:
        raise NotImplementedError("moe_ep_group_pg is not implemented.")

    def build_mesh(self):
        raise NotImplementedError("build_mesh is not implemented.")


class MpiTopology(Mapping):
    '''MPI-based mapping implementation'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_parallel_groups()

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def pp_rank(self) -> int:
        return self.rank // (self.tp_size * self.cp_size)

    @property
    def cp_rank(self) -> int:
        return self.rank % (self.tp_size * self.cp_size) // self.tp_size

    @property
    def tp_group(self) -> List[int]:
        return self.tp_groups[self.pp_rank * self.cp_size + self.cp_rank]

    @property
    def pp_group(self) -> List[int]:
        return self.pp_groups[self.cp_rank * self.tp_size + self.tp_rank]

    @property
    def cp_group(self) -> List[int]:
        return self.cp_groups[self.pp_rank * self.tp_size + self.tp_rank]

    @property
    def moe_tp_group(self) -> List[int]:
        return self.moe_tp_groups[self.pp_rank * self.moe_cluster_size *
                                  self.moe_ep_size +
                                  self.moe_cluster_rank * self.moe_ep_size +
                                  self.moe_ep_rank]

    @property
    def moe_ep_group(self) -> List[int]:
        return self.moe_ep_groups[self.pp_rank * self.moe_tp_size *
                                  self.moe_cluster_size +
                                  self.moe_tp_rank * self.moe_cluster_size +
                                  self.moe_cluster_rank]

    @property
    def moe_cluster_group(self) -> List[int]:
        return self.moe_cluster_groups[self.pp_rank * self.moe_tp_size +
                                       self.moe_tp_rank]

    def _init_parallel_groups(self):
        # init pp group
        for i in range(self.tp_size * self.cp_size):
            ranks = range(i, self.world_size, self.tp_size * self.cp_size)
            self.pp_groups.append(list(ranks))

        # init cp group
        for i in range(self.pp_size):
            for j in range(self.tp_size):
                ranks = range(i * self.tp_size * self.cp_size + j,
                              (i + 1) * self.tp_size * self.cp_size + j,
                              self.tp_size)
                self.cp_groups.append(list(ranks))

        # init tp group
        for i in range(self.pp_size):
            for j in range(self.cp_size):
                ranks = range(
                    i * self.tp_size * self.cp_size + j * self.tp_size,
                    i * self.tp_size * self.cp_size + (j + 1) * self.tp_size)
                self.tp_groups.append(list(ranks))

        # init moe tp group
        for i in range(self.pp_size):
            for j in range(self.moe_cluster_size * self.moe_ep_size):
                ranks = range(i * self.moe_tp_cluster_ep_size + j,
                              (i + 1) * self.moe_tp_cluster_ep_size,
                              self.moe_cluster_size * self.moe_ep_size)
                self.moe_tp_groups.append(list(ranks))

        # init moe cluster group
        for i in range(self.pp_size):
            for j in range(self.moe_tp_size):
                ranks = range(
                    i * self.moe_tp_cluster_ep_size +
                    j * self.moe_cluster_size * self.moe_ep_size,
                    i * self.moe_tp_cluster_ep_size +
                    (j + 1) * self.moe_cluster_size * self.moe_ep_size)
                self.moe_cluster_groups.append(list(ranks))

        # init moe ep group
        for i in range(self.pp_size):
            for j in range(self.moe_tp_size):
                for k in range(self.moe_cluster_size):
                    ranks = range(
                        i * self.moe_tp_cluster_ep_size +
                        j * self.moe_cluster_size * self.moe_ep_size +
                        k * self.moe_ep_size, i * self.moe_tp_cluster_ep_size +
                        j * self.moe_cluster_size * self.moe_ep_size +
                        (k + 1) * self.moe_ep_size)
                    self.moe_ep_groups.append(list(ranks))


class DeviceMeshTopology(DeviceMeshTopologyImpl, Mapping):
    """PyTorch DeviceMesh-based mapping implementation"""

    def __init__(self, *args, **kwargs):
        assert mpi_disabled(
        ), "DeviceMeshTopology is only available in Ray orchestrator mode."

        super().__init__(*args, **kwargs)
