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
import os
from enum import IntEnum
from typing import List

import torch
from torch.distributed import get_process_group_ranks
from torch.distributed.device_mesh import init_device_mesh

from tensorrt_llm.logger import logger


class CpType(IntEnum):
    # CP type for ulysses parallelism
    ULYSSES = 0
    # CP type for star attention
    STAR = 1
    # CP type for ring attention
    RING = 2
    # CP type for helix parallelism
    HELIX = 3


class Mapping(object):
    '''
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
    '''
    # Static variable to store the device mesh
    device_mesh = None
    tp_mesh = None

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
            moe_cluster_size=-1,  # -1 means no moe
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1,  # -1 means no moe
            attn_tp_size=-1,
            attn_cp_size=-1,
            auto_parallel=False,
            enable_attention_dp=False):
        # set default values for non-moe cases
        # or where only one MOE parallelism size is specified
        if moe_cluster_size == -1:
            moe_cluster_size = 1

        cp_type = CpType.ULYSSES if cp_config is None else cp_config.get(
            "cp_type", CpType.ULYSSES)
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

        if auto_parallel:
            if tp_size != 1 or pp_size != 1 or cp_size != 1:
                raise ValueError(
                    "When auto parallel is enabled, tp_size, pp_size, cp_size must be 1, "
                    f"but got {tp_size}, {pp_size}, {cp_size}.")
        else:
            if tp_size * pp_size * cp_size != world_size:
                raise ValueError(
                    "world_size must equal to tp_size * pp_size * cp_size, "
                    f"but got {world_size} != {tp_size} * {pp_size} * {cp_size}."
                )

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
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.moe_cluster_size = moe_cluster_size
        self.attn_tp_size = attn_tp_size
        self.attn_cp_size = attn_cp_size
        self.auto_parallel = auto_parallel
        self.world_size = world_size
        self.enable_attention_dp = enable_attention_dp
        self.rank = rank
        self.gpus_per_node = gpus_per_node

        # For Ray path. store a reference to TorchDist
        self._dist = None
        self._disable_mpi = os.environ.get("DISABLE_MPI") == "1"

        # TODO: can deprecate if moving to DeviceMesh
        self._pp_groups = []
        self._cp_groups = []
        self._tp_groups = []
        self._moe_cluster_groups = []
        self._moe_tp_groups = []
        self._moe_ep_groups = []
        self._init_parallel_groups()

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented

        return (self.world_size == other.world_size and self.rank == other.rank
                and self.gpus_per_node == other.gpus_per_node
                and self.cp_size == other.cp_size
                and self.tp_size == other.tp_size
                and self.moe_cluster_size == other.moe_cluster_size
                and self.pp_size == other.pp_size
                and self.moe_tp_size == other.moe_tp_size
                and self.moe_ep_size == other.moe_ep_size
                and self.attn_tp_size == other.attn_tp_size
                and self.attn_cp_size == other.attn_cp_size
                and self.cp_config == other.cp_config
                and self.auto_parallel == other.auto_parallel)

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
            self.auto_parallel,
        ))

    @property
    def dist(self):
        return self._dist

    @dist.setter
    def dist(self, dist):
        if self._dist is not None:
            raise RuntimeError(f"Mapping.dist is already set to {dist}")
        if not self._disable_mpi:
            logger.warning(
                "No affect on setting dist for Mapping, unless DISABLE_MPI=1.")
            return
        self._build_mesh()
        self._dist = dist

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
    def tp_rank(self):
        if Mapping.device_mesh:
            assert self.auto_parallel == False, "Auto parallel not yet implemented in Ray path."
            return self.tp_group_pg.rank()
        else:
            return 0 if self.auto_parallel else self.rank % self.tp_size

    @property
    def pp_rank(self):
        if Mapping.device_mesh:
            assert self.auto_parallel == False, "Auto parallel not yet implemented in Ray path."
            return self.pp_group_pg.rank()
        else:
            return 0 if self.auto_parallel else self.rank // (self.tp_size *
                                                              self.cp_size)

    @property
    def cp_rank(self):
        return 0 if self.auto_parallel else self.rank % (
            self.tp_size * self.cp_size) // self.tp_size

    @property
    def moe_tp_rank(self):
        return self.tp_rank // (self.moe_ep_size * self.moe_cluster_size)

    @property
    def moe_cluster_rank(self):
        return self.tp_rank % self.moe_cluster_size

    @property
    def moe_ep_rank(self):
        return self.tp_rank % self.moe_ep_size

    # TODO: Remove parity assertions and old mapping code if moving to use DeviceMesh
    # for all following xx_group functs.
    @property
    def tp_group(self):
        mapping_old = self._tp_groups[self.pp_rank * self.cp_size +
                                      self.cp_rank]
        if Mapping.device_mesh:
            mapping_dm = get_process_group_ranks(self.tp_group_pg)
            assert mapping_old == mapping_dm
            # print(
            #     f"[Rank {self.rank}] TP mapping_old: {mapping_old} mapping_dm: {mapping_dm}"
            # )
            return mapping_dm
        return mapping_old

    @property
    def pp_group(self):
        mapping_old = self._pp_groups[self.cp_rank * self.tp_size +
                                      self.tp_rank]
        if Mapping.device_mesh:
            mapping_dm = get_process_group_ranks(self.pp_group_pg)
            assert mapping_old == mapping_dm
            # print(
            #     f"[Rank {self.rank}] PP mapping_old: {mapping_old} mapping_dm: {mapping_dm}"
            # )
            return mapping_dm
        return mapping_old

    @property
    def cp_group(self):
        mapping_old = self._cp_groups[self.pp_rank * self.tp_size +
                                      self.tp_rank]
        if Mapping.device_mesh:
            mapping_dm = get_process_group_ranks(self.cp_group_pg)
            assert mapping_old == mapping_dm
            print(
                f"[Rank {self.rank}] CP mapping_old: {mapping_old} mapping_dm: {mapping_dm}"
            )
            return mapping_dm
        return mapping_old

    @property
    def moe_tp_group(self):
        mapping_old = self._moe_tp_groups[
            self.pp_rank * self.moe_cluster_size * self.moe_ep_size +
            self.moe_cluster_rank * self.moe_ep_size + self.moe_ep_rank]
        if Mapping.device_mesh:
            mapping_dm = get_process_group_ranks(self.moe_tp_group_pg)
            assert mapping_old == mapping_dm
            print(
                f"[Rank {self.rank}] MOE_TP mapping_old: {mapping_old} mapping_dm: {mapping_dm}"
            )
            return mapping_dm
        return mapping_old

    @property
    def moe_ep_group(self):
        mapping_old = self._moe_ep_groups[
            self.pp_rank * self.moe_tp_size * self.moe_cluster_size +
            self.moe_tp_rank * self.moe_cluster_size + self.moe_cluster_rank]
        if Mapping.device_mesh:
            mapping_dm = get_process_group_ranks(self.moe_ep_group_pg)
            assert mapping_old == mapping_dm
            print(
                f"[Rank {self.rank}] MOE_EP mapping_old: {mapping_old} mapping_dm: {mapping_dm}"
            )
            return mapping_dm
        return mapping_old

    @property
    def moe_cluster_group(self):
        return self._moe_cluster_groups[self.pp_rank * self.moe_tp_size +
                                        self.moe_tp_rank]

    # == For accessing the Process Group of the current rank for Ray path==
    def _get_mesh_dim_by_name(self, name: str):
        if name == 'tp':
            if 'tp' in Mapping.device_mesh.mesh_dim_names:
                return Mapping.device_mesh['tp']
            else:
                return Mapping.tp_mesh
        else:
            assert name in Mapping.device_mesh.mesh_dim_names
            return Mapping.device_mesh[name]

    @property
    def tp_group_pg(self):
        return self._get_mesh_dim_by_name('tp').get_group()

    @property
    def pp_group_pg(self):
        return self._get_mesh_dim_by_name('pp').get_group()

    @property
    def cp_group_pg(self):
        return self._get_mesh_dim_by_name('cp').get_group()

    @property
    def moe_tp_group_pg(self):
        return self._get_mesh_dim_by_name('moe_tp').get_group()

    @property
    def moe_ep_group_pg(self):
        return self._get_mesh_dim_by_name('moe_ep').get_group()

    # =========

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
            'auto_parallel': self.auto_parallel,
        }

    def _build_mesh(self):
        if Mapping.device_mesh is not None:
            return
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "DeviceMesh creation requested but torch.distributed process group "
                "has not been initialised")

        # TODO: need to decide dim order
        dims = ["cp", "pp"]
        shape = [self.cp_size, self.pp_size]

        if self.moe_ep_size > 1:
            dims += ["moe_tp", "moe_ep"]
            shape += [self.moe_tp_size, self.moe_ep_size]
        else:
            dims += ["tp"]
            shape += [self.tp_size]

        Mapping.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=tuple(shape),
            mesh_dim_names=tuple(dims),
        )

        if self.moe_ep_size > 1:
            Mapping.tp_mesh = Mapping.device_mesh["moe_tp", "moe_ep"]._flatten(
                mesh_dim_name="tp")
        print(f"Mapping.device_mesh {Mapping.device_mesh}")
        print(f"Mapping.tp_mesh {Mapping.tp_mesh}")

    def _init_parallel_groups(self):
        # init pp group
        for i in range(self.tp_size * self.cp_size):
            ranks = range(i, self.world_size, self.tp_size * self.cp_size)
            self._pp_groups.append(list(ranks))

        # init cp group
        for i in range(self.pp_size):
            for j in range(self.tp_size):
                ranks = range(i * self.tp_size * self.cp_size + j,
                              (i + 1) * self.tp_size * self.cp_size + j,
                              self.tp_size)
                self._cp_groups.append(list(ranks))

        # init tp group
        for i in range(self.pp_size):
            for j in range(self.cp_size):
                ranks = range(
                    i * self.tp_size * self.cp_size + j * self.tp_size,
                    i * self.tp_size * self.cp_size + (j + 1) * self.tp_size)
                self._tp_groups.append(list(ranks))

        # init moe tp group
        for i in range(self.pp_size):
            for j in range(self.moe_cluster_size * self.moe_ep_size):
                ranks = range(i * self.moe_tp_cluster_ep_size + j,
                              (i + 1) * self.moe_tp_cluster_ep_size,
                              self.moe_cluster_size * self.moe_ep_size)
                self._moe_tp_groups.append(list(ranks))

        # init moe cluster group
        for i in range(self.pp_size):
            for j in range(self.moe_tp_size):
                ranks = range(
                    i * self.moe_tp_cluster_ep_size +
                    j * self.moe_cluster_size * self.moe_ep_size,
                    i * self.moe_tp_cluster_ep_size +
                    (j + 1) * self.moe_cluster_size * self.moe_ep_size)
                self._moe_cluster_groups.append(list(ranks))

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
                    self._moe_ep_groups.append(list(ranks))
