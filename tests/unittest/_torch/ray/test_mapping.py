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
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm.mapping import Mapping


class TestMapping(unittest.TestCase):

    def test_device_mesh_parity(self):
        """Tentative test to ensure the parity between old Mapping and DeviceMesh
        while Ray migration is in flight.
        """
        # (tp, pp, cp, moe_tp, moe_ep)
        combos = [
            # no cp
            (2, 1, 1, -1, -1),  # -1 means no MoE in Mapping
            # 8 GPUs, no cp
            (4, 2, 1, -1, -1),
            (2, 4, 1, -1, -1),
            # 8 GPPUs with cp
            (4, 1, 2, -1, -1),
            (2, 1, 4, -1, -1),
            # with moe_tp, moe_ep
            (8, 1, 1, 2, 4),
            (2, 1, 1, 1, 2)
        ]

        num_gpus = torch.cuda.device_count()

        for tp, pp, cp, moe_tp, moe_ep in combos:
            world_size = tp * pp * cp
            print(
                f"\n\n=== TP={tp}, PP={pp}, CP={cp}, MOE_TP={moe_tp}, MOE_EP={moe_ep} ==="
            )

            if world_size > num_gpus:
                print(
                    f"SKIPPING: need {world_size} GPUs. Only have {num_gpus}.")
                continue

            mp.spawn(
                self._worker,
                args=(world_size, self._find_free_port(), tp, pp, cp, moe_tp,
                      moe_ep),
                nprocs=world_size,
                join=True,
            )

    @staticmethod
    def _worker(rank: int,
                world_size: int,
                master_port: int,
                tp=1,
                pp=1,
                cp=1,
                moe_tp=1,
                moe_ep=1) -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["DISABLE_MPI"] = "1"
        torch.cuda.set_device(rank)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        mapping = Mapping(
            world_size=world_size,
            gpus_per_node=world_size,
            tp_size=tp,
            pp_size=pp,
            cp_size=cp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
        )

        mapping.dist = dist
        mapping.rank = rank

        if rank == 0:
            print(f"Device mesh: {Mapping.device_mesh}")

        for dim in Mapping.device_mesh.mesh_dim_names:
            # local-view assertions are tentatively in Mapping attribute getters.
            getattr(mapping, f"{dim}_group")

        dist.destroy_process_group()

    @staticmethod
    def _find_free_port() -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port
