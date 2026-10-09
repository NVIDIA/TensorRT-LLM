import copy
import os
import unittest

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl
from tensorrt_llm._utils import get_free_port
from tensorrt_llm.mapping import Mapping


class TestMapping(unittest.TestCase):

    @pytest.mark.gpu2
    def test_device_mesh_parity(self):
        """To ensure parity between Ray and MPI Mapping instance."""
        # (tp, pp, cp, moe_tp, moe_ep)
        combos = [
            # no cp
            (2, 1, 1, -1, -1),  # -1 means no MoE in Mapping
            # 8 GPUs, no cp
            (4, 2, 1, -1, -1),
            (2, 4, 1, -1, -1),
            # 8 GPUs with cp
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
                args=(world_size, get_free_port(), tp, pp, cp, moe_tp, moe_ep),
                nprocs=world_size,
                join=True,
            )

    @pytest.mark.gpu2
    def test_picklable(self):
        world_size = tp = 2

        mp.spawn(
            self._worker,
            args=(world_size, get_free_port(), tp, 1, 1, -1, -1, "pickle"),
            nprocs=world_size,
            join=True,
        )

    @pytest.mark.gpu2
    def test_one_rank_mapping_inside_a_larger_job(self):
        """A Mapping with world_size 1 built while the job's mesh spans two ranks
        must report rank 0 and size-1 groups of its own, not the job's."""
        world_size = tp = 2

        mp.spawn(
            self._worker,
            args=(world_size, get_free_port(), tp, 1, 1, -1, -1, "one_rank"),
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
                moe_ep=1,
                test_type="parity") -> None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        torch.cuda.set_device(rank)

        if test_type == "parity":
            if "TLLM_DISABLE_MPI" in os.environ:
                del os.environ["TLLM_DISABLE_MPI"]

            mapping_mpi = Mapping(
                world_size=world_size,
                rank=rank,
                gpus_per_node=world_size,
                tp_size=tp,
                pp_size=pp,
                cp_size=cp,
                moe_tp_size=moe_tp,
                moe_ep_size=moe_ep,
            )

        os.environ["TLLM_DISABLE_MPI"] = "1"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        mapping_device_mesh = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=world_size,
            tp_size=tp,
            pp_size=pp,
            cp_size=cp,
            moe_tp_size=moe_tp,
            moe_ep_size=moe_ep,
        )

        if test_type == "parity":
            mapping_device_mesh.build_mesh()

            properties = []
            for dim in mapping_device_mesh.device_mesh.mesh_dim_names:
                properties.append(f"{dim}_rank")
                properties.append(f"{dim}_group")

            for prop in properties:
                mpi_value = getattr(mapping_mpi, prop)
                device_mesh_value = getattr(mapping_device_mesh, prop)

                if rank == 0:
                    print(
                        f"  {prop}: MPI={mpi_value}, DeviceMesh={device_mesh_value}"
                    )

                assert mpi_value == device_mesh_value, \
                    f"Property {prop} mismatch: MPI={mpi_value}, DeviceMesh={device_mesh_value} (rank {rank})"
        elif test_type == "one_rank":
            job = mapping_device_mesh
            job.build_mesh()
            assert job.tp_rank == rank and job.tp_group_pg.size() == world_size

            one = Mapping(world_size=1, tp_size=1, rank=0)
            assert (one.tp_rank, one.pp_rank, one.cp_rank) == (0, 0, 0)
            assert one.tp_group == [0] and one.pp_group == [
                0
            ] and one.cp_group == [0]
            for group in (one.tp_group_pg, one.pp_group_pg, one.cp_group_pg):
                assert group.size() == 1 and group.rank() == 0, \
                    f"one-rank mapping borrowed the job's group on rank {rank}"
        elif test_type == "pickle":
            mapping = mapping_device_mesh

            tp_group = mapping.tp_group
            print(f"tp_group: {tp_group}")
            assert DeviceMeshTopologyImpl.device_mesh is not None
            mapping_copy = copy.deepcopy(mapping)

            # check static mesh still exists
            assert mapping_copy.device_mesh is not None
            print(f"tp_group after deepcopy: {mapping.tp_group}")
            assert mapping.tp_group == mapping_copy.tp_group

        else:
            raise ValueError(f"Invalid test type: {test_type}")

        dist.destroy_process_group()
