import importlib
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorrt_llm._utils import get_free_port, torch_pybind11_abi


class TestCacheTransceiverComm(unittest.TestCase):

    def test_cache_transceiver_comm(self):
        mp.spawn(
            self._worker,
            args=(4, get_free_port()),
            nprocs=4,
            join=True,
        )

    @staticmethod
    def _worker(rank: int, world_size: int, master_port: int):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        try:
            dist.init_process_group(
                backend="gloo",
                rank=rank,
                world_size=world_size,
            )
            world_pg = torch.distributed.group.WORLD
            bm = importlib.import_module(
                "tensorrt_llm.bindings.internal.batch_manager")
            cacheComm = getattr(bm, "CacheTransceiverComm")
            comm = cacheComm(world_pg, torch_pybind11_abi())

            # Test split
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            color = rank // 2
            key = rank % 2
            sub = comm.split(color, key)

            expected_group_size = 2
            assert sub.get_size() == expected_group_size

            # Test allgather
            ok, gathered_ranks = sub.allgather(rank)
            assert ok is True
            expected_world_ranks = [
                r for r in range(world_size) if (r // 2) == color
            ]
            assert gathered_ranks == expected_world_ranks

            # Test allgatherv
            local_len = rank + 1
            payload = [rank] * local_len

            ok_sizes, sizes64 = sub.allgather(local_len)
            assert ok_sizes is True
            sizes = [int(x) for x in sizes64]

            ok_v, out = sub.allgatherv(payload, sizes)
            assert ok_v is True

            expected_concat = []
            for r in expected_world_ranks:
                expected_concat.extend([r] * (r + 1))
            assert out == expected_concat

            # Test allgatherv with char
            char_payload = [chr(65 + rank)] * local_len
            ok_char, char_out = sub.allgatherv(char_payload, sizes)
            assert ok_char is True

            expected_char_concat = []
            for r in expected_world_ranks:
                expected_char_concat.extend([chr(65 + r)] * (r + 1))
            assert char_out == expected_char_concat
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
