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
"""A DKV group's unified draft pages across real ranks: split over the group
(``KVCacheManagerV2.draft_page_split``) against the replicated layout, bitwise.

Each rank runs two real ``KVCacheManagerV2(dkv=True)`` with five FP8 MLA draft layers, one
splitting the draft pages (global page g on rank g % N, one page per DKV block) and one keeping
every page on every rank. The DFlash worker's context store writes the same rows into both, in two
chunks per request, as every rank of a cep group computes them: the split cache gets only the
rows of the pages this rank owns, at their rank-local positions, and reads them back
(``verify_split_context_store``). Every page this rank would send (the transceiver's draft table,
the transfer's page table and extractor) then equals the same page of the replicated layout in
every layer, and over the group each page of each prompt is sent once, by rank g % N. Needs one
node with ``WORLD_SIZE`` GPUs; ``python3 test_dkv_draft_tp.py`` under ``mpirun -n 4`` runs the
body.
"""

import pickle
import sys
import traceback

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.disaggregation.resource.cache_reuse import create_cache_reuse_adapter
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVRegionExtractorV1,
    build_page_table_from_manager,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_physical_pool,
    is_standalone_draft_group,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)
pytestmark = pytest.mark.threadleak(enabled=False)

DataType = tensorrt_llm.bindings.DataType
CacheType = tensorrt_llm.bindings.internal.batch_manager.CacheType

TPB = 64
ROW = 576  # one FP8 MLA latent
LAYERS = 5
# Prompts over several DKV blocks, with partial last pages and blocks; the first chunk's end.
PROMPTS = {1: (700, 300), 2: (1500, 1000), 3: (4 * TPB * 3 + 5, 64)}


def _manager(world: int, rank: int, draft_cp_size: int) -> KVCacheManagerV2:
    return KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(max_tokens=16384, enable_block_reuse=False),
        kv_cache_type=CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        tokens_per_block=TPB,
        max_seq_len=4096,
        max_batch_size=8,
        mapping=Mapping(world_size=world, tp_size=world, rank=rank),
        dtype=DataType.HALF,
        vocab_size=32000,
        dkv=True,
        draft_num_layers=LAYERS,
        draft_num_kv_heads=1,
        draft_head_dim=ROW,
        draft_dtype=DataType.FP8,
        draft_kv_factor=1,
        draft_total_num_kv_heads=1,
        draft_cp_size=draft_cp_size,
    )


def _store(manager: KVCacheManagerV2, requests: list) -> None:
    """The DFlash worker's context store of every request's rows, in two chunks."""
    worker = object.__new__(DFlashWorker)
    worker._ctx_kv_buf = [manager.get_draft_buffers(i, kv_layout="HND") for i in range(LAYERS)]
    worker._ctx_page_size = TPB
    worker._dflash_attention_backend = "VANILLA"
    worker._helix_mapping = None
    worker._ctx_kv_manager = manager
    worker._ctx_page_table = None
    rids = [req.py_request_id for req in requests]
    worker._ctx_block_tables = manager.get_draft_block_table(rids).cuda()
    split = manager.draft_page_split
    worker._ctx_store_shard = None if split is None else HelixDraftShard(*split, TPB)
    worker.verify_split_context_store = split is not None
    for row, rid in enumerate(rids):
        prompt, cut = PROMPTS[rid]
        for start, end in ((0, cut), (cut, prompt)):
            gen = torch.Generator().manual_seed(1000 * rid + start)
            k = torch.randn(end - start, LAYERS, 1, ROW, generator=gen).to(torch.bfloat16).cuda()
            if split is None:
                worker._store_context_kv_paged(
                    k,
                    None,
                    torch.full((end - start,), row, dtype=torch.long, device="cuda"),
                    torch.arange(start, end, device="cuda"),
                )
            else:
                worker._store_owned_context_rows(k, None, row, start, end, rid)
    torch.cuda.synchronize()


def _sent_pages(manager: KVCacheManagerV2, req) -> dict:
    """Global page -> ``[LAYERS, TPB, ROW]`` uint8 bytes this rank sends of a request's draft
    history: the transceiver's draft table through the transfer's page table and extractor."""
    table = build_page_table_from_manager(manager)
    (lg_id,) = [i for i, lg in enumerate(table.layer_groups) if is_standalone_draft_group(lg)]
    lg = table.layer_groups[lg_id]
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._reuse_adapter = create_cache_reuse_adapter(manager)
    transceiver._kv_cache_manager = manager
    ids = transceiver._describe_draft_group(req, lg_id, lg, 0)
    (view,) = lg.pool_views
    pool = get_physical_pool(table, lg_id, view.pool_idx)
    memory = convert_to_torch_tensor(
        TensorWrapper(pool.base_address, torch.uint8, [pool.num_slots * pool.slot_stride_bytes])
    )
    pages = np.flatnonzero(ids >= 0)
    ptrs = KVRegionExtractorV1(table).extract(ids, lg_id, 0).memory.ptrs
    entries = view.buffer_entries
    out = {}
    for g, ptr in zip(pages, ptrs):
        layers = []
        for offset, size in zip(entries["offset"], entries["size"]):
            start = int(ptr) - pool.base_address + int(offset)
            layers.append(memory[start : start + int(size)].view(TPB, ROW))
        out[int(g)] = torch.stack(layers).cpu()
    return out


def _body(world: int, rank: int, comm) -> None:
    torch.cuda.set_device(rank % torch.cuda.device_count())
    split = _manager(world, rank, world)
    replicated = _manager(world, rank, 1)
    try:
        assert split.draft_page_split == (world, rank) and split._draft_page_expansion == 1
        assert replicated.draft_page_split is None and replicated._draft_page_expansion == world
        token_nums = [prompt for prompt, _ in PROMPTS.values()]
        held = {}
        for manager in (split, replicated):
            requests = manager.add_dummy_requests(list(PROMPTS), token_nums=token_nums)
            assert requests is not None
            held[id(manager)] = requests
            _store(manager, requests)
        sent = {}
        for req_split, req_rep in zip(held[id(split)], held[id(replicated)]):
            prompt, _ = PROMPTS[req_split.py_request_id]
            own = _sent_pages(split, req_split)
            every = _sent_pages(replicated, req_rep)
            pages = -(-prompt // TPB)
            assert sorted(every) == list(range(pages))
            assert sorted(own) == list(range(rank, pages, world)), (rank, sorted(own))
            for g, data in own.items():
                rows = min(TPB, prompt - g * TPB)
                assert torch.equal(data[:, :rows], every[g][:, :rows]), (rank, req_split, g)
            sent[req_split.py_request_id] = sorted(own)
        for rid, (prompt, _) in PROMPTS.items():
            pages = sorted(g for everyone in comm.allgather(sent[rid]) for g in everyone)
            assert pages == list(range(-(-prompt // TPB))), (rid, pages)
        for manager in (split, replicated):
            for req in held[id(manager)]:
                manager.free_resources(req)
        if rank == 0:
            print(f"DKV draft pages: {world} ranks, prompts {PROMPTS}", flush=True)
    finally:
        split.shutdown()
        replicated.shutdown()


def _run(world: int) -> bool:
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
    try:
        _body(world, rank, comm)
    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.parametrize("world_size", [4], ids=lambda n: f"tp{n}")
def test_split_draft_pages_equal_the_replicated_layout(world_size: int) -> None:
    from mpi4py.futures import MPIPoolExecutor

    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs, has {torch.cuda.device_count()}")
    with MPIPoolExecutor(max_workers=world_size) as executor:
        assert all(executor.map(_run, [world_size] * world_size))


if __name__ == "__main__":
    _run(MPI.COMM_WORLD.Get_size())
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("DKV draft TP test passed", flush=True)
