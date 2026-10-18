# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``replicated_embedding`` on the real embedding and the real all-reduce (TP8).

Run with eight MPI ranks on two NVLink-connected four-GPU GB300 nodes and the
target checkpoint, e.g.::

    MUSESPARK_REFERENCE_CKPT=<Muse-Spark-Open-816B-A42B-NVFP4 checkpoint> \\
    srun -N2 --ntasks-per-node=4 --mpi=pmix python -m pytest <this file>

Anywhere else the test skips.

The built-in vocab-parallel ``Embedding`` is built as ``MuseSparkOpenModel``
builds it (``COLUMN``, ``gather_output``, the default all-reduce strategy) and
loaded with this rank's shard of the checkpoint's table. Every one of the
202,048 rows is looked up through the built-in path (masking, shard gather,
all-reduce) in the decode step's 8-id calls; then ``replicated_embedding.install``
builds the whole table (collective) and every row is looked up again through
the re-classed module, and the two must be equal bit for bit on every rank,
and every rank must hold the same rows. Also: the drafter's 2-id call, a
prefill-sized call (built-in path, the same rows), CUDA-graph replay of the
8-id call with the ids updated in place, and the memory the table adds.
"""

import json
import os
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_embedding import (  # noqa: E501
    embedding as replicated_embedding,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import TensorParallelMode

VOCAB, HIDDEN = 202048, 8192
KEY = "model.language_model.embed_tokens.weight"


def _shard(ckpt: str, rank: int, world: int) -> torch.Tensor:
    from safetensors import safe_open

    index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    path = os.path.join(ckpt, index["weight_map"][KEY])
    width = VOCAB // world
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_slice(KEY)[rank * width : (rank + 1) * width].contiguous()


def _digest(t: torch.Tensor) -> int:
    return zlib.crc32(t.view(torch.int16).cpu().numpy().tobytes())


@torch.inference_mode()
def test_replicated_embedding_tp8_equals_the_sharded_embedding():
    try:
        from mpi4py import MPI
    except ImportError:
        pytest.skip("Requires mpi4py")
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    if comm.Get_size() != 8:
        pytest.skip("Requires eight MPI ranks on two NVLink-connected four-GPU nodes")
    ckpt = os.environ.get("MUSESPARK_REFERENCE_CKPT")
    if not ckpt:
        pytest.skip("Requires MUSESPARK_REFERENCE_CKPT")
    rank = comm.Get_rank()
    torch.cuda.set_device(rank % 4)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Requires GB300")
    mapping = Mapping(world_size=8, tp_size=8, rank=rank, gpus_per_node=4)
    with torch.device("cuda"):
        embedding = Embedding(
            VOCAB,
            HIDDEN,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
    embedding.weight.data.copy_(_shard(ckpt, rank, 8).to("cuda"))
    text_model = SimpleNamespace(embed_tokens=embedding)

    ids = torch.arange(VOCAB, dtype=torch.int32, device="cuda")
    builtin = torch.empty(VOCAB, HIDDEN, dtype=torch.bfloat16, device="cuda")
    for start in range(0, VOCAB, 8):
        builtin[start : start + 8] = embedding(ids[start : start + 8])
    torch.cuda.synchronize()
    comm.Barrier()

    free_before = torch.cuda.mem_get_info()[0]
    replicated_embedding.install(text_model, mapping)
    added = free_before - torch.cuda.mem_get_info()[0]
    assert type(embedding) is replicated_embedding.ReplicatedEmbedding
    table = replicated_embedding.table_of(embedding)
    assert tuple(table.shape) == (VOCAB, HIDDEN)

    mismatched_rows = 0
    replicated = torch.empty_like(builtin)
    for start in range(0, VOCAB, 8):
        replicated[start : start + 8] = embedding(ids[start : start + 8])
    torch.cuda.synchronize()
    differ = (replicated.view(torch.int16) != builtin.view(torch.int16)).any(dim=1)
    mismatched_rows = int(differ.sum().item())
    negative_zero_out = int((replicated.view(torch.int16) == -32768).sum().item())
    negative_zero_table = int((table.view(torch.int16) == -32768).sum().item())

    # The drafter's call: the bonus token and the mask token.
    g = torch.Generator(device="cuda").manual_seed(rank)  # different ids per rank are fine here
    pair = torch.randint(0, VOCAB, (2,), generator=g, device="cuda", dtype=torch.int64)
    pair_ok = torch.equal(embedding(pair).view(torch.int16), builtin[pair].view(torch.int16))
    # A prefill-sized call goes to the built-in path (collective: the same size on every rank).
    seventeen = torch.arange(1000, 1017, device="cuda", dtype=torch.int32)
    prefill_ok = torch.equal(
        embedding(seventeen).view(torch.int16), builtin[1000:1017].view(torch.int16)
    )

    # CUDA graph: the 8-id call replayed with the ids updated in place.
    step_ids = torch.arange(8, dtype=torch.int32, device="cuda")
    embedding(step_ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = embedding(step_ids)
    graph_ok = True
    gg = torch.Generator(device="cuda").manual_seed(99)  # the same ids on every rank
    for _ in range(64):
        step_ids.copy_(torch.randint(0, VOCAB, (8,), generator=gg, device="cuda").int())
        graph.replay()
        torch.cuda.synchronize()
        graph_ok &= torch.equal(
            captured.view(torch.int16), builtin[step_ids.long()].view(torch.int16)
        )

    digests = comm.allgather((_digest(table[::997]), _digest(replicated[::997])))
    summary = dict(
        rank=rank,
        mismatched_rows=mismatched_rows,
        negative_zero_table=negative_zero_table,
        negative_zero_out=negative_zero_out,
        pair_ok=bool(pair_ok),
        prefill_ok=bool(prefill_ok),
        graph_ok=bool(graph_ok),
        added_gib=round(added / 2**30, 3),
    )
    print(f"replicated_embedding tp8: {summary}", flush=True)
    out_dir = os.environ.get("MODELING_V2_COMPOSE_OUT")
    if out_dir:
        with open(os.path.join(out_dir, f"replicated_embedding-tp8-rank{rank}.json"), "w") as f:
            json.dump(summary, f)
    assert mismatched_rows == 0, summary
    assert negative_zero_out == 0, summary
    assert pair_ok and prefill_ok and graph_ok, summary
    assert all(d == digests[0] for d in digests), "ranks hold different tables or rows"
    del graph
    comm.Barrier()
