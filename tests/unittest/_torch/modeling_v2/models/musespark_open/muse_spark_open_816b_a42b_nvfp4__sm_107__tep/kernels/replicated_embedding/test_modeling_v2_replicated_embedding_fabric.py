# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``replicated_embedding.install`` with the fabric table.

CPU only: the fabric mapping is a stand-in whose ``tensor`` is a CPU table into which the
peers' rows are already written (as each peer's own install writes them); the install,
the layout plan and the chunked bitwise check are the real ones.

* ``plan_layout`` puts every part on a granule, within half a granule of an even split,
  and refuses a granule that is not whole rows or a part with no rows.
* The install writes this rank's shard into the fabric table, keeps the mapping object,
  aliases the shard parameter to its rows, and the table holds every rank's bits.
* The chunked check counts the rows that differ from the gathered shards, and the install
  refuses a table that differs.
* A rank that cannot build the fabric table makes the install fall back to the replicated
  table (the outcome is agreed over the group).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.replicated_embedding import (  # noqa: E501
    embedding as replicated,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.replicated_embedding import (  # noqa: E501
    fabric_table as fabric,
)
from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import TensorParallelMode

pytestmark = pytest.mark.cpu_only

TP, WIDTH, HIDDEN = 4, 37, 16


# ---- layout ----------------------------------------------------------------------------------


def test_plan_layout_served_table():
    """The served table: 202,048 rows of 16 KiB over TP4 with 2 MiB granules."""
    layout = fabric.plan_layout(202048, 16384, 4, 2 << 20)
    assert layout.rows_per_granule == 128
    assert layout.bounds == (0, 50560, 100992, 151552, 202048)
    assert all(bound % 128 == 0 for bound in layout.bounds[:-1])
    assert layout.part_bytes == (
        50560 * 16384,
        50432 * 16384,
        50560 * 16384,
        (50496 + 64) * 16384,  # the last part is rounded up to a whole granule
    )
    assert layout.reserved_bytes == sum(layout.part_bytes)
    assert layout.reserved_bytes >= 202048 * 16384
    assert max(layout.part_bytes) <= 202048 * 16384 // 4 + (1 << 20) + 16384 * 64
    assert [layout.offset(rank) for rank in range(4)] == [
        bound * 16384 for bound in layout.bounds[:4]
    ]


@pytest.mark.parametrize("rows", [4 * 128, 4 * 128 + 1, 1000, 202048])
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_plan_layout_parts_cover_the_table(rows, world):
    if (rows, world) in ((512, 8), (513, 8)):
        # Half a granule per rank: a bound rounds onto its predecessor, so a rank holds no rows.
        with pytest.raises(ValueError, match="no rows"):
            fabric.plan_layout(rows, 16384, world, 2 << 20)
        return
    layout = fabric.plan_layout(rows, 16384, world, 2 << 20)
    assert layout.bounds[0] == 0 and layout.bounds[-1] == rows
    for rank in range(world):
        count = layout.bounds[rank + 1] - layout.bounds[rank]
        assert count > 0
        assert layout.part_bytes[rank] % (2 << 20) == 0
        assert layout.part_bytes[rank] >= count * 16384
        assert abs(layout.bounds[rank] - rank * rows / world) <= 64


def test_plan_layout_refusals():
    with pytest.raises(ValueError, match="whole number"):
        fabric.plan_layout(1000, 16384 * 3, 4, 2 << 20)
    with pytest.raises(ValueError, match="no rows"):
        fabric.plan_layout(130, 16384, 4, 2 << 20)
    with pytest.raises(ValueError, match="cannot split"):
        fabric.plan_layout(3, 16384, 4, 2 << 20)


# ---- install ---------------------------------------------------------------------------------


def _full_table(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    table = torch.randn((TP * WIDTH, HIDDEN), generator=gen).to(torch.bfloat16)
    table[3, :4] = -0.0
    return table


def _embedding(full: torch.Tensor, rank: int) -> Embedding:
    module = object.__new__(Embedding)
    nn.Module.__init__(module)
    module.weight = nn.Parameter(
        full[rank * WIDTH : (rank + 1) * WIDTH].clone(), requires_grad=False
    )
    module.num_embeddings = TP * WIDTH
    module.tp_size = TP
    module.tp_mode = TensorParallelMode.COLUMN
    module.padding_size = 0
    module.vocab_start_index = rank * WIDTH
    module.vocab_end_index = (rank + 1) * WIDTH
    return module


class _FakeFabric:
    """A fabric table whose peers' rows are already in place; this rank's rows start as garbage."""

    def __init__(self, full: torch.Tensor, rank: int):
        self.tensor = full.clone()
        self.tensor[rank * WIDTH : (rank + 1) * WIDTH] = 7.0
        self.rank = rank
        self.barriers = 0
        self.closed = False

    @property
    def local_bytes(self) -> int:
        return WIDTH * HIDDEN * 2

    def local_rows(self):
        return self.rank * WIDTH, (self.rank + 1) * WIDTH

    def barrier(self) -> None:
        self.barriers += 1

    def close(self) -> None:
        self.closed = True


class _Gather:
    """Stand-in all-gather: this rank's part from the caller, the peers' from ``full``."""

    def __init__(self, full: torch.Tensor, rank: int):
        self.full = full
        self.rank = rank
        self.begin = 0
        self.calls = 0

    def __call__(self, part, mapping, dim=0):
        assert dim == 0
        self.calls += 1
        rows = part.shape[0]
        parts = []
        for rank in range(TP):
            if rank == self.rank:
                parts.append(part.clone())
            else:
                begin = rank * WIDTH + self.begin
                parts.append(self.full[begin : begin + rows].clone())
        self.begin += rows
        return torch.cat(parts, dim=0)


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (0, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 7))
    monkeypatch.setattr(
        replicated._tp_agreement, "agreed_refusal", lambda mapping, local, facts=None: local
    )
    monkeypatch.setattr(replicated, "GATHER_ROWS", 8)
    return monkeypatch


def _mapping(rank: int):
    return SimpleNamespace(tp_size=TP, tp_rank=rank, enable_attention_dp=False)


@pytest.mark.parametrize("rank", range(TP))
def test_install_on_fabric_table(cpu, rank):
    full = _full_table(rank)
    fake = _FakeFabric(full, rank)
    cpu.setattr(fabric, "make_table", lambda rows, hidden, dtype, mapping: fake)
    gather = _Gather(full, rank)
    cpu.setattr(fabric, "allgather", gather)
    cpu.setattr(replicated, "allgather", gather)
    text_model = SimpleNamespace(embed_tokens=_embedding(full, rank))
    parameter = text_model.embed_tokens.weight
    replicated.install(text_model, _mapping(rank))
    embedding = text_model.embed_tokens
    table = replicated.table_of(embedding)
    assert table is fake.tensor
    assert embedding.__dict__[replicated.FABRIC_ATTR] is fake
    assert torch.equal(table.view(torch.int16), full.view(torch.int16))
    assert fake.barriers == 1
    assert embedding.weight is parameter
    mine = table[rank * WIDTH : (rank + 1) * WIDTH]
    assert embedding.weight.data_ptr() == mine.data_ptr()
    # The replicated table's gather never ran; only the bitwise check gathered.
    assert gather.calls == -(-WIDTH // 8)


def test_mismatched_rows_counts_differing_rows(cpu):
    rank = 1
    full = _full_table(5)
    table = full.clone()
    table[0, 3] = 100.0  # a peer's row
    table[rank * WIDTH + 2, 0] = -table[rank * WIDTH + 2, 0]  # a row of this rank
    table[TP * WIDTH - 1, HIDDEN - 1] += 1  # the last row
    cpu.setattr(fabric, "allgather", _Gather(full, rank))
    shard = full[rank * WIDTH : (rank + 1) * WIDTH]
    assert fabric.mismatched_rows(table, shard, _mapping(rank), 8) == 3


def test_install_refuses_a_differing_fabric_table(cpu):
    rank = 2
    full = _full_table(9)
    fake = _FakeFabric(full, rank)
    fake.tensor[0, 0] += 1  # a peer wrote a wrong value
    cpu.setattr(fabric, "make_table", lambda rows, hidden, dtype, mapping: fake)
    cpu.setattr(fabric, "allgather", _Gather(full, rank))
    text_model = SimpleNamespace(embed_tokens=_embedding(full, rank))
    with pytest.raises(ValueError, match="differs from the gathered shards in 1 rows"):
        replicated.install(text_model, _mapping(rank))


def test_install_falls_back_without_a_fabric_table(cpu):
    rank = 0
    full = _full_table(3)
    cpu.setattr(fabric, "make_table", lambda rows, hidden, dtype, mapping: None)
    gather = _Gather(full, rank)
    cpu.setattr(replicated, "allgather", gather)
    text_model = SimpleNamespace(embed_tokens=_embedding(full, rank))
    replicated.install(text_model, _mapping(rank))
    embedding = text_model.embed_tokens
    assert replicated.FABRIC_ATTR not in embedding.__dict__
    assert torch.equal(replicated.table_of(embedding).view(torch.int16), full.view(torch.int16))
    assert gather.calls > 0
