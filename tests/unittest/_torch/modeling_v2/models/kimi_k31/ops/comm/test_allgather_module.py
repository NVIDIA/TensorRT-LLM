# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 allgather_module entry: the engine allgather on one GB300 tray.

Each cell runs on four MPI ranks (one tray), the way a served engine runs: every rank regenerates
every rank's input from the cell's seed, calls the entry on its own slice with its TP mapping and
checks the result bit for bit against the entry's reference over all four inputs. Ragged cells
hand each rank a row slice of a larger buffer, as the indexer does. The ranks report a bit hash of
their output, so the parent also checks that every rank got the same tensor.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import pickle
import sys
import zlib

import cloudpickle
import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.allgather_module import (
    allgather_module,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
pytestmark = [
    # The MPI pool is reused across cells; its manager thread outlives the first one.
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 3), (10, 7))
        or torch.cuda.device_count() < WORLD,
        reason="certified on one GB300 tray: sm_103 and four GPUs (sm_107 driven separately)",
    ),
]


def _inputs(spec: dict, seed: int) -> list:
    """Every rank's input, in rank order, identical on every rank.

    Ragged cells (`sizes` given) split one `shape`-sized buffer along `dim` by `sizes` and hand each
    rank its slice, a view at a nonzero offset into the buffer for every rank but the first.
    Even cells give every rank its own `shape`-sized tensor.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype, sizes = spec["shape"], spec["dtype"], spec["sizes"]

    def make(*s):
        if dtype == torch.int32:
            # Index-shaped values, -1 (no index) included.
            return torch.randint(-1, 1 << 20, s, generator=gen, device="cuda", dtype=torch.int32)
        return torch.randn(*s, generator=gen, device="cuda").to(dtype)

    if sizes is None:
        return [make(*shape) for _ in range(spec["world"])]
    assert sum(sizes) == shape[spec["dim"]] and len(sizes) == spec["world"]
    return list(make(*shape).split(list(sizes), dim=spec["dim"]))


def _rank_cell(spec: dict, seed: int) -> dict:
    """One rank's half of a cell; runs inside an MPI pool worker."""
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    inputs = _inputs(spec, seed)
    sizes = None if spec["sizes"] is None else list(spec["sizes"])
    args = (inputs[rank], mapping, spec["dim"], sizes)
    before = inputs[rank].clone()
    with torch.inference_mode():
        allgather_module.is_valid(*args)
        out = allgather_module(*args)
        again = allgather_module(*args)
        ref = allgather_module.reference(inputs, spec["dim"])
        torch.cuda.synchronize()
        exact = bool(torch.equal(out, ref))
        repeat_equal = bool(torch.equal(out, again))
        bits = out.contiguous().view(torch.int32).to(torch.int64)
        weights = torch.arange(1, bits.numel() + 1, device="cuda", dtype=torch.int64) % 65521
        digest = int((bits.flatten() * weights).sum().item())
    return dict(
        rank=rank,
        shape=tuple(out.shape),
        dtype=str(out.dtype),
        exact=exact,
        repeat_equal=repeat_equal,
        hash=digest,
        input_untouched=bool(torch.equal(inputs[rank], before)),
        fresh=out.data_ptr() != inputs[rank].data_ptr(),
    )


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell", allgather_module.CELLS, ids=[c.why[4:64] for c in allgather_module.CELLS]
)
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    assert spec["world"] == WORLD
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = sorted(
        mpi_pool_executor.map(_rank_cell, *zip(*[(spec, seed)] * WORLD)), key=lambda r: r["rank"]
    )
    expected = list(spec["shape"])
    if spec["sizes"] is None:
        expected[spec["dim"]] *= WORLD
    for r in results:
        assert r["shape"] == tuple(expected) and r["dtype"] == str(spec["dtype"]), r
        assert r["fresh"] and r["input_untouched"], r
        assert r["repeat_equal"], f"rank {r['rank']}: two identical calls disagree"
        assert r["exact"], f"rank {r['rank']}: the gather is not the rank-order concatenation"
    assert len({r["hash"] for r in results}) == 1, "the ranks got different tensors"


def test_reference_refuses_a_rank_order_swap() -> None:
    """The bit-exact gate must refuse a gather that places two ranks' slices in the wrong order."""
    for cell in (allgather_module.CELLS[1], allgather_module.CELLS[-1]):
        spec = cell.spec
        inputs = _inputs(spec, seed=3)
        right = allgather_module.reference(inputs, spec["dim"])
        # Swap the last two ranks: equally sized in every cell, ragged ones included.
        assert inputs[2].shape == inputs[3].shape
        swapped = [inputs[0], inputs[1], inputs[3], inputs[2]]
        wrong = allgather_module.reference(swapped, spec["dim"])
        assert not torch.equal(wrong, right), cell.why


def test_one_rank_group_returns_the_input() -> None:
    """A one-rank group is the identity, without a kernel: the target relies on it at TP1."""
    from tensorrt_llm.mapping import Mapping

    x = torch.randn(2, 5, 16, device="cuda")
    assert allgather_module(x, Mapping(world_size=1, tp_size=1, rank=0), -1, None) is x
