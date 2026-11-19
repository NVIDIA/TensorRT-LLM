# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Kimi K3.1 attention-DP vocab-parallel LM head entries on one GB300 tray.

Each cell runs on four MPI ranks with their own row counts, on the exchange path it names (the NVLink exchange
kernels on the group's reserved workspace, or NCCL with the workspace set aside): every rank regenerates every
rank's rows and the head from the cell's seed, calls the entry with its rows and checks it bit for bit against the
replicated head, on a sample of each entry's cells (``sample_cells``).

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.comm.adp_vocab_parallel_head import (  # noqa: E501
    DEFAULT_MAX_ROWS,
    k31_adp_vocab_logits,
    k31_adp_vocab_top32,
    reserve_exchange,
)

__extra_import_path__ = ["../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)

pickle_by_value(__name__)

WORLD = 4
VOCAB = 163840
HIDDEN = 7168
pytestmark = on_ranks(WORLD, 103)


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (0.02 * torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
    rows = [
        (torch.randn(n, HIDDEN, generator=gen, device="cuda")).to(torch.bfloat16)
        for n in spec["rows"]
    ]
    return weight, rows


def _exact(got, ref) -> bool:
    got = got if isinstance(got, tuple) else (got,)
    ref = ref if isinstance(ref, tuple) else (ref,)
    return all(
        g.dtype == r.dtype
        and g.shape == r.shape
        and torch.equal(
            g.view(torch.int32) if g.dtype == torch.float32 else g,
            r.view(torch.int32) if r.dtype == torch.float32 else r,
        )
        for g, r in zip(got, ref)
    )


def _setup(world: int):
    """This rank's mapping, with the group's exchange workspace reserved (collective: every rank calls it)."""
    import tensorrt_llm
    from tensorrt_llm.mapping import Mapping

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=world, tp_size=world, rank=rank)
    reserve_exchange(mapping, DEFAULT_MAX_ROWS)
    return rank, mapping


def _rank_cell(spec: dict, seed: int, which: str) -> dict:
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__dep.ops.comm import (
        adp_vocab_parallel_head as head,
    )

    rank, mapping = _setup(spec["world"])
    weight, rows = _inputs(spec, seed)
    op = k31_adp_vocab_logits if which == "logits" else k31_adp_vocab_top32
    # An NCCL cell runs as a group without a workspace does: the reserved one is set aside for the call.
    aside = dict(head._EXCHANGES) if spec["exchange"] == "nccl" else {}
    for key in aside:
        del head._EXCHANGES[key]
    try:
        with torch.inference_mode():
            got = op(rows[rank], weight, mapping, spec["rows"])
            ref = op.reference(rows, weight, mapping, spec["rows"])
            torch.cuda.synchronize()
            exact = _exact(got, ref)
    finally:
        head._EXCHANGES.update(aside)
    return dict(rank=rank, exact=exact)


# A sample of each entry's cells (``sample_cells``).
CASES = [("logits", c) for c in sample_cells(k31_adp_vocab_logits.CELLS)[:2]] + [
    ("top32", c) for c in sample_cells(k31_adp_vocab_top32.CELLS)[:2]
]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("which, cell", CASES, ids=[c.why[:56] for _, c in CASES])
def test_certified_cells(which, cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    results = run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, which, world=WORLD)
    for r in results:
        assert r["exact"], f"rank {r['rank']}: {which} differ from the replicated head"
