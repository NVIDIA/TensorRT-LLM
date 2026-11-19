# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 latent_moe_tail entries on one GB300 tray (TP 4).

Each cell (a sample of the entry's, ``sample_cells``) runs on four MPI ranks: every rank builds the
process-wide fused tail of the group, regenerates every rank's routed and shared partials from the
cell's seed, calls the entry on its own partials and gates the result against the entry's reference
over all of them. Folded cells hand over the routed experts' output before the top-k sum; pushed
cells store the shared rows ahead (latent_moe_tail_push_shared on a side stream). The mailbox cells
run latent_moe_tail_to_mailbox and consume the mailbox with latent_moe_tail_copy_out.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx
environment.
"""

import sys
import types
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.moe.latent_moe_tail import (  # noqa: E501
    HIDDEN,
    LATENT,
    KimiK3LatentMoETail,
    UnfinalizedRoutedOutput,
    k31_latent_moe_tail,
    k31_latent_moe_tail_copy_out,
    k31_latent_moe_tail_push_shared,
    k31_latent_moe_tail_to_mailbox,
)

__extra_import_path__ = ["../../../../..", "../../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    bf16_deviation,
    on_ranks,
    pickle_by_value,
    run_per_rank,
    sample_cells,
)

pickle_by_value(__name__)

WORLD = 4
pytestmark = on_ranks(WORLD, 103, cutedsl=True)

# Gate in bf16 ulps against the row's largest magnitude (element) and the relative RMS. Against
# the reference the kernels differ in the latent norm's rsqrt / reduction order (one BF16 step in
# ~0.7% of the latent elements) and in the split-K up projection's FP32 accumulation order; each of
# the two output roundings (up projection, sum with the shared shard) can then land one step away:
# measured on GB300 up to 2.12 ulp element / 0.4 ulp RMS, ~10% of the elements differing.
ELEMENT_ULP, RMS_ULP = 3.0, 0.5


def _inputs(spec: dict, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rows, world = spec["rows"], spec["world"]

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen, device="cuda") * scale).to(torch.bfloat16)

    routed = [r(rows, LATENT) for _ in range(world)]
    shared = [r(rows, HIDDEN, scale=0.25) for _ in range(world)]
    rms_weight = (1 + 0.1 * torch.randn(LATENT, generator=gen, device="cuda")).to(torch.bfloat16)
    up_weight = r(HIDDEN, LATENT, scale=LATENT**-0.5)
    return routed, shared, rms_weight, up_weight


def _unfinalized(rows: int, gen: torch.Generator) -> UnfinalizedRoutedOutput:
    """A rank's routed experts before the top-k sum: 1..8 of each token's 16 slots on this rank, their FC2 rows in a
    random permuted order (with spare rows), BF16 routing weights."""
    local = torch.randint(1, 9, (rows,), generator=gen, device="cuda")
    perm_rows = int(local.sum())
    y = torch.randn(perm_rows + 16, LATENT, generator=gen, device="cuda").to(torch.bfloat16)
    order = torch.randperm(perm_rows, generator=gen, device="cuda").to(torch.int32)
    slots = torch.full((rows, 16), -1, dtype=torch.int32, device="cuda")
    start = 0
    for t, n in enumerate(local.tolist()):
        owned = torch.randperm(16, generator=gen, device="cuda")[:n]
        slots[t, owned] = order[start : start + n]
        start += n
    weights = (torch.rand(rows, 16, generator=gen, device="cuda") * 0.25).to(torch.bfloat16)
    return UnfinalizedRoutedOutput(y, weights, slots)


def _partials(spec: dict, seed: int):
    """Every rank's routed and shared partials and the weights, the same on every rank; with ``fold`` the routed
    partials are the routed experts' unfinalized outputs."""
    if not spec.get("fold"):
        return _inputs(spec, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    routed = [_unfinalized(spec["rows"], gen) for _ in range(spec["world"])]
    _, shared, rms_weight, up_weight = _inputs(spec, seed + 1)
    return routed, shared, rms_weight, up_weight


def _tail(spec: dict, rank: int):
    """The group's process-wide tail of the cell, kept across cells."""
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=spec["world"], tp_size=spec["world"], rank=rank)
    tails = vars(sys.modules.setdefault("_k31_tails", types.ModuleType("_k31_tails"))).setdefault(
        "by_world", {}
    )
    if spec["world"] not in tails:
        tails[spec["world"]] = KimiK3LatentMoETail.get(
            mapping,
            hidden_size=HIDDEN,
            latent_size=LATENT,
            rms_eps=spec["eps"],
        )
    return tails[spec["world"]]


def _push_ahead(shared: torch.Tensor, tail, side: torch.cuda.Stream) -> None:
    """``k31_latent_moe_tail_push_shared`` on a side stream forked from and joined back into the caller's, as the
    target's shared-expert stream does."""
    fork = torch.cuda.Event()
    fork.record()
    side.wait_event(fork)
    with torch.cuda.stream(side):
        k31_latent_moe_tail_push_shared(shared, tail)
    torch.cuda.current_stream().wait_stream(side)


def _rank_cell(spec: dict, seed: int, mailbox: bool) -> dict:
    """One rank of a ``k31_latent_moe_tail`` cell or, with ``mailbox``, of a
    ``k31_latent_moe_tail_to_mailbox`` cell, its mailbox consumed by
    ``k31_latent_moe_tail_copy_out``."""
    import tensorrt_llm

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    tail = _tail(spec, rank)
    routed, shared, rms_weight, up_weight = _partials(spec, seed)
    pushed = bool(spec.get("shared_pushed"))
    entry = k31_latent_moe_tail_to_mailbox if mailbox else k31_latent_moe_tail
    with torch.inference_mode():
        if pushed:
            _push_ahead(shared[rank], tail, torch.cuda.Stream())
        result = entry(
            routed[rank], shared[rank], rms_weight, up_weight, tail, shared_pushed=pushed
        )
        out = k31_latent_moe_tail_copy_out(result, spec["rows"], tail) if mailbox else result
        ref = entry.reference(routed, shared, rms_weight, up_weight, spec["eps"])
        element, rms = bf16_deviation(out, ref)
    return dict(rank=rank, shape=tuple(out.shape), element_ulp=element, rms_ulp=rms)


CELLS = sample_cells(k31_latent_moe_tail.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, mpi_pool_executor) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, False, world=WORLD):
        assert r["shape"] == (spec["rows"], HIDDEN), r
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, r


MAILBOX_CELLS = sample_cells(k31_latent_moe_tail_to_mailbox.CELLS, size_key="rows")[:2]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
@pytest.mark.parametrize(
    "cell",
    MAILBOX_CELLS,
    ids=[c.why[:48] for c in MAILBOX_CELLS],
)
def test_mailbox_cells(cell, mpi_pool_executor) -> None:
    """The mailbox entry, its rows consumed by the copy-out entry (whose cells are these)."""
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    for r in run_per_rank(mpi_pool_executor, _rank_cell, spec, seed, True, world=WORLD):
        assert r["shape"] == (spec["rows"], HIDDEN), r
        assert r["element_ulp"] <= ELEMENT_ULP and r["rms_ulp"] <= RMS_ULP, r
