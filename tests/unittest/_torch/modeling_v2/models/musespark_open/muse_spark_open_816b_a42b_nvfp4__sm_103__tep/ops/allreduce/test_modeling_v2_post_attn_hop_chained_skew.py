# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stale-only polling schedule under in-graph rank skew, over chained calls (8 MPI ranks, two GB300 nodes).

Every arm is a CUDA graph of ``LAYERS`` chained calls -- each layer's input depends on the
previous layer's output, so one wrong value propagates -- with a per-rank, per-layer GPU spin
captured in front of every call. Four skew patterns (none; one late rank per layer, in rank
order and mirrored; a seeded random delay per rank and layer) are captured as four graphs per
arm and replayed in turn with fresh inputs; eager calls with a delayed rank run between
replays. The stale-poll arm must equal the reference arm bit for bit on every rank:

* ``stale_poll``: ``PostAttnHopWorkspace(stale_poll=True)`` against the default fused hop (which the
  hop's own suite certifies against the built-in MNNVL all-reduce and cluster boundary),
  multicast and unicast, over row counts that mix the stale-only schedule (8 rows) with the
  default one below ``STALE_POLL_MIN_ROWS`` inside one graph; and with the stale-only
  schedule on every row count (``stale_poll_min_rows=1``).

Run it as the hop's suite is run under MPI::

    srun -N2 --segment=2 --ntasks-per-node=4 --mpi=pmix \\
        python3 -m pytest -q -p no:cacheprovider --noconftest \\
        -o addopts="-p test_common.magic_import_hooks" <this file>
"""

import importlib
import os

import pytest
import torch

try:
    from tensorrt_llm._utils import mpi_comm

    COMM = mpi_comm()
    WORLD, RANK = COMM.Get_size(), COMM.Get_rank()
except Exception:  # noqa: BLE001 - no MPI launch means no group
    COMM, WORLD, RANK = None, 1, 0

pytestmark = pytest.mark.skipif(
    WORLD != 8 or not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="needs 8 MPI ranks on two four-GPU GB300 nodes",
)

LAYERS = 16
#: Spin cycles of a late rank (about 10 us at 2 GHz).
SPIN = 20_000
PATTERNS = (0, 1, 2, 3)
REPLAYS = 24
_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)


def _agree(ok: bool) -> bool:
    return all(COMM.allgather(bool(ok)))


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.int16)


def _diff(a, b) -> int:
    return int((_bits(a) != _bits(b)).sum().item())


def _delays(pattern: int):
    """Spin cycles of this rank before each layer's call, per skew pattern."""
    if pattern == 0:
        return [0] * LAYERS
    if pattern == 1:
        return [SPIN if RANK == i % WORLD else 0 for i in range(LAYERS)]
    if pattern == 2:
        return [SPIN if RANK == (WORLD - 1 - i % WORLD) else 0 for i in range(LAYERS)]
    g = torch.Generator().manual_seed(9001)
    table = torch.randint(0, 6_000, (LAYERS, WORLD), generator=g)
    return [int(table[i, RANK]) for i in range(LAYERS)]


@pytest.fixture(scope="module")
def mapping():
    from tensorrt_llm.mapping import Mapping

    local = int(os.environ.get("SLURM_LOCALID", RANK % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    return Mapping(world_size=WORLD, tp_size=WORLD, rank=RANK, gpus_per_node=4)


def _replay_and_compare(graphs, outs, static, base, eager, pairs) -> int:
    """Replay every pattern's graphs with fresh inputs; eager delayed calls in between."""
    bad = 0
    for replay in range(REPLAYS):
        p = PATTERNS[replay % len(PATTERNS)]
        for dst, src in zip(static, base(100 + replay)):
            for d, s in zip(dst, src):
                d.copy_(s)
        graphs[("stale", p)].replay()
        graphs[("ref", p)].replay()
        if replay % 3 == 2:
            bad += eager(replay)
        torch.cuda.synchronize()
        for got, want in zip(outs[("stale", p)], outs[("ref", p)]):
            bad += sum(_diff(g, w) for g, w in pairs(got, want))
    return bad


def _capture(arms, chain, static):
    graphs, outs = {}, {}
    for p in PATTERNS:
        for name, ws in arms:
            chain(ws, static, _delays(p))
            torch.cuda.synchronize()
            COMM.Barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs[(name, p)] = chain(ws, static, _delays(p))
            graphs[(name, p)] = graph
    torch.cuda.synchronize()
    COMM.Barrier()
    return graphs, outs


# --- stale_poll: the post-attention hop -----------------------------------------------------------

HOP_HIDDEN, POST_EPS, PRE_EPS = 8192, 1e-8, 1e-5
HOP_ROWS = (8, 8, 8, 1, 8, 4, 8, 8, 6, 8, 8, 2, 8, 8, 8, 8)


def _hop_base(seed: int):
    out = []
    for i, rows in enumerate(HOP_ROWS):
        g = torch.Generator().manual_seed(seed * 1000 + i)
        channel = torch.exp(0.75 * torch.randn(HOP_HIDDEN, generator=g))
        partials = [
            (0.5 * torch.randn(rows, HOP_HIDDEN, generator=g) * channel).to(torch.bfloat16)
            for _ in range(WORLD)
        ]
        residual = 4.0 * torch.randn(rows, HOP_HIDDEN, generator=g)
        gate = 1.5 * torch.randn(HOP_HIDDEN, generator=g) / 0.3
        beta = torch.sigmoid(gate)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-gate) * (1.0 + beta), min=1e-3))
        weight = (0.2 * torch.randn(HOP_HIDDEN, generator=g)).to(torch.bfloat16)
        out.append(
            [t.cuda() for t in (partials[RANK], residual, alpha.float(), beta.float(), weight)]
        )
    return out


@pytest.fixture(scope="module")
def hop(mapping):
    mod = importlib.import_module(f"{_TARGET}.ops.allreduce.post_attn_hop")
    variants = [(POST_EPS, PRE_EPS, False)]
    ws = {
        "mc": mod.PostAttnHopWorkspace(mapping, multicast=True, variants=variants),
        "mc_stale": mod.PostAttnHopWorkspace(
            mapping, multicast=True, variants=variants, stale_poll=True
        ),
        "uc": mod.PostAttnHopWorkspace(mapping, multicast=False, variants=variants),
        "uc_stale": mod.PostAttnHopWorkspace(
            mapping, multicast=False, variants=variants, stale_poll=True
        ),
        "mc_stale_all": mod.PostAttnHopWorkspace(
            mapping, multicast=True, variants=variants, stale_poll=True, stale_poll_min_rows=1
        ),
    }
    COMM.Barrier()
    return mod.post_attn_hop, ws


@pytest.mark.parametrize(
    "stale_name, ref_name", [("mc_stale", "mc"), ("uc_stale", "uc"), ("mc_stale_all", "mc")]
)
@torch.inference_mode()
def test_stale_poll_chained_graphs_under_rank_skew(hop, stale_name, ref_name):
    op, ws = hop

    def chain(w, static, delays):
        outs = []
        h = ffn = None
        for i, rows in enumerate(HOP_ROWS):
            partial, residual, alpha, beta, weight = static[i]
            if ffn is not None:
                k = min(rows, ffn.shape[0])
                partial = partial.clone()
                partial[:k] = (partial[:k].float() + 0.25 * ffn[:k].float()).to(torch.bfloat16)
                residual = residual.clone()
                residual[:k] = residual[:k] + h[:k]
            if delays[i]:
                torch.cuda._sleep(delays[i])
            h, ffn = op(partial, residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, w)
            outs.append((h, ffn))
        return outs

    static = _hop_base(1)
    stale, ref = ws[stale_name], ws[ref_name]
    graphs, outs = _capture((("stale", stale), ("ref", ref)), chain, static)

    def eager(replay):
        partial, residual, alpha, beta, weight = _hop_base(500 + replay)[0]
        if RANK == replay % WORLD:
            torch.cuda._sleep(200_000)
        got = op(partial, residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, stale)
        want = op(partial, residual, alpha, beta, weight, POST_EPS, PRE_EPS, False, ref)
        torch.cuda.synchronize()
        return _diff(got[0], want[0]) + _diff(got[1], want[1])

    bad = _replay_and_compare(
        graphs, outs, static, _hop_base, eager, lambda got, want: list(zip(got, want))
    )
    assert _agree(bad == 0), f"rank {RANK}: {bad} values differ"
