# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the Kimi K3.1 EP exchanges against the engine strategies they replace, on one GB300 tray.

EP4 (four ranks, one tray) stands in for DEP16: the kernels' per-token work does not depend on the group size, and
the 896 experts split evenly at both. Every rank regenerates every rank's K3.1 payload (MXFP8 latent [T, 3584],
UE8M0 scales [T, 112], 16 distinct int32 expert ids of 896, BF16 routing weights) from a seed; a rank without
tokens is padded the way the engine scheduler pads it (zero rows routed alike) and every rank passes the padded
count. The MoE is emulated by a function of each received row and the rank, with values over a wide exponent range
so the combine's summation order shows.

* `K31EpAllToAll` vs the engine `NVLinkOneSided`, built with the strategy factory's arguments and the default
  fault-tolerance options, both under `MetaInitMode` as the model loader builds them, with the default counted-write
  (CFT) choice of the platform: same workspace layout, size, CFT choice and kernel timeout. Per cell, the combine
  output is bit-identical. The dispatch output is bit-identical per source block as a set of rows (activation,
  scales, ids, weights) and on the sanitized ids of the unfilled rows: the kernel places a block's rows in the
  order its CTAs claim slots (an atomic counter), which can differ from call to call, so positions are compared
  only after sorting; the rows also match the dispatch entry's reference. `create_ep_all_to_all` picks the
  one-sided exchange on this platform and shares the workspace across instances.
* `K31EpAllGather` vs the engine `AllGatherReduceScatter`: dispatch and combine outputs bit-identical.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD = 4
NUM_SLOTS, TOP_K, HIDDEN, SF = 896, 16, 3584, 112
#: The workspace's tokens per rank (model_config.max_num_tokens in the model): larger than every cell.
MAX_TOKENS = 4096
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or torch.cuda.device_count() < WORLD,
        reason="one GB300 tray: sm_100 / sm_103 and four GPUs",
    ),
]


def _inputs(rank: int, spec: dict, seed: int):
    """This rank's payload for a cell: its real tokens, then (with DP padding) zero rows up to the padded count."""
    real = spec.get("real_num_tokens", spec["all_rank_num_tokens"])[rank]
    rows = spec["all_rank_num_tokens"][rank]
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    scores = torch.rand((real, NUM_SLOTS), generator=g, device="cuda")
    ids = scores.topk(TOP_K, dim=1).indices.to(torch.int32)
    hid = torch.randint(0, 256, (real, HIDDEN), generator=g, device="cuda", dtype=torch.uint8)
    sf = torch.randint(0, 256, (real, SF), generator=g, device="cuda", dtype=torch.uint8)
    mag = torch.randint(-2, 3, (real, TOP_K), generator=g, device="cuda").float()
    sgn = torch.randint(0, 2, (real, TOP_K), generator=g, device="cuda").float() * 2 - 1
    wts = (torch.exp2(mag) * sgn).to(torch.bfloat16)
    pad = rows - real
    if pad:
        # A padding row: zero activation and scales, every row routed to the same 16 experts (the routing of
        # zero logits), one routing weight each.
        pad_ids = (torch.arange(TOP_K, device="cuda", dtype=torch.int32) * 53) % NUM_SLOTS
        ids = torch.cat([ids, pad_ids.expand(pad, TOP_K)])
        hid = torch.cat([hid, hid.new_zeros(pad, HIDDEN)])
        sf = torch.cat([sf, sf.new_zeros(pad, SF)])
        wts = torch.cat([wts, wts.new_full((pad, TOP_K), 1.0 / TOP_K)])
    return hid.view(torch.float8_e4m3fn), sf.contiguous(), ids.contiguous(), wts.contiguous()


def _moe(rank: int, hid: torch.Tensor, ids: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """Emulated MoE output of each received row on ``rank``: a function of the row alone."""
    per_rank = NUM_SLOTS // WORLD
    local = (ids >= 0) & (torch.div(ids, per_rank, rounding_mode="floor") == rank)
    w = (wts.float() * local).sum(1, keepdim=True)
    b = hid.view(torch.uint8).to(torch.int32)
    e = ((b * 7 + rank) % 40 - 20).float()
    return ((b - 128).float() * w * torch.exp2(e)).to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _row_bytes(x, sf, ids, w) -> torch.Tensor:
    return torch.cat([_bits(x), _bits(sf), _bits(ids), _bits(w)], dim=1)


def _sorted_rows(rows: torch.Tensor) -> torch.Tensor:
    """Rows in a canonical order (by an exact int64 key of their bytes; equal rows are interchangeable)."""
    g = torch.Generator(device="cuda").manual_seed(7)
    wts = torch.randint(0, 1 << 31, (rows.shape[1],), generator=g, device="cuda", dtype=torch.int64)
    key = (rows.to(torch.int64) * wts).sum(1)
    return rows[torch.sort(key, stable=True).indices]


def _a2a_timeout() -> int:
    # `torch.ops` is a module subclass cloudpickle cannot pickle with this module's functions; reach it by name.
    return getattr(torch, "ops").trtllm.moe_a2a_get_timeout()


#: Cells whose exchange is also traced for its GPU kernel sequence (decode and the first fence-only count).
TRACED_CELLS = (0, 2)


def _kernel_sequence(exchange) -> list:
    """Names of the GPU activities (kernels, copies, sets) ``exchange()`` issues, in start order."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        exchange()
        torch.cuda.synchronize()
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return [e.name for e in sorted(gpu, key=lambda e: e.time_range.start)]


def _traced_exchange(comm, rank: int, spec: dict, seed: int):
    counts = list(spec["all_rank_num_tokens"])
    x, xs, xi, xw = _inputs(rank, spec, seed)

    def exchange():
        rx, _, rxi, rxw = comm.dispatch(x, xs, xi, xw, counts, spec["use_dp_padding"])
        comm.combine(_moe(rank, rx, rxi, rxw), all_rank_max_num_tokens=max(counts))

    return exchange


def _mapping(rank: int):
    from tensorrt_llm.mapping import Mapping

    return Mapping(
        world_size=WORLD,
        rank=rank,
        gpus_per_node=WORLD,
        tp_size=WORLD,
        moe_ep_size=WORLD,
        moe_tp_size=1,
        enable_attention_dp=True,
    )


def _rank_one_sided() -> dict:
    """Engine `NVLinkOneSided` over every cell, then the family `K31EpAllToAll` over the same cells; compare."""
    from types import SimpleNamespace

    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        ep_dispatch_combine as fam,
    )
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm._torch.moe.fused_moe.wide_ep_ft import get_wide_ep_ft_options

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.zeros(1, device="cuda")
    mapping = _mapping(rank)
    cells = fam.MoeA2ADispatch.CELLS
    fails, info = [], []

    # The factory's arguments for K3.1 (no EPLB; fault-tolerance options at their defaults).
    health, watchdog_timeout_s, watchdog_poll_s = get_wide_ep_ft_options(
        SimpleNamespace(extra_attrs={}, use_cuda_graph=True, mapping=mapping)
    )
    with MetaInitMode():
        engine = NVLinkOneSided(
            mapping,
            NUM_SLOTS,
            TOP_K,
            MAX_TOKENS,
            False,
            hidden_size=HIDDEN,
            dtype=torch.bfloat16,
            num_experts=None,
            use_low_precision_combine=False,
            ep_group_health=health,
            alltoall_watchdog_timeout_s=watchdog_timeout_s,
            alltoall_watchdog_poll_interval_s=watchdog_poll_s,
        )
    engine_setup = dict(
        metainfo=engine.moe_a2a_metainfo.clone(),
        size=engine.workspace_size_per_rank,
        cft=engine.can_use_cft_counted_writes,
        timeout=_a2a_timeout(),
    )
    engine_out = []
    for i, cell in enumerate(cells):
        spec = cell.spec
        counts = list(spec["all_rank_num_tokens"])
        tmax = max(counts)
        x, xs, xi, xw = _inputs(rank, spec, 100 + i)
        rx, rxs, rxi, rxw = engine.dispatch(
            hidden_states=x,
            hidden_states_sf=xs,
            token_selected_slots=xi,
            token_final_scales=xw,
            all_rank_num_tokens=counts,
            use_dp_padding=spec["use_dp_padding"],
        )
        recv = [t.clone() for t in (rx, rxs, rxi, rxw)]
        out = engine.combine(_moe(rank, rx, rxi, rxw), all_rank_max_num_tokens=tmax)
        engine_out.append(
            dict(
                recv=recv,
                out=out.clone(),
                cft=(engine.use_cft_for_dispatch(tmax), engine.use_cft_for_combine(tmax)),
            )
        )
    engine_kernels = [
        _kernel_sequence(_traced_exchange(engine, rank, cells[i].spec, 300 + i))
        for i in TRACED_CELLS
    ]
    torch.cuda.synchronize()
    # Releases the engine's logical endpoints before the family binds its own.
    engine.destroy()

    with MetaInitMode():
        family = fam.K31EpAllToAll(
            mapping,
            num_slots=NUM_SLOTS,
            top_k=TOP_K,
            max_num_tokens_per_rank=MAX_TOKENS,
            hidden_size=HIDDEN,
            dtype=torch.bfloat16,
            use_low_precision_combine=False,
        )
        picked = fam.create_ep_all_to_all(
            mapping, NUM_SLOTS, TOP_K, MAX_TOKENS, HIDDEN, torch.bfloat16
        )
    if (
        not isinstance(picked, fam.K31EpAllToAll)
        or picked.workspace.data_ptr() != family.workspace.data_ptr()
    ):
        fails.append(("factory / workspace sharing", type(picked).__name__))
    # The setup entries' contracts on the fresh workspace (before any exchange): the initialized bytes, and the
    # combine-payload view at each cell's row count.
    init = fam.moe_a2a_initialize
    for cell in init.CELLS:
        assert cell.spec["world"] == WORLD and cell.spec["max_tokens"] == MAX_TOKENS
        args = (family.workspace, family.metainfo, rank, WORLD)
        init(*args)
        torch.cuda.synchronize()
        size = int(family.metainfo[fam._WORKSPACE_SIZE_INDEX])
        if not torch.equal(family.workspace[rank, :size], init.reference(*args)):
            fails.append(("initialize", cell.why))
    view = fam.moe_a2a_get_combine_payload_tensor
    for cell in view.CELLS:
        args = (
            family.workspace,
            family.metainfo,
            rank,
            WORLD,
            cell.spec["runtime_max"],
            torch.bfloat16,
            HIDDEN,
        )
        got, ref = view(*args), view.reference(*args)
        if got.shape != ref.shape or got.data_ptr() != ref.data_ptr() or got.dtype != ref.dtype:
            fails.append(("combine payload view", cell.why))
    if not torch.equal(family.metainfo, engine_setup["metainfo"]):
        fails.append(("metainfo", family.metainfo.tolist(), engine_setup["metainfo"].tolist()))
    if family.workspace_size_per_rank != engine_setup["size"]:
        fails.append(("workspace size", family.workspace_size_per_rank, engine_setup["size"]))
    if family.can_use_cft_counted_writes != engine_setup["cft"]:
        fails.append(("cft", family.can_use_cft_counted_writes, engine_setup["cft"]))
    if _a2a_timeout() != engine_setup["timeout"]:
        fails.append(("timeout", _a2a_timeout(), engine_setup["timeout"]))

    reference = fam.MoeA2ADispatch().reference
    for i, cell in enumerate(cells):
        spec = cell.spec
        counts = list(spec["all_rank_num_tokens"])
        tmax = max(counts)
        x, xs, xi, xw = _inputs(rank, spec, 100 + i)
        rx, rxs, rxi, rxw = family.dispatch(x, xs, xi, xw, counts, spec["use_dp_padding"])
        ex, exs, exi, exw = engine_out[i]["recv"]
        expected = reference(
            [_inputs(s, spec, 100 + i) for s in range(WORLD)], rank, WORLD, NUM_SLOTS
        )
        positional = all(
            torch.equal(_bits(a), _bits(b))
            for a, b in zip((rx, rxs, rxi, rxw), (ex, exs, exi, exw))
        )
        for s in range(WORLD):
            rows = slice(s * tmax, (s + 1) * tmax)
            filled = rxi[rows, 0] != -1
            count = int(filled.sum())
            if not torch.equal(filled, exi[rows, 0] != -1) or count != expected[s][0].shape[0]:
                fails.append(
                    ("dispatch count", cell.why, s, count, int((exi[rows, 0] != -1).sum()))
                )
                continue
            if not bool(filled[:count].all()) or not bool((rxi[rows][count:] == -1).all()):
                fails.append(("dispatch unfilled ids", cell.why, s))
            if not torch.equal(rxi[rows][count:], exi[rows][count:]):
                fails.append(("dispatch unfilled ids vs engine", cell.why, s))
            head = slice(s * tmax, s * tmax + count)
            fam_rows = _sorted_rows(_row_bytes(rx[head], rxs[head], rxi[head], rxw[head]))
            eng_rows = _sorted_rows(_row_bytes(ex[head], exs[head], exi[head], exw[head]))
            if not torch.equal(fam_rows, eng_rows):
                diff = (fam_rows != eng_rows).nonzero()
                fails.append(("dispatch rows vs engine", cell.why, s, diff[:4].tolist()))
            if not torch.equal(fam_rows, _sorted_rows(_row_bytes(*expected[s]))):
                fails.append(("dispatch rows vs reference", cell.why, s))
        out = family.combine(_moe(rank, rx, rxi, rxw), tmax)
        torch.cuda.synchronize()
        eout = engine_out[i]["out"]
        if out.shape != eout.shape or not torch.equal(_bits(out), _bits(eout)):
            d = (out.float() - eout.float()).abs()
            where = (d == d.max()).nonzero()[0].tolist() if d.numel() else []
            fails.append(
                ("combine vs engine", cell.why, float(d.max()) if d.numel() else None, where)
            )
        info.append(
            dict(cell=cell.why, cft=engine_out[i]["cft"], positional_dispatch_equal=positional)
        )
    family_kernels = [
        _kernel_sequence(_traced_exchange(family, rank, cells[i].spec, 300 + i))
        for i in TRACED_CELLS
    ]
    if family_kernels != engine_kernels:
        fails.append(("kernel sequence", engine_kernels, family_kernels))
    fam.K31EpAllToAll.release_workspaces()
    return dict(
        rank=rank,
        fails=fails,
        info=info,
        setup={k: v for k, v in engine_setup.items() if k != "metainfo"},
        kernels={cells[i].why: seq for i, seq in zip(TRACED_CELLS, engine_kernels)},
    )


def _rank_all_gather() -> dict:
    """Engine `AllGatherReduceScatter` vs the family `K31EpAllGather` over every cell."""
    import tensorrt_llm
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
        ep_dispatch_combine as fam,
    )
    from tensorrt_llm._torch.moe.fused_moe.communication.allgather_reducescatter import (
        AllGatherReduceScatter,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = _mapping(rank)
    engine = AllGatherReduceScatter(mapping)
    family = fam.K31EpAllGather(mapping, NUM_SLOTS, TOP_K, MAX_TOKENS, HIDDEN, torch.bfloat16)
    fails = []
    for i, cell in enumerate(fam.AllgatherList.CELLS):
        spec = cell.spec
        counts = list(spec["all_rank_num_tokens"])
        pad = spec["use_dp_padding"]
        x, xs, xi, xw = _inputs(rank, spec, 200 + i)
        results = []
        for comm in (engine, family):
            if comm is engine:
                recv = comm.dispatch(
                    hidden_states=x,
                    hidden_states_sf=xs,
                    token_selected_slots=xi,
                    token_final_scales=xw,
                    all_rank_num_tokens=counts,
                    use_dp_padding=pad,
                )
            else:
                recv = comm.dispatch(x, xs, xi, xw, counts, pad)
            y = _moe(rank, recv[0], recv[2], recv[3])
            if comm is engine:
                out = comm.combine(y, all_rank_max_num_tokens=max(counts))
            else:
                out = comm.combine(y, max(counts))
            results.append([t.clone() for t in recv] + [out.clone()])
        torch.cuda.synchronize()
        for name, a, b in zip(("x", "x_sf", "ids", "weights", "combine"), *results):
            if a.shape != b.shape or not torch.equal(_bits(a), _bits(b)):
                fails.append((name, cell.why))
    return dict(rank=rank, fails=fails)


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_one_sided_all_to_all_matches_engine(mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(lambda _: _rank_one_sided(), range(WORLD)))
    for r in results:
        print(f"rank {r['rank']}: setup {r['setup']}")
        for item in r["info"]:
            print(f"rank {r['rank']}: {item}")
        for why, seq in r["kernels"].items():
            print(f"rank {r['rank']}: kernels ({why}): {seq}")
    for r in results:
        assert not r["fails"], r["fails"]


@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_all_gather_matches_engine(mpi_pool_executor) -> None:
    results = list(mpi_pool_executor.map(lambda _: _rank_all_gather(), range(WORLD)))
    for r in results:
        assert not r["fails"], r["fails"]
