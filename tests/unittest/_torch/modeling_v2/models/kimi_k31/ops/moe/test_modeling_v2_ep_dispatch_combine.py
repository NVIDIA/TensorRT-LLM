# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: the Kimi K3.1 EP exchanges against the engine strategies they replace, on one GB300 tray.

EP4 (four ranks, one tray) stands in for DEP16: the kernels' per-token work does not depend on the group size, and
the 896 experts split evenly at both. Every rank regenerates every rank's K3.1 payload (MXFP8 latent [T, 3584],
UE8M0 scales [T, 112], 16 distinct int32 expert ids of 896, BF16 routing weights) from a seed; a rank without
tokens is padded the way the engine scheduler pads it (zero rows routed alike) and every rank passes the padded
count. The MoE is emulated by a function of each received row and the rank, with values over a wide exponent range
so the combine's summation order shows.

* `K31EpAllToAll` vs the engine `NVLinkOneSided`, built with the strategy factory's arguments, the default
  fault-tolerance options and the K3.1 payload's sentinel format (MXFP8), both under `MetaInitMode` as the model
  loader builds them: same workspace layout (sentinel regions included), size, sentinel token limit and kernel
  timeout, and per cell the same transport (sentinel or fence) and a bit-identical combine output (the emulated
  MoE output is a function of each received row, so a dispatch that delivers other rows shows there).
  `create_ep_all_to_all` picks the one-sided exchange on this platform and shares the workspace across instances.
* `K31EpAllGather` vs the engine `AllGatherReduceScatter`: dispatch and combine outputs bit-identical.

MPI-spawning: under srun the pool hangs, so run it under ``mpirun -n 1`` without the Slurm/PMIx environment.
"""

import pytest
import torch

__extra_import_path__ = ["../.."]
from _k31_utils import (  # noqa: E402 — needs the path declared above
    dep_mapping,
    on_ranks,
    pickle_by_value,
    run_per_rank,
)

pickle_by_value(__name__)

WORLD = 4
NUM_SLOTS, TOP_K, HIDDEN, SF = 896, 16, 3584, 112
#: The workspace's tokens per rank (model_config.max_num_tokens in the model): larger than every cell.
MAX_TOKENS = 4096
pytestmark = on_ranks(WORLD, 100, 103)


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


def _a2a_timeout() -> int:
    # `torch.ops` is a module subclass cloudpickle cannot pickle with this module's functions; reach it by name.
    return getattr(torch, "ops").trtllm.moe_a2a_get_timeout()


def _rank_one_sided() -> dict:
    """Engine `NVLinkOneSided` over every cell, then the family `K31EpAllToAll` over the same cells; compare."""
    from types import SimpleNamespace

    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe import ep_dispatch_combine as fam
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm._torch.moe.fused_moe.wide_ep_ft import get_wide_ep_ft_options

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    torch.zeros(1, device="cuda")
    mapping = dep_mapping(WORLD, rank, gpus_per_node=WORLD)
    cells = fam.MoeA2ADispatch.CELLS
    fails = []

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
            sentinel_payload_format=fam.SENTINEL_PAYLOAD_FORMAT,
        )
    engine_setup = dict(
        metainfo=engine.moe_a2a_metainfo.clone(),
        size=engine.workspace_size_per_rank,
        sentinel_max_tokens=engine.sentinel_max_tokens,
        timeout=_a2a_timeout(),
    )
    engine_out = []
    for i, cell in enumerate(cells):
        spec = cell.spec
        counts = list(spec["all_rank_num_tokens"])
        tmax = max(counts)
        x, xs, xi, xw = _inputs(rank, spec, 100 + i)
        sentinel = engine.use_sentinel_for_dispatch(tmax, x, xs, xw, None)
        rx, _, rxi, rxw = engine.dispatch(
            hidden_states=x,
            hidden_states_sf=xs,
            token_selected_slots=xi,
            token_final_scales=xw,
            all_rank_num_tokens=counts,
            use_dp_padding=spec["use_dp_padding"],
        )
        out = engine.combine(_moe(rank, rx, rxi, rxw), all_rank_max_num_tokens=tmax)
        engine_out.append((sentinel, out.clone()))
    torch.cuda.synchronize()
    # Releases the engine's workspace before the family allocates its own.
    engine.destroy()

    cache = {}
    with MetaInitMode():
        family = fam.K31EpAllToAll(
            mapping,
            num_slots=NUM_SLOTS,
            top_k=TOP_K,
            max_num_tokens_per_rank=MAX_TOKENS,
            hidden_size=HIDDEN,
            dtype=torch.bfloat16,
            use_low_precision_combine=False,
            workspace_cache=cache,
        )
        picked = fam.create_ep_all_to_all(
            mapping, NUM_SLOTS, TOP_K, MAX_TOKENS, HIDDEN, torch.bfloat16, workspace_cache=cache
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
        # The reference covers the layout before the sentinel regions (those hold the transport's armed words).
        size = int(family.metainfo[fam._WORKSPACE_SIZE_INDEX])
        if family.sentinel_max_tokens:
            size = int(family.metainfo[fam._layout_index("SENTINEL_STATE_OFFSET")])
        if not torch.equal(family.workspace[rank, :size], init.reference(*args)[:size]):
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
    if family.sentinel_max_tokens != engine_setup["sentinel_max_tokens"]:
        fails.append(
            ("sentinel limit", family.sentinel_max_tokens, engine_setup["sentinel_max_tokens"])
        )
    if _a2a_timeout() != engine_setup["timeout"]:
        fails.append(("timeout", _a2a_timeout(), engine_setup["timeout"]))

    for i, cell in enumerate(cells):
        spec = cell.spec
        counts = list(spec["all_rank_num_tokens"])
        tmax = max(counts)
        x, xs, xi, xw = _inputs(rank, spec, 100 + i)
        sentinel, eout = engine_out[i]
        if family._use_sentinel(tmax, x, xs, xw) != sentinel:
            fails.append(("transport", cell.why, sentinel))
        rx, _, rxi, rxw = family.dispatch(x, xs, xi, xw, counts, spec["use_dp_padding"])
        out = family.combine(_moe(rank, rx, rxi, rxw), tmax)
        torch.cuda.synchronize()
        if out.shape != eout.shape or not torch.equal(_bits(out), _bits(eout)):
            fails.append(("combine vs engine", cell.why))
    cache.clear()
    return dict(rank=rank, fails=fails)


def _rank_all_gather() -> dict:
    """Engine `AllGatherReduceScatter` vs the family `K31EpAllGather` over every cell."""
    import tensorrt_llm
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe import ep_dispatch_combine as fam
    from tensorrt_llm._torch.moe.fused_moe.communication.allgather_reducescatter import (
        AllGatherReduceScatter,
    )

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = dep_mapping(WORLD, rank, gpus_per_node=WORLD)
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


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_one_sided_all_to_all_matches_engine(mpi_pool_executor) -> None:
    results = run_per_rank(mpi_pool_executor, _rank_one_sided, world=WORLD)
    for r in results:
        assert not r["fails"], r["fails"]


@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [WORLD], indirect=True)
def test_all_gather_matches_engine(mpi_pool_executor) -> None:
    results = run_per_rank(mpi_pool_executor, _rank_all_gather, world=WORLD)
    for r in results:
        assert not r["fails"], r["fails"]
