# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The op-level MoE all-to-all over 4 ranks on SM107 against the engine ``NVLinkOneSided``
strategy on the same inputs: the builder's layout, the dispatched rows, and the combined output
bit for bit; each entry's cells against its collective reference. The rows each rank receives
are the ones ``a2a_expert_owner`` (the dispatch reference's rule) sends it, also on the cells
whose experts split unevenly.

Each rank builds both workspaces (the engine's and the builder's) and runs the same dispatch,
the same expert stand-in and the same combine through each. The combine takes the expert output
in the workspace region the dispatch returns or, where a cell says so, as a caller tensor that
it stages. A rank with no tokens gets an empty output, whose check is its shape. The workers
receive only plain data (cell specs): every object that holds ``torch.ops`` is built inside the
worker.
"""

import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not _SM107, reason="certified on SM107 only"),
]

#: The expert stand-in's scale: each received row's BF16 output is a fixed function of its
#: payload, so both paths combine identical inputs.
_SEED = 41


def _mod():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import rubin_moe_a2a

    return rubin_moe_a2a


def _rows(spec, world):
    rows = spec["rows"]
    return list(rows) if isinstance(rows, (tuple, list)) else [rows] * world


def _expert_rows(x4, ids, hidden):
    """The expert stand-in: a BF16 row per received token from its payload bytes and ids (rows
    whose ids are all invalid give 0, as the zeroed MoE output)."""
    base = x4.view(torch.uint8)[:, : hidden // 2].float().repeat_interleave(2, dim=1)
    live = (ids >= 0).any(dim=1, keepdim=True)
    scale = (ids.clamp_min(0).float().sum(dim=1, keepdim=True) % 7 + 1) / 64
    return torch.where(live, (base - 127.5) * scale / 128, 0).to(torch.bfloat16)


def _band(ref, low):
    """FP8 wire: each partial is rounded to E4M3 (3 mantissa bits, half an ulp is 1/16 of its
    binade) before the sum, so 2 x 1/16 of the largest output covers every token's sum of at
    most 4 partials; BF16: 2 ulps of the largest output."""
    peak = max(1e-3, ref.float().abs().max().item())
    return peak * (2.0 / 16 if low else 2.0**-7)


def _within(out, ref, low):
    """``out`` has ``ref``'s shape and lies within ``_band(ref, low)`` of it everywhere. A rank
    with no tokens has ``[0, hidden]`` outputs: the shape is all there is to check, and a max
    over no elements raises."""
    if out.shape != ref.shape:
        return False
    return ref.numel() == 0 or (out.float() - ref.float()).abs().max().item() <= _band(ref, low)


def _run_rank(spec, why):
    import os

    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm.mapping import Mapping

    mod = _mod()
    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    try:
        hidden, top_k, num_experts = spec["hidden"], spec["top_k"], spec["num_experts"]
        rows = _rows(spec, world)
        runtime_max = max(rows)
        cap = 512
        mapping = Mapping(
            world_size=world,
            rank=rank,
            tp_size=world,
            enable_attention_dp=True,
            moe_ep_size=world,
            moe_tp_size=1,
            gpus_per_node=torch.cuda.device_count(),
        )
        force = None if spec["cft"] is None else bool(spec["cft"])
        if force is False:
            os.environ["TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT"] = "0"
        low = spec.get("low_precision", True)
        in_workspace = spec.get("in_workspace", True)
        engine = NVLinkOneSided(
            mapping,
            num_experts,
            top_k,
            cap,
            payload_in_workspace=in_workspace,
            hidden_size=hidden,
            dtype=torch.bfloat16,
            use_low_precision_combine=low,
        )
        os.environ.pop("TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT", None)
        # One workspace per process may hold the counted-write endpoints: the engine's took them.
        ws = mod.rubin_moe_a2a_workspace(
            mapping,
            num_experts=num_experts,
            top_k=top_k,
            max_num_tokens_per_rank=cap,
            hidden_size=hidden,
            dtype=torch.bfloat16,
            use_low_precision_combine=low,
            force_cft=False if engine.can_use_cft_counted_writes else force,
        )
        same_layout = torch.equal(ws.metainfo, engine.moe_a2a_metainfo) or (
            engine.can_use_cft_counted_writes and not ws.can_use_cft
        )

        g = torch.Generator(device="cuda").manual_seed(_SEED + rank)
        n = rows[rank]
        x4 = torch.randint(0, 256, (n, hidden // 2), generator=g, device="cuda", dtype=torch.uint8)
        sf = torch.randint(0, 256, (n, hidden // 16), generator=g, device="cuda", dtype=torch.uint8)
        scores = torch.rand(n, num_experts, generator=g, device="cuda")
        ids = torch.topk(scores, top_k, dim=-1).indices.to(torch.int32)
        # The first tokens take ``top_k`` consecutive experts that end or start at a window
        # boundary of the dispatch's split, so the owner rule alone decides which ranks they reach.
        bounds = [mod.a2a_expert_window(num_experts, world, q)[0] for q in range(1, world)]
        runs = [start for b in bounds for start in (b - top_k, b)]
        for t, start in enumerate(runs[:n]):
            ids[t] = torch.arange(start, start + top_k, device="cuda", dtype=torch.int32)
        wts = torch.rand(n, top_k, generator=g, device="cuda")

        # The expert output goes into the combine region of the workspace, or (``in_workspace``
        # False) stays a caller tensor, which the combine stages into that region.
        e = engine.dispatch(x4, sf, ids, wts, all_rank_num_tokens=rows)
        e_payload = _expert_rows(e[0], e[2], hidden)
        if in_workspace:
            e_payload = engine.get_combine_payload_tensor_in_workspace(
                runtime_max, hidden, torch.bfloat16
            ).copy_(e_payload)
        e_out = engine.combine(e_payload, all_rank_max_num_tokens=runtime_max)

        use = mod.a2a_use_cft(ws.can_use_cft, force, mod.CFT_MAX_BATCH, runtime_max)
        *o, offset, o_payload = mod.rubin_moe_a2a_dispatch(
            x4, sf, ids, wts, ws, runtime_max, use, -1, torch.bfloat16, hidden
        )
        o_rows = _expert_rows(o[0], o[2], hidden)
        o_payload = o_payload.copy_(o_rows) if in_workspace else o_rows
        o_out = mod.rubin_moe_a2a_combine(
            o_payload, n, ws, runtime_max, offset, in_workspace, low, use
        )
        torch.cuda.synchronize()

        # Received rows as a set per source block (the kernels pack them in atomic order).
        def block_set(t, ids_):
            key = []
            for s in range(world):
                blk = slice(s * runtime_max, (s + 1) * runtime_max)
                live = (ids_[blk] >= 0).any(1)
                key.append(sorted(map(bytes, t[blk][live].view(torch.uint8).cpu().numpy())))
            return key

        same_rows = all(block_set(a, e[2]) == block_set(b, o[2]) for a, b in zip(e[:3], o[:3]))
        # The rows the dispatch reference's owner rule (``a2a_expert_owner``) sends this rank,
        # from every rank's inputs: per source, the set of rows the kernels delivered.
        every_x4, every_ids = comm.allgather(x4.cpu()), comm.allgather(ids.cpu())
        sent = []
        for s in range(world):
            mine = (mod.a2a_expert_owner(every_ids[s], num_experts, world) == rank).any(dim=1)
            sent.append(sorted(map(bytes, every_x4[s][mine].numpy())))
        owner_rows = block_set(o[0], o[2]) == sent
        # One process holds one set of counted-write endpoints: where the engine's workspace took
        # them, ours runs the fence path, and the two combines are compared within the band.
        if engine.can_use_cft_counted_writes == ws.can_use_cft:
            same_out = torch.equal(e_out, o_out)
        else:
            same_out = _within(o_out, e_out, low)
        # Each rank that owns one of a token's experts returns the stand-in row once.
        owners = mod.a2a_expert_owner(ids.cpu(), num_experts, world)
        expect = torch.zeros(n, hidden, dtype=torch.float32)
        for t in range(n):
            row = _expert_rows(x4[t : t + 1], ids[t : t + 1], hidden)[0].float().cpu()
            expect[t] = len(set(owners[t].tolist())) * row
        near = _within(o_out.cpu(), expect, low)
        engine.destroy()
        result = (bool(same_layout), bool(same_rows), bool(owner_rows), bool(same_out), bool(near))
    except Exception:
        traceback.print_exc()
        raise
    return result


def _cells():
    mod = _mod()
    seen, out = set(), []
    for cell in (*mod.rubin_moe_a2a_dispatch.CELLS, *mod.rubin_moe_a2a_combine.CELLS):
        key = repr(sorted(cell.spec.items()))
        if key not in seen:
            seen.add(key)
            out.append(cell)
    return out


@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_cells_against_the_engine_strategy(mpi_pool_executor):
    """rubin_moe_a2a_dispatch.CELLS and rubin_moe_a2a_combine.CELLS over 4 ranks (the DEP12
    window cells split their experts unevenly): the builder's metainfo is the engine's, every
    rank receives the engine's rows and the rows the owner rule sends it, and the combine is the
    engine's bit for bit (within ``_band`` where the engine took the counted-write endpoints and
    ours runs the fence path) and its reference's within ``_band``."""
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    for cell in _cells():
        results = list(
            mpi_pool_executor.map(_run_rank, *zip(*[(dict(cell.spec), cell.why)] * world))
        )
        assert all(all(r) for r in results), (
            f"{cell.why}: (layout, rows, owner rows, bits, band) {results}"
        )
