# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's helix window attention across ranks: plan, window, W_UV, combine.

Run under MPI with 2, 4 or 8 ranks on SM107 (``mpirun -n 4 pytest ...``); any other world size
skips. Every rank builds the same global draft history (one seed), keeps the pages helix gives
it (local page ``c`` = global page ``c * cp + rank``), plans and attends its own keys with
stats (``rubin_dflash2_window_triton``), projects through W_UV first and combines through the
real exchange (``rubin_dflash2_helix_combine``): NCCL, and the MNNVL FIFO where it exists.

- The combined result equals the dense window, projected, for this rank's heads.
- Sanitized rows with ``zero_kv_mask=None`` equal, bit for bit, the masked path over the same
  rows left unsanitized.
- With the FIFO, the exchange's mapping is in its cache after the warmup pass, so a captured
  forward never allocates MNNVL memory or synchronizes (a new mapping would).
- One layer captured in a CUDA graph on one input set and replayed on another equals eager.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_helix_combine import (  # noqa: E501
    rubin_dflash2_helix_combine,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    plan_buffers,
    rubin_dflash2_window_plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_triton import (  # noqa: E501
    rubin_dflash2_window_triton,
    window_workspace,
)
from tensorrt_llm._utils import get_sm_version, mpi_rank, mpi_world_size
from tensorrt_llm.mapping import Mapping


def _load_triton_test():
    path = Path(__file__).parents[1] / "ops" / "spec" / "test_rubin_dflash2_window_triton.py"
    spec = importlib.util.spec_from_file_location("_dflash2_window_triton_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


_T = _load_triton_test()

_WORLD = mpi_world_size()
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107 or _WORLD not in (2, 4, 8),
    reason="2, 4 or 8 MPI ranks on SM107",
)

#: The helix8 drafter shape on one generation server: 16 requests at draft 5, 96 heads, FP8
#: history.
_SPEC = dict(rubin_dflash2_window_triton.CELLS[1].spec, num_splits=1, stats=True)
_LENGTHS = [
    (0, 1, 5, 60, 63, 64, 66, 130, 1017, 1022, 1025, 1031, 4097, 9000, 65536, 190000),
    (190000, 9000, 4097, 1031, 1025, 1022, 1017, 130, 66, 64, 63, 60, 5, 1, 0, 7),
]
_V_DIM = 128


#: The combine's transports, both always listed: NCCL, and the MNNVL FIFO (``fifo_v2``), which a
#: host without MNNVL skips with the reason instead of leaving the case out.
_TRANSPORTS = dict(argvalues=[True, False], ids=["nccl", "fifo_v2"])


def _need_transport(nccl: bool) -> None:
    """Skip the FIFO case on a host without MNNVL. Only the capability probe decides: on an
    aarch64 host the MNNVL module must import, and a FIFO that fails to build fails the case."""
    if nccl:
        return
    if platform.machine().lower() != "aarch64":
        pytest.skip("the MNNVL FIFO runs on aarch64 hosts with MNNVL")
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    if not MnnvlMemory.supports_mnnvl():
        pytest.skip("this host has no MNNVL (not every NVLink is up)")


def _mapping(nccl: bool) -> Mapping:
    return Mapping(
        world_size=_WORLD,
        rank=mpi_rank(),
        tp_size=1,
        cp_size=_WORLD,
        cp_config={"cp_type": "HELIX", "use_nccl_for_alltoall": nccl, "fifo_version": 2},
        gpus_per_node=torch.cuda.device_count(),
    )


def _layer(spec, inp, w_uv, rank, mapping, buffers, workspace, stats, zero_kv_mask=None):
    plan = rubin_dflash2_window_plan(
        inp.num_ctx(),
        inp.local_table(rank),
        None,
        spec["block_size"],
        spec["window"],
        spec["page_size"],
        spec["cp_size"],
        rank,
        spec["num_splits"],
        buffers,
    )
    out = rubin_dflash2_window_triton(
        inp.query,
        inp.pool,
        inp.block_kv,
        plan,
        spec["scale"],
        spec["latent"],
        1.0,
        workspace,
        stats,
    )
    partial_v = torch.einsum("rhc,hvc->rhv", out, w_uv).contiguous()
    return partial_v, rubin_dflash2_helix_combine(
        partial_v, stats, mapping, spec["heads"], _V_DIM, zero_kv_mask
    )


def _expected(inp, w_uv, rank, cp):
    dense = _T.dense_window(inp)  # [R, H, latent] float64
    proj = torch.einsum("rhc,hvc->rhv", dense, w_uv.double())
    width = inp.spec["heads"] // cp
    return proj[:, rank * width : (rank + 1) * width].reshape(proj.shape[0], -1)


@torch.inference_mode()
@pytest.mark.parametrize("nccl", **_TRANSPORTS)
def test_window_and_combine_equal_the_dense_window(nccl):
    _need_transport(nccl)
    rank = mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    spec = dict(_SPEC, cp_size=_WORLD, cp_rank=rank)
    mapping = _mapping(nccl)
    inp = _T.build(spec, 11, "cuda", lengths=_LENGTHS[0])
    g = torch.Generator(device="cuda").manual_seed(13)
    w_uv = (
        0.05 * torch.randn(spec["heads"], _V_DIM, spec["latent"], generator=g, device="cuda")
    ).bfloat16()
    batch, blk, heads = len(_LENGTHS[0]), spec["block_size"], spec["heads"]
    buffers = plan_buffers(batch, blk, spec["window"], spec["page_size"], _WORLD, 1, "cuda")
    workspace = window_workspace(batch, blk, heads, spec["latent"], 1, "cuda")
    stats = torch.empty(batch * blk, heads, 2, device="cuda")

    partial_v, got = _layer(spec, inp, w_uv, rank, mapping, buffers, workspace, stats)
    torch.cuda.synchronize()
    assert torch.isfinite(got).all()
    torch.testing.assert_close(
        got.double(), _expected(inp, w_uv, rank, _WORLD), atol=3e-2, rtol=3e-2
    )
    if not nccl:
        from tensorrt_llm._torch.distributed.ops import HelixAllToAllNative

        assert mapping in HelixAllToAllNative._cache, "the exchange would allocate inside a capture"

    # The same rows unsanitized, with the mask: the exchange sanitizes them, bit for bit.
    empty = stats[..., 1] == 0  # [R, H]: a row is empty for every head or for none
    row_empty = empty.all(dim=1)
    assert torch.equal(empty, row_empty[:, None].expand_as(empty))
    poisoned_v = partial_v.masked_fill(row_empty[:, None, None], float("nan"))
    poisoned_stats = stats.clone()
    poisoned_stats[row_empty] = float("nan")
    masked = rubin_dflash2_helix_combine(
        poisoned_v, poisoned_stats, mapping, heads, _V_DIM, row_empty
    )
    assert torch.equal(masked, got)


@pytest.mark.parametrize("nccl", **_TRANSPORTS)
def test_capture_on_one_input_set_replay_on_another(nccl):
    _need_transport(nccl)
    rank = mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    spec = dict(_SPEC, cp_size=_WORLD, cp_rank=rank)
    mapping = _mapping(nccl)
    sets = [_T.build(spec, 21 + i, "cuda", lengths=lengths) for i, lengths in enumerate(_LENGTHS)]
    g = torch.Generator(device="cuda").manual_seed(13)
    w_uv = (
        0.05 * torch.randn(spec["heads"], _V_DIM, spec["latent"], generator=g, device="cuda")
    ).bfloat16()
    batch, blk, heads = len(_LENGTHS[0]), spec["block_size"], spec["heads"]
    cols = max(s.local_table(rank).shape[1] for s in sets)
    pages = max(s.pool.shape[0] for s in sets)
    dim = spec["latent"] + spec["rope"]
    live = _T.Inputs(
        spec=spec,
        lengths=list(_LENGTHS[0]),
        table=torch.zeros(batch, cols * _WORLD, dtype=torch.int64, device="cuda"),
        pool=torch.zeros(pages, spec["page_size"], dim, dtype=torch.float8_e4m3fn, device="cuda"),
        query=torch.zeros(batch * blk, heads, dim, dtype=torch.bfloat16, device="cuda"),
        block_kv=torch.zeros(batch * blk, dim, dtype=torch.float8_e4m3fn, device="cuda"),
    )
    num_ctx = torch.zeros(batch, dtype=torch.int64, device="cuda")
    table = torch.zeros(batch, cols, dtype=torch.int32, device="cuda")
    live.num_ctx = lambda: num_ctx
    live.local_table = lambda r: table
    buffers = plan_buffers(batch, blk, spec["window"], spec["page_size"], _WORLD, 1, "cuda")
    workspace = window_workspace(batch, blk, heads, spec["latent"], 1, "cuda")
    stats = torch.empty(batch * blk, heads, 2, device="cuda")

    def load(s):
        num_ctx.copy_(s.num_ctx())
        local = s.local_table(rank)
        table.zero_()
        table[:, : local.shape[1]] = local
        live.pool.view(torch.uint8).zero_()
        live.pool[: s.pool.shape[0]] = s.pool
        live.query.copy_(s.query)
        live.block_kv.copy_(s.block_kv)

    def step():
        return _layer(spec, live, w_uv, rank, mapping, buffers, workspace, stats)[1]

    load(sets[0])
    step()  # warmup: every kernel compiled, the exchange's workspace allocated
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = step()
    load(sets[1])
    graph.replay()
    replayed = graph_out.clone()
    eager = step()
    torch.cuda.synchronize()
    assert torch.isfinite(replayed).all()
    assert torch.equal(replayed, eager)
    torch.testing.assert_close(
        replayed.double(), _expected(sets[1], w_uv, rank, _WORLD), atol=3e-2, rtol=3e-2
    )
