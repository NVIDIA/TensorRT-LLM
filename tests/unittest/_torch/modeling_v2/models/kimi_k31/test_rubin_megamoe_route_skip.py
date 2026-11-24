# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's 32-row staging guard of the MegaMoE bucket path (CPU, from source).

A DEP16 generation chunk goes into MegaMoE's bucket: the top-16 (``rubin_route_topk_out``) and the
NVFP4 quantization (``rubin_megamoe_quantize``) write this rank's rows into it, and
``rubin_megamoe_in_bucket`` launches on it as it is. A rank without rows in the chunk, with more
rows than the routing kernel serves, or with a row count in ``_MOE_ROUTE_SKIP_M`` stages its rows
into the same bucket instead (``_moe_route``'s top-16, ``_moe_quantize``, then ``rubin_megamoe``).
The guard reads this rank's rows of the chunk only, whatever bucket the chunk launches, and every
rank launches the chunk's bucket. (32 rows: measured with 32 rows on every rank, a full 32-row
bucket, the bucket path's routing and quantization take longer than staging.)

The product's ``_moe_megamoe``, ``_moe_megamoe_chunk``, the bucket path's helpers
(``_moe_bucket_*``) and ``_moe_split`` run from its source; the entries they call and the staging
path's routing and quantization are recording stand-ins.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_DEP16 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__dep16" / "modeling.py"
_ROUTE = _FAMILY / "ops/moe/rubin_route_topk_out.py"
_FUNCS = (
    "_moe_megamoe",
    "_moe_megamoe_chunk",
    "_moe_bucket_rows",
    "_moe_bucket_route",
    "_moe_bucket_quantize",
    "_moe_bucket_launch",
    "_moe_split",
    "_moe_chunk_plane",
    "_moe_routed_norm",
)
# dep16's EP ranks; stand-in latent width and router width (the paths do not depend on them).
_RANKS, _LATENT, _EXPERTS, _TOP_K = 16, 64, 32, 16
_BUCKET_PATH = ["megamoe_bucket", "route_topk_out", "megamoe_quantize", "megamoe_in_bucket"]
_STAGING_PATH = ["route", "quantize", "megamoe"]


def _assigned(path: Path, name: str) -> ast.expr:
    """The value expression of the module-level assignment of ``name`` in ``path``."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [
            name
        ]:
            return node.value
    raise KeyError(name)


def _route_skip() -> frozenset:
    expr = ast.fix_missing_locations(ast.Expression(_assigned(_DEP16, "_MOE_ROUTE_SKIP_M")))
    return eval(compile(expr, _DEP16.name, "eval"), {"frozenset": frozenset})


def _route_max() -> int:
    return ast.literal_eval(_assigned(_ROUTE, "MAX_M"))


def _megamoe(log: list) -> dict:
    """The product's MegaMoE functions in a namespace of recording stand-ins."""

    def megamoe_bucket(ws, chunk_max, rows):
        log.append(("megamoe_bucket", chunk_max, rows))
        return SimpleNamespace(
            topk_idx=torch.zeros(chunk_max, _TOP_K, dtype=torch.int32),
            topk_weights=torch.zeros(chunk_max, _TOP_K),
            activation=torch.zeros(chunk_max, _LATENT // 2, dtype=torch.uint8),
            activation_sf=torch.zeros(chunk_max, _LATENT // 16, dtype=torch.uint8),
        )

    def rubin_route_topk_out(logits, bias, scale, topk_idx, topk_weights, route_approx):
        log.append(("route_topk_out", logits.shape[0], topk_idx.shape[0]))

    def rubin_megamoe_quantize(
        latent_in, scale, activation, activation_sf, topk_idx, weights, pdl=False
    ):
        assert not pdl, "the chunk code launches the quantization in stream order"
        log.append(("megamoe_quantize", latent_in.shape[0], activation.shape[0]))

    def rubin_megamoe_in_bucket(b, *weights_and_ws_rows):
        rows = weights_and_ws_rows[-1]
        log.append(("megamoe_in_bucket", b.topk_idx.shape[0], rows))
        return torch.full((rows, _LATENT), float(rows))

    def _moe_route(layer, logits):
        log.append(("route", logits.shape[0]))
        rows = logits.shape[0]
        return torch.zeros(rows, _TOP_K, dtype=torch.int32), torch.zeros(rows, _TOP_K)

    def _moe_quantize(layer, latent_in):
        log.append(("quantize", latent_in.shape[0]))
        rows = latent_in.shape[0]
        return (
            torch.zeros(rows, _LATENT // 2, dtype=torch.uint8),
            torch.zeros(rows, _LATENT // 16, dtype=torch.uint8),
        )

    def rubin_megamoe(x4, sf, ids, weights, *weights_ws_chunk_rows):
        chunk_max, rows = weights_ws_chunk_rows[-2:]
        assert x4.shape[0] == sf.shape[0] == ids.shape[0] == weights.shape[0] == rows
        log.append(("megamoe", chunk_max, rows))
        return torch.full((rows, _LATENT), float(rows))

    ns = dict(
        torch=torch,
        megamoe_bucket=megamoe_bucket,
        rubin_route_topk_out=rubin_route_topk_out,
        rubin_megamoe_quantize=rubin_megamoe_quantize,
        rubin_megamoe_in_bucket=rubin_megamoe_in_bucket,
        _moe_route=_moe_route,
        _moe_quantize=_moe_quantize,
        rubin_megamoe=rubin_megamoe,
        _MOE_ROUTE_MAX_M=_route_max(),
        _MOE_ROUTE_SKIP_M=_route_skip(),
        _MOE_SCALE=1.0,
        _MOE_TOP_K=_TOP_K,
        _MOE_SF_VEC=16,
        _MOE_EPS=1e-5,
        flashinfer_rmsnorm=lambda x, weight, eps: x,
    )
    nodes = [n for n in ast.parse(_DEP16.read_text()).body if getattr(n, "name", None) in _FUNCS]
    assert sorted(n.name for n in nodes) == sorted(_FUNCS)
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, [])), _DEP16.name, "exec"), ns)
    return ns


def _run(counts: list, rank: int, chunk_tokens: int) -> list:
    """``_moe_megamoe`` on ``rank`` of a step whose ranks hold ``counts`` rows; the calls made."""
    log = []
    ns = _megamoe(log)
    rows = counts[rank]
    layer = SimpleNamespace(
        router_bias=None,
        fc31_in_scale=None,
        fc1_w=None,
        fc1_s=None,
        fc2_w=None,
        fc2_s=None,
        fc31_alpha=None,
        fc2_alpha=None,
        fc1_norm_const=None,
        latent_norm=None,
    )
    rt = SimpleNamespace(
        megamoe=SimpleNamespace(ep_rank=rank, chunk_tokens=chunk_tokens), moe_back_planes=None
    )
    st = SimpleNamespace(all_rank_num_tokens=counts)
    out = ns["_moe_megamoe"](layer, torch.zeros(rows, _EXPERTS), torch.zeros(rows, _LATENT), st, rt)
    assert tuple(out.shape) == (rows, _LATENT)
    return log


def _ops(log: list) -> list:
    return [entry[0] for entry in log]


@pytest.mark.parametrize("rows", [1, 8, 16, 31, 32, 33, 64, 96])
def test_a_32_row_chunk_is_staged(rows):
    """Every rank holds ``rows`` rows, one chunk: 32 rows stage, the others go into the bucket."""
    log = _run([rows] * _RANKS, 0, 96)
    if rows == 32:
        assert _ops(log) == _STAGING_PATH, log
        assert log[-1] == ("megamoe", 32, 32)
    else:
        assert _ops(log) == _BUCKET_PATH, log
        assert log[0] == ("megamoe_bucket", rows, rows)
        assert log[-1] == ("megamoe_in_bucket", rows, rows)


def test_the_guard_reads_this_ranks_rows():
    """Each rank decides on its own rows; every rank launches the chunk's bucket (96 rows)."""
    counts = [32, 8, 16, 64, 96, 32, 0, 1, 31, 33, 32, 96, 64, 16, 8, 2]
    assert len(counts) == _RANKS
    for rank, n in enumerate(counts):
        log = _run(counts, rank, 96)
        if n == 32:
            assert _ops(log) == _STAGING_PATH and log[-1] == ("megamoe", 96, 32), (rank, log)
        elif n:
            assert _ops(log) == _BUCKET_PATH, (rank, log)
            assert log[0] == ("megamoe_bucket", 96, n) and log[-1] == ("megamoe_in_bucket", 96, n)
        else:
            assert log == [("megamoe", 96, 0)], (rank, log)


@pytest.mark.parametrize("rows, chunk_rows", [(64, 32), (48, 24), (96, 32)])
def test_the_guard_reads_each_chunks_rows(rows, chunk_rows):
    """A step of more rows than one chunk goes in equal chunks; a 32-row chunk stages."""
    log = _run([rows] * _RANKS, 0, 32)
    launches = [entry for entry in log if entry[0] in ("megamoe", "megamoe_in_bucket")]
    assert launches == [
        ("megamoe", chunk_rows, chunk_rows)
        if chunk_rows == 32
        else ("megamoe_in_bucket", chunk_rows, chunk_rows)
    ] * (rows // chunk_rows), log


@pytest.mark.parametrize("extra", [0, 1], ids=["max_m", "max_m_plus_1"])
def test_more_rows_than_the_routing_kernel_serves_are_staged(extra):
    """One chunk of ``MAX_M`` (2048) rows per rank goes into the bucket; one row more stages."""
    rows = _route_max() + extra
    log = _run([rows] * _RANKS, 0, 2 * _route_max())
    if extra:
        assert _ops(log) == _STAGING_PATH and log[-1] == ("megamoe", rows, rows), log
    else:
        assert _ops(log) == _BUCKET_PATH, log
        assert log[0] == ("megamoe_bucket", rows, rows), log
        assert log[-1] == ("megamoe_in_bucket", rows, rows), log


def test_chunks_follow_the_largest_rank():
    """A step goes in as many chunks as its largest rank needs; a chunk's bucket is the largest
    share of it over the ranks, and a rank's rows split near-equally, the larger first."""
    counts = [7, 0, 3, 12, 5, 1, 9, 2, 4, 6, 8, 10, 11, 3, 2, 1]
    assert len(counts) == _RANKS
    # Rank 2: 3 rows in ceil(12 / 4) = 3 chunks of 1; rank 3's 12 rows make each bucket 4 rows.
    log = _run(counts, 2, 4)
    launches = [entry for entry in log if entry[0] in ("megamoe", "megamoe_in_bucket")]
    assert launches == [("megamoe_in_bucket", 4, 1)] * 3, log
