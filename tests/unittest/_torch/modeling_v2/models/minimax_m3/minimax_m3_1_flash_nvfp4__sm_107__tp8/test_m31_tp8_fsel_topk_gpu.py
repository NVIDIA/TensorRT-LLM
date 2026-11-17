# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FSEL top-16 select (family ``minimax_m3/ops/m3_fsel_topk``, ``mv2_minimax_m3::minimax_m3_sm107tp4_fsel_topk``): the
same table as the TP8 target's SelectLongDsl (``m3_select_long_dsl``) and the select contract's reference, as a set
and in order, on every certified cell, on mixed / tied / clustered / edge rows, both score layouts, inside a CUDA
graph, and at the long-context decode rows 1..8 and the bound + 1."""

import importlib
import zlib

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import m3_fsel_topk as fsel
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.m3_fsel_topk import TOPK
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.m3_fsel_topk import m3_fsel_topk as op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

_TP8 = "tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8"

# Valid-block counts on every path: <= 16 (copied), the 6-, 16- and 18-key bodies and their edges, past the row
# (clamped, no local forcing), and zero / negative (CUDA-graph padding rows).
_MIXED_VALID = (
    0,
    5,
    16,
    17,
    200,
    1536,
    1537,
    4096,
    4097,
    4224,
    8192,
    8193,
    9000,
    1,
    300,
    6000,
    129,
    3000,
    -3,
)


def _dsl():
    """The TP8 target's SelectLongDsl entry (the select this op replaces), or None when its stack does not import."""
    try:
        from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.attention.m3_select_long_dsl import (  # noqa: E501
            m3_select_long_dsl,
        )
    # ImportError: the optional CuTe DSL stack; the comparison against the reference still runs
    except ImportError:
        return None
    return m3_select_long_dsl


def _scores(spec, g):
    nb, tq, h = spec["num_blocks"], spec["total_q"], spec["num_kv_heads"]
    s = torch.randn(h, nb, tq, generator=g, device="cuda", dtype=torch.float32)
    kind = spec["scores"]
    if kind == "ties":
        # Even tokens: a flat row with ten winners (thousands tie at the 16th-largest score: pool overflow, the radix
        # fallback's mass-tie path). Odd tokens: a coarse grid (ties broken by block index).
        s[:, :, 0::2] = 0.25
        winners = torch.randint(0, nb, (h, 10, tq), generator=g, device="cuda")
        s[:, :, 0::2].scatter_(1, winners[:, :, 0::2], 3.0)
        s[:, :, 1::2] = (s[:, :, 1::2] * 2).round() / 2
    elif kind == "clustered":
        # The top scores in one contiguous run of blocks (one warp's lanes): the witness pool's worst case.
        start = int(torch.randint(0, max(1, nb - 64), (1,), generator=g, device="cuda"))
        s[:, start : start + 48, :] += 6.0
    elif kind == "edges":
        s[:, 3::40, :] = -0.0
        s[:, 4::40, :] = 0.0
        s[:, 5::41, :] = 1e-40
        s[:, 6::43, :] = -1e-40
        s[:, 9::97, :] = 1e28
        s[:, 10::89, :] = -1e28
    elif kind == "nonfinite":
        s[:, 7::50, :] = float("inf")
        s[:, 11::70, :] = float("nan")
        s[:, :, 1::2] = float("-inf")
        s[:, 0:5, 1::2] = 1.0
    return s


def _valid(spec, g):
    tq, nb = spec["total_q"], spec["num_blocks"]
    if spec["valid"] == "mixed":
        vals = [_MIXED_VALID[i % len(_MIXED_VALID)] for i in range(tq)]
    elif spec["valid"] == "c1_150k":
        # 1 request x 8 query rows at 150,000 tokens (kv_len 149,993 .. 150,000), 128-token blocks.
        vals = [(150000 - tq + t + 1 + 127) // 128 for t in range(tq)]
    else:
        vals = (nb - torch.randint(0, 12, (tq,), generator=g, device="cuda")).tolist()
        vals[0] = nb
    return torch.tensor(vals, dtype=torch.int32, device="cuda")


def _case(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = _scores(spec, g)
    nvb = _valid(spec, g)
    out = torch.empty(
        spec["num_kv_heads"], spec["total_q"], TOPK, dtype=torch.int32, device="cuda"
    ).permute(1, 0, 2)
    return scores, nvb, out, spec["init_blocks"], spec["local_blocks"]


def _check(scores, nvb, out, init_blocks, local_blocks):
    """The op's table equals the reference (set and order: both are ascending ids, -1 tail) and SelectLongDsl's."""
    ref = op.reference(scores, nvb, out, init_blocks, local_blocks)
    assert torch.equal(out, ref), "FSEL table differs from the select contract's reference"
    dsl = _dsl()
    if dsl is not None and scores.shape[1] <= 9216:
        dsl_out = torch.empty_like(out)
        dsl(scores, nvb, dsl_out, init_blocks, local_blocks, False)
        assert torch.equal(out, dsl_out), "FSEL table differs from SelectLongDsl"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    scores, nvb, out, init_blocks, local_blocks = _case(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    op(scores, nvb, out, init_blocks, local_blocks)
    _check(scores, nvb, out, init_blocks, local_blocks)


@pytest.mark.parametrize("kind", ["random", "ties", "clustered", "edges"])
@pytest.mark.parametrize("num_blocks", [129, 8193, 9216])
@pytest.mark.parametrize("seed", [0])
def test_mixed_rows_match_selectlongdsl(kind, num_blocks, seed) -> None:
    spec = dict(
        num_blocks=num_blocks,
        total_q=len(_MIXED_VALID),
        valid="mixed",
        scores=kind,
        num_kv_heads=1,
        init_blocks=0,
        local_blocks=1,
    )
    scores, nvb, out, init_blocks, local_blocks = _case(spec, seed=seed * 1000 + num_blocks)
    op(scores, nvb, out, init_blocks, local_blocks, True)
    _check(scores, nvb, out, init_blocks, local_blocks)


@pytest.mark.parametrize("layout", ["kv_blocks_q", "kv_q_blocks"])
@pytest.mark.parametrize("valid", [17, 1172, 1537, 4097, 9216])
def test_both_score_layouts(layout, valid) -> None:
    """The scorer's block-major layout (block stride = total_q) and the row-contiguous one (block stride 1)."""
    g = torch.Generator(device="cuda").manual_seed(valid)
    nb = max(valid, 129)
    s = torch.randn(1, nb, 8, generator=g, device="cuda")
    if layout == "kv_q_blocks":
        s = s.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    nvb = torch.tensor([valid - (t % 2) for t in range(8)], dtype=torch.int32, device="cuda")
    out = torch.empty(1, 8, TOPK, dtype=torch.int32, device="cuda").permute(1, 0, 2)
    op(s, nvb, out, 0, 1, True)
    _check(s, nvb, out, 0, 1)


def test_graph_replay_selects_the_same() -> None:
    spec = op.CELLS[0].spec
    scores, nvb, out, init_blocks, local_blocks = _case(spec, seed=9)
    ref = op.reference(scores, nvb, out, init_blocks, local_blocks)
    fsel.prepare(scores.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op(scores, nvb, out, init_blocks, local_blocks, True, 0)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(scores, nvb, out, init_blocks, local_blocks, True, 0)
    for _ in range(3):
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


@pytest.mark.parametrize("total_q", [1, 8, 9])
def test_long_context_rows_and_the_bound_plus_one(total_q) -> None:
    """The long-context shape (a 1M-token cache's 8193-block scores, 150K-token rows) at every row count the TP8 step
    dispatches to FSEL (1..8) and at the bound + 1 (9: the target runs the DSL select there; the op itself stays
    exact), at the target's launch (PDL, its trigger phase, variant 6), through the target adapter."""
    modeling = importlib.import_module(f"{_TP8}.modeling")
    spec = dict(
        num_blocks=8193,
        total_q=total_q,
        valid="c1_150k",
        scores="random",
        num_kv_heads=1,
        init_blocks=0,
        local_blocks=1,
    )
    for seed in (11, 12, 13):
        scores, nvb, _, init_blocks, local_blocks = _case(spec, seed=seed)
        table = fsel.fsel_select_blocks(
            scores,
            n_valid_blocks=nvb,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            use_pdl=True,
            trigger_phase=modeling.FSEL_TRIGGER_PHASE,
            variant=modeling.FSEL_VARIANT,
        )
        assert table is not None
        _check(scores, nvb, table, init_blocks, local_blocks)
    assert (total_q <= modeling.FSEL_MAX_ROWS) == (total_q <= 8)
