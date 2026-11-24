# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one GPU, sm_107) of helix8's ``dsa_oproj_push`` claim: at the DSA o_proj shape (``[rows,
1536] @ [7168, 1536]^T``, 12 heads x 128 per rank) and helix8's tactic for each row count, the push
kernel's values (``bf16_gemm_push_values``, its store-only variant) equal today's o_proj
(``rubin_linear``, cuBLAS) word for word: the push changes where the partial goes, not its bits.
helix8 checks the same on the device once per process and row count before it pushes; this test
states that the check passes at the rows of its 1- to 12-request steps (6 to 72 rows: the 8-, 16-
and 32-token tiles; past 32 rows 128-row weight tiles in two or three N tiles of 32, or one 64-token
tile). Eager, and in a CUDA graph behind a producer."""

import importlib.util
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_linear import (  # noqa: E501
    rubin_linear,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_bf16_gemm_push import (  # noqa: E501
    bf16_gemm_push_values,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="helix8's o_proj push is certified on sm_107",
)

N, K = 7168, 1536
_PKG = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0])
_HELIX8 = (
    _PKG / "_torch/_experimental/modeling_v2/models/kimi_k31/kimi_k31_pro_nvfp4__sm_107__helix8"
)
#: helix8's generation rows: b x (k5 + 1), b = 1..12.
_ROWS = tuple(6 * b for b in range(1, 13))


def _tactic(rows: int) -> tuple:
    """helix8's ``_DSA_OPROJ_PUSH_TACTICS`` entry for ``rows``, read off its source."""
    import ast

    tree = ast.parse((_HELIX8 / "modeling.py").read_text())
    (node,) = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "_DSA_OPROJ_PUSH_TACTICS"
    ]
    for bound, tactic in ast.literal_eval(node.value):
        if rows <= bound:
            return tactic
    raise AssertionError(f"no tactic for {rows} rows")


def _operands(rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, K, generator=g, device="cuda").bfloat16()
    w = (torch.randn(N, K, generator=g, device="cuda") * K**-0.5).bfloat16()
    return x, w


@pytest.mark.parametrize("rows", _ROWS)
def test_push_values_are_todays_o_proj(rows):
    tactic = _tactic(rows)
    for seed in range(3):
        x, w = _operands(rows, 100 * rows + seed)
        want = rubin_linear(x, w, False)
        got = bf16_gemm_push_values(x, w, torch.empty_like(want), tactic)
        torch.cuda.synchronize()
        diff = int((got.view(torch.int16) != want.view(torch.int16)).sum())
        assert diff == 0, f"{rows} rows seed {seed}: {diff} words differ from today's o_proj"


@pytest.mark.parametrize("rows", [33, 40, 50, 64, 65, 70])
def test_the_rows_past_the_last_n_tile_are_not_written(rows):
    """Past 32 rows the last N tile is partial: its rows past ``M`` are neither computed into nor
    stored past the output (the guard rows after it keep their bytes)."""
    tactic = _tactic(rows)
    x, w = _operands(rows, 9 + rows)
    want = rubin_linear(x, w, False)
    buf = torch.full((rows + 8, N), 3.0, dtype=torch.bfloat16, device="cuda")
    got = bf16_gemm_push_values(x, w, buf[:rows], tactic)
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert bool((buf[rows:] == 3.0).all()), "the store wrote past the output rows"


@pytest.mark.parametrize("rows", [12, 24, 36, 54, 72])
def test_push_values_in_a_graph_behind_a_producer(rows):
    """The served launch: a CUDA graph, the GEMM behind the kernel that writes its activation."""
    tactic = _tactic(rows)
    x, w = _operands(rows, 7 + rows)
    src = x.clone()
    out = torch.empty(rows, N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            x.copy_(src * 1.0)
            bf16_gemm_push_values(x, w, out, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            x.copy_(src * 1.0)
            bf16_gemm_push_values(x, w, out, tactic)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in range(2):
        src.copy_(_operands(rows, 1000 + seed)[0])
        out.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        want = rubin_linear(src, w, False)
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), f"replay {seed}"
