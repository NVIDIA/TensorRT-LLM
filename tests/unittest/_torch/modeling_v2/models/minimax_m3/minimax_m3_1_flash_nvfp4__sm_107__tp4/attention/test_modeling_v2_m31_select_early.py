# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The trigger move (tp4 modeling.py ``SELECT_EARLY_MAX_ROWS``): the long-row select releases the sparse decode
right after its own grid wait at up to 8 query rows (the served c1 K7 step), default on, kill switch
``TRTLLM_M31_DISABLE_SELECT_EARLY``.

- The table: 8 rows and fewer take the early release, 9 and more do not; the prebuild compiles the early key; the
  decode step reads the constant and prints the SELECT_EARLY engagement line.
- The kill switch (a fresh interpreter, the switch is read at import): 0 rows, no early key (the TP 2.4 path); an
  unset spelling leaves it on.
- Exact by construction (only where the dependent may launch moves): the select's table with the early release is bit
  for bit the trigger-phase table and the 1024-thread select's, on the served shape (1 request x 8 rows, 1 KV head,
  150K tokens = 1,172 valid blocks) and on padded / clamped / tied / non-finite rows, eager and under graph replay.
"""

import ast
import inspect
import json
import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
    modeling as dispatch,
)

MODULE = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
    "minimax_m3_1_flash_nvfp4__sm_107__tp4.modeling"
)
SERVED_ROWS = 8  # c1 K7: 1 request x (1 + 7) verify rows
TOPK = 16
INIT_BLOCKS, LOCAL_BLOCKS = 0, 1  # M3's forcing (the select op's certified cells)

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the select is a CUDA kernel")


def _probe(env_extra):
    """SELECT_EARLY_MAX_ROWS and the select prebuild keys of a fresh import of the target under ``env_extra`` (every
    other TRTLLM_M31_* switch cleared)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_"))
    }
    env.update(env_extra)
    code = (
        "import json, importlib\n"
        f"m = importlib.import_module({MODULE!r})\n"
        "print('RESULT ' + json.dumps({'rows': m.SELECT_EARLY_MAX_ROWS,"
        " 'keys': [list(k) for k in m.long_select_dsl_keys()], 'dsl': bool(m.LONG_SELECT_DSL.keys)}))\n"
    )
    p = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert p.returncode == 0, f"import rc {p.returncode}: {p.stderr[-2000:]}"
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, p.stdout[-2000:]
    return json.loads(lines[0][len("RESULT ") :])


def test_the_served_step_takes_the_early_release() -> None:
    expected = 0 if dispatch.m31_disabled("SELECT_EARLY") else SERVED_ROWS
    assert dispatch.SELECT_EARLY_MAX_ROWS == expected
    for rows in range(
        1, 2 * SERVED_ROWS + 1
    ):  # the decode step's own decision (modeling's select site)
        assert (rows <= dispatch.SELECT_EARLY_MAX_ROWS) == (
            bool(expected) and rows <= SERVED_ROWS
        ), rows
    keys = dispatch.long_select_dsl_keys()
    if dispatch.LONG_SELECT_DSL.keys:
        pdl = keys[0][0]
        assert ((pdl, True, 0) in keys) == bool(expected), keys
        assert (pdl, False, dispatch.SELECT_LONG_TRIGGER_PHASE) in keys, keys


def test_the_decode_step_reads_the_constant_and_logs_the_engagement() -> None:
    src = inspect.getsource(dispatch)
    assert src.count("early_trigger = num_tokens <= SELECT_EARLY_MAX_ROWS") == 1
    assert 'm31_disabled("SELECT_EARLY")' in src
    site = src[src.index("early_trigger = num_tokens <= SELECT_EARLY_MAX_ROWS") :]
    # The select site runs to the table hand-off to the sparse decode (TP4_FSEL put the FSEL select, with its own
    # "uncovered(" line, ahead of the DSL select).
    site = site[: site.index("topk = table.permute(1, 0, 2)")]
    assert "early_trigger=early_trigger" in site, (
        "the DSL select no longer takes SELECT_EARLY's release point"
    )
    assert site.count('"SELECT_EARLY"') == 2, (
        "the SELECT_EARLY engagement line must print on the FSEL and DSL paths"
    )
    assert "if early_trigger and FSEL_TRIGGER_PHASE == 0:" in site and "if early_trigger:" in site
    assigns = {ast.unparse(n) for n in ast.parse(src).body if isinstance(n, ast.Assign)}
    assert "FSEL_TRIGGER_PHASE = 0 if SELECT_EARLY_MAX_ROWS >= FSEL_MAX_ROWS else 1" in assigns


def test_kill_switch_restores_the_trigger_phase_everywhere() -> None:
    on = _probe({})
    off = _probe({"TRTLLM_M31_DISABLE_SELECT_EARLY": "1"})
    unset = _probe({"TRTLLM_M31_DISABLE_SELECT_EARLY": "0"})
    assert on["rows"] == SERVED_ROWS and unset["rows"] == SERVED_ROWS
    assert off["rows"] == 0
    assert not any(k[1] for k in off["keys"]), (
        f"kill switch still prebuilds an early key: {off['keys']}"
    )
    if on["dsl"]:
        assert any(k[1] and k[2] == 0 for k in on["keys"]), f"no early key prebuilt: {on['keys']}"
        assert [k for k in on["keys"] if not k[1]] == off["keys"], (
            "the kill switch changes the other keys"
        )


# ---- exactness: the table is the trigger-phase table, bit for bit -------------------------------------------------


def _rows(case: str, g: torch.Generator):
    """(scores [1, n_blocks, rows] fp32, n_valid [rows] int32) for a case."""
    if case == "served":  # 1 request x 8 rows at 150K tokens = 1,172 valid blocks of 128
        nb, rows = 1172, SERVED_ROWS
        s = torch.randn(1, nb, 1, generator=g, device="cuda").expand(1, nb, rows).contiguous()
        s = s + 0.05 * torch.randn(1, nb, rows, generator=g, device="cuda")
        nvb = torch.full((rows,), nb, dtype=torch.int32, device="cuda")
        return s, nvb
    if case == "padded":  # a padded graph row (0 valid), a short row, clamped and full rows
        nb, rows = 2048, SERVED_ROWS
        s = torch.randn(1, nb, rows, generator=g, device="cuda")
        nvb = torch.tensor([0, 5, 16, 17, 200, 1536, 2048, 9000], dtype=torch.int32, device="cuda")
        return s, nvb
    if case == "ties":
        nb, rows = 4500, SERVED_ROWS
        s = torch.full((1, nb, rows), 0.25, device="cuda")
        winners = torch.randint(0, nb, (1, 10, rows), generator=g, device="cuda")
        s.scatter_(1, winners, 3.0)
        s[:, :, 1::2] = (torch.randn(1, nb, rows // 2, generator=g, device="cuda") * 2).round() / 2
        nvb = torch.full((rows,), nb, dtype=torch.int32, device="cuda")
        return s, nvb
    if case == "nonfinite":
        nb, rows = 600, SERVED_ROWS
        s = torch.randn(1, nb, rows, generator=g, device="cuda")
        s[:, 7::50, :] = float("inf")
        s[:, 11::70, :] = float("nan")
        s[:, :, 1::2] = float("-inf")
        s[:, 0:5, 1::2] = 1.0
        nvb = torch.full((rows,), nb, dtype=torch.int32, device="cuda")
        return s, nvb
    raise ValueError(case)


def _select(scores, nvb, *, early: bool, dsl: bool):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.long_select import (  # noqa: E501
        long_select_blocks,
    )

    return long_select_blocks(
        scores,
        n_valid_blocks=nvb,
        init_blocks=INIT_BLOCKS,
        local_blocks=LOCAL_BLOCKS,
        use_pdl=True,
        early_trigger=early,
        trigger_phase=dispatch.SELECT_LONG_TRIGGER_PHASE,
        fast_tail=dispatch.SELECT_LONG_SL3,
        dsl=dsl,
    )


@_cuda
@pytest.mark.parametrize("case", ["served", "padded", "ties", "nonfinite"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_early_release_selects_the_same(case: str, seed: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(4801 + seed)
    scores, nvb = _rows(case, g)
    early = _select(scores, nvb, early=True, dsl=True)
    phase = _select(scores, nvb, early=False, dsl=True)
    ref = _select(scores, nvb, early=False, dsl=False)
    torch.cuda.synchronize()
    assert early is not None and phase is not None and ref is not None
    assert early.data_ptr() != phase.data_ptr()
    assert early.shape == (SERVED_ROWS, 1, TOPK) and early.dtype is torch.int32
    assert torch.equal(early, phase), f"{case}: the early release changes the selected blocks"
    if case in (
        "served",
        "padded",
    ):  # the certified forms; ties / non-finite: same kernel, early vs phase only
        assert torch.equal(early, ref), f"{case}: differs from the 1024-thread select"


@_cuda
def test_early_release_selects_the_same_under_graph_replay() -> None:
    """The served shape in a CUDA graph: fresh scores copied in, the select with the early release, a reader of its
    table; every replay equals the eager trigger-phase table of that replay's scores."""
    g = torch.Generator(device="cuda").manual_seed(4901)
    scores, nvb = _rows("served", g)
    static = scores.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _select(static, nvb, early=True, dsl=True)  # compile / load outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        table = _select(static, nvb, early=True, dsl=True)
        read = table.clone()
    for seed in (1, 2, 3):
        fresh, _ = _rows("served", torch.Generator(device="cuda").manual_seed(5000 + seed))
        static.copy_(fresh)
        read.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        want = _select(fresh, nvb, early=False, dsl=True)
        torch.cuda.synchronize()
        assert torch.equal(read, want), f"replay {seed}: the graphed early-release table differs"
