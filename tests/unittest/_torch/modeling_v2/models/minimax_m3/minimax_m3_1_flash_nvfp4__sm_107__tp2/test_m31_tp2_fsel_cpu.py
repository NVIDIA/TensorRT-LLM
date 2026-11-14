# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the actual bounded FSEL adapter and TP2 selector fallback, no device oracle."""

import ast
import importlib
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.cpu_only


def test_actual_import_dispatch_and_target_fallback():
    family = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    )
    fsel = SimpleNamespace(
        MAX_BLOCKS=9216,
        fsel_op=Mock(return_value=object()),
        prepare=Mock(),
        log_engaged=Mock(),
        fsel_select_blocks=Mock(),
    )
    package = types.ModuleType("tp2_fsel_cpu_family")
    package.__path__ = [str(family)]
    ops = types.ModuleType("tp2_fsel_cpu_family.ops")
    ops.__path__ = [str(family / "ops")]
    ops.m3_fsel_topk = fsel
    with patch.dict(sys.modules, {"tp2_fsel_cpu_family": package, "tp2_fsel_cpu_family.ops": ops}):
        adapter = importlib.import_module("tp2_fsel_cpu_family.m31_decode_fsel")
    assert adapter.prepare(
        "cuda", enabled=True, target="sm107_tp2", rank=0, max_rows=8, trigger_phase=1
    )
    fsel.prepare.assert_called_once_with("cuda")
    assert not adapter.prepare(
        "cuda", enabled=False, target="sm107_tp2", rank=0, max_rows=8, trigger_phase=1
    )
    fsel.fsel_op.return_value = None
    assert not adapter.prepare(
        "cuda", enabled=True, target="sm107_tp2", rank=0, max_rows=8, trigger_phase=1
    )
    fsel.fsel_op.return_value = object()

    class Scores:
        def __init__(self, heads=1, blocks=8193, rows=8):
            self.shape = (heads, blocks, rows)

        def dim(self):
            return len(self.shape)

    scores = Scores()
    counts = object()
    table = object()
    fsel.fsel_select_blocks.return_value = table

    def select(scores=scores, **overrides):
        kwargs = dict(
            n_valid_blocks=counts,
            num_tokens=8,
            init_blocks=0,
            local_blocks=1,
            enabled=True,
            max_rows=8,
            use_pdl=True,
            trigger_phase=1,
        )
        kwargs.update(overrides)
        return adapter.select(scores, **kwargs)

    assert select() is table
    args = fsel.fsel_select_blocks.call_args
    assert args.args == (scores,) and args.kwargs["n_valid_blocks"] is counts
    assert args.kwargs["trigger_phase"] == 1 and args.kwargs["use_pdl"] is True
    for rows in (1, 4, 7, 8):
        assert select(num_tokens=rows) is table
    for blocks in (129, 2048, 8193, 9216):
        assert select(Scores(blocks=blocks)) is table
    for kwargs in (
        {"enabled": False},
        {"num_tokens": 0},
        {"num_tokens": 9},
        {"init_blocks": 1},
        {"local_blocks": 0},
    ):
        assert select(**kwargs) is None
    assert select(Scores(heads=2), num_kv_heads=2) is table
    assert select(Scores(heads=3), num_kv_heads=3) is None
    for bad in (Scores(heads=2), Scores(blocks=128), Scores(blocks=9217)):
        assert select(bad) is None
    fsel.fsel_select_blocks.return_value = None
    assert select() is None
    # Execute the complete actual target's selection/fallback slice, not a mirrored branch.
    source = (family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/modeling.py").read_text()
    start = source.index("            table = m31_decode_fsel.select(")
    end = source.index("            topk = table.permute", start)
    code = compile(textwrap.dedent(source[start:end]), "actual_tp2_select_slice", "exec")
    tree = ast.parse(source)
    # The slice's actual helpers: the long select's triggers and the r7 knob-taken log (stack v2).
    helpers = compile(
        ast.Module(
            body=[
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name in ("_select_triggers", "_select_knobs_taken")
            ],
            type_ignores=[],
        ),
        "actual_tp2_select_helpers",
        "exec",
    )
    for enabled, rows, heads, has_fsel, has_long, want, bound in [
        (True, 8, 1, True, True, "fsel", 8),
        (False, 8, 1, True, True, "long", 8),
        (True, 9, 1, True, True, "long", 8),
        (True, 8, 2, True, True, "fsel", 8),
        (True, 8, 1, False, True, "long", 8),
        (True, 8, 1, False, False, "stock", 8),
        # W6 FSEL_WIDE: 9..32 rows on FSEL at its exit release (phase 2); 33 keeps the DSL select at phase 0.
        (True, 9, 2, True, True, "fsel", 32),
        (True, 16, 2, True, True, "fsel", 32),
        (True, 32, 2, True, True, "fsel", 32),
        (True, 33, 2, True, True, "long", 32),
        (False, 16, 2, True, True, "long", 32),
        (True, 24, 2, False, True, "long", 32),
    ]:
        ftable, ltable, stable = object(), object(), object()
        knob_taken = Mock()
        fsel.fsel_select_blocks.return_value = ftable if has_fsel else None
        taken, missed, uncovered = Mock(), Mock(), Mock()
        long, stock = (
            Mock(return_value=ltable if has_long else None),
            Mock(return_value=stable),
        )
        ns = dict(
            m31_decode_fsel=adapter,
            max_score_kv=Scores(heads=heads, rows=rows),
            n_valid_blocks=counts,
            num_tokens=rows,
            config=SimpleNamespace(init_blocks=0, local_blocks=1, num_kv_heads=heads),
            FSEL_SELECT=SimpleNamespace(keys={"fsel"} if enabled else set(), path="fsel"),
            FSEL_MAX_ROWS=bound,
            FSEL_TRIGGER_PHASE=1,
            FSEL_WIDE_TRIGGER_PHASE=2,
            get_env_enable_pdl=lambda: True,
            SELECT_EARLY_MAX_ROWS=0,
            SELECT_LONG_TRIGGER_PHASE=1,
            SELECT_LONG_TRIGGER_MAX_ROWS=8,
            LONG_SELECT_DSL=SimpleNamespace(keys={4}, path="long"),
            LONG_SELECT=SimpleNamespace(path="longbase"),
            _select_long_dsl_kernel=SimpleNamespace(
                kernel_key=lambda *a: "ready", kernel_cache={"ready"}
            ),
            missed=missed,
            long_select_blocks=long,
            SELECT_LONG_SL3=False,
            taken=taken,
            uncovered=uncovered,
            LONG_SELECT_MIN_BLOCKS=129,
            LONG_SELECT_MAX_BLOCKS=9216,
            m3_select_blocks=stock,
            MSA_REQUIRED_TOPK=16,
            Tuple=tuple,
            SELECT_EARLY_MAX_ROWS_DEFAULT=0,
            SELECT_LONG_TRIGGER_MAX_ROWS_DEFAULT=8,
            _r7_knob_taken=knob_taken,
        )
        exec(helpers, ns)
        exec(code, ns)
        # At the knobs' defaults no select logs a knob as taken.
        knob_taken.assert_not_called()
        assert ns["table"] is {"fsel": ftable, "long": ltable, "stock": stable}[want]
        assert long.call_count == int(want != "fsel") and stock.call_count == int(want == "stock")
        if has_fsel and enabled and rows <= bound:
            assert fsel.fsel_select_blocks.call_args.kwargs["trigger_phase"] == (
                1 if rows <= 8 else 2
            )
        if want == "fsel":
            taken.assert_called_once_with("fsel", rows)
        if want == "long":
            taken.assert_called_once_with("long", rows)
        if want != "fsel":
            assert long.call_args.kwargs["early_trigger"] is False and long.call_args.kwargs[
                "trigger_phase"
            ] == (1 if rows <= 8 else 0)
    assignments = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }

    assert ast.literal_eval(assignments["SELECT_EARLY_MAX_ROWS"]) == 0
    assert (
        ast.literal_eval(assignments["FSEL_NARROW_MAX_ROWS"]) == 8
        and ast.literal_eval(assignments["FSEL_TRIGGER_PHASE"]) == 1
        and ast.literal_eval(assignments["FSEL_WIDE_TRIGGER_PHASE"]) == 2
        and ast.literal_eval(assignments["SELECT_LONG_TRIGGER_MAX_ROWS"]) == 8
    )
    bound = compile(ast.Expression(assignments["FSEL_MAX_ROWS"]), "actual_bound", "eval")
    assert eval(bound, {"FSEL_WIDE": True, "FSEL_NARROW_MAX_ROWS": 8}) == 32
    assert eval(bound, {"FSEL_WIDE": False, "FSEL_NARROW_MAX_ROWS": 8}) == 8
    wide = compile(ast.Expression(assignments["FSEL_WIDE"]), "actual_wide", "eval")
    for topk, killed, want in ((True, False, True), (True, True, False), (False, False, False)):
        got = eval(
            wide,
            {
                "FSEL_TOPK": topk,
                "m31_disabled": lambda i, k=killed: k and i == "FSEL_WIDE",
            },
        )
        assert got is want, (topk, killed, got)
    switch = compile(ast.Expression(assignments["FSEL_TOPK"]), "actual_switch", "eval")
    assert eval(switch, {"glue_disabled": lambda part, *legacy: False}) is True
    assert eval(switch, {"glue_disabled": lambda part, *legacy: True}) is False
    assert ast.unparse(assignments["FSEL_TOPK"]) == "not glue_disabled('FSEL', 'TP2_FSEL')"
    registry = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "FAST_PATHS"
    )
    assert any(isinstance(x, ast.Name) and x.id == "FSEL_SELECT" for x in registry.elts)
    assert any(
        isinstance(n, ast.ImportFrom)
        and n.level == 2
        and any(a.name == "m31_decode_fsel" for a in n.names)
        for n in tree.body
    )


if __name__ == "__main__":
    test_actual_import_dispatch_and_target_fallback()
    print("TP2_FSEL_ACTUAL_ADAPTER_AND_FALLBACK_CPU_PASS")


def test_tp2_certification_uses_two_head_cells() -> None:
    family = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    )
    tree = ast.parse((family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/modeling.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "FSEL_SELECT"
    )
    predicate = eval(compile(ast.Expression(node.value.args[3]), "actual_fsel_key", "eval"))
    assert predicate(dict(init_blocks=0, local_blocks=1, num_kv_heads=2)) == ("fsel",)
    assert predicate(dict(init_blocks=0, local_blocks=1, num_kv_heads=1)) == ()
    assert predicate(dict(init_blocks=1, local_blocks=1, num_kv_heads=2)) == ()
