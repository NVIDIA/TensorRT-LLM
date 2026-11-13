# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LOCAL_MOE (the TP8 locality MoE engine variant) on CPU, without CUDA imports: the state factory the target's
constructor installs, where the constructor installs it, and the domain-major work plan's exactness argument (every
FC1 unit keeps the stock plan's split stage, so the FC1 sums are the stock ones bit for bit)."""

import ast
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir())
TARGET = (
    ROOT
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp8"
)
PACKAGE = TARGET / "local_moe"


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        # local_moe is a subpackage of the target: its receipt reads the target's ..m31_switches, stubbed here so
        # the import stays CPU-only (no tensorrt_llm import).
        parent = types.ModuleType("_tp8_target_cpu")
        parent.__path__ = [str(TARGET)]
        switches = types.ModuleType(parent.__name__ + ".m31_switches")
        switches.m31_disabled = lambda item: False
        self.name = parent.__name__ + ".local_moe"
        package = types.ModuleType(self.name)
        package.__path__ = [str(PACKAGE)]
        self.local = types.ModuleType(self.name + ".m31_local_moe_op")
        self.local.make_state = Mock(return_value=object())
        self.modules = patch.dict(
            sys.modules,
            {
                parent.__name__: parent,
                switches.__name__: switches,
                self.name: package,
                self.local.__name__: self.local,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.bootstrap = importlib.import_module(self.name + ".bootstrap")

    def model(self):
        return NS(M3MoeState=Mock(return_value=object()), _ENGINE_STATES={})

    def test_only_the_8_row_shared_call_uses_local_factory_on_each_rank(self):
        for rank in range(8):
            model = self.model()
            stock = model.M3MoeState
            self.bootstrap.install_tp8(model, NS(tp_size=8, tp_rank=rank))
            self.assertIs(model.M3MoeState(0, 8, shared=True), self.local.make_state.return_value)
            self.local.make_state.assert_called_with(0, 8, num_ctas=None, shared=True)
            for tokens, shared in ((8, False), (16, True), (1, True), (176, True)):
                self.assertIs(model.M3MoeState(0, tokens, shared=shared), stock.return_value)
            self.assertEqual(stock.call_count, 4)

    def test_idempotent_install_and_rank_change(self):
        model = self.model()
        mapping = NS(tp_size=8, tp_rank=0)
        self.bootstrap.install_tp8(model, mapping)
        factory = model.M3MoeState
        self.bootstrap.install_tp8(model, mapping)
        self.assertIs(model.M3MoeState, factory)
        with self.assertRaisesRegex(RuntimeError, "rank changed"):
            self.bootstrap.install_tp8(model, NS(tp_size=8, tp_rank=1))

    def test_no_tp2_install(self):
        model = self.model()
        stock = model.M3MoeState
        with self.assertRaisesRegex(ValueError, "requires TP8 rank"):
            self.bootstrap.install_tp8(model, NS(tp_size=2, tp_rank=0))
        self.assertIs(model.M3MoeState, stock)
        self.local.make_state.assert_not_called()

    def test_preserves_existing_pre_engine_install_requirement(self):
        model = self.model()
        model._ENGINE_STATES["already-built"] = object()
        stock = model.M3MoeState
        with self.assertRaisesRegex(RuntimeError, "follows engine state build"):
            self.bootstrap.install_tp8(model, NS(tp_size=8, tp_rank=0))
        self.assertIs(model.M3MoeState, stock)
        self.local.make_state.assert_not_called()


def _load_plan():
    spec = importlib.util.spec_from_file_location("_tp8_local_plan_cpu", PACKAGE / "local_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LocalPlanTests(unittest.TestCase):
    """The decode shape (8 rows, TP8 width 384, 96 routed CTAs per domain) and the other row counts."""

    def setUp(self):
        self.lp = _load_plan()

    def _plans(self, m):
        tw = 0.0 if m == 1 else 1.0
        plan, nb_cap = self.lp.make_domain_phase_plan(m, 384, 96, 2, tw, True)
        stock, nb_stock = self.lp.stock_phase_plan(m, 384, 192, 2, tw, True)
        return plan, nb_cap, stock, nb_stock

    def test_domain_plan_is_valid(self):
        for m in range(1, 9):
            plan, nb_cap, _, nb_stock = self._plans(m)
            self.assertEqual(self.lp.validate_plan(plan, m, 384, 96), [], m)
            self.assertEqual(nb_cap, nb_stock)
            self.assertEqual(nb_cap, 1)

    def test_split_stages_match_stock(self):
        for m in range(1, 9):
            plan, _, stock, _ = self._plans(m)
            a = self.lp.split_structure(plan, m, 384, True)
            b = self.lp.split_structure(stock, m, 384, False)
            self.assertEqual(a.keys(), b.keys(), m)
            self.assertEqual({k: v[0] for k, v in a.items()}, {k: v[0] for k, v in b.items()}, m)

    def test_ctas_stay_in_their_domain(self):
        plan, _, _, _ = self._plans(8)
        g = self.lp.geometry(384)
        for gg in range(4, 33):
            for c in range(192):
                for p in range((gg + 3) // 4):
                    w = plan[gg][c][p]
                    i0, i1 = w & 0xFFFF, w >> 16
                    for i in range(i0, i1):
                        self.assertEqual(self.lp.remap_unit(i // 6, g)[0], c // 96)

    def test_logical_map(self):
        dom = [0] * 100 + [1] * 100 + [-1] * 12
        lmap, ext = self.lp.logical_map(dom, 96, 20)
        self.assertEqual(sorted(lmap), list(range(212)))
        self.assertTrue(all(lmap[s] < 96 for s in range(96)))
        self.assertTrue(all(96 <= lmap[s] < 192 for s in range(100, 196)))
        self.assertEqual(len(ext), 20)


class ConstructorInstallTests(unittest.TestCase):
    """The TP8 target's constructor installs the factory on its own modeling module, under the LOCAL_MOE kill switch."""

    def test_the_constructor_installs_under_the_kill_switch(self):
        src = (TARGET / "modeling.py").read_text()
        guards = [
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "not m31_disabled('LOCAL_MOE')"
        ]
        self.assertEqual(len(guards), 1)
        body = ast.unparse(ast.Module(guards[0].body, []))
        self.assertIn("from .local_moe.bootstrap import install_tp8", body)
        self.assertIn("install_tp8(sys.modules[__name__], model_config.mapping)", body)
        self.assertEqual(src.count("install_tp8(sys.modules[__name__], model_config.mapping)"), 1)
        self.assertNotIn("install_tp8", (TARGET.parent / "routing.py").read_text())


if __name__ == "__main__":
    unittest.main()
