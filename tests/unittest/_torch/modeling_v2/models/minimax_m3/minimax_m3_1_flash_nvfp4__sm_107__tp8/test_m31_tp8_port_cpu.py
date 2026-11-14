# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatch checks of the TP8 target, executable with stdlib unittest on a host without CUDA.

Execute the TP8 worker and target methods with dependency doubles. These checks establish
wiring and fallback behavior; packaged CUDA tests establish numerics and replay.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "tensorrt_llm").is_dir()))
FAMILY = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
TARGET = FAMILY / "minimax_m3_1_flash_nvfp4__sm_107__tp8"


def worker_method(
    name: str,
    namespace: dict,
    *,
    stop: str | None = None,
    source: str = "dspark.py",
    cls_name: str | None = "M3DSparkWorker",
    omit_imports: bool = False,
):
    """Compile the served class's actual method, optionally ending before a later subsystem."""
    tree = ast.parse((TARGET / source).read_text())
    scope = (
        tree.body
        if cls_name is None
        else next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)).body
    )
    fn = next((n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name))
    if stop is not None:
        fn.body = fn.body[: next((i for i, n in enumerate(fn.body) if stop in ast.unparse(n)))]
    if omit_imports:
        fn.body = [n for n in fn.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module = ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0), fn], [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(TARGET / source), "exec"), namespace)
    return namespace[name]


def module_constant(name: str, source: str = "dspark.py"):
    """A module-level literal of the served source (e.g. a tuple of names), read without importing it."""
    tree = ast.parse((TARGET / source).read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
    )
    return ast.literal_eval(node.value)


class TestTP8Chain(unittest.TestCase):
    def prebuild(self, killed: set[str], *, his=(0, 0), stop="mx ="):
        ns = {
            "_WORKER_HOOKS": module_constant("_WORKER_HOOKS"),
            "functools": functools,
            "m31_disabled": killed.__contains__,
            "s10_applies": lambda _: True,
            "logger": Mock(),
            "FC_SHARD": False,
            "TAILBK_SPEC": True,
            "FOLD_F8": True,
            "FOLD_F6": True,
            "DRAFT_APPEND_FUSE": False,
            "MAX_FAST_BATCH": 1,
            "route_his_layout": Mock(return_value=his),
            "route_mx_linears": Mock(return_value=0),
            "route_gu_swiglu": Mock(return_value=0),
            "install_draft_swiglu": Mock(return_value=5),
            "native_swiglu": Mock(),
            "fold_k_norm_weight": lambda v: v,
            "enable_parallel_refresh": lambda v: v,
        }
        for name in (
            "fused_ctx_store",
            "s10c_c_drafter_inputs",
            "advance_ctx_and_read_view",
            "refresh_block_tables",
            "forward_indices",
            "qk_norm_rope",
            "_qk_norm_rope_append_logged",
        ):
            ns[name] = Mock(name=name)
        worker = Mock(
            mapping=SimpleNamespace(tp_rank=0), spec_config=SimpleNamespace(max_draft_len=7)
        )
        worker_method("prebuild", ns, stop=stop)(worker, None, SimpleNamespace())
        return (worker, ns)

    def test_served_worker_layout_precedes_existing_providers_and_obeys_kill(self):
        for killed in (set(), {"DRAFTER_HISLAYOUT"}):
            _, ns = self.prebuild(killed)
            self.assertEqual(ns["route_his_layout"].call_count, int(not killed))
            if not killed:
                self.assertEqual(ns["route_his_layout"].call_args.args[1], [7])

    def test_served_layout_keeps_later_providers_from_overriding_it(self):
        for killed in (set(), {"DRAFTER_HISLAYOUT"}):
            _, ns = self.prebuild(killed, his=(3, 1), stop="self.prebuild_s4")
            self.assertEqual(ns["route_gu_swiglu"].call_count, int(bool(killed)))


class TestTP8Fsel(unittest.TestCase):
    def test_actual_decode_selector_prefers_fsel_and_keeps_fallbacks(self):
        for enabled, rows, native in (
            (True, 8, object()),
            (False, 8, object()),
            (True, 9, object()),
            (True, 8, None),
        ):
            with self.subTest(enabled=enabled, rows=rows, native=native):
                fsel = Mock()
                fsel.fsel_select_blocks.return_value = native
                dsl_table = object()
                ns = {
                    "FSEL_SELECT": SimpleNamespace(
                        keys={"fsel"} if enabled else set(), path="tp8_fsel"
                    ),
                    "FSEL_MAX_ROWS": 8,
                    "FSEL_TRIGGER_PHASE": 0,
                    "FSEL_VARIANT": 6,
                    "SELECT_EARLY_MAX_ROWS": 8,
                    "SELECT_LONG_TRIGGER_PHASE": 1,
                    "LONG_SELECT_DSL": SimpleNamespace(keys=set(), path="dsl"),
                    "LONG_SELECT": SimpleNamespace(path="long"),
                    "long_select_covers": lambda _: True,
                    "_fsel": fsel,
                    "long_select_blocks": Mock(return_value=dsl_table),
                    "taken": Mock(),
                    "missed": Mock(),
                    "get_env_enable_pdl": lambda: True,
                }
                config = SimpleNamespace(
                    num_index_heads=1, num_kv_heads=1, init_blocks=0, local_blocks=1
                )
                target = SimpleNamespace(
                    core=SimpleNamespace(attn=[SimpleNamespace(m3_config=config)])
                )
                scores = SimpleNamespace(shape=(1, 1172, rows))
                call = worker_method(
                    "_select", ns, source="modeling.py", cls_name="DecodeTarget", omit_imports=True
                )
                out = call(target, 0, scores, {"msa_n_valid_blocks": list(range(rows))}, rows)
                want_native = enabled and rows <= 8 and (native is not None)
                self.assertIs(out, native if want_native else dsl_table)
                self.assertEqual(fsel.fsel_select_blocks.call_count, int(enabled and rows <= 8))
                self.assertEqual(ns["long_select_blocks"].call_count, int(not want_native))


class TestTP8MoeEngine(unittest.TestCase):
    def test_actual_engine_configuration_and_independent_kills(self):
        path = TARGET / "kernels/m3_moe/op.py"
        tree = ast.parse(path.read_text())
        cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "M3MoeState"))
        init = next(
            (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        )

        def assignment_index(key):
            return next(
                (
                    i
                    for i, n in enumerate(init.body)
                    if isinstance(n, ast.Assign)
                    and any(
                        (
                            isinstance(t, ast.Subscript)
                            and isinstance(t.slice, ast.Constant)
                            and (t.slice.value == key)
                            for t in n.targets
                        )
                    )
                )
            )

        body = init.body[assignment_index("sfa_cp") : assignment_index("b2img")]
        fn = ast.parse("def configure(self, num_tokens): pass").body[0]
        fn.body = body
        module = ast.fix_missing_locations(ast.Module([fn], []))
        for rows in range(1, 9):
            for killed in (set(), {"MOE_SFA_BULK"}, {"MOE_K128"}, {"MOE_SFA_BULK", "MOE_K128"}):
                ns = {
                    "SFA_CP_BY_M": (1,) * 8,
                    "m31_disabled": killed.__contains__,
                    "_SFA_K128_LOGGED": set(range(1, 9)),
                }
                exec(compile(module, str(path), "exec"), ns)
                state = SimpleNamespace(config={})
                ns["configure"](state, rows)
                self.assertEqual(state.config.get("sfa_bulk", 0), int("MOE_SFA_BULK" not in killed))
                self.assertEqual(state.config.get("k128", 0), int("MOE_K128" not in killed))


class TestTP8Layout(unittest.TestCase):
    def test_family_row_forward_and_fused_mlp_keep_allreduce_contract(self):
        class Tensor:
            dtype = "bf16"
            device = "cuda:0"

            def __init__(self, *shape):
                self.shape = shape
                self.data = self

            def dim(self):
                return len(self.shape)

            def is_contiguous(self):
                return True

        class Linear:
            def __init__(self, shape, row):
                self.weight, self.bias = (Tensor(*shape), None)
                self.tp_mode = SimpleNamespace(name="ROW" if row else "COLUMN")
                self.reduce_output, self.tp_size = (row, 8)
                self.apply_linear, self.forward = (Mock(), Mock())
                self.all_reduce = Mock(side_effect=lambda x, **_: x)

            def __call__(self, x):
                return Tensor(x.shape[0], self.weight.shape[0])

        shapes_tree = ast.parse((TARGET / "drafter_mxfp8.py").read_text())
        shapes = ast.literal_eval(
            next(
                (
                    n.value
                    for n in shapes_tree.body
                    if isinstance(n, ast.Assign)
                    and any((isinstance(t, ast.Name) and t.id == "HIS_SHAPES" for t in n.targets))
                )
            )
        )
        linears = {
            name: Linear(shape, name in ("o_proj", "down_proj")) for name, shape in shapes.items()
        }
        mlp = SimpleNamespace(
            down_proj=linears["down_proj"],
            gate_up_proj=linears["gate_up_proj"],
            forward=Mock(),
            split_gate_up=False,
        )
        mlp_stock = mlp.forward
        row_stock = linears["o_proj"].forward
        draft = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            qkv_proj=linears["qkv_proj"], o_proj=linears["o_proj"]
                        ),
                        mlp=mlp,
                    )
                ]
            )
        )
        gemv = Mock()
        gemv.frozen.return_value = False
        gemv._declines.return_value = False
        gemv.m3_mx_gemv_bf16.side_effect = lambda x, w, *_: Tensor(x.shape[0], w.shape[0])
        gemv.m3_mx_gemv_swiglu.side_effect = lambda x, w, *_: Tensor(x.shape[0], w.shape[0])
        path = FAMILY / "m31_drafter_layout.py"
        tree = ast.parse(path.read_text())
        tree.body = [
            n
            for n in tree.body
            if not isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.ImportFrom) and n.module == "__future__")
        ]
        ns = {
            "torch": SimpleNamespace(
                Tensor=Tensor, bfloat16="bf16", zeros=lambda m, k, **_: Tensor(m, k)
            ),
            "logger": Mock(),
        }
        exec(compile(tree, str(path), "exec"), ns)
        counts = ns["route_his_layout"](
            draft,
            [7, 9],
            gemv=gemv,
            recover=lambda w: (w, w),
            swizzle=lambda x: x,
            configs={v: (1, 1) for k, v in shapes.items() if k != "down_proj"},
            down_configs={shapes["down_proj"]: (1, 1)},
        )
        self.assertEqual(counts, (3, 1))
        params = object()
        linears["o_proj"].forward(Tensor(7, shapes["o_proj"][1]), all_reduce_params=params)
        self.assertIs(linears["o_proj"].all_reduce.call_args.kwargs["all_reduce_params"], params)
        mlp.forward(Tensor(7, shapes["gate_up_proj"][1]), final_all_reduce_params=params)
        self.assertIs(linears["down_proj"].all_reduce.call_args.kwargs["all_reduce_params"], params)
        linears["o_proj"].forward(Tensor(9, shapes["o_proj"][1]), all_reduce_params=params)
        mlp.forward(Tensor(9, shapes["gate_up_proj"][1]), final_all_reduce_params=params)
        mlp_stock.assert_called_once()
        row_stock.assert_called_once()
        self.assertEqual(
            ns["route_his_layout"](
                draft,
                [7],
                gemv=gemv,
                recover=lambda w: (w, w),
                swizzle=lambda x: x,
                configs={v: (1, 1) for k, v in shapes.items() if k != "down_proj"},
                down_configs={shapes["down_proj"]: (1, 1)},
            ),
            (0, 0),
        )


class TestTP8Package(unittest.TestCase):
    def test_relative_imports_resolve_to_existing_repo_modules(self):
        for path in (
            TARGET / "dspark.py",
            TARGET / "modeling.py",
            TARGET / "drafter_mxfp8.py",
            FAMILY / "m31_drafter_layout.py",
            FAMILY / "ops/m3_fsel_topk.py",
        ):
            package = ".".join(path.relative_to(ROOT).parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level:
                    resolved = importlib.util.resolve_name(
                        "." * node.level + (node.module or ""), package
                    )
                    destination = ROOT.joinpath(*resolved.split("."))
                    self.assertTrue(
                        destination.is_dir() or destination.with_suffix(".py").is_file(),
                        f"{path.name}:{node.lineno}: {resolved}",
                    )

    def test_shared_fsel_fake_matches_native_schema(self):
        source = (FAMILY / "csrc/m31/fselTopkOp.cpp").read_text()
        definition = re.search("m.def\\((.*?)\\);", source, re.S).group(1)
        schema = "".join(
            (ast.literal_eval(x) for x in re.findall('"(?:\\\\.|[^"\\\\])*"', definition))
        )
        params = schema.split("(", 1)[1].split(") ->", 1)[0].split(",")
        names = [param.split("=", 1)[0].split()[-1] for param in params]
        fake = FAMILY / "fake_ops.py"
        fn = next(
            (
                n
                for n in ast.parse(fake.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_fsel_topk_fake"
            )
        )
        self.assertEqual([arg.arg for arg in fn.args.args], names)


if __name__ == "__main__":
    unittest.main()
