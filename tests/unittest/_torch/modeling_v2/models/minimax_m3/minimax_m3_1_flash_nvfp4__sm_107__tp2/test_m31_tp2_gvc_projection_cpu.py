# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of DRAFTER_HISLAYOUT: projection-only TP2 drafter layout, kill switches, and the default fused router."""

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.cpu_only


def _functions(path, names, scope):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(body) == len(names)
    body.insert(
        0,
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        scope,
    )
    return scope


def test_projection_only_actual_dispatch_and_ring_keys():
    family = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    )

    class Tensor:
        dtype = "bf16"
        device = "cuda"

        def __init__(self, *shape):
            self.shape = shape
            self.data = self

        def dim(self):
            return len(self.shape)

        def is_contiguous(self):
            return True

    class Linear:
        def __init__(self, shape, row):
            self.weight = Tensor(*shape)
            self.bias = None
            self.tp_mode = SimpleNamespace(name="ROW" if row else "COLUMN")
            self.reduce_output = row
            self.tp_size = 2
            self.apply_linear = Mock()
            self.forward = Mock()
            self.all_reduce = Mock(side_effect=lambda x, **kw: x)

    shapes = {
        "qkv_proj": (5120, 6144),
        "o_proj": (6144, 4096),
        "gate_up_proj": (12288, 6144),
        "down_proj": (6144, 6144),
    }

    def draft():
        linears = {n: Linear(s, n in ("o_proj", "down_proj")) for n, s in shapes.items()}
        mlp = SimpleNamespace(
            gate_up_proj=linears["gate_up_proj"],
            down_proj=linears["down_proj"],
            forward=Mock(),
            split_gate_up=False,
        )
        d = SimpleNamespace(
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
        return d, linears, mlp

    gemv = Mock()
    gemv.MAX_TOKENS = 48
    gemv.get_config.return_value = (2, 12, 1, 1, 0, 1)
    gemv.frozen.return_value = True
    gemv._declines.return_value = False
    gemv.m3_mx_gemv_bf16.side_effect = lambda x, w, *a: Tensor(x.shape[0], w.shape[0])
    ns = {
        "torch": SimpleNamespace(
            Tensor=Tensor, bfloat16="bf16", zeros=lambda m, k, **kw: Tensor(m, k)
        ),
        "logger": Mock(),
        "_HIS_TAKEN": set(),
        "MAX_ROWS": 8,
    }
    router = _functions(family / "m31_drafter_layout.py", {"route_his_layout"}, ns)[
        "route_his_layout"
    ]
    target = family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/drafter_mxfp8.py"
    scope = {
        "_gemv": gemv,
        "_route_his_layout": router,
        "mxfp8_from_bf16": lambda w: (w, w),
        "swizzle_sf": lambda s: s,
        "W8_HIS_WIDE": True,
        "HIS_WIDE_MAX_ROWS": 28,
        "_his_quantize": Mock(),
        "logger": Mock(),
        "_HIS_WIDE_LOGGED": [],
    }
    route = _functions(target, {"route_projection_layout"}, scope)["route_projection_layout"]
    for _ in range(2):
        d, linears, mlp = draft()
        stock_mlp = mlp.forward
        stock_row = {n: linears[n].forward for n in ("o_proj", "down_proj")}
        # K7 at B = 1..8: rows 7..56; the GEMV row bound (48) keeps 7..42 (what MXDRAFT served), drops 49 / 56.
        assert route(d, [7 * b for b in range(1, 9)]) == (4, 0)
        assert mlp.forward is stock_mlp and not hasattr(mlp, "_m31_hislayout_mlp")
        gemv.m3_mx_gemv_swiglu.assert_not_called()
        for name, lin in linears.items():
            expected = (2, 12, 1, 1, 0, 1)
            assert lin._m31_hislayout == expected
        params = object()
        for name in ("o_proj", "down_proj"):
            lin = linears[name]
            for rows in (7, 14, 21, 28, 42):
                lin.forward(Tensor(rows, lin.weight.shape[1]), all_reduce_params=params)
                assert lin.all_reduce.call_args.kwargs["all_reduce_params"] is params
            lin.forward(Tensor(8, lin.weight.shape[1]), all_reduce_params=params)
            lin.forward(Tensor(7, lin.weight.shape[1]), lora_params={"lora": True})
            assert stock_row[name].call_count == 2
        assert route(d, [7]) == (0, 0)
    assert route(None, [7]) == (0, 0)
    p = family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark.py"
    tree = ast.parse(p.read_text())
    parent = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and any(
            isinstance(s, ast.Assign) and any(getattr(t, "id", None) == "gvc" for t in s.targets)
            for s in n.body
        )
    )
    start = next(
        i
        for i, n in enumerate(parent.body)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "gvc" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(parent.body)
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "mx" for t in n.targets)
    )
    for off, k, draft_model, expected in (
        (False, 7, object(), 1),
        (True, 7, object(), 0),
        (False, 0, object(), 0),
        (False, 7, None, 0),
    ):
        call = Mock(return_value=(20, 0))
        env = {
            "DRAFTER_HISLAYOUT": not off,
            "rank": 0,
            "k": k,
            "draft_model": draft_model,
            "MAX_FAST_BATCH": 8,
            "route_projection_layout": call,
            "logger": Mock(),
        }
        exec(
            compile(ast.Module(body=parent.body[start:end], type_ignores=[]), str(p), "exec"),
            env,
        )
        assert call.call_count == expected
    assert any(
        isinstance(n, ast.ImportFrom) and n.level == 2 and n.module == "m31_drafter_layout"
        for n in ast.parse(target.read_text()).body
    )


def test_w8_his_wide_quantize_rows():
    """W8 HIS_WIDE on the real decline rule (mx_gemv._declines, executed from source): the drafter shapes' rows the
    bf16-input GEMV declines (gate_up / down at 14, all four at 21 / 28) run quantize + the MXFP8-input GEMV; rows past
    HIS_WIDE_MAX_ROWS and the kill switch keep the stock path; o_proj / down keep their all-reduce params."""
    family = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    )
    target = family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/drafter_mxfp8.py"
    op_src = family / "minimax_m3_1_flash_nvfp4__sm_107__tp2/kernels/mx_gemv/op.py"
    tree = ast.parse(op_src.read_text())
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in ("_mma_n", "_smem_bytes", "_declines"))
        or (
            isinstance(n, ast.Assign)
            and any(
                getattr(t, "id", "") in ("MAX_TOKENS", "MMA_WIDTHS", "_SMEM_LIMIT")
                for t in n.targets
            )
        )
    ]
    real = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(op_src), "exec"), real)
    configs = {
        (5120, 6144): (4, 12),
        (6144, 4096): (2, 12),
        (12288, 6144): (2, 12),
        (6144, 6144): (2, 12),
    }

    class Tensor:
        dtype = "bf16"
        device = "cuda"

        def __init__(self, *shape):
            self.shape = shape
            self.data = self

        def dim(self):
            return len(self.shape)

        def is_contiguous(self):
            return True

    class Linear:
        def __init__(self, shape, row):
            self.weight = Tensor(*shape)
            self.bias = None
            self.tp_mode = SimpleNamespace(name="ROW" if row else "COLUMN")
            self.reduce_output = row
            self.tp_size = 2
            self.apply_linear = Mock()
            self.forward = Mock()
            self.all_reduce = Mock(side_effect=lambda x, **kw: x)

    shapes = {
        "qkv_proj": (5120, 6144),
        "o_proj": (6144, 4096),
        "gate_up_proj": (12288, 6144),
        "down_proj": (6144, 6144),
    }
    expect_bf16 = {"qkv_proj": (7, 14), "o_proj": (7, 14), "gate_up_proj": (7,), "down_proj": (7,)}
    expect_q = {
        "qkv_proj": (21, 28),
        "o_proj": (21, 28),
        "gate_up_proj": (14, 21, 28),
        "down_proj": (14, 21, 28),
    }
    for wide_on in (True, False):
        gemv = Mock()
        gemv.MAX_TOKENS = 48
        gemv.get_config.side_effect = lambda n, k: configs[(n, k)]
        gemv.frozen.return_value = True
        gemv._declines.side_effect = real["_declines"]
        gemv.m3_mx_gemv_bf16.side_effect = lambda x, w, *a: Tensor(x.shape[0], w.shape[0])
        gemv.m3_mx_gemv.side_effect = lambda xq, xsf, w, *a: Tensor(xq.shape[0], w.shape[0])
        quantize = Mock(side_effect=lambda x: (Tensor(*x.shape), Tensor(128, x.shape[1] // 32)))
        ns = {
            "torch": SimpleNamespace(
                Tensor=Tensor, bfloat16="bf16", zeros=lambda m, k, **kw: Tensor(m, k)
            ),
            "logger": Mock(),
            "_HIS_TAKEN": set(),
            "MAX_ROWS": 8,
        }
        router = _functions(family / "m31_drafter_layout.py", {"route_his_layout"}, ns)[
            "route_his_layout"
        ]
        log = Mock()
        scope = {
            "_gemv": gemv,
            "_route_his_layout": router,
            "mxfp8_from_bf16": lambda w: (w, w),
            "swizzle_sf": lambda s: s,
            "W8_HIS_WIDE": wide_on,
            "HIS_WIDE_MAX_ROWS": 28,
            "_his_quantize": quantize,
            "logger": log,
            "_HIS_WIDE_LOGGED": [],
        }
        route = _functions(target, {"route_projection_layout"}, scope)["route_projection_layout"]
        linears = {n: Linear(s, n in ("o_proj", "down_proj")) for n, s in shapes.items()}
        stock_apply = {n: lin.apply_linear for n, lin in linears.items()}
        stock_fwd = {n: lin.forward for n, lin in linears.items()}
        d = SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            qkv_proj=linears["qkv_proj"], o_proj=linears["o_proj"]
                        ),
                        mlp=SimpleNamespace(
                            gate_up_proj=linears["gate_up_proj"],
                            down_proj=linears["down_proj"],
                            forward=Mock(),
                            split_gate_up=False,
                        ),
                    )
                ]
            )
        )
        assert route(d, [7 * b for b in range(1, 13)]) == (4, 0)
        line = log.info.call_args.args[0]
        assert line.startswith(
            "[TP2WIDE] W8 HIS rows=" + ("14|21|28 on" if wide_on else "none off")
        ), line
        for name, lin in linears.items():
            assert lin._m31_hislayout_mxq == (expect_q[name] if wide_on else ())
            row = name in ("o_proj", "down_proj")
            params = object()
            for rows in (7, 14, 21, 28, 35):
                gemv.m3_mx_gemv_bf16.reset_mock()
                gemv.m3_mx_gemv.reset_mock()
                quantize.reset_mock()
                x = Tensor(rows, lin.weight.shape[1])
                if row:
                    stock_fwd[name].reset_mock()
                    lin.forward(x, all_reduce_params=params)
                    stock_calls = stock_fwd[name].call_count
                else:
                    stock_apply[name].reset_mock()
                    lin.apply_linear(x, None)
                    stock_calls = stock_apply[name].call_count
                if rows in expect_bf16[name]:
                    assert gemv.m3_mx_gemv_bf16.call_count == 1 and stock_calls == 0
                    assert quantize.call_count == 0
                elif wide_on and rows in expect_q[name]:
                    assert quantize.call_count == 1 and gemv.m3_mx_gemv.call_count == 1
                    assert gemv.m3_mx_gemv.call_args.args[4:] == (True, configs[lin.weight.shape])
                    assert gemv.m3_mx_gemv_bf16.call_count == 0 and stock_calls == 0
                    if row:
                        assert lin.all_reduce.call_args.kwargs["all_reduce_params"] is params
                else:
                    assert stock_calls == 1 and quantize.call_count == 0
                    assert gemv.m3_mx_gemv.call_count == 0 and gemv.m3_mx_gemv_bf16.call_count == 0


def test_w8_kill_switch_reads_its_switch():
    p = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
        / "minimax_m3_1_flash_nvfp4__sm_107__tp2/drafter_mxfp8.py"
    )
    tree = ast.parse(p.read_text())
    values = {
        n.targets[0].id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    assert ast.unparse(values["W8_HIS_WIDE"]) == "not m31_disabled('DRAFTER_HISLAYOUT_WIDE')"
    assert ast.literal_eval(values["HIS_WIDE_MAX_ROWS"]) == 28
    assert ast.literal_eval(values["HIS_QUANT_TACTIC"]) == 0


def test_default_router_unchanged_for_peers():
    # TP4 / TP8 call the family router without fuse_mlp / max_rows: fused SwiGLU + down on, rows <= 8.
    family = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
    )
    fn = next(
        n
        for n in ast.parse((family / "m31_drafter_layout.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "route_his_layout"
    )
    kw = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert ast.literal_eval(kw["fuse_mlp"]) is True
    assert (
        ast.literal_eval(kw["quantize"]) is None and ast.literal_eval(kw["quantize_max_rows"]) == 0
    )
    assert isinstance(kw["max_rows"], ast.Name) and kw["max_rows"].id == "MAX_ROWS"
    src = (family / "m31_drafter_layout.py").read_text()
    assert "MAX_ROWS = 8" in src and "<= max_rows" in src


def test_kill_switches_read_at_import():
    # The kill switch turns DRAFTER_HISLAYOUT off; the default is on. Evaluated on the module's own expression.
    p = (
        Path(__file__).resolve().parents[7]
        / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3"
        / "minimax_m3_1_flash_nvfp4__sm_107__tp2/dspark.py"
    )
    node = next(
        n
        for n in ast.parse(p.read_text()).body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "DRAFTER_HISLAYOUT" for t in n.targets)
    )
    expr = ast.unparse(node.value)
    prog = (
        "import os\n"
        "_D = frozenset(k[len('TRTLLM_M31_DISABLE_'):] for k, v in os.environ.items()"
        " if k.startswith('TRTLLM_M31_DISABLE_') and v.strip().lower() in ('1', 'true', 'on', 'yes'))\n"
        "m31_disabled = _D.__contains__\n"
        f"print({expr})\n"
    )
    base = {k: v for k, v in os.environ.items() if "TRTLLM_M31" not in k}
    for extra, expected in (
        ({}, "True"),
        ({"TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT": "1"}, "False"),
        ({"TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT": "0"}, "True"),
        ({"TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT": "false"}, "True"),
    ):
        out = subprocess.run(
            [sys.executable, "-c", prog],
            env={**base, **extra},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out == expected, (extra, out)


if __name__ == "__main__":
    test_projection_only_actual_dispatch_and_ring_keys()
    test_w8_his_wide_quantize_rows()
    test_w8_kill_switch_reads_its_switch()
    test_default_router_unchanged_for_peers()
    test_kill_switches_read_at_import()
    print("TP2_GVC_PROJECTION_CPU_PASS")
