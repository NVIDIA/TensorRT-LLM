# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's ``moe_latent_up_resident`` lever: latent_up behind the latent all-reduce row kernel as
the resident GEMM (``rubin_latent_up_resident``), which loads its weight before its grid wait.

CPU (``cpu_only``): helix8's MoE block runs from its modeling.py over recording stand-ins, with
the column-parallel tail (``moe_cp_tail``) off (test_rubin_helix8_moe_cp_tail.py checks it and the
row bands' single owner):

* the constants: ``moe_latent_up_resident`` (its row bands ``moe_latent_up_resident_bands``,
  31-48, a detail logged inside it) is a generation launch constant, read into
  ``_MOE_LATENT_UP_RESIDENT_BANDS`` (none when off); the startup line states it; no environment
  variable;
* on, a CUDA graph step in its bands on the own workspace (every rank's rows in its one-shot
  buffer: up to 109 rows at 8 ranks) runs the row kernel, then the resident GEMM on its output and
  this layer's ``latent_up``, then the wait on the shared join; it logs the README's line once;
* rows outside the bands (both sides, inclusive bounds), past the own workspace's one-shot bytes
  (110 and more at 8 ranks) and the two-shot all-reduce's rows (144), an eager step and the lever
  off keep cuBLAS's latent_up and log nothing.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
import re
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, NamedTuple

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_README = _FAMILY / "README.md"
_OP = _FAMILY / "ops/gemm/rubin_latent_up_resident.py"

_HIDDEN, _LATENT = 7168, 3584
#: The lever's row bands as shipped, and the graph rows they serve (b6, b7, b8).
BANDS = "31-48"
ON_ROWS = (36, 42, 48)
_BLOCK = (
    "_TARGET",
    "_DETAILS",
    "constants_line",
    "_MoeOut",
    "_moe_cublas_linear",
    "_moe_latent_gemm",
    "_moe_in_bands",
    "_moe_cp_tail_serves",
    "_moe_latent_up",
    "_moe_one_shot",
    "_moe_tp_gen",
    "_moe_tp_gen_streams",
    "latent_moe",
)


def _tree(text: str | None = None) -> ast.Module:
    return ast.parse(text if text is not None else _MODELING.read_text(), filename=str(_MODELING))


def _switched(**values) -> ast.Module:
    """modeling.py with generation constants set to ``values`` (each key must appear once), the
    column-parallel tail off unless set."""
    text = _MODELING.read_text()
    for name, value in {"moe_cp_tail": False, **values}.items():
        rendered = f'"{value}"' if isinstance(value, str) else repr(value)
        # The generation table's line (16 spaces in), not a _DETAILS entry of the same key.
        text, hits = re.subn(rf'(?m)^( {{16}}"{name}": )[^,\n]+,$', rf"\g<1>{rendered},", text)
        assert hits == 1, (name, hits)
    return _tree(text)


def _constants(tree: ast.Module) -> dict:
    """``tree``'s module-level constants that evaluate on their own, in source order (the row-band
    parser defined first, so the bands evaluate with a lever on)."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    (parser,) = [n for n in tree.body if getattr(n, "name", None) == "_moe_row_bands"]
    exec(compile(ast.fix_missing_locations(ast.Module([parser], [])), str(_MODELING), "exec"), ns)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
        if not names or not isinstance(target, (ast.Name, ast.Tuple)):
            continue
        # A value that reads an import or a later definition is not a constant the block reads.
        try:
            value = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
        if isinstance(target, ast.Name):
            ns[target.id] = value
        else:
            ns.update(zip(names, value))
    return ns


def _definitions(tree: ast.Module, names) -> list:
    wanted, nodes, seen = set(names), [], {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if defined & wanted:
            nodes.append(copy.deepcopy(node))
            for name in defined & wanted:
                seen[name] = seen.get(name, 0) + 1
    assert seen == dict.fromkeys(wanted, 1), seen
    return nodes


class _Calls:
    """The block's kernels, event operations and log lines, in order."""

    def __init__(self):
        self.ops: list = []
        self.once: list = []


def _block(tree: ast.Module, calls: _Calls) -> dict:
    """The MoE block of ``tree`` over its constants and recording stand-ins."""

    def cublas_mm(a, b):
        calls.ops.append(
            ("latent_down" if b.shape[1] == _LATENT else "cublas_latent_up", a.shape[0])
        )
        return torch.empty(a.shape[0], b.shape[1], dtype=a.dtype)

    def rubin_latent_up_resident(x, weight):
        calls.ops.append(("resident_latent_up", x.shape[0], weight))
        return torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype)

    def rubin_latent_ar_norm_row(x, weight, eps, out, **kw):
        calls.ops.append(("latent_ar_row", x.shape[0], out, kw["use_pdl"], kw["early_trigger"]))

    def allreduce_module(x, all_reduce, fusion_op, norm_weight, eps):
        calls.ops.append((f"allreduce{fusion_op}", x.shape[0]))
        return torch.empty_like(x)

    def _moe_front(name):
        def run(*args, **kwargs):
            calls.ops.append((name,))
            return torch.empty(1)

        return run

    def _moe_route(L, logits):
        calls.ops.append(("route",))
        return None, None, None

    def _moe_quantize(L, latent_in):
        calls.ops.append(("quantize",))
        return None, None

    def _moe_fc12(L, x4, sf, ids, weights, tables, zeroed):
        calls.ops.append(("fc12",))
        return torch.empty(x_rows[0], _LATENT, dtype=torch.bfloat16)

    def _moe_shared(L, x):
        calls.ops.append(("shared",))
        return torch.empty(x.shape[0], _HIDDEN, dtype=torch.bfloat16)

    class _Event:
        def __init__(self, name):
            self.name = name

        def record(self):
            calls.ops.append(("record", self.name))

        def wait(self):
            calls.ops.append(("wait", self.name))

    class _Logger:
        def info_once(self, *msg, key):
            calls.once.append((" ".join(map(str, msg)), key))

    class _On:
        def __init__(self, stream):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    x_rows = [0]
    module = ast.fix_missing_locations(ast.Module(body=_definitions(tree, _BLOCK), type_ignores=[]))
    ns = {
        **_constants(tree),
        "NamedTuple": NamedTuple,
        "Any": Any,
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(stream=_On), empty_like=torch.empty_like, empty=torch.empty
        ),
        "logger": _Logger(),
        "cublas_mm": cublas_mm,
        "rubin_latent_up_resident": rubin_latent_up_resident,
        "rubin_latent_ar_norm_row": rubin_latent_ar_norm_row,
        "allreduce_module": allreduce_module,
        "_moe_router_gen": _moe_front("router"),
        "_moe_route": _moe_route,
        "_moe_quantize": _moe_quantize,
        "_moe_fc12": _moe_fc12,
        "_moe_fc12_zeroed": lambda rows, *a: (torch.empty(1),),
        "_moe_shared": _moe_shared,
        # The GEMM pins run today's call here (test_rubin_helix8_gemm_pins.py checks them).
        "_gemm_pinned": lambda site, a, b, today, out=None, out_dtype=None: today(),
    }
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    events = ("latent_fork", "latent_ready", "prezero_ready", "shared_fork", "shared_join")
    ns["_rt"] = SimpleNamespace(
        aux_stream="aux",
        moe_comm=SimpleNamespace(
            all_reduce="tp",
            latent_stream="latent",
            prezero_stream="prezero",
            latent_ws=dict(
                world_size=8,
                max_tokens=128,
                buffer_bytes=1 << 30,
                buffer=torch.empty(1),
                mc=torch.empty(1),
                flags=torch.empty(1),
                rank=0,
            ),
            **{e: _Event(e) for e in events},
        ),
    )
    ns["_x_rows"] = x_rows
    return ns


def _shape_only(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    """A tensor of ``shape`` over one row of memory: the stand-ins read shapes only."""
    return torch.empty(shape[-1], dtype=dtype).expand(*shape)


def _layer() -> SimpleNamespace:
    return SimpleNamespace(
        latent_up=_shape_only(_HIDDEN, _LATENT),
        latent_down=_shape_only(_LATENT, _HIDDEN),
        latent_norm=torch.empty(_LATENT, dtype=torch.bfloat16),
        fc1_w=torch.empty(1, 1),
        fc2_w=torch.empty(1, 1),
    )


def _step(tree: ast.Module, rows: int, cuda_graph: bool = True):
    """``(calls, out, layer, block)`` of one ``latent_moe`` call of ``tree``."""
    calls = _Calls()
    ns = _block(tree, calls)
    ns["_x_rows"][0] = rows
    layer = _layer()
    out = ns["latent_moe"](
        layer,
        5,
        torch.empty(rows, _HIDDEN, dtype=torch.bfloat16),
        SimpleNamespace(cuda_graph=cuda_graph),
        ns["_rt"],
    )
    assert tuple(out.routed.shape) == (rows, _HIDDEN)
    return calls, out, layer, ns


def _back(calls: _Calls) -> list:
    """The ops from the latent all-reduce on, by name."""
    names = [op[0] if op[0] not in ("record", "wait") else f"{op[0]}:{op[1]}" for op in calls.ops]
    first = next(i for i, n in enumerate(names) if n.startswith(("latent_ar_row", "allreduce9")))
    return names[first:]


def _back_lines(calls: _Calls) -> list:
    """The MoE back's log lines (the front logs its own)."""
    return [c for c in calls.once if c[0].startswith("Kimi K3.1 Rubin MoE back")]


def _on(bands: str = BANDS) -> ast.Module:
    return _switched(moe_latent_up_resident=True, moe_latent_up_resident_bands=bands)


def test_the_lever_is_a_logged_generation_constant():
    tree = _tree()
    ns = _constants(tree)
    gen = ns["_CONSTANTS"]["generation"]
    assert gen["moe_latent_up_resident"] is False
    assert gen["moe_latent_up_resident_bands"] == BANDS
    assert ns["_DETAILS"]["moe_latent_up_resident_bands"] == "moe_latent_up_resident"
    assert ns["_MOE_LATENT_UP_RESIDENT_BANDS"] == ()
    assert _constants(_on())["_MOE_LATENT_UP_RESIDENT_BANDS"] == ((31, 48),)
    assert _constants(_on("6-6,36-60"))["_MOE_LATENT_UP_RESIDENT_BANDS"] == ((6, 6), (36, 60))
    line = _block(tree, _Calls())["constants_line"]("generation")
    assert f"moe_latent_up_resident=False(moe_latent_up_resident_bands={BANDS})" in line
    # The bands never pass the own workspace's rows: the resident GEMM serves that branch only.
    on = _constants(_on())["_MOE_LATENT_UP_RESIDENT_BANDS"]
    assert max(hi for _, hi in on) <= gen["latent_ar_own_ws_max_m"]
    source = _MODELING.read_text()
    assert "os.environ" not in ast.unparse(_definitions(tree, ("_moe_latent_up",))[0])
    assert source.count("rubin_latent_up_resident(") == 1


# Up to 109 rows every rank's rows fit the own workspace's one-shot buffer (6 MiB at 8 ranks).
@pytest.mark.parametrize("rows", ON_ROWS)
def test_on_a_graph_step_runs_the_resident_gemm_behind_the_row_kernel(rows):
    calls, out, layer, ns = _step(_on(), rows)
    assert _back(calls) == ["latent_ar_row", "resident_latent_up", "wait:shared_join"]
    (ar,) = [op for op in calls.ops if op[0] == "latent_ar_row"]
    (up,) = [op for op in calls.ops if op[0] == "resident_latent_up"]
    assert ar[3:] == (True, True), "the row kernel keeps its PDL launch and early trigger"
    assert up[1] == rows and up[2] is layer.latent_up
    line = ns["_MOE_LATENT_UP_LINE"]
    assert _back_lines(calls) == [(line, line)]
    assert line.startswith("Kimi K3.1 Rubin MoE back: ") and "\n" not in line
    assert line.endswith("(resident, split)")
    assert f"\n{line}\n" in _README.read_text()


@pytest.mark.parametrize(
    "tree, rows, cuda_graph",
    [
        (None, 42, True),
        ("on", 30, True),
        ("on", 54, True),
        ("on", 66, True),
        ("on", 42, False),
        ("wide", 120, True),
        ("wide", 144, True),
        ("other", 42, True),
    ],
    ids=[
        "off",
        "below_the_bands",
        "above_the_bands",
        "b11",
        "eager",
        "past_one_shot",
        "two_shot_rows",
        "outside_other_bands",
    ],
)
def test_elsewhere_latent_up_stays_cublas(tree, rows, cuda_graph):
    tree = {
        None: _switched(moe_latent_up_resident=False),
        "on": _on(),
        "wide": _on("1-128"),
        "other": _on("1-30,49-60"),
    }[tree]
    calls, _, _, _ = _step(tree, rows, cuda_graph)
    names = [op[0] for op in calls.ops]
    assert "resident_latent_up" not in names and names.count("cublas_latent_up") == 1
    assert _back_lines(calls) == []


@pytest.mark.parametrize("rows", [31, 48])
def test_the_bands_are_inclusive(rows):
    calls, _, _, _ = _step(_on(), rows)
    assert [op[0] for op in calls.ops].count("resident_latent_up") == 1


def test_the_entry_states_the_kernels_bounds():
    """The entry's row bound, split and default budget are the kernel module's (read as text: the
    kernel module needs the GPU stack to import)."""
    kernel = (_FAMILY / "kernels/gemm/rubin_latent_up_resident.py").read_text()
    op = _OP.read_text()
    for name, value in (("SPLIT_K", "3"), ("MAX_M", "128"), ("DEFAULT_SMEM_BUDGET", "320 * 1024")):
        assert f"\n{name} = {value}\n" in kernel, name
        assert f"\n{name} = {value}\n" in op, name
