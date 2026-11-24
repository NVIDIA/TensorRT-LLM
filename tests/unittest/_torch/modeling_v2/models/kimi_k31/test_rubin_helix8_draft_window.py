# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's drafter window levers: logged generation constants, on, and handed to the window entry.

``draft_window_vec_io`` (the window kernels' 16-byte vector loads and stores) and
``draft_window_pdl`` (the window's tail as a PDL dependent of its body) are generation constants of
the helix8 target, on by default (off: today's window), stated in the startup constants
line and read once into ``_DRAFT_WINDOW_VEC_IO`` and ``_DRAFT_WINDOW_PDL``. ``draft_step`` (and the
lever path ``_draft_step_fused``) runs every drafter layer's window through ``_draft_window``, which
hands both to its ``rubin_dflash2_window_triton`` call. Under ``draft_window_vec_io``
the first drafter step of a process logs the README's INFO line with the number of layers whose
window kernels both ran ``ALIGNED`` (``window_vec_io_aligned``): the evidence a generation run
reads. The default target is untouched.

CPU (``cpu_only``), from the targets' modeling.py: the constants and the call by ``ast``, and
helix8's ``draft_step`` run over recording stand-ins, with the module's own definitions of the
other module-level functions it names (nothing is imported from the target).
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_DEFAULT = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__default" / "modeling.py"
_README = _FAMILY / "README.md"
_ENTRY = "rubin_dflash2_window_triton"
#: Each lever: (generation constant, module alias).
_LEVERS = {
    "vec_io": ("draft_window_vec_io", "_DRAFT_WINDOW_VEC_IO"),
    "pdl": ("draft_window_pdl", "_DRAFT_WINDOW_PDL"),
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _constants(tree: ast.Module) -> dict:
    """``tree``'s module-level constants that evaluate on their own, in source order."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            ns[target.id] = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
    return ns


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


def _exec(ns: dict, tree: ast.Module, *names: str) -> dict:
    body = [copy.deepcopy(_function(tree, n)) for n in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _helpers(tree: ast.Module, name: str, ns: dict) -> list[str]:
    """The module-level functions that ``name`` refers to and ``ns`` has no stand-in for, in
    source order: ``_exec`` runs them as the module defines them."""
    used = {n.id for n in ast.walk(_function(tree, name)) if isinstance(n, ast.Name)}
    wanted = used - ns.keys() - {name}
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]


def _window_calls(fn: ast.AST) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == _ENTRY
    ]


def test_the_levers_are_logged_generation_constants_and_on():
    tree = _tree(_MODELING)
    ns = _constants(tree)
    generation = ns["_CONSTANTS"]["generation"]
    ns.update(_exec(ns, tree, "constants_line"))
    line = ns["constants_line"]("generation")
    for key, alias in _LEVERS.values():
        assert generation[key] is True, key
        assert ns[alias] is True, alias
        assert f"{key}=True" in line, key
        (assign,) = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign) and [getattr(t, "id", None) for t in n.targets] == [alias]
        ]
        assert ast.unparse(assign.value) == f"_CONSTANTS['generation']['{key}']"


def test_every_drafter_layer_hands_the_levers_to_the_window_entry():
    tree = _tree(_MODELING)
    (call,) = _window_calls(_function(tree, "_draft_window"))
    passed = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert passed == {lever: alias for lever, (_, alias) in _LEVERS.items()}
    # Both drafter steps run their window through _draft_window and call no window entry directly.
    for step in ("draft_step", "_draft_step_fused"):
        fn = _function(tree, step)
        assert not _window_calls(fn), step
        called = [
            n.func.id
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        assert called.count("_draft_window") == 1, step


def test_the_default_target_keeps_todays_window():
    """The default target binds the same entry with neither lever and logs no such constant."""
    tree = _tree(_DEFAULT)
    calls = _window_calls(tree)
    assert calls and all(not c.keywords for c in calls)
    text = _DEFAULT.read_text()
    for key, alias in _LEVERS.values():
        assert key not in text and alias not in text


# draft_step over recording stand-ins

_ROWS, _BLOCK, _HEADS, _NOPE, _KV_LORA, _ROPE, _V, _HIDDEN, _QL, _I, _TAPS = (
    4,
    2,
    2,
    3,
    4,
    2,
    3,
    5,
    3,
    2,
    2,
)
_LAYERS = 5


def _layer() -> SimpleNamespace:
    z = torch.zeros
    return SimpleNamespace(
        in_norm=z(_HIDDEN),
        post_norm=z(_HIDDEN),
        attn_conv_proj=z(2 * _TAPS, _HIDDEN),
        attn_conv_base=z(2, _TAPS, _HIDDEN),
        q_a=z(_QL, _HIDDEN),
        q_a_norm=z(_QL),
        q_b=z(_HEADS * (_NOPE + _ROPE), _QL),
        k_b=z(_HEADS, _NOPE, _KV_LORA),
        kv_a=z(_KV_LORA + _ROPE, _HIDDEN),
        kv_a_norm=z(_KV_LORA),
        v_b_t=z(_HEADS, _KV_LORA, _V),
        o=z(_HIDDEN, _HEADS * _V),
        mlp_conv_proj=z(2 * _TAPS, _HIDDEN),
        mlp_conv_base=z(2, _TAPS, _HIDDEN),
        gate_up=z(2 * _I, _HIDDEN),
        down=z(_HIDDEN, _I),
        q_a_kv_a=None,
    )


def _draft(levers: dict, aligned: list):
    """helix8's ``draft_step`` with the lever aliases set to ``levers``: what each layer's window
    call received, what was asked of ``window_vec_io_aligned``, and the INFO lines."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    for lever, (_, alias) in _LEVERS.items():
        ns[alias] = levers.get(lever, False)
    # draft_step's own path (the drafter levers that hand the step to _draft_step_fused are off;
    # that path hands the window levers to the entry the same way).
    ns.update(_DRAFT_LOWM_GEMM=False, _DRAFT_CONV_NORM=False, _DRAFT_WINDOW_MLA=False)
    windows, asked, once, masks = [], [], [], []
    latent = _KV_LORA + _ROPE

    def window(*args, **kwargs):
        windows.append(kwargs)
        return torch.zeros(_ROWS, _HEADS, _KV_LORA)

    def vec_io_aligned(query, pool, block_kv, workspace, latent_dim):
        asked.append((query.shape, pool.shape, block_kv.shape, workspace, latent_dim))
        return aligned[len(asked) - 1]

    ns.update(
        _DRAFT_WINDOW=1024,
        _AR_NONE=0,
        rubin_dflash2_window_plan=lambda *a: "plan",
        flashinfer_rmsnorm=lambda x, w, eps: x,
        flashinfer_fused_add_rmsnorm=lambda a, residual, w, eps: (a, residual),
        rubin_linear=lambda x, w, tf32: torch.zeros(x.shape[0], w.shape[0]),
        dflash2_grouped_conv=lambda x, coef, base, block, group, pdl=False: x,
        bmm_out=lambda a, b, out: out,
        dflash2_mla_block_rope=lambda *a: None,
        rubin_dflash2_window_triton=window,
        window_vec_io_aligned=vec_io_aligned,
        rubin_dflash2_helix_combine=lambda p, s, m, h, v, zero_kv_mask=None: (
            masks.append(zero_kv_mask) or torch.zeros(_ROWS, h * v)
        ),
        allreduce_module=lambda x, *a: x,
        flashinfer_silu_and_mul=lambda x: x[:, : x.shape[1] // 2],
        logger=SimpleNamespace(info_once=lambda msg, key=None: once.append((msg, key))),
    )
    _exec(ns, tree, "draft_step", *_helpers(tree, "draft_step", ns))
    DW = SimpleNamespace(layers=[_layer() for _ in range(_LAYERS)], final_norm=torch.zeros(_HIDDEN))
    st = SimpleNamespace(
        rows=_ROWS,
        block=_BLOCK,
        num_ctx=torch.tensor([3, 70]),
        page_table=torch.zeros(2, 4, dtype=torch.int32),
        table_rows=torch.arange(2, dtype=torch.int32),
        page_size=64,
        positions=torch.arange(_ROWS),
        pools=[torch.zeros(3, 1, 1, 64, latent) for _ in range(_LAYERS)],
    )
    workspace = SimpleNamespace(partial_out=None, partial_stats=None)
    d = SimpleNamespace(
        eps=1e-5,
        taps=_TAPS,
        groups=1,
        group=_HIDDEN,
        cp_size=8,
        cp_rank=3,
        plan="buffers",
        block_kv=torch.zeros(8, 1, latent),
        stats=torch.zeros(8, _HEADS, 2),
        softmax_scale=0.13,
        kv_scale=1.0,
        workspace=workspace,
        freqs=None,
        block_slots=torch.arange(8),
        block_offsets=torch.zeros(8, dtype=torch.int64),
        mapping=None,
    )
    rt = SimpleNamespace(draft=d, all_reduce=None)
    h = ns["draft_step"](DW, torch.zeros(_ROWS, _HIDDEN), st, rt)
    assert h.shape == (_ROWS, _HIDDEN)
    # The Triton window writes its empty rows sanitized: the combine gets no mask.
    assert masks == [None] * _LAYERS
    return ns, windows, asked, once, workspace


_OFF = {lever: False for lever in _LEVERS}


def test_draft_step_off_runs_todays_window():
    ns, windows, asked, once, _ = _draft({}, [])
    assert windows == [_OFF] * _LAYERS
    assert asked == [] and once == []


def test_draft_step_hands_pdl_to_every_layer():
    ns, windows, asked, once, _ = _draft({"pdl": True}, [])
    assert windows == [dict(_OFF, pdl=True)] * _LAYERS
    assert asked == [] and once == []


@pytest.mark.parametrize(
    "aligned", [[True] * _LAYERS, [True, True, False, True, True]], ids=["all", "one_refused"]
)
def test_draft_step_logs_on_how_many_layers_the_window_ran_aligned(aligned):
    ns, windows, asked, once, workspace = _draft({"vec_io": True}, aligned)
    assert windows == [dict(_OFF, vec_io=True)] * _LAYERS
    # Each layer asks about its own query, its own pool layer's [pages, P, D] view, the block
    # rows' [R, D] view and the drafter's workspace, at the entry's latent width.
    shapes = (torch.Size([_ROWS, _HEADS, _KV_LORA + _ROPE]), torch.Size([3, 64, _KV_LORA + _ROPE]))
    assert asked == [(*shapes, torch.Size([_ROWS, _KV_LORA + _ROPE]), workspace, _KV_LORA)] * 5
    line = ns["_DRAFT_WINDOW_VEC_LINE"]
    assert once == [(line.format(sum(aligned), _LAYERS), line)]


def test_the_info_line_is_the_readmes():
    line = _constants(_tree(_MODELING))["_DRAFT_WINDOW_VEC_LINE"].format(_LAYERS, _LAYERS)
    assert line.startswith("Kimi K3.1 Rubin DFlash2 window: ALIGNED=True on 5 of 5 drafter layers")
    assert f"\n{line}\n" in _README.read_text()
