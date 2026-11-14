# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4_FSEL CPU checks: no torch, no GPU. The fsel kernel and its op are in the family library (csrc/m31), the op has a
fake, the family op calls the op the C++ file registers, the TP4 modeling's relative imports name modules of this tree,
and the TP4 wiring carries the kill switch, the row bound and the engagement lines (read off the sources)."""

import ast
import importlib.util
import os
import re

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))
FAM = os.path.join(ROOT, "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3")
TP4 = os.path.join(FAM, "minimax_m3_1_flash_nvfp4__sm_107__tp4")
OPS = os.path.join(FAM, "ops")
CSRC = os.path.join(FAM, "csrc")
KERNEL = os.path.join(CSRC, "m31", "fselTopkSm107.cu")
HEADER = os.path.join(CSRC, "m31", "fselTopkSm107.h")
OP = os.path.join(CSRC, "m31", "fselTopkOp.cpp")
FAKES = os.path.join(FAM, "fake_ops.py")
OP_NAME = "minimax_m3_sm107tp4_fsel_topk"


def _src(p):
    with open(p) as f:
        return f.read()


def _relative_imports(path, pkg):
    for n in ast.walk(ast.parse(_src(path))):
        if isinstance(n, ast.ImportFrom) and n.level:
            yield n.lineno, "." * n.level + (n.module or ""), pkg, [a.name for a in n.names]


def test_kernel_caps_the_include_masks_at_32_keys():
    """The per-thread include masks are 32-bit (one bit per key): the body asserts it and the 256-thread instance
    takes at most 8192 pages; wider calls run the 512-thread instance."""
    src = _src(KERNEL)
    assert "static_assert(KPT >= 1 && KPT <= 32" in src
    assert "(variant == 6 || num_cols > 256 * 32) ? 512 : 256" in src


def test_kernel_and_op_are_in_the_family_library():
    assert re.search(
        r"add_modeling_v2_shared\([^)]*SOURCES[^)]*m31/\*\.cu[^)]*m31/\*\.cpp",
        _src(os.path.join(CSRC, "CMakeLists.txt")),
        re.S,
    )
    assert os.path.isfile(HEADER) and os.path.isfile(OP)


def test_op_name_agrees_across_the_cpp_op_the_fake_and_the_family_op():
    op = _src(OP)
    assert (
        f'"{OP_NAME}(Tensor scores, Tensor n_valid_blocks, Tensor page_table, Tensor(a!) out, int "'
        in op
    )
    assert f'm.impl("{OP_NAME}", &TRTLLM_MV2_NS::torch_ext::minimaxM3Sm107Tp4FselTopk);' in op
    assert '#include "minimax_m3/csrc/m31/fselTopkSm107.h"' in op
    fakes = _src(FAKES)
    assert f'torch.library.register_fake(f"{{_NS}}::{OP_NAME}")(_fsel_topk_fake)' in fakes
    fam = _src(os.path.join(OPS, "m3_fsel_topk.py"))
    assert f"family_ops().{OP_NAME}(" in fam


def test_fake_signature_matches_the_schema():
    """register_fake checks the fake against the schema; the parameter names must be the schema's, in order."""
    schema = re.search(
        rf'"{OP_NAME}\((.*?)\) -> \(\)"', _src(OP).replace('"\n        "', ""), re.S
    ).group(1)
    names = [a.strip().split()[-1].split("=")[0] for a in schema.split(",")]
    fn = next(
        n
        for n in ast.parse(_src(FAKES)).body
        if isinstance(n, ast.FunctionDef) and n.name == "_fsel_topk_fake"
    )
    assert [a.arg for a in fn.args.args] == names


def test_every_relative_import_of_the_wiring_resolves_in_the_tree():
    """Lesson: a relative import must name a module or package of this tree, and each imported name a module
    of that package or a name its __init__ defines."""
    broken, n = [], 0
    for p in (os.path.join(TP4, "modeling.py"), os.path.join(OPS, "m3_fsel_topk.py")):
        mod = os.path.relpath(p, ROOT)[:-3].replace(os.sep, ".")
        pkg = mod.rsplit(".", 1)[0]
        for line, spec, pkg_, names in _relative_imports(p, pkg):
            n += 1
            tgt = importlib.util.resolve_name(spec, pkg_)
            base = os.path.join(ROOT, *tgt.split("."))
            if not (
                os.path.isfile(base + ".py") or os.path.isfile(os.path.join(base, "__init__.py"))
            ):
                broken.append(f"{os.path.relpath(p, ROOT)}:{line} {spec} -> {tgt}")
                continue
            if os.path.isdir(base):
                init = _src(os.path.join(base, "__init__.py"))
                for name in names:
                    if not (
                        os.path.isfile(os.path.join(base, name + ".py"))
                        or os.path.isdir(os.path.join(base, name))
                        or re.search(rf"\b{name}\b", init)
                    ):
                        broken.append(f"{os.path.relpath(p, ROOT)}:{line} {spec} import {name}")
    assert n > 20 and not broken, broken
    assert "from ..ops import m3_fsel_topk as _fsel" in _src(os.path.join(TP4, "modeling.py"))


def test_tp4_wiring_carries_the_switch_the_bound_and_the_engagement_lines():
    src = _src(os.path.join(TP4, "modeling.py"))
    assert 'FSEL_TOPK = not m31_disabled("TP4_FSEL")' in src
    assert "FSEL_MAX_ROWS = 8 " in src
    assigns = {ast.unparse(n) for n in ast.parse(src).body if isinstance(n, ast.Assign)}
    assert "FSEL_TRIGGER_PHASE = 0 if SELECT_EARLY_MAX_ROWS >= FSEL_MAX_ROWS else 1" in assigns
    assert '"tp4_fsel",' in src and "    FSEL_SELECT,\n" in src
    assert (
        "if FSEL_SELECT.keys and num_tokens <= FSEL_MAX_ROWS and 128 < max_score_kv.shape[1]:"
        in src
    )
    assert "_fsel.log_engaged(" in src and "_fsel.prepare(" in src
    fam = _src(os.path.join(OPS, "m3_fsel_topk.py"))
    assert "[M31_TP4_FSEL] engaged: target" in fam
    # The DSL select stays the fallback: nested under "if table is None", with its own compiled-key check.
    assert "            if table is None:\n                if LONG_SELECT_DSL.keys" in src
