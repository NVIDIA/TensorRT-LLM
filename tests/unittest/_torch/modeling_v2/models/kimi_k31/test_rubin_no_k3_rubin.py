# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 Rubin targets do not run Kimi K3's package (``kimi_k3_vl/k3_rubin``).

Their code is the specialized targets (including ULL generation and its wide-batch fork), the common
SM10x default, and
the family's ``rubin_*`` kernels and entries, and an import of K3's package would also apply K3's
import-time settings (such as its compiler-flag patch). These checks parse the sources;
``test_rubin_import_closure.py`` imports the targets (in the image) and records what the import
pulls in."""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[6]
PKG = REPO / "tensorrt_llm"
FAMILY = PKG / "_torch/_experimental/modeling_v2/models/kimi_k31"
ROOTS = sorted(FAMILY.glob("kimi_k31_pro_nvfp4__sm_107__*"))
ROOTS.append(FAMILY / "kimi_k31_pro_nvfp4__sm_10x__default")
#: The family files the Rubin targets add (``rubin_*`` kernels and entries).
FILES = sorted(FAMILY.glob("kernels/**/rubin_*.py")) + sorted(FAMILY.glob("ops/**/rubin_*.py"))
K3_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin"
K3_DIR = PKG / "_torch/_experimental/modeling_v2/models/kimi_k3_vl"

#: Shared modules that reach into K3's package, registered until the code they use is shared: the
#: module (relative to ``tensorrt_llm/``) -> why. ``test_rubin_import_closure.py`` reads it too.
EXCEPTIONS = {
    "_torch/modules/kimi_kda/cache_manager.py": "the batched KDA replay seed uses K3's KDA kernels",
    "_torch/moe/fused_moe/cutedsl/fused_fc12.py": "the FC12 experts module, until K3.1 injects its own",
}


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(REPO).with_suffix("").parts)


def imported_modules(path: Path):
    """``(line, module)`` of every import in ``path``, relative imports resolved against its
    package."""
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    return imported_modules_of(path.read_text(), package, str(path))


def imported_modules_of(source: str, package: str, filename: str = "<source>"):
    """``(line, module)`` of every import in ``source`` of a module of ``package``: ``import`` and
    ``from ... import`` (each name also as ``module.name``) and ``import_module`` /
    ``__import__`` calls with a literal name."""
    tree = ast.parse(source, filename=filename)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - node.level + 1]
                base = ".".join(parts + ([node.module] if node.module else []))
            found.append((node.lineno, base))
            found += [(node.lineno, f"{base}.{alias.name}") for alias in node.names]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("import_module", "__import__") and isinstance(node.args[0].value, str):
                found.append((node.lineno, node.args[0].value))
    return found


def _forbidden(module: str) -> bool:
    return module == K3_PACKAGE or module.startswith(K3_PACKAGE + ".")


def test_targets_and_family_files_do_not_import_k3_rubin():
    assert len(ROOTS) == 7 and all(root.is_dir() for root in ROOTS) and FILES, (ROOTS, len(FILES))
    offenders = []
    for path in [p for root in ROOTS for p in sorted(root.rglob("*.py"))] + FILES:
        for line, module in imported_modules(path):
            if _forbidden(module):
                offenders.append(f"{path.relative_to(REPO)}:{line}: {module}")
    assert not offenders, "\n  ".join(["imports of K3's package:", *offenders])


def test_shared_code_names_k3_rubin_only_in_the_registered_modules():
    """Outside K3's own family, only the registered shared modules name K3's package; another
    module that does is either converted or registered here with its reason."""
    k3_users = set()
    for path in sorted(PKG.rglob("*.py")):
        if K3_DIR in path.parents:
            continue
        text = path.read_text(errors="replace")
        if "kimi_k3_vl.k3_rubin" in text or "kimi_k3_vl import k3_rubin" in text:
            k3_users.add(str(path.relative_to(PKG)))
    assert k3_users <= set(EXCEPTIONS), sorted(k3_users - set(EXCEPTIONS))


def test_registered_exceptions_still_name_k3_rubin():
    """An exception whose module no longer reaches into K3's package is removed with it."""
    for rel in EXCEPTIONS:
        assert "kimi_k3_vl.k3_rubin" in (PKG / rel).read_text(), rel


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin import features\n",
            True,
        ),
        ("from ...kimi_k3_vl.k3_rubin.kda import x\n", True),
        ("from ...kimi_k3_vl import k3_rubin\n", True),
        (
            "import importlib\nimportlib.import_module("
            "'tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.kda')\n",
            True,
        ),
        ("from ...kimi_k3_vl import routing\n", False),
        ("from . import features\n", False),
    ],
)
def test_the_import_check_finds_every_form(source, expected):
    """Relative imports resolve as in a module of the package."""
    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8"
    assert any(_forbidden(module) for _, module in imported_modules_of(source, package)) is expected
