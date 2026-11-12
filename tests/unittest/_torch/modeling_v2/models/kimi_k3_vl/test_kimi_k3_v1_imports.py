# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 (``models/kimi_k3_vl``) imports no Modeling V1 code.

The Modeling V2 rule (``_experimental/modeling_v2/README.md``, "Modeling V1 code"): a target
does not import, subclass or call the model classes in ``_torch/models/``, the layers in
``_torch/modules/``, MLA and the attention backends in ``_torch/attention/``, or fused MoE in
``_torch/moe/``; from the rest of TensorRT-LLM it uses only runtime interfaces.

This file reads every ``.py`` of the family (its shared ``ops/``, ``kernels/``, ``common/``,
``routing.py`` and every SM107 target) with ``ast``, resolves every import (module level and
function level, absolute and relative, and string arguments of ``importlib.import_module`` /
``__import__``) and fails on any that reaches one of those four packages, except:

* ``RUNTIME_INTERFACES``: the runtime interfaces the family declares it uses, each with the
  reason. Symbol lists are exact; ``"*"`` takes the whole module (only the FP4 MLA KV cache
  layout contract, which ``Fp4MlaKVCacheManagerV2`` owns).
* ``MOE_KERNELS``: kernels and their custom ops that live under ``_torch/moe/`` but are not
  the fused MoE layer (``fused_moe/``) the rule forbids, with exact symbol lists.
* ``PENDING``: V1 code another line of the K3 v2 plan is still replacing, with that line's
  name. An entry no import needs any more fails ``test_no_stale_pending``, so the list only
  shrinks; it is empty when the plan is done.
* ``EXEMPT_TARGETS``: the two MXFP4 sm100 targets, which predate the rule and are not part of
  this plan. They are listed by name, so a new target is scanned.

Run it as a script to print the report: ``python3 test_kimi_k3_v1_imports.py``.
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

try:
    import pytest
except ImportError:  # the script form needs only the standard library
    pytest = None
else:
    pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[6]
_FAMILY_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
_FAMILY_DIR = _REPO.joinpath(*_FAMILY_PACKAGE.split("."))

#: The Modeling V1 packages a target must not import.
FORBIDDEN = (
    "tensorrt_llm._torch.models",
    "tensorrt_llm._torch.modules",
    "tensorrt_llm._torch.attention",
    "tensorrt_llm._torch.moe",
)

#: The MXFP4 sm100 targets: exempt from the rule.
EXEMPT_TARGETS = frozenset(
    {
        "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1",
        "kimi_k3_mxfp4__sm_100__tp16_moetp4ep4",
    }
)

_ATTENTION = "tensorrt_llm._torch.attention.backends"
_FP4_MLA = f"{_ATTENTION}.fp4_mla"

#: Runtime interfaces under the forbidden packages: module -> (symbols or "*", reason).
RUNTIME_INTERFACES: Dict[str, Tuple[object, str]] = {
    _ATTENTION: (
        {"AttentionMetadata", "TrtllmAttentionMetadata"},
        "attention metadata, the engine's per-step interface (README names it)",
    ),
    f"{_ATTENTION}.interface": ({"AttentionMetadata"}, "attention metadata (as above)"),
    f"{_ATTENTION}.trtllm": (
        {"TrtllmAttentionMetadata"},
        "attention metadata: KV pool pointers, block offsets and the FP4 MLA state",
    ),
    f"{_FP4_MLA}.config": (
        "*",
        "FP4 MLA KV cache layout contract shared with Fp4MlaKVCacheManagerV2 (PLAN §3, R2 G4)",
    ),
    f"{_FP4_MLA}.layout": ("*", "FP4 MLA KV cache layout contract (as above)"),
    f"{_FP4_MLA}.metadata": ("*", "FP4 MLA KV cache page-table metadata contract (as above)"),
    f"{_FP4_MLA}.packed_plan": (
        "*",
        "the packed decode plan geometry the metadata contract builds every step (as above)",
    ),
    "tensorrt_llm._torch.modules.kimi_kda.cache_manager": (
        {"KimiK3HybridCacheManager", "KDAReplayState"},
        "the KV cache manager the engine resolves for K3 (README: 'the KV cache managers'), "
        "a runtime interface that lives under modules/ for history only: "
        "ops/cache/k3_sm107_cache_manager.py subclasses it as the SM107 targets' "
        "RuntimeHooks.kv_cache_manager_cls, and the DSpark worker checks its KDA state update",
    ),
    "tensorrt_llm._torch.modules.mamba.mamba2_metadata": (
        {"Mamba2Metadata"},
        "the engine's mamba metadata interface (resolve_mamba_metadata_cls builds the "
        "model's mamba_metadata_cls)",
    ),
    "tensorrt_llm._torch.modules.multi_stream_utils": (
        {"do_multi_stream"},
        "engine state: the CUDA graph runner turns multi-stream on with "
        "with_multi_stream(True); common/multi_stream.py reads that one flag",
    ),
    "tensorrt_llm._torch.models.modeling_utils": (
        {"DecoderModel", "DecoderModelForCausalLM"},
        "the model lifecycle base classes (README names them)",
    ),
    "tensorrt_llm._torch.models.modeling_speculative": (
        {"SpecDecOneEngineForCausalLM"},
        "the split speculative-decoding base the README names",
    ),
    "tensorrt_llm._torch.models.checkpoints.base_weight_loader": (
        {"ConsumableWeightsDict"},
        "the engine's checkpoint weight container",
    ),
}

#: Interfaces that only one family module may import (relative to the family directory).
RUNTIME_INTERFACE_HOMES: Dict[str, str] = {
    "tensorrt_llm._torch.modules.multi_stream_utils": "common/multi_stream.py",
}

#: Kernels under ``_torch/moe/`` (not the fused MoE layer): module -> (symbols, reason).
MOE_KERNELS: Dict[str, Tuple[object, str]] = {
    "tensorrt_llm._torch.moe.kernels.rubin": (
        {"manual_mma_128dp"},
        "the 128-datapath MMA helpers of the Rubin MoE kernels, which the family's copy of the "
        "SM107 fused FC1+FC2 kernel (kernels/k3r_moe/fc12_kernel.py) calls",
    ),
    "tensorrt_llm._torch.moe.custom_ops": (
        {"cute_dsl_megamoe_custom_op"},
        "the MegaMoE CuTe DSL op the dep16 target launches (availability probe)",
    ),
    "tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op": (
        {
            "megamoe_activation_sf_bytes_per_row",
            "run_cute_dsl_megamoe_nvfp4_blackwell",
            "set_active_megamoe_cute_buffer_cache",
            "set_active_megamoe_cute_weight_cache",
        },
        "the MegaMoE CuTe DSL op: its launch, buffer / weight cache binding and SF row size",
    ),
}

#: V1 code still imported, by the line of the K3 v2 plan that replaces it:
#: module -> (symbols or "*", owner). Delete an entry with the import.
PENDING: Dict[str, Tuple[object, str]] = {}


class Import(NamedTuple):
    """One imported name: ``module`` and the ``symbol`` taken from it (None for ``import m``)."""

    path: str  # relative to the family directory
    line: int
    module: str
    symbol: Optional[str]


def _is_forbidden(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in FORBIDDEN)


def _module_name(path: Path) -> str:
    rel = path.relative_to(_FAMILY_DIR).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join([_FAMILY_PACKAGE, *parts])


def _resolve(path: Path, level: int, module: Optional[str]) -> str:
    """The absolute module of ``from <level dots><module> import ...`` in ``path``."""
    if level == 0:
        return module or ""
    package = _module_name(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    base = package[: len(package) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def _string_imports(node: ast.Call) -> Iterator[str]:
    """String module names passed to ``importlib.import_module`` / ``__import__`` /
    ``find_spec``."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name not in ("import_module", "__import__", "find_spec"):
        return
    for arg in node.args[:1]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            yield arg.value


def imports_of(path: Path) -> Iterator[Import]:
    rel = path.relative_to(_FAMILY_DIR).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Import(rel, node.lineno, alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            module = _resolve(path, node.level, node.module)
            for alias in node.names:
                # ``from a import b`` may import submodule ``a.b``: report it as such when ``a.b``
                # is under a forbidden package and ``a`` is not.
                submodule = f"{module}.{alias.name}"
                if not _is_forbidden(module) and _is_forbidden(submodule):
                    yield Import(rel, node.lineno, submodule, None)
                else:
                    yield Import(rel, node.lineno, module, alias.name)
        elif isinstance(node, ast.Call):
            for module in _string_imports(node):
                yield Import(rel, node.lineno, module, None)


def scanned_files() -> List[Path]:
    files = []
    for path in sorted(_FAMILY_DIR.rglob("*.py")):
        top = path.relative_to(_FAMILY_DIR).parts[0]
        if top in EXEMPT_TARGETS or "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def _allowed(table: Dict[str, Tuple[object, str]], imp: Import) -> Optional[str]:
    entry = table.get(imp.module)
    if entry is None:
        return None
    symbols, why = entry
    if symbols == "*" or (imp.symbol is not None and imp.symbol in symbols):
        return why
    return None


class Report(NamedTuple):
    violations: List[Import]
    interface: List[Import]
    pending: Dict[str, List[Import]]  # owner -> imports
    misplaced_interface: List[Import]


def scan() -> Report:
    violations, interface, misplaced = [], [], []
    pending: Dict[str, List[Import]] = defaultdict(list)
    for path in scanned_files():
        for imp in imports_of(path):
            if not _is_forbidden(imp.module):
                continue
            if _allowed(RUNTIME_INTERFACES, imp) is not None:
                home = RUNTIME_INTERFACE_HOMES.get(imp.module)
                if home is None or imp.path == home:
                    interface.append(imp)
                    continue
                owner = _allowed(PENDING, imp)
                if owner is not None:
                    pending[owner].append(imp)
                else:
                    misplaced.append(imp)
                continue
            if _allowed(MOE_KERNELS, imp) is not None:
                interface.append(imp)
                continue
            owner = _allowed(PENDING, imp)
            if owner is not None:
                pending[owner].append(imp)
                continue
            violations.append(imp)
    return Report(violations, interface, dict(pending), misplaced)


def _fmt(imp: Import) -> str:
    what = imp.module if imp.symbol is None else f"{imp.module} :: {imp.symbol}"
    return f"{imp.path}:{imp.line}  {what}"


def _used_pending(report: Report) -> Set[Tuple[str, Optional[str]]]:
    return {(imp.module, imp.symbol) for imps in report.pending.values() for imp in imps}


def stale_pending(report: Report) -> List[str]:
    used = _used_pending(report)
    stale = []
    for module, (symbols, owner) in sorted(PENDING.items()):
        names = [None] if symbols == "*" else sorted(symbols)
        for name in names:
            hit = any(m == module and (name is None or s == name) for m, s in used)
            if not hit:
                stale.append(f"{owner}: {module}" + ("" if name is None else f" :: {name}"))
    return stale


def test_scope_covers_every_sm107_target():
    """Every target package is scanned, except the two listed sm100 targets.

    A target is a package (``__init__.py``): a directory a checkout switch left with only
    ``__pycache__`` in it is not one.
    """
    targets = {
        p.name for p in _FAMILY_DIR.iterdir() if "__sm_" in p.name and (p / "__init__.py").is_file()
    }
    assert EXEMPT_TARGETS <= targets, "an exempt target is gone: drop it from EXEMPT_TARGETS"
    scanned = targets - EXEMPT_TARGETS
    assert scanned and all("__sm_107__" in t for t in scanned), sorted(scanned)
    scanned_tops = {p.relative_to(_FAMILY_DIR).parts[0] for p in scanned_files()}
    assert scanned <= scanned_tops


def test_resolver_reads_relative_and_string_imports(tmp_path, monkeypatch):
    """The resolver maps relative imports out of the family and string imports to modules."""
    family = tmp_path.joinpath(*_FAMILY_PACKAGE.split("."))
    target = family / "kimi_k3_nvfp4__sm_107__x"
    target.mkdir(parents=True)
    source = target / "m.py"
    source.write_text(
        "from ......modules.linear import Linear\n"
        "from tensorrt_llm._torch import moe\n"
        "def f():\n"
        "    import importlib\n"
        "    importlib.import_module('tensorrt_llm._torch.models.modeling_kimi_k25')\n"
        "    from ..common import multi_stream\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "_FAMILY_DIR", family)
    got = {(i.module, i.symbol) for i in imports_of(source)}
    assert ("tensorrt_llm._torch.modules.linear", "Linear") in got
    assert ("tensorrt_llm._torch.moe", None) in got
    assert ("tensorrt_llm._torch.models.modeling_kimi_k25", None) in got
    assert (f"{_FAMILY_PACKAGE}.common", "multi_stream") in got
    outside = {m for m, _ in got if not m.startswith(_FAMILY_PACKAGE)} - {"importlib"}
    assert all(_is_forbidden(m) for m in outside), outside


def test_no_v1_imports():
    """No import reaches Modeling V1 code beyond the declared interfaces and pending work."""
    report = scan()
    assert not report.violations, "Modeling V1 imports:\n" + "\n".join(
        _fmt(i) for i in report.violations
    )


def test_interfaces_imported_from_their_home():
    report = scan()
    assert not report.misplaced_interface, "\n".join(
        f"{_fmt(i)} (import it through {RUNTIME_INTERFACE_HOMES[i.module]})"
        for i in report.misplaced_interface
    )


def test_no_stale_pending():
    """A PENDING entry no import needs any more is deleted, so the list only shrinks."""
    stale = stale_pending(scan())
    assert not stale, "delete from PENDING:\n" + "\n".join(stale)


def _summary(imps: List[Import]) -> List[str]:
    """One line per imported module: its symbols and the files importing it."""
    by_module: Dict[str, Tuple[Set[str], Set[str]]] = defaultdict(lambda: (set(), set()))
    for imp in imps:
        symbols, files = by_module[imp.module]
        symbols.add(imp.symbol or "<module>")
        files.add(imp.path.split("/")[-1])
    return [
        f"  {module} :: {', '.join(sorted(symbols))}  [{', '.join(sorted(files))}]"
        for module, (symbols, files) in sorted(by_module.items())
    ]


def main() -> int:
    report = scan()
    if "--summary" in sys.argv:
        print(f"Kimi K3 V1 import scan: {len(scanned_files())} files")
        print(f"violations: {len(report.violations)}")
        print("\n".join(_summary(report.violations)))
        for owner in sorted(report.pending):
            print(f"pending ({owner}): {len(report.pending[owner])}")
            print("\n".join(_summary(report.pending[owner])))
        print(f"declared runtime interfaces: {len(report.interface)}")
        print("\n".join(_summary(report.interface)))
        for imp in report.misplaced_interface:
            print(f"misplaced interface: {_fmt(imp)}")
        for entry in stale_pending(report):
            print(f"stale pending: {entry}")
        return 1 if report.violations or report.misplaced_interface else 0
    print(f"Kimi K3 V1 import scan: {len(scanned_files())} files under {_FAMILY_DIR}")
    print(f"exempt targets: {', '.join(sorted(EXEMPT_TARGETS))}\n")
    print(f"violations: {len(report.violations)}")
    for imp in report.violations:
        print(f"  {_fmt(imp)}")
    for owner in sorted(report.pending):
        imps = report.pending[owner]
        print(f"pending ({owner}): {len(imps)}")
        for imp in imps:
            print(f"  {_fmt(imp)}")
    print(f"declared runtime interfaces: {len(report.interface)}")
    for imp in report.interface:
        print(f"  {_fmt(imp)}")
    for imp in report.misplaced_interface:
        print(f"misplaced interface: {_fmt(imp)}")
    for entry in stale_pending(report):
        print(f"stale pending: {entry}")
    return 1 if report.violations or report.misplaced_interface else 0


if __name__ == "__main__":
    sys.exit(main())
