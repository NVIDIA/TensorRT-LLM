# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The lender's public surface and boundaries: the eleven names, their members, what importing the
package loads, who may import its private modules, and what its public docstrings promise callers.
Every scan has a positive control that plants the fault it looks for. No GPU."""

import ast
import dataclasses
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import weakref
from pathlib import Path
from typing import List, Protocol, Set

import numpy as np
import pytest

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.kv_cache import sharing
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import (
    GroupRun,
    InPlaceLender,
    Lease,
    Part,
    PartsHold,
    Readiness,
    RegionView,
    StagingLender,
    StagingOptions,
    attach_in_place,
    attach_staging,
)

SH = "tensorrt_llm._torch.pyexecutor.kv_cache.sharing"
MGR = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
PUBLIC = [
    "GroupRun",
    "InPlaceLender",
    "Lease",
    "Part",
    "PartsHold",
    "Readiness",
    "RegionView",
    "StagingLender",
    "StagingOptions",
    "attach_in_place",
    "attach_staging",
]
MODULES = ["__init__.py", "_identity.py", "_layout.py", "_lender.py", "_manager.py", "_slots.py"]
MODULES = sorted(MODULES + ["_types.py"])
PKG_DIR = Path(sharing.__file__).resolve().parent
ROOT = Path(tensorrt_llm.__file__).resolve().parent
TESTS_DIR = Path(__file__).resolve().parent
# Manager and runtime members only the manager facade may touch.
INTERNALS = {
    "_reuse_token_source",
    "_augment_tokens_for_block_reuse",
    "_stale_block_range",
    "_resize_for_connector_prefix",
    "_fill_fresh_kv_pages",
    "_can_publish_block_reuse",
    "_stream",
    "_layer_attn_to_layer_id",
    "_sharing",
    "host_kv_cache_block_offsets",
    "kv_cache_map",
}
BACKEND_WORDS = re.compile(r"\b(kvcr|mooncake|blob|native)\b", re.IGNORECASE)


def fact(text: str) -> re.Pattern:
    """A fact a docstring states: ``text`` matched case-insensitively, any run of whitespace for a
    space, so a fact wrapped in prose or indented under a Google-style section reads the same."""
    return re.compile(r"\s+".join(re.escape(word) for word in text.split()), re.IGNORECASE)


# What callers rely on, per documented name: the thread rule, rank-local outcomes, copies serial
# with the forward, what keeps memory past the manager's shutdown, and the in-place preconditions.
THREADS = (
    fact("only on the manager's thread"),
    fact("call no lease or lender method"),
    fact("their own channel"),
)
RANK_LOCAL = (fact("this rank's own"), fact("the caller combines every rank's outcome"))
HOLD_AT_SHUTDOWN = fact(
    "A hold still open at the manager's shutdown keeps the staging memory until the process exits"
)
DOC_FACTS = {
    "attach_staging": (fact("block reuse on"), fact("commits no blocks")),
    "attach_in_place": (
        fact("on loan at the manager's shutdown"),
        fact("stay until the process exits"),
    ),
    "Lease": THREADS + (fact("read the view, access the memory it points to"),),
    "StagingLender": THREADS + (fact("read a view, access the memory it points to"),),
    "InPlaceLender": THREADS + (fact("access the lent memory"), fact("own page-table code")),
    "PartsHold": (
        HOLD_AT_SHUTDOWN,
        fact("release it on the manager's thread once deregistration is confirmed"),
        fact("dropping it unreleased keeps the memory"),
    ),
    "PartsHold.release": (fact("only on the manager's thread"),),
    "StagingLender.lend_read": RANK_LOCAL,
    "StagingLender.lend_write": (
        fact("Fails as ``lend_read``"),
        fact("no free pages"),
        fact("SWA scratch reuse"),
    ),
    "StagingLender.readiness": (
        fact("serial with the forward"),
        fact("no lender call waits for them on the CPU"),
    ),
    "InPlaceLender.lend_read": RANK_LOCAL
    + (fact("work the manager's stream queued that still writes those pages completed"),),
    "InPlaceLender.lend_write": (
        fact("all work the manager's stream queued for those pages has completed"),
        fact("unscheduled, unsuspended, unshrunk and its window still"),
    ),
    "StagingOptions": (fact("A capacity budget, not concurrency"), fact("holes")),
    "StagingLender.parts": (
        fact("an unreleased lease"),
        fact("an unreleased hold"),
        fact("a slot lost to a failed copy"),
    ),
}
FIRST_YEAR = 2026
COPYRIGHT = re.compile(
    r"# SPDX-FileCopyrightText: Copyright \(c\) (?:(20\d\d)-)?(20\d\d) NVIDIA CORPORATION & "
    r"AFFILIATES\. All rights reserved\.$"
)
LICENSE = [
    "# SPDX-License-Identifier: Apache-2.0",
    "#",
    '# Licensed under the Apache License, Version 2.0 (the "License");',
    "# you may not use this file except in compliance with the License.",
    "# You may obtain a copy of the License at",
    "#",
    "# http://www.apache.org/licenses/LICENSE-2.0",
    "#",
    "# Unless required by applicable law or agreed to in writing, software",
    '# distributed under the License is distributed on an "AS IS" BASIS,',
    "# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.",
    "# See the License for the specific language governing permissions and",
    "# limitations under the License.",
]


def has_header(source: str) -> bool:
    """The NVIDIA header: a copyright year, or a range of years, ending in 2026 or later, then the
    Apache license text."""
    lines = source.splitlines()
    match = COPYRIGHT.match(lines[0]) if lines else None
    if match is None or lines[1 : 1 + len(LICENSE)] != LICENSE:
        return False
    first, last = int(match.group(1) or match.group(2)), int(match.group(2))
    return first <= last and last >= FIRST_YEAR


def package_files() -> List[Path]:
    return sorted(PKG_DIR.glob("*.py"))


def module_name(path: Path, root: Path = ROOT) -> str:
    relative = path.resolve().relative_to(root.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def package_of(path: Path, root: Path = ROOT) -> str:
    """The package a file's relative imports resolve against."""
    name = module_name(path, root)
    return name if path.name == "__init__.py" else name.rpartition(".")[0]


def imported_modules(source: str, package: str) -> Set[str]:
    """Every module (and ``module.name``) a source imports, lazy imports and
    ``importlib.import_module`` with a literal included; relative imports resolved."""
    found: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            first = node.args[0] if node.args else None
            if name in ("import_module", "__import__") and isinstance(first, ast.Constant):
                found.add(str(first.value))
    return found


PRIVATE_IMPORT = re.compile(r"^" + re.escape(SH) + r"\._")


def private_imports(source: str, package: str) -> List[str]:
    """What a source outside the package imports of the package's private modules."""
    hits = sorted(m for m in imported_modules(source, package) if PRIVATE_IMPORT.match(m))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "kv_cache.sharing._" in node.value:
                hits.append(node.value)
    return hits


def private_importers(root: Path, pkg_dir: Path) -> dict:
    """Modules under ``root`` outside ``pkg_dir`` that import the package's private modules."""
    offenders = {}
    for path in root.rglob("*.py"):
        if pkg_dir in path.resolve().parents:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if "sharing" not in source:
            continue
        try:
            hits = private_imports(source, package_of(path, root))
        except SyntaxError:
            continue  # not importable, so it imports nothing
        if hits:
            offenders[path.relative_to(root).as_posix()] = hits
    return offenders


def disaggregation_imports(source: str, package: str) -> List[str]:
    prefix = "tensorrt_llm._torch.disaggregation"
    return sorted(m for m in imported_modules(source, package) if m.startswith(prefix))


def internal_reads(source: str) -> List[str]:
    """Attribute names (or ``getattr`` literals) of manager internals a source uses."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in INTERNALS:
            hits.append(node.attr)
        elif isinstance(node, ast.Constant) and node.value in INTERNALS:
            hits.append(str(node.value))
    return sorted(hits)


def threads_or_finalizers(source: str) -> List[str]:
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.split(".")[0] in _THREADING]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _THREADING:
            hits.append(node.module)
        elif isinstance(node, ast.ImportFrom) and any(a.name == "finalize" for a in node.names):
            hits.append("finalize")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__del__":
            hits.append("__del__")
        elif isinstance(node, ast.Attribute) and node.attr == "finalize":
            hits.append("finalize")
    return hits


_THREADING = {"threading", "_thread", "concurrent", "multiprocessing"}


def run_python(code: str) -> dict:
    """Run ``code`` in a fresh interpreter importing this tree's ``tensorrt_llm``; it prints JSON."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT.parent), env.get("PYTHONPATH")]))
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    # Proof the child imported the package under test, not another installed copy.
    assert Path(result["file"]).resolve() == Path(sharing.__file__).resolve()
    return result


# -- the names --------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_the_package_exports_exactly_eleven_names():
    assert sorted(sharing.__all__) == PUBLIC
    assert len(sharing.__all__) == len(set(sharing.__all__))
    for name in PUBLIC:
        assert getattr(sharing, name) is not None


@pytest.mark.cpu_only
def test_importing_the_package_loads_its_types_alone_and_exposes_nothing_else():
    result = run_python(
        "import json, sys\n"
        f"import {SH} as s\n"
        "print(json.dumps({'file': s.__file__, 'all': s.__all__,\n"
        "    'public': sorted(n for n in vars(s) if not n.startswith('_')),\n"
        f"    'loaded': sorted(m for m in sys.modules if m.startswith('{SH}'))}}))\n"
    )
    assert result["public"] == sorted(result["all"]) == PUBLIC
    assert result["loaded"] == [SH, f"{SH}._types"]


@pytest.mark.cpu_only
def test_the_package_is_its_init_and_six_private_modules():
    assert [p.name for p in package_files()] == MODULES
    subpackages = [p.name for p in PKG_DIR.iterdir() if p.is_dir() and p.name != "__pycache__"]
    assert subpackages == []


@pytest.mark.cpu_only
def test_the_protocols_keep_their_members_and_stay_unrelated():
    def members(cls):
        return {n for n in vars(cls) if not n.startswith("_")}

    assert members(Lease) == {"poll", "failure", "mark_arrived", "release"}
    assert members(StagingLender) == {"parts", "hold_parts", "lend_read", "lend_write", "readiness"}
    assert members(InPlaceLender) == {"lend_read", "lend_write"}
    assert members(PartsHold) == {"release"}
    assert StagingLender not in InPlaceLender.__mro__
    assert InPlaceLender not in StagingLender.__mro__
    assert Lease not in PartsHold.__mro__ and PartsHold not in Lease.__mro__
    for protocol in (Lease, StagingLender, InPlaceLender, PartsHold):
        assert Protocol in protocol.__mro__
        assert not isinstance(object(), protocol), "checkable at run time"
    assert isinstance(Lease.failure, property) and isinstance(StagingLender.parts, property)

    def params(func):
        return list(inspect.signature(func).parameters)

    assert params(Lease.poll) == ["self"]
    assert params(Lease.mark_arrived) == ["self", "masks"]
    assert params(Lease.release) == ["self"]
    assert params(PartsHold.release) == ["self"]
    assert params(StagingLender.hold_parts) == ["self"]
    for protocol in (StagingLender, InPlaceLender):
        assert params(protocol.lend_read) == ["self", "request", "start", "end"]
        assert params(protocol.lend_write) == ["self", "request", "start", "end"]
    assert params(StagingLender.readiness) == ["self", "request"]


def public_names(obj) -> Set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


LEASE_MEMBERS = {"poll", "failure", "mark_arrived", "release"}
STAGING_MEMBERS = {"parts", "hold_parts", "lend_read", "lend_write", "readiness"}


@pytest.mark.cpu_only
def test_lenders_leases_and_holds_show_only_their_protocol():
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    class Owner:  # anything a weak reference can point to
        pass

    owner = Owner()
    ref = weakref.ref(owner)
    layout = SimpleNamespace(pool_groups=(), windows=())  # all the constructors read
    shown = {
        "staging lender": (_lender.Staging(ref, layout, None, (), None, ref), STAGING_MEMBERS),
        "in-place lender": (_lender.InPlace(ref, layout), {"lend_read", "lend_write"}),
        "staging lease": (_lender._StagingLease(owner, "read", 1), LEASE_MEMBERS),
        "in-place lease": (_lender._InPlaceLease(owner, "write", None), LEASE_MEMBERS),
        "parts hold": (_lender._PartsHold(owner), {"release"}),
    }
    assert {what: public_names(obj) for what, (obj, _) in shown.items()} == {
        what: members for what, (_, members) in shown.items()
    }
    # A lease showing its view as an attribute, and a lender showing a manager hook, are found.
    leaky = _lender._StagingLease(owner, "read", 1)
    leaky.view = None
    assert public_names(leaky) - LEASE_MEMBERS == {"view"}
    hooked = type("Hooked", (_lender.InPlace,), {"on_free": _lender.InPlace._on_free})
    assert public_names(hooked(ref, layout)) - {"lend_read", "lend_write"} == {"on_free"}


@pytest.mark.cpu_only
def test_the_types_keep_their_fields():
    def fields(cls):
        return [(f.name, f.default) for f in dataclasses.fields(cls)]

    missing = dataclasses.MISSING
    assert fields(StagingOptions) == [
        ("fetch_tokens", missing),
        ("max_fetches", 1),
        ("max_bytes", None),
    ]
    assert fields(Part) == [(n, missing) for n in ("name", "address", "nbytes", "slot_bytes")] + [
        ("slots", missing)
    ]
    assert fields(GroupRun) == [
        ("layer_group", missing),
        ("ordinals", missing),
        ("names", None),
        ("addresses", None),
        ("part", None),
    ]
    assert fields(RegionView) == [("runs", missing)]
    for cls in (StagingOptions, Part, GroupRun, RegionView):
        assert cls.__dataclass_params__.frozen, cls
    assert Readiness._fields == ("usable_until", "restart_floor")
    assert issubclass(Readiness, tuple)

    def beyond_fields(cls):
        named = {f.name for f in dataclasses.fields(cls)}
        return {n for n in dir(cls) if not n.startswith("_")} - named

    assert beyond_fields(GroupRun) == {"select"}
    assert beyond_fields(RegionView) == {"num_rows", "row_masks"}
    assert beyond_fields(Part) == beyond_fields(StagingOptions) == set()
    assert {n for n in dir(Readiness) if not n.startswith("_")} == {
        "usable_until",
        "restart_floor",
        "count",
        "index",
    }


@pytest.mark.cpu_only
def test_the_attach_functions_keep_their_signatures():
    kind = inspect.Parameter

    def shape(func):
        return [(p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()]

    assert shape(attach_staging) == [
        ("manager", kind.POSITIONAL_OR_KEYWORD, kind.empty),
        ("scope", kind.KEYWORD_ONLY, kind.empty),
        ("staging", kind.KEYWORD_ONLY, kind.empty),
    ]
    assert shape(attach_in_place) == [("manager", kind.POSITIONAL_OR_KEYWORD, kind.empty)]


@pytest.mark.cpu_only
def test_a_name_is_as_long_as_the_identity_makes_it():
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _identity, _types

    assert _types._NAME_BYTES == _identity.NAME_BYTES == 54


@pytest.mark.cpu_only
def test_the_manager_gains_only_a_private_class_attribute():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    assert "_sharing" in vars(KVCacheManagerV2) and KVCacheManagerV2._sharing is None
    words = re.compile(r"lend|lent|loan|sharing|staging|retain")
    assert [n for n in dir(KVCacheManagerV2) if not n.startswith("_") and words.search(n)] == []


# -- boundaries -------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_the_scans_catch_every_import_form_they_look_for():
    package = "tensorrt_llm._torch.pyexecutor.kv_cache"
    caught = [
        f"from {SH}._lender import attach_staging",
        f"from {SH} import _types",
        f"import {SH}._slots",
        "from .sharing._identity import Identity",
        "from .sharing import _layout",
        f"import importlib\nimportlib.import_module('{SH}._manager')",
        "def later():\n    from .sharing._lender import Staging\n",
    ]
    for source in caught:
        assert private_imports(source, package), source
    for source in (f"from {SH} import GroupRun", "from .sharing import attach_staging"):
        assert private_imports(source, package) == [], source
    assert disaggregation_imports("from ....disaggregation.resource.page import MapperKind", SH)
    assert disaggregation_imports("def f():\n    import tensorrt_llm._torch.disaggregation\n", SH)
    assert disaggregation_imports("from .._layout import x", SH) == []
    assert internal_reads("def f(m):\n    return m._stream, getattr(m, 'kv_cache_map')\n") == [
        "_stream",
        "kv_cache_map",
    ]
    assert threads_or_finalizers("import threading\n") == ["threading"]
    assert threads_or_finalizers("class A:\n    def __del__(self):\n        pass\n") == ["__del__"]
    assert threads_or_finalizers("import weakref\nweakref.finalize(o, f)\n") == ["finalize"]


@pytest.fixture
def tree_with_a_private_importer(tmp_path):
    """A ``tensorrt_llm`` tree whose package imports its own private module, as it may, and one
    module outside it that imports a private module too, as none may."""
    root = tmp_path / "tensorrt_llm"
    pkg_dir = root / "_torch/pyexecutor/kv_cache/sharing"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "__init__.py").write_text("from ._types import Part\n")
    (pkg_dir / "_lender.py").write_text("from ._types import Part\n")
    (root / "_torch/pyexecutor/public_user.py").write_text(
        "from .kv_cache.sharing import attach_staging\n"
    )
    (root / "_torch/pyexecutor/private_user.py").write_text(
        "def build():\n    from .kv_cache.sharing._lender import Staging\n"
    )
    return root.resolve(), pkg_dir.resolve()


@pytest.mark.cpu_only
def test_the_boundary_scan_catches_a_module_importing_a_private_one(tree_with_a_private_importer):
    root, pkg_dir = tree_with_a_private_importer
    assert private_importers(root, pkg_dir) == {
        "_torch/pyexecutor/private_user.py": [f"{SH}._lender", f"{SH}._lender.Staging"]
    }


@pytest.mark.cpu_only
def test_nothing_outside_the_package_imports_its_private_modules():
    assert private_importers(ROOT, PKG_DIR) == {}


@pytest.mark.cpu_only
def test_the_package_imports_nothing_from_disaggregation():
    offenders = {}
    for path in package_files():
        hits = disaggregation_imports(path.read_text(), package_of(path))
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


@pytest.mark.cpu_only
def test_the_manager_never_imports_the_package():
    path = ROOT / "_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py"
    source = path.read_text()
    loaded = imported_modules(source, package_of(path))
    assert [m for m in loaded if m.startswith(SH)] == []
    planted = source + "\nfrom .sharing import attach_in_place\n"
    assert [m for m in imported_modules(planted, package_of(path)) if m.startswith(SH)]


@pytest.mark.cpu_only
def test_the_native_path_loads_nothing_of_the_package():
    result = run_python(
        "import json, sys\n"
        "import tensorrt_llm._torch.disaggregation.transceiver\n"
        f"import {MGR}\n"
        f"loaded = sorted(m for m in sys.modules if m.startswith('{SH}'))\n"
        f"import {SH} as s\n"
        "print(json.dumps({'file': s.__file__, 'loaded': loaded}))\n"
    )
    assert result["loaded"] == []


@pytest.mark.cpu_only
def test_only_the_manager_facade_touches_manager_internals():
    offenders = {}
    for path in package_files():
        if path.name == "_manager.py":
            continue
        hits = internal_reads(path.read_text())
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


@pytest.mark.cpu_only
def test_the_package_has_no_threads_locks_or_finalizers():
    offenders = {}
    for path in package_files():
        hits = threads_or_finalizers(path.read_text())
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


# -- public docstrings and headers ------------------------------------------------------------


def public_docstrings():
    """(where, docstring) for the package, each exported name and each of their public members."""
    yield SH, sharing.__doc__ or ""
    for name in PUBLIC:
        obj = getattr(sharing, name)
        yield name, obj.__doc__ or ""
        if not inspect.isclass(obj):
            continue
        for member, value in vars(obj).items():
            if member.startswith("_"):
                continue
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            doc = getattr(value, "__doc__", None)
            if doc and (inspect.isfunction(value) or isinstance(value, property)):
                yield f"{name}.{member}", doc


def backend_words(docstrings) -> List[str]:
    return [where for where, doc in docstrings if BACKEND_WORDS.search(doc)]


@pytest.mark.cpu_only
def test_public_docstrings_name_no_backend(monkeypatch):
    documented = list(public_docstrings())
    assert {where for where, _ in documented} >= {"attach_staging", "Lease.poll", "Part"}
    assert backend_words(documented) == []
    # A planted word in one member's docstring is found.
    doc = StagingLender.lend_read.__doc__
    monkeypatch.setattr(StagingLender.lend_read, "__doc__", doc + " Suits a Mooncake store.")
    assert backend_words(public_docstrings()) == ["StagingLender.lend_read"]


def missing_facts(docstrings) -> dict:
    """Per documented name of ``DOC_FACTS``, the facts its docstring does not state."""
    docs = dict(docstrings)
    missing = {}
    for where, facts in DOC_FACTS.items():
        lacking = [f.pattern for f in facts if not f.search(docs.get(where, ""))]
        if lacking:
            missing[where] = lacking
    return missing


GOOGLE_STYLE_READINESS = """Where the request may resume once its fetch settled.

    Args:
        request: The request a fetch went into.

    Returns:
        ``None`` while a fetch into the request is unsettled. Copies queue on the manager's stream,
        serial with the forward
        on the GPU; with page-locked staging no lender call waits for them on the
        CPU.
    """


@pytest.mark.cpu_only
def test_public_docstrings_state_what_callers_rely_on(monkeypatch):
    assert missing_facts(public_docstrings()) == {}
    # A Google-style docstring stating the same facts across its sections passes.
    monkeypatch.setattr(StagingLender.readiness, "__doc__", GOOGLE_STYLE_READINESS)
    assert missing_facts(public_docstrings()) == {}
    # Docstrings that drop a fact are found, and a hold whose release reads as freeing.
    monkeypatch.setattr(InPlaceLender, "__doc__", "Lends a request's own device pages.")
    monkeypatch.setattr(StagingLender.lend_read, "__doc__", "A copy of the committed blocks.")
    monkeypatch.setattr(
        StagingLender.readiness, "__doc__", GOOGLE_STYLE_READINESS.replace("serial", "parallel")
    )
    hold_doc = " ".join(PartsHold.__doc__.split())
    frees = HOLD_AT_SHUTDOWN.sub("Releasing a hold frees the staging memory", hold_doc)
    monkeypatch.setattr(PartsHold, "__doc__", frees)
    missing = missing_facts(public_docstrings())
    assert sorted(missing) == [
        "InPlaceLender",
        "PartsHold",
        "StagingLender.lend_read",
        "StagingLender.readiness",
    ]
    assert missing["PartsHold"] == [HOLD_AT_SHUTDOWN.pattern]
    assert missing["StagingLender.readiness"] == [fact("serial with the forward").pattern]


def written_files() -> List[Path]:
    return package_files() + sorted(TESTS_DIR.glob("*.py"))


@pytest.mark.cpu_only
def test_every_new_file_carries_the_license_header():
    assert [p.name for p in written_files() if not has_header(p.read_text())] == []
    body = "\n".join(LICENSE)

    def header(years: str) -> str:
        owner = "NVIDIA CORPORATION & AFFILIATES. All rights reserved."
        return f"# SPDX-FileCopyrightText: Copyright (c) {years} {owner}\n{body}"

    for years in ("2026", "2027", "2025-2026", "2026-2027", "2031"):
        assert has_header(header(years)), years
    for years in ("2025", "2024-2025", "2027-2026"):
        assert not has_header(header(years)), years
    assert not has_header("# SPDX-FileCopyrightText: Copyright (c) 2026 Someone else.\n" + body)
    assert not has_header(header("2026").splitlines()[0] + "\n# SPDX-License-Identifier: MIT\n")


def identifiers(source: str) -> Set[str]:
    """Every name the code defines or uses: variables, attributes, functions, classes, arguments
    and imported names; comments and docstrings aside."""
    found: Set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, ast.alias):
            found.add((node.asname or node.name).split(".")[-1])
    return found


@pytest.mark.cpu_only
def test_the_package_computes_no_compute_identity():
    """Which instance computed a block is the caller's: the lender names blocks by layout, scope and
    reuse key alone."""
    offenders = {}
    for path in package_files():
        hits = sorted(n for n in identifiers(path.read_text()) if "compute_id" in n)
        if hits:
            offenders[path.name] = hits
    assert offenders == {}
    planted = '"""compute_id in prose is fine."""\ndef name(key, compute_id):\n    return key\n'
    assert sorted(n for n in identifiers(planted) if "compute_id" in n) == ["compute_id"]


# -- the public types' own rules --------------------------------------------------------------


def staging_run(n=3, layer_group=0, part=0):
    return GroupRun(
        layer_group,
        np.arange(n),
        np.zeros((n, 54), np.uint8),
        np.arange(n, dtype=np.int64) * 4096,
        part,
    )


@pytest.mark.cpu_only
def test_a_group_run_carries_all_three_placements_or_none():
    ordinals = np.arange(3)
    names = np.zeros((3, 54), np.uint8)
    addresses = np.arange(3, dtype=np.int64)
    for partial in ((names, None, None), (None, addresses, 0), (names, addresses, None)):
        with pytest.raises(ValueError):
            GroupRun(0, ordinals, *partial)
    in_place = GroupRun(1, ordinals)
    assert (in_place.names, in_place.addresses, in_place.part) == (None, None, None)
    assert len(in_place) == 3 and in_place.ordinals.dtype == np.int64


@pytest.mark.cpu_only
def test_a_group_run_checks_its_shapes():
    ordinals = np.arange(3)
    addresses = np.arange(3, dtype=np.int64)
    with pytest.raises(ValueError):
        GroupRun(0, np.zeros((3, 1)))
    with pytest.raises(ValueError):
        GroupRun(0, ordinals, np.zeros((3, 53), np.uint8), addresses, 0)
    with pytest.raises(ValueError):
        GroupRun(0, ordinals, np.zeros((3, 54), np.int8), addresses, 0)
    with pytest.raises(ValueError):
        GroupRun(0, ordinals, np.zeros((3, 54), np.uint8), addresses[:2], 0)
    with pytest.raises(ValueError):
        GroupRun(-1, ordinals)
    with pytest.raises(ValueError):
        GroupRun(0, ordinals, np.zeros((3, 54), np.uint8), addresses, -1)
    with pytest.raises(TypeError):
        GroupRun(True, ordinals)
    with pytest.raises(TypeError):
        GroupRun(0, ordinals, np.zeros((3, 54), np.uint8), addresses, 1.0)


@pytest.mark.cpu_only
def test_a_group_run_s_arrays_are_read_only_views():
    ordinals = np.arange(3, dtype=np.int64)
    run = GroupRun(0, ordinals, np.zeros((3, 54), np.uint8), np.arange(3, dtype=np.int64), 0)
    for array in (run.ordinals, run.names, run.addresses):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1
    assert ordinals.flags.writeable, "the caller's own array is left as it was"


@pytest.mark.cpu_only
def test_select_keeps_the_marked_rows_in_order():
    run = staging_run(4, layer_group=2, part=1)
    picked = run.select(np.array([True, False, True, True]))
    assert picked.ordinals.tolist() == [0, 2, 3]
    assert picked.addresses.tolist() == [0, 8192, 12288]
    assert picked.names.shape == (3, 54) and (picked.layer_group, picked.part) == (2, 1)
    assert GroupRun(0, np.arange(2)).select(np.array([False, True])).names is None
    with pytest.raises(ValueError):
        run.select(np.array([True, False]))
    with pytest.raises(ValueError):
        run.select(np.array([1, 0, 1, 1]))


@pytest.mark.cpu_only
def test_a_region_view_holds_one_run_per_layer_group():
    view = RegionView([staging_run(3, 0), staging_run(2, 1, part=1)])
    assert isinstance(view.runs, tuple) and view.num_rows == 5
    masks = view.row_masks()
    assert [m.shape for m in masks] == [(3,), (2,)]
    assert all(m.dtype == np.bool_ and not m.any() and m.flags.writeable for m in masks)
    assert all(m.all() for m in view.row_masks(True))
    assert RegionView(()).num_rows == 0 and RegionView(()).row_masks() == ()
    with pytest.raises(ValueError):
        RegionView((staging_run(3, 0), staging_run(1, 0)))
    with pytest.raises(TypeError):
        RegionView((object(),))


@pytest.mark.cpu_only
def test_staging_options_take_positive_integers():
    options = StagingOptions(256)
    assert (options.fetch_tokens, options.max_fetches, options.max_bytes) == (256, 1, None)
    assert StagingOptions(256, max_fetches=4, max_bytes=1 << 30).max_bytes == 1 << 30
    for bad in (dict(fetch_tokens=0), dict(fetch_tokens=8, max_fetches=0)):
        with pytest.raises(ValueError):
            StagingOptions(**bad)
    with pytest.raises(ValueError):
        StagingOptions(8, max_bytes=0)
    for bad in (
        dict(fetch_tokens=True),
        dict(fetch_tokens=8.0),
        dict(fetch_tokens=8, max_bytes=1.5),
    ):
        with pytest.raises(TypeError):
            StagingOptions(**bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.max_fetches = 2


@pytest.mark.cpu_only
def test_readiness_is_the_interval_in_field_order():
    readiness = Readiness(96, 32)
    assert readiness == (96, 32) and readiness.usable_until == 96 and readiness.restart_floor == 32
    assert Part("p", 4096, 8192, 4096, 2) == Part("p", 4096, 8192, 4096, 2)
    assert importlib.import_module(SH).Readiness is Readiness
