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
"""The lender's public surface and boundaries: the nine names, their members, what importing the
package loads, and who may import its private modules. No GPU."""

import ast
import dataclasses
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
    LayerGroupBlocks,
    Lease,
    LeaseView,
    Lender,
    MemoryRegion,
    RegionHold,
    ResumeBounds,
    StagingOptions,
    attach_lender,
)

SH = "tensorrt_llm._torch.pyexecutor.kv_cache.sharing"
MGR = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
PUBLIC = sorted(
    "LayerGroupBlocks Lease MemoryRegion RegionHold ResumeBounds LeaseView Lender StagingOptions "
    "attach_lender".split()
)
MODULES = [
    "__init__.py",
    "_fetch_ledger.py",
    "_identity.py",
    "_layout.py",
    "_lender.py",
    "_manager.py",
    "_ranges.py",
    "_slots.py",
]
MODULES = sorted(MODULES + ["_types.py"])
PKG_DIR = Path(sharing.__file__).resolve().parent
ROOT = Path(tensorrt_llm.__file__).resolve().parent
TESTS_DIR = Path(__file__).resolve().parent
# Manager and runtime members only the manager facade may touch.
INTERNALS = {
    "impl",
    "init_config",
    "layer_grouping",
    "pool_group_descs",
    "_has_lossy_cold_page_codec",
    "get_disagg_global_layer_ids",
    "get_disagg_role_mapper_kinds",
    "get_disagg_role_layouts",
    "get_disagg_ignored_roles",
    "get_life_cycle_pool_group_indices",
    "_reuse_token_source",
    "_augment_tokens_for_block_reuse",
    "_stale_block_range",
    "_resize_for_served_prefix",
    "_fill_fresh_kv_pages",
    "_fresh_page_fill",
    "_fresh_pages_filled",
    "_can_publish_block_reuse",
    "block_reuse_policy",
    "_draft_prompt_lookahead",
    "kv_connector_manager",
    "commit_min_snapshot",
    "reuse_match_backoff",
    "_stream",
    "_layer_attn_to_layer_id",
    "_sharing",
    "host_kv_cache_block_offsets",
    "kv_cache_map",
    "py_multimodal_data",
    "multimodal_hashes",
    "multimodal_positions",
    "multimodal_lengths",
    "try_get_encoder_output_len",
    "py_return_context_logits",
    "py_additional_outputs",
    "py_result",
    "additional_context_outputs",
    "prompt_len",
    "is_draft",
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


def compute_identities(source: str) -> List[str]:
    return sorted(n for n in identifiers(source) if "compute_id" in n)


# What no file of the package holds. Which instance computed a block is the caller's: the lender
# names blocks by layout, scope and reuse key alone.
PACKAGE_SCANS = {
    "disaggregation_imports": lambda path: disaggregation_imports(
        path.read_text(), package_of(path)
    ),
    "threads_locks_or_finalizers": lambda path: threads_or_finalizers(path.read_text()),
    "compute_identities": lambda path: compute_identities(path.read_text()),
}


def run_python(code: str) -> dict:
    """Run ``code`` in a new interpreter importing this tree's ``tensorrt_llm``; it prints JSON."""
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
@pytest.mark.parametrize("raw,backoff", [(0, 0), (1, 0), (1, 2)])
def test_the_manager_exposes_read_only_raw_lookahead_and_the_lender_keeps_both_guards(raw, backoff):
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    manager = object.__new__(KVCacheManagerV2)
    manager._draft_prompt_lookahead = raw
    manager.reuse_match_backoff = backoff
    assert manager.draft_prompt_lookahead == raw
    with pytest.raises(AttributeError):
        manager.draft_prompt_lookahead = 0
    assert _manager.prompt_lookahead(manager) == max(raw, backoff)


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


def subpackages(pkg_dir: Path) -> List[str]:
    """The directories under ``pkg_dir`` that could hold modules."""
    return [p.name for p in pkg_dir.iterdir() if p.is_dir() and p.name != "__pycache__"]


@pytest.mark.cpu_only
def test_the_package_is_its_init_and_eight_private_modules(tmp_path):
    assert [p.name for p in package_files()] == MODULES
    assert subpackages(PKG_DIR) == []
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "planted").mkdir()
    assert subpackages(tmp_path) == ["planted"], "the listing sees no subpackage"


@pytest.mark.cpu_only
def test_the_protocols_keep_their_members_and_stay_unrelated():
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    def members(cls):
        return {n for n in vars(cls) if not n.startswith("_")}

    assert members(Lease) == {"poll", "failure", "mark_arrived", "release"}
    assert members(Lender) == {
        "regions",
        "hold_regions",
        "lend_read",
        "lend_write",
        "resume_bounds",
    }
    assert members(RegionHold) == {"release"}
    assert Lease not in RegionHold.__mro__ and RegionHold not in Lease.__mro__
    for protocol in (Lease, Lender, RegionHold):
        assert Protocol in protocol.__mro__
        assert not isinstance(object(), protocol), "checkable at run time"
    assert isinstance(Lease.failure, property) and isinstance(Lender.regions, property)
    # The checks are structural: a lease passes as a hold.
    lease = object.__new__(_lender._StagingLease)
    assert isinstance(lease, RegionHold)

    def params(func):
        return list(inspect.signature(func).parameters)

    assert params(Lease.poll) == ["self"]
    assert params(Lease.mark_arrived) == ["self", "masks"]
    assert params(Lease.release) == ["self"]
    assert params(RegionHold.release) == ["self"]
    assert params(Lender.hold_regions) == ["self"]
    assert params(Lender.lend_read) == ["self", "request", "start", "end"]
    assert params(Lender.lend_write) == ["self", "request", "start", "end"]
    assert params(Lender.resume_bounds) == ["self", "request"]


def public_names(obj) -> Set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


LEASE_MEMBERS = {"poll", "failure", "mark_arrived", "release"}
STAGING_MEMBERS = {"regions", "hold_regions", "lend_read", "lend_write", "resume_bounds"}


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
        "staging lease": (_lender._StagingLease(owner, "read", 1), LEASE_MEMBERS),
        "staging hold": (_lender._StagingHold(owner), {"release"}),
    }
    assert {what: public_names(obj) for what, (obj, _) in shown.items()} == {
        what: members for what, (_, members) in shown.items()
    }
    # A lease showing its view as an attribute, and a lender showing a manager hook, are found.
    leaky = _lender._StagingLease(owner, "read", 1)
    leaky.view = None
    assert public_names(leaky) - LEASE_MEMBERS == {"view"}
    hooked = type("Hooked", (_lender.Staging,), {"on_free": _lender.Staging._on_free})
    assert public_names(hooked(ref, layout, None, (), None, ref)) - STAGING_MEMBERS == {"on_free"}


@pytest.mark.cpu_only
def test_the_types_keep_their_fields():
    def fields(cls):
        """``name`` or ``name=default`` per field, in order."""
        missing = dataclasses.MISSING
        return " ".join(
            f.name if f.default is missing else f"{f.name}={f.default!r}"
            for f in dataclasses.fields(cls)
        )

    assert fields(StagingOptions) == "tokens_per_fetch planned_fetches=1 max_bytes=None"
    assert (
        fields(MemoryRegion)
        == "layout_label address nbytes slot_bytes slot_count layer_group_indices"
    )
    assert (
        fields(LayerGroupBlocks)
        == "layer_group_index block_ordinals object_keys=None addresses=None region_index=None"
    )
    assert fields(LeaseView) == "groups"
    for cls in (StagingOptions, MemoryRegion, LayerGroupBlocks, LeaseView):
        assert cls.__dataclass_params__.frozen, cls
    assert MemoryRegion("p", 4096, 8192, 4096, 2, (0,)) == MemoryRegion(
        "p", 4096, 8192, 4096, 2, (0,)
    )
    assert ResumeBounds._fields == ("max_position", "min_position")
    assert issubclass(ResumeBounds, tuple)

    def beyond_fields(cls):
        named = {f.name for f in dataclasses.fields(cls)}
        return {n for n in dir(cls) if not n.startswith("_")} - named

    assert beyond_fields(LayerGroupBlocks) == {"select"}
    assert beyond_fields(LeaseView) == {"num_block_entries", "row_masks"}
    assert beyond_fields(MemoryRegion) == beyond_fields(StagingOptions) == set()
    assert {n for n in dir(ResumeBounds) if not n.startswith("_")} == {
        "max_position",
        "min_position",
        "count",
        "index",
    }


@pytest.mark.cpu_only
def test_public_names_are_independent_of_the_lending_mechanism():
    assert {name for name in sharing.__all__ if "Staging" in name} == {"StagingOptions"}
    for name in ("StagingLender", "StagingRegion", "StagingHold", "attach_staging"):
        assert not hasattr(sharing, name), name
    assert not hasattr(Lender, "hold_staging")
    assert not hasattr(LeaseView, "block_masks")
    assert not hasattr(LayerGroupBlocks, "staging_addresses")


@pytest.mark.cpu_only
def test_views_and_regions_match_the_backend_contract_member_names():
    """Exercise the structural surface without importing the unmerged backend contract."""
    region = MemoryRegion("layout", 4096, 128, 64, 2, (3,))
    group = LayerGroupBlocks(
        layer_group_index=3,
        block_ordinals=np.array([0, 7]),
        object_keys=np.zeros((2, 54), dtype=np.uint8),
        addresses=np.array([4096, 4160]),
        region_index=0,
    )
    view = LeaseView((group,))
    assert all(
        hasattr(region, name)
        for name in ("layout_label", "address", "nbytes", "slot_bytes", "slot_count")
    )
    assert all(
        hasattr(group, name)
        for name in ("layer_group_index", "object_keys", "addresses", "region_index", "__len__")
    )
    assert all(hasattr(view, name) for name in ("groups", "row_masks"))
    assert list(inspect.signature(LeaseView.row_masks).parameters) == ["self", "fill_value"]
    false_masks = view.row_masks()
    true_masks = view.row_masks(fill_value=True)
    assert len(group) == 2 and len(false_masks) == len(true_masks) == len(view.groups) == 1
    for masks, fill in ((false_masks, False), (true_masks, True)):
        assert masks[0].shape == (2,) and masks[0].dtype == np.bool_
        assert masks[0].flags.writeable and np.all(masks[0] == fill)
    assert region.nbytes == region.slot_bytes * region.slot_count
    assert np.all(group.addresses >= region.address)
    assert np.all(group.addresses + region.slot_bytes <= region.address + region.nbytes)


@pytest.mark.cpu_only
def test_interface_docstrings_state_guarantees_without_staging_mechanisms():
    forbidden = re.compile(r"\b(staging|stream|copy|copies|host|device|gpu)\b", re.IGNORECASE)
    for cls in (MemoryRegion, LayerGroupBlocks, LeaseView, ResumeBounds, Lease, RegionHold, Lender):
        entries = [cls] + [member for name, member in vars(cls).items() if not name.startswith("_")]
        for member in entries:
            doc = inspect.getdoc(member) or ""
            assert not forbidden.search(doc), (cls.__name__, doc)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "invalid_hook", ["_on_free", "_on_shrink", "_on_reset", "_on_shutdown", "_on_caches_closed"]
)
@pytest.mark.parametrize("replacement", [None, 1, "absent"])
def test_install_checks_every_private_lifecycle_hook_before_attaching(invalid_hook, replacement):
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    hooks = {
        name: lambda *args: None
        for name in ("_on_free", "_on_shrink", "_on_reset", "_on_shutdown", "_on_caches_closed")
    }
    if replacement == "absent":
        hooks.pop(invalid_hook)
    else:
        hooks[invalid_hook] = replacement
    manager = SimpleNamespace()
    with pytest.raises(TypeError, match="all five manager lifecycle hooks"):
        _manager.install_lender(manager, SimpleNamespace(**hooks))
    assert "_sharing" not in vars(manager)


@pytest.mark.cpu_only
def test_install_accepts_the_complete_private_lifecycle_protocol():
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    lender = SimpleNamespace(
        **{
            name: lambda *args: None
            for name in ("_on_free", "_on_shrink", "_on_reset", "_on_shutdown", "_on_caches_closed")
        }
    )
    manager = SimpleNamespace()
    assert isinstance(lender, _manager._ManagerHooks)
    _manager.install_lender(manager, lender)
    assert manager._sharing is lender


@pytest.mark.cpu_only
@pytest.mark.parametrize("config_source", ["manager", "runtime", "runtime_none"])
def test_manager_facade_reads_layer_configs_by_id_with_the_existing_fallback(config_source):
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    layers = [
        SimpleNamespace(layer_id="7"),
        SimpleNamespace(layer_id=4),
        SimpleNamespace(),
        SimpleNamespace(layer_id=0),
    ]
    config = SimpleNamespace(layers=layers)
    if config_source == "manager":
        manager = SimpleNamespace(kv_cache_manager_py_config=config)
    else:
        manager = SimpleNamespace(impl=SimpleNamespace(init_config=config))
        if config_source == "runtime_none":
            manager.kv_cache_manager_py_config = None
    actual = _manager.layer_configs(manager)
    assert list(actual) == [7, 4, 2, 0]
    assert all(actual[layer_id] is layer for layer_id, layer in zip([7, 4, 2, 0], layers))


@pytest.mark.cpu_only
def test_manager_facade_preserves_layer_group_order_and_normalizes_ids():
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    grouping = [[np.int64(5), "7"], [2]]
    manager = SimpleNamespace(impl=SimpleNamespace(layer_grouping=grouping))
    actual = _manager.layer_group_layer_ids(manager)
    assert actual == [[5, 7], [2]]
    assert all(type(layer_id) is int for group in actual for layer_id in group)
    assert actual is not grouping and all(a is not b for a, b in zip(actual, grouping))


@pytest.mark.cpu_only
def test_manager_facade_returns_the_runtime_pool_group_descriptors_unchanged():
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    descriptors = [object(), object()]
    manager = SimpleNamespace(impl=SimpleNamespace(pool_group_descs=descriptors))
    assert _manager.pool_group_descriptors(manager) is descriptors


@pytest.mark.cpu_only
def test_the_attach_functions_keep_their_signatures():
    kind = inspect.Parameter

    def shape(func):
        return [(p.name, p.kind, p.default) for p in inspect.signature(func).parameters.values()]

    assert shape(attach_lender) == [
        ("manager", kind.POSITIONAL_OR_KEYWORD, kind.empty),
        ("scope", kind.KEYWORD_ONLY, kind.empty),
        ("staging", kind.KEYWORD_ONLY, kind.empty),
    ]


@pytest.mark.cpu_only
def test_a_name_is_as_long_as_the_identity_makes_it():
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _identity, _types

    assert _types._OBJECT_KEY_BYTES == _identity.OBJECT_KEY_BYTES == 54


def public_lending_names(cls) -> List[str]:
    """The public names of ``cls`` about lending."""
    words = re.compile(r"lend|lent|sharing|staging|retain")
    return [n for n in dir(cls) if not n.startswith("_") and words.search(n)]


@pytest.mark.cpu_only
def test_the_manager_has_no_public_name_about_lending():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    assert "_sharing" in vars(KVCacheManagerV2) and KVCacheManagerV2._sharing is None
    assert public_lending_names(KVCacheManagerV2) == []
    planted = type("Planted", (KVCacheManagerV2,), {"lend_read": None, "staging_part": None})
    assert public_lending_names(planted) == ["lend_read", "staging_part"], "the scan sees nothing"


# -- boundaries -------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_nothing_outside_the_package_imports_its_private_modules():
    assert private_importers(ROOT, PKG_DIR) == {}


@pytest.mark.cpu_only
@pytest.mark.parametrize("scan", list(PACKAGE_SCANS))
def test_the_package_holds_nothing_a_scan_looks_for(scan):
    hits = {path.name: PACKAGE_SCANS[scan](path) for path in package_files()}
    assert {name: found for name, found in hits.items() if found} == {}


@pytest.mark.cpu_only
def test_the_manager_never_imports_the_package():
    path = ROOT / "_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py"
    source = path.read_text()
    loaded = imported_modules(source, package_of(path))
    assert [m for m in loaded if m.startswith(SH)] == []
    planted = source + "\nfrom .sharing import attach_lender\n"
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
def test_the_context_output_check_reads_the_request_alone():
    """The fetch's context-output check reads two request attributes and never the executor's
    result, whose additional outputs concatenate their chunks at each read."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    class Unread:
        def __getattr__(self, name):
            raise AssertionError(f"read py_result.{name}")

    class Request:
        py_result = Unread()

        def __init__(self, logits, outputs):
            self.py_return_context_logits, self.py_additional_outputs = logits, outputs

    cases = [(False, None, False), (False, [], False), (True, None, True), (False, ["x"], True)]
    for logits, outputs, returns in cases:
        assert _manager.returns_context_outputs(Request(logits, outputs)) is returns


# -- headers ----------------------------------------------------------------------------------


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


# -- the public types' own rules --------------------------------------------------------------


def staging_blocks(n=3, layer_group_index=0, region_index=0):
    return LayerGroupBlocks(
        layer_group_index,
        np.arange(n),
        np.repeat(np.arange(n, dtype=np.uint8)[:, None], 54, axis=1),
        np.arange(n, dtype=np.int64) * 4096,
        region_index,
    )


@pytest.mark.cpu_only
def test_layer_group_blocks_carry_all_three_placements_or_none_in_their_shapes():
    block_ordinals = np.arange(3)
    object_keys = np.zeros((3, 54), np.uint8)
    addresses = np.arange(3, dtype=np.int64)
    refused = [
        (ValueError, (0, block_ordinals, object_keys, None, None)),
        (ValueError, (0, block_ordinals, None, addresses, 0)),
        (ValueError, (0, block_ordinals, object_keys, addresses, None)),
        (ValueError, (0, np.zeros((3, 1)))),
        (ValueError, (0, block_ordinals, np.zeros((3, 53), np.uint8), addresses, 0)),
        (ValueError, (0, block_ordinals, np.zeros((3, 54), np.int8), addresses, 0)),
        (ValueError, (0, block_ordinals, object_keys, addresses[:2], 0)),
        (ValueError, (-1, block_ordinals)),
        (ValueError, (0, block_ordinals, object_keys, addresses, -1)),
        (TypeError, (True, block_ordinals)),
        (TypeError, (0, block_ordinals, object_keys, addresses, 1.0)),
    ]
    for error, args in refused:
        with pytest.raises(error):
            LayerGroupBlocks(*args)
    unplaced = LayerGroupBlocks(1, block_ordinals)
    assert (unplaced.object_keys, unplaced.addresses, unplaced.region_index) == (
        None,
        None,
        None,
    )
    assert len(unplaced) == 3 and unplaced.block_ordinals.dtype == np.int64


@pytest.mark.cpu_only
def test_layer_group_blocks_arrays_are_read_only_views():
    block_ordinals = np.arange(3, dtype=np.int64)
    blocks = LayerGroupBlocks(
        0, block_ordinals, np.zeros((3, 54), np.uint8), np.arange(3, dtype=np.int64), 0
    )
    for array in (blocks.block_ordinals, blocks.object_keys, blocks.addresses):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array[0] = 1
    assert block_ordinals.flags.writeable, "the caller's own array is left as it was"


@pytest.mark.cpu_only
def test_select_keeps_the_marked_block_entries_in_order():
    blocks = staging_blocks(4, layer_group_index=2, region_index=1)
    picked = blocks.select(np.array([True, False, True, True]))
    assert picked.block_ordinals.tolist() == [0, 2, 3]
    assert picked.addresses.tolist() == [0, 8192, 12288]
    np.testing.assert_array_equal(picked.object_keys, blocks.object_keys[[0, 2, 3]])
    assert picked.object_keys.shape == (3, 54) and (
        picked.layer_group_index,
        picked.region_index,
    ) == (2, 1)
    assert LayerGroupBlocks(0, np.arange(2)).select(np.array([False, True])).object_keys is None
    with pytest.raises(ValueError):
        blocks.select(np.array([True, False]))
    with pytest.raises(ValueError):
        blocks.select(np.array([1, 0, 1, 1]))


@pytest.mark.cpu_only
def test_a_lease_view_holds_one_block_collection_per_layer_group():
    view = LeaseView([staging_blocks(3, 0), staging_blocks(2, 1, region_index=1)])
    assert isinstance(view.groups, tuple) and view.num_block_entries == 5
    masks = view.row_masks()
    assert [m.shape for m in masks] == [(3,), (2,)]
    assert all(m.dtype == np.bool_ and not m.any() and m.flags.writeable for m in masks)
    assert all(m.all() for m in view.row_masks(True))
    assert LeaseView(()).num_block_entries == 0 and LeaseView(()).row_masks() == ()
    with pytest.raises(ValueError):
        LeaseView((staging_blocks(3, 0), staging_blocks(1, 0)))
    with pytest.raises(TypeError):
        LeaseView((object(),))


@pytest.mark.cpu_only
def test_sparse_block_entries_keep_group_position_masks_and_placement_alignment():
    keys = np.arange(3 * 54, dtype=np.uint8).reshape(3, 54)
    blocks = LayerGroupBlocks(
        layer_group_index=7,
        block_ordinals=np.array([0, 3, 9]),
        object_keys=keys,
        addresses=np.array([12288, 4096, 8192]),
        region_index=2,
    )
    empty = LayerGroupBlocks(layer_group_index=2, block_ordinals=np.array([], dtype=np.int64))
    repeated = LayerGroupBlocks(layer_group_index=4, block_ordinals=np.array([3, 9]))
    view = LeaseView(groups=(blocks, empty, repeated))
    # Five group/block entries, but only three distinct token block ordinals.
    assert view.num_block_entries == 5
    masks = view.row_masks()
    assert [mask.shape for mask in masks] == [(3,), (0,), (2,)]
    masks[0][1] = True
    picked = view.groups[0].select(masks[0])
    assert picked.block_ordinals.tolist() == [3]
    assert picked.addresses.tolist() == [4096]
    np.testing.assert_array_equal(picked.object_keys, keys[[1]])
    assert (picked.layer_group_index, picked.region_index) == (7, 2)
    assert not masks[2].any()


@pytest.mark.cpu_only
def test_renamed_bounds_keep_maximum_first_and_region_fields_accept_keywords():
    bounds = ResumeBounds(max_position=96, min_position=32)
    assert bounds._fields == ("max_position", "min_position")
    assert tuple(bounds) == (96, 32)
    assert ResumeBounds(0, 1).min_position > ResumeBounds(0, 1).max_position
    region = MemoryRegion(
        layout_label="layout:lg0+1",
        address=4096,
        nbytes=8192,
        slot_bytes=4096,
        slot_count=2,
        layer_group_indices=(7, 2),
    )
    assert region.layout_label == "layout:lg0+1"
    assert region.layer_group_indices == (7, 2)


@pytest.mark.cpu_only
def test_staging_options_take_positive_integers():
    options = StagingOptions(256)
    assert (options.tokens_per_fetch, options.planned_fetches, options.max_bytes) == (256, 1, None)
    assert StagingOptions(256, planned_fetches=4, max_bytes=1 << 30).max_bytes == 1 << 30
    for bad in (dict(tokens_per_fetch=0), dict(tokens_per_fetch=8, planned_fetches=0)):
        with pytest.raises(ValueError):
            StagingOptions(**bad)
    with pytest.raises(ValueError):
        StagingOptions(8, max_bytes=0)
    for bad in (
        dict(tokens_per_fetch=True),
        dict(tokens_per_fetch=8.0),
        dict(tokens_per_fetch=8, max_bytes=1.5),
    ):
        with pytest.raises(TypeError):
            StagingOptions(**bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.planned_fetches = 2
