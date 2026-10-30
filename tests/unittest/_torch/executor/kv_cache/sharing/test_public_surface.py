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
"""The lender's public surface and boundaries: the twelve names, their members, what importing the
package loads, and who may import its private modules. No GPU."""

import ast
import dataclasses
import enum
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
    ALL_ARRIVED,
    HeldPages,
    HoldReason,
    Lease,
    LeaseView,
    Lender,
    LentLayerGroupRows,
    MemoryKind,
    MemoryRegion,
    RegionHold,
    ResumeBounds,
    RowSelection,
    StagingOptions,
    attach_lender,
)

SH = "tensorrt_llm._torch.pyexecutor.kv_cache.sharing"
MGR = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
PUBLIC = sorted(
    "ALL_ARRIVED Arrived HeldPages HoldReason LentLayerGroupRows Lease MemoryKind MemoryRegion "
    "RegionHold ResumeBounds RowSelection LeaseView Lender StagingOptions attach_lender".split()
)
MODULES = [
    "__init__.py",
    "_fetch_ledger.py",
    "_identity.py",
    "_layout.py",
    "_lender.py",
    "_manager.py",
    "_page_locks.py",
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
    "kv_cache_type",
    "init_config",
    "layer_grouping",
    "pool_group_descs",
    "_has_lossy_cold_page_codec",
    # The methods a manager's layout adapter builds its page table with; the lender reads the
    # table the adapter returns.
    "buffer_tensors",
    "head_split",
    "state_split",
    "untransferred_roles",
    "derived_roles",
    "layer_ids",
    "kv_heads",
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
    "capacity_only",
    "commit_min_snapshot",
    "reuse_match_backoff",
    "_stream",
    "_layer_attn_to_layer_id",
    "use_fp8_ds_mla",
    "_attention_read_window",
    "_sharing",
    "_commit_held_back",
    "take_disagg_receive_ready",
    "reuse_status",
    "_has_cp_helix",
    "_helix_cp_rank",
    "_helix_cp_size",
    "total_input_len_cp",
    "get_ssm_block_base_index",
    "py_served_prefix_end",
    "has_pending_load",
    "get_prefix_reservation",
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


# The one module of the transfer contract the package imports: the kinds of memory it lends.
CONTRACT_MEMORY = "tensorrt_llm._torch.disaggregation.base.memory"


def disaggregation_imports(source: str, package: str) -> List[str]:
    """Disaggregation modules a source imports, apart from the contract's memory module."""
    prefix = "tensorrt_llm._torch.disaggregation"
    return sorted(
        m
        for m in imported_modules(source, package)
        if m.startswith(prefix) and CONTRACT_MEMORY not in (m, m.rpartition(".")[0])
    )


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
def test_the_package_is_its_init_and_nine_private_modules(tmp_path):
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

    assert members(Lease) == {"poll", "failure", "writable", "release", "return_resumes_at_end"}
    assert members(Lender) == {
        "regions",
        "page_layout",
        "held_page_counts",
        "hold_regions",
        "lend_read",
        "lend_write",
        "return_writes",
        "progress",
        "poll_resume_bounds",
    }
    assert members(RegionHold) == {"release"}
    assert Lease not in RegionHold.__mro__ and RegionHold not in Lease.__mro__
    for protocol in (Lease, Lender, RegionHold):
        assert Protocol in protocol.__mro__
        assert not isinstance(object(), protocol), "checkable at run time"
    assert isinstance(Lease.failure, property) and isinstance(Lender.regions, property)
    assert isinstance(Lease.return_resumes_at_end, property)
    assert isinstance(Lease.writable, property)
    assert isinstance(Lender.page_layout, property)
    # The checks are structural: a lease passes as a hold.
    lease = object.__new__(_lender._StagingLease)
    assert isinstance(lease, RegionHold)

    def params(func):
        return list(inspect.signature(func).parameters)

    assert params(Lease.poll) == ["self"]
    assert params(Lease.release) == ["self"]
    assert params(RegionHold.release) == ["self"]
    assert params(Lender.hold_regions) == ["self"]
    assert params(Lender.held_page_counts) == ["self"]

    def shape(func):
        """The signature, annotated on its keyword-only parameters alone."""
        signature = inspect.signature(func)
        kept = [
            p if p.kind is p.KEYWORD_ONLY else p.replace(annotation=p.empty)
            for p in signature.parameters.values()
        ]
        return str(signature.replace(parameters=kept, return_annotation=signature.empty))

    keywords = "*, memory_kind: 'MemoryKind', rows: 'RowSelection', ready_event"
    for cls, event in (
        (Lender, "'Optional[torch.cuda.Event]'"),
        (_lender._ManagerLender, "'torch.cuda.Event | None'"),
    ):
        lend = f"(self, request, start, end, {keywords}: {event})"
        assert shape(cls.lend_read) == shape(cls.lend_write) == lend, cls
        assert [shape(cls.return_writes), shape(cls.progress)] == ["(self, entries)", "(self)"], cls
        assert params(cls.poll_resume_bounds) == ["self", "request"], cls


@pytest.mark.cpu_only
def test_all_arrived_is_one_value_that_pickles_and_copies_to_itself():
    import copy
    import pickle

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _types

    assert sharing.ALL_ARRIVED is ALL_ARRIVED is _types.ALL_ARRIVED
    assert type(ALL_ARRIVED)() is ALL_ARRIVED, "the type makes a second value"
    assert pickle.loads(pickle.dumps(ALL_ARRIVED)) is ALL_ARRIVED
    assert copy.copy(ALL_ARRIVED) is ALL_ARRIVED and copy.deepcopy(ALL_ARRIVED) is ALL_ARRIVED
    assert repr(ALL_ARRIVED) == "ALL_ARRIVED"
    assert not hasattr(ALL_ARRIVED, "__dict__"), "it carries state"
    assert not isinstance(ALL_ARRIVED, (tuple, list, np.ndarray)), "it reads as masks"


def public_names(obj) -> Set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


LEASE_MEMBERS = {"poll", "failure", "writable", "release", "return_resumes_at_end"}
LENDER_MEMBERS = {
    "regions",
    "page_layout",
    "held_page_counts",
    "hold_regions",
    "lend_read",
    "lend_write",
    "return_writes",
    "progress",
    "poll_resume_bounds",
}


@pytest.mark.cpu_only
def test_lenders_leases_and_holds_show_only_their_protocol():
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    class Owner:  # anything a weak reference can point to
        pass

    owner = Owner()
    ref = weakref.ref(owner)
    # What constructors read.
    layout = SimpleNamespace(pool_groups=(), windows=(), named_groups=(), tokens_per_block=1)
    shown = {
        "lender": (_lender._ManagerLender(ref, layout, None, (), None, ref), LENDER_MEMBERS),
        "staging lease": (_lender._StagingLease(owner, "read", 1), LEASE_MEMBERS),
        "region hold": (_lender._RegionHold(owner), {"release"}),
        "in-place lease": (_lender._InPlaceLease(owner, "read", 1), LEASE_MEMBERS),
    }
    assert {what: public_names(obj) for what, (obj, _) in shown.items()} == {
        what: members for what, (_, members) in shown.items()
    }
    # A lease showing its view as an attribute, and a lender showing a manager hook, are found.
    leaky = _lender._StagingLease(owner, "read", 1)
    leaky.view = None
    assert public_names(leaky) - LEASE_MEMBERS == {"view"}
    hooked = type("Hooked", (_lender._ManagerLender,), {"on_free": _lender._ManagerLender._on_free})
    assert public_names(hooked(ref, layout, None, (), None, ref)) - LENDER_MEMBERS == {"on_free"}


@pytest.mark.cpu_only
def test_the_types_keep_their_fields():
    def fields(cls):
        """``name`` or ``name=default`` per field, in order."""
        missing = dataclasses.MISSING
        return " ".join(
            f.name if f.default is missing else f"{f.name}={f.default!r}"
            for f in dataclasses.fields(cls)
        )

    assert fields(StagingOptions) == "guaranteed_fetch_tokens capacity_in_fetches=1 max_bytes=None"
    assert (
        fields(MemoryRegion)
        == "layout_label address nbytes slot_bytes slot_count layer_group_indices memory_kind"
    )
    assert fields(LentLayerGroupRows) == (
        "layer_group_index page_ordinals object_keys=None slots=None region_indices=None"
    )
    assert fields(LeaseView) == "groups start end"
    assert fields(HeldPages) == "deduct unreclaimable by_reason version"
    assert [reason.name for reason in HoldReason] == ["LOCKED", "DEFERRED_CLOSE"]
    for cls in (StagingOptions, MemoryRegion, LentLayerGroupRows, LeaseView, HeldPages):
        assert cls.__dataclass_params__.frozen, cls
    assert MemoryRegion("p", 4096, 8192, 4096, 2, (0,), MemoryKind.HOST) == MemoryRegion(
        "p", 4096, 8192, 4096, 2, (0,), MemoryKind.HOST
    )
    assert ResumeBounds._fields == ("max_position", "min_position")
    assert issubclass(ResumeBounds, tuple)

    def beyond_fields(cls):
        named = {f.name for f in dataclasses.fields(cls)}
        return {n for n in dir(cls) if not n.startswith("_")} - named

    assert beyond_fields(LentLayerGroupRows) == {"select"}
    assert beyond_fields(LeaseView) == {"num_rows", "row_masks"}
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
    assert not hasattr(LentLayerGroupRows, "staging_addresses")


@pytest.mark.cpu_only
def test_rows_views_and_regions_carry_the_backend_contract_s_members():
    """Rows, views and regions are what the backend contract describes, and rows carry no member
    the contract leaves out."""
    from tensorrt_llm._torch.disaggregation.base import backend as contract

    region = MemoryRegion("layout", 4096, 128, 64, 2, (3,), MemoryKind.HOST)
    group = LentLayerGroupRows(
        layer_group_index=3,
        page_ordinals=np.array([0, 7]),
        object_keys=np.zeros((2, 54), dtype=np.uint8),
        slots=np.array([0, 1]),
        region_indices=(0,),
    )
    view = LeaseView((group,), 0, 0)
    assert isinstance(region, contract.SlotRegion) and isinstance(view, contract.View)
    assert isinstance(group, contract.LayerGroupRows)
    members = {name for name in vars(contract.LayerGroupRows) if not name.startswith("_")}
    assert {f.name for f in dataclasses.fields(LentLayerGroupRows)} == members
    assert list(inspect.signature(LeaseView.row_masks).parameters) == ["self", "fill_value"]
    false_masks = view.row_masks()
    true_masks = view.row_masks(fill_value=True)
    assert len(group) == 2 and len(false_masks) == len(true_masks) == len(view.groups) == 1
    for masks, fill in ((false_masks, False), (true_masks, True)):
        assert masks[0].shape == (2,) and masks[0].dtype == np.bool_
        assert masks[0].flags.writeable and np.all(masks[0] == fill)
    assert region.nbytes == region.slot_bytes * region.slot_count
    assert np.all(group.slots < region.slot_count)


@pytest.mark.cpu_only
def test_interface_docstrings_state_guarantees_without_staging_mechanisms():
    forbidden = re.compile(r"\b(staging|stream|copy|copies|host|device|gpu)\b", re.IGNORECASE)
    for cls in (
        MemoryRegion,
        LentLayerGroupRows,
        LeaseView,
        ResumeBounds,
        Lease,
        RegionHold,
        Lender,
        RowSelection,
        HeldPages,
        HoldReason,
    ):
        entries = [cls] + [member for name, member in vars(cls).items() if not name.startswith("_")]
        for member in entries:
            # A trailing non-normative performance note may name mechanisms; guarantees precede it.
            doc = (inspect.getdoc(member) or "").split("Performance (non-normative):")[0]
            assert not forbidden.search(doc), (cls.__name__, doc)


# The private hooks the manager calls on its lender.
MANAGER_HOOKS = (
    "_on_free",
    "_on_shrink",
    "_on_resume",
    "_parking_row",
    "_parked_row",
    "_before_reset",
    "_on_reset",
    "_on_shutdown",
    "_on_caches_closed",
    "_is_locked",
    "_defer_close",
    "_keeps_pools",
    "_history_cap",
    "_page_lock_state",
)


@pytest.mark.cpu_only
@pytest.mark.parametrize("invalid_hook", MANAGER_HOOKS)
@pytest.mark.parametrize("replacement", [None, 1, "absent"])
def test_install_checks_every_private_manager_hook_before_attaching(invalid_hook, replacement):
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _manager

    hooks = {name: lambda *args: None for name in MANAGER_HOOKS}
    if replacement == "absent":
        hooks.pop(invalid_hook)
    else:
        hooks[invalid_hook] = replacement
    manager = SimpleNamespace()
    with pytest.raises(TypeError, match="all manager hooks"):
        _manager.install_lender(manager, SimpleNamespace(**hooks))
    assert "_sharing" not in vars(manager)


@pytest.mark.cpu_only
def test_install_accepts_the_complete_private_hook_protocol():
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache import _lender_hooks
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender, _manager

    lender = SimpleNamespace(**{name: lambda *args: None for name in MANAGER_HOOKS})
    manager = SimpleNamespace()
    assert isinstance(lender, _lender_hooks._ManagerHooks)
    assert set(MANAGER_HOOKS) <= set(vars(_lender_hooks._ManagerHooks)), "outside the protocol"
    assert all(callable(getattr(_lender._ManagerLender, name)) for name in MANAGER_HOOKS)
    _manager.install_lender(manager, lender)
    assert manager._sharing is lender


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


def class_definitions(source: str, name: str) -> int:
    """How many classes called ``name`` a source defines, nested ones included."""
    tree = ast.parse(source)
    return sum(isinstance(node, ast.ClassDef) and node.name == name for node in ast.walk(tree))


@pytest.mark.cpu_only
def test_the_package_and_the_contract_share_one_memory_kind():
    from tensorrt_llm._torch.disaggregation.base import memory
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender, _types

    for holder in (sharing, _types, _lender):
        assert holder.MemoryKind is memory.MemoryKind, holder.__name__
    assert MemoryKind.__module__ == memory.__name__ == CONTRACT_MEMORY
    assert memory.__all__ == ["MemoryKind"]
    assert [p.name for p in package_files() if class_definitions(p.read_text(), "MemoryKind")] == []


@pytest.mark.cpu_only
@pytest.mark.parametrize("first", ["lender", "base_memory"])
def test_the_lender_and_the_contract_s_memory_module_import_in_either_order(first):
    order = [f"{SH}._lender", CONTRACT_MEMORY]
    if first == "base_memory":
        order.reverse()
    result = run_python(
        "import json, sys\n"
        f"import {order[0]}\n"
        f"loaded = sorted(m for m in sys.modules if m.startswith('{SH}'))\n"
        f"import {order[1]}\n"
        f"s, memory = sys.modules['{SH}'], sys.modules['{CONTRACT_MEMORY}']\n"
        f"same = [m.MemoryKind is memory.MemoryKind for m in (s, sys.modules['{SH}._lender'])]\n"
        "print(json.dumps({'file': s.__file__, 'loaded': loaded, 'same': same}))\n"
    )
    assert result["same"] == [True, True]
    if first == "base_memory":
        assert result["loaded"] == [], "importing the contract's memory module loads the lender"


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


# The manager side's one module that calls a lender's hooks, and the manager's own module.
HOOKS_MODULE = ROOT / "_torch/pyexecutor/kv_cache/_lender_hooks.py"
MANAGER_MODULE = ROOT / "_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py"
# A manager's records of lending; the first belongs to the manager and the hooks module alone.
LENDING_RECORDS = ("_held_commits", "_fresh_pages_filled")
# The methods that change a dict.
DICT_WRITES = {"pop", "popitem", "setdefault", "clear", "update", "__setitem__", "__delitem__"}


def hook_uses(source: str) -> List[str]:
    """The lender's manager hooks a source names as attributes."""
    return sorted(
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute) and node.attr in MANAGER_HOOKS
    )


def record_writes(source: str, records) -> List[str]:
    """``record:line`` of each assignment, ``del`` or dict update a source makes to ``records``."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        written = None
        if isinstance(node, (ast.Attribute, ast.Subscript)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            written = node if isinstance(node, ast.Attribute) else node.value
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in DICT_WRITES
        ):
            written = node.func.value
        if isinstance(written, ast.Attribute) and written.attr in records:
            hits.append(f"{written.attr}:{node.lineno}")
    return sorted(hits)


def scan_outside_the_package(names, scan) -> dict:
    """Non-empty ``scan(source)`` of each other module outside the package naming ``names``."""
    found = {}
    for path in ROOT.rglob("*.py"):
        resolved = path.resolve()
        if PKG_DIR in resolved.parents or resolved in (HOOKS_MODULE, MANAGER_MODULE):
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        if not any(name in source for name in names):
            continue
        try:
            hits = scan(source)
        except SyntaxError:
            continue  # not importable, so it runs nothing
        if hits:
            found[path.relative_to(ROOT).as_posix()] = hits
    return found


@pytest.mark.cpu_only
def test_the_manager_side_calls_a_lender_s_hooks_through_the_hooks_module_alone():
    """Outside the package only the hooks module names a lender's manager hooks."""
    assert hook_uses(MANAGER_MODULE.read_text()) == []
    assert scan_outside_the_package(MANAGER_HOOKS, hook_uses) == {}
    assert set(hook_uses(HOOKS_MODULE.read_text())) == set(MANAGER_HOOKS)


@pytest.mark.cpu_only
def test_a_manager_s_lending_records_are_written_on_the_manager_side_alone():
    """Only the manager's module and the hooks module write a manager's lending records."""
    package = {
        path.name: record_writes(path.read_text(), LENDING_RECORDS) for path in package_files()
    }
    assert {name: hits for name, hits in package.items() if hits} == {}
    lender_state = LENDING_RECORDS[:1]
    found = scan_outside_the_package(lender_state, lambda s: record_writes(s, lender_state))
    assert found == {}


@pytest.mark.cpu_only
def test_the_manager_records_no_readiness_event_for_a_lender():
    """No module records readiness for a lender: it takes its readiness from the caller."""
    hooks = HOOKS_MODULE.read_text()
    for gone in ("record_context_ready", "discard_context_ready", "record_receive_after_reset"):
        assert gone not in hooks, gone
    for path in (MANAGER_MODULE, ROOT / "_torch/pyexecutor/kv_cache/mamba_cache_manager.py"):
        source = path.read_text()
        assert "_context_ready" not in source and "record_receive_after_reset" not in source
    package = {path.name: path.read_text() for path in package_files()}
    for name, source in package.items():
        for gone in ("take_disagg_receive_ready", "_context_ready", "record_event"):
            assert gone not in source, (name, gone)


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
    contract_memory = ROOT.parent.joinpath(*CONTRACT_MEMORY.split(".")).with_suffix(".py")
    return package_files() + [contract_memory] + sorted(TESTS_DIR.glob("*.py"))


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


def staged_rows(n=3, layer_group_index=0, region=0):
    """``n`` named rows in slots ``0 .. n - 1`` of region ``region``."""
    return LentLayerGroupRows(
        layer_group_index,
        np.arange(n),
        np.repeat(np.arange(n, dtype=np.uint8)[:, None], 54, axis=1),
        np.arange(n, dtype=np.int64),
        (region,),
    )


@pytest.mark.cpu_only
def test_memory_kinds_and_row_selections_are_plain_enums():
    assert [(m.name, m.value) for m in MemoryKind] == [("HOST", "host"), ("DEVICE", "device")]
    assert [(m.name, m.value) for m in RowSelection] == [("NAMED", "named"), ("ALL", "all")]
    assert set(MemoryKind.__members__) == {"HOST", "DEVICE"}
    assert set(RowSelection.__members__) == {"NAMED", "ALL"}
    for cls in (MemoryKind, RowSelection):
        assert cls.__bases__ == (enum.Enum,), "a member would equal its value"
    assert MemoryKind.HOST != "host" and RowSelection.NAMED != "named"
    assert MemoryKind("device") is MemoryKind.DEVICE and RowSelection("all") is RowSelection.ALL


@pytest.mark.cpu_only
def test_a_region_carries_one_memory_kind():
    for memory in MemoryKind:
        assert MemoryRegion("p", 4096, 8192, 4096, 2, (0,), memory).memory_kind is memory
    host, device = (MemoryRegion("p", 4096, 8192, 4096, 2, (0,), m) for m in MemoryKind)
    assert host != device
    for wrong in ("host", None, 0, RowSelection.NAMED):
        with pytest.raises(TypeError, match="memory_kind must be a MemoryKind"):
            MemoryRegion("p", 4096, 8192, 4096, 2, (0,), wrong)
    with pytest.raises(TypeError):
        MemoryRegion("p", 4096, 8192, 4096, 2, (0,))  # no default


_REGION = dict(
    layout_label="p",
    address=4096,
    nbytes=8192,
    slot_bytes=4096,
    slot_count=2,
    layer_group_indices=(0,),
    memory_kind=MemoryKind.HOST,
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "field, value, error",
    [
        ("address", True, TypeError),
        ("nbytes", 8192.0, TypeError),
        ("address", 0, ValueError),
        ("nbytes", 0, ValueError),
        ("slot_bytes", -1, ValueError),
        ("slot_count", 0, ValueError),
        ("slot_count", 3, ValueError),
        ("address", (1 << 63) - 8192, ValueError),
        ("layer_group_indices", [0], TypeError),
        ("layer_group_indices", (0, -1), ValueError),
        ("layer_group_indices", (False,), TypeError),
    ],
)
def test_a_region_with_a_field_out_of_its_range_is_refused(field, value, error):
    with pytest.raises(error):
        MemoryRegion(**{**_REGION, field: value})


@pytest.mark.cpu_only
def test_a_region_keeps_its_integers_as_ints_up_to_the_int64_end():
    end = (1 << 63) - 1
    region = MemoryRegion(**{**_REGION, "address": np.int64(end - 8192)})
    assert type(region.address) is int and region.address + region.nbytes == end
    indices = MemoryRegion(
        **{**_REGION, "layer_group_indices": (np.int32(2), 0)}
    ).layer_group_indices
    assert indices == (2, 0) and all(type(index) is int for index in indices)


@pytest.mark.cpu_only
def test_regions_of_one_memory_kind_that_overlap_by_a_byte_are_refused():
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _types

    def region(address, kind=MemoryKind.DEVICE):
        return MemoryRegion("r", address, 4096, 4096, 1, (0,), kind)

    _types._check_disjoint([region(8192), region(4096), region(4096, MemoryKind.HOST)])
    overlapping = [region(1 << 20), region(4096, MemoryKind.HOST), region(4096), region(8191)]
    with pytest.raises(ValueError, match="regions 2 and 3 overlap in DEVICE memory"):
        _types._check_disjoint(overlapping)


@pytest.mark.cpu_only
def test_layer_group_rows_take_slots_with_their_regions_or_neither_and_names_apart():
    page_ordinals = np.arange(3)
    object_keys = np.zeros((3, 54), np.uint8)
    slots = np.array([5, 0, 2])
    refused = [
        # Slots without their regions, or regions without slots.
        (ValueError, dict(slots=slots)),
        (ValueError, dict(object_keys=object_keys, region_indices=(0,))),
        # Shapes, dtypes and values, object_keys checked with slots or without.
        (ValueError, dict(page_ordinals=np.zeros((3, 1)))),
        (ValueError, dict(object_keys=np.zeros((3, 53), np.uint8))),
        (ValueError, dict(object_keys=np.zeros((3, 54), np.int8))),
        (ValueError, dict(object_keys=object_keys[:2], slots=slots, region_indices=(0,))),
        (ValueError, dict(layer_group_index=-1)),
        (ValueError, dict(slots=slots[:2], region_indices=(0,))),
        (ValueError, dict(slots=slots.astype(np.float64), region_indices=(0,))),
        (ValueError, dict(slots=np.array([1, -1, 2]), region_indices=(0,))),
        (ValueError, dict(slots=slots, region_indices=())),
        (ValueError, dict(slots=slots, region_indices=(1, 1))),
        (ValueError, dict(slots=slots, region_indices=(0, -1))),
        (TypeError, dict(layer_group_index=True)),
        (TypeError, dict(slots=slots, region_indices=(1.0,))),
        (TypeError, dict(slots=slots, region_indices=(True,))),
        (TypeError, dict(slots=slots, region_indices=1)),
    ]
    for error, fields in refused:
        with pytest.raises(error):
            LentLayerGroupRows(**{"layer_group_index": 0, "page_ordinals": page_ordinals, **fields})
    unplaced = LentLayerGroupRows(1, page_ordinals)
    assert len(unplaced) == 3 and unplaced.page_ordinals.dtype == np.int64
    placements = ("object_keys", "slots", "region_indices")
    assert [getattr(unplaced, name) for name in placements] == [None] * 3
    named = LentLayerGroupRows(1, page_ordinals, object_keys)
    assert named.object_keys.shape == (3, 54) and (named.slots, named.region_indices) == (
        None,
        None,
    )
    by_slot = LentLayerGroupRows(
        1, page_ordinals, slots=slots.astype(np.int32), region_indices=[np.int64(4), 1]
    )
    assert by_slot.object_keys is None
    assert by_slot.slots.dtype == np.int64 and by_slot.slots.tolist() == [5, 0, 2]
    assert by_slot.region_indices == (4, 1)
    assert all(type(index) is int for index in by_slot.region_indices)
    keyed = LentLayerGroupRows(1, page_ordinals, object_keys, slots=slots, region_indices=(0,))
    assert keyed.object_keys.shape == (3, 54) and keyed.region_indices == (0,)


def check_immutable(rows, *writable):
    """No array of ``rows`` can be written or made writable, or shares memory with ``writable``."""
    arrays = [
        getattr(rows, name)
        for name in ("page_ordinals", "object_keys", "slots")
        if getattr(rows, name) is not None
    ]
    for array in arrays:
        with pytest.raises(ValueError):
            array.flags.writeable = True
        with pytest.raises(ValueError):
            array[0] = 1
        assert not any(np.shares_memory(array, other) for other in writable)


@pytest.mark.cpu_only
def test_layer_group_rows_arrays_are_immutable_copies_of_writable_ones():
    page_ordinals = np.arange(3, dtype=np.int64)
    keys = np.zeros((3, 54), np.uint8)
    staging_slots = np.arange(3, dtype=np.int64)
    slots = np.array([4, 0, 9], dtype=np.int64)
    staged = LentLayerGroupRows(0, page_ordinals, keys, staging_slots, (0,))
    by_slot = LentLayerGroupRows(0, page_ordinals, keys, slots=slots, region_indices=(1, 0))
    check_immutable(staged, page_ordinals, keys, staging_slots)
    check_immutable(by_slot, page_ordinals, keys, slots)
    assert all(a.flags.writeable for a in (page_ordinals, keys, staging_slots, slots))
    # A caller's read-only view of its writable array is copied too.
    frozen = page_ordinals.view()
    frozen.flags.writeable = False
    check_immutable(LentLayerGroupRows(0, frozen), page_ordinals)
    # Arrays already immutable are kept, not copied again.
    kept = LentLayerGroupRows(0, by_slot.page_ordinals, slots=by_slot.slots, region_indices=(0,))
    assert kept.page_ordinals is by_slot.page_ordinals and kept.slots is by_slot.slots
    for picked in (
        staged.select(np.array([True, False, True])),
        by_slot.select(np.array([False, True, True])),
    ):
        check_immutable(picked, page_ordinals, keys, staging_slots, slots)


@pytest.mark.cpu_only
def test_select_keeps_the_marked_rows_in_order():
    rows = staged_rows(4, layer_group_index=2, region=1)
    picked = rows.select(np.array([True, False, True, True]))
    assert picked.page_ordinals.tolist() == [0, 2, 3]
    assert picked.slots.tolist() == [0, 2, 3]
    np.testing.assert_array_equal(picked.object_keys, rows.object_keys[[0, 2, 3]])
    assert picked.object_keys.shape == (3, 54) and picked.layer_group_index == 2
    assert picked.region_indices == (1,)
    assert LentLayerGroupRows(0, np.arange(2)).select(np.array([False, True])).object_keys is None
    with pytest.raises(ValueError):
        rows.select(np.array([True, False]))
    with pytest.raises(ValueError):
        rows.select(np.array([1, 0, 1, 1]))


@pytest.mark.cpu_only
def test_select_keeps_the_slots_aligned_and_their_regions():
    keys = np.arange(4 * 54, dtype=np.uint8).reshape(4, 54)
    by_slot = LentLayerGroupRows(
        3, np.array([0, 1, 4, 5]), keys, slots=np.array([7, 3, 0, 9]), region_indices=(2, 0)
    )
    picked = by_slot.select(np.array([False, True, True, False]))
    assert picked.page_ordinals.tolist() == [1, 4] and picked.slots.tolist() == [3, 0]
    np.testing.assert_array_equal(picked.object_keys, keys[[1, 2]])
    assert (picked.layer_group_index, picked.region_indices) == (3, (2, 0))
    assert not picked.slots.flags.writeable
    unnamed = LentLayerGroupRows(3, np.arange(2), slots=np.array([1, 0]), region_indices=(0,))
    none = unnamed.select(np.array([False, False]))
    assert none.object_keys is None and none.slots.tolist() == [] and none.region_indices == (0,)


def on_immutable_bytes(values, dtype=np.int64) -> np.ndarray:
    """``values`` as an array on immutable bytes, as the lender builds its own."""
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@pytest.mark.cpu_only
def test_rows_and_views_built_without_checks_are_what_the_constructors_make():
    """On values that pass the constructors' checks, the unchecked builds are what they make."""
    rng = np.random.default_rng(126)
    seen = set()
    for _ in range(300):
        n = int(rng.integers(0, 6))
        layer_group_index = int(rng.integers(0, 4))
        ordinals = on_immutable_bytes(np.sort(rng.choice(64, n, replace=False)))
        keys = (
            None
            if rng.random() < 0.4
            else on_immutable_bytes(rng.integers(0, 256, (n, 54)), np.uint8)
        )
        placed = bool(rng.random() < 0.6)
        seen.add(placed)
        fields = dict(object_keys=keys)
        if placed:
            regions = tuple(int(r) for r in rng.permutation(5)[: int(rng.integers(1, 4))])
            fields.update(
                slots=on_immutable_bytes(rng.integers(0, 1000, n)), region_indices=regions
            )
        public = LentLayerGroupRows(layer_group_index, ordinals, **fields)
        lent = LentLayerGroupRows._unchecked(layer_group_index, ordinals, **fields)
        assert type(lent) is LentLayerGroupRows
        for field in dataclasses.fields(LentLayerGroupRows):
            mine, theirs = getattr(lent, field.name), getattr(public, field.name)
            if isinstance(theirs, np.ndarray):
                assert mine is theirs, field.name
            else:
                assert (mine, type(mine)) == (theirs, type(theirs)), field.name
        assert len(lent) == len(public) and repr(lent) == repr(public)
        mask = rng.random(n) < 0.5
        assert repr(lent.select(mask)) == repr(public.select(mask))
        with pytest.raises(dataclasses.FrozenInstanceError):
            lent.slots = None
        other = LentLayerGroupRows(layer_group_index + 4, on_immutable_bytes(np.arange(2)))
        groups = (lent, other) if rng.random() < 0.5 else (other, lent)
        view, built = LeaseView(groups, 0, 0), LeaseView._unchecked(groups, 0, 0)
        assert type(built) is LeaseView and built.groups is view.groups
        assert built.num_rows == view.num_rows
        assert [m.tolist() for m in built.row_masks(True)] == [
            m.tolist() for m in view.row_masks(True)
        ]
    assert seen == {True, False}


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", ["lend_read", "lend_write"])
def test_a_lend_checks_memory_and_rows_before_anything_else(name):
    """Wrong lend arguments raise before the range check; right ones reach the shut-down error."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    class Owner:  # anything a weak reference can point to
        pass

    gone = weakref.ref(Owner())
    assert gone() is None
    layout = SimpleNamespace(pool_groups=(), windows=(), named_groups=(), tokens_per_block=4)
    lender = _lender._ManagerLender(gone, layout, None, (), None, gone)
    lend = getattr(lender, name)
    request = SimpleNamespace(py_request_id=1, prompt_len=97)
    host, named = MemoryKind.HOST, RowSelection.NAMED
    for call in (
        lambda: lend(request, 0, 4),
        lambda: lend(request, 0, 4, memory_kind=host),
        lambda: lend(request, 0, 4, rows=named),
        lambda: lend(request, 0, 4, host, named),
    ):
        with pytest.raises(TypeError):
            call()
    for memory, rows, error, why in (
        ("host", named, TypeError, "memory_kind must be a MemoryKind"),
        (None, named, TypeError, "memory_kind must be a MemoryKind"),
        (named, named, TypeError, "memory_kind must be a MemoryKind"),
        (host, "named", TypeError, "rows must be a RowSelection"),
        (host, host, TypeError, "rows must be a RowSelection"),
        (host, RowSelection.ALL, ValueError, "RowSelection.ALL is not lent"),
    ):
        with pytest.raises(error, match=why):
            # A reversed range as well.
            lend(request, 4, 0, memory_kind=memory, rows=rows, ready_event=None)
    with pytest.raises(TypeError, match="ready_event"):
        lend(request, 4, 0, memory_kind=host, rows=named)
    for wrong in (1, "event", object()):
        with pytest.raises(TypeError, match="ready_event must be an event"):
            lend(request, 4, 0, memory_kind=host, rows=named, ready_event=wrong)

    class Asked:  # an event its holder asks, which no stream can wait for
        def query(self):
            return True

    class Waited(Asked):
        def wait(self, stream=None):
            pass

    # A staging lease's copies wait for its event on a stream; an in-place lease only asks it.
    with pytest.raises(TypeError, match=r"an event with query\(\) and wait\(\) or None"):
        lend(request, 4, 0, memory_kind=host, rows=named, ready_event=Asked())
    for kind, event in ((host, Waited()), (MemoryKind.DEVICE, Asked())):
        with pytest.raises(ValueError, match="bad token range"):
            lend(request, 4, 0, memory_kind=kind, rows=named, ready_event=event)
    for boundary, value in (("start", 0.9), ("start", True), ("end", 4.9), ("end", "4")):
        start, end = (value, 4) if boundary == "start" else (0, value)
        with pytest.raises(TypeError, match=f"{boundary} must be an integer"):
            lend(
                request,
                start,
                end,
                memory_kind=MemoryKind.DEVICE,
                rows=RowSelection.ALL,
                ready_event=None,
            )
    assert lender._open_lease_count() == 0, "a refused lend opened a lease"
    with pytest.raises(ValueError, match="bad token range"):
        lend(request, 4, 0, memory_kind=host, rows=named, ready_event=None)
    with pytest.raises(RuntimeError, match=f"lends no more: {_lender._REASON_SHUT_DOWN}"):
        lend(request, 0, 4, memory_kind=host, rows=named, ready_event=None)
    assert lender._open_lease_count() == 0, "a lend that raised opened a lease"


@pytest.mark.cpu_only
def test_a_lease_view_holds_one_row_collection_per_layer_group():
    view = LeaseView([staged_rows(3, 0), staged_rows(2, 1, region=1)], 0, 0)
    assert isinstance(view.groups, tuple) and view.num_rows == 5
    masks = view.row_masks()
    assert [m.shape for m in masks] == [(3,), (2,)]
    assert all(m.dtype == np.bool_ and not m.any() and m.flags.writeable for m in masks)
    assert all(m.all() for m in view.row_masks(True))
    assert LeaseView((), 0, 0).num_rows == 0 and LeaseView((), 0, 0).row_masks() == ()
    with pytest.raises(ValueError):
        LeaseView((staged_rows(3, 0), staged_rows(1, 0)), 0, 0)
    with pytest.raises(TypeError):
        LeaseView((object(),), 0, 0)


@pytest.mark.cpu_only
def test_sparse_rows_keep_group_position_masks_and_placement_alignment():
    keys = np.arange(3 * 54, dtype=np.uint8).reshape(3, 54)
    rows = LentLayerGroupRows(
        layer_group_index=7,
        page_ordinals=np.array([0, 3, 9]),
        object_keys=keys,
        slots=np.array([3, 1, 2]),
        region_indices=(2,),
    )
    empty = LentLayerGroupRows(layer_group_index=2, page_ordinals=np.array([], dtype=np.int64))
    repeated = LentLayerGroupRows(layer_group_index=4, page_ordinals=np.array([3, 9]))
    view = LeaseView(groups=(rows, empty, repeated), start=0, end=0)
    # Five layer-group/page rows, but only three distinct page ordinals.
    assert view.num_rows == 5
    masks = view.row_masks()
    assert [mask.shape for mask in masks] == [(3,), (0,), (2,)]
    masks[0][1] = True
    picked = view.groups[0].select(masks[0])
    assert picked.page_ordinals.tolist() == [3]
    assert picked.slots.tolist() == [1]
    np.testing.assert_array_equal(picked.object_keys, keys[[1]])
    assert (picked.layer_group_index, picked.region_indices) == (7, (2,))
    assert not masks[2].any()


@pytest.mark.cpu_only
def test_resume_bounds_keep_maximum_first_and_region_fields_accept_keywords():
    bounds = ResumeBounds(max_position=96, min_position=32)
    assert tuple(bounds) == (96, 32)
    assert ResumeBounds(0, 1).min_position > ResumeBounds(0, 1).max_position
    region = MemoryRegion(
        layout_label="layout:lg0+1",
        address=4096,
        nbytes=8192,
        slot_bytes=4096,
        slot_count=2,
        layer_group_indices=(7, 2),
        memory_kind=MemoryKind.HOST,
    )
    assert region.layout_label == "layout:lg0+1"
    assert region.layer_group_indices == (7, 2)
    assert region.memory_kind is MemoryKind.HOST


@pytest.mark.cpu_only
def test_staging_options_take_positive_integers():
    options = StagingOptions(256)
    assert dataclasses.astuple(options) == (256, 1, None)
    assert StagingOptions(256, capacity_in_fetches=4, max_bytes=1 << 30).max_bytes == 1 << 30
    for bad in (
        dict(guaranteed_fetch_tokens=0),
        dict(guaranteed_fetch_tokens=8, capacity_in_fetches=0),
    ):
        with pytest.raises(ValueError):
            StagingOptions(**bad)
    with pytest.raises(ValueError):
        StagingOptions(8, max_bytes=0)
    for bad in (
        dict(guaranteed_fetch_tokens=True),
        dict(guaranteed_fetch_tokens=8.0),
        dict(guaranteed_fetch_tokens=8, max_bytes=1.5),
    ):
        with pytest.raises(TypeError):
            StagingOptions(**bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.capacity_in_fetches = 2


@pytest.mark.cpu_only
def test_attach_refuses_a_cross_attention_manager_before_allocating():
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.bindings.internal.batch_manager import CacheType

    manager = object.__new__(KVCacheManagerV2)
    manager.kv_cache_type = CacheType.CROSS
    with pytest.raises(ValueError, match="CROSS"):
        attach_lender(manager, scope=b"cross", staging=StagingOptions(32))
    assert manager._sharing is None


@pytest.mark.cpu_only
def test_a_view_takes_its_lend_s_range_and_refuses_another():
    """A view holds integer bounds ``0 <= start <= end`` and refuses any other."""
    rows = staged_rows(2, 0)
    view = LeaseView((rows,), 32, 64)
    assert (view.start, view.end) == (32, 64)
    assert (LeaseView((rows,), np.int64(0), 0).start, view.num_rows) == (0, 2)
    for start, end in ((1.0, 2), (0, "2"), (True, 2)):
        with pytest.raises(TypeError):
            LeaseView((rows,), start, end)
    for start, end in ((-1, 2), (3, 2)):
        with pytest.raises(ValueError):
            LeaseView((rows,), start, end)
