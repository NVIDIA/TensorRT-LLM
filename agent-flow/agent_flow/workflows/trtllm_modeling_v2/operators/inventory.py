# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this build can already supply, measured on the host that runs it.

:func:`run_inventory_probe` runs one read-only process under the run's own
interpreter and reports the catalog entries (certified on this architecture,
still wired, kept by another target), the registered ``torch.ops.trtllm``
ops no entry wraps, and the ModelingV2 native ops: what each family's and
target's ``csrc/`` declares, whether this installation built and registered
it, and which entries wrap it. What the checkpoint demands is the surveyor's to read,
from ``config.json`` and the HuggingFace code; joining the two is its
judgement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..infra.compute.environment import RunEnvironment

#: The inventory probe imports the engine, so it pays a real import cost, but
#: it runs no kernel and allocates nothing. Anything slower than this is not
#: the probe.
INVENTORY_TIMEOUT_S = 600

#: What the probe prints its JSON payload behind, so the parser can find it
#: in a log that also carries the engine's import-time chatter.
MARKER = "__SURVEY_JSON__"


#: An entry's ``ARCHS`` names this architecture: its cells were driven here.
RECEIPT_PASSED = "passed"
#: This architecture is not in ``ARCHS`` (or the entry declares none): unknown.
RECEIPT_ABSENT = "absent"

#: Why an entry's ``impl`` does not resolve here. ``missing``: the build has no
#: such op (renamed, removed, or a native library built from older source);
#: ``broken``: its native library is installed but does not load.
IMPL_MISSING = "missing"
IMPL_BROKEN = "broken"
#: A native op whose family library or target this installation did not build
#: (its ``ARCHS`` miss the build's): absent here, not rotted.
IMPL_UNBUILT = "unbuilt"

#: A native op a wrapper or a model queries before its real op (capability,
#: workspace sizing): never a candidate to wrap.
NATIVE_HELPER = re.compile(r"(_supported|_workspace_info|built_targets)$")

#: A native op the family library registered in this installation.
NATIVE_REGISTERED = "registered"
#: A native op ``csrc/`` declares, on a host where the probe could not load torch.
NATIVE_DECLARED = "declared"


@dataclass(frozen=True)
class NativeOp:
    """A ModelingV2 native op: ``torch.ops.<namespace>.<name>`` of a family or target library."""

    namespace: str
    name: str
    #: ``registered``, ``unbuilt``, ``missing``, ``broken``, or ``declared`` when unmeasured.
    state: str
    #: Where ``csrc/`` declares it (``path:line``); empty for an op only the library registers.
    source: str = ""

    @property
    def impl(self) -> str:
        return f"torch.ops.{self.namespace}.{self.name}"


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog entry as the host actually holds it.

    ``scope`` is empty for the catalog, ``<family>`` for family-shared ops and
    ``<family>/<target>`` for ops kept by one target.
    """

    path: str
    impl: str = ""
    receipt: str = RECEIPT_ABSENT
    impl_resolves: bool = True
    #: Why ``impl`` does not resolve (``IMPL_*``); empty when it does.
    impl_state: str = ""
    scope: str = ""
    #: How many ``Cell`` s the entry certifies; 0 when it states none.
    cells: int = 0

    @property
    def name(self) -> str:
        return Path(self.path).stem

    @property
    def category(self) -> str:
        parts = Path(self.path).parts
        return parts[0] if len(parts) > 1 else ""

    @property
    def certifiable(self) -> bool:
        """Every shared entry is an ``OpWrapper`` with cells to certify."""
        return not self.scope

    @property
    def spendable(self) -> bool:
        """Usable without a kernel writer dispatch: shared, certified here, and still wired."""
        if not self.impl_resolves or self.scope:
            return False
        return self.receipt == RECEIPT_PASSED


@dataclass(frozen=True)
class Inventory:
    """What the host supplies, measured rather than assumed."""

    arch: str = ""
    trtllm_ops: tuple[str, ...] = ()
    catalog: tuple[CatalogEntry, ...] = ()
    native_ops: tuple[NativeOp, ...] = ()
    #: Family -> ``loaded`` (with the targets it carries), ``unbuilt`` or ``broken: <why>``.
    native_libraries: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def spendable(self) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.catalog if e.spendable)

    @property
    def uncertified(self) -> tuple[CatalogEntry, ...]:
        """Present, wired, certifiable, but carrying no pass for this arch."""
        return tuple(
            e for e in self.catalog if e.certifiable and e.impl_resolves and not e.spendable
        )

    @property
    def other_targets(self) -> tuple[CatalogEntry, ...]:
        """Ops another target keeps in its own ``ops/``."""
        return tuple(e for e in self.catalog if e.scope)

    @property
    def rotted(self) -> tuple[CatalogEntry, ...]:
        """Entries whose ``impl`` does not resolve in this build (an unbuilt native op is not)."""
        return tuple(
            e for e in self.catalog if not e.impl_resolves and e.impl_state != IMPL_UNBUILT
        )

    @property
    def unbuilt(self) -> tuple[CatalogEntry, ...]:
        """Entries wrapping a native op this installation did not build."""
        return tuple(e for e in self.catalog if e.impl_state == IMPL_UNBUILT)

    @property
    def unwrapped_ops(self) -> tuple[str, ...]:
        """Registered ``torch.ops.trtllm`` ops that no catalog entry wraps."""
        wrapped = {
            e.impl.rsplit(".", 1)[-1]
            for e in self.catalog
            if e.impl.startswith("torch.ops.trtllm.")
        }
        return tuple(op for op in self.trtllm_ops if op not in wrapped)

    @property
    def unwrapped_native_ops(self) -> tuple[NativeOp, ...]:
        """Native family and target ops no entry wraps (capability and workspace queries aside)."""
        wrapped = {e.impl for e in self.catalog if e.impl}
        return tuple(
            op
            for op in self.native_ops
            if op.impl not in wrapped and not NATIVE_HELPER.search(op.name)
        )


#: Runs on the host, inside the run's own interpreter. Kept as a string
#: rather than a module so it can be shipped through any transport that
#: carries a shell command — a spool file, or a direct invocation.
PROBE_SOURCE = r"""
import ast, importlib, json, os, re, sys
from pathlib import Path

out = {"trtllm_ops": [], "catalog": [], "versions": {}, "error": ""}
repo = Path(sys.argv[1])
arch = sys.argv[2]

try:
    import torch
    out["versions"]["torch"] = torch.__version__
    import tensorrt_llm
    out["versions"]["tensorrt_llm"] = getattr(tensorrt_llm, "__version__", "?")
    # `torch.ops.<ns>` is a LAZY proxy: `dir()` lists only the names something
    # has already resolved, not every registered name. The registry is the
    # truth; `dir()` stays as a fallback for a torch without the private accessor.
    # Import the Python registration plane before enumerating: an op whose
    # `custom_ops/*.py` module has not been imported has not registered its
    # schema either.
    try:
        import importlib

        importlib.import_module("tensorrt_llm._torch.custom_ops")
        out["versions"]["custom_ops_imported"] = "yes"
    except Exception as exc:
        out["versions"]["custom_ops_imported"] = "no: %s" % type(exc).__name__
    ops = getattr(torch.ops, "trtllm", None)
    names = set()
    try:
        for schema in torch._C._jit_get_all_schemas():
            if schema.name.startswith("trtllm::"):
                names.add(schema.name.split("::", 1)[1])
    except Exception:
        pass
    if ops is not None:
        names.update(n for n in dir(ops) if not n.startswith("_"))
    out["trtllm_ops"] = sorted(names)
except Exception as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, exc)

CATALOG_PKG = "tensorrt_llm._torch._experimental.modeling_v2.catalog"

def resolves(dotted, entry_path=None):
    if not dotted:
        return True
    if dotted.startswith("torch.ops.trtllm."):
        name = dotted.rsplit(".", 1)[-1]
        if name in out["trtllm_ops"]:
            return True
        # Not in the registry YET. An op registers when the module holding
        # its `@torch.library.custom_op` is imported, and several live
        # outside `custom_ops/`. So ask the wrapper: importing it performs
        # whatever import the op needs.
        if entry_path:
            import importlib
            module = entry_path
            if "/" in module:
                module = CATALOG_PKG + "." + module[:-3].replace("/", ".")
            try:
                importlib.import_module(module)
            except Exception:
                return False
            try:
                for schema in torch._C._jit_get_all_schemas():
                    if schema.name.startswith("trtllm::"):
                        out["trtllm_ops"].append(schema.name.split("::", 1)[1])
                out["trtllm_ops"] = sorted(set(out["trtllm_ops"]))
            except Exception:
                pass
            return name in out["trtllm_ops"]
        return False
    import importlib
    parts = dotted.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return False
        return True
    return False

MODELS_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models"
models = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/models"

def native_namespace(family, target=None):
    # Mirrors `_native._namespace`: `mv2_<family>`, or `mv2_<family>_<target>` with `__` collapsed.
    if target is None:
        return "mv2_" + family
    return "mv2_%s_%s" % (family, re.sub("_{2,}", "_", target))

def owner(py):
    # (family, target) of a file under models/; target is None for the family's own code.
    parts = py.relative_to(models).parts
    target = parts[1] if len(parts) > 2 and "__" in parts[1] else None
    return parts[0], target

def native_bindings(py, tree, depth=0):
    # Module-level names bound to a native library's ops, and their namespace:
    # `x = load_family_ops(__name__)`, `x = load_target_ops(__name__)`, or a name
    # imported from a package whose `__init__.py` binds one (`from ... import native_ops`).
    try:
        family, target = owner(py)
    except ValueError:
        return {}
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            called = getattr(func, "id", None) or getattr(func, "attr", None)
            for t in node.targets:
                if not isinstance(t, ast.Name):
                    continue
                if called == "load_family_ops":
                    names[t.id] = native_namespace(family)
                elif called == "load_target_ops" and target:
                    names[t.id] = native_namespace(family, target)
        elif isinstance(node, ast.ImportFrom) and depth < 3:
            if node.level:
                pkg = py.parent
                for _ in range(node.level - 1):
                    pkg = pkg.parent
                if node.module:
                    pkg = pkg.joinpath(*node.module.split("."))
            elif (node.module or "").startswith(MODELS_PKG + "."):
                pkg = models.joinpath(*node.module[len(MODELS_PKG) + 1:].split("."))
            else:
                continue
            init = pkg / "__init__.py"
            try:
                bound = native_bindings(init, ast.parse(init.read_text(encoding="utf-8")), depth + 1)
            except (OSError, SyntaxError, ValueError):
                continue
            for alias in node.names:
                if alias.name in bound:
                    names[alias.asname or alias.name] = bound[alias.name]
    return names

# Queries a wrapper makes before its op (capability and workspace sizing), not its op.
HELPER = re.compile(r"(_supported|_workspace_info|built_targets)$")

def wrapped_call(call, bindings):
    # The op `__call__` makes: the first call into torch.ops (trtllm or a native
    # namespace), a name bound to native ops, or FlashInfer.
    found = []
    for match in re.finditer(r"torch\.ops\.(trtllm|mv2_[A-Za-z0-9_]+)\.([A-Za-z0-9_]+)", call):
        found.append((match.start(), "torch.ops.%s.%s" % match.groups()))
    for name, namespace in bindings.items():
        for match in re.finditer(r"\b%s\.([A-Za-z_][A-Za-z0-9_]*)\s*\(" % re.escape(name), call):
            found.append((match.start(), "torch.ops.%s.%s" % (namespace, match.group(1))))
    library = re.search(r"\b(flashinfer(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\(", call)
    if library:
        found.append((library.start(), library.group(1)))
    found.sort()
    main = [impl for _, impl in found if not HELPER.search(impl)]
    return (main or [impl for _, impl in found] or [""])[0]

def entry_certification(py_path):
    # An entry is an OpWrapper subclass whose class attributes are its
    # contract: `ARCHS = frozenset({Arch.SM_103})` names the architectures
    # its `CELLS` were driven on. Read from the AST rather than by importing,
    # so an entry whose op does not import here still reports what it claims.
    # Returns (archs, cells, impl).
    try:
        text = py_path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return [], 0, ""
    bindings = native_bindings(py_path, tree)
    archs, cells, call = set(), 0, ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            (isinstance(b, ast.Name) and b.id == "OpWrapper")
            or (isinstance(b, ast.Attribute) and b.attr == "OpWrapper")
            for b in node.bases
        ):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__call__":
                call = ast.get_source_segment(text, stmt) or ""
                continue
            if isinstance(stmt, ast.Assign):
                names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                names = [stmt.target.id]
            else:
                continue
            if stmt.value is None:
                continue
            for sub in ast.walk(stmt.value):
                if "ARCHS" in names:
                    if isinstance(sub, ast.Attribute) and sub.attr.startswith("SM_"):
                        archs.add(sub.attr.lower())
                    elif isinstance(sub, ast.Constant) and str(sub.value).startswith("sm_"):
                        archs.add(str(sub.value))
                if "CELLS" in names and isinstance(sub, ast.Call):
                    func = sub.func
                    if (isinstance(func, ast.Name) and func.id == "Cell") or (
                        isinstance(func, ast.Attribute) and func.attr == "Cell"
                    ):
                        cells += 1
    # The op `__call__` makes, not the first op the file mentions: an entry's
    # docstring and reference routinely name its neighbours. A tier-3 entry
    # calls a FlashInfer callable instead of a trtllm op; a family's or a
    # target's entry may call its native library.
    impl = wrapped_call(call, bindings)
    if impl:
        return sorted(archs), cells, impl
    match = re.search(r"torch\.ops\.trtllm\.([A-Za-z0-9_]+)", text)
    return sorted(archs), cells, ("torch.ops.trtllm." + match.group(1)) if match else ""

catalog = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/catalog"
# The shared vocabulary is the entry files themselves -- there is no index: one
# OpWrapper per file under catalog/<category>/; underscore modules (_op.py) are
# the catalog's own infrastructure.
for py in sorted(catalog.glob("*/*.py")) if catalog.is_dir() else ():
    if py.name.startswith("_"):
        continue
    out["catalog"].append({
        "path": "%s/%s" % (py.parent.name, py.name),
        "impl": entry_certification(py)[2],
    })
# A target's own ops: certified for that target, prior art for another.
scoped_ops = sorted({*models.glob("*/*/ops/*/*.py"), *models.glob("*/ops/*/*.py")})
for py in scoped_ops if models.is_dir() else ():
    if py.name == "__init__.py":
        continue
    target_dir = py.parents[2]
    out["catalog"].append({
        "path": "%s/%s" % (py.parent.name, py.name),
        "impl": entry_certification(py)[2],
        "scope": target_dir.relative_to(models).as_posix(),
    })
# ModelingV2 native ops: what each family's and target's csrc/ declares
# (`TRTLLM_MV2_LIBRARY_FRAGMENT(m) { m.def("op(...) -> ...") }`), then what this
# installation's family libraries register. Loading a library is the public
# `load_family_ops`, which routing calls too; it runs no kernel.
DEF = re.compile(r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)\s*[(.]')
declared, targets_of = {}, {}
for csrc in sorted([*models.glob("*/csrc"), *models.glob("*/*/csrc")]) if models.is_dir() else ():
    parts = csrc.relative_to(models).parts
    family, target = parts[0], (parts[1] if len(parts) == 3 else None)
    namespace = native_namespace(family, target)
    targets_of.setdefault(family, {})
    if target:
        targets_of[family][namespace] = target
    for src in sorted(csrc.rglob("*")):
        if src.suffix not in (".cpp", ".cc", ".cu", ".h", ".cuh"):
            continue
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in DEF.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            declared.setdefault((namespace, match.group(1)),
                                "%s:%d" % (src.relative_to(repo).as_posix(), line))
# A family or a target can also register an op of its namespace in Python:
# `@torch.library.custom_op(f"{_NS}::name")`, `_NS` being `namespace_of(__name__)`.
CUSTOM_OP = re.compile(r'''custom_op\(\s*f?["'](?:\{[^}]*\}|(mv2_[A-Za-z0-9_]+))::([A-Za-z0-9_]+)''')
for py in sorted(models.rglob("*.py")) if models.is_dir() else ():
    try:
        text = py.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    if "custom_op(" not in text:
        continue
    family, target = owner(py)
    for match in CUSTOM_OP.finditer(text):
        namespace = match.group(1) or native_namespace(family, target)
        line = text.count("\n", 0, match.start()) + 1
        targets_of.setdefault(family, {})
        if target:
            targets_of[family][native_namespace(family, target)] = target
        declared.setdefault((namespace, match.group(2)), "%s:%d" % (py.relative_to(repo).as_posix(), line))
out["native_libraries"] = {}
registered = set()
if not out["error"]:
    try:
        import importlib
        native = importlib.import_module("tensorrt_llm._torch._experimental.modeling_v2._native")
    except Exception as exc:
        native = None
        out["versions"]["modeling_v2_native"] = "no: %s" % type(exc).__name__
    for family in sorted(targets_of):
        if native is None:
            break
        try:
            native.load_family_ops("%s.%s" % (MODELS_PKG, family))
            built = getattr(torch.ops, native_namespace(family)).built_targets()
            out["native_libraries"][family] = "loaded: %s" % (built or "no targets")
        except Exception as exc:
            kind = type(exc).__name__
            if kind == "NoTarget":
                out["native_libraries"][family] = "unbuilt: %s" % exc
            else:
                out["native_libraries"][family] = "broken: %s: %s" % (kind, exc)

def refresh_registered():
    if out["error"]:
        return
    try:
        for schema in torch._C._jit_get_all_schemas():
            namespace, _, name = schema.name.partition("::")
            if namespace.startswith("mv2_") and name != "built_targets":
                registered.add((namespace, name.split(".", 1)[0]))
    except Exception:
        pass

refresh_registered()

def native_state(namespace, name):
    # registered, unbuilt (the library or the target is not in this build),
    # missing (built, but no such op), broken, or declared (not measured).
    if out["error"]:
        return "declared"
    if (namespace, name) in registered:
        return "registered"
    family = next((f for f in targets_of if namespace == native_namespace(f)
                   or namespace in targets_of[f]), None)
    library = out["native_libraries"].get(family, "")
    if library.startswith("broken"):
        return "broken"
    if not library.startswith("loaded"):
        return "unbuilt"
    target = targets_of[family].get(namespace)
    built = [t for t in library.split(": ", 1)[1].split(",") if t]
    if target and target not in built:
        return "unbuilt"
    return "missing"

for entry in out["catalog"]:
    if entry.get("scope"):
        py = models / entry["scope"] / "ops" / entry["path"]
        module = "tensorrt_llm._torch._experimental.modeling_v2.models.%s.ops.%s" % (
            entry["scope"].replace("/", "."), entry["path"][:-3].replace("/", "."))
    else:
        py = catalog / entry["path"]
        module = entry["path"]
    archs, cells, _ = entry_certification(py)
    entry["cells"] = cells
    entry["receipt"] = "passed" if arch in archs and cells else "absent"
    # Only claim an entry is unwired when the op inventory was actually
    # taken.
    if out["error"]:
        entry["impl_resolves"] = True
    elif entry["impl"].startswith("torch.ops.mv2_"):
        # An op the module registers in Python exists once the module is imported;
        # importing a target's module also loads its library (`load_target_ops`).
        try:
            importlib.import_module(module)
        except Exception:
            pass
        refresh_registered()
        state = native_state(*entry["impl"].split(".")[2:4])
        entry["impl_resolves"] = state == "registered"
        entry["impl_state"] = "" if state == "registered" else state
    else:
        entry["impl_resolves"] = bool(resolves(entry["impl"], module))
        entry["impl_state"] = "" if entry["impl_resolves"] else "missing"

refresh_registered()
out["native_ops"] = [
    {"namespace": ns, "name": name, "state": native_state(ns, name), "source": declared.get((ns, name), "")}
    for ns, name in sorted(set(declared) | registered)
]

print("__SURVEY_JSON__" + json.dumps(out))
"""


def probe_command(environment: RunEnvironment, *, repo: str | Path, arch: str) -> str:
    """The literal command that takes the inventory, fully prefixed.

    Launched as **one** process whatever the run's world size: this is
    read-only introspection, so nothing in it wants a rank per device.
    """
    return environment.python_command(
        ["-c", PROBE_SOURCE, str(repo), str(arch)], single_process=True
    )


def parse_inventory(stdout: str, *, arch: str = "") -> Inventory:
    """Pull the probe's payload out of a log that also carries engine chatter."""
    match = re.search(rf"^{re.escape(MARKER)}(.*)$", stdout, re.MULTILINE)
    if not match:
        tail = "\n".join(stdout.splitlines()[-15:])
        return Inventory(arch=arch, error=f"probe printed no payload. last lines:\n{tail}")
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        return Inventory(arch=arch, error=f"probe payload is not JSON: {exc}")
    entries = tuple(
        CatalogEntry(
            path=str(e.get("path", "")),
            impl=str(e.get("impl", "")),
            receipt=str(e.get("receipt", RECEIPT_ABSENT)),
            impl_resolves=bool(e.get("impl_resolves", True)),
            impl_state=str(e.get("impl_state", "") or ""),
            scope=str(e.get("scope", "") or ""),
            cells=int(e.get("cells", 0) or 0),
        )
        for e in data.get("catalog", ())
    )
    native = tuple(
        NativeOp(
            namespace=str(o.get("namespace", "")),
            name=str(o.get("name", "")),
            state=str(o.get("state", NATIVE_DECLARED)),
            source=str(o.get("source", "") or ""),
        )
        for o in data.get("native_ops", ())
    )
    return Inventory(
        arch=arch,
        trtllm_ops=tuple(str(o) for o in data.get("trtllm_ops", ())),
        catalog=entries,
        native_ops=native,
        native_libraries={str(k): str(v) for k, v in (data.get("native_libraries") or {}).items()},
        versions={str(k): str(v) for k, v in (data.get("versions") or {}).items()},
        error=str(data.get("error", "")),
    )


def run_inventory_probe(
    task_data: Mapping[str, Any], environment: RunEnvironment, *, shell_runner: Any
) -> Inventory:
    """Take the inventory on the host that will run the build.

    A failure here is reported, not raised.
    """
    arch = str(task_data.get("arch", ""))
    command = probe_command(environment, repo=task_data["trtllm_repo_path"], arch=arch)
    result = shell_runner(command, None, INVENTORY_TIMEOUT_S)
    if getattr(result, "timed_out", False):
        return Inventory(arch=arch, error=f"inventory probe timed out after {INVENTORY_TIMEOUT_S}s")
    stdout = getattr(result, "stdout", "") + getattr(result, "stderr", "")
    inventory = parse_inventory(stdout, arch=arch)
    # A payload that landed is kept whatever the exit code says. The probe
    # catches its own import failures and still prints what it read off
    # disk.
    landed = bool(inventory.catalog or inventory.trtllm_ops or inventory.native_ops)
    if not landed and not getattr(result, "ok", True):
        return Inventory(
            arch=arch,
            error=f"{inventory.error} (probe exited {getattr(result, 'returncode', '?')})",
        )
    return inventory


# --------------------------------------------------------------------------
# the block the surveyor reads
# --------------------------------------------------------------------------


def format_surface(
    inventory: Inventory, *, limit: int = 40, rank: Callable[[str], float] | None = None
) -> str:
    """What the build supplies, as the surveyor reads it; ``limit`` caps each list.

    With ``rank`` (``prior_art.ranker``) each list is ordered by how much each name shares
    the checkpoint's vocabulary before it is cut, so what this checkpoint is likely to
    reuse is read first, however its name sorts.
    """
    lines = ["## Surface survey", ""]
    if not inventory.ok:
        lines += [
            f"**The inventory probe reported an error** (`{inventory.error}`): what is "
            "below was still read off disk, and what is absent was not measured. Nothing "
            "here is evidence of absence; name the missing measurement in the plan's open "
            "items.",
            "",
        ]
    lines.append(
        f"Measured on `{inventory.arch or 'this arch'}` before this phase started. What the "
        "checkpoint demands is yours to read (`config.json`, the HuggingFace code); joining "
        "the two is your job."
    )
    if rank is not None:
        lines.append(
            "Each list is ordered by how much a name shares this checkpoint's `config.json` "
            "vocabulary, most first; a cut list continues, in the same order, in `surface.md`."
        )

    def listed(title: str, items: list[str]) -> None:
        if not items:
            return
        if rank is not None:
            items = sorted(items, key=lambda item: (-rank(item), item))
        lines.extend(["", f"**{title}** ({len(items)}):", ""])
        lines.extend(f"- `{item}`" for item in items[:limit])
        if len(items) > limit:
            lines.append(f"- ... {len(items) - limit} more, see `surface.md`")

    listed(
        "Spendable: shared entries certified on this arch and still wired",
        [f"{e.category}/{e.name}" for e in inventory.spendable],
    )
    listed(
        "Present but un-run on this arch: driving their cells here is what is missing",
        [f"{e.category}/{e.name}" for e in inventory.uncertified],
    )
    listed(
        "Family and target ops: reuse within the same family or copy into this target's `ops/` "
        "(targets never import sibling targets)",
        [f"{e.scope}/ops/{e.category}/{e.name}" for e in inventory.other_targets],
    )
    listed(
        "Rotted: the entry's `impl` does not resolve in this build; do not spend these",
        [
            f"{_entry_name(e)} -> {e.impl}" + (f" ({e.impl_state})" if e.impl_state else "")
            for e in inventory.rotted
        ],
    )
    listed(
        "Native ops this installation did not build (the family library or the target is not "
        "in it): absent on this host, not rotted",
        [f"{_entry_name(e)} -> {e.impl}" for e in inventory.unbuilt],
    )
    listed(
        "Registered `torch.ops.trtllm` ops no entry wraps: tier 1, a wrapper and its cells",
        list(inventory.unwrapped_ops),
    )
    listed(
        "ModelingV2 native ops no entry wraps (a family's or a target's own library; within "
        "the family, wrap one before reusing it)",
        [_native_name(op) for op in inventory.unwrapped_native_ops],
    )
    if inventory.native_libraries:
        lines.extend(["", "**ModelingV2 native libraries on this host:**", ""])
        lines.extend(
            f"- `{family}`: {state}" for family, state in sorted(inventory.native_libraries.items())
        )
    return "\n".join(lines) + "\n"


def _entry_name(entry: CatalogEntry) -> str:
    return (
        f"{entry.scope}/ops/{entry.category}/{entry.name}"
        if entry.scope
        else f"{entry.category}/{entry.name}"
    )


def _native_name(op: NativeOp) -> str:
    where = f" ({op.source})" if op.source else ""
    return f"{op.namespace}::{op.name} [{op.state}]{where}"


def surface_names(inventory: Inventory) -> list[str]:
    """Every name the surface lists, as it lists them: what a ranking is weighed over."""
    return [
        *(f"{e.category}/{e.name}" for e in inventory.catalog if not e.scope),
        *(f"{e.scope}/ops/{e.category}/{e.name}" for e in inventory.other_targets),
        *inventory.unwrapped_ops,
        *(_native_name(op) for op in inventory.unwrapped_native_ops),
    ]


def write_surface_file(
    path: str | Path, inventory: Inventory, *, rank: Callable[[str], float] | None = None
) -> Path:
    """Write the survey unabridged, with the versions the probe read, for audit and for agents."""
    out = Path(path)
    body = [format_surface(inventory, limit=1_000_000, rank=rank), "## Versions", ""]
    body.extend(f"- {k}: {v}" for k, v in sorted(inventory.versions.items()))
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    return out


__all__ = [
    "CatalogEntry",
    "IMPL_BROKEN",
    "IMPL_MISSING",
    "IMPL_UNBUILT",
    "INVENTORY_TIMEOUT_S",
    "Inventory",
    "NATIVE_DECLARED",
    "NATIVE_REGISTERED",
    "NativeOp",
    "PROBE_SOURCE",
    "RECEIPT_ABSENT",
    "RECEIPT_PASSED",
    "format_surface",
    "parse_inventory",
    "probe_command",
    "run_inventory_probe",
    "surface_names",
    "write_surface_file",
]
