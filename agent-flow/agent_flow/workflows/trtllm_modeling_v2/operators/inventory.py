# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this build can already supply, measured on the host that runs it.

:func:`run_inventory_probe` runs one read-only process under the run's own
interpreter and reports the catalog entries (certified on this architecture,
still wired, kept by another target) and the registered ``torch.ops.trtllm``
ops no entry wraps. What the checkpoint demands is the surveyor's to read,
from ``config.json`` and the HuggingFace code; joining the two is its
judgement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

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


@dataclass(frozen=True)
class CatalogEntry:
    """One catalog entry as the host actually holds it.

    ``scope`` is empty for the shared catalog and ``<family>/<target>`` for an
    op that target keeps in its own ``ops/``.
    """

    path: str
    impl: str = ""
    receipt: str = RECEIPT_ABSENT
    impl_resolves: bool = True
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
    def promotable(self) -> tuple[CatalogEntry, ...]:
        """Ops another target keeps in its own ``ops/``."""
        return tuple(e for e in self.catalog if e.scope)

    @property
    def rotted(self) -> tuple[CatalogEntry, ...]:
        """Entries whose ``impl`` does not resolve in this build."""
        return tuple(e for e in self.catalog if not e.impl_resolves)

    @property
    def unwrapped_ops(self) -> tuple[str, ...]:
        """Registered ops that no catalog entry wraps."""
        wrapped = {e.impl.rsplit(".", 1)[-1] for e in self.catalog if e.impl}
        return tuple(op for op in self.trtllm_ops if op not in wrapped)


#: Runs on the host, inside the run's own interpreter. Kept as a string
#: rather than a module so it can be shipped through any transport that
#: carries a shell command — ``srun``, a spool file, or a broker call.
PROBE_SOURCE = r"""
import ast, json, os, re, sys
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
    # calls a FlashInfer callable instead of a trtllm op.
    match = re.search(r"torch\.ops\.trtllm\.([A-Za-z0-9_]+)", call)
    if match:
        return sorted(archs), cells, "torch.ops.trtllm." + match.group(1)
    library = re.search(r"\b(flashinfer(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\s*\(", call)
    if library:
        return sorted(archs), cells, library.group(1)
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
# A target's own ops: certified for that target, promotable for another.
models = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
for py in sorted(models.glob("*/*/ops/*/*.py")) if models.is_dir() else ():
    if py.name == "__init__.py":
        continue
    target_dir = py.parents[2]
    out["catalog"].append({
        "path": "%s/%s" % (py.parent.name, py.name),
        "impl": entry_certification(py)[2],
        "scope": "%s/%s" % (target_dir.parent.name, target_dir.name),
    })
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
    entry["impl_resolves"] = True if out["error"] else bool(resolves(entry["impl"], module))

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
            scope=str(e.get("scope", "") or ""),
            cells=int(e.get("cells", 0) or 0),
        )
        for e in data.get("catalog", ())
    )
    return Inventory(
        arch=arch,
        trtllm_ops=tuple(str(o) for o in data.get("trtllm_ops", ())),
        catalog=entries,
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
    landed = bool(inventory.catalog or inventory.trtllm_ops)
    if not landed and not getattr(result, "ok", True):
        return Inventory(
            arch=arch,
            error=f"{inventory.error} (probe exited {getattr(result, 'returncode', '?')})",
        )
    return inventory


# --------------------------------------------------------------------------
# the block the surveyor reads
# --------------------------------------------------------------------------


def format_surface(inventory: Inventory, *, limit: int = 40) -> str:
    """What the build supplies, as the surveyor reads it; ``limit`` caps each list."""
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

    def listed(title: str, items: list[str]) -> None:
        if not items:
            return
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
        "Kept by another target in its own `ops/`: a promotion, not an onboard",
        [f"{e.scope}/ops/{e.category}/{e.name}" for e in inventory.promotable],
    )
    listed(
        "Rotted: the entry's `impl` does not resolve in this build; do not spend these",
        [f"{e.category}/{e.name} -> {e.impl}" for e in inventory.rotted],
    )
    listed(
        "Registered `torch.ops.trtllm` ops no entry wraps: tier 1, a wrapper and its cells",
        list(inventory.unwrapped_ops),
    )
    return "\n".join(lines) + "\n"


def write_surface_file(path: str | Path, inventory: Inventory) -> Path:
    """Write the survey unabridged, with the versions the probe read, for audit and for agents."""
    out = Path(path)
    body = [format_surface(inventory, limit=1_000_000), "## Versions", ""]
    body.extend(f"- {k}: {v}" for k, v in sorted(inventory.versions.items()))
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    return out


__all__ = [
    "CatalogEntry",
    "INVENTORY_TIMEOUT_S",
    "Inventory",
    "PROBE_SOURCE",
    "RECEIPT_ABSENT",
    "RECEIPT_PASSED",
    "format_surface",
    "parse_inventory",
    "probe_command",
    "run_inventory_probe",
    "write_surface_file",
]
