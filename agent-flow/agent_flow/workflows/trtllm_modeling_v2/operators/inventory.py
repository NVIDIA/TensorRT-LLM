# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What this build can already supply, measured on the host that runs it, for the survey.

One read-only process under the run's own interpreter lists the catalog entries (certified
on this architecture, still wired, kept by another target) and the registered
``torch.ops.trtllm`` ops no entry wraps, and writes them as the surveyor reads them. What
the checkpoint demands is the surveyor's to read; joining the two is its judgement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from ..infra.compute.environment import RunEnvironment

#: The probe imports the engine and runs no kernel: anything slower is not the probe.
INVENTORY_TIMEOUT_S = 600
#: What the probe prints its payload behind, in a log that also carries import chatter.
MARKER = "__SURVEY_JSON__"
#: Names of each list a dispatch carries; ``surface.md`` has them all.
DISPATCH_LIMIT = 40

#: Runs under the run's interpreter: ``<repo> <arch> <limit>``. An entry's claims
#: (``ARCHS``, its ``CELLS``, the op ``__call__`` makes) are read from its source, so an
#: entry whose op does not import here still says what it claims.
PROBE_SOURCE = r"""
import ast, importlib, json, re, sys
from pathlib import Path

repo, arch, limit = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
PKG = "tensorrt_llm._torch._experimental.modeling_v2"
LISTS = (
    ("spendable", "Spendable: shared entries certified on this arch and still wired"),
    ("uncertified", "Present but un-run on this arch: driving their cells here is what is missing"),
    ("kept", "Family and target ops: reuse within the family, or copy into this target's `ops/` "
             "(targets never import sibling targets)"),
    ("rotted", "Rotted: the entry's `impl` does not resolve in this build; do not spend these"),
    ("unwrapped", "Registered `torch.ops.trtllm` ops no entry wraps: tier 1, a wrapper and its cells"),
)
out = {key: [] for key, _ in LISTS}
versions, error = {}, ""

def registered():
    # The schema registry, not dir(): `torch.ops.<ns>` lists only what something resolved.
    import torch
    names = {s.name.split("::", 1)[1] for s in torch._C._jit_get_all_schemas()
             if s.name.startswith("trtllm::")}
    space = getattr(torch.ops, "trtllm", None)
    return names | {n for n in (dir(space) if space is not None else ()) if not n.startswith("_")}

ops = set()
try:
    import torch, tensorrt_llm
    versions = {"torch": torch.__version__,
                "tensorrt_llm": getattr(tensorrt_llm, "__version__", "?")}
    try:  # an op registers when the module that defines it is imported
        importlib.import_module("tensorrt_llm._torch.custom_ops")
    except Exception as exc:
        versions["custom_ops"] = "not imported: %s" % type(exc).__name__
    ops = registered()
except Exception as exc:
    error = "%s: %s" % (type(exc).__name__, exc)

def named(node):
    return getattr(node, "id", None) or getattr(node, "attr", None) or ""

def claims(path):
    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return set(), 0, ""
    archs, cells, call = set(), 0, ""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and any(named(b) == "OpWrapper" for b in node.bases)):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__call__":
                call = ast.get_source_segment(text, stmt) or ""
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
                continue
            names = [named(t) for t in getattr(stmt, "targets", None) or [stmt.target]]
            for sub in ast.walk(stmt.value):
                word = str(sub.value if isinstance(sub, ast.Constant) else named(sub)).lower()
                if "ARCHS" in names and word.startswith("sm_"):
                    archs.add(word)
                if "CELLS" in names and isinstance(sub, ast.Call) and named(sub.func) == "Cell":
                    cells += 1
    # The op `__call__` makes, not the first one the file mentions; a tier-3 entry calls
    # a FlashInfer callable instead.
    made = (re.search(r"torch\.ops\.trtllm\.\w+", call)
            or re.search(r"\bflashinfer(?:\.[A-Za-z_]\w*)+(?=\s*\()", call)
            or re.search(r"torch\.ops\.trtllm\.\w+", text))
    return archs, cells, made.group(0) if made else ""

def resolves(impl, module):
    if not impl or error:
        return True  # an inventory that was not taken calls nothing unwired
    parts = impl.split(".")
    if impl.startswith("torch.ops.trtllm."):
        if parts[-1] not in ops:
            try:  # the wrapper's own import performs whatever import registers its op
                importlib.import_module(module)
                ops.update(registered())
            except Exception:
                return False
        return parts[-1] in ops
    for cut in range(len(parts) - 1, 0, -1):
        try:
            found = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        for attr in parts[cut:]:
            found = getattr(found, attr, None)
        return found is not None
    return False

base = repo / PKG.replace(".", "/")
models = base / "models"
# The shared vocabulary is the entry files: one OpWrapper per file under
# catalog/<category>/. A family's or a target's own ops are prior art for another.
files = [(p, "") for p in sorted((base / "catalog").glob("*/*.py")) if not p.name.startswith("_")]
files += [(p, p.parents[2].relative_to(models).as_posix())
          for p in sorted({*models.glob("*/*/ops/*/*.py"), *models.glob("*/ops/*/*.py")})
          if p.name != "__init__.py"]
wrapped = set()
for path, scope in files:
    archs, cells, impl = claims(path)
    name = "%s/%s" % (path.parent.name, path.stem)
    home = ["models", *scope.split("/"), "ops"] if scope else ["catalog"]
    wrapped.add(impl.rsplit(".", 1)[-1])
    if not resolves(impl, ".".join([PKG, *home, path.parent.name, path.stem])):
        out["rotted"].append("%s%s -> %s" % (scope and scope + "/ops/", name, impl))
    elif scope:
        out["kept"].append("%s/ops/%s" % (scope, name))
    else:
        out["spendable" if arch in archs and cells else "uncertified"].append(name)
out["unwrapped"] = sorted(ops - wrapped)

def surface(limit):
    lines = ["## Surface survey", "", "Measured on `%s` before this run's first turn. What the "
             "checkpoint demands is yours to read (`config.json`, the HuggingFace code); joining "
             "the two is your job." % arch]
    if error:
        lines += ["", "**The inventory probe reported an error** (`%s`): what is below was read "
                  "off disk, and what is absent was not measured. Nothing here is evidence of "
                  "absence; name the missing measurement in the plan's open items." % error]
    for key, title in LISTS:
        if out[key]:
            lines += ["", "**%s** (%d):" % (title, len(out[key])), ""]
            lines += ["- `%s`" % name for name in out[key][:limit]]
            if limit and len(out[key]) > limit:
                lines.append("- ... %d more, see `surface.md`" % (len(out[key]) - limit))
    return "\n".join(lines) + "\n"

said = "".join("- %s: %s\n" % pair for pair in sorted(versions.items()))
headline = "%d/%d entries certified on %s, %d unwrapped op(s)" % (
    len(out["spendable"]), len(files), arch, len(out["unwrapped"]))
print("__SURVEY_JSON__" + json.dumps({
    "block": surface(limit), "surface": surface(None) + "\n## Versions\n\n" + said,
    "headline": "inventory probe failed: " + error.splitlines()[0] if error else headline}))
"""


@dataclass(frozen=True)
class Inventory:
    """What the host supplies, as the surveyor reads it."""

    #: The survey with each list cut at :data:`DISPATCH_LIMIT`, for the survey's dispatch.
    block: str
    #: The survey unabridged, with the versions the probe read: ``surface.md``.
    surface: str
    headline: str


def parse_inventory(stdout: str) -> Inventory:
    """The probe's payload, out of a log that also carries the engine's chatter.

    A probe that printed none is said in the block the surveyor reads: an inventory that
    was not taken must not read as an empty catalog.
    """
    match = re.search(rf"^{re.escape(MARKER)}(.*)$", stdout, re.MULTILINE)
    try:
        return Inventory(**json.loads(match.group(1) if match else ""))
    except (ValueError, TypeError):
        tail = "\n".join(stdout.splitlines()[-15:])
        block = (
            "## Surface survey\n\n**The inventory probe printed no payload**, so nothing was "
            "measured: plan on the assumption that you do not know what exists, and say so in "
            f"the plan's open items. Its last lines:\n\n```\n{tail}\n```\n"
        )
        return Inventory(block, block, "inventory probe failed: it printed no payload")


def run_inventory_probe(
    task_data: Mapping[str, Any], environment: RunEnvironment, *, shell_runner: Any
) -> Inventory:
    """Take the inventory with one process on the host that runs the build; a failure is reported."""
    argv = [str(task_data["trtllm_repo_path"]), str(task_data.get("arch", "")), str(DISPATCH_LIMIT)]
    command = environment.python_command(["-c", PROBE_SOURCE, *argv], single_process=True)
    result = shell_runner(command, None, INVENTORY_TIMEOUT_S)
    if getattr(result, "timed_out", False):
        return parse_inventory(f"the probe timed out after {INVENTORY_TIMEOUT_S}s")
    return parse_inventory(getattr(result, "stdout", "") + getattr(result, "stderr", ""))
