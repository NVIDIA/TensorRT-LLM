# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What changed between two packs (two states of a checkout), optionally for one target.

For a ModelingV2 target moving to a newer checkout, the facts that matter are:
the LLM API fields (those of the engine features the workflow proves first),
its family's routing and targets, the operations it imports, the test lists
and waives that run its family's tests, the reference rows of its models, the
serving configs shipped for its architecture, and the environment variables
the checkout reads. :func:`diff_packs` compares each fact of the two packs;
with ``target`` it keeps what concerns that target.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from .features import FEATURES
from .modeling_v2 import MODELING_V2_REL, MODELING_V2_TESTS_REL
from .query import Wiki

_FIELD_KEYS = ("type", "default", "status", "choices")
_OP_KEYS = ("archs", "has_cells", "calls")
_FEATURE_FIELDS = {path for spec in FEATURES.values() for path in spec.fields}


def _keyed(items: Iterable[Mapping[str, Any]], key: Callable[[Mapping[str, Any]], Any]) -> dict:
    return {key(item): item for item in items}


def _compare(old: dict, new: dict, keys: Iterable[str]) -> dict[str, Any]:
    """``{"added", "removed", "changed"}`` of two keyed fact sets; ``changed`` lists differing keys."""
    changed = []
    for name in sorted(set(old) & set(new), key=str):
        delta = {
            k: [old[name].get(k), new[name].get(k)]
            for k in keys
            if old[name].get(k) != new[name].get(k)
        }
        if delta:
            changed.append({"key": name, **delta})
    return {
        "added": [new[k] for k in sorted(set(new) - set(old), key=str)],
        "removed": [old[k] for k in sorted(set(old) - set(new), key=str)],
        "changed": changed,
    }


def _label(item: Mapping[str, Any]) -> str:
    return str(
        item.get("path") or item.get("test") or item.get("name") or item.get("family") or item
    )


def _resolve_target(wiki: Wiki, target: str) -> tuple[str, str] | None:
    """``(family, target path)`` for ``target``: a target path, its directory name, or ``family/target``."""
    wanted = target.strip("/").removeprefix(f"{MODELING_V2_REL}/").removeprefix("models/")
    for family in wiki.families():
        for entry in family["targets"]:
            if wanted in (entry["name"], f"{family['family']}/{entry['name']}", entry["path"]):
                return family["family"], entry["path"]
    return None


def _scope(old: Wiki, new: Wiki, target: str | None) -> dict[str, Any] | None:
    """What concerns ``target`` in either pack: its family, path, models, architectures, ops, tests."""
    if not target:
        return None
    found = _resolve_target(new, target) or _resolve_target(old, target)
    if found is None:
        raise ValueError(f"no target {target!r} in either pack")
    family, path = found
    families = [w.family(family) or {} for w in (old, new)]
    models = {
        c["model"]
        for f in families
        for c in (f.get("accuracy_tests") or {}).get("classes", [])
        if c.get("model")
    }
    archs = {a for f in families for a in f.get("architectures", [])}
    tests = {f"{MODELING_V2_TESTS_REL}/models/{family}/"}
    tests.update(
        f["accuracy_tests"]["path"] for f in families if f.get("accuracy_tests", {}).get("path")
    )
    ops = set()
    for wiki in (old, new):
        for op in wiki.fact("ops"):
            mine = op["family"] == family and (
                not op["target"] or path.endswith(f"/{op['target']}")
            )
            if mine or path in op.get("used_by", []):
                ops.add((op["module"], op["name"]))
                tests.update(op["tests"])
    return {
        "family": family,
        "target": path,
        "models": models,
        "archs": archs,
        "ops": ops,
        "tests": tests,
    }


def _concerns_test(scope: Mapping[str, Any] | None, path: str) -> bool:
    if scope is None:
        return True
    return any(path == t or (t.endswith("/") and path.startswith(t)) for t in scope["tests"])


def _families(old: Wiki, new: Wiki, scope: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    out = []
    olds, news = (
        _keyed(old.families(), lambda f: f["family"]),
        _keyed(new.families(), lambda f: f["family"]),
    )
    for name in sorted(set(olds) | set(news)):
        if scope is not None and name != scope["family"]:
            continue
        before, after = olds.get(name), news.get(name)
        if before is None or after is None:
            out.append({"family": name, "status": "added" if before is None else "removed"})
            continue
        archs = (set(before["architectures"]), set(after["architectures"]))
        targets = ({t["name"] for t in before["targets"]}, {t["name"] for t in after["targets"]})
        delta = {
            "architectures_added": sorted(archs[1] - archs[0]),
            "architectures_removed": sorted(archs[0] - archs[1]),
            "targets_added": sorted(targets[1] - targets[0]),
            "targets_removed": sorted(targets[0] - targets[1]),
        }
        if any(delta.values()):
            out.append(
                {"family": name, "status": "changed", **{k: v for k, v in delta.items() if v}}
            )
    return out


def _references(old: Wiki, new: Wiki, scope: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    out = []
    before, after = old.fact("accuracy_references"), new.fact("accuracy_references")
    for bench in sorted(set(before) | set(after)):
        for model in sorted(set(before.get(bench, {})) | set(after.get(bench, {}))):
            if scope is not None and model not in scope["models"]:
                continue
            rows = (before.get(bench, {}).get(model, []), after.get(bench, {}).get(model, []))
            if rows[0] != rows[1]:
                out.append({"benchmark": bench, "model": model, "old": rows[0], "new": rows[1]})
    return out


def diff_packs(old: Wiki, new: Wiki, *, target: str | None = None) -> dict[str, Any]:
    """Every fact that differs between ``old`` and ``new``; with ``target``, what concerns it."""
    scope = _scope(old, new, target)
    fields = _compare(
        _keyed(old.fact("config")["fields"], lambda f: f["path"]),
        _keyed(new.fact("config")["fields"], lambda f: f["path"]),
        _FIELD_KEYS,
    )
    touched = {f["path"] for f in fields["added"] + fields["removed"]} | {
        c["key"] for c in fields["changed"]
    }
    op_key = lambda op: (op["module"], op["name"])  # noqa: E731
    ops = _compare(_keyed(old.fact("ops"), op_key), _keyed(new.fact("ops"), op_key), _OP_KEYS)
    entry_key = lambda e: (e["list"], e["test"])  # noqa: E731
    lists = _compare(
        _keyed((e for e in old.fact("test_lists") if _concerns_test(scope, e["path"])), entry_key),
        _keyed((e for e in new.fact("test_lists") if _concerns_test(scope, e["path"])), entry_key),
        ("timeout_min", "stage", "gpu", "gpu_count"),
    )
    waive_key = lambda w: (w["test"], w.get("gpu", ""))  # noqa: E731
    waives = _compare(
        _keyed((w for w in old.fact("waives") if _concerns_test(scope, w["path"])), waive_key),
        _keyed((w for w in new.fact("waives") if _concerns_test(scope, w["path"])), waive_key),
        ("reason",),
    )
    recipe_key = lambda r: (r["path"], r["name"], r["concurrency"])  # noqa: E731
    in_scope = lambda r: scope is None or r["arch"] in scope["archs"]  # noqa: E731
    recipes = _compare(
        _keyed((r for r in old.fact("recipes") if in_scope(r)), recipe_key),
        _keyed((r for r in new.fact("recipes") if in_scope(r)), recipe_key),
        ("tp", "ep", "attention_dp", "speculative", "draft_len", "max_batch_size"),
    )
    if scope is not None:
        for part in ("added", "removed"):
            ops[part] = [op for op in ops[part] if op_key(op) in scope["ops"]]
        ops["changed"] = [c for c in ops["changed"] if c["key"] in scope["ops"]]
    names = ({v["name"] for v in old.fact("env_vars")}, {v["name"] for v in new.fact("env_vars")})
    return {
        "old": {k: old.manifest[k] for k in ("commit", "dirty")},
        "new": {k: new.manifest[k] for k in ("commit", "dirty")},
        "target": scope["target"] if scope else "",
        "feature_fields": sorted(touched & _FEATURE_FIELDS),
        "fields": fields,
        "decoding_types": {
            "added": sorted(set(new.decoding_types()) - set(old.decoding_types())),
            "removed": sorted(set(old.decoding_types()) - set(new.decoding_types())),
        },
        "families": _families(old, new, scope),
        "ops": ops,
        "test_lists": lists,
        "waives": waives,
        "accuracy_tasks": _compare(
            _keyed(old.fact("accuracy_tasks"), lambda t: t["task"]),
            _keyed(new.fact("accuracy_tasks"), lambda t: t["task"]),
            (
                "dataset_dir",
                "alpha",
                "beta",
                "sigma",
                "num_samples",
                "max_input_len",
                "max_output_len",
            ),
        ),
        "references": _references(old, new, scope),
        "recipes": recipes,
        "env_vars": {"added": sorted(names[1] - names[0]), "removed": sorted(names[0] - names[1])},
    }


def _count(section: Any) -> int:
    if isinstance(section, dict) and {"added", "removed"} <= set(section):
        return sum(len(section.get(k, [])) for k in ("added", "removed", "changed"))
    return len(section) if isinstance(section, list) else 0


def render_changes(diff: Mapping[str, Any], *, limit: int = 15) -> str:
    """The diff as text: one block per fact that changed, most specific first."""
    head = f"{diff['old']['commit'][:12]} -> {diff['new']['commit'][:12]}"
    if diff["new"]["dirty"]:
        head += f" (+uncommitted {diff['new']['dirty']})"
    lines = [f"changes {head}" + (f" for {diff['target']}" if diff["target"] else "")]
    if diff["feature_fields"]:
        lines.append(f"feature fields: {', '.join(diff['feature_fields'])}")

    def block(title: str, section: Mapping[str, Any], label: Callable[[Any], str]) -> None:
        if not _count(section):
            return
        lines.append(f"\n{title}:")
        for kind in ("removed", "added", "changed"):
            items = section.get(kind, [])
            for item in items[:limit]:
                sign = {"removed": "-", "added": "+", "changed": "~"}[kind]
                lines.append(f"  {sign} {label(item)}")
            if len(items) > limit:
                lines.append(f"  ... {len(items) - limit} more {kind}")

    def moved(was: Any, now: Any) -> str:
        if isinstance(was, list) and isinstance(now, list):
            plus = [x for x in now if x not in was]
            minus = [x for x in was if x not in now]
            return " ".join([*(f"+{x}" for x in plus), *(f"-{x}" for x in minus)]) or "reordered"
        return f"{was!r} -> {now!r}"

    def changed(item: Mapping[str, Any]) -> str:
        if "key" not in item:
            return _label(item)
        delta = "; ".join(f"{k}: {moved(*v)}" for k, v in item.items() if k != "key")
        key = item["key"] if not isinstance(item["key"], tuple) else " ".join(map(str, item["key"]))
        return f"{key} ({delta})"

    block("LLM API fields", diff["fields"], changed)
    if diff["decoding_types"]["added"] or diff["decoding_types"]["removed"]:
        lines.append(
            f"\nspeculative decoding types: +{diff['decoding_types']['added']} -{diff['decoding_types']['removed']}"
        )
    for family in diff["families"]:
        details = ", ".join(f"{k} {v}" for k, v in family.items() if k not in ("family", "status"))
        lines.append(
            f"\nfamily {family['family']}: {family['status']}"
            + (f" ({details})" if details else "")
        )
    op_label = lambda op: changed(op) if "key" in op else f"{op['name']} {op['path']}"  # noqa: E731
    block("operations", diff["ops"], op_label)
    block(
        "test lists",
        diff["test_lists"],
        lambda e: changed(e) if "key" in e else f"{e['list']}: {e['test']}",
    )
    block(
        "waives",
        diff["waives"],
        lambda w: changed(w) if "key" in w else f"{w['test']} {w.get('reason', '')}",
    )
    block("accuracy tasks", diff["accuracy_tasks"], changed)
    for ref in diff["references"]:
        lines.append(f"\nreference {ref['benchmark']} {ref['model']}: {ref['old']} -> {ref['new']}")
    block(
        "serving configs",
        diff["recipes"],
        lambda r: changed(r) if "key" in r else f"{r['path']} {r['name']}",
    )
    if diff["env_vars"]["added"] or diff["env_vars"]["removed"]:
        lines.append(
            f"\nenvironment variables: +{diff['env_vars']['added']} -{diff['env_vars']['removed']}"
        )
    if len(lines) == 1:
        lines.append("no fact changed")
    return "\n".join(lines) + "\n"
