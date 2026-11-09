# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answers from a wiki pack: fields, features, families, operations, CI lists, accuracy, recipes."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import config as config_mod
from . import features as features_mod
from . import recipes as recipes_mod
from . import tests as tests_mod
from .modeling_v2 import MODELING_V2_REL

_TEST_ROOTS = ("tests/unittest", "tests/integration/defs")


class Wiki:
    """Read-only view of one pack directory; facts load on first use."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.manifest: dict[str, Any] = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        self._facts: dict[str, Any] = {}

    def fact(self, name: str) -> Any:
        if name not in self._facts:
            target = self.path / "facts" / f"{name}.json"
            self._facts[name] = json.loads(target.read_text(encoding="utf-8"))
        return self._facts[name]

    @property
    def repo(self) -> Path:
        return Path(self.manifest["repo"])

    # -- configuration -------------------------------------------------------------

    @cached_property
    def _fields(self) -> dict[str, dict[str, Any]]:
        return {f["path"]: f for f in self.fact("config")["fields"]}

    def field(self, path: str) -> dict[str, Any] | None:
        return self._fields.get(path)

    def fields(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """The field at ``query``, else fields whose path contains it."""
        exact = self._fields.get(query)
        if exact:
            return [
                exact,
                *[f for p, f in self._fields.items() if p.startswith(f"{query}.")][: limit - 1],
            ]
        needle = query.lower()
        return [f for p, f in self._fields.items() if needle in p.lower()][:limit]

    def decoding_types(self) -> dict[str, str]:
        return dict(self.fact("config")["decoding_types"])

    def unknown_fields(self, options: Mapping[str, Any]) -> list[str]:
        """Dotted paths in ``options`` the schema does not declare (empty without a schema)."""
        removed = self.fact("config").get("removed", [])
        return config_mod.unknown_fields(self._fields.values(), options, removed=removed)

    # -- ModelingV2 ------------------------------------------------------------------

    def families(self) -> list[dict[str, Any]]:
        return list(self.fact("families"))

    def family_for(self, architecture: str) -> str | None:
        """The family directory that routes ``architecture``, else ``None``."""
        return next(
            (f["family"] for f in self.fact("families") if architecture in f["architectures"]),
            None,
        )

    def family(self, name: str) -> dict[str, Any] | None:
        return next(
            (f for f in self.fact("families") if name == f["family"] or name in f["architectures"]),
            None,
        )

    def ops(self, query: str = "", limit: int = 20) -> list[dict[str, Any]]:
        needle = query.lower()
        hits = [
            op
            for op in self.fact("ops")
            if needle in op["name"].lower()
            or needle in op["module"].lower()
            or any(needle in call.lower() for call in op["calls"])
        ]
        return hits[:limit]

    # -- tests and CI ----------------------------------------------------------------

    def ci(self, test: str) -> dict[str, Any]:
        """The test-list entries that run ``test``, its waives and the stages of its lists."""
        entries = [
            e
            for e in self.fact("test_lists")
            if tests_mod.lists(e["path"], e["node"], test, e["ignore"])
        ]
        stages = self.fact("ci_stages")
        waived = [w for w in self.fact("waives") if tests_mod.lists(w["path"], w["node"], test)]
        return {
            "test": test,
            "entries": entries,
            "waives": waived,
            "stages": {e["list"]: stages.get(e["list"], []) for e in entries},
        }

    def listed(self, test: str, kind: str = "l0") -> bool:
        """Whether a ``kind`` list (``l0``: what CI runs; ``qa``) runs ``test``."""
        return any(
            e["kind"] == kind and tests_mod.lists(e["path"], e["node"], test, e["ignore"])
            for e in self.fact("test_lists")
        )

    def _files_with(
        self, words: Iterable[str], roots: Iterable[str], limit: int
    ) -> list[dict[str, Any]]:
        words = [w for w in words if w]
        hits = []
        for root in roots:
            base = self.repo / root
            for path in sorted(base.rglob("*.py")) if base.is_dir() else ():
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                count = sum(text.count(w) for w in words)
                if count:
                    hits.append((count, path.relative_to(self.repo).as_posix()))
        hits.sort(key=lambda item: (-item[0], item[1]))
        return [{"path": path, "hits": count} for count, path in hits[:limit]]

    # -- accuracy, recipes, environment ----------------------------------------------

    def accuracy_task(self, name: str) -> dict[str, Any] | None:
        needle = name.lower()
        return next(
            (
                t
                for t in self.fact("accuracy_tasks")
                if needle in (t["task"].lower(), str(t.get("dataset", "")).lower())
            ),
            None,
        )

    def references(self, model: str, benchmark: str = "") -> dict[str, dict[str, list[dict]]]:
        """``{benchmark: {model_key: rows}}`` for model keys containing ``model``."""
        needle = model.lower()
        found: dict[str, dict[str, list[dict]]] = {}
        for bench, models in self.fact("accuracy_references").items():
            if benchmark and bench != benchmark:
                continue
            rows = {k: v for k, v in models.items() if needle in k.lower()}
            if rows:
                found[bench] = rows
        return found

    def recipes(self, **query: Any) -> list[dict[str, Any]]:
        return recipes_mod.rank_recipes(self.fact("recipes"), **query)

    def env_vars(self, query: str = "", limit: int = 50) -> list[dict[str, Any]]:
        needle = query.upper()
        return [v for v in self.fact("env_vars") if needle in v["name"]][:limit]

    # -- features --------------------------------------------------------------------

    def feature(self, name: str, *, limit: int = 12) -> dict[str, Any] | None:
        """What the checkout holds about one feature (``features.FEATURES``)."""
        key = features_mod.resolve(name)
        if key is None:
            return None
        spec = features_mod.FEATURES[key]
        fields = [self._fields.get(p) or {"path": p, "missing": True} for p in spec.fields]
        tests = self._files_with(spec.words, _TEST_ROOTS, limit)
        for test in tests:
            test["lists"] = sorted({e["list"] for e in self.ci(test["path"])["entries"]})
        env_vars = [v for v in self.fact("env_vars") if any(w in v["name"] for w in spec.env_words)]
        return {
            "feature": key,
            "title": spec.title,
            "fields": fields,
            "tests": tests,
            "modeling_v2": self._files_with(spec.words, (MODELING_V2_REL,), limit),
            "env_vars": env_vars[:limit],
        }
