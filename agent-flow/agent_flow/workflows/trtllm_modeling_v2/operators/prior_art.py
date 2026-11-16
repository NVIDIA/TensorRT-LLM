# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the engine already has for a checkpoint, read off the repository before any agent turn.

No engine import, no GPU.

- :func:`engine_model` names the engine's own model of the checkpoint's architecture,
  found by its registration: the class and what its module defines. It is
  the first thing a survey and a card are offered: the reference to copy from and to
  compare against, which a target may not import (:mod:`.v1_rule`).
- :func:`engine_constraints` states what the engine's kernels refuse for this geometry,
  read from the source that refuses it, with the design it usually implies.
- :func:`engine_kv_dtypes` is the KV-cache dtypes the engine's loader accepts.

Each is read through an anchor (:mod:`.tree_anchors`): the files that hold its
definition, wherever this checkout keeps them.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import tree_anchors
from .v1_rule import registered_classes

#: Why the engine's own model is offered, and what a target may do with it.
ENGINE_MODEL_REASON = (
    "the engine's own implementation of this architecture: the reference for behaviour and "
    "numerics. A ModelingV2 target may not import it (the ModelingV2 guide): read it, copy "
    "what you need from it into the target, and compare against it"
)
#: The names of each list an :class:`EngineModel` shows.
ENGINE_MODEL_NAMES = 12


@dataclass(frozen=True)
class EngineModel:
    """The engine's own (Modeling V1) model of the checkpoint's architecture, as names to search."""

    #: The checkpoint's architecture the engine registers a model for, and that class.
    architecture: str
    cls: str
    #: Where the registration was when it was read: evidence about one commit.
    path: str
    #: The classes the registered class's own module defines (its layers, its weight loader).
    defines: tuple[str, ...] = ()

    def search(self) -> str:
        """How the registered class is found, wherever its file is."""
        return f'register_auto_model("{self.architecture}")'

    def render(self, *, where: bool = True) -> str:
        """The first lead: the names to search for and why, in lines."""

        def listed(names: Sequence[str]) -> str:
            return ", ".join(f"`{name}`" for name in names[:ENGINE_MODEL_NAMES])

        lines = [
            f"**The engine registers a model for `{self.architecture}`**: `{self.cls}`, "
            f"{ENGINE_MODEL_REASON}.",
            f"- find it by its registration: `git grep -n '{self.search()}'`"
            + (f"; found in `{self.path}`" if where else ""),
        ]
        if len(self.defines) > 1:
            rest = [name for name in self.defines if name != self.cls]
            lines.append(f"- its module also defines: {listed(rest)}")
        return "\n".join(lines)


@dataclass
class PriorArt:
    """Whether the engine has its own model of a checkpoint's architecture (:func:`find_prior_art`)."""

    #: The checkpoint's `architectures`, and the engine's own model of one of them.
    #: ``registered``: whether the engine registers a model for one; ``None``: not known
    #: (the checkout has no built-in model, or the config names no architecture).
    architectures: tuple[str, ...] = ()
    registered: bool | None = None
    engine_model: EngineModel | None = None
    #: The classes registered under a name that is not written out: one may be this one's.
    unnamed: tuple[str, ...] = ()

    def reference(self, *, where: bool = True) -> str:
        """Whether the engine has its own model of this architecture, and its names; ``""``: not known."""
        if self.engine_model is not None:
            return self.engine_model.render(where=where)
        if self.registered is False and self.architectures:
            named = ", ".join(f"`{a}`" for a in self.architectures)
            unless = ", ".join(f"`{cls}`" for cls in self.unnamed)
            return (
                f"**The engine registers no model for {named}**: there is no implementation of "
                "this architecture in the engine to copy from. The HuggingFace reference is the "
                "ground truth."
                + (
                    f" (Registered under a name that is not written out: {unless}.)"
                    if unless
                    else ""
                )
            )
        return ""


def _registered(repo: Path) -> dict[str, tuple[str, str]]:
    """Each architecture the engine registers a model for: ``{architecture: (file, class)}``.

    Read where the registrations are concentrated: an example that registers a model of
    its own elsewhere is not a model the engine ships.
    """
    zoo = tree_anchors.resolve(repo, "builtin_models")
    found: dict[str, tuple[str, str]] = {}
    for rel in zoo.paths:
        if zoo.homes and not any(rel.startswith(f"{home}/") for home in zoo.homes):
            continue
        try:
            text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for architecture, cls in registered_classes(text):
            found.setdefault(architecture, (rel, cls))
    return found


def engine_model(
    repo: str | Path,
    config: Mapping[str, Any],
    known: Mapping[str, tuple[str, str]] | None = None,
) -> EngineModel | None:
    """The engine's own model of the checkpoint's architecture; ``None`` when it registers none.

    The class is found by its registration, never by a path, and its names are the public
    classes its own module defines.
    """
    repo = Path(repo)
    known = _registered(repo) if known is None else known
    architecture = next((str(a) for a in config.get("architectures") or () if str(a) in known), "")
    if not architecture:
        return None
    rel, cls = known[architecture]
    try:
        text = (repo / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    defines = [n for n in re.findall(r"^class (\w+)", text, re.M) if not n.startswith("_")]
    return EngineModel(architecture, cls, rel, tuple(dict.fromkeys([cls, *defines])))


def find_prior_art(repo: str | Path, config: Mapping[str, Any]) -> PriorArt:
    """Whether the tree at ``repo`` has the engine's own model of ``config``'s architecture, and which."""
    found = PriorArt(architectures=tuple(str(a) for a in config.get("architectures") or ()))
    try:
        known = _registered(Path(repo))
        found.engine_model = engine_model(repo, config, known)
    except OSError:
        return found
    if known and found.architectures:
        found.registered = found.engine_model is not None
        found.unnamed = tuple(cls for name, (_, cls) in known.items() if not name.isidentifier())
    return found


def head_dims(config: Mapping[str, Any]) -> dict[str, int]:
    """Every attention head dimension the config states, by where it says so."""
    found: dict[str, int] = {}

    def walk(node: Any, where: str) -> None:
        if not isinstance(node, Mapping):
            return
        for key, value in node.items():
            here = f"{where}.{key}" if where else str(key)
            if isinstance(value, Mapping):
                walk(value, here)
            elif key.startswith("head_dim") and isinstance(value, int) and value > 0:
                found[here] = value
        if "head_dim" not in node and isinstance(node.get("hidden_size"), int):
            heads = node.get("num_attention_heads")
            if isinstance(heads, int) and heads > 0 and node["hidden_size"] % heads == 0:
                found[f"{where}.hidden_size/num_attention_heads".lstrip(".")] = (
                    node["hidden_size"] // heads
                )

    walk(config, "")
    return found


def fmha_rule(repo: str | Path | None) -> tuple[tuple[int, ...], int, str] | None:
    """The head dimensions the trtllm-gen FMHA has kernels for and ``path:line`` it was read at.

    ``None`` when this checkout states the rule nowhere: no limit is then
    claimed, and the lint names the missing anchor.
    """
    found = tree_anchors.read(repo, "fmha_head_dims")
    if found is None:
        return None
    _, match, where = found
    listed = tuple(int(v) for v in re.findall(r"\d+", match.group(1)))
    return listed, int(match.group(2)), where


def engine_constraints(config: Mapping[str, Any], repo: str | Path | None = None) -> list[str]:
    """What the engine's kernels refuse for this checkpoint's geometry, each with its usual way out.

    One limit is read: the head dimensions the trtllm-gen FMHA has no kernel for.
    """
    rule = fmha_rule(repo)
    if rule is None:
        return []
    listed, multiple, source = rule
    unsupported: dict[int, list[str]] = {}
    for where, dim in sorted(head_dims(config).items()):
        if dim not in listed and dim % multiple:
            unsupported.setdefault(dim, []).append(f"`{where}`")
    limits = []
    for dim, wheres in sorted(unsupported.items()):
        padded = next(d for d in (*sorted(listed), *(multiple * k for k in range(1, 9))) if d > dim)
        limits.append(
            f"{', '.join(wheres)} = {dim}: the trtllm-gen FMHA has no kernel for head dim {dim} "
            f"({source}: {', '.join(map(str, listed))} or a multiple of {multiple}). The "
            f"paged-context path chunked prefill and block reuse take needs one: zero-pad the "
            f"heads to {padded} in the KV pools and keep the softmax scale at 1/sqrt({dim}) "
            f"(q_scaling = sqrt({dim}/{padded}) in the attention op's convention). The other "
            f"way is a new native kernel, which no run builds: it is a rebuild of the base."
        )
    return limits


def reference_for(model: EngineModel | None) -> str:
    """The engine's own model as one lead of a card: its class and the search that finds it.

    ``""`` when the engine registers no model for the checkpoint's architecture. A search
    and a reason, as every lead: no place.
    """
    if model is None:
        return ""
    return (
        f"`{model.cls}` (the engine's own implementation of this architecture; search "
        f"`{model.search()}`): copy what you need from it into the target; do not import it. "
        "Compare against it"
    )


#: The engine's own model as a run found it, as data, in its workspace: what each card's
#: first lead is made from (:func:`reference_for`).
PRIOR_ART_DATA = "prior_art.json"


def save(found: PriorArt | None, path: str | Path) -> None:
    """Keep the engine's own model at ``path`` as data; remove the file when there is none."""
    path = Path(path)
    if found is None or found.engine_model is None:
        path.unlink(missing_ok=True)
        return
    path.write_text(json.dumps(asdict(found.engine_model), indent=1), encoding="utf-8")


def load(path: str | Path) -> EngineModel | None:
    """What :func:`save` kept at ``path``; ``None`` when there is no such file or it is not that."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return EngineModel(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
    except (OSError, ValueError, TypeError, AttributeError):
        return None


#: The heading of the survey dispatch's section.
PRIOR_ART_HEADING = "## The engine's own model and its limits for this checkpoint"


def survey_block(found: PriorArt | None, limits: Sequence[str]) -> str:
    """What the survey's dispatch carries: the engine's own model of the architecture, and its limits."""
    first = found.reference() if found is not None else ""
    if not limits and not first:
        return ""
    lines = [PRIOR_ART_HEADING, ""]
    if first:
        lines += [first, ""]
    if limits:
        lines += [
            "**Engine limits for this geometry**: `plan.md` says how the target gets past each.",
            "",
        ]
        lines += [f"- {text}" for text in limits]
    return "\n".join(lines).rstrip() + "\n"


#: The value of the PyTorch loader's list of ``kv_cache_config.dtype`` values.
_KV_DTYPES = re.compile(r"_VALID_KV_CACHE_DTYPES\s*(?::[^=]*)?=\s*(\([^)]*\)|\[[^\]]*\])")


def engine_kv_dtypes(repo: str | Path | None) -> tuple[tuple[str, ...], str] | None:
    """The KV-cache dtypes the engine accepts and where that was read, or ``None`` when unknown."""
    found = tree_anchors.read(repo, "kv_cache_dtypes")
    if found is None:
        return None
    text, marker, where = found
    match = _KV_DTYPES.match(text, marker.start())
    try:
        values = tuple(str(v) for v in ast.literal_eval(match.group(1))) if match else ()
    except (ValueError, SyntaxError):
        return None
    return (values, where) if values else None


__all__ = [
    "PRIOR_ART_DATA",
    "PRIOR_ART_HEADING",
    "EngineModel",
    "PriorArt",
    "engine_constraints",
    "engine_kv_dtypes",
    "engine_model",
    "find_prior_art",
    "fmha_rule",
    "head_dims",
    "load",
    "reference_for",
    "save",
    "survey_block",
]
