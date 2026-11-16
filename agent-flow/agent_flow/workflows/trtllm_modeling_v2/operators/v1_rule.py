# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a ModelingV2 target may not import: Modeling V1 model code, as the guide defines it.

A target may read the engine's own implementation of an architecture or of a computation,
and compare against it. It may not import, subclass or call it. The rule is the engine's:
the ModelingV2 developer guide states it, with the places Modeling V1 model code lives in
and the runtime interfaces a target does use.

Nothing here remembers a place. :func:`rule` reads the guide (found by its title,
:mod:`.tree_anchors`): the places are the ones its own sentence lists, resolved against
where the guide is, and the runtime interfaces are the names it gives. The modules that
register a model for an architecture are found by their registration. :func:`v1_imports`
then says, for one source file, which of its imports the rule covers and what each is:
read from the definition of the name that is imported, never from the name of a file.

It judges a target's own files. A test may import the original to compare the target's
copy against it: a caller that checks a target leaves the target's tests out. What it
does not see: an import made by a call (``importlib``), and an attribute reached through
a package that was imported whole.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from . import tree_anchors

#: The guide's sentence: what a target does not import, and the places it lists.
_RULE = re.compile(r"does not import[^.:]*Modeling V1 model code:\s*(?P<list>[^.]*)\.", re.S | re.I)
#: Where the guide gives the runtime interfaces a target does use, before that sentence.
_INTERFACES = re.compile(r"runtime interfaces", re.I)
_TICKED = re.compile(r"`([^`\s]+)`")
#: How a class is registered as the model of an architecture, and the class statement after it.
#: The registration may be reached through a module, and its argument need not be a literal.
_REGISTRATION = re.compile(r"^@(?:\w+\.)*register_auto_model\(([^)\n]*)\)", re.M)
_CLASS = re.compile(r"^class (\w+)", re.M)
#: A class statement with its bases, and a name bound to another at the top of a module.
_CLASS_BASES = re.compile(r"^class (\w+)\b(?:\(([^)]*)\))?\s*:", re.M)
_ALIAS = re.compile(r"^(\w+) = (\w+)[ \t]*$", re.M)
#: What a layer is built on, in torch's own names.
_MODULE_BASES = frozenset({"Module", "ModuleList", "ModuleDict", "Sequential"})

#: What an import the rule covers is, in the guide's own terms.
MODEL_CLASS = "a model class the engine registers for an architecture"
MODEL_MODULE = "code of a module that registers a model"
ATTENTION_BACKEND = "an attention backend"
MOE_MODULE = "a fused MoE module"
LAYER = "a Modeling V1 layer"
#: The guide lists the place and not the thing: a function, a constant, a re-export.
LISTED_PLACE = "other code in a place the guide lists for Modeling V1 model code"


@dataclass(frozen=True)
class Rule:
    """The guide's rule as read from one checkout."""

    #: The guide, repository-relative.
    guide: str
    #: The rule in the guide's own words.
    sentence: str
    #: The places the sentence lists, as the guide writes them.
    listed: tuple[str, ...]
    #: The same, as directories of this checkout; one that is not there is left out.
    places: tuple[str, ...]
    #: The runtime interfaces the guide names: a target uses these.
    interfaces: tuple[str, ...]
    #: The files that register a model for an architecture, wherever they are.
    model_modules: tuple[str, ...]
    #: The classes those files register.
    registered: frozenset[str] = frozenset()
    #: Every class the places and the model modules define: ``{name: ((file, bases), ...)}``,
    #: and the names bound to another name there (``Fused = FusedBase``).
    classes: Mapping[str, tuple[tuple[str, str], ...]] = field(default_factory=dict, repr=False)
    aliases: Mapping[str, str] = field(default_factory=dict, repr=False)

    def covers(self, path: str) -> bool:
        """Whether the file at ``path`` is in a place the rule lists, or registers a model."""
        return path in self.model_modules or any(path.startswith(f"{p}/") for p in self.places)


@dataclass(frozen=True)
class V1Import:
    """One import of a source file that the rule covers."""

    line: int
    #: The module as the file imports it, and the name taken from it (empty: the module itself).
    module: str
    name: str
    #: Where the module is in this checkout.
    path: str
    #: What it is (:data:`MODEL_CLASS` and the others).
    what: str
    #: ``True``: the guide names this kind of thing. ``False``: it names only the place.
    certain: bool

    def render(self) -> str:
        said = f"`{self.name}` from `{self.module}`" if self.name else f"`{self.module}`"
        return f"line {self.line}: {said} is {self.what}"


def registered_classes(source: str) -> list[tuple[str, str]]:
    """The classes ``source`` registers as models: ``(architecture, class)``, in order.

    The class is the first one defined after the registration: other decorators, of
    any length, may stand between the two. A registration in a docstring is indented
    and is not one.
    """
    found = []
    for match in _REGISTRATION.finditer(source):
        cls = _CLASS.search(source, match.end())
        if cls:
            found.append((match.group(1).strip().strip("\"'"), cls.group(1)))
    return found


def _resolve_place(guide: str, listed: str, repo: Path) -> str:
    """A place the guide lists, as a directory of the checkout: resolved against the guide's own path."""
    parts = [p for p in listed.strip("/").split("/") if p]
    above = guide.split("/")[:-1]
    if parts and parts[0] in above:
        candidate = "/".join(above[: above.index(parts[0])] + parts)
        if (repo / candidate).is_dir():
            return candidate
    candidate = "/".join(parts)
    return candidate if parts and (repo / candidate).is_dir() else ""


def rule(repo: str | Path | None) -> Rule | None:
    """The guide's rule in the checkout at ``repo``; ``None`` when no guide states it."""
    found = tree_anchors.read(repo, "modeling_v2_guide")
    if found is None or repo is None:
        return None
    text, _, where = found
    guide = where.rsplit(":", 1)[0]
    stated = _RULE.search(text)
    if not stated:
        return None
    listed = tuple(t for t in _TICKED.findall(stated.group("list")) if "/" in t)
    places = tuple(p for p in (_resolve_place(guide, t, Path(repo)) for t in listed) if p)
    before = text[: stated.start()]
    start = [m.end() for m in _INTERFACES.finditer(before)]
    interfaces = tuple(
        dict.fromkeys(t for t in _TICKED.findall(before[start[-1] :]) if t.isidentifier())
    ) if start else ()  # fmt: skip
    registering, registered = [], set()
    for path in tree_anchors.resolve(repo, "builtin_models").paths:
        try:
            source = (Path(repo) / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found_here = registered_classes(source)
        if found_here:
            registering.append(path)
            registered.update(cls for _, cls in found_here)
    classes, aliases = _index(Path(repo), places, registering)
    return Rule(
        guide=guide,
        sentence=" ".join(stated.group(0).split()),
        listed=listed,
        places=places,
        interfaces=interfaces,
        model_modules=tuple(registering),
        registered=frozenset(registered),
        classes=classes,
        aliases=aliases,
    )


def _index(
    repo: Path, places: Iterable[str], modules: Iterable[str]
) -> tuple[dict[str, tuple[tuple[str, str], ...]], dict[str, str]]:
    """Every class the places and the model modules define, with its bases; and the aliases."""
    files = {m for m in modules}
    for place in places:
        try:
            files.update(p.relative_to(repo).as_posix() for p in (repo / place).rglob("*.py"))
        except OSError:
            continue
    classes: dict[str, list[tuple[str, str]]] = {}
    aliases: dict[str, str] = {}
    for rel in sorted(files):
        try:
            source = (repo / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for name, bases in _CLASS_BASES.findall(source):
            classes.setdefault(name, []).append((rel, bases))
        for name, other in _ALIAS.findall(source):
            if name != other:
                aliases.setdefault(name, other)
    return {name: tuple(found) for name, found in classes.items()}, aliases


def _heads(bases: str) -> list[str]:
    """The names a class is built on: a generic's parameters and keyword arguments are not bases."""
    plain = re.sub(r"\[[^\]]*\]", "", bases)
    return [b.strip().split(".")[-1] for b in plain.split(",") if b.strip() and "=" not in b]


def _kind(found: Rule, name: str, depth: int = 0) -> str:
    """What a class is through its ancestors: a backend, a fused MoE module, a layer, or ``""``."""
    if depth > 16:
        return ""
    name = found.aliases.get(name, name) if name not in found.classes else name
    layer = False
    for _, bases in found.classes.get(name, ()):
        heads = _heads(bases)
        if "AttentionBackend" in heads:
            return ATTENTION_BACKEND
        if "MoE" in heads or "MoEImplBase" in heads:
            return MOE_MODULE
        layer = layer or bool(_MODULE_BASES.intersection(heads))
        for head in heads:
            above = _kind(found, head, depth + 1) if head != name else ""
            if above in (ATTENTION_BACKEND, MOE_MODULE):
                return above
            layer = layer or above == LAYER
    return LAYER if layer else ""


def _module_file(repo: Path, parts: Iterable[str]) -> str:
    """The file a dotted module is in this checkout, repository-relative; ``""`` when it has none."""
    rel = "/".join(parts)
    for candidate in (f"{rel}.py", f"{rel}/__init__.py"):
        try:
            if rel and (repo / candidate).is_file():
                return candidate
        except OSError:
            return ""
    return ""


def _imports(tree: ast.AST, path: str) -> Iterable[tuple[int, str, list[str], str]]:
    """Every import of a file: ``(line, module as written, module parts, name)``.

    Not the imports only a type checker reads (``if TYPE_CHECKING:``).
    """
    package = path.split("/")[:-1]
    typing_only = {
        id(inner)
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test)
        for stmt in node.body
        for inner in ast.walk(stmt)
    }
    for node in ast.walk(tree):
        if id(node) in typing_only:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, alias.name.split("."), ""
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and node.level - 1 > len(package):
                continue
            base = package[: len(package) - (node.level - 1)] if node.level else []
            parts = base + (module.split(".") if module else [])
            written = "." * node.level + module
            for alias in node.names:
                yield node.lineno, written, parts, alias.name


def _definition(source: str, name: str) -> tuple[str, str]:
    """How ``name`` is defined in ``source``: ``("class", bases)``, ``("def", "")`` or ``("", "")``."""
    found = re.search(rf"^class {re.escape(name)}\b(?:\(([^)]*)\))?\s*:", source, re.M)
    if found:
        return "class", found.group(1) or ""
    if re.search(rf"^(?:async )?def {re.escape(name)}\b", source, re.M):
        return "def", ""
    return "", ""


def _reexported(repo: Path, path: str, source: str, name: str) -> tuple[str, str]:
    """Where a package's ``__init__`` takes ``name`` from, one step: ``(path, source)``."""
    if not path.endswith("/__init__.py"):
        return path, source
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return path, source
    for _, _, parts, taken in _imports(tree, path):
        if taken == name:
            target = _module_file(repo, parts)
            if target:
                return target, (repo / target).read_text(encoding="utf-8", errors="replace")
    return path, source


def _module_what(found: Rule, path: str, source: str) -> tuple[str, bool]:
    """What a module imported whole is: the kind of the classes it defines."""
    if path in found.model_modules:
        return MODEL_MODULE, True
    for name, _ in _CLASS_BASES.findall(source):
        if "Metadata" in name or "CacheManager" in name:
            continue
        kind = _kind(found, name)
        if kind:
            return kind, True
    return LISTED_PLACE, False


def _what(found: Rule, path: str, source: str, name: str) -> tuple[str, bool, str] | None:
    """What an imported name is, by its definition, and where that is; ``None`` for a runtime interface.

    A name the module does not define itself (a package that hands its classes on, a
    module that imports the name) is judged where the listed places define a class of
    that name.
    """
    if name in found.interfaces:
        return None
    if not name or name == "*":
        return (*_module_what(found, path, source), path)
    kind, bases = _definition(source, name)
    target = found.aliases.get(name, name) if not kind else name
    if not kind and target in found.classes:
        path, bases = found.classes[target][0]
        kind = "class"
    if kind != "class":
        # A function or a value: an op or a kernel a model's module defines may be called.
        return LISTED_PLACE, False, path
    if target in found.registered:
        return MODEL_CLASS, True, path
    through = _kind(found, target)
    if through in (ATTENTION_BACKEND, MOE_MODULE):
        return through, True, path
    # The guide's runtime interfaces: attention and speculative metadata, cache managers.
    if re.search(r"Metadata\b|CacheManager", f"{target} {' '.join(_heads(bases))}"):
        return None
    if path in found.model_modules:
        return MODEL_MODULE, True, path
    if through == LAYER:
        return LAYER, True, path
    return LISTED_PLACE, False, path


def v1_imports(
    repo: str | Path,
    path: str,
    *,
    text: str | None = None,
    found: Rule | None = None,
) -> list[V1Import]:
    """The imports of the file at ``path`` (repository-relative) that the guide's rule covers.

    ``text`` is the file's source when it is not the checkout's (a commit's version);
    ``found`` is :func:`rule`'s answer when the caller has it already. Empty when the
    checkout's guide states no such rule, when the file does not parse, and for a file
    with no such import. Each finding says what the imported thing is. One that is
    ``certain`` is a kind the guide names (a model class, a class of a model's own module,
    an attention backend, a fused MoE module, a layer, or a module of those imported
    whole); the others are in a place the guide lists and are not one of those kinds (a
    function, a kernel, a value): a person or a reviewer decides them.
    """
    repo = Path(repo)
    found = found or rule(repo)
    if found is None:
        return []
    try:
        source = (repo / path).read_text(encoding="utf-8") if text is None else text
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return []
    out: list[V1Import] = []
    for line, written, parts, name in _imports(tree, path):
        # `from package import module` names a module, not a thing a module defines.
        module = _module_file(repo, [*parts, name]) if name and name != "*" else ""
        taken = "" if module else name
        module = module or _module_file(repo, parts)
        if not module or not found.covers(module):
            continue
        try:
            defined = (repo / module).read_text(encoding="utf-8", errors="replace")
            if taken and taken != "*":
                module, defined = _reexported(repo, module, defined, taken)
        except OSError:
            continue
        what = _what(found, module, defined, taken)
        if what is not None:
            kind, certain, at = what
            out.append(V1Import(line, written, name, at, kind, certain))
    return sorted(out, key=lambda item: (item.line, item.name))


__all__ = [
    "ATTENTION_BACKEND",
    "LAYER",
    "LISTED_PLACE",
    "MODEL_CLASS",
    "MODEL_MODULE",
    "MOE_MODULE",
    "Rule",
    "V1Import",
    "registered_classes",
    "rule",
    "v1_imports",
]
