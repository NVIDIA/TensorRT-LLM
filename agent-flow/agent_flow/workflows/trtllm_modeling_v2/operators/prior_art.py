# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prior art for a checkpoint, found in the repository before any agent turn.

Intake reads it from disk only: no engine import, no GPU. Two products:

- :func:`find_prior_art` ranks what the tree already implements against the
  words the checkpoint's ``config.json`` uses (``sparse``, ``indexer``,
  ``topk``, ``compress`` ...): other targets' ``ops/``, the shared catalog,
  the attention backends and the custom ops. A match is a lead for the
  survey, not a verdict: the surveyor still reads the entry's cells.
- :func:`engine_constraints` states what the engine's kernels refuse for this
  geometry, read from the source that refuses it (a head dimension the
  trtllm-gen FMHA has no kernel for, say), with the design it usually implies.
  Each limit is one :data:`ENGINE_RULES` entry.
- :func:`stock_supports` says whether the stock modeling path registers the
  checkpoint's architecture, which decides whether preflight's substrate gate
  needs a proxy checkpoint.

All are printed by ``trtllm-modeling-v2 intake lint`` so the operator can put
the leads that hold into the spec's ``notes``, which every role reads.

Every place read off the tree is an anchor (:mod:`.tree_anchors`): found where
it was last seen, else by what it defines. One the checkout has nowhere is
left out and reported (:data:`PRIOR_ART_ANCHORS`), never replaced by a value
remembered here.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import tree_anchors
from .coverage import CATALOG_DIR, MODEL_OPS_DIRNAME, MODELING_V2_PACKAGE

#: Where the tree keeps implementations a target may reuse, and what each is:
#: ModelingV2's own, then the engine's, each an anchor.
TARGETS_GLOB = f"{MODELING_V2_PACKAGE}/models/*/*/{MODEL_OPS_DIRNAME}"
ENGINE_ANCHORS: tuple[tuple[str, str], ...] = (
    ("attention_backends", "attention backend"),
    ("custom_ops", "custom op"),
    ("kernels", "kernel"),
    ("moe_modules", "MoE module"),
)
KIND_TARGET_OP = "another target's op"
KIND_CATALOG = "catalog entry"

#: Words too common in configs and code to say anything about an architecture.
STOPWORDS = frozenset(
    {
        "size",
        "num",
        "dim",
        "dims",
        "type",
        "types",
        "true",
        "false",
        "none",
        "model",
        "config",
        "hidden",
        "layer",
        "max",
        "min",
        "use",
        "enabled",
        "enable",
        "list",
        "factor",
        "theta",
        "token",
        "len",
        "length",
        "eps",
        "dtype",
        "float32",
        "float16",
        "bfloat16",
        "torch",
        "the",
        "and",
        "for",
        "impl",
        "attention",
        "head",
        "norm",
        "group",
        "block",
        "method",
        "setting",
        "other",
        "architecture",
        "init",
        "value",
        "producer",
        "version",
        "ignore",
        "quant",
        "algo",
        "name",
        "path",
        "kernel",
        "op",
        "ops",
        "test",
        "tests",
        "util",
        "utils",
        "base",
        "default",
        "common",
        "module",
        "backend",
        "custom",
        "fused",
        "triton",
        "torch",
        "flashinfer",
        "trtllm",
        "tensorrt",
        "llm",
        "local",
        "shared",
        "load",
        "weight",
        "output",
        "input",
        "apply",
        "with",
        "from",
        "into",
        "per",
        "all",
        "out",
        "position",
        "embedding",
        "embeddings",
        "act",
        "select",
        "start",
        "end",
        "id",
        "ids",
        "only",
        "original",
        "cache",
        "pool",
        "cls",
        "need",
        "generation",
        "fp32",
        "top",
        "seq",
        "causal",
        "quantization",
    }
)
_SPLIT = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z])(?=[A-Z])")
_DEF = re.compile(r"^(?:def|class) ([A-Za-z_][A-Za-z0-9_]*)", re.M)


def _stem(word: str) -> str:
    """One light suffix strip, so ``indexer`` meets ``index`` and ``experts`` meets ``expert``."""
    for suffix in ("ation", "ion", "ers", "er", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def words(text: str) -> set[str]:
    """The stemmed, lower-case words of an identifier or a phrase, stop words dropped."""
    out = set()
    for raw in _SPLIT.split(text):
        word = raw.lower()
        if len(word) < 3 or word.isdigit():
            continue
        stem = _stem(word)
        if word not in STOPWORDS and stem not in STOPWORDS:
            out.add(stem)
    return out


def config_words(config: Mapping[str, Any]) -> set[str]:
    """Every word the config's keys and string values use, nested blocks included."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                found.update(words(str(key)))
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node[:8]:
                walk(item)
        elif isinstance(node, str):
            found.update(words(node))

    walk(config)
    return found


@dataclass(frozen=True)
class Candidate:
    """One implementation the tree already has, and why it may serve this checkpoint."""

    path: str
    kind: str
    matched: tuple[str, ...]
    score: float
    names: tuple[str, ...] = ()

    def render(self) -> str:
        names = f" ({', '.join(self.names[:4])})" if self.names else ""
        return f"- `{self.path}`{names} — {self.kind}; matches: {', '.join(self.matched)}"


@dataclass
class PriorArt:
    """What :func:`find_prior_art` found, best first, per kind."""

    words: tuple[str, ...]
    candidates: list[Candidate] = field(default_factory=list)

    def by_kind(self, kind: str) -> list[Candidate]:
        return [c for c in self.candidates if c.kind == kind]

    def render(self, limit: int = 12) -> str:
        if not self.candidates:
            return "No implementation in the tree shares the checkpoint's vocabulary."
        lines: list[str] = []
        for kind in dict.fromkeys(c.kind for c in self.candidates):
            ranked = self.by_kind(kind)
            lines.append(f"**{kind}** ({len(ranked)}):")
            lines += [c.render() for c in ranked[:limit]]
            if len(ranked) > limit:
                lines.append(f"- ... {len(ranked) - limit} more")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def _python_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return ()
    return (
        p for p in sorted(root.rglob("*.py")) if p.name != "__init__.py" and "test" not in p.name
    )


def _target_ops(repo: Path) -> list[tuple[str, str, set[str], tuple[str, ...]]]:
    found = []
    for ops in sorted(repo.glob(TARGETS_GLOB)):
        for path in _python_files(ops):
            rel = path.relative_to(repo).as_posix()
            found.append((rel, KIND_TARGET_OP, words(path.stem) | words(path.parent.name), ()))
    return found


def _catalog(repo: Path) -> list[tuple[str, str, set[str], tuple[str, ...]]]:
    root = repo / CATALOG_DIR
    return [
        (p.relative_to(repo).as_posix(), KIND_CATALOG, words(p.stem) | words(p.parent.name), ())
        for p in _python_files(root)
        if not p.name.startswith("_")
    ]


def _engine(repo: Path) -> list[tuple[str, str, set[str], tuple[str, ...]]]:
    found = []
    for key, kind in ENGINE_ANCHORS:
        for rel_dir in tree_anchors.resolve(repo, key).paths:
            for path in _python_files(repo / rel_dir):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                names = tuple(n for n in _DEF.findall(text) if not n.startswith("_"))
                vocab = words(path.relative_to(repo / rel_dir).as_posix())
                for name in names:
                    vocab |= words(name)
                found.append((path.relative_to(repo).as_posix(), kind, vocab, names))
    return found


def find_prior_art(repo: str | Path, config: Mapping[str, Any], *, min_words: int = 2) -> PriorArt:
    """What the tree at ``repo`` implements that shares ``config``'s vocabulary, best first.

    A candidate shares at least ``min_words`` words with the config; each word
    scores by its rarity across every candidate (inverse document frequency),
    so ``sparse`` + ``gqa`` + ``topk`` outrank common words.
    Engine files list the public functions that matched.
    """
    repo = Path(repo)
    wanted = config_words(config)
    pool = _target_ops(repo) + _catalog(repo) + _engine(repo)
    if not pool or not wanted:
        return PriorArt(words=tuple(sorted(wanted)))
    frequency: dict[str, int] = {}
    for _, _, vocab, _ in pool:
        for word in vocab & wanted:
            frequency[word] = frequency.get(word, 0) + 1
    total = len(pool)
    candidates = []
    for rel, kind, vocab, names in pool:
        matched = sorted(vocab & wanted)
        if not matched:
            continue
        score = sum(math.log(1 + total / frequency[w]) for w in matched)
        if len(matched) < min_words:
            continue
        hits = tuple(n for n in names if words(n) & wanted and len(words(n) & wanted) >= 2)
        if names and not hits:
            continue
        candidates.append(Candidate(rel, kind, tuple(matched), round(score, 2), hits))
    candidates.sort(key=lambda c: (-c.score, c.path))
    return PriorArt(words=tuple(sorted(wanted)), candidates=candidates)


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


@dataclass(frozen=True)
class EngineLimit:
    """One thing the engine's kernels refuse for this geometry, and how a plan gets past it."""

    #: The sentence: what is refused, where that is read, and the usual way out.
    text: str
    #: The value a plan that addressed it names, e.g. ``192``.
    subject: str
    #: Words one of which that plan also names: the ways out, e.g. ``("256", "native")``.
    ways_out: tuple[str, ...]


def fmha_head_dims(config: Mapping[str, Any], repo: str | Path | None = None) -> list[EngineLimit]:
    """Head dimensions the trtllm-gen FMHA has no kernel for, and the padding way out."""
    rule = fmha_rule(repo)
    if rule is None:
        return []
    listed, multiple, source = rule
    unsupported: dict[int, list[str]] = {}
    for where, dim in sorted(head_dims(config).items()):
        if dim not in listed and dim % multiple:
            unsupported.setdefault(dim, []).append(f"`{where}`")
    limits: list[EngineLimit] = []
    for dim, wheres in sorted(unsupported.items()):
        padded = next(d for d in (*sorted(listed), *(multiple * k for k in range(1, 9))) if d > dim)
        text = (
            f"{', '.join(wheres)} = {dim}: the trtllm-gen FMHA has no kernel for head dim {dim} "
            f"({source}: {', '.join(map(str, listed))} or a multiple of {multiple}). The "
            f"paged-context path chunked prefill and block reuse take needs one: zero-pad the "
            f"heads to {padded} in the KV pools and keep the softmax scale at 1/sqrt({dim}) "
            f"(q_scaling = sqrt({dim}/{padded}) in the attention op's convention), or plan a "
            f"new native kernel. Native work costs rebuilds; padding usually does not."
        )
        limits.append(EngineLimit(text, str(dim), (str(padded), "native")))
    return limits


EngineRule = Callable[[Mapping[str, Any], "str | Path | None"], list[EngineLimit]]

#: Engine limits checked against a checkpoint's geometry, each read from the source that
#: enforces it. A new limit (a GEMM shape, a KV dtype, a rope variant) adds its function here.
ENGINE_RULES: tuple[EngineRule, ...] = (fmha_head_dims,)


def engine_limits(config: Mapping[str, Any], repo: str | Path | None = None) -> list[EngineLimit]:
    """What the engine's kernels refuse for this checkpoint's geometry (:data:`ENGINE_RULES`)."""
    return [limit for rule in ENGINE_RULES for limit in rule(config, repo)]


def engine_constraints(config: Mapping[str, Any], repo: str | Path | None = None) -> list[str]:
    """What the engine's kernels refuse for this checkpoint's geometry, each with its usual way out."""
    return [limit.text for limit in engine_limits(config, repo)]


def _op_words(name: str) -> set[str]:
    """A listed implementation's words: its category and name, not the directory of its owner."""
    return words(name.split("/ops/", 1)[-1].split(" ->", 1)[0].replace("/", " "))


def ranker(config: Mapping[str, Any], names: Iterable[str]) -> Callable[[str], float]:
    """How much each of ``names`` shares ``config``'s vocabulary: rarer shared words weigh more.

    For ordering a long list before it is cut (the surface survey's), so the
    implementations this checkpoint is likely to reuse come first whatever
    their names sort as. A name outside ``names`` scores by the same weights.
    """
    wanted = config_words(config)
    pool = [_op_words(n) for n in names]
    frequency: dict[str, int] = {}
    for vocab in pool:
        for word in vocab & wanted:
            frequency[word] = frequency.get(word, 0) + 1
    total = max(len(pool), 1)

    def score(name: str) -> float:
        return sum(math.log(1 + total / frequency.get(w, 1)) for w in _op_words(name) & wanted)

    return score


#: The heading of the survey dispatch's prior-art section.
PRIOR_ART_HEADING = "## Prior art and engine limits for this checkpoint"


def survey_block(found: PriorArt | None, limits: Sequence[EngineLimit], *, limit: int = 5) -> str:
    """What the surveyor reads before it maps a step to a GAP: the leads, and the engine's limits."""
    lines = [PRIOR_ART_HEADING, ""]
    if not limits and (found is None or not found.candidates):
        return ""
    lines.append(
        "Read off this tree and the checkpoint's `config.json` before this phase started "
        "(`prior_art.md` in the workspace has the longer list). A lead is where to look, not "
        "a verdict: read its code and cells. A step you map to a GAP that writes new code "
        "names in `tried` each lead below that shares its words and why it does not serve."
    )
    if limits:
        lines += [
            "",
            "**Engine limits for this geometry** -- state the design for each in `plan.md`:",
            "",
        ]
        lines += [f"- {limit.text}" for limit in limits]
    if found is not None and found.candidates:
        lines += ["", "**Implementations this tree already has**, ranked by shared vocabulary:", ""]
        lines.append(found.render(limit=limit).rstrip())
    return "\n".join(lines) + "\n"


#: How a stock (non-modeling_v2) model module registers an architecture.
_REGISTERED = re.compile(r"""register_auto_model\(\s*["']([^"']+)["']""")


def stock_architectures(repo: str | Path) -> set[str] | None:
    """The architectures the stock modeling path registers, or ``None`` when the tree has no zoo."""
    zoo = tree_anchors.resolve(repo, "builtin_models")
    index = tree_anchors.read(repo, "builtin_model_index")
    if not zoo.found and index is None:
        return None
    found: set[str] = set()
    for rel in zoo.paths:
        for path in sorted((Path(repo) / rel).rglob("*.py")):
            try:
                found.update(
                    _REGISTERED.findall(path.read_text(encoding="utf-8", errors="replace"))
                )
            except OSError:
                continue
    try:
        tree = ast.parse(index[0]) if index is not None else None
    except (SyntaxError, ValueError):
        tree = None
    for node in ast.walk(tree) if tree is not None else ():
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == "MODEL_ARCH_TO_MODULE":
            try:
                found.update(str(k) for k in ast.literal_eval(node.value))
            except (ValueError, TypeError):
                pass
    return found


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


def stock_supports(repo: str | Path, config: Mapping[str, Any]) -> bool | None:
    """Whether the stock path registers one of the checkpoint's ``architectures``; ``None``: unknown."""
    architectures = [str(a) for a in config.get("architectures") or ()]
    known = stock_architectures(repo)
    if not architectures or not known:
        return None
    return any(a in known for a in architectures)


#: The anchors this module reads; the lint names each this checkout has nowhere.
PRIOR_ART_ANCHORS: tuple[str, ...] = (
    *(key for key, _ in ENGINE_ANCHORS),
    "fmha_head_dims",
    "builtin_models",
    "builtin_model_index",
    "kv_cache_dtypes",
)


__all__ = [
    "ENGINE_ANCHORS",
    "ENGINE_RULES",
    "PRIOR_ART_ANCHORS",
    "PRIOR_ART_HEADING",
    "Candidate",
    "EngineLimit",
    "PriorArt",
    "config_words",
    "engine_constraints",
    "engine_kv_dtypes",
    "engine_limits",
    "find_prior_art",
    "fmha_head_dims",
    "fmha_rule",
    "head_dims",
    "ranker",
    "stock_architectures",
    "stock_supports",
    "survey_block",
    "words",
]
