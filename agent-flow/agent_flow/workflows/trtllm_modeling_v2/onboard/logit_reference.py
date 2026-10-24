#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What the Hugging Face implementation predicts at every position of a fixed prompt set.

This is the logit gate's reference (``checks/logit_agreement.py``). Run it
once, in the anchors phase, under an interpreter that loads the checkpoint
with ``transformers`` (the run's HF reference environment)::

    python logit_reference.py --model <checkpoint> --out logits.reference.json.gz \
        [--prompts prompts.yaml] [--[no-]trust-remote-code] [--device-map auto]

For each prompt it records the top-k token ids and log-probs at every position:

* over the prompt, teacher-forced: one forward over the prompt, where position
  ``p`` predicts token ``p + 1``;
* over ``--continuation`` greedy tokens, each from a forward over the prompt
  and the tokens before it. Every step recomputes the whole sequence, so an
  implementation without an incremental cache runs too, and nothing ends the
  continuation early: an end token is a token like any other.

It then loads the checkpoint again in ``--spread-dtype`` (float32 by default)
and runs the same sequences teacher-forced, to measure how far the reference
moves from its own higher-precision self: the top-1 agreement and the
|delta log-prob| of the reference's top-1 token, per prompt and pooled. The gate
derives its thresholds from that spread.

The prompts, by default: three short raw-text prompts and one long prompt of
``--long-tokens`` tokens (one context chunk past TensorRT-LLM's default
``max_num_tokens``), cut from the standard library's own documentation, plus an
image prompt per ``--image``. A ``--prompts`` file replaces them: a YAML or
JSON list of ``{id, text}``, ``{id, text, images}`` or ``{id, long_tokens,
text_file?}``. An image prompt's text is raw (no chat template) and carries
``--image-placeholder`` where each image goes; the processor's own image token
replaces it here, and the engine receives the text as written. Image paths are
stored as given, with the digest of the file; the gate resolves a relative one
against the tree under test, so give paths relative to that tree and point
``--image-root`` at it.

The file is JSON, gzip-compressed when its name ends in ``.gz``, and versioned
(:data:`FORMAT`, :data:`VERSION`). Everything above the measurement section
uses the standard library only: the gate reads a file with
:func:`load_reference` in an interpreter without torch.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Collection, Iterable, Mapping, Protocol, Sequence

#: What the file says it is, and the version of its layout.
FORMAT = "trtllm-modeling-v2/logit-reference"
VERSION = 1

KIND_TEXT = "text"
KIND_LONG = "long"
KIND_IMAGE = "image"
KINDS: tuple[str, ...] = (KIND_TEXT, KIND_LONG, KIND_IMAGE)

#: Log-probs are stored to this many decimals, far below the bf16 spread the
#: gate holds the engine to.
DECIMALS = 5
#: TensorRT-LLM caps ``logprobs`` and ``prompt_logprobs`` at this many.
MAX_TOP_K = 100


class ReferenceError(ValueError):
    """A reference that cannot be read or made; the message names what and where."""


# ----------------------------------------------------------------------- the file


@dataclass(frozen=True)
class Agreement:
    """How a second prediction of the same positions agrees with the first.

    ``disagreements`` counts the positions whose top-1 token differs (a tie
    with the first's top-1 counts as agreeing); the deltas are |delta log-prob|
    of the first prediction's top-1 token. ``worst_prompt`` and
    ``worst_position`` name where the largest delta is.
    """

    positions: int = 0
    disagreements: int = 0
    mean_abs_delta: float = 0.0
    max_abs_delta: float = 0.0
    worst_prompt: str = ""
    worst_position: int = -1

    @property
    def top1_agreement(self) -> float:
        return 1.0 - self.disagreements / self.positions if self.positions else 1.0

    @classmethod
    def of(cls, prompt: str, rows: Iterable[tuple[int, bool, float]]) -> "Agreement":
        """From one prompt's ``(position, agreed, |delta|)`` rows."""
        positions = disagreements = 0
        total = worst = 0.0
        at = -1
        for position, agreed, delta in rows:
            positions += 1
            disagreements += 0 if agreed else 1
            total += delta
            if at < 0 or delta > worst:
                worst, at = delta, position
        if not positions:
            return cls()
        return cls(positions, disagreements, total / positions, worst, prompt, at)

    @classmethod
    def pooled(cls, parts: Iterable["Agreement"]) -> "Agreement":
        """Every part's positions together; the worst is the worst part's."""
        counted = [p for p in parts if p.positions]
        positions = sum(p.positions for p in counted)
        if not positions:
            return cls()
        worst = max(counted, key=lambda p: p.max_abs_delta)
        return cls(
            positions=positions,
            disagreements=sum(p.disagreements for p in counted),
            mean_abs_delta=sum(p.mean_abs_delta * p.positions for p in counted) / positions,
            max_abs_delta=worst.max_abs_delta,
            worst_prompt=worst.worst_prompt,
            worst_position=worst.worst_position,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "positions": self.positions,
            "disagreements": self.disagreements,
            "mean_abs_delta": round(self.mean_abs_delta, DECIMALS + 2),
            "max_abs_delta": round(self.max_abs_delta, DECIMALS + 2),
            "worst_prompt": self.worst_prompt,
            "worst_position": self.worst_position,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Agreement":
        return cls(
            positions=int(data.get("positions", 0)),
            disagreements=int(data.get("disagreements", 0)),
            mean_abs_delta=float(data.get("mean_abs_delta", 0.0)),
            max_abs_delta=float(data.get("max_abs_delta", 0.0)),
            worst_prompt=str(data.get("worst_prompt", "")),
            worst_position=int(data.get("worst_position", -1)),
        )


@dataclass(frozen=True)
class ImageRef:
    """An image a prompt sends: its path as given, and the SHA-256 of its bytes."""

    path: str
    sha256: str


def file_digest(path: str | Path) -> str:
    """The SHA-256 of the file at ``path``."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ReferencePrompt:
    """One prompt, and the reference's prediction at each of its positions.

    Positions run over the prompt and its continuation together: position
    ``p`` is the prediction, after ``sequence[: p + 1]``, of
    ``sequence[p + 1]``. The first :attr:`prompt_positions` are the prompt's
    own (teacher-forced); the rest are the continuation's steps, the last
    prompt token predicting the first continuation token. ``top_ids`` and
    ``top_logprobs`` hold the top-k per position, best first;
    ``target_logprobs`` the log-prob of ``sequence[p + 1]``.
    """

    id: str
    kind: str
    text: str
    input_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]
    top_ids: tuple[tuple[int, ...], ...]
    top_logprobs: tuple[tuple[float, ...], ...]
    target_logprobs: tuple[float, ...]
    images: tuple[ImageRef, ...] = ()
    spread: Agreement = field(default_factory=Agreement)

    @property
    def sequence(self) -> tuple[int, ...]:
        return self.input_ids + self.continuation_ids

    @property
    def prompt_positions(self) -> int:
        return max(len(self.input_ids) - 1, 0)

    @property
    def positions(self) -> int:
        return self.prompt_positions + len(self.continuation_ids)

    def step(self, position: int) -> int | None:
        """The continuation step ``position`` is; ``None`` for a prompt position."""
        offset = position - self.prompt_positions
        return offset if offset >= 0 else None

    def scored(self, media: Collection[int]) -> list[int]:
        """The positions a comparison reads: every one whose target is not a media placeholder.

        No model is trained to predict an image's placeholders, so what
        either side computes there says nothing; the vision tower's output
        reaches every text position after the image, where it is read.
        """
        sequence = self.sequence
        return [p for p in range(self.positions) if sequence[p + 1] not in media]


@dataclass(frozen=True)
class Reference:
    """A reference file, read."""

    path: str
    model: str
    dtype: str
    spread_dtype: str
    top_k: int
    continuation: int
    media_token_ids: frozenset[int]
    #: Every prompt's spread pooled.
    spread: Agreement
    prompts: tuple[ReferencePrompt, ...]
    created: str = ""

    def prompt(self, prompt_id: str) -> ReferencePrompt | None:
        return next((p for p in self.prompts if p.id == prompt_id), None)


def read_json(path: str | Path) -> Any:
    """The JSON document at ``path``, gunzipped first when it is compressed."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def write_reference(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``payload`` to ``path`` atomically, gzip-compressed when the name ends in ``.gz``."""
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    target = Path(path)
    if target.name.endswith(".gz"):
        data = gzip.compress(data, mtime=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.partial")
    staged.write_bytes(data)
    os.replace(staged, target)
    return target


def _prompt_from(data: Mapping[str, Any], top_k: int, continuation: int, where: str):
    prompt = ReferencePrompt(
        id=str(data["id"]),
        kind=str(data.get("kind", KIND_TEXT)),
        text=str(data.get("text", "")),
        input_ids=tuple(int(t) for t in data["input_ids"]),
        continuation_ids=tuple(int(t) for t in data["continuation_ids"]),
        top_ids=tuple(tuple(int(t) for t in row) for row in data["top_ids"]),
        top_logprobs=tuple(tuple(float(v) for v in row) for row in data["top_logprobs"]),
        target_logprobs=tuple(float(v) for v in data["target_logprobs"]),
        images=tuple(
            ImageRef(path=str(i["path"]), sha256=str(i.get("sha256", "")))
            for i in data.get("images") or ()
        ),
        spread=Agreement.from_dict(data.get("spread") or {}),
    )
    if not prompt.input_ids:
        raise ReferenceError(f"{where}: prompt {prompt.id!r} has no input tokens")
    if len(prompt.continuation_ids) != continuation:
        raise ReferenceError(
            f"{where}: prompt {prompt.id!r} has {len(prompt.continuation_ids)} continuation "
            f"tokens, the file says {continuation}"
        )
    for name in ("top_ids", "top_logprobs", "target_logprobs"):
        if len(getattr(prompt, name)) != prompt.positions:
            raise ReferenceError(
                f"{where}: prompt {prompt.id!r} has {len(getattr(prompt, name))} {name} rows "
                f"for {prompt.positions} positions"
            )
    if any(len(row) != top_k for row in (*prompt.top_ids, *prompt.top_logprobs)):
        raise ReferenceError(f"{where}: prompt {prompt.id!r} has a row that is not top-{top_k}")
    return prompt


def load_reference(path: str | Path) -> Reference:
    """Read and check the reference file at ``path``.

    Raises :class:`ReferenceError` naming the problem: an unreadable file,
    another format, a newer version, rows that do not cover the positions.
    """
    where = str(path)
    try:
        data = read_json(path)
    except OSError as exc:
        raise ReferenceError(f"cannot read the logit reference {where}: {exc}") from exc
    except ValueError as exc:
        raise ReferenceError(f"the logit reference {where} is not JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ReferenceError(f"{where} is not a logit reference (format {FORMAT!r})")
    version = data.get("version")
    if not isinstance(version, int) or version < 1 or version > VERSION:
        raise ReferenceError(
            f"{where} is version {version!r} of the logit reference; this flow reads 1 to {VERSION}"
        )
    try:
        top_k = int(data["top_k"])
        continuation = int(data["continuation"])
        prompts = tuple(
            _prompt_from(p, top_k, continuation, where) for p in data.get("prompts") or ()
        )
        reference = Reference(
            path=where,
            model=str(data.get("model", "")),
            dtype=str(data.get("dtype", "")),
            spread_dtype=str(data.get("spread_dtype", "")),
            top_k=top_k,
            continuation=continuation,
            media_token_ids=frozenset(int(t) for t in data.get("media_token_ids") or ()),
            spread=Agreement.from_dict(data.get("spread") or {}),
            prompts=prompts,
            created=str(data.get("created", "")),
        )
    except ReferenceError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise ReferenceError(f"{where} is missing or misstates a field: {exc!r}") from exc
    if not prompts:
        raise ReferenceError(f"{where} holds no prompts")
    if len({p.id for p in prompts}) != len(prompts):
        raise ReferenceError(f"{where} names a prompt id twice")
    return reference


# ------------------------------------------------------------------ the measurement
#
# What follows makes a reference. The bookkeeping (which position predicts
# which token, the greedy continuation, the spread) is plain Python over a
# `Predictor`; the predictor that runs the checkpoint (`HFPredictor`) imports
# torch and transformers when it is built.


@dataclass(frozen=True)
class PromptSpec:
    """One prompt to measure: raw text, images in it, or a long prompt cut to a token count."""

    id: str
    text: str = ""
    images: tuple[str, ...] = ()
    long_tokens: int = 0
    text_file: str = ""

    @property
    def kind(self) -> str:
        if self.images:
            return KIND_IMAGE
        return KIND_LONG if self.long_tokens else KIND_TEXT


@dataclass(frozen=True)
class Row:
    """One position's prediction, reduced: the top-k, the target's log-prob and a probe's."""

    top_ids: tuple[int, ...]
    top_logprobs: tuple[float, ...]
    target_logprob: float | None = None
    probe_logprob: float | None = None


class Predictor(Protocol):
    """A model that predicts every position of a sequence."""

    media_token_ids: frozenset[int]

    def encode(self, prompt: PromptSpec) -> list[int]:
        """The prompt's input ids, images expanded to their placeholders."""

    def rows(
        self,
        prompt: PromptSpec,
        sequence: Sequence[int],
        *,
        last_only: bool,
        top_k: int,
        targets: Sequence[int | None],
        probes: Sequence[int | None] = (),
    ) -> list[Row]:
        """One row per position of ``sequence`` (the prompt's ids, then any appended), or the last.

        ``targets[i]`` and ``probes[i]`` name a token whose log-prob row ``i``
        also reports; ``None`` for none.
        """

    def decode(self, ids: Sequence[int]) -> str:
        """``ids`` as text."""


@dataclass
class Measured:
    """One prompt, measured: its tokens, a row per position, and, once measured, its spread."""

    spec: PromptSpec
    input_ids: list[int]
    continuation_ids: list[int]
    rows: list[Row]
    images: tuple[ImageRef, ...] = ()
    spread: Agreement = field(default_factory=Agreement)


def measure_prompt(
    predictor: Predictor,
    prompt: PromptSpec,
    *,
    continuation: int,
    top_k: int,
    images: tuple[ImageRef, ...] = (),
) -> Measured:
    """The prompt's rows teacher-forced, then ``continuation`` greedy steps.

    Step 0 is the last prompt row; step ``j`` is the last row of a forward
    over the prompt and steps ``0..j-1``. A step's target is its own top-1.
    """
    ids = predictor.encode(prompt)
    if not ids:
        raise ReferenceError(f"prompt {prompt.id!r} encodes to no tokens")
    rows = predictor.rows(prompt, ids, last_only=False, top_k=top_k, targets=[*ids[1:], None])
    if len(rows) != len(ids):
        raise ReferenceError(f"prompt {prompt.id!r}: {len(rows)} rows for {len(ids)} tokens")
    generated: list[int] = []
    steps: list[Row] = []
    last = rows[-1]
    for step in range(continuation):
        if step:
            last = predictor.rows(
                prompt, [*ids, *generated], last_only=True, top_k=top_k, targets=[None]
            )[-1]
        generated.append(last.top_ids[0])
        steps.append(replace(last, target_logprob=last.top_logprobs[0]))
    return Measured(prompt, list(ids), generated, [*rows[:-1], *steps], images)


def agrees(row: Row, probe: int) -> bool:
    """Whether ``probe`` is ``row``'s top-1 token, or tied with it."""
    if row.top_ids[0] == probe:
        return True
    return row.probe_logprob is not None and row.probe_logprob >= row.top_logprobs[0]


def measure_spread(predictor: Predictor, measured: Measured, media: Collection[int]) -> Agreement:
    """How a second precision of the model moves off ``measured``, at every scored position.

    One teacher-forced forward over the prompt and its continuation; at each
    position, the log-prob of ``measured``'s top-1 token and whether it is
    still the top-1.
    """
    if predictor.encode(measured.spec) != measured.input_ids:
        raise ReferenceError(
            f"prompt {measured.spec.id!r} encodes to other tokens in the second precision"
        )
    sequence = [*measured.input_ids, *measured.continuation_ids]
    probes = [row.top_ids[0] for row in measured.rows]
    rows = predictor.rows(
        measured.spec,
        sequence,
        last_only=False,
        top_k=1,
        targets=[None] * len(sequence),
        probes=[*probes, None],
    )
    scored = []
    for position, first in enumerate(measured.rows):
        if sequence[position + 1] in media:
            continue
        second = rows[position]
        delta = abs(first.top_logprobs[0] - float(second.probe_logprob))
        scored.append((position, agrees(second, probes[position]), delta))
    return Agreement.of(measured.spec.id, scored)


def _rounded(values: Iterable[float]) -> list[float]:
    return [round(float(v), DECIMALS) for v in values]


def build_payload(
    measured: Sequence[Measured],
    *,
    model: str,
    dtype: str,
    spread_dtype: str,
    top_k: int,
    continuation: int,
    media_token_ids: Collection[int],
    continuation_texts: Mapping[str, str] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The reference file's document; ``continuation_texts`` (by prompt id) are for a reader."""
    texts = continuation_texts or {}
    prompts = []
    for m in measured:
        entry: dict[str, Any] = {
            "id": m.spec.id,
            "kind": m.spec.kind,
            "text": m.spec.text,
            "input_ids": list(m.input_ids),
            "continuation_ids": list(m.continuation_ids),
            "top_ids": [list(r.top_ids) for r in m.rows],
            "top_logprobs": [_rounded(r.top_logprobs) for r in m.rows],
            "target_logprobs": _rounded(float(r.target_logprob) for r in m.rows),
            "spread": m.spread.as_dict(),
        }
        if m.images:
            entry["images"] = [{"path": i.path, "sha256": i.sha256} for i in m.images]
        if m.spec.id in texts:
            entry["continuation_text"] = texts[m.spec.id]
        prompts.append(entry)
    return {
        "format": FORMAT,
        "version": VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "dtype": dtype,
        "spread_dtype": spread_dtype,
        "top_k": top_k,
        "continuation": continuation,
        "media_token_ids": sorted(int(t) for t in media_token_ids),
        "spread": Agreement.pooled(m.spread for m in measured).as_dict(),
        **({"meta": dict(meta)} if meta else {}),
        "prompts": prompts,
    }


# ---------------------------------------------------------------------- prompts

DEFAULT_PROMPTS: tuple[PromptSpec, ...] = (
    PromptSpec("capital", "The capital of France is"),
    PromptSpec(
        "clips",
        "Question: Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
        "Answer:",
    ),
    PromptSpec("photosynthesis", "In one sentence, photosynthesis is"),
)
#: One context chunk past TensorRT-LLM's default ``max_num_tokens`` (8192).
DEFAULT_LONG_TOKENS = 8704
DEFAULT_CONTINUATION = 32
DEFAULT_TOP_K = 5
DEFAULT_IMAGE_PLACEHOLDER = "<image>"
DEFAULT_IMAGE_TEXT = "<image>\nThis is a photo of"

#: The standard-library modules whose documentation a default long prompt is cut from.
_DOC_MODULES: tuple[str, ...] = (
    "argparse",
    "collections",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "difflib",
    "functools",
    "heapq",
    "inspect",
    "itertools",
    "json",
    "logging",
    "pathlib",
    "pickle",
    "random",
    "re",
    "shutil",
    "statistics",
    "string",
    "subprocess",
    "tarfile",
    "tempfile",
    "textwrap",
    "threading",
    "typing",
    "unittest",
    "zipfile",
)


def standard_library_text() -> str:
    """Prose to cut a long prompt from: standard-library documentation, in a fixed order.

    Each module's docstring, then those of the functions and classes it
    exports, each once. The text is stored in the reference as token ids, so
    another Python version changes only references made with it.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for name in _DOC_MODULES:
        module = importlib.import_module(name)
        docs = [module.__doc__]
        for attr in sorted(getattr(module, "__all__", ())):
            member = getattr(module, attr, None)
            owner = str(getattr(member, "__module__", ""))
            if (inspect.isfunction(member) or inspect.isclass(member)) and owner.startswith(name):
                docs.append(member.__doc__)
        for doc in docs:
            if isinstance(doc, str) and doc.strip():
                text = inspect.cleandoc(doc)
                if text not in seen:
                    seen.add(text)
                    parts.append(text)
    return "\n\n".join(parts)


def read_prompts(path: str | Path) -> list[PromptSpec]:
    """A prompts file: a YAML or JSON list of ``{id, text, images?}`` or ``{id, long_tokens, text_file?}``."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("prompts")
    if not isinstance(data, list) or not data:
        raise ReferenceError(f"{source} must hold a non-empty list of prompts")
    prompts = []
    for n, entry in enumerate(data):
        if not isinstance(entry, dict) or not str(entry.get("id") or "").strip():
            raise ReferenceError(f"{source}: prompt {n} must be a mapping with an `id`")
        unknown = set(entry) - {"id", "text", "images", "long_tokens", "text_file"}
        if unknown:
            raise ReferenceError(
                f"{source}: prompt {entry['id']!r} has unknown keys {sorted(unknown)}"
            )
        spec = PromptSpec(
            id=str(entry["id"]),
            text=str(entry.get("text") or ""),
            images=tuple(str(i) for i in entry.get("images") or ()),
            long_tokens=int(entry.get("long_tokens") or 0),
            text_file=str(entry.get("text_file") or ""),
        )
        if spec.long_tokens < 0 or (spec.long_tokens and spec.images):
            raise ReferenceError(f"{source}: prompt {spec.id!r} is long or has images, not both")
        if not spec.long_tokens and not spec.text.strip():
            raise ReferenceError(f"{source}: prompt {spec.id!r} has no text")
        prompts.append(spec)
    return prompts


def resolve_prompts(args: argparse.Namespace) -> list[PromptSpec]:
    """The prompts to measure: the ``--prompts`` file, else the defaults and ``--image`` prompts."""
    if args.prompts:
        prompts = read_prompts(args.prompts)
    else:
        prompts = list(DEFAULT_PROMPTS)
        if args.long_tokens:
            prompts.append(PromptSpec("long", long_tokens=args.long_tokens))
        for n, image in enumerate(args.image or ()):
            prompts.append(PromptSpec(f"image{n}", args.image_text, images=(image,)))
    ids = [p.id for p in prompts]
    if len(set(ids)) != len(ids):
        raise ReferenceError(f"prompt ids must be distinct: {ids}")
    return prompts


def image_location(path: str, root: str | Path) -> Path:
    """Where to open an image path as given: as is when absolute, else under ``root``."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(root) / candidate


# ---------------------------------------------------------- the Hugging Face model

#: Configuration fields that name a media placeholder's token id.
_MEDIA_ID_FIELDS: tuple[str, ...] = (
    "image_token_id",
    "image_token_index",
    "video_token_id",
    "video_token_index",
    "audio_token_id",
    "audio_token_index",
)
#: Rows of logits reduced at once: log-softmax over a 262k vocabulary in fp32
#: is 1 MiB a row.
_ROW_CHUNK = 512


def media_token_ids(config: Any, tokenizer: Any, processor: Any) -> set[int]:
    """The placeholder ids a configuration and its processor name."""
    found: set[int] = set()
    for owner in (config, *(getattr(config, s, None) for s in ("text_config", "vision_config"))):
        for name in _MEDIA_ID_FIELDS:
            value = getattr(owner, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                found.add(value)
    token = getattr(processor, "image_token", None) if processor is not None else None
    if isinstance(token, str) and token:
        value = tokenizer.convert_tokens_to_ids(token)
        if isinstance(value, int) and value != getattr(tokenizer, "unk_token_id", None):
            found.add(value)
    return found


def _device_map(value: str) -> Any:
    return value if value in ("auto", "balanced", "balanced_low_0", "sequential") else {"": value}


def _dtype_keyword(transformers: Any) -> str:
    """``from_pretrained``'s name for the weights' dtype: ``dtype`` since 4.56."""
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


class HFPredictor:
    """The checkpoint under ``transformers``, one sequence at a time, without a cache."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        processor: Any,
        *,
        image_placeholder: str,
        image_root: str | Path,
        extra_media: Collection[int] = (),
    ) -> None:
        import torch

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_placeholder = image_placeholder
        self.image_root = Path(image_root)
        self.media_token_ids = frozenset(
            media_token_ids(model.config, tokenizer, processor) | set(extra_media)
        )
        self.dtype = model.get_input_embeddings().weight.dtype
        self.device = model.device
        self._keeps_logits = "logits_to_keep" in inspect.signature(model.forward).parameters
        self._encoded: dict[str, dict[str, Any]] = {}
        self._torch = torch

    @classmethod
    def load(
        cls,
        path: str,
        *,
        dtype: str,
        device_map: str,
        trust_remote_code: bool,
        with_images: bool,
        image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER,
        image_root: str | Path = ".",
        extra_media: Collection[int] = (),
    ) -> "HFPredictor":
        """Load the checkpoint in ``dtype``: a causal LM, or an image-text model for image prompts.

        Whichever class the checkpoint does not register is tried second.
        """
        import torch
        import transformers

        wanted = getattr(torch, dtype)
        kwargs = {
            _dtype_keyword(transformers): wanted,
            "device_map": _device_map(device_map),
            "trust_remote_code": trust_remote_code,
        }
        names = ["AutoModelForCausalLM", "AutoModelForImageTextToText"]
        if with_images:
            names.reverse()
        refused: list[str] = []
        model = None
        for name in names:
            auto = getattr(transformers, name, None)
            if auto is None:
                continue
            try:
                model = auto.from_pretrained(path, **kwargs)
                break
            except ValueError as exc:
                refused.append(f"{name}: {exc}")
        if model is None:
            raise ReferenceError(f"no auto class loads {path}: {'; '.join(refused)}")
        model.eval()
        loaded = model.get_input_embeddings().weight.dtype
        if loaded != wanted:
            raise ReferenceError(f"asked for {dtype} weights, and they loaded as {loaded}")
        if with_images:
            processor = transformers.AutoProcessor.from_pretrained(
                path, trust_remote_code=trust_remote_code
            )
            tokenizer = processor.tokenizer
        else:
            processor = None
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                path, trust_remote_code=trust_remote_code
            )
        return cls(
            model,
            tokenizer,
            processor,
            image_placeholder=image_placeholder,
            image_root=image_root,
            extra_media=extra_media,
        )

    @property
    def versions(self) -> dict[str, str]:
        import transformers

        return {"torch": self._torch.__version__, "transformers": transformers.__version__}

    def close(self) -> None:
        """Release the model and its device memory."""
        self.model = None
        self._encoded.clear()
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def _long_ids(self, prompt: PromptSpec) -> list[int]:
        if prompt.text_file:
            text = Path(prompt.text_file).read_text(encoding="utf-8")
        else:
            text = prompt.text or standard_library_text()
        ids = list(self.tokenizer(text)["input_ids"])
        if len(ids) < prompt.long_tokens:
            raise ReferenceError(
                f"prompt {prompt.id!r}: the text has {len(ids)} tokens, {prompt.long_tokens} asked"
            )
        return ids[: prompt.long_tokens]

    def encode(self, prompt: PromptSpec) -> list[int]:
        torch = self._torch
        if prompt.images:
            from PIL import Image

            if self.processor is None:
                raise ReferenceError(f"prompt {prompt.id!r} has images and no processor is loaded")
            images = [
                Image.open(image_location(p, self.image_root)).convert("RGB") for p in prompt.images
            ]
            text = prompt.text
            token = getattr(self.processor, "image_token", None)
            if isinstance(token, str) and token and self.image_placeholder:
                text = text.replace(self.image_placeholder, token)
            encoded = self.processor(text=[text], images=images, return_tensors="pt")
            inputs = {}
            for name, value in encoded.items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.to(self.dtype)
                inputs[name] = value
            self._encoded[prompt.id] = inputs
            return [int(t) for t in inputs["input_ids"][0].tolist()]
        if prompt.long_tokens:
            return self._long_ids(prompt)
        return [int(t) for t in self.tokenizer(prompt.text)["input_ids"]]

    def _inputs(self, prompt: PromptSpec, sequence: Sequence[int]) -> dict[str, Any]:
        """The forward's keyword arguments for ``sequence``.

        An image prompt's processor outputs, with the tokens past the prompt
        appended: to the ids, to the attention mask (as ones) and to any other
        per-token integer tensor (as zeros, a text token's type).
        """
        torch = self._torch
        ids = torch.tensor([list(sequence)], dtype=torch.long, device=self.device)
        base = self._encoded.get(prompt.id)
        if base is None:
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        length = base["input_ids"].shape[1]
        if list(sequence[:length]) != base["input_ids"][0].tolist():
            raise ReferenceError(
                f"prompt {prompt.id!r}: the sequence does not start with its prompt"
            )
        added = len(sequence) - length
        inputs = dict(base, input_ids=ids)
        for name, value in base.items():
            if name == "input_ids" or not isinstance(value, torch.Tensor):
                continue
            if value.dim() == 2 and value.shape == (1, length) and not value.is_floating_point():
                fill = 1 if name == "attention_mask" else 0
                tail = torch.full((1, added), fill, dtype=value.dtype, device=value.device)
                inputs[name] = torch.cat([value, tail], dim=1)
        return inputs

    def rows(
        self,
        prompt: PromptSpec,
        sequence: Sequence[int],
        *,
        last_only: bool,
        top_k: int,
        targets: Sequence[int | None],
        probes: Sequence[int | None] = (),
    ) -> list[Row]:
        torch = self._torch
        kwargs = {**self._inputs(prompt, sequence), "use_cache": False}
        if last_only and self._keeps_logits:
            kwargs["logits_to_keep"] = 1
        with torch.inference_mode():
            logits = self.model(**kwargs).logits[0]
        if last_only:
            logits = logits[-1:]
        rows: list[Row] = []
        for start in range(0, logits.shape[0], _ROW_CHUNK):
            with torch.inference_mode():
                logprobs = torch.log_softmax(logits[start : start + _ROW_CHUNK].float(), dim=-1)
                values, indices = logprobs.topk(top_k, dim=-1)
                count = logprobs.shape[0]
                chosen = self._gather(logprobs, targets[start : start + count])
                probed = self._gather(logprobs, probes[start : start + count])
            for offset, (top_values, top_indices) in enumerate(
                zip(values.tolist(), indices.tolist())
            ):
                rows.append(
                    Row(
                        top_ids=tuple(int(t) for t in top_indices),
                        top_logprobs=tuple(float(v) for v in top_values),
                        target_logprob=chosen[offset],
                        probe_logprob=probed[offset],
                    )
                )
        return rows

    def _gather(self, logprobs: Any, tokens: Sequence[int | None]) -> list[float | None]:
        """``logprobs[i, tokens[i]]`` per row of ``logprobs`` [rows, vocab]; None where none is named."""
        torch = self._torch
        count = logprobs.shape[0]
        named = [i for i in range(min(count, len(tokens))) if tokens[i] is not None]
        out: list[float | None] = [None] * count
        if named:
            rows = torch.tensor(named, device=logprobs.device)
            columns = torch.tensor([int(tokens[i]) for i in named], device=logprobs.device)
            for i, value in zip(named, logprobs[rows, columns].tolist()):
                out[i] = float(value)
        return out


# ------------------------------------------------------------------------- main


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="logit_reference.py",
        description="Record the HF reference's per-position top-k log-probs for the logit gate.",
    )
    parser.add_argument("--model", required=True, help="the checkpoint (a directory or a hub id)")
    parser.add_argument("--out", required=True, type=Path, help="the reference file (.json[.gz])")
    parser.add_argument("--prompts", type=Path, help="a prompts file; replaces the defaults")
    parser.add_argument(
        "--long-tokens",
        type=int,
        default=DEFAULT_LONG_TOKENS,
        help=f"the default long prompt's length in tokens, 0 for none "
        f"(default {DEFAULT_LONG_TOKENS})",
    )
    parser.add_argument(
        "--image", action="append", help="add an image prompt sending this image (repeatable)"
    )
    parser.add_argument("--image-text", default=DEFAULT_IMAGE_TEXT, help="an --image prompt's text")
    parser.add_argument(
        "--image-root", default=".", help="where a relative image path is opened (the tree)"
    )
    parser.add_argument(
        "--image-placeholder",
        default=DEFAULT_IMAGE_PLACEHOLDER,
        help="what marks an image in a prompt's text; the processor's image token replaces it here",
    )
    parser.add_argument("--continuation", type=int, default=DEFAULT_CONTINUATION)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16"), help="the reference's dtype"
    )
    parser.add_argument(
        "--spread-dtype",
        default="float32",
        choices=("float32", "float16"),
        help="the precision the spread is measured against (default float32)",
    )
    parser.add_argument("--device-map", default="auto", help="auto, or one device such as cuda:0")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run the checkpoint's own code (default: when it ships one, an auto_map)",
    )
    parser.add_argument(
        "--media-token-id",
        type=int,
        action="append",
        default=[],
        help="a placeholder id the configuration does not name (repeatable)",
    )
    args = parser.parse_args(list(argv))
    if not 1 <= args.top_k <= MAX_TOP_K:
        parser.error(f"--top-k must be 1 to {MAX_TOP_K}")
    if args.continuation < 0 or args.long_tokens < 0:
        parser.error("--continuation and --long-tokens must be >= 0")
    if args.dtype == args.spread_dtype:
        parser.error("--spread-dtype must differ from --dtype")
    return args


def ships_code(model: str | Path) -> bool:
    """Whether the checkpoint ships its own code: an ``auto_map`` in one of its configs.

    The rule ``checks/gates.py:checkpoint_remote_code`` applies to the gates;
    stated again here because this script runs by path in the reference's own
    environment, which does not import the workflow.
    """
    for name in (
        "config.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
    ):
        try:
            data = json.loads((Path(model) / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("auto_map"):
            return True
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Measure every prompt in ``--dtype``, its spread in ``--spread-dtype``, and write the file."""
    args = _parse(sys.argv[1:] if argv is None else argv)
    if args.trust_remote_code is None:
        args.trust_remote_code = ships_code(args.model)
    prompts = resolve_prompts(args)
    with_images = any(p.images for p in prompts)
    common = dict(
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        with_images=with_images,
        image_placeholder=args.image_placeholder,
        image_root=args.image_root,
        extra_media=args.media_token_id,
    )

    started = time.monotonic()
    primary = HFPredictor.load(args.model, dtype=args.dtype, **common)
    media = primary.media_token_ids
    versions = primary.versions
    measured = []
    for prompt in prompts:
        images = tuple(
            ImageRef(path=p, sha256=file_digest(image_location(p, args.image_root)))
            for p in prompt.images
        )
        measured.append(
            measure_prompt(
                primary, prompt, continuation=args.continuation, top_k=args.top_k, images=images
            )
        )
        print(f"[{args.dtype}] {prompt.id}: {len(measured[-1].rows)} positions", flush=True)
    decoded = {m.spec.id: primary.decode(m.continuation_ids) for m in measured}
    primary.close()

    secondary = HFPredictor.load(args.model, dtype=args.spread_dtype, **common)
    for m in measured:
        m.spread = measure_spread(secondary, m, media)
        print(
            f"[spread vs {args.spread_dtype}] {m.spec.id}: top-1 agreement "
            f"{m.spread.top1_agreement:.4f}, mean |delta| {m.spread.mean_abs_delta:.4f}, "
            f"max {m.spread.max_abs_delta:.4f}",
            flush=True,
        )
    secondary.close()

    payload = build_payload(
        measured,
        model=args.model,
        dtype=args.dtype,
        spread_dtype=args.spread_dtype,
        top_k=args.top_k,
        continuation=args.continuation,
        media_token_ids=media,
        continuation_texts=decoded,
        meta={**versions, "seconds": round(time.monotonic() - started)},
    )
    write_reference(payload, args.out)
    pooled = Agreement.from_dict(payload["spread"])
    print(
        f"wrote {args.out}: {len(measured)} prompts, {pooled.positions} scored positions; spread "
        f"top-1 agreement {pooled.top1_agreement:.4f}, mean |delta| {pooled.mean_abs_delta:.4f}, "
        f"max {pooled.max_abs_delta:.4f} ({pooled.worst_prompt}@{pooled.worst_position})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
