# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The boot workload: what a boot sends, and what it must see back.

A boot starts the deployment's servers, sends this workload to the endpoint and holds
the answers to it (``checks/endpoint.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..spec.deployments import Deployment
from ..spec.task_schema import BOOT_KEYS, GATE_TIMEOUT_KEY, load_prompts_file

#: The image a multimodal unit's boot sends when the task declares none: a
#: photograph shipped in the repository's own test assets (a git-lfs file),
#: resolved against the tree under test. Never downloaded.
DEFAULT_BOOT_IMAGE = "tests/integration/test_input_files/merlion.png"


#: What the default image prompt asks, and the words any correct description
#: of that photograph (the Merlion fountain in Singapore) contains one of.
DEFAULT_BOOT_IMAGE_PROMPT = "Describe this image in one sentence."


DEFAULT_BOOT_IMAGE_KEYWORDS: tuple[str, ...] = (
    "merlion",
    "lion",
    "statue",
    "fountain",
    "singapore",
    "sculpture",
    "water",
)


@dataclass(frozen=True)
class BootPrompt:
    """One boot prompt with checks of its own, images included."""

    prompt: str
    #: Every one must appear in THIS prompt's generated text (case-insensitive).
    keywords: tuple[str, ...] = ()
    #: At least one must appear in it (case-insensitive).
    keywords_any: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    render: str = "loader"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BootPrompt":
        """Read one entry of a boot ``prompts`` list written as a mapping."""
        return cls(
            prompt=str(data.get("prompt", "")),
            keywords=tuple(str(k) for k in data.get("keywords") or ()),
            keywords_any=tuple(str(k) for k in data.get("keywords_any") or ()),
            images=tuple(str(i) for i in data.get("images") or ()),
            render=str(data.get("render") or "loader"),
        )

    def missing(self, text: str) -> list[str]:
        """What this prompt's checks did not find in ``text``."""
        lowered = (text or "").lower()
        absent = [k for k in self.keywords if k.lower() not in lowered]
        if self.keywords_any and not any(k.lower() in lowered for k in self.keywords_any):
            absent.append(f"any of {list(self.keywords_any)}")
        return absent


@dataclass(frozen=True)
class BootSpec:
    """What the boot gate sends and what it insists on seeing back.

    The keyword list starts **empty on purpose**: keywords are frozen only
    after observing what this model generates. ``prompts`` and ``keywords``
    are the plain form: each keyword must appear in SOME prompt's generated
    text. ``items`` are prompts with checks of their own, and the only ones
    that carry images. Keywords are matched against the generated texts only,
    never against the log: ``72`` matches ``CPUs 72-143`` in an engine log line.
    """

    prompts: tuple[str, ...] = (
        "The capital of France is",
        "In one sentence, photosynthesis is",
    )
    keywords: tuple[str, ...] = ()
    max_tokens: int = 32
    items: tuple[BootPrompt, ...] = ()
    #: Where the spec came from, named in a verdict.
    source: str = "the default boot prompts"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], source: str) -> "BootSpec":
        """A spec from a boot block: ``prompts`` (strings or mappings), ``keywords``, ``max_tokens``."""
        plain: list[str] = []
        items: list[BootPrompt] = []
        for entry in data.get("prompts") or ():
            if isinstance(entry, Mapping):
                items.append(BootPrompt.from_mapping(entry))
            else:
                plain.append(str(entry))
        if data.get("prompts_file"):
            plain.extend(load_prompts_file(data["prompts_file"]))
        return cls(
            prompts=tuple(plain) if (plain or items) else cls.prompts,
            keywords=tuple(str(k) for k in data.get("keywords") or ()),
            max_tokens=int(data.get("max_tokens") or cls.max_tokens),
            items=tuple(items),
            source=source,
        )

    @property
    def has_images(self) -> bool:
        return any(item.images for item in self.items)

    def requests(self, repo_root: str | Path | None) -> list[dict[str, Any]]:
        """The requests a boot sends: the plain prompts, then the items.

        A relative image path is resolved against ``repo_root``, the tree under test.
        """

        def image_path(name: str) -> str:
            path = Path(name)
            if path.is_absolute() or repo_root is None:
                return str(path)
            return str(Path(repo_root) / path)

        requests = [
            {"id": f"b{n}", "prompt": prompt, "max_tokens": self.max_tokens}
            for n, prompt in enumerate(self.prompts)
        ]
        for n, item in enumerate(self.items):
            request: dict[str, Any] = {
                "id": f"i{n}",
                "prompt": item.prompt,
                "max_tokens": self.max_tokens,
            }
            if item.images:
                request["images"] = [image_path(i) for i in item.images]
                request["render"] = item.render
            requests.append(request)
        return requests


def resolve_boot_spec(task_data: Mapping[str, Any], deployment: Deployment) -> BootSpec:
    """The boot workload ``deployment`` runs, declared in task.yaml.

    ``accuracy.boot`` (run-wide keys, with the deployment's own entry layered over them;
    core's may be written ``core``) when task.yaml declares it, else the default. A
    deployment that takes images and whose workload sends none gets the default image
    prompt appended.
    """
    accuracy = task_data.get("accuracy") or {}
    block = accuracy.get("boot") if isinstance(accuracy, Mapping) else None
    if isinstance(block, Mapping) and _declares_workload(block):
        merged = {k: v for k, v in block.items() if k in BOOT_KEYS}
        where = "accuracy.boot"
        for name in (deployment.name, *(("core",) if deployment.core else ())):
            chosen = block.get(name)
            if isinstance(chosen, Mapping) and _declares_workload(chosen):
                merged.update(chosen)
                where = f"accuracy.boot.{name}"
                break
        spec = BootSpec.from_mapping(merged, where)
    else:
        spec = BootSpec()
    if "image" in deployment.inputs and not spec.has_images:
        image = BootPrompt(
            prompt=DEFAULT_BOOT_IMAGE_PROMPT,
            keywords_any=DEFAULT_BOOT_IMAGE_KEYWORDS,
            images=(DEFAULT_BOOT_IMAGE,),
        )
        spec = replace(spec, items=(*spec.items, image))
    return spec


def _declares_workload(block: Mapping[str, Any]) -> bool:
    """Whether a level of ``accuracy.boot`` says what to send, at it or below (its timeout does not)."""
    return any(
        key in BOOT_KEYS or not isinstance(value, Mapping) or bool(value)
        for key, value in block.items()
        if key != GATE_TIMEOUT_KEY
    )


def missing_keywords(spec: BootSpec, texts: Sequence[str | None]) -> list[str]:
    """The keywords ``spec`` expects that ``texts``, its prompts' generated texts in order, lack.

    The run-wide keywords are matched against every text together; a prompt
    with keywords of its own against its own text. Matched against generated
    text only, case-insensitively. A text that is ``None`` was not asked for,
    and its prompt's own keywords are not looked for.
    """
    generated = "\n".join(text or "" for text in texts).lower()
    missing = [k for k in spec.keywords if k.lower() not in generated]
    item_texts = texts[len(spec.prompts) :]
    for item, text in zip(spec.items, item_texts):
        if text is None:
            continue
        for absent_keyword in item.missing(text):
            label = "the image prompt" if item.images else "a prompt"
            missing.append(f"{absent_keyword} (in {label} {item.prompt[:40]!r})")
    return missing
