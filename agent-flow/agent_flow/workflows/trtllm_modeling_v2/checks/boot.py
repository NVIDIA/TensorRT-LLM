# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The boot prompts: what a test's ``generate`` dataset sends (``accuracy.boot``).

Every answer must be non-empty; what it says is a benchmark's to judge. The
tree's runner sends them (``checks/e2e.py``), and ``trtllm-serve`` does for the
serve tests (``checks/serve.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..spec.task_schema import BOOT_KEYS, load_prompts_file

#: The image a multimodal unit's boot sends when the task declares none: a
#: photograph shipped in the repository's own test assets (a git-lfs file),
#: resolved against the tree under test. Never downloaded.
DEFAULT_BOOT_IMAGE = "tests/integration/test_input_files/merlion.png"


#: What the default image prompt asks of that photograph.
DEFAULT_BOOT_IMAGE_PROMPT = "Describe this image in one sentence."


@dataclass(frozen=True)
class BootPrompt:
    """One boot prompt written as a mapping: the form that carries images."""

    prompt: str
    images: tuple[str, ...] = ()
    render: str = "loader"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BootPrompt":
        """Read one entry of a boot ``prompts`` list written as a mapping."""
        return cls(
            prompt=str(data.get("prompt", "")),
            images=tuple(str(i) for i in data.get("images") or ()),
            render=str(data.get("render") or "loader"),
        )


@dataclass(frozen=True)
class BootSpec:
    """What the boot prompts are.

    ``items`` are the prompts written as mappings, the only ones that carry images.
    """

    prompts: tuple[str, ...] = (
        "The capital of France is",
        "In one sentence, photosynthesis is",
    )
    max_tokens: int = 32
    items: tuple[BootPrompt, ...] = ()
    #: Where the spec came from, named in a verdict.
    source: str = "the default boot prompts"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], source: str) -> "BootSpec":
        """A spec from a boot block: ``prompts`` (strings or mappings) and ``max_tokens``."""
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
            max_tokens=int(data.get("max_tokens") or cls.max_tokens),
            items=tuple(items),
            source=source,
        )

    @property
    def has_images(self) -> bool:
        return any(item.images for item in self.items)

    def requests(self, repo_root: str | Path | None) -> list[dict[str, Any]]:
        """Engine-driver requests: the plain prompts, then the items.

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


def resolve_boot_spec(
    task_data: Mapping[str, Any],
    unit: str | None,
    features: Sequence[str] | None = None,
) -> BootSpec:
    """The boot workload ``unit`` runs, declared in task.yaml.

    ``accuracy.boot`` (run-wide keys, with a per-unit entry layered over
    them) when task.yaml declares it, else the default. A unit whose target
    implements ``multimodal`` (and whose scope is not ``text_only``) and whose
    spec sends no image gets the default image prompt appended.
    """
    accuracy = task_data.get("accuracy") or {}
    block = accuracy.get("boot") if isinstance(accuracy, Mapping) else None
    if isinstance(block, Mapping) and _declares_workload(block):
        merged = {k: v for k, v in block.items() if k in BOOT_KEYS}
        chosen = block.get(unit) if unit is not None else None
        where = "accuracy.boot"
        if isinstance(chosen, Mapping) and _declares_workload(chosen):
            merged.update(chosen)
            where = f"accuracy.boot.{unit}"
        spec = BootSpec.from_mapping(merged, where)
    else:
        spec = BootSpec()
    implemented = {str(f) for f in (features or ())}
    if (
        "multimodal" in implemented
        and task_data.get("scope", "full") != "text_only"
        and not spec.has_images
    ):
        image = BootPrompt(prompt=DEFAULT_BOOT_IMAGE_PROMPT, images=(DEFAULT_BOOT_IMAGE,))
        spec = replace(spec, items=(*spec.items, image))
    return spec


def _declares_workload(block: Mapping[str, Any]) -> bool:
    """Whether a level of ``accuracy.boot`` says what to send, at it or below."""
    return any(
        key in BOOT_KEYS or not isinstance(value, Mapping) or bool(value)
        for key, value in block.items()
    )


def chunk_probe_tokens(max_num_tokens: int, max_seq_len: int | None) -> int | None:
    """The length of a prompt the engine must split: twice ``max_num_tokens``, within ``max_seq_len``.

    ``None`` when the engine's limits leave no room for one.
    """
    want = 2 * int(max_num_tokens) + 1
    if max_seq_len is not None:
        want = min(want, int(max_seq_len) - 2)
    return want if want > max_num_tokens else None


__all__ = [
    "BootPrompt",
    "BootSpec",
    "DEFAULT_BOOT_IMAGE",
    "DEFAULT_BOOT_IMAGE_PROMPT",
    "chunk_probe_tokens",
    "resolve_boot_spec",
]
