# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold-start generation and boot workload coverage."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.commands import ended_early
from ..infra.compute.environment import RunEnvironment
from ..spec.deployments import Deployment
from ..spec.task_schema import BOOT_KEYS, GATE_TIMEOUT_KEY, load_prompts_file, resolve_gate_timeout
from . import attest
from .check_config import check_configuration
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    ShellRunner,
    default_shell_runner,
    engine_driver_command,
    gate_workdir,
    parse_driver_records,
    with_mode,
    write_driver_spec,
)
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

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


def run_boot_gate(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    deployment: Deployment,
    workdir: str | Path,
    spec: BootSpec | None = None,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
    mode: str = MODE_REQUIRE,
) -> GateResult:
    """Cold-start ``deployment``'s engine in process and check it generates, through the engine driver.

    ``spec`` is the workload (:func:`resolve_boot_spec`). Every rank's engine is held to
    the deployment's options (``checks/attest.py``); with block reuse on, one more prompt
    is sent again and must come back with cached tokens. ``mode`` other than ``require``
    is a control run on the stock implementation, recorded as ``boot[<mode>]``.
    """
    spec = spec or BootSpec()
    env = with_mode(environment, mode)
    gate_name = "boot" if mode == MODE_REQUIRE else f"boot[{mode}]"
    _verdict = verdict_factory(inspect_tree(task_data))
    config = check_configuration(task_data, deployment, max_tokens=spec.max_tokens)
    requests = spec.requests(task_data.get("trtllm_repo_path"))
    declared_requests = len(requests)
    options = (deployment.server or deployment.ctx).options
    repeats = attest.reuse_sends(options) if attest.reuses(options) else 0
    requests += [
        {"id": f"{attest.REUSE_ID}{n}", "prompt": attest.REUSE_PROMPT, "max_tokens": 1}
        for n in range(repeats)
    ]
    if timeout is None:
        declared = resolve_gate_timeout(task_data.get("accuracy") or {}, "boot", deployment.name)
        timeout = declared or BOOT_TIMEOUT_S
    spec_path = write_driver_spec(
        gate_workdir(workdir) / f"boot.{deployment.name}.json",
        config=config,
        requests=requests,
        max_tokens=spec.max_tokens,
    )
    command = engine_driver_command(env, spec_path)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(
        name=gate_name,
        deployment=deployment.name,
        command=command,
        duration_s=elapsed,
        stdout_tail=tail,
    )

    if result.infrastructure:
        return _verdict(**lost_session(result), **ran)
    if result.timed_out and result.stalled:
        return _verdict(passed=False, detail=ended_early(result, timeout), **ran)
    if result.timed_out:
        return _verdict(
            passed=False,
            detail=f"timed out after {timeout}s — a boot gate that takes this long is "
            f"not catching catastrophes any more",
            **ran,
        )
    if not result.ok:
        return _verdict(passed=False, detail=f"exited {result.returncode}", **ran)
    attested = attest.attest(
        combined,
        {**options, "modeling_v2": mode},
        tree=task_data.get("trtllm_repo_path"),
        target=mode == MODE_REQUIRE,
    )
    ran["engine"] = (attested.record(),)
    failure = attested.failure()
    if failure:
        return _verdict(passed=False, detail=failure, **ran)

    records = {int(r.get("index", -1)): r for r in parse_driver_records(combined)}
    texts = [
        str(records[n].get("text", "")) if n in records else None for n in range(len(requests))
    ]
    absent = [requests[n]["id"] for n, text in enumerate(texts) if text is None]
    if absent:
        return _verdict(
            passed=False,
            detail=f"the engine exited cleanly but returned no record for {len(absent)} of "
            f"{len(requests)} prompt(s) ({', '.join(absent)})",
            **ran,
        )
    texts = texts[:declared_requests]
    empty = [requests[n]["id"] for n, text in enumerate(texts) if not (text or "").strip()]
    if empty:
        return _verdict(
            passed=False,
            detail=f"the engine exited cleanly but generated nothing for {', '.join(empty)}",
            **ran,
        )
    reused = ""
    if repeats:
        cached = [records[n].get("cached_tokens") for n in range(declared_requests, len(requests))]
        shortfall = attest.reuse_shortfall(cached)
        if shortfall:
            return _verdict(passed=False, detail=shortfall, **ran)
        reused = f"; a repeated prompt resumed {max(c or 0 for c in cached)} cached tokens"
    missing = missing_keywords(spec, texts)
    if missing:
        return _verdict(
            passed=False,
            detail=f"the generated texts are missing {len(missing)} expected keyword(s): "
            f"{missing}. Keywords are matched against generated text only",
            **ran,
        )
    images = sum(len(r.get("images") or ()) for r in requests)
    checked = (
        f"{len(spec.keywords) + sum(1 for i in spec.items if i.keywords or i.keywords_any)} "
        f"keyword check(s) in the generated texts"
        if spec.keywords or any(i.keywords or i.keywords_any for i in spec.items)
        else "non-empty generation"
    )
    sent = f", {images} image(s) sent" if images else ""
    return _verdict(
        passed=True,
        detail=f"engine cold-started and produced {checked} for {len(texts)} prompt(s)"
        f"{sent}; workload: {spec.source}{reused}",
        **ran,
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


__all__ = [
    "BootPrompt",
    "BootSpec",
    "DEFAULT_BOOT_IMAGE",
    "DEFAULT_BOOT_IMAGE_KEYWORDS",
    "DEFAULT_BOOT_IMAGE_PROMPT",
    "missing_keywords",
    "resolve_boot_spec",
    "run_boot_gate",
]
