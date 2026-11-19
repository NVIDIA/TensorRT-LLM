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
from ..spec.task_schema import BOOT_KEYS, GATE_TIMEOUT_KEY, load_prompts_file, resolve_gate_timeout
from . import check_config
from .check_config import EngineLimits, check_configuration, resolve_llm_api_options
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
from .runtime_evidence import reuse_failure
from .verdict import GateResult, inspect_tree, lost_session, verdict_factory

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
    """What the boot gate sends.

    An answer must not be empty; what it says is the accuracy benchmark's to
    judge. ``items`` are the prompts written as mappings, the only ones that
    carry images.
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
    workdir: str | Path,
    spec: BootSpec | None = None,
    runner: ShellRunner = default_shell_runner,
    timeout: int | None = None,
    mode: str = MODE_REQUIRE,
    features: Any | None = None,
    unit: str | None = None,
    member: str | None = None,
) -> GateResult:
    """Cold-start the engine and check it generates, through the engine driver.

    ``features`` is the features the target under test implements, not those
    the run will finish with (default the run's). ``spec`` is the workload
    (:func:`resolve_boot_spec`). An engine asked for block reuse is held to it
    (:func:`reuse_failure`). ``mode`` other than ``require`` is a control run
    on the stock implementation, recorded as ``boot[<mode>]``.
    """
    spec = spec or BootSpec()
    env = with_mode(environment, mode)
    gate_name = "boot" if mode == MODE_REQUIRE else f"boot[{mode}]"
    _verdict = verdict_factory(inspect_tree(task_data))
    config = check_configuration(
        task_data,
        environment,
        unit=unit,
        features=task_data.get("features") if features is None else features,
        max_tokens=spec.max_tokens,
        member=member,
    )
    requests = spec.requests(task_data.get("trtllm_repo_path"))
    declared_requests = len(requests)
    limits = EngineLimits.from_config(config)
    # A prompt the engine must split proves chunking only where the engine chunks: a variant
    # that turns chunked prefill off is about chunking too, and owes no split prompt.
    chunks = check_config.sets_chunking(task_data, unit) and limits.chunked_prefill
    if chunks:
        length = chunk_probe_tokens(limits.max_num_tokens, limits.max_seq_len)
        if length is None:
            return _verdict(
                name=gate_name,
                member=member,
                passed=False,
                detail=f"this unit is about chunked prefill, and its engine's limits "
                f"(max_num_tokens {limits.max_num_tokens}, max_seq_len {limits.max_seq_len}) "
                f"leave no room for a prompt the engine would have to split",
            )
        requests.append({"id": "chunk", "probe_tokens": length, "max_tokens": 1})
    if timeout is None:
        declared = resolve_gate_timeout(task_data.get("accuracy") or {}, "boot", unit)
        timeout = declared or BOOT_TIMEOUT_S
    spec_path = write_driver_spec(
        gate_workdir(workdir) / f"boot.{unit or 'unit'}.{member or 'member'}.json",
        config=config,
        requests=requests,
        max_tokens=spec.max_tokens,
        reuse_probe=True,
    )
    command = engine_driver_command(env, spec_path)

    started = time.monotonic()
    result = runner(command, None, timeout)
    elapsed = time.monotonic() - started
    combined = result.stdout + result.stderr
    tail = "\n".join(combined.splitlines()[-40:])
    ran = dict(name=gate_name, member=member, command=command, duration_s=elapsed, stdout_tail=tail)

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
    failure = reuse_failure(
        combined,
        config["llm"],
        options=resolve_llm_api_options(task_data.get("accuracy") or {}, unit),
    )
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
    chunked = ""
    if chunks:
        shortfall = chunk_shortfall(
            [records.get(n) for n in range(len(requests))], limits.max_num_tokens
        )
        if shortfall:
            return _verdict(passed=False, detail=shortfall, **ran)
        chunked = f"; a prompt past max_num_tokens {limits.max_num_tokens} was served in chunks"
    images = sum(len(r.get("images") or ()) for r in requests)
    sent = f", {images} image(s) sent" if images else ""
    return _verdict(
        passed=True,
        detail=f"engine cold-started and produced non-empty generation for {len(texts)} prompt(s)"
        f"{sent}; workload: {spec.source}{chunked}",
        **ran,
    )


def chunk_shortfall(records: Sequence[Mapping[str, Any] | None], max_num_tokens: int) -> str | None:
    """Why no request shows chunked prefill ran; ``None`` when one does.

    A request whose prompt, less the tokens the KV cache resumed, is longer
    than ``max_num_tokens`` cannot be computed in one step: one served is the
    evidence.
    """
    longest = 0
    for record in records:
        if not record:
            continue
        prompt = record.get("prompt_tokens")
        if not isinstance(prompt, int):
            continue
        cached = record.get("cached_tokens")
        computed = prompt - (cached if isinstance(cached, int) else 0)
        if computed > max_num_tokens:
            return None
        longest = max(longest, computed)
    return (
        f"no request was longer than max_num_tokens {max_num_tokens} past what the KV cache "
        f"resumed (the longest computed {longest} prompt tokens), so the engine never split a "
        f"context into chunks and this check shows nothing of chunked prefill"
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
    "chunk_shortfall",
    "resolve_boot_spec",
    "run_boot_gate",
]
