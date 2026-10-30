# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold-start generation, boot workload coverage and the optional boot boundary phases."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import (
    BOOT_KEYS,
    TaskSchemaError,
    load_prompts_file,
    members_for,
    parse_parallel_segment,
)
from ..spec.variants import is_variant
from . import check_config, divergence, feature_evidence
from .check_config import (
    EngineLimits,
    check_configuration,
    engine_declaration,
    resolve_llm_api_options,
)
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    ShellRunner,
    default_shell_runner,
    engine_driver_command,
    gate_workdir,
    parse_driver_records,
    pin_tuning,
    with_mode,
    write_driver_spec,
)
from .runtime_evidence import attest_engine, engine_failure, engine_records
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
        image = BootPrompt(
            prompt=DEFAULT_BOOT_IMAGE_PROMPT,
            keywords_any=DEFAULT_BOOT_IMAGE_KEYWORDS,
            images=(DEFAULT_BOOT_IMAGE,),
        )
        spec = replace(spec, items=(*spec.items, image))
    return spec


def _declares_workload(block: Mapping[str, Any]) -> bool:
    """Whether a level of ``accuracy.boot`` says what to send, at it or below."""
    return any(
        key in BOOT_KEYS or not isinstance(value, Mapping) or bool(value)
        for key, value in block.items()
    )


#: The top log-probs each boundary probe's tokens carry, to tell a near-tie from a defect.
BOUNDARY_TOP_K = 5
#: Kinds of member in the order the boundary phases prefer one: the member that batches most.
_BOUNDARY_KINDS = {"dep": 0, "tep": 1, "tp": 2}


def boundary_member(members: Sequence[str]) -> str | None:
    """The member a unit's boundary phases run at: attention DP first, then the widest."""

    def key(member: str) -> tuple[int, int, str]:
        try:
            kind, size = parse_parallel_segment(member)
        except TaskSchemaError:
            return (len(_BOUNDARY_KINDS) + 1, 0, member)
        return (_BOUNDARY_KINDS.get(kind, len(_BOUNDARY_KINDS)), -(size or 0), member)

    return min(members, key=key) if members else None


def _tie_gap(alone: Mapping[str, Any], loaded: Mapping[str, Any], step: int) -> float | None:
    """How far apart the two legs' tokens at ``step`` are in either leg's own log-probs.

    ``None`` when neither leg's top-k holds both tokens.
    """
    tokens = (alone["token_ids"][step], loaded["token_ids"][step])
    gaps = []
    for record, (own, other) in ((alone, tokens), (loaded, tokens[::-1])):
        rows = record.get("top_logprobs") or ()
        if step >= len(rows):
            continue
        row = {entry[0]: entry[1] for entry in rows[step] if entry and entry[0] is not None}
        if own in row and other in row:
            gaps.append(abs(float(row[own]) - float(row[other])))
    return min(gaps) if gaps else None


@dataclass(frozen=True)
class BootBoundarySpec:
    """Optional runtime-boundary phases of boot, configured by ``accuracy.envelope``.

    Sequential probes and loaded requests share the boot's engine, sampling and
    tuning choices. Cache reuse remains as declared; this is not a cold-cache
    comparison or evidence of scheduler batch sizes or CUDA graph execution.

    They run once per unit, in its boot at one member (``members``, else
    :func:`boundary_member`), never in a configuration variant's. A probe whose
    legs part where the two tokens are within ``tie_margin`` of each other in a
    leg's own log-probs is a near-tie: batch shape moves low-precision
    arithmetic, and a near-tie flips with it without any defect.
    """

    enabled: bool = False
    max_tokens: int = 16
    agree_tokens: int = 4
    long_prompts: int = 2
    probes: tuple[str, ...] = (
        "The capital of France is",
        "Water boils at a temperature of",
        "The largest planet in the solar system is",
        "Two plus two equals",
    )
    timeout: int = 2 * BOOT_TIMEOUT_S
    #: Nats between a probe's two legs' tokens that still reads as a near-tie.
    tie_margin: float = 0.15
    #: The members whose boot runs the phases; empty: one per unit.
    members: tuple[str, ...] = ()

    @classmethod
    def from_task(cls, task_data: Mapping[str, Any]) -> "BootBoundarySpec":
        block = {
            key: value
            for key, value in ((task_data.get("accuracy") or {}).get("envelope") or {}).items()
            if value is not None
        }
        for key in ("probes", "members"):
            if key in block:
                block[key] = tuple(block[key])
        return cls(**block)

    def runs_at(
        self, task_data: Mapping[str, Any], unit: str | None, member: str | None, world_size: int
    ) -> bool:
        """Whether ``unit``'s boot at ``member`` runs the phases: one member of each unit."""
        if member is None:
            return True
        if unit is not None and is_variant(task_data, unit):
            return False
        if self.members:
            return member in self.members
        return member == boundary_member(members_for(task_data, unit or "core", world_size))

    def prompt_tokens(self, limits: EngineLimits) -> int:
        """Sequence-bound prompt length, capped by the unchunked token budget."""
        length = int(limits.max_seq_len or 0) - self.max_tokens - 1
        if not limits.chunked_prefill:
            length = min(length, limits.max_num_tokens - 1)
        return max(length, 0)

    def phases(self, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Sequential baseline then bounded loaded windows, without another LLM launch."""
        limits = EngineLimits.from_config(config)
        if limits.max_seq_len is None:
            raise ValueError(
                "boot boundaries require an explicit max_seq_len in the runtime options"
            )
        if self.agree_tokens > self.max_tokens:
            raise ValueError("accuracy.envelope.agree_tokens must not exceed max_tokens")
        length = self.prompt_tokens(limits)
        if self.long_prompts and not length:
            raise ValueError(
                "boot boundaries leave no room for a long prompt and its output budget"
            )
        probes = [
            {
                "id": f"probe{n}",
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                "logprobs": BOUNDARY_TOP_K,
            }
            for n, prompt in enumerate(self.probes)
        ]
        loaded = [
            {
                "id": f"long{n}",
                "token_ids": [1000 + (n + i) % 997 for i in range(length)],
                "max_tokens": self.max_tokens,
            }
            for n in range(self.long_prompts)
        ] + probes
        concurrency = int(config["llm"]["max_batch_size"])
        for n in range(max(concurrency - len(loaded), 0)):
            loaded.append(
                {
                    "id": f"fill{n}",
                    "prompt": self.probes[n % len(self.probes)],
                    "max_tokens": 1 + n % self.max_tokens,
                }
            )
        return [
            {"id": "baseline", "requests": probes, "concurrency": 1, "ignore_eos": True},
            {"id": "loaded", "requests": loaded, "concurrency": concurrency, "ignore_eos": True},
        ]

    def failure(
        self, phases: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
    ) -> str:
        """Why the phases fail (:meth:`judge`); ``""`` when they pass."""
        return self.judge(phases, records)[0]

    def judge(
        self, phases: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
    ) -> tuple[str, list[str]]:
        """Require complete nonempty outputs, exact long lengths and agreeing probe prefixes.

        Returns why the phases fail (``""`` when they pass) and the probes whose
        legs parted at a near-tie, which pass.
        """
        outputs: dict[tuple[str, str], Mapping[str, Any]] = {}
        for record in records:
            key = (str(record.get("phase", "")), str(record.get("id", "")))
            if key in outputs:
                return f"boot boundaries returned duplicate record {key}", []
            outputs[key] = record
        for phase in phases:
            for request in phase["requests"]:
                key = (phase["id"], request["id"])
                record = outputs.get(key)
                if record is None:
                    return f"boot boundaries returned no record for {key}", []
                if not record.get("token_ids"):
                    return f"boot boundaries generated nothing for {key}", []
                if "token_ids" in request:
                    expected = len(request["token_ids"])
                    observed = record.get("prompt_tokens")
                    if type(observed) is not int or observed != expected:
                        return (
                            f"{key} reached {observed} prompt tokens, not the {expected} "
                            "required by boot boundaries"
                        ), []
        ties: list[str] = []
        for n in range(len(self.probes)):
            key = f"probe{n}"
            alone_record, loaded_record = outputs[("baseline", key)], outputs[("loaded", key)]
            alone, loaded = alone_record["token_ids"], loaded_record["token_ids"]
            need = self.agree_tokens
            said = f"{key} must agree on {need} tokens: baseline {alone[:need]} vs loaded {loaded[:need]}"
            if len(alone) < need or len(loaded) < need:
                return said, ties
            step = next((s for s in range(need) if alone[s] != loaded[s]), None)
            if step is None:
                continue
            gap = _tie_gap(alone_record, loaded_record, step)
            if gap is not None and gap <= self.tie_margin:
                ties.append(
                    f"{key} step {step}: baseline {alone[step]}, loaded {loaded[step]}, "
                    f"{gap:.3f} apart"
                )
                continue
            why = (
                f"{gap:.3f} apart at step {step}, past the near-tie margin {self.tie_margin:.3f}"
                if gap is not None
                else f"at step {step} no leg's top-{BOUNDARY_TOP_K} log-probs hold both tokens"
            )
            return f"{said} ({why})", ties
        return "", ties

    def describe(self, config: Mapping[str, Any]) -> str:
        limits = EngineLimits.from_config(config)
        length = self.prompt_tokens(limits)
        cap = (
            "" if limits.chunked_prefill else "; long input capped by unchunked max_num_tokens - 1"
        )
        return (
            f"; boot boundaries: {self.long_prompts} long request(s) with exactly {length} "
            f"input tokens{cap}; {len(self.probes)} probe(s) agree on {self.agree_tokens} tokens "
            f"at submitted concurrency up to {config['llm']['max_batch_size']}; "
            "actual scheduler batches and CUDA graph execution are not measured"
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
    (:func:`resolve_boot_spec`). The engine is held to what the gate declared
    (:func:`engine_failure`). ``mode`` other than ``require`` is a control run
    on the stock implementation, recorded as ``boot[<mode>]``. When enabled,
    ``accuracy.envelope`` appends sequential baseline and loaded boundary phases
    to this same engine, without adding a separate contract check, in one
    member's boot of each unit (:meth:`BootBoundarySpec.runs_at`).
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
    subjects = check_config.subject_fields(task_data, unit)
    # A prompt the engine must split proves chunking only where the engine chunks: a variant
    # that turns chunked prefill off is about chunking too, and owes no split prompt.
    chunks = feature_evidence.about(subjects, feature_evidence.CHUNKED) and limits.chunked_prefill
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
    # The reference's prompts, one token each with its top-k, on the same engine: the
    # model-level divergence is recorded with the verdict, never a failure by itself.
    reference, unread = None, ""
    if mode == MODE_REQUIRE:
        try:
            reference = divergence.load_reference(task_data)
        except (OSError, ValueError) as exc:
            unread = f"the reference logits cannot be read ({exc}): divergence UNKNOWN"
    reference_at = len(requests)
    if reference is not None:
        requests.extend(
            {"id": f"reference:{key}", "prompt": text, "max_tokens": 1, "logprobs": reference.top_k}
            for key, text in reference.prompts.items()
        )
    boundary = BootBoundarySpec.from_task(task_data)
    phases = []
    if (
        boundary.enabled
        and mode == MODE_REQUIRE
        and boundary.runs_at(task_data, unit, member, environment.world_size)
    ):
        try:
            phases = boundary.phases(config)
        except ValueError as exc:
            return _verdict(name=gate_name, member=member, passed=False, detail=str(exc))
        if timeout is None:
            timeout = boundary.timeout
        env = replace(
            env,
            env={**env.env, **pin_tuning(Path(workdir) / f"boot.{unit}.{member}")},
        )
    if timeout is None:
        timeout = BOOT_TIMEOUT_S
    spec_path = write_driver_spec(
        gate_workdir(workdir) / f"boot.{unit or 'unit'}.{member or 'member'}.json",
        config=config,
        requests=requests,
        max_tokens=spec.max_tokens,
        reuse_probe=True,
        phases=phases,
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
    if result.timed_out:
        return _verdict(
            passed=False,
            detail=f"timed out after {timeout}s — a boot gate that takes this long is "
            f"not catching catastrophes any more",
            **ran,
        )
    if not result.ok:
        return _verdict(passed=False, detail=f"exited {result.returncode}", **ran)
    declared = engine_declaration(config, mode=mode)
    attested = attest_engine(combined, declared)
    ran["engine"] = engine_records((attested,))
    failure = engine_failure(
        combined,
        attested,
        options=resolve_llm_api_options(task_data.get("accuracy") or {}, unit),
        asks=bool(declared.get("block_reuse")),
    )
    if failure:
        return _verdict(passed=False, detail=failure, **ran)
    unobserved = feature_evidence.unobserved(attested, subjects)
    if unobserved:
        return _verdict(passed=False, detail=feature_evidence.unobserved_detail(unobserved), **ran)

    all_records = parse_driver_records(combined)
    boundary_ties: list[str] = []
    if phases:
        failure, boundary_ties = boundary.judge(phases, all_records)
        if failure:
            return _verdict(passed=False, detail=failure, **ran)
    records = {
        int(r.get("index", -1)): r for r in all_records if not phases or r.get("phase") == "boot"
    }
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
    missing = missing_keywords(spec, texts)
    if missing:
        return _verdict(
            passed=False,
            detail=f"the generated texts are missing {len(missing)} expected keyword(s): "
            f"{missing}. Keywords are matched against generated text only",
            **ran,
        )
    compared, warnings = "", []
    if reference is not None:
        measured = {}
        for n in range(reference_at, len(requests)):
            steps = (records.get(n) or {}).get("top_logprobs") or [[]]
            row = {int(e[0]): float(e[1]) for e in steps[0] or () if e and e[0] is not None}
            measured[str(requests[n]["id"]).partition(":")[2]] = row
        found = divergence.assess(reference.first_tokens, measured, reference.bounds)
        compared = f"; {found.describe()}"
        if not found.within:
            warnings.append(
                f"divergence from the reference recorded: {len(found.crossed)} of "
                f"{found.prompts} prompt(s) past a bound (the verdict stands)"
            )
    elif unread:
        compared, warnings = f"; {unread}", [unread]
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
        f"{sent}; workload: {spec.source}{chunked}"
        + (boundary.describe(config) if phases else "")
        + (
            f"; boundary probes agreed up to {len(boundary_ties)} near-tie(s) "
            f"({'; '.join(boundary_ties)})"
            if boundary_ties
            else ""
        )
        + compared,
        warnings=tuple(warnings),
        explained=bool(compared),
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
    "BootBoundarySpec",
    "BootPrompt",
    "BootSpec",
    "DEFAULT_BOOT_IMAGE",
    "DEFAULT_BOOT_IMAGE_KEYWORDS",
    "DEFAULT_BOOT_IMAGE_PROMPT",
    "chunk_probe_tokens",
    "chunk_shortfall",
    "missing_keywords",
    "resolve_boot_spec",
    "run_boot_gate",
]
