# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared witness launch protocol and complete per-sequence continuation comparisons."""

from __future__ import annotations

import ast
import json
import re
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from .boot import BootSpec
from .check_config import check_configuration
from .check_runtime import ENGINE_DRIVER, ShellRunner, gate_workdir, pin_tuning

#: The request driver's per-sequence text prefix. Only token ids prove identity.
_GENERATED = re.compile(r"Generated text:\s*(.+?)\.\.\.\s*$", re.M)


#: Set to ``1`` on both legs of a token-comparing witness: a leg then also
#: prints each continuation whole, as ``[i] Witness token ids: [...]`` and
#: ``[i] Witness text: '...'``. Only the token ids can carry a pass
#: (:func:`missing_token_ids`).
WITNESS_OUTPUT_ENV = "TRTLLM_WITNESS_FULL_OUTPUT"


_WITNESS_TOKEN_IDS = re.compile(r"Witness token ids:\s*\[([-0-9,\s]*)\]\s*$", re.M)


_WITNESS_TEXT = re.compile(r"Witness text:\s*(.+?)\s*$", re.M)


#: What leg A resolved, exported to an equivalence witness's ``contrast_command``
#: so leg B reads it rather than restating it. ``LEG_A_ARGV`` is leg A's
#: public request-entry argv: ``engine_driver.py --request-json <JSON>``.
#: That JSON holds ``llm`` and ``sampling`` options; prompts are separately in
#: ``PROMPTS_FILE``, a JSON
#: ``{"prompts": [...]}``. Set only where they apply: ``DRAFT_LEN`` when leg A
#: speculates; ``MAX_BATCH_SIZE`` when the block declares ``max_batch_size``,
#: which leg B applies to every engine it starts; ``LEGS_ENV``, a JSON
#: ``{"VAR": "value"}`` of the block's ``legs_env``, which leg B applies to
#: every engine it starts after any setting of its own; ``TRANSCEIVER``, the
#: block's ``cache_transceiver_config`` as JSON. ``ENGINE_LAUNCH`` is the
#: command prefix that starts one engine of the member's width. Leg B's
#: constructor settings, including nested KV-cache settings, are in the JSON
#: above. Leg B must read them there rather than re-derive engine flags.
#:
#: Two ask leg B to act. ``RANK_PAIRS`` (a ``dep`` member only) is
#: ``c:g,c:g,...``, one entry per prompt (:func:`rank_pairs`): leg B pins
#: prompt ``i``'s context-only request to attention-DP rank ``c`` and its
#: generation-only request to rank ``g``, and prints
#: ``[i] Witness ranks: ctx=<c> gen=<g>``. ``ABLATE`` (:data:`ABLATE_STATE`,
#: on the self-test run only): leg B destroys, on the generation engine, the
#: state each generation-only request received, before its first decode step
#: -- every layer's recurrent state, else the request's attention KV pages --
#: and prints ``Witness ablated: <what>`` once per request.
LEG_B_PROMPTS_FILE_ENV = "MV2FLOW_WITNESS_PROMPTS_FILE"


LEG_B_MAX_TOKENS_ENV = "MV2FLOW_WITNESS_MAX_TOKENS"


LEG_B_MEMBER_ENV = "MV2FLOW_WITNESS_MEMBER"


LEG_B_DRAFT_LEN_ENV = "MV2FLOW_WITNESS_DRAFT_LEN"


LEG_B_ARGV_ENV = "MV2FLOW_WITNESS_LEG_A_ARGV"


LEG_B_TIMEOUT_ENV = "MV2FLOW_WITNESS_TIMEOUT_S"


LEG_B_MAX_BATCH_SIZE_ENV = "MV2FLOW_WITNESS_MAX_BATCH_SIZE"


LEG_B_LEGS_ENV_ENV = "MV2FLOW_WITNESS_LEGS_ENV"


LEG_B_RANK_PAIRS_ENV = "MV2FLOW_WITNESS_RANK_PAIRS"


LEG_B_ABLATE_ENV = "MV2FLOW_WITNESS_ABLATE"


LEG_B_ENGINE_LAUNCH_ENV = "MV2FLOW_WITNESS_ENGINE_LAUNCH"


LEG_B_TRANSCEIVER_ENV = "MV2FLOW_WITNESS_TRANSCEIVER"


#: What `trtllm-llmapi-launch` exports to the process it wraps. An engine that
#: inherits a spawn-proxy address dials the outer wrapper's proxy and hangs, so
#: these are unset before a contrast command runs.
LAUNCHER_EXPORTS: tuple[str, ...] = (
    "TLLM_SPAWN_PROXY_PROCESS",
    "TLLM_SPAWN_PROXY_PROCESS_IPC_ADDR",
    "TLLM_SPAWN_PROXY_PROCESS_IPC_HMAC_KEY",
    "tllm_mpi_size",
    "FLASHINFER_WORKSPACE_BASE",
    "TRTLLM_FLASHINFER_WORKSPACE_MANAGED",
)


#: Set to ``1`` on both legs of a replay or kv_quant witness: a target prints
#: its counter lines only under it.
WITNESS_COUNTER_ENV = "TRTLLM_MODELING_V2_WITNESS"


#: The counter line a target prints. Space-separated ``key=<int>`` fields on
#: one line; anything may precede the marker (a logger prefix, a rank tag).
#: Two kinds share it.
#:
#: Once per context request, when its context phase completes (after its final
#: chunk), the replay counters -- each for that one request over its whole
#: context phase::
#:
#:     MODELING_V2_WITNESS prompt_tokens=4103 forwarded_tokens=4103 decoder_tokens=128
#:
#: ``prompt_tokens``     the request's prompt length
#: ``reused_tokens``     prompt tokens served from the prefix cache
#: ``replayed_tokens``   cached tokens forwarded again in replay mode
#: ``forwarded_tokens``  rows the model's FIRST layer processed
#: ``decoder_tokens``    rows the first bounded layer -- the one after the last
#:                       KV-source layer -- processed
#:
#: Rows, not positions. A field a witness does not read may be omitted.
#:
#: Once per rank, after the KV pools are allocated, the storage line
#: ``MODELING_V2_WITNESS kv_bytes_per_token=<int>`` -- see
#: :func:`run_kv_quant_witness`. Each witness reads only the lines carrying its
#: own fields, so one target can print both kinds.
WITNESS_COUNTER_MARKER = "MODELING_V2_WITNESS"


_COUNTER_LINE = re.compile(rf"{WITNESS_COUNTER_MARKER}((?:[ \t]+[a-z_]+=-?\d+)+)")


_COUNTER_FIELD = re.compile(r"([a-z_]+)=(-?\d+)")


def parse_generated_texts(stdout: str) -> tuple[str, ...]:
    """Every generated continuation in a request run's output, in order."""
    return tuple(match.strip() for match in _GENERATED.findall(stdout))


def _unrepr(text: str) -> str:
    """The string a ``{text!r}`` printed, or ``text`` itself if it is not one."""
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    return value if isinstance(value, str) else text


def parse_witness_token_ids(stdout: str) -> tuple[tuple[int, ...], ...]:
    """Every ``Witness token ids:`` list a leg printed, in order."""
    return tuple(
        tuple(int(t) for t in re.findall(r"-?\d+", match))
        for match in _WITNESS_TOKEN_IDS.findall(stdout)
    )


def parse_witness_texts(stdout: str) -> tuple[str, ...]:
    """Every ``Witness text:`` continuation a leg printed, whole, in order."""
    return tuple(_unrepr(match) for match in _WITNESS_TEXT.findall(stdout))


def first_difference(a: Sequence[Any], b: Sequence[Any]) -> int:
    """The first position at which ``a`` and ``b`` differ.

    The shorter length when one is a prefix of the other -- a continuation cut
    short differs where it stops.
    """
    for position, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return position
    return min(len(a), len(b))


#: How far either side of a first difference a verdict shows, per unit.
_EXCERPT_WIDTH = {"token": 8, "character": 60}


_HEADLINE_WIDTH = {"token": 4, "character": 30}


#: Differing continuations an excerpt shows in full; the rest are counted.
_EXCERPT_CONTINUATIONS = 4


def _window(a: Sequence[Any], b: Sequence[Any], unit: str, width: int) -> tuple[str, Any, Any]:
    """``a`` and ``b`` around where they first differ, and a label saying where."""
    at = first_difference(a, b)
    lo, hi = max(0, at - width), min(at + width, max(len(a), len(b)))
    if isinstance(a, str):
        return f"[{lo}:{hi}] around {unit} {at}", a[lo:hi], b[lo:hi]
    return f"[{lo}:{hi}] around {unit} {at}", list(a[lo:hi]), list(b[lo:hi])


@dataclass(frozen=True)
class ContinuationComparison:
    """Two legs' continuations, compared on the strongest observable both printed.

    ``texts_a`` / ``texts_b`` are the same continuations as text -- whole where
    both legs printed them, else the 200-character prefixes -- so that a
    difference found in token ids can also be SHOWN as words.
    """

    observable: str
    unit: str
    a: tuple[Any, ...]
    b: tuple[Any, ...]
    texts_a: tuple[str, ...] = ()
    texts_b: tuple[str, ...] = ()

    @property
    def mismatched(self) -> tuple[int, ...]:
        return tuple(i for i, (x, y) in enumerate(zip(self.a, self.b)) if x != y)

    @property
    def identical(self) -> bool:
        return bool(self.a) and len(self.a) == len(self.b) and not self.mismatched

    def first_differences(self) -> dict[int, int]:
        """Continuation index -> the first differing token or character in it."""
        return {i: first_difference(self.a[i], self.b[i]) for i in self.mismatched}

    def describe_differences(self, names: tuple[str, str] = ("leg A", "leg B")) -> str:
        """Where each differing continuation first differs, and what both legs had there."""
        if len(self.a) != len(self.b):
            return f"{names[0]} printed {len(self.a)} continuation(s), {names[1]} {len(self.b)}"
        parts = []
        for i, at in self.first_differences().items():
            parts.append(
                f"#{i} first differs at {self.unit} {at} "
                f"({names[0]} {len(self.a[i])}, {names[1]} {len(self.b[i])} {self.unit}s)"
            )
        return "; ".join(parts)

    def _texts(self, i: int) -> tuple[str, str] | None:
        """Continuation ``i`` as text on both legs, where ``a``/``b`` are not text already."""
        if self.unit == "character" or i >= min(len(self.texts_a), len(self.texts_b)):
            return None
        return self.texts_a[i], self.texts_b[i]

    def headline(self, names: tuple[str, str] = ("leg A", "leg B")) -> str:
        """One line: both legs' first differing continuation around where it differs.

        As words where the legs' texts differ, since that is what a reader can
        judge; as token ids when the ids differ and the text does not.
        """
        if not self.mismatched:
            return ""
        i = self.mismatched[0]
        texts = self._texts(i)
        if texts is not None and texts[0] != texts[1]:
            where, a, b = _window(*texts, "character", _HEADLINE_WIDTH["character"])
            label = "text "
        else:
            where, a, b = _window(self.a[i], self.b[i], self.unit, _HEADLINE_WIDTH[self.unit])
            label = "token ids " if self.unit == "token" else ""
        return f"#{i} {label}{where}: {names[0]} {a!r} vs {names[1]} {b!r}"

    def excerpt(self, names: tuple[str, str] = ("leg A", "leg B")) -> str:
        """Both legs around the first difference of each differing continuation.

        The compared observable first, then -- when that is token ids -- the
        text around its own first difference, because an id list does not say
        what was said.
        """
        if len(self.a) != len(self.b) or not self.mismatched:
            return f"{names[0]}: {self.a[:2]}\n{names[1]}: {self.b[:2]}"
        pad = max(len(n) for n in names)

        def block(label: str, a: Sequence[Any], b: Sequence[Any], unit: str) -> list[str]:
            if a == b:
                return [f"  {label}: identical on both legs"]
            where, wa, wb = _window(a, b, unit, _EXCERPT_WIDTH[unit])
            return [
                f"  {label} {where}:",
                f"    {names[0]:<{pad}}: {wa!r}",
                f"    {names[1]:<{pad}}: {wb!r}",
            ]

        lines: list[str] = []
        for i in self.mismatched[:_EXCERPT_CONTINUATIONS]:
            at = first_difference(self.a[i], self.b[i])
            lines.append(
                f"continuation #{i}: first differs at {self.unit} {at} "
                f"({names[0]} {len(self.a[i])}, {names[1]} {len(self.b[i])} {self.unit}s)"
            )
            lines += block(self.observable, self.a[i], self.b[i], self.unit)
            texts = self._texts(i)
            if texts is not None:
                lines += block("text", *texts, "character")
        rest = len(self.mismatched) - _EXCERPT_CONTINUATIONS
        if rest > 0:
            lines.append(f"... and {rest} more differing continuation(s), listed in the verdict")
        return "\n".join(lines)


#: Strongest first. A leg "printed" an observable when it printed it for every
#: continuation it printed a `Generated text:` line for.
_OBSERVABLES: tuple[tuple[str, str, Callable[[str], tuple[Any, ...]]], ...] = (
    ("token ids", "token", parse_witness_token_ids),
    ("full text", "character", parse_witness_texts),
)


PREFIX_OBSERVABLE = "200-character text prefix"


def compare_continuations(out_a: str, out_b: str) -> ContinuationComparison:
    """Compare two legs' continuations on the strongest observable BOTH printed.

    Token ids, else the full text, else the 200-character prefix every
    request run prints. The weaker two can show a difference but not
    identity (:func:`missing_token_ids`).
    """
    prefix_a, prefix_b = parse_generated_texts(out_a), parse_generated_texts(out_b)
    prefix_a = tuple(_unrepr(t) for t in prefix_a)
    prefix_b = tuple(_unrepr(t) for t in prefix_b)
    texts_a, texts_b = prefix_a, prefix_b
    whole_a, whole_b = parse_witness_texts(out_a), parse_witness_texts(out_b)
    if whole_a and whole_b and len(whole_a) == len(prefix_a) and len(whole_b) == len(prefix_b):
        texts_a, texts_b = whole_a, whole_b
    for observable, unit, parse in _OBSERVABLES:
        a, b = parse(out_a), parse(out_b)
        if a and b and len(a) == len(prefix_a) and len(b) == len(prefix_b):
            return ContinuationComparison(observable, unit, a, b, texts_a, texts_b)
    return ContinuationComparison(PREFIX_OBSERVABLE, "character", prefix_a, prefix_b)


def prints_token_ids(stdout: str) -> bool:
    """Whether a leg printed a ``Witness token ids:`` line for every continuation it printed."""
    ids = parse_witness_token_ids(stdout)
    return bool(ids) and len(ids) == len(parse_generated_texts(stdout))


#: How each leg comes to print the lines a token comparison reads.
_TOKEN_IDS_FIX = (
    f"With {WITNESS_OUTPUT_ENV}=1 set, every leg must print `[i] Witness token ids: [...]` "
    f"and `[i] Witness text: '...'` after each `Generated text:` line. The public request "
    f"entry prints every sequence; a contrast_command must print token ids and text itself. "
    f"Read shared LLM/SamplingParams settings from the --request-json payload in {LEG_B_ARGV_ENV}; "
    f"keep the prompts, rank-pairs, ablation and launch-prefix protocols unchanged"
)


def missing_token_ids(
    comparison: ContinuationComparison,
    legs: Sequence[tuple[str, str]],
    minimum: int | None = None,
) -> str | None:
    """Why a comparison that found no difference still cannot pass, or ``None``.

    ``legs`` is each leg's ``(name, output)``. Only token ids carry identity:
    two token sequences can detokenize to one string, and the 200-character
    prefix is under a quarter of a 256-token continuation.
    """
    if comparison.unit == "token":
        return None
    bare = [name for name, output in legs if not prints_token_ids(output)]
    held = f", nor held to min_generated_tokens={minimum}," if minimum else ""
    return (
        f"{len(comparison.a)} continuation(s) agree as {comparison.observable}, which is not "
        f"token identity: the {' and '.join(bare) or 'one'} leg(s) did not print a `Witness "
        f"token ids:` line for every continuation, so the continuations were not compared "
        f"token by token{held} and the witness does not pass. {_TOKEN_IDS_FIX}"
    )


def too_short(comparison: ContinuationComparison, minimum: int | None) -> str | None:
    """Why ``comparison`` is too short to judge, or ``None``; only token ids carry a length."""
    if not minimum or comparison.unit != "token" or not comparison.a:
        return None
    lengths = [len(ids) for ids in (*comparison.a, *comparison.b)]
    if min(lengths) >= minimum:
        return None
    return (
        f"the continuations are too short to compare: the shortest generated "
        f"{min(lengths)} token(s), under min_generated_tokens={minimum}, so identity "
        f"would rest on the kernels' certification rather than on an end-to-end "
        f"comparison. Declare a workload with long answers in the witness block "
        f"(`prompts` / `prompts_file`, `max_tokens`)"
    )


def parse_witness_counters(stdout: str) -> list[dict[str, int]]:
    """Every counter line a target printed, one mapping per context request.

    A multi-rank run may print a request once per rank; every verdict is a
    per-line statement, so a duplicated line changes nothing.
    """
    return [
        {key: int(value) for key, value in _COUNTER_FIELD.findall(match.group(1))}
        for match in _COUNTER_LINE.finditer(stdout)
    ]


def witness_command(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int = 64,
    member: str | None = None,
    prompts: Sequence[str] | None = None,
    max_batch_size: int | None = None,
) -> str:
    """A public request entry with explicit warmup and full-sequence witness output."""
    spec = spec or BootSpec()
    config = _witness_config(
        task_data,
        environment,
        unit=unit,
        features=features,
        max_tokens=max_tokens,
        member=member,
        max_batch_size=max_batch_size,
    )
    words = [shlex.quote(str(ENGINE_DRIVER)), "--request-json", shlex.quote(json.dumps(config))]
    for prompt in spec.prompts if prompts is None else prompts:
        words.extend(("--prompt", shlex.quote(prompt)))
    return " ".join(words)


def _witness_config(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    features: Any | None,
    max_tokens: int,
    member: str | None,
    max_batch_size: int | None = None,
) -> dict[str, Any]:
    config = check_configuration(
        task_data, environment, unit=unit, features=features, max_tokens=max_tokens, member=member
    )
    if max_batch_size is not None:
        config["llm"]["max_batch_size"] = int(max_batch_size)
    config.update(warmup=True, witness=True, concurrency=1, requests=[])
    return config


def request_argv(
    task_data: Mapping[str, Any], environment: RunEnvironment, **shape: Any
) -> list[str]:
    """Leg A's runnable request entry, without prompts; the prompt file remains separate."""
    config = _witness_config(task_data, environment, **shape)
    return [shlex.quote(str(ENGINE_DRIVER)), "--request-json", shlex.quote(json.dumps(config))]


def schedule_note(max_batch_size: int | None) -> str:
    """What a verdict says about the legs' batch limit, when the block declared one."""
    if max_batch_size is None:
        return ""
    return f"; both legs at max_batch_size={max_batch_size}"


def legs_note(legs_env: Mapping[str, str] | None) -> str:
    """What a verdict says about the environment both legs shared, when declared."""
    if not legs_env:
        return ""
    return "; both legs under " + ", ".join(f"{k}={v}" for k, v in sorted(legs_env.items()))


def around(comparison: ContinuationComparison, names: tuple[str, str]) -> str:
    """``; around it, <both legs at the first difference>``, or nothing."""
    headline = comparison.headline(names)
    return f"; around it, {headline}" if headline else ""


def token_difference(comparison: ContinuationComparison, names: tuple[str, str]) -> str:
    """The failing half of a token-identity verdict: where, and what both legs had there."""
    if not comparison.a or not comparison.b:
        return (
            f" (no continuations parsed from both legs: {names[0]} {len(comparison.a)}, "
            f"{names[1]} {len(comparison.b)})"
        )
    return (
        f" (compared as {comparison.observable}): {comparison.describe_differences(names)}"
        f"{around(comparison, names)}"
    )


def leg_pins(workdir: str | Path, member: str | None) -> dict[str, str]:
    """The tuning pins (:func:`~.check_runtime.pin_tuning`) one witness run's legs share, per member."""
    return pin_tuning(gate_workdir(workdir) / str(member or "member"))


def run_leg(
    environment: RunEnvironment,
    argv: str,
    runner: ShellRunner,
    timeout: int,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str, bool]:
    """Run one leg. Returns (command, combined output, ok)."""
    environment = replace(environment, env={**environment.env, **(env or {})})
    command = environment.python_command(argv)
    result = runner(command, None, timeout)
    return command, result.stdout + result.stderr, result.ok


def contrast_leg_command(environment: RunEnvironment, contrast_command: str) -> str:
    """How an equivalence witness launches ``contrast_command``: as one process, not a rank.

    Leg B starts engines of its own, so it runs as a single process over the
    claim's devices, with :data:`LAUNCHER_EXPORTS` unset. A command whose
    first word is a ``.py`` file runs under the run's interpreter; anything
    else runs as written.
    """
    try:
        words = shlex.split(contrast_command)
    except ValueError:
        words = contrast_command.split()
    body = contrast_command
    if words and words[0].endswith(".py"):
        body = f"{shlex.quote(environment.python)} {contrast_command}"
    body = "env " + " ".join(f"-u {name}" for name in LAUNCHER_EXPORTS) + f" {body}"
    return environment.command(body, single_process=True)


def engine_launch(environment: RunEnvironment) -> str:
    """How leg B starts one engine of the member's width (:data:`LEG_B_ENGINE_LAUNCH_ENV`).

    The run's own launch of engine work over the run's interpreter, without
    its devices or environment: leg B places each engine on devices of its
    own and starts it from its own environment.
    """
    return replace(environment, devices=(), env={}).python_command("").strip()


__all__ = [
    "ContinuationComparison",
    "LAUNCHER_EXPORTS",
    "LEG_B_ABLATE_ENV",
    "LEG_B_ARGV_ENV",
    "LEG_B_DRAFT_LEN_ENV",
    "LEG_B_ENGINE_LAUNCH_ENV",
    "LEG_B_LEGS_ENV_ENV",
    "LEG_B_MAX_BATCH_SIZE_ENV",
    "LEG_B_MAX_TOKENS_ENV",
    "LEG_B_MEMBER_ENV",
    "LEG_B_PROMPTS_FILE_ENV",
    "LEG_B_RANK_PAIRS_ENV",
    "LEG_B_TIMEOUT_ENV",
    "LEG_B_TRANSCEIVER_ENV",
    "PREFIX_OBSERVABLE",
    "WITNESS_COUNTER_ENV",
    "WITNESS_COUNTER_MARKER",
    "WITNESS_OUTPUT_ENV",
    "around",
    "compare_continuations",
    "contrast_leg_command",
    "engine_launch",
    "first_difference",
    "leg_pins",
    "legs_note",
    "missing_token_ids",
    "parse_generated_texts",
    "parse_witness_counters",
    "parse_witness_texts",
    "parse_witness_token_ids",
    "prints_token_ids",
    "request_argv",
    "run_leg",
    "schedule_note",
    "token_difference",
    "too_short",
    "witness_command",
]
