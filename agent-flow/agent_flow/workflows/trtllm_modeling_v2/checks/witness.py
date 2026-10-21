# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gates for features the boot and accuracy gates cannot see.

Both standing gates pass whether or not such a feature does anything, so each
witness runs one workload twice in the same session under one changed variable
and compares a named observable. None of them measures speed.

| gate | variable | observable | verdict |
|---|---|---|---|
| residency | the offload switch | device memory after load | delta >= predicted |
| equivalence | aggregated vs disaggregated | the generated tokens | identical; not without the state |
| replay | the replay switch | per-request row counters | bounded ON, full OFF |
| kv_quant | native vs bf16 storage | bytes/token + tokens | declared; identical |
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import tempfile
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..infra.environment import RunEnvironment
from ..spec.task_schema import (
    DEFAULT_BYTES_TOLERANCE,
    WITNESS_BATCH_SIZE_KEY,
    WITNESS_LEGS_ENV_KEY,
    WITNESS_TRANSCEIVER_KEY,
    load_prompts_file,
    parse_parallel_segment,
)
from .gates import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    MODELING_V2_ENV,
    UNKNOWN_TREE_STATE,
    BootSpec,
    GateResult,
    ShellRunner,
    _default_shell_runner,
    _with_mode,
    attest_engine,
    engine_declaration,
    engine_fields,
    gate_workdir,
    inspect_tree,
    parse_engine_records,
    pin_tuning,
    quickstart_flags,
    quickstart_script,
    resolve_llm_api_options,
    resolve_max_draft_len,
    resolve_runtime,
    reuse_disabled_detail,
    reuse_shortfall,
    speculation_flags,
    topology_flags,
    with_kv_cache_config,
    with_max_batch_size,
)
from .witness_text import natural_text

#: `[0] Prompt: '...', Generated text: '...'` -- what
#: `examples/llm-api/quickstart_advanced.py` prints per request. The
#: equivalence witness compares these, and nothing else on the line.
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
#: quickstart argv without the prompts, which are in ``PROMPTS_FILE``, a JSON
#: ``{"prompts": [...]}``. Set only where they apply: ``DRAFT_LEN`` when leg A
#: speculates; ``MAX_BATCH_SIZE`` when the block declares ``max_batch_size``,
#: which leg B applies to every engine it starts; ``LEGS_ENV``, a JSON
#: ``{"VAR": "value"}`` of the block's ``legs_env``, which leg B applies to
#: every engine it starts after any setting of its own; ``TRANSCEIVER``, the
#: block's ``cache_transceiver_config`` as JSON. ``ENGINE_LAUNCH`` is the
#: command prefix that starts one engine of the member's width. Leg B's
#: environment also carries :data:`~.gates.KV_CACHE_CONFIG_ENV` when the unit's
#: options hand the quickstart KV-cache keys.
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
#: The one value :data:`LEG_B_ABLATE_ENV` takes.
ABLATE_STATE = "state"
_WITNESS_RANKS = re.compile(r"\[(\d+)\] Witness ranks:\s*ctx=(\d+)[,\s]+gen=(\d+)")
_WITNESS_ABLATED = re.compile(r"Witness ablated:\s*(\S.*?)\s*$", re.M)

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

#: The residency witness's reading, ``__DEVICE_MEMORY_MIB__ <MiB>``, which the
#: TARGET prints once per rank after its weights load and before the KV cache
#: pool is allocated (``torch.cuda.memory_allocated()`` in MiB): the pool takes
#: what an offload saved.
DEVICE_MEMORY_MARKER = "__DEVICE_MEMORY_MIB__"
_DEVICE_MEMORY = re.compile(rf"{DEVICE_MEMORY_MARKER}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)")

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

#: What each replay witness reads off a counter line.
REPLAY_COUNTER_FIELDS: dict[str, tuple[str, ...]] = {
    "decoder_replay": ("forwarded_tokens", "decoder_tokens"),
    "encoder_replay": ("prompt_tokens", "reused_tokens", "replayed_tokens", "forwarded_tokens"),
}

#: What the kv_quant witness reads: the persistent KV bytes one rank stores per
#: token of a request it serves, summed over its pools as ALLOCATED -- a ratio-r
#: compressed pool contributes its entry size over r. Fixed per-request state
#: (the SWA window, compressor carry-over) is not per-token and is excluded.
KV_BYTES_FIELD = "kv_bytes_per_token"

# One short word is at least one token under every tokenizer that splits on
# whitespace before merging, which is every tokenizer these checkpoints ship.
# So `n` words is a prompt of at least `n` tokens, and the verdict checks the
# target's own count against that floor rather than trusting it.
_FILLER = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def parse_generated_texts(stdout: str) -> tuple[str, ...]:
    """Every generated continuation in a quickstart run's output, in order."""
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
    quickstart run prints. The weaker two can show a difference but not
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
    f"and `[i] Witness text: '...'` after each `Generated text:` line. A quickstart leg "
    f"prints them from the measured tree's examples/llm-api/quickstart_advanced.py, so that "
    f"script must print `list(sequence.token_ids)` and `sequence.text!r` for each sequence "
    f"it prints when the variable is 1; a contrast_command prints them itself"
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


#: Generated tokens a token-comparing witness's continuations must reach,
#: when the block does not say. Identity over a 2-token answer rests on the
#: kernels' certification, not on an end-to-end comparison.
DEFAULT_MIN_GENERATED_TOKENS = 32


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


@dataclass(frozen=True)
class WitnessWorkload:
    """What a token-comparing witness sends, resolved from its block."""

    prompts: tuple[str, ...]
    max_tokens: int
    min_generated_tokens: int
    source: str


def resolve_witness_workload(
    task_data: Mapping[str, Any],
    block: Mapping[str, Any],
    boot: BootSpec,
    *,
    default_max_tokens: int,
) -> WitnessWorkload:
    """The witness block's own workload, else the declared acceptance prompts, else the boot prompts."""
    if block.get("prompts"):
        prompts, source = tuple(str(p) for p in block["prompts"]), "the witness block's prompts"
    elif block.get("prompts_file"):
        prompts = load_prompts_file(block["prompts_file"])
        source = f"the witness block's prompts_file ({block['prompts_file']})"
    else:
        acceptance = (task_data.get("accuracy") or {}).get("acceptance")
        acceptance = acceptance if isinstance(acceptance, Mapping) else {}
        if acceptance.get("prompts"):
            prompts = tuple(str(p) for p in acceptance["prompts"])
            source = "accuracy.acceptance.prompts"
        elif acceptance.get("prompts_file"):
            prompts = load_prompts_file(acceptance["prompts_file"])
            source = "accuracy.acceptance.prompts_file"
        else:
            prompts, source = boot.prompts, boot.source
    return WitnessWorkload(
        prompts=prompts,
        max_tokens=int(block.get("max_tokens") or default_max_tokens),
        min_generated_tokens=int(block.get("min_generated_tokens") or DEFAULT_MIN_GENERATED_TOKENS),
        source=source,
    )


def parse_device_memory_mib(stdout: str) -> float | None:
    """The last device-memory reading a witness leg printed (:data:`DEVICE_MEMORY_MARKER`), in MiB."""
    matches = _DEVICE_MEMORY.findall(stdout)
    return float(matches[-1]) if matches else None


def parse_witness_counters(stdout: str) -> list[dict[str, int]]:
    """Every counter line a target printed, one mapping per context request.

    A multi-rank run may print a request once per rank; every verdict is a
    per-line statement, so a duplicated line changes nothing.
    """
    return [
        {key: int(value) for key, value in _COUNTER_FIELD.findall(match.group(1))}
        for match in _COUNTER_LINE.finditer(stdout)
    ]


def filler_words(count: int) -> str:
    """``count`` short words: a prompt of at least ``count`` tokens, for the replay witnesses."""
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(count))


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
    """The quickstart argv both legs of a witness run, at ``member``'s topology.

    ``prompts`` overrides the boot spec's; ``max_batch_size`` replaces whatever
    batch limit the runtime block set (:func:`with_max_batch_size`).
    """
    spec = spec or BootSpec()
    quoted = " ".join(shlex.quote(p) for p in (spec.prompts if prompts is None else prompts))
    return " ".join(
        [
            # The tree's quickstart, run through the shim that builds the target the gates
            # name (`modeling_v2`) and prints each continuation whole.
            str(QUICKSTART_SHIM),
            *_quickstart_argv(
                task_data,
                environment,
                unit=unit,
                features=features,
                max_tokens=max_tokens,
                member=member,
                max_batch_size=max_batch_size,
            ),
            f"--prompt {quoted}",
        ]
    )


#: Runs the tree's quickstart as a witness leg needs it; see the module.
QUICKSTART_SHIM = Path(__file__).resolve().parent / "harness" / "quickstart_shim.py"


def _quickstart_argv(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    features: Any | None,
    max_tokens: int,
    member: str | None,
    max_batch_size: int | None = None,
) -> list[str]:
    """:func:`witness_command` without its prompts."""
    return [
        str(quickstart_script(task_data)),
        f'--model_dir="{task_data["checkpoint_path"]}"',
        f"--max_tokens={max_tokens}",
        *topology_flags(str(task_data["parallel"]), environment.world_size, member),
        *with_max_batch_size(quickstart_flags(task_data, unit, features), max_batch_size),
    ]


def _schedule_note(max_batch_size: int | None) -> str:
    """What a verdict says about the legs' batch limit, when the block declared one."""
    if max_batch_size is None:
        return ""
    return f"; both legs at max_batch_size={max_batch_size}"


def _legs_note(legs_env: Mapping[str, str] | None) -> str:
    """What a verdict says about the environment both legs shared, when declared."""
    if not legs_env:
        return ""
    return "; both legs under " + ", ".join(f"{k}={v}" for k, v in sorted(legs_env.items()))


def _around(comparison: ContinuationComparison, names: tuple[str, str]) -> str:
    """``; around it, <both legs at the first difference>``, or nothing."""
    headline = comparison.headline(names)
    return f"; around it, {headline}" if headline else ""


def _token_difference(comparison: ContinuationComparison, names: tuple[str, str]) -> str:
    """The failing half of a token-identity verdict: where, and what both legs had there."""
    if not comparison.a or not comparison.b:
        return (
            f" (no continuations parsed from both legs: {names[0]} {len(comparison.a)}, "
            f"{names[1]} {len(comparison.b)})"
        )
    return (
        f" (compared as {comparison.observable}): {comparison.describe_differences(names)}"
        f"{_around(comparison, names)}"
    )


def _leg_pins(
    workdir: str | Path | None, kind: str, task_data: Mapping[str, Any], member: str | None
) -> dict[str, str]:
    """The tuning pins (:func:`~.gates.pin_tuning`) one witness run's legs share, per member."""
    return pin_tuning(gate_workdir(workdir, f"witness-{kind}", task_data) / str(member or "member"))


def _run_leg(
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


def run_residency_witness(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    expected_mib: float,
    contrast_env: Mapping[str, str],
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    member: str | None = None,
    max_batch_size: int | None = None,
    prompts: Sequence[str] | None = None,
    max_tokens: int = 64,
    min_generated_tokens: int | None = None,
    workdir: str | Path | None = None,
) -> GateResult:
    """Prove a host-offloaded tensor is actually not resident.

    Two legs differing only by ``contrast_env``, the environment that turns
    the offload OFF. Both must produce **token-identical** output, and the
    resident leg must use at least ``expected_mib`` more device memory
    (:data:`DEVICE_MEMORY_MARKER`) than the offloaded one. ``expected_mib`` is
    computed from the checkpoint's own shapes, never from the run judged.
    """
    _verdict = partial(GateResult, name="residency", member=member)
    argv = witness_command(
        task_data,
        environment,
        unit=unit,
        spec=spec,
        features=features,
        member=member,
        max_batch_size=max_batch_size,
        prompts=prompts,
        max_tokens=max_tokens,
    )
    full_output = {WITNESS_OUTPUT_ENV: "1"}
    pins = _leg_pins(workdir, "residency", task_data, member)
    started = time.monotonic()

    on_cmd, on_out, on_ok = _run_leg(
        environment, argv, runner, timeout, env={**full_output, **pins}
    )
    if not on_ok:
        return _verdict(
            passed=False,
            detail="the offloaded leg did not complete",
            command=on_cmd,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(on_out.splitlines()[-40:]),
        )

    off_cmd, off_out, off_ok = _run_leg(
        environment, argv, runner, timeout, env={**contrast_env, **full_output, **pins}
    )
    elapsed = time.monotonic() - started
    if not off_ok:
        return _verdict(
            passed=False,
            detail=f"the resident control leg ({dict(contrast_env)}) did not complete",
            command=off_cmd,
            duration_s=elapsed,
            stdout_tail="\n".join(off_out.splitlines()[-40:]),
        )

    on_mib, off_mib = parse_device_memory_mib(on_out), parse_device_memory_mib(off_out)
    if on_mib is None or off_mib is None:
        return _verdict(
            passed=False,
            detail=f"no `{DEVICE_MEMORY_MARKER}` reading from both legs "
            f"(offloaded={on_mib}, resident={off_mib}); a witness with no number "
            f"is not a pass. The target prints `{DEVICE_MEMORY_MARKER} <MiB>` once per "
            f"rank after its weights load and before the KV cache pool is allocated",
            command=on_cmd,
            duration_s=elapsed,
        )

    comparison = compare_continuations(on_out, off_out)
    short = too_short(comparison, min_generated_tokens)
    if short:
        return _verdict(
            passed=False,
            detail=short,
            command=on_cmd,
            duration_s=elapsed,
        )
    if not comparison.identical:
        names = ("offloaded", "resident")
        return _verdict(
            passed=False,
            detail=f"the two legs did not generate identical text, so the gather is "
            f"wrong -- offloading is output-identical by construction"
            f"{_token_difference(comparison, names)}{_schedule_note(max_batch_size)}",
            command=on_cmd,
            duration_s=elapsed,
            stdout_tail=(
                comparison.excerpt(names)
                if comparison.a and comparison.b
                else "\n".join(on_out.splitlines()[-20:])
            ),
        )

    delta = off_mib - on_mib
    passed = delta >= expected_mib
    unproven = (
        missing_token_ids(
            comparison, (("offloaded", on_out), ("resident", off_out)), min_generated_tokens
        )
        if passed
        else None
    )
    if unproven:
        return _verdict(
            passed=False,
            detail=(
                f"device memory {on_mib:.0f} MiB offloaded vs {off_mib:.0f} MiB resident "
                f"(delta {delta:.0f} MiB >= predicted {expected_mib:.0f} MiB), but {unproven}"
            ),
            command=on_cmd,
            measured=delta,
            reference=expected_mib,
            duration_s=elapsed,
        )
    return _verdict(
        passed=passed,
        detail=(
            f"device memory {on_mib:.0f} MiB offloaded vs {off_mib:.0f} MiB resident "
            f"(delta {delta:.0f} MiB {'>=' if passed else '<'} predicted "
            f"{expected_mib:.0f} MiB), {len(comparison.a)} continuation(s) identical "
            f"(compared as {comparison.observable}){_schedule_note(max_batch_size)}"
        ),
        command=on_cmd,
        measured=delta,
        reference=expected_mib,
        duration_s=elapsed,
    )


def attention_dp_size(
    task_data: Mapping[str, Any], environment: RunEnvironment, member: str | None
) -> int:
    """How many attention-DP ranks one engine at ``member`` has: its width at ``dep``, else 1."""
    kind, declared = parse_parallel_segment(str(member or task_data["parallel"]))
    if kind != "dep":
        return 1
    return int(declared or environment.world_size)


def rank_pairs(ranks: int) -> tuple[tuple[int, int], ...]:
    """Every (context rank, generation rank) pair of two engines of ``ranks`` attention-DP ranks.

    The matched pairs first, then each offset in turn -- ``0:0,1:1,0:1,1:0``
    at two ranks -- so the first hops already put every rank on both sides.
    """
    return tuple((rank, (rank + shift) % ranks) for shift in range(ranks) for rank in range(ranks))


def format_rank_pairs(pairs: Sequence[tuple[int, int]]) -> str:
    """``pairs`` as :data:`LEG_B_RANK_PAIRS_ENV` spells them: ``c:g,c:g,...``."""
    return ",".join(f"{ctx}:{gen}" for ctx, gen in pairs)


def parse_witness_ranks(stdout: str) -> dict[int, tuple[int, int]]:
    """Prompt index -> the (ctx, gen) ranks leg B reported pinning its hop to."""
    return {int(index): (int(ctx), int(gen)) for index, ctx, gen in _WITNESS_RANKS.findall(stdout)}


def parse_witness_ablations(stdout: str) -> tuple[str, ...]:
    """What leg B reported destroying under :data:`LEG_B_ABLATE_ENV`, each once, in order."""
    return tuple(dict.fromkeys(match.strip() for match in _WITNESS_ABLATED.findall(stdout)))


def unpinned_hops(stdout: str, hops: Sequence[tuple[int, int]]) -> str:
    """Why leg B's report of its hops' ranks does not show ``hops``, or ``""``."""
    reported = parse_witness_ranks(stdout)
    missing = [index for index in range(len(hops)) if index not in reported]
    problems = [f"no `[i] Witness ranks:` line for continuation(s) {missing}"] if missing else []
    problems += [
        f"#{index} reported ctx={reported[index][0]} gen={reported[index][1]} for {ctx}:{gen}"
        for index, (ctx, gen) in enumerate(hops)
        if index in reported and reported[index] != (ctx, gen)
    ]
    if not problems:
        return ""
    return (
        f"leg B did not report pinning each hop to its attention-DP rank pair "
        f"({LEG_B_RANK_PAIRS_ENV}={format_rank_pairs(hops)}): {'; '.join(problems)}. Unpinned, "
        f"the router breaks ties by rank, so a leg that sends one request at a time crosses "
        f"every hop from context rank 0 to generation rank 0 and no other rank is measured. "
        f"Leg B pins prompt i's context-only and generation-only requests to the i-th pair "
        f"and prints `[i] Witness ranks: ctx=<c> gen=<g>`"
    )


def _judge_ablation(
    reference: Sequence[int],
    output: str,
    ok: bool,
    *,
    index: int,
    words: int,
    identical: str,
) -> tuple[bool, str]:
    """The self-test's verdict on its leg, and the detail that carries it.

    ``reference`` is leg A's token ids for continuation ``index``, the one
    prompt the self-test sent; ``identical`` is what the main comparison found.
    """
    leg = f"the self-test leg ({LEG_B_ABLATE_ENV}={ABLATE_STATE}, continuation #{index} alone)"
    if not ok:
        return False, (
            f"{identical}, but {leg} did not complete, so nothing shows the witness would see "
            f"a hop that loses state, and it does not pass"
        )
    destroyed = parse_witness_ablations(output)
    if not destroyed:
        return False, (
            f"{identical}, but the self-test could not run: {leg} printed no `Witness ablated:` "
            f"line, so nothing shows the witness would see a hop that loses state, and it does "
            f"not pass. Under {LEG_B_ABLATE_ENV}={ABLATE_STATE} leg B destroys, on the generation "
            f"engine, the state each generation-only request received, before its first decode "
            f"step -- every layer's recurrent state, else the request's KV pages -- and prints "
            f"`Witness ablated: <what>` for each request"
        )
    printed = parse_witness_token_ids(output)
    if len(printed) != 1:
        return False, (
            f"{identical}, but {leg} printed {len(printed)} `Witness token ids:` line(s) for its "
            f"one prompt, so its continuation cannot be compared"
        )
    what = ", ".join(destroyed)
    if tuple(printed[0]) == tuple(reference):
        return False, (
            f"INSENSITIVE, not a pass: {identical}, but with the received state destroyed "
            f"({what}) continuation #{index} ({words} words) was still token-identical to the "
            f"aggregated leg's, so this workload would pass a hop that loses state. A "
            f"continuation the attention pages alone predict -- a repeated pattern, a memorized "
            f"text -- does not read the recurrent state. Declare prompts the model has to read "
            f"to continue (prose in `prompts` / `prompts_file`, or `prompt_tokens`, which "
            f"generates prose), and check that leg B's ablation reaches the generation engine's "
            f"received state"
        )
    at = first_difference(reference, printed[0])
    return True, (
        f"; with the received state destroyed ({what}) continuation #{index} ({words} words) "
        f"first differed at token {at}, so the witness sees a hop that loses state"
    )


def run_equivalence_witness(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    contrast_env: Mapping[str, str],
    contrast_command: str,
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int = 256,
    member: str | None = None,
    prompt_tokens: Sequence[int] | None = None,
    workdir: str | Path | None = None,
    max_batch_size: int | None = None,
    legs_env: Mapping[str, str] | None = None,
    prompts: Sequence[str] | None = None,
    min_generated_tokens: int | None = None,
    transceiver: Mapping[str, Any] | None = None,
) -> GateResult:
    """Prove per-sequence state survives the prefill/decode hop: **token identity** across it.

    Leg A is one aggregated engine; leg B, ``contrast_command`` run as a
    launcher (:func:`contrast_leg_command`) under ``contrast_env``, runs the
    same workload disaggregated, handed leg A's through the ``LEG_B_*_ENV``
    variables. ``prompt_tokens`` appends one generated prose prompt per entry
    (:func:`~.witness_text.natural_text`). Before a pass, leg B runs once more
    on the longest prompt with :data:`LEG_B_ABLATE_ENV` set, and that
    continuation must DIFFER from leg A's, or the verdict is INSENSITIVE. At a
    ``dep`` member leg B pins each hop to an attention-DP rank pair. ``legs_env``
    is the environment both legs run under, where ``contrast_env`` is leg B's alone.
    """
    _verdict = partial(GateResult, name="equivalence", member=member)
    spec = spec or BootSpec()
    prompts = (
        *(spec.prompts if prompts is None else prompts),
        *(natural_text(int(n)) for n in (prompt_tokens or ())),
    )
    pairs = rank_pairs(attention_dp_size(task_data, environment, member))
    if len(pairs) > 1 and prompts and len(prompts) < len(pairs):
        prompts = tuple(prompts[i % len(prompts)] for i in range(len(pairs)))
    hops = tuple(pairs[i % len(pairs)] for i in range(len(prompts))) if len(pairs) > 1 else ()
    shape = dict(
        unit=unit,
        features=features,
        max_tokens=max_tokens,
        member=member,
        max_batch_size=max_batch_size,
    )
    argv = witness_command(task_data, environment, spec=spec, prompts=prompts, **shape)
    full_output = {WITNESS_OUTPUT_ENV: "1"}
    pins = _leg_pins(workdir, "equivalence", task_data, member)
    # Leg B's launcher re-applies `legs_env` to each engine it starts, so a
    # pinned variable carries the pinned value there too.
    shared = {str(k): pins.get(str(k), str(v)) for k, v in (legs_env or {}).items()}
    started = time.monotonic()

    agg_cmd, agg_out, agg_ok = _run_leg(
        environment, argv, runner, timeout, env={**shared, **full_output, **pins}
    )
    if not agg_ok:
        return _verdict(
            passed=False,
            detail="the aggregated leg did not complete",
            command=agg_cmd,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(agg_out.splitlines()[-40:]),
        )

    directory = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="witness-"))
    directory.mkdir(parents=True, exist_ok=True)

    def run_leg_b(
        leg_prompts: Sequence[str],
        leg_hops: Sequence[tuple[int, int]],
        extra: Mapping[str, str],
        prompts_name: str,
    ) -> tuple[str, str, bool]:
        """Leg B over ``leg_prompts``, its hops pinned to ``leg_hops``, under ``extra``."""
        prompts_file = directory / prompts_name
        prompts_file.write_text(json.dumps({"prompts": list(leg_prompts)}), encoding="utf-8")
        leg_a = {
            # Absolute: leg B is a launcher of its own and need not run from
            # the directory the orchestrator resolved the workspace against.
            LEG_B_PROMPTS_FILE_ENV: str(prompts_file.resolve()),
            LEG_B_MAX_TOKENS_ENV: str(max_tokens),
            LEG_B_ARGV_ENV: " ".join(_quickstart_argv(task_data, environment, **shape)),
            LEG_B_TIMEOUT_ENV: str(timeout),
            LEG_B_ENGINE_LAUNCH_ENV: engine_launch(environment),
        }
        if transceiver:
            leg_a[LEG_B_TRANSCEIVER_ENV] = json.dumps(dict(transceiver), sort_keys=True)
        if member:
            leg_a[LEG_B_MEMBER_ENV] = member
        if max_batch_size is not None:
            leg_a[LEG_B_MAX_BATCH_SIZE_ENV] = str(int(max_batch_size))
        if shared:
            leg_a[LEG_B_LEGS_ENV_ENV] = json.dumps(shared, sort_keys=True)
        if speculation_flags(task_data.get("features") if features is None else features):
            leg_a[LEG_B_DRAFT_LEN_ENV] = str(
                resolve_max_draft_len(task_data.get("accuracy") or {}, unit)
            )
        if leg_hops:
            leg_a[LEG_B_RANK_PAIRS_ENV] = format_rank_pairs(leg_hops)
        contrast = replace(
            environment,
            env={
                **environment.env,
                **shared,
                **contrast_env,
                **full_output,
                **leg_a,
                **extra,
                **pins,
            },
        )
        command = contrast_leg_command(contrast, contrast_command)
        result = runner(command, None, timeout)
        return command, result.stdout + result.stderr, result.ok

    dis_cmd, dis_out, dis_ok = run_leg_b(prompts, hops, {}, "prompts.json")
    elapsed = time.monotonic() - started
    if not dis_ok:
        return _verdict(
            passed=False,
            detail="the disaggregated leg did not complete. A "
            "disagg launch fails by HANGING more often than by raising -- check "
            "the re-executed interpreter and the cache transceiver before the target",
            command=dis_cmd,
            duration_s=elapsed,
            stdout_tail="\n".join(dis_out.splitlines()[-40:]),
        )

    comparison = compare_continuations(agg_out, dis_out)
    if not comparison.a or not comparison.b:
        return _verdict(
            passed=False,
            detail=f"no continuations parsed from both legs "
            f"(aggregated={len(comparison.a)}, disaggregated={len(comparison.b)})",
            command=agg_cmd,
            duration_s=elapsed,
        )

    short = too_short(comparison, min_generated_tokens)
    if short:
        return _verdict(
            passed=False,
            detail=short,
            command=agg_cmd,
            duration_s=elapsed,
        )
    compared = f"compared as {comparison.observable}"
    if comparison.observable == PREFIX_OBSERVABLE:
        compared += (
            f"; a leg printed no whole continuation under {WITNESS_OUTPUT_ENV}=1, so the "
            f"rest of each was not compared"
        )
    names = ("aggregated", "disagg")
    schedule = f"{_schedule_note(max_batch_size)}{_legs_note(shared)}"
    if not comparison.identical:
        return _verdict(
            passed=False,
            detail=(
                f"continuation(s) {list(comparison.mismatched) or 'count'} differ across the "
                f"hop ({compared}{schedule}): {comparison.describe_differences(names)}"
                f"{_around(comparison, names)}. State that did not travel is recomputed from "
                f"zero, and the model recovers into fluent, wrong text -- it shows within the "
                f"first few generated tokens. A first difference far into a continuation is a "
                f"near-tie under two speculation schedules: check logit margins before the "
                f"transfer"
            ),
            command=agg_cmd,
            measured=(
                float(min(comparison.first_differences().values()))
                if comparison.mismatched
                else None
            ),
            duration_s=elapsed,
            stdout_tail=comparison.excerpt(names),
        )
    unproven = missing_token_ids(
        comparison, ((names[0], agg_out), (names[1], dis_out)), min_generated_tokens
    )
    if unproven:
        return _verdict(
            passed=False,
            detail=unproven,
            command=agg_cmd,
            duration_s=elapsed,
        )
    identical = (
        f"{len(comparison.a)} continuation(s) token-identical across the hop ({compared}){schedule}"
    )
    if hops:
        unpinned = unpinned_hops(dis_out, hops)
        if unpinned:
            return _verdict(
                passed=False,
                detail=f"{identical}, but {unpinned}",
                command=dis_cmd,
                duration_s=elapsed,
            )
        identical += (
            f"; every attention-DP rank pair crossed ({len(pairs)} pairs over {len(hops)} hops)"
        )

    # The longest prompt: where a counting pattern hid lost state, prose must show it.
    index = max(range(min(len(prompts), len(comparison.a))), key=lambda i: len(prompts[i].split()))
    abl_cmd, abl_out, abl_ok = run_leg_b(
        (prompts[index],),
        hops[index : index + 1],
        {LEG_B_ABLATE_ENV: ABLATE_STATE},
        "prompts-ablated.json",
    )
    elapsed = time.monotonic() - started
    sensitive, verdict = _judge_ablation(
        comparison.a[index],
        abl_out,
        abl_ok,
        index=index,
        words=len(prompts[index].split()),
        identical=identical,
    )
    if not sensitive:
        return _verdict(
            passed=False,
            detail=verdict,
            command=abl_cmd,
            duration_s=elapsed,
            stdout_tail="\n".join(abl_out.splitlines()[-40:]),
        )
    return _verdict(
        passed=True,
        detail=f"{identical}{verdict}",
        command=agg_cmd,
        duration_s=elapsed,
    )


def _judge_decoder_replay(
    on: list[dict[str, int]],
    off: list[dict[str, int]],
    *,
    window: int,
    prompt_tokens: int,
) -> tuple[bool, str]:
    """Replay ON bounds every request's decoder to ``window`` rows; OFF bounds none.

    The workload check comes first: a prompt that never ran cold at full
    length satisfies "decoder rows <= window" with the feature on or off.
    """
    for leg, rows in (("ON", on), ("OFF", off)):
        longest = max(row["forwarded_tokens"] for row in rows)
        if longest < prompt_tokens:
            return False, (
                f"replay {leg}: the longest request forwarded {longest} rows, below the "
                f"declared prompt_tokens={prompt_tokens}, so the long prompt never ran "
                f"cold at full length and a bounded decoder cannot be told from a short "
                f"prompt"
            )
    over = [row for row in on if row["decoder_tokens"] > window]
    if over:
        worst = max(row["decoder_tokens"] for row in over)
        return False, (
            f"replay ON: {len(over)} request(s) ran the decoder over more than "
            f"window={window} rows (up to {worst}), so the decoder was not bounded"
        )
    unbounded = [row for row in off if row["decoder_tokens"] != row["forwarded_tokens"]]
    if unbounded:
        first = unbounded[0]
        return False, (
            f"replay OFF: {len(unbounded)} request(s) ran the decoder over "
            f"{first['decoder_tokens']} rows of {first['forwarded_tokens']} forwarded. "
            f"`contrast_env` did not turn the feature off, or the counter does not "
            f"count what it names"
        )
    return True, (
        f"replay ON: decoder <= {window} rows for each of {len(on)} request(s), the "
        f"longest having forwarded {max(r['forwarded_tokens'] for r in on)}; replay "
        f"OFF: decoder == forwarded for all {len(off)}"
    )


def _judge_encoder_replay(
    on: list[dict[str, int]],
    off: list[dict[str, int]],
    *,
    window: int,
    prefix_tokens: int,
    reuse_alignment: int,
) -> tuple[bool, str]:
    """A prefix hit replays exactly ``min(window, reused)`` rows ON, and none OFF.

    The accounting identity ``forwarded == prompt - reused + replayed`` is
    checked on both legs: a target that reports a hit and then recomputes the
    whole prefix anyway breaks it.
    """
    longest = max(row["prompt_tokens"] for row in on)
    if longest < prefix_tokens:
        return False, (
            f"replay ON: the longest prompt was {longest} tokens, below the declared "
            f"prefix_tokens={prefix_tokens}, so the shared prefix never reached the target"
        )
    for leg, rows in (("ON", on), ("OFF", off)):
        broken = [
            row
            for row in rows
            if row["forwarded_tokens"]
            != row["prompt_tokens"] - row["reused_tokens"] + row["replayed_tokens"]
        ]
        if broken:
            return False, (
                f"replay {leg}: forwarded != prompt - reused + replayed for {len(broken)} "
                f"request(s), first {broken[0]}. A cached prefix is being recomputed, or "
                f"the counter does not count what it names"
            )
    replayed_off = [row for row in off if row["replayed_tokens"] != 0]
    if replayed_off:
        return False, (
            f"replay OFF: {len(replayed_off)} request(s) replayed rows "
            f"(first {replayed_off[0]['replayed_tokens']}); `contrast_env` did not turn the "
            f"feature off"
        )
    wrong = [row for row in on if row["replayed_tokens"] != min(window, row["reused_tokens"])]
    if wrong:
        first = wrong[0]
        return False, (
            f"replay ON: {len(wrong)} request(s) replayed {first['replayed_tokens']} rows "
            f"after reusing {first['reused_tokens']}; a bounded replay re-forwards exactly "
            f"min(window={window}, reused)"
        )
    misaligned = [row for row in on if row["reused_tokens"] % reuse_alignment]
    if misaligned:
        return False, (
            f"replay ON: reuse boundary {misaligned[0]['reused_tokens']} is not a multiple of "
            f"{reuse_alignment}; a boundary that splits a compressed group leaves half of "
            f"its global KV nowhere"
        )
    hits = [row for row in on if row["reused_tokens"] > window]
    if not hits:
        most = max(row["reused_tokens"] for row in on)
        return False, (
            f"replay ON: no request reused more than window={window} cached tokens "
            f"(most: {most}), so a bounded replay cannot be told from re-forwarding the "
            f"whole prefix. Block reuse must be on and the prefix must span a KV block "
            f"past the window"
        )
    return True, (
        f"replay ON: {len(hits)} prefix hit(s) reusing up to "
        f"{max(r['reused_tokens'] for r in hits)} tokens each replayed "
        f"min(window={window}, reused) rows and forwarded suffix + replay; replay OFF: "
        f"nothing replayed over {len(off)} request(s)"
    )


def run_replay_witness(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    feature: str,
    contrast_env: Mapping[str, str],
    window: int,
    prompt_tokens: int | None = None,
    prefix_tokens: int | None = None,
    reuse_alignment: int = 1,
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    features: Any | None = None,
    max_tokens: int = 8,
    member: str | None = None,
) -> GateResult:
    """Prove a bounded replay skipped the work it exists to skip.

    Bounded replay is designed to barely move accuracy, so a unit that
    implements nothing passes the accuracy gate too. The target counts the
    rows the decoder ran (``decoder_replay``) or a prefix hit re-forwarded
    (``encoder_replay``) and prints them (:data:`WITNESS_COUNTER_MARKER`);
    they are judged against the declared ``window`` with the feature ON, then
    OFF under ``contrast_env``, which shows the switch works and the counter
    counts.
    """
    _verdict = partial(GateResult, name=f"replay[{feature}]", member=member)
    started = time.monotonic()
    prompts: tuple[str, ...]
    if feature == "encoder_replay":
        if prefix_tokens is None:
            raise ValueError("the encoder_replay witness needs prefix_tokens")
        # With block reuse off no request can hit the prefix cache.
        if not resolve_runtime(task_data, unit).get("kv_cache_reuse", True):
            return _verdict(
                passed=False,
                detail=f"the `{unit}` unit's runtime turns kv_cache_reuse off; encoder replay "
                f"only happens on a prefix hit, so there is nothing to witness",
            )
        prefix = filler_words(prefix_tokens)
        # Two requests sharing the prefix. Quickstart runs every prompt twice
        # (a batched pass, then a streamed one), so the streamed pass of each
        # hits the prefix the batched pass left in the cache.
        prompts = (f"{prefix} What comes next", f"{prefix} Say it backwards")
    elif feature == "decoder_replay":
        if prompt_tokens is None:
            raise ValueError("the decoder_replay witness needs prompt_tokens")
        prompts = (filler_words(prompt_tokens),)
    else:
        raise ValueError(f"no replay witness for feature {feature!r}")

    argv = witness_command(
        task_data,
        environment,
        unit=unit,
        features=features,
        max_tokens=max_tokens,
        member=member,
        prompts=prompts,
    )
    env = _with_mode(environment, MODE_REQUIRE)
    counting = {WITNESS_COUNTER_ENV: "1"}
    legs: dict[str, list[dict[str, int]]] = {}
    on_cmd = ""
    for leg, overlay in (("ON", counting), ("OFF", {**contrast_env, **counting})):
        command, output, ok = _run_leg(env, argv, runner, timeout, env=overlay)
        on_cmd = on_cmd or command
        if not ok:
            return _verdict(
                passed=False,
                detail=f"the replay-{leg} leg did not complete"
                + (f" ({dict(contrast_env)})" if leg == "OFF" else ""),
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(output.splitlines()[-40:]),
            )
        needed = REPLAY_COUNTER_FIELDS[feature]
        # Only the lines carrying this witness's fields: a target that also
        # implements `kv_quant` prints its storage line under the same marker.
        rows = [row for row in parse_witness_counters(output) if any(key in row for key in needed)]
        if not rows:
            return _verdict(
                passed=False,
                detail=f"the replay-{leg} leg ran but the target printed no witness counter: "
                f"no `{WITNESS_COUNTER_MARKER} key=<int> ...` line carrying {list(needed)} "
                f"with {WITNESS_COUNTER_ENV}=1 set. A witness with no number is not a pass",
                command=command,
                duration_s=time.monotonic() - started,
                stdout_tail="\n".join(output.splitlines()[-40:]),
            )
        incomplete = [row for row in rows if any(key not in row for key in needed)]
        if incomplete:
            missing = sorted({key for row in incomplete for key in needed if key not in row})
            return _verdict(
                passed=False,
                detail=f"the replay-{leg} leg printed {len(incomplete)} counter line(s) "
                f"without {missing}; the {feature} witness needs {list(needed)}",
                command=command,
                duration_s=time.monotonic() - started,
            )
        legs[leg] = rows

    if feature == "decoder_replay":
        passed, detail = _judge_decoder_replay(
            legs["ON"], legs["OFF"], window=window, prompt_tokens=int(prompt_tokens or 0)
        )
    else:
        passed, detail = _judge_encoder_replay(
            legs["ON"],
            legs["OFF"],
            window=window,
            prefix_tokens=int(prefix_tokens or 0),
            reuse_alignment=max(1, int(reuse_alignment)),
        )
    return _verdict(
        passed=passed,
        detail=detail,
        command=on_cmd,
        duration_s=time.monotonic() - started,
    )


def run_kv_quant_witness(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    expected_bytes_per_token: int,
    exact: bool = True,
    contrast_env: Mapping[str, str] | None = None,
    bytes_tolerance: float = DEFAULT_BYTES_TOLERANCE,
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int = 256,
    member: str | None = None,
    max_batch_size: int | None = None,
    prompts: Sequence[str] | None = None,
    min_generated_tokens: int | None = None,
    workdir: str | Path | None = None,
) -> GateResult:
    """Prove KV is stored natively, and -- when declared exact -- that nothing moved.

    The bytes per token the target prints (:data:`KV_BYTES_FIELD`) must match
    the declared figure. With ``exact`` (the default) greedy output under
    ``contrast_env`` (bf16 storage) must match token for token, and that leg
    must report MORE bytes per token; ``exact: false`` drops the identity leg.
    """
    _verdict = partial(GateResult, name="kv_quant", member=member)
    started = time.monotonic()
    argv = witness_command(
        task_data,
        environment,
        unit=unit,
        spec=spec,
        features=features,
        max_tokens=max_tokens,
        member=member,
        max_batch_size=max_batch_size,
        prompts=prompts,
    )
    env = _with_mode(environment, MODE_REQUIRE)
    counting = {WITNESS_COUNTER_ENV: "1", WITNESS_OUTPUT_ENV: "1"}
    # Only an identity comparison has a second leg to share tuning with.
    pins = _leg_pins(workdir, "kv_quant", task_data, member) if exact else {}

    def bytes_readings(output: str) -> list[int]:
        rows = parse_witness_counters(output)
        return [row[KV_BYTES_FIELD] for row in rows if KV_BYTES_FIELD in row]

    native_cmd, native_out, native_ok = _run_leg(
        env, argv, runner, timeout, env={**counting, **pins}
    )
    if not native_ok:
        return _verdict(
            passed=False,
            detail="the native-storage leg did not complete",
            command=native_cmd,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(native_out.splitlines()[-40:]),
        )
    native = bytes_readings(native_out)
    if not native:
        return _verdict(
            passed=False,
            detail=f"the native-storage leg ran but the target printed no witness counter: "
            f"no `{WITNESS_COUNTER_MARKER} {KV_BYTES_FIELD}=<int>` line with "
            f"{WITNESS_COUNTER_ENV}=1 set. A witness with no number is not a pass",
            command=native_cmd,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(native_out.splitlines()[-40:]),
        )
    slack = bytes_tolerance * expected_bytes_per_token
    outside = [b for b in native if abs(b - expected_bytes_per_token) > slack]
    if outside:
        return _verdict(
            passed=False,
            detail=f"native storage reports {outside[0]} B/token against the declared "
            f"{expected_bytes_per_token} (tolerance {bytes_tolerance:.0%}). The pools do not "
            f"hold the checkpoint's formats -- still bf16, or a different layout",
            command=native_cmd,
            duration_s=time.monotonic() - started,
        )
    stored = f"{max(native)} B/token (declared {expected_bytes_per_token})"
    if not exact:
        return _verdict(
            passed=True,
            detail=f"native storage at {stored}. THE WEAKER CLAIM: `exact: false` declares "
            f"a lossy attention path, so output was NOT compared with bf16 storage and "
            f"accuracy is the only output check",
            command=native_cmd,
            duration_s=time.monotonic() - started,
        )

    contrast = {**dict(contrast_env or {}), **counting, **pins}
    ref_cmd, ref_out, ref_ok = _run_leg(env, argv, runner, timeout, env=contrast)
    elapsed = time.monotonic() - started
    if not ref_ok:
        return _verdict(
            passed=False,
            detail=f"the bf16-storage leg ({dict(contrast_env or {})}) did not complete",
            command=ref_cmd,
            duration_s=elapsed,
            stdout_tail="\n".join(ref_out.splitlines()[-40:]),
        )
    reference = bytes_readings(ref_out)
    if not reference or min(reference) <= max(native):
        return _verdict(
            passed=False,
            detail=f"the bf16-storage leg reported {reference or 'no'} B/token against "
            f"native {max(native)}: `contrast_env` did not select bf16 storage, so token "
            f"identity would compare native storage with itself",
            command=ref_cmd,
            duration_s=elapsed,
        )
    comparison = compare_continuations(native_out, ref_out)
    short = too_short(comparison, min_generated_tokens)
    if short:
        return _verdict(
            passed=False,
            detail=short,
            command=native_cmd,
            duration_s=elapsed,
        )
    if not comparison.identical:
        names = ("native", "bf16")
        return _verdict(
            passed=False,
            detail=f"native storage changed the output (continuation(s) "
            f"{list(comparison.mismatched) or 'count'} differ from bf16 storage"
            f"{_token_difference(comparison, names)}{_schedule_note(max_batch_size)}). "
            f"Storing the reference's value set is output-identical by construction, so the "
            f"stored values are not the reference's -- a layout, scale or rounding that is "
            f"another format's",
            command=native_cmd,
            measured=(
                float(min(comparison.first_differences().values()))
                if comparison.mismatched
                else None
            ),
            duration_s=elapsed,
            stdout_tail=(
                comparison.excerpt(names)
                if comparison.a and comparison.b
                else f"native: {comparison.a[:2]}\nbf16:   {comparison.b[:2]}"
            ),
        )
    unproven = missing_token_ids(
        comparison, (("native", native_out), ("bf16", ref_out)), min_generated_tokens
    )
    if unproven:
        return _verdict(
            passed=False,
            detail=f"native storage at {stored} against {min(reference)} B/token in bf16, "
            f"but {unproven}",
            command=native_cmd,
            duration_s=elapsed,
        )
    return _verdict(
        passed=True,
        detail=f"native storage at {stored} against {min(reference)} B/token in bf16; "
        f"{len(comparison.a)} continuation(s) token-identical (compared as "
        f"{comparison.observable}){_schedule_note(max_batch_size)}",
        command=native_cmd,
        duration_s=elapsed,
    )


#: What each kind of witness calls the legs the shim runs, in the order it runs
#: them. An equivalence witness runs only the first through the shim.
WITNESS_LEG_NAMES: dict[str, tuple[str, ...]] = {
    "residency": ("offloaded", "resident"),
    "equivalence": ("aggregated", "disaggregated", "ablated"),
    "replay": ("replay-ON", "replay-OFF"),
    "kv_quant": ("native-storage", "bf16-storage"),
}

#: The witness each output-identical (or work-skipping) feature needs.
WITNESS_KINDS: dict[str, str] = {
    "host_offload": "residency",
    "disagg": "equivalence",
    "decoder_replay": "replay",
    "encoder_replay": "replay",
    "kv_quant": "kv_quant",
}


def run_declared_witness(
    environment: RunEnvironment,
    *,
    unit: str,
    member: str | None,
    task_data: Mapping[str, Any],
    runner: ShellRunner = _default_shell_runner,
    features: Any | None = None,
    workdir: str | Path | None = None,
    boot: BootSpec | None = None,
    kind: str | None = None,
) -> GateResult:
    """Run the witness ``accuracy.witness.<unit>`` declares, for the unit loop and the gate CLI.

    The block must already be complete (the orchestrator's
    ``_witness_block_gap``). A witness that passes fails after all when a
    leg's engine turned block reuse off, reused nothing although it had a
    prefix to reuse, or ran anything other than its own argv declares: each
    leg the shim ran is attested, leg B of an equivalence witness (a launcher)
    is not. Every verdict names the tree it measured (:func:`~.gates.inspect_tree`).
    """
    legs: list[tuple[str, str]] = []
    tree = inspect_tree(task_data)

    def recording(command: str, env: Mapping[str, str] | None = None, timeout: int = 0):
        result = runner(command, env, timeout)
        legs.append((command, f"{result.stdout}{result.stderr}"))
        return result

    def stamped(result: GateResult) -> GateResult:
        if result.tree_state != UNKNOWN_TREE_STATE:
            return result
        return replace(result, tree_state=tree.description, dirty_override=tree.override)

    kind = kind or WITNESS_KINDS[unit]
    result = _run_declared_witness(
        environment,
        unit=unit,
        member=member,
        task_data=task_data,
        runner=recording,
        features=features,
        workdir=workdir,
        boot=boot,
        kind=kind,
    )
    # The replay and kv_quant legs select the target themselves; the others
    # run under the environment's own selection.
    mode = MODE_REQUIRE if kind in ("replay", "kv_quant") else environment.env.get(MODELING_V2_ENV)
    names = WITNESS_LEG_NAMES.get(kind, ())
    shim_legs = [output for command, output in legs if str(QUICKSTART_SHIM) in command]
    attested = []
    for number, output in enumerate(shim_legs):
        records = parse_engine_records(output)
        argv = [str(a) for a in (records[-1].get("argv") or ())] if records else []
        attested.append(
            attest_engine(
                output,
                engine_declaration(task_data, unit, argv, mode=mode),
                leg=names[number] if number < len(names) else f"#{number + 1}",
            )
        )
    engine = engine_fields(attested, task_data, unit)
    result = stamped(
        replace(result, engine=engine["engine"], warnings=(*result.warnings, *engine["warnings"]))
    )
    if not result.passed:
        return result
    options = resolve_llm_api_options(task_data.get("accuracy") or {}, unit)
    for _, output in legs:
        reuse_off = reuse_disabled_detail(output, options=options)
        if reuse_off:
            return replace(result, passed=False, detail=f"a witness leg: {reuse_off}")
    for leg in attested:
        shortfall = reuse_shortfall(leg, asks=bool(leg.declared.get("block_reuse")))
        if shortfall:
            return replace(result, passed=False, detail=f"a witness leg: {shortfall}")
    for leg in attested:
        if leg.mismatches:
            return replace(result, passed=False, detail=leg.difference())
    return result


def _run_declared_witness(
    environment: RunEnvironment,
    *,
    unit: str,
    member: str | None,
    task_data: Mapping[str, Any],
    runner: ShellRunner,
    features: Any | None,
    workdir: str | Path | None,
    boot: BootSpec | None,
    kind: str,
) -> GateResult:
    """:func:`run_declared_witness` without the checks on its legs' engines."""
    # Every leg, and leg B's launcher, gets the unit's KV-cache keys the
    # quickstart has no flag for, as the boot gate's engine does.
    environment = with_kv_cache_config(environment, task_data, unit)
    witnesses = (task_data.get("accuracy") or {}).get("witness") or {}
    block = dict(witnesses.get(unit) or {}) if isinstance(witnesses, Mapping) else {}
    contrast_env = {str(k): str(v) for k, v in (block.get("contrast_env") or {}).items()}
    common = dict(
        task_data=task_data,
        runner=runner,
        unit=unit,
        features=features,
        member=member,
        # Per leg. A disagg leg boots two engines in sequence, which the
        # one-engine default is not sized for.
        timeout=int(block.get("timeout") or BOOT_TIMEOUT_S),
    )
    if kind == "replay":
        # The workload is generated from the declared sizes, not from the
        # boot prompts: two short prompts never exceed the window, and a
        # witness whose prompt fits inside the window passes for a target
        # that bounds nothing.
        return run_replay_witness(
            environment,
            feature=unit,
            contrast_env=contrast_env,
            window=int(block["window"]),
            prompt_tokens=block.get("prompt_tokens"),
            prefix_tokens=block.get("prefix_tokens"),
            reuse_alignment=int(block.get("reuse_alignment", 1)),
            **common,
        )
    # The token-comparing witnesses: both legs run at this batch limit (see
    # `task_schema.WITNESS_BATCH_SIZE_KEY` for why 1 is the useful one) over
    # the declared workload.
    batch = block.get(WITNESS_BATCH_SIZE_KEY)
    boot = boot or BootSpec()
    default_tokens = 64 if kind == "residency" else 256
    workload = resolve_witness_workload(task_data, block, boot, default_max_tokens=default_tokens)
    shaped = dict(
        max_batch_size=None if batch is None else int(batch),
        spec=boot,
        prompts=workload.prompts,
        min_generated_tokens=workload.min_generated_tokens,
    )
    if kind == "residency":
        return run_residency_witness(
            environment,
            expected_mib=resolve_expected_mib(block, member),
            contrast_env=contrast_env,
            max_tokens=workload.max_tokens,
            workdir=workdir,
            **shaped,
            **common,
        )
    if kind == "kv_quant":
        return run_kv_quant_witness(
            environment,
            expected_bytes_per_token=int(block["expected_bytes_per_token"]),
            exact=bool(block.get("exact", True)),
            contrast_env=contrast_env,
            bytes_tolerance=float(block.get("bytes_tolerance", DEFAULT_BYTES_TOLERANCE)),
            max_tokens=workload.max_tokens,
            workdir=workdir,
            **shaped,
            **common,
        )
    return run_equivalence_witness(
        environment,
        contrast_env=contrast_env,
        contrast_command=block["contrast_command"],
        transceiver=block.get(WITNESS_TRANSCEIVER_KEY),
        prompt_tokens=block.get("prompt_tokens"),
        workdir=workdir,
        legs_env={str(k): str(v) for k, v in (block.get(WITNESS_LEGS_ENV_KEY) or {}).items()},
        max_tokens=workload.max_tokens,
        **shaped,
        **common,
    )


def resolve_expected_mib(block: Mapping[str, Any], member: str | None) -> float:
    """The residency delta predicted for ``member``: a scalar, or a mapping with ``default``."""
    expected = block.get("expected_mib", 0.0)
    if isinstance(expected, Mapping):
        chosen = expected.get(str(member))
        expected = chosen if chosen is not None else expected.get("default", 0.0)
    return float(expected)


__all__ = [
    "ABLATE_STATE",
    "DEVICE_MEMORY_MARKER",
    "KV_BYTES_FIELD",
    "LAUNCHER_EXPORTS",
    "LEG_B_ARGV_ENV",
    "LEG_B_DRAFT_LEN_ENV",
    "LEG_B_ENGINE_LAUNCH_ENV",
    "LEG_B_LEGS_ENV_ENV",
    "LEG_B_MAX_BATCH_SIZE_ENV",
    "LEG_B_MAX_TOKENS_ENV",
    "LEG_B_MEMBER_ENV",
    "LEG_B_PROMPTS_FILE_ENV",
    "LEG_B_ABLATE_ENV",
    "LEG_B_RANK_PAIRS_ENV",
    "LEG_B_TIMEOUT_ENV",
    "LEG_B_TRANSCEIVER_ENV",
    "PREFIX_OBSERVABLE",
    "REPLAY_COUNTER_FIELDS",
    "WITNESS_COUNTER_ENV",
    "WITNESS_COUNTER_MARKER",
    "WITNESS_KINDS",
    "WITNESS_LEG_NAMES",
    "WITNESS_OUTPUT_ENV",
    "resolve_expected_mib",
    "run_declared_witness",
    "DEFAULT_MIN_GENERATED_TOKENS",
    "WitnessWorkload",
    "resolve_witness_workload",
    "too_short",
    "ContinuationComparison",
    "compare_continuations",
    "contrast_leg_command",
    "engine_launch",
    "attention_dp_size",
    "filler_words",
    "first_difference",
    "format_rank_pairs",
    "natural_text",
    "parse_witness_ablations",
    "parse_witness_ranks",
    "rank_pairs",
    "unpinned_hops",
    "missing_token_ids",
    "parse_device_memory_mib",
    "parse_generated_texts",
    "parse_witness_counters",
    "parse_witness_texts",
    "parse_witness_token_ids",
    "prints_token_ids",
    "run_equivalence_witness",
    "run_kv_quant_witness",
    "run_replay_witness",
    "run_residency_witness",
    "with_max_batch_size",
    "witness_command",
]
