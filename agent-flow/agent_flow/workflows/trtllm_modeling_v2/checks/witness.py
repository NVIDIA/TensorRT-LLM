# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gates for features the boot and accuracy gates cannot see.

THE PROBLEM THESE SOLVE

Boot and accuracy both ask "does the engine produce the right text". For most
features that is the whole question. For four of them it is not:

* **speculation** -- rejection sampling holds the emitted distribution to the
  target model's, so a miscomputed draft produces correct text more slowly;
* **host offload** -- the same rows are gathered and the same arithmetic runs;
  only where the bytes waited changes;
* **disaggregation** -- state that fails to cross the hop is silently
  recomputed from zero, and the model recovers into fluent, wrong text;
* **native KV storage** -- packed FP4 dequantized exactly is the same value a
  bf16 container held; only the bytes per token change.

In each case both standing gates pass whether or not the feature is doing
anything, so the unit can be declared green for a feature that was never built.
The acceptance gate is the answer for the first; this module is the answer for
the rest.

It is also the answer for a case with the opposite property:

* **bounded replay** (sliding-window replay bounded in a decoder or encoder) is
  *not* output-identical -- it truncates sliding-window state on purpose -- so
  accuracy is a real check of it. But the truncation is designed to "barely
  compromise" accuracy, which makes a unit that implements nothing pass the
  same gate. Only a count of the work skipped tells the two apart.

THE SHAPE THEY SHARE

They are **two-leg comparisons**: run one workload twice under one changed
variable, in the same session, and compare a named observable. A cross-session
comparison carries session variance into a judgement that is supposed to be
exact --- a bit-identical forward scores differently from one session to the
next. Each leg is still a process of its own, so the witnesses that
compare tokens pin the tuning a process does for itself across their legs
(:func:`~.gates.pin_tuning`). The two that read a target's counters also hold
them to a figure the spec DECLARED, because a counter judged only against
itself agrees with whatever the target printed.

What differs is the variable and the observable:

| gate | variable | observable | verdict |
|---|---|---|---|
| acceptance | `TRTLLM_MODELING_V2` | `acceptance_length` | ratio >= floor |
| residency | the offload switch | device memory after load | delta >= predicted |
| equivalence | aggregated vs disaggregated | the generated tokens | identical; not without the state |
| replay | the replay switch | per-request row counters | bounded ON, full OFF |
| kv_quant | native vs bf16 storage | bytes/token + tokens | declared; identical |

WHAT THEY DELIBERATELY DO NOT MEASURE

Speed. A residency witness says the table is not resident and the output is
unchanged; whether the per-step gather is worth having is a perf question and
belongs to tuning. A replay witness says the rows were skipped, not that
prefill got faster. Claiming more than the measurement supports is the failure
this whole package is built against.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import tempfile
import time
from dataclasses import dataclass, replace
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
    BATCH_SIZE_FLAG,
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
    launch_flags,
    parse_engine_records,
    pin_tuning,
    resolve_capture_list,
    resolve_llm_api_options,
    resolve_max_draft_len,
    resolve_runtime,
    reuse_disabled_detail,
    reuse_shortfall,
    runtime_flags,
    speculation_flags,
    topology_flags,
    with_kv_cache_config,
    with_max_batch_size,
)
from .witness_text import NATURAL_TEXT_WORDS, natural_text

#: `[0] Prompt: '...', Generated text: '...'` -- what
#: `examples/llm-api/quickstart_advanced.py` prints per request. The
#: equivalence witness compares these, and nothing else on the line.
_GENERATED = re.compile(r"Generated text:\s*(.+?)\.\.\.\s*$", re.M)

#: Set to ``1`` on both legs of a token-comparing witness. That line carries
#: the first 200 characters of a continuation -- under a quarter of 256 tokens
#: -- so under this variable a leg also prints each one whole, as
#: ``[i] Witness token ids: [...]`` and ``[i] Witness text: '...'``. Only the
#: token ids can carry a pass: a leg that prints none can still show a
#: difference in its text, but agreement there is refused (see
#: :func:`missing_token_ids`).
WITNESS_OUTPUT_ENV = "TRTLLM_WITNESS_FULL_OUTPUT"
_WITNESS_TOKEN_IDS = re.compile(r"Witness token ids:\s*\[([-0-9,\s]*)\]\s*$", re.M)
_WITNESS_TEXT = re.compile(r"Witness text:\s*(.+?)\s*$", re.M)

#: What leg A resolved, exported to an equivalence witness's ``contrast_command``
#: so leg B reads it rather than restating it. A restated workload desynchronises
#: silently at the next edit to either side, and the verdict then reads as a
#: transfer defect. ``LEG_A_ARGV`` is leg A's quickstart argv with the prompts
#: elided -- they are in ``PROMPTS_FILE``, a JSON ``{"prompts": [...]}``, because
#: one environment string is capped at 128 KiB like one argv string.
#: ``DRAFT_LEN`` is set only when leg A speculates. ``MAX_BATCH_SIZE`` is set
#: only when the witness block declares ``max_batch_size``: leg B applies it
#: to every engine it starts, and at ``1`` sends its requests one at a time.
#: ``LEGS_ENV`` is set only when the block declares ``legs_env``: the same
#: variables are already in leg B's own environment, and this JSON object
#: ``{"VAR": "value"}`` names them so leg B can apply them to every engine it
#: starts after any setting of its own, as leg A's process has them.
#: Leg B's environment also carries :data:`~.gates.KV_CACHE_CONFIG_ENV` when
#: the unit's options hand the quickstart KV-cache keys: leg A's engine has
#: them merged into its ``KvCacheConfig``, and leg B applies them to every
#: engine it starts (``engine_kwargs`` in ``checks/harness/engine_driver.py``
#: merges them as leg A's shim does).
#:
#: Two more are what a contrast command must DO, not only read:
#:
#: ``RANK_PAIRS`` is set only when leg A runs attention DP (a ``dep``
#: member): ``c:g,c:g,...``, one entry per prompt of ``PROMPTS_FILE``, in
#: order (see :func:`rank_pairs`). Leg B pins prompt ``i``'s context-only
#: request to attention-DP rank ``c`` of its context engine and its
#: generation-only request to rank ``g`` of its generation engine (strictly:
#: ``SchedulingParams(attention_dp_rank=..., attention_dp_relax=False)``), and
#: prints ``[i] Witness ranks: ctx=<c> gen=<g>`` beside the prompt's other
#: lines. A hop it does not report fails the witness.
#:
#: ``ABLATE`` is set, to :data:`ABLATE_STATE`, only on the self-test run of
#: leg B (see :func:`run_equivalence_witness`), whose ``PROMPTS_FILE`` holds one
#: prompt. Leg B destroys, on the generation engine, the state each
#: generation-only request received over the hop, once its transfer completes
#: and before its first decode step: the recurrent state (conv window and SSM
#: state) of every layer that has one -- else, for a model with none, the
#: request's attention KV pages. It prints ``Witness ablated: <what>`` once
#: per request it did that to, from any process and behind any prefix (an
#: engine tag, a rank). Without that line the self-test cannot run, and the
#: witness does not pass.
#:
#: Two more say how leg B's engines are made, for the flow's own leg B
#: (:data:`DISAGG_LEG_B`) and any launcher that wants them: ``ENGINE_LAUNCH``
#: is the command prefix that starts one engine of the member's width as the
#: run starts engine work (``mpirun ... trtllm-llmapi-launch <python>`` above
#: one rank, the interpreter alone at one), which the engine's script and
#: arguments follow; ``TRANSCEIVER`` is set only when the block declares
#: ``cache_transceiver_config``, as that JSON object.
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
#: The flow's own leg B: the member's engine split into a context and a
#: generation engine through the LLM API, honouring every variable above. The
#: equivalence witness runs it when the block names no ``contrast_command``.
DISAGG_LEG_B = Path(__file__).resolve().parent / "harness" / "disagg_leg_b.py"
_WITNESS_RANKS = re.compile(r"\[(\d+)\] Witness ranks:\s*ctx=(\d+)[,\s]+gen=(\d+)")
_WITNESS_ABLATED = re.compile(r"Witness ablated:\s*(\S.*?)\s*$", re.M)

#: What `trtllm-llmapi-launch` exports to the process it wraps. A contrast
#: command is a launcher that starts engines of its own, and an engine that
#: inherits a spawn-proxy address does not create its own: the context and the
#: generation engine would both dial the outer wrapper's proxy and hang with no
#: error. Unset before the contrast command runs, whatever launched it.
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
#: pool is allocated (``torch.cuda.memory_allocated()`` in MiB). The pool is
#: sized from the memory left, so a reading taken once the engine reports
#: ready -- by the target, or by `nvidia-smi` from outside -- finds what an
#: offload saved already taken by KV, and the two legs read alike.
DEVICE_MEMORY_MARKER = "__DEVICE_MEMORY_MIB__"
_DEVICE_MEMORY = re.compile(rf"{DEVICE_MEMORY_MARKER}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)")

#: Set to ``1`` on BOTH legs of a replay witness. A target prints its counter
#: lines only under it, so an ordinary run's log is untouched and no verdict
#: depends on a logging level.
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
#: Rows, not positions: the verdicts are statements about work done, and a
#: count of positions attended to would be the same number with the feature on
#: and off. A field a witness does not read may be omitted.
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


#: How far either side of a first difference a verdict shows, per unit. The
#: excerpt is for a reader deciding "lost state or a flipped near-tie", which
#: needs the words on both sides of the flip; the headline rides in a one-line
#: verdict and is kept short.
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

    Token ids where both legs printed them, else the full text, else the
    200-character prefix every quickstart run prints. Chosen per pair, not per
    leg: comparing one leg's ids with the other's text is not a comparison.
    The weaker two can show a difference but not identity; a witness asks
    :func:`missing_token_ids` before it passes on one.
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
    two token sequences can detokenize to one string, the 200-character prefix
    is under a quarter of a 256-token continuation, and neither has a length
    to hold to ``min_generated_tokens``. A difference found in either is a
    real one, so a witness judges that first and asks this only before it
    would pass.
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
#: The workload keys a token-comparing witness block may declare.
WORKLOAD_KEYS: tuple[str, ...] = ("prompts", "prompts_file", "max_tokens", "min_generated_tokens")


def too_short(comparison: ContinuationComparison, minimum: int | None) -> str | None:
    """Why ``comparison`` is too short to judge, or ``None``.

    Only token ids carry a length a witness can hold to a minimum. A
    comparison on text alone has none to report here, and cannot pass for
    that reason (see :func:`missing_token_ids`).
    """
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
    """The witness block's own workload, else the declared acceptance prompts, else the boot prompts.

    Declared in task.yaml, never assembled from the run: ``prompts`` or
    ``prompts_file``, ``max_tokens`` and ``min_generated_tokens`` on the
    witness block. With none, the prompts the acceptance gate declares
    (long answers by construction), else the boot prompts.
    """
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
    """The device-memory reading a witness leg printed, in MiB.

    Read from a marker rather than from `nvidia-smi`'s own format because only
    the target knows *when* to sample --- after its weights load and before
    the KV cache pool is allocated (see :data:`DEVICE_MEMORY_MARKER`), a moment
    a gate sampling from outside cannot see.
    """
    matches = _DEVICE_MEMORY.findall(stdout)
    return float(matches[-1]) if matches else None


def parse_witness_counters(stdout: str) -> list[dict[str, int]]:
    """Every counter line a target printed, one mapping per context request.

    A multi-rank run may print a request once per rank. Every verdict below is
    a per-line statement, so a duplicated line is judged twice and changes
    nothing.
    """
    return [
        {key: int(value) for key, value in _COUNTER_FIELD.findall(match.group(1))}
        for match in _COUNTER_LINE.finditer(stdout)
    ]


def filler_words(count: int) -> str:
    """``count`` short words: a prompt of at least ``count`` tokens.

    For a witness that counts work rather than compares tokens (the replay
    witnesses): a counting pattern is continued by copying whatever state
    the engine holds, so a token comparison over it cannot see lost state.
    The equivalence witness generates prose instead (:func:`natural_text`).
    """
    return " ".join(_FILLER[i % len(_FILLER)] for i in range(count))


def witness_command(
    task_data: Mapping[str, Any],
    environment: RunEnvironment,
    *,
    unit: str | None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int = 64,
    extra: tuple[str, ...] = (),
    member: str | None = None,
    prompts: Sequence[str] | None = None,
    max_batch_size: int | None = None,
) -> str:
    """The quickstart argv both legs of a witness run.

    Quickstart rather than `trtllm-eval` for the same reason the acceptance
    gate uses it: the eval harness reports neither device memory nor the
    per-request continuations, and it takes about twice as long a leg.

    `member` names the parallel configuration BOTH legs run at. A witness is
    a contrast between two legs, so the one thing it must not do is let the
    topology differ between them -- and under a `unified` segment with no
    member, `topology_flags` resolves to pure tensor parallelism, which is a
    configuration this target may not even be delivered at.

    The draft depth comes from the unit's options YAML, as it does for the
    boot gate, never a fixed depth: a unit built after `dspark` runs its
    witness with the DSpark flags, and `TorchLlmArgs` rejects a DSpark config
    whose `block_size` (from the checkpoint) differs from `max_draft_len` --
    both legs would die at startup and the witness would report a leg that did
    not complete, which reads as the unit's own defect.

    `prompts` overrides the boot spec's, for a witness whose workload is
    generated from declared sizes rather than written by hand.

    `max_batch_size` replaces whatever batch limit the runtime block set (see
    :func:`with_max_batch_size`).
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
                extra=extra,
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
    extra: tuple[str, ...],
    member: str | None,
    max_batch_size: int | None = None,
) -> list[str]:
    """:func:`witness_command` without its prompts."""
    script = Path(str(task_data["trtllm_repo_path"])) / "examples" / "llm-api"
    return [
        str(script / "quickstart_advanced.py"),
        f'--model_dir="{task_data["checkpoint_path"]}"',
        f"--max_tokens={max_tokens}",
        *topology_flags(str(task_data["parallel"]), environment.world_size, member),
        *with_max_batch_size(
            [
                *(stated := _stated_flags(task_data, unit, features, extra)),
                *launch_flags(task_data, unit, stated),
            ],
            max_batch_size,
        ),
    ]


def _stated_flags(
    task_data: Mapping[str, Any], unit: str | None, features: Any, extra: tuple[str, ...]
) -> list[str]:
    """The legs' runtime, capture, speculation and extra flags, as the unit states them."""
    runtime = resolve_runtime(task_data, unit)
    return [
        *runtime_flags(runtime, capture=resolve_capture_list(task_data, unit, runtime=runtime)),
        *speculation_flags(
            task_data.get("features") if features is None else features,
            max_draft_len=resolve_max_draft_len(task_data.get("accuracy") or {}, unit),
        ),
        *extra,
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
    """How an equivalence witness launches ``contrast_command``: as a launcher.

    Leg B of a disagg witness is two engines, a proxy and a client -- work
    that starts ranks, not a rank. Launched as engine work it would go out as
    ``mpirun -n <world> ... trtllm-llmapi-launch <python> <command>``: the
    command would run on rank 0 while idle workers held the other ranks and the
    devices' memory, and it would inherit the wrapper's spawn-proxy variables,
    which make the engines it starts hang. So it runs as ONE process, the
    branch :meth:`RunEnvironment.command` already has for work that is not
    engine work, with :data:`LAUNCHER_EXPORTS` unset. The process keeps the
    claim's devices: ``CUDA_VISIBLE_DEVICES`` names all of them.

    Under ``slurm`` one task means ``srun --gres=gpu:1``, which would take the
    launcher's devices away; there it keeps the engine-shaped ``srun`` over
    the whole claim and relies on the same unset.

    A command whose first word is a ``.py`` file runs under the run's
    interpreter; anything else -- a shell script -- runs as written.
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

    Two legs of the same workload in the same session, differing only by
    ``contrast_env`` --- the environment that turns the offload OFF. Both must
    produce **token-identical** output, and the resident leg must use at least
    ``expected_mib`` more device memory than the offloaded one.

    ``expected_mib`` is computed from the CHECKPOINT's own shapes and passed in
    by the caller, deliberately: a witness that derives its own expectation
    from the run it is judging can only ever agree with itself.

    The token-identity half is not decoration. Offloading is output-identical
    by construction, so if the two legs disagree the gather is wrong --- and
    the accuracy gate's tolerance is wide enough to hide that. Compared as
    :func:`compare_continuations` does, with both legs printing their whole
    continuations, and passed only on token ids (:func:`missing_token_ids`);
    ``max_batch_size`` runs both legs at that batch limit. The legs share their
    tuning (:func:`~.gates.pin_tuning`) under ``workdir``.

    The device-memory reading is the target's to print (see
    :data:`DEVICE_MEMORY_MARKER`); a leg without one fails.
    """
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
        return GateResult(
            name="residency",
            member=member,
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
        return GateResult(
            name="residency",
            member=member,
            passed=False,
            detail=f"the resident control leg ({dict(contrast_env)}) did not complete",
            command=off_cmd,
            duration_s=elapsed,
            stdout_tail="\n".join(off_out.splitlines()[-40:]),
        )

    on_mib, off_mib = parse_device_memory_mib(on_out), parse_device_memory_mib(off_out)
    if on_mib is None or off_mib is None:
        return GateResult(
            name="residency",
            member=member,
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
        return GateResult(
            name="residency",
            member=member,
            passed=False,
            detail=short,
            command=on_cmd,
            duration_s=elapsed,
        )
    if not comparison.identical:
        names = ("offloaded", "resident")
        return GateResult(
            name="residency",
            member=member,
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
        return GateResult(
            name="residency",
            member=member,
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
    return GateResult(
        name="residency",
        member=member,
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
    prompt the self-test sent; ``identical`` is what the main comparison
    found. A pass returns the clause the verdict appends.
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
    runner: ShellRunner = _default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int = 256,
    contrast_command: str | None = None,
    member: str | None = None,
    prompt_tokens: Sequence[int] | None = None,
    workdir: str | Path | None = None,
    max_batch_size: int | None = None,
    legs_env: Mapping[str, str] | None = None,
    prompts: Sequence[str] | None = None,
    min_generated_tokens: int | None = None,
    transceiver: Mapping[str, Any] | None = None,
) -> GateResult:
    """Prove per-sequence state survives the prefill/decode hop.

    ``contrast_command`` is how leg B is actually launched: the flow's own
    :data:`DISAGG_LEG_B` when the witness block names none (see
    :func:`run_declared_witness`), else the run's launcher; this function
    still accepts None for the one-switch shape. Disaggregation is not an
    environment variable over one process: it is a context engine and a
    generation engine with the state handed between them, which
    `quickstart_advanced.py` cannot launch. An env overlay alone would run
    leg B aggregated and the witness would compare a run to itself ---
    passing trivially, which is the worst possible outcome for a gate whose
    whole job is to catch a silent equality.

    The contract on that command is small: launch the disaggregated topology,
    drive the same prompts greedily, and print the continuations in
    `quickstart_advanced.py`'s format so both legs parse the same way -- and,
    under :data:`WITNESS_OUTPUT_ENV`, each one's token ids and whole text as
    well, without which the witness cannot pass. It runs as a launcher
    (:func:`contrast_leg_command`) and is handed leg A's resolved workload
    through the ``LEG_B_*_ENV`` variables, so it need not restate any of it.
    Two of them ask it to act -- pin its hops to ranks, destroy the state it
    received -- and are described with the variables. ``transceiver`` is the
    block's ``cache_transceiver_config``, handed over as
    :data:`LEG_B_TRANSCEIVER_ENV`.

    ``prompt_tokens`` appends one generated prompt of at least that many
    tokens per entry to the boot prompts, on BOTH legs. The boot prompts stay:
    they are the ones whose answers depend on compressed and long-range state
    (a pass key 1.3K tokens back). The generated ones supply long prompts --
    over one prefill chunk, past a page -- that hand-written prompts rarely
    reach. They are prose (:func:`~.witness_text.natural_text`), not a counting
    pattern: a pattern is continued from the attention pages alone, so a hop
    that loses the recurrent state still reproduces it.

    Leg A is one aggregated engine; leg B is the same workload with
    ``contrast_env`` applied --- here the environment that turns disaggregation
    ON, which is why the key is not named for a direction: the residency
    witness's second leg turns its feature *off* and this one's turns its
    feature *on*. Greedy, same seed, same session. The verdict is
    **token identity**, not similarity: missing recurrent state yields text
    that is fluent and wrong, and a short benchmark under a five-point
    tolerance will not catch it. It compares token ids where both legs print
    them, else the full text, else the 200-character prefix, and names which;
    only token ids can pass (:func:`missing_token_ids`).

    A failing verdict names where each continuation FIRST differs. State lost
    at the hop is read by the generation engine's first decode step, so it
    shows within the first few generated tokens; a first difference far into a
    continuation, with both legs speculating on different schedules, is a
    near-tie flipped by batch-shape-variant numerics -- a logit-margin
    question, not a transfer one.

    Identity is not trusted until the witness shows it could have seen the
    opposite. Before a pass, leg B runs once more on ONE prompt -- the
    longest, the kind a counting pattern left blind to lost state -- with
    :data:`LEG_B_ABLATE_ENV` set to :data:`ABLATE_STATE`: the state the
    generation engine received is destroyed before it decodes. That
    continuation must DIFFER from leg A's. If it does not, the workload cannot
    see lost state and the verdict is INSENSITIVE, not a pass; a leg B that
    reports destroying nothing cannot run the test and does not pass either.
    The one-switch shape runs the same test with the variable in its second
    leg's environment.

    Under attention DP (a ``dep`` member) a leg B that sends one request at
    a time crosses every hop from context rank 0 to generation rank 0: the
    router breaks ties by rank. So at such a member, and only with a
    ``contrast_command``, leg B is handed :data:`LEG_B_RANK_PAIRS_ENV`: one
    (ctx rank, gen rank) pair per prompt, covering every pair of the
    member's ranks (:func:`rank_pairs`). The prompts are repeated, on both
    legs, until there is one per pair. Leg B pins each hop and reports it; a
    hop it does not report as pinned is not a pass. The self-test's hop goes
    through the pair its prompt had.

    ``max_batch_size`` is the batch limit BOTH legs run at: leg A's argv
    carries it in place of the runtime block's, and leg B is handed it as
    :data:`LEG_B_MAX_BATCH_SIZE_ENV`. At 1 each request runs alone on both
    sides of the comparison, so the two legs' step shapes match and a
    difference cannot come from how admission happened to pack rows -- the
    batch-shape-variant numerics above are then out of the picture.

    ``legs_env`` is the environment BOTH legs run under, where ``contrast_env``
    is leg B's alone: leg A's process gets it, and leg B's launcher gets it in
    its own environment and, named, as :data:`LEG_B_LEGS_ENV_ENV` for every
    engine it starts. It pins a code path the disaggregated engines refuse or
    guard off to the same setting in the aggregated leg; otherwise the legs
    differ in that path as well as in the hop, and identity is no longer
    expected. The verdict names it, since a pass holds under it.

    For a hybrid checkpoint the exposure is the recurrent conv/SSM state and
    any sparse-attention indexer state -- not the attention pages everyone
    remembers to check.

    Both legs, and every engine leg B starts, share their tuning
    (:func:`~.gates.pin_tuning`) under ``workdir``; a pinned variable that
    ``legs_env`` also sets takes the pinned value.
    """
    spec = spec or BootSpec()
    prompts = (
        *(spec.prompts if prompts is None else prompts),
        *(natural_text(int(n)) for n in (prompt_tokens or ())),
    )
    # Only a launcher places a hop on a rank: the one-switch leg B is one engine.
    pairs = (
        rank_pairs(attention_dp_size(task_data, environment, member)) if contrast_command else ()
    )
    if len(pairs) > 1 and prompts and len(prompts) < len(pairs):
        prompts = tuple(prompts[i % len(prompts)] for i in range(len(pairs)))
    hops = tuple(pairs[i % len(pairs)] for i in range(len(prompts))) if len(pairs) > 1 else ()
    shape = dict(
        unit=unit,
        features=features,
        max_tokens=max_tokens,
        extra=(),
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
        return GateResult(
            name="equivalence",
            member=member,
            passed=False,
            detail="the aggregated leg did not complete",
            command=agg_cmd,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(agg_out.splitlines()[-40:]),
        )

    directory = None
    if contrast_command:
        directory = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="witness-"))
        directory.mkdir(parents=True, exist_ok=True)

    def run_leg_b(
        leg_prompts: Sequence[str],
        leg_hops: Sequence[tuple[int, int]],
        extra: Mapping[str, str],
        prompts_name: str,
    ) -> tuple[str, str, bool]:
        """Leg B over ``leg_prompts``, its hops pinned to ``leg_hops``, under ``extra``."""
        if directory is None or not contrast_command:
            # The one-switch shape: leg A's argv under `contrast_env`.
            leg_argv = argv
            if tuple(leg_prompts) != tuple(prompts):
                leg_argv = witness_command(
                    task_data, environment, spec=spec, prompts=leg_prompts, **shape
                )
            env = {**shared, **contrast_env, **extra, **full_output, **pins}
            return _run_leg(environment, leg_argv, runner, timeout, env=env)
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
        return GateResult(
            name="equivalence",
            member=member,
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
        return GateResult(
            name="equivalence",
            member=member,
            passed=False,
            detail=f"no continuations parsed from both legs "
            f"(aggregated={len(comparison.a)}, disaggregated={len(comparison.b)})",
            command=agg_cmd,
            duration_s=elapsed,
        )

    short = too_short(comparison, min_generated_tokens)
    if short:
        return GateResult(
            name="equivalence",
            member=member,
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
        return GateResult(
            name="equivalence",
            member=member,
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
        return GateResult(
            name="equivalence",
            member=member,
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
            return GateResult(
                name="equivalence",
                member=member,
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
        return GateResult(
            name="equivalence",
            member=member,
            passed=False,
            detail=verdict,
            command=abl_cmd,
            duration_s=elapsed,
            stdout_tail="\n".join(abl_out.splitlines()[-40:]),
        )
    return GateResult(
        name="equivalence",
        member=member,
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

    The workload check comes first because it is what makes the other two
    mean anything: a prompt that never ran cold at full length satisfies
    "decoder rows <= window" with the feature on or off.
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
    whole prefix anyway -- which is what "SWA KV is missing" invites -- breaks
    it, and nothing else in the verdict would notice.
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

    Bounded replay is NOT output-identical: both halves truncate sliding-window
    state on purpose, so the accuracy gate is a real check of them. It is not a
    sufficient one. The truncation is designed, and post-trained, to barely
    move accuracy -- which means a unit that implements nothing lands inside
    the same tolerance. The observable that differs is work: rows the decoder
    ran (``decoder_replay``), rows a prefix hit re-forwarded
    (``encoder_replay``). Only the target can count those, so the target
    prints them -- see :data:`WITNESS_COUNTER_MARKER` -- and this judges them
    against the declared ``window``.

    Two legs, one session, the same generated workload: the feature ON, then
    OFF under ``contrast_env`` (the leg WITHOUT it). Both legs set
    :data:`WITNESS_COUNTER_ENV`. The OFF leg is not decoration: it is what
    shows the switch works and the counter counts, since a counter that always
    prints ``decoder_tokens=128`` would otherwise pass.

    ``window``, ``prompt_tokens`` and ``prefix_tokens`` come from the spec,
    declared before assembly -- a witness that sized its workload from the run
    it judges could only agree with itself.
    """
    name = f"replay[{feature}]"
    started = time.monotonic()
    prompts: tuple[str, ...]
    if feature == "encoder_replay":
        if prefix_tokens is None:
            raise ValueError("the encoder_replay witness needs prefix_tokens")
        # Checked before any engine starts: with block reuse off no request can
        # hit the prefix cache, so the witness would spend two engine launches
        # to report that nothing was replayed.
        if not resolve_runtime(task_data, unit).get("kv_cache_reuse", True):
            return GateResult(
                name=name,
                member=member,
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
            return GateResult(
                name=name,
                member=member,
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
            return GateResult(
                name=name,
                member=member,
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
            return GateResult(
                name=name,
                member=member,
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
    return GateResult(
        name=name,
        member=member,
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

    Native storage of the SAME value set the reference fake-quantizes is
    output-identical by construction when attention dequantizes exactly, so
    boot and accuracy pass whether the pools hold packed FP4 or the bf16
    containers the stock path uses. Two observables close that:

    * **bytes per token** -- the target prints
      ``MODELING_V2_WITNESS kv_bytes_per_token=<int>`` (see
      :data:`WITNESS_COUNTER_MARKER`), computed from the pools as allocated,
      and it must match the figure the spec declared from the checkpoint's
      formats. bf16 containers take several times the native bytes per
      token, so the check has one way to pass.
    * **token identity against bf16 storage** (``exact``, the default) --
      greedy output under ``contrast_env`` must match token for token. A
      difference means the stored values are not the reference's: a layout,
      scale or rounding that is somebody else's format. The contrast leg must
      report MORE bytes per token than the native one, or ``contrast_env``
      selected nothing and the identity compared native storage with itself.

    ``exact: false`` is for an attention path declared lossy (an FP8-query
    kernel). The identity leg is dropped, accuracy is the only output check,
    and the verdict says it is the weaker claim.

    Identity is judged as :func:`compare_continuations` judges it -- token
    ids, which both legs must print under :data:`WITNESS_OUTPUT_ENV` for a
    pass (:func:`missing_token_ids`) -- and a failing verdict names where each
    continuation first differs and what both legs generated there.
    ``max_batch_size`` runs both legs at that batch limit; at 1 the two legs'
    schedules match, so a difference is storage and not how admission packed
    rows. The two legs share their tuning (:func:`~.gates.pin_tuning`) under
    ``workdir``.
    """
    name = "kv_quant"
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
        return GateResult(
            name=name,
            member=member,
            passed=False,
            detail="the native-storage leg did not complete",
            command=native_cmd,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(native_out.splitlines()[-40:]),
        )
    native = bytes_readings(native_out)
    if not native:
        return GateResult(
            name=name,
            member=member,
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
        return GateResult(
            name=name,
            member=member,
            passed=False,
            detail=f"native storage reports {outside[0]} B/token against the declared "
            f"{expected_bytes_per_token} (tolerance {bytes_tolerance:.0%}). The pools do not "
            f"hold the checkpoint's formats -- still bf16, or a different layout",
            command=native_cmd,
            duration_s=time.monotonic() - started,
        )
    stored = f"{max(native)} B/token (declared {expected_bytes_per_token})"
    if not exact:
        return GateResult(
            name=name,
            member=member,
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
        return GateResult(
            name=name,
            member=member,
            passed=False,
            detail=f"the bf16-storage leg ({dict(contrast_env or {})}) did not complete",
            command=ref_cmd,
            duration_s=elapsed,
            stdout_tail="\n".join(ref_out.splitlines()[-40:]),
        )
    reference = bytes_readings(ref_out)
    if not reference or min(reference) <= max(native):
        return GateResult(
            name=name,
            member=member,
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
        return GateResult(
            name=name,
            member=member,
            passed=False,
            detail=short,
            command=native_cmd,
            duration_s=elapsed,
        )
    if not comparison.identical:
        names = ("native", "bf16")
        return GateResult(
            name=name,
            member=member,
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
        return GateResult(
            name=name,
            member=member,
            passed=False,
            detail=f"native storage at {stored} against {min(reference)} B/token in bf16, "
            f"but {unproven}",
            command=native_cmd,
            duration_s=elapsed,
        )
    return GateResult(
        name=name,
        member=member,
        passed=True,
        detail=f"native storage at {stored} against {min(reference)} B/token in bf16; "
        f"{len(comparison.a)} continuation(s) token-identical (compared as "
        f"{comparison.observable}){_schedule_note(max_batch_size)}",
        command=native_cmd,
        duration_s=elapsed,
    )


#: What each kind of witness calls the legs the shim runs, in the order it runs
#: them. An equivalence witness with a ``contrast_command`` runs only the first
#: through the shim; without one, its self-test leg is the third.
WITNESS_LEG_NAMES: dict[str, tuple[str, ...]] = {
    "residency": ("offloaded", "resident"),
    "equivalence": ("aggregated", "disaggregated", "ablated"),
    "replay": ("replay-ON", "replay-OFF"),
    "kv_quant": ("native-storage", "bf16-storage"),
}

#: The witness each output-identical (or work-skipping) feature needs. The
#: orchestrator's ``WITNESS_GATED_UNITS`` is this table.
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
    """Run the witness ``accuracy.witness.<unit>`` declares, exactly as the orchestrator does.

    One entry point for the unit loop and for a standalone run of the same
    witness (``trtllm-modeling-v2 gate witness``): an assembler that reruns a
    witness by hand reruns this, not a reconstruction of it. The block must
    already be complete (see the orchestrator's ``_witness_block_gap``).

    A witness that passes fails after all when any of its legs' engines
    logged that it turned block reuse off (:func:`reuse_disabled_detail`), as
    the boot gate does: its legs ran without the reuse the unit asks for. So
    it does when a leg the shim ran asks for block reuse and its engine
    reused nothing although it had a prefix to reuse
    (:func:`~.gates.reuse_shortfall`), and when a leg's engine ran anything
    other than its argv and the unit's options declare
    (:func:`~.gates.attest_engine`): each leg the shim ran is held to the
    configuration its own argv states, which its engine record repeats. Leg B
    of an equivalence witness is a launcher, not the shim, and is not
    attested. Every verdict carries the attested legs' records, and a ``LOW
    MARGIN`` warning when one left little memory free.

    Every verdict names the tree it measured: the state of
    ``trtllm_repo_path`` as the witness began (:func:`~.gates.inspect_tree`),
    unless a snapshot stamps its commit over it afterwards -- a standalone
    ``gate witness --tree`` run is measured on the tree as it is.
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
    kind: str | None,
) -> GateResult:
    """:func:`run_declared_witness` without the block-reuse check on its legs."""
    kind = kind or WITNESS_KINDS[unit]
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
        # Without a launcher of the run's own, the flow's.
        contrast_command=block.get("contrast_command") or shlex.quote(str(DISAGG_LEG_B)),
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
    "DISAGG_LEG_B",
    "KV_BYTES_FIELD",
    "LAUNCHER_EXPORTS",
    "LEG_B_ARGV_ENV",
    "BATCH_SIZE_FLAG",
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
    "NATURAL_TEXT_WORDS",
    "PREFIX_OBSERVABLE",
    "REPLAY_COUNTER_FIELDS",
    "WITNESS_COUNTER_ENV",
    "WITNESS_COUNTER_MARKER",
    "WITNESS_KINDS",
    "WITNESS_LEG_NAMES",
    "WITNESS_OUTPUT_ENV",
    "WORKLOAD_KEYS",
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
