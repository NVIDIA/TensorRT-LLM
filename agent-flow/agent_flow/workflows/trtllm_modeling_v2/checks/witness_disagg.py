# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated state-transfer witness: rank-pinned equivalence and a state-ablation control."""

from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import parse_parallel_segment
from .boot import BootSpec
from .check_config import resolve_max_draft_len, speculation_flags
from .check_runtime import BOOT_TIMEOUT_S, ShellRunner, default_shell_runner, gate_workdir
from .verdict import GateResult
from .witness_inputs import natural_text
from .witness_protocol import (
    LEG_B_ABLATE_ENV,
    LEG_B_ARGV_ENV,
    LEG_B_DRAFT_LEN_ENV,
    LEG_B_ENGINE_LAUNCH_ENV,
    LEG_B_LEGS_ENV_ENV,
    LEG_B_MAX_BATCH_SIZE_ENV,
    LEG_B_MAX_TOKENS_ENV,
    LEG_B_MEMBER_ENV,
    LEG_B_PROMPTS_FILE_ENV,
    LEG_B_RANK_PAIRS_ENV,
    LEG_B_TIMEOUT_ENV,
    LEG_B_TRANSCEIVER_ENV,
    PREFIX_OBSERVABLE,
    WITNESS_OUTPUT_ENV,
    around,
    compare_continuations,
    contrast_leg_command,
    engine_launch,
    first_difference,
    leg_pins,
    legs_note,
    missing_token_ids,
    parse_witness_token_ids,
    request_argv,
    run_leg,
    schedule_note,
    too_short,
    witness_command,
)

#: The one value :data:`LEG_B_ABLATE_ENV` takes.
ABLATE_STATE = "state"


_WITNESS_RANKS = re.compile(r"\[(\d+)\] Witness ranks:\s*ctx=(\d+)[,\s]+gen=(\d+)")


_WITNESS_ABLATED = re.compile(r"Witness ablated:\s*(\S.*?)\s*$", re.M)


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
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int = 256,
    member: str | None = None,
    prompt_tokens: Sequence[int] | None = None,
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
    (:func:`~.witness_inputs.natural_text`). Before a pass, leg B runs once more
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
    pins = leg_pins(workdir, member)
    # Leg B's launcher re-applies `legs_env` to each engine it starts, so a
    # pinned variable carries the pinned value there too.
    shared = {str(k): pins.get(str(k), str(v)) for k, v in (legs_env or {}).items()}
    started = time.monotonic()

    agg_cmd, agg_out, agg_ok = run_leg(
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

    directory = gate_workdir(workdir)

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
            LEG_B_ARGV_ENV: " ".join(request_argv(task_data, environment, **shape)),
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
    schedule = f"{schedule_note(max_batch_size)}{legs_note(shared)}"
    if not comparison.identical:
        return _verdict(
            passed=False,
            detail=(
                f"continuation(s) {list(comparison.mismatched) or 'count'} differ across the "
                f"hop ({compared}{schedule}): {comparison.describe_differences(names)}"
                f"{around(comparison, names)}. State that did not travel is recomputed from "
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

    # The longest prose prompt is the negative-control candidate; only observed
    # divergence demonstrates sensitivity to the ablated state.
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


__all__ = [
    "ABLATE_STATE",
    "attention_dp_size",
    "format_rank_pairs",
    "parse_witness_ablations",
    "parse_witness_ranks",
    "rank_pairs",
    "run_equivalence_witness",
    "unpinned_hops",
]
