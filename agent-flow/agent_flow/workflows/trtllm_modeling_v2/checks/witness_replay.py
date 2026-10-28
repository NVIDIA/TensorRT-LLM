# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay witnesses: per-request row accounting with the feature enabled and disabled."""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Mapping

from ..infra.compute.environment import RunEnvironment
from .check_config import resolve_runtime
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    ShellRunner,
    default_shell_runner,
    with_mode,
)
from .verdict import GateResult
from .witness_inputs import filler_words
from .witness_protocol import (
    WITNESS_COUNTER_ENV,
    WITNESS_COUNTER_MARKER,
    parse_witness_counters,
    run_leg,
    witness_command,
)

#: What each replay witness reads off a counter line.
REPLAY_COUNTER_FIELDS: dict[str, tuple[str, ...]] = {
    "decoder_replay": ("forwarded_tokens", "decoder_tokens"),
    "encoder_replay": ("prompt_tokens", "reused_tokens", "replayed_tokens", "forwarded_tokens"),
}


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
    runner: ShellRunner = default_shell_runner,
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
    env = with_mode(environment, MODE_REQUIRE)
    counting = {WITNESS_COUNTER_ENV: "1"}
    legs: dict[str, list[dict[str, int]]] = {}
    on_cmd = ""
    for leg, overlay in (("ON", counting), ("OFF", {**contrast_env, **counting})):
        command, output, ok = run_leg(env, argv, runner, timeout, env=overlay)
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


__all__ = ["REPLAY_COUNTER_FIELDS", "run_replay_witness"]
