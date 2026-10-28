# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Storage witnesses: offloaded residency and native KV bytes, with token-identity controls."""

from __future__ import annotations

import re
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import DEFAULT_BYTES_TOLERANCE
from .boot import BootSpec
from .check_runtime import (
    BOOT_TIMEOUT_S,
    MODE_REQUIRE,
    ShellRunner,
    default_shell_runner,
    with_mode,
)
from .verdict import GateResult
from .witness_protocol import (
    WITNESS_COUNTER_ENV,
    WITNESS_COUNTER_MARKER,
    WITNESS_OUTPUT_ENV,
    compare_continuations,
    leg_pins,
    missing_token_ids,
    parse_witness_counters,
    run_leg,
    schedule_note,
    token_difference,
    too_short,
    witness_command,
)

#: The residency witness's reading, ``__DEVICE_MEMORY_MIB__ <MiB>``, which the
#: TARGET prints once per rank after its weights load and before the KV cache
#: pool is allocated (``torch.cuda.memory_allocated()`` in MiB): the pool takes
#: what an offload saved.
DEVICE_MEMORY_MARKER = "__DEVICE_MEMORY_MIB__"


_DEVICE_MEMORY = re.compile(rf"{DEVICE_MEMORY_MARKER}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)")


#: What the kv_quant witness reads: the persistent KV bytes one rank stores per
#: token of a request it serves, summed over its pools as ALLOCATED -- a ratio-r
#: compressed pool contributes its entry size over r. Fixed per-request state
#: (the SWA window, compressor carry-over) is not per-token and is excluded.
KV_BYTES_FIELD = "kv_bytes_per_token"


def parse_device_memory_mib(stdout: str) -> float | None:
    """The last device-memory reading a witness leg printed (:data:`DEVICE_MEMORY_MARKER`), in MiB."""
    matches = _DEVICE_MEMORY.findall(stdout)
    return float(matches[-1]) if matches else None


def run_residency_witness(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    expected_mib: float,
    contrast_env: Mapping[str, str],
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    member: str | None = None,
    max_batch_size: int | None = None,
    prompts: Sequence[str] | None = None,
    max_tokens: int = 64,
    min_generated_tokens: int | None = None,
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
    pins = leg_pins(workdir, member)
    started = time.monotonic()

    on_cmd, on_out, on_ok = run_leg(environment, argv, runner, timeout, env={**full_output, **pins})
    if not on_ok:
        return _verdict(
            passed=False,
            detail="the offloaded leg did not complete",
            command=on_cmd,
            duration_s=time.monotonic() - started,
            stdout_tail="\n".join(on_out.splitlines()[-40:]),
        )

    off_cmd, off_out, off_ok = run_leg(
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
            f"{token_difference(comparison, names)}{schedule_note(max_batch_size)}",
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
            f"(compared as {comparison.observable}){schedule_note(max_batch_size)}"
        ),
        command=on_cmd,
        measured=delta,
        reference=expected_mib,
        duration_s=elapsed,
    )


def run_kv_quant_witness(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    expected_bytes_per_token: int,
    workdir: str | Path,
    exact: bool = True,
    contrast_env: Mapping[str, str] | None = None,
    bytes_tolerance: float = DEFAULT_BYTES_TOLERANCE,
    runner: ShellRunner = default_shell_runner,
    timeout: int = BOOT_TIMEOUT_S,
    unit: str | None = None,
    spec: BootSpec | None = None,
    features: Any | None = None,
    max_tokens: int = 256,
    member: str | None = None,
    max_batch_size: int | None = None,
    prompts: Sequence[str] | None = None,
    min_generated_tokens: int | None = None,
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
    env = with_mode(environment, MODE_REQUIRE)
    counting = {WITNESS_COUNTER_ENV: "1", WITNESS_OUTPUT_ENV: "1"}
    # Only an identity comparison has a second leg to share tuning with.
    pins = leg_pins(workdir, member) if exact else {}

    def bytes_readings(output: str) -> list[int]:
        rows = parse_witness_counters(output)
        return [row[KV_BYTES_FIELD] for row in rows if KV_BYTES_FIELD in row]

    native_cmd, native_out, native_ok = run_leg(
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
    ref_cmd, ref_out, ref_ok = run_leg(env, argv, runner, timeout, env=contrast)
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
            f"{token_difference(comparison, names)}{schedule_note(max_batch_size)}). "
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
        f"{comparison.observable}){schedule_note(max_batch_size)}",
        command=native_cmd,
        duration_s=elapsed,
    )


def resolve_expected_mib(block: Mapping[str, Any], member: str | None) -> float:
    """The residency delta predicted for ``member``: a scalar, or a mapping with ``default``."""
    expected = block.get("expected_mib", 0.0)
    if isinstance(expected, Mapping):
        chosen = expected.get(str(member))
        expected = chosen if chosen is not None else expected.get("default", 0.0)
    return float(expected)


__all__ = [
    "DEVICE_MEMORY_MARKER",
    "KV_BYTES_FIELD",
    "parse_device_memory_mib",
    "resolve_expected_mib",
    "run_kv_quant_witness",
    "run_residency_witness",
]
