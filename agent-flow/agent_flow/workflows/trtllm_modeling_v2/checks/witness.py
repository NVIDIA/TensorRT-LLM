# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared feature witnesses: workload dispatch, measured-tree stamps and shared engine checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..infra.compute.environment import RunEnvironment
from ..spec.task_schema import (
    DEFAULT_BYTES_TOLERANCE,
    WITNESS_BATCH_SIZE_KEY,
    WITNESS_LEGS_ENV_KEY,
    WITNESS_TRANSCEIVER_KEY,
)
from .boot import BootSpec
from .check_config import engine_declaration, resolve_llm_api_options
from .check_runtime import (
    BOOT_TIMEOUT_S,
    ENGINE_DRIVER,
    MODE_REQUIRE,
    MODELING_V2_ENV,
    ShellRunner,
    default_shell_runner,
)
from .runtime_evidence import (
    attest_engine,
    engine_records,
    parse_engine_records,
    reuse_disabled_detail,
    reuse_shortfall,
)
from .verdict import UNKNOWN_TREE_STATE, GateResult, inspect_tree
from .witness_disagg import run_equivalence_witness
from .witness_inputs import resolve_witness_workload
from .witness_replay import run_replay_witness
from .witness_storage import resolve_expected_mib, run_kv_quant_witness, run_residency_witness

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
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    features: Any | None = None,
    boot: BootSpec | None = None,
    kind: str | None = None,
) -> GateResult:
    """Run the witness ``accuracy.witness.<unit>`` declares, for the unit loop and the gate CLI.

    The block is validated at task loading, including variant overlays. A
    witness that passes fails after all when a
    leg's engine turned block reuse off, reused nothing although it had a
    prefix to reuse, or ran anything other than its own argv declares: each
    leg the shim ran is attested, leg B of an equivalence witness (a launcher)
    is not. Every verdict names the tree it measured (:func:`~.result.inspect_tree`).
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
        return replace(result, tree_state=tree)

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
    shim_legs = [output for command, output in legs if str(ENGINE_DRIVER) in command]
    attested = []
    for number, output in enumerate(shim_legs):
        records = parse_engine_records(output)
        requested = dict(records[-1].get("requested") or {}) if records else {}
        attested.append(
            attest_engine(
                output,
                engine_declaration({"llm": requested}, mode=mode),
                leg=names[number] if number < len(names) else f"#{number + 1}",
            )
        )
    result = stamped(replace(result, engine=engine_records(attested)))
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
    workdir: str | Path,
    boot: BootSpec | None,
    kind: str,
) -> GateResult:
    """:func:`run_declared_witness` without the checks on its legs' engines."""
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


__all__ = ["WITNESS_KINDS", "WITNESS_LEG_NAMES", "run_declared_witness"]
