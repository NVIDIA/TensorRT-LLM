# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A test of the spec's own (``verify.tests``): its pytest nodes, run as the tree runs them.

Each node runs from its pytest root with a JUnit report and is judged by it
(``checks/pytest_node.py``): every selected test passed, at least one was
selected, and a skip fails it. A test that names several nodes runs them in
turn and passes when every one did.

A whole-model test (``members``) runs at one member and is handed what the
flow's own checks of its feature run with there (``spec.verify.HANDOVER_ENV``):

- ``MV2FLOW_FEATURE`` and ``MV2FLOW_MEMBER``: the feature and the member;
- ``MV2FLOW_MODEL``: the checkpoint;
- ``MV2FLOW_LLM_OPTIONS``: a YAML file of the LLM-API options, the configuration
  ``checks/check_config.check_configuration`` builds for the feature at the
  member (core's settings, the features built up to it, a variant's overlay and
  the member's parallelism) with ``modeling_v2: require``.

A test reads them with ``LLM(model, **yaml.safe_load(open(options)))``, and keeps
its own defaults for where they are not set (the CI run of the same file). A
module test (``devices``) is handed only ``MV2FLOW_FEATURE``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..infra.compute.environment import RunEnvironment
from ..infra.files import write_text_atomic
from ..spec.task_schema import TaskSchemaError
from ..spec.verify import FEATURE_ENV, LLM_OPTIONS_ENV, MEMBER_ENV, MODEL_ENV, UserTest
from . import check_config, pytest_node
from .check_runtime import MODE_REQUIRE, ShellRunner, default_shell_runner, gate_workdir
from .verdict import GateResult, inspect_tree, verdict_factory

#: The ``max_tokens`` the configuration is built with; a test sets its own sampling.
HANDOVER_MAX_TOKENS = 256


def handover(
    environment: RunEnvironment,
    task_data: Mapping[str, Any],
    test: UserTest,
    *,
    member: str,
    features: Sequence[str] | None,
    workdir: str | Path,
) -> dict[str, str]:
    """What a whole-model test is handed at ``member``: the variables, and the options file they name."""
    config = check_config.check_configuration(
        task_data,
        environment,
        unit=test.feature,
        features=features,
        max_tokens=HANDOVER_MAX_TOKENS,
        member=member,
    )
    options = dict(config["llm"])
    model = str(options.pop("model", task_data.get("checkpoint_path") or ""))
    options["modeling_v2"] = MODE_REQUIRE
    path = gate_workdir(workdir) / f"test.{test.name}.{test.feature}.{member}.llm_api.yaml"
    write_text_atomic(path, yaml.safe_dump(options, sort_keys=True), mode=0o600)
    return {
        FEATURE_ENV: test.feature,
        MEMBER_ENV: member,
        MODEL_ENV: model,
        LLM_OPTIONS_ENV: str(path),
    }


def run_user_test(
    environment: RunEnvironment,
    *,
    task_data: Mapping[str, Any],
    test: UserTest,
    workdir: str | Path,
    runner: ShellRunner = default_shell_runner,
    member: str | None = None,
    features: Sequence[str] | None = None,
) -> GateResult:
    """Run ``test`` at ``member`` (a whole-model test's) and judge its reports.

    ``task_data`` is the spec its feature's checks run under (a variant's with
    its overlay), and ``features`` the features its target implements.
    """
    env = dict(test.env or {})
    if test.whole_model:
        try:
            env.update(
                handover(
                    environment,
                    task_data,
                    test,
                    member=str(member),
                    features=features,
                    workdir=workdir,
                )
            )
        except (TaskSchemaError, ValueError, KeyError) as exc:
            return verdict_factory(inspect_tree(task_data))(
                name=test.gate,
                member=member,
                passed=False,
                detail=f"the configuration of {test.feature} at {member} could not be built: {exc}",
            )
    else:
        env[FEATURE_ENV] = test.feature
    nodes = test.node_ids(member if test.whole_model else None)
    parts: list[GateResult] = []
    for index, node in enumerate(nodes):
        part = pytest_node.run_node(
            environment,
            name=test.gate,
            node=node,
            task_data=task_data,
            workdir=workdir,
            timeout=test.timeout,
            runner=runner,
            env=env,
            unit=test.feature if len(nodes) == 1 else f"{test.feature}.part{index}",
            member=member,
            what="the test's pytest",
        )
        if part.infrastructure:
            # A run that lost its session says nothing of the target: the check is owed still.
            return part
        parts.append(part)
    return parts[0] if len(parts) == 1 else pytest_node.combine(parts, len(nodes))


__all__ = ["HANDOVER_MAX_TOKENS", "handover", "run_user_test"]
