# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's e2e test, written into the tree and committed before the first turn.

Every check of a test of datasets runs one case of the target's generated test
(``checks/e2e.py``). Preflight compiles the contract; then :func:`write` builds
one case per such check, writes ``test_e2e.py`` and ``e2e_cases.yaml`` into the
checkout and commits what changed, signed off as the run's commit identity. A
resumed run writes them again from its spec, and commits only a change. The
run's agents do not edit them: a task whose commits do is refused at
integration (``infra.git.CommitPolicy.generated_paths``). A dry run says what it
would write and writes nothing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rich.markup import escape as escape_markup

from agent_flow.console import print_message

from ...checks import check_config, e2e
from ...checks.accuracy import resolve_model_key
from ...infra.git import CommitPolicy
from ...spec import variants as variants_mod
from ...spec.task_schema import features_through, parse_parallel_segment
from ...spec.verification import RUN_DATASETS
from ..validation import contract as contract_mod


def cases(workflow) -> dict[str, Any]:
    """The run's ``e2e_cases.yaml``: one case per check of a test of datasets, in the contract's order."""
    task_data = workflow.task_data
    environment = workflow._gate_env()
    world = environment.world_size
    tests = {(t.feature, t.name): t for t in contract_mod.run_tests(task_data, world)}
    preflight = workflow._preflight
    where, problems = e2e.resolve_extra_checks(
        Path(str(task_data["trtllm_repo_path"])),
        preflight.family,
        preflight.target_rel,
        {name for t in tests.values() for name in t.extra_checks},
    )
    if problems:
        raise ValueError("extra checks the run names are not in the tree: " + "; ".join(problems))
    specs: dict[str, tuple[Any, list[str]]] = {}
    built: list[dict[str, Any]] = []
    for check in workflow._contract.checks:
        if check.runner != RUN_DATASETS:
            continue
        feature = check.unit
        if feature not in specs:
            if variants_mod.is_variant(task_data, feature):
                spec = check_config.variant_spec(
                    task_data,
                    feature,
                    world_size=world,
                    options_dir=workflow.workspace / variants_mod.OPTIONS_DIRNAME,
                )
                unit = variants_mod.base_unit(task_data, feature)
            else:
                spec, unit = task_data, feature
            specs[feature] = (spec, list(features_through(task_data, unit)))
        spec, features = specs[feature]
        size = parse_parallel_segment(check.member)[1]
        env = replace(environment, world_size=size) if size else environment
        built.append(
            e2e.build_case(env, spec, tests[(feature, check.gate)], check.member, features, where)
        )
    return e2e.build_cases(task_data, resolve_model_key(task_data), built)


def write(workflow, log) -> list[str]:
    """Write the target's generated test and cases into the checkout and commit them; what changed.

    Nothing is written when the run owes no test of datasets, or on a dry run.
    """
    if not any(c.runner == RUN_DATASETS for c in workflow._contract.checks):
        return []
    preflight = workflow._preflight
    document = cases(workflow)
    repo = Path(str(workflow.task_data["trtllm_repo_path"]))
    paths = e2e.generated_paths(preflight.family, preflight.target_rel)
    if workflow.dry_run:
        print_message(
            f"[bold cyan]→ --dry-run: the run would write {len(document['cases'])} e2e case(s) "
            f"to {escape_markup(paths[1])}[/bold cyan]",
            log,
        )
        return []
    changed = e2e.write_generated(repo, preflight.family, preflight.target_rel, document)
    if not changed:
        return []
    commit = ""
    if (repo / ".git").exists():
        policy = CommitPolicy.from_task(workflow.task_data, workspace=workflow.workspace)
        identity = policy.identity.env() if policy.identity is not None else {}
        commit = e2e.commit_generated(repo, changed, preflight.target_rel, identity)
    said = f"{len(document['cases'])} case(s) in {paths[1]}; wrote {', '.join(changed)}"
    if commit:
        said += f"; committed as {commit[:10]}"
    workflow.journal.event("e2e test", said)
    print_message(f"[bold green]✔ e2e test[/bold green] {escape_markup(said)}", log)
    return changed


__all__ = ["cases", "write"]
