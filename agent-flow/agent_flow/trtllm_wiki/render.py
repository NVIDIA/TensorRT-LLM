# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text an agent reads: the pack's ``index.md`` and the answers of each lookup."""

from __future__ import annotations

from typing import Any

from .features import FEATURES

USAGE = """\
trtllm-wiki --pack <dir> field <path|text>        LLM API / trtllm-serve field: type, default, status
trtllm-wiki --pack <dir> feature <name>           fields, tests to imitate, CI lists, ModelingV2 code, env
trtllm-wiki --pack <dir> family <family|arch>     routed architectures, targets, unit and accuracy tests
trtllm-wiki --pack <dir> op <text>                OpWrapper entries: scope, ARCHS, wrapped calls, tests
trtllm-wiki --pack <dir> ci <test path|node id>   test lists, conditions, timeouts, waives, stages
trtllm-wiki --pack <dir> accuracy <task|model>    task parameters and reference rows
trtllm-wiki --pack <dir> recipe [text] [--gpu G] [--gpus N] [--isl I] [--osl O] [--concurrency C]
trtllm-wiki --pack <dir> env [text]               TLLM_/TRTLLM_ variables and where they are read
trtllm-wiki --pack <dir> validate <options.yaml>  fields the schema does not declare
"""


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|") if value not in (None, "") else "-"


def render_index(wiki: Any) -> str:
    """The overview of a pack: what it holds, how to ask, families and features."""
    manifest = wiki.manifest
    state = manifest["commit"][:12] + (
        f" (+uncommitted {manifest['dirty']})" if manifest["dirty"] else ""
    )
    lines = [
        "# TensorRT-LLM wiki",
        "",
        f"Facts of `{manifest['repo']}` at `{state}`, built {manifest['built_at']} "
        f"in {manifest['build_s']} s. They are read from the checkout's own files; "
        "an answer names the file and line it came from, which wins over this page.",
        "",
        "| Facts | Count |",
        "|---|---|",
        *[f"| {name} | {count} |" for name, count in manifest["counts"].items()],
        "",
    ]
    if manifest["empty"]:
        lines += [f"Empty in this checkout: {', '.join(manifest['empty'])}.", ""]
    lines += ["## Ask", "", "```text", USAGE.rstrip(), "```", "", "## ModelingV2 families", ""]
    lines += ["| Family | Routes | Targets | Accuracy tests |", "|---|---|---|---|"]
    for family in wiki.families():
        targets = ", ".join(t["name"] for t in family["targets"]) or "-"
        lines.append(
            f"| `{family['family']}` | {', '.join(family['architectures']) or '-'} | {targets} "
            f"| {_cell(family['accuracy_tests']['path'])} |"
        )
    lines += ["", "## Features", "", "| Feature | What | Fields |", "|---|---|---|"]
    for key, spec in FEATURES.items():
        lines.append(
            f"| `{key}` | {spec.title} | {', '.join(f'`{f}`' for f in spec.fields) or '-'} |"
        )
    return "\n".join(lines) + "\n"


def render_field(field: dict[str, Any]) -> str:
    default = repr(field["default"]) if field.get("has_default") else "(none)"
    extra = []
    if field.get("status"):
        extra.append(f"status {field['status']}")
    if field.get("choices"):
        extra.append(f"choices {field['choices']}")
    if field.get("variants"):
        extra.append(f"in {', '.join(field['variants'])}")
    head = f"{field['path']}: {field['type']} = {default}" + (
        f"  [{'; '.join(extra)}]" if extra else ""
    )
    return head + (f"\n    {field['description']}" if field.get("description") else "")


def render_feature(page: dict[str, Any]) -> str:
    lines = [f"{page['feature']}: {page['title']}", "", "fields:"]
    for field in page["fields"]:
        lines.append(
            f"  {field['path']}: NOT IN THIS CHECKOUT'S SCHEMA"
            if field.get("missing")
            else "  " + render_field(field).replace("\n", "\n  ")
        )
    lines += ["", "tests (most mentions first; lists that run them):"]
    lines += [
        f"  {t['path']} ({t['hits']}) {', '.join(t['lists']) or 'not listed'}"
        for t in page["tests"]
    ]
    lines += ["", "ModelingV2 code that mentions it:"]
    lines += [f"  {m['path']} ({m['hits']})" for m in page["modeling_v2"]] or ["  (none)"]
    lines += ["", "environment variables:"]
    lines += [f"  {v['name']}: {', '.join(v['reads'][:2])}" for v in page["env_vars"]] or [
        "  (none)"
    ]
    return "\n".join(lines)


def render_family(family: dict[str, Any]) -> str:
    lines = [
        f"{family['family']}: routes {', '.join(family['architectures']) or '(nothing)'}",
        f"  routing: {family['routing'] or '-'}   native library: {family['has_native']}",
        "  targets:",
    ]
    for target in family["targets"]:
        lines.append(
            f"    {target['name']}  {target['path']}  classes={','.join(target['classes']) or '-'} "
            f"weights.py={target['has_weights']} csrc={target['has_native']} tests={target['unit_tests'] or '-'}"
        )
    accuracy = family["accuracy_tests"]
    lines.append(f"  accuracy tests: {accuracy['path'] or '-'}")
    for cls in accuracy["classes"]:
        lines.append(f"    {cls['name']}: {', '.join(cls['tests']) or '-'}")
    return "\n".join(lines)


def render_op(op: dict[str, Any]) -> str:
    where = (
        op["scope"]
        + (f" {op['family']}" if op["family"] else "")
        + (f"/{op['target']}" if op["target"] else "")
    )
    return (
        f"{op['name']} [{where}{'/' + op['category'] if op['category'] else ''}] {op['path']}\n"
        f"  ARCHS={','.join(op['archs']) or '-'} CELLS={'yes' if op['has_cells'] else 'no'} "
        f"calls={', '.join(op['calls']) or '-'}\n"
        f"  tests: {', '.join(op['tests']) or 'none name it'}"
    )


def render_ci(answer: dict[str, Any]) -> str:
    lines = [f"{answer['test']}:"]
    if not answer["entries"]:
        lines.append("  not in any test list (CI does not run unlisted tests)")
    elif all(e["kind"] == "qa" for e in answer["entries"]):
        lines.append("  only in QA lists: pre-merge and post-merge CI (the l0 lists) do not run it")
    for entry in answer["entries"]:
        count = entry["gpu_count"]
        gpus = f"{count.get('gte', '?')}-{count.get('lte', '?')} GPU" if count else ""
        name = entry["list"] + (" (qa)" if entry["kind"] == "qa" else "")
        where = f"{entry['stage'] or '-'} {entry['backend'] or '-'} {','.join(entry['gpu']) or '-'} {gpus}"
        timeout = f" TIMEOUT {entry['timeout_min']} min" if entry["timeout_min"] else ""
        lines.append(f"  {name} [{where}] {entry['test']}{timeout}")
    for listed, stages in answer["stages"].items():
        if stages:
            lines.append(f"  stages of {listed}: {', '.join(stages)}")
    for waive in answer["waives"]:
        lines.append(f"  WAIVED {waive['test']} {waive['reason']}")
    return "\n".join(lines)


def render_recipe(record: dict[str, Any]) -> str:
    layout = f"tp{record['tp'] or '?'} ep{record['ep'] or '?'}" + (
        " adp" if record["attention_dp"] else ""
    )
    spec = f" {record['speculative']}{record['draft_len'] or ''}" if record["speculative"] else ""
    return (
        f"{record['path']}{' :: ' + record['name'] if record['name'] else ''}\n"
        f"  {record['source']} model={record['model'] or record['arch']} gpu={record['gpu']} gpus={record['gpus']} "
        f"isl={record['isl']} osl={record['osl']} conc={record['concurrency']} {layout}{spec} "
        f"mbs={record['max_batch_size']} distance={record['distance']}"
    )
