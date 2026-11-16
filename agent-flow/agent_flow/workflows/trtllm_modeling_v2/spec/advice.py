# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a spec is likely to cost or miss: advice, printed when it is written and when a run starts.

None of it stops anything. Only a spec that cannot be read as one stops a run (a schema
error, options that cannot resolve: ``spec/task_schema.py``); a check that lacks what it
is judged by is blocked before its own load, by the check (``checks/session.py``). What is
here is what the intake lint held as opinions: an engine too narrow for the weights, a
benchmark with nothing to hold it to or with an anchor that reads as a placeholder,
where the checkpoint's ``config.json`` cannot meet a
deployment as stated, notes that carry no brief of the model, the engine's limits for
this geometry and what the tree already has for it. ``intake derive``, ``intake lint``
and ``--dry-run`` print it; the run's journal keeps it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from . import deployments as deployments_mod
from .deployments import Deployment

#: The share of a device's memory an engine's weights may take: the rest holds the KV
#: cache, activations and CUDA graphs.
ENGINE_WEIGHT_FRACTION = 0.75
#: A score under which a benchmark's anchor reads as a deliberately low placeholder.
PLAUSIBLE_ANCHOR: dict[str, float] = {"gsm8k": 60.0}
#: Where a config states its multi-token-prediction layers.
DRAFT_LAYER_KEYS = ("num_nextn_predict_layers", "num_mtp_modules", "mtp_num_hidden_layers")
#: What marks a config as a hybrid model whose layers keep recurrent state: a key that
#: configures it, or a per-layer type that names it.
_RECURRENT_KEYS = ("linear_attn_config", "linear_num_value_heads", "mamba_ssm_cfg", "ssm_cfg")
_RECURRENT_WORDS = ("mamba", "linear", "ssm", "delta", "recurrent")


def checkpoint_weight_bytes(checkpoint: str | Path | None) -> int | None:
    """The checkpoint's weights in bytes: its safetensors index's total, else its files' sizes."""
    root = Path(str(checkpoint or ""))
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        total = int((index.get("metadata") or {}).get("total_size") or 0)
    except (OSError, ValueError, TypeError, AttributeError):
        total = 0
    try:
        return total or sum(f.stat().st_size for f in root.glob("*.safetensors")) or None
    except OSError:
        return None


def unfit(task_data: Mapping[str, Any], deployment: Deployment, memory_mib: float | None) -> str:
    """Why ``deployment``'s narrowest server cannot hold the weights on such devices; else ``""``."""
    weights = checkpoint_weight_bytes(task_data.get("checkpoint_path"))
    # Expert and attention data parallel keep each rank's share of the weights no smaller
    # than tensor parallel does: the narrowest server's ranks are what must hold them.
    ranks = min(server.ranks for server in deployment.servers.values())
    room = ENGINE_WEIGHT_FRACTION * (memory_mib or 0) * 2**20 * ranks
    if not weights or not memory_mib or weights <= room:
        return ""
    return (
        f"`{deployment.name}` cannot hold the weights: {weights / 2**30:.0f} GiB on {ranks} "
        f"device(s) of {memory_mib / 1024:.0f} GiB is more than {ENGINE_WEIGHT_FRACTION:.0%} of "
        f"them, which leaves the KV cache and CUDA graphs no room; give it more ranks"
    )


def checkpoint_notes(task_data: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Where the checkpoint's ``config.json`` cannot meet a deployment as stated, one note each.

    A server's ranks split the attention heads, the KV heads and the experts; a deployment
    that drafts has draft layers; one that takes images has a tower; a quantized format
    runs on its architecture; block reuse on a hybrid checkpoint needs a snapshot policy.
    Notes, never a stop: the engine's own load decides, and a boot shows it.
    """
    text = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else {}
    # A multimodal checkpoint keeps the language model's shape under ``text_config``: its
    # heads, experts and draft layers are read through it, as derive reads them.
    shape = {**text, **{k: v for k, v in config.items() if k != "text_config"}}
    heads = shape.get("num_attention_heads")
    kv_heads = shape.get("num_key_value_heads") or heads
    experts = next(
        (
            shape[k]
            for k in ("num_experts", "n_routed_experts", "num_local_experts")
            if shape.get(k)
        ),
        None,
    )
    drafts = any(shape.get(k) for k in DRAFT_LAYER_KEYS) or any(
        str(k).startswith("dspark") for k in shape
    )
    layers = [str(t).lower() for block in (config, text) for t in block.get("layer_types") or ()]
    hybrid = any(block.get(k) for block in (config, text) for k in _RECURRENT_KEYS) or any(
        word in layer for layer in layers for word in _RECURRENT_WORDS
    )
    said: list[str] = []
    for deployment in deployments_mod.resolve(task_data):
        for role, server in deployment.servers.items():
            who = deployment.name if role == "server" else f"{deployment.name}.{role}"
            options, kv = server.options, server.options.get("kv_cache_config") or {}
            ranks = int(options.get("tensor_parallel_size") or 1)
            split = int(options.get("moe_expert_parallel_size") or 1)
            # Attention DP replicates attention; without it the ranks split its heads.
            if ranks > 1 and not options.get("enable_attention_dp"):
                if heads and heads % ranks:
                    said.append(f"{who}: {heads} attention heads do not split {ranks} ways")
                if kv_heads and kv_heads % ranks and ranks % kv_heads:
                    said.append(f"{who}: {kv_heads} KV heads do not split {ranks} ways")
            if split > 1 and experts and experts % split:
                said.append(f"{who}: {experts} experts do not split {split} ways")
            drafter = str((options.get("speculative_config") or {}).get("decoding_type") or "")
            if drafter.lower() in ("mtp", "dspark") and not drafts:
                said.append(
                    f"{who}: it drafts with {drafter}, and config.json states no draft layers"
                )
            snapshots = kv.get("mamba_state_cache_interval") or any(
                (kv.get("mamba_state_config") or {}).values()
            )
            if hybrid and kv.get("enable_block_reuse") and not snapshots:
                said.append(
                    f"{who}: block reuse on a hybrid checkpoint with no Mamba state snapshot policy: "
                    f"the engine turns reuse off, and a boot's repeated prompt then resumes nothing; "
                    f"set `kv_cache_config.mamba_state_config.periodic_snapshot_interval`"
                )
        if "image" in deployment.inputs and not {"vision_config", "audio_config"} & set(config):
            said.append(f"{deployment.name}: it takes images, and the checkpoint has no tower")
    arch = str(task_data.get("arch", ""))[3:]
    algo = str((config.get("quantization_config") or {}).get("quant_algo") or "").lower()
    if "fp4" in algo and arch.isdigit() and int(arch) < 100:
        said.append(f"the checkpoint's {algo} weights need sm_100 or later")
    return [f"checkpoint: {line}" for line in dict.fromkeys(said)]


def spec_warnings(task_data: Mapping[str, Any]) -> list[str]:
    """What the spec is likely to cost or miss, one note each; never a stop.

    The first deployment holds the weights on too few devices
    (``environment.gpu_memory_gib`` says how much one has) or has no anchor (``onboarded``
    can then only mean it boots); a benchmark has nothing to hold it to (its check will be
    blocked before the load) or an anchor that reads as a placeholder; a declared anchor
    is 1 or less on the percent scale.
    """
    from ..checks import anchors  # read here: the anchors read the deployments of this package

    try:
        found = deployments_mod.resolve(task_data)
    except Exception:  # noqa: BLE001 - advice never stops what it advises on
        return []
    said: list[str] = []
    if not found:
        return []
    core = found[0]
    memory = (task_data.get("environment") or {}).get("gpu_memory_gib")
    narrow = unfit(task_data, core, memory * 1024 if isinstance(memory, (int, float)) else None)
    if narrow:
        said.append(f"first deployment too narrow: {narrow}")
    if not core.benchmarks:
        said.append(f"no accuracy anchor: `{core.name}` names no benchmark: nothing judges it")
    for deployment in found:
        for benchmark in deployment.benchmarks:
            who = f"`{deployment.name}` on `{benchmark.name}`"
            anchor = anchors.resolve(task_data, deployment, benchmark)
            if anchor.kind == anchors.NONE:
                said.append(
                    f"no accuracy anchor: {who} has {anchor.source}, so the benchmark will be "
                    f"blocked before its load; measure one with an implementation that already "
                    f"serves the checkpoint and write the row, declare `anchor` with its "
                    f"`source`, or drop the benchmark to onboard on the session's other checks"
                )
            plausible = PLAUSIBLE_ANCHOR.get(benchmark.name)
            if anchor.value is not None and plausible is not None and anchor.value < plausible:
                said.append(
                    f"anchor reads as a placeholder: {who} is held to {anchor.value:g}, under "
                    f"the {plausible:g} a working model of this kind scores; a target that "
                    f"answers badly passes it. Replace it once a measured score is known "
                    f"(`trtllm-modeling-v2 resolve ... change-requirement` changes a frozen one "
                    f"and records why)"
                )
            if (
                benchmark.anchor is not None
                and benchmark.anchor <= 1
                and benchmark.unit == "percent"
            ):
                said.append(
                    f"anchor under one percent: {who} declares {benchmark.anchor:g} with `unit: "
                    f"percent`; if the benchmark reports a fraction, state `unit: fraction`, or "
                    f"every score passes"
                )
    return said


def kv_dtype_notes(task_data: Mapping[str, Any], repo: str | Path | None) -> list[str]:
    """Each ``kv_cache_config.dtype`` a deployment states that the tree's engine does not name.

    Read off the tree, where it says which dtypes it loads a model with; a tree that does
    not say is not guessed at. A note: the engine's own load decides.
    """
    from ..operators import prior_art

    accepted = prior_art.engine_kv_dtypes(repo)
    if accepted is None:
        return []
    values, source = accepted
    asked: dict[str, list[str]] = {}
    for deployment in deployments_mod.resolve(task_data):
        for role, server in deployment.servers.items():
            dtype = (server.options.get("kv_cache_config") or {}).get("dtype")
            if dtype is not None and str(dtype) not in values:
                who = (
                    deployment.name
                    if role == deployments_mod.SERVER
                    else f"{deployment.name}.{role}"
                )
                asked.setdefault(str(dtype), []).append(who)
    return [
        f"KV dtype: kv_cache_config.dtype {dtype!r} ({', '.join(where)}) is not one the engine "
        f"names ({source}: {', '.join(values)}): its load is likely to refuse it; `auto` keeps "
        f"the checkpoint's own KV dtype"
        for dtype, where in sorted(asked.items())
    ]


def agents_notes(task_data: Mapping[str, Any]) -> list[str]:
    """What the spec's ``agents`` block leaves unsaid that a run cannot start its workers without."""
    from ..agents.roles import AgentConfigError, resolve_role_agents

    try:
        resolve_role_agents(task_data)
    except AgentConfigError as exc:
        return [
            f"agents: {exc}; write your own backend, model and reasoning_effort into "
            f"agents.defaults"
        ]
    except Exception:  # noqa: BLE001 - advice never stops what it advises on
        return []
    return []


def loads_note(task_data: Mapping[str, Any]) -> str:
    """How many model loads the spec implies, in one line; ``""`` for a spec with no deployment.

    One session per deployment, and a session loads each of its role instances once. The
    assembly's own run of its card's check is that session: its row is read by the
    session's task, not measured again. What adds to it is said, not counted: nobody
    knows before a run how many attempts fail.
    """
    found = deployments_mod.resolve(task_data)
    if not found:
        return ""
    each = [(d.name, len(list(d.instances()))) for d in found]
    return (
        f"Model loads the spec implies: {sum(n for _, n in each)} at least ("
        + ", ".join(f"`{name}` {n}" for name, n in each)
        + "): one session per deployment, which loads each of its role instances once. "
        "Each attempt whose session fails and is repaired is a session more. The final pass "
        "measures a deployment again, one session more, only where what its session "
        "consumes changed after it passed."
    )


def milestone_note(task_data: Mapping[str, Any]) -> str:
    """How a spec of several deployments reaches its first milestone early, in one line.

    ``""`` for one deployment, and for a spec that already names an onboarded ref as its
    baseline. Advice: nothing is refused for it.
    """
    found = deployments_mod.resolve(task_data)
    if len(found) < 2 or task_data.get("onboarded"):
        return ""
    first, rest = found[0].name, ", ".join(f"`{d.name}`" for d in found[1:])
    held = ", ".join(f"`{d.name}`" for d in found[1:] if d.relative_to == first)
    return (
        f"To reach a first milestone early: run the first deployment (`{first}`) alone, "
        f"then the rest ({rest}) in a second run whose spec names the ref the first one "
        "pinned as its baseline (`onboarded: <ref>`). `onboarded` then takes one session, "
        "and no later landing has that session measured again."
        + (
            # The second spec cannot name a deployment it does not list.
            f" In the second spec {held} cannot be `relative_to: {first}`, which is not "
            f"measured there: state the score the first run measured for `{first}` as the "
            "benchmark's `anchor`, with its `source`."
            if held
            else ""
        )
    )


@dataclass
class Advice:
    """Everything a spec is told before a run: one line each, and what the tree already has."""

    warnings: list[str] = field(default_factory=list)
    #: The engine's limits for this checkpoint's geometry (``operators/prior_art.py``).
    constraints: list[str] = field(default_factory=list)
    #: Whether the engine has its own model of the architecture; ``None`` when it was not read.
    prior_art: Any = None
    #: One line per deployment: what its session is.
    deployments: str = ""
    #: How many model loads the spec implies (:func:`loads_note`).
    loads: str = ""
    #: With several deployments: how to reach a first milestone early (:func:`milestone_note`).
    milestone: str = ""

    @property
    def plan(self) -> tuple[str, ...]:
        """The lines about what the run will cost and in what order to ask for it."""
        return tuple(line for line in (self.loads, self.milestone) if line)

    def summary(self) -> str:
        return f"{len(self.warnings) + len(self.constraints)} note(s); none stops the run"

    def render(self) -> str:
        lines = ["# Intake advice", "", self.summary(), "", "## Notes", ""]
        lines += [f"- {line}" for line in (*self.warnings, *self.constraints)] or ["None."]
        if self.deployments:
            lines += ["", "## Deployments", "", self.deployments]
        for line in self.plan:
            lines += ["", line]
        if self.prior_art is not None and self.prior_art.reference():
            lines += ["", "## The engine's own model", "", self.prior_art.reference()]
        return "\n".join(lines).rstrip() + "\n"


def advise(
    task_data: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    repo: str | Path | None = None,
) -> Advice:
    """The advice a spec gets: its warnings, the checkpoint's notes, the engine's limits and own model.

    ``checkpoint`` and ``repo`` default to the spec's own. Whatever cannot be read (a
    checkpoint with no ``config.json``, a tree that is not there) is left out: advice
    never stops what it advises on.
    """
    from ..operators import prior_art  # read here: the operators read the spec's helpers

    found = Advice(warnings=list(spec_warnings(task_data)))
    found.warnings += agents_notes(task_data)
    try:
        found.deployments = deployments_mod.describe(deployments_mod.resolve(task_data))
        found.loads = loads_note(task_data)
        found.milestone = milestone_note(task_data)
    except Exception:  # noqa: BLE001 - advice never stops what it advises on
        pass
    checkpoint = checkpoint or task_data.get("checkpoint_path")
    repo = repo or task_data.get("trtllm_repo_path")
    try:
        config = json.loads((Path(str(checkpoint)) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        config = None
    if isinstance(config, Mapping):
        try:
            found.warnings += checkpoint_notes(task_data, config)
            found.warnings += kv_dtype_notes(task_data, repo)
            found.constraints = list(prior_art.engine_constraints(config, repo))
            if repo:
                found.prior_art = prior_art.find_prior_art(repo, config)
        except Exception as exc:  # noqa: BLE001 - advice never stops what it advises on
            found.warnings.append(f"the checkpoint and the tree were not read ({exc})")
    return found


def main(argv: list[str] | None = None) -> int:
    """``trtllm-modeling-v2 intake lint --task SPEC``: print the advice; 1 only for a spec that does not load."""
    import argparse

    from .task_schema import TaskSchemaError, load_and_validate_task_yaml

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 intake lint",
        description="Say what a spec is likely to cost or miss before launch. Advice: only a "
        "spec that does not load (a schema error, options that cannot resolve) exits 1. CPU "
        "only; reads the checkpoint config and the tree.",
    )
    parser.add_argument("--task", type=Path, required=True, help="The spec to read.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Default: checkpoint_path.")
    parser.add_argument("--repo", type=Path, default=None, help="Default: trtllm_repo_path.")
    parser.add_argument("--report", type=Path, default=None, help="Also write the advice here.")
    args = parser.parse_args(argv)
    try:
        task = load_and_validate_task_yaml(args.task)
    except TaskSchemaError as exc:
        print(f"error: {exc}")
        return 1
    text = advise(task, checkpoint=args.checkpoint, repo=args.repo).render()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0
