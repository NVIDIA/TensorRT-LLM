"""The deterministic half of fixing an operating point.

Establishing where to optimize is `create-sweep`'s job — a skill the
benchmark-config repo ships, carrying rules this workflow has no business
restating: which parallelism tiers a mixture-of-experts model admits, where
the tensor-parallel batch ceiling is, what a concurrency ladder looks like on
each. Those rules live in someone else's repo and will move; copying them
here would produce a second, quietly diverging answer.

But the skill is executed by an agent, and two things about that are not
safe to leave to one:

**Scope.** The skill's own Phase 3 finishes by rate-matching the generation
curve against a context anchor. That is the end-to-end join, and this
workflow's staged scope does not include it — so the last command has to
change. An instruction to "use the other post-processor" is exactly the kind
a busy agent honours on the first run and forgets on the second, and the
failure is silent: a frontier appears, it looks like a frontier, and nothing
downstream can tell it was built against an anchor nobody asked for. The
session this module was written after did precisely that, from a prompt that
said otherwise.

**The candidate set.** A context point can be *selected* rather than
computed — its objective, requests per second per context GPU, is scalar and
needs no rate match — but only if more than one candidate was measured. The
skill emits a single sized configuration, so widening it into a candidate
list is a change this workflow wants and the skill has no reason to make.

So the split: **this module decides and checks; the agent executes.** Every
command an agent runs is generated here, so the scope is not something it
can drift from, and every artefact it produces is validated here, so a
missing probe or an inconsistent row stops the run instead of quietly
narrowing what gets measured. Nothing here starts a process — the agent is
already on the cluster where the checkpoint is, and adding a second way to
reach it would defeat :mod:`.gitops`' single-door rule for no gain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from agent_flow.workflows.perf_optimize.bench_cli import GEN_ONLY_MODULE

#: Where the config repo keeps the skill. Both are checked because the two
#: copies have been observed to disagree: on the checkout this was written
#: against, ``.agents/skills`` held ``create-sweep`` and ``.claude/skills``
#: did not, while ``.claude/skills`` held a skill the other lacked. Neither
#: is a superset, so looking in one and reporting "not installed" would be
#: wrong about half the time.
SKILL_DIRS: tuple[str, ...] = (".agents/skills", ".claude/skills")
SKILL_NAME = "create-sweep"

#: The scripts the skill ships, and the only executable part of it this
#: module names. Everything else in ``SKILL.md`` is instructions.
FACTS_SCRIPT = "model_facts.py"
DESIGN_SCRIPT = "design_sweep.py"


class SweepDesignError(ValueError):
    """The design skill, or something it produced, is not usable."""


def scripts_dir(repo_root: Path) -> Path:
    """Where `create-sweep`'s scripts are in this checkout.

    Reports both places it looked, because "the skill is not installed" and
    "the skill is installed where the agent cannot see it" are different
    problems with the same symptom — and the second one is the one that
    actually happened.
    """
    root = Path(repo_root)
    for parent in SKILL_DIRS:
        found = root / parent / SKILL_NAME / "scripts"
        if (found / DESIGN_SCRIPT).is_file():
            return found
    raise SweepDesignError(
        f"no {SKILL_NAME}/scripts/{DESIGN_SCRIPT} under {root} in any of "
        f"{list(SKILL_DIRS)}. Note that an agent only discovers skills under "
        f"`.claude/skills`, so a copy that exists solely under `.agents/skills` "
        f"is present on disk and uninvokable — check for that before concluding "
        f"the skill is missing."
    )


# ------------------------------------------------------------- the commands


def facts_command(scripts: Path, model_path: str, out: Path) -> str:
    """Phase 0. Reads the checkpoint's index; touches no GPU."""
    return f"python3 - {model_path} < {Path(scripts) / FACTS_SCRIPT} > {out}"


def shapes_command(scripts: Path, facts: Path, out: Path, *, gpu: str, per_node: int) -> str:
    """The planning half of Phase 2: which shapes are legal at all."""
    return (
        f"python3 {Path(scripts) / DESIGN_SCRIPT} shapes --facts {facts} "
        f"--gpu-name {gpu} --gpus-per-node {per_node} > {out}"
    )


def ctx_command(
    scripts: Path,
    facts: Path,
    out: Path,
    *,
    gpu: str,
    per_node: int,
    isl: int,
    ratio: str,
    model_card: str,
    model_path: str,
    model_prefix: str,
    dataset_prefix: str,
) -> str:
    """The design half of Phase 1: a sized context configuration.

    One configuration, which :func:`widen_ctx_candidates` then turns into a
    candidate set. Generated first rather than hand-written so the sizing —
    weight bytes against GPU memory — stays the skill's arithmetic.
    """
    return (
        f"python3 {Path(scripts) / DESIGN_SCRIPT} ctx --facts {facts} "
        f"--gpu-name {gpu} --gpus-per-node {per_node} --isl {isl} --ratio {ratio} "
        f"--model-card {model_card} --model-path '{model_path}' "
        f"--model-prefix {model_prefix} --dataset-prefix {dataset_prefix} --out {out}"
    )


def sol_command(scripts: Path, plan: Path, out: Path) -> str:
    """The design half of Phase 3: the sweep, from the *measured* plan."""
    return f"python3 {Path(scripts) / DESIGN_SCRIPT} sol --plan {plan} --out {out}"


def postprocess_command(run_dir: Path) -> str:
    """How a generation run is scored here, and the one command that changed.

    The skill ends Phase 3 with ``ibc-bench process frontier --ctx_json``,
    which rate-matches the curve against a context anchor. That is the join,
    and the staged scope has none — so the anchor-free extractor is used
    instead. It computes the same ``tps_per_user`` from the same iteration
    logs; what it does not compute is the deployment view, which is exactly
    the part that is out of scope.
    """
    return f"python -m {GEN_ONLY_MODULE} -i {run_dir}"


# ---------------------------------------------------------- the two deltas


def widen_ctx_candidates(
    config: Mapping[str, Any], *, batches: Sequence[int], tp_sizes: Sequence[int]
) -> dict[str, Any]:
    """Turn the sized context configuration into a candidate set.

    The sweep format already expands ``max_batch`` x ``tp_size`` as a
    product, so more candidates is a longer list rather than new machinery.
    What it buys is that the context half is *selected from measurement*
    like the generation half, instead of being computed and taken on trust —
    and that costs nothing in rate matching, because requests per second per
    context GPU is entirely inside the context measurement.

    The sized values are kept in the lists whatever else is asked for: they
    are the skill's own answer, and dropping them would make the sweep unable
    to confirm or refute it.
    """
    if not batches or not tp_sizes:
        raise SweepDesignError("a candidate set needs at least one batch and one tp_size")
    widened = {key: value for key, value in config.items()}
    entries = widened.get("benchmarks")
    if not isinstance(entries, list) or not entries:
        raise SweepDesignError(
            "the generated ctx config has no 'benchmarks' entry to widen — the "
            "design script's output shape changed, so this would silently sweep "
            "nothing"
        )
    out: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SweepDesignError(f"a 'benchmarks' entry is not a mapping: {entry!r}")
        row = dict(entry)
        row["max_batch"] = sorted({*_ints(entry.get("max_batch")), *batches})
        row["tp_size"] = sorted({*_ints(entry.get("tp_size")), *tp_sizes})
        out.append(row)
    widened["benchmarks"] = out
    # The override table is dropped rather than merged: it restates the same
    # keys per GPU, and a sweep whose candidates come from two places is one
    # whose planned expansion and measured expansion can disagree without
    # anything raising.
    widened.pop("gpu_overrides", None)
    return widened


def _ints(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return [value] if isinstance(value, int) and not isinstance(value, bool) else []


# ------------------------------------------------------------ the checks


def verify_plan(plan: Mapping[str, Any], *, expected_shapes: Sequence[str]) -> list[str]:
    """Refuse a probe plan that silently measured fewer shapes than it planned.

    A probe that fails leaves its shape out of ``entries`` rather than
    failing the phase, so a plan can be well-formed, submit cleanly, and
    characterise four of six shapes. The sweep that follows would then rank
    a frontier the missing two were never on — and the report would describe
    it as the frontier.
    """
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries:
        raise SweepDesignError(
            "the probe plan carries no 'entries': no shape's max batch was "
            "measured, so the sweep would have nothing to expand"
        )
    measured: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SweepDesignError(f"a plan entry is not a mapping: {entry!r}")
        shape = entry.get("shape")
        name = _shape_name(shape) if isinstance(shape, Mapping) else str(shape)
        batch = entry.get("max_batch")
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise SweepDesignError(
                f"plan entry {name!r} has no usable max_batch ({batch!r}). The batch "
                f"is measured, never defaulted: a guessed one either wastes the "
                f"sweep on a configuration that will not fit, or leaves throughput "
                f"unmeasured on one that would have."
            )
        measured[name] = batch
    return [shape for shape in expected_shapes if shape not in measured]


def _shape_name(shape: Mapping[str, Any]) -> str:
    kind = "dep" if shape.get("adp") else "tep"
    return f"{kind}_{shape.get('tp_size')}"


#: The house rule, and `check-gen-config`'s Rule 1. Restated here only as an
#: assertion over a generated file — the rule itself stays the skill's.
def verify_sol_yaml(config: Mapping[str, Any]) -> None:
    """Check the generated sweep before an allocation is spent on it.

    Two things, both of which produce a run that succeeds and a number that
    is about something else:

    - ``max_num_tokens`` must be ``batch * (mtp + 1)``. Under it, the
      generation step is token-starved and the measured iteration time
      belongs to a smaller batch than the row claims.
    - no build source. This sweep measures the image, so an inherited
      ``trtllm_install`` would silently characterise somebody's checkout
      instead — and the operating point every later campaign freezes on
      would have been chosen on code that is not the baseline.
    """
    for key in ("trtllm_install", "trtllm_patch"):
        if config.get(key) is not None:
            raise SweepDesignError(
                f"the generated sweep carries '{key}'. Fixing the operating point "
                f"measures the image, so a build source here would choose the point "
                f"on code that is not what any campaign starts from."
            )
    for index, row in enumerate(config.get("gen_configs") or []):
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            raise SweepDesignError(f"gen_configs[{index}] is not a full row: {row!r}")
        batch, mnt, mtp = row[3], row[4], row[7]
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (batch, mnt, mtp)):
            raise SweepDesignError(f"gen_configs[{index}] has non-integer batch/mnt/mtp: {row!r}")
        if mnt != batch * (mtp + 1):
            raise SweepDesignError(
                f"gen_configs[{index}]: max_num_tokens {mnt} != batch {batch} x "
                f"(mtp {mtp} + 1) = {batch * (mtp + 1)}. Under it the generation "
                f"step is token-starved and the measured iteration time belongs to "
                f"a smaller batch than the row claims."
            )


def designer_instruction(
    *,
    scripts: Path,
    design_dir: Path,
    shapes: Sequence[str],
    facts: Path,
    ctx_config: Path,
    sol_yaml: Path,
    plan: Path,
) -> str:
    """What the design agent is asked to do, and only that.

    Everything deterministic is already generated by the time this is built,
    so the agent's job is the part that genuinely needs judgement: driving
    the probe loop, deciding whether a failed job was a memory wall or an
    infrastructure fault, and waiting on the cluster. The scored command is
    handed over verbatim rather than described, so the staged scope is not
    something the agent can restate differently on a later turn.
    """
    return (
        f"Fix this model's operating point by driving the **create-sweep** skill's "
        f"probe and sweep phases. Everything that can be generated already has "
        f"been; what is left needs judgement, which is why you are running it.\n\n"
        f"**Design directory:** `{design_dir}` — read `{DESIGN_STATE_HINT}` there "
        f"first and resume from whatever phase it records.\n\n"
        f"**Already on disk (do not regenerate):**\n"
        f"- `{facts}` — the model facts\n"
        f"- `{ctx_config}` — the ctx candidate sweep, already widened so the "
        f"context point can be *selected from measurement* rather than computed. "
        f"Submit it as-is; do not narrow it.\n\n"
        f"**Your work, in order:**\n\n"
        f"1. **Probe each shape's max batch.** Shapes: {list(shapes)}. Follow the "
        f"**test-max-batch** skill: start from a comparable model's batch, halve "
        f"on OOM, stop when one passes and twice that fails. TEP never above 256. "
        f"A shape whose probe fails for an infrastructure reason — a job that "
        f"never started, a corrupted dependency download — is retried; a shape "
        f"that genuinely OOMs at its smallest batch is recorded as not fitting. "
        f"Those are different findings and the plan must not conflate them.\n"
        f"   Record every result in `{plan}`.\n\n"
        f"2. **Submit the ctx candidate sweep** (`{ctx_config}`). It has no "
        f"dependency on the probes, so submit it first and let it run alongside "
        f"them.\n\n"
        f"3. **Generate and submit the concurrency sweep.** Build it from the "
        f"measured plan:\n"
        f"   ```bash\n   {sol_command(scripts, plan, sol_yaml)}\n   ```\n"
        f"   then validate it with the **check-gen-config** skill before "
        f"submitting.\n\n"
        f"4. **Score it — with this command and no other:**\n"
        f"   ```bash\n   {postprocess_command(Path('<each run dir>'))}\n   ```\n"
        f"   Do **not** run `ibc-bench process frontier`, and do not pass a "
        f"`--ctx_json` to anything. This campaign builds no end-to-end view: the "
        f"frontier that command produces rate-matches the generation curve "
        f"against a context anchor, which is a measurement this scope does not "
        f"take. It would still emit a curve, and the curve would still look "
        f"correct — that is precisely why the command is given here rather than "
        f"described.\n\n"
        f"**Do not select the operating point.** Your output is the measured "
        f"space; choosing a point from it happens after you, against a stated "
        f"preference you have not been given.\n\n"
        f"**Every number must be traceable to a file you can name.** A shape "
        f"whose probe or sweep failed is reported as failed — never omitted, "
        f"because a silently missing shape becomes a frontier it was never on."
    )


#: Named separately so the prompt and :mod:`.disagg_sol` cannot drift about
#: which file carries the design's resume state.
DESIGN_STATE_HINT = "state.json"


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a generated file, or say which one could not be read."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SweepDesignError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise SweepDesignError(f"{path} must be a YAML mapping, got {type(data).__name__}")
    return dict(data)
