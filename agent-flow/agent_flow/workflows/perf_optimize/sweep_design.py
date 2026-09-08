"""The deterministic half of fixing an operating point.

Establishing where to optimize is `create-sweep`'s job — a skill the
benchmark-config repo ships, carrying rules this workflow has no business
restating: which parallelism tiers a mixture-of-experts model admits, where
the tensor-parallel batch ceiling is, what a concurrency ladder looks like on
each. Those rules live in someone else's repo and will move; copying them
here would produce a second, quietly diverging answer.

**The skill runs normally, and this module checks what it produced.**

An earlier version of this file generated every command the design agent
ran — four wrappers around the skill's own scripts — on the reasoning that a
generated command cannot be drifted from. That was a bad trade, and the way
it was bad is worth recording, because the same trade will look attractive
again.

The failure it prevented was one silent one: the skill ends its sweep phase
with ``ibc-bench process frontier --ctx_json``, the end-to-end join, which
this staged scope excludes. An agent told in prose to use the other
post-processor may well forget on a later turn, and the result looks correct
— a frontier appears, and nothing downstream can tell which anchor built it.

The failure it *introduced* was worse. ``SKILL.md`` carries a rule those
wrappers had no place for: for an **existing** model directory the generated
YAMLs go into ``sweep_design/``, never on top of the curated configs. The
wrappers took the output path as a free parameter with no guard, so one
wrong argument would have overwritten a checked-in config that other people
maintain — silently, irreversibly, and outside this run. Trading a
recomputable mistake for an unrecoverable one is not a safety improvement.

More generally: the skill's phases are not four commands. They are also
gates, a resume spine, an output-placement rule, and a probe protocol.
Restating four argv strings dropped the rest of it while looking like
faithful automation.

So the skill is invoked as a skill and left to apply its own policies. What
survives here is:

- **one override** — :func:`postprocess_command`, because it is the one
  place this workflow's scope genuinely differs from the skill's, and
  because it changes semantics rather than paths;
- **checks on the artefacts** — a probe plan that quietly dropped a shape, a
  row whose ``max_num_tokens`` is inconsistent, a design sweep carrying a
  build source. These read what the skill wrote and refuse it; none of them
  can damage anything.

Nothing here starts a process or writes a file the skill owns.
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


# ------------------------------------------------------------ the one override


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


def designer_instruction(*, model_dir: str, design_dir: Path, tracks: Sequence[str]) -> str:
    """What the design agent is asked to do: run the skill, with one override.

    Deliberately short. The skill's own ``SKILL.md`` carries the phases, the
    probe protocol, the submission gates, the resume spine and — the reason
    an earlier version of this function was wrong — the rule about where
    generated YAMLs may be written. Restating any of that here would produce
    a second copy to drift from; the agent reads the skill.

    What this adds is the one thing the skill cannot know: that this
    workflow's scope excludes the end-to-end join, so the sweep is scored
    anchor-free. That is a change of meaning rather than of path, and it is
    given as the exact command because a described one is the kind an agent
    honours on the first turn and restates differently on the fourth.
    """
    return (
        f"Fix this model's operating point: invoke the **create-sweep** skill and "
        f"follow it. It carries the phases, the probe protocol, the submission "
        f"gates and the rules about where its generated YAMLs may be written -- "
        f"follow those as written, including its rule that for an EXISTING model "
        f"directory the generated configs go into `sweep_design/` and never on "
        f"top of the curated ones.\n\n"
        f"- model_dir: `{model_dir}` (existing)\n"
        f"- design directory: `{design_dir}` -- read its `{DESIGN_STATE_HINT}` "
        f"first and resume from whatever phase it records\n"
        f"- halves this campaign will optimize afterwards: {list(tracks)}\n\n"
        f"**Two departures from the skill, and only these two.**\n\n"
        f"**1. Stop after the concurrency sweep.** Do not run the predict/prune "
        f"phase or the final frontier. Both require a measured MTP accept rate "
        f"this model does not have, and the pruning is irreversible -- it would "
        f"throw away points on the strength of a multiplier nobody measured.\n\n"
        f"**2. Score the generation runs with this command and no other:**\n"
        f"```bash\n{postprocess_command(Path('<each run dir>'))}\n```\n"
        f"Do **not** run `ibc-bench process frontier`, and do not pass "
        f"`--ctx_json` to anything. This campaign builds no end-to-end view: that "
        f"command rate-matches the generation curve against a context anchor, a "
        f"measurement this scope does not take. It would still emit a curve, and "
        f"the curve would still look correct -- which is why the replacement is "
        f"given as a command rather than described.\n\n"
        f"**Do not select the operating point.** Your output is the measured "
        f"space; choosing from it happens after you, against a stated preference "
        f"you have not been given.\n\n"
        f"**A shape whose probe or sweep failed is reported as failed** -- never "
        f"omitted, because a silently missing shape becomes a frontier it was "
        f"never on. Say which failed and why: a job that never started and a "
        f"genuine memory wall are different findings."
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
