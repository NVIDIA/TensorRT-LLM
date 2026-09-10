"""Asking the design skill for an operating point, on this workflow's terms.

Establishing *where* to optimize is `create-sweep`'s job — a skill the
benchmark-config repo ships, carrying rules this workflow has no business
restating: which parallelism tiers a mixture-of-experts model admits, where
the tensor-parallel batch ceiling is, what a concurrency ladder looks like on
each, and where its generated configs may be written. Those rules live in
someone else's repo and will move; a copy here would be a second, quietly
diverging answer.

So this module does not drive the skill. It hands an agent one instruction —
invoke it, follow it, with two named departures — and reads back what the
skill wrote.

**Two departures, and why only these.** The skill ends its sweep phase with
``ibc-bench process frontier --ctx_json``: the end-to-end join, which this
staged scope excludes. And its later phases prune irreversibly against a
measured MTP accept rate this model does not have. Both are changes of
*meaning*, which the skill cannot know about, so they are stated. The scored
command is given verbatim rather than described because a described one is
the kind an agent honours on the first turn and restates on the fourth — the
session this module was written after did exactly that.

**What this module is careful not to be.** Three earlier versions of it
overreached, and the shape of each mistake is worth keeping because the same
trade will look attractive again:

- Four wrappers generated the skill's own commands, on the reasoning that a
  generated command cannot be drifted from. But ``SKILL.md`` also carries a
  rule they had no place for — for an existing model directory the generated
  YAMLs go into ``sweep_design/``, never on top of the curated configs — and
  they took the output path as an unguarded parameter. Trading a recomputable
  mistake (a frontier scored the wrong way) for an unrecoverable one
  (overwriting a config other people maintain) is not a safety improvement.
- One function rewrote the context sweep to widen its candidate list. Never
  called, and it would have been this module editing a config it does not own.
- Two checks read the skill's artefacts back and refused inconsistent ones.
  Sound in principle and never wired to anything, so they asserted nothing
  while looking like they did.

What is left is an instruction, the command that instruction overrides, and a
YAML reader. **Nothing here writes a file, edits a config, or starts a
process.**
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from agent_flow.workflows.perf_optimize.bench_cli import GEN_ONLY_MODULE


class SweepDesignError(ValueError):
    """The design skill, or something it produced, is not usable."""


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
        f"**Run END-TO-END. This instruction is the confirmation the skill's "
        f"submission gates ask for.** Do not stop to ask before submitting: you "
        f"are running non-interactively and there is no second turn in which an "
        f"answer could reach you, so a gate that waits is a gate that ends the "
        f"design. Report the case count and node estimate as the skill asks -- "
        f"and then submit.\n\n"
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
