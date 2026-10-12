# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prose blocks for the staircase-onboard role prompts.

Everything here is doctrine that binds more than one role. It is carried
across from staircase's own agent definitions and entry skills nearly
verbatim, because these sentences are the expensive part of that project —
each one has an incident behind it — while the control flow they used to
sit next to is now Python.

Role-specific guidance, including each role's Red flags table, lives in
that role's module.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..embargo import PACKAGE_PREFIX


def tidy(text: str) -> str:
    """Normalize a composed prompt so it reads as one document.

    Each block below ends with its own newline, and the f-strings that
    interpolate them add their own separators, so a naive composition
    accumulates runs of blank lines at every seam. Collapsing them is
    cosmetic for a human and not cosmetic for the model: a prompt whose
    structure is inconsistent reads as several documents stapled together.
    """
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# Where everything lives now that the catalog and the targets are in tree.
# Written out per-role rather than left implicit: the single most common
# way a ported skill goes wrong is naming staircase's old standalone paths.
IN_TREE_LAYOUT = """\
## Where things live

ModelingV2 is in the TensorRT-LLM tree, not in a separate repo:

| what | path |
|---|---|
| kernel catalog | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.{md,py}` |
| catalog index | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml` |
| catalog GPU tests | `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py` |
| targets | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/{modeling,weights}.py` |
| routing | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/routing.py` |
| accuracy anchors | `tests/integration/defs/accuracy/references/<benchmark>.yaml` |
| accuracy gates CI runs | `tests/integration/defs/accuracy/test_modeling_v2_*.py` |

Two consequences worth holding onto:

- **A catalog entry spans two trees.** The contract and wrapper live in the
  package; the GPU test lives in the tests tree, because that is where this
  repo's CI collects from and an in-package test is on no list. A receipt is
  valid only if it post-dates the last write to *every* file of its entry, so
  that check now has to look in both places.
- **Every test file is named `test_modeling_v2_*`** precisely so it cannot
  collide with the upstream test of the same op, and so the CI selector can
  match it exactly. Do not invent a different name.

There is **no version pin** in tree. A target moves with the trunk, so the
pinned-version assert is gone and receipts are keyed by GPU architecture and
nothing else. What replaced it is `REQUIRED_TRTLLM_OPS` on each target, which
`tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py`
asserts — that turns a drifting engine from a wrong answer into a loud failure.
"""

# The single most expensive mistake available in this package.
MODELING_V2_SWITCH = """\
## The modeling_v2 switch

`TRTLLM_MODELING_V2` selects the modeling path: `off` (default) is today's
behaviour byte for byte, `auto` uses a target when one matches and silently
falls back when none does, `require` raises instead of falling back.

**Anything you will attribute to modeling_v2 runs under `require`.** Under
`auto`, a configuration that misses a target's routing criteria gets the
built-in implementation — and a gate or a curve measured that way reads as
modeling_v2's. It is the most expensive mistake available here.

It must be exported **before the ranks start**, not merely before `LLM(...)`.
Worker ranks receive the environment as it stood when MPI initialized, so a
value set later reaches the driver and not them, and a driver resolving a
modeling_v2 target while its workers resolve the built-in is the silent split
the switch exists to prevent. The run's launch prefix already carries it; do
not set it yourself.
"""

# Reports are claims. This is the sentence the orchestrator is built around.
EVIDENCE_DISCIPLINE = """\
## Evidence discipline

**Your final message is this run's sole deliverable.** Nothing you did is
visible outside it, and everything in it is treated as a *claim* that will be
checked against disk before anything downstream happens.

- **Every number is verbatim from tool output.** No estimates, no "roughly",
  no number you did not watch a command print. If you did not measure it, say
  you did not measure it.
- **Re-check every claim against disk before sending.** A report that says a
  file exists, a test passed, or a receipt was recorded, when the disk says
  otherwise, costs the whole run's trust in every other line of that report.
- **Distinguish what you observed from what you inferred.** Both are useful;
  conflating them is not.
- **Report what you could not do.** A blocked step named precisely is worth
  more than a step quietly skipped.
"""

# Concurrency hygiene. Measured twice in a single staircase run.
COMMIT_DISCIPLINE = """\
## Committing

Commit **by explicit file path** — never `git add -A`, and never a bare
directory. Other agents are working in this same worktree at the same time,
and a directory pathspec sweeps in whatever they have half-written. Staircase
measured this twice in one run: 576 lines of an in-progress test file and a
mid-edit contract landed in an unrelated docs commit, splitting that entry's
attribution across two commits and leaving dangling cross-references.

`git status --short` before staging costs nothing and shows exactly whose
files are in flight.

One commit per unit of work, so attribution and revert stay per-unit. No
other git command — never `push`, never a branch operation, never a rebase.
No AI attribution lines in the message.
"""

# What a target's forward may be derived from. This survives the removal of
# the isolation hook: the hook was a mechanism, these are the invariants.
SOURCE_BOUNDARY = """\
## Reading the engine

TensorRT-LLM's source is an ordinary read, including
`_torch/models/modeling_<x>.py` and the rest of the built-in model zoo. Read
whatever answers your question.

**Read to understand an interface; never to copy a structure.** A
transliterated forward cannot satisfy the closed-vocabulary and flat-forward
rules at once, and upstream's forward does not satisfy them — so code that
starts mirroring an upstream module's shape is the signal that something has
gone wrong, and the invariants are the real defence:

- the zoo's forward calls modules and helpers that have **no catalog entry**,
  so a copy of it cannot pass the vocabulary audit;
- the zoo's forward is a **module tree**; a target's is one flat function;
- the zoo's forward is built out of **shared helpers**, which a target may
  not have.

So: read it to learn what an op expects, what the runtime allocates, which
metadata field an attention variant reads. Do not read it to learn what shape
to write. A missing op is a **vocabulary gap** for smith, never a reason to
reach into the zoo for an implementation.

Other inference engines' modeling code stays out entirely.

**Where to look is handed to you, not remembered.** The run resolves a
*runtime map* before the plan phase — the engine areas this checkpoint's own
`config.json` demands, each with the directory to read and the symbol to grep
for, narrowed to this model. It carries no signatures and no line numbers on
purpose: those are the parts that go stale between one onboarding and the
next, and a stale one reads exactly like a true one. Take the location from
the map and the content from the source.

The map's `tests` pointer is worth as much as its `code` pointer. A new
entry's test should look like the tests already sitting in that directory.
"""

# How engine knowledge gets recorded. Separate from SOURCE_BOUNDARY, which is
# about reading the engine in order to *write code* and therefore only binds
# the three roles that write code. This binds all six, because all six write
# prose that a later agent reads back -- a contract, a plan, an open risk, a
# review finding, an audit -- against an engine that has moved since.
CITATION_DISCIPLINE = """\
## Writing down what you read

**Describe the behaviour and name the symbol — never the line.** Whatever you
record about TensorRT-LLM is read back later against an engine that has moved.
Two ways to write the same fact down, and only one of them survives:

- **durable** — a file path plus a name the reader can grep: a function, a
  class, a field, a config key, or the exact assert text.
- **rots** — a line number, a pasted excerpt, or a hand-copied argument list.
  Wrong shortly after you write it, silently, and the next reader believes it
  because it reads like a citation.

This is measured, not a worry. Seven line-number citations were written into
this catalog and **all seven pointed at the wrong line within days** — one off
by about 35 lines, another by about 1100, landing on an unrelated f-string.
In the same sentences, every file path and every symbol name still resolved.

| instead of | write |
|---|---|
| `` `_torch/pyexecutor/model_engine.py:1103` `` | warmup calls `_run_autotuner_warmup` (`_torch/pyexecutor/model_engine.py`) |
| `` `llmapi/llm_args.py:4669` `` | `enable_autotuner`, a `TorchLlmArgs` field, defaults to `True` |
| `` `attentionOp.cpp:1045` `` | the `seqQOffset is nullptr.` guard in `common/attentionOp.cpp` |

One exception, and it is not a loophole: a **quoted diagnostic** such as
`RuntimeError: ... (../tensorrt_llm/runtime/torchUtils.h:113)` is the message
the engine prints, so quote it whole and verbatim. That is a symptom someone
greps for on hitting it, not a direction to navigate.

Both halves are enforced, not reviewed: a navigation citation and a named file
that no longer exists each fail a test.
"""

# The knowledge layer. Each role names its own skills; this explains the rule.
SKILL_DISCIPLINE = """\
## Your skills

Load the skill(s) named in your dispatch prompt **before anything else**. They
carry the spec, the invariants, the recipes and the traps; this prompt carries
only what must hold from your first token.

Each skill owns its own `references/`, and they are read **by need, not front
to back** — the skill says which one answers which question. The mechanism
files are not optional reading dressed up as reference: each records what an
earlier target had to establish by probing, and reading the right one is the
difference between an afternoon and a week.
"""


def decision_file_block(agent: str, decisions: Sequence[str], extra: Sequence[str] = ()) -> str:
    """Render the file-based decision protocol for one role.

    Used instead of the ``append_<agent>_progress`` tool on hosts where an
    enterprise MCP configuration forbids dynamically-registered servers, so
    the in-process tool the workflow prefers cannot be created at all.

    The instruction is deliberately blunt and repeats the path. A role that
    ends its turn without this file has done work nobody can act on: the
    orchestrator branches on ``decision`` and on nothing else it can read.
    """
    options = "\n".join(f"- `{d}`" for d in decisions)
    extra_block = ""
    if extra:
        extra_lines = "\n".join(f"  {k}: ..." for k in extra)
        extra_block = (
            f"\nThis role also carries these keys when they apply:\n\n```yaml\n{extra_lines}\n```\n"
        )
    return (
        "## Ending your turn\n"
        "\n"
        "**Your last action is to write your decision file.** Write it with the "
        "`Write` tool to:\n"
        "\n"
        "```\n"
        f"<workspace>/decisions/<unit>-{agent}-<turn>.yaml\n"
        "```\n"
        "\n"
        "The exact path is given at the end of your dispatch prompt. Write that "
        "file and no other: every turn gets a path no earlier turn used, and a "
        "decision written anywhere else is never read. "
        "The file is YAML with exactly these keys:\n"
        "\n"
        "```yaml\n"
        "summary: |\n"
        "  What you did, in a few lines. Numbers verbatim from tool output.\n"
        f"decision: <one of the values below>\n"
        "```\n"
        f"{extra_block}"
        "\n"
        "`decision` must be **exactly** one of:\n"
        "\n"
        f"{options}\n"
        "\n"
        "Nothing else is accepted — the orchestrator validates it against that "
        "list and treats anything else as a turn that produced no decision. It "
        "branches on this value and on nothing else in your report, so a turn "
        "that ends without this file is a turn nobody can act on.\n"
    )


def source_embargo_block(paths: Sequence[str]) -> str:
    """Render the run-scoped embargo on specific source files or trees.

    :data:`SOURCE_BOUNDARY` says the built-in model zoo is an ordinary
    read, and for most targets that is right — there is nothing there that
    corresponds to the target being built. It stops being right when an
    implementation of this exact checkpoint is *reachable*. Then "read to
    understand an interface, never to copy a structure" is a line nobody can
    audit from the outside: the target would be correct, and whether the
    pipeline could have produced it would be unknowable.

    So for those paths the boundary is moved from a discipline to a rule.
    Note what is **not** embargoed: running the other implementation is
    encouraged. It is the run's accuracy oracle, and using it as a black
    box costs nothing that reading it would have bought.

    **The opening sentence splits on where the paths point**, because the
    two cases are not the same claim. An in-repo entry means the engine
    already ships this checkpoint, and running it is a matter of
    ``TRTLLM_MODELING_V2=off``. An out-of-repo entry — a sibling vLLM or
    SGLang checkout on this filesystem — means another engine implements it,
    and running it means starting that engine as a black-box oracle. Saying
    "this repository" for the second case would be false, and a rule whose
    premise is visibly wrong is a rule an agent reasons its way around.
    Absolute spelling is the discriminator, and it is not a heuristic:
    :func:`~..task_schema._validate_source_embargo` accepts an absolute path
    only when it resolves outside ``trtllm_repo_path``.
    """
    if not paths:
        return ""
    # `package:<path>` (see `embargo.display_entry`) is the workflow's own
    # reference material — a worked example of this model — not an
    # implementation of the checkpoint, so it gets its own premise and no
    # "running it is expected" paragraph.
    reference = [p[len(PACKAGE_PREFIX) :] for p in paths if p.startswith(PACKAGE_PREFIX)]
    sources = [p for p in paths if not p.startswith(PACKAGE_PREFIX)]
    listed = "\n".join(
        [f"- `{p}`" for p in sources]
        + [f"- `{p}` (workflow reference material)" for p in reference]
    )
    out_of_repo = [p for p in sources if p.startswith("/")]
    in_repo = [p for p in sources if not p.startswith("/")]

    premise = []
    if in_repo:
        premise.append("This checkpoint already has an implementation in this repository.")
    if out_of_repo:
        premise.append(
            "This checkpoint is implemented in another engine's checkout on this "
            "filesystem, outside this repository."
        )
    if reference:
        premise.append(
            "This workflow ships reference material about this model — a worked example "
            "of onboarding it — which you must not read: deriving the target is the "
            "measurement, and an answer key would void it."
        )
    running = []
    if in_repo:
        running.append(
            "**Running the in-repository implementation is expected.** Under "
            "`TRTLLM_MODELING_V2=off` the engine uses exactly that implementation, "
            "which makes it this run's accuracy oracle: the same harness, the same "
            "workload, one variable. Measure against it freely."
        )
    if out_of_repo:
        running.append(
            "**Running the out-of-repository engine as a black-box oracle is "
            "expected; reading it is not.** Start its server, send it the workload "
            "and measure against what it returns — that is why the checkout is "
            "named here rather than merely avoided. What it must never be is a "
            "file you open."
        )

    return (
        "## Embargoed sources\n"
        "\n"
        f"{' '.join(premise)} For this run it is **off limits to read**:\n"
        "\n"
        f"{listed}\n"
        "\n"
        "An entry that names a directory embargoes the whole tree beneath it, not "
        "just the directory itself.\n"
        "\n"
        "Do not open these files, do not grep their contents, and do not quote them. "
        "If a search result includes one, skip it. This is not a claim that they are "
        "wrong — it is that a target derived from them would prove nothing about "
        "whether this pipeline can derive one.\n"
        "\n" + "\n\n".join(running) + "\n"
        "\n"
        "Your references are the checkpoint itself (`config.json`, the weight index, "
        "the quantization config) and the HuggingFace `transformers` implementation of "
        "this architecture. Both are outside the embargo.\n"
    )


def scope_block(scope: str) -> str:
    """Render the target's scope when it is narrower than the checkpoint."""
    if scope != "text_only":
        return ""
    return (
        "## Scope: text only\n"
        "\n"
        "This checkpoint is multimodal; **this target is the language model only.** The "
        "vision tower is out of scope — do not port it, do not write a catalog entry "
        "for it, and do not treat its absence as a gap to close.\n"
        "\n"
        "Two consequences that are easy to get wrong:\n"
        "\n"
        "- **The weight loader must ignore the vision tensors rather than fail on "
        "them.** They are present in the checkpoint and always will be. A loader that "
        "asserts every tensor was consumed will fail on a correct target.\n"
        "- **Gate it with text-only prompts.** An image prompt against a text-only "
        "target is a wrong question, and the error it raises is not a defect.\n"
    )


def environment_block(prefix: str, python: str, devices: str, world_size: int) -> str:
    """Render the launch-environment block injected into every role prompt.

    This is the block that keeps a run from re-litigating how to submit a
    job. It is generated from the resolved :class:`RunEnvironment` rather
    than written by hand so the prompt and the orchestrator's own gate
    invocations cannot drift apart.
    """
    return (
        "## Running GPU work\n"
        "\n"
        "Every GPU command in this run is launched exactly one way. Prefix it with:\n"
        "\n"
        "```\n"
        f"{prefix}\n"
        "```\n"
        "\n"
        f"The Python interpreter is `{python}`. The devices claimed for this run are: "
        f"{devices} (world size {world_size}).\n"
        "\n"
        "**Do not construct a launcher command, do not probe for one, and do not call a "
        "test-runner skill.** The environment was resolved in Python before you started, "
        "and it is not yours to rediscover. If this prefix does not fit what you need to "
        "run, that is a **symptom to report in your final message** — not a problem to "
        "solve.\n"
    )


__all__ = [
    "CITATION_DISCIPLINE",
    "COMMIT_DISCIPLINE",
    "EVIDENCE_DISCIPLINE",
    "IN_TREE_LAYOUT",
    "MODELING_V2_SWITCH",
    "SKILL_DISCIPLINE",
    "SOURCE_BOUNDARY",
    "environment_block",
    "scope_block",
    "source_embargo_block",
    "tidy",
]
