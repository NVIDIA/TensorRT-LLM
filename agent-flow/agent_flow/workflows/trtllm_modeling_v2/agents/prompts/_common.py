# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared prose blocks for the trtllm-modeling-v2 role prompts.

Everything here is doctrine that binds more than one role: the rules each
role must follow whatever it is working on. Control flow is not here; the
orchestrator owns it.

Role-specific guidance, including each role's Red flags table, lives in
that role's module.
"""

from __future__ import annotations

import re
from typing import Sequence

from ...onboard.embargo import PACKAGE_PREFIX


def tidy(text: str) -> str:
    """Normalize a composed prompt so it reads as one document.

    Each block below ends with its own newline, and the f-strings that
    interpolate them add their own separators, so a naive composition
    accumulates runs of blank lines at every seam. Collapsing them is
    cosmetic for a human and not cosmetic for the model: a prompt whose
    structure is inconsistent reads as several documents stapled together.
    """
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# Where everything lives in tree. Written out because the most common way a
# ported skill goes wrong is naming the old standalone paths.
IN_TREE_LAYOUT = """\
## Where things live

| what | path |
|---|---|
| shared catalog entry | `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py` (base: `catalog/_op.py`: `OpWrapper`, `Cell`, `Arch`) |
| its GPU test | `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<name>.py` |
| target | `tensorrt_llm/_torch/_experimental/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/{modeling,weights}.py` |
| an op only that target calls | `.../models/<family>/<ckpt>__<arch>__<parallel>/ops/<category>/<name>.py`, tested at `tests/unittest/_torch/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py` |
| a shared entry's local variant, until the promote | `.../ops/<category>/<name>__<site>.py` in the target, tested beside the target's op tests as `test_modeling_v2_<name>__<site>.py` |
| routing | `tensorrt_llm/_torch/_experimental/modeling_v2/_router_index.py` and `models/<family>/routing.py` |
| source-level gates | `tests/unittest/_torch/modeling_v2/test_modeling_v2_{claims,routing,no_stale_claims,target_contract}.py` |
| accuracy anchors | `tests/integration/defs/accuracy/references/<benchmark>.yaml` |

An entry spans two trees (the class and its test), so "every file of the
entry" means both. Every test file is named `test_modeling_v2_*`, so it never
collides with upstream's test of the same op and the CI selector matches it.
"""

# The single most expensive mistake available in this package.
MODELING_V2_SWITCH = """\
## The modeling_v2 switch

The LLM-API argument `modeling_v2` (`off` | `auto` | `require`) selects the
path. **Anything you attribute to modeling_v2 runs under `require`**: under
`auto` a configuration that misses routing silently serves the built-in model.
The gates carry it as `TRTLLM_MODELING_V2` and hand it to the LLM constructor;
a command you write yourself passes the argument.
"""

EVIDENCE_DISCIPLINE = """\
## Evidence

Your decision file is what the orchestrator acts on; your report is a claim a
reader checks against disk. Every number is verbatim from tool output, what you
inferred is marked as inferred, and what you could not do is named.
"""

# What a target's forward may be derived from: binds the roles that write code.
SOURCE_BOUNDARY = """\
## Reading the engine

TensorRT-LLM's source, the built-in model zoo included, is an ordinary read.
**Read to understand an interface; never to copy a structure**: the zoo's
forward is a module tree built on helpers with no catalog entry, and a copy of
it cannot pass the closed-vocabulary audit. A missing op is a **vocabulary
gap** for smith. Other inference engines' code is never a source for a target
or an entry. Where to look comes from the run's *runtime map*.
Take the location from the map and the content from the source.
"""

# Binds every role that writes prose a later agent reads back.
CITATION_DISCIPLINE = """\
## Writing down what you read

**Describe the behaviour and name the symbol — never the line.** A file path
plus a greppable name survives the engine moving; a line number or a pasted
excerpt goes wrong silently. A quoted diagnostic (the exact error text, its
file:line included) is quoted whole.

| instead of | write |
|---|---|
| `` `llmapi/llm_args.py:4669` `` | `enable_autotuner`, a `TorchLlmArgs` field, defaults to `True` |

The workflow's own skills are checked for this by a test; what you write is
checked only by the reader.
"""

SKILL_DISCIPLINE = """\
Load the skill(s) your dispatch names **before anything else**, and read their
`references/` by need.
"""


def decision_file_block(agent: str, decisions: Sequence[str], extra: Sequence[str] = ()) -> str:
    """Render the decision-file protocol for one role: the only way a verdict reaches the orchestrator.

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
        "## The decision file\n"
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
        "Any other value, or no file, is a turn that produced no decision.\n"
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
    ``modeling_v2=off``. An out-of-repo entry — a sibling vLLM or
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
            "**Running the in-repository implementation is expected**: under "
            "`modeling_v2=off` the engine uses it, which makes it this run's accuracy "
            "oracle. Measure against it freely."
        )
    if out_of_repo:
        running.append(
            "**Running the out-of-repository engine as a black-box oracle is "
            "expected; reading it is not.** Start its server, send it the workload "
            "and measure against what it returns."
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
        "Do not open, grep or quote them; skip one a search returns. A target derived "
        "from them would prove nothing about whether this pipeline can derive one.\n"
        "\n" + "\n\n".join(running) + "\n"
        "\n"
        "Your references are the checkpoint itself (`config.json`, the weight index, "
        "the quantization config) and the HuggingFace `transformers` implementation of "
        "this architecture. Both are outside the embargo.\n"
    )


def notes_block(notes: str) -> str:
    """Render the task's notes: the user's own guidance, verbatim, to every role."""
    if not notes or not notes.strip():
        return ""
    return (
        "## Notes from the task\n"
        "\n"
        "The user wrote these for this run. Follow them where they apply to your "
        "role; they never relax a gate, the embargo or the scope.\n"
        "\n"
        f"{notes.strip()}\n"
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


__all__ = [
    "CITATION_DISCIPLINE",
    "EVIDENCE_DISCIPLINE",
    "IN_TREE_LAYOUT",
    "MODELING_V2_SWITCH",
    "SKILL_DISCIPLINE",
    "SOURCE_BOUNDARY",
    "notes_block",
    "scope_block",
    "source_embargo_block",
    "tidy",
]
