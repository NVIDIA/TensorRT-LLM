# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prose blocks shared by the trtllm-modeling-v2 prompts.

Everything here binds more than one role: the doctrine every worker follows
whatever its task's tags, and the run-scoped blocks the bundle appends.
"""

from __future__ import annotations

import re
from typing import Sequence

from ...operators.coverage import CATALOG_TESTS_DIR, MODELING_V2_PACKAGE


def tidy(text: str) -> str:
    """Collapse the blank-line runs that composing blocks leaves behind."""
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


MODELING_V2_LAYOUT = f"""\
## ModelingV2: where things live

**Read `{MODELING_V2_PACKAGE}/README.md` first**: it
is the authoritative description of routing, layout, entries and gates, and
`catalog/_op.py` states the entry contract. The source-level gates (no GPU)
are `{CATALOG_TESTS_DIR}/test_modeling_v2_*.py`.

The existing targets (`{MODELING_V2_PACKAGE}/models/`,
one directory per family) are gate-verified exemplars, and reading them is
part of the work. Unless `source_embargo` names one, find the targets that
already solved what your task needs (the same attention kind, MoE routing,
sparse or recurrent state, a vision tower, the cache state that reuse,
chunking and disaggregation depend on) before you write: call or promote
their entries, and follow their structure and their tests. **Copy patterns,
never values**: every model-specific value is re-derived from this checkpoint.
"""

MODELING_V2_SWITCH = """\
## The modeling_v2 switch

The README's `modeling_v2` modes decide what a result is evidence of. The
gates carry the mode as `TRTLLM_MODELING_V2` and hand it to the LLM
constructor; a command you write yourself passes the argument. Under `off`
the built-in implementation, where the engine has one, is the natural
accuracy oracle: same harness, same workload, one variable.
"""

SOURCE_BOUNDARY = """\
## Prior art: read widely, then build on this engine

Most of what a task needs has been done somewhere. Before you write it, find
how others did it, and say in your report what you used:

- **TensorRT-LLM's tests**: the unit, accuracy and perf tests show working
  configurations, how each feature is deployed and tested, and the scores
  they reach (the reuse rule under How you work).
- **TensorRT-LLM's models**: the ModelingV2 targets and the built-in models
  (the modules that call `register_auto_model`): how a model of this family
  is wired, loads its weights, binds its cache state, takes multimodal input
  and speculates.
- **Kernel libraries**: the engine's own ops, FlashInfer, CUTLASS: the kernels
  the sourcing ladder reuses before anything is written.

`tree_map.md` in the workspace says where each part of this tree is in the
checkout under test. A path you were given that is not there has moved: find
what it defined by name (`git grep`), and trust the tree over any prompt.
- **Other inference engines** (vLLM, SGLang, ...), from a checkout, an
  installed package or the web where you have one: often the first to serve
  a new architecture, so their model, kernel and deployment code is a map of
  its ops, layouts and feature choices.

What you take lands on this engine's terms. The target's forward calls
catalog entries, so an op another implementation calls is an entry to reuse,
wrap or write (the sourcing ladder), not code pasted into the forward. Code
you adapt keeps its license header and names its origin. Values come from
this checkpoint. The HuggingFace reference stays the ground truth of the
math the gates judge: another implementation shows how, never what is
correct. Paths in `source_embargo` stay unread.
"""

EVIDENCE_DISCIPLINE = """\
## Evidence

Your decision file is what the orchestrator acts on; your report is a claim
the next agent builds on without re-checking. Same-protocol accuracy reruns
move 0.1-0.3 points; a smaller change is not a result. A benchmark you start
yourself runs as the gates do, CUDA graphs on: eager takes 10-50x longer and
scores a path the target does not serve. Eager is for a few prompts while
debugging.
"""

CITATION_DISCIPLINE = """\
## Writing down what you read

**Describe the behaviour and name the symbol — never the line.** A file path
plus a greppable name (a function, class, field, config key or exact assert
text) survives the engine moving; a line number goes wrong silently. A quoted
diagnostic (the exact error text, its file:line included) is quoted whole.
"""


def decision_file_block(agent: str, extra: Sequence[str] = ()) -> str:
    """Render the decision-file protocol for one role: the only way a verdict reaches the orchestrator.

    The decisions themselves are the role's ending section's, which comes first.
    """
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
        "decision: <exactly one of the decisions listed above>\n"
        "```\n"
        f"{extra_block}"
        "\n"
        "Any other value, or no file, is a turn that produced no decision.\n"
    )


def source_embargo_block(paths: Sequence[str]) -> str:
    """Render the run-scoped embargo on specific source files or trees.

    **The opening sentence splits on where the paths point**: an in-repo
    entry means the engine already ships this checkpoint, an out-of-repo
    entry means another engine implements it. Absolute spelling is the
    discriminator, and it is not a heuristic:
    :func:`~..task_schema._validate_source_embargo` accepts an absolute path
    only when it resolves outside ``trtllm_repo_path``.
    """
    if not paths:
        return ""
    listed = "\n".join(f"- `{p}`" for p in paths)
    out_of_repo = [p for p in paths if p.startswith("/")]
    in_repo = [p for p in paths if not p.startswith("/")]

    premise = []
    if in_repo:
        premise.append("This checkpoint already has an implementation in this repository.")
    if out_of_repo:
        premise.append(
            "This checkpoint is implemented in another engine's checkout on this "
            "filesystem, outside this repository."
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
        "Everything not listed stays open, and you are expected to use it: the "
        "checkpoint itself, the HuggingFace `transformers` implementation of this "
        "architecture, this tree's targets, models and tests, kernel libraries and other "
        "engines' implementations (Prior art).\n"
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
    "MODELING_V2_LAYOUT",
    "MODELING_V2_SWITCH",
    "SOURCE_BOUNDARY",
    "decision_file_block",
    "notes_block",
    "scope_block",
    "source_embargo_block",
    "tidy",
]
