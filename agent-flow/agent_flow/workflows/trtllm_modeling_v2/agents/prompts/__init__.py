# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything a worker is told, built in one place.

One turn's message is :func:`dispatch`: the base every worker reads (``worker_base``),
its task's section (``survey``, ``kernel``, ``core``, ``deployment``, ``fix``,
``config``), for a deployment the one card its options ask for (:func:`cards`), what
the run knows that the task needs (``run``), and the receipt. The texts are the
templates beside this module; the pipeline supplies the facts (``run/pipeline.py``) and
writes no prose of its own. A template that names a fact nobody supplied raises: a
prompt is never sent with a hole in it.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from jinja2 import Environment, PackageLoader, StrictUndefined

from ...operators.coverage import MAX_ROW_CELLS, MODELING_V2_PACKAGE
from ...spec.deployments import DISAGGREGATED, Deployment

#: Every role's system prompt: what a turn is told is in its message.
SYSTEM = (
    "You are a worker in a trtllm-modeling-v2 run. Your one task, its rules and the path "
    "of the receipt you end with are in the message you receive. Follow the repository's "
    "`AGENTS.md`."
)
#: The guides the texts point at, repo-relative.
MV2_README = f"{MODELING_V2_PACKAGE}/README.md"
BUILD_GUIDE = "cpp/tensorrt_llm/modeling_v2/README.md"
#: Where a kernel's implementation comes from, cheapest first.
TIERS: dict[int, str] = {
    1: "an op the engine already exports as `torch.ops.trtllm.*`",
    2: (
        "a callable already in this repository (the engine's modules, its attention "
        "backends, custom ops, or another target's `ops/`, copied into this target's own)"
    ),
    3: (
        "a kernel of a library the engine pins (FlashInfer), called through its public "
        "Python API; read that API in the run's interpreter"
    ),
    4: "no implementation exists: compose one from torch ops; correct and slow is accepted",
    5: (
        "torch cannot express it within the check's budget: a Triton kernel, adapted from "
        "an existing one where there is one"
    ),
}
#: What a kind's receipt carries beyond the keys every receipt has.
RECEIPT_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    "kernel": (
        ("entry", "the entry's path as committed"),
        ("tier", "the sourcing tier its `__call__` wraps, 1 to 5"),
    ),
}
#: What a template may name and a task need not have.
_UNSAID: dict[str, Any] = dict(
    worktree="",
    repo="",
    target="",
    gate="",
    repair="",
    note="",
    behind="",
    head="",
    history=(),
    earlier=0,
    host_facts="",
    rebuilt=(),
    kept="",
    python="",
    prefix="",
    launch=False,
    devices="",
    text_only=False,
    spec_notes="",
    card="",
)

_TEMPLATES = Environment(
    loader=PackageLoader(__package__, "templates"),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)


def render(template: str, /, **facts: Any) -> str:
    """One packaged template with ``facts``; a fact it names and lacks raises."""
    return _TEMPLATES.get_template(template).render(**facts)


def cards(deployment: Deployment) -> list[str]:
    """The cards ``deployment`` asks for by what it states; a turn loads the first."""
    servers = list(deployment.servers.values())
    caches = [s.options.get("kv_cache_config") or {} for s in servers]
    asked = {
        "disaggregated": deployment.kind == DISAGGREGATED,
        "multi_node": any(s.launch.strip() for s in servers),
        "drafter": any(s.options.get("speculative_config") for s in servers),
        "multimodal": "image" in deployment.inputs,
        "kv_dtype": any(str(c.get("dtype") or "auto").lower() != "auto" for c in caches),
        "state_dtype": any(
            str(c.get("mamba_ssm_cache_dtype") or "auto").lower() != "auto" for c in caches
        ),
    }
    return [name for name, wanted in asked.items() if wanted]


def repair(**facts: Any) -> str:
    """What a worker whose finished turn did not count is told: one short turn to put it right.

    ``problem``: its receipt is not valid. ``base`` and ``files``: its branch conflicts
    with the checkout's head. ``detail``: its worktree cannot be integrated as it stands.
    """
    said = {"problem": "", "base": "", "files": "", "detail": "", "tree": "", **facts}
    return render("repair.md.j2", **said).strip()


def dispatch(kind: str, facts: Mapping[str, Any]) -> str:
    """One turn's whole message for a task of ``kind``, from the pipeline's ``facts``."""
    named = list(facts.get("cards") or ())
    said = {
        **_UNSAID,
        "mv2_readme": MV2_README,
        "build_guide": BUILD_GUIDE,
        **facts,
        "card": bool(named),
        "extra": RECEIPT_KEYS.get(kind, ()),
        "ladder": "\n".join(f"     {tier}. {text}" for tier, text in TIERS.items()),
        "max_cells": MAX_ROW_CELLS,
    }
    said["tier_text"] = TIERS.get(said.get("tier"), "")
    parts = [render("worker_base.md.j2", **said), render(f"{kind}.md.j2", **said)]
    if named:
        parts.append(render(f"cards/{named[0]}.md.j2", **said))
    if named[1:]:
        others = ", ".join(f"`{name}`" for name in named[1:])
        parts.append(
            f"This deployment also asks for what these cards cover: {others}. Read them "
            f"under `{said['package']}/agents/prompts/templates/cards/` where your work "
            "touches them."
        )
    parts += [render("run.md.j2", **said), render("receipt.md.j2", **said)]
    text = "\n\n".join(part.strip() for part in parts if part.strip())
    return re.sub(r"\n{3,}", "\n\n", text) + "\n"
