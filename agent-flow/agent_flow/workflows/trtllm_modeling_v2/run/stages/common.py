# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants and helpers every stage of the orchestrator shares."""

from __future__ import annotations

from pathlib import Path

from agent_flow import require_tool_call_stop_hook

# The speculative features. A unit is acceptance-gated when the features its
# target implements -- its own and every earlier unit's -- include one of
# these, so the gate runs in the unit that adds speculation AND in every unit
# after it, which keeps speculation on.
#
# Speculative decoding is distribution-preserving BY CONSTRUCTION. Rejection
# sampling holds the emitted distribution to the target model's, so a draft
# path that computes the wrong thing produces correct text more slowly: boot
# passes, accuracy passes, and the only observable that moves is speed.
# Per-prompt acceptance against `=off`, a measured floor and the previous
# tree's verdict is the detector.
#
# `core` is never gated: it has no draft path, so the comparison would be
# 1.0 against 1.0 and would pass whatever core did.
ACCEPTANCE_GATED_UNITS: frozenset[str] = frozenset({"mtp", "eagle3", "dflash", "dspark"})


# The other output-identical features, and the witness each needs. Same
# argument as ACCEPTANCE_GATED_UNITS, different observable: boot and accuracy
# both pass whether or not the feature is doing anything, so a unit with no
# third gate is a unit that can be declared green for a feature nobody built.
#
#   host_offload -> device memory after weight load, against a delta predicted
#                   from the checkpoint's own shapes, plus token identity
#   disagg       -> token identity across the prefill/decode hop
#   kv_quant     -> stored bytes per token against a declared figure, plus
#                   token identity with bf16 storage when declared exact
#
# And the two that are NOT output-identical but still need one, because the
# truncation they make is designed to stay inside the accuracy tolerance and a
# unit that makes no truncation stays inside it too:
#
#   decoder_replay / encoder_replay -> per-request row counters the target
#                   prints, against a declared window and workload
#
# Each needs something from the spec that the orchestrator cannot derive: the
# environment for the contrast leg, and for most a declared figure to judge
# against. That lives in `accuracy.witness`, per unit.
WITNESS_GATED_UNITS: dict[str, str] = {
    "host_offload": "residency",
    "disagg": "equivalence",
    "decoder_replay": "replay",
    "encoder_replay": "replay",
    "kv_quant": "kv_quant",
}


#: How many times in a row one gate may be lost to its session and re-run.
#: Each re-run waits for a heartbeat first, so this only binds on a session
#: that keeps coming back and dying again -- a substrate to repair, not one to
#: outwait.
SESSION_LOSS_RERUNS = 3


# Shown on every infrastructure failure, and pointed at the file rather than
# summarized, because the summary is the part that goes stale. A run rarely
# stops here, so the operator reading that message has no reason to know this
# document exists unless it says so at the moment they need it.
TROUBLESHOOTING_HINT = (
    "This is an infrastructure failure, not a target defect — every entry in "
    f"`{Path(__file__).resolve().parents[2] / 'TROUBLESHOOTING.md'}` read as a model problem "
    "while it was happening. Search it for the error text above."
)


#: The preventive counterpart to ``TROUBLESHOOTING.md``, written into the
#: workspace and referenced by every dispatch prompt.
#:
#: The two are not the same document and must not be merged.
#: ``TROUBLESHOOTING.md`` is keyed on the verbatim error text a run already
#: hit; this is keyed on nothing, because it is meant to be read *before*
#: anything fails. It carries the host facts agents otherwise re-derive one
#: after another — pytest's exit code, the ``TMPDIR`` bound, the multi-rank
#: launch rules, which UCX line is noise, what ``$?`` means after a pipe,
#: where a venv interpreter resolves.
HOST_FACTS_FILENAME = "host_facts.md"


#: Everything below this line in ``host_facts.md`` belongs to the roles.
#: Preflight runs again on every resume, so the seeded half is rewritten
#: each time and anything a role learned would otherwise be erased by the
#: next restart — which is the same defect this file exists to fix, one
#: level down.
HOST_FACTS_APPEND_MARKER = "<!-- roles append below this line; nothing above it is preserved -->"


class _SessionLost(RuntimeError):
    """A gate was lost to its session and no session came back; the run closes resumably."""


def _compose_required_tools_hooks(required_tools: list[str]) -> dict | None:
    """Stop hooks requiring *every* listed tool, not just one of them.

    ``require_tool_call_stop_hook`` enforces "at least one of these names was
    called". One independent hook per tool therefore yields AND semantics.
    """
    if not required_tools:
        return None
    merged: dict[str, list] = {"Stop": []}
    for name in required_tools:
        merged["Stop"].extend(require_tool_call_stop_hook([name])["Stop"])
    return merged


def _same_tree(a: Path, b: Path) -> bool:
    """Whether directory ``b`` holds exactly ``a``'s files with the same bytes."""
    import filecmp

    if not b.is_dir():
        return False
    files_a = sorted(
        p.relative_to(a) for p in a.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    files_b = sorted(
        p.relative_to(b) for p in b.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    )
    return files_a == files_b and all(filecmp.cmp(a / f, b / f, shallow=False) for f in files_a)
