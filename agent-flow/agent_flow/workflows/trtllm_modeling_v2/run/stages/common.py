# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants and helpers every stage of the orchestrator shares."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_flow import require_tool_call_stop_hook

#: The speculative features. A unit is acceptance-gated when the features its
#: target implements (its own and every earlier unit's) include one of these.
#: Speculative decoding preserves the target's output distribution by
#: construction, so a draft path that computes the wrong thing passes boot and
#: accuracy and shows only as speed: per-prompt acceptance is the detector.
#: `core` has no draft path and is never gated.
ACCEPTANCE_GATED_UNITS: frozenset[str] = frozenset({"mtp", "eagle3", "dflash", "dspark"})


#: The units boot and accuracy pass whether or not their feature works, and the
#: witness each needs:
#:
#:   host_offload -> device memory after weight load, against a delta predicted
#:                   from the checkpoint's own shapes, plus token identity
#:   disagg       -> token identity across the prefill/decode hop
#:   kv_quant     -> stored bytes per token against a declared figure, plus
#:                   token identity with bf16 storage when declared exact
#:   decoder_replay / encoder_replay -> per-request row counters the target
#:                   prints, against a declared window and workload
#:
#: What the orchestrator cannot derive (the contrast leg's environment, a
#: declared figure) is in the spec's `accuracy.witness`, per unit.
WITNESS_GATED_UNITS: dict[str, str] = {
    "host_offload": "residency",
    "disagg": "equivalence",
    "decoder_replay": "replay",
    "encoder_replay": "replay",
    "kv_quant": "kv_quant",
}


#: How many times in a row one gate may be lost to its session and re-run.
#: Each re-run waits for a heartbeat first, so this only binds on a session
#: that keeps coming back and dying again.
SESSION_LOSS_RERUNS = 3


#: Shown on every infrastructure failure: the troubleshooting guide, by path.
TROUBLESHOOTING_HINT = (
    "This is an infrastructure failure, not a target defect — every entry in "
    f"`{Path(__file__).resolve().parents[2] / 'TROUBLESHOOTING.md'}` read as a model problem "
    "while it was happening. Search it for the error text above."
)


#: The host facts written into the workspace and referenced by every dispatch
#: prompt: read before anything fails, where ``TROUBLESHOOTING.md`` is keyed on
#: the error text of a failure that already happened.
HOST_FACTS_FILENAME = "host_facts.md"


#: Everything below this line in ``host_facts.md`` belongs to the roles.
#: Preflight rewrites the part above it on every start, a resume included.
HOST_FACTS_APPEND_MARKER = "<!-- roles append below this line; nothing above it is preserved -->"


class _SessionLost(RuntimeError):
    """A gate was lost to its session and no session came back; the run closes resumably."""


def _compose_required_tools_hooks(required_tools: list[str]) -> dict | None:
    """Stop hooks requiring *every* listed tool: one ``require_tool_call_stop_hook`` per tool."""
    if not required_tools:
        return None
    merged: dict[str, list] = {"Stop": []}
    for name in required_tools:
        merged["Stop"].extend(require_tool_call_stop_hook([name])["Stop"])
    return merged


def remote_turns(owner: Any):
    """``owner``'s turn executor when another session answers its turns, else ``None``.

    Such turns outlive this process (the external transport): a relaunch adopts
    them, and the run marks them consumed once it is done with them.
    """
    turns = getattr(owner, "_turns", None)
    return turns if getattr(turns, "remote", False) else None


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
