# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker: one system prompt, with each role's own turn ending."""

from __future__ import annotations

from ..decisions import ending_section
from ._common import MODELING_V2_NATIVE, tidy
from ._render import render_template

WORKER_BASE = render_template("worker_base.md.j2")
_KERNEL_WRITER_ENDING = render_template("kernel_writer_ending.md.j2")
_ASSEMBLER_ENDING = render_template("assembler_ending.md.j2")

#: What each worker role's decision file carries beyond the decision itself.
ENDING_EXTRAS: dict[str, str] = {
    "kernel_writer": _KERNEL_WRITER_ENDING,
    "assembler": _ASSEMBLER_ENDING,
}
#: The roles that write and commit code also read two facts about what they test.
COMMITTING_ROLES: tuple[str, ...] = ("kernel_writer", "assembler")


def worker_prompt(ending: str, *, commits: bool = True) -> str:
    """The worker's system prompt, ending the way one role's turn ends."""
    doctrine = (MODELING_V2_NATIVE,) if commits else ()
    return tidy(render_template("worker.md.j2", base=WORKER_BASE, doctrine=doctrine, ending=ending))


def role_prompt(role: str) -> str:
    """The worker's system prompt for a role whose decisions are in the decision table."""
    return worker_prompt(
        ending_section(role, ENDING_EXTRAS.get(role, "")), commits=role in COMMITTING_ROLES
    )


__all__ = ["COMMITTING_ROLES", "ENDING_EXTRAS", "WORKER_BASE", "role_prompt", "worker_prompt"]
