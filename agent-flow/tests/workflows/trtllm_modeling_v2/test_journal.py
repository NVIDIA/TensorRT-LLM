# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch numbers are unique across processes and survive a restart."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.run.journal import Journal


def _journal(tmp_path: Path) -> Journal:
    journal = Journal(path=tmp_path / "journal.md")
    journal.open(slug="t")
    return journal


def test_a_claimed_number_is_never_handed_out_again(tmp_path):
    """Even when its report was never written: the claim, not the report, owns it."""
    journal = _journal(tmp_path)
    claimed = journal.next_ordinal()
    assert journal.next_ordinal() == claimed + 1
    assert Journal(path=tmp_path / "journal.md").next_ordinal() == claimed + 2


def _dispatch_from_process(path: str, index: int) -> None:
    journal = Journal(path=Path(path))
    for n in range(6):
        journal.record_dispatch("kernel_writer", f"p{index} entry {n}", "prompt")


def test_concurrent_dispatches_in_processes_get_distinct_numbers(tmp_path):
    """Two orchestrator processes on one workspace, or a resume racing a dying run."""
    journal = _journal(tmp_path)
    context = multiprocessing.get_context("fork")
    procs = [
        context.Process(target=_dispatch_from_process, args=(str(journal.path), i))
        for i in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    numbers = sorted(int(p.name.split("-")[0]) for p in journal.reports_dir.glob("*.md"))
    assert numbers == list(range(1, 25))


def test_a_resumed_journal_keeps_counting(tmp_path):
    journal = _journal(tmp_path)
    journal.record_dispatch("kernel_writer", "a", "p")
    resumed = Journal(path=journal.path)
    resumed.open(slug="t")
    assert resumed.next_ordinal() == 2
