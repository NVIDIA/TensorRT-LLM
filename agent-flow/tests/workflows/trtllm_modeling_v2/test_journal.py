# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch numbers are unique across threads and processes.

A smith batch records its dispatches concurrently. Reading the highest
number and then writing the report would let two smiths take the same
``NN`` and leave duplicate dispatch numbers in the journal.
"""

from __future__ import annotations

import multiprocessing
import re
import threading
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.run.journal import Journal


def _journal(tmp_path: Path) -> Journal:
    journal = Journal(path=tmp_path / "journal.md")
    journal.open(slug="t")
    return journal


def test_ordinals_are_sequential_and_follow_existing_reports(tmp_path):
    journal = _journal(tmp_path)
    (journal.reports_dir / "07-smith-old.md").write_text("x", encoding="utf-8")
    assert journal.next_ordinal() == 8
    assert journal.next_ordinal() == 9
    path = journal.record_dispatch("assembler", "core", "prompt")
    assert path.name.startswith("10-assembler-")


def test_a_claimed_number_is_never_handed_out_again(tmp_path):
    """Even when its report was never written: the claim, not the report, owns it."""
    journal = _journal(tmp_path)
    claimed = journal.next_ordinal()
    assert journal.next_ordinal() == claimed + 1
    assert Journal(path=tmp_path / "journal.md").next_ordinal() == claimed + 2


def test_concurrent_dispatches_in_threads_get_distinct_numbers(tmp_path):
    journal = _journal(tmp_path)
    barrier = threading.Barrier(24)

    def dispatch(index: int) -> None:
        barrier.wait()
        journal.record_dispatch("smith", f"entry {index}", "prompt")

    threads = [threading.Thread(target=dispatch, args=(i,)) for i in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    numbers = sorted(int(p.name.split("-")[0]) for p in journal.reports_dir.glob("*.md"))
    assert numbers == list(range(1, 25))
    journaled = re.findall(r"dispatch (\d+) · smith", journal.path.read_text(encoding="utf-8"))
    assert len(set(journaled)) == 24


def _dispatch_from_process(path: str, index: int) -> None:
    journal = Journal(path=Path(path))
    for n in range(6):
        journal.record_dispatch("smith", f"p{index} entry {n}", "prompt")


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


def test_latest_report_ignores_the_claims(tmp_path):
    journal = _journal(tmp_path)
    first = journal.record_dispatch("smith", "norm x", "p")
    second = journal.record_dispatch("smith", "norm x", "p")
    assert journal.latest_report("smith", "norm x") == second != first


def test_the_reports_directory_holds_only_reports(tmp_path):
    """Readers list ``reports/`` and open every entry; the claims live beside it."""
    journal = _journal(tmp_path)
    journal.record_dispatch("smith", "norm x", "p")
    assert [p.is_file() for p in journal.reports_dir.iterdir()] == [True]


def test_a_new_journal_numbers_from_one_again(tmp_path):
    """``--clean`` removes the journal and the reports; the claims go with them."""
    journal = _journal(tmp_path)
    journal.record_dispatch("smith", "a", "p")
    journal.record_dispatch("smith", "b", "p")
    journal.path.unlink()
    for report in journal.reports_dir.iterdir():
        report.unlink()
    fresh = _journal(tmp_path)
    assert fresh.next_ordinal() == 1


def test_a_resumed_journal_keeps_counting(tmp_path):
    journal = _journal(tmp_path)
    journal.record_dispatch("smith", "a", "p")
    resumed = Journal(path=journal.path)
    resumed.open(slug="t")
    assert resumed.next_ordinal() == 2
