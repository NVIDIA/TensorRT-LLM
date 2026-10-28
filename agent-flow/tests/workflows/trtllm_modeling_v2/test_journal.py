# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch numbers survive a restart."""

from __future__ import annotations

from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.run.records.journal import Journal


def _journal(tmp_path: Path) -> Journal:
    journal = Journal(path=tmp_path / "journal.md")
    journal.open(slug="t")
    return journal


def test_a_resumed_journal_keeps_counting(tmp_path):
    journal = _journal(tmp_path)
    journal.record_dispatch("kernel_writer", "a", "p")
    resumed = Journal(path=journal.path)
    resumed.open(slug="t")
    assert resumed.next_ordinal() == 2
