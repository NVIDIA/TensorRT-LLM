# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command execution, with no GPU, no scheduler and no model in sight."""

from __future__ import annotations

import time

from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import (
    TIMEOUT_RETURNCODE,
    CommandSpec,
    LocalCommandExecutor,
)


def test_a_command_that_outlives_its_deadline_is_timed_out_not_merely_failed():
    started = time.monotonic()
    record = LocalCommandExecutor().execute(CommandSpec(argv="sleep 30", timeout_s=1))
    assert record.timed_out
    assert record.returncode == TIMEOUT_RETURNCODE
    assert not record.ok
    assert time.monotonic() - started < 20
