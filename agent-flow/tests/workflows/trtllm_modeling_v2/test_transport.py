# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The external agent transport: the orchestrator's turn, delivered as a file."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.infra.dispatch import DISPATCH_DIRNAME
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import TurnRequest
from agent_flow.workflows.trtllm_modeling_v2.infra.external.transport import ExternalAgentTransport


def _turn(transport, **request):
    """One turn through the transport; its text."""
    return transport.run(TurnRequest(**request)).text


def _answer_after(transport: ExternalAgentTransport, request_id: str, result: str, delay=0.05):
    """Play the driver from another thread: the result, then ``.done`` last."""

    def run():
        time.sleep(delay)
        (transport.root / f"{request_id}.result.md").write_text(result, encoding="utf-8")
        (transport.root / f"{request_id}.done").write_text("1\n", encoding="utf-8")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_the_system_prompt_travels_with_the_turn(tmp_path):
    """A driver given only the user prompt runs a DIFFERENT agent."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    _answer_after(transport, "001-assembler", "ok")
    _turn(
        transport, role="assembler", what="core", prompt="p", system_prompt="YOU ARE THE ASSEMBLER"
    )

    root = tmp_path / DISPATCH_DIRNAME
    assert (root / "001-assembler.system.md").read_text(encoding="utf-8") == "YOU ARE THE ASSEMBLER"
    request = json.loads((root / "001-assembler.request.json").read_text(encoding="utf-8"))
    assert request["system_prompt_file"].endswith("001-assembler.system.md")


def test_concurrent_turns_get_distinct_ids(tmp_path):
    """Ten kernel writers run in one batch; two sharing an id would cross their answers."""
    transport = ExternalAgentTransport(tmp_path, poll_s=0.01)
    seen: list[str] = []
    lock = threading.Lock()

    def one():
        with lock:
            n = len(seen) + 1
        request_id = f"{n:03d}-kernel_writer"
        _answer_after(transport, request_id, request_id, delay=0.2)
        with lock:
            seen.append(request_id)

    # Drive the transport itself concurrently and check the ids it minted.
    results: list[str] = []
    threads = []
    for _ in range(5):
        one()
    for _ in range(5):
        t = threading.Thread(
            target=lambda: results.append(
                _turn(transport, role="kernel_writer", what="x", prompt="p")
            )
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=10)
    assert sorted(results) == sorted(seen)


def test_a_second_orchestrator_on_one_workspace_is_refused(tmp_path):
    """Two on one workspace is not a race anybody notices.

    They share the checkpoint file, the dispatch directory and the request-id
    counter, so they mint the SAME id, overwrite each other's request JSON,
    and both unblock on one answer.
    """
    import subprocess
    import sys
    import textwrap

    workspace = tmp_path / "ws"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
        from agent_flow.workflows.trtllm_modeling_v2.cli import _claim_workspace
        from pathlib import Path
        _claim_workspace(Path({str(workspace)!r}))
        print("HELD", flush=True)
        time.sleep(30)
        """
    )
    holder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "HELD"
        second = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
        assert second.returncode == 3
        assert "already holds" in second.stderr
    finally:
        holder.kill()
