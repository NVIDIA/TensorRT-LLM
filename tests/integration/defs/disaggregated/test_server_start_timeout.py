# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPU-free unit tests for disaggregated startup helpers.

pytest -sv disaggregated/test_server_start_timeout.py
"""

from unittest.mock import MagicMock

import disagg_test_utils
from disagg_test_utils import _start_etcd, resolve_server_start_timeout


def _config(ctx_pp: int, gen_pp: int) -> dict:
    return {
        "context_servers": {"pipeline_parallel_size": ctx_pp},
        "generation_servers": {"pipeline_parallel_size": gen_pp},
    }


def test_resolve_server_start_timeout():
    """900s for PP on an SM100-family GPU, 300s otherwise; an explicit value wins.

    Budgets are written out as numbers so that a change to either constant fails here.
    """
    top_level_pp2 = {"pipeline_parallel_size": 2, "context_servers": {}, "generation_servers": {}}
    server_pp1_overrides = {
        "pipeline_parallel_size": 2,
        "context_servers": {"pipeline_parallel_size": 1},
        "generation_servers": {"pipeline_parallel_size": 1},
    }
    # (case, config, explicit budget, SM version, expected budget)
    cases = [
        ("ctx PP2 on SM100", _config(2, 1), None, 100, 900),
        ("gen PP2 on SM103", _config(1, 2), None, 103, 900),
        ("PP4 on SM107", _config(4, 4), None, 107, 900),
        ("PP4 on SM90", _config(4, 4), None, 90, 300),
        ("PP4 on SM120", _config(4, 4), None, 120, 300),
        ("no PP on SM100", _config(1, 1), None, 100, 300),
        ("no server sections", {}, None, 100, 300),
        ("top-level PP inherited", top_level_pp2, None, 100, 900),
        ("server PP1 overrides top-level", server_pp1_overrides, None, 100, 300),
        ("explicit below PP budget", _config(4, 4), 600, 100, 600),
        ("explicit above default", _config(1, 1), 1200, 100, 1200),
    ]
    failures = [
        f"{case}: got {actual}s, expected {expected}s"
        for case, config, explicit, sm_version, expected in cases
        if (actual := resolve_server_start_timeout(config, explicit, sm_version)) != expected
    ]
    assert not failures, "\n".join(failures)


def _healthy_session(peer_url: str) -> MagicMock:
    session = MagicMock()
    session.__enter__.return_value = session
    session.get.return_value.status_code = 200
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = {"members": [{"peerURLs": [peer_url]}]}
    return session


def test_start_etcd_rejects_foreign_health_endpoint(monkeypatch) -> None:
    """A healthy foreign member must not complete this attempt's readiness check."""
    foreign_session = _healthy_session("http://127.0.0.1:2380")
    owned_session = _healthy_session("http://127.0.0.1:103")
    sessions = iter([foreign_session, owned_session])
    first_process = MagicMock()
    first_process.poll.side_effect = [None, 1]
    first_process.returncode = 1
    owned_process = MagicMock()
    owned_process.poll.return_value = None
    owned_process.returncode = None
    stop = MagicMock()

    monkeypatch.setattr(disagg_test_utils.requests, "Session", lambda: next(sessions))
    monkeypatch.setattr(
        disagg_test_utils.subprocess, "Popen", MagicMock(side_effect=[first_process, owned_process])
    )
    monkeypatch.setattr(
        disagg_test_utils, "get_free_port", MagicMock(side_effect=[100, 101, 102, 103])
    )
    monkeypatch.setattr(disagg_test_utils, "_stop_etcd", stop)

    process, uri, _ = _start_etcd("/unused")

    assert process is owned_process
    assert uri == "etcd://127.0.0.1:102"
    stop.assert_called_once()
    assert stop.call_args.args[0] is first_process
    foreign_session.post.assert_called_once()
    owned_session.post.assert_called_once()
