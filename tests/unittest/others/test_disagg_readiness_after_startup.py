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
"""/health must stop promising readiness once a worker is gone.

Without a cluster manager ``is_ready()`` used to be an unconditional ``True``.
That made ``/health`` a promise about startup rather than a statement about the
workers: a ctx or gen worker could die afterwards and the coordinator kept
answering 200 for it, so a polling client had no way to learn the group was
unusable and waited out its whole timeout.

These tests exercise ``is_ready`` directly against stub routers -- no server,
no GPU, no event loop beyond ``asyncio.run``.
"""

import asyncio

import pytest

from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService


class _StubRouter:
    """Just the surface ``is_ready`` touches."""

    def __init__(self, servers):
        self._servers = list(servers)

    @property
    def servers(self):
        return self._servers

    @property
    def num_prepared_servers(self):
        return len(self._servers)


def _coordinator(ctx_servers, gen_servers, cluster_manager=None):
    """An DisaggCoordinatorService with only the readiness surface populated.

    __init__ builds sessions and config we do not need, so the object is
    created without running it -- the method under test reads three attributes.
    """
    c = object.__new__(DisaggCoordinatorService)
    c._ctx_router = _StubRouter(ctx_servers)
    c._gen_router = _StubRouter(gen_servers)
    c._disagg_cluster_manager = cluster_manager
    return c


def _ready(c):
    return asyncio.run(c.is_ready())


def test_ready_while_both_roles_have_servers():
    assert _ready(_coordinator(["ctx0"], ["gen0"])) is True


@pytest.mark.parametrize(
    "ctx,gen,gone",
    [
        ([], ["gen0"], "context"),
        (["ctx0"], [], "generation"),
        ([], [], "both"),
    ],
)
def test_not_ready_once_a_role_has_no_servers(ctx, gen, gone):
    """The regression: this returned True for every one of these."""
    assert _ready(_coordinator(ctx, gen)) is False, (
        f"coordinator reported ready with no {gone} server"
    )


def test_recovers_when_the_worker_comes_back():
    """Deliberately not sticky.

    A metadata-driven deployment adds and removes workers as a matter of
    course. Latching 'dead' on the first removal would turn a routine topology
    change into a permanently unhealthy coordinator.
    """
    c = _coordinator(["ctx0"], ["gen0"])
    assert _ready(c) is True
    c._gen_router._servers.clear()  # worker dies
    assert _ready(c) is False
    c._gen_router._servers.append("gen1")  # replacement arrives
    assert _ready(c) is True


def test_cluster_manager_path_is_untouched():
    """With a cluster manager, readiness is still delegated to it verbatim."""
    seen = {}

    class _CM:
        async def is_ready_with_router(self, n_ctx, n_gen):
            seen["args"] = (n_ctx, n_gen)
            return False  # distinct from what the fallback would say

    c = _coordinator(["ctx0", "ctx1"], ["gen0"], cluster_manager=_CM())
    assert _ready(c) is False, "the cluster manager's verdict must win"
    assert seen["args"] == (2, 1), "router counts are forwarded unchanged"


def test_static_deployment_is_unchanged():
    """No metadata server means no monitor, so the lists never shrink.

    That deployment shape keeps exactly its old behaviour -- the point of
    keying off the server lists rather than adding a new liveness source.
    """
    c = _coordinator(["ctx0"], ["gen0"])
    for _ in range(5):
        assert _ready(c) is True
