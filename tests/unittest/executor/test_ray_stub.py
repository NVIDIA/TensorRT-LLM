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
"""The stand-in for ``ray`` used when Ray is not installed.

It must be inert at import and decoration time and fail only when Ray
functionality is actually used, so that a default install can still import
``tensorrt_llm``.
"""

import importlib.util

import pytest

from tensorrt_llm.executor.ray import stub

# This file's home is the CPU-Generic stage, because that is where Ray is absent
# and the stub is the thing actually exercised (l0_cpu.yml).  Those stages run
# `pytest -m cpu_only` (jenkins/L0_Test.groovy:1476), and their conftest ignores
# any test file whose text lacks the literal string "pytest.mark.cpu_only"
# (tests/unittest/conftest.py:239).  Without this marker all six tests are
# deselected, pytest exits 5 (no tests collected), and the test_unittests_v2
# wrapper reports that as a failure rather than as an empty run.
pytestmark = pytest.mark.cpu_only

_RAY_INSTALLED = importlib.util.find_spec("ray") is not None


def test_import_is_inert() -> None:
    """Importing the stub must not raise; only *using* Ray may."""
    assert stub.remote is not None


def test_bare_decorator_defers_the_failure() -> None:
    """``@ray.remote`` must decorate cleanly and fail only when called."""

    @stub.remote
    def train_step(x: int) -> int:
        return x

    assert train_step.__name__ == "train_step"

    with pytest.raises(RuntimeError, match="train_step"):
        train_step(1)


def test_called_decorator_defers_the_failure() -> None:
    """``@ray.remote(...)`` -- the parameterised form -- behaves the same."""

    @stub.remote(num_gpus=1)
    class Worker:
        def run(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="Worker"):
        Worker()


def test_unknown_attribute_raises_and_names_itself() -> None:
    """Any other ``ray.<name>`` must raise, and say which name was wanted."""
    with pytest.raises(RuntimeError, match=r"ray\.init"):
        getattr(stub, "init")


def test_unknown_attribute_says_to_install_ray_when_ray_was_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``TLLM_DISABLE_MPI=1`` the user asked for Ray, so say so."""
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    with pytest.raises(RuntimeError, match="Please install Ray"):
        getattr(stub, "init")


@pytest.mark.skipif(_RAY_INSTALLED, reason="Ray is installed, so the fallback is not taken here")
def test_distributed_layer_falls_back_to_the_stub() -> None:
    """Without Ray, the distributed layer must import and resolve to the stub."""
    from tensorrt_llm._torch.distributed import communicator

    assert communicator.ray.__name__ == "tensorrt_llm.executor.ray.stub"
