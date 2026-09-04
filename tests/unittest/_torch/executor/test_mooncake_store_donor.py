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
"""Unit tests for lending a node's host memory to a Mooncake pool.

Runs without a Mooncake installation and without a GPU. The store is a fake
recording what `setup` was called with, since the contract under test is what
the donor asks Mooncake for and how long it holds it.
"""

import sys
from types import ModuleType

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import donor as donor_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.donor import (
    DEFAULT_DONOR_LOCAL_BUFFER_SIZE,
    donate_segment,
    maybe_donate_segment,
)
from tensorrt_llm.llmapi.llm_args import MooncakeDonationConfig

GIB = 1024**3


class FakeStore:
    """The slice of `MooncakeDistributedStore` a donor drives."""

    instances = []

    def __init__(self):
        self.setup_args = None
        self.status = 0
        FakeStore.instances.append(self)

    def setup(self, *args):
        self.setup_args = args
        return self.status


@pytest.fixture
def fake_bindings(monkeypatch):
    """Stand in for `mooncake.store`, which is not installed here."""
    FakeStore.instances = []
    package = ModuleType("mooncake")
    store = ModuleType("mooncake.store")
    store.MooncakeDistributedStore = FakeStore
    package.store = store
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store)
    return FakeStore


@pytest.fixture
def failing_bindings(fake_bindings):
    """Bindings whose `setup` refuses, as an unreachable master would."""

    class Refusing(fake_bindings):

        def setup(self, *args):
            super().setup(*args)
            return 7

    sys.modules["mooncake.store"].MooncakeDistributedStore = Refusing
    return Refusing


def test_a_donor_registers_the_segment_it_was_asked_for(fake_bindings):
    with donate_segment(
            "10.0.0.1:50051",
            32 * GIB,
            protocol="rdma",
            device_name="mlx5_0",
            metadata_server="P2PHANDSHAKE",
            hostname="10.0.0.5",
    ) as host:
        assert host == "10.0.0.5"
        (
            registered_host,
            metadata_server,
            segment_size,
            local_buffer_size,
            protocol,
            device_name,
            master,
        ) = fake_bindings.instances[0].setup_args

    assert registered_host == "10.0.0.5"
    assert metadata_server == "P2PHANDSHAKE"
    assert segment_size == 32 * GIB
    assert protocol == "rdma"
    assert device_name == "mlx5_0"
    assert master == "10.0.0.1:50051"
    assert local_buffer_size == DEFAULT_DONOR_LOCAL_BUFFER_SIZE


def test_a_donor_that_cannot_join_says_which_master_it_could_not_reach(failing_bindings):
    with pytest.raises(RuntimeError, match="status 7"):
        with donate_segment("10.0.0.1:50051", GIB, hostname="10.0.0.5"):
            pytest.fail("donation should not have yielded")


def test_a_donor_given_no_host_registers_under_the_pool_s_view_of_this_node(
        fake_bindings, monkeypatch):
    """The master and the segments registering with it must agree on the host."""
    monkeypatch.setattr(donor_module, "local_address", lambda: "10.1.2.3")

    with donate_segment("10.0.0.1:50051", GIB) as host:
        assert host == "10.1.2.3"
        assert fake_bindings.instances[0].setup_args[0] == "10.1.2.3"


def test_missing_bindings_are_reported_as_the_separate_component_they_are(monkeypatch):
    """The container's C++ transfer engine is not these Python bindings."""
    monkeypatch.setitem(sys.modules, "mooncake", None)
    monkeypatch.setitem(sys.modules, "mooncake.store", None)

    with pytest.raises(ImportError, match="mooncake-transfer-engine"):
        with donate_segment("10.0.0.1:50051", GIB):
            pytest.fail("donation should not have yielded")


@pytest.fixture
def reachable_master(monkeypatch):
    """Skip the socket probe: these tests are about what donation asks for."""
    monkeypatch.setattr(donor_module, "wait_for_master", lambda address: 0.0)


def test_a_server_that_was_asked_to_lend_memory_does(fake_bindings, reachable_master,
                                                     monkeypatch):
    """The config-driven path is what makes a generation server a donor."""
    monkeypatch.setattr(donor_module, "local_address", lambda: "10.1.2.3")
    donation = MooncakeDonationConfig(
        master_server_address="10.0.0.1:50051",
        segment_size="320GiB",
        protocol="rdma",
        device_name="mlx5_1",
    )

    with maybe_donate_segment(donation) as host:
        assert host == "10.1.2.3"
        registered_host, metadata_server, segment_size, _, protocol, device, master = (
            fake_bindings.instances[0].setup_args)

    assert registered_host == "10.1.2.3"
    assert metadata_server == "P2PHANDSHAKE"
    # A size string reaching Mooncake unparsed would be a segment of nothing.
    assert segment_size == 320 * GIB
    assert protocol == "rdma"
    assert device == "mlx5_1"
    assert master == "10.0.0.1:50051"


def test_a_server_that_was_not_asked_lends_nothing(fake_bindings):
    """Every deployment that does not lend memory takes this path."""
    with maybe_donate_segment(None) as host:
        assert host is None
    assert fake_bindings.instances == []


def test_a_published_master_address_is_read_before_joining(fake_bindings, reachable_master,
                                                           tmp_path):
    """What lets a generation server name a path instead of a scheduler's choice."""
    address_file = tmp_path / "master.addr"
    address_file.write_text("10.0.0.9:50051\n")
    donation = MooncakeDonationConfig(
        master_server_address=f"file://{address_file}",
        segment_size=GIB,
    )

    with maybe_donate_segment(donation):
        assert fake_bindings.instances[0].setup_args[6] == "10.0.0.9:50051"


def test_an_unreachable_master_is_reported_before_the_segment_is_offered(
        fake_bindings, monkeypatch):
    """Otherwise this is a status code from setup, with no address in it."""

    def refuse(address):
        raise TimeoutError(f"The Mooncake master at {address} did not accept connections")

    monkeypatch.setattr(donor_module, "wait_for_master", refuse)
    donation = MooncakeDonationConfig(master_server_address="10.0.0.1:50051")

    with pytest.raises(TimeoutError, match="10.0.0.1:50051"):
        with maybe_donate_segment(donation):
            pytest.fail("donation should not have yielded")
    assert fake_bindings.instances == []
