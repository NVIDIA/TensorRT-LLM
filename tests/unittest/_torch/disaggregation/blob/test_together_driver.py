# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KVCR-shaped test driver: the shared conformance checks in batch mode, and what its shape
adds on top of the batch-registering in-memory driver."""

from __future__ import annotations

import pytest
from test_memory_driver import TestTogetherDriverConformance as _BatchConformance
from together_driver import KVCR_LIKE_CAPABILITIES, TogetherDriver

from tensorrt_llm._torch.disaggregation.blob.store import DriverRejected, RowIO, Span
from tensorrt_llm._torch.disaggregation.testing.driver_conformance import SLOT, specs_of
from tensorrt_llm._torch.disaggregation.testing.host_memory import HostBuffer
from tensorrt_llm._torch.disaggregation.testing.memory_store import MemoryStore

pytestmark = pytest.mark.cpu_only

TIMEOUT = 5.0


class TestKvcrShapedDriverConformance(_BatchConformance):
    """The suite in batch mode, with the capabilities the KVCR driver will declare."""

    driver_cls = TogetherDriver
    capabilities = KVCR_LIKE_CAPABILITIES


def test_one_batch_at_a_time_as_a_client_opened_with_its_memory() -> None:
    driver = TogetherDriver(MemoryStore())
    first, second = HostBuffer(SLOT), HostBuffer(SLOT)
    handle = driver.register_regions(specs_of([first]))
    with pytest.raises(DriverRejected, match="one batch"):
        driver.register_regions(specs_of([second]))
    driver.unregister_regions(handle)
    again = driver.register_regions(specs_of([second]))  # the client was closed: a new one opens
    driver.unregister_regions(again)
    assert driver.drain(TIMEOUT) is True
    driver.close()


def test_drain_leaves_the_registrations_to_unregister_regions() -> None:
    """A backend drains first and revokes after; a drain that waited for the registrations
    would wait on itself."""
    driver = TogetherDriver(MemoryStore())
    memory = HostBuffer(SLOT)
    handle = driver.register_regions(specs_of([memory]))
    assert driver.drain(TIMEOUT) is True
    assert driver.live_batches() == [handle]
    with pytest.raises(DriverRejected):
        driver.close()  # the batch is still live
    driver.unregister_regions(handle)
    driver.close()


def test_hints_are_taken_and_recorded() -> None:
    driver = TogetherDriver(MemoryStore())
    memory = HostBuffer(SLOT)
    handle = driver.register_regions(specs_of([memory]))
    hint = ("peer-3", 7)
    op = driver.get([RowIO("t/v1/00", (Span(memory.address, SLOT),))], hint)
    assert driver.hints_seen == [hint]
    assert driver.access_ended(op)
    driver.unregister_regions(handle)
    assert driver.drain(TIMEOUT) is True
    driver.close()
