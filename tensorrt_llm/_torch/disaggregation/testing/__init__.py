# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test tooling for store drivers, shipped so a driver's own CI can import it from the wheel.

``driver_conformance`` holds the checks every ``StoreDriver`` must pass and the harness that runs
them; ``memory_store`` holds the in-process store and the drivers over it that stand in for a
real store, one registering regions singly and one in batches; ``broken_drivers`` holds drivers
that each break one rule, proving the checks bite; ``host_memory`` and ``object_trailer`` are
their helpers.

This package depends on pytest and is imported by tests only. No engine module imports it, and a
test guards that. It provides no public configuration: the in-memory driver cannot be selected
in a deployment.
"""

import pytest

# The checks are library modules, which pytest does not rewrite on its own; registered here, a
# failing check shows the values it compared rather than a bare AssertionError.
pytest.register_assert_rewrite(
    "tensorrt_llm._torch.disaggregation.testing.driver_conformance",
    "tensorrt_llm._torch.disaggregation.testing.broken_drivers",
)
