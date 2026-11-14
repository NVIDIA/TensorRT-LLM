# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared lifetime fixtures for PyTorch runtime unit tests."""

from collections.abc import Iterator
from contextlib import ExitStack

import pytest


@pytest.fixture
def exit_stack() -> Iterator[ExitStack]:
    with ExitStack() as stack:
        yield stack
