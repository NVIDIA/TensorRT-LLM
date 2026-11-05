# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async cleanup primitives shared by serving request owners."""

import asyncio
from typing import TypeVar

_T = TypeVar("_T")


async def drain_cleanup(cleanup: asyncio.Future[_T]) -> bool:
    """Wait for owned cleanup without forwarding repeated waiter cancellation.

    Args:
        cleanup: Retained cleanup future. Use gather(return_exceptions=True)
            when the caller needs to choose between cleanup and request errors.

    Returns:
        Whether cancellation was received while waiting. The caller must inspect
        the completed future and propagate cancellation according to its policy.
    """
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    return cancelled
