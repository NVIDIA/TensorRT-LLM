# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""asyncio helpers shared by the OpenEngine client modules."""

from __future__ import annotations

import asyncio
from typing import Any


def discard(future: asyncio.Future[Any]) -> None:
    """Drop a pending or finished future without leaving an unretrieved exception."""
    if future.done():
        if not future.cancelled():
            future.exception()
        return
    future.cancel()
    future.add_done_callback(lambda done: done.cancelled() or done.exception())
