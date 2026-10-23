# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A store driver with the shape the KVCR driver will have, for the backend tests.

KVCR opens its client with the memory it will use and closes the client to let the memory go, so
the driver registers regions together, one batch for as long as the client is open. Its gets take
a hint naming the peer to read from. It has no cancel and no probe. Its drain waits for its own
operations only, never for the registrations, which go when the batch is revoked; a drain that
waited for them would wait on the revocation that follows it. Over the in-memory store, the driver
keeps every control of ``TogetherMemoryDriver``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tensorrt_llm._torch.disaggregation.blob.store import (
    DriverCapabilities,
    DriverRejected,
    RegionSpec,
)
from tensorrt_llm._torch.disaggregation.testing.memory_store import (
    MemoryStore,
    TogetherMemoryDriver,
    _Batch,
)

KVCR_LIKE_CAPABILITIES = DriverCapabilities(
    get=True,
    put=True,
    cancel=False,
    probe=False,
    max_rows_per_op=None,
    hints=True,
    absent_is_miss=False,
)


class TogetherDriver(TogetherMemoryDriver):
    """``TogetherMemoryDriver`` with KVCR's shape: one live batch at a time, hints taken and
    recorded in ``hints_seen``, no cancel and no probe."""

    def __init__(self, store: MemoryStore, **options: Any) -> None:
        options.setdefault("capabilities", KVCR_LIKE_CAPABILITIES)
        super().__init__(store, **options)

    def register_regions(self, specs: Sequence[RegionSpec]) -> _Batch:
        with self._lock:
            if self.live_batches():
                raise DriverRejected(
                    "the client is open with its memory already: one batch at a time"
                )
        return super().register_regions(specs)
