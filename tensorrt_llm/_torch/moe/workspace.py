# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-forward scopes for native CUTLASS scratch reclamation."""

from contextlib import contextmanager
from itertools import count
from typing import Iterator

import torch

from ..modules.multi_stream_utils import do_multi_stream

_owner_ids = count(1)


class CutlassWorkspaceReclaimer:
    """Discover runners in native calls, once per runner/stream per forward."""

    def __init__(self) -> None:
        self._scope = torch.classes.trtllm.EagerMoeWorkspaceScope(next(_owner_ids))
        self._begin = self._scope.begin
        self._finish = self._scope.finish
        self._has_runners = True

    @contextmanager
    def forward(self, *, warmup: bool, num_tokens: int = 0) -> Iterator[None]:
        # A runner first seen after warmup is ineligible anyway. Avoid native
        # boundaries for models whose warmup never called CUTLASS MoE.
        if (
            (not warmup and not self._has_runners)
            or do_multi_stream()
            or torch.cuda.is_current_stream_capturing()
        ):
            yield
            return
        self._begin(warmup, num_tokens)
        completed = False
        try:
            yield
            completed = True
        finally:
            self._has_runners = self._finish(completed)
