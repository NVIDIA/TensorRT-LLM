# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared state and helpers of the VisualGen serving classes."""

import asyncio
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fastapi.responses import Response

from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.api.visual_gen.utils import \
    local_media_path_is_disallowed

if TYPE_CHECKING:
    from tensorrt_llm._utils import AdjustedSteadyClock
    from tensorrt_llm.visual_gen.params import VisualGenParams
    from tensorrt_llm.visual_gen.visual_gen import VisualGen, VisualGenResult


class VisualGenBaseServing(BaseServing):
    """Shared state and helpers of the VisualGen serving classes."""

    def __init__(self, *, generator: "VisualGen", model: str,
                 media_storage_path: Path,
                 adjusted_clock: "AdjustedSteadyClock") -> None:
        self.generator = generator
        self.model = model
        self.media_storage_path = media_storage_path
        self._adjusted_steady_clock = adjusted_clock

    async def _generate(self, prompt: str,
                        params: "VisualGenParams") -> "VisualGenResult":
        """Submit one generation off the event loop; return its handle.

        ``generate_async`` resolves and enqueues synchronously, so it runs in a
        worker thread. Callers await the handle (``await handle.aresult()``) or
        hand it to a background task.
        """
        return await asyncio.to_thread(self.generator.generate_async, prompt,
                                       params)

    def _reject_disabled_path(
            self, response_format: Optional[str]) -> Optional[Response]:
        """Return a 400 when ``response_format='path'`` but it is disabled.

        Shares the switch with ``format='path'`` on the request side; see
        :func:`local_media_path_is_disallowed`. Returns ``None`` when allowed.
        """
        if response_format != "path":
            return None
        if local_media_path_is_disallowed():
            return self.create_error_response(
                "response_format='path' is disabled on this server "
                "(TRTLLM_DISALLOW_LOCAL_MEDIA_PATH=1); it returns "
                "server-side filesystem paths and is only meaningful for "
                "co-located clients.",
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
            )
        return None
