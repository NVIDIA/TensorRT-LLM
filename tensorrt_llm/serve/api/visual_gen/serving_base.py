# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from http import HTTPStatus
from typing import Optional

from fastapi.responses import Response

from tensorrt_llm.serve.api.visual_gen.utils import \
    local_media_path_is_disallowed


class VisualGenBaseServing:

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
