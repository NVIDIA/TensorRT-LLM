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
"""The VisualGen 422 renderer shared by the route-level (``RequestValidationError``)
and handler-level (``ValidationError``) entry points. Private module."""

from http import HTTPStatus
from typing import Protocol

from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import ValidationError


class _ErrorResponseFactory(Protocol):
    """Keyword contract for the shared error-response factory."""

    def __call__(self, *, message: str, err_type: str, status_code: HTTPStatus) -> Response: ...


def render_visual_gen_validation_error(
    exc: RequestValidationError | ValidationError,
    create_error_response: _ErrorResponseFactory,
) -> Response:
    """Render a VisualGen request validation failure with HTTP status 422.

    Locations are dot-separated, omitting ``body`` segments. Unknown fields
    receive an ``extra_params`` hint; other errors use ``<loc>: <msg>`` or a
    bare message when the location is empty. Messages are joined with ``"; "``;
    an empty error list falls back to ``str(exc)``.

    Args:
        exc: Route-level or handler-level request validation failure.
        create_error_response: Called once by keyword with the message,
            ``BadRequestError`` type and HTTP status 422. The factory owns the
            response envelope and any message-derived error code.

    Returns:
        The response returned by the shared error-response factory.
    """
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(seg) for seg in err.get("loc", ()) if seg != "body")
        etype = err.get("type", "")
        msg = err.get("msg", "")
        if etype == "extra_forbidden":
            parts.append(
                f"Unknown request field {loc!r}. Pass model-specific "
                "parameters via 'extra_params' instead."
            )
        elif loc:
            parts.append(f"{loc}: {msg}")
        else:
            parts.append(msg)
    message = "; ".join(parts) if parts else str(exc)
    return create_error_response(
        message=message,
        err_type="BadRequestError",
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
    )
