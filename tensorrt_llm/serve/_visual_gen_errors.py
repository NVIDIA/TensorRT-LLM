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
from typing import Callable, Union

from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import ValidationError


def render_visual_gen_validation_error(
    exc: Union[RequestValidationError, ValidationError],
    create_error_response: Callable[..., Response],
) -> Response:
    """Render a VisualGen request validation failure as the 422 error envelope.

    Both entry points produce the same body: FastAPI's RequestValidationError
    when a route's declared request model rejects the JSON body, and pydantic's
    ValidationError when a handler builds its own request model (multipart,
    image edit, video). ``create_error_response`` is the server's error-response
    factory, so the body and the context-length code rule stay in one place.
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
