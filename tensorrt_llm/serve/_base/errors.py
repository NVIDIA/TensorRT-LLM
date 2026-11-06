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
"""Typed errors of the serving core.

The core raises them; HTTP wrappers and wire dialects render them. ServingError keeps the HTTP
status apart from the error body, because ErrorResponse.code is not always a status (an
over-length prompt carries "context_length_exceeded").
"""

from http import HTTPStatus
from typing import Union

from tensorrt_llm.executor.utils import (
    CONTEXT_LENGTH_EXCEEDED_CODE,
    is_context_length_exceeded_message,
)
from tensorrt_llm.serve.openai_protocol import ErrorResponse


def error_body(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    """Build the OpenAI error body for a rejected request.

    ``code`` is the HTTP status, except for an over-length prompt, which is recognized by its
    text because it crosses the executor IPC boundary as a plain string (RequestError), so no
    exception type survives to here.
    """
    code: Union[int, str] = status_code.value
    if is_context_length_exceeded_message(message):
        code = CONTEXT_LENGTH_EXCEEDED_CODE
    return ErrorResponse(message=message, type=err_type, code=code)


class ServingError(Exception):
    """A request the serving core rejects, with the HTTP status to answer it with.

    Attributes:
        status: The HTTP status code.
        body: The OpenAI error body; ``body.code`` is not necessarily ``status``.
    """

    def __init__(self, status: int, body: ErrorResponse) -> None:
        super().__init__(body.message)
        self.status = status
        self.body = body


def serving_error(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ServingError:
    """Return the ServingError for what OpenAIServer.create_error_response would render."""
    return ServingError(status_code.value, error_body(message, err_type, status_code))


class EngineFatalError(Exception):
    """The engine failed fatally; SIGINT has been sent and the server is shutting down.

    It carries no response; each caller decides what to answer.
    """
