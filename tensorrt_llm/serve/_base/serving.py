# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base class of the serving classes under ``tensorrt_llm/serve/api/``."""

from http import HTTPStatus
from typing import Union

from fastapi.responses import JSONResponse, Response

from tensorrt_llm.executor.utils import (CONTEXT_LENGTH_EXCEEDED_CODE,
                                         is_context_length_exceeded_message)
from tensorrt_llm.serve.openai_protocol import ErrorResponse


class BaseServing:
    """Base of every serving class.

    Holds only what all serving classes share; each subclass receives its
    dependencies as explicit constructor arguments, never by reading the
    server object.
    """

    @staticmethod
    def create_error_response(
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> Response:
        # Over-length prompts are detected from the message text because they
        # cross the executor IPC boundary as plain strings (RequestError), so
        # no exception type survives to here.
        code: Union[int, str] = status_code.value
        if is_context_length_exceeded_message(message):
            code = CONTEXT_LENGTH_EXCEEDED_CODE
        error_response = ErrorResponse(message=message,
                                       type=err_type,
                                       code=code)
        return JSONResponse(content=error_response.model_dump(),
                            status_code=status_code.value)
