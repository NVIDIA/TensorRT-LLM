# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared VisualGen request-validation renderer."""

import json
from http import HTTPStatus

import pytest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from pydantic import ValidationError

from tensorrt_llm.serve._visual_gen_errors import render_visual_gen_validation_error
from tensorrt_llm.serve.openai_protocol import ImageGenerationRequest
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

_UNKNOWN_FIELD_BODY = {
    "object": "error",
    "message": "Unknown request field 'not_a_field'. Pass model-specific "
    "parameters via 'extra_params' instead.",
    "type": "BadRequestError",
    "param": None,
    "code": 422,
}


def _pydantic_unknown_field_error() -> ValidationError:
    """Return the Pydantic validation error for an unknown request field."""
    with pytest.raises(ValidationError) as exc_info:
        ImageGenerationRequest.model_validate({"prompt": "x", "not_a_field": 1})
    return exc_info.value


def test_renderer_accepts_both_exception_types() -> None:
    """Both exception types produce the same status and response bytes."""
    pydantic_exc = _pydantic_unknown_field_error()
    fastapi_exc = RequestValidationError(
        errors=[{**err, "loc": ("body", *err["loc"])} for err in pydantic_exc.errors()]
    )

    responses = [
        render_visual_gen_validation_error(exc, OpenAIServer.create_error_response)
        for exc in (pydantic_exc, fastapi_exc)
    ]
    assert [response.status_code for response in responses] == [422, 422]
    assert responses[0].body == responses[1].body
    assert json.loads(responses[0].body) == _UNKNOWN_FIELD_BODY


def test_renderer_falls_back_to_str_without_errors() -> None:
    """An empty error list falls back to the exception's string form."""
    exc = RequestValidationError([])
    response = render_visual_gen_validation_error(exc, OpenAIServer.create_error_response)
    assert response.status_code == 422
    assert json.loads(response.body)["message"] == str(exc)


def test_renderer_delegates_to_the_factory() -> None:
    """Call the factory once by keyword and return its response unchanged."""
    calls: list[tuple[str, str, HTTPStatus]] = []
    sentinel = Response(status_code=422)

    def spy_factory(*, message: str, err_type: str, status_code: HTTPStatus) -> Response:
        calls.append((message, err_type, status_code))
        return sentinel

    result = render_visual_gen_validation_error(_pydantic_unknown_field_error(), spy_factory)
    assert result is sentinel
    assert calls == [
        (
            _UNKNOWN_FIELD_BODY["message"],
            "BadRequestError",
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )
    ]
