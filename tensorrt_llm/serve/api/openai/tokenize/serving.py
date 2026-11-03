# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``POST /_internal/tokenize``: token ids of a plain-text prompt."""

from http import HTTPStatus
from typing import Any

from fastapi.responses import JSONResponse

from tensorrt_llm.serve._base.serving import BaseServing
from tensorrt_llm.serve.openai_protocol import TokenizeRequest, TokenizeResponse


class TokenizeServing(BaseServing):
    """``POST /_internal/tokenize``: token ids of a plain-text prompt."""

    def __init__(self, *, model: str, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    async def tokenize(self, request: TokenizeRequest) -> JSONResponse:
        try:
            token_ids = self.tokenizer.encode(request.prompt)
            response = TokenizeResponse(count=len(token_ids), tokens=token_ids)
            return JSONResponse(content=response.model_dump())
        except Exception as e:
            return self.create_error_response(
                message=str(e),
                err_type="InvalidRequestError",
                status_code=HTTPStatus.BAD_REQUEST)
