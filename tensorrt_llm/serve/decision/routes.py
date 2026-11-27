# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain JSON decision endpoints for the generation and embedding services."""

import asyncio
import logging
import math
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException

from tensorrt_llm.executor.utils import RequestError, is_context_length_exceeded_message

from ..encode_batcher import QueueFullError
from . import openai, systemone
from .openai import OpenAIDecisionRequest, OpenAIDecisionResponse
from .scoring import score_decision_questions
from .systemone import SystemOneRequest, SystemOneResponse

if TYPE_CHECKING:
    from ..openai_server import OpenAIServer


def decision_route_handler(route: APIRoute):
    """Validate a public JSON request and format errors for its decision protocol."""
    handle = APIRoute.get_route_handler(route)
    systemone = route.path == "/v1/systemone"

    def error(status, message, *, param=None, code=None):
        if systemone:
            body = {"detail": message}
        else:
            category = {429: "rate_limit_error", 500: "server_error"}.get(
                status, "invalid_request_error"
            )
            body = {"error": {"message": message, "type": category, "code": code, "param": param}}
        return JSONResponse(body, status_code=status)

    async def serve(request: Request):
        media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        encoding = request.headers.get("content-encoding", "identity").strip().lower()
        if (
            media_type != "application/json"
            or encoding != "identity"
            or request.headers.get("x-trtllm-msgpack") == "1"
        ):
            return error(415, "Send an uncompressed application/json request.")
        try:
            return await handle(request)
        except RequestValidationError as exc:
            errors = [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()]
            if systemone:
                if errors[0]["type"] == "json_invalid":
                    return error(400, "Malformed JSON request.")
                return error(422, errors)
            first = errors[0]
            param = ".".join(map(str, first["loc"][1:]))
            return error(400, first["msg"], param=param or None)
        except HTTPException as exc:
            return error(
                exc.status_code,
                exc.detail,
                param="model" if exc.status_code == 404 else None,
                code="model_not_found" if exc.status_code == 404 else None,
            )
        except QueueFullError as exc:
            return error(429, str(exc))
        except ValueError as exc:
            return error(400, str(exc))
        except RuntimeError as exc:
            if isinstance(exc, RequestError) and is_context_length_exceeded_message(str(exc)):
                return error(400, str(exc))
            logging.getLogger(__name__).exception("Decision scoring failed")
            return error(500, "Decision scoring failed.")

    if systemone:
        return serve

    async def openai_response(request: Request):
        response = await serve(request)
        response.headers["x-request-id"] = uuid4().hex
        return response

    return openai_response


async def _read(raw: Request, server, state, questions):
    async def disconnected():
        while (await raw.receive())["type"] != "http.disconnect":
            pass

    scoring = asyncio.create_task(score_decision_questions(server, state, questions))
    watching = asyncio.create_task(disconnected())
    try:
        await asyncio.wait((scoring, watching), return_when=asyncio.FIRST_COMPLETED)
        if not scoring.done():
            return None
        rows = scoring.result()
        if len(rows) != len(questions) or any(
            len(row.probabilities) != len(q.names)
            or not all(0 <= p <= 1 for p in row.probabilities)
            or not math.isclose(sum(row.probabilities), 1.0, abs_tol=1e-5)
            for q, row in zip(questions, rows)
        ):
            raise RuntimeError("Invalid candidate probability distribution")
        return rows
    finally:
        watching.cancel()
        scoring.cancel()
        await asyncio.gather(watching, scoring, return_exceptions=True)


def mount_decision_endpoints(app: FastAPI, server: "OpenAIServer") -> None:
    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def create_systemone(request: SystemOneRequest, raw: Request):
        if request.model not in (
            server.model,
            str(server.generator.args.model),
            "jev-latest",
            "jev-preview",
        ):
            raise HTTPException(404, "The requested model is not served.")
        questions = [systemone.read_question(q) for q in request.questions.values()]
        rows = await _read(raw, server, request.state, questions)
        if rows is None:
            return Response(status_code=499)
        return {
            "model": server.model,
            "answers": {
                name: systemone.answer(q, row)
                for name, q, row in zip(request.questions, questions, rows, strict=True)
            },
            "usage": {
                "input_tokens": sum(r.input_tokens for r in rows),
                "output_tokens": sum(r.output_tokens for r in rows),
            },
        }

    @app.post("/v1/decisions", response_model=OpenAIDecisionResponse)
    async def decisions(request: OpenAIDecisionRequest, raw: Request):
        if request.model not in (server.model, str(server.generator.args.model)):
            raise HTTPException(404, "The requested model is not served.")
        max_choices = 26 if server.encode_batcher is None else 255
        questions = [openai.read_question(q, max_choices) for q in request.questions]
        rows = await _read(raw, server, request.evidence(), questions)
        if rows is None:
            return Response(status_code=499)
        input_tokens = sum(r.input_tokens for r in rows)
        output_tokens = sum(r.output_tokens for r in rows)
        return {
            "model": server.model,
            "answers": [openai.answer(q, r) for q, r in zip(request.questions, rows, strict=True)],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {
                    "cached_tokens": sum(r.cached_tokens for r in rows),
                    "cache_write_tokens": 0,
                },
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
