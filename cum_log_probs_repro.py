# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM logprob repro — disaggregated serving bug reproduction.

This file serves dual purpose:
  - `python repro.py prefill`  — starts a FastAPI prefill server (context_only)
  - `python repro.py test`     — runs pytest decode tests (agg + disagg)

"""

import argparse
import base64
import os
import threading
from dataclasses import dataclass, field

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.disaggregated_params import DisaggregatedParams

PROMPT = "<|user|>\nWhat is the capital of Poland?</s>\n<|assistant|>\n"
PREFILL_URL = "http://localhost:8000/prefill"
CACHE_TRANSCEIVER = {"backend": "UCX", "max_tokens_in_buffer": 2048}
MODEL_PATH = os.getenv(
    "TRTLLM_MODEL_PATH", "/home/scratch.bbuddharaju_gpu/random/hf_models/TinyLlama-1.1B-Chat-v1.0"
)


# ---------------------------------------------------------------------------
# Prefill server (runs in the prefill container)
# ---------------------------------------------------------------------------


class PrefillRequest(BaseModel):
    prompt: str
    max_tokens: int = 20
    logprobs: int | None = None
    return_generation_logits: bool = False


class PrefillResponse(BaseModel):
    """Serialized DisaggregatedParams from TRT-LLM context_only result."""

    request_type: str
    first_gen_tokens: list[int]
    ctx_request_id: int
    opaque_state: str  # Base64-encoded bytes (may include embedded logprobs).


class PrefillEngine:
    """Wraps TRT-LLM LLM in context_only mode behind a FastAPI server."""

    def __init__(self):
        self.llm: LLM = None
        self.ready = False

    def start(self):
        self._start_http_server()
        self.llm = LLM(
            model=MODEL_PATH,
            disable_overlap_scheduler=True,
            cache_transceiver_config=CACHE_TRANSCEIVER,
        )
        self.ready = True

    def _start_http_server(self):
        app = FastAPI()

        @app.get("/health")
        async def health():
            if not self.ready:
                return JSONResponse({"status": "not ready"}, status_code=503)
            return {"status": "ok"}

        @app.post("/prefill")
        async def prefill(req: PrefillRequest) -> PrefillResponse:
            if not self.ready or self.llm is None:
                return JSONResponse({"status": "not ready"}, status_code=503)
            return await self._generate_local_prefill(req)

        thread = threading.Thread(
            target=uvicorn.run,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": 8000, "log_level": "info"},
            daemon=True,
        )
        thread.start()

    async def _generate_local_prefill(self, req: PrefillRequest) -> PrefillResponse:
        """Run context_only on the local LLM. Only called via the HTTP API."""
        sp = SamplingParams(
            max_tokens=req.max_tokens,
            logprobs=req.logprobs,
            return_generation_logits=req.return_generation_logits,
        )
        result = self.llm.generate_async(
            req.prompt,
            sampling_params=sp,
            disaggregated_params=DisaggregatedParams(request_type="context_only"),
        )
        output = await result.aresult()
        dp = output.outputs[0].disaggregated_params
        out = output.outputs[0]
        logits = out.generation_logits
        logits_info = f"shape={tuple(logits.shape)}" if logits is not None else "None"
        print(
            f"  [prefill] tokens={len(out.token_ids)}, logits={logits_info}, "
            f"logprobs={out.logprobs}, text={out.text!r}"
        )
        # Logprobs are embedded in opaque_state automatically by the engine.
        return PrefillResponse(
            request_type="context_only",
            first_gen_tokens=dp.first_gen_tokens,
            ctx_request_id=dp.ctx_request_id,
            opaque_state=base64.b64encode(dp.opaque_state).decode(),
        )


# ---------------------------------------------------------------------------
# Decode engine (runs in the decode container)
# ---------------------------------------------------------------------------


@dataclass
class Engine:
    """Wraps TRT-LLM LLM for decode. Supports local (agg) and remote prefill (disagg)."""

    llm: LLM = field(default=None, init=False)

    def start(self):
        self.llm = LLM(model=MODEL_PATH, cache_transceiver_config=CACHE_TRANSCEIVER)

    def shutdown(self):
        self.llm.shutdown()

    async def generate_async(self, sampling_params: SamplingParams, remote_prefill: bool = False):
        if remote_prefill:
            async for chunk in self._generate_with_remote_prefill(sampling_params):
                yield chunk
        else:
            async for chunk in self._generate_local(sampling_params):
                yield chunk

    async def _generate_local(self, sp: SamplingParams):
        result = self.llm.generate_async(PROMPT, sampling_params=sp, streaming=True)
        async for _ in result:
            yield result.outputs[0]

    async def _generate_with_remote_prefill(self, sp: SamplingParams):
        dp = await self._remote_prefill(sp)
        print(
            f"  remote prefill returned: request_type={dp.request_type}, "
            f"first_gen_tokens={dp.first_gen_tokens}, ctx_request_id={dp.ctx_request_id}"
        )
        result = self.llm.generate_async(
            PROMPT,
            sampling_params=sp,
            streaming=True,
            disaggregated_params=dp,
        )
        async for _ in result:
            yield result.outputs[0]

    async def _remote_prefill(self, sp: SamplingParams) -> DisaggregatedParams:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                PREFILL_URL,
                json={
                    "prompt": PROMPT,
                    "max_tokens": sp.max_tokens,
                    "logprobs": sp.logprobs,
                    "return_generation_logits": sp.return_generation_logits,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Logprobs are embedded inside opaque_state; no separate handling needed.
        return DisaggregatedParams(
            request_type="generation_only",
            first_gen_tokens=data["first_gen_tokens"],
            ctx_request_id=data["ctx_request_id"],
            opaque_state=base64.b64decode(data["opaque_state"]),
        )


# ---------------------------------------------------------------------------
# Pytest tests (invoked via `pytest repro.py`)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def engine():
    e = Engine()
    e.start()
    yield e
    e.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tag,remote_prefill,sp",
    [
        # These succeed.
        ("agg_logits", False, SamplingParams(max_tokens=20, return_generation_logits=True)),
        ("disagg_logits", True, SamplingParams(max_tokens=20, return_generation_logits=True)),
        ("agg_logprobs", False, SamplingParams(max_tokens=20, logprobs=1)),
        # Regression test for https://nvbugspro.nvidia.com/bug/5926823.
        ("disagg_logprobs", True, SamplingParams(max_tokens=20, logprobs=1)),
    ],
    ids=["agg_logits", "disagg_logits", "agg_logprobs", "disagg_logprobs"],
)
async def test_streaming(engine, tag, remote_prefill, sp):
    chunk_count = 0
    async for out in engine.generate_async(sp, remote_prefill=remote_prefill):
        chunk_count += 1
        logits = out.generation_logits
        logits_info = f"shape={tuple(logits.shape)}" if logits is not None else "None"
        print(
            f"  [{tag}] tokens={len(out.token_ids)}, logits={logits_info}, "
            f"logprobs={out.logprobs}, text={out.text!r}"
        )
    assert chunk_count > 0, "Expected at least one streaming chunk"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRT-LLM logprob disagg repro")
    parser.add_argument("mode", choices=["prefill", "test"])
    args = parser.parse_args()

    if args.mode == "prefill":
        engine = PrefillEngine()
        engine.start()
        print("Prefill server running on :8000")
        threading.Event().wait()
    else:
        import sys

        sys.exit(
            pytest.main(
                [
                    __file__,
                    "-v",
                    "-s",
                    "-W",
                    "ignore::DeprecationWarning",
                    "-W",
                    "ignore::pydantic.warnings.PydanticDeprecatedSince20",
                ]
            )
        )
