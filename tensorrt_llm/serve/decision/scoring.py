# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read candidate probabilities from a generation model or a checkpoint decision head."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from tensorrt_llm.executor.utils import context_length_exceeded_message
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.render.decisions import ReadQuestion, RenderedDecision, render_decisions
from tensorrt_llm.serve.render.resources import RenderResources

if TYPE_CHECKING:
    from tensorrt_llm.serve.openai_server import OpenAIServer

__all__ = ["ReadQuestion", "ReadResult", "score_decision_questions"]

# Each generation read holds a full vocabulary row. This per-request scheduling
# limit bounds those buffers even when the engine allows larger batches.
_MAX_CONCURRENT_READS = 8


@dataclass
class ReadResult:
    probabilities: list[float]
    input_tokens: int
    output_tokens: int = 0
    label_mass: float | None = None
    cached_tokens: int = 0


@lru_cache(maxsize=16)
def _pplx_config(path: Path) -> dict | None:
    config_path = path / "decision_config.json"
    if not config_path.is_file():
        return None
    with config_path.open() as handle:
        config = json.load(handle)
    if (
        config.get("attention_mode") != "noncausal_full_attention"
        or config.get("pooling") != "last"
        or len(config.get("codes", ())) != 255
        or len(config.get("token_ids", ())) != 255
        or not (path / "readout.safetensors").is_file()
    ):
        raise ValueError("unsupported decision checkpoint configuration")
    return config


def _distribution(logits: torch.Tensor, indices: list[int], temperature: float) -> list[float]:
    selected = logits.float()[indices]
    if not torch.isfinite(selected).all():
        raise RuntimeError("the model returned non-finite candidate logits")
    # Shift before applying the temperature so a small, positive temperature cannot overflow.
    return torch.softmax((selected.double() - selected.max()) / temperature, dim=0).tolist()


async def _generate_read(server, row: RenderedDecision) -> ReadResult:
    params = SamplingParams(
        max_tokens=1,
        return_generation_logits=True,
        detokenize=False,
        add_special_tokens=False,
        ignore_eos=True,
    )
    request = server.generator.generate_async(row.token_ids, params)
    try:
        result = await request
    except asyncio.CancelledError:
        request.abort()
        raise
    output = result.outputs[0]
    if output.generation_logits is None or output.generation_logits.shape[0] != 1:
        raise RuntimeError("the model did not return first-token logits")
    logits = output.generation_logits[0, : server._logit_bias_vocab_size()].float()
    probabilities = _distribution(logits, row.label_ids, 1.0)
    mass = logits.log_softmax(dim=-1)[row.label_ids].exp().sum().item()
    if not math.isfinite(mass):
        raise RuntimeError("the model returned a non-finite vocabulary distribution")
    return ReadResult(
        probabilities,
        len(result.prompt_token_ids),
        len(output.token_ids),
        min(1.0, mass),
        result.cached_tokens,
    )


async def _pplx_read(server, row: RenderedDecision, config: dict) -> ReadResult:
    indices = row.code_indices
    if indices is None or row.label_ids != [config["token_ids"][i] for i in indices]:
        raise ValueError("the tokenizer answer codes do not match the checkpoint readout")
    result = await server.encode_batcher.submit(row.token_ids)
    logits = result.logits.flatten()
    if logits.numel() != len(config["codes"]):
        raise RuntimeError("the checkpoint did not return the configured decision head")
    temperature = float(config["temperature"])
    if not 0 < temperature < math.inf:
        raise ValueError("the decision head requires a finite positive temperature")
    return ReadResult(_distribution(logits, indices, temperature), len(row.token_ids))


async def score_decision_questions(
    server: OpenAIServer, state: Any, questions: list[ReadQuestion]
) -> list[ReadResult]:
    generator = server.generator
    model_dir = generator._hf_model_dir
    config = _pplx_config(Path(model_dir)) if model_dir else None
    if config is not None:
        limits = getattr(generator, "_encoder_executor", None)
        architectures = getattr(getattr(limits, "pretrained_config", None), "architectures", ())
        if server.encode_batcher is None or "PplxDeciderForDecision" not in (architectures or ()):
            raise ValueError("serve this decision checkpoint with trtllm-serve embeddings")
    elif server.encode_batcher is not None:
        raise ValueError("this embedding checkpoint has no decision head")
    else:
        limits = generator.args
    window = min(limits.max_batch_size or 1, _MAX_CONCURRENT_READS)
    max_context = limits.max_seq_len
    if config is None and max_context is not None:
        max_context *= limits.context_parallel_size
    rows = render_decisions(
        RenderResources.from_server(server), state, questions, config["codes"] if config else None
    )
    pending, results = [], []
    try:
        async for row in rows:
            if config is None:
                if max_context is not None and len(row.token_ids) >= max_context:
                    raise ValueError(
                        context_length_exceeded_message(max_context, len(row.token_ids))
                    )
                read = _generate_read(server, row)
            else:
                server.encode_batcher.validate_input(row.token_ids)
                read = _pplx_read(server, row, config)
            pending.append(asyncio.create_task(read))
            if len(pending) == window or (not results and max_context is None):
                results.extend(await asyncio.gather(*pending))
                pending.clear()
        results.extend(await asyncio.gather(*pending))
        return results
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await rows.aclose()
