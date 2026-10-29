#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run normalized requests through public LLM / SamplingParams / generate APIs.

The spec contains ``llm`` constructor options, ``sampling`` options and ordered
``requests``, streamed in windows of ``concurrency`` and recorded one per request.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_facts  # noqa: E402
import reuse_probe  # noqa: E402
import workload_io as wio  # noqa: E402

MODELING_V2_ENV = "TRTLLM_MODELING_V2"


def build_llm(spec: Mapping[str, Any]) -> tuple[Any, Any]:
    """Construct the measured tree's public API directly, without replacing classes."""
    from tensorrt_llm import LLM, SamplingParams

    if spec.get("dataset"):
        raise ValueError("dataset was not normalized before engine launch")
    options = dict(spec["llm"])
    mode = os.environ.get(MODELING_V2_ENV)
    if mode:
        options["modeling_v2"] = mode
    sampling = SamplingParams(**spec.get("sampling", {}))
    if getattr(sampling, "n", 1) != 1 or getattr(sampling, "use_beam_search", False):
        raise ValueError("correctness checks require one sequence without beam search")
    return LLM(**options), sampling


def sampling_for(base: Any, max_tokens: int, ignore_eos: bool, logprobs: int = 0) -> Any:
    """A copy of the declared sampling parameters with this request's budget.

    ``logprobs`` asks for each generated token's top-k log-probs, as a request
    whose tokens a gate compares up to a near-tie does.
    """
    params = copy.deepcopy(base)
    params.max_tokens = int(max_tokens)
    if ignore_eos:
        params.ignore_eos = True
    if logprobs:
        params.logprobs = int(logprobs)
    return params


def logprob_rows(entries: Any) -> list[list[list[Any]]]:
    """The engine's log-probs, one ``{token id: Logprob}`` per position, as ``[id, log-prob, rank]`` lists.

    Best first. A position the engine returned as a bare float (the simple
    format) becomes one entry with no id, which a gate reads as missing.
    """
    rows: list[list[list[Any]]] = []
    for entry in entries or ():
        if isinstance(entry, Mapping):
            row = [
                [int(token), float(getattr(lp, "logprob", lp)), getattr(lp, "rank", None)]
                for token, lp in entry.items()
            ]
            row.sort(key=lambda item: -item[1])
        else:
            row = [[None, float(entry), None]]
        rows.append(row)
    return rows


def make_input(llm: Any, request: Mapping[str, Any], *, model: str = "") -> Any:
    """Public text, token or multimodal input; images are local and never downloaded."""
    if request.get("token_ids") is not None:
        return {"prompt_token_ids": [int(t) for t in request["token_ids"]]}
    if request.get("probe_tokens") is not None:
        return {
            "prompt_token_ids": reuse_probe.probe_prompt(
                llm.tokenizer, int(request["probe_tokens"])
            )
        }
    prompt = str(request.get("prompt", ""))
    images = [str(p) for p in request.get("images") or ()]
    if not images:
        return prompt
    for image in images:
        if not Path(image).is_file():
            raise FileNotFoundError(f"image not found: {image}")
        if wio.is_lfs_pointer(image):
            raise RuntimeError(f"{image} is a git-lfs pointer, not an image")
    if request.get("render") == "raw":
        from PIL import Image

        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in images]},
        }
    from tensorrt_llm.inputs import default_multimodal_input_loader

    model_type = json.loads((Path(model) / "config.json").read_text())["model_type"]
    return default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=model,
        model_type=model_type,
        modality="image" if len(images) == 1 else "multiple_image",
        prompts=[prompt],
        media=images if len(images) == 1 else [images],
        image_data_format="pt",
        device="cpu",
    )[0]


def record_of(index: int, request: Mapping[str, Any], output: Any, steps: int) -> dict[str, Any]:
    """One request's result; retain every sequence instead of silently dropping alternatives."""
    sequences = [
        {
            "token_ids": [int(t) for t in seq.token_ids],
            "text": seq.text,
            "finish_reason": str(getattr(seq, "finish_reason", "")),
        }
        for seq in output.outputs
    ]
    if not sequences:
        raise RuntimeError("the engine returned no sequences")
    first = sequences[0]
    prompt_ids = getattr(output, "prompt_token_ids", None)
    record = {
        "index": index,
        "id": request.get("id", str(index)),
        "prompt_sha": wio.prompt_digest(str(request.get("prompt", request.get("token_ids", "")))),
        "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
        "cached_tokens": reuse_probe.cached_tokens(output),
        **first,
        "tokens": len(first["token_ids"]),
        "iterations": steps,
        "images": len(request.get("images") or ()),
        "sequences": sequences,
    }
    if request.get("logprobs"):
        record["top_logprobs"] = logprob_rows(getattr(output.outputs[0], "logprobs", None))
    return record


def _run_requests(
    llm: Any, base: Any, spec: Mapping[str, Any], *, phase: str = ""
) -> dict[str, int]:
    """Drain one phase before submitting the next, keeping the same engine and cache."""
    if spec.get("dataset"):
        raise ValueError("dataset was not normalized before engine launch")
    requests = spec.get("requests") or []
    inputs = [make_input(llm, r, model=str(spec["llm"]["model"])) for r in requests]
    if spec.get("apply_chat_template"):
        if any(r.get("images") for r in requests):
            raise ValueError("apply_chat_template requires text requests; multimodal uses render")
        inputs = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            if isinstance(p, str)
            else p
            for p in inputs
        ]
    summary = {"requests": 0, "reported": 0, "hits": 0, "reused_tokens": 0}
    concurrency = max(1, int(spec.get("concurrency", 1)))
    for start in range(0, len(requests), concurrency):
        window = requests[start : start + concurrency]
        streams = [
            llm.generate_async(
                inputs[start + i],
                sampling_for(
                    base,
                    r["max_tokens"],
                    bool(spec.get("ignore_eos")),
                    int(r.get("logprobs") or 0),
                ),
                streaming=True,
            )
            for i, r in enumerate(window)
        ]
        for offset, (request, stream) in enumerate(zip(window, streams)):
            steps, output = 0, None
            for output in stream:
                steps += 1
            if output is None:
                raise RuntimeError("the engine streamed no output for a request")
            index = start + offset
            record = record_of(index, request, output, steps)
            if phase:
                record["phase"] = phase
            wio.emit(wio.RECORD_MARKER, record)
            summary["requests"] += 1
            cached = record["cached_tokens"]
            if cached is not None:
                summary["reported"] += 1
                summary["hits"] += int(cached > 0)
                summary["reused_tokens"] += cached
    return summary


def run_workload(llm: Any, base: Any, spec: Mapping[str, Any]) -> None:
    """Run ordinary requests then optional boot phases, with one final cache probe."""
    summary = _run_requests(llm, base, spec, phase="boot" if spec.get("phases") else "")
    for phase in spec.get("phases") or ():
        result = _run_requests(
            llm,
            base,
            {**spec, **phase},
            phase=phase["id"],
        )
        for key, value in result.items():
            summary[key] += value
    probe = (
        reuse_probe.probe(llm, sampling_for(base, 1, True), spec["llm"])
        if spec.get("reuse_probe")
        else None
    )
    wio.emit(wio.REUSE_MARKER, {**summary, "probe": probe})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True)
    options = parser.parse_args(argv)
    spec = json.loads(Path(options.spec).read_text(encoding="utf-8"))
    import tensorrt_llm

    wio.emit(wio.DRIVER_MARKER, {"tensorrt_llm": str(tensorrt_llm.__file__), "api": "LLM"})
    llm, base = build_llm(spec)
    try:
        engine_facts.emit(llm, requested=spec["llm"])
        run_workload(llm, base, spec)
    finally:
        llm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
