#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a declared workload through the engine of the tree under test, and print structured records.

Usage (rank 0 under ``trtllm-llmapi-launch`` when the world size is above one)::

    <tree's python> engine_driver.py --spec spec.json [--quickstart <path>] [-- <quickstart argv>]

The engine is built by the TREE'S OWN ``examples/llm-api/quickstart_advanced.py``:
its ``parse_arguments()`` reads ``spec["quickstart_argv"]`` (the argv every
gate already derives from task.yaml: topology, runtime switches, speculation)
and its ``setup_llm()`` returns the ``LLM`` and the ``SamplingParams``. So the
engine a gate measures is configured exactly as the quickstart gates
configure it, on any tree that ships that example, and only the workload and
the output differ:

* ``requests``: ``{"id", "prompt", "max_tokens", "images", "render"}`` each.
  ``images`` are local files; ``render`` is ``loader`` (the default: the
  tree's ``default_multimodal_input_loader`` applies the model's chat template
  and image placeholders) or ``raw`` (``prompt`` is already rendered and the
  images go in ``multi_modal_data`` as they are).
* ``dataset``: ``{"source", "num", "text_field", "answer_field", "split",
  "config", "template", "skip", "max_tokens"}``, appended after ``requests``.
  Local, or a Hugging Face id read from the cache only; see
  :mod:`workload_io`.
* ``concurrency``: requests in flight at once (default 1: one at a time, so
  every request runs alone and the schedule does not depend on timing).
* ``ignore_eos``: generate ``max_tokens`` whatever the model emits.
* ``perf``: ``{"prefill_tokens", "decode_tokens", "repeats", "warmup",
  "seed"}`` -- time one prefill chunk and one decode step instead.

Output, on stdout, each one JSON line behind a marker (see :mod:`workload_io`):

* ``MV2FLOW_DRIVER`` once: which ``tensorrt_llm`` and quickstart were imported;
* ``MV2FLOW_RECORD`` per request, in order: ``index``, ``id``,
  ``prompt_sha``, ``prompt_tokens``, ``token_ids``, ``text``, ``tokens``,
  ``iterations`` (streamed steps, the first being the context step),
  ``finish_reason``, ``images``;
* ``MV2FLOW_PERF`` once in perf mode: medians and every sample.

Exits non-zero with a traceback when anything fails.
"""

from __future__ import annotations

import argparse
import copy
import functools
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import workload_io as wio  # noqa: E402 -- imported by path, see the package docstring


def load_quickstart(path: str) -> Any:
    """Import the tree's ``quickstart_advanced.py`` as a module."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"quickstart not found: {source}")
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("mv2flow_quickstart", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_llm(quickstart: Any, argv: list[str]) -> tuple[Any, Any]:
    """``(llm, sampling_params)`` exactly as the quickstart builds them from ``argv``."""
    saved = sys.argv
    sys.argv = [str(getattr(quickstart, "__file__", "quickstart_advanced.py")), *argv]
    try:
        args = quickstart.parse_arguments()
    finally:
        sys.argv = saved
    mode = os.environ.get("TRTLLM_MODELING_V2")
    llm_class = getattr(quickstart, "LLM", None)
    if mode and llm_class is not None:
        # The quickstart has no flag for target selection, which is the LLM-API
        # argument `modeling_v2` (default "off"); the gates export the switch
        # as TRTLLM_MODELING_V2. Hand it to the constructor the quickstart calls.
        quickstart.LLM = functools.partial(llm_class, modeling_v2=mode)
        try:
            return quickstart.setup_llm(args)
        finally:
            quickstart.LLM = llm_class
    return quickstart.setup_llm(args)


def sampling_for(base: Any, max_tokens: int, ignore_eos: bool) -> Any:
    """A copy of ``base`` with this request's budget."""
    params = copy.deepcopy(base)
    params.max_tokens = int(max_tokens)
    if ignore_eos:
        params.ignore_eos = True
    return params


def _model_dir(llm: Any) -> str:
    return str(getattr(llm, "_hf_model_dir", None) or getattr(llm, "model_dir", "") or "")


def make_input(llm: Any, request: dict[str, Any]) -> Any:
    """The ``generate`` input for one request: text, token ids, or text plus images."""
    if request.get("token_ids") is not None:
        return {"prompt_token_ids": [int(t) for t in request["token_ids"]]}
    prompt = str(request.get("prompt", ""))
    images = [str(p) for p in request.get("images") or ()]
    if not images:
        return prompt
    for image in images:
        if not Path(image).is_file():
            raise FileNotFoundError(f"image not found: {image}")
        if wio.is_lfs_pointer(image):
            raise RuntimeError(
                f"{image} is a git-lfs pointer, not an image: run `git lfs pull --include "
                f"{Path(image).name}` in the tree under test"
            )
    if request.get("render") == "raw":
        from PIL import Image

        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [Image.open(p).convert("RGB") for p in images]},
        }
    from tensorrt_llm.inputs import default_multimodal_input_loader

    model_dir = _model_dir(llm)
    model_type = json.loads((Path(model_dir) / "config.json").read_text())["model_type"]
    inputs = default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=model_dir,
        model_type=model_type,
        modality="image" if len(images) == 1 else "multiple_image",
        prompts=[prompt],
        media=images if len(images) == 1 else [images],
        image_data_format="pt",
        device="cpu",
    )
    return inputs[0]


def stream_one(llm: Any, inp: Any, params: Any) -> tuple[Any, list[float]]:
    """Run one request streamed; return the final output and the time of every step."""
    stamps: list[float] = []
    last = None
    for output in llm.generate_async(inp, params, streaming=True):
        stamps.append(time.perf_counter())
        last = output
    if last is None:
        raise RuntimeError("the engine streamed no output for a request")
    return last, stamps


def record_of(index: int, request: dict[str, Any], output: Any, steps: int) -> dict[str, Any]:
    """The ``MV2FLOW_RECORD`` payload of one finished request."""
    sequence = output.outputs[0]
    token_ids = [int(t) for t in sequence.token_ids]
    prompt_ids = getattr(output, "prompt_token_ids", None)
    return {
        "index": index,
        "id": request.get("id", str(index)),
        "prompt_sha": wio.prompt_digest(str(request.get("prompt", request.get("token_ids", "")))),
        "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
        "token_ids": token_ids,
        "text": sequence.text,
        "tokens": len(token_ids),
        "iterations": steps,
        "finish_reason": str(getattr(sequence, "finish_reason", "")),
        "images": len(request.get("images") or ()),
    }


def resolve_requests(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The declared requests followed by the dataset's, each with its budget."""
    default_tokens = int(spec.get("max_tokens", 64))
    requests = []
    for i, entry in enumerate(spec.get("requests") or ()):
        request = dict(entry)
        request.setdefault("id", f"r{i}")
        request.setdefault("max_tokens", default_tokens)
        requests.append(request)
    dataset = spec.get("dataset")
    if dataset:
        rows = wio.load_rows(
            dataset["source"], split=dataset.get("split", "test"), config=dataset.get("config")
        )
        chosen = wio.select_prompts(
            rows,
            num=int(dataset.get("num", 32)),
            text_field=dataset.get("text_field"),
            answer_field=dataset.get("answer_field"),
            template=dataset.get("template") or "{text}",
            skip=int(dataset.get("skip", 0)),
        )
        for i, row in enumerate(chosen):
            requests.append(
                {
                    "id": f"d{i}",
                    "prompt": row["prompt"],
                    "max_tokens": int(dataset.get("max_tokens", default_tokens)),
                }
            )
    return requests


def run_requests(llm: Any, base: Any, spec: dict[str, Any]) -> int:
    """Run every request, ``concurrency`` at a time, and print a record for each."""
    requests = resolve_requests(spec)
    concurrency = max(1, int(spec.get("concurrency", 1)))
    ignore_eos = bool(spec.get("ignore_eos", False))
    for start in range(0, len(requests), concurrency):
        window = requests[start : start + concurrency]
        if concurrency == 1:
            request = window[0]
            params = sampling_for(base, request["max_tokens"], ignore_eos)
            output, stamps = stream_one(llm, make_input(llm, request), params)
            wio.emit(wio.RECORD_MARKER, record_of(start, request, output, len(stamps)))
            continue
        futures = [
            llm.generate_async(
                make_input(llm, request),
                sampling_for(base, request["max_tokens"], ignore_eos),
                streaming=True,
            )
            for request in window
        ]
        for offset, (request, future) in enumerate(zip(window, futures)):
            steps, last = 0, None
            for output in future:
                steps += 1
                last = output
            wio.emit(wio.RECORD_MARKER, record_of(start + offset, request, last, steps))
    return len(requests)


def _vocab_size(llm: Any) -> int:
    tokenizer = getattr(llm, "tokenizer", None)
    size = getattr(tokenizer, "vocab_size", None)
    if isinstance(size, int) and size > 256:
        return size
    try:
        config = json.loads((Path(_model_dir(llm)) / "config.json").read_text())
    except (OSError, ValueError):
        return 32000
    size = config.get("vocab_size") or (config.get("text_config") or {}).get("vocab_size")
    return int(size) if size else 32000


def _median(values: list[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def run_perf(llm: Any, base: Any, perf: dict[str, Any]) -> dict[str, Any]:
    """Time one prefill chunk and one decode step; return the ``MV2FLOW_PERF`` payload.

    Wall clock on the driver, which is a stable proxy for the GPU time of a
    step at batch 1: the scheduling and response overhead around one step is
    nearly constant between two trees, so a change in the kernels moves it.

    * prefill: a request of exactly ``prefill_tokens`` random token ids and
      one output token, timed end to end -- one context chunk when the count
      is the engine's ``max_num_tokens``. Fresh ids each repeat, so no repeat
      is a prefix-cache hit.
    * decode: a short request generating ``decode_tokens`` past EOS, streamed;
      the step time is the median interval between streamed steps.

    ``warmup`` repeats of each are run first and discarded.
    """
    rng = random.Random(int(perf.get("seed", 0)))
    vocab = _vocab_size(llm)
    low, high = 100, max(101, vocab - 100)

    def ids(count: int) -> dict[str, Any]:
        return {"prompt_token_ids": [rng.randrange(low, high) for _ in range(count)]}

    prefill_tokens = int(perf.get("prefill_tokens", 2048))
    decode_tokens = int(perf.get("decode_tokens", 64))
    repeats = int(perf.get("repeats", 7))
    warmup = int(perf.get("warmup", 2))

    prefill: list[float] = []
    one = sampling_for(base, 1, True)
    for i in range(warmup + repeats):
        started = time.perf_counter()
        llm.generate([ids(prefill_tokens)], one)
        elapsed = (time.perf_counter() - started) * 1000.0
        if i >= warmup:
            prefill.append(elapsed)

    decode: list[float] = []
    tokens_per_step: list[float] = []
    many = sampling_for(base, 1 + decode_tokens, True)
    for i in range(warmup + repeats):
        output, stamps = stream_one(llm, ids(32), many)
        intervals = [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])]
        if i >= warmup and intervals:
            decode.append(_median(intervals))
            generated = len(output.outputs[0].token_ids)
            tokens_per_step.append((generated - 1) / max(1, len(stamps) - 1))
    return {
        "prefill_tokens": prefill_tokens,
        "prefill_ms": _median(prefill),
        "prefill_samples": prefill,
        "decode_tokens": decode_tokens,
        "decode_step_ms": _median(decode),
        "decode_samples": decode,
        "tokens_per_step": _median(tokens_per_step),
    }


def main(argv: list[str] | None = None) -> int:
    """Read the spec, build the engine, run the workload, print records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="the workload spec, a JSON file")
    parser.add_argument(
        "--quickstart", help="the tree's quickstart_advanced.py (default: the spec's)"
    )
    parser.add_argument(
        "quickstart_argv",
        nargs=argparse.REMAINDER,
        help="after `--`: the quickstart's argv (default: the spec's)",
    )
    options = parser.parse_args(argv)
    spec = json.loads(Path(options.spec).read_text(encoding="utf-8"))
    if options.quickstart:
        spec["quickstart"] = options.quickstart
    words = [w for w in options.quickstart_argv if w != "--"] if options.quickstart_argv else []
    if words:
        spec["quickstart_argv"] = words

    quickstart = load_quickstart(spec["quickstart"])
    import tensorrt_llm

    wio.emit(
        wio.DRIVER_MARKER,
        {
            "tensorrt_llm": str(getattr(tensorrt_llm, "__file__", "")),
            "quickstart": spec["quickstart"],
        },
    )
    llm, base = build_llm(quickstart, list(spec.get("quickstart_argv") or ()))
    try:
        if spec.get("perf"):
            wio.emit(wio.PERF_MARKER, run_perf(llm, base, dict(spec["perf"])))
        else:
            run_requests(llm, base, spec)
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
