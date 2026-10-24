#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the logit reference's prompts through the engine of the tree under test, and write its log-probs.

Usage (rank 0 under ``trtllm-llmapi-launch`` when the world size is above one)::

    <tree's python> logit_driver.py --spec spec.json [--quickstart <path>] [-- <quickstart argv>]

``harness/engine_driver.py`` (imported by path) builds the engine with the
tree's own ``quickstart_advanced.py`` from the spec's argv. The spec is a JSON
file the logit gate writes (``logit_agreement.py``):

* ``quickstart`` and ``quickstart_argv``, as for the engine driver;
* ``requests``: ``{"id", "token_ids"}`` (sent as token ids) or ``{"id",
  "prompt", "images", "render"}`` (text and images, through the engine
  driver's ``make_input``);
* ``max_tokens``: the greedy tokens each request generates, whatever they are
  (``ignore_eos``);
* ``top_k``: ``prompt_logprobs`` and ``logprobs`` per request;
* ``out``: where the results go.

Each request runs alone, in order, with a KV-cache salt of its own where the
tree's ``generate_async`` takes one: prompt log-probs come from the context
step, and a prefix another request left in the cache would get none. ``out``
is one JSON document, ``{"records": [...]}``, a record per request:
``index``, ``id``, ``prompt_token_ids``, ``token_ids``, ``finish_reason``, and
the engine's log-probs per prompt position (``prompt_logprobs``) and per
generated token (``logprobs``), each a list of ``[token id, log-prob, rank]``
best first. On stdout, each one JSON line behind a marker (see
``harness/workload_io.py``): ``MV2FLOW_DRIVER`` once, then ``MV2FLOW_LOGITS``
per request (``index``, ``id``, ``prompt_tokens``, ``tokens``).

Exits non-zero with a traceback when anything fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent

#: One per request, after its log-probs are collected.
LOGITS_MARKER = "MV2FLOW_LOGITS"


def _load(name: str, path: Path) -> Any:
    """Import the file at ``path`` as module ``name``."""
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


engine = _load("mv2flow_engine_driver", _HERE / "harness" / "engine_driver.py")
wio = _load("mv2flow_workload_io", _HERE / "harness" / "workload_io.py")


def logprob_rows(entries: Any) -> list[list[list[Any]]]:
    """The engine's log-probs, one ``{token id: Logprob}`` per position, as ``[id, log-prob, rank]`` lists.

    Best first. A position the engine returned as a bare float (the simple
    format) becomes one entry with no id, which the gate reads as missing.
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


def sampling_for(base: Any, max_tokens: int, top_k: int) -> Any:
    """``base`` for one request: ``max_tokens`` greedy tokens past any end token, with top-k log-probs."""
    params = engine.sampling_for(base, max_tokens, True)
    params.prompt_logprobs = int(top_k)
    params.logprobs = int(top_k)
    return params


def record_of(index: int, request: Mapping[str, Any], output: Any) -> dict[str, Any]:
    """One request's record: its tokens and the engine's log-probs."""
    sequence = output.outputs[0]
    prompt_ids = getattr(output, "prompt_token_ids", None) or ()
    return {
        "index": index,
        "id": str(request.get("id", index)),
        "prompt_token_ids": [int(t) for t in prompt_ids],
        "token_ids": [int(t) for t in sequence.token_ids],
        "finish_reason": str(getattr(sequence, "finish_reason", "")),
        "prompt_logprobs": logprob_rows(getattr(sequence, "prompt_logprobs", None)),
        "logprobs": logprob_rows(getattr(sequence, "logprobs", None)),
    }


def run_requests(llm: Any, base: Any, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every request alone, in order; a record each."""
    max_tokens = int(spec["max_tokens"])
    top_k = int(spec["top_k"])
    salted = "cache_salt" in inspect.signature(llm.generate_async).parameters
    records = []
    for index, request in enumerate(spec.get("requests") or ()):
        params = sampling_for(base, max_tokens, top_k)
        salt = {"cache_salt": f"mv2flow-logits-{index}"} if salted else {}
        output = llm.generate_async(engine.make_input(llm, dict(request)), params, **salt).result()
        record = record_of(index, request, output)
        records.append(record)
        wio.emit(
            LOGITS_MARKER,
            {
                "index": index,
                "id": record["id"],
                "prompt_tokens": len(record["prompt_token_ids"]),
                "tokens": len(record["token_ids"]),
            },
        )
    return records


def write_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    """``{"records": records}`` at ``path``, written whole or not at all."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.name}.partial")
    staged.write_text(json.dumps({"records": list(records)}, separators=(",", ":")), "utf-8")
    os.replace(staged, target)
    return target


def main(argv: list[str] | None = None) -> int:
    """Read the spec, build the engine, run every request, write the records."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="the logit gate's spec, a JSON file")
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

    quickstart = engine.load_quickstart(spec["quickstart"])
    import tensorrt_llm

    wio.emit(
        wio.DRIVER_MARKER,
        {
            "tensorrt_llm": str(getattr(tensorrt_llm, "__file__", "")),
            "quickstart": spec["quickstart"],
        },
    )
    llm, base = engine.build_llm(quickstart, list(spec.get("quickstart_argv") or ()))
    try:
        write_records(spec["out"], run_requests(llm, base, spec))
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
