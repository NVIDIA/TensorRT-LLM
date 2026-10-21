#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the tree's ``quickstart_advanced.py`` the way a witness leg needs it.

Usage, under the tree's interpreter (rank 0 under ``trtllm-llmapi-launch`` above
world size one)::

    <python> quickstart_shim.py <tree>/examples/llm-api/quickstart_advanced.py <quickstart argv>

The quickstart runs unchanged, and the ``LLM`` it builds gets what the
engine driver adds (:func:`engine_driver.engine_kwargs`: the target of
``TRTLLM_MODELING_V2``, the KV-cache keys of ``MV2FLOW_KV_CACHE_CONFIG``).
Under ``TRTLLM_WITNESS_FULL_OUTPUT=1`` each streamed request's sequences are
printed as ``[i] Witness token ids: [...]`` and ``[i] Witness text: '...'``,
unless the quickstart prints them itself. The ``MV2FLOW_ENGINE`` and
``MV2FLOW_REUSE`` records are printed as the engine driver prints them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_facts  # noqa: E402 -- imported by path, as the driver is
import engine_reuse  # noqa: E402
from engine_driver import (  # noqa: E402
    MODELING_V2_ENV,
    engine_kwargs,
    kv_cache_overrides,
    load_quickstart,
)

WITNESS_OUTPUT_ENV = "TRTLLM_WITNESS_FULL_OUTPUT"


def print_sequences(tag: str, output: Any) -> None:
    """The whole continuation of every sequence of one finished request."""
    for sequence in output.outputs:
        print(f"{tag} Witness token ids: {list(sequence.token_ids)}")
        print(f"{tag} Witness text: {sequence.text!r}")


def wrap_llm(
    llm_class: Any,
    *,
    mode: str | None,
    full_output: bool,
    kv_cache: Mapping[str, Any] | None = None,
    on_built: Callable[[Any], Any] | None = None,
    ledger: engine_reuse.ReuseLedger | None = None,
) -> Any:
    """``llm_class``, building the ``mode`` target and printing whole streamed continuations.

    ``on_built`` is called with each engine once its constructor returns.
    ``ledger`` keeps every request submitted.
    """
    counter = iter(range(sys.maxsize))

    class WitnessLLM(llm_class):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **engine_kwargs(kwargs, mode=mode, kv_cache=kv_cache))
            if on_built is not None:
                on_built(self)

        def generate_async(self, *args: Any, **kwargs: Any) -> Any:
            result = super().generate_async(*args, **kwargs)
            if ledger is not None:
                ledger.track(result)
            if not (full_output and kwargs.get("streaming")):
                return result

            def stream() -> Any:
                last = None
                for last in result:
                    yield last
                if last is not None:
                    print_sequences(f"[{next(counter)}]", last)

            return stream()

    WitnessLLM.__name__ = getattr(llm_class, "__name__", "LLM")
    return WitnessLLM


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit("usage: quickstart_shim.py <quickstart_advanced.py> [quickstart argv]")
    quickstart = load_quickstart(argv[0])
    full_output = os.environ.get(WITNESS_OUTPUT_ENV) == "1" and not hasattr(
        quickstart, "print_witness_output"
    )
    observer = engine_facts.WorkerObserver()
    observer.install()
    ledger = engine_reuse.ReuseLedger()
    built: list[tuple[engine_reuse.Granularity, int | None]] = []

    def on_built(llm: Any) -> None:
        engine_facts.emit(llm, observer, argv=argv[1:])
        built.append((engine_reuse.engine_granularity(llm), engine_reuse.max_batch_size(llm)))

    quickstart.LLM = wrap_llm(
        quickstart.LLM,
        mode=os.environ.get(MODELING_V2_ENV),
        full_output=full_output,
        kv_cache=kv_cache_overrides(),
        on_built=on_built,
        ledger=ledger,
    )
    sys.argv = [argv[0], *argv[1:]]
    quickstart.main()
    if built:
        granularity, concurrency = built[-1]
        engine_reuse.emit(ledger, granularity, concurrency=concurrency)


if __name__ == "__main__":
    main(sys.argv[1:])
