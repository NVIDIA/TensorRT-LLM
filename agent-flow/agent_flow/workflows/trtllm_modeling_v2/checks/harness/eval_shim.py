#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the tree's ``trtllm-eval`` in this process, and report the engine it built and what it reused.

Usage, under the tree's interpreter (rank 0 under ``trtllm-llmapi-launch`` above
world size one)::

    <python> eval_shim.py trtllm-eval <trtllm-eval argv>

``trtllm-eval`` runs unchanged, but the ``LLM`` class it builds its engine
with is replaced by a subclass that prints, as the engine driver does, the
``MV2FLOW_ENGINE`` record once the engine is up and the ``MV2FLOW_REUSE``
record once ``trtllm-eval`` returns. A tree whose ``trtllm-eval`` has no
``PyTorchLLM`` class to replace runs as is, and prints neither record.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_facts  # noqa: E402 -- imported by path, as the driver is
import engine_reuse  # noqa: E402

PROGRAM = "trtllm-eval"
EVAL_MODULE = "tensorrt_llm.commands.eval"
#: The name ``EVAL_MODULE`` builds its engine through.
LLM_ATTRIBUTE = "PyTorchLLM"


def observed_llm(
    llm_class: Any,
    observer: engine_facts.WorkerObserver,
    ledger: engine_reuse.ReuseLedger,
    built: list[tuple[engine_reuse.Granularity, int | None]],
) -> Any:
    """``llm_class``, printing its engine record once built and keeping every request it is sent.

    Each engine's reuse granularity and batch limit go to ``built`` while it
    is up: ``trtllm-eval`` shuts its engine down before it returns.
    """

    class EvalLLM(llm_class):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            engine_facts.emit(self, observer)
            built.append((engine_reuse.engine_granularity(self), engine_reuse.max_batch_size(self)))

        def generate_async(self, *args: Any, **kwargs: Any) -> Any:
            result = super().generate_async(*args, **kwargs)
            ledger.track(result)
            return result

    EvalLLM.__name__ = getattr(llm_class, "__name__", "LLM")
    return EvalLLM


def main(argv: list[str]) -> None:
    if not argv or argv[0] != PROGRAM:
        raise SystemExit(f"usage: eval_shim.py {PROGRAM} [{PROGRAM} argv]")
    observer = engine_facts.WorkerObserver()
    observer.install()
    ledger = engine_reuse.ReuseLedger()
    built: list[tuple[engine_reuse.Granularity, int | None]] = []
    module = importlib.import_module(EVAL_MODULE)
    llm_class = getattr(module, LLM_ATTRIBUTE, None)
    if isinstance(llm_class, type):
        setattr(module, LLM_ATTRIBUTE, observed_llm(llm_class, observer, ledger, built))
    sys.argv = [PROGRAM, *argv[1:]]
    try:
        module.main()
    finally:
        # The console script ends in `sys.exit`; the record goes out on the
        # way, whether the evaluation passed or raised.
        if built:
            granularity, concurrency = built[-1]
            engine_reuse.emit(ledger, granularity, concurrency=concurrency)


if __name__ == "__main__":
    main(sys.argv[1:])
