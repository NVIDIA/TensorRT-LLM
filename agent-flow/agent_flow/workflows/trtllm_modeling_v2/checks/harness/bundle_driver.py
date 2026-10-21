#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run several gates' workloads on one engine of the tree under test, one section each.

Usage (rank 0 under ``trtllm-llmapi-launch`` when the world size is above one)::

    <tree's python> bundle_driver.py --spec bundle.json

The engine is built once, as the engine driver builds it
(``engine_driver.build_llm``): the tree's own ``quickstart_advanced.py`` over
the spec's ``quickstart_argv``, with ``modeling_v2`` and the handed-over
KV-cache keys from the launch environment. Each gate in the bundle would have
built this same engine itself: the orchestrator bundles only commands whose
launch and engine argv agree (``checks/bundle.py``). Then each entry of
``sections`` runs on it, in order:

* ``driver``: an engine-driver spec (the boot and acceptance gates'): its
  requests, its reuse probe and its ``MV2FLOW_REUSE`` record, as the engine
  driver prints them (``engine_driver.run_workload``);
* ``logits``: a logit-driver spec (``checks/logit_driver.py``): its records,
  written to the section's ``out``;
* ``eval``: a ``trtllm-eval`` argv, run in this process as the eval shim runs
  it, with the ``LLM`` class it builds its engine through replaced by the
  engine already up (:class:`SharedEngine`). Its own ``shutdown`` does
  nothing: the engine is the bundle's to shut down. Its ``MV2FLOW_REUSE``
  record follows.

Output, on stdout, each one JSON line behind a marker (see ``workload_io.py``):

* ``MV2FLOW_DRIVER`` and ``MV2FLOW_ENGINE`` once, as the engine driver prints them;
* ``MV2FLOW_BUILD`` once when the engine was not built: ``error`` when loading
  the quickstart, importing the package or building raised (the traceback is
  on stderr, and the process exits 1), ``timed_out`` when building ran past
  ``build_timeout``;
* ``MV2FLOW_SECTION`` ``{"key", "begin": true}`` before a section's output,
  and ``{"key", "end": true, "ok": ..., "error" | "timed_out": ...}`` after it.

A section that raises, or a ``trtllm-eval`` that exits non-zero, ends the
bundle there: the engine may be broken, so no later section runs, and the
process exits 1. A section that runs past its own ``timeout`` ends the process
(exit 124) once its end record says so.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import engine_driver  # noqa: E402 -- imported by path, see the package docstring
import engine_facts  # noqa: E402
import engine_reuse  # noqa: E402
import eval_shim  # noqa: E402
import workload_io as wio  # noqa: E402

#: Before and after each section's own output.
SECTION_MARKER = "MV2FLOW_SECTION"
#: Once, when the engine every section shares was not built.
BUILD_MARKER = "MV2FLOW_BUILD"
#: The exit status of a bundle a deadline ended.
TIMED_OUT_EXIT = 124
#: The logit gate's job, imported by path for its request loop.
LOGIT_DRIVER = _HERE.parent / "logit_driver.py"

#: A section kind to what runs it: ``(llm, sampling params, section) -> None``.
SectionRunner = Callable[[Any, Any, Mapping[str, Any]], None]


class SectionFailed(RuntimeError):
    """A section ran to its own end and failed, so the engine is not to be trusted further."""


class Deadline:
    """Ends the process if the block it guards runs past ``seconds``, once ``record`` is printed.

    ``None`` or zero guards nothing. ``end`` is how the process ends
    (``os._exit``: the engine's threads would keep an ordinary exit waiting).
    """

    def __init__(
        self,
        seconds: float | None,
        marker: str,
        record: Mapping[str, Any],
        end: Callable[[int], Any] = os._exit,
    ) -> None:
        self.seconds = float(seconds or 0)
        self.marker = marker
        self.record = dict(record)
        self.end = end
        self._timer: threading.Timer | None = None

    def _expire(self) -> None:
        wio.emit(self.marker, self.record)
        self.end(TIMED_OUT_EXIT)

    def __enter__(self) -> "Deadline":
        if self.seconds > 0:
            self._timer = threading.Timer(self.seconds, self._expire)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._timer is not None:
            self._timer.cancel()
        return False


class SharedEngine:
    """The bundle's engine as one evaluation's ``LLM``: every request it sends is kept, and it stays up.

    Everything but ``generate_async``, ``shutdown`` and the context protocol
    is the engine's own.
    """

    def __init__(self, llm: Any, ledger: engine_reuse.ReuseLedger) -> None:
        self._llm = llm
        self._ledger = ledger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

    def generate_async(self, *args: Any, **kwargs: Any) -> Any:
        result = self._llm.generate_async(*args, **kwargs)
        self._ledger.track(result)
        return result

    def shutdown(self) -> None:
        """The bundle shuts the engine down once its last section is done."""

    def __enter__(self) -> "SharedEngine":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def _logit_driver() -> Any:
    """``checks/logit_driver.py``, imported by path once."""
    name = "mv2flow_logit_driver"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, LOGIT_DRIVER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _read_spec(section: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(Path(str(section["spec"])).read_text(encoding="utf-8"))


def run_driver(llm: Any, base: Any, section: Mapping[str, Any]) -> None:
    """An engine-driver spec's workload, printed as the engine driver prints it."""
    engine_driver.run_workload(llm, base, _read_spec(section))


def run_logits(llm: Any, base: Any, section: Mapping[str, Any]) -> None:
    """A logit-driver spec's requests, their records written to the section's ``out``."""
    spec = _read_spec(section)
    logits = _logit_driver()
    logits.write_records(section.get("out") or spec["out"], logits.run_requests(llm, base, spec))


def run_eval(llm: Any, base: Any, section: Mapping[str, Any]) -> None:
    """``trtllm-eval`` over ``section["argv"]`` in this process, on the bundle's engine.

    As the eval shim runs it (``eval_shim.main``), but the class it builds its
    engine through returns :class:`SharedEngine` over ``llm`` whatever it is
    asked; the orchestrator bundled the evaluation only after holding its
    arguments to the engine's (``checks/bundle.py``). The ``MV2FLOW_REUSE``
    record over its requests goes out whether it passed or raised. A non-zero
    exit raises :class:`SectionFailed`.
    """
    argv = [str(word) for word in section["argv"]]
    if not argv or argv[0] != eval_shim.PROGRAM:
        raise SectionFailed(f"an eval section runs {eval_shim.PROGRAM}, not {argv[:1]}")
    ledger = engine_reuse.ReuseLedger()
    shared = SharedEngine(llm, ledger)
    module = importlib.import_module(eval_shim.EVAL_MODULE)
    original = getattr(module, eval_shim.LLM_ATTRIBUTE, None)
    setattr(module, eval_shim.LLM_ATTRIBUTE, lambda *args, **kwargs: shared)
    saved = sys.argv
    sys.argv = [eval_shim.PROGRAM, *argv[1:]]
    code: Any = 0
    try:
        module.main()
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.argv = saved
        setattr(module, eval_shim.LLM_ATTRIBUTE, original)
        engine_reuse.emit(
            ledger,
            engine_reuse.engine_granularity(llm),
            concurrency=engine_reuse.max_batch_size(llm),
        )
    if code not in (0, None):
        raise SectionFailed(f"{eval_shim.PROGRAM} exited {code}")


SECTION_KINDS: dict[str, SectionRunner] = {
    "driver": run_driver,
    "logits": run_logits,
    "eval": run_eval,
}


def run_section(
    llm: Any, base: Any, section: Mapping[str, Any], end: Callable[[int], Any] = os._exit
) -> bool:
    """One section between its markers; whether it ran to its end and passed.

    Whatever the section raises is the tree under test's to report, in the
    section's end record and its traceback, not this harness's to die of
    before saying which section it was.
    """
    key = str(section["key"])
    wio.emit(SECTION_MARKER, {"key": key, "begin": True})
    expired = {"key": key, "end": True, "ok": False, "timed_out": True}
    try:
        run = SECTION_KINDS[str(section["kind"])]
        with Deadline(section.get("timeout"), SECTION_MARKER, expired, end):
            run(llm, base, section)
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        wio.emit(
            SECTION_MARKER,
            {"key": key, "end": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]},
        )
        return False
    wio.emit(SECTION_MARKER, {"key": key, "end": True, "ok": True})
    return True


def main(argv: list[str] | None = None, *, end: Callable[[int], Any] = os._exit) -> int:
    """Read the spec, build the engine once, run every section on it in order."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True, help="the bundle's spec, a JSON file")
    options = parser.parse_args(argv)
    spec = json.loads(Path(options.spec).read_text(encoding="utf-8"))
    words = [str(word) for word in spec.get("quickstart_argv") or ()]
    observer = engine_facts.WorkerObserver()
    try:
        with Deadline(spec.get("build_timeout"), BUILD_MARKER, {"timed_out": True}, end):
            quickstart = engine_driver.load_quickstart(spec["quickstart"])
            import tensorrt_llm

            wio.emit(
                wio.DRIVER_MARKER,
                {
                    "tensorrt_llm": str(getattr(tensorrt_llm, "__file__", "")),
                    "quickstart": spec["quickstart"],
                },
            )
            observer.install()
            llm, base = engine_driver.build_llm(quickstart, words)
    # Each gate's own driver would have died here the same way; the record
    # says so, so the orchestrator can tell this from a bundle cut short.
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        wio.emit(BUILD_MARKER, {"error": f"{type(exc).__name__}: {exc}"[:500]})
        return 1
    try:
        engine_facts.emit(llm, observer, argv=words)
        for section in spec.get("sections") or ():
            if not run_section(llm, base, section, end):
                return 1
        return 0
    finally:
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()


if __name__ == "__main__":
    sys.exit(main())
