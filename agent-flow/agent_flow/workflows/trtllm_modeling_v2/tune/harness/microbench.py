# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the same micro/module harness on each revision, once per revision.

A harness exports ``benchmark() -> dict[str, Callable[[], object]]``: named cases.
It constructs inputs and graph/eager callables, not timings or revision selection.
This wrapper owns CUDA-event timing, warmup, measured checkout identity and results.
Each case's callable returns what the op or module computes (a tensor, or a list, tuple
or dict of them): the control's run saves those outputs, and the candidate's run, given
them, records per case whether its own are bit-exact and their largest error relative
to the control's largest magnitude (``numerics``). :func:`diverged` reads them roughly:
it catches a broken kernel, not a rounding change.
Run by path in the engine environment; no agent-flow dependency is needed there.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import runpy
import socket
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping

#: How far a case's outputs may stray from the control's (the largest error over the
#: control's largest magnitude) before the change is taken for a broken one.
ROUGH_TOLERANCE = 0.05


def timings(raw: object) -> dict[str, float]:
    """Require nonempty, named cases with finite positive latency measurements."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError("benchmark must return a nonempty case -> latency_us mapping")
    if any(not isinstance(k, str) or not k.strip() for k in raw):
        raise ValueError("benchmark cases must have nonempty names")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
        for v in raw.values()
    ):
        raise ValueError("every latency_us must be finite and positive")
    return {k: float(v) for k, v in raw.items()}


def load_result(path: Path, *, commit: str, harness: str, cases: list[str]) -> dict:
    """Read wrapper evidence, rejecting another revision, harness, or case set."""
    data = json.loads(path.read_text())
    if data["commit"] != commit or data["harness_sha256"] != harness:
        raise ValueError("microbench revision or harness does not match this verification")
    if not data.get("hostname") or not data.get("module"):
        raise ValueError("microbench lacks host/import provenance")
    data["latency_us"] = timings(data["latency_us"])
    if set(data["latency_us"]) != set(cases):
        raise ValueError("microbench cases differ from the declared workload")
    return data


def deltas(candidate: Mapping[str, float], control: Mapping[str, float]) -> dict[str, float]:
    """Higher-is-better speedup for each identical latency case."""
    if candidate.keys() != control.keys():
        raise ValueError("microbench A/B case sets differ")
    return {f"{case}.speedup": control[case] / value - 1 for case, value in candidate.items()}


def diverged(numerics: Mapping[str, Any], tolerance: float = ROUGH_TOLERANCE) -> str:
    """The cases whose outputs left the control's, roughly; ``""`` when none did.

    A case that returned no tensor is not judged; a non-finite error (NaN, inf) always is.
    """
    problems = []
    for case, got in sorted(numerics.items()):
        if not isinstance(got, Mapping) or got.get("missing") or got.get("exact"):
            continue
        error = float(got.get("error") or 0.0)
        if not math.isfinite(error) or error > tolerance:
            problems.append(f"{case}: error {error:.3g} past {tolerance:g}")
    return "; ".join(problems)


def _tensors(value: object) -> list:
    """The tensors a case returned, in a stable order; [] for none."""
    import torch

    if isinstance(value, torch.Tensor):
        return [value.detach().to("cpu", copy=True)]
    if isinstance(value, Mapping):
        return [t for key in sorted(value) for t in _tensors(value[key])]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _tensors(item)]
    return []


def _compare(candidate: list, control: list) -> dict:
    """One case's numerics: bit-exact or not, and the largest error over the control's magnitude."""
    import torch

    if not control:
        return {"missing": "the case returned no tensor to compare"}
    if len(candidate) != len(control) or any(
        a.shape != b.shape or a.dtype != b.dtype for a, b in zip(candidate, control)
    ):
        return {"missing": "the outputs differ in number, shape or dtype"}
    exact = all(torch.equal(a, b) for a, b in zip(candidate, control))
    error = 0.0
    for a, b in zip(candidate, control):
        if a.is_floating_point():
            scale = float(b.float().abs().max()) or 1.0
            error = max(error, float((a.float() - b.float()).abs().max()) / scale)
    return {"exact": exact, "error": error, "dtype": str(control[0].dtype).removeprefix("torch.")}


def _identity(tree: Path, commit: str) -> None:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(tree), *args], text=True).strip()

    if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "-uno"):
        raise ValueError("benchmark checkout must remain at its declared clean commit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--harness-sha256", required=True)
    parser.add_argument("--outputs", type=Path, help="where this run saves its cases' outputs")
    parser.add_argument("--reference", type=Path, help="the control's outputs, to compare with")
    args = parser.parse_args()
    _identity(args.tree, args.commit)
    source = args.script.read_bytes()
    if hashlib.sha256(source).hexdigest() != args.harness_sha256:
        raise ValueError("verification harness changed")
    module = importlib.import_module("tensorrt_llm")
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(args.tree.resolve()):
        raise ValueError(f"tensorrt_llm imported from {origin}, not {args.tree}")
    import torch

    cases = runpy.run_path(str(args.script))["benchmark"]()
    if not isinstance(cases, dict) or not cases or not all(callable(fn) for fn in cases.values()):
        raise ValueError("benchmark must return named case callables, not claimed timings")
    result = {}
    for name, fn in cases.items():
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(100)
        ]
        for start, end in events:
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        result[name] = statistics.median(start.elapsed_time(end) * 1000 for start, end in events)
    result = timings(result)
    numerics = {}
    if args.outputs or args.reference:
        outputs = {name: _tensors(fn()) for name, fn in cases.items()}
        torch.cuda.synchronize()
    if args.outputs:
        args.outputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, args.outputs)
    if args.reference:
        control = torch.load(args.reference) if args.reference.is_file() else {}
        numerics = {name: _compare(outputs[name], control.get(name, [])) for name in cases}
    _identity(args.tree, args.commit)
    if args.script.read_bytes() != source:
        raise ValueError("verification harness changed while running")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "commit": args.commit,
                "harness_sha256": args.harness_sha256,
                "hostname": socket.gethostname(),
                "module": str(origin),
                "latency_us": result,
                **({"numerics": numerics} if args.reference else {}),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
