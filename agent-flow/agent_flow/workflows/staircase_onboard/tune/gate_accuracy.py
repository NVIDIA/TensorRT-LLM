#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The accuracy gate a perf keep must pass, run inside the container (by path).

Merges the identity config with every kept overlay, **forces**
``modeling_v2: require`` into the merged LLM-API yaml (selection is an LLM-API
argument; an environment variable selects nothing), runs ``trtllm-eval`` and
writes ``result.json`` with the score, the floor and the verdict. The floor is
the anchor minus ``k`` standard errors of the benchmark's own sample count.

With ``--tests``, the target's test suite runs first on the same tree: a stack
of candidates that each passed their own op tests can still break a block test
neither ran. A failing suite fails the gate without spending the eval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import resolve_config  # noqa: E402

TASK_SAMPLES = {"gsm8k": 1319, "mmlu": 14042}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", action="append", type=Path, default=[])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--anchor", type=float, required=True)
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--eval-args", default='--apply_chat_template --fewshot_as_multiturn --chat_template_kwargs={"enable_thinking":false} --random_seed=0 --max_output_length=512 --max_input_length=4096')
    p.add_argument("--tests", action="append", default=[],
                   help="repo-relative test path run with pytest from tests/unittest before the eval")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tests = None
    if args.tests:
        root = Path.cwd() / "tests" / "unittest"
        paths = [str(Path(t).relative_to("tests/unittest")) if t.startswith("tests/unittest/") else t for t in args.tests]
        tlog = args.out / "tests.log"
        with tlog.open("w") as fh:
            trc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths], cwd=root,
                                 stdout=fh, stderr=subprocess.STDOUT).returncode
        lines = [l for l in tlog.read_text(errors="replace").splitlines() if l.strip()]
        tests = {"rc": trc, "summary": lines[-1] if lines else "", "log": str(tlog), "paths": args.tests,
                 "failed": [l.split(" ", 1)[1] for l in lines if l.startswith("FAILED ")][:50]}
        if trc != 0:
            result = {"task": args.task, "score": None, "anchor": args.anchor, "verdict": "FAIL",
                      "reason": f"target tests failed: {tests['summary']}", "tests": tests,
                      "tree": os.environ.get("STAIRCASE_TREE", "")}
            (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"[gate] tests FAILED: {tests['summary']}")
            return 1
    config = resolve_config(args.config, [])
    config["modeling_v2"] = "require"
    env_overrides = config.get("env_overrides") or {}
    merged = args.out / "gate_llm_api.yaml"
    merged.write_text(yaml.safe_dump(config, sort_keys=False))
    argv = ["trtllm-eval", f"--model={args.model}", "--tp_size=1", f"--extra_llm_api_options={merged}", args.task]
    argv += [a for a in re.split(r" (?=--)", args.eval_args) if a]
    env = dict(os.environ) | {str(k): str(v) for k, v in env_overrides.items()}
    started = time.time()
    log = args.out / "eval.log"
    with log.open("w") as fh:
        rc = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    text = log.read_text(errors="replace")
    m = re.search(rf"lm-eval {args.task} average accuracy: ([0-9.]+)", text)
    score = float(m.group(1)) if m else None
    n = TASK_SAMPLES.get(args.task, 1000)
    p_ = args.anchor / 100.0
    se = 100.0 * math.sqrt(p_ * (1 - p_) / n)
    floor = args.anchor - args.k * se
    verdict = "PASS" if (rc == 0 and score is not None and score >= floor) else "FAIL"
    result = {"task": args.task, "score": score, "anchor": args.anchor, "se": round(se, 4), "k": args.k,
              "floor": round(floor, 4), "verdict": verdict, "rc": rc, "configs": [str(c) for c in args.config],
              "merged": str(merged), "seconds": round(time.time() - started, 1),
              "tree": os.environ.get("STAIRCASE_TREE", ""), "argv": argv, "tests": tests}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[gate] {args.task} score={score} floor={floor:.2f} verdict={verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
