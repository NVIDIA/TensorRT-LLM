#!/usr/bin/env python3
"""Check `serve_wait.py`'s log-against-config verdicts.

  test_serve_wait.py

`check()` is what stands between a server that came up wrong and a benchmark
that reports its numbers anyway, and it is a pure function of the log text and
the config -- no server needed to hold it to its answers.

Only the verdicts a passing server would hide are covered: the worker-count
arithmetic, a config whose quantization never took, a warmup that missed a
shape, and the fatal pattern, which aborts a healthy twenty-minute startup if
it fires on the wrong line.
"""

import importlib.util
import sys
from pathlib import Path

SERVE_WAIT = Path(__file__).resolve().parent.parent / "scripts" / "serve_wait.py"

_spec = importlib.util.spec_from_file_location("serve_wait", SERVE_WAIT)
serve_wait = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve_wait)

WARMED = (
    "Running warmup for LTX2Pipeline: 2 shape(s) [768x1280x121, 704x1280x121]\n"
    "Warmup completed in 42.0s\n"
)
GOOD_LOG = (
    "World size: 8\n"
    "Quantization: FP8\n"
    "Dynamic weight quant: True\n"
    "Attention backend: TRTLLM\n" + WARMED
)
GOOD_CFG = {
    "parallel_config": {"cfg_size": 2, "ulysses_size": 4},
    "quant_config": {"quant_algo": "FP8"},
    "attention_config": {"backend": "TRTLLM"},
    "compilation_config": {"resolutions": [[768, 1280], [704, 1280]], "num_frames": [121]},
}

# (name, log, config, the substring each expected FAIL carries)
CASES = [
    (
        "cp_size folds attn2d into the product",
        "World size: 4\nAttention backend: VANILLA\n" + WARMED,
        {
            "parallel_config": {"cfg_size": 2, "attn2d_size": [2, 2], "ulysses_size": 2},
            "compilation_config": {
                "resolutions": [[768, 1280], [704, 1280]],
                "num_frames": [121],
            },
        },
        ["world size 4 but parallel_config implies 16"],
    ),
    (
        "a quant_config that never took",
        "World size: 8\nAttention backend: TRTLLM\n" + WARMED,
        {
            "parallel_config": {"cfg_size": 2, "ulysses_size": 4},
            "quant_config": {"quant_algo": "FP8"},
        },
        ["it loaded BF16"],
    ),
    (
        "a checkpoint whose quantization the loader does not map",
        "World size: 1\n_quantization_metadata format 'modelopt_v2' is not supported\n" + WARMED,
        {"compilation_config": {"resolutions": [[768, 1280], [704, 1280]], "num_frames": [121]}},
        ["IGNORED"],
    ),
    (
        "a warmup that skipped one of the asked-for shapes",
        "World size: 1\nRunning warmup for LTX2Pipeline: 1 shape(s) [768x1280x121]\n"
        "Warmup completed in 20.0s\n",
        {"compilation_config": {"resolutions": [[768, 1280], [704, 1280]], "num_frames": [121]}},
        ["(704, 1280, 121)"],
    ),
    (
        "an attention backend other than the one configured",
        "World size: 8\nQuantization: FP8\nAttention backend: VANILLA\n" + WARMED,
        GOOD_CFG,
        ["attention backend VANILLA but config says TRTLLM"],
    ),
]

# Lines that must fire, and lines that must not -- a false positive kills a
# healthy startup.
FATAL_LINES = ["Killed", "srun: error: node1: task 0: Segmentation fault", "[FATAL] rank 3 aborted"]
BENIGN_LINES = [
    "INFO: killed 0 stale processes",
    "loading checkpoint, this is not an error",
    "Assertion coverage: 98%",
]


def main() -> int:
    problems = []

    for name, log, cfg, expected in CASES:
        fails = [message for level, message in serve_wait.check(log, cfg) if level == "FAIL"]
        for wanted in expected:
            if not any(wanted in message for message in fails):
                problems.append(f"{name}: no FAIL carrying {wanted!r}; got {fails}")

    fails = [message for level, message in serve_wait.check(GOOD_LOG, GOOD_CFG) if level == "FAIL"]
    if fails:
        problems.append(f"a server matching its config still failed: {fails}")

    for line in FATAL_LINES:
        if not serve_wait.FATAL.search(line):
            problems.append(f"FATAL misses {line!r}")
    for line in BENIGN_LINES:
        if serve_wait.FATAL.search(line):
            problems.append(f"FATAL fires on {line!r}, which would abort a healthy startup")

    for problem in problems:
        print(problem, file=sys.stderr)
    print(
        f"{len(CASES) + 1} verdicts, {len(FATAL_LINES) + len(BENIGN_LINES)} lines, "
        f"{len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
