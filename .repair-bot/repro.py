"""Verification script for bug 6029035.

Bug synopsis: trtllm-bench startup shows UCX ERROR, flash_attn kernel
override warning, and modelopt incompatibility.

This script imports tensorrt_llm in a subprocess, captures merged
stdout+stderr, and asserts none of the bug-specified noisy patterns appear.
"""

import re
import subprocess
import sys

# Patterns that should NOT appear in tensorrt_llm import / trtllm-bench
# startup output. Each entry is (label, compiled regex).
FORBIDDEN_PATTERNS = [
    ("UCX ERROR", re.compile(r"UCX\s+ERROR", re.IGNORECASE)),
    (
        "flash_attn kernel override warning",
        re.compile(
            r"flash[_-]?attn.*(override|overriding|already registered|conflict)",
            re.IGNORECASE,
        ),
    ),
    (
        "modelopt incompatibility",
        re.compile(r"incompatible with nvidia-modelopt", re.IGNORECASE),
    ),
]


def run_capture(cmd):
    """Run cmd, return (returncode, combined_output)."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
    )
    return result.returncode, result.stdout.decode("utf-8", errors="replace")


def check_output(label, output):
    """Return list of (pattern_label, matched_line) tuples for any hits."""
    hits = []
    for pat_label, pat in FORBIDDEN_PATTERNS:
        for line in output.splitlines():
            if pat.search(line):
                hits.append((pat_label, line))
                break
    return hits


def main():
    failures = []

    # Scenario 1: bare import of tensorrt_llm
    rc, out = run_capture([sys.executable, "-c", "import tensorrt_llm"])
    print(f"--- import tensorrt_llm (rc={rc}) ---")
    print(out)
    if rc != 0:
        failures.append(f"import tensorrt_llm exited non-zero ({rc})")
    hits = check_output("import", out)
    for pat_label, line in hits:
        failures.append(f"[import] forbidden pattern '{pat_label}' matched: {line!r}")

    # Scenario 2: trtllm-bench --help (the actual command in the bug synopsis)
    rc, out = run_capture(["trtllm-bench", "--help"])
    print(f"--- trtllm-bench --help (rc={rc}) ---")
    print(out)
    if rc != 0:
        failures.append(f"trtllm-bench --help exited non-zero ({rc})")
    hits = check_output("trtllm-bench", out)
    for pat_label, line in hits:
        failures.append(f"[trtllm-bench] forbidden pattern '{pat_label}' matched: {line!r}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
