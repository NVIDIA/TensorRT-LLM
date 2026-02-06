#!/usr/bin/env python3
"""Run isolated tests sequentially.

Reports failures but does not retry.
Retries are handled by the caller (Groovy/Jenkins).

Usage:
  python3 run_isolated_tests.py --llm-src /path/to/src --isolate-list /path/to/isolate.txt \
    --base-cmd "LLM_ROOT=... pytest -vv ..." --stage-name MyStage --output-dir /workspace/MyStage
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd):
    """Run command and return (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-src", required=True)
    p.add_argument("--isolate-list", required=True)
    p.add_argument("--base-cmd", required=True)
    p.add_argument("--stage-name", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    llm_src = Path(args.llm_src)
    isolate_list = Path(args.isolate_list)
    base_cmd = args.base_cmd
    stage = args.stage_name
    output_dir = Path(args.output_dir)

    if not isolate_list.exists():
        print(f"Isolate list not found: {isolate_list}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    with isolate_list.open() as f:
        tests = [line.strip() for line in f if line.strip()]

    if not tests:
        print("No isolated tests to run.")
        sys.exit(0)

    # Prepare base command tokens
    base_tokens = shlex.split(base_cmd)

    defs_cwd = llm_src.rstrip("/") + "/tests/integration/defs"
    any_failed = False
    failed_tests = []

    for i, test in enumerate(tests):
        single_test_file = str(isolate_list) + f"_isolated_{i}.txt"
        with open(single_test_file, "w") as tf:
            tf.write(test + "\n")

        cmd = []
        cmd.extend(base_tokens)
        # Append per-test args
        cmd += [f"--test-list={single_test_file}"]
        cmd += [f"--test-prefix={stage}"]
        cmd += [f"--csv={output_dir}/report_isolated_{i}.csv"]
        cmd += ["--periodic-junit-xmlpath", f"{output_dir}/results_isolated_{i}.xml"]
        cmd += ["--cov-append"]

        print("Running isolated test [{}]: {}".format(i, test))
        print("Cmd:", " ".join(shlex.quote(x) for x in cmd))

        rc, stdout, stderr = run_cmd(cmd, cwd=defs_cwd)

        if stdout:
            print(stdout)
        if stderr:
            print(stderr, file=sys.stderr)

        if rc != 0:
            print(f"Isolated test {i} (test: {test}) failed with rc={rc}", file=sys.stderr)
            any_failed = True
            failed_tests.append(
                {"index": i, "test": test, "xml": f"{output_dir}/results_isolated_{i}.xml"}
            )

        try:
            os.remove(single_test_file)
        except OSError:
            pass

    if any_failed:
        print("\n=== Failed Isolated Tests ===", file=sys.stderr)
        for failed in failed_tests:
            print(f"  [{failed['index']}] {failed['test']} -> {failed['xml']}", file=sys.stderr)
        sys.exit(1)

    print("All isolated tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
