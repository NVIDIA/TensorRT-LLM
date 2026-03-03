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


def run_cmd(cmd, cwd, env=None):
    """Run command and return (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def parse_env_and_cmd(base_cmd):
    """Extract environment variables from command string.

    Splits base_cmd into environment variables (VAR=value) and command tokens.
    Environment variables appear before the executable name.

    Returns: (env_dict, cmd_tokens)
        env_dict: dictionary of environment variables
        cmd_tokens: list of command tokens starting with executable
    """
    tokens = shlex.split(base_cmd)
    env_vars = {}
    cmd_tokens = []

    for token in tokens:
        # Check if token looks like VAR=value (contains = and doesn't start with -)
        if "=" in token and not token.startswith("-"):
            key, value = token.split("=", 1)
            # Check if key is a valid identifier
            if key.isidentifier():
                env_vars[key] = value
                continue
        # Once we hit a token that's not an env var, it's the start of the command
        cmd_tokens.append(token)

    return env_vars, cmd_tokens


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

    # Parse environment variables and command tokens from base_cmd
    base_env_vars, base_tokens = parse_env_and_cmd(base_cmd)

    # Prepare environment: inherit from parent, add extracted vars, set LLM_ROOT
    run_env = os.environ.copy()
    run_env.update(base_env_vars)

    defs_cwd = llm_src / "tests" / "integration" / "defs"
    defs_cwd = str(defs_cwd)
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

        rc, stdout, stderr = run_cmd(cmd, cwd=defs_cwd, env=run_env)

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
