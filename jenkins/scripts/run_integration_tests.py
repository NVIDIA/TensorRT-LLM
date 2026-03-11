#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Reusable script to run integration tests: regular tests, isolated tests,
# rerun failed cases, and generate the rerun report. Can be used from Jenkins
# or any shell.
#
# Usage:
#   python3 run_integration_tests.py \
#     --llm-src /path/to/TensorRT-LLM/src \
#     --stage-name L0_A10 \
#     --output-dir /workspace/L0_A10 \
#     --waives-file /path/to/waives.txt \
#     [--regular-test-list /path/to/list_regular.txt] \
#     [--isolate-test-list /path/to/list_isolate.txt] \
#     [--fail-signatures "sig1,sig2"] \
#     [--perf-mode] [--detailed-log] [--run-ray] \
#     [--tester-cores 12] [--container-port-start 10000] [--container-port-num 100] \
#     [--model-cache-dir /path/to/models]
#
# Environment: MODEL_CACHE_DIR, LD_LIBRARY_PATH (optional). Other env vars
# are set by the script from --llm-src and options.

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Default placeholder when no fail signatures are provided (test_rerun.py
# treats empty string as matching everything).
_NO_SIGNATURE_PLACEHOLDER = "__NO_SIGNATURE_MATCH__"


def _run_cmd(cmd, cwd, env=None, check=False, capture=True):
    """Run command. If capture=True, return (returncode, stdout, stderr)."""
    run_env = (env or os.environ).copy()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=run_env,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    if capture:
        return result.returncode, result.stdout or "", result.stderr or ""
    return result.returncode


def _get_trtllm_whl_path():
    """Get tensorrt_llm package location from pip."""
    rc, out, _ = _run_cmd(
        ["pip3", "show", "tensorrt_llm"],
        cwd=os.getcwd(),
    )
    if rc != 0:
        raise RuntimeError("pip3 show tensorrt_llm failed; is tensorrt_llm installed?")
    for line in out.splitlines():
        if line.startswith("Location:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Could not find Location in pip show tensorrt_llm output")


def _build_pytest_base_command(
    llm_src: Path,
    stage_name: str,
    output_dir: Path,
    waives_file: Path,
    trtllm_whl_path: str,
    coverage_config: Path,
    *,
    perf_mode: bool = False,
    detailed_log: bool = False,
    run_ray: bool = False,
    tester_cores: str = "12",
    container_port_start: int = 0,
    container_port_num: int = 0,
    model_cache_dir: str = "",
) -> tuple[list[str], dict[str, str]]:
    """Build pytest base command and env (equivalent to getPytestBaseCommandLine)."""
    pytest_timeout = "3600"
    extra_internal_env = (
        f'__LUNOWUD="-thread_pool_size={tester_cores}" '
        f"CPP_TEST_TIMEOUT_OVERRIDDEN={pytest_timeout} NCCL_DEBUG=INFO"
    )
    port_env = ""
    if container_port_start > 0 and container_port_num > 0:
        port_env = (
            f"CONTAINER_PORT_START={container_port_start} CONTAINER_PORT_NUM={container_port_num}"
        )

    env_vars = {
        "LLM_ROOT": str(llm_src),
        "LLM_BACKEND_ROOT": str(llm_src / "triton_backend"),
        "LLM_MODELS_ROOT": model_cache_dir or os.environ.get("MODEL_CACHE_DIR", ""),
        "MODEL_CACHE_DIR": model_cache_dir or os.environ.get("MODEL_CACHE_DIR", ""),
        "COLUMNS": "300",
        **dict(
            x.split("=", 1)
            for x in (extra_internal_env + " " + port_env).strip().split()
            if "=" in x
        ),
    }

    cmd = [
        "pytest",
        "-vv",
        "--timeout-method=thread",
        "--apply-test-list-correction",
        f"--timeout={pytest_timeout}",
        f"--rootdir={llm_src}/tests/integration/defs",
        f"--test-prefix={stage_name}",
        f"--waives-file={waives_file}",
        f"--output-dir={output_dir}/",
        f"--csv={output_dir}/report.csv",
        "-o",
        "junit_logging=out-err",
        f"--cov={llm_src}/examples/",
        f"--cov={llm_src}/tensorrt_llm/",
        f"--cov={trtllm_whl_path}/tensorrt_llm/",
        "--cov-report=",
        f"--cov-config={coverage_config}",
        "--periodic-junit",
        "--periodic-junit-xmlpath",
        f"{output_dir}/results.xml",
        "--periodic-batch-size=1",
        "--periodic-save-unfinished-test",
    ]
    if detailed_log:
        cmd.append("-s")
    if perf_mode:
        cmd.extend(
            [
                "--perf",
                "--perf-log-formats",
                "csv",
                "--perf-log-formats",
                "yaml",
                "--enable-gpu-clock-lock",
            ]
        )
    if run_ray:
        cmd.append("--run-ray")

    return cmd, env_vars


def _run_pytest(cmd: list[str], env: dict[str, str], llm_src: Path, extra_args: list[str]) -> int:
    """Run pytest in tests/integration/defs with merged env and optional extra args."""
    defs_cwd = llm_src / "tests" / "integration" / "defs"
    run_env = os.environ.copy()
    run_env.update(env)
    full_cmd = cmd + extra_args
    print(f"[run_integration_tests] Running: {' '.join(shlex.quote(x) for x in full_cmd)}")
    return _run_cmd(full_cmd, cwd=str(defs_cwd), env=run_env, capture=False)


def _rerun_failed_tests(
    stage_name: str,
    llm_src: Path,
    output_dir: Path,
    result_file: str,
    test_type: str,
    base_cmd: list[str],
    base_env: dict[str, str],
    fail_signatures: str,
    script_dir: Path,
) -> bool:
    """Generate rerun lists with test_rerun.py, then run pytest for rerun_1 and rerun_2.

    Return True if rerun still failed, False otherwise.
    """
    result_path = output_dir / result_file
    if not result_path.exists():
        print(f"[run_integration_tests] No {result_file} found, skipping rerun for {test_type}")
        return True

    rerun_dir = output_dir / "rerun" / test_type
    rerun_dir.mkdir(parents=True, exist_ok=True)

    sigs = fail_signatures.strip() or _NO_SIGNATURE_PLACEHOLDER
    _run_cmd(
        [
            sys.executable,
            str(script_dir / "test_rerun.py"),
            "generate_rerun_tests_list",
            f"--output-dir={rerun_dir}",
            f"--input-file={result_path}",
            f"--fail-signatures={sigs}",
        ],
        cwd=str(llm_src),
        check=True,
    )

    rerun_0 = rerun_dir / "rerun_0.txt"
    if rerun_0.exists():
        print(
            "[run_integration_tests] Some failed tests cannot be rerun (rerun_0.txt present), skipping rerun."
        )
        return True

    valid_count = 0
    for times in [1, 2]:
        rlist = rerun_dir / f"rerun_{times}.txt"
        if rlist.exists():
            count = len([line for line in rlist.read_text().splitlines() if line.strip()])
            valid_count += count
            print(
                f"[run_integration_tests] Found {count} {test_type} tests to rerun {times} time(s)"
            )

    if valid_count > 5:
        print(f"[run_integration_tests] More than 5 failed {test_type} tests, skipping rerun.")
        return True
    if valid_count == 0:
        print(f"[run_integration_tests] No failed {test_type} tests to rerun.")
        return True

    is_rerun_failed = False
    no_need = ["--splitting-algorithm", "--splits", "--group", "--cov"]
    need_change = ["--test-list", "--csv", "--periodic-junit-xmlpath"]

    for times in [1, 2]:
        rlist = rerun_dir / f"rerun_{times}.txt"
        if not rlist.exists():
            continue
        xml_file = rerun_dir / f"rerun_results_{times}.xml"
        filtered_cmd = [
            x
            for x in base_cmd
            if not any(n in x for n in no_need) and not any(c in x for c in need_change)
        ]
        extra = [
            f"--test-list={rlist}",
            f"--csv={rerun_dir}/rerun_report_{times}.csv",
            "--periodic-junit-xmlpath",
            str(xml_file),
            "--reruns",
            str(times - 1),
        ]
        rc = _run_pytest(filtered_cmd, base_env, llm_src, extra)
        if rc != 0:
            if not xml_file.exists():
                raise RuntimeError(f"{test_type} tests crashed during rerun attempt.")
            print(f"[run_integration_tests] {test_type} tests still failed after rerun attempt.")
            is_rerun_failed = True

    print(f"[run_integration_tests] isRerunFailed for {test_type}: {is_rerun_failed}")
    return is_rerun_failed


def _run_regular_tests(
    llm_src: Path,
    output_dir: Path,
    stage_name: str,
    regular_list: Path,
    base_cmd: list[str],
    base_env: dict[str, str],
    fail_signatures: str,
    script_dir: Path,
) -> tuple[bool, bool]:
    """Run regular tests. Return (any_ran, rerun_failed)."""
    lines = [line.strip() for line in regular_list.read_text().splitlines() if line.strip()]
    if not lines:
        print(f"[run_integration_tests] No regular tests to run for stage {stage_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_empty_results_xml(output_dir, stage_name)
        return False, False

    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous run (keep .coveragerc and .coverage.*)
    for f in output_dir.glob("*"):
        if f.name.startswith(".") or f.name == "rerun":
            continue
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            import shutil

            shutil.rmtree(f, ignore_errors=True)

    extra = [f"--test-list={regular_list}"]
    rc = _run_pytest(base_cmd, base_env, llm_src, extra)
    if rc == 0:
        return True, False

    rerun_failed = _rerun_failed_tests(
        stage_name,
        llm_src,
        output_dir,
        "results.xml",
        "regular",
        base_cmd,
        base_env,
        fail_signatures,
        script_dir,
    )
    return True, rerun_failed


def _write_empty_results_xml(output_dir: Path, stage_name: str) -> None:
    """Write an empty JUnit results.xml."""
    path = output_dir / "results.xml"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<testsuites>\n"
        f'<testsuite name="{stage_name}" errors="0" failures="0" skipped="0" tests="0" time="0.0">\n'
        "</testsuite>\n"
        "</testsuites>\n"
    )


def _run_isolated_tests(
    llm_src: Path,
    output_dir: Path,
    stage_name: str,
    isolate_list: Path,
    base_cmd: list[str],
    base_env: dict[str, str],
    fail_signatures: str,
    script_dir: Path,
) -> bool:
    """Run each isolated test in its own pytest invocation, then rerun failed.

    Return True if any isolated test still failed after rerun.
    """
    tests = [line.strip() for line in isolate_list.read_text().splitlines() if line.strip()]
    if not tests:
        print(f"[run_integration_tests] No isolated tests to run for stage {stage_name}")
        return False

    # Base command without test-list, test-prefix, csv, periodic-junit-xmlpath (set per run)
    filtered = [
        x
        for x in base_cmd
        if not re.search(r"^--test-list=", x)
        and not re.search(r"^--test-prefix=", x)
        and not re.search(r"^--csv=", x)
        and "--periodic-junit-xmlpath" not in x
    ]
    defs_cwd = llm_src / "tests" / "integration" / "defs"
    run_env = os.environ.copy()
    run_env.update(base_env)

    any_still_failing = False
    for i, test_name in enumerate(tests):
        single_list = output_dir / f"_isolated_{i}.txt"
        single_list.write_text(test_name + "\n")
        result_xml = output_dir / f"results_isolated_{i}.xml"
        cmd = filtered + [
            f"--test-list={single_list}",
            f"--test-prefix={stage_name}",
            f"--csv={output_dir}/report_isolated_{i}.csv",
            "--periodic-junit-xmlpath",
            str(result_xml),
            "--cov-append",
        ]
        print(f"[run_integration_tests] Running isolated test [{i}]: {test_name}")
        rc = _run_cmd(cmd, cwd=str(defs_cwd), env=run_env, capture=False)
        try:
            single_list.unlink(missing_ok=True)
        except OSError:
            pass

        if rc != 0 and result_xml.exists():
            rerun_failed = _rerun_failed_tests(
                stage_name,
                llm_src,
                output_dir,
                result_xml.name,
                f"isolated_{i}",
                base_cmd,
                base_env,
                fail_signatures,
                script_dir,
            )
            if rerun_failed:
                any_still_failing = True
        elif rc != 0:
            print(
                f"[run_integration_tests] Warning: Result XML not found for isolated {i}: {result_xml}"
            )
            any_still_failing = True

    return any_still_failing


def _generate_rerun_report(
    stage_name: str,
    llm_src: Path,
    output_dir: Path,
) -> None:
    """Collect rerun XMLs, run test_rerun generate_rerun_report and merge_junit_xmls."""
    script_dir = llm_src / "jenkins" / "scripts"
    rerun_base = output_dir / "rerun"
    regular_rerun_dir = rerun_base / "regular"

    has_regular = regular_rerun_dir.exists() and any(regular_rerun_dir.glob("rerun_results_*.xml"))
    isolated_dirs = list(rerun_base.glob("isolated_*")) if rerun_base.exists() else []
    has_isolated = any(
        (d / f).exists()
        for d in isolated_dirs
        for f in ["rerun_results_1.xml", "rerun_results_2.xml"]
    )

    isolated_with_reruns = []
    for d in isolated_dirs:
        if not d.is_dir():
            continue
        num = d.name.replace("isolated_", "")
        if any(d.glob("rerun_results_*.xml")):
            isolated_with_reruns.append(
                {
                    "dir": d,
                    "num": num,
                    "original": output_dir / f"results_isolated_{num}.xml",
                }
            )

    if not has_regular and not has_isolated:
        print("[run_integration_tests] No rerun results found, skipping report generation.")
        return

    # Normalize testsuite name in all XMLs
    for x in output_dir.rglob("*.xml"):
        text = x.read_text()
        if 'testsuite name="pytest"' in text:
            x.write_text(text.replace('testsuite name="pytest"', f'testsuite name="{stage_name}"'))

    all_input_files = []
    rerun_result_files = []

    if (output_dir / "results.xml").exists():
        all_input_files.append(str(output_dir / "results.xml"))
        if has_regular:
            rerun_result_files.append(str(output_dir / "results.xml"))

    for f in sorted(output_dir.glob("results_isolated_*.xml")):
        all_input_files.append(str(f))
    for iso in isolated_with_reruns:
        if iso["original"].exists():
            rerun_result_files.append(str(iso["original"]))
        for t in [1, 2]:
            rf = iso["dir"] / f"rerun_results_{t}.xml"
            if rf.exists():
                all_input_files.append(str(rf))
                rerun_result_files.append(str(rf))

    if has_regular:
        for t in [1, 2]:
            rf = regular_rerun_dir / f"rerun_results_{t}.xml"
            if rf.exists():
                all_input_files.append(str(rf))
                rerun_result_files.append(str(rf))

    if not all_input_files:
        print("[run_integration_tests] No valid input files for rerun report.")
        return

    _run_cmd(
        [
            sys.executable,
            str(script_dir / "test_rerun.py"),
            "generate_rerun_report",
            f"--output-file={output_dir}/rerun_results.xml",
            f"--input-files={','.join(rerun_result_files)}",
        ],
        cwd=str(llm_src),
        check=True,
    )

    _run_cmd(
        [
            sys.executable,
            str(script_dir / "test_rerun.py"),
            "merge_junit_xmls",
            f"--output-file={output_dir}/results.xml",
            f"--input-files={','.join(all_input_files)}",
            "--deduplicate",
        ],
        cwd=str(llm_src),
        check=True,
    )

    for f in output_dir.glob("results_isolated_*.xml"):
        try:
            f.unlink()
        except OSError:
            pass
    print("[run_integration_tests] Rerun report generation completed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run integration tests (regular + isolated), rerun failed, generate report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--llm-src", type=Path, required=True, help="TensorRT-LLM source root")
    parser.add_argument("--stage-name", required=True, help="Stage name (e.g. L0_A10)")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory (e.g. WORKSPACE/stageName)"
    )
    parser.add_argument("--waives-file", type=Path, required=True, help="Path to waives.txt")
    parser.add_argument(
        "--regular-test-list", type=Path, default=None, help="Path to regular test list (optional)"
    )
    parser.add_argument(
        "--isolate-test-list", type=Path, default=None, help="Path to isolate test list (optional)"
    )
    parser.add_argument(
        "--fail-signatures",
        default="",
        help="Comma-separated failure signatures for rerun (from Jenkins: getFailSignaturesList)",
    )
    parser.add_argument("--perf-mode", action="store_true")
    parser.add_argument("--detailed-log", action="store_true", help="Pass -s to pytest")
    parser.add_argument(
        "--run-ray", action="store_true", help="Add --run-ray if stage name contains -Ray-"
    )
    parser.add_argument("--tester-cores", default="12")
    parser.add_argument("--container-port-start", type=int, default=0)
    parser.add_argument("--container-port-num", type=int, default=0)
    parser.add_argument("--model-cache-dir", default="", help="Default: MODEL_CACHE_DIR env")
    args = parser.parse_args()

    llm_src = args.llm_src.resolve()
    output_dir = args.output_dir.resolve()
    waives_file = args.waives_file.resolve()
    if not llm_src.is_dir():
        print(f"Error: llm-src not a directory: {llm_src}", file=sys.stderr)
        return 1
    if not waives_file.is_file():
        print(f"Error: waives-file not found: {waives_file}", file=sys.stderr)
        return 1

    model_cache = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", "")
    trtllm_whl = _get_trtllm_whl_path()
    coverage_config = output_dir / ".coveragerc"
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_config.write_text(
        "[run]\n"
        "branch = True\n"
        f"data_file = {output_dir}/.coverage.{args.stage_name}\n"
        "[paths]\n"
        f"source =\n    {llm_src}/tensorrt_llm/\n    {trtllm_whl}/tensorrt_llm/\n"
    )

    base_cmd, base_env = _build_pytest_base_command(
        llm_src,
        args.stage_name,
        output_dir,
        waives_file,
        trtllm_whl,
        coverage_config,
        perf_mode=args.perf_mode,
        detailed_log=args.detailed_log,
        run_ray=args.run_ray,
        tester_cores=args.tester_cores,
        container_port_start=args.container_port_start,
        container_port_num=args.container_port_num,
        model_cache_dir=model_cache,
    )

    no_regular = False
    no_isolate = False
    rerun_failed = False

    script_dir = llm_src / "jenkins" / "scripts"

    # 1) Regular tests + rerun
    if args.regular_test_list and args.regular_test_list.exists():
        regular_count = len(
            [line for line in args.regular_test_list.read_text().splitlines() if line.strip()]
        )
        if regular_count > 0:
            ran, reg_rerun = _run_regular_tests(
                llm_src,
                output_dir,
                args.stage_name,
                args.regular_test_list,
                base_cmd,
                base_env,
                args.fail_signatures,
                script_dir,
            )
            if ran:
                if reg_rerun:
                    rerun_failed = True
            else:
                no_regular = True
        else:
            no_regular = True
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_empty_results_xml(output_dir, args.stage_name)
    else:
        no_regular = True
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_empty_results_xml(output_dir, args.stage_name)

    # 2) Isolated tests + rerun
    if args.isolate_test_list and args.isolate_test_list.exists():
        isolate_count = len(
            [line for line in args.isolate_test_list.read_text().splitlines() if line.strip()]
        )
        if isolate_count > 0:
            iso_failed = _run_isolated_tests(
                llm_src,
                output_dir,
                args.stage_name,
                args.isolate_test_list,
                base_cmd,
                base_env,
                args.fail_signatures,
                llm_src / "jenkins" / "scripts",
            )
            if iso_failed:
                rerun_failed = True
        else:
            no_isolate = True
    else:
        no_isolate = True

    if no_regular and no_isolate:
        print(
            "Error: No tests were executed; check test lists and process_test_list output.",
            file=sys.stderr,
        )
        return 1

    # 3) Generate report
    _generate_rerun_report(args.stage_name, llm_src, output_dir)

    if rerun_failed:
        print(
            "Error: Some tests still failed after rerun attempts; check the test report.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
