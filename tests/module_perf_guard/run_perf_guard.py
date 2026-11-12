# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Check that modules and kernels are no slower than their golden latencies.

Each target folder holds the perf tests of one model family, GPU and serving config. The folders
use the model family and target names of Staircase (Modeling V2)::

    tests/module_perf_guard/
      run_perf_guard.py       # this script
      perf_guard_utils.py     # golden file and timing tools
      kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1/
        golden.json           # latency of each module and kernel per concurrency, on one GPU
        perf_tests.py         # builds each module and kernel for a concurrency

The script times the modules and kernels of each target whose golden GPU is the current GPU.
It exits with 1 when one is slower than its golden latency by more than its noise threshold.
``--update-golden`` writes the measured latencies into the golden files instead.

Usage::

    python3 tests/module_perf_guard/run_perf_guard.py
    python3 tests/module_perf_guard/run_perf_guard.py \
        --target kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1
    python3 tests/module_perf_guard/run_perf_guard.py --update-golden
"""

import argparse
import importlib.util
from collections.abc import Callable
from pathlib import Path

import torch
from perf_guard_utils import load_golden, measure_latency_us, save_golden

PERF_GUARD_DIR = Path(__file__).resolve().parent

# Builds a module or kernel for a concurrency and returns a function that runs it once.
ModuleBuilder = Callable[[int], Callable[[], object]]


def load_module_builders(target_dir: Path) -> dict[str, ModuleBuilder]:
    """Import ``perf_tests.py`` of a target folder and return its `MODULE_NAME_TO_BUILDER`."""
    spec = importlib.util.spec_from_file_location(
        f"perf_tests_{target_dir.name}", target_dir / "perf_tests.py"
    )
    perf_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_tests)
    return perf_tests.MODULE_NAME_TO_BUILDER


def run_target(target_dir: Path, update_golden: bool) -> int:
    """Time the modules and kernels of one target folder and return how many regressed."""
    golden_path = target_dir / "golden.json"
    golden = load_golden(golden_path)
    target_name = target_dir.relative_to(PERF_GUARD_DIR)
    gpu_name = torch.cuda.get_device_name()
    if golden["gpu_name"] != gpu_name:
        print(f"Skip {target_name}: its golden latencies are for {golden['gpu_name']}")
        return 0

    module_name_to_builder = load_module_builders(target_dir)
    num_regressions = 0
    for module_name, module_golden in golden["modules"].items():
        build_module = module_name_to_builder[module_name]
        concurrency_to_latency_us = module_golden["concurrency_to_latency_us"]
        for concurrency, golden_latency_us in concurrency_to_latency_us.items():
            # Use the same inputs on every run, since some kernels' speed depends on input values.
            torch.manual_seed(0)
            latency_us = measure_latency_us(build_module(int(concurrency)))

            case_name = f"{target_name} {module_name} concurrency={concurrency}"
            if update_golden:
                concurrency_to_latency_us[concurrency] = round(latency_us, 2)
                print(f"{case_name}: recorded {latency_us:.2f} us")
            elif golden_latency_us is None:
                print(f"{case_name}: {latency_us:.2f} us, no golden latency yet")
            elif latency_us > golden_latency_us * (1 + module_golden["noise_threshold"]):
                num_regressions += 1
                print(f"{case_name}: REGRESSION {latency_us:.2f} us, golden {golden_latency_us} us")
            else:
                print(f"{case_name}: pass {latency_us:.2f} us, golden {golden_latency_us} us")

    if update_golden:
        save_golden(golden_path, golden)
    return num_regressions


def main() -> int:
    parser = argparse.ArgumentParser(description="Check modules and kernels for perf regressions.")
    parser.add_argument(
        "--target",
        help="Run only this target folder, e.g. kimi_k3_vl/kimi_k3_mxfp4__sm_100__tp16_moetp16ep1",
    )
    parser.add_argument(
        "--update-golden",
        action="store_true",
        help="Write the measured latencies into the golden files.",
    )
    args = parser.parse_args()

    if args.target is not None:
        target_dirs = [PERF_GUARD_DIR / args.target]
    else:
        target_dirs = sorted(path.parent for path in PERF_GUARD_DIR.glob("*/*/golden.json"))
    num_regressions = sum(run_target(target_dir, args.update_golden) for target_dir in target_dirs)
    print(f"{num_regressions} regression(s)")
    return 1 if num_regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
