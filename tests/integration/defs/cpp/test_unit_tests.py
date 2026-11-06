# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os as _os
import pathlib as _pl

import defs.cpp.cpp_common as _cpp
import pytest

_TEST_GROUP_DIRS = {
    "executor_bounce": "executor/bounce",
}


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("test_group", [
    "batch_manager", "common", "executor", "executor_bounce", "kernels",
    "runtime", "thop"
])
def test_unit_tests(build_google_tests, test_group, build_dir, lora_setup):

    xml_name = f"results-unit-tests-{test_group}.xml"
    test_group_dir = _TEST_GROUP_DIRS.get(test_group, test_group)

    if test_group == "executor_bounce":
        # Real-NIXL bounce tests are conditionally built when NIXL and ZMQ are
        # available. Require one here so pre-merge cannot pass with only the
        # dependency-free subset after silently losing its E2E coverage.
        required_test = (build_dir / "tests/unit_tests/executor/bounce" /
                         "bounceAgentE2ETest")
        if not required_test.is_file():
            pytest.fail(
                f"Required NIXL bounce E2E test was not built: {required_test}")

    # Discover and run the actual gtests
    ctest_command = [
        "ctest",
        "--output-on-failure",
        "--test-dir",
        f"{build_dir}/tests/unit_tests/{test_group_dir}",
        "--output-junit",
        f"{build_dir}/{xml_name}",
    ]

    parallel = _cpp.default_test_parallel
    if parallel_override := _os.environ.get("LLM_TEST_PARALLEL_OVERRIDE", None):
        parallel = int(parallel_override)

    cpp_env = {**_os.environ}

    _cpp.parallel_run_ctest(ctest_command,
                            cwd=build_dir,
                            env=cpp_env,
                            timeout=2700,
                            parallel=parallel)


@pytest.mark.parametrize("build_kv_cache_compression_tests", ["80", "100"],
                         indirect=True)
def test_kv_cache_compression_unit_tests(build_kv_cache_compression_tests,
                                         build_dir):

    xml_name = "results-unit-tests-kv_cache_compression.xml"

    # Run the binary directly: the lightweight fixture builds only this gtest,
    # so a ctest directory scan would trip over unbuilt neighbors.
    _cpp.run_command(
        [
            f"{build_dir}/tests/unit_tests/kernels/nvfp4ColdPageKernelsTest",
            f"--gtest_output=xml:{build_dir}/{xml_name}",
        ],
        cwd=build_dir,
        env={**_os.environ},
        timeout=2700,
    )


@pytest.mark.parametrize("build_fmha_head104_tests", ["100", "120"],
                         indirect=True)
def test_fmha_head104_unit_tests(build_fmha_head104_tests: None,
                                 build_dir: _pl.Path) -> None:
    """Run the dedicated FMHA target on architectures without the full gtest suite."""
    _cpp.run_command(
        [
            f"{build_dir}/tests/unit_tests/kernels/fmhaHead104Test",
            f"--gtest_output=xml:{build_dir}/results-unit-tests-fmha-head104.xml",
        ],
        cwd=build_dir,
        env={**_os.environ},
        timeout=2700,
    )
