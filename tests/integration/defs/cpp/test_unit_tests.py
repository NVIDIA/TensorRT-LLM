import os as _os

import defs.cpp.cpp_common as _cpp
import pytest


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("test_group", [
    "batch_manager", "common", "executor", "kernels", "layers", "runtime",
    "thop", "utils"
])
def test_unit_tests(build_google_tests, test_group, build_dir, lora_setup):

    xml_name = f"results-unit-tests-{test_group}.xml"

    # Discover and run the actual gtests
    ctest_command = [
        "ctest",
        "--output-on-failure",
        "--test-dir",
        f"{build_dir}/tests/unit_tests/{test_group}",
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
