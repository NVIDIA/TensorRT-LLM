# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TEMPORARY FILE - Will be removed after MoE refactor is complete.
#
# Background:
# The `enable_configurable_moe` parameter is a temporary measure during the MoE
# refactor. The old and new MoE flows will coexist for a period of time. To avoid
# large-scale changes to the existing test lists, we handle the test ID cleanup
# here. Once the refactor is complete and all tests use ConfigurableMoE by default,
# this file will no longer be needed and should be deleted.
#
# Two-phase approach:
# 1. pytest_sessionstart: Convert clean test names in CLI args back to original
#    format so pytest can find tests during collection.
# 2. pytest_collection_modifyitems: Clean up the collected test IDs for display
#    and waive matching.
import re

# Test functions that use enable_configurable_moe parameter and need ID conversion
TESTS_WITH_CONFIGURABLE_MOE = [
    "test_fused_moe_nvfp4[",
    "test_fused_moe_mxfp4_mxfp8[",
    "test_fused_moe_w4a8_nvfp4_fp8[",
    "test_fused_moe_wfp4a16[",
    "test_fused_moe_fp8_blockwise_deepgemm[",
]


def _convert_clean_to_original_moe_test_id(test_id):
    """Convert clean MoE test ID back to original format for pytest collection.

    Example: "test_fused_moe.py::test_foo[TRTLLM-dtype0]" -> "test_fused_moe.py::test_foo[-TRTLLM-dtype0]"

    This is needed because the `enable_configurable_moe` parameter uses empty string
    as ID when value is 0, resulting in test IDs like "test_foo[-TRTLLM-dtype0]".
    We clean these up in pytest_collection_modifyitems, but pytest filters tests
    during collection using the original IDs. So when user runs with clean test name,
    we need to convert it back to match the original.
    """
    if "test_fused_moe.py" not in test_id:
        return test_id

    # Match pattern like "test_name[params]" and add leading dash after "["
    # But only if params don't already start with "-" or "enable_configurable_moe"
    match = re.search(r"\[([^\]]+)\]", test_id)
    if match:
        params = match.group(1)
        # Skip if already has leading dash or starts with enable_configurable_moe
        if not params.startswith("-") and not params.startswith("enable_configurable_moe"):
            # Add leading dash to params
            new_params = "-" + params
            test_id = test_id.replace(f"[{params}]", f"[{new_params}]")

    return test_id


def pytest_sessionstart(session):
    """Convert clean MoE test IDs in config.args to original format for collection.

    This is needed because pytest filters tests during collection using original IDs.
    When user runs with clean test name, we convert it back to match the original.
    """
    args = session.config.args
    for i, arg in enumerate(args):
        if "test_fused_moe.py" in arg and "[" in arg:
            # Only apply conversion to specific tests that use enable_configurable_moe
            should_convert = any(test_name in arg for test_name in TESTS_WITH_CONFIGURABLE_MOE)
            if should_convert:
                args[i] = _convert_clean_to_original_moe_test_id(arg)


def pytest_collection_modifyitems(items):
    """Clean up test IDs by removing leading/trailing dashes from parameter IDs.

    This is needed because `enable_configurable_moe` parameter can be empty,
    resulting in ugly test IDs like "test_foo[-True]" or "test_foo[--abc]".
    We clean these up to "test_foo[True]" or "test_foo[abc]" so that:
    1. Test names in waive files and test lists remain unchanged
    2. Test reports look cleaner

    This runs BEFORE the global conftest applies waives (due to hookwrapper).
    """
    for item in items:
        if "test_fused_moe.py" in item.nodeid and "[" in item.nodeid:
            # Only apply cleanup to specific tests that use enable_configurable_moe
            should_cleanup = any(
                test_name in item.nodeid for test_name in TESTS_WITH_CONFIGURABLE_MOE
            )
            if should_cleanup:
                original_nodeid = item.nodeid
                original_name = item.name
                nodeid = item.nodeid
                name = item.name

                # Clean up leading/trailing dashes in nodeid
                nodeid = nodeid.replace("[-", "[")
                nodeid = nodeid.replace("-]", "]")

                # Clean up leading/trailing dashes in name
                name = name.replace("[-", "[")
                name = name.replace("-]", "]")

                if nodeid != original_nodeid:
                    item._nodeid = nodeid
                if name != original_name:
                    item.name = name
