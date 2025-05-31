# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import glob
import os

import yaml

from .perf.test_perf import PerfTestConfig
from .test_list_parser import get_test_name_corrections, parse_test_list


# A function to convert the mako test list or test-db to a list of test names.
def parse_test_list_or_db(test_list, trt_config):

    # For mako test list.
    if test_list.endswith("txt"):
        # Mako options that may not always be defined, but are required in test lists.
        # Defined first so that values from `trt_config` can overwrite values.
        mako_opts = {
            "level": 4,
            "idx": 0,
            "priority": 0,
            "host_mem_available_mib":
            1e8,  # Set to very large to cover all tests.
            "gpu_memory": 1e6,  # Set to very large to cover all tests.
            "system_gpu_count": 128,  # Set to very large to cover all tests.
        }
        mako_opts.update(trt_config["mako_opts"])

        test_names, _ = parse_test_list(test_list,
                                        print_mako=False,
                                        no_mako=False,
                                        mako_opts=mako_opts,
                                        test_prefix=trt_config["test_prefix"])

    # For yaml-based test db.
    elif test_list.endswith("yml"):
        with open(test_list) as f:
            test_db_data = yaml.load(f, Loader=yaml.Loader)

        test_names = []
        context_name = os.path.basename(test_list).replace(".yml", "")
        for condition in test_db_data[context_name]:
            test_names.extend(condition["tests"])

    else:
        raise ValueError(f"Unexpected test list name: {test_list}")

    # Perf tests are generated based on test lists dynamically, so separate them out from normal tests.
    non_perf_test_names = [
        x for x in test_names if "perf/test_perf.py::test_perf" not in x
    ]
    perf_test_names = [
        x for x in test_names if "perf/test_perf.py::test_perf" in x
    ]

    return non_perf_test_names, perf_test_names


# Validate perf test names, which are generated dynamically based on test lists.
def validate_perf_tests(perf_test_names) -> bool:
    passed = True
    for test_name in perf_test_names:
        config = PerfTestConfig()
        try:
            # Get only the "[...]" part in the test name.
            test_param_labels = test_name.split("[")[-1][:-1]
            # Check if perf test config can be successfully loaded.
            config.load_from_str(test_param_labels)
        except Exception as e:
            print(f"Perf test name {test_name} is invalid! Error: {e}")
            passed = False

    return passed


def test_list_validation(test_root, all_pytest_items, trt_config,
                         is_trt_environment):

    # Don't run test list validation in TRT environment because TRT uses
    # YAML-based test-db for test lists.
    if is_trt_environment:
        print(
            "Skipped TRT-LLM test list validation because the pipeline is running in TRT environment."
        )
        return

    # Glob all the test list files.
    test_list_path = os.path.join(test_root, "test_lists", "*", "*.txt")
    all_test_lists = glob.glob(test_list_path)
    assert len(all_test_lists
               ) > 0, f"Cannot find any test lists with path {test_list_path}!"

    # Glob all the test db files.
    test_db_path = os.path.join(test_root, "test_lists", "*", "*.yml")
    all_test_dbs = glob.glob(test_db_path)
    assert len(all_test_dbs
               ) > 0, f"Cannot find any test lists with path {test_db_path}!"

    # Go through test lists to get test name corrections.
    passed = True
    for test_list in (all_test_lists + all_test_dbs):
        print(f"Validating test list: {test_list} ...")
        non_perf_test_names, perf_test_names = parse_test_list_or_db(
            test_list, trt_config)

        if not validate_perf_tests(perf_test_names):
            passed = False

        corrections = get_test_name_corrections(non_perf_test_names,
                                                all_pytest_items,
                                                trt_config["test_prefix"])
        if corrections:
            err_msg = "{} errors found in test list: {}".format(
                len(corrections), test_list)
            print(err_msg)
            print("Invalid tests:")
            for name, correct in corrections.items():
                if correct is not None:
                    print("\tSUGGESTED CORRECTION: {} -> {}".format(
                        name, correct))
                else:
                    print("\tCORRECTION UNKNOWN: {}".format(name))
            passed = False

    assert passed, "Some test lists contain invalid test names!"
