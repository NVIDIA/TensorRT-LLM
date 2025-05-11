#!/usr/bin/env python3
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

#
# This script collects tensorrt_llm unit tests and transform them into TensorRT TURTLE form.
#
# Usage:
# 1. build and install tensorrt_llm python package
# 2. install pytest `pip3 install pytest`
# 3. run `python3 scripts/collect_unittests.py` in tensorrt_llm root directory.
# 4. update the collected tests into TensorRt TURTLE test.
#    - check python list `LLM_UNIT_TESTS` in `<tensorrt repo>/tests/trt-test-defs/turtle/defs/llm/test_llm_unittests.py`.
from subprocess import check_output

KEYWORDS = ["<Module", "<UnitTestCase", "<TestCaseFunction"]


def fetch_tests():
    text = check_output(["pytest", "--collect-only", "tests/"])
    text = text.decode()
    lines = text.split("\n")
    lines = [line for line in lines if any([k in line for k in KEYWORDS])]

    module, unittest, case = "<bad>", "<bad>", "<bad>"
    for line in lines:
        if "<Module" in line:
            module = line.replace("<Module ", "").replace(">", "").strip()
        elif "<UnitTestCase" in line:
            unittest = line.replace("<UnitTestCase ", "").replace(">",
                                                                  "").strip()
        elif "<TestCaseFunction" in line:
            case = line.replace("<TestCaseFunction ", "").replace(">",
                                                                  "").strip()
            print(f"LLMUnitTestCase(\"{module}\", \"{unittest}.{case}\"),")


if __name__ == "__main__":
    fetch_tests()
