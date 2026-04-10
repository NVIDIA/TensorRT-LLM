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

import pytest
from defs.conftest import get_sm_version

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)

INPUT_TEXT = """
Write a Python function `find_max(words)` to solve the following problem:\nWrite a function that accepts a list of strings.\nThe list contains different words. Return the word with maximum number\nof unique characters. If multiple strings have maximum number of unique\ncharacters, return the one which comes first in lexicographical order.\nfind_max(["name", "of", "string"]) == "string"\nfind_max(["name", "enam", "game"]) == "enam"\nfind_max(["aaaaaaa", "bb" ,"cc"]) == ""aaaaaaa"
"""
