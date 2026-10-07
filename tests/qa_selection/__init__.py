# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Select the tests a target machine can run, before Slurm allocates it.

    plugin.py        the pytest hooks and options, loaded with `-p qa_selection.plugin`
    core/
      selection.py   one run's request, and its outcome per test
      report.py      the .ids lists and the JSON record
      ladder.py      --ladder: which rung holds a test
      machine.py     --machine: may the machine's card run a test
      marks.py       what a test asks for, read from its marks

No executable statements: pytest imports this package before any conftest, to
load the plugin.
"""
