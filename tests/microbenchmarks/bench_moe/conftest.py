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
"""Register the module-perf markers for this subtree.

``attention_perf/conftest.py`` already does this for its own directory, but
there was no conftest here, so ``-m discrete`` against ``bench_moe`` ran under
``PytestUnknownMarkWarning`` -- harmless today, a hard error the moment anything
turns on ``--strict-markers``.

Note that ``test_discrete_metrics.py`` carries no ``@pytest.mark.discrete``, so
``-m discrete`` does not select it. That is not an oversight this file fixes:
those cases are driven by node id from ``qa/module_test_list.txt``, where the
discrete/continuous tag lives in the list rather than on the test. Marking them
would change which pipeline picks them up, which is not this change's business.
"""


def pytest_configure(config):
    config.addinivalue_line("markers", "discrete: zero-threshold structural assert, pre-merge gate")
    config.addinivalue_line("markers", "continuous: gpu_time vs baseline, post-merge detector")
