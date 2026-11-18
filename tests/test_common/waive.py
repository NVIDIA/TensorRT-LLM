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

"""Name matching shared by unit-test waives and CI report processing."""


def waive_candidates(nodeid: str) -> list[str]:
    """Return exact, unparametrized, class and file names, in matching order.

    Directory names are never inferred from a test's file path.
    """
    names = [nodeid]
    if nodeid.endswith("]"):
        nodeid = nodeid.split("[", 1)[0]
        names.append(nodeid)
    while "::" in nodeid:
        nodeid = nodeid.rsplit("::", 1)[0]
        names.append(nodeid)
    return list(dict.fromkeys(names))
