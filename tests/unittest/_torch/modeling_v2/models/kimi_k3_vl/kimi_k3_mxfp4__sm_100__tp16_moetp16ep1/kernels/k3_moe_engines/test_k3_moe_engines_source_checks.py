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
"""Source checks of the k3_moe_engines kernels (tcgen05 thread-sync fences); the checkers and the reasons are in
``_k3_source_checks``. The kernels are located by file path, so nothing of this target is imported (its package
loads the target's native library).

  pytest test_k3_moe_engines_source_checks.py
"""

import pytest

__extra_import_path__ = ["../../../kernels"]
import _k3_source_checks as sc

pytestmark = pytest.mark.cpu_only

KERNELS = sc.FAMILY_SOURCE / "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1" / "kernels" / "k3_moe_engines"

# Kernel modules whose tcgen05 fences are checked.
FENCED = ["k3_moe_m1_kernel", "k3_moe_m2_kernel"]


@pytest.mark.parametrize("module", FENCED)
def test_kernel_fences(module):
    sc.check_tcgen05_fences(KERNELS / f"{module}.py")
