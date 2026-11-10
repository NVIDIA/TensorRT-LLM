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
"""Source checks of the k3_sandwich kernels (cluster-scope waits on st.async mailboxes and tcgen05 thread-sync
fences); the checkers and the reasons are in ``_k3_source_checks``.

  pytest test_k3_sandwich_source_checks.py
"""

import pytest

__extra_import_path__ = [".."]
import _k3_source_checks as sc

pytestmark = pytest.mark.cpu_only

PACKAGE = f"{sc.FAMILY_KERNELS}.k3_sandwich"

# Kernel module -> the barriers in it that other CTAs complete with st.async.
MAILBOXES = {
    # rms_full: stage 0 only (filled by the cluster CTAs' st.async pushes); its other stages are completed by this
    # CTA's own arrive or bulk copy.
    "k3_sandwich_kernel": (
        "b_ready",
        "lat_full",
        r"rms_full\.subview\(0",
        "mb_rms",
        "mb_snap",
        "mb_ts",
        "mb_upd",
        "mb_sq",
    ),
}
# Kernel modules whose tcgen05 fences are checked.
FENCED = ["k3_sandwich_kernel"]


@pytest.mark.parametrize("module", list(MAILBOXES))
def test_mailbox_waits_acquire_at_cluster_scope(module):
    sc.check_mailbox_waits(f"{PACKAGE}.{module}", MAILBOXES[module])


@pytest.mark.parametrize("module", FENCED)
def test_kernel_fences(module):
    sc.check_tcgen05_fences(f"{PACKAGE}.{module}")
