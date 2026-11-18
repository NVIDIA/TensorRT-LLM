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
"""Source checks of the k3_fused_moe kernels (cluster-scope waits on st.async mailboxes, tcgen05 thread-sync fences
and fence.proxy.async before tcgen05 reads of generic-proxy writes); the checkers and the reasons are in
``_k3_source_checks``.

  pytest test_k3_fused_moe_source_checks.py
"""

import pytest

__extra_import_path__ = [".."]
import _k3_source_checks as sc

pytestmark = pytest.mark.cpu_only

PACKAGE = f"{sc.FAMILY_KERNELS}.k3_fused_moe"

# Kernel module -> the barriers in it that other CTAs complete with st.async.
MAILBOXES = {"k3_moe_front": ("mail_full",)}
# Kernel modules whose tcgen05 fences are checked.
FENCED = ["k3_moe_front", "k3_moe_kernel"]
# Kernel module -> the barriers whose stages hold shared memory that generic-proxy writes fill and a tcgen05 operation
# reads after the wait (k3_moe: the FC1 activation scales, copied in by cp.async and read by tcgen05.cp).
GENERIC_FED = {"k3_moe_kernel": ("ab_full_fc1",)}


@pytest.mark.parametrize("module", list(MAILBOXES))
def test_mailbox_waits_acquire_at_cluster_scope(module):
    sc.check_mailbox_waits(f"{PACKAGE}.{module}", MAILBOXES[module])


@pytest.mark.parametrize("module", FENCED)
def test_kernel_fences(module):
    sc.check_tcgen05_fences(f"{PACKAGE}.{module}")


@pytest.mark.parametrize("module", list(GENERIC_FED))
def test_generic_proxy_reads_fenced(module):
    sc.check_generic_proxy_reads(f"{PACKAGE}.{module}", GENERIC_FED[module])
