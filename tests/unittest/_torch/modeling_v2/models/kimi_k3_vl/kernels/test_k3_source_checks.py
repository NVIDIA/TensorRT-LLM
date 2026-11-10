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
"""The source checkers of ``_k3_source_checks`` flag the patterns they look for and accept the fixed forms. The
per-kernel ``test_*_source_checks.py`` files run them on the kernels; this file runs the mailbox check on the stock
Kimi K3 router-logits kernel, which has no kernel directory here.

  pytest test_k3_source_checks.py
"""

import _k3_source_checks as sc
import pytest

pytestmark = pytest.mark.cpu_only


def test_mailbox_checker_catches_the_pattern():
    bad = ["    while not cute.arch.mbarrier_test_wait(mail_full.data_ptr(), 0):", "        pass"]
    good = ["    while not _test_wait_cluster(mail_full.data_ptr(), 0):", "        pass"]
    assert sc.cta_scope_mailbox_waits(bad, ("mail_full",)) == [1]
    assert sc.cta_scope_mailbox_waits(good, ("mail_full",)) == []
    assert sc.waited_at_cluster_scope(good, "mail_full")
    assert not sc.waited_at_cluster_scope(bad, "mail_full")
    # Only the own-arrival block keeps CTA scope; the same wait before it or in its else branch is flagged.
    own = r"if cutlass\.const_expr\(no_cluster\):"
    branches = (
        bad + ["if cutlass.const_expr(no_cluster):", "    # own arrival"] + bad + ["else:"] + bad
    )
    assert sc.cta_scope_mailbox_waits(branches, ("mail_full",), own) == [1, 8]


def test_fence_checker_catches_the_patterns():
    """The checker flags both missing fences (and accepts the fenced forms)."""
    bad = [
        "def f():",
        "    while not cute.arch.mbarrier_try_wait(acc_full.data_ptr(), 0):",
        "        pass",
        "    acc = prims.tcgen05_ld('32x32b', ptr, num=8)",
        "    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)",
        "    prims.mbarrier_arrive(acc_empty)",
    ]
    assert [rule for _, rule in sc.violations(bad)] == [
        "tcgen05 op after a wait without fence::after_thread_sync",
        "arrive / barrier after tcgen05.wait::ld without fence::before_thread_sync",
    ]
    good = (
        bad[:3]
        + ["    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)"]
        + bad[3:5]
        + ["    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)"]
        + bad[5:]
    )
    assert sc.violations(good) == []


def test_checker_catches_an_unfenced_generic_read():
    """The proxy check flags a tcgen05 read after the wait without fence.proxy.async (and accepts the fenced form)."""
    wait = [
        "def f():",
        "    while not cute.arch.mbarrier_try_wait(",
        "        ab_full_fc1.subview(stage).data_ptr(), phase",
        "    ):",
        "        pass",
    ]
    copy = ["    prims.tcgen05_cp(shape, tmem_ptr, desc)"]
    fence = ['    prims.fence_proxy("async_shared", space=prims.SharedSpace.shared_cta)']
    assert sc.generic_reads(wait + copy, ("ab_full_fc1",)) == [(2, False)]
    assert sc.generic_reads(wait + fence + copy, ("ab_full_fc1",)) == [(2, True)]


def test_router_logits_mailbox_waits_acquire_at_cluster_scope():
    """The router-logits kernel's cross-CTA running-sum mailbox is waited on at cluster scope."""
    sc.check_mailbox_waits(
        "tensorrt_llm._torch.moe.kernels.blackwell.kimi_k3_router_logits", ("mbox_full",)
    )
