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
"""Source check of the tcgen05 thread-sync fences in Kimi K3 kernels (PTX ISA, tcgen05 memory consistency, canonical
sync patterns): a tcgen05 operation issued after an mbarrier wait follows ``tcgen05.fence::after_thread_sync``, and a
thread whose TMEM loads another thread's tcgen05 work must not overtake (an mbarrier arrive or a CTA barrier after
``tcgen05.wait::ld``) issues ``tcgen05.fence::before_thread_sync`` first. Without them ptxas may move the TMEM access
across the synchronization; no test of values can see it, so the kernels' sources are read.

  pytest test_k3_tcgen05_fences.py
"""

import importlib.util
import re

import pytest

KERNELS = [
    "tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.k3_ctm_gemv_kernel",
    "tensorrt_llm._torch.cute_dsl_kernels.k3_decode_gemv.k3_decode_gemv_kernel",
]

# Thread syncs that order other threads' work before a tcgen05 operation of this thread. Waits on a barrier only TMA
# completes (the weight / activation rings) are not: TMA writes and tcgen05 reads are both async proxy, ordered by the
# barrier's complete_tx, the CUTLASS mainloop pattern.
WAIT = re.compile(
    r"mbarrier_try_wait\(|mbarrier_test_wait\(|_(try|test)_wait_cluster\(|(?<![\w.])_wait\(|barrier_cta_sync\(|"
    r"barrier_cluster_wait\(|sync_warp\(|cute\.arch\.barrier\("
)
TMA_FED = re.compile(r"\b(full|ab_full|fc2_full|w_full|x_full)\.(subview|data_ptr)\(")
TCGEN05_OP = re.compile(r"tcgen05_(ld|cp|mma\w*)\(")
AFTER = "Tcgen05Fence.AFTER_THREAD_SYNC"
BEFORE = "Tcgen05Fence.BEFORE_THREAD_SYNC"
LOAD_WAIT = "Tcgen05Wait.LOAD"
SYNC = re.compile(r"mbarrier_arrive\(|barrier_cta_sync\(|cute\.arch\.barrier\(")


def violations(lines):
    """(line number, rule) of every tcgen05 operation reached from a wait without the after-fence, and every arrive /
    barrier reached from a TMEM load wait without the before-fence (scanning back within the function)."""
    code = [ln.split("#", 1)[0] for ln in lines]
    out = []
    for i, ln in enumerate(code):
        if TCGEN05_OP.search(ln) and not ln.lstrip().startswith("def "):
            for j in range(i - 1, -1, -1):
                if AFTER in code[j] or code[j].lstrip().startswith("def "):
                    break
                if WAIT.search(code[j]):
                    if not TMA_FED.search(
                        code[j] + code[j + 1]
                    ):  # the barrier name may wrap to the next line
                        out.append(
                            (i + 1, "tcgen05 op after a wait without fence::after_thread_sync")
                        )
                    break
        if SYNC.search(ln):
            for j in range(i - 1, -1, -1):
                if BEFORE in code[j] or code[j].lstrip().startswith("def ") or SYNC.search(code[j]):
                    break
                if LOAD_WAIT in code[j]:
                    out.append(
                        (
                            i + 1,
                            "arrive / barrier after tcgen05.wait::ld without fence::before_thread_sync",
                        )
                    )
                    break
    return out


def test_checker_catches_the_patterns():
    """The checker flags both missing fences (and accepts the fenced forms)."""
    bad = [
        "def f():",
        "    while not cute.arch.mbarrier_try_wait(acc_full.data_ptr(), 0):",
        "        pass",
        "    acc = prims.tcgen05_ld('32x32b', ptr, num=8)",
        "    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)",
        "    prims.mbarrier_arrive(acc_empty)",
    ]
    assert [rule for _, rule in violations(bad)] == [
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
    assert violations(good) == []


@pytest.mark.parametrize("module", KERNELS, ids=[m.rsplit(".", 1)[1] for m in KERNELS])
def test_kernel_fences(module):
    path = importlib.util.find_spec(module).origin
    with open(path) as f:
        found = violations(f.read().split("\n"))
    assert not found, f"{path}: {found}"
