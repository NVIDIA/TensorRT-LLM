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
"""Source checks of the synchronization in Kimi K3 kernels, shared by the per-kernel ``test_*_source_checks.py``
files. No test of values can see these orderings, so the kernels' sources are read.

Cluster-scope mailbox waits: a barrier that other CTAs complete with st.async releases at cluster scope (complete_tx),
so the waiting CTA must acquire at cluster scope (``mbarrier.{test,try}_wait.parity.acquire.cluster``, the kernels'
``_test_wait_cluster`` / ``_try_wait_cluster``); the DSL's ``mbarrier_test_wait`` / ``mbarrier_try_wait`` acquire at
CTA scope, which the PTX memory model does not let synchronize with another CTA's release.

tcgen05 thread-sync fences (PTX ISA, tcgen05 memory consistency, canonical sync patterns): a tcgen05 operation issued
after an mbarrier wait follows ``tcgen05.fence::after_thread_sync``, and a thread whose TMEM loads another thread's
tcgen05 work must not overtake (an mbarrier arrive or a CTA barrier after ``tcgen05.wait::ld``) issues
``tcgen05.fence::before_thread_sync`` first. Without them ptxas may move the TMEM access across the synchronization. A
tcgen05 operation that reads shared memory filled by generic-proxy writes (cp.async) also follows
``fence.proxy.async.shared::cta`` after the wait that orders those writes.

Not a test file (no ``test_`` prefix): the per-kernel files import it by name.
"""

import importlib.util
import re
from pathlib import Path

FAMILY_KERNELS = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.kernels"
# The family's source directory, for checks that locate a kernel by its file instead of importing its package.
FAMILY_SOURCE = (
    Path(__file__).resolve().parents[7] / "tensorrt_llm/_torch/modeling_v2/models/kimi_k3_vl"
)

# ---------------------------------------------------------------------------------------------------------------
# Cluster-scope mailbox waits
# ---------------------------------------------------------------------------------------------------------------
CTA_WAIT = re.compile(r"mbarrier_(test|try)_wait\(")
CLUSTER_WAIT = re.compile(r"_(test|try)_wait_cluster\(")


def block_header(code, i):
    """The header of the block that line i is in: the nearest earlier code line indented less."""
    indent = len(code[i]) - len(code[i].lstrip())
    for ln in reversed(code[:i]):
        if ln.strip() and len(ln) - len(ln.lstrip()) < indent:
            return ln.strip()
    return ""


def cta_scope_mailbox_waits(lines, mailboxes, own_arrival=None):
    """Line numbers of CTA-scope waits on a mailbox barrier (the barrier named on the wait's line or the next;
    a name is a regular expression), except those directly in a block whose header matches ``own_arrival``."""
    code = [ln.split("#", 1)[0] for ln in lines]
    found = []
    for i, ln in enumerate(code):
        if CTA_WAIT.search(ln):
            window = ln + (code[i + 1] if i + 1 < len(code) else "")
            if any(re.search(rf"\b{name}\b", window) for name in mailboxes):
                if own_arrival is None or not re.fullmatch(own_arrival, block_header(code, i)):
                    found.append(i + 1)
    return found


def waited_at_cluster_scope(lines, name):
    """Whether a cluster-scope wait names the barrier (on the wait's line or the next)."""
    code = [ln.split("#", 1)[0] for ln in lines]
    return any(
        CLUSTER_WAIT.search(ln)
        and re.search(rf"\b{name}\b", ln + (code[i + 1] if i + 1 < len(code) else ""))
        for i, ln in enumerate(code)
    )


# ---------------------------------------------------------------------------------------------------------------
# tcgen05 fences
# ---------------------------------------------------------------------------------------------------------------
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
PROXY_FENCE = re.compile(r"fence_proxy\(\s*(\"async_shared\"|prims\.Proxy\.ASYNC_SHARED)")


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


def generic_reads(lines, barriers):
    """(line number of the wait, whether it is fenced) for each wait on a listed barrier (named on the wait's line or
    the next) that reaches a tcgen05 operation; fenced means ``fence.proxy.async`` lies between the two."""
    code = [ln.split("#", 1)[0] for ln in lines]
    out = []
    for i, ln in enumerate(code):
        window = ln + (code[i + 1] if i + 1 < len(code) else "")
        if WAIT.search(ln) and any(re.search(rf"\b{name}\b", window) for name in barriers):
            fenced = False
            for later in code[i + 1 :]:
                if later.lstrip().startswith("def "):
                    break
                fenced = fenced or bool(PROXY_FENCE.search(later))
                if TCGEN05_OP.search(later):
                    out.append((i + 1, fenced))
                    break
    return out


# ---------------------------------------------------------------------------------------------------------------
# The checks of one kernel module
# ---------------------------------------------------------------------------------------------------------------
def source_lines(module):
    """The path and the source lines of a kernel module: a dotted module name (located with find_spec, which imports
    the module's parent packages) or the ``Path`` of its source file (read without importing anything). The checks
    below take either."""
    path = module if isinstance(module, Path) else importlib.util.find_spec(module).origin
    with open(path) as f:
        return path, f.read().split("\n")


def check_mailbox_waits(module, mailboxes, own_arrival=None):
    """No CTA-scope wait on the module's st.async mailboxes, and a cluster-scope wait on each of them."""
    path, lines = source_lines(module)
    found = cta_scope_mailbox_waits(lines, mailboxes, own_arrival)
    assert not found, f"{path}: CTA-scope waits on st.async mailboxes at lines {found}"
    # Every listed mailbox is still waited on, so a renamed barrier or helper cannot pass unchecked.
    missing = [name for name in mailboxes if not waited_at_cluster_scope(lines, name)]
    assert not missing, f"{path}: no cluster-scope wait on {missing}"


def check_tcgen05_fences(module):
    """Both tcgen05 thread-sync fences wherever the module needs them."""
    path, lines = source_lines(module)
    found = violations(lines)
    assert not found, f"{path}: {found}"


def check_generic_proxy_reads(module, barriers):
    """A wait on one of ``barriers`` reaches a tcgen05 operation, and every such wait is fenced."""
    path, lines = source_lines(module)
    reads = generic_reads(lines, barriers)
    assert reads, f"{path}: no wait on {barriers} reaches a tcgen05 operation"
    unfenced = [line for line, fenced in reads if not fenced]
    assert not unfenced, (
        f"{path}: no fence.proxy.async between the waits at lines {unfenced} and their tcgen05 reads"
    )
