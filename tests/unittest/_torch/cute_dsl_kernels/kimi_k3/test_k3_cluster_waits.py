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
"""Source check of the mailbox waits in Kimi K3 kernels whose barriers other CTAs complete with st.async: the
complete_tx releases at cluster scope, so the waiting CTA must acquire at cluster scope
(``mbarrier.{test,try}_wait.parity.acquire.cluster``, the kernels' ``_test_wait_cluster`` / ``_try_wait_cluster``);
the DSL's ``mbarrier_test_wait`` / ``mbarrier_try_wait`` acquire at CTA scope, which the PTX memory model does not let
synchronize with another CTA's release.

  pytest test_k3_cluster_waits.py
"""

import importlib.util
import re

import pytest

# Kernel module -> the barriers in it that other CTAs complete with st.async.
MAILBOXES = {
    "tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn.k3_kda_attn_kernel": ("mbox_bar", "ss_ready"),
    "tensorrt_llm._torch.cute_dsl_kernels.k3_kda_attn.k3_kda_decode_kernel": ("ss_ready",),
    "tensorrt_llm._torch.cute_dsl_kernels.k3_kda_verify.k3_kda_verify_kernel": ("ss_ready",),
    "tensorrt_llm._torch.cute_dsl_kernels.k3_drafter.k3_drafter_attn_kernel": ("mail_full",),
    "tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.k3_ctm_gemv_kernel": ("mail_full",),
    "tensorrt_llm._torch.cute_dsl_kernels.k3_fused_moe.k3_moe_front": ("mail_full",),
    "tensorrt_llm._torch.cute_dsl_kernels.k3_mla.k3_mla_attn_kernel": ("ml_full",),
    "tensorrt_llm._torch.cute_dsl_kernels.k3_mla.k3_mla_q_kernel": (
        "qn_full",
        "norm_full",
        "kb_full",
        "mail_full",
    ),
    # rms_full: stage 0 only (filled by the cluster CTAs' st.async pushes); its other stages are completed by this
    # CTA's own arrive or bulk copy.
    "tensorrt_llm._torch.cute_dsl_kernels.k3_sandwich.k3_sandwich_kernel": (
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
# Kernel module -> the header of the blocks whose mailbox waits are on this CTA's own arrival and keep CTA scope
# (k3_mla_attn's no_cluster mode arrives on ml_full itself and acquires the other CTAs' (m, l) through a counter).
OWN_ARRIVAL = {
    "tensorrt_llm._torch.cute_dsl_kernels.k3_mla.k3_mla_attn_kernel": r"if cutlass\.const_expr\(no_cluster\):",
}
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


def test_checker_catches_the_pattern():
    bad = ["    while not cute.arch.mbarrier_test_wait(mail_full.data_ptr(), 0):", "        pass"]
    good = ["    while not _test_wait_cluster(mail_full.data_ptr(), 0):", "        pass"]
    assert cta_scope_mailbox_waits(bad, ("mail_full",)) == [1]
    assert cta_scope_mailbox_waits(good, ("mail_full",)) == []
    assert waited_at_cluster_scope(good, "mail_full")
    assert not waited_at_cluster_scope(bad, "mail_full")
    # Only the own-arrival block keeps CTA scope; the same wait before it or in its else branch is flagged.
    own = r"if cutlass\.const_expr\(no_cluster\):"
    branches = (
        bad + ["if cutlass.const_expr(no_cluster):", "    # own arrival"] + bad + ["else:"] + bad
    )
    assert cta_scope_mailbox_waits(branches, ("mail_full",), own) == [1, 8]


@pytest.mark.parametrize("module", list(MAILBOXES), ids=[m.rsplit(".", 1)[1] for m in MAILBOXES])
def test_mailbox_waits_acquire_at_cluster_scope(module):
    path = importlib.util.find_spec(module).origin
    with open(path) as f:
        lines = f.read().split("\n")
    found = cta_scope_mailbox_waits(lines, MAILBOXES[module], OWN_ARRIVAL.get(module))
    assert not found, f"{path}: CTA-scope waits on st.async mailboxes at lines {found}"
    # Every listed mailbox is still waited on, so a renamed barrier or helper cannot pass unchecked.
    missing = [name for name in MAILBOXES[module] if not waited_at_cluster_scope(lines, name)]
    assert not missing, f"{path}: no cluster-scope wait on {missing}"
