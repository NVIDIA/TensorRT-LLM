# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter and DFlash worker installs of a composition, in install order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import dispatch as _dispatch


def adopt_drafter(shell, plan: "_dispatch.Plan") -> None:
    """Install ``plan``'s drafter and worker fast paths on ``shell.llm`` once the drafter has loaded.

    Each install is idempotent, logs only what it changed, raises ``ValueError``
    for a fast path of the plan it cannot install, and imports its module only
    when the plan has it.
    """
    block_conv = plan.enabled("drafter_block_conv")
    block_decode = plan.enabled("drafter_block_decode")
    if block_conv or block_decode:
        from .kernels.drafter_block_decode import block_decode as _draft

        _draft.install(shell.llm.draft_model, block_conv, block_decode)
    if plan.enabled("wide_drafter_window_attention"):
        # On the block decode drafter_block_decode just bound.
        from .kernels.wide_drafter_window_attention import attention as _wide_window

        _wide_window.install(shell.llm.draft_model)
    if plan.enabled("draft_head"):
        from .kernels.draft_head import head as _draft_head

        _draft_head.install(shell.llm)
    if plan.enabled("wide_draft_topk"):
        # On the key all-reduce draft_head just attached.
        from .kernels.wide_draft_topk import exchange as _wide_topk

        _wide_topk.install(shell.llm)
    fc_shard = plan.enabled("drafter_fc_shard")
    fused_accept = plan.enabled("fused_accept")
    fixed_gemms = plan.enabled("drafter_fixed_gemms")
    if fc_shard or fused_accept or fixed_gemms:
        from . import spec_worker as _drafter_worker

        _drafter_worker.install(
            getattr(shell.llm, "draft_model", None),
            getattr(shell.llm, "spec_worker", None),
            shell._mapping,
            fc_shard,
            fused_accept,
            fixed_gemms,
            # Passed only when selected, as the load-phase installers' keywords are.
            **({"fc_inbox": True} if plan.enabled("drafter_fc_inbox") else {}),
        )
    if plan.enabled("context_kv_fork"):
        # Derives from the class draft_head / fused_accept left the worker with;
        # verify_fold then composes on top of it.
        from .kernels.context_kv_fork import fork as _context_kv_fork

        _context_kv_fork.install(shell.llm, shell.llm.model)
    verify_fold = plan.enabled("verify_fold")
    path_walk_tail = plan.enabled("path_walk_tail")
    if verify_fold or path_walk_tail:
        from .kernels.verify_fold import fold as _spec_loop_fold

        _spec_loop_fold.install(
            getattr(shell.llm, "spec_worker", None), verify_fold, path_walk_tail
        )
    if plan.enabled("drafter_hops"):
        from .kernels.drafter_hops import hops as _drafter_hops

        _drafter_hops.install(getattr(shell.llm, "draft_model", None), shell._mapping)
        if plan.enabled("wide_drafter_hops"):
            _drafter_hops.install_wide(getattr(shell.llm, "draft_model", None), shell._mapping)
    if plan.enabled("drafter_pdl_chain"):
        # After the drafter_fixed_gemms GEMM methods and the drafter_hops hops:
        # the entry triggers rely on both.
        from .kernels.drafter_pdl_chain import chain as _drafter_pdl

        _drafter_pdl.install(shell.llm, shell._mapping)
    # After draft_head / fused_accept: the head features re-class the worker as a class
    # derived from whatever class those left it with.
    if plan.enabled("split_sampler"):
        from .kernels.split_sampler import target_head as _target_head

        _target_head.install(
            shell.llm,
            shell._mapping,
            plan.enabled("lm_head_inbox"),
            plan.enabled("vocab_parallel_head"),
        )
        if plan.enabled("wide_split_sampler"):
            _target_head.widen(shell.llm)
    # Last: the arena compaction derives its worker class from whatever class every
    # other draft and head fast path left the worker with.
    if plan.enabled("drafter_window_compact"):
        from .kernels.drafter_window_compact import compact as _window_compact

        _window_compact.install(getattr(shell.llm, "spec_worker", None))
