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
    when the plan has it. ``rejection_sampler`` installs only when the model's
    speculative config asks for rejection sampling, and an install it refuses
    leaves the built-in rejection path (one warning, on every rank alike).

    First, under forced acceptance on a deployment the policy applies to (sm107
    TP8 with ``rejection_sampler`` not turned off), the model's speculative config
    is checked to have rejection sampling off (``rs_common.apply_forced_policy``;
    the route switched the engine's config off before the model was built),
    before every install reads it. Elsewhere it is left as it is.
    """
    from .kernels.rs_sampler import rs_common as _rs_common

    table = getattr(plan, "levers", None)
    _rs_common.apply_forced_policy(shell, table.deployment if table is not None else None)
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
    if plan.enabled("draft_head_sampled"):
        # On the worker class draft_head just set, before fused_accept derives from it.
        from .kernels.draft_head import head as _draft_head

        _draft_head.install_sampled(shell.llm)
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
            **(
                {
                    "fc_inbox_rows": plan.param("drafter_fc_inbox_wide", "MAX_ROWS"),
                    "fc_inbox_programs": plan.param("drafter_fc_inbox_wide", "PROGRAMS"),
                }
                if plan.enabled("drafter_fc_inbox_wide")
                else {}
            ),
        )
    if plan.enabled("context_kv_fork"):
        # Derives from the class draft_head / fused_accept left the worker with;
        # verify_fold then composes on top of it.
        from .kernels.context_kv_fork import fork as _context_kv_fork

        parallel = plan.enabled("windowed_commit_parallel")
        _context_kv_fork.install(
            shell.llm,
            shell.llm.model,
            side_chain=plan.enabled("drafter_kv_side_chain"),
            early_fc=plan.enabled("drafter_fc_early_fork"),
            glue=plan.enabled("drafter_kv_glue"),
            fork_before_head=plan.enabled("drafter_kv_fork_before_head"),
            parallel_min_requests=(
                plan.param("windowed_commit_parallel", "MIN_REQUESTS") if parallel else None
            ),
            late_join=plan.enabled("context_kv_late_join"),
            declined=plan.declined,
            paged_commit=plan.enabled("paged_context_commit"),
            paged_tables=plan.enabled("paged_ctx_block_tables"),
            paged_store=plan.enabled("paged_ctx_kv_store"),
        )
    verify_fold = plan.enabled("verify_fold")
    path_walk_tail = plan.enabled("path_walk_tail")
    distributed_head_sampler = plan.enabled("distributed_head_sampler")
    if verify_fold or path_walk_tail:
        from .kernels.verify_fold import fold as _spec_loop_fold

        # distributed_head_sampler: the fold loads before its wait behind the sampler's early
        # release. drafter_preamble_late_join: the preamble carries the side chain's join into
        # the drafter forward.
        _spec_loop_fold.install(
            getattr(shell.llm, "spec_worker", None),
            verify_fold,
            path_walk_tail,
            preamble=plan.enabled("drafter_preamble_from_commit"),
            declined=plan.declined,
            verify_pdl=distributed_head_sampler,
            late_join=plan.enabled("drafter_preamble_late_join"),
            rejection=plan.enabled("rejection_fold"),
        )
    if plan.enabled("drafter_hops"):
        from .kernels.drafter_hops import hops as _drafter_hops

        _drafter_hops.install(getattr(shell.llm, "draft_model", None), shell._mapping)
        if plan.enabled("wide_drafter_hops"):
            _drafter_hops.install_wide(getattr(shell.llm, "draft_model", None), shell._mapping)
    drafter_folds = plan.enabled("drafter_folds")
    if drafter_folds:
        # drafter_folds: the layer-0 input norm releasing its dependent at entry.
        from .kernels.ox_entry import entry_norm as _ox_entry_norm

        _ox_entry_norm.install(getattr(shell.llm, "draft_model", None))
    if plan.enabled("drafter_pdl_chain"):
        # After the drafter_fixed_gemms GEMM methods and the drafter_hops hops:
        # the entry triggers rely on both.
        from .kernels.drafter_pdl_chain import chain as _drafter_pdl

        _drafter_pdl.install(shell.llm, shell._mapping)
    if drafter_folds:
        # drafter_folds: the QK norm + RoPE on a PDL copy, the window attention's split loading
        # before its wait.
        from .kernels.drafter_attn_pdl import attn_pdl as _drafter_attn_pdl

        _drafter_attn_pdl.install(getattr(shell.llm, "draft_model", None))
        # drafter_folds: elementwise steps folded into the drafter's GEMVs. After the block
        # installs, the fixed GEMMs (the tactics) and the PDL chain (the MLP class it consults).
        from .kernels.dr3_folds import folds as _dr3_folds

        _dr3_folds.install(getattr(shell.llm, "draft_model", None))
    if plan.enabled("draft_head_tail"):
        # draft_head_tail: the draft head's tail after the shard GEMM as CUDA launches. After
        # draft_head (its worker class and key exchange). Collective.
        from .kernels.dr_draft_tail import tail as _dr_draft_tail

        _dr_draft_tail.install(shell.llm, shell._mapping)
    elif plan.enabled("rejection_sampler"):
        # rejection_sampler without draft_head_tail (a drafter that declares a logit transform, or
        # the tail's knob off): the tail for the rejection-sampled steps only. Collective.
        from .kernels.dr_draft_tail import tail as _dr_draft_tail

        _dr_draft_tail.install(shell.llm, shell._mapping, rs_only=True)
    # After draft_head / fused_accept: the head features re-class the worker as a class
    # derived from whatever class those left it with.
    if plan.enabled("split_sampler"):
        from .kernels.split_sampler import target_head as _target_head

        multimem = plan.enabled("head_multimem_gather")
        coop = plan.enabled("head_sampler_coop_fold")
        head = _target_head.install(
            shell.llm,
            shell._mapping,
            plan.enabled("lm_head_inbox"),
            plan.enabled("vocab_parallel_head"),
            multimem_gather=multimem,
            coop_max_rows=plan.param("head_sampler_coop_fold", "MAX_ROWS") if coop else 0,
            distributed=distributed_head_sampler,
        )
        if multimem:
            # The gather's refusal is agreed over the TP group, so every rank reports alike.
            plan.installed(
                "head_multimem_gather", head.multimem is not None, str(head.multimem_refusal)
            )
        if distributed_head_sampler:
            # distributed_head_sampler: the interleaved LM-head copy and the sampler-state
            # exchange. After the head (it attaches to it). Collective.
            from .kernels.s2_sampler import sampler as _s2_sampler

            # rejection_sampler: whether the speculative config asks for rejection sampling (as
            # the forced-acceptance policy above left it) and the forced mode, agreed by the s2
            # install's refusal before it sizes the fabric (with the RS areas when it does).
            rs_facts = None
            if plan.enabled("rejection_sampler"):
                from .kernels.rs_sampler import rs as _rs_sampler

                rs_facts = _rs_sampler.agreement_facts(shell.llm)
            _s2_sampler.install(shell.llm, shell._mapping, rs_facts=rs_facts)
            if rs_facts is not None and rs_facts[0]:
                # rejection_sampler: fused rejection sampling on the s2 head and the draft
                # head tail's walk. After s2 (whose fabric holds its areas). Collective.
                _rs_sampler.install(shell.llm, shell._mapping)
        if plan.enabled("wide_split_sampler"):
            _target_head.widen(shell.llm)
    if drafter_folds:
        # drafter_folds: the drafter's layer-0 front as a per-token table. After the head paths:
        # the table is built by running the layer-0 modules as every install above left them,
        # and the verify fold's worker resolves the mask id the noise ids carry.
        from .kernels.tt_table import drafter_l0 as _tt_drafter_l0

        _tt_drafter_l0.install(shell.llm)
    # Last: the arena compaction derives its worker class from whatever class every
    # other draft and head fast path left the worker with.
    if plan.enabled("step_window_compact"):
        from .kernels.drafter_window_compact import compact as _window_compact

        _window_compact.install_steps(getattr(shell.llm, "spec_worker", None))
