# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every MiniMax-M3 modeling_v2 target declares a copy plan that validates, merges what its decode step needs merged,
and needs no barrier: no device reader precedes the replay. Import-level: no target is constructed."""

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.spec.copy_sites import (
    M3SpecCopySite,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import MsaCopySite
from tensorrt_llm._torch.pyexecutor.device_copy import CopyPlan, CopySite, InGraph, copy_plan_of

_TARGET = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__{}.modeling"
)
_SM103 = ("sm_103__tp4", "sm_103__tp8")
_SM107 = ("sm_107__tp2", "sm_107__tp4", "sm_107__tp8")

# The sites every target merges: graph inputs, decoder positions, attention lengths and the host-staged MSA tables.
_SHARED = {
    CopySite.GRAPH_INPUT_IDS,
    CopySite.GRAPH_POSITION_IDS,
    CopySite.INPUT_POSITION_IDS,
    CopySite.ATTN_KV_LENS,
    CopySite.ATTN_PROMPT_LENS,
    MsaCopySite.SHORT_TABLE,
    MsaCopySite.N_VALID_BLOCKS,
    MsaCopySite.OUT_CACHE_LOC,
    MsaCopySite.KV_INDICES,
    MsaCopySite.BLOCK_TABLE,
    MsaCopySite.SEQ_LENS,
    MsaCopySite.QO_LENS,
    MsaCopySite.CU_Q_LENS,
    MsaCopySite.CU_KV_LENS,
    MsaCopySite.Q_BATCH_ROW,
    MsaCopySite.Q_INTRA,
    MsaCopySite.KV_LENS_STAGED,
}
_SPEC = {
    CopySite.SPEC_RNG_REQUEST_SEEDS,
    CopySite.SPEC_RNG_REQUEST_OFFSETS,
    CopySite.SPEC_RNG_SEEDS,
    CopySite.SPEC_RNG_OFFSETS,
    M3SpecCopySite.FOLD_SLOTS,
    M3SpecCopySite.FOLD_CTRL,
}


def _module(target: str):
    return importlib.import_module(_TARGET.format(target))


def _plan(target: str) -> CopyPlan:
    module = _module(target)
    classes = [
        cls
        for name, cls in vars(module).items()
        if name.startswith("ModelingV2MiniMaxM3Nvfp4") and isinstance(cls, type)
    ]
    assert len(classes) == 1, classes
    plan = copy_plan_of(classes[0])
    assert plan is module._COPY_PLAN
    return plan


@pytest.mark.parametrize("target", _SM103 + _SM107)
def test_every_target_merges_the_shared_sites_without_a_barrier(target):
    """No device work reads a merged destination before the replay, so no target needs a flush_after barrier."""
    plan = _plan(target)
    assert _SHARED <= plan.merge
    assert not plan.flush_after
    # The plan is a valid CopyPlan: rebuilding it from its fields runs the same validation.
    assert (
        CopyPlan(
            merge=plan.merge,
            flush_after=plan.flush_after,
            changed_span=plan.changed_span,
            parallel_scatter=plan.parallel_scatter,
            final_flush=plan.final_flush,
        )
        == plan
    )


@pytest.mark.parametrize("target", _SM103)
def test_sm103_targets_flush_eagerly(target):
    plan = _plan(target)
    assert plan.merge == _SHARED
    assert plan.final_flush is None and not plan.parallel_scatter and not plan.changed_span


@pytest.mark.parametrize(
    "target, step_epoch, changed_span",
    [("sm_107__tp2", False, False), ("sm_107__tp4", True, True), ("sm_107__tp8", True, True)],
)
def test_sm107_targets_flush_in_graph(target, step_epoch, changed_span):
    module = _module(target)
    plan = _plan(target)
    assert plan.merge == _SHARED | _SPEC
    assert plan.parallel_scatter
    assert isinstance(plan.final_flush, InGraph)
    assert plan.final_flush.early_copy
    assert plan.final_flush.prologue_op is module.m3_step_prologue
    assert plan.final_flush.step_epoch is step_epoch
    assert plan.changed_span == ({MsaCopySite.BLOCK_TABLE} if changed_span else set())


@pytest.mark.parametrize("target", _SM107)
def test_sm107_runtime_hooks_keep_only_step_loop_choices(target):
    hooks = _module(target)._runtime_hooks(None)
    assert hooks.spec_offsets_op is not None
    assert hooks.loop_on_execution_stream and hooks.merge_tuned_tactics_across_ranks
    assert hooks.first_content_emit_iters == 2
    assert (
        hooks.mxfp8_pinned_tactics
        and hooks.late_compile_detect
        and hooks.iteration_timing_one_event
    )
