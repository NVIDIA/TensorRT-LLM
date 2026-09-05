# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding state that is keyed by live-request identity must be
sized by the sequence-slot pool, not by max_batch_size.

Under the attention-DP overlap headroom the two differ by 2x
(``compute_max_num_sequences``): a finished request holds its slot for one more
iteration while its replacement is already admitted (nvbug-6627795). Two
distinct families follow from that, and only the first needs the pool size:

* keyed by ``py_seq_slot`` / a per-request ``SlotManager`` slot -- must span the
  pool. ``SpecMetadata.num_seq_slots`` (draft_probs, full_draft_probs,
  penalty_state), ``MTPHiddenStatesManager``'s hidden-state pools,
  ``DynamicTreeSlotStorage``, ``Eagle3ResourceManager.slot_manager`` and the
  ``SuffixAutomatonManager`` slot pool.
* keyed by *batch position* -- ``max_num_requests`` is correct and deliberately
  unchanged, because the micro-batch scheduler caps every forward at
  max_batch_size (its ``no_schedule_after_state=GENERATION_TO_COMPLETE`` default
  keeps the retiring requests out of the batch entirely). ``SpecTreeManager``'s
  per-forward work buffers and ``batch_indices_cuda`` are in this family.
"""

import ast
import inspect
import textwrap
import types

import pytest
import torch

from tensorrt_llm._torch.speculative.eagle3 import (
    Eagle3OneModelDynamicTreeResourceManager,
    Eagle3ResourceManager,
)
from tensorrt_llm._torch.speculative.mtp import MTPHiddenStatesManager
from tensorrt_llm._torch.speculative.mtp_dynamic_tree import MTPEagleDynamicTreeResourceManager
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
from tensorrt_llm._torch.speculative.suffix_automaton import SAConfig, SuffixAutomatonManager
from tensorrt_llm._torch.speculative.utils import (
    _build_spec_metadata,
    get_spec_metadata,
    get_spec_resource_manager,
)

R, POOL = 8, 16  # max_batch_size, 2 * max_batch_size (overlap headroom)


@pytest.mark.cpu_only
def test_slot_pool_size_is_applied_centrally(monkeypatch):
    """``get_spec_metadata`` stamps the pool size onto whatever mode was built.

    This is the property that fixes the review finding: previously only the
    MTP-eagle branch forwarded ``num_seq_slots``, so vanilla MTP, Eagle3
    one-model, PARD, DFlash/DSpark and draft-target one-model all sized their
    slot-indexed buffers at ``max_num_requests``. Applying it once at the single
    exit point makes it impossible for a mode -- including a future one -- to be
    missed, so the assertion deliberately does not name any mode.
    """
    built = types.SimpleNamespace()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.utils._build_spec_metadata", lambda *a, **k: built
    )
    spec_config = types.SimpleNamespace(enable_penalty=False)

    out = get_spec_metadata(
        spec_config,
        model_config=object(),
        max_num_requests=R,
        max_num_tokens=128,
        num_seq_slots=POOL,
    )

    assert out is built
    assert out.num_seq_slots == POOL


@pytest.mark.cpu_only
def test_unknown_slot_pool_leaves_the_max_num_requests_fallback(monkeypatch):
    """``num_seq_slots=None`` must not be written as a literal.

    Both allocators resolve the pool as ``self.num_seq_slots or
    self.max_num_requests``, so leaving the dataclass default (0) in place is how
    a caller that does not know the pool size keeps the old sizing. Writing
    ``None`` would raise in the ``+ 1`` scratch-row arithmetic instead.
    """
    built = types.SimpleNamespace()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.utils._build_spec_metadata", lambda *a, **k: built
    )
    spec_config = types.SimpleNamespace(enable_penalty=False)

    get_spec_metadata(
        spec_config,
        model_config=object(),
        max_num_requests=R,
        max_num_tokens=128,
        num_seq_slots=None,
    )

    assert not hasattr(built, "num_seq_slots")


@pytest.mark.cpu_only
def test_per_mode_builder_does_not_take_the_pool_size():
    """Guard the central-application invariant structurally.

    Re-plumbing ``num_seq_slots`` through the per-mode constructors is what let a
    branch be forgotten in the first place; keep the builder free of it.
    """
    assert "num_seq_slots" not in inspect.signature(_build_spec_metadata).parameters


def _mtp_config():
    return types.SimpleNamespace(max_draft_len=2, use_relaxed_acceptance_for_thinking=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MTP hidden-state pools are CUDA tensors")
@pytest.mark.parametrize(
    "num_seq_slots,expected_pool",
    [
        (POOL, POOL + 1),
        (None, R + 1),
    ],
)
def test_mtp_hidden_states_pool_spans_the_slot_pool(num_seq_slots, expected_pool):
    """The pool must cover every *resident* request, plus the CUDA-graph dummy.

    ``add_slot`` runs on a request's first context chunk and the slot is only
    returned by ``free_resources``, which the overlap scheduler defers -- so at
    ``max_num_requests + 1`` the replacement request raises ``NoFreeSlotsError``.
    ``None`` keeps the pre-existing sizing for callers that do not know the pool.
    """
    mgr = MTPHiddenStatesManager(
        _mtp_config(), torch.float16, hidden_size=8, max_num_requests=R, num_seq_slots=num_seq_slots
    )

    assert mgr.slot_manager.max_num_requests == expected_pool
    assert mgr.mtp_past_hidden_states_pool.shape[0] == expected_pool
    assert mgr.mtp_past_tokens_pool.shape[0] == expected_pool
    assert mgr.mtp_relaxed_delta_pool.shape[0] == expected_pool
    # Batch-position state is unaffected: the forward batch is still capped at
    # max_batch_size by the micro-batch scheduler.
    assert mgr.get_max_resource_count() == R


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MTP hidden-state pools are CUDA tensors")
def test_mtp_slot_pool_survives_a_full_overlap_turnover():
    """R retiring + R admitted must both hold slots at once.

    This is the exact interleaving the overlap scheduler produces and the one
    that used to exhaust the pool.
    """
    mgr = MTPHiddenStatesManager(
        _mtp_config(), torch.float16, hidden_size=8, max_num_requests=R, num_seq_slots=POOL
    )

    retiring = [mgr.slot_manager.add_slot(rid) for rid in range(R)]
    # Replacements are admitted before the deferred teardown frees the slots.
    incoming = [mgr.slot_manager.add_slot(rid) for rid in range(R, 2 * R)]

    assert len(set(retiring) | set(incoming)) == 2 * R
    assert all(0 <= slot < POOL + 1 for slot in retiring + incoming)


# ---------------------------------------------------------------------------
# Resource managers. Unlike SpecMetadata there is no single exit point to stamp
# the pool onto, so the plumbing is per-branch -- which is exactly how three
# managers were missed in a row. The AST guard below makes forgetting a branch a
# test failure instead of a runtime IndexError.
# ---------------------------------------------------------------------------

#: Managers that legitimately do not take a slot pool. Adding a name here must be
#: a deliberate act with a reason, which is the point of the allow-list.
_MANAGERS_WITHOUT_A_SLOT_POOL = {
    # The n-gram pool is keyed by pattern, not by request identity, and NGRAM is
    # absent from SpeculativeDecodingMode.support_overlap_scheduler(), so
    # py_executor_creator forces the overlap scheduler off and the headroom can
    # never apply.
    "NGramPoolManager",
    # Hidden-state export path; no per-request slot pool.
    "SaveHiddenStatesResourceManager",
}

_MANAGERS_WITH_A_SLOT_POOL = (
    MTPHiddenStatesManager,
    MTPEagleDynamicTreeResourceManager,
    Eagle3ResourceManager,
    Eagle3OneModelDynamicTreeResourceManager,
    SuffixAutomatonManager,
    SpecTreeManager,
)


@pytest.mark.cpu_only
def test_every_resource_manager_branch_forwards_the_slot_pool():
    """Mechanical guard: no branch of ``get_spec_resource_manager`` may omit it.

    ``num_seq_slots`` is computed once at the top of the function and then has to
    reach every manager it builds. A new speculation mode -- or a new manager in
    an existing mode's branch -- fails here rather than in production, where the
    symptom is an out-of-range ``py_seq_slot`` write into a pool sized for
    max_batch_size.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(get_spec_resource_manager)))

    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name is None or not name.endswith("Manager") or name in _MANAGERS_WITHOUT_A_SLOT_POOL:
            continue
        if not any(kw.arg == "num_seq_slots" for kw in node.keywords):
            missing.append(name)

    assert not missing, (
        f"get_spec_resource_manager builds {sorted(set(missing))} without forwarding "
        "num_seq_slots; slot-keyed pools would be sized at max_batch_size. Either pass "
        "it or justify the exemption in _MANAGERS_WITHOUT_A_SLOT_POOL."
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("manager", _MANAGERS_WITH_A_SLOT_POOL, ids=lambda m: m.__name__)
def test_slot_pool_managers_accept_an_optional_pool_size(manager):
    """The receiving end of the same contract, with ``None`` as the default.

    ``None`` -- not ``max_num_requests`` -- has to be the default so that a caller
    which does not know the pool (PP, no attention DP, overlap disabled) keeps the
    established sizing without every call site having to restate it.
    """
    param = inspect.signature(manager.__init__).parameters.get("num_seq_slots")

    assert param is not None, f"{manager.__name__} cannot be told its slot pool"
    assert param.default is None, f"{manager.__name__} must default to None, got {param.default!r}"


def _tree_manager(num_seq_slots):
    return SpecTreeManager(
        max_num_requests=R,
        use_dynamic_tree=True,
        max_total_draft_tokens=3,
        max_draft_len=3,
        eagle_choices=None,
        dynamic_tree_max_topK=2,
        num_seq_slots=num_seq_slots,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dynamic-tree slot storage is on CUDA")
@pytest.mark.parametrize("num_seq_slots,expected_slots", [(POOL, POOL), (None, R)])
def test_dynamic_tree_slot_storage_spans_the_slot_pool(num_seq_slots, expected_slots):
    """``DynamicTreeSlotStorage`` is documented as indexed by ``py_seq_slot``.

    It was nonetheless sized from ``num_trees`` (== max_batch_size), so the two
    disagreed by 2x once the headroom was on. The dummy row sits one past the
    pool, so every buffer is ``pool + 1`` deep.
    """
    storage = _tree_manager(num_seq_slots).slot_storage

    assert storage.dummy_slot_id == expected_slots
    for name in (
        "has_tree",
        "packed_mask",
        "position_offsets",
        "retrieve_index",
        "retrieve_next_token",
        "retrieve_next_sibling",
    ):
        assert getattr(storage, name).shape[0] == expected_slots + 1, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dynamic-tree slot storage is on CUDA")
def test_dynamic_tree_work_buffers_stay_at_max_batch_size():
    """The other family must not be widened along with it.

    ``num_trees`` indexes the build kernel's output by batch position, and the
    micro-batch scheduler caps the forward at max_batch_size. Widening it would
    waste memory quadratically in the tree dimensions for no benefit.
    """
    mgr = _tree_manager(POOL)

    assert mgr.num_trees == R
    assert mgr.retrieve_index.shape[0] == R
    assert mgr.retrieve_next_token.shape[0] == R
    assert mgr.retrieve_next_sibling.shape[0] == R
    assert mgr.num_slots == POOL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dynamic-tree slot storage is on CUDA")
def test_marking_a_high_slot_invalid_needs_the_pool():
    """The concrete failure, plus a negative control that it was reachable.

    ``Eagle3OneModelDynamicTreeResourceManager.free_resources`` calls
    ``mark_invalid(request.py_seq_slot)``, and with the headroom on ``py_seq_slot``
    ranges over the whole pool. Sized at max_batch_size the write is out of
    range, so the second half of the assertion is what proves the first half is
    not vacuous.
    """
    _tree_manager(POOL).slot_storage.mark_invalid(POOL - 1)

    with pytest.raises(IndexError):
        _tree_manager(None).slot_storage.mark_invalid(POOL - 1)


def _eagle_config():
    # Deliberately not an EagleDecodingConfig: that keeps max_total_draft_tokens
    # on the max_draft_len branch and leaves spec_tree_manager unbuilt, so this
    # exercises slot_manager sizing only.
    return types.SimpleNamespace(
        max_draft_len=2,
        num_capture_layers=1,
        use_relaxed_acceptance_for_thinking=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Eagle3 hidden states are CUDA tensors")
@pytest.mark.parametrize("num_seq_slots,expected_pool", [(POOL, POOL + 1), (None, R + 1)])
def test_eagle3_slot_manager_spans_the_slot_pool(num_seq_slots, expected_pool):
    """``Eagle3ResourceManager`` sized its ``SlotManager`` from ``max_seq_len``.

    That is a token count standing in for a slot count -- accidentally generous
    for most configurations, but not for ``max_batch_size == max_seq_len``, where
    the pool lands exactly one slot short of a full overlap turnover.
    """
    mgr = Eagle3ResourceManager(
        _eagle_config(),
        torch.float16,
        hidden_size=8,
        max_num_requests=R,
        max_seq_len=4,
        max_num_tokens=64,
        num_seq_slots=num_seq_slots,
    )

    assert mgr.slot_manager.max_num_requests == expected_pool
    assert mgr.relaxed_delta_pool.shape[0] == expected_pool
    assert len(mgr.seq_lens) == expected_pool
    assert len(mgr.start_indices) == expected_pool
    # Batch-position state is untouched.
    assert mgr.batch_indices_cuda.shape[0] == R


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Eagle3 hidden states are CUDA tensors")
def test_eagle3_keeps_the_max_seq_len_floor():
    """Existing deployments must not shrink.

    ``max_seq_len`` stays a floor so that every configuration where it already
    exceeded the slot pool allocates exactly what it did before this change.
    """
    mgr = Eagle3ResourceManager(
        _eagle_config(),
        torch.float16,
        hidden_size=8,
        max_num_requests=R,
        max_seq_len=1024,
        max_num_tokens=64,
        num_seq_slots=POOL,
    )

    assert mgr.slot_manager.max_num_requests == 1024 + 1


def _sa_manager(num_seq_slots, **config_kwargs):
    config = SAConfig(max_seq_len=1024, max_slots=R, **config_kwargs)
    return SuffixAutomatonManager(config, R, 1024, num_seq_slots=num_seq_slots)


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_seq_slots,expected_pool", [(POOL, POOL), (None, R)])
def test_sa_pool_spans_the_slot_pool(num_seq_slots, expected_pool):
    """SA slots are held for a request id's lifetime, so the pool follows it.

    The dummy slot index is derived from ``pool_size``, so it moves with the pool
    rather than colliding with a real slot.
    """
    mgr = _sa_manager(num_seq_slots)

    assert mgr.pool_size == expected_pool
    assert len(mgr._free_slots) == expected_pool
    assert mgr._dummy_slot_index == expected_pool


@pytest.mark.cpu_only
def test_sa_pool_survives_a_full_overlap_turnover():
    """2 * max_batch_size concurrent slots, with a negative control.

    Without the pool the (max_batch_size + 1)-th allocation has nothing free and
    nothing retained to evict, which is a hard ``RuntimeError`` mid-run.
    """
    mgr = _sa_manager(POOL)
    slots = [mgr._allocate_slot() for _ in range(POOL)]
    assert len(set(slots)) == POOL

    starved = _sa_manager(None)
    for _ in range(R):
        starved._allocate_slot()
    with pytest.raises(RuntimeError, match="No free or retained slots"):
        starved._allocate_slot()


@pytest.mark.cpu_only
def test_an_explicit_sa_pool_is_honoured_but_validated():
    """``global_pool_size`` is a memory contract, so it is never grown silently.

    It is validated against the sequence-slot pool instead, turning what would be
    a mid-run slot exhaustion into a startup error that names the real bound.
    """
    grown = _sa_manager(POOL, enable_global_pool=True, global_pool_size=64)
    assert grown.pool_size == 64

    with pytest.raises(ValueError, match="sequence slots"):
        _sa_manager(POOL, enable_global_pool=True, global_pool_size=R)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dynamic-tree slot storage is on CUDA")
def test_eagle3_one_model_dynamic_tree_forwards_the_slot_pool():
    """End-to-end for the manager whose ``free_resources`` triggers the write."""
    config = types.SimpleNamespace(
        use_dynamic_tree=True,
        max_draft_len=3,
        tokens_per_gen_step=4,
        eagle_choices=None,
        dynamic_tree_max_topK=2,
    )

    mgr = Eagle3OneModelDynamicTreeResourceManager(config, R, num_seq_slots=POOL)

    assert mgr.spec_tree_manager.slot_storage.dummy_slot_id == POOL
    assert mgr.spec_tree_manager.num_trees == R
    assert mgr.batch_indices_cuda.shape[0] == R
    mgr.free_resources(types.SimpleNamespace(py_seq_slot=POOL - 1))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
