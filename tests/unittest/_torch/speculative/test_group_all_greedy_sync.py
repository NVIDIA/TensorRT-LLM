# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the group-synchronized ``is_all_greedy_sample`` override.

Under ADP + LM-head TP with rejection sampling, the greedy-vs-advanced path
choice gates group collectives, so the model engine all-gathers the per-rank
flags and stores the group AND in ``SpecMetadata.group_all_greedy_sample``;
``_scan_one_model_sampling`` must then re-apply it on every rescan (populate
runs after the CUDA graph key is built and would otherwise resurrect the
rank-local value).

These tests call ``_scan_one_model_sampling`` unbound on a SimpleNamespace
stand-in, mirroring test_rejection_buffers_guard.py, so no GPU or full
SpecMetadata construction is needed.
"""

import types

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import SpecMetadata


def _fake_request(temperature=None, top_k=None, top_p=None, min_p=None, slot=0):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[temperature] if temperature is not None else None,
            top_k=[top_k] if top_k is not None else None,
            top_p=[top_p] if top_p is not None else None,
            min_p=[min_p] if min_p is not None else None,
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
    )


def _fake_meta(group_all_greedy_sample=None):
    meta = types.SimpleNamespace(
        runtime_draft_len=2,
        dummy_slot_row=0,
        spec_dec_mode=types.SimpleNamespace(use_one_engine=lambda: True),
        group_all_greedy_sample=group_all_greedy_sample,
    )
    # update_is_all_greedy_sample dispatches through self, so the stand-in needs the
    # method bound to itself -- calling the unbound SpecMetadata function with a plain
    # namespace is not enough.
    meta._scan_one_model_sampling = lambda requests: SpecMetadata._scan_one_model_sampling(
        meta, requests
    )
    return meta


def _scan(meta, requests):
    return SpecMetadata._scan_one_model_sampling(meta, requests)


def _refresh(meta, requests):
    return SpecMetadata.update_is_all_greedy_sample(meta, requests)


def test_local_value_used_when_no_group_sync():
    meta = _fake_meta(group_all_greedy_sample=None)
    _scan(meta, [_fake_request(), _fake_request()])
    assert meta.is_all_greedy_sample is True

    _scan(meta, [_fake_request(), _fake_request(temperature=0.8)])
    assert meta.is_all_greedy_sample is False


def test_group_override_pulls_greedy_rank_onto_advanced_path():
    # This rank's batch is all-greedy, but another rank in the LM-head-TP
    # group has a sampling request: the group AND (False) must win so the
    # whole group takes the advanced path together.
    meta = _fake_meta(group_all_greedy_sample=False)
    _scan(meta, [_fake_request(), _fake_request()])
    assert meta.is_all_greedy_sample is False


def test_group_override_survives_rescan():
    # populate_sampling_params_for_one_model rescans after the CUDA graph key
    # is built; the override must keep applying so the key, the buffers, and
    # the worker branches all agree.
    meta = _fake_meta(group_all_greedy_sample=False)
    for _ in range(3):
        _scan(meta, [_fake_request()])
        assert meta.is_all_greedy_sample is False


def test_group_override_true_keeps_greedy():
    meta = _fake_meta(group_all_greedy_sample=True)
    _scan(meta, [_fake_request()])
    assert meta.is_all_greedy_sample is True


def test_refresh_discards_warmup_group_value_for_real_request():
    meta = _fake_meta()

    # Model warmup carries no sampling parameters and is synchronized as an
    # all-greedy group decision.
    warmup = _fake_request()
    warmup.state = LlmRequestState.CONTEXT_INIT
    _scan(meta, [warmup])
    assert meta.is_all_greedy_sample is True
    meta.group_all_greedy_sample = True

    # The first real request requires rejection sampling. Refresh must derive
    # its local non-greedy value without reapplying the warmup decision; the
    # model engine synchronizes a fresh group value immediately afterwards.
    real_request = _fake_request(temperature=0.7, top_k=50, top_p=0.9)
    _refresh(meta, [real_request])

    assert meta.group_all_greedy_sample is None
    assert meta.is_all_greedy_sample is False


# --- min_p's effect on the synchronized flag (hazard B5) ---------------------------
#
# min_p now participates in params_imply_greedy_decoding, so it can move
# is_all_greedy_sample -- the flag that gates group collectives under ADP + LM-head TP,
# and that doubles as a CUDA-graph key. What matters is that it moves the *same* flag the
# group AND is taken over, so a rank holding the only min_p request in the group still
# ends up on the same path as every other rank.
#
# The all-gather itself needs more than one GPU and is not covered here; this is the
# rank-local half, which is where the classification bug would live.


def test_min_p_only_request_makes_the_rank_non_greedy():
    # The silent failure this guards: min_p classified as greedy sends the request down
    # the argmax fast path, where min_p is not applied at all and nothing reports it.
    meta = _fake_meta(group_all_greedy_sample=None)
    _scan(meta, [_fake_request(), _fake_request(min_p=0.05)])
    assert meta.is_all_greedy_sample is False


def test_min_p_one_is_explicit_greedy_and_leaves_the_flag_alone():
    # min_p == 1.0 keeps only the argmax, so it is greedy by definition; a rank whose
    # only sampling knob is min_p=1.0 must not drag its group onto the advanced path.
    meta = _fake_meta(group_all_greedy_sample=None)
    _scan(meta, [_fake_request(), _fake_request(min_p=1.0)])
    assert meta.is_all_greedy_sample is True


def test_group_override_pulls_a_min_p_free_rank_onto_the_advanced_path():
    # This rank's batch has no min_p and no other sampling knob, but some other rank in
    # the LM-head-TP group holds a min_p request. The group AND (False) must win, or the
    # two ranks disagree about whether to run the collectives at all.
    meta = _fake_meta(group_all_greedy_sample=False)
    _scan(meta, [_fake_request(), _fake_request()])
    assert meta.is_all_greedy_sample is False


def test_min_p_rank_still_yields_to_a_greedy_group_decision():
    # The mirror image, and the one that says min_p goes through the same override rather
    # than around it: a local min_p request cannot pull this rank off a group-greedy
    # decision on its own.
    meta = _fake_meta(group_all_greedy_sample=True)
    _scan(meta, [_fake_request(min_p=0.05)])
    assert meta.is_all_greedy_sample is True
