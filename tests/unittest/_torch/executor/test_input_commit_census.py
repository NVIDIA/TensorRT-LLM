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
"""Engagement accounting of the GEN host bundle (TRTLLM_GEN_HOST_BUNDLE=1).

The log lines that prove the bundle engaged must come from live decode steps: an engine warm-up
runs the commit kernels too, and a line (or counter) from there says nothing about serving. These
tests pin that down: warm-up commits are neither counted nor logged nor charged to the VERIFY
budget, the census counts live steps only, every skipped step carries its reason, and every part
reports its launches and fallbacks.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor import input_commit as ic
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm.mapping import CpType

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch):
    """Everything input_commit.py logs, as plain strings; a fresh census for the test."""
    logged = []
    monkeypatch.setattr(ic, "logger", SimpleNamespace(info=logged.append, warning=logged.append))
    census = ic.bundle_census()
    census.reset()
    yield logged
    census.reset()


def _first(logged, text):
    return [m for m in logged if text in m]


# ---------------------------------------------------------------------------------------------
# The census itself (no GPU)
# ---------------------------------------------------------------------------------------------


def test_census_counts_live_steps_only(messages) -> None:
    census = ic.BundleCensus(first_log_steps=1000)
    census.count("commits")  # before any step: not live
    census.begin_step(live=False, skipped="warmup")
    census.count("commits")
    assert census.steps == 0 and census.get("commits") == 0
    census.begin_step(live=True)
    census.count("commits")
    census.count("staged_copies", 30)
    census.count("overlap_commits")
    census.begin_step(live=True, skipped="dummy_requests")
    census.begin_step(live=True, skipped="dummy_requests")
    census.begin_step(live=True, skipped="no_graph")
    assert census.steps == 4 and census.get("commits") == 1
    summary = census.summary()
    assert "steps=4 commits=1 overlap_commits=1 verified_commits=0" in summary
    assert "staged_copies_per_commit=30.0" in summary
    assert "skipped=dummy_requests:2,no_graph:1" in summary
    census.begin_step(live=False)  # the engine goes back to warm-up (e.g. a capture): paused again
    census.count("commits")
    assert census.get("commits") == 1
    assert not messages


def test_census_logs_at_decades_and_at_the_end(messages) -> None:
    census = ic.BundleCensus(first_log_steps=2)
    census.log()  # nothing happened yet: silent
    assert not messages
    for _ in range(200):
        census.begin_step(live=True)
    lines = _first(messages, "InputCommit census:")
    assert len(lines) == 3  # after 2, 20 and 200 live steps
    assert "steps=2 " in lines[0] and "steps=20 " in lines[1] and "steps=200 " in lines[2]
    census.log(final=True)
    assert "InputCommit census (final): steps=200 " in messages[-1]


def test_census_logs_every_linear_step_count_after_the_decades(messages) -> None:
    census = ic.BundleCensus(first_log_steps=2, linear_log_steps=20)
    for _ in range(100):
        census.begin_step(live=True)
    lines = _first(messages, "InputCommit census:")
    # decades up to the linear bound (2, 20), then every 20 live steps: a run that is killed without an engine
    # cleanup still has a census line close to its end
    assert [int(m.split("steps=")[1].split()[0]) for m in lines] == [2, 20, 40, 60, 80, 100]


def test_census_default_schedule_is_decades_then_every_ten_thousand(messages) -> None:
    census = ic.BundleCensus()
    for _ in range(35000):
        census.begin_step(live=True)
    lines = _first(messages, "InputCommit census:")
    assert [int(m.split("steps=")[1].split()[0]) for m in lines] == [100, 1000, 10000, 20000, 30000]


def test_census_paused_ignores_counts_and_restores_state() -> None:
    census = ic.BundleCensus()
    census.begin_step(live=True)
    with census.paused():
        census.count("commits")
        assert not census.live
    census.count("commits")
    assert census.live and census.get("commits") == 1


def test_malformed_env_int_is_ignored(monkeypatch: pytest.MonkeyPatch, messages) -> None:
    monkeypatch.setenv("TLLM_TEST_KNOB", "100")
    assert ic._env_int("TLLM_TEST_KNOB", 7) == 100
    monkeypatch.setenv("TLLM_TEST_KNOB", "-1")
    assert ic._env_int("TLLM_TEST_KNOB", 7) == -1
    monkeypatch.setenv("TLLM_TEST_KNOB", " ")
    assert ic._env_int("TLLM_TEST_KNOB", 7) == 7
    monkeypatch.delenv("TLLM_TEST_KNOB")
    assert ic._env_int("TLLM_TEST_KNOB", 7) == 7
    assert not messages
    monkeypatch.setenv("TLLM_TEST_KNOB", "all")
    assert ic._env_int("TLLM_TEST_KNOB", 7) == 7
    assert "TLLM_TEST_KNOB='all' is not an integer" in messages[-1]


def test_disable_switches_every_part_off_once(monkeypatch: pytest.MonkeyPatch, messages) -> None:
    monkeypatch.setattr(ic, "_GEN_HOST_BUNDLE", True)
    assert ic.gen_host_bundle_kernels_enabled()
    ic.disable_gen_host_bundle("attention DP is enabled")
    assert not ic.gen_host_bundle_kernels_enabled()
    assert len(messages) == 1 and "attention DP is enabled" in messages[0]
    ic.disable_gen_host_bundle("again")  # already off: silent
    assert len(messages) == 1
    monkeypatch.setattr(ic, "_GEN_HOST_BUNDLE", False)
    ic.disable_gen_host_bundle("never enabled")  # a run without the bundle logs nothing
    assert len(messages) == 1


# ---------------------------------------------------------------------------------------------
# Warm-up is not engagement (GPU)
# ---------------------------------------------------------------------------------------------


def _step(commit: ic.InputCommit, out: torch.Tensor, with_fixup: bool, value: int = 1) -> None:
    """One decode-like commit: three staged copies and, optionally, the overlap gathers."""
    commit.begin()
    commit.stage(out[0:4], [value] * 4)
    commit.stage(out[4:8], np.arange(4, dtype=np.int32) + value)
    commit.stage(out[8:12], [value, 0, value, 0])
    if with_fixup:
        device = out.device
        commit.set_overlap_fixup(
            previous_slots=[1, 0],
            tokens_per_req=2,
            draft_width=1,
            new_tokens=torch.arange(8, dtype=torch.int32, device=device).reshape(2, 2, 2)[:, :, :1],
            new_tokens_lens=torch.ones(2, dtype=torch.int32, device=device),
            next_draft_tokens=torch.zeros((2, 1), dtype=torch.int32, device=device),
            input_ids=out[16:32],
            input_ids_start=0,
            draft_tokens=out[32:48],
            draft_tokens_start=0,
            pos_offsets=torch.zeros(4, dtype=torch.int32, device=device),
            pos_offsets_start=0,
            kv_offsets=torch.zeros(2, dtype=torch.int32, device=device),
            kv_offsets_start=0,
        )
    commit.commit()


@requires_gpu
@pytest.mark.parametrize("bundle", [False, True])
def test_warmup_commits_do_not_count_or_log(
    bundle: bool, monkeypatch: pytest.MonkeyPatch, messages
) -> None:
    monkeypatch.setattr(ic, "_GEN_HOST_BUNDLE", bundle)
    census = ic.bundle_census()
    census.begin_step(live=True)  # a live census must not pick up the warm-up either
    commit = ic.InputCommit()
    ic.warmup_input_commit_kernels(commit)
    assert commit._num_commits == 0 and not commit._logged_overlap
    assert commit._events == [None, None] and commit._index == 0 and not commit.is_open
    assert census.live and census.get("commits") == 0 and census.get("staged_copies") == 0
    assert not messages, messages  # the old lines "first batched input commit (1 staged copies ..."

    # The first real steps log their own lines, once, with their own numbers.
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    _step(commit, out, with_fixup=False)
    _step(commit, out, with_fixup=True)
    _step(commit, out, with_fixup=True, value=5)
    torch.cuda.synchronize()
    assert _first(messages, "first batched input commit (3 staged copies, overlap fixup: False)")
    assert len(_first(messages, "first batched input commit")) == 1
    assert len(_first(messages, "first overlap-scheduler commit (3 staged copies)")) == 1
    assert commit._num_commits == 3
    assert census.get("commits") == 3 and census.get("overlap_commits") == 2
    assert census.get("staged_copies") == 9


@requires_gpu
def test_warmup_commits_do_not_use_up_the_verify_budget(
    monkeypatch: pytest.MonkeyPatch, messages
) -> None:
    monkeypatch.setattr(ic, "_VERIFY_COMMITS", 2)
    census = ic.bundle_census()
    census.begin_step(live=True)
    commit = ic.InputCommit()
    ic.warmup_input_commit_kernels(commit)  # two verified warm-up commits, none counted
    assert commit._num_commits == 0
    assert not _first(messages, "verify")
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    _step(commit, out, with_fixup=True)
    assert not _first(messages, "commits matched")  # one real commit of two
    _step(commit, out, with_fixup=True, value=3)
    assert len(_first(messages, "InputCommit verify: the first 2 commits matched")) == 1
    _step(commit, out, with_fixup=True, value=4)  # past the budget: no more verification
    assert census.get("verified_commits") == 2 and census.get("commits") == 3
    torch.cuda.synchronize()


@requires_gpu
def test_verify_all_counts_every_real_commit(monkeypatch: pytest.MonkeyPatch, messages) -> None:
    monkeypatch.setattr(ic, "_VERIFY_COMMITS", -1)
    census = ic.bundle_census()
    census.begin_step(live=True)
    commit = ic.InputCommit(capacity=64)  # arena growth under verification as well
    ic.warmup_input_commit_kernels(commit)
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    for step in range(6):
        _step(commit, out, with_fixup=bool(step % 2), value=step)
    assert census.get("verified_commits") == census.get("commits") == 6
    assert "commits matched" not in " ".join(messages)  # no fixed budget to announce


# ---------------------------------------------------------------------------------------------
# Every part reports its launches and fallbacks (GPU)
# ---------------------------------------------------------------------------------------------


def _block_offsets_case(num_pools: int, num_blocks: int, num_seqs: int):
    host = torch.randint(
        0, 1 << 20, (num_pools, num_seqs + 1, 2, num_blocks), dtype=torch.int32
    ).pin_memory()
    copy_idx = torch.arange(num_seqs, dtype=torch.int32).pin_memory()
    scales = torch.ones(num_pools, dtype=torch.int32).pin_memory()
    kv_offset = torch.zeros(num_pools, dtype=torch.int32).pin_memory()
    dst = torch.zeros((num_pools, num_seqs, 2, num_blocks), dtype=torch.int32, device="cuda")
    return host, dst, copy_idx, scales, kv_offset


@requires_gpu
def test_parts_report_launches_and_fallbacks(messages) -> None:
    census = ic.bundle_census()
    census.begin_step(live=True)

    # Block offsets: one launch per call, the C++ kernel when the layout does not qualify.
    case = _block_offsets_case(2, 976, 3)
    assert ic.copy_batch_block_offsets_from_host(*case)
    bad = list(case)
    bad[1] = bad[1].to(torch.int64)
    assert not ic.copy_batch_block_offsets_from_host(*bad)
    jobs = [ic.block_offsets_copy_job(*_block_offsets_case(1, 512, 2)) for _ in range(3)]
    ic.copy_block_offsets_jobs(jobs)  # one pair launch and one single launch
    assert census.get("block_offsets_triton") == 2 and census.get("block_offsets_cpp") == 1
    assert census.get("block_offsets_pair") == 1

    # State upload kernel: qualifies, or falls back to the two copies.
    src = [
        torch.arange(5, dtype=torch.int32).pin_memory(),
        torch.ones(5, dtype=torch.bool).pin_memory(),
    ]
    dst = [
        torch.zeros(5, dtype=torch.int32, device="cuda"),
        torch.zeros(5, dtype=torch.bool, device="cuda"),
    ]
    assert ic.copy_small_pinned_pair(dst[0], src[0], dst[1], src[1])
    assert not ic.copy_small_pinned_pair(dst[0], src[0].clone(), dst[1], src[1])  # pageable
    assert census.get("state_upload_kernel") == 1 and census.get("state_upload_fallback") == 1

    # Spec store scatter.
    slots = torch.tensor([2, 0], device="cuda")
    store = [
        torch.zeros((3, 4), dtype=torch.int32, device="cuda"),
        torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
        torch.zeros(4, dtype=torch.int32, device="cuda"),
        torch.zeros((4, 1), dtype=torch.int32, device="cuda"),
    ]
    srcs = [
        torch.ones((2, 3), dtype=torch.int32, device="cuda"),
        torch.ones((2, 2), dtype=torch.int32, device="cuda"),
        torch.ones(2, dtype=torch.int32, device="cuda"),
        torch.ones((2, 1), dtype=torch.int32, device="cuda"),
    ]
    assert ic.scatter_spec_store(slots, *store, *srcs)
    srcs[0] = torch.ones((2, 4), dtype=torch.int32, device="cuda")  # wrong width: layout mismatch
    assert not ic.scatter_spec_store(slots, *store, *srcs)
    assert census.get("spec_scatter_kernel") == 1 and census.get("spec_scatter_fallback") == 1
    torch.cuda.synchronize()


@requires_gpu
def test_parts_do_not_count_while_the_engine_warms_up(messages) -> None:
    census = ic.bundle_census()
    census.begin_step(live=False)
    case = _block_offsets_case(1, 64, 2)
    assert ic.copy_batch_block_offsets_from_host(*case)
    ic.warmup_input_commit_kernels(ic.InputCommit())
    assert census.get("block_offsets_triton") == 0
    census.begin_step(live=True)
    assert ic.copy_batch_block_offsets_from_host(*case)
    assert census.get("block_offsets_triton") == 1
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------
# The engine reports why a step did not use the commit
# ---------------------------------------------------------------------------------------------


def _engine(**overrides) -> PyTorchModelEngine:
    """The attributes _input_commit_for_step reads: a one-model spec engine, TP, no extras."""
    engine = object.__new__(PyTorchModelEngine)
    engine.__dict__.update(
        _input_commit=object(),
        _is_warmup=False,
        use_mrope=False,
        max_beam_width=1,
        is_draft_model=False,
        enable_attention_dp=False,
        lora_model_config=None,
        guided_decoder=None,
        enable_spec_decode=True,
        spec_config=SimpleNamespace(spec_dec_mode=SimpleNamespace(has_draft_model=lambda: False)),
        mapping=SimpleNamespace(cp_config={}, has_cp_helix=lambda: False),
        _is_encoder_decoder_model=lambda: False,
    )
    engine.__dict__.update(overrides)
    return engine


def _mrope_model():
    # use_mrope is derived from the model's rope scaling.
    config = SimpleNamespace(rope_scaling={"type": "mrope"})
    return SimpleNamespace(model_config=SimpleNamespace(pretrained_config=config))


def _requests(num_context=0, num_generation=3, dummy=False):
    generation = [
        SimpleNamespace(is_dummy=dummy and i == num_generation - 1) for i in range(num_generation)
    ]
    return SimpleNamespace(
        num_context_requests=num_context,
        num_generation_requests=num_generation,
        generation_requests=generation,
    )


@pytest.mark.parametrize(
    "overrides,step,expected",
    [
        ({}, {}, None),
        ({}, {"can_run_graph": False}, "no_graph"),
        ({}, {"requests": _requests(num_context=1)}, "context"),
        ({}, {"promoted": frozenset({3})}, "context"),
        ({}, {"requests": _requests(num_generation=0)}, "no_generation"),
        ({}, {"requests": _requests(dummy=True)}, "dummy_requests"),
        ({"model": _mrope_model()}, {}, "mrope_or_beam"),
        ({"max_beam_width": 2}, {}, "mrope_or_beam"),
        ({"is_draft_model": True}, {}, "draft_model"),
        ({"enable_attention_dp": True}, {}, "attention_dp"),
        ({"lora_model_config": object()}, {}, "lora_or_guided"),
        ({"guided_decoder": object()}, {}, "lora_or_guided"),
        ({"enable_spec_decode": False}, {}, "not_one_model_spec"),
        (
            {
                "spec_config": SimpleNamespace(
                    spec_dec_mode=SimpleNamespace(has_draft_model=lambda: True)
                )
            },
            {},
            "not_one_model_spec",
        ),
        (
            {
                "mapping": SimpleNamespace(
                    cp_config={"cp_type": CpType.STAR}, has_cp_helix=lambda: False
                )
            },
            {},
            "star_cp",
        ),
        (
            {"mapping": SimpleNamespace(cp_config={}, has_cp_helix=lambda: True)},
            {},
            "helix",
        ),
        ({}, {"metadata": object()}, "attn_backend"),
        ({}, {"metadata_fields": {"enable_flash_mla": True}}, "flash_mla"),
        ({}, {"metadata_fields": {"high_precision_kv_pool": object()}}, "high_precision_kv"),
    ],
)
def test_engine_reports_why_a_step_skips_the_commit(overrides, step, expected, messages) -> None:
    engine = _engine(**overrides)
    metadata = step.get("metadata")
    if metadata is None:
        metadata = object.__new__(TrtllmAttentionMetadata)
        metadata.__dict__.update(step.get("metadata_fields", {}))
    census = ic.bundle_census()
    result = engine._input_commit_for_step(
        step.get("requests", _requests()),
        SimpleNamespace(),
        metadata,
        step.get("can_run_graph", True),
        step.get("promoted", frozenset()),
    )
    assert (result is None) == (expected is not None)
    assert census.steps == 1
    assert census.summary().count(f"skipped={expected}:1" if expected else "skipped=none") == 1


def test_engine_census_ignores_warmup_and_a_disabled_commit(messages) -> None:
    census = ic.bundle_census()
    metadata = object.__new__(TrtllmAttentionMetadata)
    args = (_requests(), SimpleNamespace(), metadata, True, frozenset())
    assert _engine(_is_warmup=True)._input_commit_for_step(*args) is None
    assert census.steps == 0  # a warm-up step is not a live step
    assert _engine(_input_commit=None)._input_commit_for_step(*args) is None
    assert census.steps == 0  # the bundle is off: nothing to account
    assert _engine()._input_commit_for_step(*args) is not None
    assert census.steps == 1 and "skipped=none" in census.summary()
