# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical/state tests of the pure-Torch minimum-length helper without native initialization."""

from __future__ import annotations

import ast
import copy
import gc
import importlib.util
import json
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[5]
_PATH = _ROOT / "tensorrt_llm/_torch/speculative/minimum_length.py"
_SPEC = importlib.util.spec_from_file_location("_dspark_minimum_length_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
minimum = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = minimum
_SPEC.loader.exec_module(minimum)


@dataclass
class _Request:
    py_seq_slot: int | None = 0
    py_min_length: int = 0
    py_end_id: int = 0
    py_original_end_id: int = 0
    py_stop_words_list: list[list[int]] | None = None
    py_client_id: int = 1
    py_request_id: int | None = 1
    py_orig_prompt_len: int = 1000
    generated: int = 0
    is_dummy: bool = False
    is_context_init_state: bool = False
    is_last_context_chunk: bool = True

    def get_num_tokens(self, beam: int) -> int:
        assert beam == 0
        return self.py_orig_prompt_len + self.generated


def _metadata(slots: int = 4, batch: int = 3, device: str = "cpu") -> SimpleNamespace:
    return SimpleNamespace(
        supports_min_tokens=True,
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: True),
        num_seq_slots=slots,
        max_num_requests=batch,
        vocab_size=8,
        minimum_length_state=minimum.MinimumLengthState.create(slots, device),
        batch_slot_ids=torch.empty(batch, dtype=torch.long, device=device),
        is_ragged_verify=False,
    )


def _populate(meta: SimpleNamespace, requests: list[_Request]) -> None:
    slots = [minimum.request_slot(r, meta.minimum_length_state.slots) for r in requests]
    meta.batch_slot_ids[: len(slots)].copy_(torch.tensor(slots, device=meta.batch_slot_ids.device))
    minimum.populate(meta, requests)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "minimum_value,generated,expected", [(0, 0, 0), (3, 0, 3), (3, 2, 1), (3, 3, 0), (3, 90, 0)]
)
def test_remaining_counts_only_generated_tokens(
    minimum_value: int, generated: int, expected: int
) -> None:
    assert minimum.minimum_remaining(minimum_value, generated) == expected
    with pytest.raises(ValueError, match="nonnegative"):
        minimum.minimum_remaining(-1, generated)


@pytest.mark.cpu_only
def test_forced_acceptance_does_not_admit_minimum_length(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: True), draft_is_embedded_in_target=False
    )
    monkeypatch.setenv("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", "2")
    assert not minimum.supports_config(cfg)


@pytest.mark.cpu_only
@pytest.mark.parametrize("standalone,dspark", [(True, True), (False, True), (True, False)])
def test_capability_requires_standalone_dspark(
    standalone: bool, dspark: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS", raising=False)
    cfg = SimpleNamespace(
        spec_dec_mode=SimpleNamespace(is_dspark=lambda: dspark),
        draft_is_embedded_in_target=not standalone,
    )
    assert minimum.supports_config(cfg) == (standalone and dspark)
    assert not minimum.supports_config(None)


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_mixed_context_recovery_and_bonus_positions_restore_raw_scores(dtype: torch.dtype) -> None:
    meta = _metadata()
    requests = [
        _Request(py_seq_slot=2, py_min_length=1, py_original_end_id=4),
        _Request(py_seq_slot=0, py_min_length=2, py_stop_words_list=[[3], [5, 6]]),
        _Request(py_seq_slot=1, py_min_length=4),
    ]
    _populate(meta, requests)
    logits = torch.arange(72, dtype=dtype).reshape(9, 8)
    original = logits.clone()
    drafts = torch.tensor([[1, 2, 3], [1, 2, 3]])
    returned, saved = minimum.apply(logits, drafts, 1, 3, meta)
    expected = original.clone()
    expected[0, 4] = -float("inf")
    expected[1:3, 0] = -float("inf")
    expected[1:3, 3] = -float("inf")
    expected[5:9, 0] = -float("inf")  # Includes the bonus row while still below four.
    assert returned is logits
    assert torch.equal(logits, expected)
    minimum.restore(logits, saved)
    assert torch.equal(logits, original)


@pytest.mark.cpu_only
@pytest.mark.parametrize("offset", [0, 4, 8])
def test_vocab_shard_owns_only_its_stop_columns(offset: int) -> None:
    meta = _metadata(batch=1)
    meta.vocab_size = 12
    request = _Request(py_min_length=1, py_stop_words_list=[[4], [7]])
    _populate(meta, [request])
    logits = torch.tensor([[20.0, 10.0, 5.0, 1.0]])
    original = logits.clone()
    _, saved = minimum.apply(
        logits, torch.empty((1, 0), dtype=torch.int), 0, 1, meta, vocab_offset=offset
    )
    expected = original.clone()
    for token in (0, 4, 7):
        if offset <= token < offset + 4:
            expected[0, token - offset] = -float("inf")
    assert torch.equal(logits, expected)
    minimum.restore(logits, saved)
    assert torch.equal(logits, original)


@pytest.mark.cpu_only
def test_ragged_verify_windows_do_not_borrow_neighbor_positions() -> None:
    meta = _metadata()
    requests = [
        _Request(py_seq_slot=0, py_min_length=1),
        _Request(py_seq_slot=1, py_min_length=2),
        _Request(py_seq_slot=2, py_min_length=1),
    ]
    _populate(meta, requests)
    meta.is_ragged_verify = True
    meta.verify_lens = torch.tensor([2, 4], dtype=torch.int32)
    meta.qo_indptr = torch.tensor([0, 2, 6], dtype=torch.int32)
    meta.total_verify_tokens = 6
    logits = torch.zeros((7, 8))
    _, saved = minimum.apply(logits, torch.zeros((2, 3), dtype=torch.int), 1, 3, meta)
    assert torch.isneginf(logits[:, 0]).tolist() == [True, True, True, True, False, False, False]
    minimum.restore(logits, saved)
    assert torch.equal(logits, torch.zeros_like(logits))


@pytest.mark.cpu_only
@pytest.mark.parametrize("counts", [[1, 1, 1], [1, 3, 4], [4, 2, 1]])
def test_progress_counts_actual_accepted_prefix_not_proposal_width(counts: list[int]) -> None:
    meta = _metadata()
    requests = [
        _Request(py_seq_slot=2, py_min_length=7),
        _Request(py_seq_slot=0, py_min_length=7),
        _Request(py_seq_slot=1, py_min_length=7),
    ]
    _populate(meta, requests)
    accepted = (torch.full((3, 4), 7, dtype=torch.int), torch.tensor(counts))
    assert minimum.commit(accepted, 3, meta) is accepted
    assert meta.minimum_length_state.remaining.tolist() == [
        7 - counts[1],
        7 - counts[2],
        7 - counts[0],
        0,
        0,
    ]


@pytest.mark.cpu_only
def test_host_lag_and_graph_bucket_views_share_progress_and_scratch() -> None:
    meta = _metadata(slots=8, batch=4)
    view = copy.copy(meta)
    view.max_num_requests = 2
    minimum.prepare(view)
    assert view.minimum_length_state is meta.minimum_length_state
    assert view.dummy_slot_row == 8
    req = _Request(py_seq_slot=7, py_min_length=5)
    dummy = _Request(py_seq_slot=0, is_dummy=True, py_min_length=100)
    _populate(view, [req, dummy])
    assert view.batch_slot_ids[:2].tolist() == [7, 8]
    minimum.commit((torch.zeros((2, 3), dtype=torch.int), torch.tensor([2, 3])), 2, view)
    assert meta.minimum_length_state.remaining[7].item() == 3
    assert meta.minimum_length_state.remaining[8].item() == 0
    _populate(view, [req, dummy])  # Host still has no generated tokens.
    assert meta.minimum_length_state.remaining[7].item() == 3
    req.generated = 2
    _populate(view, [req, dummy])
    assert meta.minimum_length_state.remaining[7].item() == 3


@pytest.mark.cpu_only
def test_slot_reuse_and_weak_ownership_reset_even_when_numeric_ids_repeat() -> None:
    meta = _metadata(batch=1)
    req = _Request(py_min_length=5)
    _populate(meta, [req])
    minimum.commit((torch.ones((1, 3), dtype=torch.int), torch.tensor([3])), 1, meta)
    ref = weakref.ref(req)
    del req
    gc.collect()
    assert ref() is None
    new = _Request(py_min_length=5, generated=1)
    _populate(meta, [new])
    assert meta.minimum_length_state.remaining[0].item() == 4
    new.py_min_length = 6
    with pytest.raises(ValueError, match="changed within"):
        _populate(meta, [new])


@pytest.mark.cpu_only
def test_chunk_only_context_does_not_advance_the_minimum() -> None:
    meta = _metadata(batch=1)
    req = _Request(py_min_length=2, is_context_init_state=True, is_last_context_chunk=False)
    _populate(meta, [req])
    accepted = (torch.ones((1, 1), dtype=torch.int), torch.ones(1, dtype=torch.int))
    minimum.commit(accepted, 1, meta)
    assert meta.minimum_length_state.remaining[0].item() == 2
    req.is_last_context_chunk = True
    _populate(meta, [req])
    minimum.commit(accepted, 1, meta)
    assert meta.minimum_length_state.remaining[0].item() == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("minimum_value", [0, 3])
def test_stable_owner_population_skips_unchanged_device_uploads(
    minimum_value: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = _metadata(batch=1)
    request = _Request(py_min_length=minimum_value)
    _populate(meta, [request])

    def unexpected_upload(*args: object) -> None:
        pytest.fail("unchanged minimum-length state must not upload")

    monkeypatch.setattr(minimum, "_device", unexpected_upload)
    minimum.populate(meta, [request])


@pytest.mark.cpu_only
def test_zero_minimum_and_ignore_eos_keep_neutral_and_original_token_semantics() -> None:
    meta = _metadata(batch=1)
    req = _Request(py_min_length=0, py_stop_words_list=[[i] for i in range(100)])
    _populate(meta, [req])
    raw = torch.randn(1, 8)
    original = raw.clone()
    _, saved = minimum.apply(raw, torch.empty((1, 0), dtype=torch.int), 0, 1, meta)
    assert torch.equal(raw, original)
    minimum.restore(raw, saved)
    req = _Request(py_min_length=1, py_end_id=-1, py_original_end_id=0)
    _populate(meta, [req])
    _, saved = minimum.apply(raw, torch.empty((1, 0), dtype=torch.int), 0, 1, meta)
    assert torch.isneginf(raw[0, 0]) and raw[0, -1] == original[0, -1]
    minimum.restore(raw, saved)
    assert torch.equal(raw, original)


@pytest.mark.cpu_only
def test_invalid_slots_stop_bounds_and_layouts_fail_closed() -> None:
    meta = _metadata(batch=1)
    for slot in (-1, 4, 5):
        with pytest.raises(ValueError, match="sequence slot"):
            _populate(meta, [_Request(py_seq_slot=slot, py_min_length=1)])
    with pytest.raises(ValueError, match="vocabulary"):
        _populate(meta, [_Request(py_min_length=1, py_original_end_id=8)])
    with pytest.raises(ValueError, match="cannot cover"):
        _populate(meta, [_Request(py_min_length=1, py_stop_words_list=[[i] for i in range(8)])])
    with pytest.raises(ValueError, match="at most"):
        minimum.termination_ids(_Request(py_stop_words_list=[[i] for i in range(65)]))
    with pytest.raises(ValueError, match="nonnegative"):
        minimum.termination_ids(_Request(py_stop_words_list=[[-2]]))
    _populate(meta, [_Request(py_min_length=1)])
    with pytest.raises(RuntimeError, match="row count"):
        minimum.apply(torch.zeros(2, 8), torch.empty((1, 0)), 0, 1, meta)
    with pytest.raises(RuntimeError, match="request layout"):
        minimum.apply(torch.zeros(1, 8), torch.empty((1, 0)), 0, 1, meta, vocab_offset=-1)
    with pytest.raises(RuntimeError, match="accepted-prefix"):
        minimum.commit((torch.zeros(1, 2), torch.tensor([1, 1])), 1, meta)


@pytest.mark.cpu_only
def test_all_but_one_termination_tokens_leave_a_valid_target_choice() -> None:
    meta = _metadata(batch=1)
    request = _Request(py_min_length=1, py_stop_words_list=[[i] for i in range(7)])
    _populate(meta, [request])
    logits = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    original = logits.clone()
    _, saved = minimum.apply(logits, torch.empty((1, 0)), 0, 1, meta)
    assert torch.isneginf(logits[0, :7]).all()
    assert logits.argmax(1).item() == 7
    minimum.restore(logits, saved)
    assert torch.equal(logits, original)


@pytest.mark.cpu_only
def test_recovery_then_fully_accepted_prefix_allows_eos_at_the_exact_minimum() -> None:
    meta = _metadata(batch=1)
    request = _Request(py_min_length=3)
    _populate(meta, [request])
    raw = torch.tensor([[10.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    original = raw.clone()
    # An EOS proposal at position zero cannot match the masked target. The
    # target recovery contributes exactly one generated non-EOS token.
    _, saved = minimum.apply(raw, torch.tensor([[0, 0]]), 0, 1, meta)
    assert raw.argmax(1).tolist() == [1, 1, 1]
    minimum.commit((torch.tensor([[1, -1, -1]]), torch.tensor([1])), 1, meta)
    minimum.restore(raw, saved)
    _populate(meta, [request])  # The host token list still lags this commit.
    _, saved = minimum.apply(raw, torch.tensor([[1, 1]]), 0, 1, meta)
    assert raw.argmax(1).tolist() == [1, 1, 0]
    # Two matched proposals plus the EOS bonus finish at three non-EOS tokens.
    minimum.commit((torch.tensor([[1, 1, 0]]), torch.tensor([3])), 1, meta)
    minimum.restore(raw, saved)
    assert meta.minimum_length_state.remaining[0].item() == 0
    assert torch.equal(raw, original)


def _real_accept_methods() -> type:
    """Use the production transaction with controlled acceptance outputs, without native imports."""
    path = _ROOT / "tensorrt_llm/_torch/speculative/interface.py"
    tree = ast.parse(path.read_text())
    worker = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SpecWorkerBase"
    )
    methods = [
        copy.deepcopy(n)
        for n in worker.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_accept_draft_tokens", "_commit_occurrence_counts"}
    ]
    namespace = {
        "torch": torch,
        "minimum_length_ops": minimum,
        "VocabShardedLogits": type("VocabShardedLogits", (), {}),
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.ClassDef(name="Worker", bases=[], keywords=[], body=methods, decorator_list=[]),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Worker"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("rejection", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_real_accept_transaction_restores_scores_and_commits_only_result(
    rejection: bool, raises: bool
) -> None:
    Worker = _real_accept_methods()
    worker = Worker()
    worker._apply_occurrence_penalties = lambda logits, *_: logits
    worker._can_use_rejection_sampling = lambda _: rejection
    worker._rejection_buffers_valid = lambda *args: True
    meta = _metadata(batch=1)
    meta.enable_penalty = False
    meta.draft_probs_last_dim = meta.draft_probs_vocab_size = 8
    meta.draft_probs = torch.ones(5, 2, 8) / 8
    req = _Request(py_min_length=3)
    _populate(meta, [req])
    original_q = meta.draft_probs.clone()
    logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    original = logits.clone()
    accepted = (torch.tensor([[1, -1, -1]]), torch.tensor([1]))

    def consume(scores: torch.Tensor, *args: object) -> tuple[torch.Tensor, torch.Tensor]:
        assert torch.isneginf(scores[:, 0]).all()
        assert torch.equal(meta.draft_probs, original_q)
        if raises:
            raise ValueError("controlled sampler failure")
        return accepted

    worker._sample_and_accept_draft_tokens_base = consume
    worker._sample_and_accept_draft_tokens_rejection = consume
    if raises:
        with pytest.raises(ValueError, match="controlled sampler"):
            worker._accept_draft_tokens(logits, torch.ones(1, 2, dtype=torch.int), 0, 1, meta)
        assert meta.minimum_length_state.remaining[0].item() == 3
    else:
        assert (
            worker._accept_draft_tokens(logits, torch.ones(1, 2, dtype=torch.int), 0, 1, meta)
            is accepted
        )
        assert meta.minimum_length_state.remaining[0].item() == 2
    assert torch.equal(logits, original)


@pytest.mark.skipif(torch.version.cuda is None, reason="requires CUDA Torch and an allocated GPU")
def test_cuda_graphs_read_new_minima_and_share_progress_across_buckets() -> None:
    meta = _metadata(slots=2, batch=2, device="cuda")
    meta.batch_slot_ids.copy_(torch.tensor([0, 2], device="cuda"))
    graphs, results, logits = [], [], []
    static_inputs = []
    for batch in (1, 2):
        view = copy.copy(meta)
        view.max_num_requests = batch
        raw = torch.tensor([[10.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device="cuda").repeat(
            batch, 1
        )
        drafts = torch.empty((batch, 0), dtype=torch.int, device="cuda")
        counts = torch.ones(batch, dtype=torch.int, device="cuda")

        def step() -> torch.Tensor:
            _, saved = minimum.apply(raw, drafts, 0, batch, view)
            tokens = raw.argmax(dim=1).to(torch.int).unsqueeze(1)
            minimum.commit((tokens, counts), batch, view)
            minimum.restore(raw, saved)
            return tokens

        for _ in range(2):
            step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens = step()
        graphs.append(graph)
        results.append(tokens)
        logits.append(raw)
        # CUDA graphs retain addresses; external input tensors must stay alive.
        static_inputs.append((drafts, counts))
    req = _Request(py_min_length=2)
    _populate(meta, [req, _Request(py_seq_slot=None, is_dummy=True)])
    for index, expected, remaining in ((0, 1, 1), (1, 1, 0), (0, 0, 0)):
        graphs[index].replay()
        torch.cuda.synchronize()
        print(
            "MINIMUM_GRAPH_REPLAY",
            json.dumps(
                {
                    "bucket": index,
                    "counts": static_inputs[index][1].tolist(),
                    "remaining": meta.minimum_length_state.remaining.tolist(),
                    "token": results[index][0, 0].item(),
                    "commit_flags": meta.minimum_length_state.commit.tolist(),
                }
            ),
        )
        assert results[index][0, 0].item() == expected
        assert meta.minimum_length_state.remaining[0].item() == remaining
        assert torch.equal(static_inputs[index][1], torch.ones_like(static_inputs[index][1]))
        assert logits[index][0, 0].item() == 10.0
    assert meta.minimum_length_state.remaining[0].item() == 0
    assert meta.minimum_length_state.remaining[2].item() == 0
