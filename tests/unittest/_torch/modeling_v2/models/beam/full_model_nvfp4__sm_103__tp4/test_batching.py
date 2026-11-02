# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched engine steps of the Beam attention contract: request spans, rejections and rows.

The native runtime orders a step's requests contexts first, then generations; each request
has its own cached prefix and KV length. ``check_batch_metadata`` must return each request's
``(past, new tokens)`` from the prepared metadata and reject a step whose rows or requests
disagree; ``check_metadata`` keeps the one-request contract; the full model selects the last
row of each request for the sampler. Needs one CUDA device (a small KVCacheManagerV2); no
model files.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4 import (
    _attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.model import (  # noqa: E501
    BeamFullModelNvfp4Sm103Tp4,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

KV_HEADS, HEAD_DIM, TOKENS_PER_BLOCK, MAX_SEQ_LEN = 2, 128, 32, 4096
#: (cached tokens, new tokens) per request: two contexts (fresh, after a cached prefix) and
#: two generation requests at different depths.
CONTEXTS = ((0, 40), (64, 20))
GENERATIONS = ((100, 1), (7, 1))


@contextlib.contextmanager
def prepared_step(contexts, generations, *, cache_reuse=True):
    """A one-layer BF16 KVCacheManagerV2 and prepared metadata for one engine step."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    requests = list(contexts) + list(generations)
    mgr = KVCacheManagerV2(
        KvCacheConfig(max_tokens=64 * TOKENS_PER_BLOCK * len(requests), enable_block_reuse=False),
        CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=len(requests),
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.BF16,
    )
    try:
        request_ids = list(range(1, len(requests) + 1))
        mgr.add_dummy_requests(request_ids, token_nums=[past + n for past, n in requests])
        md = TrtllmAttentionMetadata(
            max_num_requests=len(requests),
            max_num_tokens=8192,
            kv_cache_manager=mgr,
            num_contexts=len(contexts),
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[past for past, _ in requests]
            ),
            seq_lens=torch.tensor([n for _, n in requests], dtype=torch.int),
            request_ids=request_ids,
            # The engine's context length of a context request is its new tokens (this chunk).
            prompt_lens=[n for _, n in contexts] + [past + n for past, n in generations],
            runtime_features=AttentionRuntimeFeatures(cache_reuse=cache_reuse),
        )
        md.prepare()
        yield md
    finally:
        mgr.shutdown()


def _spans(md, num_tokens):
    return _attention.check_batch_metadata(
        md, num_tokens, layer_idx=0, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM
    )


def test_mixed_step_spans():
    requests = list(CONTEXTS) + list(GENERATIONS)
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        assert _spans(md, sum(n for _, n in requests)) == requests


def test_generation_only_step_spans():
    with prepared_step((), GENERATIONS) as md:
        assert _spans(md, len(GENERATIONS)) == list(GENERATIONS)


def test_one_request_contract_is_kept():
    with prepared_step(CONTEXTS[1:], ()) as md:
        past = _attention.check_metadata(
            md, CONTEXTS[1][1], layer_idx=0, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM
        )
        assert past == CONTEXTS[1][0]
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        with pytest.raises(ValueError, match="one request per call"):
            _attention.check_metadata(md, 62, layer_idx=0, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM)


@pytest.mark.parametrize("num_tokens", [61, 63, 1])
def test_rows_must_be_the_requests_new_tokens(num_tokens):
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        with pytest.raises(ValueError, match="do not fill"):
            _spans(md, num_tokens)


def test_cached_prefix_needs_paged_context_fmha():
    with prepared_step(CONTEXTS, GENERATIONS, cache_reuse=False) as md:
        with pytest.raises(ValueError, match="paged-context FMHA"):
            _spans(md, 62)


def test_generation_request_appends_one_token():
    with prepared_step(CONTEXTS, ((100, 2),)) as md:
        with pytest.raises(ValueError, match="exactly one token"):
            _spans(md, 62)


def test_unheld_layer_is_rejected():
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        with pytest.raises(ValueError, match="does not hold logical layer 3"):
            _attention.check_batch_metadata(
                md, 62, layer_idx=3, num_kv_heads=KV_HEADS, head_dim=HEAD_DIM
            )


REQUESTS = list(CONTEXTS) + list(GENERATIONS)
NUM_TOKENS = sum(n for _, n in REQUESTS)


def _call_state(diagnostics, manager_checks):
    """The per-call state of the full model that ``check_call`` reads (no weights)."""
    return SimpleNamespace(
        max_tokens=8192,
        device=torch.device("cuda", torch.cuda.current_device()),
        local_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        kv_cache_dtype=DataType.BF16,
        diagnostics=diagnostics,
        _checked_manager=None,
        check_cache_manager=manager_checks.append,
    )


def _call_inputs():
    ids = torch.zeros(NUM_TOKENS, dtype=torch.int64, device="cuda")
    positions = torch.cat([torch.arange(p, p + n) for p, n in REQUESTS]).cuda()
    return ids, positions


check_call = BeamFullModelNvfp4Sm103Tp4.check_call


@pytest.mark.parametrize("diagnostics", [False, True], ids=["off", "on"])
def test_call_returns_spans_and_compares_positions_in_diagnostic_mode_only(diagnostics):
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        state = _call_state(diagnostics, [])
        ids, positions = _call_inputs()
        flat, flat_positions, spans = check_call(state, ids.view(1, -1), positions, md)
        assert spans == REQUESTS
        assert torch.equal(flat, ids) and torch.equal(flat_positions, positions)
        shifted = positions + 1
        if diagnostics:
            with pytest.raises(ValueError, match="positions must be exactly"):
                check_call(state, ids, shifted, md)
        else:
            _, flat_positions, spans = check_call(state, ids, shifted, md)
            assert spans == REQUESTS and torch.equal(flat_positions, shifted)


@pytest.mark.parametrize("diagnostics", [False, True], ids=["off", "on"])
@pytest.mark.parametrize(
    "fault,match",
    [
        ("ids_dtype", "torch.int64"),
        ("ids_shape", r"\[T\] or \[1, T\]"),
        ("rows", "do not fill"),
        ("positions_float", "integer tensor"),
        ("positions_count", "integer tensor"),
        ("positions_device", "integer tensor"),
        ("generation_rows", "exactly 2 tokens"),
        ("extra_kv_tokens", "extra KV tokens"),
        ("no_manager", "needs a KV cache manager"),
    ],
)
def test_call_rejects_malformed_required_inputs_in_every_mode(diagnostics, fault, match):
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        state = _call_state(diagnostics, [])
        ids, positions = _call_inputs()
        kwargs = {}
        if fault == "ids_dtype":
            ids = ids.to(torch.int32)
        elif fault == "ids_shape":
            ids = ids.view(2, -1)
        elif fault == "rows":
            ids, positions = ids[:-1], positions[:-1]
        elif fault == "positions_float":
            positions = positions.float()
        elif fault == "positions_count":
            positions = positions[:-1]
        elif fault == "positions_device":
            positions = positions.cpu()
        elif fault == "generation_rows":
            kwargs["generation_rows"] = 2
        elif fault == "extra_kv_tokens":
            kwargs["extra_kv_tokens"] = 6
        else:
            md.kv_cache_manager = None
        with pytest.raises(ValueError, match=match):
            check_call(state, ids, positions, md, **kwargs)


def test_cache_manager_geometry_is_checked_once_per_manager():
    checks = []
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        state = _call_state(False, checks)
        ids, positions = _call_inputs()
        for _ in range(3):
            check_call(state, ids, positions, md)
        assert checks == [md.kv_cache_manager]
    with prepared_step(CONTEXTS, GENERATIONS) as md:
        check_call(state, ids, positions, md)
        assert len(checks) == 2 and checks[-1] is md.kv_cache_manager


def test_rejected_cache_manager_is_checked_again():
    def reject(manager):
        raise ValueError("the KV cache manager must hold logical layers 0..51")

    with prepared_step(CONTEXTS, GENERATIONS) as md:
        state = _call_state(False, [])
        state.check_cache_manager = reject
        ids, positions = _call_inputs()
        for _ in range(2):
            with pytest.raises(ValueError, match="must hold logical layers"):
                check_call(state, ids, positions, md)
        assert state._checked_manager is None


@pytest.mark.parametrize("diagnostics", [False, True], ids=["off", "on"])
def test_gather_id_values_are_compared_in_diagnostic_mode_only(diagnostics):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    state = SimpleNamespace(
        device=torch.device("cuda", torch.cuda.current_device()), diagnostics=diagnostics
    )
    rows = [3, 4, 5]
    check = BeamFullModelNvfp4Sm103Tp4._check_gather_ids
    check(state, torch.tensor(rows, device="cuda"), rows)
    with pytest.raises(ValueError, match="gathers logits rows"):
        check(state, torch.tensor(rows[:2], device="cuda"), rows)
    with pytest.raises(ValueError, match="gathers logits rows"):
        check(state, torch.tensor(rows), rows)
    wrong = torch.tensor([3, 4, 6], device="cuda")
    if diagnostics:
        with pytest.raises(ValueError, match="need the rows"):
            check(state, wrong, rows)
    else:
        check(state, wrong, rows)


@pytest.mark.parametrize(
    "spans,expected",
    [
        ([(0, 40)], [39]),
        ([(0, 40), (64, 20), (100, 1), (7, 1)], [39, 59, 60, 61]),
        ([(5, 1), (9, 1), (2, 1)], [0, 1, 2]),
    ],
)
def test_default_rows_are_each_requests_last_row(spans, expected):
    select = BeamFullModelNvfp4Sm103Tp4._select_rows
    assert select(None, None, spans) == expected
    assert select(None, "all", spans) == list(range(sum(n for _, n in spans)))
