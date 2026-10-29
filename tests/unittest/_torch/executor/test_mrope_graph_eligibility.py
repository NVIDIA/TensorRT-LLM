# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.mapping import Mapping


def _request(slot: int | None, **changes: object) -> SimpleNamespace:
    fields = dict(
        py_seq_slot=slot,
        py_mrope_delta_cache_slot=None,
        py_mrope_position_delta=None,
        py_multimodal_data=None,
        py_draft_tokens=[],
        is_dummy=False,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def _generation_batch(*requests: SimpleNamespace) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.generation_requests = list(requests)
    return batch


def _captured_runner(batch: ScheduledRequests) -> tuple[CUDAGraphRunner, tuple]:
    config = CUDAGraphRunnerConfig(
        use_cuda_graph=False,
        cuda_graph_padding_enabled=False,
        cuda_graph_batch_sizes=[batch.batch_size],
        max_cuda_graph_batch_size=batch.batch_size,
        max_beam_width=1,
        max_num_tokens=batch.batch_size,
        spec_config=None,
        use_mrope=True,
        original_max_total_draft_tokens=0,
        enable_attention_dp=False,
        is_encoder_decoder=False,
        batch_size=batch.batch_size,
        mapping=Mapping(),
        dist=None,
    )
    # Selection needs recorded metadata, but neither device buffers nor capture.
    runner = CUDAGraphRunner(config)
    runner.enabled = True
    key = runner.get_graph_key(batch)
    metadata = object()
    runner.graph_metadata[key] = {"attn_metadata": metadata, "spec_metadata": None}
    return runner, (metadata, None, key)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "multimodal_data",
    [None, {}, {"mrope_config": {}}, {"mrope_config": {"mrope_position_deltas": None}}],
    ids=["absent", "empty", "empty-mrope", "no-delta"],
)
def test_text_requests_with_unseeded_slots_select_captured_graph(
    multimodal_data: dict | None,
) -> None:
    text = _request(0, py_multimodal_data=multimodal_data)
    seeded = _request(1, py_mrope_position_delta=torch.tensor([3]), py_mrope_delta_cache_slot=1)
    batch = _generation_batch(text, seeded)
    runner, captured = _captured_runner(batch)

    assert runner.maybe_get_cuda_graph(batch, False, None) == captured


@pytest.mark.cpu_only
@pytest.mark.parametrize("delta_source", ["direct", "multimodal"])
@pytest.mark.parametrize("delta", [0, 7])
@pytest.mark.parametrize(
    ("slot", "cached_slot", "is_dummy", "eligible"),
    [
        (1, None, False, False),
        (1, 0, False, False),
        (1, 1, False, True),
        (1, None, True, True),
        (None, None, False, True),
    ],
    ids=["unseeded", "reseated", "fresh", "dummy", "unassigned"],
)
def test_graph_selection_requires_only_real_stale_deltas(
    delta_source: str,
    delta: int,
    slot: int | None,
    cached_slot: int | None,
    is_dummy: bool,
    eligible: bool,
) -> None:
    request = _request(slot, py_mrope_delta_cache_slot=cached_slot, is_dummy=is_dummy)
    value = torch.tensor([delta])
    if delta_source == "direct":
        request.py_mrope_position_delta = value
    else:
        request.py_multimodal_data = {"mrope_config": {"mrope_position_deltas": value}}
    batch = _generation_batch(_request(0), request)
    runner, captured = _captured_runner(batch)

    selected = runner.maybe_get_cuda_graph(batch, False, None)

    assert selected == (captured if eligible else (None, None, None))


@pytest.mark.cpu_only
@pytest.mark.parametrize("guard", ["disabled", "context", "missing-capture"])
def test_text_requests_preserve_other_graph_guards(guard: str) -> None:
    batch = _generation_batch(_request(0))
    runner, _ = _captured_runner(batch)
    if guard == "disabled":
        runner.enabled = False
    elif guard == "context":
        batch.context_requests_last_chunk = batch.generation_requests
        batch.generation_requests = []
    else:
        runner.graph_metadata.clear()

    assert runner.maybe_get_cuda_graph(batch, False, None) == (None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mRoPE cache seeding writes CUDA tensors")
def test_cache_seeding_enables_graph_and_tracks_slot_reassignment() -> None:
    direct = _request(0, py_mrope_position_delta=torch.tensor([[0]], dtype=torch.int64))
    nested = _request(
        1,
        py_multimodal_data={
            "mrope_config": {
                "mrope_position_deltas": torch.tensor([[7.0]], device="cuda", dtype=torch.float64)
            }
        },
    )
    text = _request(2)
    dummy = _request(3, is_dummy=True, py_mrope_position_delta=torch.tensor([13]))
    unassigned = _request(None, py_mrope_position_delta=torch.tensor([17]))
    batch = _generation_batch(direct, nested, text, dummy, unassigned)
    runner, captured = _captured_runner(batch)
    cache = torch.full((batch.batch_size + 1,), -1, dtype=torch.int32, device="cuda")
    preparer = DecoderInputPreparer(
        buffers=None,
        config=SimpleNamespace(use_mrope=True, is_spec_decode=False, spec_config=None),
        mapping=Mapping(),
        dist=None,
        model=SimpleNamespace(mrope_position_deltas_cache=cache),
        get_runtime_tokens_per_gen_step=None,
        iter_states={},
        lora=None,
        use_beam_search=False,
        steady_gen_cache_supported=False,
    )

    assert runner.maybe_get_cuda_graph(batch, False, None) == (None, None, None)
    preparer.pad_batch_seed_mrope_delta_cache(batch)
    assert cache.cpu().tolist() == [0, 7, -1, -1, -1, -1]
    assert runner.maybe_get_cuda_graph(batch, False, None) == captured

    # Fresh slots keep their cached value until the request acquires a new slot.
    direct.py_mrope_position_delta.fill_(11)
    preparer.pad_batch_seed_mrope_delta_cache(batch)
    assert cache.cpu().tolist() == [0, 7, -1, -1, -1, -1]

    direct.py_seq_slot = 4
    assert runner.maybe_get_cuda_graph(batch, False, None) == (None, None, None)
    preparer.pad_batch_seed_mrope_delta_cache(batch)
    assert cache.cpu().tolist() == [0, 7, -1, -1, 11, -1]
    assert runner.maybe_get_cuda_graph(batch, False, None) == captured
