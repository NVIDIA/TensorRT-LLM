# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import weakref
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import KVCacheParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.beam_compaction import BeamCompaction
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.workspace import EagerWorkspaceReclaimer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def make_metadata(batch, capacity):
    count = batch * capacity
    metadata = TrtllmAttentionMetadata(
        max_num_requests=batch,
        max_num_tokens=count,
        max_num_sequences=count,
        seq_lens=torch.ones(count, dtype=torch.int32),
        num_contexts=0,
        beam_width=capacity,
        max_seq_len=64,
    )
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[4] * count)
    metadata.request_ids = list(range(batch))
    metadata.prompt_lens = [3] * count
    metadata.kv_lens.fill_(5)
    metadata.kv_lens_cuda.fill_(5)
    metadata.prompt_lens_cpu.fill_(3)
    metadata.prompt_lens_cuda.fill_(3)
    metadata.host_request_types.fill_(1)
    metadata.kv_cache_block_offsets = torch.arange(
        count * 8, dtype=torch.int32, device="cuda"
    ).reshape(1, count, 2, 4)
    metadata.cache_indirection = torch.zeros(batch, capacity, 64, dtype=torch.int32, device="cuda")
    return metadata


def make_inputs(batch, capacity):
    ids = torch.arange(batch * capacity, device="cuda", dtype=torch.int32)
    return dict(
        input_ids=ids, position_ids=ids[None, :] + 10, attn_metadata=make_metadata(batch, capacity)
    )


def fake_forward(**inputs):
    values = inputs["input_ids"].float() + inputs["position_ids"][0]
    values += inputs["attn_metadata"].kv_lens_cuda[: values.numel()]
    return torch.stack((values, values + 1), dim=-1)


def forward(inputs):
    runner = SimpleNamespace(_model_caller=fake_forward, _eager_workspace_reclaimer=None)
    return DecoderRunner.model_forward(runner, is_dummy=False, **inputs)


def request(width, next_width=None, dummy=False):
    return SimpleNamespace(
        is_dummy=dummy,
        get_beam_width_by_iter=lambda for_next_iteration=False: (
            next_width if for_next_iteration and next_width is not None else width
        ),
    )


@pytest.mark.parametrize("previous,expected", [(False, 2), (True, 4)])
def test_width_tracks_input_iteration_and_ignores_padding(previous, expected):
    batch = SimpleNamespace(
        context_requests=[],
        generation_requests=[request(2, 4), request(2, 4), request(5, dummy=True)],
    )
    assert BeamCompaction(5, (2, 4)).execution_width(batch, previous) == expected


@pytest.mark.parametrize(
    "requests,contexts",
    [
        ([request(3)], []),
        ([request(2), request(4)], []),
        ([request(5, dummy=True)], []),
        ([request(2)], [object()]),
    ],
)
def test_unsupported_batches_use_capacity(requests, contexts):
    batch = SimpleNamespace(context_requests=contexts, generation_requests=requests)
    assert BeamCompaction(5, (2, 4)).execution_width(batch, False) == 0


@pytest.mark.parametrize(
    "batch,capacity,width", [(1, 5, 2), (2, 5, 4), (2, 500, 20), (2, 500, 300)]
)
@pytest.mark.parametrize("graph_enabled", [False, True])
def test_active_rows_and_repeated_metadata_updates(batch, capacity, width, graph_enabled):
    inputs = make_inputs(batch, capacity)
    source = inputs["attn_metadata"]
    compaction = BeamCompaction(capacity, (width,))
    compaction.prepare_inputs(inputs, width)
    view = inputs["beam_compaction"]
    pointer = view.metadata.kv_lens_cuda.data_ptr()
    if graph_enabled:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward(inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = forward(inputs)
    for step in range(3):
        inputs["input_ids"].add_(1)
        inputs["position_ids"].add_(2)
        source.kv_lens.add_(1)
        source.kv_lens_cuda.add_(1)
        source.kv_cache_block_offsets.add_(8)
        # Parent IDs remain request-local; their numeric values are not remapped.
        source.cache_indirection.fill_(step % width)
        compaction.prepare_inputs(inputs, width)
        assert inputs["beam_compaction"] is view
        assert view.metadata.kv_lens_cuda.data_ptr() == pointer
        if graph_enabled:
            graph.replay()
        else:
            result = forward(inputs)
        expected = fake_forward(**inputs).reshape(batch, capacity, 2)
        actual = result.reshape(batch, capacity, 2)
        torch.testing.assert_close(actual[:, :width], expected[:, :width], rtol=0, atol=0)
        assert torch.isneginf(actual[:, width:]).all()
        torch.testing.assert_close(
            view.metadata.kv_cache_block_offsets,
            source.kv_cache_block_offsets.index_select(1, view.row_indices),
        )
        torch.testing.assert_close(
            view.metadata.cache_indirection, source.cache_indirection[:, :width]
        )


def test_metadata_views_do_not_share_plans_or_cross_batch_rows():
    inputs = make_inputs(2, 5)
    compaction = BeamCompaction(5, (2, 4))
    compaction.prepare_inputs(inputs, 2)
    narrow = inputs["beam_compaction"]
    compaction.prepare_inputs(inputs, 4)
    wide = inputs["beam_compaction"]
    assert narrow.row_indices_list == [0, 1, 5, 6]
    assert wide.row_indices_list == [0, 1, 2, 3, 5, 6, 7, 8]
    assert narrow.metadata.fmha_plan_caches is not wide.metadata.fmha_plan_caches
    compaction.prepare_inputs(inputs, 0)
    assert "beam_compaction" not in inputs
    compaction.clear()
    assert not compaction._views


def test_multitoken_and_lora_inputs_use_capacity():
    inputs = make_inputs(1, 5)
    compaction = BeamCompaction(5, (2,))
    inputs["lora_params"] = object()
    compaction.prepare_inputs(inputs, 2)
    assert "beam_compaction" not in inputs
    inputs.pop("lora_params")
    inputs["attn_metadata"].seq_lens = torch.full((5,), 2, dtype=torch.int32)
    compaction.prepare_inputs(inputs, 2)
    assert "beam_compaction" not in inputs


def test_workspace_reclaimer_keeps_source_identity():
    inputs = make_inputs(1, 5)
    source = inputs["attn_metadata"]
    source.workspace = torch.empty(64, dtype=torch.uint8, device="cuda")
    reclaimer = EagerWorkspaceReclaimer(source)
    old_workspace = weakref.ref(source.workspace)
    source.workspace.resize_(1024)
    compaction = BeamCompaction(5, (2,))
    compaction.prepare_inputs(inputs, 2)
    view = inputs["beam_compaction"]

    def model(**kwargs):
        assert kwargs["attn_metadata"].workspace is source.workspace
        source.workspace.resize_(max(source.workspace.numel(), 64))
        return fake_forward(**kwargs)

    runner = SimpleNamespace(_model_caller=model, _eager_workspace_reclaimer=reclaimer)
    for _ in range(5):
        DecoderRunner.model_forward(runner, is_dummy=False, **inputs)
        assert view.metadata.workspace is None
    assert source.workspace.untyped_storage().nbytes() == 64
    assert old_workspace() is None


def test_graph_runner_captures_widths_and_replays_updated_rows(monkeypatch):
    from tensorrt_llm._torch.pyexecutor import cuda_graph_runner as graph_module
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
        CUDAGraphRunner,
        CUDAGraphRunnerConfig,
        KeyType,
    )
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setattr(graph_module, "_STRICT_BUFFER_CHECK", True)
    config = CUDAGraphRunnerConfig(
        use_cuda_graph=True,
        cuda_graph_padding_enabled=True,
        cuda_graph_batch_sizes=[2],
        max_cuda_graph_batch_size=2,
        max_beam_width=5,
        max_num_tokens=10,
        spec_config=None,
        cuda_graph_mem_pool=None,
        use_mrope=False,
        original_max_draft_len=0,
        original_max_total_draft_tokens=0,
        enable_attention_dp=False,
        is_encoder_decoder=False,
        batch_size=2,
        mapping=Mapping(),
        dist=None,
        kv_cache_manager_key=None,
        compact_beam_widths=(2, 4),
    )
    runner = CUDAGraphRunner(config)
    inputs = make_inputs(2, 5)
    inputs["attn_metadata"].is_cuda_graph = True
    key = KeyType(2, 0, False)
    runner.is_warmup_only = True
    runner.capture(key, forward, inputs)
    runner.is_warmup_only = False
    runner.capture(key, forward, inputs)
    assert {k.beam_execution_width for k in runner.graphs} == {0, 2, 4}
    eager_inputs = make_inputs(2, 5)
    eager_ids, eager_positions = eager_inputs["input_ids"], eager_inputs["position_ids"]
    for step, width in enumerate((2, 4, 0, 2)):
        eager_count = (1 + step % 2) * 5
        eager_inputs["attn_metadata"].seq_lens = torch.ones(eager_count, dtype=torch.int32)
        eager_inputs["input_ids"] = eager_ids[:eager_count]
        eager_inputs["position_ids"] = eager_positions[:, :eager_count]
        runner.beam_compaction.prepare_inputs(eager_inputs, 2)
        inputs["input_ids"].add_(7)
        inputs["position_ids"].add_(1)
        inputs["attn_metadata"].kv_lens_cuda.add_(3)
        runner.beam_compaction.prepare_inputs(inputs, width)
        result = runner.replay(key._replace(beam_execution_width=width), inputs)
        expected = fake_forward(**inputs).reshape(2, 5, 2)
        actual = result.reshape(2, 5, 2)
        active = width or 5
        torch.testing.assert_close(actual[:, :active], expected[:, :active], rtol=0, atol=0)
        assert torch.isneginf(actual[:, active:]).all()
    inputs.pop("beam_compaction")
    with pytest.raises(RuntimeError, match="refreshed compact metadata"):
        runner.replay(key._replace(beam_execution_width=2), inputs)


@pytest.mark.parametrize("widths,expected", [([], []), ([4, 2, 2, 5], [2, 4])])
def test_config_normalizes_execution_widths(widths, expected):
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    args = TorchLlmArgs.model_construct(max_beam_width=5, compact_beam_widths=widths)
    assert args.validate_compact_beam_widths().compact_beam_widths == expected


def test_config_rejects_width_above_capacity():
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    args = TorchLlmArgs.model_construct(max_beam_width=5, compact_beam_widths=[6])
    with pytest.raises(ValueError, match="max_beam_width"):
        args.validate_compact_beam_widths()


@pytest.mark.parametrize("width", [0, -1])
def test_config_width_schema_requires_positive_integers(width):
    from pydantic import TypeAdapter, ValidationError

    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    schema = TypeAdapter(TorchLlmArgs.model_fields["compact_beam_widths"].annotation)
    with pytest.raises(ValidationError):
        schema.validate_python([width])


def test_config_keeps_width_one_on_capacity_path():
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    args = TorchLlmArgs.model_construct(max_beam_width=5, compact_beam_widths=[1])
    with pytest.raises(ValueError, match="at least 2"):
        args.validate_compact_beam_widths()
    batch = SimpleNamespace(context_requests=[], generation_requests=[request(1)])
    assert BeamCompaction(5, (2, 4)).execution_width(batch, False) == 0


def test_metadata_host_indices_ignore_default_device():
    inputs = make_inputs(2, 5)
    compaction = BeamCompaction(5, (2,))
    with torch.device("cuda"):
        compaction.prepare_inputs(inputs, 2)
    view = inputs["beam_compaction"]
    assert view.row_indices_cpu.device.type == "cpu"
    assert view.metadata.seq_lens.device.type == "cpu"
    assert view.row_indices.is_cuda


@pytest.mark.parametrize("capacity,width", [(5, 2), (5, 4), (500, 20), (500, 300)])
def test_eager_cache_releases_previous_batch_views(capacity: int, width: int) -> None:
    """Changing batch size releases obsolete views and keeps active rows correct."""
    inputs = make_inputs(3, capacity)
    source = inputs["attn_metadata"]
    input_ids, position_ids = inputs["input_ids"], inputs["position_ids"]
    compaction = BeamCompaction(capacity, (width,))
    previous_view = previous_buffer = None
    for batch in (1, 2, 3, 1, 3, 2):
        count = batch * capacity
        source.seq_lens = torch.ones(count, dtype=torch.int32)
        source.request_ids = list(range(batch))
        source.kv_lens.add_(1)
        source.kv_lens_cuda.add_(1)
        inputs["input_ids"] = input_ids[:count]
        inputs["position_ids"] = position_ids[:, :count]
        compaction.prepare_inputs(inputs, width)
        view = inputs["beam_compaction"]
        if previous_view is not None:
            assert previous_view() is None
            assert previous_buffer() is None
        assert len(compaction._views) == 1
        assert view.batch_size == batch
        pointer = view.metadata.kv_lens_cuda.data_ptr()
        compaction.prepare_inputs(inputs, width)
        assert inputs["beam_compaction"] is view
        assert view.metadata.kv_lens_cuda.data_ptr() == pointer
        actual = forward(inputs).reshape(batch, capacity, 2)
        expected = fake_forward(**inputs).reshape(batch, capacity, 2)
        torch.testing.assert_close(actual[:, :width], expected[:, :width], rtol=0, atol=0)
        assert torch.isneginf(actual[:, width:]).all()
        previous_view = weakref.ref(view)
        previous_buffer = weakref.ref(view.metadata.kv_lens_cuda)
        del view


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_eager_cache_keeps_other_sources_and_widths(graph_enabled: bool) -> None:
    """Eviction is isolated from other widths and eager or CUDA graph sources."""
    inputs = make_inputs(3, 5)
    other_inputs = make_inputs(2, 5)
    other_inputs["attn_metadata"].is_cuda_graph = graph_enabled
    compaction = BeamCompaction(5, (2, 4))
    compaction.prepare_inputs(inputs, 4)
    wide = inputs["beam_compaction"]
    compaction.prepare_inputs(other_inputs, 2)
    other = other_inputs["beam_compaction"]
    other_pointer = other.metadata.kv_lens_cuda.data_ptr()
    compaction.prepare_inputs(inputs, 2)
    inputs["attn_metadata"].seq_lens = torch.ones(5, dtype=torch.int32)
    inputs["input_ids"] = inputs["input_ids"][:5]
    inputs["position_ids"] = inputs["position_ids"][:, :5]
    compaction.prepare_inputs(inputs, 2)
    assert len(compaction._views) == 3
    assert wide in compaction._views.values()
    assert other in compaction._views.values()
    compaction.prepare_inputs(other_inputs, 2)
    assert other_inputs["beam_compaction"] is other
    assert other.metadata.kv_lens_cuda.data_ptr() == other_pointer
