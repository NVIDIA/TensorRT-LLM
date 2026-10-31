# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused CPU tests for DSpark decoder-runner ragged runtime state."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch import utils as torch_utils
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, DSparkCUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.engine import metadata as metadata_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as graphs_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import inputs as decoder_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import (
    metadata as decoder_metadata_module,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import DecoderMetadata
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm._torch.speculative.ragged_helpers import ragged_gather_index_lists


def _runner(*, secondary=True):
    return SimpleNamespace(
        enabled=True,
        agreed_ragged_bucket=None,
        ragged_pad_verify_len=0,
        ragged_zero_real_high_rows=0,
        supported_batch_sizes=[4, 16],
        secondary_padding_dummy_requests={5: object()} if secondary else {},
        _round_up_batch_size=lambda rows: 4 if rows <= 4 else 16,
        will_pad_to=lambda *_args: True,
    )


def _engine(runner, buckets):
    engine = object.__new__(DecoderInputPreparer)
    engine.cuda_graph_runner = runner
    engine._config = SimpleNamespace(
        spec_config=SimpleNamespace(max_draft_len=5),
        max_batch_size=8,
        cuda_graph_batch_sizes=[1, 4, 8],
        use_mrope=False,
    )
    engine._dspark_confidence_enabled = True
    engine._dspark_trims_submitted_tokens = True
    engine._dspark_device_windows = False
    engine._dspark_last_padded_bs = None
    engine.ragged_verify_token_buckets = lambda _rows: list(buckets)
    return engine


def test_pinned_host_ping_pongs_without_private_cuda_events() -> None:
    engine = SimpleNamespace(
        _pinned_host_cache={},
        _pinned_host_active={},
    )

    first = DecoderInputPreparer._pinned_host(engine, "rows", [1, 2], torch.long)
    second = DecoderInputPreparer._pinned_host(engine, "rows", [3, 4], torch.long)
    third = DecoderInputPreparer._pinned_host(engine, "rows", [5, 6], torch.long)

    assert first.data_ptr() == third.data_ptr()
    assert first.data_ptr() != second.data_ptr()
    assert third.tolist() == [5, 6]


@pytest.mark.parametrize(
    ("verifier_budget", "scheduled_window", "high_rows"),
    [(80, 4, 0), (81, 5, 1), (88, 5, 8)],
)
def test_zero_real_exact_fit_uses_one_scheduled_dummy(verifier_budget, scheduled_window, high_rows):
    runner = _runner()
    engine = _engine(runner, [48, 64, 80, 81, 88, 96])
    dummy = SimpleNamespace(is_attention_dp_dummy=True, py_verify_len=None)

    bucket = DecoderInputPreparer.fit_ragged_verify_lens(
        engine,
        [dummy],
        [scheduled_window],
        peer_stats=[[16, 88, 1], [0, 0, 1]],
        exact_shape=(16, verifier_budget, 5),
        exact_zero_real=True,
    )

    assert bucket == verifier_budget
    assert dummy.py_verify_len == scheduled_window
    assert engine._dspark_last_num_real == 0
    assert runner.ragged_pad_verify_len == 4
    assert runner.ragged_zero_real_high_rows == high_rows


def test_zero_real_exact_fit_requires_the_secondary_dummy():
    runner = _runner(secondary=False)
    engine = _engine(runner, [88])
    dummy = SimpleNamespace(is_attention_dp_dummy=True, py_verify_len=None)

    with pytest.raises(RuntimeError, match="disappeared after"):
        DecoderInputPreparer.fit_ragged_verify_lens(
            engine,
            [dummy],
            [5],
            peer_stats=[[16, 88, 1], [0, 0, 1]],
            exact_shape=(16, 88, 5),
            exact_zero_real=True,
        )


def test_exact_fit_publishes_the_measured_bucket_and_pad_window():
    runner = _runner()
    engine = _engine(runner, [8, 24])
    requests = [SimpleNamespace(py_verify_len=None) for _ in range(2)]

    bucket = DecoderInputPreparer.fit_ragged_verify_lens(
        engine,
        requests,
        [2, 2],
        peer_stats=[[2, 6, 1]],
        exact_shape=(4, 8, 1),
    )

    assert bucket == 8
    assert [request.py_verify_len for request in requests] == [2, 2]
    assert runner.agreed_ragged_bucket == 8
    assert runner.ragged_pad_verify_len == 0


def test_full_k_bucket_preserves_the_native_static_graph():
    runner = _runner()
    engine = _engine(runner, [24])
    requests = [SimpleNamespace(py_verify_len=None) for _ in range(2)]

    bucket = DecoderInputPreparer.fit_ragged_verify_lens(
        engine,
        requests,
        [5, 5],
        peer_stats=[[2, 12, 1]],
        exact_shape=(4, 24, 6),
    )

    assert bucket is None
    assert runner.agreed_ragged_bucket is None
    assert all(request.py_verify_len is None for request in requests)


def _warmup_metadata(position_helper, mask_helper):
    ratios = [1, 4]
    return SimpleNamespace(
        _compress_ratios_sorted=ratios,
        max_draft_tokens=5,
        past_kv_lens_cuda={ratio: torch.empty(8, dtype=torch.int32) for ratio in ratios},
        cu_new_comp_kv_cuda={ratio: torch.empty(9, dtype=torch.int32) for ratio in ratios},
        new_comp_kv_lens_cuda={ratio: torch.empty(8, dtype=torch.int32) for ratio in ratios},
        compressed_position_ids_cuda={
            ratio: torch.empty(48, dtype=torch.int32) for ratio in ratios
        },
        compressed_mask_cuda={ratio: torch.empty(48, dtype=torch.bool) for ratio in ratios},
        batch_size_cuda=torch.tensor([17], dtype=torch.int32),
        _compute_gen_compressed_position_ids=position_helper,
        _compute_compressed_mask=mask_helper,
    )


def test_ragged_compressor_warmup_primes_and_cleans_intermediate_shape(monkeypatch):
    position_calls = []
    mask_calls = []

    def record_positions(*args):
        position_calls.append((torch.is_inference_mode_enabled(), args[3:]))

    def record_mask(new_comp, cu_new_comp, compressed_mask, batch_size, compress_ratios):
        mask_calls.append((torch.is_inference_mode_enabled(), batch_size.tolist(), compress_ratios))

    metadata = _warmup_metadata(record_positions, record_mask)
    engine = SimpleNamespace(
        _ragged_verification_enabled=True,
        attn_metadata=metadata,
        _config=SimpleNamespace(max_batch_size=8, cuda_graph_batch_sizes=[1, 4, 8]),
    )
    sync_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))

    DecoderMetadata._warmup_dspark_ragged_compressor_metadata(engine)

    assert position_calls == [(True, (0, 7, 6, [1, 4], {1: 0, 4: 0}))]
    assert mask_calls == [(True, [7], [1, 4])]
    assert metadata.batch_size_cuda.tolist() == [17]
    for ratio, token_count in ((1, 42), (4, 14)):
        assert not metadata.past_kv_lens_cuda[ratio][:7].any()
        assert not metadata.new_comp_kv_lens_cuda[ratio][:7].any()
        assert not metadata.cu_new_comp_kv_cuda[ratio][:8].any()
        assert not metadata.compressed_position_ids_cuda[ratio][:token_count].any()
        assert not metadata.compressed_mask_cuda[ratio][:token_count].any()
    assert sync_calls == [True, True]


def test_ragged_compressor_warmup_cleans_after_helper_failure(monkeypatch):
    def fail_positions(*args):
        args[2][1][:18].fill_(7)
        raise RuntimeError("compile boom")

    metadata = _warmup_metadata(fail_positions, lambda *_args: None)
    engine = SimpleNamespace(
        _ragged_verification_enabled=True,
        attn_metadata=metadata,
        _config=SimpleNamespace(max_batch_size=4, cuda_graph_batch_sizes=[1, 4]),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="compile boom"):
        DecoderMetadata._warmup_dspark_ragged_compressor_metadata(engine)

    assert not metadata.past_kv_lens_cuda[1][:3].any()
    assert not metadata.new_comp_kv_lens_cuda[1][:3].any()
    assert not metadata.cu_new_comp_kv_cuda[1][:4].any()
    assert not metadata.compressed_position_ids_cuda[1][:18].any()
    assert not metadata.compressed_mask_cuda[1][:18].any()
    assert metadata.batch_size_cuda.tolist() == [17]


@pytest.mark.parametrize("fail_after_mask", [False, True])
def test_ragged_compressor_warmup_calls_current_metadata_helpers(monkeypatch, fail_after_mask):
    """Run the real five-argument mask and position helper bodies on CPU tensors."""
    actual_mask = DeepseekV4TrtllmAttentionMetadata._compute_compressed_mask

    def mask(new_comp, cu_new_comp, compressed_mask, batch_size, compress_ratios):
        actual_mask(new_comp, cu_new_comp, compressed_mask, batch_size, compress_ratios)
        assert batch_size is metadata.batch_size_cuda
        assert batch_size.tolist() == [7]
        assert compressed_mask[1][:42].all()
        assert compressed_mask[4][:14].all()
        assert metadata.compressed_position_ids_cuda[1][:42].tolist() == list(range(6)) * 7
        assert metadata.compressed_position_ids_cuda[4][:14].tolist() == [0, 4] * 7
        if fail_after_mask:
            raise RuntimeError("mask cleanup probe")

    metadata = _warmup_metadata(
        DeepseekV4TrtllmAttentionMetadata._compute_gen_compressed_position_ids, mask
    )
    original_storage = metadata.batch_size_cuda.data_ptr()
    engine = SimpleNamespace(
        _ragged_verification_enabled=True,
        attn_metadata=metadata,
        _config=SimpleNamespace(max_batch_size=8, cuda_graph_batch_sizes=[1, 4, 8]),
    )
    monkeypatch.setattr(torch_utils, "is_piecewise_running", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    if fail_after_mask:
        with pytest.raises(RuntimeError, match="mask cleanup probe"):
            DecoderMetadata._warmup_dspark_ragged_compressor_metadata(engine)
    else:
        DecoderMetadata._warmup_dspark_ragged_compressor_metadata(engine)
    assert metadata.batch_size_cuda.data_ptr() == original_storage
    assert metadata.batch_size_cuda.tolist() == [17]
    for ratio, token_count in ((1, 42), (4, 14)):
        assert not metadata.past_kv_lens_cuda[ratio][:7].any()
        assert not metadata.new_comp_kv_lens_cuda[ratio][:7].any()
        assert not metadata.cu_new_comp_kv_cuda[ratio][:8].any()
        assert not metadata.compressed_position_ids_cuda[ratio][:token_count].any()
        assert not metadata.compressed_mask_cuda[ratio][:token_count].any()


def test_ragged_compressor_warmup_requires_trimmed_tokens():
    engine = SimpleNamespace(
        _ragged_verification_enabled=False,
    )

    DecoderMetadata._warmup_dspark_ragged_compressor_metadata(engine)


def test_ragged_compressor_warmup_rejects_missing_metadata_contract():
    engine = SimpleNamespace(
        _ragged_verification_enabled=True,
        attn_metadata=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="missing .*max_draft_tokens"):
        DecoderMetadata._warmup_dspark_ragged_compressor_metadata(engine)


@pytest.mark.parametrize("runtime_draft_len", [0, 2, 5])
def test_publish_gen_token_layout_uses_call_draft_length(runtime_draft_len: int) -> None:
    engine = _engine(_runner(), [])
    engine.get_runtime_tokens_per_gen_step = lambda draft_len: draft_len + 1
    requests = [
        SimpleNamespace(py_verify_len=min(1, runtime_draft_len)),
        SimpleNamespace(py_verify_len=min(3, runtime_draft_len)),
    ]
    metadata = SimpleNamespace(
        runtime_tokens_per_gen_step=0,
        ragged_verify_lens=None,
        device_windows_mode=False,
    )
    assert not hasattr(engine, "runtime_draft_len")

    engine._publish_gen_token_layout(metadata, requests, runtime_draft_len=runtime_draft_len)

    assert metadata.runtime_tokens_per_gen_step == runtime_draft_len + 1
    assert metadata.ragged_verify_lens == [request.py_verify_len + 1 for request in requests]
    assert not metadata.device_windows_mode
    assert not hasattr(engine, "runtime_draft_len")


def test_attach_ragged_layout_refreshes_persistent_buffers_in_place() -> None:
    runner = _engine(_runner(), [])
    runner._pinned_host_cache = {}
    runner._pinned_host_active = {}
    runner._buffers = SimpleNamespace()
    runner._buffers.ragged_verify_lens_cuda = torch.empty(9, dtype=torch.int32)
    runner._buffers.ragged_qo_indptr_cuda = torch.empty(10, dtype=torch.int32)
    attention = SimpleNamespace(ragged_verify_lens=None)
    speculation = SimpleNamespace()
    requests = [SimpleNamespace(py_verify_len=1), SimpleNamespace(py_verify_len=3)]

    runner._attach_ragged_verify_layout(speculation, attention, requests)
    first_lens_ptr = speculation.verify_lens.data_ptr()
    first_prefix_ptr = speculation.qo_indptr.data_ptr()

    assert speculation.verify_lens.tolist() == [2, 4]
    assert speculation.qo_indptr.tolist() == [0, 2, 6]
    assert speculation.total_verify_tokens == 6
    assert attention.ragged_verify_lens == [2, 4]

    requests[0].py_verify_len = 0
    requests[1].py_verify_len = 2
    runner._attach_ragged_verify_layout(speculation, attention, requests)

    assert speculation.verify_lens.tolist() == [1, 3]
    assert speculation.qo_indptr.tolist() == [0, 1, 4]
    assert speculation.total_verify_tokens == 4
    assert attention.ragged_verify_lens == [1, 3]
    assert speculation.verify_lens.data_ptr() == first_lens_ptr
    assert speculation.qo_indptr.data_ptr() == first_prefix_ptr


@pytest.mark.parametrize("trim_enabled", [False, True])
def test_missing_or_disabled_windows_clear_stale_ragged_views(trim_enabled: bool) -> None:
    runner = _engine(_runner(), [])
    runner._dspark_trims_submitted_tokens = trim_enabled
    attention = SimpleNamespace(ragged_verify_lens=[2, 4])
    speculation = SimpleNamespace(
        verify_lens=torch.tensor([2, 4]),
        qo_indptr=torch.tensor([0, 2, 6]),
        total_verify_tokens=6,
    )

    runner._attach_ragged_verify_layout(
        speculation,
        attention,
        [SimpleNamespace(py_verify_len=1), SimpleNamespace(py_verify_len=None)],
    )

    assert speculation.verify_lens is None
    assert speculation.qo_indptr is None
    assert speculation.total_verify_tokens is None
    assert attention.ragged_verify_lens is None


def test_model_engine_delegates_ragged_state_to_decoder_owner() -> None:
    inputs = _engine(_runner(), [])
    runner = object.__new__(DecoderRunner)
    runner._inputs = inputs
    inputs._dspark_sps_cost_table = object()
    inputs._dspark_exact_candidate_cells = ((4, 12),)
    inputs._dspark_exact_identity_words = tuple(range(8))
    inputs.fit_ragged_verify_lens = lambda *args, **kwargs: (args, kwargs)
    worker = object()
    runner._model = SimpleNamespace(spec_worker=worker)
    engine = object.__new__(ModelEngine)
    engine._cleanup_done = True
    engine._runner = runner

    assert engine._dspark_confidence_enabled
    assert engine._dspark_trims_submitted_tokens
    assert engine._dspark_sps_cost_table is runner._dspark_sps_cost_table
    assert engine._dspark_exact_candidate_cells == ((4, 12),)
    assert engine._dspark_exact_identity_words == tuple(range(8))
    assert engine._dspark_device_budget is None
    assert engine._get_spec_worker() is worker
    engine._dspark_device_budget = 7
    assert runner._dspark_device_budget == 7
    assert engine.fit_ragged_verify_lens([1], exact_shape=(4, 12, 3)) == (
        ([1],),
        {"exact_shape": (4, 12, 3)},
    )


def test_model_engine_non_decoder_keeps_confidence_inactive() -> None:
    engine = object.__new__(ModelEngine)
    engine._cleanup_done = True
    engine._runner = SimpleNamespace()

    assert not engine._dspark_confidence_enabled
    assert not engine._dspark_trims_submitted_tokens
    assert engine._dspark_sps_cost_table is None
    assert engine._dspark_exact_candidate_cells == ()
    assert engine._dspark_exact_identity_words == (0,) * 8
    assert engine._dspark_device_budget is None
    assert engine._get_spec_worker() is None


def _metadata_inputs(metadata_type):
    model = SimpleNamespace(
        pretrained_config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=1),
        sparse_attention_config=None,
        enable_flash_mla=False,
    )
    kwargs = dict(
        max_batch_size=2,
        max_num_tokens=8,
        max_beam_width=1,
        attention_backend=SimpleNamespace(Metadata=metadata_type),
        attention_runtime_features=SimpleNamespace(cache_reuse=False, chunked_prefill=False),
        mapping=object(),
        cache_indirection=None,
        enable_context_mla_with_cached_kv=False,
    )
    return model, kwargs


@pytest.mark.parametrize("explicit_false", [False, True])
def test_metadata_helper_keeps_native_constructor_kwargs(explicit_false):
    seen = []

    class NativeMetadata:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    model, kwargs = _metadata_inputs(NativeMetadata)
    if explicit_false:
        kwargs["enable_ragged_verification"] = False
    metadata_module.build_attention_metadata(model, **kwargs)

    assert "enable_ragged_verification" not in seen[0]
    assert seen[0]["max_num_heads_per_kv"] == 4


def test_metadata_helper_enables_only_supported_ragged_metadata():
    class RaggedMetadata(DSAtrtllmAttentionMetadata):
        def __init__(self, **kwargs):
            self.constructor_flag = kwargs.pop("enable_ragged_verification", False)
            self.enable_ragged_verification = self.constructor_flag
            self.ragged_verify_lens = [1, 2]

    model, kwargs = _metadata_inputs(RaggedMetadata)
    result = metadata_module.build_attention_metadata(
        model, enable_ragged_verification=True, **kwargs
    )
    assert result.constructor_flag is True
    result._validate_ragged_verification()

    kwargs["attention_backend"] = SimpleNamespace(Metadata=object)
    with pytest.raises(ValueError, match="DSA target metadata"):
        metadata_module.build_attention_metadata(model, enable_ragged_verification=True, **kwargs)


@pytest.mark.parametrize("enabled", [False, True])
def test_decoder_enables_ragged_metadata_during_construction(monkeypatch, enabled):
    seen = []

    class MetadataProbe(DSAtrtllmAttentionMetadata):
        def __init__(self, **kwargs):
            self.enable_ragged_verification = kwargs.pop("enable_ragged_verification", False)
            self.kv_cache_manager = kwargs["kv_cache_manager"]
            self.ragged_verify_lens = None
            seen.append(self.enable_ragged_verification)

    model, kwargs = _metadata_inputs(MetadataProbe)
    decoder = object.__new__(DecoderMetadata)
    decoder.attn_metadata = None
    decoder.model = SimpleNamespace(model_config=model)
    decoder._config = SimpleNamespace(
        **kwargs, spec_config=SimpleNamespace(enable_confidence_scheduling=enabled)
    )
    decoder.mapping = kwargs["mapping"]
    decoder._attention_runtime_features = kwargs["attention_runtime_features"]
    decoder._buffers = SimpleNamespace(cache_indirection=None)
    monkeypatch.setattr(decoder_metadata_module, "resolve_mamba_metadata_cls", lambda _model: None)
    kv = object()

    result = decoder.setup_attention(kv)
    if enabled:
        result.ragged_verify_lens = [1, 2]
    result._validate_ragged_verification()

    assert seen == [enabled]
    assert decoder.setup_attention(kv) is result
    assert seen == [enabled]


@pytest.mark.parametrize("graph_type", [CUDAGraphRunner, DSparkCUDAGraphRunner])
def test_current_main_warmup_keeps_serving_timeout_only_for_capture(monkeypatch, graph_type):
    graph = graph_type.__new__(graph_type)
    graph.clear = lambda **_kwargs: None
    graph._capture_allowed = False
    graph.padding_dummy_requests = {5: object()}
    if graph_type is DSparkCUDAGraphRunner:
        graph.secondary_padding_dummy_requests = {5: object()}
        graph.ragged_zero_real_high_rows = 2
        graph.adp_shape_agreement = object()
    serving_phase = False
    calls = []

    @contextmanager
    def serving_timeouts():
        nonlocal serving_phase
        serving_phase = True
        try:
            yield
        finally:
            serving_phase = False

    def warmup(_resources):
        calls.append((graph.is_warmup_only, serving_phase, bool(graph.padding_dummy_requests)))
        assert graph._capture_allowed
        if not graph.is_warmup_only and graph_type is DSparkCUDAGraphRunner:
            assert graph.secondary_padding_dummy_requests == {}
            assert graph.ragged_zero_real_high_rows == 0
            assert graph.adp_shape_agreement is None

    monkeypatch.setattr(graphs_module, "moe_comm_serving_timeouts", serving_timeouts)
    decoder = object.__new__(DecoderCudaGraphs)
    decoder.cuda_graph_runner = graph
    decoder.run_cuda_graph_warmup = warmup
    decoder.warmup_and_capture(object())

    assert calls == [(True, False, True), (False, True, False)]
    assert not graph._capture_allowed
    assert not graph.is_warmup_only


@pytest.mark.parametrize("worker", [None, SimpleNamespace(verify_planner=None)])
def test_device_window_prologue_declines_an_unavailable_planner(
    worker: SimpleNamespace | None,
) -> None:
    decoder = object.__new__(DecoderInputPreparer)
    decoder.model = SimpleNamespace(spec_worker=worker)
    decoder.cuda_graph_runner = SimpleNamespace(agreed_ragged_bucket=10)
    decoder._dspark_device_budget = 10
    decoder._dspark_prev_covers_batch = True

    assert not decoder._apply_device_window_prologue({}, None)
    assert decoder._dspark_device_budget is None


def test_confidence_without_an_sps_table_is_eager_only() -> None:
    decoder = object.__new__(DecoderInputPreparer)
    decoder.cuda_graph_runner = SimpleNamespace(enabled=False)
    decoder._dspark_confidence_enabled = True
    decoder._dspark_trims_submitted_tokens = True
    decoder._dspark_sps_cost_table = None
    decoder._config = SimpleNamespace(
        spec_config=SimpleNamespace(draft_len_schedule=None, max_draft_len=3)
    )

    graphs = object.__new__(DecoderCudaGraphs)
    graphs._inputs = decoder
    graphs._config = decoder._config
    graphs.cuda_graph_runner = decoder.cuda_graph_runner
    graphs._capture_generation_cuda_graphs(None)
    with pytest.raises(RuntimeError, match="authenticated exact SPS table"):
        graphs._get_graphs_to_capture([1, 4])


@pytest.mark.parametrize("use_fused_gather", [False, True])
@pytest.mark.parametrize(
    ("trims", "prefix_window", "previous_windows"),
    [(True, 1, [3, 3]), (True, 1, [1, 2]), (False, None, [None, None])],
)
def test_overlap_offsets_follow_the_packed_prefix(
    trims: bool,
    prefix_window: int | None,
    previous_windows: list[int | None],
    use_fused_gather: bool,
) -> None:
    """Exercise the full input packer with CPU buffers and metadata-only leaves."""
    decoder = object.__new__(DecoderInputPreparer)
    mode = SimpleNamespace(
        support_overlap_scheduler=lambda: True,
        extend_ctx=lambda _backend: False,
        use_one_engine=lambda: True,
    )
    spec_config = SimpleNamespace(spec_dec_mode=mode, is_linear_tree=True, max_draft_len=3)
    decoder._config = SimpleNamespace(
        use_mrope=False,
        max_num_tokens=32,
        max_beam_width=1,
        is_spec_decode=True,
        disable_overlap_scheduler=False,
        spec_config=spec_config,
        attention_backend="TRTLLM",
        enable_attention_dp=False,
        prefill_cuda_graph_backend=decoder_module.PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=None,
    )
    decoder.mapping = SimpleNamespace(pp_size=1, cp_config={}, has_cp_helix=lambda: False)
    decoder.dist = None
    decoder.guided_decoder = None
    decoder.model = SimpleNamespace(
        model_config=SimpleNamespace(pretrained_config=SimpleNamespace())
    )
    decoder._dspark_trims_submitted_tokens = trims
    decoder._dspark_device_windows = False
    decoder._prepare_inputs_fast_path = lambda *_args, **_kwargs: None
    decoder._can_use_steady_gen_fast_prepare = lambda *_args: False
    decoder._steady_mtp = None
    decoder.breakable_cuda_graph_runner = None
    decoder._new_extra_inputs = lambda: SimpleNamespace(
        add_generation_request=lambda _request: None, build=lambda *_args: {}
    )
    decoder._lora = SimpleNamespace(build=lambda *_args, **_kwargs: None)
    decoder._should_use_full_generation_page_table = lambda *_args: False
    decoder.get_runtime_tokens_per_gen_step = lambda draft_len: draft_len + 1
    decoder.iter_states = {}
    decoder.log_cached_kv_tokens_per_req = False
    decoder.use_beam_search = False
    decoder._mm_item_scheduler = None
    decoder._pinned_host_cache = {}
    decoder._pinned_host_active = {}
    decoder._staged_previous_batch_indices = None
    decoder._staged_previous_pos_indices = None
    gather_calls = []

    def gather(*args) -> bool:
        # A CPU kernel leaf verifies the production caller's packed offsets.
        new, draft, lens, slots, positions, rows, width, draft_width = args[:8]
        ids, input_begin, drafts, draft_begin, pos, pos_begin, kv, kv_begin = args[8:]
        selected_slots = slots[:rows].long()
        gather_calls.append((input_begin, pos_begin, kv_begin))
        ids[input_begin : input_begin + rows * width].copy_(
            new.transpose(0, 1)[selected_slots, :width].flatten()
        )
        drafts[draft_begin : draft_begin + rows * draft_width].copy_(
            draft[selected_slots, :draft_width].flatten()
        )
        pos[pos_begin : pos_begin + rows * width].copy_(lens[positions[: rows * width].long()])
        kv[kv_begin : kv_begin + rows].copy_(lens[selected_slots] - width)
        return True

    decoder._step_input_gather = SimpleNamespace(gather=gather) if use_fused_gather else None
    decoder._ragged_gather_indices = lambda slots, lengths: tuple(
        torch.tensor(indices, dtype=torch.long)
        for indices in ragged_gather_index_lists(slots, lengths)
    )
    decoder._buffers = SimpleNamespace()
    for name in (
        "input_ids_cuda",
        "draft_tokens_cuda",
        "previous_pos_id_offsets_cuda",
        "previous_kv_lens_offsets_cuda",
        "previous_pos_indices_cuda",
        "previous_batch_indices_cuda",
        "position_ids_cuda",
        "gather_ids_cuda",
    ):
        setattr(decoder._buffers, name, torch.full((32,), -99, dtype=torch.int32))
    decoder._gather_ids_buffer = decoder_module.DeviceCopyBuffer(decoder._buffers.gather_ids_cuda)
    windows = [prefix_window, *previous_windows]
    requests = [
        SimpleNamespace(
            py_request_id=index + 100,
            py_seq_slot=index,
            py_batch_idx=None if index == 0 else index,
            py_draft_tokens=[11, 12, 13],
            py_verify_len=window,
            is_dummy=False,
            is_attention_dp_dummy=False,
            is_cuda_graph_dummy=False,
            py_is_first_draft=False,
            py_num_accepted_draft_tokens=1,
            max_beam_num_tokens=8,
            py_prompt_len=4,
            py_num_compressed_tokens=0,
            get_last_tokens=lambda _beam: 10,
        )
        for index, window in enumerate(windows)
    ]
    scheduled = SimpleNamespace(
        context_requests=[],
        generation_requests=requests,
        num_context_requests=0,
        num_generation_requests=3,
        is_attention_dp_phase_idle=False,
        attention_dp_phase=None,
    )
    state = object.__new__(decoder_module.SampleStateTensorsSpec)
    state.new_tokens = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    state.new_tokens_lens = torch.tensor([2, 3, 4], dtype=torch.int32)
    state.next_draft_tokens = torch.arange(9, dtype=torch.int32).reshape(3, 3)
    metadata = SimpleNamespace(is_cuda_graph=False, prepare=lambda: None)

    prepared = decoder.prepare_inputs(
        scheduled,
        None,
        metadata,
        new_tensors_device=state,
        enable_spec_decode=True,
        runtime_draft_len=3,
        is_dummy=False,
    )

    inputs, draft_len = prepared.kwargs, prepared.runtime_draft_len
    lengths = [1 + window if trims else 4 for window in windows]
    prefix_len, *previous_lengths = lengths
    expected = [0] * prefix_len + [3] * previous_lengths[0] + [4] * previous_lengths[1]
    assert decoder._buffers.previous_pos_id_offsets_cuda[: sum(lengths)].tolist() == expected
    assert decoder._buffers.previous_pos_id_offsets_cuda[sum(lengths) :].count_nonzero().item() == 0
    assert decoder._buffers.previous_kv_lens_offsets_cuda[:3].tolist() == [
        0,
        3 - previous_lengths[0],
        4 - previous_lengths[1],
    ]
    assert inputs["input_ids"].numel() == sum(lengths)
    assert metadata.seq_lens.tolist() == lengths
    assert metadata.kv_cache_params.num_extra_kv_tokens == 2
    assert decoder._gather_ids_buffer.tensor[: sum(lengths)].tolist() == list(range(sum(lengths)))
    assert draft_len == 3
    assert gather_calls == (
        [(prefix_len, prefix_len, 1)] if use_fused_gather and previous_lengths == [4, 4] else []
    )


@pytest.mark.parametrize("trims", [False, True])
def test_graph_overlap_offset_zeroing_preserves_ragged_bucket_padding(trims: bool) -> None:
    decoder = object.__new__(DecoderInputPreparer)
    decoder._config = SimpleNamespace(
        max_num_tokens=16, max_beam_width=1, disable_overlap_scheduler=False, is_spec_decode=True
    )
    decoder._dspark_trims_submitted_tokens = trims
    decoder._staged_previous_batch_indices = None
    decoder._staged_previous_pos_indices = None
    decoder._step_input_gather = None
    decoder.get_runtime_tokens_per_gen_step = lambda _length: 4
    decoder._ragged_gather_indices = lambda slots, lengths: tuple(
        torch.tensor(indices, dtype=torch.long)
        for indices in ragged_gather_index_lists(slots, lengths)
    )
    decoder._buffers = SimpleNamespace(
        **{
            name: torch.full((16,), -99, dtype=torch.int32)
            for name in (
                "input_ids_cuda",
                "draft_tokens_cuda",
                "previous_batch_indices_cuda",
                "previous_pos_indices_cuda",
                "previous_pos_id_offsets_cuda",
                "previous_kv_lens_offsets_cuda",
            )
        }
    )
    lengths = [1, 2] if trims else [4, 4]
    host = SimpleNamespace(
        lists=SimpleNamespace(
            input_ids=[],
            draft_tokens=[],
            draft_lens=[],
            position_ids=list(range(sum(lengths))),
            previous_batch_indices=[0, 1],
            previous_pos_indices=[0] * lengths[0] + [1] * lengths[1],
            extend_requests=[object(), object()],
            extend_dummy_requests=[],
            generation_requests=[],
            unwritten_input_id_rows=[],
        ),
        previous_batch_tokens_per_request=lengths if trims else None,
        previous_batch_token_start=0,
    )
    decoder._copy_token_ids_to_device(
        host,
        SimpleNamespace(is_attention_dp_phase_idle=False, num_context_requests=0),
        torch.arange(8, dtype=torch.int32).reshape(4, 2, 1),
        torch.tensor([1, 2], dtype=torch.int32),
        torch.arange(6, dtype=torch.int32).reshape(2, 3),
        3,
        attn_metadata=SimpleNamespace(is_cuda_graph=True),
        enable_spec_decode=True,
    )
    expected_tail = 0 if trims else -99
    assert decoder._buffers.previous_pos_id_offsets_cuda[sum(lengths) :].tolist() == [
        expected_tail
    ] * (16 - sum(lengths))
    assert decoder._buffers.previous_kv_lens_offsets_cuda[2:].tolist() == [expected_tail] * 14


@pytest.mark.parametrize("release_owners", [False, True])
def test_graph_release_preserves_nccl_window_owner_choice(release_owners: bool) -> None:
    calls = []
    graphs = object.__new__(DecoderCudaGraphs)
    graphs.cuda_graph_runner = SimpleNamespace(
        clear=lambda **kwargs: calls.append(("native", kwargs))
    )
    graphs.breakable_cuda_graph_runner = SimpleNamespace(
        clear=lambda **kwargs: calls.append(("breakable", kwargs))
    )

    graphs.release(release_nccl_window_owners=release_owners)

    assert calls == [
        ("native", {"release_nccl_window_owners": release_owners}),
        ("breakable", {"release_nccl_window_owners": release_owners}),
    ]
