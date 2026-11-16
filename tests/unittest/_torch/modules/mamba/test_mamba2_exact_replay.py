# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mamba2 exact replay: prefill, resumed prefill and decode share the chunked-scan bits."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba import cache_manager as mamba2_cache_manager
from tensorrt_llm._torch.modules.mamba import exact_replay
from tensorrt_llm._torch.modules.mamba.cache_manager import (
    Mamba2ExactReplayState,
    NemotronHybridCacheManager,
)
from tensorrt_llm._torch.modules.mamba.exact_replay import (
    pin_ssd_autotune_configs,
    prepare_exact_replay_rows,
    replay_ssd,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    Mamba2Metadata,
    cu_seqlens_to_chunk_indices_offsets_triton,
)
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update
from tensorrt_llm._torch.modules.mamba.ssd_combined import mamba_chunk_scan_combined
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaStateLayout
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig, MambaStateConfig
from tensorrt_llm.mapping import Mapping

CHUNK = 128
HEADS, GROUPS, HEAD_DIM, DSTATE = 4, 2, 64, 128
D_INNER = HEADS * HEAD_DIM
CONV_DIM = D_INNER + 2 * GROUPS * DSTATE
WIDTH = D_INNER + CONV_DIM + HEADS


@pytest.fixture(autouse=True)
def _pinned_tiles():
    """Pin the SSD tiles for these tests and restore the autotune lists after."""
    kernels = [entry[0] for entry in exact_replay._PINNED_SSD_CONFIGS]
    saved = [(kernel, list(kernel.configs)) for kernel in kernels]
    was_pinned = exact_replay._pinned
    pin_ssd_autotune_configs()
    yield
    for kernel, configs in saved:
        kernel.configs = configs
        kernel.cache.clear()
    exact_replay._pinned = was_pinned


def _randn(*shape, dtype=torch.bfloat16, scale=0.1):
    return torch.randn(*shape, device="cuda", dtype=dtype) * scale


def test_pinned_configurations_leave_one_tile_per_kernel():
    for kernel, kwargs, num_stages, num_warps in exact_replay._PINNED_SSD_CONFIGS:
        assert len(kernel.configs) == 1
        (config,) = kernel.configs
        assert all(config.kwargs[key] == value for key, value in kwargs.items())
        assert num_stages in (None, config.num_stages)
        assert num_warps in (None, config.num_warps)


@pytest.mark.parametrize("length", [5, 128, 300, 513])
@pytest.mark.parametrize("initial", [False, True])
def test_replay_ssd_matches_the_default_scan_of_a_lone_sequence(length, initial):
    torch.manual_seed(length + int(initial))
    x = _randn(1, length, HEADS, HEAD_DIM)
    dt = _randn(1, length, HEADS)
    B = _randn(1, length, GROUPS, DSTATE)
    C = _randn(1, length, GROUPS, DSTATE)
    A = -torch.rand(HEADS, device="cuda")
    D = torch.randn(HEADS, device="cuda")
    dt_bias = torch.randn(HEADS, device="cuda") * 0.1
    init = _randn(1, HEADS, HEAD_DIM, DSTATE, dtype=torch.float32) if initial else None
    cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    seq_idx = torch.zeros(1, length, device="cuda", dtype=torch.int32)
    chunk_indices = chunk_offsets = None
    if initial:
        chunk_indices, chunk_offsets = cu_seqlens_to_chunk_indices_offsets_triton(
            cu, CHUNK, total_seqlens=length
        )
    expected = torch.empty_like(x)
    mamba_chunk_scan_combined(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=CHUNK,
        D=D,
        z=None,
        dt_bias=dt_bias,
        initial_states=init,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        cu_seqlens=cu,
        seq_idx=seq_idx,
        return_varlen_states=True,
        return_final_states=False,
        out=expected,
        state_dtype=torch.float32,
    )
    out = torch.empty_like(x)
    states, final = replay_ssd(x, dt, A, B, C, CHUNK, D, dt_bias, init, True, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert states.shape == (1, -(-length // CHUNK), HEADS, HEAD_DIM, DSTATE)
    assert states.dtype == final.dtype == torch.float32
    if init is not None:
        torch.testing.assert_close(states[:, 0], init, rtol=0, atol=0)


def _make_layer():
    layer = Mamba2Mixer.__new__(Mamba2Mixer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.tp_nheads, layer.tp_ngroups = HEADS, GROUPS
    layer.head_dim, layer.d_state, layer.chunk_size = HEAD_DIM, DSTATE, CHUNK
    layer.tp_d_inner, layer.tp_conv_dim = D_INNER, CONV_DIM
    layer.conv1d = SimpleNamespace(weight=_randn(CONV_DIM, 4), bias=_randn(CONV_DIM))
    layer.A = -torch.rand(HEADS, device="cuda")
    layer.D = torch.randn(HEADS, device="cuda")
    layer.dt_bias = torch.randn(HEADS, device="cuda") * 0.1
    layer.delta_softplus = True
    layer._mamba_ssm_cache_dtype = torch.float32
    layer._token_major_conv = True
    layer._use_flashinfer = False
    layer._stochastic_rounding_for_flashinfer = False
    layer.selective_state_update_func = selective_state_update
    layer.norm = SimpleNamespace(is_nvfp4=False)
    layer.cache_derived_state()
    layer.aux_steram = torch.cuda.Stream()
    layer.events = [torch.cuda.Event(), torch.cuda.Event()]
    return layer


class _Cache:
    """One layer's recurrent cache plus the replay buffers of Mamba2ExactReplayState."""

    def __init__(self, slots):
        self.conv = _randn(slots, CONV_DIM, 3)
        self.temporal = torch.zeros(slots, HEADS, HEAD_DIM, DSTATE, device="cuda")
        self.replay_x = torch.zeros(
            slots, CHUNK, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
        )
        self.replay_dt = torch.zeros(slots, CHUNK, HEADS, device="cuda", dtype=torch.bfloat16)
        self.replay_B = torch.zeros(
            slots, CHUNK, GROUPS, DSTATE, device="cuda", dtype=torch.bfloat16
        )
        self.replay_num_tokens = torch.zeros(slots, dtype=torch.int32, device="cuda")

    def clone(self):
        other = _Cache.__new__(_Cache)
        for name, value in vars(self).items():
            setattr(other, name, value.clone())
        return other

    def layer_cache(self):
        return SimpleNamespace(**vars(self))

    def assert_equal(self, other, slots):
        """Bitwise equality of the state that later steps read: boundary states,
        conv states, token counts and the live part of the partial-chunk buffers."""
        torch.testing.assert_close(self.conv, other.conv, rtol=0, atol=0)
        torch.testing.assert_close(self.replay_num_tokens, other.replay_num_tokens, rtol=0, atol=0)
        for slot in slots:
            count = int(self.replay_num_tokens[slot])
            live = count % CHUNK
            if count >= CHUNK:
                torch.testing.assert_close(
                    self.temporal[slot], other.temporal[slot], rtol=0, atol=0
                )
            for name in ("replay_x", "replay_dt", "replay_B"):
                torch.testing.assert_close(
                    getattr(self, name)[slot, :live],
                    getattr(other, name)[slot, :live],
                    rtol=0,
                    atol=0,
                )


def _metadata(max_batch):
    return SimpleNamespace(
        chunk_size=CHUNK,
        state_indices=torch.zeros(max_batch, dtype=torch.int32, device="cuda"),
        cu_seqlens=torch.zeros(max_batch + 1, dtype=torch.int32, device="cuda"),
        _arange_buffer=torch.arange(max_batch + 1, dtype=torch.int32, device="cuda"),
        _arange_buffer_long=torch.arange(max_batch + 1, dtype=torch.long, device="cuda"),
        has_initial_states=torch.zeros(max_batch, dtype=torch.bool, device="cuda"),
        replay_pos=torch.zeros(max_batch, dtype=torch.long, device="cuda"),
        replay_has_boundary=torch.zeros(max_batch, dtype=torch.bool, device="cuda"),
        replay_conv_has_initial=torch.ones(max_batch, dtype=torch.bool, device="cuda"),
        replay_prefill_n_pre=[],
        replay_prefill_has_boundary=[],
    )


def _prepare(metadata, cache, lengths, prefills, slots, initial):
    """What Mamba2Metadata.prepare does for the replay rows of one step."""
    batch = len(lengths)
    metadata.state_indices[:batch].copy_(torch.tensor(slots, dtype=torch.int32))
    cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    metadata.cu_seqlens[: batch + 1].copy_(cu)
    metadata.has_initial_states[:batch].copy_(torch.tensor(initial))
    context_lens = torch.tensor(lengths[:prefills], dtype=torch.int32, device="cuda")
    prepare_exact_replay_rows(metadata, cache.replay_num_tokens, batch, prefills, context_lens)


def _attn(cache, lengths, prefills):
    return SimpleNamespace(
        num_contexts=prefills,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        num_ctx_tokens=sum(lengths[:prefills]),
        num_tokens=sum(lengths),
        kv_cache_manager=SimpleNamespace(
            mamba_layer_cache=lambda _: cache.layer_cache(),
            is_speculative=lambda: False,
        ),
    )


def _run(layer, cache, projected, lengths, prefills, slots, initial, output=None):
    metadata = _metadata(len(lengths))
    _prepare(metadata, cache, lengths, prefills, slots, initial)
    if output is None:
        output = torch.empty(projected.shape[0], D_INNER, device="cuda", dtype=torch.bfloat16)
    layer.forward_core(projected, _attn(cache, lengths, prefills), metadata, None, output)
    return output


@pytest.mark.parametrize("length", [5, 128, 129, 300])
def test_resumed_prefill_and_decode_match_single_shot_prefill(length):
    torch.manual_seed(length)
    layer = _make_layer()
    projected = _randn(length, WIDTH)
    base = _Cache(3)

    full_cache = base.clone()
    full = _run(layer, full_cache, projected, [length], 1, [1], [False])

    # Prefill in three pieces; the later pieces resume from the stored
    # boundary state and partial chunk.
    split_cache = base.clone()
    cuts = sorted({0, length // 3, (2 * length) // 3, length})
    pieces = [
        _run(layer, split_cache, projected[a:b], [b - a], 1, [1], [a > 0])
        for a, b in zip(cuts, cuts[1:])
    ]
    torch.testing.assert_close(torch.cat(pieces), full, rtol=0, atol=0)
    split_cache.assert_equal(full_cache, [1])

    # Prefill a prefix, then decode token by token.
    decode_cache = base.clone()
    prefix = max(1, length // 2)
    outputs = [_run(layer, decode_cache, projected[:prefix], [prefix], 1, [1], [False])]
    for t in range(prefix, length):
        outputs.append(_run(layer, decode_cache, projected[t : t + 1], [1], 0, [1], [True]))
    torch.testing.assert_close(torch.cat(outputs), full, rtol=0, atol=0)
    decode_cache.assert_equal(full_cache, [1])
    assert int(decode_cache.replay_num_tokens[1]) == length


@pytest.mark.parametrize("stale", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_decode_ignores_what_a_slot_held_before(stale):
    """A newly acquired slot holds arbitrary bytes, including non-finite values."""
    torch.manual_seed(11)
    layer = _make_layer()
    length = 8
    projected = _randn(length, WIDTH)
    base = _Cache(3)
    full = _run(layer, base.clone(), projected, [length], 1, [1], [False])

    dirty = base.clone()
    for name in ("replay_x", "replay_dt", "replay_B"):
        getattr(dirty, name).fill_(stale)
    outputs = [_run(layer, dirty, projected[:5], [5], 1, [1], [False])]
    for name in ("replay_x", "replay_dt", "replay_B"):
        assert not getattr(dirty, name)[1, 5:].any()
    for t in range(5, length):
        outputs.append(_run(layer, dirty, projected[t : t + 1], [1], 0, [1], [True]))
    torch.testing.assert_close(torch.cat(outputs), full, rtol=0, atol=0)


def test_mixed_batch_matches_isolated_rows():
    torch.manual_seed(7)
    layer = _make_layer()
    cache = _Cache(6)
    # Histories: a partial chunk without a boundary (120), a boundary with an
    # empty partial chunk (128), a full partial chunk whose next token folds
    # (255) and a boundary with a partial chunk (300).
    for slot, tokens in {1: 120, 2: 128, 3: 255, 4: 300}.items():
        _run(layer, cache, _randn(tokens, WIDTH), [tokens], 1, [slot], [False])

    lengths, slots = [7, 13, 1, 1, 1], [0, 1, 2, 3, 4]
    initial = [False, True, True, True, True]
    projected = _randn(sum(lengths) + 3, WIDTH)
    original = projected.clone()
    mixed_cache = cache.clone()
    mixed = _run(layer, mixed_cache, projected, lengths, 2, slots, initial)

    outputs, start = [], 0
    for row, length in enumerate(lengths):
        outputs.append(
            _run(
                layer,
                cache,
                projected[start : start + length],
                [length],
                int(row < 2),
                [slots[row]],
                [initial[row]],
            )
        )
        start += length
    torch.testing.assert_close(mixed[:start], torch.cat(outputs), rtol=0, atol=0)
    assert torch.count_nonzero(mixed[start:]) == 0
    torch.testing.assert_close(projected, original, rtol=0, atol=0)
    mixed_cache.assert_equal(cache, slots)
    assert mixed_cache.replay_num_tokens.tolist() == [7, 133, 129, 256, 301, 0]


def test_decode_replay_under_cuda_graph():
    torch.manual_seed(11)
    layer = _make_layer()
    cache = _Cache(4)
    for slot, tokens in {0: 127, 1: 130, 2: 5}.items():
        _run(layer, cache, _randn(tokens, WIDTH), [tokens], 1, [slot], [False])
    slots, lengths = [0, 1, 2], [1, 1, 1]
    inputs = [_randn(3, WIDTH) for _ in range(2)]
    projected = torch.empty_like(inputs[0])
    output = torch.empty(3, D_INNER, device="cuda", dtype=torch.bfloat16)

    def prepare(metadata, cache_):
        _prepare(metadata, cache_, lengths, 0, slots, [True] * 3)

    eager_cache, expected = cache.clone(), []
    metadata = _metadata(3)
    for values in inputs:
        projected.copy_(values)
        prepare(metadata, eager_cache)
        layer.forward_core(projected, _attn(eager_cache, lengths, 0), metadata, None, output)
        expected.append(output.clone())

    # Warm up on a scratch copy, then capture one decode step on a fresh copy.
    scratch, metadata = cache.clone(), _metadata(3)
    projected.copy_(inputs[0])
    prepare(metadata, scratch)
    layer.forward_core(projected, _attn(scratch, lengths, 0), metadata, None, output)
    torch.cuda.synchronize()
    graph_cache, metadata = cache.clone(), _metadata(3)
    attn = _attn(graph_cache, lengths, 0)
    prepare(metadata, graph_cache)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layer.forward_core(projected, attn, metadata, None, output)
    graph.replay()
    torch.testing.assert_close(output, expected[0], rtol=0, atol=0)
    # Slot 0 folded its chunk in the first step; the second step reads the new
    # boundary and position through the metadata buffers, not the graph.
    projected.copy_(inputs[1])
    prepare(metadata, graph_cache)
    graph.replay()
    torch.testing.assert_close(output, expected[1], rtol=0, atol=0)
    graph_cache.assert_equal(eager_cache, slots)


def test_exact_replay_state_declares_slot_buffers_and_binds_their_views():
    layout = MambaStateLayout(
        layer_mask=(True, False, True),
        pp_layers=(0, 1, 2),
        mamba_pp_layers=(0, 2),
        mapping=Mapping(),
        conv_state_shape=(CONV_DIM, 3),
        ssm_state_shape=(HEADS, HEAD_DIM, DSTATE),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        n_groups_per_rank=GROUPS,
        max_batch_size=4,
        state_index_capacity=5,
        slot_capacity=5,
        spec_config=None,
    )
    state = Mamba2ExactReplayState(CHUNK)
    assert state.replay_shares_state_slot and state.provides_layer_cache
    assert not state.uses_replay
    layouts = [state.replay_layout(layout, layer_id) for layer_id in layout.mamba_pp_layers]
    shapes = [sorted(shape for _, shape in layer.values()) for layer in layouts]
    buffers = sorted([[CHUNK, HEADS, HEAD_DIM], [CHUNK, HEADS], [CHUNK, GROUPS, DSTATE]])
    # The first local layer also holds the token-count row.
    assert shapes == [sorted([*buffers, [4]]), buffers]
    assert sorted(str(dtype) for dtype, _ in layouts[0].values()) == sorted(
        ["torch.bfloat16"] * 3 + ["torch.int32"]
    )

    views = [
        {
            role: torch.zeros(5, *shape, dtype=dtype, device="cuda")
            for role, (dtype, shape) in layer.items()
        }
        for layer in layouts
    ]
    ssm = [torch.zeros(5, HEADS, HEAD_DIM, DSTATE, device="cuda") for _ in range(2)]
    conv = [torch.zeros(5, CONV_DIM, 3, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    with pytest.raises(RuntimeError, match="slot buffers"):
        Mamba2ExactReplayState(CHUNK).bind(layout, ssm, conv)
    state.bind_replay_pool(views)
    state.bind(layout, ssm, conv)
    assert state.replay_num_tokens.shape == (5,) and state.replay_num_tokens.dtype == torch.int32
    layer_cache = state.make_layer_cache(1, conv[1], ssm[1])
    assert layer_cache.conv is conv[1] and layer_cache.temporal is ssm[1]
    assert layer_cache.replay_x.shape == (5, CHUNK, HEADS, HEAD_DIM)
    assert layer_cache.replay_dt.shape == (5, CHUNK, HEADS)
    assert layer_cache.replay_B.shape == (5, CHUNK, GROUPS, DSTATE)
    assert layer_cache.replay_x in views[1].values()
    assert layer_cache.replay_num_tokens is state.replay_num_tokens

    state.replay_num_tokens[1] = 9
    state.replay_num_tokens[3] = 7
    # A context request keeps its count; a request that starts in generation restarts it.
    state.reset_slots(torch.tensor([1], device="cuda"), [1])
    assert state.replay_num_tokens.tolist() == [0, 9, 0, 7, 0]
    for layer_views in views:
        for buffer in layer_views.values():
            buffer.fill_(1)
    state.replay_num_tokens.copy_(torch.tensor([0, 9, 0, 7, 0]))
    state.reset_replay_slots(torch.tensor([3], device="cuda"))
    assert state.replay_num_tokens.tolist() == [0, 9, 0, 0, 0]
    # ... and its buffers, which no prefill has cleared, are zero.
    for name in ("replay_x", "replay_dt", "replay_B"):
        buffer = getattr(state.make_layer_cache(1, conv[1], ssm[1]), name)
        assert not buffer[3].any() and buffer[[0, 1, 2, 4]].all()
    state.shutdown()
    assert state.replay_num_tokens is None


def _build_manager(monkeypatch, **manager_kwargs):
    """A real Nemotron cache manager (one Mamba2 layer, one attention layer) in deterministic mode."""
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    monkeypatch.setattr(mamba2_cache_manager, "get_sm_version", lambda: 100)
    kwargs = dict(
        mamba_d_state=DSTATE,
        mamba_d_conv=4,
        mamba_num_heads=HEADS,
        mamba_n_groups=GROUPS,
        mamba_head_dim=HEAD_DIM,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        mamba_chunk_size=CHUNK,
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="per_request"),
            mamba_state_config=MambaStateConfig(additional_snapshot_offsets_from_start=[160]),
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=512,
        max_num_tokens=8192,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        spec_config=None,
        layer_mask=[False, True],
        vocab_size=4096,
        dtype=DataType.HALF,
    )
    kwargs.update(manager_kwargs)
    return NemotronHybridCacheManager(**kwargs)


def _request(request_id, tokens):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=8,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _manager_attn(manager, requests, lengths, prefills, cached):
    seq_lens = torch.tensor(lengths, dtype=torch.int)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_seqs=len(lengths),
        num_contexts=prefills,
        num_ctx_tokens=sum(lengths[:prefills]),
        num_tokens=sum(lengths),
        kv_cache_manager=manager,
        request_ids=[request.py_request_id for request in requests],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor(cached, dtype=torch.int)
        ),
    )


def _context_chunk(manager, layer, metadata, request, projected):
    """Schedule and run one context chunk of `request` through the manager and the mixer."""
    start = request.context_current_position
    length = projected.shape[0]
    assert manager.resize_context(request, num_tokens=length)
    batch = ScheduledRequests()
    batch.append_context_request(request)
    manager.prepare_resources(batch)
    attn = _manager_attn(manager, [request], [length], 1, [start])
    metadata.prepare(attn)
    output = torch.empty(length, D_INNER, device="cuda", dtype=torch.bfloat16)
    layer.forward_core(projected, attn, metadata, None, output)
    torch.cuda.synchronize()
    request.context_current_position = start + length
    manager.update_context_resources(batch)
    return output


def _slot_state(manager, request):
    """Clones of what later steps read from the request's recurrent slot."""
    slot = manager.get_state_indices([request.py_request_id], [False])[0]
    cache = manager.mamba_layer_cache(0)
    count = int(cache.replay_num_tokens[slot])
    live = count % CHUNK
    return {
        "count": count,
        "conv": cache.conv[slot].clone(),
        "temporal": cache.temporal[slot].clone(),
        "replay_x": cache.replay_x[slot, :live].clone(),
        "replay_dt": cache.replay_dt[slot, :live].clone(),
        "replay_B": cache.replay_B[slot, :live].clone(),
    }


def _assert_same_state(actual, expected):
    assert actual["count"] == expected["count"]
    names = ["conv", "replay_x", "replay_dt", "replay_B"]
    if expected["count"] >= CHUNK:
        names.append("temporal")
    for name in names:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_manager_keeps_the_replay_buffers_in_the_recurrent_state_slot(monkeypatch):
    manager = _build_manager(monkeypatch)
    try:
        assert isinstance(manager._speculative_state, Mamba2ExactReplayState)
        # No replay region of its own: the buffers are addressed by the state slot.
        assert manager.replay_layer_group_id is None
        cache = manager.mamba_layer_cache(0)
        slots = cache.temporal.shape[0]
        assert cache.replay_x.shape == (slots, CHUNK, HEADS, HEAD_DIM)
        assert cache.replay_dt.shape == (slots, CHUNK, HEADS)
        assert cache.replay_B.shape == (slots, CHUNK, GROUPS, DSTATE)
        assert cache.replay_num_tokens.shape == (slots,)
        replay_bytes = CHUNK * (HEADS * HEAD_DIM + HEADS + GROUPS * DSTATE) * 2 + 16
        assert manager._mamba_state_bytes_per_slot() == (
            manager.ssm_bytes + manager.conv_bytes + replay_bytes
        )
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "override",
    [{"is_disagg": True}, {"mamba_ssm_cache_dtype": torch.bfloat16}],
    ids=["disaggregated", "bf16_state"],
)
def test_unmet_preconditions_keep_the_default_kernels(monkeypatch, override):
    manager = _build_manager(monkeypatch, **override)
    try:
        assert not isinstance(manager._speculative_state, Mamba2ExactReplayState)
        assert manager.mamba_replay_num_tokens is None
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "override, sm, reason",
    [
        ({}, 100, None),
        ({}, 103, None),
        ({"spec_config": object()}, 100, "speculative decoding"),
        ({"is_disagg": True}, 100, "disaggregated serving"),
        ({"ssm_cache_dtype": torch.bfloat16}, 100, "float32"),
        ({}, 90, "SM100 and SM103"),
        ({"chunk_size": None}, 100, "chunk size"),
    ],
)
def test_exact_replay_preconditions(monkeypatch, override, sm, reason):
    monkeypatch.setattr(mamba2_cache_manager, "get_sm_version", lambda: sm)
    arguments = dict(
        spec_config=None, ssm_cache_dtype=torch.float32, chunk_size=CHUNK, is_disagg=False
    )
    arguments.update(override)
    actual = mamba2_cache_manager._exact_replay_disabled_reason(**arguments)
    if reason is None:
        assert actual is None
    else:
        assert reason in actual


def test_snapshot_reuse_and_suspension_carry_the_replay_state(monkeypatch):
    """A request that resumes from another request's snapshot, or from its own
    suspended cache, continues with the bits of a single-shot prefill."""
    torch.manual_seed(23)
    manager = _build_manager(monkeypatch)
    layer = _make_layer()
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=CHUNK)
    # The snapshot point has a boundary state (128) and a partial chunk (32 tokens).
    point, tail = 160, 20
    shared = _randn(point, WIDTH)
    tail_inputs = _randn(tail, WIDTH)
    full_inputs = torch.cat((shared, tail_inputs))
    prefix_tokens = list(range(1, point + 1))
    try:
        # The owner prefills to the snapshot point, which commits its slot, and goes on.
        owner = _request(1, prefix_tokens + list(range(500, 513)))
        manager.prepare_expect_snapshot_points([owner])
        assert owner.expect_snapshot_points == [point]
        assert manager.prepare_context(owner)
        _context_chunk(manager, layer, metadata, owner, shared)
        assert manager.kv_cache_map[1].num_committed_tokens == point
        _context_chunk(manager, layer, metadata, owner, _randn(13, WIDTH))

        # The reference runs the sibling's prompt in one piece, without reuse.
        reference = _request(2, list(range(2000, 2000 + point + tail)))
        manager.prepare_expect_snapshot_points([reference])
        assert manager.prepare_context(reference)
        assert reference.context_current_position == 0
        expected = _context_chunk(manager, layer, metadata, reference, full_inputs)
        expected_state = _slot_state(manager, reference)
        assert expected_state["count"] == point + tail

        # The sibling shares the first `point` tokens and resumes from the snapshot.
        sibling = _request(3, prefix_tokens + list(range(900, 900 + tail)))
        manager.prepare_expect_snapshot_points([sibling])
        assert manager.prepare_context(sibling)
        assert sibling.context_current_position == point
        resumed = _context_chunk(manager, layer, metadata, sibling, tail_inputs)
        torch.testing.assert_close(resumed, expected[point:], rtol=0, atol=0)
        _assert_same_state(_slot_state(manager, sibling), expected_state)

        # Suspending and resuming the sibling's cache keeps the state it resumes with.
        cache = manager.kv_cache_map[3]
        cache.suspend()
        assert cache.resume(manager._stream.cuda_stream)
        manager._setup_state_indices([sibling, reference], num_contexts=0)
        torch.cuda.synchronize()
        _assert_same_state(_slot_state(manager, sibling), expected_state)

        # Both decode the same next token from their slots.
        step = _randn(1, WIDTH).expand(2, WIDTH).contiguous()
        attn = _manager_attn(manager, [sibling, reference], [1, 1], 0, [0, 0])
        metadata.prepare(attn)
        output = torch.empty(2, D_INNER, device="cuda", dtype=torch.bfloat16)
        layer.forward_core(step, attn, metadata, None, output)
        torch.testing.assert_close(output[0], output[1], rtol=0, atol=0)
        _assert_same_state(_slot_state(manager, sibling), _slot_state(manager, reference))
    finally:
        manager.shutdown()


def test_metadata_prepare_snapshots_and_advances_the_slot_counts():
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)
    counter = torch.tensor([0, 13, 8, 0], dtype=torch.int32, device="cuda")
    manager = SimpleNamespace(
        mamba_replay_num_tokens=counter,
        get_state_indices=lambda request_ids, is_padding: [2, 1, 0][: len(request_ids)],
    )
    # Row 0: a new prefill on slot 2 (its old count restarts). Row 1: a resumed
    # prefill on slot 1 (13 tokens so far). Row 2: a decode on slot 0.
    seq_lens = torch.tensor([4, 3, 1], dtype=torch.int)
    attn_metadata = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=2,
        num_ctx_tokens=7,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor([0, 13], dtype=torch.int),
        ),
    )
    metadata.prepare(attn_metadata)
    assert metadata.replay_prefill_n_pre == [0, 5]
    assert metadata.replay_prefill_has_boundary == [False, True]
    assert metadata.replay_pos[:3].tolist() == [0, 5, 0]
    assert metadata.replay_has_boundary[:3].tolist() == [False, True, False]
    assert metadata.replay_conv_has_initial[:3].tolist() == [False, True, True]
    assert counter.tolist() == [1, 16, 4, 0]

    # Decode-only steps of the same requests: a full prepare(), then the steady
    # step that skips the state-index staging. Both advance the rows.
    decode_lens = torch.tensor([1, 1, 1], dtype=torch.int)
    decode_metadata = SimpleNamespace(
        seq_lens=decode_lens,
        seq_lens_cuda=decode_lens.cuda(),
        num_contexts=0,
        num_ctx_tokens=0,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor([4, 16, 1], dtype=torch.int),
        ),
    )
    metadata.prepare(decode_metadata)
    assert metadata.replay_pos[:3].tolist() == [4, 0, 1]
    assert counter.tolist() == [2, 17, 5, 0]
    metadata.prepare_steady_gen_step(decode_metadata)
    assert metadata.replay_pos[:3].tolist() == [5, 1, 2]
    assert metadata.replay_has_boundary[:3].tolist() == [False, True, False]
    assert counter.tolist() == [3, 18, 6, 0]
