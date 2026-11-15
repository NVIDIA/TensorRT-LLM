# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN exact replay: prefill, resumed prefill and decode share the chunked delta-rule bits."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fla.cache_manager import (
    GDNExactReplayState,
    Qwen35HybridCacheManager,
)
from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule
from tensorrt_llm._torch.modules.mamba import gdn_exact_replay
from tensorrt_llm._torch.modules.mamba.exact_replay import prepare_exact_replay_rows
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import fused_gdn_post_conv
from tensorrt_llm._torch.modules.mamba.gdn_exact_replay import (
    GDN_CHUNK_SIZE,
    pin_gdn_autotune_configs,
    replay_gdn,
)
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaStateLayout
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig, MambaStateConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

CHUNK = GDN_CHUNK_SIZE
K_HEADS, V_HEADS, HEAD_K, HEAD_V = 4, 8, 128, 128
KEY_DIM, VALUE_DIM = K_HEADS * HEAD_K, V_HEADS * HEAD_V
QKV_DIM = 2 * KEY_DIM + VALUE_DIM


@pytest.fixture(autouse=True)
def _pinned_tiles():
    """Pin the chunk-kernel tiles for these tests and restore the autotune lists after."""
    tuners = [
        gdn_exact_replay._autotuner(entry[0]) for entry in gdn_exact_replay._PINNED_GDN_CONFIGS
    ]
    saved = [(tuner, list(tuner.configs)) for tuner in tuners]
    was_pinned = gdn_exact_replay._pinned
    pin_gdn_autotune_configs()
    yield
    for tuner, configs in saved:
        tuner.configs = configs
        tuner.cache.clear()
    gdn_exact_replay._pinned = was_pinned


def _randn(*shape, dtype=torch.bfloat16, scale=0.1):
    return torch.randn(*shape, device="cuda", dtype=dtype) * scale


def test_pinned_configurations_leave_one_tile_per_kernel():
    for kernel, kwargs, num_stages, num_warps in gdn_exact_replay._PINNED_GDN_CONFIGS:
        (config,) = gdn_exact_replay._autotuner(kernel).configs
        assert all(config.kwargs[key] == value for key, value in kwargs.items())
        assert (config.num_stages, config.num_warps) == (num_stages, num_warps)


@pytest.mark.parametrize("length", [64, 128, 320])
@pytest.mark.parametrize("initial", [False, True])
def test_replay_gdn_matches_the_default_chunk_rule_of_a_lone_sequence(length, initial):
    torch.manual_seed(length + int(initial))
    q = torch.nn.functional.normalize(_randn(1, length, K_HEADS, HEAD_K, scale=1.0).float(), dim=-1)
    k = torch.nn.functional.normalize(_randn(1, length, K_HEADS, HEAD_K, scale=1.0).float(), dim=-1)
    q, k = q.bfloat16(), k.bfloat16()
    v = _randn(1, length, V_HEADS, HEAD_V)
    g = -torch.rand(1, length, V_HEADS, device="cuda") * 0.1
    beta = torch.rand(1, length, V_HEADS, device="cuda")
    state = _randn(1, V_HEADS, HEAD_V, HEAD_K, dtype=torch.float32)
    if not initial:
        state.zero_()
    expected, expected_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    out = torch.empty_like(v)
    final = replay_gdn(q, k, v, g, beta, HEAD_K**-0.5, state.clone(), out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert final.dtype == torch.float32
    torch.testing.assert_close(final, expected_state, rtol=0, atol=0)


def test_normalized_inputs_do_not_depend_on_the_row_of_a_token():
    """With one token per program, a token alone, inside a window and inside a long
    prefill gets the same q, k, v, g and beta bits."""
    torch.manual_seed(3)
    tokens = 1024
    conv = _randn(QKV_DIM, tokens, scale=1.0)
    a, b = _randn(tokens, V_HEADS, scale=2.0), _randn(tokens, V_HEADS, scale=2.0)
    A_log = torch.randn(V_HEADS, device="cuda") * 0.1
    dt_bias = torch.randn(V_HEADS, device="cuda") * 0.1

    def run(start, stop):
        return fused_gdn_post_conv(
            conv[:, start:stop].contiguous(),
            None,
            a[start:stop],
            b[start:stop],
            A_log,
            dt_bias,
            K_HEADS,
            HEAD_K,
            V_HEADS,
            HEAD_V,
            block_tokens=1,
        )

    full = run(0, tokens)
    for start, stop in [(5, 300), *((t, t + 1) for t in range(0, tokens, 7))]:
        for whole, part in zip(full, run(start, stop)):
            torch.testing.assert_close(part[0], whole[0, start:stop], rtol=0, atol=0)


def _make_layer(conv_scale=0.1):
    layer = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 0
    layer.attn_tp_size = 1
    layer.num_k_heads = layer.num_k_heads_per_tp = K_HEADS
    layer.num_v_heads = layer.num_v_heads_per_tp = V_HEADS
    layer.head_k_dim, layer.head_v_dim = HEAD_K, HEAD_V
    layer.key_dim_per_tp, layer.value_dim_per_tp = KEY_DIM, VALUE_DIM
    layer.activation = "silu"
    layer.conv1d = SimpleNamespace(weight=_randn(QKV_DIM, 4, scale=conv_scale), bias=None)
    layer.A_log = torch.randn(V_HEADS, device="cuda") * 0.1
    layer.dt_bias = torch.randn(V_HEADS, device="cuda") * 0.1
    layer.prefill_graphable = True
    return layer


class _Cache:
    """One layer's recurrent cache plus the replay buffers of GDNExactReplayState."""

    def __init__(self, slots):
        self.conv = _randn(slots, QKV_DIM, 3)
        self.temporal = torch.zeros(slots, V_HEADS, HEAD_V, HEAD_K, device="cuda")
        self.replay_q = torch.zeros(
            slots, CHUNK, K_HEADS, HEAD_K, device="cuda", dtype=torch.bfloat16
        )
        self.replay_k = torch.zeros_like(self.replay_q)
        self.replay_v = torch.zeros(
            slots, CHUNK, V_HEADS, HEAD_V, device="cuda", dtype=torch.bfloat16
        )
        self.replay_g = torch.zeros(slots, CHUNK, V_HEADS, device="cuda")
        self.replay_beta = torch.zeros(slots, CHUNK, V_HEADS, device="cuda")
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
            for name in ("replay_q", "replay_k", "replay_v", "replay_g", "replay_beta"):
                torch.testing.assert_close(
                    getattr(self, name)[slot, :live],
                    getattr(other, name)[slot, :live],
                    rtol=0,
                    atol=0,
                )


def _metadata(max_batch):
    return SimpleNamespace(
        chunk_size=128,  # The SSD chunk size of the shared metadata; GDN passes its own.
        state_indices=torch.zeros(max_batch, dtype=torch.int32, device="cuda"),
        query_start_loc=torch.zeros(max_batch + 1, dtype=torch.int32, device="cuda"),
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
    metadata.query_start_loc[: batch + 1].copy_(cu)
    metadata.has_initial_states[:batch].copy_(torch.tensor(initial))
    context_lens = torch.tensor(lengths[:prefills], dtype=torch.int32, device="cuda")
    prepare_exact_replay_rows(
        metadata, cache.replay_num_tokens, batch, prefills, context_lens, CHUNK
    )


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


def _inputs(tokens, scale=0.1):
    return _randn(tokens, QKV_DIM, scale=scale), _randn(tokens, V_HEADS), _randn(tokens, V_HEADS)


def _run(layer, cache, inputs, lengths, prefills, slots, initial, output=None):
    mixed_qkv, a, b = inputs
    metadata = _metadata(len(lengths))
    _prepare(metadata, cache, lengths, prefills, slots, initial)
    if output is None:
        output = torch.empty(
            1, mixed_qkv.shape[0], V_HEADS, HEAD_V, device="cuda", dtype=torch.bfloat16
        )
    layer.forward_core(mixed_qkv, a, b, _attn(cache, lengths, prefills), metadata, None, output)
    return output[0]


def _slice(inputs, start, stop):
    return tuple(tensor[start:stop] for tensor in inputs)


# Wide pre-activations reach the deep-negative SiLU region, where the
# channel-major convolution kernel switches formula for a whole warp of tokens.
@pytest.mark.parametrize("wide", [False, True], ids=["narrow", "wide"])
@pytest.mark.parametrize("length", [5, 64, 65, 150])
def test_resumed_prefill_and_decode_match_single_shot_prefill(length, wide):
    torch.manual_seed(length)
    layer = _make_layer(conv_scale=1.0 if wide else 0.1)
    inputs = _inputs(length, scale=3.0 if wide else 0.1)
    base = _Cache(3)

    full_cache = base.clone()
    full = _run(layer, full_cache, inputs, [length], 1, [1], [False])

    # Prefill in three pieces; the later pieces resume from the stored
    # boundary state and partial chunk.
    split_cache = base.clone()
    cuts = sorted({0, length // 3, (2 * length) // 3, length})
    pieces = [
        _run(layer, split_cache, _slice(inputs, a, b), [b - a], 1, [1], [a > 0])
        for a, b in zip(cuts, cuts[1:])
    ]
    torch.testing.assert_close(torch.cat(pieces), full, rtol=0, atol=0)
    split_cache.assert_equal(full_cache, [1])

    # Prefill a prefix, then decode token by token.
    decode_cache = base.clone()
    prefix = max(1, length // 2)
    outputs = [_run(layer, decode_cache, _slice(inputs, 0, prefix), [prefix], 1, [1], [False])]
    for t in range(prefix, length):
        outputs.append(_run(layer, decode_cache, _slice(inputs, t, t + 1), [1], 0, [1], [True]))
    torch.testing.assert_close(torch.cat(outputs), full, rtol=0, atol=0)
    decode_cache.assert_equal(full_cache, [1])
    assert int(decode_cache.replay_num_tokens[1]) == length


@pytest.mark.parametrize("stale", [float("nan"), float("inf"), 1e30], ids=["nan", "inf", "1e30"])
def test_replay_ignores_what_a_slot_held_before(stale):
    """A newly acquired slot holds arbitrary bytes, including non-finite values."""
    torch.manual_seed(11)
    layer = _make_layer()
    length = 8
    inputs = _inputs(length)
    base = _Cache(3)
    full = _run(layer, base.clone(), inputs, [length], 1, [1], [False])

    buffers = ("replay_q", "replay_k", "replay_v", "replay_g", "replay_beta")
    dirty = base.clone()
    for name in buffers:
        getattr(dirty, name).fill_(stale)
    outputs = [_run(layer, dirty, _slice(inputs, 0, 5), [5], 1, [1], [False])]
    for name in buffers:
        assert not getattr(dirty, name)[1, 5:].any()
    for t in range(5, length):
        outputs.append(_run(layer, dirty, _slice(inputs, t, t + 1), [1], 0, [1], [True]))
    torch.testing.assert_close(torch.cat(outputs), full, rtol=0, atol=0)


def test_mixed_batch_matches_isolated_rows():
    torch.manual_seed(7)
    layer = _make_layer()
    cache = _Cache(6)
    # Histories: a partial chunk without a boundary (60), a boundary with an
    # empty partial chunk (64), a full partial chunk whose next token folds
    # (127) and a boundary with a partial chunk (150).
    for slot, tokens in {1: 60, 2: 64, 3: 127, 4: 150}.items():
        _run(layer, cache, _inputs(tokens), [tokens], 1, [slot], [False])

    lengths, slots = [7, 13, 1, 1, 1], [0, 1, 2, 3, 4]
    initial = [False, True, True, True, True]
    inputs = _inputs(sum(lengths) + 3)
    originals = [tensor.clone() for tensor in inputs]
    mixed_cache = cache.clone()
    mixed = _run(layer, mixed_cache, inputs, lengths, 2, slots, initial)

    outputs, start = [], 0
    for row, length in enumerate(lengths):
        outputs.append(
            _run(
                layer,
                cache,
                _slice(inputs, start, start + length),
                [length],
                int(row < 2),
                [slots[row]],
                [initial[row]],
            )
        )
        start += length
    torch.testing.assert_close(mixed[:start], torch.cat(outputs), rtol=0, atol=0)
    assert torch.count_nonzero(mixed[start:]) == 0
    for tensor, original in zip(inputs, originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    mixed_cache.assert_equal(cache, slots)
    assert mixed_cache.replay_num_tokens.tolist() == [7, 73, 65, 128, 151, 0]


def test_decode_replay_under_cuda_graph():
    torch.manual_seed(11)
    layer = _make_layer()
    cache = _Cache(4)
    for slot, tokens in {0: 63, 1: 66, 2: 5}.items():
        _run(layer, cache, _inputs(tokens), [tokens], 1, [slot], [False])
    slots, lengths = [0, 1, 2], [1, 1, 1]
    steps = [_inputs(3) for _ in range(2)]
    live = tuple(torch.empty_like(tensor) for tensor in steps[0])
    output = torch.empty(1, 3, V_HEADS, HEAD_V, device="cuda", dtype=torch.bfloat16)

    eager_cache = cache.clone()
    expected = [
        _run(layer, eager_cache, step, lengths, 0, slots, [True] * 3).clone() for step in steps
    ]

    graph_cache = cache.clone()
    metadata = _metadata(3)
    attn = _attn(graph_cache, lengths, 0)
    # Warm up outside the graph on a disposable cache.
    _run(layer, cache.clone(), steps[0], lengths, 0, slots, [True] * 3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    _prepare(metadata, graph_cache, lengths, 0, slots, [True] * 3)
    for tensor, value in zip(live, steps[0]):
        tensor.copy_(value)
    with torch.cuda.graph(graph):
        layer.forward_core(*live, attn, metadata, None, output)
    graph.replay()
    torch.testing.assert_close(output[0], expected[0], rtol=0, atol=0)
    # The next step: new inputs, new positions, and slot 0 folded its chunk.
    _prepare(metadata, graph_cache, lengths, 0, slots, [True] * 3)
    for tensor, value in zip(live, steps[1]):
        tensor.copy_(value)
    graph.replay()
    torch.testing.assert_close(output[0], expected[1], rtol=0, atol=0)
    graph_cache.assert_equal(eager_cache, slots)


def test_prefill_graph_is_disabled_and_snapshots_are_rejected():
    layer = _make_layer()
    cache = _Cache(2)
    attn = _attn(cache, [3], 1)
    attn.mamba_metadata = _metadata(1)
    assert layer.prefill_graph_key(attn) is None
    metadata = _metadata(1)
    _prepare(metadata, cache, [3], 1, [0], [False])
    metadata.inforward_split = object()
    with pytest.raises(RuntimeError, match="in-forward state snapshots"):
        layer.forward_core(*_inputs(3), attn, metadata, None, None)


def test_replay_state_declares_slot_buffers_and_binds_their_views():
    layout = MambaStateLayout(
        layer_mask=(True, False, True),
        pp_layers=(0, 1, 2),
        mamba_pp_layers=(0, 2),
        mapping=Mapping(),
        conv_state_shape=(QKV_DIM, 3),
        ssm_state_shape=(V_HEADS, HEAD_V, HEAD_K),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        n_groups_per_rank=K_HEADS,
        max_batch_size=3,
        state_index_capacity=5,
        slot_capacity=5,
        spec_config=None,
    )
    slots = 5
    state = GDNExactReplayState(CHUNK)
    assert state.replay_shares_state_slot and state.provides_layer_cache
    assert not state.uses_replay
    layouts = [state.replay_layout(layout, layer_id) for layer_id in layout.mamba_pp_layers]
    per_layer = (2 * KEY_DIM + VALUE_DIM) * 2 * CHUNK + 2 * V_HEADS * 4 * CHUNK
    sizes = [
        sum(dtype.itemsize * torch.Size(shape).numel() for dtype, shape in layer.values())
        for layer in layouts
    ]
    # The first local layer also holds the 16-byte token-count row.
    assert sizes == [per_layer + 16, per_layer]

    views = [
        {
            role: torch.zeros(slots, *shape, dtype=dtype, device="cuda")
            for role, (dtype, shape) in layer.items()
        }
        for layer in layouts
    ]
    ssm = [torch.zeros(slots, V_HEADS, HEAD_V, HEAD_K, device="cuda") for _ in range(2)]
    conv = [_randn(slots, QKV_DIM, 3) for _ in range(2)]
    with pytest.raises(RuntimeError, match="slot buffers"):
        GDNExactReplayState(CHUNK).bind(layout, ssm, conv)
    state.bind_replay_pool(views)
    state.bind(layout, ssm, conv)
    cache = state.make_layer_cache(1, conv[1], ssm[1])
    assert cache.conv is conv[1] and cache.temporal is ssm[1]
    assert cache.replay_q.shape == cache.replay_k.shape == (slots, CHUNK, K_HEADS, HEAD_K)
    assert cache.replay_q is not cache.replay_k
    assert cache.replay_v.shape == (slots, CHUNK, V_HEADS, HEAD_V)
    assert cache.replay_g.shape == cache.replay_beta.shape == (slots, CHUNK, V_HEADS)
    assert cache.replay_g.dtype == cache.replay_beta.dtype == torch.float32
    assert cache.replay_num_tokens is state.replay_num_tokens
    assert state.replay_num_tokens.shape == (slots,)

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
    cache = state.make_layer_cache(1, conv[1], ssm[1])
    for name in _REPLAY_BUFFERS:
        buffer = getattr(cache, name)
        assert not buffer[3].any() and buffer[[0, 1, 2, 4]].all()
    state.shutdown()
    assert state.replay_num_tokens is None


def _build_manager(monkeypatch, **manager_kwargs):
    """A real Qwen3.5 cache manager (one GDN layer, one attention layer) in deterministic mode."""
    monkeypatch.setenv("FORCE_DETERMINISTIC", "1")
    monkeypatch.setattr(gdn_exact_replay, "get_sm_version", lambda: 100)
    kwargs = dict(
        mamba_d_state=HEAD_K,
        mamba_d_conv=4,
        mamba_num_heads=V_HEADS,
        mamba_n_groups=K_HEADS,
        mamba_head_dim=HEAD_V,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="per_request"),
            mamba_state_config=MambaStateConfig(
                additional_snapshot_offsets_from_start=[160],
                # Offsets from the prompt end select the in-forward snapshots by default.
                additional_snapshot_offsets_from_end=[2],
            ),
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
    return Qwen35HybridCacheManager(**kwargs)


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


def _context_chunk(manager, layer, metadata, request, inputs):
    """Schedule and run one context chunk of `request` through the manager and the mixer."""
    start = request.context_current_position
    length = inputs[0].shape[0]
    assert manager.resize_context(request, num_tokens=length)
    batch = ScheduledRequests()
    batch.append_context_request(request)
    manager.prepare_resources(batch)
    assert manager.inforward_snapshot_plan == {}
    attn = _manager_attn(manager, [request], [length], 1, [start])
    metadata.prepare(attn)
    output = torch.empty(1, length, V_HEADS, HEAD_V, device="cuda", dtype=torch.bfloat16)
    layer.forward_core(*inputs, attn, metadata, None, output)
    torch.cuda.synchronize()
    request.context_current_position = start + length
    manager.update_context_resources(batch)
    return output[0]


_REPLAY_BUFFERS = ("replay_q", "replay_k", "replay_v", "replay_g", "replay_beta")


def _slot_state(manager, request):
    """Clones of what later steps read from the request's recurrent slot."""
    slot = manager.get_state_indices([request.py_request_id], [False])[0]
    cache = manager.mamba_layer_cache(0)
    count = int(cache.replay_num_tokens[slot])
    live = count % CHUNK
    state = {
        "count": count,
        "conv": cache.conv[slot].clone(),
        "temporal": cache.temporal[slot].clone(),
    }
    state.update({name: getattr(cache, name)[slot, :live].clone() for name in _REPLAY_BUFFERS})
    return state


def _assert_same_state(actual, expected):
    assert actual["count"] == expected["count"]
    names = ["conv", *_REPLAY_BUFFERS]
    if expected["count"] >= CHUNK:
        names.append("temporal")
    for name in names:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_manager_keeps_the_replay_buffers_in_the_recurrent_state_slot(monkeypatch):
    manager = _build_manager(monkeypatch)
    try:
        assert isinstance(manager._speculative_state, GDNExactReplayState)
        assert manager.mamba_replay_chunk_size == CHUNK
        # No replay region of its own: the buffers are addressed by the state slot.
        assert manager.replay_layer_group_id is None
        # A snapshot point ends a context chunk instead of being staged inside one.
        assert manager.inforward_snapshot_rows == 0
        cache = manager.mamba_layer_cache(0)
        slots = cache.temporal.shape[0]
        assert cache.replay_q.shape == cache.replay_k.shape == (slots, CHUNK, K_HEADS, HEAD_K)
        assert cache.replay_v.shape == (slots, CHUNK, V_HEADS, HEAD_V)
        assert cache.replay_g.shape == cache.replay_beta.shape == (slots, CHUNK, V_HEADS)
        assert cache.replay_num_tokens.shape == (slots,)
        replay_bytes = (2 * KEY_DIM + VALUE_DIM) * 2 * CHUNK + 2 * V_HEADS * 4 * CHUNK + 16
        assert manager._mamba_state_bytes_per_slot() == (
            manager.ssm_bytes + manager.conv_bytes + replay_bytes
        )
    finally:
        manager.shutdown()


def test_disaggregated_serving_keeps_the_default_kernels(monkeypatch):
    manager = _build_manager(monkeypatch, is_disagg=True)
    try:
        assert not isinstance(manager._speculative_state, GDNExactReplayState)
        assert manager.mamba_replay_num_tokens is None
        assert getattr(manager.mamba_layer_cache(0), "replay_q", None) is None
    finally:
        manager.shutdown()


def test_snapshot_reuse_and_suspension_carry_the_replay_state(monkeypatch):
    """A request that resumes from another request's snapshot, or from its own
    suspended cache, continues with the bits of a single-shot prefill."""
    torch.manual_seed(29)
    manager = _build_manager(monkeypatch)
    layer = _make_layer()
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=128)
    # The snapshot point has two boundary states behind it and a partial chunk (32 tokens).
    point, tail = 160, 20
    shared = _inputs(point)
    tail_inputs = _inputs(tail)
    full_inputs = tuple(torch.cat(pair) for pair in zip(shared, tail_inputs))
    prefix_tokens = list(range(1, point + 1))
    try:
        # The owner prefills to the snapshot point, which commits its slot, and goes on.
        owner = _request(1, prefix_tokens + list(range(500, 513)))
        manager.prepare_expect_snapshot_points([owner])
        assert point in owner.expect_snapshot_points
        assert manager.prepare_context(owner)
        _context_chunk(manager, layer, metadata, owner, shared)
        assert manager.kv_cache_map[1].num_committed_tokens == point
        _context_chunk(manager, layer, metadata, owner, _inputs(11))

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
        step = tuple(tensor.expand(2, -1).contiguous() for tensor in _inputs(1))
        attn = _manager_attn(manager, [sibling, reference], [1, 1], 0, [0, 0])
        metadata.prepare(attn)
        output = torch.empty(1, 2, V_HEADS, HEAD_V, device="cuda", dtype=torch.bfloat16)
        layer.forward_core(*step, attn, metadata, None, output)
        torch.testing.assert_close(output[0, 0], output[0, 1], rtol=0, atol=0)
        _assert_same_state(_slot_state(manager, sibling), _slot_state(manager, reference))
    finally:
        manager.shutdown()
