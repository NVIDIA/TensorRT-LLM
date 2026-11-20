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
    DECODE_GROUP_ROWS,
    pin_ssd_autotune_configs,
    prepare_exact_replay_rows,
    replay_ssd,
)
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import (
    cu_seqlens_to_chunk_indices_offsets_triton,
)
from tensorrt_llm._torch.modules.mamba.mamba2_mixer import Mamba2Mixer
from tensorrt_llm._torch.modules.mamba.replay_metadata import ReplayMamba2Metadata
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


# The autotune lists as the kernels declare them, before any test pins them.
_DECLARED_SSD_CONFIGS = [
    (entry[0], list(entry[0].configs)) for entry in exact_replay._PINNED_SSD_CONFIGS
]


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
        # Tokens each slot's request has run; the executor keeps this count
        # (num_cached_tokens_per_seq), not the cache.
        self.counts = [0] * slots

    def clone(self):
        other = _Cache.__new__(_Cache)
        for name, value in vars(self).items():
            setattr(other, name, list(value) if name == "counts" else value.clone())
        return other

    def layer_cache(self):
        return SimpleNamespace(
            **{name: value for name, value in vars(self).items() if name != "counts"}
        )

    def assert_equal(self, other, slots):
        """Bitwise equality of the state that later steps read: boundary states,
        conv states, token counts and the live part of the partial-chunk buffers."""
        torch.testing.assert_close(self.conv, other.conv, rtol=0, atol=0)
        assert self.counts == other.counts
        for slot in slots:
            count = self.counts[slot]
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
    """The replay rows of ReplayMamba2Metadata, without a cache manager."""
    return ReplayMamba2Metadata(max_batch, CHUNK)


def _prepare(metadata, cache, lengths, prefills, slots, initial):
    """What ReplayMamba2Metadata.prepare does for the replay rows of one step.

    A row continues its slot's request (``initial``) or starts a new one.
    """
    batch = len(lengths)
    metadata.state_indices[:batch].copy_(torch.tensor(slots, dtype=torch.int32))
    cu = torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32)
    metadata.cu_seqlens[: batch + 1].copy_(cu)
    cached = [cache.counts[slot] if keep else 0 for slot, keep in zip(slots, initial)]
    prepare_exact_replay_rows(metadata, cached, batch, prefills, CHUNK)
    for slot, count, length in zip(slots, cached, lengths):
        cache.counts[slot] = count + length


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
    assert decode_cache.counts[1] == length


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
    for name in ("replay_x", "replay_dt", "replay_B", "temporal", "conv"):
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
    assert mixed_cache.counts == [7, 133, 129, 256, 301, 0]


@pytest.mark.parametrize("group_tokens", [None, 2 * CHUNK], ids=["one_call", "split_calls"])
def test_prefill_groups_match_isolated_rows(monkeypatch, group_tokens):
    """Sequences with the same number of chunks run as one dense batch; every
    row keeps the bits it has alone, fresh or resumed."""
    if group_tokens is not None:
        monkeypatch.setattr(exact_replay, "PREFILL_GROUP_TOKENS", group_tokens)
    torch.manual_seed(19)
    layer = _make_layer()
    cache = _Cache(8)
    # Histories: none, a partial chunk (5), a boundary with a partial chunk (130).
    for slot, tokens in {1: 5, 2: 130, 5: 5}.items():
        _run(layer, cache, _randn(tokens, WIDTH), [tokens], 1, [slot], [False])
    # Two chunks each: a fresh row, a resumed partial chunk, a resumed row that
    # ends on a chunk boundary, a fresh row one token into its second chunk.
    # One chunk each: a short row, an exact chunk, a resumed partial chunk.
    lengths = [200, 150, 2 * CHUNK - 2, CHUNK + 1, 7, CHUNK, 100]
    slots = [0, 1, 2, 3, 4, 6, 5]
    initial = [False, True, True, False, False, False, True]
    projected = _randn(sum(lengths), WIDTH)
    grouped_cache = cache.clone()
    grouped = _run(layer, grouped_cache, projected, lengths, len(lengths), slots, initial)

    outputs, start = [], 0
    for length, slot, keep in zip(lengths, slots, initial):
        outputs.append(
            _run(layer, cache, projected[start : start + length], [length], 1, [slot], [keep])
        )
        start += length
    torch.testing.assert_close(grouped, torch.cat(outputs), rtol=0, atol=0)
    grouped_cache.assert_equal(cache, slots)
    # The rows decode on from the grouped prefill like from the isolated one.
    step = _randn(len(slots), WIDTH)
    torch.testing.assert_close(
        _run(layer, grouped_cache, step, [1] * len(slots), 0, slots, [True] * len(slots)),
        _run(layer, cache, step, [1] * len(slots), 0, slots, [True] * len(slots)),
        rtol=0,
        atol=0,
    )


def test_a_step_makes_no_synchronizing_call():
    """Prefill rows (alone, grouped, resumed) and decode rows issue no blocking host sync."""
    torch.manual_seed(31)
    layer = _make_layer()
    cache = _Cache(6)
    for slot, tokens in {1: 5, 4: CHUNK - 1, 5: 40}.items():
        _run(layer, cache, _randn(tokens, WIDTH), [tokens], 1, [slot], [False])
    lengths, slots = [300, 20, 30, 150, 1, 1], [0, 1, 2, 3, 4, 5]
    initial = [False, True, False, False, True, True]
    projected = _randn(sum(lengths), WIDTH)
    output = torch.empty(sum(lengths), D_INNER, device="cuda", dtype=torch.bfloat16)
    # Compile every kernel variant first, on a copy of the cache.
    _run(layer, cache.clone(), projected, lengths, 4, slots, initial)
    metadata = _metadata(len(lengths))
    _prepare(metadata, cache, lengths, 4, slots, initial)
    attn = _attn(cache, lengths, 4)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        layer.forward_core(projected, attn, metadata, None, output)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_decode_groups_match_isolated_rows():
    """A decode batch is scanned in groups; every row keeps the bits it has alone."""
    torch.manual_seed(13)
    layer = _make_layer()
    rows = 2 * DECODE_GROUP_ROWS + 5
    cache = _Cache(rows)
    # Histories that end before, at and after a chunk boundary.
    for slot in range(rows):
        tokens = (CHUNK - 3, CHUNK - 1, CHUNK, CHUNK + 2, 1)[slot % 5] + slot // 5
        _run(layer, cache, _randn(tokens, WIDTH), [tokens], 1, [slot], [False])
    projected = _randn(rows, WIDTH)
    slots = list(range(rows))
    batch_cache = cache.clone()
    together = _run(layer, batch_cache, projected, [1] * rows, 0, slots, [True] * rows)
    alone = [
        _run(layer, cache, projected[slot : slot + 1], [1], 0, [slot], [True]) for slot in slots
    ]
    torch.testing.assert_close(together, torch.cat(alone), rtol=0, atol=0)
    batch_cache.assert_equal(cache, slots)


def test_a_request_that_starts_in_generation_decodes_like_a_one_token_prefill():
    """A one-token prompt promoted to a decode row has no cached tokens."""
    torch.manual_seed(17)
    layer = _make_layer()
    projected = _randn(3, WIDTH)
    base = _Cache(2)
    prefilled = base.clone()
    expected = [_run(layer, prefilled, projected[:1], [1], 1, [1], [False])]
    # The slot's previous owner left a boundary state and a convolution state;
    # the manager clears the replay buffers of a request that starts in generation.
    decoded = base.clone()
    decoded.temporal.fill_(float("nan"))
    decoded.conv.fill_(float("nan"))
    actual = [_run(layer, decoded, projected[:1], [1], 0, [1], [False])]
    for t in (1, 2):
        expected.append(_run(layer, prefilled, projected[t : t + 1], [1], 0, [1], [True]))
        actual.append(_run(layer, decoded, projected[t : t + 1], [1], 0, [1], [True]))
    torch.testing.assert_close(torch.cat(actual), torch.cat(expected), rtol=0, atol=0)


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

    def prepare(step_metadata, step_cache):
        _prepare(step_metadata, step_cache, lengths, 0, slots, [True] * 3)

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
    assert not state.uses_replay
    layouts = [state.replay_layout(layout, layer_id) for layer_id in layout.mamba_pp_layers]
    shapes = [sorted(shape for _, shape in layer.values()) for layer in layouts]
    buffers = sorted([[CHUNK, HEADS, HEAD_DIM], [CHUNK, HEADS], [CHUNK, GROUPS, DSTATE]])
    assert shapes == [buffers, buffers]
    assert {dtype for layer in layouts for dtype, _ in layer.values()} == {torch.bfloat16}

    views = [
        {
            role: torch.ones(5, *shape, dtype=dtype, device="cuda")
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
    layer_cache = state.make_layer_cache(1, conv[1], ssm[1])
    assert layer_cache.conv is conv[1] and layer_cache.temporal is ssm[1]
    assert layer_cache.replay_x.shape == (5, CHUNK, HEADS, HEAD_DIM)
    assert layer_cache.replay_dt.shape == (5, CHUNK, HEADS)
    assert layer_cache.replay_B.shape == (5, CHUNK, GROUPS, DSTATE)
    assert layer_cache.replay_x in views[1].values()

    # A context request keeps what it buffered; a request that starts in
    # generation gets cleared buffers, which no prefill has cleared for it.
    state.reset_slots(torch.tensor([1], device="cuda"), [1])
    state.reset_replay_slots(torch.tensor([3], device="cuda"))
    for layer_offset in (0, 1):
        layer_cache = state.make_layer_cache(layer_offset, conv[1], ssm[1])
        for name in ("replay_x", "replay_dt", "replay_B"):
            buffer = getattr(layer_cache, name)
            assert not buffer[3].any() and buffer[[0, 1, 2, 4]].all()
    state.shutdown()


def _build_manager(monkeypatch, switch="FORCE_DETERMINISTIC", **manager_kwargs):
    """A real Nemotron cache manager (one Mamba2 layer, one attention layer).

    ``switch`` names the environment variable that asks for the deterministic
    kernels; None leaves both unset.
    """
    for name in ("FORCE_DETERMINISTIC", "FORCE_MAMBA_KERNEL_DETERMINISTIC"):
        monkeypatch.delenv(name, raising=False)
    if switch is not None:
        monkeypatch.setenv(switch, "1")
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


def _slot_state(manager, request, count):
    """Clones of what later steps read from the slot of a request that has run `count` tokens."""
    slot = manager.get_state_indices([request.py_request_id], [False])[0]
    cache = manager.mamba_layer_cache(0)
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
        assert manager.mamba_exact_replay_chunk_size == CHUNK
        # No replay region of its own: the buffers are addressed by the state slot.
        assert manager.replay_layer_group_id is None
        cache = manager.mamba_layer_cache(0)
        slots = cache.temporal.shape[0]
        assert cache.replay_x.shape == (slots, CHUNK, HEADS, HEAD_DIM)
        assert cache.replay_dt.shape == (slots, CHUNK, HEADS)
        assert cache.replay_B.shape == (slots, CHUNK, GROUPS, DSTATE)
        replay_bytes = CHUNK * (HEADS * HEAD_DIM + HEADS + GROUPS * DSTATE) * 2
        assert manager._mamba_state_bytes_per_slot() == (
            manager.ssm_bytes + manager.conv_bytes + replay_bytes
        )
        # The size the cost estimate reserves before the manager exists.
        assert replay_bytes == mamba2_cache_manager._replay_layout_bytes(
            manager._speculative_state, manager._state_layout
        )
    finally:
        manager.shutdown()


def test_building_the_manager_pins_the_tiles(monkeypatch):
    # Start from the autotune lists as the kernels declare them.
    for kernel, configs in _DECLARED_SSD_CONFIGS:
        kernel.configs = list(configs)
    exact_replay._pinned = False
    assert any(len(kernel.configs) > 1 for kernel, _ in _DECLARED_SSD_CONFIGS)
    manager = _build_manager(monkeypatch)
    try:
        assert all(len(kernel.configs) == 1 for kernel, _ in _DECLARED_SSD_CONFIGS)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "switch, override",
    [
        ("FORCE_DETERMINISTIC", {"is_disagg": True}),
        ("FORCE_DETERMINISTIC", {"mamba_ssm_cache_dtype": torch.bfloat16}),
        ("FORCE_DETERMINISTIC", {"mamba_cache_dtype": torch.float16}),
        (None, {}),
    ],
    ids=["disaggregated", "bf16_state", "float16_model", "switch_unset"],
)
def test_unmet_preconditions_keep_the_default_kernels(monkeypatch, switch, override):
    manager = _build_manager(monkeypatch, switch=switch, **override)
    try:
        assert not isinstance(manager._speculative_state, Mamba2ExactReplayState)
        assert manager.mamba_exact_replay_chunk_size is None
        assert getattr(manager.mamba_layer_cache(0), "replay_x", None) is None
        replay_free = manager.ssm_bytes + manager.conv_bytes
        assert manager._mamba_state_bytes_per_slot() == replay_free
    finally:
        manager.shutdown()


def test_the_component_switch_selects_exact_replay_alone(monkeypatch):
    manager = _build_manager(monkeypatch, switch="FORCE_MAMBA_KERNEL_DETERMINISTIC")
    try:
        assert isinstance(manager._speculative_state, Mamba2ExactReplayState)
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "override, sm, reason",
    [
        ({}, 100, None),
        ({}, 103, None),
        ({}, 90, "SM100 and SM103 only (this GPU is SM90)"),
        # The hardware is reported first: a float32 state cannot help on SM90.
        ({"ssm_cache_dtype": torch.bfloat16}, 90, "SM100 and SM103"),
        ({"spec_config": object()}, 100, "speculative decoding"),
        ({"is_disagg": True}, 100, "disaggregated serving"),
        ({"model_dtype": torch.float16}, 100, "model dtype"),
        ({"ssm_cache_dtype": torch.bfloat16}, 100, "kv_cache_config.mamba_ssm_cache_dtype"),
        ({"chunk_size": None}, 100, "chunk size"),
    ],
)
def test_exact_replay_preconditions(monkeypatch, override, sm, reason):
    monkeypatch.setattr(mamba2_cache_manager, "get_sm_version", lambda: sm)
    arguments = dict(
        spec_config=None,
        model_dtype=torch.bfloat16,
        ssm_cache_dtype=torch.float32,
        chunk_size=CHUNK,
        is_disagg=False,
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
    metadata = manager.mamba_metadata_cls(max_batch_size=4, chunk_size=CHUNK)
    # The snapshot point has a boundary state (128) and a partial chunk (32 tokens).
    point, tail = 160, 20
    total = point + tail
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
        reference = _request(2, list(range(2000, 2000 + total)))
        manager.prepare_expect_snapshot_points([reference])
        assert manager.prepare_context(reference)
        assert reference.context_current_position == 0
        expected = _context_chunk(manager, layer, metadata, reference, full_inputs)
        expected_state = _slot_state(manager, reference, total)

        # The sibling shares the first `point` tokens and resumes from the snapshot.
        sibling = _request(3, prefix_tokens + list(range(900, 900 + tail)))
        manager.prepare_expect_snapshot_points([sibling])
        assert manager.prepare_context(sibling)
        assert sibling.context_current_position == point
        resumed = _context_chunk(manager, layer, metadata, sibling, tail_inputs)
        torch.testing.assert_close(resumed, expected[point:], rtol=0, atol=0)
        _assert_same_state(_slot_state(manager, sibling, total), expected_state)

        # Suspending and resuming the sibling's cache keeps the state it resumes with.
        cache = manager.kv_cache_map[3]
        cache.suspend()
        assert cache.resume(manager._stream.cuda_stream)
        manager._setup_state_indices([sibling, reference], num_contexts=0)
        torch.cuda.synchronize()
        _assert_same_state(_slot_state(manager, sibling, total), expected_state)

        # Both decode the same next token from their slots.
        step = _randn(1, WIDTH).expand(2, WIDTH).contiguous()
        attn = _manager_attn(manager, [sibling, reference], [1, 1], 0, [total, total])
        metadata.prepare(attn)
        output = torch.empty(2, D_INNER, device="cuda", dtype=torch.bfloat16)
        layer.forward_core(step, attn, metadata, None, output)
        torch.testing.assert_close(output[0], output[1], rtol=0, atol=0)
        _assert_same_state(
            _slot_state(manager, sibling, total + 1), _slot_state(manager, reference, total + 1)
        )
    finally:
        manager.shutdown()


def test_decode_through_the_manager_matches_a_single_shot_prefill(monkeypatch):
    """Prefill, a full prepare() and steady decode steps, through the manager and its metadata."""
    torch.manual_seed(29)
    manager = _build_manager(monkeypatch)
    layer = _make_layer()
    metadata = manager.mamba_metadata_cls(max_batch_size=4, chunk_size=CHUNK)
    assert isinstance(metadata, ReplayMamba2Metadata)
    # The decode steps cross the chunk boundary at 128 tokens.
    prompt, steps = CHUNK - 2, 5
    inputs = _randn(prompt + steps, WIDTH)
    try:
        reference = _request(1, list(range(1, prompt + steps + 1)))
        assert manager.prepare_context(reference)
        expected = _context_chunk(manager, layer, metadata, reference, inputs)

        request = _request(2, list(range(1000, 1000 + prompt)))
        assert manager.prepare_context(request)
        outputs = [_context_chunk(manager, layer, metadata, request, inputs[:prompt])]
        manager._setup_state_indices([request], num_contexts=0)
        output = torch.empty(1, D_INNER, device="cuda", dtype=torch.bfloat16)
        for step in range(steps):
            # The executor passes the tokens the request has run; a steady
            # step keeps the previous metadata and only moves the rows on.
            attn = _manager_attn(manager, [request], [1], 0, [prompt + step])
            if step == 0:
                metadata.prepare(attn)
            else:
                metadata.prepare_steady_gen_step(attn)
            assert metadata.replay_pos[0].item() == (prompt + step) % CHUNK
            assert metadata.replay_has_boundary[0].item() == (prompt + step >= CHUNK)
            layer.forward_core(
                inputs[prompt + step : prompt + step + 1], attn, metadata, None, output
            )
            outputs.append(output.clone())
        torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)
        _assert_same_state(
            _slot_state(manager, request, prompt + steps),
            _slot_state(manager, reference, prompt + steps),
        )
    finally:
        manager.shutdown()


def test_metadata_sets_the_rows_from_the_cached_tokens_and_advances_them():
    metadata = ReplayMamba2Metadata(max_batch_size=4, chunk_size=8)
    manager = SimpleNamespace(
        mamba_exact_replay_chunk_size=8,
        get_state_indices=lambda request_ids, is_padding: [2, 1, 0][: len(request_ids)],
    )
    # Row 0: a new prefill. Row 1: a resumed prefill (13 tokens so far). Row 2:
    # a decode whose request has run 8 tokens.
    seq_lens = torch.tensor([4, 3, 1], dtype=torch.int)
    attn_metadata = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=2,
        num_ctx_tokens=7,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor([0, 13, 8], dtype=torch.int),
        ),
    )
    metadata.prepare(attn_metadata)
    assert metadata.replay_prefill_n_pre == [0, 5]
    assert metadata.replay_prefill_has_boundary == [False, True]
    assert metadata.replay_pos[:3].tolist() == [0, 5, 0]
    assert metadata.replay_has_boundary[:3].tolist() == [False, True, True]
    assert metadata.replay_conv_has_initial[:3].tolist() == [False, True, True]

    # Decode-only steps of the same requests: a full prepare(), then steady
    # steps that skip the host walk. A list of counts works like a tensor.
    decode_lens = torch.tensor([1, 1, 1], dtype=torch.int)
    decode_metadata = SimpleNamespace(
        seq_lens=decode_lens,
        seq_lens_cuda=decode_lens.cuda(),
        num_contexts=0,
        num_ctx_tokens=0,
        kv_cache_manager=manager,
        request_ids=[10, 11, 12],
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[6, 16, 0]),
    )
    metadata.prepare(decode_metadata)
    assert metadata.replay_pos[:3].tolist() == [6, 0, 0]
    assert metadata.replay_has_boundary[:3].tolist() == [False, True, False]
    assert metadata.replay_conv_has_initial[:3].tolist() == [True, True, False]
    metadata.prepare_steady_gen_step(decode_metadata)
    assert metadata.replay_pos[:3].tolist() == [7, 1, 1]
    assert metadata.replay_has_boundary[:3].tolist() == [False, True, False]
    assert metadata.replay_conv_has_initial[:3].tolist() == [True, True, True]
    # The first row's chunk folds on the next step.
    metadata.prepare_steady_gen_step(decode_metadata)
    assert metadata.replay_pos[:3].tolist() == [0, 2, 2]
    assert metadata.replay_has_boundary[:3].tolist() == [True, True, False]

    # Without the exact-replay state the rows stay untouched.
    manager.mamba_exact_replay_chunk_size = None
    metadata.prepare_steady_gen_step(decode_metadata)
    assert metadata.replay_pos[:3].tolist() == [0, 2, 2]
