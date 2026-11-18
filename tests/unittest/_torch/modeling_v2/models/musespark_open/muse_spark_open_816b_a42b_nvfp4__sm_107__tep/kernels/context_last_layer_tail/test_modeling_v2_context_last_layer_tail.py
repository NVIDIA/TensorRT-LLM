# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host tests of ``context_last_layer_tail``: the rows the LM head reads, the padded row count, where each
needed row's residual comes from, which steps the last layer's library is offered, that an unarmed
call runs the library behind it with the manager's arguments, and the depth cap: which steps it
leaves unchanged, and that such a step runs the decoder layer's own post-attention path."""

import contextlib
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_last_layer_tail import (  # noqa: E501
    tail as tail_path,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType


def test_needed_rows_are_each_context_requests_last_row_then_every_generation_row():
    assert tail_path.needed_rows([8192], 8192, 8192) == [8191]
    assert tail_path.needed_rows([3000, 5192], 8192, 8192) == [2999, 8191]
    assert tail_path.needed_rows([1, 1, 6], 8, 8) == [0, 1, 7]
    # Generation rows (a mixed step) are all read by the one-engine speculative head.
    assert tail_path.needed_rows([5], 5, 9) == [4, 5, 6, 7, 8]
    assert tail_path.needed_rows([], 0, 3) == [0, 1, 2]


@pytest.mark.parametrize(
    "count, padded", [(0, 1), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (96, 128), (128, 128)]
)
def test_padded_count_is_the_next_power_of_two(count, padded):
    assert tail_path.padded_count(count) == padded


def test_owner_plan_takes_each_row_from_the_rank_that_holds_it():
    shard = 2048  # an 8192-row step over 4 ranks
    rows = [100, 2047, 2048, 8191, 8191]
    for rank in range(4):
        slots, sources, flat = tail_path.owner_plan(rows, shard, rank)
        owned = [s for s, r in enumerate(rows) if r // shard == rank]
        assert slots == owned
        assert sources == [rows[s] - rank * shard for s in owned]
        assert all(0 <= src < shard for src in sources)
        # The all-gathered buffer is [rank 0 slots | rank 1 slots | ...]: owner * count + slot.
        assert flat == [(r // shard) * len(rows) + s for s, r in enumerate(rows)]


def _metadata(contexts, generations, beam_width=1):
    return types.SimpleNamespace(
        num_contexts=contexts, num_generations=generations, beam_width=beam_width
    )


def test_pure_context_steps_only():
    assert tail_path.takes_pure_context(_metadata(2, 0))
    assert not tail_path.takes_pure_context(_metadata(0, 3))
    assert not tail_path.takes_pure_context(_metadata(2, 1))
    assert not tail_path.takes_pure_context(_metadata(2, 0, beam_width=2))


class _Behind:
    def __init__(self, supported=True):
        self.calls = []
        self.supported = supported

    def is_supported(self, *args, **kwargs):
        return self.supported

    def forward(self, *args):
        self.calls.append(args)


class _Attention:
    """The attributes ``PhasedFmha`` reads from its owner (held through a weak reference)."""

    is_mla_enable = False
    kv_lora_rank = None
    v_head_dim = None
    head_dim = 64
    layer_idx = 61
    num_heads = 32
    num_kv_heads = 4


def _forward_args(input_type=AttentionInputType.context_only):
    return types.SimpleNamespace(attention_input_type=input_type, output=object(), output_sf=None)


def test_the_library_is_offered_on_pure_context_steps_whatever_the_arming():
    behind, fallback, state = _Behind(), _Behind(), tail_path.TailState()
    owner = _Attention()
    lib = tail_path.TailPureContextFmha(owner, behind, fallback, state)
    for armed in (False, True):
        state.armed = armed
        assert lib._is_supported(
            None, None, None, _metadata(2, 0), _forward_args(), phase=FmhaPhase.CONTEXT
        )
        assert not lib._is_supported(None, None, None, _metadata(2, 0), _forward_args())
        assert not lib._is_supported(
            None, None, None, _metadata(2, 1), _forward_args(), phase=FmhaPhase.CONTEXT
        )
        assert not lib._is_supported(
            None,
            None,
            None,
            _metadata(2, 0),
            _forward_args(AttentionInputType.generation_only),
            phase=FmhaPhase.CONTEXT,
        )
    fallback.supported = False
    assert not lib._is_supported(
        None, None, None, _metadata(2, 0), _forward_args(), phase=FmhaPhase.CONTEXT
    )


def test_an_unarmed_call_runs_the_library_behind_with_the_managers_arguments():
    behind, fallback, state = _Behind(), _Behind(), tail_path.TailState()
    owner = _Attention()
    lib = tail_path.TailPureContextFmha(owner, behind, fallback, state)
    call = ("q", None, None, _metadata(1, 0), _forward_args())
    lib.forward(*call)
    assert behind.calls == [call]
    assert state.passed_calls == 1 and state.pruned_calls == 0
    with pytest.raises(RuntimeError, match="pure context steps only"):
        lib.run_generation(None)


def _depth_metadata(kv_lens, contexts, seq_lens=None, cached=None):
    meta = types.SimpleNamespace(num_contexts=contexts)
    if kv_lens is not None:
        meta.kv_lens_runtime = torch.tensor(kv_lens, dtype=torch.int32)
    if seq_lens is not None:
        meta.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        meta.kv_cache_params = types.SimpleNamespace(num_cached_tokens_per_seq=cached)
    return meta


def test_the_deepest_context_depth_is_the_longest_context_requests_keys_after_the_chunk():
    # kv_lens_runtime holds each request's cached prefix plus this step's rows.
    assert tail_path.deepest_context_depth(_depth_metadata([270336, 8192, 77], 2)) == 270336
    # Generation rows past the context requests are not context depth.
    assert tail_path.deepest_context_depth(_depth_metadata([8192, 500000], 1)) == 8192
    assert tail_path.deepest_context_depth(_depth_metadata([500000], 0)) == 0
    # Without the runtime view: cached prefix plus the chunk.
    meta = _depth_metadata(None, 2, seq_lens=[8192, 4096], cached=[262144, 300000])
    assert tail_path.deepest_context_depth(meta) == 304096
    assert tail_path.deepest_context_depth(types.SimpleNamespace(num_contexts=1)) == 0


class _Step:
    def __init__(self):
        self.active = False


def _tail_with(monkeypatch, meta, max_depth, step=None):
    layer = types.SimpleNamespace(layer_idx=61)
    model = types.SimpleNamespace(layers=[layer], _sp_mapping=types.SimpleNamespace(tp_size=4))
    state = tail_path.TailState()
    monkeypatch.setattr(
        tail_path, "get_model_extra_attrs", lambda: {"attention_metadata": lambda: meta}
    )
    return tail_path.LastLayerTail(model, None, state, max_depth=max_depth, step=step), state


@pytest.mark.parametrize(
    "depth, max_depth, armed",
    [
        (262143, 262144, True),
        (262144, 262144, False),
        (458752, 262144, False),
        (75776, 262144, True),
        (458752, None, True),
        (458752, 1 << 30, True),
        (8192, 1, False),
    ],
)
def test_a_step_at_or_past_the_depth_cap_leaves_the_library_unarmed(
    monkeypatch, depth, max_depth, armed
):
    tail, state = _tail_with(monkeypatch, _depth_metadata([depth], 1), max_depth)
    tail._arm()
    assert state.armed is armed
    assert tail._unchanged is (not armed)
    assert state.depth_declined_steps == int(not armed)


def test_the_tail_break_runs_the_layer_on_a_capped_step_and_the_tail_otherwise(monkeypatch):
    meta = _depth_metadata([300000], 1)
    tail, state = _tail_with(monkeypatch, meta, 262144)
    calls = []
    monkeypatch.setattr(tail, "_run_layer", lambda *a: calls.append("layer"))
    monkeypatch.setattr(tail, "_run_tail", lambda *a: calls.append("tail"))
    for kv in (300000, 100000, 262144, 262143):
        meta.kv_lens_runtime[0] = kv
        tail._arm()
        tail._tail(None, None, None)
        # Every break leaves the next step undecided and the library unarmed.
        assert not state.armed and not tail._unchanged
    assert calls == ["layer", "tail", "layer", "tail"]
    assert state.depth_declined_steps == 2


def test_a_capped_step_runs_the_decoder_layers_own_post_attention_path(monkeypatch):
    """``_run_layer`` against ``MuseSparkOpenDecoderLayer.forward`` (sequence-parallel arm, no
    next-layer norm) plus the sequence-parallel forward's final norm and all-gather, over the
    same stand-in modules: the same values, bit for bit, and the MoE called alike."""
    from tensorrt_llm._torch.attention import attention as attention_module
    from tensorrt_llm._torch.models import modeling_musespark_open as builtin

    tp, rows, hidden_size, width = 4, 10, 16, 12
    shard = -(-rows // tp)
    gen = torch.Generator().manual_seed(7)
    attention_rows = torch.randn(rows, width, generator=gen).to(torch.bfloat16)
    residual = torch.randn(shard, hidden_size, generator=gen)
    w_o = torch.randn(width, hidden_size, generator=gen).to(torch.bfloat16)
    gain = torch.randn(hidden_size, generator=gen)

    def reducescatter(t, mapping, dim=0):
        assert dim == 0 and t.shape[0] % mapping.tp_size == 0
        parts = t.float().view(mapping.tp_size, -1, t.shape[1])
        return (parts * torch.arange(1, mapping.tp_size + 1).view(-1, 1, 1)).sum(0).to(t.dtype)

    def allgather(t, mapping, dim=0):
        assert dim == 0
        return torch.cat([t + r for r in range(mapping.tp_size)], dim=0)

    for module in (builtin, tail_path):
        monkeypatch.setattr(module, "reducescatter", reducescatter, raising=False)
        monkeypatch.setattr(module, "allgather", allgather, raising=False)

    mapping = types.SimpleNamespace(
        tp_size=tp, has_cp_helix=lambda: False, enable_attention_dp=False
    )
    step = _Step()
    moe_calls = []

    class OProj:
        def __call__(self, x, all_reduce_params=None, lora_params=None, layer_idx=None):
            assert all_reduce_params is not None and not all_reduce_params.enable_allreduce
            assert lora_params is None and layer_idx == 61
            return x @ w_o

    class Attention:
        layer_idx = 61
        o_proj = OProj()

        def __call__(self, position_ids, hidden_states, attn_metadata, **kwargs):
            # The built-in attention module's default projection of its attention output.
            return attention_module._helix_cp_output_projection(
                self.o_proj,
                attention_rows,
                attn_metadata,
                kwargs["all_reduce_params"],
                mapping,
                mapping,
                self.layer_idx,
                None,
            )

    class AddAttn:
        def _add_after_rmsnorm_and_norm(self, res, branch, eps, next_norm, gain=None):
            assert gain is None
            updated = res + branch.float() * 0.75 + eps
            return updated, next_norm(updated)

    class AddFfn:
        def add_after_rmsnorm(self, res, branch, eps, gain):
            return res + branch.float() * gain + eps

    def feed_forward(x, attn_metadata, sp_num_tokens=None):
        moe_calls.append((sp_num_tokens, step.active, attn_metadata))
        return (x.float() * 3 - 1).to(torch.bfloat16)

    layer = types.SimpleNamespace(
        layer_idx=61,
        mapping=mapping,
        self_attn=Attention(),
        add_attn=AddAttn(),
        add_ffn=AddFfn(),
        feed_forward=feed_forward,
        pre_ffn_norm=lambda h: (h * 0.5).to(torch.bfloat16),
        post_attn_norm=types.SimpleNamespace(variance_epsilon=1e-6),
        post_ffn_norm=types.SimpleNamespace(variance_epsilon=1e-5, gain=lambda: gain),
        nvtx=lambda name: contextlib.nullcontext(),
        _overlap_sp_projection_enabled=False,
        _pad_sp_projection_enabled=False,
        _sp_attn_input_gather=lambda attn_input, n: attn_input,
    )
    model = types.SimpleNamespace(
        layers=[layer], _sp_mapping=mapping, dtype=torch.bfloat16, norm=lambda h: h * 2
    )
    meta = _depth_metadata([300000], 1)
    meta.all_rank_num_tokens = None

    hidden = builtin.MuseSparkOpenDecoderLayer.forward(
        layer,
        position_ids=None,
        hidden_states=residual.clone(),
        attn_metadata=meta,
        precomputed_attn_input=torch.zeros(shard, hidden_size, dtype=torch.bfloat16),
        next_layer_norm=None,
        sp_num_tokens=rows,
    )
    reference = allgather(model.norm(hidden.to(model.dtype)), mapping, dim=0)[:rows]

    monkeypatch.setattr(
        tail_path, "get_model_extra_attrs", lambda: {"attention_metadata": lambda: meta}
    )
    tail = tail_path.LastLayerTail(model, None, tail_path.TailState(), max_depth=262144, step=step)
    output = torch.zeros(rows, hidden_size, dtype=torch.bfloat16)
    tail._arm()
    assert tail._unchanged
    tail._tail(attention_rows, residual.clone(), output)
    assert torch.equal(output, reference)
    assert moe_calls[0][0] == moe_calls[1][0] == rows
    assert moe_calls[1][1] is True and not step.active
    assert moe_calls[1][2] is meta
