# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decoder bounded replay planner entry (dep2): the engine's plan against the restated kept-row rule, on a minimal
context-step metadata outside attention DP (host planning, exact)."""

from types import SimpleNamespace

import pytest
import torch
from _v41_util import DEV

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (  # noqa: E501
    csa2_bounded_replay_tokens,
    csa2_decoder_replay_enabled,
    csa2_decoder_replay_plan,
    csa2_requires_full_decoder_prefill,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the plan's rows go to the device"
)


def _metadata(spec):
    ctx, gen = spec["ctx"], spec["gen"]
    lens = [n for n, _, _ in ctx] + [4] * gen
    cached = [c for _, c, _ in ctx] + [100] * gen
    ends = tuple(e for _, _, e in ctx) + (0,) * gen
    seq = torch.tensor(lens, dtype=torch.int32)
    return SimpleNamespace(
        is_cuda_graph=False,
        mapping=None,
        padded_num_tokens=None,
        num_tokens=sum(lens),
        num_contexts=len(ctx),
        num_chunked_ctx_requests=0,
        kv_cache_manager=SimpleNamespace(enable_block_reuse=False),
        seq_lens=seq,
        seq_lens_cuda=seq.to(DEV),
        num_seqs=len(lens),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=cached),
        decoder_context_ends=ends,
        prompt_lens=[e or 1000 for e in ends],
    )


@pytest.mark.parametrize(
    "cell", csa2_decoder_replay_plan.CELLS, ids=[c.why[:44] for c in csa2_decoder_replay_plan.CELLS]
)
def test_cells(cell, certifying):  # noqa: F811
    certifying(csa2_decoder_replay_plan)
    md = _metadata(cell.spec)
    got = csa2_decoder_replay_plan(md, cell.spec["window"], None)
    want = csa2_decoder_replay_plan.reference(md, cell.spec["window"], None)
    if want is None:
        assert got is None
        return
    rows, kept, floors = want
    assert (
        torch.equal(got.rows.cpu(), rows)
        and got.replay_seq_lens.tolist() == kept
        and got.swa_floors == floors
    )
    assert got.num_encoder_tokens == md.num_tokens


def test_zero_window_plans_nothing():
    """No window disables the replay: the engine's planner and the reference both return None (the reference used to
    plan a zero-row replay of a 600-row prompt)."""
    md = _metadata(csa2_decoder_replay_plan.CELLS[0].spec)
    assert csa2_decoder_replay_plan(md, 0, None) is None
    assert csa2_decoder_replay_plan.reference(md, 0, None) is None


def test_switches_and_predicates(monkeypatch):
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    assert not csa2_decoder_replay_enabled()
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    assert csa2_decoder_replay_enabled()
    monkeypatch.setenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", "200")
    assert csa2_bounded_replay_tokens(128) == 200
    assert csa2_requires_full_decoder_prefill(SimpleNamespace(py_return_context_logits=True))
    assert not csa2_requires_full_decoder_prefill(SimpleNamespace())


def test_encoder_recovery_rows_and_consumption():
    """`csa2_encoder_recovery` reads the context requests' Encoder recovery as the engine's metadata does: an unconsumed
    replay gives its Global end (historical rows below it) and its start as the SWA floor; a consumed one keeps only
    the floor; a checkpoint or no plan gives neither. `csa2_complete_encoder_replay`
    consumes the replays only."""
    from types import SimpleNamespace as NS

    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
        csa2_complete_encoder_replay,
        csa2_encoder_recovery,
        csa2_encoder_replay_tokens,
    )
    from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderCheckpoint, EncoderReplay

    cold = EncoderReplay(request_id=1, cache_identity=0, global_end=4096, start=3956)
    warm = EncoderReplay(request_id=2, cache_identity=0, global_end=8192, start=8052)
    done = EncoderReplay(request_id=3, cache_identity=0, global_end=100, start=0, consumed=True)
    ckpt = EncoderCheckpoint(request_id=4, cache_identity=0, global_end=512)
    reqs = [NS(py_ced_replay=p) for p in (cold, warm, done, ckpt, None)]
    assert csa2_encoder_recovery(reqs, 6) == ([4096, 8192, 0, 0, 0, 0], [3956, 8052, 0, 0, 0, 0])
    assert csa2_encoder_recovery(reqs[3:], 2) is None and csa2_encoder_recovery(None, 2) is None
    assert [csa2_encoder_replay_tokens(r) for r in reqs] == [140, 140, 0, 0, 0]
    csa2_complete_encoder_replay(reqs)
    assert cold.consumed and warm.consumed and csa2_encoder_replay_tokens(reqs[0]) == 0
    assert csa2_encoder_recovery(reqs, 5) == ([0] * 5, [3956, 8052, 0, 0, 0])


def _policy_inputs(**kw):
    """A released-checkpoint text config and a model config without speculation (overrides in `kw`)."""
    cfg = SimpleNamespace(
        kv_source_layer_ids=[2, 8, 14, 20],
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        engram_layer_ids=[1, 14],
        sliding_window=128,
        num_hidden_layers=40,
    )
    mc = SimpleNamespace(
        mapping=SimpleNamespace(pp_size=1, cp_size=1, enable_attention_dp=True),
        spec_config=None,
        extra_attrs={},
    )
    for k, v in kw.items():
        setattr(cfg if hasattr(cfg, k) else mc, k, v)
    return cfg, mc


def _dspark(embedded=True, layers=(1, 20, 39)):
    mode = SimpleNamespace(is_dspark=lambda: True)
    return SimpleNamespace(
        spec_dec_mode=mode, draft_is_embedded_in_target=embedded, target_layer_ids=list(layers)
    )


def test_replay_policy_follows_stock(monkeypatch):
    """`csa2_replay_policy` is stock's `_resolve_replay_policy` on this layout: layer 20 and a 140-token window by
    default, the engine's switches honoured, and full prefill (None, 0) where stock refuses."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
        csa2_replay_policy,
    )

    monkeypatch.delenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", raising=False)
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    assert csa2_replay_policy(*_policy_inputs(), captures=True) == (20, 140)
    assert csa2_replay_policy(*_policy_inputs(), captures=False) == (20, 140)
    monkeypatch.setenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS", "256")
    assert csa2_replay_policy(*_policy_inputs(), captures=True) == (20, 256)
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    assert csa2_replay_policy(*_policy_inputs(), captures=True) == (None, 0)
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.delenv("TRTLLM_V41_BOUNDED_REPLAY_TOKENS")
    # embedded DSpark: replayed with its capture contract (the target writes the captures' row map), refused otherwise
    assert csa2_replay_policy(*_policy_inputs(spec_config=_dspark()), captures=True) == (20, 140)
    refused = [
        _policy_inputs(spec_config=_dspark()) + (False,),
        _policy_inputs(spec_config=_dspark(embedded=False)) + (True,),
        _policy_inputs(spec_config=_dspark(layers=())) + (True,),
        _policy_inputs(mapping=SimpleNamespace(pp_size=2, cp_size=1, enable_attention_dp=True))
        + (True,),
        _policy_inputs(extra_attrs={"bounded_replay_on_generation": True}) + (True,),
        _policy_inputs(engram_layer_ids=[1, 21]) + (True,),
        _policy_inputs(compress_ratios=[0, 0] + [2] * 38) + (True,),
        _policy_inputs(kv_source_layer_ids=[]) + (True,),
    ]
    for cfg, mc, captures in refused:
        assert csa2_replay_policy(cfg, mc, captures=captures) == (None, 0)


def _model(window=140):
    return SimpleNamespace(
        decoder_replay_split=20, decoder_replay_window=window, ced_kv_precompute=True
    )


def test_step_plan_refusals():
    """`csa2_step_replay_plan` / `csa2_replay_plan` fall back to every row exactly where stock's `_plan_bounded_replay`
    / `_make_bounded_replay_plan` do: a caller needing every row's state (context logits), block reuse without private
    decoder pages, a CUDA-graph step and a per-request prefill graph under attention DP, and a model without a split;
    and they reject a plan whose exchanged counts are missing or stale, or that compacts rows a caller still needs."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
        csa2_replay_plan,
        csa2_step_replay_plan,
    )
    from tensorrt_llm._torch.utils import (
        get_per_request_prefill_cuda_graph_flag,
        set_per_request_prefill_cuda_graph_flag,
    )

    spec = dict(window=140, ctx=((600, 0, 600), (300, 0, 300)), gen=1)
    md = _metadata(spec)
    model = _model()
    # without attention DP the plan is made in the forward: compacting, unless every row's state is needed
    plan = csa2_step_replay_plan(model, md, False, None)
    assert (
        plan is not None
        and plan.replays_local_tokens
        and plan.replay_seq_lens.tolist() == [140, 140, 4]
    )
    assert (
        csa2_step_replay_plan(model, md, True, None) is None
    )  # context logits: every row (stock's conservative default)
    assert (
        csa2_step_replay_plan(SimpleNamespace(decoder_replay_split=None), md, False, None) is None
    )
    reuse = _metadata(spec)
    reuse.kv_cache_manager = SimpleNamespace(enable_block_reuse=True)
    assert (
        csa2_replay_plan(model, reuse, False, None) is None
    )  # reused window pages without private decoder pages
    full = SimpleNamespace(py_return_context_logits=True, prompt_len=600)
    assert csa2_replay_plan(
        model, md, False, [full, SimpleNamespace(prompt_len=300)]
    ).replay_seq_lens.tolist() == [600, 140, 4]
    # attention DP: the plan prepare_adp_inputs made, checked against the exchanged counts (the engine clears the
    # per-request prefill graph flag, on by default, for a batch that runs no prefill graph)
    flag = get_per_request_prefill_cuda_graph_flag()
    set_per_request_prefill_cuda_graph_flag(False)
    try:
        _check_adp_plan(model, spec, set_per_request_prefill_cuda_graph_flag)
    finally:
        set_per_request_prefill_cuda_graph_flag(flag)


def _check_adp_plan(model, spec, set_flag):
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
        csa2_replay_plan,
        csa2_step_replay_plan,
    )

    adp = _metadata(spec)
    adp.mapping, adp.tp_rank = SimpleNamespace(enable_attention_dp=True, tp_rank=0), 0
    adp.all_rank_num_tokens = [adp.num_tokens, 7]
    adp.decoder_replay_plan = csa2_replay_plan(model, adp, False, None)
    with pytest.raises(ValueError, match="Exchange ADP token counts"):
        csa2_step_replay_plan(model, adp, False, None)
    adp.decoder_replay_plan.replay_all_rank_num_tokens = [284, 7]
    adp.decoder_replay_plan.saved_all_rank_num_tokens = list(adp.all_rank_num_tokens)
    assert csa2_step_replay_plan(model, adp, False, None) is adp.decoder_replay_plan
    with pytest.raises(ValueError, match="Full token states"):
        csa2_step_replay_plan(model, adp, True, None)
    adp.all_rank_num_tokens = [adp.num_tokens + 1, 7]
    with pytest.raises(ValueError, match="changed after exchanging"):
        csa2_step_replay_plan(model, adp, False, None)
    adp.is_cuda_graph = True
    assert csa2_step_replay_plan(model, adp, False, None) is None
    adp.is_cuda_graph = False
    set_flag(True)
    assert csa2_step_replay_plan(model, adp, False, None) is None
