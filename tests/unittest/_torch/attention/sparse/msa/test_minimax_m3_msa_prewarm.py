# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the MiniMax-M3 MSA prewarm planning (m3ax_msa_prewarm)."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.m3ax_msa_prewarm import (
    core,
    hook,
    sdwarm,
)

# q / output rows of 32 heads x 128 bf16, top-k rows of 16 int32, int32 seq_lens, a 4096-wide int32 block table.
_ROW_BYTES = (32 * 128 * 2, 32 * 128 * 2, 16 * 4, 4, 4096 * 4)


def _sig(t, r, row_bytes):
    q, out, topk, seq, bt = row_bytes
    return tuple((n % 16) == 0 for n in (t * q, t * out, t * topk, r * seq, r * bt))


@pytest.mark.parametrize("row_bytes", [_ROW_BYTES, (3, 5, 6, 4, 12)])
@pytest.mark.parametrize("b,max_batch", [(1, 1), (1, 8), (5, 8), (8, 8)])
def test_plan_variants_covers_every_reachable_signature(row_bytes, b, max_batch):
    planned = sdwarm.plan_variants(b, max_batch, *row_bytes)
    sigs = [sig for sig, _, _ in planned]
    assert len(sigs) == len(set(sigs))
    assert planned[0] == (_sig(0, 0, row_bytes), 0, 0)
    for r in range(1, max_batch - b + 1):
        for t in range(r, r + 64):
            assert _sig(t, r, row_bytes) in sigs


def test_plan_enumerates_every_local_batch():
    batches = {b for b, _, _, _ in sdwarm.plan(16, _ROW_BYTES)}
    assert batches == set(range(1, 17))


def test_row_outermost():
    assert sdwarm.row_outermost((10, 32, 128), (4096, 128, 1), 0)
    # Head-major top-k view [kv_heads, tokens, topk] of a token-major buffer.
    assert sdwarm.row_outermost((2, 10, 16), (16, 32, 1), 1)
    assert not sdwarm.row_outermost((2, 10, 16), (160, 16, 1), 1)


def _runner(spec, draft_len=0, mapping=None):
    cfg = SimpleNamespace(is_spec_decode=spec, original_max_draft_len=draft_len)
    return SimpleNamespace(_config=cfg, _dynamic_draft_len_mapping=mapping)


@pytest.mark.parametrize(
    "runner,expected",
    [
        (_runner(False), set()),
        (_runner(True, 3), {4}),
        (_runner(True, 7), {8}),
        (_runner(True, 4, {1: 4, 8: 2, 64: 0}), {5, 3, 1}),
    ],
)
def test_decode_query_lens(runner, expected):
    assert sdwarm.decode_query_lens(runner) == expected


def test_families_drop_dense_and_cover_proxy_split():
    fams = {f["family"]: f for f in core.families_for(2, 0, 128)}
    assert set(fams) == {"proxy", "proxy-split"}
    assert fams["proxy"]["split"] is False and fams["proxy-split"]["split"] is True
    assert fams["proxy"]["hq"] == 2 and fams["proxy"]["hkv"] == 1


def test_hook_disabled_is_a_no_op(monkeypatch):
    monkeypatch.setattr(hook, "ENABLED", False)
    monkeypatch.setattr(core, "prewarm_for_runner", pytest.fail)
    monkeypatch.setattr(core, "set_phase", pytest.fail)
    hook.before_msa_indexer_prewarm(object(), object())
    with hook.warmup_scope(object()):
        pass


def test_warmup_scope_propagates_and_sets_serve(monkeypatch):
    monkeypatch.setattr(hook, "ENABLED", True)
    finished = []
    monkeypatch.setattr(sdwarm, "finish", lambda runner, run: finished.append(run))
    with pytest.raises(RuntimeError):
        with hook.warmup_scope(object()):
            assert core._PHASE[0] == "warmup"
            raise RuntimeError("warmup failed")
    assert finished == [False]
    assert core._PHASE[0] == "serve"
