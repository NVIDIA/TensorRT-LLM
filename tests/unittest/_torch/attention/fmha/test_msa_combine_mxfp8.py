# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 CTX: when the NVFP4 sparse prefill's combine writes the o_proj input as MXFP8 (msa_combine_mxfp8)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import msa_combine_mxfp8 as m
from tensorrt_llm._torch.modules import linear

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _one_stream(monkeypatch):
    # The slot records its producing stream; on CPU tensors every set and take is on one stream.
    stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


class _Sparse:
    def sparse_atten_nvfp4_kv_func(self, q, *, out=None, o_fp8=None, sf_out=None, write_bf16=True):
        pass


class _OldSparse:
    def sparse_atten_nvfp4_kv_func(self, q, *, out=None):
        pass


def _out(rows=200, heads=32):
    return torch.zeros((rows, heads, 128), dtype=torch.bfloat16)


CTX = SimpleNamespace(num_generations=0)


def test_engages_only_when_armed_pure_context_in_place_and_supported(monkeypatch):
    monkeypatch.setattr(m, "ENABLED", True)
    out = _out()
    assert m.combine_kwargs(_Sparse(), out, CTX, in_place=True) is None, "not armed"
    with m.armed():
        kw = m.combine_kwargs(_Sparse(), out, CTX, in_place=True)
        assert kw is not None and kw["write_bf16"] is True
        assert kw["o_fp8"].shape == (200, 32 * 128) and kw["o_fp8"].dtype == torch.float8_e4m3fn
        assert kw["sf_out"].numel() == 256 * 32 * 128 // 32
        assert int(kw["sf_out"][-128 * 32 * 128 // 32 :].sum()) == 0, "padding tile zeroed"
        assert (
            m.combine_kwargs(_Sparse(), out, SimpleNamespace(num_generations=3), in_place=True)
            is None
        )
        assert m.combine_kwargs(_Sparse(), out, CTX, in_place=False) is None
        assert m.combine_kwargs(_OldSparse(), out, CTX, in_place=True) is None
        assert m.combine_kwargs(_Sparse(), out.view(200, -1), CTX, in_place=True) is None
        assert m.combine_kwargs(_Sparse(), _out().float(), CTX, in_place=True) is None
    assert m.combine_kwargs(_Sparse(), out, CTX, in_place=True) is None, "disarmed on exit"


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(m, "ENABLED", False)
    with m.armed():
        assert m.combine_kwargs(_Sparse(), _out(), CTX, in_place=True) is None


class _Log:
    def __init__(self):
        self.info, self.warning = [], []

    def info_once(self, msg, key=None):
        self.info.append(msg)

    def warning_once(self, msg, key=None):
        self.warning.append(msg)


def _publish(monkeypatch):
    monkeypatch.setattr(m, "ENABLED", True)
    log = _Log()
    monkeypatch.setattr(m, "logger", log)
    out = _out()
    with m.armed():
        kw = m.combine_kwargs(_Sparse(), out, CTX, in_place=True)
    m.publish(out, kw)
    return out, kw, log


def test_engagement_is_logged_only_when_the_o_proj_takes_the_pair(monkeypatch):
    out, kw, log = _publish(monkeypatch)
    assert not log.info, "publishing alone is not engagement"
    o = out.view(200, -1)  # the o_proj input: same storage and shape
    act, sf = linear.take_prequantized_mxfp8(o)
    assert act.data_ptr() == kw["o_fp8"].data_ptr() and sf is kw["sf_out"]
    m.after_o_proj(o)
    assert linear._PREQUANTIZED_MXFP8[0] is None
    assert log.info == ["[M3 CTX] combine MXFP8 o_proj input taken"] and not log.warning


def test_a_take_on_mismatched_rows_is_a_miss(monkeypatch):
    out, _, log = _publish(monkeypatch)
    padded = torch.zeros((256, 32 * 128), dtype=torch.bfloat16)  # e.g. a padded o_proj input
    assert linear.take_prequantized_mxfp8(padded) is None, "dropped, the o_proj quantizes"
    m.after_o_proj(padded)
    assert not log.info and len(log.warning) == 1 and "not consumed" in log.warning[0]


def test_a_pair_the_o_proj_never_looked_at_is_cleared_and_a_miss(monkeypatch):
    _, _, log = _publish(monkeypatch)
    m.after_o_proj()
    assert linear._PREQUANTIZED_MXFP8[0] is None
    assert not log.info and len(log.warning) == 1


def test_slot_audit_counts_takes_drops_unconsumed_and_declines(monkeypatch):
    monkeypatch.setattr(linear, "_PREQUANTIZED_MXFP8_STATS", {})
    monkeypatch.setattr(m, "ENABLED", True)
    monkeypatch.setattr(m, "logger", _Log())
    out = _out()
    o = out.view(200, -1)
    for take_input in (o, torch.zeros((256, 32 * 128), dtype=torch.bfloat16), None):
        with m.armed("general"):
            kw = m.combine_kwargs(_Sparse(), out, CTX, in_place=True)
            m.publish(out, kw)
        if take_input is not None:
            linear.take_prequantized_mxfp8(take_input)
        m.after_o_proj(take_input)
    with m.armed("general"):
        assert (
            m.combine_kwargs(_Sparse(), out, SimpleNamespace(num_generations=1), in_place=True)
            is None
        )
        assert m.combine_kwargs(_Sparse(), out, CTX, in_place=False) is None
    stats = linear._PREQUANTIZED_MXFP8_STATS
    assert stats["combine_general"] == [3, 1, 1, 1]  # set, taken, dropped, unconsumed
    assert stats["combine_general_declined_mixed_step"][0] == 1
    assert stats["combine_general_declined_not_in_place"][0] == 1


def _mxfp8_o_proj(use_cutlass=True, lora=None):
    method = object.__new__(linear.MXFP8LinearMethod)
    method.use_cutlass = use_cutlass
    return SimpleNamespace(quant_method=method, lora=lora)


def test_o_proj_takes_slot_only_for_an_mxfp8_cutlass_linear():
    assert m.o_proj_takes_slot(_mxfp8_o_proj())
    assert not m.o_proj_takes_slot(_mxfp8_o_proj(use_cutlass=False))
    assert m.o_proj_takes_slot(_mxfp8_o_proj(lora=object())), (
        "a LoRA layer reads only when lora_params are passed"
    )
    assert not m.o_proj_takes_slot(SimpleNamespace(quant_method=object(), lora=None))


def test_bf16_skip_only_when_enabled_and_proven(monkeypatch):
    monkeypatch.setattr(m, "ENABLED", True)
    out = _out()
    for skip, ok, expect_write in ((False, True, True), (True, False, True), (True, True, False)):
        monkeypatch.setattr(m, "SKIP_BF16", skip)
        with m.armed("general", skip_bf16_ok=ok):
            assert (
                m.combine_kwargs(_Sparse(), out, CTX, in_place=True)["write_bf16"] is expect_write
            )


def test_bf16_skipped_pair_not_taken_raises(monkeypatch):
    monkeypatch.setattr(m, "ENABLED", True)
    monkeypatch.setattr(m, "SKIP_BF16", True)
    monkeypatch.setattr(m, "logger", _Log())
    out = _out()
    with m.armed("general", skip_bf16_ok=True):
        kw = m.combine_kwargs(_Sparse(), out, CTX, in_place=True)
        m.publish(out, kw)
    with pytest.raises(RuntimeError, match="bf16 input was not written"):
        m.after_o_proj(torch.zeros((256, 32 * 128), dtype=torch.bfloat16))
    # Taken: no raise.
    with m.armed("general", skip_bf16_ok=True):
        kw = m.combine_kwargs(_Sparse(), out, CTX, in_place=True)
        m.publish(out, kw)
    o = out.view(200, -1)
    assert linear.take_prequantized_mxfp8(o) is not None
    m.after_o_proj(o)
    assert "[M3 CTX] combine MXFP8 o_proj input taken (bf16 store skipped)" in m.logger.info
