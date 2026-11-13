# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A model's opt-ins on the DFlash worker (the ``enable_*`` methods) and the speculative worker's override hooks.

Every opt-in is off on a freshly built worker, an installed op is the one the stock step then hands its operands to,
and a declining op (None / False) leaves the stock lines to run. The opt-ins that configure the draft model touch only
a draft model that reads them. CPU only.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.cpu_only

TAG = "[test] int32 accept counts taken:"


def _worker(**attrs) -> DFlashWorker:
    worker = DFlashWorker.__new__(DFlashWorker)
    worker._lean_glue = False
    worker._fused_ctx_store = None
    worker._fused_prefill_store = None
    worker._fused_block_refresh = None
    worker._fused_drafter_inputs = None
    worker._fused_ctx_advance = None
    worker._fused_kv_rewind = None
    worker._int32_accept_counts = None
    worker.mapping = SimpleNamespace(tp_size=2, tp_rank=0, enable_attention_dp=False)
    for name, value in attrs.items():
        setattr(worker, name, value)
    return worker


def test_class_level_opt_ins_are_off():
    assert DFlashWorker._full_slot_arena is False
    assert DFlashWorker._disagg_draft_kv_engaged is False
    assert DFlashWorker._disagg_draft_ctx_seeds is None
    assert SpecWorkerBase._draft_logits_gather_op is None
    assert SpecWorkerBase._block_tokens_keep_int32 is False


def test_full_slot_arena():
    worker = _worker()
    worker.enable_full_slot_arena()
    assert worker._full_slot_arena is True
    assert DFlashWorker._full_slot_arena is False


@pytest.mark.parametrize(
    "method, attr",
    [
        ("enable_fused_ctx_store", "_fused_ctx_store"),
        ("enable_fused_prefill_store", "_fused_prefill_store"),
        ("enable_fused_ctx_advance", "_fused_ctx_advance"),
        ("enable_fused_drafter_inputs", "_fused_drafter_inputs"),
        ("enable_fused_kv_rewind", "_fused_kv_rewind"),
        ("enable_fused_block_refresh", "_fused_block_refresh"),
    ],
)
def test_worker_ops_install_and_remove(method, attr):
    worker = _worker()

    def op(*args, **kwargs):
        return None

    assert getattr(worker, method)(op) is True and getattr(worker, attr) is op
    assert getattr(worker, method)(None) is False and getattr(worker, attr) is None


def test_int32_accept_counts_need_both_fused_ops():
    """Taken only with the fused store and advance installed (a stock path converts on first use)."""

    def store(*args, **kwargs):
        return True

    def advance(*args):
        return None

    worker = _worker(_fused_ctx_store=store, _fused_ctx_advance=advance)
    assert worker.enable_int32_accept_counts(TAG) is True and worker._int32_accept_counts == TAG
    for has_store, has_advance in ((False, True), (True, False), (False, False)):
        worker = _worker(
            _fused_ctx_store=store if has_store else None,
            _fused_ctx_advance=advance if has_advance else None,
        )
        assert worker.enable_int32_accept_counts(TAG) is False
        assert worker._int32_accept_counts is None


def test_a_none_tag_disarms_the_int32_accept_counts():
    worker = _worker(_fused_ctx_store=lambda *a, **k: True, _fused_ctx_advance=lambda *a: None)
    worker.enable_int32_accept_counts(TAG)
    assert worker.enable_int32_accept_counts(None) is False and worker._int32_accept_counts is None


@pytest.mark.parametrize(
    "method, attr, value",
    [
        ("enable_fused_forward_indices", "_dflash_fused_indices", lambda *a: None),
        ("enable_fused_qk_norm_rope", "_dflash_fused_qk_norm_rope", lambda *a: None),
        ("enable_fused_qk_norm_rope_append", "_dflash_fused_qk_norm_rope_append", lambda *a: None),
        ("enable_attention_pdl", "_dflash_attn_pdl", True),
    ],
)
def test_draft_model_opt_ins_touch_only_a_draft_model_that_reads_them(method, attr, value):
    plain = SimpleNamespace()
    assert getattr(_worker(), method)(plain, value) is False
    assert not hasattr(plain, attr)
    reader = SimpleNamespace(**{attr: None})
    assert getattr(_worker(), method)(reader, value) is True
    assert getattr(reader, attr) is value


def test_window_capped_decode_and_lean_glue():
    plain = SimpleNamespace()
    worker = _worker()
    worker.enable_window_capped_decode(plain, 4)
    worker.enable_lean_glue(plain, True)
    assert worker._lean_glue is True and vars(plain) == {}
    reader = SimpleNamespace(_dflash_window_cap_max_batch=0, _dflash_lean_glue=False)
    worker.enable_window_capped_decode(reader, 4)
    worker.enable_lean_glue(reader, True)
    assert reader._dflash_window_cap_max_batch == 4 and reader._dflash_lean_glue is True
    worker.enable_lean_glue(reader, False)
    assert worker._lean_glue is False and reader._dflash_lean_glue is False


def test_fused_ar_norm_needs_a_draft_model_that_reads_it():
    worker = _worker()
    assert worker.enable_fused_ar_norm(SimpleNamespace()) is False
    assert worker.enable_fused_ar_norm_op(SimpleNamespace(), lambda *a: None) is False
    # A reader without layers (or with TP 1) stays off.
    reader = SimpleNamespace(_dflash_fused_ar_norm=True, _dflash_fused_ar_norm_op=None)
    assert worker.enable_fused_ar_norm(reader) is False and reader._dflash_fused_ar_norm is False
    # The op is installed only while the fused AR + norm is on.
    assert worker.enable_fused_ar_norm_op(reader, lambda *a: None) is False
    assert reader._dflash_fused_ar_norm_op is None


def test_fc_tp_split_needs_a_draft_model_that_reads_it():
    fc = SimpleNamespace(weight=torch.zeros(4, 8), bias=None)
    plain = SimpleNamespace(fc=fc)
    assert _worker().enable_fc_tp_split(plain) is False
    assert not hasattr(plain, "_dflash_fc_split")
    reader = SimpleNamespace(fc=fc, _dflash_fc_split="stale")
    assert _worker().enable_fc_tp_split(reader, enabled=False) is False
    assert reader._dflash_fc_split is None
    single = _worker(mapping=SimpleNamespace(tp_size=1, tp_rank=0))
    assert single.enable_fc_tp_split(reader) is False


def _rewind_metadata(num_seqs):
    calls = []
    return SimpleNamespace(
        kv_lens_cuda=torch.arange(10, 10 + num_seqs, dtype=torch.int32),
        update_for_spec_dec=lambda: calls.append("update"),
        calls=calls,
    )


def test_fused_kv_rewind_matches_the_stock_lines():
    num_accepted = torch.tensor([1, 3, 2], dtype=torch.int32)
    stock = _worker()
    stock_meta = _rewind_metadata(3)
    stock._prepare_kv_for_draft_forward(stock_meta, num_accepted, 1, 3)

    def rewind(counts, kv_lens, nc, bs):
        kv_lens[nc:bs] += 1
        return 1 - counts[nc:bs]

    fused = _worker(_fused_kv_rewind=rewind)
    fused_meta = _rewind_metadata(3)
    fused._prepare_kv_for_draft_forward(fused_meta, num_accepted, 1, 3)
    assert torch.equal(fused_meta.kv_lens_cuda, stock_meta.kv_lens_cuda)
    assert torch.equal(fused._kv_rewind_amount, stock._kv_rewind_amount)
    assert (fused._kv_rewind_nc, fused._kv_rewind_bs) == (1, 3)
    assert fused_meta.calls == stock_meta.calls == ["update"]


def test_a_declining_kv_rewind_runs_the_stock_lines():
    num_accepted = torch.tensor([2, 2], dtype=torch.int32)
    worker = _worker(_fused_kv_rewind=lambda *args: None)
    meta = _rewind_metadata(2)
    worker._prepare_kv_for_draft_forward(meta, num_accepted, 0, 2)
    assert meta.kv_lens_cuda.tolist() == [11, 12]
    assert worker._kv_rewind_amount.tolist() == [-1, -1]


def _refresh_worker(full_rows, block_offset=0):
    seen = []

    def refresh(src, pool_idx, num_seqs, divisor, tables, counts):
        seen.append((pool_idx, num_seqs, divisor))
        return True

    worker = _worker(
        _fused_block_refresh=refresh,
        _ctx_block_tables=torch.zeros(4, 8, dtype=torch.int32),
        _ctx_block_counts=torch.zeros(4, dtype=torch.long),
        _ctx_unified_cache=False,
        _ctx_pool_idx=0,
        _ctx_block_divisor=2,
        _ctx_block_offset=block_offset,
        _ctx_counts_full_row=full_rows,
        _ctx_table_width=0,
    )
    return worker, seen


def test_fused_block_refresh_is_taken_only_where_every_row_counts_full():
    worker, seen = _refresh_worker(full_rows=True)
    meta = SimpleNamespace(draft_kv_cache_block_offsets=torch.zeros(1, 4, 2, 8, dtype=torch.int32))
    assert worker._refresh_ctx_block_tables(meta, 2) is True
    assert seen == [(0, 2, 2)] and worker._ctx_table_width == 8


@pytest.mark.parametrize("full_rows, block_offset", [(False, 0), (True, 3)])
def test_fused_block_refresh_is_not_taken_otherwise(monkeypatch, full_rows, block_offset):
    from tensorrt_llm._torch.speculative import dflash

    decoded = []
    monkeypatch.setattr(
        dflash, "decode_draft_block_table", lambda *args, **kwargs: decoded.append(kwargs)
    )
    worker, seen = _refresh_worker(full_rows=full_rows, block_offset=block_offset)
    meta = SimpleNamespace(draft_kv_cache_block_offsets=torch.zeros(1, 4, 2, 8, dtype=torch.int32))
    assert worker._refresh_ctx_block_tables(meta, 2) is True
    assert seen == [] and decoded == [{"block_offset": block_offset}]


def test_draft_logits_gather_op():
    class _Gathering(SpecWorkerBase):
        def _forward_impl(self, *args, **kwargs):  # pragma: no cover - never run
            raise NotImplementedError

        @property
        def max_draft_len(self) -> int:
            return 1

    worker = object.__new__(_Gathering)
    worker.mapping = SimpleNamespace(tp_size=2, enable_attention_dp=False)
    meta = SimpleNamespace(is_all_greedy_sample=False, draft_vocab_size=8, vocab_size=8)
    shard = torch.arange(8, dtype=torch.float32).view(2, 4)
    full = torch.cat([shard, shard + 100], dim=-1)
    assert worker.enable_draft_logits_gather_op(lambda logits: full) is True
    assert worker.maybe_gather_sharded_draft_logits(shard, meta) is full
    # A replicated full-vocab row is not gathered at all.
    assert worker.maybe_gather_sharded_draft_logits(full, meta) is full
    assert worker.enable_draft_logits_gather_op(None) is False
    assert worker._draft_logits_gather_op is None


def test_chain_speculative_sampling_override(monkeypatch):
    monkeypatch.setattr(interface, "_CHAIN_SPEC_OVERRIDE", None)
    out = (torch.zeros(1, 3, dtype=torch.int32), torch.ones(1, dtype=torch.int32))
    calls = []

    def override(*args):
        calls.append(args)
        return out

    interface.set_chain_speculative_sampling_override(override)
    args = (torch.zeros(1, 2, 4), torch.zeros(1, 2, dtype=torch.int32), torch.zeros(1, 3, 4))
    seed, offset = torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=torch.int64)
    assert interface.rejection_sampling_one_model(*args, seed=seed, offset=offset) is out
    assert len(calls) == 1 and calls[0][3] is seed and calls[0][4] is offset
    interface.set_chain_speculative_sampling_override(None)
    assert interface._CHAIN_SPEC_OVERRIDE is None
    slot_override = object()
    interface.set_chain_speculative_sampling_slot_override(slot_override)
    assert interface._CHAIN_SPEC_SLOT_OVERRIDE is slot_override
    interface.set_chain_speculative_sampling_slot_override(None)
    assert interface._CHAIN_SPEC_SLOT_OVERRIDE is None
