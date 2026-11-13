# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F1's routing in the tp8 target's S4 worker, without GPUs or a model: a rejection-sampled draft block arms the
renorm's row store with the block's slot rows (only while padding rows keep the renorm's output) and always disarms
it; the store after the sampler is skipped only when the renorm reports those rows stored, else it stores as before.
The renorm's values are certified by sampling/test_modeling_v2_m31_fold_f1_gpu.py."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark_markov_worker,
    )
except NoTarget as no_target:
    pytest.skip(str(no_target), allow_module_level=True)
from tensorrt_llm._torch.speculative import interface

pytestmark = pytest.mark.cpu_only

K = 7


class _Base:
    def advanced_sample_draft_block(self, gen_logits, spec_metadata, num_contexts, batch_size):
        self.seen = dspark_markov_worker._topp_det._ROW_STORE[0]
        if getattr(self, "fail", False):
            raise RuntimeError("sampler failed")
        return "tokens"


class _Worker(dspark_markov_worker.S4WorkerMixin, _Base):
    pass


def _meta(**kw):
    base = dict(
        use_rejection_sampling=True,
        draft_probs=torch.zeros(4, K, 16),
        batch_slot_ids=torch.tensor([3, 1, 2], dtype=torch.int64),
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def worker(monkeypatch):
    # The engine attribute the worker reads with a False default (absent on this engine).
    monkeypatch.setattr(interface, "_CHEAP_PADDING_ROWS", False, raising=False)
    monkeypatch.setattr(dspark_markov_worker, "FOLD_F1", True)
    dspark_markov_worker._topp_det.disarm_row_store()
    return _Worker.__new__(_Worker)


def test_a_rejection_block_arms_the_renorm_with_its_slot_rows(worker):
    meta = _meta()
    assert worker.advanced_sample_draft_block(torch.zeros(2, K, 16), meta, 1, 3) == "tokens"
    dst, slot_ids, k = worker.seen
    assert dst is meta.draft_probs and k == K
    assert slot_ids.data_ptr() == meta.batch_slot_ids[1:].data_ptr() and slot_ids.numel() == 2
    topp_det = dspark_markov_worker._topp_det
    assert topp_det._ROW_STORE[0] is None and topp_det._ROWS_STORED[0] is None


@pytest.mark.parametrize(
    "case", ["off", "cheap padding", "no rejection", "no draft probs", "no slot ids", "2-D logits"]
)
def test_other_blocks_are_not_armed(worker, monkeypatch, case):
    if case == "off":
        monkeypatch.setattr(dspark_markov_worker, "FOLD_F1", False)
    if case == "cheap padding":
        monkeypatch.setattr(interface, "_CHEAP_PADDING_ROWS", True, raising=False)
    meta = _meta(
        use_rejection_sampling=case != "no rejection",
        **({"draft_probs": None} if case == "no draft probs" else {}),
        **({"batch_slot_ids": None} if case == "no slot ids" else {}),
    )
    logits = torch.zeros(2 * K, 16) if case == "2-D logits" else torch.zeros(2, K, 16)
    assert worker.advanced_sample_draft_block(logits, meta, 1, 3) == "tokens"
    assert worker.seen is None


def test_a_failed_block_disarms(worker):
    worker.fail = True
    with pytest.raises(RuntimeError, match="sampler failed"):
        worker.advanced_sample_draft_block(torch.zeros(2, K, 16), _meta(), 1, 3)
    assert worker.seen is not None and dspark_markov_worker._topp_det._ROW_STORE[0] is None


def test_the_store_is_skipped_only_for_rows_the_renorm_stored(monkeypatch):
    monkeypatch.setattr(dspark_markov_worker, "FOLD_F1", True)
    monkeypatch.setattr(dspark_markov_worker, "TAIL16_G3", False)
    slot_ids, probs = torch.tensor([2]), torch.ones(1, K, 16)

    def store(draft_probs):
        dspark_markov_worker.S4WorkerMixin._store_block_draft_probs(draft_probs, slot_ids, probs)
        return bool(draft_probs[2].eq(1).all())

    for stored in (True, False):
        monkeypatch.setattr(dspark_markov_worker._topp_det, "rows_stored", lambda *a, s=stored: s)
        assert store(torch.zeros(4, K, 16)) is not stored, stored
    monkeypatch.setattr(dspark_markov_worker, "FOLD_F1", False)
    monkeypatch.setattr(dspark_markov_worker._topp_det, "rows_stored", lambda *a: True)
    assert store(torch.zeros(4, K, 16))


def _fold_f1_in_subprocess(value):
    name = "TRTLLM_M31_DISABLE_FOLD_TOPP_SLOT_WRITE"
    env = {k: v for k, v in os.environ.items() if k != name}
    if value is not None:
        env[name] = value
    code = (
        "from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
        "minimax_m3_1_flash_nvfp4__sm_107__tp8 import dspark_markov_worker as m; print('FOLD_F1=' + str(m.FOLD_F1))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr
    return [ln for ln in out.stdout.splitlines() if ln.startswith("FOLD_F1=")][-1]


def test_fold_f1_kill_switch_subprocess():
    """FOLD_F1 is on by default; TRTLLM_M31_DISABLE_FOLD_TOPP_SLOT_WRITE=1 (read at import) turns it off."""
    assert _fold_f1_in_subprocess(None) == "FOLD_F1=True"
    assert _fold_f1_in_subprocess("0") == "FOLD_F1=True"
    assert _fold_f1_in_subprocess("1") == "FOLD_F1=False"
