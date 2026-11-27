# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F4A's routing in the MSA metadata, without GPUs or a model: with the policy's kv_update_tail and a tail past
the live rows, on_update_kv_lens hands the out_cache_loc tail fill to that launch (no fill of its own, kv_update not
called); when it declines, there is no such op, or the tail runs over the live rows, the
metadata fills the tail itself as before. The launch's values are certified by test_modeling_v2_msa_kv_update.py."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_backend
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy

pytestmark = pytest.mark.cpu_only

Meta = msa_backend.MiniMaxM3MsaSparseAttentionMetadata
STALE = 7


def _meta(monkeypatch, *, tail="takes", ready=True, start=6, total_q=4, length=10):
    calls = []

    def kv_update(*args):
        calls.append(("kv_update", args[11].clone()))
        return True

    def kv_update_tail(*args):
        calls.append(("tail", args[-1], args[11].clone()))
        return tail == "takes"

    monkeypatch.setattr(
        TrtllmAttentionMetadata, "on_update_kv_lens", lambda self: calls.append(("super",))
    )
    monkeypatch.setattr(Meta, "num_seqs", property(lambda self: 2))
    monkeypatch.setattr(Meta, "num_tokens", property(lambda self: total_q))
    md = Meta.__new__(Meta)
    md.msa_policy = MsaRuntimePolicy(
        kv_update=kv_update, kv_update_tail=kv_update_tail if tail != "none" else None
    )
    md._msa_subpage_expanded = True
    md._msa_tail_start = start
    md.msa_out_cache_loc = torch.full((length,), STALE, dtype=torch.int32)
    md._msa_fields_ready = ready
    md._msa_kv_lens_dynamic = True
    md.kv_cache_manager = SimpleNamespace(tokens_per_block=128)
    md._msa_prefill_n_valid_blocks = None
    md.msa_n_valid_blocks = torch.zeros(length, dtype=torch.int32)
    md.msa_cu_kv_lens = None
    for name in ("kv_lens_cuda", "msa_kv_lens_staged", "seq_lens_cuda", "msa_seq_lens_cuda"):
        # a read-only property (seq_lens_cuda on the attention interface) is fed through its backing field
        attr = getattr(Meta, name, None)
        if isinstance(attr, property) and attr.fset is None:
            name = "_" + name
        setattr(md, name, torch.zeros(4, dtype=torch.int32))
    for name in ("msa_q_batch_row", "msa_q_intra"):
        setattr(md, name, torch.zeros(length, dtype=torch.int32))
    md.msa_block_table = torch.zeros(2, 3, dtype=torch.int32)
    return md, calls


def _tail_filled(t, start):
    return bool((t[start:] == -1).all()) and bool((t[:start] == STALE).all())


def test_the_tail_goes_to_the_kv_update_launch(monkeypatch):
    md, calls = _meta(monkeypatch)
    md.on_update_kv_lens()
    assert [c[0] for c in calls] == ["super", "tail"]
    assert calls[1][1] == 6 and bool((calls[1][2] == STALE).all())
    assert bool(
        (md.msa_out_cache_loc == STALE).all()
    )  # the fake launch writes nothing: no fill of its own ran
    assert md._msa_subpage_expanded is False


@pytest.mark.parametrize(
    "tail, start, expected",
    [
        (
            "declines",
            6,
            ["super", "tail", "kv_update"],
        ),  # a declined tail launch: fill, then the stock update
        ("none", 6, ["super", "kv_update"]),  # no tail op: the tail is filled first
        ("takes", 3, ["super", "kv_update"]),  # a tail over the live rows keeps the fill
    ],
)
def test_the_metadata_fills_the_tail_otherwise(monkeypatch, tail, start, expected):
    md, calls = _meta(monkeypatch, tail=tail, start=start, total_q=4)
    md.on_update_kv_lens()
    assert [c[0] for c in calls] == expected and _tail_filled(calls[-1][-1], start)
    if tail == "declines":
        assert bool((calls[1][2] == STALE).all())
