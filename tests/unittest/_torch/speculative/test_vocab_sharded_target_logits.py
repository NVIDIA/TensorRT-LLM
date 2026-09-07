# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for verifying against vocab-sharded target logits.

A one-engine target LM head is COLUMN tensor-parallel and all-gathers its vocab
shards into full-vocab logits. When the only consumer is the greedy argmax in
``_sample_tokens_for_batch``, ``SpecDecOneEngineForCausalLM.forward`` keeps the
shard instead and the worker recovers the same token from the per-rank maxima.

Two things have to hold for that to be lossless, and both are tested here:
  * the gate only opens when nothing else reads the logits' values, and when
    dropping the gather really leaves each rank its own contiguous, unpadded
    slice of the vocabulary;
  * combining the shards' ``(index, value)`` pairs picks exactly the token a
    full-vocab argmax would, ties included.

Exercised unbound on ``types.SimpleNamespace`` stand-ins (mirroring
test_group_all_greedy_sync.py), with ``allgather`` stubbed by the rank-order
concatenation it performs, so no GPU or distributed process group is needed.
"""

import functools
import types

import torch

from tensorrt_llm._torch.models.modeling_speculative import _lm_head_shard_is_self_contained
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase


def _fake_lm_head(tp_mode=TensorParallelMode.COLUMN, gather_output=True, tp_size=2, padding_size=0):
    return types.SimpleNamespace(
        tp_mode=tp_mode, gather_output=gather_output, tp_size=tp_size, padding_size=padding_size
    )


def _fake_worker(tp_size=2, enable_attention_dp=False, **overrides):
    worker = types.SimpleNamespace(
        guided_decoder=None,
        use_dynamic_tree=False,
        spec_config=types.SimpleNamespace(use_relaxed_acceptance_for_thinking=False),
        mapping=types.SimpleNamespace(tp_size=tp_size, enable_attention_dp=enable_attention_dp),
    )
    for key, value in overrides.items():
        setattr(worker, key, value)
    return worker


def _fake_meta(is_all_greedy_sample=True, enable_penalty=False):
    return types.SimpleNamespace(
        is_all_greedy_sample=is_all_greedy_sample, enable_penalty=enable_penalty
    )


def _can_verify(worker, meta):
    return Eagle3OneModelWorker.can_verify_vocab_sharded_target_logits(worker, meta)


# --------------------------------------------------------------------------
# The head-geometry precondition
# --------------------------------------------------------------------------


def test_column_parallel_head_shard_is_self_contained():
    assert _lm_head_shard_is_self_contained(_fake_lm_head())


def test_padded_vocab_shard_is_not_self_contained():
    # The padding columns are only trimmed on the gathered path, so an
    # ungathered shard would score token ids that do not exist.
    assert not _lm_head_shard_is_self_contained(_fake_lm_head(padding_size=4))


def test_untensor_parallel_head_has_no_shard_to_keep():
    assert not _lm_head_shard_is_self_contained(_fake_lm_head(tp_size=1))
    assert not _lm_head_shard_is_self_contained(_fake_lm_head(tp_mode=TensorParallelMode.ROW))
    assert not _lm_head_shard_is_self_contained(_fake_lm_head(gather_output=False))


# --------------------------------------------------------------------------
# The worker-side gate
# --------------------------------------------------------------------------


def test_plain_tp_greedy_batch_can_verify_sharded():
    assert _can_verify(_fake_worker(), _fake_meta())


def test_full_vocab_consumers_close_the_gate():
    assert not _can_verify(_fake_worker(guided_decoder=object()), _fake_meta())
    assert not _can_verify(_fake_worker(use_dynamic_tree=True), _fake_meta())
    assert not _can_verify(
        _fake_worker(spec_config=types.SimpleNamespace(use_relaxed_acceptance_for_thinking=True)),
        _fake_meta(),
    )
    assert not _can_verify(_fake_worker(), _fake_meta(is_all_greedy_sample=False))
    assert not _can_verify(_fake_worker(), _fake_meta(enable_penalty=True))


def test_replicated_full_vocab_logits_close_the_gate():
    # Nothing is sharded on a single rank, and under attention DP each rank
    # owns different requests behind a replicated head.
    assert not _can_verify(_fake_worker(tp_size=1), _fake_meta())
    assert not _can_verify(_fake_worker(enable_attention_dp=True), _fake_meta())


def test_base_worker_does_not_opt_in():
    assert not SpecWorkerBase.can_verify_vocab_sharded_target_logits(_fake_worker(), _fake_meta())


# --------------------------------------------------------------------------
# The shard combine itself
# --------------------------------------------------------------------------


def _shard_combining_worker(tp_size, tp_rank, monkeypatch, shards):
    """A stand-in whose ``allgather`` returns the rank-order concatenation."""
    import tensorrt_llm._torch.distributed.ops as dist_ops

    def fake_allgather(tensor, mapping, dim=0, sizes=None):
        del mapping, sizes
        # Every rank contributes its own (index, value) pair for the same rows;
        # allgather along the last dim concatenates them in rank order.
        return torch.cat(shards["combined"], dim=dim)

    monkeypatch.setattr(dist_ops, "allgather", fake_allgather)

    worker = types.SimpleNamespace(mapping=types.SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank))
    for name in (
        "_get_local_max_and_combined",
        "_get_draft_tokens_from_gathered",
        "_greedy_argmax_over_vocab_shards",
    ):
        setattr(worker, name, functools.partial(getattr(SpecWorkerBase, name), worker))
    return worker


def _combined_argmax(logits, tp_size, monkeypatch):
    """Run every rank's local reduction, then one rank's global pick."""
    shard_width = logits.shape[-1] // tp_size
    shards = {"combined": []}
    for tp_rank in range(tp_size):
        worker = _shard_combining_worker(tp_size, tp_rank, monkeypatch, shards)
        local = logits[:, tp_rank * shard_width : (tp_rank + 1) * shard_width]
        shards["combined"].append(worker._get_local_max_and_combined(local, worker.mapping))
    # All ranks see the same gathered pairs and therefore the same token.
    worker = _shard_combining_worker(tp_size, 0, monkeypatch, shards)
    return worker._greedy_argmax_over_vocab_shards(logits[:, :shard_width], worker.mapping)


def test_shard_combine_matches_full_vocab_argmax(monkeypatch):
    torch.manual_seed(0)
    logits = torch.randn(19, 512, dtype=torch.bfloat16)
    expected = torch.argmax(logits.float(), dim=-1)
    assert torch.equal(_combined_argmax(logits, 2, monkeypatch).long(), expected)


def test_shard_combine_breaks_ties_like_full_vocab_argmax(monkeypatch):
    # Only 4 distinct values over 512 columns, so the maximum repeats both
    # within and across shards. argmax returns the lowest index holding it;
    # the shards are contiguous and rank-ordered, so the first rank holding
    # the maximum also holds its lowest global index.
    torch.manual_seed(0)
    logits = torch.randint(0, 4, (19, 512)).to(torch.bfloat16)
    expected = torch.argmax(logits.float(), dim=-1)
    assert torch.equal(_combined_argmax(logits, 4, monkeypatch).long(), expected)


def test_bfloat16_argmax_is_invariant_under_the_fp32_upcast(monkeypatch):
    # The gathered path upcasts before the argmax; the sharded path does not.
    # The cast is exact and monotonic, so it cannot move the argmax.
    torch.manual_seed(0)
    logits = torch.randn(19, 512, dtype=torch.bfloat16)
    assert torch.equal(torch.argmax(logits, dim=-1), torch.argmax(logits.float(), dim=-1))
