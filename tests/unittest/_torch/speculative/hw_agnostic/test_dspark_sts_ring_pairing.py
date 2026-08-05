# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""STS pairing must join on draft-pass identity, never on arrival order.

The recorder's label (accepted count) and its feature (confidence logits) are
produced by different steps: the block verified at step ``i`` was drafted at
step ``i-1``, and under the overlap scheduler the sampler consumes the label
while pass ``i+1`` has already overwritten the shared confidence buffer. The
single mutable stash this ring replaced had NO execution order under which
the pair was correct -- measured on job 2562577, 82.2% of rows paired
confidence(t+1) with accepted(t), correlation 0.14 -- and the shards looked
perfectly healthy.

These tests pin the join contract: a pair is appended only when the ring
snapshot's per-row stamp equals the pass the label verifies, everything else
is a *counted* decline, and the shard carries provenance the fitter can
refuse. No CUDA: the recorder itself is device-agnostic (CPU tensors skip the
pinned-copy event), which is exactly what lets this run in pre-merge CI.
"""

import importlib.util
import os
import pathlib
import sys
import tempfile

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_sts import DSparkStsRecorder

BLOCK = 5
ROWS = 8


def _recorder(tmp, flush_every=1000):
    return DSparkStsRecorder(path_stem=os.path.join(tmp, "shard"),
                             block_size=BLOCK, rank=0,
                             flush_every=flush_every)


def _stage(rec, seq, written_rows, fill):
    """Stage one pass's snapshot: `written_rows` stamped `seq`, rest stale.

    Mirrors the producer: the in-graph scatter stamps only the rows the pass
    actually drafted; every other row keeps whatever stamp it had (0 for
    never-written, matching the worker's zero-initialized stamp buffer).
    """
    logits = torch.full((ROWS, BLOCK), float(fill))
    stamps = torch.zeros(ROWS, dtype=torch.int32)
    stamps[list(written_rows)] = seq
    rec.stage_snapshot(device_logits=logits, device_stamps=stamps,
                       staged_seq=seq)


def test_pairs_by_pass_identity_not_arrival_order():
    """A label for pass 1 gets pass-1 logits even after pass 2 was staged."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        _stage(rec, seq=1, written_rows=[2], fill=1.0)
        _stage(rec, seq=2, written_rows=[2], fill=2.0)  # overwrite arrives first
        rec.record(row=2, accepted=3, target_seq=1)
        rec.record(row=2, accepted=1, target_seq=2)
        assert rec.stats["recorded"] == 2
        assert rec._logits[0][0].item() == 1.0  # pass 1 content, not latest
        assert rec._logits[1][0].item() == 2.0


def test_row_not_written_for_that_pass_is_declined():
    """A row the pass never drafted must not contribute a fabricated pair."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        _stage(rec, seq=1, written_rows=[2], fill=1.0)
        rec.record(row=3, accepted=2, target_seq=1)  # row 3 stamp is 0, not 1
        assert rec.stats["recorded"] == 0
        assert rec.stats["stale_stamp"] == 1


def test_ring_eviction_declines_instead_of_mispairing():
    """A label older than the ring depth is dropped, never matched to the

    evicting pass's content: the slot's stamps say who actually wrote it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        for seq in range(1, 2 + DSparkStsRecorder._RING_DEPTH):
            _stage(rec, seq=seq, written_rows=[2], fill=float(seq))
        rec.record(row=2, accepted=3, target_seq=1)  # slot now holds seq 5
        assert rec.stats["recorded"] == 0
        assert rec.stats["stale_stamp"] == 1


def test_every_giving_up_path_is_counted():
    """Silent declines are how the mispaired fit shipped; none may be silent."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        rec.record(row=2, accepted=1, target_seq=None)   # no pass id
        rec.record(row=2, accepted=1, target_seq=7)      # ring slot never staged
        _stage(rec, seq=1, written_rows=[2], fill=1.0)
        rec.record(row=None, accepted=1, target_seq=1)   # unresolvable request
        rec.record(row=ROWS + 3, accepted=1, target_seq=1)
        assert rec.stats == {"recorded": 0, "no_snapshot": 2,
                             "stale_stamp": 0, "row_out_of_range": 1,
                             "no_row": 1, "snapshots_staged": 1}


def test_staging_without_stamps_is_a_noop():
    """No stamps means no way to verify identity -- refuse to guess."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        rec.stage_snapshot(device_logits=torch.zeros(ROWS, BLOCK),
                           device_stamps=None, staged_seq=1)
        rec.stage_snapshot(device_logits=torch.zeros(ROWS, BLOCK),
                           device_stamps=torch.zeros(ROWS, dtype=torch.int32),
                           staged_seq=None)
        assert rec.stats["snapshots_staged"] == 0
        rec.record(row=2, accepted=1, target_seq=1)
        assert rec.stats["no_snapshot"] == 1


def test_shard_carries_provenance_and_prefix_semantics():
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        _stage(rec, seq=1, written_rows=[2], fill=1.5)
        rec.record(row=2, accepted=3, target_seq=1)
        rec.flush()
        blob = torch.load(os.path.join(tmp, "shard.r0.0.pt"),
                          map_location="cpu")
        assert blob["meta"]["pairing"] == "draft_seq_ring"
        assert blob["meta"]["stats"]["recorded"] == 1
        assert blob["logits"].shape == (1, BLOCK)
        # accepted=3 -> positions 0..2 were ALL accepted: the prefix event
        # whose probability survival[k] estimates.
        assert blob["prefix_mask"].tolist() == [[1.0, 1.0, 1.0, 0.0, 0.0]]


# ---------------------------------------------------------------------------
# The fitter must refuse shards from the pre-ring recorders: their pairs are
# mislabeled in ways the tensors themselves cannot reveal.
# ---------------------------------------------------------------------------

_SPEC = importlib.util.spec_from_file_location(
    "dspark_fit_sts",
    pathlib.Path(__file__).resolve().parents[4] / "microbenchmarks" /
    "dspark_fit_sts.py")
_FITTER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FITTER
_SPEC.loader.exec_module(_FITTER)


def test_fitter_refuses_shards_without_ring_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        torch.save({"logits": torch.zeros(4, BLOCK),
                    "prefix_mask": torch.zeros(4, BLOCK)},
                   os.path.join(tmp, "old.r0.0.pt"))
        with pytest.raises(SystemExit, match="draft_seq_ring"):
            _FITTER.load_shards(os.path.join(tmp, "*.pt"))


def test_fitter_accepts_ring_shards():
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        _stage(rec, seq=1, written_rows=[2, 3], fill=0.5)
        rec.record(row=2, accepted=2, target_seq=1)
        rec.record(row=3, accepted=5, target_seq=1)
        rec.flush()
        logits, mask = _FITTER.load_shards(os.path.join(tmp, "*.pt"))
        assert logits.shape == (2, BLOCK) and mask.shape == (2, BLOCK)
