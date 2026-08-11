# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSpark STS calibration subsystem: recorder ring pairing, shard provenance,
fitter acceptance, collection-regime guards, and confidence-head resolution.
Runs on CPU only (the recorder is device-agnostic), so it fits pre-merge CI.
"""

import contextlib
import importlib.util
import os
import pathlib
import sys
import tempfile

import pytest
import torch

from tensorrt_llm._torch.speculative.dspark_sts import (DSparkStsRecorder,
                                                        STS_COLLECT_ENV,
                                                        make_recorder_from_env)

BLOCK = 5
ROWS = 8

_PIN = "TLLM_DSPARK_FORCE_VERIFY_LEN"


def _recorder(tmp, flush_every=1000):
    return DSparkStsRecorder(path_stem=os.path.join(tmp, "shard"),
                             block_size=BLOCK, rank=0,
                             flush_every=flush_every)


def _stage(rec, seq, written_rows, fill):
    """Stage one pass's snapshot: `written_rows` stamped `seq`, rest stale."""
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
    evicting pass's content."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        for seq in range(1, 2 + DSparkStsRecorder._RING_DEPTH):
            _stage(rec, seq=seq, written_rows=[2], fill=float(seq))
        rec.record(row=2, accepted=3, target_seq=1)  # slot now holds seq 5
        assert rec.stats["recorded"] == 0
        assert rec.stats["stale_stamp"] == 1


def test_every_giving_up_path_is_counted():
    """No decline path may be silent; every one increments a named counter."""
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
    """Flushed shards carry ring provenance, stats, and prefix-mask labels."""
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


_SPEC = importlib.util.spec_from_file_location(
    "dspark_fit_sts",
    pathlib.Path(__file__).resolve().parents[4] / "microbenchmarks" /
    "dspark_fit_sts.py")
_FITTER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FITTER
_SPEC.loader.exec_module(_FITTER)


def test_fitter_refuses_shards_without_ring_provenance():
    """The fitter must refuse shards lacking draft_seq_ring provenance."""
    with tempfile.TemporaryDirectory() as tmp:
        torch.save({"logits": torch.zeros(4, BLOCK),
                    "prefix_mask": torch.zeros(4, BLOCK)},
                   os.path.join(tmp, "old.r0.0.pt"))
        with pytest.raises(SystemExit, match="draft_seq_ring"):
            _FITTER.load_shards(os.path.join(tmp, "*.pt"))


def test_fitter_accepts_ring_shards():
    """Shards produced by the ring recorder load cleanly in the fitter."""
    with tempfile.TemporaryDirectory() as tmp:
        rec = _recorder(tmp)
        _stage(rec, seq=1, written_rows=[2, 3], fill=0.5)
        rec.record(row=2, accepted=2, target_seq=1)
        rec.record(row=3, accepted=5, target_seq=1)
        rec.flush()
        logits, mask = _FITTER.load_shards(os.path.join(tmp, "*.pt"))
        assert logits.shape == (2, BLOCK) and mask.shape == (2, BLOCK)


@contextlib.contextmanager
def collecting(pin=None):
    """Set the collection env for the block, restore it after."""
    saved = {k: os.environ.get(k) for k in (STS_COLLECT_ENV, _PIN)}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[STS_COLLECT_ENV] = os.path.join(tmp, "collect")
            if pin is None:
                os.environ.pop(_PIN, None)
            else:
                os.environ[_PIN] = str(pin)
            yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_collection_off_builds_nothing():
    """The serving default must be untouched by any of this."""
    os.environ.pop(STS_COLLECT_ENV, None)
    assert make_recorder_from_env(block_size=5) is None
    assert make_recorder_from_env(block_size=5, has_cost_table=True,
                                  ragged_mode="compact") is None


def test_a_loaded_cost_table_without_a_pin_is_refused():
    """A table means the planner trims, and trimming censors the label."""
    with collecting():
        with pytest.raises(ValueError, match="censors the acceptance label"):
            make_recorder_from_env(block_size=5, has_cost_table=True)


def test_a_pinned_window_makes_the_table_harmless():
    """Every uncensored regime (pinned window, no table, static mode) must
    build a recorder."""
    with collecting(pin=5):
        rec = make_recorder_from_env(block_size=5, has_cost_table=True)
        assert rec is not None and rec.block_size == 5
    with collecting():
        assert make_recorder_from_env(block_size=5,
                                      has_cost_table=False) is not None
        assert make_recorder_from_env(block_size=5,
                                      ragged_mode="static") is not None


def test_compact_mode_is_refused():
    """Padding rows contribute prefix labels that measure nothing; refused
    even with the pin set."""
    with collecting(pin=5):
        with pytest.raises(ValueError, match="Padded verify rows"):
            make_recorder_from_env(block_size=5, ragged_mode="compact")


class _Head:
    pass


class _Bare:
    """DSparkDraftModel layout: stages under .mtp_layers."""

    def __init__(self, head):
        stage = type("Stage", (), {})()
        if head is not None:
            stage.confidence_head = head
        self.mtp_layers = [object(), object(), stage]


class _Wrapper:
    """DSparkForCausalLM layout: bare model under .dspark_model."""

    def __init__(self, head):
        self.dspark_model = _Bare(head)


def test_head_resolves_across_both_layouts_and_absence_is_none():
    """resolve_confidence_head finds the head under either model layout and
    returns None when absent."""
    from tensorrt_llm._torch.speculative.dspark_sts import resolve_confidence_head
    h = _Head()
    assert resolve_confidence_head(_Bare(h)) is h
    assert resolve_confidence_head(_Wrapper(h)) is h
    assert resolve_confidence_head(_Bare(None)) is None
    assert resolve_confidence_head(_Wrapper(None)) is None
    assert resolve_confidence_head(object()) is None
