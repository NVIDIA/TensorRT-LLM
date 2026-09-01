# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-window and replay-shape checks of the layer-wise benchmarks Calibrator.

No GPU and no nsys: these read the replay database and nothing else. They live
apart from test_layer_wise_benchmarks.py, which is the nsys integration suite --
its module-scoped autouse fixture skips that whole module when nsys cannot trace
CUDA, which is the right answer for a trace-and-parse test and the wrong one for
these.

The calibrator is built by hand rather than through Calibrator.init(), which
cannot be used here: _init_replay_mode() decodes every record and moves the slots
to CUDA.
"""

from typing import Iterable

import pytest

from tensorrt_llm.tools.layer_wise_benchmarks.calibrator import Calibrator, Mode


def _replay_calibrator(
    iterations: Iterable[int],
    tokens: int = 32,
    top_k: int = 6,
    layers: int = 4,
) -> Calibrator:
    calibrator = Calibrator()
    calibrator.mode = Mode.REPLAY
    calibrator._replay_db = {
        i: {
            "metadata": [
                {
                    "layer_idx": k,
                    "num_slots": 256,
                    "token_selected_slots_shape": [tokens, top_k],
                }
                for k in range(layers)
            ]
        }
        for i in iterations
    }
    return calibrator


def test_missing_replay_iterations_none_when_the_window_fits() -> None:
    calibrator = _replay_calibrator(range(100, 126))
    assert calibrator.get_missing_replay_iterations(105, 125) == []
    assert calibrator.get_missing_replay_iterations(100, 125) == []


def test_missing_replay_iterations_past_the_end() -> None:
    calibrator = _replay_calibrator(range(100, 126))
    assert calibrator.get_missing_replay_iterations(124, 128) == [126, 127, 128]
    assert calibrator.get_missing_replay_iterations(0, 3) == [0, 1, 2, 3]


def test_missing_replay_iterations_sees_a_hole() -> None:
    """Report the case a first/last comparison cannot answer.

    get_replay_iteration_range() raises on a non-contiguous calibration, so a bounds
    check has nothing to compare against and the KeyError comes back at pre_step().
    A window that stays inside one contiguous run is still legal and must pass.
    """
    calibrator = _replay_calibrator(list(range(100, 111)) + list(range(113, 126)))
    assert calibrator.get_missing_replay_iterations(113, 125) == []
    assert calibrator.get_missing_replay_iterations(105, 125) == [111, 112]


def test_missing_replay_iterations_requires_replay_mode() -> None:
    with pytest.raises(ValueError, match="only valid in REPLAY mode"):
        Calibrator().get_missing_replay_iterations(0, 1)


def test_replay_token_count_when_every_layer_agrees() -> None:
    assert _replay_calibrator(range(100, 103), tokens=64).get_replay_token_count() == 64


def test_replay_token_count_sums_the_chunks_of_one_layer() -> None:
    """MoE chunking records one entry per (layer, chunk), not one per layer.

    maybe_collect_or_replay_slots() is called from _forward_chunk_impl(), which
    _forward_multiple_chunks() runs once per chunk, so one record holds a chunk's
    token count rather than the iteration's.
    """
    calibrator = _replay_calibrator([100], layers=1)
    calibrator._replay_db[100]["metadata"] = [
        {"layer_idx": 0, "num_slots": 256, "token_selected_slots_shape": [n, 6]}
        for n in (2048, 2048)
    ]
    assert calibrator.get_replay_token_count() == 4096


def test_replay_token_count_accepts_an_uneven_chunk_split() -> None:
    """split_chunk(4096, 3) is [1366, 1365, 1365]: three shapes, one iteration.

    Such a window is replayable, and narrowing it cannot help, because the shapes
    come from inside one iteration rather than from across the window.
    """
    calibrator = _replay_calibrator([100], layers=1)
    calibrator._replay_db[100]["metadata"] = [
        {"layer_idx": 0, "num_slots": 256, "token_selected_slots_shape": [n, 6]}
        for n in (1366, 1365, 1365)
    ]
    assert calibrator.get_replay_token_count() == 4096


def test_replay_token_count_rejects_layers_that_disagree() -> None:
    """Name the shapes and where they were recorded.

    A calibration whose layers disagree cannot be replayed under one CUDA graph,
    and the message belongs next to the data that explains it.
    """
    calibrator = _replay_calibrator(range(100, 103), tokens=64)
    calibrator._replay_db[101]["metadata"][0]["token_selected_slots_shape"] = [32, 6]
    with pytest.raises(ValueError, match=r"2 different routing shapes"):
        calibrator.get_replay_token_count()


def test_replay_token_count_compares_the_whole_shape() -> None:
    """Reject a range that agrees on tokens and differs in top_k.

    One CUDA graph holds one shape, and [64, 6] is not [64, 8]; comparing only the
    token dimension would call this range replayable.
    """
    calibrator = _replay_calibrator(range(100, 103), tokens=64, top_k=6)
    calibrator._replay_db[101]["metadata"][0]["token_selected_slots_shape"] = [64, 8]
    with pytest.raises(ValueError, match=r"different routing shapes"):
        calibrator.get_replay_token_count()


def test_replay_token_count_rejects_chunks_disagreeing_on_top_k() -> None:
    """Chunks of one layer are summed, so their trailing dims have to match.

    Summing [2048, 6] and [2048, 8] into "4096 tokens" would invent a shape that
    was never recorded.
    """
    calibrator = _replay_calibrator([100], layers=1)
    calibrator._replay_db[100]["metadata"] = [
        {"layer_idx": 0, "num_slots": 256, "token_selected_slots_shape": [2048, top_k]}
        for top_k in (6, 8)
    ]
    with pytest.raises(ValueError, match=r"disagree on the routing shape"):
        calibrator.get_replay_token_count()


def test_replay_token_count_is_scoped_to_the_window() -> None:
    """Ignore records outside the window being replayed.

    Unscoped, a single stray iteration at another shape makes the whole file look
    inconsistent and takes a perfectly replayable window down with it.
    """
    calibrator = _replay_calibrator(range(105, 126), tokens=64)
    calibrator._replay_db[99] = {
        "metadata": [
            {"layer_idx": k, "num_slots": 256, "token_selected_slots_shape": [32, 6]}
            for k in range(4)
        ]
    }
    with pytest.raises(ValueError, match=r"different routing shapes"):
        calibrator.get_replay_token_count()
    assert calibrator.get_replay_token_count(105, 125) == 64
    with pytest.raises(ValueError, match=r"different routing shapes"):
        calibrator.get_replay_token_count(99, 125)


def test_replay_token_count_rejects_an_empty_window() -> None:
    calibrator = _replay_calibrator(range(100, 126), tokens=64)
    with pytest.raises(ValueError, match=r"No routing recorded over"):
        calibrator.get_replay_token_count(200, 300)


def test_replay_token_count_requires_replay_mode() -> None:
    with pytest.raises(ValueError, match="only valid in REPLAY mode"):
        Calibrator().get_replay_token_count()
