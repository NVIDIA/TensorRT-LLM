# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay-window, replay-shape and decoder-layer checks of the Calibrator.

No GPU and no nsys: these read the replay database or wrap stand-in layers, and
nothing else. They live apart from test_layer_wise_benchmarks.py, which is the
nsys integration suite -- its module-scoped autouse fixture skips that whole
module when nsys cannot trace CUDA, which is the right answer for a
trace-and-parse test and the wrong one for these.

The calibrator is built by hand rather than through Calibrator.init(), which
cannot be used here: _init_replay_mode() decodes every record and moves the slots
to CUDA.
"""

from typing import Iterable

import pytest

from tensorrt_llm.tools.layer_wise_benchmarks.calibrator import (
    Calibrator,
    Mode,
    NoDecoderLayers,
    _decoder_layers,
)

pytestmark = pytest.mark.cpu_only


class _ScanCapped(dict):
    """A replay database that raises once it has been probed `cap` times."""

    def __init__(self, entries: dict, cap: int) -> None:
        super().__init__(entries)
        self._cap = cap
        self._probes = 0

    def __contains__(self, key: object) -> bool:
        self._probes += 1
        if self._probes > self._cap:
            raise AssertionError(f"scanned more than {self._cap} times")
        return super().__contains__(key)


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
    assert calibrator.get_missing_replay_iterations(105, 125) == (0, [])
    assert calibrator.get_missing_replay_iterations(100, 125) == (0, [])


def test_missing_replay_iterations_past_the_end() -> None:
    calibrator = _replay_calibrator(range(100, 126))
    assert calibrator.get_missing_replay_iterations(124, 128) == (3, [126, 127, 128])
    assert calibrator.get_missing_replay_iterations(0, 3) == (4, [0, 1, 2, 3])


def test_missing_replay_iterations_sees_a_hole() -> None:
    """Report the case a first/last comparison cannot answer.

    get_replay_iteration_range() raises on a non-contiguous calibration, so a bounds
    check has nothing to compare against and the KeyError comes back at pre_step().
    A window that stays inside one contiguous run is still legal and must pass.
    """
    calibrator = _replay_calibrator(list(range(100, 111)) + list(range(113, 126)))
    assert calibrator.get_missing_replay_iterations(113, 125) == (0, [])
    assert calibrator.get_missing_replay_iterations(105, 125) == (2, [111, 112])


def test_missing_replay_iterations_counts_more_than_it_names() -> None:
    """The count is the whole answer; the examples are capped at `limit`."""
    calibrator = _replay_calibrator(range(100, 126))
    assert calibrator.get_missing_replay_iterations(126, 200) == (
        75,
        [126, 127, 128, 129, 130, 131, 132, 133],
    )
    assert calibrator.get_missing_replay_iterations(126, 200, limit=2) == (75, [126, 127])


def test_missing_replay_iterations_is_bounded_by_the_pack() -> None:
    """A fat-fingered window has to stay cheap.

    --replay-stop-iter is a bare `type=int`, so 1000000000 is one keystroke away.
    Enumerating that window costs tens of gigabytes and takes the box down before
    run.py can print the error this check exists to give.
    """
    calibrator = _replay_calibrator(range(100, 126))
    # Refuses to be scanned rather than counting the scans and asserting afterwards:
    # letting an unbounded scan run to completion to measure it is the failure.
    calibrator._replay_db = _ScanCapped(calibrator._replay_db, cap=100)
    assert calibrator.get_missing_replay_iterations(105, 10**9) == (
        10**9 - 105 + 1 - 21,
        [126, 127, 128, 129, 130, 131, 132, 133],
    )


def test_missing_replay_iterations_of_an_inverted_window() -> None:
    """An inverted window holds nothing and misses nothing, not a negative count.

    run.py rejects one before it reaches here, but a count below zero is not an
    answer to give a caller that asks directly.
    """
    calibrator = _replay_calibrator(range(100, 126))
    assert calibrator.get_missing_replay_iterations(125, 100) == (0, [])


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


class _Layer:
    """Stands in for a decoder layer: the one thing the calibrator wraps."""

    def forward(self, *args, **kwargs):
        return None


class _Holder:
    def __init__(self, **attrs) -> None:
        self.__dict__.update(attrs)


def _wrapped_chain(unwraps: int, layers) -> _Holder:
    """`layers` behind `unwraps` levels of `.model`."""
    obj = _Holder(layers=layers)
    for _ in range(unwraps):
        obj = _Holder(model=obj)
    return obj


@pytest.mark.parametrize("attr", ["layers", "block", "blocks", "h"])
def test_decoder_layers_under_each_known_name(attr: str) -> None:
    layers = [_Layer()]
    assert _decoder_layers(_Holder(model=_Holder(**{attr: layers}))) is layers


def test_decoder_layers_descends_into_llm() -> None:
    """Qwen3.5-VL keeps the causal LM under `llm`: `model.llm.model.layers`."""
    layers = [_Layer(), _Layer()]
    vl = _Holder(llm=_Holder(model=_Holder(layers=layers)))
    assert _decoder_layers(vl) is layers


def test_decoder_layers_descends_into_language_model() -> None:
    """The HF-style `*ForConditionalGeneration` shape."""
    layers = [_Layer()]
    vl = _Holder(language_model=_Holder(model=_Holder(layers=layers)))
    assert _decoder_layers(vl) is layers


def test_decoder_layers_skips_an_empty_candidate_for_a_real_one() -> None:
    layers = [_Layer()]
    assert _decoder_layers(_Holder(model=_Holder(layers=[], blocks=layers))) is layers


def test_decoder_layers_skips_a_mapping_named_h() -> None:
    """A dict passes len(), then raises KeyError on [0]; the search must go on."""
    layers = [_Layer()]
    assert _decoder_layers(_Holder(h={"attn": 1}, model=_Holder(layers=layers))) is layers


def test_decoder_layers_raises_when_only_a_mapping_is_present() -> None:
    with pytest.raises(NoDecoderLayers):
        _decoder_layers(_Holder(model=_Holder(h={"attn": 1})))


def test_decoder_layers_rejects_items_without_forward() -> None:
    """A sized, indexable attribute is not by itself a decoder stack."""
    with pytest.raises(NoDecoderLayers, match="none of"):
        _decoder_layers(_Holder(model=_Holder(blocks=[{"hidden": 4096}])))


def test_decoder_layers_stops_on_a_self_referential_wrapper() -> None:
    holder = _Holder()
    holder.model = holder
    with pytest.raises(NoDecoderLayers):
        _decoder_layers(holder)


def test_decoder_layers_stops_on_a_two_object_cycle() -> None:
    """A cycle longer than one object must stop, not walk the depth budget."""
    outer = _Holder()
    inner = _Holder(llm=outer)
    outer.model = inner
    with pytest.raises(NoDecoderLayers, match=r"searched along _Holder -> _Holder \("):
        _decoder_layers(outer)


def test_decoder_layers_falls_through_to_a_later_inner_attr_after_a_cycle() -> None:
    """A back-reference under `model` must not hide the real stack under `llm`."""
    layers = [_Layer()]
    outer = _Holder()
    inner = _Holder(model=outer, llm=_Holder(model=_Holder(layers=layers)))
    outer.model = inner
    assert _decoder_layers(outer) is layers


@pytest.mark.parametrize("unwraps", [0, 1, 2, 3, 4])
def test_decoder_layers_searches_the_whole_budget(unwraps: int) -> None:
    layers = [_Layer()]
    assert _decoder_layers(_wrapped_chain(unwraps, layers)) is layers


def test_decoder_layers_stops_one_level_past_the_budget() -> None:
    with pytest.raises(NoDecoderLayers):
        _decoder_layers(_wrapped_chain(5, [_Layer()]))


def test_decoder_layers_names_the_type_and_chain_it_searched() -> None:
    class Outer(_Holder):
        pass

    class Inner(_Holder):
        pass

    with pytest.raises(NoDecoderLayers) as excinfo:
        _decoder_layers(Outer(llm=Inner(decoder=[_Layer()])))
    message = str(excinfo.value)
    assert message.startswith("Outer keeps its decoder layers under none of layers, block")
    assert "Outer -> Inner" in message


def test_no_decoder_layers_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _decoder_layers(_Holder())


def test_maybe_wrap_model_wraps_layers_behind_a_nested_wrapper() -> None:
    """Every layer of `model.llm.model.layers` is wrapped and still delegates.

    Goes through the public entry point; MARK mode reaches _wrap_layer_forward
    without touching CUDA.
    """
    layers = [_Layer(), _Layer(), _Layer()]
    model = _Holder(llm=_Holder(model=_Holder(layers=layers)))
    originals = [layer.forward for layer in layers]

    calibrator = Calibrator()
    calibrator.mode = Mode.MARK
    assert calibrator.maybe_wrap_model(model) is model

    for idx, (layer, original) in enumerate(zip(layers, originals)):
        assert layer.forward is not original, f"layer {idx} left unwrapped"
        # Bound methods are rebuilt on each attribute access, so compare by `==`.
        assert layer.forward.__wrapped__ == original
        assert layer.forward() is None
