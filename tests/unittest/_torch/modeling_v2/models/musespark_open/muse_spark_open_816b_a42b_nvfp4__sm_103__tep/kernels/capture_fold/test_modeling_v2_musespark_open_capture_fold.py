# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host test of ``capture_fold`` (``fold.py``): what it builds on, the capture view, and
when the re-classed decoder layer leaves the DFlash capture to the boundary kernel.

No GPU and no collective: the parent layer's forward and the built-in capture are
replaced by recorders, so this checks only the wiring. The kernel's bytes are the
8-rank op test's (``comm/test_modeling_v2_unpool_gather_boundary.py``, the ``capture``
cells).
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.capture_fold import (  # noqa: E501
    fold as capture_fold,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers import (  # noqa: E501
    gathers as projection_gathers,
)
from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode

HIDDEN = 8192
LAYERS = (1, 16, 30, 44, 59)


def test_it_builds_on_the_gathers():
    assert dispatch.closure({"capture_fold"}) == {"capture_fold", "projection_gathers"}
    with pytest.raises(ValueError, match="builds on"):
        dispatch.Plan(frozenset({"capture_fold"}))


def _spec_config(mode, layers):
    return types.SimpleNamespace(
        spec_dec_mode=mode, target_layer_ids=layers, draft_is_embedded_in_target=False
    )


def test_capture_layers_come_from_the_dflash_config():
    assert (
        capture_fold.capture_layers(
            _spec_config(SpeculativeDecodingMode.DFLASH, [59, 1, 30, 16, 44])
        )
        == LAYERS
    )
    for config in (
        None,
        _spec_config(SpeculativeDecodingMode.NONE, list(LAYERS)),
        # A standalone DSpark drafter captures through its own metadata, which
        # ``destination`` never folds.
        _spec_config(SpeculativeDecodingMode.DSPARK, list(LAYERS)),
        _spec_config(SpeculativeDecodingMode.DFLASH, None),
        _spec_config(SpeculativeDecodingMode.DFLASH, []),
    ):
        with pytest.raises(ValueError):
            capture_fold.capture_layers(config)


def _metadata(rows=16, dtype=torch.bfloat16, hidden=HIDDEN):
    metadata = object.__new__(DFlashSpecMetadata)
    metadata.captured_hidden_states = torch.zeros(rows, hidden * len(LAYERS), dtype=dtype)
    metadata._layer_to_idx = {layer: i for i, layer in enumerate(LAYERS)}
    metadata.hidden_size = hidden
    return metadata


def test_destination_is_the_slice_the_builtin_capture_writes():
    metadata = _metadata()
    hidden_states = torch.zeros(8, HIDDEN)
    for index, layer in enumerate(LAYERS):
        for rows in (1, 3, 8):
            view = capture_fold.destination(metadata, layer, hidden_states[:rows])
            want = metadata.captured_hidden_states[:rows, index * HIDDEN : (index + 1) * HIDDEN]
            assert view is not None
            assert (view.data_ptr(), view.shape, view.stride()) == (
                want.data_ptr(),
                want.shape,
                want.stride(),
            )

    class Other(DFlashSpecMetadata):
        def maybe_capture_hidden_states(self, layer_id, hidden_states, residual=None):
            pass

    other = object.__new__(Other)
    other.__dict__.update(metadata.__dict__)
    no_buffer = _metadata()
    no_buffer.captured_hidden_states = None
    # (metadata, layer, residual, whether the built-in capture still writes)
    for spec_metadata, layer, states, writes in (
        (None, 30, hidden_states, False),
        (other, 30, hidden_states, True),
        (no_buffer, 30, hidden_states, False),
        (metadata, 2, hidden_states, False),
        (metadata, 30, torch.zeros(9, HIDDEN), False),
        (metadata, 30, torch.zeros(0, HIDDEN), False),
        (metadata, 30, hidden_states.to(torch.bfloat16), True),
        (metadata, 30, hidden_states.unsqueeze(0), True),
        (_metadata(dtype=torch.float32), 30, hidden_states, True),
        (_metadata(rows=4), 30, hidden_states, True),
        (_metadata(hidden=4096), 30, torch.zeros(8, 4096), True),
    ):
        assert capture_fold.destination(spec_metadata, layer, states) is None
        view, reason = capture_fold._destination(spec_metadata, layer, states)
        assert view is None and (reason is not None) == writes, reason
    assert capture_fold._destination(metadata, 30, hidden_states)[1] is None


class _Recorder:
    def __init__(self):
        self.parent_kwargs = []
        self.captures = []
        self.store = True
        self.stored_value = None
        self.warnings = []


def _layer(monkeypatch, layer_idx: int, folded: bool = True):
    recorder = _Recorder()
    monkeypatch.setattr(
        capture_fold,
        "logger",
        types.SimpleNamespace(warning=recorder.warnings.append, info=lambda message: None),
    )

    def parent_forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        precomputed_attn_input=None,
        next_layer_norm=None,
        sp_num_tokens=None,
        **kwargs,
    ):
        recorder.parent_kwargs.append(dict(kwargs))
        updated = hidden_states + 1.5
        view = self.add_ffn._capture_fold
        if recorder.store and view is not None:
            value = updated if recorder.stored_value is None else recorder.stored_value
            view.copy_(value.to(view.dtype))
            self.add_ffn._capture_stored = True
        return updated, updated.to(torch.bfloat16)

    def builtin_capture(self, layer_id, hidden_states, residual=None):
        recorder.captures.append((layer_id, hidden_states, residual))

    monkeypatch.setattr(projection_gathers.MuseSparkOpenTepDecoderLayer, "forward", parent_forward)
    monkeypatch.setattr(DFlashSpecMetadata, "maybe_capture_hidden_states", builtin_capture)
    layer = object.__new__(capture_fold.MuseSparkOpenTepCaptureDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = layer_idx
    layer.add_ffn = types.SimpleNamespace(_capture_fold=None, _capture_stored=False)
    layer._capture_folded = folded
    layer._capture_fold_fallbacks = frozenset()
    return layer, recorder


def _call(layer, metadata, capture_fold=True, rows=8, next_norm=object()):
    return layer(
        None,
        torch.randn(rows, HIDDEN),
        None,
        precomputed_attn_input=None,
        next_layer_norm=next_norm,
        sp_num_tokens=None,
        capture_fold=capture_fold,
        spec_metadata=metadata,
        fused_unpool_gather=True,
    )


def test_a_stored_capture_replaces_the_builtin_call(monkeypatch):
    layer, recorder = _layer(monkeypatch, 30)
    metadata = _metadata()
    result = _call(layer, metadata)
    assert recorder.captures == []
    assert "spec_metadata" not in recorder.parent_kwargs[0]
    assert recorder.parent_kwargs[0]["fused_unpool_gather"] is True
    assert "capture_fold" not in recorder.parent_kwargs[0]
    view = metadata.captured_hidden_states[:8, 2 * HIDDEN : 3 * HIDDEN]
    assert torch.equal(view, result[0].to(torch.bfloat16))
    assert layer.add_ffn._capture_fold is None and layer.add_ffn._capture_stored is False
    assert recorder.warnings == []


def test_the_builtin_capture_runs_when_the_boundary_did_not_store(monkeypatch):
    layer, recorder = _layer(monkeypatch, 44)
    recorder.store = False
    result = _call(layer, _metadata())
    assert len(recorder.captures) == 1
    layer_id, states, residual = recorder.captures[0]
    assert layer_id == 44 and states is result[0] and residual is None
    assert "spec_metadata" not in recorder.parent_kwargs[0]
    # A folded layer says so, once per reason.
    assert len(recorder.warnings) == 1
    assert "layer 44" in recorder.warnings[0] and "fused boundary" in recorder.warnings[0]
    _call(layer, _metadata())
    assert len(recorder.captures) == 2 and len(recorder.warnings) == 1


@pytest.mark.parametrize(
    "case", ["feature_off", "layer_not_captured", "last_layer", "no_metadata", "nine_rows"]
)
def test_otherwise_the_builtin_layer_runs_unchanged(monkeypatch, case):
    # The last decoder layer is never folded (``folded_layers``).
    layer, recorder = _layer(
        monkeypatch, 2 if case == "layer_not_captured" else 16, folded=case != "last_layer"
    )
    metadata = None if case == "no_metadata" else _metadata()
    _call(
        layer,
        metadata,
        capture_fold=case != "feature_off",
        rows=9 if case == "nine_rows" else 8,
        next_norm=None if case == "last_layer" else object(),
    )
    assert recorder.parent_kwargs[0].get("spec_metadata") is metadata
    assert "capture_fold" not in recorder.parent_kwargs[0]
    assert recorder.captures == []
    assert layer.add_ffn._capture_fold is None
    # Nothing the boundary could have stored: the built-in capture writes nothing it
    # could have written (or the boundary is not fused at these rows either).
    assert recorder.warnings == []


def test_the_join_is_disarmed_when_the_layer_raises(monkeypatch):
    layer, _ = _layer(monkeypatch, 59)

    def failing(self, *args, **kwargs):
        assert self.add_ffn._capture_fold is not None
        raise RuntimeError("boom")

    monkeypatch.setattr(projection_gathers.MuseSparkOpenTepDecoderLayer, "forward", failing)
    with pytest.raises(RuntimeError, match="boom"):
        _call(layer, _metadata())
    assert layer.add_ffn._capture_fold is None and layer.add_ffn._capture_stored is False


@pytest.mark.parametrize(
    "case, phrase",
    [
        ("bf16_residual", "torch.bfloat16"),
        ("own_capture_method", "has its own maybe_capture_hidden_states"),
        ("no_next_norm", "no next-layer norm"),
        ("short_buffer", "capture buffer (4, 40960)"),
    ],
)
def test_a_folded_layer_logs_why_its_capture_falls_back(monkeypatch, case, phrase):
    layer, recorder = _layer(monkeypatch, 59)
    metadata = _metadata(rows=4) if case == "short_buffer" else _metadata()
    if case == "own_capture_method":

        class Other(DFlashSpecMetadata):
            def maybe_capture_hidden_states(self, layer_id, hidden_states, residual=None):
                pass

        other = object.__new__(Other)
        other.__dict__.update(metadata.__dict__)
        metadata = other
    for _ in range(3):
        layer(
            None,
            torch.randn(8, HIDDEN).to(torch.bfloat16 if case == "bf16_residual" else torch.float32),
            None,
            next_layer_norm=None if case == "no_next_norm" else object(),
            capture_fold=True,
            spec_metadata=metadata,
        )
    # The built-in layer ran with the metadata (and so made the capture itself).
    assert all(kwargs.get("spec_metadata") is metadata for kwargs in recorder.parent_kwargs)
    assert len(recorder.warnings) == 1, recorder.warnings
    assert "layer 59" in recorder.warnings[0] and phrase in recorder.warnings[0]
    # Same call on a layer install did not count as folded: silent.
    quiet, quiet_recorder = _layer(monkeypatch, 59, folded=False)
    quiet(
        None,
        torch.randn(8, HIDDEN),
        None,
        next_layer_norm=None,
        capture_fold=True,
        spec_metadata=_metadata(),
    )
    assert quiet_recorder.warnings == []


def _model(layer_class=projection_gathers.MuseSparkOpenTepDecoderLayer):
    layers = []
    for index in range(4):
        layer = object.__new__(layer_class)
        torch.nn.Module.__init__(layer)
        layer.layer_idx = index
        join = object.__new__(projection_gathers.MuseSparkOpenTepGatedResidual)
        torch.nn.Module.__init__(join)
        layer.add_ffn = join
        layers.append(layer)
    return types.SimpleNamespace(layers=layers)


def test_install_reclasses_every_layer():
    config = _spec_config(SpeculativeDecodingMode.DFLASH, [1, 3])
    # Layer 3 is the last of four: it has no fused boundary, so it keeps the built-in copy.
    assert capture_fold.folded_layers(_model(), config) == frozenset({1})
    model = _model()
    assert capture_fold.install(model, config) == {capture_fold.LAYER_KWARG: True}
    assert all(
        type(layer) is capture_fold.MuseSparkOpenTepCaptureDecoderLayer for layer in model.layers
    )
    assert [layer._capture_folded for layer in model.layers] == [False, True, False, False]
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenDecoderLayer

    with pytest.raises(ValueError):
        capture_fold.install(_model(MuseSparkOpenDecoderLayer), config)
    with pytest.raises(ValueError):
        capture_fold.install(_model(), None)


# --------------------------------------------------------------------------
# The early-release junction in ``adopt``: which capture layers it is told to keep
# --------------------------------------------------------------------------
#: The model's decoder layers; every one but the last has a boundary -> QKV edge.
NUM_LAYERS = 62


def _text_model(num_layers: int = NUM_LAYERS):
    """A built-in text model as far as ``adopt`` reads it with the projection gathers, the early
    release and the capture fold.

    The layers are the re-classed projection-gather decoder layers with the fused post-FFN join
    (an inbox set), as ``projection_gathers.install`` leaves them; that install is
    replaced by its result in ``junction``.
    """
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel

    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    torch.nn.Module.__init__(model)
    layers = []
    for index in range(num_layers):
        layer = object.__new__(projection_gathers.MuseSparkOpenTepDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.layer_idx = index
        join = object.__new__(projection_gathers.MuseSparkOpenTepGatedResidual)
        torch.nn.Module.__init__(join)
        join.__dict__["_gather_inbox"] = object()
        layer.add_ffn = join
        layers.append(layer)
    model.layers = torch.nn.ModuleList(layers)
    model.model_config = types.SimpleNamespace(
        spec_config=_spec_config(SpeculativeDecodingMode.DFLASH, [59, 1, 30, 16, 44])
    )
    return model


@pytest.fixture
def junction(monkeypatch):
    """``adopt`` with the projection gathers' install replaced by its result, and the early release's recorded.

    The stand-in layers have no attention, so the norm gate's trigger is taken
    away (no attention class); every other step of ``adopt`` that the projection
    gathers / early release / capture fold plan runs as in the model, including the install
    refusals and
    ``pdl_triggers.install_boundary_triggers`` itself.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
        triggers as pdl_triggers,
    )

    monkeypatch.setattr(pdl_triggers, "NORM_GATE_TRIGGER", None)
    monkeypatch.setattr(
        projection_gathers,
        "install",
        lambda model, unpool, pool: {"fused_unpool_gather": unpool, "fused_pool_gather": pool},
    )
    told = []
    install = pdl_triggers.install_boundary_triggers

    def recorded(model, captured=frozenset()):
        told.append(frozenset(captured))
        return install(model, captured)

    monkeypatch.setattr(pdl_triggers, "install_boundary_triggers", recorded)
    return told


def _adopt(*paths):
    model = _text_model()
    return modeling.MuseSparkOpenTepModelingV2.adopt(model, dispatch.plan_of(*paths)), model


def _triggers(model) -> list:
    return [layer.add_ffn._boundary_trigger for layer in model.layers[:-1]]


def test_with_capture_fold_early_release_keeps_no_capture_layer_and_converts_every_edge(junction):
    core, model = _adopt("boundary_early_release", "capture_fold")
    assert junction == [frozenset()]
    assert _triggers(model) == ["entry"] * (NUM_LAYERS - 1)
    assert core._layer_kwargs.get(capture_fold.LAYER_KWARG) is True
    assert all(
        type(layer) is capture_fold.MuseSparkOpenTepCaptureDecoderLayer for layer in model.layers
    )
    assert [layer.layer_idx for layer in model.layers if layer._capture_folded] == list(LAYERS)


def test_without_capture_fold_early_release_keeps_the_capture_layers(junction):
    core, model = _adopt("boundary_early_release")
    assert junction == [frozenset(LAYERS)]
    assert _triggers(model) == [
        "exit" if index in LAYERS else "entry" for index in range(NUM_LAYERS - 1)
    ]
    assert capture_fold.LAYER_KWARG not in core._layer_kwargs


# --------------------------------------------------------------------------
# The ``spec_metadata`` contract of the stored path
# --------------------------------------------------------------------------
def _classes_that_see_the_layer_keywords() -> list:
    """This target's classes on the path of a capture-fold layer's keywords to its attention.

    The layer classes between ``MuseSparkOpenTepCaptureDecoderLayer`` and the
    built-in layer, and every attention class ``adopt`` can install
    (``_attention_class`` over the prep x a decode-attention plan x the norm gate).
    """
    classes = set(capture_fold.MuseSparkOpenTepCaptureDecoderLayer.__mro__[1:])
    for preattn_prep_on in (False, True):
        for plan in (None, object()):
            for norm_gate in (False, True):
                attention = target_attention._attention_class(preattn_prep_on, plan, norm_gate)
                if attention is not None:
                    classes.update(attention.__mro__)
    package = modeling.__package__
    return sorted(
        (cls for cls in classes if cls.__module__.startswith(package)),
        key=lambda cls: (cls.__module__, cls.__qualname__),
    )
