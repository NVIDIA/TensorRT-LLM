# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.2 tp8 target's speculative worker: its adoption, the phase gate of every override,
the draft loop it runs on a decode step, and the draft layer's live-row arming.

Host-only: the built-in worker is created without its constructor and carries only the
attributes the overrides read. The kernel entry points are stand-ins, and the built-in methods
an override falls back to or calls are recorders.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import (  # noqa: E501
    dispatch,
    layers,
    spec_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_live_rows import (  # noqa: E501
    step0,
)
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker, MTPEagleWorker
from tensorrt_llm._torch.speculative.mtp_dynamic_tree import MTPEagleDynamicTreeWorker

pytestmark = pytest.mark.cpu_only

PREFILL, DECODE = dispatch.Phase.PREFILL, dispatch.Phase.DECODE
GlmMtpWorker = spec_worker.GlmMtpWorker
OVERRIDES = ["_forward_impl", "_forward_linear_draft_loop", "_run_draft_forward"]


def _bare(cls, **attributes):
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    for name, value in attributes.items():
        setattr(obj, name, value)
    return obj


def _keys(module):
    return (
        [name for name, _ in module.named_parameters()],
        [name for name, _ in module.named_buffers()],
        [name for name, _ in module.named_modules()],
    )


# --- adoption -----------------------------------------------------------------------------


def test_adoption_reclasses_the_builtin_worker_in_place():
    worker = _bare(MTPEagleWorker, is_mtp_eagle=True)
    worker.register_buffer("probe", torch.zeros(1))
    before = _keys(worker)

    assert GlmMtpWorker.adopt(worker) is worker

    assert type(worker) is GlmMtpWorker
    assert worker._draft_loop_ops is None and "_draft_loop_ops" not in vars(worker)
    assert _keys(worker) == before


def _other_worker():
    return _bare(type("_OtherMTPEagleWorker", (MTPEagleWorker,), {}))


@pytest.mark.parametrize(
    "make, got",
    [
        (lambda: _bare(Eagle3OneModelWorker), "Eagle3OneModelWorker"),
        (lambda: _bare(MTPEagleDynamicTreeWorker), "MTPEagleDynamicTreeWorker"),
        (_other_worker, "_OtherMTPEagleWorker"),
        (lambda: GlmMtpWorker.adopt(_bare(MTPEagleWorker)), "GlmMtpWorker"),
    ],
    ids=["the Eagle3 worker", "the dynamic-tree worker", "a subclass", "an adopted worker"],
)
def test_adoption_refuses_anything_but_the_exact_builtin_worker(make, got):
    with pytest.raises(TypeError, match=f"adopts the built-in MTPEagleWorker, got {got}$"):
        GlmMtpWorker.adopt(make())


def test_the_class_declares_no_state_of_its_own():
    assert not {"__slots__", "__init__", "forward"} & set(vars(GlmMtpWorker))
    assert GlmMtpWorker._draft_loop_ops is None


@pytest.mark.parametrize("has_worker", [True, False], ids=["a worker", "no worker"])
def test_the_worker_is_adopted_after_the_draft_layer(monkeypatch, has_worker):
    order = []
    for cls, label in (
        (layers.GlmDecoderLayer, "decoder layer"),
        (layers.GlmMtpLayer, "draft layer"),
        (GlmMtpWorker, "worker"),
    ):
        monkeypatch.setattr(
            cls, "adopt", classmethod(lambda c, module, label=label: order.append((label, module)))
        )
    core = types.SimpleNamespace(num_hidden_layers=2, layers=["layer 0", "layer 1", "draft"])
    worker = object() if has_worker else None

    modeling.adopt_model(types.SimpleNamespace(model=core, spec_worker=worker), plan=None)

    expected = [
        ("decoder layer", "layer 0"),
        ("decoder layer", "layer 1"),
        ("draft layer", "draft"),
    ]
    assert order == expected + ([("worker", worker)] if has_worker else [])


# --- the phase gate -----------------------------------------------------------------------


def _method(name):
    tree = ast.parse(Path(spec_worker.__file__).read_text())
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "GlmMtpWorker")
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize("name", OVERRIDES)
def test_each_override_starts_with_the_phase_gate(name):
    """``if dispatch.phase() is not DECODE: return super().<name>(<every built-in parameter by
    name>)``, after the built-in docstring if the method has one."""
    node = _method(name)
    docstring = isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
    gate = node.body[1] if docstring else node.body[0]
    assert ast.unparse(gate.test) == "dispatch.phase() is not dispatch.Phase.DECODE"
    assert len(gate.body) == 1 and not gate.orelse
    call = gate.body[0].value
    assert ast.unparse(call.func) == f"super().{name}" and not call.args
    parameters = inspect.signature(getattr(Eagle3OneModelWorker, name)).parameters
    assert [k.arg for k in call.keywords] == [p for p in parameters if p != "self"]
    assert all(k.arg == ast.unparse(k.value) for k in call.keywords)
    assert list(inspect.signature(getattr(GlmMtpWorker, name)).parameters) == list(parameters)


def _record(monkeypatch, owner, name, calls):
    def recorder(*args, **kwargs):
        calls.append((name, args, kwargs))
        return f"{owner.__name__}.{name}"

    monkeypatch.setattr(owner, name, recorder)


@pytest.mark.parametrize("phase", [PREFILL, None], ids=["prefill", "outside a step"])
def test_off_decode_every_override_runs_the_builtin_method(monkeypatch, phase):
    worker = GlmMtpWorker.adopt(_bare(MTPEagleWorker, is_mtp_eagle=True))
    calls = []
    for name in OVERRIDES:
        _record(monkeypatch, Eagle3OneModelWorker, name, calls)
    monkeypatch.setattr(dispatch, "is_enabled", lambda path: pytest.fail(path))
    monkeypatch.setattr(dispatch, "get_impl", lambda *a: pytest.fail(str(a)))

    with dispatch.step(phase) if phase is not None else contextlib.nullcontext():
        results = [
            worker._forward_impl("ids", "pos", "h", "logits", "md", "spec", "draft", "rm"),
            worker._forward_linear_draft_loop(
                "inputs", "md", "spec", "draft", "kv", 0, 2, "accepted", "ranks"
            ),
            worker._run_draft_forward("draft", "inputs", "spec", 1),
        ]

    assert results == [f"Eagle3OneModelWorker.{name}" for name in OVERRIDES]
    assert all(args == (worker,) for _, args, _ in calls)
    assert [(name, kwargs) for name, _, kwargs in calls] == [
        (
            "_forward_impl",
            dict(
                input_ids="ids",
                position_ids="pos",
                hidden_states="h",
                logits="logits",
                attn_metadata="md",
                spec_metadata="spec",
                draft_model="draft",
                resource_manager="rm",
            ),
        ),
        (
            "_forward_linear_draft_loop",
            dict(
                inputs="inputs",
                attn_metadata="md",
                spec_metadata="spec",
                draft_model="draft",
                draft_kv_cache_manager="kv",
                num_contexts=0,
                batch_size=2,
                num_accepted_tokens="accepted",
                original_all_rank_num_tokens="ranks",
            ),
        ),
        (
            "_run_draft_forward",
            dict(draft_model="draft", inputs="inputs", spec_metadata="spec", step_idx=1),
        ),
    ]
    assert "_draft_loop_ops" not in vars(worker)


# --- the decode draft loop ----------------------------------------------------------------

BATCH = 2
HIDDEN = 4


class _DraftLoopOps:
    """Stands in for the draft-loop operations of one forward; records each call."""

    def __init__(self, events, next_new_tokens="ops next new tokens"):
        self.events = events
        self.batch_size = BATCH
        self.gather_ids = torch.arange(BATCH, dtype=torch.int64)
        self.next_new_tokens = next_new_tokens

    def accept(self, logits):
        self.events.append(("accept", logits))
        return "ops accepted", "ops num accepted"

    def first_drafter_inputs(self, position_ids, hidden_states):
        self.events.append(("first_drafter_inputs", tuple(position_ids.shape), hidden_states))
        return "ops first inputs"

    def step0_rows(self, hidden_states):
        self.events.append(("step0_rows", hidden_states.shape[0]))
        return self.gather_ids, hidden_states[:BATCH]

    def step_update(self, attn_metadata, step, num_accepted, position_ids, gather_ids):
        self.events.append(("step_update", step, position_ids is not None, gather_ids is not None))
        return types.SimpleNamespace(position_ids=f"positions after step {step}")

    def finalize(self, tokens):
        self.events.append(("finalize", list(tokens)))
        return "ops stacked"


def _worker(events, **methods):
    worker = GlmMtpWorker.adopt(
        _bare(
            MTPEagleWorker,
            is_mtp_eagle=True,
            sa_enhancer=None,
            guided_decoder=None,
            model_config=None,
            _uses_external_shared_target_kv=False,
            _is_mamba_hybrid_cache=False,
            _auxiliary_state_handlers=[],
        )
    )
    for name, result in methods.items():

        def method(*args, _name=name, _result=result, **kwargs):
            events.append((_name,))
            return _result(worker) if callable(_result) else _result

        setattr(worker, name, method)
    return worker


class _FastPaths:
    """``dispatch.is_enabled`` / ``get_impl`` with stand-in draft-loop entry points."""

    def __init__(self, events, enabled, entries):
        self.events, self.enabled, self.entries = events, frozenset(enabled), entries

    def is_enabled(self, path):
        assert path in dispatch.FAST_PATH_NAMES, path
        return path in self.enabled

    def get_impl(self, path, name):
        assert path in self.enabled, path
        entry = self.entries[(path, name)]

        def call(*args):
            self.events.append((name,) + args)
            return entry(*args)

        return call


@pytest.mark.parametrize("mode", ["engaged", "ops without next new tokens", "declined", "off"])
def test_the_decode_forward_runs_the_draft_loop_operations(monkeypatch, mode):
    events = []
    ops = _DraftLoopOps(events, None if mode == "ops without next new tokens" else "ops tokens")
    worker = _worker(
        events,
        _execute_guided_decoder_if_present=None,
        sample_and_accept_draft_tokens=("accepted", "num accepted"),
        _prepare_attn_metadata_for_spec_dec=None,
        prepare_1st_drafter_inputs="first inputs",
        get_draft_kv_cache_manager="draft kv",
        _forward_draft_loop=lambda w: ("draft tokens", w._draft_loop_ops),
        _restore_attn_metadata_from_spec_dec=None,
        _prepare_next_new_tokens="next new tokens",
    )
    paths = _FastPaths(
        events,
        () if mode == "off" else {"draft_loop"},
        {("draft_loop", "draft_loop_ops"): lambda *a: None if mode == "declined" else ops},
    )
    monkeypatch.setattr(dispatch, "is_enabled", paths.is_enabled)
    monkeypatch.setattr(dispatch, "get_impl", paths.get_impl)
    md = types.SimpleNamespace(num_seqs=BATCH, num_contexts=0, all_rank_num_tokens=None)
    spec = types.SimpleNamespace(
        runtime_draft_len=2, spec_resource_manager=None, batch_indices_cuda="batch indices"
    )

    with dispatch.step(DECODE):
        out = worker._forward_impl(
            "ids", torch.zeros(1, 6), "hidden", "logits", md, spec, "draft model", "rm"
        )

    engaged = mode in ("engaged", "ops without next new tokens")
    start = [("_execute_guided_decoder_if_present",)]
    if mode != "off":
        start.append(("draft_loop_ops", worker, md, spec))
    if engaged:
        assert events == start + [
            ("accept", "logits"),
            ("_prepare_attn_metadata_for_spec_dec",),
            ("first_drafter_inputs", (6,), "hidden"),
            ("get_draft_kv_cache_manager",),
            ("_forward_draft_loop",),
            ("_restore_attn_metadata_from_spec_dec",),
        ] + ([("_prepare_next_new_tokens",)] if ops.next_new_tokens is None else [])
        assert out["new_tokens"] == "ops accepted"
        assert out["new_tokens_lens"] == "ops num accepted"
        assert out["next_draft_tokens"] == ("draft tokens", ops)
        assert out["next_new_tokens"] == (ops.next_new_tokens or "next new tokens")
    else:
        assert events == start + [
            ("sample_and_accept_draft_tokens",),
            ("_prepare_attn_metadata_for_spec_dec",),
            ("prepare_1st_drafter_inputs",),
            ("get_draft_kv_cache_manager",),
            ("_forward_draft_loop",),
            ("_restore_attn_metadata_from_spec_dec",),
            ("_prepare_next_new_tokens",),
        ]
        assert out["new_tokens"] == "accepted"
        assert out["next_draft_tokens"] == ("draft tokens", None)
        assert out["next_new_tokens"] == "next new tokens"
    # The operations live for one forward only.
    assert worker._draft_loop_ops is None and md.use_spec_decoding is True
    assert out["logits"] == "logits"


class _DraftLayerForward:
    """The draft layer's decode forward as the draft loop sees it: it records the rows it was
    armed with and, when armed, returns only the live rows, as the live-row hop does."""

    def __init__(self, layer, compacts):
        self.layer, self.compacts, self.calls = layer, compacts, []

    def __call__(self, **kwargs):
        rows = self.layer._draft_live_rows
        self.calls.append((rows, kwargs["embed_tokens"], kwargs["all_rank_num_tokens"]))
        if rows is not None and self.compacts:
            self.layer._draft_live_rows_compacted = True
            return torch.ones(rows.shape[0], HIDDEN)
        return torch.zeros(3 * BATCH if len(self.calls) == 1 else BATCH, HIDDEN)


@pytest.mark.parametrize("compacts", [True, False], ids=["compacted", "kept every row"])
def test_the_decode_draft_loop_runs_the_draft_fast_paths(monkeypatch, compacts):
    """Each draft-loop fast path is decided once before the steps; step 0 runs armed with the
    draft-loop operations' live rows; the draft tail makes each step's tokens; the step updates
    replace the native position advance and metadata update."""
    events = []
    ops = _DraftLoopOps(events)
    draft_layer = _bare(layers.GlmMtpLayer)
    forward = _DraftLayerForward(draft_layer, compacts)
    draft_layer.forward = forward
    draft_model = types.SimpleNamespace(embed_tokens="embedding table", mtp_layers=[draft_layer])
    worker = _worker(events, draft_kv_cache_context=lambda w: contextlib.nullcontext(md))
    worker._draft_loop_ops = ops
    tail_tokens = iter(["tokens 0", "tokens 1"])
    paths = _FastPaths(
        events,
        {"draft_loop", "draft_tail", "draft_live_rows"},
        {
            ("draft_tail", "select_draft_tail"): lambda *a: (
                lambda rows: (events.append(("draft tail", rows.shape[0])), next(tail_tokens))[1]
            ),
            ("draft_live_rows", "select_step0"): lambda w, dm, s, o: step0.Step0LiveRows(
                dm.mtp_layers[0], o.gather_ids
            ),
            ("draft_loop", "draft_embedding"): lambda table: ("draft embedding", table),
        },
    )
    monkeypatch.setattr(dispatch, "is_enabled", paths.is_enabled)
    monkeypatch.setattr(dispatch, "get_impl", paths.get_impl)
    md = types.SimpleNamespace()
    spec = types.SimpleNamespace(
        runtime_draft_len=2,
        all_rank_num_tokens=[3 * BATCH],
        subseq_all_rank_num_tokens=[BATCH],
        wants_advanced_draft_sampling=False,
    )
    inputs = {"input_ids": "ids", "position_ids": "positions", "hidden_states": "h"}

    with dispatch.step(DECODE):
        out = worker._forward_linear_draft_loop(
            inputs, md, spec, draft_model, None, 0, BATCH, "num accepted", None
        )

    assert out == "ops stacked"
    assert events[0] == ("select_draft_tail", worker, draft_model, spec)
    assert events[1] == ("select_step0", worker, draft_model, spec, ops)
    assert events[2] == ("draft_kv_cache_context",)
    step_events = [e for e in events if e[0] in ("step0_rows", "draft tail", "step_update")]
    assert step_events == ([] if compacts else [("step0_rows", 3 * BATCH)]) + [
        ("draft tail", BATCH),
        ("step_update", 0, True, True),
        ("draft tail", BATCH),
        ("step_update", 1, False, False),
    ]
    assert events[-1] == ("finalize", ["tokens 0", "tokens 1"])
    # Step 0 ran armed with the live rows; step 1 did not; the layer is disarmed afterwards.
    armed = [rows for rows, _, _ in forward.calls]
    assert armed[0] is ops.gather_ids and armed[1] is None
    assert [c[1:] for c in forward.calls] == [
        (("draft embedding", "embedding table"), [3 * BATCH]),
        (("draft embedding", "embedding table"), [BATCH]),
    ]
    assert draft_layer._draft_live_rows is None and draft_layer._draft_live_rows_compacted is False


@pytest.mark.parametrize("draft_loop", [True, False], ids=["draft loop", "no draft loop"])
def test_the_decode_draft_forward_takes_the_draft_loop_embedding(monkeypatch, draft_loop):
    events = []
    calls = []
    layer = lambda **kwargs: calls.append(kwargs) or "draft layer output"  # noqa: E731
    draft_model = types.SimpleNamespace(embed_tokens="embedding table", mtp_layers=[layer])
    worker = _worker(events)
    paths = _FastPaths(
        events,
        {"draft_loop"} if draft_loop else (),
        {("draft_loop", "draft_embedding"): lambda table: ("draft embedding", table)},
    )
    monkeypatch.setattr(dispatch, "is_enabled", paths.is_enabled)
    monkeypatch.setattr(dispatch, "get_impl", paths.get_impl)
    spec = types.SimpleNamespace(all_rank_num_tokens=[6], subseq_all_rank_num_tokens=[2])

    with dispatch.step(DECODE):
        out = worker._run_draft_forward(draft_model, {"input_ids": "ids"}, spec, 0)

    assert out == ("draft layer output", None)
    table = ("draft embedding", "embedding table") if draft_loop else "embedding table"
    assert calls == [dict(embed_tokens=table, all_rank_num_tokens=[6], input_ids="ids")]


# --- the draft layer's live-row arming ----------------------------------------------------


def _recording_draft_layer(writes):
    class _Recording(layers.GlmMtpLayer):
        def __setattr__(self, name, value):
            if name.startswith("_draft_live_rows"):
                writes.append((name, value))
            super().__setattr__(name, value)

    return _bare(_Recording)


def test_the_draft_layer_is_armed_and_disarmed_through_its_api():
    layer = _bare(layers.GlmMtpLayer)
    rows = torch.arange(BATCH)
    layer.arm_live_rows(rows)
    assert layer._draft_live_rows is rows and layer._draft_live_rows_compacted is False
    layer._draft_live_rows_compacted = True
    assert layer.disarm_live_rows() is True
    assert layer._draft_live_rows is None and layer._draft_live_rows_compacted is False
    assert layer.disarm_live_rows() is False


@pytest.mark.parametrize("outcome", ["compacted", "kept every row", "raised"])
def test_step0_arming_writes_the_layer_state_in_the_builtin_order(outcome):
    """Arm (rows, then not compacted), the draft layer's own write, then clear (rows, then
    compacted), also when the step raises."""
    writes = []
    layer = _recording_draft_layer(writes)
    rows = torch.arange(BATCH)
    live = step0.Step0LiveRows(layer, rows)

    with pytest.raises(RuntimeError) if outcome == "raised" else contextlib.nullcontext():
        with live.armed() as armed:
            assert armed is live and layer._draft_live_rows is rows
            if outcome != "kept every row":
                layer._draft_live_rows_compacted = True
            if outcome == "raised":
                raise RuntimeError("draft step failed")

    assert live.compacted is (outcome == "compacted")
    own = [("_draft_live_rows_compacted", True)] if outcome != "kept every row" else []
    assert writes == [
        ("_draft_live_rows", rows),
        ("_draft_live_rows_compacted", False),
        *own,
        ("_draft_live_rows", None),
        ("_draft_live_rows_compacted", False),
    ]
