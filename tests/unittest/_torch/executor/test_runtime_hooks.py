# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the engine reads the step-loop hooks a model declares (``runtime_hooks_of``).

The engine's call sites run their original statements when a hook is off: every
object a hook attaches to declares it off at class level, every call a call
site makes on a hook is a method of the hook's protocol, every read of a hook in
the engine files is decided by its all-off value toward the original statements
(checked on the source), and call sites of each hook run on stand-ins issue
their original work with the hook off and hand it to the hook when it is on.
"""

import ast
import contextlib
import dataclasses
import inspect
import types
from pathlib import Path

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    ExecutorEvents,
    RuntimeHooks,
    SamplerStoreWriter,
    StepInputStager,
    runtime_hooks_of,
)

pytestmark = pytest.mark.cpu_only


class _PlainModel(torch.nn.Module):
    """A model that declares no hooks, like every model outside the opt-in."""


class _ClassHooksModel(torch.nn.Module):
    runtime_hooks = RuntimeHooks(loop_on_execution_stream=True)


class _Events:
    def after_warmup(self) -> None:
        pass

    def before_overlap_loop(self) -> None:
        pass

    def at_shutdown(self) -> None:
        pass


@pytest.mark.parametrize("model", [_PlainModel(), object()], ids=["module", "object"])
def test_a_model_without_hooks_gets_the_all_off_default(model):
    hooks = runtime_hooks_of(model)

    assert hooks is DEFAULT_RUNTIME_HOOKS
    assert hooks == RuntimeHooks()
    assert not hooks.loop_on_execution_stream
    assert hooks.step_input_stager is None
    assert hooks.sampler_store_writer is None
    assert hooks.decode_prologue is None
    assert hooks.executor_events is None


def test_every_field_defaults_to_off():
    for field in dataclasses.fields(RuntimeHooks):
        assert getattr(DEFAULT_RUNTIME_HOOKS, field.name) in (False, None), field.name


def test_hooks_declared_as_none_mean_the_default():
    model = _PlainModel()
    model.runtime_hooks = None

    assert runtime_hooks_of(model) is DEFAULT_RUNTIME_HOOKS


def test_class_attribute_hooks_are_returned_as_declared():
    assert runtime_hooks_of(_ClassHooksModel()) is _ClassHooksModel.runtime_hooks


def test_instance_attribute_hooks_are_returned_as_declared():
    hooks = RuntimeHooks(executor_events=_Events())
    model = _PlainModel()
    model.runtime_hooks = hooks

    assert runtime_hooks_of(model) is hooks


def test_instance_attribute_overrides_class_attribute():
    hooks = RuntimeHooks()
    model = _ClassHooksModel()
    model.runtime_hooks = hooks

    assert runtime_hooks_of(model) is hooks


@pytest.mark.parametrize(
    "declared",
    [True, {"loop_on_execution_stream": True}, "RuntimeHooks()"],
    ids=["bool", "dict", "str"],
)
def test_hooks_of_the_wrong_type_raise(declared):
    model = _PlainModel()
    model.runtime_hooks = declared

    with pytest.raises(TypeError, match=r"_PlainModel\.runtime_hooks must be a RuntimeHooks"):
        runtime_hooks_of(model)


def test_hooks_are_immutable_once_declared():
    hooks = RuntimeHooks()

    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.loop_on_execution_stream = True


def test_engine_classes_declare_the_stager_off():
    """Every object a hook attaches to declares the attribute off at class level."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelSpecMetadata
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    for cls in (
        TrtllmAttentionMetadata,
        DSAtrtllmAttentionMetadata,
        SpecMetadata,
        Eagle3OneModelSpecMetadata,
    ):
        assert cls.step_input_stager is None, cls
        # A class attribute, not a dataclass field: the constructors do not take it.
        assert "step_input_stager" not in {f.name for f in dataclasses.fields(cls)}, cls
    assert DecoderInputPreparer.step_input_stager is None
    assert DecoderInputPreparer.decode_prologue is None
    assert DecoderRunner.runtime_hooks is DEFAULT_RUNTIME_HOOKS
    assert KVCacheManagerV2.step_input_stager is None
    assert CUDAGraphRunner.staged_static_input_tokens == 0
    assert SpecSampler.store_writer is None
    assert PyExecutor._execution_stream_loop_armed is False
    assert PyExecutor._loop_on_execution_stream is False
    assert PyExecutor._executor_events is None


# Per hook protocol: the local names and attribute names a call site holds the hook in, and
# the engine files with call sites.
_HOOK_CONSUMERS = {
    StepInputStager: (
        ("stager", "stager_source", "step_input_stager"),
        ("step_input_stager",),
        (
            "pyexecutor/engine/runners/decoder/runner.py",
            "pyexecutor/engine/runners/decoder/inputs.py",
            "pyexecutor/engine/runners/common.py",
            "pyexecutor/kv_cache/kv_cache_manager_v2.py",
            "attention/backends/trtllm.py",
            "attention/backends/sparse/dsa/metadata.py",
            "attention/backends/sparse/dsa/indexer.py",
            "speculative/eagle3.py",
            "speculative/interface.py",
        ),
    ),
    SamplerStoreWriter: (
        ("store_writer",),
        ("store_writer",),
        ("speculative/spec_sampler_base.py",),
    ),
    ExecutorEvents: ((), ("_executor_events",), ("pyexecutor/py_executor.py",)),
}


def _protocol_methods(protocol) -> set:
    return {
        name
        for name, value in vars(protocol).items()
        if inspect.isfunction(value) and not name.startswith("_")
    }


def _hook_calls(tree: ast.AST, local_names, attr_names):
    """``(method, line)`` of every call on a hook held in one of the locals or attributes."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        receiver = node.func.value
        if (isinstance(receiver, ast.Name) and receiver.id in local_names) or (
            isinstance(receiver, ast.Attribute) and receiver.attr in attr_names
        ):
            yield node.func.attr, node.lineno


@pytest.mark.parametrize("protocol", list(_HOOK_CONSUMERS), ids=lambda p: p.__name__)
def test_every_hook_call_site_uses_its_protocol(protocol):
    """A call site calls only its hook's protocol methods, and the protocol has no unused method."""
    local_names, attr_names, consumers = _HOOK_CONSUMERS[protocol]
    root = Path(tensorrt_llm.__file__).resolve().parent / "_torch"
    methods = _protocol_methods(protocol)
    used, unknown = set(), []
    for rel in consumers:
        tree = ast.parse((root / rel).read_text())
        calls = list(_hook_calls(tree, local_names, attr_names))
        assert calls, f"{rel} has no {protocol.__name__} call site"
        for method, line in calls:
            used.add(method)
            if method not in methods:
                unknown.append(f"{rel}:{line}: {method}")
    assert not unknown, f"calls outside the {protocol.__name__} protocol:\n" + "\n".join(unknown)
    assert used == methods


class _StoreWriterModel(torch.nn.Module):
    def __init__(self, writer) -> None:
        super().__init__()
        self.runtime_hooks = RuntimeHooks(sampler_store_writer=writer)


def _instantiate_sampler(monkeypatch, model, sampler):
    """``instantiate_sampler`` for a speculative engine whose spec decoder is ``sampler``."""
    from tensorrt_llm._torch.pyexecutor import _util

    monkeypatch.setattr(_util, "confidential_compute_enabled", lambda: False)
    monkeypatch.setattr(_util, "resolve_max_num_sequences", lambda *args, **kwargs: 8)
    monkeypatch.setattr(_util, "create_torch_sampler_args", lambda **kwargs: "sampler args")
    monkeypatch.setattr(_util, "get_spec_decoder", lambda args, spec_config: sampler)
    engine = types.SimpleNamespace(
        max_seq_len=64,
        spec_config=types.SimpleNamespace(
            spec_dec_mode=types.SimpleNamespace(has_spec_decoder=lambda: True)
        ),
        model=model,
    )
    llm_args = types.SimpleNamespace(
        sampler_force_async_worker=False,
        disable_overlap_scheduler=False,
        enable_speculative_beam_history_d2h=False,
    )
    return _util.instantiate_sampler(
        engine,
        llm_args,
        None,
        max_batch_size=8,
        max_beam_width=1,
        mm_encoder_only=False,
        speculative_config=None,
    )


def test_instantiate_sampler_binds_no_writer_without_hooks(monkeypatch):
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    sampler = SpecSampler.__new__(SpecSampler)

    assert _instantiate_sampler(monkeypatch, _PlainModel(), sampler) is sampler
    assert "store_writer" not in vars(sampler)
    assert sampler.store_writer is None


def test_instantiate_sampler_binds_the_declared_writer(monkeypatch):
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    writer = object()
    sampler = SpecSampler.__new__(SpecSampler)

    assert _instantiate_sampler(monkeypatch, _StoreWriterModel(writer), sampler) is sampler
    assert sampler.store_writer is writer


def test_executor_events_protocol():
    assert _protocol_methods(ExecutorEvents) == {
        "after_warmup",
        "before_overlap_loop",
        "at_shutdown",
    }
    events = _Events()
    assert RuntimeHooks(executor_events=events).executor_events is events


class _StreamRecorder:
    """Stands in for ``torch.cuda.stream``: records the streams it is entered with."""

    def __init__(self) -> None:
        self.entered = []

    @contextlib.contextmanager
    def __call__(self, stream):
        self.entered.append(stream)
        yield


def _scope_executor(armed: bool, **overrides):
    """A ``PyExecutor`` with only the attributes ``_overlap_loop_stream_scope`` reads."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = PyExecutor.__new__(PyExecutor)
    executor._execution_stream_loop_armed = armed
    executor.execution_stream = object()
    attrs = dict(
        drafter=None,
        draft_model_engine=None,
        guided_decoder=None,
        kv_connector_manager=None,
        is_encoder_decoder=False,
        _mm_encoder_item_scheduling_enabled=False,
    )
    attrs.update(overrides)
    for name, value in attrs.items():
        setattr(executor, name, value)
    return executor


def test_overlap_loop_scope_is_a_no_op_when_not_armed(monkeypatch):
    executor = _scope_executor(armed=False)
    recorder = _StreamRecorder()
    monkeypatch.setattr(torch.cuda, "stream", recorder)

    with executor._overlap_loop_stream_scope():
        assert executor._loop_on_execution_stream is False
    assert recorder.entered == []
    assert "_loop_on_execution_stream" not in vars(executor)


def test_overlap_loop_scope_engages_when_armed(monkeypatch):
    executor = _scope_executor(armed=True)
    recorder = _StreamRecorder()
    monkeypatch.setattr(torch.cuda, "stream", recorder)

    with executor._overlap_loop_stream_scope():
        assert executor._loop_on_execution_stream is True
    assert recorder.entered == [executor.execution_stream]
    assert executor._loop_on_execution_stream is False


def test_overlap_loop_scope_resets_the_flag_when_the_loop_raises(monkeypatch):
    executor = _scope_executor(armed=True)
    monkeypatch.setattr(torch.cuda, "stream", _StreamRecorder())

    with pytest.raises(RuntimeError, match="loop failed"):
        with executor._overlap_loop_stream_scope():
            raise RuntimeError("loop failed")
    assert executor._loop_on_execution_stream is False


@pytest.mark.parametrize(
    "name, value",
    [
        ("drafter", object()),
        ("draft_model_engine", object()),
        ("guided_decoder", object()),
        ("kv_connector_manager", object()),
        ("is_encoder_decoder", True),
        ("_mm_encoder_item_scheduling_enabled", True),
    ],
)
def test_overlap_loop_scope_stays_off_with_side_streams(monkeypatch, name, value):
    executor = _scope_executor(armed=True, **{name: value})
    recorder = _StreamRecorder()
    monkeypatch.setattr(torch.cuda, "stream", recorder)

    with executor._overlap_loop_stream_scope():
        assert executor._loop_on_execution_stream is False
    assert recorder.entered == []


def test_instantiate_sampler_leaves_other_spec_decoders_alone(monkeypatch):
    sampler = types.SimpleNamespace()

    assert _instantiate_sampler(monkeypatch, _StoreWriterModel(object()), sampler) is sampler
    assert "store_writer" not in vars(sampler)


# ---------------------------------------------------------------------------------------------
# With every hook off, every call site runs its original statements.
#
# The check below reads the engine files that consume a hook and evaluates each read of a hook
# at its all-off value. An ``if`` that reads a hook must be decided by that value; the branch it
# takes is checked like any other code and the other branch is not run, so it may use the hook
# freely. Outside such tests, live code may only bind a hook value to a local (or to an
# attribute that declares the hook) at its all-off value. Any other read is reported with its
# file and line.
# ---------------------------------------------------------------------------------------------

_SEAM_CONSUMERS = (
    "pyexecutor/engine/runners/decoder/runner.py",
    "pyexecutor/engine/runners/decoder/inputs.py",
    "pyexecutor/cuda_graph_runner.py",
    "pyexecutor/kv_cache/kv_cache_manager_v2.py",
    "pyexecutor/py_executor.py",
    "pyexecutor/_util.py",
    "attention/backends/trtllm.py",
    "attention/backends/sparse/dsa/metadata.py",
    "attention/backends/sparse/dsa/indexer.py",
    "speculative/eagle3.py",
    "speculative/interface.py",
    "speculative/spec_sampler_base.py",
)

# Attributes that hold a hook, and their value when every hook is off.
_SEAM_ATTRS = {
    "step_input_stager": None,
    "store_writer": None,
    "_executor_events": None,
    "_loop_on_execution_stream": False,
    "_execution_stream_loop_armed": False,
    "staged_static_input_tokens": 0,
    "runtime_hooks": DEFAULT_RUNTIME_HOOKS,
    "decode_prologue": None,
}
# Locals the call sites hold a hook (or a decision on one) in.
_SEAM_LOCALS = ("stager", "stager_source", "store_writer", "prologue", "staged", "staged_store")
_HOOK_FIELDS = {field.name for field in dataclasses.fields(RuntimeHooks)}


class _Unknown:
    """A value the all-off evaluation cannot decide."""


_UNKNOWN = _Unknown()


class _AllOffWalk:
    """Walks one engine file with every hook at its all-off value."""

    def __init__(self, rel: str) -> None:
        self.rel = rel
        self.problems = []
        self.decided = 0

    def report(self, node: ast.AST, what: str) -> None:
        self.problems.append(f"{self.rel}:{node.lineno}: {what}")

    # -- reads of a hook ------------------------------------------------------------------
    @staticmethod
    def _is_seam_call(node: ast.AST) -> bool:
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            return False
        if node.func.id == "runtime_hooks_of":
            return True
        return (
            node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "step_input_stager"
        )

    def reads_seam(self, node: ast.AST, env: dict) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in _SEAM_ATTRS:
                return True
            if isinstance(sub, ast.Name) and sub.id in env:
                return True
            if self._is_seam_call(sub):
                return True
        return False

    def value(self, node: ast.AST, env: dict):
        """The all-off value of ``node``, or ``_UNKNOWN``."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id, _UNKNOWN)
        if isinstance(node, ast.Attribute):
            receiver = self.value(node.value, env)
            if receiver is DEFAULT_RUNTIME_HOOKS and node.attr in _HOOK_FIELDS:
                return getattr(DEFAULT_RUNTIME_HOOKS, node.attr)
            if node.attr in _SEAM_ATTRS:
                return _SEAM_ATTRS[node.attr]
            return _UNKNOWN
        if self._is_seam_call(node):
            return DEFAULT_RUNTIME_HOOKS if node.func.id == "runtime_hooks_of" else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = self.value(node.operand, env)
            return _UNKNOWN if operand is _UNKNOWN else not operand
        if isinstance(node, ast.BoolOp):
            values = [self.value(v, env) for v in node.values]
            known = [v for v in values if v is not _UNKNOWN]
            if isinstance(node.op, ast.And):
                if any(not v for v in known):
                    return False
                return True if len(known) == len(values) else _UNKNOWN
            if any(known):
                return True
            return False if len(known) == len(values) else _UNKNOWN
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = self.value(node.left, env), self.value(node.comparators[0], env)
            op = node.ops[0]
            if isinstance(op, (ast.Is, ast.IsNot)) and left is not _UNKNOWN:
                if isinstance(node.comparators[0], ast.Constant) and right is None:
                    return (left is None) == isinstance(op, ast.Is)
            if (
                isinstance(op, ast.Eq)
                and isinstance(node.left, ast.Attribute)
                and node.left.attr == "staged_static_input_tokens"
            ):
                # No tokens were staged; a graph replay has at least one token.
                return False
        return _UNKNOWN

    # -- statements -------------------------------------------------------------------------
    def block(self, body, env: dict) -> bool:
        """Walk live statements; True when the block always ends in return or raise."""
        for stmt in body:
            if self.stmt(stmt, env):
                return True
        return False

    def bind(self, stmt, targets, value, env: dict) -> None:
        reads = self.reads_seam(value, env)
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        attrs = [t for t in targets if isinstance(t, ast.Attribute) and t.attr in _SEAM_ATTRS]
        if not (reads or attrs or any(name in env or name in _SEAM_LOCALS for name in names)):
            return
        if len(names) + len(attrs) != len(targets):
            self.report(stmt, "a hook value is bound to something other than a local")
            return
        off = self.value(value, env)
        if off is _UNKNOWN:
            if reads or attrs:
                self.report(stmt, "a hook value with every hook off is not decided")
            for name in names:
                env.pop(name, None)
            return
        for name in names:
            env[name] = off
        for attr in attrs:
            if off != _SEAM_ATTRS[attr.attr]:
                self.report(stmt, f"{attr.attr} is set to a value other than its all-off value")

    def stmt(self, stmt: ast.stmt, env: dict) -> bool:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.block(stmt.body, dict(env))
            return False
        if isinstance(stmt, ast.ClassDef):
            self.block(stmt.body, {})
            return False
        if isinstance(stmt, (ast.If, ast.While)):
            if not self.reads_seam(stmt.test, env):
                body_ends = self.block(stmt.body, env)
                else_ends = self.block(stmt.orelse, env)
                return isinstance(stmt, ast.If) and body_ends and else_ends
            taken = self.value(stmt.test, env)
            if taken is _UNKNOWN:
                self.report(stmt, "a test of a hook is not decided with every hook off")
                return False
            self.decided += 1
            if isinstance(stmt, ast.While) and taken:
                self.report(stmt, "a loop on a hook runs with every hook off")
                return False
            return self.block(stmt.body if taken else stmt.orelse, env)
        if isinstance(stmt, ast.Assign):
            self.bind(stmt, stmt.targets, stmt.value, env)
            return False
        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self.bind(stmt, [stmt.target], stmt.value, env)
            return False
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if self.reads_seam(item.context_expr, env):
                    self.report(stmt, "a hook is entered as a context")
            return self.block(stmt.body, env)
        if isinstance(stmt, ast.Try):
            ends = self.block(stmt.body, env)
            for handler in stmt.handlers:
                self.block(handler.body, env)
            self.block(stmt.orelse, env)
            return self.block(stmt.finalbody, env) or (ends and not stmt.handlers)
        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            if self.reads_seam(stmt.iter, env):
                self.report(stmt, "a loop iterates over a hook")
            self.block(stmt.body, env)
            self.block(stmt.orelse, env)
            return False
        if self.reads_seam(stmt, env):
            self.report(stmt, "a hook is used outside a test decided with every hook off")
        return isinstance(stmt, (ast.Return, ast.Raise))


@pytest.mark.parametrize("rel", _SEAM_CONSUMERS)
def test_every_seam_read_has_a_native_branch(rel):
    """With every hook off, each call site takes the branch with its original statements."""
    root = Path(tensorrt_llm.__file__).resolve().parent / "_torch"
    walk = _AllOffWalk(rel)
    walk.block(ast.parse((root / rel).read_text()).body, {})
    assert not walk.problems, "\n".join(walk.problems)
    assert walk.decided, f"{rel} has no call site of a hook"


def test_the_seam_check_reports_undecided_reads():
    source = """
def step(self, x):
    stager = self.step_input_stager
    if stager is not None:
        stager.copy(x)
    else:
        x.copy_()
    stager.flush()
    if x or stager:
        pass
    self.step_input_stager = x
"""
    walk = _AllOffWalk("sample.py")
    walk.block(ast.parse(source).body, {})
    assert walk.decided == 1
    assert walk.problems == [
        "sample.py:8: a hook is used outside a test decided with every hook off",
        "sample.py:9: a test of a hook is not decided with every hook off",
        "sample.py:11: a hook value with every hook off is not decided",
    ]


def test_the_decode_prologue_call_site_uses_its_protocol():
    """The input preparer calls the prologue with the protocol's positional and keyword parameters."""
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import DecodePrologue

    signature = inspect.signature(DecodePrologue.__call__)
    positional = [
        p.name
        for p in signature.parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.name != "self"
    ]
    keywords = {
        p.name for p in signature.parameters.values() if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    root = Path(tensorrt_llm.__file__).resolve().parent / "_torch"
    tree = ast.parse((root / "pyexecutor/engine/runners/decoder/inputs.py").read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "prologue"
    ]
    assert len(calls) == 1
    assert len(calls[0].args) == len(positional)
    assert {keyword.arg for keyword in calls[0].keywords} == keywords


# -- the decode prologue ------------------------------------------------------------------------


class _AttnMetadata:
    def __init__(self) -> None:
        self.kv_lens_updates = 0

    def on_update_kv_lens(self) -> None:
        self.kv_lens_updates += 1


def _prologue_preparer(prologue):
    """A ``DecoderInputPreparer`` with only what ``preprocess_inputs`` reads without spec decoding."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    preparer = DecoderInputPreparer.__new__(DecoderInputPreparer)
    if prologue is not None:
        preparer.decode_prologue = prologue
    preparer.guided_decoder = None
    preparer._config = types.SimpleNamespace(disable_overlap_scheduler=True)
    preparer.mapping = types.SimpleNamespace(has_cp_helix=lambda: False)
    return preparer


def test_preprocess_inputs_runs_the_native_prologue_without_a_hook():
    preparer = _prologue_preparer(None)
    inputs = {"attn_metadata": _AttnMetadata()}

    assert preparer.decode_prologue is None
    assert preparer.preprocess_inputs(inputs, enable_spec_decode=False, runtime_draft_len=0) is (
        inputs
    )
    assert inputs["attn_metadata"].kv_lens_updates == 1


@pytest.mark.parametrize("engaged", [False, True], ids=["declined", "engaged"])
def test_preprocess_inputs_keeps_the_native_prologue_when_the_hook_declines(engaged):
    calls = []

    def prologue(preparer, inputs, *, enable_spec_decode, runtime_draft_len):
        calls.append((preparer, inputs, enable_spec_decode, runtime_draft_len))
        return engaged

    preparer = _prologue_preparer(prologue)
    inputs = {"attn_metadata": _AttnMetadata()}

    assert preparer.preprocess_inputs(inputs, enable_spec_decode=False, runtime_draft_len=3) is (
        inputs
    )
    assert calls == [(preparer, inputs, False, 3)]
    assert inputs["attn_metadata"].kv_lens_updates == (0 if engaged else 1)


# -- the execution-stream loop --------------------------------------------------------------------


class _Stream:
    def __init__(self, name: str, log: list) -> None:
        self.name, self.log = name, log

    def wait_stream(self, other: "_Stream") -> None:
        self.log.append(f"{self.name} waits for {other.name}")


def _forward_step_executor(monkeypatch, log: list):
    """A ``PyExecutor`` with only what ``_forward_step`` reads, logging its stream work."""
    from tensorrt_llm._torch.pyexecutor import py_executor
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    current = _Stream("current", log)

    @contextlib.contextmanager
    def stream(s):
        log.append(f"enter {s.name}")
        yield
        log.append(f"leave {s.name}")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current)
    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(py_executor, "nvtx_range", lambda message: (lambda fn: fn))

    executor = PyExecutor.__new__(PyExecutor)
    executor.iter_counter = 0
    executor.execution_stream = _Stream("execution", log)
    executor.resource_manager = object()
    executor.sampler = types.SimpleNamespace(get_cache_indirection=lambda: None)
    executor.model_engine = types.SimpleNamespace(
        forward=lambda *args, **kwargs: log.append("forward") or "outputs"
    )
    executor._iter_adp_dummy_ctx_tokens = 0
    executor._iter_adp_dummy_gen_tokens = 0
    executor._compute_adp_dummy_tokens = lambda requests: (0, 0)
    executor._maybe_record_hang_diagnostic_phase = lambda *args, **kwargs: None
    executor._attach_encoder_output_to_execution_stream = lambda requests: None
    executor._mark_cross_kv_projection_consumed = lambda requests: None
    executor._kv_connector_wait_for_save = lambda: None
    return executor


_NO_REQUESTS = types.SimpleNamespace(
    context_requests=[], num_context_requests=0, num_generation_requests=0
)


def test_forward_step_orders_the_streams_off_the_execution_stream_loop(monkeypatch):
    log = []
    executor = _forward_step_executor(monkeypatch, log)

    assert executor._forward_step(_NO_REQUESTS) == "outputs"
    assert log == [
        "execution waits for current",
        "enter execution",
        "forward",
        "leave execution",
        "current waits for execution",
    ]


def test_forward_step_skips_the_waits_on_the_execution_stream_loop(monkeypatch):
    log = []
    executor = _forward_step_executor(monkeypatch, log)
    executor._loop_on_execution_stream = True

    assert executor._forward_step(_NO_REQUESTS) == "outputs"
    assert log == ["enter execution", "forward", "leave execution"]


# -- executor events --------------------------------------------------------------------------


def _cleanup_executor(events, completed: bool):
    """A ``PyExecutor`` with only what ``_executor_loop_cleanup`` reads, without pipeline stages."""
    import threading

    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    executor = PyExecutor.__new__(PyExecutor)
    executor.response_cv = threading.Condition()
    executor.shutdown_event = threading.Event()
    executor.num_micro_batches = 0
    executor._event_loop_completed = completed
    if events is not None:
        executor._executor_events = events
    return executor


class _RecordingEvents(_Events):
    def __init__(self) -> None:
        self.calls = []

    def at_shutdown(self) -> None:
        self.calls.append("at_shutdown")


@pytest.mark.parametrize("completed", [False, True])
def test_executor_loop_cleanup_without_events(completed):
    executor = _cleanup_executor(None, completed)

    executor._executor_loop_cleanup()
    assert executor.is_shutdown and executor.shutdown_event.is_set()


@pytest.mark.parametrize("completed", [False, True])
def test_executor_loop_cleanup_calls_at_shutdown_after_a_completed_loop(completed):
    events = _RecordingEvents()
    executor = _cleanup_executor(events, completed)

    executor._executor_loop_cleanup()
    assert executor.is_shutdown
    assert events.calls == (["at_shutdown"] if completed else [])


# -- the step-input stager ------------------------------------------------------------------


class _RecordingStager:
    def __init__(self) -> None:
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


def _block_offsets_manager(monkeypatch, log: list):
    """A ``KVCacheManagerV2`` with only what ``copy_batch_block_offsets`` reads (one page table)."""
    from tensorrt_llm._torch.pyexecutor.kv_cache import kv_cache_manager_v2
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

    monkeypatch.setattr(
        kv_cache_manager_v2,
        "copy_batch_block_offsets_to_device",
        lambda *args: log.append(("native", args)),
    )
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager._publish_sparse_metadata = lambda: None
    manager.max_copy_beam_width = 1
    manager.kv_cache_type = None
    manager._use_per_layer_page_tables = False
    manager.index_mapper = types.SimpleNamespace(
        get_copy_index=lambda request_ids, num_contexts, beam_width, cross: torch.tensor([1, 0])
    )
    manager.host_kv_cache_block_offsets = torch.zeros(2, 4, dtype=torch.int32)
    manager.index_scales = torch.ones(1, dtype=torch.int32)
    manager._index_scale_ints = [1]
    manager.kv_offset = torch.zeros(1, dtype=torch.int32)
    manager._stream = types.SimpleNamespace(cuda_stream=7)
    return manager


def test_kv_cache_block_offsets_copy_natively_without_a_stager(monkeypatch):
    log = []
    manager = _block_offsets_manager(monkeypatch, log)
    dst = torch.empty(2, 4, dtype=torch.int32)

    manager.copy_batch_block_offsets(dst, [3, 4], 1, 0, 2)
    assert len(log) == 1
    kind, args = log[0]
    assert kind == "native"
    assert args[0] is manager.host_kv_cache_block_offsets and args[1] is dst
    assert args[2].tolist() == [1, 0]
    assert args[3] is manager.index_scales and args[4] is manager.kv_offset and args[5] == 7


def test_kv_cache_block_offsets_are_staged_with_a_stager(monkeypatch):
    log = []
    manager = _block_offsets_manager(monkeypatch, log)
    manager.step_input_stager = _RecordingStager()
    dst = torch.empty(2, 4, dtype=torch.int32)

    manager.copy_batch_block_offsets(dst, [3, 4], 1, 0, 2)
    assert log == []
    ((name, args, _),) = manager.step_input_stager.calls
    assert name == "block_offsets"
    assert args[0] is manager.host_kv_cache_block_offsets and args[1] is dst
    assert args[2:] == ([1, 0], [1], [0])


def _slot_mapping_inputs(tokens: int):
    metadata = types.SimpleNamespace(
        host_indexer_k_cache_block_offsets=torch.tensor([[0, 1]], dtype=torch.int32),
        host_slot_mapping_fp8=torch.zeros(tokens, dtype=torch.int64),
        host_slot_mapping_scale=torch.zeros(tokens, dtype=torch.int64),
        slot_mapping_fp8=torch.full((tokens,), -1, dtype=torch.int64),
        slot_mapping_scale=torch.full((tokens,), -1, dtype=torch.int64),
    )
    params = types.SimpleNamespace(
        batch_size=1,
        tokens_per_block=4,
        head_dim=8,
        new_kv_tokens=torch.tensor([tokens]),
        data_bytes_per_token=8,
        cached_kv_tokens=torch.tensor([1]),
        quant_block_size=8,
    )
    return metadata, params


def test_indexer_slot_mappings_copy_natively_without_a_stager():
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer

    metadata, params = _slot_mapping_inputs(3)

    Indexer.recompute_slot_mappings(metadata, params)
    assert torch.equal(metadata.slot_mapping_fp8, metadata.host_slot_mapping_fp8)
    assert torch.equal(metadata.slot_mapping_scale, metadata.host_slot_mapping_scale)


def test_indexer_slot_mappings_are_staged_with_a_stager():
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer

    metadata, params = _slot_mapping_inputs(3)
    metadata.step_input_stager = _RecordingStager()

    Indexer.recompute_slot_mappings(metadata, params)
    assert [name for name, _, _ in metadata.step_input_stager.calls] == ["copy", "copy"]
    assert bool((metadata.slot_mapping_fp8 == -1).all())
    assert bool((metadata.slot_mapping_scale == -1).all())


# -- the step-input window in the decoder runner -------------------------------------------------


class _ReferenceStager:
    """A step-input stager that accepts every step it is offered and logs the window's lifecycle."""

    def __init__(self, log: list, accept: bool = True, tokens: int = 3) -> None:
        self.log, self.accept, self.tokens = log, accept, tokens

    def begin_if_eligible(
        self,
        preparer,
        requests,
        attn_metadata,
        spec_metadata,
        new_tensors_device,
        *,
        graph_replay,
        static_inputs,
        is_dummy,
        enable_spec_decode,
        promoted,
    ):
        self.log.append(("begin", graph_replay, static_inputs))
        return self if self.accept else None

    def end(self, record_event: bool = True) -> None:
        self.log.append(("end", record_event))

    def discard(self) -> None:
        self.log.append("discard")

    def set_completion_event(self, event) -> None:
        self.log.append(("completion_event", event))

    def static_inputs_tokens(self) -> int:
        return self.tokens


class _WindowGraphs:
    """The ``DecoderCudaGraphs`` calls ``_forward_decoder`` makes, with one selected graph key.

    ``run`` follows ``DecoderCudaGraphs._run``: ``before_replay`` runs only right before the
    replay of an already captured graph, and the replay consumes the staged static inputs.
    """

    def __init__(self, log: list, key, *, needs_capture=False, warmup_only=False, fail=None):
        self.log, self.key, self.fail = log, key, fail
        self.cuda_graph_runner = types.SimpleNamespace(
            needs_capture=lambda key: needs_capture,
            is_warmup_only=warmup_only,
            shared_static_tensors={"input_ids": torch.zeros(4, dtype=torch.int64)},
            staged_static_input_tokens=0,
        )

    def attention_dp_execution_batch(self, requests):
        return requests

    def promote_final_contexts(self, requests, kv_cache_manager, **kwargs):
        return requests, frozenset()

    def pad_batch(self, requests, resource_manager, runtime_draft_len):
        return contextlib.nullcontext(requests)

    def select(self, requests, resource_manager, **kwargs):
        return types.SimpleNamespace(
            key=self.key,
            use_lora_graph=False,
            attn_metadata=kwargs["attn_metadata"],
            spec_metadata=None,
        )

    def captures_model_body(self, key) -> bool:
        return False

    def run(self, key, inputs, run_model, postprocess_inputs, *, before_replay=None, **kwargs):
        runner = self.cuda_graph_runner
        if self.fail == "before_replay":
            raise RuntimeError("graph run failed")
        if key is None:
            self.log.append("eager")
        elif runner.needs_capture(key) or runner.is_warmup_only:
            self.log.append("capture")
        else:
            if before_replay is not None:
                before_replay()
            self.log.append(("replay", runner.staged_static_input_tokens))
            runner.staged_static_input_tokens = 0
            if self.fail == "replay":
                raise RuntimeError("replay failed")
        return "outputs"


class _WindowPreparer:
    """The ``DecoderInputPreparer`` calls ``_forward_decoder`` makes."""

    def __init__(self, log: list, fail: bool = False) -> None:
        self.log, self.fail = log, fail
        self.input_copy_event = "input-copy-event"

    def prepare_inputs(self, *args, **kwargs):
        self.log.append("prepare")
        if self.fail:
            raise RuntimeError("preparation failed")
        return types.SimpleNamespace(kwargs={}, runtime_draft_len=0, gather_ids=None)

    def record_input_copy(self) -> None:
        self.log.append("record_input_copy")

    def postprocess_inputs(self, *args, **kwargs) -> None:
        pass


def _window_runner(monkeypatch, graphs, preparer, stager):
    """A ``DecoderRunner`` with only what ``_forward_decoder`` reads (no speculation, no CED)."""
    from tensorrt_llm._torch.pyexecutor import ced_replay
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as runner_module
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

    monkeypatch.setattr(runner_module, "get_draft_kv_cache_manager", lambda *args: None)
    monkeypatch.setattr(ced_replay, "uses_private_decoder_cache", lambda *args: False)
    attn_metadata = types.SimpleNamespace()
    runner = DecoderRunner.__new__(DecoderRunner)
    runner._config = types.SimpleNamespace(spec_config=None, use_mrope=False, is_spec_decode=False)
    runner._metadata = types.SimpleNamespace(
        setup_attention=lambda *args: attn_metadata,
        attn_metadata=attn_metadata,
        spec_metadata=None,
    )
    runner._graphs = graphs
    runner._inputs = preparer
    runner._stage_in_graph_sampling = None
    runner._cuda_graph_metadata_prepare_scope = lambda *args, **kwargs: contextlib.nullcontext()
    runner._execute_logit_post_processors = lambda *args: None
    runner.forward_pass_callable = None
    runner.moe_load_balancer = None
    runner.model = types.SimpleNamespace()
    runner.kv_cache_manager_key = "kv_cache_manager"
    runner.runtime_hooks = RuntimeHooks(step_input_stager=stager)
    return runner


def _decode_step(runner):
    forward_inputs = types.SimpleNamespace(
        batch=types.SimpleNamespace(num_context_requests=0),
        new_tensors_device=None,
        cache_indirection_buffer=None,
        gather_context_logits=False,
        enable_spec_decode=False,
        runtime_draft_len=0,
    )
    resource_manager = types.SimpleNamespace(get_resource_manager=lambda key: object())
    return runner._forward_decoder(forward_inputs, resource_manager, is_dummy=False)


def _names(log: list) -> list:
    return [entry if isinstance(entry, str) else entry[0] for entry in log]


_GRAPH_KEY = (1, 0, False)
_STAGED_STEP = ["begin", "prepare", "end", "record_input_copy", "completion_event", "replay"]


def test_a_replay_issues_the_staged_preparation_before_the_replay(monkeypatch):
    log = []
    graphs = _WindowGraphs(log, _GRAPH_KEY)
    runner = _window_runner(monkeypatch, graphs, _WindowPreparer(log), _ReferenceStager(log))

    assert _decode_step(runner) == "outputs"
    assert log == [
        ("begin", True, graphs.cuda_graph_runner.shared_static_tensors),
        "prepare",
        ("end", False),
        "record_input_copy",
        ("completion_event", "input-copy-event"),
        ("replay", 3),
    ]


@pytest.mark.parametrize(
    "key, needs_capture, warmup_only, ran",
    [
        (None, False, False, "eager"),
        (_GRAPH_KEY, True, False, "capture"),
        (_GRAPH_KEY, False, True, "capture"),
    ],
    ids=["eager", "initial_capture", "warmup_only"],
)
def test_a_step_that_is_not_a_replay_never_opens_a_window(
    monkeypatch, key, needs_capture, warmup_only, ran
):
    log = []
    graphs = _WindowGraphs(log, key, needs_capture=needs_capture, warmup_only=warmup_only)
    runner = _window_runner(monkeypatch, graphs, _WindowPreparer(log), _ReferenceStager(log))

    _decode_step(runner)
    assert log == ["prepare", "record_input_copy", ran]
    assert graphs.cuda_graph_runner.staged_static_input_tokens == 0


def test_a_declined_replay_prepares_natively(monkeypatch):
    log = []
    graphs = _WindowGraphs(log, _GRAPH_KEY)
    stager = _ReferenceStager(log, accept=False)
    runner = _window_runner(monkeypatch, graphs, _WindowPreparer(log), stager)

    _decode_step(runner)
    assert _names(log) == ["begin", "prepare", "record_input_copy", "replay"]
    assert log[-1] == ("replay", 0)


def test_a_failed_preparation_discards_the_window(monkeypatch):
    log = []
    graphs = _WindowGraphs(log, _GRAPH_KEY)
    runner = _window_runner(
        monkeypatch, graphs, _WindowPreparer(log, fail=True), _ReferenceStager(log)
    )

    with pytest.raises(RuntimeError, match="preparation failed"):
        _decode_step(runner)
    assert _names(log) == ["begin", "prepare", "discard"]


@pytest.mark.parametrize(
    "fail, expected",
    [
        ("before_replay", ["begin", "prepare", "discard"]),
        ("replay", _STAGED_STEP),
    ],
    ids=["window_open", "window_closed"],
)
def test_a_failed_run_discards_only_a_window_still_open(monkeypatch, fail, expected):
    log = []
    graphs = _WindowGraphs(log, _GRAPH_KEY, fail=fail)
    runner = _window_runner(monkeypatch, graphs, _WindowPreparer(log), _ReferenceStager(log))

    with pytest.raises(RuntimeError):
        _decode_step(runner)
    assert _names(log) == expected


def test_replay_eager_replay_opens_one_window_per_replay(monkeypatch):
    log = []
    graphs = _WindowGraphs(log, _GRAPH_KEY)
    runner = _window_runner(monkeypatch, graphs, _WindowPreparer(log), _ReferenceStager(log))

    _decode_step(runner)
    graphs.key = None
    _decode_step(runner)
    graphs.key = _GRAPH_KEY
    _decode_step(runner)
    assert _names(log) == [
        *_STAGED_STEP,
        "prepare",
        "record_input_copy",
        "eager",
        *_STAGED_STEP,
    ]
    assert [entry for entry in log if entry[0] == "replay"] == [("replay", 3), ("replay", 3)]


def _replay_graphs(monkeypatch, log: list, *, needs_capture: bool, warmup_only: bool):
    """A ``DecoderCudaGraphs`` with only what ``_run`` reads, logging capture and replay."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import graphs as graphs_module
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs

    monkeypatch.setattr(graphs_module, "prepare_attn_metadata_for_draft_replay", lambda *args: None)
    monkeypatch.setattr(
        graphs_module, "restore_attn_metadata_after_draft_replay", lambda *args: None
    )
    monkeypatch.setattr(
        graphs_module, "MoeLoadBalancerIterContext", lambda balancer: contextlib.nullcontext()
    )
    graphs = DecoderCudaGraphs.__new__(DecoderCudaGraphs)
    graphs.breakable_cuda_graph_runner = None
    graphs._torch_compile_bypass = contextlib.nullcontext
    graphs.cuda_graph_runner = types.SimpleNamespace(
        needs_capture=lambda key: needs_capture,
        is_warmup_only=warmup_only,
        capture=lambda key, fn, inputs, **kwargs: log.append("capture"),
        replay=lambda key, inputs, **kwargs: log.append("replay"),
    )
    return graphs


@pytest.mark.parametrize(
    "key, needs_capture, warmup_only, expected",
    [
        (_GRAPH_KEY, False, False, ["before_replay", "replay"]),
        (_GRAPH_KEY, True, False, ["capture", "replay"]),
        (_GRAPH_KEY, True, True, ["capture"]),
        (None, False, False, ["eager"]),
    ],
    ids=["replay", "capture_then_replay", "warmup_only", "eager"],
)
def test_before_replay_runs_only_before_the_replay_of_a_captured_graph(
    monkeypatch, key, needs_capture, warmup_only, expected
):
    log = []
    graphs = _replay_graphs(monkeypatch, log, needs_capture=needs_capture, warmup_only=warmup_only)

    graphs._run(
        key,
        {"input_ids": torch.zeros(2, dtype=torch.int64)},
        lambda inputs: log.append("eager"),
        lambda inputs: None,
        attn_metadata=None,
        draft_kv_cache_manager=None,
        moe_load_balancer=None,
        enable_spec_decode=True,
        gather_context_logits=False,
        position_id_offsets=None,
        before_replay=lambda: log.append("before_replay"),
    )
    assert log == expected


# -- the speculative sampler's store writer -----------------------------------------------------


class _ReferenceStoreWriter:
    """Stores like the native ``index_copy_`` path, then copies the store to the host tensors."""

    def __init__(self, supported: bool = True) -> None:
        self._supported = supported
        self.calls = []

    def supported(self, store, new_tokens, next_new_tokens, new_tokens_lens, next_draft_tokens):
        self.calls.append("supported")
        return self._supported

    def store_and_copy_out(
        self,
        store,
        slots,
        new_tokens,
        next_new_tokens,
        new_tokens_lens,
        next_draft_tokens,
        host_new_tokens,
        host_new_tokens_lens,
        host_next_draft_tokens,
        record_event=True,
    ) -> None:
        self.calls.append(("store_and_copy_out", list(slots), record_event))
        index = torch.tensor(slots, dtype=torch.long)
        store.new_tokens.squeeze(-1).T.index_copy_(0, index, new_tokens)
        store.next_new_tokens.squeeze(-1).T.index_copy_(0, index, next_new_tokens)
        store.new_tokens_lens.index_copy_(0, index, new_tokens_lens)
        store.next_draft_tokens.index_copy_(0, index, next_draft_tokens)
        host_new_tokens.copy_(store.new_tokens)
        host_new_tokens_lens.copy_(store.new_tokens_lens)
        host_next_draft_tokens.copy_(store.next_draft_tokens)

    def set_completion_event(self, event) -> None:
        self.calls.append(("completion_event", event))


def _cpu_sampling(monkeypatch, *, pinned: bool) -> None:
    """Run ``SpecSampler.sample_async`` on CPU: device transfers stay on the host."""
    from tensorrt_llm._torch.speculative import spec_sampler_base

    original_to = torch.Tensor.to
    original_empty_like = torch.empty_like

    def to_cpu(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs = {**kwargs, "device": "cpu"}
        return original_to(tensor, *args, **kwargs)

    def empty_like(tensor, *args, pin_memory=False, **kwargs):
        # Pinning needs a CUDA context; the writer's host buffers are plain host tensors here.
        return original_empty_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to_cpu)
    monkeypatch.setattr(torch, "empty_like", empty_like)
    monkeypatch.setattr(spec_sampler_base, "prefer_pinned", lambda: pinned)
    monkeypatch.setattr(spec_sampler_base, "handle_stop_criteria", lambda *args, **kwargs: False)


def _store_sampler(writer, *, async_worker: bool = False):
    """A ``SpecSampler`` with a six-slot store, draft length 3 and the given store writer."""
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    sampler = SpecSampler.__new__(SpecSampler)
    sampler.draft_len = 3
    sampler.max_accepted_path_len = 4
    sampler.max_seq_len = 128
    sampler.store = types.SimpleNamespace(
        new_tokens=torch.zeros(4, 6, 1, dtype=torch.int32),
        next_new_tokens=torch.zeros(4, 6, 1, dtype=torch.int32),
        next_draft_tokens=torch.zeros(6, 3, dtype=torch.int32),
        new_tokens_lens=torch.zeros(6, dtype=torch.int32),
    )
    sampler.store_writer = writer
    sampler._async_worker_active = lambda: async_worker
    sampler._copy_to_host = lambda tensor: tensor.clone()
    sampler._record_sampler_event = lambda: types.SimpleNamespace(cuda_event="sampler-event")
    return sampler


def _sample_generation_step(sampler, *, verify_lens: bool = False):
    """Three generation requests in nonconsecutive slots 4, 1 and 3."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

    def request(request_id: int, slot: int):
        return types.SimpleNamespace(
            py_request_id=request_id,
            py_seq_slot=slot,
            state=LlmRequestState.GENERATION_IN_PROGRESS,
            py_draft_tokens=[10, 11, 12],
            py_draft_tokens_effective_len=None,
            py_verify_len=None,
            py_decoding_iter=0,
        )

    scheduled = types.SimpleNamespace(
        context_requests_chunking=[],
        context_requests_last_chunk=[],
        generation_requests=[request(7, 4), request(8, 1), request(9, 3)],
    )
    outputs = dict(
        new_tokens=torch.arange(12, dtype=torch.int32).reshape(3, 4) + 100,
        new_tokens_lens=torch.tensor([1, 3, 2], dtype=torch.int32),
        next_draft_tokens=torch.arange(9, dtype=torch.int32).reshape(3, 3) + 200,
        next_new_tokens=torch.arange(12, dtype=torch.int32).reshape(3, 4) + 300,
    )
    if verify_lens:
        outputs["verify_lens"] = torch.tensor([2, 3, 1], dtype=torch.int32)
        outputs["verify_lens_in_output_order"] = True
    return sampler.sample_async(scheduled, outputs, [])


def _assert_same_stores(sampler, state, native, native_state) -> None:
    for name in ("new_tokens", "next_new_tokens", "next_draft_tokens", "new_tokens_lens"):
        assert torch.equal(getattr(sampler.store, name), getattr(native.store, name)), name
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(state.host, name), getattr(native_state.host, name)), name


def test_the_store_writer_matches_the_native_stores_on_nonconsecutive_slots(monkeypatch):
    _cpu_sampling(monkeypatch, pinned=True)
    native = _store_sampler(None)
    writer = _ReferenceStoreWriter()
    sampler = _store_sampler(writer)

    native_state = _sample_generation_step(native)
    state = _sample_generation_step(sampler)
    assert writer.calls == [
        "supported",
        ("store_and_copy_out", [4, 1, 3], False),
        ("completion_event", "sampler-event"),
    ]
    assert native.store.new_tokens_lens.tolist() == [0, 3, 0, 2, 1, 0]
    _assert_same_stores(sampler, state, native, native_state)
    assert state.host.verify_lens is None
    assert state.device.new_tokens is sampler.store.next_new_tokens


@pytest.mark.parametrize("case", ["declined", "verify_lens", "async_worker", "not_pinned"])
def test_the_store_writer_falls_back_to_the_native_stores(monkeypatch, case):
    _cpu_sampling(monkeypatch, pinned=case != "not_pinned")
    native = _store_sampler(None)
    writer = _ReferenceStoreWriter(supported=case != "declined")
    sampler = _store_sampler(writer, async_worker=case == "async_worker")

    native_state = _sample_generation_step(native, verify_lens=case == "verify_lens")
    state = _sample_generation_step(sampler, verify_lens=case == "verify_lens")
    assert writer.calls == (["supported"] if case == "declined" else [])
    _assert_same_stores(sampler, state, native, native_state)
    if case == "verify_lens":
        assert state.host.verify_lens.tolist() == [2, 3, 1]
