# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the engine reads the step-loop hooks a model declares (``runtime_hooks_of``).

The engine's call sites run their original statements when a hook is off: every
object that holds a hook declares it off at class level, every call a call site
makes on a hook is a method of the hook's protocol, every read of a hook in the
engine files is decided by its all-off value toward the original statements
(checked on the source), and call sites of each hook run on stand-ins with the
hook off and on.
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
    MsaRuntimePolicy,
    RuntimeHooks,
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
    assert hooks.decode_prologue is None
    assert hooks.executor_events is None


def test_every_field_defaults_to_off():
    for field in dataclasses.fields(RuntimeHooks):
        value = getattr(DEFAULT_RUNTIME_HOOKS, field.name)
        if dataclasses.is_dataclass(value):
            assert all(
                getattr(value, item.name) in (False, None) for item in dataclasses.fields(value)
            )
        else:
            assert value in (False, None), field.name


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


def test_engine_classes_declare_the_hooks_off():
    """Every object that holds a hook declares it off at class level."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    assert DecoderInputPreparer.decode_prologue is None
    assert DecoderRunner.runtime_hooks is DEFAULT_RUNTIME_HOOKS
    assert PyExecutor._execution_stream_loop_armed is False
    assert PyExecutor._loop_on_execution_stream is False
    assert PyExecutor._executor_events is None


# Per hook protocol: the local names and attribute names a call site holds the hook in, and
# the engine files with call sites.
_HOOK_CONSUMERS = {
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
    "pyexecutor/engine/runners/decoder/inputs.py",
    "pyexecutor/py_executor.py",
)

# Attributes that hold a hook, and their value when every hook is off.
_SEAM_ATTRS = {
    "_executor_events": None,
    "_loop_on_execution_stream": False,
    "_execution_stream_loop_armed": False,
    "runtime_hooks": DEFAULT_RUNTIME_HOOKS,
    "decode_prologue": None,
}
# Locals the call sites hold a hook (or a decision on one) in.
_SEAM_LOCALS = ("prologue",)
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
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "runtime_hooks_of"
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
        if isinstance(node, ast.IfExp):
            condition = self.value(node.test, env)
            if condition is not _UNKNOWN:
                return self.value(node.body if condition else node.orelse, env)
            body, other = self.value(node.body, env), self.value(node.orelse, env)
            return body if body is not _UNKNOWN and body is other else _UNKNOWN
        if self._is_seam_call(node):
            return DEFAULT_RUNTIME_HOOKS
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
    events = self._executor_events
    if events is not None:
        events.after_warmup()
    else:
        x.copy_()
    events.at_shutdown()
    if x or events:
        pass
    self._executor_events = x
"""
    walk = _AllOffWalk("sample.py")
    walk.block(ast.parse(source).body, {})
    assert walk.decided == 1
    assert walk.problems == [
        "sample.py:8: a hook is used outside a test decided with every hook off",
        "sample.py:9: a test of a hook is not decided with every hook off",
        "sample.py:11: a hook value with every hook off is not decided",
    ]


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("self.runtime_hooks.spec_step_fold if enabled else None", None),
        ("self.runtime_hooks.spec_step_fold if enabled else object()", _UNKNOWN),
    ],
)
def test_conditional_hook_binding_requires_both_branches_to_agree(expression, expected):
    walk = _AllOffWalk("sample.py")
    assert walk.value(ast.parse(expression, mode="eval").body, {}) is expected


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

    def wait_event(self, event: "_Event") -> None:
        self.log.append(f"{self.name} waits for {event.stream.name}")


class _Event:
    """A reused ordering event: a wait sees the stream of its latest record."""

    stream: _Stream | None = None

    def record(self, stream: _Stream) -> None:
        self.stream = stream


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
    executor.enable_spec_decode = False
    executor.runtime_draft_len = 0
    executor.execution_stream = _Stream("execution", log)
    executor.resource_manager = object()
    executor.sampler = types.SimpleNamespace(get_cache_indirection=lambda: None)
    executor.model_engine = types.SimpleNamespace(
        forward=lambda *args, **kwargs: log.append("forward") or "outputs",
        route_capture=None,
    )
    executor._iter_adp_dummy_ctx_tokens = 0
    executor._iter_adp_dummy_gen_tokens = 0
    executor._compute_adp_dummy_tokens = lambda requests: (0, 0)
    executor._maybe_record_hang_diagnostic_phase = lambda *args, **kwargs: None
    executor._attach_encoder_output_to_execution_stream = lambda requests: None
    executor._mark_cross_kv_projection_consumed = lambda requests: None
    executor._kv_connector_wait_for_save = lambda: None
    executor._stream_ordering_events = (_Event(), _Event())
    executor._finish_inforward_snapshots = False
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


class _M3PlainModel(torch.nn.Module):
    """A model that declares no hooks, like every model outside the opt-in."""


class _M3ClassHooksModel(torch.nn.Module):
    runtime_hooks = RuntimeHooks(pack_step_inputs=True, skip_unchanged_h2d=True)


def test_m3_model_without_hooks_gets_the_all_off_default():
    hooks = runtime_hooks_of(_M3PlainModel())

    assert hooks is DEFAULT_RUNTIME_HOOKS
    assert hooks == RuntimeHooks()
    assert not hooks.pack_step_inputs
    assert not hooks.skip_unchanged_h2d
    assert not hooks.loop_on_execution_stream
    assert not hooks.merge_tuned_tactics_across_ranks
    assert not hooks.skip_greedy_rng_upload
    assert hooks.spec_step_fold is None
    assert hooks.msa == MsaRuntimePolicy()
    assert not hooks.msa.short_sequence_identity
    assert hooks.msa.kv_update is None
    assert not hooks.msa.skip_unchanged_block_offsets
    assert not hooks.msa.reuse_pool_views


def test_m3_hooks_declared_as_none_mean_the_default():
    model = _M3PlainModel()
    model.runtime_hooks = None

    assert runtime_hooks_of(model) is DEFAULT_RUNTIME_HOOKS


def test_m3_class_attribute_hooks_are_returned_as_declared():
    model = _M3ClassHooksModel()

    assert runtime_hooks_of(model) is _M3ClassHooksModel.runtime_hooks


def test_m3_instance_attribute_hooks_are_returned_as_declared():
    hooks = RuntimeHooks(
        loop_on_execution_stream=True,
        msa=MsaRuntimePolicy(short_sequence_identity=True, reuse_pool_views=True),
    )
    model = _M3PlainModel()
    model.runtime_hooks = hooks

    assert runtime_hooks_of(model) is hooks


def test_m3_instance_attribute_overrides_class_attribute():
    hooks = RuntimeHooks(skip_greedy_rng_upload=True)
    model = _M3ClassHooksModel()
    model.runtime_hooks = hooks

    assert runtime_hooks_of(model) is hooks


@pytest.mark.parametrize(
    "declared",
    [True, {"pack_step_inputs": True}, MsaRuntimePolicy(), "RuntimeHooks()"],
    ids=["bool", "dict", "msa_policy", "str"],
)
def test_m3_hooks_of_the_wrong_type_raise(declared):
    model = _M3PlainModel()
    model.runtime_hooks = declared

    with pytest.raises(TypeError, match=r"_M3PlainModel\.runtime_hooks must be a RuntimeHooks"):
        runtime_hooks_of(model)


def test_m3_hooks_are_immutable_once_declared():
    hooks = RuntimeHooks()

    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.pack_step_inputs = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        hooks.msa.reuse_pool_views = True


@pytest.mark.parametrize(
    "hooks, skips",
    [
        (RuntimeHooks(), False),
        (RuntimeHooks(pack_step_inputs=True, skip_greedy_rng_upload=True), False),
        (RuntimeHooks(skip_unchanged_h2d=True), True),
        (RuntimeHooks(msa=MsaRuntimePolicy(skip_unchanged_block_offsets=True)), True),
    ],
    ids=["default", "no_skip_of_metadata", "h2d_skip", "msa_block_offset_skip"],
)
def test_m3_every_metadata_upload_skip_asks_for_private_metadata_buffers(hooks, skips):
    assert hooks.skips_unchanged_metadata_uploads is skips
