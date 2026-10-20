# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the engine reads the step-loop hooks a model declares (``runtime_hooks_of``).

With every hook off (``DEFAULT_RUNTIME_HOOKS``), each engine call site runs its
original statement and nothing else. The call sites that guard one host-to-device
write are run as written in the engine, from their ``stager =
self.step_input_stager`` binding, on stand-in values: with the hook off they make
exactly the calls of their original statement; with a stager that records the
write they hand it the very buffers the original statement writes and skip that
statement; with a stager that declines it they hand it over and then run the
original statement. The overlap-scheduler branch, the prepare window, the CUDA
graph replay's static inputs, the speculative sampler's store and the sampler
binding are checked the same way through their own code.
"""

import ast
import builtins
import dataclasses
import inspect
import textwrap
import types
from pathlib import Path
from unittest import mock

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    RuntimeHooks,
    SamplerStoreWriter,
    StepInputStager,
    StepInputStagerSource,
    runtime_hooks_of,
)

pytestmark = pytest.mark.cpu_only


class _PlainModel(torch.nn.Module):
    """A model that declares no hooks, like every model outside the opt-in."""


class _ClassHooksModel(torch.nn.Module):
    runtime_hooks = RuntimeHooks(step_input_stager=object())


# -- RuntimeHooks and runtime_hooks_of --------------------------------------------------------


@pytest.mark.parametrize("model", [_PlainModel(), object()], ids=["module", "object"])
def test_a_model_without_hooks_gets_the_all_off_default(model):
    hooks = runtime_hooks_of(model)

    assert hooks is DEFAULT_RUNTIME_HOOKS
    assert hooks == RuntimeHooks()
    assert hooks.step_input_stager is None
    assert hooks.sampler_store_writer is None


def test_every_field_defaults_to_off():
    for field in dataclasses.fields(RuntimeHooks):
        assert getattr(DEFAULT_RUNTIME_HOOKS, field.name) is None, field.name


def test_hooks_declared_as_none_mean_the_default():
    model = _PlainModel()
    model.runtime_hooks = None

    assert runtime_hooks_of(model) is DEFAULT_RUNTIME_HOOKS


def test_class_attribute_hooks_are_returned_as_declared():
    assert runtime_hooks_of(_ClassHooksModel()) is _ClassHooksModel.runtime_hooks


def test_property_hooks_are_returned_as_declared():
    hooks = RuntimeHooks(sampler_store_writer=object())

    class _PropertyModel(torch.nn.Module):
        @property
        def runtime_hooks(self):
            return hooks

    assert runtime_hooks_of(_PropertyModel()) is hooks


def test_instance_attribute_overrides_class_attribute():
    hooks = RuntimeHooks()
    model = _ClassHooksModel()
    model.runtime_hooks = hooks

    assert runtime_hooks_of(model) is hooks


@pytest.mark.parametrize(
    "declared", [True, {"step_input_stager": None}, "RuntimeHooks()"], ids=["bool", "dict", "str"]
)
def test_hooks_of_the_wrong_type_raise(declared):
    model = _PlainModel()
    model.runtime_hooks = declared

    with pytest.raises(TypeError, match=r"_PlainModel\.runtime_hooks must be a RuntimeHooks"):
        runtime_hooks_of(model)


def test_hooks_are_immutable_once_declared():
    with pytest.raises(dataclasses.FrozenInstanceError):
        RuntimeHooks().step_input_stager = object()


def test_the_protocols_state_the_call_site_methods():
    def methods(protocol):
        return {
            name
            for name, value in vars(protocol).items()
            if inspect.isfunction(value) and not name.startswith("_")
        }

    # What RuntimeHooks.step_input_stager declares: it opens one window per step.
    assert methods(StepInputStagerSource) == {"begin_if_eligible"}
    # What a window sets as ``step_input_stager`` and the call sites call.
    assert methods(StepInputStager) == {
        "copy",
        "overlap_inputs",
        "block_offsets",
        "end",
        "discard",
        "static_inputs",
    }
    assert methods(SamplerStoreWriter) == {"store_and_copy_out"}


def test_engine_classes_declare_the_hooks_off():
    """Every object a hook attaches to declares it off at class level."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm._torch.speculative.interface import SpecMetadata
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    for cls in (TrtllmAttentionMetadata, SpecMetadata, DFlashSpecMetadata):
        assert cls.step_input_stager is None, cls
        # A class attribute, not a dataclass field: the constructors do not take it.
        assert "step_input_stager" not in {f.name for f in dataclasses.fields(cls)}, cls
    assert DecoderRunner.step_input_stager is None
    assert DecoderRunner.runtime_hooks is DEFAULT_RUNTIME_HOOKS
    assert CUDAGraphRunner.staged_static_inputs is None
    assert SpecSampler.store_writer is None


def test_no_zoo_model_declares_runtime_hooks():
    """Read off disk: no model of the zoo names the hooks, so each gets the all-off default."""
    models = Path(tensorrt_llm.__file__).resolve().parent / "_torch" / "models"
    offenders = [
        str(path.relative_to(models))
        for path in sorted(models.rglob("*.py"))
        if "runtime_hooks" in path.read_text(errors="ignore")
    ]
    assert not offenders, offenders


def test_zoo_model_classes_read_as_the_all_off_hooks():
    """Every model class of the zoo that imports here, without its constructor, reads as the
    all-off hooks."""
    import importlib

    from tensorrt_llm._torch.models._arch_index import MODEL_CLASS_TO_MODULE

    checked = 0
    for name, module in sorted(MODEL_CLASS_TO_MODULE.items()):
        try:
            providing = importlib.import_module(f"tensorrt_llm._torch.models.{module}")
        except ImportError as err:
            if (getattr(err, "name", None) or "").split(".")[0] == "tensorrt_llm":
                raise
            continue
        cls = getattr(providing, name)
        assert runtime_hooks_of(cls.__new__(cls)) is DEFAULT_RUNTIME_HOOKS, name
        checked += 1
    assert checked


# -- call sites that guard one write, run as written ------------------------------------------
#
# Each site runs from its ``stager = self.step_input_stager`` binding: the test sets the
# attribute on ``self``, as a stager's window does, and runs the binding and the guarded
# statement. A site that read another object, or bound something else, would leave the
# stager unasked.


class _Stager:
    """Records the hook calls a call site makes and answers them with ``accept``."""

    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.accept

        return call


class _Sym:
    """A stand-in value named by the expression that produced it.

    Attribute, item and call results are ``_Sym`` objects memoized by expression, so
    evaluating one expression twice gives the same object and two different expressions
    give different objects. Every call is logged as ``(expression, args, kwargs)``.
    """

    def __init__(self, expr: str, log: list) -> None:
        object.__setattr__(self, "_expr", expr)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_derived", {})

    def _derive(self, key, expr):
        if key not in self._derived:
            self._derived[key] = _Sym(expr, self._log)
        return self._derived[key]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._derive(("attr", name), f"{self._expr}.{name}")

    def __setattr__(self, name, value):
        self._derived[("attr", name)] = value

    def __getitem__(self, key):
        text = _text(key)
        return self._derive(("item", text), f"{self._expr}[{text}]")

    def __call__(self, *args, **kwargs):
        self._log.append((self._expr, args, kwargs))
        text = ", ".join([_text(a) for a in args] + [f"{k}={_text(v)}" for k, v in kwargs.items()])
        return self._derive(("call", text), f"{self._expr}({text})")

    def __len__(self):
        return 2

    def __repr__(self):
        return self._expr


def _text(value):
    if isinstance(value, slice):
        return ":".join(
            "" if v is None else _text(v) for v in (value.start, value.stop, value.step)
        )
    if isinstance(value, tuple):
        return "(" + ", ".join(_text(v) for v in value) + ")"
    return repr(value)


def _texts(log):
    return [
        (expr, tuple(_text(a) for a in args), tuple((k, _text(v)) for k, v in kwargs.items()))
        for expr, args, kwargs in log
    ]


def _decline_test(test, method):
    """Whether ``test`` is ``stager is None or not stager.<method>(...)``."""
    if not (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or) and len(test.values) == 2):
        return False
    is_none, declined = test.values
    return (
        isinstance(is_none, ast.Compare)
        and isinstance(is_none.left, ast.Name)
        and is_none.left.id == "stager"
        and isinstance(is_none.ops[0], ast.Is)
        and isinstance(declined, ast.UnaryOp)
        and isinstance(declined.op, ast.Not)
        and isinstance(declined.operand, ast.Call)
        and isinstance(declined.operand.func, ast.Attribute)
        and isinstance(declined.operand.func.value, ast.Name)
        and declined.operand.func.value.id == "stager"
        and declined.operand.func.attr == method
    )


def _binds_stager(node):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        targets = [node.target]
    else:
        return False
    return any(isinstance(t, ast.Name) and t.id == "stager" for t in targets)


def _bound_sites(func, selects):
    """``(binding, statement)`` for every statement of ``func`` that ``selects`` picks.

    ``binding`` is the one statement of the statement's function scope (a nested function
    is a scope of its own) that binds ``stager``; it comes before the statement.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found = []

    def scope(function):
        bindings, statements, inner = [], [], []
        stack = list(function.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner.append(node)
                continue
            if _binds_stager(node):
                bindings.append(node)
            if selects(node):
                statements.append(node)
            stack.extend(ast.iter_child_nodes(node))
        if statements:
            assert len(bindings) == 1, (function.name, [ast.unparse(b) for b in bindings])
            (binding,) = bindings
            for statement in sorted(statements, key=lambda s: s.lineno):
                assert binding.lineno < statement.lineno, (function.name, statement.lineno)
                found.append((binding, statement))
        for nested in inner:
            scope(nested)

    scope(tree.body[0])
    return found


def _run(nodes, ns):
    exec(compile(ast.Module(body=list(nodes), type_ignores=[]), "<call site>", "exec"), ns)


def _eval(node, ns):
    return eval(compile(ast.Expression(body=node), "<call site>", "eval"), ns)


def _free_names(nodes):
    return {
        n.id
        for node in nodes
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and not hasattr(builtins, n.id)
    } - {"stager"}


def _sym_namespace(nodes, stager):
    """A ``_Sym`` for every free name of ``nodes``, with ``self.step_input_stager = stager``."""
    log = []
    ns = {name: _Sym(name, log) for name in _free_names(nodes)}
    ns.setdefault("self", _Sym("self", log)).step_input_stager = stager
    return ns, log


def _write_sites():
    """``(binding, site, method)`` of every call site that guards one write with a stager."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    groups = [
        ("runner", DecoderRunner._prepare_tp_inputs, "copy", 7),
        ("trtllm", TrtllmAttentionMetadata.prepare, "copy", 2),
        ("trtllm", TrtllmAttentionMetadata.prepare, "block_offsets", 1),
        ("dflash", DFlashSpecMetadata.prepare, "copy", 2),
        ("interface", SpecMetadata._populate_request_rng_state, "copy", 1),
    ]
    out = []
    for name, func, method, expected in groups:
        sites = _bound_sites(
            func, lambda node, m=method: isinstance(node, ast.If) and _decline_test(node.test, m)
        )
        assert len(sites) == expected, (name, method, len(sites))
        out += [
            pytest.param(binding, site, method, id=f"{name}-{method}-L{site.lineno}")
            for binding, site in sites
        ]
    return out


_WRITE_SITES = _write_sites()


def _original_call(site):
    """The guarded original statement and its call: ``dst.copy_(src, non_blocking=True)`` or
    ``manager.copy_batch_block_offsets(dst, ...)``."""
    (original,) = site.body
    assert isinstance(original, ast.Expr) and isinstance(original.value, ast.Call)
    return original, original.value


def _hook_operands(call, method, ns):
    """What the original call writes, as the stager must receive it, evaluated in ``ns``:
    ``(dst, src)`` of a copy, ``(manager, dst, ...)`` of the block-offset copy."""
    receiver = _eval(call.func.value, ns)
    operands = [_eval(arg, ns) for arg in call.args]
    if method == "copy":
        assert call.func.attr == "copy_" and len(operands) == 1
        assert [(k.arg, k.value.value) for k in call.keywords] == [("non_blocking", True)]
        return (receiver, operands[0])
    assert call.func.attr == "copy_batch_block_offsets"
    return (receiver, *operands)


@pytest.mark.parametrize("binding, site, method", _WRITE_SITES)
def test_a_write_site_runs_exactly_its_original_statement_with_the_hook_off(binding, site, method):
    original, _ = _original_call(site)
    expected_ns, expected = _sym_namespace([original], None)
    _run([original], expected_ns)
    assert expected, "the original statement makes calls"

    ns, log = _sym_namespace([binding, site], None)
    _run([binding, site], ns)
    assert _texts(log) == _texts(expected)


@pytest.mark.parametrize("binding, site, method", _WRITE_SITES)
def test_a_write_site_hands_its_write_to_an_accepting_stager(binding, site, method):
    _, call = _original_call(site)
    stager = _Stager(accept=True)
    ns, log = _sym_namespace([binding, site], stager)
    _run([binding, site], ns)

    ((name, args, kwargs),) = stager.calls
    assert name == method and not kwargs
    # The original call never ran ...
    assert not [expr for expr, _, _ in log if expr.endswith(f".{call.func.attr}")]
    # ... and the stager got the very buffers the original call writes.
    expected = _hook_operands(call, method, ns)
    assert len(args) == len(expected)
    assert all(a is e for a, e in zip(args, expected)), (args, expected)


@pytest.mark.parametrize("binding, site, method", _WRITE_SITES)
def test_a_write_site_runs_its_original_statement_after_a_stager_declines(binding, site, method):
    _, call = _original_call(site)
    stager = _Stager(accept=False)
    ns, log = _sym_namespace([binding, site], stager)
    _run([binding, site], ns)

    ((name, args, _),) = stager.calls
    assert name == method
    (own,) = [entry for entry in log if entry[0].endswith(f".{call.func.attr}")]
    expected = _hook_operands(call, method, ns)
    assert len(args) == len(expected)
    assert all(a is e for a, e in zip(args, expected)), (args, expected)
    # The original call ran once, on what the stager declined.
    expr, own_args, own_kwargs = own
    assert expr == f"{expected[0]!r}.{call.func.attr}"
    assert len(own_args) == len(expected) - 1
    assert all(a is e for a, e in zip(own_args, expected[1:]))
    assert set(own_kwargs) == {k.arg for k in call.keywords}


# -- the runner's overlap-scheduler branch ---------------------------------------------------


def _mock_namespace(nodes, **values):
    """A mock for every free name the statements read, except builtins and ``values``."""
    ns = {name: mock.MagicMock(name=name) for name in _free_names(nodes)}
    ns.update(values)
    return ns


def _overlap_chain():
    """The ``stager`` binding of ``_prepare_tp_inputs`` and its overlap-scheduler chain."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

    chains = _bound_sites(
        DecoderRunner._prepare_tp_inputs,
        lambda node: isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Attribute) and n.attr == "overlap_inputs" for n in ast.walk(node.test)
        ),
    )
    assert len(chains) == 1
    return chains[0]


def _run_overlap(stager, next_draft_tokens_device=mock.sentinel.drafts):
    binding, chain = _overlap_chain()
    ns = _mock_namespace(
        [binding, chain],
        next_draft_tokens_device=next_draft_tokens_device,
        previous_batch_len=0,
        draft_lens=[3],
    )
    ns["self"].step_input_stager = stager
    _run([binding, chain], ns)
    return ns, chain


def test_the_overlap_branch_runs_natively_with_the_hook_off():
    ns, _ = _run_overlap(None)
    self = ns["self"]
    # The native branch, once: the offsets are zeroed and the step width read once.
    assert self.get_runtime_tokens_per_gen_step.call_count == 1
    assert mock.call.previous_pos_id_offsets_cuda.__imul__(0) in self.mock_calls
    assert mock.call.previous_kv_lens_offsets_cuda.__imul__(0) in self.mock_calls


def test_the_overlap_branch_is_handed_to_an_accepting_stager():
    stager = _Stager(accept=True)
    ns, chain = _run_overlap(stager)
    self = ns["self"]
    ((name, args, kwargs),) = stager.calls
    assert name == "overlap_inputs" and args == (self,)
    assert set(kwargs) == {
        "new_tokens_device",
        "new_tokens_lens_device",
        "next_draft_tokens_device",
        "previous_batch_indices",
        "previous_pos_indices",
        "num_tokens",
        "num_draft_tokens",
        "tokens_per_gen_step",
        "num_extend_requests",
    }
    # Each argument is the prepare's own value of that name.
    (hook_call,) = [
        n
        for n in ast.walk(chain.test)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "overlap_inputs"
    ]
    for keyword in hook_call.keywords:
        if isinstance(keyword.value, ast.Name):
            assert keyword.arg == keyword.value.id
            assert kwargs[keyword.arg] is ns[keyword.value.id], keyword.arg
    assert kwargs["next_draft_tokens_device"] is mock.sentinel.drafts
    assert kwargs["tokens_per_gen_step"] is self.get_runtime_tokens_per_gen_step.return_value
    self.get_runtime_tokens_per_gen_step.assert_called_once_with(ns["runtime_draft_len"])
    # Only the stager's argument read the step width; the native branch did not run.
    assert mock.call.previous_pos_id_offsets_cuda.__imul__(0) not in self.mock_calls
    ns["previous_seq_slots_device"].assert_not_called()


def test_the_overlap_branch_runs_natively_after_a_stager_declines():
    stager = _Stager(accept=False)
    ns, _ = _run_overlap(stager)
    self = ns["self"]
    assert [c[0] for c in stager.calls] == ["overlap_inputs"]
    assert self.get_runtime_tokens_per_gen_step.call_count == 2
    assert mock.call.previous_pos_id_offsets_cuda.__imul__(0) in self.mock_calls


def test_without_previous_drafts_the_overlap_branch_never_asks_the_stager():
    stager = _Stager(accept=True)
    ns, _ = _run_overlap(stager, next_draft_tokens_device=None)
    assert stager.calls == []
    # The chain's next branch (previous tokens without drafts) ran.
    ns["previous_seq_slots_device"].assert_called_once_with()


# -- the runner's prepare window ---------------------------------------------------------------


def _window():
    """The statements of ``_forward_decoder`` from ``stager = None`` through the window's end."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

    tree = ast.parse(textwrap.dedent(inspect.getsource(DecoderRunner._forward_decoder)))
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body):
                if (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "stager_source"
                ):
                    assert ast.unparse(body[i - 1]) == "stager = None"
                    assert isinstance(body[i + 2], ast.Try)
                    end = body[i + 3]
                    assert isinstance(end, ast.If) and ast.unparse(end.test) == "stager is not None"
                    return body[i - 1 : i + 4], body[i + 2].body
    raise AssertionError("no step-input window in DecoderRunner._forward_decoder")


class _GraphRunner:
    is_warmup_only = False

    def __init__(self, log):
        self.log = log

    def needs_capture(self, key):
        self.log.append("needs_capture")
        return False


def _run_window(stager_source, prepare_error=None, graph_runner=None):
    window, prepare = _window()
    log = []
    ns = _mock_namespace(window)
    self = ns["self"]
    self.runtime_hooks = (
        DEFAULT_RUNTIME_HOOKS
        if stager_source is None
        else RuntimeHooks(step_input_stager=stager_source)
    )
    self.cuda_graph_runner = _GraphRunner(log) if graph_runner is None else graph_runner

    def prepare_inputs(*args, **kwargs):
        log.append("prepare")
        if prepare_error is not None:
            raise prepare_error
        return ("inputs", "gather_ids", 3)

    self._prepare_inputs.side_effect = prepare_inputs
    if stager_source is not None:
        stager_source.log = log
    return window, prepare, ns, log


def test_the_prepare_window_runs_only_the_prepare_with_the_hook_off():
    window, (prepare,), ns, log = _run_window(None)
    before = dict(ns)

    _run(window, ns)
    assert log == ["prepare"], "no eligibility check, no graph-key lookup"
    # Exactly the original call: the same arguments, positional and keyword.
    call = ns["self"]._prepare_inputs.call_args
    assert ns["self"]._prepare_inputs.call_count == 1
    assert call.args == tuple(before[a.id] for a in prepare.value.args)
    assert call.kwargs == {k.arg: before[k.value.id] for k in prepare.value.keywords}
    assert "staged_static_inputs" not in vars(ns["self"].cuda_graph_runner)
    assert (ns["inputs"], ns["gather_ids"], ns["runtime_draft_len"]) == ("inputs", "gather_ids", 3)


class _Window:
    def __init__(self, log, static):
        self.log = log
        self.static = static

    def end(self, inputs):
        self.log.append(("end", inputs))

    def discard(self):
        self.log.append("discard")

    def static_inputs(self):
        self.log.append("static_inputs")
        return self.static


class _Source:
    def __init__(self, eligible=True, static=(1, 2, 3)):
        self.log = []
        self.eligible = eligible
        self.static = static
        self.begin_args = None

    def begin_if_eligible(self, *args, **kwargs):
        self.log.append("begin")
        self.begin_args = (args, kwargs)
        return _Window(self.log, self.static) if self.eligible else None


def test_the_prepare_window_brackets_the_prepare_with_a_stager():
    source = _Source()
    window, _, ns, log = _run_window(source)
    _run(window, ns)
    assert log == ["needs_capture", "begin", "prepare", ("end", "inputs"), "static_inputs"]
    args, kwargs = source.begin_args
    assert args == (
        ns["self"],
        ns["execution_requests"],
        ns["attn_metadata"],
        ns["spec_metadata"],
        ns["new_tensors_device"],
    )
    assert set(kwargs) == {"graph_replay", "is_dummy", "enable_spec_decode", "promoted"}
    assert ns["self"].cuda_graph_runner.staged_static_inputs == (1, 2, 3)


def test_the_prepare_window_discards_on_a_failed_prepare():
    source = _Source()
    window, _, ns, log = _run_window(source, prepare_error=RuntimeError("prepare failed"))
    ns["self"].cuda_graph_runner.staged_static_inputs = (1, 2, 3)
    with pytest.raises(RuntimeError, match="prepare failed"):
        _run(window, ns)
    assert log == ["needs_capture", "begin", "prepare", "discard"]
    assert ns["self"].cuda_graph_runner.staged_static_inputs is None


def test_an_ineligible_step_prepares_natively_and_drops_staged_static_inputs():
    source = _Source(eligible=False)
    window, _, ns, log = _run_window(source)
    # Left by an earlier step whose replay never ran.
    ns["self"].cuda_graph_runner.staged_static_inputs = (1, 2, 3)
    _run(window, ns)
    assert log == ["needs_capture", "begin", "prepare"]
    assert ns["self"].cuda_graph_runner.staged_static_inputs is None


# -- the CUDA graph replay's static inputs -----------------------------------------------------


def _graph_runner(log):
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType

    key = KeyType(batch_size=1, draft_len=3, is_first_draft=False)
    runner = CUDAGraphRunner.__new__(CUDAGraphRunner)
    attn = object()
    runner.graph_metadata = {key: {"attn_metadata": attn, "spec_metadata": None}}
    runner._check_strict_buffer_stability = lambda meta: None
    runner.shared_static_tensors = {
        "input_ids": torch.full((8,), -1, dtype=torch.int32),
        "position_ids": torch.full((1, 8), -1, dtype=torch.int32),
    }
    runner._get_num_tokens_for_key = lambda k: 4
    runner.config = types.SimpleNamespace(use_mrope=False)
    runner.graphs = {
        key: types.SimpleNamespace(replay=lambda: log.append("replay"), reset=lambda: None)
    }
    runner.graph_outputs = {key: "outputs"}
    runner.memory_pool = None
    inputs = {
        "attn_metadata": attn,
        "input_ids": torch.arange(4, dtype=torch.int32),
        "position_ids": torch.arange(10, 14, dtype=torch.int32).unsqueeze(0),
    }
    return runner, key, inputs


def test_the_replay_copies_its_static_inputs_with_the_hook_off():
    log = []
    runner, key, inputs = _graph_runner(log)

    assert runner.replay(key, inputs) == "outputs"
    static = runner.shared_static_tensors
    assert static["input_ids"].tolist() == [0, 1, 2, 3, -1, -1, -1, -1]
    assert static["position_ids"].tolist() == [[10, 11, 12, 13, -1, -1, -1, -1]]
    assert log == ["replay"]
    assert "staged_static_inputs" not in vars(runner)


def test_the_replay_skips_the_static_inputs_a_stager_wrote_once():
    log = []
    runner, key, inputs = _graph_runner(log)
    runner.staged_static_inputs = (
        inputs["input_ids"].data_ptr(),
        inputs["position_ids"].data_ptr(),
        4,
    )

    runner.replay(key, inputs)
    static = runner.shared_static_tensors
    assert static["input_ids"].tolist() == [-1] * 8
    assert static["position_ids"].tolist() == [[-1] * 8]
    assert runner.staged_static_inputs is None, "consumed by one replay"
    runner.replay(key, inputs)
    assert static["input_ids"].tolist()[:4] == [0, 1, 2, 3]


def test_the_replay_copies_static_inputs_staged_for_other_tensors():
    runner, key, inputs = _graph_runner([])
    runner.staged_static_inputs = (inputs["input_ids"].data_ptr(), 0, 4)

    runner.replay(key, inputs)
    assert runner.shared_static_tensors["input_ids"].tolist()[:4] == [0, 1, 2, 3]
    assert runner.staged_static_inputs is None


def test_a_declined_step_after_an_unreplayed_staged_step_copies_its_static_inputs():
    """A step's window staged the static inputs, its replay never ran (the forward failed
    first); the next step, which the stager declines, presents the same pointers and token
    count, and its replay still copies its inputs."""
    graph_runner, key, inputs = _graph_runner([])
    graph_runner.is_warmup_only = False
    graph_runner.needs_capture = lambda key: False
    staged = (inputs["input_ids"].data_ptr(), inputs["position_ids"].data_ptr(), 4)

    window, _, ns, _ = _run_window(_Source(static=staged), graph_runner=graph_runner)
    _run(window, ns)
    assert graph_runner.staged_static_inputs == staged

    window, _, ns, _ = _run_window(_Source(eligible=False), graph_runner=graph_runner)
    _run(window, ns)
    graph_runner.replay(key, inputs)
    static = graph_runner.shared_static_tensors
    assert static["input_ids"].tolist()[:4] == [0, 1, 2, 3]
    assert static["position_ids"].tolist()[0][:4] == [10, 11, 12, 13]


# -- the speculative sampler's store -------------------------------------------------------------


class _TorchProxy:
    """``torch`` for ``spec_sampler_base`` on a host without a GPU: records the slot upload."""

    def __init__(self, log):
        self._log = log
        log_ = log

        class _Functional:
            def __getattr__(self, name):
                return getattr(torch.nn.functional, name)

            @staticmethod
            def pad(t, pad):
                log_.append("pad")
                return torch.nn.functional.pad(t, pad)

        self.nn = types.SimpleNamespace(functional=_Functional())

    def __getattr__(self, name):
        return getattr(torch, name)

    def as_tensor(self, data, dtype=None):
        tensor = torch.as_tensor(data, dtype=dtype)
        log = self._log

        class _Host:
            def to(self, device=None, non_blocking=False):
                log.append(("slots", device, non_blocking))
                return tensor

        return _Host()


def _spec_sampler(monkeypatch, log, writer=None):
    from tensorrt_llm._torch.speculative import spec_sampler_base
    from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler

    monkeypatch.setattr(spec_sampler_base, "torch", _TorchProxy(log))
    sampler = SpecSampler.__new__(SpecSampler)
    S, T, D = 4, 4, 3
    sampler.store = types.SimpleNamespace(
        new_tokens=torch.zeros((T, S, 1), dtype=torch.int32),
        next_new_tokens=torch.zeros((T, S, 1), dtype=torch.int32),
        new_tokens_lens=torch.zeros((S,), dtype=torch.int32),
        next_draft_tokens=torch.zeros((S, D), dtype=torch.int32),
    )
    sampler.draft_len = D
    sampler._async_worker = None
    sampler._copy_to_host = lambda t: log.append("d2h") or t.clone()
    sampler._record_sampler_event = lambda: log.append("event") or "event"
    if writer is not None:
        sampler.store_writer = writer
    requests = [
        types.SimpleNamespace(
            py_seq_slot=slot, py_draft_tokens=[0] * D, py_draft_tokens_effective_len=None
        )
        for slot in (2, 0)
    ]
    scheduled = types.SimpleNamespace(
        context_requests_chunking=[], context_requests_last_chunk=[], generation_requests=requests
    )
    # The worker's outputs are one step narrower than the store: the pads run.
    outputs = {
        "new_tokens": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32),
        "new_tokens_lens": torch.tensor([3, 1], dtype=torch.int32),
        "next_draft_tokens": torch.tensor([[7, 8], [9, 10]], dtype=torch.int32),
        "next_new_tokens": torch.tensor([[11, 12, 13], [14, 15, 16]], dtype=torch.int32),
    }
    return sampler, scheduled, outputs


def test_the_sampler_stores_and_copies_natively_with_the_hook_off(monkeypatch):
    log = []
    sampler, scheduled, outputs = _spec_sampler(monkeypatch, log)

    state = sampler.sample_async(scheduled, outputs, [])
    # The slot upload sits before the pads, as it always did.
    assert log == [("slots", "cuda", True), "pad", "pad", "pad", "d2h", "d2h", "d2h", "event"]
    store = sampler.store
    assert store.new_tokens_lens.tolist() == [1, 0, 3, 0]
    assert store.new_tokens[:, 2, 0].tolist() == [1, 2, 3, 0]
    assert store.next_draft_tokens[0].tolist() == [9, 10, 0]
    assert torch.equal(state.host.new_tokens, store.new_tokens)
    assert state.host.verify_lens is None


def test_the_sampler_hands_the_store_to_an_accepting_writer(monkeypatch):
    log = []
    host = tuple(object() for _ in range(3))
    calls = []

    class _Writer:
        def store_and_copy_out(self, *args):
            calls.append(args)
            log.append("writer")
            return host

    sampler, scheduled, outputs = _spec_sampler(monkeypatch, log, writer=_Writer())
    state = sampler.sample_async(scheduled, outputs, [])
    assert log == ["pad", "pad", "pad", "writer", "event"]
    ((owner, slots, *tensors, verify_lens),) = calls
    assert owner is sampler and slots == [2, 0] and verify_lens is None and len(tensors) == 4
    assert int(sampler.store.new_tokens_lens.abs().sum()) == 0, "the store is the writer's"
    assert (state.host.new_tokens, state.host.new_tokens_lens, state.host.next_draft_tokens) == host


def test_the_sampler_stores_natively_after_a_writer_declines(monkeypatch):
    log = []

    class _Writer:
        def store_and_copy_out(self, *args):
            log.append("writer")
            return None

    sampler, scheduled, outputs = _spec_sampler(monkeypatch, log, writer=_Writer())
    sampler.sample_async(scheduled, outputs, [])
    assert log == [
        "pad",
        "pad",
        "pad",
        "writer",
        ("slots", "cuda", True),
        "d2h",
        "d2h",
        "d2h",
        "event",
    ]
    assert sampler.store.new_tokens_lens.tolist() == [1, 0, 3, 0]


# -- binding the sampler store writer ------------------------------------------------------------


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


def test_instantiate_sampler_leaves_other_spec_decoders_alone(monkeypatch):
    sampler = types.SimpleNamespace()

    assert _instantiate_sampler(monkeypatch, _StoreWriterModel(object()), sampler) is sampler
    assert "store_writer" not in vars(sampler)


# -- the runner reads its own model's hooks ------------------------------------------------------


class _Stop(Exception):
    """Raised by the stand-in config: the rest of the runner's constructor needs an engine."""


class _StopConfig:
    def __getattr__(self, name):
        raise _Stop(name)


@pytest.mark.parametrize("declared", [False, True])
def test_the_runner_reads_the_hooks_of_its_model(declared):
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner

    model = _PlainModel()
    hooks = RuntimeHooks(step_input_stager=object())
    if declared:
        model.runtime_hooks = hooks
    runner = DecoderRunner.__new__(DecoderRunner)
    others = {
        name: mock.MagicMock(name=name)
        for name in inspect.signature(DecoderRunner.__init__).parameters
        if name not in ("self", "model", "config")
    }
    with pytest.raises(_Stop):
        DecoderRunner.__init__(runner, model, _StopConfig(), **others)
    assert runner.runtime_hooks is (hooks if declared else DEFAULT_RUNTIME_HOOKS)
