# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression boundaries for decorated signatures and cached import bindings."""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest
from cbts.coverage.selection.declaration_policy import DecoratorFacts, analyze_decorators
from cbts.coverage.selection.python_change_analysis import analyze_python_changes
from cbts.coverage.selection.selector import CoverageSelector
from cbts.rules._helpers import iter_diff_deleted_post_lines, iter_diff_post_line_numbers

from .test_selector import _FakeDB

pytestmark = pytest.mark.cpu_only
REPO = Path(__file__).resolve().parents[4]
PATH = "tensorrt_llm/_torch/pyexecutor/example.py"


def _images(path: str) -> tuple[str, str] | None:
    file = REPO / path
    if not file.is_file():
        return None
    source = file.read_text()
    return source, source


def _analyze(before: str, source: str, proof: DecoratorFacts | None = None):
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    return analyze_python_changes(
        source,
        iter_diff_post_line_numbers(diff),
        iter_diff_deleted_post_lines(diff),
        pre_source=before,
        per_call_functions=proof.per_call if proof else set(),
        safe_decorated_functions=proof.signature_safe if proof else set(),
    )


@pytest.mark.parametrize(
    ("import_line", "decorator", "source_proven"),
    (
        ("from ..._utils import nvtx_range as profile", 'profile("work")', False),
        ("from ..hostfunc import hostfunc as callback", "callback", True),
    ),
)
def test_repository_wrappers_require_source_proofs_for_decoration(
    import_line: str, decorator: str, source_proven: bool
):
    before = f"{import_line}\n@{decorator}\ndef caller(value) -> int:\n    return value\n"
    source = before.replace("caller(value) -> int", "caller(value, option=0) -> None")
    proof = analyze_decorators(PATH, source, before, _images)
    assert bool(proof.signature_safe) is source_proven
    assert not proof.per_call
    assert bool(proof.wrapper_bounds) is source_proven
    facts = _analyze(before, source, proof)
    assert bool(facts.limitation) is not source_proven
    assert _analyze(before, source).limitation


@pytest.mark.parametrize(
    "prefix",
    (
        "from somewhere import nvtx_range as profile\n",
        "from ..._utils import nvtx_range as profile\nprofile = cache\n",
        "from ..._utils import nvtx_range as profile\n"
        "def replace():\n    global profile\n    profile = cache\n",
        "from ..._utils import nvtx_range as profile\n"
        "def replace():\n    profile.wrapper = cache\n",
        "from ..._utils import nvtx_range as profile\nglobals()['profile'] = cache\n",
    ),
)
def test_lookalike_or_mutated_decorators_gain_no_exception(prefix: str):
    source = prefix + '@profile("work")\ndef caller():\n    return 1\n'
    assert not analyze_decorators(PATH, source, source, _images).signature_safe


@pytest.mark.parametrize(
    "decorator", ("profile(make_name())", 'profile("work", **options)', "cache")
)
def test_unresolved_decorator_options_or_extra_cache_remain_untrusted(decorator: str):
    source = (
        f"from ..._utils import nvtx_range as profile\n@{decorator}\ndef caller():\n    return 1\n"
    )
    assert not analyze_decorators(PATH, source, source, _images).signature_safe


@pytest.mark.parametrize("old_provider", (False, True))
def test_provider_implementation_changes_are_checked_in_both_images(old_provider: bool):
    source = "from project.hooks import decorate\n@decorate\ndef caller():\n    return 1\n"
    provider = "import external\ndef decorate(action):\n    return action\n"
    proof = analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (provider, provider) if path == "project/hooks.py" else None,
    )
    assert proof.per_call == {"caller"}

    def images(path: str):
        if path != "project/hooks.py":
            return None
        changed = provider.replace("return action", "return external.register(action)")
        return (provider, changed) if old_provider else (changed, provider)

    assert not analyze_decorators(PATH, source, source, images).signature_safe


def test_cached_and_profiled_function_still_requires_complete_callers():
    source = (
        "from ..._utils import nvtx_range\nfrom functools import cache\n"
        '@nvtx_range("work")\n@cache\ndef caller():\n    return 1\n'
    )
    assert not analyze_decorators(PATH, source, source, _images).signature_safe
    facts = _analyze(source, source.replace("return 1", "return 2"))
    assert "caller" in facts.caller_bound_required


@pytest.mark.parametrize(
    "mutation", ("", "wrapped = cache(wrapped)\n", "functools.wraps = cache\n")
)
def test_external_wraps_is_not_assumed_metadata_only(mutation: str):
    source = (
        "import functools\ndef wrapped(value):\n    return value\n"
        + mutation
        + "@functools.wraps(wrapped)\ndef caller(value):\n    return wrapped(value)\n"
    )
    proof = analyze_decorators(PATH, source, source, _images)
    assert not proof.per_call
    assert not proof.signature_safe


@pytest.mark.parametrize("at_eof", (False, True))
def test_deleting_last_method_statement_preserves_its_body_consumer(at_eof: bool):
    suffix = "" if at_eof else "\n    def other(self):\n        return 0\n"
    before = "class Worker:\n    def run(self):\n        consume()\n        return 1\n" + suffix
    source = before.replace("        return 1\n", "")
    facts = _analyze(before, source)
    assert not facts.limitation
    assert "Worker.run" in facts.binding_consumers


@pytest.mark.parametrize("statement", ("    value = register()\n", "    value = 1\n"))
def test_class_field_deletion_is_not_reclassified_as_a_method_body(statement: str):
    before = "class Worker:\n" + statement + "\n    def run(self):\n        consume()\n"
    assert _analyze(before, before.replace(statement, "")).limitation


@pytest.mark.parametrize("annotation", ("factory()", "Unknown | None", "custom.Generic[int]"))
def test_return_annotation_changes_still_reject_executable_or_unknown_types(annotation: str):
    before = f"def caller() -> {annotation}:\n    return 1\n"
    assert _analyze(before, "def caller() -> None:\n    return 1\n").limitation


@pytest.mark.parametrize(
    ("provider", "cached", "allowed"),
    (
        ("class Budget:\n    pass\n", True, True),
        ("class Base:\n    pass\nclass Budget(Base):\n    pass\n", True, True),
        ("class Budget(metaclass=Meta):\n    pass\n", True, False),
        ("@register\nclass Budget:\n    pass\n", True, False),
        ("class Budget(ExternalBase):\n    pass\n", True, False),
        ("class Budget:\n    pass\nBudget = factory()\n", True, False),
        ("class Budget:\n    pass\nfrom other import *\n", True, False),
        ("class Budget:\n    pass\nexec(code)\n", True, False),
        (
            "class Budget:\n    pass\ndef change():\n    global Budget\n    Budget = factory()\n",
            True,
            False,
        ),
        ("class Budget:\n    pass\n", False, False),
    ),
)
def test_imported_annotations_use_sanity_and_direct_rows(
    tmp_path: Path, provider: str, cached: bool, allowed: bool
):
    prefix = "from typing import Optional\n"
    before_import = "from .helpers import Existing\n" if cached else ""
    after_import = (
        "from .helpers import Existing, Budget\n" if cached else "from .helpers import Budget\n"
    )
    before = prefix + before_import + "def caller():\n    return 1\n"
    source = prefix + after_import + "def caller(budget: Optional[Budget] = None):\n    return 2\n"
    path = "tensorrt_llm/example.py"
    sources = {path: source, "tensorrt_llm/helpers.py": "class Existing:\n    pass\n" + provider}
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    db = _FakeDB()
    db.rows["caller"] = {"A10-PyTorch-1/first", "A10-PyTorch-1/second"}
    db.known_by_family = lambda: {"A10-PyTorch": {"first", "second", "unrelated"}}
    result = CoverageSelector(
        db,
        tmp_path,
        import_sanity_files={path},
        read_source=sources.get,
        external_references=lambda _p, _n: set(),
    ).decide([path], {path: diff})
    assert result.ok, result.reason
    assert result.impacted == {"A10-PyTorch": {"first", "second"}}
    assert result.skippable == {"A10-PyTorch": {"unrelated"}}


@pytest.mark.parametrize(
    "mutation",
    (
        "queue.Empty = factory()\n",
        "del queue.Empty\n",
        "setattr(queue, 'Empty', factory())\n",
        "setattr(queue, name, factory())\n",
    ),
)
def test_cached_exception_import_does_not_hide_provider_mutation(mutation: str):
    prefix = "import queue\nfrom queue import Queue\n" + mutation
    before = prefix + "def caller():\n    return 1\n"
    source = prefix + "from queue import Empty\ndef caller():\n    raise Empty\n"
    facts = _analyze(before, source)
    assert facts.limitation or any(
        target.module == "queue" and target.name == "Empty" for target in facts.new_import_targets
    )


def test_cached_exception_import_still_requires_target_resolution():
    before = "from queue import Queue\ndef caller():\n    return 1\n"
    source = "from queue import Empty, Queue\ndef caller():\n    raise Empty\n"
    facts = _analyze(before, source)
    assert not facts.limitation
    assert any(
        target.module == "queue" and target.name == "Empty" for target in facts.new_import_targets
    )
    assert facts.binding_consumers == {"caller"}


@pytest.mark.parametrize(
    "provider",
    (
        (
            "def arbitrary_name(action):\n"
            "    def invoke(*args, **kwargs):\n"
            "        return action(*args, **kwargs)\n"
            "    return invoke\n"
        ),
        (
            "def make_label():\n"
            "    return 'decorate'\n"
            "def arbitrary_name(action):\n"
            "    label = make_label()\n"
            "    alias = action\n"
            "    def invoke(*args, **kwargs):\n"
            "        return alias(*args, **kwargs)\n"
            "    return invoke\n"
        ),
        (
            "def arbitrary_name(action):\n"
            "    cache = {}\n"
            "    def invoke(*args, **kwargs):\n"
            "        if args not in cache:\n"
            "            cache[args] = action(*args, **kwargs)\n"
            "        return cache[args]\n"
            "    return invoke\n"
        ),
    ),
)
def test_arbitrary_repository_decorators_use_wrapper_bounds(provider: str):
    source = "from project.hooks import arbitrary_name as decorate\n@decorate\ndef caller():\n    return 1\n"
    facts = analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (provider, provider) if path == "project/hooks.py" else None,
    )
    assert facts.signature_safe == {"caller"}
    assert not facts.per_call
    assert facts.wrapper_bounds == {
        "caller": {("project/hooks.py", "arbitrary_name.<locals>.invoke")}
    }


def test_factory_aliases_and_refactored_closures_follow_both_images():
    before = (
        "def factory(label):\n"
        "    def decorate(fn):\n"
        "        def original(*a, **kw):\n"
        "            return fn(*a, **kw)\n"
        "        return original\n"
        "    return decorate\n"
    )
    after = before.replace("def original", "def renamed").replace(
        "return original", "result = renamed\n        return result"
    )
    source = "from project.hooks import factory\n@factory('work')\ndef caller():\n    return 1\n"
    facts = analyze_decorators(
        PATH, source, source, lambda path: (after, before) if path == "project/hooks.py" else None
    )
    assert facts.signature_safe == {"caller"}
    assert facts.wrapper_bounds["caller"] == {
        ("project/hooks.py", "factory.<locals>.decorate.<locals>.original"),
        ("project/hooks.py", "factory.<locals>.decorate.<locals>.renamed"),
    }


@pytest.mark.parametrize(
    "effect",
    (
        "schema = action.__annotations__",
        "lookup = {action.__annotations__['return']: 0}",
        "external.register(action)",
        "external.audit()",
        "registry['entry'] = action",
        "action()",
    ),
)
@pytest.mark.parametrize("changed_image", (0, 1))
def test_signature_inspection_registration_and_invocation_decline_in_either_image(
    effect: str, changed_image: int
):
    provider = (
        "import external\n"
        "registry = {}\n"
        "def arbitrary_name(action):\n"
        "    def invoke(*args, **kwargs):\n"
        "        return action(*args, **kwargs)\n"
        "    return invoke\n"
    )
    changed = provider.replace("    def invoke", f"    {effect}\n    def invoke")
    images = [provider, provider]
    images[changed_image] = changed
    source = (
        "from project.hooks import arbitrary_name\n@arbitrary_name\ndef caller():\n    return 1\n"
    )
    facts = analyze_decorators(
        PATH, source, source, lambda path: tuple(images) if path == "project/hooks.py" else None
    )
    assert not facts.signature_safe


@pytest.mark.parametrize("helper_change", (False, True))
@pytest.mark.parametrize("wrapper_recorded", (False, True))
def test_custom_cache_does_not_expand_direct_lookup(
    tmp_path: Path, helper_change: bool, wrapper_recorded: bool
):
    provider = (
        "def memoize(work):\n"
        "    cache = {}\n"
        "    def entry(*args, **kwargs):\n"
        "        if args not in cache:\n"
        "            cache[args] = work(*args, **kwargs)\n"
        "        return cache[args]\n"
        "    return entry\n"
    )
    before = (
        "from .wrappers import memoize\n"
        + ("def helper():\n    return 1\n" if helper_change else "")
        + "@memoize\ndef caller():\n    return "
        + ("helper()" if helper_change else "1")
        + "\n"
    )
    source = before.replace("return 1", "return 2")
    if not helper_change:
        source = source.replace("caller():", "caller(option=None):")
    path = "tensorrt_llm/example.py"
    sources = {path: source, "tensorrt_llm/wrappers.py": provider}
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    db = _FakeDB()
    db.rows = {
        "helper": {"A10-PyTorch-1/first"},
        "caller": {"A10-PyTorch-1/first", "A10-PyTorch-1/deferred"},
        "memoize.<locals>.entry": {"A10-PyTorch-1/first", "A10-PyTorch-1/second"}
        if wrapper_recorded
        else set(),
    }
    db.known_by_family = lambda: {"A10-PyTorch": {"first", "second", "deferred", "unrelated"}}
    result = CoverageSelector(
        db,
        tmp_path,
        import_sanity_files={path},
        read_source=sources.get,
        external_reference_paths=lambda _p, _n: {},
    ).decide([path], {path: diff})
    assert result.ok, result.reason
    expected = {"first"} if helper_change else {"first", "deferred"}
    assert result.impacted == {"A10-PyTorch": expected}
    assert result.skippable == {
        "A10-PyTorch": {"first", "second", "deferred", "unrelated"} - expected
    }


@pytest.mark.parametrize("parameter", ("args", "kwargs"))
def test_wrapper_variadic_annotations_are_checked_at_declaration_time(parameter: str):
    provider = (
        "def trace(action):\n"
        "    def entry(*args, **kwargs):\n"
        "        return action(*args, **kwargs)\n"
        "    return entry\n"
    ).replace(
        parameter + "," if parameter == "args" else parameter + ")",
        parameter + ": action.__annotations__['return']" + ("," if parameter == "args" else ")"),
        1,
    )
    source = "from project.hooks import trace\n@trace\ndef caller():\n    return 1\n"
    facts = analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (provider, provider) if path == "project/hooks.py" else None,
    )
    assert not facts.signature_safe


@pytest.mark.parametrize("changed_image", (0, 1))
def test_unparsable_repository_module_cannot_borrow_an_external_api_contract(changed_image: int):
    source = "import nvtx\n@nvtx.annotate('work')\ndef caller():\n    return 1\n"
    images = ["def annotate(value):\n    return value\n"] * 2
    images[changed_image] = "def annotate("
    facts = analyze_decorators(
        PATH, source, source, lambda path: tuple(images) if path == "nvtx.py" else None
    )
    assert not facts.signature_safe


@pytest.mark.parametrize(
    "decorator",
    (
        "click.command('demo')",
        "click.option('--value')",
        "click.argument('value')",
        "nvtx.annotate('work')",
        "functools.wraps(original)",
        "external.decorate",
    ),
)
def test_external_decorator_names_never_supply_a_proof(decorator: str):
    source = (
        "import click, nvtx, functools, external\n"
        "def original():\n    return 1\n"
        f"@{decorator}\ndef caller():\n    return 1\n"
    )
    proof = analyze_decorators(PATH, source, source, lambda _path: None)
    assert not proof.signature_safe
    assert not proof.per_call
    assert not proof.wrapper_bounds


@pytest.mark.parametrize("effect", ("external.prepare()", "value = external.lookup()"))
def test_opaque_factory_calls_decline_even_before_receiving_the_callable(effect: str):
    provider = (
        "import external\n"
        "def factory():\n"
        f"    {effect}\n"
        "    def forward(action):\n        return action\n"
        "    return forward\n"
    )
    source = "from project.hooks import factory\n@factory()\ndef caller():\n    return 1\n"
    proof = analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (provider, provider) if path == "project/hooks.py" else None,
    )
    assert not proof.signature_safe
    assert not proof.per_call
    assert not proof.wrapper_bounds


@pytest.mark.parametrize(
    ("module", "member"),
    (
        ("project.hooks", "factory"),
        ("click", "command"),
        ("nvtx", "annotate"),
        ("functools", "wraps"),
    ),
)
def test_supplied_source_is_analyzed_independently_of_library_names(module: str, member: str):
    provider = (
        f"def {member}(*labels, **options):\n"
        "    def forward(action):\n"
        "        return action\n"
        "    return forward\n"
    )
    source = f"import {module} as provider\n@provider.{member}('work', enabled=True)\ndef caller():\n    return 1\n"
    proof = analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (provider, provider) if path == module.replace(".", "/") + ".py" else None,
    )
    assert proof.signature_safe == {"caller"}
    assert proof.per_call == {"caller"}
    assert not proof.wrapper_bounds


@pytest.mark.parametrize(
    "mutation",
    (
        "decorate = external.cache(decorate)\n",
        "decorate.wrapper = external.cache\n",
        "setattr(decorate, attribute, external.cache)\n",
        "def replace():\n    global decorate\n    decorate = external.cache(decorate)\n",
    ),
)
def test_source_proven_decorator_rebinding_and_mutation_still_decline(mutation: str):
    provider = "import external\ndef decorate(action):\n    return action\n"
    source = "from project.hooks import decorate\n@decorate\ndef caller():\n    return 1\n"
    before = analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (provider, provider) if path == "project/hooks.py" else None,
    )
    assert before.per_call == {"caller"}
    changed = provider + mutation
    after = analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (changed, provider) if path == "project/hooks.py" else None,
    )
    assert not after.signature_safe
    assert not after.per_call


@pytest.mark.parametrize("extra_cache", (False, True))
def test_opaque_command_helper_uses_only_changed_helper_rows(tmp_path: Path, extra_cache: bool):
    before = (
        "import click\nfrom functools import cache\nimport importlib\n"
        "def helper():\n    return 1\n"
        "@click.command('demo')\n"
        + ("@cache\n" if extra_cache else "")
        + "def caller():\n    return helper()\n"
        "def load(name):\n    return getattr(importlib.import_module(name), 'handler')\n"
    )
    source = before.replace("return 1", "return 2")
    path = "tensorrt_llm/example.py"
    db = _FakeDB()
    db.rows = {"helper": {"A10-PyTorch-1/affected"}, "caller": {"A10-PyTorch-1/affected"}}
    db.known_by_family = lambda: {"A10-PyTorch": {"affected", "unrelated"}}
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    result = CoverageSelector(
        db,
        tmp_path,
        read_source=lambda target: source if target == path else None,
        external_reference_paths=lambda _path, _names: {},
    ).decide([path], {path: diff})
    assert result.ok, result.reason
    assert result.impacted == {"A10-PyTorch": {"affected"}}
    assert result.skippable == {"A10-PyTorch": {"unrelated"}}


@pytest.mark.parametrize(
    "effect",
    (
        "metadata = f'{0:{action.__annotations__}}'",
        "metadata = f'{external.value}'",
        "def entry(*args: action.__annotations__):\n        return action(*args)",
    ),
)
def test_new_syntax_cannot_hide_callable_metadata_or_opaque_calls(effect: str):
    provider = (
        "from __future__ import annotations\nimport external\n"
        "def decorate(action):\n"
        f"    {effect}\n"
        "    def wrapper(*args):\n        return action(*args)\n"
        "    return wrapper\n"
    )
    source = "from project.helpers import decorate\n@decorate\ndef caller():\n    return 1\n"
    assert not analyze_decorators(
        PATH,
        source,
        source,
        lambda path: (provider, provider) if path == "project/helpers.py" else None,
    ).signature_safe
