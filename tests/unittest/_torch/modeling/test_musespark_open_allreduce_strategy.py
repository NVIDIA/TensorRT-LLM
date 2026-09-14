# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The all-reduce strategy MuseSparkOpen pins, and the knob that overrides it.

The row-parallel attention output projection takes its strategy from the model
config, which defaults to AUTO. At TP4 on one node a decode step reduces a tiny
message, and AUTO routes tiny messages through the NCCL symmetric-memory
all-reduce, which reuses a shared buffer across launches without
synchronising: ranks overwrite each other's payload and the layer output comes
back non-finite. Argmax over an all-NaN row is index 0, which is the token-0
collapse the bring-up chased.

So NCCL is the default here, on both the attention and the MoE all-reduce, and
``MUSESPARK_OPEN_ALLREDUCE`` names a different one for a single attribution leg.
A typo in that variable must not silently decide correctness, so an unparsable
name keeps the pin instead of failing engine construction.
"""

import ast
import inspect
import pathlib

import pytest

from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import _attention_allreduce_strategy
from tensorrt_llm.functional import AllReduceStrategy


def _method_node(module, class_name: str, method: str) -> ast.FunctionDef:
    """The ast node of one method, parsed from the module file.

    ``inspect.getsource`` on a method needs the function object; going through
    the file keeps this a check on the code as written, and works the same
    whichever way the module was loaded. The node keeps the file's line
    numbers, so statement order inside the method is readable off it.
    """
    path = inspect.getsourcefile(module)
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method:
                    return sub
    raise AssertionError(f"{class_name}.{method} not found in {path}")


def _method_source(module, class_name: str, method: str) -> str:
    """The source text of one method."""
    path = inspect.getsourcefile(module)
    return ast.get_source_segment(
        pathlib.Path(path).read_text(encoding="utf-8"),
        _method_node(module, class_name, method),
    )


def test_the_default_is_nccl_not_auto(monkeypatch):
    """AUTO is never selected by default. That is the whole point of the pin."""
    monkeypatch.delenv("MUSESPARK_OPEN_ALLREDUCE", raising=False)
    assert _attention_allreduce_strategy() is AllReduceStrategy.NCCL


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_value_is_the_default(monkeypatch, value):
    """An exported-but-empty variable is not a request for something else."""
    monkeypatch.setenv("MUSESPARK_OPEN_ALLREDUCE", value)
    assert _attention_allreduce_strategy() is AllReduceStrategy.NCCL


@pytest.mark.parametrize("name", ["AUTO", "NCCL", "ONESHOT", "TWOSHOT"])
def test_the_knob_selects_a_named_strategy(monkeypatch, name):
    """Including AUTO: a control leg needs the racy default to reproduce the fault."""
    monkeypatch.setenv("MUSESPARK_OPEN_ALLREDUCE", name)
    assert _attention_allreduce_strategy() is AllReduceStrategy[name]


@pytest.mark.parametrize("spelling", ["auto", "Auto", "  auto  "])
def test_the_knob_is_case_and_whitespace_insensitive(monkeypatch, spelling):
    monkeypatch.setenv("MUSESPARK_OPEN_ALLREDUCE", spelling)
    assert _attention_allreduce_strategy() is AllReduceStrategy.AUTO


@pytest.mark.parametrize("name", ["NCLL", "nccl2", "0", "none"])
def test_an_unparsable_name_keeps_the_pin(monkeypatch, name):
    """A typo must not quietly restore the failing default."""
    monkeypatch.setenv("MUSESPARK_OPEN_ALLREDUCE", name)
    assert _attention_allreduce_strategy() is AllReduceStrategy.NCCL


def test_the_attention_projection_uses_the_resolved_strategy():
    """``__init__`` must build its AllReduce from the resolver, not the config.

    A source check: constructing the attention module needs a full model config
    and a mapping with a real TP group. What can regress cheaply is the wiring
    -- someone dropping the override and letting ``o_proj`` keep the config's
    AUTO -- and that is what this pins.
    """
    source = _method_source(modeling_musespark_open, "MuseSparkOpenAttention", "__init__")
    assert "self.o_proj.all_reduce = AllReduce(" in source
    assert "strategy=_attention_allreduce_strategy()" in source


def test_the_moe_allreduce_is_pinned_to_nccl():
    """The MoE side is pinned outright: no knob, no AUTO, no tunable autotuner."""
    source = _method_source(modeling_musespark_open, "MuseSparkOpenMoE", "__init__")
    assert "strategy=AllReduceStrategy.NCCL" in source
    assert "AllReduceStrategy.AUTO" not in source


def test_the_moe_combine_is_cast_down_before_it_is_reduced():
    """The combine crosses the wire in the activation dtype, not fp32.

    The finalize produces FP32 and the unpooler wants the activation dtype, so
    the cast happens either way; only its position decides whether the
    collective moves two bytes per element or four. Putting it after the reduce
    costs about 0.6 ms of every decode iteration and is a one-line regression
    that no unit assertion on the module's output could see -- the output dtype
    is the same on both sides of it. So it is pinned as an ordering check on
    the code, in the same spirit as the strategy pins above.
    """
    forward = _method_node(modeling_musespark_open, "MuseSparkOpenMoE", "forward")

    casts = [
        node.lineno
        for node in ast.walk(forward)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "combined" for t in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "to"
    ]
    reduces = [
        node.lineno
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "all_reduce"
    ]

    assert casts, "the combine no longer casts to the activation dtype"
    assert reduces, "the combine no longer all-reduces"
    assert max(casts) < min(reduces), (
        "the MoE combine is all-reduced before it is cast down, so the collective moves FP32 bytes"
    )
