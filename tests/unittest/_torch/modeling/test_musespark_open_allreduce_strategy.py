# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The all-reduce strategy MuseSparkOpen builds its all-reduces with.

AUTO, the configured default, fails this model at TP4. On the row-parallel
attention output projection it routes each decode step's tiny message through
the NCCL symmetric-memory all-reduce, which reuses a shared buffer across
launches without synchronising: ranks overwrite each other's payload, the layer
output comes back non-finite, and argmax over the all-NaN row collapses every
request to token 0. On the MoE combine its autotuner hangs engine warmup. Both
all-reduces therefore take NCCL in place of AUTO. An explicit NCCL_SYMMETRIC,
the corrupting path itself, is rejected; any other configured strategy is used
as given.
"""

import ast
import inspect
import pathlib

import pytest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_musespark_open
from tensorrt_llm._torch.models.modeling_musespark_open import _allreduce_strategy
from tensorrt_llm.functional import AllReduceStrategy

pytestmark = pytest.mark.cpu_only


def _method_node(module, class_name: str, method: str) -> ast.FunctionDef:
    """The ast node of one method, parsed from the module file.

    `inspect.getsource` on a method needs the function object; going through
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


def test_auto_becomes_nccl():
    """AUTO is the configured default, and it is never what this model builds."""
    model_config = ModelConfig()
    assert model_config.allreduce_strategy is AllReduceStrategy.AUTO
    assert _allreduce_strategy(model_config) is AllReduceStrategy.NCCL


@pytest.mark.parametrize(
    "strategy",
    [
        strategy
        for strategy in AllReduceStrategy
        if strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.NCCL_SYMMETRIC)
    ],
    ids=lambda strategy: strategy.name,
)
def test_an_explicit_strategy_is_used_as_configured(strategy):
    assert _allreduce_strategy(ModelConfig(allreduce_strategy=strategy)) is strategy


@pytest.mark.parametrize("strategy", [AllReduceStrategy.NCCL_SYMMETRIC, "NCCL_SYMMETRIC"])
def test_an_explicit_nccl_symmetric_is_rejected(strategy):
    """An explicit setting must not bring back the corruption AUTO is remapped to avoid."""
    with pytest.raises(ValueError, match="NCCL_SYMMETRIC"):
        _allreduce_strategy(ModelConfig(allreduce_strategy=strategy))


def test_a_strategy_named_in_the_llm_args_is_used_as_configured():
    """`allreduce_strategy` reaches the model config as a string."""
    model_config = ModelConfig(allreduce_strategy="ONESHOT")
    assert _allreduce_strategy(model_config) is AllReduceStrategy.ONESHOT


@pytest.mark.parametrize("class_name", ["MuseSparkOpenAttention", "MuseSparkOpenMoE"])
def test_both_allreduces_use_the_resolved_strategy(class_name):
    """`__init__` must build its AllReduce from the resolver, not the config.

    A source check: constructing either module needs a full model config and a
    mapping with a real TP group. What can regress cheaply is the wiring --
    someone dropping the override and letting the config's AUTO through -- and
    that is what this pins.
    """
    source = _method_source(modeling_musespark_open, class_name, "__init__")
    assert "strategy=_allreduce_strategy(model_config)" in source
    assert "AllReduceStrategy.AUTO" not in source


def test_the_moe_combine_is_cast_down_before_it_is_reduced():
    """The combine crosses the wire in the activation dtype, not fp32.

    The finalize produces FP32 and the unpooler wants the activation dtype, so
    the cast happens either way; only its position decides whether the
    collective moves two bytes per element or four. Putting it after the reduce
    costs about 0.6 ms of every decode iteration and is a one-line regression
    that no unit assertion on the module's output could see -- the output dtype
    is the same on both sides of it. So it is pinned as an ordering check on
    the code, in the same spirit as the strategy check above.
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
