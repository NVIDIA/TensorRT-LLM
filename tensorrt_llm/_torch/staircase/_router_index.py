# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture name -> routing module, and the resolver that reads the table.

Staircase targets are keyed by a *synthetic* architecture name that no
checkpoint declares. The checkpoint's own ``architectures[0]`` stays what it
always was (``GptOssForCausalLM``, ``DeepseekV3ForCausalLM``), so it cannot
also be the target selector; ``staircase_resolve`` rewrites it into the
synthetic name, and ``AutoModelForCausalLM._resolve_class`` then looks that up
in the ordinary registry.

The pattern is upstream's own: ``MTPDraftModelForCausalLM`` is likewise a name
no config.json declares, reached only through a rewrite in ``_resolve_class``.

Two levels, on purpose:

* This table maps ``architectures[0]`` -> routing module. Its keys are things
  that already exist in every checkpoint's config.json, so adding a new
  checkpoint, GPU arch or parallel topology never touches it -- only a new
  architecture family does.
* The routing module owns the rest of the decision as one forward-reading
  tree. Reading that single file tells you where any configuration lands, and
  ``explain.py`` replays the same tree to say *why*.

Routing modules are imported lazily: resolving a GptOss config never imports
the DeepSeek tree, and a target's modeling code is imported only once its
routing module has claimed the config.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig

_PACKAGE = "tensorrt_llm._torch.staircase"

#: The switch. An environment variable rather than an LLM-API field, so that
#: nothing outside this package has to carry the concept: the only upstream
#: change staircase needs is the ``_resolve_class`` hook itself.
#:
#: It has to be exported **before the ranks start**, not merely before
#: ``LLM(...)``. Worker ranks receive the environment as it stood when MPI
#: initialized, and long-lived ranks under ``trtllm-llmapi-launch`` receive it
#: once at launch, so a value set later reaches the driver and not them -- and
#: a driver that resolves a staircase target while its workers resolve the
#: built-in is exactly the silent split this package exists to prevent. Export
#: it in the shell, or before ``import tensorrt_llm``.
STAIRCASE_ENV = "TRTLLM_STAIRCASE"

# architectures[0] -> routing module, relative to this package.
STAIRCASE_ROUTERS = {
    "GptOssForCausalLM": "models.gpt_oss.routing",
    "DeepseekV3ForCausalLM": "models.deepseek_v3.routing",
}


class StaircaseMode(str, enum.Enum):
    """What to do when a config reaches the staircase resolver."""

    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"

    @classmethod
    def from_env(cls) -> "StaircaseMode":
        """Read ``TRTLLM_STAIRCASE``; unset means off.

        An unknown value raises rather than falling back. Silently reading a
        typo as "off" would hand back the built-in implementation while the
        caller believed they had asked for a staircase target, and a number
        measured that way is attributed to the wrong system -- the failure the
        ``require`` mode below exists to prevent, arriving through the door
        instead of the window.
        """
        raw = os.environ.get(STAIRCASE_ENV)
        if raw is None or raw == "":
            return cls.OFF
        try:
            return cls(raw.strip().lower())
        except ValueError:
            raise ValueError(
                f"{STAIRCASE_ENV}={raw!r} is not a staircase mode; expected "
                f"one of {', '.join(m.value for m in cls)}"
            ) from None


@dataclass(frozen=True)
class StaircaseContext:
    """Everything a routing decision is allowed to depend on.

    The admission rule is one line: a quantity may live here only if it is
    already known when ``_resolve_class`` runs *and* does not change for the
    rest of the engine's life. Config shape, SM version, the parallel mapping,
    the quant and speculative configs and the disagg flag all qualify. Batch
    composition, ``num_contexts``, "is this step pure decode" do not -- they
    move every forward, and a target selected from them would be selected
    once and then be wrong.

    That is a necessary condition, not a sufficient one. ``max_num_tokens``
    and ``cuda_graph_config.max_batch_size`` are per-instance constants too,
    but they are tuning knobs: they belong in a ``configs/`` variant. Only an
    instance constant that changes the *structure of the forward* earns a
    target of its own.

    ``is_disagg`` is declared but **not yet plumbed**: nothing sets it on
    ``ModelConfig``, so it reads False in every deployment, disaggregated or
    not. It is here so the field exists the day it is, and a claim test
    forbids any routing module from reading it until then -- a branch on a
    value that is always False would take the wrong side silently, which is
    the failure this package's ``require`` mode exists to prevent. The useful
    disagg dimension is the instance's ``ServerRole``, and that stops at the
    server layer today; wiring it into ``LlmArgs`` is a prerequisite, not
    part of this work.
    """

    pretrained_config: Any
    mapping: Any
    sm: Tuple[int, int]
    quant_config: Any
    spec_config: Any
    is_disagg: bool

    @classmethod
    def from_model_config(cls, config: "ModelConfig") -> "StaircaseContext":
        import torch

        assert torch.cuda.is_available(), (
            "staircase routes on the SM version of the device it will run on; "
            "no CUDA device is visible"
        )
        return cls(
            pretrained_config=config.pretrained_config,
            mapping=config.mapping,
            sm=torch.cuda.get_device_capability(),
            quant_config=config.quant_config,
            spec_config=config.spec_config,
            is_disagg=getattr(config, "is_disagg", False),
        )


class Trace:
    """Records the criteria a routing tree evaluated, for ``explain``.

    Routing modules call ``check``/``resolve`` instead of a bare ``if`` so the
    same tree that decides can also narrate. ``NULL_TRACE`` makes both a no-op
    and is the default, so the resolve path pays nothing.
    """

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps: List[Tuple[str, Any, Any]] = []

    def check(self, label: str, value: Any, ok: bool) -> bool:
        """Record a pass/fail criterion and return it unchanged."""
        self.steps.append((label, value, ok or None))
        return ok

    def resolve(self, label: str, value: Any, outcome: Any) -> Any:
        """Record a criterion that names something, and return that name."""
        self.steps.append((label, value, outcome))
        return outcome


class _NullTrace(Trace):
    __slots__ = ()

    def __init__(self) -> None:  # no list to append to
        pass

    def check(self, label: str, value: Any, ok: bool) -> bool:
        return ok

    def resolve(self, label: str, value: Any, outcome: Any) -> Any:
        return outcome


NULL_TRACE = _NullTrace()


def routing_module(arch: str):
    """Import and return the routing module for ``arch``, or None."""
    name = STAIRCASE_ROUTERS.get(arch)
    if name is None:
        return None
    return import_module(f"{_PACKAGE}.{name}")


def staircase_resolve(config: "ModelConfig") -> Optional[str]:
    """Rewrite ``architectures[0]`` into a staircase target's class name.

    Driven by ``TRTLLM_STAIRCASE``; see ``STAIRCASE_ENV`` for why it is an
    environment variable and when it has to be set.

    Returns None when staircase is off, when no routing module claims the
    architecture, or when the routing tree finds no matching target -- in
    ``auto`` the caller then falls back to the built-in implementation. In
    ``require`` a non-match raises instead, because the failure this mode
    exists to prevent is silent: asking for a target that does not exist,
    getting the in-tree implementation, and reading the resulting curve as
    staircase's.
    """
    mode = StaircaseMode.from_env()
    if mode is StaircaseMode.OFF:
        return None

    pretrained_config = config.pretrained_config
    if not getattr(pretrained_config, "architectures", None):
        return None

    ctx = StaircaseContext.from_model_config(config)
    arch = pretrained_config.architectures[0]
    trace = Trace() if mode is StaircaseMode.REQUIRE else NULL_TRACE

    routing = routing_module(arch)
    target = None if routing is None else routing.route(ctx, trace)

    if target is None:
        if mode is StaircaseMode.REQUIRE:
            raise ValueError(explain_no_match(arch, routing, trace))
        return None

    # The synthetic name is only a registry key: importing the target module
    # is what puts the class behind it. Nothing else would -- the built-in
    # static index does not, and must not, carry staircase names.
    import_module(f"{_PACKAGE}.{routing.TARGET_MODULES[target]}")
    return target


def explain_no_match(arch: str, routing, trace: Trace) -> str:
    """Say which criterion the configuration failed, not just that it did."""
    if routing is None:
        known = ", ".join(sorted(STAIRCASE_ROUTERS)) or "(none)"
        return (
            f"staircase is set to 'require' but no staircase target "
            f"exists for architecture {arch!r}; routed architectures: "
            f"{known}"
        )
    lines = [
        f"staircase is set to 'require' but no target matched "
        f"architecture {arch!r}. The decision tree in "
        f"{routing.__name__.rpartition('.')[0].replace('.', '/')}/routing.py "
        f"got as far as:"
    ]
    for label, value, outcome in trace.steps:
        mark = "no match" if outcome is None else f"-> {outcome}"
        lines.append(f"  {label:<10} {value!r:<48} {mark}")
    if not trace.steps:
        lines.append("  (the tree rejected the configuration before its first recorded criterion)")
    return "\n".join(lines)
