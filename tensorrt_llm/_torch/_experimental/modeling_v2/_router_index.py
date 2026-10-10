# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture name -> routing module, and the resolver that reads the table.

ModelingV2 targets are keyed by a *synthetic* architecture name that no
checkpoint declares. The checkpoint's own ``architectures[0]`` stays what it
always was (``GptOssForCausalLM``, ``DeepseekV3ForCausalLM``), so it cannot
also be the target selector; ``modeling_v2_resolve`` decides on the synthetic
name, the model loader records it on ``ModelConfig.modeling_v2_target``, and
``AutoModelForCausalLM._resolve_class`` then looks that up in the ordinary
registry.

The pattern is upstream's own: the Eagle3 rewrite in ``_resolve_class`` builds
``EAGLE3<Arch>`` the same way -- a name no config.json declares, reached only
through that rewrite.

Two levels, on purpose:

* This table maps ``architectures[0]`` -> routing module. Its keys are things
  that already exist in every checkpoint's config.json, so adding a new
  checkpoint, GPU arch or parallel topology never touches it -- only a new
  architecture family does.
* The routing module owns the rest of the decision as one forward-reading
  tree, in two stages. ``route`` reads the configuration's *identity* --
  checkpoint shape, GPU architecture, parallel topology -- and names a target.
  ``within_bounds`` then reads the *deployment* -- the LLM API arguments as
  the engine will run with them, model defaults included -- and says whether
  that target was certified for it.
  Reading that single file tells you where any configuration lands, and
  ``explain.py`` replays both stages to say *why*.

Routing modules are imported lazily: resolving a GptOss config never imports
the DeepSeek tree, and a target's modeling code is imported only once its
routing module has claimed the config and accepted the deployment.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"

#: The switch: ``TorchLlmArgs.modeling_v2``, an LLM API argument.
#:
#: An argument rather than an environment variable because the arguments are
#: what every rank is handed. A variable assigned from a script after MPI
#: initialized reached the driver and not the worker ranks, and a driver that
#: resolves a modeling_v2 target while its workers resolve the built-in is
#: exactly the silent split this package exists to prevent.
MODELING_V2_ARG = "modeling_v2"

# architectures[0] -> routing module, relative to this package.
MODELING_V2_ROUTERS = {
    "GptOssForCausalLM": "models.gpt_oss.routing",
    "DeepseekV3ForCausalLM": "models.deepseek_v3.routing",
}


class ModelingV2Mode(str, enum.Enum):
    """What to do when a deployment reaches the modeling_v2 resolver.

    Pydantic admits only these three values for ``TorchLlmArgs.modeling_v2``,
    so a typo is refused when the arguments are built rather than read as
    "off". Silently reading it as "off" would hand back the built-in
    implementation while the caller believed they had asked for a modeling_v2
    target, and a number measured that way is attributed to the wrong system
    -- the failure the ``require`` mode below exists to prevent, arriving
    through the door instead of the window.
    """

    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"

    @classmethod
    def of(cls, llm_args: "TorchLlmArgs") -> "ModelingV2Mode":
        return cls(getattr(llm_args, MODELING_V2_ARG))


@dataclass(frozen=True)
class ModelingV2Context:
    """Everything the identity stage of a routing decision may depend on.

    The admission rule is one line: a quantity may live here only if it is
    already known when the model class is resolved *and* does not change for
    the rest of the engine's life. Config shape, SM version, the parallel
    mapping, the quant and speculative configs and the disagg flag all
    qualify. Batch composition, ``num_contexts``, "is this step pure decode"
    do not -- they move every forward, and a target selected from them would
    be selected once and then be wrong.

    That is a necessary condition, not a sufficient one. ``max_num_tokens``
    and ``cuda_graph_config.max_batch_size`` are per-instance constants too,
    but they are tuning knobs: they are LLM API arguments, not identity. Only
    an instance constant that changes the *structure of the forward* earns a
    target of its own. Knobs belong to the second stage: a target's
    ``within_bounds`` reads the LLM API arguments directly and says whether
    the deployment is one it was certified for.

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
    def from_model_config(
        cls, config: "ModelConfig", sm: Optional[Tuple[int, int]] = None
    ) -> "ModelingV2Context":
        """Build the context the identity stage routes on.

        ``sm`` defaults to the device this process will run on, which is what
        the engine wants. ``explain`` passes it explicitly so a configuration
        can be explained from a host with no GPU -- every other field it reads
        off the same ``ModelConfig`` the engine built, rather than restating
        one, so the two cannot disagree about what a checkpoint is.
        """
        if sm is None:
            import torch

            assert torch.cuda.is_available(), (
                "modeling_v2 routes on the SM version of the device it will run on; "
                "no CUDA device is visible"
            )
            sm = torch.cuda.get_device_capability()
        return cls(
            pretrained_config=config.pretrained_config,
            mapping=config.mapping,
            sm=sm,
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
    name = MODELING_V2_ROUTERS.get(arch)
    if name is None:
        return None
    return import_module(f"{_PACKAGE}.{name}")


def modeling_v2_resolve(config: "ModelConfig", llm_args: "TorchLlmArgs") -> Optional[str]:
    """Decide which modeling_v2 target, if any, this deployment builds.

    Returns the target's class name -- a synthetic architecture name no
    checkpoint declares -- or None for the built-in implementation: when
    ``llm_args.modeling_v2`` is ``off``, when no routing module claims the
    architecture, when the identity tree names no target, or when the target
    it names says the deployment is outside its bounds. In ``auto`` the caller
    then builds the built-in implementation. In ``require`` every one of those
    but ``off`` raises instead, because the failure that mode exists to
    prevent is silent: asking for a target, getting the in-tree
    implementation, and reading the resulting curve as modeling_v2's.

    The model loader calls this when it is about to build the model, after
    the built-in model's defaults have been applied to ``llm_args``: a
    target's ``within_bounds`` therefore judges the deployment as it will
    actually run, model defaults included.
    """
    mode = ModelingV2Mode.of(llm_args)
    if mode is ModelingV2Mode.OFF:
        return None

    pretrained_config = config.pretrained_config
    if not getattr(pretrained_config, "architectures", None):
        return None

    ctx = ModelingV2Context.from_model_config(config)
    arch = pretrained_config.architectures[0]
    trace = Trace() if mode is ModelingV2Mode.REQUIRE else NULL_TRACE

    routing = routing_module(arch)
    target = None if routing is None else routing.route(ctx, trace)
    accepted = target is not None and routing.within_bounds(target, llm_args, ctx, trace)

    if not accepted:
        if mode is ModelingV2Mode.REQUIRE:
            raise ValueError(explain_no_match(arch, routing, trace, target))
        if target is not None:
            logger.info(
                f"modeling_v2: {target} claims {arch} but this deployment is outside "
                f"its bounds; using the built-in implementation"
            )
        return None

    # The synthetic name is only a registry key: importing the target module
    # is what puts the class behind it. Nothing else would -- the built-in
    # static index does not, and must not, carry modeling_v2 names.
    import_module(f"{_PACKAGE}.{routing.TARGET_MODULES[target]}")
    return target


def explain_no_match(arch: str, routing, trace: Trace, target: Optional[str] = None) -> str:
    """Say which criterion the configuration failed, not just that it did.

    ``target`` is the name the identity stage selected when the deployment
    then fell outside that target's bounds; None when identity itself found
    nothing.
    """
    asked = f"{MODELING_V2_ARG}={ModelingV2Mode.REQUIRE.value!r}"
    if routing is None:
        known = ", ".join(sorted(MODELING_V2_ROUTERS)) or "(none)"
        return f"{asked}, but no target exists for architecture {arch!r}; routed architectures: {known}"
    where = routing.__name__.rpartition(".")[0].replace(".", "/") + "/routing.py"
    if target is None:
        lines = [
            f"{asked}, but no target matched architecture {arch!r}. "
            f"The decision tree in {where} got as far as:"
        ]
    else:
        lines = [
            f"{asked}, but {target} claims architecture {arch!r} and this deployment "
            f"is outside its bounds. The decision tree in {where} got as far as:"
        ]
    for label, value, outcome in trace.steps:
        mark = "no match" if outcome is None else f"-> {outcome}"
        lines.append(f"  {label:<10} {value!r:<48} {mark}")
    if not trace.steps:
        lines.append("  (the tree rejected the configuration before its first recorded criterion)")
    return "\n".join(lines)
