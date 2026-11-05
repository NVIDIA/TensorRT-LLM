# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture name -> ``route``, and the resolver that consults it.

The checkpoint's own ``architectures[0]`` (``GptOssForCausalLM``,
``DeepseekV3ForCausalLM``) selects a family's ``route``, never a target: one
architecture has many deployments and a target serves exactly one of them.
``modeling_v2_resolve`` returns the target *class*, the model loader records it
on ``ModelConfig.modeling_v2_target``, and ``AutoModelForCausalLM._resolve_class``
builds that class instead of the registry's.

Two levels, on purpose:

* Each family's ``routing.py`` declares the ``architectures[0]`` values its
  ``route`` serves with ``@routes(...)``, and ``models/__init__.py`` imports
  every family's ``routing.py``, which is what registers them. The names are
  things that already exist in every checkpoint's config.json, so adding a
  new checkpoint, GPU arch or parallel topology never touches either -- only
  a new architecture family does.
* That ``route`` owns the rest of the decision as one
  forward-reading tree over the facts that fix which code runs. It receives
  the ``ModelConfig``, the LLM arguments and the GPU's compute capability, and
  reads from the first two the checkpoint's config, the parallel mapping, the
  speculative-decoding config, the KV-cache algorithm the engine will run with
  (:func:`engine_kv_cache_quant_algo`) and any other engine choice its targets
  depend on. It returns the target class, or raises :class:`NoTarget` with the
  reason in words. Reading that one file tells you where any configuration
  lands.

Tuning knobs -- ``max_batch_size``, ``max_num_tokens``, CUDA graph settings --
are deliberately not routing inputs. They do not change which code runs, so a
target claims a configuration regardless of them; ``require`` promises that the
target ran, not that a particular knob setting was measured.

Routing modules are imported on the first resolve, all of them, and hold no
modeling code: a target's modeling code is imported only on the ``route``
branch that returns it, so resolving a GptOss config never imports a DeepSeek
target.
"""

from __future__ import annotations

import enum
import os
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# architectures[0] -> the ``route`` that decides where it lands, filled by ``@routes``.
_ROUTES: Dict[str, Callable[..., type]] = {}


def routes(*architectures: str):
    """Declare that the decorated ``route`` decides where these ``architectures[0]`` values land."""

    def register(route):
        for arch in architectures:
            assert arch not in _ROUTES, (
                f"{arch} routed by both {_ROUTES[arch].__module__} and {route.__module__}"
            )
            _ROUTES[arch] = route
        return route

    return register


def registered_routes() -> Dict[str, Callable[..., type]]:
    """architectures[0] -> ``route``, for every family ``models/__init__.py`` imports."""
    from . import models  # noqa: F401

    return _ROUTES


#: Turns the step contract check on. Off by default, so a served engine runs
#: the forward and nothing else: the checks describe facts fixed at engine
#: construction, and a check that can only ever hold is a check whose failure
#: mode is its own bug taking down a working deployment.
#:
#: An environment variable rather than an LLM arg while modeling_v2 is
#: experimental; it has to reach the ranks, so a test passes it through
#: ``env_overrides``. When this stops being experimental it belongs in the
#: LLM args, where a caller can set it per engine rather than per process.
MODELING_V2_VALIDATE_ENV = "TRTLLM_MODELING_V2_VALIDATE"


def step_contract_enabled() -> bool:
    """Whether targets should check their step contract on the first forward.

    Read once per model instance at construction rather than per forward: the
    answer cannot change within a process, and the forward is the hot path.
    """
    return os.environ.get(MODELING_V2_VALIDATE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class ModelingV2Mode(str, enum.Enum):
    """What to do when a deployment reaches the modeling_v2 resolver."""

    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"

    @classmethod
    def of(cls, llm_args: "TorchLlmArgs") -> "ModelingV2Mode":
        return cls(llm_args.modeling_v2)


class NoTarget(ValueError):
    """Raised by a family's ``route`` when no target fits, or by a target at construction when the runtime
    the engine arguments set is not the one it is written for.

    The message says, in words, which fact ruled the targets out. Under
    ``auto`` the resolver logs it and falls back to the built-in implementation
    (for a refusal at construction, ``AutoModelForCausalLM.from_config`` does);
    under ``require`` it becomes the error the user sees.
    """


# The mode each target class was last resolved under (the loader builds the class right after resolving it).
_RESOLVED_MODES: Dict[type, ModelingV2Mode] = {}


def resolved_mode(target: type) -> ModelingV2Mode:
    """The mode ``modeling_v2_resolve`` returned ``target`` under; ``auto`` for a class built without it.

    A target reads it at construction: under ``require`` its in-target dispatch
    raises where it would otherwise fall back to a built-in path inside the
    shapes it claims.
    """
    return _RESOLVED_MODES.get(target, ModelingV2Mode.AUTO)


def tp_reaches_across_nodes(mapping) -> bool:
    """Whether an attention-TP allreduce over ``mapping`` can reach every rank
    through the workspaces the default targets build.

    Inside one node the IPC workspace (``comm/allreduce``) serves it. Across
    nodes the targets take the MNNVL workspace (``comm/mnnvl_allreduce``), which
    needs one NVLink domain over those nodes, as on GB200 and GB300 trays. This
    is the check the built-in ``AllReduce`` makes before it takes MNNVL. Where it
    fails, the built-in model falls back to NCCL, which the targets do not
    implement, so a routing module refuses and ``auto`` keeps the built-in model.
    Attention DP reduces nothing in its norm slots, and its allgather and
    reduce-scatter run over NCCL.
    """
    if mapping.enable_attention_dp or mapping.tp_size == 1 or not mapping.is_multi_node():
        return True
    import torch

    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    # bf16 is the activation dtype of both default targets.
    return MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16)


def _device_compute_capability() -> Tuple[int, int]:
    import torch

    assert torch.cuda.is_available(), (
        "modeling_v2 routes on the compute capability of the device it will run on; "
        "no CUDA device is visible"
    )
    return torch.cuda.get_device_capability()


def engine_kv_cache_quant_algo(config: "ModelConfig", llm_args: "TorchLlmArgs"):
    """The KV-cache algorithm the engine will run with, which routing runs too early to read.

    The loader resolves the target before `validate_and_set_kv_cache_quant`
    applies `kv_cache_config.dtype` to the config, so this applies the same
    rule here: ``auto`` keeps the checkpoint's algorithm, ``fp8`` and
    ``nvfp4`` replace it. Any other dtype comes back as the string given, so a
    route refuses it by name: ``fp8_ds_mla`` is FP8 to the loader but also
    selects a packed sparse-MLA cache layout, and an unknown dtype is one the
    loader raises on later.
    """
    from tensorrt_llm.quantization.mode import QuantAlgo

    dtype = llm_args.kv_cache_config.dtype
    if dtype == "auto":
        algo = config.quant_config.kv_cache_quant_algo
        return None if algo is None else QuantAlgo(algo)
    return {"fp8": QuantAlgo.FP8, "nvfp4": QuantAlgo.NVFP4}.get(dtype, dtype)


def modeling_v2_resolve(config: "ModelConfig", llm_args: "TorchLlmArgs") -> Optional[type]:
    """Decide which modeling_v2 target, if any, this deployment builds.

    Returns the target class, or None for the built-in implementation:
    when ``llm_args.modeling_v2`` is ``off``, when no ``route`` is registered
    for the architecture, or when that ``route`` raises ``NoTarget``.
    In ``require`` every one of those but ``off`` raises ``ValueError`` instead.

    """
    mode = ModelingV2Mode.of(llm_args)
    if mode is ModelingV2Mode.OFF:
        return None

    pretrained_config = config.pretrained_config
    if not getattr(pretrained_config, "architectures", None):
        return None
    arch = pretrained_config.architectures[0]

    try:
        route = registered_routes().get(arch)
        if route is None:
            known = ", ".join(sorted(registered_routes())) or "(none)"
            raise NoTarget(f"no route for this architecture; routed architectures: {known}")
        target = route(config, llm_args, _device_compute_capability())
    except NoTarget as why:
        if mode is ModelingV2Mode.REQUIRE:
            raise ValueError(f"modeling_v2='require', but no target fits {arch!r}: {why}") from None
        logger.info(
            f"modeling_v2: no target fits {arch!r} ({why}); using the built-in implementation"
        )
        return None
    if target is not None:
        _RESOLVED_MODES[target] = mode
    return target


def _qualified(target: Optional[type]) -> Optional[str]:
    return None if target is None else f"{target.__module__}.{target.__qualname__}"


def modeling_v2_resolve_on_all_ranks(
    config: "ModelConfig", llm_args: "TorchLlmArgs"
) -> Optional[type]:
    """:func:`modeling_v2_resolve`, agreed by every rank of the engine.

    Each rank routes on its own facts -- its device, and whether this
    installation built the target's native code -- so one rank can land on the
    built-in model while its peers build the target, and the first collective
    then hangs instead of failing. Every rank therefore reports its outcome, and
    all of them raise unless the outcomes agree. A rank's error is reported
    after the exchange, not before, so its peers are never left waiting in it.

    ``off`` returns before any exchange: nothing in this package runs then.
    """
    if ModelingV2Mode.of(llm_args) is ModelingV2Mode.OFF:
        return None

    target = error = None
    try:
        target = modeling_v2_resolve(config, llm_args)
    except Exception as caught:  # noqa: BLE001 -- re-raised below, after every rank has reported
        error = caught
    outcome = (_qualified(target), None if error is None else f"{type(error).__name__}: {error}")

    mapping = config.mapping
    outcomes = [outcome]
    if mapping.world_size > 1:
        from tensorrt_llm._torch.distributed.communicator import Distributed

        outcomes = Distributed.get(mapping).allgather(outcome)

    if error is not None:
        raise error
    failed = {rank: why for rank, (_, why) in enumerate(outcomes) if why is not None}
    if failed:
        detail = "; ".join(f"rank {rank}: {why}" for rank, why in sorted(failed.items()))
        raise RuntimeError(f"modeling_v2: other ranks failed to resolve a target ({detail})")
    if len({built for built, _ in outcomes}) > 1:
        detail = "; ".join(
            f"rank {rank}: {built or 'built-in'}" for rank, (built, _) in enumerate(outcomes)
        )
        raise RuntimeError(
            f"modeling_v2: ranks disagree on the implementation to build ({detail}); "
            "every rank has to run the same code"
        )
    if target is not None and mapping.rank == 0:
        logger.warning(f"modeling_v2: building {_qualified(target)} on all {len(outcomes)} rank(s)")
    return target
