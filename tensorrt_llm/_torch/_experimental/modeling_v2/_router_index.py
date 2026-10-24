# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture name -> routing module, and the resolver that reads the table.

The checkpoint's own ``architectures[0]`` (``GptOssForCausalLM``,
``DeepseekV3ForCausalLM``) selects a routing module, never a target: one
architecture has many deployments and a target serves exactly one of them.
``modeling_v2_resolve`` returns the target *class*, the model loader records it
on ``ModelConfig.modeling_v2_target``, and ``AutoModelForCausalLM._resolve_class``
builds that class instead of the registry's.

Two levels, on purpose:

* This table maps ``architectures[0]`` -> routing module. Its keys are things
  that already exist in every checkpoint's config.json, so adding a new
  checkpoint, GPU arch or parallel topology never touches it -- only a new
  architecture family does.
* The routing module's ``route`` owns the rest of the decision as one
  forward-reading tree over the facts that fix which code runs: the
  checkpoint's config, the parallel mapping, the GPU's compute capability, the
  speculative-decoding config and the KV-cache algorithm the engine will run
  with. It also receives the ``ModelConfig`` and the LLM arguments, from which a
  family derives any other engine choice its targets depend on. It returns the
  target class, or raises :class:`NoTarget` with the reason in words. Reading
  that one file tells you where any configuration lands.

Tuning knobs -- ``max_batch_size``, ``max_num_tokens``, CUDA graph settings --
are deliberately not routing inputs. They do not change which code runs, so a
target claims a configuration regardless of them; ``require`` promises that the
target ran, not that a particular knob setting was measured.

Routing modules are imported lazily: resolving a GptOss config never imports
the DeepSeek tree, and a target's modeling code is imported only on the
``route`` branch that returns it.
"""

from __future__ import annotations

import enum
import os
from importlib import import_module
from typing import TYPE_CHECKING, Optional, Tuple

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"

# architectures[0] -> routing module, relative to this package.
MODELING_V2_ROUTERS = {
    "GptOssForCausalLM": "models.gpt_oss.routing",
    "DeepseekV3ForCausalLM": "models.deepseek_v3.routing",
    "KimiK3ForConditionalGeneration": "models.kimi_k3_vl.routing",
    # Qwen3.8-Flash-Next publishes ``Qwen4ExpForConditionalGeneration``, and the
    # config loader's composite branch writes the same name back.
    "Qwen4ExpForConditionalGeneration": "models.qwen4_exp.routing",
    "GlmMoeDsaForCausalLM": "models.glm_moe_dsa.routing",
}


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


class NoTarget(Exception):
    """Raised by a routing module's ``route`` when no target fits.

    The message says, in words, which fact ruled the targets out. Under
    ``auto`` the resolver logs it and falls back to the built-in implementation;
    under ``require`` it becomes the error the user sees.
    """


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


def routing_module(arch: str):
    """Import and return the routing module for ``arch``, or None."""
    name = MODELING_V2_ROUTERS.get(arch)
    if name is None:
        return None
    return import_module(f"{_PACKAGE}.{name}")


def _device_compute_capability() -> Tuple[int, int]:
    import torch

    assert torch.cuda.is_available(), (
        "modeling_v2 routes on the compute capability of the device it will run on; "
        "no CUDA device is visible"
    )
    return torch.cuda.get_device_capability()


def _kv_cache_quant_algo(config: "ModelConfig", llm_args: "TorchLlmArgs"):
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
    when ``llm_args.modeling_v2`` is ``off``, when no routing module claims the
    architecture, or when the routing module's ``route`` raises ``NoTarget``.
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
        routing = routing_module(arch)
        if routing is None:
            known = ", ".join(sorted(MODELING_V2_ROUTERS)) or "(none)"
            raise NoTarget(
                f"no routing module for this architecture; routed architectures: {known}"
            )
        return routing.route(
            pretrained_config,
            config.mapping,
            _device_compute_capability(),
            config.spec_config,
            _kv_cache_quant_algo(config, llm_args),
            model_config=config,
            llm_args=llm_args,
        )
    except NoTarget as why:
        if mode is ModelingV2Mode.REQUIRE:
            raise ValueError(f"modeling_v2='require', but no target fits {arch!r}: {why}") from None
        logger.info(
            f"modeling_v2: no target fits {arch!r} ({why}); using the built-in implementation"
        )
        return None
