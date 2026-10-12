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
* The routing module's ``route`` owns the rest of the decision as one
  forward-reading tree over the facts that fix which code runs: the
  checkpoint's config, the parallel mapping, the GPU's compute capability and
  the speculative-decoding config. It returns a target name, or raises
  :class:`NoTarget` with the reason in words. Reading that one file tells you
  where any configuration lands.

Tuning knobs -- ``max_batch_size``, ``max_num_tokens``, CUDA graph settings --
are deliberately not routing inputs. They do not change which code runs, so a
target claims a configuration regardless of them; ``require`` promises that the
target ran, not that a particular knob setting was measured.

Routing modules are imported lazily: resolving a GptOss config never imports
the DeepSeek tree, and a target's modeling code is imported only once its
routing module has named it.
"""

from __future__ import annotations

import enum
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


def modeling_v2_resolve(config: "ModelConfig", llm_args: "TorchLlmArgs") -> Optional[str]:
    """Decide which modeling_v2 target, if any, this deployment builds.

    Returns the target's class name, or None for the built-in implementation:
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
        target = routing.route(
            pretrained_config, config.mapping, _device_compute_capability(), config.spec_config
        )
    except NoTarget as why:
        if mode is ModelingV2Mode.REQUIRE:
            raise ValueError(f"modeling_v2='require', but no target fits {arch!r}: {why}") from None
        logger.info(
            f"modeling_v2: no target fits {arch!r} ({why}); using the built-in implementation"
        )
        return None

    # The synthetic name is only a registry key: importing the target module
    # is what puts the class behind it. Nothing else would -- the built-in
    # static index does not, and must not, carry modeling_v2 names.
    import_module(f"{_PACKAGE}.{routing.TARGET_MODULES[target]}")
    return target
