from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeVar

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.sd_ops import SDOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.simple_builder import SimpleBuilder
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import FuseRule

_Model = TypeVar("_Model", bound=torch.nn.Module)


@dataclass(frozen=True, slots=True)
class QuantizationPolicy:
    """Checkpoint stages, shell rewrites, and LoRA fusion for model loading."""

    sd_ops: tuple[SDOps, ...]
    fuse_rule: FuseRule
    module_ops: tuple[ModuleOps, ...] = ()


def compose_quantization(builder: SimpleBuilder[_Model], policy: QuantizationPolicy) -> SimpleBuilder[_Model]:
    """Append the policy's checkpoint stages and module operations and use its fuse rule.
    Compose before compilation: compiled blocks no longer match the policy's module operations.
    """
    return replace(
        builder,
        sd_ops=(*builder.sd_ops, *policy.sd_ops),
        module_ops=(*builder.module_ops, *policy.module_ops),
        fuse_rule=policy.fuse_rule,
    )
