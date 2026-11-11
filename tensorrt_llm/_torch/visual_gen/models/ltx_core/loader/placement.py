"""Retention placement values and policies for unit specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import LoRAUnitSpec, PatchedUnitSpec, UnitSpec


def packs_storage(device: torch.device) -> bool:
    """Return whether whole-unit storage on ``device`` is one packed byte owner with tensor views.
    Host owners are packed: a unit then moves to a device or another rank as one byte range. Device storage gets one
    allocation per tensor, so every weight starts on an allocator block: device kernels are slower on weights at
    arbitrary offsets into a shared owner, and MPS gathers such as ``embedding`` copy the whole weight on every call
    when it is a view past offset zero.
    """
    return device.type == "cpu"


@dataclass(frozen=True, slots=True)
class Placement:
    """Device and CPU pinning for a retained state; ``None`` disables retention."""

    device: torch.device | None
    pin_memory: bool = False

    def __post_init__(self) -> None:
        """Require CPU pinning and make CUDA and MPS device indices explicit, as tensors report them."""
        if self.pin_memory and (self.device is None or self.device.type != "cpu"):
            raise ValueError("pin_memory requires a CPU placement")
        if self.device is not None and self.device.type == "cuda" and self.device.index is None:
            object.__setattr__(self, "device", torch.device("cuda", torch.cuda.current_device()))
        if self.device is not None and self.device.type == "mps" and self.device.index is None:
            object.__setattr__(self, "device", torch.device("mps", 0))


NO_RETENTION = Placement(None)


class PlacementPolicy(Protocol):
    """Return a retention placement for a unit specification."""

    def placement(self, spec: UnitSpec | PatchedUnitSpec) -> Placement: ...


@dataclass(frozen=True, slots=True)
class RetentionPolicy(PlacementPolicy):
    """Use ``clean`` for model units, ``active`` for patched units, and ``lora`` for LoRA factor units."""

    clean: Placement = NO_RETENTION
    active: Placement = NO_RETENTION
    lora: Placement = NO_RETENTION

    def placement(self, spec: UnitSpec | PatchedUnitSpec) -> Placement:
        """Choose by spec type: LoRA factor, active model or clean model."""
        if isinstance(spec, LoRAUnitSpec):
            return self.lora
        if isinstance(spec, PatchedUnitSpec):
            return self.active
        return self.clean


NO_RETENTION_POLICY = RetentionPolicy()
