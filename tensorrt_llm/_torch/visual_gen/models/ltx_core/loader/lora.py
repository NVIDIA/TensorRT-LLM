"""Bind checkpoint LoRA factors and patch selected model units."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.devices import allow_async_transfer
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.placement import Placement
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import (
    MPS_HOST_FUSED_FP8,
    FuseRule,
    LoadMapping,
    LoRAFactorKeys,
    LoRAUnitSpec,
    MetadataUnit,
    PatchedUnitSpec,
    TensorMetadata,
    TensorsUnit,
    factor_pairs,
)


@dataclass(frozen=True, slots=True)
class LoRAFuser:
    """Patch model units one weight at a time from supplied LoRA factors.
    ``apply_device`` selects where each A/B product is computed; ``None`` uses
    the model weight's device. When the retention policy does not keep a factor
    unit, the state loader uses ``unretained_factor_placement`` to load it for
    one build. The patched spec supplies the fusion rule.
    """

    apply_device: torch.device | None = None
    unretained_factor_placement: Placement = field(
        default_factory=lambda: Placement(torch.device("cpu"), pin_memory=True)
    )

    def patch(
        self, spec: PatchedUnitSpec, clean: Mapping[str, torch.Tensor], factors: Sequence[TensorsUnit]
    ) -> dict[str, torch.Tensor]:
        """Return the tensors the unit's LoRAs rewrite: each bound weight and the companions its rule emits.
        ``clean`` holds the unit's clean tensors, at least its bound weights and their companions, and is not
        modified. ``factors`` supplies one tensor unit per LoRA, in the spec's order. Results are new tensors
        with the dtypes of the clean tensors they replace.
        """
        with torch.no_grad():
            return {name: value.to(dtype=clean[name].dtype) for name, value in self._fused(spec, clean, factors)}

    def patch_into(
        self,
        spec: PatchedUnitSpec,
        clean: Mapping[str, torch.Tensor],
        factors: Sequence[TensorsUnit],
        dst: Mapping[str, torch.Tensor],
    ) -> frozenset[str]:
        """Write the tensors the unit's LoRAs rewrite into the storage of ``dst``; return their keys.
        Each fused weight is copied as soon as it is computed, so at most one rewritten tensor and its companions
        are alive at a time, whatever the unit size. Other keys of ``dst`` are left as they are. ``clean`` may be
        ``dst`` itself.
        """
        written: set[str] = set()
        with torch.no_grad():
            for name, value in self._fused(spec, clean, factors):
                if dst[name] is not value:
                    dst[name].copy_(value)
                written.add(name)
        return frozenset(written)

    def application_device(self, weight_device: torch.device) -> torch.device:
        """Device on which deltas for weights living on ``weight_device`` are computed."""
        return torch.device(weight_device) if self.apply_device is None else torch.device(self.apply_device)

    def factor_loading_placement(self, weight_device: torch.device) -> Placement:
        """Choose temporary factor storage when the retention policy keeps no factor state."""
        if self.unretained_factor_placement.pin_memory and self.application_device(weight_device).type != "cuda":
            return Placement(torch.device("cpu"))
        return self.unretained_factor_placement

    def _fused(
        self, spec: PatchedUnitSpec, clean: Mapping[str, torch.Tensor], factors: Sequence[TensorsUnit]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield each rewritten tensor as its weight is fused, one bound weight at a time."""
        if len(spec.loras) != len(factors):
            raise ValueError("LoRA selections and factor units differ in count")
        selected = spec.clean.unit.metadata
        rule = spec.fuse_rule
        # Only weights some LoRA binds: a key intersection, not a pass over the whole unit.
        targets = set().union(*(lora.bindings.keys() & selected.keys() for lora in spec.loras))
        for key in targets:
            delta = self._delta(key, clean[key], rule, spec.loras, factors)
            if delta is None:
                continue
            emitted = rule(key, clean[key], delta, clean)
            del delta
            companions = {f"{key.removesuffix('.weight')}{suffix}" for suffix in rule.requires_companions}
            if stray := [name for name in emitted if name != key and name not in companions]:
                # Footprint swaps restore and exchange only bound weights and declared companions.
                raise ValueError(f"fuse rule {rule.name!r} wrote undeclared keys {stray}")
            yield from emitted.items()
            del emitted

    def _delta(
        self,
        key: str,
        weight: torch.Tensor,
        rule: FuseRule,
        adapters: Sequence[LoRAUnitSpec],
        factors: Sequence[TensorsUnit],
    ) -> torch.Tensor | None:
        delta: torch.Tensor | None = None
        for adapter, factor_unit in zip(adapters, factors, strict=True):
            values = factor_unit.tensors
            pair = adapter.bindings.get(key)
            if pair is None:
                continue
            a, b = values[pair.a], values[pair.b]
            apply = self.application_device(weight.device)
            aggregate = rule.dtype_for(apply)
            a = a.to(device=apply, dtype=aggregate, non_blocking=allow_async_transfer(apply))
            b = b.to(device=apply, dtype=aggregate, non_blocking=allow_async_transfer(apply))
            product = (b * adapter.strength) @ a
            delta = product if delta is None else delta + product
        if delta is None or delta.device == weight.device:
            return delta
        return delta.to(weight.device, non_blocking=allow_async_transfer(weight.device))


def bind_lora_factors(
    load_mapping: LoadMapping,
    model_metadata: Mapping[str, TensorMetadata],
) -> dict[str, LoRAFactorKeys]:
    """Select checkpoint LoRA A/B pairs targeting model weights."""
    result: dict[str, LoRAFactorKeys] = {}
    factors = load_mapping.tensor_mapping.destination_metadata
    for target, pair in factor_pairs(factors).items():
        if target not in model_metadata:
            continue
        result[target] = pair
    return result


def whole_adapter_spec(
    load_mapping: LoadMapping,
    bindings: Mapping[str, LoRAFactorKeys],
    unit_id: str,
    *,
    strength: float = 1.0,
) -> LoRAUnitSpec:
    """Select applicable LoRA factors as one unit sharing the full checkpoint recipe."""
    keys = dict.fromkeys(key for pair in bindings.values() for key in pair)
    if not keys:
        raise ValueError("adapter has no factor pairs for selected model weights")
    metadata = load_mapping.tensor_mapping.destination_metadata
    selected = MetadataUnit(unit_id, {key: metadata[key] for key in keys})
    return LoRAUnitSpec(selected, load_mapping, strength)


def patched_keys(spec: PatchedUnitSpec) -> frozenset[str]:
    """Keys of the unit that ``spec``'s adapters can rewrite: bound weights and their rule companions."""
    metadata = spec.clean.unit.metadata
    targets = set().union(*(adapter.bindings.keys() & metadata.keys() for adapter in spec.loras))
    companions = {
        name
        for key in targets
        for suffix in spec.fuse_rule.requires_companions
        if (name := f"{key.removesuffix('.weight')}{suffix}") in metadata
    }
    return frozenset(targets | companions)


def _float_patch(
    key: str, weight: torch.Tensor, delta: torch.Tensor, _clean: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if not weight.is_floating_point() or weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"{key!r} needs an explicit quantization patch rule and companions")
    return {key: weight.to(delta.dtype) + delta}


def scaled_fp8_patch(
    key: str, weight: torch.Tensor, delta: torch.Tensor, clean: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    """Fuse into a per-tensor scaled FP8 weight and its companion scale."""
    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return _float_patch(key, weight, delta, clean)
    scale_key = f"{key.removesuffix('.weight')}.weight_scale"
    scale = clean[scale_key]
    combined = weight.float() * scale.float() + delta.float()
    limit = torch.finfo(weight.dtype).max
    new_scale = combined.abs().amax() / limit
    new_scale = torch.where(new_scale == 0, torch.ones_like(new_scale), new_scale)
    quantized = (combined / new_scale).clamp(-limit, limit).to(weight.dtype)
    return {key: quantized, scale_key: new_scale.reshape(scale.shape).to(scale.dtype)}


FLOAT_FUSE_RULE = FuseRule("bf16-add", torch.bfloat16, _float_patch)


SCALED_FP8_FUSE_RULE = FuseRule(
    "scaled-fp8", torch.float32, scaled_fp8_patch, (".weight_scale",), host_fused_dtypes=MPS_HOST_FUSED_FP8
)
