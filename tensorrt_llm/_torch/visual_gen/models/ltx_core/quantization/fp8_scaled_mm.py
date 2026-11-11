from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.checkpoint import CheckpointReader
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import FuseRule
from tensorrt_llm._torch.visual_gen.models.ltx_core.quantization.policy import QuantizationPolicy


class FP8Linear(nn.Module):
    """Linear layer with FP8 weight storage for scaled matrix multiplication."""

    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn, device=device))
        self.weight_scale = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
        self.input_scale = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape

        # Clamp before cast: out-of-range values cast to NaN/saturated FP8, which
        # produces black-screen output on some checkpoints (e.g. ltx-2-19b-dev-fp8).
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        qinput = torch.clamp(x * self.input_scale.reciprocal(), fp8_min, fp8_max).to(torch.float8_e4m3fn)
        if qinput.dim() == 3:
            qinput = qinput.reshape(-1, qinput.shape[-1])
        output = torch._scaled_mm(
            qinput,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )

        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        return output


def quantize_weight_to_fp8_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to ``float8_e4m3fn`` with a per-tensor scale."""
    weight_fp32 = weight.to(torch.float32)

    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    max_abs = torch.amax(torch.abs(weight_fp32))
    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), fp8_max / max_abs)

    @torch.compiler.disable
    def _quantize(
        weight_fp32: torch.Tensor, scale: torch.Tensor, fp8_min: torch.Tensor, fp8_max: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_weight = torch.clamp(weight_fp32 * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        weight_scale = scale.reciprocal()
        return quantized_weight, weight_scale

    quantized_weight, weight_scale = _quantize(weight_fp32, scale, fp8_min, fp8_max)
    return quantized_weight, weight_scale


def _linear_to_fp8linear(layer: nn.Linear) -> FP8Linear:
    """Create an ``FP8Linear`` matching the shape/bias of *layer*."""
    return FP8Linear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
    )


def _swap_linears_to_fp8(model: nn.Module, should_swap: Callable[[str], bool]) -> nn.Module:
    """Replace nn.Linear layers with FP8Linear where ``should_swap(name)`` returns True."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
            continue
        if not should_swap(name):
            continue

        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name

        replacements.append((parent, attr_name, module))

    for parent, attr_name, linear in replacements:
        setattr(parent, attr_name, _linear_to_fp8linear(linear))

    return model


def _fuse_scaled_weight(
    key: str, weight: torch.Tensor, delta: torch.Tensor, scale: torch.Tensor
) -> dict[str, torch.Tensor]:
    scale_key = f"{key.removesuffix('.weight')}.weight_scale"
    decoded = weight.float() * scale.float()
    payload, new_scale = quantize_weight_to_fp8_per_tensor(decoded + delta.float())
    return {key: payload, scale_key: new_scale.reshape(scale.shape).to(scale.dtype)}


def _fp8_scaled_unit_patch(
    key: str, weight: torch.Tensor, delta: torch.Tensor, clean: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if weight.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        if not weight.is_floating_point():
            raise ValueError(f"{key!r} is not a floating fallback weight")
        return {key: (weight.to(delta.dtype) + delta).to(weight.dtype)}
    scale_key = f"{key.removesuffix('.weight')}.weight_scale"
    return _fuse_scaled_weight(key, weight, delta, clean[scale_key])


FP8_SCALED_UNIT_FUSE_RULE = FuseRule("fp8-scaled-mm", torch.bfloat16, _fp8_scaled_unit_patch, (".weight_scale",))


def selected_fp8_layers(paths: Sequence[Path]) -> frozenset[str]:
    """Discover supported FP8/scale pairs from safetensors headers alone."""
    with CheckpointReader.open(paths) as reader:
        _, metadata = reader.scan()
    selected = frozenset(
        key.removesuffix(".weight_scale")
        for key in metadata
        if key.endswith(".weight_scale")
        and metadata.get(f"{key.removesuffix('.weight_scale')}.weight", (None, None))[1] == torch.float8_e4m3fn
    )
    if not selected:
        raise ValueError("scaled FP8 requires at least one F8_E4M3 weight with a weight_scale companion")
    return selected


def swap_module_op(paths: Sequence[Path]) -> ModuleOps:
    """Replace only linears whose checkpoint headers select scaled FP8 weights."""
    selected = selected_fp8_layers(paths)

    def should_swap(name: str) -> bool:
        return any(path == name or path.endswith(f".{name}") for path in selected)

    fingerprint = sha256("\0".join(sorted(selected)).encode()).hexdigest()[:12]
    return ModuleOps(
        f"fp8-scaled-swap:{fingerprint}",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=lambda model: _swap_linears_to_fp8(model, should_swap),
    )


def build_policy(checkpoint_path: str | Path) -> QuantizationPolicy:
    """FP8 scaled matmul for checkpoints pre-quantized with per-tensor scales.
    The set of layers to swap to ``FP8Linear`` is discovered from the
    checkpoint's ``.weight_scale`` tensors via suffix-matching against the
    model's named modules. Requires a pre-quantized checkpoint; for BF16
    checkpoints, use :func:`ltx_core.quantization.fp8_cast.build_policy`.
    """
    return QuantizationPolicy(
        sd_ops=(), module_ops=(swap_module_op((Path(checkpoint_path),)),), fuse_rule=FP8_SCALED_UNIT_FUSE_RULE
    )
