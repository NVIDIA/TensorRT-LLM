import dataclasses
import functools
from collections.abc import Mapping
from pathlib import Path

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.kernels import TRITON_AVAILABLE
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.module_ops import ModuleOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.sd_ops import KeyMatching, KeyReplacement, KeyValueOperation, SDOps
from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.units import MPS_HOST_FUSED_FP8, FuseRule
from tensorrt_llm._torch.visual_gen.models.ltx_core.quantization.policy import QuantizationPolicy

BLOCK_SIZE = 1024


def fused_add_round_launch(target_weight: torch.Tensor, original_weight: torch.Tensor, seed: int) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "fused_add_round_launch requires Triton, which is not available on this platform. "
            "Callers should gate on ltx_core.loader.kernels.TRITON_AVAILABLE and use a "
            "deterministic-rounding fallback instead."
        )
    import triton  # noqa: PLC0415

    from tensorrt_llm._torch.visual_gen.models.ltx_core.loader.kernels import fused_add_round_kernel  # noqa: PLC0415

    if original_weight.dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif original_weight.dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15  # noqa: F841
    else:
        raise ValueError("Unsupported dtype")

    if target_weight.dtype != torch.bfloat16:
        raise ValueError("target_weight dtype must be bfloat16")

    # Calculate grid and block sizes
    n_elements = original_weight.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    fused_add_round_kernel[grid](
        original_weight,
        target_weight,
        seed,
        n_elements,
        exponent_bias,
        mantissa_bits,
        BLOCK_SIZE,
    )
    return target_weight


# Triton's fp8e4nv (torch.float8_e4m3fn) is only a legal kernel dtype on
# sm_89+ (Ada). Below that, tl.load of an e4m3fn tensor fails at compile with
# "type fp8e4nv not supported in this architecture". PyTorch eager conversion
# still works -- that is the software path used when Triton is missing.
_TRITON_FP8E4NV_MIN_CAPABILITY = (8, 9)


def _triton_supports_fp8e4nv(device: torch.device) -> bool:
    """Return whether Triton's fused fp8 kernel can compile for *device*.
    ``device`` is the tensor's device, not the process's current CUDA device,
    so mixed-GPU hosts pick the capability that will actually run the kernel.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) >= _TRITON_FP8E4NV_MIN_CAPABILITY


def _use_triton_fp8_kernel(device: torch.device) -> bool:
    """True when ``fused_add_round_launch`` can compile on *device*."""
    return TRITON_AVAILABLE and _triton_supports_fp8e4nv(device)


def _upcast_and_round(
    weight: torch.Tensor, dtype: torch.dtype, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.Tensor:
    """
    Upcast the weight to the given dtype and optionally apply stochastic rounding.
    Input weight needs to have float8_e4m3fn or float8_e5m2 dtype.
    Stochastic rounding is implemented via a Triton kernel. When Triton is not
    available (e.g., on Windows) or cannot compile ``fp8e4nv`` (pre-Ada GPUs),
    this falls back to deterministic (nearest) rounding via ``weight.to(dtype)``.
    MPS moves FP8 storage as bytes to CPU for conversion, then returns the
    converted tensor to MPS.
    """
    if weight.device.type == "mps":
        cpu_weight = weight.view(torch.uint8).cpu().view(weight.dtype).view(weight.shape)
        return cpu_weight.to(dtype).to(weight.device)
    if not with_stochastic_rounding or dtype != torch.bfloat16 or not _use_triton_fp8_kernel(weight.device):
        return weight.to(dtype)
    return fused_add_round_launch(torch.zeros_like(weight, dtype=dtype), weight, seed)


class Fp8CastLinear(torch.nn.Linear):
    """nn.Linear storing weights in fp8, upcasting to input dtype during forward.
    Used via __class__ reassignment (not subclassing) so existing weight tensors
    are preserved in-place. Class-level forward is required for torch.compile
    compatibility — instance-level closure monkey-patches cause graph breaks.
    """

    _with_stochastic_rounding: bool
    _seed: int

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002, type: ignore[override]
        w_up = _upcast_and_round(self.weight, input.dtype, self._with_stochastic_rounding, self._seed)
        b_up = (
            _upcast_and_round(self.bias, input.dtype, self._with_stochastic_rounding, self._seed)
            if self.bias is not None
            else None
        )
        return torch.nn.functional.linear(input, w_up, b_up)


def _replace_fwd_with_upcast(layer: torch.nn.Linear, with_stochastic_rounding: bool = False, seed: int = 0) -> None:
    """
    Intended to be applied via __class__ reassignment to existing nn.Linear
    instances. Forward remains defined at the class level, which is required for
    torch.compile compatibility — instance-level closure monkey-patches cause
    graph breaks.
    Also retypes ``weight`` and ``bias`` to fp8 so the meta param dtype matches
    the post-load tensor dtype (sd_ops downcasts checkpoint bf16 -> fp8 at load).
    Block streaming relies on this to derive pool buffer layout from the meta
    model without an eager checkpoint read.
    """
    layer.__class__ = Fp8CastLinear
    layer._with_stochastic_rounding = with_stochastic_rounding
    layer._seed = seed
    layer.weight = torch.nn.Parameter(
        torch.empty(layer.weight.shape, dtype=torch.float8_e4m3fn, device=layer.weight.device),
        requires_grad=layer.weight.requires_grad,
    )
    if layer.bias is not None:
        layer.bias = torch.nn.Parameter(
            torch.empty(layer.bias.shape, dtype=torch.float8_e4m3fn, device=layer.bias.device),
            requires_grad=layer.bias.requires_grad,
        )


# Module-name suffixes for the Linears that participate in fp8 cast. Used by
# both the upcast matcher and the sd_ops downcast map so the two cannot drift.
# - ``.to_q`` / ``.to_k`` / ``.to_v`` / ``.to_out.0`` have a leading dot so they
#   only match the attention Linears at ``...attnN.to_q`` etc.
# - ``ff.net.0.proj`` / ``ff.net.2`` are intentionally **dotless** so they match
#   both video FF (``...ff.net.0.proj``) and audio FF (``...audio_ff.net.0.proj``).
_FP8_CAST_KEY_PREFIX = "transformer_blocks."
_FP8_CAST_LINEAR_SUFFIXES: tuple[str, ...] = (
    ".to_q",
    ".to_k",
    ".to_v",
    ".to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def _is_fp8_cast_linear(module_name: str) -> bool:
    """Return True if *module_name* names a Linear that should be fp8-cast."""
    if _FP8_CAST_KEY_PREFIX not in module_name:
        return False
    return any(module_name.endswith(suffix) for suffix in _FP8_CAST_LINEAR_SUFFIXES)


def _amend_forward_with_upcast(
    model: torch.nn.Module, with_stochastic_rounding: bool = False, seed: int = 0
) -> torch.nn.Module:
    """
    Replace the forward method of the fp8-cast Linear layers (per
    :data:`_FP8_CAST_LINEAR_SUFFIXES`) to forward with upcast and optional
    stochastic rounding.
    Only the Linears whose weights are downcast by :func:`fp8_cast_downcast_stage`
    are retyped. Linears outside that subset (e.g. ``to_gate_logits``) are left as
    plain ``nn.Linear`` so the meta-model param dtype matches the loaded checkpoint
    dtype.
    """
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and _is_fp8_cast_linear(name):
            _replace_fwd_with_upcast(m, with_stochastic_rounding, seed)
    return model


def fuse_cast_fp8_weight(
    delta_bf16: torch.Tensor,
    weight_fp8: torch.Tensor,
) -> torch.Tensor:
    """Return ``(delta_bf16 + dequantize(weight_fp8)).to(weight_fp8.dtype)``.
    CUDA with Triton on sm_89+ uses stochastic rounding via the fused kernel;
    otherwise falls back to a deterministic bf16 add (no Triton, or a GPU
    whose Triton cannot compile ``fp8e4nv``). The CUDA/CPU path mutates the
    BF16 accumulator; MPS fuses on CPU and returns a new MPS tensor.
    """
    if delta_bf16.dtype != torch.bfloat16:
        raise ValueError(f"delta_bf16 must be bfloat16, got {delta_bf16.dtype}")
    if weight_fp8.device.type == "mps":
        cpu_weight = weight_fp8.view(torch.uint8).cpu().view(weight_fp8.dtype).view(weight_fp8.shape)
        cpu_result = fuse_cast_fp8_weight(delta_bf16.cpu(), cpu_weight)
        return cpu_result.view(torch.uint8).to(weight_fp8.device).view(weight_fp8.dtype).view(weight_fp8.shape)
    if _use_triton_fp8_kernel(weight_fp8.device):
        fused_add_round_launch(delta_bf16, weight_fp8, seed=0)
    else:
        delta_bf16.add_(weight_fp8.to(dtype=torch.bfloat16))
    return delta_bf16.to(dtype=weight_fp8.dtype)


E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
# fp8 x fp8 GEMMs (torch._scaled_mm) need compute capability 8.9+ (Ada, Hopper, Blackwell).
_FP8_GEMM_MIN_CAPABILITY = (8, 9)


def supports_fp8_gemm(device: torch.device) -> bool:
    """Whether *device* runs the e4m3 x e4m3 GEMMs of :class:`Fp8RowwiseLinear`."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device) >= _FP8_GEMM_MIN_CAPABILITY


class Fp8RowwiseLinear(Fp8CastLinear):
    """:class:`Fp8CastLinear` that multiplies long activations in e4m3 instead of upcasting the weight.
    A bf16 input of at least ``_min_rows`` rows on a device with fp8 GEMMs (:func:`supports_fp8_gemm`)
    is rounded to e4m3 with one scale per row (the row's absolute maximum maps to the e4m3 maximum)
    and multiplied with the stored e4m3 weight at scale 1 through ``torch._scaled_mm`` (unit scales,
    fp32 accumulation, bf16 product); the row scales and the bias are then applied in fp32. Other
    inputs, including any on a device without fp8 GEMMs, take the upcast path. Lossy:
    the activations keep e4m3's 3 mantissa bits. The cast is plain tensor ops so a compile
    backend can fuse it into the kernel producing the activation.
    """

    _min_rows: int

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002, type: ignore[override]
        in_features = input.shape[-1]
        rows = input.numel() // in_features
        if (
            rows < self._min_rows
            or input.dtype != torch.bfloat16
            or not supports_fp8_gemm(input.device)
            or self._with_stochastic_rounding
        ):
            return super().forward(input)
        x = input.reshape(rows, in_features)
        amax = x.abs().amax(dim=-1, keepdim=True).float()
        row_scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
        x8 = (x.float() / row_scale).to(torch.float8_e4m3fn)
        # Unit (tensor-wise) scales select the cuBLASLt fp8 GEMM; per-row scales would select a slower CUTLASS
        # kernel with per-launch host setup. The row scale and the bias are applied to the bf16 product as plain
        # tensor ops, which a compile backend fuses into the kernel consuming this Linear's output.
        unit = torch.ones((), device=input.device, dtype=torch.float32)
        out = torch._scaled_mm(
            x8, self.weight.t(), scale_a=unit, scale_b=unit, out_dtype=torch.bfloat16, use_fast_accum=False
        )
        out = out.float() * row_scale
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(torch.bfloat16).view(*input.shape[:-1], self.out_features)


def _amend_forward_with_rowwise_fp8_gemm(model: torch.nn.Module, min_rows: int) -> torch.nn.Module:
    """Retype every :class:`Fp8CastLinear` of *model* to :class:`Fp8RowwiseLinear`."""
    for m in model.modules():
        if type(m) is Fp8CastLinear:
            m.__class__ = Fp8RowwiseLinear
            m._min_rows = min_rows
    return model


def with_rowwise_fp8_gemm(policy: QuantizationPolicy, min_rows: int = 8192) -> QuantizationPolicy:
    """*policy* (an fp8-cast policy) with its Linears' long inputs multiplied in e4m3 (:class:`Fp8RowwiseLinear`).
    Weights and their loading are unchanged, so a stage on this policy shares its weights with one
    on *policy*. Only for devices where :func:`supports_fp8_gemm` holds. ``min_rows`` is the per-rank
    activation row count from which the e4m3 GEMM is taken: the video token sequences of long passes,
    not the audio or text-context projections.
    """
    if policy.fuse_rule is not FP8_CAST_UNIT_FUSE_RULE:
        raise ValueError("with_rowwise_fp8_gemm needs an fp8-cast policy (weights stored in e4m3)")
    op = ModuleOps(
        name=f"fp8_rowwise_gemm_from_{min_rows}_rows",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=functools.partial(_amend_forward_with_rowwise_fp8_gemm, min_rows=min_rows),
    )
    return dataclasses.replace(policy, module_ops=(*policy.module_ops, op))


def _fp8_cast_unit_patch(
    key: str, weight: torch.Tensor, delta: torch.Tensor, _clean: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return {key: fuse_cast_fp8_weight(delta.to(torch.bfloat16), weight)}
    if not weight.is_floating_point() or weight.dtype == torch.uint8:
        raise ValueError(f"{key!r} has an unsupported FP8 cast fuse payload")
    return {key: (weight.to(delta.dtype) + delta).to(weight.dtype)}


FP8_CAST_UNIT_FUSE_RULE = FuseRule(
    "fp8-cast", torch.bfloat16, _fp8_cast_unit_patch, host_fused_dtypes=MPS_HOST_FUSED_FP8
)


def _fold_unit_scale(scale_key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    parent = scale_key.removesuffix("_scale")
    if parent not in inputs:
        raise ValueError(f"prequant scale {scale_key!r} has no parent tensor")
    scale, weight = inputs[scale_key], inputs[parent]
    if scale.ndim != 0:
        raise ValueError(f"prequant scale {scale_key!r} must be scalar")
    return {parent: (weight.float() * scale.float()).to(torch.bfloat16)}


def _drop_activation_scale(_key: str, _inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {}


def fp8_cast_prequant_fold_stage() -> SDOps:
    """Fold checkpoint weight and bias scales into their parents before downcasting.
    FP8 casting keeps activations in bf16, so static activation scales (``.input_scale``) are dropped.
    """
    fold = KeyValueOperation(
        _fold_unit_scale, auxiliary_key_specs=((KeyMatching(suffix="_scale"), KeyReplacement("_scale", "")),)
    )
    return (
        SDOps("fp8-cast-fold-prequant")
        .with_kv_operation(fold, key_suffix=".weight_scale")
        .with_kv_operation(fold, key_suffix=".bias_scale")
        .with_kv_operation(_drop_activation_scale, key_suffix=".input_scale")
    )


def _downcast_unit_weight(key: str, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: inputs[key].to(torch.float8_e4m3fn)}


def fp8_cast_downcast_stage() -> SDOps:
    """Downcast the selected transformer linear weights and biases after folding."""
    stage = SDOps("fp8-cast-downcast")
    for suffix in _FP8_CAST_LINEAR_SUFFIXES:
        for parameter in (".weight", ".bias"):
            stage = stage.with_kv_operation(
                _downcast_unit_weight, key_prefix=_FP8_CAST_KEY_PREFIX, key_suffix=suffix + parameter
            )
    return stage


def fp8_cast_upcast_module_op(*, stochastic: bool = False, seed: int = 0) -> ModuleOps:
    """Rewrite selected transformer linears to upcast their FP8 weights at use."""
    return ModuleOps(
        name=f"fp8-cast-upcast:stochastic={stochastic}:seed={seed if stochastic else 0}",
        matcher=lambda model: hasattr(model, "transformer_blocks"),
        mutator=lambda model: _amend_forward_with_upcast(model, stochastic, seed),
    )


def build_policy(_checkpoint_path: str | Path) -> QuantizationPolicy:
    """FP8 casting with upcasting during inference.
    The checkpoint path is accepted for the shared policy-factory interface.
    Checkpoint scale siblings are folded when the model is materialized.
    """
    return QuantizationPolicy(
        sd_ops=(fp8_cast_prequant_fold_stage(), fp8_cast_downcast_stage()),
        module_ops=(fp8_cast_upcast_module_op(),),
        fuse_rule=FP8_CAST_UNIT_FUSE_RULE,
    )
