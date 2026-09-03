# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A MoE layer's activation: one carrier, one backend declaration, one adapter.

``MoEActivation``
    What a model builds. A union of per-kind dataclasses, so a kind and its
    constants cannot disagree, and a constant a kind does not have cannot be
    written down at all.

``MoEActivationSupport``
    What a backend publishes, as a class attribute. Which kinds its kernels
    execute, and what shape each constant must reach them in.

``materialize_activation_params``
    The only place a semantic name becomes a kernel register, and the only
    place a scalar is broadcast to per-expert or a per-expert tensor reduced
    to a scalar.

The split matters because the two floats every gated activation carries are
not one shared concept: they are two registers in the activation functor that
unrelated kinds borrow for unrelated jobs. ``SwigluBias`` reads ``alpha`` as a
scale inside the sigmoid and ``beta`` as an additive offset (neutral ``0.0``);
``SiTu`` reads both as tanh soft-cap magnitudes that must be positive (neutral
``1.0``). See ``cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/
moe_kernels.cuh`` (``SwigluBiasAdaptor`` / ``SiTuAdaptor``) and
``GemmGatedActOptions.h``. A single nullable ``beta`` slot therefore has no
coherent default and no readable meaning until you know the kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar

import torch

from ...utils import ActivationType

__all__ = [
    "ACTIVATION_CONSTANT_NAMES",
    "activation_constant_names",
    "install_activation_params",
    "ACTIVATION_PAYLOAD",
    "ActivationConstant",
    "ActivationConstants",
    "ActivationParamShape",
    "DEFAULT_MOE_ACTIVATION",
    "MaterializedActivation",
    "MoEActivation",
    "MoEActivationSupport",
    "resolve_activation_support",
    "SimpleActivation",
    "SiTuActivation",
    "SwigluActivation",
    "SwigluBiasActivation",
    "materialize_activation_params",
]

#: One value for the whole layer, or one per local expert. Which form reaches
#: the kernel is the backend's call (``MoEActivationSupport``).
ActivationConstant = torch.Tensor | float


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
#
# Above the carriers: their ``__post_init__`` calls these, and
# ``DEFAULT_MOE_ACTIVATION`` constructs one at module scope.


def _reject_non_positive_clamp(value: ActivationConstant | None) -> None:
    """Reject a clamp the kernel ABI cannot distinguish from "no clamp".

    ``swiglu_impl`` derives presence from the value itself
    (``HAS_SWIGLU_LIMIT = swiglu_limit is not None and swiglu_limit > 0.0`` in
    ``torch_custom_ops.py``), and the CuteDSL epilogue spells absence as
    ``+inf``. So a clamp of zero silently means "no clamp".
    """
    if value is None:
        return
    smallest = (
        float(value.detach().min().item()) if isinstance(value, torch.Tensor) else float(value)
    )
    if not smallest > 0:
        raise ValueError(
            f"activation clamp must be positive because the kernel ABI encodes an absent "
            f"clamp as a non-positive value; got {smallest}."
        )


def _reject_non_positive(value: ActivationConstant, *, name: str) -> None:
    smallest = (
        float(value.detach().min().item()) if isinstance(value, torch.Tensor) else float(value)
    )
    if not smallest > 0:
        raise ValueError(
            f"SiTu {name} must be positive because the kernel divides by it; got {smallest}."
        )


# ---------------------------------------------------------------------------
# Carrier
# ---------------------------------------------------------------------------
#
# ``eq=False`` throughout: these hold tensors, and the generated ``__eq__``
# would compare elementwise and then call ``bool()`` on the result, which raises.


@dataclass(frozen=True, eq=False)
class ActivationConstants:
    """The activation functor's register triple, in kernel-ABI order.

    The one representation that speaks ``alpha`` / ``beta``, because at this
    boundary those *are* the names: ``ActFn::alpha`` / ``::beta`` / ``::limit``
    in ``moe_kernels.cuh``, ``gemm1_alpha`` / ``gemm1_beta`` in trtllm-gen.
    Produced only by ``MoEActivation.constants()``; never authored by a caller.
    """

    alpha: ActivationConstant | None = None
    beta: ActivationConstant | None = None
    limit: ActivationConstant | None = None


@dataclass(frozen=True, eq=False)
class SwigluActivation:
    """``silu(gate) * linear``, optionally clamped."""

    clamp: ActivationConstant | None = None

    kind: ClassVar[ActivationType] = ActivationType.Swiglu

    def __post_init__(self) -> None:
        _reject_non_positive_clamp(self.clamp)

    def constants(self) -> ActivationConstants:
        return ActivationConstants(limit=self.clamp)


@dataclass(frozen=True, eq=False)
class SwigluBiasActivation:
    """gpt-oss / MiniMax gated SwiGLU.

    ``g*sigmoid(g*gate_sigmoid_scale)*(l + linear_offset)`` where ``g`` / ``l``
    are the gate / linear halves of FC1 clamped by ``clamp``.

    ``linear_offset`` is not the MoE ``bias``: the kernel adds the per-expert
    bias pointer to both halves *before* clamping, then adds this constant to
    the clamped linear half (``moe_kernels.cuh`` ``SwigluBiasAdaptor``). Naming
    it ``linear_bias`` would collide with that.
    """

    gate_sigmoid_scale: ActivationConstant
    linear_offset: ActivationConstant
    clamp: ActivationConstant | None = None

    kind: ClassVar[ActivationType] = ActivationType.SwigluBias

    def __post_init__(self) -> None:
        _reject_non_positive_clamp(self.clamp)

    def constants(self) -> ActivationConstants:
        return ActivationConstants(
            alpha=self.gate_sigmoid_scale, beta=self.linear_offset, limit=self.clamp
        )


@dataclass(frozen=True, eq=False)
class SiTuActivation:
    """Kimi K3 SiTU: two independently soft-capped branches.

    ``softcap(gate, gate_softcap) * sigmoid(gate) * softcap(linear,
    linear_softcap)`` with ``softcap(x, c) = c*tanh(x / c)``, i.e. a smooth
    saturation of ``x`` to ``+-c``. There is no separate clamp.

    Both caps must be positive because the kernel divides by them
    (``GemmGatedActOptions.h``: ``SiTuGlu`` uses ``1/alpha`` and ``1/beta``), so
    zero is not a neutral value here the way it is for ``SwigluBias``.

    Checkpoint provenance: ``gate_softcap`` is Kimi's ``activation_situ_beta``
    and ``linear_softcap`` its ``activation_situ_linear_beta``.
    """

    gate_softcap: ActivationConstant
    linear_softcap: ActivationConstant

    kind: ClassVar[ActivationType] = ActivationType.SiTu

    def __post_init__(self) -> None:
        for name in ("gate_softcap", "linear_softcap"):
            _reject_non_positive(getattr(self, name), name=name)

    def constants(self) -> ActivationConstants:
        return ActivationConstants(alpha=self.gate_softcap, beta=self.linear_softcap)


@dataclass(frozen=True, eq=False)
class SimpleActivation:
    """A kind whose kernels take no activation constants (Silu, Geglu, Relu2 ...)."""

    kind: ActivationType

    def __post_init__(self) -> None:
        payload = ACTIVATION_PAYLOAD.get(ActivationType(self.kind))
        if payload is None:
            raise ValueError(
                f"{ActivationType(self.kind).name} is not a MoE activation kind; "
                f"known kinds: {', '.join(sorted(k.name for k in ACTIVATION_PAYLOAD))}"
            )
        if payload is not SimpleActivation:
            raise ValueError(
                f"{ActivationType(self.kind).name} takes activation constants; "
                f"build a {payload.__name__} instead of SimpleActivation."
            )

    def constants(self) -> ActivationConstants:
        return ActivationConstants()


MoEActivation = SwigluActivation | SwigluBiasActivation | SiTuActivation | SimpleActivation


#: Entry point for a config-driven caller that knows only a kind: the member's
#: ``__init__`` signature *is* the constant list for that kind.
ACTIVATION_PAYLOAD: dict[ActivationType, type[MoEActivation]] = {
    ActivationType.Gelu: SimpleActivation,
    ActivationType.Relu: SimpleActivation,
    ActivationType.Silu: SimpleActivation,
    ActivationType.Relu2: SimpleActivation,
    ActivationType.Geglu: SimpleActivation,
    ActivationType.Swiglu: SwigluActivation,
    ActivationType.SwigluBias: SwigluBiasActivation,
    ActivationType.SiTu: SiTuActivation,
}

#: Plain SwiGLU, no constants -- the historical default of every MoE signature.
#: Safe to share: the carrier is frozen.
DEFAULT_MOE_ACTIVATION: MoEActivation = SwigluActivation()


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


class ActivationParamShape(Enum):
    """Form a backend's kernel boundary requires for a per-expert constant."""

    #: Kernel dereferences a ``float*`` indexed by local expert.
    PER_EXPERT_TENSOR = auto()
    #: Kernel bakes one value for the whole layer (constexpr, or passed by value).
    UNIFORM_SCALAR = auto()
    #: Kernel has no such parameter.
    UNSUPPORTED = auto()


@dataclass(frozen=True)
class MoEActivationSupport:
    """What one backend's kernels can actually do with an activation.

    ``kinds`` declares what the backend **executes**, not what its gate
    happens to accept. Several backends silently narrow an accepted kind to
    SwiGLU or SiLU; those kinds do not belong here.

    ``alpha_beta`` keeps the ABI names on purpose: it describes the shape of the
    functor's register pair, and it is only ever applied to an
    ``ActivationConstants`` produced by ``MoEActivation.constants()``.

    ``limit_when_absent`` exists because some clamp ABIs have no "absent"
    encoding -- the CuteDSL epilogue always applies the clamp functor, so "no
    clamp" has to be spelled as a value. A backend whose ABI accepts None
    leaves this unset.
    """

    kinds: frozenset[ActivationType]
    alpha_beta: ActivationParamShape = ActivationParamShape.UNSUPPORTED
    limit: ActivationParamShape = ActivationParamShape.UNSUPPORTED
    limit_when_absent: float | None = None

    def __post_init__(self) -> None:
        if self.limit_when_absent is not None and self.limit is ActivationParamShape.UNSUPPORTED:
            raise ValueError(
                "limit_when_absent names the value a clamp-less layer must still pass, so it "
                "is meaningless with limit=UNSUPPORTED: declare a shape, or drop the value."
            )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MaterializedActivation:
    """The activation in exactly the forms one backend's kernels take.

    Kind-neutral names on purpose. The C++ boundary calls these registers
    ``swiglu_alpha`` / ``swiglu_beta`` / ``swiglu_limit`` (``moe_kernels.h``'s
    ``ActivationParams``, and the ``moe_op`` schema), but SiTU fills the same
    slots with tanh soft-caps. The op-call sites keep the schema names; nothing
    above them does.

    One field per register, not one per ABI form: a clamp reaches its kernel
    either as a ``float*`` indexed by expert or as a value, but which one is
    already pinned by the backend's ``MoEActivationSupport``.
    """

    activation_type: ActivationType
    alpha: ActivationConstant | None = None
    beta: ActivationConstant | None = None
    clamp: ActivationConstant | None = None


def materialize_activation_params(
    activation: MoEActivation,
    support: MoEActivationSupport,
    *,
    num_local_experts: int,
    device: torch.device | str | None = None,
    owner: str,
) -> MaterializedActivation:
    """Convert an activation to the exact forms ``support`` declares.

    Takes ``activation.constants()`` -- the sole semantic-name-to-register seam
    -- then broadcasts scalars to ``float32[num_local_experts]`` for
    ``PER_EXPERT_TENSOR`` and reduces uniform tensors to a float for
    ``UNIFORM_SCALAR``. A non-uniform tensor bound for ``UNIFORM_SCALAR`` is
    rejected: the kernel bakes the value, so no single scalar is faithful.

    Rejections name ``owner`` and the offending parameter so a mis-declared
    backend is identifiable without reading the factory.
    """
    kind = ActivationType(activation.kind)
    if kind not in support.kinds:
        supported = ", ".join(sorted(k.name for k in support.kinds))
        raise ValueError(
            f"{owner} does not implement activation {kind.name}; it executes: {supported}."
        )

    constants = activation.constants()
    alpha = _materialize_to_declared_shape(
        constants.alpha,
        support.alpha_beta,
        name="alpha",
        kind=kind,
        num_local_experts=num_local_experts,
        device=device,
        owner=owner,
    )
    beta = _materialize_to_declared_shape(
        constants.beta,
        support.alpha_beta,
        name="beta",
        kind=kind,
        num_local_experts=num_local_experts,
        device=device,
        owner=owner,
    )
    # Substituted before materialization, not after: it is a plain float standing in for
    # a missing constant, so a backend declaring PER_EXPERT_TENSOR needs it
    # broadcast to the per-expert buffer like any caller-supplied scalar.
    limit = _materialize_to_declared_shape(
        constants.limit if constants.limit is not None else support.limit_when_absent,
        support.limit,
        name="clamp",
        kind=kind,
        num_local_experts=num_local_experts,
        device=device,
        owner=owner,
    )

    return MaterializedActivation(
        activation_type=kind,
        alpha=alpha,
        beta=beta,
        clamp=limit,
    )


def resolve_activation_support(module: torch.nn.Module) -> MoEActivationSupport:
    """The declaration that applies to ``module``, class attribute or override.

    Static for ten of the eleven backends. TRTLLM-Gen is the documented
    exception: its clamp ABI is a per-expert tensor for the FP4 fused-activation
    cubins but a by-value ``double`` for the FP8 block-scale separate-activation
    kernel, which is not a property of the class, so it defines
    ``resolve_activation_support`` and narrows the shape per instance.
    """
    override = getattr(module, "resolve_activation_support", None)
    if callable(override):
        return override()
    support = getattr(type(module), "activation_support", None)
    if support is None:
        raise TypeError(
            f"{type(module).__name__} must declare an ``activation_support`` class "
            f"attribute stating which activations its kernels execute."
        )
    return support


def install_activation_params(
    module: torch.nn.Module, *, device: torch.device | str | None = None
) -> None:
    """Assign the ``act_*`` slots ``module``'s kernels read, from ``module.activation``.

    The one place a layer or execution unit turns its declared activation into
    the three attributes the forward paths and the quantization layer read. Runs
    at construction, and again once ``ConfigurableMoE`` has synced
    ``expert_size_per_partition`` -- the only thing that changes the per-expert
    length -- before any weight is created.

    Safe to repeat only before weights exist: it re-materializes from
    ``module.activation``, undoing any in-place transform a quant method applied.
    """
    support = resolve_activation_support(module)
    params = materialize_activation_params(
        module.activation,
        support,
        num_local_experts=module.expert_size_per_partition,
        device=device,
        owner=type(module).__name__,
    )
    module.activation_params = params
    _write_activation_slot(module, "act_alpha", params.alpha)
    _write_activation_slot(module, "act_beta", params.beta)
    _write_activation_slot(module, "act_clamp", params.clamp)


def _write_activation_slot(
    module: torch.nn.Module, name: str, value: ActivationConstant | None
) -> None:
    """Write one slot so the constant travels with the weights it is used with.

    A tensor constant is registered rather than assigned, because a plain
    attribute is the one thing ``nn.Module.to()`` does not move: the weights
    reach the execution device that way, and a constant left behind arrives at
    the kernel on the wrong device. ``persistent=False`` keeps it out of the
    state dict, which is where a plain attribute already was -- these are
    backend configuration, not checkpoint values.

    A slot that is already a parameter stays one. ``TRTLLMGenFusedMoE`` promotes
    the SiTu slots so they do travel in the state dict, and the exclude-modules
    pass clears ``_weights_created`` without unregistering them, so this runs
    again over a live parameter.
    """
    if isinstance(value, torch.Tensor) and name in getattr(module, "_parameters", {}):
        setattr(module, name, torch.nn.Parameter(value, requires_grad=False))
        return
    # Installing happens more than once -- EPLB slot sync and the layerwise
    # quant config each redo it -- so clear the previous binding, of whichever
    # kind, before making the new one. ``delattr`` rather than popping
    # ``_buffers``, so a slot going from tensor back to scalar also leaves
    # ``_non_persistent_buffers_set``.
    if hasattr(module, name):
        delattr(module, name)
    if isinstance(value, torch.Tensor):
        module.register_buffer(name, value, persistent=False)
    else:
        setattr(module, name, value)


#: The ABI register names, in the order ``ActivationConstants`` declares them.
#: ``alpha`` and ``beta`` share one declared shape (``alpha_beta``); ``clamp``
#: has its own (``limit``).
ACTIVATION_CONSTANT_NAMES: tuple[str, ...] = ("alpha", "beta", "clamp")


def activation_constant_names(activation: MoEActivation | None) -> frozenset[str]:
    """Which of the three ABI registers this activation actually fills.

    The selection layer needs this and cannot get it from the kind alone: SwiGLU
    with a clamp and SwiGLU without one are the same ``ActivationType`` but not
    the same request, and a backend declaring ``limit=UNSUPPORTED`` can serve
    only the second. Returning names rather than the values keeps ``MoEProblem``
    hashable and JSON-serializable, which a tuning key has to be.
    """
    if activation is None:
        return frozenset()
    constants = activation.constants()
    return frozenset(
        name
        for name in ACTIVATION_CONSTANT_NAMES
        if getattr(constants, "limit" if name == "clamp" else name) is not None
    )


def _materialize_to_declared_shape(
    value: ActivationConstant | None,
    shape: ActivationParamShape,
    *,
    name: str,
    kind: ActivationType,
    num_local_experts: int,
    device: torch.device | str | None,
    owner: str,
) -> ActivationConstant | None:
    """Put one constant in the form ``shape`` declares, or refuse to.

    Not a lenient conversion: a value bound for ``UNSUPPORTED`` raises rather
    than being dropped, and a per-expert tensor bound for ``UNIFORM_SCALAR``
    raises unless its values are exactly equal.
    """
    if value is None:
        return None
    if shape is ActivationParamShape.UNSUPPORTED:
        raise ValueError(
            f"{owner} kernels take no activation {name}, but {kind.name} supplied one. "
            f"Either drop it or select a backend that declares it."
        )
    if shape is ActivationParamShape.UNIFORM_SCALAR:
        return _reduce_to_uniform_scalar(value, name=name, owner=owner)
    return _broadcast_to_per_expert(
        value, name=name, num_local_experts=num_local_experts, device=device, owner=owner
    )


def _reduce_to_uniform_scalar(value: ActivationConstant, *, name: str, owner: str) -> float:
    """Reduce a per-expert constant to the one float the kernel can bake."""
    if not isinstance(value, torch.Tensor):
        return float(value)
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        raise ValueError(f"{owner} received an empty activation {name}.")
    first = flat[0]
    # Exact equality, not allclose: the kernel bakes ``first`` and every other
    # expert inherits it, so a tolerance would discard the values that differ
    # rather than approximate them.
    if flat.numel() > 1 and bool((flat != first).any()):
        raise ValueError(
            f"{owner} only supports a uniform (per-layer) activation {name} because the "
            f"kernel bakes it as a compile-time scalar; got per-expert values "
            f"{flat.cpu().tolist()}."
        )
    return float(first.item())


def _broadcast_to_per_expert(
    value: ActivationConstant,
    *,
    name: str,
    num_local_experts: int,
    device: torch.device | str | None,
    owner: str,
) -> torch.Tensor:
    """Produce the ``float32[num_local_experts]`` buffer the kernel indexes.

    ``device=None`` stays ``None`` so the constant lands where ``create_weights``
    puts the weights it is combined with: both create with the ambient device
    (``nn.Parameter(torch.empty(shape, dtype=...))``), and
    ``install_activation_params`` registers this as a buffer so the same
    ``.to()`` moves both. Naming a device here instead makes the two disagree --
    a module built and loaded on CPU, then moved, divides by the FC31 scale
    while still on CPU.

    Meta is the one exception. These are configuration, not checkpoint values,
    so nothing reloads them after materialization; a meta constant would just be
    empty.
    """
    if device is None and torch.get_default_device().type == "meta":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(value, torch.Tensor):
        return torch.full((num_local_experts,), float(value), dtype=torch.float32, device=device)
    flat = value.detach().reshape(-1).to(dtype=torch.float32)
    if flat.numel() == num_local_experts:
        # ``.clone()`` because ``quantization.py`` divides these buffers in place
        # by the FC31 scale, and ``detach``/``reshape``/``to`` are no-ops for an
        # already-matching tensor -- so returning it would divide the caller's own
        # constant.
        return flat.to(device=device).clone()
    if flat.numel() == 1:
        return flat.to(device=device).expand(num_local_experts).contiguous()
    raise ValueError(
        f"{owner} indexes activation {name} by local expert, so it must hold "
        f"{num_local_experts} values (or one to broadcast); got {flat.numel()}."
    )
