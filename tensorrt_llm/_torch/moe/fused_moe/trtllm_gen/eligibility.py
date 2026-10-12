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
"""The eligibility gates the leaves compose into ``can_implement``.

Free functions rather than methods on :class:`.TrtllmGenFusedMoEBase`: the
family base must not implement ``can_implement``, or every leaf would
inherit an answer that belongs to no single identity. Each leaf composes its
own gates explicitly, so one can diverge without unpicking an inherited
default.

Each gate reads only ``cls``, ``MoEProblem`` and ``MoEDeployment``. No
``get_sm_version()``, no ``os.environ``, no import probe, so an offline tuner
on a GPU-less host gets the same verdict a serving process does.
"""

import torch

from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm._utils import is_sm_100f

from ..impl_contract import (
    MoEDeployment,
    MoEEligibility,
    MoEProblem,
    MoERejectReason,
    check_quant_matches_identity,
)
from ..impl_environment import MoEDep, MoEEnvFlag
from ..interface import _reject


def check_trtllm_gen_capabilities(
    cls: type, p: MoEProblem, d: MoEDeployment
) -> MoEEligibility | None:
    """Capability gates every TRTLLM-Gen leaf shares, or ``None`` to admit.

    Every leaf restating the same gates is how the SM, dtype, routing and
    activation answers drift apart, so each leaf checks its own quantization
    format and then defers here.
    """
    # The cubin drop is sm_100f (family-compatible) plus arch-specific
    # sm_100a/sm_103a/sm_107a, so the whole SM100 family is servable; the C++
    # selector (KernelRunner.cpp isSMCompatible) picks sm_100f on family
    # members without their own arch build. A leaf whose dtype pair has no
    # sm_100f build (W4A8 NVFP4 FP8) narrows this in its own can_implement.
    if not is_sm_100f(d.env.sm):
        return _reject(
            MoERejectReason.SM_UNSUPPORTED,
            f"{cls.__name__} requires the SM100 family, got SM{d.env.sm}",
        )

    # run_moe asserts x.dtype == torch.bfloat16
    if p.dtype_act != torch.bfloat16:
        return _reject(
            MoERejectReason.DTYPE_UNSUPPORTED,
            f"{cls.__name__} only supports bfloat16 activation, got {p.dtype_act}",
        )

    if d.smart_router:
        return _reject(
            MoERejectReason.TOPOLOGY_UNSUPPORTED,
            f"{cls.__name__} has no smart-router path (moe_cluster_size={d.cluster_size})",
        )

    # Whether the gpt-oss SwiGLU package (expert bias plus alpha/beta) has a
    # fused cubin is a per-quant fact; the leaf declares it, this only applies it.
    if p.swiglu_gptoss_style and not cls.supports_gptoss_style:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} has no fused bias/swiglu-parameter cubin for its "
            f"quantization format (quant={p.quant})",
        )

    # SiTu exists only as a fused FC1 epilogue, with no standalone activation
    # kernel to fall back to, and reaching one takes both a cubin family that
    # ships it and a provider that calls it -- so unlike gpt-oss above, the
    # leaf holds the answer, not the format.
    if p.activation_type is ActivationType.SiTu and not cls.supports_situ:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} reaches no fused SiTu cubin (quant={p.quant}, "
            f"provider={cls.provider})",
        )

    return None


def check_flashinfer_provider(cls: type, p: MoEProblem, d: MoEDeployment) -> MoEEligibility | None:
    """Gates that separate the FlashInfer provider from the ``trtllm`` one.

    Only the quantized leaves come through here. The unquantized one is
    FlashInfer-exclusive and reached without the opt-in flag, so it states its
    own dependency gate instead.
    """
    if not d.env.has_dep(MoEDep.FLASHINFER):
        return _reject(MoERejectReason.DEP_MISSING, f"{cls.__name__} requires the FlashInfer wheel")

    # Opt-in, not a capability: the trtllm provider serves these formats too,
    # so routing traffic here without being asked would change which kernel a
    # previously-working deployment runs.
    if d.env.env_flag(MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER) != "1":
        return _reject(
            MoERejectReason.PATH_NOT_ENABLED,
            f"{cls.__name__} is opt-in; set "
            f"{MoEEnvFlag.TRTLLM_GEN_USE_FLASHINFER.value}=1 to select the "
            f"FlashInfer provider for quantized TRTLLM-Gen",
        )

    # SiTu is a native TRTLLM-Gen cubin and is absent from FlashInfer's
    # activation enum; Relu2 has no FlashInfer path either.
    if p.activation_type in (ActivationType.SiTu, ActivationType.Relu2):
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} does not implement {p.activation} "
            f"(FlashInfer's activation enum has no such kernel)",
        )

    # These two fuse routing in a form only the native runner accepts. Matched
    # on the algorithm the problem carries, not on the value handed to the C++
    # kernels: a method no kernel implements borrows a recognized encoding
    # there, and matching on the borrowed one would reject a method this gate
    # has nothing against.
    if p.routing_method_type is not None and p.routing_method_type.name in (
        "DeepSeekV3",
        "Default",
    ):
        return _reject(
            MoERejectReason.ROUTING_UNSUPPORTED,
            f"{cls.__name__} does not implement {p.routing} routing",
        )

    return None


def check_flashinfer_shard_alignment(
    cls: type,
    p: MoEProblem,
    d: MoEDeployment,
    *,
    weight_alignment: int,
    input_hidden_alignment: int | None = None,
) -> MoEEligibility | None:
    """Per-rank shard alignment the FlashInfer kernels need, or ``None``.

    Two distinct spellings of "does not apply": an absent shape on ``p`` means
    the caller did not say, so abstain; an ``input_hidden_alignment`` of
    ``None`` means the kernels constrain the hidden dimension not at all.
    """
    if p.intermediate_size is not None:
        inter = p.intermediate_size
        if d.tp_size > 1:
            if inter % d.tp_size != 0:
                return _reject(
                    MoERejectReason.SHAPE_UNALIGNED,
                    f"{cls.__name__} requires intermediate_size ({inter}) "
                    f"divisible by moe_tp_size ({d.tp_size})",
                )
            inter = inter // d.tp_size
        if inter % weight_alignment != 0:
            return _reject(
                MoERejectReason.SHAPE_UNALIGNED,
                f"{cls.__name__} requires intermediate_size_per_partition "
                f"({inter}) to be a multiple of {weight_alignment}",
            )

    if (
        input_hidden_alignment is not None
        and p.hidden_size is not None
        and p.hidden_size % input_hidden_alignment != 0
    ):
        return _reject(
            MoERejectReason.SHAPE_UNALIGNED,
            f"{cls.__name__} requires hidden_size ({p.hidden_size}) to be a "
            f"multiple of {input_hidden_alignment}",
        )

    return None


def check_no_expert_bias(cls: type, p: MoEProblem) -> MoEEligibility | None:
    """No standalone expert bias, for the leaves whose runner reads none.

    Narrower than two gates that look like they would cover it.
    ``supports_gptoss_style`` sees only the bias-plus-alpha-beta package as a
    whole, and a checkpoint can carry a plain expert bias without it.
    ``capabilities.supports_expert_bias`` is declared once for the whole
    family, and ``create_moe`` reads it against the class resolution already
    picked -- so a leaf that answered only there would be chosen and then
    refuse, instead of standing aside for a sibling that can serve.

    A gate rather than only the ``_check_configs`` assert for the same reason:
    some leaves reaching here have a sibling on the other provider that does
    take a bias, and standing aside during selection is what lets it be
    tried; for the rest it is the difference between a structured verdict
    and an assert after resolution has committed.

    Its own function rather than a line inside
    :func:`check_mxfp4_flashinfer_shape`, because a leaf whose shape gate is
    conditional still has to answer about bias unconditionally, and one copy
    of the rule keeps every leaf giving the same reason.
    """
    if p.bias:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} takes no expert bias",
        )
    return None


def check_no_activation_constants(cls: type, p: MoEProblem) -> MoEEligibility | None:
    """No activation constants at all, for the runners that take none.

    Companion to :func:`check_no_expert_bias` and the same argument: these
    runners have no parameter for alpha, beta or the clamp and pin
    ``act_type``, so a constant the carrier materialized would be dropped
    rather than applied. ``supports_gptoss_style`` does not cover it -- that
    gate sees the bias-plus-alpha-beta package as a whole, while a checkpoint
    can carry a plain clamp on its own, which is not part of the package at
    all.

    Any nonempty set, rather than a named subset, because for these leaves the
    kernel applies none of them.
    """
    if p.activation_constants:
        return _reject(
            MoERejectReason.ACTIVATION_UNSUPPORTED,
            f"{cls.__name__} takes no activation constants, got {sorted(p.activation_constants)}",
        )
    return None


def check_mxfp4_flashinfer_shape(
    cls: type,
    p: MoEProblem,
    d: MoEDeployment,
    *,
    weight_alignment: int,
    input_hidden_alignment: int,
) -> MoEEligibility | None:
    """The quantized FlashInfer leaves' shape gate: no bias, then alignment.

    Bundled rather than folded into
    :func:`check_flashinfer_shard_alignment`, because the unquantized leaf
    shares the alignment rule but answers about bias through
    ``swiglu_gptoss_style``.
    """
    bias_verdict = check_no_expert_bias(cls, p)
    if bias_verdict is not None:
        return bias_verdict
    return check_flashinfer_shard_alignment(
        cls,
        p,
        d,
        weight_alignment=weight_alignment,
        input_hidden_alignment=input_hidden_alignment,
    )


def nvfp4_needs_padded_method(activation_type: ActivationType, has_alpha_constant: bool) -> bool:
    """Whether NVFP4 needs the padded quant method rather than the base one.

    One definition read from two sides -- ``can_implement`` asks it of the
    problem, ``_get_quant_method`` of the instance -- so that a configuration
    cannot be admitted by one and laid out by the other.
    """
    return has_alpha_constant or activation_type in (
        ActivationType.SiTu,
        ActivationType.Relu2,
        ActivationType.Silu,
    )


def check_trtllm_gen_leaf(
    cls: type, p: MoEProblem, d: MoEDeployment, *provider_gates: MoEEligibility | None
) -> MoEEligibility:
    """Compose one leaf's verdict: identity, shared capability, then provider.

    First rejection wins, in a fixed order so that two leaves differing only in
    provider give the same reason for a problem neither can serve.
    ``provider_gates`` are already-evaluated verdicts, not callables: the gates
    are pure, so there is nothing to defer.
    """
    for verdict in (
        check_quant_matches_identity(cls, p),
        check_trtllm_gen_capabilities(cls, p, d),
        *provider_gates,
    ):
        if verdict is not None:
            return verdict
    return MoEEligibility.ok()
