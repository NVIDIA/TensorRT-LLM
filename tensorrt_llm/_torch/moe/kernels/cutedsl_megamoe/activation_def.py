# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The elementwise function between an expert's FC1 and FC2."""

import enum


class ExpertActivation(str, enum.Enum):
    """The activation one MegaMoE expert applies to its FC1 output before FC2.

    A gated kind reads FC1's rows as [gate 16, up 16] groups and emits one feature per pair, so
    intermediate_gateup_size (FC1's rows) is twice FC2's K. relu2 emits one feature per row, so the
    two extents are equal. Constants keep their own ProblemDesc fields: gate_up_clamp and
    swiglu_alpha/beta (bias-free SwigluBias) for swiglu, situ_beta/situ_linear_beta for situ;
    relu2 takes none.
    """

    swiglu = "swiglu"
    situ = "situ"
    relu2 = "relu2"

    # Pinned like QuantKind (quant_def.py): the value keys the compiled-kernel name, so it must not
    # become "ExpertActivation.relu2" on an interpreter upgrade.
    __str__ = str.__str__
    __format__ = str.__format__

    @property
    def is_gated(self) -> bool:
        """Whether FC1's rows come in [gate, up] pairs that fold into one FC2 input each."""
        return self is not ExpertActivation.relu2

    def intermediate_downproj(self, intermediate_gateup_size):
        """FC2's K extent, which is also the width of the FC1 output FC2 reads (int or Int32)."""
        return intermediate_gateup_size // 2 if self.is_gated else intermediate_gateup_size

    @classmethod
    def from_problem_desc(cls, problem_desc) -> "ExpertActivation":
        """The declared ``expert_activation``; absent or None infers the kinds that predate it.

        Before the field existed a descriptor selected SiTU by setting ``situ_beta`` and SwiGLU
        otherwise, so those descriptors keep their meaning. An unknown name raises ValueError.
        """
        declared = problem_desc.get("expert_activation")
        if declared is not None:
            return cls(declared)
        return cls.situ if problem_desc.get("situ_beta") is not None else cls.swiglu
