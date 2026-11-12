# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The copy sites of the MiniMax-M3 modeling_v2 targets' own per-step uploads (device_copy.CopySiteEnum)."""

from tensorrt_llm._torch.pyexecutor.device_copy import CopySiteEnum, Direction


class M3SpecCopySite(CopySiteEnum):
    """M3SpecStepFold.begin_step's arming of the forward's tail: the armed rows' sequence slots and the row count.

    Only that step's forward reads them, so a plan may merge both into the step's pack.
    """

    FOLD_SLOTS = ("m3_spec_fold.slots", Direction.H2D)
    FOLD_CTRL = ("m3_spec_fold.ctrl", Direction.H2D)
