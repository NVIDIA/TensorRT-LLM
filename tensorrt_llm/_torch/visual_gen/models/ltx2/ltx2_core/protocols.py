# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports from the pinned November 4 LTX core."""

from tensorrt_llm._torch.visual_gen.models.ltx_core.components.protocols import (
    DiffusionStepProtocol as DiffusionStepProtocol,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.components.protocols import (
    Patchifier as Patchifier,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.components.protocols import (
    SchedulerProtocol as SchedulerProtocol,
)
