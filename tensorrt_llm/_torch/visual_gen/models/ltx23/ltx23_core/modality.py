# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from dataclasses import dataclass, field

import torch

from ...ltx2.ltx2_core.modality import Modality


@dataclass(frozen=True)
class LTX23Modality(Modality):
    """LTX-2 Modality plus the global denoising sigma."""

    sigma: torch.Tensor = field(kw_only=True)  # (B,) drives prompt K/V
