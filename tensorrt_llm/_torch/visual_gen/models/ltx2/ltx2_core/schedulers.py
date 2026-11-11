# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Run the upstream schedule on the latent device expected by TRTLLM."""

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.components.schedulers import (
    BASE_SHIFT_ANCHOR as BASE_SHIFT_ANCHOR,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.components.schedulers import (
    MAX_SHIFT_ANCHOR as MAX_SHIFT_ANCHOR,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.components.schedulers import (
    LTX2Scheduler as UpstreamScheduler,
)


class LTX2Scheduler(UpstreamScheduler):
    def execute(self, steps, latent=None, **kwargs):
        device = latent.device if latent is not None else torch.device("cpu")
        # Explicit transfer also works when runtime disables TorchFunction modes,
        # in which case the torch.device context cannot redirect tensor factories.
        return super().execute(steps, latent=latent, **kwargs).to(device=device)
