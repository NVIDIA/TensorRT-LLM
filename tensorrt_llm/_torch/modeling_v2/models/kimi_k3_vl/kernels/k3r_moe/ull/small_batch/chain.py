# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# TensorRT-LLM local change: update imports for the Kimi K3 family layout.
"""Model-owned MoE front storage published by the o-projection consumer."""

import torch

from .front_tp16 import initialize_designated


class SmallBatchBox:
    """Front rows and epoch hints shared serially across MoE layers."""

    def __init__(self, device, tp):
        self.batches = (1,) if tp == 8 else (1, 2)
        self.xmoe = torch.empty((2, 1, 4, 7168), dtype=torch.bfloat16, device=device)
        self.xmoe.view(torch.int32).fill_(-0x80000000)
        self.epoch = torch.zeros(256, dtype=torch.int32, device=device)
        if tp == 16:
            sms = torch.cuda.get_device_properties(device).multi_processor_count
            self.designated = torch.zeros(sms + 16, dtype=torch.int32, device=device)
            initialize_designated(self.designated)
        self.front_views = tuple(self.xmoe[p] for p in range(2))
        self.hint_views = tuple(self.epoch[p * 32 : p * 32 + 224] for p in range(2))
