# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

_verify_skipped_device_copies = os.environ.get("TLLM_DEBUG_MODE", "")[0:1] == "1"


class DeviceCopyBuffer:
    def __init__(self, tensor: torch.Tensor, *, pin_memory: bool = False):
        self._tensor = tensor
        self._pin_memory = pin_memory
        self._last_host_values = None

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def update(self, host: torch.Tensor, *, force: bool = False) -> None:
        flat = host.reshape(-1)
        n = flat.numel()
        last = self._last_host_values
        if not force and last is not None and last.numel() >= n and torch.equal(last[:n], flat):
            if _verify_skipped_device_copies:
                if self._tensor.is_cuda:
                    if torch.cuda.is_current_stream_capturing():
                        return
                    torch.cuda.synchronize(self._tensor.device)
                held = self._tensor.view(-1)[:n].cpu()
                assert torch.equal(held, flat.to(held.dtype)), (
                    f"DeviceCopyBuffer skipped a copy, but the buffer holds {held.tolist()} "
                    f"instead of {flat.tolist()}: something else wrote it"
                )
            return
        staging = torch.empty_like(flat, device="cpu", pin_memory=self._pin_memory)
        staging.copy_(flat)
        self._tensor.view(-1)[:n].copy_(staging, non_blocking=True)
        if last is None or last.numel() <= n:
            self._last_host_values = flat.clone()
        else:
            last = last.clone()
            last[:n] = flat
            self._last_host_values = last
