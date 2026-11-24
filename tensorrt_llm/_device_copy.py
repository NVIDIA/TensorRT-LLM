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

    def holds(self, host: torch.Tensor) -> bool:
        """Whether the leading elements of the buffer were last written from ``host``'s values,
        i.e. whether ``update(host)`` would skip its copy."""
        flat = host.reshape(-1)
        n = flat.numel()
        last = self._last_host_values
        return last is not None and last.numel() >= n and torch.equal(last[:n], flat)

    def record_written(self, host: torch.Tensor) -> None:
        """Record that the caller wrote ``host``'s values to the leading elements of the buffer
        (stream-ordered, as ``update`` does), so later updates compare against them."""
        flat = host.reshape(-1)
        n = flat.numel()
        last = self._last_host_values
        if last is None or last.numel() <= n:
            self._last_host_values = flat.clone()
        else:
            last = last.clone()
            last[:n] = flat
            self._last_host_values = last

    def update(self, host: torch.Tensor, *, force: bool = False) -> None:
        flat = host.reshape(-1)
        n = flat.numel()
        if not force and self.holds(flat):
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
        self.record_written(flat)
