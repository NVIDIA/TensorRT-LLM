# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

_verify_skipped_device_copies = os.environ.get("TLLM_DEBUG_MODE", "")[0:1] == "1"


class DeviceCopyBuffer:
    def __init__(self, tensor: torch.Tensor, *, pin_memory: bool = False) -> None:
        self._tensor = tensor
        self._pin_memory = pin_memory
        self._last_host_values = None
        self._last_host_shape: torch.Size | None = None
        self._copy_key: object = None

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def invalidate(self) -> None:
        """Forget a pending or externally overwritten copy."""
        self._last_host_values = None
        self._last_host_shape = None
        self._copy_key = None

    def matches(self, host: torch.Tensor, *, key: object = None, exact: bool = False) -> bool:
        """Whether the recorded host inputs match, including any copy-specific layout key.

        ``exact`` requires the same host shape instead of accepting a matching prefix.
        Callers that transform or defer their copies must record only successful submissions
        and invalidate the record if the submission is discarded or another writer intervenes.
        """
        last = self._last_host_values
        flat = host.reshape(-1)
        return (
            last is not None
            and key == self._copy_key
            and last.dtype == host.dtype
            and (not exact or self._last_host_shape == host.shape)
            and last.numel() >= flat.numel()
            and torch.equal(last[: flat.numel()], flat)
        )

    def record(self, host: torch.Tensor, *, key: object = None) -> None:
        """Remember the inputs of a copy issued by this buffer or a specialized copy path."""
        flat = host.reshape(-1)
        last = self._last_host_values
        if (
            last is None
            or key != self._copy_key
            or last.dtype != host.dtype
            or last.numel() <= flat.numel()
        ):
            self._last_host_values = flat.clone()
        else:
            last = last.clone()
            last[: flat.numel()] = flat
            self._last_host_values = last
        self._last_host_shape = host.shape
        self._copy_key = key

    def update(self, host: torch.Tensor, *, force: bool = False) -> None:
        flat = host.reshape(-1)
        n = flat.numel()
        if not force and self.matches(host):
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
        self.record(host)
