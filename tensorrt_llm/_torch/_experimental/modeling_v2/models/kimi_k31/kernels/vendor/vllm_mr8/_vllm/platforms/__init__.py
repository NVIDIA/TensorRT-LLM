# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vllm.platforms stand-in (CUDA only)."""
import torch


class _Cuda:
    device_type = "cuda"
    device_name = "cuda"

    @staticmethod
    def current_device():
        return torch.cuda.current_device()

    @staticmethod
    def is_rocm():
        return False

    @staticmethod
    def is_cuda():
        return True

    @staticmethod
    def get_device_capability(device_id=0):
        return torch.cuda.get_device_capability(device_id)


current_platform = _Cuda()
