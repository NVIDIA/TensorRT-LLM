# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Install TRTLLM linear factories before checkpoint materialization."""

import torch


def use_linear_factory(module, factory):
    if factory is None:
        return
    for name, child in module.named_children():
        if type(child) is torch.nn.Linear:
            setattr(
                module,
                name,
                factory(child.in_features, child.out_features, bias=child.bias is not None),
            )
        else:
            use_linear_factory(child, factory)
