# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch


class PixArtAlphaTextProjection(torch.nn.Module):
    """Projects caption embeddings (from PixArt-Alpha)."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
        make_linear=None,
    ):
        super().__init__()
        if make_linear is None:
            make_linear = torch.nn.Linear
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = make_linear(in_features, hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = make_linear(hidden_size, out_features, bias=True)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
