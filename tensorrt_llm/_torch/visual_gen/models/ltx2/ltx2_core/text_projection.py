# Ported from https://github.com/Lightricks/LTX-2
# packages/ltx-core/src/ltx_core/model/transformer/text_projection.py

import torch


class PixArtAlphaTextProjection(torch.nn.Module):
    """Projects caption embeddings (from PixArt-Alpha)."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int | None = None,
        act_fn: str = "gelu_tanh",
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = torch.nn.Linear(
            in_features=in_features, out_features=hidden_size, bias=True
        )
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = torch.nn.Linear(
            in_features=hidden_size, out_features=out_features, bias=True
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
