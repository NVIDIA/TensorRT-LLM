from enum import Enum

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.common.normalization import NormType, build_normalization_layer


class AttentionType(Enum):
    """Enum for specifying the attention mechanism type."""

    VANILLA = "vanilla"
    LINEAR = "linear"
    NONE = "none"


class AttnBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # b,hw,c
        k = k.reshape(b, c, h * w).contiguous()  # b,c,hw
        w_ = torch.bmm(q, k).contiguous()  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w).contiguous()
        w_ = w_.permute(0, 2, 1).contiguous()  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_).contiguous()  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w).contiguous()

        h_ = self.proj_out(h_)

        return x + h_


def make_attn(
    in_channels: int,
    attn_type: AttentionType = AttentionType.VANILLA,
    norm_type: NormType = NormType.GROUP,
) -> torch.nn.Module:
    match attn_type:
        case AttentionType.VANILLA:
            return AttnBlock(in_channels, norm_type=norm_type)
        case AttentionType.NONE:
            return torch.nn.Identity()
        case AttentionType.LINEAR:
            raise NotImplementedError(f"Attention type {attn_type.value} is not supported yet.")
        case _:
            raise ValueError(f"Unknown attention type: {attn_type}")
