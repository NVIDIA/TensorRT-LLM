from typing import List, Tuple

import torch
from torch import nn

from ..attention_backend.interface import RopeParams
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        rope_params: RopeParams,
        *,
        head_dim: int,
        is_neox: bool = True,
    ):
        super().__init__()
        self.rope_params = rope_params
        self.head_dim = head_dim
        self.is_neox = is_neox
        self.max_positions = rope_params.max_positions
        self.rotary_cos_sin = rope_params.create_rope_const_params(
            interleave=False)[1].reshape(rope_params.max_positions, 2, -1)

    def forward(
        self,
        position_ids: torch.Tensor,
        targets: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply RoPE to any number of target tensors with the same position_ids.
        This is useful if q_len = k_len, in which case we may RoPE q and k with the same cos and sin values.
        However, if k is cached without positional embedding, we need to apply rope to q and k with different values, so we need a separate call for each.
        """
        if IS_FLASHINFER_AVAILABLE and len(targets) == 2:
            from ..custom_ops import \
                flashinfer_apply_rope_with_cos_sin_cache_inplace
            q = targets[0]
            k = targets[1]
            flashinfer_apply_rope_with_cos_sin_cache_inplace(
                position_ids.view(-1),
                q,
                k,
                self.head_dim,
                self.rotary_cos_sin.view(self.max_positions, -1),
                self.is_neox,
            )
            return [q, k]

        # it is assumed all targets are of the same rank
        q_or_k = targets[0]
        remove_input_padding = (len(q_or_k.size()) == 2)

        cos_sin = self.rotary_cos_sin[position_ids.view(-1)]
        cos, sin = cos_sin[:, 0, :], cos_sin[:, 1, :]
        cos = cos.to(dtype=q_or_k.dtype).unsqueeze(0)
        sin = sin.to(dtype=q_or_k.dtype).unsqueeze(0)

        if remove_input_padding:
            bsz = 1
            seq_len, _ = q_or_k.size()
        else:
            bsz, seq_len, _ = q_or_k.size()

        def rope_target(target):
            target = target.view(bsz, seq_len, -1,
                                 self.head_dim).transpose(1, 2)
            target = RotaryEmbedding.apply_rotary_pos_emb(target,
                                                          cos,
                                                          sin,
                                                          is_neox=self.is_neox)
            target = target.transpose(1, 2).contiguous()
            if remove_input_padding:
                target = target.view(seq_len, -1)
            else:
                target = target.view(bsz, seq_len, -1)
            return target

        return [rope_target(target) for target in targets]

    @staticmethod
    def apply_rotary_pos_emb(q_or_k: torch.Tensor,
                             cos: torch.Tensor,
                             sin: torch.Tensor,
                             unsqueeze_dim: int = 1,
                             is_neox: bool = True) -> torch.Tensor:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q_or_k (`torch.Tensor`): The query/key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
            is_neox (bool): Whether to use Neox style RoPE, True by default.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        rot_dim = cos.shape[-1] * 2
        # If q_or_k_pass is empty, rotary pos embedding is applied to all tensor
        q_or_k, q_or_k_pass = q_or_k[..., :rot_dim], q_or_k[..., rot_dim:]

        if is_neox:
            x1 = q_or_k[..., :q_or_k.shape[-1] // 2]
            x2 = q_or_k[..., q_or_k.shape[-1] // 2:]
        else:
            x1 = q_or_k[..., ::2]
            x2 = q_or_k[..., 1::2]

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

        if is_neox:
            return torch.cat((o1, o2, q_or_k_pass), dim=-1)
        else:
            embed = torch.stack((o1, o2), dim=-1).flatten(-2)
            return torch.cat((embed, q_or_k_pass), dim=-1)


class MRotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        rope_params: RopeParams,
        *,
        head_dim: int,
        mrope_section: List[int],
        is_neox: bool = True,
    ):
        super().__init__(rope_params, head_dim=head_dim, is_neox=is_neox)
        self.mrope_section = mrope_section

    def get_cos_sin(
            self,
            position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 3:
            cos_sin = self.rotary_cos_sin[position_ids.view(3, -1)]
            cos, sin = cos_sin[:, :, 0, :], cos_sin[:, :, 1, :]
            cos = torch.cat([
                m[i]
                for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)
            sin = torch.cat([
                m[i]
                for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
            ],
                            dim=-1)
        else:
            # Fallback to the original RoPE where position_ids is 2D for dummy requests
            cos_sin = self.rotary_cos_sin[position_ids.view(-1)]
            cos, sin = cos_sin[:, 0, :], cos_sin[:, 1, :]

        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return cos, sin

    def forward(
        self,
        position_ids: torch.Tensor,
        targets: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        # it is assumed all targets are of the same rank
        q_or_k = targets[0]
        remove_input_padding = (len(q_or_k.size()) == 2)

        # TODO(yechank-nvidia): Re-visit this to achieve cos_sin caching
        cos, sin = self.get_cos_sin(position_ids)
        cos, sin = cos.to(dtype=q_or_k.dtype), sin.to(dtype=q_or_k.dtype)
        if remove_input_padding:
            bsz = 1
            seq_len, _ = q_or_k.size()
        else:
            bsz, seq_len, _ = q_or_k.size()

        def rope_target(target):
            target = target.view(bsz, seq_len, -1,
                                 self.head_dim).transpose(1, 2)
            target = RotaryEmbedding.apply_rotary_pos_emb(target,
                                                          cos,
                                                          sin,
                                                          is_neox=self.is_neox)
            target = target.transpose(1, 2).contiguous()
            if remove_input_padding:
                target = target.view(seq_len, -1)
            else:
                target = target.view(bsz, seq_len, -1)
            return target

        return [rope_target(target) for target in targets]
