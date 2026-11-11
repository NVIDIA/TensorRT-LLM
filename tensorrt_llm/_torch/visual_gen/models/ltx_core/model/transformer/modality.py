from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenTimesteps:
    """Timestep values and the optional per-token index into their compact table.
    ``values`` is ``(B, T)`` when dense or ``(B, K)`` when ``indices`` supplies
    the ``(B, T)`` token layout. ``zero_index`` is the column of a compact table known, on the
    host, to hold timestep zero in every row, or ``None`` when the table has no such column.
    """

    values: torch.Tensor
    indices: torch.Tensor | None = None
    zero_index: int | None = None

    def per_token(self) -> torch.Tensor:
        """The timestep of each token, with a trailing singleton dimension."""
        if self.indices is None:
            return self.values.unsqueeze(-1) if self.values.ndim == 2 else self.values
        return torch.gather(self.values, 1, self.indices.to(torch.int64)).unsqueeze(-1)

    def select_tokens(self, index: slice | torch.Tensor) -> TokenTimesteps:
        if self.indices is None:
            return TokenTimesteps(self.values[:, index])
        return TokenTimesteps(self.values, self.indices[:, index], self.zero_index)

    def pad_tokens(self, pad: int) -> TokenTimesteps:
        """Append ``pad`` tokens with timestep zero.
        Padded tokens stay visible as keys to video-to-audio cross-attention, so their timestep
        reaches real outputs and must match dense padding. An indexed table keeps its shape when
        ``zero_index`` names a zero column, and the padded tokens point at it; otherwise it gains
        one zero column, which becomes ``zero_index``.
        """
        if self.indices is None:
            pad_shape = list(self.values.shape)
            pad_shape[1] = pad
            return TokenTimesteps(torch.cat([self.values, self.values.new_zeros(pad_shape)], dim=1))
        values, zero_index = self.values, self.zero_index
        if zero_index is None:
            values = torch.cat([values, values.new_zeros(values.shape[0], 1)], dim=1)
            zero_index = self.values.shape[1]
        pad_indices = self.indices.new_full((self.indices.shape[0], pad), zero_index)
        return TokenTimesteps(values, torch.cat([self.indices, pad_indices], dim=1), zero_index)

    def split(self, sizes: list[int]) -> list[TokenTimesteps]:
        values = self.values.split(sizes, dim=0)
        indices = self.indices.split(sizes, dim=0) if self.indices is not None else [None] * len(sizes)
        return [
            TokenTimesteps(chunk_values, chunk_indices, self.zero_index)
            for chunk_values, chunk_indices in zip(values, indices, strict=True)
        ]


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    Attributes:
        latent: Patchified latent tokens, shape ``(B, T, D)`` where *B* is
            the batch size, *T* is the total number of tokens (noisy +
            conditioning), and *D* is the input dimension.
        timesteps: Per-token timestep values or a compact table and its token indices.
        positions: Per-token patch coordinates used to build the RoPE
            frequencies. With the default ``use_middle_indices_grid=True``,
            shape is ``(B, n_pos_dims, T, 2)`` where ``n_pos_dims=3`` for
            video (time, height, width) and ``n_pos_dims=1`` for audio
            (time); the last dim of size 2 holds the ``[start, end)``
            index bounds of each patch, and RoPE is evaluated at the
            *middle* of that range -- hence the flag name. Taking the
            patch midpoint produces a smoother and more accurate
            positional signal than indexing by the patch's start when
            patches span more than one spatial / temporal unit.
            When ``use_middle_indices_grid=False``, the legacy 3-D form
            ``(B, n_pos_dims, T)`` of integer positional indices is
            accepted instead and used as-is (no midpoint derivation).
        context: Text conditioning embeddings from the prompt encoder.
        enabled: Whether this modality is active in the current forward pass.
        context_mask: Optional mask for the text context tokens.
        attention_mask: Optional 2-D self-attention mask, shape ``(B, T, T)``.
            Values in ``[0, 1]`` where ``1`` = full attention and ``0`` = no
            attention. ``None`` means unrestricted (full) attention between
            all tokens. Built incrementally by conditioning items; see
            :class:`~ltx_core.conditioning.types.attention_strength_wrapper.ConditioningItemAttentionStrengthWrapper`.
        keyframes_mask: Optional per-token marker, shape ``(B, T, 1)`` -- aligned with the latent
            token axis -- non-zero for tokens whose latent encodes a *single standalone pixel
            frame* rather than the usual multi-frame span. That is the target's first latent frame
            (the video encoder is causal) plus any generated keyframe slots. Selects the tokens
            that receive the model's learned keyframe absolute-position embedding. ``None`` means
            no token is marked, which is also the effective behaviour of every model built without
            ``use_keyframes_abs_pos_embedding``.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    sigma: torch.Tensor  # Shape: (B,). Current sigma value, used for cross-attention timestep calculation.
    timesteps: TokenTimesteps
    # Shape: (B, n_pos_dims, T, 2) by default (use_middle_indices_grid=True);
    # n_pos_dims=3 for video, 1 for audio; last dim holds [start, end) patch bounds.
    # Legacy form (B, n_pos_dims, T) when use_middle_indices_grid=False.
    positions: torch.Tensor
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None  # Shape: (B, T, 1), non-zero on single-pixel-frame latents

    def split(self, sizes: list[int]) -> list[Modality]:
        """Split along the batch dimension into chunks of the given sizes."""
        n = len(sizes)
        split_fields: dict[str, list[torch.Tensor | TokenTimesteps | None] | list[bool]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                split_fields[f.name] = list(value.split(sizes, dim=0))
            elif isinstance(value, TokenTimesteps):
                split_fields[f.name] = value.split(sizes)
            elif value is None or isinstance(value, bool):
                split_fields[f.name] = [value] * n
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(n)]
