"""Utilities for building 2D self-attention masks for conditioning items."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.types import LatentState


def resolve_cross_mask(
    attention_mask: float | int | torch.Tensor,
    num_new_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an attention_mask (scalar or tensor) to a (B, M) cross_mask tensor.
    Args:
        attention_mask: Scalar value applied uniformly, 1D tensor of shape (M,)
            broadcast across batch, or 2D tensor of shape (B, M).
        num_new_tokens: Number of new conditioning tokens M.
        batch_size: Batch size B.
        device: Device for the output tensor.
        dtype: Data type for the output tensor.
    Returns:
        Cross-mask tensor of shape (B, M).
    """
    if isinstance(attention_mask, (int, float)):
        return torch.full(
            (batch_size, num_new_tokens),
            fill_value=float(attention_mask),
            device=device,
            dtype=dtype,
        )
    mask = attention_mask.to(device=device, dtype=dtype)

    # Handle scalar (0-D) tensor like a Python scalar.
    if mask.dim() == 0:
        return torch.full(
            (batch_size, num_new_tokens),
            fill_value=float(mask.item()),
            device=device,
            dtype=dtype,
        )

    if mask.dim() == 1:
        if mask.shape[0] != num_new_tokens:
            raise ValueError(
                f"1-D attention_mask length must equal num_new_tokens ({num_new_tokens}), got shape {tuple(mask.shape)}"
            )
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    elif mask.dim() == 2:
        b, m = mask.shape
        if m != num_new_tokens:
            raise ValueError(
                f"2-D attention_mask second dimension must equal num_new_tokens ({num_new_tokens}), "
                f"got shape {tuple(mask.shape)}"
            )
        if b not in (batch_size, 1):
            raise ValueError(
                f"2-D attention_mask batch dimension must equal batch_size ({batch_size}) or 1, "
                f"got shape {tuple(mask.shape)}"
            )
        if b == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
    else:
        raise ValueError(
            f"attention_mask tensor must be 0-D, 1-D, or 2-D, got {mask.dim()}-D with shape {tuple(mask.shape)}"
        )
    return mask


def extend_keyframes_mask(
    latent_state: LatentState,
    num_new_tokens: int,
    *,
    marked: bool,
) -> torch.Tensor | None:
    """Extend :attr:`LatentState.keyframes_mask` to cover newly appended tokens.
    Every conditioning item that appends tokens must call this, otherwise the per-token marker
    goes out of sync with the token sequence -- or is dropped entirely by items that build a fresh
    :class:`LatentState` instead of using :func:`dataclasses.replace`.
    Args:
        latent_state: State being extended.
        num_new_tokens: Number of tokens the item appends.
        marked: Whether the new tokens encode a single standalone pixel frame in the *generated*
            stream. True only for generated keyframe slots; given-content conditioning tokens
            (image guidance, reference latents) are never marked, matching the reference
            implementation, which marks only the keyframe conditioning methods.
    Returns:
        The extended mask, or ``None`` when there was no mask and the new tokens are unmarked.
    """
    existing = latent_state.keyframes_mask
    if existing is None and not marked:
        return None
    if existing is None:
        existing = torch.zeros_like(latent_state.denoise_mask.tensor, device=latent_state.latent.device)
    new = torch.ones if marked else torch.zeros
    pad = new(
        (existing.shape[0], num_new_tokens, existing.shape[2]),
        device=existing.device,
        dtype=existing.dtype,
    )
    return torch.cat([existing, pad], dim=1)


def update_attention_mask(
    latent_state: LatentState,
    attention_mask: float | torch.Tensor | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build or update the self-attention mask for newly appended conditioning tokens.
    If *attention_mask* is ``None`` and no existing mask is present, returns
    ``None``.  If *attention_mask* is ``None`` but an existing mask is present,
    the mask is expanded with full attention (1s) for the new tokens so that
    its dimensions stay consistent with the growing latent sequence.  Otherwise,
    resolves *attention_mask* to a per-token cross-mask and expands the 2-D
    attention mask via :func:`build_attention_mask`.
    Args:
        latent_state: Current latent state (provides the existing mask and total
            existing-token count).
        attention_mask: Per-token attention weight. Scalar, 1-D ``(M,)``, 2-D
            ``(B, M)`` tensor, or ``None`` (no-op).
        num_noisy_tokens: Number of original noisy tokens (from
            ``latent_tools.target_shape.token_count()``).
        num_new_tokens: Number of new conditioning tokens being appended.
        batch_size: Batch size.
        device: Device for the output tensor.
        dtype: Data type for the output tensor.
    Returns:
        Updated attention mask of shape ``(B, N+M, N+M)``, or ``None`` if no
        masking is needed.
    """
    if attention_mask is None:
        if latent_state.attention_mask is None:
            return None
        # Existing mask present but no new mask requested: pad with 1s (full
        # attention) so the mask dimensions stay consistent with the growing
        # latent sequence.
        cross_mask = torch.ones(batch_size, num_new_tokens, device=device, dtype=dtype)
        return build_attention_mask(
            existing_mask=latent_state.attention_mask,
            num_noisy_tokens=num_noisy_tokens,
            num_new_tokens=num_new_tokens,
            num_existing_tokens=latent_state.latent.shape[1],
            cross_mask=cross_mask,
            device=device,
            dtype=dtype,
        )

    cross_mask = resolve_cross_mask(attention_mask, num_new_tokens, batch_size, device, dtype)
    return build_attention_mask(
        existing_mask=latent_state.attention_mask,
        num_noisy_tokens=num_noisy_tokens,
        num_new_tokens=num_new_tokens,
        num_existing_tokens=latent_state.latent.shape[1],
        cross_mask=cross_mask,
        device=device,
        dtype=dtype,
    )


def build_attention_mask(
    existing_mask: torch.Tensor | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    num_existing_tokens: int,
    cross_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Expand the attention mask to include newly appended conditioning tokens.
    Each conditioning item appends M new reference tokens to the sequence. This function
    builds a (B, N+M, N+M) attention mask with the following block structure:
                     noisy      prev_ref    new_ref
                   (N_noisy)   (N-N_noisy)    (M)
                 ┌───────────┬───────────┬───────────┐
        noisy    │           │           │           │
       (N_noisy) │  existing │  existing │   cross   │
                 │           │           │           │
                 ├───────────┼───────────┼───────────┤
       prev_ref  │           │           │           │
      (N-N_noisy)│  existing │  existing │     0     │
                 │           │           │           │
                 ├───────────┼───────────┼───────────┤
       new_ref   │           │           │           │
         (M)     │   cross   │     0     │     1     │
                 │           │           │           │
                 └───────────┴───────────┴───────────┘
    Where:
      - **existing**: preserved from the previous mask (or 1.0 if first conditioning)
      - **cross**: values from *cross_mask* (shape B, M), in [0, 1]
      - **0**: no attention between different reference groups
    Args:
        existing_mask: Current attention mask of shape (B, N, N), or None if no mask exists yet.
            When None, the top-left NxN block is filled with 1s (full attention between all
            existing tokens including any prior reference tokens that had no mask).
        num_noisy_tokens: Number of original noisy tokens (always at positions [0:num_noisy_tokens]).
        num_new_tokens: Number of new conditioning tokens M being appended.
        num_existing_tokens: Total number of current tokens N (noisy + any prior conditioning tokens).
        cross_mask: Per-token attention weight of shape (B, M) controlling attention between
            new reference tokens and noisy tokens. Values in [0, 1].
        device: Device for the output tensor.
        dtype: Data type for the output tensor.
    Returns:
        Attention mask of shape (B, N+M, N+M) with values in [0, 1].
    """
    batch_size = cross_mask.shape[0]
    total = num_existing_tokens + num_new_tokens

    # Start with zeros
    mask = torch.zeros((batch_size, total, total), device=device, dtype=dtype)

    # Top-left: preserve existing mask or fill with 1s for noisy tokens
    if existing_mask is not None:
        mask[:, :num_existing_tokens, :num_existing_tokens] = existing_mask
    else:
        mask[:, :num_existing_tokens, :num_existing_tokens] = 1.0

    # Bottom-right: new reference tokens fully attend to themselves
    mask[:, num_existing_tokens:, num_existing_tokens:] = 1.0

    # Cross-attention between noisy tokens and new reference tokens
    # cross_mask shape: (B, M) -> broadcast to (B, N_noisy, M) and (B, M, N_noisy)

    # Noisy tokens attending to new reference tokens: [0:N_noisy, N:N+M]
    # Each column j in this block gets cross_mask[:, j]
    mask[:, :num_noisy_tokens, num_existing_tokens:] = cross_mask.unsqueeze(1)

    # New reference tokens attending to noisy tokens: [N:N+M, 0:N_noisy]
    # Each row i in this block gets cross_mask[:, i]
    mask[:, num_existing_tokens:, :num_noisy_tokens] = cross_mask.unsqueeze(2)

    # [N_noisy:N, N:N+M] and [N:N+M, N_noisy:N] remain 0 (no cross-ref attention)

    return mask
