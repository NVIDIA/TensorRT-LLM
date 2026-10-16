# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch helpers shared by the Vanilla sparse MLA golden references.

The DSA and DeepSeek-V4 references read TRTLLM-style attention metadata: a batch
packs its context requests before its generation requests, ``seq_lens`` holds
each request's new tokens and ``kv_lens_cuda`` its total KV length including
them. The helpers below derive phase bounds and token positions from those
fields, and rotate absorbed MLA tensors without dispatching the fused RoPE
kernels the references validate.
"""

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


class VanillaRotaryEmbedding(RotaryEmbedding):
    """Apply pure-PyTorch RoPE to packed, token-major tensors.

    Targets share token positions but may have different head counts or layouts:
    [T, H, R], [T, H * R], or [T, R] for a single head. Here T is the packed token
    count and R is this object's head_dim. Sparse MLA callers pass only the
    positional slices of Q/K, so R is the RoPE width, not the full attention or
    indexer width. The inherited constructor builds the cos/sin table from the
    model's RoPE configuration, including its frequency and scaling settings.

    For each dimension pair (x1, x2), using the cached cosine c and sine s for
    that token's position and that pair's frequency, the forward rotation is::

        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s

    With standard RoPE, c = cos(position * frequency) and
    s = sin(position * frequency). Applying these rotations to Q and K makes
    their dot product depend on relative position. Neox layout pairs the first
    and second halves of the rotated dimensions; interleaved layout pairs
    adjacent dimensions. inverse=True reverses the signs of the sine terms.

    RotaryEmbedding.forward can dispatch a Q/K pair to an in-place FlashInfer
    kernel, and its PyTorch fallback interprets rank-three inputs as
    [batch, sequence, features]. This wrapper explicitly adapts packed layouts
    and calls the static apply_rotary_pos_emb helper instead. That helper uses
    ordinary PyTorch operations without fused dispatch, so sharing its rotation
    math and layout conventions keeps the references independent of the fused
    kernels without duplicating the equations.
    """

    def forward(
        self, position_ids: torch.Tensor, targets: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Rotate targets with shared positions, preserving each target's shape.

        Args:
            position_ids: Integer positions flattening to [T], one per packed
                token. Positions can restart for each request in the batch.
            targets: Floating-point Q/K slices with T leading rows and one or
                more heads of width self.head_dim per row.

        Returns:
            Rotated tensors in input order, with each input's shape and dtype.
            Inputs are not modified in place.
        """
        position_ids = position_ids.reshape(-1)
        num_tokens = position_ids.numel()
        if num_tokens == 0:
            return targets

        # Look up positions once for all targets; packed row indices need not
        # equal token positions within their respective requests.
        cos_sin = self.rotary_cos_sin[position_ids]
        cos, sin = cos_sin[:, 0, :], cos_sin[:, 1, :]

        def apply_rope(target: torch.Tensor) -> torch.Tensor:
            original_shape = target.shape
            if target.shape[0] != num_tokens:
                raise ValueError(
                    "Packed RoPE targets must have one leading row per position: "
                    f"got target shape {tuple(target.shape)} and {num_tokens} positions"
                )
            # Normalize each layout to [T, H, R], then [1, H, T, R]. The helper
            # broadcasts the position-dependent cos/sin values across heads;
            # the leading singleton is a layout adapter, not a request count.
            target = target.reshape(num_tokens, -1, self.head_dim)
            target = target.transpose(0, 1).unsqueeze(0)
            target = RotaryEmbedding.apply_rotary_pos_emb(
                target,
                cos.to(dtype=target.dtype).unsqueeze(0),
                sin.to(dtype=target.dtype).unsqueeze(0),
                is_neox=self.is_neox,
                inverse=self.inverse,
            )
            return target.squeeze(0).transpose(0, 1).reshape(original_shape)

        return [apply_rope(target) for target in targets]


def apply_mla_rope(
    rotary_emb: VanillaRotaryEmbedding,
    fused_q: Optional[torch.Tensor],
    q_pe: Optional[torch.Tensor],
    latent_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads: int,
    kv_lora_rank: int,
    apply_q: bool = True,
    apply_k: bool = True,
) -> None:
    """Rotate the RoPE parts of an absorbed query and its latent rows in place.

    An absorbed query head is [kv_lora_rank + R] wide with the RoPE slice last,
    and a latent row is [kv_lora_rank + R] = [compressed KV | k_pe].

    Args:
        rotary_emb: Table and layout (R = rotary_emb.head_dim) to rotate with.
        fused_q: [T, num_heads * (kv_lora_rank + R)] absorbed query; the last
            R columns of each head receive the rotated ``q_pe``.
        q_pe: [T, num_heads, R] or [T, num_heads * R] unrotated query RoPE
            part. It may alias the RoPE slots of ``fused_q``.
        latent_cache: [T, kv_lora_rank + R] latent rows; the trailing R
            columns (k_pe) are rotated in place.
        positions: [T] absolute token positions.
        num_heads: Query heads in ``fused_q``.
        kv_lora_rank: Width of the non-positional part of a head and of a
            latent row.
        apply_q: Rotate the query side.
        apply_k: Rotate the latent k_pe.
    """
    num_tokens = latent_cache.shape[0]
    rope_dim = rotary_emb.head_dim
    targets = []
    if apply_q:
        if fused_q is None or q_pe is None:
            raise ValueError("Q-side MLA RoPE requires fused_q and q_pe")
        targets.append(q_pe.reshape(num_tokens, num_heads * rope_dim))
    if apply_k:
        targets.append(latent_cache[..., kv_lora_rank:])

    rotated = iter(rotary_emb(positions, targets))
    if apply_q:
        q_pe_rot = next(rotated)
        fused_q.view(num_tokens, num_heads, kv_lora_rank + rope_dim)[..., kv_lora_rank:] = (
            q_pe_rot.view(num_tokens, num_heads, rope_dim)
        )
    if apply_k:
        latent_cache[..., kv_lora_rank:] = next(rotated)


def cached_lens(metadata: "TrtllmAttentionMetadata") -> list[int]:
    """Return each request's KV length before this step's tokens are appended.

    ``kv_lens_cuda`` holds every request's total KV length including this
    step's tokens, so subtracting ``seq_lens`` (new tokens per request) leaves
    the cached prefix. That value is the position of the request's first new
    token, shared by RoPE positions, cache-append offsets and causal limits.
    """
    seq_lens = metadata.seq_lens.tolist()
    kv_lens = metadata.kv_lens_cuda[: metadata.num_seqs].tolist()
    return [kv_len - seq_len for kv_len, seq_len in zip(kv_lens, seq_lens, strict=True)]


def phase_bounds(
    metadata: "TrtllmAttentionMetadata", is_generation: Optional[bool]
) -> tuple[int, int, int, int]:
    """Return (seq_start, seq_end, token_start, token_end) in the packed batch.

    Bounds are half-open, with context requests/tokens before generation.
    None selects the whole batch; False/True selects context/generation.
    """
    if is_generation is None:
        return 0, metadata.num_seqs, 0, metadata.num_tokens
    if is_generation:
        return (
            metadata.num_contexts,
            metadata.num_seqs,
            metadata.num_ctx_tokens,
            metadata.num_tokens,
        )
    return 0, metadata.num_contexts, 0, metadata.num_ctx_tokens


def phase_lens(
    metadata: "TrtllmAttentionMetadata", is_generation: Optional[bool]
) -> tuple[int, int, list[int], list[int]]:
    """Return (seq_start, seq_end, seq_lens, past_lens) for the selected requests.

    ``seq_lens`` and ``past_lens`` hold the new and cached token counts of
    requests ``seq_start`` through ``seq_end - 1``, in packed batch order.
    """
    seq_start, seq_end, _, _ = phase_bounds(metadata, is_generation)
    seq_lens = metadata.seq_lens.tolist()[seq_start:seq_end]
    past_lens = cached_lens(metadata)[seq_start:seq_end]
    return seq_start, seq_end, seq_lens, past_lens


def token_positions(
    seq_lens: list[int], past_lens: list[int], device: torch.device
) -> torch.Tensor:
    """Return int32 [sum(seq_lens)] absolute positions of the new tokens.

    Request i contributes ``past_lens[i], ..., past_lens[i] + seq_lens[i] - 1``.
    These positions select RoPE angles and bound causal selections.
    """
    pieces = [
        torch.arange(past, past + seq_len, dtype=torch.int32, device=device)
        for seq_len, past in zip(seq_lens, past_lens, strict=True)
    ]
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int32, device=device)


def phase_is_generation(attention_input_type: AttentionInputType) -> bool:
    """Map a phase to ``is_generation``; the sparse MLA modules never run a mixed batch at once."""
    if attention_input_type == AttentionInputType.context_only:
        return False
    if attention_input_type == AttentionInputType.generation_only:
        return True
    raise ValueError("Sparse MLA requires a context-only or generation-only input")


__all__ = [
    "VanillaRotaryEmbedding",
    "apply_mla_rope",
    "cached_lens",
    "phase_bounds",
    "phase_is_generation",
    "phase_lens",
    "token_positions",
]
