"""Small regression head that predicts shot duration from frozen Connector outputs.
The audio and video Connectors in ``embeddings_connector.py`` consume a Gemma caption
embedding and emit per-token hidden states. Their output dimension and sequence length
are fixed by the production model, so a thin regression head on top has all the signal
it needs to predict the natural duration of the implied shot -- without running the full
diffusion pipeline.
The head is modality-agnostic: pass either or both of {audio, video} connector outputs.
Modality-specific input projections map each stream to a shared pooler hidden dim,
learnable modality embeddings tag the streams so an attention pooler can tell them
apart, and a small MLP turns the pooled vector into a log-duration prediction. Targets
(when training) are in log-seconds so the loss spreads its budget evenly across orders
of magnitude; at inference time callers exponentiate the prediction to get seconds.
Unlike the training-time head, this module never sees packed/multi-sample sequences, so
it has no segment-id bookkeeping. It also needs no attention mask: the connector always
substitutes learnable registers for padded positions and marks the result as fully
attendable, so every token handed to this module is already valid.
"""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.disposable import Disposable


class AttentionPooler(nn.Module):
    """Cross-attend ``num_queries`` learnable tokens against ``tokens``.
    Produces a fixed-shape ``(B, num_queries, hidden_dim)`` output regardless of the
    input sequence length. Every position in ``tokens`` is attendable -- see the module
    docstring for why no mask is needed.
    """

    def __init__(self, hidden_dim: int = 256, num_queries: int = 1, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return pooled


class DurationHead(nn.Module, Disposable):
    """Predict log-duration in seconds from one or both Connector outputs.
    Args:
        video_cross_attention_dim: Video connector output dim.
        audio_cross_attention_dim: Audio connector output dim.
        pooler_hidden_dim: Shared hidden dim both modalities are projected into.
        num_queries: Number of learnable pooling queries.
        num_pooler_heads: Attention heads used by the pooler.
        mlp_hidden: Hidden width of the output MLP.
    """

    def __init__(
        self,
        video_cross_attention_dim: int = 4096,
        audio_cross_attention_dim: int = 2048,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.pooler_hidden_dim = pooler_hidden_dim

        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)

        self.attention_pooler = AttentionPooler(
            hidden_dim=pooler_hidden_dim,
            num_queries=num_queries,
            num_heads=num_pooler_heads,
        )
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict duration in seconds.
        The regression target is trained in log-seconds (spreads the loss budget evenly
        across orders of magnitude -- see the module docstring), so the MLP output is a
        log-duration internally; ``exp`` is applied here so callers always get seconds.
        Args:
            video_tokens: ``(B, T_v, video_cross_attention_dim)``, or ``None``.
            audio_tokens: ``(B, T_a, audio_cross_attention_dim)``, or ``None``.
        Returns:
            Duration prediction in seconds, shape ``(B,)``.
        """
        if video_tokens is None and audio_tokens is None:
            raise ValueError("DurationHead.forward requires at least one of video_tokens / audio_tokens")

        token_groups: list[torch.Tensor] = []
        if video_tokens is not None:
            token_groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            token_groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)

        tokens = torch.cat(token_groups, dim=1)
        pooled = self.attention_pooler(tokens)
        pooled_flat = pooled.reshape(pooled.shape[0], -1)
        hidden = torch.nn.functional.gelu(self.mlp_hidden(pooled_flat), approximate="tanh")
        log_duration = self.mlp_out(hidden).squeeze(-1)
        return log_duration.exp()
