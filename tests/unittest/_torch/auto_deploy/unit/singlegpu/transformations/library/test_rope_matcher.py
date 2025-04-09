import pytest
import torch
from _graph_test_helpers import run_test
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.transformations.library.rope import (
    match_rope_v1,
    match_rope_v2,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

torch.manual_seed(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _precompute_freqs_cis(seq_len: int, head_dim: int, rope_theta: float):
    dtype = torch.float32
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # Expected shape: (B, seq, head_dim//2) and complex dtype.
):
    # Reshape the inputs to pair the last dimension.
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # Multiply with frequencies. Note that freqs_cis is expected to broadcast with an extra head dim.
    xq_out = torch.view_as_real(xq_complex * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _precompute_freqs_cis_v2(seq_len: int, head_dim: int, rope_theta: float):
    """
    Compute the frequency tensor for the complex multiplication RoPE variant.
    Returns a complex tensor of shape (seq_len, head_dim//2).
    """
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim // 2, dtype=torch.float32) / (head_dim // 2))
    )
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (seq_len, head_dim//2)
    # Create a complex tensor from magnitude=1 and the computed angles.
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


class RotaryModel(torch.nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int, layout: str = "BNSD"):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.layout = layout
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.linear(x)
        k = self.linear(x)
        # Simulate a single-head scenario by unsqueezing the head dimension.
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)  # [B, 1, S, D]
        batch, _, seq, hidden = q.shape
        if self.layout == "BSND":
            # Transpose to [B, S, N, D]
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            unsqueeze_dim = 2
        else:
            # For BNSD, layout remains [B, N, S, D]
            unsqueeze_dim = 1

        # Precompute cosine-sine cache. shape: [max_seq_len, hidden].
        cos, sin = _precompute_freqs_cis(seq, hidden, rope_theta=10000)
        cos = cos.to(q.device)
        sin = sin.to(q.device)
        cos = cos.unsqueeze(0).expand(batch, -1, -1)  # [B, max_seq_len, hidden]
        sin = sin.unsqueeze(0).expand(batch, -1, -1)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return (q_embed + k_embed).to(torch.float16)

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=16)}


class RotaryModelV2(torch.nn.Module):
    def __init__(self, hidden_size: int, max_seq_len: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.linear(x)
        k = self.linear(x)
        q = q.unsqueeze(2)
        k = k.unsqueeze(2)  # [B, S, 1, D]
        batch, seq, n_head, hidden = q.shape

        # Precompute the complex frequency tensor.
        freqs_cis = _precompute_freqs_cis_v2(seq, hidden, rope_theta=10000)
        freqs_cis = freqs_cis.to(q.device)  # [seq, hidden//2]
        # Expand to batch dimension; expected shape: [B, seq, hidden//2]
        freqs_cis = freqs_cis.unsqueeze(0).expand(batch, -1, -1)

        q_embed, k_embed = apply_rotary_emb(q, k, freqs_cis)
        out = (q_embed + k_embed).to(torch.float16)
        return out

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=16)}


@pytest.mark.parametrize(
    "layout",
    ["BNSD", "BSND"],
)
@torch.inference_mode()
def test_match_rope(layout):
    batch_size, seq_len = 8, 16
    hidden_size = 64
    max_position_embeddings = seq_len

    model = RotaryModel(hidden_size, max_position_embeddings, layout=layout).to(
        "cuda", dtype=torch.float16
    )
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    _ = run_test(
        model,
        x,
        match_rope_v1,
        lambda gm: any(is_op(n, torch.ops.rope.flashinfer) for n in gm.graph.nodes),
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )


@torch.inference_mode()
def test_match_rope_v2():
    batch_size, seq_len = 8, 16
    hidden_size = 64
    max_position_embeddings = seq_len

    model = RotaryModelV2(hidden_size, max_position_embeddings).to("cuda", dtype=torch.float16)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    _ = run_test(
        model,
        x,
        match_rope_v2,
        lambda gm: any(is_op(n, torch.ops.rope.flashinfer) for n in gm.graph.nodes),
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )
