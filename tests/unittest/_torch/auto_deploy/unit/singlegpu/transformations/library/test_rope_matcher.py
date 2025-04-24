import pytest
import torch
from _graph_test_helpers import run_test
from _model_test_utils import (
    apply_rotary_pos_emb_complex,
    apply_rotary_pos_emb_ds,
    apply_rotary_pos_emb_explicit,
)
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.transformations.library.rope import (
    match_complex_rope,
    match_explicit_rope,
    optimize_rope,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

torch.manual_seed(0)


def _precompute_freqs_cis_explicit(seq_len: int, head_dim: int, rope_theta: float):
    dtype = torch.float32
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def _precompute_freqs_cis_complex(seq_len: int, head_dim: int, rope_theta: float):
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


class RoPEModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_seq_len: int,
        num_heads: int,
        num_kv_heads: int,
        variant: str = "explicit",  # "explicit" or "complex"
        mode: str = "match",  # "match" or "optimize"
        layout: str = "BNSD",  # only for explicit variants
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.variant = variant
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.layout = layout if variant == "explicit" else None

        self.linear_q = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.linear_k = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.linear_q(x)
        k = self.linear_k(x)

        if self.variant == "explicit":
            # reshape and permute if BNSD layout
            q = q.view(b, s, self.num_heads, self.head_dim)
            k = k.view(b, s, self.num_kv_heads, self.head_dim)
            if self.layout == "BNSD":
                q = q.permute(0, 2, 1, 3).contiguous()
                k = k.permute(0, 2, 1, 3).contiguous()
                unsq_dim = 1
            else:
                unsq_dim = 2

            cos, sin = _precompute_freqs_cis_explicit(s, self.head_dim, rope_theta=10000)
            cos = cos.to(x.device).unsqueeze(0).expand(b, -1, -1)
            sin = sin.to(x.device).unsqueeze(0).expand(b, -1, -1)

            if self.mode == "match":
                q_out, k_out = apply_rotary_pos_emb_explicit(q, k, cos, sin, unsq_dim)
            else:  # optimize
                q_out, k_out = torch.ops.rope.torch_apply_explicit_rope(q, k, cos, sin, unsq_dim)

            # revert layout and flatten
            if self.layout == "BNSD":
                q_out = q_out.permute(0, 2, 1, 3).reshape(b, s, -1)
                k_out = k_out.permute(0, 2, 1, 3).reshape(b, s, -1)
            else:
                q_out = q_out.reshape(b, s, -1)
                k_out = k_out.reshape(b, s, -1)

        else:  # complex variant
            q = q.view(b, s, self.num_heads, self.head_dim)
            k = k.view(b, s, self.num_kv_heads, self.head_dim)
            freqs = _precompute_freqs_cis_complex(s, self.head_dim, rope_theta=10000)
            freqs = freqs.to(x.device).unsqueeze(0).expand(b, -1, -1)

            if self.mode == "match":
                q_out, k_out = apply_rotary_pos_emb_complex(q, k, freqs)
            else:
                q_out, k_out = torch.ops.rope.torch_apply_complex_rope(q, k, freqs)

            q_out = q_out.reshape(b, s, -1)
            k_out = k_out.reshape(b, s, -1)

        out = torch.cat([q_out, k_out], dim=-1)
        return out.to(torch.float16) if self.mode == "match" else out

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=16)}


@pytest.mark.parametrize(
    "transformation,variant,layout,batch_size,seq_len,num_heads,num_kv_heads,atol,rtol",
    [
        ("match", "explicit", "BNSD", 8, 16, 8, 8, 1e-3, 1e-3),
        ("match", "explicit", "BSND", 8, 16, 8, 4, 1e-2, 1e-2),
        ("match", "complex", None, 8, 16, 8, 8, 1e-3, 1e-3),
        ("match", "complex", None, 8, 16, 8, 4, 1e-3, 1e-3),
        ("optimize", "explicit", "BNSD", 4, 12, 8, 8, 1e-3, 1e-3),
        ("optimize", "explicit", "BSND", 4, 12, 8, 4, 1e-3, 1e-3),
        ("optimize", "complex", None, 4, 12, 8, 8, 1e-3, 1e-3),
        ("optimize", "complex", None, 4, 12, 8, 4, 1e-3, 1e-3),
    ],
)
@torch.inference_mode()
def test_rope_variants(
    transformation,
    variant,
    layout,
    batch_size,
    seq_len,
    num_heads,
    num_kv_heads,
    atol,
    rtol,
):
    hidden_size = 512
    model = RoPEModel(
        hidden_size,
        seq_len,
        num_heads,
        num_kv_heads,
        variant=variant,
        mode=transformation,
        layout=layout or "BNSD",
    ).to("cuda", torch.float16)
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16)
    dyn = model.get_dynamic_shapes()

    if transformation == "match":
        fn = match_explicit_rope if variant == "explicit" else match_complex_rope
        check_op = (
            torch.ops.rope.torch_apply_explicit_rope
            if variant == "explicit"
            else torch.ops.rope.torch_apply_complex_rope
        )

        def checker(gm):
            return any(is_op(n, check_op) for n in gm.graph.nodes)

    else:
        fn = optimize_rope

        def checker(gm):
            return any(is_op(n, torch.ops.rope.flashinfer) for n in gm.graph.nodes)

    _ = run_test(
        model,
        x,
        fn,
        checker,
        lambda n: n,
        atol=atol,
        rtol=rtol,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dyn,
    )


class DSRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # returns [seq_len, head_dim] cos & sin
        return self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype)


class DSModel(torch.nn.Module):
    def __init__(self, hidden_size, max_seq, n_head, n_kv, layout="BNSD"):
        super().__init__()
        self.hdim = hidden_size // n_head
        self.layout = layout
        self.q_lin = torch.nn.Linear(hidden_size, n_head * self.hdim)
        self.k_lin = torch.nn.Linear(hidden_size, n_kv * self.hdim)
        self.rotary = DSRotaryEmbedding(self.hdim, max_seq, base=10000, device="cuda")

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_lin(x).view(b, s, -1, self.hdim)
        k = self.k_lin(x).view(b, s, -1, self.hdim)
        if self.layout == "BNSD":
            # to [B, N, S, D]
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            unsq_dim = 1
        else:
            unsq_dim = 2
        cos, sin = self.rotary(x, seq_len=s)
        # build position_ids [B, S]
        pos_ids = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        q_out, k_out = apply_rotary_pos_emb_ds(q, k, cos, sin, pos_ids, unsqueeze_dim=unsq_dim)
        if self.layout == "BNSD":
            # back to [B, S, N*D]
            q_out = q_out.permute(0, 2, 1, 3).reshape(b, s, -1)
            k_out = k_out.permute(0, 2, 1, 3).reshape(b, s, -1)
        else:
            q_out = q_out.reshape(b, s, -1)
            k_out = k_out.reshape(b, s, -1)
        return torch.cat([q_out, k_out], dim=-1)

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=16)}


@pytest.mark.parametrize("layout,num_heads,num_kv_heads", [("BNSD", 8, 8), ("BSND", 8, 4)])
@torch.inference_mode()
def test_match_rope_deepseek(layout, num_heads, num_kv_heads):
    batch, seq, hid = 4, 12, 512
    model = DSModel(hid, 16, num_heads, num_kv_heads, layout=layout).to("cuda", torch.float16)
    x = torch.randn(batch, seq, hid, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()

    _ = run_test(
        model,
        x,
        match_explicit_rope,
        lambda gm: any(
            is_op(n, torch.ops.rope.torch_apply_rope_with_qk_interleaving) for n in gm.graph.nodes
        ),
        lambda num_p: num_p,
        atol=1e-3,
        rtol=1e-3,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )
