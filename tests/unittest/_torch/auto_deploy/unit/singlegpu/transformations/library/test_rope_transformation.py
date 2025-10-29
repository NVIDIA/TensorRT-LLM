import pytest
import torch
from _graph_test_helpers import run_test_transformed_gm
from _model_test_utils import (
    apply_rotary_pos_emb_complex,
    apply_rotary_pos_emb_ds,
    apply_rotary_pos_emb_explicit,
)
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_output_tuple, is_op

torch.manual_seed(0)


def _precompute_freqs_cis_explicit(
    seq_len: int, head_dim: int, rope_theta: float, dtype: torch.dtype = torch.float32
):
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
        layout: str = "BNSD",  # "BNSD" or "BSND"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.variant = variant
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.layout = layout

        self.linear_q = torch.nn.Linear(hidden_size, num_heads * self.head_dim)
        self.linear_k = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.linear_q(x)
        k = self.linear_k(x)

        if self.variant == "explicit" or self.variant == "explicit_pm":
            # reshape and permute if BNSD layout
            q = q.view(b, s, self.num_heads, self.head_dim)
            k = k.view(b, s, self.num_kv_heads, self.head_dim)
            if self.layout == "BNSD":
                q = q.permute(0, 2, 1, 3).contiguous()
                k = k.permute(0, 2, 1, 3).contiguous()
                unsq_dim = 1
            else:
                unsq_dim = 2

            cos, sin = _precompute_freqs_cis_explicit(
                s, self.head_dim, rope_theta=10000, dtype=x.dtype
            )
            cos = cos.to(x.device).unsqueeze(0).expand(b, -1, -1)
            sin = sin.to(x.device).unsqueeze(0).expand(b, -1, -1)

            if self.mode == "match":
                q_out, k_out = apply_rotary_pos_emb_explicit(q, k, cos, sin, unsq_dim)
            else:  # optimize
                q_out, k_out = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
                    q, k, cos, sin, unsq_dim
                )

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
            if self.layout == "BNSD":
                q = q.permute(0, 2, 1, 3).contiguous()
                k = k.permute(0, 2, 1, 3).contiguous()
                unsq_dim = 1
            else:
                unsq_dim = 2

            freqs = _precompute_freqs_cis_complex(s, self.head_dim, rope_theta=10000)
            freqs = freqs.to(x.device).unsqueeze(0).expand(b, -1, -1)

            if self.mode == "match":
                q_out, k_out = apply_rotary_pos_emb_complex(q, k, freqs, unsq_dim)
            else:
                q_out, k_out = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
                    q, k, freqs, unsq_dim
                )

            # revert layout and flatten
            if self.layout == "BNSD":
                q_out = q_out.permute(0, 2, 1, 3).reshape(b, s, -1)
                k_out = k_out.permute(0, 2, 1, 3).reshape(b, s, -1)
            else:
                q_out = q_out.reshape(b, s, -1)
                k_out = k_out.reshape(b, s, -1)

        out = torch.cat([q_out, k_out], dim=-1)
        return out.to(torch.float16) if self.mode == "match" else out

    def get_dynamic_shapes(self):
        return {0: Dim("batch_size", max=8), 1: Dim("seq_len", max=16)}


@pytest.mark.parametrize(
    "transformation,variant,layout,batch_size,seq_len,num_heads,num_kv_heads,atol,rtol, target_layout",
    [
        ("match", "explicit", "BNSD", 8, 16, 8, 8, 1e-2, 1e-2, None),
        ("match", "explicit", "BSND", 8, 16, 8, 4, 1e-2, 1e-2, None),
        ("match", "complex", "BNSD", 8, 16, 8, 8, 1e-3, 1e-3, None),
        ("match", "complex", "BSND", 8, 16, 8, 4, 1e-3, 1e-3, None),
        ("match_layout", "explicit", "BNSD", 4, 12, 8, 8, 1e-3, 1e-3, "BSND"),
        ("match_layout", "explicit", "BNSD", 4, 12, 8, 8, 1e-3, 1e-3, "BNSD"),
        ("match_layout", "complex", "BNSD", 4, 12, 8, 8, 1e-3, 1e-3, "BSND"),
        ("match_layout", "complex", "BSND", 4, 12, 8, 8, 1e-3, 1e-3, "BSND"),
        pytest.param(
            "optimize",
            "explicit",
            "BNSD",
            4,
            12,
            8,
            8,
            1e-3,
            1e-3,
            None,
            marks=pytest.mark.xfail(
                reason="flashinfer op does not support BNSD layout", strict=True
            ),
        ),
        ("optimize", "explicit", "BSND", 4, 12, 8, 4, 1e-3, 1e-3, None),
        pytest.param(
            "optimize",
            "complex",
            "BNSD",
            4,
            12,
            8,
            8,
            1e-3,
            1e-3,
            None,
            marks=pytest.mark.xfail(
                reason="flashinfer op does not support BNSD layout", strict=True
            ),
        ),
        ("optimize", "complex", "BSND", 4, 12, 8, 4, 1e-3, 1e-3, None),
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
    target_layout,
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
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dyn,), clone=True)

    if transformation == "match":
        gm_transformed = InferenceOptimizer(
            None,
            {
                "match_rope_pattern": {
                    "stage": "pattern_matcher",
                },
            },
        )(None, gm)
        gm_transformed.to("cuda")

        check_op = (
            torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin
            if variant == "explicit" or variant == "explicit_pm"
            else torch.ops.auto_deploy.torch_rope_with_complex_freqs
        )

        def checker(gm):
            return any(is_op(n, check_op) for n in gm.graph.nodes)

        run_test_transformed_gm(
            model,
            x,
            gm_transformed,
            checker,
            lambda n: n,
            atol,  # atol
            rtol,  # rtol
            True,  # test_load_hook
            True,  # strict_loading
            dyn,  # dynamic_shapes
            1,  # check_num_matches
            False,  # skip_output_assert
        )

    elif transformation == "match_layout":
        gm_transformed = InferenceOptimizer(
            None,
            {
                "match_rope_layout": {
                    "stage": "pattern_matcher",
                    "expected_layout": target_layout,
                },
            },
        )(None, gm)
        gm_transformed.to("cuda")

        def checker(gm):
            for n in gm.graph.nodes:
                if is_op(
                    n,
                    {
                        torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin,
                        torch.ops.auto_deploy.torch_rope_with_complex_freqs,
                    },
                ):
                    q_arg, k_arg, *rest = n.args
                    if not (
                        is_op(q_arg, torch.ops.aten.contiguous)
                        and is_op(k_arg, torch.ops.aten.contiguous)
                    ):
                        matched = False
                        break

                    old_q, old_k = extract_output_tuple(n, 2)
                    if old_q is None or old_k is None:
                        matched = False
                        break
                    q_transposed = any(is_op(u, torch.ops.aten.transpose) for u in old_q.users)
                    k_transposed = any(is_op(u, torch.ops.aten.transpose) for u in old_k.users)
                    matched = q_transposed and k_transposed

            return matched if layout != target_layout else not matched

        run_test_transformed_gm(
            model,
            x,
            gm_transformed,
            checker,
            lambda n: n,
            atol,  # atol
            rtol,  # rtol
            True,  # test_load_hook
            True,  # strict_loading
            dyn,  # dynamic_shapes
            None,  # check_num_matches
            False,  # skip_output_assert
        )

    else:  # optimize
        gm_transformed = InferenceOptimizer(
            None,
            {
                "optimize_rope": {
                    "stage": "pattern_matcher",
                },
            },
        )(None, gm)
        gm_transformed.to("cuda")

        def checker(gm):
            return any(is_op(n, torch.ops.auto_deploy.flashinfer_rope) for n in gm.graph.nodes)

        run_test_transformed_gm(
            model,
            x,
            gm_transformed,
            checker,
            lambda n: n,
            atol,  # atol
            rtol,  # rtol
            True,  # test_load_hook
            True,  # strict_loading
            dyn,  # dynamic_shapes
            None,  # check_num_matches
            False,  # skip_output_assert
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
    def __init__(self, hidden_size, max_seq, n_head, n_kv, layout="BNSD", mode: str = "match"):
        super().__init__()
        self.hdim = hidden_size // n_head
        self.layout = layout
        self.q_lin = torch.nn.Linear(hidden_size, n_head * self.hdim)
        self.k_lin = torch.nn.Linear(hidden_size, n_kv * self.hdim)
        self.rotary = DSRotaryEmbedding(self.hdim, max_seq, base=10000, device="cuda")
        self.mode = mode  # "match" or "optimize"

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
        if self.mode == "match":
            q_out, k_out = apply_rotary_pos_emb_ds(q, k, cos, sin, pos_ids, unsqueeze_dim=unsq_dim)
        else:
            cos = cos[pos_ids]
            sin = sin[pos_ids]
            q_out, k_out = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
                q, k, cos, sin, unsq_dim
            )
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


@pytest.mark.parametrize(
    "layout,num_heads,num_kv_heads,mode, target_layout",
    [
        ("BNSD", 8, 8, "match", None),
        ("BSND", 8, 4, "match", None),
        ("BNSD", 8, 8, "match_layout", "BNSD"),
        ("BSND", 8, 4, "match_layout", "BNSD"),
        ("BSND", 8, 4, "match_layout", "BSND"),
        ("BNSD", 8, 4, "match_layout", "BSND"),
    ],
)
@torch.inference_mode()
def test_match_and_layout_deepseek(layout, num_heads, num_kv_heads, mode, target_layout):
    batch, seq, hid = 4, 12, 512
    model = DSModel(hid, 16, num_heads, num_kv_heads, layout=layout, mode=mode)
    model = model.to("cuda", torch.float16)

    x = torch.randn(batch, seq, hid, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)

    if mode == "match":
        gm_transformed = InferenceOptimizer(
            None,
            {
                "match_rope_pattern": {
                    "stage": "pattern_matcher",
                },
            },
        )(None, gm)
        gm_transformed.to("cuda")

        def checker(gm):
            return any(
                is_op(n, torch.ops.auto_deploy.torch_rope_with_qk_interleaving)
                for n in gm.graph.nodes
            )

        run_test_transformed_gm(
            model,
            x,
            gm_transformed,
            checker,
            lambda num_p: num_p,
            1e-3,  # atol
            1e-3,  # rtol
            True,  # test_load_hook
            True,  # strict_loading
            dynamic_shapes,  # dynamic_shapes
            1,  # check_num_matches
            False,  # skip_output_assert
        )

    else:  # mode == "match_layout"
        gm_transformed = InferenceOptimizer(
            None,
            {
                "match_rope_layout": {
                    "stage": "pattern_matcher",
                    "expected_layout": target_layout,
                },
            },
        )(None, gm)
        gm_transformed.to("cuda")

        def checker(gm):
            for n in gm.graph.nodes:
                if is_op(n, torch.ops.auto_deploy.torch_rope_with_qk_interleaving):
                    q_arg, k_arg, *rest = n.args
                    if not (
                        is_op(q_arg, torch.ops.aten.contiguous)
                        and is_op(k_arg, torch.ops.aten.contiguous)
                    ):
                        matched = False
                        break

                    old_q, old_k = extract_output_tuple(n, 2)
                    if old_q is None or old_k is None:
                        matched = False
                        break
                    q_transposed = any(is_op(u, torch.ops.aten.transpose) for u in old_q.users)
                    k_transposed = any(is_op(u, torch.ops.aten.transpose) for u in old_k.users)
                    matched = q_transposed and k_transposed

            return matched if layout != target_layout else not matched

        run_test_transformed_gm(
            model,
            x,
            gm_transformed,
            checker,
            lambda num_p: num_p,
            1e-3,  # atol
            1e-3,  # rtol
            True,  # test_load_hook
            True,  # strict_loading
            dynamic_shapes,  # dynamic_shapes
            None,  # check_num_matches
            False,  # skip_output_assert
        )
