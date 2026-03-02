from functools import partial

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
from tensorrt_llm._torch.auto_deploy.models.custom.mla_rope_utils import (
    _rope_deinterleave_load_hook,
)
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
        return {0: Dim.DYNAMIC, 1: Dim.DYNAMIC}


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
            matched = False
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
        return {0: Dim.DYNAMIC, 1: Dim.DYNAMIC}


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
            matched = False
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


@pytest.mark.parametrize(
    "num_heads,num_kv_heads",
    [
        (8, 8),  # Standard MHA
        (8, 1),  # MQA (DeepSeek-style)
    ],
)
@torch.inference_mode()
def test_optimize_interleaved_rope(num_heads, num_kv_heads):
    """Test that optimize_rope replaces torch_rope_with_qk_interleaving
    with triton_rope_on_interleaved_qk_inputs by tracing back through
    aten.index.Tensor to find the cached cos/sin and position_ids."""
    batch, seq, hid = 4, 12, 512
    model = DSModel(hid, 16, num_heads, num_kv_heads, layout="BSND", mode="optimize")
    model = model.to("cuda", torch.float16)

    x = torch.randn(batch, seq, hid, device="cuda", dtype=torch.float16)
    dynamic_shapes = model.get_dynamic_shapes()
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)

    # Verify the graph contains torch_rope_with_qk_interleaving before optimization
    assert any(
        is_op(n, torch.ops.auto_deploy.torch_rope_with_qk_interleaving) for n in gm.graph.nodes
    ), "Expected torch_rope_with_qk_interleaving in graph before optimization"

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
        has_triton = any(
            is_op(n, torch.ops.auto_deploy.triton_rope_on_interleaved_qk_inputs)
            for n in gm.graph.nodes
        )
        no_torch_rope = not any(
            is_op(n, torch.ops.auto_deploy.torch_rope_with_qk_interleaving) for n in gm.graph.nodes
        )
        return has_triton and no_torch_rope

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        checker,
        lambda num_p: num_p,
        1e-2,  # atol
        1e-2,  # rtol
        True,  # test_load_hook
        True,  # strict_loading
        dynamic_shapes,  # dynamic_shapes
        None,  # check_num_matches
        False,  # skip_output_assert
    )


class DSExplicitRotaryEmbedding(torch.nn.Module):
    """Rotary embedding that stores cos/sin as buffers (full tables).

    When indexed with position_ids in the model forward, this creates the
    ``aten.index.Tensor(table, [position_ids])`` graph pattern that triggers
    the full-table optimization path in ``optimize_rope``.
    """

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
        return self.cos_cached, self.sin_cached


class MLASelfAttn(torch.nn.Module):
    """Simplified MLA self-attention for testing interleaved weights + RoPE.

    Mirrors the GLM4/DeepSeek MLA attention:
    - q_b_proj output is split into nope + rope parts via torch.split
    - kv_a_proj_with_mqa output is split into compressed_kv + k_pe
    - torch.split produces non-contiguous q_pe/k_pe (strided tensors)
    - RoPE is applied on the rope parts only

    rope_mode:
    - "explicit": uses torch_rope_with_explicit_cos_sin (NeoX layout)
    - "triton": uses torch_rope_with_qk_interleaving (interleaved layout)
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        q_lora_rank,
        rope_mode="explicit",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.rope_mode = rope_mode

        self.q_a_proj = torch.nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_b_proj = torch.nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = torch.nn.Linear(
            hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False
        )

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_b_proj(self.q_a_proj(x))
        q = q.view(b, s, self.num_heads, self.qk_head_dim)
        # split produces non-contiguous q_pe (strided tensor)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(b, s, 1, self.qk_rope_head_dim)

        if self.rope_mode == "explicit":
            q_pe_rot, k_pe_rot = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
                q_pe,
                k_pe,
                cos,
                sin,
                2,
            )
        else:  # triton
            q_pe_rot, k_pe_rot = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
                q_pe,
                k_pe,
                cos,
                sin,
                2,
            )

        q_out = torch.cat([q_nope, q_pe_rot], dim=-1).reshape(b, s, -1)
        k_out = k_pe_rot.reshape(b, s, -1)
        return torch.cat([q_out, k_out, compressed_kv], dim=-1)


class DSMLAModelWithExplicitCosSinRopE(torch.nn.Module):
    """MLA-style model with RoPE on split (non-contiguous) tensors.

    End-to-end test model mirroring the DeepSeek/GLM4 MLA pipeline:
    - model.layers[0].self_attn.{q_b_proj, kv_a_proj_with_mqa}: MLA weight layout
    - torch.split creates non-contiguous q_pe/k_pe (tests flashinfer strided path)
    - cos/sin table indexing with position_ids (tests full-table optimization)

    rope_mode:
    - "explicit": NeoX layout, optimized to flashinfer_rope. Needs de-interleave
      load hook when loading interleaved checkpoints.
    - "triton": interleaved layout, optimized to triton_rope. No load hook needed.
    """

    def __init__(
        self,
        hidden_size,
        max_seq,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        q_lora_rank,
        rope_mode="explicit",
    ):
        super().__init__()
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.rope_mode = rope_mode
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = MLASelfAttn(
            hidden_size,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            kv_lora_rank,
            q_lora_rank,
            rope_mode=rope_mode,
        )
        self.model.layers = torch.nn.ModuleList([layer])
        self.rotary = DSExplicitRotaryEmbedding(
            qk_rope_head_dim, max_seq, base=10000, device="cuda"
        )

    def forward(self, x):
        b, s, _ = x.shape
        cos_table, sin_table = self.rotary(x)
        pos_ids = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)
        cos = cos_table[pos_ids]
        sin = sin_table[pos_ids]
        return self.model.layers[0].self_attn(x, cos, sin)

    def get_dynamic_shapes(self):
        return {0: Dim.DYNAMIC, 1: Dim.DYNAMIC}


@pytest.mark.parametrize(
    "num_heads",
    [4, 8],
)
@torch.inference_mode()
def test_optimize_mla_rope(num_heads):
    """Test both explicit and triton RoPE paths on the MLA pipeline.

    Creates a triton model (interleaved weights, no load hook) and an explicit
    model (de-interleaved weights via load hook). After optimize_rope:
    - explicit path should use flashinfer_rope
    - triton path should use triton_rope_on_interleaved_qk_inputs
    - both paths should produce the same output
    """
    hidden_size = 256
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64  # must be multiple of 64 for flashinfer
    kv_lora_rank = 64
    q_lora_rank = 128
    batch, seq, max_seq = 4, 12, 16
    model_args = (
        hidden_size,
        max_seq,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        q_lora_rank,
    )

    x = torch.randn(batch, seq, hidden_size, device="cuda", dtype=torch.float16)

    # --- Triton path: interleaved weights, no load hook ---
    model_triton = DSMLAModelWithExplicitCosSinRopE(
        *model_args,
        rope_mode="triton",
    ).to("cuda", torch.float16)

    dynamic_shapes = model_triton.get_dynamic_shapes()
    gm_triton = torch_export_to_gm(
        model_triton,
        args=(x,),
        dynamic_shapes=(dynamic_shapes,),
        clone=True,
    )
    gm_triton_opt = InferenceOptimizer(
        None,
        {"optimize_rope": {"stage": "pattern_matcher"}},
    )(None, gm_triton)
    gm_triton_opt.to("cuda")

    assert any(
        is_op(n, torch.ops.auto_deploy.triton_rope_on_interleaved_qk_inputs)
        for n in gm_triton_opt.graph.nodes
    ), "Expected triton_rope_on_interleaved_qk_inputs in triton-optimized graph"

    # --- Explicit path: de-interleaved weights via load hook ---
    model_explicit = DSMLAModelWithExplicitCosSinRopE(
        *model_args,
        rope_mode="explicit",
    ).to("cuda", torch.float16)

    # Load triton model's (interleaved) weights through the de-interleave hook
    model_explicit._register_load_state_dict_pre_hook(
        partial(
            _rope_deinterleave_load_hook,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            num_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            num_layers=1,
        )
    )
    model_explicit.load_state_dict(model_triton.state_dict())

    gm_explicit = torch_export_to_gm(
        model_explicit,
        args=(x,),
        dynamic_shapes=(dynamic_shapes,),
        clone=True,
    )
    gm_explicit_opt = InferenceOptimizer(
        None,
        {"optimize_rope": {"stage": "pattern_matcher"}},
    )(None, gm_explicit)
    gm_explicit_opt.to("cuda")

    assert any(
        is_op(n, torch.ops.auto_deploy.flashinfer_rope) for n in gm_explicit_opt.graph.nodes
    ), "Expected flashinfer_rope in explicit-optimized graph"

    # --- Compare outputs: both paths should produce the same result ---
    y_triton = model_triton(x)
    y_explicit = model_explicit(x)
    torch.testing.assert_close(y_triton, y_explicit, atol=1e-2, rtol=1e-2)
