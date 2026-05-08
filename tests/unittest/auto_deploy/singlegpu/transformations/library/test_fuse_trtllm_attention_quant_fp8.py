import pytest
import torch
import torch.nn as nn
from _torch_test_utils import fp8_compatible

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.trtllm_attention import (
    get_trtllm_attention_fp8_input_scale,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op


def _build_fp8_linear_args(hidden_size: int, out_features: int, device: torch.device):
    weight_fp8 = torch.randn(out_features, hidden_size, device=device, dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.tensor(1.0, device=device, dtype=torch.float32)
    return weight_fp8, weight_scale


class AttentionMixedConsumers(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        device = torch.device("cuda")
        w_fp8, w_scale = _build_fp8_linear_args(hidden_size, hidden_size, device)
        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", w_scale)
        self.register_buffer("input_scale", torch.tensor(2.0, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        attn = torch.ops.auto_deploy.torch_attention.default(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=None,
            logit_cap=None,
            layout="bsnd",
        )
        attn_flat = attn.reshape(b, s, self.hidden_size)
        fp8_out = torch.ops.auto_deploy.trtllm_quant_fp8_linear.default(
            attn_flat, self.weight_fp8, None, self.input_scale, self.weight_scale
        )
        non_fp8 = self.out_proj(attn_flat)
        return fp8_out + non_fp8


class AttentionDifferentScales(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        device = torch.device("cuda")
        w1_fp8, w1_scale = _build_fp8_linear_args(hidden_size, hidden_size, device)
        w2_fp8, w2_scale = _build_fp8_linear_args(hidden_size, hidden_size, device)
        self.register_buffer("weight1_fp8", w1_fp8)
        self.register_buffer("weight1_scale", w1_scale)
        self.register_buffer("weight2_fp8", w2_fp8)
        self.register_buffer("weight2_scale", w2_scale)
        self.register_buffer("input_scale_a", torch.tensor(2.0, device=device, dtype=torch.float32))
        self.register_buffer("input_scale_b", torch.tensor(3.0, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        attn = torch.ops.auto_deploy.torch_attention.default(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=None,
            logit_cap=None,
            layout="bsnd",
        )
        attn_flat = attn.reshape(b, s, self.hidden_size)
        out_a = torch.ops.auto_deploy.trtllm_quant_fp8_linear.default(
            attn_flat, self.weight1_fp8, None, self.input_scale_a, self.weight1_scale
        )
        out_b = torch.ops.auto_deploy.trtllm_quant_fp8_linear.default(
            attn_flat, self.weight2_fp8, None, self.input_scale_b, self.weight2_scale
        )
        return out_a + out_b


class AttentionSharedScales(nn.Module):
    def __init__(self, hidden_size: int = 16, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        device = torch.device("cuda")
        w1_fp8, w1_scale = _build_fp8_linear_args(hidden_size, hidden_size, device)
        w2_fp8, w2_scale = _build_fp8_linear_args(hidden_size, hidden_size, device)
        self.register_buffer("weight1_fp8", w1_fp8)
        self.register_buffer("weight1_scale", w1_scale)
        self.register_buffer("weight2_fp8", w2_fp8)
        self.register_buffer("weight2_scale", w2_scale)
        self.register_buffer("input_scale", torch.tensor(2.0, device=device, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_heads, self.head_dim)
        attn = torch.ops.auto_deploy.torch_attention.default(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=None,
            logit_cap=None,
            layout="bsnd",
        )
        attn_flat = attn.reshape(b, s, self.hidden_size)
        out_a = torch.ops.auto_deploy.trtllm_quant_fp8_linear.default(
            attn_flat, self.weight1_fp8, None, self.input_scale, self.weight1_scale
        )
        out_b = torch.ops.auto_deploy.trtllm_quant_fp8_linear.default(
            attn_flat, self.weight2_fp8, None, self.input_scale, self.weight2_scale
        )
        return out_a + out_b


def _find_nodes(gm, op):
    return [node for node in gm.graph.nodes if is_op(node, op)]


@pytest.mark.skipif(not fp8_compatible(), reason="Requires FP8 support")
@pytest.mark.parametrize(
    "model_cls",
    [AttentionMixedConsumers, AttentionDifferentScales],
    ids=["mixed_consumers", "different_scales"],
)
def test_fuse_trtllm_attention_quant_fp8_negative_no_contract_no_out_dtype(model_cls):
    model = model_cls().to("cuda").eval()
    x = torch.randn(2, 4, model.hidden_size, device="cuda", dtype=torch.float16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm = InferenceOptimizer(
        None,
        {
            "fuse_trtllm_attn_quant_fp8": {
                "stage": "post_load_fusion",
                "enabled": True,
            },
        },
    )(None, gm)

    attn_nodes = _find_nodes(gm, torch.ops.auto_deploy.torch_attention.default)
    assert len(attn_nodes) == 1
    assert get_trtllm_attention_fp8_input_scale(attn_nodes[0]) is None

    fp8_linear_nodes = _find_nodes(gm, torch.ops.auto_deploy.trtllm_quant_fp8_linear.default)
    assert len(fp8_linear_nodes) >= 1
    assert all(node.kwargs.get("out_dtype", None) is None for node in fp8_linear_nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires FP8 support")
def test_fuse_trtllm_attention_quant_fp8_sets_input_scale_contract_only():
    model = AttentionSharedScales().to("cuda").eval()
    x = torch.randn(2, 4, model.hidden_size, device="cuda", dtype=torch.float16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # Start from graph where attention has FP8 input-scale contract.
    gm = InferenceOptimizer(
        None,
        {
            "fuse_trtllm_attn_quant_fp8": {
                "stage": "post_load_fusion",
                "enabled": True,
            },
        },
    )(None, gm)

    attn_nodes = _find_nodes(gm, torch.ops.auto_deploy.torch_attention.default)
    assert len(attn_nodes) == 1
    input_scale_contract = get_trtllm_attention_fp8_input_scale(attn_nodes[0])
    assert input_scale_contract is not None
    assert len(_find_nodes(gm, torch.ops.aten.reciprocal.default)) == 0


@pytest.mark.skipif(not fp8_compatible(), reason="Requires FP8 support")
def test_insert_cached_attention_trtllm_materializes_out_scale_reciprocal():
    model = AttentionSharedScales().to("cuda").eval()
    x = torch.randn(2, 4, model.hidden_size, device="cuda", dtype=torch.float16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm = InferenceOptimizer(
        None,
        {
            "fuse_trtllm_attn_quant_fp8": {
                "stage": "post_load_fusion",
                "enabled": True,
            },
        },
    )(None, gm)
    attn_nodes = _find_nodes(gm, torch.ops.auto_deploy.torch_attention.default)
    assert len(attn_nodes) == 1

    reciprocal_nodes_before = _find_nodes(gm, torch.ops.aten.reciprocal.default)
    assert len(reciprocal_nodes_before) == 0

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=128,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=64,
        max_batch_size=4,
        max_num_tokens=256,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    gm = InferenceOptimizer(
        None,
        {
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "trtllm",
            },
        },
    )(cm, gm)

    cached_nodes = _find_nodes(gm, torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default)
    assert len(cached_nodes) == 1

    # The ``out_scale`` argument materialises ``1/input_scale`` either as a
    # per-step ``aten.reciprocal`` op (dynamic scale) or as a pre-computed
    # static buffer (``get_attr`` to ``_trtllm_recip_*``) when ``input_scale``
    # is a static module buffer.  Either form is correct; the latter avoids
    # a per-step kernel launch.
    (out_scale_arg,) = extract_op_args(cached_nodes[0], "out_scale")
    assert isinstance(out_scale_arg, torch.fx.Node)
    if is_op(out_scale_arg, torch.ops.aten.reciprocal.default):
        # Dynamic-scale form: the reciprocal node must precede the cached attn.
        graph_nodes = list(gm.graph.nodes)
        assert graph_nodes.index(out_scale_arg) < graph_nodes.index(cached_nodes[0])
    else:
        # Static-scale form: a fresh get_attr to a pre-folded recip buffer.
        assert out_scale_arg.op == "get_attr", (
            f"Expected out_scale to be aten.reciprocal or a get_attr; "
            f"got op={out_scale_arg.op}, target={out_scale_arg.target}"
        )
        assert "recip" in str(out_scale_arg.target), (
            f"Expected pre-folded recip buffer name to contain 'recip'; "
            f"got target={out_scale_arg.target}"
        )


@pytest.mark.skipif(not fp8_compatible(), reason="Requires FP8 support")
def test_insert_cached_attention_trtllm_fallback_without_fp8_contract():
    model = AttentionSharedScales().to("cuda").eval()
    x = torch.randn(2, 4, model.hidden_size, device="cuda", dtype=torch.float16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)

    # No fusion transform run: input-scale contract is absent and cache-init should keep
    # origin/main behavior (no out_scale contract, no reciprocal insertion).
    attn_nodes = _find_nodes(gm, torch.ops.auto_deploy.torch_attention.default)
    assert len(attn_nodes) == 1
    assert get_trtllm_attention_fp8_input_scale(attn_nodes[0]) is None

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=128,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=64,
        max_batch_size=4,
        max_num_tokens=256,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    gm = InferenceOptimizer(
        None,
        {
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "trtllm",
            },
        },
    )(cm, gm)

    assert len(_find_nodes(gm, torch.ops.aten.reciprocal.default)) == 0
    cached_nodes = _find_nodes(gm, torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default)
    assert len(cached_nodes) == 1
    (out_scale_arg,) = extract_op_args(cached_nodes[0], "out_scale")
    assert out_scale_arg is None
