"""Tests for QKV GEMM fusion chained with TRT-LLM attention cache insertion.

These tests are split from ``test_gemm_fusion.py`` because they depend on the
TRT-LLM attention backend (``insert_cached_attention`` with ``backend=trtllm``),
which is unavailable in the standalone auto_deploy package. The standalone
package excludes this file via ``EXCLUDE_TEST_FILES`` in
``examples/auto_deploy/create_standalone_package.py``.
"""

import operator

import torch
import torch.nn as nn
from test_gemm_fusion import TestModel  # type: ignore

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 (registers torch_attention op)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op, is_op
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

torch.manual_seed(0)


def _count_split_output_nodes(gm):
    """Count getitem nodes that extract slices from a split_output/split_with_sizes."""
    count = 0
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is operator.getitem:
            source = n.args[0]
            if isinstance(source, torch.fx.Node) and source.op == "call_function":
                count += 1
    return count


class QKVAttentionModel(TestModel):
    """Model with separate Q, K, V projections feeding into torch_attention.

    Mimics the attention pattern in transformer models where Q, K, V are
    projected from the same input. fuse_gemms_mixed_children should fuse the
    3 projections into one GEMM with 3 narrow views.
    """

    def __init__(
        self,
        batch_size=2,
        seq_len=8,
        hidden_size=64,
        num_heads=4,
        num_kv_heads=None,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def get_input(self, **kwargs):
        return torch.randn(self.batch_size, self.seq_len, self.hidden_size, **kwargs)

    @property
    def keys_to_pop(self):
        return ("q_proj.weight", "k_proj.weight", "v_proj.weight")

    @property
    def num_gemms_after_fusion(self) -> int:
        return 2  # 1 fused QKV + 1 o_proj

    @property
    def expected_narrow_count(self) -> int:
        return 3  # Q, K, V slices

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)

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
        out = attn.reshape(b, s, self.hidden_size)
        return self.o_proj(out)


@torch.inference_mode()
def test_fuse_qkv_with_trtllm_cache_insertion():
    """Chain QKV fusion → TRT-LLM cache insertion and verify the pipeline works.

    This tests that fuse_gemms_mixed_children produces correct meta['val']
    shapes that the insert_cached_attention transform can consume when using
    the TRT-LLM attention backend.
    """
    model = QKVAttentionModel(hidden_size=64, num_heads=4).to(device="cuda", dtype=torch.float16)
    x = model.get_input(device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm = InferenceOptimizer(
        None,
        {"fuse_gemms_mixed_children": {"stage": "post_load_fusion"}},
    )(None, gm)

    assert _count_split_output_nodes(gm) == 3
    assert sum(is_linear_op(n) for n in gm.graph.nodes) == 2

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=128,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=64,
        max_batch_size=4,
        max_num_tokens=64 * 4,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    gm = InferenceOptimizer(
        None,
        {"insert_cached_attention": {"stage": "cache_init", "backend": "trtllm"}},
    )(cm, gm)

    cached_attn_nodes = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default)
    ]
    assert len(cached_attn_nodes) == 1, (
        f"Expected 1 trtllm_attention_mha_with_cache node, got {len(cached_attn_nodes)}"
    )

    prep_meta_nodes = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.trtllm_attention_prepare_metadata.default)
    ]
    assert len(prep_meta_nodes) == 1, (
        f"Expected 1 prepare_metadata node, got {len(prep_meta_nodes)}"
    )


@torch.inference_mode()
def test_fuse_qkv_gqa_with_trtllm_cache_insertion():
    """Same pipeline but with GQA (num_kv_heads < num_heads).

    Verifies that asymmetric Q/KV projection sizes work through the full
    fusion → cache insertion pipeline.
    """
    model = QKVAttentionModel(
        hidden_size=64,
        num_heads=4,
        num_kv_heads=2,
    ).to(device="cuda", dtype=torch.float16)
    x = model.get_input(device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm = InferenceOptimizer(
        None,
        {"fuse_gemms_mixed_children": {"stage": "post_load_fusion"}},
    )(None, gm)

    assert _count_split_output_nodes(gm) == 3

    # Verify the fused GEMM produces 3 split outputs (Q, K, V)
    assert _count_split_output_nodes(gm) == 3

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=128,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=64,
        max_batch_size=4,
        max_num_tokens=64 * 4,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    gm = InferenceOptimizer(
        None,
        {"insert_cached_attention": {"stage": "cache_init", "backend": "trtllm"}},
    )(cm, gm)

    cached_attn_nodes = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default)
    ]
    assert len(cached_attn_nodes) == 1
