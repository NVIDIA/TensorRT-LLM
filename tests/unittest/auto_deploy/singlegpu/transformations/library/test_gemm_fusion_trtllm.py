"""Tests for QKV GEMM fusion chained with TRT-LLM attention cache insertion.

These tests are split from ``test_gemm_fusion.py`` because they depend on the
TRT-LLM attention backend (``insert_cached_attention`` with ``backend=trtllm``),
which is unavailable in the standalone auto_deploy package. The standalone
package excludes this file via ``EXCLUDE_TEST_FILES`` in
``examples/auto_deploy/create_standalone_package.py``.
"""

import operator

import torch
from test_gemm_fusion import QKVAttentionModel, _get_narrow_nodes  # type: ignore

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 (registers torch_attention op)
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op, is_op

torch.manual_seed(0)


def _count_split_output_nodes(gm):
    """Count torch.narrow nodes produced by fuse_gemms_mixed_children.

    These narrow calls split the fused GEMM output back into per-projection
    slices. (The legacy split_output closure path was replaced by
    ``torch.narrow + .contiguous`` in the GEMM fusion code.)
    """
    count = 0
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is torch.narrow:
            count += 1
    return count


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
        max_num_tokens=256,
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

    # The QKV split (3 getitems) must survive cache insertion; cache insertion
    # itself can add more getitems (e.g. from the metadata-prep tuple), so
    # require at least 3 rather than exactly 3.
    assert _count_split_output_nodes(gm) >= 3


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

    # Asymmetric Q/K/V split sizes must be preserved.  The fusion may emit
    # either ``torch.narrow`` ops or a ``split_with_sizes``-style tuple split
    # depending on the dtype path.  In the split-tuple form the getitem nodes
    # may lack ``meta['val']`` (the closure isn't shape-propagated), but the
    # downstream ``view`` they feed into has the right shape — read from there.
    narrow_sizes = sorted([n.args[3] for n in _get_narrow_nodes(gm)])
    if not narrow_sizes:
        getitem_sizes = []
        for n in gm.graph.nodes:
            if (
                n.op == "call_function"
                and n.target is operator.getitem
                and isinstance(n.args[0], torch.fx.Node)
                and n.args[0].op == "call_function"
            ):
                val = n.meta.get("val")
                if val is None:
                    # Fall back to the view consumer's shape: (B, S, n_heads, head_dim)
                    for user in n.users:
                        uval = user.meta.get("val")
                        if uval is not None and len(uval.shape) >= 4:
                            getitem_sizes.append(int(uval.shape[-1] * uval.shape[-2]))
                            break
                else:
                    getitem_sizes.append(int(val.shape[-1]))
        narrow_sizes = sorted(getitem_sizes)
    assert narrow_sizes == [32, 32, 64], f"Unexpected QKV split sizes for GQA: {narrow_sizes}"

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
        {"insert_cached_attention": {"stage": "cache_init", "backend": "trtllm"}},
    )(cm, gm)

    cached_attn_nodes = [
        n
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default)
    ]
    assert len(cached_attn_nodes) == 1


class QKVRopeAttentionModel(QKVAttentionModel):
    """``QKVAttentionModel`` with HF-style explicit cos/sin RoPE on Q and K.

    This shape is what ``fuse_rope_into_trtllm_attention`` looks for: a
    ``torch_rope_with_explicit_cos_sin`` node sits between Q/K projections
    and the ``torch_attention`` call, and Q/K/V all trace back to the same
    fused QKV GEMM output so the QKV-passthrough leg can fire.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ``fuse_rope_into_trtllm_attention`` reads cos/sin as 2D
        # ``[max_pos, head_dim]`` buffers (HF-style table).
        self.register_buffer(
            "rope_cos_table",
            torch.zeros(self.seq_len, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin_table",
            torch.zeros(self.seq_len, self.head_dim),
            persistent=False,
        )

    def forward(self, x):
        b, s, _ = x.shape
        # Index cos/sin tables by position so the extractor can trace through
        # ``aten.index.Tensor`` back to the underlying 2D table.
        position_ids = torch.arange(s, device=x.device)
        cos = self.rope_cos_table[position_ids].unsqueeze(0).expand(b, -1, -1)
        sin = self.rope_sin_table[position_ids].unsqueeze(0).expand(b, -1, -1)
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin.default(q, k, cos, sin, 2)
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
def test_fuse_qkv_passthrough_with_rope():
    """Exercise the QKV-passthrough leg of fuse_rope_into_trtllm_attention.

    Builds a graph where Q/K/V come from a fused QKV GEMM and Q/K go
    through ``torch_rope_with_explicit_cos_sin`` before attention.  After
    rope fusion the ``torch_attention`` node must carry the
    ``_trtllm_fused_qkv`` marker plus non-zero head/dim hints — proving
    that the passthrough actually fired.
    """
    model = QKVRopeAttentionModel(hidden_size=64, num_heads=4).to(
        device="cuda", dtype=torch.float16
    )
    x = model.get_input(device="cuda", dtype=torch.float16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)

    gm = InferenceOptimizer(
        None,
        {
            "fuse_gemms_mixed_children": {"stage": "post_load_fusion"},
            "fuse_rope_into_trtllm_attention": {"stage": "post_load_fusion"},
        },
    )(None, gm)

    # The torch_attention node should now carry passthrough metadata.
    torch_attn_nodes = [
        n for n in gm.graph.nodes if is_op(n, torch.ops.auto_deploy.torch_attention.default)
    ]
    assert len(torch_attn_nodes) == 1
    src = torch_attn_nodes[0]
    assert src.meta.get("_trtllm_fused_qkv") is True, (
        "fuse_rope_into_trtllm_attention should have set _trtllm_fused_qkv"
    )
    assert src.meta.get("_trtllm_num_heads") == 4
    assert src.meta.get("_trtllm_num_kv_heads") == 4
    assert src.meta.get("_trtllm_head_dim") == 16
    # Q/K/V args must all alias the same fused-QKV node.
    assert src.args[0] is src.args[1] is src.args[2]
