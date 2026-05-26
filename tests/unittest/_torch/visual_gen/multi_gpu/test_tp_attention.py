"""Multi-GPU tests for Tensor Parallel (TP) Attention.

Validates that TP-sharded attention (column-parallel QKV, row-parallel output
with all-reduce) produces the same result as F.scaled_dot_product_attention
with full (unsharded) weights.

Also includes a combined TP + Ulysses test to verify both parallelisms compose.

Run with:
    pytest tests/unittest/_torch/visual_gen/multi_gpu/test_tp_attention.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import math
from typing import Callable

import pytest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

try:
    from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl
    from tensorrt_llm._torch.visual_gen.config import AttentionConfig, DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.mapping import VisualGenMapping
    from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
    from tensorrt_llm._utils import get_free_port
    from tensorrt_llm.mapping import Mapping

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


# =============================================================================
# Distributed helpers (same pattern as test_ulysses_attention.py)
# =============================================================================


def _init_worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, test_fn, port, *args):
    try:
        _init_worker(rank, world_size, port)
        test_fn(rank, world_size, *args)
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        raise
    finally:
        _cleanup()


def _run(world_size: int, test_fn: Callable, *args):
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    port = get_free_port()
    mp.spawn(
        _distributed_worker, args=(world_size, test_fn, port, *args), nprocs=world_size, join=True
    )


# =============================================================================
# Helpers for building configs and sharding weights
# =============================================================================


def _setup_vgm(rank, world_size, tp_size=1, ulysses_size=1):
    """Create a VisualGenMapping which calls init_pg (required by the C++ all-reduce backend)."""
    DeviceMeshTopologyImpl.device_mesh = None
    if tp_size == 1 and ulysses_size == 1:
        return None
    return VisualGenMapping(
        world_size=world_size,
        rank=rank,
        tp_size=tp_size,
        ulysses_size=ulysses_size,
    )


def _make_config(vgm=None, backend="VANILLA", **mapping_kwargs):
    """Build a DiffusionModelConfig for testing.

    When a VisualGenMapping is provided, its to_llm_mapping() supplies the
    Mapping backed by the device mesh.  Otherwise a plain Mapping is created
    from ``mapping_kwargs`` (tp_size, rank, etc.).
    """
    if vgm is not None and (vgm.tp_size > 1 or vgm.ulysses_size > 1):
        mapping = vgm.to_llm_mapping()
    else:
        mapping = Mapping(**mapping_kwargs)
    return DiffusionModelConfig(
        mapping=mapping,
        visual_gen_mapping=vgm,
        attention=AttentionConfig(backend=backend),
        skip_create_weights_in_init=False,
    )


def _broadcast_params(module):
    """Initialize weights with small random values and broadcast from rank 0.

    TRT-LLM Linear uses torch.empty (uninitialized memory), so we must
    explicitly fill with valid data before use.
    """
    for p in module.parameters():
        torch.nn.init.normal_(p.data, mean=0.0, std=0.02)
        dist.broadcast(p.data, src=0)


def _shard_tp_weights(ref_attn, tp_attn, tp_rank, tp_size, qkv_mode=QKVMode.FUSE_QKV):
    """Copy the appropriate TP shard of ref_attn's weights into tp_attn.

    Column-parallel (QKV): split output dim (dim 0)
    Row-parallel (output): split input dim (dim 1)
    RMSNorm (TP-enabled): split weight
    """
    with torch.no_grad():
        if qkv_mode == QKVMode.FUSE_QKV:
            # Fused QKV: weight is [q_dim + 2*kv_dim, hidden_size]
            full_w = ref_attn.qkv_proj.weight.data
            q_dim = ref_attn.q_dim
            kv_dim = ref_attn.kv_dim
            q_w, k_w, v_w = full_w.split([q_dim, kv_dim, kv_dim], dim=0)

            q_shard = _shard_dim0(q_w, tp_rank, tp_size)
            k_shard = _shard_dim0(k_w, tp_rank, tp_size)
            v_shard = _shard_dim0(v_w, tp_rank, tp_size)
            tp_attn.qkv_proj.weight.data.copy_(torch.cat([q_shard, k_shard, v_shard], dim=0))

            if ref_attn.qkv_proj.bias is not None:
                full_b = ref_attn.qkv_proj.bias.data
                q_b, k_b, v_b = full_b.split([q_dim, kv_dim, kv_dim], dim=0)
                tp_attn.qkv_proj.bias.data.copy_(
                    torch.cat(
                        [
                            _shard_dim0(q_b, tp_rank, tp_size),
                            _shard_dim0(k_b, tp_rank, tp_size),
                            _shard_dim0(v_b, tp_rank, tp_size),
                        ],
                        dim=0,
                    )
                )
        else:
            for name in ("to_q", "to_k", "to_v"):
                ref_proj = getattr(ref_attn, name)
                tp_proj = getattr(tp_attn, name)
                tp_proj.weight.data.copy_(_shard_dim0(ref_proj.weight.data, tp_rank, tp_size))
                if ref_proj.bias is not None:
                    tp_proj.bias.data.copy_(_shard_dim0(ref_proj.bias.data, tp_rank, tp_size))

        # Output projection: row-parallel (split input dim = dim 1)
        ref_out = ref_attn.to_out[0]
        tp_out = tp_attn.to_out[0]
        shard_size = math.ceil(ref_out.weight.shape[1] / tp_size)
        start = tp_rank * shard_size
        end = min(start + shard_size, ref_out.weight.shape[1])
        tp_out.weight.data.copy_(ref_out.weight.data[:, start:end].contiguous())
        if ref_out.bias is not None:
            tp_out.bias.data.copy_(ref_out.bias.data)

        # QK norm weights (if TP-enabled, they're sharded)
        if hasattr(ref_attn, "norm_q") and hasattr(tp_attn, "norm_q"):
            if tp_attn.norm_q.enable_tp:
                shard_size = ref_attn.norm_q.weight.shape[0] // tp_size
                start = tp_rank * shard_size
                end = start + shard_size
                tp_attn.norm_q.weight.data.copy_(ref_attn.norm_q.weight.data[start:end])
                tp_attn.norm_k.weight.data.copy_(ref_attn.norm_k.weight.data[start:end])
            else:
                tp_attn.norm_q.weight.data.copy_(ref_attn.norm_q.weight.data)
                tp_attn.norm_k.weight.data.copy_(ref_attn.norm_k.weight.data)


def _shard_dim0(tensor, tp_rank, tp_size):
    """Shard a tensor along dim 0 (works for both 1D bias and 2D weight)."""
    shard_size = math.ceil(tensor.shape[0] / tp_size)
    start = tp_rank * shard_size
    end = min(start + shard_size, tensor.shape[0])
    return tensor[start:end].contiguous()


# =============================================================================
# Manual F.sdpa reference
# =============================================================================


def _wb(module):
    """Return (weight, bias_or_None) for a Linear module."""
    return module.weight.data, (module.bias.data if module.bias is not None else None)


def _sdpa_ref(q, k, v, out_weight, out_bias, num_heads, head_dim):
    """Reshape Q/K/V [B,S,H*D] → run F.sdpa → output projection."""
    B, S, _ = q.shape
    q = q.view(B, S, num_heads, head_dim).transpose(1, 2)
    k = k.view(B, S, num_heads, head_dim).transpose(1, 2)
    v = v.view(B, S, num_heads, head_dim).transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(head_dim))
    out = out.transpose(1, 2).reshape(B, S, num_heads * head_dim)

    return F.linear(out, out_weight, out_bias)


def _manual_attention(x, qkv_weight, qkv_bias, out_weight, out_bias, num_heads, head_dim):
    """Manual F.sdpa reference with fused QKV projection."""
    q_dim = num_heads * head_dim
    qkv = F.linear(x, qkv_weight, qkv_bias)
    q, k, v = qkv.split([q_dim, q_dim, q_dim], dim=-1)
    return _sdpa_ref(q, k, v, out_weight, out_bias, num_heads, head_dim)


def _manual_attention_separate(x, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, num_heads, head_dim):
    """Manual F.sdpa reference with separate Q/K/V projections."""
    q = F.linear(x, q_w, q_b)
    k = F.linear(x, k_w, k_b)
    v = F.linear(x, v_w, v_b)
    return _sdpa_ref(q, k, v, out_w, out_b, num_heads, head_dim)


def _manual_attention_with_norm(
    x, qkv_weight, qkv_bias, out_weight, out_bias, norm_q_w, norm_k_w, eps, num_heads, head_dim
):
    """Manual F.sdpa reference with fused QKV + RMSNorm on Q/K."""
    q_dim = num_heads * head_dim
    qkv = F.linear(x, qkv_weight, qkv_bias)
    q, k, v = qkv.split([q_dim, q_dim, q_dim], dim=-1)

    def _rms_norm(t, weight):
        t_f32 = t.to(torch.float32)
        var = t_f32.pow(2).mean(-1, keepdim=True)
        return weight * (t_f32 * torch.rsqrt(var + eps)).to(t.dtype)

    q = _rms_norm(q, norm_q_w)
    k = _rms_norm(k, norm_k_w)
    return _sdpa_ref(q, k, v, out_weight, out_bias, num_heads, head_dim)


# =============================================================================
# Reusable test body
# =============================================================================


def _run_tp_with_params(
    rank, world_size, batch, seq, hidden_size, num_heads, qkv_mode=QKVMode.FUSE_QKV, qk_norm=False
):
    """Run a single TP-vs-F.sdpa comparison with the given dimensions."""
    tp_size = world_size
    device = torch.device(f"cuda:{rank}")
    head_dim = hidden_size // num_heads

    vgm = _setup_vgm(rank, world_size, tp_size=tp_size)

    # Build ref module for weight extraction
    config_ref = _make_config(world_size=1, rank=0, tp_size=1)
    attn_ref = Attention(
        hidden_size,
        num_heads,
        qkv_mode=qkv_mode,
        qk_norm=qk_norm,
        qk_norm_mode="full",
        config=config_ref,
    ).to(device)
    _broadcast_params(attn_ref)

    # Build TP module, shard weights from ref
    config_tp = _make_config(vgm=vgm)
    attn_tp = Attention(
        hidden_size,
        num_heads,
        qkv_mode=qkv_mode,
        qk_norm=qk_norm,
        qk_norm_mode="full",
        config=config_tp,
    ).to(device)
    _shard_tp_weights(attn_ref, attn_tp, rank, tp_size, qkv_mode=qkv_mode)

    # Same input on all ranks
    torch.manual_seed(42)
    x = torch.randn(batch, seq, hidden_size, device=device, dtype=torch.bfloat16)

    # Build manual F.sdpa reference
    out_w, out_b = _wb(attn_ref.to_out[0])

    if qkv_mode == QKVMode.FUSE_QKV:
        qkv_w, qkv_b = _wb(attn_ref.qkv_proj)
        if qk_norm:
            ref_out = _manual_attention_with_norm(
                x,
                qkv_w,
                qkv_b,
                out_w,
                out_b,
                attn_ref.norm_q.weight.data,
                attn_ref.norm_k.weight.data,
                attn_ref.norm_q.variance_epsilon,
                num_heads,
                head_dim,
            )
        else:
            ref_out = _manual_attention(x, qkv_w, qkv_b, out_w, out_b, num_heads, head_dim)
    else:
        ref_out = _manual_attention_separate(
            x,
            *_wb(attn_ref.to_q),
            *_wb(attn_ref.to_k),
            *_wb(attn_ref.to_v),
            out_w,
            out_b,
            num_heads,
            head_dim,
        )

    tp_out = attn_tp(x)

    torch.testing.assert_close(tp_out, ref_out, rtol=1e-2, atol=1e-2)


# =============================================================================
# Test logic functions (module-level for pickling by mp.spawn)
# =============================================================================


def _logic_tp_fused_qkv(rank, world_size):
    """TP with fused QKV matches F.sdpa reference."""
    _run_tp_with_params(rank, world_size, batch=2, seq=16, hidden_size=256, num_heads=8)


def _logic_tp_separate_qkv(rank, world_size):
    """TP with separate Q/K/V projections matches F.sdpa reference."""
    _run_tp_with_params(
        rank,
        world_size,
        batch=2,
        seq=16,
        hidden_size=256,
        num_heads=8,
        qkv_mode=QKVMode.SEPARATE_QKV,
    )


def _logic_tp_with_qk_norm(rank, world_size):
    """TP with QK norm (full mode) matches F.sdpa reference."""
    _run_tp_with_params(
        rank, world_size, batch=2, seq=16, hidden_size=256, num_heads=8, qk_norm=True
    )


def _logic_tp_batch_size_1(rank, world_size):
    _run_tp_with_params(rank, world_size, batch=1, seq=16, hidden_size=256, num_heads=8)


def _logic_tp_hidden_128(rank, world_size):
    _run_tp_with_params(rank, world_size, batch=2, seq=16, hidden_size=128, num_heads=4)


def _logic_tp_hidden_512(rank, world_size):
    _run_tp_with_params(rank, world_size, batch=2, seq=16, hidden_size=512, num_heads=4)


def _logic_tp_heads_not_divisible(rank, world_size):
    """TP when num_heads % tp_size != 0. Expected to fail until uneven sharding is implemented."""
    _run_tp_with_params(rank, world_size, batch=2, seq=16, hidden_size=320, num_heads=5)


def _logic_tp_world_size_4(rank, world_size):
    _run_tp_with_params(rank, world_size, batch=2, seq=16, hidden_size=512, num_heads=16)


def _logic_tp_ulysses_combined(rank, world_size, ulysses_size, tp_size):
    """TP + Ulysses combined matches F.sdpa reference on the full sequence.

    4 GPUs: tp_size=2, ulysses_size=2.
    Ulysses shards the sequence; TP shards the heads/weights.
    """
    assert tp_size * ulysses_size == world_size
    device = torch.device(f"cuda:{rank}")

    hidden_size = 512
    num_heads = 16
    head_dim = hidden_size // num_heads
    batch = 2
    seq_per_rank = 8
    seq_full = seq_per_rank * ulysses_size

    vgm = _setup_vgm(rank, world_size, tp_size=tp_size, ulysses_size=ulysses_size)

    # Build ref module for weight extraction
    config_ref = _make_config(world_size=1, rank=0, tp_size=1)
    attn_ref = Attention(hidden_size, num_heads, qk_norm=False, config=config_ref).to(device)
    _broadcast_params(attn_ref)

    qkv_w, qkv_b = _wb(attn_ref.qkv_proj)
    out_w, out_b = _wb(attn_ref.to_out[0])

    # Build combined TP + Ulysses module
    config_combined = _make_config(vgm=vgm)
    attn_combined = Attention(hidden_size, num_heads, qk_norm=False, config=config_combined).to(
        device
    )
    _shard_tp_weights(attn_ref, attn_combined, vgm.tp_rank, tp_size)

    # Same full input on all ranks
    torch.manual_seed(42)
    x_full = torch.randn(batch, seq_full, hidden_size, device=device, dtype=torch.bfloat16)

    # F.sdpa reference on full sequence
    ref_out = _manual_attention(x_full, qkv_w, qkv_b, out_w, out_b, num_heads, head_dim)

    # Each Ulysses rank takes its sequence shard
    ulysses_rank = vgm.ulysses_rank
    x_shard = x_full[
        :, ulysses_rank * seq_per_rank : (ulysses_rank + 1) * seq_per_rank
    ].contiguous()
    expected_shard = ref_out[:, ulysses_rank * seq_per_rank : (ulysses_rank + 1) * seq_per_rank]

    combined_out = attn_combined(x_shard)

    torch.testing.assert_close(combined_out, expected_shard, rtol=1e-2, atol=1e-2)


# =============================================================================
# Test classes
# =============================================================================


class TestTPAttention:
    """Core correctness tests for TP-sharded Attention."""

    def test_tp_fused_qkv(self):
        _run(2, _logic_tp_fused_qkv)

    def test_tp_separate_qkv(self):
        _run(2, _logic_tp_separate_qkv)

    def test_tp_with_qk_norm(self):
        _run(2, _logic_tp_with_qk_norm)


class TestTPAttentionEdgeCases:
    """Edge case tests for TP attention."""

    def test_batch_size_1(self):
        _run(2, _logic_tp_batch_size_1)

    def test_hidden_128(self):
        _run(2, _logic_tp_hidden_128)

    def test_hidden_512(self):
        _run(2, _logic_tp_hidden_512)

    @pytest.mark.xfail(reason="Uneven head sharding not yet implemented", raises=Exception)
    def test_tp_heads_not_divisible(self):
        _run(2, _logic_tp_heads_not_divisible)

    def test_tp_world_size_4(self):
        _run(4, _logic_tp_world_size_4)


class TestTPUlyssesCombined:
    """Tests for combined TP + Ulysses parallelism."""

    def test_tp_ulysses_combined(self):
        tp_size = 2
        ulysses_size = 2
        world = ulysses_size * tp_size
        _run(world, _logic_tp_ulysses_combined, ulysses_size, tp_size)

    def test_tp_2_ulysses_4(self):
        tp_size = 2
        ulysses_size = 4
        world = ulysses_size * tp_size
        _run(world, _logic_tp_ulysses_combined, ulysses_size, tp_size)

    def test_tp_4_ulysses_2(self):
        tp_size = 4
        ulysses_size = 2
        world = ulysses_size * tp_size
        _run(world, _logic_tp_ulysses_combined, ulysses_size, tp_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
