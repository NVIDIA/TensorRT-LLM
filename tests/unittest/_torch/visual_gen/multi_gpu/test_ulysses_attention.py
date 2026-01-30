"""Multi-GPU tests for Ulysses Attention.

These tests use torch.multiprocessing.spawn to launch multiple processes internally.
Run with:
    pytest tests/visual_gen/multi_gpu/test_ulysses_attention.py -v
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

# Try to import the modules - skip tests if not available
try:
    from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
    from tensorrt_llm._torch.distributed import all_to_all_4d
    from tensorrt_llm._torch.visual_gen.attention_backend import UlyssesAttention, VanillaAttention
    from tensorrt_llm._utils import get_free_port

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    """Clean up TLLM_DISABLE_MPI env var after tests complete."""
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


def init_distributed_worker(rank: int, world_size: int, backend: str = "gloo", port: int = 29500):
    """Initialize distributed environment for a worker process."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Use gloo backend for CPU, nccl for GPU
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_worker(rank, world_size, backend, test_fn, port):
    """Worker function that runs in each process. Module-level for pickling."""
    try:
        init_distributed_worker(rank, world_size, backend, port)
        test_fn(rank, world_size)
    except Exception as e:
        print(f"Rank {rank} failed with error: {e}")
        raise
    finally:
        cleanup_distributed()


def run_test_in_distributed(world_size: int, test_fn: Callable, use_cuda: bool = True):
    """Run a test function in a distributed environment with multiple processes.

    Args:
        world_size: Number of processes to spawn
        test_fn: Test function to run (must be module-level for pickling).
                 Should accept (rank, world_size) as arguments.
        use_cuda: Whether to use CUDA (requires sufficient GPUs)
    """
    if not MODULES_AVAILABLE:
        pytest.skip("Required modules not available")

    if use_cuda and torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, only {torch.cuda.device_count()} available")

    backend = "nccl" if use_cuda else "gloo"

    port = get_free_port()

    # Spawn processes
    mp.spawn(
        _distributed_worker, args=(world_size, backend, test_fn, port), nprocs=world_size, join=True
    )


# =============================================================================
# Test logic functions (module-level so they can be pickled by mp.spawn)
# =============================================================================


def _logic_a2a_seq_to_head(rank, world_size):
    """all_to_all_4d: sequence sharding to head sharding."""
    batch = 2
    seq_per_rank = 4
    heads = 8
    head_dim = 64

    if heads % world_size != 0:
        heads = world_size * 2

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    input_tensor = (
        torch.randn(batch, seq_per_rank, heads, head_dim, device=device, dtype=torch.float32)
        + rank * 100
    )

    output = all_to_all_4d(
        input_tensor,
        scatter_dim=2,
        gather_dim=1,
        process_group=None,
    )

    expected_shape = (batch, seq_per_rank * world_size, heads // world_size, head_dim)
    assert output.shape == expected_shape, (
        f"Rank {rank}: Expected shape {expected_shape}, got {output.shape}"
    )
    assert output.device == device


def _logic_a2a_head_to_seq(rank, world_size):
    """all_to_all_4d: head sharding to sequence sharding."""
    batch = 2
    seq = 16
    heads_per_rank = 2
    head_dim = 64

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    input_tensor = torch.randn(
        batch, seq, heads_per_rank, head_dim, device=device, dtype=torch.float32
    )

    output = all_to_all_4d(
        input_tensor,
        scatter_dim=1,
        gather_dim=2,
        process_group=None,
    )

    expected_shape = (batch, seq // world_size, heads_per_rank * world_size, head_dim)
    assert output.shape == expected_shape, (
        f"Rank {rank}: Expected shape {expected_shape}, got {output.shape}"
    )


def _logic_a2a_roundtrip(rank, world_size):
    """all_to_all_4d: forward and backward are inverses."""
    batch = 2
    seq_per_rank = 4
    heads = world_size * 4
    head_dim = 64

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    original = torch.randn(batch, seq_per_rank, heads, head_dim, device=device, dtype=torch.float32)

    intermediate = all_to_all_4d(original, scatter_dim=2, gather_dim=1, process_group=None)
    reconstructed = all_to_all_4d(intermediate, scatter_dim=1, gather_dim=2, process_group=None)

    assert reconstructed.shape == original.shape
    torch.testing.assert_close(reconstructed, original, rtol=1e-5, atol=1e-5)


def _logic_a2a_single_process(rank, world_size):
    """all_to_all_4d: single process returns input unchanged."""
    batch, seq, heads, head_dim = 2, 8, 4, 64
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(batch, seq, heads, head_dim, device=device)

    output = all_to_all_4d(input_tensor, scatter_dim=2, gather_dim=1, process_group=None)

    torch.testing.assert_close(output, input_tensor)


def _logic_ulysses_init(rank, world_size):
    """UlyssesAttention initialization."""
    num_heads = world_size * 4
    head_dim = 64

    inner = VanillaAttention(num_heads=num_heads // world_size, head_dim=head_dim)
    attention = UlyssesAttention(
        inner_backend=inner,
        process_group=None,
    )

    assert attention.num_heads == num_heads
    assert attention.head_dim == head_dim
    assert attention.world_size == world_size
    assert rank >= 0 and rank < world_size


def _logic_ulysses_forward(rank, world_size):
    """UlyssesAttention forward pass."""
    batch = 2
    seq_per_rank = 8
    num_heads = world_size * 4
    head_dim = 64

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    inner = VanillaAttention(num_heads=num_heads // world_size, head_dim=head_dim)
    attention = UlyssesAttention(
        inner_backend=inner,
        process_group=None,
    ).to(device)

    q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)

    output = attention(q, k, v, batch_size=batch, seq_len=seq_per_rank * world_size)

    assert output.shape == q.shape, f"Rank {rank}: Expected shape {q.shape}, got {output.shape}"
    assert output.device == device


def _logic_ulysses_with_mask(rank, world_size):
    """UlyssesAttention with attention mask."""
    batch = 2
    seq_per_rank = 8
    seq_full = seq_per_rank * world_size
    num_heads = world_size * 4
    head_dim = 64

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    inner = VanillaAttention(num_heads=num_heads // world_size, head_dim=head_dim)
    attention = UlyssesAttention(
        inner_backend=inner,
        process_group=None,
    ).to(device)

    q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)

    mask = PredefinedAttentionMask.CAUSAL

    output = attention(q, k, v, batch_size=batch, seq_len=seq_full, attention_mask=mask)

    assert output.shape == q.shape


def _logic_ulysses_vs_standard_multi_gpu(rank, world_size):
    """UlyssesAttention across multiple GPUs matches standard attention on the full sequence."""
    batch = 2
    seq_per_rank = 8
    seq_full = seq_per_rank * world_size
    num_heads = world_size * 4
    head_dim = 64

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Every rank generates identical full tensors using the same seed.
    torch.manual_seed(42)
    q_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    k_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)
    v_full = torch.randn(batch, seq_full, num_heads, head_dim, device=device)

    # Each rank takes its sequence shard.
    q_shard = q_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    k_shard = k_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()
    v_shard = v_full[:, rank * seq_per_rank : (rank + 1) * seq_per_rank].contiguous()

    # Ulysses attention on shards.
    inner = VanillaAttention(num_heads=num_heads // world_size, head_dim=head_dim)
    attention = UlyssesAttention(
        inner_backend=inner,
        process_group=None,
    ).to(device)

    ulysses_output = attention(q_shard, k_shard, v_shard, batch_size=batch, seq_len=seq_full)

    # Standard attention on the full tensors.
    q_std = q_full.transpose(1, 2)  # [B, H, S, D]
    k_std = k_full.transpose(1, 2)
    v_std = v_full.transpose(1, 2)

    std_output = F.scaled_dot_product_attention(
        q_std, k_std, v_std, scale=1.0 / math.sqrt(head_dim), dropout_p=0.0
    )
    std_output = std_output.transpose(1, 2).contiguous()  # [B, S, H, D]

    # Compare the shard slice.
    expected_shard = std_output[:, rank * seq_per_rank : (rank + 1) * seq_per_rank]
    torch.testing.assert_close(
        ulysses_output,
        expected_shard,
        rtol=1e-4,
        atol=1e-4,
        msg=f"Rank {rank}: Ulysses multi-GPU output differs from standard attention",
    )


def _logic_ulysses_invalid_heads(rank, world_size):
    """Invalid head count (not divisible by world_size) cannot be sharded."""
    assert rank >= 0 and rank < world_size

    num_heads = world_size * 4 + 1  # Not divisible
    head_dim = 64

    # With the decorator pattern, the caller is responsible for sharding heads.
    # num_heads // world_size truncates, so the wrapper's computed full head
    # count won't match the original.
    sharded_heads = num_heads // world_size
    inner = VanillaAttention(num_heads=sharded_heads, head_dim=head_dim)
    attention = UlyssesAttention(inner_backend=inner, process_group=None)
    assert attention.num_heads != num_heads  # Truncation means mismatch


def _logic_different_batch_sizes(rank, world_size):
    """Various batch sizes."""
    num_heads = world_size * 4
    head_dim = 64
    seq_per_rank = 8
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    inner = VanillaAttention(num_heads=num_heads // world_size, head_dim=head_dim)
    attention = UlyssesAttention(
        inner_backend=inner,
        process_group=None,
    ).to(device)

    for batch_size in [1, 2, 4, 8]:
        q = torch.randn(batch_size, seq_per_rank, num_heads, head_dim, device=device)
        k = torch.randn(batch_size, seq_per_rank, num_heads, head_dim, device=device)
        v = torch.randn(batch_size, seq_per_rank, num_heads, head_dim, device=device)

        output = attention(q, k, v, batch_size=batch_size, seq_len=seq_per_rank * world_size)
        assert output.shape == q.shape


def _logic_different_head_dims(rank, world_size):
    """Various head dims."""
    batch = 2
    seq_per_rank = 8
    num_heads = world_size * 4
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    for head_dim in [32, 64, 128]:
        inner = VanillaAttention(num_heads=num_heads // world_size, head_dim=head_dim)
        attention = UlyssesAttention(
            inner_backend=inner,
            process_group=None,
        ).to(device)

        q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)

        output = attention(q, k, v, batch_size=batch, seq_len=seq_per_rank * world_size)
        assert output.shape == q.shape


def _logic_world_size_4(rank, world_size):
    """4-GPU test."""
    batch = 2
    seq_per_rank = 16
    num_heads = world_size * 8  # 32 heads total
    head_dim = 64

    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")

    inner = VanillaAttention(num_heads=num_heads // world_size, head_dim=head_dim)
    attention = UlyssesAttention(
        inner_backend=inner,
        process_group=None,
    ).to(device)

    q = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    k = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)
    v = torch.randn(batch, seq_per_rank, num_heads, head_dim, device=device)

    output = attention(q, k, v, batch_size=batch, seq_len=seq_per_rank * world_size)
    assert output.shape == q.shape


# =============================================================================
# Test classes
# =============================================================================


class TestAllToAll4D:
    """Tests for all_to_all_4d function."""

    def test_all_to_all_4d_sequence_to_head(self):
        """Test sequence sharding to head sharding transformation."""
        run_test_in_distributed(world_size=2, test_fn=_logic_a2a_seq_to_head, use_cuda=True)

    def test_all_to_all_4d_head_to_sequence(self):
        """Test head sharding to sequence sharding transformation."""
        run_test_in_distributed(world_size=2, test_fn=_logic_a2a_head_to_seq, use_cuda=True)

    def test_all_to_all_4d_roundtrip(self):
        """Test that forward and backward all-to-all are inverses."""
        run_test_in_distributed(world_size=2, test_fn=_logic_a2a_roundtrip, use_cuda=True)

    def test_all_to_all_4d_single_process(self):
        """Test that single process returns input unchanged."""
        run_test_in_distributed(world_size=1, test_fn=_logic_a2a_single_process, use_cuda=True)


class TestUlyssesAttention:
    """Tests for UlyssesAttention module."""

    def test_ulysses_attention_initialization(self):
        """Test UlyssesAttention initialization."""
        run_test_in_distributed(world_size=2, test_fn=_logic_ulysses_init, use_cuda=True)

    def test_ulysses_attention_forward(self):
        """Test UlyssesAttention forward pass."""
        run_test_in_distributed(world_size=2, test_fn=_logic_ulysses_forward, use_cuda=True)

    def test_ulysses_attention_with_mask(self):
        """Test UlyssesAttention with attention mask."""
        run_test_in_distributed(world_size=2, test_fn=_logic_ulysses_with_mask, use_cuda=True)

    def test_ulysses_vs_standard_attention_single_gpu(self):
        """Compare UlyssesAttention with standard attention on single GPU."""
        if not MODULES_AVAILABLE:
            pytest.skip("Required modules not available")

        if not torch.cuda.is_available():
            pytest.skip("Test requires CUDA")

        batch = 2
        seq = 16
        num_heads = 8
        head_dim = 64
        device = torch.device("cuda:0")

        inner = VanillaAttention(num_heads=num_heads, head_dim=head_dim)
        ulysses_attn = UlyssesAttention(
            inner_backend=inner,
            process_group=None,
        ).to(device)

        torch.manual_seed(42)
        q = torch.randn(batch, seq, num_heads, head_dim, device=device)
        k = torch.randn(batch, seq, num_heads, head_dim, device=device)
        v = torch.randn(batch, seq, num_heads, head_dim, device=device)

        ulysses_output = ulysses_attn(q, k, v, batch_size=batch, seq_len=seq)

        q_std = q.transpose(1, 2)  # [B, H, S, D]
        k_std = k.transpose(1, 2)
        v_std = v.transpose(1, 2)

        std_output = F.scaled_dot_product_attention(
            q_std, k_std, v_std, scale=1.0 / math.sqrt(head_dim), dropout_p=0.0
        )
        std_output = std_output.transpose(1, 2).contiguous()  # [B, S, H, D]

        torch.testing.assert_close(
            ulysses_output,
            std_output,
            rtol=1e-4,
            atol=1e-4,
            msg="Ulysses attention output differs from standard attention",
        )

    def test_ulysses_vs_standard_attention_multi_gpu(self):
        """Compare UlyssesAttention across GPUs with standard attention on full sequence."""
        run_test_in_distributed(
            world_size=2, test_fn=_logic_ulysses_vs_standard_multi_gpu, use_cuda=True
        )

    def test_ulysses_attention_invalid_heads(self):
        """Test that invalid head count raises error."""
        run_test_in_distributed(world_size=2, test_fn=_logic_ulysses_invalid_heads, use_cuda=False)


class TestUlyssesAttentionEdgeCases:
    """Edge case tests for UlyssesAttention."""

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        run_test_in_distributed(world_size=2, test_fn=_logic_different_batch_sizes, use_cuda=True)

    def test_different_head_dims(self):
        """Test with various head dims."""
        run_test_in_distributed(world_size=2, test_fn=_logic_different_head_dims, use_cuda=True)

    def test_world_size_4(self):
        """Test with 4 GPUs."""
        run_test_in_distributed(world_size=4, test_fn=_logic_world_size_4, use_cuda=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
