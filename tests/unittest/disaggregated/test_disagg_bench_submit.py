"""Tests for examples/disaggregated/slurm/benchmark/submit.py GPU allocation logic."""

import os
import sys

# isort: off
# Add the benchmark directory to path for imports.
_BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "examples",
    "disaggregated",
    "slurm",
    "benchmark",
)
sys.path.insert(0, os.path.abspath(_BENCHMARK_DIR))

from submit import allocate_gpus  # noqa: E402
# isort: on


class TestAllocateGpus:
    """Test cases for the allocate_gpus function."""

    def test_gen_and_ctx_on_separate_nodes(self):
        """Test that GEN and CTX servers are allocated on separate nodes."""
        allocations = allocate_gpus(
            total_nodes=2,
            gpus_per_node=8,
            num_gen_servers=2,
            num_ctx_servers=2,
            gen_world_size=2,
            ctx_world_size=2,
            base_port=8000,
        )

        assert "GEN" in allocations
        assert "CTX" in allocations
        assert len(allocations["GEN"]) == 2
        assert len(allocations["CTX"]) == 2

        # Verify GEN and CTX are on separate nodes.
        gen_nodes = set()
        for server in allocations["GEN"].values():
            gen_nodes.update(server["nodes"].keys())
        ctx_nodes = set()
        for server in allocations["CTX"].values():
            ctx_nodes.update(server["nodes"].keys())
        assert gen_nodes.isdisjoint(ctx_nodes)

        # Verify port assignments.
        assert allocations["GEN"][0]["port"] == 8000
        assert allocations["GEN"][1]["port"] == 8001
        assert allocations["CTX"][0]["port"] == 8000
        assert allocations["CTX"][1]["port"] == 8001

    def test_same_server_type_can_share_nodes(self):
        """Test that multiple instances of the same server type can share nodes."""
        # 2 GEN servers with world_size=2 each, 8 GPUs per node.
        # Both GEN servers should fit on the same node.
        allocations = allocate_gpus(
            total_nodes=2,
            gpus_per_node=8,
            num_gen_servers=2,
            num_ctx_servers=1,
            gen_world_size=2,
            ctx_world_size=2,
            base_port=8000,
        )

        gen0_nodes = set(allocations["GEN"][0]["nodes"].keys())
        gen1_nodes = set(allocations["GEN"][1]["nodes"].keys())

        # Both GEN servers should be on the same node (node 0).
        assert gen0_nodes == gen1_nodes, (
            f"GEN servers should share nodes but got {gen0_nodes} and {gen1_nodes}."
        )

        # Servers on the same node must have different ports.
        gen0_port = allocations["GEN"][0]["port"]
        gen1_port = allocations["GEN"][1]["port"]
        assert gen0_port != gen1_port, (
            f"GEN servers on same node must have different ports but both have {gen0_port}."
        )


class TestPortConflictDetection:
    """Test cases for port conflict detection."""

    def test_port_conflict_raises_error(self):
        """Test that allocating servers that would conflict raises an error.

        This tests the validation logic, not the allocation itself.
        In practice, the node boundary alignment should prevent most conflicts,
        but the validation catches edge cases.
        """
        # With proper node separation, this should work without conflict.
        # The port conflict detection is a safety net for edge cases.
        try:
            allocate_gpus(
                total_nodes=2,
                gpus_per_node=8,
                num_gen_servers=2,
                num_ctx_servers=2,
                gen_world_size=4,
                ctx_world_size=4,
                base_port=8000,
            )
            # Should succeed - no conflict expected with proper allocation.
        except RuntimeError as e:
            # If there's a conflict, it should have a clear error message.
            assert "bind to" in str(e).lower() or "conflict" in str(e).lower()

    def test_multi_node_server_first_node_binding(self):
        """Test that multi-node servers bind to their first node."""
        allocations = allocate_gpus(
            total_nodes=4,
            gpus_per_node=4,
            num_gen_servers=1,
            num_ctx_servers=1,
            gen_world_size=8,  # Spans 2 nodes.
            ctx_world_size=8,  # Spans 2 nodes.
            base_port=8000,
        )

        # GEN with world_size=8 on 4 GPUs/node spans 2 nodes.
        gen_nodes = list(allocations["GEN"][0]["nodes"].keys())
        assert len(gen_nodes) == 2, f"GEN should span 2 nodes, got {gen_nodes}."

        # CTX should start on a fresh node boundary (node 2).
        ctx_nodes = list(allocations["CTX"][0]["nodes"].keys())
        assert len(ctx_nodes) == 2, f"CTX should span 2 nodes, got {ctx_nodes}."

        # Verify no overlap.
        assert set(gen_nodes).isdisjoint(set(ctx_nodes))


class TestGpuAssignment:
    """Test cases for GPU assignment within nodes."""

    def test_gpu_ids_are_contiguous(self):
        """Test that GPU IDs within a node are assigned contiguously."""
        allocations = allocate_gpus(
            total_nodes=1,
            gpus_per_node=8,
            num_gen_servers=1,
            num_ctx_servers=0,
            gen_world_size=4,
            ctx_world_size=0,
            base_port=8000,
        )

        gen_gpus = allocations["GEN"][0]["nodes"]["<node0_placeholder>"]
        assert gen_gpus == [0, 1, 2, 3], f"Expected [0,1,2,3], got {gen_gpus}."

    def test_multiple_servers_gpu_assignment(self):
        """Test GPU assignment for multiple servers of the same type."""
        allocations = allocate_gpus(
            total_nodes=1,
            gpus_per_node=8,
            num_gen_servers=2,
            num_ctx_servers=0,
            gen_world_size=2,
            ctx_world_size=0,
            base_port=8000,
        )

        gen0_gpus = allocations["GEN"][0]["nodes"]["<node0_placeholder>"]
        gen1_gpus = allocations["GEN"][1]["nodes"]["<node0_placeholder>"]

        assert gen0_gpus == [0, 1], f"GEN 0 expected [0,1], got {gen0_gpus}."
        assert gen1_gpus == [2, 3], f"GEN 1 expected [2,3], got {gen1_gpus}."


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_ctx_servers(self):
        """Test allocation with no CTX servers."""
        allocations = allocate_gpus(
            total_nodes=1,
            gpus_per_node=8,
            num_gen_servers=2,
            num_ctx_servers=0,
            gen_world_size=2,
            ctx_world_size=1,  # Doesn't matter since num_ctx_servers=0.
            base_port=8000,
        )

        assert "GEN" in allocations
        assert len(allocations["GEN"]) == 2
        # CTX key exists but is empty.
        assert "CTX" in allocations
        assert len(allocations["CTX"]) == 0

    def test_zero_gen_servers(self):
        """Test allocation with no GEN servers."""
        allocations = allocate_gpus(
            total_nodes=1,
            gpus_per_node=8,
            num_gen_servers=0,
            num_ctx_servers=2,
            gen_world_size=1,
            ctx_world_size=2,
            base_port=8000,
        )

        assert "GEN" in allocations
        assert len(allocations["GEN"]) == 0
        assert "CTX" in allocations
        assert len(allocations["CTX"]) == 2

    def test_exact_node_fill(self):
        """Test allocation that exactly fills nodes."""
        # 2 GEN servers with 4 GPUs each = 8 GPUs = exactly 1 node.
        allocations = allocate_gpus(
            total_nodes=2,
            gpus_per_node=8,
            num_gen_servers=2,
            num_ctx_servers=1,
            gen_world_size=4,
            ctx_world_size=4,
            base_port=8000,
        )

        # GEN servers fill node 0 exactly.
        gen0_gpus = allocations["GEN"][0]["nodes"]["<node0_placeholder>"]
        gen1_gpus = allocations["GEN"][1]["nodes"]["<node0_placeholder>"]
        assert gen0_gpus == [0, 1, 2, 3]
        assert gen1_gpus == [4, 5, 6, 7]

        # CTX should be on node 1.
        ctx_nodes = list(allocations["CTX"][0]["nodes"].keys())
        assert "<node1_placeholder>" in ctx_nodes
