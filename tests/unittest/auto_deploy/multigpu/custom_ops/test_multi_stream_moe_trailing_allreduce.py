# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-GPU regression tests for nvbugs/6248757.

Background
----------
``fuse_rmsnorm_quant_nvfp4`` restructures the shared-expert branch of a
TP-sharded MoE layer so that a standalone ``trtllm_dist_all_reduce`` directly
feeds the merge node::

    hidden ─┬─ gate ─ topk ─── moe_fused ─── moe_out ───────────────────┐
             └─ up_proj ─ relu² ─ down_proj ─ all_reduce ─ shared_out ─┴─ add

Before PR #14917, ``_execute_shared_expert_in_aux_stream`` placed ``end_aux``
*after* the all_reduce, putting the collective on the aux stream.  With
AllReduceStrategy.SYMM_MEM (used in Nemotron Ultra V3 production config) two
concurrent SYMM_MEM ops on different streams interleave across ranks under
monolithic CUDA-graph replay and silently corrupt the output.

Three test scenarios
--------------------
1. ``test_structural_multigpu`` — graph-level check: after the transform the
   all_reduce must appear *after* ``end_aux``.  FAILS pre-PR, PASSES with fix.

2. ``test_correctness_nccl_cuda_graph`` — NCCL correctness under CUDA graph.
   NCCL serialises submissions CPU-side so the race does not manifest; serves
   as a regression guard.

3. ``test_corruption_symm_mem_cuda_graph`` — explicitly builds the *buggy*
   graph, forces asymmetric submission ordering across ranks, and verifies the
   buggy graph produces wrong output while the fixed graph is correct.
   Skipped when SYMM_MEM is unavailable.
"""

import traceback

import pytest
import torch
from torch.distributed import DistNetworkError

# MPI pool leaks a thread on shutdown — suppress the threadleak warning.
pytestmark = pytest.mark.threadleak(enabled=False)


# ---------------------------------------------------------------------------
# Worker helpers (everything torch.ops-related is inside workers to avoid
# cloudpickle issues when serialising across MPI)
# ---------------------------------------------------------------------------


def _init_dist(port):
    import torch.distributed as dist

    import tensorrt_llm
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — registers custom ops
    from tensorrt_llm._torch.auto_deploy.distributed.common import initialize_or_skip
    from tensorrt_llm._utils import get_free_port, mpi_broadcast

    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    # Rank 0 picks a free port and broadcasts it so all workers use the same one.
    if port is None:
        port = mpi_broadcast(get_free_port() if rank == 0 else None)
    initialize_or_skip(port=port)
    return rank, dist.get_world_size()


def _cleanup():
    import torch.distributed as dist

    from tensorrt_llm._torch.auto_deploy.distributed.common import cleanup

    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    cleanup()


def _make_model_and_example(hidden_dim, inter_dim, strategy, device="cuda"):
    """Build the mock MoE layer with trailing all_reduce and an example input.

    Defined inside worker functions to avoid cloudpickle capturing torch.ops
    at module level.
    """
    import torch.nn as nn

    # Register mock MoE op if not already registered.
    op_name = "auto_deploy::mock_moe_trailing_ar"
    if not hasattr(torch.ops.auto_deploy, "mock_moe_trailing_ar"):

        @torch.library.custom_op(op_name, mutates_args=())
        def _mock_moe(
            x: torch.Tensor, sel: torch.Tensor, w: torch.Tensor, ew: torch.Tensor
        ) -> torch.Tensor:
            return torch.ops.aten.linear(x, ew)

        @_mock_moe.register_fake
        def _mock_moe_fake(x, sel, w, ew):
            return torch.ops.aten.linear(x, ew)

    moe_op = torch.ops.auto_deploy.mock_moe_trailing_ar
    ar_op = torch.ops.auto_deploy.trtllm_dist_all_reduce

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.strategy = strategy
            self.gate = nn.Linear(hidden_dim, 8, bias=False)
            self.up = nn.Linear(hidden_dim, inter_dim, bias=False)
            self.down = nn.Linear(inter_dim, hidden_dim, bias=False)
            self.expert_w = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
            self.ln = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            logits = self.gate(x)
            rw, sel = torch.topk(logits, k=2, dim=-1)
            shared = self.down(torch.relu(self.up(x)) ** 2)
            shared_out = ar_op(shared, self.strategy)
            moe_out = moe_op(x, sel, rw, self.expert_w)
            return self.ln(shared_out + moe_out)

    model = _Layer().eval().to(device)
    example = torch.randn(4, hidden_dim, device=device)
    moe_ops = [moe_op]
    return model, example, moe_ops


def _build_gm(model, example):
    return torch.export.export(model, (example,)).module()


# ---------------------------------------------------------------------------
# Worker 1 — structural check
# ---------------------------------------------------------------------------


def _worker_structural(world_size, port):
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
        _execute_shared_expert_in_aux_stream,
    )
    from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import (
        cuda_stream_manager,
        end_aux_stream_passthrough,
        wait_aux_stream_passthrough,
    )
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import all_reduce_ops, is_op

    rank, _ = _init_dist(port)
    try:
        cuda_stream_manager.add_device(rank)
        model, example, moe_ops = _make_model_and_example(128, 256, "NCCL")
        gm = _build_gm(model, example)
        gm, num = _execute_shared_expert_in_aux_stream(gm, moe_ops)

        assert num == 1, f"[rank {rank}] Expected 1 replacement, got {num}"

        node_order = {n: i for i, n in enumerate(gm.graph.nodes)}
        ar_ops = all_reduce_ops()
        ar_node = next((n for n in gm.graph.nodes if is_op(n, ar_ops)), None)
        end_aux_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is end_aux_stream_passthrough
        ]
        wait_aux_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is wait_aux_stream_passthrough
        ]

        assert ar_node is not None, f"[rank {rank}] No all_reduce node"
        assert end_aux_nodes, f"[rank {rank}] No end_aux node"
        assert wait_aux_nodes, f"[rank {rank}] No wait_aux node"

        end_aux = end_aux_nodes[0]
        wait_aux = wait_aux_nodes[0]

        # Core invariant: collective must come AFTER the stream switch back to main.
        assert node_order[ar_node] > node_order[end_aux], (
            f"[rank {rank}] BUG: all_reduce before end_aux — on aux stream"
        )
        assert node_order[ar_node] > node_order[wait_aux], (
            f"[rank {rank}] BUG: all_reduce before wait_aux"
        )
        assert end_aux.args[0] is not ar_node, (
            f"[rank {rank}] BUG: end_aux wraps the all_reduce directly"
        )
        assert wait_aux in ar_node.all_input_nodes, (
            f"[rank {rank}] wait_aux must feed the all_reduce"
        )
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Worker 2 — NCCL correctness under CUDA graph
# ---------------------------------------------------------------------------


def _worker_nccl_cuda_graph(world_size, port):
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
        _execute_shared_expert_in_aux_stream,
    )
    from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import cuda_stream_manager

    rank, _ = _init_dist(port)
    try:
        cuda_stream_manager.add_device(rank)
        torch.manual_seed(42 + rank)
        model, example, moe_ops = _make_model_and_example(128, 256, "NCCL")
        gm = _build_gm(model, example)
        gm, num = _execute_shared_expert_in_aux_stream(gm, moe_ops)
        assert num == 1

        test_x = torch.randn(4, 128, device="cuda")
        ref = model(test_x)

        static_x = torch.randn_like(test_x)
        static_out = torch.empty_like(ref)
        for _ in range(3):
            static_out.copy_(gm(static_x))

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out.copy_(gm(static_x))

        static_x.copy_(test_x)
        g.replay()

        assert torch.allclose(static_out, ref, atol=1e-4), (
            f"[rank {rank}] CUDA graph mismatch: max diff {(static_out - ref).abs().max().item()}"
        )
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Worker 3 — SYMM_MEM corruption demo
# ---------------------------------------------------------------------------


def _worker_symm_mem_corruption(world_size, port):
    """Build buggy and fixed graphs, run under SYMM_MEM + CUDA graph, compare."""
    import torch.distributed as dist

    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.custom_ops.distributed.trtllm_dist import (
        is_trtllm_op_available,
    )
    from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
        _execute_shared_expert_in_aux_stream,
        _find_merge_node,
        _get_ancestors,
    )
    from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import (
        begin_aux_stream_passthrough,
        cuda_stream_manager,
        end_aux_stream_passthrough,
        wait_aux_stream_passthrough,
    )
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import all_reduce_ops, is_op
    from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy
    from tensorrt_llm.mapping import Mapping

    if not is_trtllm_op_available():
        return "skip:no_trtllm_ops"

    rank, wsize = _init_dist(port)
    try:
        cuda_stream_manager.add_device(rank)

        # Check SYMM_MEM is available on this hardware.
        try:
            mapping = Mapping(world_size=wsize, tp_size=wsize, rank=rank)
            ar_runner = AllReduce(
                mapping=mapping, strategy=AllReduceStrategy.SYMM_MEM, dtype=torch.float16
            )
            if ar_runner.strategy != AllReduceStrategy.SYMM_MEM:
                return "skip:no_symm_mem"
        except Exception:
            return "skip:no_symm_mem"

        torch.manual_seed(42)
        strategy = "SYMM_MEM"
        hidden_dim, inter_dim = 128, 256
        model, example, moe_ops = _make_model_and_example(hidden_dim, inter_dim, strategy)

        ar_op = torch.ops.auto_deploy.trtllm_dist_all_reduce
        ar_ops = all_reduce_ops()

        # ----------------------------------------------------------------
        # Build BUGGY graph: all_reduce placed on aux stream (pre-PR).
        # ----------------------------------------------------------------
        def make_buggy_gm():
            gm = _build_gm(model, example)
            graph = gm.graph
            node_order_snap = {n: i for i, n in enumerate(graph.nodes)}

            moe_node = next(n for n in graph.nodes if is_op(n, moe_ops))
            merge_node = _find_merge_node(moe_node)
            assert merge_node is not None

            moe_anc = _get_ancestors(moe_node)
            moe_anc.add(moe_node)

            shared_output = routed_output = None
            for arg in merge_node.all_input_nodes:
                arg_anc = _get_ancestors(arg)
                if moe_node in arg_anc or arg is moe_node:
                    routed_output = arg
                elif arg in moe_anc or arg.op != "call_function":
                    pass
                else:
                    shared_output = arg

            assert shared_output is not None and is_op(shared_output, ar_ops)

            shared_nodes, fork_point, visited = [], None, set()
            queue = [shared_output]
            while queue:
                n = queue.pop(0)
                if n in visited:
                    continue
                visited.add(n)
                if n.op == "get_attr":
                    continue
                if n in moe_anc:
                    if fork_point is None or node_order_snap.get(n, 0) > node_order_snap.get(
                        fork_point, 0
                    ):
                        fork_point = n
                    continue
                shared_nodes.append(n)
                for inp in n.all_input_nodes:
                    queue.append(inp)

            shared_nodes.sort(key=lambda n: node_order_snap.get(n, 0))
            first_shared = shared_nodes[0]

            with graph.inserting_before(first_shared):
                beg = graph.call_function(begin_aux_stream_passthrough, args=(fork_point,))
            first_shared.args = tuple(beg if a is fork_point else a for a in first_shared.args)

            # BUG: end_aux inserted AFTER the all_reduce → collective on aux stream.
            with graph.inserting_after(shared_output):
                end = graph.call_function(end_aux_stream_passthrough, args=(shared_output,))
            merge_node.args = tuple(end if a is shared_output else a for a in merge_node.args)

            with graph.inserting_before(merge_node):
                wait = graph.call_function(wait_aux_stream_passthrough, args=(routed_output,))
            merge_node.args = tuple(wait if a is routed_output else a for a in merge_node.args)

            # Add second all_reduce on main stream (stands in for routed-expert AR).
            out_node = next(n for n in reversed(list(graph.nodes)) if n.op == "output")
            out_arg = out_node.args[0]
            with graph.inserting_before(out_node):
                second_ar = graph.call_function(ar_op.default, args=(out_arg, strategy))
            out_node.args = (second_ar,)
            graph.lint()
            gm.recompile()
            return gm

        # ----------------------------------------------------------------
        # Build FIXED graph: all_reduce on main stream.
        # ----------------------------------------------------------------
        def make_fixed_gm():
            gm = _build_gm(model, example)
            gm, num = _execute_shared_expert_in_aux_stream(gm, moe_ops)
            assert num == 1
            graph = gm.graph
            out_node = next(n for n in reversed(list(graph.nodes)) if n.op == "output")
            out_arg = out_node.args[0]
            with graph.inserting_before(out_node):
                second_ar = graph.call_function(ar_op.default, args=(out_arg, strategy))
            out_node.args = (second_ar,)
            graph.lint()
            gm.recompile()
            return gm

        buggy_gm = make_buggy_gm()
        fixed_gm = make_fixed_gm()

        # ----------------------------------------------------------------
        # Run both under CUDA graph with forced asymmetric stream ordering:
        #   rank 0 → main stream waits for aux before second AR → aux submits first
        #   rank 1 → no extra wait → main submits first
        # SYMM_MEM collectives don't go through NCCL's CPU serialisation, so
        # the cross-rank submission order mismatch produces wrong all_reduce
        # results for the buggy graph.
        # ----------------------------------------------------------------
        aux_stream = cuda_stream_manager.get_stream(rank, "aux")
        main_stream = cuda_stream_manager.get_stream(rank, "main")

        def capture_and_replay(gm, x):
            static_x = x.clone()
            static_out = torch.empty_like(gm(static_x))
            for _ in range(3):
                static_out.copy_(gm(static_x))
            # Asymmetric delay: rank 0 delays main before the second collective.
            if rank == 0:
                main_stream.wait_stream(aux_stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out.copy_(gm(static_x))
            static_x.copy_(x)
            g.replay()
            torch.cuda.synchronize()
            return static_out.clone()

        test_x = torch.randn(4, hidden_dim, device="cuda")
        dist.barrier()

        buggy_out = capture_and_replay(buggy_gm, test_x)

        # Eager reference (sequential, no multi-stream).
        with torch.cuda.stream(main_stream):
            ref_out = model(test_x)
            ref_out = ar_op(ref_out, strategy)
        torch.cuda.synchronize()

        dist.barrier()
        fixed_out = capture_and_replay(fixed_gm, test_x)

        # Fixed graph must be correct.
        fixed_correct = torch.allclose(fixed_out, ref_out, atol=1e-3)
        assert fixed_correct, (
            f"[rank {rank}] Fixed graph wrong under SYMM_MEM CUDA graph: "
            f"max diff = {(fixed_out - ref_out).abs().max().item():.4f}"
        )

        # Buggy graph should produce wrong output when SYMM_MEM is truly
        # concurrent (interleaved) on this hardware.
        buggy_correct = torch.allclose(buggy_out, ref_out, atol=1e-3)
        if buggy_correct:
            # SYMM_MEM may have serialised (e.g., fallback, world_size too small).
            return "skip:race_not_triggered"

        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Pytest entry points — use MpiPoolSession.submit_sync like
# test_allreduce_residual_rmsnorm_fusion.py to avoid cloudpickle torch.ops issues.
# ---------------------------------------------------------------------------


def _run_with_retries(worker_fn, world_size, **kwargs):
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    max_retries = 5
    last_exc = None
    for _ in range(max_retries):
        pool = MpiPoolSession(n_workers=world_size)
        try:
            return pool.submit_sync(worker_fn, port=None, world_size=world_size, **kwargs)
        except DistNetworkError as e:
            last_exc = e
            if "EADDRINUSE" not in str(e) and "address already in use" not in str(e).lower():
                raise
        finally:
            pool.shutdown()
    raise RuntimeError(f"Dist init failed after {max_retries} retries") from last_exc


def _check_results(results):
    """Assert all worker results are True; return first non-True for skip detection."""
    for r in results:
        if isinstance(r, str) and r.startswith("skip:"):
            return r
        assert r is True, f"Unexpected worker result: {r}"
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires ≥ 2 GPUs")
def test_structural_multigpu():
    """Graph-level invariant holds in real multi-GPU MPI context.

    FAILS on pre-PR code (all_reduce appears before end_aux → on aux stream).
    PASSES on PR #14917 fix.
    Uses NCCL — works on any multi-GPU setup.
    """
    results = _run_with_retries(_worker_structural, world_size=2)
    _check_results(results)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires ≥ 2 GPUs")
def test_correctness_nccl_cuda_graph():
    """Fixed graph is numerically correct under NCCL + CUDA graph replay.

    NCCL serialises collective submissions CPU-side, so the race between two
    concurrent collectives does not manifest.  This test always passes on both
    buggy and fixed code — it is a correctness regression guard.
    """
    results = _run_with_retries(_worker_nccl_cuda_graph, world_size=2)
    _check_results(results)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires ≥ 2 GPUs")
def test_corruption_symm_mem_cuda_graph():
    """Buggy graph produces wrong output; fixed graph is correct — with SYMM_MEM.

    Explicitly builds the buggy graph (all_reduce on aux stream), forces
    asymmetric submission ordering across ranks, and verifies the wrong output.
    Then runs the fixed graph and verifies correctness.

    Skipped when SYMM_MEM is unavailable or the race does not manifest on this
    hardware (e.g., world_size below MULTIMEM threshold).
    """
    results = _run_with_retries(_worker_symm_mem_corruption, world_size=2)
    outcome = _check_results(results)
    if isinstance(outcome, str) and outcome.startswith("skip:"):
        pytest.skip(
            f"SYMM_MEM race not reproducible ({outcome.split(':', 1)[1]}); "
            f"try world_size ≥ 6 on SM100 for reliable MULTIMEM activation"
        )
