# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The graph-pool hazard driven through the real FusedMoeRunner, no model.

A companion file used to hold the capture-pool block in a Python list, which
proved the allocator bug but not that TRT-LLM's own holder reaches it. That file
(`test_cuda_graph_pool_hazard.py`) is no longer in the tree, so this is the only
reproducer left, and the only place the C++ holder is exercised at all. The real
holder is C++: `FusedMoeRunner` caches its workspace in
`std::map<cudaStream_t, WorkspaceInfo>`, reached from `MoERunner.runner_dict` --
a class attribute, so it outlives the executor that captured the graph. That is
what `CUDAGraphRunner.clear()` releases, and this file is what asserts the
release matters.

It is also the answer to "is this model-specific?". The geometry is a parameter,
so the same three arms run with a small shape and with the routed-expert
configuration of a real MoE model that is NOT the one the bug was found on. If
the verdict moved with the geometry, the diagnosis would be wrong.

Measured: fault 2/2 and fix 0/2 for both geometries, and the large one raises
PyTorch's own assertion at the decrement site --

    RuntimeError: block->pool->owner_PrivatePool->cudaMalloc_count > 0
    INTERNAL ASSERT FAILED at ".../c10/cuda/CUDACachingAllocator.cpp"

Each arm is a fresh subprocess: a SIGSEGV cannot be caught in-process, and an
already initialised CUDA context changes the allocator layout under test. That
makes the file cost about 50 s per arm, essentially all of it `import
tensorrt_llm`, so the arm count is kept down: four for the fault and its ordered
release, three for the reservation rule, and two for the trigger that is not
teardown at all. The reservation cannot be checked anywhere cheaper, because it
lives in moeOp.cpp.

One of those runs with expandable_segments OFF on purpose. The earlier off-pool
rule was measured only with it ON and raises cudaErrorStreamCaptureInvalidated
without it, which is TRT-LLM's default -- a whole configuration went unmeasured
because every arm shared one setting. There is no cheaper place to check it any
more: the model-free control that used to carry it went with
test_cuda_graph_pool_hazard.py.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

# One expansion inside the capture, carved into large-pool blocks, a scattered
# subset held. That shape is what makes the pool's cudaMalloc_count reach zero
# with live blocks outstanding; the MoE workspace is allocated inside the same
# capture and is what the C++ side keeps.
_PRE_EXPAND_MIB = 256
_N_BLOCKS = 64
_BLOCK_BYTES = 2 << 20

_SCRIPT = textwrap.dedent("""
    import gc
    import os
    import sys

    import torch

    import tensorrt_llm  # noqa: F401  (registers the trtllm custom ops)
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MoERunner

    arm = sys.argv[1]
    num_experts = int(os.environ["MOE_EXPERTS"])
    hidden = int(os.environ["MOE_HIDDEN"])
    inter = int(os.environ["MOE_INTER"])
    top_k = int(os.environ["MOE_TOPK"])
    dtype = torch.bfloat16

    def make_inputs(n_tokens=8):
        return dict(
            x=torch.randn(n_tokens, hidden, dtype=dtype, device="cuda"),
            w1=torch.randn(num_experts, 2 * inter, hidden, dtype=dtype, device="cuda"),
            w2=torch.randn(num_experts, hidden, inter, dtype=dtype, device="cuda"),
            sel=torch.randint(0, num_experts, (n_tokens, top_k), dtype=torch.int32,
                              device="cuda"),
            scl=torch.ones(n_tokens, top_k, dtype=torch.float32, device="cuda"),
        )

    def moe(t):
        return torch.ops.trtllm.fused_moe(
            t["x"], t["sel"], t["scl"], t["w1"], None, t["w2"], None, dtype,
            quant_scales=[], tp_size=1, tp_rank=0, ep_size=1, ep_rank=0,
            cluster_size=1, cluster_rank=0, enable_alltoall=False,
        )

    t = make_inputs()
    moe(t)                       # eager once, so the runner for this shape exists
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    filler = []
    with torch.cuda.graph(graph):
        pre = torch.empty({pre_expand} << 20, dtype=torch.int8, device="cuda")
        del pre
        for _ in range({n_blocks}):
            filler.append(torch.empty({block_bytes}, dtype=torch.int8, device="cuda"))
        out = moe(t)             # the C++ workspace is now a block of THIS pool
    torch.cuda.synchronize()

    held = filler[::2]
    del filler
    gc.collect()

    if arm == "second_capture":
        # No teardown at all. torch.cuda.graph.__enter__ calls
        # torch.cuda.empty_cache() before every capture, which is the same
        # release_cached_blocks() that erases PrivatePools -- so opening a
        # SECOND capture can erase the first graph's pool while the C++ map
        # still holds a workspace carved out of it. Ordering the teardown
        # cannot cover this; it does not happen at teardown.
        second = torch.cuda.CUDAGraph()
        with torch.cuda.graph(second):
            out2 = moe(t)
        torch.cuda.synchronize()
        del out2
        second.reset()
        del second
        MoERunner.clear_all_workspaces()
        gc.collect()
        torch.cuda.empty_cache()
        print("survived")
        raise SystemExit(0)

    if arm == "fix":
        # What CUDAGraphRunner.clear() does: drop the C++ workspaces while the
        # pool that backs them is still alive.
        MoERunner.clear_all_workspaces()

    # Exactly CUDAGraphRunner.clear().
    del out, t
    graph.reset()
    del graph
    del held
    gc.collect()
    torch.cuda.empty_cache()

    # The step that faults on the real run: the C++ map still holds a workspace
    # carved out of the pool that empty_cache() just erased.
    MoERunner.clear_all_workspaces()
    gc.collect()
    torch.cuda.empty_cache()
    print("survived")
""").format(pre_expand=_PRE_EXPAND_MIB, n_blocks=_N_BLOCKS, block_bytes=_BLOCK_BYTES)

# A toy shape, and the routed-expert configuration of a real MoE model that is
# not the one this bug was found on.
_GEOMETRIES = {
    "toy": dict(MOE_EXPERTS="4", MOE_HIDDEN="512", MOE_INTER="1024", MOE_TOPK="2"),
    # 288 experts / top-8 is the routing shape of a real MoE model that is not
    # the one this bug was found on; hidden and intermediate are scaled down so
    # the weights are ~0.6 GB instead of ~9 GB. The full-size version was
    # measured too (fault 2/2, fix 0/2) -- the routing shape is what makes the
    # "different model" point, and the weight size only costs lane time.
    "large": dict(MOE_EXPERTS="288", MOE_HIDDEN="1024", MOE_INTER="512", MOE_TOPK="8"),
}


# This PR makes the reservation the default, so the arms that reproduce the
# pre-fix behaviour have to switch it off explicitly. Without this they would
# exercise the fix and the fault assertions would silently invert.
_RESERVE_OFF = {"TLLM_MOE_CAPTURE_WORKSPACE_RESERVE": "0"}


def _run(
    arm: str,
    geometry: str,
    *,
    alloc_conf: str = "expandable_segments:True",
    extra_env: dict | None = None,
) -> int:
    env = dict(os.environ, **_GEOMETRIES[geometry], PYTORCH_CUDA_ALLOC_CONF=alloc_conf)
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT, arm], env=env, capture_output=True, timeout=900
    ).returncode


requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("geometry", sorted(_GEOMETRIES))
def test_workspace_held_across_pool_teardown_faults(geometry):
    """The C++ workspace surviving the pool teardown faults the allocator.

    Parametrized over the geometry on purpose: the bug is in the allocator, not
    in any model, so the verdict must not depend on the expert count or hidden
    size. If one geometry stops faulting, the diagnosis needs re-examining
    rather than the test relaxing.
    """
    assert _run("hazard", geometry, extra_env=_RESERVE_OFF) != 0, (
        f"[{geometry}] expected the held MoE workspace to fault the allocator; "
        "if this now passes, the PyTorch allocator behaviour has changed and "
        "CUDAGraphRunner.clear() should be re-examined"
    )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("geometry", sorted(_GEOMETRIES))
def test_releasing_workspaces_first_avoids_the_fault(geometry):
    """clear_all_workspaces() before the teardown, which is what clear() does."""
    assert _run("fix", geometry, extra_env=_RESERVE_OFF) == 0


# The reservation arms run the HAZARD script -- no clear_all_workspaces() before
# the teardown -- because that is the claim: with the workspace allocated
# eagerly it is not a block of the graph's private pool, so erasing the pool
# cannot strand it and the teardown order stops mattering for this fault.
@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("geometry", sorted(_GEOMETRIES))
def test_reserved_workspace_survives_the_pool_teardown(geometry):
    """The reservation removes the fault without the ordered release."""
    assert _run("hazard", geometry, extra_env={"TLLM_MOE_CAPTURE_WORKSPACE_RESERVE": "1"}) == 0, (
        f"[{geometry}] the reserved workspace still faulted the allocator; it is "
        "supposed to be an ordinary eager block that the private pool does not own"
    )


@pytest.mark.gpu
@requires_gpu
def test_reserved_workspace_does_not_break_capture_without_expandable_segments():
    """The configuration that killed the off-pool rule.

    That rule allocated the capture-time workspace on a pool stream, which
    raises cudaErrorStreamCaptureInvalidated unless expandable_segments is on --
    and it is not TRT-LLM's default. The reservation allocates nothing during
    capture at all, so the capture has to complete here. The hazard itself is
    not expected either way in this configuration; what is under test is that
    the capture survives.
    """
    assert (
        _run(
            "fix",
            "toy",
            alloc_conf="",
            extra_env={"TLLM_MOE_CAPTURE_WORKSPACE_RESERVE": "1"},
        )
        == 0
    ), (
        "the capture failed with the reservation on and expandable_segments off; "
        "an allocation is reaching the capturing stream that should not be"
    )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize(
    "reserve", [pytest.param("0", id="shipped"), pytest.param("1", id="reserved")]
)
def test_pool_erased_by_a_later_capture(reserve):
    """The trigger that is not teardown, and that teardown ordering cannot cover.

    torch.cuda.graph.__enter__ runs torch.cuda.empty_cache() before every
    capture. A workspace cached from the first capture is a block of the first
    graph's pool, and opening the second capture is enough to erase that pool.
    The reservation is an ordinary eager block and is not in any pool, so it has
    nothing to lose here.

    Not asserted as a fault for the shipped rule: whether the erase lands
    depends on the pool reaching cudaMalloc_count == 0, which the geometry only
    makes likely. What is asserted is the reservation surviving; the shipped arm
    is recorded for comparison.
    """
    rc = _run("second_capture", "toy", extra_env={"TLLM_MOE_CAPTURE_WORKSPACE_RESERVE": reserve})
    if reserve == "1":
        assert rc == 0, (
            "the reservation faulted when a later capture erased the first graph's "
            "pool; it is supposed to be an ordinary block that no pool owns"
        )
