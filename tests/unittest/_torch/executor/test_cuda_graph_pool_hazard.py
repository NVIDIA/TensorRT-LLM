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
"""Reproduces the hazard that CUDAGraphRunner.clear() must avoid.

`clear()` releases the C++ MoE workspaces before `torch.cuda.empty_cache()`
erases the CUDA graph's private memory pool. This test shows why that ordering
matters, without a model, without MoE and without TRT-LLM in the picture: an
allocation made during graph capture and still held when the pool is torn down
faults the caching allocator when it is finally freed.

The reproduction needs two things together, and neither alone is enough:

    expandable_segments   pre-expansion   result
    -------------------   -------------   ------
    on                    yes             SIGSEGV
    on                    no              clean
    off                   yes             clean
    off                   no              clean

`expandable_segments:True` is also a measured necessary condition on the real
crash this guards against: the same Inkling NVFP4 run that segfaults reliably
runs clean with that allocator mode switched off and no other change.

A SIGSEGV cannot be caught in-process, so each arm runs in a fresh subprocess
and the assertion is on its exit status. Fresh matters -- an already initialised
CUDA context changes the allocator layout that is under test.
"""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

# One large allocation that grows the segment once, then freed so everything
# below is carved out of that space without growing it again.
_PRE_EXPAND_MIB = 256
_N_BLOCKS = 64
_BLOCK_BYTES = 2 << 20

_SCRIPT = textwrap.dedent("""
    import gc
    import sys

    import torch

    RELEASE_EARLY = sys.argv[1] == "release_early"

    x = torch.zeros(1024, device="cuda")
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    blocks = []
    with torch.cuda.graph(graph):
        pre = torch.empty({pre_expand} << 20, dtype=torch.int8, device="cuda")
        del pre
        for _ in range({n_blocks}):
            blocks.append(torch.empty({block_bytes}, dtype=torch.int8, device="cuda"))
        x.add_(1.0)
    torch.cuda.synchronize()

    # Hold a scattered subset, so the freed blocks between them cannot coalesce.
    # This stands in for the C++ FusedMoeRunner holding its workspace tensor.
    held = blocks[::2]
    del blocks
    gc.collect()

    if RELEASE_EARLY:
        # What the fix does: drop the held allocations while the pool still
        # exists, before empty_cache() tears it down.
        del held
        held = None
        gc.collect()

    # Exactly CUDAGraphRunner.clear(): reset the graphs, drop the pool handle,
    # then empty_cache().
    del x
    graph.reset()
    del graph
    gc.collect()
    torch.cuda.empty_cache()

    if held is not None:
        del held
        gc.collect()
        torch.cuda.empty_cache()

    print("survived")
""").format(pre_expand=_PRE_EXPAND_MIB,
            n_blocks=_N_BLOCKS,
            block_bytes=_BLOCK_BYTES)


def _run(arm: str, expandable: bool) -> int:
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"} if expandable else {}
    return subprocess.run([sys.executable, "-c", _SCRIPT, arm],
                          env={**os.environ, **env},
                          capture_output=True,
                          timeout=300).returncode


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_releasing_before_pool_teardown_avoids_the_fault():
    """The ordering clear() implements: release first, and the teardown is safe."""
    assert _run("release_early", expandable=True) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_holding_across_pool_teardown_faults():
    """Holding a capture-pool allocation past the teardown crashes the allocator.

    This is the hazard clear() closes. If a future PyTorch release fixes the
    allocator, this assertion is the thing that will tell us -- at which point
    the early release becomes belt-and-braces rather than load-bearing, and this
    test should be revisited rather than silently relaxed.
    """
    rc = _run("hold", expandable=True)
    assert rc != 0, ("expected the held allocation to fault the allocator; "
                     "if this now passes cleanly, the PyTorch allocator behaviour "
                     "has changed and CUDAGraphRunner.clear() should be re-examined")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fault_requires_expandable_segments():
    """Control: the identical arm is clean without expandable_segments.

    Pins down which allocator mode the hazard belongs to, and keeps the test
    above from being read as "holding an allocation is always fatal".
    """
    assert _run("hold", expandable=False) == 0
