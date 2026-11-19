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
"""``rubin_latent_ar_norm``: CuTe one-shot MNNVL all-reduce + RMSNorm vs the C++ kernel.

On every rank: the CuTe row kernel (eagerly and under CUDA graph replay) must match
``mnnvl_fusion_allreduce`` (``RMS_NORM``) bit for bit while it shares its Lamport workspace with
the C++ one-shot and two-shot kernels in any order. The row kernel re-polls only the slots still
missing and rotates the buffers while it polls, so it also runs with the ranks arriving in varying
orders.
"""

import os
import traceback

import pytest
import torch

import tensorrt_llm
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../.."]
from _k31_utils import (
    on_sm,
    pickle_by_value,  # noqa: E402 — needs the path declared above
)

pickle_by_value(__name__)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 3584
EPS = 1e-6
MAX_TOKENS = 64
#: The targets' ``latent_ar_oneshot_max_bytes``: one-shot rows of the own workspace.
ONESHOT_MAX_BYTES = 6 << 20
ROWS = (1, 2, 7, 14, 16, 18)


def _inputs(world, num_tokens, seed):
    """Every rank's partial; seed 2 adds large values, exact zeros and BF16 -0."""
    gen = torch.Generator(device="cuda")
    xs = []
    for r in range(world):
        gen.manual_seed(seed * 1009 + r)
        x = torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda")
        if seed % 3 == 2:
            x[:, ::7] *= 300.0
            x[:, 1::11] = 0.0
            x[:, 2::13] = -0.0
        xs.append(x.bfloat16())
    return xs


def _ar_norm_rank(world):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    try:
        from tensorrt_llm._torch.distributed import AllReduce
        from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import (
            rubin_latent_ar_norm as kan,
        )
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.comm.rubin_comm import (  # noqa: E501
            rubin_latent_ar_workspace,
        )
        from tensorrt_llm.functional import AllReduceStrategy

        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16)
        assert ar.mnnvl_allreduce is not None
        shared_ws = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        own = rubin_latent_ar_workspace(mapping, world, HIDDEN, MAX_TOKENS, ONESHOT_MAX_BYTES)
        comm = shared_ws["comm"]
        torch.manual_seed(0)
        gamma = (torch.rand(HIDDEN, device="cuda") + 0.5).bfloat16()

        def cpp(x, op, workspace):
            if workspace is own:
                buf, flags = own["buffer"], own["flags"]
            else:
                buf = workspace["uc_buffer"].view(torch.bfloat16).view(3, -1)
                flags = workspace["buffer_flags"]
            norm = op != AllReduceFusionOp.NONE
            (out,) = torch.ops.trtllm.mnnvl_fusion_allreduce(
                x,
                gamma if norm else None,
                None,
                EPS if norm else None,
                buf,
                flags,
                norm,
                None,
                int(op),
            )
            return out

        def row_ar(x):
            out = torch.empty_like(x)
            kan.run_k3_latent_ar_norm_row(
                x,
                gamma,
                EPS,
                out,
                uc_addr=own["buffer"].data_ptr(),
                mc_addr=own["mc"].data_ptr(),
                flags=own["flags"],
                rank=rank,
                world_size=world,
            )
            return out

        def same(a, b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

        # Bit-identical to the C++ kernel; the C++ one-shot and two-shot kernels use the same
        # workspace before and after it.
        for num_tokens in ROWS:
            for seed in range(3):
                x = _inputs(world, num_tokens, seed)[rank]
                ref = cpp(x, AllReduceFusionOp.RMS_NORM, shared_ws)
                same(cpp(x, AllReduceFusionOp.RMS_NORM, own), ref)
                # Two-shot sizes on the same workspace in between.
                big = _inputs(world, MAX_TOKENS, seed + 10)[rank]
                same(
                    cpp(big, AllReduceFusionOp.RMS_NORM, own),
                    cpp(big, AllReduceFusionOp.RMS_NORM, shared_ws),
                )
                # The row kernel on the same workspace (it re-arms the previous call's buffer).
                same(row_ar(x), ref)
                same(row_ar(x), ref)
                same(
                    cpp(big, AllReduceFusionOp.RMS_NORM, own),
                    cpp(big, AllReduceFusionOp.RMS_NORM, shared_ws),
                )
                same(row_ar(x), ref)
        torch.cuda.synchronize()

        # The row kernel with the ranks arriving in varying orders (each rank sleeps a drawn number
        # of cycles first; the same draws on every rank): its re-polls of the missing slots and its
        # buffer rotation while it polls.
        delay_gen = torch.Generator().manual_seed(1234)
        for num_tokens in (1, 7, 14):
            for trial in range(6):
                x = _inputs(world, num_tokens, trial % 3)[rank]
                ref = cpp(x, AllReduceFusionOp.RMS_NORM, shared_ws)
                delays = torch.randint(0, 40000, (world,), generator=delay_gen).tolist()
                torch.cuda.synchronize()
                comm.Barrier()
                torch.cuda._sleep(delays[rank])
                same(row_ar(x), ref)
        torch.cuda.synchronize()

        # CUDA graph: the row kernel three times in a row per replay, each on the previous one's
        # output, new inputs each replay.
        for num_tokens in (7, 14):
            static_x = torch.empty((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            static_x.copy_(_inputs(world, num_tokens, 0)[rank])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                row_ar(static_x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outs = []
                    src = static_x
                    for i in range(3):
                        outs.append(row_ar(src))
                        src = outs[-1]
            torch.cuda.synchronize()
            for step in range(3):
                static_x.copy_(_inputs(world, num_tokens, 30 + step)[rank])
                comm.Barrier()
                graph.replay()
                torch.cuda.synchronize()
                ref = static_x
                for out in outs:
                    ref = cpp(ref, AllReduceFusionOp.RMS_NORM, shared_ws)
                    same(out, ref)
            del graph

        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


@on_sm(100, 103, 107)
@pytest.mark.gpu4
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_latent_ar_norm(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    results = mpi_pool_executor.map(_ar_norm_rank, *zip(*[(world,)] * world))
    assert all(r is True for r in results)
