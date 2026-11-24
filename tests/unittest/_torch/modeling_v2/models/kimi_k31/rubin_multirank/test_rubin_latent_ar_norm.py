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

On every rank: the CuTe row kernel (early and late PDL trigger, eagerly and under CUDA graph
replay) must match ``mnnvl_fusion_allreduce`` (``RMS_NORM``, and ``NONE`` for the plain sum) bit
for bit, its MXFP8 side output must be the low-M FP8 GEMM's quantization of the normalized rows,
while it shares its Lamport workspace with the C++ one-shot and two-shot kernels in any order. The
16-rank chunked summation order is checked on the available GPUs by filling the other ranks' slots
of the Lamport buffer through the multicast address (sum only, against an FP32 reference). The
row kernel re-polls only the slots still missing and rotates the buffers while it polls, so it
also runs with the ranks arriving in varying orders.
"""

import os
import pickle
import sys
import traceback

import _torch.modeling_v2.models.kimi_k31.kernels.gemm.rubin_mxfp8_test_utils as mxfp8_ref
import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm.functional import AllReduceFusionOp
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(mxfp8_ref)
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

#: This GPU's compute capability; None without one (the tests below then skip).
_CAP = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

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
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm import (
            rubin_latent_ar_norm as kan,
        )
        from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_comm import (  # noqa: E501
            rubin_latent_ar_workspace,
        )
        from tensorrt_llm._torch.distributed import AllReduce
        from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
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

        def row_ar(x, early=True, norm=True, world_size=world, mxfp8=False, early_state=False):
            out = torch.empty_like(x)
            mx = None
            if mxfp8:
                mx = (
                    torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device),
                    torch.empty((x.shape[0], HIDDEN // 32), dtype=torch.uint8, device=x.device),
                )
            kan.run_k3_latent_ar_norm_row(
                x,
                gamma,
                EPS,
                out,
                uc_addr=own["buffer"].data_ptr(),
                mc_addr=own["mc"].data_ptr(),
                flags=own["flags"],
                rank=rank,
                world_size=world_size,
                early_trigger=early,
                norm=norm,
                mxfp8_out=mx,
                early_state=early_state,
            )
            return out if mx is None else (out, *mx)

        def same(a, b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

        # Bit-identical to the C++ kernel; the C++ one-shot and two-shot kernels use the same
        # workspace before and after it.
        for num_tokens in ROWS:
            for seed in range(3):
                x = _inputs(world, num_tokens, seed)[rank]
                ref = cpp(x, AllReduceFusionOp.RMS_NORM, shared_ws)
                ref_sum = cpp(x, AllReduceFusionOp.NONE, shared_ws)
                same(cpp(x, AllReduceFusionOp.RMS_NORM, own), ref)
                # The MXFP8 side output: the low-M GEMM's quantization of the normalized rows.
                exq, exs = mxfp8_ref.mxfp8_quantize_reference(ref)
                # Two-shot sizes on the same workspace in between.
                big = _inputs(world, MAX_TOKENS, seed + 10)[rank]
                same(
                    cpp(big, AllReduceFusionOp.RMS_NORM, own),
                    cpp(big, AllReduceFusionOp.RMS_NORM, shared_ws),
                )
                # The row kernel on the same workspace (it re-arms the previous call's buffer).
                assert kan.can_run_row(x, gamma, world, own["buffer_bytes"])
                assert not kan.can_run_row(x, gamma, world, num_tokens * world * HIDDEN * 2 - 16)
                same(row_ar(x), ref)
                same(row_ar(x, early=False), ref)
                # early_state, as the column-parallel MoE tail runs it: state, count and re-arm
                # before the grid wait.
                same(row_ar(x, early_state=True), ref)
                same(row_ar(x, early=False, early_state=True), ref)
                same(row_ar(x), ref)
                same(row_ar(x, norm=False), ref_sum)
                out, xq, xs = row_ar(x, mxfp8=True)
                same(out, ref)
                assert torch.equal(xq.view(torch.uint8), exq.view(torch.uint8))
                assert torch.equal(xs, exs)
                out, xq, xs = row_ar(x, mxfp8=True, early_state=True)
                same(out, ref)
                assert torch.equal(xq.view(torch.uint8), exq.view(torch.uint8))
                assert torch.equal(xs, exs)
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
                same(row_ar(x, early_state=trial % 2 > 0), ref)
        torch.cuda.synchronize()

        # CUDA graph: the row kernel three times in a row per replay, each on the previous one's
        # output, new inputs each replay. early_state only on the first launch: its previous call
        # on the workspace (the last launch of the previous replay) has completed, while the next
        # ones' producer is that call itself.
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
                        outs.append(row_ar(src, early_state=i == 0))
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

        # 16 ranks on `world` GPUs (sum only): rank r also stands for ranks r + world, ...;
        # their slots of the current buffer are filled through the multicast address first.
        if 16 % world == 0 and world < 16:
            buf_bytes = int(own["flags"][2].item())
            mc = own["mc"].view(torch.bfloat16)

            def fill_other_slots(xs, num_tokens):
                torch.cuda.synchronize()
                start = int(own["flags"][0].item()) * buf_bytes // 2
                stage = mc[start : start + num_tokens * 16 * HIDDEN].view(num_tokens, 16, HIDDEN)
                for v in range(rank + world, 16, world):
                    stage[:, v, :].copy_(torch.where(xs[v] == 0, torch.zeros_like(xs[v]), xs[v]))

            for num_tokens in (1, 7, 16):
                if num_tokens * 16 * HIDDEN * 2 > buf_bytes:
                    # The workspace is sized for the real world size.
                    continue
                xs = _inputs(16, num_tokens, 3)
                # The FP32 reference: two chunks of eight ranks, each summed from 0.
                chunks = []
                for base in (0, 8):
                    c = torch.zeros((num_tokens, HIDDEN), dtype=torch.float32, device="cuda")
                    for v in range(base, base + 8):
                        c = c + xs[v].float()
                    chunks.append(c)
                expect = ((torch.zeros_like(chunks[0]) + chunks[0]) + chunks[1]).bfloat16()
                # The row kernel (its own slot from the registers) on the 16-rank layout.
                fill_other_slots(xs, num_tokens)
                torch.cuda.synchronize()
                comm.Barrier()
                same(row_ar(xs[rank], norm=False, world_size=16), expect)
                comm.Barrier()
                # Again with the real ranks' slots arriving in varying orders.
                for trial in range(4):
                    fill_other_slots(xs, num_tokens)
                    delays = torch.randint(0, 40000, (world,), generator=delay_gen).tolist()
                    torch.cuda.synchronize()
                    comm.Barrier()
                    torch.cuda._sleep(delays[rank])
                    same(row_ar(xs[rank], norm=False, world_size=16), expect)
                    comm.Barrier()
            # The C++ kernel resumes on the workspace after the 16-rank launches.
            x = _inputs(world, 14, 1)[rank]
            same(cpp(x, AllReduceFusionOp.RMS_NORM, own), row_ar(x))
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


@pytest.mark.skipif(
    _CAP not in ((10, 0), (10, 3), (10, 7)),
    reason="SM100 / SM103 / SM107 kernel",
)
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_latent_ar_norm(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    results = mpi_pool_executor.map(_ar_norm_rank, *zip(*[(world,)] * world))
    assert all(r is True for r in results)
