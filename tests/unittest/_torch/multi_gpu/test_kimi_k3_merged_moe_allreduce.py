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
"""Kimi K3 MoE block back-end: the all-reduce variants of the routed latent.

``KimiK3MoERuntime.forward`` runs on every rank with stand-ins for the routed experts (a
per-rank latent partial) and the TP-sharded shared expert (a per-rank partial, either reduced by
the module itself like the BF16 ``GatedMLP`` or by ``shared_down_all_reduce`` after an FP8
weight-read down projection), real MNNVL all-reduces and the real latent RMSNorm.

- Merged all-reduce (BF16 latent up-projection): must match the two-all-reduce path and an FP32
  reference, eagerly (with and without the shared-expert stream) and under CUDA graph replay.
- ``TRTLLM_K3_LATENT_AR_QUANT`` (FP8 weight-read latent up-projection on SM107): the all-reduce
  that also quantizes the normed latent to FP8 1x128 (R128c4 scales) must give the same bits as
  the all-reduce + RMSNorm followed by the up-projection's own quantization, eagerly and under
  CUDA graph replay.
"""

import importlib.util
import os
import pickle
import sys
import traceback
from types import SimpleNamespace

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 7168
LATENT = 3584
SHARED_SHARD = 384
EPS = 1e-6
# 1 and 7 run one-shot at TP2 with the default threshold, 224 two-shot.
SEQ_LENS = (1, 7, 32, 224)


class _RoutedStandIn:
    """Returns this rank's latent partial; writes it into ``moe_output`` when asked to."""

    def __init__(self, all_reduce, honor_output):
        self.all_reduce = all_reduce
        self.honor_output = honor_output
        self.partial = None
        self.routing_method = None
        self.use_dp = False
        self.backend = SimpleNamespace()
        self.wrote_into_output = False

    def __call__(self, x, router_logits, *, all_rank_num_tokens=None, moe_output=None, **kwargs):
        if moe_output is not None and self.honor_output:
            moe_output.copy_(self.partial)
            self.wrote_into_output = True
            return moe_output
        return self.partial.clone()


class _SharedStandIn(torch.nn.Module):
    """TP-sharded shared expert: BF16 (reduces itself) or FP8 down projection (does not)."""

    def __init__(self, all_reduce, fp8_down):
        super().__init__()
        self.all_reduce = all_reduce
        self.partial = None
        self.activation = None
        self.down_proj = fp8_down if fp8_down is not None else torch.nn.Identity()

    def forward(self, x, final_all_reduce_params=None):
        if not isinstance(self.down_proj, torch.nn.Identity):
            return self.down_proj(self.activation)
        out = self.partial.clone()
        if final_all_reduce_params is None or final_all_reduce_params.enable_allreduce:
            out = self.all_reduce(out)
        return out


def _rank_inputs(world, rank, num_tokens, seed):
    """Per-rank partials (and every rank's, for the reference) and the shared residual."""
    gen = torch.Generator(device="cuda")
    latents, shareds, acts = [], [], []
    for r in range(world):
        gen.manual_seed(seed * 1009 + r)
        latents.append(torch.randn((num_tokens, LATENT), generator=gen, device="cuda").bfloat16())
        shareds.append(torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda").bfloat16())
        acts.append(
            torch.randn((num_tokens, SHARED_SHARD), generator=gen, device="cuda").bfloat16()
        )
    gen.manual_seed(seed * 1009 + 999)
    residual = (torch.randn((num_tokens, HIDDEN), generator=gen, device="cuda") * 4).bfloat16()
    return latents, shareds, acts, residual


def _assert_moe_close(actual, expected, residual, rtol, atol):
    """Compare two ``residual=`` outputs, ``residual + BF16(routed + shared)``, with the tolerance
    relative to the operands of that final add, ``|residual| + |routed + shared|``.

    Two BF16 rounding steps feed the result: the MoE sum and the final add. Where the residual
    (std 4 here) nearly cancels the MoE sum, one rounding step of that sum (2**-5 at magnitude 4-8)
    lands on a small result, and a tolerance relative to the result alone rejects it. The operand
    magnitudes bound both steps, and are never below the result's, so this only admits rounding
    that a result-relative tolerance would reject because of cancellation.
    """
    a, e, r = actual.float(), expected.float(), residual.float()
    assert a.shape == e.shape == r.shape
    assert torch.isfinite(a).all() and torch.isfinite(e).all()
    bad = (a - e).abs() > atol + rtol * (r.abs() + (e - r).abs())
    if bad.any():
        i = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(
            f"{int(bad.sum())} / {bad.numel()} elements beyond atol {atol} + rtol {rtol} x "
            f"(|residual| + |moe|); first at {i}: actual {a[i].item()}, expected "
            f"{e[i].item()}, residual {r[i].item()}"
        )


def _make_runtime(mkl, routed, shared, shared_down_all_reduce, up_proj):
    """``KimiK3MoERuntime`` around the stand-ins, with the real latent RMSNorm and ``up_proj``.

    The two-all-reduce configuration: callers switch on the merged all-reduce themselves.
    """
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    runtime = mkl.KimiK3MoERuntime.__new__(mkl.KimiK3MoERuntime)
    torch.nn.Module.__init__(runtime)
    runtime.hidden_size = HIDDEN
    runtime.moe_hidden_size = LATENT
    runtime.gate = SimpleNamespace(
        compute_logits=lambda h: torch.zeros((h.shape[0], 8), device=h.device)
    )
    runtime.routed_experts = routed
    runtime.shared_experts = shared
    runtime.shared_down_all_reduce = shared_down_all_reduce
    runtime.routed_expert_down_proj = torch.nn.Identity()
    runtime.routed_expert_norm = RMSNorm(hidden_size=LATENT, eps=EPS, dtype=torch.bfloat16).cuda()
    runtime.routed_expert_norm.weight.data = (torch.rand(LATENT, device="cuda") + 0.5).bfloat16()
    runtime.routed_expert_up_proj = up_proj
    runtime._routed_projection = lambda x, projection: projection(x)
    runtime._can_fuse_router_down = lambda h: False
    runtime._reduce_routed_output = True
    runtime._defer_shared_allreduce = False
    runtime._merge_moe_allreduce = False
    runtime.shared_expert_stream = torch.cuda.Stream()
    runtime.latent_down_stream = None
    runtime.moe_main_event = torch.cuda.Event()
    runtime.moe_shared_event = torch.cuda.Event()
    runtime.moe_routed_event = runtime.moe_latent_fork_event = None
    runtime.moe_latent_ready_event = None
    return runtime


def _merged_moe_rank(world, honor_output, shared_fp8):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    try:
        from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
        from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
        ar_routed = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16
        )
        ar_shared = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16
        )
        assert ar_routed.mnnvl_allreduce is not None

        torch.manual_seed(0)
        fp8_down = None
        direct_writes = []
        if shared_fp8:
            from tensorrt_llm._torch.cute_dsl_kernels.blackwell import k3_fp8w_lowm_gemm as lowm

            gemm_out = lowm.k3_fp8w_lowm_gemm_out

            def counting_gemm_out(x, w, w_scale, out):
                direct_writes.append(out.data_ptr())
                return gemm_out(x, w, w_scale, out)

            lowm.k3_fp8w_lowm_gemm_out = counting_gemm_out
            mkl._K3_FP8W_LOWM_GEMM = True
            dense = torch.nn.Linear(SHARED_SHARD, HIDDEN, bias=False, dtype=torch.bfloat16)
            dense.weight.data = (torch.randn_like(dense.weight) * 0.05).cuda()
            fp8_down = mkl._Fp8BlockScaleWeightReadLinear.from_linear(dense)

        up_proj = torch.nn.Linear(LATENT, HIDDEN, bias=False, dtype=torch.bfloat16).cuda()
        up_proj.weight.data.normal_(0, 0.02)
        runtime = _make_runtime(
            mkl,
            _RoutedStandIn(ar_routed, honor_output),
            _SharedStandIn(ar_shared, fp8_down),
            ar_shared if shared_fp8 else None,
            up_proj,
        )
        mkl._K3_MERGED_MOE_AR_MAX_TOKENS = 0
        norm_w = runtime.routed_expert_norm.weight.float()

        def set_inputs(latents, shareds, acts):
            runtime.routed_experts.partial = latents[rank]
            runtime.shared_experts.partial = shareds[rank]
            runtime.shared_experts.activation = acts[rank]

        def run(merge, multi_stream, hidden, residual):
            runtime._merge_moe_allreduce = merge
            with with_multi_stream(multi_stream), torch.inference_mode():
                return runtime(hidden, residual=residual)

        def reference(latents, shareds, acts, residual):
            lat = sum(x.float() for x in latents).bfloat16().float()
            lat = (lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + EPS) * norm_w).bfloat16()
            routed = runtime.routed_expert_up_proj(lat).float()
            if shared_fp8:
                shareds = [fp8_down(a) for a in acts]
            shared = sum(x.float() for x in shareds).bfloat16().float()
            return residual.float() + (routed + shared).bfloat16().float()

        for num_tokens in SEQ_LENS:
            latents, shareds, acts, residual = _rank_inputs(world, rank, num_tokens, num_tokens)
            set_inputs(latents, shareds, acts)
            hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            ref = reference(latents, shareds, acts, residual)
            old = run(False, False, hidden, residual)
            _assert_moe_close(old, ref, residual, rtol=0.02, atol=0.05)
            for multi_stream in (False, True):
                runtime.routed_experts.wrote_into_output = False
                direct_writes.clear()
                new = run(True, multi_stream, hidden, residual)
                assert runtime.routed_experts.wrote_into_output == honor_output
                if shared_fp8 and lowm.can_run(acts[rank], fp8_down.weight, fp8_down.weight_scale):
                    # The low-M down projection wrote straight into the merged buffer.
                    assert len(direct_writes) == 1
                _assert_moe_close(new, ref, residual, rtol=0.02, atol=0.05)
                # Same sums and the same RMSNorm math: at most one BF16 rounding step apart.
                _assert_moe_close(new, old, residual, rtol=2**-7, atol=2**-6)

            # CUDA graph: capture the merged path once, replay it on new partials.
            static_lat = latents[rank].clone()
            static_sh = shareds[rank].clone()
            static_act = acts[rank].clone()
            static_res = residual.clone()
            runtime.routed_experts.partial = static_lat
            runtime.shared_experts.partial = static_sh
            runtime.shared_experts.activation = static_act
            for _ in range(2):
                run(True, True, hidden, static_res)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            runtime._merge_moe_allreduce = True
            with with_multi_stream(True), torch.inference_mode(), torch.cuda.graph(graph):
                graph_out = runtime(hidden, residual=static_res)
            for step in range(3):
                latents, shareds, acts, residual = _rank_inputs(
                    world, rank, num_tokens, 100 + 7 * step + num_tokens
                )
                static_lat.copy_(latents[rank])
                static_sh.copy_(shareds[rank])
                static_act.copy_(acts[rank])
                static_res.copy_(residual)
                graph.replay()
                torch.cuda.synchronize()
                set_inputs(latents, shareds, acts)
                old = run(False, False, hidden, residual)
                _assert_moe_close(graph_out, old, residual, rtol=2**-7, atol=2**-6)
                _assert_moe_close(
                    graph_out,
                    reference(latents, shareds, acts, residual),
                    residual,
                    rtol=0.02,
                    atol=0.05,
                )
            del graph

        # Above TRTLLM_K3_MERGED_MOE_AR_MAX_TOKENS the runtime keeps the two all-reduces.
        num_tokens = SEQ_LENS[-1]
        mkl._K3_MERGED_MOE_AR_MAX_TOKENS = num_tokens - 1
        latents, shareds, acts, residual = _rank_inputs(world, rank, num_tokens, 5)
        set_inputs(latents, shareds, acts)
        hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
        runtime.routed_experts.wrote_into_output = False
        limited = run(True, True, hidden, residual)
        assert not runtime.routed_experts.wrote_into_output
        torch.testing.assert_close(limited, run(False, True, hidden, residual), rtol=0, atol=0)
        mkl._K3_MERGED_MOE_AR_MAX_TOKENS = 0
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


def _latent_ar_quant_rank(world):
    """``TRTLLM_K3_LATENT_AR_QUANT`` on vs off around an FP8 weight-read latent up-projection."""
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    quant_flag = mkl._K3_LATENT_AR_QUANT
    try:
        from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

        mapping = Mapping(world_size=world, rank=rank, gpus_per_node=world, tp_size=world)
        ar_routed = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16
        )
        ar_shared = AllReduce(
            mapping=mapping, strategy=AllReduceStrategy.AUTO, dtype=torch.bfloat16
        )
        assert ar_routed.mnnvl_allreduce is not None

        torch.manual_seed(0)
        dense_up = torch.nn.Linear(LATENT, HIDDEN, bias=False, dtype=torch.bfloat16)
        dense_up.weight.data = (torch.randn_like(dense_up.weight) * 0.02).cuda()
        up_proj = mkl._Fp8BlockScaleWeightReadLinear.from_linear(dense_up)
        assert up_proj.supports_prequantized_input
        prequantized_rows = []
        forward_prequantized = up_proj.forward_prequantized

        def counting_forward_prequantized(activation, activation_scale):
            prequantized_rows.append(activation.shape[0])
            return forward_prequantized(activation, activation_scale)

        up_proj.forward_prequantized = counting_forward_prequantized
        runtime = _make_runtime(
            mkl,
            _RoutedStandIn(ar_routed, honor_output=False),
            _SharedStandIn(ar_shared, None),
            None,
            up_proj,
        )

        def set_inputs(latent, shared):
            runtime.routed_experts.partial = latent
            runtime.shared_experts.partial = shared

        def run(quantize, hidden, residual):
            mkl._K3_LATENT_AR_QUANT = quantize
            with with_multi_stream(True), torch.inference_mode():
                return runtime(hidden, residual=residual)

        for num_tokens in SEQ_LENS:
            latents, shareds, _, residual = _rank_inputs(world, rank, num_tokens, num_tokens)
            set_inputs(latents[rank], shareds[rank])
            hidden = torch.zeros((num_tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            prequantized_rows.clear()
            unfused = run(False, hidden, residual)
            assert prequantized_rows == []
            fused = run(True, hidden, residual)
            assert prequantized_rows == [num_tokens]
            # Same reduction, RMSNorm, FP8 bytes, scales and GEMM: the same bits.
            assert torch.equal(fused, unfused)

            # CUDA graph: capture the fused path once, replay it on new partials.
            static_lat = latents[rank].clone()
            static_sh = shareds[rank].clone()
            static_res = residual.clone()
            set_inputs(static_lat, static_sh)
            for _ in range(2):
                run(True, hidden, static_res)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            mkl._K3_LATENT_AR_QUANT = True
            captured_rows = len(prequantized_rows)
            with with_multi_stream(True), torch.inference_mode(), torch.cuda.graph(graph):
                graph_out = runtime(hidden, residual=static_res)
            # The capture itself took the fused route (the spy runs while the graph is recorded), so the replays
            # below exercise the fused all-reduce and not a silent fallback to the unfused chain.
            assert prequantized_rows[captured_rows:] == [num_tokens]
            for step in range(3):
                latents, shareds, _, residual = _rank_inputs(
                    world, rank, num_tokens, 100 + 7 * step + num_tokens
                )
                static_lat.copy_(latents[rank])
                static_sh.copy_(shareds[rank])
                static_res.copy_(residual)
                graph.replay()
                torch.cuda.synchronize()
                set_inputs(latents[rank], shareds[rank])
                assert torch.equal(graph_out, run(False, hidden, residual))
                set_inputs(static_lat, static_sh)
            del graph
        return True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        mkl._K3_LATENT_AR_QUANT = quant_flag
        os.environ.pop("TLLM_TEST_MNNVL", None)
        torch.cuda.synchronize()


def _rubin_prequantized_gemm_available():
    """Two GPUs with the SM107 CuTe DSL MXFP8 GEMM, which reads the prequantized latent."""
    if torch.cuda.device_count() < 2:
        return False
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _run(mpi_pool_executor, honor_output, shared_fp8):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(
        _merged_moe_rank, *zip(*[(world, honor_output, shared_fp8)] * world)
    )
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("honor_output", [True, False], ids=["direct", "copied"])
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_merged_moe_allreduce_bf16_shared(mpi_pool_executor, honor_output):
    _run(mpi_pool_executor, honor_output, shared_fp8=False)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the FP8 low-M GEMM runs on SM100/SM103",
)
@pytest.mark.skipif(
    importlib.util.find_spec("tensorrt_llm._torch.cute_dsl_kernels.blackwell.k3_fp8w_lowm_gemm")
    is None,
    reason="the FP8 low-M GEMM is not on this branch",
)
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_merged_moe_allreduce_fp8_shared_tp4(mpi_pool_executor):
    _run(mpi_pool_executor, honor_output=True, shared_fp8=True)


@pytest.mark.skipif(
    not _rubin_prequantized_gemm_available(),
    reason="needs 2 GPUs and the Rubin (SM107) CuTe DSL MXFP8 GEMM",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_latent_ar_quant_matches_unfused(mpi_pool_executor):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_latent_ar_quant_rank, *zip(*[(world,)] * world))
    for r in results:
        assert r is True
