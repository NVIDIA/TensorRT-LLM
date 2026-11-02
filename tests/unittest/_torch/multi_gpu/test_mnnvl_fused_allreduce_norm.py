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
"""The CuTe DSL all-reduce + residual + RMSNorm kernels of MNNVLAllReduce against an FP32 reference over
all ranks' inputs: the slot (one-shot) and two-shot launches, the shared-expert gate fold (with or
without the in-kernel MoE finalize), the FP8 output, and the FP8 GEMM + all-reduce + norm launch."""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams, MoEAllReduceParams
from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import MnnvlAllReduceNorm
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
        reason="the kernels are enabled on SM100/SM103",
    ),
]

H = 4096
EPS = 1e-6
# Kernel launch: slot below 64 rows, two-shot from 64. ``ar(...)`` takes the CuTe kernels only up to the 1 MiB
# one-shot message (rows * H * tp * 2 bytes), the library op above, so the two-shot rows are also checked through
# ``ar.fused_norm`` directly. Consecutive calls also cycle the Lamport buffers.
ROWS = (1, 7, 32, 63, 64, 96, 200)


def _rand(shape, seed, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(dtype)


def _ref_norm(total, residual, norm_weight):
    """(norm, residual_out) in FP32 of the all-reduced ``total`` [M, H] FP32."""
    res = total + residual.float()
    var = res.pow(2).mean(-1, keepdim=True)
    return res * torch.rsqrt(var + EPS) * norm_weight.float(), res


def _check(got, ref, tag, rtol=2e-2):
    torch.testing.assert_close(
        got.float(), ref, rtol=rtol, atol=rtol * ref.abs().max().item(), msg=lambda m: f"{tag}: {m}"
    )


def _mnnvl(tp):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    os.environ["TLLM_TEST_MNNVL"] = "1"
    MPI.COMM_WORLD.barrier()
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    ar = MNNVLAllReduce(mapping, torch.bfloat16, max_num_tokens=max(ROWS), hidden_size=H)
    assert MnnvlAllReduceNorm.is_supported(mapping, torch.bfloat16, H)
    assert ar.fused_norm is not None, "CuTe DSL kernels not built"
    return ar, rank


def _worker(fn, tp):
    try:
        with torch.inference_mode():
            fn(tp)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        os.environ.pop("TLLM_TEST_MNNVL", None)
    return True


def _norm_cases(tp):
    ar, rank = _mnnvl(tp)
    norm_weight = _rand((H,), 1, 0.1) + 1
    scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")
    for m in ROWS:
        assert ar.fused_norm.supports_rows(m), m
        xs = [_rand((m, H), 100 * m + r) for r in range(tp)]
        residual = _rand((m, H), 7 * m)
        ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
        params = AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
        )
        norm, res = ar(xs[rank], all_reduce_params=params)
        _check(norm, ref_norm, ("norm", m))
        _check(res, ref_res, ("residual", m))
        norm, res = ar.fused_norm(residual, norm_weight, EPS, x=xs[rank])
        _check(norm, ref_norm, ("kernel norm", m))
        _check(res, ref_res, ("kernel residual", m))
        norm, fp8, res = ar.fused_norm(
            residual, norm_weight, EPS, x=xs[rank], scale=scale, norm_out=True
        )
        _check(norm, ref_norm, ("norm with fp8", m))
        # E4M3 holds 3 mantissa bits: half an ulp is 6.25 % relative.
        _check(fp8.float() * scale, ref_norm.clamp(-448 * 0.05, 448 * 0.05), ("fp8", m), rtol=7e-2)
    # More rows than two CTAs per SM: the two-shot kernel takes several tokens per CTA.
    m = 2 * torch.cuda.get_device_properties(rank).multi_processor_count + 8
    assert ar.fused_norm.supports_rows(m), m
    xs = [_rand((m, H), 100 * m + r) for r in range(tp)]
    residual = _rand((m, H), 7 * m)
    ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
    norm, res = ar.fused_norm(residual, norm_weight, EPS, x=xs[rank])
    _check(norm, ref_norm, ("kernel norm", m))
    _check(res, ref_res, ("kernel residual", m))


def _moe_fold_cases(tp, top_k=10):
    ar, rank = _mnnvl(tp)
    norm_weight = _rand((H,), 2, 0.1) + 1
    for m in ROWS:
        assert ar.supports_moe_fold(m, H), m
        routed, finalized, args = [], [], None
        for r in range(tp):
            s = 1000 * m + 10 * r
            # FC2 rows of the permuted (token, k) pairs plus padding rows, with a stride past H.
            fc2 = _rand((m * top_k + 16, H + 64), s)[:, :H]
            g = torch.Generator(device="cuda").manual_seed(s + 1)
            perm = torch.randperm(m * top_k + 16, generator=g, device="cuda")[: m * top_k]
            perm = perm.view(m, top_k).to(torch.int32)
            topw = torch.softmax(_rand((m, top_k), s + 2, dtype=torch.float32), -1).to(
                torch.bfloat16
            )
            shared = _rand((m, H), s + 3)
            gate = _rand((m, 1), s + 4)
            fin = (fc2.float()[perm.long()] * topw.float().unsqueeze(-1)).sum(1)
            routed.append(fin + torch.sigmoid(gate.float()) * shared.float())
            fin_bf16 = fin.to(torch.bfloat16)
            finalized.append(fin_bf16.float() + torch.sigmoid(gate.float()) * shared.float())
            if r == rank:
                args = (fc2, topw, perm, shared, gate, fin_bf16)
        residual = _rand((m, H), 9 * m)
        ref_norm, ref_res = _ref_norm(sum(routed), residual, norm_weight)
        fc2, topw, perm, shared, gate, fin_bf16 = args
        params = MoEAllReduceParams(
            expanded_idx_to_permuted_idx=perm,
            expert_scale_factor=topw,
            shared_expert_output=shared,
            shared_expert_gate=gate,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        )
        norm, res = ar(fc2, all_reduce_params=params)
        _check(norm, ref_norm, ("moe fold norm", m))
        _check(res, ref_res, ("moe fold residual", m))
        # A finalized routed output: only the shared-expert combine is folded.
        ref_norm, ref_res = _ref_norm(sum(finalized), residual, norm_weight)
        params = MoEAllReduceParams(
            shared_expert_output=shared,
            shared_expert_gate=gate,
            residual=residual,
            norm_weight=norm_weight,
            eps=EPS,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        )
        norm, res = ar(fin_bf16, all_reduce_params=params)
        _check(norm, ref_norm, ("finalized fold norm", m))
        _check(res, ref_res, ("finalized fold residual", m))


def _gemm_norm_cases(tp, k=1024):
    from tensorrt_llm._torch.modules.low_m_mma_gemm import default_tactic

    ar, rank = _mnnvl(tp)
    assert ar.gemm_norm is not None, "GEMM + all-reduce + norm kernel not built"
    norm_weight = _rand((H,), 3, 0.1) + 1
    weight_scale = torch.tensor([0.02], dtype=torch.float32, device="cuda")
    input_scale = torch.tensor([4.0 / 448], dtype=torch.float32, device="cuda")
    for m in (1, 5, 8, 16, 32):
        outs, args = [], None
        for r in range(tp):
            x = _rand((m, k), 77 * m + r)
            w = _rand((H, k), 5000 + r, 4.0).to(torch.float8_e4m3fn)
            xq = (x.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            outs.append((xq.float() @ w.float().t()) * input_scale * weight_scale)
            if r == rank:
                args = (x, w)
        residual = _rand((m, H), 11 * m)
        ref_norm, ref_res = _ref_norm(sum(outs), residual, norm_weight)
        x, w = args
        # fp8_payload only changes two-shot calls; this one-launch kernel exchanges BF16 either way.
        for fp8_payload in (False, True):
            params = AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm_weight,
                eps=EPS,
                fp8_payload=fp8_payload,
            )
            out = ar.gemm_allreduce_norm(
                x, w, weight_scale, input_scale, params, default_tactic(m, H, k)[:3]
            )
            assert out is not None, ("GEMM + all-reduce + norm declined", m, fp8_payload)
            norm, res = out
            _check(norm, ref_norm, ("gemm norm", m, fp8_payload))
            _check(res, ref_res, ("gemm residual", m, fp8_payload))


def _meta_init_cases(tp):
    """Kernels built under MetaInitMode, as the model loader builds the model, hold no meta tensor and run."""
    from tensorrt_llm._torch.distributed.mnnvl_fused_allreduce import (
        GemmAllReduceNorm,
        _LamportTable,
    )
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode

    ar, rank = _mnnvl(tp)
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)["comm"]
    with MetaInitMode():
        table = _LamportTable(mapping, torch.bfloat16)
        norm_kernel = MnnvlAllReduceNorm(mapping, comm, H, table)
        gemm_kernel = GemmAllReduceNorm(mapping, H, table)
    for kernel in (norm_kernel, gemm_kernel):
        meta = [k for k, v in vars(kernel).items() if isinstance(v, torch.Tensor) and v.is_meta]
        assert not meta, (type(kernel).__name__, meta)
    m = 7
    norm_weight = _rand((H,), 3, 0.1) + 1
    xs = [_rand((m, H), 300 + r) for r in range(tp)]
    residual = _rand((m, H), 301)
    ref_norm, ref_res = _ref_norm(sum(x.float() for x in xs), residual, norm_weight)
    norm, res = norm_kernel(residual, norm_weight, EPS, x=xs[rank])
    _check(norm, ref_norm, ("meta-init norm", m))
    _check(res, ref_res, ("meta-init residual", m))


def _opt_in_cases(tp):
    """AllReduce builds the kernels only where the model declares them (extra_attrs
    ``allreduce_mnnvl_fused_norm``); otherwise its MNNVL path keeps the library op."""
    from tensorrt_llm._torch.distributed import AllReduce
    from tensorrt_llm._torch.utils import model_extra_attrs
    from tensorrt_llm.functional import AllReduceStrategy

    _, rank = _mnnvl(tp)
    mapping = Mapping(world_size=tp, tp_size=tp, rank=rank)
    shape = {"allreduce_max_num_tokens": max(ROWS), "allreduce_hidden_size": H}
    for declared in (False, True):
        with model_extra_attrs({**shape, "allreduce_mnnvl_fused_norm": declared}):
            ar = AllReduce(mapping, strategy=AllReduceStrategy.MNNVL, dtype=torch.bfloat16)
        assert ar.mnnvl_allreduce is not None
        assert (ar.mnnvl_allreduce.fused_norm is not None) == declared


CASES = {
    "norm": _norm_cases,
    "moe_fold": _moe_fold_cases,
    "gemm_norm": _gemm_norm_cases,
    "meta_init": _meta_init_cases,
    "opt_in": _opt_in_cases,
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("mpi_pool_executor", [2, 4], indirect=True)
def test_mnnvl_fused_allreduce_norm(case, mpi_pool_executor):
    tp = mpi_pool_executor.num_workers
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    results = mpi_pool_executor.map(_worker, [CASES[case]] * tp, [tp] * tp)
    assert all(results)
