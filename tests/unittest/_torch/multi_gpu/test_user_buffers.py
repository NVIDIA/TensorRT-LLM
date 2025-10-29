import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
import torch.nn as nn
from mpi4py import MPI
from utils.util import skip_pre_blackwell_unittest

import tensorrt_llm
import tensorrt_llm.bindings.internal.userbuffers as ub
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, AllReduceStrategy,
                                             userbuffers_allreduce_finalize)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)


def init_userbuffers_allocator(tp_size, rank, max_ub_size):
    ub.initialize_userbuffers_manager(tp_size, 1, 1, rank,
                                      torch.cuda.device_count(), max_ub_size,
                                      False)


def create_userbuffers_tensor(shape, dtype):
    # WAR pickle error
    def func(shape, dtype):
        return torch.ops.trtllm.create_userbuffers_tensor(shape, dtype)

    return func(shape, dtype)


def quant(input, scale):
    finfo = torch.finfo(torch.float8_e4m3fn)
    inv_scale = scale.reciprocal()
    qinput = (input.float() * inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qinput.to(torch.float8_e4m3fn)


def dequant(input, scale, dtype):
    dqinput = input.to(torch.float32) * scale
    return dqinput.to(dtype)


# This rms_norm aligns with ub impl that calculate gamma * hidden in high
# precision
def rms_norm(input, gamma, eps):
    variance = input.pow(2).mean(-1, keepdim=True)
    hidden_states = input * torch.rsqrt(variance + eps)
    return gamma.to(torch.float32) * hidden_states


def run_single_rank(tensor_parallel_size, a, b, c):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        support = ub.ub_supported()
        if not support:
            return True

        a = a.cuda()
        b = b.cuda()
        c = c.cuda()
        ref = torch.matmul(a, b) + c

        ub_size = c.nelement() * c.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)

        def run_ub_impl(a, b, c):
            ub0_tensor = create_userbuffers_tensor(c.size(), a.dtype)
            ub1_tensor = create_userbuffers_tensor(c.size(), a.dtype)
            assert ub0_tensor is not None
            assert ub1_tensor is not None
            internal = torch.matmul(a, b, out=ub0_tensor)
            res = torch.add(internal, c, out=ub1_tensor)
            return res

        res = run_ub_impl(a, b, c)
        res = run_ub_impl(a, b, c)

        torch.cuda.synchronize()
        torch.testing.assert_close(ref, res)

    except Exception:
        traceback.print_exc()
        raise
    return True


def run_single_rank_ar_rms_norm(tensor_parallel_size, a, b, c, gamma):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        support = ub.ub_supported()
        if not support:
            return True
        eps = 1e-6

        a_partial = torch.chunk(a, tensor_parallel_size, 1)
        b_partial = torch.chunk(b, tensor_parallel_size, 0)

        a_local = a_partial[rank].cuda()
        b_local = b_partial[rank].cuda()
        c = c.cuda()
        gamma = gamma.cuda()

        ub_size = c.nelement() * c.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)

        ub0_tensor = create_userbuffers_tensor(c.size(), a.dtype)
        hidden = torch.matmul(a_local, b_local, out=ub0_tensor)
        mapping = Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=rank,
        )
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.UB)
        ar_params = AllReduceParams(
            strategy=AllReduceStrategy.UB,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=c,
            norm_weight=gamma,
            eps=eps)
        res_ub, residual = ar.forward(hidden, all_reduce_params=ar_params)
        res = res_ub.clone()
        residual = userbuffers_allreduce_finalize(residual, False)

        torch.cuda.synchronize()
        if rank == 0:
            # Fully simulate matmul + allreduce behavior
            ax = [a_partial[i].cuda() for i in range(0, tensor_parallel_size)]
            bx = [b_partial[i].cuda() for i in range(0, tensor_parallel_size)]
            h1 = [
                torch.matmul(ax[i], bx[i])
                for i in range(0, tensor_parallel_size)
            ]
            sum = h1[0]
            for i in range(1, tensor_parallel_size):
                sum = sum + h1[i]
            ref_residual = sum + c
            ref = rms_norm(ref_residual.to(torch.float32), gamma,
                           eps).to(res.dtype)
            torch.testing.assert_close(ref, res, atol=5e-1, rtol=1e-2)
            torch.testing.assert_close(ref_residual,
                                       residual,
                                       atol=5e-1,
                                       rtol=1e-2)

    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("mnk", [(256, 512, 256), (32, 16, 64)],
                         ids=lambda x: f"m{x[0]}_n{x[1]}_k{x[2]}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_user_buffers_ar_rms_norm(mnk, mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = 2
    dtype = torch.float16
    m = mnk[0]
    n = mnk[1]
    k = mnk[2]
    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.randn((m, n), dtype=dtype)
    gamma = torch.randn((n), dtype=dtype)

    results = mpi_pool_executor.map(
        run_single_rank_ar_rms_norm,
        *zip(*[(tensor_parallel_size, a, b, c, gamma)] * tensor_parallel_size))
    for r in results:
        assert r is True


def run_single_rank_ar_rms_norm_fp8(tensor_parallel_size, a, b, c, gamma,
                                    scale):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        support = ub.ub_supported()
        if not support:
            return True
        eps = 1e-6

        a_partial = torch.chunk(a, tensor_parallel_size, 1)
        b_partial = torch.chunk(b, tensor_parallel_size, 0)

        a_local = a_partial[rank].cuda()
        b_local = b_partial[rank].cuda()
        c = c.cuda()
        gamma = gamma.cuda()
        scale = scale.cuda()

        ub_size = c.nelement() * c.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)

        ub0_tensor = create_userbuffers_tensor(c.size(), a.dtype)
        hidden = torch.matmul(a_local, b_local, out=ub0_tensor)
        mapping = Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=rank,
        )
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.UB)
        ar_params = AllReduceParams(
            strategy=AllReduceStrategy.UB,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8,
            residual=c,
            norm_weight=gamma,
            scale=scale,
            eps=eps)
        res, residual = ar.forward(hidden, all_reduce_params=ar_params)
        res = dequant(res, scale, torch.float16)
        residual = userbuffers_allreduce_finalize(residual, False)

        torch.cuda.synchronize()
        if rank == 0:
            # Fully simulate matmul + allreduce behavior
            ax = [a_partial[i].cuda() for i in range(0, tensor_parallel_size)]
            bx = [b_partial[i].cuda() for i in range(0, tensor_parallel_size)]
            h1 = [
                torch.matmul(ax[i], bx[i])
                for i in range(0, tensor_parallel_size)
            ]
            sum = h1[0]
            for i in range(1, tensor_parallel_size):
                sum = sum + h1[i]
            ref_residual = sum + c
            ref = dequant(
                quant(rms_norm(ref_residual.to(torch.float32), gamma, eps),
                      scale), scale, torch.float16)
            torch.testing.assert_close(ref, res, atol=5e-1, rtol=1e-2)
            torch.testing.assert_close(ref_residual,
                                       residual,
                                       atol=5e-1,
                                       rtol=1e-2)

    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_user_buffers_basic(mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = 2
    dtype = torch.float32
    m = 16
    n = 32
    k = 16
    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.randn((m, n), dtype=dtype)
    results = mpi_pool_executor.map(
        run_single_rank,
        *zip(*[(tensor_parallel_size, a, b, c)] * tensor_parallel_size))
    for r in results:
        assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("mnk", [(256, 512, 256), (32, 16, 64)],
                         ids=lambda x: f"m{x[0]}_n{x[1]}_k{x[2]}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_user_buffers_ar_rms_norm_fp8(mnk, mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = 2
    dtype = torch.float16
    m = mnk[0]
    n = mnk[1]
    k = mnk[2]
    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.randn((m, n), dtype=dtype)
    gamma = torch.randn((n), dtype=dtype)
    scale = torch.randn((1), dtype=torch.float32)

    results = mpi_pool_executor.map(
        run_single_rank_ar_rms_norm_fp8,
        *zip(*[(tensor_parallel_size, a, b, c, gamma, scale)] *
             tensor_parallel_size))
    for r in results:
        assert r is True


class UBTestModel(nn.Module):

    def __init__(self, tp_size, rank, hidden_size, dtype, eps, l0_weight,
                 l0_input_scale, l0_weight_scale, l1_weight, l1_input_scale,
                 l1_weight_scale, l2_weight, l2_input_scale, l2_weight_scale,
                 l3_weight, l3_input_scale, l3_weight_scale, l4_weight,
                 l4_input_scale, l4_weight_scale, norm0_gamma, norm1_gamma,
                 norm2_gamma):
        super().__init__()
        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.FP8
        quant_config.layer_quant_mode
        self.rank = rank
        self.tp_size = tp_size
        mapping = Mapping(
            world_size=tp_size,
            tp_size=tp_size,
            rank=rank,
        )
        self.l0 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=mapping,
                         tensor_parallel_mode=TensorParallelMode.ROW,
                         quant_config=quant_config).cuda()
        self.l1 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=mapping,
                         tensor_parallel_mode=TensorParallelMode.COLUMN,
                         quant_config=quant_config).cuda()
        self.l2 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=mapping,
                         tensor_parallel_mode=TensorParallelMode.ROW,
                         quant_config=quant_config).cuda()
        self.l3 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=mapping,
                         tensor_parallel_mode=TensorParallelMode.COLUMN,
                         quant_config=quant_config).cuda()
        self.l4 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=mapping,
                         tensor_parallel_mode=TensorParallelMode.ROW,
                         quant_config=quant_config).cuda()
        self.norm0 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()
        self.norm1 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()
        self.norm2 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()

        self.l0.load_weights([
            dict(weight=l0_weight,
                 input_scale=l0_input_scale,
                 weight_scale=l0_weight_scale)
        ])
        self.l1.load_weights([
            dict(weight=l1_weight,
                 input_scale=l1_input_scale,
                 weight_scale=l1_weight_scale)
        ])
        self.l2.load_weights([
            dict(weight=l2_weight,
                 input_scale=l2_input_scale,
                 weight_scale=l2_weight_scale)
        ])
        self.l3.load_weights([
            dict(weight=l3_weight,
                 input_scale=l3_input_scale,
                 weight_scale=l3_weight_scale)
        ])
        self.l4.load_weights([
            dict(weight=l4_weight,
                 input_scale=l4_input_scale,
                 weight_scale=l4_weight_scale)
        ])
        self.norm0.weight.data.copy_(norm0_gamma)
        self.norm1.weight.data.copy_(norm1_gamma)
        self.norm2.weight.data.copy_(norm2_gamma)
        torch.cuda.current_stream().synchronize()

    def forward(self, input):
        local = torch.chunk(input, self.tp_size,
                            1)[self.rank].contiguous()  # [token, hidden/tp]
        hidden0 = self.l0(
            local
        )  # [token, hidden/tp] * [hidden/tp, hidden] -> [token, hidden], row parallel
        hidden1 = hidden0 + input  # [token, hidden]
        norm0 = self.norm0(hidden1)  # [token, hidden]
        hidden2 = self.l1(
            norm0
        )  # [token, hidden] * [hidden, hidden/tp] -> [token, hidden/tp], col parallel, no gather
        hidden3 = self.l2(
            hidden2
        )  # # [token, hidden/tp] * [hidden/tp, hidden] -> [token, hidden], row parallel
        hidden4 = hidden3 + hidden1
        norm1 = self.norm1(hidden4)
        hidden5 = self.l3(
            norm1
        )  # [token, hidden] * [hidden, hidden/tp] -> [token, hidden/tp], col parallel, no gather
        hidden6 = self.l4(
            hidden5
        )  # # [token, hidden/tp] * [hidden/tp, hidden] -> [token, hidden], row parallel
        hidden7 = hidden6 + hidden4
        res = self.norm2(hidden7)
        return res


def run_single_rank_ub_pass(
        tensor_parallel_size, input, l0_weight, l0_input_scale, l0_weight_scale,
        l1_weight, l1_input_scale, l1_weight_scale, l2_weight, l2_input_scale,
        l2_weight_scale, l3_weight, l3_input_scale, l3_weight_scale, l4_weight,
        l4_input_scale, l4_weight_scale, norm0_gamma, norm1_gamma, norm2_gamma):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        support = ub.ub_supported()
        if not support:
            return True

        eps = 1e-6
        hidden_size = input.size(-1)
        dtype = input.dtype
        input = input.cuda()
        ub_size = input.nelement() * input.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)
        model = UBTestModel(
            tensor_parallel_size, rank, hidden_size, dtype, eps,
            quant(l0_weight, l0_weight_scale), l0_input_scale, l0_weight_scale,
            quant(l1_weight, l1_weight_scale), l1_input_scale, l1_weight_scale,
            quant(l2_weight, l2_weight_scale), l2_input_scale, l2_weight_scale,
            quant(l3_weight, l3_weight_scale), l3_input_scale, l3_weight_scale,
            quant(l4_weight, l4_weight_scale), l4_input_scale, l4_weight_scale,
            norm0_gamma, norm1_gamma, norm2_gamma)
        backend = Backend(enable_inductor=False, enable_userbuffers=True)
        model_opt = torch.compile(model, backend=backend, fullgraph=True)
        with torch.inference_mode():
            output_fused = model_opt(input)
        # 3 AR_NORM fusion happens first
        # 2 AR_NORM fused with Quant
        # 3 AR_NORM replacement
        # 3 Scaled MM Prologue
        # 2 UB Finalize Removal
        assert backend.match_count == [3, 0, 2, 0, 3, 0, 3, 0, 2, 0]
        torch.cuda.synchronize()

        if rank == 0:

            def ref_scaled_mm_ar(xs, w, in_s, w_s):
                ws = torch.chunk(w, tensor_parallel_size, 1)
                ys = [
                    torch.ops.trtllm.cublas_scaled_mm(
                        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                            xs[i].contiguous(), in_s)[0],
                        ws[i].contiguous().t(),
                        in_s,
                        w_s,
                        bias=None,
                        out_dtype=dtype) for i in range(0, len(xs))
                ]
                y = ys[0]
                for i in range(1, tensor_parallel_size):
                    y = y + ys[i]
                return y

            def ref_scaled_mm_col(x, w, in_s, w_s):
                ws = torch.chunk(w, tensor_parallel_size, 0)
                ys = [
                    torch.ops.trtllm.cublas_scaled_mm(x,
                                                      ws[i].contiguous().t(),
                                                      in_s,
                                                      w_s,
                                                      bias=None,
                                                      out_dtype=dtype)
                    for i in range(0, len(ws))
                ]
                return ys

            mm0 = ref_scaled_mm_ar(
                torch.chunk(input, tensor_parallel_size, 1),
                quant(l0_weight.cuda(), l0_weight_scale.cuda()),
                l0_input_scale.cuda(), l0_weight_scale.cuda()) + input
            norm1 = quant(
                rms_norm(mm0.to(torch.float32), norm0_gamma.cuda(), eps),
                l1_input_scale.cuda())
            mm1 = ref_scaled_mm_col(
                norm1, quant(l1_weight.cuda(), l1_weight_scale.cuda()),
                l1_input_scale.cuda(), l1_weight_scale.cuda())
            mm2 = ref_scaled_mm_ar(
                mm1, quant(l2_weight.cuda(), l2_weight_scale.cuda()),
                l2_input_scale.cuda(), l2_weight_scale.cuda()) + mm0
            norm2 = quant(
                rms_norm(mm2.to(torch.float32), norm1_gamma.cuda(), eps),
                l3_input_scale.cuda())
            mm3 = ref_scaled_mm_col(
                norm2, quant(l3_weight.cuda(), l3_weight_scale.cuda()),
                l3_input_scale.cuda(), l3_weight_scale.cuda())
            mm4 = ref_scaled_mm_ar(
                mm3, quant(l4_weight.cuda(), l4_weight_scale.cuda()),
                l4_input_scale.cuda(), l4_weight_scale.cuda()) + mm2
            norm3 = rms_norm(mm4.to(torch.float32), norm2_gamma.cuda(),
                             eps).to(dtype)
            torch.testing.assert_close(output_fused,
                                       norm3,
                                       atol=5e-1,
                                       rtol=1e-2)
    except Exception:
        traceback.print_exc()

        return False
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden", [512, 32], ids=lambda x: f"_hidden{x}")
@pytest.mark.parametrize("tokens", [256, 16], ids=lambda x: f"_tokens{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=lambda x: "fp16" if x == torch.float16 else "bf16")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_user_buffers_pass(hidden, tokens, dtype, mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = 2

    input = torch.randn((tokens, hidden), dtype=dtype)
    l0_weight = torch.randn((hidden, hidden), dtype=dtype)
    l1_weight = torch.randn((hidden, hidden), dtype=dtype)
    l2_weight = torch.randn((hidden, hidden), dtype=dtype)
    l3_weight = torch.randn((hidden, hidden), dtype=dtype)
    l4_weight = torch.randn((hidden, hidden), dtype=dtype)
    norm0_gamma = torch.randn((hidden, ), dtype=dtype)
    norm1_gamma = torch.randn((hidden, ), dtype=dtype)
    norm2_gamma = torch.randn((hidden, ), dtype=dtype)
    l0_input_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l0_weight_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l1_input_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l1_weight_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l2_input_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l2_weight_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l3_input_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l3_weight_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l4_input_scale = torch.full((1, ), 0.1, dtype=torch.float32)
    l4_weight_scale = torch.full((1, ), 0.1, dtype=torch.float32)

    results = mpi_pool_executor.map(
        run_single_rank_ub_pass,
        *zip(*[(tensor_parallel_size, input, l0_weight, l0_input_scale,
                l0_weight_scale, l1_weight, l1_input_scale, l1_weight_scale,
                l2_weight, l2_input_scale, l2_weight_scale, l3_weight,
                l3_input_scale, l3_weight_scale, l4_weight, l4_input_scale,
                l4_weight_scale, norm0_gamma, norm1_gamma, norm2_gamma)] *
             tensor_parallel_size))
    for r in results:
        assert r is True


def run_single_rank_ar_rms_norm_fp4(tensor_parallel_size, a, b, c, gamma):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        support = ub.ub_supported()
        if not support:
            return True
        eps = 1e-5

        ref_residual = a @ b + c
        ref_internal = rms_norm(ref_residual.to(torch.float32), gamma,
                                eps).to(a.dtype)
        internal_global_sf = (448 * 6) / ref_internal.abs().max().float()

        a_partial = torch.chunk(a, tensor_parallel_size, 1)
        b_partial = torch.chunk(b, tensor_parallel_size, 0)

        a_local = a_partial[rank].cuda()
        b_local = b_partial[rank].cuda()
        c = c.cuda()
        gamma = gamma.cuda()
        internal_global_sf = internal_global_sf.cuda()

        ub_size = c.nelement() * c.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)

        ub0_tensor = create_userbuffers_tensor(c.size(), a.dtype)
        hidden = torch.matmul(a_local, b_local, out=ub0_tensor)
        mapping = Mapping(
            world_size=tensor_parallel_size,
            tp_size=tensor_parallel_size,
            rank=rank,
        )
        ar = AllReduce(mapping=mapping, strategy=AllReduceStrategy.UB)
        ar_params = AllReduceParams(
            strategy=AllReduceStrategy.UB,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
            residual=c,
            norm_weight=gamma,
            scale=internal_global_sf,
            eps=eps)
        q_internal, scale_ub, residual = ar.forward(hidden,
                                                    all_reduce_params=ar_params)

        def e2m1_and_ufp8sf_scale_to_float_v2(e2m1_tensor,
                                              ufp8_scale_tensor,
                                              global_scale_tensor,
                                              sf_vec_size,
                                              ufp8_type,
                                              is_sf_swizzled_layout=True):
            return torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
                e2m1_tensor, ufp8_scale_tensor, global_scale_tensor,
                sf_vec_size, ufp8_type, is_sf_swizzled_layout)

        def quant_fp4(x, sf):
            return torch.ops.trtllm.fp4_quantize(x, sf, 16, False)

        ref, ref_sf = quant_fp4(ref_internal.cuda(), internal_global_sf)
        t1 = e2m1_and_ufp8sf_scale_to_float_v2(q_internal.cpu(), scale_ub.cpu(),
                                               1 / internal_global_sf.cpu(), 16,
                                               1)
        t2 = e2m1_and_ufp8sf_scale_to_float_v2(ref.cpu(), ref_sf.cpu(),
                                               1 / internal_global_sf.cpu(), 16,
                                               1)
        # FP4 is very sensitive to the value difference. On blackwell, the multimem.ld_reduce
        # instruction generates slight difference.
        torch.testing.assert_close(t1, t2, atol=1.5, rtol=1e-2)

        residual = userbuffers_allreduce_finalize(residual, False)
        torch.testing.assert_close(ref_residual,
                                   residual.cpu(),
                                   atol=5e-1,
                                   rtol=1e-2)

    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("mnk", [(256, 512, 256), (32, 64, 64)],
                         ids=lambda x: f"m{x[0]}_n{x[1]}_k{x[2]}")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@skip_pre_blackwell_unittest
def test_user_buffers_ar_rms_norm_fp4(mnk, mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = 2
    dtype = torch.float16
    m = mnk[0]
    n = mnk[1]
    k = mnk[2]
    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.randn((m, n), dtype=dtype)
    gamma = torch.randn((n), dtype=dtype)

    results = mpi_pool_executor.map(
        run_single_rank_ar_rms_norm_fp4,
        *zip(*[(tensor_parallel_size, a, b, c, gamma)] * tensor_parallel_size))
    for r in results:
        assert r is True


class UBMMAddModel(nn.Module):

    def __init__(self, tp_size, rank, hidden_size, dtype, eps, norm0_gamma,
                 norm1_gamma, norm2_gamma):
        super().__init__()
        self.tp_size = tp_size
        self.rank = rank
        self.hidden_size = hidden_size
        self.dtype = dtype
        mapping = Mapping(
            world_size=tp_size,
            tp_size=tp_size,
            rank=rank,
        )
        self.ar_0 = AllReduce(mapping=mapping).cuda()
        self.ar_1 = AllReduce(mapping=mapping).cuda()
        self.ar_2 = AllReduce(mapping=mapping).cuda()
        self.norm0 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()
        self.norm1 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()
        self.norm2 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()
        self.norm0.weight.data.copy_(norm0_gamma)
        self.norm1.weight.data.copy_(norm1_gamma)
        self.norm2.weight.data.copy_(norm2_gamma)

    def forward(self, mm0_input_0, mm0_input_1, mm1_input_0, mm1_input_1,
                residual_0, residual_1):
        mm0_output = torch.matmul(mm0_input_0, mm0_input_1)
        ar0_output = self.ar_0(mm0_output)
        ar0_add = ar0_output + residual_0
        norm0_output = self.norm0(ar0_add)

        mm1_output = torch.matmul(mm1_input_0, mm1_input_1)
        ar1_output = self.ar_1(mm1_output)
        ar1_add = ar1_output + residual_1
        norm1_output = self.norm1(ar1_add)
        sum = norm0_output + norm1_output
        ar2_output = self.ar_2(sum)
        ar2_add = ar2_output + ar1_add
        return self.norm2(ar2_add)


def run_single_rank_ub_mm_add_pass(tensor_parallel_size, num_tokens,
                                   hidden_size, dtype, norm0_gamma, norm1_gamma,
                                   norm2_gamma):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        support = ub.ub_supported()
        if not support:
            return True

        eps = 1e-6

        mm0_input_0 = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
        mm0_input_1 = torch.randn((hidden_size, hidden_size),
                                  dtype=dtype).cuda()
        mm1_input_0 = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
        mm1_input_1 = torch.randn((hidden_size, hidden_size),
                                  dtype=dtype).cuda()
        residual_0 = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
        residual_1 = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()

        ub_size = mm0_input_0.nelement() * mm0_input_0.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)
        model = UBMMAddModel(tensor_parallel_size, rank, hidden_size, dtype,
                             eps, norm0_gamma, norm1_gamma, norm2_gamma)
        backend = Backend(enable_inductor=False, enable_userbuffers=True)
        model_opt = torch.compile(model, backend=backend, fullgraph=True)
        with torch.inference_mode():
            output_fused = model_opt(mm0_input_0, mm0_input_1, mm1_input_0,
                                     mm1_input_1, residual_0, residual_1)
            torch.cuda.synchronize()
            del output_fused
            # Run twice
            output_fused = model_opt(mm0_input_0, mm0_input_1, mm1_input_0,
                                     mm1_input_1, residual_0, residual_1)
            torch.cuda.synchronize()
            output_ref = model(mm0_input_0, mm0_input_1, mm1_input_0,
                               mm1_input_1, residual_0, residual_1)
        # 3 AR_NORM fusion happens first
        # 0 AR_NORM fused with Quant
        # 3 AR_NORM replacement
        # 3 Prologue
        # 1 UB Finalize Removal
        assert backend.match_count == [3, 0, 0, 3, 0, 3, 0, 1, 0]
        torch.cuda.synchronize()

        if rank == 0:
            torch.testing.assert_close(output_fused,
                                       output_ref,
                                       atol=6e-1,
                                       rtol=1e-2)
    except Exception:
        traceback.print_exc()

        return False
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden", [512, 32], ids=lambda x: f"_hidden{x}")
@pytest.mark.parametrize("tokens", [256, 16], ids=lambda x: f"_tokens{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=lambda x: "fp16" if x == torch.float16 else "bf16")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_user_buffers_mm_add_prologue(hidden, tokens, dtype, mpi_pool_executor):
    torch.manual_seed(42)
    tensor_parallel_size = 2

    norm0_gamma = torch.randn((hidden, ), dtype=dtype)
    norm1_gamma = torch.randn((hidden, ), dtype=dtype)
    norm2_gamma = torch.randn((hidden, ), dtype=dtype)

    results = mpi_pool_executor.map(
        run_single_rank_ub_mm_add_pass,
        *zip(*[(tensor_parallel_size, tokens, hidden, dtype, norm0_gamma,
                norm1_gamma, norm2_gamma)] * tensor_parallel_size))
    for r in results:
        assert r is True


class UBFp4TestModel(nn.Module):

    def __init__(self, tp_size, rank, hidden_size, dtype, eps, l0_weight,
                 l0_input_scale, l0_weight_scale1, l0_weight_scale2, l1_weight,
                 l1_input_scale, l1_weight_scale1, l1_weight_scale2, l2_weight,
                 l2_input_scale, l2_weight_scale1, l2_weight_scale2, l3_weight,
                 l3_input_scale, l3_weight_scale1, l3_weight_scale2, l4_weight,
                 l4_input_scale, l4_weight_scale1, l4_weight_scale2,
                 norm0_gamma, norm1_gamma, norm2_gamma):
        super().__init__()
        quant_config = QuantConfig()
        quant_config.quant_algo = QuantAlgo.NVFP4
        quant_config.layer_quant_mode
        self.rank = rank
        self.tp_size = tp_size
        self.l0 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=Mapping(
                             world_size=tp_size,
                             tp_size=tp_size,
                             rank=rank,
                         ),
                         tensor_parallel_mode=TensorParallelMode.ROW,
                         quant_config=quant_config).cuda()
        self.l1 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=Mapping(
                             world_size=tp_size,
                             tp_size=tp_size,
                             rank=rank,
                         ),
                         tensor_parallel_mode=TensorParallelMode.COLUMN,
                         quant_config=quant_config).cuda()
        self.l2 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=Mapping(
                             world_size=tp_size,
                             tp_size=tp_size,
                             rank=rank,
                         ),
                         tensor_parallel_mode=TensorParallelMode.ROW,
                         quant_config=quant_config).cuda()
        self.l3 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=Mapping(
                             world_size=tp_size,
                             tp_size=tp_size,
                             rank=rank,
                         ),
                         tensor_parallel_mode=TensorParallelMode.COLUMN,
                         quant_config=quant_config).cuda()
        self.l4 = Linear(in_features=hidden_size,
                         out_features=hidden_size,
                         bias=False,
                         dtype=dtype,
                         mapping=Mapping(
                             world_size=tp_size,
                             tp_size=tp_size,
                             rank=rank,
                         ),
                         tensor_parallel_mode=TensorParallelMode.ROW,
                         quant_config=quant_config).cuda()
        self.norm0 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()
        self.norm1 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()
        self.norm2 = RMSNorm(hidden_size=hidden_size, eps=eps,
                             dtype=dtype).cuda()

        self.l0.load_weights([
            dict(weight=l0_weight,
                 input_scale=1.0 / l0_input_scale,
                 weight_scale=l0_weight_scale1,
                 weight_scale_2=1.0 / l0_weight_scale2)
        ])
        self.l1.load_weights([
            dict(weight=l1_weight,
                 input_scale=1.0 / l1_input_scale,
                 weight_scale=l1_weight_scale1,
                 weight_scale_2=1.0 / l1_weight_scale2)
        ])
        self.l2.load_weights([
            dict(weight=l2_weight,
                 input_scale=1.0 / l2_input_scale,
                 weight_scale=l2_weight_scale1,
                 weight_scale_2=1.0 / l2_weight_scale2)
        ])
        self.l3.load_weights([
            dict(weight=l3_weight,
                 input_scale=1.0 / l3_input_scale,
                 weight_scale=l3_weight_scale1,
                 weight_scale_2=1.0 / l3_weight_scale2)
        ])
        self.l4.load_weights([
            dict(weight=l4_weight,
                 input_scale=1.0 / l4_input_scale,
                 weight_scale=l4_weight_scale1,
                 weight_scale_2=1.0 / l4_weight_scale2)
        ])
        self.norm0.weight.data.copy_(norm0_gamma)
        self.norm1.weight.data.copy_(norm1_gamma)
        self.norm2.weight.data.copy_(norm2_gamma)
        torch.cuda.current_stream().synchronize()

    def forward(self, input):
        local = torch.chunk(input, self.tp_size,
                            1)[self.rank].contiguous()  # [token, hidden/tp]
        hidden0 = self.l0(
            local
        )  # [token, hidden/tp] * [hidden/tp, hidden] -> [token, hidden], row parallel
        hidden1 = hidden0 + input  # [token, hidden]
        norm0 = self.norm0(hidden1)  # [token, hidden]
        hidden2 = self.l1(
            norm0
        )  # [token, hidden] * [hidden, hidden/tp] -> [token, hidden/tp], col parallel, no gather
        hidden3 = self.l2(
            hidden2
        )  # # [token, hidden/tp] * [hidden/tp, hidden] -> [token, hidden], row parallel
        hidden4 = hidden3 + hidden1
        norm1 = self.norm1(hidden4)
        hidden5 = self.l3(
            norm1
        )  # [token, hidden] * [hidden, hidden/tp] -> [token, hidden/tp], col parallel, no gather
        hidden6 = self.l4(
            hidden5
        )  # # [token, hidden/tp] * [hidden/tp, hidden] -> [token, hidden], row parallel
        hidden7 = hidden6 + hidden4
        res = self.norm2(hidden7)
        return res


def run_single_rank_ub_pass_fp4(
        tensor_parallel_size, input, l0_weight, l0_input_scale, l0_weight_scale,
        l1_weight, l1_input_scale, l1_weight_scale, l2_weight, l2_input_scale,
        l2_weight_scale, l3_weight, l3_input_scale, l3_weight_scale, l4_weight,
        l4_input_scale, l4_weight_scale, norm0_gamma, norm1_gamma, norm2_gamma):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    try:
        support = ub.ub_supported()
        if not support:
            return True

        eps = 1e-5
        hidden_size = input.size(-1)
        dtype = input.dtype
        input = input.cuda()
        ub_size = input.nelement() * input.element_size()
        init_userbuffers_allocator(tensor_parallel_size, rank, ub_size)

        def quant_fp4(x, sf):
            return torch.ops.trtllm.fp4_quantize(x, sf, 16, False)

        l0_weight, l0_weight_scale1 = quant_fp4(l0_weight.cuda(),
                                                l0_weight_scale.cuda())
        l1_weight, l1_weight_scale1 = quant_fp4(l1_weight.cuda(),
                                                l1_weight_scale.cuda())
        l2_weight, l2_weight_scale1 = quant_fp4(l2_weight.cuda(),
                                                l2_weight_scale.cuda())
        l3_weight, l3_weight_scale1 = quant_fp4(l3_weight.cuda(),
                                                l3_weight_scale.cuda())
        l4_weight, l4_weight_scale1 = quant_fp4(l4_weight.cuda(),
                                                l4_weight_scale.cuda())

        def block_scale_unswizzled(scale):
            sz = fp4_utils.pad_up(hidden_size, 128)
            return torch.ops.trtllm.block_scale_interleave_reverse(
                scale.cpu().view(sz, -1))[0:hidden_size]

        l0_weight_scale_block_unswizzled = block_scale_unswizzled(
            l0_weight_scale1)
        l1_weight_scale_block_unswizzled = block_scale_unswizzled(
            l1_weight_scale1)
        l2_weight_scale_block_unswizzled = block_scale_unswizzled(
            l2_weight_scale1)
        l3_weight_scale_block_unswizzled = block_scale_unswizzled(
            l3_weight_scale1)
        l4_weight_scale_block_unswizzled = block_scale_unswizzled(
            l4_weight_scale1)

        model = UBFp4TestModel(
            tensor_parallel_size, rank, hidden_size, dtype, eps, l0_weight,
            l0_input_scale,
            l0_weight_scale_block_unswizzled.view(torch.float8_e4m3fn),
            l0_weight_scale, l1_weight, l1_input_scale,
            l1_weight_scale_block_unswizzled.view(torch.float8_e4m3fn),
            l1_weight_scale, l2_weight, l2_input_scale,
            l2_weight_scale_block_unswizzled.view(torch.float8_e4m3fn),
            l2_weight_scale, l3_weight, l3_input_scale,
            l3_weight_scale_block_unswizzled.view(torch.float8_e4m3fn),
            l3_weight_scale, l4_weight, l4_input_scale,
            l4_weight_scale_block_unswizzled.view(torch.float8_e4m3fn),
            l4_weight_scale, norm0_gamma, norm1_gamma, norm2_gamma)

        backend = Backend(enable_inductor=False, enable_userbuffers=True)
        model_opt = torch.compile(model, backend=backend, fullgraph=True)
        with torch.inference_mode():
            output_ref = model(input)
            output_fused = model_opt(input)

        # 3 AR_NORM fusion happens first
        # 2 AR_NORM fused with Quant
        # 3 AR_NORM replacement
        # 3 Scaled MM Prologue
        # 2 UB Finalize Removal
        assert backend.match_count == [3, 0, 2, 0, 3, 0, 3, 0, 2, 0]
        torch.cuda.synchronize()
        torch.testing.assert_close(output_fused,
                                   output_ref,
                                   atol=1.6,
                                   rtol=1e-2)
    except Exception:
        traceback.print_exc()

        return False
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("hidden", [512, 128], ids=lambda x: f"_hidden{x}")
@pytest.mark.parametrize("tokens", [256, 16], ids=lambda x: f"_tokens{x}")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         ids=lambda x: "fp16" if x == torch.float16 else "bf16")
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
@skip_pre_blackwell_unittest
def test_user_buffers_fp4_pass(hidden, tokens, dtype, mpi_pool_executor):
    torch.manual_seed(43)
    tensor_parallel_size = 2

    eps = 1e-5

    input = torch.randn((tokens, hidden), dtype=dtype) * 0.05
    l0_weight = torch.randn((hidden, hidden), dtype=dtype) * 0.05
    l1_weight = torch.randn((hidden, hidden), dtype=dtype) * 0.05
    l2_weight = torch.randn((hidden, hidden), dtype=dtype) * 0.05
    l3_weight = torch.randn((hidden, hidden), dtype=dtype) * 0.05
    l4_weight = torch.randn((hidden, hidden), dtype=dtype) * 0.05
    norm0_gamma = torch.randn((hidden, ), dtype=dtype)
    norm1_gamma = torch.randn((hidden, ), dtype=dtype)
    norm2_gamma = torch.randn((hidden, ), dtype=dtype)

    def fp4_scale(x):
        return (448 * 6) / x.abs().max().float()

    input_scale = fp4_scale(input)
    l0_weight_scale = fp4_scale(l0_weight)
    l1_weight_scale = fp4_scale(l1_weight)
    l2_weight_scale = fp4_scale(l2_weight)
    l3_weight_scale = fp4_scale(l3_weight)
    l4_weight_scale = fp4_scale(l4_weight)

    l1_residual = input @ l0_weight.t() + input
    l0_res = rms_norm(l1_residual, norm0_gamma, eps).to(dtype)
    l0_res_scale = fp4_scale(l0_res)

    l1_res = l0_res @ l1_weight.t()
    l1_res_scale = fp4_scale(l1_res)
    l2_residual = l1_res @ l2_weight.t() + l1_residual
    l2_res = rms_norm(l2_residual, norm1_gamma, eps).to(dtype)
    l2_res_scale = fp4_scale(l2_res)
    l3_res = l2_res @ l3_weight.t()
    l3_res_scale = fp4_scale(l3_res)

    results = mpi_pool_executor.map(
        run_single_rank_ub_pass_fp4,
        *zip(*[(tensor_parallel_size, input, l0_weight, input_scale,
                l0_weight_scale, l1_weight, l0_res_scale, l1_weight_scale,
                l2_weight, l1_res_scale, l2_weight_scale, l3_weight,
                l2_res_scale, l3_weight_scale, l4_weight, l3_res_scale,
                l4_weight_scale, norm0_gamma, norm1_gamma, norm2_gamma)] *
             tensor_parallel_size))
    for r in results:
        assert r is True
