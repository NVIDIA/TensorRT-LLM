import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor

import tensorrt_llm
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.distributed import (AllReduce, AllReduceFusionOp,
                                             AllReduceParams, AllReduceStrategy,
                                             ParallelConfig, TensorParallelMode,
                                             userbuffers_allreduce_finalize)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def init_runtime(tp_size, max_ub_size):
    ub.ub_initialize(tp_size)
    allocated_buffer_0 = ub.ub_allocate(0, max_ub_size)
    allocated_buffer_1 = ub.ub_allocate(1, max_ub_size)
    return allocated_buffer_0, allocated_buffer_1


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
        ub0_addr, ub1_addr = init_runtime(tensor_parallel_size, ub_size)
        assert ub.ub_is_initialized()

        ub0 = ub.ub_get(0)
        assert not ub0.invalid()
        assert ub0_addr == ub0.addr
        ub1 = ub.ub_get(1)
        assert not ub1.invalid()
        assert ub1_addr == ub1.addr

        ub0_tensor = convert_to_torch_tensor(
            TensorWrapper(ub0.addr, a.dtype, c.size()))
        ub1_tensor = convert_to_torch_tensor(
            TensorWrapper(ub1.addr, a.dtype, c.size()))
        internal = torch.matmul(a, b, out=ub0_tensor)
        res = torch.add(internal, c, out=ub1_tensor)

        torch.cuda.synchronize()
        torch.testing.assert_close(ref, res)

        ub.ub_deallocate(ub0_addr)
        ub.ub_deallocate(ub1_addr)

    except Exception:
        traceback.print_exc()
        raise
    return True


def run_single_rank_ar_rms_norm(tensor_parallel_size, a, b, c, gamma, scale):
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
        ub0_addr, ub1_addr = init_runtime(tensor_parallel_size, ub_size)
        assert ub.ub_is_initialized()

        ub0 = ub.ub_get(0)
        assert not ub0.invalid()
        assert ub0_addr == ub0.addr
        ub0_tensor = convert_to_torch_tensor(
            TensorWrapper(ub0.addr, a.dtype, c.size()))
        hidden = torch.matmul(a_local, b_local, out=ub0_tensor)
        parallel_config = ParallelConfig(
            tensor_parallel_rank=rank,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_mode=TensorParallelMode.COLUMN)
        ar = AllReduce(parallel_config, strategy=AllReduceStrategy.UB)
        ar_params = AllReduceParams(
            strategy=AllReduceStrategy.UB,
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=c,
            norm_weight=gamma,
            scale=scale,
            eps=eps)
        q_res, residual = ar.forward(hidden, all_reduce_params=ar_params)
        res = dequant(q_res, scale, torch.float16)
        residual = userbuffers_allreduce_finalize(residual)

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
            torch.testing.assert_close(ref, res)
            torch.testing.assert_close(ref_residual, residual)

    except Exception:
        traceback.print_exc()
        raise
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
def test_user_buffers():
    torch.manual_seed(42)
    tensor_parallel_size = 2
    dtype = torch.float32
    m = 16
    n = 32
    k = 16
    a = torch.randn((m, k), dtype=dtype)
    b = torch.randn((k, n), dtype=dtype)
    c = torch.randn((m, n), dtype=dtype)
    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank,
            *zip(*[(tensor_parallel_size, a, b, c)] * tensor_parallel_size))
        for r in results:
            assert r is True


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason='needs 2 GPUs to run this test')
@pytest.mark.parametrize("mnk", [(256, 512, 256), (32, 16, 64)],
                         ids=lambda x: f"m{x[0]}_n{x[1]}_k{x[2]}")
def test_user_buffers_ar_rms_norm(mnk):
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

    with MPIPoolExecutor(max_workers=tensor_parallel_size) as executor:
        results = executor.map(
            run_single_rank_ar_rms_norm,
            *zip(*[(tensor_parallel_size, a, b, c, gamma, scale)] *
                 tensor_parallel_size))
        for r in results:
            assert r is True


if __name__ == '__main__':
    test_user_buffers()
    test_user_buffers_ar_rms_norm((256, 512, 256))
    test_user_buffers_ar_rms_norm((32, 16, 64))
