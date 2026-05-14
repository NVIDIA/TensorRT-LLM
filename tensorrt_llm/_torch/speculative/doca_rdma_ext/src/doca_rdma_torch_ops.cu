/*
 * PyTorch custom ops for DOCA RDMA GPU-initiated transfers
 *
 * These ops integrate with PyTorch's stream management for CUDA graph compatibility.
 */

// DOCA GPUNetIO device-side headers (must be first, like NIXL)
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#include <doca_gpunetio_dev_verbs_twosided.cuh>
// Note: doca_gpunetio_dev_buf.cuh requires libdoca_gpunetio_device.a which has ABI issues
// with PyTorch (ABI v8 vs v7). Disabled until NVIDIA resolves this.
// #include <doca_gpunetio_dev_buf.cuh>
#include <cuda.h>
#include <cuda/atomic>

#include <ATen/cuda/CUDAContext.h>
#include <cstdlib>
#include <cstring>
#include <torch/extension.h>

#ifndef MLX5_OPCODE_RDMA_WRITE_IMM
#define MLX5_OPCODE_RDMA_WRITE_IMM 0x09
#endif
#ifndef MLX5_OPCODE_RDMA_WRITE
#define MLX5_OPCODE_RDMA_WRITE 0x08
#endif

// Use system mlx5_cqe64 definition
#include <infiniband/mlx5dv.h>

// Device helper for byte swap
__device__ __forceinline__ uint32_t device_ntohl(uint32_t x)
{
    return ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24);
}

__device__ __forceinline__ uint16_t device_ntohs(uint16_t x)
{
    return (uint16_t) (((x & 0x00FFU) << 8) | ((x & 0xFF00U) >> 8));
}

// Global error counter to limit spam (managed memory for host access)
__device__ uint32_t g_rdma_error_count = 0;
constexpr uint32_t MAX_ERROR_PRINTS = 5;

// Queue sizes (must match doca_rdma_endpoint.cpp)
constexpr uint64_t RQ_SIZE = 64;

//
// CUDA Kernels
//

__global__ void rdma_write_imm_kernel(doca_gpu_dev_verbs_qp* qp, void* local_buf,
    uint32_t lkey,      // Host-endian from Python
    uint64_t remote_addr,
    uint32_t rkey,      // Host-endian from Python
    size_t size,
    uint32_t imm_value, // Host-endian from Python
    uint64_t* wqe_idx_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // This DOCA build expects callers to pass swapped keys/immediates.
    uint32_t lkey_be = device_ntohl(lkey);
    uint32_t rkey_be = device_ntohl(rkey);
    uint32_t imm_be = device_ntohl(imm_value);

    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, 1);
    doca_gpu_dev_verbs_wqe* wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE_IMM,
        DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, imm_be, remote_addr, rkey_be, (uint64_t) local_buf, lkey_be, size);

    doca_gpu_dev_verbs_mark_wqes_ready(qp, wqe_idx, wqe_idx);
    doca_gpu_dev_verbs_submit(qp, wqe_idx + 1);

    if (wqe_idx_out)
        *wqe_idx_out = wqe_idx;
}

__global__ void poll_send_cq_kernel(doca_gpu_dev_verbs_qp* qp, uint64_t wqe_idx, int* completed)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
    *completed = (doca_gpu_dev_verbs_poll_cq_at(cq, wqe_idx) == 0) ? 1 : 0;
}

__global__ void wait_send_cq_kernel(doca_gpu_dev_verbs_qp* qp,
    int64_t* wqe_idx_ptr // Read from device memory for graph compatibility
)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    uint64_t wqe_idx = (uint64_t) *wqe_idx_ptr;
    doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);

    if (!cq)
        return;

    int poll_count = 0;
    int const max_polls = 10000000;
    int status = -1;
    while (poll_count < max_polls)
    {
        status = doca_gpu_dev_verbs_poll_cq_at(cq, wqe_idx);
        if (status == 0)
        {
            return;
        }
        poll_count++;
    }

    uint32_t err_num = atomicAdd(&g_rdma_error_count, 1);
    if (err_num < MAX_ERROR_PRINTS)
    {
        printf("[GPU] Send CQ TIMEOUT: polls=%d wqe_idx=%llu last_status=%d\n", poll_count,
            (unsigned long long) wqe_idx, status);
    }
}

__global__ void poll_recv_cq_kernel(doca_gpu_dev_verbs_qp* qp, uint64_t* cq_idx, uint32_t* imm_out, int* completed)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    uint64_t idx = *cq_idx;

    if (doca_gpu_dev_verbs_poll_cq_at(cq, idx) == 0)
    {
        uint8_t* cqe_base = (uint8_t*) __ldg((uintptr_t*) &cq->cqe_daddr);
        uint32_t cqe_num = __ldg(&cq->cqe_num);
        // Use modulo instead of bitmask so this remains correct even if queue depth
        // is not a power of two on some provider/runtime combinations.
        uint32_t cqe_slot = (uint32_t) (idx % cqe_num);

        struct mlx5_cqe64* cqe = (struct mlx5_cqe64*) (cqe_base + cqe_slot * 64);
        *imm_out = device_ntohl(__ldg(&cqe->imm_inval_pkey));
        *cq_idx = idx + 1;
        *completed = 1;

        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
        doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&cq->cqe_ci, idx + 1);
    }
    else
    {
        *completed = 0;
    }
}

__global__ void wait_recv_cq_kernel(doca_gpu_dev_verbs_qp* qp, uint64_t* cq_idx, uint32_t* imm_out,
    uint8_t* recv_buf = nullptr // Optional: unused, kept for API compatibility
)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_rq(qp);
    if (!cq)
        return;

    uint8_t* cqe_base = (uint8_t*) __ldg((uintptr_t*) &cq->cqe_daddr);
    uint32_t cqe_num = __ldg(&cq->cqe_num);

    uint64_t idx = *cq_idx;
    // Use modulo instead of bitmask so this remains correct even if queue depth
    // is not a power of two on some provider/runtime combinations.
    uint32_t cqe_slot = (uint32_t) (idx % cqe_num);
    volatile struct mlx5_cqe64* cqe = (volatile struct mlx5_cqe64*) (cqe_base + cqe_slot * 64);

    int poll_count = 0;
    int const max_polls = 50000000;
    uint8_t op_own;
    uint8_t expected_owner = (idx / cqe_num) & 1;

    while (poll_count < max_polls)
    {
        op_own = cqe->op_own;
        uint8_t opcode = op_own >> 4;
        uint8_t owner = op_own & 1;

        if (owner == expected_owner && opcode != 0x0f)
        {
            *imm_out = device_ntohl(cqe->imm_inval_pkey);
            *cq_idx = idx + 1;

            doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
            doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
                &cq->cqe_ci, idx + 1);
            return;
        }

        poll_count++;
    }

    uint32_t err_num = atomicAdd(&g_rdma_error_count, 1);
    if (err_num < MAX_ERROR_PRINTS)
    {
        printf("[GPU] Recv CQ TIMEOUT after %d polls!\n", poll_count);
    }

    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&cq->cqe_ci, idx + 1);
}

__global__ void post_recv_wrs_kernel(doca_gpu_dev_verbs_qp* qp, int num_wrs,
    uint64_t* recv_wq_idx // Cumulative index tracking
)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    uint64_t start_idx = *recv_wq_idx;

    // For WRITE_WITH_IMM, recv WQEs are empty. Pre-initialize once, then only bump CI + doorbell.
    if (start_idx == 0)
    {
        for (uint64_t i = 0; i < RQ_SIZE; i++)
        {
            struct mlx5_wqe_data_seg* rwqe_ptr = doca_gpu_dev_verbs_get_rwqe_ptr(qp, i);
            if (!rwqe_ptr)
                return;
            doca_gpu_dev_verbs_wqe_prepare_recv(qp, rwqe_ptr, 0, 0, 0);
        }
    }

    uint64_t new_idx = start_idx + num_wrs;
    *recv_wq_idx = new_idx;

    // Submit with cumulative count using GPU SM doorbell handler.
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, new_idx);
}

__global__ void post_recv_wrs_kernel_auto(doca_gpu_dev_verbs_qp* qp, int num_wrs,
    uint64_t* recv_wq_idx // Cumulative index tracking
)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    uint64_t start_idx = *recv_wq_idx;

    if (start_idx == 0)
    {
        for (uint64_t i = 0; i < RQ_SIZE; i++)
        {
            struct mlx5_wqe_data_seg* rwqe_ptr = doca_gpu_dev_verbs_get_rwqe_ptr(qp, i);
            if (!rwqe_ptr)
                return;
            doca_gpu_dev_verbs_wqe_prepare_recv(qp, rwqe_ptr, 0, 0, 0);
        }
    }

    uint64_t new_idx = start_idx + num_wrs;
    *recv_wq_idx = new_idx;

    // Explicit AUTO fallback for platforms where GPU SM DB is unavailable.
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO, DOCA_GPUNETIO_VERBS_QP_RQ>(qp, new_idx);
}

//
// PyTorch Op Wrappers
//

void rdma_write_with_imm(int64_t qp_ptr, int64_t local_addr, int64_t local_size, int64_t lkey, int64_t remote_addr,
    int64_t rkey, int64_t imm_value, torch::Tensor wqe_idx_out)
{
    TORCH_CHECK(wqe_idx_out.is_cuda(), "wqe_idx_out must be on CUDA");
    TORCH_CHECK(wqe_idx_out.dtype() == torch::kInt64, "wqe_idx_out must be int64");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rdma_write_imm_kernel<<<1, 1, 0, stream>>>((doca_gpu_dev_verbs_qp*) qp_ptr, (void*) local_addr, (uint32_t) lkey,
        (uint64_t) remote_addr, (uint32_t) rkey, (size_t) local_size, (uint32_t) imm_value,
        (uint64_t*) wqe_idx_out.data_ptr());
}

void wait_send_completion(int64_t qp_ptr, torch::Tensor wqe_idx)
{
    TORCH_CHECK(wqe_idx.is_cuda(), "wqe_idx must be on CUDA");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Kernel reads from device memory for graph compatibility
    wait_send_cq_kernel<<<1, 1, 0, stream>>>((doca_gpu_dev_verbs_qp*) qp_ptr, (int64_t*) wqe_idx.data_ptr());
}

void wait_send_completion_v2(int64_t qp_ptr, torch::Tensor buf, torch::Tensor wqe_idx_tensor)
{
    // buf is the tensor we're waiting to send - used for dispatch and graph dependencies
    // wqe_idx_tensor allows graph capture (kernel reads from device memory)
    TORCH_CHECK(buf.is_cuda(), "buf must be on CUDA");
    TORCH_CHECK(wqe_idx_tensor.is_cuda(), "wqe_idx_tensor must be on CUDA");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    wait_send_cq_kernel<<<1, 1, 0, stream>>>((doca_gpu_dev_verbs_qp*) qp_ptr, (int64_t*) wqe_idx_tensor.data_ptr());
}

torch::Tensor poll_recv_cq(int64_t qp_ptr, torch::Tensor cq_idx, torch::Tensor imm_out)
{
    TORCH_CHECK(cq_idx.is_cuda(), "cq_idx must be on CUDA");
    TORCH_CHECK(imm_out.is_cuda(), "imm_out must be on CUDA");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto completed = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(cq_idx.device()));

    poll_recv_cq_kernel<<<1, 1, 0, stream>>>((doca_gpu_dev_verbs_qp*) qp_ptr, (uint64_t*) cq_idx.data_ptr(),
        (uint32_t*) imm_out.data_ptr(), (int*) completed.data_ptr());

    return completed;
}

void wait_recv_completion(int64_t qp_ptr, torch::Tensor cq_idx, torch::Tensor imm_out,
    torch::Tensor recv_buf // Optional: for debugging recv buffer contents
)
{
    TORCH_CHECK(cq_idx.is_cuda(), "cq_idx must be on CUDA");
    TORCH_CHECK(imm_out.is_cuda(), "imm_out must be on CUDA");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Pass recv_buf address if provided (numel > 0), otherwise nullptr
    uint8_t* recv_buf_ptr = recv_buf.numel() > 0 ? (uint8_t*) recv_buf.data_ptr() : nullptr;

    wait_recv_cq_kernel<<<1, 1, 0, stream>>>(
        (doca_gpu_dev_verbs_qp*) qp_ptr, (uint64_t*) cq_idx.data_ptr(), (uint32_t*) imm_out.data_ptr(), recv_buf_ptr);
}

void post_recv_wrs(int64_t qp_ptr,
    torch::Tensor recv_wq_idx, // Cumulative index tracking tensor
    int64_t num_wrs)
{
    TORCH_CHECK(recv_wq_idx.is_cuda(), "recv_wq_idx must be on CUDA");
    TORCH_CHECK(recv_wq_idx.dtype() == torch::kInt64, "recv_wq_idx must be int64");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Keep recv posting handler aligned with the selected QP export mode.
    // Default is GPU SM DB (handler 2). Override with:
    //   DOCA_RDMA_POST_RECV_HANDLER=auto
    char const* handler_env = std::getenv("DOCA_RDMA_POST_RECV_HANDLER");
    bool const use_auto
        = handler_env != nullptr && (std::strcmp(handler_env, "auto") == 0 || std::strcmp(handler_env, "0") == 0);

    if (use_auto)
    {
        post_recv_wrs_kernel_auto<<<1, 1, 0, stream>>>(
            (doca_gpu_dev_verbs_qp*) qp_ptr, (int) num_wrs, (uint64_t*) recv_wq_idx.data_ptr());
    }
    else
    {
        post_recv_wrs_kernel<<<1, 1, 0, stream>>>(
            (doca_gpu_dev_verbs_qp*) qp_ptr, (int) num_wrs, (uint64_t*) recv_wq_idx.data_ptr());
    }
}

void fused_send_recv(torch::Tensor input, torch::Tensor send_buf, torch::Tensor recv_buf, int64_t qp_ptr,
    int64_t local_addr, int64_t local_size, int64_t lkey, int64_t remote_addr, int64_t rkey, int64_t imm_value,
    torch::Tensor wqe_idx_out, torch::Tensor recv_wq_idx, torch::Tensor cq_idx, torch::Tensor imm_out, int64_t num_wrs)
{
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(send_buf.is_cuda(), "send_buf must be on CUDA");
    TORCH_CHECK(recv_buf.is_cuda(), "recv_buf must be on CUDA");
    TORCH_CHECK(send_buf.scalar_type() == input.scalar_type(), "send_buf must match input dtype");
    TORCH_CHECK(send_buf.numel() == input.numel(), "send_buf and input must have the same number of elements");
    TORCH_CHECK(wqe_idx_out.is_cuda(), "wqe_idx_out must be on CUDA");
    TORCH_CHECK(wqe_idx_out.dtype() == torch::kInt64, "wqe_idx_out must be int64");
    TORCH_CHECK(recv_wq_idx.is_cuda(), "recv_wq_idx must be on CUDA");
    TORCH_CHECK(recv_wq_idx.dtype() == torch::kInt64, "recv_wq_idx must be int64");
    TORCH_CHECK(cq_idx.is_cuda(), "cq_idx must be on CUDA");
    TORCH_CHECK(cq_idx.dtype() == torch::kInt64, "cq_idx must be int64");
    TORCH_CHECK(imm_out.is_cuda(), "imm_out must be on CUDA");
    TORCH_CHECK(imm_out.dtype() == torch::kInt32, "imm_out must be int32");

    int64_t input_bytes = input.numel() * input.element_size();
    TORCH_CHECK(input_bytes == local_size, "input size in bytes must match local_size");

    send_buf.copy_(input);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    rdma_write_imm_kernel<<<1, 1, 0, stream>>>((doca_gpu_dev_verbs_qp*) qp_ptr, (void*) local_addr, (uint32_t) lkey,
        (uint64_t) remote_addr, (uint32_t) rkey, (size_t) local_size, (uint32_t) imm_value,
        (uint64_t*) wqe_idx_out.data_ptr());

    post_recv_wrs_kernel<<<1, 1, 0, stream>>>(
        (doca_gpu_dev_verbs_qp*) qp_ptr, (int) num_wrs, (uint64_t*) recv_wq_idx.data_ptr());

    wait_send_cq_kernel<<<1, 1, 0, stream>>>((doca_gpu_dev_verbs_qp*) qp_ptr, (int64_t*) wqe_idx_out.data_ptr());

    uint8_t* recv_buf_ptr = recv_buf.numel() > 0 ? (uint8_t*) recv_buf.data_ptr() : nullptr;

    wait_recv_cq_kernel<<<1, 1, 0, stream>>>(
        (doca_gpu_dev_verbs_qp*) qp_ptr, (uint64_t*) cq_idx.data_ptr(), (uint32_t*) imm_out.data_ptr(), recv_buf_ptr);
}

// Wrap a GPU memory address as a PyTorch tensor (zero-copy)
torch::Tensor wrap_gpu_memory(int64_t addr, int64_t size, int64_t device_id)
{
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_id);
    return torch::from_blob(reinterpret_cast<void*>(addr), {size}, options);
}

//
// Register PyTorch ops
//

TORCH_LIBRARY(doca_rdma, m)
{
    m.def(
        "write_with_imm(int qp_ptr, int local_addr, int local_size, int lkey, int remote_addr, int rkey, int "
        "imm_value, Tensor(a!) wqe_idx_out) -> ()");
    // NOTE: write_with_imm_buf_arr is disabled due to ABI incompatibility between
    // libdoca_gpunetio_device.a (CUDA ABI v8) and PyTorch (forces ABI v7 via compute_80)
    m.def("wait_send(int qp_ptr, Tensor buf, Tensor(a!) wqe_idx) -> ()");
    m.def("poll_recv(int qp_ptr, Tensor(a!) cq_idx, Tensor(b!) imm_out) -> Tensor");
    m.def("wait_recv(int qp_ptr, Tensor(a!) cq_idx, Tensor(b!) imm_out, Tensor(c!) recv_buf) -> ()");
    m.def("post_recv_wrs(int qp_ptr, Tensor(a!) recv_wq_idx, int num_wrs) -> ()");
    m.def(
        "fused_send_recv(Tensor input, Tensor(a!) send_buf, Tensor(b!) recv_buf, int qp_ptr, int local_addr, int "
        "local_size, int lkey, int remote_addr, int rkey, int imm_value, Tensor(c!) wqe_idx_out, Tensor(d!) "
        "recv_wq_idx, Tensor(e!) cq_idx, Tensor(f!) imm_out, int num_wrs) -> ()");
    m.def("wrap_gpu_memory(int addr, int size, int device_id) -> Tensor");
}

TORCH_LIBRARY_IMPL(doca_rdma, CUDA, m)
{
    m.impl("write_with_imm", rdma_write_with_imm);
    m.impl("wait_send", wait_send_completion_v2);
    m.impl("poll_recv", poll_recv_cq);
    m.impl("wait_recv", wait_recv_completion);
    m.impl("post_recv_wrs", post_recv_wrs);
    m.impl("fused_send_recv", fused_send_recv);
}

// Register wrap_gpu_memory as CompositeExplicitAutograd (works without tensor args)
TORCH_LIBRARY_IMPL(doca_rdma, CompositeExplicitAutograd, m)
{
    m.impl("wrap_gpu_memory", wrap_gpu_memory);
}
