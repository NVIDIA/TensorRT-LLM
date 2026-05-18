/*
 * CUDA Kernels for GPU-initiated RDMA operations using DOCA GPUNetIO
 *
 * These kernels enable RDMA WRITE_WITH_IMM operations directly from GPU,
 * allowing CUDA graph capture and avoiding host-side synchronization.
 */

// DOCA GPUNetIO device-side headers (must be first, like NIXL)
#include <cuda.h>
#include <cuda/atomic>
#include <doca_gpunetio_dev_verbs_onesided.cuh>
#include <doca_gpunetio_dev_verbs_twosided.cuh>

#include <cstdint>

// MLX5 opcodes
#ifndef MLX5_OPCODE_RDMA_WRITE
#define MLX5_OPCODE_RDMA_WRITE 0x08
#endif

#ifndef MLX5_OPCODE_RDMA_WRITE_IMM
#define MLX5_OPCODE_RDMA_WRITE_IMM 0x09
#endif

// Use system mlx5_cqe64 definition
#include <infiniband/mlx5dv.h>

__device__ __forceinline__ uint32_t device_ntohl(uint32_t x)
{
    return ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8) | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24);
}

/*
 * GPU-initiated RDMA WRITE WITH IMMEDIATE
 *
 * Posts an RDMA WRITE_WITH_IMM operation from the GPU. The immediate value
 * will be delivered to the receiver's completion queue, serving as a
 * notification mechanism (no separate signal buffer needed).
 *
 * @param qp            Device-accessible QP pointer
 * @param local_buf     Local GPU buffer to send
 * @param lkey          Local memory key
 * @param remote_addr   Remote address to write to
 * @param rkey          Remote memory key
 * @param size          Size of data to write
 * @param imm_value     32-bit immediate value (sequence number)
 * @param wqe_idx_out   Output: WQE index for completion tracking
 */
extern "C" __global__ void rdma_write_with_imm_kernel(doca_gpu_dev_verbs_qp* qp, void* local_buf, uint32_t lkey,
    uint64_t remote_addr, uint32_t rkey, size_t size, uint32_t imm_value, uint64_t* wqe_idx_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Reserve WQE slot
    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, 1);

    // Get WQE pointer
    doca_gpu_dev_verbs_wqe* wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    // Prepare RDMA WRITE WITH IMMEDIATE
    // The function signature based on NIXL code:
    // doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe, wqe_idx, opcode, ctrl_flags, imm,
    //                                      remote_addr, rkey, local_addr, lkey, size)
    doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe_ptr, wqe_idx,
        MLX5_OPCODE_RDMA_WRITE_IMM,            // RDMA WRITE with immediate
        DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, // Request completion
        imm_value,                             // 32-bit immediate value
        remote_addr, rkey, (uint64_t) local_buf, lkey, size);

    // Mark WQE ready and submit
    doca_gpu_dev_verbs_mark_wqes_ready(qp, wqe_idx, wqe_idx);
    doca_gpu_dev_verbs_submit(qp, wqe_idx + 1);

    // Output WQE index for completion tracking
    if (wqe_idx_out)
    {
        *wqe_idx_out = wqe_idx;
    }
}

/*
 * Poll send completion queue
 *
 * Polls the send CQ for a specific WQE completion.
 *
 * @param cq              Device-accessible CQ pointer
 * @param expected_wqe_idx WQE index to wait for
 * @param completed       Output: 1 if completed, 0 otherwise
 */
extern "C" __global__ void poll_send_cq_kernel(doca_gpu_dev_verbs_cq* cq, uint64_t expected_wqe_idx, int* completed)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Poll CQ at expected index
    int status = doca_gpu_dev_verbs_poll_cq_at(cq, expected_wqe_idx);

    *completed = (status == 0) ? 1 : 0;
}

/*
 * Poll receive completion queue for incoming WRITE_WITH_IMM
 *
 * Polls the receive CQ for incoming RDMA WRITE_WITH_IMM completions.
 * When a WRITE_WITH_IMM arrives, a receive request is consumed and
 * a CQE is generated with the immediate value.
 *
 * @param cq              Device-accessible receive CQ pointer
 * @param cq_idx          Input/output: current CQ consumer index
 * @param imm_value_out   Output: immediate value from CQE (sequence number)
 * @param completed       Output: 1 if completion found, 0 otherwise
 */
extern "C" __global__ void poll_recv_cq_for_imm_kernel(
    doca_gpu_dev_verbs_cq* cq, uint64_t* cq_idx, uint32_t* imm_value_out, int* completed)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    uint64_t idx = *cq_idx;

    // Poll CQ at current index
    int status = doca_gpu_dev_verbs_poll_cq_at(cq, idx);

    if (status == 0)
    {
        // Completion found - extract immediate value
        // The immediate value is in the CQE at this index
        // Access CQE to get imm_data field
        uint8_t* cqe_base = (uint8_t*) __ldg((uintptr_t*) &cq->cqe_daddr);
        uint32_t cqe_num = __ldg(&cq->cqe_num);
        uint32_t cqe_idx = idx & (cqe_num - 1);

        struct mlx5_cqe64* cqe = (struct mlx5_cqe64*) (cqe_base + cqe_idx * 64);
        uint32_t imm = __ldg(&cqe->imm_inval_pkey);

        *imm_value_out = device_ntohl(imm);
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

/*
 * Blocking poll for receive completion with immediate
 *
 * Spins until a receive completion arrives, then returns the immediate value.
 * Use with caution - can block indefinitely if no completion arrives.
 *
 * @param cq              Device-accessible receive CQ pointer
 * @param cq_idx          Input/output: current CQ consumer index
 * @param imm_value_out   Output: immediate value from CQE
 */
extern "C" __global__ void wait_recv_imm_kernel(doca_gpu_dev_verbs_cq* cq, uint64_t* cq_idx, uint32_t* imm_value_out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    uint64_t idx = *cq_idx;

    // Spin until completion
    while (doca_gpu_dev_verbs_poll_cq_at(cq, idx) != 0)
    {
        __threadfence_system();
    }

    // Extract immediate value
    uint8_t* cqe_base = (uint8_t*) __ldg((uintptr_t*) &cq->cqe_daddr);
    uint32_t cqe_num = __ldg(&cq->cqe_num);
    uint32_t cqe_idx = idx & (cqe_num - 1);

    struct mlx5_cqe64* cqe = (struct mlx5_cqe64*) (cqe_base + cqe_idx * 64);
    uint32_t imm = __ldg(&cqe->imm_inval_pkey);

    *imm_value_out = device_ntohl(imm);
    *cq_idx = idx + 1;

    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&cq->cqe_ci, idx + 1);
}

/*
 * Combined write-and-wait kernel for request/response pattern
 *
 * Performs RDMA WRITE_WITH_IMM to send request, then waits for response
 * via incoming WRITE_WITH_IMM. Suitable for synchronous request/response.
 *
 * @param qp              Send QP
 * @param send_cq         Send CQ
 * @param recv_cq         Receive CQ
 * @param local_send_buf  Local buffer with request data
 * @param send_lkey       Local key for send buffer
 * @param remote_recv_addr Remote address to write request to
 * @param remote_recv_rkey Remote key for recv buffer
 * @param send_size       Size of request data
 * @param seq_num         Sequence number for this request
 * @param recv_cq_idx     Receive CQ consumer index (input/output)
 * @param response_imm    Output: immediate value from response
 */
extern "C" __global__ void write_and_wait_kernel(doca_gpu_dev_verbs_qp* qp, doca_gpu_dev_verbs_cq* send_cq,
    doca_gpu_dev_verbs_cq* recv_cq, void* local_send_buf, uint32_t send_lkey, uint64_t remote_recv_addr,
    uint32_t remote_recv_rkey, size_t send_size, uint32_t seq_num, uint64_t* recv_cq_idx, uint32_t* response_imm)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // Step 1: Post RDMA WRITE_WITH_IMM
    uint64_t wqe_idx = doca_gpu_dev_verbs_reserve_wq_slots(qp, 1);
    doca_gpu_dev_verbs_wqe* wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);

    doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe_ptr, wqe_idx, MLX5_OPCODE_RDMA_WRITE_IMM,
        DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE, seq_num, remote_recv_addr, remote_recv_rkey, (uint64_t) local_send_buf,
        send_lkey, send_size);

    doca_gpu_dev_verbs_mark_wqes_ready(qp, wqe_idx, wqe_idx);
    doca_gpu_dev_verbs_submit(qp, wqe_idx + 1);

    // Step 2: Wait for send completion
    while (doca_gpu_dev_verbs_poll_cq_at(send_cq, wqe_idx) != 0)
    {
        __threadfence_system();
    }

    // Step 3: Wait for response (incoming WRITE_WITH_IMM)
    uint64_t idx = *recv_cq_idx;
    while (doca_gpu_dev_verbs_poll_cq_at(recv_cq, idx) != 0)
    {
        __threadfence_system();
    }

    // Extract response immediate value
    uint8_t* cqe_base = (uint8_t*) __ldg((uintptr_t*) &recv_cq->cqe_daddr);
    uint32_t cqe_num = __ldg(&recv_cq->cqe_num);
    uint32_t cqe_idx = idx & (cqe_num - 1);

    struct mlx5_cqe64* cqe = (struct mlx5_cqe64*) (cqe_base + cqe_idx * 64);
    *response_imm = device_ntohl(__ldg(&cqe->imm_inval_pkey));
    *recv_cq_idx = idx + 1;

    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&recv_cq->cqe_ci, idx + 1);
}
