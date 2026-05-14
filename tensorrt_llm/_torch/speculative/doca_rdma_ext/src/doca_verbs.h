/*
 * DOCA Verbs Wrapper Classes for GPU-initiated RDMA
 *
 * Provides GPU-aware QP/CQ/MR infrastructure using DOCA verbs APIs.
 * Based on NIXL's approach for ABI-safe GPU RDMA.
 */

#ifndef DOCA_VERBS_H
#define DOCA_VERBS_H

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unistd.h>

#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_verbs_def.h>
#include <doca_log.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

namespace doca_verbs
{

// CQ with GPU-allocated umem for GPU-initiated polling
class Cq
{
public:
    Cq(doca_gpu* gpu_dev, doca_dev* dev, doca_verbs_context* verbs_ctx, doca_verbs_pd* verbs_pd, uint16_t ncqe);
    ~Cq();

    // Non-copyable
    Cq(Cq const&) = delete;
    Cq& operator=(Cq const&) = delete;

    doca_verbs_cq* get_cq() const
    {
        return cq_verbs_;
    }

private:
    doca_verbs_cq* create_cq();
    void destroy_cq();

    doca_gpu* gpu_dev_;
    doca_dev* dev_;
    doca_verbs_context* verbs_ctx_;
    doca_verbs_pd* verbs_pd_;
    uint16_t ncqe_;

    doca_verbs_cq* cq_verbs_ = nullptr;
    void* cq_umem_gpu_ptr_ = nullptr;
    doca_umem* cq_umem_ = nullptr;
    doca_uar* external_uar_ = nullptr;
};

// QP with GPU-allocated umem + GPU export for kernel access
class Qp
{
public:
    Qp(doca_gpu* gpu_dev, doca_dev* dev, doca_verbs_context* verbs_ctx, doca_verbs_pd* verbs_pd, uint16_t sq_nwqe,
        uint16_t rq_nwqe, doca_gpu_dev_verbs_nic_handler nic_handler);
    ~Qp();

    // Non-copyable
    Qp(Qp const&) = delete;
    Qp& operator=(Qp const&) = delete;

    // Host-side QP for connection setup
    doca_verbs_qp* get_qp() const
    {
        return qp_verbs_;
    }

    // GPU-side QP for kernel access
    doca_gpu_verbs_qp* get_qp_gpu() const
    {
        return qp_gverbs_;
    }

    doca_gpu_dev_verbs_qp* get_qp_gpu_dev() const
    {
        return qp_gdev_verbs_;
    }

    // Get QP number for connection handshake
    uint32_t get_qpn() const;

private:
    doca_verbs_qp* create_qp();
    void destroy_qp();

    doca_gpu* gpu_dev_;
    doca_dev* dev_;
    doca_verbs_context* verbs_ctx_;
    doca_verbs_pd* verbs_pd_;
    uint16_t sq_nwqe_;
    uint16_t rq_nwqe_;
    doca_gpu_dev_verbs_nic_handler nic_handler_;

    doca_verbs_qp* qp_verbs_ = nullptr;
    void* qp_umem_gpu_ptr_ = nullptr;
    doca_umem* qp_umem_ = nullptr;
    void* qp_umem_dbr_gpu_ptr_ = nullptr;
    doca_umem* qp_umem_dbr_ = nullptr;
    doca_uar* external_uar_ = nullptr;
    doca_gpu_verbs_qp* qp_gverbs_ = nullptr;
    doca_gpu_dev_verbs_qp* qp_gdev_verbs_ = nullptr;

    std::unique_ptr<Cq> cq_sq_;
    std::unique_ptr<Cq> cq_rq_;
};

// MR using dmabuf (no fallback - fail with clear error if dmabuf not supported)
class Mr
{
public:
    // Register local GPU memory
    Mr(doca_gpu* gpu_dev, void* addr, size_t size, ibv_pd* pd);

    // Create remote MR reference (no registration, just stores rkey)
    Mr(void* addr, size_t size, uint32_t rkey);

    ~Mr();

    // Non-copyable
    Mr(Mr const&) = delete;
    Mr& operator=(Mr const&) = delete;

    ibv_mr* get_mr() const
    {
        return ibmr_;
    }

    // Keys in big-endian format for MLX5 WQEs
    uint32_t get_lkey() const
    {
        return lkey_;
    }

    uint32_t get_rkey() const
    {
        return rkey_;
    }

    void* get_addr() const
    {
        return addr_;
    }

    size_t get_size() const
    {
        return size_;
    }

    bool is_remote() const
    {
        return remote_;
    }

private:
    doca_gpu* gpu_dev_ = nullptr;
    void* addr_ = nullptr;
    size_t size_ = 0;
    ibv_pd* pd_ = nullptr;
    ibv_mr* ibmr_ = nullptr;
    uint32_t lkey_ = 0;
    uint32_t rkey_ = 0;
    bool remote_ = false;
    int dmabuf_fd_ = -1;
};

// Helper to open DOCA verbs context from IB device name
doca_verbs_context* open_verbs_context(char const* ib_dev_name);

} // namespace doca_verbs

#endif // DOCA_VERBS_H
