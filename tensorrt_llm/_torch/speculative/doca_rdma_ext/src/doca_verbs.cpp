/*
 * DOCA Verbs Wrapper Implementation
 */

#include "doca_verbs.h"
#include <cstdlib>
#include <cstring>
#include <endian.h>

DOCA_LOG_REGISTER(DOCA_VERBS);

#define MAX_SEND_SGES 1
#define MAX_RECV_SGES 1
#define DBR_SIZE 8

#define ROUND_UP(size, align) (((size) + (align) -1) & (~((align) -1)))

namespace doca_verbs
{

static size_t get_page_size()
{
    long ret = sysconf(_SC_PAGESIZE);
    return (ret == -1) ? 4096 : static_cast<size_t>(ret);
}

static uint32_t align_up(uint32_t value, uint32_t alignment)
{
    uint64_t remainder = value % alignment;
    if (remainder == 0)
        return value;
    return static_cast<uint32_t>(value + (alignment - remainder));
}

static uint32_t calc_cq_umem_size(uint32_t ncqe)
{
    uint32_t cqe_buf_size = ncqe * sizeof(struct mlx5_cqe64);
    return align_up(cqe_buf_size + DBR_SIZE, get_page_size());
}

static uint32_t calc_qp_umem_size(uint32_t rq_nwqe, uint32_t sq_nwqe)
{
    uint32_t rq_size = (rq_nwqe != 0) ? rq_nwqe * sizeof(struct mlx5_wqe_data_seg) : 0;
    uint32_t sq_size = (sq_nwqe != 0) ? sq_nwqe * sizeof(doca_gpu_dev_verbs_wqe) : 0;
    return align_up(rq_size + sq_size, get_page_size());
}

static void init_cqes(struct mlx5_cqe64* cqes, uint32_t ncqe)
{
    for (uint32_t i = 0; i < ncqe; i++)
    {
        cqes[i].op_own = (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) | MLX5_CQE_OWNER_MASK;
    }
}

// ============================================================================
// Cq Implementation
// ============================================================================

Cq::Cq(doca_gpu* gpu_dev, doca_dev* dev, doca_verbs_context* verbs_ctx, doca_verbs_pd* verbs_pd, uint16_t ncqe)
    : gpu_dev_(gpu_dev)
    , dev_(dev)
    , verbs_ctx_(verbs_ctx)
    , verbs_pd_(verbs_pd)
    , ncqe_(ncqe)
{
    cq_verbs_ = create_cq();
}

Cq::~Cq()
{
    destroy_cq();
}

doca_verbs_cq* Cq::create_cq()
{
    doca_error_t status;
    doca_verbs_cq_attr* attr = nullptr;
    doca_verbs_cq* cq = nullptr;
    uint32_t umem_size = calc_cq_umem_size(ncqe_);

    status = doca_verbs_cq_attr_create(&attr);
    if (status != DOCA_SUCCESS)
    {
        throw std::runtime_error("Failed to create CQ attributes");
    }

    status = doca_verbs_cq_attr_set_external_datapath_en(attr, 1);
    if (status != DOCA_SUCCESS)
    {
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to enable external datapath for CQ");
    }

    // Allocate CQ buffer in GPU memory
    status
        = doca_gpu_mem_alloc(gpu_dev_, umem_size, get_page_size(), DOCA_GPU_MEM_TYPE_GPU, &cq_umem_gpu_ptr_, nullptr);
    if (status != DOCA_SUCCESS)
    {
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to allocate GPU memory for CQ");
    }

    // Initialize CQEs on host then copy to GPU
    struct mlx5_cqe64* cq_host = static_cast<struct mlx5_cqe64*>(calloc(umem_size, 1));
    if (!cq_host)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to allocate host memory for CQ init");
    }
    init_cqes(cq_host, ncqe_);

    cudaError_t cuda_err = cudaMemcpy(cq_umem_gpu_ptr_, cq_host, umem_size, cudaMemcpyDefault);
    free(cq_host);
    if (cuda_err != cudaSuccess)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to copy CQ init data to GPU");
    }

    // Register GPU memory as umem
    status = doca_umem_gpu_create(gpu_dev_, dev_, cq_umem_gpu_ptr_, umem_size,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
            | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
        &cq_umem_);
    if (status != DOCA_SUCCESS)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to create GPU umem for CQ");
    }

    status = doca_verbs_cq_attr_set_external_umem(attr, cq_umem_, 0);
    if (status != DOCA_SUCCESS)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to set CQ external umem");
    }

    status = doca_verbs_cq_attr_set_cq_size(attr, ncqe_);
    if (status != DOCA_SUCCESS)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to set CQ size");
    }

    status = doca_verbs_cq_attr_set_cq_overrun(attr, 1);
    if (status != DOCA_SUCCESS)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to set CQ overrun");
    }

    // Create UAR for doorbell
    status = doca_uar_create(dev_, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &external_uar_);
    if (status != DOCA_SUCCESS)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to create UAR for CQ");
    }

    status = doca_verbs_cq_attr_set_external_uar(attr, external_uar_);
    if (status != DOCA_SUCCESS)
    {
        destroy_cq();
        doca_verbs_cq_attr_destroy(attr);
        throw std::runtime_error("Failed to set CQ external UAR");
    }

    status = doca_verbs_cq_create(verbs_ctx_, attr, &cq);
    doca_verbs_cq_attr_destroy(attr);
    if (status != DOCA_SUCCESS)
    {
        destroy_cq();
        throw std::runtime_error("Failed to create DOCA verbs CQ");
    }

    return cq;
}

void Cq::destroy_cq()
{
    if (cq_verbs_)
    {
        doca_verbs_cq_destroy(cq_verbs_);
        cq_verbs_ = nullptr;
    }
    if (external_uar_)
    {
        doca_uar_destroy(external_uar_);
        external_uar_ = nullptr;
    }
    if (cq_umem_)
    {
        doca_umem_destroy(cq_umem_);
        cq_umem_ = nullptr;
    }
    if (cq_umem_gpu_ptr_)
    {
        doca_gpu_mem_free(gpu_dev_, cq_umem_gpu_ptr_);
        cq_umem_gpu_ptr_ = nullptr;
    }
}

// ============================================================================
// Qp Implementation
// ============================================================================

Qp::Qp(doca_gpu* gpu_dev, doca_dev* dev, doca_verbs_context* verbs_ctx, doca_verbs_pd* verbs_pd, uint16_t sq_nwqe,
    uint16_t rq_nwqe, doca_gpu_dev_verbs_nic_handler nic_handler)
    : gpu_dev_(gpu_dev)
    , dev_(dev)
    , verbs_ctx_(verbs_ctx)
    , verbs_pd_(verbs_pd)
    , sq_nwqe_(sq_nwqe)
    , rq_nwqe_(rq_nwqe)
    , nic_handler_(nic_handler)
{

    // Create CQs first
    cq_sq_ = std::make_unique<Cq>(gpu_dev, dev, verbs_ctx, verbs_pd, sq_nwqe);
    cq_rq_ = std::make_unique<Cq>(gpu_dev, dev, verbs_ctx, verbs_pd, rq_nwqe);

    // Create QP
    qp_verbs_ = create_qp();

    // Export QP to GPU
    doca_error_t status = doca_gpu_verbs_export_qp(
        gpu_dev_, dev_, qp_verbs_, nic_handler_, qp_umem_gpu_ptr_, cq_sq_->get_cq(), cq_rq_->get_cq(), &qp_gverbs_);
    if (status != DOCA_SUCCESS)
    {
        throw std::runtime_error("Failed to export QP to GPU");
    }

    status = doca_gpu_verbs_get_qp_dev(qp_gverbs_, &qp_gdev_verbs_);
    if (status != DOCA_SUCCESS)
    {
        throw std::runtime_error("Failed to get GPU device QP handle");
    }

    DOCA_LOG_DBG("Created GPU-aware QP with device handle %p", (void*) qp_gdev_verbs_);
}

Qp::~Qp()
{
    if (qp_gverbs_)
    {
        doca_gpu_verbs_unexport_qp(gpu_dev_, qp_gverbs_);
        qp_gverbs_ = nullptr;
    }
    destroy_qp();
}

doca_verbs_qp* Qp::create_qp()
{
    doca_error_t status;
    doca_verbs_qp_init_attr* attr = nullptr;
    doca_verbs_qp* qp = nullptr;
    uint32_t umem_size = calc_qp_umem_size(rq_nwqe_, sq_nwqe_);
    size_t dbr_size = ROUND_UP(DBR_SIZE, get_page_size());

    status = doca_verbs_qp_init_attr_create(&attr);
    if (status != DOCA_SUCCESS)
    {
        throw std::runtime_error("Failed to create QP init attributes");
    }

    status = doca_verbs_qp_init_attr_set_external_datapath_en(attr, 1);
    if (status != DOCA_SUCCESS)
    {
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to enable external datapath for QP");
    }

    // Create UAR
    status = doca_uar_create(dev_, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &external_uar_);
    if (status != DOCA_SUCCESS)
    {
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to create UAR for QP");
    }

    status = doca_verbs_qp_init_attr_set_external_uar(attr, external_uar_);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP external UAR");
    }

    // Allocate WQ buffer in GPU memory
    status
        = doca_gpu_mem_alloc(gpu_dev_, umem_size, get_page_size(), DOCA_GPU_MEM_TYPE_GPU, &qp_umem_gpu_ptr_, nullptr);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to allocate GPU memory for QP WQ");
    }

    status = doca_umem_gpu_create(gpu_dev_, dev_, qp_umem_gpu_ptr_, umem_size,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
            | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
        &qp_umem_);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to create GPU umem for QP WQ");
    }

    status = doca_verbs_qp_init_attr_set_external_umem(attr, qp_umem_, 0);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP external umem");
    }

    // Allocate DBR in GPU memory
    status = doca_gpu_mem_alloc(
        gpu_dev_, dbr_size, get_page_size(), DOCA_GPU_MEM_TYPE_GPU, &qp_umem_dbr_gpu_ptr_, nullptr);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to allocate GPU memory for QP DBR");
    }

    status = doca_umem_gpu_create(gpu_dev_, dev_, qp_umem_dbr_gpu_ptr_, dbr_size,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
            | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
        &qp_umem_dbr_);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to create GPU umem for QP DBR");
    }

    status = doca_verbs_qp_init_attr_set_external_dbr_umem(attr, qp_umem_dbr_, 0);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP external DBR umem");
    }

    // Set QP parameters
    status = doca_verbs_qp_init_attr_set_pd(attr, verbs_pd_);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP PD");
    }

    status = doca_verbs_qp_init_attr_set_sq_wr(attr, sq_nwqe_);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP SQ size");
    }

    status = doca_verbs_qp_init_attr_set_rq_wr(attr, rq_nwqe_);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP RQ size");
    }

    status = doca_verbs_qp_init_attr_set_qp_type(attr, DOCA_VERBS_QP_TYPE_RC);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP type");
    }

    status = doca_verbs_qp_init_attr_set_send_cq(attr, cq_sq_->get_cq());
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP send CQ");
    }

    status = doca_verbs_qp_init_attr_set_receive_cq(attr, cq_rq_->get_cq());
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP recv CQ");
    }

    status = doca_verbs_qp_init_attr_set_send_max_sges(attr, MAX_SEND_SGES);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP send max SGEs");
    }

    status = doca_verbs_qp_init_attr_set_receive_max_sges(attr, MAX_RECV_SGES);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        doca_verbs_qp_init_attr_destroy(attr);
        throw std::runtime_error("Failed to set QP recv max SGEs");
    }

    status = doca_verbs_qp_create(verbs_ctx_, attr, &qp);
    doca_verbs_qp_init_attr_destroy(attr);
    if (status != DOCA_SUCCESS)
    {
        destroy_qp();
        throw std::runtime_error("Failed to create DOCA verbs QP");
    }

    return qp;
}

void Qp::destroy_qp()
{
    if (qp_verbs_)
    {
        doca_verbs_qp_destroy(qp_verbs_);
        qp_verbs_ = nullptr;
    }
    if (external_uar_)
    {
        doca_uar_destroy(external_uar_);
        external_uar_ = nullptr;
    }
    if (qp_umem_)
    {
        doca_umem_destroy(qp_umem_);
        qp_umem_ = nullptr;
    }
    if (qp_umem_gpu_ptr_)
    {
        doca_gpu_mem_free(gpu_dev_, qp_umem_gpu_ptr_);
        qp_umem_gpu_ptr_ = nullptr;
    }
    if (qp_umem_dbr_)
    {
        doca_umem_destroy(qp_umem_dbr_);
        qp_umem_dbr_ = nullptr;
    }
    if (qp_umem_dbr_gpu_ptr_)
    {
        doca_gpu_mem_free(gpu_dev_, qp_umem_dbr_gpu_ptr_);
        qp_umem_dbr_gpu_ptr_ = nullptr;
    }
}

uint32_t Qp::get_qpn() const
{
    return doca_verbs_qp_get_qpn(qp_verbs_);
}

// ============================================================================
// Mr Implementation
// ============================================================================

Mr::Mr(doca_gpu* gpu_dev, void* addr, size_t size, ibv_pd* pd)
    : gpu_dev_(gpu_dev)
    , addr_(addr)
    , size_(size)
    , pd_(pd)
    , remote_(false)
{

    if (!gpu_dev || !addr || size == 0 || !pd)
    {
        throw std::invalid_argument("Invalid MR parameters");
    }

    static size_t page_size = get_page_size();
    doca_error_t status;

    // Check alignment requirements
    if ((size % page_size) != 0)
    {
        throw std::runtime_error("MR size must be page-aligned for dmabuf registration");
    }
    if ((reinterpret_cast<uintptr_t>(addr) % page_size) != 0)
    {
        throw std::runtime_error("MR address must be page-aligned for dmabuf registration");
    }

    // Get dmabuf fd from DOCA
    status = doca_gpu_dmabuf_fd(gpu_dev, addr, size, &dmabuf_fd_);
    if (status != DOCA_SUCCESS)
    {
        throw std::runtime_error(
            "doca_gpu_dmabuf_fd failed - GPU dmabuf export not supported. "
            "Ensure CUDA 12.8+, dmabuf-capable kernel, and proper GPU driver.");
    }

    // Register with libibverbs
    ibmr_ = ibv_reg_dmabuf_mr(pd, 0, size, reinterpret_cast<uint64_t>(addr), dmabuf_fd_,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
    if (!ibmr_)
    {
        close(dmabuf_fd_);
        dmabuf_fd_ = -1;
        throw std::runtime_error("ibv_reg_dmabuf_mr failed - check kernel dmabuf/RDMA support");
    }

    // Store keys in big-endian for MLX5 WQEs
    lkey_ = htobe32(ibmr_->lkey);
    rkey_ = htobe32(ibmr_->rkey);

    DOCA_LOG_DBG("Registered MR: addr=%p size=%zu lkey=0x%x rkey=0x%x (BE: 0x%x 0x%x)", addr, size, ibmr_->lkey,
        ibmr_->rkey, lkey_, rkey_);
}

Mr::Mr(void* addr, size_t size, uint32_t rkey)
    : addr_(addr)
    , size_(size)
    , rkey_(rkey)
    , remote_(true)
{
    // Remote MR - no local registration needed
}

Mr::~Mr()
{
    if (!remote_ && ibmr_)
    {
        ibv_dereg_mr(ibmr_);
        ibmr_ = nullptr;
    }
    if (dmabuf_fd_ >= 0)
    {
        close(dmabuf_fd_);
        dmabuf_fd_ = -1;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

doca_verbs_context* open_verbs_context(char const* ib_dev_name)
{
    int num_devices = 0;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);

    if (!dev_list || num_devices == 0)
    {
        throw std::runtime_error("No IB devices found");
    }

    ibv_device* target_dev = nullptr;
    for (int i = 0; i < num_devices; i++)
    {
        if (strcmp(ibv_get_device_name(dev_list[i]), ib_dev_name) == 0)
        {
            target_dev = dev_list[i];
            break;
        }
    }

    if (!target_dev)
    {
        ibv_free_device_list(dev_list);
        throw std::runtime_error(std::string("IB device not found: ") + ib_dev_name);
    }

    doca_verbs_context* ctx = nullptr;
    doca_error_t status
        = doca_verbs_bridge_verbs_context_create(target_dev, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &ctx);

    ibv_free_device_list(dev_list);

    if (status != DOCA_SUCCESS)
    {
        throw std::runtime_error("Failed to create DOCA verbs context");
    }

    return ctx;
}

// QP state transition helpers removed - endpoint has its own connect_qp logic

} // namespace doca_verbs
