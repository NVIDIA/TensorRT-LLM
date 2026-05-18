/*
 * DOCA RDMA Endpoint Implementation
 *
 * Simplified wrapper around DOCA GPUNetIO for GPU-initiated RDMA operations.
 * Supports RDMA_WRITE_WITH_IMM for efficient signaling.
 */

#include "doca_rdma_endpoint.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <endian.h>
#include <mutex>
#include <netdb.h>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_buf_array.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_gpunetio.h>
#include <doca_gpunetio_verbs_def.h>
#include <doca_log.h>
#include <doca_mmap.h>
#include <doca_rdma_bridge.h>
#include <doca_uar.h>
#include <doca_umem.h>
#include <doca_verbs.h>
#include <doca_verbs_bridge.h>

#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

DOCA_LOG_REGISTER(DOCA_RDMA_ENDPOINT);

#define CHECK_DOCA(call, msg)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        doca_error_t _err = (call);                                                                                    \
        if (_err != DOCA_SUCCESS)                                                                                      \
        {                                                                                                              \
            DOCA_LOG_ERR("%s: %s", msg, doca_error_get_descr(_err));                                                   \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDA(call, msg)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t _err = (call);                                                                                     \
        if (_err != cudaSuccess)                                                                                       \
        {                                                                                                              \
            DOCA_LOG_ERR("%s: %s", msg, cudaGetErrorString(_err));                                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

static const uint32_t SQ_SIZE = 64;
static const uint32_t RQ_SIZE = 64;
static const uint32_t DBR_SIZE = 8;

static size_t get_page_size()
{
    long ret = sysconf(_SC_PAGESIZE);
    return (ret == -1) ? 4096 : (size_t) ret;
}

static uint32_t align_up(uint32_t value, uint32_t align)
{
    return (value + align - 1) & ~(align - 1);
}

static uint32_t calc_cq_umem_size(uint32_t ncqe)
{
    uint32_t cqe_buf_size = ncqe * sizeof(struct mlx5_cqe64);
    return align_up(cqe_buf_size + DBR_SIZE, get_page_size());
}

static uint32_t calc_qp_umem_size(uint32_t sq_size, uint32_t rq_size)
{
    uint32_t sq_ring = sq_size * sizeof(doca_gpu_dev_verbs_wqe);
    uint32_t rq_ring = rq_size * sizeof(struct mlx5_wqe_data_seg);
    return align_up(sq_ring + rq_ring, get_page_size());
}

static void init_cqes(struct mlx5_cqe64* cqes, uint32_t ncqe)
{
    for (uint32_t i = 0; i < ncqe; i++)
    {
        cqes[i].op_own = (MLX5_CQE_INVALID << DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT) | MLX5_CQE_OWNER_MASK;
    }
}

static doca_mtu_size mtu_from_value(uint32_t mtu_bytes)
{
    switch (mtu_bytes)
    {
    case 256: return DOCA_MTU_SIZE_256_BYTES;
    case 512: return DOCA_MTU_SIZE_512_BYTES;
    case 1024: return DOCA_MTU_SIZE_1K_BYTES;
    case 2048: return DOCA_MTU_SIZE_2K_BYTES;
    case 4096: return DOCA_MTU_SIZE_4K_BYTES;
    default:
        throw std::runtime_error("Invalid GLUE_RDMA_PATH_MTU value: " + std::to_string(mtu_bytes)
            + ". Must be 256, 512, 1024, 2048, or 4096");
    }
}

static uint32_t read_env_uint32(char const* name, uint32_t default_val)
{
    char const* val = std::getenv(name);
    if (val == nullptr)
    {
        return default_val;
    }
    try
    {
        return static_cast<uint32_t>(std::stoul(val));
    }
    catch (...)
    {
        DOCA_LOG_WARN("Invalid value for %s: %s, using default %u", name, val, default_val);
        return default_val;
    }
}

static bool read_env_enabled(char const* name, bool default_val = false)
{
    char const* val = std::getenv(name);
    if (val == nullptr || *val == '\0')
    {
        return default_val;
    }

    std::string s(val);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static std::vector<uint32_t> load_nic_handler_candidates()
{
    // Default behavior is unchanged from existing code.
    std::vector<uint32_t> handlers{
        static_cast<uint32_t>(DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB),
    };

    // CPU-proxy and AUTO handlers are disabled by default for this endpoint,
    // because this code path relies on GPU-submitted verbs ops.
    // Opt-in only for targeted bring-up with explicit proxy progression.
    bool allow_proxy_handlers = read_env_enabled("GLUE_RDMA_ALLOW_PROXY_HANDLERS", false);

    // Optional override list, for example: GLUE_RDMA_NIC_HANDLERS=0,1,2
    // Parsed as numeric IDs to avoid depending on enum name differences across
    // DOCA SDK revisions.
    char const* raw = std::getenv("GLUE_RDMA_NIC_HANDLERS");
    if (raw == nullptr || *raw == '\0')
    {
        return handlers;
    }

    std::vector<uint32_t> parsed;
    bool filtered_proxy_handlers = false;
    std::stringstream ss(raw);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty())
        {
            continue;
        }
        try
        {
            uint32_t handler = static_cast<uint32_t>(std::stoul(token));
            if (!allow_proxy_handlers
                && (handler == static_cast<uint32_t>(DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO)
                    || handler == static_cast<uint32_t>(DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY)))
            {
                DOCA_LOG_WARN(
                    "Ignoring GLUE_RDMA_NIC_HANDLERS token %u; proxy handlers are disabled. "
                    "Set GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 to enable.",
                    handler);
                std::fprintf(stderr,
                    "[doca_rdma] Ignoring GLUE_RDMA_NIC_HANDLERS token %u; proxy handlers are disabled. "
                    "Set GLUE_RDMA_ALLOW_PROXY_HANDLERS=1 to enable.\n",
                    handler);
                filtered_proxy_handlers = true;
                continue;
            }
            parsed.push_back(handler);
        }
        catch (...)
        {
            DOCA_LOG_WARN("Ignoring invalid GLUE_RDMA_NIC_HANDLERS token: %s", token.c_str());
        }
    }

    if (parsed.empty() && filtered_proxy_handlers)
    {
        DOCA_LOG_WARN(
            "GLUE_RDMA_NIC_HANDLERS=%s produced no usable handlers after filtering; "
            "falling back to default GPU handler.",
            raw);
        std::fprintf(stderr,
            "[doca_rdma] GLUE_RDMA_NIC_HANDLERS=%s produced no usable handlers after filtering; "
            "falling back to default GPU handler.\n",
            raw);
    }

    if (!parsed.empty())
    {
        handlers = std::move(parsed);
    }

    return handlers;
}

static RdmaConfig load_rdma_config()
{
    RdmaConfig config{};

    config.path_mtu = read_env_uint32("GLUE_RDMA_PATH_MTU", 512);
    config.gid_index = read_env_uint32("GLUE_RDMA_GID_INDEX", 0);
    config.max_dest_rd_atomic = static_cast<uint8_t>(read_env_uint32("GLUE_RDMA_MAX_DEST_RD_ATOMIC", 16));
    config.min_rnr_timer = static_cast<uint8_t>(read_env_uint32("GLUE_RDMA_MIN_RNR_TIMER", 1));

    config.timeout = static_cast<uint8_t>(read_env_uint32("GLUE_RDMA_TIMEOUT", 14));
    config.retry_cnt = static_cast<uint8_t>(read_env_uint32("GLUE_RDMA_RETRY_CNT", 7));
    config.rnr_retry = static_cast<uint8_t>(read_env_uint32("GLUE_RDMA_RNR_RETRY", 1));
    config.max_rd_atomic = static_cast<uint8_t>(read_env_uint32("GLUE_RDMA_MAX_RD_ATOMIC", 16));

    DOCA_LOG_INFO(
        "RDMA config: path_mtu=%u, gid_index=%u, max_dest_rd_atomic=%u, min_rnr_timer=%u, "
        "timeout=%u, retry_cnt=%u, rnr_retry=%u, max_rd_atomic=%u",
        config.path_mtu, config.gid_index, config.max_dest_rd_atomic, config.min_rnr_timer, config.timeout,
        config.retry_cnt, config.rnr_retry, config.max_rd_atomic);

    return config;
}

struct DocaRdmaEndpoint::CqState
{
    doca_verbs_cq* cq = nullptr;
    void* umem_gpu_ptr = nullptr;
    doca_umem* umem = nullptr;
    doca_uar* uar = nullptr;
    doca_gpu_dev_verbs_cq* dev_cq = nullptr;
    uint32_t size = 0;
};

struct DocaRdmaEndpoint::QpState
{
    doca_verbs_qp* qp = nullptr;
    void* umem_gpu_ptr = nullptr;
    doca_umem* umem = nullptr;
    void* dbr_gpu_ptr = nullptr;
    void* dbr_host_alloc = nullptr; // CPU_PROXY path: backing calloc'd memory to free
    bool dbr_is_host = false;
    doca_umem* dbr_umem = nullptr;
    doca_uar* uar = nullptr;
    doca_gpu_verbs_qp* gpu_qp = nullptr;
    doca_gpu_dev_verbs_qp* dev_qp = nullptr;
    std::unique_ptr<CqState> send_cq;
    std::unique_ptr<CqState> recv_cq;
    uint32_t qpn = 0;
    uint32_t psn = 0;
};

struct DocaRdmaEndpoint::DocaState
{
    doca_dev* ddev = nullptr;
    doca_gpu* gpu = nullptr;
    doca_verbs_context* verbs_ctx = nullptr;
    doca_verbs_pd* verbs_pd = nullptr;
    doca_verbs_ah_attr* ah_attr = nullptr;
    ibv_context* ibv_ctx = nullptr;
    ibv_pd* pd = nullptr;
    ibv_port_attr port_attr;

    uint8_t gid[16];
    uint16_t lid;
    int gid_index = 0;

    std::vector<std::unique_ptr<QpState>> peer_qps;

    struct MemReg
    {
        void* addr;
        size_t len;
        ibv_mr* mr;
        uint32_t lkey;
        uint32_t rkey;
    };

    std::vector<MemReg> mem_regs;

    // Track different registration approaches for cleanup and debugging
    struct DataUmem
    {
        void* addr;
        size_t len;
        doca_umem* umem;
        ibv_mr* mr;
        char const* approach;
    };

    std::vector<DataUmem> data_umems;

    struct DataMmap
    {
        void* addr;
        size_t len;
        doca_mmap* mmap;
        ibv_mr* mr;
        char const* approach;
    };

    std::vector<DataMmap> data_mmaps;

    struct DocaBuffer
    {
        void* addr;
        size_t len;
        ibv_mr* mr;
        char const* approach;
    };

    std::vector<DocaBuffer> doca_buffers;

    // New: Buffer array registration using doca_mmap + doca_buf_arr
    struct BufArrReg
    {
        void* user_ptr; // Original PyTorch pointer
        size_t length;
        doca_mmap* mmap;
        doca_buf_arr* buf_arr;
        doca_gpu_buf_arr* gpu_buf_arr; // GPU-accessible handle
        int index;                     // Index in buf_arr_regs vector
    };

    std::vector<BufArrReg> buf_arr_regs;

    // Standalone objects kept alive for debugging
    std::vector<doca_umem*> standalone_umems;
    std::vector<doca_mmap*> standalone_mmaps;
};

DocaRdmaEndpoint::DocaRdmaEndpoint()
    : initialized_(false)
    , gpu_id_(0)
    , doca_state_(std::make_unique<DocaState>())
{
}

DocaRdmaEndpoint::~DocaRdmaEndpoint()
{
    cleanup();
}

bool DocaRdmaEndpoint::init(int gpu_id, std::string const& nic_name)
{
    if (initialized_)
    {
        DOCA_LOG_WARN("Already initialized");
        return true;
    }

    gpu_id_ = gpu_id;
    nic_name_ = nic_name;
    local_name_ = "gpu_" + std::to_string(gpu_id);
    rdma_config_ = load_rdma_config();

    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(gpu_id), "Failed to set CUDA device");
    CHECK_CUDA(cudaFree(0), "Failed to initialize CUDA context");

    if (!setup_doca_devices())
    {
        DOCA_LOG_ERR("Failed to setup DOCA devices");
        return false;
    }

    initialized_ = true;
    DOCA_LOG_INFO("DocaRdmaEndpoint initialized: %s", local_name_.c_str());
    return true;
}

bool DocaRdmaEndpoint::setup_doca_devices()
{
    doca_error_t result;

    {
        static std::once_flag log_backend_once;
        static doca_error_t log_backend_result = DOCA_SUCCESS;
        std::call_once(log_backend_once, []() { log_backend_result = doca_log_backend_create_standard(); });
        if (log_backend_result != DOCA_SUCCESS)
        {
            DOCA_LOG_WARN("Failed to create DOCA log backend (may already exist)");
        }
    }

    int num_devices;
    ibv_device** dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0)
    {
        DOCA_LOG_ERR("No IB devices found");
        return false;
    }

    ibv_device* ib_dev = nullptr;
    for (int i = 0; i < num_devices; i++)
    {
        if (nic_name_ == ibv_get_device_name(dev_list[i]))
        {
            ib_dev = dev_list[i];
            break;
        }
    }

    if (!ib_dev)
    {
        DOCA_LOG_ERR("IB device %s not found", nic_name_.c_str());
        ibv_free_device_list(dev_list);
        return false;
    }

    // Create DOCA verbs context from IB device
    result
        = doca_verbs_bridge_verbs_context_create(ib_dev, DOCA_VERBS_CONTEXT_CREATE_FLAGS_NONE, &doca_state_->verbs_ctx);
    ibv_free_device_list(dev_list);

    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create DOCA verbs context: %s", doca_error_get_descr(result));
        return false;
    }

    doca_state_->ibv_ctx = doca_verbs_bridge_get_ibv_ctx(doca_state_->verbs_ctx);
    if (!doca_state_->ibv_ctx)
    {
        DOCA_LOG_ERR("Failed to get ibv_context from DOCA verbs context");
        return false;
    }

    // Create DOCA verbs PD
    result = doca_verbs_pd_create(doca_state_->verbs_ctx, &doca_state_->verbs_pd);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create DOCA verbs PD: %s", doca_error_get_descr(result));
        return false;
    }

    doca_state_->pd = doca_verbs_bridge_verbs_pd_get_ibv_pd(doca_state_->verbs_pd);
    if (!doca_state_->pd)
    {
        DOCA_LOG_ERR("Failed to get ibv_pd from DOCA verbs PD");
        return false;
    }

    // Get port attributes
    if (ibv_query_port(doca_state_->ibv_ctx, 1, &doca_state_->port_attr))
    {
        DOCA_LOG_ERR("Failed to query port");
        return false;
    }
    doca_state_->lid = doca_state_->port_attr.lid;

    // Get GID (using gid_index from config)
    doca_state_->gid_index = static_cast<int>(rdma_config_.gid_index);
    ibv_gid gid;
    if (ibv_query_gid(doca_state_->ibv_ctx, 1, doca_state_->gid_index, &gid))
    {
        DOCA_LOG_ERR("Failed to query GID");
        return false;
    }
    memcpy(doca_state_->gid, gid.raw, 16);

    // Create DOCA dev from PD
    result = doca_rdma_bridge_open_dev_from_pd(doca_state_->pd, &doca_state_->ddev);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create DOCA dev from PD: %s", doca_error_get_descr(result));
        return false;
    }
    DOCA_LOG_DBG("DOCA device created from PD: ddev=%p", doca_state_->ddev);

    // Create DOCA GPU device
    char pci_bus_id[32];
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), gpu_id_), "Failed to get GPU PCI bus ID");

    DOCA_LOG_DBG("Creating DOCA GPU for PCI bus ID: %s", pci_bus_id);
    result = doca_gpu_create(pci_bus_id, &doca_state_->gpu);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create DOCA GPU: %s", doca_error_get_descr(result));
        return false;
    }
    DOCA_LOG_DBG("DOCA GPU created: gpu=%p", doca_state_->gpu);

    // Test GPU memory allocation to verify DOCA GPUNetIO is working
    void* test_ptr = nullptr;
    result = doca_gpu_mem_alloc(doca_state_->gpu, 4096, 4096, DOCA_GPU_MEM_TYPE_GPU, &test_ptr, nullptr);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("DOCA GPU mem alloc test FAILED: %s - GPUNetIO may not be properly initialized",
            doca_error_get_descr(result));
        // Continue anyway, will fail later with more context
    }
    else
    {
        DOCA_LOG_DBG("DOCA GPU mem alloc test passed: ptr=%p", test_ptr);

        // Test CUDA dmabuf export on DOCA-allocated memory
        int test_dmabuf_fd = -1;
        CUresult cu_res = cuMemGetHandleForAddressRange(
            &test_dmabuf_fd, (CUdeviceptr) test_ptr, 4096, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (cu_res == CUDA_SUCCESS && test_dmabuf_fd >= 0)
        {
            DOCA_LOG_DBG("CUDA dmabuf export test passed: fd=%d", test_dmabuf_fd);
            close(test_dmabuf_fd);
        }
        else
        {
            char const* err_name = nullptr;
            cuGetErrorName(cu_res, &err_name);
            DOCA_LOG_WARN("CUDA dmabuf export test failed: %s (%d) - will try DOCA dmabuf",
                err_name ? err_name : "unknown", cu_res);

            // Try DOCA's dmabuf export
            int doca_dmabuf_fd = -1;
            doca_error_t doca_res = doca_gpu_dmabuf_fd(doca_state_->gpu, test_ptr, 4096, &doca_dmabuf_fd);
            if (doca_res == DOCA_SUCCESS && doca_dmabuf_fd >= 0)
            {
                DOCA_LOG_DBG("DOCA dmabuf export test passed: fd=%d", doca_dmabuf_fd);
                close(doca_dmabuf_fd);
            }
            else
            {
                DOCA_LOG_WARN("DOCA dmabuf export test failed: %s", doca_error_get_descr(doca_res));
            }
        }

        // Test UMEM creation with DOCA-allocated memory
        doca_umem* test_umem = nullptr;
        result = doca_umem_gpu_create(doca_state_->gpu, doca_state_->ddev, test_ptr, 4096,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
                | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &test_umem);
        if (result != DOCA_SUCCESS)
        {
            DOCA_LOG_ERR(
                "DOCA UMEM GPU create test FAILED: %s (result=%d)", doca_error_get_descr(result), (int) result);
            DOCA_LOG_ERR("This usually means:");
            DOCA_LOG_ERR("  1. GPU and NIC are not on the same PCIe root complex");
            DOCA_LOG_ERR("  2. MLNX_OFED version incompatible with DOCA version");
            DOCA_LOG_ERR("  3. GPUDirect RDMA not enabled in driver");
            DOCA_LOG_ERR("  4. Missing gdrdrv kernel module");
            DOCA_LOG_ERR("Check: lspci -t, dmesg | grep -i gpu, lsmod | grep gdr");
        }
        else
        {
            DOCA_LOG_DBG("DOCA UMEM GPU create test passed");
            doca_umem_destroy(test_umem);
        }

        doca_gpu_mem_free(doca_state_->gpu, test_ptr);
    }

    doca_verbs_addr_type addr_type = (doca_state_->port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND)
        ? DOCA_VERBS_ADDR_TYPE_IB_NO_GRH
        : DOCA_VERBS_ADDR_TYPE_IPv4;
    result = doca_verbs_ah_attr_create(doca_state_->verbs_ctx, &doca_state_->ah_attr);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create AH attr: %s", doca_error_get_descr(result));
        return false;
    }

    result = doca_verbs_ah_attr_set_addr_type(doca_state_->ah_attr, addr_type);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to set AH addr type: %s", doca_error_get_descr(result));
        return false;
    }

    result = doca_verbs_ah_attr_set_sgid_index(doca_state_->ah_attr, doca_state_->gid_index);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to set AH sgid index: %s", doca_error_get_descr(result));
        return false;
    }

    DOCA_LOG_INFO("DOCA devices setup complete: NIC=%s, GPU=%s", nic_name_.c_str(), pci_bus_id);
    return true;
}

std::unique_ptr<DocaRdmaEndpoint::CqState> DocaRdmaEndpoint::create_cq(uint32_t ncqe)
{
    auto cq = std::make_unique<CqState>();
    cq->size = ncqe;
    doca_error_t result;

    // Ensure CUDA context is set to our GPU
    cudaError_t cuda_err = cudaSetDevice(gpu_id_);
    if (cuda_err != cudaSuccess)
    {
        DOCA_LOG_ERR("Failed to set CUDA device %d: %s", gpu_id_, cudaGetErrorString(cuda_err));
        return nullptr;
    }

    doca_verbs_cq_attr* cq_attr = nullptr;
    result = doca_verbs_cq_attr_create(&cq_attr);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create CQ attr: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_cq_attr_set_external_datapath_en(cq_attr, 1);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_WARN("Failed to set CQ external datapath (may be unsupported): %s", doca_error_get_descr(result));
    }

    uint32_t umem_size = calc_cq_umem_size(ncqe);
    DOCA_LOG_DBG("Allocating CQ GPU memory: size=%u, page_size=%zu, gpu=%p, ddev=%p", umem_size, get_page_size(),
        doca_state_->gpu, doca_state_->ddev);

    result = doca_gpu_mem_alloc(
        doca_state_->gpu, umem_size, get_page_size(), DOCA_GPU_MEM_TYPE_GPU, &cq->umem_gpu_ptr, nullptr);
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_cq_attr_destroy(cq_attr);
        DOCA_LOG_ERR("Failed to alloc GPU memory for CQ: %s", doca_error_get_descr(result));
        return nullptr;
    }
    DOCA_LOG_DBG("CQ GPU memory allocated at %p", cq->umem_gpu_ptr);

    // Initialize CQEs on host and copy to GPU
    auto* cq_host = (struct mlx5_cqe64*) calloc(umem_size, 1);
    init_cqes(cq_host, ncqe);
    cuda_err = cudaMemcpy(cq->umem_gpu_ptr, cq_host, umem_size, cudaMemcpyDefault);
    free(cq_host);
    if (cuda_err != cudaSuccess)
    {
        DOCA_LOG_ERR("cudaMemcpy failed for CQ init: %s", cudaGetErrorString(cuda_err));
    }

    // Create UMEM from GPU memory
    DOCA_LOG_DBG("Creating UMEM for CQ: gpu=%p, ddev=%p, ptr=%p, size=%u", doca_state_->gpu, doca_state_->ddev,
        cq->umem_gpu_ptr, umem_size);
    result = doca_umem_gpu_create(doca_state_->gpu, doca_state_->ddev, cq->umem_gpu_ptr, umem_size,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
            | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
        &cq->umem);
    if (result != DOCA_SUCCESS)
    {
        doca_gpu_mem_free(doca_state_->gpu, cq->umem_gpu_ptr);
        doca_verbs_cq_attr_destroy(cq_attr);
        DOCA_LOG_ERR("Failed to create UMEM for CQ: %s (result=%d)", doca_error_get_descr(result), (int) result);
        return nullptr;
    }
    DOCA_LOG_DBG("CQ UMEM created successfully");

    result = doca_verbs_cq_attr_set_external_umem(cq_attr, cq->umem, 0);
    if (result != DOCA_SUCCESS)
    {
        doca_umem_destroy(cq->umem);
        doca_gpu_mem_free(doca_state_->gpu, cq->umem_gpu_ptr);
        doca_verbs_cq_attr_destroy(cq_attr);
        DOCA_LOG_ERR("Failed to set CQ external umem: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_cq_attr_set_cq_size(cq_attr, ncqe);
    if (result != DOCA_SUCCESS)
    {
        doca_umem_destroy(cq->umem);
        doca_gpu_mem_free(doca_state_->gpu, cq->umem_gpu_ptr);
        doca_verbs_cq_attr_destroy(cq_attr);
        DOCA_LOG_ERR("Failed to set CQ size: %s", doca_error_get_descr(result));
        return nullptr;
    }

    doca_verbs_cq_attr_set_cq_overrun(cq_attr, 1); // Optional, ignore errors

    result = doca_uar_create(doca_state_->ddev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &cq->uar);
    if (result != DOCA_SUCCESS)
    {
        doca_umem_destroy(cq->umem);
        doca_gpu_mem_free(doca_state_->gpu, cq->umem_gpu_ptr);
        doca_verbs_cq_attr_destroy(cq_attr);
        DOCA_LOG_ERR("Failed to create UAR for CQ: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_cq_attr_set_external_uar(cq_attr, cq->uar);
    if (result != DOCA_SUCCESS)
    {
        doca_uar_destroy(cq->uar);
        doca_umem_destroy(cq->umem);
        doca_gpu_mem_free(doca_state_->gpu, cq->umem_gpu_ptr);
        doca_verbs_cq_attr_destroy(cq_attr);
        DOCA_LOG_ERR("Failed to set CQ external UAR: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_cq_create(doca_state_->verbs_ctx, cq_attr, &cq->cq);
    doca_verbs_cq_attr_destroy(cq_attr);
    if (result != DOCA_SUCCESS)
    {
        doca_uar_destroy(cq->uar);
        doca_umem_destroy(cq->umem);
        doca_gpu_mem_free(doca_state_->gpu, cq->umem_gpu_ptr);
        DOCA_LOG_ERR("Failed to create CQ: %s", doca_error_get_descr(result));
        return nullptr;
    }

    return cq;
}

std::unique_ptr<DocaRdmaEndpoint::QpState> DocaRdmaEndpoint::create_qp()
{
    auto qp = std::make_unique<QpState>();
    qp->psn = rand() & 0xFFFFFF;
    doca_error_t result;

    // Create send and recv CQs
    qp->send_cq = create_cq(SQ_SIZE);
    if (!qp->send_cq)
    {
        DOCA_LOG_ERR("Failed to create send CQ");
        return nullptr;
    }

    qp->recv_cq = create_cq(RQ_SIZE);
    if (!qp->recv_cq)
    {
        DOCA_LOG_ERR("Failed to create recv CQ");
        return nullptr;
    }

    // Create QP attributes
    doca_verbs_qp_init_attr* qp_attr = nullptr;
    result = doca_verbs_qp_init_attr_create(&qp_attr);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create QP attr: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_qp_init_attr_set_external_datapath_en(qp_attr, 1);
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to set QP external datapath: %s", doca_error_get_descr(result));
        return nullptr;
    }

    // Create UAR for QP
    result = doca_uar_create(doca_state_->ddev, DOCA_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED, &qp->uar);
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to create UAR for QP: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_qp_init_attr_set_external_uar(qp_attr, qp->uar);
    if (result != DOCA_SUCCESS)
    {
        doca_uar_destroy(qp->uar);
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to set QP external UAR: %s", doca_error_get_descr(result));
        return nullptr;
    }

    // Allocate GPU memory for QP ring
    uint32_t qp_umem_size = calc_qp_umem_size(SQ_SIZE, RQ_SIZE);
    result = doca_gpu_mem_alloc(
        doca_state_->gpu, qp_umem_size, get_page_size(), DOCA_GPU_MEM_TYPE_GPU, &qp->umem_gpu_ptr, nullptr);
    if (result != DOCA_SUCCESS)
    {
        doca_uar_destroy(qp->uar);
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to alloc GPU memory for QP: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_umem_gpu_create(doca_state_->gpu, doca_state_->ddev, qp->umem_gpu_ptr, qp_umem_size,
        DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
            | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
        &qp->umem);
    if (result != DOCA_SUCCESS)
    {
        doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
        doca_uar_destroy(qp->uar);
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to create UMEM for QP: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_qp_init_attr_set_external_umem(qp_attr, qp->umem, 0);
    if (result != DOCA_SUCCESS)
    {
        doca_umem_destroy(qp->umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
        doca_uar_destroy(qp->uar);
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to set QP external umem: %s", doca_error_get_descr(result));
        return nullptr;
    }

    // Allocate doorbell record memory. For CPU_PROXY mode, DBR must be HOST
    // memory so the CPU progress thread can write it (matches DOCA sample
    // verbs_high_level.c CPU_PROXY branch using calloc()).
    uint32_t dbr_size = align_up(DBR_SIZE, get_page_size());
    bool use_host_dbr = false;
    {
        char const* hh = std::getenv("GLUE_RDMA_NIC_HANDLERS");
        char const* allow = std::getenv("GLUE_RDMA_ALLOW_PROXY_HANDLERS");
        bool allow_proxy = (allow && (allow[0] == '1' || allow[0] == 'y' || allow[0] == 'Y'));
        if (allow_proxy && hh)
        {
            std::string s(hh);
            if (s.find('0') != std::string::npos || s.find('1') != std::string::npos)
                use_host_dbr = true;
        }
    }
    qp->dbr_is_host = use_host_dbr;
    if (use_host_dbr)
    {
        void* host_ptr = std::calloc(dbr_size, 1);
        if (host_ptr == nullptr)
        {
            doca_umem_destroy(qp->umem);
            doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
            doca_uar_destroy(qp->uar);
            doca_verbs_qp_init_attr_destroy(qp_attr);
            DOCA_LOG_ERR("Failed to calloc CPU memory for DBR");
            return nullptr;
        }
        qp->dbr_host_alloc = host_ptr;
        qp->dbr_gpu_ptr = nullptr; // important: keeps later gpu_mem_free(NULL) a no-op
        result = doca_umem_create(doca_state_->ddev, host_ptr, dbr_size,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
                | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &qp->dbr_umem);
    }
    else
    {
        result = doca_gpu_mem_alloc(
            doca_state_->gpu, dbr_size, get_page_size(), DOCA_GPU_MEM_TYPE_GPU, &qp->dbr_gpu_ptr, nullptr);
        if (result != DOCA_SUCCESS)
        {
            doca_umem_destroy(qp->umem);
            doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
            doca_uar_destroy(qp->uar);
            doca_verbs_qp_init_attr_destroy(qp_attr);
            DOCA_LOG_ERR("Failed to alloc GPU memory for DBR: %s", doca_error_get_descr(result));
            return nullptr;
        }
        result = doca_umem_gpu_create(doca_state_->gpu, doca_state_->ddev, qp->dbr_gpu_ptr, dbr_size,
            DOCA_ACCESS_FLAG_LOCAL_READ_WRITE | DOCA_ACCESS_FLAG_RDMA_WRITE | DOCA_ACCESS_FLAG_RDMA_READ
                | DOCA_ACCESS_FLAG_RDMA_ATOMIC,
            &qp->dbr_umem);
    }
    if (result != DOCA_SUCCESS)
    {
        doca_gpu_mem_free(doca_state_->gpu, qp->dbr_gpu_ptr);
        doca_umem_destroy(qp->umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
        doca_uar_destroy(qp->uar);
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to create UMEM for DBR: %s", doca_error_get_descr(result));
        return nullptr;
    }

    result = doca_verbs_qp_init_attr_set_external_dbr_umem(qp_attr, qp->dbr_umem, 0);
    if (result != DOCA_SUCCESS)
    {
        doca_umem_destroy(qp->dbr_umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->dbr_gpu_ptr);
        doca_umem_destroy(qp->umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
        doca_uar_destroy(qp->uar);
        doca_verbs_qp_init_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to set QP external DBR umem: %s", doca_error_get_descr(result));
        return nullptr;
    }

    // Set QP parameters
    doca_verbs_qp_init_attr_set_pd(qp_attr, doca_state_->verbs_pd);
    doca_verbs_qp_init_attr_set_sq_wr(qp_attr, SQ_SIZE);
    doca_verbs_qp_init_attr_set_rq_wr(qp_attr, RQ_SIZE);
    doca_verbs_qp_init_attr_set_qp_type(qp_attr, DOCA_VERBS_QP_TYPE_RC);
    doca_verbs_qp_init_attr_set_send_cq(qp_attr, qp->send_cq->cq);
    doca_verbs_qp_init_attr_set_receive_cq(qp_attr, qp->recv_cq->cq);
    doca_verbs_qp_init_attr_set_send_max_sges(qp_attr, 1);
    doca_verbs_qp_init_attr_set_receive_max_sges(qp_attr, 1);

    // Create QP
    result = doca_verbs_qp_create(doca_state_->verbs_ctx, qp_attr, &qp->qp);
    doca_verbs_qp_init_attr_destroy(qp_attr);
    if (result != DOCA_SUCCESS)
    {
        doca_umem_destroy(qp->dbr_umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->dbr_gpu_ptr);
        doca_umem_destroy(qp->umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
        doca_uar_destroy(qp->uar);
        DOCA_LOG_ERR("Failed to create QP: %s", doca_error_get_descr(result));
        return nullptr;
    }

    qp->qpn = doca_verbs_qp_get_qpn(qp->qp);

    // Export QP to GPU. On some platform/driver combinations (for example
    // newer hardware bring-up) a different NIC handler ID may be required.
    // Keep default behavior, but allow opt-in overrides via
    // GLUE_RDMA_NIC_HANDLERS (comma-separated integer list).
    result = DOCA_ERROR_INVALID_VALUE;
    auto nic_handlers = load_nic_handler_candidates();
    for (uint32_t handler_raw : nic_handlers)
    {
        result = doca_gpu_verbs_export_qp(doca_state_->gpu, doca_state_->ddev, qp->qp,
            static_cast<decltype(DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB)>(handler_raw), qp->umem_gpu_ptr,
            qp->send_cq->cq, qp->recv_cq->cq, &qp->gpu_qp);
        if (result == DOCA_SUCCESS)
        {
            DOCA_LOG_INFO("Exported QP to GPU with NIC handler id=%u", handler_raw);
            break;
        }
        DOCA_LOG_WARN("QP export with NIC handler id=%u failed: %s", handler_raw, doca_error_get_descr(result));
    }
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_qp_destroy(qp->qp);
        doca_umem_destroy(qp->dbr_umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->dbr_gpu_ptr);
        doca_umem_destroy(qp->umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
        doca_uar_destroy(qp->uar);
        DOCA_LOG_ERR("Failed to export QP to GPU: %s", doca_error_get_descr(result));
        return nullptr;
    }

    // Get device pointer
    result = doca_gpu_verbs_get_qp_dev(qp->gpu_qp, &qp->dev_qp);
    if (result != DOCA_SUCCESS)
    {
        doca_gpu_verbs_unexport_qp(doca_state_->gpu, qp->gpu_qp);
        doca_verbs_qp_destroy(qp->qp);
        doca_umem_destroy(qp->dbr_umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->dbr_gpu_ptr);
        doca_umem_destroy(qp->umem);
        doca_gpu_mem_free(doca_state_->gpu, qp->umem_gpu_ptr);
        doca_uar_destroy(qp->uar);
        DOCA_LOG_ERR("Failed to get QP device pointer: %s", doca_error_get_descr(result));
        return nullptr;
    }

    DOCA_LOG_DBG("Created QP: qpn=%u, psn=%u, dev_qp=%p", qp->qpn, qp->psn, qp->dev_qp);
    return qp;
}

void DocaRdmaEndpoint::cleanup()
{
    if (!initialized_)
        return;

    // Cleanup memory registrations
    for (auto& mr : doca_state_->mem_regs)
    {
        if (mr.mr)
            ibv_dereg_mr(mr.mr);
    }
    doca_state_->mem_regs.clear();

    // Cleanup peer QPs
    for (auto& pqp : doca_state_->peer_qps)
    {
        // TODO: proper cleanup of DOCA verbs objects
    }
    doca_state_->peer_qps.clear();

    // Cleanup peers
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        for (auto& peer : peers_)
        {
            if (peer.sock_fd >= 0)
                close(peer.sock_fd);
        }
        peers_.clear();
    }

    // Cleanup DOCA state
    if (doca_state_->gpu)
    {
        doca_gpu_destroy(doca_state_->gpu);
        doca_state_->gpu = nullptr;
    }
    if (doca_state_->ddev)
    {
        doca_dev_close(doca_state_->ddev);
        doca_state_->ddev = nullptr;
    }
    if (doca_state_->pd)
    {
        ibv_dealloc_pd(doca_state_->pd);
        doca_state_->pd = nullptr;
    }
    if (doca_state_->ibv_ctx)
    {
        ibv_close_device(doca_state_->ibv_ctx);
        doca_state_->ibv_ctx = nullptr;
    }

    initialized_ = false;
}

BufferInfo DocaRdmaEndpoint::register_buffer(void* gpu_ptr, size_t length)
{
    BufferInfo info = {0};

    if (!initialized_)
    {
        DOCA_LOG_ERR("Not initialized");
        return info;
    }
    if (length == 0)
    {
        DOCA_LOG_ERR("Invalid buffer registration: length=0");
        return info;
    }

    DOCA_LOG_DBG("=== BUFFER REGISTRATION: ptr=%p, len=%zu ===", gpu_ptr, length);

    static size_t page_size = get_page_size();

    // Round up to page-size multiple for dmabuf registration.
    size_t aligned_length = ((length + page_size - 1) / page_size) * page_size;
    if (aligned_length != length)
    {
        DOCA_LOG_DBG("Rounding up registration from %zu to %zu (page_size=%zu)", length, aligned_length, page_size);
    }

    // ============================================================
    // Preferred path: register existing CUDA allocation via dmabuf
    // ============================================================
    if (gpu_ptr != nullptr)
    {
        CUdeviceptr base_ptr = 0;
        size_t alloc_size = 0;
        CUresult cu_res = cuMemGetAddressRange(&base_ptr, &alloc_size, (CUdeviceptr) gpu_ptr);
        if (cu_res == CUDA_SUCCESS && base_ptr != 0 && alloc_size != 0)
        {
            size_t offset = (size_t) ((uintptr_t) gpu_ptr - (uintptr_t) base_ptr);
            if (offset + length <= alloc_size)
            {
                size_t reg_length = aligned_length;
                if (offset + reg_length > alloc_size)
                {
                    DOCA_LOG_WARN(
                        "Aligned length exceeds CUDA allocation, using unaligned length: "
                        "offset=%zu len=%zu aligned=%zu alloc_size=%zu",
                        offset, length, reg_length, alloc_size);
                    reg_length = length;
                }

                int dmabuf_fd = -1;
                cu_res = cuMemGetHandleForAddressRange(
                    &dmabuf_fd, base_ptr, alloc_size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
                if (cu_res == CUDA_SUCCESS && dmabuf_fd >= 0)
                {
                    ibv_mr* mr = ibv_reg_dmabuf_mr(doca_state_->pd, offset, reg_length, (uint64_t) gpu_ptr, dmabuf_fd,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ
                            | IBV_ACCESS_REMOTE_ATOMIC);
                    close(dmabuf_fd);
                    if (mr)
                    {
                        doca_state_->doca_buffers.push_back({gpu_ptr, reg_length, mr, "cuda_dmabuf"});

                        info.addr = (uint64_t) gpu_ptr;
                        info.length = length;
                        info.lkey = mr->lkey;
                        info.rkey = mr->rkey;
                        info.buf_arr_index = -1;
                        info.gpu_buf_arr = 0;
                        info.original_addr = (uint64_t) gpu_ptr;

                        DOCA_LOG_DBG(
                            "=== REGISTRATION SUCCESS (CUDA dmabuf): addr=0x%lx, "
                            "lkey=0x%x, rkey=0x%x (host-endian) ===",
                            info.addr, info.lkey, info.rkey);
                        return info;
                    }

                    DOCA_LOG_ERR("ibv_reg_dmabuf_mr (CUDA dmabuf) failed: errno=%d (%s)", errno, strerror(errno));
                }
                else
                {
                    char const* err_name = nullptr;
                    cuGetErrorName(cu_res, &err_name);
                    DOCA_LOG_WARN(
                        "cuMemGetHandleForAddressRange failed: %s (%d)", err_name ? err_name : "unknown", cu_res);
                }
            }
            else
            {
                DOCA_LOG_ERR("Requested range exceeds CUDA allocation: offset=%zu len=%zu alloc_size=%zu", offset,
                    length, alloc_size);
            }
        }
        else
        {
            char const* err_name = nullptr;
            cuGetErrorName(cu_res, &err_name);
            DOCA_LOG_WARN(
                "cuMemGetAddressRange failed for ptr=%p: %s (%d)", gpu_ptr, err_name ? err_name : "unknown", cu_res);
        }
    }

    // ============================================================
    // Fallback: try DOCA dmabuf export for existing DOCA allocations
    // ============================================================
    if (gpu_ptr != nullptr)
    {
        int dmabuf_fd = -1;
        doca_error_t res = doca_gpu_dmabuf_fd(doca_state_->gpu, gpu_ptr, aligned_length, &dmabuf_fd);
        if (res == DOCA_SUCCESS && dmabuf_fd >= 0)
        {
            ibv_mr* mr = ibv_reg_dmabuf_mr(doca_state_->pd, 0, aligned_length, (uint64_t) gpu_ptr, dmabuf_fd,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
            close(dmabuf_fd);
            if (mr)
            {
                doca_state_->doca_buffers.push_back({gpu_ptr, aligned_length, mr, "doca_dmabuf"});

                info.addr = (uint64_t) gpu_ptr;
                info.length = length;
                info.lkey = mr->lkey;
                info.rkey = mr->rkey;
                info.buf_arr_index = -1;
                info.gpu_buf_arr = 0;
                info.original_addr = (uint64_t) gpu_ptr;

                DOCA_LOG_DBG(
                    "=== REGISTRATION SUCCESS (DOCA dmabuf): addr=0x%lx, "
                    "lkey=0x%x, rkey=0x%x (host-endian) ===",
                    info.addr, info.lkey, info.rkey);
                return info;
            }
            DOCA_LOG_ERR("ibv_reg_dmabuf_mr (DOCA dmabuf) failed: errno=%d (%s)", errno, strerror(errno));
        }
        else if (res != DOCA_SUCCESS)
        {
            DOCA_LOG_WARN("doca_gpu_dmabuf_fd failed: %s", doca_error_get_descr(res));
        }
    }

    // ============================================================
    // Last resort: allocate DOCA GPU memory and register it
    // ============================================================
    void* doca_ptr = nullptr;
    doca_error_t res
        = doca_gpu_mem_alloc(doca_state_->gpu, aligned_length, page_size, DOCA_GPU_MEM_TYPE_GPU, &doca_ptr, nullptr);
    if (res != DOCA_SUCCESS || !doca_ptr)
    {
        DOCA_LOG_ERR("doca_gpu_mem_alloc failed: %s", doca_error_get_descr(res));
        return info;
    }
    DOCA_LOG_DBG("DOCA GPU memory allocated: %p (zero-copy, user accesses directly)", doca_ptr);

    int dmabuf_fd = -1;
    res = doca_gpu_dmabuf_fd(doca_state_->gpu, doca_ptr, aligned_length, &dmabuf_fd);
    if (res != DOCA_SUCCESS || dmabuf_fd < 0)
    {
        DOCA_LOG_ERR(
            "doca_gpu_dmabuf_fd failed: %s - GPU dmabuf export not supported. "
            "Ensure CUDA 12.8+, dmabuf-capable kernel, and proper GPU driver.",
            doca_error_get_descr(res));
        doca_gpu_mem_free(doca_state_->gpu, doca_ptr);
        return info;
    }
    DOCA_LOG_DBG("Got dmabuf fd=%d from DOCA", dmabuf_fd);

    ibv_mr* mr = ibv_reg_dmabuf_mr(doca_state_->pd, 0, aligned_length, (uint64_t) doca_ptr, dmabuf_fd,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
    if (!mr)
    {
        DOCA_LOG_ERR(
            "ibv_reg_dmabuf_mr failed: errno=%d (%s) - check kernel dmabuf/RDMA support", errno, strerror(errno));
        close(dmabuf_fd);
        doca_gpu_mem_free(doca_state_->gpu, doca_ptr);
        return info;
    }
    close(dmabuf_fd);
    DOCA_LOG_DBG("ibv_reg_dmabuf_mr success: lkey=0x%x, rkey=0x%x", mr->lkey, mr->rkey);

    doca_state_->doca_buffers.push_back({doca_ptr, aligned_length, mr, "doca_alloc"});

    info.addr = (uint64_t) doca_ptr;
    info.length = length;
    info.lkey = mr->lkey;
    info.rkey = mr->rkey;
    info.buf_arr_index = -1;
    info.gpu_buf_arr = 0;
    info.original_addr = (uint64_t) gpu_ptr;

    DOCA_LOG_DBG(
        "=== REGISTRATION SUCCESS (DOCA alloc): addr=0x%lx, lkey=0x%x, "
        "rkey=0x%x (host-endian) ===",
        info.addr, info.lkey, info.rkey);
    return info;
}

void DocaRdmaEndpoint::deregister_buffer(BufferInfo const& buf)
{
    auto it = std::find_if(doca_state_->mem_regs.begin(), doca_state_->mem_regs.end(),
        [&buf](DocaState::MemReg const& r) { return r.addr == (void*) buf.addr; });
    if (it != doca_state_->mem_regs.end())
    {
        ibv_dereg_mr(it->mr);
        doca_state_->mem_regs.erase(it);
    }
}

bool DocaRdmaEndpoint::connect_to_peer(std::string const& host, int port, std::string const& peer_name)
{
    DOCA_LOG_DBG("Connecting to peer %s at %s:%d", peer_name.c_str(), host.c_str(), port);

    // Create socket and connect
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        DOCA_LOG_ERR("Failed to create socket");
        return false;
    }

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);

    if (host == "localhost" || host == "127.0.0.1")
    {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    }
    else
    {
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
        {
            hostent* he = gethostbyname(host.c_str());
            if (!he)
            {
                DOCA_LOG_ERR("Failed to resolve host: %s", host.c_str());
                close(sock);
                return false;
            }
            memcpy(&addr.sin_addr, he->h_addr, he->h_length);
        }
    }

    if (connect(sock, (sockaddr*) &addr, sizeof(addr)) < 0)
    {
        DOCA_LOG_ERR("Failed to connect to %s:%d", host.c_str(), port);
        close(sock);
        return false;
    }

    // Create peer entry
    RemotePeer peer;
    peer.name = peer_name;
    peer.sock_fd = sock;

    // Create QP for this peer
    if (!create_qp_for_peer(peer))
    {
        DOCA_LOG_ERR("Failed to create QP for peer %s", peer_name.c_str());
        close(sock);
        return false;
    }

    // Exchange QP info
    QpInfo local_qp;
    local_qp.qpn = peer.qp_info.qpn;
    local_qp.psn = peer.qp_info.psn;
    local_qp.lid = doca_state_->lid;
    memcpy(local_qp.gid, doca_state_->gid, 16);

    if (!exchange_qp_info(sock, local_qp, peer.qp_info))
    {
        DOCA_LOG_ERR("Failed to exchange QP info with peer %s", peer_name.c_str());
        close(sock);
        return false;
    }

    // Add peer to list before connecting QP (connect_qp uses find_peer_index)
    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peers_.push_back(peer);
    }

    if (!connect_qp(peer))
    {
        DOCA_LOG_ERR("Failed to connect QP to peer %s", peer_name.c_str());
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peers_.pop_back();
        close(sock);
        return false;
    }

    DOCA_LOG_INFO("Connected to peer %s", peer_name.c_str());
    return true;
}

bool DocaRdmaEndpoint::accept_peer(int port, std::string const& peer_name)
{
    DOCA_LOG_DBG("Listening for peer %s on port %d", peer_name.c_str(), port);

    int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_sock < 0)
    {
        DOCA_LOG_ERR("Failed to create listen socket");
        return false;
    }

    int const kEnableReuseAddr = 1;
    if (setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &kEnableReuseAddr, sizeof(kEnableReuseAddr)) < 0)
    {
        DOCA_LOG_ERR("Failed to set SO_REUSEADDR on listen socket");
        close(listen_sock);
        return false;
    }

    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    if (bind(listen_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
    {
        DOCA_LOG_ERR("Failed to bind listen socket on port %d", port);
        close(listen_sock);
        return false;
    }

    if (::listen(listen_sock, 1) < 0)
    {
        DOCA_LOG_ERR("Failed to listen on port %d", port);
        close(listen_sock);
        return false;
    }

    int sock = accept(listen_sock, nullptr, nullptr);
    close(listen_sock);
    if (sock < 0)
    {
        DOCA_LOG_ERR("Failed to accept peer connection");
        return false;
    }

    RemotePeer peer;
    peer.name = peer_name;
    peer.sock_fd = sock;

    if (!create_qp_for_peer(peer))
    {
        DOCA_LOG_ERR("Failed to create QP for peer %s", peer_name.c_str());
        close(sock);
        return false;
    }

    QpInfo local_qp;
    local_qp.qpn = peer.qp_info.qpn;
    local_qp.psn = peer.qp_info.psn;
    local_qp.lid = doca_state_->lid;
    memcpy(local_qp.gid, doca_state_->gid, 16);

    if (!exchange_qp_info(sock, local_qp, peer.qp_info))
    {
        DOCA_LOG_ERR("Failed to exchange QP info with peer %s", peer_name.c_str());
        close(sock);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peers_.push_back(peer);
    }

    if (!connect_qp(peer))
    {
        DOCA_LOG_ERR("Failed to connect QP to peer %s", peer_name.c_str());
        std::lock_guard<std::mutex> lock(peers_mutex_);
        peers_.pop_back();
        close(sock);
        return false;
    }

    DOCA_LOG_INFO("Accepted peer %s", peer_name.c_str());
    return true;
}

bool DocaRdmaEndpoint::create_qp_for_peer(RemotePeer& peer)
{
    auto qp = create_qp();
    if (!qp)
    {
        DOCA_LOG_ERR("Failed to create QP for peer");
        return false;
    }

    peer.qp_info.qpn = qp->qpn;
    peer.qp_info.psn = qp->psn;

    doca_state_->peer_qps.push_back(std::move(qp));
    return true;
}

bool DocaRdmaEndpoint::connect_qp(RemotePeer& peer)
{
    int idx = find_peer_index(peer.name);
    if (idx < 0)
    {
        DOCA_LOG_ERR("Peer not found: %s", peer.name.c_str());
        return false;
    }

    auto& qp = doca_state_->peer_qps[idx];
    doca_error_t result;

    // Create QP modify attributes
    doca_verbs_qp_attr* qp_attr = nullptr;
    result = doca_verbs_qp_attr_create(&qp_attr);
    if (result != DOCA_SUCCESS)
    {
        DOCA_LOG_ERR("Failed to create QP modify attr: %s", doca_error_get_descr(result));
        return false;
    }

    doca_verbs_gid remote_gid;
    memcpy(remote_gid.raw, peer.qp_info.gid, 16);
    result = doca_verbs_ah_attr_set_gid(doca_state_->ah_attr, remote_gid);
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_qp_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed to set remote GID: %s", doca_error_get_descr(result));
        return false;
    }

    // For IB, set DLID
    if (doca_state_->port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND)
    {
        result = doca_verbs_ah_attr_set_dlid(doca_state_->ah_attr, peer.qp_info.lid);
        if (result != DOCA_SUCCESS)
        {
            doca_verbs_qp_attr_destroy(qp_attr);
            DOCA_LOG_ERR("Failed to set DLID: %s", doca_error_get_descr(result));
            return false;
        }
    }

    // Configure QP attributes (using env var tunables from rdma_config_)
    doca_verbs_qp_attr_set_path_mtu(qp_attr, mtu_from_value(rdma_config_.path_mtu));
    doca_verbs_qp_attr_set_rq_psn(qp_attr, peer.qp_info.psn);
    doca_verbs_qp_attr_set_sq_psn(qp_attr, qp->psn);
    doca_verbs_qp_attr_set_port_num(qp_attr, 1);
    doca_verbs_qp_attr_set_ack_timeout(qp_attr, rdma_config_.timeout);
    doca_verbs_qp_attr_set_retry_cnt(qp_attr, rdma_config_.retry_cnt);
    doca_verbs_qp_attr_set_rnr_retry(qp_attr, rdma_config_.rnr_retry);
    doca_verbs_qp_attr_set_min_rnr_timer(qp_attr, rdma_config_.min_rnr_timer);
    doca_verbs_qp_attr_set_allow_remote_write(qp_attr, 1);
    doca_verbs_qp_attr_set_allow_remote_read(qp_attr, 1);
    doca_verbs_qp_attr_set_atomic_mode(qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC);
    doca_verbs_qp_attr_set_max_rd_atomic(qp_attr, rdma_config_.max_rd_atomic);
    doca_verbs_qp_attr_set_max_dest_rd_atomic(qp_attr, rdma_config_.max_dest_rd_atomic);
    doca_verbs_qp_attr_set_ah_attr(qp_attr, doca_state_->ah_attr);
    doca_verbs_qp_attr_set_dest_qp_num(qp_attr, peer.qp_info.qpn);

    // RST -> INIT
    doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_INIT);
    result = doca_verbs_qp_modify(qp->qp, qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_ATOMIC_MODE
            | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM);
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_qp_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed RST->INIT: %s", doca_error_get_descr(result));
        return false;
    }

    // INIT -> RTR
    doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_RTR);
    result = doca_verbs_qp_modify(qp->qp, qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN | DOCA_VERBS_QP_ATTR_DEST_QP_NUM
            | DOCA_VERBS_QP_ATTR_PATH_MTU | DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER
            | DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC);
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_qp_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed INIT->RTR: %s", doca_error_get_descr(result));
        return false;
    }

    // RTR -> RTS
    doca_verbs_qp_attr_set_next_state(qp_attr, DOCA_VERBS_QP_STATE_RTS);
    result = doca_verbs_qp_modify(qp->qp, qp_attr,
        DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN | DOCA_VERBS_QP_ATTR_ACK_TIMEOUT
            | DOCA_VERBS_QP_ATTR_RETRY_CNT | DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC);
    if (result != DOCA_SUCCESS)
    {
        doca_verbs_qp_attr_destroy(qp_attr);
        DOCA_LOG_ERR("Failed RTR->RTS: %s", doca_error_get_descr(result));
        return false;
    }

    doca_verbs_qp_attr_destroy(qp_attr);
    DOCA_LOG_INFO("QP connected: local_qpn=%u -> remote_qpn=%u", qp->qpn, peer.qp_info.qpn);
    return true;
}

int DocaRdmaEndpoint::find_peer_index(std::string const& peer_name)
{
    std::lock_guard<std::mutex> lock(peers_mutex_);
    for (size_t i = 0; i < peers_.size(); i++)
    {
        if (peers_[i].name == peer_name)
            return (int) i;
    }
    return -1;
}

bool DocaRdmaEndpoint::exchange_qp_info(int sock_fd, QpInfo& local, QpInfo& remote)
{
    // Send local QP info
    if (send(sock_fd, &local, sizeof(local), 0) != sizeof(local))
    {
        DOCA_LOG_ERR("Failed to send QP info");
        return false;
    }

    // Receive remote QP info
    if (recv(sock_fd, &remote, sizeof(remote), MSG_WAITALL) != sizeof(remote))
    {
        DOCA_LOG_ERR("Failed to receive QP info");
        return false;
    }

    DOCA_LOG_DBG("QP exchange: local_qpn=%u, remote_qpn=%u", local.qpn, remote.qpn);
    return true;
}

bool DocaRdmaEndpoint::exchange_buffer_info(
    std::string const& peer_name, BufferInfo const& local_send, BufferInfo const& local_recv)
{
    std::lock_guard<std::mutex> lock(peers_mutex_);

    auto it = std::find_if(peers_.begin(), peers_.end(), [&](RemotePeer const& p) { return p.name == peer_name; });
    if (it == peers_.end())
    {
        DOCA_LOG_ERR("Peer not found: %s", peer_name.c_str());
        return false;
    }

    // Send our buffer info (what remote should write to)
    if (send(it->sock_fd, &local_recv, sizeof(local_recv), 0) != sizeof(local_recv))
    {
        DOCA_LOG_ERR("Failed to send buffer info");
        return false;
    }

    // Receive remote's buffer info (what we should write to)
    // Keys are received in host-endian (LPU uses standard ibverbs)
    // Conversion to big-endian happens when passing to GPU kernels
    if (recv(it->sock_fd, &it->recv_buf, sizeof(it->recv_buf), MSG_WAITALL) != sizeof(it->recv_buf))
    {
        DOCA_LOG_ERR("Failed to receive buffer info");
        return false;
    }

    it->send_buf = local_send;

    DOCA_LOG_INFO("Buffer exchange with %s: remote_addr=0x%lx, remote_rkey=0x%x", peer_name.c_str(), it->recv_buf.addr,
        it->recv_buf.rkey);
    return true;
}

void* DocaRdmaEndpoint::get_qp_dev_ptr(std::string const& peer_name)
{
    int idx = find_peer_index(peer_name);
    if (idx < 0 || idx >= (int) doca_state_->peer_qps.size())
        return nullptr;
    return doca_state_->peer_qps[idx]->dev_qp;
}

void* DocaRdmaEndpoint::get_send_cq_dev_ptr(std::string const& peer_name)
{
    int idx = find_peer_index(peer_name);
    if (idx < 0 || idx >= (int) doca_state_->peer_qps.size())
        return nullptr;
    return doca_state_->peer_qps[idx]->send_cq ? doca_state_->peer_qps[idx]->send_cq->dev_cq : nullptr;
}

void* DocaRdmaEndpoint::get_recv_cq_dev_ptr(std::string const& peer_name)
{
    int idx = find_peer_index(peer_name);
    if (idx < 0 || idx >= (int) doca_state_->peer_qps.size())
        return nullptr;
    return doca_state_->peer_qps[idx]->recv_cq ? doca_state_->peer_qps[idx]->recv_cq->dev_cq : nullptr;
}

void* DocaRdmaEndpoint::get_buf_arr_gpu_handle(int buffer_index)
{
    if (buffer_index < 0 || buffer_index >= (int) doca_state_->buf_arr_regs.size())
    {
        DOCA_LOG_WARN("Invalid buffer index: %d (have %zu)", buffer_index, doca_state_->buf_arr_regs.size());
        return nullptr;
    }
    return doca_state_->buf_arr_regs[buffer_index].gpu_buf_arr;
}

BufferInfo DocaRdmaEndpoint::get_remote_recv_buf(std::string const& peer_name)
{
    std::lock_guard<std::mutex> lock(peers_mutex_);
    auto it = std::find_if(peers_.begin(), peers_.end(), [&](RemotePeer const& p) { return p.name == peer_name; });
    if (it != peers_.end())
    {
        return it->recv_buf;
    }
    return BufferInfo{0};
}

BufferInfo DocaRdmaEndpoint::get_local_send_buf(std::string const& peer_name)
{
    std::lock_guard<std::mutex> lock(peers_mutex_);
    auto it = std::find_if(peers_.begin(), peers_.end(), [&](RemotePeer const& p) { return p.name == peer_name; });
    if (it != peers_.end())
    {
        return it->send_buf;
    }
    return BufferInfo{0};
}

bool DocaRdmaEndpoint::post_recv(std::string const& peer_name, void* buf, size_t length, uint32_t lkey)
{
    int idx = find_peer_index(peer_name);
    if (idx < 0 || idx >= (int) doca_state_->peer_qps.size())
        return false;
    auto& qp_state = doca_state_->peer_qps[idx];
    if (!qp_state->qp)
        return false;

    struct ibv_sge sge
    {
    };

    sge.addr = reinterpret_cast<uintptr_t>(buf);
    sge.length = (uint32_t) length;
    sge.lkey = lkey;

    struct ibv_recv_wr wr
    {
    };

    wr.wr_id = 0;
    wr.next = nullptr;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    struct ibv_recv_wr* bad = nullptr;
    int rc = doca_verbs_bridge_post_recv(qp_state->qp, &wr, &bad);
    if (rc != 0)
    {
        DOCA_LOG_ERR("doca_verbs_bridge_post_recv failed: rc=%d", rc);
        return false;
    }
    return true;
}

bool DocaRdmaEndpoint::send_ready_signal(std::string const& peer_name)
{
    std::lock_guard<std::mutex> lock(peers_mutex_);
    auto it = std::find_if(peers_.begin(), peers_.end(), [&](RemotePeer const& p) { return p.name == peer_name; });
    if (it == peers_.end())
    {
        DOCA_LOG_ERR("Peer not found: %s", peer_name.c_str());
        return false;
    }

    char ready = 'R';
    if (send(it->sock_fd, &ready, 1, 0) != 1)
    {
        DOCA_LOG_ERR("Failed to send ready signal to %s", peer_name.c_str());
        return false;
    }

    DOCA_LOG_INFO("Sent ready signal to %s", peer_name.c_str());
    return true;
}

BufferInfo DocaRdmaEndpoint::alloc_doca_gpu_buffer(size_t length)
{
    BufferInfo info = {0};

    if (!initialized_)
    {
        DOCA_LOG_ERR("Not initialized");
        return info;
    }

    // Co-existence with PyTorch / TRT-LLM in the same process: when the
    // caller opts in via ``GLUE_DOCA_HOST_PINNED_BUFFERS=1`` we allocate
    // the buffer in host-pinned RAM and hand the device-mapped pointer
    // out as if it were GPU memory. The NIC accesses it via dmabuf
    // registration against the host-pinned page (OS pinned, can't be
    // migrated), while the GPU kernel sees a normal GPU virtual address.
    // This sidesteps the torch caching allocator vs DOCA GPU mem pool
    // page-migration conflict observed in ``phase13_doca_cpu_proxy.md``.
    {
        char const* env = std::getenv("GLUE_DOCA_HOST_PINNED_BUFFERS");
        bool host_pinned = env && (env[0] == '1' || env[0] == 'y' || env[0] == 'Y');
        if (host_pinned)
        {
            // Round up to page so the dmabuf FD covers the whole alloc;
            // matches what register_buffer does for the GPU path.
            size_t page = get_page_size();
            size_t aligned = ((length + page - 1) / page) * page;
            void* host_ptr = nullptr;
            cudaError_t cerr = cudaHostAlloc(&host_ptr, aligned, cudaHostAllocPortable | cudaHostAllocMapped);
            if (cerr != cudaSuccess || host_ptr == nullptr)
            {
                DOCA_LOG_ERR("cudaHostAlloc(%zu) failed: %s", aligned, cudaGetErrorString(cerr));
                return info;
            }
            std::memset(host_ptr, 0, aligned);
            void* dev_ptr = nullptr;
            cerr = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
            if (cerr != cudaSuccess || dev_ptr == nullptr)
            {
                cudaFreeHost(host_ptr);
                DOCA_LOG_ERR("cudaHostGetDevicePointer failed: %s", cudaGetErrorString(cerr));
                return info;
            }
            // Register MR against the host-pinned mapping. The NIC then
            // DMAs to/from host RAM (over PCIe), but GPU kernels still
            // see ``dev_ptr`` as a normal GPU VA via UVA.
            ibv_mr* mr = ibv_reg_mr(doca_state_->pd, host_ptr, aligned,
                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
            if (!mr)
            {
                cudaFreeHost(host_ptr);
                DOCA_LOG_ERR("ibv_reg_mr (host pinned) failed: errno=%d (%s)", errno, strerror(errno));
                return info;
            }
            // Track the host alloc so ``deregister_buffer`` can free it.
            doca_state_->doca_buffers.push_back({host_ptr, aligned, mr, "host_pinned"});
            info.addr = (uint64_t) dev_ptr;
            info.length = length;
            info.lkey = mr->lkey;
            info.rkey = mr->rkey;
            info.buf_arr_index = -1;
            DOCA_LOG_INFO("Host-pinned DOCA buffer: dev=%p host=%p len=%zu lkey=0x%x rkey=0x%x", dev_ptr, host_ptr,
                length, info.lkey, info.rkey);
            return info;
        }
    }

    // Allocate GPU memory using DOCA
    void* gpu_ptr = nullptr;
    doca_error_t result = doca_gpu_mem_alloc(doca_state_->gpu, length, 4096, DOCA_GPU_MEM_TYPE_GPU, &gpu_ptr, nullptr);
    if (result != DOCA_SUCCESS || !gpu_ptr)
    {
        DOCA_LOG_ERR("doca_gpu_mem_alloc failed: %s", doca_error_get_descr(result));
        return info;
    }

    DOCA_LOG_DBG("DOCA GPU memory allocated: ptr=%p, length=%zu", gpu_ptr, length);

    // Now register it for RDMA
    return register_buffer(gpu_ptr, length);
}

bool DocaRdmaEndpoint::rdma_write_with_imm(
    std::string const& peer_name, void* local_buf, size_t length, uint32_t imm_value)
{
    DOCA_LOG_WARN("Host-side RDMA not implemented - use CUDA kernels");
    return false;
}

bool DocaRdmaEndpoint::poll_for_completion(std::string const& peer_name, uint32_t* imm_value)
{
    DOCA_LOG_WARN("Host-side polling not implemented - use CUDA kernels");
    return false;
}

bool DocaRdmaEndpoint::listen(int port)
{
    return accept_peer(port, "rdma_peer");
}

int DocaRdmaEndpoint::cpu_proxy_enabled(std::string const& peer_name)
{
    int idx = find_peer_index(peer_name);
    if (idx < 0 || idx >= (int) doca_state_->peer_qps.size())
        return -1;
    auto& qp_state = doca_state_->peer_qps[idx];
    if (qp_state->gpu_qp == nullptr)
        return -2;
    uint8_t enabled = 0;
    doca_error_t r = doca_gpu_verbs_cpu_proxy_enabled(qp_state->gpu_qp, &enabled);
    if (r != DOCA_SUCCESS)
        return -10 - (int) r;
    return (int) enabled;
}

int DocaRdmaEndpoint::cpu_proxy_progress(std::string const& peer_name)
{
    int idx = find_peer_index(peer_name);
    if (idx < 0 || idx >= (int) doca_state_->peer_qps.size())
        return -1;
    auto& qp_state = doca_state_->peer_qps[idx];
    if (qp_state->gpu_qp == nullptr)
        return -2;

    bool debug_trace = read_env_enabled("TLLM_RDMA_DEBUG_TRACE", false);
    if (!debug_trace)
    {
        return (int) doca_gpu_verbs_cpu_proxy_progress(qp_state->gpu_qp);
    }

    // Diagnostic: when *qp_cpu[0] (producer counter) transitions to non-zero,
    // dump the relevant qp_cpu fields. This path is intentionally opt-in
    // because CPU proxy progress may run in a tight loop.
    static thread_local uint64_t dbg_last_v0 = (uint64_t) -1;
    static thread_local int dbg_events = 0;
    static thread_local int dbg_cap_transitions = 200;
    static thread_local int dbg_cap_posts = 400;
    // Periodic v0 snapshot (every ~200 ms) regardless of transitions —
    // captures cases where the producer counter rises rapidly past the
    // values we already printed.
    static thread_local auto dbg_last_snapshot_t = std::chrono::steady_clock::now();
    static thread_local int dbg_snapshot_count = 0;
    unsigned char* p = reinterpret_cast<unsigned char*>(qp_state->gpu_qp);
    uint64_t v0_before = 0;
    if (p)
    {
        void* p0 = *reinterpret_cast<void**>(p + 0);
        v0_before = p0 ? *reinterpret_cast<uint64_t volatile*>(p0) : 0;
        if (v0_before != dbg_last_v0 && dbg_events < dbg_cap_transitions)
        {
            uint32_t qpn_ds = *reinterpret_cast<uint32_t*>(p + 0x2c);
            unsigned char b30 = *(p + 0x30);
            void* dbr_addr = *reinterpret_cast<void**>(p + 0x20);
            uint32_t dbr_val = dbr_addr ? *reinterpret_cast<uint32_t volatile*>(dbr_addr) : 0;
            void* uar_addr = *reinterpret_cast<void**>(p + 0x18);
            std::fprintf(stderr,
                "[progress transition] prod=%lu qpn_ds@0x2c=0x%08x [0x30]=%u UAR=%p DBR=%p (*DBR=0x%08x)\n",
                (unsigned long) v0_before, qpn_ds, (unsigned) b30, uar_addr, dbr_addr, dbr_val);
            dbg_last_v0 = v0_before;
            dbg_events++;
        }
    }
    doca_error_t r = doca_gpu_verbs_cpu_proxy_progress(qp_state->gpu_qp);
    // Periodic snapshot of v0 (the SQ producer counter) every ~200 ms,
    // regardless of whether we caught a transition. Helps catch the
    // "rises but we missed it" case.
    if (p && dbg_snapshot_count < 200)
    {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - dbg_last_snapshot_t).count() >= 200)
        {
            void* p0_now = *reinterpret_cast<void**>(p + 0);
            uint64_t v0_now = p0_now ? *reinterpret_cast<uint64_t volatile*>(p0_now) : 0;
            uint64_t last_seen_now = *reinterpret_cast<uint64_t*>(p + 0x10);
            void* dbr_addr2 = *reinterpret_cast<void**>(p + 0x20);
            uint32_t dbr_val2 = dbr_addr2 ? *reinterpret_cast<uint32_t volatile*>(dbr_addr2) : 0;
            std::fprintf(stderr, "[progress snapshot] v0=%lu last_seen=%lu *DBR=0x%08x\n", (unsigned long) v0_now,
                (unsigned long) last_seen_now, dbr_val2);
            dbg_last_snapshot_t = now;
            dbg_snapshot_count++;
        }
    }
    if (p && v0_before != 0 && dbg_events < dbg_cap_posts)
    {
        void* dbr_addr = *reinterpret_cast<void**>(p + 0x20);
        uint32_t dbr_val = dbr_addr ? *reinterpret_cast<uint32_t volatile*>(dbr_addr) : 0;
        std::fprintf(stderr, "[progress post] r=%d *DBR=0x%08x last_seen=%lu\n", (int) r, dbr_val,
            (unsigned long) *reinterpret_cast<uint64_t*>(p + 0x10));
        dbg_events++;
    }
    return (int) r;
}

int DocaRdmaEndpoint::poll_send_cq_cpu(std::string const& peer_name, uint64_t* out, int max_out)
{
    int idx = find_peer_index(peer_name);
    if (idx < 0 || idx >= (int) doca_state_->peer_qps.size())
        return -1;
    auto& qp_state = doca_state_->peer_qps[idx];
    if (!qp_state->send_cq || !qp_state->send_cq->umem_gpu_ptr)
        return -2;

    // CQ memory lives on GPU; copy to host buffer and walk CQEs.
    uint32_t ncqe = qp_state->send_cq->size;
    if (ncqe == 0)
        return 0;
    size_t bytes = (size_t) ncqe * sizeof(struct mlx5_cqe64);
    std::vector<uint8_t> host_buf(bytes);
    cudaError_t cerr = cudaMemcpy(host_buf.data(), qp_state->send_cq->umem_gpu_ptr, bytes, cudaMemcpyDefault);
    if (cerr != cudaSuccess)
        return -3;
    auto* cqes = reinterpret_cast<struct mlx5_cqe64*>(host_buf.data());
    int n = 0;
    for (uint32_t i = 0; i < ncqe && n < max_out; ++i)
    {
        uint8_t op_own = cqes[i].op_own;
        uint8_t opcode = (op_own & 0xF0) >> 4;
        uint8_t owner = op_own & 0x1;
        // Anything that's not the initial (invalid / hw-owner) byte 0x00 nor 0xFF tells us NIC touched it.
        // Pack opcode | owner | byte_cnt for upstream interpretation.
        if (op_own != 0x00 && op_own != 0xF0 && op_own != 0xFF)
        {
            uint32_t byte_cnt = __builtin_bswap32(cqes[i].byte_cnt);
            uint32_t wqe_ctr = __builtin_bswap16(cqes[i].wqe_counter);
            uint64_t pack = ((uint64_t) opcode << 56) | ((uint64_t) owner << 48) | ((uint64_t) (wqe_ctr & 0xFFFF) << 32)
                | (byte_cnt & 0xFFFFFFFFu);
            out[n++] = pack;
            // Pretty-print this CQE; if it's an error CQE (REQ_ERR=13, RESP_ERR=14),
            // also decode the mlx5_err_cqe fields (syndrome / vendor_err_synd).
            unsigned char* rb = host_buf.data() + (size_t) i * sizeof(struct mlx5_cqe64);
            std::fprintf(stderr, "[send_cq cqe%u] op_own=0x%02x opcode=%u owner=%u wqe_counter=%u byte_cnt=%u\n", i,
                op_own, opcode, owner, wqe_ctr, byte_cnt);
            if (opcode == 13 /*MLX5_CQE_REQ_ERR*/ || opcode == 14 /*MLX5_CQE_RESP_ERR*/)
            {
                // err_cqe: byte 54=vendor_err_synd, byte 55=syndrome, bytes 56..59=s_wqe_opcode_qpn (BE)
                uint8_t vendor_synd = rb[54];
                uint8_t syndrome = rb[55];
                uint32_t wqe_op_qpn_be;
                std::memcpy(&wqe_op_qpn_be, rb + 56, 4);
                uint32_t wqe_op_qpn = __builtin_bswap32(wqe_op_qpn_be);
                std::fprintf(stderr, "[send_cq cqe%u ERR] syndrome=0x%02x vendor=0x%02x s_wqe_op_qpn=0x%08x\n", i,
                    syndrome, vendor_synd, wqe_op_qpn);
            }
        }
    }
    return n;
}
