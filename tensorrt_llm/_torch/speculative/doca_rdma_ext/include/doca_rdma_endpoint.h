#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Configuration for RDMA QP parameters, all tunable via environment variables.
// These should match the settings on the LPU side (lib_verbs_server).
struct RdmaConfig
{
    // RTR state parameters
    uint32_t path_mtu = 512;         // GLUE_RDMA_PATH_MTU (256, 512, 1024, 2048, 4096)
    uint32_t gid_index = 0;          // GLUE_RDMA_GID_INDEX
    uint8_t max_dest_rd_atomic = 16; // GLUE_RDMA_MAX_DEST_RD_ATOMIC
    uint8_t min_rnr_timer = 1;       // GLUE_RDMA_MIN_RNR_TIMER (0-31)

    // RTS state parameters
    uint8_t timeout = 14;       // GLUE_RDMA_TIMEOUT (0-31)
    uint8_t retry_cnt = 7;      // GLUE_RDMA_RETRY_CNT (0-7)
    uint8_t rnr_retry = 1;      // GLUE_RDMA_RNR_RETRY (0-7, 7=infinite)
    uint8_t max_rd_atomic = 16; // GLUE_RDMA_MAX_RD_ATOMIC
};

struct QpInfo
{
    uint32_t qpn;
    uint32_t psn;
    uint16_t lid;
    uint8_t gid[16];
};

struct BufferInfo
{
    uint64_t addr; // DOCA-allocated address (used for RDMA)
    uint32_t rkey;
    uint32_t lkey;
    size_t length; // Logical size (user-requested)
    // New fields for doca_buf_arr approach
    int buf_arr_index;      // Index in buf_arr_regs (-1 if using fallback)
    uint64_t gpu_buf_arr;   // doca_gpu_buf_arr* for kernel access
    uint64_t original_addr; // Original PyTorch tensor address (for sync back)
};

struct RemotePeer
{
    std::string name;
    QpInfo qp_info;
    BufferInfo send_buf;
    BufferInfo recv_buf;
    int sock_fd;
};

#ifdef __cplusplus
extern "C"
{
#endif

    // Forward declarations for DOCA types (opaque pointers for header)
    struct doca_dev;
    struct doca_gpu;
    struct doca_gpu_dev_verbs_qp;
    struct doca_gpu_dev_verbs_cq;

#ifdef __cplusplus
}
#endif

class DocaRdmaEndpoint
{
public:
    DocaRdmaEndpoint();
    ~DocaRdmaEndpoint();

    // Initialization
    bool init(int gpu_id, std::string const& nic_name);
    void cleanup();

    // Buffer registration
    BufferInfo register_buffer(void* gpu_ptr, size_t length);
    void deregister_buffer(BufferInfo const& buf);

    // Connection management
    bool listen(int port);
    bool accept_peer(int port, std::string const& peer_name);
    bool connect_to_peer(std::string const& host, int port, std::string const& peer_name);
    bool exchange_buffer_info(std::string const& peer_name, BufferInfo const& local_send, BufferInfo const& local_recv);

    // Get device pointers for CUDA kernels
    void* get_qp_dev_ptr(std::string const& peer_name);
    void* get_send_cq_dev_ptr(std::string const& peer_name);
    void* get_recv_cq_dev_ptr(std::string const& peer_name);

    // Get GPU buf_arr handle for buffer (for use with doca_gpu_dev_buf_* functions)
    void* get_buf_arr_gpu_handle(int buffer_index);

    // Get remote buffer info for RDMA operations
    BufferInfo get_remote_recv_buf(std::string const& peer_name);
    BufferInfo get_local_send_buf(std::string const& peer_name);

    // Post receive work requests (for WRITE_WITH_IMM reception)
    bool post_recv(std::string const& peer_name, void* buf, size_t length, uint32_t lkey);

    // Signal that GPU is ready to receive (after posting recv WRs)
    bool send_ready_signal(std::string const& peer_name);

    // Allocate DOCA GPU memory (for testing if DOCA-allocated memory works better)
    BufferInfo alloc_doca_gpu_buffer(size_t length);

    // High-level RDMA operations (host-side, for testing)
    bool rdma_write_with_imm(std::string const& peer_name, void* local_buf, size_t length, uint32_t imm_value);
    bool poll_for_completion(std::string const& peer_name, uint32_t* imm_value);

    // CPU_PROXY mode: enabled-flag query + progress tick (call from a CPU thread)
    int cpu_proxy_enabled(std::string const& peer_name);
    int cpu_proxy_progress(std::string const& peer_name);
    // Poll the send CQ from the CPU side; returns # completions found, -1 on error.
    // Writes back up to max_out completion words: bits [63:32]=opcode, [31:0]=syndrome|wqe_idx.
    int poll_send_cq_cpu(std::string const& peer_name, uint64_t* out, int max_out);

    // Accessors
    std::string const& get_local_name() const
    {
        return local_name_;
    }

    bool is_initialized() const
    {
        return initialized_;
    }

private:
    bool initialized_;
    std::string local_name_;
    int gpu_id_;
    std::string nic_name_;
    RdmaConfig rdma_config_;

    struct DocaState;
    struct CqState;
    struct QpState;
    std::unique_ptr<DocaState> doca_state_;

    // Connected peers
    std::mutex peers_mutex_;
    std::vector<RemotePeer> peers_;

    bool setup_doca_devices();
    bool setup_verbs_context();
    std::unique_ptr<CqState> create_cq(uint32_t ncqe);
    std::unique_ptr<QpState> create_qp();
    bool create_qp_for_peer(RemotePeer& peer);
    bool connect_qp(RemotePeer& peer);
    bool exchange_qp_info(int sock_fd, QpInfo& local, QpInfo& remote);
    int find_peer_index(std::string const& peer_name);
};

// CUDA kernel declarations (extern "C" for CUDA linkage)
#ifdef __CUDACC__
extern "C"
{

    __global__ void rdma_write_with_imm_kernel(doca_gpu_dev_verbs_qp* qp, void* local_buf, uint32_t lkey,
        uint64_t remote_addr, uint32_t rkey, size_t size, uint32_t imm_value, uint64_t* wqe_idx_out);

    __global__ void poll_send_cq_kernel(doca_gpu_dev_verbs_cq* cq, uint64_t expected_wqe_idx, int* completed);

    __global__ void poll_recv_cq_for_imm_kernel(
        doca_gpu_dev_verbs_cq* cq, uint64_t* cq_idx, uint32_t* imm_value_out, int* completed);
}
#endif
