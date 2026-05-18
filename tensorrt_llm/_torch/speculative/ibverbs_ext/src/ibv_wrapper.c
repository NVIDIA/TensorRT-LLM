/*
 * libibverbs exposes ibv_post_send, ibv_post_recv, and ibv_poll_cq as inline
 * functions in verbs.h on some platforms. Python ctypes cannot resolve those
 * inline-only symbols from libibverbs.so, so this tiny shim exports stable
 * wrapper functions for the ibverbs endpoint backend.
 */
#include <infiniband/verbs.h>

int wrap_ibv_post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr)
{
    return ibv_post_send(qp, wr, bad_wr);
}

int wrap_ibv_post_recv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr)
{
    return ibv_post_recv(qp, wr, bad_wr);
}

int wrap_ibv_poll_cq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc)
{
    return ibv_poll_cq(cq, num_entries, wc);
}
