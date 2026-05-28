# PEARL CUDA IPC Transport (`transport=cudaipc`)

Same-machine alternative to `transport=shm` where the **data slots live
in GPU device memory** shared across processes via
`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`. The CPU shared-memory
header (head / tail / stop / ready / IPC handle) is kept so polling
stays a cheap CPU memory read.

Compared to the pure-CPU `shm` backend, the only practical advantage on
this workload turned out to be **architectural** (data resident on the
GPU so verify and send can share the same kernel chain). Raw tok/s is
within measurement noise of `shm` / `ibverbs` baselines because the
channel layer still issues a per-send stream sync. The real cycle
shortening will come from graph-capturing the send kernel together with
the model's forward pass -- which is left to a follow-up because it
requires deeper hooks into TRT-LLM's PyExecutor.

## Four-stage progression

The implementation arrived in four toggleable stages. Each stage was
verified byte-identical against the `ibverbs` baseline (same prompt,
same PEARL counters 2321/834, total_accept_rate=0.3593).

| Stage | Mechanism | Env | Tok/s |
|---|---|---|---|
| 1 | Host-orchestrated. `cudaMemcpyAsync(host->device, 100B)` per send. | (default) | 68.11 |
| 2 | Kernel-based send. Frame travels as kernel param (no host memcpy). | `PEARL_CUDAIPC_KERNEL_SEND=1` | 68.34 |
| 3 | Compose-and-send. Kernel reads verified `last_token` directly from the GPU verify-output tensor; CPU only hands over scalar protocol fields. | `PEARL_CUDAIPC_GPU_COMPOSE=1` | 68.43 |
| 4 | Symmetric on the draft side -- draft auto-uses stage 2 kernel send via `_CudaIpcBackend`. Going further requires keeping `draft_tokens` GPU-resident end-to-end, which is a draft-backend refactor outside the transport. | (auto) | 68.36 |

Baselines: ibverbs 68.29, shm 68.31. All deltas within measurement
noise on this workload; the value of these stages is making the next
layer of optimization (graph capture of the send kernel) tractable.

## Architecture

```
target process                                              draft process
                                                                  
[ verify sampler                                                  ]
[ → accepted_tokens (GPU)                                         ]
[                                                                 ]
[ Phase B prefetch -> pinned CPU                                  ]
[ (skipped on stage 3 hot path)                                   ]
[                                                                 ]
[ stage 3 set_next_send_gpu_tokens(                               ]
[     accepted_tokens_row.data_ptr(),                             ]
[     num_accepted_ptr.data_ptr(),                                ]
[     ...)                                                        ]
[                                                                 ]
[ channel.send_for_request(...)                                   ]
[ → _CudaIpcBackend.send                                          ]
[     → RingComposeLauncher.compose_and_send(...) ◀── reads from  ]
[         (1-thread kernel that reads GPU verify   GPU tensor    ]
[          output, assembles 100-byte frame,                      ]
[          writes to peer IPC ring slot,                          ]
[          __threadfence_system())                                ]
[                                                                 ]
[ cudaStreamSynchronize(self._stream)                             ]
[ meta_out.set_head(head + 1)         CPU shm head bump          ]
                                                                  
                                              /dev/shm/<name>_meta_t2d
                                              [head u64 | tail u64 | stop | ready | IPC handle 64B]
                                              GPU memory (256 slots * 100B) shared via cudaIpc
                                                  ┃
                                                  ▼
                                              [pearl-draft consumer reads slot]
                                              [→ DraftApiProtocol.decode(96B)]
                                              [→ next backend_step ...]
```

CPU meta region per direction (`<name>_meta_t2d` / `<name>_meta_d2t`):

```
[0:8]    head u64 (producer writes monotonically)
[8:16]   tail u64 (consumer writes monotonically)
[16:17]  stop u8
[17:24]  reserved
[24:28]  ready magic u32 (server writes last)
[28:32]  reserved
[32:96]  IPC mem handle (64 bytes -- server publishes, client opens)
[96:128] reserved
```

GPU device memory per direction: 256 slots × 100 bytes = 25 600 B.

## Source map

| File | Role |
|------|------|
| [cudaipc_endpoint.py](../../../tensorrt_llm/_torch/speculative/cudaipc_endpoint.py) | `_CudaIpcBackend` implementing the `Endpoint` protocol; CPU meta + GPU rings + IPC handshake; stages 1/2/3 send paths gated by env. |
| [pearl_ring_kernel.py](../../../tensorrt_llm/_torch/speculative/pearl_ring_kernel.py) | NVRTC-compiled CUBIN containing `pearl_write_ring_slot` (stage 2) and `pearl_compose_and_send` (stage 3); `RingWriteLauncher` / `RingComposeLauncher` Python wrappers with pre-allocated ctypes scratch for hot-path launches. |
| [ibverbs_draft_offload.py](../../../tensorrt_llm/_torch/speculative/ibverbs_draft_offload.py) | Spec dec layer that calls `set_next_send_gpu_tokens` when the endpoint supports it (stage 3 handoff). |
| [draft_rdma_server.py](draft_rdma_server.py) | Server side `_new_channel` recognizes `transport=cudaipc`. |

## Running

```bash
# Draft (any free GPU, e.g. 6)
cd /code/tensorrt_llm
DRAFT_VISIBLE_GPUS=6 TRANSPORT=cudaipc \
    bash examples/llm-api/rdma/run_pearl_ibverbs_draft_server.sh

# Target (TP=2 on free GPUs; cudaipc handles GPU0:GPU0..N P2P access)
TARGET_VISIBLE_GPUS=2,7 TP_SIZE=2 \
    ATTN_BACKEND=TRTLLM CUDA_GRAPH=1 TRACE=0 \
    TRANSPORT=cudaipc \
    bash examples/llm-api/rdma/run_pearl_ibverbs_target_one_case.sh
```

Stages 2 and 3 are on by default. Disable for A/B comparison:

```bash
# Disable stage 3 (back to stage 2)
PEARL_CUDAIPC_GPU_COMPOSE=0 ...

# Disable stage 2 + 3 (back to stage 1)
PEARL_CUDAIPC_KERNEL_SEND=0 PEARL_CUDAIPC_GPU_COMPOSE=0 ...
```

Override the CPU meta region name (mandatory for multiple PEARL pairs
on one host):

```bash
CUDAIPC_NAME=pearl_ipc_pair_a TRANSPORT=cudaipc ...
```

## What's NOT yet captured into a CUDA graph

The send kernel + meta head bump still runs **eagerly** -- one CUDA
launch + one stream sync per cycle. The original promise of stage 3
("fuse verify+send into a single graph-captured kernel chain") needs
the spec-dec offload layer to be invoked from inside the model's
graph capture region, which TRT-LLM's PyExecutor does not currently
expose. Two follow-up paths exist:

1. **Inline the launch into the model's CUDA graph.** Requires
   patching `TorchSampler` (or whichever step owns graph capture) to
   call the compose launcher as a tail kernel before the graph node
   ends. Removes the per-cycle stream sync entirely.

2. **Capture an endpoint-owned mini-graph.** Build a one-node
   `cudaGraph` containing the compose-and-send launch; replay it via
   `cudaGraphLaunch` per cycle with `cudaGraphExecKernelNodeSetParams`
   updating the kernel scalars. Smaller scope than (1) and doesn't
   touch the model executor, but the per-replay parameter update
   itself has API overhead -- needs measurement before committing.

Both are bounded follow-ups; the GPU-resident data plane delivered by
stages 1-3 is the prerequisite.

## Correctness check

```bash
# Baseline (ibverbs or shm)
TRANSPORT=ibverbs ... > target_ibverbs.log

# CUDA IPC
TRANSPORT=cudaipc ... > target_cudaipc.log

# PEARL counters + decoded text should be byte-identical (modulo wall
# clock tokens_per_sec which is noise).
diff <(grep -E "(generated_tokens|accept_rate|reuse_rate|hit_rate)" target_ibverbs.log) \
     <(grep -E "(generated_tokens|accept_rate|reuse_rate|hit_rate)" target_cudaipc.log)
```

Empty diff confirms semantic equivalence across all four stages.
