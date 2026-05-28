# PEARL Shared-Memory Transport (`transport=shm`)

Same-machine alternative to `transport=ibverbs` / `tcp` / `doca` for the PEARL
draft offload data plane.

- **No NIC, no RDMA driver required.** Two POSIX shared-memory regions
  under `/dev/shm/` carry the 100-byte protocol frames between draft and
  target processes.
- **Drop-in.** The wire frame is the existing 4-byte `imm_data` + 96-byte
  `DraftApiProtocol` payload, so `SpecDecodeChannel` and the protocol
  layer are unchanged.
- **Sub-microsecond latency.** Two SPSC ring buffers + busy-poll, no
  syscalls in the hot path. Aligned 8-byte head/tail counters are atomic
  on x86_64 TSO; no locks.
- **Reusable draft.** Unlike the ibverbs single-use handshake, one draft
  server process can serve multiple back-to-back target sessions over the
  same shared-memory pair.

## When to use

- Hardware lacks RDMA NICs (laptop, dev box, CPU-only container host).
- You want to bring up PEARL correctness without dealing with `mlx5_0`,
  gid/pkey indices, or DOCA driver versions.
- Two-process debugging on the same node (e.g., draft on GPU 7, target
  on GPU 6 / TP=2 on GPU 5,6).

Not for cross-host deployment — `/dev/shm` is host-local. Use `ibverbs`
or `doca` for those.

## How it works

```
target process                                 draft process
   ▲     │                                          ▲     │
   │ ┌───┴─── /dev/shm/<shm_name>_t2d ──────────┐   │     │
   │ │ [head u64][tail u64][stop u8][ready u32] │   │     │
   │ │ [slot 0: 100B] [slot 1] ... [slot 255]   │ ◄─┘     │
   │ └──────────────────────────────────────────┘         │ target→draft
   │                                                       │
   │ ┌──────── /dev/shm/<shm_name>_d2t ──────────┐         │
   └─┤ [head u64][tail u64][stop u8][ready u32]  │         │ draft→target
     │ [slot 0: 100B] [slot 1] ... [slot 255]    │ ◄───────┘
     └───────────────────────────────────────────┘
```

- Each ring: 64-byte header + 256 slots × 100 byte = ~26 KB.
- Server (draft) creates both regions and writes the ready magic last.
- Client (target) attaches by name and busy-polls for both regions to
  publish the magic before exchanging packets.
- Both peers explicitly `resource_tracker.unregister` their handles so a
  client process exit doesn't `unlink` the server-owned segment.

Source: [shm_endpoint.py](../../../tensorrt_llm/_torch/speculative/shm_endpoint.py).

## Run

The same draft / target scripts work — switch one env var.

### 0. Verify the host has `/dev/shm` writable

```bash
ls -ld /dev/shm
# drwxrwxrwt ...   — world-writable tmpfs, what you want
```

### 1. Start the draft server (terminal 1)

```bash
cd /code/tensorrt_llm
TRANSPORT=shm \
  bash examples/llm-api/rdma/run_pearl_ibverbs_draft_server.sh
```

Wait for `[draft-rdma] control listener: port=47331` and then `[pearl-draft]
waiting for TcpModelInit on port 47331 ...`. The control plane is still
TCP on `--control-port` (cheap, one-shot per session); only the data
plane switches to shared memory.

### 2. Run the target (terminal 2)

```bash
cd /code/tensorrt_llm
TARGET_VISIBLE_GPUS=1,2 TP_SIZE=2 \
ATTN_BACKEND=TRTLLM CUDA_GRAPH=1 TRACE=0 \
TRANSPORT=shm \
  bash examples/llm-api/rdma/run_pearl_ibverbs_target_one_case.sh
```

After it finishes you should see the same `=== PEARL draft summary ===`
counts as the ibverbs reference (byte-identical on greedy decoding with
the default prompt).

### 3. Re-running

The draft does **not** need a restart between target runs — the same
shared-memory pair is reused for the next session. Just re-launch
the target. Verified back-to-back:

| Run | warmup tok/s | perf tok/s | accept rate |
|-----|--------------|------------|-------------|
| 1   | 68.34        | 68.07      | 0.3593      |
| 2   | 68.37        | 68.28      | 0.3593      |

Compare to ibverbs baseline `perf: 68.29 tok/s, accept: 0.3593` — within
measurement noise.

## Overrides

| Env var | Default | Meaning |
|---|---|---|
| `TRANSPORT` | `ibverbs` | Set to `shm` to enable this backend. |
| `SHM_NAME` | `pearl_shm_default` | Prefix of `/dev/shm/<name>_t2d` and `<name>_d2t`. Override for multiple PEARL pairs on the same host. |
| `DRAFT_CONTROL_PORT` | `47331` | TCP control channel (model-init handshake). Still required even with `shm` data plane. |

Other knobs (`CUDA_GRAPH`, `TRACE`, `ATTN_BACKEND`, `TP_SIZE`, etc.) are
documented in the top-level [HOW_TO_TEST.md](../../../../HOW_TO_TEST.md);
they're orthogonal to the transport.

### Multiple concurrent pairs

```bash
# Pair A
SHM_NAME=pearl_pair_a TRANSPORT=shm bash ...draft_server.sh
SHM_NAME=pearl_pair_a TRANSPORT=shm DRAFT_CONTROL_PORT=47331 bash ...target...

# Pair B (different SHM_NAME, different control port)
SHM_NAME=pearl_pair_b TRANSPORT=shm DRAFT_CONTROL_PORT=47332 bash ...draft_server.sh
SHM_NAME=pearl_pair_b TRANSPORT=shm DRAFT_CONTROL_PORT=47332 bash ...target...
```

`/dev/shm/` listing will show:

```
pearl_pair_a_t2d  pearl_pair_a_d2t
pearl_pair_b_t2d  pearl_pair_b_d2t
```

## Gotchas

- **Stale segments from a crashed draft.** Server-side cleans up via
  `_create_or_recreate` (best-effort `unlink` of the same `SHM_NAME` on
  next start), but if a previous run left a stale `/dev/shm/pearl_shm_*`
  with bad header content and you can't trust it, remove it manually:
  ```bash
  rm -f /dev/shm/pearl_shm_default_t2d /dev/shm/pearl_shm_default_d2t
  ```

- **Two different `SHM_NAME` values between draft and target → handshake
  hang.** The target retries for `handshake_timeout_s` (120 s default)
  then errors out. Symptom: target prints
  `shm endpoint: timed out waiting for server-side shm regions`.

- **Container `/dev/shm` size.** Default Docker `/dev/shm` is 64 MB; this
  backend uses ~52 KB per pair, so it's never the bottleneck. If you
  changed `--shm-size` you should still be fine.

- **Cross-arch portability.** SPSC counters here rely on x86_64 TSO. A
  port to ARMv8 needs release/acquire barriers — track in
  [shm_endpoint.py docstring](../../../tensorrt_llm/_torch/speculative/shm_endpoint.py).

## Correctness check

Run against ibverbs (or tcp) baseline with the same prompt and greedy
sampling; output should be byte-identical:

```bash
# Baseline (ibverbs)
TRANSPORT=ibverbs TARGET_VISIBLE_GPUS=5,6 TP_SIZE=2 ATTN_BACKEND=TRTLLM \
  CUDA_GRAPH=1 TRACE=0 bash examples/llm-api/rdma/run_pearl_ibverbs_target_one_case.sh \
  > target_ibverbs.log

# Shared memory
TRANSPORT=shm TARGET_VISIBLE_GPUS=5,6 TP_SIZE=2 ATTN_BACKEND=TRTLLM \
  CUDA_GRAPH=1 TRACE=0 bash examples/llm-api/rdma/run_pearl_ibverbs_target_one_case.sh \
  > target_shm.log

# Diff the generated text and PEARL accept rates
diff <(grep -E "(tokens_per_sec|accept_rate|generated_tokens)" target_ibverbs.log) \
     <(grep -E "(tokens_per_sec|accept_rate|generated_tokens)" target_shm.log)
```

Empty diff (except possibly the wall-clock tok/s which is noise-level)
confirms semantic equivalence.
