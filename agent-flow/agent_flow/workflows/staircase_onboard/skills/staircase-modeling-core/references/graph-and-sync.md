# Graph safety and sync-free rules for a target forward

> Established on B200 (`sm_100`). The
> three-class taxonomy and the sync-free rewrites are properties of CUDA graph
> capture and of torch, not of any particular engine build; what an engine
> change can move is which values the engine hands the step, so re-check the
> step-args classification rather than this file's rules.

These are **writing** invariants, not tuning advice. A forward that violates
them is wrong the first time it is captured — which is why they live here,
where the assembler writing the first version and the tuner editing it later
read the same copy.

## Three constraints that shape every forward

| Constraint | What it forbids |
|---|---|
| thop custom ops are permanently invisible to `torch.compile` | no compile-based structure in the forward, and no compile-fusion reasoning about a trtllm-backed call |
| **the KV red line** — K/V are written *inside* the attention op; the target only forwards the address book | nothing in the forward may take ownership of KV, move it, or reshape it |
| the hot path reads weights **by reference** | a per-step weight transform is a bug; derived state belongs in `post_load_weights` |

## What a replayed forward may contain

Decode replay executes captured kernels only. Classify every value the
step consumes — each target's `step_args` is the single audit point:

| Class | Meaning | Replay behavior |
|---|---|---|
| reference | engine-owned persistent buffer, refreshed in place each step | replay reads fresh contents automatically (all step-args tensors) |
| GPU-derived | produced by captured kernels | recomputed on replay |
| host-derived | Python scalar baked into the capture | frozen — legal only as a per-capture constant (e.g. `num_contexts == 0` in a decode graph) |

What breaks replay or stalls the host:

| Pattern in the step path | Why it is broken |
|---|---|
| `.item()` / `.cpu()` / `.tolist()` / `print(tensor)` | host-device sync (cuStreamSynchronize); under capture the read value is frozen anyway |
| Python `if`/`while` on tensor **values** | syncs to read the value, and the branch taken is frozen into the graph — branch on tensor metadata or per-capture constants only |
| host-side shape math from tensor values (`nonzero`, `masked_select`, slice bounds read from CUDA tensors) | data-dependent output size forces a sync and bakes one shape into the capture |
| `torch.tensor(py_obj, device='cuda')` built per step | blocking host-to-device creation; make it a load-time constant in `post_load_weights` |
| allocation sized from tensor **values** | needs the value on host — sync |
| allocation sized from tensor **metadata** (`empty([x.shape[0], ...])`) | fine — the caching allocator serves stable blocks under capture; upstream captures the same pattern |

## Sync detection

```python
torch.cuda.set_sync_debug_mode("warn")  # or "error" to hard-fail on sync
# ... run the step under test ...
torch.cuda.set_sync_debug_mode(0)
```

This catches only syncs going through torch's wrapped APIs — thop ops and
libraries calling CUDA sync APIs directly are invisible to it. For those:

```bash
nsys profile --capture-range=cudaProfilerApi --python-sampling=true \
             --backtrace=dwarf <cmd>
```

then search the CUDA API row for `cudaStreamSynchronize` /
`cudaEventSynchronize` / `cudaDeviceSynchronize`; the call stack names the
Python line. Verify a fix by re-running detection, not by reading the diff.

Classify before fixing: **false dependencies** (debug prints, leftover
`.item()`, per-step tensor creation from Python objects) are deleted or
moved to load time; **true dependencies** (the CPU genuinely needs a GPU
value) are restructured onto the GPU or kept outside the captured region.

## Sync-free quick table (inference-forward rows)

| Sync-inducing pattern | Sync-free shape |
|---|---|
| `.item()` in a conditional / `if gpu_tensor:` / `torch.is_nonzero(t)` | GPU-side select (`torch.where`-shaped) or branch on metadata |
| Python `max(a, b)` on GPU tensors | `torch.max`-shaped GPU op |
| `torch.tensor(obj, device='cuda')` / `torch.as_tensor(..., device='cuda')` | precompute at load time; if truly per-step, create on CPU and `.to('cuda', non_blocking=True)` |
| `x_gpu[idx_cpu]` / `x_gpu[python_list]` | index with a resident GPU index tensor |
| `x[i:j]` with bounds read from CUDA tensors | metadata-derived bounds, or a GPU `arange`-shaped index |
| `x_gpu[mask_gpu]` / `torch.masked_select(x, mask)` | fixed-shape `torch.where(mask, x, 0)` |
| `torch.nonzero(mask)` | `where`-shaped rewrite, or move outside the captured region |
| `torch.repeat_interleave(x, r)` | pass `output_size=N` when known |
| `torch.unique(x)` | avoid in the step path; precompute |

The right-hand column names torch shapes, not licenses: in a target
forward the actual call must be a catalog entry. A replacement with no
entry in
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml` is a
vocabulary gap — never paste raw torch math
into the forward to work around a sync.
