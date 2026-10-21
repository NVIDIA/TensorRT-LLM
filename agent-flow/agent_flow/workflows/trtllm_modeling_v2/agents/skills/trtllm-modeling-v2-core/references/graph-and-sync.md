# Graph safety and sync-free rules for a target forward

> Established on B200 (`sm_100`). The three-class taxonomy is a property of
> CUDA graph capture, not of any particular engine build; what an engine
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

## Finding and removing a sync

Detection (`torch.cuda.set_sync_debug_mode`, and nsys for syncs that bypass
torch's wrapped APIs, which thop ops do) and the sync-free rewrites of each
pattern above are general torch practice: the repository's
`perf-torch-sync-free` and `perf-torch-cuda-graphs` skills cover them.

What is specific to a target is where a rewrite may land. A rewrite into one
native torch callable is a direct call, as Invariant 1 allows. A rewrite that
needs a computation composed of several torch calls, or a kernel, is a
vocabulary gap — an entry in the catalog or the target's `ops/` — not math
pasted into the forward to work around the sync.
