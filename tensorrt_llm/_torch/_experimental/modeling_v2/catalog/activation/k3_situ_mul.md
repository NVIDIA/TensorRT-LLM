---
receipts: {}
---

# k3_situ_mul

**Wraps** `torch.ops.trtllm.k3_situ_mul` (one call).

## Semantics

The SiTU-gated multiply of the Kimi K3 dense MLP (`SituAndMul` in `tensorrt_llm/_torch/modules/situ.py`) for at
most 8 tokens, as one CuTe DSL kernel that lets the next kernel launch at once under programmatic dependent launch.
The last dim of the input holds the gate half followed by the up half:

```
K = gu.shape[1] // 2;   g = gu[m, k],  u = gu[m, K + k]            0 <= m < M <= 8, 0 <= k < K, in fp32
a         = (beta * tanh(g / beta)) * sigmoid(g)                    sigmoid(g) = 1 / (1 + exp(-g))
v         = linear_beta * tanh(u / linear_beta)                     if linear_beta is not None, else v = u
out[m, k] = bf16(a * v)
```

The arithmetic is fp32 in this order, with IEEE division and the full-precision `exp` and `tanh`, and the result
is rounded to bf16 once, round-to-nearest-even. That is the order of `SituAndMul`'s eager path. The `exp` and
`tanh` implementations are not torch's or Triton's, so the result is not bit-identical to `SituAndMul` (eager or
fused); they agree within the tolerance below. Rows are independent: a token's output row has the same bits
whatever `M` is.

Fusion boundary: the single call computes the activation only. The caller owns the gate_up projection that
produces `gu` and the down projection that consumes the output. There is no quantization of the output.

## Signature

```python
def k3_situ_mul(gu: torch.Tensor, beta: float = 1.0, linear_beta: float | None = None) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `gu` | `[M, 2 * K]`, gate half first, `1 <= M <= 8`, `K % 8 == 0` | bf16 | contiguous, 16-byte-aligned | CUDA |
| `beta` | scalar | Python float | — | — |
| `linear_beta` | scalar or `None` | Python float, not 0 (the wrapper asserts) | — | — |
| returns | `[M, K]` | bf16 | newly allocated, contiguous | `gu`'s device |

Every element of the returned tensor is written; `gu` is not mutated.

### Certified arguments

The dense MLP of Kimi K3 at TP16 (gate_up output `[M, 4224]`, activation `[M, 2112]`) at every `M` in 1..8, with
`(beta, linear_beta)` = `(4.0, 25.0)` (the Kimi K3 checkpoint's) and `(1.0, None)` (the defaults). Inputs are
`gu ~ 2 * N(0, 1)`, rounded to bf16. The gate per cell: `max|y - ref| / max|ref| <= 8e-3` against the formula
above evaluated in fp64 on the same bf16 input, identical bits on a repeated call, and each `M`'s rows
bit-identical to the same rows of the 8-row call. Also certified, with both settings at `M` 1 and 8: calls
captured in a CUDA graph after an eager call per key, replayed with `gu` rewritten in place, return the bits of
eager calls on the new `gu`. Other widths and `beta` / `linear_beta` values are accepted but not certified.

## Metadata consumed

None. The op reads no attention metadata, KV cache or module state; every input is an argument, and the kernel
runs on the current CUDA stream of `gu`'s device.

One process-global cache sits behind it, and it is result-neutral: the compiled kernel per key
`(K, linear_beta is not None, PDL on/off)`. The first call with a new key compiles the kernel with the CuTe DSL,
which costs host time; `M`, `beta` and the value of `linear_beta` are runtime arguments and never recompile. That
first call refuses to run under CUDA-graph capture: it raises `RuntimeError` ("run once per shape outside
CUDA-graph capture first") before launching anything. Call every key once eagerly, then capture; captured and
later calls reuse the compiled kernel. The PDL setting is read from `TRTLLM_ENABLE_PDL` on every call, so
changing it within a process adds a key.

## Preconditions

The op checks these and raises `ValueError` ("k3_situ_mul: unsupported call ...") when one fails:

- `gu` is a 2-D contiguous CUDA bf16 tensor with `1 <= M <= 8` rows.
- `gu.shape[1] % 16 == 0`, so each half is a whole number of 16-byte vectors.
- `gu` starts at a 16-byte-aligned address.

Certified refusals: 0 and 9 rows, a width of 4216, a `gu` starting 2 bytes past a 16-byte boundary, fp16, a
row-strided `gu` and a 1-D `gu`.

The wrapper adds one check, because the op does not fail on it: `linear_beta` must not be `0.0`. The op hands the
kernel `linear_beta or 1.0`, so `linear_beta=0.0` would run as `1.0` (`v = tanh(u)`) with no error, where the
formula gives `0 * tanh(u / 0)`. The wrapper raises `AssertionError` before calling the op (certified).

Not checked by the op:

- `K >= 8`; an empty `gu` (width 0) passes the op's check.
- The architecture. The kernel needs no SM 10.x feature (no tcgen05, no clusters; `griddepcontrol` needs SM 9.0
  or newer), but only sm_100 has been measured.
- The CuTe DSL (`cutlass`) and `cuda-python` (`cuda.bindings`) are importable. The op imports the kernel module
  after its check, so without them a call raises `ImportError`.

## Notes

- Programmatic dependent launch (PDL). With `TRTLLM_ENABLE_PDL` unset or `1` the kernel is launched with PDL; any
  other value launches it without. There is no `trigger_early` argument: every CTA executes
  `griddepcontrol.launch_dependents` first, then `griddepcontrol.wait`, and only then reads `gu`. So the next
  kernel on the stream, if launched with PDL, launches at once and can stream its own weights while this kernel
  and its predecessor run (the CTM GEMVs do); it must execute `griddepcontrol.wait`
  (`cudaGridDependencySynchronize`) before it reads the output, as those GEMVs do before they read their
  activation. Kernels launched without PDL, torch's included, start after this one completes as usual. None of
  this changes results.
- Two identical calls return identical bits.
- Grid: `ceil(K / 1024)` x 8 CTAs of 128 threads, 8 columns per thread; rows past `M` exit at once.
- Only sm_100 has been measured.
