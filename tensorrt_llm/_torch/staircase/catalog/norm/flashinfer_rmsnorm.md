---
receipts:
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 126}
---

<!--
The `sm_100` key was REMOVED, not lost. It recorded a 1.3.0rc21 run against a
contract and test file that no longer exist: both were rewritten on 2026-09-14
to add the V4.1 column, so that receipt predated every file in the entry and a
receipt that predates its own test file is exactly what the freshness rule
exists to catch. No sm_100 device was available to rerun the current files on,
and "key absent = unknown" is the honest state. It is not a claim that the op
fails there.
-->


# flashinfer_rmsnorm

**Wraps** `torch.ops.trtllm.flashinfer_rmsnorm` (one call).

## Semantics

Root-mean-square normalization over the last dimension, with elementwise
weight scaling:

```
out[..., i] = x[..., i] / sqrt(mean(x[..., :]^2) + eps) * weight[i]
```

The squared-mean reduction and normalization are accumulated in fp32
inside the kernel; the result is cast back to the input dtype.

Fusion boundary: the single call computes normalization and weight scaling
only. There is no residual add (see `flashinfer_fused_add_rmsnorm` for
that), no `(1 + weight)` gemma-style scaling (see `flashinfer_gemma_rmsnorm`),
and no quantization. The caller owns everything else.

## Signature

```python
def flashinfer_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[num_tokens, hidden]` or `[batch, num_heads, head_dim]` | fp16 / bf16 / fp32 | last-dim stride must be 1; non-contiguous row strides are handled | CUDA |
| `weight` | `[hidden]` (or `[head_dim]` for 3D input) | same as `x` | contiguous | CUDA (same device as `x`) |
| `eps` | scalar | Python float | — | — |
| returns | same shape as `x` | same as `x` | newly allocated | CUDA |

The output is a new tensor (`x` is not mutated).

## Certified coverage

What the receipt covers, exactly. Every row is one or more cases in
`tests/unittest/_torch/staircase/norm/test_staircase_flashinfer_rmsnorm.py`;
the receipt is that file passing on sm_103.

| axis | certified values | cases |
|---|---|---|
| **DeepSeek-V4.1-Flash widths x rows** | **the full cross-product**: `{5120, 1280, 512, 128}` x 17 row buckets, bf16, at `eps = 1e-20` | 68 |
| V4.1 widths | `5120` attn_norm / ffn_norm / final norm (`args.dim`), `1280` q_norm (`q_lora_rank`), `512` kv_norm and the per-layer compressor norm (`head_dim`), `128` indexer k_norm (`index_head_dim`) | — |
| row buckets | 1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096 | — |
| dtypes | bf16 / fp16 / fp32, each V4.1 width, **`M=64`, `eps = 1e-20`** | 12 |
| epsilon magnitude | each V4.1 width x constant-row `c` in `{1e-9, 1e-10, 1e-11, 1e-12}`, **`M=8`, bf16, `eps = 1e-20`** | 16 |
| degenerate rows | all-zero rows at each width, **`M=4`, bf16 and fp32, `eps = 1e-20`**, plus one zeroed row inside an ordinary batch (**width 5120, `M=8`, bf16**) | 8 + 1 |
| PDL | on vs off at each V4.1 width, **`M=129`, bf16, `eps = 1e-20`** | 4 |
| strided rows | a `[129, 1280]` column slice of a `[129, 8192]` buffer, **bf16, `eps = 1e-20`** | 1 |
| discrimination | sum-instead-of-mean control, **width 5120, `M=129`, bf16, `eps = 1e-20`** | 1 |
| dtype domain | 8 input dtypes, **width 512, `M=4`, `eps = 1e-20`** — see the Preconditions table for each observed result | 8 |
| fp8 caller hazard | both fp8 types accepted, **width 512, `M=4`, `eps = 1e-20`** | 1 |
| pre-existing column | eps `1e-5`/`1e-6`, widths 4096/5120/2048/128/111/1152, 2D and 3D, fp16/fp32, strided | 6 |
| total | | **126**, all passing |

Every row above except the first is a **single-shape** measurement at the `M`,
dtype and epsilon stated in it, and is not claimed to generalize to another
geometry. Only the cross-product row spans row counts.

Uncertified and therefore not claimed: widths outside the list above at
`eps = 1e-20`, `M > 4096`, epsilons strictly between `1e-20` and `1e-6`, and any
arch other than sm_103 — including sm_100, whose historical receipt was removed
rather than carried forward (see the frontmatter comment).

### Every V4.1 norm input is bf16

All four placements sit on the bf16 text path, so bf16 is the only dtype this
checkpoint actually feeds the op; fp16 and fp32 are extra coverage kept for the
entry's own robustness, not target semantics.

This is worth stating because the obvious reading of the source says otherwise.
`Compressor.__init__` builds `wkv` (and `wgate`) with
`dtype=torch.float32` when `compress_ratio > 1`, which looks like an fp32 norm
input. It is not: `Compressor.forward` captures `dtype = x.dtype` on entry,
does the pooling in fp32, and returns `self.norm(kv.to(dtype))` — and the
`ratio == 1` branch returns `self.norm(self.wkv(x))` on a bf16 `wkv`. **The
norm sees bf16 on both branches.** Reading the declaration without the call site
is what produced the opposite claim in an earlier revision of this contract.

### `eps = 1e-20` is honoured, and that is measured rather than assumed

This checkpoint sets `rms_norm_eps = 1e-20`, fourteen orders below anything this
entry was previously certified at. Epsilon is the one term here a kernel can
silently drop, clamp to a minimum, or lose to a downcast in the binding, **and
no ordinary activation would reveal it**: with `mean(x^2)` of order 1, an
epsilon of 1e-20 and an epsilon of 0 give bit-identical results. The certified
column therefore contains two independent witnesses that it reaches the kernel:

* **Magnitude.** Rows of a constant `c` make `mean(x^2) = c^2` exactly, and `c`
  is walked down through 1e-10 where `c^2` meets eps. The kernel matches an
  eps-honouring fp32 reference **exactly (`0.000e+00`)** at every width and every
  `c`, while an eps-dropping reference diverges by `3.91e-03` at `c=1e-9`,
  `2.93e-01` at `c=1e-10`, `9.00e-01` at `c=1e-11` and `9.90e-01` at `c=1e-12`.
  The test asserts both directions, and refuses to pass a `c` where the two
  references are closer than `1e-3` — a case that cannot discriminate must not
  report a pass.
* **Degenerate rows.** An all-zero row is `0 * rsqrt(0 + eps)`. With epsilon it
  is `0`; with `eps = 0` it is `0 * inf` = `NaN`, which would propagate through
  the whole forward. Measured `nan=0, inf=0, max_abs=0` at every width in bf16
  and fp32, and a zeroed row inside an ordinary batch leaves its neighbours
  bit-identical.

**What is NOT separable at this epsilon.** Applying epsilon to the norm instead
of to the mean square — `x / (sqrt(mean) + eps)` rather than
`x * rsqrt(mean + eps)` — is indistinguishable on ordinary data at
`eps = 1e-20`: the domain probe measures **0** elements outside the default gate
for that variant at every width and row count. That is by construction, not by
luck, and it is recorded here so nobody reads the clean tolerance table as
evidence about epsilon placement. Placement is separable only in the
constant-`c` regime above, which is the other reason that regime is in the
certified column.

### The tolerance the receipt is measured at

`torch.testing.assert_close`'s **defaults for the dtype, unloosened**, both rtol
and atol, everywhere in the file. That is measured: across every V4.1 width at
`M=1` and `M=4096` in bf16 and fp32, **0 elements** lie outside the default
elementwise bound, with the worst correct error `1.56e-02` on a tensor of scale
`15.1`.

**Discrimination.** Reducing with `sum` instead of `mean` — same shape, same
values, same epsilon, off by `sqrt(width)` — puts **`660,410` of `660,480`**
elements outside that same default bound at `(5120, M=129)`, with a max-error
ratio of `729.6x`. Not literally every element: `70` survive, which is what a
`sqrt(width)` scaling error looks like where the correct value is already near
zero. The unloosened gate sees the error at every other measured point too —
5,120 / 20,970,344 / 524,270 elements outside across the probe's table.

## Metadata consumed

None. Stateless.

## Preconditions

- `x` is 2D or 3D on a CUDA device; the normalized dim is the last one.
- `weight.shape == (x.shape[-1],)` and `weight.dtype == x.dtype`.
- `x.stride(-1) == 1` (rows may be strided, e.g. a column slice of a wider
  buffer; the kernel selects a strided code path in that case).
- **The dtype domain, every case driven on sm_103 and asserted by
  `test_dtype_domain`.** The previous version of this bullet was inherited from
  a 2026-07-28 note and one of its six claims is false on this install, so it is
  replaced by what was measured:

  | input dtype | observed |
  |---|---|
  | `bfloat16` / `float16` / `float32` | accepted, correct (the certified path) |
  | `float64` | `KeyError: torch.float64` |
  | `int8` | `KeyError: torch.int8` |
  | `uint8` | `KeyError: torch.uint8` |
  | `float8_e5m2` | **accepted** — returns a `float8_e5m2` tensor, `4.73e-01` from an fp32 reference on a scale of `7.53` |
  | `float8_e4m3fn` | **accepted** — returns a `float8_e4m3fn` tensor, `2.11e-01` from an fp32 reference on a scale of `6.79` |

  Those two distances are seeded (`manual_seed(2026)`, width 512, `M=4`) and the
  probe and the GPU test print the same numbers. An earlier revision quoted
  unseeded values, which moved between runs.

  **`float8_e5m2` was documented as raising `KeyError` and does not.** That is
  the expensive direction to be wrong in: the old bullet promised a guard the
  caller would then not write. Both fp8 types are accepted and normalize the
  stored fp8 values *as if they were the real numbers* — no dequantization —
  so a caller who forgets to dequantize gets a plausible-looking answer, not an
  error. This matters for this checkpoint specifically, whose dense path
  quantizes activations to MXFP8 before every projection.

  **No wrapper guard was added for it**, deliberately: two already-certified
  targets (`deepseek_v3/r1_0528_nvfp4/sm_103/dep4` and
  `gpt_oss/gpt_oss_120b/sm_103/tp1`) call this wrapper on hot paths, and adding
  an assert to a shared wrapper would change behaviour for artifacts whose gates
  are hour-scale accuracy runs I cannot re-earn from inside this Goal. The
  hazard is documented and tested here instead; a target that wants the check
  owns it at its own call site.

  The uncertified CUDA-JIT path (`FLASHINFER_USE_CUDA_NORM=1`) is a different
  code path and nothing here describes it.
- `hidden` (last dim) has no divisibility requirement: sizes not divisible
  by the 128-bit vector width (e.g. 111, 1152) fall back to a smaller
  vector size and were verified correct on this machine.

## Notes

- The op is registered only when flashinfer is importable
  (`IS_FLASHINFER_AVAILABLE`); this pinned install ships
  flashinfer-python 0.6.14.
- Programmatic dependent launch (PDL) is controlled by the env var
  `TRTLLM_ENABLE_PDL` (default enabled) inside the trtllm custom op; it affects
  scheduling only, not results. **Driven, not inherited:** at `eps = 1e-20` and
  `M=129`, PDL on and PDL off are `torch.equal` at all four V4.1 widths, and the
  test asserts it. The default is *on*, so that is the path a target actually
  runs — a contract claim about it is worth nothing untested.
- TRT-LLM's own `RMSNorm` module routes to this op only for fp16/bf16 inputs.
  fp32 also works and is covered by this entry's sm_103 tests.
- Weight scaling is plain `w * normed(x)`, not the gemma `(1 + w)` form.
