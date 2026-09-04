# Third-party notices

This package is vendored from
[`github.com/NVlabs/Sana`](https://github.com/NVlabs/Sana), branch
[`sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine), at commit
[`5fe5feb`](https://github.com/NVlabs/Sana/commit/5fe5feb) (2026-08-17;
best-effort reconstruction from vendoring-date file timestamps and upstream
commit history, not an exact recorded pin from the original port -- see the
pull request for how this was verified). Checked against the current branch
tip ([`83e54df`](https://github.com/NVlabs/Sana/commit/83e54df), 2026-08-20)
on 2026-08-27; the only upstream change since that touches this subset is a
merge whose SM89 work is out of scope here. The rest of that merge (MPS/Metal
Apple Silicon backend, RTX 4090/5090 configs) is likewise out of scope for
this CUDA/Blackwell-only subset.

**Note for future currency checks.** These files are linted and formatted to
this repository's style (`ruff check` and `ruff format`, line length 100)
rather than kept byte-identical to upstream, so a direct `diff` against
upstream shows formatting noise as well as real changes. Upstream wraps at
roughly 80 columns; most of the difference is expressions joined onto one
line. To compare semantics, run `ruff format` over the upstream copy first and
diff the normalized results -- that is how the currency check above was done.

## Scope of the vendored subset

Only the pieces needed for the architectures TensorRT-LLM ships are carried:

| Carried | Not carried |
|---|---|
| `interface.py`, `preprocess.py`, `common/` | `sm89/`, `sm90/` (incl. `sm90/_compat/`) |
| `sm100/` (B200 / GB200) | `triton_ref/` Triton reference attention |
| | `sm120/` (RTX Blackwell) |
| | `_vendor/flash_attn/` (see below) |

The upstream package vendored a copy of FlashAttention's CuTe DSL helpers
under `sol_attn/_vendor/flash_attn/cute/`. That copy is **not** carried here:
TensorRT-LLM already depends on
[`flash-attn-4`](https://github.com/Dao-AILab/flash-attention) (pinned in
`requirements.txt`), which provides the same `flash_attn.cute` modules, and
the SM100 kernels import them from that dependency directly. This was
verified on B200 to produce bit-identical output to the vendored copy across a
shape/tau sweep. FlashAttention's BSD-3-Clause license is retained at
`sol_attn/sm100/LICENSE.flash-attention` because portions of the SM100 design
scaffold still derive from that project.

`preprocess.py` implements the routing/threshold stage in Triton, so Triton is
a required runtime dependency on every Sol-Attn path, not only a fallback.

## A derived file outside this directory

`../sol_attn_backend.py` is **not** part of the vendored package above, but it
is a derivative work and is recorded here because this file is where a future
currency check starts. It is adapted from upstream's
`techniques/sparse_backends/sol_attn_backend.py` (same branch and commit as the
package). Only the kernel-wrapper subset is carried -- the shape/dtype guard,
the dense fallback (routed to cute_dsl_fmha_fwd here, not torch SDPA), and the
call counters. Upstream's model-integration
half is not carried: the diffusers self-attention dispatch hook, HunyuanVideo's
padded `[video, text]` MMDiT handling, and model-level Morton ordering.

Deliberate divergences from upstream in that file, all of which a re-sync must
preserve rather than overwrite:

| Divergence | Why |
|---|---|
| `logger.warning_once` replaces `print()` | fallbacks must be suppressible and routed through the repo's logger |
| `dense_fallback_calls` counter added | makes a silently-degraded run countable, not just visible in stderr |
| `sol_attn_ineligible_reason()` added | names the specific reason (arch / head_dim / dtype) instead of one boolean |
| `SOL_ATTN_STRICT=1` also covers the eligibility path | upstream raises only on kernel exceptions, so an ineligible run stayed silent |
| `@torch.compiler.disable` on `_run_sol_attn_bthd` | see below |
| dense paths routed to `cute_dsl_fmha_fwd` via `dense_fn` | upstream's dense fallback is torch SDPA; staying in-backend is what makes a `backend: CUTEDSL` A/B isolate sparsity |

**Upstream solves the `torch.compile` problem differently, and arguably
better.** Its `sol_attn_backend.py` wraps the same call in a
`torch.library.custom_op` (`sana_sol_attn::self_attention`) with a
`register_fake` returning `torch.empty_like(q)`, which keeps the kernel in the
compiled graph as an opaque node instead of breaking the graph at it; a second
consumer (`models/ltx2.5-refiner/GB200/sol_attention.py`) applies
`torch.compiler.disable` at the call site behind a flag. This repository uses
`@torch.compiler.disable` on the launch boundary instead, matching the
convention every other CuTe DSL entry point here already follows
(`attention_backend/cute_dsl/fmha.py`,
`cute_dsl_kernels/blackwell/video_sparse_attention/interface.py`). Without some
such guard Dynamo traces into the CuTe DSL JIT builder and retraces on every
call -- measured at near two orders of magnitude slower on B200. Migrating to
the `custom_op` form would
remove the per-layer graph break and is a reasonable follow-up; it was not done
here because the `torch.compiler.disable` form is what this repository's other
kernels use and what the measurements above were taken with.

The runtime also depends on NVIDIA CUTLASS / CuTe DSL, cuda-python, and
PyTorch. Those dependencies are not redistributed by this repository and
remain subject to their respective licenses.

SM120 (RTX Blackwell) was carried in an earlier revision of this port and has
been dropped: it had kernel-level evidence only, no end-to-end validation, and
no `cute_dsl_fmha_fwd` exists for that architecture, so Sol-Attn's dense
fallback could not match its own backend there. With SM100 alone, Sol-Attn's
architecture set is a subset of the dense CuTe DSL FMHA kernel's. The
cuDNN-frontend attribution that covered the SM120 execution skeleton was
removed with it.
