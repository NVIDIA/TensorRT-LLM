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
| `sm120/` (RTX Blackwell) | `_vendor/flash_attn/` (see below) |

The upstream package vendored a copy of FlashAttention's CuTe DSL helpers
under `sol_attn/_vendor/flash_attn/cute/`. That copy is **not** carried here:
TensorRT-LLM already depends on
[`flash-attn-4`](https://github.com/Dao-AILab/flash-attention) (pinned in
`requirements.txt`), which provides the same `flash_attn.cute` modules, and
the SM100/SM120 kernels import them from that dependency directly. This was
verified on B200 to produce bit-identical output to the vendored copy across a
shape/tau sweep. FlashAttention's BSD-3-Clause license is retained at
`sol_attn/sm100/LICENSE.flash-attention` because portions of the SM100 design
scaffold still derive from that project.

`preprocess.py` implements the routing/threshold stage in Triton, so Triton is
a required runtime dependency on every Sol-Attn path, not only a fallback.

The runtime also depends on NVIDIA CUTLASS / CuTe DSL, cuda-python, and
PyTorch. Those dependencies are not redistributed by this repository and
remain subject to their respective licenses.

The SM120 warp-MMA/TMA execution skeleton and online-softmax helpers are
adapted from
[NVIDIA cuDNN Frontend's block-sparse-attention reference](https://github.com/NVIDIA/cudnn-frontend/tree/74785165de2da954a2c879a5e3e6f95411c2292d)
at commit `74785165de2da954a2c879a5e3e6f95411c2292d`. That source is
licensed under the Apache License 2.0; adapted files retain the
corresponding SPDX header.
