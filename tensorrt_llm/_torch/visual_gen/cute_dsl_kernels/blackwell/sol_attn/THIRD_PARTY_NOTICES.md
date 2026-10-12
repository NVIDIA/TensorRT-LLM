# Third-party notices

This package is vendored from
[`github.com/NVlabs/Sana`](https://github.com/NVlabs/Sana), branch
[`sol-engine`](https://github.com/NVlabs/Sana/tree/sol-engine), at commit
[`5fe5feb`](https://github.com/NVlabs/Sana/commit/5fe5feb) (2026-08-17).
Checked against the branch tip
([`83e54df`](https://github.com/NVlabs/Sana/commit/83e54df), 2026-08-20) on
2026-08-27.

## Scope of the vendored subset

Only the pieces needed for the architectures TensorRT-LLM ships are carried:

| Carried | Not carried |
|---|---|
| `interface.py`, `preprocess.py`, `common/` | `sm89/`, `sm90/` (incl. `sm90/_compat/`) |
| `sm100/` — serves SM100 (B200/GB200) and SM103 (B300/GB300) | `triton_ref/` Triton reference attention |
| | `sm120/` (RTX Blackwell) |
| | `_vendor/flash_attn/` (see below) |

`../sol_attn_backend.py` is not part of the vendored package but is a
derivative work of upstream's
`techniques/sparse_backends/sol_attn_backend.py` (same branch and commit).
Only the kernel-wrapper subset is carried; upstream's model-integration half
(diffusers dispatch hook, HunyuanVideo MMDiT padding, model-level Morton
ordering) is not.

Implementation divergences from upstream are documented in the source itself,
in the module docstrings of `../sol_attn_backend.py` and
`attention_backend/cute_dsl/sol_attn.py`.

## Licensing

The upstream package vendored a copy of FlashAttention's CuTe DSL helpers
under `sol_attn/_vendor/flash_attn/cute/`. That copy is **not** carried here:
TensorRT-LLM already depends on
[`flash-attn-4`](https://github.com/Dao-AILab/flash-attention) (pinned in
`requirements.txt`), which provides the same `flash_attn.cute` modules, and
the SM100 kernels import them from that dependency directly. FlashAttention's
BSD-3-Clause license is retained at `sol_attn/sm100/LICENSE.flash-attention`
because portions of the SM100 design scaffold still derive from that project.

`preprocess.py` implements the routing/threshold stage in Triton, so Triton is
a required runtime dependency on every Sol-Attn path, not only a fallback.

The runtime also depends on NVIDIA CUTLASS / CuTe DSL, cuda-python, and
PyTorch. Those dependencies are not redistributed by this repository and
remain subject to their respective licenses.
