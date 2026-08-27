# Third-party notices

This package is vendored wholesale from `github.com/NVlabs/Sana`, branch
`sol-engine`, at commit `5fe5feb` (2026-08-17; best-effort reconstruction from
vendoring-date file timestamps and upstream commit history, not an exact
recorded pin from the original port -- see PR discussion for how this was
verified). Checked against the current branch tip (`83e54df`, 2026-08-20) on
2026-08-27: the only relevant upstream change since is a merge whose SM89
touch is described as "resolve SM89 documentation conflicts, format the
backend" -- `sm89/kernel.py` diffs byte-identical and `sm89/mainloop.py` /
`sm89/__init__.py` are structurally identical (same constants, functions,
control flow) against the current upstream tip, so this vendored copy is
functionally current as of that check. The rest of the Aug 20 merge (MPS/
Metal Apple Silicon backend, RTX 4090/5090 configs) is out of scope for this
CUDA/Blackwell-only vendored subset.

The files under `sol_attn/_vendor/flash_attn/cute/` and portions of the SM89,
SM90, and SM100 design scaffold derive from the FlashAttention project. Its
BSD-3-Clause license is included at
`sol_attn/sm100/LICENSE.flash-attention`.

The runtime also depends on NVIDIA CUTLASS / CuTe DSL, cuda-python, PyTorch,
and Triton. Those dependencies are not redistributed by this repository and
remain subject to their respective licenses.

The SM120 warp-MMA/TMA execution skeleton and online-softmax helpers are
adapted from NVIDIA cuDNN Frontend's block-sparse-attention reference at commit
`74785165de2da954a2c879a5e3e6f95411c2292d`. That source is licensed under the
Apache License 2.0; adapted files retain the corresponding SPDX header.
