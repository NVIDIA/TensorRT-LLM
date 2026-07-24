# SHA-256 (vendored from Bitcoin Core, modified)

This directory contains the SHA-256 implementation from the
[Bitcoin Core](https://github.com/bitcoin/bitcoin) project, **reduced and
adapted by NVIDIA** to the minimal single-block hasher needed by TensorRT-LLM.
It provides a portable scalar SHA-256 with runtime dispatch to a
hardware-accelerated transform (x86 SHA-NI or ARMv8 crypto extensions), exposed
through the `CSHA256` class.

TensorRT-LLM uses it in the C++ KVCacheManagerV2 (`blockRadixTree`) to hash
token sequences into block keys, byte-identically to the Python backend's
`hashlib.sha256` block keys.

## Provenance

- **Upstream project:** Bitcoin Core — https://github.com/bitcoin/bitcoin
- **Source path upstream:** `src/crypto/`
- **Upstream commit:** `70d9ec7f3d452789d04dce81dc02db0b3b778bb5` (branch `master`)
- **Retrieved:** 2026-07-17
- **License:** MIT (see `LICENSE`, copied from the upstream `COPYING`). Each
  source file retains its original MIT header.

## Contents

| File | Origin | Notes |
|------|--------|-------|
| `sha256.h` | Bitcoin, modified | `CSHA256` API; `SHA256D64` declaration removed |
| `sha256.cpp` | Bitcoin, modified | Scalar core + `CSHA256` + slim runtime dispatch |
| `sha256_x86_shani.cpp` | Bitcoin, modified | x86 SHA-NI transform (guard changed) |
| `sha256_arm_shani.cpp` | Bitcoin, modified | ARMv8 crypto transform (guard changed) |
| `attributes.h` | Bitcoin, unmodified | `ALWAYS_INLINE` macro used by the x86 transform |
| `sha256_endian.h` | **NVIDIA-authored** | Big-endian helpers replacing upstream `common.h` |
| `LICENSE` | Bitcoin | Upstream MIT `COPYING` |

## NVIDIA modifications

The files are **not** as-received; they were reduced to the single-block path
TensorRT-LLM needs. Per NVIDIA's open-source guidance for permissively licensed
software used with modifications, each edited Bitcoin file keeps its original
MIT header and carries an added NVIDIA copyright/modification notice; the whole
directory remains under the MIT license.

Changes vs. upstream:

- **Removed** the public `SHA256D64` double-hash API (and its scalar
  `TransformD64`) and the standalone SSE4 / SSE4.1 / AVX2 multi-block transform
  files (unused — TensorRT-LLM only calls single-block `CSHA256`). The two
  SHA-NI files retain their upstream 2-way `Transform_2way` helpers; these are
  now unreferenced but were left in place to keep the transforms byte-close to
  upstream.
- **Removed** the upstream support-header chain (`crypto/common.h`,
  `compat/endian.h`, `compat/byteswap.h`, `compat/cpuid.h`), which pulled in
  C++20 (`<bit>`, `<concepts>`). Endian helpers are now the small,
  NVIDIA-authored `sha256_endian.h`; CPU detection uses a `CPUID` leaf-7 check
  via `__get_cpuid_count` (x86) / `getauxval(AT_HWCAP)` (aarch64). The vendored
  sources now build as plain **C++17**.
- **Flattened** the directory (no `crypto/` / `compat/` subtree) and switched
  the HW-transform build guards from the upstream `ENABLE_*` macros to target
  architecture macros. Build wiring (per-file `-msha` / `-march=armv8-a+crypto`)
  lives in `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/CMakeLists.txt`.

## Updating

Re-fetch `src/crypto/sha256*.{h,cpp}` from the desired upstream tag and re-apply
the modifications above (they are localized to file headers, includes, build
guards, and the removed `SHA256D64`/SIMD sections). Update the commit hash and
date. Verify with the standard `sha256("abc")` vector and against the Python
`hashlib.sha256` block-key chain.
