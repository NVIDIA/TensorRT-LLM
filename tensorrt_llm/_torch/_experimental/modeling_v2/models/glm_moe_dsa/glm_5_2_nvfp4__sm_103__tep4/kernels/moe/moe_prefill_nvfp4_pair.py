# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compilation boundary for the experimental GLM-5.2 prefill NVFP4 GG pair.

This is an experimental example, not a serving default or a TRT-LLM adapter.
Fixed model dimensions: H6144, intermediate2048, 64 local experts, top-K8.
Both kernels use compact 128-row expert tiles and K256; N is selected from T
before graph capture.

Example (SM103 device; ``use_pdl`` must equal TRT-LLM's ``TRTLLM_ENABLE_PDL``)::

    width = selected_width(num_tokens)
    gg1 = compile_gg1(tile_n=width, num_sms=148, use_pdl=True)
    gg2 = compile_gg2(tile_n=width, num_sms=148, use_pdl=True)

The handles accept runtime tensor/scalar arguments in the order of the
respective kernel module's moe entry, excluding constexpr parameters/stream.
FP4 storage uses Float4E2M1FNx2 views, scales Float8E4M3FN; all pointers must
be 16-byte aligned. Layout/stride orders are defined by the fake tensors below.

GG1 takes original A[1,T,3072], linear SFA[T,384,1], W1/SFW1, per-expert alpha,
norm_const and current routing metadata. Its outputs are C[1,rows,1024] and
SFC[1,tiles,32,512]. GG2 consumes those same outputs and routing epoch, W2/SFW2,
token weights[T,8] and per-expert alpha, and atomically accumulates BF16[T,6144].
The bias argument remains in the ABI but is unused by the selected specialization.

Caller contracts (not enforced by these compile-only entry points):
* rows = 128 * tile capacity; SF atom capacity = tile capacity; metadata arrays
  cover that capacity. num_tiles_m is the device-resident actual tile count.
* Expert IDs are local [0,64). For each valid permuted row, token_mapping stores
  expanded_idx = token_id * 8 + topk_slot in [0,T*8), NOT a token ID in [0,T).
  GG1 gathers token expanded_idx // 8; GG2 uses the same index to select its
  routing weight and output token. mn_limit is an absolute exclusive row bound:
  tile i's valid prefix is [128*i, mn_limit[i]); padding is not a token.
* Retain nonalias buffers/views and current metadata through every graph replay.
* Routing/quantization must complete before GG1 consumes their data; GG2 must
  wait for GG1 and for output zero-initialization, including cross-stream events.
  PDL is not a replacement for the auxiliary-stream zero-ready dependency.
* Neither shared-expert merge nor a subsequent collective is included here.

Metadata producer compatibility:
``mtp_moe_routing.routing_host_with_workspace`` with LOCAL_EXPERTS=64, TILE=128,
num_expanded=T*8, the rank's local_expert_offset, and its documented workspace
and routing_schedule_bucket() satisfies this layout contract. Bind its outputs:
* tile_idx_to_expert_idx -> both kernels' tile_idx_to_expert_idx;
* tile_idx_to_mn_limit -> GG1 mn_limit / GG2 tile_idx_to_mn_limit;
* permuted_idx_to_expanded_idx -> GG1 token_mapping / GG2's same-named map;
* num_non_exiting_tiles -> both kernels' device num_tiles_m.
total_num_padded_tokens describes the active padded extent, not capacity. The
inverse expanded_idx_to_permuted_idx is not the GG pair's row map. Empty experts
contribute no tiles; busy experts may contribute several adjacent M128 tiles.

Layout compatibility does not establish execution readiness. The general
routing single-CTA/cached-cluster paths launch with PDL and do not issue an
input dependency wait. Callers must establish selected-ID visibility before
those kernels read it; GG1's later wait cannot repair stale routing metadata.
This example does not handle that producer synchronization.

Compile/cache keys must include tile_n, num_sms, PDL and source/compiler identity.
Do not mutate module globals or the PDL environment after import. Fixed weight
policies are GG1 evict_last and GG2 evict_first, independent of global L2 overrides.
"""

import importlib


def selected_width(num_tokens: int) -> int:
    """Select the measured width family; unsupported T is an error, not fallback.

    The checkpoint uses N128 through T1536, then N256. The supported prefill
    domain is T64..8192; decode below 64 belongs to the existing decode kernels.
    """
    if type(num_tokens) is not int or not 64 <= num_tokens <= 8192:
        raise ValueError("prefill pair requires integer T in [64,8192]")
    return 128 if num_tokens <= 1536 else 256


def _require_contract(tile_n, num_sms, use_pdl):
    """Reject unsupported architecture/configuration before loading CUDA modules."""
    if type(tile_n) is not int or tile_n not in (128, 256):
        raise ValueError("tile_n must be integer 128 or 256")
    if type(num_sms) is not int or num_sms <= 0:
        raise ValueError("num_sms must be a positive integer")
    if type(use_pdl) is not bool:
        raise ValueError("use_pdl must be bool")
    import torch

    from tensorrt_llm._torch.kernels.blackwell.utils import TRTLLM_ENABLE_PDL

    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("this mixed-K prefill kernel pair requires an SM103 device")
    if use_pdl != bool(TRTLLM_ENABLE_PDL):
        raise ValueError("use_pdl must match TRT-LLM's TRTLLM_ENABLE_PDL")


def _load_kernel(name, policy, use_pdl):
    """Reject a module cached under incompatible PDL or weight-cache settings."""
    module = importlib.import_module(f".{name}", __package__)
    if module.USE_PDL != use_pdl or module.L2_POLICY_B != policy:
        raise ValueError("cached kernel policy differs; use a fresh process")
    return module


def compile_gg1(*, tile_n: int, num_sms: int, use_pdl: bool):
    """Compile gather/GG1/SwiGLU/NVFP4 using the actual unpadded input-scale ABI."""
    _require_contract(tile_n, num_sms, use_pdl)
    module = _load_kernel("moe_prefill_nvfp4_gg1", "evict_last", use_pdl)
    return _compile_gg1(module, tile_n, num_sms)


def compile_gg2(*, tile_n: int, num_sms: int, use_pdl: bool):
    """Compile compact-M128 GG2/weighted BF16 atomic finalize (no bias)."""
    _require_contract(tile_n, num_sms, use_pdl)
    module = _load_kernel("moe_prefill_nvfp4_gg2", "evict_first", use_pdl)
    return _compile_gg2(module, tile_n, num_sms)


def _compile_gg1(module, tile_n, num_sms, bindings=None):
    """Build the packed GG1 ABI; inputs and linear scales share unpadded T."""
    if bindings is None:
        cutlass = importlib.import_module("cutlass")
        cute = importlib.import_module("cutlass.cute")
        runtime = importlib.import_module("cutlass.cute.runtime")
    else:
        cutlass, cute, runtime = bindings
    tokens = cute.sym_int64()
    rows = cute.sym_int64(divisibility=128)
    tiles = cute.sym_int64()

    def fake(dtype, shape, order):
        """Create an aligned symbolic tensor without allocating device storage."""
        return runtime.make_fake_compact_tensor(dtype, shape, stride_order=order, assumed_align=16)

    return cute.compile(
        module.moe,
        fake(cutlass.Float4E2M1FNx2, (64, 3072, 4096), (2, 0, 1)),
        fake(cutlass.Float8E4M3FN, (64, 32, 96, 512), (3, 2, 1, 0)),
        fake(cutlass.Float4E2M1FNx2, (1, rows, 1024), (2, 1, 0)),
        fake(cutlass.Int32, (tiles,), (0,)),
        fake(cutlass.Float4E2M1FNx2, (1, tokens, 3072), (2, 1, 0)),
        fake(cutlass.Float8E4M3FN, (tokens, 384, 1), (2, 1, 0)),
        fake(cutlass.Float8E4M3FN, (1, tiles, 32, 512), (3, 2, 1, 0)),
        fake(cutlass.Float32, (1,), (0,)),
        fake(cutlass.Float32, (64,), (0,)),
        fake(cutlass.Int32, (rows,), (0,)),
        fake(cutlass.Int32, (tiles,), (0,)),
        cutlass.Int32(6144),
        8,
        fake(cutlass.Int32, (1,), (0,)),
        cutlass.Int32(128),
        cutlass.Int32(32),
        tile_n,
        256,
        False,
        num_sms,
        runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        False,
        False,
        False,
        False,
        options="--enable-tvm-ffi --generate-line-info",
    )


def _compile_gg2(module, tile_n, num_sms, bindings=None):
    """Build compact-M128 GG2 ABI with distinct dynamic row/tile/SF extents."""
    if bindings is None:
        cutlass = importlib.import_module("cutlass")
        cute = importlib.import_module("cutlass.cute")
        runtime = importlib.import_module("cutlass.cute.runtime")
    else:
        cutlass, cute, runtime = bindings
    tokens = cute.sym_int64()
    rows = cute.sym_int64(divisibility=128)
    tiles = cute.sym_int64()
    sf_atoms = cute.sym_int64()

    def fake(dtype, shape, order):
        """Create an aligned symbolic tensor without allocating device storage."""
        return runtime.make_fake_compact_tensor(dtype, shape, stride_order=order, assumed_align=16)

    return cute.compile(
        module.moe,
        fake(cutlass.Float4E2M1FNx2, (1, rows, 1024), (2, 1, 0)),
        fake(cutlass.Float4E2M1FNx2, (64, 1024, 6144), (2, 0, 1)),
        fake(cutlass.Float8E4M3FN, (1, sf_atoms, 32, 512), (3, 2, 1, 0)),
        fake(cutlass.Float8E4M3FN, (64, 48, 32, 512), (3, 2, 1, 0)),
        fake(cutlass.BFloat16, (tokens, 6144), (1, 0)),
        fake(cutlass.Int32, (tiles,), (0,)),
        fake(cutlass.Float32, (64, 6144), (1, 0)),
        fake(cutlass.Int32, (rows,), (0,)),
        fake(cutlass.Float32, (tokens, 8), (1, 0)),
        fake(cutlass.Float32, (64,), (0,)),
        fake(cutlass.Int32, (tiles,), (0,)),
        cutlass.Int32(2048),
        cutlass.Int32(6144),
        cutlass.Int32(8),
        fake(cutlass.Int32, (1,), (0,)),
        cutlass.Int32(128),
        tile_n,
        256,
        True,
        1,
        False,
        False,
        0,
        False,
        num_sms,
        runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        0,
        0,
        False,
        False,
        True,
        True,
        128,
        options="--enable-tvm-ffi --generate-line-info",
    )
