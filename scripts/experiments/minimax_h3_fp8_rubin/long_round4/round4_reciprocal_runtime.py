# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Postload exact derived constants for one immutable H3 calibration/model.

Only the norm producer receives the reciprocal pointer. GEMMs and all model
scale tensors retain their original values. Unsupported scales use the original
kernel. No activation or output is cached. Captures use stable device constants.
"""

import sys

import fp8_producers as producer
import round4_norm_kernel as candidate
import torch
import triton
import triton.language as tl


@triton.jit
def exact_reciprocal_kernel(scale, inverse):
    value = tl.div_rn(1.0, tl.load(scale).to(tl.float32))
    tl.store(inverse, value)


class Runtime:
    def __init__(self, blocks):
        self.original = producer._adaln_fp8_kernel
        self.mode = "off"
        self.entries = {}
        self.inventory = []
        self.calls = 0
        self.fallbacks = 0
        mixed = sys.modules.get("h3_mixed_layout")
        if mixed is not None and mixed.active():
            mixed.assert_fp8_subset(blocks)
        else:
            assert len(blocks) == 50
        for layer, block in enumerate(blocks):
            assert not block.training
            for role, linear in [("qkv", block.attn.qkv_proj), ("gate_up", block.ff.gate_up_proj)]:
                scale = linear.input_scale
                assert (
                    scale.is_cuda
                    and scale.dtype == torch.float32
                    and scale.numel() == 1
                    and scale.is_contiguous()
                )
                assert bool(torch.isfinite(scale).all()) and bool((scale > 0).all())
                key = (scale.device.index, scale.data_ptr())
                if key not in self.entries:
                    inverse = torch.empty_like(scale)
                    exact_reciprocal_kernel[(1,)](scale, inverse, num_warps=4)
                    self.entries[key] = (scale, scale._version, inverse)
                self.inventory.append(
                    dict(
                        layer=layer,
                        role=role,
                        scale_value=float(scale),
                        pointer=scale.data_ptr(),
                        version=scale._version,
                    )
                )
        torch.cuda.synchronize()

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            # This adapter executes inside the existing opaque norm custom op.
            scale = args[4]
            key = (scale.device.index, scale.data_ptr())
            entry = self.entries.get(key)
            selected = self.mode == "all" or (self.mode == "norm_only" and not kwargs["HAS_RES"])
            if selected and entry is not None:
                self.calls += 1
                arguments = list(args)
                arguments[4] = entry[2]
                return candidate._adaln_fp8_kernel[grid](*arguments, **kwargs)
            self.fallbacks += 1
            return self.original[grid](*args, **kwargs)

        return run

    def configure(self, mode):
        assert mode in ("off", "norm_only", "all")
        if mode != "off":
            for scale, version, inverse in self.entries.values():
                assert scale._version == version
                assert scale.device == inverse.device
        self.mode = mode
        producer._adaln_fp8_kernel = self.original if mode == "off" else self
