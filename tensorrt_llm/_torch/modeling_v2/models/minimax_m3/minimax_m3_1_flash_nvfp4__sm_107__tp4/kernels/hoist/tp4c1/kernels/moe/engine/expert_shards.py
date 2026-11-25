# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""uGPU split-N expert shards: each die's half of the routed weights, held in that die's memory pool."""

import torch

_st = {}


def _shards(mw):
    """The two die shards of mw's routed weights (row slices of the prepared buffers), read in place by the engines.
    The vLLM chain registers production's shards in _st[id(mw)]; otherwise they are cut here into each die's pool."""
    key = id(mw)
    if key not in _st:
        import flashinfer_ugpu as fu  # the reference runtime's die pools (the target registers its shards first)

        rt = fu.get_runtime(torch.device("cuda", torch.cuda.current_device()))
        pools = [rt.mem_pool(0), rt.mem_pool(1)] if rt is not None else [None, None]
        w13, w13s, w2, w2s = mw.experts
        i, h = mw.inter, 6144

        def cut(t, p, n):
            src = t[:, p * n : (p + 1) * n]
            if pools[p] is not None:
                with torch.cuda.use_mem_pool(pools[p]):
                    d = torch.empty(src.shape, dtype=src.dtype, device=src.device)
            else:
                d = torch.empty(src.shape, dtype=src.dtype, device=src.device)
            d.copy_(src)
            return d

        _st[key] = [
            (cut(w13, p, i), cut(w13s, p, i), cut(w2, p, h // 2), cut(w2s, p, h // 2))
            for p in range(2)
        ]
    return _st[key]
