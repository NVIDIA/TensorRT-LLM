# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The glm_kpool JIT-prefetch planner covers the indexer kernels' variants.

One real call per phase (as the warmup forwards make them) fills the launch
template; the planner then enumerates the batch-dependent variants, and the
real launches of other batches -- other row counts, generation rows behind
any number of context rows, single-request verify steps -- must each hit a
planned Triton cache key.
"""

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool import kernels
from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.jit_prefetch import GlmKpoolProvider

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

HD, KPOOL, TPB, H, SELECT_K, WIDTH = 128, 16, 64, 32, 128, 2112
SLOTS, PAGES, MAX_BATCH = 64, 8, 32


def _keys(calls):
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key

    keys = set()
    for call in calls:
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        for kw in variants:
            kw = dict(kw)
            kw["debug"] = kw.get("debug", jit_fn.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            _, kkc, _, _, binder = jit_fn.device_caches[driver.active.get_current_device()]
            _, spec, opts = binder(*call.args, **kw)
            keys.add((jit_fn.fn.__name__, compute_cache_key(kkc, spec, opts)))
    return keys


class _Engine:
    """Engine-sized buffers laid out as the glm_kpool backend sees them."""

    def __init__(self):
        dev = torch.device("cuda")
        storage = torch.zeros(SLOTS, TPB, 3 * HD + 64, dtype=torch.bfloat16, device=dev)
        self.index_pool = storage[..., : 3 * HD]
        self.block_tables = torch.zeros(MAX_BATCH, PAGES, dtype=torch.int64, device=dev)
        self.kv_lens = torch.full((MAX_BATCH,), 100, dtype=torch.int32, device=dev)
        self.ape = torch.zeros(KPOOL, HD, dtype=torch.bfloat16, device=dev)
        self.dev = dev

    def _qw(self, rows):
        q = torch.zeros(rows, H, HD, dtype=torch.bfloat16, device=self.dev)
        fused = torch.zeros(rows, 2 * HD + H, dtype=torch.bfloat16, device=self.dev)
        return q, fused[:, 2 * HD :]  # head weights: a column slice of the fused GEMM

    def prefill(self, tokens, nreq):
        """Packed context rows of ``nreq`` requests."""
        q, w = self._qw(tokens)
        req = torch.arange(tokens, device=self.dev, dtype=torch.int32) % nreq
        visible = torch.arange(1, tokens + 1, device=self.dev, dtype=torch.int64)
        tables = self.block_tables[:nreq]
        pos = torch.arange(tokens, device=self.dev, dtype=torch.int64)
        kernels.kpool_update(self.index_pool, tables, pos[::KPOOL].contiguous(), self.ape, TPB,
                             head_dim=HD, kpool=KPOOL, request_ids=req[::KPOOL].contiguous())  # fmt: skip
        kernels.kpool_score(q, w, self.index_pool, tables, visible, TPB, num_pools_max=PAGES * TPB // KPOOL,
                            head_dim=HD, kpool=KPOOL, q_scale=1.0, w_scale=1.0, precision="tf32",
                            rows_per_program=16, request_ids=req)  # fmt: skip
        sel = torch.zeros(tokens, SELECT_K, dtype=torch.int32, device=self.dev)
        kernels.kpool_expand(sel, visible, tables, TPB, base_row=0, rows_per_slot=TPB, kpool=KPOOL,
                             out_width=WIDTH, request_ids=req)  # fmt: skip

    def decode(self, nctx, ngen):
        """``ngen`` one-token generation rows behind ``nctx`` context rows."""
        batch = nctx + ngen
        q, w = self._qw(ngen)
        tables = self.block_tables[:batch][nctx:]
        lens = self.kv_lens[:batch][nctx:]
        kernels.kpool_update(self.index_pool, tables, lens - 1, self.ape, TPB, head_dim=HD,
                             kpool=KPOOL)  # fmt: skip
        kernels.kpool_score(q, w, self.index_pool, tables, lens, TPB, num_pools_max=PAGES * TPB // KPOOL,
                            head_dim=HD, kpool=KPOOL, q_scale=1.0, w_scale=1.0, precision="tf32")  # fmt: skip
        sel = torch.zeros(ngen, SELECT_K, dtype=torch.int32, device=self.dev)
        kernels.kpool_expand(sel, lens, tables, TPB, base_row=0, rows_per_slot=TPB, kpool=KPOOL,
                             out_width=WIDTH)  # fmt: skip

    def verify(self, nctx, ngen, steps):
        """MTP verification: ``steps`` tokens per generation request."""
        batch = nctx + ngen
        tables = self.block_tables[:batch][nctx:]
        lens = self.kv_lens[:batch][nctx:]
        positions = (lens.long() - steps).unsqueeze(1) + torch.arange(steps, device=self.dev)
        for step in range(steps):
            kernels.kpool_update(self.index_pool, tables, positions[:, step].contiguous(), self.ape,
                                 TPB, head_dim=HD, kpool=KPOOL)  # fmt: skip
        rows = ngen * steps
        q, w = self._qw(rows)
        rep = tables.repeat_interleave(steps, dim=0)
        visible = (positions + 1).reshape(-1)
        kernels.kpool_score(q, w, self.index_pool, rep, visible, TPB, num_pools_max=PAGES * TPB // KPOOL,
                            head_dim=HD, kpool=KPOOL, q_scale=1.0, w_scale=1.0, precision="tf32")  # fmt: skip
        sel = torch.zeros(rows, SELECT_K, dtype=torch.int32, device=self.dev)
        kernels.kpool_expand(sel, visible, rep, TPB, base_row=0, rows_per_slot=TPB, kpool=KPOOL,
                             out_width=WIDTH)  # fmt: skip


def _recorded(fn, *args):
    with jp.shadow_launches() as rec:
        fn(*args)
    return _keys(rec)


def test_planned_keys_cover_every_phase_and_batch():
    eng = _Engine()
    kernels.KPOOL_LAUNCH_TEMPLATE.clear()
    # Warmup-like calls fill the template: one prefill, a pure decode, a verify.
    eng.prefill(128, 2)
    eng.decode(0, 4)
    eng.verify(0, 4, 4)
    torch.cuda.synchronize()
    provider = GlmKpoolProvider(max_num_tokens=4096)
    assert provider
    planned = _keys(provider.plan_all())
    cases = [
        ("prefill", eng.prefill, (1, 1)),
        ("prefill", eng.prefill, (17, 3)),
        ("prefill", eng.prefill, (512, 5)),
        ("decode", eng.decode, (0, 1)),
        ("decode", eng.decode, (1, 3)),  # kv_lens slice 4 bytes in
        ("decode", eng.decode, (2, 16)),
        ("decode", eng.decode, (3, 17)),
        ("verify", eng.verify, (0, 1, 4)),  # one request: strided step positions
        ("verify", eng.verify, (5, 3, 4)),
    ]
    for phase, fn, args in cases:
        real = _recorded(fn, *args)
        assert real, f"{phase}{args}: recorded no launches"
        missing = real - planned
        assert not missing, f"{phase}{args}: unplanned {sorted(name for name, _ in missing)}"
