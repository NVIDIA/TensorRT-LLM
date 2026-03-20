# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression test: partial block must not rebase onto full tree block."""

import gc
import itertools
import os
import unittest
from importlib.util import find_spec
from typing import cast

TYPE_CHECKING = False
if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import (
        DEFAULT_BEAM_INDEX,
        CudaStream,
        DataRole,
        KVCacheManager,
        LayerId,
        TokenId,
        TokenIdExt,
    )
    from kv_cache_manager_v2._common import MemAddress
    from kv_cache_manager_v2._utils import (
        TemporaryCudaStream,
        exact_div,
        init_cuda_once,
        round_up,
        temporary_sys_path,
    )
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        DEFAULT_BEAM_INDEX,
        CudaStream,
        DataRole,
        KVCacheManager,
        LayerId,
        TokenId,
        TokenIdExt,
    )
    from tensorrt_llm.runtime.kv_cache_manager_v2._common import MemAddress
    from tensorrt_llm.runtime.kv_cache_manager_v2._utils import (
        TemporaryCudaStream,
        exact_div,
        init_cuda_once,
        round_up,
        temporary_sys_path,
    )

with temporary_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from kernels import check_values, fill_values
    from test_kv_cache_manager_v2 import create_config

KEY = DataRole("key")
LID = LayerId(0)


class TestPartialBlockRebase(unittest.TestCase):
    def setUp(self) -> None:
        init_cuda_once()
        self._tok = itertools.count(1)
        gc.collect()
        gc.disable()
        self.tpb = 8
        self.kv_buf_size = 8192
        self.cfg = create_config(self.tpb, 16 << 20, 0, 0, 2, None, 0, self.kv_buf_size)
        self.manager = KVCacheManager(self.cfg)

    def tearDown(self) -> None:
        gc.enable()
        self.manager.shutdown()

    def tok(self) -> TokenIdExt:
        return TokenId(next(self._tok))

    def _page_addr(self, kv, block_ordinal):
        lc = self.manager._storage._layer_to_life_cycle_ids[LID]
        base_idx = list(kv.get_base_page_indices(lc, DEFAULT_BEAM_INDEX))[block_ordinal]
        pool = self.manager.get_mem_pool_base_address(LID, KEY)
        stride = self.manager.get_page_stride(LID, KEY)
        scale = self.manager.get_page_index_scale(LID, KEY)
        return MemAddress(pool + stride * base_idx * scale)

    def _tok_bytes(self):
        return exact_div(self.kv_buf_size, self.tpb)

    def _write(self, kv, ordinal, offset, tokens, stream):
        addr = self._page_addr(kv, ordinal)
        tb = self._tok_bytes()
        fill_values(
            MemAddress(addr + tb * offset),
            tb,
            1,
            self.tpb,
            LID,
            0,
            DEFAULT_BEAM_INDEX,
            tokens,
            stream,
        )

    def _check(self, kv, ordinal, offset, tokens, stream):
        addr = self._page_addr(kv, ordinal)
        tb = self._tok_bytes()
        check_values(
            MemAddress(addr + tb * offset),
            tb,
            1,
            self.tpb,
            LID,
            0,
            DEFAULT_BEAM_INDEX,
            tokens,
            stream,
        )

    def _prefill(self, prompt, reuse, stream):
        kv = self.manager.create_kv_cache(None, reuse)
        kv.resume(stream)
        kv.resize(round_up(len(prompt), 1))
        n = kv.num_committed_tokens
        for i in range(n, len(prompt)):
            self._write(kv, i // self.tpb, i % self.tpb, [prompt[i]], stream)
        if n < len(prompt):
            kv.commit(prompt[n:])
        kv.stop_committing()
        return kv

    def test_partial_block_not_rebased(self) -> None:
        """Partial block rebase must not let C write onto the shared tree page.

        A0: 13 tokens → block 1 partial (5/8), commit to tree.
        A:  16 tokens (same prefix + 3) → extends block 1 to full (8/8).
        B:  reuse 12, process token 12, commit + stop_committing.
            Then B writes tokens at positions 13, 14 (block 1 offsets 5, 6).

        Without fix: rebase makes B's block 1 point to the tree page (shared
        with A).  B's writes at offsets 5, 6 overwrite A's data.
        With fix: B keeps its own copy page.  A's data is untouched.
        """
        prefix = [self.tok() for _ in range(13)]  # 1 full + partial
        extra = [self.tok() for _ in range(3)]  # extends to full
        gen_tokens = [self.tok(), self.tok()]  # B generates 2 tokens

        with TemporaryCudaStream([]) as s:
            stream = cast(CudaStream, s.handle)

            kv_a0 = self._prefill(prefix, prefix, stream)
            kv_a = self._prefill(prefix + extra, prefix + extra, stream)

            # Verify A's block 1 data
            self._check(kv_a, 1, 5, [extra[0]], stream)  # offset 5
            self._check(kv_a, 1, 6, [extra[1]], stream)  # offset 6

            # B: reuse 12, process token 12, commit
            kv_b = self.manager.create_kv_cache(None, prefix[:12])
            assert kv_b.num_committed_tokens == 12
            kv_b.resume(stream)
            kv_b.resize(round_up(len(prefix) + len(extra) + len(gen_tokens), 1))
            self._write(kv_b, 1, 4, [prefix[12]], stream)

            kv_b.commit(prefix[12:])
            kv_b.stop_committing()

            # B "generates" 2 tokens at positions 13, 14 (block 1 offsets 5, 6)
            self._write(kv_b, 1, 5, [gen_tokens[0]], stream)
            self._write(kv_b, 1, 6, [gen_tokens[1]], stream)

            # A's data must NOT be corrupted by B's writes
            self._check(kv_a, 1, 5, [extra[0]], stream)
            self._check(kv_a, 1, 6, [extra[1]], stream)

            kv_a0.close()
            kv_a.close()
            kv_b.close()

        s.take_finish_event().synchronize()
        self.manager.clear_reusable_blocks()


if __name__ == "__main__":
    unittest.main()
