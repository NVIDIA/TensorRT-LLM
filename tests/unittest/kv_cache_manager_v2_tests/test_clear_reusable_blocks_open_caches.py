# Unit tests for clear_reusable_blocks() with open KV caches (T1-refit).
import unittest
from typing import cast

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    GpuCacheTierConfig, KVCacheManager, KVCacheManagerConfig, ReuseScope)
from tensorrt_llm.runtime.kv_cache_manager_v2._core._kv_cache import _KVCache
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import TemporaryCudaStream
from tensorrt_llm.runtime.kv_cache_manager_v2._common import CudaStream, TokenId
from kv_cache_manager_v2_tests.test_kv_cache_manager_v2 import (
    TestKVCacheManagerV2)


class TestClearReusableBlocksWithOpenCaches(TestKVCacheManagerV2):
    """clear_reusable_blocks() must behave like V1: drop the reuse tree, keep living caches."""

    def _prompt(self, n: int) -> list[TokenId]:
        return [TokenId(i) for i in range(n)]

    def test_clear_with_open_cache_keeps_pages_and_stops_commits(self) -> None:
        tokens_per_block = 32
        self.prepare(32 << 20, 0, 0, 4, 128, 1, kv_buf_size=32768)
        prompt = self._prompt(tokens_per_block * 4)
        req = self.new_request(0, None, len(prompt), tokens_per_block)
        with TemporaryCudaStream([]) as s:
            assert req.kv_cache.resume(cast(CudaStream, s.handle))
            self.run_request(req, tokens_per_block, True)  # commits 4 full blocks
        s.take_finish_event().synchronize()
        self.assertEqual(self.manager.probe_reuse(None, prompt), len(prompt))
        # 1. No LogicError with the cache still open (the assertion this replaces).
        self.manager.clear_reusable_blocks()
        # 2. Tree is empty; living cache still open, pages still addressable.
        self.assertEqual(self.manager.probe_reuse(None, prompt), 0)
        self.assertNotEqual(req.kv_cache.status, _KVCache.Status.CLOSED)
        for ord_ in range(4):
            self.assertIsNotNone(req.kv_cache.get_base_page_indices(0)[ord_])
        # 3. Further commits are no-ops (no stale block re-registered, no dangling prev).
        req.kv_cache.commit(self._prompt(tokens_per_block), is_end=False)
        self.assertEqual(self.manager.probe_reuse(None, prompt + self._prompt(tokens_per_block)), 0)
        # 4. Close releases the orphan blocks' pages back to the pool (no leak / no assert).
        req.kv_cache.close()
        self.manager.clear_reusable_blocks()
        self.manager.shutdown()

    def test_clear_with_never_committing_dummy_cache(self) -> None:
        # Mirrors the CUDA-graph padding dummy: stop_committing() right after resume.
        self.prepare(32 << 20, 0, 0, 4, 128, 1, kv_buf_size=32768)
        dummy = self.manager.create_kv_cache(ReuseScope(lora_id=None), self._prompt(1))
        with TemporaryCudaStream([]) as s:
            assert dummy.resume(cast(CudaStream, s.handle))
            dummy.stop_committing()
            assert dummy.resize(1)
        self.manager.clear_reusable_blocks()  # must not raise
        self.manager.clear_reusable_blocks()  # idempotent
        dummy.close()
        self.manager.shutdown()

    def test_clear_then_new_request_reuses_nothing_and_commits_fresh(self) -> None:
        tokens_per_block = 32
        self.prepare(32 << 20, 0, 0, 4, 128, 1, kv_buf_size=32768)
        prompt = self._prompt(tokens_per_block * 2)
        req0 = self.new_request(0, None, len(prompt), 0)
        with TemporaryCudaStream([]) as s:
            assert req0.kv_cache.resume(cast(CudaStream, s.handle))
            self.run_request(req0, tokens_per_block, True)
        s.take_finish_event().synchronize()
        self.manager.clear_reusable_blocks()  # req0 still open
        req1 = self.new_request(1, None, len(prompt), 0)
        self.assertEqual(req1.kv_cache.num_committed_tokens, 0)  # no stale match
        with TemporaryCudaStream([]) as s:
            assert req1.kv_cache.resume(cast(CudaStream, s.handle))
            self.run_request(req1, tokens_per_block, True)  # commits into the fresh tree
        s.take_finish_event().synchronize()
        self.assertEqual(self.manager.probe_reuse(None, prompt), len(prompt))
        req0.kv_cache.close(); req1.kv_cache.close()
        self.manager.clear_reusable_blocks(); self.manager.shutdown()


if __name__ == "__main__":
    unittest.main()
