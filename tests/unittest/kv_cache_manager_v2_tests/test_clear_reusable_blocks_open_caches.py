# clear_reusable_blocks() with KV caches still open: a weight refit resets the reuse tree while
# requests (and the CUDA-graph padding dummy) are in flight. V1 semantics: drop the reusable
# blocks, keep every living cache and its pages; the living caches must not re-register their
# pre-reset blocks.
import unittest
from typing import cast

from kv_cache_manager_v2_tests import test_kv_cache_manager_v2 as base


class TestClearReusableBlocksWithOpenCaches(base.TestKVCacheManagerV2):
    Request = base.TestNoBatching.Request
    new_request = base.TestNoBatching.new_request
    run_request = base.TestNoBatching.run_request

    TPB = 32

    def _prepare(self) -> None:
        self.prepare(32 << 20, 0, 0, 4, 128, 1, tokens_per_block=self.TPB, kv_buf_size=32768)

    def _resume_and_run(self, req) -> None:
        with base.TemporaryCudaStream([]) as s:
            assert req.kv_cache.resume(cast(base.CudaStream, s.handle))
            self.run_request(req, self.TPB, True)
        s.take_finish_event().synchronize()

    def test_clear_with_open_cache_keeps_it_active_and_stops_commits(self) -> None:
        self._prepare()
        req = self.new_request(0, None, self.TPB * 4, 0)
        self._resume_and_run(req)  # commits 4 full blocks
        self.assertEqual(self.manager.probe_reuse(None, req.prompt), len(req.prompt))

        self.manager.clear_reusable_blocks()  # must not raise with the cache open

        self.assertEqual(self.manager.probe_reuse(None, req.prompt), 0)
        self.assertEqual(req.kv_cache.status, base._KVCache.Status.ACTIVE)
        # Further commits are no-ops for the tree: no stale block is re-registered.
        extra = [self.next_token() for _ in range(self.TPB)]
        assert req.kv_cache.resize(len(req.prompt) + self.TPB)
        req.kv_cache.commit(extra)
        self.assertEqual(self.manager.probe_reuse(None, req.prompt + extra), 0)
        self.assertEqual(self.manager.probe_reuse(None, req.prompt), 0)
        # ... but the tokens are still history: the bookkeeping that suspend() and
        # close() check (committed <= history <= capacity) continues under VIRTUAL_STOP.
        self.assertEqual(req.kv_cache.num_committed_tokens, len(req.prompt) + self.TPB)
        self.assertEqual(req.kv_cache.history_length, len(req.prompt) + self.TPB)
        req.kv_cache.suspend()
        self.assertEqual(req.kv_cache.status, base._KVCache.Status.SUSPENDED)
        req.kv_cache.close()
        self.manager.clear_reusable_blocks()

    def test_clear_with_never_committing_dummy_cache(self) -> None:
        # Mirrors the CUDA-graph padding dummy: stop_committing() right after resume.
        self._prepare()
        dummy = self.manager.create_kv_cache(base.ReuseScope(lora_id=None), [self.next_token()])
        with base.TemporaryCudaStream([]) as s:
            assert dummy.resume(cast(base.CudaStream, s.handle))
            dummy.stop_committing()
            assert dummy.resize(1)
        s.take_finish_event().synchronize()
        self.manager.clear_reusable_blocks()  # must not raise
        self.manager.clear_reusable_blocks()  # idempotent
        self.assertEqual(dummy.status, base._KVCache.Status.ACTIVE)
        dummy.close()

    def test_clear_then_same_prompt_reuses_nothing_and_commits_fresh(self) -> None:
        self._prepare()
        req0 = self.new_request(0, None, self.TPB * 2, 0)
        self._resume_and_run(req0)
        self.assertEqual(self.manager.probe_reuse(None, req0.prompt), len(req0.prompt))

        self.manager.clear_reusable_blocks()  # req0 still open

        # Same prompt again: nothing to reuse, and the fresh commits rebuild the tree.
        req1 = self.Request(
            1,
            self.manager.create_kv_cache(base.ReuseScope(lora_id=None), req0.prompt),
            req0.prompt,
            0,
        )
        self.assertEqual(req1.kv_cache.num_committed_tokens, 0)
        self._resume_and_run(req1)
        self.assertEqual(self.manager.probe_reuse(None, req0.prompt), len(req0.prompt))
        req0.kv_cache.close()
        req1.kv_cache.close()
        self.manager.clear_reusable_blocks()

    def test_clear_releases_the_stale_swa_pages_held_for_a_commit(self) -> None:
        # With commit_min_snapshot off (the default) a sliding-window layer keeps the
        # out-of-window pages of its UNcommitted blocks held while commits are allowed:
        # the commit that publishes them may still come. The reset ends that possibility,
        # so it must also end the hold -- a cache that cannot commit holds no stale pages
        # (what stop_committing() does, and what the sanity check requires outside ALLOWED).
        self._prepare()  # 2 SWA layers (window 128, 1 sink token) + 2 full-attention layers
        req = self.new_request(0, None, self.TPB * 4, 0)
        self._resume_and_run(req)  # the prompt's 4 blocks are committed; history 128
        kv = req.kv_cache
        # Generation-style growth with the commit deferred: capacity one block ahead,
        # history following. At history 320 the live window is tokens 193..320, so SWA
        # blocks 1..5 are stale: 1..3 are committed (dropped from the sequence when they
        # left the window), 4 and 5 are uncommitted and held for the commit.
        for capacity in range(self.TPB * 5, self.TPB * 11, self.TPB):
            assert kv.resize(capacity, capacity - self.TPB)
        assert kv.resize(self.TPB * 10, self.TPB * 10)
        active_before = base._introspection.active_page_stats(kv)

        self.manager.clear_reusable_blocks()

        # The two held stale pages are gone: the pages the cache still locks or holds
        # (neither free nor evictable) are exactly its live ones -- for the SWA layers the
        # sink block and the 4 window blocks (6..9), for the full-attention layers all 10
        # blocks. (The committed SWA pages of blocks 1..3 stay allocated but evictable:
        # the orphaned tree blocks the sequence references keep them until it closes.)
        swa_lcs = set(base._introspection.swa_life_cycle_ids(self.manager))
        pool_group_of = list(self.manager.get_life_cycle_pool_group_indices(base.GPU_LEVEL))

        def unavailable_pages() -> list[int]:
            return [
                s.total - s.free - s.evictable
                for s in self.manager.get_storage_statistics(base.GPU_LEVEL)
            ]

        expected = [0] * len(unavailable_pages())
        for lc, pool_group in enumerate(pool_group_of):
            expected[pool_group] += 5 if lc in swa_lcs else 10
        self.assertEqual(unavailable_pages(), expected)
        self.assertEqual(base._introspection.active_page_stats(kv), active_before)
        self.assertEqual(kv.status, base._KVCache.Status.ACTIVE)
        # The cache keeps running without a commit: a block that leaves the window now is
        # released at once (nothing can commit it), and suspend() still finds the
        # bookkeeping consistent.
        assert kv.resize(self.TPB * 11, self.TPB * 11)
        for lc, pool_group in enumerate(pool_group_of):
            expected[pool_group] += 0 if lc in swa_lcs else 1  # SWA: block 10 in, 6 out
        self.assertEqual(unavailable_pages(), expected)
        kv.suspend()
        self.assertEqual(kv.status, base._KVCache.Status.SUSPENDED)
        kv.close()
        self.manager.clear_reusable_blocks()
        # Closing the cache drops the orphaned tree blocks with it: nothing stays behind.
        stats = self.manager.get_storage_statistics(base.GPU_LEVEL)
        self.assertEqual([s.free for s in stats], [s.total for s in stats])


if __name__ == "__main__":
    unittest.main()
