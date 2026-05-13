import os
import tempfile
import time
from unittest.mock import patch


# ---------------------------------------------------------------------------
# PerfTimer tests
# ---------------------------------------------------------------------------
class TestPerfTimer:
    def _make_timer(self):
        from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfTimer

        return PerfTimer()

    def test_task_latency_positive(self):
        timer = self._make_timer()
        timer.record_task_start(0)
        time.sleep(0.01)
        timer.record_task_end(0)
        assert timer.get_task_latency(0) > 0

    def test_transfer_throughput_positive(self):
        timer = self._make_timer()
        size_bytes = 1024 * 1024  # 1 MiB
        timer.record_transfer_sizes(0, size_bytes, count=4)
        timer.record_transfer_start(0)
        time.sleep(0.01)
        timer.record_transfer_end(0)

        throughput = timer.get_transfer_throughput(0)
        assert throughput > 0  # MB/s

    def test_unrecorded_peer_returns_zero(self):
        timer = self._make_timer()
        assert timer.get_task_latency(99) == 0.0
        assert timer.get_transfer_latency(99) == 0.0
        assert timer.get_transfer_throughput(99) == 0.0
        assert timer.get_prepare_args_latency(99) == 0.0
        assert timer.get_queue_latency(99) == 0.0
        assert timer.get_transfer_size(99) == 0
        assert timer.get_average_segment_size(99) == 0.0
        assert timer.get_transfer_entry_count(99) == 0


# ---------------------------------------------------------------------------
# PerfLogManager tests
# ---------------------------------------------------------------------------
def _reset_singleton():
    """Reset PerfLogManager singleton so each test gets a fresh instance."""
    from tensorrt_llm._torch.disaggregation.native import perf_logger

    perf_logger.PerfLogManager._instance = None


class TestPerfLogManager:
    def setup_method(self):
        _reset_singleton()

    def teardown_method(self):
        _reset_singleton()

    @patch.dict("os.environ", {"TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO": "1"}, clear=False)
    def test_enabled_when_env_set(self):
        from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfLogManager

        mgr = PerfLogManager()
        assert mgr.enabled is True

    @patch.dict("os.environ", {}, clear=False)
    def test_disabled_when_env_unset(self):
        import os

        os.environ.pop("TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO", None)
        from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfLogManager

        mgr = PerfLogManager()
        assert mgr.enabled is False
        # log() should be a no-op, not crash
        mgr.log("inst", 0, "csv_line", "info_msg")

    def test_log_gen_transfer_summary_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                "os.environ",
                {"TRTLLM_KVCACHE_TIME_OUTPUT_PATH": tmpdir},
                clear=False,
            ):
                from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfLogManager

                mgr = PerfLogManager()
                mgr.log_gen_transfer_summary(
                    unique_rid=42,
                    instance_name="test_inst",
                    instance_rank=0,
                    gen_side_transfer_time_ms=12.345,
                    kv_cache_size=1024,
                )
                csv_path = os.path.join(tmpdir, "test_inst_0_gen_transfer_summary.csv")
                assert os.path.exists(csv_path), f"CSV file not created at {csv_path}"
                with open(csv_path) as f:
                    lines = f.readlines()
                assert len(lines) >= 2  # header + at least 1 data row
                assert "gen_side_transfer_time(ms)" in lines[0]
                assert "42" in lines[1]
                assert "12.345" in lines[1]
                assert "1024" in lines[1]

    def test_log_gen_transfer_summary_disabled_without_env(self):
        import os as _os

        _os.environ.pop("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", None)
        from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfLogManager

        mgr = PerfLogManager()
        # Should be a no-op, not crash
        mgr.log_gen_transfer_summary(
            unique_rid=1,
            instance_name="test",
            instance_rank=0,
            gen_side_transfer_time_ms=0.0,
            kv_cache_size=0,
        )

    @patch.dict("os.environ", {"TLLM_ENABLE_CACHE_TRANSFER_PERF_INFO": "1"}, clear=False)
    def test_log_task_perf_does_not_crash(self):
        from tensorrt_llm._torch.disaggregation.native.perf_logger import PerfLogManager, PerfTimer

        mgr = PerfLogManager()
        timer = PerfTimer()
        timer.record_task_start(0)
        timer.record_task_end(0)
        timer.record_transfer_sizes(0, 4096, count=2)
        timer.record_transfer_start(0)
        timer.record_transfer_end(0)
        # Should not raise — just logs to stdout via logger.info
        mgr.log_task_perf(
            task_type="KVSendTask",
            unique_rid=1,
            peer_rank=0,
            instance_name="test",
            instance_rank=0,
            perf_timer=timer,
        )
