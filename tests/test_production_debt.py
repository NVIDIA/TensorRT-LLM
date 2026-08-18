import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../tensorrt_llm/runtime/production_debt.py",
)
spec = importlib.util.spec_from_file_location("trt_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["trt_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtRuntimeGate = production_debt_mod.ProductionDebtRuntimeGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtRuntimeGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtRuntimeGate(
            never_equate_intent_to_approval=True,
            max_acceptable_trtdi=12.0,
        )

    def test_clean_engine_passes_readiness(self) -> None:
        report = self.gate.evaluate_runtime_engine(
            engine_id="trt_llama3_70b_tp8_node01",
            allocated_slot_bytes=16000000000,
            peak_batching_bytes=16600000000,
            allreduce_barrier_latency_us=28.5,
            engine_rebuild_cycles=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.trtdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_engine_fails_debt(self) -> None:
        report = self.gate.evaluate_runtime_engine(
            engine_id="uncalibrated_multi_gpu_engine",
            allocated_slot_bytes=16000000000,
            peak_batching_bytes=38000000000,  # High batching sprawl (2.3x)
            allreduce_barrier_latency_us=180.0,  # High barrier latency
            engine_rebuild_cycles=3,  # 3 unoptimized engine rebuilds
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.trtdi_score, 50.0)
        self.assertIn("HIGH_INFLIGHT_BATCH_SPRAWL_2.38X", report.critical_smells)
        self.assertIn("HIGH_ALLREDUCE_BARRIER_LATENCY_180.0US", report.critical_smells)
        self.assertIn("DETECTED_3_UNOPTIMIZED_ENGINE_REBUILDS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_RUNTIME_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_runtime_engine("engine-1")
        self.gate.evaluate_runtime_engine("engine-2")
        self.gate.evaluate_runtime_engine("engine-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
