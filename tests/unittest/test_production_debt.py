# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../tensorrt_llm/runtime/production_debt.py",
)
spec = importlib.util.spec_from_file_location("trtllm_runtime_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["trtllm_runtime_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtModelRunnerGate = production_debt_mod.ProductionDebtModelRunnerGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtModelRunnerGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtModelRunnerGate(
            never_equate_intent_to_approval=True,
            max_acceptable_tdi=12.0,
        )

    def test_clean_runner_step_passes_readiness(self) -> None:
        report = self.gate.evaluate_runner_step(
            runner_step_id="trtllm_cuda_graph_h100_step",
            allocated_workspace_bytes=16000000000,
            utilized_workspace_bytes=16800000000,
            runner_step_latency_ms=2.6,
            cuda_graph_invalidation_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.tdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_runner_step_fails_debt(self) -> None:
        report = self.gate.evaluate_runner_step(
            runner_step_id="uncalibrated_trtllm_runner_step",
            allocated_workspace_bytes=16000000000,
            utilized_workspace_bytes=45000000000,  # 2.81x workspace allocation sprawl
            runner_step_latency_ms=35.0,  # High runner step latency
            cuda_graph_invalidation_stalls=3,  # 3 CUDA graph invalidation stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.tdi_score, 50.0)
        self.assertIn("HIGH_WORKSPACE_ALLOCATION_SPRAWL_2.81X", report.critical_smells)
        self.assertIn("HIGH_RUNNER_STEP_LATENCY_35.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_CUDA_GRAPH_INVALIDATION_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_QUANT_SCALE_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_runner_step("step-1")
        self.gate.evaluate_runner_step("step-2")
        self.gate.evaluate_runner_step("step-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
