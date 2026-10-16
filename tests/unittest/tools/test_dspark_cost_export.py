# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU artifact controls, not measured GPU costs or performance evidence."""

import copy
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tensorrt_llm/tools/dspark_cost_export.py"
SPEC = importlib.util.spec_from_file_location("public_dspark_cost_export_cpu", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _costs(k: int = 3, graphs: tuple[int, ...] = (16, 64), world: int = 8) -> dict:
    """Typed synthetic fixtures only; these numbers are not benchmark prices."""
    fingerprint = {
        "global_graph_batch_sizes": [g * world for g in graphs],
        "gpu": "CPU_SCHEMA_FIXTURE_NOT_GPU",
        "gpu_count": world,
        "gpu_snapshot_sha256": "a" * 64,
        "max_draft_len": k,
        "rank_local_graph_batch_sizes": list(graphs),
        "runtime_snapshot": "CPU_FIXTURE_NOT_NATIVE",
        "source_diff_sha256": "b" * 64,
        "source_head": "ORIGINAL_RECORDED_CPU_FIXTURE_SOURCE",
        "topology": "CPU_FIXTURE_NOT_FABRIC",
    }
    tables, measurements = {}, []
    for g in graphs:
        budgets = [0] if k == 1 else [0, g, g * 2]
        values = [float(index + 1) for index in range(len(budgets))]
        tables[str(g)] = {"token_counts": budgets, "step_time_ms": values}
        measurements.extend(
            {
                "rank_local_graph_batch_size": g,
                "rank_local_verifier_budget": v,
                "source_result_sha256": "c" * 64,
                "step_time_ms": value,
            }
            for v, value in zip(budgets, values)
        )
    return {
        "schema_version": 2,
        "cost_tables": tables,
        "engine_fingerprint": fingerprint,
        "engine_fingerprint_sha256": _digest(fingerprint),
        "minimum_predicted_gain": 0.01,
        "measurements": measurements,
    }


def _check(payload: dict, k: int = 3, graphs: tuple[int, ...] = (16, 64)) -> dict:
    return MODULE.check_cost_bytes(
        json.dumps(payload).encode(), physical_k=k, graph_batch_sizes=list(graphs)
    )


class PublicCostExportTests(unittest.TestCase):
    def _whole_step_summary(self, k: int = 3) -> dict:
        """Exercise the real reducer on explicitly synthetic CPU raw records."""
        utilities = MODULE._source_module(
            "cost_bridge_test_reducer", ROOT / "tensorrt_llm/tools/verify_costs.py"
        )
        costs = _costs(k=k, graphs=(16,), world=2)
        declared = []
        shapes = []
        for v in costs["cost_tables"]["16"]["token_counts"]:
            shape = {
                "workload": "dspark_chain",
                "physical_k": k,
                "graph_batch_size": 16,
                "verifier_budget": v,
                "attention_layout": "uniform" if v == 0 else "packed_chain_rows",
                "captured_head": True,
            }
            shapes.append(shape)
            declared.append(utilities.canonical_sha256(shape))
        identity = {
            "cpu_fixture_not_gpu": True,
            "generations_per_shape": 3,
            "declared_shape_ids": declared,
            "recorded_engine_fingerprint": costs["engine_fingerprint"],
            "profile": {
                "physical_k": k,
                "record_confidence_head": True,
                "measurement_scope": "whole-step",
                "cells": [
                    {"graph_batch_size": 16, "verifier_budget": shape["verifier_budget"]}
                    for shape in shapes
                ],
            },
        }
        raw = []
        for cell_index, (shape, declared_id) in enumerate(zip(shapes, declared)):
            for generation in range(3):
                for rank in range(2):
                    raw.append(
                        {
                            "schema": "verify-cost-raw-v1",
                            "rank": rank,
                            "world_size": 2,
                            "generation": generation,
                            "generation_closed": True,
                            "run_complete": True,
                            "model_shutdown_completed": True,
                            "identity": identity,
                            "timing_scope": MODULE.WHOLE_STEP_SCOPE,
                            "declared_shape_id": declared_id,
                            "execution_shape": {**shape, "observed_graph_key": [16, k]},
                            "samples": [
                                {
                                    "sample_index": cell_index * 42 + generation * 14 + i,
                                    "executor_iteration": cell_index * 42
                                    + generation * 14
                                    + i
                                    + 100,
                                    "elapsed_ms": float(rank + i + cell_index + 1),
                                    "eligible": True,
                                }
                                for i in range(14)
                            ],
                        }
                    )
        return utilities.reduce_measurements(raw)

    def test_complete_loop_reducer_to_actual_consumer_preserves_values(self) -> None:
        for k in (1, 2, 3, 5, 8):
            summary = self._whole_step_summary(k)
            before = copy.deepcopy(summary)
            data = json.dumps(summary).encode()
            output, report = MODULE.runtime_cost_bytes(data, physical_k=k, graph_batch_sizes=[16])
            payload = json.loads(output)
            self.assertEqual(summary, before)
            self.assertEqual(
                payload["engine_fingerprint"], summary["identity"]["recorded_engine_fingerprint"]
            )
            self.assertTrue(report["direct_whole_step_measurements_exported"])
            self.assertFalse(report["performance_or_economic_admission_qualified"])
            self.assertNotIn("iteration_drain_guard", payload)
            expected = {
                cell["execution_shape"]["verifier_budget"]: statistics.median(
                    cell["retained_rank_max_samples_ms"]
                )
                for cell in summary["cells"]
            }
            for row in payload["measurements"]:
                self.assertEqual(row["step_time_ms"], expected[row["rank_local_verifier_budget"]])
                self.assertEqual(row["source_result_sha256"], hashlib.sha256(data).hexdigest())
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "costs.json"
                path.write_bytes(output)
                consumer = MODULE._source_module(
                    "bridge_test_actual_consumer",
                    ROOT / "tensorrt_llm/_torch/speculative/dspark_planner.py",
                )
                table, _ = consumer.load_runtime_sps_cost_table(
                    path, max_draft_len=k, graph_batch_sizes=[16], validate_freshness=False
                )
                self.assertEqual(table.step_time(0, 16), expected[0])

    def test_complete_loop_export_refuses_missing_or_repriced_cells(self) -> None:
        for kind in ("missing", "median", "fingerprint", "K", "layout", "duplicate", "floor"):
            summary = self._whole_step_summary()
            if kind == "missing":
                summary["cells"].pop()
            elif kind == "median":
                summary["cells"][0]["rank_max_median_ms"] += 1.0
            elif kind == "fingerprint":
                del summary["identity"]["recorded_engine_fingerprint"]
            elif kind == "K":
                summary["cells"][0]["execution_shape"]["physical_k"] = 5
            elif kind == "layout":
                summary["cells"][0]["execution_shape"]["attention_layout"] = "other"
            elif kind == "duplicate":
                summary["cells"].append(copy.deepcopy(summary["cells"][0]))
            else:
                summary["sample_floors"]["pooled_per_shape"] = 1
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                MODULE.runtime_cost_bytes(
                    json.dumps(summary).encode(), physical_k=3, graph_batch_sizes=[16]
                )

    def test_existing_costs_use_actual_consumer_without_freshness_gate(self) -> None:
        for k in (1, 2, 3, 4, 5, 8):
            for world in (1, 2, 8):
                with self.subTest(k=k, world=world):
                    payload = _costs(k=k, graphs=(8, 192, 256), world=world)
                    before = copy.deepcopy(payload)
                    report = _check(payload, k=k, graphs=(8, 192, 256))
                    subset_report = _check(payload, k=k, graphs=(8,))
                    self.assertTrue(subset_report["structural_validation_passed"])
                    self.assertEqual(payload, before)
                    self.assertTrue(report["structural_validation_passed"])
                    self.assertFalse(report["destination_freshness_checked"])
                    self.assertFalse(report["new_measurements_created"])
                    self.assertFalse(report["performance_or_economic_admission_qualified"])
                    self.assertEqual(
                        report["recorded_source_head"],
                        payload["engine_fingerprint"]["source_head"],
                    )

    def test_optional_measured_drain_values_retained_without_invention(self) -> None:
        payload = _costs()
        payload["iteration_drain_guard"] = {
            "loss_multiplier": 1.5,
            "mean_output_tokens_per_request_iteration": 3.25,
            "minimum_group_value_ms": 0.2,
            "source_result_sha256": "d" * 64,
            "tail_graph_batch_size": 16,
        }
        before = copy.deepcopy(payload)
        _check(payload)
        self.assertEqual(payload, before)
        self.assertNotIn("iteration_drain_guard", _costs())

    def test_actual_runtime_candidate_bounds_refuse_before_copy(self) -> None:
        payload = _costs(graphs=(16,))
        budgets = [0, 16, 17, 18, 19, 20]
        payload["cost_tables"]["16"] = {
            "token_counts": budgets,
            "step_time_ms": [1.0] * len(budgets),
        }
        payload["measurements"] = [
            {
                "rank_local_graph_batch_size": 16,
                "rank_local_verifier_budget": v,
                "source_result_sha256": "c" * 64,
                "step_time_ms": 1.0,
            }
            for v in budgets
        ]
        with self.assertRaisesRegex(ValueError, "compact V cells"):
            _check(payload, graphs=(16,))

    def test_actual_consumer_rejects_invalid_schema_integrity_or_cells(self) -> None:
        for kind in (
            "physical_k",
            "graph_coverage",
            "fingerprint",
            "measurement",
            "native",
            "negative",
            "extra",
            "bool",
            "unknown_scope",
        ):
            payload = _costs()
            k, graphs = 3, (16, 64)
            if kind == "physical_k":
                k = 5
            elif kind == "graph_coverage":
                graphs = (16, 32)
            elif kind == "fingerprint":
                payload["engine_fingerprint_sha256"] = "f" * 64
            elif kind == "measurement":
                payload["measurements"][0]["step_time_ms"] = 99.0
            elif kind == "native":
                payload["cost_tables"]["16"]["token_counts"][0] = 1
            elif kind == "negative":
                payload["cost_tables"]["16"]["step_time_ms"][0] = -1.0
            elif kind == "extra":
                payload["measurements"].append(copy.deepcopy(payload["measurements"][0]))
            elif kind == "bool":
                payload["cost_tables"]["16"]["token_counts"][1] = True
            else:
                payload["timing_scope"] = "partial-scope-cannot-be-hidden"
            with self.subTest(kind=kind), self.assertRaises((TypeError, ValueError)):
                _check(payload, k=k, graphs=graphs)

    def test_partial_summary_refuses_even_if_eligibility_is_toggled(self) -> None:
        for schema in ("verify-cost-raw-v1", "verify-cost-measured-summary-v1"):
            for eligible in (False, True):
                payload = {
                    "schema": schema,
                    "timing_scope": "original-graph-replay-entry-including-static-copies",
                    "identity": {"identity_scope": "DIAGNOSTIC_PARTIAL_LOADED_RUNTIME"},
                    "production_sps_table_eligible": eligible,
                }
                with self.subTest(schema=schema, eligible=eligible):
                    with self.assertRaisesRegex(ValueError, "whole-step producer"):
                        _check(payload)

    def test_ambiguous_or_nonfinite_json_refuses_before_export(self) -> None:
        for data in (b'{"schema_version":2,"schema_version":2}', b'{"value":NaN}', b"[]"):
            with self.subTest(data=data), self.assertRaises((ValueError, TypeError)):
                MODULE.check_cost_bytes(data, physical_k=3, graph_batch_sizes=[16, 64])

    def test_real_standalone_cli_preserves_bytes_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "costs.json", root / "new-costs.json"
            original = json.dumps(_costs(), indent=3).encode() + b"\n\n"
            source.write_bytes(original)
            command = [
                sys.executable,
                "-I",
                "-B",
                str(SOURCE),
                "--input",
                str(source),
                "--physical-k",
                "3",
                "--graph-batch-sizes",
                "16",
                "64",
                "--output",
                str(output),
            ]
            result = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(output.read_bytes(), original)
            report = json.loads(result.stdout)
            self.assertEqual(report["source_artifact_sha256"], hashlib.sha256(original).hexdigest())
            again = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(output.read_bytes(), original)

    def test_real_complete_loop_cli_writes_only_original_direct_cells(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "CPU-summary.json", root / "CPU-costs.json"
            summary = self._whole_step_summary(k=5)
            source.write_text(json.dumps(summary), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(SOURCE),
                    "--input",
                    str(source),
                    "--physical-k",
                    "5",
                    "--graph-batch-sizes",
                    "16",
                    "--output",
                    str(output),
                ],
                cwd=root,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            payload = json.loads(output.read_bytes())
            self.assertEqual(
                payload["engine_fingerprint"], summary["identity"]["recorded_engine_fingerprint"]
            )
            self.assertEqual(payload["minimum_predicted_gain"], 0.01)
            self.assertFalse(
                json.loads(result.stdout)["performance_or_economic_admission_qualified"]
            )

    def test_real_partial_export_cli_creates_no_cost_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "partial.json", root / "must-not-exist.json"
            source.write_text(json.dumps({"schema": "verify-cost-measured-summary-v1"}))
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(SOURCE),
                    "--input",
                    str(source),
                    "--physical-k",
                    "3",
                    "--graph-batch-sizes",
                    "16",
                    "64",
                    "--output",
                    str(output),
                ],
                cwd=root,
                capture_output=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"whole-step producer", result.stderr)
            self.assertFalse(output.exists())

    def test_cpu_dependency_loading_never_initializes_native_package(self) -> None:
        _check(_costs())
        self.assertNotIn("tensorrt_llm", sys.modules)
        self.assertNotIn("torch", sys.modules)
        source = SOURCE.read_text(encoding="utf-8")
        for token in ("/scratch/", "aws-pdx", "K3", "admitted=True"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
