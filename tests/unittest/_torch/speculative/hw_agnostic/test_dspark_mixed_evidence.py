# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, deliberately synthetic protocol fixtures; no production evidence."""

import copy
import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tensorrt_llm._torch.speculative import dspark_mixed_evidence as mixed
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, DSparkMixedEvidenceConfig


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity():
    return mixed.MixedRuntimeIdentity(*("abcdef"[index] * 64 for index in range(6)))


def geometry():
    # Rank 0 prefills 64 rows; rank 1 decodes with native 32 versus compact 16.
    # Every bootstrap/dummy generation is still charged its full K+1 rows.
    features = (
        (64, 4096, 8192, 0, 0, 0, 0, 1, 0, 1, 0, 0),
        (0, 0, 0, 0, 1024, 128, 0, 0, 8, 0, 8, 0),
    ) + ((0,) * 12,) * 6
    return mixed.MixedGeometry(features, 96, (64, 16, 0, 0, 0, 0, 0, 0), (64, 32, 0, 0, 0, 0, 0, 0))


def fixture():
    live = identity()
    table = {
        "schema": "dspark-mixed-breakable-costs-v1",
        "identity": live.payload(),
        "protocol_sha256": "1" * 64,
        "action_semantics": mixed._ACTION,
        "pricing_semantics": mixed._PRICE_SEMANTICS,
        "synthetic": False,
        "provenance": "closed_measured_breakable_mixed",
        "minimum_predicted_gain": 0.01,
        "cells": [
            {
                "geometry": geometry().payload(),
                "native_lower_ms": 7.0,
                "candidate_upper_ms": 6.9,
                "measurement_result_sha256": "2" * 64,
            }
        ],
    }
    receipt = {
        "schema": "dspark-mixed-breakable-admission-v1",
        "identity": live.payload(),
        "costs_sha256": sha(encode(table)),
        "protocol_sha256": "1" * 64,
        "action_semantics": mixed._ACTION,
        "pricing_semantics": mixed._PRICE_SEMANTICS,
        "production_admission": True,
        "source_and_measurements_verified": True,
        "accuracy_pass": True,
        "runtime_invariants_pass": True,
        "matched_workload_pass": True,
        "accuracy_result_sha256": "3" * 64,
        "runtime_invariants_result_sha256": "4" * 64,
        "matched_workload_result_sha256": "5" * 64,
        "selector_replay_sha256": "6" * 64,
        "measurement_results_sha256": sha(encode(["2" * 64])),
        "economics": {
            "policy_steps": 10,
            "compact_choices": 4,
            "gross_compact_value_ms_lower_bound": 3.0,
            "fixed_confidence_path_overhead_ms_upper_bound": 0.1,
            "safety_margin_ms": 0.5,
        },
    }
    return table, receipt, live


def resolve(table, receipt, live=None, *, update_receipt=True):
    table_raw = encode(table)
    receipt = copy.deepcopy(receipt)
    if update_receipt:
        receipt["costs_sha256"] = sha(table_raw)
    receipt_raw = encode(receipt)
    binding = mixed.MixedEvidenceBinding(sha(table_raw), sha(receipt_raw), "1" * 64)
    return mixed.resolve_mixed_evidence(
        binding, costs_raw=table_raw, admission_raw=receipt_raw, live_identity=live or identity()
    )


class TestMixedEvidence(unittest.TestCase):
    def assert_native(self, table, receipt, live=None, **kwargs):
        result = resolve(table, receipt, live, **kwargs)
        self.assertIsNone(result.costs)
        self.assertTrue(result.reason.startswith("mixed_evidence_invalid_native:"))

    def test_01_actual_binding_positive_synthetic_fixture_only(self):
        table, receipt, live = fixture()
        resolved = resolve(table, receipt, live)
        self.assertEqual(resolved.reason, "mixed_evidence_bound")
        self.assertEqual(resolved.costs.lookup(geometry()).candidate_upper_ms, 6.9)
        self.assertEqual(resolved.costs.minimum_predicted_gain, 0.01)

    def test_02_bad_fixed_restored_old_validation_schema(self):
        table, receipt, live = fixture()
        table["schema"] = "dspark-host-ragged-bcg-prices-v1"
        self.assert_native(table, receipt, live)
        table["schema"] = "dspark-mixed-breakable-costs-v1"
        self.assertIsNotNone(resolve(table, receipt, live).costs)
        table["schema"] = "dspark-host-ragged-bcg-prices-v1"
        self.assert_native(table, receipt, live)

    def test_03_absent_optional_binding_does_not_read_payloads(self):
        result = mixed.resolve_mixed_evidence(
            None, costs_raw=None, admission_raw=None, live_identity=identity()
        )
        self.assertEqual(result.reason, "mixed_evidence_absent_native")
        self.assertIsNone(result.costs)

    def test_04_decode_policy_object_and_flag_unchanged_for_all_outcomes(self):
        decode = {"enabled": True, "identity": "existing-decode", "table": object()}
        before = dict(decode)
        table, receipt, live = fixture()
        for binding in (None, mixed.MixedEvidenceBinding("0" * 64, "1" * 64, "2" * 64)):
            mixed.resolve_mixed_evidence(
                binding, costs_raw=b"bad", admission_raw=b"bad", live_identity=live
            )
            self.assertEqual(decode, before)
            self.assertTrue(decode["enabled"])
        self.assertIsNotNone(resolve(table, receipt, live).costs)
        self.assertEqual(decode, before)

    def test_05_live_identity_is_not_read_from_table(self):
        table, receipt, live = fixture()
        for field in (
            "source_tree_sha256",
            "runtime_snapshot_sha256",
            "model_config_sha256",
            "gpu_class_sha256",
            "topology_sha256",
            "selector_identity_sha256",
        ):
            with self.subTest(field=field):
                self.assert_native(table, receipt, dataclasses.replace(live, **{field: "9" * 64}))

    def test_06_invalid_supported_runtime_and_mutable_grid_rejected(self):
        live = identity()
        for overrides in (
            {"physical_k": 2},
            {"physical_k": True},
            {"world_size": 1},
            {"execution_mode": "FULL"},
            {"body_buckets": list(live.body_buckets)},
            {"body_buckets": (96, 128, 192, 256)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                dataclasses.replace(live, **overrides)

    def test_07_payloads_are_deeply_frozen_into_typed_tuples(self):
        table, receipt, live = fixture()
        costs = resolve(table, receipt, live).costs
        table["cells"][0]["geometry"]["rank_features"][1]["eligible_generations"] = 0
        table["cells"][0]["native_lower_ms"] = 0
        self.assertEqual(costs.lookup(geometry()).native_lower_ms, 7.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            costs.cells[0].candidate_upper_ms = 1

    def test_08_uncovered_geometry_is_native_no_interpolation(self):
        table, receipt, live = fixture()
        costs = resolve(table, receipt, live).costs
        unseen = dataclasses.replace(geometry(), actual_rank_rows=(64, 15, 0, 0, 0, 0, 0, 0))
        self.assertIsNone(costs.lookup(unseen))
        self.assertIsNone(costs.lookup(geometry().payload()))

    def test_09_duplicate_geometry_rejects_favorable_reselection(self):
        table, receipt, _ = fixture()
        table["cells"].append(copy.deepcopy(table["cells"][0]))
        table["cells"][1]["candidate_upper_ms"] = 1
        self.assert_native(table, receipt)

    def test_10_cell_list_empty_or_unbounded_rejected(self):
        table, receipt, _ = fixture()
        for cells in ([], [table["cells"][0]] * 257):
            bad = copy.deepcopy(table)
            bad["cells"] = cells
            self.assert_native(bad, receipt)

    def test_11_context_and_fixed_generation_rows_cannot_be_discounted(self):
        original = geometry()
        for change in (
            {"actual_rank_rows": (63, 16, 0, 0, 0, 0, 0, 0)},
            {"actual_rank_rows": (64, 7, 0, 0, 0, 0, 0, 0)},
            {"native_rank_rows": (64, 31, 0, 0, 0, 0, 0, 0)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                dataclasses.replace(original, **change)
        features = list(original.rank_features)
        features[1] = (0, 0, 0, 0, 1024, 128, 1, 0, 7, 0, 8, 0)
        with self.assertRaises(ValueError):
            dataclasses.replace(original, rank_features=tuple(features))

    def test_12_decode_only_and_noncompact_geometry_rejected(self):
        original = geometry()
        features = list(original.rank_features)
        features[0] = (64, 4096, 8192, 0, 0, 0, 0, 0, 0, 1, 0, 0)
        with self.assertRaises(ValueError):
            dataclasses.replace(original, rank_features=tuple(features))
        with self.assertRaises(ValueError):
            dataclasses.replace(original, actual_rank_rows=original.native_rank_rows)

    def test_13_rank_order_is_not_coalesced(self):
        table, receipt, live = fixture()
        costs = resolve(table, receipt, live).costs
        original = geometry()
        reordered = dataclasses.replace(
            original,
            rank_features=original.rank_features[::-1],
            actual_rank_rows=original.actual_rank_rows[::-1],
            native_rank_rows=original.native_rank_rows[::-1],
        )
        self.assertIsNone(costs.lookup(reordered))

    def test_14_unknown_null_and_validation_only_fields_reject(self):
        table, receipt, _ = fixture()
        for field, value in (
            ("validation_only", False),
            ("pricing_sequence", [96]),
            ("confidence_sps_table_path", "old.json"),
            ("unknown", None),
        ):
            bad = copy.deepcopy(table)
            bad[field] = value
            self.assert_native(bad, receipt)

    def test_15_synthetic_prices_and_claim_only_receipts_reject(self):
        table, receipt, _ = fixture()
        for field in (
            "production_admission",
            "source_and_measurements_verified",
            "accuracy_pass",
            "runtime_invariants_pass",
            "matched_workload_pass",
        ):
            bad = copy.deepcopy(receipt)
            bad[field] = False
            self.assert_native(table, bad)
        table["synthetic"] = True
        self.assert_native(table, receipt)

    def test_16_proxy_is_never_relabelled_as_whole_step(self):
        table, receipt, _ = fixture()
        for key in ("table", "receipt"):
            bad_table, bad_receipt = copy.deepcopy(table), copy.deepcopy(receipt)
            (bad_table if key == "table" else bad_receipt)["pricing_semantics"] = (
                "qualified_group_whole_step"
            )
            self.assert_native(bad_table, bad_receipt)

    def test_17_protocol_action_and_measurement_binding_reject_drift(self):
        table, receipt, _ = fixture()
        for key, value in (
            ("protocol_sha256", "9" * 64),
            ("action_semantics", "uniform-L"),
            ("measurement_results_sha256", "9" * 64),
        ):
            bad = copy.deepcopy(receipt)
            bad[key] = value
            self.assert_native(table, bad)

    def test_18_all_result_hashes_required(self):
        table, receipt, _ = fixture()
        for field in (
            "accuracy_result_sha256",
            "runtime_invariants_result_sha256",
            "matched_workload_result_sha256",
            "selector_replay_sha256",
            "measurement_results_sha256",
        ):
            bad = copy.deepcopy(receipt)
            bad[field] = "not-a-hash"
            self.assert_native(table, bad)

    def test_19_positive_economics_recomputed_no_boolean_override(self):
        table, receipt, _ = fixture()
        for change in (
            {"compact_choices": 0},
            {"gross_compact_value_ms_lower_bound": 1.0},
            {"policy_steps": True},
            {"safety_margin_ms": -1.0},
            {"compact_choices": 11},
        ):
            bad = copy.deepcopy(receipt)
            bad["economics"].update(change)
            self.assert_native(table, bad)

    def test_20_no_relaxation_of_one_percent_immediate_gate(self):
        table, receipt, _ = fixture()
        for margin in (0.0, 0.009, True):
            table["minimum_predicted_gain"] = margin
            self.assert_native(table, receipt)
        table["minimum_predicted_gain"] = 0.02
        self.assertEqual(resolve(table, receipt).costs.minimum_predicted_gain, 0.02)

    def test_21_nonpositive_and_nonfinite_costs_rejected(self):
        table, receipt, _ = fixture()
        for field in ("native_lower_ms", "candidate_upper_ms"):
            for value in (0, -1, True, "7.0"):
                bad = copy.deepcopy(table)
                bad["cells"][0][field] = value
                self.assert_native(bad, receipt)
        raw = encode(table).replace(b'"candidate_upper_ms":6.9', b'"candidate_upper_ms":1e9999')
        receipt["costs_sha256"] = sha(raw)
        self.assert_raw_native(raw, encode(receipt))

    def assert_raw_native(self, table_raw, receipt_raw):
        binding = mixed.MixedEvidenceBinding(sha(table_raw), sha(receipt_raw), "1" * 64)
        result = mixed.resolve_mixed_evidence(
            binding, costs_raw=table_raw, admission_raw=receipt_raw, live_identity=identity()
        )
        self.assertIsNone(result.costs)

    def test_22_duplicate_nonfinite_malformed_deep_json_rejected(self):
        _, receipt, _ = fixture()
        for raw in (
            b'{"schema":"x","schema":"y"}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b"{malformed",
            b"\xff",
            b"[" * 1500 + b"]" * 1500,
        ):
            self.assert_raw_native(raw, encode(receipt))

    def test_23_missing_raw_oversized_and_digest_drift_rejected(self):
        table, receipt, live = fixture()
        raw, qraw = encode(table), encode(receipt)
        binding = mixed.MixedEvidenceBinding(sha(raw), sha(qraw), "1" * 64)
        for value in (None, raw + b" ", bytearray(raw), b"", b" " * (mixed._MAX_BYTES + 1)):
            result = mixed.resolve_mixed_evidence(
                binding, costs_raw=value, admission_raw=qraw, live_identity=live
            )
            self.assertIsNone(result.costs)

    def test_24_table_receipt_cross_binding_rejected(self):
        table, receipt, _ = fixture()
        table["cells"][0]["candidate_upper_ms"] = 6.8
        self.assert_native(table, receipt, update_receipt=False)

    def test_25_rank_extents_strict_types_and_wire_limits(self):
        original = geometry()
        for value in (True, 1.0, -1, 2**63):
            features = list(original.rank_features)
            changed = list(features[0])
            changed[2] = value
            features[0] = tuple(changed)
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                dataclasses.replace(original, rank_features=tuple(features))

    def test_26_decode_schema_is_not_mixed_evidence(self):
        _, receipt, _ = fixture()
        self.assert_raw_native(encode({"schema_version": 2}), encode(receipt))

    def test_27_valid_bootstrap_reserves_full_k_plus_one_rows(self):
        original = geometry()
        features = list(original.rank_features)
        features[1] = (0, 0, 0, 0, 1024, 128, 1, 0, 7, 0, 8, 4)
        valid = dataclasses.replace(
            original, rank_features=tuple(features), actual_rank_rows=(64, 20, 0, 0, 0, 0, 0, 0)
        )
        self.assertEqual(valid.rank_features[1][-1], 4)
        with self.assertRaises(ValueError):
            dataclasses.replace(valid, actual_rank_rows=(64, 10, 0, 0, 0, 0, 0, 0))

    def test_28_bad_fixed_restored_cost_pin_is_not_relabelled(self):
        table, receipt, live = fixture()
        raw, qraw = encode(table), encode(receipt)
        bad_binding = mixed.MixedEvidenceBinding("0" * 64, sha(qraw), "1" * 64)
        for binding, expected in (
            (bad_binding, False),
            (mixed.MixedEvidenceBinding(sha(raw), sha(qraw), "1" * 64), True),
            (bad_binding, False),
        ):
            result = mixed.resolve_mixed_evidence(
                binding, costs_raw=raw, admission_raw=qraw, live_identity=live
            )
            self.assertEqual(result.costs is not None, expected)

    def test_29_file_binding_requires_independent_live_fingerprint(self):
        table, receipt, live = fixture()
        table_raw, receipt_raw = encode(table), encode(receipt)
        live_raw = encode(
            {"schema": "dspark-mixed-live-fingerprint-v1", "identity": live.payload()}
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("costs.json", "receipt.json", "live.json")]
            for path, raw in zip(paths, (table_raw, receipt_raw, live_raw)):
                path.write_bytes(raw)
            config = SimpleNamespace(
                costs_path=str(paths[0]),
                costs_sha256=sha(table_raw),
                admission_path=str(paths[1]),
                admission_sha256=sha(receipt_raw),
                protocol_sha256="1" * 64,
                live_fingerprint_path=str(paths[2]),
                live_fingerprint_sha256=sha(live_raw),
            )
            self.assertIsNotNone(mixed.load_mixed_evidence(config).costs)
            config.live_fingerprint_sha256 = "0" * 64
            self.assertIsNone(mixed.load_mixed_evidence(config).costs)
            self.assertEqual(mixed.load_mixed_evidence(None).reason, "mixed_evidence_absent_native")

    def test_30_typed_config_rejects_invalid_pins_and_unknown_fields(self):
        data = dict(
            costs_path="costs.json",
            costs_sha256="1" * 64,
            admission_path="receipt.json",
            admission_sha256="2" * 64,
            protocol_sha256="3" * 64,
            live_fingerprint_path="live.json",
            live_fingerprint_sha256="4" * 64,
        )
        self.assertEqual(DSparkMixedEvidenceConfig(**data).costs_path, "costs.json")
        with self.assertRaises(ValueError):
            DSparkMixedEvidenceConfig(**{**data, "costs_sha256": "invalid"})
        with self.assertRaises(ValueError):
            DSparkMixedEvidenceConfig(**data, force=True)

    def test_31_feature_off_rejects_mixed_input_and_k1_clears_it(self):
        config = SimpleNamespace(
            enable_confidence_scheduling=False,
            enable_fused_confidence_scheduler=False,
            confidence_sts_path=None,
            confidence_sps_table_path=None,
            confidence_sps_live_fingerprint_path=None,
            confidence_admission_receipt_path=None,
            confidence_admission_receipt_sha256=None,
            confidence_mixed_evidence=object(),
            max_draft_len=3,
        )
        with self.assertRaisesRegex(ValueError, "require enable_confidence_scheduling"):
            DSparkDecodingConfig.validate_confidence_scheduling(config)
        config.enable_confidence_scheduling, config.max_draft_len = True, 1
        self.assertIs(DSparkDecodingConfig.validate_confidence_scheduling(config), config)
        self.assertFalse(config.enable_confidence_scheduling)
        self.assertIsNone(config.confidence_mixed_evidence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
