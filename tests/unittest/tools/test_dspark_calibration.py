# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual CPU STS maths, input contracts and standalone CLI controls."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tensorrt_llm/tools/dspark_calibration.py"
SPEC = importlib.util.spec_from_file_location("standalone_dspark_calibration", SCRIPT)
calibration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(calibration)


class CalibrationCpuTests(unittest.TestCase):
    def test_known_cumulative_prefix_target_and_grid_order(self) -> None:
        logits = torch.zeros((4, 2), dtype=torch.float32)
        labels = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 0]], dtype=torch.float32)
        grid = calibration.default_temperature_grid()
        result = calibration.fit_sts_temperatures(logits=logits, prefix_mask=labels, grid=grid)
        self.assertEqual(grid.numel(), 41)
        self.assertEqual(result["temperatures"], [float(grid[0]), float(grid[0])])
        self.assertEqual(result["ece_before"], [0.0, 0.0])
        self.assertEqual(result["ece_after"], [0.0, 0.0])

    def test_weighted_ece_not_unweighted_bin_mean(self) -> None:
        probs = torch.tensor([0.1, 0.1, 0.1, 0.9])
        targets = torch.tensor([0.0, 0.0, 0.0, 0.0])
        expected = float((probs.to(torch.float64).sum() / 4).item())
        self.assertAlmostEqual(
            calibration.expected_calibration_error(probs=probs, targets=targets, num_bins=10),
            expected,
        )

    def test_physical_k_is_not_fixed_to_pilot_three(self) -> None:
        for width in (1, 2, 3, 4, 5, 8):
            with self.subTest(width=width):
                logits = torch.zeros((4, width))
                labels = torch.zeros((4, width))
                result = calibration.fit_sts_temperatures(
                    logits=logits, prefix_mask=labels, grid=calibration.default_temperature_grid()
                )
                self.assertEqual(len(result["temperatures"]), width)
                self.assertTrue(all(value > 0 for value in result["temperatures"]))

    def test_malformed_rows_and_censored_prefix_order_refused(self) -> None:
        grid = calibration.default_temperature_grid()
        rows = [
            (torch.empty((0, 3)), torch.empty((0, 3))),
            (torch.zeros((2, 3)), torch.zeros((2, 2))),
            (torch.zeros((2, 3), dtype=torch.int32), torch.zeros((2, 3))),
            (torch.tensor([[float("nan")]]), torch.zeros((1, 1))),
            (torch.zeros((1, 1)), torch.tensor([[float("inf")]])),
            (torch.zeros((1, 1)), torch.tensor([[0.5]])),
            (torch.zeros((1, 3)), torch.tensor([[1, 0, 1]])),
        ]
        for logits, labels in rows:
            with self.subTest(shape=list(logits.shape)), self.assertRaises(ValueError):
                calibration.fit_sts_temperatures(logits=logits, prefix_mask=labels, grid=grid)
        for bad in (torch.tensor([]), torch.tensor([0.0]), torch.tensor([float("inf")])):
            with self.subTest(grid=bad), self.assertRaises(ValueError):
                calibration.fit_sts_temperatures(
                    logits=torch.zeros((2, 1)), prefix_mask=torch.zeros((2, 1)), grid=bad
                )
        with self.assertRaises(ValueError):
            calibration.fit_sts_temperatures(
                logits=torch.zeros((2, 1)), prefix_mask=torch.zeros((2, 1)), grid=grid, num_bins=0
            )

    def test_safe_shard_pairing_width_and_duplicate_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paired.pt"
            logits = torch.zeros((4, 3))
            mask = torch.zeros((4, 3))
            torch.save(
                {"meta": {"pairing": "draft_seq_ring"}, "logits": logits, "prefix_mask": mask}, path
            )
            actual_logits, actual_mask, hashes = calibration.load_shards([path], 3)
            self.assertTrue(torch.equal(logits, actual_logits))
            self.assertTrue(torch.equal(mask, actual_mask))
            self.assertEqual(hashes[str(path)], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                calibration.load_shards([path, path], 3)
            with self.assertRaisesRegex(ValueError, "width"):
                calibration.load_shards([path], 5)
            torch.save(
                {
                    "meta": {"pairing": "unpaired_live_buffer"},
                    "logits": logits,
                    "prefix_mask": mask,
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "same stamped draft"):
                calibration.load_shards([path], 3)
            torch.save({"logits": logits, "prefix_mask": mask}, path)
            with self.assertRaisesRegex(ValueError, "pairing metadata"):
                calibration.load_shards([path], 3)

    def test_real_cli_from_unrelated_cwd_and_exclusive_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shards, output = base / "paired.pt", base / "sts.json"
            logits = torch.zeros((4, 3))
            mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)
            torch.save(
                {"meta": {"pairing": "draft_seq_ring"}, "logits": logits, "prefix_mask": mask},
                shards,
            )
            argv = [
                sys.executable,
                "-I",
                "-B",
                str(SCRIPT),
                "--shards",
                str(shards),
                "--physical-k",
                "3",
                "--output",
                str(output),
                "--minimum-samples",
                "4",
            ]
            result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_bytes())
            self.assertEqual(payload["sts_temperatures"], payload["temperatures"])
            self.assertEqual(payload["num_samples"], 4)
            self.assertEqual(payload["physical_k"], 3)
            self.assertFalse(payload["task_accuracy_or_economic_admission_qualified"])
            self.assertFalse(payload["artifact_freshness_validated"])
            before = output.read_bytes()
            again = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("FileExistsError", again.stderr)
            self.assertEqual(output.read_bytes(), before)

    def test_no_native_trtllm_import_and_output_does_not_fabricate_admission(self) -> None:
        text = SCRIPT.read_text()
        self.assertNotIn("import tensorrt_llm", text)
        self.assertNotIn("torch.cuda", text)
        self.assertIn("weights_only=True", text)
        self.assertNotIn('"admitted": True', text)
        self.assertNotIn("/scratch/", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
