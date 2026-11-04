# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RULER input/telemetry checks without starting a model."""

import hashlib
import importlib.util
import itertools
import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

source = Path(__file__).parents[1] / "integration/defs/accuracy/modeling_v2/ruler.py"
spec = importlib.util.spec_from_file_location("ruler_accuracy", source)
ruler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ruler)

spec = importlib.util.spec_from_file_location("spec_evidence", source.with_name("spec_evidence.py"))
spec_evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spec_evidence)


class TestLongSpecEvidence(unittest.TestCase):
    def test_worker_snapshot_uses_client_id_for_result_join(self):
        req = SimpleNamespace(
            py_client_id=7,
            request_id=9001,
            is_dummy=False,
            orig_prompt_len=1048300,
            get_num_tokens=lambda beam: 1048305,
            py_draft_tokens=[3, 5, 8],
        )
        result = spec_evidence.request_snapshot(req, "generation")
        self.assertEqual(result["id"], 7)
        self.assertEqual(result["executor_id"], 9001)
        self.assertEqual(result["draft_tokens"], 3)

    def setUp(self):
        self.predictions = [
            dict(
                id=["niah_single_1", 1048576, i],
                request_id=i,
                phase=phase,
                prompt_tokens=1048300,
                cached_tokens=950000,
                score=1,
                spec_decoding=dict(
                    accepted=5,
                    drafted=9,
                    rejected=4,
                    drafted_per_position=[3, 3, 3, 0],
                    accepted_per_position=[3, 2, 0, 0],
                ),
            )
            for i, phase in enumerate(("first_b", "warm_a"))
        ]
        self.steps = [
            dict(
                iteration=i,
                rank=rank,
                context_tokens=32764 if rank == 0 else 0,
                requests=[
                    dict(
                        id=i if rank == 0 else -1,
                        dummy=rank != 0,
                        phase="generation",
                        prompt_tokens=1048300,
                        tokens=1048301,
                        draft_tokens=3,
                    )
                ],
            )
            for i in range(2)
            for rank in range(4)
        ]
        cold = deepcopy(self.predictions[0])
        cold.update(
            id=["ruler_vt", 65536, 0],
            request_id=2,
            phase="mixed",
            prompt_tokens=65536,
            cached_tokens=0,
        )
        self.predictions.append(cold)
        for step in self.steps:
            if step["rank"] == 0:
                step["requests"].append(
                    dict(id=2, dummy=False, phase="context", prompt_tokens=65536, draft_tokens=0)
                )

    def summarize(self):
        return spec_evidence.summarize(self.predictions, self.steps, 1_000_000, 4, True)

    def test_linked_long_evidence_and_positional_al(self):
        result = self.summarize()
        self.assertFalse(result["failures"])
        self.assertEqual(result["verified_requests"], [0, 1])
        self.assertAlmostEqual(
            result["by_length_phase"]["1048576/warm_a"]["acceptance_length"], 1 + 5 / 3
        )

    def test_short_or_unrelated_requests_do_not_prove_long_verification(self):
        for mutation in ("length", "request", "width", "dummy", "positions"):
            with self.subTest(mutation=mutation):
                steps, predictions = deepcopy(self.steps), deepcopy(self.predictions)
                for step in steps:
                    req = step["requests"][0]
                    if mutation == "length":
                        req["prompt_tokens"] = 32768
                    elif mutation == "request":
                        req["id"] = 99
                    elif mutation == "width":
                        req["draft_tokens"] = 0
                    elif mutation == "dummy":
                        req["dummy"] = True
                if mutation == "positions":
                    for row in predictions:
                        row["spec_decoding"]["drafted_per_position"] = [3, 0, 0]
                result = spec_evidence.summarize(predictions, steps, 1_000_000, 4, True)
                self.assertTrue(any("four-token" in failure for failure in result["failures"]))

    def test_separate_prefill_and_decode_are_not_mixed_evidence(self):
        self.steps[0]["requests"] = []
        for step in self.steps[1:]:
            step["context_tokens"] = 0
        self.assertTrue(any("same-step" in f for f in self.summarize()["failures"]))

    def test_cached_short_dummy_or_unrelated_context_is_not_chunked_prefill(self):
        for mutation in ("cached", "short", "dummy", "unrelated"):
            with self.subTest(mutation=mutation):
                steps, predictions = deepcopy(self.steps), deepcopy(self.predictions)
                if mutation == "cached":
                    predictions[-1]["cached_tokens"] = 65535
                for step in steps:
                    for req in step["requests"]:
                        if req["phase"] != "context":
                            continue
                        if mutation == "short":
                            req["prompt_tokens"] = 32768
                        elif mutation == "dummy":
                            req["dummy"] = True
                        elif mutation == "unrelated":
                            req["id"] = 99
                result = spec_evidence.summarize(predictions, steps, 1_000_000, 4, True)
                self.assertTrue(any("same-step" in f for f in result["failures"]))

    def test_missing_peer_is_not_dep_padding_evidence(self):
        self.steps = [step for step in self.steps if step["rank"] != 3]
        self.assertTrue(any("DEP dummy" in f for f in self.summarize()["failures"]))

    def test_no_rejection_or_no_reuse_is_inconclusive(self):
        for row in self.predictions:
            row["cached_tokens"] = 0
            row["spec_decoding"]["accepted"] = 9
        failures = self.summarize()["failures"]
        self.assertTrue(any("rejection" in f for f in failures))
        self.assertTrue(any("warm/cached-prefix" in f for f in failures))

    def test_context_budget_is_checked_per_step(self):
        self.steps[0]["context_tokens"] = 32769
        with self.assertRaises(AssertionError):
            self.summarize()


class TestRulerAccuracy(unittest.TestCase):
    def setUp(self):
        self.row = {"prompt_ids": [10, 20, 30], "max_gen_toks": 2}
        self.output = SimpleNamespace(
            finish_reason="stop", token_ids=[40], request_perf_metrics=object()
        )
        self.result = SimpleNamespace(
            prompt_token_ids=[10, 20, 30], outputs=[self.output], cached_tokens=2
        )

    def test_missing_metrics_are_not_cold_evidence(self):
        self.output.request_perf_metrics = None
        self.result.cached_tokens = 0
        with self.assertRaisesRegex(AssertionError, "missing cache telemetry"):
            ruler.cached_tokens(self.result, self.row)

    def test_illegal_cache_counts_fail(self):
        for hit in (-1, 4, True):
            with self.subTest(hit=hit):
                self.result.cached_tokens = hit
                with self.assertRaises(AssertionError):
                    ruler.cached_tokens(self.result, self.row)

    def test_valid_cold_and_warm_counts(self):
        self.assertEqual(ruler.cached_tokens(self.result, self.row), 2)
        self.result.cached_tokens = 0
        self.assertEqual(ruler.cached_tokens(self.result, self.row), 0)

    def test_accuracy_output_does_not_require_cache_telemetry(self):
        self.output.request_perf_metrics = None
        ruler.validate_output(self.result, self.row)

    def test_prompt_mutation_is_rejected(self):
        self.result.prompt_token_ids = [10, 20, 31]
        with self.assertRaisesRegex(AssertionError, "Prompt changed"):
            ruler.validate_output(self.result, self.row)

    def test_abort_is_not_a_completion(self):
        self.output.finish_reason = "abort"
        with self.assertRaises(AssertionError):
            ruler.validate_output(self.result, self.row)

    def test_changed_dataset_is_rejected_before_generation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "ruler_lite.jsonl").write_bytes(b"modified input")
            with self.assertRaisesRegex(AssertionError, "Frozen RULER inputs changed"):
                ruler.load_inputs(path, "quick", path / "tekken.json")

    def test_missing_data_points_to_setup(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "RULER.md"):
                ruler.load_inputs(Path(directory), "quick", Path("unused"))

    def test_full_uses_yaml_floors(self):
        references = ruler.load_references(
            "mistralai/Mistral-Large-4-1T-A52B", "MIXED_PRECISION", "FP8"
        )
        first = {
            (task, length, idx): 1.0
            for task in ruler.TASKS
            for length in ruler.LENGTHS
            for idx in range(50)
        }
        result = ruler.summarize(first, "full", references)
        self.assertEqual(result["n"], 1800)
        self.assertFalse(result["accuracy_failures"])
        self.assertEqual(
            result["threshold_by_length"],
            {length: row["threshold"] for length, row in references.items()},
        )
        for idx in range(50):
            first[("ruler_qa_hotpot", 32768, idx)] = 0.0
        result = ruler.summarize(first, "full", references)
        self.assertEqual(len(result["accuracy_failures"]), 1)
        self.assertIn("mean at 32768 below", result["accuracy_failures"][0])

    def test_quick_has_retrieval_gate_without_full_aggregate_floor(self):
        first = {
            (task, length, idx): 0.0 if task == "ruler_qa_hotpot" else 1.0
            for task in ruler.TASKS
            for length in ruler.LENGTHS
            for idx in range(4)
        }
        result = ruler.summarize(first, "quick", {})
        self.assertEqual(result["n"], 144)
        self.assertFalse(result["threshold_by_length"])
        self.assertFalse(result["accuracy_failures"])
        first[("niah_single_1", 4096, 0)] = 0.0
        result = ruler.summarize(first, "quick", {})
        self.assertEqual(result["accuracy_failures"], ["INCONCLUSIVE: niah_single_1@4096"])


class TestRulerTokenizerDigest(unittest.TestCase):
    def test_model_settings_change_no_token_id(self):
        with TemporaryDirectory() as scratch:
            left, right, other = (Path(scratch) / name for name in ("a.json", "b.json", "c.json"))
            tekken = {"config": {"version": "v15"}, "vocab": [{"rank": 0, "token_bytes": "QQ=="}]}
            left.write_text(json.dumps({**tekken, "model_settings_builder": {"default": "a"}}))
            right.write_text(json.dumps({"model_settings_builder": {"default": "b"}, **tekken}))
            other.write_text(json.dumps({**tekken, "vocab": []}))
            self.assertEqual(ruler.tokenizer_digest(left), ruler.tokenizer_digest(right))
            self.assertNotEqual(ruler.tokenizer_digest(left), ruler.tokenizer_digest(other))


def _first(profile, value=1.0):
    return {
        (task, length, idx): value
        for task in ruler.TASKS
        for length, count in ruler.cell_counts(profile).items()
        for idx in range(count)
    }


def _load_rows(directory, profile):
    """load_inputs with a stand-in tokenizer; the GPU test checks the checkpoint's tekken.json."""
    with TemporaryDirectory() as scratch:
        tokenizer = Path(scratch) / "tekken.json"
        tokenizer.write_text(json.dumps({"vocab": ["stand-in"]}))
        with patch.object(ruler, "TOKENIZER_SHA256", ruler.tokenizer_digest(tokenizer)):
            return ruler.load_inputs(directory, profile, tokenizer)


def _first_long(profile, value=1.0):
    return {
        (task, length, idx): value
        for (task, length), rows in ruler.LONG_ROWS[profile].items()
        for idx in rows
    }


class TestRulerLongProfile(unittest.TestCase):
    def test_cell_counts(self):
        self.assertEqual(ruler.cell_counts("quick"), dict.fromkeys(ruler.LENGTHS, 4))
        self.assertEqual(ruler.cell_counts("full"), dict.fromkeys(ruler.LENGTHS, 50))
        for profile in ("long_lite", "long_full"):
            self.assertEqual(ruler.cell_counts(profile), {65536: 10, 131072: 10, 1048576: 2})

    def test_long_rows_per_cell_are_checked(self):
        rows = [
            {"task": task, "max_length": length, "idx": idx}
            for task in ruler.TASKS
            for length, count in ruler.cell_counts("long_full").items()
            for idx in range(count)
        ]
        for dropped in (False, True):
            with self.subTest(dropped=dropped), TemporaryDirectory() as directory:
                kept = rows[:-1] if dropped else rows
                content = b"".join(json.dumps(row).encode() + b"\n" for row in kept)
                (Path(directory) / "ruler_long.jsonl").write_bytes(content)
                profile = (
                    "ruler_long.jsonl",
                    hashlib.sha256(content).hexdigest(),
                    {65536: 10, 131072: 10, 1048576: 2},
                )
                with patch.dict(ruler.PROFILES, {"long_lite": profile, "long_full": profile}):
                    if dropped:
                        with self.assertRaises(AssertionError):
                            _load_rows(Path(directory), "long_full")
                    else:
                        # These stand-in rows have no niah_single_1 rows 7 / 9 at 1M.
                        self.assertEqual(len(_load_rows(Path(directory), "long_full")), 181)
                        self.assertEqual(len(_load_rows(Path(directory), "long_lite")), 8)

    def test_long_data_beside_the_v1_files(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(FileNotFoundError, "ruler_long.jsonl.*RULER.md"):
                ruler.load_inputs(path, "long_lite", path / "tekken.json")
            (path / "ruler_long.jsonl").write_bytes(b"modified input")
            with self.assertRaisesRegex(AssertionError, "Frozen RULER inputs changed"):
                ruler.load_inputs(path, "long_lite", path / "tekken.json")

    def test_long_lite_means_are_reported_not_gated(self):
        first = _first_long("long_lite")
        result = ruler.summarize(first, "long_lite", {})
        self.assertEqual(result["n"], 10)
        self.assertEqual(result["accuracy_failures"], [])
        self.assertEqual(result["threshold_by_length"], {})
        self.assertEqual(set(result["mean_by_length"]), {"65536", "131072", "1048576"})
        self.assertEqual(result["mean_by_cell"]["niah_single_1@1048576"], 100)
        # Without its 1M rows (attention DP), long_lite reports 64K and 128K only.
        below = {key: value for key, value in first.items() if key[1] <= 131072}
        self.assertEqual(
            set(ruler.summarize(below, "long_lite", {})["mean_by_length"]), {"65536", "131072"}
        )
        first[("niah_single_1", 65536, 1)] = 0.0
        result = ruler.summarize(first, "long_lite", {})
        self.assertEqual(result["accuracy_failures"], ["INCONCLUSIVE: niah_single_1@65536"])
        self.assertEqual(result["mean_by_length"]["65536"], 75)

    def test_long_full_gates_64k_and_128k(self):
        references = ruler.load_references(
            "mistralai/Mistral-Large-4-1T-A52B", "MIXED_PRECISION", "FP8", "long_full"
        )
        first = _first_long("long_full")
        result = ruler.summarize(first, "long_full", references)
        self.assertEqual(result["n"], 183)
        self.assertEqual(result["accuracy_failures"], [])
        self.assertEqual(result["threshold_by_length"], {"65536": 93.03, "131072": 92.22})
        for idx in range(10):
            first[("ruler_qa_hotpot", 65536, idx)] = 0.0
        first[("niah_multikey_2", 131072, 3)] = 0.0
        result = ruler.summarize(first, "long_full", references)
        self.assertEqual(
            result["accuracy_failures"],
            ["FAIL: niah_multikey_2@131072", "FAIL: RULER mean at 65536 below 93.03"],
        )

    def test_quick_and_full_summaries_unchanged(self):
        references = ruler.load_references(
            "mistralai/Mistral-Large-4-1T-A52B", "MIXED_PRECISION", "FP8"
        )
        for profile, refs in (("quick", {}), ("full", references)):
            with self.subTest(profile=profile):
                result = ruler.summarize(_first(profile), profile, refs)
                self.assertEqual(
                    list(result),
                    ["n", "mean_by_length", "threshold_by_length", "accuracy_failures"],
                )
                self.assertEqual(list(result["mean_by_length"]), [str(n) for n in ruler.LENGTHS])

    def test_mixed_rows(self):
        rows = [
            {"task": task, "max_length": length, "idx": idx}
            for length in (1048576, 65536, 131072)
            for task in ruler.TASKS
            for idx in range(10)
        ]
        self.assertEqual(ruler.mixed_rows(rows, "quick", 16), rows[:8])
        self.assertEqual(ruler.mixed_rows(rows, "full", 64), rows[:32])
        picked = ruler.mixed_rows(rows, "long_full", 128)
        self.assertEqual(len(picked), 64)
        self.assertTrue(all(r["max_length"] < 1048576 for r in picked))
        self.assertTrue(all(ruler.exact_task("long_full", r["task"]) for r in picked))
        self.assertEqual(ruler.mixed_rows(rows, "long_lite", 16), rows[90:98])

    def test_long_pairs(self):
        lengths = ruler.cell_counts("long_lite")
        self.assertEqual([length for length, _, _ in ruler.LONG_PAIRS], [131072, 1048576])
        for length, a, b in ruler.LONG_PAIRS:
            self.assertLess(a, b)
            self.assertIn(length, lengths)
            for profile in ("long_lite", "long_full"):
                self.assertIn(a, ruler.LONG_ROWS[profile][("niah_single_1", length)])
                self.assertIn(b, ruler.LONG_ROWS[profile][("niah_single_1", length)])

    def test_dep_mixed_panel_preserves_rank_coverage(self):
        anchors = [
            dict(task="niah_single_1", max_length=131072, idx=i, prompt_ids=[1] * 10000 + [i])
            for i in (1, 8)
        ]
        cold = [
            dict(task="niah_single_1", max_length=65536, idx=i, prompt_ids=[2, i]) for i in range(4)
        ]
        scored = dict(task="ruler_vt", max_length=131072, idx=0, prompt_ids=[3])
        for world, pairs in ((4, 4), (8, 5)):
            with self.subTest(world=world):
                mixed = []

                def generate(batch, salts, phase):
                    if phase in ("first_b", "warm_a"):
                        return [10000]
                    if phase == "mixed":
                        mixed.append((batch, salts))
                        return [10000, 0] * (len(batch) // 2)
                    return [0] * len(batch)

                self.assertEqual(
                    ruler._reuse_panel(
                        anchors + cold + [scored],
                        generate,
                        16 * world,
                        "long_lite",
                        attention_dp_size=world,
                    ),
                    [],
                )
                batch, salts = mixed[0]
                self.assertEqual(len(batch), 2 * pairs)
                self.assertGreaterEqual(len(batch) - 2, world - 1)
                self.assertEqual(batch[1:9:2], cold)
                self.assertTrue(all(row == anchors[0] for row in batch[::2]))
                self.assertEqual(len(set(salts[1::2])), pairs)
                self.assertTrue(set(salts[::2]).isdisjoint(salts[1::2]))

    def test_long_witness_needs_half_the_shared_prefix(self):
        rows = [
            {
                "task": "niah_single_1",
                "max_length": length,
                "idx": idx,
                "prompt_ids": [length] * 10000 + [idx] * 10,
            }
            for length, a, b in ruler.LONG_PAIRS
            for idx in (a, b)
        ] + [
            {"task": "ruler_vt", "max_length": length, "idx": 0, "prompt_ids": [1]}
            for length in (65536, 1048576)
        ]
        for suffix, missing in ((10000, False), (5000, False), (4999, True)):
            with self.subTest(suffix=suffix):
                cold = []

                def generate(batch, salts, phase):
                    if phase == "first_b":
                        return [suffix]
                    if phase == "warm_a":
                        return [len(batch[0]["prompt_ids"])]
                    if phase == "mixed":
                        return [100, 0] * (len(batch) // 2)
                    if phase == "cold_a":
                        cold.append(batch[0]["max_length"])
                    return [0] * len(batch)

                expected = [
                    f"INCONCLUSIVE: missing warm/cached-context witness at {length}"
                    for length, _, _ in ruler.LONG_PAIRS
                ]
                self.assertEqual(
                    ruler._reuse_panel(rows, generate, 2, "long_lite"),
                    expected if missing else [],
                )
                # No fresh-salt replay of the 1M row: it would prefill 1M tokens again.
                self.assertEqual(cold, [131072])
        # Rows that stop below 1M (long_lite under attention DP) witness the 128K pair only.
        phases = []

        def record(batch, salts, phase):
            phases.append((phase, batch[0]["max_length"]))
            if phase == "first_b":
                return [10000]
            if phase == "warm_a":
                return [len(batch[0]["prompt_ids"])]
            if phase == "mixed":
                return [100, 0] * (len(batch) // 2)
            return [0] * len(batch)

        below = [row for row in rows if row["max_length"] <= 131072] + [
            {"task": "ruler_vt", "max_length": 131072, "idx": 0, "prompt_ids": [2]}
        ]
        self.assertEqual(ruler._reuse_panel(below, record, 2, "long_lite"), [])
        self.assertEqual({length for phase, length in phases if phase != "mixed"}, {131072})

    def test_shared_prefix(self):
        self.assertEqual(ruler.shared_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(ruler.shared_prefix([1, 2], [1, 2, 3]), 2)
        self.assertEqual(ruler.shared_prefix([5], [1]), 0)

    def test_longest_completed(self):
        rows = [
            {"task": "niah_single_1", "max_length": 1048576, "idx": 7, "prompt_ids": [0] * 7},
            {"task": "ruler_vt", "max_length": 65536, "idx": 0, "prompt_ids": [0] * 3},
        ]
        with TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(ruler.longest_completed(out, rows), 0)
            (out / "ruler_predictions.jsonl").write_text(
                json.dumps({"id": ["ruler_vt", 65536, 0]}) + "\n"
            )
            self.assertEqual(ruler.longest_completed(out, rows), 3)
            with (out / "ruler_predictions.jsonl").open("a") as stream:
                stream.write(json.dumps({"id": ["niah_single_1", 1048576, 7]}) + "\n")
            self.assertEqual(ruler.longest_completed(out, rows), 7)


DATA = Path(os.environ.get("TRTLLM_RULER_DATA", "/nonexistent"))


@unittest.skipUnless(
    (DATA / "ruler_long.jsonl").exists(), "TRTLLM_RULER_DATA has no ruler_long.jsonl"
)
class TestRulerLongData(unittest.TestCase):
    """The frozen long rows themselves: digest, lengths and the cache-witness pairs."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _load_rows(DATA, "long_full")

    def test_rows_fit_the_1m_envelope(self):
        for profile, count in (("long_lite", 10), ("long_full", 183)):
            with self.subTest(profile=profile):
                rows = _load_rows(DATA, profile)
                self.assertEqual(len(rows), count)
                for row in rows:
                    self.assertEqual(row["n_prompt_tokens"], len(row["prompt_ids"]))
                    self.assertLessEqual(row["n_prompt_tokens"], row["max_length"])
                    self.assertLessEqual(row["n_prompt_tokens"] + row["max_gen_toks"], 1048576)
                longest = max(len(row["prompt_ids"]) for row in rows)
                self.assertGreaterEqual(longest, ruler.LONG_WITNESS_TOKENS)

    def test_pairs_share_the_longest_prefix(self):
        for length, a, b in ruler.LONG_PAIRS:
            with self.subTest(length=length):
                rows = {
                    row["idx"]: row["prompt_ids"]
                    for row in self.rows
                    if row["task"] == "niah_single_1" and row["max_length"] == length
                }
                shared = ruler.shared_prefix(rows[a], rows[b])
                best = max(
                    ruler.shared_prefix(rows[x], rows[y])
                    for x, y in itertools.combinations(rows, 2)
                )
                self.assertEqual(shared, best)
                self.assertGreater(shared, 0.9 * len(rows[a]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
