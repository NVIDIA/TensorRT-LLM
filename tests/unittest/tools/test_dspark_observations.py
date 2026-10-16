# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source/CPU controls only; finite parents do not qualify native overlap/CUDA."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / "tensorrt_llm/tools"


def load(name, path, source=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if source is None:
        spec.loader.exec_module(module)
    else:
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


observations = load("tensorrt_llm.tools.dspark_observations", PUBLIC / "dspark_observations.py")
RUNTIME = ROOT / "tensorrt_llm/_torch/pyexecutor/dspark_observation_runtime.py"
runtime = load("_public_observation_runtime", RUNTIME)


def request(identifier, slot):
    return SimpleNamespace(
        py_request_id=identifier,
        py_seq_slot=slot,
        py_decoding_iter=0,
        is_dummy=False,
        py_draft_tokens=[7, 8, 9],
    )


class FiniteSampler:
    """Deferred copies covered by the finite ORIGINAL sampler event, not a GPU event."""

    def __init__(self):
        self.copies = []
        self.waits = 0

    def _copy_to_host(self, tensor):
        host = torch.zeros_like(tensor, device="cpu")
        self.copies.append((host, tensor.clone()))
        return host

    def event(self):
        copies, self.copies = self.copies, []

        def synchronize():
            self.waits += 1
            for host, snapshot in copies:
                host.copy_(snapshot)

        return SimpleNamespace(synchronize=synchronize)


class FiniteWorker:
    def __init__(self):
        self.sequence = 0
        self.confidence = torch.zeros((4, 3))
        self.stamps = torch.zeros(4, dtype=torch.int32)

    def bump_draft_seq(self):
        old = self.sequence
        self.sequence += 1
        return old

    def staged_confidence_buffer(self):
        return self.confidence

    def confidence_stamp_buffer(self):
        return self.stamps

    def confidence_row_for(self, identifier):
        return {41: 0, 42: 3}[identifier]


class FiniteOriginal:
    """Only the parent's finite call/order/buffer contract is modeled here."""

    def __init__(self):
        self.is_warmup = False
        self.iter_counter = 10
        self.worker = FiniteWorker()
        self.model_engine = SimpleNamespace(
            _collect_calibration=True,
            model=SimpleNamespace(spec_worker=self.worker),
            spec_config=SimpleNamespace(max_draft_len=3),
        )
        self.sampler = FiniteSampler()
        self.tokens = torch.zeros((3, 3), dtype=torch.int32)

    def _forward_step(self, batch, new_tensors_device=None):
        for req in batch.generation_requests:
            row = self.worker.confidence_row_for(req.py_request_id)
            self.worker.confidence[row] = torch.tensor(
                [
                    self.iter_counter + row,
                    self.iter_counter + row + 0.1,
                    self.iter_counter + row + 0.2,
                ]
            )
            self.worker.stamps[row] = self.worker.sequence
            self.tokens[req.py_seq_slot] = torch.tensor(
                [self.iter_counter * 10 + i for i in range(3)]
            )
        return {"next_draft_tokens": self.tokens}

    def _sample_async(self, batch, output):
        host = SimpleNamespace(
            next_draft_tokens=self.sampler._copy_to_host(self.tokens),
            new_tokens_lens=self.sampler._copy_to_host(torch.tensor([2, 1, 3])),
            verify_lens=None,
        )
        return SimpleNamespace(
            host=host,
            device=SimpleNamespace(next_draft_tokens=self.tokens),
            sampler_event=self.sampler.event(),
            requests=list(batch.generation_requests),
            runtime_draft_len=3,
            draft_lens=[3] * len(batch.generation_requests),
        )

    def _update_requests(self, state, resource_manager=None):
        state.sampler_event.synchronize()
        for req in state.requests:
            if getattr(req, "retired", False):
                continue
            req.py_decoding_iter += 1
            req.py_draft_tokens = state.host.next_draft_tokens[req.py_seq_slot].tolist()


def complete_overlap(mixin):
    class Executor(mixin, FiniteOriginal):
        pass

    executor = Executor()
    reqs = [request(41, 2), request(42, 0)]
    batch = SimpleNamespace(generation_requests=reqs)
    output = executor._forward_step(batch)
    source = executor._sample_async(batch, output)
    executor.iter_counter = 11
    reordered = SimpleNamespace(generation_requests=list(reversed(reqs)))
    output = executor._forward_step(reordered, source.device)
    executor._update_requests(source)
    target = executor._sample_async(reordered, output)
    executor._update_requests(target)
    return executor


def run_record(mode):
    inputs = [{"input_id": 0, "prompt_token_ids": [4, 5]}]
    return {
        "schema": "dspark-public-run-v1",
        "mode": mode,
        "run_complete": True,
        "model_shutdown_completed": True,
        "calibration_collection": False,
        "comparison_contract": {
            "physical_k": 3,
            "batch_size": 1,
            "output_tokens": 2,
            "same_cpu_fixture": True,
        },
        "elapsed_seconds": 2 if mode == "static" else 1.5,
        "resolved_confidence_enabled": mode == "dynamic",
        "runtime_confidence_enabled": mode == "dynamic",
        "routing": {"native_replays": 2, "compact_replays": 1 if mode == "dynamic" else 0},
        "inputs": inputs,
        "responses": [{**inputs[0], "output_token_ids": [6, 7], "finish_reason": "length"}],
    }


class Tests(unittest.TestCase):
    def test_whole_overlap_reordered_slots_and_delayed_original_event(self):
        executor = complete_overlap(runtime.ObservationExecutorMixin)
        payload = executor._native_pairs.payload()
        self.assertEqual(executor.sampler.waits, 2)
        self.assertEqual(
            payload["logits"].tolist()[0], [13.0, 13.100000381469727, 13.199999809265137]
        )
        self.assertEqual(
            payload["logits"].tolist()[1], [10.0, 10.100000381469727, 10.199999809265137]
        )
        self.assertEqual(payload["prefix_mask"].tolist(), [[1, 0, 0], [1, 1, 0]])
        self.assertTrue(all(join["pair_key"][2:] == [10, 11] for join in payload["meta"]["joins"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.pt"
            torch.save(payload, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(loaded["meta"]["pairing"], "draft_seq_ring")

    def test_whole_old_fixed_restored_persistent_input_overwrite(self):
        source = RUNTIME.read_text(encoding="utf-8")
        before = "input_host = self.sampler._copy_to_host(new_tensors_device.next_draft_tokens)"
        old = source.replace(before, "input_host = new_tensors_device.next_draft_tokens", 1)
        for index, variant in enumerate((old, source, old)):
            loaded = load(f"_source_boundary_{index}", RUNTIME, variant)
            if index == 1:
                self.assertEqual(
                    complete_overlap(loaded.ObservationExecutorMixin)._native_pairs.stats[
                        "recorded"
                    ],
                    2,
                )
            else:
                with self.assertRaisesRegex(ValueError, "verification input differs"):
                    complete_overlap(loaded.ObservationExecutorMixin)

    def test_wrong_step_incarnation_stale_confidence_and_censored_negatives(self):
        executor = complete_overlap(runtime.ObservationExecutorMixin)
        source = next(iter(executor._observation_states.values()))
        req = source.requests[41][0]
        arguments = dict(
            source=source,
            request=req,
            target_iteration=12,
            input_proposal=source.state.host.next_draft_tokens[2],
            accepted=2,
            verified=3,
            target_slot=2,
        )
        for fields, message in (
            ({"target_iteration": 10}, "after"),
            ({"request": request(41, 2)}, "incarnation"),
            ({"verified": 2}, "censors"),
            ({"accepted": 4}, "outside"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                observations.NativePairs(3).record(**{**arguments, **fields})
        source.stamp_host[0] += 1
        with self.assertRaisesRegex(ValueError, "producing draft"):
            observations.NativePairs(3).record(**arguments)

    def test_retired_target_is_not_labeled(self):
        class Executor(runtime.ObservationExecutorMixin, FiniteOriginal):
            pass

        executor = Executor()
        req = request(41, 2)
        batch = SimpleNamespace(generation_requests=[req])
        state = executor._sample_async(batch, executor._forward_step(batch))
        executor._update_requests(state)
        executor.iter_counter += 1
        output = executor._forward_step(batch, state.device)
        target = executor._sample_async(batch, output)
        req.retired = True
        executor._update_requests(target)
        self.assertEqual(executor._native_pairs.stats["recorded"], 0)
        self.assertEqual(executor._native_pairs.stats["retired_or_not_advanced"], 1)

    def test_same_state_cannot_update_twice(self):
        executor = complete_overlap(runtime.ObservationExecutorMixin)
        state = list(executor._observation_states.values())[-1].state
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            executor._update_requests(state)

    def test_configurable_physical_k(self):
        for k in (1, 2, 3, 4, 5, 8):
            req = request(41, 0)
            tokens = torch.arange(k).view(1, -1)
            state = SimpleNamespace(host=SimpleNamespace(next_draft_tokens=tokens))
            snapshot = observations.ProposalSnapshot(
                1, state, torch.zeros((1, k)), {41: (req, 0, 0)}, torch.tensor([8]), 8, True
            )
            pairs = observations.NativePairs(k)
            pairs.record(
                source=snapshot,
                request=req,
                target_iteration=2,
                input_proposal=tokens[0],
                accepted=k - 1,
                verified=k,
                target_slot=0,
            )
            self.assertEqual(pairs.payload()["prefix_mask"].tolist(), [[1] * (k - 1) + [0]])

    def test_prefill_placeholder_is_not_labeled_as_scored_draft(self):
        class Executor(runtime.ObservationExecutorMixin, FiniteOriginal):
            pass

        executor = Executor()
        req = request(41, 2)
        empty = SimpleNamespace(generation_requests=[])
        source = executor._sample_async(empty, executor._forward_step(empty))
        executor.iter_counter += 1
        batch = SimpleNamespace(generation_requests=[req])
        output = executor._forward_step(batch, source.device)
        executor._update_requests(source)
        target = executor._sample_async(batch, output)
        executor._update_requests(target)
        self.assertEqual(executor._native_pairs.stats["recorded"], 0)
        self.assertEqual(executor._native_pairs.stats["prefill_placeholder_no_scored_proposal"], 1)

    def test_literal_collect_payload_to_approved_public_fitter(self):
        fitter = PUBLIC / "dspark_calibration.py"
        self.assertTrue(fitter.is_file(), "Assemble the approved public fitter with this endpoint")
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            shard = cwd / "cpu-source-contract-pairs.pt"
            torch.save(
                complete_overlap(runtime.ObservationExecutorMixin)._native_pairs.payload(), shard
            )
            output = cwd / "cpu-source-contract-temperatures.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(fitter),
                    "--shards",
                    str(shard),
                    "--physical-k",
                    "3",
                    "--minimum-samples",
                    "2",
                    "--output",
                    str(output),
                ],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_bytes())
            self.assertEqual(payload["physical_k"], 3)
            self.assertEqual(payload["num_samples"], 2)
            self.assertFalse(payload["task_accuracy_or_economic_admission_qualified"])

    def test_complete_public_run_parser_has_configurable_k_bs_and_no_internal_kwargs(self):
        cli = load("_public_cli_parser_control", PUBLIC / "dspark_reproduction.py")
        name = "tensorrt_llm._torch.speculative.dspark_reproduction"
        original = sys.modules.get(name)
        seen = []
        sys.modules[name] = SimpleNamespace(run_native=lambda args: seen.append(args))
        try:
            for mode in ("static", "dynamic", "collect"):
                cli.main(
                    [
                        "run",
                        "--mode",
                        mode,
                        "--model",
                        "model",
                        "--config",
                        "config.yml",
                        "--inputs",
                        "input.jsonl",
                        "--source-root",
                        "checkout",
                        "--physical-k",
                        "4",
                        "--batch-size",
                        "128",
                        "--output-tokens",
                        "64",
                        "--timeout",
                        "25200",
                        "--output",
                        "private-output",
                    ]
                )
            self.assertEqual([args.mode for args in seen], ["static", "dynamic", "collect"])
            self.assertTrue(all(args.physical_k == 4 and args.batch_size == 128 for args in seen))
            self.assertTrue(all("executor_cls" not in vars(args) for args in seen))
        finally:
            if original is None:
                del sys.modules[name]
            else:
                sys.modules[name] = original

    def test_complete_literal_cli_compare_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            files = []
            for mode in ("static", "dynamic"):
                path = cwd / f"{mode}.json"
                path.write_text(json.dumps(run_record(mode)), encoding="utf-8")
                files.append(path)
            output = cwd / "comparison.json"
            command = [
                sys.executable,
                "-I",
                "-B",
                str(PUBLIC / "dspark_reproduction.py"),
                "compare",
                "--runs",
                *map(str, files),
                "--output",
                str(output),
            ]
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_bytes())
            self.assertAlmostEqual(report["dynamic_over_static_pct"], 100 / 3)
            self.assertIsNone(report["confidence_interval"])
            self.assertFalse(report["task_accuracy_qualified"])
            self.assertNotEqual(
                subprocess.run(command, cwd=cwd, capture_output=True, timeout=60).returncode, 0
            )

    def test_static_fallback_no_compact_unmatched_and_partial_refused(self):
        static, dynamic = run_record("static"), run_record("dynamic")
        for field, value, message in (
            ("resolved_confidence_enabled", False, "fell back"),
            ("runtime_confidence_enabled", False, "fell back"),
            ("routing", {"native_replays": 3, "compact_replays": 0}, "never executed"),
            ("run_complete", False, "closed"),
        ):
            bad = {**dynamic, field: value}
            with self.assertRaisesRegex(ValueError, message):
                observations.compare_runs([static, bad])
        different = copy.deepcopy(dynamic)
        different["comparison_contract"]["physical_k"] = 4
        with self.assertRaisesRegex(ValueError, "differ"):
            observations.compare_runs([static, different])
        different = copy.deepcopy(dynamic)
        different["responses"][0]["prompt_token_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "prompt"):
            observations.compare_runs([static, different])

    def test_internal_constructor_kwargs_not_public_llm_args(self):
        # Real argument-validation and constructor controls live in the separate
        # native-runtime test. This CPU guard prevents accidental private API injection.
        driver = (ROOT / "tensorrt_llm/_torch/speculative/dspark_reproduction.py").read_text()
        self.assertNotIn("executor_cls", driver)
        self.assertNotIn("runtime_constructors", driver)


if __name__ == "__main__":
    unittest.main()
