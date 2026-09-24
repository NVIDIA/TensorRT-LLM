# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests; GPU numerical/concurrent admission runs separately."""

import __future__

import ast
import inspect
import json
import os
import unittest
from contextlib import nullcontext
from copy import deepcopy
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[4]
_SOURCE = _ROOT / "tensorrt_llm/_torch/cute_dsl_kernels/megamoe_shared_fc12.py"
_AUTOTUNER = _ROOT / "tensorrt_llm/_torch/autotuner.py"


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(index=0)
        self.is_cuda = True
        self.ndim = len(shape)

    def is_contiguous(self):
        return True

    def stride(self):
        strides = []
        n = 1
        for extent in reversed(self.shape):
            strides.append(n)
            n *= extent
        return tuple(reversed(strides))


def inputs(tokens):
    return [
        Tensor((tokens, 7168)),
        Tensor((1, 6144, 7168), "e4m3"),
        Tensor((1, 1376256), "uint8"),
        Tensor((1, 7168, 3072), "e4m3"),
        Tensor((1, 688128), "uint8"),
    ]


class TuningConfig(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(constraint_specs=(), distributed_tuning_strategy="independent", **kwargs)
        self.distributed_tuning_strategy = SimpleNamespace(value="independent")


class TunableRunner:
    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)


class ProfilingCache:
    def __init__(self):
        self.cache = {}

    def get_cache_key(self, op, runner, shapes, config, apply_map_to_tuning_buckets=True):
        shapes = list(shapes)
        if apply_map_to_tuning_buckets:
            shapes[0] = (runner.token_bucket(shapes[0][0]), shapes[0][1])
        return op, type(runner).__name__, str(runner.unique_id()), tuple(shapes)

    def search_cache(self, op, runners, shapes, config, apply_map_to_tuning_buckets=True):
        key = self.get_cache_key(op, runners[0], shapes, config, apply_map_to_tuning_buckets)
        return (True, *self.cache[key]) if key in self.cache else (False, 0, -1, float("inf"))


class CpuContractTests(unittest.TestCase):
    def setUp(self):
        self.tuner = SimpleNamespace(
            is_tuning_mode=False, _active_capture=None, profiling_cache=ProfilingCache()
        )
        properties = SimpleNamespace(multi_processor_count=212, major=10, minor=7)
        torch = SimpleNamespace(
            Tensor=Tensor,
            bfloat16="bf16",
            float8_e4m3fn="e4m3",
            uint8="uint8",
            cuda=SimpleNamespace(
                get_device_properties=lambda device: properties,
                device=lambda device: nullcontext(),
                current_stream=lambda device: SimpleNamespace(cuda_stream=17),
            ),
            empty_like=lambda x: Tensor(x.shape, x.dtype),
        )
        self.ns = dict(
            torch=torch,
            AutoTuner=SimpleNamespace(get=lambda: self.tuner),
            DynamicTensorSpec=SimpleNamespace,
            TunableRunner=TunableRunner,
            TuningConfig=TuningConfig,
            lru_cache=lru_cache,
            deepcopy=deepcopy,
            product=product,
        )
        module = ast.parse(_SOURCE.read_text())
        module.body = [
            node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        exec(
            compile(module, str(_SOURCE), "exec", flags=__future__.annotations.compiler_flag),
            self.ns,
        )
        self.creations = []
        self.launches = []

        def factory(*args):
            self.creations.append(args)
            launches = self.launches

            class Compiled:
                audit = {"tactic": list(args[-1])}

                def __call__(self, *tensors):
                    launches.append((args[-1], tensors[0].shape[0]))
                    return tensors[0]

            return Compiled()

        self.ns["_get_runner"] = factory

    def runner(self, sms=204, capacity=8192, clamp=10.0, stream=17):
        return self.ns["SharedFc12TunableRunner"](
            0, 7168, 3072, capacity, sms, clamp, stream, ("fixed_weight_layout",)
        )

    def test_catalog_includes_both_baselines_and_resource_decisions(self):
        catalog = self.ns["shared_fc12_candidate_tactics"]()
        self.assertEqual(len(catalog), 48)
        self.assertEqual(len(set(catalog)), 48)
        self.assertIn((1, 256, 128, "atomic_counter", True), catalog)
        self.assertIn((2, 128, 128, "grid_stride", True), catalog)
        rejected = catalog[0]
        calls = []

        def resource_check(*args):
            calls.append(args)
            if args[-1] == rejected:
                raise ValueError("insufficient test resource")

        self.ns["_make_kernel"] = resource_check
        for sms in (204, 212):
            runner = self.runner(sms)
            valid = runner.get_valid_tactics([], None)
            self.assertEqual(set(valid), set(catalog) - {rejected})
            self.assertEqual(runner.get_valid_tactics([], None), valid)
            self.assertEqual(
                runner.audit_state()["rejected_tactics"][0]["stage"], "descriptor_resources"
            )
        self.assertEqual(len(calls), 96)
        with self.assertRaises(ValueError):
            self.ns["shared_fc12_tactic_descriptor"]((1, 256, 512, "atomic_counter", True))

    def test_bucket_and_persistent_key_separation(self):
        r = self.runner()
        self.assertEqual(
            [r.token_bucket(x) for x in (1, 2, 128, 129, 8191, 8192)],
            [1, 128, 128, 256, 8192, 8192],
        )
        self.assertEqual(self.runner(capacity=300).buckets, (1, 128, 256, 300))
        self.assertNotEqual(r.unique_id(), self.runner(212).unique_id())
        self.assertNotEqual(r.unique_id(), self.runner(clamp=None).unique_id())
        self.assertNotEqual(r.unique_id(), self.runner(capacity=4096).unique_id())
        self.assertEqual(r.unique_id(), self.runner(stream=18).unique_id())
        for invalid in (0, 8193):
            with self.assertRaises(ValueError):
                r.token_bucket(invalid)

    def test_preparation_is_not_fallback_and_dynamic_t_reuses_compilation(self):
        r = self.runner()
        self.tuner.is_tuning_mode = True
        self.assertIsNone(r.forward(inputs(8192), do_preparation=True))
        self.assertEqual(self.creations, [])
        with self.assertRaises(RuntimeError):
            r.forward(inputs(8192), tactic=-1)
        tactic = (2, 128, 128, "atomic_counter", True)
        for tokens in (8192, 129, 4096, 256, 8192):
            x = inputs(tokens)
            self.assertIs(r.forward(x, tactic=tactic), x[0])
        self.assertEqual(len(self.creations), 1)
        self.assertEqual([tokens for _, tokens in self.launches], [8192, 129, 4096, 256, 8192])
        self.tuner.is_tuning_mode = False
        r.forward(inputs(256), tactic=-1)
        self.assertEqual(self.creations[-1][-1], (1, 256, 128, "atomic_counter", True))

    def test_public_dispatch_obeys_chooser_and_distinguishes_replay_fallback(self):
        tactic = (2, 64, 256, "grid_stride", False)
        choices = []

        def choose(op, runners, config, tensors):
            runner = runners[0]
            key = self.tuner.profiling_cache.get_cache_key(
                op, runner, tuple(t.shape for t in tensors), config
            )
            self.tuner.profiling_cache.cache[key] = (0, tactic, 0.25)
            choices.append(key)
            return runner, tactic

        self.tuner.choose_one = choose
        for tokens in (129, 200):
            self.ns["run_shared_fc12"](*inputs(tokens), swiglu_limit=10.0, sm_count=204)
        state = self.ns["shared_fc12_autotune_state"]()
        row = state["instances"][0]["selections"][0]
        self.assertEqual(len(choices), 2)
        self.assertEqual(choices[0], choices[1])
        self.assertEqual(len(self.creations), 1)
        self.assertEqual(row["tactic"], list(tactic))
        self.assertEqual(row["calls"], 2)
        self.assertTrue(row["cache_hit"])
        self.assertFalse(row["fallback"])
        self.assertEqual(row["min_time_ms"], 0.25)
        runner = next(iter(self.ns["_tunable_runners"].values()))
        self.tuner._active_capture = SimpleNamespace(is_replaying=lambda: True)
        runner.record_selection(self.tuner, inputs(129), tactic)
        self.tuner._active_capture = None
        runner.record_selection(self.tuner, inputs(129), -1)
        current = self.ns["shared_fc12_autotune_state"]()
        self.assertTrue(current["instances"][0]["selections"][1]["replay"])
        self.assertFalse(current["instances"][0]["selections"][2]["cache_hit"])
        self.assertTrue(current["instances"][0]["selections"][2]["fallback"])
        json.dumps(current, allow_nan=False)
        state["instances"][0]["selections"][0]["calls"] = 999
        self.assertEqual(runner.audit_state()["selections"][0]["calls"], 2)

    def test_original_autotuner_primes_each_cached_tactic_before_serving(self):
        source = ast.parse(_AUTOTUNER.read_text())
        autotuner_class = next(
            n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "AutoTuner"
        )
        method = deepcopy(
            next(
                n
                for n in autotuner_class.body
                if isinstance(n, ast.FunctionDef) and n.name == "_prime_cached_tactics"
            )
        )
        namespace = dict(
            os=os,
            inspect=inspect,
            nvtx_range=lambda name: nullcontext(),
            DistributedTuningStrategy=SimpleNamespace(MERGE=object()),
        )
        exec(
            compile(
                ast.Module(body=[method], type_ignores=[]),
                str(_AUTOTUNER),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        r = self.runner()
        tactics = [(2, 128, 128, "grid_stride", True), (1, 256, 128, "atomic_counter", True)]
        tensors = [inputs(256), inputs(8192)]
        profiles = [
            SimpleNamespace(get_opt_shapes=lambda x=x: tuple(t.shape for t in x)) for x in tensors
        ]
        for x, tactic in zip(tensors, tactics):
            key = self.tuner.profiling_cache.get_cache_key(
                "trtllm::megamoe_shared_fc12", r, tuple(t.shape for t in x), r.tuning_config
            )
            self.tuner.profiling_cache.cache[key] = (0, tactic, 0.1)
        self.tuner._optimization_profiles = lambda config, x: profiles
        self.tuner._prepare_input_tensors = lambda profile, x: inputs(
            profile.get_opt_shapes()[0][0]
        )
        self.tuner._primed_cached_tactics = set()
        self.tuner.is_tuning_mode = True
        r.tuning_config.inputs_pre_hook = None
        with patch.dict(os.environ, {"TLLM_AUTOTUNER_PRIME_CACHED_TACTICS": "1"}):
            for _ in range(2):
                namespace["_prime_cached_tactics"](
                    self.tuner, "trtllm::megamoe_shared_fc12", [r], r.tuning_config, inputs(8192)
                )
        self.assertEqual(len(self.creations), 2)
        self.assertEqual([t for t, _ in self.launches], tactics)


if __name__ == "__main__":
    unittest.main()
