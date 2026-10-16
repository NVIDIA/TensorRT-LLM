# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local CPU-only checks of the native AllReduce bucket source and Python consumers."""

import ast
import copy
import ctypes
import itertools
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _Config:
    dynamic_tensor_specs: tuple
    tune_max_num_tokens: int | None = None


@unittest.skipUnless(shutil.which("g++"), "g++ is required")
class TestAllReduceTuningBuckets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (_ROOT / "cpp/tensorrt_llm/thop/allreduceOp.cpp").read_text()
        native = (
            "struct AllReduceTacticCacheKey"
            + source.split("struct AllReduceTacticCacheKey", 1)[1].split(
                "AllReduceInputShapes getAllReduceStaticInputShapes", 1
            )[0]
        )
        prelude = """
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <stdexcept>
#include <tuple>
#include <vector>
using AllReduceInputShapes = std::array<std::vector<int64_t>, 6>;
using AllReduceFusionOp = int;
namespace c10 { using ScalarType = int; }
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
"""
        bridge = """
extern "C" {
int configure_ceiling(int64_t ceiling) {
    try { getAllReduceTacticCache().configure(ceiling); return 1; }
    catch (std::runtime_error const&) { return 0; }
}
int64_t bucket_count() { return getAllReduceTacticCache().getBuckets().size(); }
int64_t bucket_at(int64_t index) { return getAllReduceTacticCache().getBuckets().at(index); }
int64_t map_tokens(int64_t tokens) { return getAllReduceTacticCache().mapToBucket(tokens); }
int insert_tactic(int64_t group, int64_t bucket, int64_t tactic) {
    try {
        getAllReduceTacticCache().insert({{group}, 0, false, 0, {}}, bucket, tactic);
        return 1;
    } catch (std::runtime_error const&) { return 0; }
}
int64_t lookup_tactic(int64_t group, int64_t tokens) {
    return getAllReduceTacticCache().find({{group}, 0, false, 0, {}}, tokens).value_or(-2);
}
void clear_tactics() { getAllReduceTacticCache().clear(); }
int check_lazy_bucket_lifecycle() {
    AllReduceTacticCache engine;
    AllReduceTacticCacheKey key{{0}, 0, false, 0, {}};
    if (engine.mapToBucket(1) != -1 || engine.find(key, 1).has_value()) return 1;
    auto buckets = engine.configure(40000);
    if (buckets.back() != 40000 || engine.mapToBucket(40000) != 40000) return 2;
    engine.insert(key, 40000, 8);
    engine.configure(40000);
    if (engine.find(key, 40000) != 8) return 3;
    AllReduceTacticCache standalone;
    if (standalone.mapToBucket(1) != -1) return 4;
    if (standalone.getBuckets().back() != 16384) return 5;
    standalone.insert(key, 16384, 8);
    standalone.getBuckets();
    if (standalone.find(key, 16384) != 8) return 6;
    AllReduceTacticCache directRegistration;
    directRegistration.insert(key, 8192, 8);
    if (directRegistration.find(key, 8192) != 8) return 7;
    return 0;
}
}
"""
        cls._directory = tempfile.TemporaryDirectory(prefix="allreduce-buckets-")
        cls.addClassCleanup(cls._directory.cleanup)
        cpp = Path(cls._directory.name) / "buckets.cpp"
        library = Path(cls._directory.name) / "buckets.so"
        cpp.write_text(prelude + native + bridge)
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                "-shared",
                "-pthread",
                str(cpp),
                "-o",
                str(library),
            ],
            check=True,
        )
        cls.native = ctypes.CDLL(str(library))
        for name, count, result in (
            ("configure_ceiling", 1, ctypes.c_int),
            ("bucket_count", 0, ctypes.c_int64),
            ("bucket_at", 1, ctypes.c_int64),
            ("map_tokens", 1, ctypes.c_int64),
            ("insert_tactic", 3, ctypes.c_int),
            ("lookup_tactic", 2, ctypes.c_int64),
            ("clear_tactics", 0, None),
            ("check_lazy_bucket_lifecycle", 0, ctypes.c_int),
        ):
            function = getattr(cls.native, name)
            function.argtypes = [ctypes.c_int64] * count
            function.restype = result

    def setUp(self) -> None:
        self.configure(16384)
        self.native.clear_tactics()

    def buckets(self) -> tuple[int, ...]:
        return tuple(self.native.bucket_at(i) for i in range(self.native.bucket_count()))

    def configure(self, ceiling: int) -> tuple[int, ...]:
        if not self.native.configure_ceiling(ceiling):
            raise ValueError("Invalid ceiling")
        return self.buckets()

    def test_native_buckets_are_initialized_on_demand(self) -> None:
        self.assertEqual(self.native.check_lazy_bucket_lifecycle(), 0)

    def test_bucket_generation(self) -> None:
        base = tuple(1 << exponent for exponent in range(15))
        for ceiling, expected in (
            (1, (1,)),
            (3, (1, 2, 3)),
            (4096, base[:13]),
            (6000, (*base[:13], 6000)),
            (8192, base[:14]),
            (16384, base),
            (16385, (*base, 16385)),
            (20000, (*base, 20000)),
            (40000, (*base, 32768, 40000)),
            (65536, (*base, 32768, 65536)),
        ):
            self.assertEqual(self.configure(ceiling), expected)
        for ceiling in (0, -1):
            with self.assertRaises(ValueError):
                self.configure(ceiling)
        # Doubling must not overflow at the signed integer limit.
        maximum = (1 << 63) - 1
        buckets = self.configure(maximum)
        self.assertEqual(buckets[-2:], (1 << 62, maximum))
        self.assertEqual(len(buckets), 64)

    def test_lookup_and_registered_winners_share_boundaries(self) -> None:
        for ceiling in (1, 6000, 8192, 16384, 20000, 40000, 65536):
            buckets = self.configure(ceiling)
            for bucket in buckets:
                self.assertEqual(self.native.insert_tactic(0, bucket, bucket), 1)
            for tokens in (
                0,
                1,
                8191,
                8192,
                16383,
                16384,
                32767,
                32768,
                39999,
                40000,
                40001,
                65536,
                65537,
            ):
                expected = (
                    max(bucket for bucket in buckets if bucket <= tokens)
                    if 1 <= tokens <= buckets[-1]
                    else -1
                )
                self.assertEqual(self.native.map_tokens(tokens), expected)
                self.assertEqual(
                    self.native.lookup_tactic(0, tokens), expected if expected != -1 else -2
                )

    def test_cache_configuration_and_operation_isolation(self) -> None:
        self.configure(40000)
        self.assertEqual(self.native.insert_tactic(0, 40000, 10), 1)
        self.assertEqual(self.native.insert_tactic(1, 40000, 8), 1)
        self.assertEqual(self.native.lookup_tactic(0, 40000), 10)
        self.assertEqual(self.native.lookup_tactic(1, 40000), 8)
        self.assertEqual(self.native.lookup_tactic(0, 32768), -2)
        self.assertEqual(self.native.insert_tactic(0, 30000, 9), 0)
        self.assertEqual(self.native.insert_tactic(0, 0, 9), 0)
        self.configure(40000)
        self.assertEqual(self.native.lookup_tactic(0, 40000), 10)
        with self.assertRaises(ValueError):
            self.configure(0)
        self.assertEqual(self.native.lookup_tactic(0, 40000), 10)
        self.configure(65536)
        self.assertEqual(self.native.lookup_tactic(0, 40000), -2)
        self.assertEqual(self.native.insert_tactic(0, 65536, 9), 1)
        self.native.clear_tactics()
        self.assertEqual(self.native.lookup_tactic(0, 65536), -2)
        self.assertEqual(self.native.map_tokens(65536), 65536)

    def autotuner_namespace(self) -> dict:
        tree = ast.parse((_ROOT / "tensorrt_llm/_torch/autotuner.py").read_text())
        namespace = dict(
            dataclass=dataclass, field=field, lru_cache=lru_cache, copy=copy, itertools=itertools
        )
        for node in tree.body:
            if getattr(node, "name", "") in (
                "DynamicTensorSpec",
                "ConstraintSpec",
                "StaticDim",
                "DynamicDim",
                "OptimizationProfile",
                "_spec_in_bounds",
            ):
                exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
        tuner = next(node for node in tree.body if getattr(node, "name", "") == "AutoTuner")
        for node in tuner.body:
            if getattr(node, "name", "") in ("_find_nearest_profile", "_optimization_profiles"):
                exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
        return namespace

    def test_python_lookup_uses_native_mapper_and_reuses_lru(self) -> None:
        buckets = self.configure(40000)
        namespace = self.autotuner_namespace()
        lookup = namespace["_find_nearest_profile"].__func__
        spec_type = namespace["DynamicTensorSpec"]

        def mapper(tokens: int) -> int:
            return self.native.map_tokens(tokens)

        for _ in range(100):
            spec = spec_type(0, 0, buckets, mapper)
            self.assertEqual(lookup(None, ((32769, 7168),), (spec,), (), 40000), ((32768, 7168),))
        self.assertEqual(lookup.cache_info().currsize, 1)
        self.assertEqual(lookup.cache_info().hits, 99)
        self.assertEqual(lookup(None, ((40000, 7168),), (spec,), (), 40000), ((40000, 7168),))
        self.assertEqual(lookup(None, ((40001, 7168),), (spec,), (), 40000), ((-1, 7168),))

    def test_existing_profiler_bounds_large_warmup_without_slicing(self) -> None:
        buckets = self.configure(40000)
        namespace = self.autotuner_namespace()

        class Tensor:
            def size(self):
                return (50000, 7168)

        namespace["torch"] = SimpleNamespace(Tensor=Tensor)
        config = SimpleNamespace(
            dynamic_tensor_specs=(
                namespace["DynamicTensorSpec"](0, 0, buckets, self.native.map_tokens),
            ),
            constraint_specs=(),
            tune_max_num_tokens=40000,
        )
        tuner = SimpleNamespace(skip_dynamic_tuning_buckets=False, _debug_logger=lambda _: None)
        profiles = namespace["_optimization_profiles"](tuner, config, [Tensor()])
        self.assertEqual(tuple(profile.get_opt_shapes()[0][0] for profile in profiles), buckets)

    def test_runner_configures_once_and_registers_only_winners(self) -> None:
        namespace = self.autotuner_namespace()
        tree = ast.parse((_ROOT / "tensorrt_llm/_torch/custom_ops/torch_custom_ops.py").read_text())
        runner = next(node for node in tree.body if getattr(node, "name", "") == "AllReduceRunner")
        registered, configured, default_reads = [], [], []
        mapper = SimpleNamespace(default=self.native.map_tokens)

        def configure(ceiling):
            configured.append(ceiling)
            return self.configure(ceiling)

        def get_buckets():
            default_reads.append(True)
            return self.buckets()

        ops = SimpleNamespace(
            configure_allreduce_tuning_buckets=configure,
            get_allreduce_tuning_buckets=get_buckets,
            map_to_allreduce_tuning_bucket=mapper,
            register_allreduce_tactic=lambda *args: registered.append(args[-2:]),
        )
        namespace.update(
            replace=replace, torch=SimpleNamespace(ops=SimpleNamespace(trtllm=ops), Size=tuple)
        )
        cache = SimpleNamespace(
            get_cache_key=lambda *args, **kwargs: args[2][0][0],
            cache={bucket: (0, 10, 1.0) for bucket in self.configure(40000)},
        )
        namespace["AutoTuner"] = SimpleNamespace(
            get=lambda: SimpleNamespace(
                profiling_cache=cache, _get_input_sizes=lambda _: [(40000, 7168)] + [(0,)] * 5
            )
        )
        methods = {}
        for node in runner.body:
            if getattr(node, "name", "") in (
                "configure_tuning_buckets",
                "_set_tuning_buckets",
                "__init__",
                "unique_id",
                "_register_native_tactics",
            ):
                exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
                methods[node.name] = namespace[node.name]
        methods["tuning_config"] = _Config(())
        runner_type = type("Runner", (), methods)
        runner_type.configure_tuning_buckets(40000)
        instance = runner_type(8, list(range(8)), None, 0, 1e-6, False)
        identity = instance.unique_id()
        instance._register_native_tactics([None] * 6)
        instance._register_native_tactics([None] * 6)
        self.assertEqual(configured, [40000])
        self.assertEqual(default_reads, [])
        self.assertEqual(registered, [(bucket, 10) for bucket in self.buckets()] * 2)
        self.assertIs(
            runner_type.tuning_config.dynamic_tensor_specs[0].map_to_tuning_buckets, mapper.default
        )
        self.assertEqual(runner_type.tuning_config.tune_max_num_tokens, 40000)
        runner_type.configure_tuning_buckets(16384)
        self.assertNotEqual(identity, instance.unique_id())

        # Without engine configuration, initialize once from the native defaults.
        standalone_type = type("StandaloneRunner", (), methods)
        standalone_type(8, list(range(8)), None, 0, 1e-6, False)
        standalone_type(8, list(range(8)), None, 0, 1e-6, False)
        self.assertEqual(default_reads, [True])
        self.assertEqual(standalone_type.tuning_config.tune_max_num_tokens, 16384)

        # Executing the real class-level tuning configuration must not touch
        # the native bucket getter or mapper during module import.
        config_node = next(
            node
            for node in runner.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "tuning_config"
                for target in node.targets
            )
        )
        namespace.update(
            TuningConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            DistributedTuningStrategy=SimpleNamespace(MERGE="merge"),
            torch=SimpleNamespace(),
        )
        exec(ast.unparse(config_node), namespace)
        self.assertEqual(default_reads, [True])

    def test_dispatch_preserves_original_inputs_and_signature(self) -> None:
        tree = ast.parse((_ROOT / "tensorrt_llm/_torch/custom_ops/torch_custom_ops.py").read_text())
        function = next(
            node for node in tree.body if getattr(node, "name", "") == "tunable_allreduce"
        )
        self.assertEqual(len(function.args.args), 11)
        function.decorator_list = []

        class Tensor:
            dtype = None
            is_cuda = False

        input_tensor, residual = Tensor(), Tensor()
        inputs = [input_tensor, residual, None, None, None, None]
        profiled, dispatched = [], []

        class Runner:
            tuning_config = _Config(())

            def __init__(self, *args):
                pass

            def _register_native_tactics(self, actual_inputs):
                pass

            def __call__(self, actual_inputs, tactic):
                dispatched.append((actual_inputs, tactic))
                return [actual_inputs[0]]

        def choose_one(op, runners, config, actual_inputs):
            profiled.append(actual_inputs)
            return runners[0], -1

        namespace = dict(
            AutoTuner=SimpleNamespace(get=lambda: SimpleNamespace(choose_one=choose_one)),
            AllReduceRunner=Runner,
            torch=SimpleNamespace(Tensor=Tensor),
        )
        exec("from __future__ import annotations\n" + ast.unparse(function), namespace)
        result = namespace[function.name](*inputs, [0, 1], 0, 0, 1e-6, False)
        self.assertIs(result[0], input_tensor)
        self.assertEqual(profiled, [inputs])
        self.assertEqual(dispatched, [(inputs, -1)])

    def test_skip_sweep_profiles_align_with_native_lookup(self) -> None:
        namespace = self.autotuner_namespace()
        lookup = namespace["_find_nearest_profile"].__func__
        buckets = self.configure(40000)

        class Tensor:
            def __init__(self, tokens):
                self.tokens = tokens

            def size(self):
                return (self.tokens, 7168)

        namespace["torch"] = SimpleNamespace(Tensor=Tensor)
        spec = namespace["DynamicTensorSpec"](0, 0, buckets, lambda n: self.native.map_tokens(n))
        config = SimpleNamespace(
            dynamic_tensor_specs=(spec,), constraint_specs=(), tune_max_num_tokens=40000
        )
        tuner = SimpleNamespace(skip_dynamic_tuning_buckets=True, _debug_logger=lambda _: None)
        cache = {}
        for tokens, expected, mapped in (
            (0, (), -1),
            (33000, (32768, 33000), 32768),
            (40000, (40000,), 40000),
            (40001, (), -1),
        ):
            profiles = namespace["_optimization_profiles"](tuner, config, [Tensor(tokens)])
            self.assertEqual(tuple(p.get_opt_shapes()[0][0] for p in profiles), expected)
            for profile in profiles:
                shape = profile.get_opt_shapes()
                key = lookup(None, tuple(tuple(s) for s in shape), (spec,), (), 40000, False)
                # Give each measured shape a distinct tactic so wrong-bucket
                # hits cannot accidentally pass.
                tactic = shape[0][0]
                cache[key] = tactic
                if tactic in buckets:
                    self.assertEqual(self.native.insert_tactic(0, tactic, tactic), 1)
            query = lookup(None, ((tokens, 7168),), (spec,), (), 40000)
            self.assertEqual(query, ((mapped, 7168),))
            self.assertEqual(cache.get(query, -2), self.native.lookup_tactic(0, tokens))

        # An explicit-shape warmup outside the default range cannot use the
        # ceiling's winner, so it should do no profiling and remain a miss.
        buckets = self.configure(16384)
        config.dynamic_tensor_specs = (
            namespace["DynamicTensorSpec"](0, 0, buckets, lambda n: self.native.map_tokens(n)),
        )
        config.tune_max_num_tokens = 16384
        profiles = namespace["_optimization_profiles"](tuner, config, [Tensor(20000)])
        self.assertEqual(profiles, [])
        self.assertEqual(self.native.insert_tactic(0, 16384, 8), 1)
        self.assertEqual(self.native.lookup_tactic(0, 20000), -2)

    def test_engine_passes_resolved_budget(self) -> None:
        tree = ast.parse((_ROOT / "tensorrt_llm/_torch/pyexecutor/model_engine.py").read_text())
        method = next(
            node
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "PyTorchModelEngine"
            for node in cls.body
            if getattr(node, "name", "") == "_configure_allreduce_tuning"
        )
        method.body = [node for node in method.body if not isinstance(node, ast.ImportFrom)]
        configured = []
        namespace = dict(
            AllReduceRunner=SimpleNamespace(configure_tuning_buckets=configured.append)
        )
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
        engine = SimpleNamespace(
            llm_args=SimpleNamespace(enable_autotuner=True, max_num_tokens=8192),
            max_num_tokens=40000,
            encoder_max_num_tokens=65536,
            _is_encoder_decoder_model=lambda: False,
        )
        namespace[method.name](engine)
        self.assertEqual(configured, [40000])
        engine._is_encoder_decoder_model = lambda: True
        namespace[method.name](engine)
        self.assertEqual(configured, [40000, 65536])
        engine.encoder_max_num_tokens = 4096
        namespace[method.name](engine)
        self.assertEqual(configured, [40000, 65536, 40000])
        engine.max_num_tokens = 6000
        namespace[method.name](engine)
        self.assertEqual(configured[-1], 6000)
        engine.llm_args.enable_autotuner = False
        namespace[method.name](engine)
        self.assertEqual(configured, [40000, 65536, 40000, 6000])

    def test_configuration_precedes_runner_warmup(self) -> None:
        tree = ast.parse((_ROOT / "tensorrt_llm/_torch/pyexecutor/model_engine.py").read_text())
        method = next(
            node
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "PyTorchModelEngine"
            for node in cls.body
            if getattr(node, "name", "") == "_warmup_impl"
        )
        method.decorator_list = []
        events = []

        class PackedRunner:
            def warmup(self):
                events.append("packed")

            def capture_graphs(self):
                events.append("packed_capture")

        class ScheduledRunner:
            def warmup(self, resource_manager):
                events.append("scheduled")

            def capture_graphs(self, resource_manager):
                events.append("scheduled_capture")

        namespace = dict(PackedModelRunner=PackedRunner, ScheduledModelRunner=ScheduledRunner)
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
        resources = SimpleNamespace(get_resource_manager=lambda key: None)
        for runner in (PackedRunner(), ScheduledRunner()):
            engine = SimpleNamespace(
                _runner=runner,
                _is_packed_runner=isinstance(runner, PackedRunner),
                _is_encoder_decoder_model=lambda: False,
                kv_cache_manager_key="kv_cache",
                _configure_allreduce_tuning=lambda: events.append("configure"),
            )
            namespace[method.name](engine, resources)
        self.assertEqual(
            events,
            ["configure", "packed", "configure", "scheduled"],
        )


if __name__ == "__main__":
    unittest.main()
