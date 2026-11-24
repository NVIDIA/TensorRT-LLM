# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax-H3 selective cuDNN -> trtllm-gen-fp16sm hybrid FMHA routing."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

U4 = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_fp8__sm_103_107__u4"
)


class _Scalar:
    def __init__(self, v: float):
        self.v = float(v)

    def __mul__(self, other):
        return _Scalar(self.v * (other.v if isinstance(other, _Scalar) else other))

    def __rmul__(self, other):
        return _Scalar(other * self.v)

    def __truediv__(self, other):
        return _Scalar(self.v / (other.v if isinstance(other, _Scalar) else other))

    def __rtruediv__(self, other):
        return _Scalar(other / self.v)

    def clamp(self, min=None, max=None):
        x = self.v
        if min is not None:
            mv = min.v if isinstance(min, _Scalar) else min
            x = x if x >= mv else mv
        if max is not None:
            xv = max.v if isinstance(max, _Scalar) else max
            x = x if x <= xv else xv
        return _Scalar(x)

    def bfloat16(self):
        return self

    def float(self):
        return self


def _load_complementary_descales():
    tree = ast.parse((U4 / "qkv_exchange.py").read_text())
    keep = {"_LOG2E", "_FP16SM_K_SHIFT_SCALE", "complementary_fp16sm_descales"}
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in keep)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in keep for t in node.targets)
        )
    ]
    ns = {
        "math": math,
        "torch": SimpleNamespace(Tensor=object, stack=lambda xs: [x.v for x in xs]),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "qkv_exchange.py"), "exec"), ns)
    return ns


def _load_u4_hybrid_classes(capability: tuple[int, int]):
    tree = ast.parse((U4 / "pipeline.py").read_text())
    keep = {
        "_HYBRID_CUDNN_INITIAL_STEPS",
        "_HYBRID_DIRECT_FP8_OUT_START_STEP",
        "_HYBRID_SKIP_CORR_T1_START_STEP",
        "Fp8UlyssesAttention",
        "_initialize_sm107_hybrid_fmha",
        "MiniMaxH3Fp8U4Pipeline",
    }
    body = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in keep)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in keep for t in node.targets)
        )
    ]

    class _FakeBaseAttention:
        def __init__(self, *a, **k):
            self.norm_q = SimpleNamespace(weight="WQ", variance_epsilon=1e-6)
            self.norm_k = SimpleNamespace(weight="WK")
            self.to_out = [SimpleNamespace(input_scale=None)]
            self.use_async_ulysses = False

    class _FakeBasePipeline:
        def forward(self, *args, fail=False, **kwargs):
            for _ in range(10):
                self._step_hook(None, None, None)
            if fail:
                raise ValueError("boom")
            return "OK"

    class _FakeCuDNN:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.quant_dtype = "fp8"
            self.dtype = "BF16"
            self.preferred_layout = "NHD"
            self.quant_attention_config = "FP8_CFG"

    class _FakeUlysses:
        def __init__(self, inner, group):
            self.inner_backend = inner
            self.process_group = group
            self.world_size = 4
            self.head_dim = 128
            self.sharded_num_heads = 14
            self.sharded_num_kv_heads = 14
            self._ulysses_rank = 0

        def forward_packed_fp8(self, send, fp8_descales=None, return_pre_a2a=False, **kwargs):
            return SimpleNamespace(
                ndim=4, shape=(1, 4, 14, 128), dtype="BF16", is_cuda=True, flatten=lambda _: send
            )

    calls = []

    def fake_qk_pack(*args, **kwargs):
        calls.append(("pack", bool(kwargs.get("gen_fp16sm", False))))
        return "SEND", "DESCALES"

    ns = {
        "__name__": "tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_103_107__u4.pipeline",
        "__package__": "tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_103_107__u4",
        "torch": SimpleNamespace(
            Tensor=object,
            bfloat16="BF16",
            float8_e4m3fn="FP8",
            float32="FP32",
            uint8="U8",
            compiler=SimpleNamespace(disable=lambda fn: fn),
            cuda=SimpleNamespace(get_device_capability=lambda dev: capability),
            isfinite=lambda x: SimpleNamespace(all=lambda: True),
            ops=SimpleNamespace(
                trtllm=SimpleNamespace(fmha_dense_fp8_is_supported=lambda *a: True)
            ),
            distributed=SimpleNamespace(is_initialized=lambda: False),
        ),
        "AttentionTensorLayout": SimpleNamespace(NHD="NHD"),
        "_a3_fmha_dense_fp8": lambda *a, **k: "RAW_FP8",
        "CuDNNAttention": _FakeCuDNN,
        "UlyssesAttention": _FakeUlysses,
        "PipelineComponent": SimpleNamespace(TRANSFORMER="t"),
        "MiniMaxH3Pipeline": _FakeBasePipeline,
        "MiniMaxH3Attention": _FakeBaseAttention,
        "Fp8UlyssesTransformer": object,
        "static_fp8_output_scale": lambda linear: linear.input_scale,
        "exchange_fp8_output": lambda raw, s, g: "FP8_EXCH",
        "exchange_prequantized_fp8_output": lambda raw, g: "PREQ_FP8_EXCH",
        "exchange_bf16_output": lambda raw, g: "BF16_EXCH",
        "pack_qkv": lambda *a, **k: ("S", "D"),
        "qk_norm_rope_pack_qkv": fake_qk_pack,
        "initialize_block_ops": lambda t: None,
        "configure_u4_video_vae_pre_load": lambda *a, **k: False,
        "configure_u4_video_vae_post_load": lambda *a, **k: False,
        "install_audio_vae_cuda_graph": lambda *a, **k: False,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "pipeline.py"), "exec"), ns)
    ns["_calls"] = calls
    return ns


class TestFp8HybridFmhaContracts(unittest.TestCase):
    def test_complementary_fp16sm_descales_satisfies_unit_log2e_fold(self):
        ns = _load_complementary_descales()
        fn = ns["complementary_fp16sm_descales"]
        sm_scale = 128.0**-0.5
        for s_k in (0.015, 0.032, 0.085, 0.25):
            dq, dk, dv = fn([_Scalar(0.035), _Scalar(s_k), _Scalar(0.042)])
            self.assertAlmostEqual(dk, 32.0 * s_k, places=7)
            self.assertAlmostEqual(dv, 0.042, places=7)
            self.assertAlmostEqual(dq * dk * sm_scale * math.log2(math.e), 1.0, places=6)

    def test_complementary_fp16sm_descales_guards_tiny_k_against_q_zero_underflow(self):
        ns = _load_complementary_descales()
        fn = ns["complementary_fp16sm_descales"]
        sm_scale = 128.0**-0.5
        # Q max = 8.0 (s_q = 8/448), K max = 0.01 (s_k = 0.01/448)
        s_q = 8.0 / 448.0
        s_k = 0.01 / 448.0
        dq, dk, _ = fn([_Scalar(s_q), _Scalar(s_k), _Scalar(0.02)])
        self.assertAlmostEqual(dq * dk * sm_scale * math.log2(math.e), 1.0, places=6)
        self.assertGreaterEqual(8.0 / dq, 0.125 - 1e-6)
        # Normal low-magnitude attention layers must stay on the exact 32 * s_k power-of-two grid
        dq_n, dk_n, _ = fn([_Scalar(0.008), _Scalar(0.015), _Scalar(0.02)])
        self.assertAlmostEqual(dk_n, 32.0 * 0.015, places=7)

    def test_sm103_stays_on_cudnn_and_sm107_switches_at_shipped_steps_with_reset(self):
        # SM103 must never attach trtllm-gen-fp16sm
        ns103 = _load_u4_hybrid_classes((10, 3))
        t103 = SimpleNamespace(
            proj_out=SimpleNamespace(weight=SimpleNamespace(device=SimpleNamespace(type="cuda"))),
            transformer_blocks=[],
        )
        self.assertIsNone(ns103["_initialize_sm107_hybrid_fmha"](t103))

        # SM107 routes steps 0..10 to cuDNN, 11..19 to Gen t=0, 20..24 to Direct FP8 t=0, 25+ to t=1.0
        ns107 = _load_u4_hybrid_classes((10, 7))
        attn = ns107["Fp8UlyssesAttention"]()
        attn.attn = ns107["UlyssesAttention"](ns107["CuDNNAttention"](), "GRP")
        class _FakeScale:
            _version = 1

            def data_ptr(self):
                return 777

            def __gt__(self, other):
                return SimpleNamespace(all=lambda: True)

        attn.to_out = [SimpleNamespace(input_scale=_FakeScale())]
        direct_calls = []
        attn._direct_fp8_packed_exchange = (
            lambda send, descales, out_scale, rows, backend, skip_t, **kw: direct_calls.append(
                skip_t
            )
            or "DIRECT_FP8"
        )
        hooks = []
        tr = SimpleNamespace(
            proj_out=SimpleNamespace(weight=SimpleNamespace(device=SimpleNamespace(type="cuda"))),
            transformer_blocks=[SimpleNamespace(attn=attn)],
            register_forward_hook=lambda fn: hooks.append(fn),
        )
        state = ns107["_initialize_sm107_hybrid_fmha"](tr)
        self.assertIsNotNone(state)
        qkv = SimpleNamespace(
            ndim=3, shape=(1, 1, 3 * 56 * 128), dtype="BF16", is_cuda=True, is_contiguous=lambda: True
        )
        for step in range(30):
            state["step"] = step
            attn._fused_packed_forward(qkv, ("C", "S"))
        self.assertEqual(attn._hybrid_cudnn_fmha_calls, 11)
        self.assertEqual(attn._hybrid_gen_fmha_calls, 19)
        self.assertEqual(attn._hybrid_t1_calls, 5)
        self.assertEqual(direct_calls, [0.0] * 5 + [1.0] * 5)
        self.assertEqual(
            [flag for _, flag in ns107["_calls"]],
            [False] * 11 + [True] * 19,
        )

        # Pipeline forward resets step state both on success and after an exception
        pipe = ns107["MiniMaxH3Fp8U4Pipeline"]()
        pipe._hybrid_step_state = state
        pipe._step_hook = hooks[0]
        state["step"] = 5
        self.assertEqual(pipe.forward(), "OK")
        self.assertEqual(state["step"], 0)
        with self.assertRaises(ValueError):
            pipe.forward(fail=True)
        self.assertEqual(state["step"], 0)

    def test_scratch_cache_is_bounded_o1_per_device_and_stream(self):
        tree = ast.parse((U4 / "qkv_exchange.py").read_text())
        keep = {"_U4_SCRATCH", "_U1_SCRATCH", "_stream_key", "_get_u4_scratch", "_get_u1_scratch"}
        body = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in keep)
            or (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and (
                    (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in keep)
                    or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in keep for t in node.targets))
                )
            )
        ]
        stream_holder = {"id": 101}
        ns = {
            "torch": SimpleNamespace(
                device=object,
                Tensor=object,
                bfloat16="BF16",
                int32="I32",
                float32="F32",
                uint8="U8",
                empty=lambda shape, dtype=None, device=None: (shape, dtype),
                cuda=SimpleNamespace(
                    current_stream=lambda dev: SimpleNamespace(cuda_stream=stream_holder["id"])
                ),
            ),
            "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
        }
        exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "qkv_exchange.py"), "exec"), ns)
        class _FakeDev:
            type = "cuda"

        dev = _FakeDev()
        # Different shapes on the same stream must evict the prior entry (O(1) memory)
        s1 = ns["_get_u4_scratch"](100, 1, dev)
        s2 = ns["_get_u4_scratch"](200, 1, dev)
        self.assertEqual(len(ns["_U4_SCRATCH"]), 1)
        self.assertIsNot(s1, s2)
        self.assertIs(ns["_get_u4_scratch"](200, 1, dev), s2)
        # Different CUDA stream gets its own isolated entry (no cross-stream aliasing)
        stream_holder["id"] = 202
        s3 = ns["_get_u4_scratch"](200, 1, dev)
        self.assertEqual(len(ns["_U4_SCRATCH"]), 2)
        self.assertIsNot(s2, s3)
        # U1 scratch cache obeys the exact same O(1) eviction and stream isolation
        u1_a = ns["_get_u1_scratch"](100, 4, dev)
        u1_b = ns["_get_u1_scratch"](200, 4, dev)
        self.assertEqual(len(ns["_U1_SCRATCH"]), 1)
        self.assertIsNot(u1_a, u1_b)
        stream_holder["id"] = 303
        u1_c = ns["_get_u1_scratch"](200, 4, dev)
        self.assertEqual(len(ns["_U1_SCRATCH"]), 2)
        self.assertIsNot(u1_b, u1_c)

    def test_audio_decode_graph_guards_default_stream_nested_capture_and_max_shapes(self):
        tree = ast.parse((U4 / "vae_opt.py").read_text())
        body = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AudioDecodeGraph"
        ]
        state = {"stream": 0, "capturing": False, "eager_calls": 0, "replays": 0}

        class _FakeGraph:
            def replay(self):
                state["replays"] += 1

        class _FakeCtx:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        class _FakeTensor:
            def __init__(self, shape):
                self.shape = shape
                self.dtype = "FP32"
                self.device = "cuda:0"
                self.is_cuda = True

            def clone(self):
                return _FakeTensor(self.shape)

            def copy_(self, other):
                return self

        ns = {
            "torch": SimpleNamespace(
                Tensor=object,
                is_grad_enabled=lambda: False,
                cuda=SimpleNamespace(
                    CUDAGraph=_FakeGraph,
                    graph=lambda g, stream=None: _FakeCtx(),
                    is_current_stream_capturing=lambda: state["capturing"],
                    current_stream=lambda dev: SimpleNamespace(cuda_stream=state["stream"]),
                ),
            )
        }
        exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "vae_opt.py"), "exec"), ns)

        def eager(x, *, return_dict=True):
            state["eager_calls"] += 1
            return (x,)

        wrapper = ns["AudioDecodeGraph"](eager, max_shapes=2)
        # Legacy default stream (0) must run eagerly without capturing
        wrapper(_FakeTensor((1, 8)), return_dict=False)
        self.assertEqual(len(wrapper._cache), 0)
        self.assertEqual(state["eager_calls"], 1)
        # Nested capture must run eagerly
        state["stream"] = 99
        state["capturing"] = True
        wrapper(_FakeTensor((1, 8)), return_dict=False)
        self.assertEqual(len(wrapper._cache), 0)
        # Side stream captures up to max_shapes=2, then falls back eagerly for a 3rd shape
        state["capturing"] = False
        wrapper(_FakeTensor((1, 8)), return_dict=False)
        wrapper(_FakeTensor((1, 16)), return_dict=False)
        wrapper(_FakeTensor((1, 24)), return_dict=False)
        self.assertEqual(len(wrapper._cache), 2)
        self.assertEqual(state["replays"], 2)

    def test_inference_tensor_scale_version_runtime_error_is_handled(self):
        ns = _load_u4_hybrid_classes((10, 7))
        attn = ns["Fp8UlyssesAttention"]()
        attn.attn = ns["UlyssesAttention"](ns["CuDNNAttention"](), "GRP")

        class _InferenceScale:
            @property
            def _version(self):
                raise RuntimeError("Inference tensors do not track version counter.")

            def data_ptr(self):
                return 12345

            def __gt__(self, other):
                return SimpleNamespace(all=lambda: True)

        scale = _InferenceScale()
        attn.to_out = [SimpleNamespace(input_scale=scale)]
        self.assertIs(attn._validated_fp8_output_scale(), scale)


if __name__ == "__main__":
    unittest.main()
