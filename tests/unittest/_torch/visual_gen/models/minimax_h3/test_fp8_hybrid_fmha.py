# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for MiniMax-H3 SM107 pure trtllm-gen-fp16sm routing, scratch bounds, audio graph, and FP8 YAMLs."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[6]
U4 = REPO / "tensorrt_llm/_torch/visual_gen/models/minimax_h3/minimax_h3_fp8__sm_103_107__u4"
CONFIGS = REPO / "examples/visual_gen/configs"


def _load_u4_gen_classes(capability: tuple[int, int]):
    tree = ast.parse((U4 / "pipeline.py").read_text())
    keep = {
        "Fp8UlyssesAttention",
        "_initialize_sm107_gen_fmha",
        "MiniMaxH3Fp8U4Pipeline",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in keep
    ]

    class _FakeBaseAttention:
        def __init__(self, *a, **k):
            self.norm_q = SimpleNamespace(weight="WQ", variance_epsilon=1e-6)
            self.norm_k = SimpleNamespace(weight="WK")
            self.to_out = [SimpleNamespace(input_scale=None)]
            self.use_async_ulysses = False

    class _FakeBasePipeline:
        def forward(self, *args, **kwargs):
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

    def fake_dyn_pack(*args, **kwargs):
        calls.append("dyn_pack")
        return "SEND_DYN", "DESCALES_DYN"

    def fake_static_pack(*args, **kwargs):
        calls.append("static_pack")
        return "SEND_STATIC", "DESCALES_STATIC"

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
        "pack_qkstatic_u4": fake_static_pack,
        "qk_norm_rope_pack_qkv": fake_dyn_pack,
        "initialize_block_ops": lambda t: None,
        "configure_u4_video_vae_pre_load": lambda *a, **k: False,
        "configure_u4_video_vae_post_load": lambda *a, **k: False,
        "install_audio_vae_cuda_graph": lambda *a, **k: False,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(U4 / "pipeline.py"), "exec"), ns)
    ns["_calls"] = calls
    return ns


class TestFp8PureGenFmhaContracts(unittest.TestCase):
    def test_sm103_stays_on_cudnn_and_sm107_uses_pure_trtllm_gen_with_direct_fp8(self):
        # SM103 must never attach trtllm-gen-fp16sm
        ns103 = _load_u4_gen_classes((10, 3))
        t103 = SimpleNamespace(
            proj_out=SimpleNamespace(weight=SimpleNamespace(device=SimpleNamespace(type="cuda"))),
            transformer_blocks=[],
        )
        self.assertFalse(ns103["_initialize_sm107_gen_fmha"](t103))

        # SM107 routes 100% of calls to pack_qkstatic_u4 + trtllm-gen-fp16sm (+ Direct FP8 out on static-FP8 blocks)
        ns107 = _load_u4_gen_classes((10, 7))
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
        tr = SimpleNamespace(
            proj_out=SimpleNamespace(weight=SimpleNamespace(device=SimpleNamespace(type="cuda"))),
            transformer_blocks=[SimpleNamespace(attn=attn)],
        )
        self.assertTrue(ns107["_initialize_sm107_gen_fmha"](tr))
        qkv = SimpleNamespace(
            ndim=3, shape=(1, 1, 3 * 56 * 128), dtype="BF16", is_cuda=True, is_contiguous=lambda: True
        )
        for _ in range(5):
            self.assertEqual(attn._fused_packed_forward(qkv, ("C", "S")), "DIRECT_FP8")
        self.assertEqual(attn._hybrid_cudnn_fmha_calls, 0)
        self.assertEqual(attn._hybrid_gen_fmha_calls, 5)
        self.assertEqual(direct_calls, [0.0] * 5)
        self.assertEqual(ns107["_calls"], ["static_pack"] * 5)

    def test_scratch_caches_are_bounded_o1_per_device_and_stream(self):
        tree = ast.parse((U4 / "qkv_exchange.py").read_text())
        keep = {
            "_LOG2E",
            "_QK_STATIC_DESCALE_128",
            "_U4_SCRATCH",
            "_QKSTATIC_SCRATCH",
            "_stream_key",
            "_get_u4_scratch",
            "_get_qkstatic_scratch",
        }
        body = [
            node
            for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in keep)
            or (
                isinstance(node, (ast.Assign, ast.AnnAssign))
                and (
                    (
                        isinstance(node, ast.AnnAssign)
                        and isinstance(node.target, ast.Name)
                        and node.target.id in keep
                    )
                    or (
                        isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id in keep for t in node.targets)
                    )
                )
            )
        ]
        stream_holder = {"id": 101}
        ns = {
            "math": math,
            "torch": SimpleNamespace(
                device=object,
                Tensor=object,
                bfloat16="BF16",
                float8_e4m3fn="FP8",
                int32="I32",
                float32="F32",
                uint8="U8",
                empty=lambda shape, dtype=None, device=None: (shape, dtype),
                tensor=lambda vals, dtype=None, device=None: (tuple(vals), dtype),
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
        s1 = ns["_get_u4_scratch"](100, 1, dev)
        s2 = ns["_get_u4_scratch"](200, 1, dev)
        self.assertEqual(len(ns["_U4_SCRATCH"]), 1)
        self.assertIsNot(s1, s2)
        q1 = ns["_get_qkstatic_scratch"](100, dev)
        q2 = ns["_get_qkstatic_scratch"](200, dev)
        self.assertEqual(len(ns["_QKSTATIC_SCRATCH"]), 1)
        self.assertIsNot(q1, q2)
        stream_holder["id"] = 202
        q3 = ns["_get_qkstatic_scratch"](200, dev)
        self.assertEqual(len(ns["_QKSTATIC_SCRATCH"]), 2)
        self.assertIsNot(q2, q3)

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
        wrapper(_FakeTensor((1, 8)), return_dict=False)
        self.assertEqual(len(wrapper._cache), 0)
        state["stream"] = 99
        state["capturing"] = True
        wrapper(_FakeTensor((1, 8)), return_dict=False)
        self.assertEqual(len(wrapper._cache), 0)
        state["capturing"] = False
        wrapper(_FakeTensor((1, 8)), return_dict=False)
        wrapper(_FakeTensor((1, 16)), return_dict=False)
        wrapper(_FakeTensor((1, 24)), return_dict=False)
        self.assertEqual(len(wrapper._cache), 2)
        self.assertEqual(state["replays"], 2)

    def test_fp8_example_yamls_present_and_aligned(self):
        for fname, ulysses in (
            ("minimax-h3-fp8-sm107-1gpu.yaml", "ulysses_size: 1"),
            ("minimax-h3-fp8-4gpu.yaml", "ulysses_size: 4"),
        ):
            txt = (CONFIGS / fname).read_text()
            self.assertIn("backend: CUDNN", txt)
            self.assertIn("qk_dtype: fp8", txt)
            self.assertIn("v_dtype: fp8", txt)
            self.assertIn(ulysses, txt)
            self.assertIn("visual_gen_scenario: require", txt)

    def test_inference_tensor_scale_version_runtime_error_is_handled(self):
        ns = _load_u4_gen_classes((10, 7))
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
