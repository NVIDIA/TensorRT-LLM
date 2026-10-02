# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import functools
import json
import os
import tempfile
import unittest
from copy import deepcopy
from dataclasses import dataclass
from unittest import mock

import torch
import torch.nn.functional as F
from _torch.helpers import create_mock_cuda_graph_runner
from parameterized import parameterized
from transformers import ApertusConfig
from transformers import ApertusForCausalLM as HFApertusForCausalLM
from transformers.activations import XIELUActivation
from utils.llm_data import llm_models_root
from utils.util import default_dtype

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_apertus import (
    Apertus1p5ForConditionalGeneration,
    ApertusForCausalLM,
)
from tensorrt_llm._torch.modules.xielu import XIELU, xielu_reference
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# A 2-layer Apertus. RoPE uses llama3 scaling with a short original context
# so that the scaling changes most rotary frequencies and a model that ignored
# it would fail the HF comparison at the short positions used here.
APERTUS_TINY_CONFIG = {
    "architectures": ["ApertusForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "xielu",
    "hidden_size": 512,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "max_position_embeddings": 4096,
    "mlp_bias": False,
    "model_type": "apertus",
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "num_key_value_heads": 2,
    "pad_token_id": 3,
    "qk_norm": True,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {
        "rope_type": "llama3",
        "rope_theta": 10000.0,
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 32,
    },
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 1024,
}

APERTUS_8B_INSTRUCT = "Apertus-8B-Instruct-2509"
APERTUS_1P5_8B = "Apertus-v1.5-8B"


@dataclass(repr=False)
class Scenario:
    backend: str
    use_cuda_graph: bool = False

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}-use_cuda_graph:{self.use_cuda_graph}"


def _randomize_xielu(hf_model: HFApertusForCausalLM, generator: torch.Generator) -> None:
    """Give every layer distinct xIELU coefficients far from the init defaults."""
    for layer in hf_model.model.layers:
        act = layer.mlp.act_fn
        for param in (act.alpha_p, act.alpha_n):
            values = torch.empty(param.shape, dtype=torch.float32, device="cpu").uniform_(
                -2.0, 2.0, generator=generator
            )
            param.data.copy_(values.to(param.device, param.dtype))


def _xielu_float64(x: torch.Tensor, a_p: float, a_n: float, beta: float, eps: float):
    xd = x.double()
    pos = a_p * xd * xd + beta * xd
    neg = (torch.expm1(torch.clamp_max(xd, eps)) - xd) * a_n + beta * xd
    return torch.where(xd > 0, pos, neg)


def _xielu_test_input(dtype: torch.dtype, device: str) -> torch.Tensor:
    """Covers x > 0, x in (eps, 0], x <= eps, very negative x, and exact zeros."""
    parts = [
        torch.linspace(-30.0, 30.0, 4097),
        torch.linspace(-1e-5, 1e-5, 257),
        torch.tensor([0.0, -0.0, -1e-6, -2e-6, -5e-7, 1e-7, -1e4, 1e4]),
        torch.randn(8192) * 4,
    ]
    return torch.cat(parts).to(device=device, dtype=dtype)


class TestXIELU(unittest.TestCase):
    def _make_pair(self, dtype, device, alpha_p, alpha_n):
        hf_act = XIELUActivation(dtype=torch.float32).to(device)
        hf_act.alpha_p.data.fill_(alpha_p)
        hf_act.alpha_n.data.fill_(alpha_n)
        act = XIELU().to(device)
        act.load_weights([dict(hf_act.state_dict())])
        return hf_act, act

    def test_load_weights_computes_coefficients(self):
        hf_act, act = self._make_pair(torch.bfloat16, "cpu", 0.3, -1.7)
        self.assertAlmostEqual(act.a_p, F.softplus(torch.tensor(0.3)).item(), places=6)
        self.assertAlmostEqual(act.a_n, 0.5 + F.softplus(torch.tensor(-1.7)).item(), places=6)
        self.assertEqual(act.beta_value, hf_act.beta.item())
        self.assertEqual(act.eps_value, hf_act.eps.item())

    def test_load_weights_requires_alphas(self):
        act = XIELU()
        with self.assertRaises(AssertionError):
            act.load_weights([{"beta": torch.tensor(0.5), "eps": torch.tensor(-1e-6)}])

    @parameterized.expand(
        [
            (dtype, device)
            for dtype in (torch.bfloat16, torch.float16)
            for device in ("cpu", "cuda")
        ],
        lambda f, n, p: f"{f.__name__}[{p.args[0]}-{p.args[1]}]",
    )
    def test_matches_reference(self, dtype, device):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        hf_act, act = self._make_pair(dtype, device, 0.3, -1.7)
        x = _xielu_test_input(dtype, device)
        if dtype == torch.float16:
            x = x.clamp(-60000.0, 60000.0)

        out = act(x)
        self.assertEqual(out.dtype, dtype)

        # One rounding from fp32 to the output dtype: within 1 ulp of the
        # correctly rounded float64 result.
        expected = _xielu_float64(x, act.a_p, act.a_n, act.beta_value, act.eps_value)
        finite = torch.isfinite(expected.to(dtype))
        torch.testing.assert_close(
            out[finite].double(),
            expected[finite].to(dtype).double(),
            atol=0,
            rtol=torch.finfo(dtype).eps,
        )

        # HF's fp32 path is the same expression.
        torch.testing.assert_close(out, hf_act(x.float()).to(dtype), atol=0, rtol=0)

        # Exact zeros take the negative branch: a_n * expm1(eps) - 0 + 0.
        zero_out = act(torch.zeros(4, dtype=dtype, device=device)).float()
        expected_zero = torch.tensor(act.a_n * torch.expm1(torch.tensor(act.eps_value)).item())
        torch.testing.assert_close(
            zero_out.cpu(), expected_zero.to(dtype).float().expand(4), atol=0, rtol=0
        )

    def test_reference_function_matches_module(self):
        act = XIELU()
        x = _xielu_test_input(torch.bfloat16, "cpu")
        torch.testing.assert_close(
            act(x), xielu_reference(x, act.a_p, act.a_n, act.beta_value, act.eps_value)
        )


class TestApertus(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")

    def _build_models(self, backend: str):
        torch.random.manual_seed(0)
        generator = torch.Generator().manual_seed(1234)
        config = ApertusConfig.from_dict(deepcopy(APERTUS_TINY_CONFIG))
        dtype = config.torch_dtype
        device = torch.device("cuda")

        with torch.device(device), default_dtype(dtype):
            hf_model = HFApertusForCausalLM(config).eval()
            _randomize_xielu(hf_model, generator)

            model_config = ModelConfig(pretrained_config=config, attn_backend=backend)
            model = ApertusForCausalLM(model_config).to(dtype).to(device)
            model.load_weights(hf_model.state_dict())
            model.post_load_weights()
        return config, hf_model, model

    def test_rejects_other_activations(self):
        config = ApertusConfig.from_dict({**deepcopy(APERTUS_TINY_CONFIG), "hidden_act": "silu"})
        with torch.device("cuda"), self.assertRaisesRegex(ValueError, "hidden_act"):
            ApertusForCausalLM(ModelConfig(pretrained_config=config))

    def test_lora_params_reach_mlp(self):
        config = ApertusConfig.from_dict(deepcopy(APERTUS_TINY_CONFIG))
        with torch.device("cuda"), default_dtype(config.torch_dtype):
            model = ApertusForCausalLM(ModelConfig(pretrained_config=config))
        layer = model.model.layers[0]
        hidden = torch.randn(3, config.hidden_size, device="cuda", dtype=config.torch_dtype)
        lora_params = {"sentinel": True}
        with (
            mock.patch.object(layer.self_attn, "forward", return_value=hidden) as attn,
            mock.patch.object(layer.mlp, "forward", return_value=hidden) as mlp,
        ):
            layer(
                position_ids=None,
                hidden_states=hidden,
                attn_metadata=None,
                residual=None,
                lora_params=lora_params,
            )
        self.assertIs(attn.call_args.kwargs["lora_params"], lora_params)
        self.assertIs(mlp.call_args.kwargs["lora_params"], lora_params)

    def test_weights_loaded(self):
        _, hf_model, model = self._build_models("TRTLLM")
        for hf_layer, layer in zip(hf_model.model.layers, model.model.layers):
            act = layer.mlp.activation
            self.assertIsInstance(act, XIELU)
            hf_act = hf_layer.mlp.act_fn
            self.assertAlmostEqual(act.a_p, F.softplus(hf_act.alpha_p.float()).item(), places=5)
            self.assertAlmostEqual(
                act.a_n,
                hf_act.beta.float().item() + F.softplus(hf_act.alpha_n.float()).item(),
                places=5,
            )
            torch.testing.assert_close(
                layer.attention_layernorm.weight, hf_layer.attention_layernorm.weight
            )
            torch.testing.assert_close(
                layer.feedforward_layernorm.weight, hf_layer.feedforward_layernorm.weight
            )
            torch.testing.assert_close(
                layer.self_attn.q_norm.weight, hf_layer.self_attn.q_norm.weight
            )
            torch.testing.assert_close(
                layer.self_attn.k_norm.weight, hf_layer.self_attn.k_norm.weight
            )

    @parameterized.expand(
        [
            Scenario(backend="VANILLA"),
            Scenario(backend="FLASHINFER"),
            Scenario(backend="TRTLLM"),
            Scenario(backend="TRTLLM", use_cuda_graph=True),
        ],
        lambda testcase_func, param_num, param: f"{testcase_func.__name__}[{param.args[0]}]",
    )
    @torch.no_grad()
    def test_apertus_allclose_to_hf(self, scenario: Scenario) -> None:
        metadata_cls = get_attention_backend(scenario.backend).Metadata
        config, hf_model, model = self._build_models(scenario.backend)
        device = torch.device("cuda")

        num_gen_steps = 4
        input_ids = torch.tensor(
            [100, 200, 300, 100, 200, 100, 400, 500, 17, 901, 3, 64],
            dtype=torch.int,
            device=device,
        )
        gen_tokens = [600, 7, 1000, 42]
        prompt_len = input_ids.size(-1)

        head_dim = config.hidden_size // config.num_attention_heads
        tokens_per_block = 128
        kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=tokens_per_block),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=tokens_per_block,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        request_ids = [1]
        kv_cache_manager.add_dummy_requests(request_ids, [prompt_len + num_gen_steps])

        def make_metadata(num_tokens: int, num_cached: int, num_contexts: int):
            return metadata_cls(
                seq_lens=torch.tensor([num_tokens], dtype=torch.int),
                num_contexts=num_contexts,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=[num_cached]
                ),
                max_num_requests=1,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=request_ids,
                prompt_lens=[prompt_len],
            )

        # Prefill (CUDA graphs are only used for generation).
        attn_metadata = make_metadata(prompt_len, 0, 1)
        position_ids = torch.arange(0, prompt_len, device=device).unsqueeze(0)
        ref = hf_model.forward(
            input_ids=input_ids.unsqueeze(0).long(), position_ids=position_ids, use_cache=True
        )
        # Runs first so that the prefill below overwrites the KV cache it wrote.
        self._assert_xielu_is_observable(
            model, input_ids, position_ids, attn_metadata, ref.logits[:, -1].float()
        )
        attn_metadata.prepare()
        logits = model.forward(
            input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
        )
        self._assert_logits_close(logits, ref.logits[:, -1].float(), "prefill")
        past_key_values = ref.past_key_values

        graph_runner = create_mock_cuda_graph_runner(1) if scenario.use_cuda_graph else None
        graph_metadata = None
        try:
            for step, token in enumerate(gen_tokens):
                gen_input_ids = torch.tensor([token], dtype=torch.int, device=device)
                gen_position_ids = torch.tensor([[prompt_len + step]], device=device)
                num_cached = prompt_len + step

                if graph_runner is None:
                    attn_metadata = make_metadata(1, num_cached, 0)
                    attn_metadata.prepare()
                    logits = model.forward(
                        input_ids=gen_input_ids,
                        position_ids=gen_position_ids,
                        attn_metadata=attn_metadata,
                    )
                else:
                    key = (1, 0, False)
                    if graph_metadata is None:
                        graph_metadata = make_metadata(1, num_cached, 0).create_cuda_graph_metadata(
                            1
                        )
                    else:
                        graph_metadata.kv_cache_params = KVCacheParams(
                            use_cache=True, num_cached_tokens_per_seq=[num_cached]
                        )
                    inputs = {
                        "input_ids": gen_input_ids,
                        "position_ids": gen_position_ids,
                        "attn_metadata": graph_metadata,
                    }
                    graph_metadata.prepare()
                    if step == 0:
                        graph_runner.capture(key, lambda inputs: model.forward(**inputs), inputs)
                    # Replay twice to catch buffers reallocated in prepare().
                    for _ in range(2):
                        graph_metadata.prepare()
                        logits = graph_runner.replay(key, inputs)

                ref = hf_model.forward(
                    input_ids=gen_input_ids.unsqueeze(0).long(),
                    position_ids=gen_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = ref.past_key_values
                self._assert_logits_close(logits, ref.logits[:, -1].float(), f"decode {step}")
        finally:
            if graph_runner is not None:
                graph_runner.clear()
            kv_cache_manager.shutdown()

    # The tiny random model has small logits, so a tolerance as loose as the
    # Llama test's (0.4) would not detect a wrong activation. This is checked
    # directly by _assert_xielu_is_observable.
    _ATOL = 0.05
    _RTOL = 0.05

    def _assert_logits_close(self, actual, expected, what):
        torch.testing.assert_close(
            actual, expected, atol=self._ATOL, rtol=self._RTOL, msg=lambda m: f"{what}: {m}"
        )

    def _assert_xielu_is_observable(self, model, input_ids, position_ids, attn_metadata, ref):
        """The comparison above must fail if the checkpoint's xIELU coefficients are ignored."""
        saved = []
        for layer in model.model.layers:
            act = layer.mlp.activation
            saved.append((act.a_p, act.a_n))
            act.a_p, act.a_n = 0.8, 0.8
        try:
            attn_metadata.prepare()
            wrong = model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
        finally:
            for layer, (a_p, a_n) in zip(model.model.layers, saved):
                layer.mlp.activation.a_p, layer.mlp.activation.a_n = a_p, a_n
        with self.assertRaises(AssertionError):
            self._assert_logits_close(wrong, ref, "default xIELU coefficients")


# Shape of the swiss-ai/Apertus-v1.5-8B config.json, with the audio and
# vision tokenizer sub-configs reduced to what the loader must ignore.
def _apertus1p5_config_dict(text_config: dict) -> dict:
    return {
        "architectures": ["Apertus1p5ForConditionalGeneration"],
        "model_type": "apertus1p5",
        "image_token_id": 131079,
        "audio_token_id": 131085,
        "audio_tokenizer_config": {"model_type": "wavtokenizer", "dtype": "float32"},
        "vision_tokenizer_config": {
            "model_type": "apertus1p5_vision_tokenizer",
            "dtype": "float32",
        },
        "text_config": {**text_config, "model_type": "apertus1p5_text"},
        "tie_word_embeddings": False,
    }


def _to_apertus1p5_state_dict(hf_state_dict: dict, output_vocab_size: int) -> dict:
    """Lay an Apertus state dict out like an Apertus 1.5 checkpoint."""
    weights = {}
    for name, tensor in hf_state_dict.items():
        if name == "lm_head.weight":
            weights[name] = tensor[:output_vocab_size]
        else:
            weights[name.replace("model.", "model.language_model.", 1)] = tensor
    weights["model.vision_tokenizer.encoder.conv_in.weight"] = torch.zeros(4, 3, 3, 3)
    weights["model.audio_tokenizer.head.linear.bias"] = torch.zeros(4)
    return weights


class TestApertus1p5Config(unittest.TestCase):
    def test_text_config_is_flattened(self):
        text_config = {
            k: v
            for k, v in APERTUS_TINY_CONFIG.items()
            if k not in ("architectures", "model_type", "torch_dtype")
        }
        text_config.update(dtype="bfloat16", vocab_size=2048, output_vocab_size=1024)
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(_apertus1p5_config_dict(text_config), f)
            config = load_pretrained_config(model_dir)

        self.assertIsInstance(config, ApertusConfig)
        self.assertEqual(config.architectures, ["Apertus1p5ForConditionalGeneration"])
        # The runtime's vocab_size is the output vocabulary (it bounds request
        # token ids and sizes the logits); the embedding uses input_vocab_size.
        self.assertEqual(config.vocab_size, 1024)
        self.assertEqual(config.input_vocab_size, 2048)
        self.assertEqual(config.torch_dtype, torch.bfloat16)
        self.assertEqual(config.num_hidden_layers, APERTUS_TINY_CONFIG["num_hidden_layers"])
        self.assertEqual(config.rope_parameters["rope_type"], "llama3")
        self.assertIsNone(getattr(config, "quantization_config", None))

    def test_top_level_quantization_config_is_kept(self):
        """Quantized 1.5 exports put quantization_config next to text_config."""
        text_config = {
            k: v
            for k, v in APERTUS_TINY_CONFIG.items()
            if k not in ("architectures", "model_type", "torch_dtype")
        }
        text_config.update(dtype="bfloat16", vocab_size=2048, output_vocab_size=1024)
        quantization_config = {"quant_method": "compressed-tensors", "format": "float-quantized"}
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        **_apertus1p5_config_dict(text_config),
                        "quantization_config": quantization_config,
                    },
                    f,
                )
            config = load_pretrained_config(model_dir)
        self.assertEqual(config.quantization_config, quantization_config)


class TestApertus1p5(unittest.TestCase):
    @parameterized.expand(
        [
            ("plain", "model.language_model.layers.0.mlp.down_proj"),
            ("regex", r"re:model\.language_model\.layers\.0\.mlp\.down_proj"),
        ]
    )
    def test_quant_exclusions_use_checkpoint_names(self, _, excluded):
        """1.5 quantization metadata names modules as the checkpoint does."""
        config_dict = deepcopy(APERTUS_TINY_CONFIG)
        config_dict["vocab_size"] = self.INPUT_VOCAB_SIZE
        config = ApertusConfig.from_dict(
            {**config_dict, "output_vocab_size": self.OUTPUT_VOCAB_SIZE}
        )
        quant_config = QuantConfig(quant_algo=QuantAlgo.FP8, exclude_modules=[excluded, "lm_head"])
        with torch.device("cuda"), default_dtype(config.torch_dtype):
            model = Apertus1p5ForConditionalGeneration(
                ModelConfig(pretrained_config=config, quant_config=quant_config)
            )

        def quantized(linear):
            return (
                linear.quant_config is not None
                and linear.quant_config.layer_quant_mode.has_any_quant()
            )

        self.assertFalse(quantized(model.model.layers[0].mlp.down_proj))
        self.assertTrue(quantized(model.model.layers[0].mlp.up_proj))
        self.assertTrue(quantized(model.model.layers[1].mlp.down_proj))
        # The caller's QuantConfig is not modified.
        self.assertEqual(quant_config.exclude_modules, [excluded, "lm_head"])

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")

    INPUT_VOCAB_SIZE = 2048
    OUTPUT_VOCAB_SIZE = 1024

    @torch.no_grad()
    def test_text_logits_match_hf(self):
        torch.random.manual_seed(0)
        generator = torch.Generator().manual_seed(4321)
        config_dict = deepcopy(APERTUS_TINY_CONFIG)
        config_dict["vocab_size"] = self.INPUT_VOCAB_SIZE
        hf_config = ApertusConfig.from_dict(config_dict)
        text_config = {
            k: v for k, v in config_dict.items() if k not in ("architectures", "model_type")
        }
        text_config["output_vocab_size"] = self.OUTPUT_VOCAB_SIZE
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(_apertus1p5_config_dict(text_config), f)
            config = load_pretrained_config(model_dir)
        dtype = config.torch_dtype
        device = torch.device("cuda")

        with torch.device(device), default_dtype(dtype):
            hf_model = HFApertusForCausalLM(hf_config).eval()
            _randomize_xielu(hf_model, generator)
            model_config = ModelConfig(pretrained_config=config, attn_backend="TRTLLM")
            model = Apertus1p5ForConditionalGeneration(model_config).to(dtype).to(device)
            model.load_weights(
                _to_apertus1p5_state_dict(hf_model.state_dict(), self.OUTPUT_VOCAB_SIZE)
            )
            model.post_load_weights()

        self.assertEqual(model.model.embed_tokens.weight.shape[0], self.INPUT_VOCAB_SIZE)
        self.assertEqual(model.lm_head.weight.shape[0], self.OUTPUT_VOCAB_SIZE)

        # Includes ids beyond the output vocabulary, as image and audio codes are.
        input_ids = torch.tensor(
            [100, 200, 1500, 300, 2047, 100, 400, 1024], dtype=torch.int, device=device
        )
        prompt_len = input_ids.size(-1)
        head_dim = config.hidden_size // config.num_attention_heads
        kv_cache_manager = KVCacheManager(
            KvCacheConfig(max_tokens=128),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            tokens_per_block=128,
            max_seq_len=128,
            max_batch_size=1,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        try:
            kv_cache_manager.add_dummy_requests([1], [prompt_len])
            attn_metadata = get_attention_backend("TRTLLM").Metadata(
                seq_lens=torch.tensor([prompt_len], dtype=torch.int),
                num_contexts=1,
                kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
                max_num_requests=1,
                max_num_tokens=8192,
                kv_cache_manager=kv_cache_manager,
                request_ids=[1],
                prompt_lens=[prompt_len],
            )
            position_ids = torch.arange(0, prompt_len, device=device).unsqueeze(0)
            attn_metadata.prepare()
            logits = model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
            ref = hf_model.forward(
                input_ids=input_ids.unsqueeze(0).long(), position_ids=position_ids
            )
        finally:
            kv_cache_manager.shutdown()

        self.assertEqual(logits.shape[-1], self.OUTPUT_VOCAB_SIZE)
        torch.testing.assert_close(
            logits,
            ref.logits[:, -1, : self.OUTPUT_VOCAB_SIZE].float(),
            atol=TestApertus._ATOL,
            rtol=TestApertus._RTOL,
        )


class _RealCheckpointGreedyTest:
    """Greedy decoding of a real checkpoint through the LLM API, checked against HF.

    Two bf16 implementations do not produce identical greedy sequences: on
    near-tied tokens (measured on Apertus 1.5: 0.02 and 0.09 logits in an fp32
    reference, below bf16 resolution at these magnitudes) accumulation order
    alone decides. So instead of exact sequence equality, HF runs once over the
    prompt plus TRT-LLM's generated tokens and every TRT-LLM choice must be
    within MAX_MARGIN logits of HF's best token at that position. A model with
    wrong weights or activations picks tokens several logits below the best.
    """

    CHECKPOINT: str
    PROMPTS = [
        "The capital of Switzerland is",
        "Write a haiku about glaciers.",
        "Explain in one paragraph why the sky is blue.",
    ]
    MAX_TOKENS = 64
    # generation_config.json: </s>, <|assistant_end|>, <|tools_suffix|>
    STOP_TOKEN_IDS = [2, 68, 72]
    # Two bf16 ulps for logits of magnitude 32-64.
    MAX_MARGIN = 0.5

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        root = llm_models_root()
        path = os.path.join(root, self.CHECKPOINT) if root is not None else None
        if path is None or not os.path.isdir(path):
            self.skipTest(f"{self.CHECKPOINT} not found under LLM_MODELS_ROOT")
        self.model_path = path

    def _load_hf_model(self):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map="cuda"
        )

    @staticmethod
    def _use_fp32_xielu(hf_model):
        """Run HF's own xIELU formula in fp32 and round once, as TRT-LLM does.

        HF's path in bf16 rounds after every operation, which by itself flips
        near-tied greedy choices. HF's implementation is used (not the one
        under test) so the reference stays independent.
        """
        for layer in hf_model.model.layers:
            act = layer.mlp.act_fn.float()
            act.forward = functools.partial(
                lambda x, act: act._xielu_python(x.float()).to(x.dtype), act=act
            )

    def _trtllm_greedy(self, prompt_ids):
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import KvCacheConfig as LlmKvCacheConfig

        sampling_params = SamplingParams(
            max_tokens=self.MAX_TOKENS, temperature=0.0, stop_token_ids=self.STOP_TOKEN_IDS
        )
        with LLM(
            model=self.model_path,
            kv_cache_config=LlmKvCacheConfig(free_gpu_memory_fraction=0.5),
            max_batch_size=len(prompt_ids),
            max_seq_len=1024,
        ) as llm:
            outputs = llm.generate(
                [{"prompt_token_ids": ids} for ids in prompt_ids], sampling_params
            )
        return [list(o.outputs[0].token_ids) for o in outputs]

    def test_greedy_consistent_with_hf(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in self.PROMPTS
        ]
        self.assertIn("<|user_start|>", prompts[0])
        self.assertTrue(prompts[0].rstrip().endswith("<|assistant_start|>"), prompts[0])
        prompt_ids = [tokenizer(p, add_special_tokens=False).input_ids for p in prompts]

        trt_outputs = self._trtllm_greedy(prompt_ids)

        hf_model = self._load_hf_model().eval()
        self._use_fp32_xielu(hf_model)
        vocab = hf_model.lm_head.out_features
        try:
            for prompt, ids, gen in zip(self.PROMPTS, prompt_ids, trt_outputs):
                self.assertGreater(len(gen), 0, prompt)
                with torch.inference_mode():
                    seq = torch.tensor([ids + gen], device="cuda")
                    logits = hf_model(seq).logits[0, len(ids) - 1 : -1, :vocab].float()
                    hf_greedy = hf_model.generate(
                        torch.tensor([ids], device="cuda"),
                        max_new_tokens=self.MAX_TOKENS,
                        do_sample=False,
                        eos_token_id=self.STOP_TOKEN_IDS,
                    )[0, len(ids) :].tolist()
                chosen = torch.tensor(gen, device="cuda")
                margins = logits.max(-1).values - logits.gather(1, chosen[:, None]).squeeze(1)
                prefix = next(
                    (i for i, (a, b) in enumerate(zip(gen, hf_greedy)) if a != b),
                    min(len(gen), len(hf_greedy)),
                )
                print(
                    f"{self.CHECKPOINT} {prompt!r}: {len(gen)} tokens, "
                    f"max margin {margins.max().item():.3f}, "
                    f"argmax agreement {(margins == 0).float().mean().item():.3f}, "
                    f"identical prefix with HF greedy {prefix}"
                )
                self.assertLessEqual(
                    margins.max().item(),
                    self.MAX_MARGIN,
                    f"{prompt!r}: token {int(margins.argmax())} of "
                    f"{tokenizer.decode(gen)!r} is not a near-argmax under HF",
                )
        finally:
            del hf_model
            torch.cuda.empty_cache()


class TestApertus8BInstruct(_RealCheckpointGreedyTest, unittest.TestCase):
    CHECKPOINT = APERTUS_8B_INSTRUCT


class TestApertus1p5_8B(_RealCheckpointGreedyTest, unittest.TestCase):
    """Apertus 1.5 has no upstream transformers class; the reference is HF's
    Apertus decoder on the checkpoint's text weights with the text-only lm_head."""

    CHECKPOINT = APERTUS_1P5_8B

    def _load_hf_model(self):
        from safetensors import safe_open

        with open(os.path.join(self.model_path, "config.json")) as f:
            text_config = dict(json.load(f)["text_config"])
        text_config["model_type"] = "apertus"
        config = ApertusConfig.from_dict(text_config)
        with torch.device("cuda"), default_dtype(torch.bfloat16):
            hf_model = HFApertusForCausalLM(config)
            hf_model.lm_head = torch.nn.Linear(
                config.hidden_size, text_config["output_vocab_size"], bias=False
            )

        with open(os.path.join(self.model_path, "model.safetensors.index.json")) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        state_dict = {}
        for shard in shards:
            with safe_open(os.path.join(self.model_path, shard), framework="pt") as f:
                for name in f.keys():
                    if name.startswith("model.language_model."):
                        key = "model." + name[len("model.language_model.") :]
                        state_dict[key] = f.get_tensor(name)
                    elif name == "lm_head.weight":
                        state_dict[name] = f.get_tensor(name)
        missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        return hf_model


if __name__ == "__main__":
    unittest.main()
