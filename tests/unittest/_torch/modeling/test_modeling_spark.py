# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Spark2_5 (XHToken Spark-X2.5/X2.6) PyTorch-backend model.

The pure-function tests (output gate, per-layer-type RoPE builders) are
cpu_only; the model-level tests require a GPU.
"""

import unittest
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.configs import Spark2_5Config
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_spark import (
    Spark2_5ForCausalLM,
    _apply_output_gate,
    _build_rope_params,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.functional import RotaryScalingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

# Tiny variant of the real 1.7B config (3:1 sliding/full pattern preserved).
SPARK_TINY_CONFIG = {
    "architectures": ["Spark2_5ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "gate_attn_act_mode": "sigmoid",
    "head_dim": 64,
    "headwise_attn_output_gate": True,
    "hidden_act": "gelu",
    "hidden_size": 128,
    "intermediate_size": 256,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "max_position_embeddings": 4096,
    "mlp_bias": False,
    "model_type": "spark2_5",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 2,
    "pad_token_id": 2,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "full_attention": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 5000000,
        },
        "sliding_attention": {
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000,
        },
    },
    "sliding_window": 8,
    "tie_word_embeddings": True,
    "vocab_size": 256,
    "dtype": "bfloat16",
    "torch_dtype": "bfloat16",
}


def _tiny_config() -> Spark2_5Config:
    return Spark2_5Config(**SPARK_TINY_CONFIG)


@pytest.mark.cpu_only
class TestSparkOutputGate:
    def test_apply_output_gate_per_head(self):
        torch.manual_seed(0)
        num_tokens, num_heads, head_dim = 5, 4, 8
        attn_output = torch.randn(num_tokens, num_heads * head_dim)
        gate_logits = torch.randn(num_tokens, num_heads)

        result = _apply_output_gate(attn_output, gate_logits, num_heads, head_dim)

        expected = (
            attn_output.view(num_tokens, num_heads, head_dim)
            * torch.sigmoid(gate_logits).unsqueeze(-1)
        ).view(num_tokens, num_heads * head_dim)
        torch.testing.assert_close(result, expected)
        assert result.shape == attn_output.shape

    def test_gate_is_sigmoid_not_softplus(self):
        num_tokens, num_heads, head_dim = 3, 2, 4
        attn_output = torch.ones(num_tokens, num_heads * head_dim)
        gate_logits = torch.randn(num_tokens, num_heads)

        result = _apply_output_gate(attn_output, gate_logits, num_heads, head_dim)

        sigmoid_expected = (
            torch.sigmoid(gate_logits)
            .unsqueeze(-1)
            .expand(num_tokens, num_heads, head_dim)
            .reshape(num_tokens, -1)
        )
        softplus = (
            F.softplus(gate_logits)
            .unsqueeze(-1)
            .expand(num_tokens, num_heads, head_dim)
            .reshape(num_tokens, -1)
        )
        torch.testing.assert_close(result, sigmoid_expected)
        assert not torch.allclose(result, softplus)

    def test_gate_computed_in_float32(self):
        num_tokens, num_heads, head_dim = 3, 2, 4
        attn_output = torch.ones(num_tokens, num_heads * head_dim, dtype=torch.bfloat16)
        # -20 underflows to 0 in bf16 sigmoid but not in fp32 sigmoid.
        gate_logits = torch.full((num_tokens, num_heads), -20.0, dtype=torch.bfloat16)

        result = _apply_output_gate(attn_output, gate_logits, num_heads, head_dim)

        expected = (
            torch.sigmoid(gate_logits.float())
            .to(torch.bfloat16)
            .unsqueeze(-1)
            .expand(num_tokens, num_heads, head_dim)
            .reshape(num_tokens, -1)
        )
        torch.testing.assert_close(result, expected)
        assert result.dtype == torch.bfloat16

    def test_gate_act_mode_silu(self):
        num_tokens, num_heads, head_dim = 3, 2, 4
        attn_output = torch.randn(num_tokens, num_heads * head_dim)
        gate_logits = torch.randn(num_tokens, num_heads)

        result = _apply_output_gate(attn_output, gate_logits, num_heads, head_dim, act_mode="silu")

        expected = (
            attn_output.view(num_tokens, num_heads, head_dim) * F.silu(gate_logits).unsqueeze(-1)
        ).view(num_tokens, num_heads * head_dim)
        torch.testing.assert_close(result, expected)

    def test_gate_invalid_act_mode(self):
        with pytest.raises(ValueError, match="gate_attn_act_mode"):
            _apply_output_gate(torch.ones(1, 8), torch.ones(1, 2), 2, 4, act_mode="softplus")

    def test_gate_zero_logits_half_scale(self):
        # A zero g_proj produces sigmoid(0)=0.5 scaling, not a zero output;
        # pin the exact semantics so the model-level test can rely on it.
        num_tokens, num_heads, head_dim = 2, 2, 4
        attn_output = torch.randn(num_tokens, num_heads * head_dim)
        gate_logits = torch.zeros(num_tokens, num_heads)

        result = _apply_output_gate(attn_output, gate_logits, num_heads, head_dim)

        torch.testing.assert_close(result, attn_output * 0.5)


@pytest.mark.cpu_only
class TestSparkRopeParams:
    def test_build_rope_params_full_layer(self):
        config = _tiny_config()
        rp = _build_rope_params(config, "full_attention")
        assert rp.theta == 5000000
        # head_dim=64, partial_rotary_factor=0.25 -> 16 rotary dims.
        assert rp.dim == int(config.head_dim * 0.25) == 16
        assert rp.scale_type == RotaryScalingType.none
        assert rp.scale == 1.0
        assert rp.max_positions == config.max_position_embeddings

    def test_build_rope_params_sliding_layer(self):
        config = _tiny_config()
        rp = _build_rope_params(config, "sliding_attention")
        assert rp.theta == 10000
        # Full rotation for sliding layers.
        assert rp.dim == config.head_dim == 64
        assert rp.scale_type == RotaryScalingType.none

    def test_build_rope_params_real_1_7b_dims(self):
        config = Spark2_5Config(
            head_dim=256,
            max_position_embeddings=1048576,
            rope_parameters={
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
                "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
            },
        )
        full = _build_rope_params(config, "full_attention")
        sliding = _build_rope_params(config, "sliding_attention")
        assert (full.dim, full.theta) == (64, 5000000)
        assert (sliding.dim, sliding.theta) == (256, 10000)
        assert full.max_positions == 1048576

    def test_build_rope_params_defaults_when_missing(self):
        # rope_parameters absent -> reference defaults (theta 10000, prf 1.0).
        config = Spark2_5Config(head_dim=128)
        rp = _build_rope_params(config, "full_attention")
        assert rp.theta == 10000
        assert rp.dim == 128

    def test_build_rope_params_max_positions_override(self):
        config = _tiny_config()
        rp = _build_rope_params(config, "full_attention", max_positions=128)
        assert rp.max_positions == 128


# ---------------------------------------------------------------------------
# GPU model-level tests
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class Scenario:
    backend: str

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}"


def _scenario_id(func, num, param):
    return f"{func.__name__}[{param.args[0]}]"


def _get_kv_cache_manager(
    dtype: torch.dtype,
    config: Spark2_5Config,
    tokens_per_block: int,
    max_seq_len: int,
    batch_size: int,
    num_blocks: int,
):
    if dtype == torch.half:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    else:
        raise ValueError("Invalid dtype")

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        enable_partial_reuse=False,
        copy_on_partial_reuse=False,
        max_tokens=num_blocks * tokens_per_block,
    )
    return KVCacheManagerV2(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )


def _build_tiny_model(backend: str, device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    config = Spark2_5Config(**SPARK_TINY_CONFIG)
    model_config = ModelConfig(pretrained_config=config, attn_backend=backend)
    model = Spark2_5ForCausalLM(model_config).to(config.torch_dtype).to(device)
    return model, model_config, config


@unittest.skipUnless(torch.cuda.is_available(), "TestSparkModel requires CUDA")
class TestSparkModel(unittest.TestCase):
    def _assert_most_elems_close(self, actual, ref, atol, rtol, max_failed_fraction):
        matches = torch.isclose(actual.float(), ref.float(), atol=atol, rtol=rtol)
        failed_fraction = (~matches).float().mean().item()
        assert failed_fraction <= max_failed_fraction, (
            f"Exceeded tolerance: {failed_fraction * 100:.2f}% of elements "
            f"differ more than allowed (max {max_failed_fraction * 100:.2f}%)"
        )

    @parameterized.expand(
        [
            Scenario(backend="TRTLLM"),
            Scenario(backend="VANILLA"),
            Scenario(backend="FLASHINFER"),
        ],
        _scenario_id,
    )
    @torch.no_grad()
    def test_spark_sanity(self, scenario: Scenario):
        device = torch.device("cuda")
        model, model_config, config = _build_tiny_model(scenario.backend, device)
        dtype = config.torch_dtype

        input_ids = torch.tensor([100, 200, 30, 40, 50, 60, 70, 80], dtype=torch.int, device=device)
        context_sequence_lengths = [3, 2, 1]
        sequence_lengths = context_sequence_lengths + [1, 1]
        past_seen_tokens = [0, 0, 0, 20, 30]
        request_ids = list(range(len(sequence_lengths)))
        token_nums = (torch.tensor(past_seen_tokens) + torch.tensor(sequence_lengths)).tolist()
        prompt_lens = token_nums[:3] + past_seen_tokens[3:]

        num_blocks = 100
        tokens_per_block = 64
        max_seq_len = num_blocks * tokens_per_block
        batch_size = len(sequence_lengths)
        kv_cache_manager = _get_kv_cache_manager(
            dtype, config, tokens_per_block, max_seq_len, batch_size, num_blocks
        )
        self.addCleanup(kv_cache_manager.shutdown)
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor(sequence_lengths, dtype=torch.int),
            num_contexts=len(context_sequence_lengths),
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=past_seen_tokens,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=batch_size,
            max_num_tokens=8192,
        )

        position_ids = []
        for i, tokens in enumerate(past_seen_tokens):
            seq_len = context_sequence_lengths[i] if i < len(context_sequence_lengths) else 1
            position_ids.append(torch.arange(tokens, tokens + seq_len, device=device))
        position_ids = torch.cat(position_ids).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )
        self.assertEqual(len(sequence_lengths), logits.shape[0])
        self.assertEqual(config.vocab_size, logits.shape[-1])
        self.assertTrue(torch.isfinite(logits.float()).all())

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )
        self.assertEqual(input_ids.shape, logits.shape[:-1])
        self.assertTrue(torch.isfinite(logits.float()).all())

    @torch.no_grad()
    def test_spark_generation_multistep(self):
        """Incremental decode must match a single full-context replay."""
        device = torch.device("cuda")
        backend = "TRTLLM"
        input_ids = torch.tensor(
            [100, 200, 30, 40, 50, 60, 70, 80, 90, 110], dtype=torch.int, device=device
        )

        num_blocks = 10
        tokens_per_block = 64
        max_seq_len = num_blocks * tokens_per_block

        # Pass 1: context of 8 tokens + two decode steps.
        model, model_config, config = _build_tiny_model(backend, device, seed=0)
        dtype = config.torch_dtype
        kv_cache_manager = _get_kv_cache_manager(
            dtype, config, tokens_per_block, max_seq_len, 1, num_blocks
        )
        self.addCleanup(kv_cache_manager.shutdown)
        kv_cache_manager.add_dummy_requests([0], [10])
        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata

        def make_metadata(seq_len, num_contexts, cached):
            return metadata_cls(
                seq_lens=torch.tensor([seq_len], dtype=torch.int),
                num_contexts=num_contexts,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[cached],
                ),
                kv_cache_manager=kv_cache_manager,
                request_ids=[0],
                prompt_lens=[10],
                max_num_requests=1,
                max_num_tokens=8192,
            )

        gen_logits = None
        with torch.inference_mode():
            md = make_metadata(8, 1, 0)
            md.prepare()
            model.forward(
                input_ids=input_ids[:8],
                position_ids=torch.arange(8, device=device).unsqueeze(0),
                attn_metadata=md,
            )
            for step in range(8, 10):
                md = make_metadata(1, 0, step)
                md.prepare()
                gen_logits = model.forward(
                    input_ids=input_ids[step : step + 1],
                    position_ids=torch.tensor([[step]], device=device),
                    attn_metadata=md,
                )

        # Pass 2: fresh identical model replays all 10 tokens as one context.
        model2, _, config2 = _build_tiny_model(backend, device, seed=0)
        kv_cache_manager2 = _get_kv_cache_manager(
            dtype, config2, tokens_per_block, max_seq_len, 1, num_blocks
        )
        self.addCleanup(kv_cache_manager2.shutdown)
        kv_cache_manager2.add_dummy_requests([0], [10])
        with torch.inference_mode():
            md2 = metadata_cls(
                seq_lens=torch.tensor([10], dtype=torch.int),
                num_contexts=1,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[0],
                ),
                kv_cache_manager=kv_cache_manager2,
                request_ids=[0],
                prompt_lens=[10],
                max_num_requests=1,
                max_num_tokens=8192,
            )
            md2.prepare()
            replay_logits = model2.forward(
                input_ids=input_ids,
                position_ids=torch.arange(10, device=device).unsqueeze(0),
                attn_metadata=md2,
            )

        self._assert_most_elems_close(
            gen_logits, replay_logits[-1:], atol=0.1, rtol=0.1, max_failed_fraction=0.01
        )

    @torch.no_grad()
    def test_spark_sliding_window_effect(self):
        """A token beyond the sliding window must not affect sliding layers
        but must affect full-attention layers."""
        device = torch.device("cuda")
        model, model_config, config = _build_tiny_model("TRTLLM", device)
        dtype = config.torch_dtype
        seq_len = 16  # > sliding_window (8)

        torch.manual_seed(42)
        hidden = torch.randn(seq_len, config.hidden_size, dtype=dtype, device=device)
        hidden_perturbed = hidden.clone()
        hidden_perturbed[0] += 1.0  # position 0 sits 15 tokens behind the last
        # query (position 15), far outside the 8-token window

        kv_cache_manager = _get_kv_cache_manager(dtype, config, 64, 1024, 1, 16)
        self.addCleanup(kv_cache_manager.shutdown)
        kv_cache_manager.add_dummy_requests([0], [seq_len])
        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        def run_layer(layer_idx, hidden_states):
            md = metadata_cls(
                seq_lens=torch.tensor([seq_len], dtype=torch.int),
                num_contexts=1,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[0],
                ),
                kv_cache_manager=kv_cache_manager,
                request_ids=[0],
                prompt_lens=[seq_len],
                max_num_requests=1,
                max_num_tokens=8192,
            )
            md.prepare()
            out, _ = model.model.layers[layer_idx](
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=md,
            )
            return out

        with torch.inference_mode():
            # Layer 0 is a sliding-attention layer (layer_types[0]).
            assert config.layer_types[0] == "sliding_attention"
            base = run_layer(0, hidden)
            perturbed = run_layer(0, hidden_perturbed)
            torch.testing.assert_close(base[-1], perturbed[-1])

            # Layer 3 is a full-attention layer.
            assert config.layer_types[3] == "full_attention"
            base_full = run_layer(3, hidden)
            perturbed_full = run_layer(3, hidden_perturbed)
            self.assertFalse(torch.allclose(base_full[-1], perturbed_full[-1]))
            # Sanity: the perturbed position itself does change for sliding.
            self.assertFalse(torch.allclose(base[0], perturbed[0]))

    @torch.no_grad()
    def test_spark_gating_effect_in_forward(self):
        """sigmoid(+60) == 1.0 must match a bypassed gate; sigmoid(-60) ~ 0
        must change the logits."""
        device = torch.device("cuda")
        model, model_config, config = _build_tiny_model("TRTLLM", device)
        dtype = config.torch_dtype

        input_ids = torch.tensor([100, 200, 30, 40], dtype=torch.int, device=device)
        kv_cache_manager = _get_kv_cache_manager(dtype, config, 64, 1024, 1, 16)
        self.addCleanup(kv_cache_manager.shutdown)
        kv_cache_manager.add_dummy_requests([0], [4])
        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        position_ids = torch.arange(4, device=device).unsqueeze(0)

        def run():
            md = metadata_cls(
                seq_lens=torch.tensor([4], dtype=torch.int),
                num_contexts=1,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[0],
                ),
                kv_cache_manager=kv_cache_manager,
                request_ids=[0],
                prompt_lens=[4],
                max_num_requests=1,
                max_num_tokens=8192,
            )
            md.prepare()
            return model.forward(input_ids=input_ids, position_ids=position_ids, attn_metadata=md)

        def set_constant_gate(value: float):
            for layer in model.model.layers:
                attn = layer.self_attn
                num_heads_local = attn.num_heads

                def const_gate(x, _v=value, _n=num_heads_local):
                    return torch.full((x.shape[0], _n), _v, dtype=dtype, device=x.device)

                attn.g_proj.forward = const_gate
                attn.use_headwise_gate = True

        def bypass_gate():
            for layer in model.model.layers:
                layer.self_attn.use_headwise_gate = False

        with torch.inference_mode():
            set_constant_gate(60.0)
            logits_gate_open = run()
            bypass_gate()
            logits_bypass = run()
            set_constant_gate(-60.0)
            logits_gate_closed = run()

        # sigmoid(60) rounds to 1.0 in fp32 -> identical to no gate.
        torch.testing.assert_close(logits_gate_open, logits_bypass)
        # sigmoid(-60) ~ 0 -> attention output suppressed -> different logits.
        self.assertFalse(torch.allclose(logits_gate_closed, logits_bypass, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.isfinite(logits_gate_closed.float()).all())

    def test_gate_disables_quantized_attention_output(self):
        """The headwise gate needs a plain tensor: _use_quantize_output must
        opt out the same way the base class does for attn_output_gate, or an
        FP8/NVFP4 o_proj would make the kernel emit a pre-quantized output
        (fp8 dtype or Fp4QuantizedTensor) that the gate math cannot consume."""
        device = torch.device("cuda")
        model, _, _ = _build_tiny_model("TRTLLM", device)
        attn = model.model.layers[0].self_attn
        self.assertTrue(attn.use_headwise_gate)

        # Simulate a quantized checkpoint: the base opt-in condition becomes true.
        attn.has_quant_scale = True
        self.assertFalse(attn._use_quantize_output())

        # Without the headwise gate the same setup keeps the base behavior.
        attn.use_headwise_gate = False
        self.assertTrue(attn._use_quantize_output())

    @torch.no_grad()
    def test_spark_partial_rope_numerics(self):
        """Full-attention layers rotate only the first head_dim * 0.25 dims,
        matching the HF reference math (theta=5e6, neox rotate-half)."""
        device = torch.device("cuda")
        config_dict = dict(SPARK_TINY_CONFIG)
        config_dict["dtype"] = "float32"
        config_dict["torch_dtype"] = "float32"
        torch.manual_seed(0)
        config = Spark2_5Config(**config_dict)
        model_config = ModelConfig(pretrained_config=config)
        model = Spark2_5ForCausalLM(model_config).to(torch.float32).to(device)

        layer_idx = 3  # full_attention layer
        attn = model.model.layers[layer_idx].self_attn
        rope_dim = int(config.head_dim * 0.25)
        theta = 5000000.0

        num_tokens = 5
        position_ids = torch.arange(num_tokens, device=device).unsqueeze(0)
        torch.manual_seed(1)
        q_dim = config.num_attention_heads * config.head_dim
        kv_dim = config.num_key_value_heads * config.head_dim
        qkv = torch.randn(num_tokens, q_dim + 2 * kv_dim, dtype=torch.float32, device=device)

        q, k, v = attn.apply_rope(qkv, None, None, position_ids)
        q = q.view(num_tokens, config.num_attention_heads, config.head_dim)
        k = k.view(num_tokens, config.num_key_value_heads, config.head_dim)

        # HF reference (modeling_spark.py compute_rope_cos_sin +
        # apply_rotary_pos_emb).
        inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
        freqs = torch.outer(torch.arange(num_tokens, dtype=torch.float32), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1).to(device)
        cos = emb.cos()[:, None, :]
        sin = emb.sin()[:, None, :]

        def rotate_half(t):
            t1 = t[..., : t.shape[-1] // 2]
            t2 = t[..., t.shape[-1] // 2 :]
            return torch.cat((-t2, t1), dim=-1)

        def ref_rope(x):
            x_rot = x[..., :rope_dim]
            x_pass = x[..., rope_dim:]
            rotated = x_rot * cos + rotate_half(x_rot) * sin
            return torch.cat([rotated, x_pass], dim=-1)

        q_ref = ref_rope(
            qkv[:, :q_dim].view(num_tokens, config.num_attention_heads, config.head_dim)
        )
        k_ref = ref_rope(
            qkv[:, q_dim : q_dim + kv_dim].view(
                num_tokens, config.num_key_value_heads, config.head_dim
            )
        )
        torch.testing.assert_close(q, q_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(k, k_ref, atol=1e-4, rtol=1e-4)
        # v passes through untouched.
        torch.testing.assert_close(v.reshape(num_tokens, kv_dim), qkv[:, q_dim + kv_dim :])

    @torch.no_grad()
    def test_spark_weight_loading_synthetic(self):
        """HF-style checkpoint (fused q_k_v_proj, embedding/out_proj names)
        loads through SparkHfWeightMapper into the TRT-LLM module layout."""
        device = torch.device("cuda")
        model, _, config = _build_tiny_model("TRTLLM", device)
        dtype = config.torch_dtype
        nh, nkv, hd = (config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        q_dim, kv_dim = nh * hd, nkv * hd
        hidden, inter, vocab = (config.hidden_size, config.intermediate_size, config.vocab_size)

        def t(*shape):
            return torch.randn(*shape, dtype=dtype, device=device)

        sd = {"model.embedding.weight": t(vocab, hidden)}
        for i in range(config.num_hidden_layers):
            p = f"model.layers.{i}"
            q, k_, v_ = t(q_dim, hidden), t(kv_dim, hidden), t(kv_dim, hidden)
            sd[f"{p}.self_attn.q_k_v_proj.weight"] = torch.cat([q, k_, v_], dim=0)
            sd[f"{p}.self_attn.g_proj.weight"] = t(nh, hidden)
            sd[f"{p}.self_attn.out_proj.weight"] = t(hidden, q_dim)
            sd[f"{p}.mlp.gate_proj.weight"] = t(inter, hidden)
            sd[f"{p}.mlp.up_proj.weight"] = t(inter, hidden)
            sd[f"{p}.mlp.down_proj.weight"] = t(hidden, inter)
            sd[f"{p}.input_layernorm.weight"] = t(hidden)
            sd[f"{p}.post_attention_layernorm.weight"] = t(hidden)
            # Stash expectations for layer 0 before names go out of scope.
            if i == 0:
                expected = {
                    "qkv": torch.cat([q, k_, v_], dim=0),
                    "g": sd[f"{p}.self_attn.g_proj.weight"],
                    "o": sd[f"{p}.self_attn.out_proj.weight"],
                    "gate_up": torch.cat(
                        [
                            sd[f"{p}.mlp.gate_proj.weight"],
                            sd[f"{p}.mlp.up_proj.weight"],
                        ],
                        dim=0,
                    ),
                    "down": sd[f"{p}.mlp.down_proj.weight"],
                }
        sd["model.norm.weight"] = t(hidden)

        model.load_weights(sd)

        attn0 = model.model.layers[0].self_attn
        torch.testing.assert_close(attn0.qkv_proj.weight.data, expected["qkv"])
        torch.testing.assert_close(attn0.g_proj.weight.data, expected["g"])
        torch.testing.assert_close(attn0.o_proj.weight.data, expected["o"])
        mlp0 = model.model.layers[0].mlp
        torch.testing.assert_close(mlp0.gate_up_proj.weight.data, expected["gate_up"])
        torch.testing.assert_close(mlp0.down_proj.weight.data, expected["down"])
        torch.testing.assert_close(
            model.model.embed_tokens.weight.data, sd["model.embedding.weight"]
        )
        torch.testing.assert_close(model.model.norm.weight.data, sd["model.norm.weight"])
        # tie_word_embeddings: lm_head shares the embedding storage.
        self.assertEqual(
            model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr()
        )


def _load_hf_reference():
    """Load the checkpoint's reference implementation (modeling_spark.py +
    configuration_spark.py) from SPARK_HF_REFERENCE_DIR as a synthetic
    package, so its relative imports resolve."""
    import importlib.util
    import os
    import sys
    import types

    ref_dir = os.environ.get("SPARK_HF_REFERENCE_DIR")
    if not ref_dir:
        pytest.skip("SPARK_HF_REFERENCE_DIR not set; skipping HF alignment")
    modeling_path = os.path.join(ref_dir, "modeling_spark.py")
    config_path = os.path.join(ref_dir, "configuration_spark.py")
    if not (os.path.isfile(modeling_path) and os.path.isfile(config_path)):
        pytest.skip(f"reference files not found under {ref_dir}")

    if "spark_ref.modeling_spark" in sys.modules:
        return sys.modules["spark_ref.modeling_spark"]

    pkg = types.ModuleType("spark_ref")
    pkg.__path__ = [ref_dir]
    sys.modules["spark_ref"] = pkg
    for name, path in (("configuration_spark", config_path), ("modeling_spark", modeling_path)):
        full_name = f"spark_ref.{name}"
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    modeling = sys.modules["spark_ref.modeling_spark"]
    # The checkpoint's modeling_spark.py targets transformers 4.57, where
    # _tied_weights_keys is a list; transformers 5.x requires the
    # {target: source} dict form and raises AttributeError in post_init
    # otherwise. Tying does not affect forward parity: the reference forward
    # computes logits from the embedding weight directly when
    # tie_word_embeddings is set.
    import transformers

    tied = getattr(modeling.Spark2_5ForCausalLM, "_tied_weights_keys", None)
    if int(transformers.__version__.split(".")[0]) >= 5 and isinstance(tied, (list, tuple)):
        modeling.Spark2_5ForCausalLM._tied_weights_keys = {
            "lm_head.weight": "model.embedding.weight"
        }

    # The reference calls create_causal_mask with the transformers-4.57
    # keyword set (`input_embeds`, `cache_position`); transformers 5.x renamed
    # the former to `inputs_embeds` and dropped the latter. Wrap the two mask
    # builders the reference imported so either signature works.
    import inspect

    def _compat_mask_fn(fn):
        sig = inspect.signature(fn)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        def wrapped(**kwargs):
            if "input_embeds" not in params and "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            elif "inputs_embeds" not in params and "inputs_embeds" in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            if not accepts_var_kw:
                kwargs = {k: v for k, v in kwargs.items() if k in params}
            return fn(**kwargs)

        return wrapped

    for fn_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        if hasattr(modeling, fn_name):
            setattr(modeling, fn_name, _compat_mask_fn(getattr(modeling, fn_name)))
    return modeling


def _hf_style_state_dict(
    config: Spark2_5Config, device: torch.device, dtype: torch.dtype, seed: int = 7
) -> dict:
    """Synthetic Spark HF-checkpoint state dict (fused q_k_v_proj, embedding /
    out_proj names), shared by the TRT-LLM and HF-reference models."""
    torch.manual_seed(seed)
    nh, nkv, hd = (config.num_attention_heads, config.num_key_value_heads, config.head_dim)
    q_dim, kv_dim = nh * hd, nkv * hd
    hidden, inter, vocab = (config.hidden_size, config.intermediate_size, config.vocab_size)

    def t(*shape):
        return torch.randn(*shape, dtype=dtype, device=device) * 0.02

    sd = {"model.embedding.weight": t(vocab, hidden)}
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        sd[f"{p}.self_attn.q_k_v_proj.weight"] = t(q_dim + 2 * kv_dim, hidden)
        sd[f"{p}.self_attn.g_proj.weight"] = t(nh, hidden)
        sd[f"{p}.self_attn.out_proj.weight"] = t(hidden, q_dim)
        sd[f"{p}.mlp.gate_proj.weight"] = t(inter, hidden)
        sd[f"{p}.mlp.up_proj.weight"] = t(inter, hidden)
        sd[f"{p}.mlp.down_proj.weight"] = t(hidden, inter)
        sd[f"{p}.input_layernorm.weight"] = torch.ones(hidden, dtype=dtype, device=device)
        sd[f"{p}.post_attention_layernorm.weight"] = torch.ones(hidden, dtype=dtype, device=device)
    sd["model.norm.weight"] = torch.ones(hidden, dtype=dtype, device=device)
    return sd


@dataclass(repr=False)
class AlignScenario:
    backend: str
    dtype: torch.dtype
    kv_dtype: "tensorrt_llm.bindings.DataType"
    atol: float
    rtol: float
    max_failed_fraction: float

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_dtype:{str(self.dtype).split('.')[-1]}"


@unittest.skipUnless(torch.cuda.is_available(), "TestSparkHfAlignment requires CUDA")
class TestSparkHfAlignment(unittest.TestCase):
    """Logits parity against the checkpoint's HF reference implementation on a
    tiny config with identical synthetic weights.

    VANILLA runs in fp32 (the reference computes internally in fp32) with a
    tight tolerance. TRTLLM runs in bf16 -- its FMHA kernels do not take an
    fp32 KV cache -- so it uses the loose tolerance convention of
    test_modeling_gemma3.py's HF comparison.
    """

    def _assert_most_elems_close(self, actual, ref, atol, rtol, max_failed_fraction):
        matches = torch.isclose(actual.float(), ref.float(), atol=atol, rtol=rtol)
        failed_fraction = (~matches).float().mean().item()
        assert failed_fraction <= max_failed_fraction, (
            f"Exceeded tolerance: {failed_fraction * 100:.2f}% of elements "
            f"differ more than allowed (max {max_failed_fraction * 100:.2f}%)"
        )

    @parameterized.expand(
        [
            AlignScenario(
                backend="VANILLA",
                dtype=torch.float32,
                kv_dtype=tensorrt_llm.bindings.DataType.FLOAT,
                atol=1e-2,
                rtol=1e-2,
                max_failed_fraction=0.01,
            ),
            AlignScenario(
                backend="TRTLLM",
                dtype=torch.bfloat16,
                kv_dtype=tensorrt_llm.bindings.DataType.BF16,
                atol=0.4,
                rtol=0.4,
                max_failed_fraction=0.001,
            ),
        ],
        lambda func, num, param: f"{func.__name__}[{param.args[0]}]",
    )
    @torch.no_grad()
    def test_spark_hf_alignment(self, scenario: AlignScenario):
        ref = _load_hf_reference()
        device = torch.device("cuda")
        dtype_str = "float32" if scenario.dtype == torch.float32 else "bfloat16"

        config_dict = dict(SPARK_TINY_CONFIG)
        config_dict["dtype"] = dtype_str
        config_dict["torch_dtype"] = dtype_str
        torch.manual_seed(0)
        config = Spark2_5Config(**config_dict)
        model_config = ModelConfig(pretrained_config=config, attn_backend=scenario.backend)
        model = Spark2_5ForCausalLM(model_config).to(scenario.dtype).to(device)

        sd = _hf_style_state_dict(config, device, scenario.dtype)
        model.load_weights(sd)

        ref_config = ref.Spark2_5Config(**config_dict)
        ref_model = ref.Spark2_5ForCausalLM(ref_config).to(scenario.dtype).to(device).eval()
        missing, unexpected = ref_model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all(k.startswith("lm_head") for k in missing), f"unexpected missing keys: {missing}"

        input_ids = torch.tensor(
            [[100, 200, 30, 40, 50, 60, 70, 80]], dtype=torch.long, device=device
        )

        ref_logits = ref_model.forward(input_ids=input_ids, use_cache=False).logits[0]

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(
                enable_block_reuse=False,
                enable_partial_reuse=False,
                copy_on_partial_reuse=False,
                max_tokens=16 * 64,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            tokens_per_block=64,
            max_seq_len=1024,
            max_batch_size=1,
            mapping=mapping,
            dtype=scenario.kv_dtype,
        )
        self.addCleanup(kv_cache_manager.shutdown)
        kv_cache_manager.add_dummy_requests([0], [8])

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([8], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=[0],
            prompt_lens=[8],
            max_num_requests=1,
            max_num_tokens=8192,
        )

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(
                input_ids=input_ids[0],
                position_ids=torch.arange(8, device=device).unsqueeze(0),
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )

        self._assert_most_elems_close(
            logits,
            ref_logits,
            atol=scenario.atol,
            rtol=scenario.rtol,
            max_failed_fraction=scenario.max_failed_fraction,
        )


GOLDEN_PATH = Path(__file__).parent / "spark_golden" / "spark2_5_golden.pt"


@dataclass(repr=False)
class GoldenScenario:
    backend: str
    dtype: torch.dtype
    kv_dtype: "tensorrt_llm.bindings.DataType"
    atol: float
    rtol: float
    max_failed_fraction: float
    min_row_cosine: float
    min_top1_agreement: float

    def __repr__(self) -> str:
        return f"backend:{self.backend.lower()}_dtype:{str(self.dtype).split('.')[-1]}"


@unittest.skipUnless(torch.cuda.is_available(), "TestSparkGoldenParity requires CUDA")
class TestSparkGoldenParity(unittest.TestCase):
    """CI-enforced parity against the HF reference implementation via a
    committed golden artifact (spark_golden/generate_spark2_5_golden.py).

    The golden weights use HF checkpoint naming (fused q_k_v_proj, model.embedding,
    out_proj), so loading them also validates SparkHfWeightMapper's renames and
    the q|k|v split order: a wrong split order cannot reproduce the reference
    logits. VANILLA runs fp32 against the fp32 golden with a tight tolerance;
    TRTLLM runs bf16 (its fused RoPE path included) with a tolerance derived
    from bf16 precision (~0.4% relative per op over a 2-layer stack). Both
    scenarios additionally assert scale-independent shape-of-distribution
    checks (per-row cosine similarity and top-1 agreement) so a systematic
    gate/RoPE/window error cannot hide inside the absolute tolerance band of
    the small-magnitude golden logits.
    """

    @parameterized.expand(
        [
            GoldenScenario(
                backend="VANILLA",
                dtype=torch.float32,
                kv_dtype=tensorrt_llm.bindings.DataType.FLOAT,
                atol=1e-3,
                rtol=1e-3,
                max_failed_fraction=0.01,
                min_row_cosine=0.999,
                min_top1_agreement=1.0,
            ),
            GoldenScenario(
                backend="TRTLLM",
                dtype=torch.bfloat16,
                kv_dtype=tensorrt_llm.bindings.DataType.BF16,
                atol=0.1,
                rtol=0.1,
                max_failed_fraction=0.02,
                min_row_cosine=0.99,
                min_top1_agreement=0.9,
            ),
        ],
        lambda func, num, param: f"{func.__name__}[{param.args[0]}]",
    )
    @torch.no_grad()
    def test_spark_golden_parity(self, scenario: GoldenScenario):
        golden = torch.load(GOLDEN_PATH, map_location="cpu", weights_only=True)
        device = torch.device("cuda")
        dtype_str = "float32" if scenario.dtype == torch.float32 else "bfloat16"

        config_dict = dict(golden["config"])
        config_dict["dtype"] = dtype_str
        config_dict["torch_dtype"] = dtype_str
        config = Spark2_5Config(**config_dict)
        model_config = ModelConfig(pretrained_config=config, attn_backend=scenario.backend)
        model = Spark2_5ForCausalLM(model_config).to(scenario.dtype).to(device)
        model.load_weights(
            {k: v.to(scenario.dtype).to(device) for k, v in golden["state_dict"].items()}
        )

        input_ids = torch.tensor(golden["input_ids"], dtype=torch.int, device=device)
        seq_len = input_ids.numel()

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_manager = KVCacheManagerV2(
            KvCacheConfig(
                enable_block_reuse=False,
                enable_partial_reuse=False,
                copy_on_partial_reuse=False,
                max_tokens=64,
            ),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            tokens_per_block=16,
            max_seq_len=64,
            max_batch_size=1,
            mapping=mapping,
            dtype=scenario.kv_dtype,
        )
        self.addCleanup(kv_cache_manager.shutdown)
        kv_cache_manager.add_dummy_requests([0], [seq_len])

        metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([seq_len], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=[0],
            prompt_lens=[seq_len],
            max_num_requests=1,
            max_num_tokens=8192,
        )

        with torch.inference_mode():
            attn_metadata.prepare()
            logits = model.forward(
                input_ids=input_ids,
                position_ids=torch.arange(seq_len, device=device).unsqueeze(0),
                attn_metadata=attn_metadata,
                return_context_logits=True,
            )

        ref_logits = golden["logits"]
        got = logits.float().cpu()
        ref = ref_logits.float()
        matches = torch.isclose(got, ref, atol=scenario.atol, rtol=scenario.rtol)
        failed_fraction = (~matches).float().mean().item()
        assert failed_fraction <= scenario.max_failed_fraction, (
            f"Exceeded tolerance vs HF-reference golden: "
            f"{failed_fraction * 100:.2f}% of logits differ more than allowed "
            f"(max {scenario.max_failed_fraction * 100:.2f}%)"
        )

        # Scale-independent checks: a systematic gate/RoPE/window error shifts
        # the logit distribution in ways the absolute band above could hide at
        # this golden's small logit magnitude (std ~0.2).
        row_cosine = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
        worst_row = int(row_cosine.argmin().item())
        assert row_cosine.min().item() >= scenario.min_row_cosine, (
            f"Row cosine similarity {row_cosine.min().item():.4f} at position {worst_row} "
            f"below {scenario.min_row_cosine}"
        )
        top1 = (got.argmax(-1) == ref.argmax(-1)).float().mean().item()
        assert top1 >= scenario.min_top1_agreement, (
            f"Top-1 agreement {top1:.2%} below {scenario.min_top1_agreement:.2%}"
        )
