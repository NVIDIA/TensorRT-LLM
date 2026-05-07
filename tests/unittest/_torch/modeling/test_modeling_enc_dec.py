# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for the PyTorch-flow encoder-decoder modules.

Tests that modules can be constructed and run forward passes on dummy tensors.
Most cases use the VANILLA attention backend for isolated unit testing; the
TRTLLM cross-attention tests additionally validate cached-KV correctness
against the VANILLA reference. The TRTLLM cross-attn path runs on Blackwell
via the ``trtllm_gen`` sub-path and on Hopper / Ampere / earlier via the
legacy ``thop.attention`` C++ wrapper.
"""

import unittest
from copy import deepcopy

import torch
from transformers import BartConfig, T5Config

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_bart import BartDecoderLayer, BartEncoderLayer, BartModel
from tensorrt_llm._torch.models.modeling_t5 import (
    T5DecoderLayer,
    T5Encoder,
    T5EncoderLayer,
    T5Model,
)
from tensorrt_llm._torch.modules.cross_attention import CrossAttention


def _make_vanilla_metadata(seq_lens, device="cuda"):
    """Create a minimal VanillaAttentionMetadata for testing."""
    metadata_cls = get_attention_backend("VANILLA").Metadata
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32)
    total_tokens = sum(seq_lens)
    num_requests = len(seq_lens)
    metadata = metadata_cls(
        max_num_requests=num_requests,
        max_num_tokens=total_tokens,
        kv_cache_manager=None,
        request_ids=list(range(num_requests)),
        prompt_lens=seq_lens,
        seq_lens=seq_lens_tensor,
        num_contexts=num_requests,
    )
    metadata.max_seq_len = max(seq_lens)
    metadata.prepare()
    return metadata


# Small T5 config for fast testing
SMALL_T5_CONFIG = {
    "architectures": ["T5ForConditionalGeneration"],
    "d_model": 64,
    "d_kv": 8,
    "d_ff": 128,
    "num_heads": 8,
    "num_layers": 2,
    "num_decoder_layers": 2,
    "vocab_size": 100,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "relu",
    "is_encoder_decoder": True,
    "is_gated_act": False,
    "model_type": "t5",
    "decoder_start_token_id": 0,
    "pad_token_id": 0,
    "eos_token_id": 1,
    "torch_dtype": "bfloat16",
}

# Small BART config for fast testing
SMALL_BART_CONFIG = {
    "architectures": ["BartForConditionalGeneration"],
    "d_model": 64,
    "encoder_ffn_dim": 128,
    "decoder_ffn_dim": 128,
    "encoder_layers": 2,
    "decoder_layers": 2,
    "encoder_attention_heads": 8,
    "decoder_attention_heads": 8,
    "vocab_size": 100,
    "max_position_embeddings": 128,
    "activation_function": "gelu",
    "is_encoder_decoder": True,
    "model_type": "bart",
    "decoder_start_token_id": 2,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "bos_token_id": 0,
    "torch_dtype": "bfloat16",
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCrossAttention(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)

    def test_cross_attention_forward(self):
        """CrossAttention projects K/V from encoder and outputs correct shape."""
        device = torch.device("cuda")
        dtype = torch.bfloat16
        hidden_size = 64
        num_heads = 8
        num_tokens_decoder = 4
        num_tokens_encoder = 8

        t5_cfg = deepcopy(SMALL_T5_CONFIG)
        t5_cfg["torch_dtype"] = "bfloat16"
        config = ModelConfig(
            pretrained_config=T5Config.from_dict(t5_cfg),
            attn_backend="VANILLA",
        )
        cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            encoder_hidden_size=hidden_size,
            bias=False,
            layer_idx=0,
            dtype=dtype,
            config=config,
        ).to(device)

        decoder_hs = torch.randn(num_tokens_decoder, hidden_size, device=device, dtype=dtype)
        encoder_hs = torch.randn(num_tokens_encoder, hidden_size, device=device, dtype=dtype)
        metadata = _make_vanilla_metadata([num_tokens_decoder])

        with torch.inference_mode():
            output = cross_attn(
                hidden_states=decoder_hs,
                encoder_hidden_states=encoder_hs,
                attn_metadata=metadata,
                skip_cross_kv_projection=False,
            )
        self.assertEqual(output.shape, (num_tokens_decoder, hidden_size))


def _build_trtllm_cross_metadata(
    decoder_seq_lens,
    encoder_seq_lens,
    *,
    num_kv_heads,
    head_dim,
    dtype,
    skip_cross_kv_projection: bool = False,
    kv_managers=None,
    kv_cache_manager_cls=None,
):
    """Build a TrtllmAttentionMetadata + cross sub-metadata for CrossAttention.

    Sets up a proper KV-cache-managed cross pool (CacheType.CROSS) so the
    TRTLLM ``trtllm-gen`` backend can read paged K/V offsets. The decoder
    self-attention metadata uses a small (unused) SELF pool just to satisfy
    the wrapper's metadata expectations; only the cross sub-metadata is
    used by the cross-attention forward call. When ``kv_managers`` is
    provided, reuse the existing SELF/CROSS managers so generation tests can
    read encoder K/V written during an earlier context pass.

    ``kv_cache_manager_cls`` selects the KV cache manager class for both
    pools (V1 ``KVCacheManager`` or V2 ``KVCacheManagerV2``).  Defaults
    to V2 to preserve backward compatibility with existing call sites. The
    parametrized sibling test classes below cover the V1 production lane.
    """
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    if kv_cache_manager_cls is None:
        kv_cache_manager_cls = KVCacheManagerV2

    metadata_cls = get_attention_backend("TRTLLM").Metadata
    num_seqs = len(decoder_seq_lens)
    assert len(encoder_seq_lens) == num_seqs

    if dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    elif dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    else:
        raise ValueError(f"Unsupported KV cache dtype: {dtype}")

    page_size = 32
    max_encoder_len = max(int(x) for x in encoder_seq_lens)
    max_decoder_len = max(int(x) for x in decoder_seq_lens)
    blocks_per_seq = max(1, (max_encoder_len + page_size - 1) // page_size)
    cross_max_seq_len = blocks_per_seq * page_size

    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    cross_cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.CROSS
    self_cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF

    request_ids = list(range(num_seqs))
    if kv_managers is None:
        enc_dec_kv_cache_manager = kv_cache_manager_cls(
            KvCacheConfig(max_tokens=num_seqs * cross_max_seq_len),
            cross_cache_type,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_seq_len=cross_max_seq_len,
            max_batch_size=num_seqs,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )
        self_kv_cache_manager = kv_cache_manager_cls(
            KvCacheConfig(max_tokens=num_seqs * page_size),
            self_cache_type,
            num_layers=1,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=page_size,
            max_seq_len=page_size,
            max_batch_size=num_seqs,
            mapping=mapping,
            dtype=kv_cache_dtype,
        )

        enc_dec_kv_cache_manager.add_dummy_requests(request_ids, [int(x) for x in encoder_seq_lens])
        self_kv_cache_manager.add_dummy_requests(request_ids, [int(x) for x in decoder_seq_lens])
    else:
        self_kv_cache_manager, enc_dec_kv_cache_manager = kv_managers

    decoder_seq_lens_tensor = torch.tensor([int(x) for x in decoder_seq_lens], dtype=torch.int32)
    encoder_seq_lens_tensor = torch.tensor([int(x) for x in encoder_seq_lens], dtype=torch.int32)

    metadata = metadata_cls(
        max_num_requests=num_seqs,
        max_num_tokens=sum(int(x) for x in decoder_seq_lens),
        kv_cache_manager=self_kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=[int(x) for x in decoder_seq_lens],
        seq_lens=decoder_seq_lens_tensor,
        num_contexts=0 if skip_cross_kv_projection else num_seqs,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
    )
    metadata.max_seq_len = max(max_decoder_len, page_size)
    metadata.prepare()

    encoder_cached = (
        [int(x) for x in encoder_seq_lens] if skip_cross_kv_projection else [0] * num_seqs
    )
    cross_metadata = metadata.create_cross_metadata(
        encoder_seq_lens=encoder_seq_lens_tensor,
        enc_dec_kv_cache_manager=enc_dec_kv_cache_manager,
        encoder_num_cached_tokens_per_seq=encoder_cached,
    )
    cross_metadata.prepare()
    return metadata, cross_metadata, (self_kv_cache_manager, enc_dec_kv_cache_manager)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCrossAttentionTrtllmBackend(unittest.TestCase):
    """Validate CrossAttention on the TRTLLM backend.

    On Blackwell (SM100/SM103) the request flows through the ``trtllm_gen``
    sub-path; on Hopper / Ampere / earlier it flows through the legacy
    ``thop.attention`` sub-path.

    Subclasses override ``kv_cache_manager_cls`` to run the same correctness
    cases on the V1 ``KVCacheManager`` (the production lane and default
    target) and the V2 ``KVCacheManagerV2`` (the additive secondary path).
    The base class defaults to V2 so the existing CI lanes keep their
    current coverage; ``TestCrossAttentionTrtllmBackendV1`` re-runs the
    same suite on V1.
    """

    kv_cache_manager_cls = None  # ``None`` lets the helper default to V2.

    def setUp(self):
        torch.random.manual_seed(42)

    def _make_cross_attn(self, hidden_size, num_heads, head_dim, dtype, *, backend="TRTLLM"):
        t5_cfg = deepcopy(SMALL_T5_CONFIG)
        t5_cfg["d_model"] = hidden_size
        t5_cfg["num_heads"] = num_heads
        t5_cfg["d_kv"] = head_dim
        t5_cfg["torch_dtype"] = "bfloat16" if dtype == torch.bfloat16 else "float16"
        pretrained_config = T5Config.from_dict(t5_cfg)
        pretrained_config.head_dim = head_dim
        config = ModelConfig(
            pretrained_config=pretrained_config,
            attn_backend=backend,
        )
        cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            encoder_hidden_size=hidden_size,
            bias=False,
            layer_idx=0,
            dtype=dtype,
            config=config,
        )
        # ``Linear.create_weights`` allocates ``torch.empty`` parameters.
        # The unit test never calls ``load_weights``, so initialise the
        # projection weights with a small Gaussian so the forward pass
        # exercises real arithmetic instead of uninitialised memory.
        for proj in (cross_attn.q_proj, cross_attn.k_proj, cross_attn.v_proj, cross_attn.o_proj):
            torch.nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        return cross_attn

    def _make_cross_attn_pair(self, hidden_size, num_heads, head_dim, dtype, device):
        trtllm_cross_attn = self._make_cross_attn(
            hidden_size,
            num_heads,
            head_dim,
            dtype,
            backend="TRTLLM",
        )
        vanilla_cross_attn = self._make_cross_attn(
            hidden_size,
            num_heads,
            head_dim,
            dtype,
            backend="VANILLA",
        )
        vanilla_cross_attn.load_state_dict(trtllm_cross_attn.state_dict())
        return trtllm_cross_attn.to(device), vanilla_cross_attn.to(device)

    def _assert_matches_vanilla_reference(
        self, trtllm_output, vanilla_output, *, max_abs_tol, mean_abs_tol
    ):
        self.assertEqual(trtllm_output.shape, vanilla_output.shape)
        self.assertTrue(torch.isfinite(trtllm_output).all())
        self.assertTrue(torch.isfinite(vanilla_output).all())
        abs_diff = (trtllm_output.float() - vanilla_output.float()).abs()
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()
        self.assertLess(
            max_abs_diff,
            max_abs_tol,
            f"max abs diff {max_abs_diff} exceeded tolerance {max_abs_tol}",
        )
        self.assertLess(
            mean_abs_diff,
            mean_abs_tol,
            f"mean abs diff {mean_abs_diff} exceeded tolerance {mean_abs_tol}",
        )

    def _make_vanilla_cross_metadata(self, decoder_seq_lens, encoder_seq_lens, device):
        vanilla_metadata = _make_vanilla_metadata(decoder_seq_lens, device)
        vanilla_cross_metadata = vanilla_metadata.create_cross_metadata(
            encoder_seq_lens=torch.tensor([int(x) for x in encoder_seq_lens], dtype=torch.int32),
            enc_dec_kv_cache_manager=None,
        )
        vanilla_cross_metadata.prepare()
        return vanilla_metadata, vanilla_cross_metadata

    def test_attn_backend_selection(self):
        """CrossAttention picks the TRTLLM backend on every architecture."""
        cross_attn = self._make_cross_attn(64, 8, 8, torch.bfloat16)
        self.assertEqual(type(cross_attn.attn).__name__, "TrtllmAttention")

    def test_cross_attention_context_runs(self):
        """Context phase: project K/V from encoder, write to cross pool, run FMHA.

        ``head_dim`` is constrained to ``{32, 64, 72, 128, 256}`` by the
        cross-attention KV-cache-update kernel (see
        ``invokeUpdateKvCacheForCrossAttention`` in
        ``cpp/tensorrt_llm/kernels/unfusedAttentionKernels``); we pick 64.
        """
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_heads = 8
        head_dim = 64
        hidden_size = num_heads * head_dim
        decoder_seq_lens = [4]
        encoder_seq_lens = [8]

        cross_attn = self._make_cross_attn(hidden_size, num_heads, head_dim, dtype).to(device)
        decoder_hs = torch.randn(sum(decoder_seq_lens), hidden_size, device=device, dtype=dtype)
        encoder_hs = torch.randn(sum(encoder_seq_lens), hidden_size, device=device, dtype=dtype)

        metadata, cross_metadata, kv_managers = _build_trtllm_cross_metadata(
            decoder_seq_lens,
            encoder_seq_lens,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            kv_cache_manager_cls=self.kv_cache_manager_cls,
        )

        try:
            with torch.inference_mode():
                output = cross_attn(
                    hidden_states=decoder_hs,
                    encoder_hidden_states=encoder_hs,
                    attn_metadata=metadata,
                    cross_attn_metadata=cross_metadata,
                    skip_cross_kv_projection=False,
                )
        finally:
            for mgr in kv_managers:
                mgr.shutdown()

        self.assertEqual(output.shape, (sum(decoder_seq_lens), hidden_size))
        self.assertTrue(
            torch.isfinite(output).all(), "TRTLLM cross-attn output has non-finite values"
        )

    def test_cross_attention_context_matches_vanilla_reference(self):
        """Context phase matches the VANILLA reference within a tight BF16 band."""
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_heads = 2
        head_dim = 64
        hidden_size = num_heads * head_dim
        # Cross-attention should support asymmetric Q/KV lengths. Keep the
        # decoder and encoder lengths intentionally different across requests.
        decoder_seq_lens = [4, 3]
        encoder_seq_lens = [8, 5]

        trtllm_cross_attn, vanilla_cross_attn = self._make_cross_attn_pair(
            hidden_size,
            num_heads,
            head_dim,
            dtype,
            device,
        )
        decoder_hs = torch.randn(sum(decoder_seq_lens), hidden_size, device=device, dtype=dtype)
        encoder_hs = torch.randn(sum(encoder_seq_lens), hidden_size, device=device, dtype=dtype)
        vanilla_metadata, vanilla_cross_metadata = self._make_vanilla_cross_metadata(
            decoder_seq_lens, encoder_seq_lens, device
        )
        trtllm_metadata, trtllm_cross_metadata, kv_managers = _build_trtllm_cross_metadata(
            decoder_seq_lens,
            encoder_seq_lens,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            kv_cache_manager_cls=self.kv_cache_manager_cls,
        )

        try:
            with torch.inference_mode():
                trtllm_output = trtllm_cross_attn(
                    hidden_states=decoder_hs,
                    encoder_hidden_states=encoder_hs,
                    attn_metadata=trtllm_metadata,
                    cross_attn_metadata=trtllm_cross_metadata,
                    skip_cross_kv_projection=False,
                )
                vanilla_output = vanilla_cross_attn(
                    hidden_states=decoder_hs,
                    encoder_hidden_states=encoder_hs,
                    attn_metadata=vanilla_metadata,
                    cross_attn_metadata=vanilla_cross_metadata,
                    skip_cross_kv_projection=False,
                )
        finally:
            for mgr in kv_managers:
                mgr.shutdown()

        # Tolerances cover both trtllm-gen on Blackwell and legacy
        # ``thop.attention`` FMHA on Hopper / Ampere / earlier. The two paths
        # produce numerically equivalent cross-attention outputs within a
        # BF16-friendly band; we observed up to ``mean_abs ≈ 0.017`` and
        # ``max_abs ≈ 0.06`` on H100 vs the VANILLA SDPA reference, so set
        # tolerances slightly above to keep the test as a real correctness
        # gate against bugs while accommodating fused-kernel float noise.
        self._assert_matches_vanilla_reference(
            trtllm_output,
            vanilla_output,
            max_abs_tol=0.10,
            mean_abs_tol=0.025,
        )

    def test_cross_attention_generation_matches_vanilla_reference(self):
        """Generation matches VANILLA when reading encoder K/V from cache."""
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_heads = 2
        head_dim = 64
        hidden_size = num_heads * head_dim
        context_decoder_seq_lens = [4, 3]
        generation_decoder_seq_lens = [1, 1]
        encoder_seq_lens = [8, 5]

        trtllm_cross_attn, vanilla_cross_attn = self._make_cross_attn_pair(
            hidden_size,
            num_heads,
            head_dim,
            dtype,
            device,
        )
        context_decoder_hs = torch.randn(
            sum(context_decoder_seq_lens), hidden_size, device=device, dtype=dtype
        )
        generation_decoder_hs = torch.randn(
            sum(generation_decoder_seq_lens), hidden_size, device=device, dtype=dtype
        )
        encoder_hs = torch.randn(sum(encoder_seq_lens), hidden_size, device=device, dtype=dtype)
        vanilla_metadata, vanilla_cross_metadata = self._make_vanilla_cross_metadata(
            generation_decoder_seq_lens, encoder_seq_lens, device
        )
        context_metadata, context_cross_metadata, kv_managers = _build_trtllm_cross_metadata(
            context_decoder_seq_lens,
            encoder_seq_lens,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            kv_cache_manager_cls=self.kv_cache_manager_cls,
        )

        try:
            with torch.inference_mode():
                _ = trtllm_cross_attn(
                    hidden_states=context_decoder_hs,
                    encoder_hidden_states=encoder_hs,
                    attn_metadata=context_metadata,
                    cross_attn_metadata=context_cross_metadata,
                    skip_cross_kv_projection=False,
                )

            generation_metadata, generation_cross_metadata, _ = _build_trtllm_cross_metadata(
                generation_decoder_seq_lens,
                encoder_seq_lens,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                skip_cross_kv_projection=True,
                kv_managers=kv_managers,
            )

            with torch.inference_mode():
                trtllm_output = trtllm_cross_attn(
                    hidden_states=generation_decoder_hs,
                    encoder_hidden_states=None,
                    attn_metadata=generation_metadata,
                    cross_attn_metadata=generation_cross_metadata,
                    skip_cross_kv_projection=True,
                )
                vanilla_output = vanilla_cross_attn(
                    hidden_states=generation_decoder_hs,
                    encoder_hidden_states=encoder_hs,
                    attn_metadata=vanilla_metadata,
                    cross_attn_metadata=vanilla_cross_metadata,
                    skip_cross_kv_projection=False,
                )
        finally:
            for mgr in kv_managers:
                mgr.shutdown()

        # See note above ``test_cross_attention_context_matches_vanilla_reference``
        # on tolerances. Generation goes through the masked-FMHA decoder
        # path; observed deltas vs VANILLA on H100 stayed below
        # ``max_abs ≈ 0.063`` / ``mean_abs ≈ 0.017``.
        self._assert_matches_vanilla_reference(
            trtllm_output,
            vanilla_output,
            max_abs_tol=0.10,
            mean_abs_tol=0.025,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCrossAttentionTrtllmBackendLegacy(TestCrossAttentionTrtllmBackend):
    """Validate cross-attention through the legacy ``thop.attention`` path.

    The wrapper in ``trtllm.py`` prefers the trtllm-gen sub-path whenever
    ``trtllm_gen.is_supported(...)`` returns ``True`` (which it does on
    Blackwell), so on a B200 dev host the inherited tests above only exercise
    the trtllm-gen sub-path. To run the legacy C++ plumbing
    (``cross_attention`` / ``cross_kv`` / ``encoder_input_lengths`` in
    ``cpp/tensorrt_llm/thop/attentionOp.cpp`` + nanobind binding), we force
    ``trtllm_gen.is_supported`` to return ``False`` for the duration of each
    test, which steers ``TrtllmAttention._run()`` into the ``else: thop.attention(...)``
    branch on every architecture, including Blackwell. The same inherited
    ``CrossAttention`` forward calls + numerical comparisons against the
    VANILLA reference therefore re-run on the legacy compute path.
    """

    def setUp(self):
        super().setUp()
        # Local import to avoid pulling ``unittest.mock`` into the module scope
        # for the (much larger) set of unrelated tests in this file.
        from unittest.mock import patch

        from tensorrt_llm._torch.attention_backend import trtllm as trtllm_backend

        patcher = patch.object(
            trtllm_backend.trtllm_gen,
            "is_supported",
            return_value=(False, "forced legacy thop.attention path for testing"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_attn_backend_selection(self):
        """Backend selection is independent of the trtllm-gen vs legacy split."""
        super().test_attn_backend_selection()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCrossAttentionTrtllmBackendV1(TestCrossAttentionTrtllmBackend):
    """Re-run the dual-pool cross-attention suite on V1 ``KVCacheManager``.

    V1 is the **default and production target** for encoder-decoder
    deployments (``KvCacheConfig.use_kv_cache_manager_v2=False``); V2 is
    an additive secondary path validated by the base class.  Subclassing
    ``TestCrossAttentionTrtllmBackend`` re-runs the same context /
    generation correctness cases against the V1 dual-pool stack.
    """

    @classmethod
    def setUpClass(cls):
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

        cls.kv_cache_manager_cls = KVCacheManager


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCrossAttentionTrtllmBackendV1Legacy(TestCrossAttentionTrtllmBackendLegacy):
    """Re-run the legacy ``thop.attention`` sub-path on V1 ``KVCacheManager``.

    Doubles the V1 production-lane coverage by also forcing the legacy
    ``thop.attention`` sub-path so that Hopper / Ampere / earlier
    deployments (which never hit the trtllm-gen sub-path) are exercised
    against the V1 dual-pool stack.
    """

    @classmethod
    def setUpClass(cls):
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

        cls.kv_cache_manager_cls = KVCacheManager


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCrossAttentionDualPoolSmokeBenchmark(unittest.TestCase):
    """Smoke benchmark for V1 dual-pool cross-attention.

    Times one decoder context cross-attention call followed by one
    decoder generation cross-attention call against the V1 dual-pool
    stack and prints wall-clock latency + tokens/s. Asserts only loose
    upper bounds so the test acts as a smoke gate (it should not flake
    on CI noise) while still exposing pathological regressions in the
    V1 production lane.

    For sustained throughput / TTFT / TPOT measurements, use ``trtllm-bench``.
    This bench only validates that the V1 dual-pool model + backend + cache
    stack boots and runs at a sensible order of magnitude.
    """

    def setUp(self):
        torch.random.manual_seed(42)

    def test_v1_dual_pool_cross_attention_smoke(self):
        from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_heads = 8
        head_dim = 64
        hidden_size = num_heads * head_dim
        decoder_seq_lens = [4, 4, 4, 4]
        encoder_seq_lens = [16, 16, 16, 16]
        num_warmup = 2
        num_iters = 5

        cross_attn = (
            TestCrossAttentionTrtllmBackend()
            ._make_cross_attn(hidden_size, num_heads, head_dim, dtype)
            .to(device)
        )
        decoder_hs = torch.randn(sum(decoder_seq_lens), hidden_size, device=device, dtype=dtype)
        encoder_hs = torch.randn(sum(encoder_seq_lens), hidden_size, device=device, dtype=dtype)

        # Build the V1 dual-pool metadata once for context, reuse the same
        # SELF/CROSS managers across iterations to mimic steady state.
        context_metadata, context_cross_metadata, kv_managers = _build_trtllm_cross_metadata(
            decoder_seq_lens,
            encoder_seq_lens,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            kv_cache_manager_cls=KVCacheManager,
        )

        try:
            # Warmup
            for _ in range(num_warmup):
                with torch.inference_mode():
                    cross_attn(
                        hidden_states=decoder_hs,
                        encoder_hidden_states=encoder_hs,
                        attn_metadata=context_metadata,
                        cross_attn_metadata=context_cross_metadata,
                        skip_cross_kv_projection=False,
                    )
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iters):
                with torch.inference_mode():
                    cross_attn(
                        hidden_states=decoder_hs,
                        encoder_hidden_states=encoder_hs,
                        attn_metadata=context_metadata,
                        cross_attn_metadata=context_cross_metadata,
                        skip_cross_kv_projection=False,
                    )
            end.record()
            torch.cuda.synchronize()
            ms_per_iter = start.elapsed_time(end) / num_iters
        finally:
            for mgr in kv_managers:
                mgr.shutdown()

        total_decoder_tokens = sum(decoder_seq_lens)
        tokens_per_sec = total_decoder_tokens * 1000.0 / max(ms_per_iter, 1e-6)
        print(
            f"\n[V1 dual-pool cross-attn smoke] "
            f"decoder_tokens={total_decoder_tokens} encoder_tokens={sum(encoder_seq_lens)} "
            f"ms/iter={ms_per_iter:.3f} tokens/s={tokens_per_sec:.1f}",
            flush=True,
        )

        # Loose smoke bounds: 100 ms/iter is generous enough to absorb
        # CI jitter and small-shape kernel-launch overhead while still
        # catching catastrophic regressions (e.g. accidental fall-through
        # to a CPU reference path).
        self.assertLess(
            ms_per_iter,
            100.0,
            f"V1 dual-pool cross-attn smoke is suspiciously slow ({ms_per_iter:.2f} ms/iter)",
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestT5Modules(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        self.model_config = ModelConfig(
            pretrained_config=self.hf_config,
            attn_backend="VANILLA",
        )

    def test_t5_encoder_layer_forward(self):
        """Single T5 encoder layer produces correct output shape."""
        layer = T5EncoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_tokens = 6
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens], self.device)

        output = layer(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_t5_decoder_layer_forward(self):
        """Single T5 decoder layer with cross-attention produces correct shape."""
        layer = T5DecoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_dec = 4
        num_enc = 8
        decoder_hs = torch.randn(
            num_dec, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        encoder_hs = torch.randn(
            num_enc, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_dec], self.device)

        output = layer(
            position_ids=torch.arange(num_dec, device=self.device),
            hidden_states=decoder_hs,
            attn_metadata=metadata,
            encoder_hidden_states=encoder_hs,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (num_dec, self.hf_config.d_model))

    def test_t5_encoder_stack_forward(self):
        """T5 encoder stack runs all layers and applies final norm."""
        encoder = T5Encoder(self.model_config).to(self.device)
        num_tokens = 10
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens], self.device)

        output = encoder(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_t5_model_forward(self):
        """T5Model encoder-decoder body runs end-to-end with encoder_input_ids."""
        model = T5Model(self.model_config).to(self.device)
        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, self.hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, self.hf_config.vocab_size, (dec_len,), device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        dec_metadata = _make_vanilla_metadata([dec_len], self.device)

        output = model(
            attn_metadata=dec_metadata,
            input_ids=decoder_ids,
            encoder_input_ids=encoder_ids,
            encoder_attn_metadata=enc_metadata,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (dec_len, self.hf_config.d_model))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBartModules(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.hf_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        self.model_config = ModelConfig(
            pretrained_config=self.hf_config,
            attn_backend="VANILLA",
        )

    def test_bart_encoder_layer_forward(self):
        """Single BART encoder layer produces correct output shape."""
        layer = BartEncoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_tokens = 6
        hidden_states = torch.randn(
            num_tokens, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_tokens], self.device)

        output = layer(hidden_states=hidden_states, attn_metadata=metadata)
        self.assertEqual(output.shape, (num_tokens, self.hf_config.d_model))

    def test_bart_decoder_layer_forward(self):
        """Single BART decoder layer with cross-attention produces correct shape."""
        layer = BartDecoderLayer(self.model_config, layer_idx=0).to(self.device)
        num_dec = 4
        num_enc = 8
        decoder_hs = torch.randn(
            num_dec, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        encoder_hs = torch.randn(
            num_enc, self.hf_config.d_model, device=self.device, dtype=self.dtype
        )
        metadata = _make_vanilla_metadata([num_dec], self.device)

        output = layer(
            position_ids=torch.arange(num_dec, device=self.device),
            hidden_states=decoder_hs,
            attn_metadata=metadata,
            encoder_hidden_states=encoder_hs,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (num_dec, self.hf_config.d_model))

    def test_bart_model_forward(self):
        """BartModel encoder-decoder body runs end-to-end."""
        model = BartModel(self.model_config).to(self.device)
        self.assertEqual(model.position_id_offset, 2)
        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, self.hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, self.hf_config.vocab_size, (dec_len,), device=self.device)
        # BART position IDs start at offset 2 (padding_idx + 1) per HF convention.
        # Use the same offset here so the test exercises valid embedding indices.
        offset = 2
        enc_positions = torch.arange(offset, offset + enc_len, device=self.device)
        dec_positions = torch.arange(offset, offset + dec_len, device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        dec_metadata = _make_vanilla_metadata([dec_len], self.device)

        output = model(
            attn_metadata=dec_metadata,
            input_ids=decoder_ids,
            encoder_input_ids=encoder_ids,
            encoder_position_ids=enc_positions,
            position_ids=dec_positions,
            encoder_attn_metadata=enc_metadata,
            skip_cross_kv_projection=False,
        )
        self.assertEqual(output.shape, (dec_len, self.hf_config.d_model))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestModelRegistration(unittest.TestCase):
    def test_t5_registered(self):
        """T5ForConditionalGeneration is discoverable via MODEL_CLASS_MAPPING."""
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("T5ForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_bart_registered(self):
        """BartForConditionalGeneration is discoverable."""
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("BartForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_mbart_registered(self):
        """MBartForConditionalGeneration is discoverable."""
        from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

        self.assertIn("MBartForConditionalGeneration", MODEL_CLASS_MAPPING)

    def test_model_config_enc_dec_flag(self):
        """ModelConfig.is_encoder_decoder is True for T5/BART configs."""
        t5_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        mc = ModelConfig(pretrained_config=t5_config)
        self.assertTrue(mc.is_encoder_decoder)

        bart_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        mc = ModelConfig(pretrained_config=bart_config)
        self.assertTrue(mc.is_encoder_decoder)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestT5WeightLoading(unittest.TestCase):
    """Verify T5 HF weights load into TRT-LLM and produce matching outputs."""

    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_t5_load_weights_and_encoder_parity(self):
        """Load HF T5 weights and verify encoder output matches HF exactly.

        This tests that the relative position bias is correctly loaded and
        applied, giving numerical parity on the encoder side (self-attention
        only, no cross-attention complications).
        """
        import transformers

        hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        hf_model = transformers.T5ForConditionalGeneration(hf_config).to(self.device).to(self.dtype)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration as TllmT5

        tllm_model = TllmT5(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)
        tllm_model.eval()

        enc_len = 8
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(
                input_ids=encoder_ids,
            ).last_hidden_state.squeeze(0)

        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0))

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
            )

        hf_flat = hf_enc_out.to(self.dtype)
        tllm_flat = tllm_enc_out.to(self.dtype)
        max_diff = (hf_flat - tllm_flat).abs().max().item()
        self.assertLess(max_diff, 1e-3, f"T5 encoder output mismatch: max_diff={max_diff}")

    def test_t5_load_weights_runs_forward(self):
        """Load HF T5 weights into TRT-LLM T5 and verify forward succeeds.

        Full decoder-side parity requires a cross-attention-capable attention
        backend.  This test verifies that weight loading succeeds and the model
        produces finite outputs with the correct shape.
        """
        import transformers

        hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        hf_model = transformers.T5ForConditionalGeneration(hf_config)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration as TllmT5

        tllm_model = TllmT5(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)
        tllm_model.eval()

        enc_len = 8
        dec_len = 4
        encoder_ids = torch.randint(0, hf_config.vocab_size, (enc_len,), device=self.device)
        decoder_ids = torch.randint(0, hf_config.vocab_size, (dec_len,), device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        dec_metadata = _make_vanilla_metadata([dec_len], self.device)

        with torch.inference_mode():
            tllm_out = tllm_model(
                attn_metadata=dec_metadata,
                input_ids=decoder_ids,
                encoder_input_ids=encoder_ids,
                encoder_attn_metadata=enc_metadata,
                skip_cross_kv_projection=False,
            )

        self.assertEqual(tllm_out.shape[-1], hf_config.vocab_size)
        self.assertTrue(torch.isfinite(tllm_out).all(), "Output contains non-finite values")

    def test_t5_for_conditional_generation_load_weights(self):
        """T5ForConditionalGeneration.load_weights runs without error."""
        import transformers

        hf_config = T5Config.from_dict(deepcopy(SMALL_T5_CONFIG))
        hf_model = transformers.T5ForConditionalGeneration(hf_config)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration

        tllm_model = T5ForConditionalGeneration(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBartWeightLoading(unittest.TestCase):
    """Verify BART HF weights load into TRT-LLM and produce matching outputs."""

    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_bart_load_weights_and_encoder_parity(self):
        """Load HF BART weights and verify encoder output matches HF exactly.

        Full decoder-side numerical parity requires a cross-attention-capable
        attention backend. The VANILLA backend's
        ``no_kv_cache_forward`` path uses ``flash_attn_varlen_func`` with
        identical Q/K sequence lengths, which is incorrect for cross-attention
        where K/V lengths differ from Q. Decoder parity can be tightened once
        that path supports mismatched Q and K/V lengths.

        This test verifies:
        1. All HF weights load successfully.
        2. The encoder path (which doesn't involve cross-attention) produces
           outputs identical to HF.
        """
        import transformers

        hf_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        hf_model = transformers.BartModel(hf_config).to(self.device).to(self.dtype)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_bart import (
            BartForConditionalGeneration as TllmBart,
        )
        from tensorrt_llm._torch.models.modeling_bart import _convert_hf_bart_weights

        tllm_model = TllmBart(model_config).to(self.device)
        tllm_weights = _convert_hf_bart_weights(hf_weights, hf_config)
        loaded_count = 0
        for name, module in tllm_model.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            if name not in tllm_weights:
                continue
            w = tllm_weights[name]
            if hasattr(module, "load_weights"):
                module.load_weights(weights=w)
            else:
                for n, p in module.named_parameters(recurse=False):
                    if n in w[0]:
                        p.data.copy_(w[0][n][:])
            loaded_count += 1

        self.assertGreater(loaded_count, 0, "No weights were loaded")
        tllm_model.eval()

        # Verify encoder output parity (no cross-attention involved)
        enc_len = 8
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        import math

        embed_scale = math.sqrt(hf_config.d_model)
        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(
                inputs_embeds=hf_model.shared(encoder_ids) * embed_scale,
            ).last_hidden_state.squeeze(0)

        offset = 2
        enc_positions = torch.arange(offset, offset + enc_len, device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0)) * embed_scale

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
                position_ids=enc_positions,
            )

        hf_flat = hf_enc_out.to(self.dtype)
        tllm_flat = tllm_enc_out.to(self.dtype)
        max_diff = (hf_flat - tllm_flat).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"BART encoder output mismatch: max_diff={max_diff}")

    def test_bart_for_conditional_generation_load_weights(self):
        """BartForConditionalGeneration.load_weights runs without error."""
        import transformers

        hf_config = BartConfig.from_dict(deepcopy(SMALL_BART_CONFIG))
        hf_model = transformers.BartForConditionalGeneration(hf_config)
        hf_model.eval()
        hf_weights = hf_model.state_dict()

        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_bart import BartForConditionalGeneration

        tllm_model = BartForConditionalGeneration(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)


def _get_llm_models_root():
    """Return the path to the LLM models root directory, or None if unavailable."""
    import os
    from pathlib import Path

    root = Path("/home/scratch.trt_llm_data/llm-models/")
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    return root if root.exists() else None


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestT5SmallRealWeights(unittest.TestCase):
    """Verify T5-small (real pre-trained weights) encoder parity with HF.

    t5-small ships as float32.  The test loads it with torch_dtype=bfloat16
    so that the precision-conversion path (float32 → bfloat16) is exercised,
    mirroring the legacy TRT path's ``convert_weight_to_dtype`` logic.
    """

    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        models_root = _get_llm_models_root()
        if models_root is None:
            self.skipTest("LLM_MODELS_ROOT not found")
        self.model_path = str(models_root / "t5-small")
        import os

        if not os.path.isdir(self.model_path):
            self.skipTest(f"t5-small not found at {self.model_path}")

    def test_t5_small_encoder_parity(self):
        """Load real t5-small (float32) as bfloat16 and verify encoder parity."""
        import transformers

        hf_model = (
            transformers.T5ForConditionalGeneration.from_pretrained(self.model_path)
            .to(self.device)
            .to(self.dtype)
        )
        hf_model.eval()
        hf_config = hf_model.config
        hf_weights = hf_model.state_dict()

        hf_config.torch_dtype = self.dtype
        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_t5 import T5ForConditionalGeneration as TllmT5

        tllm_model = TllmT5(model_config).to(self.device)
        tllm_model.load_weights(hf_weights)
        tllm_model.eval()

        for name, p in tllm_model.named_parameters():
            self.assertEqual(
                p.dtype,
                self.dtype,
                f"Parameter {name} has dtype {p.dtype}, expected {self.dtype}",
            )

        enc_len = 16
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(
                input_ids=encoder_ids,
            ).last_hidden_state.squeeze(0)

        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0))

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
            )

        max_diff = (hf_enc_out - tllm_enc_out).abs().max().item()
        # bf16 accumulates more error than float32 across 6 encoder layers
        self.assertLess(max_diff, 0.05, f"T5-small encoder output mismatch: max_diff={max_diff}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBartLargeCNNRealWeights(unittest.TestCase):
    """Verify bart-large-cnn (real pre-trained weights) encoder parity with HF.

    bart-large-cnn ships as float32.  The test loads it with
    torch_dtype=bfloat16 so that the precision-conversion path
    (float32 → bfloat16) is exercised, mirroring the legacy TRT path's
    ``convert_weight_to_dtype`` logic.
    """

    def setUp(self):
        torch.random.manual_seed(42)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

        models_root = _get_llm_models_root()
        if models_root is None:
            self.skipTest("LLM_MODELS_ROOT not found")
        self.model_path = str(models_root / "bart-large-cnn")
        import os

        if not os.path.isdir(self.model_path):
            self.skipTest(f"bart-large-cnn not found at {self.model_path}")

    def test_bart_large_cnn_encoder_parity(self):
        """Load real bart-large-cnn (float32) as bfloat16 and verify encoder parity."""
        import math

        import transformers

        hf_model = (
            transformers.BartModel.from_pretrained(self.model_path).to(self.device).to(self.dtype)
        )
        hf_model.eval()
        hf_config = hf_model.config
        hf_weights = hf_model.state_dict()

        hf_config.torch_dtype = self.dtype
        model_config = ModelConfig(
            pretrained_config=hf_config,
            attn_backend="VANILLA",
        )
        from tensorrt_llm._torch.models.modeling_bart import (
            BartForConditionalGeneration as TllmBart,
        )
        from tensorrt_llm._torch.models.modeling_bart import _convert_hf_bart_weights

        tllm_model = TllmBart(model_config).to(self.device)
        tllm_weights = _convert_hf_bart_weights(hf_weights, hf_config, dtype=self.dtype)
        for name, module in tllm_model.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue
            if name not in tllm_weights:
                continue
            w = tllm_weights[name]
            if hasattr(module, "load_weights"):
                module.load_weights(weights=w)
            else:
                for n, p in module.named_parameters(recurse=False):
                    if n in w[0]:
                        p.data.copy_(w[0][n][:])
        tllm_model.eval()

        for name, p in tllm_model.named_parameters():
            self.assertEqual(
                p.dtype,
                self.dtype,
                f"Parameter {name} has dtype {p.dtype}, expected {self.dtype}",
            )

        enc_len = 16
        encoder_ids = torch.randint(0, hf_config.vocab_size, (1, enc_len), device=self.device)

        embed_scale = math.sqrt(hf_config.d_model)
        with torch.inference_mode():
            hf_enc_out = hf_model.encoder(
                inputs_embeds=hf_model.shared(encoder_ids) * embed_scale,
            ).last_hidden_state.squeeze(0)

        offset = 2
        enc_positions = torch.arange(offset, offset + enc_len, device=self.device)
        enc_metadata = _make_vanilla_metadata([enc_len], self.device)
        enc_embeds = tllm_model.model.shared_embedding(encoder_ids.squeeze(0)) * embed_scale

        with torch.inference_mode():
            tllm_enc_out = tllm_model.model.encoder(
                hidden_states=enc_embeds,
                attn_metadata=enc_metadata,
                position_ids=enc_positions,
            )

        max_diff = (hf_enc_out - tllm_enc_out).abs().max().item()
        # bf16 accumulates more error than float32 across 12 encoder layers
        self.assertLess(
            max_diff, 0.1, f"BART-large-CNN encoder output mismatch: max_diff={max_diff}"
        )


if __name__ == "__main__":
    unittest.main()
