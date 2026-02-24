from typing import List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from _model_test_utils import GQA
from _torch_test_utils import all_close

# Initialize resources first (KVPagedResourceHandler is used within tests below)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import KVPagedResourceHandler
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.factory import (
    FullModelExportInfo,
    ModelFactory,
    SubModuleExportInfo,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import Stages, TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import InitializeCache, ResizeKVCache
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm.llmapi.llm_args import KvCacheConfig


class DummyFactory(ModelFactory):
    """Dummy factory to pass cache_config_updates for testing."""

    def __init__(self, model, cache_config_updates):
        self._model = model
        self.cache_config_updates = cache_config_updates

    def build_model(self, device: str):
        return self._model.to(device=device)

    def _build_model(self, device: str):
        return

    def _load_checkpoint(self, model, device):
        return

    def get_cache_config_updates(self):
        return self.cache_config_updates

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        return [FullModelExportInfo()]


# Class that uses SDPA directly instead of the regular attention mechanism
class GQAWithSdpaAndEmbedding(GQA):
    """GQA model with embedding layer that uses SDPA directly instead of the regular attention."""

    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
        vocab_size: int = 1000,
    ):
        super().__init__(num_attention_heads, hidden_size, num_key_value_heads)
        # Store the head dimensions explicitly
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.vocab_size = vocab_size

        # Add embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        if self.num_heads != self.num_kv_heads:
            self.num_key_value_groups = self.num_heads // self.num_kv_heads
        else:
            self.num_key_value_groups = None

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with input tokens and optional position ids.
        position_ids parameter added to match expected interface in kvcache.py
        """
        # Embed input_ids: [b, s] -> [b, s, hidden]
        x = self.embed_tokens(input_ids)

        b, s, _ = x.shape

        # Project input to q, k, v representations
        q = self.q_proj(x)  # [b, s, n*h_d]
        k = self.k_proj(x)  # [b, s, n_kv*h_d]
        v = self.v_proj(x)  # [b, s, n_kv*h_d]

        # Reshape to [b, s, n, h_d]
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_kv_heads, self.head_dim)
        v = v.view(b, s, self.num_kv_heads, self.head_dim)

        # Use grouped SDPA in bsnd layout
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=None,
            logit_cap=None,
            layout="bsnd",
        )

        # SDPA output is already in [b, s, n, h_d] format
        # Reshape to [b, s, n*h_d]
        attn_output = attn_output.reshape(b, s, -1)

        # Apply output projection
        return self.o_proj(attn_output)


# TODO (lucaslie): consider rewriting this test with a custom InferenceOptimizer config
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32],
    ids=["float16", "float32"],
)
@pytest.mark.parametrize(
    "attn_backend",
    ["triton", "flashinfer"],
    ids=["triton", "flashinfer"],
)
@pytest.mark.parametrize(
    "gqa_config",
    [
        (16, 1024, 16),  # Regular attention (num_heads = num_kv_heads)
        (16, 1024, 4),  # GQA with 4 kv heads
        (16, 1024, 1),  # MQA with 1 kv head
    ],
    ids=["regular", "gqa", "mqa"],
)
@torch.inference_mode()
def test_sdpa_with_kv_cache(dtype, attn_backend, gqa_config):
    """Test the SDPA transformation with KV cache."""
    # flashinfer doesn't support float32 data type
    if attn_backend == "flashinfer" and dtype == torch.float32:
        pytest.skip("flashinfer doesn't support float32 data type")

    # Unpack the GQA configuration
    num_attention_heads, hidden_size, num_key_value_heads = gqa_config

    # some config
    atol = 1e-3
    rtol = 1e-3
    batch_size, seq_len = 16, 64
    num_reset_steps = 2
    num_random_steps = 4
    max_position_embeddings = 128
    vocab_size = 1000

    # set up sequence+cache objects using CachedSequenceInterface
    # Use tokens_per_block=max_position_embeddings so each sequence fits in 1 page for the test
    kv_cache_config = KvCacheConfig(
        tokens_per_block=max_position_embeddings,
        max_tokens=batch_size * max_position_embeddings,
        free_gpu_memory_fraction=0.0,  # Disable dynamic resizing for test
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_position_embeddings,
        max_batch_size=batch_size,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    # Create the model with embedding layer and SDPA, wrap it in a fake factory
    model = GQAWithSdpaAndEmbedding(
        num_attention_heads,
        hidden_size,
        num_key_value_heads,
        vocab_size=vocab_size,
    ).to(dtype=dtype, device="cuda")

    # Create input token ids and position_ids
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
    position_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")

    # Get the model's regular output
    y_model = model(input_ids, position_ids)  # b, s, d

    # Apply the transformation
    optimizer = InferenceOptimizer(
        DummyFactory(model, cache_config_updates={}),
        {
            "build_model": {
                "stage": "factory",
                "run_per_gm": False,
                "device": "cuda",
                "run_graph_cleanup": False,
                "requires_clean_graph": False,
            },
            "export_to_gm": {
                "stage": "export",
                "strict": False,
                "run_per_gm": False,
                "clone_state_dict": True,
                "run_graph_cleanup": False,
                "requires_clean_graph": False,
            },
            "cleanup_input_constraints": {
                "stage": "post_export",
            },
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": attn_backend,
            },
        },
    )  # type: ignore
    gm = optimizer(cm)

    gm.to("cuda")
    num_caches = cm.initialize_resources()
    print(f"num_caches: {num_caches}")

    # Helper function to call the model with proper sequence nesting
    def _call_and_unnest(x, input_pos):
        # Use nest_sequences to properly set input_ids and automatically update position_ids
        cm.info.nest_sequences(x, input_pos=input_pos)

        # Use the cm.args as is - it already contains the correct position_ids
        y = gm(**cm.named_args)

        # Unnest the output sequences
        return torch.stack(cm.info.unnest_sequences(y))

    # Test 1: Regular inference (all tokens at once)
    cm.info.reset()
    y_no_cache = _call_and_unnest(input_ids, 0)
    assert all_close(y_model, y_no_cache, atol=atol, rtol=rtol)

    # Test 2: Autoregressive inference with KV cache
    cm.info.reset()
    y_with_cache = torch.empty_like(y_model)
    for i_p in range(input_ids.shape[1]):
        # Just pass the current token
        y_with_cache[:, i_p : i_p + 1] = _call_and_unnest(input_ids[:, i_p : i_p + 1], i_p)
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # Test 3: Cache continuation after random tokens
    for i_p in range(
        input_ids.shape[1] - num_reset_steps,
        input_ids.shape[1] - num_reset_steps + num_random_steps,
    ):
        random_tokens = torch.randint(0, vocab_size, (batch_size, 1), device="cuda")
        _call_and_unnest(random_tokens, i_p)

    # Continue inference from previous context
    cm.info.reset()
    for i_p in range(input_ids.shape[1] - num_reset_steps, input_ids.shape[1]):
        y_with_cache[:, i_p : i_p + 1] = _call_and_unnest(input_ids[:, i_p : i_p + 1], i_p)
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # Test 4: Exportability of the transformed model
    exported_gm = torch_export_to_gm(gm, args=(), kwargs=cm.named_args)
    assert exported_gm is not None


# =============================================================================
# Transform Unit Tests for Refactored Pipeline
# =============================================================================


@pytest.fixture
def dummy_cached_interface():
    """Create a CachedSequenceInterface for transform testing."""
    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=256,
        free_gpu_memory_fraction=0.0,
    )
    return CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )


def test_initialize_cache_transform_calls_initialize_resources(dummy_cached_interface):
    """Verify InitializeCache transform calls cm.initialize_resources()."""
    # Create a mock module
    mock_module = MagicMock()

    # Create the transform with a proper config
    transform = InitializeCache(config=TransformConfig(stage=Stages.PATTERN_MATCHER))

    # Add a resource to verify initialize_resources is called
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
        KVPagedResourceHandler,
    )

    dummy_cached_interface.add_resource(
        "kv_cache_0", KVPagedResourceHandler(8, 64, dtype=torch.float16)
    )

    # Mock the factory and shared_config
    mock_factory = MagicMock()
    mock_shared_config = MagicMock()

    # Run the transform
    result_mod, info = transform._apply_to_full_model(
        mock_module, dummy_cached_interface, mock_factory, mock_shared_config
    )

    # Verify caches were initialized
    assert info.skipped is False
    assert info.num_matches >= 1
    assert dummy_cached_interface.kv_cache_manager is not None


def test_resize_kv_cache_transform_skipped_when_not_needed(dummy_cached_interface):
    """Verify ResizeKVCache transform is skipped when resize not needed."""
    dummy_cached_interface.add_resource(
        "kv_cache_0", KVPagedResourceHandler(8, 64, dtype=torch.float16)
    )
    dummy_cached_interface.initialize_resources()

    # Create the transform with a proper config
    transform = ResizeKVCache(config=TransformConfig(stage=Stages.PATTERN_MATCHER))

    # Create a mock module
    mock_module = MagicMock()

    # Mock forward call
    mock_module.side_effect = lambda **kwargs: None

    mock_factory = MagicMock()
    mock_shared_config = MagicMock()

    # Run the transform - should be skipped since free_gpu_memory_fraction=0.0
    result_mod, info = transform._apply_to_full_model(
        mock_module, dummy_cached_interface, mock_factory, mock_shared_config
    )

    # Verify transform was skipped
    assert info.skipped is True


def test_resize_kv_cache_transform_runs_when_needed():
    """Verify ResizeKVCache transform runs when resize is needed."""
    # Create interface with resizing enabled
    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=256,
        free_gpu_memory_fraction=0.5,  # Enable resizing
    )
    cm = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=4,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    cm.add_resource("kv_cache_0", KVPagedResourceHandler(8, 64, dtype=torch.float16))
    cm.initialize_resources()

    # Create the transform with a proper config
    transform = ResizeKVCache(config=TransformConfig(stage=Stages.PATTERN_MATCHER))

    # Create a simple mock module that just returns None
    class MockModule:
        def __call__(self, **kwargs):
            return None

    mock_module = MockModule()
    mock_factory = MagicMock()
    mock_shared_config = MagicMock()

    # Run the transform
    result_mod, info = transform._apply_to_full_model(
        mock_module, cm, mock_factory, mock_shared_config
    )

    # Verify transform was not skipped
    assert info.skipped is False


def test_insert_cached_attention_uses_add_resource():
    """Verify InsertCachedAttention uses cm.add_resource() for cache registration."""
    # This test verifies the integration point between InsertCachedAttention
    # and CachedSequenceInterface.add_resource() by checking that after the
    # transform, resources are registered in the interface.

    num_attention_heads = 8
    hidden_size = 512
    num_key_value_heads = 8
    vocab_size = 1000
    batch_size = 4
    max_seq_len = 64

    kv_cache_config = KvCacheConfig(
        tokens_per_block=max_seq_len,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    # Create a model
    model = GQAWithSdpaAndEmbedding(
        num_attention_heads,
        hidden_size,
        num_key_value_heads,
        vocab_size=vocab_size,
    ).to(dtype=torch.float16, device="cuda")

    # Apply transformation
    optimizer = InferenceOptimizer(
        DummyFactory(model, cache_config_updates={}),
        {
            "build_model": {
                "stage": "factory",
                "run_per_gm": False,
                "device": "cuda",
                "run_graph_cleanup": False,
                "requires_clean_graph": False,
            },
            "export_to_gm": {
                "stage": "export",
                "strict": False,
                "run_per_gm": False,
                "clone_state_dict": True,
                "run_graph_cleanup": False,
                "requires_clean_graph": False,
            },
            "cleanup_input_constraints": {
                "stage": "post_export",
            },
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "triton",
            },
        },
    )

    optimizer(cm)

    # Verify resources were added
    assert len(cm._resource_lookup) > 0
    # Should have k_cache and v_cache resources registered
    resource_names = list(cm._resource_lookup.keys())
    assert any("k_cache" in name for name in resource_names)
    assert any("v_cache" in name for name in resource_names)


def test_insert_cached_attention_passes_kv_cache_config():
    """Verify InsertCachedAttention passes cm.kv_cache_config to get_cache_initializers."""
    # This test verifies that the KvCacheConfig from the interface is used
    # when initializing cache resources (e.g., for dtype configuration).

    num_attention_heads = 8
    hidden_size = 512
    num_key_value_heads = 8
    vocab_size = 1000
    batch_size = 4
    max_seq_len = 64

    # Use specific dtype in kv_cache_config
    kv_cache_config = KvCacheConfig(
        tokens_per_block=max_seq_len,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
        dtype="bfloat16",  # Specify explicit dtype
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    # Verify kv_cache_config is accessible
    assert cm.kv_cache_config.dtype == "bfloat16"

    # Create a model
    model = GQAWithSdpaAndEmbedding(
        num_attention_heads,
        hidden_size,
        num_key_value_heads,
        vocab_size=vocab_size,
    ).to(dtype=torch.bfloat16, device="cuda")

    # Apply transformation
    optimizer = InferenceOptimizer(
        DummyFactory(model, cache_config_updates={}),
        {
            "build_model": {
                "stage": "factory",
                "run_per_gm": False,
                "device": "cuda",
                "run_graph_cleanup": False,
                "requires_clean_graph": False,
            },
            "export_to_gm": {
                "stage": "export",
                "strict": False,
                "run_per_gm": False,
                "clone_state_dict": True,
                "run_graph_cleanup": False,
                "requires_clean_graph": False,
            },
            "cleanup_input_constraints": {
                "stage": "post_export",
            },
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "triton",
            },
        },
    )

    optimizer(cm)

    # Initialize resources
    cm.initialize_resources()

    assert not any(handler.is_paged for handler in cm._resource_lookup.values()), (
        "triton should not use paged resources"
    )
    assert cm._caches, "at least some resources should be present"

    # Verify cache dtype matches config
    for name, handler in cm._resource_lookup.items():
        if hasattr(handler, "dtype"):
            assert handler.dtype == torch.bfloat16
