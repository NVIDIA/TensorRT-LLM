from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from _model_test_utils import GQA, default_max_num_tokens
from _torch_test_utils import all_close

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig

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
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
    InitializeCache,
    InsertCachedMLAAttention,
    ResizeKVCache,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


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

    @property
    def max_seq_len(self) -> int:
        return 512


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
        """Forward pass with input tokens and optional position ids.

        position_ids parameter added to match expected interface in kvcache.py.
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


class SpanMaskedAttentionModel(torch.nn.Module):
    """Minimal model whose source graph uses a semantic multimodal mask op."""

    def __init__(self, hidden_size: int = 8, num_heads: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size).to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mm_token_positions: torch.Tensor,
        mm_token_lengths: torch.Tensor,
        mm_item_cu_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        del position_ids
        x = self._embed(input_ids)
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        attn_mask = torch.ops.auto_deploy.gemma4_multimodal_mask.default(
            input_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        )
        return torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            layout="bsnd",
        )


def _create_cpu_seq_info() -> CachedSequenceInterface:
    return CachedSequenceInterface(
        max_seq_len=16,
        max_batch_size=4,
        max_num_tokens=default_max_num_tokens(16, 4),
        device="cpu",
        kv_cache_config=KvCacheConfig(tokens_per_block=16),
    )


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
        max_num_tokens=default_max_num_tokens(max_position_embeddings, batch_size),
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
        # x is [batch_size, seq_len] tensor
        bs, sl = x.shape
        cu_seqlen = list(range(0, bs * sl + 1, sl))
        if isinstance(input_pos, int):
            input_pos = [input_pos] * bs
        cm.info.nest_sequences(
            x.flatten().tolist(),
            cu_seqlen=cu_seqlen,
            input_pos=input_pos,
            cache_loc=list(range(bs)),
            cu_num_pages=list(range(bs + 1)),
            slot_idx=list(range(bs)),
            gather_context_logits=True,
        )

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


@torch.inference_mode()
def test_source_graph_uses_semantic_multimodal_mask_op():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )

    semantic_nodes = [
        node for node in gm.graph.nodes if is_op(node, torch.ops.auto_deploy.gemma4_multimodal_mask)
    ]
    assert len(semantic_nodes) == 1


@torch.inference_mode()
def test_insert_cached_attention_lowers_semantic_mask_for_torch_backend():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )
    cm = _create_cpu_seq_info()
    gm_transformed = InferenceOptimizer(
        None,
        {
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "torch",
            },
        },
    )(cm, gm)

    assert not any(
        is_op(node, torch.ops.auto_deploy.torch_attention) for node in gm_transformed.graph.nodes
    )
    assert any(
        is_op(node, torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)
        for node in gm_transformed.graph.nodes
    )

    cached_nodes = [
        node
        for node in gm_transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_cached_attention_with_cache)
    ]
    assert len(cached_nodes) == 1
    assert is_op(cached_nodes[0].args[-1], torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)


@torch.inference_mode()
def test_insert_cached_attention_lowers_semantic_mask_for_triton_paged_backend():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )
    cm = _create_cpu_seq_info()
    gm_transformed = InferenceOptimizer(
        None,
        {
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "triton_paged",
            },
        },
    )(cm, gm)

    assert any(
        is_op(node, torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)
        for node in gm_transformed.graph.nodes
    )
    cached_nodes = [
        node
        for node in gm_transformed.graph.nodes
        if is_op(node, torch.ops.auto_deploy.triton_paged_mha_with_cache)
    ]
    assert len(cached_nodes) == 1
    assert is_op(cached_nodes[0].args[-1], torch.ops.auto_deploy.gemma4_prepare_multimodal_mask)


@torch.inference_mode()
def test_insert_cached_attention_rejects_unsupported_semantic_mask_backend():
    model = SpanMaskedAttentionModel().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int64)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).repeat(input_ids.shape[0], 1)
    mm_token_positions = torch.tensor([1, 3], dtype=torch.int32)
    mm_token_lengths = torch.tensor([2, 2], dtype=torch.int32)
    mm_item_cu_seqlen = torch.tensor([0, 2], dtype=torch.int32)

    gm = torch_export_to_gm(
        model,
        args=(
            input_ids,
            position_ids,
            mm_token_positions,
            mm_token_lengths,
            mm_item_cu_seqlen,
        ),
        clone=True,
    )
    cm = _create_cpu_seq_info()

    with pytest.raises(
        RuntimeError,
        match=(
            "Cached attention backend 'flashinfer' does not support lowering semantic mask op"
            ".*gemma4_multimodal_mask.*Supported backends: torch, triton_paged"
        ),
    ):
        InferenceOptimizer(
            None,
            {
                "insert_cached_attention": {
                    "stage": "cache_init",
                    "backend": "flashinfer",
                },
            },
        )(cm, gm)


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
        max_num_tokens=default_max_num_tokens(128, 4),
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
        max_num_tokens=default_max_num_tokens(128, 4),
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
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
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
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
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


def _make_fake_mla_node(kv_lora_rank: int, qk_rope_head_dim: int):
    return SimpleNamespace(
        args=[
            None,
            None,
            SimpleNamespace(meta={"val": torch.empty(1, 1, kv_lora_rank)}),
            SimpleNamespace(meta={"val": torch.empty(1, 1, 1, qk_rope_head_dim)}),
        ]
    )


def test_insert_cached_mla_attention_keeps_supported_flashinfer_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))

    attn_node = _make_fake_mla_node(kv_lora_rank=512, qk_rope_head_dim=64)
    backend = InsertCachedMLAAttention.resolve_backend_for_node("flashinfer_mla", attn_node)

    assert backend == "flashinfer_mla"


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
def test_insert_cached_mla_attention_falls_back_for_rank256(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: capability)

    attn_node = _make_fake_mla_node(kv_lora_rank=256, qk_rope_head_dim=64)
    backend = InsertCachedMLAAttention.resolve_backend_for_node("flashinfer_mla", attn_node)

    expected = "flashinfer_trtllm_mla" if capability >= (10, 0) else "torch_mla"
    assert backend == expected


def test_insert_cached_mla_attention_preserves_non_flashinfer_backend():
    attn_node = _make_fake_mla_node(kv_lora_rank=256, qk_rope_head_dim=64)
    backend = InsertCachedMLAAttention.resolve_backend_for_node("torch_mla", attn_node)

    assert backend == "torch_mla"


# =============================================================================
# Sliding Window KV Cache Integration Tests
# =============================================================================


class SlidingWindowGQA(GQAWithSdpaAndEmbedding):
    """GQA model with a configurable per-layer sliding_window for testing."""

    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
        vocab_size: int = 1000,
        sliding_window: Optional[int] = None,
    ):
        super().__init__(num_attention_heads, hidden_size, num_key_value_heads, vocab_size)
        self._sliding_window = sliding_window

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=self._sliding_window,
            logit_cap=None,
            layout="bsnd",
        )
        return self.o_proj(attn_output.reshape(b, s, -1))


class VSWAModel(nn.Module):
    """Model with two attention layers: one sliding-window, one full-attention (VSWA)."""

    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
        vocab_size: int = 1000,
        sliding_window: int = 32,
    ):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.q_proj_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj_0 = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj_0 = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj_0 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_proj_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj_1 = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj_1 = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj_1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self._sliding_window = sliding_window

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        b, s, _ = x.shape

        # Layer 0: sliding window attention
        q0 = self.q_proj_0(x).view(b, s, self.num_heads, self.head_dim)
        k0 = self.k_proj_0(x).view(b, s, self.num_kv_heads, self.head_dim)
        v0 = self.v_proj_0(x).view(b, s, self.num_kv_heads, self.head_dim)
        a0 = torch.ops.auto_deploy.torch_attention(
            q0,
            k0,
            v0,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=self._sliding_window,
            logit_cap=None,
            layout="bsnd",
        )
        x = x + self.o_proj_0(a0.reshape(b, s, -1))

        # Layer 1: full attention (no sliding window)
        q1 = self.q_proj_1(x).view(b, s, self.num_heads, self.head_dim)
        k1 = self.k_proj_1(x).view(b, s, self.num_kv_heads, self.head_dim)
        v1 = self.v_proj_1(x).view(b, s, self.num_kv_heads, self.head_dim)
        a1 = torch.ops.auto_deploy.torch_attention(
            q1,
            k1,
            v1,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=None,
            sinks=None,
            sliding_window=None,
            logit_cap=None,
            layout="bsnd",
        )
        return x + self.o_proj_1(a1.reshape(b, s, -1))


def _build_optimizer_with_backend(model, backend="triton_paged"):
    """Helper to create an InferenceOptimizer for testing."""
    return InferenceOptimizer(
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
                "backend": backend,
            },
        },
    )


@torch.inference_mode()
def test_insert_cached_attention_extracts_sliding_window():
    """Verify insert_cached_attention sets max_attention_window from graph sliding_window."""
    sliding_window = 32
    max_seq_len = 128
    batch_size = 4

    kv_cache_config = KvCacheConfig(
        tokens_per_block=max_seq_len,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    assert cm.kv_cache_config.max_attention_window is None

    model = SlidingWindowGQA(
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        sliding_window=sliding_window,
    ).to(dtype=torch.float16, device="cuda")

    optimizer = _build_optimizer_with_backend(model)
    optimizer(cm)

    assert cm.kv_cache_config.max_attention_window is not None
    assert cm.kv_cache_config.max_attention_window == [sliding_window]


@torch.inference_mode()
def test_insert_cached_attention_no_sliding_window_leaves_config_unchanged():
    """Verify insert_cached_attention does not set max_attention_window for non-SWA models."""
    max_seq_len = 128
    batch_size = 4

    kv_cache_config = KvCacheConfig(
        tokens_per_block=max_seq_len,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    model = GQAWithSdpaAndEmbedding(
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
    ).to(dtype=torch.float16, device="cuda")

    optimizer = _build_optimizer_with_backend(model)
    optimizer(cm)

    assert cm.kv_cache_config.max_attention_window is None


@torch.inference_mode()
def test_insert_cached_attention_respects_user_override():
    """Verify insert_cached_attention does not overwrite user-set max_attention_window."""
    max_seq_len = 128
    batch_size = 4
    user_window = [64]

    kv_cache_config = KvCacheConfig(
        tokens_per_block=max_seq_len,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
        max_attention_window=user_window,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    model = SlidingWindowGQA(
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        sliding_window=32,
    ).to(dtype=torch.float16, device="cuda")

    optimizer = _build_optimizer_with_backend(model)
    optimizer(cm)

    # User-provided value must be preserved
    assert cm.kv_cache_config.max_attention_window == user_window


@torch.inference_mode()
def test_insert_cached_attention_vswa_preserves_per_layer_windows():
    """Verify VSWA model preserves per-layer window sizes for proportional allocation."""
    sliding_window = 32
    max_seq_len = 128
    batch_size = 4

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    model = VSWAModel(
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        sliding_window=sliding_window,
    ).to(dtype=torch.float16, device="cuda")

    optimizer = _build_optimizer_with_backend(model)
    optimizer(cm)

    # VSWA preserves per-layer windows: [32, 128] (not collapsed)
    assert cm.kv_cache_config.max_attention_window is not None
    assert len(cm.kv_cache_config.max_attention_window) == 2
    assert cm.kv_cache_config.max_attention_window == [sliding_window, max_seq_len]

    # Window groups should be registered on SequenceInfo
    assert cm.info.num_window_groups == 2
    assert cm.info.window_groups == [sliding_window, max_seq_len]
    assert cm.info.window_group_map == {sliding_window: 0, max_seq_len: 1}


@torch.inference_mode()
def test_kv_cache_manager_initialized_with_sliding_window():
    """Verify KVCacheManager receives max_attention_window_vec from SWA model.

    Runs the full insert_cached_attention + initialize_cache pipeline.
    """
    import math

    sliding_window = 32
    max_seq_len = 128
    batch_size = 4
    tokens_per_block = 16

    kv_cache_config = KvCacheConfig(
        tokens_per_block=tokens_per_block,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    model = SlidingWindowGQA(
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        sliding_window=sliding_window,
    ).to(dtype=torch.float16, device="cuda")

    # Run insert_cached_attention + initialize_cache
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
                "backend": "triton_paged",
            },
            "initialize_cache": {
                "stage": "cache_init",
                "run_per_gm": False,
            },
        },
    )
    optimizer(cm)

    # KVCacheManager should exist and carry the window vector
    mgr = cm.kv_cache_manager
    assert mgr.max_attention_window_vec == [sliding_window]

    # max_blocks_per_seq is based on max_seq_len (not window) because sequences
    # temporarily need full blocks during prefill; SWA eviction frees them during decode.
    expected_max_blocks = math.ceil(max_seq_len / tokens_per_block)
    assert cm.info.max_blocks_per_seq == expected_max_blocks


@torch.inference_mode()
def test_kv_cache_manager_vswa_proportional_allocation():
    """Verify VSWA models get proportional pool allocation in KVCacheManager.

    KVCacheManager detects VSWA from the per-layer max_attention_window vector
    and allocates separate block pools per window size.
    """
    import math

    sliding_window = 32
    max_seq_len = 128
    batch_size = 4
    tokens_per_block = 16

    kv_cache_config = KvCacheConfig(
        tokens_per_block=tokens_per_block,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    model = VSWAModel(
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        sliding_window=sliding_window,
    ).to(dtype=torch.float16, device="cuda")

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
                "backend": "triton_paged",
            },
            "initialize_cache": {
                "stage": "cache_init",
                "run_per_gm": False,
            },
        },
    )
    optimizer(cm)

    # KVCacheManager should detect VSWA with per-layer windows.  A single C++
    # KVCacheManager now hosts both pools via the per-window head_dim/dtype
    # overrides — there is no MultiPoolKVCacheManager wrapper anymore.
    mgr = cm.kv_cache_manager
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

    assert isinstance(mgr, KVCacheManager)
    # The unified manager exposes one pool per effective window via the C++ impl.
    assert len(mgr.impl.size_per_head_per_window) == 2

    # max_blocks_per_seq reflects the larger window (full-attention layer)
    expected_max_blocks = math.ceil(max_seq_len / tokens_per_block)
    assert cm.info.max_blocks_per_seq == expected_max_blocks

    # Per-group cache tensors should be registered on SequenceInfo
    assert cm.info.num_window_groups == 2
    assert "cache_loc_g1" in cm.info.available_args
    assert "cu_num_pages_g1" in cm.info.available_args


# =============================================================================
# VSWA SequenceInfo and Graph Wiring Tests
# =============================================================================


@torch.inference_mode()
def test_sequence_info_register_window_groups():
    """Verify register_window_groups creates per-group tensors in InputBuffer."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo

    max_seq_len = 128
    batch_size = 4
    tokens_per_block = 16

    seq_info = SequenceInfo(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        tokens_per_block=tokens_per_block,
    )

    # Before registration: no window groups
    assert seq_info.num_window_groups == 0
    assert seq_info.window_groups == []

    # Register two groups: [32, 128]
    seq_info.register_window_groups([32, 128])

    assert seq_info.num_window_groups == 2
    assert seq_info.window_groups == [32, 128]
    assert seq_info.window_group_map == {32: 0, 128: 1}

    # Group 0 reuses existing tensors (cache_loc, cu_num_pages, etc.)
    # Group 1 gets new tensors
    assert "cache_loc_g1" in seq_info.available_args
    assert "cu_num_pages_g1" in seq_info.available_args
    assert "cu_num_pages_g1_host" in seq_info.available_args
    assert "last_page_len_g1" in seq_info.available_args
    assert "last_page_len_g1_host" in seq_info.available_args
    assert "extra_page_per_seq_g1" in seq_info.available_args

    # Group 0 names should NOT appear with _g0 suffix
    assert "cache_loc_g0" not in seq_info.available_args


@torch.inference_mode()
def test_vswa_graph_has_per_group_placeholders():
    """Verify VSWA model graph contains per-group cache_loc/cu_num_pages placeholders."""
    sliding_window = 32
    max_seq_len = 128
    batch_size = 4

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32,
        max_tokens=batch_size * max_seq_len,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        max_num_tokens=default_max_num_tokens(max_seq_len, batch_size),
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    model = VSWAModel(
        num_attention_heads=8,
        hidden_size=512,
        num_key_value_heads=8,
        sliding_window=sliding_window,
    ).to(dtype=torch.float16, device="cuda")

    optimizer = _build_optimizer_with_backend(model)
    gm = optimizer(cm)

    # Check that per-group graph placeholders exist
    placeholder_names = [n.target for n in gm.graph.nodes if n.op == "placeholder"]
    assert "cache_loc" in placeholder_names
    assert "cu_num_pages" in placeholder_names
    # Group 1 should have its own placeholders
    assert "cache_loc_g1" in placeholder_names
    assert "cu_num_pages_g1" in placeholder_names


# =============================================================================
# Phase 3: sink_token_length wiring through get_constants
# =============================================================================


def test_trtllm_get_constants_does_not_include_sink_token_length():
    """Verify TrtllmAttention.get_constants returns node-level constants only.

    sink_token_length is a runtime config (KvCacheConfig), not a model parameter.
    It is appended by the kvcache transform at insertion time, not by get_constants.
    """
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention.trtllm_attention import (
        TrtllmAttention,
    )
    from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
    from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

    model = (
        SlidingWindowGQA(
            num_attention_heads=8,
            hidden_size=512,
            num_key_value_heads=8,
            sliding_window=32,
        )
        .eval()
        .to(dtype=torch.float16, device="cpu")
    )

    input_ids = torch.randint(0, 1000, (1, 4))
    gm = torch_export_to_gm(model, (input_ids,))

    source_op = TrtllmAttention.get_source_attention_op()
    source_nodes = [n for n in gm.graph.nodes if is_op(n, source_op)]
    assert len(source_nodes) == 1

    constants = TrtllmAttention.get_constants(source_nodes[0])
    # Last constant is None (out placeholder); sink_token_length is NOT here
    # (it's appended at the transform site from KvCacheConfig, not by get_constants)
    assert constants[-1] is None
    assert len(constants) == 6  # scale, sw, kv_so, kv_qs, out_scale, out
