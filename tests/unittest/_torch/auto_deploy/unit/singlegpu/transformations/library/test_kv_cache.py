from typing import Optional

import pytest
import torch
from _graph_test_helpers import FakeFactory
from _model_test_utils import GQA
from _torch_test_utils import all_close

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import CacheConfig, SequenceInfo
from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_attention import FlashInferAttention
from tensorrt_llm._torch.auto_deploy.custom_ops.triton_attention import TritonAttention
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import InferenceOptimizerConfig
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.transformations.library import update_in_out_nodes
from tensorrt_llm._torch.auto_deploy.transformations.library.kvcache import insert_cached_attention


# Class that uses SDPA directly instead of the regular attention mechanism
class GQAWithSdpa(GQA):
    """GQA model that uses SDPA directly instead of the regular attention."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Store the head dimensions explicitly
        self.num_heads = args[0]  # First argument is num_attention_heads
        self.num_kv_heads = args[2]  # Third argument is num_key_value_heads
        self.head_dim = args[1] // self.num_heads  # hidden_size / num_heads

        if self.num_heads != self.num_kv_heads:
            self.num_key_value_groups = self.num_heads // self.num_kv_heads
        else:
            self.num_key_value_groups = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with input tokens and optional position ids.
        position_ids parameter added to match expected interface in kvcache.py
        """
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
        attn_output = torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa(
            q, k, v, None, 0.0, True, None
        )

        # SDPA output is already in [b, s, n, h_d] format
        # Reshape to [b, s, n*h_d]
        attn_output = attn_output.reshape(b, s, -1)

        # Apply output projection
        return self.o_proj(attn_output)


def _get_optimizer_config() -> InferenceOptimizerConfig:
    return {
        "build_model": {
            "stage": "factory",
            "device": "cuda",
            "run_graph_cleanup": False,
            "requires_clean_graph": False,
        },
        "export_to_gm": {
            "stage": "export",
            "strict": False,
            "clone_state_dict": True,
            "run_graph_cleanup": False,
            "requires_clean_graph": False,
        },
        "cleanup_input_constraints": {
            "stage": "post_export",
        },
    }


class SequenceEmbeddingInfo(SequenceInfo):
    hidden_size: int
    dtype: torch.dtype

    def set_example_sequence(self) -> None:
        super().set_example_sequence()
        # set input ids to a 3D tensor (actually input embeddings)
        self.input_ids = torch.rand(
            *self.input_ids.shape,
            self.hidden_size,
            device=self.input_ids.device,
            dtype=self.dtype,
        )


# TODO (lucaslie): consider rewriting this test with a custom InferenceOptimizer config
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32],
    ids=["float16", "float32"],
)
@pytest.mark.parametrize(
    "attn_descriptor",
    [TritonAttention, FlashInferAttention],
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
def test_sdpa_with_kv_cache(dtype, attn_descriptor, gqa_config):
    """Test the SDPA transformation with KV cache."""
    # flashinfer doesn't support float32 data type
    if attn_descriptor == FlashInferAttention and dtype == torch.float32:
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

    # set up sequence+cache objects
    ci = SequenceEmbeddingInfo(
        max_seq_len=max_position_embeddings,
        max_batch_size=batch_size,
    )
    ci.hidden_size = hidden_size
    ci.dtype = dtype
    cm = CachedSequenceInterface(sequence_info=ci, device="cuda")

    # Create the model with SDPA and wrap it in a fake factory
    model = GQAWithSdpa(
        num_attention_heads,
        hidden_size,
        num_key_value_heads,
    ).to(dtype=dtype, device="cuda")
    factory = FakeFactory(model)

    # Create input tensor and position_ids
    x = torch.rand(batch_size, seq_len, hidden_size).to(device="cuda", dtype=dtype)
    position_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")

    # Get the model's regular output
    y_model = model(x, position_ids)  # b, s, d

    # run modular inference optimizer up to post_export
    optimizer = InferenceOptimizer(factory, _get_optimizer_config())  # type: ignore
    gm = optimizer(cm)

    y_gm = gm(x, position_ids)
    assert all_close(y_model, y_gm, atol=atol, rtol=rtol)

    # Set up cache configuration
    cache_config = CacheConfig()

    # Get input node(s)
    update_in_out_nodes(gm, cm)

    # Apply the transformation
    insert_cached_attention(gm, cm, attn_descriptor=attn_descriptor, cache_config=cache_config)
    gm.to("cuda")
    cm.initialize_caches()

    # Helper function to call the model with proper sequence nesting
    def _call_and_unnest(x):
        # Use nest_sequences to properly set input_ids and automatically update position_ids
        cm.info.nest_sequences(x, allow_realloc=True)

        # Use the cm.args as is - it already contains the correct position_ids
        y = gm(*cm.args)

        # Unnest the output sequences
        return torch.stack(cm.info.unnest_sequences(y))

    # Test 1: Regular inference (all tokens at once)
    cm.info.reset()
    y_no_cache = _call_and_unnest(x)
    assert all_close(y_model, y_no_cache, atol=atol, rtol=rtol)

    # Test 2: Autoregressive inference with KV cache
    cm.info.reset()
    y_with_cache = torch.empty_like(y_model)
    for i in range(x.shape[1]):
        # Just pass the current token
        y_with_cache[:, i : i + 1] = _call_and_unnest(x[:, i : i + 1])
        # Update position for next token
        cm.info.update_pos(1)  # This automatically updates position_ids too
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # Test 3: Cache continuation after random tokens
    cm.info.update_pos(-num_reset_steps)  # Rewind position
    for i in range(num_random_steps):
        _call_and_unnest(torch.rand_like(x[:, :1]))
        cm.info.update_pos(1)

    # Continue inference from previous context
    cm.info.reset()
    cm.info.update_pos(x.shape[1] - num_reset_steps)
    for i in range(x.shape[1] - num_reset_steps, x.shape[1]):
        y_with_cache[:, i : i + 1] = _call_and_unnest(x[:, i : i + 1])
        cm.info.update_pos(1)
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # Test 4: Exportability of the transformed model
    exported_gm = torch_export_to_gm(gm, args=cm.args)
    assert exported_gm is not None
