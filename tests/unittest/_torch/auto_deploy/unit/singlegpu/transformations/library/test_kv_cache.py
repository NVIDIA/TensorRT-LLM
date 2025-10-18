from typing import List, Optional

import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import SequenceEmbeddingInfo
from _model_test_utils import GQA
from _torch_test_utils import all_close

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import CacheConfig
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.factory import (
    FullModelExportInfo,
    ModelFactory,
    SubModuleExportInfo,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer


class DummyFactory(ModelFactory):
    """Dummy factory to pass cache_config for testing."""

    def __init__(self, model, cache_config):
        self._model = model
        self.cache_config = cache_config

    def build_model(self, device: str):
        return self._model.to(device=device)

    def _build_model(self, device: str):
        return

    def _load_checkpoint(self, model, device):
        return

    def get_cache_config(self):
        return self.cache_config

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        return [FullModelExportInfo()]


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
    def forward(
        self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with input tokens and optional position ids.
        position_ids parameter added to match expected interface in kvcache.py
        """
        b, s, _ = input_ids.shape

        # Project input to q, k, v representations
        q = self.q_proj(input_ids)  # [b, s, n*h_d]
        k = self.k_proj(input_ids)  # [b, s, n_kv*h_d]
        v = self.v_proj(input_ids)  # [b, s, n_kv*h_d]

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

    # set up sequence+cache objects
    ci = SequenceEmbeddingInfo(
        max_seq_len=max_position_embeddings,
        max_batch_size=batch_size,
        hidden_size=hidden_size,
        dtype=dtype,
    )
    cm = CachedSequenceInterface(sequence_info=ci, device="cuda")

    # Create the model with SDPA and wrap it in a fake factory
    model = GQAWithSdpa(
        num_attention_heads,
        hidden_size,
        num_key_value_heads,
    ).to(dtype=dtype, device="cuda")

    # Create input tensor and position_ids
    x = torch.rand(batch_size, seq_len, hidden_size).to(device="cuda", dtype=dtype)
    position_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")

    # Get the model's regular output
    y_model = model(x, position_ids)  # b, s, d

    # Apply the transformation
    optimizer = InferenceOptimizer(
        DummyFactory(model, CacheConfig()),
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
            "update_in_out_nodes": {
                "stage": "cache_init",
            },
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": attn_backend,
            },
        },
    )  # type: ignore
    gm = optimizer(cm)

    gm.to("cuda")
    num_caches = cm.initialize_caches()
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
    y_no_cache = _call_and_unnest(x, 0)
    assert all_close(y_model, y_no_cache, atol=atol, rtol=rtol)

    # Test 2: Autoregressive inference with KV cache
    cm.info.reset()
    y_with_cache = torch.empty_like(y_model)
    for i_p in range(x.shape[1]):
        # Just pass the current token
        y_with_cache[:, i_p : i_p + 1] = _call_and_unnest(x[:, i_p : i_p + 1], i_p)
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # Test 3: Cache continuation after random tokens
    for i_p in range(x.shape[1] - num_reset_steps, x.shape[1] - num_reset_steps + num_random_steps):
        _call_and_unnest(torch.rand_like(x[:, :1]), i_p)

    # Continue inference from previous context
    cm.info.reset()
    for i_p in range(x.shape[1] - num_reset_steps, x.shape[1]):
        y_with_cache[:, i_p : i_p + 1] = _call_and_unnest(x[:, i_p : i_p + 1], i_p)
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # Test 4: Exportability of the transformed model
    exported_gm = torch_export_to_gm(gm, args=(), kwargs=cm.named_args)
    assert exported_gm is not None
