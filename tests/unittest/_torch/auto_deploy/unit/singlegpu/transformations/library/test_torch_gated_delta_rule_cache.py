"""Transform unit test for insert_cached_gated_delta_rule with the torch_gated_delta backend.

Verifies the full pipeline:
  1. Build a mock model that calls ``torch_gated_delta_rule`` internally.
  2. Export to GraphModule.
  3. Apply the ``insert_cached_gated_delta_rule`` transform.
  4. Test full-sequence prefill matches uncached model output.
  5. Test token-by-token autoregressive matches uncached model output.

The stale-cache fix zeroes delta_cache tensors between phases because
``cm.info.reset()`` only clears sequence metadata, not physical cache data.
"""

from typing import List, Optional

import pytest
import torch
import torch.nn as nn
from _torch_test_utils import all_close

# Register all auto_deploy custom ops
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.factory import (
    FullModelExportInfo,
    ModelFactory,
    SubModuleExportInfo,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm.llmapi.llm_args import KvCacheConfig

# ---------------------------------------------------------------------------
# Dummy factory (same pattern as test_kv_cache.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Mock model that uses torch_gated_delta_rule
# ---------------------------------------------------------------------------


class GatedDeltaRuleModel(nn.Module):
    """Minimal model that projects embeddings through torch_gated_delta_rule.

    Supports GVA: q/k use num_k_heads, v/g/beta use num_v_heads.
    L2 normalization and repeat_interleave are handled inside the op.

    Architecture:
      input_ids -> embedding -> linear projections -> torch_gated_delta_rule -> output proj
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        key_dim: int,
        value_dim: int,
    ):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, num_k_heads * key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_k_heads * key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_v_heads * value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.beta_proj = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.o_proj = nn.Linear(num_v_heads * value_dim, hidden_size, bias=False)

    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)  # [B, S, hidden]
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.num_k_heads, self.key_dim)
        k = self.k_proj(x).view(b, s, self.num_k_heads, self.key_dim)
        v = self.v_proj(x).view(b, s, self.num_v_heads, self.value_dim)

        # g should be negative (decay), beta should be in (0, 1)
        g = -torch.nn.functional.softplus(self.g_proj(x))  # [B, S, HV]
        beta = torch.sigmoid(self.beta_proj(x))  # [B, S, HV]

        # L2 norm and GVA head expansion are handled inside the op
        attn_out = torch.ops.auto_deploy.torch_gated_delta_rule(q, k, v, g, beta)
        # attn_out: [B, S, HV, V]

        attn_out = attn_out.reshape(b, s, -1)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_k_heads,num_v_heads", [(2, 2), (2, 4)])
@torch.inference_mode()
def test_torch_gated_delta_rule_cache(num_k_heads, num_v_heads):
    """Test the insert_cached_gated_delta_rule transform with torch_gated_delta backend."""
    # Configuration
    dtype = torch.float32
    atol = 1e-3
    rtol = 1e-3
    batch_size = 4
    seq_len = 16
    vocab_size = 100
    hidden_size = 32
    key_dim = 8
    value_dim = 8
    max_position_embeddings = 64

    # Create CachedSequenceInterface
    kv_cache_config = KvCacheConfig(
        tokens_per_block=max_position_embeddings,
        max_tokens=batch_size * max_position_embeddings,
        free_gpu_memory_fraction=0.0,
    )
    cm = CachedSequenceInterface(
        max_seq_len=max_position_embeddings,
        max_batch_size=batch_size,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )

    # Create model
    model = GatedDeltaRuleModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        key_dim=key_dim,
        value_dim=value_dim,
    ).to(dtype=dtype, device="cuda")

    # Create input data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

    # Get uncached reference output
    y_model = model(input_ids)  # [B, S, hidden]

    # Apply the transformation pipeline
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
            "insert_cached_gated_delta_rule": {
                "stage": "cache_init",
                "backend": "torch_gated_delta",
            },
        },
    )  # type: ignore

    gm = optimizer(cm)
    gm.to("cuda")
    num_caches = cm.initialize_resources()
    print(f"num_caches: {num_caches}")

    # Helper function to call the model with proper sequence nesting
    def _call_and_unnest(x, input_pos):
        cm.info.nest_sequences(x, input_pos=input_pos)
        y = gm(**cm.named_args)
        return torch.stack(cm.info.unnest_sequences(y))

    # ---- Test 1: Full-sequence prefill ----
    cm.info.reset()
    # Zero caches for a clean start
    for cache in cm._caches.values():
        if cache is not None:
            cache.zero_()
    y_no_cache = _call_and_unnest(input_ids, 0)
    assert all_close(y_model, y_no_cache, atol=atol, rtol=rtol), (
        "Prefill output does not match uncached model output"
    )

    # ---- Test 2: Token-by-token autoregressive ----
    cm.info.reset()
    # CRITICAL: Zero the delta_cache tensors between test phases.
    # cm.info.reset() only clears sequence metadata, NOT the physical cache.
    # For recurrent state caches, the decode path unconditionally loads state
    # from delta_cache when num_decode > 0, so stale data from the prior
    # prefill test would corrupt the first autoregressive token.
    for cache in cm._caches.values():
        if cache is not None:
            cache.zero_()

    y_with_cache = torch.empty_like(y_model)
    for i_p in range(seq_len):
        y_with_cache[:, i_p : i_p + 1] = _call_and_unnest(
            input_ids[:, i_p : i_p + 1],
            i_p,
        )
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol), (
        "Autoregressive output does not match uncached model output"
    )
