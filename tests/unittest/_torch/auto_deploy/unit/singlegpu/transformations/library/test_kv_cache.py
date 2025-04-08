from typing import Optional

import pytest
import torch
from _model_test_utils import GQA
from _torch_test_utils import all_close

from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import CacheConfig, SequenceInfo
from tensorrt_llm._torch.auto_deploy.custom_ops.flashinfer_attention import FlashInferAttention
from tensorrt_llm._torch.auto_deploy.custom_ops.triton_attention import TritonWithFlattenedInputs
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transformations.export import torch_export, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations.library import (
    check_in_out_nodes,
    insert_mha_with_kv_cache,
)


# Class that inherits from GQA but uses fused_mha directly
class GQAWithFusedMHA(GQA):
    """GQA model that uses torch.ops.attention.fused_mha directly instead of SDPA."""

    def __init__(
        self,
        *args,
        pos_embd_mode: Optional[str] = None,
        rope_theta: float = 10000.0,
        rope_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pos_embd_mode = pos_embd_mode
        self.rope_theta = rope_theta if pos_embd_mode == "rope" else None
        self.rope_scale = rope_scale if pos_embd_mode == "rope" else None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        # Project input to q, k, v representations
        q = self.q_proj(x)  # [b, s, n*h_d]
        k = self.k_proj(x)  # [b, s, n_kv*h_d]
        v = self.v_proj(x)  # [b, s, n_kv*h_d]

        # Call fused_mha directly
        y = torch.ops.attention.fused_mha(
            q=q,
            k=k,
            v=v,
            head_dim=self.head_dim,
            pos_embd_mode=self.pos_embd_mode,
            rope_theta=self.rope_theta,
            rope_scale=self.rope_scale,
        )

        # Apply output projection
        return self.o_proj(y)


@pytest.mark.parametrize(
    "use_rope,atol,rtol",
    [
        (True, 1e-2, 1e-2),  # Much more relaxed tolerance for RoPE due to float16 precision issues
        (False, 1e-3, 1e-3),  # Default tolerance for non-RoPE cases
    ],
    ids=["with_rope", "without_rope"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32],
    ids=["float16", "float32"],
)
@pytest.mark.parametrize(
    "attention_op",
    [FlashInferAttention, TritonWithFlattenedInputs],
    ids=["flashinfer", "triton"],
)
@torch.inference_mode()
def test_model_with_kv_cache(use_rope, atol, rtol, dtype, attention_op):
    # some config
    batch_size, seq_len = 16, 64
    num_reset_steps = 2
    num_random_steps = 10
    # Use 16 heads with 64 dimensions each to maintain hidden_size=1024
    # This ensures compatibility with FlashInfer which requires head_dim=64
    num_attention_heads = 16  # Changed from 32 to 16
    hidden_size = 1024
    num_key_value_heads = 16  # Changed from 32 to 16

    # FlashInfer doesn't support float32 data type
    if attention_op == FlashInferAttention and dtype == torch.float32:
        pytest.skip("FlashInfer doesn't support float32 data type")

    max_position_embeddings = 128

    # set up sequence+cache objects
    ci = SequenceInfo(
        max_seq_len=max_position_embeddings,
        max_batch_size=batch_size,
    )
    cm = CachedSequenceInterface(sequence_info=ci, device="cuda")

    # Use the model with fused MHA directly instead of regular GQA
    model = GQAWithFusedMHA(
        num_attention_heads,
        hidden_size,
        num_key_value_heads,
        pos_embd_mode="rope" if use_rope else None,
        rope_theta=10000.0,
        rope_scale=1.0,
    ).to(device="cuda", dtype=dtype)

    x = torch.rand(batch_size, seq_len, hidden_size).to(device="cuda", dtype=dtype)

    # get the model's regular output
    y_model = model(x)  # b, s, d

    # export + check (we clone the state dict to have a bit more freedom in testing below)
    gm = torch_export_to_gm(
        model,
        args=(x,),
        clone=True,
        dynamic_shapes=cm.dynamic_shapes[:1],
    )
    y_gm = gm(x)
    assert all_close(y_model, y_gm, atol=atol, rtol=rtol)

    # Since we're already using fused_mha, we can skip the fusion step
    # and directly insert KV cache
    cache_config = CacheConfig()
    # get input node
    input_node = check_in_out_nodes(gm)
    gm_transformed = insert_mha_with_kv_cache(
        gm, cm, attention_op=attention_op, cache_config=cache_config, input_node=input_node
    )
    gm_transformed.to("cuda")
    cm.initialize_caches()

    def _call_and_unnest(x):
        cm.info.nest_sequences(x)
        y = gm_transformed(*cm.args)
        return torch.stack(cm.info.unnest_sequences(y))

    # run regular inference
    cm.info.reset()
    y_no_cache = _call_and_unnest(x)
    assert all_close(y_model, y_no_cache, atol=atol, rtol=rtol)

    # run inference with kv cache
    cm.info.reset()
    y_with_cache = torch.empty_like(y_model)
    for i in range(x.shape[1]):
        y_with_cache[:, i : i + 1] = _call_and_unnest(x[:, i : i + 1])
        cm.info.update_pos(1)
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # try running some garbage through the caches and then bring back input_pos to see
    # if that works
    cm.info.update_pos(-num_reset_steps)  # should be x.shape[1] - num_reset
    for i in range(num_random_steps):
        _call_and_unnest(torch.rand_like(x[:, :1]))
        cm.info.update_pos(1)

    # go back and run inference again
    cm.info.reset()
    cm.info.update_pos(x.shape[1] - num_reset_steps)
    for i in range(x.shape[1] - 2, x.shape[1]):
        y_with_cache[:, i : i + 1] = _call_and_unnest(x[:, i : i + 1])
        cm.info.update_pos(1)
    assert all_close(y_model, y_with_cache, atol=atol, rtol=rtol)

    # check if we can still export the model as expected
    torch_export(gm_transformed, args=cm.args)
    torch_export_to_gm(gm_transformed, args=cm.args)
