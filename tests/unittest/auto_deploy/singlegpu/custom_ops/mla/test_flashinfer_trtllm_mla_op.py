from unittest.mock import patch

import pytest
import torch
from auto_deploy.singlegpu.custom_ops.mla.test_flashinfer_mla_op import (
    _create_mla_inputs,
    _create_paged_cache_and_metadata,
    _create_unpaged_cache_and_metadata,
)

import tensorrt_llm._torch.auto_deploy  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.mla import flashinfer_trtllm_mla as trtllm_mla_mod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FlashInfer TRTLLM MLA tests require CUDA",
)


def _make_combined_cache(ckv_cache: torch.Tensor, kpe_cache: torch.Tensor) -> torch.Tensor:
    combined = torch.zeros(
        ckv_cache.shape[0],
        ckv_cache.shape[1],
        1,
        ckv_cache.shape[2] + kpe_cache.shape[2],
        dtype=ckv_cache.dtype,
        device=ckv_cache.device,
    )
    combined[:, :, 0, : ckv_cache.shape[2]] = ckv_cache
    combined[:, :, 0, ckv_cache.shape[2] :] = kpe_cache
    return combined


def test_flashinfer_trtllm_mla_decode_falls_back_to_torch_on_hopper():
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("requires Hopper+ for MLA test setup")

    # Force the non-Blackwell fallback path so this test exercises the
    # reference decode even when running on Blackwell GPUs.
    with patch.object(trtllm_mla_mod, "_is_blackwell_decode_supported", return_value=False):
        _run_fallback_decode_test()


def _run_fallback_decode_test():
    batch_size = 2
    seq_len = 1
    prefill_seq_length = 128
    num_heads = 4
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 256
    v_head_dim = 128
    dtype = torch.bfloat16
    device = "cuda"
    page_size = 64
    max_seq_len = prefill_seq_length + seq_len + 64
    max_num_pages = batch_size * (max_seq_len // page_size + 2)

    inputs = _create_mla_inputs(
        batch_size,
        seq_len,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )

    seq_lengths = [seq_len] * batch_size
    input_positions = [prefill_seq_length] * batch_size

    torch_meta = _create_unpaged_cache_and_metadata(
        batch_size,
        max_seq_len,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )
    torch_meta["mla_cache"][:, :prefill_seq_length].normal_()

    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )
    for batch_idx in range(batch_size):
        page_start = int(flashinfer_meta["cu_num_pages_host"][batch_idx].item())
        page_end = int(flashinfer_meta["cu_num_pages_host"][batch_idx + 1].item())
        for rel_page_idx, flat_page_idx in enumerate(range(page_start, page_end)):
            page_idx = int(flashinfer_meta["cache_loc"][flat_page_idx].item())
            tokens_in_page = min(page_size, prefill_seq_length - rel_page_idx * page_size)
            if tokens_in_page > 0:
                prefix = torch_meta["mla_cache"][
                    batch_idx,
                    rel_page_idx * page_size : rel_page_idx * page_size + tokens_in_page,
                ]
                flashinfer_meta["ckv_cache"][page_idx, :tokens_in_page] = prefix[:, :kv_lora_rank]
                flashinfer_meta["kpe_cache"][page_idx, :tokens_in_page] = prefix[:, kv_lora_rank:]

    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )

    torch_output = torch.ops.auto_deploy.torch_cached_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        torch_meta["batch_info_host"],
        torch_meta["seq_len"],
        torch_meta["input_pos"],
        torch_meta["slot_idx"],
        torch_meta["cu_seqlen"],
        torch_meta["mla_cache"],
        None,
        kv_lora_rank,
    )

    trtllm_output = torch.ops.auto_deploy.flashinfer_trtllm_mla_with_cache(
        inputs["q_nope"],
        inputs["q_pe"],
        inputs["compressed_kv"],
        inputs["kpe"],
        inputs["kv_b_proj_weight"],
        flashinfer_meta["batch_info_host"],
        flashinfer_meta["cu_seqlen_host"],
        flashinfer_meta["cu_num_pages"],
        flashinfer_meta["cu_num_pages_host"],
        flashinfer_meta["cache_loc"],
        flashinfer_meta["last_page_len"],
        flashinfer_meta["last_page_len_host"],
        flashinfer_meta["seq_len_with_cache_host"],
        combined_cache,
        None,
        kv_lora_rank,
    )

    assert torch.allclose(trtllm_output.float(), torch_output.float(), atol=5e-2, rtol=5e-2), (
        "Hopper fallback path should match torch_cached_mla_with_cache"
    )


def test_flashinfer_trtllm_mla_uses_blackwell_decode_kernel_when_available():
    batch_size = 2
    seq_len = 1
    num_heads = 4
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    kv_lora_rank = 256
    v_head_dim = 128
    dtype = torch.bfloat16
    device = "cuda"
    page_size = 64
    max_num_pages = batch_size * 4

    inputs = _create_mla_inputs(
        batch_size,
        seq_len,
        num_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        kv_lora_rank,
        v_head_dim,
        dtype,
        device,
    )
    seq_lengths = [seq_len] * batch_size
    input_positions = [64] * batch_size
    flashinfer_meta = _create_paged_cache_and_metadata(
        batch_size,
        max_num_pages,
        page_size,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        device,
        seq_lengths,
        input_positions,
    )
    combined_cache = _make_combined_cache(
        flashinfer_meta["ckv_cache"], flashinfer_meta["kpe_cache"]
    )

    latent_out = torch.ones(batch_size, 1, num_heads, kv_lora_rank, dtype=dtype, device=device)
    weight_reshaped = inputs["kv_b_proj_weight"].view(
        num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
    )
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]
    expected = torch.einsum("bsnk,nvk->bsnv", latent_out, w_v)

    with patch.object(trtllm_mla_mod, "_is_blackwell_decode_supported", return_value=True):
        with patch.object(
            trtllm_mla_mod.flashinfer.mla,
            "trtllm_batch_decode_with_kv_cache_mla",
            return_value=latent_out,
        ) as kernel_mock:
            output = torch.ops.auto_deploy.flashinfer_trtllm_mla_with_cache(
                inputs["q_nope"],
                inputs["q_pe"],
                inputs["compressed_kv"],
                inputs["kpe"],
                inputs["kv_b_proj_weight"],
                flashinfer_meta["batch_info_host"],
                flashinfer_meta["cu_seqlen_host"],
                flashinfer_meta["cu_num_pages"],
                flashinfer_meta["cu_num_pages_host"],
                flashinfer_meta["cache_loc"],
                flashinfer_meta["last_page_len"],
                flashinfer_meta["last_page_len_host"],
                flashinfer_meta["seq_len_with_cache_host"],
                combined_cache,
                None,
                kv_lora_rank,
            )

    kernel_mock.assert_called_once()
    assert torch.allclose(output.float(), expected.float(), atol=1e-4, rtol=1e-4)
