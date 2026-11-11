# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real V2 shell and vision tower through shared embedding preparation."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._multimodal import (
    check_target_multimodal_contract,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default import (
    vision as V,
)
from tensorrt_llm._torch.configs import Qwen4ExpConfig, Qwen4ExpVisionConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.multimodal.model import MultimodalModelMixin, MultimodalRuntimeMixin
from tensorrt_llm._torch.pyexecutor.model_engine import _spec_decode_fast_paths
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks, runtime_hooks_of
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalInput,
    MultimodalItemMetadata,
    MultimodalParams,
    MultimodalRuntimeData,
)
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.cpu_only
Shell = M.ModelingV2Qwen38FlashNextNvfp4Sm103Default


def make_shell(monkeypatch, cache_bytes=4096):
    config = Qwen4ExpVisionConfig(
        depth=0,
        hidden_size=2,
        num_heads=1,
        in_channels=1,
        patch_size=1,
        temporal_patch_size=1,
        spatial_merge_size=2,
        num_position_embeddings=4,
        out_hidden_size=2,
    )
    weights = {
        "embed": torch.zeros(16, 2),
        "vis_patch_w": torch.zeros(2, 1, 1, 1, 1),
        "vis_patch_b": torch.zeros(2),
        "vis_pos": torch.zeros(4, 2),
        "vis_m_n_w": torch.ones(2),
        "vis_m_n_b": torch.zeros(2),
        "vis_m_fc1_w": torch.zeros(8, 8),
        "vis_m_fc1_b": torch.zeros(8),
        "vis_m_fc2_w": torch.zeros(2, 8),
        "vis_m_fc2_b": torch.zeros(2),
    }
    tower = V.VisionTower(weights, config)
    calls = []

    def run(pixels, grid):
        calls.append(tuple(float(value) for value in pixels[:, 0].unique()))
        return pixels[::4].expand(-1, 2).clone()

    monkeypatch.setattr(tower, "_run", run)
    shell = Shell.__new__(Shell)
    torch.nn.Module.__init__(shell)
    shell.model_config = ModelConfig(
        pretrained_config=Qwen4ExpConfig(
            text_config={"hidden_size": 2, "vocab_size": 16},
            vision_config=config.to_dict(),
            dtype="float32",
        ),
        multimodal_config=MultimodalConfig(encoder_cache_max_bytes=cache_bytes),
    )
    shell.vision_tower = tower
    shell._mm_token_ids_value = (7, 8)
    shell._mm_token_ids_cache = None
    shell._initialize_multimodal_encoder_cache(cache_bytes)
    return shell, calls


def make_params(markers, lengths, start=0, end=None, kwargs_hash="same"):
    total = sum(lengths)
    positions = []
    offset = 0
    for length in lengths:
        positions.append(offset)
        offset += length
    return MultimodalParams(
        multimodal_input=MultimodalInput(
            multimodal_hashes=[[marker] * 8 for marker in markers],
            multimodal_positions=positions,
            multimodal_lengths=lengths,
        ),
        multimodal_data={
            "image": {
                "pixel_values": torch.cat(
                    [
                        torch.full((4 * length, 1), float(marker))
                        for marker, length in zip(markers, lengths, strict=True)
                    ]
                ),
                "image_grid_thw": torch.tensor([[1, 2, 2 * length] for length in lengths]),
            },
            "mm_processor_kwargs_hash": kwargs_hash,
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                item_refs=[("image", i) for i in range(len(markers))],
                encoder_token_lengths=[4 * length for length in lengths],
                output_embedding_lengths=lengths,
            ),
        },
        multimodal_runtime=MultimodalRuntimeData(
            past_seen_token_num=start,
            chunk_end_pos=total if end is None else end,
            embed_mask_cumsum=torch.arange(1, total + 1),
        ),
    )


def prepare(shell, params):
    count = params.multimodal_runtime.num_mm_tokens_in_chunk
    ids = torch.full((count,), 7, dtype=torch.int32)
    prepared = shell.prepare_multimodal_embeddings(
        input_ids=ids,
        multimodal_params=[params],
        num_context_requests=1,
        mm_token_indices=torch.arange(count),
    )
    assert prepared.input_ids is ids
    assert not prepared.extra_embeds
    return V.multimodal_embeds(prepared)


def test_chunks_and_repeated_requests_share_encoder_outputs(monkeypatch):
    shell, calls = make_shell(monkeypatch)
    params = make_params([10], [6], end=2)
    torch.testing.assert_close(prepare(shell, params)[0], torch.full((2, 2), 10.0))
    params.multimodal_runtime = make_params([10], [6], start=2, end=5).multimodal_runtime
    torch.testing.assert_close(prepare(shell, params)[0], torch.full((3, 2), 10.0))
    torch.testing.assert_close(prepare(shell, make_params([10], [6]))[0], torch.full((6, 2), 10.0))
    assert calls == [(10.0,)]


def test_partial_hits_select_only_missing_items_and_restore_prompt_order(monkeypatch):
    shell, calls = make_shell(monkeypatch)
    prepare(shell, make_params([10], [2]))
    mixed = make_params([10, 20], [2, 2])
    rows = prepare(shell, mixed)[0]
    torch.testing.assert_close(rows[:, 0], torch.tensor([10.0, 10.0, 20.0, 20.0]))
    reordered = prepare(shell, make_params([20, 10], [2, 2]))[0]
    torch.testing.assert_close(reordered[:, 0], torch.tensor([20.0, 20.0, 10.0, 10.0]))
    assert calls == [(10.0,), (20.0,)]
    assert mixed.multimodal_data["image"]["pixel_values"].shape == (16, 1)


@pytest.mark.parametrize("cache_bytes", [0, 8])
def test_disabled_or_undersized_persistent_cache_keeps_chunk_reuse(monkeypatch, cache_bytes):
    shell, calls = make_shell(monkeypatch, cache_bytes)
    params = make_params([10], [6], end=2)
    prepare(shell, params)
    params.multimodal_runtime = make_params([10], [6], start=2, end=4).multimodal_runtime
    prepare(shell, params)
    assert len(calls) == 1
    prepare(shell, make_params([10], [6]))
    assert len(calls) == 2


def test_processor_kwargs_are_part_of_persistent_cache_key(monkeypatch):
    shell, calls = make_shell(monkeypatch)
    prepare(shell, make_params([10], [2]))
    prepare(shell, make_params([10], [2], kwargs_hash="different"))
    assert len(calls) == 2


def test_position_only_params_are_filtered_and_attached_embeddings_are_consumed(monkeypatch):
    shell, calls = make_shell(monkeypatch)
    positions = MultimodalParams(multimodal_data={"mrope_config": {}})
    attached = MultimodalParams(multimodal_data={"multimodal_embedding": torch.ones(2, 2)})
    chosen = shell.select_multimodal_params([positions, attached], 2)
    assert len(chosen) == 1 and chosen[0] is attached
    result = shell.prepare_multimodal_embeddings(
        input_ids=torch.tensor([7, 7]),
        multimodal_params=[positions, attached],
        num_context_requests=2,
    )
    torch.testing.assert_close(result.multimodal_embeddings[0], torch.ones(2, 2))
    assert not calls


def test_decoder_contracts_and_speculative_hooks_keep_the_integrated_shell(monkeypatch):
    shell, _ = make_shell(monkeypatch)
    assert isinstance(shell, MultimodalRuntimeMixin)
    assert not isinstance(shell, MultimodalModelMixin)
    # The runtime mixin delegates nothing to a separate language model.
    assert not hasattr(shell, "language_model")
    assert not hasattr(shell, "set_guided_decoder")
    assert Shell.vocab_size_padded is DecoderModelForCausalLM.vocab_size_padded
    assert Shell.infer_max_seq_len is DecoderModelForCausalLM.infer_max_seq_len
    assert Shell.forward is not MultimodalModelMixin.forward
    assert shell.embedding_dim == 2 and shell.dtype == torch.float32
    shell.spec_decode_fast_paths = True
    shell.runtime_hooks = RuntimeHooks(loop_on_execution_stream=True)
    assert _spec_decode_fast_paths(shell)
    assert runtime_hooks_of(shell).loop_on_execution_stream


def test_the_shell_meets_the_multimodal_runtime_contract(monkeypatch):
    shell, _ = make_shell(monkeypatch)
    Shell.check_multimodal_class_contract()
    shell.check_multimodal_instance_contract()
    check_target_multimodal_contract("Qwen4ExpForConditionalGeneration", Shell)
    assert Shell.supports_concurrent_encoder
    assert not Shell.supports_mm_encoder_item_scheduling
    assert shell.max_encoder_tokens_per_item("image") == V.VISION_CALL_MAX_PATCHES
    assert shell.max_encoder_tokens_per_item("video") == V.VISION_CALL_MAX_PATCHES
    assert shell.allows_breakable_prefill_cuda_graph(None)
