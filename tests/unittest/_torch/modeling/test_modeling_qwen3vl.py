import copy
import os
from dataclasses import dataclass
from typing import List, Optional

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalLM
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.hf.qwen3vl_weight_mapper import Qwen3VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VisionModel,
    Qwen3VLInputProcessorBase,
    Qwen3VLModel,
    _triton_pos_embed_interpolate,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

QWEN3_VL_8B_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 4,
        # NOTE: Only 4 layers for testing, 36 layers for full model.
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
    "_attn_implementation": "flash_attention_2",
    "_name_or_path": str(os.path.join(llm_models_root(), "Qwen3", "Qwen3-VL-8B-Instruct")),
}


@dataclass(repr=False)
class TestQwen3VLScenario(MultimodalScenario):
    disable_fuse_rope: bool = False

    def __repr__(self) -> str:
        """Generate a human-readable string representation of the scenario."""
        features = []
        features.append(f"modality:{self.modality.lower()}")
        if self.use_cuda_graph:
            features.append("cuda_graph")
        if self.disable_fuse_rope:
            features.append("no_fuse_rope")
        if self.chunked_prefill:
            features.append("chunked_prefill")
        if self.kv_cache_reuse:
            features.append("kv_cache_reuse")
        return "-".join(features)


class TestQwen3VL(TestModelingMultimodal):
    def get_model_config(self):
        """Return the model configuration dictionary."""
        return QWEN3_VL_8B_CONFIG

    def get_trtllm_model_class(self):
        return Qwen3VLModel

    def get_hf_model_class(self):
        return HFQwen3VLForConditionalLM

    def get_weight_mapper_class(self):
        return Qwen3VLHfWeightMapper

    def get_model_type(self):
        return "qwen3_vl"

    def get_model_config_class(self):
        return Qwen3VLConfig

    def get_trtllm_inputs(
        self,
        input_ids,
        multimodal_params_list,
        is_gen: bool = False,
        num_cached_tokens_per_seq: List[int] = None,
        total_prompt_len: Optional[int] = None,
    ):
        trtllm_inputs = super().get_trtllm_inputs(
            input_ids,
            multimodal_params_list,
            is_gen,
            num_cached_tokens_per_seq,
            total_prompt_len=total_prompt_len,
        )

        if is_gen:
            mrope_gen_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_gen_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]["mrope_position_deltas"]
                )
            mrope_gen_position_ids = torch.cat(mrope_gen_position_ids, dim=-1).to(self.device)
            trtllm_inputs["position_ids"] = (
                (trtllm_inputs["position_ids"] + mrope_gen_position_ids).expand(3, -1, 1).cuda()
            )
            gen_multimodal_params_list = []
            for multimodal_param in multimodal_params_list:
                multimodal_param.strip_for_generation()
                multimodal_param.to_device(
                    "multimodal_data",
                    self.device,
                    pin_memory=True,
                    target_keywords=["mrope_config.mrope_position_deltas"],
                )
                gen_multimodal_params_list.append(multimodal_param)
            trtllm_inputs["multimodal_params"] = gen_multimodal_params_list
        else:
            # Mrope position ids. For chunked prefill / KV cache reuse we must
            # mirror production ``PyTorchModelEngine`` behavior and slice the
            # request's full ``mrope_position_ids`` to the current chunk's
            # range -- the model now indexes mrope cos/sin by batch-flat
            # per-token index, so the position_ids tensor must contain only
            # the tokens of the current forward (chunk-local) with their
            # correct (T, H, W) values.
            chunk_len = input_ids.shape[-1]
            if num_cached_tokens_per_seq is None:
                begin_offsets = [0] * len(multimodal_params_list)
            elif isinstance(num_cached_tokens_per_seq, int):
                begin_offsets = [num_cached_tokens_per_seq] * len(multimodal_params_list)
            else:
                begin_offsets = list(num_cached_tokens_per_seq)
            mrope_position_ids = []
            for multimodal_param, begin in zip(multimodal_params_list, begin_offsets):
                full_mrope = multimodal_param.multimodal_data["mrope_config"]["mrope_position_ids"]
                mrope_position_ids.append(full_mrope[:, :, begin : begin + chunk_len])
            position_ids = torch.cat(mrope_position_ids, dim=-1)
            position_ids = position_ids.cuda()
            trtllm_inputs["position_ids"] = position_ids

        return trtllm_inputs

    def init_kv_cache_manager(self, scenario: TestQwen3VLScenario):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen3-VL model."""
        cache_config = self.get_kv_cache_config(scenario)
        tokens_per_block = cache_config["tokens_per_block"]
        max_seq_len = cache_config["max_seq_len"]
        batch_size = cache_config["batch_size"]

        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block

        self.kv_cache_manager = self.get_kv_cache_manager(
            dtype=self.model_config.pretrained_config.torch_dtype,
            config=self.model_config.pretrained_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks,
        )

        self.kv_cache_manager.add_dummy_requests(
            request_ids=[1],
            token_nums=[max_seq_len],
            # NOTE: Qwen3-VL model uses mrope
            use_mrope=True,
        )

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen3-VL model."""
        if not use_cuda_graph:
            trtllm_inputs["attn_metadata"].prepare()
            return self.trtllm_model.forward(**trtllm_inputs)
        else:
            # NOTE: Qwen3-VL model uses mrope
            graph_runner = create_mock_cuda_graph_runner(1, True)
            trtllm_inputs["attn_metadata"] = trtllm_inputs[
                "attn_metadata"
            ].create_cuda_graph_metadata(1)

            # Prepare metadata before capture (like in working Qwen2.5-VL test)
            trtllm_inputs["attn_metadata"].prepare()

            key = (1, 0, False)
            graph_runner.capture(
                key=key,
                forward_fn=lambda inputs: self.trtllm_model.forward(**inputs),
                initial_inputs=trtllm_inputs,
            )
            for _ in range(2):
                # Run it twice. This helps us catch problems if buffers are accidentally reallocated in prepare().
                trtllm_inputs["attn_metadata"].prepare()
                logits = graph_runner.replay(key=key, current_inputs=trtllm_inputs)
            return logits.clone()

    def get_scenarios(self) -> List[TestQwen3VLScenario]:
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestQwen3VLScenario(
                modality="video",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            TestQwen3VLScenario(
                modality="multiple_image",
                use_cuda_graph=False,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
            # ==== CUDA Graph Scenarios ====
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=True,
                disable_fuse_rope=False,
                chunked_prefill=False,
                kv_cache_reuse=False,
            ),
        ]
        # Paged context FMHA (triggered by chunked_prefill / kv_cache_reuse)
        # is forced on for correctness on Hopper (SM90); the trtllm-gen
        # kernel set on Blackwell (SM100) falls back to an unfused MHA path
        # whose output diverges from the non-paged context kernel. Gate
        # those scenarios to SM90 until the Blackwell fallback matches.
        if torch.cuda.is_available() and get_sm_version() == 90:
            scenarios.extend(
                [
                    # ==== Chunked Prefill Scenarios ====
                    TestQwen3VLScenario(
                        modality="image",
                        use_cuda_graph=False,
                        disable_fuse_rope=False,
                        chunked_prefill=True,
                        kv_cache_reuse=False,
                    ),
                    # ==== KV Cache Reuse Scenarios ====
                    TestQwen3VLScenario(
                        modality="image",
                        use_cuda_graph=False,
                        disable_fuse_rope=False,
                        chunked_prefill=False,
                        kv_cache_reuse=True,
                    ),
                ]
            )
        # ==== Disable fuse rope scenarios ====
        # Run last: setup_scenario rebuilds trtllm_model with
        # disable_fuse_rope=True for this scenario, and the rebuild is not
        # undone afterwards. Keeping it at the tail prevents the rebuilt
        # model from leaking into chunked-prefill / kv-cache-reuse
        # scenarios (where it surfaces as a cos/sin vs. q/k seq-len
        # mismatch in MRotaryEmbedding.forward).
        scenarios.append(
            TestQwen3VLScenario(
                modality="image",
                use_cuda_graph=False,
                disable_fuse_rope=True,
                chunked_prefill=False,
                kv_cache_reuse=False,
            )
        )
        return scenarios

    def get_hf_inputs(self, modality: str, prompt, media):
        processor_inputs = super().get_hf_inputs(modality, prompt, media)

        # For video: the parent class already deleted mm_token_type_ids, which
        # causes HF to fall back to 1D position IDs (no MRope). Qwen3VL encodes
        # video frames as separate timestamp-separated vision segments, so the
        # correct approach is to compute 3D MRope position IDs via TRT-LLM's
        # get_rope_index (which expands video_grid_thw per-frame) and pass them
        # explicitly to HF's forward(), bypassing compute_3d_position_ids.
        if modality == "video" and "video_grid_thw" in processor_inputs:
            position_ids, _ = Qwen3VLInputProcessorBase.get_rope_index(
                self.hf_config,
                processor_inputs["input_ids"],
                image_grid_thw=processor_inputs.get("image_grid_thw"),
                video_grid_thw=processor_inputs["video_grid_thw"],
                attention_mask=processor_inputs.get("attention_mask"),
            )
            processor_inputs["position_ids"] = position_ids.to(processor_inputs["input_ids"].device)

        return processor_inputs

    def setup_scenario(self, scenario: TestQwen3VLScenario):
        super().setup_scenario(scenario)
        if scenario.disable_fuse_rope:
            self.trtllm_model, self.model_config = self.create_trtllm_model(
                load_weights=True,
                hf_model_state_dict=self.hf_model.state_dict(),
                disable_fuse_rope=True,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen3vl_init_preserves_caller_quant_config():
    """Building Qwen3VLModel must not mutate the caller's quant_config."""
    hf_config = Qwen3VLConfig.from_dict(copy.deepcopy(QWEN3_VL_8B_CONFIG))
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    model_config = ModelConfig(
        pretrained_config=hf_config,
        quant_config=quant_config,
        skip_create_weights_in_init=True,
    )

    model = Qwen3VLModel(model_config)

    # Outer LLM keeps the caller's quant_config unchanged (same object, same FP8 KV-cache setting).
    assert model.model_config.quant_config is quant_config
    assert model.model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8

    # Vision encoder operates on an independent copy whose quant settings have been reset to the
    # defaults (no quantization).
    assert model.mm_encoder.model_config.quant_config is not quant_config
    assert model.mm_encoder.model_config.quant_config.kv_cache_quant_algo is None
    assert model.mm_encoder.model_config.quant_config.quant_algo is None


# ---------------------------------------------------------------------------
# Accuracy tests for the fused Triton bilinear position-embedding kernel used
# by the Qwen3-VL vision tower. Mirrors vLLM's
# ``tests/kernels/core/test_vit_bilinear_pos_embed.py`` so the two
# implementations stay aligned.
# ---------------------------------------------------------------------------

_VIT_POS_EMBED_DTYPES = [torch.float32, torch.bfloat16]
# Qwen3-VL defaults
_VIT_POS_EMBED_NUM_GRID_PER_SIDE = 48
_VIT_POS_EMBED_SPATIAL_MERGE_SIZE = 2
_VIT_POS_EMBED_HIDDEN_DIM = 1152

_VIT_POS_EMBED_GRIDS = [
    (1, 4, 4),
    (1, 16, 16),
    (1, 32, 32),
    (1, 48, 48),
    (1, 8, 16),
    (1, 14, 20),
    (1, 32, 48),
    (1, 60, 80),
]


def _vit_pos_embed_native_reference(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Eager PyTorch reference for one (t, h, w) grid — the pre-Triton
    implementation, kept here only as the test oracle for
    ``_triton_pos_embed_interpolate``.
    """
    hidden_dim = embed_weight.shape[1]
    device = embed_weight.device

    h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=device)

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    h_grid_idx = h_grid * num_grid_per_side

    indices = (h_grid_idx + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype)

    embeds = embed_weight[indices]
    embeds *= weights
    combined = embeds.sum(dim=0)

    combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
    return combined.expand(t, -1, -1).reshape(-1, hidden_dim).to(dtype)


@pytest.mark.parametrize("dtype", _VIT_POS_EMBED_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize(
    "grid_thw",
    _VIT_POS_EMBED_GRIDS,
    ids=[f"{t}x{h}x{w}" for t, h, w in _VIT_POS_EMBED_GRIDS],
)
def test_vit_pos_embed_triton_matches_native(grid_thw, dtype):
    """Triton kernel output must match the PyTorch reference within
    bf16 / fp32 ULP-level rounding."""
    t, h, w = grid_thw
    device = "cuda"

    # Scale to match the real Qwen3-VL pos_embed weight distribution
    # (std ~ 0.23). Larger scales magnify the bf16 rounding gap between
    # the two op orderings.
    torch.manual_seed(42)
    embed_weight = (
        torch.randn(
            _VIT_POS_EMBED_NUM_GRID_PER_SIDE * _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
            _VIT_POS_EMBED_HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )
        * 0.25
    )

    native_out = _vit_pos_embed_native_reference(
        embed_weight,
        t,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )
    triton_out = _triton_pos_embed_interpolate(
        embed_weight,
        t,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )

    assert native_out.shape == triton_out.shape

    # Single-ULP differences come from the precomputed scalar h/w_scale
    # in the Triton kernel vs ``torch.linspace`` in the reference. Match
    # vLLM's tolerances for the same kernel.
    atol = {torch.float32: 5e-5, torch.bfloat16: 1e-2}[dtype]
    rtol = {torch.float32: 1e-5, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(triton_out, native_out, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", _VIT_POS_EMBED_DTYPES, ids=lambda d: str(d).split(".")[-1])
def test_vit_pos_embed_temporal_repeat(dtype):
    """t > 1 must repeat the (h, w) spatial pattern verbatim."""
    device = "cuda"
    h, w = 16, 16
    t_single, t_multi = 1, 3

    torch.manual_seed(42)
    embed_weight = (
        torch.randn(
            _VIT_POS_EMBED_NUM_GRID_PER_SIDE * _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
            _VIT_POS_EMBED_HIDDEN_DIM,
            device=device,
            dtype=dtype,
        )
        * 0.25
    )

    out_single = _triton_pos_embed_interpolate(
        embed_weight,
        t_single,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )
    out_multi = _triton_pos_embed_interpolate(
        embed_weight,
        t_multi,
        h,
        w,
        _VIT_POS_EMBED_NUM_GRID_PER_SIDE,
        _VIT_POS_EMBED_SPATIAL_MERGE_SIZE,
        dtype,
    )

    expected = out_single.repeat(t_multi, 1)
    torch.testing.assert_close(out_multi, expected, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Accuracy tests for ``Qwen3VisionModel.rot_pos_emb``'s pos_ids construction.
# The new implementation lifts pos_ids generation to CPU (numpy + lru_cache)
# and ships a single H->D copy at the end. Previously pos_ids were built on
# device via a per-image torch.arange/expand/stack chain. The freq_table
# lookup is identical in both paths, so we only verify pos_ids equality —
# that guarantees the final ``freq_table[pos_ids]`` output matches bit-for-bit.
# ---------------------------------------------------------------------------


def _rot_pos_ids_gpu_reference(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Pre-vectorization GPU reference for pos_ids.

    Mirrors the original on-device implementation: per-image torch.arange
    + broadcast/expand/stack/repeat, written into a contiguous (total, 2)
    buffer on the same device as ``grid_thw``.
    """
    device = grid_thw.device
    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

    offset = 0
    for num_frames, height, width in grid_thw:
        merged_h, merged_w = height // spatial_merge_size, width // spatial_merge_size

        block_rows = torch.arange(merged_h, device=device)
        block_cols = torch.arange(merged_w, device=device)
        intra_row = torch.arange(spatial_merge_size, device=device)
        intra_col = torch.arange(spatial_merge_size, device=device)

        row_idx = (
            block_rows[:, None, None, None] * spatial_merge_size + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * spatial_merge_size + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)
        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    return pos_ids


_ROT_POS_IDS_GRIDS = [
    # Single image, varying (h, w)
    [(1, 4, 4)],
    [(1, 16, 16)],
    [(1, 32, 48)],
    [(1, 14, 20)],
    # Multi-frame video (t > 1)
    [(3, 16, 16)],
    [(5, 8, 12)],
    # Multi-image batch (mixed sizes)
    [(1, 16, 16), (1, 32, 32)],
    [(1, 8, 12), (1, 14, 20), (1, 32, 32)],
    # Mix of images and a video clip
    [(1, 16, 16), (3, 8, 12), (1, 14, 20)],
]


@pytest.mark.parametrize(
    "grid_thw_list",
    _ROT_POS_IDS_GRIDS,
    ids=["_".join(f"{t}x{h}x{w}" for t, h, w in cfg) for cfg in _ROT_POS_IDS_GRIDS],
)
def test_rot_pos_ids_matches_gpu_reference(grid_thw_list):
    """``rot_pos_ids`` + cat (new path) must produce the same pos_ids tensor
    as the pre-vectorization on-device implementation."""
    spatial_merge_size = _VIT_POS_EMBED_SPATIAL_MERGE_SIZE
    device = "cuda"

    grid_thw = torch.tensor(grid_thw_list, dtype=torch.long, device=device)
    expected = _rot_pos_ids_gpu_reference(grid_thw, spatial_merge_size)

    # New path: per-(h, w) CPU build (lru_cache), per-image repeat, then
    # cat + single H->D transfer.
    pieces = [
        Qwen3VisionModel.rot_pos_ids(h, w, spatial_merge_size)
        if t == 1
        else Qwen3VisionModel.rot_pos_ids(h, w, spatial_merge_size).repeat(t, 1)
        for t, h, w in grid_thw_list
    ]
    actual = torch.cat(pieces, dim=0).to(device, non_blocking=True)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_rot_pos_ids_lru_cache_hit():
    """Repeated (h, w, spatial_merge_size) keys must hit the lru_cache and
    return the same underlying tensor object (no recompute)."""
    Qwen3VisionModel.rot_pos_ids.cache_clear()
    a = Qwen3VisionModel.rot_pos_ids(16, 16, 2)
    b = Qwen3VisionModel.rot_pos_ids(16, 16, 2)
    assert a is b, "lru_cache should return the cached tensor object on hit"
    info = Qwen3VisionModel.rot_pos_ids.cache_info()
    assert info.hits >= 1 and info.misses >= 1


# ---------------------------------------------------------------------------
# L2 cache: per-(t, h, w) GPU rotary embeddings via _rotary_pos_emb_thw.
#
# Mirrors vLLM's get_rope_by_thw pattern: identical (t, h, w) returns the
# same on-device tensor object so subsequent rot_pos_emb calls only do a
# device-side torch.cat with no H->D transfer.
# ---------------------------------------------------------------------------


def _make_qwen3_vision_model_for_l2():
    """Build a minimally-initialized Qwen3VisionModel that exposes
    ``_rotary_pos_emb_thw`` and ``rot_pos_emb`` without loading any weights.

    Bypasses the heavy ``__init__`` (blocks, mergers, attn metadata buffers)
    by going through ``__new__`` so the class-level decorators (``@staticmethod``
    on ``rot_pos_ids`` and ``@lru_cache`` on ``_rotary_pos_emb_thw``) stay intact.
    The mock wires only the attrs those cached methods touch:
    ``spatial_merge_size``, ``rotary_pos_emb`` (a stub exposing the
    ``rotary_cos_sin`` cache that ``_rotary_pos_emb_thw`` indexes into),
    and ``patch_embed.proj.weight`` (read by ``device``).
    """
    obj = Qwen3VisionModel.__new__(Qwen3VisionModel)
    torch.nn.Module.__init__(obj)
    obj.spatial_merge_size = 2

    # Mirror the real init: rope_dim = head_dim // 2 = 36, theta = 10000.
    rope_dim = 36
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    max_rope_seqlen = 8192
    seq = torch.arange(max_rope_seqlen, dtype=torch.float32)
    freqs = torch.outer(seq, inv_freq)
    # ``_rotary_pos_emb_thw`` slices ``rotary_pos_emb.rotary_cos_sin`` and
    # indexes with ``pos_ids``. Stack cos/sin along dim 1 to match the
    # real ``RotaryEmbedding`` buffer layout ``(max_pos, 2, freq_dim)``.
    rotary_cos_sin = torch.stack([freqs.cos(), freqs.sin()], dim=1).to("cuda")
    rotary_pos_emb = torch.nn.Module()
    rotary_pos_emb.rotary_cos_sin = rotary_cos_sin
    obj.rotary_pos_emb = rotary_pos_emb

    # patch_embed.proj.weight is what ``Qwen3VisionModel.device`` reads.
    obj.patch_embed = torch.nn.Module()
    obj.patch_embed.proj = torch.nn.Linear(1, 1, bias=False).to("cuda")
    return obj


def test_rotary_pos_emb_thw_returns_device_tensor():
    """Cached per-(t, h, w) rotary (cos, sin) pair must live on CUDA so
    subsequent rot_pos_emb calls do device-side cats (no H->D transfer)."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    cos, sin = vm._rotary_pos_emb_thw(1, 16, 16)
    assert cos.is_cuda and sin.is_cuda, "L2 cache must store cos/sin on device"
    assert cos.shape == sin.shape
    assert cos.shape[0] == 1 * 16 * 16
    # After gather + flatten the per-token dim is 2 * freq_dim (= 36).
    assert cos.shape[1] % 2 == 0


def test_rotary_pos_emb_thw_lru_cache_hit():
    """Same (t, h, w) must return the same cached (cos, sin) tuple
    (object identity, not equality)."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()
    a_cos, a_sin = vm._rotary_pos_emb_thw(1, 16, 16)
    b_cos, b_sin = vm._rotary_pos_emb_thw(1, 16, 16)
    assert a_cos is b_cos and a_sin is b_sin, (
        "L2 lru_cache must return the cached device tensors on hit"
    )
    info = vm._rotary_pos_emb_thw.cache_info()
    assert info.hits >= 1 and info.misses >= 1


def test_rot_pos_emb_l2_matches_per_tile():
    """rot_pos_emb on a multi-image grid must equal the per-tile (cos, sin)
    tensors concatenated along dim 0, bit-exact."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    grid_thw_list = [(1, 16, 16), (1, 32, 32), (1, 16, 16), (3, 8, 12)]
    cos_out, sin_out = vm.rot_pos_emb(grid_thw_list)

    cos_exp = torch.cat(
        [vm._rotary_pos_emb_thw(t, h, w)[0] for t, h, w in grid_thw_list],
        dim=0,
    )
    sin_exp = torch.cat(
        [vm._rotary_pos_emb_thw(t, h, w)[1] for t, h, w in grid_thw_list],
        dim=0,
    )
    torch.testing.assert_close(cos_out, cos_exp, atol=0, rtol=0)
    torch.testing.assert_close(sin_out, sin_exp, atol=0, rtol=0)


def test_rot_pos_emb_l2_no_device_transfer_on_hit():
    """After the first call warms the L2 cache for a (t, h, w), a second
    rot_pos_emb call on the same grid must not trigger any new miss."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    grid = [(1, 16, 16), (1, 32, 32), (1, 16, 16)]
    vm.rot_pos_emb(grid)  # warm
    miss_count = vm._rotary_pos_emb_thw.cache_info().misses

    vm.rot_pos_emb(grid)  # all hits
    info = vm._rotary_pos_emb_thw.cache_info()
    assert info.misses == miss_count, (
        f"second rot_pos_emb call should be all cache hits; got "
        f"{info.misses - miss_count} extra miss(es)"
    )
    # Per-tile cos/sin must be on CUDA so forward's torch.cat is device-side.
    for t, h, w in grid:
        cos, sin = vm._rotary_pos_emb_thw(t, h, w)
        assert cos.is_cuda and sin.is_cuda


def test_rot_pos_emb_cos_sin_matches_old_repeat_chain():
    """The new pre-computed-cos/sin path must produce the same per-token
    (cos, sin) after the forward's ``.repeat(1, 2)`` as the prior
    ``freqs.repeat(1, 2).cos()/.sin()`` chain."""
    vm = _make_qwen3_vision_model_for_l2()
    vm._rotary_pos_emb_thw.cache_clear()

    grid_thw_list = [(1, 16, 16), (1, 32, 32), (3, 8, 12)]

    # New path: gathered cos/sin then forward's ``.repeat(1, 2)``.
    cos_new, sin_new = vm.rot_pos_emb(grid_thw_list)
    cos_new = cos_new.repeat(1, 2)
    sin_new = sin_new.repeat(1, 2)

    # Old path replica: build freqs via outer on the fly, then
    # ``flatten(1).repeat(1, 2).cos()/.sin()``.
    rope_dim = 36
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    inv_freq = inv_freq.to("cuda")
    cos_pieces, sin_pieces = [], []
    for t, h, w in grid_thw_list:
        pos_ids = Qwen3VisionModel.rot_pos_ids(h, w, vm.spatial_merge_size)
        if t > 1:
            pos_ids = pos_ids.repeat(t, 1)
        pos_ids = pos_ids.to("cuda")
        max_hw = max(h, w)
        seq = torch.arange(max_hw, device="cuda", dtype=torch.float32)
        freq_table = torch.outer(seq, inv_freq)  # (max_hw, 18)
        freqs = freq_table[pos_ids].flatten(1)  # (total, 36)
        emb = freqs.repeat(1, 2)  # (total, 72)
        cos_pieces.append(emb.cos())
        sin_pieces.append(emb.sin())
    cos_old = torch.cat(cos_pieces, dim=0)
    sin_old = torch.cat(sin_pieces, dim=0)

    # The cos/sin formulas are identical; the only difference is that
    # the new path computes the outer product host-side at init, while
    # the old path computes it on-device per call. That's a single-ULP
    # fp32 gap (~1e-7).
    torch.testing.assert_close(cos_new, cos_old, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sin_new, sin_old, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Pre-computed cos/sin buffer (RotaryEmbedding.rotary_cos_sin) used by
# ``_rotary_pos_emb_thw`` to gather without firing per-forward
# ``.cos()`` / ``.sin()`` kernels.
# ---------------------------------------------------------------------------


def test_rope_cos_sin_buffers_match_hf_rotary():
    """The cos/sin cache exposed via ``rotary_pos_emb.rotary_cos_sin`` must
    bit-match HF's ``Qwen3VLVisionRotaryEmbedding(dim=36)(max_hw).cos()``
    / ``.sin()`` output."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionRotaryEmbedding as HFQwen3VLVisionRotaryEmbedding,
    )

    vm = _make_qwen3_vision_model_for_l2()
    hf = HFQwen3VLVisionRotaryEmbedding(36).to("cuda")
    atol = rtol = 1e-6
    # ``rotary_cos_sin`` is stacked along dim 1 as (max_pos, 2, freq_dim);
    # index 0 holds cos, index 1 holds sin.
    cos_cache = vm.rotary_pos_emb.rotary_cos_sin[:, 0, :]
    sin_cache = vm.rotary_pos_emb.rotary_cos_sin[:, 1, :]
    for max_hw in [16, 32, 48, 64, 100, 128]:
        ref_freqs = hf(max_hw)
        torch.testing.assert_close(cos_cache[:max_hw], ref_freqs.cos(), atol=atol, rtol=rtol)
        torch.testing.assert_close(sin_cache[:max_hw], ref_freqs.sin(), atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# argsort(window_index) equivalence with the prior scatter-assign inverse.
#
# Qwen2.5-VL's vision tower swaps
#   reverse_indices = torch.empty_like(window_index)
#   reverse_indices[window_index] = torch.arange(N, device=..., dtype=...)
# for
#   reverse_indices = torch.argsort(window_index)
# The two are mathematically the inverse permutation; argsort is one
# fused op vs the prior alloc + scatter pair.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# async_tensor_h2d helper (tensorrt_llm._utils).
# ---------------------------------------------------------------------------


def test_async_tensor_h2d_sequence_input():
    """List/tuple input -> device tensor with correct dtype and values."""
    from tensorrt_llm._utils import async_tensor_h2d

    out = async_tensor_h2d([1, 2, 3, 4], dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    assert out.is_cuda and out.dtype == torch.int32
    torch.testing.assert_close(out.cpu(), torch.tensor([1, 2, 3, 4], dtype=torch.int32))


def test_async_tensor_h2d_tensor_input():
    """CPU tensor input -> device tensor with cast applied and values
    preserved."""
    from tensorrt_llm._utils import async_tensor_h2d

    src = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64)
    out = async_tensor_h2d(src, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    assert out.is_cuda and out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), src.to(torch.float32))


def test_maybe_pin_memory_idempotent_on_already_pinned():
    """A second call on an already-pinned tensor must return it unchanged
    (same Python object) -- the new ``is_pinned()`` guard skips the
    redundant ``.pin_memory()`` dispatch."""
    from tensorrt_llm._utils import maybe_pin_memory

    t = torch.tensor([1.0, 2.0, 3.0])
    pinned = maybe_pin_memory(t)
    if not pinned.is_pinned():
        pytest.skip("prefer_pinned() is False in this environment (confidential compute mode)")
    again = maybe_pin_memory(pinned)
    assert again is pinned


@pytest.mark.parametrize("n", [16, 256, 1024])
def test_argsort_matches_scatter_inverse(n):
    """argsort(perm) must equal the scatter-assign inverse permutation."""
    torch.manual_seed(n)
    window_index = torch.randperm(n, device="cuda", dtype=torch.long)

    # Old scatter-assign path.
    old = torch.empty_like(window_index)
    old[window_index] = torch.arange(n, device="cuda", dtype=window_index.dtype)
    # New argsort path.
    new = torch.argsort(window_index)

    torch.testing.assert_close(new, old, atol=0, rtol=0)
    # Functional check: applying ``new`` to ``window_index`` is identity.
    identity = window_index[new]
    torch.testing.assert_close(
        identity, torch.arange(n, device="cuda", dtype=window_index.dtype), atol=0, rtol=0
    )


# ---------------------------------------------------------------------------
# End-to-end Qwen3VLVisionBlock forward path.
#
# The pre-EXAONE-4.5 ``apply_rotary_pos_emb_vision`` (HF) accepted
# ``cos / sin`` of shape ``(seq, head_dim)``; the post-EXAONE-4.5
# ``RotaryEmbedding.apply_rotary_pos_emb`` (TRT-LLM) expects
# ``(seq, head_dim // 2)`` and broadcasts. This test pushes one block
# through with the post-EXAONE shape so any future regression (e.g.
# re-introducing a stray ``.repeat(1, 2)`` on cos/sin) surfaces here
# rather than only at full-model CI.
# ---------------------------------------------------------------------------


def test_qwen3vl_vision_block_forward_end_to_end():
    """One ``Qwen3VLVisionBlock.forward`` call with random weights and the
    correct (post-EXAONE) ``cos/sin`` shape must not raise a broadcasting
    error in ``apply_rope``."""
    from transformers import Qwen3VLConfig

    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLVisionBlock

    hf_config = Qwen3VLConfig(
        text_config={
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 256,
            "head_dim": 64,
            "rope_scaling": {
                "mrope_section": [8, 8, 8],
                "mrope_interleaved": True,
                "rope_type": "default",
            },
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "num_key_value_heads": 4,
            "vocab_size": 1024,
            "dtype": "bfloat16",
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "model_type": "qwen3_vl_text",
        },
        vision_config={
            "hidden_size": 1152,
            "num_heads": 16,  # head_dim = 72
            "depth": 1,
            "intermediate_size": 4304,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 2304,
            "out_hidden_size": 256,
            "deepstack_visual_indexes": [],
            "hidden_act": "gelu_pytorch_tanh",
            "model_type": "qwen3_vl",
            "initializer_range": 0.02,
        },
        architectures=["Qwen3VLForConditionalGeneration"],
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )

    mc = ModelConfig(pretrained_config=hf_config, attn_backend="TRTLLM")
    block = Qwen3VLVisionBlock(mc, layer_idx=0).to("cuda", dtype=torch.bfloat16)

    seq_len = 256
    head_dim = 72
    freq_dim = head_dim // 2  # 36 -- the (post-EXAONE) cos/sin last-dim

    hidden = torch.randn(seq_len, 1152, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(seq_len, freq_dim, device="cuda", dtype=torch.float32)
    sin = torch.randn(seq_len, freq_dim, device="cuda", dtype=torch.float32)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda")

    metadata_cls = get_attention_backend("TRTLLM").Metadata
    attn_metadata = metadata_cls(
        max_num_requests=64,
        max_num_tokens=seq_len,
        kv_cache_manager=None,
    )
    attn_metadata.num_contexts = 1
    attn_metadata.request_ids = [1]
    attn_metadata.prompt_lens = [seq_len]
    attn_metadata.seq_lens = torch.tensor([seq_len], dtype=torch.int)
    attn_metadata.max_seq_len = seq_len
    attn_metadata.prepare()

    out = block(
        hidden,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
        attn_metadata=attn_metadata,
    )
    assert out.shape == hidden.shape
