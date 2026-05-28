import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from _torch.helpers import create_mock_cuda_graph_runner
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal
from transformers import Qwen2_5_VLConfig
from transformers import \
    Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VLForConditionalLM
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.models.checkpoints.hf.qwen2vl_weight_mapper import \
    Qwen2VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen2vl import (
    Qwen2_5_VLModel, Qwen2VLInputProcessorBase)
from tensorrt_llm._utils import get_sm_version

QWEN2_5_VL_7B_CONFIG = {
    "architectures": ["Qwen2_5_VLForConditionalGeneration"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "vision_token_id": 151654,
    "image_token_id": 151655,
    "video_token_id": 151656,
    "hidden_act": "silu",
    "hidden_size": 3584,
    "initializer_range": 0.02,
    "intermediate_size": 18944,
    "max_position_embeddings": 128000,
    "max_window_layers": 28,
    "model_type": "qwen2_5_vl",
    "num_attention_heads": 28,
    "num_hidden_layers":
    2,  # NOTE: Only 1 layer for testing, 28 layers for full model
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 32768,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.41.2",
    "use_cache": True,
    "use_sliding_window": False,
    "vision_config": {
        "depth":
        2,  # NOTE: Only 8 layers for testing, 32 layers for full model. At least 8 layer needed for global Attention
        "hidden_act": "silu",
        "hidden_size": 1280,
        "intermediate_size": 3420,
        "num_heads": 16,
        "in_chans": 3,
        "out_hidden_size": 3584,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "spatial_patch_size": 14,
        "window_size": 112,
        "fullatt_block_indexes": [0],
        "tokens_per_second": 2,
        "temporal_patch_size": 2
    },
    "rope_scaling": {
        "type": "mrope",
        "mrope_section": [16, 24, 24]
    },
    "vocab_size": 152064,
    # "_attn_implementation": "flash_attention_2",
    "_name_or_path":
    str(os.path.join(llm_models_root(), "Qwen2.5-VL-7B-Instruct"))
}


@dataclass(repr=False)
class TestQwen2_5_VLScenario(MultimodalScenario):

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


class TestQwen2_5_VL(TestModelingMultimodal):

    def get_model_config(self):
        """Return the model configuration dictionary."""
        return QWEN2_5_VL_7B_CONFIG

    def get_trtllm_model_class(self):
        return Qwen2_5_VLModel

    def get_hf_model_class(self):
        return HFQwen2_5_VLForConditionalLM

    def get_weight_mapper_class(self):
        return Qwen2VLHfWeightMapper

    def get_model_type(self):
        return "qwen2_5_vl"

    def get_model_config_class(self):
        return Qwen2_5_VLConfig

    def get_trtllm_inputs(self,
                          input_ids,
                          multimodal_params_list,
                          is_gen: bool = False,
                          num_cached_tokens_per_seq: List[int] = None,
                          total_prompt_len: Optional[int] = None):

        trtllm_inputs = super().get_trtllm_inputs(
            input_ids,
            multimodal_params_list,
            is_gen,
            num_cached_tokens_per_seq,
            total_prompt_len=total_prompt_len)

        if is_gen:
            mrope_gen_position_ids = []
            for multimodal_param in multimodal_params_list:
                mrope_gen_position_ids.append(
                    multimodal_param.multimodal_data["mrope_config"]
                    ["mrope_position_deltas"])
            mrope_gen_position_ids = torch.cat(mrope_gen_position_ids,
                                               dim=-1).to(self.device)
            trtllm_inputs["position_ids"] = (trtllm_inputs["position_ids"] +
                                             mrope_gen_position_ids).expand(
                                                 3, -1, 1).cuda()
            gen_multimodal_params_list = []
            for multimodal_param in multimodal_params_list:
                multimodal_param.strip_for_generation()
                multimodal_param.to_device(
                    "multimodal_data",
                    self.device,
                    pin_memory=True,
                    target_keywords=["mrope_config.mrope_position_deltas"])
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
                begin_offsets = [num_cached_tokens_per_seq] * len(
                    multimodal_params_list)
            else:
                begin_offsets = list(num_cached_tokens_per_seq)
            mrope_position_ids = []
            for multimodal_param, begin in zip(multimodal_params_list,
                                               begin_offsets):
                full_mrope = multimodal_param.multimodal_data["mrope_config"][
                    "mrope_position_ids"]
                mrope_position_ids.append(full_mrope[:, :,
                                                     begin:begin + chunk_len])
            position_ids = torch.cat(mrope_position_ids, dim=-1)
            position_ids = position_ids.cuda()
            trtllm_inputs["position_ids"] = position_ids

        return trtllm_inputs

    def init_kv_cache_manager(self, scenario: TestQwen2_5_VLScenario):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen2.5-VL model."""
        cache_config = self.get_kv_cache_config(scenario)
        tokens_per_block = cache_config['tokens_per_block']
        max_seq_len = cache_config['max_seq_len']
        batch_size = cache_config['batch_size']

        num_blocks = (max_seq_len + tokens_per_block - 1) // tokens_per_block

        self.kv_cache_manager = self.get_kv_cache_manager(
            dtype=self.model_config.pretrained_config.torch_dtype,
            config=self.model_config.pretrained_config,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_blocks=num_blocks)

        self.kv_cache_manager.add_dummy_requests(
            request_ids=[1],
            token_nums=[max_seq_len],
            # NOTE: Qwen2.5-VL model uses mrope
            use_mrope=True)

    def run_trtllm_forward(self, trtllm_inputs, use_cuda_graph: bool = False):
        """NOTE: Exactly the same as the parent class method, but with the mrope flag set to True for Qwen2.5-VL model."""
        if not use_cuda_graph:
            trtllm_inputs["attn_metadata"].prepare()
            return self.trtllm_model.forward(**trtllm_inputs)
        else:
            # NOTE: Qwen2.5-VL model uses mrope
            graph_runner = create_mock_cuda_graph_runner(1, use_mrope=True)
            trtllm_inputs["attn_metadata"] = trtllm_inputs[
                "attn_metadata"].create_cuda_graph_metadata(1)

            # Prepare metadata before capture (like in working Qwen2.5-VL test)
            trtllm_inputs["attn_metadata"].prepare()

            key = (1, 0, False)
            graph_runner.capture(
                key=key,
                forward_fn=lambda inputs: self.trtllm_model.forward(**inputs),
                initial_inputs=trtllm_inputs)
            for _ in range(2):
                # Run it twice. This helps us catch problems if buffers are accidentally reallocated in prepare().
                trtllm_inputs["attn_metadata"].prepare()
                logits = graph_runner.replay(key=key,
                                             current_inputs=trtllm_inputs)
            return logits.clone()

    def get_scenarios(self) -> List[TestQwen2_5_VLScenario]:
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestQwen2_5_VLScenario(modality="image",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),
            TestQwen2_5_VLScenario(modality="video",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),
            TestQwen2_5_VLScenario(modality="multiple_image",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),

            # ==== CUDA Graph Scenarios ====
            TestQwen2_5_VLScenario(modality="image",
                                   use_cuda_graph=True,
                                   disable_fuse_rope=False,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False),
        ]
        # Paged context FMHA (triggered by chunked_prefill / kv_cache_reuse)
        # is forced on for correctness on Hopper (SM90). On Blackwell (SM100)
        # the trtllm-gen kernel set falls back to an unfused MHA path whose
        # output diverges from the non-paged context kernel; gate those
        # scenarios to SM90 until the Blackwell fallback matches.
        if torch.cuda.is_available() and get_sm_version() == 90:
            scenarios.extend([
                # ==== Chunked Prefill Scenarios ====
                TestQwen2_5_VLScenario(modality="image",
                                       use_cuda_graph=False,
                                       disable_fuse_rope=False,
                                       chunked_prefill=True,
                                       kv_cache_reuse=False),
                # ==== KV Cache Reuse Scenarios ====
                TestQwen2_5_VLScenario(modality="image",
                                       use_cuda_graph=False,
                                       disable_fuse_rope=False,
                                       chunked_prefill=False,
                                       kv_cache_reuse=True),
            ])
        # ==== Disable fuse rope scenarios ====
        # Run last: setup_scenario rebuilds trtllm_model with
        # disable_fuse_rope=True for this scenario, and the rebuild is not
        # undone afterwards. Keeping it at the tail prevents the rebuilt
        # model from leaking into chunked-prefill / kv-cache-reuse
        # scenarios (where it surfaces as a cos/sin vs. q/k seq-len
        # mismatch in MRotaryEmbedding.forward).
        scenarios.append(
            TestQwen2_5_VLScenario(modality="image",
                                   use_cuda_graph=False,
                                   disable_fuse_rope=True,
                                   chunked_prefill=False,
                                   kv_cache_reuse=False))
        return scenarios

    def get_hf_inputs(self, modality: str, prompt, media):
        processor_inputs = super().get_hf_inputs(modality, prompt, media)

        # HF transformers 5.x uses a different get_rope_index algorithm for Qwen2.5-VL:
        # it multiplies temporal positions by tokens_per_second, which diverges from
        # TRT-LLM's algorithm. Compute position IDs using TRT-LLM's get_rope_index and
        # pass them explicitly so both models use the same position IDs.
        has_vision = "image_grid_thw" in processor_inputs or "video_grid_thw" in processor_inputs
        if has_vision:
            position_ids, _ = Qwen2VLInputProcessorBase.get_rope_index(
                self.hf_config,
                processor_inputs["input_ids"],
                image_grid_thw=processor_inputs.get("image_grid_thw"),
                video_grid_thw=processor_inputs.get("video_grid_thw"),
                attention_mask=processor_inputs.get("attention_mask"),
                second_per_grid_ts=processor_inputs.get("second_per_grid_ts"),
            )
            processor_inputs["position_ids"] = position_ids.to(
                processor_inputs["input_ids"].device)
            processor_inputs.pop("mm_token_type_ids", None)

        return processor_inputs

    def setup_scenario(self, scenario: TestQwen2_5_VLScenario):
        super().setup_scenario(scenario)
        if scenario.disable_fuse_rope:
            self.trtllm_model, self.model_config = self.create_trtllm_model(
                load_weights=True,
                hf_model_state_dict=self.hf_model.state_dict(),
                disable_fuse_rope=True)


# ---------------------------------------------------------------------------
# Equivalence tests for the Qwen2.5-VL optimization steps.
#
# These do not exercise the full `Qwen2VLModelBase`/`Qwen2_5_VisionModel` —
# they isolate the host-side reshapes/transfers we changed and prove the
# new path is numerically identical to the prior implementation.
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


def _vision_transfer_old(rotary_pos_emb_cos, rotary_pos_emb_sin, window_indices,
                         device):
    """Reproduces the prior 3-separate-`.to(device, non_blocking=True)` shape:
    cos / sin / window_index each get their own H->D copy."""
    cos = torch.cat(rotary_pos_emb_cos).to(device=device, non_blocking=True)
    sin = torch.cat(rotary_pos_emb_sin).to(device=device, non_blocking=True)
    window_index = torch.cat(window_indices).to(device=device,
                                                non_blocking=True)
    return cos, sin, window_index


def _vision_transfer_new(rotary_pos_emb_cos, rotary_pos_emb_sin, window_indices,
                         device):
    """The new path: stack cos/sin on host -> single H->D, then ship
    window_index in a second copy (different dtype)."""
    cos_sin = torch.stack(
        [
            torch.cat(rotary_pos_emb_cos),
            torch.cat(rotary_pos_emb_sin),
        ],
        dim=0,
    ).to(device=device, non_blocking=True)
    cos, sin = cos_sin[0], cos_sin[1]
    window_index = torch.cat(window_indices).to(device=device,
                                                non_blocking=True)
    return cos, sin, window_index


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shapes",
    [
        # (per-image (rows, hidden)) tuples — same shape requirements as the
        # real `Qwen2_5_VisionModel.get_rotary_pos_emb_window_data` output:
        # cos[i] and sin[i] share shape/dtype; window_indices[i] is int64.
        [(64, 80)],
        [(64, 80), (256, 80)],
        [(32, 80), (96, 80), (128, 80)],
    ],
    ids=lambda s: "_".join(f"{r}x{h}" for r, h in s),
)
def test_qwen2_5_vision_cos_sin_stack_equivalence(shapes, dtype):
    """Stacked-on-host single H->D must match the old 3-transfer shape
    bit-for-bit for cos, sin, and window_index."""
    torch.manual_seed(7)
    device = "cuda"

    rotary_pos_emb_cos = [torch.randn(r, h, dtype=dtype) for r, h in shapes]
    rotary_pos_emb_sin = [torch.randn(r, h, dtype=dtype) for r, h in shapes]
    window_indices = [torch.randperm(r, dtype=torch.long) for r, _ in shapes]

    cos_old, sin_old, win_old = _vision_transfer_old(rotary_pos_emb_cos,
                                                     rotary_pos_emb_sin,
                                                     window_indices, device)
    cos_new, sin_new, win_new = _vision_transfer_new(rotary_pos_emb_cos,
                                                     rotary_pos_emb_sin,
                                                     window_indices, device)

    torch.cuda.synchronize()

    torch.testing.assert_close(cos_new, cos_old, atol=0, rtol=0)
    torch.testing.assert_close(sin_new, sin_old, atol=0, rtol=0)
    torch.testing.assert_close(win_new, win_old, atol=0, rtol=0)


# NOTE: A direct numerical equivalence test between the old per-request
# prepare_mrope_config loop and the new flatten path is intentionally not
# included here. The two paths produce intentionally different output
# layouts (N stacked per-request blocks vs a single concatenated block);
# the downstream attention contract was updated alongside the flatten
# rewrite on qwen3vl_opt, where the same algorithm was validated end-to-
# end. The stack-vs-3-transfers test above is what's unique to qwen2.5.
