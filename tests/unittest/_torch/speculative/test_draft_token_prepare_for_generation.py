import math
import os
import sys
import unittest

import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.drafting_loops import TreeDraftingLoopWrapper
from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager, Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
from tensorrt_llm.llmapi import Eagle3DecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = None
        self.config = None
        self.model = {}

    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass


def test_draft_token_static_tree_prepare_for_generation():
    # Fix parameters
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"  # It will not actually be used.
    use_dynamic_tree = False
    max_new_tokens = 128
    kv_cache_manager = None

    # Create related object and run test
    def run_test(
        max_batch_size,
        prepare_for_layer_idx,
        max_total_draft_tokens,
        max_draft_len,
        eagle_choices,
        input_seq_lens_cuda,
        input_kv_lens_cuda,
        input_num_accepted_draft_tokens,
        input_hidden_states_write_indices,
        input_hidden_states_read_indices,
        input_position_ids,
        ref_position_ids,
        ref_attn_metadata,
        ref_spec_metadata,
    ):
        # 1) Create attention metadata
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=max_batch_size,
            max_num_tokens=max_new_tokens,
            kv_cache_manager=kv_cache_manager,
        )

        # Set initial values
        attn_metadata._seq_lens_cuda = input_seq_lens_cuda  # set from input
        attn_metadata.kv_lens_cuda = input_kv_lens_cuda  # set from input
        attn_metadata._seq_lens = torch.zeros([max_batch_size], device="cpu")
        attn_metadata.host_request_types = torch.zeros([max_batch_size], device="cuda")
        attn_metadata.spec_decoding_position_offsets = torch.zeros(
            [max_batch_size, max_total_draft_tokens + 1],
            dtype=torch.int,
            device="cuda",
        )
        attn_metadata.spec_decoding_packed_mask = torch.zeros(
            [
                max_batch_size,
                max_total_draft_tokens + 1,
                math.ceil((max_total_draft_tokens + 1) / 32),
            ],
            dtype=torch.int,
            device="cuda",
        )
        attn_metadata.spec_decoding_generation_lengths = torch.zeros(
            [max_batch_size],
            dtype=torch.int,
            device="cuda",
        )

        # 2) Create spec metadata
        spec_config = Eagle3DecodingConfig(
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            speculative_model=eagle_model_dir,
            eagle3_one_model=False,
            eagle_choices=eagle_choices,
            use_dynamic_tree=use_dynamic_tree,
        )

        eagle3_resource_manager = Eagle3ResourceManager(
            spec_config,
            torch.bfloat16,
            1024,
            max_batch_size,
            max_new_tokens,
            max_new_tokens,
        )

        spec_tree_manager = SpecTreeManager(
            max_num_requests=max_batch_size,
            use_dynamic_tree=spec_config.use_dynamic_tree,
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            eagle_choices=spec_config.eagle_choices,
            dynamic_tree_max_topK=spec_config.dynamic_tree_max_topK,
        )

        spec_metadata = Eagle3SpecMetadata(
            max_draft_len=spec_config.max_draft_len,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_num_requests=max_batch_size,
            num_layers=32,
            hidden_size=1024,
            max_num_tokens=max_new_tokens,
            dtype=torch.bfloat16,
            is_draft_model=True,
            eagle3_resource_manager=eagle3_resource_manager,
            layers_to_capture=spec_config.eagle3_layers_to_capture,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            eagle_choices=spec_config.eagle_choices,
            is_spec_dec_tree=spec_config.eagle_choices is not None or spec_config.use_dynamic_tree,
            is_spec_dec_dynamic_tree=spec_config.use_dynamic_tree,
        )

        # Set initial values
        spec_metadata.num_accepted_draft_tokens = input_num_accepted_draft_tokens  # set from input
        spec_metadata.num_tokens = 0
        spec_metadata.hidden_states_write_indices = (
            input_hidden_states_write_indices  # set from input
        )
        spec_metadata.hidden_states_read_indices = (
            input_hidden_states_read_indices  # set from input
        )

        # 3) Create TreeDraftingLoopWrapper
        tree_drafting_loop_wrapper = TreeDraftingLoopWrapper(
            max_batch_size=max_batch_size,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            draft_model=DummyModel(),
        )

        # 3) Run the function
        tree_drafting_loop_wrapper.prepare_for_generation(
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            spec_tree_manager=spec_tree_manager,
            position_ids=input_position_ids,
        )

        # Compare input_ids and position_ids
        print(
            f"tree_drafting_loop_wrapper.position_ids_buffer: {tree_drafting_loop_wrapper.position_ids_buffer}, \
            ref_output_position_ids: {ref_position_ids}"
        )

        # Compare the attention metadata
        print(
            f"attn_metadata.kv_lens_cuda: {attn_metadata.kv_lens_cuda}, \
            ref_attn_metadata.kv_lens_cuda: {ref_attn_metadata['kv_lens_cuda']}"
        )
        print(
            f"attn_metadata._seq_lens: {attn_metadata._seq_lens}, \
            ref_attn_metadata._seq_lens: {ref_attn_metadata['_seq_lens']}"
        )
        print(
            f"attn_metadata._seq_lens_cuda: {attn_metadata._seq_lens_cuda}, \
            ref_attn_metadata._seq_lens_cuda: {ref_attn_metadata['_seq_lens_cuda']}"
        )
        print(
            f"attn_metadata.host_request_types: {attn_metadata.host_request_types}, \
            ref_attn_metadata.host_request_types: {ref_attn_metadata['host_request_types']}"
        )
        print(
            f"attn_metadata.num_contexts: {attn_metadata.num_contexts}, \
            ref_attn_metadata.num_contexts: {ref_attn_metadata['num_contexts']}"
        )
        print(
            f"attn_metadata.spec_decoding_position_offsets: {attn_metadata.spec_decoding_position_offsets}, \
            ref_attn_metadata.spec_decoding_position_offsets: {ref_attn_metadata['spec_decoding_position_offsets']}"
        )
        print(
            f"attn_metadata.spec_decoding_packed_mask: {attn_metadata.spec_decoding_packed_mask}, \
            ref_attn_metadata.spec_decoding_packed_mask: {ref_attn_metadata['spec_decoding_packed_mask']}"
        )
        print(
            f"attn_metadata.spec_decoding_generation_lengths: {attn_metadata.spec_decoding_generation_lengths}, \
            ref_attn_metadata.spec_decoding_generation_lengths: {ref_attn_metadata['spec_decoding_generation_lengths']}"
        )

        # Compare the spec metadata
        print(
            f"spec_metadata.num_tokens: {spec_metadata.num_tokens}, \
            ref_spec_metadata.num_tokens: {ref_spec_metadata['num_tokens']}"
        )
        print(
            f"spec_metadata.hidden_states_read_indices: {spec_metadata.hidden_states_read_indices}, \
            ref_spec_metadata.hidden_states_read_indices: {ref_spec_metadata['hidden_states_read_indices']}"
        )
        print(
            f"spec_metadata.hidden_states_write_indices: {spec_metadata.hidden_states_write_indices}, \
            ref_spec_metadata.hidden_states_write_indices: {ref_spec_metadata['hidden_states_write_indices']}"
        )

        assert torch.all(tree_drafting_loop_wrapper.position_ids_buffer == ref_position_ids)
        assert torch.all(attn_metadata.kv_lens_cuda == ref_attn_metadata["kv_lens_cuda"])
        assert torch.all(attn_metadata._seq_lens == ref_attn_metadata["_seq_lens"])
        assert torch.all(attn_metadata._seq_lens_cuda == ref_attn_metadata["_seq_lens_cuda"])
        assert torch.all(
            attn_metadata.host_request_types == ref_attn_metadata["host_request_types"]
        )
        assert torch.all(
            torch.tensor(attn_metadata.num_contexts)
            == torch.tensor(ref_attn_metadata["num_contexts"])
        )
        assert torch.all(
            attn_metadata.spec_decoding_generation_lengths
            == ref_attn_metadata["spec_decoding_generation_lengths"]
        )
        assert torch.all(
            attn_metadata.spec_decoding_position_offsets[:max_batch_size, :].reshape(-1)
            == ref_attn_metadata["spec_decoding_position_offsets"].reshape(-1)
        )
        assert torch.all(
            attn_metadata.spec_decoding_packed_mask[:max_batch_size, :, :].reshape(-1)
            == ref_attn_metadata["spec_decoding_packed_mask"].reshape(-1)
        )
        assert torch.all(
            torch.tensor(spec_metadata.num_tokens) == torch.tensor(ref_spec_metadata["num_tokens"])
        )

        output_hidden_states_read_indices = spec_metadata.hidden_states_read_indices[
            : max_batch_size * (max_total_draft_tokens + 1)
        ].reshape(max_batch_size, max_total_draft_tokens + 1)
        assert torch.all(
            # We do not compare the last element of the hidden_states_read_indices, because it is padding.
            output_hidden_states_read_indices[:, :-1]
            == ref_spec_metadata["hidden_states_read_indices"][:, :-1]
        )

        output_hidden_states_write_indices = spec_metadata.hidden_states_write_indices[
            : max_batch_size * (max_total_draft_tokens + 1)
        ].reshape(max_batch_size, max_total_draft_tokens + 1)
        assert torch.all(
            output_hidden_states_write_indices == ref_spec_metadata["hidden_states_write_indices"]
        )

    ##### CASE 1 static tree, batch size = 1, prefill, prepare_for_layer_idx = 1 #############
    max_batch_size = 1
    prepare_for_layer_idx = 1
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [
        [0],
        [1],
        [2],
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [2, 0],
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
    ]

    prompt_len_1 = 15
    input_position_ids = torch.arange(prompt_len_1, dtype=torch.int32, device="cuda").reshape(
        1, prompt_len_1
    )
    input_seq_lens_cuda = torch.tensor([prompt_len_1], dtype=torch.int32, device="cuda")
    input_kv_lens_cuda = torch.tensor([prompt_len_1], dtype=torch.int32, device="cuda")
    input_num_accepted_draft_tokens = torch.tensor(
        [prompt_len_1 - 1], dtype=torch.int32, device="cuda"
    )
    input_hidden_states_write_indices = torch.zeros(
        [max_new_tokens], dtype=torch.long, device="cuda"
    )
    input_hidden_states_write_indices[:prompt_len_1] = torch.arange(
        prompt_len_1, dtype=torch.long, device="cuda"
    )
    input_hidden_states_read_indices = torch.zeros(
        [max_new_tokens], dtype=torch.long, device="cuda"
    )

    ref_position_ids = torch.tensor(
        [[15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 0]], dtype=torch.int32, device="cuda"
    )

    ref_attn_metadata = {}
    # prompt_len_1 + max_total_draft_tokens + 1
    ref_attn_metadata["kv_lens_cuda"] = torch.tensor([28], dtype=torch.int32, device="cuda")

    # max_total_draft_tokens + 1
    ref_attn_metadata["_seq_lens"] = torch.tensor([13], dtype=torch.int32, device="cpu")

    # max_total_draft_tokens + 1
    ref_attn_metadata["_seq_lens_cuda"] = torch.tensor([13], dtype=torch.int32, device="cuda")

    ref_attn_metadata["host_request_types"] = torch.tensor([0], dtype=torch.int32, device="cuda")
    ref_attn_metadata["num_contexts"] = 0
    ref_attn_metadata["spec_decoding_position_offsets"] = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0], dtype=torch.int32, device="cuda"
    ).repeat(max_batch_size)  # [max_batch_size * (max_total_draft_tokens + 1)]
    ref_attn_metadata["spec_decoding_packed_mask"] = torch.tensor(
        [1, 2, 4, 9, 17, 33, 66, 130, 260, 521, 1041, 2114, 0], dtype=torch.int32, device="cuda"
    ).repeat(max_batch_size)  # [max_batch_size * (max_total_draft_tokens + 1) * 1]
    ref_attn_metadata["spec_decoding_generation_lengths"] = torch.tensor(
        [13], dtype=torch.int32, device="cuda"
    )

    ref_spec_metadata = {}
    ref_spec_metadata["num_tokens"] = 13
    ref_spec_metadata["hidden_states_read_indices"] = torch.tensor(
        [[14, 14, 14, 15, 15, 15, 16, 16, 17, 18, 19, 21, 0]], dtype=torch.int32, device="cuda"
    )  # [max_batch_size, max_total_draft_tokens + 1]
    ref_spec_metadata["hidden_states_write_indices"] = torch.tensor(
        [[15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]], dtype=torch.int32, device="cuda"
    )  # [max_batch_size, max_total_draft_tokens + 1]

    run_test(
        max_batch_size,
        prepare_for_layer_idx,
        max_total_draft_tokens,
        max_draft_len,
        eagle_choices,
        input_seq_lens_cuda,
        input_kv_lens_cuda,
        input_num_accepted_draft_tokens,
        input_hidden_states_write_indices,
        input_hidden_states_read_indices,
        input_position_ids,
        ref_position_ids,
        ref_attn_metadata,
        ref_spec_metadata,
    )

    ##### CASE 2 static tree, batch size = 2, one prefill, one decode, prepare_for_layer_idx = 1 #####
    max_batch_size = 2
    prepare_for_layer_idx = 1
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [
        [0],
        [1],
        [2],
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [2, 0],
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 0],
    ]

    prompt_len_1 = 15  # prefill
    prompt_len_2 = 18
    seq_len_2 = (
        3 + 1
    )  # accepted 2 draft tokens. For the 0-th drafter layer, the sequence length will be pad to max_draft_len + 1

    input_position_ids_1 = torch.arange(prompt_len_1, dtype=torch.int32, device="cuda").reshape(
        1, prompt_len_1
    )
    input_position_ids_2 = torch.tensor([18, 19, 20, 21], dtype=torch.int32, device="cuda").reshape(
        1, max_draft_len + 1
    )  # for target model
    input_position_ids = torch.cat([input_position_ids_1, input_position_ids_2], dim=1)

    input_seq_lens_cuda = torch.tensor([prompt_len_1, seq_len_2], dtype=torch.int32, device="cuda")
    input_kv_lens_cuda = torch.tensor(
        [prompt_len_1, prompt_len_2 + seq_len_2], dtype=torch.int32, device="cuda"
    )
    input_num_accepted_draft_tokens = torch.tensor(
        [prompt_len_1 - 1, 2], dtype=torch.int32, device="cuda"
    )  # Suppose 2 are received.
    input_hidden_states_write_indices = torch.zeros(
        [max_new_tokens], dtype=torch.long, device="cuda"
    )
    input_hidden_states_write_indices[: prompt_len_1 + seq_len_2] = torch.arange(
        prompt_len_1 + seq_len_2, dtype=torch.long, device="cuda"
    )
    input_hidden_states_read_indices = torch.zeros(
        [max_new_tokens], dtype=torch.long, device="cuda"
    )
    ref_position_ids = torch.tensor(
        [
            [15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 0],
            [21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    ref_attn_metadata = {}
    ref_attn_metadata["kv_lens_cuda"] = torch.tensor([28, 34], dtype=torch.int32, device="cuda")
    ref_attn_metadata["_seq_lens"] = torch.tensor([13, 13], dtype=torch.int32, device="cpu")
    ref_attn_metadata["_seq_lens_cuda"] = torch.tensor([13, 13], dtype=torch.int32, device="cuda")
    ref_attn_metadata["host_request_types"] = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    ref_attn_metadata["num_contexts"] = 0
    ref_attn_metadata["spec_decoding_position_offsets"] = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0], dtype=torch.int32, device="cuda"
    ).repeat(max_batch_size)
    ref_attn_metadata["spec_decoding_packed_mask"] = torch.tensor(
        [1, 2, 4, 9, 17, 33, 66, 130, 260, 521, 1041, 2114, 0], dtype=torch.int32, device="cuda"
    ).repeat(max_batch_size)
    ref_attn_metadata["spec_decoding_generation_lengths"] = torch.tensor(
        [13, 13], dtype=torch.int32, device="cuda"
    )

    ref_spec_metadata = {}
    ref_spec_metadata["num_tokens"] = 26
    ref_spec_metadata["hidden_states_read_indices"] = torch.tensor(
        [
            [14, 14, 14, 15, 15, 15, 16, 16, 17, 18, 19, 21, 0],
            [17, 17, 17, 18, 18, 18, 19, 19, 20, 21, 22, 24, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    ref_spec_metadata["hidden_states_write_indices"] = torch.tensor(
        [
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    run_test(
        max_batch_size,
        prepare_for_layer_idx,
        max_total_draft_tokens,
        max_draft_len,
        eagle_choices,
        input_seq_lens_cuda,
        input_kv_lens_cuda,
        input_num_accepted_draft_tokens,
        input_hidden_states_write_indices,
        input_hidden_states_read_indices,
        input_position_ids,
        ref_position_ids,
        ref_attn_metadata,
        ref_spec_metadata,
    )


if __name__ == "__main__":
    unittest.main()
