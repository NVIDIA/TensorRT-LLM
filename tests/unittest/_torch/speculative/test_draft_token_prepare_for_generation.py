import math
import os
import sys
import unittest

import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.drafting_loops import StaticTreeDraftingLoopWrapper, DynamicTreeDraftingLoopWrapper
from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager, Eagle3SpecMetadata
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
from tensorrt_llm.llmapi import EagleDecodingConfig
from tensorrt_llm._torch.speculative.model_drafter import ModelDrafter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = None
        self.config = None
        self.model = {}
        self.model_is_wrapped = True

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
        spec_config = EagleDecodingConfig(
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            speculative_model_dir=eagle_model_dir,
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

        # 3) Create StaticTreeDraftingLoopWrapper
        static_tree_drafting_loop_wrapper = StaticTreeDraftingLoopWrapper(
            max_batch_size=max_batch_size,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            draft_model=DummyModel(),
        )

        # 3) Run the function
        static_tree_drafting_loop_wrapper.prepare_for_generation(
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            spec_tree_manager=spec_tree_manager,
            position_ids=input_position_ids,
        )

        # Compare input_ids and position_ids
        print(
            f"static_tree_drafting_loop_wrapper.position_ids_buffer: {static_tree_drafting_loop_wrapper.position_ids_buffer}, \
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

        assert torch.all(static_tree_drafting_loop_wrapper.position_ids_buffer == ref_position_ids)
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



def test_dynamic_tree_update_draft_tokens_and_scores():
    # Fix parameters
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"  # It will not actually be used.
    use_dynamic_tree = True
    max_new_tokens = 128
    kv_cache_manager = None
    
    def run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK,
                 cur_draft_idx, new_draft_tokens, new_draft_scores, previous_draft_scores,
                 input_spec_decoding_packed_mask, input_hidden_states_write_indices, input_hidden_states_read_indices, 
                 ref_draft_tokens_buffer, ref_history_draft_tokens_buffer, ref_history_draft_tokens_parent_buffer, 
                 ref_history_score_buffer, ref_spec_decoding_packed_mask, ref_hidden_states_read_indices):

        # 1) Create attention metadata
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=max_batch_size,
            max_num_tokens=max_new_tokens,
            kv_cache_manager=kv_cache_manager,
        )
        # Set initial values
        attn_metadata.spec_decoding_packed_mask = input_spec_decoding_packed_mask
        attn_metadata.spec_decoding_generation_lengths = torch.zeros(
            [max_batch_size],
            dtype=torch.int,
            device="cuda",
        )

        # 3) Create spec metadata
        spec_config = EagleDecodingConfig(
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            speculative_model_dir=eagle_model_dir,
            eagle3_one_model=False,
            eagle_choices=None,
            use_dynamic_tree=use_dynamic_tree,
            dynamic_tree_max_topK=dynamic_tree_max_topK,
        )
        eagle3_resource_manager = Eagle3ResourceManager(
            config=spec_config,
            dtype=torch.bfloat16,
            hidden_size=1024,
            max_num_requests=max_batch_size,
            max_seq_len=max_new_tokens,
            max_num_tokens=max_new_tokens,
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
        spec_metadata.hidden_states_write_indices = input_hidden_states_write_indices
        spec_metadata.hidden_states_read_indices = input_hidden_states_read_indices

        # 4) Create DynamicTreeDraftingLoopWrapper
        dynamic_tree_drafting_loop_wrapper = DynamicTreeDraftingLoopWrapper(
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            max_batch_size=max_batch_size,
            dynamic_tree_max_topK=dynamic_tree_max_topK,
            draft_model=DummyModel(),
        )

        # 5) Run the function
        cur_scores = dynamic_tree_drafting_loop_wrapper.update_draft_tokens_and_scores(
            cur_draft_idx=cur_draft_idx,
            batch_size=max_batch_size,
            new_draft_tokens=new_draft_tokens,
            new_draft_scores=new_draft_scores,
            previous_draft_scores=previous_draft_scores,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
        )
        
        # 5) Check the results
        print("==================================")
        print(f"======= ref_draft_tokens_buffer: {ref_draft_tokens_buffer} ========")
        print(f"======= dynamic_tree_drafting_loop_wrapper.draft_tokens_buffer: {dynamic_tree_drafting_loop_wrapper.draft_tokens_buffer} ========")
        
        print(f"======= ref_history_draft_tokens_buffer: {ref_history_draft_tokens_buffer} ========")
        print(f"======= dynamic_tree_drafting_loop_wrapper.history_draft_tokens_buffer: {dynamic_tree_drafting_loop_wrapper.history_draft_tokens_buffer} ========")
        
        print(f"======= ref_history_draft_tokens_parent_buffer: {ref_history_draft_tokens_parent_buffer} ========")
        print(f"======= dynamic_tree_drafting_loop_wrapper.history_draft_tokens_parent_buffer: {dynamic_tree_drafting_loop_wrapper.history_draft_tokens_parent_buffer} ========")
        
        print(f"======= ref_history_score_buffer: {ref_history_score_buffer} ========")
        print(f"======= dynamic_tree_drafting_loop_wrapper.history_score_buffer: {dynamic_tree_drafting_loop_wrapper.history_score_buffer} ========")
        
        print(f"======= ref_spec_decoding_packed_mask: {ref_spec_decoding_packed_mask} ========")
        print(f"======= attn_metadata.spec_decoding_packed_mask: {attn_metadata.spec_decoding_packed_mask} ========")
        
        print(f"======= ref_hidden_states_read_indices: {ref_hidden_states_read_indices} ========")
        print(f"======= spec_metadata.hidden_states_read_indices: {spec_metadata.hidden_states_read_indices} ========")


        assert torch.all(dynamic_tree_drafting_loop_wrapper.draft_tokens_buffer == ref_draft_tokens_buffer)
        assert torch.all(dynamic_tree_drafting_loop_wrapper.history_draft_tokens_buffer == ref_history_draft_tokens_buffer)
        if ref_history_draft_tokens_parent_buffer is not None:
            assert torch.all(dynamic_tree_drafting_loop_wrapper.history_draft_tokens_parent_buffer == ref_history_draft_tokens_parent_buffer)
        assert torch.allclose(dynamic_tree_drafting_loop_wrapper.history_score_buffer, ref_history_score_buffer, atol=1e-3)
        if ref_spec_decoding_packed_mask is not None:
            assert torch.all(attn_metadata.spec_decoding_packed_mask == ref_spec_decoding_packed_mask)
        if ref_hidden_states_read_indices is not None:
            assert torch.all(spec_metadata.hidden_states_read_indices == ref_hidden_states_read_indices)

    ##### CASE 1 dynamic tree, batch size = 1, cur_draft_idx = 0 #############
    max_batch_size = 1
    max_draft_len = 3
    max_total_draft_tokens = 15
    dynamic_tree_max_topK = 3
    cur_draft_idx = 0

    new_draft_tokens = torch.tensor([2, 6, 3], dtype=torch.int32, device="cuda")
    new_draft_scores = torch.tensor([0.3, 0.2, 0.1], dtype=torch.float32, device="cuda")
    previous_draft_scores = None

    input_spec_decoding_packed_mask = torch.zeros(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32), dtype=torch.int32, device="cuda")
    input_hidden_states_write_indices = torch.arange(1, max_batch_size * (max_total_draft_tokens + 1) + 1, dtype=torch.int32, device="cuda")
    input_hidden_states_read_indices = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda")

    ref_draft_tokens_buffer = torch.zeros(max_batch_size, max_total_draft_tokens + 1, dtype=torch.int32, device="cuda")
    ref_draft_tokens_buffer[:, :3] = new_draft_tokens

    ref_history_draft_tokens_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.int32, device="cuda")
    ref_history_draft_tokens_buffer[:, :3] = new_draft_tokens

    ref_history_draft_tokens_parent_buffer = torch.ones(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.int32, device="cuda") * -1
    ref_history_draft_tokens_parent_buffer[:, 0:12] = torch.tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32, device="cuda")

    ref_history_score_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.float32, device="cuda")
    ref_history_score_buffer[:, :3] = new_draft_scores

    ref_spec_decoding_packed_mask = torch.tensor([1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda").reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))
    ref_hidden_states_read_indices = None
    

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK,
             cur_draft_idx, new_draft_tokens, new_draft_scores, previous_draft_scores,
             input_spec_decoding_packed_mask, input_hidden_states_write_indices, input_hidden_states_read_indices, 
             ref_draft_tokens_buffer, ref_history_draft_tokens_buffer, ref_history_draft_tokens_parent_buffer, 
             ref_history_score_buffer, ref_spec_decoding_packed_mask, ref_hidden_states_read_indices)

             
    ##### CASE 2 dynamic tree, batch size = 1, cur_draft_idx = 1 #############
    max_batch_size = 1
    max_draft_len = 3
    max_total_draft_tokens = 15
    dynamic_tree_max_topK = 3
    cur_draft_idx = 1

    # new_draft_tokens: [[48, 47, 46], [45, 44, 43], ..., [6, 5, 4], [3, 2, 1]]
    # new_draft_scores: [[0.48, 0.47, 0.46], [0.45, 0.44, 0.43], ..., [0.06, 0.05, 0.04], [0.03, 0.02, 0.01]]
    # But the valuable draft tokens are new_draft_tokens[3:]
    new_draft_tokens = torch.arange(
        max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK, 0, -1,
        dtype=torch.int32, device="cuda"
    ).reshape(max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK)
    new_draft_scores = torch.arange(
        max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK, 0, -1,
        dtype=torch.float32, device="cuda"
    ).reshape(max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK) * 0.01
    previous_draft_scores = torch.tensor([[1.5, 0.7, 0.4]], dtype=torch.float32, device="cuda")

    input_spec_decoding_packed_mask = torch.tensor([1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda").reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))
    input_hidden_states_write_indices = torch.arange(1, max_batch_size * (max_total_draft_tokens + 1) + 1, dtype=torch.int32, device="cuda")
    input_hidden_states_read_indices = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda")

    ref_draft_tokens_buffer = torch.zeros(max_batch_size, max_total_draft_tokens + 1, dtype=torch.int32, device="cuda")
    ref_draft_tokens_buffer[:, 3:6] = torch.tensor([48, 47, 46], dtype=torch.int32, device="cuda")

    ref_history_draft_tokens_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.int32, device="cuda")
    ref_history_draft_tokens_buffer[:, 3:12] = torch.tensor([48, 47, 46, 45, 44, 43, 42, 41, 40], dtype=torch.int32, device="cuda")

    ref_history_draft_tokens_parent_buffer = torch.ones(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.int32, device="cuda") * -1
    ref_history_draft_tokens_parent_buffer[:, 12:12+9] = torch.tensor([3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=torch.int32, device="cuda")

    ref_history_score_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.float32, device="cuda")
    ref_history_score_buffer[:, 3:12] = torch.tensor([1.98, 1.97, 1.96, 1.15, 1.14, 1.13, 0.82, 0.81, 0.80], dtype=torch.float32, device="cuda")

    ref_spec_decoding_packed_mask = torch.tensor([1, 2, 4, 9, 17, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda").reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))
    ref_hidden_states_read_indices = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda")
    

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK,
             cur_draft_idx, new_draft_tokens, new_draft_scores, previous_draft_scores,
             input_spec_decoding_packed_mask, input_hidden_states_write_indices, input_hidden_states_read_indices, 
             ref_draft_tokens_buffer, ref_history_draft_tokens_buffer, ref_history_draft_tokens_parent_buffer, 
             ref_history_score_buffer, ref_spec_decoding_packed_mask, ref_hidden_states_read_indices)


    ##### CASE 2 dynamic tree, batch size = 1, cur_draft_idx = 1 #############
    max_batch_size = 1
    max_draft_len = 3
    max_total_draft_tokens = 15
    dynamic_tree_max_topK = 3
    cur_draft_idx = 2

    # new_draft_tokens: [[48, 47, 46], [45, 44, 43], ..., [6, 5, 4], [3, 2, 1]]
    # new_draft_scores: [[0.48, 0.47, 0.46], [0.45, 0.44, 0.43], ..., [0.06, 0.05, 0.04], [0.03, 0.02, 0.01]]
    # But the valuable draft tokens are new_draft_tokens[3:]
    new_draft_tokens = torch.arange(
        max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK, 0, -1,
        dtype=torch.int32, device="cuda"
    ).reshape(max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK)
    new_draft_scores = torch.arange(
        max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK, 0, -1,
        dtype=torch.float32, device="cuda"
    ).reshape(max_batch_size * (max_total_draft_tokens + 1) * dynamic_tree_max_topK) * 0.01
    previous_draft_scores = torch.tensor([[1.5, 0.7, 0.4]], dtype=torch.float32, device="cuda")

    input_spec_decoding_packed_mask = torch.tensor([1, 2, 4, 9, 17, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda").reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))
    input_hidden_states_write_indices = torch.arange(1, max_batch_size * (max_total_draft_tokens + 1) + 1, dtype=torch.int32, device="cuda")
    input_hidden_states_read_indices = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda")

    ref_draft_tokens_buffer = torch.zeros(max_batch_size, max_total_draft_tokens + 1, dtype=torch.int32, device="cuda")
    ref_draft_tokens_buffer[:, 6:9] = torch.tensor([39, 38, 37], dtype=torch.int32, device="cuda")

    ref_history_draft_tokens_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.int32, device="cuda")
    ref_history_draft_tokens_buffer[:, 12:21] = torch.tensor([39, 38, 37, 36, 35, 34, 33, 32, 31], dtype=torch.int32, device="cuda")

    ref_history_draft_tokens_parent_buffer = None # will not be updated for this layer

    ref_history_score_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.float32, device="cuda")
    ref_history_score_buffer[:, 12:21] = torch.tensor([1.89, 1.88, 1.87, 1.06, 1.05, 1.04, 0.73, 0.72, 0.71], dtype=torch.float32, device="cuda")

    ref_spec_decoding_packed_mask = torch.tensor([1, 2, 4, 9, 17, 33, 73, 137, 265, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda").reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))
    ref_hidden_states_read_indices = torch.tensor([0, 0, 0, 1, 1, 1, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda")
    

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK,
             cur_draft_idx, new_draft_tokens, new_draft_scores, previous_draft_scores,
             input_spec_decoding_packed_mask, input_hidden_states_write_indices, input_hidden_states_read_indices, 
             ref_draft_tokens_buffer, ref_history_draft_tokens_buffer, ref_history_draft_tokens_parent_buffer, 
             ref_history_score_buffer, ref_spec_decoding_packed_mask, ref_hidden_states_read_indices)


    ##### CASE 4 dynamic tree, batch size = 2, cur_draft_idx = 1 #############
    max_batch_size = 2
    max_draft_len = 3
    max_total_draft_tokens = 15
    dynamic_tree_max_topK = 3
    cur_draft_idx = 1

    # new_draft_tokens: [[48, 47, 46], [45, 44, 43], ..., [6, 5, 4], [3, 2, 1]]
    # new_draft_scores: [[0.48, 0.47, 0.46], [0.45, 0.44, 0.43], ..., [0.06, 0.05, 0.04], [0.03, 0.02, 0.01]]
    # But the valuable draft tokens are new_draft_tokens[3:]
    new_draft_tokens = torch.arange(
        (max_total_draft_tokens + 1) * dynamic_tree_max_topK, 0, -1,
        dtype=torch.int32, device="cuda"
    ).reshape((max_total_draft_tokens + 1) * dynamic_tree_max_topK).repeat(max_batch_size)
    new_draft_scores = torch.arange(
        (max_total_draft_tokens + 1) * dynamic_tree_max_topK, 0, -1,
        dtype=torch.float32, device="cuda"
    ).reshape((max_total_draft_tokens + 1) * dynamic_tree_max_topK).repeat(max_batch_size) * 0.01
    previous_draft_scores = torch.tensor([[1.5, 0.7, 0.4], [2.5, 1.7, 1.4]], dtype=torch.float32, device="cuda")

    input_spec_decoding_packed_mask = torch.tensor([[1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32, device="cuda").reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))
    input_hidden_states_write_indices = torch.arange(1, max_batch_size * (max_total_draft_tokens + 1) + 1, dtype=torch.int32, device="cuda")
    input_hidden_states_read_indices = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # req1
                                                     16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # req2
                                                     ], dtype=torch.int32, device="cuda")

    ref_draft_tokens_buffer = torch.zeros(max_batch_size, max_total_draft_tokens + 1, dtype=torch.int32, device="cuda")
    ref_draft_tokens_buffer[:, 3:6] = torch.tensor([48, 47, 46], dtype=torch.int32, device="cuda")

    ref_history_draft_tokens_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.int32, device="cuda")
    ref_history_draft_tokens_buffer[:, 3:12] = torch.tensor([48, 47, 46, 45, 44, 43, 42, 41, 40], dtype=torch.int32, device="cuda")

    ref_history_draft_tokens_parent_buffer = torch.ones(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.int32, device="cuda") * -1
    ref_history_draft_tokens_parent_buffer[:, 12:12+9] = torch.tensor([3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=torch.int32, device="cuda")

    ref_history_score_buffer = torch.zeros(max_batch_size, dynamic_tree_max_topK + dynamic_tree_max_topK * dynamic_tree_max_topK * (max_draft_len - 1), dtype=torch.float32, device="cuda")
    ref_history_score_buffer[0:, 3:12] = torch.tensor([1.98, 1.97, 1.96, 1.15, 1.14, 1.13, 0.82, 0.81, 0.80], dtype=torch.float32, device="cuda")
    ref_history_score_buffer[1:, 3:12] = torch.tensor([2.98, 2.97, 2.96, 2.15, 2.14, 2.13, 1.82, 1.81, 1.80], dtype=torch.float32, device="cuda")

    ref_spec_decoding_packed_mask = torch.tensor([1, 2, 4, 9, 17, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # req1
                                                  1, 2, 4, 9, 17, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # req2
                                                  ], dtype=torch.int32, device="cuda").reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))
    ref_hidden_states_read_indices = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # req1
                                                   16, 16, 16, 17, 17, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # req2
                                                   ], dtype=torch.int32, device="cuda")
    

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK,
             cur_draft_idx, new_draft_tokens, new_draft_scores, previous_draft_scores,
             input_spec_decoding_packed_mask, input_hidden_states_write_indices, input_hidden_states_read_indices, 
             ref_draft_tokens_buffer, ref_history_draft_tokens_buffer, ref_history_draft_tokens_parent_buffer, 
             ref_history_score_buffer, ref_spec_decoding_packed_mask, ref_hidden_states_read_indices)


def test_dynamic_tree_restruct_tree():
    # Fix parameters
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"  # It will not actually be used.
    use_dynamic_tree = True
    max_new_tokens = 128
    kv_cache_manager = None

    def run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK, 
                 cur_topk_score_indices, cur_history_draft_tokens_parent_buffer, 
                 ref_eagle_paths, ref_spec_dec_packed_mask, ref_spec_dec_position_offsets):
        # 1) Create spec metadata
        spec_config = EagleDecodingConfig(
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            speculative_model_dir=eagle_model_dir,
            eagle3_one_model=False,
            eagle_choices=None,
            use_dynamic_tree=use_dynamic_tree,
            dynamic_tree_max_topK=dynamic_tree_max_topK,
        )

        eagle3_resource_manager = Eagle3ResourceManager(
            config=spec_config,
            dtype=torch.bfloat16,
            hidden_size=1024,
            max_num_requests=max_batch_size,
            max_seq_len=max_new_tokens,
            max_num_tokens=max_new_tokens,
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

        # 2) Create model drafter
        model_drafter = ModelDrafter(
            spec_config=spec_config,
            draft_model_engine=DummyModel(),
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            draft_seq_slot_manager=None,
            sampler=None,
            spec_resource_manager=eagle3_resource_manager,
            guided_decoder=None,
        )

        # 3) Reconstruct the dynamic tree
        model_drafter.reconstruct_dynamic_tree(
            0,
            cur_topk_score_indices,
            cur_history_draft_tokens_parent_buffer,
            spec_tree_manager)

        print("==================================")
        print(f"ref_eagle_paths: {ref_eagle_paths}")
        print(f"spec_tree_manager.eagle_paths: {spec_tree_manager.eagle_paths}")

        print(f"ref_spec_dec_packed_mask: {ref_spec_dec_packed_mask}")
        print(f"spec_tree_manager.spec_dec_packed_mask: {spec_tree_manager.spec_dec_packed_mask}")

        print(f"ref_spec_dec_position_offsets: {ref_spec_dec_position_offsets}")
        print(f"spec_tree_manager.spec_dec_position_offsets: {spec_tree_manager.spec_dec_position_offsets}")

        import pdb; pdb.set_trace()

        assert torch.all(ref_eagle_paths == spec_tree_manager.eagle_paths)
        assert torch.all(ref_spec_dec_packed_mask == spec_tree_manager.spec_dec_packed_mask) 
        assert torch.all(ref_spec_dec_position_offsets == spec_tree_manager.spec_dec_position_offsets) 

    ##### CASE 1 dynamic tree, batch size = 1 #############
    max_batch_size = 1
    max_draft_len = 3
    max_total_draft_tokens = 10
    dynamic_tree_max_topK = 3

    cur_topk_score_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 12], dtype=torch.int32, device="cpu")
    cur_history_draft_tokens_parent_buffer = torch.tensor(
        [-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=torch.int32, device="cpu")

    ref_eagle_paths = torch.tensor(
        [[[ 0, -1, -1, -1],                             
         [ 0,  1, -1, -1],
         [ 0,  2, -1, -1],          
         [ 0,  3, -1, -1],                            
         [ 0,  1,  4, -1],                                                                                                                 
         [ 0,  1,  5, -1],
         [ 0,  1,  6, -1],
         [ 0,  2,  7, -1],                                                                                                                 
         [ 0,  2,  8, -1],    
         [ 0,  3,  9, -1],
         [ 0,  1,  4, 10]]], dtype=torch.int32, device='cpu', pin_memory=True,
    )

    ref_spec_dec_packed_mask = torch.tensor(
        [1, 3, 5, 9, 19, 35, 67, 133, 261, 521, 1043], dtype=torch.int32, device="cuda"
    ).reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))

    ref_spec_dec_position_offsets = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3]], dtype=torch.int32, device="cuda")

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK,
                 cur_topk_score_indices, cur_history_draft_tokens_parent_buffer, 
                 ref_eagle_paths, ref_spec_dec_packed_mask, ref_spec_dec_position_offsets)

    ##### CASE 2 dynamic tree, batch size = 1 #############
    max_batch_size = 1
    max_draft_len = 3
    max_total_draft_tokens = 9
    dynamic_tree_max_topK = 3

    cur_topk_score_indices = torch.tensor([0, 1, 2, 3, 4, 6, 12, 13, 18], dtype=torch.int32, device="cpu")
    cur_history_draft_tokens_parent_buffer = torch.tensor(
        [-1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 6, 6, 6], dtype=torch.int32, device="cpu")

    ref_eagle_paths = torch.tensor(
        [[[ 0, -1, -1, -1],                             
         [ 0,  1, -1, -1],
         [ 0,  2, -1, -1],          
         [ 0,  3, -1, -1],                            
         [ 0,  1,  4, -1],                                                                                                                 
         [ 0,  1,  5, -1],
         [ 0,  2,  6, -1],
         [ 0,  1,  4, 7],                                                                                                                 
         [ 0,  1,  4, 8],    
         [ 0,  2,  6, 9]]], dtype=torch.int32, device='cpu', pin_memory=True,
    )

    ref_spec_dec_packed_mask = torch.tensor(
        [1, 3, 5, 9, 19, 35, 69, 147, 275, 581], dtype=torch.int32, device="cuda"
    ).reshape(max_batch_size, max_total_draft_tokens + 1, math.ceil((max_total_draft_tokens + 1) / 32))

    ref_spec_dec_position_offsets = torch.tensor([[0, 1, 1, 1, 2, 2, 2, 3, 3, 3]], dtype=torch.int32, device="cuda")

    run_test(max_batch_size, max_draft_len, max_total_draft_tokens, dynamic_tree_max_topK,
                 cur_topk_score_indices, cur_history_draft_tokens_parent_buffer, 
                 ref_eagle_paths, ref_spec_dec_packed_mask, ref_spec_dec_position_offsets)


if __name__ == "__main__":
    unittest.main()
