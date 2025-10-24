import os
import sys
import unittest

import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        SamplingConfig)
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.speculative.spec_tree_manager import SpecTreeManager
from tensorrt_llm.llmapi import EagleDecodingConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def run_test(eagle_model_dir, max_seq_len, beam_width, use_dynamic_tree,
             max_new_tokens, max_batch_size, input_request, input_new_tokens,
             draft_layer_id, max_total_draft_tokens, max_draft_len,
             eagle_choices, ref_num_accepted_draft_tokens, ref_mtokens):
    spec_config = EagleDecodingConfig(
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        speculative_model_dir=eagle_model_dir,
        eagle3_one_model=False,
        eagle_choices=eagle_choices,
        use_dynamic_tree=use_dynamic_tree,
    )
    spec_tree_manager = SpecTreeManager(
        max_num_requests=max_batch_size,
        use_dynamic_tree=spec_config.use_dynamic_tree,
        max_draft_len=spec_config.max_draft_len,
        max_total_draft_tokens=spec_config.max_total_draft_tokens,
        eagle_choices=spec_config.eagle_choices,
        dynamic_tree_max_topK=spec_config.dynamic_tree_max_topK,
    )
    spec_tree_manager.cur_draft_layer_idx = draft_layer_id
    torch_sampler = TorchSampler(
        TorchSampler.Args(
            max_seq_len=max_seq_len,
            max_draft_len=spec_config.max_draft_len,
            max_total_draft_tokens=spec_config.max_total_draft_tokens,
            max_num_sequences=max_batch_size,
            max_beam_width=beam_width,
        ))

    input_new_tokens_list = input_new_tokens.tolist()
    num_accepted_draft_tokens = torch_sampler._process_draft_tokens_tree(
        request=input_request,
        new_tokens_tensor=input_new_tokens,
        new_tokens_list=input_new_tokens_list,
        spec_tree_manager=spec_tree_manager)

    print(f"num_accepted_draft_tokens: {num_accepted_draft_tokens}")
    print(f"ref_num_accepted_draft_tokens: {ref_num_accepted_draft_tokens}")
    print(f"input_request.get_tokens(0): {input_request.get_tokens(0)}")
    print(f"ref_mtokens: {ref_mtokens}")

    # For draft model, no tokens will be accepted.
    assert num_accepted_draft_tokens == ref_num_accepted_draft_tokens

    # Check mtokens by calling get_tokens()
    assert input_request.get_tokens(0) == ref_mtokens


# For the draft model, we will not do the tree verification logic, but only add the draft tokens of the previous layer.
def test_static_tree_verification_for_draft_model():
    # Fix parameters
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"  # It will not actually be used.
    max_seq_len = 1024
    beam_width = 1
    use_dynamic_tree = False
    max_new_tokens = 128
    max_batch_size = 1  # Each request will call tree_verification separately, so batch_size is always 1.

    # We dn not need to test the case of draft_layer_id = 0, because _update_requests() is one step delay.
    # And we do not need to extract the root node of in the draft_layer_id = 0.

    ################## CASE 1 static tree, draft model's request, draft_layer_id = 1 ##########################
    draft_layer_id = 0
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    input_request = LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=max_new_tokens,
        input_tokens=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )
    # shape: [max_total_draft_tokens + 1, max_batch_size, beam_width]
    input_new_tokens = torch.zeros(max_total_draft_tokens + 1,
                                   max_batch_size,
                                   beam_width,
                                   dtype=torch.int,
                                   device='cpu')
    input_new_tokens[:3, 0,
                     0] = torch.tensor([20, 21, 22
                                        ])  # The draft tokens of the 1st layer.

    ref_mtokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22]

    run_test(eagle_model_dir=eagle_model_dir,
             max_seq_len=max_seq_len,
             beam_width=beam_width,
             use_dynamic_tree=use_dynamic_tree,
             max_new_tokens=max_new_tokens,
             max_batch_size=max_batch_size,
             input_request=input_request,
             input_new_tokens=input_new_tokens,
             draft_layer_id=draft_layer_id,
             max_draft_len=max_draft_len,
             max_total_draft_tokens=max_total_draft_tokens,
             eagle_choices=eagle_choices,
             ref_num_accepted_draft_tokens=0,
             ref_mtokens=ref_mtokens)

    ################## CASE 2 static tree, draft model's request, draft_layer_id = 2 ##########################
    draft_layer_id = 1
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    input_request = LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=max_new_tokens,
        input_tokens=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )
    # shape: [max_total_draft_tokens + 1, max_batch_size, beam_width]
    input_new_tokens = torch.zeros(max_total_draft_tokens + 1,
                                   max_batch_size,
                                   beam_width,
                                   dtype=torch.int,
                                   device='cpu')
    input_new_tokens[:6, 0,
                     0] = torch.tensor([20, 21, 22, 23, 24, 25
                                        ])  # The draft tokens of the 2nd layer.

    ref_mtokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 24, 25]

    run_test(eagle_model_dir=eagle_model_dir,
             max_seq_len=max_seq_len,
             beam_width=beam_width,
             use_dynamic_tree=use_dynamic_tree,
             max_new_tokens=max_new_tokens,
             max_batch_size=max_batch_size,
             input_request=input_request,
             input_new_tokens=input_new_tokens,
             draft_layer_id=draft_layer_id,
             max_draft_len=max_draft_len,
             max_total_draft_tokens=max_total_draft_tokens,
             eagle_choices=eagle_choices,
             ref_num_accepted_draft_tokens=0,
             ref_mtokens=ref_mtokens)

    ################## CASE 3 static tree, draft model's request, draft_layer_id = 3 ##########################
    draft_layer_id = 2
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    input_request = LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=max_new_tokens,
        input_tokens=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )
    # shape: [max_total_draft_tokens + 1, max_batch_size, beam_width]
    input_new_tokens = torch.zeros(max_total_draft_tokens + 1,
                                   max_batch_size,
                                   beam_width,
                                   dtype=torch.int,
                                   device='cpu')
    input_new_tokens[:3, 0,
                     0] = torch.tensor([26, 27, 28
                                        ])  # The draft tokens of the 3rd layer.

    ref_mtokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 26, 27, 28]

    run_test(eagle_model_dir=eagle_model_dir,
             max_seq_len=max_seq_len,
             beam_width=beam_width,
             use_dynamic_tree=use_dynamic_tree,
             max_new_tokens=max_new_tokens,
             max_batch_size=max_batch_size,
             input_request=input_request,
             input_new_tokens=input_new_tokens,
             draft_layer_id=draft_layer_id,
             max_draft_len=max_draft_len,
             max_total_draft_tokens=max_total_draft_tokens,
             eagle_choices=eagle_choices,
             ref_num_accepted_draft_tokens=0,
             ref_mtokens=ref_mtokens)


# For the target model, we will do the tree verification logic.
def test_static_tree_verification_for_target_model():
    # Fix parameters
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"  # It will not actually be used.
    max_seq_len = 1024
    beam_width = 1
    use_dynamic_tree = False
    max_new_tokens = 128
    max_batch_size = 1  # Each request will call tree_verification separately, so batch_size is always 1.

    ################## CASE 1 static tree, target model's request, no draft tokens are accepted ##########################
    draft_layer_id = 1  # Not be used.
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    input_request = LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=max_new_tokens,
        input_tokens=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )
    input_request.py_draft_tokens = torch.tensor(
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        dtype=torch.int,
        device='cpu')  # all draft tokens
    # shape: [max_total_draft_tokens + 1, max_batch_size, beam_width]
    input_new_tokens = torch.tensor(
        [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 0],
        dtype=torch.int,
        device='cpu').reshape(
            (max_total_draft_tokens + 1, max_batch_size, beam_width))

    ref_mtokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                   31]  # Only accept the top-1 of the first draft layer.
    ref_num_accepted_draft_tokens = 0

    run_test(eagle_model_dir=eagle_model_dir,
             max_seq_len=max_seq_len,
             beam_width=beam_width,
             use_dynamic_tree=use_dynamic_tree,
             max_new_tokens=max_new_tokens,
             max_batch_size=max_batch_size,
             input_request=input_request,
             input_new_tokens=input_new_tokens,
             draft_layer_id=draft_layer_id,
             max_draft_len=max_draft_len,
             max_total_draft_tokens=max_total_draft_tokens,
             eagle_choices=eagle_choices,
             ref_num_accepted_draft_tokens=ref_num_accepted_draft_tokens,
             ref_mtokens=ref_mtokens)

    ################## CASE 2 static tree, target model's request, only one path is accepted, not the longest one ##########################
    draft_layer_id = 1  # Not be used.
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    input_request = LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=max_new_tokens,
        input_tokens=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )
    input_request.py_draft_tokens = torch.tensor(
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        dtype=torch.int,
        device='cpu')  # all draft tokens, [max_total_draft_tokens]
    # shape: [max_total_draft_tokens + 1, max_batch_size, beam_width]
    input_new_tokens = torch.tensor(
        [11, 15, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112
         ],  # [max_total_draft_tokens + 1]
        dtype=torch.int,
        device='cpu').reshape(
            (max_total_draft_tokens + 1, max_batch_size, beam_width))

    ref_mtokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 15, 105]
    ref_num_accepted_draft_tokens = 2

    run_test(eagle_model_dir=eagle_model_dir,
             max_seq_len=max_seq_len,
             beam_width=beam_width,
             use_dynamic_tree=use_dynamic_tree,
             max_new_tokens=max_new_tokens,
             max_batch_size=max_batch_size,
             input_request=input_request,
             input_new_tokens=input_new_tokens,
             draft_layer_id=draft_layer_id,
             max_draft_len=max_draft_len,
             max_total_draft_tokens=max_total_draft_tokens,
             eagle_choices=eagle_choices,
             ref_num_accepted_draft_tokens=ref_num_accepted_draft_tokens,
             ref_mtokens=ref_mtokens)

    ################## CASE 3 static tree, target model's request, only one path is accepted, which is also the longest one ##########################
    draft_layer_id = 1  # Not be used.
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    input_request = LlmRequest(
        request_id=0,
        seq_slot=0,
        max_new_tokens=max_new_tokens,
        input_tokens=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
        is_streaming=False,
    )
    input_request.py_draft_tokens = torch.tensor(
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        dtype=torch.int,
        device='cpu')  # all draft tokens, [max_total_draft_tokens]
    # shape: [max_total_draft_tokens + 1, max_batch_size, beam_width]
    input_new_tokens = torch.tensor(
        [11, 14, 102, 103, 20, 105, 106, 107, 108, 109, 110, 111, 112
         ],  # [max_total_draft_tokens + 1]
        dtype=torch.int,
        device='cpu').reshape(
            (max_total_draft_tokens + 1, max_batch_size, beam_width))

    ref_mtokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 20, 110]
    ref_num_accepted_draft_tokens = 3

    run_test(eagle_model_dir=eagle_model_dir,
             max_seq_len=max_seq_len,
             beam_width=beam_width,
             use_dynamic_tree=use_dynamic_tree,
             max_new_tokens=max_new_tokens,
             max_batch_size=max_batch_size,
             input_request=input_request,
             input_new_tokens=input_new_tokens,
             draft_layer_id=draft_layer_id,
             max_draft_len=max_draft_len,
             max_total_draft_tokens=max_total_draft_tokens,
             eagle_choices=eagle_choices,
             ref_num_accepted_draft_tokens=ref_num_accepted_draft_tokens,
             ref_mtokens=ref_mtokens)


if __name__ == "__main__":
    unittest.main()
