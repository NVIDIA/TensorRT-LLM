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


def test_draft_token_static_tree_sampling():
    # Fix parameters
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"  # It will not actually be used.
    beam_width = 1
    use_dynamic_tree = False
    max_new_tokens = 128

    # Create related object and run test
    def run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
                 max_draft_len, eagle_choices, logits, ref_new_tokens,
                 num_new_draft_tokens):
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
                max_draft_len=spec_config.max_draft_len,
                max_seq_len=1024,
                max_total_draft_tokens=spec_config.max_total_draft_tokens,
                max_num_sequences=max_batch_size,
                max_beam_width=beam_width))

        # Prepare tree sampling inputs
        scheduled_requests = []
        for req_id in range(max_batch_size):
            req = LlmRequest(
                request_id=req_id,
                max_new_tokens=max_new_tokens,
                input_tokens=range(req_id * 10, (req_id + 1) * 10),
                sampling_config=SamplingConfig(
                    SamplingParams()._get_sampling_config()),
                is_streaming=False,
            )
            scheduled_requests.append(req)
        seq_slots = torch.tensor(range(max_batch_size),
                                 dtype=torch.int64,
                                 device='cuda')
        new_tokens = torch.zeros((spec_config.max_total_draft_tokens + 1,
                                  max_batch_size, beam_width),
                                 dtype=torch.int,
                                 device='cuda')

        model_outputs = {
            "logits": logits,
        }

        new_tokens_host = torch_sampler._tree_sampling_batch(
            requests=scheduled_requests,
            max_num_sequences=max_batch_size,
            seq_slots=seq_slots,
            model_outputs=model_outputs,
            spec_tree_manager=spec_tree_manager)
        new_tokens_host = new_tokens_host.squeeze(dim=-1).transpose(
            0, 1)  # shape: [max_batch_size, max_total_draft_tokens + 1]

        print(
            f"new_tokens_host.shape: {new_tokens_host.shape}, new_tokens_host: {new_tokens_host}"
        )
        print(
            f"ref_new_tokens.shape: {ref_new_tokens.shape}, ref_new_tokens: {ref_new_tokens}"
        )

        assert torch.all(new_tokens_host[:, :num_new_draft_tokens] ==
                         ref_new_tokens[:, :num_new_draft_tokens])

    ################## CASE 1 static tree, batch size = 1, draft_layer_id = 0 ##########################
    max_batch_size = 1
    draft_layer_id = 0
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    logits = torch.tensor(
        [
            [0.1, 1.1, 0.3, 0.4, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top3 indices = [4, 1, 9]
        ],
        device='cuda')
    ref_new_tokens = torch.tensor(
        [
            [4, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 3
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)

    ################## CASE 2 static tree, batch size = 1, draft_layer_id = 1 ##########################
    max_batch_size = 1
    draft_layer_id = 1
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]

    logits = torch.tensor(
        [
            [0.1, 1.1, 0.3, 0.4, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top3 indices = [4, 1, 9]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 1.2, 0.9, 1.0
             ],  # top2 indices = [7, 1]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2
             ],  # top1 indices = [9]
        ],
        device='cuda')
    ref_new_tokens = torch.tensor(
        [
            [4, 1, 9, 7, 1, 9, 0, 0, 0, 0, 0, 0, 0],
        ], device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 6
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)

    ################## CASE 3 static tree, batch size = 1, draft_layer_id = 2 ##########################
    max_batch_size = 1
    draft_layer_id = 2
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]
    logits = torch.tensor(
        [
            [0.1, 1.1, 0.3, 0.4, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top1 indices = [4]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 1.2, 0.9, 1.0
             ],  # top1 indices = [7]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2
             ],  # top1 indices = [9]
        ],
        device='cuda')
    ref_new_tokens = torch.tensor(
        [
            [4, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 3
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)

    ################## CASE 4 static tree, batch size = 2, draft_layer_id = 0 ##########################
    max_batch_size = 2
    draft_layer_id = 0
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]
    logits = torch.tensor(
        [
            [0.1, 1.1, 0.3, 0.4, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top3 indices = [4, 1, 9]
            [0.1, 0.3, 1.1, 0.4, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top3 indices = [4, 2, 9]
        ],
        device='cuda')
    ref_new_tokens = torch.tensor(
        [
            [4, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 3
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)

    ################## CASE 5 static tree, batch size = 2, draft_layer_id = 1 ##########################
    max_batch_size = 2
    draft_layer_id = 1
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]
    logits = torch.tensor(
        [
            [0.1, 1.1, 0.3, 0.4, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top3 indices = [4, 1, 9]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 1.2, 0.9, 1.0
             ],  # top2 indices = [7, 1]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2
             ],  # top1 indices = [9]
            [0.1, 1.1, 0.3, 0.4, 0.6, 1.2, 0.7, 0.8, 1.0, 0.9
             ],  # top3 indices = [5, 1, 8]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 1.2, 0.8, 0.9, 1.0
             ],  # top2 indices = [6, 1]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.2, 1.0
             ],  # top1 indices = [8]
        ],
        device='cuda')
    ref_new_tokens = torch.tensor(
        [
            [4, 1, 9, 7, 1, 9, 0, 0, 0, 0, 0, 0, 0],
            [5, 1, 8, 6, 1, 8, 0, 0, 0, 0, 0, 0, 0],
        ],
        device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 6
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)

    ################## CASE 6 static tree, batch size = 2, draft_layer_id = 2 ##########################
    max_batch_size = 2
    draft_layer_id = 2
    max_total_draft_tokens = 12
    max_draft_len = 3
    eagle_choices = [[0], [1], [2], [0, 0], [0, 1], [0, 2], [1, 0], [1, 1],
                     [2, 0], [0, 0, 0], [0, 1, 1], [1, 0, 0]]
    logits = torch.tensor(
        [
            [0.1, 1.1, 0.3, 0.4, 1.2, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top1 indices = [4]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 1.2, 0.9, 1.0
             ],  # top1 indices = [7]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2
             ],  # top1 indices = [9]
            [0.1, 1.1, 0.3, 0.4, 0.6, 1.2, 0.7, 0.8, 0.9, 1.0
             ],  # top1 indices = [5]
            [0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.2, 1.0
             ],  # top1 indices = [8]
            [1.2, 0.1, 1.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0
             ],  # top1 indices = [0]
        ],
        device='cuda')
    ref_new_tokens = torch.tensor(
        [
            [4, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 3
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)

    ################## CASE 7 static tree, batch size = 1, draft_layer_id = 0, bigger tree ##########################
    max_batch_size = 1
    draft_layer_id = 0
    max_total_draft_tokens = 63
    max_draft_len = 4
    eagle_choices = [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2],
                     [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5],
                     [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8],
                     [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0],
                     [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5],
                     [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0],
                     [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0],
                     [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2],
                     [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0],
                     [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]  # mc_sim_7b_63

    logits = torch.tensor([[
        1.1, 1.9, 0.7, 0.3, 4.4, 4.3, 4.9, 1.2, 3.9, 4.7, 1.4, 2.5, 3.2, 0.8,
        2.0, 3.0, 1.8, 2.3, 4.2, 1.3
    ]],
                          device='cuda')
    ref_new_tokens = torch.tensor(
        [
            [
                6, 9, 4, 5, 18, 8, 12, 15, 11, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0
            ],
        ],
        device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 10
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)

    ################## CASE 8 static tree, batch size = 1, draft_layer_id = 0, bigger tree ##########################
    max_batch_size = 1
    draft_layer_id = 1
    max_total_draft_tokens = 63
    max_draft_len = 4
    eagle_choices = [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2],
                     [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5],
                     [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8],
                     [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0],
                     [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5],
                     [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0],
                     [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0],
                     [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2],
                     [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0],
                     [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]  # mc_sim_7b_63

    logits = torch.tensor([
        [
            3.8, 2.5, 1.9, 0.2, 4.5, 0.9, 0.5, 3.9, 4.0, 2.9, 2.7, 0.7, 2.8,
            2.1, 1.2, 1.4, 3.3, 3.7, 2.4, 5.4
        ],
        [
            3.2, 5.4, 0.1, 4.6, 1.9, 4.8, 3.8, 0.9, 5.3, 3.7, 4.4, 0.4, 4.3,
            1.3, 0.3, 3.6, 2.5, 6.0, 3.1, 3.9
        ],
        [
            4.1, 2.8, 3.9, 4.2, 5.1, 5.8, 3.7, 1.5, 0.8, 1.4, 2.2, 0.0, 1.9,
            2.5, 3.2, 0.4, 3.5, 5.0, 2.9, 4.3
        ],
        [
            0.7, 1.4, 0.6, 2.0, 3.0, 0.9, 1.7, 0.0, 3.4, 2.8, 4.7, 3.6, 5.6,
            4.2, 1.1, 4.5, 1.2, 1.3, 3.9, 3.2
        ],
        [
            3.6, 5.5, 4.0, 6.0, 4.7, 2.3, 5.8, 0.8, 3.2, 0.3, 4.8, 2.1, 0.0,
            3.1, 0.1, 2.5, 2.2, 5.0, 3.4, 3.9
        ],
        [
            3.3, 3.8, 2.5, 2.4, 6.0, 5.5, 3.6, 0.6, 1.8, 2.6, 4.7, 1.2, 3.9,
            4.0, 4.8, 5.0, 1.1, 1.9, 0.4, 0.3
        ],
        [
            3.1, 1.7, 0.0, 4.6, 3.7, 3.2, 2.8, 4.7, 2.7, 3.8, 6.0, 4.2, 1.1,
            3.4, 5.5, 2.1, 2.6, 1.8, 0.1, 1.3
        ],
        [
            3.6, 5.1, 2.3, 5.3, 3.5, 1.8, 2.7, 2.4, 3.2, 3.0, 4.8, 5.9, 2.9,
            1.4, 5.7, 0.9, 1.0, 1.3, 2.0, 3.9
        ],
        [
            3.9, 0.2, 4.5, 1.5, 1.4, 0.1, 4.3, 0.3, 3.2, 3.4, 0.4, 3.8, 4.8,
            3.5, 5.6, 1.9, 2.8, 4.1, 1.0, 5.0
        ],
        [
            5.9, 1.8, 3.5, 0.3, 1.7, 2.5, 3.0, 3.3, 2.1, 4.3, 6.0, 0.1, 0.5,
            0.9, 4.0, 5.4, 1.1, 2.7, 5.3, 3.4
        ],
    ],
                          device='cuda')
    ref_new_tokens = torch.tensor(
        [[
            19,
            4,
            8,
            7,
            0,
            17,
            16,
            9,
            12,
            10,  # top-10
            17,
            1,
            8,
            5,
            3,
            10,
            12,  # top-7
            5,
            4,
            17,  # top-3
            12,
            10,  # top-2
            3,  # top-1
            4,  # top-1
            10,  # top-1
            11,  # top-1
            14,  # top-1
            10,  # top-1
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]],
        device='cpu')  # shape: [max_batch_size, max_total_draft_tokens + 1]
    num_new_draft_tokens = 28
    run_test(max_batch_size, draft_layer_id, max_total_draft_tokens,
             max_draft_len, eagle_choices, logits, ref_new_tokens,
             num_new_draft_tokens)


if __name__ == "__main__":
    unittest.main()
